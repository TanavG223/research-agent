from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from research_agent import tools


@pytest.fixture(autouse=True)
def _clear_tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


def test_web_search_returns_fresh_source_ids_when_key_is_missing() -> None:
    first = tools.web_search("first question")
    second = tools.web_search("second question")

    assert not first.success
    assert first.error == "TAVILY_API_KEY is not configured"
    assert first.content == {"query": "first question", "results": []}
    assert first.source_id != second.source_id
    uuid.UUID(first.source_id)
    uuid.UUID(second.source_id)


def test_web_search_normalizes_tavily_results(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeTavilyClient:
        def __init__(self, *, api_key: str) -> None:
            seen["api_key"] = api_key

        def search(self, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {
                "results": [
                    {
                        "url": "https://example.test/a",
                        "title": "Example A",
                        "content": "A raw Tavily snippet.",
                        "score": 0.99,
                    },
                    {
                        "url": "https://example.test/b",
                        "title": "Example B",
                        "snippet": "Fallback snippet field.",
                    },
                ]
            }

    real_import = tools.import_module

    def fake_import(name: str) -> Any:
        if name == "tavily":
            return SimpleNamespace(TavilyClient=FakeTavilyClient)
        return real_import(name)

    monkeypatch.setattr(tools, "import_module", fake_import)
    result = tools.web_search("  grounded agents  ", max_results=2, api_key="test-key", timeout_s=3)

    assert result.success
    assert result.error is None
    assert result.content == {
        "query": "grounded agents",
        "results": [
            {
                "url": "https://example.test/a",
                "title": "Example A",
                "snippet": "A raw Tavily snippet.",
            },
            {
                "url": "https://example.test/b",
                "title": "Example B",
                "snippet": "Fallback snippet field.",
            },
        ],
    }
    assert seen == {
        "api_key": "test-key",
        "query": "grounded agents",
        "max_results": 2,
        "search_depth": "basic",
        "timeout": 3.0,
    }


def test_web_search_converts_network_error_and_redacts_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key

        def search(self, **_: Any) -> dict[str, Any]:
            raise ConnectionError(f"request with {self.api_key} failed")

    monkeypatch.setattr(
        tools,
        "import_module",
        lambda _: SimpleNamespace(TavilyClient=FailingClient),
    )
    result = tools.web_search("question", api_key="secret-value")

    assert not result.success
    assert result.content["results"] == []
    assert result.error is not None
    assert "Tavily search failed" in result.error
    assert "secret-value" not in result.error
    assert "[REDACTED]" in result.error


def test_web_search_returns_structured_failure_when_tavily_sdk_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = tools.import_module

    def missing_tavily(name: str) -> Any:
        if name == "tavily":
            raise ModuleNotFoundError("No module named 'tavily'")
        return real_import(name)

    monkeypatch.setattr(tools, "import_module", missing_tavily)
    result = tools.web_search("question", api_key="configured")

    assert not result.success
    assert result.content == {"query": "question", "results": []}
    assert "Tavily SDK is unavailable" in (result.error or "")


def test_run_python_captures_successful_output() -> None:
    result = tools.run_python("print(sum(range(6)))", timeout_s=2)

    assert result.success, result.error
    assert result.content["stdout"] == "15\n"
    assert result.content["stderr"] == ""
    assert result.content["exit_code"] == 0
    assert result.content["timed_out"] is False
    uuid.UUID(result.source_id)


def test_run_python_returns_nonzero_exit_as_structured_failure() -> None:
    result = tools.run_python("raise ValueError('bad calculation')", timeout_s=2)

    assert not result.success
    assert result.content["exit_code"] != 0
    assert "ValueError: bad calculation" in result.content["stderr"]
    assert result.error is not None
    assert result.error.startswith("Python exited with code")


def test_run_python_enforces_wall_timeout_without_raising() -> None:
    started_at = time.monotonic()
    result = tools.run_python("while True:\n    pass", timeout_s=0.15)
    elapsed = time.monotonic() - started_at

    assert not result.success
    assert result.content["timed_out"] is True
    assert "timed out" in (result.error or "")
    assert elapsed < 2.0


def test_run_python_enforces_memory_limit_from_parent_watchdog() -> None:
    result = tools.run_python(
        "chunks = []\nwhile True:\n    chunks.append([0] * 500000)",
        timeout_s=3,
        memory_limit_mb=64,
    )

    assert not result.success
    assert result.content["timed_out"] is False
    if result.content["memory_limit_exceeded"]:
        assert result.content["peak_rss_bytes"] > 64 * 1024 * 1024
        assert result.error == "Python exceeded the 64 MiB memory limit"
    else:
        # Platforms with a working RLIMIT_AS can terminate allocation before
        # the independent parent watchdog observes an over-limit RSS sample.
        assert "MemoryError" in result.content["stderr"]


def test_run_python_caps_stdout() -> None:
    result = tools.run_python("print('x' * 10000)", timeout_s=2, max_output_bytes=512)

    assert len(result.content["stdout"].encode()) <= 512
    assert result.content["stdout_truncated"] is True
    # Exceeding the OS file-size limit may terminate Python; either way the
    # tool returns a bounded structured result rather than raising.
    assert isinstance(result.success, bool)


@pytest.mark.parametrize(
    ("code", "expected_error"),
    [
        ("print(open('/etc/passwd').read())", "name 'open' is not allowed"),
        ("import os\nprint(os.listdir('/'))", "import 'os' is not allowed"),
        ("print((1).__class__)", "attribute '__class__' is not allowed"),
        (
            "import statistics\nprint(statistics.sys.modules['os'])",
            "attribute 'modules' is not allowed",
        ),
        (
            "g = (x for x in ())\nprint(g.gi_frame.f_builtins['open']('/etc/passwd').read())",
            "attribute 'f_builtins' is not allowed",
        ),
        (
            "from statistics import __builtins__ as b\nprint(b['open']('/etc/passwd').read())",
            "imported name '__builtins__' is not allowed",
        ),
        (
            "match 1:\n    case int(__class__=klass):\n        print(klass)",
            "Match is not allowed",
        ),
        (
            'print("{0.__globals__[sys].modules[os]}".format(sum))',
            "attribute 'format' is not allowed",
        ),
    ],
)
def test_run_python_rejects_filesystem_and_introspection_escape_paths(
    code: str,
    expected_error: str,
) -> None:
    result = tools.run_python(code, timeout_s=2)

    assert not result.success
    assert result.content["stdout"] == ""
    assert result.error == f"Python code rejected by sandbox: {expected_error}"


def test_run_python_safe_module_proxy_blocks_transitive_builtins_escape() -> None:
    result = tools.run_python(
        "import fractions\n"
        "fractions.re.enum.bltns.exec(\"print(open('/etc/hosts').read())\")",
        timeout_s=2,
    )

    assert not result.success
    assert "localhost" not in result.content["stdout"]
    assert result.error == "Python exited with code 1"


def test_run_python_parser_recursion_is_a_bounded_structured_failure() -> None:
    result = tools.run_python("+".join(["1"] * 3000), timeout_s=2)

    assert not result.success
    assert result.error == (
        "Python code rejected by sandbox: parser rejected code safely: RecursionError"
    )


def test_run_python_caps_validation_error_text() -> None:
    result = tools.run_python(
        "import " + "a" * 90_000,
        timeout_s=2,
        max_output_bytes=512,
    )

    assert not result.success
    assert result.error is not None
    assert len(result.error.encode("utf-8")) <= 512


def test_run_python_does_not_use_preexec_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = tools.subprocess.Popen
    seen_kwargs: dict[str, Any] = {}

    def tracking_popen(*args: Any, **kwargs: Any):
        seen_kwargs.update(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(tools.subprocess, "Popen", tracking_popen)
    result = tools.run_python("print(42)", timeout_s=2)

    assert result.success, result.error
    assert "preexec_fn" not in seen_kwargs


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt-specific regression test")
def test_macos_seatbelt_denies_reads_outside_the_scratch_directory(tmp_path: Any) -> None:
    protected_file = Path(__file__).resolve().parents[1] / "README.md"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    command = tools._seatbelt_command(
        [sys.executable, "-I", "-S", "-c", f"print(open({str(protected_file)!r}).read())"],
        str(scratch),
    )

    completed = subprocess.run(
        command,
        cwd=scratch,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode != 0
    assert "Grounded Research Agent" not in completed.stdout
    assert "Operation not permitted" in completed.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt-specific regression test")
def test_macos_seatbelt_denies_outbound_signals_and_native_exec(tmp_path: Any) -> None:
    sleeper = subprocess.Popen(["/bin/sleep", "5"])
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    try:
        signal_command = tools._seatbelt_command(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                f"import os, signal; os.kill({sleeper.pid}, signal.SIGTERM)",
            ],
            str(scratch),
        )
        signal_result = subprocess.run(
            signal_command,
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        exec_command = tools._seatbelt_command(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                'import os; os.execve("/bin/echo", ["echo", "escaped"], {})',
            ],
            str(scratch),
        )
        exec_result = subprocess.run(
            exec_command,
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

        assert signal_result.returncode != 0
        assert "Operation not permitted" in signal_result.stderr
        assert sleeper.poll() is None
        assert exec_result.returncode != 0
        assert "Operation not permitted" in exec_result.stderr
        assert "escaped" not in exec_result.stdout
    finally:
        if sleeper.poll() is None:
            sleeper.terminate()
            sleeper.wait(timeout=2)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt-specific regression test")
def test_macos_seatbelt_denies_parent_environment_sysctl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    canary = "sandbox-parent-canary-42"
    monkeypatch.setenv("SANDBOX_PARENT_CANARY", canary)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    code = (
        "import os, types\n"
        # Prevent the broad sysctl rule from stopping ctypes during import;
        # this ensures the narrower cross-process rule is truly exercised.
        "os.uname = lambda: types.SimpleNamespace(release='25.0.0')\n"
        "import ctypes\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "mib = (ctypes.c_int * 3)(1, 49, os.getppid())\n"
        "size = ctypes.c_size_t(1024 * 1024)\n"
        "buf = ctypes.create_string_buffer(size.value)\n"
        "rc = libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0)\n"
        f"print(rc, {canary.encode()!r} in buf.raw[:size.value])\n"
    )
    command = tools._seatbelt_command(
        [sys.executable, "-I", "-S", "-c", code],
        str(scratch),
    )

    completed = subprocess.run(
        command,
        cwd=scratch,
        env={"PATH": "", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": str(scratch)},
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "-1 False"
    assert canary not in completed.stdout


class _FakeCollection:
    def __init__(self) -> None:
        self.upsert_payload: dict[str, Any] | None = None
        self.upsert_payloads: list[dict[str, Any]] = []
        self.query_payload: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_payload = kwargs
        return {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Alpha text", "Beta text"]],
            "metadatas": [[{"source": "alpha.pdf", "page": 3}, {"page": 4}]],
            "distances": [[0.1, 0.25]],
        }

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_payload = kwargs
        self.upsert_payloads.append(kwargs)


class _FakeChroma:
    def __init__(self, collection: _FakeCollection) -> None:
        self.collection = collection
        self.paths: list[str] = []
        self.collection_names: list[str] = []

    def PersistentClient(self, *, path: str) -> _FakeChroma:  # noqa: N802
        self.paths.append(path)
        return self

    def get_or_create_collection(self, *, name: str) -> _FakeCollection:
        self.collection_names.append(name)
        return self.collection


def _install_fake_chroma(
    monkeypatch: pytest.MonkeyPatch, collection: _FakeCollection
) -> _FakeChroma:
    fake_chroma = _FakeChroma(collection)
    real_import = tools.import_module

    def fake_import(name: str) -> Any:
        if name == "chromadb":
            return fake_chroma
        return real_import(name)

    monkeypatch.setattr(tools, "import_module", fake_import)
    return fake_chroma


def test_search_docs_returns_chunks_with_document_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    collection = _FakeCollection()
    fake_chroma = _install_fake_chroma(monkeypatch, collection)

    result = tools.search_docs(
        "alpha",
        top_k=2,
        persist_directory=tmp_path,
        collection_name="research",
    )

    assert result.success, result.error
    assert result.content["chunks"] == [
        {
            "text": "Alpha text",
            "document_source": "alpha.pdf",
            "metadata": {"source": "alpha.pdf", "page": 3},
            "distance": 0.1,
            "document_id": "chunk-1",
        },
        {
            "text": "Beta text",
            "document_source": "unknown",
            "metadata": {"page": 4},
            "distance": 0.25,
            "document_id": "chunk-2",
        },
    ]
    assert collection.query_payload == {
        "query_texts": ["alpha"],
        "n_results": 2,
        "include": ["documents", "metadatas", "distances"],
    }
    assert fake_chroma.collection_names == ["research"]


def test_add_documents_upserts_deterministic_ids_and_source_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    collection = _FakeCollection()
    _install_fake_chroma(monkeypatch, collection)

    first = tools.add_documents(
        [" Alpha text ", "Beta text"],
        ["alpha.pdf", "beta.html"],
        metadatas=[{"page": 3}, {"source": "wrong", "section": "intro"}],
        persist_directory=tmp_path,
    )
    second = tools.add_documents(
        [" Alpha text ", "Beta text"],
        ["alpha.pdf", "beta.html"],
        persist_directory=tmp_path,
    )

    assert first.success, first.error
    assert second.success, second.error
    assert first.source_id != second.source_id
    assert first.content["document_ids"] == second.content["document_ids"]
    assert all(uuid.UUID(document_id) for document_id in first.content["document_ids"])
    assert collection.upsert_payload is not None
    assert collection.upsert_payloads[0]["metadatas"] == [
        {"page": 3, "source": "alpha.pdf"},
        {"source": "beta.html", "section": "intro"},
    ]
    assert collection.upsert_payload["metadatas"] == [
        {"source": "alpha.pdf"},
        {"source": "beta.html"},
    ]


def test_search_docs_returns_structured_failure_when_chromadb_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    real_import = tools.import_module

    def missing_chroma(name: str) -> Any:
        if name == "chromadb":
            raise ModuleNotFoundError("No module named 'chromadb'")
        return real_import(name)

    monkeypatch.setattr(tools, "import_module", missing_chroma)
    result = tools.search_docs("query", persist_directory=tmp_path)

    assert not result.success
    assert result.content == {"query": "query", "chunks": []}
    assert "Document search failed" in (result.error or "")


def test_add_documents_rejects_length_mismatch_without_importing_chroma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools,
        "import_module",
        lambda name: pytest.fail(f"unexpected optional import: {name}"),
    )

    result = tools.add_documents(["one", "two"], ["only-one-source"])

    assert not result.success
    assert result.content == {"document_ids": [], "count": 0}
    assert result.error == "documents and sources must have the same length"
