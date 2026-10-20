"""Tool implementations used by the research-agent loop.

Every public operation is deliberately fail-closed: it returns a ``ToolResult``
for expected configuration, dependency, network, database, and subprocess
errors instead of allowing those errors to escape into the agent loop.

``run_python`` is intended for short, generated calculations.  It validates a
restricted Python AST, then combines an isolated interpreter, a temporary
working directory, a minimal environment, an in-child resource bootstrap,
wall-clock limits, capped output files, and (when available on macOS) a
Seatbelt profile that denies network access, user-data reads, writes outside
the temporary directory, and child-process creation.  This is useful defence
in depth, but it is not a substitute for a separately isolated host or
container when executing code from untrusted users.
"""

from __future__ import annotations

import ast
import base64
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import Any, BinaryIO

import psutil

from research_agent.schemas import ToolResult

DEFAULT_CHROMA_COLLECTION = "documents"
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
RESOURCE_POLL_INTERVAL_S = 0.001


def _source_id() -> str:
    """Return a fresh citation identifier for every invocation, including failures."""

    return str(uuid.uuid4())


def _finish(
    *,
    source_id: str,
    tool_name: str,
    started_at: float,
    success: bool,
    content: dict[str, Any],
    error: str | None = None,
) -> ToolResult:
    return ToolResult(
        source_id=source_id,
        tool_name=tool_name,
        success=success,
        content=content,
        error=error,
        duration_ms=round((time.monotonic() - started_at) * 1_000, 3),
    )


def _error_text(prefix: str, exc: BaseException, secret: str | None = None) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    if secret:
        detail = detail.replace(secret, "[REDACTED]")
    return f"{prefix}: {detail}"


def web_search(
    query: str,
    max_results: int = 5,
    api_key: str | None = None,
    timeout_s: float = 15.0,
) -> ToolResult:
    """Search Tavily and return normalized URL/title/snippet records.

    The Tavily SDK is imported inside the function so installations that only
    use the other tools still import successfully.
    """

    source_id = _source_id()
    started_at = time.monotonic()
    empty_content: dict[str, Any] = {"query": query, "results": []}

    if not isinstance(query, str) or not query.strip():
        return _finish(
            source_id=source_id,
            tool_name="web_search",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="web_search requires a non-empty query",
        )
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        return _finish(
            source_id=source_id,
            tool_name="web_search",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="max_results must be a positive integer",
        )
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
        return _finish(
            source_id=source_id,
            tool_name="web_search",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="timeout_s must be positive",
        )

    resolved_key = api_key or os.environ.get("TAVILY_API_KEY")
    if not resolved_key:
        return _finish(
            source_id=source_id,
            tool_name="web_search",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="TAVILY_API_KEY is not configured",
        )

    try:
        tavily_module = import_module("tavily")
        client_class = tavily_module.TavilyClient
    except (ImportError, AttributeError) as exc:
        return _finish(
            source_id=source_id,
            tool_name="web_search",
            started_at=started_at,
            success=False,
            content=empty_content,
            error=_error_text("Tavily SDK is unavailable", exc),
        )

    try:
        client = client_class(api_key=resolved_key)
        response = client.search(
            query=query.strip(),
            max_results=max_results,
            search_depth="basic",
            timeout=float(timeout_s),
        )
        raw_results = response.get("results", []) if isinstance(response, Mapping) else response
        if raw_results is None:
            raw_results = []
        if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
            raise TypeError("Tavily returned a non-list results payload")

        results: list[dict[str, str]] = []
        for item in raw_results[:max_results]:
            if not isinstance(item, Mapping):
                continue
            results.append(
                {
                    "url": str(item.get("url") or ""),
                    "title": str(item.get("title") or ""),
                    "snippet": str(item.get("content") or item.get("snippet") or ""),
                }
            )

        return _finish(
            source_id=source_id,
            tool_name="web_search",
            started_at=started_at,
            success=True,
            content={"query": query.strip(), "results": results},
        )
    except Exception as exc:
        return _finish(
            source_id=source_id,
            tool_name="web_search",
            started_at=started_at,
            success=False,
            content=empty_content,
            error=_error_text("Tavily search failed", exc, resolved_key),
        )


_ALLOWED_PYTHON_IMPORTS = frozenset({"decimal", "fractions", "math", "statistics"})
_BLOCKED_PYTHON_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "classmethod",
        "compile",
        "copyright",
        "credits",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "importlib",
        "license",
        "locals",
        "object",
        "open",
        "os",
        "pathlib",
        "property",
        "quit",
        "resource",
        "setattr",
        "socket",
        "staticmethod",
        "subprocess",
        "super",
        "sys",
        "type",
        "vars",
    }
)
_BLOCKED_PYTHON_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.FunctionDef,
    ast.Global,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
)
_BLOCKED_PYTHON_ATTRIBUTES = frozenset(
    {
        "attrgetter",
        "builtins",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "format",
        "format_map",
        "gi_code",
        "gi_frame",
        "methodcaller",
        "modules",
        "open",
        "os",
        "pathlib",
        "resource",
        "socket",
        "subprocess",
        "sys",
        "tb_frame",
        "tb_next",
    }
)


def _validate_python_code(code: str) -> str | None:
    """Restrict execution to calculation-oriented Python before spawning it."""

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"invalid syntax on line {exc.lineno or 1}"
    except (MemoryError, RecursionError, ValueError) as exc:
        return f"parser rejected code safely: {type(exc).__name__}"

    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_PYTHON_NODES):
            return f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.Name) and (
            node.id in _BLOCKED_PYTHON_NAMES
            or (
                node.id.startswith("_") and node.id != "_"
            )
        ):
            return f"name {node.id!r} is not allowed"
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _BLOCKED_PYTHON_ATTRIBUTES
        ):
            return f"attribute {node.attr!r} is not allowed"
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            return "dunder names in string literals are not allowed"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in _ALLOWED_PYTHON_IMPORTS:
                    return f"import {alias.name!r} is not allowed"
                if alias.asname and (
                    alias.asname.startswith("_")
                    or alias.asname in (_BLOCKED_PYTHON_NAMES | _BLOCKED_PYTHON_ATTRIBUTES)
                ):
                    return f"import alias {alias.asname!r} is not allowed"
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in _ALLOWED_PYTHON_IMPORTS:
                return f"import from {node.module!r} is not allowed"
            if any(alias.name == "*" for alias in node.names):
                return "wildcard imports are not allowed"
            for alias in node.names:
                if alias.name.startswith("_") or alias.name in (
                    _BLOCKED_PYTHON_NAMES | _BLOCKED_PYTHON_ATTRIBUTES
                ):
                    return f"imported name {alias.name!r} is not allowed"
                if alias.asname and (
                    alias.asname.startswith("_")
                    or alias.asname in (_BLOCKED_PYTHON_NAMES | _BLOCKED_PYTHON_ATTRIBUTES)
                ):
                    return f"import alias {alias.asname!r} is not allowed"
    return None


_RESOURCE_BOOTSTRAP = """
import base64
import builtins
import decimal
import fractions
import math
import os
import statistics
import sys
import types

try:
    import resource
except ImportError:
    resource = None

def set_limit(name, value):
    if resource is None:
        return
    limit = getattr(resource, name, None)
    if limit is None:
        return
    try:
        resource.setrlimit(limit, (value, value))
    except (OSError, ValueError):
        pass

timeout_s = float(sys.argv[1])
memory_limit_mb = int(sys.argv[2])
max_output_bytes = int(sys.argv[3])
os.umask(0o077)
set_limit("RLIMIT_CORE", 0)
set_limit("RLIMIT_CPU", max(1, int(timeout_s) + 1))
set_limit("RLIMIT_AS", memory_limit_mb * 1024 * 1024)
set_limit("RLIMIT_FSIZE", max_output_bytes + 1)
set_limit("RLIMIT_NOFILE", 32)
set_limit("RLIMIT_NPROC", 1)

math_names = (
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil",
    "comb", "copysign", "cos", "cosh", "degrees", "dist", "e", "erf", "erfc",
    "exp", "exp2", "expm1", "fabs", "factorial", "floor", "fmod", "frexp", "fsum",
    "gamma", "gcd", "hypot", "inf", "isclose", "isfinite", "isinf", "isnan",
    "isqrt", "lcm", "ldexp", "lgamma", "log", "log10", "log1p", "log2", "modf",
    "nan", "nextafter", "perm", "pi", "pow", "prod", "radians", "remainder", "sin",
    "sinh", "sqrt", "tan", "tanh", "tau", "trunc", "ulp",
)
statistics_names = (
    "NormalDist", "correlation", "covariance", "fmean", "geometric_mean",
    "harmonic_mean", "linear_regression", "mean", "median", "median_grouped",
    "median_high", "median_low", "mode", "multimode", "pstdev", "pvariance",
    "quantiles", "stdev", "variance",
)
decimal_names = (
    "BasicContext", "Context", "Decimal", "DefaultContext", "ExtendedContext",
    "MAX_EMAX", "MIN_EMIN", "ROUND_05UP", "ROUND_CEILING", "ROUND_DOWN",
    "ROUND_FLOOR", "ROUND_HALF_DOWN", "ROUND_HALF_EVEN", "ROUND_HALF_UP", "ROUND_UP",
    "getcontext", "localcontext", "setcontext",
)

def proxy(module, names):
    return types.SimpleNamespace(
        **{name: getattr(module, name) for name in names if hasattr(module, name)}
    )

safe_modules = {
    "math": proxy(math, math_names),
    "statistics": proxy(statistics, statistics_names),
    "decimal": proxy(decimal, decimal_names),
    "fractions": types.SimpleNamespace(Fraction=fractions.Fraction),
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    del globals, locals, fromlist
    if level or name not in safe_modules:
        raise ImportError("module is not available in the calculation sandbox")
    return safe_modules[name]

safe_builtin_names = (
    "ArithmeticError", "Exception", "OverflowError", "TypeError", "ValueError",
    "ZeroDivisionError", "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "hex", "int", "isinstance", "issubclass", "len",
    "list", "map", "max", "min", "next", "oct", "ord", "pow", "print", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
)
safe_builtins = {name: getattr(builtins, name) for name in safe_builtin_names}
safe_builtins["__import__"] = safe_import
source = base64.b64decode(sys.argv[4]).decode("utf-8")
exec(
    compile(source, "<agent-code>", "exec"),
    {"__name__": "__main__", "__builtins__": safe_builtins},
)
""".strip()


def _seatbelt_command(command: list[str], temporary_directory: str) -> list[str]:
    """Wrap a command in the macOS sandbox when the system utility exists."""

    sandbox_exec = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if not sandbox_exec:
        return command

    def escaped(path: str | os.PathLike[str]) -> str:
        return str(Path(path).resolve()).replace("\\", "\\\\").replace('"', '\\"')

    escaped_directory = escaped(temporary_directory)
    readable_runtime_paths = {
        escaped(sys.prefix),
        escaped(sys.base_prefix),
        escaped(Path(sys.executable).resolve().parent),
        escaped_directory,
    }
    protected_paths = {
        escaped(Path.home()),
        "/Network",
        "/Volumes",
    }
    profile = "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny process-fork)",
            "(deny signal)",
            "(deny sysctl-read)",
            "(deny process-info*)",
            "(allow process-info* (target self))",
            "(deny process-exec)",
            "(allow process-exec "
            f'(literal "{escaped(sys.executable)}") '
            f'(literal "{escaped(Path(sys.executable).resolve())}"))',
            "(deny file-write*)",
            f'(allow file-write* (subpath "{escaped_directory}"))',
            "(deny file-read-data "
            + " ".join(f'(subpath "{path}")' for path in sorted(protected_paths))
            + ")",
            "(allow file-read-data "
            + " ".join(f'(subpath "{path}")' for path in sorted(readable_runtime_paths))
            + ")",
        ]
    )
    return [sandbox_exec, "-p", profile, *command]


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        with suppress(OSError, ProcessLookupError):
            process.kill()


def _read_capped(stream: BinaryIO, max_output_bytes: int) -> tuple[str, bool]:
    stream.seek(0)
    raw = stream.read(max_output_bytes + 1)
    truncated = len(raw) > max_output_bytes
    visible = raw[:max_output_bytes]
    text = visible.decode("utf-8", errors="replace")
    return text, truncated


def _process_tree_rss_bytes(process: subprocess.Popen[bytes]) -> int | None:
    """Return resident memory for the sandbox process tree when it still exists."""

    try:
        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running())
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except psutil.AccessDenied as exc:
        raise RuntimeError("sandbox memory monitoring is unavailable") from exc


def _wait_for_sandbox(
    process: subprocess.Popen[bytes],
    *,
    timeout_s: float,
    memory_limit_bytes: int,
) -> tuple[bool, bool, int]:
    """Wait while enforcing wall time and RSS from the parent process.

    ``RLIMIT_AS`` remains the fast in-child guard on platforms that enforce it.
    Darwin does not reliably honor that limit for Python, so this independent
    watchdog also makes the configured bound effective there.
    """

    deadline = time.monotonic() + timeout_s
    peak_rss_bytes = 0
    while process.poll() is None:
        rss_bytes = _process_tree_rss_bytes(process)
        if rss_bytes is not None:
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            if rss_bytes > memory_limit_bytes:
                _kill_process_group(process)
                process.wait(timeout=2)
                return False, True, peak_rss_bytes

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process_group(process)
            process.wait(timeout=2)
            return True, False, peak_rss_bytes
        time.sleep(min(RESOURCE_POLL_INTERVAL_S, remaining))

    return False, False, peak_rss_bytes


def run_python(
    code: str,
    timeout_s: float = 5.0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    memory_limit_mb: int = 256,
) -> ToolResult:
    """Execute Python with bounded time, resources, environment, and output."""

    source_id = _source_id()
    started_at = time.monotonic()
    empty_content: dict[str, Any] = {
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "timed_out": False,
        "memory_limit_exceeded": False,
        "peak_rss_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }

    if not isinstance(code, str) or not code.strip():
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="run_python requires non-empty code",
        )
    if len(code) > 100_000:
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="Python code exceeds the 100000 character limit",
        )
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="timeout_s must be positive",
        )
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes < 1
    ):
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="max_output_bytes must be a positive integer",
        )
    if (
        isinstance(memory_limit_mb, bool)
        or not isinstance(memory_limit_mb, int)
        or memory_limit_mb < 32
    ):
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="memory_limit_mb must be an integer of at least 32",
        )

    try:
        validation_error = _validate_python_code(code)
    except Exception as exc:  # parser/validator failures must stay inside the tool contract
        validation_error = f"validator failed safely: {type(exc).__name__}"
    if validation_error:
        encoded_error = f"Python code rejected by sandbox: {validation_error}".encode()
        bounded_error = encoded_error[:max_output_bytes].decode("utf-8", errors="replace")
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=False,
            content=empty_content,
            error=bounded_error,
        )

    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="research-agent-python-") as temporary_directory:
            python_command = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-u",
                "-X",
                "utf8",
                "-c",
                _RESOURCE_BOOTSTRAP,
                str(float(timeout_s)),
                str(memory_limit_mb),
                str(max_output_bytes),
                base64.b64encode(code.encode("utf-8")).decode("ascii"),
            ]
            command = _seatbelt_command(python_command, temporary_directory)
            child_environment = {
                "PATH": "",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "TMPDIR": temporary_directory,
            }

            with tempfile.TemporaryFile(
                mode="w+b", dir=temporary_directory
            ) as stdout_file, tempfile.TemporaryFile(
                mode="w+b", dir=temporary_directory
            ) as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=temporary_directory,
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    close_fds=True,
                    start_new_session=True,
                )
                timed_out, memory_limit_exceeded, peak_rss_bytes = _wait_for_sandbox(
                    process,
                    timeout_s=float(timeout_s),
                    memory_limit_bytes=memory_limit_mb * 1024 * 1024,
                )

                stdout, stdout_truncated = _read_capped(stdout_file, max_output_bytes)
                stderr, stderr_truncated = _read_capped(stderr_file, max_output_bytes)
                exit_code = process.returncode

        content = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "memory_limit_exceeded": memory_limit_exceeded,
            "peak_rss_bytes": peak_rss_bytes,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        if timed_out:
            return _finish(
                source_id=source_id,
                tool_name="run_python",
                started_at=started_at,
                success=False,
                content=content,
                error=f"Python execution timed out after {float(timeout_s):g} seconds",
            )
        if memory_limit_exceeded:
            return _finish(
                source_id=source_id,
                tool_name="run_python",
                started_at=started_at,
                success=False,
                content=content,
                error=f"Python exceeded the {memory_limit_mb} MiB memory limit",
            )
        if exit_code != 0:
            return _finish(
                source_id=source_id,
                tool_name="run_python",
                started_at=started_at,
                success=False,
                content=content,
                error=f"Python exited with code {exit_code}",
            )
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=True,
            content=content,
        )
    except Exception as exc:
        if process is not None and process.poll() is None:
            _kill_process_group(process)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
        return _finish(
            source_id=source_id,
            tool_name="run_python",
            started_at=started_at,
            success=False,
            content=empty_content,
            error=_error_text("Python execution failed", exc),
        )


def _resolve_chroma_directory(persist_directory: str | os.PathLike[str] | None) -> str:
    configured = persist_directory or os.environ.get("CHROMA_DB_PATH")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str((Path.cwd() / ".research_agent_chroma").resolve())


def _get_chroma_collection(persist_directory: str, collection_name: str) -> Any:
    chromadb_module = import_module("chromadb")
    client = chromadb_module.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(name=collection_name)


def _first_query_row(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    if value and isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes)):
        return list(value[0])
    return list(value)


def search_docs(
    query: str,
    top_k: int = 5,
    persist_directory: str | os.PathLike[str] | None = None,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
) -> ToolResult:
    """Return the top matching local Chroma chunks with source metadata."""

    source_id = _source_id()
    started_at = time.monotonic()
    empty_content: dict[str, Any] = {"query": query, "chunks": []}

    if not isinstance(query, str) or not query.strip():
        return _finish(
            source_id=source_id,
            tool_name="search_docs",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="search_docs requires a non-empty query",
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        return _finish(
            source_id=source_id,
            tool_name="search_docs",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="top_k must be a positive integer",
        )
    if not isinstance(collection_name, str) or not collection_name.strip():
        return _finish(
            source_id=source_id,
            tool_name="search_docs",
            started_at=started_at,
            success=False,
            content=empty_content,
            error="collection_name must be non-empty",
        )

    try:
        resolved_directory = _resolve_chroma_directory(persist_directory)
        collection = _get_chroma_collection(resolved_directory, collection_name.strip())
        raw_result = collection.query(
            query_texts=[query.strip()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not isinstance(raw_result, Mapping):
            raise TypeError("Chroma returned a non-mapping query payload")

        documents = _first_query_row(raw_result.get("documents"))
        metadatas = _first_query_row(raw_result.get("metadatas"))
        distances = _first_query_row(raw_result.get("distances"))
        document_ids = _first_query_row(raw_result.get("ids"))
        chunks: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata_value = metadatas[index] if index < len(metadatas) else {}
            metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
            document_source = str(metadata.get("source") or "unknown")
            chunks.append(
                {
                    "text": str(document or ""),
                    "document_source": document_source,
                    "metadata": metadata,
                    "distance": distances[index] if index < len(distances) else None,
                    "document_id": document_ids[index] if index < len(document_ids) else None,
                }
            )

        return _finish(
            source_id=source_id,
            tool_name="search_docs",
            started_at=started_at,
            success=True,
            content={
                "query": query.strip(),
                "chunks": chunks,
                "collection": collection_name.strip(),
                "persist_directory": resolved_directory,
            },
        )
    except Exception as exc:
        return _finish(
            source_id=source_id,
            tool_name="search_docs",
            started_at=started_at,
            success=False,
            content=empty_content,
            error=_error_text("Document search failed", exc),
        )


def add_documents(
    documents: Sequence[str],
    sources: Sequence[str],
    ids: Sequence[str] | None = None,
    metadatas: Sequence[Mapping[str, Any]] | None = None,
    persist_directory: str | os.PathLike[str] | None = None,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
) -> ToolResult:
    """Idempotently add or update local chunks, always recording their source."""

    source_id = _source_id()
    started_at = time.monotonic()
    empty_content: dict[str, Any] = {"document_ids": [], "count": 0}

    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        error = "documents must be a sequence of strings"
    elif isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        error = "sources must be a sequence of strings"
    elif not documents:
        error = "documents must not be empty"
    elif len(documents) != len(sources):
        error = "documents and sources must have the same length"
    elif any(not isinstance(document, str) or not document.strip() for document in documents):
        error = "every document must be a non-empty string"
    elif any(not isinstance(source, str) or not source.strip() for source in sources):
        error = "every source must be a non-empty string"
    elif ids is not None and (
        isinstance(ids, (str, bytes))
        or not isinstance(ids, Sequence)
        or len(ids) != len(documents)
    ):
        error = "ids must have the same length as documents"
    elif metadatas is not None and (
        isinstance(metadatas, (str, bytes))
        or not isinstance(metadatas, Sequence)
        or len(metadatas) != len(documents)
    ):
        error = "metadatas must have the same length as documents"
    elif not isinstance(collection_name, str) or not collection_name.strip():
        error = "collection_name must be non-empty"
    else:
        error = None

    if error:
        return _finish(
            source_id=source_id,
            tool_name="add_documents",
            started_at=started_at,
            success=False,
            content=empty_content,
            error=error,
        )

    normalized_documents = [document.strip() for document in documents]
    normalized_sources = [source.strip() for source in sources]
    if ids is None:
        normalized_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}\0{document}"))
            for document, source in zip(normalized_documents, normalized_sources, strict=True)
        ]
    else:
        normalized_ids = [str(document_id) for document_id in ids]

    normalized_metadatas: list[dict[str, Any]] = []
    for index, source in enumerate(normalized_sources):
        supplied_metadata = metadatas[index] if metadatas is not None else {}
        if not isinstance(supplied_metadata, Mapping):
            return _finish(
                source_id=source_id,
                tool_name="add_documents",
                started_at=started_at,
                success=False,
                content=empty_content,
                error="every metadata item must be a mapping",
            )
        metadata = dict(supplied_metadata)
        metadata["source"] = source
        normalized_metadatas.append(metadata)

    try:
        resolved_directory = _resolve_chroma_directory(persist_directory)
        collection = _get_chroma_collection(resolved_directory, collection_name.strip())
        collection.upsert(
            ids=normalized_ids,
            documents=normalized_documents,
            metadatas=normalized_metadatas,
        )
        return _finish(
            source_id=source_id,
            tool_name="add_documents",
            started_at=started_at,
            success=True,
            content={
                "document_ids": normalized_ids,
                "count": len(normalized_documents),
                "collection": collection_name.strip(),
                "persist_directory": resolved_directory,
            },
        )
    except Exception as exc:
        return _finish(
            source_id=source_id,
            tool_name="add_documents",
            started_at=started_at,
            success=False,
            content=empty_content,
            error=_error_text("Adding documents failed", exc),
        )


__all__ = ["add_documents", "run_python", "search_docs", "web_search"]
