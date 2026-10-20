from __future__ import annotations

import json
from pathlib import Path

from research_agent.config import Settings
from research_agent.executors import ProductionToolExecutor, RecordedBenchmarkToolExecutor
from research_agent.schemas import BenchmarkItem, ToolResult


def _settings(root: Path) -> Settings:
    return Settings(
        project_root=root,
        openai_api_key=None,
        tavily_api_key=None,
        openai_model="unused",
        judge_model="unused",
        max_steps=8,
        python_timeout_s=2,
        web_timeout_s=2,
        chroma_path=root / "chroma",
        run_dir=root / "runs",
        input_cost_per_million=0,
        output_cost_per_million=0,
        tavily_cost_per_call=0,
        allowed_hosts=("localhost",),
        api_docs_enabled=False,
    )


def _benchmark_item() -> BenchmarkItem:
    return BenchmarkItem(
        id="case-001",
        question="Find the author, then identify the country's capital.",
        known_correct_answer="Ottawa",
        category="multi_step",
        answerable=True,
    )


def test_production_executor_identifies_live_tavily_backend(tmp_path: Path) -> None:
    executor = ProductionToolExecutor(_settings(tmp_path))

    assert executor.retrieval_mode == "live_web"
    assert executor.web_search_provider == "tavily"


def _write_corpus(path: Path) -> None:
    payload = {
        "version": 1,
        "cases": [
            {
                "id": "case-001",
                "actions": [
                    {
                        "tool": "web_search",
                        "input": {"query": "author of Pride and Prejudice"},
                        "result": {
                            "source_id": "recorded-id-must-not-be-reused",
                            "success": True,
                            "raw_content": {
                                "results": [
                                    {
                                        "url": "https://recorded.test/austen",
                                        "title": "Pride and Prejudice reference",
                                        "snippet": "Jane Austen wrote Pride and Prejudice.",
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "tool": "web_search",
                        "input": {"query": "capital of Canada"},
                        "result": {
                            "source_id": "another-recorded-id",
                            "success": True,
                            "raw_content": {
                                "results": [
                                    {
                                        "url": "https://recorded.test/canada",
                                        "title": "Canada reference",
                                        "snippet": "Ottawa is the capital of Canada.",
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "tool": "run_python",
                        "input": {"code": "print('fixture output')"},
                        "result": {
                            "source_id": "ignored-python-fixture",
                            "success": True,
                            "raw_content": {"stdout": "fixture output\n"},
                        },
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _executor(tmp_path: Path) -> RecordedBenchmarkToolExecutor:
    corpus_path = tmp_path / "recorded-corpus.json"
    _write_corpus(corpus_path)
    return RecordedBenchmarkToolExecutor(
        _settings(tmp_path),
        corpus_path=corpus_path,
        benchmark=[_benchmark_item()],
    )


def test_recorded_web_matches_paraphrases_and_issues_fresh_source_ids(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    question = _benchmark_item().question

    first = executor.execute(
        "web_search",
        {"query": "Who penned the novel Pride & Prejudice?"},
        question=question,
    )
    repeated = executor.execute(
        "web_search",
        {"query": "Who penned the novel Pride & Prejudice?"},
        question=question,
    )

    assert first.success is True
    assert first.source_id.startswith("src_recorded_")
    assert repeated.source_id.startswith("src_recorded_")
    assert repeated.source_id != first.source_id
    assert first.source_id != "recorded-id-must-not-be-reused"
    assert first.content["retrieval_mode"] == "recorded_web_corpus"
    assert first.content["recorded_query"] == "author of Pride and Prejudice"
    assert first.content["query"] == "Who penned the novel Pride & Prejudice?"
    assert first.content["results"][0]["snippet"] == "Jane Austen wrote Pride and Prejudice."


def test_recorded_web_selects_the_semantically_relevant_entry(tmp_path: Path) -> None:
    executor = _executor(tmp_path)

    result = executor.execute(
        "web_search",
        {"query": "What city is Canada's capital?"},
        question=_benchmark_item().question,
    )

    assert result.success is True
    assert result.content["recorded_query"] == "capital of Canada"
    assert result.content["results"][0]["snippet"] == "Ottawa is the capital of Canada."


def test_recorded_web_fails_closed_for_unrelated_or_unscoped_queries(tmp_path: Path) -> None:
    executor = _executor(tmp_path)

    unrelated = executor.execute(
        "web_search",
        {"query": "tomorrow weather forecast"},
        question=_benchmark_item().question,
    )
    unscoped = executor.execute("web_search", {"query": "capital of Canada"})

    assert unrelated.success is False
    assert "no sufficiently similar" in (unrelated.error or "")
    assert unrelated.content["retrieval_mode"] == "recorded_web_corpus"
    assert unscoped.success is False
    assert "no web entries" in (unscoped.error or "")
    assert unrelated.source_id != unscoped.source_id


def test_recorded_executor_runs_real_python_instead_of_fixture_output(tmp_path: Path) -> None:
    executor = _executor(tmp_path)

    result = executor.execute(
        "run_python",
        {"code": "print(6 * 7)"},
        question=_benchmark_item().question,
    )

    assert result.success is True
    assert result.content["stdout"] == "42\n"
    assert "fixture output" not in result.content["stdout"]
    assert result.source_id != "ignored-python-fixture"


def test_recorded_executor_delegates_search_docs_to_real_tool_layer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _executor(tmp_path)
    seen: dict = {}

    def fake_search_docs(*, query: str, top_k: int, persist_directory: Path) -> ToolResult:
        seen.update(query=query, top_k=top_k, persist_directory=persist_directory)
        return ToolResult(
            source_id="fresh-doc-source",
            tool_name="search_docs",
            success=True,
            content={"chunks": [{"text": "record from the real retrieval seam"}]},
        )

    monkeypatch.setattr("research_agent.tools.search_docs", fake_search_docs)

    result = executor.execute(
        "search_docs",
        {"query": "local evidence", "top_k": 3},
        question=_benchmark_item().question,
    )

    assert result.success is True
    assert result.source_id == "fresh-doc-source"
    assert seen == {
        "query": "local evidence",
        "top_k": 3,
        "persist_directory": tmp_path / "chroma",
    }
