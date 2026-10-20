"""Validated dispatch for production tools and deterministic benchmark fixtures."""

from __future__ import annotations

import copy
import json
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from research_agent.config import Settings
from research_agent.schemas import BenchmarkItem, ToolResult


class ToolExecutor(Protocol):
    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        question: str | None = None,
    ) -> ToolResult: ...


class ProductionToolExecutor:
    """Execute live Tavily search plus the production Python/docs tools."""

    retrieval_mode = "live_web"
    web_search_provider = "tavily"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        question: str | None = None,
    ) -> ToolResult:
        del question
        # Lazy import keeps data validation and offline benchmarks usable in minimal installs.
        from research_agent.tools import run_python, search_docs, web_search

        if tool_name == "web_search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return _invalid_result(tool_name, "query must be a non-empty string")
            return web_search(
                query=query,
                api_key=self.settings.tavily_api_key,
                timeout_s=self.settings.web_timeout_s,
            )
        if tool_name == "run_python":
            code = arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                return _invalid_result(tool_name, "code must be a non-empty string")
            return run_python(code=code, timeout_s=self.settings.python_timeout_s)
        if tool_name == "search_docs":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return _invalid_result(tool_name, "query must be a non-empty string")
            return search_docs(
                query=query,
                top_k=int(arguments.get("top_k", 5)),
                persist_directory=self.settings.chroma_path,
            )
        return _invalid_result(tool_name, f"unknown tool: {tool_name}")


class FixtureToolExecutor:
    """Resolve exact scripted calls to stored raw results."""

    def __init__(self, fixtures_path: Path, benchmark: list[BenchmarkItem]) -> None:
        payload = json.loads(fixtures_path.read_text(encoding="utf-8"))
        question_by_id = {item.id: item.question for item in benchmark}
        self._results: dict[tuple[str, str, str], dict[str, Any]] = {}
        for case in payload["cases"]:
            question = question_by_id[case["id"]]
            for action in case["actions"]:
                key = (question, action["tool"], _canonical(action["input"]))
                previous = self._results.get(key)
                if previous is not None and previous != action["result"]:
                    raise ValueError(f"fixture call collision with different results: {key}")
                self._results[key] = action["result"]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        question: str | None = None,
    ) -> ToolResult:
        fixture = self._results.get((question or "", tool_name, _canonical(arguments)))
        if fixture is None:
            return _invalid_result(tool_name, "no exact offline fixture for this call")
        return ToolResult(
            source_id=fixture["source_id"],
            tool_name=tool_name,
            success=bool(fixture["success"]),
            content=fixture.get("raw_content", {}),
            error=fixture.get("error"),
            duration_ms=float(fixture.get("duration_ms", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class _RecordedWebEntry:
    query: str
    success: bool
    content: dict[str, Any]
    error: str | None


_SEMANTIC_STOP_WORDS = {
    "a",
    "an",
    "and",
    "did",
    "do",
    "does",
    "find",
    "first",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "then",
    "to",
    "was",
    "what",
    "where",
    "which",
    "who",
    "with",
}
_SEMANTIC_ALIASES = {
    "authored": "author",
    "biggest": "largest",
    "nation": "country",
    "penned": "author",
    "writer": "author",
    "wrote": "author",
    "written": "author",
}
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _semantic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", text.casefold()):
        if raw in _SEMANTIC_STOP_WORDS:
            continue
        normalized = _NUMBER_WORDS.get(raw, _SEMANTIC_ALIASES.get(raw, raw))
        tokens.add(normalized)
    return tokens


def _entry_search_text(entry: _RecordedWebEntry) -> str:
    fields = [entry.query]
    results = entry.content.get("results", [])
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                fields.extend(str(result.get(key) or "") for key in ("title", "snippet"))
    return " ".join(fields)


def _semantic_match_score(query: str, entry: _RecordedWebEntry) -> tuple[int, float]:
    query_tokens = _semantic_tokens(query)
    recorded_tokens = _semantic_tokens(entry.query)
    evidence_tokens = _semantic_tokens(_entry_search_text(entry))
    overlap = query_tokens & evidence_tokens
    if not query_tokens or not recorded_tokens or not overlap:
        return 0, 0.0
    query_coverage = len(overlap) / len(query_tokens)
    recorded_overlap = query_tokens & recorded_tokens
    recorded_coverage = len(recorded_overlap) / len(recorded_tokens)
    sequence_similarity = SequenceMatcher(
        None,
        " ".join(sorted(query_tokens)),
        " ".join(sorted(recorded_tokens)),
    ).ratio()
    score = 0.65 * query_coverage + 0.25 * recorded_coverage + 0.10 * sequence_similarity
    return len(overlap), score


class RecordedBenchmarkToolExecutor:
    """Use a recorded web corpus while keeping local calculation/retrieval real.

    This is intentionally not a live-web executor. Web calls are matched only
    against the recorded entries belonging to the current benchmark question.
    Each invocation receives a new source ID, including repeated corpus hits.
    """

    retrieval_mode = "recorded_web_corpus"

    def __init__(
        self,
        settings: Settings,
        *,
        corpus_path: Path,
        benchmark: Sequence[BenchmarkItem],
        minimum_similarity: float = 0.34,
    ) -> None:
        if not 0 < minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be in (0, 1]")
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            raise ValueError("recorded corpus must contain a cases list")
        question_by_id = {item.id: item.question for item in benchmark}
        if len(question_by_id) != len(benchmark):
            raise ValueError("benchmark IDs must be unique")
        self.settings = settings
        self.corpus_path = corpus_path.resolve()
        self.minimum_similarity = minimum_similarity
        self._production = ProductionToolExecutor(settings)
        self._web_entries: dict[str, list[_RecordedWebEntry]] = {}
        for case in cases:
            if not isinstance(case, dict) or case.get("id") not in question_by_id:
                continue
            question = question_by_id[str(case["id"])]
            entries = self._web_entries.setdefault(question, [])
            actions = case.get("actions", [])
            if not isinstance(actions, list):
                raise ValueError(f"recorded corpus actions must be a list for {case['id']}")
            for action in actions:
                if not isinstance(action, dict) or action.get("tool") != "web_search":
                    continue
                arguments = action.get("input")
                result = action.get("result")
                if not isinstance(arguments, dict) or not isinstance(result, dict):
                    raise ValueError(f"invalid recorded web entry for {case['id']}")
                recorded_query = arguments.get("query")
                raw_content = result.get("raw_content", {})
                if not isinstance(recorded_query, str) or not recorded_query.strip():
                    raise ValueError(f"recorded web query must be non-empty for {case['id']}")
                if not isinstance(raw_content, dict):
                    raise ValueError(f"recorded web content must be an object for {case['id']}")
                entries.append(
                    _RecordedWebEntry(
                        query=recorded_query.strip(),
                        success=bool(result.get("success")),
                        content=copy.deepcopy(raw_content),
                        error=str(result["error"]) if result.get("error") is not None else None,
                    )
                )

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        question: str | None = None,
    ) -> ToolResult:
        if tool_name in {"run_python", "search_docs"}:
            return self._production.execute(tool_name, arguments, question=question)
        if tool_name != "web_search":
            return _invalid_result(tool_name, f"unknown tool: {tool_name}")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return self._recorded_failure("", "query must be a non-empty string")
        if question is None or question not in self._web_entries:
            return self._recorded_failure(
                query.strip(),
                "recorded corpus has no web entries for this benchmark question",
            )
        entries = self._web_entries[question]
        if not entries:
            return self._recorded_failure(
                query.strip(),
                "recorded corpus has no web entries for this benchmark question",
            )
        candidates = [
            (*_semantic_match_score(query, entry), index, entry)
            for index, entry in enumerate(entries)
        ]
        overlap, score, _, entry = max(
            candidates,
            key=lambda candidate: (candidate[1], -candidate[2]),
        )
        required_overlap = min(2, len(_semantic_tokens(query)))
        if overlap < required_overlap or score < self.minimum_similarity:
            return self._recorded_failure(
                query.strip(),
                "recorded corpus contains no sufficiently similar web query",
            )
        started = time.monotonic()
        content = copy.deepcopy(entry.content)
        content["query"] = query.strip()
        content["retrieval_mode"] = self.retrieval_mode
        content["recorded_query"] = entry.query
        return ToolResult(
            source_id=f"src_recorded_{uuid.uuid4().hex}",
            tool_name="web_search",
            success=entry.success,
            content=content,
            error=entry.error,
            duration_ms=(time.monotonic() - started) * 1000,
        )

    def _recorded_failure(self, query: str, error: str) -> ToolResult:
        return ToolResult(
            source_id=f"src_recorded_{uuid.uuid4().hex}",
            tool_name="web_search",
            success=False,
            content={
                "query": query,
                "results": [],
                "retrieval_mode": self.retrieval_mode,
            },
            error=error,
        )


def _canonical(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _invalid_result(tool_name: str, message: str) -> ToolResult:
    from uuid import uuid4

    return ToolResult(
        source_id=f"src_{uuid4().hex}",
        tool_name=tool_name,
        success=False,
        content={},
        error=message,
    )
