from __future__ import annotations

from typing import Any

from research_agent.schemas import Claim, JudgeVerdict, ToolResult, Usage
from research_agent.verifier import CitationVerifier


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def log(self, event: str, **payload: Any) -> None:
        self.records.append((event, payload))


class _SelectiveJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[Claim, ToolResult]] = []

    def supports(self, *, claim: Claim, source: ToolResult) -> JudgeVerdict:
        self.calls.append((claim, source))
        supported = claim.text == "Supported claim"
        return JudgeVerdict(
            value=supported,
            reason="direct support" if supported else "the raw content does not entail this claim",
            usage=Usage(input_tokens=2, output_tokens=1, cost_usd=0.001, latency_ms=3),
        )


def test_verifier_deletes_invalid_failed_and_unsupported_citations_and_logs_each() -> None:
    successful_source = ToolResult(
        source_id="source-ok",
        tool_name="search_docs",
        success=True,
        content={"chunks": [{"text": "Supported claim"}]},
    )
    failed_source = ToolResult(
        source_id="source-failed",
        tool_name="web_search",
        success=False,
        content={"results": []},
        error="network unavailable",
    )
    claims = [
        Claim(text="Unknown citation", source_id="source-missing"),
        Claim(text="Failed citation", source_id="source-failed"),
        Claim(text="Unsupported claim", source_id="source-ok"),
        Claim(text="Supported claim", source_id="source-ok"),
    ]
    judge = _SelectiveJudge()
    logger = _RecordingLogger()

    result = CitationVerifier(judge).verify(
        claims=claims,
        history=[successful_source, failed_source],
        logger=logger,  # type: ignore[arg-type]
    )

    assert result.claims == [claims[-1]]
    assert [deletion.reason for deletion in result.deletions] == [
        "unknown_source_id",
        "failed_source",
        "unsupported_by_source",
    ]
    assert [claim.text for claim, _ in judge.calls] == ["Unsupported claim", "Supported claim"]
    assert result.usage == Usage(
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.002,
        latency_ms=6,
    )

    deletion_events = [payload for event, payload in logger.records if event == "claim_deleted"]
    assert len(deletion_events) == len(result.deletions)
    assert [payload["deletion"].reason for payload in deletion_events] == [
        "unknown_source_id",
        "failed_source",
        "unsupported_by_source",
    ]
    assert [event for event, _ in logger.records].count("claim_verified") == 1


class _BrokenJudge:
    def supports(self, *, claim: Claim, source: ToolResult) -> JudgeVerdict:
        del claim, source
        raise TimeoutError("judge timed out")


def test_verifier_fails_closed_and_logs_when_the_judge_errors() -> None:
    logger = _RecordingLogger()
    claim = Claim(text="A claim", source_id="source-ok")
    source = ToolResult(
        source_id="source-ok",
        tool_name="run_python",
        success=True,
        content={"stdout": "A claim\n", "stderr": "", "exit_code": 0},
    )

    result = CitationVerifier(_BrokenJudge()).verify(
        claims=[claim],
        history=[source],
        logger=logger,  # type: ignore[arg-type]
    )

    assert result.claims == []
    assert len(result.deletions) == 1
    assert result.deletions[0].reason == "judge_error"
    assert "TimeoutError: judge timed out" in result.deletions[0].detail
    assert [event for event, _ in logger.records] == ["claim_deleted"]
