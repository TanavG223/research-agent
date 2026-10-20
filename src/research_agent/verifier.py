"""Fail-closed citation verifier."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.providers import JudgeClient
from research_agent.run_logging import JsonlRunLogger
from research_agent.schemas import (
    Claim,
    ToolResult,
    Usage,
    VerificationDeletion,
    VerificationResult,
)


class CitationVerifier:
    def __init__(self, judge: JudgeClient) -> None:
        self.judge = judge

    def verify(
        self,
        *,
        claims: Sequence[Claim],
        history: Sequence[ToolResult],
        logger: JsonlRunLogger | None = None,
    ) -> VerificationResult:
        by_source = {result.source_id: result for result in history}
        kept: list[Claim] = []
        deletions: list[VerificationDeletion] = []
        usage = Usage()

        for claim in claims:
            source = by_source.get(claim.source_id)
            if source is None:
                deletion = VerificationDeletion(
                    claim=claim,
                    reason="unknown_source_id",
                    detail="The cited source_id does not exist in this question's step history.",
                )
                deletions.append(deletion)
                _log_deletion(logger, deletion)
                continue
            if not source.success:
                deletion = VerificationDeletion(
                    claim=claim,
                    reason="failed_source",
                    detail="The cited tool result was unsuccessful.",
                )
                deletions.append(deletion)
                _log_deletion(logger, deletion)
                continue
            try:
                verdict = self.judge.supports(claim=claim, source=source)
                usage = usage + verdict.usage
            except Exception as exc:  # fail closed when the judge is unavailable
                deletion = VerificationDeletion(
                    claim=claim,
                    reason="judge_error",
                    detail=f"Citation judge failed: {type(exc).__name__}: {exc}",
                )
                deletions.append(deletion)
                _log_deletion(logger, deletion)
                continue
            if not verdict.value:
                deletion = VerificationDeletion(
                    claim=claim,
                    reason="unsupported_by_source",
                    detail=verdict.reason,
                )
                deletions.append(deletion)
                _log_deletion(logger, deletion)
                continue
            kept.append(claim)
            if logger:
                logger.log(
                    "claim_verified",
                    claim=claim,
                    source_id=source.source_id,
                    judge_reason=verdict.reason,
                )

        return VerificationResult(claims=kept, deletions=deletions, usage=usage)


def _log_deletion(logger: JsonlRunLogger | None, deletion: VerificationDeletion) -> None:
    if logger:
        logger.log("claim_deleted", deletion=deletion)

