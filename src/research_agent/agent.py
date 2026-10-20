"""Bounded research loop that never exposes an unverified claim."""

from __future__ import annotations

import time
from collections.abc import Sequence
from uuid import uuid4

from research_agent.executors import ToolExecutor
from research_agent.providers import ModelClient
from research_agent.run_logging import JsonlRunLogger
from research_agent.schemas import (
    AgentAnswer,
    Claim,
    StepRecord,
    ToolResult,
    Usage,
)
from research_agent.verifier import CitationVerifier

VERBOSE_TOOL_DESCRIPTIONS = """\
- web_search(query): Search the public web. Returns {results: [{url, title, snippet}]}.
  Use it for current or externally verifiable facts. A snippet supports only what it states.
- run_python(code): Run Python in a resource-limited local subprocess. Returns
  {stdout, stderr, exit_code}. It accepts calculation-oriented Python and the math, statistics,
  decimal, and fractions modules; filesystem, network, process, dynamic-code, private-attribute,
  class, and function-definition escape paths are rejected. Print a self-contained labeled result
  or equation because citation verification sees stdout, not the executed code.
- search_docs(query, top_k=5): Retrieve local ChromaDB chunks. Returns
  {chunks: [{text, document_source, distance, metadata}]}. Cite only a matching chunk.
Each tool result has a unique source_id and success flag. Failed results are evidence of failure,
not evidence for an answer. You may call only one tool per decision.
"""


class ResearchAgent:
    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolExecutor,
        verifier: CitationVerifier,
        max_steps: int = 8,
        tool_descriptions: str = VERBOSE_TOOL_DESCRIPTIONS,
        planning_subgoals: bool = False,
    ) -> None:
        if not 1 <= max_steps <= 32:
            raise ValueError("max_steps must be between 1 and 32")
        self.model = model
        self.tools = tools
        self.verifier = verifier
        self.max_steps = max_steps
        self.tool_descriptions = tool_descriptions
        self.planning_subgoals = planning_subgoals

    def answer(
        self,
        question: str,
        *,
        logger: JsonlRunLogger | None = None,
    ) -> AgentAnswer:
        if not question.strip():
            raise ValueError("question must not be empty")
        started = time.perf_counter()
        history: list[ToolResult] = []
        steps: list[StepRecord] = []
        total_usage = Usage()
        if logger:
            logger.log("question_started", question=question, max_steps=self.max_steps)

        for step_number in range(1, self.max_steps + 1):
            try:
                turn = self.model.decide(
                    question=question,
                    tool_descriptions=self.tool_descriptions,
                    history=history,
                    steps=steps,
                )
                total_usage = total_usage + turn.usage
                if (
                    self.planning_subgoals
                    and turn.decision.kind == "tool_call"
                    and getattr(turn.decision, "sub_goal", None) is None
                ):
                    raise ValueError(
                        "planning_subgoals requires a one-sentence sub_goal before every tool call"
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                record = StepRecord(
                    step=step_number,
                    action="model_error",
                    success=False,
                    error=error,
                )
                steps.append(record)
                if logger:
                    logger.log("model_error", step=step_number, error=error)
                return self._finish(
                    question=question,
                    status="insufficient_information",
                    answer="insufficient information",
                    steps=steps,
                    history=history,
                    usage=total_usage,
                    started=started,
                    logger=logger,
                )

            decision = turn.decision
            if logger:
                logger.log(
                    "model_decision",
                    step=step_number,
                    decision=decision,
                    usage=turn.usage,
                    raw_response=turn.raw_response,
                )

            if decision.kind == "tool_call":
                arguments = decision.arguments.compact()
                tool_started = time.perf_counter()
                try:
                    result = ToolResult.model_validate(
                        self.tools.execute(
                            decision.tool_name or "",
                            arguments,
                            question=question,
                        )
                    )
                    if any(previous.source_id == result.source_id for previous in history):
                        raise ValueError(f"duplicate source_id returned: {result.source_id}")
                except Exception as exc:  # tools must be recoverable even if an adapter is buggy
                    result = ToolResult(
                        source_id=f"src_{uuid4().hex}",
                        tool_name=decision.tool_name or "unknown",
                        success=False,
                        content={},
                        error=f"Unhandled tool error: {type(exc).__name__}: {exc}",
                    )
                elapsed_ms = (time.perf_counter() - tool_started) * 1000
                if result.duration_ms == 0:
                    result = result.model_copy(update={"duration_ms": elapsed_ms})
                history.append(result)
                record = StepRecord(
                    step=step_number,
                    action="tool_call",
                    sub_goal=getattr(decision, "sub_goal", None),
                    tool_name=decision.tool_name,
                    arguments=arguments,
                    source_id=result.source_id,
                    success=result.success,
                    error=result.error,
                    latency_ms=turn.usage.latency_ms + result.duration_ms,
                )
                steps.append(record)
                if logger:
                    logger.log("tool_result", step=step_number, result=result)
                continue

            generated_claims = decision.claims
            if not generated_claims:
                steps.append(
                    StepRecord(
                        step=step_number,
                        action="final",
                        success=True,
                        latency_ms=turn.usage.latency_ms,
                    )
                )
                return self._finish(
                    question=question,
                    status="declined",
                    answer="insufficient information",
                    generated_claims=generated_claims,
                    steps=steps,
                    history=history,
                    usage=total_usage,
                    started=started,
                    logger=logger,
                )

            verification = self.verifier.verify(
                claims=generated_claims,
                history=history,
                logger=logger,
            )
            total_usage = total_usage + verification.usage
            steps.append(
                StepRecord(
                    step=step_number,
                    action="final",
                    success=bool(verification.claims),
                    latency_ms=turn.usage.latency_ms + verification.usage.latency_ms,
                )
            )
            if not verification.claims:
                return self._finish(
                    question=question,
                    status="unverified",
                    answer="I could not verify an answer.",
                    generated_claims=generated_claims,
                    deletions=verification.deletions,
                    steps=steps,
                    history=history,
                    usage=total_usage,
                    started=started,
                    logger=logger,
                )
            answer_text = " ".join(claim.text.strip() for claim in verification.claims)
            return self._finish(
                question=question,
                status="answered",
                answer=answer_text,
                claims=verification.claims,
                generated_claims=generated_claims,
                deletions=verification.deletions,
                steps=steps,
                history=history,
                usage=total_usage,
                started=started,
                logger=logger,
            )

        if logger:
            logger.log("max_steps_reached", max_steps=self.max_steps)
        return self._finish(
            question=question,
            status="insufficient_information",
            answer="insufficient information",
            steps=steps,
            history=history,
            usage=total_usage,
            started=started,
            logger=logger,
        )

    def _finish(
        self,
        *,
        question: str,
        status: str,
        answer: str,
        steps: Sequence[StepRecord],
        history: Sequence[ToolResult],
        usage: Usage,
        started: float,
        logger: JsonlRunLogger | None,
        claims: Sequence[Claim] = (),
        generated_claims: Sequence[Claim] = (),
        deletions: Sequence = (),
    ) -> AgentAnswer:
        result = AgentAnswer(
            question=question,
            status=status,
            answer=answer,
            claims=list(claims),
            generated_claims=list(generated_claims),
            deletions=list(deletions),
            steps=list(steps),
            sources=list(history),
            max_steps=self.max_steps,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if logger:
            logger.log("question_completed", result=result)
        return result
