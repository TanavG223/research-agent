"""Shared, JSON-serializable contracts for tools, models, verification, and runs."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolResult(StrictModel):
    source_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    success: bool
    content: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Claim(StrictModel):
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class ToolArguments(StrictModel):
    query: str | None = None
    code: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)

    def compact(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class AgentDecision(StrictModel):
    kind: Literal["tool_call", "final"]
    tool_name: Literal["web_search", "run_python", "search_docs"] | None = None
    arguments: ToolArguments = Field(default_factory=ToolArguments)
    claims: list[Claim] = Field(default_factory=list)
    decline_reason: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> AgentDecision:
        if self.kind == "tool_call":
            if self.tool_name is None:
                raise ValueError("tool_call decisions require tool_name")
            if self.claims or self.decline_reason:
                raise ValueError("tool_call decisions cannot include a final answer")
        else:
            if self.tool_name is not None or self.arguments.compact():
                raise ValueError("final decisions cannot include a tool call")
            if bool(self.claims) == bool(self.decline_reason):
                raise ValueError("final decisions require exactly one of claims or decline_reason")
        return self


class PlanningAgentDecision(AgentDecision):
    """Decision contract used only by the planning-subgoal experiment."""

    sub_goal: str | None = Field(
        default=None,
        min_length=1,
        max_length=240,
    )

    @field_validator("sub_goal", mode="before")
    @classmethod
    def normalize_sub_goal(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("sub_goal must not be blank")
        if re.search(
            r"[!?…](?=\s*\S)|(?<![A-Z]\.[A-Z])\.(?=\s+\S)|\.(?=[A-Z](?!\.))",
            normalized,
        ):
            raise ValueError("sub_goal must contain only one sentence")
        return normalized

    @model_validator(mode="after")
    def require_tool_sub_goal(self) -> PlanningAgentDecision:
        if self.kind == "tool_call" and self.sub_goal is None:
            raise ValueError("tool_call decisions require a sub_goal when planning is enabled")
        if self.kind == "final" and self.sub_goal is not None:
            raise ValueError("final decisions cannot include a sub_goal")
        return self


class Usage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
        )


class ModelTurn(StrictModel):
    decision: AgentDecision | PlanningAgentDecision
    usage: Usage = Field(default_factory=Usage)
    raw_response: str | None = None


class JudgeVerdict(StrictModel):
    value: bool
    reason: str
    usage: Usage = Field(default_factory=Usage)


class VerificationDeletion(StrictModel):
    claim: Claim
    reason: Literal[
        "unknown_source_id",
        "failed_source",
        "unsupported_by_source",
        "judge_error",
    ]
    detail: str


class VerificationResult(StrictModel):
    claims: list[Claim] = Field(default_factory=list)
    deletions: list[VerificationDeletion] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class StepRecord(StrictModel):
    step: int = Field(ge=1)
    action: Literal["tool_call", "final", "model_error"]
    sub_goal: str | None = Field(
        default=None,
        min_length=1,
        max_length=240,
        exclude_if=lambda value: value is None,
    )
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    source_id: str | None = None
    success: bool
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class AgentAnswer(StrictModel):
    question: str
    status: Literal["answered", "declined", "insufficient_information", "unverified"]
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    generated_claims: list[Claim] = Field(default_factory=list)
    deletions: list[VerificationDeletion] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    sources: list[ToolResult] = Field(default_factory=list)
    max_steps: int = Field(ge=1)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = Field(default=0.0, ge=0.0)

    def public_dump(self) -> dict[str, Any]:
        """Serialize an outward response without exposing rejected claim text."""
        payload = self.model_dump(mode="json")
        payload.pop("generated_claims", None)
        payload["claims_generated"] = len(self.generated_claims)
        payload["claims_deleted"] = len(self.deletions)
        safe_details = {
            "unknown_source_id": "The citation was not present in this run's source history.",
            "failed_source": "The cited tool source did not complete successfully.",
            "unsupported_by_source": "The cited source did not support the generated claim.",
            "judge_error": "The citation check did not complete, so the claim was removed.",
        }
        payload["deletions"] = [
            {"reason": deletion.reason, "detail": safe_details[deletion.reason]}
            for deletion in self.deletions
        ]
        return payload


class BenchmarkItem(StrictModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    known_correct_answer: str = Field(min_length=1)
    category: Literal["factual_lookup", "multi_step", "computational", "unanswerable"]
    answerable: bool

    @model_validator(mode="after")
    def validate_answerability(self) -> BenchmarkItem:
        if (self.category == "unanswerable") == self.answerable:
            raise ValueError("only unanswerable items may have answerable=false")
        if not self.known_correct_answer.strip():
            raise ValueError("known_correct_answer must not be blank")
        return self
