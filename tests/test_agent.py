from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from research_agent.agent import ResearchAgent
from research_agent.run_logging import JsonlRunLogger
from research_agent.schemas import (
    AgentDecision,
    Claim,
    JudgeVerdict,
    ModelTurn,
    PlanningAgentDecision,
    StepRecord,
    ToolArguments,
    ToolResult,
)
from research_agent.verifier import CitationVerifier


class _RepeatingToolModel:
    def __init__(self) -> None:
        self.history_sizes: list[int] = []

    def decide(
        self,
        *,
        question: str,
        tool_descriptions: str,
        history: Sequence[ToolResult],
        steps: Sequence[StepRecord],
    ) -> ModelTurn:
        del question, tool_descriptions, steps
        self.history_sizes.append(len(history))
        return ModelTurn(
            decision=AgentDecision(
                kind="tool_call",
                tool_name="web_search",
                arguments=ToolArguments(query="keep searching"),
            )
        )


class _SuccessfulTools:
    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        question: str | None = None,
    ) -> ToolResult:
        del arguments, question
        self.calls += 1
        return ToolResult(
            source_id=f"source-{self.calls}",
            tool_name=tool_name,
            success=True,
            content={"results": []},
        )


class _VerifierMustNotRun:
    def verify(self, **_: Any) -> None:
        raise AssertionError("the verifier must not run before a final decision")


def test_agent_stops_exactly_at_max_steps_and_logs_each_step(tmp_path: Any) -> None:
    model = _RepeatingToolModel()
    tools = _SuccessfulTools()
    logger = JsonlRunLogger(tmp_path, "bounded-run", "question")
    agent = ResearchAgent(
        model=model,
        tools=tools,
        verifier=_VerifierMustNotRun(),  # type: ignore[arg-type]
        max_steps=3,
    )

    result = agent.answer("A question that never reaches a final decision", logger=logger)

    assert result.status == "insufficient_information"
    assert result.answer == "insufficient information"
    assert result.max_steps == 3
    assert len(result.steps) == 3
    assert len(result.sources) == 3
    assert model.history_sizes == [0, 1, 2]
    assert tools.calls == 3
    assert [step.step for step in result.steps] == [1, 2, 3]
    assert all(step.action == "tool_call" for step in result.steps)

    records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert events.count("model_decision") == 3
    assert events.count("tool_result") == 3
    assert events.count("max_steps_reached") == 1
    assert events[-1] == "question_completed"


class _PlanningModel:
    def decide(
        self,
        *,
        question: str,
        tool_descriptions: str,
        history: Sequence[ToolResult],
        steps: Sequence[StepRecord],
    ) -> ModelTurn:
        del question, tool_descriptions
        if not history:
            return ModelTurn(
                decision=PlanningAgentDecision(
                    kind="tool_call",
                    sub_goal="Find the first required fact.",
                    tool_name="web_search",
                    arguments=ToolArguments(query="first required fact"),
                )
            )
        assert steps[0].sub_goal == "Find the first required fact."
        return ModelTurn(
            decision=AgentDecision(kind="final", decline_reason="No more evidence.")
        )


def test_planning_subgoal_is_stored_and_passed_to_the_next_model_turn() -> None:
    tools = _SuccessfulTools()
    agent = ResearchAgent(
        model=_PlanningModel(),
        tools=tools,
        verifier=_VerifierMustNotRun(),  # type: ignore[arg-type]
        max_steps=2,
        planning_subgoals=True,
    )

    result = agent.answer("Resolve this in stages")

    assert result.status == "declined"
    assert result.steps[0].action == "tool_call"
    assert result.steps[0].sub_goal == "Find the first required fact."
    assert tools.calls == 1


def test_default_step_serialization_does_not_add_a_null_subgoal() -> None:
    step = StepRecord(step=1, action="final", success=True)

    assert "sub_goal" not in step.model_dump(mode="json")


def test_planning_mode_refuses_a_tool_call_without_a_subgoal() -> None:
    tools = _SuccessfulTools()
    agent = ResearchAgent(
        model=_RepeatingToolModel(),
        tools=tools,
        verifier=_VerifierMustNotRun(),  # type: ignore[arg-type]
        planning_subgoals=True,
    )

    result = agent.answer("Require a plan")

    assert result.status == "insufficient_information"
    assert result.steps[0].action == "model_error"
    assert "planning_subgoals requires" in (result.steps[0].error or "")
    assert result.sources == []
    assert tools.calls == 0


class _RecoveryModel:
    def __init__(self) -> None:
        self.histories: list[list[ToolResult]] = []
        self.step_histories: list[list[StepRecord]] = []

    def decide(
        self,
        *,
        question: str,
        tool_descriptions: str,
        history: Sequence[ToolResult],
        steps: Sequence[StepRecord],
    ) -> ModelTurn:
        del question, tool_descriptions
        self.histories.append([item.model_copy(deep=True) for item in history])
        self.step_histories.append([item.model_copy(deep=True) for item in steps])
        if not history:
            decision = AgentDecision(
                kind="tool_call",
                tool_name="run_python",
                arguments=ToolArguments(code="raise RuntimeError('adapter failure')"),
            )
        elif len(history) == 1:
            assert history[0].success is False
            assert "Unhandled tool error: RuntimeError: subprocess adapter broke" in (
                history[0].error or ""
            )
            decision = AgentDecision(
                kind="tool_call",
                tool_name="search_docs",
                arguments=ToolArguments(query="verified value"),
            )
        else:
            assert history[1].success is True
            decision = AgentDecision(
                kind="final",
                claims=[
                    Claim(
                        text="The verified value is 42.",
                        source_id=history[1].source_id,
                    )
                ],
            )
        return ModelTurn(decision=decision)


class _RecoveryTools:
    def __init__(self) -> None:
        self.tool_names: list[str] = []

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        question: str | None = None,
    ) -> ToolResult:
        del arguments, question
        self.tool_names.append(tool_name)
        if tool_name == "run_python":
            raise RuntimeError("subprocess adapter broke")
        return ToolResult(
            source_id="source-recovered",
            tool_name=tool_name,
            success=True,
            content={"chunks": [{"text": "The verified value is 42."}]},
        )


class _SupportingJudge:
    def supports(self, *, claim: Claim, source: ToolResult) -> JudgeVerdict:
        return JudgeVerdict(
            value=claim.text in str(source.content),
            reason="claim is present in the cited raw content",
        )


def test_tool_exception_is_appended_and_model_can_recover_with_another_tool() -> None:
    model = _RecoveryModel()
    tools = _RecoveryTools()
    agent = ResearchAgent(
        model=model,
        tools=tools,
        verifier=CitationVerifier(_SupportingJudge()),  # type: ignore[arg-type]
        max_steps=3,
    )

    result = agent.answer("Find the verified value")

    assert result.status == "answered"
    assert result.answer == "The verified value is 42."
    assert result.claims == [Claim(text="The verified value is 42.", source_id="source-recovered")]
    assert tools.tool_names == ["run_python", "search_docs"]
    assert [step.action for step in result.steps] == ["tool_call", "tool_call", "final"]
    assert [source.success for source in result.sources] == [False, True]
    assert result.sources[0].error is not None
    assert "subprocess adapter broke" in result.sources[0].error
    assert len(model.histories) == 3
    assert [len(history) for history in model.step_histories] == [0, 1, 2]
    assert model.step_histories[1][0].arguments == {
        "code": "raise RuntimeError('adapter failure')"
    }
    assert model.step_histories[2][1].arguments == {"query": "verified value"}


class _UnknownCitationModel:
    def decide(
        self,
        *,
        question: str,
        tool_descriptions: str,
        history: Sequence[ToolResult],
        steps: Sequence[StepRecord],
    ) -> ModelTurn:
        del question, tool_descriptions, history, steps
        return ModelTurn(
            decision=AgentDecision(
                kind="final",
                claims=[Claim(text="An unsupported answer.", source_id="missing-source")],
            )
        )


class _ToolsMustNotRun:
    def execute(self, *_: Any, **__: Any) -> ToolResult:
        raise AssertionError("no tool call was requested")


class _JudgeMustNotRun:
    def supports(self, **_: Any) -> JudgeVerdict:
        raise AssertionError("unknown source ids must be rejected before judging")


def test_all_deleted_claims_return_the_exact_fail_closed_fallback() -> None:
    agent = ResearchAgent(
        model=_UnknownCitationModel(),
        tools=_ToolsMustNotRun(),  # type: ignore[arg-type]
        verifier=CitationVerifier(_JudgeMustNotRun()),  # type: ignore[arg-type]
    )

    result = agent.answer("Give me an answer")

    assert result.status == "unverified"
    assert result.answer == "I could not verify an answer."
    assert result.claims == []
    assert len(result.generated_claims) == 1
    assert [deletion.reason for deletion in result.deletions] == ["unknown_source_id"]

    public = result.public_dump()
    assert "generated_claims" not in public
    assert public["claims"] == []
    assert public["claims_generated"] == 1
    assert public["claims_deleted"] == 1
    assert public["deletions"] == [
        {
            "reason": "unknown_source_id",
            "detail": "The citation was not present in this run's source history.",
        }
    ]
    assert "An unsupported answer." not in str(public)
