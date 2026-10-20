from __future__ import annotations

from typing import Any

import pytest

from research_agent.providers import OllamaClient, OpenAIClient
from research_agent.schemas import (
    AgentDecision,
    Claim,
    PlanningAgentDecision,
    StepRecord,
    ToolArguments,
    ToolResult,
    Usage,
)


def _client_without_network() -> OpenAIClient:
    client = object.__new__(OpenAIClient)
    client.model = "test-agent"
    client.judge_model = "test-judge"
    client.input_cost_per_million = 0.0
    client.output_cost_per_million = 0.0
    return client


def test_openai_model_receives_prior_call_arguments_and_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_response(
        self: OpenAIClient,
        *,
        model: str,
        instructions: str,
        input_text: str,
        response_model: type,
    ) -> tuple[AgentDecision, Usage, str]:
        del self, instructions, response_model
        seen.update(model=model, input_text=input_text)
        return AgentDecision(kind="final", decline_reason="No more evidence."), Usage(), "{}"

    monkeypatch.setattr(OpenAIClient, "_responses_model", fake_response)
    client = _client_without_network()
    result = ToolResult(
        source_id="source-python",
        tool_name="run_python",
        success=True,
        content={"stdout": "42\n", "stderr": "", "exit_code": 0},
    )
    step = StepRecord(
        step=1,
        action="tool_call",
        tool_name="run_python",
        arguments={"code": "print(40 + 2)"},
        source_id=result.source_id,
        success=True,
    )

    client.decide(
        question="What is the result?",
        tool_descriptions="run_python(code)",
        history=[result],
        steps=[step],
    )

    assert seen["model"] == "test-agent"
    assert '"code": "print(40 + 2)"' in seen["input_text"]
    assert '"source_id": "source-python"' in seen["input_text"]
    assert '"stdout": "42\\n"' in seen["input_text"]


def test_citation_judge_cannot_treat_search_query_as_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class ParsedVerdict:
        supported = False
        reason = "no result supports the claim"

    def fake_response(
        self: OpenAIClient,
        *,
        model: str,
        instructions: str,
        input_text: str,
        response_model: type,
    ) -> tuple[ParsedVerdict, Usage, str]:
        del self, model, instructions, response_model
        seen["input_text"] = input_text
        return ParsedVerdict(), Usage(), "{}"

    monkeypatch.setattr(OpenAIClient, "_responses_model", fake_response)
    client = _client_without_network()
    source = ToolResult(
        source_id="source-search",
        tool_name="web_search",
        success=True,
        content={
            "query": "The secret answer is Zeta-9",
            "results": [{"title": "No match", "url": "https://example.test", "snippet": ""}],
        },
    )

    verdict = client.supports(
        claim=Claim(text="The secret answer is Zeta-9.", source_id=source.source_id),
        source=source,
    )

    assert verdict.value is False
    assert "The secret answer is Zeta-9" in seen["input_text"].split("CITED SOURCE:", 1)[0]
    assert "The secret answer is Zeta-9" not in seen["input_text"].split("CITED SOURCE:", 1)[1]


def _ollama_response(content: str, *, input_tokens: int = 0, output_tokens: int = 0) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": input_tokens,
        "eval_count": output_tokens,
    }


def test_ollama_decide_uses_native_strict_json_and_measured_usage() -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []

    def transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict:
        calls.append((url, payload, timeout_s))
        return _ollama_response(
            '{"kind":"tool_call","tool_name":"web_search",'
            '"arguments":{"query":"capital of France"},"claims":[],"decline_reason":null}',
            input_tokens=321,
            output_tokens=17,
        )

    client = OllamaClient(
        model="qwen3:4b",
        base_url="http://127.0.0.1:11434",
        timeout_s=12,
        num_ctx=4096,
        transport=transport,
        runtime_version="0.21.1",
    )
    result = ToolResult(
        source_id="prior-source",
        tool_name="run_python",
        success=True,
        content={"stdout": "42\n", "stderr": "", "exit_code": 0},
    )
    step = StepRecord(
        step=1,
        action="tool_call",
        tool_name="run_python",
        arguments={"code": "print(40 + 2)"},
        source_id=result.source_id,
        success=True,
    )

    turn = client.decide(
        question="What is the capital?",
        tool_descriptions="web_search(query)",
        history=[result],
        steps=[step],
    )

    assert turn.decision.kind == "tool_call"
    assert turn.decision.tool_name == "web_search"
    assert turn.decision.arguments.query == "capital of France"
    assert turn.usage.input_tokens == 321
    assert turn.usage.output_tokens == 17
    assert turn.usage.cost_usd == 0
    assert turn.usage.latency_ms >= 0
    assert len(calls) == 1
    url, payload, timeout_s = calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert timeout_s == 12
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 4096,
        "num_predict": 512,
    }
    assert payload["format"]["title"] == "AgentDecision"
    assert [branch["title"] for branch in payload["format"]["oneOf"]] == [
        "FinalClaimsDecision",
        "ToolCallDecision",
        "FinalDeclineDecision",
    ]
    assert "sub_goal" not in payload["format"]["oneOf"][1]["properties"]
    assert "Return only JSON matching this exact schema" in payload["messages"][0]["content"]
    assert "Do not finalize an intermediate fact" in payload["messages"][0]["content"]
    assert "self-contained labeled statement" in payload["messages"][0]["content"]
    assert '"code": "print(40 + 2)"' in payload["messages"][1]["content"]
    assert '"sub_goal"' not in payload["messages"][1]["content"]
    assert client.manifest_metadata["runtime_version"] == "0.21.1"
    assert client.manifest_metadata["cost_basis"].startswith("local_unbilled")
    assert client.manifest_metadata["planning_subgoals"] is False


def test_ollama_planning_subgoal_is_strict_and_present_in_next_turn_context() -> None:
    calls: list[dict[str, Any]] = []

    def transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict:
        del url, timeout_s
        calls.append(payload)
        return _ollama_response(
            '{"kind":"tool_call","sub_goal":"Identify the next missing fact.",'
            '"tool_name":"web_search","arguments":{"query":"next missing fact"},'
            '"claims":[],"decline_reason":null}'
        )

    client = OllamaClient(
        planning_subgoals=True,
        transport=transport,
        runtime_version="test",
    )
    prior_source = ToolResult(
        source_id="prior-source",
        tool_name="web_search",
        success=True,
        content={"results": [{"snippet": "An intermediate fact."}]},
    )
    prior_step = StepRecord(
        step=1,
        action="tool_call",
        sub_goal="Identify the intermediate fact.",
        tool_name="web_search",
        arguments={"query": "intermediate fact"},
        source_id=prior_source.source_id,
        success=True,
    )

    turn = client.decide(
        question="Resolve a two-stage question.",
        tool_descriptions="web_search(query)",
        history=[prior_source],
        steps=[prior_step],
    )

    assert turn.decision.kind == "tool_call"
    assert turn.decision.sub_goal == "Identify the next missing fact."
    payload = calls[0]
    assert payload["format"]["title"] == "PlanningAgentDecision"
    assert all("sub_goal" in branch["required"] for branch in payload["format"]["oneOf"])
    tool_schema = payload["format"]["oneOf"][1]
    assert tool_schema["properties"]["sub_goal"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": 240,
    }
    assert payload["format"]["oneOf"][0]["properties"]["sub_goal"] == {"type": "null"}
    assert "Before every tool call" in payload["messages"][0]["content"]
    assert '"sub_goal": "Identify the intermediate fact."' in payload["messages"][1][
        "content"
    ]
    assert client.manifest_metadata["planning_subgoals"] is True


def test_ollama_planning_schema_rejects_missing_subgoal_and_final_subgoal() -> None:
    responses = iter(
        (
            _ollama_response(
                '{"kind":"tool_call","tool_name":"web_search",'
                '"arguments":{"query":"missing plan"},"claims":[],"decline_reason":null}'
            ),
            _ollama_response(
                '{"kind":"final","sub_goal":"This must not be present.",'
                '"tool_name":null,"arguments":{},"claims":[],'
                '"decline_reason":"No evidence."}'
            ),
        )
    )

    def transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict:
        del url, payload, timeout_s
        return next(responses)

    client = OllamaClient(
        planning_subgoals=True,
        transport=transport,
        runtime_version="test",
    )

    for expected in ("missing", "final"):
        with pytest.raises(RuntimeError, match="invalid PlanningAgentDecision"):
            client.decide(
                question=f"Test {expected} sub-goal",
                tool_descriptions="web_search(query)",
                history=[],
                steps=[],
            )


@pytest.mark.parametrize(
    "sub_goal",
    (
        "Find the first fact. Then find the second fact.",
        "Find the first fact! Then find the second fact.",
        "Find the first fact?Then find the second fact.",
        "Find the first fact… Then find the second fact.",
    ),
)
def test_planning_subgoal_contract_rejects_non_sentence_values(sub_goal: str) -> None:
    with pytest.raises(ValueError, match="sub_goal"):
        PlanningAgentDecision(
            kind="tool_call",
            sub_goal=sub_goal,
            tool_name="web_search",
            arguments=ToolArguments(query="next fact"),
        )


def test_planning_subgoal_contract_accepts_one_sentence_without_terminal_mark() -> None:
    decision = PlanningAgentDecision(
        kind="tool_call",
        sub_goal="Identify the country with the maple leaf flag",
        tool_name="web_search",
        arguments=ToolArguments(query="maple leaf flag country"),
    )

    assert decision.sub_goal == "Identify the country with the maple leaf flag"

    abbreviation = PlanningAgentDecision(
        kind="tool_call",
        sub_goal="Find the U.S. capital",
        tool_name="web_search",
        arguments=ToolArguments(query="U.S. capital"),
    )
    assert abbreviation.sub_goal == "Find the U.S. capital"


def test_ollama_judges_parse_booleans_and_hide_query_from_citation_evidence() -> None:
    calls: list[dict[str, Any]] = []

    def transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict:
        del url, timeout_s
        calls.append(payload)
        title = payload["format"]["title"]
        if title == "_CitationSupport":
            return _ollama_response(
                '{"supported":false,"reason":"the result does not support the claim"}',
                input_tokens=20,
                output_tokens=5,
            )
        assert title == "_AnswerCorrectness"
        return _ollama_response(
            '{"correct":true,"reason":"the candidate matches the reference"}',
            input_tokens=15,
            output_tokens=4,
        )

    client = OllamaClient(transport=transport, runtime_version="test")
    source = ToolResult(
        source_id="source-search",
        tool_name="web_search",
        success=True,
        content={
            "query": "The secret answer is Zeta-9",
            "results": [{"url": "https://example.test", "title": "No match", "snippet": ""}],
        },
    )

    support = client.supports(
        claim=Claim(text="The secret answer is Zeta-9.", source_id=source.source_id),
        source=source,
    )
    correctness = client.correct(
        question="What is two plus two?",
        known_answer="4",
        candidate_answer="Four",
    )

    assert support.value is False
    assert support.usage.input_tokens == 20
    assert support.usage.output_tokens == 5
    assert support.usage.cost_usd == 0
    assert support.usage.latency_ms >= 0
    citation_input = calls[0]["messages"][1]["content"]
    assert "The secret answer is Zeta-9" in citation_input.split("CITED SOURCE:", 1)[0]
    assert "The secret answer is Zeta-9" not in citation_input.split("CITED SOURCE:", 1)[1]
    assert correctness.value is True
    assert correctness.usage.input_tokens == 15
    assert correctness.usage.output_tokens == 4


def test_ollama_invalid_structured_output_fails_after_one_call() -> None:
    calls = 0

    def transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict:
        nonlocal calls
        del url, payload, timeout_s
        calls += 1
        return _ollama_response(
            '{"kind":"tool_call","tool_name":"web_search",'
            '"arguments":{"query":"find a fact"},'
            '"claims":[{"text":"unsupported","source_id":"missing"}],'
            '"decline_reason":null}'
        )

    client = OllamaClient(transport=transport, runtime_version="test")

    with pytest.raises(RuntimeError, match="invalid AgentDecision structured output"):
        client.decide(
            question="Find a fact",
            tool_descriptions="web_search(query)",
            history=[],
            steps=[],
        )

    assert calls == 1


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:11434",
        "http://192.168.1.20:11434",
        "http://localhost.example:11434",
        "http://user:password@localhost:11434",
        "http://localhost:11434/api",
    ],
)
def test_ollama_rejects_non_loopback_or_non_origin_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback|origin"):
        OllamaClient(base_url=base_url, transport=lambda *_: {}, runtime_version="test")


def test_ollama_accepts_ipv4_localhost_and_ipv6_loopback_origins() -> None:
    transport = lambda *_: {}  # noqa: E731

    assert OllamaClient(
        base_url="http://127.0.0.1:11434",
        transport=transport,
        runtime_version="test",
    ).base_url == "http://127.0.0.1:11434"
    assert OllamaClient(
        base_url="http://localhost:11434/",
        transport=transport,
        runtime_version="test",
    ).base_url == "http://localhost:11434"
    assert OllamaClient(
        base_url="http://[::1]:11434",
        transport=transport,
        runtime_version="test",
    ).base_url == "http://[::1]:11434"
