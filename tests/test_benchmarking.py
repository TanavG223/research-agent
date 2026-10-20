from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from research_agent.agent import ResearchAgent
from research_agent.benchmarking import (
    EXPECTED_COUNTS,
    _validity_metadata,
    load_benchmark,
    prepare_run_directory,
    render_comparison,
    render_error_analysis,
    run_benchmark,
)
from research_agent.schemas import (
    AgentDecision,
    BenchmarkItem,
    JudgeVerdict,
    ModelTurn,
)
from research_agent.verifier import CitationVerifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_has_exact_required_size_categories_and_answerability() -> None:
    items = load_benchmark(PROJECT_ROOT / "benchmark.json")

    assert len(items) == 50
    assert Counter(item.category for item in items) == EXPECTED_COUNTS
    assert len({item.id for item in items}) == 50
    assert all(item.known_correct_answer for item in items)
    assert sum(item.answerable for item in items) == 40
    assert sum(not item.answerable for item in items) == 10
    assert all((item.category == "unanswerable") is (not item.answerable) for item in items)


def test_benchmark_item_rejects_a_missing_known_answer() -> None:
    with pytest.raises(ValueError, match="known_correct_answer"):
        BenchmarkItem.model_validate(
            {
                "id": "invalid",
                "question": "What is missing?",
                "known_correct_answer": None,
                "category": "factual_lookup",
                "answerable": True,
            }
        )


def test_run_directories_reject_traversal_and_reuse(tmp_path: Path) -> None:
    run_id, run_path = prepare_run_directory(tmp_path, "baseline-20260818T120000Z")

    assert run_id == "baseline-20260818T120000Z"
    assert run_path == tmp_path / run_id
    with pytest.raises(ValueError, match="already exists"):
        prepare_run_directory(tmp_path, run_id)
    with pytest.raises(ValueError, match="run_id"):
        prepare_run_directory(tmp_path, "../../outside")


def test_fixture_workflows_match_each_benchmark_category_contract() -> None:
    items = load_benchmark(PROJECT_ROOT / "benchmark.json")
    fixtures = json.loads(
        (PROJECT_ROOT / "data" / "offline_fixtures.json").read_text(encoding="utf-8")
    )
    cases = {case["id"]: case for case in fixtures["cases"]}

    assert set(cases) == {item.id for item in items}
    for item in items:
        case = cases[item.id]
        actions = case["actions"]
        if item.category == "factual_lookup":
            assert [action["tool"] for action in actions] == ["web_search"]
        elif item.category == "multi_step":
            assert len(actions) >= 2
            assert all(action["tool"] == "web_search" for action in actions)
            assert len({action["input"]["query"].casefold() for action in actions}) >= 2
            assert item.question.startswith("Use distinct web lookups:")
        elif item.category == "computational":
            assert [action["tool"] for action in actions] == ["run_python"]
        else:
            assert item.category == "unanswerable"
            assert case["expected_final"].get("claims", []) == []


def test_fixture_fault_injections_exercise_verifier_deletions() -> None:
    fixtures = json.loads(
        (PROJECT_ROOT / "data" / "offline_fixtures.json").read_text(encoding="utf-8")
    )
    injected_cases = [case for case in fixtures["cases"] if "generated_claims" in case]

    assert len(injected_cases) >= 4
    for case in injected_cases:
        generated = case["generated_claims"]
        expected = case["expected_final"]["claims"]
        assert all(claim in generated for claim in expected)
        assert len(generated) > len(expected)


def _comparison_payload(max_steps: int) -> dict[str, Any]:
    return {
        "mode": "offline_fixture",
        "configuration": {
            "max_steps": max_steps,
            "citation_verifier": "enabled_fail_closed",
            "tool_descriptions": "verbose",
            "plan_revision": "allowed_each_turn",
        },
        "metrics": {
            "accuracy_pct": 90.0,
            "hallucination_rate_pct": 0.0,
            "verifier_catch_rate_pct": 25.0,
            "average_steps_per_question": 2.5,
            "average_cost_usd_per_question": 0.01,
            "average_latency_ms_per_question": 123.0,
        },
    }


def test_comparison_accepts_max_steps_as_the_only_changed_variable(tmp_path: Path) -> None:
    output_path = tmp_path / "RESULTS.md"

    render_comparison(
        baseline=_comparison_payload(8),
        experiment=_comparison_payload(3),
        output_path=output_path,
    )

    rendered = output_path.read_text(encoding="utf-8")
    assert "| MAX_STEPS | 8 | 3 |" in rendered
    assert "The only changed\nconfiguration field was `MAX_STEPS` (8 → 3)." in rendered


def test_comparison_rejects_the_wrong_max_steps_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline MAX_STEPS=8 and experiment MAX_STEPS=3"):
        render_comparison(
            baseline=_comparison_payload(9),
            experiment=_comparison_payload(4),
            output_path=tmp_path / "RESULTS.md",
        )


def test_comparison_renders_live_interpretation_and_rejects_mixed_modes(
    tmp_path: Path,
) -> None:
    baseline = _comparison_payload(8)
    experiment = _comparison_payload(3)
    baseline["mode"] = experiment["mode"] = "live"
    output_path = tmp_path / "RESULTS.md"

    render_comparison(baseline=baseline, experiment=experiment, output_path=output_path)
    rendered = output_path.read_text(encoding="utf-8")
    assert "These are live OpenAI + Tavily measurements" in rendered
    assert "**not** measurements of live-model" not in rendered

    experiment["mode"] = "offline_fixture"
    with pytest.raises(ValueError, match="modes must match"):
        render_comparison(
            baseline=baseline,
            experiment=experiment,
            output_path=output_path,
        )


def test_comparison_renders_local_inference_boundary(tmp_path: Path) -> None:
    baseline = _comparison_payload(8)
    experiment = _comparison_payload(3)
    baseline["mode"] = experiment["mode"] = "local_fixture_tools"
    baseline["metrics"]["average_latency_ms_per_question"] = 1_250.0
    experiment["metrics"]["average_latency_ms_per_question"] = 1_100.0
    output_path = tmp_path / "RESULTS.md"

    render_comparison(baseline=baseline, experiment=experiment, output_path=output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "local Ollama inference + recorded web evidence" in rendered
    assert "1.25 s" in rendered
    assert "no provider-billed token cost" in rendered


def test_comparison_renders_local_live_tavily_boundary(tmp_path: Path) -> None:
    baseline = _comparison_payload(8)
    experiment = _comparison_payload(3)
    baseline["mode"] = experiment["mode"] = "local_live_tavily"
    output_path = tmp_path / "RESULTS.md"

    render_comparison(baseline=baseline, experiment=experiment, output_path=output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "local Ollama inference + live Tavily search" in rendered
    assert "Search results are time-varying" in rendered


def test_local_live_tavily_validity_metadata_is_live_and_model_backed() -> None:
    assert _validity_metadata("local_live_tavily") == {
        "valid_for_live_model_accuracy": True,
        "valid_for_model_accuracy": True,
        "valid_for_end_to_end_search_accuracy": True,
    }

    with pytest.raises(ValueError, match="unsupported benchmark mode"):
        _validity_metadata("ambiguous-live-mode")


class _SyntheticDecliningModel:
    model = "synthetic-local-model"

    def decide(self, **kwargs) -> ModelTurn:
        del kwargs
        return ModelTurn(
            decision=AgentDecision(kind="final", decline_reason="Insufficient evidence.")
        )


class _SyntheticJudge:
    judge_model = "synthetic-local-judge"

    def supports(self, **kwargs) -> JudgeVerdict:
        del kwargs
        return JudgeVerdict(value=False, reason="unused")

    def correct(self, **kwargs) -> JudgeVerdict:
        del kwargs
        return JudgeVerdict(value=False, reason="synthetic mismatch")


class _SyntheticTools:
    def execute(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("declining model must not execute tools")


def test_planning_experiment_is_explicit_in_result_configuration_and_manifest(
    tmp_path: Path,
) -> None:
    judge = _SyntheticJudge()
    agent = ResearchAgent(
        model=_SyntheticDecliningModel(),
        tools=_SyntheticTools(),  # type: ignore[arg-type]
        verifier=CitationVerifier(judge),  # type: ignore[arg-type]
        max_steps=8,
        planning_subgoals=True,
    )
    items = [
        BenchmarkItem(
            id="answerable",
            question="What is the answer?",
            known_correct_answer="42",
            category="factual_lookup",
            answerable=True,
        ),
        BenchmarkItem(
            id="unanswerable",
            question="What cannot be known?",
            known_correct_answer="No public answer exists",
            category="unanswerable",
            answerable=False,
        ),
    ]

    payload = run_benchmark(
        items=items,
        agent=agent,
        judge=judge,  # type: ignore[arg-type]
        run_dir=tmp_path / "runs",
        run_id="planning-metadata-test",
        mode="local_live_tavily",
    )

    assert payload["experiment_variable"] == "planning_subgoals"
    assert payload["configuration"]["max_steps"] == 8
    assert payload["configuration"]["planning_subgoals"] is True
    assert payload["configuration"]["manifest"]["planning_subgoals"] is True


def test_error_analysis_lists_every_failed_item_without_rejected_text(
    tmp_path: Path,
) -> None:
    baseline = _comparison_payload(8)
    experiment = _comparison_payload(3)
    failed_item = {
        "id": "multi_step_001",
        "question": "Question?",
        "known_correct_answer": "Boron",
        "category": "multi_step",
        "answerable": True,
        "agent": {
            "status": "insufficient_information",
            "answer": "insufficient information",
            "max_steps": 3,
            "steps": [
                {"action": "tool_call", "tool_name": "web_search", "success": True},
                {"action": "tool_call", "tool_name": "web_search", "success": True},
                {"action": "tool_call", "tool_name": "web_search", "success": True},
            ],
        },
        "evaluation": {
            "correct": False,
            "answer_correct": False,
            "tool_contract_satisfied": True,
            "reason": "answer mismatch",
        },
    }
    baseline["run_id"] = "baseline-local"
    experiment["run_id"] = "experiment-local"
    baseline["results"] = []
    experiment["results"] = [failed_item]
    output_path = tmp_path / "ERROR_ANALYSIS.md"

    render_error_analysis(baseline=baseline, experiment=experiment, output_path=output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "Incorrect outcomes: **1 / 1**" in rendered
    assert "max_steps_before_final_answer" in rendered
    assert "[multi_step_001](runs/experiment-local/multi_step_001.jsonl)" in rendered
    assert "unverified secret zeta-9" not in rendered.lower()


@pytest.mark.parametrize("change_extra_field", [False, True])
def test_comparison_rejects_zero_or_multiple_variable_changes(
    tmp_path: Path,
    change_extra_field: bool,
) -> None:
    baseline = _comparison_payload(8)
    experiment = deepcopy(baseline)
    if change_extra_field:
        experiment["configuration"]["max_steps"] = 3
        experiment["configuration"]["citation_verifier"] = "disabled"

    with pytest.raises(ValueError, match="comparison must change exactly max_steps"):
        render_comparison(
            baseline=baseline,
            experiment=experiment,
            output_path=tmp_path / "RESULTS.md",
        )


def test_shipped_results_cover_the_full_benchmark_and_change_only_max_steps() -> None:
    benchmark_ids = {item.id for item in load_benchmark(PROJECT_ROOT / "benchmark.json")}
    baseline = json.loads((PROJECT_ROOT / "results_baseline.json").read_text(encoding="utf-8"))
    experiment = json.loads(
        (PROJECT_ROOT / "results_experiment.json").read_text(encoding="utf-8")
    )

    assert len(baseline["results"]) == len(experiment["results"]) == 50
    assert {result["id"] for result in baseline["results"]} == benchmark_ids
    assert {result["id"] for result in experiment["results"]} == benchmark_ids
    changed = {
        key
        for key in baseline["configuration"] | experiment["configuration"]
        if baseline["configuration"].get(key) != experiment["configuration"].get(key)
    }
    assert changed == {"max_steps"}
    assert baseline["configuration"]["max_steps"] == 8
    assert experiment["configuration"]["max_steps"] == 3
    for result in (baseline, experiment):
        assert result["valid_for_live_model_accuracy"] is (result["mode"] == "live")
        assert result["valid_for_model_accuracy"] is (
            result["mode"] in {"live", "local_fixture_tools"}
        )
        assert result["valid_for_end_to_end_search_accuracy"] is (
            result["mode"] == "live"
        )
        assert result["metrics"]["accuracy_denominator"] == 40
        assert result["metrics"]["hallucination_denominator"] == 10
        assert result["metrics"]["claims_deleted"] >= 0
        assert result["metrics"]["verifier_catch_rate_pct"] >= 0
        manifest = result["configuration"]["manifest"]
        assert manifest["agent_model"]
        assert manifest["judge_model"]
        assert len(manifest["benchmark_sha256"]) == 64
        assert len(manifest["fixtures_sha256"]) == 64
        assert len(manifest["tool_descriptions_sha256"]) == 64
        assert len(manifest["source_tree_sha256"]) == 64
        assert len(manifest["dependency_lock_sha256"]) == 64
        for question_result in result["results"]:
            outward = question_result["agent"]
            assert "generated_claims" not in outward
            assert all("claim" not in deletion for deletion in outward["deletions"])
            if question_result["answerable"] and question_result["evaluation"]["correct"]:
                assert question_result["evaluation"]["tool_contract_satisfied"] is True


def test_shipped_runs_have_a_complete_step_and_evaluation_log_per_question() -> None:
    for result_name in ("results_baseline.json", "results_experiment.json"):
        result = json.loads((PROJECT_ROOT / result_name).read_text(encoding="utf-8"))
        run_path = PROJECT_ROOT / "runs" / result["run_id"]
        assert (run_path / "summary.json").exists()

        log_files = list(run_path.glob("*.jsonl"))
        assert len(log_files) == 50
        for question_result in result["results"]:
            records = [
                json.loads(line)
                for line in (run_path / f"{question_result['id']}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            events = [record["event"] for record in records]
            steps = question_result["agent"]["steps"]
            assert events[0] == "question_started"
            assert events.count("model_decision") == len(steps)
            assert events.count("tool_result") == sum(
                step["action"] == "tool_call" for step in steps
            )
            assert events.count("question_completed") == 1
            assert events[-1] == "benchmark_evaluation"
