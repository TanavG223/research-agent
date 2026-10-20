"""Full-suite benchmark execution, metrics, and comparison rendering."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_agent.agent import ResearchAgent
from research_agent.providers import JudgeClient
from research_agent.run_logging import JsonlRunLogger, validate_log_identifier
from research_agent.schemas import BenchmarkItem, Usage

EXPECTED_COUNTS = {
    "factual_lookup": 15,
    "multi_step": 15,
    "computational": 10,
    "unanswerable": 10,
}

BENCHMARK_MODES = frozenset(
    {"offline_fixture", "local_fixture_tools", "local_live_tavily", "live"}
)


def _validity_metadata(mode: str) -> dict[str, bool]:
    if mode not in BENCHMARK_MODES:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    live_search = mode in {"live", "local_live_tavily"}
    return {
        "valid_for_live_model_accuracy": live_search,
        "valid_for_model_accuracy": mode != "offline_fixture",
        "valid_for_end_to_end_search_accuracy": live_search,
    }


def prepare_run_directory(run_dir: Path, run_id: str) -> tuple[str, Path]:
    canonical_id = validate_log_identifier(run_id, label="run_id")
    path = run_dir / canonical_id
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"run directory already exists: {path}") from exc
    return canonical_id, path


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_tree_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((project_root / "src" / "research_agent").rglob("*.py"))
    paths.append(project_root / "pyproject.toml")
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "chromadb",
        "fastapi",
        "httpx",
        "openai",
        "psutil",
        "pydantic",
        "tavily-python",
        "uvicorn",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _run_manifest(
    *,
    items: list[BenchmarkItem],
    agent: ResearchAgent,
    judge: JudgeClient,
    tavily_cost_per_call: float,
) -> dict[str, Any]:
    model = agent.model
    tool_settings = getattr(agent.tools, "settings", None)
    fixture_path = (
        getattr(model, "fixtures_path", None)
        or getattr(agent.tools, "fixtures_path", None)
        or getattr(agent.tools, "corpus_path", None)
    )
    project_root = Path(__file__).resolve().parents[2]
    lock_path = project_root / "requirements.lock"
    return {
        "agent_model": getattr(model, "model", type(model).__name__),
        "judge_model": getattr(judge, "judge_model", type(judge).__name__),
        "tool_backend": type(agent.tools).__name__,
        "web_search_mode": getattr(agent.tools, "retrieval_mode", None),
        "web_search_provider": getattr(agent.tools, "web_search_provider", None),
        "planning_subgoals": bool(getattr(agent, "planning_subgoals", False)),
        "model_provider": getattr(
            model, "provider", getattr(model, "provider_name", type(model).__name__)
        ),
        "model_endpoint": getattr(model, "base_url", None),
        "model_runtime_version": getattr(model, "runtime_version", None),
        "model_options": getattr(
            model, "generation_options", getattr(model, "manifest_metadata", None)
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "pricing_assumptions": {
            "input_cost_per_million": getattr(model, "input_cost_per_million", 0.0),
            "output_cost_per_million": getattr(model, "output_cost_per_million", 0.0),
            "tavily_cost_per_call": tavily_cost_per_call,
        },
        "timeouts": {
            "python_seconds": getattr(tool_settings, "python_timeout_s", None),
            "web_seconds": getattr(tool_settings, "web_timeout_s", None),
        },
        "benchmark_sha256": _sha256_json(
            [item.model_dump(mode="json") for item in items]
        ),
        "fixtures_sha256": (
            hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest()
            if fixture_path is not None
            else None
        ),
        "tool_descriptions_sha256": hashlib.sha256(
            agent.tool_descriptions.encode("utf-8")
        ).hexdigest(),
        "source_tree_sha256": _source_tree_sha256(project_root),
        "dependency_lock_sha256": (
            hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
        ),
    }


def _tool_contract(item: BenchmarkItem, answer: Any) -> tuple[bool, str]:
    successful_steps = [
        step for step in answer.steps if step.action == "tool_call" and step.success
    ]
    successful_tools = [step.tool_name for step in successful_steps]
    if item.category == "factual_lookup":
        satisfied = successful_tools.count("web_search") >= 1
        return satisfied, "requires at least one successful web_search call"
    if item.category == "multi_step":
        distinct_queries = {
            str(step.arguments.get("query", "")).strip().casefold()
            for step in successful_steps
            if step.tool_name == "web_search" and str(step.arguments.get("query", "")).strip()
        }
        satisfied = len(distinct_queries) >= 2
        return satisfied, "requires at least two distinct successful web_search calls"
    if item.category == "computational":
        satisfied = successful_tools.count("run_python") >= 1
        return satisfied, "requires at least one successful run_python call"
    return True, "unanswerable items have no positive tool-use requirement"


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("benchmark.json must contain a JSON list")
    items = [BenchmarkItem.model_validate(item) for item in raw]
    ids = [item.id for item in items]
    for item_id in ids:
        validate_log_identifier(item_id, label="benchmark id")
    if len(items) != 50:
        raise ValueError(f"benchmark must contain 50 questions, found {len(items)}")
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark ids must be unique")
    counts = Counter(item.category for item in items)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"category counts must be {EXPECTED_COUNTS}, found {dict(counts)}")
    return items


def run_benchmark(
    *,
    items: list[BenchmarkItem],
    agent: ResearchAgent,
    judge: JudgeClient,
    run_dir: Path,
    run_id: str,
    mode: str,
    tavily_cost_per_call: float = 0.0,
) -> dict[str, Any]:
    run_id, run_path = prepare_run_directory(run_dir, run_id)
    results: list[dict[str, Any]] = []
    correctness_count = 0
    answerable_count = sum(item.answerable for item in items)
    hallucination_count = 0
    unanswerable_count = len(items) - answerable_count
    total_generated_claims = 0
    total_deleted_claims = 0
    step_counts: list[int] = []
    costs: list[float] = []
    latencies: list[float] = []

    for item in items:
        logger = JsonlRunLogger(run_dir, run_id, item.id)
        answer = agent.answer(item.question, logger=logger)
        evaluation: dict[str, Any]
        judge_usage = Usage()
        if item.answerable:
            tool_contract_satisfied, tool_contract = _tool_contract(item, answer)
            try:
                verdict = judge.correct(
                    question=item.question,
                    known_answer=item.known_correct_answer,
                    candidate_answer=answer.answer,
                )
                judge_usage = verdict.usage
                answer_correct = verdict.value
                is_correct = answer_correct and tool_contract_satisfied
                evaluation = {
                    "correct": is_correct,
                    "answer_correct": answer_correct,
                    "tool_contract_satisfied": tool_contract_satisfied,
                    "tool_contract": tool_contract,
                    "reason": (
                        verdict.reason
                        if tool_contract_satisfied
                        else f"tool contract failed: {tool_contract}"
                    ),
                    "judge_usage": judge_usage.model_dump(mode="json"),
                }
            except Exception as exc:
                is_correct = False
                evaluation = {
                    "correct": False,
                    "answer_correct": False,
                    "tool_contract_satisfied": tool_contract_satisfied,
                    "tool_contract": tool_contract,
                    "reason": f"judge_error: {type(exc).__name__}: {exc}",
                    "judge_usage": judge_usage.model_dump(mode="json"),
                }
            correctness_count += int(is_correct)
        else:
            confident_answer = answer.status == "answered" and bool(answer.claims)
            hallucination_count += int(confident_answer)
            evaluation = {
                "correct": not confident_answer,
                "confident_answer": confident_answer,
                "reason": (
                    "agent produced verified claims for an unanswerable item"
                    if confident_answer
                    else "agent declined or returned insufficient information"
                ),
                "judge_usage": judge_usage.model_dump(mode="json"),
            }

        web_calls = sum(step.tool_name == "web_search" for step in answer.steps)
        estimated_cost = (
            answer.usage.cost_usd + judge_usage.cost_usd + web_calls * tavily_cost_per_call
        )
        total_latency = answer.latency_ms + judge_usage.latency_ms
        total_generated_claims += len(answer.generated_claims)
        total_deleted_claims += len(answer.deletions)
        step_counts.append(len(answer.steps))
        costs.append(estimated_cost)
        latencies.append(total_latency)
        logger.log(
            "benchmark_evaluation",
            evaluation=evaluation,
            estimated_cost_usd=estimated_cost,
            total_latency_ms=total_latency,
        )
        results.append(
            {
                "id": item.id,
                "question": item.question,
                "known_correct_answer": item.known_correct_answer,
                "category": item.category,
                "answerable": item.answerable,
                "agent": answer.public_dump(),
                "evaluation": evaluation,
                "estimated_cost_usd": estimated_cost,
                "total_latency_ms": total_latency,
            }
        )

    catch_rate = (
        total_deleted_claims / total_generated_claims if total_generated_claims else 0.0
    )
    metrics = {
        "accuracy": correctness_count / answerable_count,
        "accuracy_pct": 100 * correctness_count / answerable_count,
        "accuracy_numerator": correctness_count,
        "accuracy_denominator": answerable_count,
        "hallucination_rate": hallucination_count / unanswerable_count,
        "hallucination_rate_pct": 100 * hallucination_count / unanswerable_count,
        "hallucination_numerator": hallucination_count,
        "hallucination_denominator": unanswerable_count,
        "verifier_catch_rate": catch_rate,
        "verifier_catch_rate_pct": 100 * catch_rate,
        "claims_deleted": total_deleted_claims,
        "claims_generated": total_generated_claims,
        "average_steps_per_question": statistics.fmean(step_counts),
        "average_cost_usd_per_question": statistics.fmean(costs),
        "average_latency_ms_per_question": statistics.fmean(latencies),
    }
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        **_validity_metadata(mode),
        "experiment_variable": (
            "planning_subgoals"
            if getattr(agent, "planning_subgoals", False)
            else "MAX_STEPS"
        ),
        "configuration": {
            "max_steps": agent.max_steps,
            "planning_subgoals": bool(getattr(agent, "planning_subgoals", False)),
            "citation_verifier": "enabled_fail_closed",
            "tool_descriptions": "verbose",
            "plan_revision": "allowed_each_turn",
            "manifest": _run_manifest(
                items=items,
                agent=agent,
                judge=judge,
                tavily_cost_per_call=tavily_cost_per_call,
            ),
        },
        "metrics": metrics,
        "results": results,
    }
    summary_path = run_path / "summary.json"
    summary_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_comparison(
    *,
    baseline: dict[str, Any],
    experiment: dict[str, Any],
    output_path: Path,
) -> None:
    baseline_config = baseline["configuration"]
    experiment_config = experiment["configuration"]
    changed = [
        key
        for key in sorted(set(baseline_config) | set(experiment_config))
        if baseline_config.get(key) != experiment_config.get(key)
    ]
    if changed != ["max_steps"]:
        raise ValueError(f"comparison must change exactly max_steps; changed fields: {changed}")
    if baseline_config["max_steps"] != 8 or experiment_config["max_steps"] != 3:
        raise ValueError("comparison requires baseline MAX_STEPS=8 and experiment MAX_STEPS=3")
    if baseline["mode"] != experiment["mode"]:
        raise ValueError("baseline and experiment modes must match")

    b = baseline["metrics"]
    e = experiment["metrics"]
    if baseline["mode"] == "live":
        run_note = "live OpenAI + Tavily"
    elif baseline["mode"] == "local_live_tavily":
        run_note = "local Ollama inference + live Tavily search"
    elif baseline["mode"] == "local_fixture_tools":
        run_note = "local Ollama inference + recorded web evidence"
    else:
        run_note = "offline fixtures"
    rows = [
        ("MAX_STEPS", str(baseline_config["max_steps"]), str(experiment_config["max_steps"])),
        ("Accuracy", f"{b['accuracy_pct']:.1f}%", f"{e['accuracy_pct']:.1f}%"),
        (
            "Hallucination rate",
            f"{b['hallucination_rate_pct']:.1f}%",
            f"{e['hallucination_rate_pct']:.1f}%",
        ),
        (
            "Verifier catch rate",
            f"{b['verifier_catch_rate_pct']:.1f}%",
            f"{e['verifier_catch_rate_pct']:.1f}%",
        ),
        (
            "Average steps / question",
            f"{b['average_steps_per_question']:.2f}",
            f"{e['average_steps_per_question']:.2f}",
        ),
        (
            "Average cost / question",
            f"${b['average_cost_usd_per_question']:.6f}",
            f"${e['average_cost_usd_per_question']:.6f}",
        ),
        (
            "Average latency / question",
            _format_latency(b["average_latency_ms_per_question"]),
            _format_latency(e["average_latency_ms_per_question"]),
        ),
    ]
    table = "\n".join(f"| {name} | {left} | {right} |" for name, left, right in rows)
    delta_accuracy = e["accuracy_pct"] - b["accuracy_pct"]
    if baseline["mode"] == "live":
        interpretation = (
            "These are live OpenAI + Tavily measurements for the recorded models, pricing "
            "assumptions, dependencies, and data snapshot. Web results and provider behavior are "
            "time-varying, so compare future runs using the embedded manifests and preserved logs."
        )
    elif baseline["mode"] == "local_live_tavily":
        interpretation = (
            "These measurements use real local Ollama inference for the research agent and LLM "
            "judges with live Tavily search. Search results are time-varying, while local "
            "inference has no provider-billed token cost; the reported model cost excludes "
            "electricity and hardware depreciation. Tavily cost follows the embedded pricing "
            "assumption."
        )
    elif baseline["mode"] == "local_fixture_tools":
        interpretation = (
            "These measurements use real local Ollama inference for both the research agent and "
            "LLM judges. Web evidence comes from the fixed recorded benchmark corpus and Python "
            "calculations execute in the real local sandbox, so this measures local-model behavior "
            "without measuring live search quality. Local inference has no provider-billed token "
            "cost; the reported $0 excludes electricity and hardware depreciation."
        )
    else:
        interpretation = (
            "Offline-fixture runs validate orchestration, logging, metric computation, and the "
            "controlled `MAX_STEPS` comparison; they are **not** measurements of live-model "
            "research accuracy. Run the same commands in `live` mode with API credentials before "
            "making model-quality or deployment claims."
        )

    text = f"""# Benchmark results

Run mode: **{run_note}**. Citation verification was fail-closed in both runs. The only changed
configuration field was `MAX_STEPS` (8 → 3).

| Metric | Baseline | Experiment |
|---|---:|---:|
{table}

Reducing the step budget changed accuracy by {delta_accuracy:+.1f} percentage points. Questions
that need more than two tool calls cannot both gather all evidence and emit a final answer within
three loop iterations, so the shorter run declines at the hard limit. The verifier, tool wording,
model/judge implementation, benchmark data, and plan-revision behavior were held constant.

## Interpretation boundary

{interpretation}
"""
    output_path.write_text(text, encoding="utf-8")


def _format_latency(latency_ms: float) -> str:
    if latency_ms >= 1_000:
        return f"{latency_ms / 1_000:.2f} s"
    return f"{latency_ms:.2f} ms"


def _failure_cause(result: dict[str, Any]) -> str:
    agent = result["agent"]
    evaluation = result["evaluation"]
    steps = agent.get("steps", [])
    if not result["answerable"] and evaluation.get("confident_answer"):
        return "hallucinated_answer"
    if str(evaluation.get("reason", "")).startswith("judge_error:"):
        return "correctness_judge_error"
    if any(step.get("action") == "model_error" for step in steps):
        return "model_or_schema_error"
    if any(
        step.get("action") == "tool_call" and not step.get("success", False)
        for step in steps
    ):
        return "tool_failure_or_recorded_query_miss"
    if agent.get("status") == "insufficient_information" and len(steps) >= agent.get(
        "max_steps", 0
    ):
        return "max_steps_before_final_answer"
    if evaluation.get("tool_contract_satisfied") is False:
        return "required_tool_chain_not_completed"
    if agent.get("status") == "unverified":
        return "citation_verifier_removed_all_claims"
    if agent.get("status") in {"declined", "insufficient_information"}:
        return "agent_declined"
    if evaluation.get("answer_correct") is False:
        return "incorrect_verified_answer"
    return "other_evaluation_failure"


def _markdown_cell(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_error_analysis(
    *,
    baseline: dict[str, Any],
    experiment: dict[str, Any],
    output_path: Path,
) -> None:
    """Write an auditable summary of every incorrect benchmark outcome."""

    sections: list[str] = [
        "# Error analysis",
        "",
        (
            "This report includes every item whose benchmark evaluation was incorrect. "
            "Answers shown here are outward, verifier-approved text only; rejected claim text "
            "remains confined to the JSONL audit logs."
        ),
    ]
    for label, payload in (("Baseline", baseline), ("Experiment", experiment)):
        failures = [item for item in payload["results"] if not item["evaluation"]["correct"]]
        causes = Counter(_failure_cause(item) for item in failures)
        sections.extend(
            [
                "",
                f"## {label}: MAX_STEPS={payload['configuration']['max_steps']}",
                "",
                f"Incorrect outcomes: **{len(failures)} / {len(payload['results'])}**.",
                "",
                "Cause counts: "
                + (
                    ", ".join(
                        f"`{cause}`={count}" for cause, count in sorted(causes.items())
                    )
                    or "none"
                ),
                "",
                (
                    "| ID | Category | Cause | Status | Known answer | Returned answer | "
                    "Trace | Judge reason |"
                ),
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for item in failures:
            agent = item["agent"]
            trace = " → ".join(
                (
                    f"{step.get('tool_name')}"
                    f"({'ok' if step.get('success') else 'failed'})"
                    if step.get("action") == "tool_call"
                    else str(step.get("action"))
                )
                for step in agent.get("steps", [])
            )
            log_path = f"runs/{payload['run_id']}/{item['id']}.jsonl"
            item_id = f"[{item['id']}]({log_path})"
            sections.append(
                "| "
                + " | ".join(
                    (
                        item_id,
                        _markdown_cell(item["category"]),
                        _failure_cause(item),
                        _markdown_cell(agent.get("status", "")),
                        _markdown_cell(item.get("known_correct_answer", "")),
                        _markdown_cell(agent.get("answer", "")),
                        _markdown_cell(trace),
                        _markdown_cell(item["evaluation"].get("reason", "")),
                    )
                )
                + " |"
            )
        if not failures:
            sections.append("| — | — | — | — | — | — | — | — |")

    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
