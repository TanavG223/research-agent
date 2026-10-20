from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from research_agent import api
from research_agent.config import Settings
from research_agent.schemas import AgentAnswer, Claim, VerificationDeletion


def _settings(project_root: Path, *, openai_api_key: str | None = None) -> Settings:
    return Settings(
        project_root=project_root,
        openai_api_key=openai_api_key,
        tavily_api_key=None,
        openai_model="test-model",
        judge_model="test-judge",
        max_steps=8,
        python_timeout_s=1.0,
        web_timeout_s=1.0,
        chroma_path=project_root / "chroma",
        run_dir=project_root / "runs",
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        tavily_cost_per_call=0.0,
        allowed_hosts=("127.0.0.1", "localhost", "testserver"),
        api_docs_enabled=False,
    )


def test_ask_without_an_openai_key_returns_a_safe_service_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected_build(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the live stack must not be built without credentials")

    monkeypatch.setattr(api, "build_live_agent", unexpected_build)
    app = api.create_app(_settings(tmp_path))

    with TestClient(app) as client:
        health = client.get("/api/health")
        response = client.post("/api/ask", json={"question": "What is grounded?"})

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "live_ready": False,
        "openai_configured": False,
        "tavily_configured": False,
        "max_steps": 8,
    }
    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Live mode is not configured. Set OPENAI_API_KEY "
            "(and TAVILY_API_KEY for web search)."
        )
    }
    assert "traceback" not in response.text.lower()
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert "script-src 'self'" in health.headers["content-security-policy"]


def test_results_endpoint_returns_both_result_documents(tmp_path: Path) -> None:
    baseline = {"run_id": "baseline-test", "metrics": {"accuracy_pct": 75.0}}
    experiment = {"run_id": "experiment-test", "metrics": {"accuracy_pct": 50.0}}
    (tmp_path / "results_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (tmp_path / "results_experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    app = api.create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/results")

    assert response.status_code == 200
    assert response.json() == {"baseline": baseline, "experiment": experiment}


def test_log_path_supports_a_run_directory_outside_the_project(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    external_log = tmp_path / "external-runs" / "run" / "question.jsonl"

    assert api._display_log_path(project_root / "runs" / "one.jsonl", project_root) == (
        "runs/one.jsonl"
    )
    assert api._display_log_path(external_log, project_root) == str(external_log.resolve())


def test_ask_hides_rejected_claim_text_and_supports_external_run_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    settings = _settings(project_root, openai_api_key="configured")
    settings = replace(settings, run_dir=tmp_path / "external-runs")

    class FakeAgent:
        def answer(self, question: str, *, logger) -> AgentAnswer:
            del logger
            rejected = Claim(text="The unverified secret is Zeta-9.", source_id="missing")
            return AgentAnswer(
                question=question,
                status="unverified",
                answer="I could not verify an answer.",
                claims=[],
                generated_claims=[rejected],
                deletions=[
                    VerificationDeletion(
                        claim=rejected,
                        reason="unsupported_by_source",
                        detail="The source does not support: The unverified secret is Zeta-9.",
                    )
                ],
                max_steps=8,
            )

    monkeypatch.setattr(api, "build_live_agent", lambda _: (FakeAgent(), object()))
    app = api.create_app(settings)

    with TestClient(app) as client:
        response = client.post("/api/ask", json={"question": "What is the secret?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "I could not verify an answer."
    assert payload["claims"] == []
    assert payload["claims_generated"] == 1
    assert payload["claims_deleted"] == 1
    assert payload["deletions"] == [
        {
            "reason": "unsupported_by_source",
            "detail": "The cited source did not support the generated claim.",
        }
    ]
    assert "Zeta-9" not in response.text
    assert payload["log_file"].startswith(str(settings.run_dir.resolve()))
