from __future__ import annotations

from pathlib import Path

import pytest

from research_agent import cli, factory
from research_agent.config import Settings
from research_agent.executors import ProductionToolExecutor


def _settings(root: Path, *, tavily_api_key: str | None) -> Settings:
    return Settings(
        project_root=root,
        openai_api_key=None,
        tavily_api_key=tavily_api_key,
        openai_model="unused",
        judge_model="unused",
        max_steps=8,
        python_timeout_s=2,
        web_timeout_s=2,
        chroma_path=root / "chroma",
        run_dir=root / "runs",
        input_cost_per_million=0,
        output_cost_per_million=0,
        tavily_cost_per_call=0,
        allowed_hosts=("localhost",),
        api_docs_enabled=False,
    )


def test_local_live_cli_mode_builds_local_model_with_live_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (object(), object())
    seen: dict[str, object] = {}

    def fake_builder(
        settings: Settings,
        *,
        max_steps: int,
        planning_subgoals: bool,
    ):
        seen.update(
            settings=settings,
            max_steps=max_steps,
            planning_subgoals=planning_subgoals,
        )
        return expected

    monkeypatch.setattr(cli, "build_local_live_agent", fake_builder)
    settings = _settings(tmp_path, tavily_api_key="configured")

    built = cli._stack(
        mode="local-live",
        max_steps=8,
        settings=settings,
        items=[],
        fixtures_path=tmp_path / "unused.json",
    )

    assert built == expected
    assert seen == {
        "settings": settings,
        "max_steps": 8,
        "planning_subgoals": False,
    }
    assert cli._result_mode("local-live") == "local_live_tavily"
    parsed = cli.build_parser().parse_args(
        ["benchmark", "--mode", "local-live", "--output", "result.json"]
    )
    assert parsed.mode == "local-live"
    assert parsed.planning_subgoals is False

    parsed_planning = cli.build_parser().parse_args(
        ["benchmark", "--mode", "local-live", "--planning-subgoals"]
    )
    assert parsed_planning.planning_subgoals is True


def test_unknown_stack_mode_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown benchmark mode"):
        cli._stack(
            mode="typo",
            max_steps=8,
            settings=_settings(tmp_path, tavily_api_key=None),
            items=[],
            fixtures_path=tmp_path / "unused.json",
        )

    with pytest.raises(ValueError, match="supported only with --mode local-live"):
        cli._stack(
            mode="local",
            max_steps=8,
            settings=_settings(tmp_path, tavily_api_key=None),
            items=[],
            fixtures_path=tmp_path / "unused.json",
            planning_subgoals=True,
        )

    with pytest.raises(ValueError, match="requires --max-steps 8"):
        cli._stack(
            mode="local-live",
            max_steps=3,
            settings=_settings(tmp_path, tavily_api_key="configured"),
            items=[],
            fixtures_path=tmp_path / "unused.json",
            planning_subgoals=True,
        )


def test_local_live_factory_requires_tavily_without_requiring_openai(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        factory.build_local_live_agent(
            _settings(tmp_path, tavily_api_key=None),
            max_steps=8,
        )

    local_client = object()
    client_options: dict[str, object] = {}

    def fake_ollama(**kwargs):
        client_options.update(kwargs)
        return local_client

    monkeypatch.setattr(factory, "OllamaClient", fake_ollama)
    settings = _settings(tmp_path, tavily_api_key="configured")

    agent, judge = factory.build_local_live_agent(
        settings,
        max_steps=8,
        planning_subgoals=True,
    )

    assert agent.model is local_client
    assert judge is local_client
    assert isinstance(agent.tools, ProductionToolExecutor)
    assert agent.tools.settings is settings
    assert agent.max_steps == 8
    assert agent.planning_subgoals is True
    assert client_options["planning_subgoals"] is True
