"""Composition helpers for live and deterministic stacks."""

from __future__ import annotations

from pathlib import Path

from research_agent.agent import ResearchAgent
from research_agent.config import Settings
from research_agent.executors import (
    FixtureToolExecutor,
    ProductionToolExecutor,
    RecordedBenchmarkToolExecutor,
)
from research_agent.providers import (
    FixtureJudgeClient,
    OllamaClient,
    OpenAIClient,
    ScriptedModelClient,
)
from research_agent.schemas import BenchmarkItem
from research_agent.verifier import CitationVerifier


def build_live_agent(
    settings: Settings,
    *,
    max_steps: int | None = None,
) -> tuple[ResearchAgent, OpenAIClient]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for live agent requests")
    client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        judge_model=settings.judge_model,
        input_cost_per_million=settings.input_cost_per_million,
        output_cost_per_million=settings.output_cost_per_million,
    )
    agent = ResearchAgent(
        model=client,
        tools=ProductionToolExecutor(settings),
        verifier=CitationVerifier(client),
        max_steps=max_steps or settings.max_steps,
    )
    return agent, client


def build_offline_agent(
    *,
    items: list[BenchmarkItem],
    fixtures_path: Path,
    max_steps: int,
) -> tuple[ResearchAgent, FixtureJudgeClient]:
    model = ScriptedModelClient(benchmark=items, fixtures_path=fixtures_path)
    judge = FixtureJudgeClient(benchmark=items, fixtures_path=fixtures_path)
    agent = ResearchAgent(
        model=model,
        tools=FixtureToolExecutor(fixtures_path, items),
        verifier=CitationVerifier(judge),
        max_steps=max_steps,
    )
    return agent, judge


def build_local_benchmark_agent(
    settings: Settings,
    *,
    items: list[BenchmarkItem],
    fixtures_path: Path,
    max_steps: int,
) -> tuple[ResearchAgent, OllamaClient]:
    """Use real local inference against the fixed recorded-search benchmark corpus."""

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        judge_model=settings.ollama_judge_model,
        timeout_s=settings.ollama_timeout_s,
    )
    tools = RecordedBenchmarkToolExecutor(
        settings,
        corpus_path=fixtures_path,
        benchmark=items,
    )
    agent = ResearchAgent(
        model=client,
        tools=tools,
        verifier=CitationVerifier(client),
        max_steps=max_steps,
    )
    return agent, client


def build_local_live_agent(
    settings: Settings,
    *,
    max_steps: int | None = None,
    planning_subgoals: bool = False,
) -> tuple[ResearchAgent, OllamaClient]:
    """Use real local Ollama inference with live Tavily-backed web search."""

    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is required for local-live requests")
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        judge_model=settings.ollama_judge_model,
        timeout_s=settings.ollama_timeout_s,
        planning_subgoals=planning_subgoals,
    )
    agent = ResearchAgent(
        model=client,
        tools=ProductionToolExecutor(settings),
        verifier=CitationVerifier(client),
        max_steps=max_steps or settings.max_steps,
        planning_subgoals=planning_subgoals,
    )
    return agent, client
