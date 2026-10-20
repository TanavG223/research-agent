"""Command line interface for questions, ingestion, and reproducible benchmark runs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from research_agent.benchmarking import (
    load_benchmark,
    render_comparison,
    render_error_analysis,
    run_benchmark,
    write_result,
)
from research_agent.config import Settings
from research_agent.factory import (
    build_live_agent,
    build_local_benchmark_agent,
    build_local_live_agent,
    build_offline_agent,
)
from research_agent.run_logging import JsonlRunLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _stack(
    *,
    mode: str,
    max_steps: int,
    settings: Settings,
    items,
    fixtures_path: Path,
    planning_subgoals: bool = False,
):
    if planning_subgoals and mode != "local-live":
        raise ValueError("--planning-subgoals is supported only with --mode local-live")
    if planning_subgoals and max_steps != 8:
        raise ValueError("--planning-subgoals experiment requires --max-steps 8")
    if mode == "offline":
        return build_offline_agent(
            items=items,
            fixtures_path=fixtures_path,
            max_steps=max_steps,
        )
    if mode == "local":
        return build_local_benchmark_agent(
            settings,
            items=items,
            fixtures_path=fixtures_path,
            max_steps=max_steps,
        )
    if mode == "local-live":
        return build_local_live_agent(
            settings,
            max_steps=max_steps,
            planning_subgoals=planning_subgoals,
        )
    if mode == "live":
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is required for a live full benchmark")
        return build_live_agent(settings, max_steps=max_steps)
    raise ValueError(f"unknown benchmark mode: {mode}")


def _result_mode(cli_mode: str) -> str:
    return {
        "offline": "offline_fixture",
        "local": "local_fixture_tools",
        "local-live": "local_live_tavily",
        "live": "live",
    }[cli_mode]


def command_benchmark(args: argparse.Namespace) -> int:
    settings = Settings.from_env(PROJECT_ROOT)
    benchmark_path = Path(args.benchmark).resolve()
    fixtures_path = Path(args.fixtures).resolve()
    items = load_benchmark(benchmark_path)
    agent, judge = _stack(
        mode=args.mode,
        max_steps=args.max_steps,
        settings=settings,
        items=items,
        fixtures_path=fixtures_path,
        planning_subgoals=args.planning_subgoals,
    )
    run_id = args.run_id or f"{args.mode}-steps{args.max_steps}-{_timestamp()}"
    payload = run_benchmark(
        items=items,
        agent=agent,
        judge=judge,
        run_dir=settings.run_dir,
        run_id=run_id,
        mode=_result_mode(args.mode),
        tavily_cost_per_call=settings.tavily_cost_per_call,
    )
    output = Path(args.output).resolve()
    write_result(output, payload)
    print(json.dumps({"output": str(output), "metrics": payload["metrics"]}, indent=2))
    return 0


def command_experiment(args: argparse.Namespace) -> int:
    settings = Settings.from_env(PROJECT_ROOT)
    items = load_benchmark(Path(args.benchmark).resolve())
    fixtures_path = Path(args.fixtures).resolve()
    stamp = _timestamp()
    runs = []
    for label, max_steps, output_name in (
        ("baseline", 8, "results_baseline.json"),
        ("experiment", 3, "results_experiment.json"),
    ):
        agent, judge = _stack(
            mode=args.mode,
            max_steps=max_steps,
            settings=settings,
            items=items,
            fixtures_path=fixtures_path,
        )
        payload = run_benchmark(
            items=items,
            agent=agent,
            judge=judge,
            run_dir=settings.run_dir,
            run_id=f"{label}-{args.mode}-{stamp}",
            mode=_result_mode(args.mode),
            tavily_cost_per_call=settings.tavily_cost_per_call,
        )
        write_result(PROJECT_ROOT / output_name, payload)
        runs.append(payload)
    render_comparison(
        baseline=runs[0],
        experiment=runs[1],
        output_path=PROJECT_ROOT / "RESULTS.md",
    )
    render_error_analysis(
        baseline=runs[0],
        experiment=runs[1],
        output_path=PROJECT_ROOT / "ERROR_ANALYSIS.md",
    )
    print(
        json.dumps(
            {
                "baseline": runs[0]["metrics"],
                "experiment": runs[1]["metrics"],
                "comparison": str(PROJECT_ROOT / "RESULTS.md"),
                "error_analysis": str(PROJECT_ROOT / "ERROR_ANALYSIS.md"),
            },
            indent=2,
        )
    )
    return 0


def command_ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env(PROJECT_ROOT)
    agent, _ = build_live_agent(settings)
    logger = JsonlRunLogger(settings.run_dir, f"cli-{_timestamp()}", "question")
    result = agent.answer(args.question, logger=logger)
    print(json.dumps(result.public_dump(), indent=2, ensure_ascii=False))
    return 0


def command_ingest(args: argparse.Namespace) -> int:
    settings = Settings.from_env(PROJECT_ROOT)
    from research_agent.tools import add_documents

    documents: list[str] = []
    sources: list[str] = []
    for raw_path in args.paths:
        path = Path(raw_path).resolve()
        text = path.read_text(encoding="utf-8")
        chunk_size = args.chunk_size
        for start in range(0, len(text), chunk_size):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                documents.append(chunk)
                sources.append(f"{path}#chars={start}-{start + len(chunk)}")
    result = add_documents(
        documents=documents,
        sources=sources,
        persist_directory=settings.chroma_path,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.success else 1


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "research_agent.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="ask one live, verified question")
    ask.add_argument("question")
    ask.set_defaults(func=command_ask)

    benchmark = subparsers.add_parser("benchmark", help="run one 50-question suite")
    benchmark.add_argument(
        "--mode",
        choices=("offline", "local", "local-live", "live"),
        default="offline",
    )
    benchmark.add_argument("--max-steps", type=int, default=8)
    benchmark.add_argument(
        "--planning-subgoals",
        action="store_true",
        help="require one planning sub-goal before each tool call (local-live only)",
    )
    benchmark.add_argument("--benchmark", default=str(PROJECT_ROOT / "benchmark.json"))
    benchmark.add_argument(
        "--fixtures", default=str(PROJECT_ROOT / "data" / "offline_fixtures.json")
    )
    benchmark.add_argument("--output", default=str(PROJECT_ROOT / "results.json"))
    benchmark.add_argument("--run-id")
    benchmark.set_defaults(func=command_benchmark)

    experiment = subparsers.add_parser(
        "experiment", help="run MAX_STEPS=8 baseline and MAX_STEPS=3 experiment"
    )
    experiment.add_argument(
        "--mode",
        choices=("offline", "local", "local-live", "live"),
        default="offline",
    )
    experiment.add_argument("--benchmark", default=str(PROJECT_ROOT / "benchmark.json"))
    experiment.add_argument(
        "--fixtures", default=str(PROJECT_ROOT / "data" / "offline_fixtures.json")
    )
    experiment.set_defaults(func=command_experiment)

    ingest = subparsers.add_parser("ingest", help="chunk text/Markdown files into ChromaDB")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--chunk-size", type=int, default=1200)
    ingest.set_defaults(func=command_ingest)

    serve = subparsers.add_parser("serve", help="start the FastAPI web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=command_serve)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
