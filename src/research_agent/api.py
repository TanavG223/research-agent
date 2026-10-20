"""FastAPI service for verified research answers and benchmark results."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import Response

from research_agent.config import Settings
from research_agent.factory import build_live_agent
from research_agent.run_logging import JsonlRunLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=10_000)


def _display_log_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path.resolve())


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env(PROJECT_ROOT)
    docs_url = "/docs" if configured.api_docs_enabled else None
    app = FastAPI(
        title="Grounded Research Agent",
        version="0.1.0",
        description="Bounded tool use with fail-closed per-claim citation verification.",
        docs_url=docs_url,
        redoc_url="/redoc" if configured.api_docs_enabled else None,
        openapi_url="/openapi.json" if configured.api_docs_enabled else None,
    )
    app.state.settings = configured
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(configured.allowed_hosts))

    @app.middleware("http")
    async def security_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        return response

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "live_ready": bool(configured.openai_api_key and configured.tavily_api_key),
            "openai_configured": bool(configured.openai_api_key),
            "tavily_configured": bool(configured.tavily_api_key),
            "max_steps": configured.max_steps,
        }

    @app.post("/api/ask")
    async def ask(request: AskRequest) -> dict:
        if not configured.openai_api_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Live mode is not configured. Set OPENAI_API_KEY "
                    "(and TAVILY_API_KEY for web search)."
                ),
            )
        try:
            agent, _ = build_live_agent(configured)
            run_id = f"api-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
            question_id = f"question-{uuid4().hex}"
            logger = JsonlRunLogger(configured.run_dir, run_id, question_id)
            result = await asyncio.to_thread(agent.answer, request.question, logger=logger)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Agent request failed safely: {type(exc).__name__}",
            ) from exc
        payload = result.public_dump()
        payload["cost_usd"] = result.usage.cost_usd
        payload["run_id"] = run_id
        payload["log_file"] = _display_log_path(logger.path, configured.project_root)
        return payload

    @app.get("/api/results")
    def results() -> dict:
        def read(name: str) -> dict | None:
            path = configured.project_root / name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        return {
            "baseline": read("results_baseline.json"),
            "experiment": read("results_experiment.json"),
        }

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app/")

    static_path = PROJECT_ROOT / "src" / "research_agent" / "static"
    if static_path.exists():
        app.mount("/app", StaticFiles(directory=static_path, html=True), name="frontend")
    return app


app = create_app()
