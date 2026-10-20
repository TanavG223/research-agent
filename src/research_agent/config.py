"""Environment-backed configuration with conservative defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    openai_api_key: str | None
    tavily_api_key: str | None
    openai_model: str
    judge_model: str
    max_steps: int
    python_timeout_s: float
    web_timeout_s: float
    chroma_path: Path
    run_dir: Path
    input_cost_per_million: float
    output_cost_per_million: float
    tavily_cost_per_call: float
    allowed_hosts: tuple[str, ...]
    api_docs_enabled: bool
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"
    ollama_judge_model: str = "qwen3:4b"
    ollama_timeout_s: float = 120.0

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env", override=False)
        max_steps = int(os.getenv("MAX_STEPS", "8"))
        if not 1 <= max_steps <= 32:
            raise ValueError("MAX_STEPS must be between 1 and 32")
        return cls(
            project_root=root,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            judge_model=os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini")),
            max_steps=max_steps,
            python_timeout_s=_float_env("PYTHON_TIMEOUT_SECONDS", 5.0),
            web_timeout_s=_float_env("WEB_TIMEOUT_SECONDS", 15.0),
            chroma_path=Path(os.getenv("CHROMA_PATH", root / "data" / "chroma")).resolve(),
            run_dir=Path(os.getenv("RUN_DIR", root / "runs")).resolve(),
            input_cost_per_million=_float_env("OPENAI_INPUT_COST_PER_MILLION", 0.0),
            output_cost_per_million=_float_env("OPENAI_OUTPUT_COST_PER_MILLION", 0.0),
            tavily_cost_per_call=_float_env("TAVILY_COST_PER_CALL", 0.0),
            allowed_hosts=tuple(
                host.strip()
                for host in os.getenv(
                    "ALLOWED_HOSTS", "127.0.0.1,localhost,testserver"
                ).split(",")
                if host.strip()
            ),
            api_docs_enabled=_bool_env("ENABLE_API_DOCS", False),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            ollama_judge_model=os.getenv(
                "OLLAMA_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:4b")
            ),
            ollama_timeout_s=_float_env("OLLAMA_TIMEOUT_SECONDS", 120.0),
        )

    def with_max_steps(self, max_steps: int) -> Settings:
        if not 1 <= max_steps <= 32:
            raise ValueError("max_steps must be between 1 and 32")
        values = {field: getattr(self, field) for field in self.__dataclass_fields__}
        values["max_steps"] = max_steps
        return Settings(**values)
