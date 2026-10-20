"""Append-only JSONL logging used by API and benchmark runs."""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def validate_log_identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must be 1-128 characters, start with an alphanumeric character, "
            "and contain only letters, numbers, '.', '_', or '-'"
        )
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


class JsonlRunLogger:
    """Thread-safe, append-only event logger with one file per question."""

    def __init__(self, run_dir: Path, run_id: str, question_id: str) -> None:
        safe_run = validate_log_identifier(run_id, label="run_id")
        safe_question = validate_log_identifier(question_id, label="question_id")
        self.path = run_dir / safe_run / f"{safe_question}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        line = json.dumps(record, default=_json_default, sort_keys=True, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
