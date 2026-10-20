#!/usr/bin/env python3
"""Targeted Playwright smoke test for the local FastAPI frontend."""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import Route, sync_playwright

BASE_URL = os.getenv("RESEARCH_AGENT_TEST_URL", "http://127.0.0.1:8765/app/")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
CHROME_PATH = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def mock_answer(route: Route) -> None:
    route.fulfill(
        status=200,
        content_type="application/json",
        json={
            "answer": "The verified UI fixture answer is 42.",
            "claims": [
                {
                    "text": "The verified UI fixture answer is 42.",
                    "source_id": "ui-src-1",
                }
            ],
            "claims_generated": 2,
            "claims_deleted": 1,
            "steps": [
                {
                    "step": 1,
                    "action": "tool_call",
                    "tool_name": "run_python",
                    "arguments": {"code": "print(42)"},
                    "source_id": "ui-src-1",
                    "success": True,
                    "latency_ms": 2.0,
                },
                {"step": 2, "action": "final", "success": True, "latency_ms": 1.0},
            ],
            "deletions": [
                {
                    "reason": "unknown_source_id",
                    "detail": "not in history",
                }
            ],
            "latency_ms": 3.0,
            "usage": {"cost_usd": 0.0},
            "status": "answered",
        },
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    with sync_playwright() as playwright:
        launch_args = {"headless": True}
        if CHROME_PATH.exists():
            launch_args["executable_path"] = str(CHROME_PATH)
        browser = playwright.chromium.launch(**launch_args)

        desktop = browser.new_page(viewport={"width": 1440, "height": 1000})
        desktop.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        desktop.route("**/api/ask", mock_answer)
        response = desktop.goto(BASE_URL, wait_until="networkidle", timeout=15_000)
        assert response and response.ok
        assert desktop.get_by_role("heading", name="Answers with an audit trail.").is_visible()
        assert desktop.get_by_text("Setup needed").is_visible()
        accuracy_values = desktop.get_by_text("65.0%", exact=True)
        assert accuracy_values.count() >= 2
        assert accuracy_values.first.is_visible()

        desktop.get_by_label("Research question").fill("What is the fixture answer?")
        desktop.get_by_role("button", name="Run research").click()
        desktop.get_by_text("The verified UI fixture answer is 42.").wait_for(timeout=5_000)
        assert desktop.get_by_text("ui-src-1").first.is_visible()
        assert desktop.get_by_text("Verifier audit · 1 claim removed").is_visible()
        desktop.screenshot(path=ARTIFACT_DIR / "ui-desktop.png", full_page=True)

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(BASE_URL, wait_until="networkidle", timeout=15_000)
        assert mobile.get_by_role("heading", name="Answers with an audit trail.").is_visible()
        mobile.screenshot(path=ARTIFACT_DIR / "ui-mobile.png", full_page=True)

        browser.close()

    if console_errors:
        raise AssertionError(f"browser console errors: {console_errors}")
    print(f"UI smoke passed; screenshots written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
