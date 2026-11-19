"""The engine's LLM dispatch — Devon's brain as judge.

REWORK of ``engine/agent.py`` from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma). Upstream shelled out to
headless ``claude -p`` with ``--permission-mode bypassPermissions``
— a paid Claude Code subscription, which Devon does not use and
does not carry. The port replaces the subprocess with Devon's
brain through an injected ``suggest`` callable ``(prompt) -> text``
— the exact pattern ``nomorals/triage/engine.py`` already uses:
with no model the code rung runs and model verdicts are recorded
as unavailable, never fabricated.

``bypassPermissions`` is dropped entirely: every agent action runs
under Devon's Capability gating, and the read-only default stays.
``extract_json`` (fenced-JSON-last) ports verbatim — it is the
verdict contract between the judge and the orchestrator.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from ...core.logging_setup import get_logger

__all__ = ["run_agent", "extract_json"]

_log = get_logger(__name__)


def run_agent(task: str, *,
              suggest: Callable[[str], str] | None = None,
              timeout: float = 30.0) -> dict[str, Any]:
    """Ask the judge (Devon's brain) to run one hunt/validate task.

    Returns ``{"result": str, "error": str | None, "duration_s": float}``
    — the same shape upstream's ``run_agent`` returned, minus the
    Claude-CLI fields (``cost_usd``, ``num_turns``), which do not
    exist for Devon's brain. ``error`` is ``"no-model"`` when no
    ``suggest`` callable was injected: the caller records the item
    as not actually tested (``blocked``), never as clean.
    """
    t0 = time.time()
    if suggest is None:
        return {"result": "", "error": "no-model",
                "duration_s": round(time.time() - t0, 1)}
    try:
        raw = suggest(task) or ""
    except Exception as exc:  # noqa: BLE001 - a dead model is "no model"
        _log.debug("bughunter agent call failed", exc_info=True)
        return {"result": "", "error": f"exec:{exc}",
                "duration_s": round(time.time() - t0, 1)}
    if "usage limit" in raw.lower() or "session limit" in raw.lower():
        return {"result": raw, "error": "rate-limited",
                "duration_s": round(time.time() - t0, 1)}
    return {"result": raw, "error": None,
            "duration_s": round(time.time() - t0, 1)}


def extract_json(text: str | None) -> Any:
    """Pull the last valid JSON array/object out of a judge reply."""
    if not text:
        return None
    blocks = re.findall(r"```(?:json\s*|\s*)(.*?)```", text, re.S)
    for block in reversed(blocks):
        try:
            value = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    for b in reversed(re.findall(r"(\[.*\]|\{.*\})", text, re.S)):
        try:
            return json.loads(b)
        except Exception:  # noqa: BLE001 - not JSON, keep looking
            pass
    return None


def agent_selftest() -> bool:
    """Upstream's offline self-test of the JSON extractor (no agent call)."""
    assert extract_json('blah ```json\n[{"a":1}]\n``` end') == [{"a": 1}]
    assert extract_json('text {"x": "y"} more') == {"x": "y"}
    assert extract_json("no json here") is None
    # Prefers fenced/last.
    assert extract_json(
        'first {"a":1} then ```json\n{"b":2}\n```') == {"b": 2}
    # No model is an honest error, never a fabricated verdict.
    assert run_agent("task")["error"] == "no-model"
    return True
