"""Integration bridge: opt-in agentic mode for the owner-chat path.

Does NOT modify agent_loop.py or coremind. The existing rigid pipeline
stays the default. Call ``maybe_run_agentic`` from the chat path where
you want the ReAct loop as an alternative mode.

Enable via environment: ``NM_AGENTIC_MODE=1`` — or call ``run_agentic``
directly with the live LLM + registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .context import LoopMemory
from .loop import AgenticLoop, LoopResult, run_agentic
from .tools import ToolAdapter

_log = logging.getLogger(__name__)

__all__ = ["maybe_run_agentic", "agentic_mode_enabled"]


def agentic_mode_enabled() -> bool:
    """True when the owner has opted into agentic mode."""
    return os.environ.get("NM_AGENTIC_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def maybe_run_agentic(
    text: str,
    *,
    llm: Any,
    registry: Any,
    history: list[tuple[str, str]] | None = None,
    step_budget: int = 10,
    actor: str = "owner-loop",
    capabilities: Any = None,
) -> LoopResult | None:
    """Run the agentic loop if agentic mode is enabled, else return None.

    Returns None when disabled so the caller falls through to the normal
    rigid pipeline. When enabled, returns the LoopResult.
    """
    if not agentic_mode_enabled():
        return None
    if llm is None or registry is None:
        _log.warning("agentic mode enabled but llm or registry missing")
        return None
    try:
        return run_agentic(
            text,
            llm=llm,
            registry=registry,
            history=history,
            step_budget=step_budget,
            actor=actor,
            capabilities=capabilities,
        )
    except Exception as exc:  # noqa: BLE001 — agentic mode must never sink chat
        _log.warning("agentic loop failed: %s", exc)
        return None
