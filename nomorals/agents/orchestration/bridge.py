"""Integration bridge: opt-in agentic mode for the owner-chat path.

Does NOT modify agent_loop.py or coremind. The existing rigid pipeline
stays the default. Call ``maybe_run_agentic`` from the chat path where
you want the ReAct loop as an alternative mode.

STANDING RULE — exact commands first: ``maybe_run_agentic`` must only be
called AFTER deterministic slash-command parsing. Exact ``/commands``
always parse first, before any model call. This bridge never sees them.

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

__all__ = ["maybe_run_agentic", "agentic_mode_enabled", "is_model_available"]


def agentic_mode_enabled() -> bool:
    """True when the owner has opted into agentic mode."""
    return os.environ.get("NM_AGENTIC_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def is_model_available(llm: Any) -> bool:
    """True when there is a usable model behind ``llm``.

    Cheap and non-invasive: no probe calls. Returns False when llm is
    None, or when the provider explicitly reports itself unavailable
    (e.g. an ``available`` / ``healthy`` attribute set to False, or a
    provider with no configured credentials). Anything else is assumed
    usable — actual call failures are handled gracefully by the loop.
    """
    if llm is None:
        return False
    for attr in ("available", "healthy", "is_available", "is_healthy"):
        val = getattr(llm, attr, None)
        if val is not None:
            # Attribute exists — respect it (callable or plain bool)
            try:
                return bool(val() if callable(val) else val)
            except Exception:  # noqa: BLE001 — a broken flag means unavailable
                return False
    # Some providers expose credential presence
    for attr in ("has_credentials", "credentials_configured", "api_key"):
        val = getattr(llm, attr, None)
        if val is not None:
            try:
                resolved = val() if callable(val) else val
                if not resolved:
                    return False
            except Exception:  # noqa: BLE001
                return False
    return True


def maybe_run_agentic(
    text: str,
    *,
    llm: Any,
    registry: Any,
    history: list[tuple[str, str]] | None = None,
    step_budget: int = 10,
    actor: str = "owner-loop",
    capabilities: Any = None,
    resume_from: dict[str, Any] | None = None,
    db: Any = None,
) -> LoopResult | None:
    """Run the agentic loop if agentic mode is enabled, else return None.

    Returns None when disabled OR when no model is available, so the
    caller falls through to the normal rigid pipeline. When enabled and
    a model is present, returns the LoopResult.

    Callers must parse exact ``/commands`` deterministically BEFORE
    calling this — the agentic loop never sees slash commands.

    ``db``: skill database for the Hermes distillation hook. When None
    the hook falls back to the app's default storage path.

    ``resume_from``: a ``LoopResult.memory_snapshot`` from a previous run
    that paused with ``ask`` — continues the plan on the user's answer.
    Persist ``result.memory_snapshot`` whenever ``result.asked_user`` is
    True and hand it back here on the next message.
    """
    if not agentic_mode_enabled():
        return None
    if not is_model_available(llm) or registry is None:
        _log.info("agentic mode enabled but no usable model or registry — "
                  "falling through to rigid pipeline")
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
            resume_from=resume_from,
            db=db,
        )
    except Exception as exc:  # noqa: BLE001 — agentic mode must never sink chat
        _log.warning("agentic loop failed: %s", exc)
        return None
