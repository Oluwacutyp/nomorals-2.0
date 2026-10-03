"""Power mode: online-first capability detection.

The owner was right — "fully offline" thinking made the system dumb.  The
philosophy is now **online-first with offline fallback**:

- ``FULL``: a real model is live and answering.  Use the model for titles,
  chapters, routing disambiguation — everything.  Heuristics are not consulted.
- ``DEGRADED``: a model is configured but unhealthy, or the active provider is
  weak/unverified.  Try the model, but expect fallback.
- ``OFFLINE``: no model at all (mock/offline/test provider, or ``NM_OFFLINE``).
  Heuristics and templates are the only option — and they should be honest
  about it.

Every organ that has a model path AND a heuristic path must consult
:func:`power_state` and prefer the model whenever the state is not OFFLINE.
The state is cheap to compute and cached briefly on the context.
"""

from __future__ import annotations

import time
from typing import Any

#: mock/offline provider names — never "full power"
_MOCK_PROVIDERS = {"mock", "offline", "test", "none", ""}

#: cache TTL for the power probe (seconds)
_PROBE_TTL = 60.0


def power_state(context: Any) -> str:
    """Return ``"full"``, ``"degraded"``, or ``"offline"``.

    Cheap and never raises.  Result is cached on the context for
    ``_PROBE_TTL`` seconds so hot paths don't re-probe.
    """
    now = time.monotonic()
    try:
        cached = getattr(context, "_power_state_cache", None)
        if cached and (now - cached[1]) < _PROBE_TTL:
            return cached[0]
    except Exception:  # noqa: BLE001
        pass

    state = _probe(context)
    try:
        context._power_state_cache = (state, now)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    return state


def _probe(context: Any) -> str:
    # explicit offline flag always wins
    try:
        settings = getattr(context, "settings", None)
        if settings is not None and bool(getattr(settings, "offline", False)):
            return "offline"
    except Exception:  # noqa: BLE001
        pass

    router = getattr(context, "router", None)
    if router is None:
        return "offline"

    snapshot_fn = getattr(router, "stats_snapshot", None)
    if snapshot_fn is None:
        return "degraded"
    try:
        snap = snapshot_fn()
    except Exception:  # noqa: BLE001
        return "degraded"

    active = str(snap.get("active") or "").lower()
    if active in _MOCK_PROVIDERS:
        return "offline"

    chain = [str(p).lower() for p in (snap.get("chain") or [])]
    real_providers = [p for p in chain if p not in _MOCK_PROVIDERS]
    if not real_providers:
        return "offline"

    # a real provider is registered — but is it healthy?
    health = snap.get("health") or {}
    active_health = health.get(active) or {}
    try:
        failures = int(active_health.get("consecutive_failures", 0) or 0)
    except (TypeError, ValueError):
        failures = 0
    if failures >= 3:
        return "degraded"

    return "full"


def model_usable(context: Any) -> bool:
    """True when the model path should be attempted first.

    ``"full"`` always; ``"degraded"`` yes (with fallback expected);
    ``"offline"`` no.
    """
    return power_state(context) in ("full", "degraded")


def invalidate(context: Any) -> None:
    """Drop the cached power state (call after a provider switch)."""
    try:
        context._power_state_cache = None  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
