"""Error-system spine tools — plain-language access to health and budgets.

- system_health: "is everything healthy?" — one snapshot.
- error_budget_status: per-subsystem burn rates and remaining budget.
- incident_history: recent incidents, filterable by subsystem.
"""

from __future__ import annotations

from typing import Any


def _es(context: Any) -> Any:
    es = getattr(context, "error_system", None)
    if es is None:
        from ..core.error_system import get_error_system
        es = get_error_system()
    return es


def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "system_health",
        description=(
            "Check the health of every subsystem. ('is everything healthy?', "
            "'any errors lately?', 'system status'). Returns overall ok, "
            "active budget alerts, degraded subsystems, and top failures "
            "in the last 24h."
        ),
        capability=Capability.DB_READ,
    )
    def system_health() -> dict[str, Any]:
        es = _es(context)
        if es is None:
            return {"ok": False,
                    "error": "error system not initialized at boot"}
        return {"ok": True, **es.health()}

    @registry.register(
        "error_budget_status",
        description=(
            "Show error-budget burn for each subsystem. ('error budgets', "
            "'which service is failing most?', 'burn rates'). Real ratios "
            "from recorded heartbeats — not failure-only."
        ),
        capability=Capability.DB_READ,
    )
    def error_budget_status(subsystem: str = "") -> dict[str, Any]:
        es = _es(context)
        if es is None:
            return {"ok": False,
                    "error": "error system not initialized at boot"}
        if subsystem:
            b = es.budgets.budget_for(subsystem)
            return {"ok": True, "budget": b.status()}
        return {"ok": True, "budgets": es.budgets.status_all()}

    @registry.register(
        "incident_history",
        description=(
            "Recent error incidents. ('what broke recently?', 'show errors "
            "for telegram', 'incident history'). Optional subsystem filter "
            "and limit."
        ),
        capability=Capability.DB_READ,
    )
    def incident_history(subsystem: str = "", limit: int = 20) -> dict[str, Any]:
        es = _es(context)
        if es is None:
            return {"ok": False,
                    "error": "error system not initialized at boot"}
        limit = max(1, min(100, int(limit or 20)))
        try:
            rows = es.journal.recent_incidents(
                subsystem=subsystem or None, limit=limit)
        except AttributeError:
            rows = []
        out = []
        for r in rows:
            d = dict(r) if hasattr(r, "keys") else r
            out.append({
                "id": d.get("id"),
                "subsystem": d.get("subsystem"),
                "signature": d.get("signature"),
                "first_seen": d.get("first_seen"),
                "last_seen": d.get("last_seen"),
                "count": d.get("count"),
            })
        return {"ok": True, "incidents": out}
