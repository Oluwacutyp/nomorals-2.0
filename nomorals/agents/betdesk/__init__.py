"""The betting desk — crash pattern lab, signal calls, self-provisioned data.

Extends the sports analyst (``agents/sports_bet.py``) into a standing
desk: continuous crash-game study with verdicts in writing, timed
signal calls with >= 2 minutes of lead off lab-confirmed time
structure, and its data needs wired through Devon's real
self-provisioning + shared source registry.  Analysis only — the
desk studies, signals and proposes; it never places a bet.
"""

from __future__ import annotations

from typing import Any

from .crash import CrashRound, CrashStore, betdesk_dir
from .desk import BettingDesk, render_library, render_status
from .jobs import ensure_betdesk_jobs

__all__ = [
    "BettingDesk", "CrashRound", "CrashStore", "betdesk_dir",
    "ensure_betdesk_jobs", "render_library", "render_status",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``betdesk`` agent tool (study / signals / status /
    library / collect / provision)."""
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "betdesk",
        description=(
            "Betting desk: action=study (collect crash rounds, run the "
            "pattern lab, refresh the library, scan sports fixtures) | "
            "signals (publish callable timed signal calls, >=2m lead, "
            "off lab-confirmed time structure only) | status (rounds, "
            "verdicts, signal calibration, sources, keys) | library "
            "(per-game study records with evidence) | collect (fetch "
            "providers + import inbox only) | provision (register the "
            "desk's API services with self-provisioning)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — study|signals|status|library|collect|provision",
            "deliver": "bool — publish signal calls when action=signals",
        },
    )
    def betdesk(*, action: str = "status",
                deliver: bool = False) -> dict[str, Any]:
        act = (action or "status").strip().lower()
        desk = BettingDesk(context)
        if act == "status":
            return desk.status()
        if act == "library":
            return {"library": desk.library()}
        if act == "collect":
            from .collectors import collect_all
            rep = collect_all(desk.store, registry=desk.registry)
            rep["imports"] = desk.import_inbox()
            return rep
        if act == "study":
            return desk.study(deliver_signals=bool(deliver))
        if act == "signals":
            desk.analyze_all()
            return desk.run_signals(deliver=bool(deliver))
        if act == "provision":
            from .sources import ensure_provisioning
            return {"services": ensure_provisioning()}
        return {"error": f"unknown action {act!r} — "
                         "study|signals|status|library|collect|provision"}
