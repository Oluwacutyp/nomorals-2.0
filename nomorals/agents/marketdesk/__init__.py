"""The markets desk — BTC/stocks pattern lab, signal calls,
self-knowledge of its feeds.

Extends the price class (``integrations/market_data``, the
keyless-first OHLCV chain) into a standing desk: continuous
BTCUSD + US-major study with verdicts in writing, timed signal
calls with >= 2 minutes of lead off lab-confirmed hour structure,
and its data needs wired through Devon's shared source registry.
Analysis only — the desk studies, signals and proposes; it never
trades.
"""

from __future__ import annotations

from typing import Any

from .collectors import collect_all, collect_symbol
from .desk import MarketsDesk, render_collect, render_library, render_status
from .jobs import ensure_marketdesk_jobs
from .store import Candle, CandleStore, marketdesk_dir

__all__ = [
    "MarketsDesk", "Candle", "CandleStore", "marketdesk_dir",
    "collect_symbol", "collect_all",
    "ensure_marketdesk_jobs", "render_collect", "render_library",
    "render_status",
    "register",
]


def register(registry: Any) -> None:
    """Register the ``marketdesk`` agent tool (study / signals / status /
    library / collect / sources / provision)."""
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "marketdesk",
        description=(
            "Markets desk: action=study (collect BTCUSD + US-major bars, "
            "run the pattern lab, refresh the library) | signals "
            "(publish callable timed signal calls, >=2m lead, off "
            "lab-confirmed hour structure only) | status (bars, verdicts, "
            "signal calibration, sources, keyed feeds) | library "
            "(per-symbol study records with evidence) | collect (fetch "
            "daily + hourly bars per watched symbol) | sources (the "
            "desk's market sources on the shared registry + keyed-feed "
            "state)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — study|signals|status|library|collect|sources",
            "deliver": "bool — publish signal calls when action=signals",
        },
    )
    def marketdesk(*, action: str = "status",
                   deliver: bool = False) -> dict[str, Any]:
        act = (action or "status").strip().lower()
        desk = MarketsDesk(context)
        if act == "status":
            return desk.status()
        if act == "library":
            return {"library": desk.library()}
        if act == "collect":
            return desk.collect()
        if act == "study":
            return desk.study(deliver_signals=bool(deliver))
        if act == "signals":
            desk.analyze_all()
            return desk.run_signals(deliver=bool(deliver))
        if act == "sources":
            from .sources import keyed_feed_state, market_status
            return {"sources": market_status(desk.registry),
                    "keyed_feeds": keyed_feed_state()}
        return {"error": f"unknown action {act!r} — "
                         "study|signals|status|library|collect|sources"}
