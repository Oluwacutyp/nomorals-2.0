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

import re
from typing import Any

from .collectors import collect_all, collect_symbol
from .desk import MarketsDesk, render_collect, render_library, render_status
from .jobs import ensure_marketdesk_jobs
from .patterns import (
    BUILTIN_ID,
    Pattern,
    PatternStore,
    backtest,
    check_watches,
    list_patterns,
    parse_rule,
    pattern_status,
    remove_pattern,
    render_check,
    render_pattern_list,
    render_study,
    set_watching,
    study_rule,
)
from .patterns import (
    render_status as render_pattern_status,
)
from .store import Candle, CandleStore, marketdesk_dir

__all__ = [
    "MarketsDesk", "Candle", "CandleStore", "marketdesk_dir",
    "collect_symbol", "collect_all",
    "ensure_marketdesk_jobs", "render_collect", "render_library",
    "render_status",
    "Pattern", "PatternStore", "BUILTIN_ID",
    "parse_rule", "backtest", "study_rule", "list_patterns",
    "set_watching", "check_watches", "pattern_status", "remove_pattern",
    "render_study", "render_pattern_list", "render_check",
    "render_pattern_status",
    "register",
]


def _patterns_action(context: Any, *, verb: str = "", rule: str = "",
                     symbol: str = "", pattern_id: str = "") -> dict[str, Any]:
    """The marketdesk tool's ``patterns`` action — one verb, one call."""
    from .patterns import RULE_SYNTAX

    settings = getattr(context, "settings", None)
    v = (verb or "list").strip().lower()
    if v == "study":
        if not (rule or "").strip():
            return {"ok": False,
                    "error": "study needs a rule — " + RULE_SYNTAX}
        text = rule.strip()
        if ((symbol or "").strip()
                and not re.search(r"\bfor\s+[A-Za-z]{2,12}\s*$",
                                  text, flags=re.IGNORECASE)):
            text = f"{text} for {symbol.strip().upper()}"
        return study_rule(text, settings=settings)
    if v == "list":
        return list_patterns(settings=settings)
    if v in ("watch", "unwatch"):
        if not (pattern_id or "").strip():
            return {"ok": False,
                    "error": f"{v} needs a pattern_id — see verb=list"}
        return set_watching(pattern_id, v == "watch", settings=settings)
    if v == "check":
        return check_watches(settings=settings)
    if v == "status":
        if not (pattern_id or "").strip():
            return {"ok": False,
                    "error": "status needs a pattern_id — see verb=list"}
        return pattern_status(pattern_id, settings=settings)
    if v == "remove":
        if not (pattern_id or "").strip():
            return {"ok": False,
                    "error": "remove needs a pattern_id — see verb=list"}
        return remove_pattern(pattern_id, settings=settings)
    return {"ok": False,
            "error": f"unknown patterns verb {verb!r} — "
                     "study|list|watch|unwatch|check|status|remove"}


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
            "state) | patterns (study plain-words market patterns against "
            "real history, watch them live — verb=study|list|watch|"
            "unwatch|check|status|remove, rule=the plain-words rule, "
            "symbol=XAUUSD by default, pattern_id=the pattern to act on)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — study|signals|status|library|collect|sources|patterns",
            "deliver": "bool — publish signal calls when action=signals",
            "verb": "str — patterns action only: study|list|watch|unwatch|check|status|remove",
            "rule": "str — patterns action, verb=study: the plain-words rule, e.g. 'Friday high < Thursday high => Monday low <= Friday low for XAUUSD'",
            "symbol": "str — patterns action: symbol the pattern studies (default XAUUSD)",
            "pattern_id": "str — patterns action: which pattern to watch/unwatch/status/remove",
        },
    )
    def marketdesk(*, action: str = "status",
                   deliver: bool = False,
                   verb: str = "", rule: str = "",
                   symbol: str = "", pattern_id: str = "") -> dict[str, Any]:
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
        if act == "patterns":
            return _patterns_action(
                context, verb=verb, rule=rule, symbol=symbol,
                pattern_id=pattern_id)
        return {"error": f"unknown action {act!r} — "
                        "study|signals|status|library|collect|sources|patterns"}
