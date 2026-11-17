"""The markets desk — Devon's standing BTC/stocks study organ.

The desk EXTENDS the price class (``integrations/market_data``,
the keyless-first OHLCV chain) with the things a desk does that a
quote doesn't:

* **continuous study** — scheduled cycles that collect daily +
  hourly bars, re-run the pattern lab, refresh the Library, and
  scan for callable time structure (analysis only);
* **a pattern lab** (:mod:`.lab`) testing trend, momentum,
  volatility regime, levels, mean reversion, autocorrelation,
  streaks, and the hour-of-day effect, per symbol, with verdicts
  in writing;
* **signal calls** (:mod:`.signals`) — timed alerts with >= 2
  minutes of lead, issued only off lab-confirmed hour structure;
* **a Library** — one structured record per symbol: sample sizes,
  the tests run, the verdict with its evidence, and what would
  change it.  Findings persist across restarts and cycles;
* **self-knowledge of its feeds** — its data sources live in the
  shared source registry (:mod:`.sources`), kind ``markets``.

Doctrine (the owner's, non-negotiable): the desk studies, signals,
proposes.  It contains NO execution path and never will —
GoldenSwing / Devon trading remain the execution arms; real money
moves only on the owner's explicit word, outside this code.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from . import lab as _lab
from .collectors import DEFAULT_SYMBOLS, SYMBOL_MARKETS, collect_all
from .failures import record_failure
from .signals import SignalEngine
from .store import CandleStore, marketdesk_dir

__all__ = ["MarketsDesk", "render_status", "render_library",
           "render_collect"]


class MarketsDesk:
    """One desk instance.  ``context`` is the runtime context when
    bound (Notifier delivery); CLI/tests run context-free.

    The registry is RESOLVED through
    :func:`.sources.market_registry_for_context` (context-KV-bound),
    so the desk's seeds and outcomes persist into the same state
    the health loop verifies."""

    def __init__(self, context: Any = None, desk_dir: str = "",
                 store: CandleStore | None = None,
                 notifier: Any = None):
        self.context = context
        self.dir = desk_dir or marketdesk_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.store = store or CandleStore(self.dir)
        if notifier is None and context is not None:
            try:
                from ..notifier import Notifier
                notifier = Notifier(context)
            except Exception as e:  # delivery is optional kit
                record_failure("desk.__init__.notifier", e)
                notifier = None
        self.signals = SignalEngine(self.dir, notifier=notifier)
        self._registry = None

    # ── shared source registry (lazy; the ONE registry) ──────────────

    @property
    def registry(self) -> Any:
        if self._registry is None:
            from .sources import ensure_market_sources, market_registry_for_context
            self._registry = market_registry_for_context(self.context)
            ensure_market_sources(self._registry)
        return self._registry

    # ── library ───────────────────────────────────────────────────────

    def _library_path(self) -> str:
        return os.path.join(self.dir, "library.json")

    def library(self) -> dict[str, Any]:
        """The desk Library: per-symbol study records, keyed by symbol."""
        if os.path.exists(self._library_path()):
            try:
                with open(self._library_path(), encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError) as e:
                record_failure("desk.library", e)
                pass
        return {}

    def _save_library(self, lib: dict[str, Any]) -> None:
        tmp = self._library_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(lib, fh, indent=1)
        os.replace(tmp, self._library_path())

    # ── study cycle ───────────────────────────────────────────────────

    def analyze_all(self) -> dict[str, Any]:
        """Run the lab over every symbol in the store; refresh Library."""
        lib = self.library()
        verdicts: dict[str, Any] = {}
        for symbol in self.store.symbols():
            daily = self.store.bars(symbol, "1d")
            hourly = self.store.bars(symbol, "1h")
            v = _lab.analyze_symbol(daily, hourly, symbol=symbol)
            verdicts[symbol] = v.to_dict()
            rec = lib.get(symbol, {})
            rec.update(v.to_dict())
            rec["updated_at"] = time.time()
            rec["market"] = SYMBOL_MARKETS.get(symbol, "")
            lib[symbol] = rec
        self._save_library(lib)
        return verdicts

    def collect(self) -> dict[str, Any]:
        """One collection cycle: daily + hourly bars per watched symbol.

        Every symbol reports its own outcome; a dead symbol never
        kills the cycle.
        """
        out: dict[str, Any] = {"at": time.time()}
        try:
            out["symbols"] = collect_all(self.store)
        except Exception as exc:  # noqa: BLE001 - a leg reports, not raises
            out["symbols"] = {}
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["added"] = sum((s or {}).get("added", 0)
                           for s in (out.get("symbols") or {}).values())
        return out

    def study(self, deliver_signals: bool = False) -> dict[str, Any]:
        """One full desk cycle: collect -> lab -> (optionally) signals.

        Every leg reports its own outcome; a dead leg never kills the
        cycle.
        """
        out: dict[str, Any] = {"at": time.time()}
        coll = self.collect()
        out["collect"] = coll
        out["verdicts"] = self.analyze_all()
        out["signals"] = self.run_signals(deliver=deliver_signals)
        return out

    # ── signals ───────────────────────────────────────────────────────

    def run_signals(self, deliver: bool = True) -> dict[str, Any]:
        """Evaluate every Library verdict for a call; publish the
        callable ones; score any due calls.  No callable verdict ->
        an explicit no-call report, never silence."""
        lib = self.library()
        calls, reasons = [], []
        for symbol, rec in sorted(lib.items()):
            v = _verdict_from_record(symbol, rec)
            decision = self.signals.evaluate(v)
            if decision["call"] is not None:
                calls.append(self.signals.publish(decision["call"],
                                                  deliver=deliver))
            else:
                reasons.append(decision["reason"])
        return {"published": calls, "no_call_reasons": reasons,
                "scoring": self.signals.score_due(self.store)}

    # ── status ────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        from .sources import keyed_feed_state, market_status
        lib = self.library()
        symbols = []
        for symbol in self.store.symbols():
            rec = lib.get(symbol, {})
            symbols.append({
                "symbol": symbol,
                "market": SYMBOL_MARKETS.get(symbol, ""),
                "bars_daily": len(self.store.bars(symbol, "1d")),
                "bars_hourly": len(self.store.bars(symbol, "1h")),
                "verdict": rec.get("verdict", "NOT_STUDIED"),
                "callable_structure": rec.get("callable_structure", ""),
                "evidence": rec.get("evidence", ""),
            })
        scoring = self.signals.score_due(self.store)
        try:
            sources = market_status(self.registry)
        except Exception as exc:  # noqa: BLE001
            sources = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "bars_total": self.store.count(),
            "symbols": symbols,
            "signals": scoring,
            "min_lead_seconds": self.signals.min_lead,
            "sources": sources,
            "keyed_feeds": keyed_feed_state(),
            "library_records": len(lib),
            "watchlist": list(DEFAULT_SYMBOLS),
        }


def _verdict_from_record(symbol: str, rec: dict[str, Any]) -> Any:
    """Rebuild the lab verdict fields the signal engine reads."""
    from types import SimpleNamespace
    v = SimpleNamespace(
        symbol=symbol,
        verdict=rec.get("verdict", "INSUFFICIENT_DATA"),
        callable_structure=rec.get("callable_structure", ""),
        best_hours=list(rec.get("best_hours", [])),
        hour_rates={int(k): float(vv)
                    for k, vv in (rec.get("hour_rates") or {}).items()},
        base_success_rate=float(rec.get("base_success_rate", 0.0)),
        evidence=rec.get("evidence", ""))
    return v


# ── rendering ──────────────────────────────────────────────────────────

def render_status(st: dict[str, Any]) -> str:
    L = [f"📈 markets desk — {st.get('bars_total', 0)} bars in the "
         f"ledger, {st.get('library_records', 0)} library records"]
    for s in st.get("symbols", []):
        L.append(f"  {s['symbol']} ({s.get('market', '')}): "
                 f"{s['bars_daily']} daily + {s['bars_hourly']} hourly "
                 f"bars — {s['verdict']}"
                 + (f" (callable: {s['callable_structure']})"
                    if s.get("callable_structure") else ""))
        if s.get("evidence"):
            L.append(f"    {s['evidence'][:220]}")
    sig = st.get("signals", {})
    if sig.get("calls_total"):
        L.append(f"signals: {sig['calls_scored']}/{sig['calls_total']} "
                 f"calls scored, hit rate "
                 f"{(sig.get('hit_rate') or 0):.0%}")
    else:
        L.append("signals: no calls published yet (nothing has validated "
                 "as callable — see /markets library)")
    L.append(f"signal lead floor: {st.get('min_lead_seconds', 120)}s")
    for f in st.get("keyed_feeds", []):
        if not f["key_present"]:
            L.append(f"key {f['env_key']}: missing — {f['status']}")
        else:
            L.append(f"key {f['env_key']}: present")
    src = (st.get("sources") or {}).get("sources", [])
    if src:
        L.append("sources: " + ", ".join(
            f"{s['domain']} [{s['health']}]" for s in src))
    missing = [s for s in st.get("watchlist", [])
               if s not in {x["symbol"] for x in st.get("symbols", [])}]
    if missing:
        L.append("not yet collected: " + ", ".join(missing)
                 + " — run /markets study")
    return "\n".join(L)


def render_collect(rep: dict[str, Any]) -> str:
    """Render one collection cycle in the desk's house style — every
    symbol named with its own outcome."""
    L = [f"collect: +{rep.get('added', 0)} bars"]
    if rep.get("error"):
        L.append(f"  collect failed: {rep['error']}")
    for symbol, s in sorted((rep.get("symbols") or {}).items()):
        if s.get("ok"):
            tf = s.get("timeframes", {})
            L.append(f"  {symbol}: +{s.get('added', 0)} bars "
                     f"(1d +{tf.get('1d', {}).get('added', 0)}, "
                     f"1h +{tf.get('1h', {}).get('added', 0)})")
        else:
            L.append(f"  {symbol}: failed — {s.get('error', '')}")
    return "\n".join(L)


def render_library(lib: dict[str, Any]) -> str:
    if not lib:
        return ("the desk library is empty — no symbol has been studied "
                "yet (collect bars first: /markets study)")
    blocks = []
    for symbol, rec in sorted(lib.items()):
        lines = [f"📚 {symbol} — {rec.get('verdict', '?')} "
                 f"({rec.get('n_daily', 0)} daily, "
                 f"{rec.get('n_hourly', 0)} hourly bars)"]
        if rec.get("evidence"):
            lines.append(f"  evidence: {rec['evidence'][:400]}")
        if rec.get("what_would_change"):
            lines.append(f"  what would change it: "
                         f"{rec['what_would_change'][:200]}")
        for t in rec.get("tests", []):
            if t.get("skipped"):
                lines.append(f"  - {t['name']}: skipped ({t['skipped']})")
            else:
                lines.append(f"  - {t['name']}: p={t['p_value']:.3g} — "
                             f"{t.get('detail', '')[:120]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
