"""Standing whole-market read — Devon trading system, Pass 1.

One organ that, for a given instrument, assembles the whole-market
picture the rest of the trading system consumes: D1·H4·H1·M30·M15·M5
bars + live price from the existing market-data providers, per-
timeframe structure bias and zones from the existing TA library
(``ta/structure`` + ``ta/smc``), session context from ``ta/sessions``.

Doctrine shapes (settled, GoldenSwing): the hierarchy is Swing D1 /
Intraday H4 / Scalp H1 — a hierarchy, never a vote; bias and phase
come from market STRUCTURE (swing sequence), never from an EMA read;
the spine is asset-agnostic with XAUUSD the primary instrument.

Feed honesty: every timeframe block names its provider, freshness and
last-bar age from the provider frame's own attrs. A timeframe whose
data is missing or stale is marked ``missing`` / ``stale`` in the
read — it is never silently filled from another timeframe, and its
analysis is computed from its own bars or not at all.

This module is the assembly + contract only: no new indicators, no
judgment, no order authority. Pass 2 playbooks and Pass 3 judgment
consume :func:`whole_market_read`'s dict (and the journaled snapshots
via :func:`recent_reads`). It lives in ``agents/`` beside
FinancialExpert because that is the layer from which the market-data
integrations, the TA library and the finance journal are all legal
static imports (``ta/`` itself cannot statically import
``integrations/`` — see ``ta/feeds.py``'s import_module pattern).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..finance import trading_journal
from ..integrations import market_data
from ..ta import sessions, smc, structure

_log = logging.getLogger(__name__)

#: The read's timeframes, highest first (provider spellings).
TIMEFRAMES: tuple[str, ...] = ("1d", "4h", "1h", "30m", "15m", "5m")
TF_LABELS = {"1d": "D1", "4h": "H4", "1h": "H1",
             "30m": "M30", "15m": "M15", "5m": "M5"}
#: Doctrine roles: the hierarchy levels plus the execution context.
TF_ROLES = {"1d": "swing", "4h": "intraday", "1h": "scalp",
            "30m": "execution", "15m": "execution", "5m": "execution"}
#: (level, timeframe) — Swing D1 / Intraday H4 / Scalp H1.
HIERARCHY: tuple[tuple[str, str], ...] = (
    ("swing", "1d"), ("intraday", "4h"), ("scalp", "1h"))
#: Bars fetched per timeframe (enough for swings, zones, and the
#: 100-bar premium/discount dealing range on every timeframe).
DEFAULT_BARS = {"1d": 260, "4h": 300, "1h": 300,
                "30m": 300, "15m": 320, "5m": 320}
#: Below this many bars, structure analysis is not attempted.
_MIN_STRUCTURE_BARS = 30


def _zone_row(row: Any) -> dict[str, Any]:
    return {"top": float(row["top"]), "bottom": float(row["bottom"]),
            "mid": float(row["mid"]), "touches": int(row["touches"]),
            "strength": float(row["strength"]),
            "kind": str(row["kind"]), "state": str(row["state"])}


def _gap_row(row: Any, kind_word: str) -> dict[str, Any]:
    return {"kind": kind_word, "top": float(row["top"]),
            "bottom": float(row["bottom"]), "mid": float(row["mid"])}


def _nearest_pair(frame: Any, ref: float, kind_of: Any) -> dict[str, Any]:
    """Nearest zone-like rows above/below ``ref`` from a mid column."""
    if frame is None or len(frame) == 0 or "mid" not in frame.columns:
        return {"above": None, "below": None, "open_count": 0}
    above = frame[frame["mid"] > ref]
    below = frame[frame["mid"] < ref]
    out: dict[str, Any] = {"above": None, "below": None,
                           "open_count": int(len(frame))}
    if not above.empty:
        out["above"] = kind_of(above.sort_values("mid").iloc[0])
    if not below.empty:
        out["below"] = kind_of(below.sort_values("mid").iloc[-1])
    return out


def _analyze(df: Any, ref: float, block: dict[str, Any]) -> None:
    """Fill one timeframe block's structure analysis from its own bars.

    Bias/phase are structure reads (swing sequence via
    ``structure.market_bias``); zones, gaps and blocks come from the
    TA library unchanged. Any failure degrades only the analysis
    fields — the block's data status stays as the provider reported.
    """
    if len(df) < _MIN_STRUCTURE_BARS:
        block["analysis_note"] = (
            f"insufficient bars for structure ({len(df)} "
            f"< {_MIN_STRUCTURE_BARS})")
        return
    try:
        pts = structure.swing_points(df)
        if not pts.empty:
            bias = structure.market_bias(structure.label_swings(pts))
            block["bias"] = {"bias": int(bias["bias"]),
                             "label": str(bias["label"]),
                             "reason": str(bias["reason"])}
            # Phase is the structure read, per doctrine — the swing
            # sequence's label, never an indicator cross.
            block["phase"] = str(bias["label"]).lower()
        zones = structure.sr_zones(df)
        levels = structure.nearest_levels(zones, ref)
        inside = ([_zone_row(r) for _, r in
                   zones[zones["state"] == "inside"].iterrows()]
                  if not zones.empty else [])
        block["zones"] = {
            "support": (_zone_row(levels["support"])
                        if levels["support"] is not None else None),
            "resistance": (_zone_row(levels["resistance"])
                           if levels["resistance"] is not None else None),
            "inside": inside}
        fvgs = smc.unmitigated_fvgs(df)
        block["fvgs"] = _nearest_pair(
            fvgs, ref,
            lambda r: _gap_row(r, "bullish" if r["kind"] > 0 else "bearish"))
        blocks = smc.order_blocks(df)
        open_blocks = blocks[~blocks["mitigated"]] \
            if not blocks.empty else blocks
        block["order_blocks"] = _nearest_pair(
            open_blocks, ref,
            lambda r: _gap_row(r, "bullish" if r["kind"] > 0 else "bearish"))
        pd_frame = smc.premium_discount(df)
        last = pd_frame.iloc[-1]
        block["premium_discount"] = {
            "premium": float(last["premium"]),
            "zone": str(last["zone"]),
            "range_high": float(last["range_high"]),
            "range_low": float(last["range_low"])}
        breaks = structure.structure_breaks(df)
        if not breaks.empty:
            ev = breaks.iloc[-1]
            block["last_break"] = {
                "kind": str(ev["kind"]), "level": float(ev["level"]),
                "bias_before": int(ev["bias_before"])}
    except Exception as exc:  # noqa: BLE001 - analysis degrades, read stands
        block["analysis_error"] = f"{type(exc).__name__}: {exc}"
        _log.warning("market_read: analysis failed: %s", exc)


def _read_timeframe(symbol: str, market: str, timeframe: str,
                    ref_price: float | None, bars: int,
                    now: float) -> dict[str, Any]:
    block: dict[str, Any] = {
        "timeframe": timeframe, "label": TF_LABELS[timeframe],
        "role": TF_ROLES[timeframe], "status": "missing",
        "source": None, "freshness": None, "last_bar_ts": None,
        "bar_age_s": None, "bars": 0, "bias": None, "phase": None,
        "zones": None, "fvgs": None, "order_blocks": None,
        "premium_discount": None, "last_break": None, "error": None}
    try:
        df = market_data.get_ohlcv(symbol, market=market,
                                   timeframe=timeframe, bars=bars)
    except Exception as exc:  # noqa: BLE001 - one TF down ≠ read down
        block["error"] = f"{type(exc).__name__}: {exc}"
        _log.info("market_read: %s %s unavailable: %s",
                  symbol, timeframe, exc)
        return block
    if df is None or len(df) == 0:
        block["error"] = "provider returned no bars"
        return block
    attrs = getattr(df, "attrs", None) or {}
    block["source"] = attrs.get("source")
    block["freshness"] = attrs.get("freshness") or "unknown"
    last_ts = attrs.get("last_bar_ts")
    block["last_bar_ts"] = float(last_ts) if last_ts is not None else None
    if block["last_bar_ts"] is not None:
        block["bar_age_s"] = round(max(0.0, now - block["last_bar_ts"]), 1)
    if attrs.get("source_demoted"):
        block["source_demoted"] = True
    block["bars"] = int(len(df))
    block["status"] = "stale" if block["freshness"] == "stale" else "ok"
    ref = ref_price if ref_price is not None else float(df["close"].iloc[-1])
    _analyze(df, ref, block)
    return block


def _live_price(symbol: str, market: str) -> tuple[dict[str, Any],
                                                   float | None]:
    try:
        q = market_data.quote(symbol, market=market)
    except Exception as exc:  # noqa: BLE001 - bars still carry the read
        _log.info("market_read: live quote for %s failed: %s", symbol, exc)
        return {"price": None,
                "error": f"{type(exc).__name__}: {exc}"}, None
    block: dict[str, Any] = {
        "price": float(q["price"]), "source": q.get("source"),
        "currency": q.get("currency"), "quote_ts": q.get("quote_ts"),
        "quote_age_s": q.get("quote_age_s"),
        "freshness": q.get("freshness"), "bid": q.get("bid"),
        "ask": q.get("ask"), "spread_bps": q.get("spread_bps"),
        "change_pct_24h": q.get("change_pct_24h")}
    if q.get("source_demoted"):
        block["source_demoted"] = True
    return block, float(q["price"])


def _iso_utc(taken: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(taken))


def _session_block(taken: float) -> dict[str, Any] | None:
    try:
        ts = _iso_utc(taken)
        return {"name": sessions.session_at(ts),
                "killzone": bool(sessions.in_killzone(ts))}
    except Exception as exc:  # noqa: BLE001 - context only, never fatal
        _log.debug("market_read: session lookup failed: %s", exc)
        return None


def whole_market_read(symbol: str, market: str = "crypto", *,
                      journal: bool = False, settings: Any = None,
                      journal_path: Any = None,
                      now: float | None = None,
                      bars: dict[str, int] | None = None) -> dict[str, Any]:
    """Assemble the standing whole-market read for one instrument.

    Returns the typed snapshot dict (see module docstring): per-
    timeframe structure/zones with provider provenance, the Swing /
    Intraday / Scalp hierarchy, live price and session context.
    With ``journal=True`` the snapshot is appended to the trading
    system journal as a ``market_read`` event for later grading.
    """
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    price_block, price = _live_price(sym, market)
    timeframes = {}
    for tf in TIMEFRAMES:
        want = (bars or {}).get(tf, DEFAULT_BARS[tf])
        timeframes[tf] = _read_timeframe(sym, market, tf, price,
                                         want, taken)
    hierarchy: dict[str, Any] = {}
    for level, tf in HIERARCHY:
        blk = timeframes[tf]
        hierarchy[level] = {
            "timeframe": tf, "label": TF_LABELS[tf],
            "status": blk["status"],
            "bias": blk["bias"]["bias"] if blk["bias"] else None,
            "bias_label": blk["bias"]["label"] if blk["bias"] else None}
    signs = [hierarchy[level]["bias"] for level, _tf in HIERARCHY]
    if any(s is None for s in signs):
        alignment = "insufficient"
    elif all(s == 1 for s in signs):
        alignment = "aligned_up"
    elif all(s == -1 for s in signs):
        alignment = "aligned_down"
    elif all(s == 0 for s in signs):
        alignment = "flat"
    else:
        alignment = "mixed"
    degraded = [tf for tf in TIMEFRAMES
                if timeframes[tf]["status"] != "ok"]
    read: dict[str, Any] = {
        "kind": "market_read", "version": 1, "symbol": sym,
        "market": market, "taken_at": taken,
        "taken_at_utc": _iso_utc(taken),
        "price": price_block, "session": _session_block(taken),
        "hierarchy": hierarchy, "alignment": alignment,
        "timeframes": timeframes, "degraded": degraded,
        "complete": not degraded}
    if journal:
        trading_journal.append_event(
            "market_read", settings=settings, path=journal_path,
            symbol=sym, snapshot=read)
    return read


def recent_reads(symbol: str | None = None, *, limit: int = 10,
                 settings: Any = None,
                 journal_path: Any = None) -> list[dict[str, Any]]:
    """Journaled read snapshots, newest first (optionally per symbol)."""
    out: list[dict[str, Any]] = []
    for rec in trading_journal.read_events(
            limit=0, event="market_read", settings=settings,
            path=journal_path):
        snap = rec.get("snapshot")
        if not isinstance(snap, dict):
            continue
        if symbol is not None and snap.get("symbol") != \
                symbol.strip().upper():
            continue
        out.append(snap)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _fmt_price(value: Any) -> str:
    if value is None:
        return "?"
    return f"{float(value):,.2f}"


def render_read(read: dict[str, Any]) -> str:
    """Human rendering of one read — the chat/CLI surface text."""
    price = read.get("price") or {}
    sess = read.get("session") or {}
    lines = [f"{read.get('symbol')} whole-market read — "
             f"{(read.get('taken_at_utc') or '')[:16].replace('T', ' ')} UTC"]
    if price.get("price") is not None:
        age = price.get("quote_age_s")
        age_txt = f", {age:.0f}s old" if age is not None else ""
        lines.append(
            f"price {_fmt_price(price['price'])} via "
            f"{price.get('source') or '?'} "
            f"({price.get('freshness') or 'unknown'}{age_txt})")
    else:
        lines.append(f"live price unavailable ({price.get('error')})")
    if sess:
        lines.append(f"session {sess.get('name')}"
                     + (" · killzone" if sess.get("killzone") else ""))
    hier = read.get("hierarchy") or {}
    hier_txt = " · ".join(
        f"{level.capitalize()} {hier.get(level, {}).get('label')}: "
        f"{hier.get(level, {}).get('bias_label') or '—'}"
        for level, _tf in HIERARCHY)
    lines.append(f"{hier_txt} → {read.get('alignment')}")
    for tf in TIMEFRAMES:
        blk = read["timeframes"][tf]
        label = blk["label"]
        if blk["status"] == "missing":
            lines.append(f"{label:<4} NO DATA — {blk.get('error')}")
            continue
        zones = blk.get("zones") or {}
        sup = (zones.get("support") or {}).get("mid")
        res = (zones.get("resistance") or {}).get("mid")
        pd_block = blk.get("premium_discount") or {}
        stale = " STALE" if blk["status"] == "stale" else ""
        lines.append(
            f"{label:<4} {(blk.get('phase') or '—'):<10} "
            f"S {_fmt_price(sup)} / R {_fmt_price(res)} · "
            f"{pd_block.get('zone') or '—'} "
            f"({pd_block.get('premium', 0):.2f}) · "
            f"{blk.get('source') or '?'}{stale}")
    return "\n".join(lines)
