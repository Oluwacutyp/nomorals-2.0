"""Afternoon pulse — midday market and news check-in.

Fires daily at 13:00 America/Denver (user's timezone). Lighter than the
morning pulse: market movers, breaking news since morning, weather
afternoon outlook. Same pipeline (news → briefing → voice → delivery)
with graceful degradation.

Scheduler: ``ensure_afternoon_pulse_job`` registers one durable
``afternoon-pulse`` job (idempotent by name), ``daily 13:00`` in
America/Denver.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

PULSE_JOB_NAME = "afternoon-pulse"
DEFAULT_PULSE_TIME = "13:00"
PULSE_TIMEZONE = "America/Denver"

PULSE_POLICIES = {
    "missed_fire_policy": "fire_now",
    "overlap_policy": "skip",
    "max_runtime_s": 1800,
}


def pulse_time(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("afternoon_pulse_time") or DEFAULT_PULSE_TIME)


def pulse_timezone(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("pulse_timezone") or PULSE_TIMEZONE)


def pulse_enabled(context: Any) -> bool:
    prefs = _prefs(context)
    return bool(prefs.get("afternoon_pulse_enabled", True))


def _prefs(context: Any) -> dict[str, Any]:
    try:
        from .morning_pulse import _prefs as _morning_prefs
        return _morning_prefs(context)
    except Exception:
        return {}


def ensure_afternoon_pulse_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``afternoon-pulse`` job (idempotent)."""
    from .scheduler import Scheduler

    sched = Scheduler(context)
    want = pulse_time(context)
    tz = pulse_timezone(context)
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == PULSE_JOB_NAME]
    except Exception:  # noqa: BLE001
        have = []
    if have:
        job0 = have[0]
        same_policies = all(
            job0.get(k) == v for k, v in PULSE_POLICIES.items())
        if job0.get("spec") == f"daily {want}" and same_policies:
            return {"name": PULSE_JOB_NAME, "already_scheduled": True,
                    "job_id": job0.get("id")}
    for j in have:
        try:
            sched.remove(j["id"])
        except Exception:  # noqa: BLE001
            pass
    job = sched.add(PULSE_JOB_NAME, f"daily {want}", "tool",
                    {"tool": "pulse", "args": {"action": "run_afternoon"}},
                    timezone=tz, **PULSE_POLICIES)
    _log.info("scheduled afternoon-pulse job: %s (daily %s %s)",
              PULSE_JOB_NAME, want, tz)
    return {"name": PULSE_JOB_NAME, "scheduled": True,
            "job_id": job.get("id")}


def run_afternoon_pulse(context: Any) -> dict[str, Any]:
    """Run the afternoon pulse pipeline. Never raises.

    Focus: market midday movers, breaking news since morning,
    afternoon weather outlook. Shorter than morning — a check-in,
    not a full briefing.
    """
    from .morning_pulse import _run_stage, _fetch_news
    from ..core.tz import safe_zoneinfo
    from datetime import datetime

    started = datetime.now().timestamp()
    result: dict[str, Any] = {"ok": True, "stages": {}}

    # Stage 1: fresh news since morning
    ok, news_items, secs, attempts = _run_stage(
        "news", lambda: _fetch_news(context))
    result["stages"]["news"] = {
        "ok": ok, "items": len(news_items) if ok else 0, "seconds": secs}

    # Stage 2: market snapshot (midday movers)
    def _market_snapshot() -> list[dict[str, Any]]:
        from ..integrations.market_data import quote
        symbols = ["BTC", "ETH", "XAUUSD", "EURUSD"]
        out = []
        for sym in symbols:
            try:
                market = "commodities" if sym == "XAUUSD" else (
                    "forex" if sym == "EURUSD" else "crypto")
                q = quote(sym, market=market)
                if q:
                    out.append(q)
            except Exception:  # noqa: BLE001 — one dead symbol is fine
                continue
        return out

    ok, market_data, secs, attempts = _run_stage("market", _market_snapshot)
    result["stages"]["market"] = {
        "ok": ok, "symbols": len(market_data) if ok else 0, "seconds": secs}

    # Stage 3: compose afternoon briefing text
    def _compose() -> str:
        from ..llm.brain import brain_for
        brain = brain_for(context)
        tz = safe_zoneinfo(pulse_timezone(context))
        now = datetime.now(tz).strftime("%I:%M %p")
        news_summary = "\n".join(
            f"- {n.get('title', '')}" for n in (news_items or [])[:8])
        market_summary = "\n".join(
            f"- {m.get('symbol', '')}: {m.get('price', '?')} "
            f"({m.get('change_pct_24h', 0):+.1f}%)"
            for m in (market_data or []))
        prompt = (
            f"Write a brief afternoon check-in for {now}. Keep it under "
            f"150 words, conversational.\n\nTop news:\n{news_summary}\n\n"
            f"Markets:\n{market_summary}\n\nFocus on what's moved since "
            f"morning and what to watch this afternoon.")
        resp = brain.chat([{"role": "user", "content": prompt}])
        return resp.text if hasattr(resp, "text") else str(resp)

    ok, briefing_text, secs, attempts = _run_stage("compose", _compose)
    result["stages"]["compose"] = {"ok": ok, "seconds": secs}
    if not ok or not briefing_text:
        briefing_text = "Afternoon check-in: markets and news update unavailable."
        result["ok"] = False

    # Stage 4: deliver
    def _deliver() -> bool:
        from .morning_pulse import _ledger
        _ledger(context, "afternoon-pulse", briefing_text[:200])
        # Publish via notifier
        try:
            notifier = getattr(context, "notifier", None)
            if notifier:
                notifier.publish("afternoon-pulse", briefing_text)
                return True
        except Exception:  # noqa: BLE001
            pass
        _log.info("afternoon pulse: %s", briefing_text[:500])
        return True

    ok, _, secs, attempts = _run_stage("deliver", _deliver)
    result["stages"]["deliver"] = {"ok": ok, "seconds": secs}
    result["seconds"] = datetime.now().timestamp() - started
    result["briefing"] = briefing_text
    return result
