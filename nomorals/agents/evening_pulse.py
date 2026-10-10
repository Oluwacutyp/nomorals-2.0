"""Evening pulse — end-of-day recap and tomorrow preview.

Fires daily at 21:00 America/Denver (user's timezone). Focus: day's
market close, top stories of the day, tomorrow's outlook, any alerts
or reminders for tomorrow. Same pipeline with graceful degradation.

Scheduler: ``ensure_evening_pulse_job`` registers one durable
``evening-pulse`` job (idempotent by name), ``daily 21:00`` in
America/Denver.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

PULSE_JOB_NAME = "evening-pulse"
DEFAULT_PULSE_TIME = "21:00"
PULSE_TIMEZONE = "America/Denver"

PULSE_POLICIES = {
    "missed_fire_policy": "fire_now",
    "overlap_policy": "skip",
    "max_runtime_s": 1800,
}


def pulse_time(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("evening_pulse_time") or DEFAULT_PULSE_TIME)


def pulse_timezone(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("pulse_timezone") or PULSE_TIMEZONE)


def pulse_enabled(context: Any) -> bool:
    prefs = _prefs(context)
    return bool(prefs.get("evening_pulse_enabled", True))


def _prefs(context: Any) -> dict[str, Any]:
    try:
        from .morning_pulse import _prefs as _morning_prefs
        return _morning_prefs(context)
    except Exception:
        return {}


def ensure_evening_pulse_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``evening-pulse`` job (idempotent)."""
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
                    {"tool": "pulse", "args": {"action": "run_evening"}},
                    timezone=tz, **PULSE_POLICIES)
    _log.info("scheduled evening-pulse job: %s (daily %s %s)",
              PULSE_JOB_NAME, want, tz)
    return {"name": PULSE_JOB_NAME, "scheduled": True,
            "job_id": job.get("id")}


def run_evening_pulse(context: Any) -> dict[str, Any]:
    """Run the evening pulse pipeline. Never raises.

    Focus: market close recap, day's top stories, tomorrow's calendar
    and outlook, overnight watch items.
    """
    from .morning_pulse import _run_stage, _fetch_news
    from ..core.tz import safe_zoneinfo
    from datetime import datetime

    started = datetime.now().timestamp()
    result: dict[str, Any] = {"ok": True, "stages": {}}

    # Stage 1: day's news
    ok, news_items, secs, attempts = _run_stage(
        "news", lambda: _fetch_news(context))
    result["stages"]["news"] = {
        "ok": ok, "items": len(news_items) if ok else 0, "seconds": secs}

    # Stage 2: market close data
    def _market_close() -> list[dict[str, Any]]:
        from ..integrations.market_data import quote
        symbols = ["BTC", "ETH", "XAUUSD", "EURUSD", "SPY"]
        out = []
        for sym in symbols:
            try:
                if sym == "XAUUSD":
                    market = "commodities"
                elif sym in ("EURUSD",):
                    market = "forex"
                elif sym == "SPY":
                    market = "stocks"
                else:
                    market = "crypto"
                q = quote(sym, market=market)
                if q:
                    out.append(q)
            except Exception:  # noqa: BLE001
                continue
        return out

    ok, market_data, secs, attempts = _run_stage("market", _market_close)
    result["stages"]["market"] = {
        "ok": ok, "symbols": len(market_data) if ok else 0, "seconds": secs}

    # Stage 3: tomorrow's outlook (calendar + weather)
    def _tomorrow_preview() -> dict[str, Any]:
        preview: dict[str, Any] = {}
        # Weather for tomorrow
        try:
            from .weather import geocode, fetch_forecast
            place = geocode("Lagos")
            if place:
                fc = fetch_forecast(place.latitude, place.longitude,
                                    days=2, units="metric")
                if fc and fc.get("daily"):
                    preview["weather_tomorrow"] = fc["daily"][1] if len(
                        fc["daily"]) > 1 else fc["daily"][0]
        except Exception:  # noqa: BLE001
            pass
        return preview

    ok, tomorrow, secs, attempts = _run_stage("tomorrow", _tomorrow_preview)
    result["stages"]["tomorrow"] = {"ok": ok, "seconds": secs}

    # Stage 4: compose evening recap
    def _compose() -> str:
        from ..llm.brain import brain_for
        brain = brain_for(context)
        tz = safe_zoneinfo(pulse_timezone(context))
        now = datetime.now(tz).strftime("%I:%M %p")
        news_summary = "\n".join(
            f"- {n.get('title', '')}" for n in (news_items or [])[:10])
        market_summary = "\n".join(
            f"- {m.get('symbol', '')}: {m.get('price', '?')} "
            f"({m.get('change_pct_24h', 0):+.1f}%)"
            for m in (market_data or []))
        weather_str = ""
        if tomorrow and tomorrow.get("weather_tomorrow"):
            wt = tomorrow["weather_tomorrow"]
            weather_str = (
                f"\nTomorrow's weather: high {wt.get('temp_max', '?')}°, "
                f"low {wt.get('temp_min', '?')}°.")
        prompt = (
            f"Write an evening recap for {now}. Under 200 words, warm and "
            f"conversational.\n\nDay's top stories:\n{news_summary}\n\n"
            f"Market close:\n{market_summary}{weather_str}\n\n"
            f"Cover: how the day went, what mattered, and one thing to "
            f"watch overnight or tomorrow morning.")
        resp = brain.chat([{"role": "user", "content": prompt}])
        return resp.text if hasattr(resp, "text") else str(resp)

    ok, briefing_text, secs, attempts = _run_stage("compose", _compose)
    result["stages"]["compose"] = {"ok": ok, "seconds": secs}
    if not ok or not briefing_text:
        briefing_text = "Evening recap: day summary unavailable."
        result["ok"] = False

    # Stage 5: deliver
    def _deliver() -> bool:
        from .morning_pulse import _ledger
        _ledger(context, "evening-pulse", briefing_text[:200])
        try:
            notifier = getattr(context, "notifier", None)
            if notifier:
                notifier.publish("evening-pulse", briefing_text)
                return True
        except Exception:  # noqa: BLE001
            pass
        _log.info("evening pulse: %s", briefing_text[:500])
        return True

    ok, _, secs, attempts = _run_stage("deliver", _deliver)
    result["stages"]["deliver"] = {"ok": ok, "seconds": secs}
    result["seconds"] = datetime.now().timestamp() - started
    result["briefing"] = briefing_text
    return result
