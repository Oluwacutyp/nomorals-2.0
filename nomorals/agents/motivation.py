"""The Motivation class — real motivation, twice a day, never canned.

Owner's directive (2026-10-22): a motivation class that fires at a
**randomized** time inside 05:00–06:00 Nigeria time (Africa/Lagos,
WAT) every morning and again in an evening window, changing every
day, with REAL motivation drawn from real material — the Library and
the live net as co-equal feeds into the Teachings ledger
(:mod:`nomorals.agents.teachings`), never a canned quote rotation.

Scheduling — the trap this design avoids: the agents scheduler's
``start_jitter_s`` randomizes only a job's *first* firing, so a daily
job + jitter is random on day 1 and fixed forever.  Motivation
therefore runs on **self-rescheduling one-shots**: each firing
delivers, then rolls the *next* day's instant inside the window —
``PulseAnchor.fire`` on Africa/Lagos at a uniformly drawn minute —
and arms it.  Randomness is re-drawn daily by construction, durable
in ``schedule_jobs`` across restarts.  A daily **backstop** job
re-arms anything missing and runs the catch-up check, so a runner
that died before re-arming (or a host asleep overnight) never
silently eats a morning: catch-up delivers once on wake inside the
same Lagos day-part (morning before 11:00 WAT, evening before
23:00 WAT), and never after — a 3 PM "morning motivation" is worse
than none.

Composition: theme from the ledger's wheel (48h theme cooldown) →
one item drawn from the combined pool (30-day item cooldown, mix
floors) → the brain synthesizes a 120–220 word piece in Devon's
voice, grounded in the item's passage.  Degraded path (no brain /
offline): the passage itself is delivered as a teaching card —
passage + provenance + one framing line.  Real material either way;
there is no stored quote-of-the-day anywhere in this path.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "MORNING_JOB",
    "EVENING_JOB",
    "BACKSTOP_JOB",
    "WINDOW_DEFAULTS",
    "ensure_motivation_jobs",
    "check_motivation_catchup",
    "run_motivation",
    "motivation_backstop",
    "motivation_status",
    "motivation_history",
    "roll_instant",
    "next_fire",
    "register",
]

_log = logging.getLogger(__name__)

MORNING_JOB = "motivation morning"
EVENING_JOB = "motivation evening"
BACKSTOP_JOB = "motivation backstop"

LAGOS = "Africa/Lagos"

#: Window defaults (owner's directive for the morning; the evening
#: bounds and catch-up cutoffs are proposed defaults — one prefs
#: value each moves them: settings.motivation.<window>_window /
#: <window>_catchup_until).
WINDOW_DEFAULTS: dict[str, dict[str, str]] = {
    "morning": {"start": "05:00", "end": "06:00",
                "catchup_until": "11:00", "job": MORNING_JOB},
    "evening": {"start": "19:00", "end": "21:00",
                "catchup_until": "23:00", "job": EVENING_JOB},
}

#: One-shot execution policies: fire even if the host was off at the
#: rolled instant (the runner's day-part guard decides deliver vs
#: skip), never stack two runs, generous cap for synthesis + TTS.
_SHOT_POLICIES = {
    "missed_fire_policy": "fire_now",
    "overlap_policy": "skip",
    "run_timeout_s": 900.0,
}


# ── prefs + anchor ───────────────────────────────────────────────────

def _prefs(context: Any) -> dict[str, Any]:
    """settings.motivation as a flat dict (dict or attrs).  Never raises."""
    out: dict[str, Any] = {"enabled": True, "voice": False}
    for window, spec in WINDOW_DEFAULTS.items():
        out[f"{window}_start"] = spec["start"]
        out[f"{window}_end"] = spec["end"]
        out[f"{window}_catchup_until"] = spec["catchup_until"]
    try:
        block = getattr(getattr(context, "settings", None),
                        "motivation", None)
        if isinstance(block, dict):
            src = block
        elif block is not None and hasattr(block, "__dict__"):
            src = dict(vars(block))
        else:
            src = {}
        for key, val in src.items():
            if val is not None:
                out[str(key)] = val
        # nested per-window blocks: {morning: {start: "05:15", ...}}
        for window in WINDOW_DEFAULTS:
            sub = src.get(window)
            if isinstance(sub, dict):
                for key, val in sub.items():
                    if val is not None:
                        out[f"{window}_{key}"] = val
    except Exception:  # noqa: BLE001
        _log.debug("motivation: prefs read failed", exc_info=True)
    return out


def _anchor() -> Any:
    from ..pulses.timezone import PulseAnchor

    return PulseAnchor(LAGOS)


def _hhmm(value: str) -> tuple[int, int]:
    parts = str(value or "").strip().split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        hour, minute = 0, 0
    return max(0, min(23, hour)), max(0, min(59, minute))


def _window_cfg(prefs: dict[str, Any], window: str) -> dict[str, Any]:
    spec = WINDOW_DEFAULTS[window]
    start = _hhmm(str(prefs.get(f"{window}_start") or spec["start"]))
    end = _hhmm(str(prefs.get(f"{window}_end") or spec["end"]))
    cutoff = _hhmm(str(prefs.get(f"{window}_catchup_until")
                       or spec["catchup_until"]))
    return {
        "window": window,
        "job": spec["job"],
        "start_min": start[0] * 60 + start[1],
        "end_min": end[0] * 60 + end[1],
        "cutoff_min": cutoff[0] * 60 + cutoff[1],
    }


# ── the daily re-roll ────────────────────────────────────────────────

def roll_instant(anchor: Any, cfg: dict[str, Any], on_date: Any,
                 rng: random.Random) -> datetime:
    """One uniformly drawn fire instant inside the window, Lagos time.

    Minute grain (``fire_at_local``'s grain): the offset is drawn
    uniformly over the window's minutes, both bounds inclusive.
    """
    width = max(0, int(cfg["end_min"]) - int(cfg["start_min"]))
    offset = rng.randint(0, width) if width else 0
    total = int(cfg["start_min"]) + offset
    return anchor.fire(total // 60, total % 60, on_date=on_date)


def next_fire(anchor: Any, cfg: dict[str, Any], *, now_ts: float,
              rng: random.Random) -> datetime:
    """The next rolled instant strictly ahead of ``now_ts``.

    Checks today's Lagos date first, then tomorrow's — so a boot
    before the window arms *today's* roll, and a firing re-arms
    *tomorrow's*.
    """
    lagos_today = datetime.fromtimestamp(now_ts, anchor.tzinfo).date()
    for day in (lagos_today, lagos_today + timedelta(days=1)):
        instant = roll_instant(anchor, cfg, day, rng)
        if instant.timestamp() > now_ts + 30:
            return instant
    # Unreachable in practice (tomorrow's roll is always ahead), but
    # never arm the past: fall back to tomorrow's window start.
    day = lagos_today + timedelta(days=1)
    total = int(cfg["start_min"])
    return anchor.fire(total // 60, total % 60, on_date=day)


def _at_spec(instant: datetime) -> str:
    """Scheduler ``at`` spec for an aware UTC instant.

    The scheduler parses ``at`` specs as *host-local* wall time
    (``_parse_timestamp`` → naive ``strptime`` → ``.timestamp()``), so
    the instant is rendered in the host's own zone: the parse reads
    back the identical instant on any host timezone, at the minute
    grain ``fire_at_local`` works at.
    """
    local = instant.astimezone()
    return f"at {local.strftime('%Y-%m-%d %H:%M')}"


# ── job arming ───────────────────────────────────────────────────────

def _pending_shot(sched: Any, job_name: str, now_ts: float) -> dict | None:
    """The window's live one-shot, if one is armed and unfired."""
    try:
        for job in sched.list_jobs():
            if job.get("name") != job_name or not job.get("enabled"):
                continue
            if job.get("kind") != "at":
                continue
            nxt = job.get("next_run")
            if nxt is None or float(nxt) > now_ts - 3600:
                return job
    except Exception:  # noqa: BLE001
        _log.debug("motivation: pending-shot scan failed", exc_info=True)
    return None


def _arm_window(context: Any, prefs: dict[str, Any], window: str, *,
                now_ts: float, rng: random.Random) -> dict[str, Any]:
    from .scheduler import Scheduler

    sched = Scheduler(context)
    cfg = _window_cfg(prefs, window)
    pending = _pending_shot(sched, cfg["job"], now_ts)
    if pending is not None:
        return {"window": window, "already_scheduled": True,
                "job_id": pending.get("id"),
                "next_run": pending.get("next_run")}
    anchor = _anchor()
    instant = next_fire(anchor, cfg, now_ts=now_ts, rng=rng)
    job = sched.add(cfg["job"], _at_spec(instant), "tool",
                    {"tool": "motivation",
                     "args": {"action": "run", "window": window}},
                    timezone=LAGOS, routine=True, **_SHOT_POLICIES)
    _log.info("motivation: armed %s one-shot for %s Lagos (%s UTC)",
              window,
              instant.astimezone(anchor.tzinfo).strftime("%Y-%m-%d %H:%M"),
              instant.strftime("%Y-%m-%d %H:%M"))
    return {"window": window, "scheduled": True, "job_id": job.get("id"),
            "next_run": job.get("next_run"),
            "lagos": instant.astimezone(anchor.tzinfo).isoformat()}


def ensure_motivation_jobs(context: Any, *,
                           rng: random.Random | None = None,
                           now: float | None = None) -> dict[str, Any]:
    """Arm the one-shots + the daily backstop.  Idempotent, boot-safe.

    Mirrors ``ensure_pulse_job``: safe to call on every boot; a window
    with a live one-shot is left exactly as it is (its roll stands).
    """
    rng = rng or random.Random()
    now_ts = time.time() if now is None else now
    prefs = _prefs(context)
    if not prefs.get("enabled", True):
        return {"enabled": False}
    out: dict[str, Any] = {"enabled": True, "windows": {}}
    for window in WINDOW_DEFAULTS:
        try:
            out["windows"][window] = _arm_window(
                context, prefs, window, now_ts=now_ts, rng=rng)
        except Exception as exc:  # noqa: BLE001
            out["windows"][window] = {
                "error": f"{type(exc).__name__}: {exc}"}
    # the backstop: daily 06:30 Lagos — re-arms anything missing and
    # runs catch-up, so a broken chain still lands the morning piece
    # (via catch-up, before 11:00) instead of eating it silently.
    try:
        from .scheduler import Scheduler

        sched = Scheduler(context)
        have = {j.get("name") for j in sched.list_jobs()
                if j.get("enabled")}
        if BACKSTOP_JOB in have:
            out["backstop"] = {"already_scheduled": True}
        else:
            job = sched.add(
                BACKSTOP_JOB, "daily 06:30", "tool",
                {"tool": "motivation", "args": {"action": "backstop"}},
                timezone=LAGOS, routine=True,
                missed_fire_policy="fire_now", overlap_policy="skip",
                run_timeout_s=900.0)
            out["backstop"] = {"scheduled": True, "job_id": job.get("id")}
    except Exception as exc:  # noqa: BLE001
        out["backstop"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def motivation_backstop(context: Any, *, now: float | None = None,
                        rng: random.Random | None = None) -> dict[str, Any]:
    """The daily backstop's whole job: re-arm, then catch up."""
    return {
        "ensure": ensure_motivation_jobs(context, rng=rng, now=now),
        "catchup": check_motivation_catchup(context, now=now, rng=rng),
    }


# ── delivery marker (catch-up + status observability) ───────────────

def _marker_path(context: Any) -> Path | None:
    try:
        ws = getattr(getattr(context, "settings", None),
                     "workspace_dir", "")
        if not ws:
            return None
        return Path(ws) / "motivation_last_run.json"
    except Exception:  # noqa: BLE001
        return None


def _read_marker(context: Any) -> dict[str, Any]:
    try:
        p = _marker_path(context)
        if p is not None and p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _write_marker(context: Any, window: str, entry: dict[str, Any]) -> None:
    try:
        p = _marker_path(context)
        if p is None:
            return
        data = _read_marker(context)
        data[window] = entry
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001
        _log.debug("motivation: marker write failed", exc_info=True)


# ── composition ──────────────────────────────────────────────────────

_FRAMING = {
    "morning": "Start the day on this:",
    "evening": "Close the day on this:",
}


def _synthesize(context: Any, item: dict[str, Any], window: str) -> str | None:
    """The brain's piece: 120–220 words, Devon's voice, passage-spined.

    Returns None when no brain answers or the answer is off-size —
    the caller degrades to the teaching card, never to canned text.
    """
    try:
        from ..llm.brain import brain_for

        brain = brain_for(context)
        if brain is None:
            return None
        src = item.get("source") or {}
        passage = str(src.get("passage") or "").strip()
        label = str(src.get("label") or "").strip()
        theme = str(item.get("theme") or "").strip()
        when = "morning" if window == "morning" else "evening"
        prompt = (
            f"Write a {when} motivation for your owner, in your own "
            "voice — direct, warm, unpatronizing, specific over "
            "abstract, no filler, no greeting fluff, no sign-off. "
            "120 to 220 words. The passage below is the spine of the "
            "piece: build on what it actually says, and connect it to "
            "a real life of building, goals and discipline without "
            "inventing facts about him.\n\n"
            f"Theme: {theme or 'the day'}\n"
            f"Passage (from {label or 'her library'}):\n{passage}"
        )
        resp = brain.chat([{"role": "user", "content": prompt}])
        text = resp.text if hasattr(resp, "text") else str(resp)
        text = (text or "").strip()
        words = len(text.split())
        if not (70 <= words <= 320):
            _log.info("motivation: synthesis off-size (%d words); "
                      "degrading to card", words)
            return None
        return text
    except Exception:  # noqa: BLE001 - no brain == degraded card
        _log.debug("motivation: synthesis unavailable", exc_info=True)
        return None


def _card(item: dict[str, Any], window: str) -> str:
    """The degraded path: the passage itself, as a teaching card.

    Passage + provenance + one framing line (the answer_card shape).
    The teaching is real material from the ledger — never a stored
    quote rotation.
    """
    src = item.get("source") or {}
    passage = str(src.get("passage") or "").strip()
    label = str(src.get("label") or "").strip()
    title = str(item.get("title") or "").strip()
    lines = [_FRAMING.get(window, _FRAMING["morning"])]
    if title and title.lower() not in passage.lower():
        lines.append(f"**{title}**")
    lines.append(f"“{passage}”")
    if label:
        lines.append(f"— {label}")
    ref = str(src.get("ref") or "")
    if src.get("type") in ("web", "news") and ref.startswith("http"):
        lines.append(ref)
    return "\n".join(lines)


def _piece_body(item: dict[str, Any], window: str,
                synthesized: str | None) -> str:
    """Final body: the synthesis (or card) + a provenance footer."""
    src = item.get("source") or {}
    label = str(src.get("label") or "").strip()
    if synthesized:
        body = synthesized
        if label:
            kind = src.get("type")
            origin = ("From her library" if kind in ("wisdom", "book")
                      else "Found today" if kind in ("web", "news")
                      else "From your past" if kind == "memory"
                      else "Source")
            body += f"\n\n— {origin}: {label}"
            ref = str(src.get("ref") or "")
            if kind in ("web", "news") and ref.startswith("http"):
                body += f"\n{ref}"
        return body
    return _card(item, window)


def _send_voice(context: Any, text: str) -> bool:
    """Best-effort voice note of the piece (prefs-gated, off default).

    Reuses the pulse's machinery: UniversalTTS → ogg → send_voice_note
    on the owner's first reachable chat.  Never raises.
    """
    try:
        from ..social.chat.gateway import resolve_owner_targets
        from ..social.chat.voice_transport import send_voice_note
        from ..voice.tts import UniversalTTS
        from .morning_pulse import _to_voice_ogg
        from .notifier import resolve_gateway

        gateway = resolve_gateway(context)
        if gateway is None:
            return False
        engine = UniversalTTS(backend="system", audience="private")
        out = engine.speak(text[:1200])
        wav = (out or {}).get("path") or ""
        if not wav:
            return False
        ogg, _mime = _to_voice_ogg(wav)
        targets = resolve_owner_targets(context) or []
        for plat, ref in targets:
            adapter = gateway._adapter_for(plat)
            if adapter is None or getattr(gateway, "dry_run", False):
                continue
            delivery = send_voice_note(adapter, ref, ogg)
            if getattr(delivery, "ok", False):
                return True
        return False
    except Exception:  # noqa: BLE001
        _log.debug("motivation: voice note failed", exc_info=True)
        return False


# ── the run ──────────────────────────────────────────────────────────

def _ledger(context: Any, kind: str, summary: str, *, ok: bool = True,
            metadata: dict[str, Any] | None = None) -> None:
    """Journal to the unified autonomy ledger.  Never raises."""
    try:
        from .autonomy_ledger import record_ledger

        record_ledger(context, "motivation", kind, "motivation", summary,
                      ok=ok, metadata=metadata or {})
    except Exception:  # noqa: BLE001
        _log.debug("motivation: ledger write failed", exc_info=True)


def run_motivation(context: Any, window: str = "morning", *,
                   force: bool = False, now: float | None = None,
                   rng: random.Random | None = None) -> dict[str, Any]:
    """One window's firing: compose from the ledger, deliver, re-arm.

    The re-arm is in a ``finally``: the chain continues even when a
    firing fails, and the daily backstop is the second net under it.
    ``force`` (owner's /motivation now) skips the day-part guards and
    the already-delivered check, but still marks served — a forced
    piece spends its item/theme like any other.
    """
    window = (window or "morning").strip().lower()
    if window not in WINDOW_DEFAULTS:
        return {"ok": False, "error": f"unknown window {window!r}"}
    rng = rng or random.Random()
    now_ts = time.time() if now is None else now
    prefs = _prefs(context)
    anchor = _anchor()
    cfg = _window_cfg(prefs, window)
    lagos_now = datetime.fromtimestamp(now_ts, anchor.tzinfo)
    today = lagos_now.date().isoformat()
    result: dict[str, Any] = {"ok": True, "window": window,
                              "lagos_date": today, "delivered": False}
    try:
        if not prefs.get("enabled", True) and not force:
            result["reason"] = "motivation disabled"
            return result
        minute_of_day = lagos_now.hour * 60 + lagos_now.minute
        if not force:
            if minute_of_day < cfg["start_min"]:
                result["reason"] = "before window"
                return result
            if minute_of_day > cfg["cutoff_min"]:
                result["reason"] = "past catch-up cutoff — skipped"
                _ledger(context, window, result["reason"], ok=True)
                return result
            marker = _read_marker(context).get(window) or {}
            if marker.get("lagos_date") == today and marker.get("delivered"):
                result["reason"] = "already delivered today"
                return result

        from .teachings import TeachingsLedger

        ledger = TeachingsLedger(context)
        result["stocked"] = ledger.stock(now=now_ts)
        item = ledger.pick(surface=f"motivation-{window}", now=now_ts,
                           rng=rng)
        if item is None:
            result["reason"] = "no material in the ledger yet"
            _ledger(context, window, result["reason"], ok=False)
            return result

        synthesized = _synthesize(context, item, window)
        body = _piece_body(item, window, synthesized)
        theme = str(item.get("theme") or "").strip()
        emoji = "🌅" if window == "morning" else "🌙"
        title = f"{emoji} {window.capitalize()} motivation"
        if theme:
            title += f" — {theme}"

        delivered = False
        try:
            notifier = getattr(context, "notifier", None)
            if notifier is None:
                from .notifier import Notifier

                notifier = Notifier(context)
            res = notifier.publish("motivation", title, body, force=True)
            delivered = bool(res.get("delivered")) if isinstance(
                res, dict) else True
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
            _ledger(context, window, f"delivery failed: {exc}", ok=False)
            return result
        result["delivered"] = delivered
        result["degraded_card"] = synthesized is None
        result["theme"] = theme
        result["source_type"] = (item.get("source") or {}).get("type")

        served = ledger.mark_served(
            item, f"motivation-{window}", body=body, kind="motivation",
            now=now_ts)
        result["item_id"] = served.get("id")
        _write_marker(context, window, {
            "lagos_date": today, "ts": now_ts, "delivered": delivered,
            "item_id": served.get("id"), "theme": theme,
            "degraded_card": synthesized is None,
        })
        _ledger(context, window,
                f"{window} motivation delivered ({theme or 'no theme'}, "
                f"{result['source_type'] or '?'})",
                metadata={"item_id": served.get("id"),
                          "degraded_card": synthesized is None})
        if prefs.get("voice") and delivered:
            result["voice_sent"] = _send_voice(context, body)
        return result
    finally:
        # The chain must survive this firing, whatever happened above
        # (unless the owner has switched the class off entirely).
        if prefs.get("enabled", True):
            try:
                _arm_window(context, prefs, window, now_ts=now_ts, rng=rng)
            except Exception:  # noqa: BLE001
                _log.debug("motivation: re-arm failed (backstop covers)",
                           exc_info=True)


def check_motivation_catchup(context: Any, *, now: float | None = None,
                             rng: random.Random | None = None
                             ) -> dict[str, Any]:
    """Boot-time catch-up, mirroring ``check_pulse_catchup``.

    For each window: if the host is awake inside the same Lagos
    day-part after the window passed (before the cutoff) and no piece
    has delivered today, run it once now.  After the cutoff: silent.
    Never raises.
    """
    out: dict[str, Any] = {}
    now_ts = time.time() if now is None else now
    prefs = _prefs(context)
    anchor = _anchor()
    lagos_now = datetime.fromtimestamp(now_ts, anchor.tzinfo)
    today = lagos_now.date().isoformat()
    minute_of_day = lagos_now.hour * 60 + lagos_now.minute
    for window in WINDOW_DEFAULTS:
        try:
            if not prefs.get("enabled", True):
                out[window] = {"catchup": False, "reason": "disabled"}
                continue
            cfg = _window_cfg(prefs, window)
            if minute_of_day <= cfg["end_min"]:
                out[window] = {"catchup": False,
                               "reason": "window not passed"}
                continue
            if minute_of_day > cfg["cutoff_min"]:
                out[window] = {"catchup": False,
                               "reason": "past catch-up cutoff"}
                continue
            marker = _read_marker(context).get(window) or {}
            if marker.get("lagos_date") == today and marker.get("delivered"):
                out[window] = {"catchup": False,
                               "reason": "already delivered"}
                continue
            _log.info("motivation: catch-up firing for %s", window)
            res = run_motivation(context, window, now=now_ts, rng=rng)
            res["catchup"] = True
            out[window] = res
        except Exception as exc:  # noqa: BLE001
            out[window] = {"catchup": False,
                           "reason": f"{type(exc).__name__}: {exc}"}
    return out


# ── surfaces ─────────────────────────────────────────────────────────

def motivation_status(context: Any, *, now: float | None = None
                      ) -> dict[str, Any]:
    """Schedules, last runs, and ledger state for the owner surfaces."""
    now_ts = time.time() if now is None else now
    prefs = _prefs(context)
    anchor = _anchor()
    marker = _read_marker(context)
    out: dict[str, Any] = {
        "enabled": bool(prefs.get("enabled", True)),
        "voice": bool(prefs.get("voice")),
        "timezone": LAGOS,
        "windows": {},
    }
    jobs: dict[str, Any] = {}
    try:
        from .scheduler import Scheduler

        for job in Scheduler(context).list_jobs():
            jobs[job.get("name")] = job
    except Exception:  # noqa: BLE001
        jobs = {}
    for window, spec in WINDOW_DEFAULTS.items():
        cfg = _window_cfg(prefs, window)
        job = jobs.get(spec["job"]) or {}
        nxt = job.get("next_run")
        out["windows"][window] = {
            "window": (f"{cfg['start_min'] // 60:02d}:"
                       f"{cfg['start_min'] % 60:02d}–"
                       f"{cfg['end_min'] // 60:02d}:"
                       f"{cfg['end_min'] % 60:02d} Lagos"),
            "job_id": job.get("id"),
            "next_run_utc": (datetime.fromtimestamp(
                float(nxt), timezone.utc).isoformat() if nxt else None),
            "next_run_lagos": (datetime.fromtimestamp(
                float(nxt), anchor.tzinfo).strftime("%Y-%m-%d %H:%M")
                if nxt else None),
            "last": marker.get(window),
        }
    backstop = jobs.get(BACKSTOP_JOB) or {}
    out["backstop"] = {"job_id": backstop.get("id"),
                       "next_run": backstop.get("next_run")}
    try:
        from .teachings import TeachingsLedger

        out["ledger"] = TeachingsLedger(context).stats(now=now_ts)
    except Exception:  # noqa: BLE001
        out["ledger"] = {}
    return out


def motivation_history(context: Any, *, limit: int = 10
                       ) -> list[dict[str, Any]]:
    """Recently served motivation pieces (title/theme/source/when)."""
    try:
        from .teachings import TeachingsLedger

        items = TeachingsLedger(context).served_items(
            surface_prefix="motivation")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for it in items[: max(1, int(limit))]:
        served = it.get("served") or []
        out.append({
            "id": it.get("id"), "title": it.get("title"),
            "theme": it.get("theme"),
            "source": (it.get("source") or {}).get("label"),
            "source_type": (it.get("source") or {}).get("type"),
            "surfaces": [s.get("surface") for s in served],
            "last_served_at": max(
                (float(s.get("at") or 0) for s in served), default=0.0),
        })
    return out


def _current_window(context: Any, now_ts: float) -> str:
    """The window a bare 'now' belongs to, by Lagos day-part."""
    anchor = _anchor()
    lagos_now = datetime.fromtimestamp(now_ts, anchor.tzinfo)
    return "morning" if lagos_now.hour < 12 else "evening"


def register(registry: Any) -> None:
    """Register the ``motivation`` tool (status | now | history | run).

    The scheduler's one-shots call ``action=run``; the owner's
    ``/motivation`` rides the same tool: ``now`` forces a piece,
    ``status`` shows the armed rolls + last runs, ``history`` lists
    what she has served, ``ensure``/``backstop`` re-arm the chain.
    """
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "motivation",
        description=(
            "Daily motivation in Devon's voice, drawn from her Library "
            "and the live net through the Teachings ledger — randomized "
            "fire time inside 05:00–06:00 Lagos every morning and an "
            "evening window, re-rolled daily, never a canned rotation. "
            "action=status (armed rolls + ledger) | now (deliver a "
            "piece immediately) | history (recent pieces) | run "
            "(scheduler entry) | backstop (re-arm + catch-up). "
            "window=morning|evening (default: current Lagos day-part)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — status|now|history|run|backstop|ensure",
            "window": "str — morning|evening (optional)",
        },
    )
    def motivation(*, action: str = "", window: str = "") -> dict[str, Any]:
        act = (action or "status").strip().lower()
        win = (window or "").strip().lower()
        if act == "status":
            return {"ok": True, **motivation_status(context)}
        if act == "history":
            return {"ok": True, "history": motivation_history(context)}
        if act == "ensure":
            return {"ok": True, **ensure_motivation_jobs(context)}
        if act == "backstop":
            return {"ok": True, **motivation_backstop(context)}
        if act in ("run", "now"):
            if win not in WINDOW_DEFAULTS:
                win = _current_window(context, time.time())
            return run_motivation(context, win, force=(act == "now"))
        return {"ok": False, "error": f"unknown action {act!r}"}
