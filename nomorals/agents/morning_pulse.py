"""Morning pulse — the autonomous daily briefing with voice.

The user's priority demonstration of system autonomy: news → briefing →
voice → delivery, all automatic, no manual commands.  Fires daily at
23:00 America/Denver (the phone's timezone, user's explicit choice).

Pipeline (``run_pulse``):
1. **News radar** — ``NewsAgent.run()`` fetches fresh items so the pulse
   never reports stale news.
2. **Compose** — ``BriefingComposer`` builds the text briefing; the LLM
   then writes a conversational two-host script from it.
3. **Voice** — ``UniversalTTS`` synthesizes each host turn; turns are
   concatenated into one audio file, converted to OGG for the voice note.
4. **Deliver** — text via ``Notifier.publish``, audio via the live gateway
   as a Telegram voice bubble (``voice_note=True``).

Every stage degrades gracefully: news failure → pulse still goes out with
the briefing; voice failure → text still delivers; total failure → a short
fallback message, never silence.  Never raises.

Scheduler: ``ensure_pulse_job`` registers one durable ``morning-pulse``
job (idempotent by name), ``daily 23:00`` in America/Denver.  Safe to call
on every boot.  Wired into ``PartnerRuntime`` alongside
``ensure_briefing_job``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from ..llm.brain import brain_for

_log = logging.getLogger(__name__)

PULSE_JOB_NAME = "morning-pulse"
DEFAULT_PULSE_TIME = "23:00"
#: The phone's timezone — user's explicit choice (2026-10-09).
PULSE_TIMEZONE = "America/Denver"

#: Two hosts for the conversational format (cf. the Setorch reference's
#: "Brian & Emma").  espeak voice variants: plain + female variant.
HOST_1_NAME = "Devon"
HOST_1_VOICE = ""          # default espeak voice
HOST_2_NAME = "Pulse"
HOST_2_VOICE = "en+f3"    # espeak female variant


# ── prefs ────────────────────────────────────────────────────────────────

def _prefs(context: Any) -> dict[str, Any]:
    """Pulse prefs: settings.pulse namespace, else JSON sidecar."""
    out: dict[str, Any] = {"time": DEFAULT_PULSE_TIME,
                           "timezone": PULSE_TIMEZONE,
                           "enabled": True}
    bs = getattr(getattr(context, "settings", None), "pulse", None)
    if bs is not None and hasattr(bs, "__dict__"):
        for k in out:
            v = getattr(bs, k, None)
            if k == "enabled":
                # a falsy switch is a real setting, not a missing one
                if v is not None:
                    out[k] = bool(v)
            elif v:
                out[k] = v
        return out
    try:
        sidecar = (Path(context.settings.workspace_dir)
                   / "pulse_prefs.json")
        if sidecar.is_file():
            out.update(json.loads(sidecar.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return out


def pulse_time(context: Any) -> str:
    """Daily pulse time HH:MM (default 23:00)."""
    return str(_prefs(context).get("time") or DEFAULT_PULSE_TIME)


def pulse_timezone(context: Any) -> str:
    """Pulse timezone (default America/Denver — the phone)."""
    return str(_prefs(context).get("timezone") or PULSE_TIMEZONE)


def pulse_enabled(context: Any) -> bool:
    return bool(_prefs(context).get("enabled", True))


# ── last-run marker (catch-up + "did it deliver?" observability) ──

def _marker_path(context: Any) -> Path | None:
    """Sidecar JSON: the last pulse run's outcome.  Never raises."""
    try:
        ws = getattr(getattr(context, "settings", None), "workspace_dir", "")
        if not ws:
            return None
        return Path(ws) / "pulse_last_run.json"
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


def _write_marker(context: Any, result: dict[str, Any]) -> None:
    try:
        p = _marker_path(context)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "ts": time.time(),
            "stages": result.get("stages", []),
            "delivered_text": bool(result.get("delivered_text")),
            "delivered_audio": bool(result.get("delivered_audio")),
            "elapsed_s": result.get("elapsed_s", 0),
        }), encoding="utf-8")
    except Exception:  # noqa: BLE001
        _log.debug("morning-pulse: marker write failed", exc_info=True)


def last_pulse_run(context: Any) -> dict[str, Any]:
    """When the pulse last ran and whether it reached the owner."""
    return _read_marker(context)


def _pulse_due_today(context: Any) -> float:
    """Epoch of today's pulse time in the pulse timezone (0 if the
    timezone math is unavailable).  Never raises."""
    try:
        from datetime import datetime
        from ..core.tz import safe_zoneinfo
        tz = safe_zoneinfo(pulse_timezone(context))
        hh, mm = (pulse_time(context).split(":") + ["0"])[:2]
        return datetime.now(tz).replace(hour=int(hh), minute=int(mm),
                                        second=0, microsecond=0).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def check_pulse_catchup(context: Any) -> dict[str, Any]:
    """Boot-time catch-up for the pulse (mirrors briefing.check_catchup).

    If the pulse is enabled, today's pulse time has passed, and no pulse
    has *delivered* since then, run it once now — never a backlog, never
    a duplicate.  This covers the case the scheduler's ``fire_now``
    policy misses: a pulse that fired but failed to deliver.  Safe to
    call on every boot.  Never raises.
    """
    try:
        if not pulse_enabled(context):
            return {"catchup": False, "reason": "pulse disabled"}
        due = _pulse_due_today(context)
        now = time.time()
        if not due or now < due:
            return {"catchup": False, "reason": "pulse time not reached"}
        marker = _read_marker(context)
        last_ts = float(marker.get("ts") or 0)
        if last_ts >= due and (marker.get("delivered_text")
                               or marker.get("delivered_audio")):
            return {"catchup": False, "reason": "already delivered"}
        _log.info("morning-pulse: catch-up firing (last delivered run "
                  "before today's %s)", pulse_time(context))
        result = run_pulse(context)
        result["catchup"] = True
        return result
    except Exception as exc:  # noqa: BLE001
        _log.warning("morning-pulse catch-up failed: %s", exc)
        return {"catchup": False, "reason": f"{type(exc).__name__}: {exc}"}


# ── scheduler registration ───────────────────────────────────────────────

#: Execution policies for the pulse job: it must fire even if the phone
#: was off at 23:00 (fire_now), must never stack two pulses
#: (overlap=skip), and gets a generous wall-clock cap for TTS.
PULSE_POLICIES = {
    "missed_fire_policy": "fire_now",
    "overlap_policy": "skip",
    "run_timeout_s": 1800.0,
}


def _pulse_spec_matches(job_spec: str, want: str) -> bool:
    """True when the stored job spec is the wanted pulse time.

    The stored spec may carry the timezone (``"23:00 America/Denver"``);
    only the wall-clock part is compared.
    """
    return bool(job_spec) and job_spec.split()[0] == want


def ensure_pulse_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``morning-pulse`` job (idempotent).

    Runs ``daily HH:MM`` at the configured pulse time (default 23:00) in
    America/Denver, with ``missed_fire_policy=fire_now`` (a phone that
    was off at 23:00 still gets its pulse on next boot),
    ``overlap_policy=skip`` (pulses never stack), and a 30-minute
    run cap for TTS.  Safe to call on every boot.  If the owner changed
    the time — or the policies drifted — the job is replaced.
    """
    from .scheduler import Scheduler

    sched = Scheduler(context)
    want = pulse_time(context)
    tz = pulse_timezone(context)
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == PULSE_JOB_NAME]
    except Exception:  # noqa: BLE001 — scheduler table may not exist yet
        have = []
    if have:
        job0 = have[0]
        same_policies = all(
            job0.get(k) == v for k, v in PULSE_POLICIES.items())
        if _pulse_spec_matches(job0.get("spec") or "", want) and same_policies:
            return {"name": PULSE_JOB_NAME, "already_scheduled": True,
                    "job_id": job0.get("id")}
    for j in have:  # stale time or policies → replace
        try:
            sched.remove(j["id"])
        except Exception:  # noqa: BLE001
            pass
    job = sched.add(PULSE_JOB_NAME, f"daily {want}", "tool",
                    {"tool": "pulse", "args": {"action": "run"}},
                    timezone=tz, **PULSE_POLICIES)
    _log.info("scheduled morning-pulse job: %s (daily %s %s)",
              PULSE_JOB_NAME, want, tz)
    return {"name": PULSE_JOB_NAME, "scheduled": True,
            "job_id": job.get("id")}


# ── stage runner ─────────────────────────────────────────────────────────

def _run_stage(name: str, fn: Any, *, attempts: int = 2,
               base_delay_s: float = 5.0) -> tuple[bool, Any, float, int]:
    """Run one pipeline stage with exponential-backoff retries.

    Returns ``(ok, value, seconds, attempts_used)``.  Never raises — a
    stage that exhausts its attempts reports ``ok=False`` and the
    pipeline degrades gracefully to the next stage.
    """
    started = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            value = fn()
            return True, value, time.time() - started, attempt
        except Exception as exc:  # noqa: BLE001 - a stage failure is a result
            last_exc = exc
            _log.warning("morning-pulse: stage %s attempt %d/%d failed: %s",
                         name, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(min(60.0, base_delay_s * (2.0 ** (attempt - 1))))
    _log.error("morning-pulse: stage %s failed after %d attempt(s): %s",
               name, attempts, last_exc)
    return False, None, time.time() - started, attempts


def _ledger(context: Any, kind: str, summary: str, *,
            cost_seconds: float = 0.0, ok: bool = True,
            learned: str = "",
            metadata: dict[str, Any] | None = None) -> None:
    """Journal to the unified autonomy ledger.  Never raises."""
    try:
        from .autonomy_ledger import record_ledger

        record_ledger(context, "pulse", kind, PULSE_JOB_NAME, summary,
                      cost_seconds=cost_seconds, ok=ok, learned=learned,
                      metadata=metadata)
    except Exception:  # noqa: BLE001
        _log.debug("morning-pulse ledger write failed", exc_info=True)


def _emit_bus(data: dict[str, Any]) -> None:
    """Publish ``pulse.finished`` for other systems.  Fail-open."""
    try:
        from ..core.events import Event, global_bus

        global_bus.publish(Event(topic="pulse.finished", data=data,
                                 source="nomorals.agents.morning_pulse"))
    except Exception:  # noqa: BLE001
        _log.debug("morning-pulse bus publish failed", exc_info=True)


# ── the pipeline ─────────────────────────────────────────────────────────

def _fetch_news(context: Any) -> list[dict[str, Any]]:
    """News radar stage.  Returns the items (may be empty); raises only
    on unexpected breakage — the stage runner converts that to a
    graceful degradation."""
    from .news import NewsAgent

    agent = NewsAgent(context)
    report = agent.run()
    items = agent.recent(limit=12) if report.get("ok") else []
    _log.info("morning-pulse: news radar got %d items", len(items))
    return items


def _compose_briefing(context: Any) -> str:
    """Briefing compose stage.  Returns the text (possibly a fallback)."""
    from .morning_briefing import BriefingComposer
    from datetime import datetime
    from ..core.tz import safe_zoneinfo

    composer = BriefingComposer()
    tz = safe_zoneinfo(pulse_timezone(context))
    date = datetime.now(tz).strftime("%Y-%m-%d")
    briefing = composer.compose(context, date)
    return briefing.render_text()


def run_pulse(context: Any) -> dict[str, Any]:
    """Run the full morning-pulse pipeline.  Never raises.

    Every stage runs through the stage runner (exponential-backoff
    retries), lands a ledger entry with its cost, and degrades
    gracefully: news failure → pulse still goes out with the briefing;
    voice failure → text still delivers; total failure → a short
    fallback message, never silence.

    Returns {ok, text, audio_path, delivered_text, delivered_audio,
    stages} — stages names each completed step for observability.
    ``ok`` means the pipeline ran to completion (it never raises);
    whether anything reached the owner is in ``delivered_text`` /
    ``delivered_audio`` (and the ledger/bus event).
    """
    stages: list[str] = []
    started = time.time()

    # ── 1. news radar ────────────────────────────────────────────────
    ok, news_items, secs, att = _run_stage(
        "news", lambda: _fetch_news(context))
    news_items = news_items or []
    _ledger(context, "stage",
            f"news radar: {'ok' if ok else 'FAILED'} "
            f"({len(news_items)} items, {att} attempt(s))",
            cost_seconds=secs, ok=ok,
            metadata={"stage": "news", "items": len(news_items),
                      "attempts": att})
    if ok:
        stages.append("news")

    # ── 2. compose ───────────────────────────────────────────────────
    ok, briefing_text, secs, att = _run_stage(
        "compose", lambda: _compose_briefing(context))
    briefing_text = briefing_text or "(briefing unavailable tonight)"
    _ledger(context, "stage",
            f"compose: {'ok' if ok else 'fallback'} ({att} attempt(s))",
            cost_seconds=secs, ok=True,  # fallback text still delivers
            metadata={"stage": "compose", "attempts": att,
                      "fallback": not ok})
    stages.append("compose")

    # ── 3. script (two-host conversational) ──────────────────────────
    script = _write_script(context, briefing_text, news_items)
    stages.append("script")

    # ── 4. voice ─────────────────────────────────────────────────────
    ok, audio_path, secs, att = _run_stage(
        "voice", lambda: _synthesize(context, script))
    audio_path = audio_path or ""
    _ledger(context, "stage",
            f"voice: {'ok' if ok and audio_path else 'FAILED'} "
            f"({att} attempt(s))",
            cost_seconds=secs, ok=bool(audio_path),
            learned="" if audio_path else "TTS failed — text-only delivery",
            metadata={"stage": "voice", "attempts": att})
    if audio_path:
        stages.append("voice")

    # ── 5. deliver ───────────────────────────────────────────────────
    text_ok = _deliver_text(context, briefing_text)
    audio_ok = _deliver_audio(context, audio_path) if audio_path else False
    stages.append("deliver")

    elapsed = round(time.time() - started, 1)
    delivered = text_ok or audio_ok
    _log.info("morning-pulse done in %ss: stages=%s text=%s audio=%s",
              elapsed, stages, text_ok, audio_ok)
    _ledger(context, "run",
            f"morning pulse: stages={','.join(stages)} "
            f"text={'delivered' if text_ok else 'FAILED'} "
            f"audio={'delivered' if audio_ok else 'n/a' if not audio_path else 'FAILED'}",
            cost_seconds=elapsed, ok=delivered,
            learned="" if delivered else "nothing delivered — check gateway",
            metadata={"stages": stages, "delivered_text": text_ok,
                      "delivered_audio": audio_ok})
    _emit_bus({
        "stages": stages,
        "delivered_text": text_ok,
        "delivered_audio": audio_ok,
        "elapsed_s": elapsed,
        "ok": delivered,
    })
    result = {"ok": True, "stages": stages, "text": briefing_text,
                "audio_path": audio_path, "delivered_text": text_ok,
                "delivered_audio": audio_ok, "elapsed_s": elapsed}
    _write_marker(context, result)
    return result


def _write_script(context: Any, briefing_text: str,
                  news_items: list[dict[str, Any]]) -> str:
    """Write the two-host conversational script via the LLM.

    Falls back to a single-voice template when no brain is available.
    Returns "Host: line" format, one turn per line.
    """
    # Try the LLM first.
    try:
        router = getattr(context, "router", None)
        if router is not None:
            from ..llm.base import Message
            headlines = "\n".join(
                f"- {n.get('title', '')}" for n in news_items[:5])
            prompt = (
                f"You are writing a 90-second morning news pulse for a "
                f"personal AI's daily briefing. Two hosts: {HOST_1_NAME} "
                f"(warm, direct) and {HOST_2_NAME} (sharp, playful). "
                f"Write ONLY dialogue lines, one per line, in this exact "
                f"format:\n{HOST_1_NAME}: <line>\n{HOST_2_NAME}: <line>\n\n"
                f"Cover these in conversational order — no bullet lists, "
                f"no headlines read verbatim, talk like humans:\n"
                f"{briefing_text[:1500]}\n\n"
                f"Top stories:\n{headlines}\n\n"
                f"Rules: every line a complete spoken sentence. "
                f"Numbers written out (twenty-three, not 23). "
                f"No URLs, no markup, no stage directions. "
                f"Keep it under 200 words total."
            )
            response = brain_for(context).chat([Message(role="user", content=prompt)], task_kind="creative")
            text = (getattr(response, "text", "") or "").strip()
            # accept the LLM script when it actually looks like dialogue
            # (Speaker: line turns) — a missing host name alone shouldn't
            # discard a good script.
            dialogue_lines = [ln for ln in text.splitlines()
                              if ":" in ln and ln.partition(":")[2].strip()]
            if len(dialogue_lines) >= 2 or HOST_1_NAME in text:
                _log.info("morning-pulse: LLM wrote %d-char script",
                          len(text))
                return text
            _log.info("morning-pulse: LLM reply wasn't dialogue-shaped; "
                      "using template fallback")
    except Exception as exc:  # noqa: BLE001
        _log.warning("morning-pulse: LLM script failed: %s", exc)

    # Fallback: single-host template from the briefing.
    _log.info("morning-pulse: using fallback single-host script")
    body = " ".join(briefing_text.split())[:800]
    return f"{HOST_1_NAME}: Good evening. Here's your pulse. {body}"


def _synthesize(context: Any, script: str) -> str:
    """Synthesize the script to a WAV file.  Returns path or ''.

    Two-host format: each "Host: line" turn is synthesized with that
    host's voice, then concatenated.  Single voice when only one host
    appears.
    """
    from ..voice.tts import UniversalTTS

    turns: list[tuple[str, str]] = []
    for line in script.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        speaker, _, text = line.partition(":")
        speaker, text = speaker.strip(), text.strip()
        if text:
            turns.append((speaker, text))
    if not turns:
        return ""

    workdir = Path(os.environ.get("TMPDIR", "/tmp")) / "morning-pulse"
    workdir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    parts: list[str] = []

    for i, (speaker, text) in enumerate(turns):
        # Host 2 gets the alternate voice; everyone else the default.
        voice_lang = (HOST_2_VOICE if speaker == HOST_2_NAME
                      else HOST_1_VOICE)
        out = str(workdir / f"turn-{stamp}-{i}.wav")
        try:
            # SystemTTSBackend honors SYSTEM_TTS_LANG for espeak -v.
            if voice_lang:
                os.environ["SYSTEM_TTS_LANG"] = voice_lang
            elif "SYSTEM_TTS_LANG" in os.environ:
                del os.environ["SYSTEM_TTS_LANG"]
            engine = UniversalTTS(backend="system", audience="private")
            res = engine.speak(text, out_path=out)
            path = str(res.get("path", "") or "")
            if path and os.path.isfile(path):
                parts.append(path)
        except Exception as exc:  # noqa: BLE001 — one bad turn skips
            _log.warning("morning-pulse: turn %d TTS failed: %s", i, exc)
    # restore env
    if "SYSTEM_TTS_LANG" in os.environ:
        del os.environ["SYSTEM_TTS_LANG"]

    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    # Concatenate WAVs.
    import wave
    out_path = str(workdir / f"pulse-{stamp}.wav")
    try:
        with wave.open(parts[0], "rb") as w0:
            params = w0.getparams()
        with wave.open(out_path, "wb") as out:
            out.setparams(params)
            for p in parts:
                with wave.open(p, "rb") as w:
                    out.writeframes(w.readframes(w.getnframes()))
    except Exception as exc:  # noqa: BLE001
        _log.warning("morning-pulse: WAV concat failed: %s", exc)
        return parts[0]
    _log.info("morning-pulse: synthesized %d turns → %s",
              len(parts), out_path)
    return out_path


def _deliver_text(context: Any, text: str) -> bool:
    """Deliver the text pulse via Notifier.  Never raises."""
    from .notifier import Notifier
    title = "🌙 Morning pulse"
    try:
        notifier = Notifier(context)
        res = notifier.publish("pulse", title, text[:3500], force=True)
        delivered = bool(res.get("delivered"))
        if not delivered:
            _log.warning(
                "morning-pulse: text not delivered — gateway=%s",
                _delivery_diagnosis(context))
        return delivered
    except Exception as exc:  # noqa: BLE001
        _log.warning("morning-pulse: text delivery failed: %s", exc)
        return False


def _deliver_audio(context: Any, audio_path: str) -> bool:
    """Deliver the voice note via the live gateway.  Never raises."""
    if not audio_path or not os.path.isfile(audio_path):
        return False
    try:
        from ..social.chat.base import ChatRef, MediaRef
        from .notifier import resolve_gateway
        gateway = resolve_gateway(context)
        if gateway is None:
            _log.warning("morning-pulse: audio not delivered — %s",
                         _delivery_diagnosis(context))
            return False
        # Telegram voice bubbles want OGG/Opus; try conversion, else WAV.
        send_path, mime = _to_voice_ogg(audio_path)
        media = MediaRef(path=send_path, kind="audio",
                         name="morning-pulse.ogg", mime=mime,
                         voice_note=True)
        settings = getattr(context, "settings", None)
        partner = getattr(settings, "partner", None)
        raw = getattr(partner, "owner_chats", "") or ""
        reached = 0
        for key in raw.split(","):
            plat, _, cid = key.strip().partition(":")
            if not (plat and cid):
                continue
            try:
                status = gateway.status()
                if not status.get(plat, {}).get("running_in_session"):
                    continue
                adapter = gateway._adapter_for(plat)
                if adapter is None or getattr(gateway, "dry_run", False):
                    continue
                result = adapter.send_media(
                    ChatRef(platform=plat, chat_id=cid), media)
                if getattr(result, "ok", False):
                    reached += 1
            except Exception:  # noqa: BLE001 — one dead channel skips
                continue
        _log.info("morning-pulse: voice note reached %d channel(s)", reached)
        return reached > 0
    except Exception as exc:  # noqa: BLE001
        _log.warning("morning-pulse: audio delivery failed: %s", exc)
        return False


def _to_voice_ogg(wav_path: str) -> tuple[str, str]:
    """Convert WAV → OGG/Opus for Telegram voice bubbles.

    Returns (path, mime).  Falls back to the WAV itself when ffmpeg is
    missing or conversion fails.
    """
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return wav_path, "audio/wav"
    ogg_path = os.path.splitext(wav_path)[0] + ".ogg"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", wav_path, "-c:a", "libopus",
             "-b:a", "64k", ogg_path],
            check=True, capture_output=True, timeout=120)
        if os.path.isfile(ogg_path):
            return ogg_path, "audio/ogg"
    except Exception as exc:  # noqa: BLE001
        _log.warning("morning-pulse: ffmpeg conversion failed: %s", exc)
    return wav_path, "audio/wav"


def _delivery_diagnosis(context: Any) -> str:
    """One-line diagnosis of WHY delivery can't reach the owner.

    The /notify audit found silent 0-deliveries; this makes the cause
    visible in logs instead of a bare False.
    """
    from .notifier import resolve_gateway
    if resolve_gateway(context) is None:
        return "no live gateway in context (scheduler context?)"
    settings = getattr(context, "settings", None)
    partner = getattr(settings, "partner", None)
    raw = (getattr(partner, "owner_chats", "") or "").strip()
    if not raw:
        return "owner_chats is empty — no destination configured"
    return (f"owner_chats={raw[:60]!r} but no platform reports "
            "running_in_session")


# ── tool registration ──────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Register the ``pulse`` tool (briefing slots: morning/afternoon/evening)."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "pulse",
        description=(
            "Daily briefings: morning (news → briefing → voice), afternoon "
            "(markets + fresh news), evening (day wrap + tomorrow). "
            "action=run (deliver now) | status (schedules + last runs) | "
            "config (show times) | catchup (run undelivered slots). "
            "slot=morning|afternoon|evening (default morning)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — run|status|config|catchup",
            "slot": "str — morning|afternoon|evening (default morning)",
        },
    )
    def pulse(*, action: str = "", slot: str = "morning") -> dict[str, Any]:
        slot = (slot or "morning").strip().lower()
        if slot not in BRIEFING_SLOTS and slot != "all":
            return {"error": f"unknown slot {slot!r}"}
        act = (action or "").strip().lower()
        if act == "run":
            if slot == "all":
                return {s: run_slot(context, s) for s in BRIEFING_SLOTS}
            return run_slot(context, slot)
        if act == "status":
            from .scheduler import Scheduler
            sched = Scheduler(context)
            try:
                jobs = {j.get("name"): j for j in sched.list_jobs()}
            except Exception:  # noqa: BLE001
                jobs = {}
            out: dict[str, Any] = {}
            for s, spec in BRIEFING_SLOTS.items():
                out[s] = {
                    "job": jobs.get(spec["job_name"]),
                    "time": slot_time(context, s),
                    "timezone": str(_slot_prefs(context, s).get("timezone")),
                    "enabled": slot_enabled(context, s),
                    "voice": slot_voice(context, s),
                    "last_run": (_read_slot_marker(context, s)
                                 if s != "morning"
                                 else last_pulse_run(context)),
                }
            return out if slot == "all" else out.get(slot, {})
        if act == "catchup":
            if slot == "all":
                return check_all_catchup(context)
            return check_slot_catchup(context, slot)
        if act == "config":
            if slot == "all":
                return {s: {
                    "time": slot_time(context, s),
                    "timezone": str(_slot_prefs(context, s).get("timezone")),
                    "enabled": slot_enabled(context, s),
                    "voice": slot_voice(context, s),
                } for s in BRIEFING_SLOTS}
            # flat morning format for backward compat
            return {"time": pulse_time(context),
                    "timezone": pulse_timezone(context),
                    "enabled": pulse_enabled(context),
                    "hosts": [HOST_1_NAME, HOST_2_NAME]}
        return {"error": f"unsupported pulse action {act!r}"}


# ═══════════════════════════════════════════════════════════════════════
# Multi-slot briefings: morning / afternoon / evening
# Extends the morning pulse with two shorter briefings.  Each slot has
# its own schedule, enable switch, last-run marker, and content builder.
# A per-day seen-store keeps slots from repeating each other's items.
# ═══════════════════════════════════════════════════════════════════════

#: Slot definitions.  ``voice`` controls whether the slot synthesizes
#: audio (morning does; afternoon/evening are text-first by default).
BRIEFING_SLOTS: dict[str, dict[str, Any]] = {
    "morning": {
        "job_name": PULSE_JOB_NAME,          # "morning-pulse" (existing)
        "default_time": DEFAULT_PULSE_TIME,  # "23:00"
        "title": "🌙 Morning pulse",
        "voice": True,
        "marker": "pulse_last_run.json",     # existing marker
    },
    "afternoon": {
        "job_name": "afternoon-pulse",
        "default_time": "13:00",
        "title": "☀️ Afternoon briefing",
        "voice": False,
        "marker": "pulse_last_run_afternoon.json",
    },
    "evening": {
        "job_name": "evening-pulse",
        "default_time": "19:00",
        "title": "🌆 Evening wrap",
        "voice": False,
        "marker": "pulse_last_run_evening.json",
    },
}

#: Market symbols for the afternoon snapshot (best-effort each).
AFTERNOON_SYMBOLS = ["XAUUSD", "BTC", "ETH", "EURUSD"]


def _slot_prefs(context: Any, slot: str) -> dict[str, Any]:
    """Per-slot prefs: settings.pulse.<slot>_time / <slot>_enabled.

    Falls back to the legacy flat keys for the morning slot
    (``time``/``enabled``) so existing config keeps working.
    """
    slot = (slot or "morning").strip().lower()
    spec = BRIEFING_SLOTS.get(slot, BRIEFING_SLOTS["morning"])
    out = {"time": spec["default_time"], "timezone": PULSE_TIMEZONE,
           "enabled": True, "voice": spec["voice"]}
    bs = getattr(getattr(context, "settings", None), "pulse", None)
    if bs is not None and hasattr(bs, "__dict__"):
        d = bs.__dict__
        t = d.get(f"{slot}_time")
        if t:
            out["time"] = str(t)
        elif slot == "morning":
            t0 = d.get("time")
            if t0:
                out["time"] = str(t0)
        e = d.get(f"{slot}_enabled")
        if e is not None:
            out["enabled"] = bool(e)
        elif slot == "morning":
            e0 = d.get("enabled")
            if e0 is not None:
                out["enabled"] = bool(e0)
        tz = d.get("timezone")
        if tz:
            out["timezone"] = str(tz)
        v = d.get(f"{slot}_voice")
        if v is not None:
            out["voice"] = bool(v)
        return out
    # JSON sidecar fallback
    try:
        ws = getattr(getattr(context, "settings", None),
                     "workspace_dir", "")
        sidecar = Path(ws) / "pulse_prefs.json" if ws else None
        if sidecar is not None and sidecar.is_file():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                t = data.get(f"{slot}_time") or (
                    data.get("time") if slot == "morning" else None)
                if t:
                    out["time"] = str(t)
                e = data.get(f"{slot}_enabled")
                if e is None and slot == "morning":
                    e = data.get("enabled")
                if e is not None:
                    out["enabled"] = bool(e)
                if data.get("timezone"):
                    out["timezone"] = str(data["timezone"])
                v = data.get(f"{slot}_voice")
                if v is not None:
                    out["voice"] = bool(v)
    except Exception:  # noqa: BLE001
        pass
    return out


def slot_time(context: Any, slot: str = "morning") -> str:
    return str(_slot_prefs(context, slot).get("time")
               or BRIEFING_SLOTS.get(slot, {}).get("default_time")
               or DEFAULT_PULSE_TIME)


def slot_enabled(context: Any, slot: str = "morning") -> bool:
    return bool(_slot_prefs(context, slot).get("enabled", True))


def slot_voice(context: Any, slot: str = "morning") -> bool:
    return bool(_slot_prefs(context, slot).get("voice", False))


# ── per-slot markers ─────────────────────────────────────────────────

def _slot_marker_path(context: Any, slot: str) -> Path | None:
    try:
        ws = getattr(getattr(context, "settings", None), "workspace_dir", "")
        if not ws:
            return None
        spec = BRIEFING_SLOTS.get(slot, BRIEFING_SLOTS["morning"])
        return Path(ws) / spec["marker"]
    except Exception:  # noqa: BLE001
        return None


def _read_slot_marker(context: Any, slot: str) -> dict[str, Any]:
    try:
        p = _slot_marker_path(context, slot)
        if p is not None and p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _write_slot_marker(context: Any, slot: str,
                       result: dict[str, Any]) -> None:
    try:
        p = _slot_marker_path(context, slot)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "ts": time.time(),
            "slot": slot,
            "stages": result.get("stages", []),
            "delivered_text": bool(result.get("delivered_text")),
            "delivered_audio": bool(result.get("delivered_audio")),
            "elapsed_s": result.get("elapsed_s", 0),
        }), encoding="utf-8")
    except Exception:  # noqa: BLE001
        _log.debug("pulse: marker write failed for slot %s", slot,
                   exc_info=True)


# ── seen-store: no repeats across slots ─────────────────────────────

def _seen_path(context: Any) -> Path | None:
    try:
        ws = getattr(getattr(context, "settings", None), "workspace_dir", "")
        return Path(ws) / "pulse_seen.json" if ws else None
    except Exception:  # noqa: BLE001
        return None


def _read_seen(context: Any) -> dict[str, Any]:
    """{day: {item_key: slot}} — item keys already delivered today."""
    try:
        from datetime import datetime
        from ..core.tz import safe_zoneinfo
        p = _seen_path(context)
        if p is None or not p.is_file():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        tz = safe_zoneinfo(PULSE_TIMEZONE)
        today = datetime.now(tz).strftime("%Y-%m-%d")
        # prune to today only
        return {today: data.get(today, {})}
    except Exception:  # noqa: BLE001
        return {}


def _mark_seen(context: Any, keys: list[str], slot: str) -> None:
    if not keys:
        return
    try:
        from datetime import datetime
        from ..core.tz import safe_zoneinfo
        p = _seen_path(context)
        if p is None:
            return
        tz = safe_zoneinfo(PULSE_TIMEZONE)
        today = datetime.now(tz).strftime("%Y-%m-%d")
        data: dict[str, Any] = {}
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:  # noqa: BLE001
                pass
        day = data.get(today, {})
        if not isinstance(day, dict):
            day = {}
        for k in keys:
            day[str(k)] = slot
        data = {today: day}  # drop older days
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001
        _log.debug("pulse: seen-store write failed", exc_info=True)


def _filter_unseen(context: Any, items: list[dict[str, Any]],
                   key_fn: Any) -> list[dict[str, Any]]:
    """Drop items whose key was already delivered today by any slot."""
    seen = _read_seen(context)
    day_keys: set[str] = set()
    for day in seen.values():
        if isinstance(day, dict):
            day_keys.update(str(k) for k in day)
    out = []
    for it in items:
        try:
            k = str(key_fn(it))
        except Exception:  # noqa: BLE001
            k = ""
        if k and k in day_keys:
            continue
        out.append(it)
    return out


# ── afternoon content: markets + fresh news + today's remainder ─────

def _afternoon_markets(context: Any) -> tuple[str, list[str]]:
    """Market snapshot lines + seen keys.  Never raises."""
    lines: list[str] = []
    keys: list[str] = []
    try:
        from ..integrations.market_data import quote
        for sym in AFTERNOON_SYMBOLS:
            try:
                q = quote(sym)
            except Exception:  # noqa: BLE001
                continue
            if not q or q.get("price") is None:
                continue
            chg = q.get("change_pct_24h")
            chg_s = (f" ({chg:+.1f}% 24h)"
                     if isinstance(chg, (int, float)) else "")
            price = q["price"]
            if isinstance(price, float) and price >= 100:
                ps = f"${price:,.2f}"
            elif isinstance(price, float):
                ps = f"${price:,.4f}"
            else:
                ps = str(price)
            lines.append(f"• {sym}: {ps}{chg_s}")
            keys.append(f"mkt-{sym}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines), keys


def _fresh_news(context: Any, limit: int = 6) -> tuple[str, list[str]]:
    """News not yet delivered today.  Never raises."""
    try:
        from .news import NewsAgent
        agent = NewsAgent(context)
        report = agent.run()
        items = agent.recent(limit=15) if report.get("ok") else []
        fresh = _filter_unseen(
            context, items,
            lambda it: it.get("url") or it.get("title", ""))
        lines = []
        keys = []
        for it in fresh[:limit]:
            title = str(it.get("title", "")).strip()
            if title:
                lines.append(f"• {title}")
                keys.append(str(it.get("url") or title))
        return "\n".join(lines), keys
    except Exception:  # noqa: BLE001
        return "", []


def _today_remainder(context: Any) -> str:
    """Scheduler jobs still due today.  Never raises."""
    try:
        from datetime import datetime
        from ..core.tz import safe_zoneinfo
        from .scheduler import Scheduler
        sched = Scheduler(context)
        jobs = sched.list_jobs()
        tz = safe_zoneinfo(PULSE_TIMEZONE)
        now = datetime.now(tz)
        lines = []
        for j in jobs or []:
            name = str(j.get("name", ""))
            if not name or "pulse" in name:
                continue
            spec = str(j.get("spec", ""))
            # crude: daily HH:MM specs later today
            if spec.startswith("daily "):
                try:
                    hh, mm = spec.split()[1].split(":")[:2]
                    if (int(hh), int(mm)) > (now.hour, now.minute):
                        lines.append(f"• {name} — {hh}:{mm}")
                except Exception:  # noqa: BLE001
                    continue
        return "\n".join(lines[:8])
    except Exception:  # noqa: BLE001
        return ""


def _compose_afternoon(context: Any) -> tuple[str, list[str]]:
    """Afternoon briefing text + delivered item keys."""
    keys: list[str] = []
    parts = ["☀️ *Afternoon briefing* — what's moved since this morning.\n"]
    markets, mkeys = _afternoon_markets(context)
    keys.extend(mkeys)
    if markets:
        parts.append("*Markets*\n" + markets + "\n")
    news, nkeys = _fresh_news(context)
    keys.extend(nkeys)
    if news:
        parts.append("*Fresh headlines*\n" + news + "\n")
    rest = _today_remainder(context)
    if rest:
        parts.append("*Still ahead today*\n" + rest + "\n")
    if len(parts) == 1:
        parts.append("Quiet afternoon — nothing new since the morning pulse.")
    return "\n".join(parts).strip(), keys


# ── evening content: day wrap + tomorrow + pending ───────────────────

def _day_wrap(context: Any) -> str:
    """What happened today, from the autonomy ledger.  Never raises."""
    try:
        from .autonomy_ledger import AutonomyLedger
        ledger = AutonomyLedger(context)
        rows = ledger.recent(limit=30, since_hours=14)
        if not rows:
            return ""
        # group by system, keep the interesting bits
        by_sys: dict[str, list[str]] = {}
        for r in rows:
            sys = str(r.get("system", "?"))
            summ = str(r.get("summary", "")).strip()
            if not summ or len(summ) > 120:
                continue
            by_sys.setdefault(sys, []).append(summ)
        lines = []
        for sys, sums in sorted(by_sys.items()):
            for s in sums[:3]:
                lines.append(f"• [{sys}] {s}")
        return "\n".join(lines[:12])
    except Exception:  # noqa: BLE001
        return ""


def _tomorrow_preview(context: Any) -> str:
    """Tomorrow's scheduled jobs.  Never raises."""
    try:
        from .scheduler import Scheduler
        sched = Scheduler(context)
        jobs = sched.list_jobs()
        lines = []
        for j in jobs or []:
            name = str(j.get("name", ""))
            if not name or "pulse" in name:
                continue
            spec = str(j.get("spec", ""))
            if spec.startswith("daily "):
                try:
                    hhmm = spec.split()[1][:5]
                    lines.append(f"• {name} — {hhmm}")
                except Exception:  # noqa: BLE001
                    continue
        return "\n".join(lines[:8])
    except Exception:  # noqa: BLE001
        return ""


def _pending_items(context: Any) -> str:
    """Open proposals, fired alerts, running missions.  Never raises."""
    lines: list[str] = []
    # proposals awaiting decision
    try:
        rows = context.db.query(
            "SELECT id FROM proactive_log WHERE status = 'pending' LIMIT 5")
        for r in rows or []:
            lines.append(f"• proposal {r['id']} awaiting decision")
    except Exception:  # noqa: BLE001
        pass
    # finance alerts armed
    try:
        from ..finance.alerts import AlertStore
        store = AlertStore(context.db)
        alerts = store.list_enabled() if hasattr(store, "list_enabled") else []
        if alerts:
            lines.append(f"• {len(alerts)} price alert(s) armed")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def _compose_evening(context: Any) -> tuple[str, list[str]]:
    """Evening wrap text + delivered item keys."""
    parts = ["🌆 *Evening wrap* — closing out the day.\n"]
    wrap = _day_wrap(context)
    if wrap:
        parts.append("*Today*\n" + wrap + "\n")
    tom = _tomorrow_preview(context)
    if tom:
        parts.append("*Tomorrow*\n" + tom + "\n")
    pend = _pending_items(context)
    if pend:
        parts.append("*Still open*\n" + pend + "\n")
    if len(parts) == 1:
        parts.append("Quiet day — nothing major to report.")
    return "\n".join(parts).strip(), []


# ── generalized slot runner ──────────────────────────────────────────

def _deliver_slot_text(context: Any, slot: str, text: str) -> bool:
    """Deliver a slot's text via Notifier.  Never raises."""
    spec = BRIEFING_SLOTS.get(slot, BRIEFING_SLOTS["morning"])
    from .notifier import Notifier
    try:
        notifier = Notifier(context)
        res = notifier.publish("pulse", spec["title"], text[:3500],
                               force=True)
        delivered = bool(res.get("delivered"))
        if not delivered:
            _log.warning("pulse/%s: text not delivered — %s", slot,
                         _delivery_diagnosis(context))
        return delivered
    except Exception as exc:  # noqa: BLE001
        _log.warning("pulse/%s: text delivery failed: %s", slot, exc)
        return False


def run_slot(context: Any, slot: str = "morning") -> dict[str, Any]:
    """Run one briefing slot end-to-end.  Never raises.

    Morning keeps the full pipeline (news → briefing → voice →
    deliver).  Afternoon/evening are shorter text-first briefings
    built from fresh, not-yet-delivered items.
    """
    slot = (slot or "morning").strip().lower()
    if slot not in BRIEFING_SLOTS:
        return {"ok": False, "error": f"unknown slot {slot!r}"}
    if slot == "morning":
        # The existing full pipeline; marker written by run_pulse.
        result = run_pulse(context)
        result["slot"] = "morning"
        return result

    started = time.time()
    stages: list[str] = []
    if slot == "afternoon":
        text, keys = _compose_afternoon(context)
    else:
        text, keys = _compose_evening(context)
    stages.append("compose")

    # voice only when explicitly enabled for the slot
    audio_path = ""
    if slot_voice(context, slot):
        ok, audio_path, secs, att = _run_stage(
            f"{slot}-voice",
            lambda: _synthesize(
                context,
                f"{BRIEFING_SLOTS[slot]['title']}: "
                + " ".join(text.split())[:800]))
        audio_path = audio_path or ""
        if audio_path:
            stages.append("voice")

    text_ok = _deliver_slot_text(context, slot, text)
    audio_ok = _deliver_audio(context, audio_path) if audio_path else False
    stages.append("deliver")

    if text_ok and keys:
        _mark_seen(context, keys, slot)

    elapsed = round(time.time() - started, 1)
    delivered = text_ok or audio_ok
    _ledger(context, "run",
            f"{slot} briefing: stages={','.join(stages)} "
            f"text={'delivered' if text_ok else 'FAILED'}",
            cost_seconds=elapsed, ok=delivered)
    _emit_bus({"slot": slot, "stages": stages,
               "delivered_text": text_ok,
               "delivered_audio": audio_ok,
               "elapsed_s": elapsed, "ok": delivered})
    result = {"ok": True, "slot": slot, "stages": stages, "text": text,
              "audio_path": audio_path, "delivered_text": text_ok,
              "delivered_audio": audio_ok, "elapsed_s": elapsed}
    _write_slot_marker(context, slot, result)
    _log.info("pulse/%s done in %ss: text=%s audio=%s", slot, elapsed,
              text_ok, audio_ok)
    return result


# ── per-slot scheduling + catch-up ───────────────────────────────────

def _slot_due_today(context: Any, slot: str) -> float:
    try:
        from datetime import datetime
        from ..core.tz import safe_zoneinfo
        prefs = _slot_prefs(context, slot)
        tz = safe_zoneinfo(str(prefs.get("timezone") or PULSE_TIMEZONE))
        hh, mm = (str(prefs.get("time") or "12:00").split(":") + ["0"])[:2]
        return datetime.now(tz).replace(hour=int(hh), minute=int(mm),
                                        second=0, microsecond=0).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def ensure_slot_job(context: Any, slot: str = "morning") -> dict[str, Any]:
    """Register one durable briefing job for ``slot`` (idempotent)."""
    from .scheduler import Scheduler

    slot = (slot or "morning").strip().lower()
    spec = BRIEFING_SLOTS.get(slot)
    if spec is None:
        return {"error": f"unknown slot {slot!r}"}
    if not slot_enabled(context, slot):
        # enabled=False → make sure no stale job lingers
        try:
            sched = Scheduler(context)
            for j in sched.list_jobs():
                if j.get("name") == spec["job_name"]:
                    sched.remove(j["id"])
        except Exception:  # noqa: BLE001
            pass
        return {"name": spec["job_name"], "disabled": True}
    if slot == "morning":
        return ensure_pulse_job(context)  # existing path, unchanged

    sched = Scheduler(context)
    want = slot_time(context, slot)
    tz = str(_slot_prefs(context, slot).get("timezone") or PULSE_TIMEZONE)
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == spec["job_name"]]
    except Exception:  # noqa: BLE001
        have = []
    if have:
        job0 = have[0]
        same = all(job0.get(k) == v for k, v in PULSE_POLICIES.items())
        if _pulse_spec_matches(job0.get("spec") or "", want) and same:
            return {"name": spec["job_name"], "already_scheduled": True,
                    "job_id": job0.get("id")}
    for j in have:
        try:
            sched.remove(j["id"])
        except Exception:  # noqa: BLE001
            pass
    job = sched.add(spec["job_name"], f"daily {want}", "tool",
                    {"tool": "pulse",
                     "args": {"action": "run", "slot": slot}},
                    timezone=tz, **PULSE_POLICIES)
    _log.info("scheduled %s job: %s (daily %s %s)",
              slot, spec["job_name"], want, tz)
    return {"name": spec["job_name"], "scheduled": True,
            "job_id": job.get("id")}


def ensure_pulse_jobs(context: Any) -> dict[str, Any]:
    """Register all enabled briefing slots.  Idempotent, boot-safe."""
    out: dict[str, Any] = {}
    for slot in BRIEFING_SLOTS:
        try:
            out[slot] = ensure_slot_job(context, slot)
        except Exception as exc:  # noqa: BLE001
            out[slot] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def check_slot_catchup(context: Any, slot: str = "morning") -> dict[str, Any]:
    """Boot-time catch-up for one slot.  Never raises."""
    slot = (slot or "morning").strip().lower()
    if slot == "morning":
        return check_pulse_catchup(context)  # existing path
    try:
        if not slot_enabled(context, slot):
            return {"catchup": False, "reason": f"{slot} disabled"}
        due = _slot_due_today(context, slot)
        now = time.time()
        if not due or now < due:
            return {"catchup": False, "reason": f"{slot} time not reached"}
        marker = _read_slot_marker(context, slot)
        last_ts = float(marker.get("ts") or 0)
        if last_ts >= due and (marker.get("delivered_text")
                               or marker.get("delivered_audio")):
            return {"catchup": False, "reason": "already delivered"}
        _log.info("pulse/%s: catch-up firing", slot)
        result = run_slot(context, slot)
        result["catchup"] = True
        return result
    except Exception as exc:  # noqa: BLE001
        _log.warning("pulse/%s catch-up failed: %s", slot, exc)
        return {"catchup": False,
                "reason": f"{type(exc).__name__}: {exc}"}


def check_all_catchup(context: Any) -> dict[str, Any]:
    """Boot-time catch-up for every enabled slot.  Never raises."""
    out: dict[str, Any] = {}
    for slot in BRIEFING_SLOTS:
        try:
            out[slot] = check_slot_catchup(context, slot)
        except Exception as exc:  # noqa: BLE001
            out[slot] = {"catchup": False,
                         "reason": f"{type(exc).__name__}: {exc}"}
    return out

