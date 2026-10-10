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
            if v:
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
    return {"ok": True, "stages": stages, "text": briefing_text,
            "audio_path": audio_path, "delivered_text": text_ok,
            "delivered_audio": audio_ok, "elapsed_s": elapsed}


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
            if text and HOST_1_NAME in text:
                _log.info("morning-pulse: LLM wrote %d-char script",
                          len(text))
                return text
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
    """Register the ``pulse`` tool (agent-callable morning pulse)."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "pulse",
        description=(
            "Morning pulse: the autonomous daily briefing with voice. "
            "action=run (news → briefing → voice → deliver now) | "
            "status (job schedule + last run) | config (show time/timezone)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "action": "str — run|status|config",
        },
    )
    def pulse(*, action: str = "") -> dict[str, Any]:
        act = (action or "").strip().lower()
        if act == "run":
            return run_pulse(context)
        if act == "status":
            from .scheduler import Scheduler
            sched = Scheduler(context)
            jobs = [j for j in sched.list_jobs()
                    if j.get("name") == PULSE_JOB_NAME]
            return {"job": jobs[0] if jobs else None,
                    "time": pulse_time(context),
                    "timezone": pulse_timezone(context),
                    "enabled": pulse_enabled(context)}
        if act == "config":
            return {"time": pulse_time(context),
                    "timezone": pulse_timezone(context),
                    "enabled": pulse_enabled(context),
                    "hosts": [HOST_1_NAME, HOST_2_NAME]}
        return {"error": f"unsupported pulse action {act!r}"}
