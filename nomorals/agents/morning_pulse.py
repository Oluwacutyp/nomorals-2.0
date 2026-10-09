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

def ensure_pulse_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``morning-pulse`` job (idempotent).

    Runs ``daily HH:MM`` at the configured pulse time (default 23:00) in
    America/Denver.  Safe to call on every boot.  If the owner changed the
    time, the job is replaced.
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
    if have and have[0].get("spec") == want:
        return {"name": PULSE_JOB_NAME, "already_scheduled": True,
                "job_id": have[0].get("id")}
    for j in have:  # stale time → replace
        try:
            sched.remove(j["id"])
        except Exception:  # noqa: BLE001
            pass
    job = sched.add(PULSE_JOB_NAME, f"daily {want}", "tool",
                    {"tool": "pulse", "args": {"action": "run"}},
                    timezone=tz)
    _log.info("scheduled morning-pulse job: %s (daily %s %s)",
              PULSE_JOB_NAME, want, tz)
    return {"name": PULSE_JOB_NAME, "scheduled": True,
            "job_id": job.get("id")}


# ── the pipeline ─────────────────────────────────────────────────────────

def run_pulse(context: Any) -> dict[str, Any]:
    """Run the full morning-pulse pipeline.  Never raises.

    Returns {ok, text, audio_path, delivered_text, delivered_audio,
    stages} — stages names each completed step for observability.
    """
    stages: list[str] = []
    started = time.time()

    # ── 1. news radar ────────────────────────────────────────────────
    news_items: list[dict[str, Any]] = []
    try:
        from .news import NewsAgent
        agent = NewsAgent(context)
        report = agent.run()
        if report.get("ok"):
            news_items = agent.recent(limit=12)
        stages.append("news")
        _log.info("morning-pulse: news radar got %d items",
                  len(news_items))
    except Exception as exc:  # noqa: BLE001 — news must not kill the pulse
        _log.warning("morning-pulse: news radar failed: %s", exc)

    # ── 2. compose ───────────────────────────────────────────────────
    briefing_text = ""
    try:
        from .morning_briefing import BriefingComposer
        composer = BriefingComposer()
        from datetime import datetime
        from ..core.tz import safe_zoneinfo
        tz = safe_zoneinfo(pulse_timezone(context))
        date = datetime.now(tz).strftime("%Y-%m-%d")
        briefing = composer.compose(context, date)
        briefing_text = briefing.render_text()
        stages.append("compose")
    except Exception as exc:  # noqa: BLE001
        _log.warning("morning-pulse: briefing compose failed: %s", exc)
        briefing_text = "(briefing unavailable tonight)"

    # ── 3. script (two-host conversational) ──────────────────────────
    script = _write_script(context, briefing_text, news_items)
    stages.append("script")

    # ── 4. voice ─────────────────────────────────────────────────────
    audio_path = ""
    try:
        audio_path = _synthesize(context, script) or ""
        if audio_path:
            stages.append("voice")
    except Exception as exc:  # noqa: BLE001 — voice must not kill delivery
        _log.warning("morning-pulse: voice synthesis failed: %s", exc)

    # ── 5. deliver ───────────────────────────────────────────────────
    text_ok = _deliver_text(context, briefing_text)
    audio_ok = _deliver_audio(context, audio_path) if audio_path else False
    stages.append("deliver")

    elapsed = round(time.time() - started, 1)
    _log.info("morning-pulse done in %ss: stages=%s text=%s audio=%s",
              elapsed, stages, text_ok, audio_ok)
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
            response = router.chat([Message(role="user", content=prompt)])
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
