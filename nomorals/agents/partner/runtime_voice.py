"""RuntimeVoiceMixin: PartnerRuntime command group (voice)."""

from __future__ import annotations

import os
from typing import Any, Callable
from ...core.logging_setup import get_logger
_log = get_logger(__name__)


class RuntimeVoiceMixin:
    """RuntimeVoiceMixin for :class:`PartnerRuntime`."""


    # ── voice: /tts /stt ─────────────────────────────────────────────────────
    def _control_tts(self, tail: str, *, chat_key: str) -> str:
        """Speak text: synthesize with the best available engine, send the file."""
        text = (tail or "").strip()
        if not text:
            return "usage: /tts <text to speak>"
        outcome = self.context.tools.call("speak", text=text[:4000])
        if not outcome.ok:
            return f"tts failed: {getattr(outcome.error, 'message', outcome.error)}"
        info = outcome.value
        chat = self._ref_from_key(chat_key)
        try:
            result = self.gateway.send_file(chat.platform, chat, info["path"],
                                            caption=f"[{info.get('engine')}]")
            if getattr(result, "ok", False):
                return f"🔊 spoken ({info.get('engine')}, {info.get('bytes', 0) // 1024} KB, {info.get('seconds')}s)"
        except Exception:  # noqa: BLE001 - console and other adapters
            pass
        return f"🔊 spoken ({info.get('engine')}) — saved at {info.get('path')}"

    def _control_stt(self, tail: str) -> str:
        """Transcribe an audio file with the best available STT backend."""
        path = (tail or "").strip()
        if not path:
            return "usage: /stt <audio file path>"
        outcome = self.context.tools.call("transcribe", path=path)
        if not outcome.ok:
            return f"stt failed: {getattr(outcome.error, 'message', outcome.error)}"
        value = outcome.value
        text = value.get("text") or "(no speech detected)"
        return f"🎧 [{value.get('provider')}, {value.get('seconds')}s]\n{text[:1500]}"

    # ── vision: /look (screen reader) ────────────────────────────────────────
    def _control_look(self, tail: str, *, chat_key: str) -> str:
        """Screen-reader analysis of a screenshot (path or URL)."""
        parts = (tail or "").strip().split(None, 1)
        if not parts:
            return "usage: /look <path or url> [what to focus on]"
        target = parts[0]
        focus = parts[1] if len(parts) > 1 else ""
        kwargs = {"url": target} if target.startswith(("http://", "https://")) else {"path": target}
        outcome = self.context.tools.call("vision_screen", focus=focus, **kwargs)
        if not outcome.ok:
            return f"vision failed: {getattr(outcome.error, 'message', outcome.error)}"
        value = outcome.value
        description = value.get("description") or "(vision model unavailable — metadata only)"
        ocr_text = str(value.get("ocr_text") or "").strip()
        text = (f"👁 screen read ({value.get('format', '?')}, {value.get('width', '?')}x{value.get('height', '?')}) "
                f"[{value.get('provider') or 'no provider'}]\n{description}")
        if ocr_text:
            text += f"\n\n— verbatim text from the pixels (OCR) —\n{ocr_text[:2000]}"
        chat = self._ref_from_key(chat_key)
        return self._send_long_checked(chat.platform, chat, text[:6000])

    def _control_speak(self, tail: str, chat_key: str) -> str:
        """`/speak <text>` — I say it: neural TTS voice note sent back here."""
        text = (tail or "").strip()
        if not text:
            return ("usage: /speak <text> — I'll say it as a voice note "
                    "(tags like [happy] [whisper] [laughs] [pause:300] work)")
        # bridge the current mood into the voice, best-effort
        mood, mood_level = "", 5
        mood_box = self.context.extras.get("mood")
        if mood_box is not None:
            mood = str(getattr(mood_box, "mood", "") or
                       (mood_box.get("mood") if isinstance(mood_box, dict) else "") or "")
            try:
                mood_level = int(getattr(mood_box, "level", 5) or 5)
            except (TypeError, ValueError):
                mood_level = 5
        outcome = self.context.tools.call(
            "tts_say", text=text, mood=mood, mood_level=str(mood_level))
        if not outcome.ok:
            return f"speak failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        ref = self._ref_from_key(chat_key)
        gateway = self.context.extras.get("gateway")
        sent = False
        if gateway is not None and ref.platform in getattr(gateway, "adapters", {}):
            try:
                # Caption ceiling is ~1024 chars on the platforms; the full
                # transcript rides beside the voice note so it's readable
                # and searchable in chat.
                result = gateway.send_file(ref.platform, ref, v["path"],
                                           caption=text[:1024])
                sent = bool(getattr(result, "ok", False))
                if sent and len(text) > 1024:
                    try:
                        gateway.send(ref.platform, ref, text)
                    except Exception:  # noqa: BLE001 - transcript is a bonus
                        _log.debug("voice-note transcript send failed",
                                   exc_info=True)
            except Exception:  # noqa: BLE001 - report the file, don't crash
                sent = False
        backend = v.get("backend", "?")
        return self._deliver_voice_note(chat_key, v["path"], text, backend,
                                        size_kb=v["bytes"] // 1024)

    def _deliver_voice_note(self, chat_key: str, path: str, text: str,
                            backend: str, size_kb: int = 0) -> str:
        """Send a wav back into the chat as a voice note; reply text."""
        ref = self._ref_from_key(chat_key)
        gateway = self.context.extras.get("gateway")
        sent = False
        if gateway is not None and ref.platform in getattr(gateway, "adapters", {}):
            try:
                # Caption ceiling is ~1024 chars on the platforms; the full
                # transcript rides beside the voice note so it's readable
                # and searchable in chat.
                result = gateway.send_file(ref.platform, ref, path,
                                           caption=text[:1024])
                sent = bool(getattr(result, "ok", False))
                if sent and len(text) > 1024:
                    try:
                        gateway.send(ref.platform, ref, text)
                    except Exception:  # noqa: BLE001 - transcript is a bonus
                        _log.debug("voice-note transcript send failed",
                                   exc_info=True)
            except Exception:  # noqa: BLE001 - report the file, don't crash
                sent = False
        if not size_kb and os.path.exists(path):
            size_kb = os.path.getsize(path) // 1024
        if sent:
            return f"🎙️ said it ({backend}, {size_kb} KB voice note)"
        return (f"🎙️ voice note ready ({backend}, {size_kb} KB): {path}")

    # ── voice catalogue: /voice (Telegram / WhatsApp / console) ─────────────
    def _control_voice(self, tail: str, chat_key: str,
                       message: Any = None) -> str:
        """The voice catalogue, live from any chat.

        /voice list                              catalogue + active voice
        /voice use <name>                        switch this chat's voice now
        /voice say <text>                        speak as the chat's voice
        /voice clone <name> [path]               clone a voice note / file
        /voice transcript <name> <text>          set a clone's prompt text
        /voice describe <name> <text>            describe a catalogue voice
        /voice backend <name> <backend>          change a voice's backend
        /voice rm <name>                         drop a catalogue voice
        """
        from ...voice.catalogue import default_catalogue

        cat = default_catalogue()
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        # Unified voice pipeline (#108) subcommands — the architecture under
        # all of Phase 24. Kept under /voice so the chat surface stays one.
        if verb in ("pipeline", "engines", "keyterms"):
            from ...audio.pipeline import control_voice as _pipeline_control
            return _pipeline_control(tail or "", context=self.context,
                                     chat=chat_key)

        if verb in ("list", "ls"):
            voices = cat.list()
            if not voices:
                return ("no voices in the catalogue yet — clone one:\n"
                        "/voice clone <name> (attach a voice note to the "
                        "command message, or give a file path)")
            lines = ["voices:"]
            for v in voices:
                mark = "●" if v["active"] else "○"
                chat_mark = (" [this chat]"
                             if cat.chat_overrides.get(chat_key) == v["name"]
                             else "")
                desc = f" — {v['description']}" if v["description"] else ""
                prof = f" (profile: {v['profile']})" if v["profile"] else ""
                lines.append(f"  {mark} {v['name']} [{v['backend']}]"
                             f"{prof}{chat_mark}{desc}")
            return "\n".join(lines)

        if verb == "use":
            name = rest.strip()
            if not name:
                return "usage: /voice use <name>"
            try:
                cat.set_chat_voice(chat_key, name)
            except KeyError:
                return f"unknown voice {name!r} — /voice list"
            return f"🎙️ this chat now speaks as '{name}'"

        if verb == "say":
            text = rest.strip()
            if not text:
                return "usage: /voice say <text>"
            mood = self._voice_mood()
            try:
                out = cat.speak(text, chat_key=chat_key, mood=mood or "neutral")
            except Exception as exc:  # noqa: BLE001
                return f"voice say failed: {exc}"
            return self._deliver_voice_note(chat_key, out["path"], text,
                                            out.get("backend", "?"))

        if verb == "clone":
            cparts = rest.strip().split(None, 1)
            if not cparts:
                return ("usage: /voice clone <name> [audio path] — attach a "
                        "voice note to this message or pass a file path")
            name, path = cparts[0], (cparts[1] if len(cparts) > 1 else "")
            if not path and message is not None:
                for media in getattr(message, "media", None) or []:
                    if getattr(media, "kind", "") in ("audio", "voice"):
                        path = getattr(media, "path", "")
                        break
            if not path or not os.path.exists(path):
                return ("no audio found — attach a voice note to the /voice "
                        "clone message or pass a file path")
            transcript = ""
            try:
                outcome = self.context.tools.call("transcribe", path=path)
                if outcome.ok and outcome.value:
                    transcript = outcome.value.get("text", "") or ""
            except Exception:  # noqa: BLE001 - transcript is a bonus
                transcript = ""
            try:
                voice = cat.clone(name, path, transcript=transcript,
                                  description="cloned by owner via /voice clone")
            except Exception as exc:  # noqa: BLE001
                return f"clone failed: {exc}"
            note = (f" (transcript: {transcript[:80]}…)"
                    if transcript else " (no transcript — set one with "
                    "/voice transcript <name> <text>)")
            return (f"🎙️ cloned '{voice.name}' from your voice note{note}\n"
                    f"say something: /voice say hello there")

        if verb == "transcript":
            tparts = rest.strip().split(None, 1)
            if len(tparts) < 2:
                return "usage: /voice transcript <name> <text>"
            name, text = tparts
            profile = cat.library.get(name)
            if profile is None:
                return f"unknown voice {name!r} — /voice list"
            profile.prompt_text = text.strip()
            cat.library._save_index()
            return f"transcript set for '{name}' ({len(text)} chars)"

        if verb == "describe":
            dparts = rest.strip().split(None, 1)
            if len(dparts) < 2:
                return "usage: /voice describe <name> <text>"
            voice = cat.get(dparts[0])
            if voice is None:
                return f"unknown voice {dparts[0]!r} — /voice list"
            voice.description = dparts[1].strip()
            cat._save()
            return f"description set for '{voice.name}'"

        if verb == "backend":
            bparts = rest.strip().split(None, 1)
            if len(bparts) < 2:
                return "usage: /voice backend <name> <backend>"
            voice = cat.get(bparts[0])
            if voice is None:
                return f"unknown voice {bparts[0]!r} — /voice list"
            voice.backend = bparts[1].strip()
            cat._save()
            return f"'{voice.name}' now prefers backend '{voice.backend}'"

        if verb in ("rm", "remove", "delete"):
            name = rest.strip()
            if not name:
                return "usage: /voice rm <name>"
            if not cat.remove(name):
                return f"unknown voice {name!r} — /voice list"
            return f"removed '{name}' from the catalogue"

        return ("usage: /voice list | use <name> | say <text> | clone <name> "
                "[path] | transcript <name> <text> | describe <name> <text> | "
                "backend <name> <backend> | rm <name>")

    def _voice_mood(self) -> str:
        """Best-effort current mood label for voice performances."""
        mood_box = self.context.extras.get("mood")
        if mood_box is None:
            return ""
        return str(getattr(mood_box, "mood", "") or
                   (mood_box.get("mood") if isinstance(mood_box, dict)
                    else "") or "")
