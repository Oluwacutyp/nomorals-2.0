"""RuntimeMediaMixin: PartnerRuntime command group (media)."""

from __future__ import annotations

import contextlib
import contextvars
import os
import re
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ...search.adaptive import adaptive_result_limit
from .failures import record_failure

_log = get_logger(__name__)

_AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac",
               ".wav"}
_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class _MixGatewaySender:
    """``mix_surface.Sender`` over the chat gateway: audio and voice
    notes ride ``gateway.send_file`` with the matching kind."""

    def __init__(self, gateway: Any) -> None:
        self._gateway = gateway

    def send_audio(self, chat: Any, path: str, **kwargs: Any) -> Any:
        from ...social.chat.base import ChatRef

        ref = ChatRef.parse(str(chat))
        return self._gateway.send_file(
            ref.platform, ref.key, path, kind="audio",
            caption=f"🎧 {Path(path).name}",
        )

    def send_voice(self, chat: Any, path: str, **kwargs: Any) -> Any:
        from ...social.chat.base import ChatRef

        ref = ChatRef.parse(str(chat))
        return self._gateway.send_file(
            ref.platform, ref.key, path, kind="voice",
            caption=f"🎧 {Path(path).name}",
        )


def _send_kind_for(path: str) -> str:
    """The delivery kind a downloaded file IS, by extension —
    so a download lands as a playable video, an audio message
    or a photo (delivery-shape law, 2026-11-07), never as a
    document wearing a file name.  Anything unrecognized
    keeps the plain file shape ("")."""
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    return ""


#: The ack-message handle of the background download job running on
#: this context: ``(chat_ref, ack_message_id)``.  Set by
#: ``_play_download_bg`` around the job body so bodies can edit the
#: ack at real milestones (``_job_ack_progress``) without threading
#: the handle through every signature.
_RUNNING_JOB_ACK: contextvars.ContextVar = contextvars.ContextVar(
    "devon_running_job_ack", default=None)

#: The operation owner for pick-lists opened by the play turn in
#: flight (Living Chat P1 targeting): the control dispatch in
#: runtime.py sets it from the requesting message, and
#: :meth:`RuntimeMediaMixin._control_play` reads it when no
#: explicit owner was passed — so a pick-list binds to the user
#: who asked, without changing _control_play's call shape (that
#: seam is doubled with the two-argument signature).
play_owner_var: contextvars.ContextVar = contextvars.ContextVar(
    "devon_play_owner", default="")

#: The intake command-claim key for the play turn in flight
#: (dupfix4 D4): the control dispatch in runtime.py sets it from
#: the requesting message, and
#: :meth:`RuntimeMediaMixin._control_play` reads it and threads it
#: to :meth:`_play_download_bg`, whose async body re-checks who
#: holds the key at SEND time — so a sibling that failed over and
#: claimed meanwhile stands this job down instead of landing the
#: second pick list. A ContextVar, like play_owner_var above, so
#: _control_play's two-argument call shape is unchanged (that seam
#: is doubled in tests with the two-argument signature).
play_claim_var: contextvars.ContextVar = contextvars.ContextVar(
    "devon_play_claim", default="")


def _pick_expired_reply() -> Any:
    """The honest answer when a /play pick-list is gone.

    One wording for both paths that can meet a dead list — the
    explicit ``/play pick <token> <n>`` command and the bare-number
    reply hook — so the two can never drift apart again (the
    2026-11-12 live defect: the hook had no answer at all, and the
    owner's "1" met silence). The "Search again" button re-runs
    /play; on surfaces without buttons it stays as tappable text.
    """
    from ...social.chat.tgbot_buttons import actionable

    return actionable(
        "that pick-list expired (30 min) — pick fast next time 😄",
        [[("🎵 Search again", "play")]])


def _looks_like_path_or_url(ref: str) -> bool:
    """Heuristic: is this an image path/URL (lookup) or a text prompt (generate)?"""
    s = (ref or "").strip()
    if not s:
        return False
    # URLs
    if re.match(r"(?i)^(https?|ftp|data|file)://", s):
        return True
    # Absolute or home-relative paths
    if s.startswith(("/", "~", "./", "../")):
        return True
    # Windows paths
    if re.match(r"(?i)^[a-z]:[\\/]", s):
        return True
    # Existing file (relative path)
    if os.path.exists(os.path.expanduser(s)):
        return True
    # Looks like a filename with an image extension
    if re.search(r"(?i)\.(png|jpe?g|gif|webp|bmp|tiff?|avif|heic|svg)$", s.split()[0]):
        return True
    return False


def _resolve_3d_input(ref: str, *, settings: Any, tag: str) -> str:
    """A path-or-URL -> a local file path (audit gap R2-25).

    A URL is fetched (``core.http.HttpClient`` — the same route vision
    uses) into the generated-media dir; a path is used as-is.
    """
    import time
    from pathlib import Path

    s = (ref or "").strip()
    if s.startswith(("http://", "https://")):
        from ...core.http import HttpClient

        body = HttpClient(timeout=60.0).get(s).body
        out_dir = Path(settings.resolve("data/media/generated"))
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(s.split("?", 1)[0]).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
                          ".glb", ".obj", ".fbx", ".vrm", ".gltf", ".dae"):
            suffix = ".jpg"
        dest = out_dir / f"{tag}_in_{time.time_ns()}{suffix}"
        dest.write_bytes(body)
        return str(dest)
    return str(Path(os.path.expanduser(s)))


def _tool_result_dict(outcome: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Unwrap a registry.call Outcome into the tool's dict.

    Unlike ``_tool_data`` (which reports "unknown" when a failing dict
    carries no "error" key), this keeps the organ's own "detail" — the
    3D wrappers report failures in "detail", the Blender tool in
    "error". Returns (data, error_text); exactly one is None.
    """
    if not getattr(outcome, "ok", False):
        err = getattr(outcome, "error", None)
        return None, str(getattr(err, "message", err) or "unknown")
    data = getattr(outcome, "value", None)
    if not isinstance(data, dict):
        return None, "tool returned no result"
    if not data.get("ok"):
        return None, str(data.get("detail") or data.get("error") or "failed")
    return data, None


#: Sentinel for ``_edit_chat_message``'s ``buttons``: the caller said
#: nothing about the keyboard — the settled-render clear
#: (``buttons=[]``) every existing caller relies on.
_EDIT_BUTTONS_UNSET: Any = object()


class RuntimeMediaMixin:
    """RuntimeMediaMixin for :class:`PartnerRuntime`."""


    # ── wave 72 systems: media · execution · archives · builders ───────────

    def _active_draft_path(self, chat_key: str) -> str | None:
        drafts = getattr(self, "_active_drafts", None) or {}
        return drafts.get(chat_key)

    def _set_active_draft(self, chat_key: str, path: str) -> None:
        drafts = getattr(self, "_active_drafts", None)
        if drafts is None:
            drafts = self._active_drafts = {}
        drafts[chat_key] = path

    def _judge_draft_or_warn(self, draft: Any) -> str:
        """Judge a draft; "" when judged, else an honest note.

        Judging must never break drafting — a failure is logged +
        recorded and the reply says the draft went unjudged.
        """
        from ...media.song_draft import judge_draft
        try:
            return f"\n\n{judge_draft(draft)['verdict']}"
        except Exception as exc:  # noqa: BLE001
            _log.error("draft judge failed: %s", exc)
            record_failure(
                "runtime_media.RuntimeMediaMixin._judge_draft_or_warn", exc)
            return (f"\n\n⚠️ I couldn't judge this draft ({exc}) — it went "
                    "out unjudged; the critique loop and bible check "
                    "didn't run.")

    def _control_music_draft(self, tail: str, chat_key: str = "") -> str:
        """ /music draft <topic> [style] — the artist's notebook: a full
        song draft (lyrics + cadence + melody contour + emotional arc +
        artist notes). The draft is inspectable and revisable. """
        from ...media.music import STYLES
        from ...media.song_draft import draft_song, render_notebook, save_draft
        words = (tail or "").strip().split()
        if not words:
            return ("usage: /music draft <topic> [style]\n"
                    f"styles: {', '.join(STYLES)}")
        style = "pop"
        if words[-1].lower() in STYLES and len(words) > 1:
            style = words[-1].lower()
            words = words[:-1]
        topic = " ".join(words)
        try:
            draft = draft_song(topic, style=style, context=self.context)
        except Exception as exc:  # noqa: BLE001
            return f"draft failed: {exc}"
        try:
            path = save_draft(draft)
            self._set_active_draft(chat_key, path)
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._control_music_draft", e)
        nb = render_notebook(draft)
        # chat gets the notebook; keep it readable, not the whole wall
        lines = nb.splitlines()
        preview = "\n".join(lines[:40])
        if len(lines) > 40:
            preview += f"\n… ({len(lines) - 40} more lines)"
        from ...social.chat.tgbot_buttons import actionable

        return actionable(
            f"📝 draft: “{draft.title}”\n\n{preview}"
            f"{self._judge_draft_or_warn(draft)}\n\n"
            f"revise: /music revise <note>",
            [[("🎤 Perform it", "music perform")]])

    def _control_music_revise(self, tail: str, chat_key: str = "") -> str:
        """ /music revise <note> — revise the active draft in plain words. """
        from ...media.song_draft import load_draft, render_notebook, revise_draft, save_draft
        note = (tail or "").strip()
        if not note:
            return "usage: /music revise <note> — e.g. /music revise make the chorus hit harder"
        path = self._active_draft_path(chat_key)
        if not path:
            return "no active draft — start one with /music draft <topic>"
        try:
            draft = load_draft(path)
            new = revise_draft(draft, note, context=self.context)
            new_path = save_draft(new)
            self._set_active_draft(chat_key, new_path)
        except Exception as exc:  # noqa: BLE001
            return f"revise failed: {exc}"
        nb = render_notebook(new)
        lines = nb.splitlines()
        preview = "\n".join(lines[:30])
        if len(lines) > 30:
            preview += f"\n… ({len(lines) - 30} more lines)"
        return (f"📝 revised: “{new.title}”\n\n{preview}"
                f"{self._judge_draft_or_warn(new)}")

    def _control_music_perform(self, chat_key: str = "") -> str:
        """ /music perform — perform the active draft: beat + vocals shaped
        together from the draft's own melody contours. """
        from ...media.draft_perform import perform_draft
        from ...media.song_draft import load_draft
        path = self._active_draft_path(chat_key)
        if not path:
            return "no active draft — start one with /music draft <topic>"
        try:
            draft = load_draft(path)
        except Exception as exc:  # noqa: BLE001
            return f"couldn't load the draft: {exc}"
        res = perform_draft(draft, context=self.context)
        if not res.get("ok"):
            return f"performance failed: {res.get('reason')}"
        out_path = res.get("path", "")
        if out_path:
            try:
                self.gateway.send_file(
                    "telegram", chat_key, out_path,
                    caption=f"🎵 {res.get('draft_title', '')} — performed from draft", kind="audio")
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._control_music_perform", e)
        return f"{res.get('note', '')}\n{out_path}"

    def _control_music_freestyle(self, tail: str, chat_key: str = "") -> str:
        """ /music freestyle [seed] [bars N] — live improvisation over a
        beat. Genuinely generative, bar by bar. """
        from ...media.freestyle import start_session
        words = (tail or "").strip().split()
        n_bars = 8
        seed_words: list[str] = []
        i = 0
        while i < len(words):
            if words[i].lower() == "bars" and i + 1 < len(words):
                try:
                    n_bars = max(1, min(32, int(words[i + 1])))
                except ValueError as e:
                    record_failure("runtime_media.RuntimeMediaMixin._control_music_freestyle", e)
                i += 2
            else:
                seed_words.append(words[i])
                i += 1
        seed = " ".join(seed_words)
        try:
            sess = start_session(seed=seed, bpm=92, energy=0.7,
                                 feel="hungry, in the pocket")
            bars = sess.spit(n_bars, context=self.context)
        except Exception as exc:  # noqa: BLE001
            return f"freestyle failed: {exc}"
        out = [f"🎤 freestyle — {sess.bpm} BPM · {len(bars)} bars",
               f"*seed: {seed or '(open)'}*", ""]
        out.extend(f"{b}" for b in bars)
        out.append(f"\n*say /music freestyle {seed} bars 8 for another round*"
                   if seed else "\n*say /music freestyle <seed> bars 8 for another round*")
        return "\n".join(out)

    def _control_music_bed(self, tail: str, chat_key: str = "") -> str:
        """ /music bed <topic> [style] — AI instrumental bed via ACE-Step 1.5.

        Lyrics come from MusicCreator's real engine; the bed is AI-generated.
        Fails honestly when the model isn't installed — never fake audio.
        """
        from ...media.ace_step import ACEModelUnavailable, make_bed, parse_bed_request
        from ...media.music import STYLES

        req = parse_bed_request(f"bed {tail}", STYLES)
        if req is None:
            return ("usage: /music bed <topic> [style]\n"
                    "e.g. /music bed lagos nights afrobeats")
        try:
            res = make_bed(req.topic, style=req.style,
                           duration_s=req.duration_s, context=self.context)
        except ACEModelUnavailable as exc:
            return f"🎹 couldn't make the AI bed:\n{exc}"
        except Exception as exc:  # noqa: BLE001
            return f"music bed error: {exc}"
        text = (f"🎹 “{res.title}” — AI instrumental bed "
                f"[{res.variant}, {res.duration_s:.0f}s]\n"
                f"tags: {res.tags}\n{res.note}")
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None and res.audio_path:
            try:
                self.gateway.send_file(
                    chat.platform, f"{chat.platform}:{chat.chat_id}",
                    res.audio_path,
                    caption=f"🎹 {res.title} — AI bed (vocals come next)", kind="audio")
                # The file landing is the delivery — no "sent …
                # to this chat" narration (presentation law).
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._control_music_bed", e)
                text += f"\naudio: {res.audio_path}"
        else:
            text += f"\naudio: {res.audio_path or '(render failed)'}"
        return text

    def _control_music_full(self, tail: str, chat_key: str = "") -> str:
        """ /music full <topic> [style] [voice_id] — the whole song.

        #57 ACE-Step bed + DiffSinger vocal + RVC voice → mixed master.
        Honest labeling: "AI-generated instrumental + AI vocals."
        Fails honestly when a stage can't run — never fake audio.
        """
        from ...media.music import STYLES
        from ...media.vocals import (
            RVCVoiceRegistry,
            VocalModelUnavailable,
            make_full_song,
            parse_full_request,
        )

        req = parse_full_request(f"/music full {tail}", STYLES)
        if req is None:
            return ("usage: /music full <topic> [style] [voice_id]\n"
                    "e.g. /music full lagos nights afrobeats owner-xtts\n"
                    "/music voices lists registered voices")
        # voice_id is the last word if it's a known voice, else default
        registry = RVCVoiceRegistry()
        words = req["topic"].split()
        voice_id = ""
        if words and words[-1] in registry.all():
            voice_id = words[-1]
            req["topic"] = " ".join(words[:-1]).strip() or "untitled"
        if not voice_id:
            default = registry.default()
            voice_id = default.voice_id if default is not None else ""
        try:
            res = make_full_song(req["topic"], style=req["style"],
                                 voice_id=voice_id, audience="private",
                                 context=self.context)
        except Exception as exc:  # noqa: BLE001 - both model errors
            return f"🎵 couldn't make the full song:\n{exc}"
        vlabel = voice_id or "tts vocals"
        text = (f"🎵 “{res.title}” — full song [{vlabel}]\n{res.note}")
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None and res.master_path:
            vocal_txt = "AI vocals" if res.vocal_path else "instrumental"
            try:
                self.gateway.send_file(
                    chat.platform, f"{chat.platform}:{chat.chat_id}",
                    res.master_path,
                    caption=f"🎵 {res.title} — AI instrumental + {vocal_txt}", kind="audio")
                # No "sent … to this chat" narration (presentation
                # law) — the file landing is the delivery.
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._control_music_full", e)
                text += f"\nmaster: {res.master_path}"
        else:
            text += f"\nmaster: {res.master_path or '(render failed)'}"
        return text

    def _control_music_chain(self, tail: str, chat_key: str = "") -> str:
        """ /music chain [lang:] <lyrics…> — the lyrics-first chain.

        Tone-aware draft melody from the owner's own lyrics (tonal
        G2P fixes the melody to the lyric tones), then ACE-Step bed →
        DiffSinger vocal → RVC timbre → master, with a licence-audited
        manifest. Stages without a provisioned backend are marked
        skipped with the reason — the MIDI draft, spec and manifest
        are real artifacts either way. Distinct from /music full
        (topic → composed song): the chain starts from supplied lyrics.
        """
        import re as _re
        import time as _time
        from pathlib import Path as _Path

        from ...media.musicgen_chain import ChainRequest, run_chain

        raw = (tail or "").strip()
        if not raw:
            return ("usage: /music chain [lang:] <lyrics…>\n"
                    "e.g. /music chain Mo dé, ayọ dé, ìfẹ́ ni mo ní\n"
                    "prefix a language as yo: / en: / ha: … (default yo "
                    "— the chain is tone-aware)")
        language = "yo"
        m = _re.match(r"^([a-z]{2}):\s*(.+)$", raw, _re.DOTALL)
        if m:
            language, raw = m.group(1), m.group(2).strip()
        if not raw:
            return "give me the lyrics: /music chain [lang:] <lyrics…>"
        title = _re.sub(r"[^\w\-]+", "-",
                        "-".join(raw.split()[:4]).lower()).strip("-") \
            or "chain"
        workdir = str(_Path("musicgen_out") / f"{title}-{int(_time.time())}")
        req = ChainRequest(title=title, lyrics=raw, language=language,
                           workdir=workdir)
        try:
            res = run_chain(req)
        except Exception as exc:  # noqa: BLE001
            return f"🎵 chain failed: {exc}"
        lines = [f"🎵 chain “{res.title}” — lyrics-first generation"]
        for s in res.stages:
            mark = {"real": "✅", "fake": "⚠️",
                    "skipped": "⏭"}.get(s.status, "·")
            detail = f" — {s.detail}" if s.detail else ""
            lines.append(f"  {mark} {s.name}: {s.status}{detail}")
        if res.note:
            lines.append(res.note)
        if res.error:
            lines.append(f"error: {res.error}")
        chat = self._ref_from_key(chat_key) if chat_key else None
        sent = False
        if chat is not None:
            for path, caption in (
                    (res.master_path, f"🎵 {res.title} — chain master"),
                    (res.midi_path,
                     f"🎼 {res.title} — draft melody (MIDI)"),
                    (res.manifest_path,
                     f"📋 {res.title} — chain manifest (licence audit)")):
                if path:
                    try:
                        self.gateway.send_file(
                            chat.platform,
                            f"{chat.platform}:{chat.chat_id}",
                            path, caption=caption)
                        sent = True
                    except Exception as e:
                        record_failure(
                            "runtime_media.RuntimeMediaMixin."
                            "_control_music_chain", e)
        if not sent:
            lines.append(
                f"artifacts in {workdir}: MIDI {res.midi_path or '—'}, "
                f"spec {res.spec_path or '—'}, manifest "
                f"{res.manifest_path or '—'}"
                + (f", master {res.master_path}" if res.master_path
                   else ""))
        return "\n".join(lines)

    def _control_music_voices(self, tail: str, chat_key: str = "") -> str:
        """ /music voices [add <id> <model.pth> <source>] — RVC voices. """
        from ...media.vocals import RVCVoiceRegistry, VocalModelUnavailable
        registry = RVCVoiceRegistry()
        words = (tail or "").split()
        if words and words[0].lower() == "add":
            if len(words) < 4:
                return ("usage: /music voices add <id> <model.pth> "
                        "<source>\nsource: xtts (private-only) | "
                        "chatterbox | rvc")
            try:
                v = registry.add(words[1], words[2], words[3])
            except VocalModelUnavailable as exc:
                return f"couldn't register the voice:\n{exc}"
            return (f"voice “{v.voice_id}” registered "
                    f"({v.source}, {v.license})")
        voices = registry.all()
        if not voices:
            return ("no RVC voices registered.\n"
                    "/music voices add <id> <model.pth> <source>")
        return ("RVC voices:\n" + "\n".join(
            f"  {v.voice_id:16s} {v.source:12s} {v.license}"
            for v in voices.values()))

    def _control_music(self, tail: str, chat_key: str = "") -> str:
        """ /music <topic> [style] — composes a real song and sends the
        audio + the score PDF (lead sheet) straight to this chat.
        /music styles | /music song [slug] | /music bed <topic> [style]
        | /music full <topic> [style] [voice_id] | /music voices
        | /music chain [lang:] <lyrics…> (lyrics-first generation chain)
        | /music draft <topic> [style] | /music revise <note>
        | /music perform | /music freestyle [seed] [bars N]."""
        from ...media.music import STYLES, MusicCreator

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /music <topic> [style]  |  /music styles  |  "
                    "/music song [slug]\nstyles: " + ", ".join(STYLES))
        words = tail.split()
        if words and words[0].lower() in ("sources", "enrich", "health", "radio", "registry-play"):
            # Legal music registry surface — licence always shown.
            # The runtime context rides along (registry health +
            # the wired play pipeline consult it) and, in chat, a
            # deliver_fn over the gateway so a fetched registry
            # track is SENT, not just described (doc 7).
            from ...music_sources.chat import control_music_sources
            _reg_chat = self._ref_from_key(chat_key) if chat_key else None
            _reg_deliver = None
            if _reg_chat is not None:
                def _reg_deliver(path: str, caption: str,
                                 _chat: Any = _reg_chat) -> tuple[bool, str]:
                    try:
                        res = self.gateway.send_file(
                            _chat.platform,
                            f"{_chat.platform}:{_chat.chat_id}",
                            path, caption=caption[:1024],
                            kind=_send_kind_for(path))
                        if res is not None and getattr(
                                res, "ok", True) is False:
                            return False, str(
                                getattr(res, "error", "")
                                or "send failed")
                        return True, ""
                    except Exception as exc:  # noqa: BLE001
                        return False, str(exc)
            return control_music_sources(
                tail, context=self.context, deliver_fn=_reg_deliver)
        if words[0].lower() == "styles":
            return ("styles:\n" + "\n".join(
                f"  {k:12s} {v.label}  ({v.mode}, {v.tempo[0]}-{v.tempo[1]} bpm)"
                for k, v in STYLES.items()))
        if words[0].lower() == "bed":
            return self._control_music_bed(" ".join(words[1:]), chat_key)
        if words[0].lower() == "full":
            return self._control_music_full(" ".join(words[1:]), chat_key)
        if words[0].lower() == "voices":
            return self._control_music_voices(" ".join(words[1:]), chat_key)
        if words[0].lower() == "chain":
            return self._control_music_chain(" ".join(words[1:]), chat_key)
        if words[0].lower() == "draft":
            return self._control_music_draft(" ".join(words[1:]), chat_key)
        if words[0].lower() == "revise":
            return self._control_music_revise(" ".join(words[1:]), chat_key)
        if words[0].lower() == "perform":
            return self._control_music_perform(chat_key)
        if words[0].lower() == "freestyle":
            return self._control_music_freestyle(" ".join(words[1:]), chat_key)
        if words[0].lower() == "song":
            from ...media.music import _saved_songs
            lookup = " ".join(words[1:]).strip()
            out = _saved_songs(self.context, lookup)
            if lookup:
                if out.get("found"):
                    s = out["song"]
                    return f"found “{s['title']}” → {s['midi']}\n/play it to play."
                names = ", ".join(x["title"] for x in out.get("songs", [])[:8])
                return (f"no saved song matching {lookup!r}. Saved: "
                        f"{names or '(none)'}")
            names = [x["title"] for x in out.get("songs", [])]
            return ("saved songs:\n" + "\n".join(f"  {n}" for n in names)
                    or "saved songs: (none)\n/music <topic> composes one.")
        topic = tail
        # trailing vocals flag: /music <topic> [style] [vocals|novocals]
        # (vocals are on by default — lightweight TTS hook, Termux-safe).
        # Strip the flag BEFORE style detection so "pop novocals" parses.
        style = "pop"
        with_vocals = True
        if words and words[-1].lower() in ("vocals", "novocals"):
            with_vocals = words[-1].lower() == "vocals"
            words = words[:-1]
        if words and words[-1].lower() in STYLES and len(words) > 1:
            style = words[-1].lower()
            topic = " ".join(words[:-1])
        else:
            topic = " ".join(words)
        if not topic:
            return "usage: /music <topic> [style] [vocals|novocals]"
        try:
            song = MusicCreator(self.context).compose(
                topic, style=style, with_vocals=with_vocals)
        except Exception as exc:  # noqa: BLE001
            return f"music error: {exc}"
        n_lines = sum(len(sec.lyrics) for sec in song.sections)
        text = (f"🎵 “{song.title}”  [{song.style}, {song.key}, {song.tempo} bpm]\n"
                f"{n_lines} lyric lines across {len(song.sections)} sections")
        if song.synth_backend:
            text += f"\nrendered with: {song.synth_backend}"
        if song.vocal_note:
            text += f"\n🎤 {song.vocal_note}"
        if song.synth_note:
            text += f"\n💡 {song.synth_note}"
        # deliver the audio + the written score to this chat
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None:
            sent = []
            try:
                if song.audio_path:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        song.audio_path,
                        caption=f"🎵 {song.title} — listen", kind="audio")
                    sent.append("audio")
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._control_music", e)
            try:
                if song.score_pdf_path:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        song.score_pdf_path,
                        caption=f"🎼 {song.title} — score (chords + lyrics)")
                    sent.append("score")
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._control_music:376", e)
            if sent:
                # The report must reflect whether vocals are in the
                # audio — never imply a full song when the vocal
                # track was skipped (vocals requested, none rendered).
                # The delivery itself needs no narration line.
                if with_vocals and not (song.vocal_path or "").strip():
                    text += ("\n⚠ instrumental only — the vocal track "
                             "was skipped")
            else:
                text += (f"\n audio: {song.audio_path or '(render failed)'}"
                         f"\n score: {song.score_pdf_path or '(render failed)'}")
        else:
            text += (f"\n audio: {song.audio_path or '(render failed)'}"
                     f"\n score: {song.score_pdf_path or '(render failed)'}"
                     f"\n midi: {song.midi_path or '(not written)'}")
        return text

    def _control_distribute(self, tail: str, chat_key: str = "") -> str:
        """ /distribute [song path] — the release walk-through (owner-only).

        Phase 1: Devon prepares everything (metadata, splits, AI
        disclosure, checklist) — the USER clicks submit on the
        distributor. Splits are mandatory and must sum to 100.
        """
        from ...finance.ledger import Ledger
        from ...media import distribute as _dist

        tail = (tail or "").strip()
        if tail.lower() in ("legal", "terms"):
            return ("**Legal weather:**\n" +
                    "\n".join(f"  {n}" for n in _dist.legal_notes()))
        song_path = tail
        if song_path and not os.path.isfile(song_path):
            return (f"audio not found: {song_path!r}\n"
                    "usage: /distribute [song .wav path]\n"
                    "or reply with the path. /distribute legal shows terms.")
        draft = _dist.start_draft(chat_key, song_path=song_path)
        return _dist.draft_prompt(draft)

    def _distribute_answer(self, chat_key: str, text: str) -> str | None:
        """Advance a pending /distribute draft. None when no draft pending."""
        from ...finance.ledger import Ledger
        from ...media import distribute as _dist

        draft = _dist.pending_draft(chat_key)
        if draft is None:
            return None
        nxt = _dist.advance_draft(draft, text)
        if nxt == "cancel":
            _dist.clear_draft(chat_key)
            return "📦 release cancelled — no packet built, nothing submitted."
        if nxt == "done":
            _dist.clear_draft(chat_key)
            try:
                packet = _dist.prepare(
                    draft.song_path, title=draft.title, artist=draft.artist,
                    platforms=draft.platforms, splits=draft.splits,
                    ledger=Ledger())
            except _dist.DistributionError as exc:
                return f"📦 couldn't build the packet:\n{exc}"
            return _dist.format_packet(packet)
        return nxt

    @staticmethod
    def _youtube_error(prefix: str, exc: Any, chat: Any) -> str:
        """Forced-YouTube failure with the cookies hint when the cause
        is a bot-check.  Chat users are on phones: they get the short
        /cookies pointer, never the yt-dlp command dump.  CLI users
        CAN run commands, so they get the full FAQ-grounded help."""
        from ...media.cookies import BOT_COOKIE_HELP, BOT_COOKIE_SHORT_HELP, is_bot_detection_error
        msg = f"{prefix}: {exc}"
        if is_bot_detection_error(exc):
            msg += f"\n{BOT_COOKIE_SHORT_HELP if chat is not None else BOT_COOKIE_HELP}"
        return msg

    def _control_play(self, tail: str, chat_key: str = "", *,
                      owner_key: str = "") -> str:
        """Play transport.  <query> is ONE thing — a path, a URL, or a
        song title — never whitespace-split into word-paths.

        In chat (chat_key set): the resolved audio file is SENT to the
        conversation via the gateway instead of playing locally — nobody
        hears mpv on the server.  On CLI (no chat_key): local playback
        via mpv/ffmpeg as before.

        /play <workspace path to audio/midi>
        /play <song title>   → dynamic strategy chain (SourceResolver):
                              local file → SoundCloud search → YouTube
                              search → Spotify (when linked)
        /play <any URL>     → yt-dlp universal extraction (1000+ sites),
                              SoundCloud API as metadata optimization
                              with automatic yt-dlp fallback
        /play spotify:<query>  → force Spotify (honest when not linked)
        /play youtube:<query|url> → force YouTube (needs yt-dlp)
        /play queue|status|pause|…   (transport actions)
        """
        # Pick-list owner binding (Living Chat P1): an explicit
        # owner wins; otherwise the dispatch's in-flight owner
        # (play_owner_var) applies; "" keeps the chat slot.
        owner_key = owner_key or play_owner_var.get("")
        # Intake command-claim key (dupfix4 D4): the dispatch's
        # in-flight key (play_claim_var) applies; "" means no
        # claim to re-check at send time.
        claim_key = play_claim_var.get("")
        from ...connectors.wiring import wire_streaming_adapters
        from ...media.playback import PlaybackEngine

        # chat never wired these before (only `nm music` did) — share the
        # same wiring so Spotify/SoundCloud work from the DM too.
        try:
            wire_streaming_adapters(self.context)
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._control_play", e)

        tail = (tail or "").strip()
        # Registry play path: /play registry <query> [--source <id>] —
        # search → download → enrich, licence always shown. Falls back
        # to the normal transport for every other tail.
        if tail.lower().startswith("registry ") or tail.lower() == "registry":
            from ...music_sources import play as _registry_play
            query = tail[len("registry"):].strip()
            source_id = None
            if "--source" in query:
                query, _, source_id = query.partition("--source")
                query = query.strip()
                source_id = (source_id.strip().split() or [None])[0]
            from ...music_sources import registry_bridge as _ms_bridge
            from ...music_sources.live import live_collaborators
            # Doc 7: wire the REAL search/download steps — the same
            # resolver sweep and playback-engine downloader ordinary
            # /play uses — instead of the helper's unwired defaults
            # (no_search / plan_only presented as results).
            _search_fn, _download_fn = live_collaborators(self.context)
            _result = _registry_play.play_via_registry(
                query, search_fn=_search_fn, download_fn=_download_fn,
                source_id=source_id,
                registry=_ms_bridge.music_registry_for_context(
                    self.context))
            _payload = _result.download.payload \
                if _result.download is not None else None
            if _result.status == "ready" \
                    and isinstance(_payload, dict) \
                    and _payload.get("path"):
                _chat = self._ref_from_key(chat_key) if chat_key else None
                if _chat is not None:
                    # The journey ends like ordinary /play ends: the
                    # verified file delivered to the chat as audio.
                    return self._play_deliver_fetched(
                        _chat, _result, _payload)
            return _registry_play.registry_play_text(_result)
        actions = {"add", "pause", "resume", "stop", "seek", "volume",
                   "next", "prev", "queue", "remove", "clear", "status",
                   "pick"}
        parts = tail.split(None, 1)
        first = parts[0].lower() if parts else ""
        if first in ("l", "album"):
            # /play L <artist> — a random album from the artist, every
            # track sent. A verb on /play, not a new command.
            try:
                return self._control_play_album(
                    parts[1].strip() if len(parts) > 1 else "",
                    chat_key, owner_key=owner_key,
                    claim_key=claim_key)
            except Exception as exc:  # noqa: BLE001
                return f"play error: {exc}"
        if first in actions:
            action, rest = first, (parts[1] if len(parts) > 1 else "")
        else:
            action, rest = "play", tail
        rest = (rest or "").strip()
        # forced-source prefixes: "spotify:<query>" / "youtube:<query|url>"
        forced: str = ""
        low_rest = rest.lower()
        if low_rest.startswith("spotify:") and not _looks_like_spotify_uri(rest):
            forced, rest = "spotify", rest.split(":", 1)[1].strip()
        elif low_rest.startswith("youtube:"):
            forced, rest = "youtube", rest.split(":", 1)[1].strip()
        try:
            engine = PlaybackEngine(self.context)
            if action == "pick":
                return self._play_pick(rest, chat_key)
            if action in ("play", "add"):
                if not rest:
                    st = engine.status()
                    return (f"queue ({st['queue']}): "
                            + (f"now {st['current']}" if st["queue"] else "empty")
                            + "\n/play <song title, path, or url> to queue and play")
                if forced == "spotify":
                    return self._play_forced_spotify(engine, rest)
                if forced == "youtube":
                    chat = self._ref_from_key(chat_key) if chat_key else None
                    try:
                        added = engine._add_youtube(rest)
                    except Exception as exc:  # noqa: BLE001
                        return self._youtube_error(
                            f"can't play {rest[:80]!r} from YouTube",
                            exc, chat)
                    if chat is not None and added:
                        yt_title = str(added[0].get("title") or "audio")
                        return self._play_download_bg(
                            chat, chat_key, f"⏳ {yt_title}",
                            lambda: self._play_send_in_chat_result(
                                engine, added[0], chat),
                            title=yt_title, claim_key=claim_key)
                    # CLI: play the item _add_youtube just enqueued.
                    # Do NOT call engine.play_youtube(rest) — it
                    # re-adds the same video, double-enqueuing it.
                    try:
                        q = engine.queue()
                        n = len(q) if isinstance(q, list) else 0
                        res = engine.play(n - 1 if n else 0)
                    except Exception as exc:  # noqa: BLE001
                        return self._youtube_error(
                            f"can't play {rest[:80]!r} from YouTube",
                            exc, chat)
                    return self._fmt_started(res, engine)
                # Dynamic resolution: the strategy chain in
                # nomorals/media/resolver.py tries local file → yt-dlp
                # (universal) → SoundCloud API (with yt-dlp fallback) →
                # text search, recording every attempt.  Never raises.
                from ...media.picklist import (
                    PICK_LIMIT,
                    PickCache,
                    format_picklist,
                    is_specific,
                    parse_title_artist,
                    search_query_for,
                )
                from ...media.resolver import SourceResolver
                low_r = rest.lower()
                _is_urlish = low_r.startswith(("http://", "https://",
                                               "spotify:"))
                _is_pathish = ("/" in rest or "\\" in rest) and bool(
                    re.search(r"\.(mp3|wav|flac|ogg|m4a|mid|midi)$", rest, re.I))
                chat = self._ref_from_key(chat_key) if chat_key else None
                # Version requests (/play <song> reverb|nightcore|
                # slowed|sped up, or /play reverb <song>): the version
                # words come off the query — the search runs on the
                # clean song, and the version rides into the
                # background job. A requested version is the answer,
                # never a fallback.
                version, clean_rest = self._play_parse_version(rest)
                if (chat is not None and not _is_urlish and not _is_pathish
                        and not is_specific(clean_rest)):
                    # Vague query in chat → old-school pick-list instead of
                    # auto-downloading the top guess.  Never raises.
                    try:
                        candidates = SourceResolver(
                            self.context).search_candidates(
                                clean_rest, limit=PICK_LIMIT)
                    except Exception as e:
                        record_failure("runtime_media.RuntimeMediaMixin._control_play:579", e)
                        candidates = []
                    if candidates:
                        token = PickCache.store(
                            candidates, chat_key or "",
                            owner_key=owner_key)
                        if token:
                            return format_picklist(
                                candidates, clean_rest, token)
                    # No candidates (or cache hiccup) → fall through to
                    # the normal auto-resolve so the user still gets an
                    # honest tried-list instead of silence.
                title_q, artist_q = parse_title_artist(clean_rest)
                search_q = search_query_for(title_q, artist_q) or clean_rest
                # Step 4 Ekiti lexicon: search-normalize Ekiti → standard
                # Yoruba for source queries; the ORIGINAL stays in `rest`
                # for display/TTS.  Never raises.
                try:
                    from ...media.source_registry import EkitiLexicon
                    search_q = EkitiLexicon().normalize_query(
                        search_q)["search"] or search_q
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._control_play:599", e)
                # Measured progress (P1): in chat, the search is the
                # first stage of the operation the owner is about to
                # watch — the tracker created here rides into the
                # background job below, so the ack edits continue
                # this same event stream end to end.
                _play_tracker = None
                if action == "play" and chat is not None:
                    from ...core.ids import new_short_id
                    from ...core.progress_events import SEARCHING, ProgressTracker
                    _play_tracker = ProgressTracker(
                        op_id=new_short_id("dl"), title=rest)
                    _play_tracker.set_stage(
                        SEARCHING, note="looking for the track")
                if action == "play" and chat is not None:
                    # ACK FIRST (2026-11-21 wedge): the chat path used
                    # to run resolve() + the full candidate sweep
                    # SYNCHRONOUSLY before the first ack — two
                    # sequential full-chain sweeps, 30s HTTP legs and
                    # 120s yt-dlp legs, no budget — so /play took
                    # minutes to even get processed.  The ack goes out
                    # now, on the owner's own words; the resolve +
                    # sweep + send run on the background worker behind
                    # it (:meth:`_play_resolve_and_send`), and the ack
                    # is edited to the outcome.  The pick-list path
                    # above (vague query) already returned; the add
                    # action and the CLI path keep the synchronous
                    # flow below — they need the queue receipt, not a
                    # download.
                    return self._play_download_bg(
                        chat, chat_key, f"⏳ {rest}",
                        lambda: self._play_resolve_and_send(
                            engine, search_q, rest, title_q, artist_q,
                            chat,
                            sweep=(not _is_urlish and not _is_pathish),
                            version=version),
                        title=rest, tracker=_play_tracker,
                        claim_key=claim_key)
                resolution = SourceResolver(self.context).resolve(search_q)
                if not resolution.ok:
                    # Step 6: never a dead end — tried + why + hint via
                    # PlayService (pure formatting, no extra network).
                    try:
                        from ...media.play_service import PlayService
                        detail = PlayService(
                            self.context).explain_failure(resolution)
                        # explain_failure already includes "hint: ..." —
                        # appending resolution.hint again would print the
                        # hint body twice.
                        hint = ""
                    except Exception as e:
                        record_failure("runtime_media.RuntimeMediaMixin._control_play:613", e)
                        detail = ("tried: "
                                  + ("; ".join(resolution.attempts)
                                     or "no strategies"))
                        hint = (f"\n{resolution.hint}"
                                if resolution.hint else "")
                    msg = f"couldn't find {rest[:80]!r} —\n{detail}{hint}"
                    if chat is not None:
                        # Chat users are on phones: the resolver's
                        # full yt-dlp command dump is not actionable
                        # there — shorten to the /cookies pointer.
                        from ...media.cookies import BOT_COOKIE_HELP as _FULL_HELP
                        from ...media.cookies import BOT_COOKIE_SHORT_HELP as _SHORT_HELP
                        msg = msg.replace(_FULL_HELP, _SHORT_HELP)
                    return msg
                if version and chat is None:
                    # CLI: no chat to send to — download the winner,
                    # render the version, queue the render for local
                    # playback. Never silently play the original.
                    return self._play_cli_version(
                        engine, version, resolution, rest,
                        title_q, artist_q)
                added = []
                for r in (resolution, *resolution.extra_tracks):
                    added.append(engine._enqueue(
                        r.path_or_url, r.kind, r.title or r.path_or_url,
                        artist=r.artist, duration=r.duration))
                queue_len = len(engine.queue())
                out = (f"queued {len(added)} (queue {queue_len}):\n"
                       + "\n".join(f"  - {a.get('title') or a['path']}"
                                   for a in added))
                # The tried trail goes to the journal only
                # (presentation law, 2026-11-05) — attempts record
                # every strategy in order (failed + the winner), and
                # the durable copy is the log, never a chat receipt.
                self._play_log_tried(resolution.attempts)
                if action == "play":
                    st = engine.play()
                    if st.get("status") == "playing":
                        out += f"\n▶ playing “{st.get('current', '')}” " \
                               f"[{st.get('backend')}]"
                    else:
                        out += (f"\nstatus: {st.get('status')} "
                                f"{st.get('error') or st.get('hint', '')}")
                elif chat is not None:
                    # /play add in chat: the titles, no queue-count
                    # receipt (presentation law) — the counts form is
                    # the CLI transport readout only.
                    return ("added to the queue:\n" + "\n".join(
                        f"  - {a.get('title') or a['path']}"
                        for a in added))
                return out
            if action == "seek":
                if not rest:
                    return ("usage: /play seek <seconds> — jump to a "
                            "position in the current track.")
                try:
                    secs = float(rest.split()[0])
                except (TypeError, ValueError):
                    return (f"“{rest.split()[0]}” isn't a number — "
                            "usage: /play seek <seconds>.")
                return f"seeked to {rest.split()[0]}s: {engine.seek(secs)}"
            if action == "volume":
                if not rest:
                    return ("usage: /play volume <0-100> — set the "
                            "playback volume.")
                try:
                    level = int(rest.split()[0])
                except (TypeError, ValueError):
                    return (f"“{rest.split()[0]}” isn't a number — "
                            "usage: /play volume <0-100>.")
                return f"volume: {engine.volume(level)}"
            if action == "remove":
                if not rest:
                    return ("usage: /play remove <number> — remove a "
                            "track from the queue (/play queue to see "
                            "numbers).")
                try:
                    idx = int(rest.split()[0])
                except (TypeError, ValueError):
                    return (f"“{rest.split()[0]}” isn't a number — "
                            "usage: /play remove <number>.")
                try:
                    return f"removed: {engine.remove(idx)}"
                except Exception as exc:  # noqa: BLE001
                    return f"can't remove item {idx}: {exc}"
            if action == "queue":
                items = engine.queue()
                return (f"queue ({len(items)}):\n"
                        + "\n".join(f"  {i:2d}. {it['title']} [{it['kind']}]"
                                    for i, it in enumerate(items))
                        or "queue (0): empty")
            if action in ("pause", "resume", "stop", "next", "prev", "clear"):
                return f"{action}: {getattr(engine, action)()}"
            st = engine.status()
            return (f"backend: {st['backend'].get('name') if isinstance(st['backend'], dict) else st['backend']}\n"
                    f"playing: {st['playing']}  paused: {st['paused']}\n"
                    f"current: {st['current'] or '(none)'} "
                    f"[{st['position']}/{st['queue']} in queue]\n"
                    f"volume: {st['volume']}")
        except Exception as exc:  # noqa: BLE001
            return f"play error: {exc}"

    def _control_play_album(self, artist: str, chat_key: str = "", *,
                            owner_key: str = "", claim_key: str = "") -> str:
        """``/play L <artist>`` — pick a RANDOM album from the artist,
        download every track in it, send them in album order.

        Chat: one background job behind one ack (the same
        :meth:`_play_download_bg` seam a single /play uses) — the
        album search + pick + per-track send all run on the worker
        thread, and the ack is edited to the final summary (album,
        how many tracks landed, which are missing and why). CLI (no
        chat_key): synchronous — resolve each track, queue it, play.
        Never raises.
        """
        from ...core.ids import new_short_id
        from ...core.progress_events import SEARCHING, ProgressTracker
        from ...media.play_album import (
            find_album_for_artist,
            format_album_summary,
        )
        from ...media.playback import PlaybackEngine

        artist = (artist or "").strip()
        if not artist:
            return ("usage: /play L <artist> — pick a random album "
                    "from the artist and send every track in it.")
        chat = self._ref_from_key(chat_key) if chat_key else None
        try:
            engine = PlaybackEngine(self.context)
        except Exception as exc:  # noqa: BLE001
            return f"play error: {exc}"
        if chat is None:
            # CLI: queue the whole album, then play it.
            result = find_album_for_artist(artist)
            if result.error or result.album is None:
                return result.error or f"no albums found for {artist!r}."
            delivered: list[str] = []
            missing: list[tuple[str, str]] = []
            for track in result.tracks:
                _t, sent = self._play_album_track_cli(
                    engine, result.album.artist, track.title)
                if sent:
                    delivered.append(track.title)
                else:
                    missing.append((track.title, _t))
            summary = format_album_summary(
                result.album, delivered, missing)
            if delivered:
                try:
                    st = engine.play()
                    if st.get("status") == "playing":
                        summary += (f"\n▶ playing "
                                    f"“{st.get('current', '')}” "
                                    f"[{st.get('backend')}]")
                    else:
                        summary += (f"\nstatus: {st.get('status')} "
                                    f"{st.get('error') or st.get('hint', '')}")
                except Exception as exc:  # noqa: BLE001
                    summary += f"\nplay error: {exc}"
            return summary
        tracker = ProgressTracker(op_id=new_short_id("dl"), title=artist)
        tracker.set_stage(SEARCHING, note="finding a random album")
        return self._play_download_bg(
            chat, chat_key, f"⏳ {artist} — finding a random album…",
            lambda: self._play_album_send(engine, artist, chat),
            title=f"album: {artist}", tracker=tracker,
            claim_key=claim_key)

    def _play_album_send(self, engine: Any, artist: str,
                         chat: Any) -> tuple[str, bool]:
        """Background body for /play L: find the album, send every
        track in album order through the same fall-through a single
        /play uses per track. Returns (summary, any_sent) — the ack
        is edited to the summary. Never raises."""
        from ...media.play_album import (
            find_album_for_artist,
            format_album_summary,
        )

        try:
            result = find_album_for_artist(artist)
        except Exception as exc:  # noqa: BLE001 - find never raises,
            # belt and braces: the ack must land as an honest line
            return f"couldn't pick an album for {artist!r}: {exc}", False
        if result.error or result.album is None:
            return (result.error or f"no albums found for {artist!r}.",
                    False)
        album = result.album
        delivered: list[str] = []
        missing: list[tuple[str, str]] = []
        for track in result.tracks:
            text, sent = self._play_album_track(
                engine, album.artist, track.title, chat)
            if sent:
                delivered.append(track.title)
            else:
                missing.append((track.title, text))
        return format_album_summary(album, delivered, missing), \
            bool(delivered)

    def _play_album_track(self, engine: Any, artist: str, title: str,
                          chat: Any) -> tuple[str, bool]:
        """Send ONE album track through the single-/play fall-through:
        resolve → candidate sweep → fall-through send with a
        (title, artist) match gate. Returns (text, sent); the text is
        the delivered line on success, a short failure reason on
        failure. Never raises."""
        from ...media.picklist import search_query_for
        from ...media.resolver import SourceResolver

        search_q = search_query_for(title, artist) \
            or f"{artist} {title}"
        try:
            from ...media.source_registry import EkitiLexicon
            search_q = EkitiLexicon().normalize_query(
                search_q)["search"] or search_q
        except Exception as e:  # noqa: BLE001
            record_failure("runtime_media.RuntimeMediaMixin"
                           "._play_album_track:lexicon", e)
        try:
            resolution = SourceResolver(self.context).resolve(search_q)
        except Exception as exc:  # noqa: BLE001 - resolve never
            # raises, belt and braces
            return f"couldn't find {title!r}: {exc}", False
        if not resolution.ok:
            self._play_log_tried(resolution.attempts)
            return f"couldn't find {title!r}", False
        try:
            cands, detailed = SourceResolver(
                self.context).search_candidates_detailed(search_q)
        except Exception as e:  # noqa: BLE001
            record_failure("runtime_media.RuntimeMediaMixin"
                           "._play_album_track:candidates", e)
            cands, detailed = [], []
        fallbacks = [
            {"kind": c.kind, "path": c.path_or_url,
             "title": c.title or c.path_or_url,
             "artist": c.artist or "",
             "duration": c.duration or 0.0,
             "downloadable": c.downloadable,
             "block_reason": str(
                 getattr(c, "block_reason", "") or ""),
             "preview_only": bool(
                 getattr(c, "preview_only", False))}
            for c in (cands or [])]
        first_item = dict(engine._enqueue(
            resolution.path_or_url, resolution.kind,
            resolution.title or resolution.path_or_url,
            artist=resolution.artist, duration=resolution.duration))
        # The chain winner can carry the raw search query as its
        # "title" — the candidate sweep has the SAME track with its
        # real metadata; prefer it (same fix as single /play).
        for c in (cands or []):
            if c.kind == first_item.get("kind") \
                    and c.path_or_url == first_item.get("path"):
                cur = str(first_item.get("title") or "")
                if c.title and (not cur
                                or cur.strip().lower() in (
                                    search_q.strip().lower(),
                                    f"{title} {artist}".strip().lower())):
                    first_item["title"] = c.title
                if c.artist and not first_item.get("artist"):
                    first_item["artist"] = c.artist
                break
        first_item["downloadable"] = resolution.downloadable
        first_item["block_reason"] = str(
            getattr(resolution, "block_reason", "") or "")
        first_item["preview_only"] = bool(
            getattr(resolution, "preview_only", False))
        merged = self._merge_search_attempts(
            resolution.attempts, detailed)
        # Preview gate (same law as single /play): the best every
        # source has is a 30s preview and no full-track candidate
        # exists — drop the queue row and name the miss, never send
        # a preview as the track.
        if self._play_is_preview_item(first_item) \
                and not any(
                    fb.get("downloadable") is not False
                    and not self._play_is_preview_item(fb)
                    for fb in fallbacks):
            self._play_log_tried(merged)
            self._drop_matching_queued(engine, first_item)
            return f"only a 30s preview exists for {title!r}", False
        text, sent = self._play_send_with_fallthrough(
            engine, first_item, fallbacks, merged, chat,
            match=(title, artist), query=f"{artist} {title}")
        if sent:
            return text, True
        # The fall-through's failure text is multi-line; the album
        # summary carries the first line, truncated there.
        first_line = (text or "").strip().splitlines()
        return (first_line[0][:120] if first_line
                else f"couldn't download {title!r}"), False

    def _play_album_track_cli(self, engine: Any, artist: str,
                              title: str) -> tuple[str, bool]:
        """CLI half of one album track: resolve + enqueue, no send.
        Returns (reason, queued). Never raises."""
        from ...media.picklist import search_query_for
        from ...media.resolver import SourceResolver

        search_q = search_query_for(title, artist) \
            or f"{artist} {title}"
        try:
            from ...media.source_registry import EkitiLexicon
            search_q = EkitiLexicon().normalize_query(
                search_q)["search"] or search_q
        except Exception as e:  # noqa: BLE001
            record_failure("runtime_media.RuntimeMediaMixin"
                           "._play_album_track_cli:lexicon", e)
        try:
            resolution = SourceResolver(self.context).resolve(search_q)
        except Exception as exc:  # noqa: BLE001
            return f"couldn't find {title!r}: {exc}", False
        if not resolution.ok:
            return f"couldn't find {title!r}", False
        try:
            engine._enqueue(
                resolution.path_or_url, resolution.kind,
                resolution.title or resolution.path_or_url,
                artist=resolution.artist,
                duration=resolution.duration)
        except Exception as exc:  # noqa: BLE001
            return f"couldn't queue {title!r}: {exc}", False
        return "", True

    def _play_pick(self, rest: str, chat_key: str = "") -> str:
        """`/play pick <token> <n>` — download+send pick-list choice n.

        The token comes from a Telegram inline button, a WhatsApp
        list-row tap, or the WhatsApp number-reply hook.  Single-use:
        the pick-list is consumed on a successful pick.  Never raises.
        """
        from ...media.picklist import PickCache
        from ...media.playback import PlaybackEngine

        parts = (rest or "").split()
        if len(parts) < 2:
            return ("usage: /play pick <token> <number> — or just reply "
                    "with the number after /play shows the list.")
        token, num_s = parts[0].strip().lower(), parts[1].strip()
        try:
            n = int(num_s)
        except (TypeError, ValueError):
            return f"“{num_s}” isn't a track number — reply with a number from the list."
        # Peek, don't consume: the list is consumed on a SUCCESSFUL
        # pick.  Consuming up front loses the whole list when the
        # download fails, forcing the user to re-run /play (and
        # re-search every source) just to try another number.
        entry = PickCache.get(token)
        if entry is None:
            return _pick_expired_reply()
        if PickCache.is_superseded(token):
            # A newer list replaced this one in the chat: a tap on
            # the stale card must not download from a list the
            # chat has moved past (Living Chat P1 targeting).
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                "that pick-list has been replaced by a newer one "
                "— pick from the latest list 🙂",
                [[("🎵 Search again", "play")]])
        cands = entry.candidates or []
        if n < 1 or n > len(cands):
            return (f"pick a number 1–{len(cands)} — “{num_s}” is out of range.")
        cand = cands[n - 1]
        chat = self._ref_from_key(chat_key) if chat_key else None
        try:
            engine = PlaybackEngine(self.context)
            item = engine._enqueue(
                cand.path_or_url, cand.kind,
                cand.title or cand.path_or_url,
                artist=cand.artist or "", duration=cand.duration or 0.0)
        except Exception as exc:  # noqa: BLE001
            return f"couldn't queue “{cand.title}”: {exc}"
        out = f"picked {n}/{len(cands)}:\n  - {cand.title}"
        if chat is not None:
            # Background: the pick is consumed on the job thread ONLY
            # when the file actually went out (on_sent) — a failed
            # download leaves the list alive for another try, exactly
            # like the old synchronous peek-don't-consume rule.
            return self._play_download_bg(
                chat, chat_key, f"⏳ {cand.title}",
                lambda: self._play_send_in_chat_result(
                    engine, item, chat),
                on_sent=lambda: PickCache.consume(token),
                title=str(cand.title or ""))
        try:
            st = engine.play()
            if st.get("status") == "playing":
                out += f"\n▶ playing “{st.get('current', '')}”"
                PickCache.consume(token)
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._play_pick", e)
        return out

    def _play_pick_hook(self, message: ChatMessage) -> str | None:
        """Bare number reply after a /play pick-list → download that track.

        Old-school WhatsApp-bot style: the pick-list says "reply with the
        number", the user replies "3", this resolves it through the same
        ``/play pick`` path as the Telegram buttons.  Returns a reply
        string when a pending pick-list and a bare number both match,
        else None — EXCEPT when the chat HAD a pick-list that is now
        gone: then it answers honestly ("that pick-list expired")
        instead of silence (2026-11-12: the owner's "1" five minutes
        after the list met nothing at all).  A bare number in a chat
        that never had a pick-list still returns None and the turn
        flows on untouched.  Never raises.
        """
        from ...games.players import canonical_whatsapp_sender_id
        from ...media.picklist import PICK_LIMIT, PickCache

        text = (message.text or "").strip()
        if not text.isdigit():
            return None
        n = int(text)
        if n < 1 or n > PICK_LIMIT:
            return None
        # Same owner derivation as _pick_owner_key (runtime.py):
        # WhatsApp reports one human as phone JID or LID "in
        # either order", so the raw sender id is folded to the
        # canonical phone form — otherwise a reply arriving on
        # the other form misses a live pick and meets silence.
        meta = getattr(message, "meta", None)
        alt_id = ""
        if isinstance(meta, dict):
            alt_id = str(meta.get("sender_alt_id") or "").strip()
        sender_id = str(getattr(message, "sender_id", "") or "")
        owner_key = (canonical_whatsapp_sender_id(sender_id, alt_id)
                     or str(getattr(message, "sender", "") or ""))
        found = PickCache.get_for_chat(message.chat.key, owner_key)
        if found is None:
            if PickCache.had_expired_pick(message.chat.key, owner_key):
                return _pick_expired_reply()
            return None
        token, _entry = found
        try:
            return self._control_play(f"pick {token} {n}", message.chat.key,
                                      owner_key=owner_key)
        except Exception as exc:  # noqa: BLE001 — _control_play
            # shouldn't raise, belt and braces
            return f"couldn't pick track {n}: {exc}"

    @staticmethod
    def _play_query(context: Any, raw: str) -> str:
        """One query from a possibly-messy tail.

        Pasted help prose ("queue it with: /play <path>", multi-line
        text) collapses to the actual query: the last non-empty line,
        stripped of quotes and a leading /play.  URLs and existing paths
        pass through untouched; bare titles go to workspace/SoundCloud
        resolution in :func:`_resolve_play_title`.
        """
        import re
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # drop pure-prose lines ("queue it with:", "here's the song:")
        query_lines = [ln for ln in lines
                       if not re.match(r"^[a-z ]{1,40}:$", ln, re.I)]
        text = query_lines[-1] if query_lines else raw.strip()
        text = re.sub(r"^/play\s+", "", text, flags=re.I).strip()
        text = text.strip("\"“”'").strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered.startswith(("http://", "https://", "spotify:")):
            return text
        # an existing path (workspace-relative or absolute) passes through
        if re.search(r"\.(mp3|wav|flac|ogg|m4a|mid|midi)$", text, re.I) \
                and ("/" in text or "\\" in text):
            return text
        return _resolve_play_title(context, text)

    def _play_forced_spotify(self, engine: Any, query: str) -> str:
        """`/play spotify:<query>` — Spotify only, honest when not linked."""
        from ...connectors.wiring import spotify_connector, spotify_link_help, spotify_linked
        if not query:
            return "play spotify:<what> — give me a song or artist to search."
        if not spotify_linked(self.context):
            return spotify_link_help()
        try:
            res = engine.play_spotify(query)
        except Exception as exc:  # noqa: BLE001 - adapter's own message
            return f"Spotify couldn't play {query[:80]!r}: {exc}"
        return self._fmt_started(res, engine)

    def _download_produce(self, url: str, audio_only: bool,
                          chat: Any, swarm: bool = False,
                          *, want: str = "video",
                          quality: str = "best") -> tuple[str, bool]:
        """The /download download+send body — runs on the job worker
        thread in chat mode, synchronously on CLI.  Returns (reply
        text, sent): ``sent`` is True only when the file actually
        went out.  Never raises.

        Cobalt (52-websites #1): this body is the uniform
        ``fetch_media`` contract, not a second produce path — the
        chain, its verification and its metadata all live in
        :mod:`nomorals.media.downloader`; this method only decides
        what goes out in chat and how the outcome reads.  ``want``
        is video | audio | image | gif; ``quality`` one of the
        Cobalt labels (video only).
        """
        from ...core.jobs import report_progress
        from ...media.downloader import fetch_media

        if audio_only and want == "video":
            want = "audio"
        report_progress(f"downloading {url[:80]}")
        result = fetch_media(url, want=want, quality=quality,
                             context=self.context, swarm=swarm)
        if not result.ok:
            # The refusal is the canonical account: the service and
            # the reason, in plain words — never a bare "failed".
            lines = [f"couldn't download {url[:80]}:",
                     f"  {result.likely_cause}"]
            if result.hint:
                lines.append(f"  {result.hint}")
            if result.stages:
                lines.append(f"  tried: {', '.join(result.stages)}")
            return "\n".join(lines), False

        path = result.path
        title = result.title or "download"
        kind = result.kind or ("audio" if audio_only else "video")
        emoji = {"audio": "🎵", "video": "🎬", "image": "🖼️"}.get(kind, "🎬")
        if chat is not None:
            try:
                from ...core.progress_events import UPLOADING, emit_stage
                emit_stage(UPLOADING, note="sending the file")
                self.gateway.send_file(
                    chat.platform, f"{chat.platform}:{chat.chat_id}",
                    path, caption=f"{emoji} {title}"[:1024],
                    kind=kind)
                # The item's own line is the final state — no size
                # receipt, no "sent to this chat" (presentation law).
                return f"{emoji} {title}", True
            except Exception as exc:  # noqa: BLE001
                return (f"downloaded “{title}” but couldn't send it: "
                        f"{exc} (it's saved on the server at {path})"), False
        return f"{emoji} downloaded: {path}", False

    def _control_download(self, tail: str, chat_key: str = "") -> str:
        """Cobalt (52-websites #1): /download <url> [want] [quality].

        Paste a link, get the file — video, audio-only, GIF or photo,
        at the quality named.  YouTube, TikTok, Instagram, X/Twitter,
        SoundCloud, direct MP3/MP4 links — yt-dlp handles 1000+ sites
        as the universal extractor, with the proxy-lab and browser
        stages as fallback when direct fails.  The fetch itself is
        the uniform :func:`nomorals.media.downloader.fetch_media`
        contract; this handler only parses the request and routes it.

        /download <url>               → best quality video (or whatever the URL is)
        /download <url> audio         → audio only (mp3)
        /download <url> gif           → first 10 seconds as a GIF (needs ffmpeg)
        /download <url> image         → the photo (a direct image, or the
                                         poster's thumbnail for a video link)
        /download <url> 720p          → video at 720p (best | 1080p | 720p |
                                         480p | 360p)
        /download <url> audio 720p    → audio (quality is video-only)
        /download <url> --swarm       → parallel segmented fetch for a
                                    large file — a direct link, or a
                                    page/video link whose file URL the
                                    extractor resolves first (byte-range
                                    workers, resume, hash check);
                                    falls back to the single stream
                                    when the server will not serve
                                    ranges
        /download swarm          → the swarm queue's state (queued /
                                    running / done / failed, with
                                    segment counts)
        /download swarm cancel <job-id|title> → stop a queued or
                                    running swarm job (part files
                                    discarded, a queued job never
                                    runs)

        The platform routes (TikTok, Instagram) run only for the
        plain video/audio request they were built for — a format or
        quality request goes through the uniform contract, which
        honours the want instead of silently dropping it.  Spotify
        always takes its identify-and-match route (its descriptor
        refuses downloads; the match is the product).

        In chat the download runs as a background job (owner decision
        2026-10-19): the handler returns an ack immediately and the
        file is sent to the chat when the download lands.
        """
        tail = (tail or "").strip()
        if not tail:
            return ("usage: /download <url> [video|audio|image|gif] "
                    "[best|1080p|720p|480p|360p] [--swarm]\n"
                    "paste any media URL — YouTube, TikTok, Instagram, X, "
                    "SoundCloud, direct mp3/mp4 links, 1000+ sites via yt-dlp.\n"
                    "/download status — the source registry's growth "
                    "state (seeds, grown sources, health, candidates).\n"
                    "/download swarm — the download swarm's queue "
                    "(large files fetched in parallel segments).\n"
                    "/download swarm cancel <job-id> — stop a "
                    "queued or running swarm job.")
        words = tail.split()
        if words and words[0].lower() == "cancel":
            return self._cancel_download_job(
                words[1] if len(words) > 1 else "", chat_key)
        if words and words[0].lower() == "status":
            return self._download_sources_status()
        if words and words[0].lower() == "swarm" and len(words) == 1:
            return self._download_swarm_status()
        if words and len(words) >= 2 and words[0].lower() == "swarm" \
                and words[1].lower() == "cancel":
            return self._download_swarm_cancel(" ".join(words[2:]))
        lowered = [w.lower() for w in words]
        audio_only = any(w in ("audio", "--audio", "-a") for w in lowered)
        swarm = any(w in ("swarm", "--swarm") for w in lowered)
        want = "video"
        if "gif" in lowered:
            want = "gif"
        elif any(w in ("image", "photo", "picture") for w in lowered):
            want = "image"
        elif audio_only:
            want = "audio"
        quality = "best"
        for label in ("1080p", "720p", "480p", "360p"):
            if label in lowered:
                quality = label
                break
        if "best" in lowered:
            quality = "best"
        url = " ".join(w for w in words
                       if w.lower() not in (
                           "audio", "--audio", "-a", "video",
                           "image", "photo", "picture", "gif",
                           "best", "1080p", "720p", "480p", "360p",
                           "swarm", "--swarm")).strip()
        if not url:
            return ("usage: /download <url> [video|audio|image|gif] "
                    "[best|1080p|720p|480p|360p]")
        if not url.lower().startswith(("http://", "https://")):
            return (f"that doesn't look like a URL: {url[:80]!r}\n"
                    "paste the full link starting with https://")

        # Platform routing (FEI gap wave A): one surface, the
        # platform-shaped route per link — Spotify identifies and
        # matches (its own audio is previews; the chain's descriptor
        # gate would only refuse), TikTok gets the watermark-honest
        # route, Instagram the login-wall-aware one.  The platform
        # routes run only for the plain request they were built for;
        # a format or quality request takes the uniform contract
        # below, which honours the want instead of dropping it.
        from ...media.platforms import detect_platform
        _platform = detect_platform(url)
        if _platform == "spotify":
            return self._control_spotify(url, chat_key)

        plain_request = (want in ("video", "audio")
                         and quality == "best")
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None:
            if _platform == "tiktok" and plain_request:
                return self._play_download_bg(
                    chat, chat_key, "⏳ fetching that TikTok…",
                    lambda: self._tiktok_produce(url, audio_only, chat),
                    title=url[:80])
            if _platform == "instagram" and want == "video" \
                    and quality == "best":
                return self._play_download_bg(
                    chat, chat_key, "⏳ fetching that Instagram post…",
                    lambda: self._instagram_produce(url, chat),
                    title=url[:80])
            return self._play_download_bg(
                chat, chat_key, f"⏳ {url[:60]}",
                lambda: self._download_produce(url, audio_only, chat,
                                               swarm=swarm, want=want,
                                               quality=quality),
                title=url[:80])
        if _platform == "tiktok" and plain_request:
            text, _sent = self._tiktok_produce(url, audio_only, None)
            return text
        if _platform == "instagram" and want == "video" \
                and quality == "best":
            text, _sent = self._instagram_produce(url, None)
            return text
        text, _sent = self._download_produce(url, audio_only, None,
                                             swarm=swarm, want=want,
                                             quality=quality)
        return text

    def _download_swarm_status(self) -> str:
        """/download swarm — the download swarm's durable queue.

        The queue is the record of large-file swarm jobs (queued /
        running / done / failed, with segment counts); a job
        interrupted mid-download keeps its segment state beside the
        file, so the next run resumes it instead of restarting.
        """
        from ...media.download_swarm import SwarmQueue

        try:
            return SwarmQueue(context=self.context).status_text()
        except Exception as exc:  # noqa: BLE001 — status never raises
            return f"couldn't read the swarm queue: {exc}"

    def _download_swarm_cancel(self, arg: str) -> str:
        """/download swarm cancel <job-id|title> — stop a swarm job.

        Queued jobs never run; running jobs stop at their next
        segment boundary and their part files are discarded, never
        left as resume bait.  Resolution is by the queue's own ids
        (``sw-…``) or the file name, exact first, then a unique
        substring — the same shape as /download cancel's title
        resolution, but over the swarm queue (which is global, not
        per-chat; the whole /download surface is owner-only, so no
        chat-ownership check applies here).  Unknown and finished
        ids refuse in words.  Never raises.
        """
        from ...media.download_swarm import SwarmQueue

        arg = (arg or "").strip()
        if not arg:
            return ("usage: /download swarm cancel <job-id> — name "
                    "the job the way /download swarm lists it (its "
                    "id, or the file name), and I'll stop it.")
        try:
            queue = SwarmQueue(context=self.context)
        except Exception as exc:  # noqa: BLE001 — cancel never raises
            return f"couldn't read the swarm queue: {exc}"
        job, ambiguous = self._resolve_swarm_job(queue, arg)
        if ambiguous:
            return (f"more than one swarm job matches “{arg[:60]}” — "
                    "name the job id from /download swarm.")
        if job is None:
            return (f"no swarm job matching “{arg[:60]}” — it may "
                    "already have finished, or the id is wrong.")
        job_id = str(job.get("id") or "")
        status = str(job.get("status") or "")
        if status in ("done", "failed"):
            return (f"swarm job {job_id} already finished ({status}) "
                    "— nothing left to cancel.")
        if status == "cancelled":
            return f"swarm job {job_id} is already cancelled."
        outcome = queue.cancel(job_id)
        if outcome == "cancelled":
            if status == "running":
                return (f"🛑 swarm job {job_id} cancelled — it stops "
                        "at its next segment boundary and its part "
                        "files are discarded.")
            return (f"🛑 swarm job {job_id} cancelled — it was still "
                    "queued, so it will never run.")
        if outcome == "already-cancelled":
            return f"swarm job {job_id} is already cancelled."
        if outcome == "finished":
            return (f"swarm job {job_id} finished before the "
                    "cancellation landed.")
        return (f"no swarm job matching “{arg[:60]}” — it may "
                "already have finished, or the id is wrong.")

    def _resolve_swarm_job(self, queue: Any, arg: str
                           ) -> tuple[Any | None, bool]:
        """Resolve an id, file name, or unique substring to one
        swarm job.  Returns (job, ambiguous) — (None, False) when
        nothing matches, (None, True) when several do.  Never
        raises."""
        from pathlib import Path

        try:
            norm = " ".join(arg.split()).lower()
            jobs = queue.list_jobs()

            def _title(job: Any) -> str:
                return Path(str(job.get("dest") or "")).name.lower()

            for job in jobs:
                if str(job.get("id") or "").lower() == norm:
                    return job, False
            for job in jobs:
                if _title(job) and _title(job) == norm:
                    return job, False
            hits = [job for job in jobs
                    if norm in _title(job)
                    or norm in str(job.get("url") or "").lower()
                    or norm in str(job.get("id") or "").lower()]
            if len(hits) == 1:
                return hits[0], False
            if len(hits) > 1:
                return None, True
            return None, False
        except Exception as exc:  # noqa: BLE001 — resolution never raises
            record_failure(
                "runtime_media.RuntimeMediaMixin._resolve_swarm_job",
                exc)
            return None, False

    def _download_sources_status(self) -> str:
        """/download status — the source registry's growth state
        (pass #86): seed vs grown sources, health, outcome counts,
        candidates pending/shed.  Renders the SAME registry the
        download path credits (the downloader's context-bound
        instance), so what this shows is what downloads learned."""
        from ...media.downloader import MediaDownloader
        from ...media.source_registry import render_status_text

        reg = MediaDownloader(self.context).source_registry()
        return render_status_text(reg.status())

    def _control_explore(self, tail: str, chat_key: str = "") -> Any:
        """/explore — the exploration board (WS4 P3): Devon
        watches the video field she fetches from.

        /explore              → the board: last sweep's truth
                                (failures named as failures), the
                                next sweep due, current claims,
                                candidate standings, and the proxy
                                dependency stated plainly
        /explore run          → one sweep NOW through the same
                                cycle entry point the durable jobs
                                fire (the owner's hand is ungated
                                by the quiet-hours / daily-cap
                                discipline), ending in fresh claims
        /explore trends       → the current claims with their
                                basis and freshness
        /explore candidates   → the candidate pool's standings
        /explore niches       → the shared niche registry: the
                                shipped floor + discovered niche
                                proposals with their evidence
        /explore adopt <ref>  → the owner's adoption act on a
                                proposal (#n from /explore niches,
                                or its slug): flips the overlay
                                record + scaffold-generates a
                                DRAFT plugin — never a production
                                plugin, never auto-registered
        /explore dismiss <ref> → the owner's other act: dismiss
                                a proposal (same evidence won't
                                re-propose it)

        Owner-only (CONTROL_COMMANDS, never PUBLIC_COMMANDS).
        Buttons (button-fix lane 5, reversing the earlier
        text-complete stance as an explicit decision): the board
        carries its four real moves (run / trends / candidates /
        niches) and the niches view carries Adopt / Dismiss per
        proposal — every payload is the typed verb for the same
        act, tap-tested by effect; the text stays complete without
        them.
        """
        from ...media.explore import board as explore_board
        from ...media.explore.jobs import run_sweep_cycle

        words = (tail or "").strip().split()
        verb = words[0].lower() if words else ""
        # Presentation Phase 3 remainder: every explore view is
        # a Card framing the board module's own lines verbatim
        # (the board module is shared with the durable jobs, so
        # the wrap stays at this call site).
        from ...social.chat.typography import card_for_text

        if verb == "run":
            cycle = run_sweep_cycle(self.context, manual=True)
            return card_for_text(
                explore_board.render_sweep_cycle(cycle), "briefing",
                title="Explore sweep")
        if verb == "trends":
            return card_for_text(
                explore_board.render_explore_trends(self.context),
                "briefing", title="Explore trends")
        if verb == "candidates":
            return card_for_text(
                explore_board.render_explore_candidates(
                    self.context),
                "briefing", title="Explore candidates")
        if verb == "niches":
            rendered = explore_board.render_explore_niches(
                self.context)
            rendered = card_for_text(rendered, "briefing",
                                     title="Explore niches")
            from ...media.explore.niches import proposal_board_order

            store = explore_board.resolve_store(self.context)
            proposals = proposal_board_order(store)
            if not proposals:
                return rendered
            from ...social.chat.tgbot_buttons import actionable, grid_layout
            buttons: list[tuple[str, str]] = []
            for i in range(1, len(proposals) + 1):
                buttons.append(
                    (f"✅ Adopt #{i}", f"explore adopt {i}"))
                buttons.append(
                    (f"🗑 Dismiss #{i}", f"explore dismiss {i}"))
            return actionable(rendered, grid_layout([buttons]))
        if verb in ("adopt", "dismiss"):
            from ...media.explore import niches as explore_niches

            ref = words[1] if len(words) > 1 else ""
            if not ref:
                return (f"usage: /explore {verb} <#n or slug> — "
                        "/explore niches lists the proposals "
                        "with their numbers")
            store = explore_board.resolve_store(self.context)
            act = (explore_niches.adopt_niche if verb == "adopt"
                   else explore_niches.dismiss_niche)
            result = act(store, ref)
            rendered = explore_board.render_niche_act(result, verb)
            if not result.get("ok"):
                # A refused act stays a plain line — it is not
                # a board.
                return rendered
            return card_for_text(rendered, "briefing",
                                 title=f"Niche {verb}ed")
        if verb == "":
            rendered = explore_board.render_explore_board(
                self.context,
                scheduler=getattr(self, "_scheduler", None))
            rendered = card_for_text(rendered, "leaderboard",
                                     title="Explore")
            from ...social.chat.tgbot_buttons import actionable, grid_layout
            buttons = [("▶️ Run sweep", "explore run"),
                       ("📈 Trends", "explore trends"),
                       ("🎯 Candidates", "explore candidates"),
                       ("🧩 Niches", "explore niches")]
            return actionable(rendered, grid_layout([buttons]))
        return ("usage: /explore [run | trends | candidates | "
                "niches | adopt <ref> | dismiss <ref>]\n"
                "bare /explore is the board: last sweep, what's "
                "moving with its basis, the candidate pool, and "
                "when the next sweep is due.")

    # ── dedicated platform downloaders (FEI gap wave A) ─────────────
    #
    # /tiktok, /instagram, /spotify and /aio are entry points into
    # the SAME routes /download uses — nomorals.media.platforms does
    # detection + the platform-shaped fetches over MediaDownloader,
    # and /download itself routes Spotify/TikTok/Instagram links
    # through them.  One implementation per platform, several doors.

    @staticmethod
    def _first_url(text: str) -> str:
        """The first http(s) URL inside *text* ("" when none)."""
        m = re.search(r"https?://[^\s<>\"']+", str(text or ""))
        return m.group(0).rstrip(").,") if m else ""

    def _send_platform_file(self, chat: Any, path: str, caption: str,
                            size_bytes: int) -> tuple[str, bool]:
        """Send a downloaded platform file to a chat.  (text, sent).

        Over the platform's inline-media cap the gateway sends as a
        document instead — the reply says so, so a big video never
        looks like it vanished.  Never raises.
        """
        if chat is None:
            return f"downloaded: {path}", False
        note = ""
        try:
            from ...media.platforms import ensure_media_extension
            from ...social.chat.platform_limits import for_platform
            # A file that reaches this seam named `.bin` (or with no
            # extension at all) is repaired from its own bytes
            # before it goes out — sniff-only, no guessing: the
            # downloader already applied the kind fallback upstream,
            # so an unidentifiable file here stays as it is rather
            # than wearing a wrong label. The repair also fixes the
            # delivery kind below: a video named `.bin` would ride
            # as a document instead of a playable video.
            path = ensure_media_extension(path,
                                          allow_kind_fallback=False)
            cap_mb = for_platform(chat.platform).max_file_mb
            if cap_mb and size_bytes > cap_mb * 1024 * 1024:
                note = (f"\n(it's {size_bytes // (1024 * 1024)}MB — over "
                        f"{chat.platform}'s {cap_mb:g}MB inline-media "
                        "cap, so it goes as a document)")
        except Exception as e:  # noqa: BLE001 — the cap note is garnish
            record_failure("runtime_media._send_platform_file.cap", e)
        try:
            self.gateway.send_file(
                chat.platform, f"{chat.platform}:{chat.chat_id}",
                path, caption=caption[:1024], kind=_send_kind_for(path))
            return note.strip(), True
        except Exception as exc:  # noqa: BLE001
            return (f"downloaded to {path} but couldn't send it: {exc}",
                    False)

    def _tiktok_produce(self, url: str, audio_only: bool,
                        chat: Any) -> tuple[str, bool]:
        """The /tiktok download+send body (job worker / sync CLI)."""
        from ...core.jobs import report_progress
        from ...media.downloader import MediaDownloader
        from ...media.platforms import fetch_tiktok

        report_progress(f"fetching TikTok {url[:60]}")
        res = fetch_tiktok(url, downloader=MediaDownloader(self.context),
                           audio_only=audio_only)
        if not res.ok:
            return f"couldn't fetch that TikTok — {res.failure}", False
        title = res.title or "TikTok video"
        if audio_only:
            label = f"🎵 {title} — audio via {res.route}"
        elif res.watermarked is False:
            label = f"🎬 {title} — no watermark (via {res.route})"
        elif res.watermarked is True:
            label = (f"🎬 {title} — this one carries the TikTok "
                     f"watermark (via {res.route})")
        else:
            label = f"🎬 {title} — via {res.route}"
        if chat is None:
            return f"{label}\ndownloaded: {res.path}", False
        note, sent = self._send_platform_file(
            chat, res.path, label, res.size_bytes)
        if not sent:
            return note, False
        return (label + (f"\n{note}" if note else "")), True

    def _control_tiktok(self, tail: str, chat_key: str = "") -> str:
        """/tiktok <url> [audio] — fetch a TikTok video.

        The no-watermark rendition is fetched when a route genuinely
        provides one, and the reply names the route and the watermark
        truth: "no watermark" is only ever said when a clean
        rendition is what came down.
        """
        tail = (tail or "").strip()
        audio_only = bool(re.search(r"(^|\s)(audio|--audio|-a)(\s|$)",
                                    tail, re.I))
        url = self._first_url(tail)
        if not url:
            return ("usage: /tiktok <tiktok url> [audio]\n"
                    "paste the video's link — vm.tiktok.com and "
                    "vt.tiktok.com share links work too.")
        from ...media.platforms import detect_platform
        if detect_platform(url) != "tiktok":
            return (f"that's not a TikTok link: {url[:80]}\n"
                    "/download takes any link; /tiktok is TikTok-only.")
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None:
            return self._play_download_bg(
                chat, chat_key, "⏳ fetching that TikTok…",
                lambda: self._tiktok_produce(url, audio_only, chat),
                title=url[:80])
        text, _sent = self._tiktok_produce(url, audio_only, None)
        return text

    def _instagram_produce(self, url: str,
                           chat: Any) -> tuple[str, bool]:
        """The /instagram download+send body (job worker / sync CLI)."""
        from ...core.jobs import report_progress
        from ...media.downloader import MediaDownloader
        from ...media.platforms import fetch_instagram

        report_progress(f"fetching Instagram {url[:60]}")
        res = fetch_instagram(url,
                              downloader=MediaDownloader(self.context))
        if not res.ok:
            return res.failure, False
        title = res.title or "Instagram post"
        label = f"📸 {title} — via {res.route}"
        if chat is None:
            return f"{label}\ndownloaded: {res.path}", False
        note, sent = self._send_platform_file(
            chat, res.path, label, res.size_bytes)
        if not sent:
            return note, False
        return (label + (f"\n{note}" if note else "")), True

    def _control_instagram(self, tail: str, chat_key: str = "") -> str:
        """/instagram <url> — fetch a public Instagram reel/post.

        Public content comes down through the download chain;
        private, story and age-gated posts hit Instagram's login
        wall, and the reply names the wall instead of failing raw.
        """
        url = self._first_url(tail or "")
        if not url:
            return ("usage: /instagram <instagram url>\n"
                    "paste the reel/post link — public posts come "
                    "down; private ones need a login I don't hold.")
        from ...media.platforms import detect_platform
        if detect_platform(url) != "instagram":
            return (f"that's not an Instagram link: {url[:80]}\n"
                    "/download takes any link; /instagram is "
                    "Instagram-only.")
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None:
            return self._play_download_bg(
                chat, chat_key, "⏳ fetching that Instagram post…",
                lambda: self._instagram_produce(url, chat),
                title=url[:80])
        text, _sent = self._instagram_produce(url, None)
        return text

    #: Spotify collections resolve at most this many tracks per call —
    #: each match is a full ladder run, and the summary says plainly
    #: when a collection was capped.
    _SPOTIFY_COLLECTION_CAP = 10

    def _spotify_match(self, query: str, duration_hint: float = 0.0):
        """One track through the /play full-source ladder.

        Returns ``(item, attempts, source_label)`` — item None when
        no full-track source has the track.  The preview law binds
        here exactly as in /play: a resolution that is preview-only
        or not downloadable is not a match, and the candidate sweep
        only accepts full candidates.  Never raises.
        """
        from ...media.resolver import SourceResolver

        attempts: list[str] = []
        try:
            resolver = SourceResolver(self.context)
            res = resolver.resolve(query)
            attempts.extend(res.attempts or [])
            if res.ok and res.downloadable and not res.preview_only:
                return ({
                    "kind": res.kind, "path": res.path_or_url,
                    "title": res.title or query,
                    "artist": res.artist or "",
                    "duration": res.duration or duration_hint or 0.0,
                    "downloadable": True, "preview_only": False,
                }, attempts, res.source_name or res.kind)
            for c in resolver.search_candidates(query):
                if c.downloadable and not getattr(
                        c, "preview_only", False):
                    return ({
                        "kind": c.kind, "path": c.path_or_url,
                        "title": c.title or query,
                        "artist": c.artist or "",
                        "duration": c.duration or duration_hint or 0.0,
                        "downloadable": True, "preview_only": False,
                    }, attempts, c.source_name or c.kind)
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime_media.RuntimeMediaMixin._spotify_match", e)
        return None, attempts, ""

    def _spotify_match_and_deliver(self, query: str, meta: Any,
                                   chat: Any, chat_key: str,
                                   track: Any = None) -> str:
        """Match one Spotify-identified track and deliver it."""
        from ...media.playback import PlaybackEngine

        hint = float(getattr(track, "duration_s", 0.0) or 0.0)
        item, attempts, label = self._spotify_match(query, hint)
        name = (getattr(track, "title", "") or getattr(meta, "name", "")
                or query)
        artist = (getattr(track, "artist", "")
                  or getattr(meta, "artist", "") or "")
        head = f"🎧 “{name}”" + (f" — {artist}" if artist else "")
        if item is None:
            # The tried trail is journal-only (presentation law).
            self._play_log_tried(attempts)
            return (f"{head}\nfound it on Spotify, but no full-track "
                    "source has it — the only audio out there is "
                    "previews/metadata, and I won't pass a 30s preview "
                    "off as the song.")
        engine = PlaybackEngine(self.context)
        added = engine._enqueue(
            item["path"], item["kind"], item["title"],
            artist=item["artist"], duration=item["duration"])
        if chat is not None:
            return self._play_download_bg(
                chat, chat_key, f"⏳ {name}",
                lambda: self._play_send_in_chat_result(
                    engine, added, chat,
                    note=(f"matched via {label} — Spotify itself "
                          "serves only previews")),
                title=str(added.get("title") or "audio"))
        out = (f"{head}\nSpotify itself serves only previews — the "
               f"full track was matched via {label}.")
        st = engine.play()
        if st.get("status") == "playing":
            out += f"\n▶ playing “{st.get('current', '')}” [{st.get('backend')}]"
        else:
            out += (f"\nqueued — playback status: {st.get('status')} "
                    f"{st.get('error') or st.get('hint', '')}")
        return out

    def _spotify_send_many(self, engine: Any, items: list,
                           summary: str, chat: Any) -> tuple[str, bool]:
        """Chat produce for a Spotify collection: send each matched
        file, then report the count honestly.  Never raises."""
        sent = 0
        failures: list[str] = []
        for item in items:
            text, ok = self._play_send_in_chat_result(
                engine, item, chat)
            if ok:
                sent += 1
            else:
                failures.append(
                    f"  - “{item.get('title') or 'track'}”: "
                    f"{text.strip()}")
        lines = [summary, f"sent {sent} of {len(items)} files"]
        lines.extend(failures)
        return "\n".join(lines), sent > 0

    def _spotify_collection(self, meta: Any, chat: Any,
                            chat_key: str) -> str:
        """Match a Spotify album/playlist track by track (capped)."""
        from ...media.playback import PlaybackEngine

        cap = self._SPOTIFY_COLLECTION_CAP
        tracks = list(meta.tracks or [])
        shown = tracks[:cap]
        engine = PlaybackEngine(self.context)
        winners: list[tuple[Any, dict, str]] = []
        misses: list[Any] = []
        for track in shown:
            item, _attempts, label = self._spotify_match(
                track.query, track.duration_s)
            if item is None:
                misses.append(track)
            else:
                winners.append((track, item, label))
        head = (f"🎧 Spotify {meta.kind} “{meta.name}”"
                + (f" — {meta.artist}" if meta.artist else "")
                + f" · {len(tracks)} tracks")
        if len(tracks) > cap:
            head += f" (fetching the first {cap} in full)"
        lines = [head,
                 f"matched {len(winners)} of {len(shown)} in full:"]
        added_items = []
        for track, item, label in winners:
            added = engine._enqueue(
                item["path"], item["kind"], item["title"],
                artist=item["artist"], duration=item["duration"])
            added_items.append(added)
            lines.append(f"  ✓ “{track.title}”"
                         + (f" — {track.artist}" if track.artist else "")
                         + f" · via {label}")
        if misses:
            lines.append("no full-track match (previews/metadata only):")
            for track in misses:
                lines.append(f"  ✗ “{track.title}”"
                             + (f" — {track.artist}"
                                if track.artist else ""))
        summary = "\n".join(lines)
        if chat is not None:
            if not added_items:
                return summary
            return self._play_download_bg(
                chat, chat_key,
                f"⏳ {meta.name or 'Spotify collection'}",
                lambda: self._spotify_send_many(
                    engine, added_items, summary, chat),
                title=meta.name or "spotify collection")
        if added_items:
            st = engine.play()
            if st.get("status") == "playing":
                summary += (f"\n▶ playing “{st.get('current', '')}” "
                            f"[{st.get('backend')}]")
        return summary

    def _control_spotify(self, tail: str, chat_key: str = "") -> str:
        """/spotify <url> — identify a Spotify link, fetch it in full.

        Spotify's own audio is DRM previews, so the honest shape is:
        identify the track/album/playlist from the link (keyless,
        from Spotify's public pages), then fetch the FULL track
        through the /play full-source ladder and label where it
        actually came from ("matched via YouTube").  A bare title
        instead of a link runs the same match as a text search.
        Playlists/albums resolve track by track, capped, with a
        matched/failed summary.
        """
        from ...media import spotify_meta as sm

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /spotify <spotify url>\n"
                    "paste a track, album or playlist link — I'll "
                    "identify it on Spotify and fetch the full track "
                    "from a downloadable source, labeled by where it "
                    "came from. Spotify's own files are 30s previews; "
                    "those are never sent.")
        chat = self._ref_from_key(chat_key) if chat_key else None
        ref = sm.parse_spotify_ref(tail)
        if ref is None and "spoti.fi" in tail:
            short = self._first_url(tail)
            if short:
                from ...media.downloader import expand_short_url
                ref = sm.parse_spotify_ref(expand_short_url(short))
        if ref is None:
            # Bare text: a title search through the same ladder.
            return self._spotify_match_and_deliver(
                tail, None, chat, chat_key)
        kind, sid = ref
        meta = sm.fetch_spotify_meta(kind, sid)
        if meta is None:
            return ("couldn't identify that Spotify link — it may be "
                    "dead, region-locked, or Spotify changed its "
                    f"pages.\n{tail[:80]}")
        if kind in ("track", "episode"):
            track = meta.tracks[0] if meta.tracks else sm.SpotifyTrack(
                title=meta.name, artist=meta.artist,
                duration_s=meta.duration_s)
            return self._spotify_match_and_deliver(
                track.query, meta, chat, chat_key, track=track)
        if not meta.tracks:
            note = meta.note or "couldn't list its tracks"
            return (f"🎧 Spotify {kind} “{meta.name}” — {note}. Paste a "
                    "single track link and I'll fetch that one in full.")
        return self._spotify_collection(meta, chat, chat_key)

    def _control_aio(self, tail: str, chat_key: str = "") -> str:
        """/aio <url> [audio] — all-in-one downloader.

        Paste any supported link; the platform is detected from the
        URL and routed to the same fetch /download uses (TikTok and
        Instagram get their dedicated routes, Spotify gets
        identify-and-match, everything else the universal chain).
        A link no route reaches gets an honest refusal naming what
        IS supported, up front — not a burned download chain.
        """
        from ...media.platforms import (
            SUPPORTED_PLATFORMS,
            detect_platform,
            is_media_file_url,
            platform_label,
        )

        tail = (tail or "").strip()
        url = self._first_url(tail)
        if not url:
            return ("usage: /aio <url> [audio]\n"
                    "paste a link from: "
                    + ", ".join(platform_label(p)
                                for p in SUPPORTED_PLATFORMS)
                    + " — or any direct media file link.")
        if not detect_platform(url) and not is_media_file_url(url):
            host = re.sub(r"^https?://", "", url).split("/")[0]
            return (f"I can't fetch from {host} — no download route "
                    "reaches that site.\nwhat I can fetch: "
                    + ", ".join(platform_label(p)
                                for p in SUPPORTED_PLATFORMS)
                    + ", or a direct media file link (mp4/mp3/…).")
        return self._control_download(tail, chat_key)

    # ── background downloads (owner decision 2026-10-19; presentation
    # law 2026-11-05) ──────────────────────────────────────────────────
    #
    # Long downloads run as background jobs on the shared JobRegistry:
    # the chat handler never blocks on a multi-minute download.  The
    # job body is the EXISTING synchronous download+send
    # (``_play_send_in_chat_result`` / ``_play_send_with_fallthrough``
    # / the /download body) on the job's worker thread.
    #
    # Presentation law (owner, 2026-11-05, live-bot screenshot): the
    # whole exchange is ONE message edited in place.  The wrapper
    # sends the ack itself, the worker edits it as real milestones
    # land (a candidate being tried), and the finished state REPLACES
    # the ack — the delivered item's line on success, the honest
    # failure on failure.  No queue counts, no "[job …]" ids, no
    # "sources tried" lists, no "sent … to this chat" receipts in
    # chat: the tried trail goes to the journal (_play_log_tried),
    # and the file landing is the delivery.  Where a surface has no
    # edit verb at all, the ack stays a one-line fetch note and a
    # failure still posts exactly once — never a three-message chain.

    #: Label marking a job as a chat download; the _process harvester
    #: filters on it (+ the chat key) so outcomes surface per chat.
    _PLAY_DL_JOB_OWNER = "play"

    # ── the ack handle: one message, edited in place ───────────────

    def _chat_adapter(self, platform: str) -> Any | None:
        """The gateway's adapter for *platform* (or None)."""
        try:
            return self.gateway._adapter_for(platform)
        except Exception as e:  # noqa: BLE001 - edits are best-effort
            record_failure(
                "runtime_media.RuntimeMediaMixin._chat_adapter", e)
            return None

    def _derive_edit_buttons(self, adapter: Any, chat: Any,
                             text: str) -> Any:
        """The keyboard an edited message carries when the caller asked
        for derivation (``buttons=None``): the adapter's own derivation
        first — the exact one ``send()`` uses — else the shared content
        patterns (adapters whose edit verb takes buttons but which
        derive nothing themselves). None when nothing fits. Never
        raises.
        """
        derive = getattr(adapter, "_derive_buttons", None)
        if callable(derive):
            try:
                return derive(chat, text)
            except Exception as e:  # noqa: BLE001 - derivation never breaks an edit
                record_failure(
                    "runtime_media.RuntimeMediaMixin._derive_edit_buttons",
                    e)
                return None
        try:
            from ...social.chat.tgbot_buttons import buttons_for_text

            return buttons_for_text(text)
        except Exception as e:  # noqa: BLE001 - derivation never breaks an edit
            record_failure(
                "runtime_media.RuntimeMediaMixin._derive_edit_buttons", e)
            return None

    def _edit_chat_message(self, chat: Any, message_id: str,
                           text: str, *, progress: bool = False,
                           buttons: Any = _EDIT_BUTTONS_UNSET) -> bool:
        """Edit one of Devon's own messages in *chat* in place.

        Verb ladder: ``edit_message`` (the unthrottled final edit —
        Bot API editMessageText / Telethon edit / the WhatsApp
        bridges' edit command) → ``edit_progress`` (the throttled
        progressive-display seam).  ``progress=True`` goes straight
        to the throttled seam: mid-job milestones must never spam
        edits.  Success shapes differ per adapter (API dict, verdict
        dict with ``ok``, None on refusal/throttle) — normalized
        here to a bool.  Never raises.

        ``buttons`` (pick-list buttons, 2026-11-12): the default
        (``_EDIT_BUTTONS_UNSET``) keeps today's settled-render clear
        (``buttons=[]``); ``None`` derives the keyboard from the
        final text — the same derivation ``send()`` uses — so a live
        choice surface (the /play pick-list offer) rides the edit as
        tappable rows instead of losing its keyboard; a list passes
        through as-is, and an explicit empty list clears the
        keyboard.
        """
        if not message_id:
            return False
        adapter = self._chat_adapter(chat.platform)
        if adapter is None:
            return False
        if buttons is _EDIT_BUTTONS_UNSET:
            buttons = []
        elif buttons is None:
            buttons = self._derive_edit_buttons(adapter, chat, text)
        verbs: list[str] = (["edit_progress"] if progress else
                            ["edit_message", "edit_progress"])
        for name in verbs:
            verb = getattr(adapter, name, None)
            if not callable(verb):
                continue
            try:
                if name == "edit_message":
                    try:
                        out = verb(chat, str(message_id), text,
                                   buttons=buttons)
                    except TypeError:
                        out = verb(chat, str(message_id), text)
                else:
                    out = verb(chat, str(message_id), text)
            except Exception as e:  # noqa: BLE001 - best-effort edit
                record_failure(
                    "runtime_media.RuntimeMediaMixin._edit_chat_message",
                    e)
                continue
            if isinstance(out, dict):
                if out.get("ok") is False:
                    continue  # a named refusal — try the next verb
                return True
            if out is not None:
                return True
        return False

    def _job_ack_progress(self, text: str) -> None:
        """Mid-job milestone edit of the running job's ack message.

        Bodies call this at REAL milestones (a candidate being
        tried); the handle rides a contextvar the job wrapper sets
        for the body's thread, so concurrent jobs never cross-edit.
        Throttled at the adapter seam.  A no-op outside a background
        job or on surfaces without an edit verb.  Never raises.
        """
        try:
            handle = _RUNNING_JOB_ACK.get()
            if not handle:
                return
            chat, message_id = handle
            if chat is not None and message_id:
                self._edit_chat_message(chat, message_id, text,
                                        progress=True)
        except Exception as e:  # noqa: BLE001 - progress never breaks
            record_failure(
                "runtime_media.RuntimeMediaMixin._job_ack_progress", e)

    def _play_download_bg(self, chat: Any, chat_key: str,
                          ack: str, produce: Any, *,
                          on_sent: Any = None, title: str = "",
                          tracker: Any = None,
                          claim_key: str = "") -> str:
        """Start a chat download+send as a background job behind ONE
        ack message, edited in place (presentation law, 2026-11-05).

        This wrapper sends *ack* itself and captures its message id;
        ``produce()`` — the synchronous download+send body, run on
        the job's worker thread — sends the FILE itself and returns
        ``(text, sent)``: the outcome line (final state, no ack
        prefix) plus whether the item went out.  When the body
        finishes, the ack is EDITED to that outcome — the delivered
        item's line on success, the honest failure/offer on failure —
        and a successful delivery fires the expression organ's
        celebration moment.  Only when the edit is impossible (no
        edit verb on the surface, edit refused) does a failure post
        as one fallback message; a success never posts a receipt.
        ``on_sent`` runs on the worker thread only when the file
        actually went out (e.g. consuming a pick-list).  The result
        dict records ``delivered`` + ``ack_id`` so
        :meth:`_harvest_play_downloads` never double-reports an
        outcome and can still edit the ack after a restart.

        If the registry itself refuses the job, fall back to running
        the body synchronously rather than dropping the download.
        Returns "" — the ack already posted (this kind family is
        MEDIA_DELIVERED).  Never raises.
        """
        from ...core.jobs import current_job, report_progress, shared_registry

        # Measured progress (Play & Downloads P1): ONE tracker per
        # download operation.  Its renders ride the EXISTING ack
        # progress seam (adapter-throttled) and mirror into the job's
        # progress line — no second progress surface.  The tracker's
        # terminal state is recorded for other consumers but NOT
        # rendered onto the ack: the outcome edit at the end of the
        # body owns the ack's final state, promptly and exactly once.
        if tracker is None:
            from ...core.ids import new_short_id
            from ...core.progress_events import ProgressTracker

            tracker = ProgressTracker(
                op_id=new_short_id("dl"), title=title or "")

        # Living card (Living Chat P2): the download's durable
        # card — family "progress", identity = the tracker's op
        # id.  The tracker stays the measurement source; the card
        # is the record that survives the process (its snapshots
        # are folded in below, terminal included, even though the
        # ack's own final state is owned by the outcome edit).
        _dl_card_holder: dict[str, str] = {}

        def _card_sync(event: Any, _tracker: Any = tracker) -> None:
            card_id = _dl_card_holder.get("card_id") or ""
            if not card_id:
                return
            try:
                from ...social.chat.living_cards import apply_progress_snapshot
                store = self._living_card_store()
                card = store.get(card_id)
                if card is None:
                    return
                apply_progress_snapshot(card, _tracker.snapshot())
                old = card.revision
                card.revision = old + 1
                store.save(card, expect_revision=old)
            except Exception as e:  # noqa: BLE001 - a record, not a gate
                record_failure(
                    "runtime_media.RuntimeMediaMixin._play_download_bg"
                    ".card_sync", e)

        def _render_progress(event: Any, _tracker: Any = tracker) -> None:
            _card_sync(event, _tracker)
            text = _tracker.render_text()
            if event.terminal is None:
                # The ack's own text already says "working": an edit
                # carrying no stage and no measurement yet would only
                # flash noise — and a job that fails before any stage
                # must land as exactly ONE edit, the outcome.  A new
                # source taking over IS news and renders at once.
                snap = _tracker.snapshot()
                if snap["stage"] or snap["bytes_received"] \
                        or snap["source_index"] > 1:
                    self._job_ack_progress(text)
            try:
                report_progress(text)
            except Exception as e:  # noqa: BLE001 - a mirror, not a gate
                record_failure(
                    "runtime_media.RuntimeMediaMixin._play_download_bg"
                    ".progress_mirror", e)

        tracker.on_event = _render_progress

        # Duplicate guard: the same chat asking for the same title
        # while it is already downloading is told so in plain words —
        # never a second download of the same file (and never the
        # file sent twice).  Internal ids never print (owner law), so
        # /download cancel resolves the job by title.
        try:
            for snap in shared_registry().list():
                labels = snap.get("labels") or {}
                if (snap.get("status") in ("pending", "running")
                        and labels.get("owner") == self._PLAY_DL_JOB_OWNER
                        and labels.get("chat") == (chat_key or "")
                        and labels.get("title") == (title or "")[:80]):
                    return (
                        f"⏳ “{(title or 'that download')[:60]}” is "
                        "already on its way — it lands here when it's "
                        "ready. "
                        f"/download cancel {(title or '')[:60]} "
                        "stops it.")
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._play_download_bg", e)

        # Send the ack NOW — its message id is the handle every later
        # state of this exchange edits.  When even the ack can't post
        # (surface down), the job still runs: completion then falls
        # back to posting the outcome as one message.
        ack_id = ""
        try:
            sent_ack = self.gateway.send(chat.platform, chat, ack)
            if sent_ack is not None and getattr(sent_ack, "ok", False):
                ack_id = str(getattr(sent_ack, "message_id", "") or "")
        except Exception as e:  # noqa: BLE001 - the ack is best-effort
            record_failure(
                "runtime_media.RuntimeMediaMixin._play_download_bg.ack", e)

        # Open the download's living card on the ack: the ack
        # message IS the card's live render (same message id), and
        # the durable card record is what a restart rebinds to.
        try:
            from ...social.chat.living_cards import progress_card
            _dl_card = progress_card(
                tracker, chat_key=chat_key or "",
                platform=str(getattr(chat, "platform", "") or ""),
                chat_id=str(getattr(chat, "chat_id", "") or ""))
            _dl_card.message_id = ack_id
            _dl_card.delivered = bool(ack_id)
            self._living_card_store().save(_dl_card)
            _dl_card_holder["card_id"] = _dl_card.card_id
        except Exception as e:  # noqa: BLE001 - a record, not a gate
            record_failure(
                "runtime_media.RuntimeMediaMixin._play_download_bg"
                ".card_open", e)

        # The worker body needs its own job id (for the durable
        # record) before submit() returns it — a one-cell holder the
        # submitting thread fills in immediately after submit.
        holder: dict[str, str] = {}

        def _body() -> dict[str, Any]:
            from ...core.progress_events import (
                reset_current_tracker,
                set_current_tracker,
            )

            token = _RUNNING_JOB_ACK.set(
                (chat, ack_id) if ack_id else None)
            ttoken = set_current_tracker(tracker)
            try:
                report_progress("download started")
                # Send-time exactly-once (dupfix4 D4): the intake
                # claim was taken minutes ago; re-check who holds
                # it before the file/pick list goes out. A sibling
                # that failed over and claimed meanwhile owns the
                # command — this job stands down without sending,
                # instead of landing the second copy the owner
                # screenshotted. No ledger, no key, or this surface
                # holding the claim: proceed (fail open). The ack
                # is left as the searching card — the sibling's
                # answer is the visible outcome — and the harvester
                # skips a "skipped" result, so no failure is
                # invented for a deliberate stand-down.
                if claim_key and self._command_claim_lost(
                        claim_key, chat):
                    tracker.finish(
                        "skipped",
                        note="stood down: sibling surface holds the "
                        "command claim")
                    self._play_dl_mark_finished(holder)
                    return {"ok": False, "delivered": False,
                            "sent": False, "title": title, "text": "",
                            "ack_id": ack_id, "skipped": True,
                            "error": "stood-down: sibling surface "
                            "holds the command claim"}
                text, sent = produce()
            except Exception as exc:  # noqa: BLE001 - failure is data
                tracker.finish(
                    "failed",
                    error=f"{type(exc).__name__}: {exc}"[:300])
                self._play_dl_mark_finished(holder)
                return {"ok": False, "delivered": False, "sent": False,
                        "title": title, "text": "", "ack_id": ack_id,
                        "error": f"{type(exc).__name__}: {exc}"[:300]}
            finally:
                _RUNNING_JOB_ACK.reset(token)
                reset_current_tracker(ttoken)
            if sent and on_sent is not None:
                try:
                    on_sent()
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._body", e)
            final = str(text or "").strip()
            late = False
            job_now = current_job()
            if job_now is not None:
                with job_now.lock:
                    late = job_now.status == "presumed-dead"
            if late and final:
                final = ("⏳ that download finished late — the quiet "
                         "stretch was it still working, not a drop:\n"
                         + final)
            # The finished state REPLACES the ack.  A success with no
            # outcome line keeps the ack's own line as the final
            # state; a failure always lands — edited in when
            # possible, posted once when not.
            edited = False
            if ack_id and final:
                # A pick-list offer is a LIVE choice surface, not a
                # settled render: the ack-edit must carry the derived
                # keyboard (the ListMessage rows on WhatsApp) instead
                # of the clearing edit — the buttons were never sent
                # on this path, so the owner saw plain text
                # (2026-11-12 /play pick-list). Every other final
                # keeps the settled-render clear.
                from ...media.picklist import PICK_MARKER_RE

                if PICK_MARKER_RE.search(final):
                    edited = self._edit_chat_message(
                        chat, ack_id, final, buttons=None)
                else:
                    edited = self._edit_chat_message(chat, ack_id, final)
            delivered = False
            if edited:
                delivered = True
            elif final and not sent:
                try:
                    self.gateway.send(chat.platform, chat, final)
                    delivered = True
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._body:1105", e)
                    delivered = False
            elif sent:
                # The file landing in the chat IS the report — no
                # receipt posts.  Mark delivered so the harvester
                # does not invent a "its report never reached this
                # chat" failure for a delivery that landed on purpose.
                delivered = True
            if sent:
                self._celebrate_delivery(chat, chat_key, title)
            # The operation's terminal truth for the progress stream:
            # delivered only when the file actually went out.
            tracker.finish("delivered" if sent else "failed")
            self._play_dl_mark_finished(holder)
            return {"ok": True, "delivered": delivered,
                    "sent": bool(sent), "title": title, "text": final,
                    "ack_id": ack_id, "late": late}

        try:
            job = shared_registry().submit(
                _body, tool="play-download",
                labels={"owner": self._PLAY_DL_JOB_OWNER,
                        "chat": chat_key or "",
                        "title": (title or "")[:80]})
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._play_download_bg:1118", e)
            from ...core.progress_events import (
                reset_current_tracker,
                set_current_tracker,
            )

            ttoken = set_current_tracker(tracker)
            try:
                text, _sent = produce()
            finally:
                reset_current_tracker(ttoken)
            tracker.finish("delivered" if _sent else "failed")
            return str(text or "")
        holder["job_id"] = job.id
        self._play_dl_record_start(job.id, chat_key, title)
        # Link the card to its job so the restart harvester can
        # close the exact card a lost download belonged to.
        try:
            _card_id = _dl_card_holder.get("card_id") or ""
            if _card_id:
                _store = self._living_card_store()
                _card = _store.get(_card_id)
                if _card is not None:
                    _card.state = {**_card.state, "job_id": job.id}
                    _old = _card.revision
                    _card.revision = _old + 1
                    _store.save(_card, expect_revision=_old)
        except Exception as e:  # noqa: BLE001 - a record, not a gate
            record_failure(
                "runtime_media.RuntimeMediaMixin._play_download_bg"
                ".card_link", e)
        return ""

    def _celebrate_delivery(self, chat: Any, chat_key: str,
                            title: str) -> None:
        """Expression law (owner, 2026-11-05): a delivery is a moment
        that earns one — fire the expression organ's celebration for
        the room when its own settings allow.  DM rooms carry no
        group settings, so the organ's defaults apply; failures never
        reach here.  Best effort: a celebration must never break
        a delivery."""
        try:
            service = self._expression_service()
            if service is None:
                return
            from types import SimpleNamespace

            shim = SimpleNamespace(
                chat=chat, platform=str(getattr(chat, "platform", "")),
                chat_id=str(getattr(chat, "chat_id", "")),
                text="", sender_id="", kind=None)
            service.fire(
                "celebration", room=chat_key,
                platform=str(getattr(chat, "platform", "")), settings={},
                send=lambda _moment_chat, media_ref:
                    self._expression_send(shim, media_ref))
        except Exception as e:  # noqa: BLE001 - decoration, never fatal
            record_failure(
                "runtime_media.RuntimeMediaMixin._celebrate_delivery", e)

    # ── durable record of chat downloads (restart honesty) ─────────
    #
    # The job registry is in-memory: a process restart mid-download
    # used to lose the job silently — the chat was never told.  Every
    # chat download now also writes a row (job id, chat, title, boot
    # id) to the DB at submit; the body flips it to "done" when it
    # finishes.  A row still "running" under a DIFFERENT boot id can
    # only mean the process that owned it died — the harvester says
    # so, once, instead of the download vanishing.

    def _maybe_cover_video(self, chat: Any, path: str, title: str,
                           artist: str) -> None:
        """Procedural animated cover for a delivered song (doc 8,
        demonstrator 1 second half): rendered from the track's real
        audio envelope + artwork by :mod:`nomorals.media.cover_video`
        — no generation model — and sent as a native video AFTER the
        track itself.

        Gated by the ``media_cover_video`` setting (default OFF —
        doc 7 §5: no decorative messages by default for every song).
        strictly best-effort: a render or send failure is recorded
        and changes nothing about the delivery that already landed.
        Never raises."""
        try:
            settings = getattr(self.context, "settings", None)
            enabled = False
            if isinstance(settings, dict):
                enabled = bool(settings.get("media_cover_video"))
            elif settings is not None:
                enabled = bool(getattr(settings, "media_cover_video",
                                       False))
            if not enabled or not path or not os.path.isfile(path):
                return
            import time

            from ...media.cover_video import render_cover_video

            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(
                os.path.dirname(path),
                f"{stem}.cover-{time.time_ns() % 10**8}.mp4")
            report = render_cover_video(path, out, title=title,
                                        artist=artist)
            if not report.ok:
                record_failure(
                    "runtime_media.RuntimeMediaMixin._maybe_cover_video",
                    RuntimeError(report.error or "cover render failed"))
                return
            from ...core.progress_events import UPLOADING, emit_stage
            emit_stage(UPLOADING, note="sending the animated cover")
            self.gateway.send_file(
                chat.platform, f"{chat.platform}:{chat.chat_id}",
                out, caption=f"🎬 {title} — animated cover"[:1024],
                kind="video")
        except Exception as e:  # noqa: BLE001 - decoration, never fatal
            record_failure(
                "runtime_media.RuntimeMediaMixin._maybe_cover_video", e)

    def _play_dl_boot_id(self) -> str:
        """This process's boot id (one per runtime instance)."""
        bid = getattr(self, "_play_dl_boot", "")
        if not bid:
            import uuid

            bid = uuid.uuid4().hex[:12]
            self._play_dl_boot = bid
        return bid

    def _play_dl_db(self) -> Any | None:
        """The context DB with the downloads table ensured, or None."""
        ctx = getattr(self, "context", None)
        db = getattr(ctx, "db", None) if ctx is not None else None
        if db is None:
            return None
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS play_download_jobs ("
                "job_id TEXT PRIMARY KEY, chat_key TEXT NOT NULL "
                "DEFAULT '', title TEXT NOT NULL DEFAULT '', "
                "boot_id TEXT NOT NULL DEFAULT '', status TEXT NOT "
                "NULL DEFAULT 'running', created_at REAL NOT NULL "
                "DEFAULT 0)")
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._play_dl_db", e)
            return None
        return db

    def _play_dl_record_start(self, job_id: str, chat_key: str,
                              title: str) -> None:
        import time

        db = self._play_dl_db()
        if db is None or not job_id:
            return
        try:
            db.execute(
                "INSERT OR REPLACE INTO play_download_jobs "
                "(job_id, chat_key, title, boot_id, status, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (job_id, chat_key or "", (title or "")[:120],
                 self._play_dl_boot_id(), "running", time.time()))
            # Prune finished rows older than a week — the table is a
            # restart ledger, not an archive.
            db.execute(
                "DELETE FROM play_download_jobs WHERE status != "
                "'running' AND created_at < ?",
                (time.time() - 7 * 24 * 3600,))
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._play_dl_record_start", e)

    def _play_dl_record_finish(self, job_id: str) -> None:
        db = self._play_dl_db()
        if db is None or not job_id:
            return
        try:
            db.execute(
                "UPDATE play_download_jobs SET status = 'done' "
                "WHERE job_id = ? AND status = 'running'", (job_id,))
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._play_dl_record_finish", e)

    def _play_dl_mark_finished(self, holder: dict[str, str]) -> None:
        """Record the body's job as finished in the durable ledger.

        The body starts before submit() returns its job id; the
        submitting thread fills *holder* immediately after, so a
        fast body waits briefly for the id rather than skipping the
        record (which would later read as a lost download).
        """
        import time

        job_id = holder.get("job_id") or ""
        for _ in range(40):
            if job_id:
                break
            time.sleep(0.05)
            job_id = holder.get("job_id") or ""
        if job_id:
            self._play_dl_record_finish(job_id)

    def _cancel_download_job(self, arg: str, chat_key: str) -> str:
        """/download cancel <title|job-id> — cooperative cancellation.

        Only the chat that owns the job may cancel it.  Cancellation
        is cooperative: the flag is set on the job and the download
        backends stop at their next progress checkpoint (yt-dlp hook,
        direct-HTTP chunk hook, CLI subprocess poll) — bytes actually
        stop; a job that already finished is reported as finished.

        The argument is the download's TITLE as the owner knows it
        (internal ids never print, presentation law 2026-11-05); an
        exact job id still resolves for surfaces that carry one.
        Title resolution looks at THIS chat's live download jobs:
        exact title first, then a unique substring match.
        """
        from ...core.jobs import shared_registry

        arg = (arg or "").strip()
        if not arg:
            return ("usage: /download cancel <title> — name the "
                    "download the way the ack did (its title or "
                    "link), and I'll stop it.")
        registry = shared_registry()
        job = registry.poll(arg)
        if job is None:
            job = self._find_chat_download_job(arg, chat_key)
            if job is None:
                return (f"no live download matching “{arg[:60]}” in "
                        "this chat — it may already have finished.")
        labels = job.labels or {}
        if labels.get("owner") != self._PLAY_DL_JOB_OWNER \
                or labels.get("chat") != (chat_key or ""):
            return "that download doesn't belong to this chat."
        title = str(labels.get("title") or "download")
        if job.done():
            return (f"“{title[:60]}” already finished ({job.status}) "
                    "— nothing left to cancel.")
        if registry.cancel(job.id):
            return (f"🛑 cancellation requested for “{title[:60]}” — "
                    "the download stops at its next progress "
                    "checkpoint. If it had already finished, its "
                    "result still arrives.")
        return (f"“{title[:60]}” finished before the cancellation "
                "landed.")

    def _find_chat_download_job(self, arg: str, chat_key: str) -> Any:
        """Resolve a title (exact, then unique substring) to one of
        this chat's live download jobs, or None.  Never raises."""
        try:
            from ...core.jobs import shared_registry

            norm = " ".join(arg.split()).lower()
            live = []
            for snap in shared_registry().list():
                labels = snap.get("labels") or {}
                if snap.get("status") not in ("pending", "running"):
                    continue
                if labels.get("owner") != self._PLAY_DL_JOB_OWNER:
                    continue
                if labels.get("chat") != (chat_key or ""):
                    continue
                live.append(snap)
            for snap in live:
                if " ".join(str(snap["labels"].get("title") or "")
                            .split()).lower() == norm:
                    return shared_registry().poll(str(snap.get("id")))
            hits = [snap for snap in live
                    if norm in " ".join(
                        str(snap["labels"].get("title") or "")
                        .split()).lower()]
            if len(hits) == 1:
                return shared_registry().poll(str(hits[0].get("id")))
            return None
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime_media.RuntimeMediaMixin._find_chat_download_job",
                e)
            return None

    def _harvest_play_downloads(self, chat_key: str, chat: Any) -> None:
        """Surface this chat's download jobs that never got to report.

        The job body delivers its own outcome (file + follow-up text)
        on success AND on honest download failure — those need no
        harvest.  What remains:

        - ``failed``: the worker crashed before it could report;
        - ``presumed-dead``: heartbeat silence — honest UNKNOWN (the
          download may still be running; never a fake failure, never
          a blind retry);
        - ``done``/late results whose own follow-up send failed:
          re-send the body's stored text.

        Collected exactly once (``collect_ready`` marks collected).
        Runs on every inbound message for the chat.  Never raises.
        """
        from ...core.jobs import shared_registry

        try:
            jobs = shared_registry().collect_ready(
                labels={"owner": self._PLAY_DL_JOB_OWNER,
                        "chat": chat_key or ""})
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._harvest_play_downloads", e)
            return
        target = f"{chat.platform}:{chat.chat_id}"
        for job in jobs:
            try:
                self._play_dl_record_finish(job.id)
                res = job.result if isinstance(job.result, dict) else {}
                if res.get("delivered"):
                    continue  # body already reported — never twice
                if res.get("skipped"):
                    continue  # stood down at send time (dupfix4 D4):
                    # the sibling surface holds the command claim
                    # and sent the answer — the outcome landed,
                    # there is nothing to harvest and no failure
                    # to invent for a deliberate stand-down.
                title = str(job.labels.get("title")
                            or res.get("title") or "download")
                if job.status == "failed":
                    self.gateway.send(
                        chat.platform, target,
                        f"⚠️ the background download “{title}” broke "
                        f"before it could report: "
                        f"{job.error or 'unknown error'}")
                elif job.status == "presumed-dead" \
                        and not job.late_result:
                    self.gateway.send(
                        chat.platform, target,
                        f"⏳ “{title}” went quiet while downloading "
                        "(no heartbeat) — outcome unknown: it may "
                        "still be running, so I won't retry it "
                        "blindly. Run /play again if you want it "
                        "re-queued.")
                else:
                    # done-but-undelivered, or a late finish whose own
                    # outcome never landed: the outcome REPLACES the
                    # ack when the ack's id survived (edit-in-place
                    # law); only when there is no ack to edit does the
                    # outcome post as a message.
                    text = str(res.get("text") or "")
                    if not text:
                        text = (f"⚠️ the background download of "
                                f"“{title}” finished, but its report "
                                "never reached this chat.")
                    ack_id = str(res.get("ack_id") or "")
                    if not (ack_id and self._edit_chat_message(
                            chat, ack_id, text)):
                        self.gateway.send(chat.platform, target, text)
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._harvest_play_downloads:1311", e)
                continue
        # Restart losses: rows still "running" under a previous boot
        # id belong to a process that died mid-download.  Say so —
        # once — instead of the download vanishing silently.
        db = self._play_dl_db()
        if db is not None:
            try:
                rows = db.query(
                    "SELECT job_id, title FROM play_download_jobs "
                    "WHERE chat_key = ? AND status = 'running' "
                    "AND boot_id != ?",
                    (chat_key or "", self._play_dl_boot_id()))
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._harvest_play_downloads:1324", e)
                rows = []
            for row in rows or []:
                lost_title = str(row.get("title") or "download")
                try:
                    self.gateway.send(
                        chat.platform, target,
                        f"⚠️ the download “{lost_title}” was lost when "
                        "I restarted — it never finished. Ask me again "
                        "and I'll redo it. (If the file did reach you "
                        "before the restart, ignore this.)")
                    db.execute(
                        "UPDATE play_download_jobs SET status = "
                        "'lost-reported' WHERE job_id = ?",
                        (row.get("job_id"),))
                    # Close the lost download's living card with
                    # the same honesty (Living Chat P2): the card
                    # rebinds after a restart, and its terminal
                    # state is this loss — never a phantom "still
                    # downloading" left on the record.
                    try:
                        _store = self._living_card_store()
                        for _card in _store.live_for_chat(
                                chat_key or ""):
                            if _card.family != "progress":
                                continue
                            if _card.state.get("job_id") != \
                                    row.get("job_id"):
                                continue
                            _card.status = "lost"
                            _card.terminal = True
                            _card.state = {
                                **_card.state,
                                "restart_loss": True}
                            _old = _card.revision
                            _card.revision = _old + 1
                            _store.save(
                                _card, expect_revision=_old)
                    except Exception as e:  # noqa: BLE001
                        record_failure(
                            "runtime_media.RuntimeMediaMixin"
                            "._harvest_play_downloads.card_close", e)
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._harvest_play_downloads:1339", e)
                    continue

    def _play_send_in_chat(self, engine: Any, item: dict[str, Any],
                           chat: Any) -> str:
        """Chat path for /play: download the track, send it as a file.

        Never plays locally — the user receives the audio in the
        conversation.  Honest on every failure path.  Never raises.

        Wording matters: when the download fails there was nothing to
        send, so the failure is "couldn't download", never
        "couldn't send".  "Couldn't send" is reserved for the case
        where the file WAS downloaded but the send itself failed.
        The success text is the delivered item's own line — the file
        landing is the delivery, never a "sent … to this chat"
        receipt (presentation law, 2026-11-05).
        """
        text, _sent = self._play_send_in_chat_result(engine, item, chat)
        return text

    def _play_deliver_fetched(self, chat: Any, result: Any,
                              payload: dict[str, Any]) -> str:
        """Deliver a file a wired journey already fetched (the
        registry play path, doc 7): tag a delivery copy and send it
        natively, exactly like the ordinary /play send paths.  The
        full-track verdict rides the payload from fetch time; an
        UNVERIFIED verdict is labelled on the delivered line, never
        passed off as proof.  Never raises.
        """
        path = str(payload.get("path") or "")
        title = str(payload.get("title") or result.query or "audio")
        artist = str(payload.get("artist") or "")
        line = f"🎵 {title}"
        if artist and artist.lower() not in title.lower():
            line += f" — {artist}"
        line += (f"\nvia {result.source_name} · licence: {result.licence}"
                 f" · fetched via "
                 f"{payload.get('served_via') or 'a wired source'}")
        if payload.get("verify_state") == "unverified":
            line += (f"\n(full-track length unverified — "
                     f"{payload.get('verify_why') or 'not probed'})")
        send_path = self._delivery_path(path, title, artist)
        send_path = self._tag_delivery_copy(send_path, title, artist)
        try:
            from ...core.progress_events import UPLOADING, emit_stage
            emit_stage(UPLOADING, note="sending the file")
            res = self.gateway.send_file(
                chat.platform, f"{chat.platform}:{chat.chat_id}",
                send_path, caption=line[:1024],
                kind=_send_kind_for(send_path))
            if res is not None and getattr(res, "ok", True) is False:
                err = str(getattr(res, "error", "") or "send failed")
                return (f"had the file but couldn't send it: {err} "
                        f"(it's saved on the server at {path})")
            return line
        except Exception as exc:  # noqa: BLE001
            return (f"had the file but couldn't send it: {exc} "
                    f"(it's saved on the server at {path})")

    def _play_dl_relabel_current_job(self, title: str) -> None:
        """Relabel the running play-download job with the resolved title.

        The ack-first branch (2026-11-21 wedge) registers the job
        under the raw query — the resolved title is unknown until the
        worker resolves.  The duplicate guard, ``/download cancel``,
        the harvester and job lookups all key on the title label, so
        the worker relabels its own job once the resolve lands.
        Best-effort: never raises, no-op outside a job.
        """
        try:
            from ...core.jobs import current_job

            job = current_job()
            if job is None:
                return
            with job.lock:
                job.labels["title"] = (title or "")[:80]
        except Exception:
            pass

    def _play_resolve_and_send(self, engine: Any, search_q: str,
                               rest: str, title_q: str, artist_q: str,
                               chat: Any, *, sweep: bool = True,
                               version: str | None = None
                               ) -> tuple[str, bool]:
        """Background body for a chat /play — resolve + sweep + send.

        Runs on the download job's worker thread, behind the ack the
        chat thread already sent (:meth:`_control_play`'s ack-first
        branch).  The resolve + candidate sweep used to run
        synchronously on the chat thread before the first ack — two
        sequential full-chain sweeps — so /play took minutes to even
        get processed (2026-11-21).  Returns ``(text, sent)`` for the
        ack edit: the delivered item's line, or the honest failure.
        Never raises.
        """
        from ...media.resolver import SourceResolver

        resolution = SourceResolver(self.context).resolve(search_q)
        if not resolution.ok:
            # Never a dead end — tried + why + hint via PlayService
            # (pure formatting, no extra network).
            try:
                from ...media.play_service import PlayService
                detail = PlayService(
                    self.context).explain_failure(resolution)
                # explain_failure already includes "hint: ..." —
                # appending resolution.hint again would print the
                # hint body twice.
                hint = ""
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._play_resolve_and_send", e)
                detail = ("tried: "
                          + ("; ".join(resolution.attempts)
                             or "no strategies"))
                hint = (f"\n{resolution.hint}"
                        if resolution.hint else "")
            msg = f"couldn't find {rest[:80]!r} —\n{detail}{hint}"
            if chat is not None:
                # Chat users are on phones: the resolver's full
                # yt-dlp command dump is not actionable there —
                # shorten to the /cookies pointer.
                from ...media.cookies import BOT_COOKIE_HELP as _FULL_HELP
                from ...media.cookies import BOT_COOKIE_SHORT_HELP as _SHORT_HELP
                msg = msg.replace(_FULL_HELP, _SHORT_HELP)
            return msg, False
        # The job was registered under the raw query; the lookups key
        # on the resolved title — relabel now that it is known.
        self._play_dl_relabel_current_job(resolution.title or rest)
        # A Spotify link resolves ok but not downloadable — route it
        # through the identify-and-match ladder instead of the
        # download that hard-refuses it (DRM).
        if (resolution.ok and str(resolution.kind) == "spotify"
                and not resolution.downloadable):
            return self._play_spotify_link_in_chat(
                rest, chat, version=version)
        # A requested version is the answer, never a fallback.
        if version:
            return self._play_version_send(
                engine, version, search_q, rest, title_q, artist_q,
                chat, resolution)
        added = []
        for r in (resolution, *resolution.extra_tracks):
            added.append(engine._enqueue(
                r.path_or_url, r.kind, r.title or r.path_or_url,
                artist=r.artist, duration=r.duration))
        if not sweep or resolution.extra_tracks:
            # URL/path, or an album/playlist hit: the first track
            # goes out the same way the URL/path path sends it — no
            # candidate sweep (a URL is already the answer).
            self._play_log_tried(resolution.attempts)
            return self._play_send_in_chat_result(engine, added[0], chat)
        # Text query, single track: download fall-through across ALL
        # candidates.  A search-ok source whose download fails is a
        # download failure, not success — keep trying until one both
        # finds AND downloads.  The helper renders the full
        # tried-list, so don't append the resolver's partial list here.
        try:
            cands, detailed = SourceResolver(
                self.context).search_candidates_detailed(search_q)
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._play_resolve_and_send.sweep", e)
            cands, detailed = [], []
        fallbacks = [
            {"kind": c.kind, "path": c.path_or_url,
             "title": c.title or c.path_or_url,
             "artist": c.artist or "",
             "duration": c.duration or 0.0,
             "downloadable": c.downloadable,
             "block_reason": str(
                 getattr(c, "block_reason", "") or ""),
             "preview_only": bool(
                 getattr(c, "preview_only", False))}
            for c in (cands or [])]
        # The chain winner can carry the raw search query as its
        # "title" (the YouTube search strategies do this) — the
        # candidate sweep has the SAME track with its real metadata;
        # prefer it, so nothing downstream ever displays or delivers
        # the query text as if it were the track's name.
        first_item = dict(added[0])
        for c in (cands or []):
            if c.kind == first_item.get("kind") \
                    and c.path_or_url == first_item.get("path"):
                cur = str(first_item.get("title") or "")
                if c.title and (not cur
                                or cur.strip().lower() in (
                                    search_q.strip().lower(),
                                    rest.strip().lower())):
                    first_item["title"] = c.title
                if c.artist and not first_item.get("artist"):
                    first_item["artist"] = c.artist
                break
        # The chain verdict rides onto the first item: a
        # found-but-blocked resolution (protected streams,
        # metadata — or a PREVIEW) must not be attempted
        # as if it were downloadable.
        first_item["downloadable"] = resolution.downloadable
        first_item["block_reason"] = str(
            getattr(resolution, "block_reason", "") or "")
        first_item["preview_only"] = bool(
            getattr(resolution, "preview_only", False))
        merged = self._merge_search_attempts(
            resolution.attempts, detailed)
        # The sweep may have refined the title (the chain winner can
        # carry the raw query as its "title") — relabel again so the
        # job, the cancel verb and the harvester all name the track.
        self._play_dl_relabel_current_job(
            str(first_item.get("title") or rest))
        # ── preview gate ────────────────────────────────────────
        # The best every source has is a 30s preview and no
        # full-track candidate exists: do NOT queue a download job
        # for a preview-grade candidate.  Drop the queue row, log the
        # trail, and say the honest thing in plain words.
        if self._play_is_preview_item(first_item) \
                and not any(
                    fb.get("downloadable") is not False
                    and not self._play_is_preview_item(fb)
                    for fb in fallbacks):
            self._play_log_tried(merged)
            self._drop_matching_queued(engine, first_item)
            return self._play_preview_line(first_item), False
        return self._play_send_with_fallthrough(
            engine, first_item, fallbacks, merged,
            chat, match=(title_q, artist_q), query=rest)

    def _play_send_in_chat_result(self, engine: Any, item: dict[str, Any],
                                  chat: Any, *,
                                  note: str = "") -> tuple[str, bool]:
        """Same as :meth:`_play_send_in_chat`, plus whether the file
        was actually sent — callers that consume state on success
        (e.g. the pick-list) need the signal, not a string search.
        ``sent`` is True only when send_file succeeded.  ``note`` is
        an optional provenance line appended under the delivered
        item's line (e.g. where a Spotify match actually came from).
        Never raises.
        """
        from ...core.jobs import report_progress
        from ...media.cookies import BOT_COOKIE_SHORT_HELP, is_bot_detection_error
        # Progress note: counts as a heartbeat when this runs as a
        # background job; a no-op on the synchronous path.
        report_progress(
            f"downloading {item.get('title') or 'audio'}")
        from ...core.progress_events import emit_source
        emit_source(str(item.get("title") or "audio"))
        try:
            dl = engine.download(item)
        except Exception as exc:  # noqa: BLE001 - download() shouldn't raise, belt and braces
            # Undeliverable: drop the just-enqueued queue row so the
            # durable queue doesn't keep a broken item.
            self._drop_matching_queued(engine, item)
            return f"couldn't download the audio file: {exc}", False
        if not dl.get("ok"):
            reason = str(dl.get("reason", "unknown error"))
            self._drop_matching_queued(engine, item)
            msg = f"couldn't download the audio: {reason}"
            if is_bot_detection_error(reason):
                msg += f"\n{BOT_COOKIE_SHORT_HELP}"
            return msg, False
        if dl.get("substituted_from"):
            # The engine could not have THIS track — its source
            # blocked it — and fetched a different recording (a
            # YouTube copy) instead.  This path exists to deliver
            # one SPECIFIC item (a pasted URL, a pick-list choice):
            # passing the substitute off as that item would be a
            # silent swap, so it is refused here, by name.  The
            # by-name fall-through path is where substitutes are
            # allowed — and there they are declared.
            self._drop_matching_queued(engine, item)
            label = str(dl.get("substituted_from") or "the source")
            why = self._play_clean_reason(
                dl.get("substitute_reason") or dl.get("note") or "")
            why_part = f" ({why})" if why else ""
            title = str(item.get("title") or "that track")
            return (
                f"{label} has that exact track blocked{why_part} — "
                f"and I won't swap in a different recording for it: "
                f"you asked for this track. Ask for “{title}” by "
                "name and I'll fetch the best copy from any source, "
                "declared as the substitute it is."), False
        path = dl["path"]
        # Verify what arrived, via the shared acceptance contract
        # (Phase 2): a preview-length file for a full-length track —
        # or a positively wrong file, like an HTML error page — is a
        # failed download, never the song; don't send it.  Only a
        # REJECTION refuses delivery; an UNVERIFIED file is delivered
        # (absence of evidence is not a failure) under its plain
        # identity line — the line claims the track's identity, never
        # a verification the state doesn't support.  Surfaces that
        # REPORT the journey (the registry path) label the state
        # explicitly.
        from ...media.download_orchestrator import DownloadOrchestrator
        acc = DownloadOrchestrator().verify(
            path, "song",
            expected_duration=float(item.get("duration") or 0))
        if not acc.ok:
            self._drop_matching_queued(engine, item)
            return (f"that download wasn't the song — "
                    f"{acc.reason}. I won't pass it off as the "
                    "track."), False
        title = str(dl.get("title") or item.get("title", "audio"))
        artist = str(dl.get("artist") or item.get("artist", "") or "")
        send_path = self._delivery_path(path, title, artist)
        send_path = self._tag_delivery_copy(send_path, title, artist)
        line = f"🎵 {title}"
        if artist and artist.lower() not in title.lower():
            line += f" — {artist}"
        if note:
            line += f"\n{note}"
        try:
            from ...core.progress_events import UPLOADING, emit_stage
            emit_stage(UPLOADING, note="sending the file")
            res = self.gateway.send_file(
                chat.platform, f"{chat.platform}:{chat.chat_id}",
                send_path, caption=line[:1024], kind=_send_kind_for(send_path))
            # The gateway reports a failed media send as a RESULT,
            # not a raise — an unchecked ok=False here would edit
            # the ack to the delivery line for a file that never
            # arrived (a false success on the owner's own surface).
            if res is not None and getattr(res, "ok", True) is False:
                err = str(getattr(res, "error", "") or "send failed")
                return (f"had the file but couldn't send it: {err} "
                        f"(it's saved on the server at {path})"), False
            return line, True
        except Exception as exc:  # noqa: BLE001
            return (f"had the file but couldn't send it: {exc} "
                    f"(it's saved on the server at {path})"), False

    def _play_download_only(self, engine: Any,
                            item: dict[str, Any]) -> tuple[str, str]:
        """Download one item WITHOUT sending it: ``(path, "")`` on a
        landed, verified file, ``("", reason)`` otherwise. The
        version render and the Spotify ladder both need the file
        before deciding what goes out. Never raises."""
        try:
            dl = engine.download(item)
        except Exception as exc:  # noqa: BLE001 - download() shouldn't raise
            return "", f"couldn't download the audio: {exc}"
        if not dl.get("ok"):
            return "", (f"couldn't download the audio: "
                       f"{dl.get('reason', 'unknown error')}")
        path = str(dl.get("path") or "")
        from ...media.download_orchestrator import DownloadOrchestrator
        acc = DownloadOrchestrator().verify(
            path, "song",
            expected_duration=float(item.get("duration") or 0))
        if not acc.ok:
            return "", (f"that download wasn't the song — "
                       f"{acc.reason}. I won't pass it off as the "
                       "track.")
        return path, ""

    def _play_send_rendered(self, engine: Any, item: dict[str, Any],
                            path: str, version: str, chat: Any,
                            title: str, artist: str) -> tuple[str, bool]:
        """Render *version* onto the landed file at *path* with the
        DSP chain and send the render as the version — tagged and
        captioned as the version, honestly. Returns (text, sent).
        Never raises."""
        spec = self._PLAY_VERSIONS[version]
        from ...audio.edit import apply_fx
        try:
            rendered = apply_fx(path, spec["chain"],
                                suffix=spec["slug"])
        except Exception as exc:  # noqa: BLE001 - render never dies
            rendered = {"ok": False, "reason": str(exc)}
        if not rendered.get("ok"):
            return (f"got the track but couldn't render the "
                    f"{spec['label']} version: "
                    f"{rendered.get('reason', 'render failed')}"), False
        version_title = f"{title} ({spec['label']} version)"
        rendered_item = engine._enqueue(
            str(rendered["output"]), "file", version_title,
            artist=artist, duration=0.0)
        return self._play_send_in_chat_result(
            engine, rendered_item, chat,
            note=(f"{spec['label']} version — rendered with Devon's "
                  f"{spec['label']} chain"))

    def _play_cli_version(self, engine: Any, version: str,
                          resolution: Any, rest: str,
                          title_q: str, artist_q: str) -> str:
        """CLI /play <song> <version>: download the winner, render
        the version, queue the render for local playback. Never
        silently plays the original. Returns the queue receipt."""
        spec = self._PLAY_VERSIONS[version]
        item = {"kind": str(resolution.kind),
                "path": str(resolution.path_or_url),
                "title": str(resolution.title or rest),
                "artist": str(resolution.artist or ""),
                "duration": float(resolution.duration or 0.0)}
        path, why = self._play_download_only(engine, item)
        if not path:
            return (f"couldn't fetch {rest[:80]!r} to render the "
                    f"{spec['label']} version from —\n{why}")
        from ...audio.edit import apply_fx
        try:
            rendered = apply_fx(path, spec["chain"],
                                suffix=spec["slug"])
        except Exception as exc:  # noqa: BLE001 - render never dies
            rendered = {"ok": False, "reason": str(exc)}
        if not rendered.get("ok"):
            return (f"got the track but couldn't render the "
                    f"{spec['label']} version: "
                    f"{rendered.get('reason', 'render failed')}")
        title = title_q or item["title"]
        artist = artist_q or item["artist"]
        version_title = f"{title} ({spec['label']} version)"
        engine._enqueue(
            str(rendered["output"]), "file", version_title,
            artist=artist, duration=0.0)
        queue_len = len(engine.queue())
        out = (f"queued 1 (queue {queue_len}):\n"
               f"  - {version_title}"
               + (f" — {artist}" if artist else "")
               + f"\nrendered with Devon's {spec['label']} chain")
        self._play_log_tried(resolution.attempts)
        return out

    def _play_version_send(self, engine: Any, version: str,
                           search_q: str, rest: str, title_q: str,
                           artist_q: str, chat: Any,
                           resolution: Any) -> tuple[str, bool]:
        """A requested /play version, in order: FIRST pull the
        version from the source (a version upload), ELSE render the
        effect onto the fetched full track with the DSP chain. The
        version is the answer, never a demoted fallback — the
        edit-demotion sort below never sees this path. Returns
        (text, sent). Never raises."""
        from ...media.picklist import is_confident_match
        from ...media.resolver import SourceResolver

        spec = self._PLAY_VERSIONS[version]
        # ── 1. source-first: the version as uploaded ─────────────
        try:
            cands, _detailed = SourceResolver(
                self.context).search_candidates_detailed(
                    f"{search_q} {spec['search']}")
        except Exception as e:  # noqa: BLE001 - a dead sweep is no version
            record_failure(
                "runtime_media.RuntimeMediaMixin._play_version_send", e)
            cands = []
        for cand in (cands or []):
            if not cand.downloadable or bool(
                    getattr(cand, "preview_only", False)):
                continue
            if not self._play_version_marker_match(
                    version, str(cand.title or "")):
                continue
            if not is_confident_match(
                    str(cand.title or ""), str(cand.artist or ""),
                    title_q, artist_q):
                continue
            added = engine._enqueue(
                cand.path_or_url, cand.kind,
                str(cand.title or cand.path_or_url),
                artist=str(cand.artist or artist_q or ""),
                duration=float(cand.duration or 0.0))
            source = str(cand.source_name or cand.kind or "the source")
            return self._play_send_in_chat_result(
                engine, added, chat,
                note=(f"{spec['label']} version — pulled from "
                      f"{source}"))
        # ── 2. DSP fallback: render onto the fetched original ────
        originals: list[dict[str, Any]] = [{
            "kind": str(resolution.kind),
            "path": str(resolution.path_or_url),
            "title": str(resolution.title or rest),
            "artist": str(resolution.artist or ""),
            "duration": float(resolution.duration or 0.0)}]
        try:
            sweep, _det = SourceResolver(
                self.context).search_candidates_detailed(search_q)
        except Exception as e:  # noqa: BLE001 - the winner alone then
            record_failure(
                "runtime_media.RuntimeMediaMixin._play_version_send.sweep",
                e)
            sweep = []
        seen_paths = {originals[0]["path"]}
        for cand in (sweep or []):
            if not cand.downloadable or bool(
                    getattr(cand, "preview_only", False)):
                continue
            if str(cand.path_or_url) in seen_paths:
                continue
            seen_paths.add(str(cand.path_or_url))
            originals.append({
                "kind": str(cand.kind), "path": str(cand.path_or_url),
                "title": str(cand.title or cand.path_or_url),
                "artist": str(cand.artist or ""),
                "duration": float(cand.duration or 0.0)})
        req_title = title_q or originals[0]["title"]
        req_artist = artist_q or originals[0]["artist"]
        last_why = ""
        for orig in originals:
            path, why = self._play_download_only(engine, orig)
            if not path:
                last_why = why
                continue
            return self._play_send_rendered(
                engine, orig, path, version, chat,
                req_title, req_artist)
        return (f"couldn't fetch the original to render the "
                f"{spec['label']} version from"
                + (f" — {last_why}" if last_why else "")), False

    def _play_spotify_track_in_chat(self, track: Any, meta: Any,
                                    chat: Any, *,
                                    version: str | None = None
                                    ) -> tuple[str, bool]:
        """One Spotify-identified track, matched and delivered in
        chat — the background-safe half of
        :meth:`_spotify_match_and_deliver` (no nested background
        job: the caller already runs inside one). With a version
        request the matched full track is rendered, not sent raw.
        Returns (text, sent). Never raises."""
        from ...media.playback import PlaybackEngine

        hint = float(getattr(track, "duration_s", 0.0) or 0.0)
        item, attempts, label = self._spotify_match(
            track.query, hint)
        name = (getattr(track, "title", "")
                or getattr(meta, "name", "") or track.query)
        artist = (getattr(track, "artist", "")
                  or getattr(meta, "artist", "") or "")
        head = f"🎧 “{name}”" + (f" — {artist}" if artist else "")
        if item is None:
            self._play_log_tried(attempts)
            return (f"{head}\nfound it on Spotify, but no full-track "
                    "source has it — the only audio out there is "
                    "previews/metadata, and I won't pass a 30s preview "
                    "off as the song."), False
        engine = PlaybackEngine(self.context)
        if version:
            return self._play_render_spotify(
                engine, item, version, chat, name, artist)
        added = engine._enqueue(
            item["path"], item["kind"], item["title"],
            artist=item["artist"], duration=item["duration"])
        return self._play_send_in_chat_result(
            engine, added, chat,
            note=(f"matched via {label} — Spotify itself "
                  "serves only previews"))

    def _play_render_spotify(self, engine: Any, item: dict,
                             version: str, chat: Any,
                             name: str, artist: str) -> tuple[str, bool]:
        """Render a version onto a Spotify-matched full track and
        send the render. Returns (text, sent). Never raises."""
        path, why = self._play_download_only(engine, item)
        if not path:
            return why, False
        return self._play_send_rendered(
            engine, item, path, version, chat, name, artist)

    def _play_spotify_link_in_chat(self, url: str, chat: Any, *,
                                   version: str | None = None
                                   ) -> tuple[str, bool]:
        """A Spotify link in chat: identify it on Spotify (keyless,
        from Spotify's public pages), then fetch the FULL track
        through the /play ladder and label where it actually came
        from — the same ladder /spotify uses, never the DRM refusal.
        Returns (text, sent). Never raises."""
        from ...media import spotify_meta as sm

        ref = sm.parse_spotify_ref(url)
        if ref is None and "spoti.fi" in url:
            short = self._first_url(url)
            if short:
                from ...media.downloader import expand_short_url
                ref = sm.parse_spotify_ref(expand_short_url(short))
        if ref is None:
            return (f"couldn't identify that Spotify link — it may be "
                    "dead, region-locked, or Spotify changed its "
                    f"pages.\n{url[:80]}"), False
        kind, sid = ref
        meta = sm.fetch_spotify_meta(kind, sid)
        if meta is None:
            return (f"couldn't identify that Spotify link — it may be "
                    "dead, region-locked, or Spotify changed its "
                    f"pages.\n{url[:80]}"), False
        if kind in ("track", "episode"):
            track = meta.tracks[0] if meta.tracks else sm.SpotifyTrack(
                title=meta.name, artist=meta.artist,
                duration_s=meta.duration_s)
            return self._play_spotify_track_in_chat(
                track, meta, chat, version=version)
        if not meta.tracks:
            note = meta.note or "couldn't list its tracks"
            return (f"🎧 Spotify {kind} “{meta.name}” — {note}. Paste a "
                    "single track link and I'll fetch that one in "
                    "full."), False
        # Collection: match track by track (capped), send each.
        from ...media.playback import PlaybackEngine

        cap = self._SPOTIFY_COLLECTION_CAP
        tracks = list(meta.tracks or [])
        shown = tracks[:cap]
        engine = PlaybackEngine(self.context)
        winners: list[tuple[Any, dict, str]] = []
        misses: list[Any] = []
        for track in shown:
            item, _attempts, label = self._spotify_match(
                track.query, track.duration_s)
            if item is None:
                misses.append(track)
            else:
                winners.append((track, item, label))
        head = (f"🎧 Spotify {meta.kind} “{meta.name}”"
                + (f" — {meta.artist}" if meta.artist else "")
                + f" · {len(tracks)} tracks")
        if len(tracks) > cap:
            head += f" (fetching the first {cap} in full)"
        lines = [head,
                 f"matched {len(winners)} of {len(shown)} in full:"]
        sent = 0
        for track, item, label in winners:
            name = track.title or meta.name
            artist = track.artist or meta.artist or ""
            if version:
                text, ok = self._play_render_spotify(
                    engine, item, version, chat, name, artist)
            else:
                added = engine._enqueue(
                    item["path"], item["kind"], item["title"],
                    artist=item["artist"], duration=item["duration"])
                text, ok = self._play_send_in_chat_result(
                    engine, added, chat,
                    note=(f"matched via {label} — Spotify itself "
                          "serves only previews"))
            if ok:
                sent += 1
                lines.append(f"  ✓ “{name}”"
                             + (f" — {artist}" if artist else "")
                             + f" · via {label}"
                             + (f" · {version} version, rendered"
                                if version else ""))
            else:
                lines.append(f"  ✗ “{name}”"
                             + (f" — {artist}" if artist else "")
                             + f" — {text.strip()}")
        if misses:
            lines.append("no full-track match (previews/metadata only):")
            for track in misses:
                lines.append(f"  ✗ “{track.title}”"
                             + (f" — {track.artist}"
                                if track.artist else ""))
        return "\n".join(lines), sent > 0

    # ── honest artifact identity (Law 6) ────────────────────────────
    #
    # A delivered file must carry the track's real identity in its
    # own metadata: chat clients render title/artist from the file's
    # tags, and a source that shipped the track under a fan-edit name
    # with no artist tag shows the owner "Unknown artist" for a song
    # Devon knew perfectly well.  Tagging writes to a staged copy —
    # the cache original is never mutated.

    #: Title words marking a candidate as an EDIT of the recording
    #: (fan edit, remix, cover…) rather than the canonical recording
    #: itself.  Whole-token matches only — "credited" is not "edit".
    _EDIT_MARKER_TOKENS = frozenset({
        "nightcore", "remix", "remixed", "cover", "bootleg",
        "mashup", "reverb", "slowed", "spedup", "edit", "flip",
        "remake", "parody", "sped", "daycore",
    })

    #: /play version requests — a version is a real variant of the
    #: recording, never a fallback. FIRST pull the version from the
    #: source (a "song reverb version" upload), ELSE render it onto
    #: the fetched full track with the DSP chain. The definitions
    #: are honest, in code: nightcore is tempo-up AND pitch-up
    #: (tape varispeed ×1.25); slowed is tempo-down (varispeed ×0.8,
    #: pitch falls with it); sped up is the same varispeed family
    #: at ×1.25 — nightcore and sped up differ only in which
    #: uploads the source search looks for. ``chain`` is an
    #: :func:`nomorals.audio.edit.apply_fx` spec; ``search`` is
    #: what the source-first search appends to the song query.
    _PLAY_VERSIONS = {
        "reverb": {"label": "reverb", "slug": "reverb",
                   "chain": "reverb wet=0.35, normalize",
                   "search": "reverb version"},
        "nightcore": {"label": "nightcore", "slug": "nightcore",
                      "chain": "varispeed factor=1.25, normalize",
                      "search": "nightcore"},
        "slowed": {"label": "slowed", "slug": "slowed",
                   "chain": "varispeed factor=0.8, normalize",
                   "search": "slowed"},
        "sped up": {"label": "sped up", "slug": "spedup",
                    "chain": "varispeed factor=1.25, normalize",
                    "search": "sped up"},
        "slowed + reverb": {"label": "slowed + reverb",
                            "slug": "slowedreverb",
                            "chain": ("varispeed factor=0.8, "
                                      "reverb wet=0.35, normalize"),
                            "search": "slowed reverb version"},
    }

    #: Version phrases as token tuples — longest first, so a combined
    #: "slowed reverb" is never read as plain "slowed".
    _PLAY_VERSION_PHRASES = (
        (("slowed", "reverb"), "slowed + reverb"),
        (("reverb", "slowed"), "slowed + reverb"),
        (("reverb",), "reverb"),
        (("nightcore",), "nightcore"),
        (("slowed",), "slowed"),
        (("sped", "up"), "sped up"),
    )

    @classmethod
    def _play_parse_version(cls, rest: str):
        """Split a /play version request off the query.

        ``/play <song> reverb`` and ``/play reverb <song>`` both
        parse; a trailing "version" is filler ("/play <song> reverb
        version"). Returns ``(version_key | None, clean_query)`` —
        the clean query keeps the user's own casing for display.
        A bare version word with no song left is not a request.
        Never raises.
        """
        try:
            text = (rest or "").strip()
            if not text:
                return None, text
            words = text.split()
            # Tokenise for matching, tracking which word each token
            # came from: "+" is a separator, "spedup" / "speed up"
            # are the "sped up" phrase.
            toks: list[str] = []
            owners: list[int] = []
            for wi, word in enumerate(words):
                low = word.lower().strip("+")
                if not low:
                    continue
                if low == "spedup":
                    toks.extend(["sped", "up"])
                    owners.extend([wi, wi])
                elif low == "speed":
                    toks.append("sped")
                    owners.append(wi)
                else:
                    toks.append(low)
                    owners.append(wi)
            # A trailing "version" is filler, not the request.
            if toks and toks[-1] == "version":
                toks = toks[:-1]
                owners = owners[:-1]
            if not toks:
                return None, text

            def _clean(lo: int, hi: int) -> str:
                """The words OUTSIDE toks[lo:hi], in order — a word
                that contributed no token (a "+" separator) never
                survives into the query either."""
                drop = set(owners[lo:hi])
                keep = [w for wi, w in enumerate(words)
                        if wi in set(owners) and wi not in drop]
                return " ".join(keep).strip()

            for phrase, key in cls._PLAY_VERSION_PHRASES:
                n = len(phrase)
                if len(toks) > n and tuple(toks[:n]) == phrase:
                    clean = _clean(0, n)
                    if clean:
                        return key, clean
                if len(toks) > n and tuple(toks[-n:]) == phrase:
                    clean = _clean(len(toks) - n, len(toks))
                    if clean:
                        return key, clean
            return None, text
        except Exception:  # noqa: BLE001 - unparsable input is plain /play
            return None, (rest or "")

    @staticmethod
    def _play_version_marker_match(version: str, title: str) -> bool:
        """Does *title* carry the requested version's marker? The
        combined version needs BOTH markers; "sped up" is a
        two-token marker. Never raises."""
        try:
            tokens = re.findall(r"[a-z0-9]+", str(title or "").lower())
            if version == "slowed + reverb":
                return "slowed" in tokens and "reverb" in tokens
            if version == "sped up":
                return "sped" in tokens and "up" in tokens
            return version in tokens
        except Exception:  # noqa: BLE001 - no match on garbage
            return False

    @classmethod
    def _edit_marker_of(cls, title: str) -> str:
        """The edit marker in a candidate's title ("" when the title
        reads as the plain recording).  "sped up" is a two-token
        marker; the rest are single tokens."""
        try:
            tokens = re.findall(r"[a-z0-9]+", str(title or "").lower())
            if "sped" in tokens and "up" in tokens:
                return "sped up"
            for tok in tokens:
                if tok in cls._EDIT_MARKER_TOKENS:
                    return "nightcore" if tok == "daycore" else tok
            return ""
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _tag_delivery_copy(path: str, title: str, artist: str, *,
                           comment: str = "") -> str:
        """Stage a tag-corrected copy of *path* when its own tags
        don't already carry *title*/*artist*.

        Ladder: mutagen when importable → ffmpeg stream-copy with
        -metadata → the path unchanged.  Best-effort throughout: a
        tagging failure never costs the delivery itself.  Returns
        the path to send.
        """
        try:
            if not path or not os.path.isfile(path) or not title:
                return path
            from ...media.library import read_metadata

            meta = read_metadata(path)
            have_title = str(meta.get("title") or "").strip()
            have_artist = str(meta.get("artist") or "").strip()

            def _norm(s: str) -> str:
                return " ".join(re.findall(r"[a-z0-9]+", s.lower()))

            if _norm(have_title) == _norm(title) \
                    and (_norm(have_artist) == _norm(artist)
                         or not artist) \
                    and not comment:
                return path
            # No tagger on either rung → no staged copy at all
            # (a copy that can't be tagged is litter, not identity).
            try:
                import mutagen  # noqa: F401
                tagger = True
            except ImportError:
                import shutil as _sh
                tagger = bool(_sh.which("ffmpeg"))
            if not tagger:
                return path
            import shutil

            # Real post-processing is happening (a staged copy is
            # being tagged): the progress stream hears about it as
            # PROCESSING — stage time measured from here, not from
            # the download (Phase 2 processing emitters).
            from ...core.progress_events import PROCESSING, emit_stage
            emit_stage(PROCESSING, note="tagging the delivery copy")
            stem, ext = os.path.splitext(path)
            target = f"{stem}.del{ext or '.mp3'}"
            if os.path.abspath(target) != os.path.abspath(path):
                shutil.copyfile(path, target)
            else:
                target = path
            if RuntimeMediaMixin._write_tags(
                    target, title, artist, comment):
                return target
            return path
        except Exception as e:  # noqa: BLE001 - tags are garnish
            record_failure(
                "runtime_media.RuntimeMediaMixin._tag_delivery_copy", e)
            return path

    @staticmethod
    def _write_tags(path: str, title: str, artist: str,
                    comment: str) -> bool:
        """Write title/artist/comment tags onto *path* in place.

        mutagen first (pure-Python, per-format tag classes); ffmpeg
        stream-copy second (rewrites the container to a temp file,
        then swaps it over *path*).  Returns True when a rung
        landed.  Never raises.
        """
        try:
            import mutagen

            audio = mutagen.File(path, easy=True)
            if audio is not None:
                if audio.tags is None:
                    audio.add_tags()
                audio["title"] = [title]
                if artist:
                    audio["artist"] = [artist]
                if comment:
                    # Not all formats take a comment frame —
                    # best-effort, never a failed delivery.
                    with contextlib.suppress(Exception):
                        audio["comment"] = [comment]
                audio.save()
                return True
        except ImportError as exc:
            # mutagen absent on this host: tags via mutagen are
            # unavailable (the ffmpeg rung below still runs) — record
            # the degradation instead of letting it look like success.
            record_failure(
                "runtime_media.RuntimeMediaMixin._write_tags.mutagen_import",
                exc)
        except Exception as e:  # noqa: BLE001 - fall to the ffmpeg rung
            record_failure(
                "runtime_media.RuntimeMediaMixin._write_tags.mutagen", e)
        try:
            import shutil
            import subprocess

            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                return False
            tmp = f"{path}.tagtmp{os.path.splitext(path)[1] or '.mp3'}"
            cmd = [ffmpeg, "-y", "-v", "error", "-i", path,
                   "-codec", "copy",
                   "-metadata", f"title={title}"]
            if artist:
                cmd += ["-metadata", f"artist={artist}"]
            if comment:
                cmd += ["-metadata", f"comment={comment}"]
            cmd.append(tmp)
            proc = subprocess.run(  # noqa: S603 - fixed argv, own file
                cmd, capture_output=True, timeout=60)
            if proc.returncode == 0 and os.path.isfile(tmp) \
                    and os.path.getsize(tmp) > 0:
                os.replace(tmp, path)
                return True
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            return False
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime_media.RuntimeMediaMixin._write_tags.ffmpeg", e)
            return False

    @staticmethod
    def _queued_index_of(engine: Any, item: dict[str, Any]) -> int | None:
        """Index of the LAST queue row matching (kind, path), or None.

        The just-enqueued row is the last match — removing exactly
        that one never touches an older duplicate.  Best-effort:
        engines without a list-shaped queue() return None.
        """
        try:
            rows = engine.queue()
            if not isinstance(rows, list):
                return None
            kind = str(item.get("kind", ""))
            path = str(item.get("path", ""))
            found: int | None = None
            for r in rows:
                if not isinstance(r, dict):
                    continue
                if str(r.get("kind", "")) == kind \
                        and str(r.get("path", "")) == path:
                    idx = r.get("index")
                    if isinstance(idx, int):
                        found = idx
            return found
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._queued_index_of", e)
            return None

    @classmethod
    def _drop_matching_queued(cls, engine: Any, item: dict[str, Any]) -> None:
        """Remove the just-enqueued queue row for *item*.  Best-effort."""
        try:
            idx = cls._queued_index_of(engine, item)
            if idx is not None:
                engine.remove(idx)
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._drop_matching_queued", e)

    @staticmethod
    def _delivery_path(path: str, title: str, artist: str) -> str:
        """The path to actually SEND for a downloaded track.

        The delivered file must carry the track's real identity
        (owner delivery contract, 2026-10-20): a download whose name
        already carries the title — yt-dlp's normal
        ``"<title> [<id>].<ext>"`` — goes out as-is.  A downloader
        template artifact (``playlist [playlist].mp3``) or any other
        title-free name would show the user garbage and "Unknown
        artist", so a copy named ``"<Artist> - <Title>.<ext>"`` is
        staged NEXT TO the original and sent instead.  The original
        is a cache file (e.g. YouTube's id-keyed cache) — copied,
        never renamed.  Never raises: any failure returns *path*.
        """
        try:
            if not path or not os.path.isfile(path):
                return path
            title_tokens = {w for w in re.findall(
                r"[a-z0-9]+", (title or "").lower()) if len(w) > 1}
            if not title_tokens:
                return path
            stem = os.path.splitext(os.path.basename(path))[0]
            stem_tokens = {w for w in re.findall(
                r"[a-z0-9]+", stem.lower()) if len(w) > 1}
            if title_tokens <= stem_tokens:
                return path
            from ...core.text import safe_filename

            label = (f"{artist} - {title}"
                     if (artist or "").strip() else (title or ""))
            ext = os.path.splitext(path)[1] or ".mp3"
            name = safe_filename(f"{label}{ext}", fallback="")
            if not name:
                return path
            target = os.path.join(os.path.dirname(path), name)
            if os.path.abspath(target) == os.path.abspath(path):
                return path
            if not (os.path.isfile(target)
                    and os.path.getsize(target) == os.path.getsize(path)):
                import shutil

                shutil.copyfile(path, target)
            return target
        except Exception as e:
            # breaks a delivery that would otherwise succeed
            record_failure("runtime_media.RuntimeMediaMixin._delivery_path", e)
            return path

    @classmethod
    def _replace_queued_with(cls, engine: Any, old_item: dict[str, Any],
                             new_item: dict[str, Any]) -> None:
        """The queue must hold what was actually SENT, not the failed
        first pick: swap the old row for the delivered candidate.
        Best-effort — never raises, never changes the reply text."""
        try:
            cls._drop_matching_queued(engine, old_item)
            engine._enqueue(
                str(new_item.get("path", "")),
                str(new_item.get("kind", "")),
                str(new_item.get("title", "") or new_item.get("path", "")),
                artist=str(new_item.get("artist", "") or ""),
                duration=float(new_item.get("duration", 0.0) or 0.0))
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._replace_queued_with", e)

    # Candidate kind -> search-attempt name, for merging download
    # outcomes into the tried-list.
    _KIND_TO_ATTEMPT_NAME = {
        "file": "workspace-scan",
        "netnaija": "netnaija-search",
        "notjustok": "notjustok-search",
        "tooxclusive": "tooxclusive-search",
        "soundcloud": "soundcloud-search",
        "bandcamp": "bandcamp-search",
        "youtube": "youtube-search",
        "youtube-music": "youtube-music-search",
        "archiveorg": "archiveorg-search",
        "boomplay": "boomplay-search",
        "musicbrainz": "musicbrainz-search",
        "openverse": "openverse-search",
        "wikimedia": "wikimedia-search",
        "itunes": "itunes-search",
        "deezer": "deezer-search",
        "spotify": "spotify-search",
        "audiomack": "audiomack",
        "url": "direct-url",
    }

    #: Kinds whose only audio is a ~30s preview (sources.py's
    #: PREVIEW_ONLY_SOURCES — proven there from the APIs' response
    #: shapes).  A preview-class item is never downloaded and never
    #: offered as a pick: it only identifies a track.
    _PREVIEW_KINDS = frozenset({"itunes", "deezer"})

    @classmethod
    def _music_chain_index(cls, kind: str) -> int:
        """Position of a source kind in the owner-ordered music chain.

        The pick / attempt order is the match score first and the
        chain order as the tiebreak: when a real source and a demoted
        one (mdundo, owner order 2026-11-18) match equally well, the
        real source is attempted first.  Unknown kinds sort last —
        never ahead of a known source.  The index is built once from
        ``MUSIC_SEARCH_SOURCES`` (the one canonical order) and cached
        on the class; a hiccup returns "last" rather than raising.
        """
        try:
            idx = cls.__dict__.get("_CHAIN_INDEX")
            if idx is None:
                from ...media.sources import MUSIC_SEARCH_SOURCES

                idx = {
                    name.removesuffix("-search"): pos
                    for pos, name in enumerate(MUSIC_SEARCH_SOURCES)
                }
                cls._CHAIN_INDEX = idx
            return idx.get(str(kind or ""), len(idx))
        except Exception:  # noqa: BLE001
            return 1 << 30

    @classmethod
    def _play_is_preview_item(cls, item: Any) -> bool:
        """Is this candidate preview-class (flag or preview kind)?"""
        try:
            if not isinstance(item, dict):
                return False
            if item.get("preview_only"):
                return True
            return str(item.get("kind", "") or "").lower() \
                in cls._PREVIEW_KINDS
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _play_clean_reason(reason: Any) -> str:
        """A download-block reason safe for a user reply: no raw URLs
        (asset links are internal diagnostics), no doubled failure
        wording ("download failed: download failed:"), whitespace
        collapsed, length-capped.  Never raises."""
        try:
            text = re.sub(r"https?://\S+", "", str(reason or ""))
            text = re.sub(
                r"(?i)\b(download (?:failed|blocked):)"
                r"(?:\s*download (?:failed|blocked):)+",
                r"\1", text)
            text = re.sub(r"\s+", " ", text).strip(" -—")
            return text[:120]
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _play_preview_line(item: Any) -> str:
        """The honest-absence sentence for a preview-only find."""
        title = str((item or {}).get("title") or "that track")
        artist = str((item or {}).get("artist") or "").strip()
        who = f" by {artist}" if artist else ""
        return (f"I found “{title}”{who} — but only as a 30-second "
                "preview. No source has the full track, so there's "
                "nothing to send.")

    @staticmethod
    def _play_preview_note(item: Any) -> str:
        """One-line preview fact for a mixed report (other facts
        follow, so this doesn't claim nothing exists)."""
        title = str((item or {}).get("title") or "that track")
        artist = str((item or {}).get("artist") or "").strip()
        who = f" by {artist}" if artist else ""
        return (f"⚠ found “{title}”{who} — but only as a 30-second "
                "preview; a preview is not the song, so I didn't "
                "send it.")

    @staticmethod
    def _play_substitution_note(
            winner: Any,
            prejudged_blocks: list[tuple[dict[str, Any], str]]) -> str:
        """The substitution declaration for a delivered copy: the
        requested match was pre-judged blocked on its own source and
        a DIFFERENT source's recording is what landed.  "" when the
        winner is from the same source as every blocked match (that
        source served the song after all — no substitution).  Never
        raises."""
        try:
            from ...media.picklist import _KIND_LABELS

            winner_kind = str((winner or {}).get("kind", "") or "")
            for blocked_item, blocked_reason in prejudged_blocks:
                blocked_kind = str(
                    (blocked_item or {}).get("kind", "") or "")
                if blocked_kind == winner_kind:
                    continue
                blocked_label = _KIND_LABELS.get(
                    blocked_kind, "") or blocked_kind or "the source"
                winner_label = _KIND_LABELS.get(
                    winner_kind, "") or winner_kind or "another source"
                detail = str(blocked_reason or "").removeprefix(
                    "not downloadable — ")
                if detail.startswith("not downloadable"):
                    detail = "that copy is not downloadable"
                return (f"substitute copy — the {blocked_label} "
                        f"original wasn't obtainable ({detail}); "
                        f"this is {winner_label}'s copy of the song")
            return ""
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _play_block_sentence(block_reason: Any) -> str:
        """The human sentence for a structured block class
        (preview / drm / region / login / private) — the connector
        vocabulary's own text, so every surface names a block the
        same way.  "" when the block is unclassified: an unclassified
        refusal is reported as-is, never dressed in a class it
        didn't earn.  Never raises."""
        try:
            reason = str(block_reason or "").strip()
            if not reason:
                return ""
            from ...connectors.soundcloud import block_reason_text
            return block_reason_text(reason)
        except Exception:  # noqa: BLE001
            return ""

    @classmethod
    def _play_blocked_line(cls, item: Any, reason: Any) -> str:
        """One found-but-blocked fact for a user reply: the track,
        the source's human label, and the block reason cleaned (no
        raw URLs, no doubled wording)."""
        try:
            from ...media.picklist import _KIND_LABELS

            kind = str((item or {}).get("kind", "") or "")
            label = _KIND_LABELS.get(kind, "") or kind
            who = (f" — {(item or {}).get('artist')}"
                   if (item or {}).get("artist") else "")
            clean = cls._play_clean_reason(reason)
            suffix = f" ({clean})" if clean else ""
            return (f"⚠ found “{(item or {}).get('title', '')}”{who} "
                    f"on {label} — but couldn't download it{suffix}.")
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _play_log_tried(search_attempts: Any,
                        outcomes: Any = None) -> None:
        """Record the per-source tried/outcome trail in the journal
        (log).  This is the ONLY home of the tried-list now:
        presentation law (owner, 2026-11-05) — tried-lists and
        receipts never post in chat; the trail lives here and in the
        job record for on-demand reads.  Never raises."""
        try:
            from ...core.logging_setup import get_logger

            lines = [str(a) for a in (search_attempts or [])
                     if isinstance(a, str)]
            for name, outcome in (outcomes or {}).items():
                lines.append(f"{name} — {outcome}")
            if lines:
                get_logger(__name__).info(
                    "play tried: %s", "; ".join(lines))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _play_verify_full_track_state(path: str,
                                      item: Any) -> tuple[str, str]:
        """Three-valued full-track verdict (doc 7, §2):
        ``(state, why)`` with state verified / rejected / UNVERIFIED
        — see :func:`nomorals.media.library.full_track_verdict`.
        A downloaded file whose length could not be checked is
        UNVERIFIED, and delivery surfaces label it as such; it is
        never silently a pass.  Never raises.
        """
        try:
            expected = float((item or {}).get("duration") or 0)
        except (TypeError, ValueError):
            expected = 0.0
        from ...media.library import full_track_verdict

        return full_track_verdict(path, expected)

    @staticmethod
    def _play_verify_full_track(path: str, item: Any) -> str:
        """Verify a downloaded file is the FULL track, not a clip.

        When the candidate's metadata carries the track's real
        duration, the file on disk must not be preview-length: a
        ~30s file for a minutes-long song is a FAILED download (a
        source served its preview stream anyway), reported as such
        and never sent as the song.  Returns the rejection reason,
        or "" when the file is not rejected — note "" now covers
        both VERIFIED and UNVERIFIED outcomes; surfaces that report
        a delivery use :meth:`_play_verify_full_track_state` and
        label the unverified state honestly.  Never raises.
        """
        state, why = RuntimeMediaMixin._play_verify_full_track_state(
            path, item)
        return why if state == "rejected" else ""

    @staticmethod
    def _merge_search_attempts(resolution_attempts: Any,
                               detailed_attempts: Any) -> list[str]:
        """Full search audit: the detailed candidate sweep (every
        candidate source) plus any resolve-chain-only source
        (workspace-scan) it lacks.  Never raises."""
        try:
            merged = list(detailed_attempts or [])
            names = {a.split(" — ", 1)[0].strip() for a in merged
                     if isinstance(a, str)}
            missing = [a for a in (resolution_attempts or [])
                       if isinstance(a, str)
                       and a.split(" — ", 1)[0].strip() not in names]
            return missing + merged
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._merge_search_attempts", e)
            return list(detailed_attempts or []) or \
                list(resolution_attempts or [])

    @staticmethod
    def _format_tried_list(search_attempts: Any,
                           outcomes: dict[str, str]) -> list[str]:
        """Merge search-stage attempts with download-stage outcomes,
        one line per source actually tried.  Never raises."""
        try:
            merged: list[str] = []
            seen: set[str] = set()
            for attempt in (search_attempts or []):
                if not isinstance(attempt, str):
                    continue
                name = attempt.split(" — ", 1)[0].strip()
                seen.add(name)
                if name in outcomes:
                    merged.append(f"{attempt}, {outcomes[name]}")
                else:
                    merged.append(attempt)
            for name, outcome in outcomes.items():
                if name not in seen:
                    merged.append(f"{name} — {outcome}")
            return merged
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._format_tried_list", e)
            return list(search_attempts or [])

    def _play_send_with_fallthrough(
            self, engine: Any, first_item: dict[str, Any],
            fallbacks: Any, search_attempts: Any,
            chat: Any, *, match: Any = None,
            query: str = "") -> tuple[str, bool]:
        """Chat path: try candidates in order until one DOWNLOADS.

        A source whose search succeeded but whose download failed is a
        *download* failure, not success — the loop keeps trying the
        next source until one both finds AND downloads, or all
        candidates are exhausted.  Returns ``(text, sent)``: on
        success the text is the delivered item's own line; on total
        failure the honest failure (never receipts — the per-source
        tried trail goes to the journal via :meth:`_play_log_tried`,
        presentation law 2026-11-05).  The short /cookies hint shows
        ONLY when YouTube was actually attempted AND bot-blocked.
        Never raises.

        Preview law (owner, 2026-11-05): preview-class candidates
        (iTunes/Deezer 30s clips) are partitioned out FIRST — never
        download-attempted, never offered as picks; they only
        identify the track for the honest report.  A download whose
        file turns out preview-length for a full-length track is a
        failed download too (verified on arrival, never sent).
        Reasons quoted in the reply are cleaned: no raw asset URLs,
        no doubled failure wording.

        Delivery contract (owner, 2026-10-20): when ``match`` is a
        ``(title, artist)`` pair parsed from the request, ONLY
        candidates that confidently match it
        (:func:`nomorals.media.picklist.is_confident_match`) are
        download-attempted for auto-delivery — a found-but-blocked
        match keeps the chain trying OTHER sources for the SAME
        track.  Candidates that don't match are never silently
        substituted: when nothing confident downloads, the owner
        gets the found-but-blocked report plus a pick-list of the
        rest, and their pick downloads exactly what they chose.
        Without ``match`` the behaviour is the historical one.
        A match a source PRE-JUDGED blocked (its block classified
        before any attempt — preview policy, DRM, region, login)
        that is then served by a DIFFERENT source's copy is a
        substitution too: the delivered line declares it and names
        the specific block, and the file's tags carry the same
        declaration.

        Right-artifact law (owner, 2026-11-05): among confident
        matches the CANONICAL recording is attempted first — a
        candidate whose title carries an edit marker (Nightcore,
        remix, sped up, cover…) is attempted only after every clean
        match failed, and when an edit is what lands, its caption,
        final line and file tags SAY it's an edit of the requested
        track — never passed off as the original.
        """
        from ...media.cookies import BOT_COOKIE_SHORT_HELP, is_bot_detection_error
        from ...media.download_failures import (
            SOURCE_DEAD_KINDS,
            classify_failure,
            dead_kind_for_url,
            record_url_failure,
        )

        title = str(first_item.get("title") or "audio")
        items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for it in (first_item, *(fallbacks or [])):
            if not isinstance(it, dict):
                continue
            key = (str(it.get("kind", "")), str(it.get("path", "")))
            if key in seen:
                continue
            seen.add(key)
            items.append(it)
        # ── preview partition ───────────────────────────────────────
        # Preview-class candidates (iTunes/Deezer 30s clips) leave
        # the candidate set HERE, before any download or pick logic:
        # queueing a download for a preview is the defect this whole
        # path exists to prevent.  They survive only as preview_finds
        # — evidence for the honest report ("found it, but only as a
        # preview").  In gated mode only previews that actually match
        # the request count as finds; a namesake preview is not the
        # song and is dropped silently.
        preview_items = [it for it in items
                         if self._play_is_preview_item(it)]
        items = [it for it in items
                 if not self._play_is_preview_item(it)]
        preview_finds: list[dict[str, Any]] = []
        if preview_items:
            if match is not None:
                from ...media.picklist import is_confident_match

                q_t = str((match or ("", ""))[0] or "")
                q_a = str((match or ("", ""))[1] or "")
                preview_finds = [
                    it for it in preview_items
                    if is_confident_match(str(it.get("title", "") or ""),
                                          str(it.get("artist", "") or ""),
                                          q_t, q_a)]
            else:
                preview_finds = preview_items[:1]
        # ── delivery-contract partition ─────────────────────────────
        # Confident matches are attempted (best score first, chain
        # order breaking ties); everything else downloadable becomes
        # the pick-list if no match lands.  blocked_matches collects
        # the found-but-blocked facts for the final report.
        attempt_items = items
        pick_items: list[dict[str, Any]] = []
        blocked_matches: list[tuple[dict[str, Any], str]] = []
        gated = match is not None
        q_title = ""
        q_artist = ""
        if gated:
            from ...media.picklist import is_confident_match, match_confidence

            q_title = str((match or ("", ""))[0] or "")
            q_artist = str((match or ("", ""))[1] or "")
            confident: list[tuple[float, dict[str, Any]]] = []
            for it in items:
                it_title = str(it.get("title", "") or "")
                it_artist = str(it.get("artist", "") or "")
                if is_confident_match(it_title, it_artist,
                                      q_title, q_artist):
                    confident.append((match_confidence(
                        it_title, it_artist, q_title, q_artist), it))
                elif it.get("downloadable") is not False:
                    pick_items.append(it)
            # Right-artifact law: the canonical recording first —
            # edit-marked candidates (Nightcore/remix/cover…) sort
            # AFTER every clean confident match, best score first
            # within each class, chain order breaking ties — so when
            # a real source and a demoted one (mdundo, owner order
            # 2026-11-18) match equally well, the real source is
            # attempted first and mdundo surfaces only when nothing
            # real matches.  An edit may still win, but only when
            # no original is obtainable — and then it is labeled as
            # an edit (see the success path below).
            confident.sort(key=lambda pair: (
                1 if self._edit_marker_of(
                    str(pair[1].get("title", "") or "")) else 0,
                -pair[0],
                self._music_chain_index(str(pair[1].get("kind", "")))))
            attempt_items = [it for _score, it in confident]
        outcomes: dict[str, str] = {}
        # Failure memory (lane 26): source kinds that already failed
        # on THIS song — a 403 / timeout / 5xx / site error on one
        # copy means the source is hostile right now, so its other
        # copies for the same song are skipped instead of burning the
        # chain. dead_kind_reasons carries the kind that condemned
        # each. skipped_dead flags that at least one candidate was
        # skipped for a recorded failure (the preview-only line must
        # not fire when full tracks were found but dead).
        dead_kinds: set[str] = set()
        dead_kind_reasons: dict[str, str] = {}
        skipped_dead = False
        # Matches a source PRE-JUDGED undownloadable (never attempted:
        # its block was known before any byte moved).  When a
        # different source's copy is what lands, the delivery must
        # SAY it is a substitute and why — a policy-blocked track may
        # not be silently swapped for another source's recording.
        prejudged_blocks: list[tuple[dict[str, Any], str]] = []
        verify_notes: list[str] = []
        youtube_bot_blocked = False
        for attempt in (search_attempts or []):
            if isinstance(attempt, str) \
                    and attempt.startswith("youtube-search") \
                    and is_bot_detection_error(attempt):
                youtube_bot_blocked = True
        youtube_already_attempted = False
        for cand_no, item in enumerate(attempt_items, start=1):
            kind = str(item.get("kind", ""))
            name = self._KIND_TO_ATTEMPT_NAME.get(
                kind, f"{kind}-search" if kind else "unknown")
            cand_path = str(item.get("path", "") or "")
            if kind == "youtube" and youtube_already_attempted:
                # A previous download already ran the YouTube fallback
                # (e.g. SoundCloud DRM -> YouTube) — don't attempt
                # YouTube twice; record it as already attempted.
                outcomes[name] = \
                    "already attempted via fallback — blocked"
                continue
            if item.get("downloadable") is False:
                # The specific block, when the source classified it
                # (preview policy vs DRM vs region vs login) — the
                # generic "protected streams" survives only for
                # blocks no source classified.
                block_sentence = self._play_block_sentence(
                    item.get("block_reason"))
                blocked_reason = (
                    f"not downloadable — {block_sentence}"
                    if block_sentence else
                    "not downloadable (protected streams)")
                outcomes[name] = blocked_reason
                prejudged_blocks.append((item, blocked_reason))
                if gated:
                    blocked_matches.append((item, blocked_reason))
                continue
            # Progress note per candidate: counts as a heartbeat when
            # this runs as a background job; a no-op synchronously.
            # The same milestone edits the ack in place (one message,
            # edited as the job progresses — presentation law).
            from ...core.jobs import report_progress
            # ── failure memory ──────────────────────────────
            # A link that failed recently is not re-attempted: the
            # next source's copy goes first, and the skip is named
            # in the journal trail instead of burning minutes on
            # the same dead end (owner report 2026-11-17).
            recorded_kind = dead_kind_for_url(cand_path)
            if recorded_kind is not None:
                outcomes[name] = (
                    f"skipped — this link failed recently "
                    f"({recorded_kind})")
                skipped_dead = True
                if gated:
                    blocked_matches.append(
                        (item,
                         f"link failed recently ({recorded_kind})"))
                continue
            if kind in dead_kinds:
                outcomes[name] = (
                    f"skipped — {kind} already failed on this song "
                    f"({dead_kind_reasons.get(kind, '')})")
                skipped_dead = True
                if gated:
                    blocked_matches.append(
                        (item, f"{kind} failed on this song"))
                continue
            report_progress(
                f"trying {name} ({cand_no} of {len(attempt_items)})")
            self._job_ack_progress(
                f"⏳ {title} — trying "
                f"{name.removesuffix('-search')}…")
            # A new source takes over the measurement (P1): attempt
            # restarts at 1 and the previous source's percentage is
            # gone — it never described this transfer.
            from ...core.progress_events import emit_source
            emit_source(name)
            try:
                dl = engine.download(item)
            except Exception as exc:  # noqa: BLE001
                dl = {"ok": False, "reason": f"download error: {exc}"}
            if dl.get("ok"):
                path = dl["path"]
                dl_title = dl.get("title") or item.get("title", "audio")
                # Verify what arrived before anything is sent, via
                # the shared per-type acceptance contract (Phase 2):
                # a preview-length file for a full-length track is a
                # failed download, and so is a positively wrong file
                # (an HTML error page renamed .mp3) — record either,
                # keep falling through.  Only a REJECTION fails the
                # route; an UNVERIFIED property rides the facts and
                # the plain identity line claims no verification.
                from ...media.download_orchestrator import DownloadOrchestrator
                acc = DownloadOrchestrator().verify(
                    path, "song",
                    expected_duration=float(item.get("duration") or 0))
                if not acc.ok:
                    if acc.facts.get("verify_state") == "rejected":
                        outcomes[name] = (
                            f"downloaded a clip, not the track "
                            f"({acc.reason})")
                    else:
                        outcomes[name] = (
                            f"downloaded the wrong file "
                            f"({acc.reason})")
                    verify_notes.append(acc.reason)
                    # The URL served the wrong file — remember it, so
                    # the next attempt does not download the same
                    # clip / error page again.
                    record_url_failure(cand_path, "wrong_file",
                                       source=kind)
                    if gated:
                        blocked_matches.append((item, acc.reason))
                    continue
                outcomes[name] = "downloaded ✓"
                # The tried trail's only home is the journal now —
                # the chat gets the delivered item's line, no receipt.
                self._play_log_tried(search_attempts, outcomes)
                note = dl.get("note")
                if not note:
                    # No engine-level note — but a pre-judged blocked
                    # match from another source makes this delivery a
                    # substitution, and it is declared, not silent.
                    note = self._play_substitution_note(
                        item, prejudged_blocks)
                # Right-artifact labeling: an edit-marked winner is
                # delivered AS an edit of the requested track — the
                # caption, the final line and the file's own tags all
                # carry the request's identity plus the edit label,
                # never the fan-edit name passed off as the original.
                marker = self._edit_marker_of(
                    str(item.get("title", "") or "")) if gated else ""
                if marker:
                    req_title = q_title or str(dl_title)
                    req_artist = q_artist or str(
                        item.get("artist", "") or "")
                    tag_title = f"{req_title} ({marker} edit)"
                    comment = (
                        f"This copy is a {marker} edit of "
                        f"“{req_title}” — the original recording "
                        "wasn't obtainable.")
                    send_path = self._delivery_path(
                        path, tag_title, req_artist)
                    send_path = self._tag_delivery_copy(
                        send_path, tag_title, req_artist,
                        comment=comment)
                    line = (f"🎵 {req_title}"
                            + (f" — {req_artist}" if req_artist else "")
                            + f" · {marker} edit — the original "
                              "wasn't obtainable")
                else:
                    win_artist = str(dl.get("artist")
                                     or item.get("artist", "") or "")
                    send_path = self._delivery_path(
                        path, str(dl_title), win_artist)
                    send_path = self._tag_delivery_copy(
                        send_path, str(dl_title), win_artist,
                        comment=str(note or ""))
                    line = f"🎵 {dl_title}"
                    if win_artist and win_artist.lower() \
                            not in str(dl_title).lower():
                        line += f" — {win_artist}"
                if note:
                    line += f"\n({note})"
                try:
                    from ...core.progress_events import UPLOADING, emit_stage
                    emit_stage(UPLOADING, note="sending the file")
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        send_path, caption=line[:1024], kind=_send_kind_for(send_path))
                    # The queue must hold what was actually sent: if
                    # the winner differs from the first pick, swap
                    # the queued row so a later /play next doesn't
                    # play the failed first pick.
                    if (str(item.get("kind", "")), str(item.get("path", ""))) != \
                            (str(first_item.get("kind", "")),
                             str(first_item.get("path", ""))):
                        self._replace_queued_with(engine, first_item, item)
                    # Expression follow-on (doc 8 demonstrator 1,
                    # second half): the procedural animated cover,
                    # when the owner enabled it — after the track,
                    # never instead of it, never blocking it.
                    self._maybe_cover_video(
                        chat, send_path, str(dl_title),
                        str(item.get("artist", "") or ""))
                    return line, True
                except Exception as exc:  # noqa: BLE001
                    return (f"had the file but couldn't send it: {exc} "
                            f"(it's saved on the server at {path})"), False
            reason = str(dl.get("reason", "unknown error"))
            # Record the failure so the next /play never re-picks
            # this link — and so the sweep demotes it. A 403 /
            # timeout / 5xx / site error also condemns the source
            # for the rest of THIS song (its other copies are
            # skipped above); a generic error condemns only its own
            # URL.
            fail_kind = classify_failure(reason)
            record_url_failure(cand_path, fail_kind, source=kind)
            if fail_kind in SOURCE_DEAD_KINDS:
                dead_kinds.add(kind)
                dead_kind_reasons[kind] = fail_kind
            if gated:
                blocked_matches.append((item, reason))
            if dl.get("youtube_fallback_attempted"):
                youtube_already_attempted = True
                if is_bot_detection_error(reason):
                    youtube_bot_blocked = True
                    outcomes[name] = (
                        "download blocked — YouTube fallback also "
                        "bot-blocked")
                else:
                    outcomes[name] = (
                        f"download blocked "
                        f"({self._play_clean_reason(reason)}) — "
                        "YouTube fallback also tried")
                continue
            if kind == "youtube" and is_bot_detection_error(reason):
                youtube_bot_blocked = True
            outcomes[name] = (
                f"download blocked ({self._play_clean_reason(reason)})")
        # Total failure: the tried trail goes to the journal; the
        # chat gets the outcome and the found-but-blocked facts.
        self._play_log_tried(search_attempts, outcomes)
        # The pre-enqueued first pick never downloaded — drop its
        # queue row like the single-send path does, so the durable
        # queue doesn't keep a broken item.
        self._drop_matching_queued(engine, first_item)
        if gated and pick_items:
            # Nothing that MATCHES could be downloaded, but other
            # results exist: they are the owner's to choose from —
            # never auto-sent as if they were the request.
            offer = self._play_picklist_offer(
                pick_items, blocked_matches, preview_finds,
                query or title, chat,
                youtube_bot_blocked=youtube_bot_blocked)
            if offer:
                return offer, False
        attempted_download = any(
            isinstance(v, str)
            and v.startswith(("download blocked", "downloaded"))
            for v in outcomes.values())
        if preview_finds and not blocked_matches \
                and not attempted_download and not skipped_dead:
            # Previews were the ONLY thing any source had: the plain
            # honest-absence line instead of the generic failure
            # line.  Nothing was sent, nothing queued for download.
            return self._play_preview_line(preview_finds[0]), False
        lines = [f"couldn't download “{title}” from any source."]
        for blocked_item, blocked_reason in blocked_matches[:3]:
            blocked_line = self._play_blocked_line(
                blocked_item, blocked_reason)
            if blocked_line:
                lines.append(blocked_line)
        # An arrived-but-wrong file (a preview-length clip for a
        # full-length track) is its own honesty fact — it rides the
        # report even outside the gated path, where no blocked
        # lines are collected (inside it, the blocked lines above
        # already carry the same fact).
        if not gated:
            for note in verify_notes[:2]:
                lines.append(f"⚠ {note}.")
        if preview_finds:
            lines.append(self._play_preview_note(preview_finds[0]))
        if youtube_bot_blocked:
            lines.append(BOT_COOKIE_SHORT_HELP)
        return "\n".join(lines), False

    def _play_picklist_offer(
            self, pick_items: list[dict[str, Any]],
            blocked_matches: list[tuple[dict[str, Any], str]],
            preview_finds: list[dict[str, Any]],
            query: str, chat: Any,
            *, youtube_bot_blocked: bool = False,
            owner_key: str = "") -> str:
        """The no-confident-download outcome: honest report + choice.

        The reply carries (1) every found-but-blocked match BY NAME
        with its block reason (cleaned: no raw asset URLs, no doubled
        wording), (2) the preview truth when the track also turned
        up as a 30s clip, and (3) the standard pick-list over the
        remaining downloadable FULL-TRACK results — preview-class
        results are never in the list — stored in the PickCache so a
        number reply or a button tap downloads exactly the chosen
        candidate.  The tried-list is NOT in the reply (journal-only,
        presentation law 2026-11-05).  Returns "" when the pick-list
        can't be stored (caller falls back to the plain failure
        report).  Never raises.
        """
        # Owner scope (Living Chat P2, closing the P1 remainder):
        # the offer binds to the requesting user exactly like the
        # vague-query pick-list does — an explicit owner wins,
        # otherwise the dispatch's in-flight owner (play_owner_var,
        # set by the runtime's /play branch and carried into the
        # download job's thread by the registry's context copy)
        # applies; "" keeps the legacy chat slot. Without this,
        # the blocked-matches offer stayed chat-scoped while every
        # sibling pick surface was owner-scoped, so another user's
        # bare number in a group could consume the offer.
        owner_key = owner_key or play_owner_var.get("")
        try:
            from ...media.cookies import BOT_COOKIE_SHORT_HELP
            from ...media.picklist import PICK_LIMIT, PickCache, format_picklist
            from ...media.resolver import ResolvedAudio

            cands = [
                ResolvedAudio(
                    ok=True, path_or_url=str(it.get("path", "") or ""),
                    title=str(it.get("title", "") or it.get("path", "")),
                    artist=str(it.get("artist", "") or ""),
                    duration=float(it.get("duration", 0.0) or 0.0),
                    kind=str(it.get("kind", "") or ""),
                    source_name=str(it.get("kind", "") or ""),
                    downloadable=True)
                for it in pick_items[:PICK_LIMIT]]
            if not cands:
                return ""
            chat_key = ""
            try:
                # The chat's own key (thread-aware) so the
                # number-reply hook finds this list; fall back to the
                # plain platform:id form.
                chat_key = str(getattr(chat, "key", "") or "")
                if not chat_key:
                    chat_key = f"{chat.platform}:{chat.chat_id}"
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._play_picklist_offer", e)
                chat_key = ""
            token = PickCache.store(
                cands, chat_key, owner_key=owner_key) if chat_key else ""
            if not token:
                return ""
            lines = []
            if blocked_matches or preview_finds:
                lines.append(
                    f"couldn't download “{query}” — every match I "
                    "found was blocked or preview-only:")
            for it, reason in blocked_matches:
                line = self._play_blocked_line(it, reason)
                if line:
                    lines.append(line)
            for it in (preview_finds or [])[:1]:
                lines.append(self._play_preview_note(it))
            if blocked_matches or preview_finds:
                lines.append(
                    "the other results didn't match your request "
                    "closely enough to send one blind — pick one:")
            else:
                lines.append(
                    "none of the results matched your request closely "
                    "enough to send one blind — pick one:")
            if youtube_bot_blocked:
                lines.append(BOT_COOKIE_SHORT_HELP)
            lines.append(format_picklist(cands, query, token))
            return "\n".join(lines)
        except Exception as e:
            # the honest failure report the caller still has
            record_failure("runtime_media.RuntimeMediaMixin._play_picklist_offer:1806", e)
            return ""

    @staticmethod
    def _fmt_started(res: dict[str, Any], engine: Any) -> str:
        """One-line human summary of a play_* result."""
        st = res.get("status", "")
        cur = res.get("current", "") or res.get("title", "")
        if isinstance(cur, dict):
            cur = cur.get("title", "")
        via = res.get("via", "")
        if st == "playing":
            return f"▶ playing “{cur}” [{res.get('backend', via or '?')}]"
        if st == "no-backend":
            out = f"queued (no audio backend): {cur}\n  {res.get('hint', '')}"
            for key in ("stream_url", "watch_url"):
                if res.get(key):
                    out += f"\n  {key}: {res[key]}"
            return out
        return (f"status: {st} "
                f"{res.get('error') or res.get('hint', '')}".rstrip())

    # ── end _control_play ────────────────────────────────────────────────

    def _control_dj(self, tail: str, chat_key: str = "") -> str:
        """/dj [trending|<genre>] — Devon FM radio show.

        /dj            → full show: trending charts + own tracks, voice breaks
        /dj trending   → just list what's hot right now
        /dj <genre>    → show built around a genre (uk-drill, afrobeats, …)
        /dj live [genre,…] → LIVE set in this chat: real-time performance,
                             reads the room, takes requests, talks
        /dj live stop  → end the live set
        /dj mix <genre|songs…> → MIX-DROP: fetch real tracks, beatmatch +
                             harmonically mix into ONE continuous file,
                             drop it in chat. Then mix the next.
        /dj long [minutes] <genre|songs…> → LONG MIX: plan a 30-60 min
                             set (energy arc, drop-aware transitions) and
                             render it as resumable real-audio segments
                             on disk; the chat gets the tracklist.
        /dj desk [fetch <song>|live [genre|songs…]|eq <A|B> <band> <0..2>
                             |fx <A|B> <filter|echo|reverb|flanger> …
                             |fader|keylock|jog|crate|wave|record|provision]
                             → the DESK you play: real fetched tracks on
                             two decks with beat grid, hot cues, pads,
                             loops, crossfader and tempo sync — plus
                             3-band EQ, FX, jog/scratch, key lock,
                             crates, waveforms and recording.
        """
        from ...media.dj import DJ

        tail = (tail or "").strip()
        if tail.lower().startswith("long"):
            from ...media.dj_long import control_dj_long
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is None:
                return "run /dj long from a chat — the mix lands there."
            gateway = self.gateway

            def _dt_long(text: str) -> None:
                gateway.send(chat.platform,
                             f"{chat.platform}:{chat.chat_id}", text)
            return control_dj_long(tail[4:].strip(), chat_key=chat_key,
                                   deliver_text=_dt_long,
                                   context=self.context)
        if tail.lower().startswith("mix"):
            from ...media.dj_mixdrop import control_dj_mix
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is None:
                return "run /dj mix from a chat — the mix drops there."
            gateway = self.gateway

            def _da(path: str, caption: str) -> None:
                gateway.send_file(chat.platform,
                                  f"{chat.platform}:{chat.chat_id}",
                                  path, caption=caption[:1024], kind="audio")

            def _dt(text: str) -> None:
                gateway.send(chat.platform,
                             f"{chat.platform}:{chat.chat_id}", text)
            return control_dj_mix(tail[3:].strip(), chat_key=chat_key,
                                  deliver_audio=_da, deliver_text=_dt,
                                  context=self.context)
        if tail.lower().startswith("live"):
            from ...media.dj_live import control_dj_live
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is None:
                return "run /dj live from a chat — the DJ needs a room."
            gateway = self.gateway

            def _da(path: str, caption: str) -> None:
                gateway.send_file(chat.platform,
                                  f"{chat.platform}:{chat.chat_id}",
                                  path, caption=caption[:1024], kind="audio")

            def _dt(text: str) -> None:
                gateway.send(chat.platform,
                             f"{chat.platform}:{chat.chat_id}", text)
            return control_dj_live(tail[4:].strip(), chat_key=chat_key,
                                   deliver_audio=_da, deliver_text=_dt,
                                   context=self.context)
        if tail.lower().startswith("desk"):
            from ...media.dj_desk import control_dj_desk
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is None:
                return ("run /dj desk from a chat — the live set drops "
                        "there.")
            gateway = self.gateway

            def _da_desk(path: str, caption: str) -> None:
                gateway.send_file(chat.platform,
                                  f"{chat.platform}:{chat.chat_id}",
                                  path, caption=caption[:1024], kind="audio")

            def _dt_desk(text: str) -> None:
                gateway.send(chat.platform,
                             f"{chat.platform}:{chat.chat_id}", text)
            return control_dj_desk(tail[4:].strip(), chat_key=chat_key,
                                   deliver_audio=_da_desk,
                                   deliver_text=_dt_desk,
                                   context=self.context)
        dj = DJ(self.context)
        if tail.lower() == "trending":
            res = dj.fetch_trending()
            if not res.get("ok"):
                return f"charts unavailable: {res.get('reason', '?')}"
            lines = [f"🔥 trending ({res['source']}):"]
            for t in res["tracks"][:10]:
                lines.append(f"  #{t.get('rank', '?')} {t['title']} — {t['artist']}")
            return "\n".join(lines)

        genre = tail
        try:
            show = dj.build_show(genre=genre, n_tracks=4)
        except Exception as exc:  # noqa: BLE001
            return f"dj error: {exc}"
        if not show.get("ok"):
            return f"couldn't build the show: {show.get('reason', '?')}"
        lines = [f"📻 Devon FM ({show['genre']}) — "
                 f"{show['duration_s']}s, {len(show['tracklist'])} tracks"]
        for t in show["tracklist"]:
            mark = "▶" if t.get("status") == "played" else "⏭"
            lines.append(f"  {mark} {t['label']}")
        text = "\n".join(lines)
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None and show.get("path"):
            try:
                self.gateway.send_file(
                    chat.platform, f"{chat.platform}:{chat.chat_id}",
                    show["path"], caption="📻 Devon FM — full show", kind="audio")
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._control_dj", e)
        return text

    def _control_mix(self, tail: str, chat_key: str = "") -> str:
        """``/mix`` — shareable DJ mixes built from the queue.

        The surface lives in :mod:`nomorals.media.mix_chat`; this is
        the thin adapter that injects the gateway sender (audio /
        voice note) and the mixes workdir. ``/mix send`` delivers
        through the same gateway the DJ drops use.
        """
        from ...media.mix_chat import control_mix

        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is None:
            return "run /mix from a chat — mixes live in the chat they were built in."
        workdir = None
        settings = getattr(getattr(self, "context", None), "settings", None)
        home = getattr(settings, "home", "") if settings is not None else ""
        if home:
            workdir = str(Path(home) / ".devon" / "mixes")
        return control_mix(
            tail, chat_key=chat_key, sender=_MixGatewaySender(self.gateway),
            context=self.context, workdir=workdir,
        )

    def _control_produce(self, tail: str, chat_key: str = "") -> str:
        """ /produce — the Producer composes an ORIGINAL piece.

        * ``/produce`` (no args) → from your taste profile ("make me something")
        * ``/produce <spotify-url>`` → analyze the track's lane + record taste
        * ``/produce <vibe words>`` → compose from the description + record taste
        * ``/produce harder|faster|slower|…`` → iterate on the last production
        """
        from ...media.producer import Producer, describe_to_profile
        from ...media.taste import last_production_profile, load_taste, modify_profile

        tail = (tail or "").strip()
        store = load_taste(getattr(self, "context", None))
        producer = Producer(self.context)

        def _deliver(res: dict, intro: str) -> str:
            if not res.get("ok"):
                return f"couldn't produce: {res.get('reason', '?')}"
            lines = [f"{intro} {res['title']} — {res['bpm']:g} BPM, "
                     f"{res['key']} {res['mode']}"]
            lines.append(res["notes"])
            text = "\n".join(lines)
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None and res.get("path"):
                try:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        res["path"], caption=f"🎛 {res['title']} (original)", kind="audio")
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._deliver", e)
            return text

        try:
            # ── no args: from taste ──────────────────────────────────
            if not tail:
                prof = store.suggest_profile()
                if prof is None:
                    # cold start: something dark and driving, and say so
                    prof = describe_to_profile("dark driving edm")
                    note = ("no taste profile yet — made you something dark "
                            "and driving. tell me what you think and I'll "
                            "learn your taste.")
                else:
                    note = ("based on your taste profile — "
                            f"{store.profile.production_count} productions "
                            "and counting.")
                res = producer.produce_from_profile(prof, source_note=note)
                if res.get("ok"):
                    store.record_production(res["profile"], source="taste",
                                            ref="taste profile")
                return _deliver(res, "🎛 produced from your taste:")

            # ── iteration on the last production ─────────────────────
            low = tail.lower()
            iter_words = ("harder", "faster", "slower", "more melodic",
                          "softer", "darker", "lighter", "heavier",
                          "speed it up", "slow it down", "chill out")
            if len(tail.split()) <= 3 and any(w in low for w in iter_words):
                base = last_production_profile(store)
                if base is not None:
                    tweaked = modify_profile(base, tail)
                    if tweaked is not base:
                        note = (f"iterating on the last one — {tail}. "
                                f"{base.bpm:g}→{tweaked.bpm:g} BPM.")
                        res = producer.produce_from_profile(
                            tweaked, source_note=note)
                        if res.get("ok"):
                            store.record_production(res["profile"],
                                                    source="iterate", ref=tail)
                        return _deliver(res, "🎛 reworked:")

            # ── link or description ──────────────────────────────────
            # LLM-first: the model writes the song as a SongSpec, the
            # producer renders it.  Spotify links contribute their musical
            # DNA as style context; descriptions go straight to composition.
            from ...media.composer_llm import compose_song_spec
            if "open.spotify.com" in tail:
                from ...media.producer import analyze_reference
                ref_prof = analyze_reference(tail)
                style_ctx = (
                    f"{ref_prof.genre} at {ref_prof.bpm:g} BPM in "
                    f"{ref_prof.key} {ref_prof.mode}, mood: {ref_prof.mood or ', '.join(ref_prof.energy_words[:2])}"
                    if ref_prof.ok else "")
                spec = compose_song_spec(
                    self.context, tail, style_hint=style_ctx)
                res = producer.produce_from_spec(spec)
                if res.get("ok"):
                    store.record_production(
                        describe_to_profile(tail), source="link", ref=tail)
                return _deliver(res, "🎛 produced:")
            # pure description → LLM composes the song spec directly
            spec = compose_song_spec(self.context, tail)
            res = producer.produce_from_spec(spec)
            if res.get("ok"):
                store.record_production(describe_to_profile(tail),
                                        source="description", ref=tail)
            return _deliver(res, "🎛 produced:")
        except Exception as exc:  # noqa: BLE001
            return f"produce error: {exc}"

    def _control_like(self, tail: str) -> str:
        """/like [notes…] — the last production was good; learn from it."""
        from ...media.taste import load_taste
        store = load_taste(getattr(self, "context", None))
        return store.record_feedback(True, tail or "")

    def _control_dislike(self, tail: str) -> str:
        """"/dislike [notes…] — the last production missed; learn from it."""
        from ...media.taste import load_taste
        store = load_taste(getattr(self, "context", None))
        return store.record_feedback(False, tail or "not feeling it")

    def _control_cookies(self, tail: str,
                         message: Any = None) -> str:
        """/cookies — cookie state in two families.  The
        bundle family (import|status|clear) moves the
        owner's logged-in sessions into Devon's own
        browser profile; everything else is the YouTube
        cookie file status this command has always shown.
        Owner-only like /pair: control dispatch never
        reaches non-operators."""
        words = (tail or "").split()
        sub = words[0].lower() if words else ""
        if sub in ("import", "status", "clear"):
            return self._control_cookies_bundle(
                sub, words[1:], message)
        return self._control_cookies_youtube(tail)

    def _control_cookies_youtube(self, tail: str) -> str:
        """/cookies [check] — YouTube cookie file status and setup help."""
        from ...media.cookies import BOT_COOKIE_HELP, cookies_status
        st = cookies_status()
        found = st.get("found")
        valid = st.get("valid_youtube")
        # Pass #76: the browser-profile export is part of the store
        # downloads actually resolve at fetch time — the status must
        # say so, or it lies by omission (it used to report only the
        # user file while downloads ran on profile cookies).
        bvalid = st.get("browser_valid") or 0
        browser_line = ""
        if bvalid:
            domains = ", ".join(st.get("browser_domains") or [])
            browser_line = (
                f"🍪 browser profile: {bvalid} valid cookie"
                f"{'s' if bvalid != 1 else ''}"
                f"{f' ({domains})' if domains else ''} — downloads use "
                "these automatically for those sites.")
        elif st.get("browser_export"):
            browser_line = ("🍪 browser profile: a cookie export "
                            "exists, but every cookie in it is expired.")
        lines = []
        if found:
            if valid:
                lines.append(
                    f"✅ cookies file found: {found} "
                    f"({valid} valid YouTube cookie"
                    f"{'s' if valid != 1 else ''})")
            else:
                # A file that exists but holds no valid YouTube
                # cookies fails the bot check exactly like no file —
                # saying only "found" would be false confidence.
                lines.append(f"⚠️ cookies file found: {found} — but it "
                             "has no valid (non-expired) YouTube cookies.")
                lines.append("Re-export your browser cookies (see below) "
                             "— an expired file won't pass the bot check.")
                lines.append("")
                lines.append(BOT_COOKIE_HELP)
            if browser_line:
                lines.append(browser_line)
        elif bvalid:
            lines.append("❌ no YouTube cookies file found — but "
                         "downloads are not cookieless:")
            lines.append(browser_line)
        else:
            lines.append("❌ no YouTube cookies file found.")
            if browser_line:
                lines.append(browser_line)
            lines.append("")
            lines.append(BOT_COOKIE_HELP)
        if (tail or "").strip().lower() == "check" and not found:
            lines.append("")
            lines.append("Checked: " + ", ".join(st.get("checked", [])))
        return "\n".join(lines)

    def _control_cookies_bundle(self, sub: str,
                                rest: list[str],
                                message: Any = None) -> str:
        """/cookies import|status|clear — the owner's
        session bundles for Devon's browser profile.  The
        organ (social.chat.cookie_bundle) owns parsing,
        merging, journaling and rendering; this handler
        only resolves where things live and where the
        bundle file is."""
        from pathlib import Path

        from ...social.chat import cookie_bundle

        profile_dir = self._webshot_profile_dir(
            self._webshot_book())
        if profile_dir is None:
            return ("🍪 my browser profile isn't set up on "
                    "this host, so there is nowhere for a "
                    "cookie bundle to land.")
        context = getattr(self, "context", None)
        settings = getattr(context, "settings", None)
        ws_dir = getattr(settings, "workspace_dir", ".")
        journal_path = (Path(ws_dir) / "data" / "social"
                        / "personal" / "journal.jsonl")
        if sub == "status":
            return cookie_bundle.render_status(profile_dir)
        if sub == "clear":
            target = (rest[0] if rest else "").strip()
            if not target:
                return ("🍪 name what to clear: /cookies "
                        "clear <domain> for one site, or "
                        "/cookies clear all for everything "
                        "imported.")
            return cookie_bundle.clear_imported(
                profile_dir, target,
                journal_path=journal_path)
        src = self._cookie_bundle_source(message)
        if src is None:
            return ("🍪 attach the cookie bundle file to "
                    "the command, or reply to the message "
                    "holding it with /cookies import — I "
                    "take cookies.txt and JSON exports.")
        outcome = cookie_bundle.import_bundle_file(
            src, profile_dir, journal_path=journal_path)
        return cookie_bundle.render_import_reply(outcome)

    def _cookie_bundle_source(self, message: Any):  # type: ignore[no-untyped-def]
        """The bundle file for /cookies import, resolved
        the way file-consuming commands resolve media in
        this codebase: the command message's own attachment
        (a document first), then the replied message's
        already-downloaded bytes, then the replied
        message's platform handle downloaded on demand.
        Returns a Path or None."""
        from pathlib import Path

        if message is None:
            return None
        media_rows = list(
            getattr(message, "media", None) or [])
        for want_document in (True, False):
            for media in media_rows:
                path = str(getattr(media, "path", "") or "")
                if not path:
                    continue
                if want_document and str(
                        getattr(media, "kind", "")
                        or "") != "document":
                    continue
                return Path(path)
        meta = getattr(message, "meta", None) or {}
        local = meta.get("replied_media_local") or {}
        if isinstance(local, dict) and local.get("path"):
            return Path(str(local["path"]))
        replied = meta.get("replied_media") or {}
        if isinstance(replied, dict) and replied.get("file_id"):
            platform = str(getattr(
                getattr(message, "chat", None), "platform",
                "") or "")
            adapter = self._adapter_for(platform) \
                if platform else None
            download = getattr(adapter, "download_file",
                               None)
            if callable(download):
                try:
                    path = download(
                        str(replied["file_id"]),
                        str(replied.get("name")
                            or "cookies-bundle.txt"))
                except Exception:  # noqa: BLE001 - a
                    path = ""          # failed download is
                    # no bundle, not a crash in chat
                if path:
                    return Path(str(path))
        return None

    def _control_sham(self, tail: str, chat_key: str = "",
                      message: Any = None) -> str:
        """ /sham — identify the song in a replied-to voice note or audio.

        Reply to an audio/voice message with /sham, or send audio with
        /sham as the caption. /songrecon is the same command under the
        system's own name. Native-first: Devon's local fingerprint
        memory first; on a recording miss, her melody catalog offers
        ranked hypotheses for covers and hummed voice notes (never
        identifications); then AudD for the global catalogue, honest
        acoustic description when neither can name it. A recording
        identification links straight into /play.
        """
        try:
            return self._sham_identify(message)
        except Exception as exc:  # noqa: BLE001 - never raises
            return f"couldn't identify that audio ({exc})"

    @staticmethod
    def _sham_receipt(local: dict[str, Any]) -> dict[str, Any]:
        from ...audio import songrecon as _songrecon

        return _songrecon.receipt_of(local)

    def _sham_local_match_reply(self, local: dict[str, Any]) -> str:
        """Render a local hit from its SongRecon receipt."""
        from ...audio import songrecon as _songrecon

        return _songrecon.local_match_reply(local)

    def _sham_melodic_hypotheses(self, audio_path: str) -> str | None:
        """SR3 melodic step: ranked hypotheses after a class-1 miss.

        Voice notes are the hum intake — a hummed, sung or covered
        query is matched by tune against the enrolled melody
        catalog. Returns the rendered reply, or None when no
        hypothesis clears the floor (the AudD chain then proceeds
        exactly as before SR3). Never raises; hypotheses never
        render as identifications and carry no /play line.
        """
        from ...audio import songrecon as _songrecon

        return _songrecon.melodic_hypotheses(audio_path)

    def _sham_external_match_reply(
            self, res: dict[str, Any],
            local: dict[str, Any]) -> str:
        """Render an AudD hit, explicitly labelled as fallback."""
        from ...audio import songrecon as _songrecon

        return _songrecon.external_match_reply(res, local)

    def _sham_unmatched_reply(
            self, local: dict[str, Any], *, lead: str, reason: str,
            external_status: str, heard: str,
            external_searched: bool = False) -> str:
        """Render an honest non-match with the actual search scope."""
        from ...audio import songrecon as _songrecon

        return _songrecon.unmatched_reply(
            local, lead=lead, reason=reason,
            external_status=external_status, heard=heard,
            external_searched=external_searched)

    def _sham_identify(self, message: Any) -> str:
        """Core /sham logic. Returns the reply text. Never raises.

        The staging lives in :mod:`nomorals.audio.songrecon`
        (local fingerprint → melodic hypotheses → AudD → honest
        ears); this method only finds the audio in the message and
        opens AudD through the vault.
        """
        from ...audio import songrecon as _songrecon
        from ...connectors.audd import open_audd

        audio_path = self._sham_find_audio(message)
        if not audio_path:
            return (
                "nothing to identify — reply to a voice note or audio "
                "message with /sham, or send audio with /sham as the caption."
            )
        audd, unavailable = open_audd(
            getattr(self.context, "db", None),
            self.context, actor="shazam")
        return _songrecon.recognize(
            audio_path, audd=audd,
            audd_unavailable=unavailable)["text"]

    def _sham_find_audio(self, message: Any) -> str:
        """Locate the audio file for /sham. Returns a local path or "".

        1. Audio attached to the command message itself (caption flow).
        2. Replied-to audio on Telegram (file_id stashed in message meta,
           downloaded on demand via the adapter).
        Never raises.
        """
        try:
            media = getattr(message, "media", None) or []
            for m in media:
                # "voice" too: a voice note can carry a /sham caption,
                # and the Bot API now marks voice notes kind="voice"
                # instead of collapsing them to "audio".
                if getattr(m, "kind", "") in ("audio", "voice") and getattr(m, "path", ""):
                    return str(m.path)
            # Replied-to audio whose bytes the surface already
            # downloaded at inbound (WhatsApp quoted media, the
            # Telegram userbot's replied download): a local path.
            meta = getattr(message, "meta", None) or {}
            local = meta.get("replied_media_local") or {}
            if isinstance(local, dict) and local.get("path") \
                    and str(local.get("kind") or "") in (
                        "audio", "voice"):
                return str(local["path"])
            # Replied-to audio (Telegram): download on demand.
            file_id = str(meta.get("replied_audio_file_id") or "")
            if file_id and self.gateway is not None:
                adapter = self.gateway.adapters.get("telegram")
                download = getattr(adapter, "download_file", None)
                if callable(download):
                    path = download(file_id, "sham.ogg")
                    if path:
                        return path
        except Exception as e:
            record_failure("runtime_media.RuntimeMediaMixin._sham_find_audio", e)
        return ""

    def _control_musicnote(self, tail: str, chat_key: str = "") -> str:
        """ /musicnote find | create — answer the music voice-note ask.

        The ask (owner-only) stashes the clip in
        :data:`nomorals.voice.music_note.PENDING` when it goes out;
        tapping a button consumes the entry (a double-tap can never
        run Find/Create twice). Find runs the /sham music-ID path on
        the stashed clip; Create runs LLM-first production seeded by
        the clip's tempo/key/feel. Never raises.
        """
        from ...voice import music_note as _music_note

        action = (tail or "").strip().split()[0].lower() if (tail or "").strip() else ""
        if action not in ("find", "create"):
            return ("usage: /musicnote find | create — tap Find to identify "
                    "the song in your voice note, or Create to produce a "
                    "new song from it.")
        entry = _music_note.PENDING.take(chat_key)
        if entry is None:
            return ("no music voice note waiting — send a music voice note "
                    "first, then tap Find or Create.")
        path = str(entry.get("path") or "")
        heard = str(entry.get("heard") or "")
        try:
            if action == "find":
                from ...social.chat.base import ChatKind, ChatMessage, ChatRef, MediaRef
                platform, _, cid = (chat_key or "").partition(":")
                msg = ChatMessage(
                    chat=ChatRef(platform=platform or "telegram",
                                 chat_id=cid or "0", kind=ChatKind.DM),
                    incoming=True, text="", sender="owner",
                    media=[MediaRef(path=path, kind="voice",
                                    mime="audio/ogg",
                                    name=(path.rsplit("/", 1)[-1] or "note.ogg"))])
                return self._sham_identify(msg)
            return self._musicnote_produce(path, heard, chat_key)
        except Exception as exc:  # noqa: BLE001 - never raises
            return f"couldn't run /musicnote {action} ({exc})"

    def _musicnote_produce(self, path: str, heard: str,
                           chat_key: str = "") -> str:
        """Produce a NEW original song seeded by a music voice note.

        The clip's tempo/key (from the fingerprint analyzer) seed the
        Producer profile; the result is a new composition in that
        lane, never a copy of the clip. The production is recorded in
        taste memory, indexed for future /sham, and delivered as
        audio. Never raises.
        """
        from ...media.producer import Producer, describe_to_profile
        from ...media.taste import load_taste

        # Seed tempo/key from the clip — analysis failure falls back
        # to a driving default, never to a crash.
        seed_note = (heard or "").strip() or "music from your voice note"
        prof = describe_to_profile(f"{seed_note} driving electronic")
        try:
            from ...audio.fingerprint import analyze as _analyze
            a = _analyze(path) if path else None
            if a is not None and getattr(a, "ok", False):
                tempo = float(getattr(a, "tempo_bpm", 0.0) or 0.0)
                tconf = float(getattr(a, "tempo_confidence", 0.0) or 0.0)
                if tempo > 0 and tconf >= 0.15:
                    prof.bpm = tempo
                key = str(getattr(a, "key", "") or "").strip()
                kconf = float(getattr(a, "key_confidence", 0.0) or 0.0)
                if key and kconf >= 0.55:
                    # Analyzer keys look like "C" or "A minor" — split
                    # a trailing mode word off when present.
                    parts = key.split()
                    prof.key = parts[0]
                    if len(parts) > 1 and parts[1].lower() in (
                            "major", "minor"):
                        prof.mode = parts[1].lower()
        except Exception as e:  # noqa: not the path
            record_failure("runtime_media.RuntimeMediaMixin._musicnote_produce", e)
        try:
            producer = Producer(self.context)
            res = producer.produce_from_profile(
                prof,
                source_note=f"seeded by your music voice note ({seed_note}).")
            if not res.get("ok"):
                return f"couldn't produce: {res.get('reason', '?')}"
            ref_profile = res.get("profile")
            if ref_profile is not None:
                try:
                    store = load_taste(getattr(self, "context", None))
                    store.record_production(ref_profile, source="musicnote",
                                            ref=seed_note)
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._musicnote_produce:2313", e)
            try:
                if res.get("path"):
                    from ...audio.fingerprint import index_own
                    index_own(res["path"], title=str(res.get("title", "")),
                              artist="Devon")
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._musicnote_produce:2320", e)
            lines = [f"🎛 produced from your voice note: {res['title']} — "
                     f"{res['bpm']:g} BPM, {res['key']} {res['mode']}"]
            lines.append(res["notes"])
            text = "\n".join(lines)
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None and res.get("path"):
                try:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        res["path"], caption=f"🎛 {res['title']} (original)", kind="audio")
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._musicnote_produce:2332", e)
            return text
        except Exception as exc:  # noqa: BLE001 - never raises
            return f"couldn't produce: {exc}"

    def _produce_hook(self, message: Any) -> str | None:
        """NL music production for the owner DM (non-slash only).

        "make me something for the gym" → produce with a mood-derived
        profile.  "I like this" / "too slow" right after a production →
        taste feedback.  Returns a reply or None; never raises.
        """
        from ...media.producer import Producer, describe_to_profile
        from ...media.taste import detect_feedback, detect_produce_intent, load_taste, mood_hint

        text = (getattr(message, "text", None) or "").strip()
        if not text or text.startswith("/"):
            return None
        store = load_taste(getattr(self, "context", None))

        # feedback first — but only when there's a production to judge
        if store.profile.last_production:
            fb = detect_feedback(text)
            if fb is not None:
                liked, notes = fb
                return store.record_feedback(liked, notes)

        intent = detect_produce_intent(text)
        if intent is None:
            return None
        try:
            producer = Producer(self.context)
            hint = mood_hint(text)
            if "open.spotify.com" in intent:
                res = producer.produce(intent)
                src, ref = "link", intent
            elif hint:
                prof = describe_to_profile(" ".join(hint) + " electronic")
                # nudge bpm by context: gym → fast, chill → slow
                low = text.lower()
                if any(w in low for w in ("gym", "workout", "party", "hype",
                                          "pumped")):
                    prof.bpm = max(prof.bpm, 140.0)
                if any(w in low for w in ("chill", "relax", "focus", "study",
                                          "sleep")):
                    prof.bpm = min(prof.bpm, 110.0)
                note = f"mood read: {', '.join(hint)} — composed from that."
                res = producer.produce_from_profile(prof, source_note=note)
                src, ref = "mood", text
            else:
                res = producer.produce(intent)
                src, ref = "description", intent
            if not res.get("ok"):
                return f"couldn't produce: {res.get('reason', '?')}"
            ref_profile = res.get("profile")
            if ref_profile is not None:
                store.record_production(ref_profile, source=src, ref=ref)
            # index the production into Devon's local recognition memory —
            # from now on /sham knows this song without any API
            try:
                if res.get("path"):
                    from ...audio.fingerprint import index_own
                    index_own(res["path"], title=str(res.get("title", "")),
                              artist="Devon")
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._produce_hook", e)
                _log.debug("produce fingerprint index failed",
                           exc_info=True)
            lines = [f"🎛 produced: {res['title']} — {res['bpm']:g} BPM, "
                     f"{res['key']} {res['mode']}"]
            lines.append(res["notes"])
            chat_key = getattr(getattr(message, "chat", None), "key", "")
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None and res.get("path"):
                try:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        res["path"], caption=f"🎛 {res['title']} (original)", kind="audio")
                except Exception as e:
                    record_failure("runtime_media.RuntimeMediaMixin._produce_hook:2411", e)
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            _log.exception("produce hook failed")
            return f"produce error: {exc}"

    def _control_video(self, tail: str, chat_key: str = "",
                       message: Any = None) -> str:
        """/video <query> [platform] | /video download <url> [audio] | platforms.

        Devon Studio generation lives here too (additive — the search
        paths below are untouched):
        /video gen "<prompt>" [--backend auto|ltx|wan|motion] [--seconds N]
        /video qc <path> | /video qc "<prompt>" [--seconds N]
        /video explain <preset> <content> [--values "a|b"] [--seconds N]
        /video motion lyric <audio> "<lyrics>" [--preset NAME]
        /video motion visual <audio> [--preset phonk]
        /video motion slideshow <img1,img2> [audio]
        /video motion trailer <clip1,clip2> [--title T]
        /video capability | /video list
        Watching lives here too: /video watch <url|path> [question]
        (owner-only — it spends vision-model calls per frame).
        Briefing lives here too: /video brief <url|path> (owner-only —
        it downloads the video and transcribes it when no captions
        exist), /video brief list, /video brief show <id>.
        """
        from ...media.video import _PLATFORM_SITES, VideoFinder

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /video <query> [platform]  |  "
                    "/video download <url> [audio]  |  /video platforms\n"
                    "studio: /video gen \"<prompt>\" | /video motion … | "
                    "/video capability | /video list\n"
                    "watch: /video watch <url|path> [question]\n"
                    "brief: /video brief <url|path> | /video brief list | "
                    "/video brief show <id>")
        words = tail.split()
        first = words[0].lower()
        if first == "watch":
            return self._control_video_watch(
                tail[len(words[0]):].strip(),
                chat_key=chat_key, message=message)
        if first == "brief":
            return self._control_video_brief(
                tail[len(words[0]):].strip(),
                chat_key=chat_key, message=message)
        # studio subcommands — "motion" only triggers on a known product
        # kind so searches like "/video motion graphics" still search
        _motion_kinds = ("lyric", "visual", "slideshow", "trailer")
        if first in ("gen", "generate", "capability", "list", "chain",
                     "qc", "explain") or (
                first == "motion" and len(words) > 1
                and words[1].lower() in _motion_kinds):
            return self._control_video_studio(first,
                                              tail[len(words[0]):].strip())
        if words[0].lower() == "platforms":
            return "platforms: " + ", ".join(sorted(_PLATFORM_SITES))
        if words[0].lower() == "download":
            rest = words[1:]
            audio = any(w.lower() == "audio" for w in rest)
            url = " ".join(w for w in rest if w.lower() != "audio").strip()
            if not url:
                return "usage: /video download <url> [audio]"
            try:
                out = VideoFinder(self.context).download(url,
                                                         audio_only=audio)
                return f"⬇ downloaded: {out.get('path', out)}"
            except Exception as exc:  # noqa: BLE001
                return f"download failed: {exc}"
        query = tail
        platform = ""
        if words[-1].lower() in _PLATFORM_SITES and len(words) > 1:
            platform = words[-1].lower()
            query = " ".join(words[:-1])
        try:
            # adaptive breadth: the query's own phrasing sets the result count
            n = adaptive_result_limit(query, base=8, floor=4, ceiling=12)
            out = VideoFinder(self.context).find(query, max_results=n,
                                                 platform=platform)
        except Exception as exc:  # noqa: BLE001
            return f"video error: {exc}"
        if not out["results"]:
            return (f"no video results for “{query}”"
                    + (f" on {platform}" if platform else "")
                    + " — try a broader query or another platform.")
        lines = [f"🎬 {out['count']} result(s) for “{query}”:"]
        for i, r in enumerate(out["results"], 1):
            title = (r.get("title") or r["url"])[:64]
            dur = f"  [{r['duration']}]" if r.get("duration") else ""
            extra = f" · {r['author']}" if r.get("author") else ""
            lines.append(f"  {i}. {title}{dur}{extra}\n     {r['url']}")
        lines.append("download: /video download <url>")
        return "\n".join(lines)

    def _control_video_brief(self, tail: str, chat_key: str = "",
                             message: Any = None) -> str:
        """`/video brief <url|path> | list | show <id>` — a long video as
        a sectioned brief.

        Captions first, ASR fallback, honest refusal when neither
        exists — the brief names which transcript it was built from.
        Owner-only: every brief downloads the video (when no captions
        exist) and transcribes it, the same cost profile as /video
        watch, so the same gate. The public /video grant admits
        everyone, so this subverb gates itself. Never raises.
        """
        from ...media.video_brief import (
            VideoBrief,
            VideoBriefStore,
            brief_video,
            format_brief_reply,
        )
        from ...partner.gating import is_owner_chat

        def _send(text: str) -> str:
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None:
                return self._send_long_checked(chat.platform, chat,
                                               text[:6000])
            return text[:6000]

        if message is not None:
            if not self._is_operator(message):
                return "that one's just for the owner."
        elif chat_key:
            chat = self._ref_from_key(chat_key)
            if not is_owner_chat(chat, owner_chats=getattr(
                    self, "_owner_chats", ())):
                return "that one's just for the owner."
        else:
            return "that one's just for the owner."

        settings = getattr(self.context, "settings", None)
        home = None
        if settings is not None:
            home = getattr(settings, "home_path", None) or getattr(
                settings, "home", None)
        store = VideoBriefStore(home=home)
        tail = (tail or "").strip()
        tokens = tail.split()
        if tokens and tokens[0].lower() == "list":
            rows = store.list()
            if not rows:
                return _send("no video briefs yet — /video brief <url> "
                             "to make the first one")
            lines = ["🎬 video briefs:"]
            for row in rows:
                lines.append(
                    f"  {row['id']} — {row['title']} "
                    f"({row['sections']} sections, "
                    f"{row['transcript_source'] or '?'})")
            return _send("\n".join(lines))
        if tokens and tokens[0].lower() == "show":
            wanted = tokens[1] if len(tokens) > 1 else ""
            record = store.get(wanted) if wanted else None
            if record is None:
                return _send("no stored brief "
                             + (f"'{wanted}'" if wanted else "(give an id)"))
            return _send(format_brief_reply(
                VideoBrief.from_dict(record)))
        abstractive = "--abstractive" in tokens
        targets = [t for t in tokens if t != "--abstractive"]
        if not targets:
            return _send("usage: /video brief <url|path> [--abstractive] | "
                         "/video brief list | /video brief show <id>")
        target = targets[0]
        try:
            brief = brief_video(
                target, home=home, context=self.context,
                abstractive=abstractive, store=store)
            return _send(format_brief_reply(brief))
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return _send(f"video brief error: {exc}")

    def _control_video_watch(self, tail: str, chat_key: str = "",
                             message: Any = None) -> str:
        """`/video watch <url|path> [question]` — Devon truly watches a video.

        Attached video wins over any path argument (same shape as
        /vision's attached image); a URL is downloaded through the
        same VideoFinder route /video download uses; a path is watched
        as-is. The organ samples frames (vision) and transcribes the
        audio (STT) and answers from both — or names the rung that is
        down, never inventing what it could not see or hear.

        Owner-only: every watch spends vision-model calls (one per
        sampled frame) and sends frames to the model provider — the
        same cost/privacy profile as /vision, so the same gate. The
        public /video grant admits everyone, so this subverb gates
        itself. Never raises.
        """
        import shlex

        from ...media.video_watch import (
            render_watch,
            watch_capabilities,
            watch_video,
        )
        from ...partner.gating import is_owner_chat

        def _send(text: str) -> str:
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None:
                return self._send_long_checked(chat.platform, chat,
                                               text[:6000])
            return text[:6000]

        if message is not None:
            if not self._is_operator(message):
                return "that one's just for the owner."
        elif chat_key:
            chat = self._ref_from_key(chat_key)
            if not is_owner_chat(chat, owner_chats=getattr(
                    self, "_owner_chats", ())):
                return "that one's just for the owner."
        else:
            return "that one's just for the owner."

        tail = (tail or "").strip()
        try:
            tokens = shlex.split(tail)
        except ValueError:
            tokens = tail.split()
        if tokens and tokens[0].lower() == "info":
            caps = watch_capabilities(self.context)
            lines = ["🎬 video watch on this machine:"]
            lines.append("  ffmpeg: "
                         + ("yes (" + caps["ffmpeg_path"] + ")"
                            if caps["ffmpeg"] else "NO"))
            lines.append("  vision model: "
                         + ("yes" if caps["vision"] else "NO"))
            lines.append("  speech-to-text: "
                         + str(caps["stt"] or "NO"))
            for note in caps["notes"]:
                lines.append(f"  ⚠️ {note}")
            return _send("\n".join(lines))

        # attached video wins over any path argument
        attached = ""
        if message is not None:
            for media in getattr(message, "media", None) or []:
                if getattr(media, "kind", "") in ("video", "video_note"):
                    attached = getattr(media, "path", "") or ""
                    break

        def _looks_like_target(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return False
            if s.startswith(("http://", "https://", "/", "./", "~/")):
                return True
            if len(s) > 3 and "." in s:
                low = s.lower()
                if any(low.endswith(ext) for ext in
                       (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")):
                    return True
            return False

        targets = [t for t in tokens if _looks_like_target(t)]
        if attached:
            targets = [attached] + [t for t in targets if t != attached]
        if not targets:
            return _send("usage: /video watch <url|path> [question] — "
                         "attach a video, or pass its link or path")
        target = targets[0]
        question = " ".join(t for t in tokens if t != target).strip()
        try:
            if target.startswith(("http://", "https://")):
                from ...media.video import VideoFinder

                out = VideoFinder(self.context).download(
                    target, audio_only=False)
                path = str(out.get("path") or out)
            else:
                path = target
            result = watch_video(path, context=self.context,
                                 question=question)
            return _send(render_watch(result))
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return _send(f"video watch error: {exc}")

    def _control_video_studio(self, action: str, tail: str) -> str:
        """Devon Studio generation subcommands for /video. Never raises."""
        try:
            if action in ("gen", "generate"):
                return self._video_studio_gen(tail)
            if action == "explain":
                return self._control_video_explain(tail)
            if action == "qc":
                return self._video_studio_qc(tail)
            if action == "motion":
                return self._video_studio_motion(tail)
            if action == "capability":
                from ...media.videogen import capability_report
                return capability_report()
            if action == "list":
                from ...media.motion_studio._core import ledger_status, read_ledger
                ledger = read_ledger(limit=10)
                if not ledger:
                    return "no studio renders yet."
                lines = ["🎬 recent studio renders:"]
                for e in ledger:
                    lines.append(f"  {e.get('kind', '?'):22} {e.get('path', '?')}")
                total = ledger_status()["total_entries"]
                if total > len(ledger):
                    lines.append(f"  …and {total - len(ledger)} more in the "
                                 f"full ledger ({total} total)")
                return "\n".join(lines)
            if action == "chain":
                return ("usage: /video chain needs a scenes JSON file — "
                        "use `nm video chain scenes.json` on the CLI")
            return f"unknown /video studio action {action!r}"
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return f"video studio error: {exc}"

    def _control_alternatives(self, tail: str) -> str:
        """Software alternatives: /alternatives <software> [flags].

        Crowd-ranked alternatives from the AlternativeTo catalogue
        (seeded, grown by live fetch) — Devon herself first where she
        does the same job. Flags: --platform <name>, --free,
        --open-source, --limit N. Never raises: bad input returns
        usage, lookup failures name the problem.
        """
        import re

        from ...sources.software import (
            AlternativeToSource,
            SoftwareCatalog,
            format_alternatives_reply,
        )

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /alternatives <software> [--platform <name>] "
                    "[--free | --open-source] [--limit N]\n"
                    "example: /alternatives Photoshop --platform linux "
                    "--free")
        platform = None
        license_tier = None
        open_source = None
        limit = 10
        m = re.search(r"--platform\s+(\S+)", tail)
        if m:
            platform = m.group(1)
            tail = (tail[:m.start()] + tail[m.end():]).strip()
        m = re.search(r"--limit\s+(\d+)", tail)
        if m:
            try:
                limit = max(1, int(m.group(1)))
            except ValueError:
                limit = 10
            tail = (tail[:m.start()] + tail[m.end():]).strip()
        if "--open-source" in tail:
            open_source = True
            tail = tail.replace("--open-source", " ").strip()
        if "--free" in tail:
            license_tier = "Free"
            tail = tail.replace("--free", " ").strip()
        name = re.sub(r"\s+", " ", tail).strip()
        if not name:
            return ("usage: /alternatives <software> [--platform <name>] "
                    "[--free | --open-source] [--limit N]")
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — seeds still answer
                home = None
            source = AlternativeToSource(catalog=SoftwareCatalog(home))
            results = source.alternatives(
                name, platform=platform, license_tier=license_tier,
                open_source=open_source, limit=limit)
            return format_alternatives_reply(name, results)
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"alternatives lookup failed: {exc}"

    def _control_compute(self, tail: str) -> str:
        """Computed answers: /compute <question> [--rung wolfram|sympy|stdlib].

        Arithmetic, units, dates and symbolic math on the local
        ladder; curated facts through WolframAlpha when an AppID is
        configured. Every answer names its rung and shows its
        derivation; a question no rung can answer says so and names
        what would help. Never raises: bad input returns usage.
        """
        import re

        from ...codews.compute import compute_answer, format_compute_reply

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /compute <question> "
                    "[--rung wolfram|sympy|stdlib]\n"
                    "example: /compute integrate x^2 sin(x)")
        rung = "auto"
        m = re.search(r"--rung\s+(\S+)", tail)
        if m:
            rung = m.group(1).lower()
            tail = (tail[:m.start()] + tail[m.end():]).strip()
        question = re.sub(r"\s+", " ", tail).strip()
        if not question:
            return ("usage: /compute <question> "
                    "[--rung wolfram|sympy|stdlib]")
        try:
            appid = ""
            try:
                settings = getattr(self, "settings", None)
                wolfram = getattr(settings, "wolfram", None) \
                    if settings is not None else None
                appid = str(getattr(wolfram, "appid", "") or "") \
                    if wolfram is not None else ""
            except Exception:  # noqa: BLE001 — local rungs still answer
                appid = ""
            comp = compute_answer(question, appid=appid or None, rung=rung)
            return format_compute_reply(comp)
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"compute failed: {exc}"

    def _control_breach(self, tail: str) -> str:
        """Breach exposure: /breach <email|domain> | password <pw> | watch ….

        Owner-only. An email/domain check names every breach with its
        date and the data classes it exposed (HIBP v3, key-gated). A
        password check is Pwned Passwords k-anonymity — only 5 hash
        characters leave the box; the password itself is never stored.
        The watch keeps the owner's own addresses and pages him when
        a new breach names one. Never raises: bad input returns usage.
        """
        from ...security.breach import (
            BreachWatchStore,
            check_account,
            check_password,
            format_breach_reply,
            run_breach_watch_tick_once,
        )

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /breach <email|domain> | /breach password "
                    "<password> | /breach watch add <email> | remove "
                    "<email> | list | status | check")
        parts = tail.split(None, 1)
        verb = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""
        if verb == "password":
            if not rest:
                return "usage: /breach password <password>"
            return format_breach_reply(check_password(rest))
        if verb == "watch":
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — watch needs a home
                home = None
            if home is None:
                return "breach watch failed: no home directory"
            from pathlib import Path

            path = Path(home) / "data" / "breach_watch.json"
            store = BreachWatchStore(path)
            wparts = rest.split(None, 1)
            wverb = wparts[0].lower() if wparts else "list"
            warg = wparts[1].strip() if len(wparts) > 1 else ""
            if wverb == "add":
                if not warg or "@" not in warg:
                    return "usage: /breach watch add <email>"
                if store.add(warg):
                    return (f"watching {warg.lower()} — a new breach "
                            "naming it pages you once.")
                return f"{warg.lower()} is already watched."
            if wverb == "remove":
                if store.remove(warg):
                    return f"no longer watching {warg.lower()}."
                return f"{warg.lower()} was not watched."
            if wverb == "list":
                watched = store.list_watched()
                if not watched:
                    return ("no watched addresses yet — /breach watch "
                            "add <email> to watch one.")
                return "watched addresses:\n" + "\n".join(
                    f"  • {addr}" for addr in watched)
            if wverb == "status":
                watched = store.list_watched()
                last = store.last_check()
                lines = [f"watched: {len(watched)} address(es)"]
                for addr in watched:
                    seen = store.seen_names(addr)
                    lines.append(f"  • {addr} — {len(seen)} known "
                                 f"breach(es)")
                if last:
                    import datetime

                    lines.append("last check: " + datetime.datetime.fromtimestamp(
                        last).strftime("%Y-%m-%d %H:%M"))
                else:
                    lines.append("last check: never")
                return "\n".join(lines)
            if wverb == "check":
                result = run_breach_watch_tick_once(
                    context=self, notifier=None, state_path=path)
                if not result.get("ok"):
                    return ("breach watch check failed: "
                            f"{result.get('error') or 'unknown error'}")
                new = result.get("new") or []
                if not new:
                    return (f"watch check done — {len(result.get('checked') or [])} "
                            "address(es) checked, nothing new.")
                return ("new breach(es) on watched addresses:\n" + "\n".join(
                    f"  • {name}" for name in new))
            return ("usage: /breach watch add <email> | remove <email> | "
                    "list | status | check")
        if verb in {"help", "--help"}:
            return ("usage: /breach <email|domain> | /breach password "
                    "<password> | /breach watch add <email> | remove "
                    "<email> | list | status | check")
        try:
            key = ""
            try:
                settings = getattr(self, "settings", None)
                osint = getattr(settings, "osint", None) \
                    if settings is not None else None
                key = str(getattr(osint, "hibp_key", "") or "") \
                    if osint is not None else ""
            except Exception:  # noqa: BLE001 — no key is an honest refusal
                key = ""
            return format_breach_reply(check_account(tail, key=key))
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"breach check failed: {exc}"

    def _control_soundtrack(self, tail: str) -> str:
        """Soundtrack index: what song plays in a film or TV scene.

        Public. /soundtrack <show|film> [S1E2] | song <track>
        [by <artist>] | clip <path> | add <work> | S1E2 | <track> —
        <artist> | <scene> | vote <work> | <track> | confirm|deny.
        Every placement shows its provenance and confidence; curated
        entries are labelled, never presented as verified. An unknown
        work or song is refused honestly. Never raises: bad input
        returns usage.
        """
        from ...music_sources.placements import (
            Placement,
            format_placement_reply,
            format_reverse_reply,
            lookup,
            open_store,
            parse_soundtrack_query,
            placements_for_clip,
        )

        tail = (tail or "").strip()
        usage = ("usage: /soundtrack <show|film> [S1E2] | song <track> "
                 "[by <artist>] | clip <path> | add <work> | S1E2 | "
                 "<track> — <artist> | <scene> | vote <work> | <track> | "
                 "confirm|deny")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb == "song":
                if not rest:
                    return "usage: /soundtrack song <track> [by <artist>]"
                track, _, artist = rest.partition(" by ")
                works = store.works_for_track(track.strip(),
                                              artist.strip() or None)
                if not works:
                    return (f"no placements recorded for “{track.strip()}” "
                            f"yet — add one with /soundtrack add.")
                return format_reverse_reply(works)
            if verb == "clip":
                if not rest:
                    return "usage: /soundtrack clip <path>"
                from ...audio.fingerprint import identify as _identify

                ans = placements_for_clip(store, rest, identify_fn=_identify)
                if ans.get("ok"):
                    return format_reverse_reply(ans["placements"])
                return format_placement_reply(ans)
            if verb == "add":
                fields = [f.strip() for f in rest.split("|")]
                if len(fields) < 3 or not fields[0] or not fields[2]:
                    return ("usage: /soundtrack add <work> | S1E2 | "
                            "<track> — <artist> | <scene>")
                parsed = parse_soundtrack_query(fields[1])
                track, _, artist = fields[2].partition(" — ")
                if not artist:
                    track, _, artist = fields[2].partition(" - ")
                placement = Placement(
                    work_kind="tv" if parsed["episode"] is not None
                    else "film",
                    work_title=fields[0],
                    season=parsed["season"] if parsed["season"] is not None
                    else -1,
                    episode=parsed["episode"] if parsed["episode"] is not None
                    else -1,
                    scene_note=fields[3] if len(fields) > 3 else "",
                    track_title=track.strip(),
                    track_artist=artist.strip(),
                    provenance="owner", confidence=0.95)
                added = store.add(placement)
                return (f"recorded: {placement.track_title} — "
                        f"{placement.track_artist} in "
                        f"{placement.work_title}"
                        + (" — already recorded" if not added else ""))
            if verb == "vote":
                fields = [f.strip() for f in rest.split("|")]
                if len(fields) < 3 or fields[2].lower() not in (
                        "confirm", "deny"):
                    return ("usage: /soundtrack vote <work> | <track> | "
                            "confirm|deny")
                resolved = store.resolve_work(fields[0])
                if not resolved:
                    return f"“{fields[0]}” is not in the catalogue"
                updated = store.vote(resolved[0]["work_id"], fields[1],
                                     confirm=(fields[2].lower() == "confirm"))
                if updated is None:
                    return (f"no community placement for “{fields[1]}” in "
                            f"{resolved[0]['work_title']} to vote on")
                return (f"{updated.track_title} — {updated.trust_label()}")
            if verb in {"help", "--help"}:
                return usage
            return format_placement_reply(lookup(store, tail))
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"soundtrack failed: {exc}"

    def _control_evidence(self, tail: str) -> str:
        """Research answers from real papers (Elicit core).

        Public. /evidence <research question> asks the open scholarly
        corpus (OpenAlex, Semantic Scholar, arXiv), screens the paper
        set, extracts a cited table per paper (claim, method, result,
        stance — every cell quotes its paper verbatim) and synthesises
        what agrees, what disagrees and what is still unanswered.
        /evidence list shows past reviews; /evidence show <id>
        re-renders one; /evidence table <id> re-exports its CSV.
        A question with no papers is refused honestly — nothing is
        ever invented. Never raises: bad input returns usage.
        """
        from ...research.extraction import (
            EvidenceTable,
            answer_question,
            export_csv,
            format_evidence_reply,
            open_review_store,
        )

        tail = (tail or "").strip()
        usage = ("usage: /evidence <research question> | list | "
                 "show <id> | table <id>")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_review_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                reviews = store.list()
                if not reviews:
                    return "no evidence reviews yet — ask me a research " \
                           "question with /evidence <question>"
                lines = ["evidence reviews:"]
                for review in reviews[:10]:
                    lines.append(
                        f"  {review['id']} — {review['question']} "
                        f"({review['papers_included']} papers)")
                return "\n".join(lines)
            if verb in {"show", "table"}:
                if not rest:
                    return f"usage: /evidence {verb} <id>"
                review = store.get(rest)
                if review is None:
                    return f"no evidence review “{rest}”"
                if verb == "show":
                    return format_evidence_reply({**review, "ok": True})
                table = EvidenceTable.from_dict(review.get("table") or {})
                path = review.get("csv_path") or ""
                if not path:
                    from ...research.extraction import (
                        TABLE_DIRNAME,
                        _default_home,
                    )

                    path = export_csv(
                        table, _default_home(home) / TABLE_DIRNAME
                        / f"{rest}.csv")
                    review["csv_path"] = path
                    store.save(review)
                return (f"extraction table for “{review['question']}”: "
                        f"{len(table.cells)} cells over "
                        f"{len(table.paper_set.papers)} papers — {path}")
            return format_evidence_reply(answer_question(home, tail))
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"evidence failed: {exc}"

    def _control_devask(self, tail: str) -> str:
        """Developer answers with cited sources and executed code.

        Public. /devask <question or pasted stack trace> asks the dev
        source pack (Devon's own code index first, then the live web
        backends) and synthesises a direct answer — explanation plus
        code — with every claim cited to a real source. Every
        self-contained code block is EXECUTED (through the run_code
        tool when the runtime's tool registry is reachable) before
        it is presented; a block that did not run green is labelled
        not verified, never presented as working. /devask list shows
        past answers; /devask show <id> re-renders one. A question
        no source answers is refused honestly — nothing is ever
        invented. Never raises: bad input returns usage.
        """
        from ...research.dev_answer import (
            answer_dev_question,
            format_dev_answer_reply,
            open_dev_answer_store,
        )

        tail = (tail or "").strip()
        usage = "usage: /devask <question or pasted stack trace> | " \
                "list | show <id>"
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_dev_answer_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                answers = store.list()
                if not answers:
                    return "no dev answers yet — ask me a developer " \
                           "question with /devask <question>"
                lines = ["dev answers:"]
                for row in answers[:10]:
                    lines.append(
                        f"  {row['answer_id']} — {row['question']}")
                return "\n".join(lines)
            if verb == "show":
                if not rest:
                    return "usage: /devask show <id>"
                row = store.get(rest)
                if row is None:
                    return f"no dev answer “{rest}”"
                return format_dev_answer_reply({**row, "ok": True})
            context = getattr(self, "context", None)
            tools = getattr(context, "tools", None) if context else None
            run_fn = None
            if tools is not None:
                def _run(code: str, lang: str):
                    try:
                        outcome = tools.call(
                            "run_code", action="run", code=code,
                            lang=lang, timeout=30.0)
                    except Exception:  # noqa: BLE001 — unverified then
                        return None
                    if not getattr(outcome, "ok", False):
                        return None
                    value = getattr(outcome, "value", None)
                    return value if isinstance(value, dict) else None

                run_fn = _run
            llm_fn = None
            if context is not None:
                from ...tools.paper_reader import _llm_from_context

                llm_fn = _llm_from_context(context)
            return format_dev_answer_reply(answer_dev_question(
                home, tail, store=store, run_fn=run_fn, llm_fn=llm_fn))
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"devask failed: {exc}"

    def _control_explain(self, tail: str) -> str:
        """Explain a shell command fragment by fragment.

        Public. /explain <shell command> annotates each fragment —
        the command, every option, arguments, pipes and redirects —
        with the exact help text for that fragment, grounded in the
        man pages of the binaries actually installed on this box,
        in --help output harvested in the sandbox, or in tldr
        pages. A fragment with no grounded doc is marked unknown,
        never filled in silently. /explain list shows past
        explanations; /explain show <id> re-renders one;
        /explain ingest rebuilds the man corpus; /explain status
        reports it. The explainer never executes the command.
        Never raises: bad input returns usage.
        """
        from ...codews.explain import (
            explain_command,
            format_explanation_reply,
            open_explanation_store,
        )
        from ...codews.man_corpus import fetch_tldr_page, open_man_corpus

        tail = (tail or "").strip()
        usage = "usage: /explain <shell command> | list | show <id> | " \
                "ingest | status"
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_explanation_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                rows = store.list()
                if not rows:
                    return "no explanations yet — explain a command " \
                           "with /explain <shell command>"
                lines = ["explanations:"]
                for row in rows[:10]:
                    lines.append(
                        f"  {row['explanation_id']} — {row['command']}")
                return "\n".join(lines)
            if verb == "show":
                if not rest:
                    return "usage: /explain show <id>"
                row = store.get(rest)
                if row is None:
                    return f"no explanation “{rest}”"
                return format_explanation_reply({**row, "ok": True})
            if verb == "status":
                corpus = open_man_corpus(home)
                try:
                    status = corpus.status()
                finally:
                    corpus.close()
                lines = [f"man corpus: {status['pages']} pages, "
                         f"{status['options']} options",
                         f"stored at {status['path']}",
                         f"explanations stored: {len(store.list())}"]
                if status["last_ingest"]:
                    lines.append(
                        f"last ingest: {status['last_ingest']}")
                return "\n".join(lines)
            if verb == "ingest":
                corpus = open_man_corpus(home)
                try:
                    stats = corpus.ingest()
                finally:
                    corpus.close()
                return (f"man corpus ingested: {stats['pages']} pages, "
                        f"{stats['options']} options "
                        f"({stats['skipped']} unchanged)")
            context = getattr(self, "context", None)
            tools = getattr(context, "tools", None) if context else None
            help_fetcher = None
            if tools is not None:
                def _help(cmd: str):
                    try:
                        outcome = tools.call(
                            "shell_run", command=f"{cmd} --help",
                            timeout=15.0)
                    except Exception:  # noqa: BLE001 — harvest is a miss
                        return None
                    if not getattr(outcome, "ok", False):
                        return None
                    value = getattr(outcome, "value", None)
                    if not isinstance(value, dict):
                        return None
                    if value.get("exit_code") != 0:
                        return None
                    text = (value.get("stdout") or "").strip()
                    return text or None

                help_fetcher = _help
            llm_fn = None
            if context is not None:
                from ...tools.paper_reader import _llm_from_context

                llm_fn = _llm_from_context(context)
            return format_explanation_reply(explain_command(
                home, tail, store=store, llm_fn=llm_fn,
                help_fetcher=help_fetcher,
                tldr_fetcher=fetch_tldr_page))
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"explain failed: {exc}"

    def _control_regex(self, tail: str) -> str:
        """Regex workbench (Regex101 core).

        Public. /regex <pattern> [:: <text>] tests the pattern and
        explains every token in plain English from a real pattern
        parser — never an LLM guess. /regex gen <description>
        generates a pattern, then PROVES it against positive and
        negative cases before answering (a pattern that fails its
        proof is never presented as working). /regex debug <pattern>
        :: <text> traces the match attempt with a step budget, so a
        catastrophic-backtracking pattern is caught and reported,
        never hung on. /regex sub <pattern> :: <replacement> ::
        <text> tests substitution. /regex code <id> <lang> generates
        code from a stored session and EXECUTES it to prove it
        reproduces the session's matches. /regex list shows past
        sessions; /regex show <id> re-renders one. A pattern a
        flavour does not support is refused by name; a flavour with
        no engine here is labelled not verifiable, never simulated.
        Never raises: bad input returns usage.
        """
        from ...tools.regex_lab import (
            format_regex_reply,
            generate_code,
            open_regex_store,
            run_regex_session,
            verify_code,
        )

        tail = (tail or "").strip()
        usage = ("usage: /regex <pattern> [:: <text>] | gen "
                 "<description> | debug <pattern> :: <text> | sub "
                 "<pattern> :: <replacement> :: <text> | code <id> "
                 "<lang> | list | show <id>")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_regex_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                sessions = store.list()
                if not sessions:
                    return "no regex sessions yet — test a pattern " \
                           "with /regex <pattern> :: <text>"
                lines = ["regex sessions:"]
                for row in sessions[:10]:
                    lines.append(
                        f"  {row['session_id']} — "
                        f"{row['pattern']} ({row['flavour']})")
                return "\n".join(lines)
            if verb == "show":
                if not rest:
                    return "usage: /regex show <id>"
                row = store.get(rest)
                if row is None:
                    return f"no regex session “{rest}”"
                return format_regex_reply({**row, "ok": True})
            context = getattr(self, "context", None)
            tools = getattr(context, "tools", None) if context else None
            run_fn = None
            if tools is not None:
                def _run(code: str, lang: str):
                    try:
                        outcome = tools.call(
                            "run_code", action="run", code=code,
                            lang=lang, timeout=30.0)
                    except Exception:  # noqa: BLE001 — unverified
                        return None
                    if not getattr(outcome, "ok", False):
                        return None
                    value = getattr(outcome, "value", None)
                    return value if isinstance(value, dict) else None

                run_fn = _run
            llm_fn = None
            if context is not None:
                from ...tools.paper_reader import _llm_from_context

                llm_fn = _llm_from_context(context)
            if verb == "gen":
                if not rest:
                    return "usage: /regex gen <description>"
                record = run_regex_session(
                    home, description=rest, llm_fn=llm_fn,
                    run_fn=run_fn, store=store)
                return format_regex_reply(record)
            if verb == "code":
                fields = rest.split()
                if len(fields) < 2:
                    return "usage: /regex code <id> <lang>"
                row = store.get(fields[0])
                if row is None:
                    return f"no regex session “{fields[0]}”"
                snippet = generate_code(
                    str(row.get("pattern", "")),
                    str(row.get("flavour", "python")), fields[1],
                    texts=[str(r.get("text", "")) for r in
                           (row.get("matches") or {}).get(
                               "results", [])])
                if not snippet.get("ok"):
                    return ("code generation failed: " +
                            str(snippet.get("error") or ""))
                record = dict(row)
                record["code"] = snippet
                expected = (row.get("matches") or {}).get(
                    "results", [])
                record["code_verification"] = verify_code(
                    snippet["code"], fields[1], expected,
                    run_fn=run_fn)
                return format_regex_reply(record)
            if verb in {"debug", "sub"}:
                segments = [s.strip() for s in rest.split("::")]
                if verb == "debug":
                    if len(segments) < 2 or not segments[0]:
                        return "usage: /regex debug <pattern> :: " \
                               "<text>"
                    record = run_regex_session(
                        home, pattern=segments[0],
                        trace_text=segments[1], run_fn=run_fn,
                        store=store)
                    return format_regex_reply(record)
                if len(segments) < 3 or not segments[0]:
                    return "usage: /regex sub <pattern> :: " \
                           "<replacement> :: <text>"
                record = run_regex_session(
                    home, pattern=segments[0],
                    substitution=segments[1],
                    texts=[segments[2]], run_fn=run_fn,
                    store=store)
                return format_regex_reply(record)
            segments = [s.strip() for s in tail.split("::")]
            pattern = segments[0]
            if not pattern:
                return usage
            texts = [ln for ln in segments[1].splitlines()
                     if ln.strip()] if len(segments) > 1 else []
            record = run_regex_session(
                home, pattern=pattern, texts=texts, run_fn=run_fn,
                store=store)
            return format_regex_reply(record)
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"regex failed: {exc}"

    def _control_fmt(self, tail: str) -> str:
        """Text-transformation family (CodeBeautify core).

        Public. /fmt beautifies, minifies, validates or converts
        text — JSON, JSONL, YAML, TOML, XML, CSV, SQL, CSS, HTML (and
        JS via prettier when installed). The input is parsed by a
        real parser first, so a malformed input returns a located
        error (line, column, machine-readable kind), never a guess.
        /fmt <text> auto-detects the format; /fmt <format> <text>
        names it; /fmt validate|minify <format> <text> checks or
        compacts; /fmt convert <from> <to> <text> converts through
        the shared data model; /fmt fix <text> repairs damaged JSON
        and proves the repair with a real parse. Never raises: bad
        input returns usage.
        """
        from ...codews.textfmt import format_transform_reply, transform

        tail = (tail or "").strip()
        usage = ("usage: /fmt <text> | <format> <text> | validate "
                 "<format> <text> | minify <format> <text> | convert "
                 "<from> <to> <text> | fix <text> | detect <text>")
        if not tail:
            return usage
        try:
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "detect":
                if not rest:
                    return usage
                return format_transform_reply(transform(
                    rest, operation="detect"))
            if verb == "fix":
                if not rest:
                    return usage
                return format_transform_reply(transform(
                    rest, operation="fix"))
            if verb in {"validate", "minify"}:
                fields = rest.split(None, 1)
                if len(fields) < 2:
                    return usage
                return format_transform_reply(transform(
                    fields[1], operation=verb,
                    format_in=fields[0].lower()))
            if verb == "convert":
                fields = rest.split(None, 2)
                if len(fields) < 3:
                    return usage
                return format_transform_reply(transform(
                    fields[2], operation="convert",
                    format_in=fields[0].lower(),
                    format_out=fields[1].lower()))
            fields = tail.split(None, 1)
            if len(fields) == 2 and fields[0].lower() in (
                    "json", "jsonl", "yaml", "toml", "xml", "csv", "sql",
                    "css", "html", "js"):
                return format_transform_reply(transform(
                    fields[1], operation="beautify",
                    format_in=fields[0].lower()))
            return format_transform_reply(transform(
                tail, operation="beautify", format_in="auto"))
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"fmt failed: {exc}"

    def _control_consensus(self, tail: str) -> str:
        """What the literature agrees on (Consensus meter).

        Public. /consensus <yes/no research question> asks the open
        scholarly corpus (OpenAlex, Semantic Scholar, arXiv), then
        classifies each paper's finding as yes / no / possibly /
        mixed, weighs the papers by quality (citations, recency,
        study design — never count alone) and reports the
        distribution as a meter with the dissent named by name and
        what would change it computed from the meter's own weights.
        A question that is not yes/no gets prose instead of a meter;
        a question with no papers is refused honestly — nothing is
        ever invented. /consensus list shows past reviews;
        /consensus show <id> re-renders one; /consensus meter <id>
        re-renders just the meter. "deep" (or --deep) asks for the
        deeper set (25 papers instead of 10). Never raises: bad
        input returns usage.
        """
        from ...research.consensus import (
            answer_consensus,
            format_consensus_reply,
            open_consensus_store,
        )

        tail = (tail or "").strip()
        usage = ("usage: /consensus <yes/no research question> "
                 "[deep] | list | show <id> | meter <id>")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_consensus_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                reviews = store.list()
                if not reviews:
                    return "no consensus reviews yet — ask me a research " \
                           "question with /consensus <does X …?>"
                lines = ["consensus reviews:"]
                for review in reviews[:10]:
                    meter = review.get("meter") or {}
                    lines.append(
                        f"  {review['id']} — {review['question']} "
                        f"({meter.get('strength', '')}, "
                        f"{meter.get('n', 0)} papers)")
                return "\n".join(lines)
            if verb in {"show", "meter"}:
                if not rest:
                    return f"usage: /consensus {verb} <id>"
                review = store.get(rest)
                if review is None:
                    return f"no consensus review “{rest}”"
                return format_consensus_reply({**review, "ok": True})
            depth = "quick"
            question = tail
            if verb == "deep":
                depth = "deep"
                question = rest
            elif tail.lower().endswith(" --deep"):
                depth = "deep"
                question = tail[:-len(" --deep")].strip()
            if not question:
                return usage
            return format_consensus_reply(
                answer_consensus(home, question, store=store, depth=depth))
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"consensus failed: {exc}"

    def _control_papergraph(self, tail: str, message: Any = None) -> str:
        """The paper graph around a paper (Connected Papers core).

        Public. /papergraph <title or DOI> resolves the seed paper
        against the open scholarly corpus (OpenAlex, Semantic Scholar
        fallback), counts the citation neighbourhood from reference
        sets (co-citation + bibliographic coupling, pure counting),
        and reports the clusters and the reading order — what to read
        next and why. The diagram is drawn as a PNG and sent in chat
        beside the text, fail-open: a failed render or send never
        takes the answer away. /papergraph list shows past graphs;
        /papergraph show <id> re-renders one; /papergraph graph <id>
        sends just the diagram. "deep" (or --deep) asks for the
        deeper pool (100 seed references, 50 citers, 60 nodes).
        Never raises: bad input returns usage.
        """
        from ...research.paper_graph import (
            build_graph,
            format_graph_reply,
            open_graph_store,
        )

        tail = (tail or "").strip()
        usage = ("usage: /papergraph <title or DOI> [deep] | list | "
                 "show <id> | graph <id>")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            from pathlib import Path

            home_path = (Path(home) if home else
                         Path("~/.nomorals").expanduser())
            store = open_graph_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                reviews = store.list()
                if not reviews:
                    return "no paper graphs yet — map one with " \
                           "/papergraph <title or DOI>"
                lines = ["paper graphs:"]
                for review in reviews[:10]:
                    lines.append(
                        f"  {review['id']} — {review['seed']} "
                        f"({review['nodes']} papers)")
                return "\n".join(lines)
            if verb in {"show", "graph"}:
                if not rest:
                    return f"usage: /papergraph {verb} <id>"
                review = store.get(rest)
                if review is None:
                    return f"no paper graph “{rest}”"
                if verb == "show":
                    return format_graph_reply({**review, "ok": True})
                return self._send_papergraph_diagram(review, message,
                                                     home_path)
            depth = "quick"
            query = tail
            if verb == "deep":
                depth = "deep"
                query = rest
            elif tail.lower().endswith(" --deep"):
                depth = "deep"
                query = tail[:-len(" --deep")].strip()
            if not query:
                return usage
            answer = build_graph(home, query, store=store, depth=depth)
            reply = format_graph_reply(answer)
            if answer.get("ok"):
                self._maybe_send_papergraph_diagram(answer, message,
                                                    home_path)
            return reply
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"papergraph failed: {exc}"

    def _send_papergraph_diagram(self, review: dict[str, Any],
                                 message: Any, home_path: Any) -> str:
        """Render a stored graph's diagram and send the PNG in chat.

        The diagram is the answer for /papergraph graph, so a
        failure is reported in words with the reason named — never
        silence, never a fake image.
        """
        from ...research.paper_graph import PaperGraph, graph_diagram
        from ...visuals.render import render_png

        graph = PaperGraph.from_dict(review.get("graph") or {})
        visuals_dir = home_path / "visuals"
        try:
            visuals_dir.mkdir(parents=True, exist_ok=True)
            path = render_png(
                graph_diagram(graph),
                visuals_dir / f"paper-graph-{review['id']}.png")
        except Exception as exc:  # noqa: BLE001 — honest refusal
            return f"graph failed: {exc}"
        if message is None:
            return (f"rendered to {path} — run /papergraph graph from "
                    "a chat to have the image sent there.")
        failed = self._scene_send_file(
            str(path), message,
            f"paper graph — “{graph.seed_title}”")
        if failed is not None:
            return failed
        return "paper graph sent."

    def _maybe_send_papergraph_diagram(self, answer: dict[str, Any],
                                       message: Any,
                                       home_path: Any) -> None:
        """After a fresh /papergraph map, send the diagram beside the text.

        Fail-open by design: the text is the answer and is already
        returned; anything missing here (no chat, a failed render)
        simply means no diagram — never an exception into the reply
        path.
        """
        try:
            if not answer.get("ok") or message is None:
                return
            self._send_papergraph_diagram(
                {"id": answer.get("review_id", ""),
                 "graph": answer.get("graph") or {}}, message, home_path)
        except Exception:  # noqa: BLE001 — diagram is additive only
            return

    def _control_scholar(self, tail: str) -> str:
        """The full Semantic Scholar record of a paper.

        Public. /scholar <title or DOI> fetches the full record —
        authors, venue, year, citation counts, the influential
        citations named with their intents, a one-line TLDR
        (Semantic Scholar's own, or one generated locally from the
        abstract when the API carries none), fields of study, and
        the open-access PDF when one exists. /scholar list shows
        stored records; /scholar show <id> re-renders one;
        /scholar watch <topic> | feeds | unwatch <topic> manage
        the topic feeds the daily tick checks for new papers. A
        paper with no record is refused honestly — nothing here is
        invented. Never raises: bad input returns usage.
        """
        from ...research.paper_records import (
            ensure_tldr,
            fetch_influential_citations,
            fetch_paper_record,
            format_record_reply,
            open_record_store,
        )

        tail = (tail or "").strip()
        usage = ("usage: /scholar <title or DOI> | list | show <id> | "
                 "watch <topic> | feeds | unwatch <topic>")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_record_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                rows = store.list()
                if not rows:
                    return ("no paper records yet — fetch one with "
                            "/scholar <title or DOI>")
                lines = ["paper records:"]
                for row in rows[:10]:
                    lines.append(f"  {row['paper_id']} — {row['title']} "
                                 f"({row.get('year') or '?'})")
                return "\n".join(lines)
            if verb == "feeds":
                feeds = store.feeds()
                if not feeds:
                    return ("no topic feeds watched — "
                            "/scholar watch <topic>")
                return "watched topics:\n" + "\n".join(
                    f"  {topic}" for topic in feeds)
            if verb == "watch":
                if not rest:
                    return "usage: /scholar watch <topic>"
                store.watch(rest)
                return (f"watching '{rest}' — new papers surface on "
                        "the daily tick")
            if verb == "unwatch":
                if not rest:
                    return "usage: /scholar unwatch <topic>"
                if not store.unwatch(rest):
                    return f"no feed watched for '{rest}'"
                return f"stopped watching '{rest}'"
            if verb == "show":
                if not rest:
                    return "usage: /scholar show <id>"
                record = store.get(rest)
                if record is None:
                    return f"no stored record '{rest}'"
                return format_record_reply({"ok": True, "record": record})
            record, note = fetch_paper_record(tail)
            if record is None:
                return note
            stored = store.get(record.paper_id) if record.paper_id else None
            if stored is not None and stored.tldr:
                record = stored
            else:
                record = ensure_tldr(record)
                store.save(record)
            if record.paper_id:
                record.influential = fetch_influential_citations(
                    record.paper_id)
                store.save(record)
            return format_record_reply({"ok": True, "record": record})
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"scholar failed: {exc}"

    def _control_readpaper(self, tail: str) -> str:
        """Read a paper and explain it in plain language.

        Public. /readpaper <DOI or title> reads the paper — the
        open-access PDF when one exists, the arXiv PDF, or the
        abstract when the full text is not openly available — and
        explains it: what it asks, what it did, what it found
        (every finding quoted verbatim), why it matters, and the
        hard parts (equations, tables) unpacked. /readpaper text
        <pasted paper> reads pasted text. /readpaper ask <id>
        <question> answers against the stored paper's own
        sentences; /readpaper explain <id> <equation 1|table 1>
        unpacks one hard part; /readpaper cite <id> [style]
        cites it (apa|mla|chicago|bibtex); /readpaper paraphrase
        <text> says it more simply without losing a number or a
        name. /readpaper list | show <id> recall stored readings.
        An unknown paper, or a question the paper does not cover,
        is refused honestly — nothing here is invented. Never
        raises: bad input returns usage.
        """
        from ...research.paper_reader import (
            PaperText,
            ask_paper,
            explain_span,
            format_citation,
            format_reading_reply,
            open_reading_store,
            paraphrase_span,
            read_paper,
        )

        tail = (tail or "").strip()
        usage = ("usage: /readpaper <DOI or title> | text <pasted paper> "
                 "| list | show <id> | ask <id> <question> | explain <id> "
                 "<equation 1|table 1> | cite <id> [apa|mla|chicago|bibtex] "
                 "| paraphrase <text>")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            store = open_reading_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                rows = store.list()
                if not rows:
                    return ("no paper readings yet — read one with "
                            "/readpaper <DOI or title>")
                lines = ["paper readings:"]
                for row in rows[:10]:
                    lines.append(f"  {row['reading_id']} — {row['title']} "
                                 f"({row.get('year') or '?'})")
                return "\n".join(lines)
            if verb == "show":
                if not rest:
                    return "usage: /readpaper show <id>"
                row = store.get(rest)
                if row is None:
                    return f"no stored reading '{rest}'"
                return format_reading_reply(
                    {"ok": True,
                     "reading_id": row["paper"]["paper_id"],
                     "paper": row["paper"],
                     "explanation": row["explanation"]})
            if verb == "ask":
                sub = rest.split(None, 1)
                if len(sub) < 2:
                    return "usage: /readpaper ask <id> <question>"
                row = store.get(sub[0])
                if row is None:
                    return (f"no stored reading '{sub[0]}' — read the "
                            "paper first")
                result = ask_paper(PaperText.from_dict(row["paper"]),
                                   sub[1])
                if not result.get("ok"):
                    return result.get("error") or \
                        "the paper does not cover that"
                store.log_question(sub[0], sub[1], result["answer"])
                lines = [result["answer"]]
                for quote in result.get("quotes") or []:
                    lines.append(f"  “{quote}”")
                return "\n".join(lines)
            if verb == "explain":
                sub = rest.split(None, 1)
                if len(sub) < 2:
                    return ("usage: /readpaper explain <id> "
                            "<equation 1|table 1>")
                row = store.get(sub[0])
                if row is None:
                    return (f"no stored reading '{sub[0]}' — read the "
                            "paper first")
                result = explain_span(PaperText.from_dict(row["paper"]),
                                      sub[1])
                if not result.get("ok"):
                    return result.get("error") or \
                        "no such span in this paper"
                part = result["explanation"]
                lines = [f"{part.span_ref} ({part.kind}): "
                         f"{part.plain_text}"]
                for quote in part.grounded_quotes:
                    lines.append(f"  “{quote}”")
                return "\n".join(lines)
            if verb == "cite":
                sub = rest.split(None, 1)
                if not sub:
                    return ("usage: /readpaper cite <id> "
                            "[apa|mla|chicago|bibtex]")
                row = store.get(sub[0])
                if row is None:
                    return (f"no stored reading '{sub[0]}' — read the "
                            "paper first")
                from ...research.paper_records import PaperRecord

                record = PaperRecord.from_dict(row["paper"])
                try:
                    return format_citation(
                        record,
                        sub[1].strip().lower() if len(sub) > 1 else "apa")
                except ValueError as exc:
                    return str(exc)
            if verb == "paraphrase":
                if not rest:
                    return "usage: /readpaper paraphrase <text>"
                result = paraphrase_span(rest)
                if not result.get("ok"):
                    return result.get("error") or "nothing to paraphrase"
                reply = result["paraphrase"]
                if result.get("method") == "quoted":
                    reply += ("\n(kept verbatim — a simpler version "
                              "would lose a number or a name)")
                return reply
            if verb == "text":
                if not rest:
                    return "usage: /readpaper text <pasted paper>"
                result = read_paper(rest, home=home)
                return result.get("reply") or result.get("error") or \
                    "paper read failed"
            result = read_paper(tail, home=home)
            return result.get("reply") or result.get("error") or \
                "paper read failed"
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"readpaper failed: {exc}"

    def _control_soundscape(self, tail: str, message: Any = None) -> str:
        """Soundscape machine: compose, tune and play ambient sound.

        Public. /soundscape <rain|forest|cafe|brown noise|sleep|…>
        renders the soundscape and sends the WAV in chat. Natural
        words work — “rain sounds, more thunder, less birds” — and a
        sleep timer (“fade out in 40 minutes”) ends in silence, not
        a cut. /soundscape list shows the generators, presets and
        saved mixes; /soundscape save <name> keeps the last render
        as a mix one word brings back; /soundscape mixes lists them;
        /soundscape tune <stem|band> <+dB|-dB> re-renders the last
        one adjusted; /soundscape tone <freq> plays a calibration
        tone; /soundscape calibrate <band Hz> <gain dB> sets the
        hearing profile, applied to every render after. Everything
        is synthesised in code — no recording is ever faked — and an
        unknown name is refused with the real list. Never raises:
        bad input returns usage.
        """
        from ...audio.soundscapes.engine import BAND_FREQS, render_tone
        from ...audio.soundscapes.generators import (
            GENERATORS,
            PRESETS,
            parse_soundscape_query,
        )
        from ...audio.soundscapes.render import render_soundscape
        from ...audio.soundscapes.store import open_store

        tail = (tail or "").strip()
        usage = ("usage: /soundscape <rain|forest|cafe|brown noise|sleep|…> "
                 "[more <stem>|less <stem>] [fade out in <40 minutes>] | "
                 "list | mixes | save <name> | tune <stem|band> <+dB|-dB> | "
                 "tone <freq> | calibrate <band Hz> <gain dB>")
        if not tail:
            return usage
        try:
            home = None
            try:
                home = str(self.settings.home_path)
            except Exception:  # noqa: BLE001 — default home then
                home = None
            home = home or "~/.nomorals"
            store = open_store(home)
            parts = tail.split(None, 1)
            verb = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if verb in {"help", "--help"}:
                return usage
            if verb == "list":
                lines = ["soundscapes — generators:"]
                for spec in GENERATORS.values():
                    lines.append(f"• {spec.name} ({spec.id}) — "
                                 f"{spec.description}")
                lines.append("presets: " + ", ".join(sorted(PRESETS)))
                mixes = store.list_mixes()
                if mixes:
                    lines.append("your mixes: " + ", ".join(mixes))
                return "\n".join(lines)
            if verb == "mixes":
                mixes = store.list_mixes()
                if not mixes:
                    return "no saved mixes yet — render one, then /soundscape save <name>."
                lines = ["your mixes:"]
                for name in mixes:
                    mix = store.load_mix(name)
                    if mix:
                        lines.append(f"• {mix['name']} — {mix['generator']}")
                return "\n".join(lines)
            if verb == "save":
                if not rest:
                    return "usage: /soundscape save <name>"
                last = store.last()
                if not last or not last.get("generator"):
                    return ("nothing rendered yet — ask for a soundscape "
                            "first, then /soundscape save <name>.")
                from ...audio.soundscapes.generators import resolve_generator

                spec = resolve_generator(str(last["generator"]))
                sliders = last.get("sliders") or {}
                values = [float(sliders.get(label, 0.0))
                          for label in spec.sliders]
                store.save_mix(rest, spec.id, tuple(values),
                               animate=bool(last.get("animate", False)),
                               duration_s=float(
                                   last.get("duration_s") or 60.0))
                return f"saved mix “{rest}” — {spec.name}."
            if verb == "tone":
                try:
                    freq = float(rest)
                except (TypeError, ValueError):
                    return "usage: /soundscape tone <freq Hz>"
                path = render_tone(freq, duration_s=2.0)
                text = f"calibration tone — {freq:g} Hz."
                if message is not None:
                    failed = self._scene_send_file(path, message,
                                                   "calibration tone")
                    if failed is not None:
                        return text + f"\n{failed}"
                    return text + "\nsent."
                return text + f"\nfile: {path}"
            if verb == "calibrate":
                fields = rest.split()
                if len(fields) != 2:
                    bands = ", ".join(f"{f:g}" for f in BAND_FREQS)
                    return ("usage: /soundscape calibrate <band Hz> <gain dB>\n"
                            f"bands: {bands}")
                try:
                    band_hz = float(fields[0])
                    gain = float(fields[1])
                except (TypeError, ValueError):
                    return ("usage: /soundscape calibrate <band Hz> <gain dB>")
                if band_hz not in BAND_FREQS:
                    bands = ", ".join(f"{f:g}" for f in BAND_FREQS)
                    return (f"{band_hz:g} Hz is not a calibration band — "
                            f"try: {bands}")
                store.set_hearing_band(band_hz, gain)
                return (f"hearing: {band_hz:g} Hz {gain:+.1f} dB — applied "
                        f"to every render from now on.")
            if verb == "tune":
                # The dB is the last word; the label is everything
                # before it, so a band ("1000 Hz") parses whole.
                label, sep, db_text = rest.rpartition(" ")
                if not sep or not label:
                    return ("usage: /soundscape tune <stem|band> <+dB|-dB>")
                last = store.last()
                if not last or not last.get("generator"):
                    return ("nothing rendered yet — ask for a soundscape "
                            "first, then tune it.")
                try:
                    delta = float(db_text)
                except (TypeError, ValueError):
                    return ("usage: /soundscape tune <stem|band> <+dB|-dB>")
                from ...audio.soundscapes.generators import resolve_generator

                spec = resolve_generator(str(last["generator"]))
                if label not in spec.sliders:
                    return (f"“{label}” is not a slider of "
                            f"{spec.name} — try: "
                            + ", ".join(spec.sliders))
                ans = render_soundscape(
                    home, str(last["generator"]),
                    duration_s=float(last.get("duration_s") or 60.0),
                    animate=bool(last.get("animate", False)),
                    sliders=dict(last.get("sliders") or {}),
                    adjust={label: delta})
                return self._soundscape_send(ans, message)
            parsed = parse_soundscape_query(tail)
            if store.load_mix(tail) is not None:
                # The owner's own mix name wins over any preset word
                # inside it ("deep focus" is his mix, not the "focus"
                # preset) — render_soundscape loads the mix itself.
                ans = render_soundscape(home, tail)
            else:
                target = parsed.get("generator") or tail
                ans = render_soundscape(
                    home, target,
                    timer_min=parsed.get("timer_min"),
                    animate=parsed.get("animate"),
                    adjust=dict(parsed.get("adjust") or {}))
            return self._soundscape_send(ans, message)
        except Exception as exc:  # noqa: BLE001 — never raise in chat
            return f"soundscape failed: {exc}"

    def _soundscape_send(self, ans: dict[str, Any],
                         message: Any = None) -> str:
        """Send a rendered soundscape: reply text + the WAV file."""
        if not ans.get("ok"):
            return ans.get("error") or "soundscape failed."
        text = ans.get("reply") or f"{ans.get('name')} rendered."
        path = ans.get("path")
        if not path:
            return text
        if message is not None:
            failed = self._scene_send_file(str(path), message,
                                           "soundscape")
            if failed is not None:
                return text + f"\n{failed}"
            return text + "\nsent."
        return text + f"\nfile: {path}"

    def _control_video_explain(self, tail: str) -> str:
        """Explainer presets: /video explain <preset> <content>.

        ``/video explain code-trace x = 1 | x = x + 1 --values "x = 1 |
        x = 2"`` renders the named Manim-class preset on the code
        backend — deterministic seek(t) frames, narrated by default.
        Never raises: bad input returns usage, render failures name
        the problem. An unknown preset is refused with the list of
        real ones — never a silent default.
        """
        import re

        from ...media.videogen import VideogenError, list_presets, render_preset

        def _usage() -> str:
            lines = ["usage: /video explain <preset> <content> "
                     "[--values \"a|b\"] [--title \"T\"] [--seconds N] "
                     "[--silent]"]
            for entry in list_presets():
                lines.append(f"  {entry['name']}: {entry['description']}")
            return "\n".join(lines)

        tail = (tail or "").strip()
        seconds = 5.0
        narrate = True
        values = ""
        title = ""
        if "--silent" in tail:
            narrate = False
            tail = tail.replace("--silent", " ").strip()
        sm = re.search(r"--seconds\s+([\d.]+)", tail)
        if sm:
            try:
                seconds = float(sm.group(1))
            except ValueError:
                seconds = 5.0
            tail = (tail[:sm.start()] + tail[sm.end():]).strip()
        vm = re.search(r'--values\s+"([^"]*)"', tail)
        if vm:
            values = vm.group(1)
            tail = (tail[:vm.start()] + tail[vm.end():]).strip()
        tm = re.search(r'--title\s+"([^"]*)"', tail)
        if tm:
            title = tm.group(1)
            tail = (tail[:tm.start()] + tail[tm.end():]).strip()
        if not tail:
            return _usage()
        name, _, content = tail.partition(" ")
        name = name.strip().lower()
        content = content.strip()
        known = {entry["name"] for entry in list_presets()}
        if name not in known:
            return (f"unknown explainer preset {name!r} — known "
                    f"presets: {', '.join(sorted(known))}")
        if not content:
            return _usage()
        params: dict[str, Any] = {}
        if name == "equation-walkthrough":
            params["lines"] = content
        elif name == "diagram-build":
            params["nodes"] = content
        elif name == "code-trace":
            params["code"] = content
            if values:
                params["values"] = values
        else:  # chart-build
            params["series"] = content
        if title:
            params["title"] = title
        try:
            res = render_preset(name, duration_s=seconds,
                                narrate=narrate, **params)
        except VideogenError as exc:
            return f"explain: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return f"explain error: {exc}"
        return res.message()

    def _control_code_format(self, kind: str, tail: str) -> str:
        """Code-video format commands: /map-route, /headline, /teardown.

        Each builds its scene spec from the tail and renders it on the
        videogen code backend — deterministic seek(t) frames, no model.
        Never raises: bad input returns usage, render failures name the
        problem.
        """
        import re

        from ...media.videogen import VideogenError, render_format

        tail = (tail or "").strip()
        seconds = 5.0
        narrate = True
        captions = True
        if "--silent" in tail:
            narrate = False
            tail = tail.replace("--silent", " ").strip()
        if "--no-captions" in tail:
            captions = False
            tail = tail.replace("--no-captions", " ").strip()
        sm = re.search(r"--seconds\s+([\d.]+)", tail)
        if sm:
            try:
                seconds = float(sm.group(1))
            except ValueError:
                seconds = 5.0
            tail = (tail[:sm.start()] + tail[sm.end():]).strip()
        try:
            if kind == "map-route":
                if not tail:
                    return ("usage: /map-route <A> to <B> [via <C>…] | "
                            "/map-route <A>, <B>, <C> [--seconds N] "
                            "[--silent] [--no-captions]")
                res = render_format("map-route", route=tail,
                                    duration_s=seconds, narrate=narrate,
                                    captions=captions)
            elif kind == "headline":
                lines = None
                lm = re.search(r'--lines\s+"([^"]*)"', tail)
                if lm:
                    lines = lm.group(1)
                    tail = (tail[:lm.start()] + tail[lm.end():]).strip()
                if not tail:
                    return ('usage: /headline <text> [--lines "a|b"] '
                            "[--seconds N] [--silent] [--no-captions]")
                res = render_format("headline", headline=tail,
                                    lines=lines, duration_s=seconds,
                                    narrate=narrate, captions=captions)
            else:  # teardown
                if not tail:
                    return ("usage: /teardown <thing> : <part>, <part>… "
                            "[--seconds N] [--silent] [--no-captions]")
                thing, sep, parts = tail.partition(":")
                if not sep:
                    thing, sep, parts = tail.partition("—")
                if not sep or not parts.strip():
                    return ("usage: /teardown <thing> : <part>, <part>… "
                            "[--seconds N] [--silent] [--no-captions]")
                res = render_format("teardown", thing=thing.strip(),
                                    parts=parts, duration_s=seconds,
                                    narrate=narrate, captions=captions)
        except VideogenError as exc:
            return f"{kind}: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return f"{kind} error: {exc}"
        return res.message()

    def _control_videocall(self, tail: str, chat_key: str = "",
                           message: Any = None) -> str:
        """LiveKit video from chat: /videocall.

        A thin dispatch to the same ``video_live_*`` tools the brain
        uses — no session construction duplicated here, so the chat
        surface and the brain surface can never drift apart. Never
        raises: bad input returns usage, tool failures name the
        problem in words.

        ``/videocall <room>`` starts publishing Devon's video in the
        room (title card by default; ``--source image|scene`` with
        ``--image PATH`` picks the picture; ``--seconds N`` bounds
        the session). ``status`` shows readiness and the running
        session; ``stop`` ends it with the report; ``snapshot``
        sends the latest frame she has seen from the room.
        """
        import re

        tail = (tail or "").strip()
        words = tail.split()
        verb = (words[0].lower() if words else "status")
        tools = getattr(self.context, "tools", None)
        if tools is None:
            return "videocall needs a live session — no tool registry here."

        def _call(name: str, **kwargs: Any):
            outcome = tools.call(name, **kwargs)
            if not outcome.ok:
                return None, (f"{getattr(outcome.error, 'message', outcome.error)}")
            return outcome.value or {}, None

        if verb == "status":
            value, err = _call("video_live_status")
            if err is not None:
                return f"videocall status failed: {err}"
            status = value.get("status") or {}
            lines = ["📹 live video:"]
            lines.append(f"  SDK: {'installed (' + str(status.get('sdk_version')) + ')' if status.get('sdk_installed') else 'NOT installed — nm voice live provision'}")
            if status.get("configured"):
                lines.append("  server: configured")
            else:
                lines.append("  server: NOT configured — missing: "
                             + "; ".join(status.get("missing") or []))
            session = status.get("session")
            if session:
                lines.append(f"  running: room {session.get('room')} — "
                             f"{session.get('state')} "
                             f"({session.get('source')}, "
                             f"{session.get('frames_published')} frames "
                             f"published, "
                             f"{sum((session.get('frames_received') or {}).values())} received)")
            last = status.get("last_report")
            if not session and last:
                lines.append(f"  last: room {last.get('room')} — "
                             f"{last.get('frames_published')} frames, "
                             f"ended {last.get('end_reason')}")
            return "\n".join(lines)

        if verb == "stop":
            value, err = _call("video_live_end")
            if err is not None:
                return f"videocall stop failed: {err}"
            report = value.get("report") or {}
            return (f"📹 video ended in room {report.get('room')} — "
                    f"{report.get('frames_published')} frames published, "
                    f"{sum((report.get('frames_received') or {}).values())} received, "
                    f"ended {report.get('end_reason')}")

        if verb == "snapshot":
            identity = words[1] if len(words) > 1 else ""
            value, err = _call("video_live_snapshot",
                               identity=identity)
            if err is not None:
                return f"videocall snapshot failed: {err}"
            snap = value.get("snapshot") or {}
            path = str(snap.get("path") or "")
            if not path or message is None:
                return (f"snapshot from {snap.get('identity')}: {path} "
                        f"({snap.get('frames_received')} frames received, "
                        f"{snap.get('age_s')}s old)")
            chat = message.chat
            outcome = tools.call(
                "file_send",
                platform=str(getattr(chat, "platform", "") or ""),
                chat_id=str(getattr(chat, "chat_id", "") or ""),
                path=path,
                caption=(f"live video snapshot — {snap.get('identity')} "
                         f"({snap.get('frames_received')} frames, "
                         f"{snap.get('age_s')}s old)"))
            if not outcome.ok:
                return (f"snapshot saved to {path}, but sending it "
                        f"failed: "
                        f"{getattr(outcome.error, 'message', outcome.error)}")
            return "snapshot sent."

        if verb == "start":
            words = words[1:]
        room = words[0] if words else ""
        if not room:
            return ("usage: /videocall <room> [--source title|image|scene] "
                    "[--image PATH] [--seconds N] | /videocall status | "
                    "/videocall stop | /videocall snapshot [identity]")
        rest = " ".join(words[1:])
        source = "title"
        sm = re.search(r"--source\s+(\S+)", rest)
        if sm:
            source = sm.group(1)
        image = ""
        im = re.search(r"--image\s+(\S+)", rest)
        if im:
            image = im.group(1)
        max_duration = 0.0
        dm = re.search(r"--seconds\s+([\d.]+)", rest)
        if dm:
            try:
                max_duration = float(dm.group(1))
            except ValueError:
                max_duration = 0.0
        kwargs: dict[str, Any] = {"room": room, "source": source,
                                   "max_duration": max_duration}
        if image:
            kwargs["image"] = image
        value, err = _call("video_live_start", **kwargs)
        if err is not None:
            return f"videocall failed: {err}"
        return (f"📹 live video in room {room} "
                f"({value.get('source')}) — session "
                f"{value.get('session_id')}. /videocall snapshot to see "
                f"the room, /videocall stop to end it.")

    def _video_studio_gen(self, tail: str) -> str:
        import re

        from ...media.videogen import generate
        # /video gen "<prompt>" [--backend X] [--seconds N] [--seed N]
        #   [--model NAME] [--lora NAME[:W]] [--lora NAME2 ...]
        m = re.match(r'"([^"]+)"\s*(.*)$', tail) or re.match(r"'([^']+)'\s*(.*)$", tail)
        if m:
            prompt, flags = m.group(1), m.group(2)
        else:
            parts = tail.split("--", 1)
            prompt, flags = parts[0].strip(), ("--" + parts[1] if len(parts) > 1 else "")
        if not prompt:
            return ('usage: /video gen "<prompt>" [--backend auto|ltx|wan|motion|code] '
                    '[--seconds N] [--seed N] [--model NAME] '
                    '[--lora NAME[:WEIGHT]] [--lora NAME2 ...]')
        backend = "auto"
        seconds = 5.0
        seed = 0
        bm = re.search(r"--backend\s+(\S+)", flags)
        if bm:
            backend = bm.group(1)
        sm = re.search(r"--seconds\s+([\d.]+)", flags)
        if sm:
            seconds = float(sm.group(1))
        nm = re.search(r"--seed\s+(\d+)", flags)
        if nm:
            seed = int(nm.group(1))
        mm = re.search(r"--model\s+(\S+)", flags)
        model = mm.group(1) if mm else None
        loras = re.findall(r"--lora\s+(\S+)", flags)
        res = generate(prompt, backend=backend, duration_s=seconds,
                       seed=seed, model=model, lora=loras or None)
        return res.message()

    def _video_studio_qc(self, tail: str) -> str:
        """`/video qc <path>` | `/video qc "<prompt>" [--seconds N]`.

        An existing file gets the single-render QC; anything else is
        treated as a prompt and renders the variants matrix. Never
        raises: bad input returns usage, failures name the problem.
        """
        import re
        from pathlib import Path

        from ...media.videogen import VideogenError, matrix_message, qc_existing, render_matrix
        seconds = 5.0
        sm = re.search(r"--seconds\s+([\d.]+)", tail)
        if sm:
            try:
                seconds = float(sm.group(1))
            except ValueError:
                seconds = 5.0
            tail = (tail[:sm.start()] + tail[sm.end():]).strip()
        m = re.match(r'"([^"]+)"\s*$', tail) or re.match(r"'([^']+)'\s*$",
                                                        tail)
        target = m.group(1) if m else tail.strip()
        if not target:
            return ('usage: /video qc <path-to-render> | /video qc '
                    '"<prompt>" [--seconds N]')
        try:
            if Path(target).expanduser().is_file():
                out = qc_existing(target)
                lines = [f"🎬 QC — {target}",
                         f"  verdict: {out['verdict'].upper()}",
                         f"  gate ship: {out['ship']} (edit-engine gate "
                         f"needs a document — see the visual verdict)"]
                for reason in out["reasons"]:
                    lines.append(f"  — {reason}")
                lines.append(f"  contact sheet: {out['contact_sheet']}")
                lines.append(f"  report: {out['report_path']}")
                return "\n".join(lines)
            report = render_matrix(prompt=target, duration_s=seconds)
            return matrix_message(report)
        except VideogenError as exc:
            return f"video qc: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return f"video qc error: {exc}"

    def _video_studio_motion(self, tail: str) -> str:
        import shlex

        from ...media.motion_studio import studio
        try:
            parts = shlex.split(tail)
        except ValueError:
            parts = tail.split()
        if not parts:
            return ("usage: /video motion lyric|visual|slideshow|trailer …\n"
                    "/video motion lyric <audio> \"<lyrics>\" [--preset NAME]\n"
                    "/video motion visual <audio> [--preset phonk]\n"
                    "/video motion slideshow <img1,img2> [audio]\n"
                    "/video motion trailer <clip1,clip2> [--title T]")
        kind = parts[0].lower()
        rest = parts[1:]
        preset = "neon_pop"
        skip_next = False
        positional = []
        title = ""
        for i, p in enumerate(rest):
            if skip_next:
                skip_next = False
                continue
            if p == "--preset" and i + 1 < len(rest):
                preset = rest[i + 1]
                skip_next = True
                continue
            if p == "--title" and i + 1 < len(rest):
                title = rest[i + 1]
                skip_next = True
                continue
            if p.startswith("--"):
                continue
            positional.append(p)
        if kind == "lyric":
            if len(positional) < 2:
                return 'usage: /video motion lyric <audio> "<lyrics>" [--preset NAME]'
            path = studio.make_lyric_video(positional[0], positional[1],
                                           preset=preset)
            return f"🎤 lyric video → {path}"
        if kind == "visual":
            if not positional:
                return "usage: /video motion visual <audio> [--preset phonk]"
            vpreset = preset if preset != "neon_pop" else "phonk"
            path = studio.make_music_visualizer(positional[0], preset=vpreset)
            return f"🎛️ visualizer → {path}"
        if kind == "slideshow":
            if not positional:
                return "usage: /video motion slideshow <img1,img2> [audio]"
            images = [i for i in positional[0].split(",") if i]
            audio = positional[1] if len(positional) > 1 else None
            path = studio.make_slideshow(images, audio=audio)
            return f"🖼️ slideshow → {path}"
        if kind == "trailer":
            if not positional:
                return "usage: /video motion trailer <clip1,clip2> [--title T]"
            clips = [c for c in positional[0].split(",") if c]
            path = studio.make_trailer(clips, title=title)
            return f"🎬 trailer → {path}"
        return f"unknown /video motion kind {kind!r}"

    def _control_director(self, tail: str) -> str:
        """/director — Devon directs: coverage → plan → shoot → QA.

        /director cover <script-file>                 reader's coverage
        /director plan <script-file> [--style PACK] [--refine]
        /director shoot <production-id> [--backend X] [--model M]
                        [--shots 1.1,2.3] [--no-qa]
        /director regen <production-id> <shot-id>
        /director show [production-id] | /director styles
        Never raises."""
        import re
        import shlex
        from pathlib import Path

        from ...media.director import STYLE_PACKS, Director, DirectorError

        usage = (
            "🎬 /director — coverage → breakdown + design → shoot → QA\n"
            "  /director cover <script-file>\n"
            "  /director plan <script-file> [--style neutral|naturalist|"
            "noir|neon|golden] [--refine]\n"
            "  /director shoot <production-id> [--backend auto|ltx|wan|"
            "motion] [--model M] [--shots 1.1,2.3] [--no-qa]\n"
            "  /director regen <production-id> <shot-id>\n"
            "  /director show [production-id] | /director styles")
        try:
            parts = shlex.split(tail or "")
        except ValueError:
            parts = (tail or "").split()
        if not parts:
            return usage
        action = parts[0].lower()
        rest = parts[1:]
        director = Director(context=self.context)

        def read_script(path_arg: str) -> str:
            p = Path(path_arg)
            if not p.is_file():
                ws = getattr(self.context, "workspace", None)
                if ws and (Path(ws) / path_arg).is_file():
                    p = Path(ws) / path_arg
                else:
                    raise DirectorError(f"script file not found: {path_arg}")
            return p.read_text(encoding="utf-8", errors="replace")

        def flag(name: str, default: str = "") -> str:
            m = re.search(rf"--{name}\s+(\S+)", " ".join(rest))
            return m.group(1) if m else default

        try:
            if action == "styles":
                lines = ["🎞️ director style packs (data over the "
                         "design primitives — the engine bakes no look):"]
                for name, pack in sorted(STYLE_PACKS.items()):
                    lines.append(f"  {name:12} {pack['label']}")
                return "\n".join(lines)
            if action == "cover":
                if not rest:
                    return "usage: /director cover <script-file>"
                text = read_script(rest[0])
                cov = director.cover(text)
                path = director.save_coverage(cov)
                return cov.report() + f"\n\nsaved: {path}"
            if action == "plan":
                if not rest:
                    return ("usage: /director plan <script-file> "
                            "[--style PACK] [--refine]")
                text = read_script(rest[0])
                prod = director.plan(
                    text, style_pack=flag("style", "neutral"),
                    refine="--refine" in rest)
                lines = [f"🎬 production {prod.id} — {prod.title}",
                         f"style pack: {prod.style_pack} · "
                         f"{len(prod.screenplay.scenes)} scenes · "
                         f"{len(prod.shots)} shots designed"]
                for s in prod.shots:
                    frag = ""
                    if s.design:
                        frag = (f" [{s.design.camera.shot_size}, "
                                f"{s.design.lighting.key_source}]")
                    lines.append(f"  {s.id} {s.kind}: "
                                 f"{s.description[:70]}{frag}")
                lines.append(f"plan: {prod.dir}/report.md")
                lines.append(f"shoot it: /director shoot {prod.id}")
                return "\n".join(lines)
            if action == "shoot":
                if not rest:
                    return ("usage: /director shoot <production-id> "
                            "[--shots 1.1,2.3] [--no-qa]")
                shot_ids = [s for s in flag("shots").split(",") if s]
                prod = director.shoot(
                    rest[0], shot_ids=shot_ids or None,
                    backend=flag("backend", "auto"),
                    model=flag("model") or None,
                    qa="--no-qa" not in rest)
                return director.summary(prod.id)
            if action == "regen":
                if len(rest) < 2:
                    return ("usage: /director regen <production-id> "
                            "<shot-id>")
                prod = director.regen(rest[0], rest[1],
                                      backend=flag("backend", "auto"),
                                      model=flag("model") or None)
                shot = prod.shot(rest[1])
                return (f"🎬 shot {shot.id} regenerated → {shot.status}\n"
                        f"clip: {shot.clip_path}\n"
                        f"QA: {shot.qa.get('notes', '')}")
            if action == "show":
                return director.summary(rest[0] if rest else "")
            return f"unknown /director action {action!r}\n{usage}"
        except DirectorError as exc:
            return f"director: {exc}"
        except ValueError as exc:  # unknown style pack etc.
            return f"director: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return f"director error: {exc}"

    def _control_caption(self, tail: str, chat_key: str = "",
                         message: Any = None) -> str:
        """`/caption [style]` — burn AI subtitles into a video.

        Attach a video to the command message, or pass a file path.
        Styles: hormozi (default), mrbeast, karaoke, minimal.
        """
        from ...media_edit.captions import caption_styles, caption_video
        from ...media_edit.videos import MediaEditError

        parts = (tail or "").strip().split(None, 1)
        style = "hormozi"
        path_arg = ""
        if parts:
            # First word might be a style or a path.
            if parts[0].lower() in caption_styles():
                style = parts[0].lower()
                path_arg = parts[1] if len(parts) > 1 else ""
            else:
                path_arg = tail.strip()

        # Find the video: attached media first, then path argument.
        video_path = ""
        if message is not None:
            for media in getattr(message, "media", None) or []:
                if getattr(media, "kind", "") == "video":
                    video_path = getattr(media, "path", "")
                    break
        if not video_path and path_arg:
            video_path = path_arg
        if not video_path:
            return ("usage: /caption [style] — attach a video to the command "
                    f"or pass a path. styles: {', '.join(caption_styles())}")

        try:
            res = caption_video(video_path, style=style)
        except MediaEditError as exc:
            return f"caption failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"caption failed: {exc}"

        out_path = res.get("output", "")
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None and out_path:
            try:
                self.gateway.send_file(
                    chat.platform, f"{chat.platform}:{chat.chat_id}",
                    out_path,
                    caption=f"🎬 captioned ({style} style)", kind="video")
                # The captioned video IS the reply — no "video sent"
                # receipt after it (owner no-meta-narration rule,
                # 2026-11-01). Failures still return their text below.
                return ""
            except Exception as e:
                record_failure("runtime_media.RuntimeMediaMixin._control_caption", e)
        return f"🎬 captioned ({style}): {out_path or '(render failed)'}"

    _VISION_SUBCOMMANDS = (
        "describe", "read-text", "read", "locate", "compare", "analyze",
        "faces", "qr", "layout", "hash", "exif", "colors", "info",
        "capabilities", "document",
    )

    def _control_vision(self, tail: str, chat_key: str = "",
                        message: Any = None) -> str:
        """`/vision [subcommand] [args]` — Devon's eyes in chat.

        Subcommands (image from attachment, path, or URL):
          describe <img> [question]  — vision-model description (default)
          read-text <img>            — transcription, native OCR first,
                                       plus the picture as an editable
                                       document file
          document <img...> [format] — one step: picture(s) → editable
                                       document file (Word, HTML, text,
                                       PDF, spreadsheet — the shape is
                                       chosen at runtime from the content
                                       and your words; name a format to
                                       win, e.g. `document shot.png word`)
          locate "<target>" <img>    — find a UI element / object
          compare <imgA> <imgB>      — native pixel diff + model reading
          analyze <img>              — native deep report (no model)
          faces|qr|layout|hash|exif|colors <img> — one native analysis
          info                       — what Devon can do with images here

        Bare `/vision <img> [question]` still means describe.
        """
        import shlex

        def _looks_like_target(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return False
            if s.startswith(("http://", "https://", "/", "./", "~/")):
                return True
            if len(s) > 3 and "." in s:
                low = s.lower()
                if any(low.endswith(ext) for ext in
                       (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
                        ".tiff", ".heic")):
                    return True
            return False

        def _send(text: str) -> str:
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None:
                return self._send_long_checked(chat.platform, chat,
                                               text[:6000])
            return text[:6000]

        def _fail(text: str) -> str:
            return _send(text)

        tail = (tail or "").strip()
        try:
            tokens = shlex.split(tail)
        except ValueError:
            tokens = tail.split()

        sub = "describe"
        rest = tokens
        if tokens and tokens[0].lower() in self._VISION_SUBCOMMANDS:
            sub = tokens[0].lower()
            rest = tokens[1:]
        if sub == "read":
            sub = "read-text"
        if sub == "capabilities":
            sub = "info"

        # attached image wins over any path argument
        attached = ""
        attached_all: list[str] = []
        if message is not None:
            for media in getattr(message, "media", None) or []:
                if getattr(media, "kind", "") == "image":
                    path = getattr(media, "path", "") or ""
                    if path:
                        attached_all.append(path)
                    if not attached:
                        attached = path

        def _pick_targets(count: int) -> list[str] | None:
            found = [t for t in rest if _looks_like_target(t)]
            if attached:
                found = [attached] + [t for t in found if t != attached]
            if len(found) < count:
                return None
            return found[:count]

        def _question_minus(targets: list[str]) -> str:
            words = [t for t in rest if t not in targets]
            return " ".join(words).strip()

        def _call(name: str, **kw: Any) -> Any:
            outcome = self.context.tools.call(name, **kw)
            if not outcome.ok:
                raise RuntimeError(
                    getattr(outcome.error, "message", None) or str(outcome.error))
            return outcome.value or {}

        def _all_targets() -> list[str]:
            """Every image target: all attached images first, then any
            path/URL tokens, in order — several pictures go into ONE
            document."""
            found = [t for t in rest if _looks_like_target(t)]
            if attached_all:
                found = attached_all + [t for t in found
                                        if t not in attached_all]
            return found

        def _deliver_document(targets: list[str], format: str | None,
                              instruction: str) -> dict[str, Any]:
            """Pictures → editable document file, sent with the reply.

            Extract + correct each picture, let the model choose and
            shape the document at runtime, invoke the chosen builder
            through the registry, and send the staged file. A failure
            raises (the caller turns it into an honest failure text) —
            no fake document, no empty file.
            """
            from ...tools.vision import resolve_image
            from ...tools.vision_document import picture_to_document

            images: list[tuple[bytes, str]] = []
            for target in targets:
                if target.startswith(("http://", "https://")):
                    data, source = resolve_image(self.context, url=target)
                else:
                    data, source = resolve_image(self.context, path=target)
                images.append((data, source))
            result = picture_to_document(
                self.context, self.context.tools, images=images,
                instruction=instruction, format=format)
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None:
                try:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        result["path"],
                        caption=(f"📄 {result['filename']} "
                                 f"({result['format']})"),
                        kind="document")
                except Exception as exc:  # noqa: BLE001
                    record_failure(
                        "runtime_media.RuntimeMediaMixin._control_vision",
                        exc)
                    result = dict(result, send_error=str(exc))
            return result

        try:
            if sub == "info":
                value = _call("vision_capabilities")
                return _send(self._render_capabilities(value))

            if sub in ("analyze", "faces", "qr", "hash", "exif",
                       "colors"):
                targets = _pick_targets(1)
                if not targets:
                    return _fail(
                        f"usage: /vision {sub} <image> — attach an image or "
                        "pass a path/URL")
                value = _call("vision_analyze", path=targets[0])
                return _send(self._render_analyze(value, sub))

            if sub == "layout":
                targets = _pick_targets(1)
                if not targets:
                    return _fail("usage: /vision layout <image> — attach "
                                 "an image or pass a path/URL")
                value = _call("vision_layout", path=targets[0])
                return _send(self._render_layout(value))

            if sub == "compare":
                targets = _pick_targets(2)
                if not targets:
                    return _fail("usage: /vision compare <imageA> <imageB> — "
                                 "attach one image and pass the other, or pass "
                                 "two paths/URLs")
                value = _call("vision_compare", path_a=targets[0],
                              path_b=targets[1])
                return _send(self._render_compare(value))

            if sub == "document":
                from ...tools.vision_document import FORMATS

                targets = _all_targets()
                if not targets:
                    return _fail("usage: /vision document <image> [format] — "
                                 "attach image(s) or pass path(s)/URL(s); "
                                 "name a format to win (word, html, txt, "
                                 "pdf, xlsx)")
                words = _question_minus(targets).split()
                fmt = None
                if words and words[0].lower() in FORMATS:
                    fmt = words[0].lower()
                    words = words[1:]
                instruction = " ".join(words).strip()
                result = _deliver_document(targets, fmt, instruction)
                lines = [f"📄 {result['filename']} "
                         f"({result['format']} — {result['reason']})"]
                if not result["corrected"]:
                    lines.append(f"[{result['correction_notes'][0]}]")
                if result["uncertain"]:
                    lines.append(f"[{result['uncertain']} uncertain region(s) "
                                 "marked [illegible]]")
                if result.get("send_error"):
                    lines.append(f"[the file could not be sent here: "
                                 f"{result['send_error']} — it is saved at "
                                 f"{result['path']}]")
                lines.append("")
                lines.append(result["text_preview"])
                return _send("\n".join(lines))

            if sub == "read-text":
                targets = _all_targets()
                if not targets:
                    return _fail("usage: /vision read-text <image> — attach "
                                 "an image or pass a path/URL")
                question = _question_minus(targets)
                result = _deliver_document(targets, None, question)
                lines = [f"📝 [{result['method']}]", result["text"]]
                if not result["corrected"]:
                    lines.append(f"[{result['correction_notes'][0]}]")
                lines.append(f"📄 document: {result['filename']} "
                             f"({result['format']} — {result['reason']})")
                if result["uncertain"]:
                    lines.append(f"[{result['uncertain']} uncertain region(s) "
                                 "marked [illegible]]")
                if result.get("send_error"):
                    lines.append(f"[the file could not be sent here: "
                                 f"{result['send_error']} — it is saved at "
                                 f"{result['path']}]")
                return _send("\n".join(lines))

            if sub == "locate":
                targets = _pick_targets(1)
                question = _question_minus(targets or [])
                if not targets or not question:
                    return _fail('usage: /vision locate "<target>" <image> — '
                                 'e.g. /vision locate "the send button" shot.png')
                value = _call("vision_locate", path=targets[0],
                              target=question)
                return _send(self._render_locate(value))

            # describe (default)
            targets = _pick_targets(1)
            question = _question_minus(targets or [])
            if not targets:
                return _fail("usage: /vision [question] — attach an image or "
                             "pass a path/URL")
            value = _call("vision_describe", path=targets[0],
                          prompt=question)
        except RuntimeError as exc:
            return _fail(f"vision failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            return _fail(f"vision failed: {exc}")

        description = value.get("description") or "(no description returned)"
        provider = value.get("provider") or "vision"
        # Vision output is untrusted data — the tool already marks it.
        return _send(f"👁 [{provider}]\n{description}")

    # ── /vision renderers ────────────────────────────────────────────────

    @staticmethod
    def _render_locate(value: dict[str, Any]) -> str:
        target = value.get("target", "?")
        if not value.get("found"):
            return (f"🎯 '{target}': not found "
                    f"(method: {value.get('method', '?')})\n"
                    f"{value.get('disclaimer', '')}")
        method = value.get("method", "?")
        score = value.get("score", value.get("confidence", "?"))
        return (f"🎯 '{target}': x={value['x']} y={value['y']} "
                f"w={value['w']} h={value['h']} "
                f"(score {score}, method: {method})\n"
                f"{value.get('disclaimer', '')}")

    @staticmethod
    def _render_compare(value: dict[str, Any]) -> str:
        lines = []
        native = value.get("native") or {}
        if native:
            if native.get("identical"):
                lines.append("🔍 native diff: pixel-identical")
            else:
                frac = (native.get("changed_fraction") or 0.0) * 100
                lines.append(f"🔍 native diff: {frac:.1f}% of pixels changed")
                bbox = native.get("changed_bbox_1000")
                if bbox:
                    lines.append(f"   region: x={bbox['x']} y={bbox['y']} "
                                 f"w={bbox['w']} h={bbox['h']} (0-1000)")
        desc = (value.get("description") or "").strip()
        if desc and value.get("method") != "native":
            lines.append("")
            lines.append(desc)
        elif not native:
            lines.append(desc or "(no comparison returned)")
        return "\n".join(lines) or "(no comparison returned)"

    @staticmethod
    def _render_analyze(value: dict[str, Any], section: str) -> str:
        if section == "analyze":
            return RuntimeMediaMixin._render_analyze_full(value)
        if section == "colors":
            colors = (value.get("colors") or {})
            if not colors.get("available", True):
                return f"🎨 colors unavailable: {colors.get('why', '?')}"
            dom = colors.get("dominant") or []
            lines = ["🎨 palette:"]
            for c in dom[:5]:
                lines.append(f"  {c['hex']} — {c['share'] * 100:.1f}%")
            lines.append(f"brightness {colors.get('brightness')}/255 · "
                         f"contrast {colors.get('contrast')} · "
                         f"saturation {colors.get('saturation_pct')}%")
            return "\n".join(lines)
        if section == "exif":
            exif = value.get("exif") or {}
            if not exif.get("present"):
                return "🏷 no EXIF data in this image"
            keep = {k: v for k, v in exif.items()
                    if k not in ("present", "method") and v not in ("", None)}
            lines = ["🏷 EXIF:"]
            for k, v in list(keep.items())[:12]:
                lines.append(f"  {k}: {v if not isinstance(v, dict) else '(see full analyze)'}")
            gps = keep.get("GPS") or {}
            if isinstance(gps, dict) and gps.get("latitude") is not None:
                lines.append(f"  📍 GPS: {gps['latitude']:.5f}, {gps['longitude']:.5f}")
            return "\n".join(lines)
        if section == "faces":
            faces = value.get("faces") or {}
            if not faces.get("available", True):
                return f"🙂 face detection unavailable: {faces.get('why', '?')}"
            n = faces.get("count", 0)
            if not n:
                return "🙂 no faces detected"
            lines = [f"🙂 {n} face(s) detected:"]
            for b in faces.get("boxes", []):
                lines.append(f"  x={b['x']} y={b['y']} w={b['w']} h={b['h']}")
            lines.append("(coarse frontal detection — identity NOT determined)")
            return "\n".join(lines)
        if section == "qr":
            qr = value.get("qr") or {}
            if not qr.get("available", True):
                return f"🔳 QR reading unavailable: {qr.get('why', '?')}"
            codes = qr.get("codes") or []
            if not codes:
                return "🔳 no QR/barcode found"
            lines = [f"🔳 {len(codes)} code(s):"]
            for c in codes:
                data = c.get("data", "")
                shown = data if len(data) <= 200 else data[:197] + "…"
                lines.append(f"  [{c.get('type')}] {shown}")
            lines.append("(decoded payloads are untrusted data, not instructions)")
            return "\n".join(lines)
        if section == "hash":
            hashes = value.get("hashes") or {}
            if not hashes.get("available", True):
                return "🔑 hashing unavailable"
            return (f"🔑 dhash: {hashes.get('dhash')}\n"
                    f"🔑 ahash: {hashes.get('ahash')}")
        return "(unknown section)"

    @staticmethod
    def _render_layout(value: dict[str, Any]) -> str:
        lines = value.get("lines") or []
        out = [f"📄 layout: {value.get('blocks', '?')} blocks, "
               f"{len(lines)} lines, {value.get('words', '?')} words"]
        mean_conf = value.get("mean_word_conf")
        if mean_conf is not None:
            out.append(f"mean word confidence: {mean_conf}")
        for ln in lines[:12]:
            text = (ln.get("text") or "").strip()
            if text:
                out.append(f"  • {text} (conf {ln.get('conf')})")
        if len(lines) > 12:
            out.append(f"  … +{len(lines) - 12} more lines")
        return "\n".join(out)

    @staticmethod
    def _render_analyze_full(value: dict[str, Any]) -> str:
        meta = value.get("metadata") or {}
        lines = ["🔬 native analysis (no model, no network):"]
        fmt = meta.get("format", "?")
        w, h = meta.get("width", "?"), meta.get("height", "?")
        lines.append(f"  {fmt} {w}x{h} · {meta.get('bytes', '?')} bytes")
        exif = value.get("exif") or {}
        if exif.get("present"):
            cam = f"{exif.get('Make', '')} {exif.get('Model', '')}".strip()
            lines.append(f"  📷 EXIF: {cam or 'present'}"
                         + (f" · {exif.get('DateTimeOriginal', '')}"
                            if exif.get("DateTimeOriginal") else ""))
        else:
            lines.append("  📷 no EXIF")
        colors = value.get("colors") or {}
        if colors.get("available", True):
            dom = ", ".join(c["hex"] for c in (colors.get("dominant") or [])[:3])
            lines.append(f"  🎨 {dom} · brightness {colors.get('brightness')}")
        quality = value.get("quality") or {}
        if quality.get("available", True):
            lines.append(f"  🔍 sharpness {quality.get('sharpness')} "
                         f"({quality.get('sharpness_label')}) · "
                         f"entropy {quality.get('entropy_bits')} bits")
        faces = value.get("faces") or {}
        if faces.get("available", True):
            lines.append(f"  🙂 faces: {faces.get('count', 0)}")
        qr = value.get("qr") or {}
        if qr.get("available", True) and qr.get("count"):
            lines.append(f"  🔳 QR/barcodes: {qr.get('count')}")
        hashes = value.get("hashes") or {}
        if hashes.get("available", True):
            lines.append(f"  🔑 dhash {hashes.get('dhash')}")
        return "\n".join(lines)

    @staticmethod
    def _render_capabilities(value: dict[str, Any]) -> str:
        lines = [f"👁 vision capabilities (profile: {value.get('profile', '?')}):",
                 "", "native — offline, no key, no tokens:"]
        for name, info in (value.get("native") or {}).items():
            if info.get("available"):
                lines.append(f"  ✅ {name}: {info.get('what', '')}")
            else:
                lines.append(f"  ❌ {name}: {info.get('why', '')}"
                             + (f" ({info.get('install')})"
                                if info.get("install") else ""))
        lines.append("")
        lines.append("needs a vision model:")
        for name, why in (value.get("needs_model") or {}).items():
            lines.append(f"  🧠 {name}: {why}")
        if value.get("router_vision"):
            lines.append("\nmodel path: available ✅")
        else:
            lines.append("\nmodel path: not configured ❌ "
                         "(native analyses still work)")
        return "\n".join(lines)

    def _control_image(self, tail: str, *, chat_key: str = "",
                       message: Any = None) -> str:
        ref = (tail or "").strip()
        if not ref:
            return "usage: /image <path-or-url> (lookup) or /image <prompt> (generate)"
        if _looks_like_path_or_url(ref):
            return self._image_lookup(ref)
        return self._image_generate(ref, chat_key=chat_key,
                                    message=message)

    def _image_lookup(self, ref: str) -> str:
        outcome = self.context.tools.call("image_lookup", path=ref)
        data, error = self._tool_data(outcome)
        if data is None:
            return f"image lookup failed: {error}"
        lines = [f"🖼 {data.get('format', '?')} · {data.get('width') or '?'}×{data.get('height') or '?'} · "
                 f"{data.get('bytes', 0) // 1024} KB"]
        if data.get("seen_before"):
            where = f" (last: {data['seen_where']})" if data.get("seen_where") else ""
            lines.append(f"seen before{data.get('seen_days', '')}{where}")
        near = data.get("near_duplicates") or []
        if near:
            lines.append("near duplicates: " + "; ".join(near[:3]))
        return "\n".join(lines)

    def _image_generate(self, prompt: str, *, chat_key: str = "",
                        message: Any = None) -> str:
        """Generate an image from a text prompt via the generative backend."""
        try:
            from ...media_edit.generate import GenerativeEditError, get_backend
        except ImportError as exc:
            return f"image generation unavailable: {exc}"
        try:
            backend = get_backend()
        except GenerativeEditError as exc:
            return f"image generation unavailable: {exc}"
        try:
            images = backend.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - backend errors surface as text
            # A missing credential is not a dead end on owner surfaces:
            # offer the self-provisioning drive and remember this
            # request so it retries once the key lands.
            from ...core.errors import MissingCredentialError

            if isinstance(exc, MissingCredentialError):
                offer = self._provision_offer_for_missing(
                    exc, retry=("image_generate", prompt),
                    chat_key=chat_key, message=message)
                base = f"image generation failed: {exc}"
                return f"{base}\n{offer}" if offer else base
            return f"image generation failed: {exc}"
        if not images:
            return "image generation failed: backend returned no images"
        # Save the first image to the workspace and report the path
        try:
            import time
            from pathlib import Path
            out_dir = Path(self.context.settings.resolve("data/media/generated"))
            out_dir.mkdir(parents=True, exist_ok=True)
            # Nanosecond stamp: a 1-second name made any two generations
            # in the same second (any chat) overwrite each other while
            # both replies named the same file.
            fname = f"img_{time.time_ns()}.png"
            out_path = out_dir / fname
            img = images[0]
            if hasattr(img, "save"):
                img.save(str(out_path))
            else:
                return f"🎨 generated image for: {prompt!r} (could not save — unsupported image type)"
            return (f"🎨 generated: {prompt}\n"
                    f"saved: {out_path}\n"
                    f"backend: {backend.describe()}")
        except Exception as exc:  # noqa: BLE001 - save errors surface as text
            return f"image generated but save failed: {exc}"

    def _control_imggen(self, tail: str) -> str:
        """Devon's own image studio in chat.

        /imggen <prompt> [--seed N] [--ar 16:9] [--negative "..."]
            [--dtype bf16] [--quant int8]
        /imggen upscale <path> [--scale 2]
        /imggen checkpoints | /imggen dashboard <run>
        """
        ref = (tail or "").strip()
        if not ref:
            return ("imggen — Devon's own image studio:\n"
                    "/imggen <prompt> [--seed N] [--ar 16:9] "
                    "[--negative \"...\"] [--dtype bf16] "
                    "[--quant int8]\n"
                    "/imggen upscale <path> [--scale 2]\n"
                    "/imggen checkpoints\n"
                    "/imggen dashboard <run>")
        parts = ref.split(None, 1)
        action = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        try:
            if action == "checkpoints":
                from ...media.imggen.registry import list_models

                entries = list_models()
                if not entries:
                    return ("no native checkpoints yet — the first "
                            "/imggen <prompt> fetches the stock SD1.5 "
                            "checkpoint itself; or train one: "
                            "`nm imggen train --data <photo-folder>`, "
                            "or drop an SD1.5 .safetensors into "
                            "~/.nomorals/imggen")
                lines = []
                for e in entries:
                    status = ("loadable" if e["loadable"] else
                              f"not loadable: {e['loadable_reason']}")
                    lines.append(
                        f"• {e['name']} [{e['model_class']}]: "
                        f"{e['path']} ({e['bytes'] / 1e6:.1f} MB) "
                        f"— {status}")
                # Readiness names the model that would actually load,
                # at which dtype/quant, on which device.
                from ...media.imggen.studio import Studio

                ok, line = Studio().readiness()
                lines.append("readiness: " + line if ok
                             else f"readiness: not ready — {line}")
                # The local render median, when any local render has
                # been timed — the number behind the CPU-slow route.
                timing = Studio().timing_line()
                if timing:
                    lines.append(timing)
                return "\n".join(lines)
            if action == "dashboard":
                from ...media.imggen.studio import training_dashboard

                return training_dashboard(rest or "devon-ddpm")
            if action == "upscale":
                from ...media.imggen.studio import Studio

                bits = rest.split()
                scale = 2.0
                path = rest
                for i, b in enumerate(bits):
                    if b == "--scale" and i + 1 < len(bits):
                        try:
                            scale = float(bits[i + 1])
                        except ValueError as _partner_exc:
                            record_failure("runtime_media.RuntimeMediaMixin._control_imggen", _partner_exc)
                        path = " ".join(
                            bits[:i] + bits[i + 2:]).strip()
                        break
                paths = Studio().upscale(path, scale=scale)
                return f"🎨 upscaled ×{scale}: {paths[0]}"
            # Default: generation. Parse --flags inline. The
            # prompt is the WHOLE tail: the first word is not a
            # subcommand (those are handled above), and splitting
            # on ``rest`` ate it — "/imggen cyberpunk city"
            # generated "city" alone.
            import shlex

            try:
                toks = shlex.split(ref)
            except ValueError:
                toks = ref.split()
            kwargs: dict = {}
            prompt_toks: list[str] = []
            i = 0
            while i < len(toks):
                t = toks[i]
                if t == "--seed" and i + 1 < len(toks):
                    kwargs["seed"] = int(toks[i + 1]); i += 2
                elif t == "--ar" and i + 1 < len(toks):
                    from ...media.imggen.pipeline import resolve_aspect_ratio

                    w, h = resolve_aspect_ratio(toks[i + 1])
                    kwargs["width"], kwargs["height"] = w, h; i += 2
                elif t == "--negative" and i + 1 < len(toks):
                    kwargs["negative_prompt"] = toks[i + 1]; i += 2
                elif t == "--steps" and i + 1 < len(toks):
                    kwargs["steps"] = int(toks[i + 1]); i += 2
                elif t == "--dtype" and i + 1 < len(toks):
                    kwargs["dtype"] = toks[i + 1]
                    i += 2
                elif t == "--quant" and i + 1 < len(toks):
                    kwargs["quant"] = toks[i + 1]
                    i += 2
                else:
                    prompt_toks.append(t); i += 1
            prompt = " ".join(prompt_toks).strip()
            if not prompt:
                return "usage: /imggen <prompt> [--seed N] [--ar 16:9]"
            # One front door, not the Studio alone: native
            # (self-provisioning the stock checkpoint) → diffusers
            # → HF serverless. A bare prompt works out of the box —
            # no training, no file drop, no flag.
            from ...media_edit.generate import generate_image

            images, rung, notes = generate_image(prompt, **kwargs)
            if not images:
                return "imggen failed: backend returned no images"
            import time
            from pathlib import Path

            out_dir = Path(self.context.settings.resolve(
                "data/media/generated"))
            out_dir.mkdir(parents=True, exist_ok=True)
            # Nanosecond stamp: a 1-second name made any two
            # generations in the same second overwrite each other.
            fname = f"img_{time.time_ns()}.png"
            out_path = out_dir / fname
            img = images[0]
            if hasattr(img, "save"):
                img.save(str(out_path))
            else:
                return (f"🎨 generated image for: {prompt!r} (could "
                        "not save — unsupported image type)")
            lines = [f"🎨 generated: {prompt}", f"saved: {out_path}"]
            if rung == "native":
                lines.extend(n for n in notes
                             if n.startswith("provision: "))
            else:
                lines.append(f"via {rung}")
                for n in notes:
                    if n.startswith("native:"):
                        lines.append(
                            "local unavailable: "
                            + n[len("native: "):])
                        break
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"imggen failed: {exc}"

    def _control_poster(self, tail: str) -> str:
        """A finished poster in one call: generated art + exact text.

        /poster <art prompt> :: <title> [:: <line> ...]
                [--ar 9:16] [--seed N] [--steps N]
        The first text is the title (center, large); the rest join
        into one footer block at the bottom. Any line may carry a
        position prefix (top:TEXT, bottom:TEXT, ...). Flags may sit
        anywhere in the tail.
        """
        ref = (tail or "").strip()
        usage = ("poster — generated art + exact typography in one call:\n"
                 "/poster <art prompt> :: <title> [:: <line> ...] "
                 "[--ar 9:16] [--seed N] [--steps N]\n"
                 "example: /poster a Lagos owambe at night :: OWAMBE "
                 ":: Saturday 8 PM")
        if not ref or "::" not in ref:
            return usage
        try:
            import shlex

            from ...media.upgrade.typography import POSITIONS

            # Flags may sit in any segment; pull them out of each
            # segment before the "::" split becomes prompt + texts.
            cleaned: list[str] = []
            call_kwargs: dict = {}
            for part in [p.strip() for p in ref.split("::")]:
                try:
                    toks = shlex.split(part)
                except ValueError:
                    toks = part.split()
                kept: list[str] = []
                i = 0
                while i < len(toks):
                    t = toks[i]
                    if t == "--seed" and i + 1 < len(toks):
                        call_kwargs["seed"] = int(toks[i + 1])
                        i += 2
                    elif t == "--ar" and i + 1 < len(toks):
                        call_kwargs["aspect"] = toks[i + 1]
                        i += 2
                    elif t == "--steps" and i + 1 < len(toks):
                        call_kwargs["steps"] = int(toks[i + 1])
                        i += 2
                    else:
                        kept.append(t)
                        i += 1
                cleaned.append(" ".join(kept).strip())
            prompt = cleaned[0]
            texts = [s for s in cleaned[1:] if s]
            if not prompt or not texts:
                return usage
            layers: list[dict] = []
            footer: list[str] = []
            for j, seg in enumerate(texts):
                pos = None
                text = seg
                if ":" in seg:
                    maybe, rest = seg.split(":", 1)
                    if maybe.strip().lower() in POSITIONS and rest.strip():
                        pos, text = maybe.strip().lower(), rest.strip()
                if j == 0:
                    layers.append({"text": text,
                                   "position": pos or "center",
                                   "size": 96})
                elif pos is not None:
                    layers.append({"text": text, "position": pos,
                                   "size": 48})
                else:
                    footer.append(text)
            if footer:
                layers.append({"text": "\n".join(footer),
                               "position": "bottom", "size": 48})
            from ...media.poster import make_poster

            result = make_poster(prompt, layers, **call_kwargs)
            return f"🖼️ poster: {prompt}\nsaved: {result['path']}"
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"poster failed: {exc}"

    # ── 3D run/steer (audit gap R2-25) ─────────────────────────────────
    def _control_trellis(self, tail: str, *, chat_key: str = "",
                         message: Any = None) -> str:
        """`/trellis <image path-or-url> [out] [--seed N] [--resolution ...]`
        — one photo -> a textured GLB, via TRELLIS.2.

        Attached photo wins over any path argument (the /video watch
        shape); a URL is fetched; a path is used as-is. The organ's own
        GPU gate (Linux + NVIDIA 24 GB+) fires inside the tool before
        any run — a refusal names the measured numbers. Owner-only: a
        run spends a 24 GB GPU for up to 30 minutes. Never raises.
        """
        if message is not None:
            if not self._is_operator(message):
                return "that one's just for the owner."
        elif chat_key:
            from ...partner.gating import is_owner_chat

            chat = self._ref_from_key(chat_key)
            if not is_owner_chat(chat, owner_chats=getattr(
                    self, "_owner_chats", ())):
                return "that one's just for the owner."
        else:
            return "that one's just for the owner."
        usage = ("trellis — one photo -> a textured GLB (TRELLIS.2):\n"
                 "/trellis <image path-or-url> [out] [--seed N] "
                 "[--resolution 512|1024|1024_cascade|1536_cascade]\n"
                 "attach a photo, or pass its link or path")
        try:
            import shlex
            import time
            from pathlib import Path

            attached = ""
            if message is not None:
                for media in getattr(message, "media", None) or []:
                    if getattr(media, "kind", "") == "image":
                        attached = getattr(media, "path", "") or ""
                        break
            try:
                tokens = shlex.split(tail or "")
            except ValueError:
                tokens = (tail or "").split()
            seed = 42
            resolution = ""
            positional: list[str] = []
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t == "--seed" and i + 1 < len(tokens):
                    try:
                        seed = int(tokens[i + 1])
                    except ValueError:
                        return usage
                    i += 2
                elif t == "--resolution" and i + 1 < len(tokens):
                    resolution = tokens[i + 1]
                    i += 2
                else:
                    positional.append(t)
                    i += 1
            image_ref = attached or (positional[0] if positional else "")
            if not image_ref:
                return usage
            out = positional[1] if len(positional) > 1 else ""
            image_path = _resolve_3d_input(
                image_ref, settings=self.context.settings, tag="trellis")
            if not out:
                out_dir = Path(self.context.settings.resolve(
                    "data/media/generated"))
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(image_path).stem
                out = str(out_dir / f"{stem}_{time.time_ns()}.glb")
            kwargs: dict[str, Any] = {"image": image_path, "out": out,
                                      "seed": seed}
            if resolution:
                kwargs["resolution"] = resolution
            outcome = self.context.tools.call("trellis_image_to_3d",
                                              **kwargs)
            data, error = _tool_result_dict(outcome)
            if data is None:
                return str(error)
            lines = [f"🧊 {data.get('detail') or 'GLB written'}"]
            if data.get("glb"):
                lines.append(f"saved: {data['glb']}")
                lines.append(f"import it live: /blender import {data['glb']}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"trellis failed: {exc}"

    def _control_unirig(self, tail: str, *, chat_key: str = "",
                        message: Any = None) -> str:
        """`/unirig <mesh path-or-url> [out] [--seed N]` — any mesh -> a
        rigged, animatable character, via UniRig.

        A URL is fetched; a path is used as-is. The organ's own GPU gate
        (Linux + CUDA 8 GB+) fires inside the tool before any run — a
        refusal names the measured numbers. Owner-only: a run spends a
        GPU for up to 30 minutes. Never raises.
        """
        if message is not None:
            if not self._is_operator(message):
                return "that one's just for the owner."
        elif chat_key:
            from ...partner.gating import is_owner_chat

            chat = self._ref_from_key(chat_key)
            if not is_owner_chat(chat, owner_chats=getattr(
                    self, "_owner_chats", ())):
                return "that one's just for the owner."
        else:
            return "that one's just for the owner."
        usage = ("unirig — any mesh -> a rigged character (UniRig):\n"
                 "/unirig <mesh path-or-url> [out] [--seed N]\n"
                 "pass a mesh file (.glb/.obj/.fbx/.vrm/.gltf/.dae), "
                 "its link or path")
        try:
            import shlex
            import time
            from pathlib import Path

            try:
                tokens = shlex.split(tail or "")
            except ValueError:
                tokens = (tail or "").split()
            seed = 12345
            positional: list[str] = []
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t == "--seed" and i + 1 < len(tokens):
                    try:
                        seed = int(tokens[i + 1])
                    except ValueError:
                        return usage
                    i += 2
                else:
                    positional.append(t)
                    i += 1
            if not positional:
                return usage
            mesh_ref = positional[0]
            out = positional[1] if len(positional) > 1 else ""
            mesh_path = _resolve_3d_input(
                mesh_ref, settings=self.context.settings, tag="unirig")
            if not out:
                out_dir = Path(self.context.settings.resolve(
                    "data/media/generated"))
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(mesh_path).stem
                out = str(out_dir / f"{stem}_{time.time_ns()}.fbx")
            outcome = self.context.tools.call(
                "unirig_rig_mesh", mesh=mesh_path, out=out, seed=seed)
            data, error = _tool_result_dict(outcome)
            if data is None:
                return str(error)
            lines = [f"🦴 {data.get('detail') or 'rigged'}"]
            if data.get("rigged"):
                lines.append(f"saved: {data['rigged']}")
                lines.append(
                    f"import it live: /blender import {data['rigged']}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"unirig failed: {exc}"

    def _control_blender(self, tail: str, *, chat_key: str = "",
                         message: Any = None) -> str:
        """`/blender status | scene [query] | look [mode] | code <bpy> |
        import <glb|fbx>` — steer the operator's live Blender session.

        Every verb connects first; when no live Blender is reachable
        the tool's own refusal names the setup step. Devon never starts
        Blender herself. Owner-only. Never raises.
        """
        if message is not None:
            if not self._is_operator(message):
                return "that one's just for the owner."
        elif chat_key:
            from ...partner.gating import is_owner_chat

            chat = self._ref_from_key(chat_key)
            if not is_owner_chat(chat, owner_chats=getattr(
                    self, "_owner_chats", ())):
                return "that one's just for the owner."
        else:
            return "that one's just for the owner."
        usage = ("blender — steer the operator's live Blender:\n"
                 "/blender status | scene [query] | look [mode] | "
                 "code <bpy> | import <glb|fbx>\n"
                 "start Blender with the MCP for Blender add-on running, "
                 "then try again")
        try:
            import shlex

            try:
                tokens = shlex.split(tail or "")
            except ValueError:
                tokens = (tail or "").split()
            if not tokens:
                return usage
            action = tokens[0].lower()
            rest = tokens[1:]
            if action == "status":
                call_kwargs: dict[str, Any] = {"action": "status"}
            elif action == "scene":
                call_kwargs = {"action": "scene",
                               "query": " ".join(rest)}
            elif action == "look":
                call_kwargs = {"action": "look",
                               "mode": rest[0] if rest else "angles"}
            elif action == "code":
                if not rest:
                    return usage
                call_kwargs = {"action": "code", "code": " ".join(rest)}
            elif action == "import":
                if not rest:
                    return usage
                call_kwargs = {"action": "import", "path": rest[0]}
            else:
                return usage
            outcome = self.context.tools.call("blender_live",
                                              **call_kwargs)
            data, error = _tool_result_dict(outcome)
            if data is None:
                return str(error)
            text = str(data.get("text") or data.get("detail") or
                       "blender: ok")
            if action == "look" and data.get("image_b64"):
                text += ("\n(screenshot captured — the image rides the "
                         "tool result; the caption above is what Blender "
                         "showed)")
            return text
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"blender failed: {exc}"

    def _control_shorts(self, tail: str) -> str:
        """Devon's short-form content empire in chat.

        /shorts make <niche> "<topic>" [--now] [--style NAME]
            [--footage stock|stills|auto] [--voice edge|catalogue]
            — queue or render now (--footage/--voice opt into the
            factory rungs: stock footage, Edge TTS voiceover)
        /shorts status [job_id]                    — job list / inspect
        /shorts resume <run_id>                    — resume a failed render
        /shorts niches                             — list niches
        /shorts calendar [--due]                   — content calendar
        /shorts ledger                             — what got posted where
        /shorts estimate <niche> "<topic>"         — time/cost estimate
        /shorts hero "<prompt>" [--seconds N] [--seed N] [--model NAME]
            — one neural hero clip (videogen), or its honest decline
        """
        import shlex

        ref = (tail or "").strip()
        if not ref:
            return ("🎬 /shorts — Devon's short-form content empire:\n"
                    "/shorts make <niche> \"<topic>\" [--now] [--style NAME] "
                    "[--footage stock|stills|auto] [--voice edge|catalogue]\n"
                    "/shorts niches — list content niches\n"
                    "/shorts status [job] · /shorts resume <run_id>\n"
                    "/shorts calendar [--due] · /shorts ledger\n"
                    "/shorts estimate <niche> \"<topic>\"\n"
                    "/shorts hero \"<prompt>\" — one neural hero clip")
        try:
            toks = shlex.split(ref)
        except ValueError:
            toks = ref.split()
        if not toks:
            return "usage: /shorts make|status|resume|niches|calendar|ledger|estimate|hero ..."
        action = toks[0].lower()
        rest = toks[1:]
        if action == "hero":
            return self._shorts_hero(rest)
        try:
            from ...media.contentops.pipeline import ShortPipeline

            pipe = ShortPipeline()
            if action == "niches":
                from ...media.contentops.niches import get_niche, list_niches

                lines = ["🎬 niches:"]
                for name in list_niches():
                    plugin = get_niche(name)
                    lines.append(f"• {name} ({getattr(plugin, 'cadence', '?')}/day) — "
                                 f"{getattr(plugin, 'thesis', '')[:70]}")
                return "\n".join(lines)
            if action == "make":
                if len(rest) < 2:
                    return ("usage: /shorts make <niche> \"<topic>\" "
                            "[--now] [--style phonk|documentary|vlog|minimal] "
                            "[--footage stock|stills|auto] "
                            "[--voice edge|catalogue]")
                niche = rest[0]
                toks = rest[1:]
                now = "--now" in toks
                style = ""
                if "--style" in toks:
                    si = toks.index("--style")
                    if si + 1 < len(toks):
                        style = toks[si + 1]
                        del toks[si:si + 2]
                footage = ""
                if "--footage" in toks:
                    fi = toks.index("--footage")
                    if fi + 1 < len(toks):
                        footage = toks[fi + 1]
                        del toks[fi:fi + 2]
                voice = ""
                if "--voice" in toks:
                    vi = toks.index("--voice")
                    if vi + 1 < len(toks):
                        voice = toks[vi + 1]
                        del toks[vi:vi + 2]
                toks = [t for t in toks if t != "--now"]
                topic = " ".join(toks).strip("\"'").strip()
                job = pipe.plan(niche, topic, style=style,
                                footage=footage, voice=voice)
                if now:
                    result = pipe.run(job)
                    if result.ok:
                        return (f"🎬 rendered {job.id}\n📁 {result.final_path}\n"
                                f"stages: " + ", ".join(
                                    f"{k}={v}ms" for k, v in
                                    (result.stages or {}).items()))
                    return f"🎬 render failed: {result.error}"
                style_note = f" [{style}]" if style else ""
                return (f"🎬 queued {job.id}{style_note} ({niche} — {topic[:60]})\n"
                        f"render now: /shorts resume-later via `nm shorts` "
                        f"or re-run with --now")
            if action == "status":
                jobs = pipe.jobs.list()
                if rest:
                    q = rest[0]
                    jobs = [j for j in jobs if j.id == q or j.id.startswith(q)]
                if not jobs:
                    return "no jobs yet"
                return "\n".join(
                    f"• {j.id} [{j.status}] {j.niche} — {j.topic[:50]}"
                    for j in jobs[-10:])
            if action == "resume":
                if not rest:
                    return "usage: /shorts resume <run_id>"
                result = pipe.resume(rest[0])
                if result.ok:
                    return f"🎬 resumed → {result.final_path}"
                return f"🎬 resume failed: {result.error}"
            if action == "calendar":
                posts = pipe.calendar.due() if "--due" in rest else pipe.calendar.list()
                if not posts:
                    return "calendar empty"
                return "\n".join(
                    f"• {p.id} [{p.status}] {p.niche} — {p.topic[:50]}"
                    for p in posts[-10:])
            if action == "ledger":
                from ...media.contentops.publish.ledger import PublishLedger

                entries = PublishLedger().list()
                if not entries:
                    return "nothing posted yet"
                return "\n".join(
                    f"• {e.get('at', '')} {e.get('platform', '')} "
                    f"[{e.get('status', '')}] {str(e.get('title', ''))[:50]}"
                    for e in entries[-10:])
            if action == "estimate":
                if len(rest) < 2:
                    return "usage: /shorts estimate <niche> \"<topic>\""
                import json as _json

                est = pipe.estimate(rest[0], " ".join(rest[1:]))
                return "⏱ estimate:\n" + _json.dumps(est, indent=1, default=str)[:1500]
            return f"unknown /shorts action {action!r}"
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"shorts failed: {exc}"

    # ── trendops footage ask ──────────────────────────────────────────
    # An Apply tap on a trendops proposal (or a bare
    # ``/trendops apply <format>``) carries no footage — the format
    # alone cannot cut.  The command arms a one-shot ask in this chat
    # and the owner's next message IS the footage: a sent video/file,
    # or a path/URL in text.  The ask is answered in _process_inner
    # (runtime.py), after the proposal reason ask, and expires after
    # the same 15-minute TTL.  "cancel" drops it — typed, or
    # the [Cancel] button on the ask (Telegram and WhatsApp),
    # which re-enters as "/trendops cancel" and is answered
    # in _control_trendops.
    def _trendops_ask_store(self) -> dict[str, dict[str, Any]]:
        store = getattr(self, "_trendops_ask_store_v", None)
        if store is None:
            store = {}
            self._trendops_ask_store_v = store
        return store

    def _note_trendops_ask(self, chat_key: str, fmt_name: str) -> None:
        import time

        self._trendops_ask_store()[chat_key] = {
            "action": "apply", "format": fmt_name, "at": time.time()}

    def _trendops_ask_for_footage(self, chat_key: str,
                                  fmt_name: str) -> Any:
        """Arm the footage ask; the prompt with its Cancel button, or
        None when there is no chat to arm it on (then the caller falls
        back to usage).  The button taps ``/trendops cancel`` — the
        same drop the typed word performs."""
        if not chat_key:
            return None
        self._note_trendops_ask(chat_key, fmt_name)
        from ...social.chat.tgbot_buttons import actionable

        return actionable(
            f"🎬 applying the {fmt_name} format — send the footage "
            "(a video file, or its path/URL) and I'll cut it. "
            "('cancel' drops it.)",
            [[("❌ Cancel", "trendops cancel")]])

    def _trendops_ask_drop(self, chat_key: str) -> str | None:
        """Drop an armed footage ask; the drop line, or None when
        nothing is armed in this chat.  The one drop path for both
        the typed "cancel" and the Cancel button tap — the tap must
        do exactly what typing does."""
        if self._trendops_ask_store().pop(chat_key, None) is None:
            return None
        return "trendops apply dropped — no footage, nothing cut."

    def _trendops_ask_answer(self, message) -> str | None:  # type: ignore[no-untyped-def]
        """Answer an armed footage ask with the owner's next message.

        Returns the apply result's text, the drop line for "cancel",
        or None when no ask is armed, it expired, or the message
        carries no footage (the message then flows on untouched).
        """
        import time

        store = self._trendops_ask_store()
        chat_key = getattr(getattr(message, "chat", None), "key", "") or ""
        ask = store.get(chat_key)
        if not ask:
            return None
        if time.time() - float(ask.get("at") or 0) > self._PROPOSAL_ASK_TTL_S:
            store.pop(chat_key, None)
            return None
        text = (getattr(message, "text", "") or "").strip()
        media = list(getattr(message, "media", None) or [])
        if text.lower() == "cancel" and not media:
            return self._trendops_ask_drop(chat_key)
        footage = ""
        for item in media:
            path = getattr(item, "path", "") or ""
            if path:
                footage = path
                break
        if not footage:
            footage = text
        if not footage:
            return None
        store.pop(chat_key, None)
        result = self._trendops_apply(ask.get("format") or "", [footage])
        return result.text if hasattr(result, "text") else str(result)

    def _trendops_apply(self, fmt_name: str, footage: list[str], *,
                        music: str = "", caption: str = "",
                        beats: list[float] | None = None) -> Any:
        """Cut footage to a trend format and queue the post draft.

        The shared execution behind the ``apply`` command branch and
        the footage-ask answer: the render itself, then the drafts
        organ (the #43 rule) — the owner is asked "post directly or
        send for review?" through the draft, never handed a bare path.
        A draft failure never loses the render; the plain reply is
        the fallback and the miss is named.
        """
        from ...media.trendops import TrendOps

        ops = TrendOps(context=self.context)
        result = ops.apply_format(
            footage[0] if len(footage) == 1 else footage,
            fmt_name, music=music or None,
            caption=caption or None, beats=beats or None)
        review = result.get("review", {})
        problems = review.get("problems") or []
        note = ((" (review flags: " + "; ".join(problems) + ")")
                if problems else "")
        render_line = (f"🎬 cut to {result['format_name']} → "
                       f"{result['output']}\n"
                       f"{result['duration']:.1f}s · "
                       f"{result['width']}x{result['height']} · "
                       f"{result['segments']} segments · "
                       f"{result['beats_used']} beats{note}")
        # The temporal QA verdict rides with the reply so the owner
        # sees whether the cut actually landed — pass and fail both
        # speak; an unmeasurable render stays silent, as before.
        qa = review.get("qa") or {}
        if qa.get("status") in ("pass", "fail"):
            render_line += (f"\n🎞️ temporal QA "
                            f"{qa['status']}: {qa.get('notes', '')}")
        try:
            from ...media.trendops.editor import draft_for_render, stamp_render_draft
            from ...social.drafts import propose_post

            content, metadata = draft_for_render(
                result, footage=footage, music=music or None,
                caption=caption or None)
            queue = self._draft_queue()
            try:
                draft = queue.create_draft(
                    content, platforms=None, metadata=metadata)
                queue.propose(draft.id)
            finally:
                queue.close()
            # The cut's Museum card now names the draft that
            # carries it — a stamp failure never loses the
            # draft (stamp_render_draft never raises).
            stamp_render_draft(result["output"], draft.id)
            text, buttons = propose_post(draft)
            from ...social.chat.tgbot_buttons import actionable, grid_layout

            return actionable(f"{render_line}\n\n{text}",
                              grid_layout([buttons]))
        except Exception as exc:  # noqa: BLE001 - draft is delivery, not the render
            _log.warning("trendops: draft failed for %s: %s",
                         result.get("output"), exc)
            return (f"{render_line}\n"
                    f"(couldn't queue the draft: {exc})")

    @staticmethod
    def _animato_opts(toks: list[str]) -> dict[str, Any]:
        """Parse /animato tail tokens into {name|idea, --options}.

        Value options: --project/--niche/--style/--voice/--music/
        --channel/--schedule/--publish. Flag: --private. The first
        positional is the project name (``new``); for ``spark`` all
        positionals joined are the idea.
        """
        opts: dict[str, Any] = {}
        rest: list[str] = []
        value_opts = {"project", "niche", "style", "voice", "music",
                      "channel", "schedule", "publish"}
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == "--private":
                opts["private"] = True
            elif (tok.startswith("--") and tok[2:] in value_opts
                    and i + 1 < len(toks)):
                opts[tok[2:]] = toks[i + 1]
                i += 1
            else:
                rest.append(tok)
            i += 1
        if rest:
            opts["name"] = rest[0]
            opts["idea"] = " ".join(rest).strip("\"'").strip()
        return opts

    def _control_animato(self, tail: str) -> str:
        """Animato Studio in chat: spark → build → share.

        /animato new <name> [--niche N] [--style S] [--voice V]
            [--music M] [--channel C] [--schedule S]
        /animato list                          — studio projects
        /animato show <project>                — project detail
        /animato spark "<idea>" [--project NAME] [--publish youtube]
            [--private]                        — run one episode end to end
        /animato queue <project> "<topic>"     — queue an episode idea
        /animato cast <project> <name> [--voice V] [--face-ref PATH]
            — add a cast member; the face reference locks that
            character's face into every scene of every episode
        /animato post <project> <episode> [--private]
        /animato export <project>              — export the studio repo
            (brief + manifest + render scripts)
        /animato export <project> <episode>    — hand back the file
        /animato status [--project NAME]       — episode status
        /animato autopost start|stop|status|run [--project NAME] [--force]
            — the scheduled mission: due projects post themselves
        """
        import shlex

        ref = (tail or "").strip()
        if not ref:
            return ("🎬 /animato — Animato Studio: spark an idea into a "
                    "finished video.\n"
                    "/animato new <name> — start a series project\n"
                    "/animato spark \"<idea>\" [--project NAME] "
                    "[--publish youtube] — run one episode\n"
                    "/animato list · /animato show <project> · "
                    "/animato status\n"
                    "/animato queue <project> \"<topic>\" · "
                    "/animato cast <project> <name> [--face-ref PATH] · "
                    "/animato post <project> <episode> · "
                    "/animato export <project> [<episode>]\n"
                    "/animato autopost start|stop|status|run "
                    "[--project NAME] [--force]")
        try:
            toks = shlex.split(ref)
        except ValueError:
            toks = ref.split()
        if not toks:
            return "usage: /animato new|list|show|spark|queue|post|export|status ..."
        action = toks[0].lower()
        rest = toks[1:]
        from ...media.contentops.publish.youtube import YouTubePublisher
        from ...media.contentops.studio import StudioRunner

        # L5 may import the L5 publisher directly — pass the class as
        # the factory so the L4 runner never imports it itself.
        runner = StudioRunner(publisher_factory=YouTubePublisher)
        if action == "list":
            projects = runner.store.list()
            if not projects:
                return ("no studio projects yet — "
                        "/animato new <name> to start one")
            lines = ["🎬 studio projects:"]
            for p in projects:
                lines.append(f"• {p.slug} ({p.niche}) — "
                             f"{len(p.episodes)} episodes, "
                             f"{len(p.topic_queue)} queued")
            return "\n".join(lines)
        if action == "new":
            if not rest:
                return "usage: /animato new <name>"
            opts = self._animato_opts(rest)
            try:
                project = runner.store.create(
                    opts["name"], niche=opts.get("niche", "did_you_know"),
                    style=opts.get("style", ""), voice=opts.get("voice", "edge"),
                    music=opts.get("music", ""), channel=opts.get("channel", ""),
                    schedule=opts.get("schedule", ""))
            except ValueError as exc:
                return f"animato: {exc}"
            return (f"created studio project {project.slug} "
                    f"({project.niche} — {project.name})")
        if action == "show":
            if not rest:
                return "usage: /animato show <project>"
            try:
                project = runner.store.load(rest[0])
            except FileNotFoundError as exc:
                return f"animato: {exc}"
            lines = [f"{project.slug} — {project.name}",
                     f"niche: {project.niche}  style: "
                     f"{project.style or '(default)'}  voice: {project.voice}",
                     f"music: {project.music or '(none)'}  channel: "
                     f"{project.channel or '(not set)'}  schedule: "
                     f"{project.schedule or '(not set)'}"]
            if project.topic_queue:
                lines.append("queued topics:")
                lines.extend(f"• {t}" for t in project.topic_queue)
            if project.episodes:
                lines.append("episodes:")
                for e in project.episodes[-10:]:
                    lines.append(f"• {e.id}  {e.status}  {e.topic[:50]}")
            return "\n".join(lines)
        if action == "queue":
            if len(rest) < 2:
                return "usage: /animato queue <project> \"<topic>\""
            try:
                project = runner.store.queue_topic(
                    rest[0], " ".join(rest[1:]).strip("\"'").strip())
            except FileNotFoundError as exc:
                return f"animato: {exc}"
            return (f"queued in {project.slug} "
                    f"({len(project.topic_queue)} queued)")
        if action == "cast":
            if len(rest) < 2:
                return ("usage: /animato cast <project> <name> "
                        "[--voice V] [--face-ref PATH]")
            from ...media.contentops.studio import CastMember

            voice = "edge"
            face_ref = ""
            character_id = ""
            member = ""
            i = 1
            while i < len(rest):
                tok = rest[i]
                if tok in ("--voice", "--face-ref", "--character-id") \
                        and i + 1 < len(rest):
                    if tok == "--voice":
                        voice = rest[i + 1]
                    elif tok == "--face-ref":
                        face_ref = rest[i + 1]
                    else:
                        character_id = rest[i + 1]
                    i += 2
                elif not member:
                    member = tok
                    i += 1
                else:
                    i += 1
            if not member:
                return ("usage: /animato cast <project> <name> "
                        "[--voice V] [--face-ref PATH]")
            if face_ref:
                from pathlib import Path

                if not Path(face_ref).expanduser().is_file():
                    return f"animato: no face reference image at {face_ref!r}"
            try:
                project = runner.store.add_cast(
                    rest[0], CastMember(name=member, voice=voice,
                                        face_ref=face_ref,
                                        character_id=character_id))
            except FileNotFoundError as exc:
                return f"animato: {exc}"
            return (f"cast {member} → {project.slug} (voice: {voice}"
                    f"{', ref: ' + face_ref if face_ref else ', no reference'})")
        if action == "spark":
            opts = self._animato_opts(rest)
            idea = opts.get("idea", "")
            if not idea:
                return ("usage: /animato spark \"<idea>\" "
                        "[--project NAME] [--publish youtube]")
            slug = opts.get("project", "")
            try:
                if slug:
                    project = runner.store.load(slug)
                else:
                    from ...media.contentops.studio import slugify

                    slug = slugify(idea)
                    try:
                        project = runner.store.load(slug)
                    except FileNotFoundError:
                        project = runner.store.create(
                            idea, niche="did_you_know",
                            style=opts.get("style", ""),
                            voice=opts.get("voice", "edge"))
            except FileNotFoundError as exc:
                return f"animato: {exc}"
            if opts.get("style"):
                project.style = opts["style"]
            if opts.get("voice"):
                project.voice = opts["voice"]
            result = runner.spark(project, idea,
                                  publish=opts.get("publish", ""),
                                  private=bool(opts.get("private")))
            if not result.get("ok"):
                return "animato: " + "; ".join(result.get("notes", [])
                                               or ["spark failed"])
            out = (f"🎬 episode {result['episode_id']} → "
                   f"{result['video_path']}")
            if result.get("published"):
                out += f"\nposted: {result['published']}"
            return out
        if action == "post":
            if len(rest) < 2:
                return "usage: /animato post <project> <episode> [--private]"
            try:
                result = runner.publish_episode(
                    rest[0], rest[1], private="--private" in rest)
            except (FileNotFoundError, KeyError) as exc:
                return f"animato: {exc}"
            if not result.get("ok"):
                return f"animato: {result.get('reason', 'publish failed')}"
            return (f"posted episode {result['episode_id']} → "
                    f"{result.get('video_id', '')}")
        if action == "export":
            if not rest:
                return "usage: /animato export <project> [<episode>]"
            try:
                if len(rest) == 1:
                    # The step-5 form: the whole project as a studio repo.
                    result = runner.export_project(rest[0])
                    if not result.get("ok"):
                        return (f"animato: "
                                f"{result.get('reason', 'export failed')}")
                    cast = result.get("cast") or []
                    return (f"🎬 exported {result['project']} → "
                            f"{result['path']} ({len(result['files'])} "
                            f"files, {result['topics']} topics"
                            f"{', cast ' + ', '.join(cast) if cast else ''})")
                result = runner.export_episode(rest[0], rest[1])
            except (FileNotFoundError, KeyError) as exc:
                return f"animato: {exc}"
            if not result.get("ok"):
                return f"animato: {result.get('reason', 'export failed')}"
            return result["path"]
        if action == "status":
            opts = self._animato_opts(rest)
            slug = opts.get("project", "")
            try:
                projects = ([runner.store.load(slug)] if slug
                            else runner.store.list())
            except FileNotFoundError as exc:
                return f"animato: {exc}"
            rows = [(p.slug, e) for p in projects for e in p.episodes]
            if not rows:
                return ("no episodes yet — "
                        "/animato spark \"<idea>\" to make one")
            return "\n".join(
                f"{s}  {e.id}  {e.status}  {e.topic[:50]}"
                for s, e in rows[-20:])
        if action == "autopost":
            from ...media.contentops.autopost import (
                AutopostStore,
                autopost_status,
                run_autopost,
            )

            sub = (rest[0].lower() if rest else "status")
            opts = self._animato_opts(rest[1:])
            store = AutopostStore(runner.store.root)
            if sub == "start":
                store.set_enabled(True)
                return ("animato autopost started — due projects post "
                        "themselves every 30 minutes")
            if sub == "stop":
                store.set_enabled(False)
                return ("animato autopost stopped — the schedule will "
                        "not fire (manual run with --force still works)")
            if sub == "status":
                result = autopost_status(runner=runner, store=store)
                lines = [f"autopost: {'on' if result['enabled'] else 'off'}"]
                for row in result["projects"]:
                    state = ("DUE" if row["due"] else
                             f"last {row['last_status'] or 'never'} "
                             f"{row['last_run_at'] or ''}").strip()
                    lines.append(f"• {row['project']} ({row['schedule']}) — "
                                 f"{row['queued_topics']} queued, {state}")
                for run in result["recent"]:
                    lines.append(f"• {run['at']} {run['project']} "
                                 f"{run['status']} {run.get('topic', '')[:40]}"
                                 + (f" — {run['reason']}"
                                    if run.get('reason') else ""))
                return "\n".join(lines)
            if sub == "run":
                result = run_autopost(
                    runner=runner, store=store,
                    project=opts.get("project", ""),
                    force=bool(opts.get("force")))
                if not result.get("ok"):
                    return f"animato: {result.get('reason', 'autopost run failed')}"
                if not result.get("enabled") and not opts.get("force"):
                    return ("autopost is off — /animato autopost start "
                            "to run the schedule")
                lines = [f"autopost: ran {result['ran']} project(s), "
                         f"{result['posted']} posted, "
                         f"{result['failed']} failed, "
                         f"{result['skipped']} skipped"]
                for o in result["outcomes"]:
                    lines.append(f"• {o['project']} {o['status']} "
                                 f"{o.get('topic', '')[:40]}"
                                 + (f" → {o['video_id']}"
                                    if o.get('video_id') else "")
                                 + (f" — {o['reason']}"
                                    if o.get('reason') else ""))
                return "\n".join(lines)
            return ("usage: /animato autopost "
                    "start|stop|status|run [--project NAME] [--force]")
        return "usage: /animato new|list|show|spark|queue|post|export|status|autopost ..."

    def _control_clip(self, tail: str) -> str:
        """TrendClip in chat: find what's hot, cut it into a vertical clip.

        /clip <url> [--strategy gaming|conversation|educational|comedy|craft] [--no-brain]
            — ingest, transcribe locally, score moments (deterministic +
              brain) and leave the job ready with its drafts
        /clip render <job> <pick> [--layout balanced|punch|clean] [--dest youtube,facebook,tiktok]
            — render one picked moment into a validated vertical clip + package
        /clip status [job] — job list / inspect
        /clip queue <job> [platforms] — queue the rendered clip for the publisher sweep
        """
        import shlex

        from ...media.trendclip import TrendclipPipeline, topic_from_query, trendclip_root

        ref = (tail or "").strip()
        if not ref:
            return ("🎬 /clip — TrendClip: find what's hot, cut it into a vertical clip:\\n"
                    "/clip <url> [--strategy NAME] [--no-brain]\\n"
                    "/clip render <job> <pick> [--layout balanced|punch|clean]\\n"
                    "/clip status [job] · /clip queue <job> [platforms]\\n"
                    "Manual publish only — nothing auto-posts.")
        try:
            toks = shlex.split(ref)
        except ValueError:
            toks = ref.split()
        if not toks:
            return ("usage: /clip <url> | /clip render <job> <pick> | "
                    "/clip status [job] | /clip queue <job> [platforms]")
        action = toks[0].lower()
        pipe = TrendclipPipeline(trendclip_root(self.settings))

        def _pick_fn(prompt: str) -> Any:
            """The brain rung: chat_json wrapped to the organ's
            ``pick_fn(prompt) -> parsed JSON | None`` shape."""
            from ...llm.base import Message, SamplingParams
            from ...llm.brain import brain_for
            try:
                data, _response = brain_for(self.context).chat_json(
                    [Message.user(prompt)], task_kind="judge",
                    params=SamplingParams(temperature=0.2, max_tokens=800))
            except Exception as exc:  # noqa: BLE001 - a dead brain is a note
                _log.warning("trendclip brain pick failed: %s", exc)
                return None
            return data if isinstance(data, dict) else None

        def _copy_fn(prompt: str) -> Any:
            from ...llm.base import Message, SamplingParams
            from ...llm.brain import brain_for
            try:
                data, _response = brain_for(self.context).chat_json(
                    [Message.user(prompt)], task_kind="judge",
                    params=SamplingParams(temperature=0.2, max_tokens=500))
            except Exception as exc:  # noqa: BLE001
                _log.warning("trendclip brain copy failed: %s", exc)
                return None
            return data if isinstance(data, dict) else None

        if action == "status":
            if len(toks) > 1:
                try:
                    info = pipe.status(toks[1])
                except KeyError:
                    return f"no trendclip job {toks[1]!r}"
                lines = [f"🎬 trendclip {info['job_id']} — {info['stage']}",
                         f"topic: {info['topic']} · source: {info['source_url']}",
                         f"moments: {info['moments']} · brain picks: {info['brain_picks']}",
                         f"clip: {info['clip'] or 'not rendered yet'}",
                         f"validation: {'passed' if info['validation_passed'] else 'not passed'}"]
                if info["error"]:
                    lines.append(f"error: {info['error']}")
                return "\n".join(lines)
            jobs = pipe.list_jobs()
            if not jobs:
                return "no trendclip jobs yet — /clip <url> to start one"
            return "\n".join(
                f"{j['job_id']} — {j['stage']} · {j['topic']} · "
                f"{j['moments']} moments"
                for j in jobs[:10])

        if action == "render":
            if len(toks) < 3:
                return "usage: /clip render <job> <pick> [--layout NAME]"
            layout = "balanced"
            destinations = None
            i = 3
            while i < len(toks):
                if toks[i] == "--layout" and i + 1 < len(toks):
                    layout = toks[i + 1]
                    i += 2
                elif toks[i] == "--dest" and i + 1 < len(toks):
                    destinations = [d.strip() for d in toks[i + 1].split(",")
                                    if d.strip()]
                    i += 2
                else:
                    i += 1
            try:
                pick_n = int(toks[2])
            except ValueError:
                return f"pick must be a number, got {toks[2]!r}"
            pipe_with_brain = TrendclipPipeline(
                trendclip_root(self.settings), copy_fn=_copy_fn)
            try:
                job = pipe_with_brain.render(
                    toks[1], pick_n, layout=layout,
                    destinations=destinations)
            except (KeyError, RuntimeError) as exc:
                return f"render failed: {exc}"
            render_info = (job.receipt or {}).get("render") or {}
            validation = (job.receipt or {}).get("validation") or {}
            if job.stage == "failed":
                problems = "; ".join(validation.get("problems") or [])
                return (f"render failed: {job.error}"
                        + (f" — {problems}" if problems else ""))
            package = render_info.get("package") or {}
            return ("\n".join([
                f"🎬 clip ready — {render_info.get('clip_path')}",
                f"duration {render_info.get('duration')}s · "
                f"layout {render_info.get('layout')} · "
                f"validation {'passed' if validation.get('passed') else 'FAILED'}",
                f"package: {package.get('dir', '')}",
                "Review draft — third-party media. Manual upload only."]))

        if action == "queue":
            if len(toks) < 2:
                return "usage: /clip queue <job> [platforms...]"
            try:
                queued = pipe.queue(toks[1], platforms=toks[2:] or None)
            except (KeyError, RuntimeError) as exc:
                return f"queue failed: {exc}"
            return (f"🎬 queued for the publisher sweep — post "
                    f"{queued['post_id']} "
                    f"({', '.join(queued['platforms']) or 'no platforms named'}). "
                    "The sweep delivers the pre-rendered clip; nothing re-renders.")

        url = toks[0]
        if not url.startswith("http"):
            return ("usage: /clip <url> — give me a video URL to cut a clip from "
                    "(or /clip render|status|queue ...)")
        strategy = "conversation"
        use_brain = True
        i = 1
        while i < len(toks):
            if toks[i] == "--strategy" and i + 1 < len(toks):
                strategy = toks[i + 1]
                i += 2
            elif toks[i] == "--no-brain":
                use_brain = False
                i += 1
            else:
                i += 1
        topic = topic_from_query(url, strategy=strategy)
        pipe_with_brain = TrendclipPipeline(
            trendclip_root(self.settings),
            pick_fn=_pick_fn if use_brain else None)
        job = pipe_with_brain.analyze(url, topic, brain=use_brain)
        if job.stage == "failed":
            return f"analysis failed: {job.error or 'unknown error'}"
        receipt = job.receipt or {}
        brain_info = receipt.get("brain") or {}
        lines = [f"🎬 analyzed — job {job.job_id}",
                 f"source: {(receipt.get('source') or {}).get('title') or url}",
                 f"rights: {(receipt.get('rights') or {}).get('label') or ''}",
                 f"words: {(receipt.get('transcript') or {}).get('words', 0)} · "
                 f"strategy: {receipt.get('strategy')}",
                 "",
                 receipt.get("editorial_summary") or "no moments scored",
                 ""]
        if brain_info.get("ran") and brain_info.get("picks"):
            lines.append("brain picks:")
            for pick in brain_info["picks"]:
                lines.append(f"  #{pick['n']}: {pick['reason']}")
        elif brain_info.get("error"):
            lines.append(f"brain rung: {brain_info['error']}")
        lines.append("")
        lines.append(f"render one with: /clip render {job.job_id} <pick>")
        return "\n".join(lines)

    def _control_trend(self, tail: str) -> str:
        """What's hot right now: /trend <query> [--limit N]."""
        import shlex

        from ...media.trendclip import TrendclipPipeline, topic_from_query, trendclip_root

        ref = (tail or "").strip()
        if not ref:
            return "usage: /trend <query> [--limit N] — e.g. /trend GTA 6 gameplay"
        try:
            toks = shlex.split(ref)
        except ValueError:
            toks = ref.split()
        limit = 10
        query_parts: list[str] = []
        i = 0
        while i < len(toks):
            if toks[i] == "--limit" and i + 1 < len(toks):
                with contextlib.suppress(ValueError):
                    limit = max(1, min(20, int(toks[i + 1])))
                i += 2
            else:
                query_parts.append(toks[i])
                i += 1
        query = " ".join(query_parts).strip()
        if not query:
            return "usage: /trend <query> [--limit N]"
        topic = topic_from_query(query)
        items = TrendclipPipeline(
            trendclip_root(self.settings)).discover(topic, limit=limit)
        if items and "error" in items[0]:
            return f"trend search failed: {items[0]['error']}"
        if not items:
            return f"nothing trending found for {query!r}"
        lines = [f"🔥 trending — {query}:"]
        for idx, item in enumerate(items[:limit], 1):
            lines.append(
                f"{idx}. {item.get('title') or 'untitled'} "
                f"({item.get('channel') or '?'}) — "
                f"{item.get('views_per_hour', 0):,.0f} views/hr · "
                f"score {item.get('feed_score', 0)}\n"
                f"   {item.get('url') or ''}")
        lines.append("\ncut one with: /clip <url from the list>")
        return "\n".join(lines)

    def _control_trendops(self, tail: str, chat_key: str = "") -> Any:
        """Trend-driven video editing in chat.

        /trendops formats                                  — live trend formats
        /trendops apply <format> <footage> [--music <audio>] [--caption "text"]
                                                           — cut footage to a format
        /trendops apply <format>                           — arms a footage ask
        /trendops cancel                                   — drop an armed footage ask
        /trendops capture <url> [--name <slug>] [--niche X] — catch a trending clip
        /trendops propose                                  — peaking formats, ranked
        /trendops remove <slug>                            — drop a format
        """
        import shlex

        ref = (tail or "").strip()
        if not ref:
            return ("🎬 /trendops — trend-driven video editing:\n"
                    "/trendops formats — live trend formats\n"
                    "/trendops apply <format> <footage> [--music <audio>] "
                    "[--caption \"text\"]\n"
                    "/trendops capture <url> [--name <slug>] [--niche X]\n"
                    "/trendops propose — peaking formats, ranked\n"
                    "/trendops remove <slug>")
        try:
            toks = shlex.split(ref)
        except ValueError:
            toks = ref.split()
        if not toks:
            return ("usage: /trendops formats|apply|capture|propose|remove ...")
        action = toks[0].lower()
        rest = toks[1:]
        try:
            from ...media.trendops import TrendOps

            ops = TrendOps(context=self.context)
            if action == "formats":
                rows = ops.list_formats()
                if not rows:
                    return "no live trend formats — capture one: /trendops capture <url>"
                lines = []
                for r in rows:
                    left = (f"{r['days_left']:.0f}d left"
                            if r["days_left"] is not None
                            else "never expires")
                    lines.append(
                        f"• {r['slug']} ({r['niche'] or 'no niche'}) — "
                        f"{left}, used {r['uses']}x — "
                        f"{r['description'][:60]}")
                return "\n".join(lines)
            if action == "apply":
                if len(rest) < 1:
                    return ("usage: /trendops apply <format> <footage> "
                            "[--music <audio>] [--caption \"text\"]")
                fmt_name = rest[0]
                if len(rest) < 2:
                    # No footage on the line — the Apply button's
                    # shape.  Ask for it instead of usage text.
                    prompt = self._trendops_ask_for_footage(chat_key,
                                                            fmt_name)
                    if prompt is not None:
                        return prompt
                    return ("usage: /trendops apply <format> <footage> "
                            "[--music <audio>] [--caption \"text\"]")
                music = ""
                caption = ""
                beats: list[float] = []
                footage: list[str] = []
                args = rest[1:]
                i = 0
                while i < len(args):
                    a = args[i]
                    if a == "--music" and i + 1 < len(args):
                        music = args[i + 1]
                        i += 2
                    elif a == "--caption" and i + 1 < len(args):
                        caption = args[i + 1]
                        i += 2
                    elif a == "--beats" and i + 1 < len(args):
                        try:
                            beats = [float(b) for b in
                                     args[i + 1].split(",") if b.strip()]
                        except ValueError:
                            return ("--beats takes a comma list of seconds, "
                                    "e.g. --beats 0.5,1.0,1.5")
                        i += 2
                    elif a.startswith("--"):
                        i += 1
                    else:
                        footage.append(a)
                        i += 1
                if not footage:
                    prompt = self._trendops_ask_for_footage(chat_key,
                                                            fmt_name)
                    if prompt is not None:
                        return prompt
                    return "usage: /trendops apply <format> <footage> [...]"
                return self._trendops_apply(fmt_name, footage, music=music,
                                            caption=caption,
                                            beats=beats or None)
            if action == "cancel":
                # The Cancel button on the footage ask (and the typed
                # word, via _trendops_ask_answer) — one drop path.
                dropped = self._trendops_ask_drop(chat_key)
                if dropped is not None:
                    return dropped
                return "no trendops footage ask is armed in this chat."
            if action == "capture":
                if not rest:
                    return ("usage: /trendops capture <url> "
                            "[--name <slug>] [--niche X]")
                url = rest[0]
                name = ""
                niche = ""
                i = 1
                while i < len(rest):
                    if rest[i] == "--name" and i + 1 < len(rest):
                        name = rest[i + 1]
                        i += 2
                    elif rest[i] == "--niche" and i + 1 < len(rest):
                        niche = rest[i + 1]
                        i += 2
                    else:
                        i += 1
                got = ops.capture_trend(url, name=name or None, niche=niche)
                measured = got.get("measured", {})
                return (f"🎬 caught the trend → format {got['slug']}\n"
                        f"{measured.get('duration', 0):.1f}s · "
                        f"{measured.get('beat_count', 0)} beats measured · "
                        f"expires in {got['expires_in_days']:.0f}d\n"
                        f"cut it: /trendops apply {got['slug']} <footage>")
            if action == "propose":
                rows = ops.propose()
                if not rows:
                    return "no live trend formats — capture one: /trendops capture <url>"
                return "\n".join(
                    f"• {r['why']}\n  {r['apply']}" for r in rows)
            if action == "remove":
                if not rest:
                    return "usage: /trendops remove <slug>"
                if ops.remove_format(rest[0]):
                    return f"removed trend format {rest[0]}"
                return f"no trend format {rest[0]!r}"
            return f"unknown /trendops action {action!r}"
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"trendops failed: {exc}"

    def _shorts_hero(self, toks: list) -> str:
        """``/shorts hero "<prompt>" [--seconds N] [--seed N] [--model M]
        [--lora NAME[:W]]…`` — one neural hero clip via videogen.

        Returns the clip path, or the honest decline (reason +
        suggestion) verbatim — a hero shot is never faked with motion
        graphics under a neural label.
        """
        import re

        from ...media.videogen import request_hero_clip
        text = " ".join(toks)
        seconds = 5.0
        seed = 0
        model = None
        sm = re.search(r"--seconds\s+([\d.]+)", text)
        if sm:
            seconds = float(sm.group(1))
            text = text[:sm.start()] + text[sm.end():]
        nm = re.search(r"--seed\s+(\d+)", text)
        if nm:
            seed = int(nm.group(1))
            text = text[:nm.start()] + text[nm.end():]
        mm = re.search(r"--model\s+(\S+)", text)
        if mm:
            model = mm.group(1)
            text = text[:mm.start()] + text[mm.end():]
        loras = re.findall(r"--lora\s+(\S+)", text)
        text = re.sub(r"--lora\s+\S+", "", text)
        prompt = text.strip().strip("\"'").strip()
        if not prompt:
            return ('usage: /shorts hero "<prompt>" [--seconds N] '
                    '[--seed N] [--model NAME] [--lora NAME[:W]]')
        res = request_hero_clip(prompt, duration_s=seconds, seed=seed,
                                model=model, lora=loras or None)
        if res.get("ok"):
            extra = f" · model: {res['model']}" if res.get("model") else ""
            if res.get("loras"):
                extra += f" · loras: {', '.join(res['loras'])}"
            return f"🎬 hero clip → {res['path']} ({res['backend']}{extra})"
        out = f"🎬 hero clip unavailable: {res.get('reason', 'unknown')}"
        if res.get("suggestion"):
            out += f"\n   plan B: {res['suggestion']}"
        return out

    def _control_lens(self, tail: str) -> str:
        ref = (tail or "").strip()
        if not ref:
            return "usage: /lens <path-or-url>"
        outcome = self.context.tools.call("reverse_image_search", path=ref)
        data, error = self._tool_data(outcome)
        if data is None:
            return f"reverse search failed: {error}"
        lines = ["🔍 reverse image search:"]
        for name, link in (data.get("lookup_links") or {}).items():
            lines.append(f"  {name}: {link}")
        matches = data.get("unverified_matches") or []
        if matches:
            lines.append("possible related (unverified, from Bing):")
            lines.extend(f"  {m}" for m in matches[:5])
        if data.get("note"):
            lines.append(data["note"])
        return "\n".join(lines)

    def _control_gen(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split(None, 2)
        if len(parts) < 2:
            outcome = self.context.tools.call("script_kinds")
            kinds = outcome.value["kinds"] if outcome.ok else {}
            lines = ["usage: /gen <kind> <name> [json config]"]
            for kind, cfg in sorted(kinds.items()):
                lines.append(f"  {kind}: " + ", ".join(cfg))
            return "\n".join(lines)
        kind, name = parts[0], parts[1]
        config = parts[2].strip() if len(parts) > 2 else ""
        outcome = self.context.tools.call("script_gen", kind=kind, name=name, config=config)
        if not outcome.ok:
            return f"gen failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        return (f"✅ generated {v['path']} ({v['bytes']} B, validated {v['ext']})\n"
                f"{v['preview']}")

    def _control_apps(self, tail: str) -> str:
        """/apps [list|stacks|build <name> --stack …|info <name>]."""
        from ...builders.app_builder import STACKS, AppBuilder

        tail = (tail or "").strip()
        tokens = tail.split() if tail else []
        action = tokens[0].lower() if tokens and tokens[0].lower() in (
            "list", "stacks", "build", "info", "serve", "stop", "served",
            "deploy", "stop_deploy", "deployed") else "list"
        rest = tokens[1:] if action != "list" else tokens
        def _port_flag():
            if "--port" in rest:
                i = rest.index("--port")
                if i + 1 < len(rest):
                    try:
                        return int(rest[i + 1])
                    except ValueError:
                        return 0
            return 0
        try:
            b = AppBuilder(self.context)
            if action == "served":
                out = b.served()
                entries = out.get("served") or []
                if not entries:
                    return "no apps currently serving"
                return (f"serving ({out['count']}):\n" + "\n".join(
                    f"  {e['app']}  {e['url']}  (pid {e['pid']}, "
                    f"alive={e['alive']})" for e in entries))
            if action == "serve":
                if not rest:
                    return "usage: /apps serve <name> [--port N]"
                out = b.serve(rest[0], port=_port_flag())
                health = out.get("health") or {}
                text = (f"🖥 serving {out['app']} ({out['stack']}) → "
                        f"{out['url']}  (pid {out.get('pid')}, "
                        f"health={'ok' if health.get('ok') else health.get('error', 'unknown')})")
                if out.get("note"):
                    text += f"\n{out['note']}"
                return text
            if action == "stop":
                if not rest:
                    return "usage: /apps stop <name>"
                out = b.stop(rest[0])
                return (f"stopped {out['app']} (was pid {out['was_pid']}) "
                        f"→ {'stopped' if out['stopped'] else 'already dead'}")
            if action == "stacks":
                return "stacks: " + ", ".join(STACKS)
            if action == "info":
                if not rest:
                    return "usage: /apps info <name>"
                out = b.info(rest[0])
                return (f"{out.get('name')} ({out.get('stack')}) · "
                        f"{len(out.get('files', []))} files\nrun: {out.get('run')}")
            if action == "build":
                name = " ".join(
                    w for w in rest
                    if not w.startswith("--")).strip()
                def _flag(flag, default=""):
                    if flag in rest:
                        i = rest.index(flag)
                        if i + 1 < len(rest):
                            return rest[i + 1]
                    return default
                if not name or name.startswith("--"):
                    return ("usage: /apps build <name> --stack "
                            "static|flask|fastapi|express|react-vite|cli-python|"
                            "django|nextjs|bot-telegram|go-cli "
                            "[--title T] [--features a,b,c]")
                stack = _flag("--stack", "static")
                feats = [f.strip() for f in _flag("--features", "").split(",")
                         if f.strip()]
                out = b.build({"name": name, "stack": stack,
                               "title": _flag("--title", name),
                               "features": feats,
                               "overwrite": True})
                v = out["validation"]
                return (f"🏗 built {out['app']} ({out['stack']}) → {out['dir']}\n"
                        f"files: {len(out['files'])} · validation: "
                        f"{'OK' if v['ok'] else 'FAILED ' + str(v['failed'])}\n"
                        f"run: {out['run']}")
            if action == "deployed":
                out = b.deployed()
                entries = out.get("deployments") or []
                if not entries:
                    return "no apps currently deployed"
                return (f"deployed ({out['count']}):\n" + "\n".join(
                    f"  {e['app']}  {e['url']}  (proxy pid {e['pid']}, "
                    f"→ :{e.get('backend_port')}, alive={e['alive']})"
                    for e in entries))
            if action == "deploy":
                def _flag(flag, default=""):
                    if flag in rest:
                        i = rest.index(flag)
                        if i + 1 < len(rest):
                            return rest[i + 1]
                    return default
                name = " ".join(
                    w for w in rest
                    if not w.startswith("--")).strip()
                if not name:
                    return ("usage: /apps deploy <name> [--domain d] "
                            "[--path /x] [--port N]")
                out = b.deploy(name, host=_flag("--host", "0.0.0.0"),
                               port=int(_flag("--port", "0") or 0),
                               domain=_flag("--domain"),
                               path=_flag("--path"))
                health = out.get("health") or {}
                text = (f"🌐 deployed {out['app']} → {out['url']}\n"
                        f"  real reverse proxy (pid {out.get('pid')}) "
                        f"→ 127.0.0.1:{out.get('backend_port')}\n"
                        f"  domain: {out.get('domain') or '(none)'}   "
                        f"path: {out.get('path') or '/'}   "
                        f"health={'ok' if health.get('ok') else health.get('error', 'unknown')}")
                if out.get("note"):
                    text += f"\n{out['note']}"
                return text
            if action == "stop_deploy":
                if not rest:
                    return "usage: /apps stop_deploy <name>"
                out = b.stop_deploy(rest[0])
                return (f"stopped proxy for {out['app']} "
                        f"({'stopped' if out['stopped'] else 'already dead'}) "
                        f"was {out.get('url', '')}")
            out = b.list_apps()
            if not out["apps"]:
                return ("no apps yet — /apps build myapp --stack flask "
                        "(stacks: " + ", ".join(STACKS) + ")")
            return (f"apps ({out['count']}):\n" + "\n".join(
                f"  {a['name']}  [{a['stack']}]  →  {a['run']}"
                for a in out["apps"]))
        except Exception as exc:  # noqa: BLE001
            return f"apps error: {exc}"

    def _control_podcast(self, tail: str, chat_key: str = "") -> str:
        """/podcast <query…> [platform] — the podcast pipeline, one call."""
        return self._control_hub(f"podcast {tail}", chat_key=chat_key)

    def _control_record(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb == "start":
            if len(parts) < 2:
                return "usage: /record start <name>"
            outcome = self.context.tools.call("record_start", name=parts[1])
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            return (f"● recording {parts[1]} — every tool call you send me is captured. "
                    f"/record stop when done; it becomes /macro {parts[1]}.")
        if verb == "stop":
            outcome = self.context.tools.call("record_stop")
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            return (f"■ saved macro {v['steps']} steps → call it with "
                    f"/macro {v['saved']} (or ask me to run it)")
        if verb == "step":
            if len(parts) < 2:
                return "usage: /record step <tool> [json args]"
            args = " ".join(parts[2:]) if len(parts) > 2 else ""
            outcome = self.context.tools.call("record_step", tool=parts[1], args=args)
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            return f"step {outcome.value['steps']} added ({parts[1]})"
        if verb == "status":
            outcome = self.context.tools.call("record_status")
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            if not v.get("recording"):
                return "not recording — /record start <name>"
            return (f"● recording {v['name']}: {v['steps']} steps "
                    f"({v['seconds']}s, last: {v.get('last')})")
        return "usage: /record start <name> | stop | step <tool> [json] | status"

    def _control_publish(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        if len(parts) < 3:
            return "usage: /publish <platform> <chat_id> <markdown file> [format]"
        platform, chat_id, path = parts[0], parts[1], parts[2]
        fmt = parts[3].lower() if len(parts) > 3 else "pdf"
        try:
            from ...tools.filesystem import safe_path

            source = safe_path(self.context, path, must_exist=True)
        except Exception as exc:  # noqa: BLE001
            return f"publish failed: cannot read {path!r} ({exc})"
        content = source.read_text(encoding="utf-8", errors="replace")
        title = source.stem.replace("-", " ").replace("_", " ").strip() or "Report"
        outcome = self.context.tools.call(
            "report_publish", content=content, title=title, platform=platform,
            chat_id=chat_id, format=fmt, name=source.stem)
        if not outcome.ok:
            return (f"publish failed: {getattr(outcome.error, 'message', outcome.error)}")
        v = outcome.value
        size = v.get("bytes", 0)
        where = f" → {platform}:{chat_id}" if v.get("sent") else " (not sent)"
        return (f"📄 published {os.path.basename(v['path'])} "
                f"({size // 1024} KB, {v['format']}){where}")



def _looks_like_spotify_uri(text: str) -> bool:
    """True for real Spotify URIs/links (not a `spotify:<query>` force)."""
    t = (text or "").strip()
    if t.startswith("spotify:"):
        parts = t.split(":")
        return len(parts) == 3 and bool(parts[2])
    low = t.lower()
    return "open.spotify.com" in low or "play.spotify.com" in low


def _resolve_play_title(context: Any, title: str) -> str:
    """Resolve a bare song title to something playable.

    1. workspace scan — audio files whose filename matches the title
    2. Spotify search (only when linked) — top track's URI
    3. SoundCloud search (keyless) — top track's permalink URL
    4. YouTube search (needs yt-dlp) — top video's watch URL
    5. the raw title (the player will fail loudly, not silently)
    """
    import os
    import re
    from difflib import SequenceMatcher

    query = title.strip()
    if not query:
        return ""
    # 1. workspace scan
    try:
        settings = getattr(context, "settings", None)
        root = (str(getattr(settings, "workspace_dir", "")) or "").strip()
        if root and os.path.isdir(root):
            exts = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mid", ".midi")
            best: tuple[float, str] | None = None
            words = [w for w in re.findall(r"[a-z0-9]+", query.lower())
                     if len(w) > 2]
            for dirpath, _dirnames, filenames in os.walk(root):
                # skip noise dirs
                if "/." in dirpath or "__pycache__" in dirpath:
                    continue
                for fn in filenames:
                    if not fn.lower().endswith(exts):
                        continue
                    stem = os.path.splitext(fn)[0].lower()
                    score = 0.0
                    if words:
                        hits = sum(1 for w in words if w in stem)
                        score = hits / len(words)
                    else:
                        score = SequenceMatcher(
                            None, query.lower(), stem).ratio() * 0.9
                    if score >= 0.5 and (
                            best is None or score > best[0]):
                        best = (score, os.path.join(dirpath, fn))
            if best is not None:
                return best[1]
    except Exception as e:
        record_failure("runtime_media._resolve_play_title", e)
    # 2. Spotify search — only when OAuth is actually linked; a locked
    # vault or dead token just skips to the next source.
    try:
        from ...connectors.wiring import spotify_connector, spotify_linked
        if spotify_linked(context):
            sp = spotify_connector(context)
            results = sp.search(query, types=["track"], limit=1)
            items = ((results.get("tracks") or {}).get("items") or [])
            if items:
                top = items[0]
                uri = str(top.get("uri") or
                          f"spotify:track:{top.get('id', '')}")
                if uri and uri != "spotify:track:":
                    return uri
    except Exception as e:
        record_failure("runtime_media._resolve_play_title:3762", e)
    # 3. SoundCloud search (keyless, auto client_id)
    try:
        from ...accounts.vault import open_vault
        from ...connectors import create_connector
        vault = open_vault(getattr(context, "db", None), context=context,
                           actor="soundcloud-search")
        sc = create_connector("soundcloud", vault)
        tracks = sc.search_tracks(query, limit=3)
        for t in tracks or []:
            url = (t.get("permalink_url") or t.get("url") or "").strip()
            if url:
                return url
    except Exception as e:
        record_failure("runtime_media._resolve_play_title:3776", e)
    # 4. YouTube search (keyless via yt-dlp; skipped when not installed)
    try:
        from ...media.playback import PlaybackEngine
        video_id = PlaybackEngine._youtube_search_id(query)
        if video_id:
            return PlaybackEngine._youtube_watch_url(video_id)
    except Exception as e:
        record_failure("runtime_media._resolve_play_title:3784", e)
    # 5. give up honestly — the player reports the failure
    return query

