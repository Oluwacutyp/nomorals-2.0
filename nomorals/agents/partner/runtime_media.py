"""RuntimeMediaMixin: PartnerRuntime command group (media)."""

from __future__ import annotations

import os
import re
from typing import Any

from ...search.adaptive import adaptive_result_limit


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

    def _control_music_draft(self, tail: str, chat_key: str = "") -> str:
        """ /music draft <topic> [style] — the artist's notebook: a full
        song draft (lyrics + cadence + melody contour + emotional arc +
        artist notes). The draft is inspectable and revisable. """
        from ...media.song_draft import draft_song, render_notebook, save_draft
        from ...media.music import STYLES
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
        except Exception:  # noqa: BLE001
            pass
        nb = render_notebook(draft)
        # chat gets the notebook; keep it readable, not the whole wall
        lines = nb.splitlines()
        preview = "\n".join(lines[:40])
        if len(lines) > 40:
            preview += f"\n… ({len(lines) - 40} more lines — /music perform to hear it)"
        return (f"📝 draft: “{draft.title}”\n\n{preview}\n\n"
                f"revise: /music revise <note>  |  perform: /music perform")

    def _control_music_revise(self, tail: str, chat_key: str = "") -> str:
        """ /music revise <note> — revise the active draft in plain words. """
        from ...media.song_draft import revise_draft, render_notebook, save_draft, load_draft
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
        return f"📝 revised: “{new.title}”\n\n{preview}"

    def _control_music_perform(self, chat_key: str = "") -> str:
        """ /music perform — perform the active draft: beat + vocals shaped
        together from the draft's own melody contours. """
        from ...media.song_draft import load_draft
        from ...media.draft_perform import perform_draft
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
                    caption=f"🎵 {res.get('draft_title', '')} — performed from draft")
            except Exception:  # noqa: BLE001
                pass
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
                except ValueError:
                    pass
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
        from ...media.ace_step import (
            ACEModelUnavailable, make_bed, parse_bed_request)
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
                    caption=f"🎹 {res.title} — AI bed (vocals come next)")
                text += "\nsent the bed to this chat."
            except Exception:  # noqa: BLE001
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
        from ...media.vocals import (
            RVCVoiceRegistry, VocalModelUnavailable,
            make_full_song, parse_full_request)
        from ...media.music import STYLES

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
                    caption=f"🎵 {res.title} — AI instrumental + {vocal_txt}")
                text += "\nsent the master to this chat."
            except Exception:  # noqa: BLE001
                text += f"\nmaster: {res.master_path}"
        else:
            text += f"\nmaster: {res.master_path or '(render failed)'}"
        return text

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
        | /music draft <topic> [style] | /music revise <note>
        | /music perform | /music freestyle [seed] [bars N]."""
        from ...media.music import STYLES, MusicCreator

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /music <topic> [style]  |  /music styles  |  "
                    "/music song [slug]\nstyles: " + ", ".join(STYLES))
        words = tail.split()
        if words and words[0].lower() in ("sources", "enrich", "health", "registry-play"):
            # Legal music registry surface — licence always shown.
            from ...music_sources.chat import control_music_sources
            return control_music_sources(tail)
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
                        caption=f"🎵 {song.title} — listen")
                    sent.append("audio")
            except Exception:  # noqa: BLE001 - text fallback below
                pass
            try:
                if song.score_pdf_path:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        song.score_pdf_path,
                        caption=f"🎼 {song.title} — score (chords + lyrics)")
                    sent.append("score")
            except Exception:  # noqa: BLE001
                pass
            if sent:
                # The sent line must reflect whether vocals are in
                # the audio — never imply a full song when the vocal
                # track was skipped (vocals requested, none rendered).
                vocal_gap = ""
                if with_vocals and not (song.vocal_path or "").strip():
                    vocal_gap = (" (instrumental only — vocal track "
                                 "skipped)")
                text += (f"\n sent {' + '.join(sent)} to this chat."
                         f"{vocal_gap}")
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
        from ...media import distribute as _dist
        from ...finance.ledger import Ledger

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
        from ...media import distribute as _dist
        from ...finance.ledger import Ledger

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
        from ...media.cookies import (
            BOT_COOKIE_HELP, BOT_COOKIE_SHORT_HELP, is_bot_detection_error)
        msg = f"{prefix}: {exc}"
        if is_bot_detection_error(exc):
            msg += f"\n{BOT_COOKIE_SHORT_HELP if chat is not None else BOT_COOKIE_HELP}"
        return msg

    def _control_play(self, tail: str, chat_key: str = "") -> str:
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
        from ...media.playback import PlaybackEngine
        from ...connectors.wiring import wire_streaming_adapters

        # chat never wired these before (only `nm music` did) — share the
        # same wiring so Spotify/SoundCloud work from the DM too.
        try:
            wire_streaming_adapters(self.context)
        except Exception:  # noqa: BLE001 - wiring is best-effort
            pass

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
            return _registry_play.registry_play_text(
                _registry_play.play_via_registry(query, source_id=source_id))
        actions = {"add", "pause", "resume", "stop", "seek", "volume",
                   "next", "prev", "queue", "remove", "clear", "status",
                   "pick"}
        parts = tail.split(None, 1)
        first = parts[0].lower() if parts else ""
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
                        out = f"queued 1:\n  - {added[0].get('title')}"
                        return self._play_download_bg(
                            chat, chat_key, out,
                            lambda: self._play_send_in_chat_result(
                                engine, added[0], out, chat),
                            title=str(added[0].get("title") or ""))
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
                from ...media.resolver import SourceResolver
                from ...media.picklist import (
                    is_specific, parse_title_artist, search_query_for,
                    PickCache, format_picklist, PICK_LIMIT,
                )
                low_r = rest.lower()
                _is_urlish = low_r.startswith(("http://", "https://",
                                               "spotify:"))
                _is_pathish = ("/" in rest or "\\" in rest) and bool(
                    re.search(r"\.(mp3|wav|flac|ogg|m4a|mid|midi)$", rest, re.I))
                chat = self._ref_from_key(chat_key) if chat_key else None
                if (chat is not None and not _is_urlish and not _is_pathish
                        and not is_specific(rest)):
                    # Vague query in chat → old-school pick-list instead of
                    # auto-downloading the top guess.  Never raises.
                    try:
                        candidates = SourceResolver(
                            self.context).search_candidates(
                                rest, limit=PICK_LIMIT)
                    except Exception:  # noqa: BLE001
                        candidates = []
                    if candidates:
                        token = PickCache.store(
                            candidates, chat_key or "")
                        if token:
                            return format_picklist(
                                candidates, rest, token)
                    # No candidates (or cache hiccup) → fall through to
                    # the normal auto-resolve so the user still gets an
                    # honest tried-list instead of silence.
                title_q, artist_q = parse_title_artist(rest)
                search_q = search_query_for(title_q, artist_q) or rest
                # Step 4 Ekiti lexicon: search-normalize Ekiti → standard
                # Yoruba for source queries; the ORIGINAL stays in `rest`
                # for display/TTS.  Never raises.
                try:
                    from ...media.source_registry import EkitiLexicon
                    search_q = EkitiLexicon().normalize_query(
                        search_q)["search"] or search_q
                except Exception:  # noqa: BLE001 — lexicon is advisory
                    pass
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
                    except Exception:  # noqa: BLE001
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
                        from ...media.cookies import (
                            BOT_COOKIE_HELP as _FULL_HELP,
                            BOT_COOKIE_SHORT_HELP as _SHORT_HELP)
                        msg = msg.replace(_FULL_HELP, _SHORT_HELP)
                    return msg
                added = []
                for r in (resolution, *resolution.extra_tracks):
                    added.append(engine._enqueue(
                        r.path_or_url, r.kind, r.title or r.path_or_url,
                        artist=r.artist, duration=r.duration))
                queue_len = len(engine.queue())
                out = (f"queued {len(added)} (queue {queue_len}):\n"
                       + "\n".join(f"  - {a.get('title') or a['path']}"
                                   for a in added))
                # Chat context: SEND the audio file instead of local mpv.
                # Nobody hears the server's speakers — the user gets the track.
                # Text query, single track: download fall-through across
                # ALL candidates.  A search-ok source whose download
                # fails is a download failure, not success — keep
                # trying until one both finds AND downloads.  The
                # helper renders the full tried-list, so don't append
                # the resolver's partial list here.
                if (action == "play" and chat is not None
                        and not _is_urlish and not _is_pathish
                        and not resolution.extra_tracks):
                    try:
                        from ...media.resolver import SourceResolver
                        cands, detailed = SourceResolver(
                            self.context).search_candidates_detailed(
                                search_q)
                    except Exception:  # noqa: BLE001
                        cands, detailed = [], []
                    fallbacks = [
                        {"kind": c.kind, "path": c.path_or_url,
                         "title": c.title or c.path_or_url,
                         "artist": c.artist or "",
                         "duration": c.duration or 0.0,
                         "downloadable": c.downloadable}
                        for c in (cands or [])]
                    # The chain winner can carry the raw search query
                    # as its "title" (the YouTube search strategies do
                    # this) — the candidate sweep has the SAME track
                    # with its real metadata; prefer it, so nothing
                    # downstream ever displays or delivers the query
                    # text as if it were the track's name.
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
                    merged = self._merge_search_attempts(
                        resolution.attempts, detailed)
                    return self._play_download_bg(
                        chat, chat_key, out,
                        lambda: (self._play_send_with_fallthrough(
                            engine, first_item, fallbacks, merged, out,
                            chat, match=(title_q, artist_q), query=rest),
                                 False),
                        title=str(added[0].get("title") or "audio"))
                # Sources tried — visible, so the user sees MANY sources
                # were consulted, not 1–2.  Attempts already record
                # every strategy in order (failed + the winner).
                tried = [a for a in (resolution.attempts or [])]
                if tried:
                    out += ("\nsources tried "
                            f"({len(tried)}): " + "; ".join(tried))
                if action == "play" and chat is not None:
                    return self._play_download_bg(
                        chat, chat_key, out,
                        lambda: self._play_send_in_chat_result(
                            engine, added[0], out, chat),
                        title=str(added[0].get("title") or ""))
                if action == "play":
                    st = engine.play()
                    if st.get("status") == "playing":
                        out += f"\n▶ playing “{st.get('current', '')}” " \
                               f"[{st.get('backend')}]"
                    else:
                        out += (f"\nstatus: {st.get('status')} "
                                f"{st.get('error') or st.get('hint', '')}")
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

    def _play_pick(self, rest: str, chat_key: str = "") -> str:
        """`/play pick <token> <n>` — download+send pick-list choice n.

        The token comes from a Telegram inline button or the WhatsApp
        number-reply hook.  Single-use: the pick-list is consumed on a
        successful pick.  Never raises.
        """
        from ...media.playback import PlaybackEngine
        from ...media.picklist import PickCache

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
            return ("that pick-list expired (5 min) — run /play again and "
                    "pick fast 😄")
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
                chat, chat_key, out,
                lambda: self._play_send_in_chat_result(
                    engine, item, out, chat),
                on_sent=lambda: PickCache.consume(token),
                title=str(cand.title or ""))
        try:
            st = engine.play()
            if st.get("status") == "playing":
                out += f"\n▶ playing “{st.get('current', '')}”"
                PickCache.consume(token)
        except Exception:  # noqa: BLE001
            pass
        return out

    def _play_pick_hook(self, message: ChatMessage) -> str | None:
        """Bare number reply after a /play pick-list → download that track.

        Old-school WhatsApp-bot style: the pick-list says "reply with the
        number", the user replies "3", this resolves it through the same
        ``/play pick`` path as the Telegram buttons.  Returns a reply
        string when a pending pick-list and a bare number both match,
        else None.  Never raises.
        """
        from ...media.picklist import PickCache, PICK_LIMIT

        text = (message.text or "").strip()
        if not text.isdigit():
            return None
        n = int(text)
        if n < 1 or n > PICK_LIMIT:
            return None
        found = PickCache.get_for_chat(message.chat.key)
        if found is None:
            return None
        token, _entry = found
        try:
            return self._control_play(f"pick {token} {n}", message.chat.key)
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
        from ...connectors.wiring import (spotify_connector, spotify_linked,
                                          spotify_link_help)
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
                          chat: Any) -> tuple[str, bool]:
        """The /download download+send body — runs on the job worker
        thread in chat mode, synchronously on CLI.  Returns (reply
        text, sent): ``sent`` is True only when the file actually
        went out.  Never raises."""
        from ...core.jobs import report_progress
        from ...media.downloader import MediaDownloader

        downloader = MediaDownloader(self.context)
        report_progress(f"downloading {url[:80]}")
        report = downloader.download(url, audio_only=audio_only)
        if not report.ok:
            lines = [f"couldn't download {url[:80]}:"]
            for stage in report.stages:
                err = stage.error or "failed"
                lines.append(f"  {stage.stage}: {err[:120]}")
            if report.hint:
                lines.append(f"hint: {report.hint}")
            return "\n".join(lines), False

        path = report.path
        title = report.title or "download"
        emoji = "🎵" if audio_only else "🎬"
        if chat is not None:
            try:
                self.gateway.send_file(
                    chat.platform, f"{chat.platform}:{chat.chat_id}",
                    path, caption=f"{emoji} {title}"[:1024])
                return (f"{emoji} downloaded “{title}” "
                        f"({report.size_bytes // 1024}KB) — sent to this chat."), True
            except Exception as exc:  # noqa: BLE001
                return (f"downloaded “{title}” to {path} "
                        f"but couldn't send it: {exc}"), False
        return f"{emoji} downloaded: {path}", False

    def _control_download(self, tail: str, chat_key: str = "") -> str:
        """/download <url> [audio] — download any media URL and send it here.

        Like @Anyurldownloader_bot: paste a link, get the file.  YouTube,
        TikTok, Instagram, X/Twitter, SoundCloud, direct MP3/MP4 links —
        yt-dlp handles 1000+ sites as the universal extractor, with the
        proxy-lab and browser stages as fallback when direct fails.

        /download <url>          → best quality video (or whatever the URL is)
        /download <url> audio    → audio only (mp3)

        In chat the download runs as a background job (owner decision
        2026-10-19): the handler returns an ack immediately and the
        file is sent to the chat when the download lands.
        """
        tail = (tail or "").strip()
        if not tail:
            return ("usage: /download <url> [audio]\n"
                    "paste any media URL — YouTube, TikTok, Instagram, X, "
                    "SoundCloud, direct mp3/mp4 links, 1000+ sites via yt-dlp.")
        words = tail.split()
        if words and words[0].lower() == "cancel":
            return self._cancel_download_job(
                words[1] if len(words) > 1 else "", chat_key)
        audio_only = any(w.lower() in ("audio", "--audio", "-a") for w in words)
        url = " ".join(w for w in words
                       if w.lower() not in ("audio", "--audio", "-a")).strip()
        if not url:
            return "usage: /download <url> [audio]"
        if not url.lower().startswith(("http://", "https://")):
            return (f"that doesn't look like a URL: {url[:80]!r}\n"
                    "paste the full link starting with https://")

        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None:
            return self._play_download_bg(
                chat, chat_key,
                f"⏳ downloading {url[:60]} in the background — I'll "
                "send the file here when it's ready.",
                lambda: self._download_produce(url, audio_only, chat),
                title=url[:80])
        text, _sent = self._download_produce(url, audio_only, None)
        return text

    # ── background downloads (owner decision 2026-10-19) ────────────────
    #
    # Long downloads run as background jobs on the shared JobRegistry
    # and get SENT once done: the chat handler never blocks on a
    # multi-minute download again.  The job body is the EXISTING
    # synchronous download+send (``_play_send_in_chat_result`` /
    # ``_play_send_with_fallthrough`` / the /download body) — every
    # pass-#6 behavior (fall-through, honest wording, tried-list,
    # queue hygiene, hints) is preserved because the same code runs,
    # just on the job's worker thread instead of the chat pump thread.

    #: Label marking a job as a chat download; the _process harvester
    #: filters on it (+ the chat key) so outcomes surface per chat.
    _PLAY_DL_JOB_OWNER = "play"

    @staticmethod
    def _play_job_delta_text(out_prefix: str, full_text: str) -> str:
        """Follow-up text for a finished download job: the part of the
        synchronous reply past what the ack already showed.  The ack
        prints the queue line; the follow-up carries the outcome
        (tried-list, sent confirmation, or honest failure).  Defensive:
        when the body didn't start with the ack prefix, return the
        whole text rather than a mangled slice.  Never raises."""
        try:
            if out_prefix and full_text.startswith(out_prefix):
                delta = full_text[len(out_prefix):].lstrip("\n")
                if delta:
                    return delta
            return full_text
        except Exception:  # noqa: BLE001
            return full_text

    def _play_download_bg(self, chat: Any, chat_key: str,
                          out_prefix: str, produce: Any, *,
                          on_sent: Any = None, title: str = "") -> str:
        """Start a chat download+send as a background job; return the ack.

        ``produce()`` returns ``(full_text, sent)`` — the synchronous
        download+send body, run on the job's worker thread.  It sends
        the FILE itself; this wrapper then posts the body's outcome
        text (only the delta past ``out_prefix`` — the ack already
        showed the queue line) to the SAME chat.  ``on_sent`` runs on
        the worker thread only when the file actually went out (e.g.
        consuming a pick-list).  The result dict records ``delivered``
        so :meth:`_harvest_play_downloads` never double-reports an
        outcome the body already posted.

        If the registry itself refuses the job, fall back to running
        the body synchronously (today's behavior) rather than dropping
        the download.  Never raises.
        """
        from ...core.jobs import (
            current_job, report_progress, shared_registry)

        # Duplicate guard: the same chat asking for the same title
        # while it is already downloading gets the EXISTING job named
        # back — never a second download of the same file (and never
        # the file sent twice).
        try:
            for snap in shared_registry().list():
                labels = snap.get("labels") or {}
                if (snap.get("status") in ("pending", "running")
                        and labels.get("owner") == self._PLAY_DL_JOB_OWNER
                        and labels.get("chat") == (chat_key or "")
                        and labels.get("title") == (title or "")[:80]):
                    return (
                        out_prefix
                        + f"\n⏳ “{(title or 'that download')[:60]}” is "
                          "already downloading in the background "
                          f"[job {snap.get('id')}] — I won't start a "
                          "second copy. "
                          f"Send /download cancel {snap.get('id')} "
                          "to stop it.")
        except Exception:  # noqa: BLE001 - guard must never block a dl
            pass

        # The worker body needs its own job id (for the durable
        # record) before submit() returns it — a one-cell holder the
        # submitting thread fills in immediately after submit.
        holder: dict[str, str] = {}

        def _body() -> dict[str, Any]:
            try:
                report_progress("download started")
                text, sent = produce()
            except Exception as exc:  # noqa: BLE001 - failure is data
                self._play_dl_mark_finished(holder)
                return {"ok": False, "delivered": False, "sent": False,
                        "title": title, "text": "",
                        "error": f"{type(exc).__name__}: {exc}"[:300]}
            if sent and on_sent is not None:
                try:
                    on_sent()
                except Exception:  # noqa: BLE001 - bookkeeping only
                    pass
            final = self._play_job_delta_text(out_prefix, text)
            late = False
            job_now = current_job()
            if job_now is not None:
                with job_now.lock:
                    late = job_now.status == "presumed-dead"
            if late and final:
                final = ("⏳ that download finished late — the quiet "
                         "stretch was it still working, not a drop:\n"
                         + final)
            delivered = False
            if final:
                try:
                    self.gateway.send(chat.platform, chat, final)
                    delivered = True
                except Exception:  # noqa: BLE001 - harvest retries
                    delivered = False
            self._play_dl_mark_finished(holder)
            return {"ok": True, "delivered": delivered,
                    "sent": bool(sent), "title": title, "text": final,
                    "late": late}

        try:
            job = shared_registry().submit(
                _body, tool="play-download",
                labels={"owner": self._PLAY_DL_JOB_OWNER,
                        "chat": chat_key or "",
                        "title": (title or "")[:80]})
        except Exception:  # noqa: BLE001 - registry down: degrade to sync
            text, _sent = produce()
            return text
        holder["job_id"] = job.id
        self._play_dl_record_start(job.id, chat_key, title)
        return (out_prefix
                + "\n⏳ downloading in the background — I'll send it "
                  f"here when it's ready. [job {job.id}]")

    # ── durable record of chat downloads (restart honesty) ─────────
    #
    # The job registry is in-memory: a process restart mid-download
    # used to lose the job silently — the chat was never told.  Every
    # chat download now also writes a row (job id, chat, title, boot
    # id) to the DB at submit; the body flips it to "done" when it
    # finishes.  A row still "running" under a DIFFERENT boot id can
    # only mean the process that owned it died — the harvester says
    # so, once, instead of the download vanishing.

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
        except Exception:  # noqa: BLE001 - bookkeeping never breaks dl
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
        except Exception:  # noqa: BLE001 - bookkeeping never breaks dl
            pass

    def _play_dl_record_finish(self, job_id: str) -> None:
        db = self._play_dl_db()
        if db is None or not job_id:
            return
        try:
            db.execute(
                "UPDATE play_download_jobs SET status = 'done' "
                "WHERE job_id = ? AND status = 'running'", (job_id,))
        except Exception:  # noqa: BLE001 - bookkeeping never breaks dl
            pass

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

    def _cancel_download_job(self, job_id: str, chat_key: str) -> str:
        """/download cancel <job-id> — cooperative cancellation.

        Only the chat that owns the job may cancel it.  Cancellation
        is cooperative: the flag is set on the job and the download
        backends stop at their next progress checkpoint (yt-dlp hook,
        direct-HTTP chunk hook, CLI subprocess poll) — bytes actually
        stop; a job that already finished is reported as finished.
        """
        from ...core.jobs import shared_registry

        job_id = (job_id or "").strip()
        if not job_id:
            return ("usage: /download cancel <job-id> — the id is in "
                    "the ack I sent when the download started.")
        registry = shared_registry()
        job = registry.poll(job_id)
        if job is None:
            return (f"no download job {job_id} on record — it may "
                    "already have finished and been swept.")
        labels = job.labels or {}
        if labels.get("owner") != self._PLAY_DL_JOB_OWNER \
                or labels.get("chat") != (chat_key or ""):
            return "that job isn't a download belonging to this chat."
        if job.done():
            return (f"job {job_id} already finished ({job.status}) — "
                    "nothing left to cancel.")
        if registry.cancel(job_id):
            title = str(labels.get("title") or "download")
            return (f"🛑 cancellation requested for “{title[:60]}” "
                    f"[job {job_id}] — the download stops at its next "
                    "progress checkpoint. If it had already finished, "
                    "its result still arrives.")
        return f"job {job_id} finished before the cancellation landed."

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
        except Exception:  # noqa: BLE001 - harvest never breaks chat
            return
        target = f"{chat.platform}:{chat.chat_id}"
        for job in jobs:
            try:
                self._play_dl_record_finish(job.id)
                res = job.result if isinstance(job.result, dict) else {}
                if res.get("delivered"):
                    continue  # body already reported — never twice
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
                    # follow-up never landed: replay the body's text.
                    text = str(res.get("text") or "")
                    if not text:
                        text = (f"⚠️ the background download of "
                                f"“{title}” finished, but its report "
                                "never reached this chat.")
                    self.gateway.send(chat.platform, target, text)
            except Exception:  # noqa: BLE001 - one job never blocks rest
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
            except Exception:  # noqa: BLE001 - sweep never breaks chat
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
                except Exception:  # noqa: BLE001 - one row never blocks
                    continue

    def _play_send_in_chat(self, engine: Any, item: dict[str, Any],
                           out: str, chat: Any) -> str:
        """Chat path for /play: download the track, send it as a file.

        Never plays locally — the user receives the audio in the
        conversation.  Honest on every failure path.  Never raises.

        Wording matters: when the download fails there was nothing to
        send, so the failure is "couldn't download", never
        "couldn't send".  "Couldn't send" is reserved for the case
        where the file WAS downloaded but the send itself failed.
        """
        text, _sent = self._play_send_in_chat_result(engine, item, out, chat)
        return text

    def _play_send_in_chat_result(self, engine: Any, item: dict[str, Any],
                                  out: str, chat: Any) -> tuple[str, bool]:
        """Same as :meth:`_play_send_in_chat`, plus whether the file
        was actually sent — callers that consume state on success
        (e.g. the pick-list) need the signal, not a string search.
        ``sent`` is True only when send_file succeeded.  Never raises.
        """
        from ...core.jobs import report_progress
        from ...media.cookies import (
            BOT_COOKIE_SHORT_HELP, is_bot_detection_error)
        # Progress note: counts as a heartbeat when this runs as a
        # background job; a no-op on the synchronous path.
        report_progress(
            f"downloading {item.get('title') or 'audio'}")
        try:
            dl = engine.download(item)
        except Exception as exc:  # noqa: BLE001 - download() shouldn't raise, belt and braces
            # Undeliverable: drop the just-enqueued queue row so the
            # durable queue doesn't keep a broken item.
            self._drop_matching_queued(engine, item)
            return out + f"\ncouldn't download the audio file: {exc}", False
        if not dl.get("ok"):
            reason = str(dl.get("reason", "unknown error"))
            self._drop_matching_queued(engine, item)
            msg = out + f"\ncouldn't download the audio: {reason}"
            if is_bot_detection_error(reason):
                msg += f"\n{BOT_COOKIE_SHORT_HELP}"
            return msg, False
        path = dl["path"]
        title = dl.get("title") or item.get("title", "audio")
        send_path = self._delivery_path(
            path, str(title), str(item.get("artist", "") or ""))
        try:
            self.gateway.send_file(
                chat.platform, f"{chat.platform}:{chat.chat_id}",
                send_path, caption=f"🎵 {title}")
            return out + f"\n▶ sent “{title}” to this chat.", True
        except Exception as exc:  # noqa: BLE001
            return out + f"\nhad the file ({path}) but couldn't send it: {exc}", False

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
        except Exception:  # noqa: BLE001 - queue hygiene never breaks chat
            return None

    @classmethod
    def _drop_matching_queued(cls, engine: Any, item: dict[str, Any]) -> None:
        """Remove the just-enqueued queue row for *item*.  Best-effort."""
        try:
            idx = cls._queued_index_of(engine, item)
            if idx is not None:
                engine.remove(idx)
        except Exception:  # noqa: BLE001
            pass

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
        except Exception:  # noqa: BLE001 - identity staging never
            # breaks a delivery that would otherwise succeed
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
        except Exception:  # noqa: BLE001
            pass

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
        except Exception:  # noqa: BLE001 - audit trail is best-effort
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
        except Exception:  # noqa: BLE001
            return list(search_attempts or [])

    def _play_send_with_fallthrough(
            self, engine: Any, first_item: dict[str, Any],
            fallbacks: Any, search_attempts: Any, out: str,
            chat: Any, *, match: Any = None, query: str = "") -> str:
        """Chat path: try candidates in order until one DOWNLOADS.

        A source whose search succeeded but whose download failed is a
        *download* failure, not success — the loop keeps trying the
        next source until one both finds AND downloads, or all
        candidates are exhausted.  The reply lists EVERY source
        actually tried (search + download stage), says "couldn't
        download" (never "couldn't send") on total failure, and shows
        the short /cookies hint ONLY when YouTube was actually
        attempted AND bot-blocked.  Never raises.

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
        """
        from ...media.cookies import (
            BOT_COOKIE_SHORT_HELP, is_bot_detection_error)

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
        # ── delivery-contract partition ─────────────────────────────
        # Confident matches are attempted (best score first, chain
        # order breaking ties); everything else downloadable becomes
        # the pick-list if no match lands.  blocked_matches collects
        # the found-but-blocked facts for the final report.
        attempt_items = items
        pick_items: list[dict[str, Any]] = []
        blocked_matches: list[tuple[dict[str, Any], str]] = []
        gated = match is not None
        if gated:
            from ...media.picklist import (
                is_confident_match, match_confidence)

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
            confident.sort(key=lambda pair: -pair[0])
            attempt_items = [it for _score, it in confident]
        outcomes: dict[str, str] = {}
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
            if kind == "youtube" and youtube_already_attempted:
                # A previous download already ran the YouTube fallback
                # (e.g. SoundCloud DRM -> YouTube) — don't attempt
                # YouTube twice; record it as already attempted.
                outcomes[name] = \
                    "already attempted via fallback — blocked"
                continue
            if item.get("downloadable") is False:
                outcomes[name] = "not downloadable (protected streams)"
                if gated:
                    blocked_matches.append(
                        (item, "not downloadable (protected streams)"))
                continue
            # Progress note per candidate: counts as a heartbeat when
            # this runs as a background job; a no-op synchronously.
            from ...core.jobs import report_progress
            report_progress(
                f"trying {name} ({cand_no} of {len(attempt_items)})")
            try:
                dl = engine.download(item)
            except Exception as exc:  # noqa: BLE001
                dl = {"ok": False, "reason": f"download error: {exc}"}
            if dl.get("ok"):
                path = dl["path"]
                dl_title = dl.get("title") or item.get("title", "audio")
                outcomes[name] = "downloaded ✓"
                tried = self._format_tried_list(search_attempts, outcomes)
                suffix = (f"\nsources tried ({len(tried)}): "
                          + "; ".join(tried))
                note = dl.get("note")
                if note:
                    suffix += f"\n({note})"
                send_path = self._delivery_path(
                    path, str(dl_title),
                    str(item.get("artist", "") or ""))
                try:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        send_path, caption=f"🎵 {dl_title}")
                    # The queue must hold what was actually sent: if
                    # the winner differs from the first pick, swap
                    # the queued row so a later /play next doesn't
                    # play the failed first pick.
                    if (str(item.get("kind", "")), str(item.get("path", ""))) != \
                            (str(first_item.get("kind", "")),
                             str(first_item.get("path", ""))):
                        self._replace_queued_with(engine, first_item, item)
                    return (out + suffix
                            + f"\n▶ sent “{dl_title}” to this chat.")
                except Exception as exc:  # noqa: BLE001
                    return (out + suffix + f"\nhad the file ({path}) "
                            f"but couldn't send it: {exc}")
            reason = str(dl.get("reason", "unknown error"))
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
                        f"download blocked ({reason[:120]}) — YouTube "
                        "fallback also tried")
                continue
            if kind == "youtube" and is_bot_detection_error(reason):
                youtube_bot_blocked = True
            outcomes[name] = f"download blocked ({reason[:120]})"
        tried = self._format_tried_list(search_attempts, outcomes)
        # Total failure: the pre-enqueued first pick never downloaded —
        # drop its queue row like the single-send path does, so the
        # durable queue doesn't keep a broken item.
        self._drop_matching_queued(engine, first_item)
        if gated and pick_items:
            # Nothing that MATCHES could be downloaded, but other
            # results exist: they are the owner's to choose from —
            # never auto-sent as if they were the request.
            offer = self._play_picklist_offer(
                pick_items, blocked_matches, out, tried,
                query or title, chat,
                youtube_bot_blocked=youtube_bot_blocked)
            if offer:
                return offer
        msg = (out + f"\nsources tried ({len(tried)}): "
               + "; ".join(tried))
        msg += f"\ncouldn't download {title!r} from any source."
        if youtube_bot_blocked:
            msg += f"\n{BOT_COOKIE_SHORT_HELP}"
        return msg

    def _play_picklist_offer(
            self, pick_items: list[dict[str, Any]],
            blocked_matches: list[tuple[dict[str, Any], str]],
            out: str, tried: list[str], query: str, chat: Any,
            *, youtube_bot_blocked: bool = False) -> str:
        """The no-confident-download outcome: honest report + choice.

        The reply carries (1) the full tried-list, (2) every
        found-but-blocked match BY NAME with its block reason, and
        (3) the standard pick-list over the remaining downloadable
        results, stored in the PickCache so a number reply or a
        button tap downloads exactly the chosen candidate.  Returns
        "" when the pick-list can't be stored (caller falls back to
        the plain failure report).  Never raises.
        """
        try:
            from ...media.cookies import BOT_COOKIE_SHORT_HELP
            from ...media.picklist import (
                PICK_LIMIT, PickCache, format_picklist)
            from ...media.picklist import _KIND_LABELS
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
            except Exception:  # noqa: BLE001
                chat_key = ""
            token = PickCache.store(cands, chat_key) if chat_key else ""
            if not token:
                return ""
            lines = [out,
                     f"sources tried ({len(tried)}): " + "; ".join(tried)]
            if blocked_matches:
                for it, reason in blocked_matches:
                    kind = str(it.get("kind", "") or "")
                    label = _KIND_LABELS.get(kind, "") or kind
                    who = (f" — {it.get('artist')}"
                           if it.get("artist") else "")
                    lines.append(
                        f"⚠ found “{it.get('title', '')}”{who} on "
                        f"{label} — but couldn't download it "
                        f"({str(reason)[:120]}).")
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
        except Exception:  # noqa: BLE001 - the offer never breaks
            # the honest failure report the caller still has
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
        """
        from ...media.dj import DJ

        tail = (tail or "").strip()
        if tail.lower().startswith("mix"):
            from ...media.dj_mixdrop import control_dj_mix
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is None:
                return "run /dj mix from a chat — the mix drops there."
            gateway = self.gateway

            def _da(path: str, caption: str) -> None:
                gateway.send_file(chat.platform,
                                  f"{chat.platform}:{chat.chat_id}",
                                  path, caption=caption[:1024])

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
                                  path, caption=caption[:1024])

            def _dt(text: str) -> None:
                gateway.send(chat.platform,
                             f"{chat.platform}:{chat.chat_id}", text)
            return control_dj_live(tail[4:].strip(), chat_key=chat_key,
                                   deliver_audio=_da, deliver_text=_dt,
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
                    show["path"], caption="📻 Devon FM — full show")
            except Exception:  # noqa: BLE001
                pass
        return text

    def _control_produce(self, tail: str, chat_key: str = "") -> str:
        """ /produce — the Producer composes an ORIGINAL piece.

        * ``/produce`` (no args) → from your taste profile ("make me something")
        * ``/produce <spotify-url>`` → analyze the track's lane + record taste
        * ``/produce <vibe words>`` → compose from the description + record taste
        * ``/produce harder|faster|slower|…`` → iterate on the last production
        """
        from ...media.producer import Producer, describe_to_profile
        from ...media.taste import (load_taste, modify_profile,
                                    last_production_profile)

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
                        res["path"], caption=f"🎛 {res['title']} (original)")
                except Exception:  # noqa: BLE001
                    pass
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

    def _control_cookies(self, tail: str) -> str:
        """/cookies [check] — YouTube cookie file status and setup help."""
        from ...media.cookies import BOT_COOKIE_HELP, cookies_status
        st = cookies_status()
        found = st.get("found")
        valid = st.get("valid_youtube")
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
        else:
            lines.append("❌ no YouTube cookies file found.")
            lines.append("")
            lines.append(BOT_COOKIE_HELP)
        if (tail or "").strip().lower() == "check" and not found:
            lines.append("")
            lines.append("Checked: " + ", ".join(st.get("checked", [])))
        return "\n".join(lines)

    def _control_sham(self, tail: str, chat_key: str = "",
                      message: Any = None) -> str:
        """ /sham — identify the song in a replied-to voice note or audio.

        Reply to an audio/voice message with /sham, or send audio with
        /sham as the caption. Native-first: Devon's local fingerprint
        memory first, AudD for the global catalogue, honest acoustic
        description when neither can name it. The result links straight
        into /play.
        """
        try:
            return self._sham_identify(message)
        except Exception as exc:  # noqa: BLE001 - never raises
            return f"couldn't identify that audio ({exc})"

    def _sham_identify(self, message: Any) -> str:
        """Core /sham logic. Returns the reply text. Never raises.

        Native-first strategy chain:

        1. **Devon's own memory** — local fingerprint match against her
           own productions + the indexed library. Zero API, zero cost.
        2. **AudD** — the global commercial catalogue, only when a key
           is configured and the local memory drew a blank.
        3. **Honest ears** — when neither can name it, Devon says what
           she actually hears (tempo, key, loudness) instead of
           shrugging.
        """
        from ...audio.fingerprint import describe_audio, match_local_db
        from ...connectors.registry import create_connector
        from ...accounts.vault import CredentialVault

        audio_path = self._sham_find_audio(message)
        if not audio_path:
            return (
                "nothing to identify — reply to a voice note or audio "
                "message with /sham, or send audio with /sham as the caption."
            )
        local = match_local_db(audio_path)
        if local.get("ok"):
            artist = local.get("artist") or "unknown artist"
            title = local.get("title") or "unknown title"
            score = local.get("score", 0)
            return (f"🎵 I know this one — {artist} — {title}\n"
                    f"(recognized from my own library, "
                    f"confidence {score})\n"
                    f"/play {artist} {title}")
        heard = describe_audio(audio_path)
        vault = CredentialVault(
            getattr(self.context, "db", None),
            master_passphrase=__import__("os").environ.get(
                "NM_VAULT_PASSPHRASE", ""))
        try:
            audd = create_connector("audd", vault)
        except Exception:
            audd = None
        if audd is None:
            return (
                "couldn't name that track — music recognition isn't set "
                "up (it's one free AudD API key away), and it's not in "
                "my local library.\n"
                f"👂 but here's what I hear: {heard}")
        res = audd.recognize(audio_path)
        if not res.get("ok"):
            if res.get("needs_key"):
                # Surface the connector's own reason — it carries the
                # free-API setup guidance (dashboard URL, 300 free
                # requests). A bare "no key configured" strands the user.
                reason = str(res.get("reason") or "no AudD key configured")
                return (
                    f"couldn't name that track — {reason}\n"
                    f"it's not in my local library either.\n"
                    f"👂 but here's what I hear: {heard}")
            return (f"{res.get('reason') or 'could not identify that audio'}\n"
                    f"👂 what I hear: {heard}")
        artist = res.get("artist", "Unknown artist")
        title = res.get("title", "Unknown title")
        album = res.get("album", "")
        release = res.get("release_date", "")
        song_link = res.get("song_link", "")
        lines = [f"🎵 {artist} — {title}"]
        if album:
            detail = album
            if release:
                detail += f" ({release[:4]})"
            lines.append(f"💿 {detail}")
        spotify = res.get("spotify") or {}
        sp_url = ""
        if isinstance(spotify, dict):
            sp_url = str(
                (spotify.get("external_urls") or {}).get("spotify") or "")
        if song_link:
            lines.append(f"🔗 {song_link}")
        # One tap into /play: the resolver's strategy chain takes it.
        lines.append(f"/play {artist} {title}")
        return "\n".join(lines)

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
            # Replied-to audio (Telegram): download on demand.
            meta = getattr(message, "meta", None) or {}
            file_id = str(meta.get("replied_audio_file_id") or "")
            if file_id and self.gateway is not None:
                adapter = self.gateway.adapters.get("telegram")
                download = getattr(adapter, "download_file", None)
                if callable(download):
                    path = download(file_id, "sham.ogg")
                    if path:
                        return path
        except Exception:  # noqa: BLE001 - best-effort
            pass
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
                from ...social.chat.base import (
                    ChatKind, ChatMessage, ChatRef, MediaRef)
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
        except Exception:  # noqa: BLE001 - seed is a bonus, not the path
            pass
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
                except Exception:  # noqa: BLE001 - taste is a bonus
                    pass
            try:
                if res.get("path"):
                    from ...audio.fingerprint import index_own
                    index_own(res["path"], title=str(res.get("title", "")),
                              artist="Devon")
            except Exception:  # noqa: BLE001 - indexing is a bonus
                pass
            lines = [f"🎛 produced from your voice note: {res['title']} — "
                     f"{res['bpm']:g} BPM, {res['key']} {res['mode']}"]
            lines.append(res["notes"])
            text = "\n".join(lines)
            chat = self._ref_from_key(chat_key) if chat_key else None
            if chat is not None and res.get("path"):
                try:
                    self.gateway.send_file(
                        chat.platform, f"{chat.platform}:{chat.chat_id}",
                        res["path"], caption=f"🎛 {res['title']} (original)")
                except Exception:  # noqa: BLE001
                    pass
            return text
        except Exception as exc:  # noqa: BLE001 - never raises
            return f"couldn't produce: {exc}"

    def _produce_hook(self, message: Any) -> str | None:
        """NL music production for the owner DM (non-slash only).

        "make me something for the gym" → produce with a mood-derived
        profile.  "I like this" / "too slow" right after a production →
        taste feedback.  Returns a reply or None; never raises.
        """
        from ...media.taste import (detect_feedback, detect_produce_intent,
                                    load_taste, mood_hint)
        from ...media.producer import Producer, describe_to_profile

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
            except Exception:  # noqa: BLE001 - indexing is a bonus
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
                        res["path"], caption=f"🎛 {res['title']} (original)")
                except Exception:  # noqa: BLE001
                    pass
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            _log.exception("produce hook failed")
            return f"produce error: {exc}"

    def _control_video(self, tail: str) -> str:
        """/video <query> [platform] | /video download <url> [audio] | platforms.

        Devon Studio generation lives here too (additive — the search
        paths below are untouched):
        /video gen "<prompt>" [--backend auto|ltx|wan|motion] [--seconds N]
        /video motion lyric <audio> "<lyrics>" [--preset NAME]
        /video motion visual <audio> [--preset phonk]
        /video motion slideshow <img1,img2> [audio]
        /video motion trailer <clip1,clip2> [--title T]
        /video capability | /video list
        """
        from ...media.video import _PLATFORM_SITES, VideoFinder

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /video <query> [platform]  |  "
                    "/video download <url> [audio]  |  /video platforms\n"
                    "studio: /video gen \"<prompt>\" | /video motion … | "
                    "/video capability | /video list")
        words = tail.split()
        first = words[0].lower()
        # studio subcommands — "motion" only triggers on a known product
        # kind so searches like "/video motion graphics" still search
        _motion_kinds = ("lyric", "visual", "slideshow", "trailer")
        if first in ("gen", "generate", "capability", "list", "chain") or (
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

    def _control_video_studio(self, action: str, tail: str) -> str:
        """Devon Studio generation subcommands for /video. Never raises."""
        try:
            if action in ("gen", "generate"):
                return self._video_studio_gen(tail)
            if action == "motion":
                return self._video_studio_motion(tail)
            if action == "capability":
                from ...media.videogen import capability_report
                return capability_report()
            if action == "list":
                from ...media.motion_studio._core import read_ledger
                ledger = read_ledger()
                if not ledger:
                    return "no studio renders yet."
                lines = ["🎬 recent studio renders:"]
                for e in ledger[-10:]:
                    lines.append(f"  {e.get('kind', '?'):22} {e.get('path', '?')}")
                return "\n".join(lines)
            if action == "chain":
                return ("usage: /video chain needs a scenes JSON file — "
                        "use `nm video chain scenes.json` on the CLI")
            return f"unknown /video studio action {action!r}"
        except Exception as exc:  # noqa: BLE001 - chat surfaces as text
            return f"video studio error: {exc}"

    def _video_studio_gen(self, tail: str) -> str:
        import re
        from ...media.videogen import generate
        # /video gen "<prompt>" [--backend X] [--seconds N] [--seed N]
        m = re.match(r'"([^"]+)"\s*(.*)$', tail) or re.match(r"'([^']+)'\s*(.*)$", tail)
        if m:
            prompt, flags = m.group(1), m.group(2)
        else:
            parts = tail.split("--", 1)
            prompt, flags = parts[0].strip(), ("--" + parts[1] if len(parts) > 1 else "")
        if not prompt:
            return ('usage: /video gen "<prompt>" [--backend auto|ltx|wan|motion] '
                    '[--seconds N] [--seed N]')
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
        res = generate(prompt, backend=backend, duration_s=seconds, seed=seed)
        return res.message()

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
                    caption=f"🎬 captioned ({style} style)")
                return f"🎬 captioned video sent ({style} style)."
            except Exception:  # noqa: BLE001
                pass
        return f"🎬 captioned ({style}): {out_path or '(render failed)'}"

    _VISION_SUBCOMMANDS = (
        "describe", "read-text", "read", "locate", "compare", "analyze",
        "faces", "qr", "layout", "hash", "exif", "colors", "info",
        "capabilities",
    )

    def _control_vision(self, tail: str, chat_key: str = "",
                        message: Any = None) -> str:
        """`/vision [subcommand] [args]` — Devon's eyes in chat.

        Subcommands (image from attachment, path, or URL):
          describe <img> [question]  — vision-model description (default)
          read-text <img>            — transcription, native OCR first
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
        if message is not None:
            for media in getattr(message, "media", None) or []:
                if getattr(media, "kind", "") == "image":
                    attached = getattr(media, "path", "") or ""
                    break

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

            if sub == "read-text":
                targets = _pick_targets(1)
                if not targets:
                    return _fail("usage: /vision read-text <image> — attach "
                                 "an image or pass a path/URL")
                value = _call("vision_read_text", path=targets[0])
                method = value.get("method", "?")
                return _send(f"📝 [{method}]\n{value.get('text', '')}\n"
                             f"[{value.get('confidence_note', '')}]")

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

    def _control_image(self, tail: str) -> str:
        ref = (tail or "").strip()
        if not ref:
            return "usage: /image <path-or-url> (lookup) or /image <prompt> (generate)"
        if _looks_like_path_or_url(ref):
            return self._image_lookup(ref)
        return self._image_generate(ref)

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

    def _image_generate(self, prompt: str) -> str:
        """Generate an image from a text prompt via the generative backend."""
        try:
            from ...media_edit.generate import get_backend, GenerativeEditError
        except ImportError as exc:
            return f"image generation unavailable: {exc}"
        try:
            backend = get_backend()
        except GenerativeEditError as exc:
            return f"image generation unavailable: {exc}"
        try:
            images = backend.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - backend errors surface as text
            return f"image generation failed: {exc}"
        if not images:
            return "image generation failed: backend returned no images"
        # Save the first image to the workspace and report the path
        try:
            from pathlib import Path
            import time
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
        /imggen upscale <path> [--scale 2]
        /imggen checkpoints | /imggen dashboard <run>
        """
        ref = (tail or "").strip()
        if not ref:
            return ("imggen — Devon's own image studio:\n"
                    "/imggen <prompt> [--seed N] [--ar 16:9] "
                    "[--negative \"...\"]\n"
                    "/imggen upscale <path> [--scale 2]\n"
                    "/imggen checkpoints\n"
                    "/imggen dashboard <run>")
        parts = ref.split(None, 1)
        action = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        try:
            if action == "checkpoints":
                from ...media.imggen.pipeline import (
                    list_native_checkpoints)

                cks = list_native_checkpoints()
                if not cks:
                    return ("no native checkpoints yet — train one: "
                            "`nm imggen train --data <photo-folder>`")
                return "\n".join(
                    f"• {c['run']}: {c['path']} "
                    f"({c['bytes'] / 1e6:.1f} MB)" for c in cks)
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
                        except ValueError:
                            pass
                        path = " ".join(
                            bits[:i] + bits[i + 2:]).strip()
                        break
                paths = Studio().upscale(path, scale=scale)
                return f"🎨 upscaled ×{scale}: {paths[0]}"
            # Default: generation. Parse --flags inline.
            import shlex

            try:
                toks = shlex.split(rest or ref)
            except ValueError:
                toks = (rest or ref).split()
            kwargs: dict = {}
            prompt_toks: list[str] = []
            i = 0
            while i < len(toks):
                t = toks[i]
                if t == "--seed" and i + 1 < len(toks):
                    kwargs["seed"] = int(toks[i + 1]); i += 2
                elif t == "--ar" and i + 1 < len(toks):
                    from ...media.imggen.pipeline import (
                        resolve_aspect_ratio)

                    w, h = resolve_aspect_ratio(toks[i + 1])
                    kwargs["width"], kwargs["height"] = w, h; i += 2
                elif t == "--negative" and i + 1 < len(toks):
                    kwargs["negative_prompt"] = toks[i + 1]; i += 2
                elif t == "--steps" and i + 1 < len(toks):
                    kwargs["steps"] = int(toks[i + 1]); i += 2
                else:
                    prompt_toks.append(t); i += 1
            prompt = " ".join(prompt_toks).strip()
            if not prompt:
                return "usage: /imggen <prompt> [--seed N] [--ar 16:9]"
            from ...media.imggen.studio import Studio

            paths = Studio().generate(prompt, **kwargs)
            return f"🎨 generated: {prompt}\nsaved: {paths[0]}"
        except Exception as exc:  # noqa: BLE001 - chat never raises
            return f"imggen failed: {exc}"

    def _control_shorts(self, tail: str) -> str:
        """Devon's short-form content empire in chat.

        /shorts make <niche> "<topic>" [--now] [--style NAME] — queue or render now
        /shorts status [job_id]                    — job list / inspect
        /shorts resume <run_id>                    — resume a failed render
        /shorts niches                             — list niches
        /shorts calendar [--due]                   — content calendar
        /shorts ledger                             — what got posted where
        /shorts estimate <niche> "<topic>"         — time/cost estimate
        """
        import shlex

        ref = (tail or "").strip()
        if not ref:
            return ("🎬 /shorts — Devon's short-form content empire:\n"
                    "/shorts make <niche> \"<topic>\" [--now] [--style NAME]\n"
                    "/shorts niches — list content niches\n"
                    "/shorts status [job] · /shorts resume <run_id>\n"
                    "/shorts calendar [--due] · /shorts ledger\n"
                    "/shorts estimate <niche> \"<topic>\"")
        try:
            toks = shlex.split(ref)
        except ValueError:
            toks = ref.split()
        if not toks:
            return "usage: /shorts make|status|resume|niches|calendar|ledger|estimate ..."
        action = toks[0].lower()
        rest = toks[1:]
        try:
            from ...media.contentops.pipeline import ShortPipeline

            pipe = ShortPipeline()
            if action == "niches":
                from ...media.contentops.niches import list_niches, get_niche

                lines = ["🎬 niches:"]
                for name in list_niches():
                    plugin = get_niche(name)
                    lines.append(f"• {name} ({getattr(plugin, 'cadence', '?')}/day) — "
                                 f"{getattr(plugin, 'thesis', '')[:70]}")
                return "\n".join(lines)
            if action == "make":
                if len(rest) < 2:
                    return ("usage: /shorts make <niche> \"<topic>\" "
                            "[--now] [--style phonk|documentary|vlog|minimal]")
                niche = rest[0]
                toks = rest[1:]
                now = "--now" in toks
                style = ""
                if "--style" in toks:
                    si = toks.index("--style")
                    if si + 1 < len(toks):
                        style = toks[si + 1]
                        del toks[si:si + 2]
                toks = [t for t in toks if t != "--now"]
                topic = " ".join(toks).strip("\"'").strip()
                job = pipe.plan(niche, topic, style=style)
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
        from ...builders import AppBuilder, STACKS

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
    except Exception:  # noqa: BLE001 - scan is best-effort
        pass
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
    except Exception:  # noqa: BLE001 - search is best-effort
        pass
    # 3. SoundCloud search (keyless, auto client_id)
    try:
        from ...connectors import create_connector
        from ...accounts.vault import CredentialVault
        vault = CredentialVault(
            getattr(context, "db", None),
            master_passphrase=os.environ.get("NM_VAULT_PASSPHRASE", ""))
        sc = create_connector("soundcloud", vault)
        tracks = sc.search_tracks(query, limit=3)
        for t in tracks or []:
            url = (t.get("permalink_url") or t.get("url") or "").strip()
            if url:
                return url
    except Exception:  # noqa: BLE001 - search is best-effort
        pass
    # 4. YouTube search (keyless via yt-dlp; skipped when not installed)
    try:
        from ...media.playback import PlaybackEngine
        video_id = PlaybackEngine._youtube_search_id(query)
        if video_id:
            return PlaybackEngine._youtube_watch_url(video_id)
    except Exception:  # noqa: BLE001 - search is best-effort
        pass
    # 5. give up honestly — the player reports the failure
    return query

