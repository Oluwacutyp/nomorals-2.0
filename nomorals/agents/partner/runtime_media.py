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
                text += f"\n sent {' + '.join(sent)} to this chat."
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
                        return f"can't play {rest[:80]!r} from YouTube: {exc}"
                    if chat is not None and added:
                        out = f"queued 1:\n  - {added[0].get('title')}"
                        return self._play_send_in_chat(engine, added[0], out, chat)
                    try:
                        res = engine.play_youtube(rest)
                    except Exception as exc:  # noqa: BLE001
                        return f"can't play {rest[:80]!r} from YouTube: {exc}"
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
                resolution = SourceResolver(self.context).resolve(search_q)
                if not resolution.ok:
                    tried = "; ".join(resolution.attempts) or "no strategies"
                    hint = (f" {resolution.hint}"
                            if resolution.hint else "")
                    return (f"couldn't find {rest[:80]!r} — "
                            f"tried: {tried}.{hint}")
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
                if action == "play" and chat is not None:
                    return self._play_send_in_chat(engine, added[0], out, chat)
                if action == "play":
                    st = engine.play()
                    if st.get("status") == "playing":
                        out += f"\n▶ playing “{st.get('current', '')}” " \
                               f"[{st.get('backend')}]"
                    else:
                        out += (f"\nstatus: {st.get('status')} "
                                f"{st.get('error') or st.get('hint', '')}")
                return out
            if action == "seek" and rest:
                return f"seeked to {rest.split()[0]}s: {engine.seek(float(rest.split()[0]))}"
            if action == "volume" and rest:
                return f"volume: {engine.volume(int(rest.split()[0]))}"
            if action == "remove" and rest:
                return f"removed: {engine.remove(int(rest.split()[0]))}"
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
        entry = PickCache.consume(token)
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
            return self._play_send_in_chat(engine, item, out, chat)
        try:
            st = engine.play()
            if st.get("status") == "playing":
                out += f"\n▶ playing “{st.get('current', '')}”"
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

    def _play_send_in_chat(self, engine: Any, item: dict[str, Any],
                           out: str, chat: Any) -> str:
        """Chat path for /play: download the track, send it as a file.

        Never plays locally — the user receives the audio in the
        conversation.  Honest on every failure path.  Never raises.
        """
        try:
            dl = engine.download(item)
        except Exception as exc:  # noqa: BLE001 - download() shouldn't raise, belt and braces
            return out + f"\ncouldn't get the audio file: {exc}"
        if not dl.get("ok"):
            return out + f"\ncouldn't send the audio: {dl.get('reason', 'unknown error')}"
        path = dl["path"]
        title = dl.get("title") or item.get("title", "audio")
        try:
            self.gateway.send_file(
                chat.platform, f"{chat.platform}:{chat.chat_id}",
                path, caption=f"🎵 {title}")
            return out + f"\n▶ sent “{title}” to this chat."
        except Exception as exc:  # noqa: BLE001
            return out + f"\nhad the file ({path}) but couldn't send it: {exc}"

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
        """
        from ...media.dj import DJ

        tail = (tail or "").strip()
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
        lines = []
        if found:
            lines.append(f"✅ cookies file found: {found}")
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
        /sham as the caption. Uses AudD music recognition; the result
        links straight into /play.
        """
        try:
            return self._sham_identify(message)
        except Exception as exc:  # noqa: BLE001 - never raises
            return f"couldn't identify that audio ({exc})"

    def _sham_identify(self, message: Any) -> str:
        """Core /sham logic. Returns the reply text. Never raises."""
        from ...connectors.registry import create_connector
        from ...accounts.vault import CredentialVault

        audio_path = self._sham_find_audio(message)
        if not audio_path:
            return (
                "nothing to identify — reply to a voice note or audio "
                "message with /sham, or send audio with /sham as the caption."
            )
        vault = CredentialVault(
            getattr(self.context, "db", None),
            master_passphrase=__import__("os").environ.get(
                "NM_VAULT_PASSPHRASE", ""))
        try:
            audd = create_connector("audd", vault)
        except Exception:
            return (
                "music recognition isn't wired up yet — it's one free API "
                "key away."
            )
        res = audd.recognize(audio_path)
        if not res.get("ok"):
            return str(res.get("reason") or "couldn't identify that audio")
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
                if getattr(m, "kind", "") == "audio" and getattr(m, "path", ""):
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
        """/video <query> [platform] | /video download <url> [audio] | platforms."""
        from ...media.video import _PLATFORM_SITES, VideoFinder

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /video <query> [platform]  |  "
                    "/video download <url> [audio]  |  /video platforms")
        words = tail.split()
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

    def _control_vision(self, tail: str, chat_key: str = "",
                        message: Any = None) -> str:
        """`/vision [question]` — analyze an image with the vision model.

        Attach an image to the command message, or pass a path/URL.
        The path/URL can come first or last: `/vision /tmp/img.png what's
        this?` and `/vision what's this? /tmp/img.png` both work.
        """
        def _looks_like_target(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return False
            if s.startswith(("http://", "https://", "/", "./", "~/")):
                return True
            # Windows paths and bare filenames with extensions.
            if len(s) > 3 and "." in s:
                low = s.lower()
                if any(low.endswith(ext) for ext in
                       (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
                        ".tiff", ".heic")):
                    return True
            return False

        tail = (tail or "").strip()
        target = ""
        question = tail
        if tail:
            tokens = tail.split()
            if _looks_like_target(tokens[0]):
                target = tokens[0]
                question = " ".join(tokens[1:]).strip()
            elif len(tokens) > 1 and _looks_like_target(tokens[-1]):
                target = tokens[-1]
                question = " ".join(tokens[:-1]).strip()

        # Attached image takes priority over a path argument.
        if message is not None:
            for media in getattr(message, "media", None) or []:
                if getattr(media, "kind", "") == "image":
                    target = getattr(media, "path", "")
                    break
        if not target:
            return ("usage: /vision [question] — attach an image or pass "
                    "a path/URL")

        try:
            outcome = self.context.tools.call(
                "vision_describe", path=target, prompt=question)
        except Exception as exc:  # noqa: BLE001
            return f"vision failed: {exc}"
        if not outcome.ok:
            return (f"vision failed: "
                    f"{getattr(outcome.error, 'message', outcome.error)}")
        value = outcome.value or {}
        description = value.get("description") or "(no description returned)"
        provider = value.get("provider") or "vision"
        text = f"👁 [{provider}]\n{description}"
        # Vision output is untrusted data — the tool already marks it.
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None:
            return self._send_long_checked(chat.platform, chat,
                                           text[:6000])
        return text[:6000]

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
            fname = f"img_{int(time.time())}.png"
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

        /shorts make <niche> "<topic>" [--now]     — queue or render now
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
                    "/shorts make <niche> \"<topic>\" [--now]\n"
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
                    return "usage: /shorts make <niche> \"<topic>\" [--now]"
                niche, topic = rest[0], " ".join(rest[1:])
                now = topic.endswith("--now")
                if now:
                    topic = topic[: -len("--now")].strip().strip("\"'")
                job = pipe.plan(niche, topic.strip().strip("\"'"))
                if now:
                    result = pipe.run(job)
                    if result.ok:
                        return (f"🎬 rendered {job.id}\n📁 {result.final_path}\n"
                                f"stages: " + ", ".join(
                                    f"{k}={v}ms" for k, v in
                                    (result.stages or {}).items()))
                    return f"🎬 render failed: {result.error}"
                return (f"🎬 queued {job.id} ({niche} — {topic[:60]})\n"
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

