"""RuntimeMediaMixin: PartnerRuntime command group (media)."""

from __future__ import annotations

import os
import re

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
            if default is None:
                known = ", ".join(sorted(registry.all())) or "(none)"
                return ("no voice registered yet — register one first:\n"
                        "/music voices add <id> <model.pth> <source>\n"
                        f"known voices: {known}")
            voice_id = default.voice_id
        try:
            res = make_full_song(req["topic"], style=req["style"],
                                 voice_id=voice_id, audience="private",
                                 context=self.context)
        except Exception as exc:  # noqa: BLE001 - both model errors
            return f"🎵 couldn't make the full song:\n{exc}"
        text = (f"🎵 “{res.title}” — full song [{voice_id}]\n{res.note}")
        chat = self._ref_from_key(chat_key) if chat_key else None
        if chat is not None and res.master_path:
            try:
                self.gateway.send_file(
                    chat.platform, f"{chat.platform}:{chat.chat_id}",
                    res.master_path,
                    caption=f"🎵 {res.title} — AI instrumental + AI vocals")
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
        | /music full <topic> [style] [voice_id] | /music voices."""
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
                   "next", "prev", "queue", "remove", "clear", "status"}
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
                resolution = SourceResolver(self.context).resolve(rest)
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
                chat = self._ref_from_key(chat_key) if chat_key else None
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
            res = producer.produce(tail)
            if res.get("ok"):
                src = "link" if "open.spotify.com" in tail else "description"
                ref_profile = res.get("profile")
                if ref_profile is not None:
                    store.record_production(ref_profile, source=src, ref=tail)
                else:  # pragma: no cover - safety net
                    store.record_production(describe_to_profile(tail),
                                            source=src, ref=tail)
            return _deliver(res, "🎛 produced:")
        except Exception as exc:  # noqa: BLE001
            return f"produce error: {exc}"

    def _control_like(self, tail: str) -> str:
        """/like [notes…] — the last production was good; learn from it."""
        from ...media.taste import load_taste
        store = load_taste(getattr(self, "context", None))
        return store.record_feedback(True, tail or "")

    def _control_dislike(self, tail: str) -> str:
        """/dislike [notes…] — the last production missed; learn from it."""
        from ...media.taste import load_taste
        store = load_taste(getattr(self, "context", None))
        return store.record_feedback(False, tail or "not feeling it")

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

