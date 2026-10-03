"""The Media system — playback, music creation, and video finding.

Three engines, one façade:

* :class:`.playback.PlaybackEngine` — durable queue + detached-player
  transport (mpv IPC when available; graceful console fallback).
* :class:`.music.MusicCreator` — style-aware lyrics + structure + chord
  plan + real playable MIDI (pure-Python SMF writer).
* :class:`.video.VideoFinder` — cross-web video search, ranked + enriched
  (oEmbed/yt-dlp), with download through the media pipeline.

    from nomorals.media import MediaHub
    m = MediaHub(context)
    m.add("workspace/song.mp3"); m.play(); m.volume(60); m.pause()
    song = m.make_song("city lights at 2am", style="afrobeats")
    hits = m.find_video("amapiano tutorials", max_results=5)

Every capability is also registered as its own tool (``player``,
``music_writer``, ``video_finder``) so the main AI and sub-agents can call
them directly.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from ..core.errors import ToolError
from ..core.policy import Capability
from .music import STYLES, MusicCreator, Song, StyleSpec
from .playback import Backend, PlaybackEngine, detect_backend
from .video import VideoFinder

__all__ = [
    "MediaHub", "MusicCreator", "Song", "StyleSpec", "STYLES",
    "PlaybackEngine", "Backend", "detect_backend",
    "VideoFinder",
]


class MediaHub:
    """One object over the whole media stack."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.playback = PlaybackEngine(context)
        self.music = MusicCreator(context)
        self.video = VideoFinder(context)

    # ── playback ──────────────────────────────────────────────────────────
    def add(self, *targets: str, title: str = "") -> dict[str, Any]:
        return self.playback.add(*targets, title=title)

    def play(self, index: int | None = None) -> dict[str, Any]:
        return self.playback.play(index)

    def pause(self) -> dict[str, Any]:
        return self.playback.pause()

    def resume(self) -> dict[str, Any]:
        return self.playback.resume()

    def stop(self) -> dict[str, Any]:
        return self.playback.stop()

    def seek(self, seconds: float) -> dict[str, Any]:
        return self.playback.seek(seconds)

    def volume(self, level: int | float) -> dict[str, Any]:
        return self.playback.volume(level)

    def next(self) -> dict[str, Any]:
        return self.playback.next()

    def prev(self) -> dict[str, Any]:
        return self.playback.prev()

    def queue(self) -> list[dict[str, Any]]:
        return self.playback.queue()

    def remove(self, index: int) -> dict[str, Any]:
        return self.playback.remove(index)

    def clear(self) -> dict[str, Any]:
        return self.playback.clear()

    def status(self) -> dict[str, Any]:
        return self.playback.status()

    # ── music creation ────────────────────────────────────────────────────
    def make_song(self, topic: str, *, style: str = "pop", title: str = "",
                  key: str = "", seed: int | None = None,
                  with_midi: bool = True) -> Song:
        return self.music.compose(topic, style=style, title=title, key=key,
                                  seed=seed, with_midi=with_midi)

    def styles(self) -> dict[str, dict[str, Any]]:
        return {k: {"label": v.label, "tempo": list(v.tempo), "mode": v.mode,
                    "energy": v.energy} for k, v in STYLES.items()}

    # ── video finding ─────────────────────────────────────────────────────
    def find_video(self, query: str, *, max_results: int = 10,
                   platform: str = "", freshness: str = "") -> dict[str, Any]:
        return self.video.find(query, max_results=max_results,
                               platform=platform, freshness=freshness)

    def download_media(self, url: str, *, audio_only: bool = False) -> dict[str, Any]:
        return self.video.download(url, audio_only=audio_only)

    # ── the run() orchestrator: one call, whole pipeline ──────────────────
    def run(self, mode: str = "song", *, topic: str = "", query: str = "",
            style: str = "pop", platform: str = "", seed: int | None = None,
            play: bool = True, send_transcript: bool | None = None,
            send_to: tuple[str, str] = ("", ""),
            max_results: int = 5) -> dict[str, Any]:
        """Compose-or-find → (download → transcribe) → queue → play.

        mode:
          * ``song``    — compose a song on ``topic``, queue its MIDI, play
          * ``video``   — find a video for ``query``, download it, queue, play
          * ``podcast`` — find, download audio, transcribe (STT when a
            backend exists), summarize into chapters, save transcript, and
            — by default (send_transcript=None) — deliver the transcript
            file to your newest live chat on any connected platform
            (send_transcript=False keeps it local)
        """
        mode = (mode or "song").strip().lower()
        want = topic or query
        if not want.strip():
            raise ToolError("media_hub.run needs topic= (song) or query= "
                            "(video/podcast)")
        if mode == "song":
            song = self.make_song(want, style=style, seed=seed)
            out: dict[str, Any] = {"mode": "song", "song": song.to_dict()}
            target = song.midi_path
            if target and play:
                try:
                    self.playback.add(target)
                    out["playback"] = self.playback.play()
                except ToolError as exc:
                    out["playback"] = {"status": "error", "error": str(exc)}
            elif play:
                out["playback"] = {"status": "skipped",
                                   "reason": "no MIDI file was produced"}
            return out
        if mode in ("video", "podcast"):
            found = self.find_video(query or want, max_results=max_results,
                                    platform=platform)
            results = found.get("results") or []
            if not results:
                return {"mode": mode, "found": found,
                        "error": "no video results — pipeline stopped"}
            pick = results[0]
            out = {"mode": mode, "pick": pick, "found_count": found.get("count")}
            dl = self.download_media(pick["url"],
                                     audio_only=(mode == "podcast"))
            out["download"] = dl
            path = dl.get("path", "")
            if mode == "podcast":
                out.update(self._podcast_transcribe(path, query or want,
                                                    send_transcript=
                                                    send_transcript,
                                                    send_to=send_to))
                return out
            if path and play:
                try:
                    self.playback.add(path)
                    out["playback"] = self.playback.play()
                except ToolError as exc:
                    out["playback"] = {"status": "error", "error": str(exc)}
            return out
        raise ToolError(f"unknown media_hub mode {mode!r} "
                        "(song|video|podcast)")

    # ── transcript delivery: newest live chat on an online platform ──────
    def auto_send_target(self) -> tuple[str, str]:
        """(platform, chat_id) of the best live delivery target.

        Walks the chat gateway's status, keeps the CONNECTED platforms,
        then picks the most recently active chat on one of them from the
        chat registry (owner chats first).  Returns ("", "") when
        nothing is online — callers treat that as "no target"."""
        gw = getattr(self.context, "gateway", None)
        if gw is None:
            return "", ""
        try:
            st = gw.status()
        except Exception:  # noqa: BLE001
            return "", ""
        online = [n for n, h in st.items()
                  if n != "_stats" and isinstance(h, dict)
                  and h.get("connected")]
        if not online:
            return "", ""
        db = getattr(self.context, "db", None)
        if db is not None:
            try:
                marks = ",".join("?" * len(online))
                row = db.query_one(
                    f"SELECT platform, chat_id FROM chats "
                    f"WHERE platform IN ({marks}) "
                    f"ORDER BY is_owner DESC, last_active DESC LIMIT 1",
                    tuple(online))
                if row:
                    return (row["platform"], row["chat_id"])
            except Exception:  # noqa: BLE001
                pass
        return online[0], ""

    # ── podcast: download → transcribe → summarize → chapters ────────────
    def podcast(self, query: str, *, platform: str = "",
                max_results: int = 5, send_transcript: bool | None = None,
                send_to: tuple[str, str] = ("", ""),
                play: bool = False) -> dict[str, Any]:
        return self.run("podcast", query=query, platform=platform,
                        max_results=max_results,
                        send_transcript=send_transcript, send_to=send_to,
                        play=play)

    def _podcast_transcribe(self, audio_path: str, title: str, *,
                            send_transcript: bool | None = None,
                            send_to: tuple[str, str] = ("", ""),
                            ) -> dict[str, Any]:
        out: dict[str, Any] = {"transcript": None, "chapters": [],
                               "summary": "", "transcript_path": ""}
        if not audio_path:
            out["error"] = "no audio file to transcribe"
            return out
        try:
            from ..tools.audio import stt as _stt

            audio_s = getattr(getattr(self.context, "settings", None),
                              "audio", None)
            llm_s = getattr(getattr(self.context, "settings", None),
                            "llm", None)
            base = (getattr(audio_s, "stt_base_url", "")
                    or os.environ.get("NM_AUDIO_STT_BASE_URL", "")
                    or getattr(llm_s, "base_url", "") or "")
            key = (getattr(audio_s, "stt_api_key", "")
                   or os.environ.get("NM_AUDIO_STT_API_KEY", "")
                   or getattr(llm_s, "api_key", "") or "")
            model = getattr(audio_s, "stt_model", "") or "whisper-1"
            res = _stt(audio_path, base_url=base, api_key=key, model=model)
            text = (res.get("text") or "").strip()
            out["transcript"] = text
            out["stt_provider"] = res.get("provider", "")
        except ToolError as exc:
            out["stt_error"] = str(exc)
            out["note"] = ("transcription needs an STT backend — "
                           "NM_AUDIO_STT_BASE_URL + NM_AUDIO_STT_API_KEY "
                           "(OpenAI-compatible) or whisper.cpp; the audio "
                           "file is saved for later")
            return out
        except Exception as exc:  # noqa: BLE001
            out["stt_error"] = str(exc)
            return out
        if not text:
            out["note"] = "transcript came back empty"
            return out

        # save transcript + chapters
        try:
            from ..tools.filesystem import safe_path
            import re

            slug = re.sub(r"[^a-z0-9]+", "-", (title or "podcast").lower())
            slug = slug.strip("-")[:48] or "podcast"
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = safe_path(self.context, f"podcasts/{slug}-{stamp}.txt",
                             must_exist=False)
            base.parent.mkdir(parents=True, exist_ok=True)
            summary = _summarize(text,
                                  router=getattr(self.context, "router", None))
            chapters = _chapters(text, max_chapters=8)
            blob = [f"Podcast: {title}",
                    f"source: {audio_path}",
                    f"summary: {summary}", "", "chapters:"]
            for ch in chapters:
                blob.append(f"  [{ch['start']}-{ch['end']}] {ch['title']}")
            blob += ["", "transcript:", text]
            base.write_text("\n".join(blob), encoding="utf-8")
            out["transcript_path"] = str(base)
            out["chapters"] = chapters
            out["summary"] = summary
        except Exception as exc:  # noqa: BLE001
            out["save_error"] = str(exc)
        # deliver: explicit ask, or AUTO — newest live chat on any
        # connected platform (the default once a platform is online)
        want_send = send_transcript is not False
        if want_send and out.get("transcript_path"):
            gateway = getattr(self.context, "gateway", None)
            plat, chat = send_to
            if not plat:
                plat, chat = self.auto_send_target()
            if gateway is None or not plat:
                out["send"] = (("transcript saved (no platform online — "
                                "auto-send skipped)")
                               if send_transcript is None else
                               ("transcript saved but no send target — "
                                "pass send_to=(platform, chat)"))
                return out
            try:
                res = gateway.send_file(plat, chat, out["transcript_path"],
                                        caption=f"podcast transcript: {title}")
                out["send"] = {"ok": bool(res.ok),
                               "platform": plat, "chat": chat,
                               "error": getattr(res, "error", "")}
            except Exception as exc:  # noqa: BLE001
                out["send_error"] = str(exc)
        return out


def _summarize(text: str, router: Any = None,
               max_sentences: int = 8) -> str:
    """Model summarization when a router is available; extractive otherwise."""
    sentences = [s_.strip() for s_ in re.split(r"(?<=[.!?])\s+", text)
                 if len(s_.strip()) > 20]
    if not sentences:
        return text[:300]
    if router is not None:
        try:
            from ..llm.base import Message, SamplingParams

            resp = router.chat(
                [Message.system(
                    "Summarize this transcript in at most 8 tight sentences. "
                    "Plain text only, no preamble."),
                 Message.user(text[:24000])],
                SamplingParams(temperature=0.2, max_tokens=700))
            if resp.ok and (resp.text or "").strip():
                return resp.text.strip()
        except Exception:  # noqa: BLE001 — fall through to extractive
            pass
    freq: dict[str, int] = {}
    for s_ in sentences:
        for w in re.findall(r"[a-z']{4,}", s_.lower()):
            freq[w] = freq.get(w, 0) + 1
    scored = []
    for i, s_ in enumerate(sentences):
        words = re.findall(r"[a-z']{4,}", s_.lower())
        score = sum(freq.get(w, 0) for w in words) / max(1, len(words))
        scored.append((score, i, s_))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked = sorted(t[1] for t in scored[:max_sentences])
    return " ".join(sentences[i] for i in picked)


def _chapters(text: str, max_chapters: int = 8) -> list[dict[str, Any]]:
    """Split a transcript into titled chapters (~2-4 min each at 150 wpm)."""
    words = text.split()
    if not words:
        return []
    per = max(1, len(words) // max_chapters)
    minutes_total = max(1, round(len(words) / 2.5))  # ~150 wpm
    chapters = []
    for i in range(0, len(words), per):
        chunk = words[i:i + per]
        start_min = round(i / 2.5)
        end_min = round((i + len(chunk)) / 2.5)
        title_words = " ".join(chunk[:8]).strip(" .,;:")
        chapters.append({
            "title": title_words[:60] or f"part {len(chapters) + 1}",
            "start": f"{start_min}m",
            "end": f"{min(end_min, minutes_total)}m",
        })
    return chapters[:max_chapters + 1]


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "media_hub",
        description=(
            "The Media system's one-call orchestrator. action=run (mode="
            "song|video|podcast, topic/query, style, platform, play) | "
            "podcast (query, platform) | status. song: compose→queue→play. "
            "video: find→download→queue→play. podcast: find→download audio→"
            "transcribe (STT when a backend is configured)→summarize→chapters"
            "+transcript saved under podcasts/ and — by default — sent to "
            "your newest live chat on any connected platform "
            "(send_transcript=no keeps it local; send_platform/send_chat "
            "force the target)."
        ),
        capability=Capability.FS_WRITE,
    )
    def media_hub(action: str = "run", mode: str = "song", topic: str = "",
                  query: str = "", style: str = "pop", platform: str = "",
                  seed: int = 0, play: bool = True,
                  send_transcript: str = "auto", send_platform: str = "",
                  send_chat: str = "", max_results: int = 5) -> dict[str, Any]:
        hub = MediaHub(context)
        if action == "status":
            return hub.status()
        send_to = (send_platform, send_chat)
        t = str(send_transcript or "auto").strip().lower()
        st = None if t in ("", "auto") else (t not in ("no", "false",
                                                       "0", "off"))
        if action == "podcast":
            return hub.podcast(query or topic, platform=platform,
                               max_results=max_results,
                               send_transcript=st, send_to=send_to)
        if action != "run":
            raise ToolError(f"unknown media_hub action {action!r}")
        return hub.run(mode, topic=topic, query=query, style=style,
                       platform=platform, seed=seed or None, play=play,
                       send_transcript=st, send_to=send_to,
                       max_results=max_results)
