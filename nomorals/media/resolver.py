"""Dynamic audio source resolver — a strategy chain, not hardcoded branches.

The old ``/play`` path grew hardcoded ``if "soundcloud.com" in url``
branches: when one service's API had a bad day, that whole source died
instead of falling back.  This resolver treats every input the same way —
a chain of strategies tried in order, each recording what happened:

1. **local file** — the query is an existing path (workspace-relative or
   absolute) → used directly, zero network involved.
2. **yt-dlp (universal)** — for ANY URL, yt-dlp's extractors route
   dynamically (SoundCloud, YouTube, Bandcamp, Audiomack, 1000+ sites).
   No per-domain code anywhere in the main path.
3. **SoundCloud API** — an *optimization* for SoundCloud URLs: fast
   metadata without extraction.  When the api-v2 has issues it is NEVER
   fatal — the chain falls back to yt-dlp for the same URL and logs it.
   (This was the bug: an API hiccup used to kill SoundCloud playback
   outright instead of degrading.)
4. **text search** — a plain title goes workspace scan → SoundCloud
   search (keyless) → YouTube search (``ytsearch1:`` via yt-dlp) →
   Spotify search (only when linked; device playback, no download).

``resolve()`` never raises.  It returns a ``ResolvedAudio`` carrying
``ok``, ``path_or_url``, ``title``, ``attempts`` (everything tried, in
order), ``errors`` and a ``hint`` — so callers report *what was tried*
instead of a bare failure.

Layering: media is L4, connectors are L5 — the SoundCloud/Spotify
adapters are *injected* (constructor kwargs, or the
``soundcloud_adapter`` / ``spotify_adapter`` context attributes the
``nm music`` CLI wires up), never imported.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..tools.media import probe

_log = get_logger(__name__)

__all__ = [
    "SourceResolver",
    "ResolvedAudio",
    "ResolutionError",
    "YT_DLP_HINT",
]


YT_DLP_HINT = (
    "yt-dlp is not installed — `pip install yt-dlp` (Termux: "
    "`pip install yt-dlp`) unlocks YouTube, SoundCloud, Bandcamp, "
    "Audiomack and 1000+ other sites"
)

_AUDIO_EXTS = (
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus",
    ".aac", ".wma", ".aiff", ".mid", ".midi",
)

#: Applicability guard for the SoundCloud-API *optimization* strategy.
#: The main path stays branch-free — each strategy decides for itself
#: whether it applies; this one just avoids a wasted API round-trip on
#: URLs that are obviously not SoundCloud's.
_SC_HOST_RE = re.compile(
    r"https?://([a-z0-9-]+\.)*soundcloud\.com/", re.IGNORECASE)

_YT_ID_RE = re.compile(r"[A-Za-z0-9_\-]{11}\Z")

#: SoundCloud api-v2 ``policy`` values that are NOT full streams.
#: "ALLOW" (or empty — yt-dlp probes carry no policy) means full audio.
_PREVIEW_POLICIES = {"PREVIEW", "SNIP", "BLOCK", "DENY"}

_SPOTIFY_URI_KINDS = (
    "track", "album", "playlist", "episode", "show", "artist",
)


class ResolutionError(Exception):
    """One strategy failed.  Carries the strategy name for the audit trail."""

    def __init__(self, strategy: str, message: str) -> None:
        super().__init__(message)
        self.strategy = strategy
        self.message = message


def _info_is_preview_only(info: dict) -> bool:
    """Does a yt-dlp probe describe a preview-only SoundCloud track?

    The signal is the same one the SoundCloud adapter's ``stream_url``
    uses on final stream URLs: SoundCloud's ``/preview/`` path in any
    format/stream URL. yt-dlp probes carry no ``policy`` field, so this
    is the only pre-download signal available.
    """
    urls: list[str] = []
    if info.get("url"):
        urls.append(str(info["url"]))
    for fmt in info.get("formats") or []:
        if not isinstance(fmt, dict):
            continue
        for key in ("url", "format_note"):
            if fmt.get(key):
                urls.append(str(fmt[key]))
    return any("/preview/" in u for u in urls)


@dataclass
class ResolvedAudio:
    """The outcome of ``SourceResolver.resolve()`` — never raises."""

    ok: bool = False
    path_or_url: str = ""     # local path, watch URL, permalink, or URI
    title: str = ""
    artist: str = ""
    duration: float = 0.0
    kind: str = ""             # file | youtube | soundcloud | url | spotify
    source_name: str = ""      # human-readable: how it was resolved
    attempts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    hint: str = ""
    downloadable: bool = True  # False for Spotify (DRM, device playback)
    extra_tracks: list["ResolvedAudio"] = field(default_factory=list)


class SourceResolver:
    """Strategy-chain audio source resolution."""

    def __init__(self, context: Any = None, *,
                 soundcloud: Any = None, spotify: Any = None) -> None:
        self.context = context
        self._soundcloud = soundcloud if soundcloud is not None else \
            getattr(context, "soundcloud_adapter", None)
        self._spotify = spotify if spotify is not None else \
            getattr(context, "spotify_adapter", None)

    # ── public API ────────────────────────────────────────────────────

    def resolve(self, query: Any) -> ResolvedAudio:
        """Resolve anything playable.  Never raises."""
        result = ResolvedAudio()
        q = query.strip() if isinstance(query, str) else ""
        if not q:
            result.attempts.append("input validation — failed")
            result.errors.append("empty query")
            result.hint = "usage: /play <song title, file path, or URL>"
            return result
        try:
            if self._is_url(q):
                return self._resolve_url(q, result)
            if self._is_path_like(q):
                self._run_chain(
                    result, [("local-file",
                              lambda: self._s_local_file(q))])
                if result.ok:
                    return result
                # Not actually a file — fall through to text search,
                # keeping the attempt log so the report stays truthful.
                result.ok = False
            return self._resolve_text(q, result)
        except Exception as exc:  # noqa: BLE001 — resolve() never raises
            result.errors.append(f"resolver crashed: {exc}")
            _log.warning("SourceResolver.resolve crashed on %r: %s",
                         q[:80], exc)
            return result

    def search_candidates(self, query: str,
                          limit: int = 8) -> list[ResolvedAudio]:
        """All text-search candidates for a pick-list — no auto-picking.

        Runs the downloadable text strategies (SoundCloud search,
        YouTube search) and returns every usable candidate, deduped by
        (title, artist), best first.  Never raises; empty list on total
        failure.
        """
        out: list[ResolvedAudio] = []
        q = (query or "").strip()
        if not q:
            return out
        limit = max(1, min(int(limit or 8), 25))
        try:
            out.extend(self._soundcloud_search_many(q, limit))
        except Exception as exc:  # noqa: BLE001 — one source dying
            # must not kill the other
            _log.info("resolver candidates: soundcloud failed: %s", exc)
        try:
            out.extend(self._netnaija_search_many(q, limit))
        except Exception as exc:  # noqa: BLE001
            _log.info("resolver candidates: netnaija failed: %s", exc)
        try:
            out.extend(self._youtube_search_many(q, limit))
        except Exception as exc:  # noqa: BLE001
            _log.info("resolver candidates: youtube failed: %s", exc)
        try:
            out.extend(self._boomplay_search_many(q, limit))
        except Exception as exc:  # noqa: BLE001
            _log.info("resolver candidates: boomplay failed: %s", exc)
        seen: set[tuple[str, str]] = set()
        deduped: list[ResolvedAudio] = []
        for r in out:
            title = (r.title or "").strip()
            if not title:
                continue
            key = (title.lower(), (r.artist or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped[:limit]

    def _soundcloud_search_many(self, query: str,
                                limit: int) -> list[ResolvedAudio]:
        adapter = self._soundcloud
        if adapter is None:
            return []
        tracks = adapter.search_tracks(query, limit=limit)
        out: list[ResolvedAudio] = []
        for tr in tracks or []:
            try:
                out.append(self._track_result(tr, "SoundCloud search"))
            except ResolutionError:
                continue
        return out

    def _youtube_search_many(self, query: str,
                             limit: int) -> list[ResolvedAudio]:
        from .playback import PlaybackEngine  # lazy: same layer

        out: list[ResolvedAudio] = []
        for row in PlaybackEngine._youtube_search_many(query, limit):
            title = row.get("title") or query
            out.append(ResolvedAudio(
                ok=True,
                path_or_url=PlaybackEngine._youtube_watch_url(
                    row["video_id"]),
                title=title,
                artist=row.get("uploader") or "",
                duration=row.get("duration") or 0.0,
                kind="youtube",
                source_name="YouTube search (yt-dlp)"))
        return out

    def _netnaija_search_many(self, query: str,
                             limit: int) -> list[ResolvedAudio]:
        from .sources import NetNaijaSource  # lazy: same layer

        out: list[ResolvedAudio] = []
        for c in NetNaijaSource().search(query, limit=limit):
            out.append(ResolvedAudio(
                ok=True, path_or_url=c.url, title=c.title,
                artist=c.artist, duration=c.duration, kind="netnaija",
                source_name="NetNaija search"))
        return out

    def _boomplay_search_many(self, query: str,
                             limit: int) -> list[ResolvedAudio]:
        from .sources import BoomplaySource  # lazy: same layer

        out: list[ResolvedAudio] = []
        for c in BoomplaySource().search(query, limit=limit):
            out.append(ResolvedAudio(
                ok=True, path_or_url=c.url, title=c.title,
                artist=c.artist, kind="boomplay",
                source_name="Boomplay search", downloadable=False,
                hint="Boomplay streams are protected — open in the "
                     "Boomplay app"))
        return out

    # ── chain machinery ───────────────────────────────────────────────

    def _run_chain(self, result: ResolvedAudio,
                   strategies: list[tuple[str, Callable[[], ResolvedAudio]]]
                   ) -> ResolvedAudio:
        """Try each strategy in order; first success wins.

        Failures are recorded, never fatal — the chain only reports
        failure when EVERY strategy has failed.
        """
        for name, fn in strategies:
            try:
                out = fn()
            except ResolutionError as exc:
                result.attempts.append(f"{exc.strategy} — failed")
                result.errors.append(f"{exc.strategy}: {exc.message}")
                _log.info("resolver: %s failed: %s",
                          exc.strategy, exc.message)
                continue
            except Exception as exc:  # noqa: BLE001 — a strategy bug
                # is recorded, never fatal to the whole resolution
                result.attempts.append(f"{name} — error")
                result.errors.append(f"{name}: unexpected error: {exc}")
                _log.warning("resolver strategy %s raised: %s", name, exc)
                continue
            if not isinstance(out, ResolvedAudio) or not out.ok:
                result.attempts.append(f"{name} — no result")
                result.errors.append(f"{name}: returned no audio")
                continue
            result.attempts.append(f"{name} — ok ({out.source_name})")
            result.ok = True
            result.path_or_url = out.path_or_url
            result.title = out.title
            result.artist = out.artist
            result.duration = out.duration
            result.kind = out.kind
            result.source_name = out.source_name
            result.downloadable = out.downloadable
            result.extra_tracks = out.extra_tracks
            if out.hint and not result.hint:
                result.hint = out.hint
            return result
        return self._finalize_failure(result)

    @staticmethod
    def _finalize_failure(result: ResolvedAudio) -> ResolvedAudio:
        blob = " ".join(result.errors).lower()
        if "yt-dlp" in blob or "yt_dlp" in blob:
            result.hint = YT_DLP_HINT
        elif not result.hint:
            result.hint = \
                "tried every source — check the title/URL and try again"
        return result

    # ── input classification (not domain routing) ─────────────────────

    @staticmethod
    def _is_url(q: str) -> bool:
        return q.lower().startswith(("http://", "https://"))

    @staticmethod
    def _is_path_like(q: str) -> bool:
        if os.path.exists(os.path.expanduser(q)):
            return True
        low = q.lower()
        return low.endswith(_AUDIO_EXTS) and (
            "/" in q or "\\" in q or q.startswith(("~", ".")))

    # ── URL path ──────────────────────────────────────────────────────

    def _resolve_url(self, url: str, result: ResolvedAudio) -> ResolvedAudio:
        low = url.lower()
        if low.startswith("spotify:") or "open.spotify.com" in low \
                or "play.spotify.com" in low:
            # Spotify serves no downloadable stream — handled explicitly
            # and honestly, never through extraction.
            return self._run_chain(
                result, [("spotify-link",
                          lambda: self._s_spotify_url(url))])
        # One uniform chain for every URL.  yt-dlp routes dynamically;
        # the SoundCloud API is purely a metadata optimization with a
        # yt-dlp fallback — its failure is never fatal.
        return self._run_chain(result, [
            ("audiomack", lambda: self._s_audiomack_url(url)),
            ("soundcloud-api", lambda: self._s_soundcloud_url(url)),
            ("yt-dlp", lambda: self._s_ytdlp_url(url)),
            ("direct-url", lambda: self._s_direct_url(url)),
        ])

    def _resolve_text(self, query: str,
                      result: ResolvedAudio) -> ResolvedAudio:
        # Downloadable sources first; Spotify last — it can only play on
        # a linked device, never produce a file for chat.  Nigerian
        # sources (NetNaija) get priority for direct MP3s; Boomplay is
        # metadata-only (protected streams) so it sits with Spotify.
        return self._run_chain(result, [
            ("workspace-scan", lambda: self._s_workspace_scan(query)),
            ("netnaija-search", lambda: self._s_netnaija_search(query)),
            ("soundcloud-search",
             lambda: self._s_soundcloud_search(query)),
            ("youtube-search", lambda: self._s_youtube_search(query)),
            ("boomplay-search", lambda: self._s_boomplay_search(query)),
            ("spotify-search", lambda: self._s_spotify_search(query)),
        ])

    # ── strategies: local ─────────────────────────────────────────────

    def _s_local_file(self, query: str) -> ResolvedAudio:
        from ..tools.filesystem import safe_path

        try:
            p = safe_path(self.context, query, must_exist=True)
            if p.is_file():
                return self._file_result(str(p), "local file (workspace)")
        except Exception:  # noqa: BLE001 — fall through to absolute
            pass
        ap = os.path.abspath(os.path.expanduser(query))
        if os.path.isfile(ap):
            return self._file_result(ap, "local file")
        raise ResolutionError("local-file",
                              f"no audio file at {query!r}")

    @staticmethod
    def _file_result(path: str, source_name: str) -> ResolvedAudio:
        return ResolvedAudio(ok=True, path_or_url=path,
                             title=os.path.basename(path),
                             kind="file", source_name=source_name)

    # ── strategies: SoundCloud API (optimization, never fatal) ─────────

    def _s_soundcloud_url(self, url: str) -> ResolvedAudio:
        if not _SC_HOST_RE.search(url or ""):
            raise ResolutionError("soundcloud-api",
                                  "not a SoundCloud URL")
        adapter = self._soundcloud
        if adapter is None:
            raise ResolutionError("soundcloud-api",
                                  "SoundCloud adapter not wired")
        try:
            resolved = adapter.resolve(url)
        except Exception as exc:  # noqa: BLE001 — API hiccups fall
            # through to yt-dlp; this must never kill the resolution
            raise ResolutionError(
                "soundcloud-api",
                f"API resolve failed ({exc}); falling back") from exc
        kind = str((resolved or {}).get("kind", ""))
        if kind == "track":
            return self._track_result(resolved.get("track"),
                                      "SoundCloud API")
        if kind == "playlist":
            try:
                tracks = adapter.playlist_tracks(url)
            except Exception as exc:  # noqa: BLE001
                raise ResolutionError(
                    "soundcloud-api",
                    f"playlist expand failed ({exc}); falling back"
                ) from exc
            return self._track_list_result(tracks, url, "SoundCloud API")
        if kind == "user":
            try:
                tracks = adapter.user_tracks(url, limit=25)
            except Exception as exc:  # noqa: BLE001
                raise ResolutionError(
                    "soundcloud-api",
                    f"user tracks failed ({exc}); falling back") from exc
            return self._track_list_result(tracks, url, "SoundCloud API")
        raise ResolutionError("soundcloud-api",
                              f"unsupported SoundCloud kind {kind!r}")

    @classmethod
    def _preview_reason(cls, tr: Any) -> str:
        """Why this SoundCloud track dict is preview-only, or "" when
        it's a full stream. Checks the api-v2 ``policy`` AND explicit
        preview flags (yt-dlp probes carry no policy, so the flag is
        the only signal there)."""
        tr = tr or {}
        policy = str(tr.get("policy") or "").upper()
        if policy and policy != "ALLOW":
            return f"policy={policy}"
        if tr.get("preview") or tr.get("preview_url"):
            return "preview flag set"
        return ""

    @classmethod
    def _track_result(cls, tr: Any, source_name: str) -> ResolvedAudio:
        tr = tr or {}
        permalink = str(tr.get("permalink_url") or "")
        if not permalink:
            raise ResolutionError("soundcloud-api",
                                  "track has no permalink URL")
        # Preview-only tracks (~30s clips) are useless — fall through to
        # full-track sources (YouTube, Audiomack, etc.) instead of
        # silently downloading a preview.
        reason = cls._preview_reason(tr)
        if reason:
            raise ResolutionError(
                "soundcloud-api",
                f"track {tr.get('title', '')!r} is preview-only "
                f"({reason}); trying full-track sources")
        return ResolvedAudio(
            ok=True, path_or_url=permalink,
            title=str(tr.get("title") or permalink),
            artist=str(tr.get("artist") or ""),
            duration=float(tr.get("duration_ms") or 0) / 1000.0,
            kind="soundcloud", source_name=source_name)

    def _track_list_result(self, tracks: Any, url: str,
                           source_name: str) -> ResolvedAudio:
        tracks = [t for t in (tracks or [])
                  if (t or {}).get("permalink_url")]
        if not tracks:
            raise ResolutionError("soundcloud-api",
                                  f"no playable tracks at {url!r}")
        playable = [t for t in tracks
                    if not self._preview_reason(t or {})]
        skipped = len(tracks) - len(playable)
        if not playable:
            raise ResolutionError(
                "soundcloud-api",
                f"all {len(tracks)} tracks at {url!r} are preview-only; "
                f"trying full-track sources")
        if skipped:
            _log.info("resolver: skipped %d preview-only tracks at %s",
                      skipped, url)
        first = self._track_result(playable[0], source_name)
        first.extra_tracks = [self._track_result(t, source_name)
                              for t in playable[1:]]
        return first

    def _s_soundcloud_search(self, query: str) -> ResolvedAudio:
        adapter = self._soundcloud
        if adapter is None:
            raise ResolutionError("soundcloud-search",
                                  "SoundCloud adapter not wired")
        try:
            tracks = adapter.search_tracks(query, limit=1)
        except Exception as exc:  # noqa: BLE001
            raise ResolutionError("soundcloud-search",
                                  f"search failed: {exc}") from exc
        if not tracks:
            raise ResolutionError(
                "soundcloud-search",
                f"no SoundCloud results for {query!r}")
        return self._track_result(tracks[0], "SoundCloud search")

    # ── strategies: yt-dlp (the dynamic universal path) ───────────────

    def _s_ytdlp_url(self, url: str) -> ResolvedAudio:
        try:
            info = probe(url)
        except Exception as exc:  # noqa: BLE001 — missing yt-dlp or
            # extraction failure; the chain keeps going
            raise ResolutionError("yt-dlp", str(exc)) from exc
        info = info or {}
        extractor = str(info.get("extractor") or "").lower()
        title = str(info.get("title") or url)
        duration = float(info.get("duration") or 0)
        webpage = str(info.get("webpage_url") or url)
        if "youtube" in extractor:
            vid = str(info.get("id") or "")
            if not _YT_ID_RE.match(vid):
                raise ResolutionError(
                    "yt-dlp",
                    "no single video id extracted "
                    "(playlist URLs need a watch link)")
            return ResolvedAudio(
                ok=True,
                path_or_url=f"https://www.youtube.com/watch?v={vid}",
                title=title, duration=duration, kind="youtube",
                source_name=f"yt-dlp ({extractor})")
        if "soundcloud" in extractor:
            # yt-dlp routed a SoundCloud URL dynamically — keep the
            # soundcloud kind so the player still prefers the API for
            # fresh stream URLs, with yt-dlp as the download fallback.
            if _info_is_preview_only(info):
                # SoundCloud only serves a ~30s preview for this track —
                # don't claim it; fall through to full-track sources.
                return self._soundcloud_preview_fallback(
                    title, str(info.get("uploader") or ""), url)
            out = self._track_result(
                {"permalink_url": webpage, "title": title,
                 "artist": str(info.get("uploader") or ""),
                 "duration_ms": int(duration * 1000)},
                f"yt-dlp ({extractor})")
            return out
        return ResolvedAudio(ok=True, path_or_url=url, title=title,
                             duration=duration, kind="url",
                             source_name=f"yt-dlp ({extractor})")

    def _soundcloud_preview_fallback(self, title: str, artist: str,
                                     url: str) -> ResolvedAudio:
        """A SoundCloud URL that only yields a preview: try YouTube for
        the full track before giving up. Raises ResolutionError when
        even that finds nothing — the chain then reports honestly."""
        from .playback import PlaybackEngine  # lazy: same layer

        query = f"{artist} {title}".strip() or title
        _log.info("resolver: %s is preview-only on SoundCloud — "
                  "falling back to YouTube search for %r", url, query[:60])
        try:
            video_id = PlaybackEngine._youtube_search_id(query)
        except Exception as exc:  # noqa: BLE001
            raise ResolutionError(
                "yt-dlp",
                f"SoundCloud only serves a preview (~30s) for {title!r}; "
                f"YouTube fallback failed: {exc}") from exc
        if not _YT_ID_RE.match(str(video_id or "")):
            raise ResolutionError(
                "yt-dlp",
                f"SoundCloud only serves a preview (~30s) for {title!r}; "
                f"no YouTube full-track match either")
        return ResolvedAudio(
            ok=True,
            path_or_url=PlaybackEngine._youtube_watch_url(video_id),
            title=title, artist=artist, kind="youtube",
            source_name="YouTube search (SoundCloud preview fallback)",
            hint=f"SoundCloud only serves a ~30s preview of {title!r} — "
                 f"playing the full track from YouTube instead")

    def _s_youtube_search(self, query: str) -> ResolvedAudio:
        from .playback import PlaybackEngine  # lazy: same layer

        try:
            video_id = PlaybackEngine._youtube_search_id(query)
        except Exception as exc:  # noqa: BLE001 — carries the
            # yt-dlp install hint when yt-dlp is missing
            raise ResolutionError("youtube-search", str(exc)) from exc
        video_id = str(video_id or "")
        if not _YT_ID_RE.match(video_id):
            raise ResolutionError("youtube-search",
                                  f"no YouTube results for {query!r}")
        return ResolvedAudio(
            ok=True,
            path_or_url=PlaybackEngine._youtube_watch_url(video_id),
            title=query, kind="youtube",
            source_name="YouTube search (yt-dlp)")

    def _s_audiomack_url(self, url: str) -> ResolvedAudio:
        """First-class Audiomack URLs — yt-dlp's audiomack extractor
        handles the download (verified in yt-dlp 2026.08.19)."""
        from .sources import AudiomackSource  # lazy: same layer

        src = AudiomackSource()
        if not src.handles_url(url):
            raise ResolutionError("audiomack", "not an Audiomack track URL")
        title = url
        try:
            info = probe(url)
            title = str((info or {}).get("title") or url)
        except Exception:  # noqa: BLE001 — title is a nicety, not required
            pass
        _log.info("resolver: audiomack URL -> %s", title[:60])
        return ResolvedAudio(ok=True, path_or_url=url, title=title,
                             kind="audiomack",
                             source_name="Audiomack (yt-dlp)")

    def _s_netnaija_search(self, query: str) -> ResolvedAudio:
        """NetNaija text search — Nigerian download blog, direct MP3s."""
        from .sources import NetNaijaSource  # lazy: same layer

        cands = NetNaijaSource().search(query, limit=5)
        if not cands:
            raise ResolutionError(
                "netnaija-search", f"no NetNaija results for {query!r}")
        c = cands[0]
        _log.info("resolver: netnaija hit -> %s", c.title[:60])
        return ResolvedAudio(ok=True, path_or_url=c.url, title=c.title,
                             artist=c.artist, kind="netnaija",
                             source_name="NetNaija search")

    def _s_boomplay_search(self, query: str) -> ResolvedAudio:
        """Boomplay text search — metadata only (protected streams)."""
        from .sources import BoomplaySource  # lazy: same layer

        cands = BoomplaySource().search(query, limit=5)
        if not cands:
            raise ResolutionError(
                "boomplay-search", f"no Boomplay results for {query!r}")
        c = cands[0]
        return ResolvedAudio(
            ok=True, path_or_url=c.url, title=c.title, artist=c.artist,
            kind="boomplay", source_name="Boomplay search",
            downloadable=False,
            hint="Boomplay streams are protected — open in the Boomplay app")

    # ── strategies: leftovers ─────────────────────────────────────────

    def _s_direct_url(self, url: str) -> ResolvedAudio:
        # Last resort: the URL itself as a plain remote file.  Only
        # claimed when it actually looks like media — otherwise the
        # chain reports honestly instead of pretending.
        path_part = url.split("?", 1)[0].lower()
        if not path_part.endswith(_AUDIO_EXTS):
            raise ResolutionError("direct-url", "not a direct media link")
        name = path_part.rsplit("/", 1)[-1] or url
        return ResolvedAudio(ok=True, path_or_url=url, title=name,
                             kind="url", source_name="direct URL")

    def _s_workspace_scan(self, query: str) -> ResolvedAudio:
        settings = getattr(self.context, "settings", None) \
            if self.context is not None else None
        root = str(getattr(settings, "workspace_dir", "") or "").strip() \
            if settings is not None else ""
        if not root or not os.path.isdir(root):
            raise ResolutionError("workspace-scan",
                                  "no workspace directory")
        from difflib import SequenceMatcher

        words = [w for w in re.findall(r"[a-z0-9]+", query.lower())
                 if len(w) > 2]
        best: tuple[float, str] | None = None
        for dirpath, _dirnames, filenames in os.walk(root):
            if "/." in dirpath or "__pycache__" in dirpath:
                continue
            for fn in filenames:
                if not fn.lower().endswith(_AUDIO_EXTS):
                    continue
                stem = os.path.splitext(fn)[0].lower()
                if words:
                    hits = sum(1 for w in words if w in stem)
                    score = hits / len(words)
                else:
                    score = SequenceMatcher(
                        None, query.lower(), stem).ratio() * 0.9
                if score >= 0.5 and (best is None or score > best[0]):
                    best = (score, os.path.join(dirpath, fn))
        if best is None:
            raise ResolutionError(
                "workspace-scan", f"no local audio matches {query!r}")
        return self._file_result(best[1], "workspace scan")

    # ── strategies: Spotify (honest, no extraction) ───────────────────

    def _s_spotify_url(self, url: str) -> ResolvedAudio:
        text = (url or "").strip()
        uri = ""
        if text.startswith("spotify:"):
            parts = text.split(":")
            if len(parts) == 3 and parts[1] in _SPOTIFY_URI_KINDS \
                    and parts[2]:
                uri = text
        else:
            m = re.search(
                r"open\.spotify\.com/(?:intl-[a-z\-]+/)?"
                r"(track|album|playlist|episode|show|artist)/"
                r"([A-Za-z0-9]+)", text)
            if m:
                uri = f"spotify:{m.group(1)}:{m.group(2)}"
        if not uri:
            raise ResolutionError("spotify-link",
                                  f"not a Spotify URI or link: {url!r}")
        out = ResolvedAudio(ok=True, path_or_url=uri, title=uri,
                            kind="spotify", source_name="Spotify link",
                            downloadable=False)
        out.hint = ("Spotify serves no downloadable stream — it plays on "
                    "your linked Spotify device, not as a file in chat.")
        return out

    def _s_spotify_search(self, query: str) -> ResolvedAudio:
        adapter = self._spotify
        if adapter is None:
            raise ResolutionError("spotify-search", "Spotify not linked")
        try:
            results = adapter.search(query, types=["track"], limit=1)
        except Exception as exc:  # noqa: BLE001
            raise ResolutionError("spotify-search",
                                  f"search failed: {exc}") from exc
        items = ((results or {}).get("tracks") or {}).get("items") or []
        if not items:
            raise ResolutionError("spotify-search",
                                  f"no Spotify results for {query!r}")
        top = items[0]
        uri = str(top.get("uri") or f"spotify:track:{top.get('id', '')}")
        artists = ", ".join(
            a.get("name", "") for a in top.get("artists", [])
            if isinstance(a, dict))
        name = str(top.get("name") or "")
        title = f"{artists} – {name}" if artists and name else (
            name or uri)
        out = ResolvedAudio(ok=True, path_or_url=uri, title=title,
                            artist=artists, kind="spotify",
                            source_name="Spotify search",
                            downloadable=False)
        out.hint = ("Spotify serves no downloadable stream — it plays on "
                    "your linked Spotify device, not as a file in chat.")
        return out
