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
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..tools.media import probe
from .cookies import BOT_COOKIE_HELP, YTDLP_INSTALL_HELP

_log = get_logger(__name__)

__all__ = [
    "SourceResolver",
    "ResolvedAudio",
    "ResolutionError",
    "YT_DLP_HINT",
    "ytdlp_available",
]


#: Install hint, single-sourced from cookies.py (official yt-dlp docs) —
#: never a second, drifted copy.
YT_DLP_HINT = YTDLP_INSTALL_HELP


def ytdlp_available() -> bool:
    """Is yt-dlp actually usable here — Python module OR CLI on PATH?

    Verified, not guessed: the failure hint must never claim "not
    installed" when yt-dlp just ran and hit a bot check. Callers that
    need a fast answer in tests monkeypatch this name on the module.
    """
    import importlib.util
    import shutil

    try:
        if importlib.util.find_spec("yt_dlp") is not None:
            return True
    except Exception:  # noqa: BLE001 — a broken spec lookup is "not found"
        pass
    try:
        return shutil.which("yt-dlp") is not None
    except Exception:  # noqa: BLE001
        return False

#: Every text-search source /play tries, in chain order — single source
#: of truth, imported from sources.py so the count is testable.
#: Owner rule: MANY sources, never 1–2.
from .sources import MUSIC_SEARCH_SOURCES as PLAY_TEXT_STRATEGY_NAMES  # noqa: E402
from .sources import MUSIC_SOURCE_DOMAINS  # noqa: E402

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
                 soundcloud: Any = None, spotify: Any = None,
                 source_registry: Any = None) -> None:
        self.context = context
        self._soundcloud = soundcloud if soundcloud is not None else \
            getattr(context, "soundcloud_adapter", None)
        self._spotify = spotify if spotify is not None else \
            getattr(context, "spotify_adapter", None)
        # Source registry (Pass #88): the ONE registry the music
        # chain credits outcomes to, takes its ordering from, and
        # grows through.  Tests may inject; otherwise the shared
        # DEFAULT_REGISTRY, bound to the context's KV store exactly
        # as MediaDownloader binds it (Pass #86 pattern) — and only
        # when the context carries a database, so db-less callers
        # (pure/offline contexts) keep the pre-#88 behaviour exactly.
        self._source_registry_injected = source_registry
        self._source_registry_cache: Any = None
        self._source_registry_resolved = False

    # ── source registry: outcomes, ordering, growth (Pass #88) ──────

    def _registry(self) -> Any:
        """The source registry for music outcomes, or None when this
        resolver has no registry to speak to (no injection and a
        context without a database).  Never raises."""
        if self._source_registry_resolved:
            return self._source_registry_cache
        self._source_registry_resolved = True
        if self._source_registry_injected is not None:
            self._source_registry_cache = self._source_registry_injected
            return self._source_registry_cache
        try:
            db = getattr(self.context, "db", None) \
                if self.context is not None else None
            if db is None:
                return None
            from .source_registry import DEFAULT_REGISTRY
            reg = DEFAULT_REGISTRY
            try:
                from ..storage.kv import KVStore
                reg.bind_store(KVStore(db))
            except Exception:  # noqa: BLE001 - persistence is an
                # enhancement; its absence never sinks a resolution
                pass
            self._source_registry_cache = reg
            return reg
        except Exception:  # noqa: BLE001
            return None

    def _desc_for_strategy(self, reg: Any, name: str) -> Any:
        """The registry descriptor a chain strategy serves from
        (``sources.MUSIC_SOURCE_DOMAINS``), or None.  Never raises."""
        try:
            domain = MUSIC_SOURCE_DOMAINS.get(name)
            if not domain:
                return None
            return reg.for_domain(domain)
        except Exception:  # noqa: BLE001
            return None

    def _record_served(self, reg: Any, desc: Any, latency_s: float,
                       out: Any = None) -> None:
        """A strategy served: credit its source, and note any foreign
        host the winning candidate points at as an UNTRUSTED
        discovery candidate (search results genuinely carry such
        hosts — Openverse results live on creator domains the
        registry has never seen).  Best-effort; never raises."""
        try:
            if desc is not None:
                reg.record_outcome(desc, True, latency_s=latency_s)
            url = getattr(out, "path_or_url", "") if out is not None \
                else ""
            if url:
                reg.note_candidate(url, via="music-search-result")
        except Exception:  # noqa: BLE001 - bookkeeping never
            # breaks resolution
            pass

    def _record_no_result(self, reg: Any, desc: Any) -> None:
        """A search reached the source and found nothing: weak
        evidence (see SourceRegistry.record_no_result)."""
        try:
            if desc is not None:
                reg.record_no_result(desc)
        except Exception:  # noqa: BLE001
            pass

    def _record_failure(self, reg: Any, desc: Any, error_class: str,
                        latency_s: float = 0.0) -> None:
        """An operational failure against the serving source, with
        the error class the code actually knows."""
        try:
            if desc is not None:
                reg.record_outcome(desc, False, latency_s=latency_s,
                                   error_class=error_class)
        except Exception:  # noqa: BLE001
            pass

    def _record_strategy_error(self, reg: Any, name: str, desc: Any,
                               exc: ResolutionError,
                               latency_s: float) -> None:
        """Classify one strategy's ResolutionError into registry
        evidence.  The taxonomy is honest about what the seam can
        observe (the sources' ``search()`` never raises, so most
        catalogue misses arrive as 'no … results'):

        * not-applicable / not-configured (wrong URL shape, adapter
          unwired, Spotify unlinked, no workspace, yt-dlp binary
          missing for the yt-dlp strategies) → NO evidence: these
          are facts about the query or this install, not the source;
        * operational failures ('… failed …') → a real failure with
          an error class, feeding the circuit breaker;
        * everything else (no results, preview-only catalogue hits)
          → a no-result: weak evidence, score-only.
        """
        try:
            if desc is None:
                return
            msg = (exc.message or "").lower()
            if ("not wired" in msg or "not linked" in msg
                    or msg.startswith("not a ")
                    or msg.startswith("not an ")
                    or "no workspace" in msg
                    or "no audio file" in msg
                    or "no local audio" in msg):
                return
            if name in ("youtube-search", "youtube-music-search") \
                    and not ytdlp_available():
                return  # the binary is missing — an install fact
            if "failed" in msg or "error" in msg:
                self._record_failure(
                    reg, desc,
                    "search_failed" if "search" in msg
                    else "resolve_failed",
                    latency_s)
            else:
                self._record_no_result(reg, desc)
        except Exception:  # noqa: BLE001
            pass

    def _music_shim(self, reg: Any, name: str, fn: Callable,
                    desc: Any) -> Callable:
        """Wrap one chain strategy so its outcome lands in the
        registry (Pass #88).  Transparent: same return, same raise."""

        def run() -> ResolvedAudio:
            t0 = time.perf_counter()
            try:
                out = fn()
            except ResolutionError as exc:
                self._record_strategy_error(
                    reg, name, desc, exc, time.perf_counter() - t0)
                raise
            except Exception:
                self._record_failure(reg, desc, "strategy_error",
                                     time.perf_counter() - t0)
                raise
            latency = time.perf_counter() - t0
            if isinstance(out, ResolvedAudio) and out.ok:
                self._record_served(reg, desc, latency, out)
            else:
                self._record_no_result(reg, desc)
            return out

        return run

    def _order_music(self, reg: Any,
                     items: list[tuple[str, Any, Any]]
                     ) -> list[tuple[str, Any, Any]]:
        """Outcome-aware chain order (Pass #88), through the
        registry's existing score/health semantics:

        * NO learned state anywhere in the chain → the list is
          returned untouched: day-one behaviour is today's chain
          order, byte-for-byte;
        * otherwise active sources sort by registry score (stable —
          ties keep chain order) and stood-down / disabled / dead
          sources are demoted to the end, in chain order: they leave
          the active ordering (the #86 vocabulary) but the chain
          still tries them last rather than pretending they do not
          exist — the attempt log stays truthful.
        Never raises; a registry hiccup returns the chain as-is."""
        try:
            evidence = False
            for _name, _fn, desc in items:
                if desc is None:
                    continue
                st = reg.stats_for(desc)
                if st["ok"] or st["fail"] or st["no_result"] \
                        or not reg.is_active(desc):
                    evidence = True
                    break
            if not evidence:
                return items
            active = [it for it in items
                      if it[2] is None or reg.is_active(it[2])]
            inactive = [it for it in items
                        if it[2] is not None and not reg.is_active(it[2])]
            active.sort(key=lambda it: -reg.score(it[2])
                        if it[2] is not None else 0.0)
            return active + inactive
        except Exception:  # noqa: BLE001
            return items

    def _grown_music(self, reg: Any) -> list[Any]:
        """Registry descriptors for GROWN music sources: candidates
        the health loop promoted (origin ``grown``, kind ``music``).
        Candidates that were never promoted are NOT here — they
        serve nothing.  Never raises."""
        try:
            return [d for d in reg.all()
                    if reg.origin_for(d) == "grown"
                    and "music" in d.kinds]
        except Exception:  # noqa: BLE001
            return []

    def _s_grown_search(self, query: str, desc: Any) -> ResolvedAudio:
        """One grown music source, served through the generic
        WordPress blog shape (sources.DiscoveredBlogSource)."""
        from .sources import DiscoveredBlogSource  # lazy: same layer

        src = DiscoveredBlogSource(desc.domain)
        cands = src.search(query, limit=5)
        if not cands:
            raise ResolutionError(
                f"grown:{desc.domain}",
                f"no results on grown source {desc.domain} for {query!r}")
        c = cands[0]
        return ResolvedAudio(
            ok=True, path_or_url=c.url, title=c.title, artist=c.artist,
            kind=f"grown:{desc.domain}",
            source_name=f"{desc.name or desc.domain} (grown source)")

    def _grown_search_many(self, desc: Any, query: str,
                           limit: int) -> list[ResolvedAudio]:
        from .sources import DiscoveredBlogSource  # lazy: same layer

        src = DiscoveredBlogSource(desc.domain)
        return [ResolvedAudio(
            ok=True, path_or_url=c.url, title=c.title, artist=c.artist,
            kind=f"grown:{desc.domain}",
            source_name=f"{desc.name or desc.domain} (grown source)")
            for c in src.search(query, limit=limit)]

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

        Runs EVERY downloadable text strategy (all of
        MUSIC_SEARCH_SOURCES that can return candidates) and returns
        every usable candidate, deduped by (title, artist), best first.
        One source failing never kills the others.  Never raises; empty
        list on total failure.
        """
        out, _attempts = self.search_candidates_detailed(query, limit=limit)
        return out

    def search_candidates_detailed(self, query: str, limit: int = 8
                                   ) -> tuple[list[ResolvedAudio],
                                              list[str]]:
        """Same candidate sweep as :meth:`search_candidates`, plus the
        per-source audit trail.

        Returns ``(candidates, attempts)`` where ``attempts`` has one
        entry per source tried — ``"<name> — ok (N)"``,
        ``"<name> — no results"``, or ``"<name> — failed: <reason>"``.
        Only DOWNLOADABLE candidates are returned: a metadata-only
        (MusicBrainz) or protected-stream (Boomplay) hit is not
        something the caller can download, so it must never show up
        as a pickable/downloadable candidate — the audit trail still
        records that it was tried.  The chat download fall-through
        needs this: a source whose search succeeded but whose
        download failed is a *download* failure, not success, and the
        reply must list EVERY source actually tried — not just the
        ones the first-ok chain reached.  Never raises.
        """
        out: list[ResolvedAudio] = []
        attempts: list[str] = []
        q = (query or "").strip()
        if not q:
            return out, attempts
        limit = max(1, min(int(limit or 8), 25))
        # Regional blogs first (direct MP3s), then the global catalogs.
        fns: list[tuple[str, Any]] = [
            ("netnaija-search", self._netnaija_search_many),
            ("notjustok-search", self._notjustok_search_many),
            ("tooxclusive-search", self._tooxclusive_search_many),
            ("soundcloud-search", self._soundcloud_search_many),
            ("bandcamp-search", self._bandcamp_search_many),
            ("youtube-search", self._youtube_search_many),
            ("youtube-music-search", self._youtube_music_search_many),
            ("archiveorg-search", self._archiveorg_search_many),
            ("musicbrainz-search", self._musicbrainz_search_many),
            ("openverse-search", self._openverse_search_many),
            ("wikimedia-search", self._wikimedia_search_many),
            ("itunes-search", self._itunes_search_many),
            ("deezer-search", self._deezer_search_many),
        ]
        # Pass #88: the sweep is a real music path — its outcomes
        # land in the same registry (served / no-result / failure per
        # source), its order follows the same learned ordering, and
        # grown sources sweep too.  With no registry the sweep is
        # exactly the list above, in this order.
        reg = self._registry()
        items: list[tuple[str, Any, Any]] = []
        if reg is not None:
            for name, fn in fns:
                items.append(
                    (name, fn, self._desc_for_strategy(reg, name)))
            for desc in self._grown_music(reg):
                items.append((
                    f"grown:{desc.domain}",
                    lambda d=desc: self._grown_search_many(d, q, limit),
                    desc))
            items = self._order_music(reg, items)
            fns = [(name, fn) for name, fn, _d in items]
        descs = {name: desc for name, _fn, desc in items}
        for name, fn in fns:
            desc = descs.get(name)
            t0 = time.perf_counter()
            try:
                got = fn(q, limit)
            except Exception as exc:  # noqa: BLE001 — one source dying
                # must not kill the others
                attempts.append(f"{name} — failed: {exc}")
                _log.info("resolver candidates: %s failed: %s",
                          getattr(fn, "__name__", "?"), exc)
                if reg is not None and desc is not None:
                    self._record_failure(reg, desc, "search_failed",
                                         time.perf_counter() - t0)
                continue
            if got:
                attempts.append(f"{name} — ok ({len(got)})")
                out.extend(got)
                if reg is not None and desc is not None:
                    self._record_served(
                        reg, desc, time.perf_counter() - t0, got[0])
                    for extra in got[1:]:
                        try:
                            reg.note_candidate(
                                extra.path_or_url,
                                via="music-search-result")
                        except Exception:  # noqa: BLE001
                            pass
            else:
                attempts.append(f"{name} — no results")
                if reg is not None and desc is not None \
                        and not (name == "soundcloud-search"
                                 and self._soundcloud is None):
                    # An unwired SoundCloud adapter returning [] is a
                    # configuration fact, not source evidence.
                    self._record_no_result(reg, desc)
        seen: set[tuple[str, str]] = set()
        deduped: list[ResolvedAudio] = []
        for r in out:
            title = (r.title or "").strip()
            if not title:
                continue
            # Non-downloadable hits (metadata-only / protected
            # streams) are not candidates — nothing can download
            # them, so offering them to pick/download only produces
            # a guaranteed failure.  The attempts trail above still
            # records that the source was tried.
            if r.downloadable is False:
                continue
            key = (title.lower(), (r.artist or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped[:limit], attempts

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

    def _youtube_music_search_many(self, query: str,
                                   limit: int) -> list[ResolvedAudio]:
        """YouTube Music candidates (yt-dlp ``ytmsearch``).

        YouTube Music's catalog carries the K-pop, J-pop, Latin and
        regional music plain YouTube search misses.
        """
        from .playback import PlaybackEngine  # lazy: same layer

        out: list[ResolvedAudio] = []
        for row in PlaybackEngine._youtube_search_many(
                query, limit, prefix="ytmsearch"):
            title = row.get("title") or query
            out.append(ResolvedAudio(
                ok=True,
                path_or_url=PlaybackEngine._youtube_watch_url(
                    row["video_id"]),
                title=title,
                artist=row.get("uploader") or "",
                duration=row.get("duration") or 0.0,
                kind="youtube-music",
                source_name="YouTube Music search (yt-dlp)"))
        return out

    def _api_source_search_many(self, query: str, limit: int,
                                cls_name: str, kind: str, label: str,
                                downloadable: bool) -> list[ResolvedAudio]:
        """Candidates from a keyless JSON-API source class in sources.py."""
        import importlib

        mod = importlib.import_module("nomorals.media.sources")
        cls = getattr(mod, cls_name)
        out: list[ResolvedAudio] = []
        for c in cls().search(query, limit=limit):
            out.append(ResolvedAudio(
                ok=True, path_or_url=c.direct_url or c.url,
                title=c.title, artist=c.artist, duration=c.duration,
                kind=kind, source_name=label, downloadable=downloadable,
                hint=str((c.extra or {}).get("note") or "")))
        return out

    def _musicbrainz_search_many(self, query: str,
                                 limit: int) -> list[ResolvedAudio]:
        return self._api_source_search_many(
            query, limit, "MusicBrainzSource", "musicbrainz",
            "MusicBrainz search", downloadable=False)

    def _openverse_search_many(self, query: str,
                               limit: int) -> list[ResolvedAudio]:
        return self._api_source_search_many(
            query, limit, "OpenverseSource", "openverse",
            "Openverse search", downloadable=True)

    def _wikimedia_search_many(self, query: str,
                               limit: int) -> list[ResolvedAudio]:
        return self._api_source_search_many(
            query, limit, "WikimediaSource", "wikimedia",
            "Wikimedia Commons search", downloadable=True)

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

    def _blog_search_many(self, query: str, limit: int,
                          cls_name: str, kind: str,
                          label: str) -> list[ResolvedAudio]:
        import importlib

        mod = importlib.import_module("nomorals.media.sources")
        cls = getattr(mod, cls_name)
        out: list[ResolvedAudio] = []
        for c in cls().search(query, limit=limit):
            out.append(ResolvedAudio(
                ok=True, path_or_url=c.url, title=c.title,
                artist=c.artist, kind=kind, source_name=label))
        return out

    def _notjustok_search_many(self, query: str,
                               limit: int) -> list[ResolvedAudio]:
        return self._blog_search_many(query, limit, "NotJustOkSource",
                                      "notjustok", "NotJustOk search")

    def _tooxclusive_search_many(self, query: str,
                                 limit: int) -> list[ResolvedAudio]:
        return self._blog_search_many(query, limit, "TooXclusiveSource",
                                      "tooxclusive", "TooXclusive search")

    def _bandcamp_search_many(self, query: str,
                              limit: int) -> list[ResolvedAudio]:
        from .sources import BandcampSource  # lazy: same layer

        out: list[ResolvedAudio] = []
        for c in BandcampSource().search(query, limit=limit):
            out.append(ResolvedAudio(
                ok=True, path_or_url=c.url, title=c.title,
                artist=c.artist, kind="bandcamp",
                source_name="Bandcamp search"))
        return out

    def _archiveorg_search_many(self, query: str,
                                limit: int) -> list[ResolvedAudio]:
        from .sources import ArchiveOrgSource  # lazy: same layer

        src = ArchiveOrgSource()
        out: list[ResolvedAudio] = []
        for c in src.search(query, limit=limit):
            out.append(ResolvedAudio(
                ok=True, path_or_url=c.url, title=c.title,
                artist=c.artist, kind="archiveorg",
                source_name="Internet Archive search"))
        return out

    def _preview_search_many(self, query: str, limit: int,
                             cls_name: str, kind: str,
                             label: str) -> list[ResolvedAudio]:
        import importlib

        mod = importlib.import_module("nomorals.media.sources")
        cls = getattr(mod, cls_name)
        out: list[ResolvedAudio] = []
        for c in cls().search(query, limit=limit):
            if not (c.extra or {}).get("preview_only"):
                continue
            out.append(ResolvedAudio(
                ok=True, path_or_url=c.direct_url or c.url,
                title=c.title, artist=c.artist, kind=kind,
                source_name=label,
                hint=str((c.extra or {}).get("note") or "")))
        return out

    def _itunes_search_many(self, query: str,
                            limit: int) -> list[ResolvedAudio]:
        return self._preview_search_many(query, limit, "ITunesSource",
                                         "itunes", "iTunes search (30s preview)")

    def _deezer_search_many(self, query: str,
                            limit: int) -> list[ResolvedAudio]:
        return self._preview_search_many(query, limit, "DeezerSource",
                                         "deezer", "Deezer search (30s preview)")

    # ── chain machinery ───────────────────────────────────────────────

    def _run_chain(self, result: ResolvedAudio,
                   strategies: list[tuple[str, Callable[[], ResolvedAudio]]]
                   ) -> ResolvedAudio:
        """Try each strategy in order; first DOWNLOADABLE success wins.

        Failures are recorded, never fatal — the chain only reports
        failure when EVERY strategy has failed.

        A hit that is found but NOT downloadable (Boomplay's protected
        streams, MusicBrainz metadata, Spotify DRM) is recorded and the
        chain KEEPS GOING: a later downloadable source must win over
        an earlier undownloadable one — "found but can't download" is
        not success.  When nothing downloadable is found, the FIRST
        non-downloadable hit is returned (found-but-blocked) with the
        full attempt log, so the caller can explain what was found and
        why it can't be delivered instead of "couldn't find".
        """
        first_blocked: ResolvedAudio | None = None
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
            if out.downloadable is False:
                # Found but not deliverable — record and keep going.
                result.attempts.append(
                    f"{name} — ok ({out.source_name}; not downloadable)")
                if first_blocked is None:
                    first_blocked = out
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
        if first_blocked is not None:
            # Nothing downloadable — return the first found-but-blocked
            # hit so the caller can name the source and its block.
            result.ok = True
            result.path_or_url = first_blocked.path_or_url
            result.title = first_blocked.title
            result.artist = first_blocked.artist
            result.duration = first_blocked.duration
            result.kind = first_blocked.kind
            result.source_name = first_blocked.source_name
            result.downloadable = False
            result.extra_tracks = first_blocked.extra_tracks
            if first_blocked.hint and not result.hint:
                result.hint = first_blocked.hint
            return result
        return self._finalize_failure(result)

    @staticmethod
    def _finalize_failure(result: ResolvedAudio) -> ResolvedAudio:
        # Hint classification from VERIFIED facts, never substring
        # matches: a bot-check error contains "yt-dlp" in its GitHub
        # FAQ URL, which must NOT read as "yt-dlp not installed".
        blob = " ".join(result.errors).lower()
        if ("not a bot" in blob or "confirm you're not a bot" in blob
                or "sign in to confirm" in blob):
            # yt-dlp RAN and hit a bot check — the fix is cookies.
            result.hint = BOT_COOKIE_HELP
        elif not ytdlp_available() and ("yt-dlp" in blob or "yt_dlp" in blob):
            # yt-dlp verified missing AND an error says so — install hint.
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
            out = self._run_chain(
                result, [("spotify-link",
                          lambda: self._s_spotify_url(url))])
            self._record_url_outcome(url, out)
            return out
        # One uniform chain for every URL.  yt-dlp routes dynamically;
        # the SoundCloud API is purely a metadata optimization with a
        # yt-dlp fallback — its failure is never fatal.
        out = self._run_chain(result, [
            ("audiomack", lambda: self._s_audiomack_url(url)),
            ("soundcloud-api", lambda: self._s_soundcloud_url(url)),
            ("yt-dlp", lambda: self._s_ytdlp_url(url)),
            ("direct-url", lambda: self._s_direct_url(url)),
        ])
        self._record_url_outcome(url, out)
        return out

    def _record_url_outcome(self, url: str, out: ResolvedAudio) -> None:
        """Credit/blame the source a pasted URL belongs to (Pass
        #88): a resolve that yielded a playable URL is a success for
        the URL's host source; a total failure is recorded against
        it with the resolve_failed class.  A host the registry does
        not know is noted as an UNTRUSTED discovery candidate (via
        "play-url") — the play path's analogue of the downloader's
        download-url intake (#86).  Best-effort; never raises."""
        try:
            reg = self._registry()
            if reg is None:
                return
            desc = reg.for_url(url)
            if desc is None:
                reg.note_candidate(url, via="play-url")
            elif out.ok:
                reg.record_outcome(desc, True)
            else:
                reg.record_outcome(desc, False,
                                   error_class="resolve_failed")
        except Exception:  # noqa: BLE001 - bookkeeping never
            # breaks resolution
            pass

    def _resolve_text(self, query: str,
                      result: ResolvedAudio) -> ResolvedAudio:
        # MANY sources, in chain order (see MUSIC_SEARCH_SOURCES).
        # Downloadable/full-track sources first; regional blogs before
        # globals; metadata-only (Boomplay, MusicBrainz) + 30s-preview
        # APIs (iTunes/Deezer) + Spotify last. Boomplay is back in the
        # chain via browser-backed search (verified live 2026-10-18).
        # GLOBAL coverage: Nigeria/Africa is one region
        # among many — the chain spans worldwide catalogs (YouTube
        # Music, MusicBrainz, Openverse, Wikimedia, iTunes, Deezer).
        # One source failing never ends the search — the chain keeps
        # going through ALL of them.
        strategies: list[tuple[str, Callable[[], ResolvedAudio]]] = [
            ("workspace-scan", lambda: self._s_workspace_scan(query)),
            ("netnaija-search", lambda: self._s_netnaija_search(query)),
            ("notjustok-search", lambda: self._s_blog_search(
                query, "NotJustOkSource", "notjustok")),
            ("tooxclusive-search", lambda: self._s_blog_search(
                query, "TooXclusiveSource", "tooxclusive")),
            ("soundcloud-search",
             lambda: self._s_soundcloud_search(query)),
            ("bandcamp-search", lambda: self._s_bandcamp_search(query)),
            ("youtube-search", lambda: self._s_youtube_search(query)),
            ("youtube-music-search",
             lambda: self._s_youtube_music_search(query)),
            ("archiveorg-search", lambda: self._s_archiveorg_search(query)),
            ("boomplay-search", lambda: self._s_boomplay_search(query)),
            ("musicbrainz-search", lambda: self._s_metadata_search(
                query, "MusicBrainzSource", "musicbrainz",
                "MusicBrainz metadata only — no audio; find the track "
                "via the full-track sources")),
            ("openverse-search", lambda: self._s_blog_search(
                query, "OpenverseSource", "openverse")),
            ("wikimedia-search", lambda: self._s_blog_search(
                query, "WikimediaSource", "wikimedia")),
            ("itunes-search", lambda: self._s_preview_search(
                query, "ITunesSource", "itunes")),
            ("deezer-search", lambda: self._s_preview_search(
                query, "DeezerSource", "deezer")),
            ("spotify-search", lambda: self._s_spotify_search(query)),
        ]
        # Pass #88: the chain learns.  Each strategy is wrapped so
        # its outcome lands in the ONE source registry against the
        # source that served (sources.MUSIC_SOURCE_DOMAINS), the
        # order becomes outcome-aware through the registry's
        # score/health semantics, and health-promoted GROWN sources
        # are appended as strategies of their own.  With no registry
        # (db-less context) or no learned state, the chain below is
        # exactly the list above, in this order.
        reg = self._registry()
        if reg is None:
            return self._run_chain(result, strategies)
        items: list[tuple[str, Any, Any]] = []
        for name, fn in strategies:
            desc = self._desc_for_strategy(reg, name)
            items.append((name, self._music_shim(reg, name, fn, desc),
                          desc))
        for desc in self._grown_music(reg):
            name = f"grown:{desc.domain}"
            items.append((name, self._music_shim(
                reg, name,
                lambda d=desc: self._s_grown_search(query, d), desc),
                desc))
        ordered = self._order_music(reg, items)
        return self._run_chain(
            result, [(name, fn) for name, fn, _desc in ordered])

    def _s_blog_search(self, query: str, cls_name: str,
                       kind: str) -> ResolvedAudio:
        import importlib

        mod = importlib.import_module("nomorals.media.sources")
        cls = getattr(mod, cls_name)
        cands = cls().search(query, limit=5)
        if not cands:
            raise ResolutionError(
                f"{kind}-search", f"no {cls_name} results for {query!r}")
        c = cands[0]
        _log.info("resolver: %s hit -> %s", kind, c.title[:60])
        return ResolvedAudio(ok=True, path_or_url=c.url, title=c.title,
                             artist=c.artist, kind=kind,
                             source_name=f"{cls_name} search")

    def _s_bandcamp_search(self, query: str) -> ResolvedAudio:
        from .sources import BandcampSource  # lazy: same layer

        cands = BandcampSource().search(query, limit=5)
        if not cands:
            raise ResolutionError(
                "bandcamp-search", f"no Bandcamp results for {query!r}")
        c = cands[0]
        return ResolvedAudio(ok=True, path_or_url=c.url, title=c.title,
                             artist=c.artist, kind="bandcamp",
                             source_name="Bandcamp search")

    def _s_archiveorg_search(self, query: str) -> ResolvedAudio:
        from .sources import ArchiveOrgSource  # lazy: same layer

        src = ArchiveOrgSource()
        cands = src.search(query, limit=5)
        if not cands:
            raise ResolutionError(
                "archiveorg-search",
                f"no Internet Archive results for {query!r}")
        c = cands[0]
        return ResolvedAudio(ok=True, path_or_url=c.url, title=c.title,
                             artist=c.artist, kind="archiveorg",
                             source_name="Internet Archive search")

    def _s_preview_search(self, query: str, cls_name: str,
                          kind: str) -> ResolvedAudio:
        """30s-preview APIs (iTunes/Deezer): last-resort only.

        The candidate is a real preview MP3, but only 30s — the chain
        reaches here only after every full-track source failed, and
        the hint says so honestly.
        """
        import importlib

        mod = importlib.import_module("nomorals.media.sources")
        cls = getattr(mod, cls_name)
        cands = cls().search(query, limit=5)
        if not cands:
            raise ResolutionError(
                f"{kind}-search", f"no {cls_name} results for {query!r}")
        c = cands[0]
        if not (c.extra or {}).get("preview_only"):
            raise ResolutionError(
                f"{kind}-search", f"{cls_name} returned no preview for {query!r}")
        out = ResolvedAudio(
            ok=True, path_or_url=c.direct_url or c.url, title=c.title,
            artist=c.artist, kind=kind,
            source_name=f"{cls_name} search (30s preview)")
        out.hint = str((c.extra or {}).get("note") or
                       "30-second preview only — full track not found "
                       "on any source")
        return out

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
        # Only the hard preview policies count: MONETIZE is a full
        # stream (the real stream URL is the only preview-only proof,
        # checked by the adapter's stream_url). Treating MONETIZE as
        # preview-only discards real full tracks.
        if policy in _PREVIEW_POLICIES:
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
            # Fetch up to 5 and skip preview-only ones: fetching exactly
            # 1 fails the whole strategy when the first hit is a clip.
            tracks = adapter.search_tracks(query, limit=5)
        except Exception as exc:  # noqa: BLE001
            raise ResolutionError("soundcloud-search",
                                  f"search failed: {exc}") from exc
        if not tracks:
            raise ResolutionError(
                "soundcloud-search",
                f"no SoundCloud results for {query!r}")
        playable = [t for t in tracks if not self._preview_reason(t or {})]
        if not playable:
            raise ResolutionError(
                "soundcloud-search",
                f"all {len(tracks)} SoundCloud search results are "
                "preview-only; trying full-track sources")
        return self._track_result(playable[0], "SoundCloud search")

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

    def _s_youtube_music_search(self, query: str) -> ResolvedAudio:
        """YouTube Music search (yt-dlp ``ytmsearch1:``)."""
        from .playback import PlaybackEngine  # lazy: same layer

        try:
            video_id = PlaybackEngine._youtube_search_id(
                query, prefix="ytmsearch")
        except Exception as exc:  # noqa: BLE001 — carries the
            # yt-dlp install hint when yt-dlp is missing
            raise ResolutionError("youtube-music-search", str(exc)) from exc
        video_id = str(video_id or "")
        if not _YT_ID_RE.match(video_id):
            raise ResolutionError("youtube-music-search",
                                  f"no YouTube Music results for {query!r}")
        return ResolvedAudio(
            ok=True,
            path_or_url=PlaybackEngine._youtube_watch_url(video_id),
            title=query, kind="youtube-music",
            source_name="YouTube Music search (yt-dlp)")

    def _s_metadata_search(self, query: str, cls_name: str,
                           kind: str, note: str) -> ResolvedAudio:
        """Metadata-only sources (MusicBrainz): discovery, no audio."""
        import importlib

        mod = importlib.import_module("nomorals.media.sources")
        cls = getattr(mod, cls_name)
        cands = cls().search(query, limit=5)
        if not cands:
            raise ResolutionError(
                f"{kind}-search", f"no {cls_name} results for {query!r}")
        c = cands[0]
        return ResolvedAudio(
            ok=True, path_or_url=c.url, title=c.title, artist=c.artist,
            kind=kind, source_name=f"{cls_name} search",
            downloadable=False, hint=note)

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
        """Boomplay text search — metadata only (protected streams).

        In the active chain since 2026-10-18: browser-backed SPA
        search (session cookies + browser headers) returns real
        candidates where plain-HTTP search returned none.
        """
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
