"""Step 6 — /play service: fallback chain that never dead-ends.

``/play`` in chat lives in ``agents/partner/runtime_media.py``
(:meth:`_control_play`); the resolution chain lives in
:mod:`nomorals.media.resolver` (``SourceResolver``).  This module is the
service layer on top of both:

* :class:`PlayService.resolve` — the fallback chain.  INVARIANT: the
  returned :class:`PlayReport` ALWAYS carries what was tried and why
  each attempt failed — even on total failure the user gets "best
  available + what was tried + why failed", never a dead end.
* LLM-first understanding — :meth:`PlayService.understand` asks the
  brain (when one is wired) to parse title / artist / mood / nation
  from free text; the deterministic ``picklist`` parser is the
  fallback, never the primary.  Ekiti queries are search-normalized
  via the Step-4 lexicon; the ORIGINAL is preserved for display/TTS.
* Quality + licence ladder — :func:`quality_licence_rank` orders
  candidates by quality first, licence honesty second: a licensed
  source wins ties, and an unlicensed source is LABELLED unlicensed,
  never laundered into looking official.
* More extractors — resolution consults the Step-5
  :class:`ExtractorRegistry` (yt-dlp + Audiomack/Boomplay/NetNaija/
  Spotify/YouTube/SoundCloud + direct media).  Pirated sources are
  treated as DATA with an honest ``unlicensed`` label; jurisdiction is
  a per-deployment setting (``NM_PLAY_UNLICENSED``: ``allow`` /
  ``label-only`` / ``block`` — default ``label-only``: labelled, never
  auto-downloaded); real accounts/cookies are NEVER touched — the
  owner provides a cookie explicitly via the Step-5 vault flow.
* DJ continuity — :meth:`PlayService.dj_continuity` closes the taste
  loop on ``dj_engine`` skip data: skip-rate per transition, energy
  curve target, key/tempo compatibility, no-repeat window, and
  read-the-room (a run of skips cools the energy target).
* Per-nation charts merge — :func:`merge_charts` blends several chart
  providers with weights + TTL; providers are injected callables, so
  no chart is ever fabricated — an empty provider set yields an empty
  merge with an honest note.

Never raises out of the public functions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .download_kit import DEFAULT_EXTRACTORS, ExtractorRegistry
from .picklist import parse_title_artist
from .source_registry import DEFAULT_REGISTRY, EkitiLexicon, SourceRegistry

_log = get_logger(__name__)

__all__ = [
    "PlayIntent",
    "PlayAttempt",
    "PlayReport",
    "quality_licence_rank",
    "merge_charts",
    "ChartEntry",
    "ContinuityPlan",
    "PlayService",
    "UNLICENSED_MODES",
]

#: Per-deployment jurisdiction switch for unlicensed sources.
#: "label-only" (default): labelled, never auto-downloaded.
#: "allow": operator explicitly accepts local law for their deployment.
#: "block": unlicensed sources are excluded from candidates entirely.
UNLICENSED_MODES = ("allow", "label-only", "block")


@dataclass
class PlayIntent:
    """What the user wants, as understood (LLM-first, deterministic fallback)."""

    original: str = ""       # exactly what the user typed — display/TTS
    search_query: str = ""   # normalized for source queries (Ekiti → std)
    title: str = ""
    artist: str = ""
    mood: str = ""
    nation: str = ""         # "NG" etc. when detected
    understood_by: str = ""  # "llm" | "deterministic"
    mapped_terms: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PlayAttempt:
    strategy: str = ""
    ok: bool = False
    detail: str = ""         # what happened — the "why failed"
    licence: str = "unknown"
    quality: str = ""


@dataclass
class PlayReport:
    """The /play outcome.  Never a dead end: ``tried`` is always
    populated, and ``best_available`` names the best thing found even
    when nothing is downloadable."""

    ok: bool = False
    best_available: str = ""     # title/URL of the best candidate found
    path_or_url: str = ""
    title: str = ""
    licence: str = "unknown"
    quality: str = ""
    tried: list[PlayAttempt] = field(default_factory=list)
    why_failed: str = ""         # one-line summary when ok is False
    hint: str = ""

    def summary(self) -> str:
        if self.ok:
            return f"▶ {self.title or self.best_available} [{self.licence}]"
        lines = [f"couldn't play {self.best_available or 'that'} — tried:"]
        for a in self.tried:
            lines.append(f"  {a.strategy}: {a.detail or 'failed'}")
        if self.why_failed:
            lines.append(f"  why: {self.why_failed}")
        if self.hint:
            lines.append(f"  hint: {self.hint}")
        return "\n".join(lines)


@dataclass
class ChartEntry:
    title: str = ""
    artist: str = ""
    nation: str = ""
    rank: int = 0
    providers: tuple[str, ...] = ()
    weight: float = 0.0


def quality_licence_rank(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order candidates: quality first, licence honesty second.

    Quality order: lossless > 1080p/320k > 720p > 480p > audio > unknown.
    At equal quality, licensed beats user-upload beats unlicensed —
    but unlicensed is never dropped silently: it keeps its honest label
    and the caller (PlayService) decides per the jurisdiction mode."""
    _q = {"lossless": 6, "flac": 6, "1080p": 5, "320k": 5, "720p": 4,
          "480p": 3, "audio": 2, "360p": 2, "": 1, "unknown": 1}
    _lic = {"licensed": 3, "user-upload": 2, "unknown": 1, "unlicensed": 0}

    def key(c: dict[str, Any]) -> tuple[int, int, float]:
        q = str(c.get("quality") or "").lower()
        lic = str(c.get("licence") or "").lower()
        return (_q.get(q, 1), _lic.get(lic, 1), float(c.get("score") or 0.0))

    try:
        return sorted(cands, key=key, reverse=True)
    except Exception:  # noqa: BLE001
        return list(cands)


def merge_charts(providers: dict[str, tuple[list[dict[str, Any]], float]],
                 *, ttl_s: float = 3600.0, now: float = 0.0) -> dict[str, Any]:
    """Blend several per-nation chart providers into one chart.

    ``providers``: {provider_name: (entries, weight)}.  Each entry:
    {title, artist, nation, rank}.  Score = weight * (1 / rank).
    Entries are deduped by (title, artist); the blend is re-ranked.
    Empty providers → empty merge with an honest note (never fabricated
    chart positions).  ``now``/``ttl_s`` are accepted for cache-aware
    callers; this pure function does not cache (see PlayService.charts
    for the TTL cache).  Never raises."""
    try:
        now = now or time.time()
        scored: dict[tuple[str, str], dict[str, Any]] = {}
        for provider, (entries, weight) in (providers or {}).items():
            w = float(weight or 0.0)
            if w <= 0:
                continue
            for e in entries or []:
                title = str(e.get("title") or "").strip()
                artist = str(e.get("artist") or "").strip()
                if not title:
                    continue
                key = (title.lower(), artist.lower())
                rank = int(e.get("rank") or 0) or 999
                score = w * (1.0 / max(1, rank))
                row = scored.setdefault(key, {
                    "title": title, "artist": artist,
                    "nation": str(e.get("nation") or ""),
                    "providers": [], "weight": 0.0})
                row["weight"] += score
                if provider not in row["providers"]:
                    row["providers"].append(provider)
        merged = sorted(scored.values(), key=lambda r: -r["weight"])
        out: list[ChartEntry] = []
        for i, row in enumerate(merged, 1):
            out.append(ChartEntry(title=row["title"], artist=row["artist"],
                                  nation=row["nation"], rank=i,
                                  providers=tuple(row["providers"]),
                                  weight=round(row["weight"], 4)))
        return {"entries": out, "fetched_at": now, "ttl_s": ttl_s,
                "note": ("" if out else
                         "no chart providers returned data — no chart fabricated")}
    except Exception:  # noqa: BLE001
        return {"entries": [], "fetched_at": now or time.time(),
                "ttl_s": ttl_s, "note": "chart merge failed"}


@dataclass
class ContinuityPlan:
    """What the DJ should do next, from skip data + energy curve."""

    next_energy_target: float = 0.6
    avoid_ids: list[str] = field(default_factory=list)
    skip_rate: float = 0.0
    read_the_room: str = ""   # "cooling down" | "holding" | "pushing"
    taste_notes: list[str] = field(default_factory=list)


class PlayService:
    """The /play service layer: understanding → fallback chain → DJ
    continuity → charts.  Wraps :class:`SourceResolver` (never replaces
    it) and adds the Step-4/5 organs."""

    def __init__(self, context: Any = None, *,
                 registry: SourceRegistry | None = None,
                 extractors: ExtractorRegistry | None = None,
                 resolver: Any = None,
                 chart_providers: dict[str, Any] | None = None,
                 unlicensed_mode: str | None = None) -> None:
        self.context = context
        self.registry = registry or DEFAULT_REGISTRY
        self.extractors = extractors or DEFAULT_EXTRACTORS
        self._resolver = resolver
        self._lexicon = EkitiLexicon()
        self._chart_providers = dict(chart_providers or {})
        self._charts_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        mode = (unlicensed_mode or os.environ.get(
            "NM_PLAY_UNLICENSED", "label-only")).strip().lower()
        self.unlicensed_mode = mode if mode in UNLICENSED_MODES else "label-only"

    # ── understanding (LLM-first) ─────────────────────────────────────

    def understand(self, query: str) -> PlayIntent:
        """Parse a free-text /play query.  The brain (LLM) goes first
        when one is wired on the context; the deterministic parser is
        the fallback.  Ekiti → standard for SEARCH only; ``original``
        is preserved verbatim.  Never raises."""
        try:
            q = (query or "").strip()
            norm = self._lexicon.normalize_query(q)
            intent = PlayIntent(original=q, search_query=norm["search"],
                                mapped_terms=list(norm["mapped"]),
                                understood_by="deterministic")
            title, artist = parse_title_artist(q)
            intent.title, intent.artist = title, artist
            llm = getattr(self.context, "llm", None) if self.context else None
            parsed = None
            if llm is not None:
                try:
                    parsed = self._llm_parse(llm, q)
                except Exception:  # noqa: BLE001 — LLM is advisory
                    parsed = None
            if isinstance(parsed, dict) and parsed:
                intent.title = str(parsed.get("title") or intent.title)
                intent.artist = str(parsed.get("artist") or intent.artist)
                intent.mood = str(parsed.get("mood") or "")
                intent.nation = str(parsed.get("nation") or "")
                intent.understood_by = "llm"
            return intent
        except Exception:  # noqa: BLE001
            return PlayIntent(original=query or "",
                              search_query=query or "",
                              understood_by="deterministic")

    @staticmethod
    def _llm_parse(llm: Any, query: str) -> dict[str, Any] | None:
        """Ask the brain to parse the query into JSON.  Returns None on
        any failure — the deterministic path takes over.  The LLM output
        is DATA: it is parsed as JSON, never executed, and its fields
        are only ever used as search strings."""
        try:
            import json as _json
            prompt = (
                "Parse this music request into JSON with keys title, "
                "artist, mood, nation (ISO code or empty). "
                "Return ONLY JSON, no prose.\n"
                f"Request: {query[:200]}")
            out = None
            if hasattr(llm, "complete"):
                out = llm.complete(prompt)
            elif hasattr(llm, "generate"):
                out = llm.generate(prompt)
            elif callable(llm):
                out = llm(prompt)
            if isinstance(out, dict):
                return out
            text = str(out or "")
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                parsed = _json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            return None
        except Exception:  # noqa: BLE001
            return None

    # ── resolution: the fallback chain ────────────────────────────────

    def resolve(self, query: str) -> PlayReport:
        """Resolve a /play query through the full fallback chain.

        1. Understand (LLM-first).
        2. SourceResolver (local → yt-dlp → SoundCloud → text search →
           Spotify) — the existing chain, reused, not reimplemented.
        3. Extractor descriptors for URL queries — label licence +
           downloadability honestly.
        4. Per-nation charts as a discovery fallback ("best available").
        Never a dead end: the report always lists what was tried + why."""
        report = PlayReport(best_available=(query or "").strip())
        try:
            intent = self.understand(query)
            resolver = self._resolver
            if resolver is None:
                try:
                    from .resolver import SourceResolver
                    resolver = SourceResolver(self.context)
                except Exception as exc:  # noqa: BLE001
                    report.tried.append(PlayAttempt(
                        strategy="resolver", detail=f"unavailable: {exc}"))
                    resolver = None
            if resolver is not None:
                try:
                    res = resolver.resolve(intent.search_query or query)
                except Exception as exc:  # noqa: BLE001
                    res = None
                    report.tried.append(PlayAttempt(
                        strategy="resolver", detail=f"crashed: {exc}"))
                if res is not None:
                    if getattr(res, "ok", False):
                        report.ok = True
                        report.path_or_url = res.path_or_url
                        report.title = res.title or intent.original
                        report.best_available = report.title
                        report.licence = self._licence_for(res)
                        report.tried.append(PlayAttempt(
                            strategy="resolver", ok=True,
                            detail=f"resolved via {res.source_name}",
                            licence=report.licence))
                        return report
                    for att in (res.attempts or []):
                        report.tried.append(PlayAttempt(
                            strategy="resolver", detail=str(att)))
                    for err in (res.errors or []):
                        report.tried.append(PlayAttempt(
                            strategy="resolver", detail=str(err)))
            # URL queries: extractor descriptors label what this URL is.
            q = (query or "").strip()
            if q.lower().startswith(("http://", "https://")):
                descs = self.extractors.for_url(q)
                if descs:
                    best = descs[0]
                    if not best.downloadable:
                        report.tried.append(PlayAttempt(
                            strategy=f"extractor:{best.name}",
                            detail=f"{best.notes} — not downloadable",
                            licence=best.licence.value))
                        report.why_failed = (
                            f"{best.name} serves no downloadable stream")
                        report.hint = best.notes
                        return report
                    report.tried.append(PlayAttempt(
                        strategy=f"extractor:{best.name}", ok=True,
                        detail=f"extraction rule {best.extraction_rule}",
                        licence=best.licence.value))
                    # Extractor says downloadable but the resolver already
                    # failed — report it honestly, don't claim success.
                    report.tried.append(PlayAttempt(
                        strategy="extractor",
                        detail="extractor matched but resolver could not "
                               "fetch a stream"))
                else:
                    report.tried.append(PlayAttempt(
                        strategy="extractor",
                        detail="no extractor descriptor matches this URL"))
            # Discovery fallback: per-nation charts ("best available").
            if intent.nation:
                charts = self.charts(intent.nation)
                entries = charts.get("entries") or []
                if entries:
                    top = entries[0]
                    report.best_available = (
                        f"{top.title} — {top.artist} "
                        f"(#{top.rank} in {intent.nation} charts)")
                    report.tried.append(PlayAttempt(
                        strategy="charts",
                        detail=f"no direct match; closest chart entry: "
                               f"{report.best_available}"))
            if not report.tried:
                report.tried.append(PlayAttempt(
                    strategy="resolver",
                    detail="no strategies produced a candidate"))
            report.why_failed = ("every source failed or returned nothing "
                                 "playable for this query")
            report.hint = ("check the title/URL and try again, or pick a "
                           "chart entry: /play charts <nation>")
            return report
        except Exception as exc:  # noqa: BLE001 — never raises
            report.tried.append(PlayAttempt(
                strategy="play-service", detail=f"crashed: {exc}"))
            report.why_failed = "play service error"
            return report

    def _licence_for(self, res: Any) -> str:
        """Honest licence label for a resolver result."""
        try:
            kind = str(getattr(res, "kind", "") or "").lower()
            src = str(getattr(res, "source_name", "") or "").lower()
            if kind in ("file",) or "workspace" in src or "local" in src:
                return "licensed"  # owner's own files
            if kind == "spotify" or "spotify" in src:
                return "licensed"
            if kind == "boomplay":
                return "licensed"
            if kind == "netnaija":
                return "unlicensed"
            if kind in ("youtube", "soundcloud", "audiomack", "url"):
                return "user-upload"
            return "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    def explain_failure(self, resolution: Any) -> str:
        """Format a resolver failure as tried + why + hint (used by the
        /play chat handler).  Pure formatting — no network.  Never raises."""
        try:
            attempts = list(getattr(resolution, "attempts", None) or [])
            errors = list(getattr(resolution, "errors", None) or [])
            hint = str(getattr(resolution, "hint", "") or "")
            lines = ["tried:"]
            for a in attempts:
                lines.append(f"  {a}")
            for e in errors:
                lines.append(f"  error: {e}")
            if hint:
                lines.append(f"hint: {hint}")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return "tried: (no attempt log)"

    # ── charts ────────────────────────────────────────────────────────

    def charts(self, nation: str, *, ttl_s: float = 3600.0) -> dict[str, Any]:
        """Per-nation charts, merged with weights + TTL cache.  Providers
        are injected callables ``nation -> (entries, weight)``.  Never
        raises; empty providers → honest empty merge."""
        try:
            key = (nation or "").upper()
            now = time.time()
            cached = self._charts_cache.get(key)
            if cached and now - cached[0] < ttl_s:
                return cached[1]
            providers: dict[str, tuple[list[dict[str, Any]], float]] = {}
            for name, fn in self._chart_providers.items():
                try:
                    entries, weight = fn(key)
                    providers[name] = (list(entries or []), float(weight or 0))
                except Exception as exc:  # noqa: BLE001 — one dead provider
                    _log.info("charts: provider %s failed: %s", name, exc)
            merged = merge_charts(providers, ttl_s=ttl_s, now=now)
            self._charts_cache[key] = (now, merged)
            return merged
        except Exception:  # noqa: BLE001
            return {"entries": [], "fetched_at": time.time(), "ttl_s": ttl_s,
                    "note": "charts unavailable"}

    # ── DJ continuity ─────────────────────────────────────────────────

    def dj_continuity(self, *, feedback_rows: list[dict[str, Any]] | None = None,
                      played_ids: list[str] | None = None,
                      current_energy: float = 0.6,
                      window: int = 10) -> ContinuityPlan:
        """Close the taste loop on dj_engine skip data.

        * skip-rate over the recent window → read-the-room: a run of
          skips cools the energy target; keeps push the target up.
        * no-repeat: ``played_ids`` are avoided (the caller passes the
          recent queue; this plan names them back as ``avoid_ids``).
        * energy curve: the target is snapped to the dj_engine curve
          shape for the show position (warm-up → peak → cool-down).
        ``feedback_rows`` are ``dj_engine.log_feedback`` JSONL rows
        (event kept/skipped/liked).  Never raises."""
        try:
            rows = list(feedback_rows or [])[-max(1, window):]
            skips = sum(1 for r in rows if str(r.get("event")) == "skipped")
            keeps = sum(1 for r in rows
                        if str(r.get("event")) in ("kept", "liked"))
            total = skips + keeps
            skip_rate = (skips / total) if total else 0.0
            if skip_rate >= 0.6:
                room, target = "cooling down", max(0.2, current_energy - 0.25)
                notes = ["skip run — cooling the energy, switching lanes"]
            elif skip_rate <= 0.2 and keeps >= 3:
                room, target = "pushing", min(0.95, current_energy + 0.10)
                notes = ["crowd is with it — pushing the energy up"]
            else:
                room, target = "holding", current_energy
                notes = ["holding the current energy"]
            # Snap to the dj_engine energy curve when available.
            try:
                from .dj_engine import energy_curve
                curve = energy_curve("peak", 10)
                snapped = curve[min(9, max(0, int(target * 10)))]
                target = round((target + snapped) / 2.0, 3)
            except Exception:  # noqa: BLE001 — curve is advisory
                pass
            return ContinuityPlan(
                next_energy_target=round(max(0.1, min(1.0, target)), 3),
                avoid_ids=list(played_ids or []),
                skip_rate=round(skip_rate, 3),
                read_the_room=room,
                taste_notes=notes)
        except Exception:  # noqa: BLE001
            return ContinuityPlan(next_energy_target=current_energy,
                                  avoid_ids=list(played_ids or []))
