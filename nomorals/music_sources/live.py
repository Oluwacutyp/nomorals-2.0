"""Live collaborators for the registry play path (doc 7 wiring).

``play.play_via_registry`` takes injected ``search_fn`` /
``download_fn`` steps.  The production call sites (``/play registry``
and ``/music registry-play``) originally injected NOTHING, so the
branch answered ``no_search`` / ``plan_only`` — metadata and plans
presented as results.  This module builds the real steps from a
runtime context, sharing the exact services ordinary ``/play`` uses:

* search — :class:`nomorals.media.resolver.SourceResolver`'s
  candidate sweep (the same strategies, the same audit posture);
* retrieval — :class:`nomorals.media.playback.PlaybackEngine`'s
  ``download`` (the same per-kind downloaders, cache and honest
  DRM refusals), with fall-through across candidates exactly like
  the chat fall-through sender: a candidate whose download fails —
  or whose file the full-track verdict rejects as a preview-length
  clip — is a failed ROUTE, and the next candidate is tried;
* verification — :func:`nomorals.media.library.full_track_verdict`,
  three-valued: verified / rejected / unverified.  The verdict
  rides the result payload so the delivering surface labels an
  unverified delivery as unverified, never as proof.

Registry framing stays honest: a candidate whose resolver kind IS a
registry source id names that source (exact attribution); the other
candidates ride the helper's metadata fallback for the licence
ladder, and the download detail names the route that actually
served.  A source EXPLICITLY picked with ``--source`` keeps its own
semantics — purchase stays a checkout, metadata stays a plan,
produced by the registry's own per-class downloader.  A source the
SEARCH selected (a named candidate's source, or the metadata
fallback) frames licence/attribution only: the journey fetches
through the resolver's real candidates, because the track the user
asked for exists there — stopping at the fallback source's plan
would re-create the very defect this module exists to fix.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .downloader import DownloadResult
from .downloader import download as _plan_download
from .registry import MusicSource, by_id

__all__ = ["live_collaborators"]

#: Classes an EXPLICITLY picked source may fetch per-track.  Every
#: other class keeps its plan when named with --source (see module
#: docstring); search-selected journeys fetch via the candidates.
_FETCH_CLASSES = ("open", "artist_enabled")

#: Classes that must never be auto-downloaded under a candidate's
#: naming — purchase and pool rows keep their plan semantics even
#: when a resolver candidate kind happens to share the id.
_NO_AUTOFETCH_CLASSES = ("purchase", "pool")


def _registry_id_for_kind(kind: str) -> Optional[str]:
    """The registry source a resolver candidate kind names, if any.

    Exact id match only, and never a purchase/pool row: those
    classes resolve to plans, not fetches."""
    if not kind or ":" in kind:
        return None
    source = by_id(kind)
    if source is None or source.source_class in _NO_AUTOFETCH_CLASSES:
        return None
    return source.id


class _LiveJourney:
    """One wired registry journey: search once, download with
    fall-through.  State is per-journey (the two closures share it);
    a fresh pair is built per command invocation."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self._candidates: Optional[List[Any]] = None
        self._engine: Any = None

    # ── search step ──────────────────────────────────────────────

    def _search(self, query: str) -> List[Any]:
        from ..media.resolver import SourceResolver

        cands = SourceResolver(self.context).search_candidates(
            query, limit=8)
        self._candidates = list(cands or [])
        return self._candidates

    def search(self, query: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cand in self._search(query):
            row: Dict[str, Any] = {
                "title": cand.title or "",
                "artist": cand.artist or "",
                "kind": cand.kind or "",
                "duration": cand.duration or 0.0,
            }
            named = _registry_id_for_kind(str(cand.kind or ""))
            if named:
                row["source_id"] = named
            out.append(row)
        return out

    # ── download step ────────────────────────────────────────────

    def _engine_or_none(self) -> Any:
        if self._engine is None:
            from ..media.playback import PlaybackEngine

            self._engine = PlaybackEngine(self.context)
        return self._engine

    def download(self, source: MusicSource,
                 query: str) -> DownloadResult:
        from ..media.download_orchestrator import DownloadOrchestrator

        if self._candidates is None:
            # Explicit --source pick (the helper never ran search):
            # the named source's own semantics rule — a purchase is
            # a checkout, a metadata source serves no file.  Only
            # fetch classes go on to search-and-fetch below.
            if source.source_class not in _FETCH_CLASSES:
                return _plan_download(source, query=query)
            self._search(query)
        elif not self._candidates:
            # Search ran and returned nothing fetchable: the
            # selected source's own per-class result stands.
            return _plan_download(source, query=query)
        cands = list(self._candidates or [])
        # The selected source's own candidates first, then the rest
        # in search order — fall-through mirrors ordinary /play.
        preferred = [c for c in cands
                     if _registry_id_for_kind(str(c.kind or ""))
                     == source.id]
        ordered = preferred + [c for c in cands if c not in preferred]
        engine = self._engine_or_none()
        failures: List[str] = []
        fetchable = []
        for cand in ordered:
            if getattr(cand, "preview_only", False) \
                    or getattr(cand, "downloadable", True) is False:
                failures.append(
                    f"{cand.kind or 'source'}: preview/protected — "
                    "not a deliverable track")
                continue
            fetchable.append(cand)
        # The candidate loop itself is the shared orchestrator
        # (Phase 2): retrieve → per-type acceptance (the song
        # contract carries the full-track verdict) → next candidate
        # on any failure, reasons journaled in its attempt story.
        landed: dict[int, dict[str, Any]] = {}

        def _retrieve(cand: Any) -> tuple[str | None, str]:
            item = {
                "kind": cand.kind or "",
                "path": cand.path_or_url or "",
                "title": cand.title or "",
                "artist": cand.artist or "",
                "duration": cand.duration or 0.0,
            }
            try:
                dl = engine.download(item)
            except Exception as exc:  # noqa: BLE001 - a route failure
                return None, str(exc)
            if not dl.get("ok"):
                return None, str(dl.get("reason", "download failed"))
            landed[id(cand)] = dl
            return str(dl.get("path") or ""), ""

        result = DownloadOrchestrator().run(
            fetchable, _retrieve, kind="song",
            label_of=lambda c: str(c.kind or "source"),
            expected_duration_of=lambda c: float(c.duration or 0.0),
            manage_sources=False)
        failures.extend(
            f"{a.label}: {a.reason}" for a in result.attempts if not a.ok)
        if result.ok and result.candidate is not None:
            cand = result.candidate
            dl = landed.get(id(cand), {})
            acc = result.acceptance
            state = str((acc.facts.get("verify_state") if acc else None)
                        or "unverified")
            why = str((acc.facts.get("verify_why") if acc else None)
                      or "")
            served = (getattr(cand, "source_name", "") or cand.kind
                      or "a wired source")
            payload = {
                "path": result.path,
                "title": str(dl.get("title") or cand.title or ""),
                "artist": str(dl.get("artist") or cand.artist or ""),
                "duration": float(cand.duration or 0.0),
                "served_via": served,
                "verify_state": state,
                "verify_why": why,
            }
            return DownloadResult(
                source_id=source.id, licence=source.licence,
                downloaded=True, action="fetch", attempted=True,
                detail=(f"fetched via {served} — full-track "
                        f"{state} ({why})"),
                payload=payload)
        detail = "no route delivered the track"
        if failures:
            detail += ": " + "; ".join(failures[:4])
        return DownloadResult(
            source_id=source.id, licence=source.licence,
            downloaded=False, action="fetch", attempted=True,
            detail=detail)


def live_collaborators(
    context: Any,
) -> Tuple[Optional[Callable[[str], Any]],
           Optional[Callable[[MusicSource, str], DownloadResult]]]:
    """Build the (search_fn, download_fn) pair ``play_via_registry``
    consumes, wired to the real resolver/downloader services for
    ``context``.  A fresh pair per invocation — the closures share
    one journey's candidate state.

    With NO context there is nothing real to wire — the playback
    engine requires a database-carrying context by design — so the
    pair is (None, None) and the helper answers with its honest
    unwired floor instead of a fabricated journey."""
    if context is None:
        return None, None
    journey = _LiveJourney(context)
    return journey.search, journey.download
