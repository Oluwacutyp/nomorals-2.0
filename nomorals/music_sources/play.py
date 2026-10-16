"""Registry-backed /play path: search → resolve → download → enrich.

``play_via_registry`` is the spine that connects the music registry
to playback. Every step is injected, so the function is testable
without network and honest without wiring:

* ``search_fn(query)`` — metadata search, returns candidate mappings
  (each optionally naming a ``source_id``). No search wired →
  ``status="no_search"`` with the registry-browsing hint, never a
  fabricated candidate.
* ``download_fn(source, query)`` — defaults to
  :func:`downloader.download` with no fetcher (plan-only). A caller
  may inject a real fetcher-backed download.
* ``enrich_chain`` — an :class:`enrichment.EnrichmentChain`; without
  one, enrichment is recorded as ``missing`` and playback proceeds
  with the download result alone.

Every :class:`PlayResult` carries the source's licence label and its
quality ceiling — licence is shown on every /play result, never
hidden. Purchase sources resolve to a checkout plan, never to a
claimed download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .downloader import DownloadResult, download as _download
from .enrichment import EnrichmentChain, EnrichmentResult
from .registry import MusicSource, by_class, by_id

__all__ = ["PlayResult", "play_via_registry", "registry_play_text"]


@dataclass
class PlayResult:
    query: str
    status: str = ""  # ready | plan_only | checkout | no_search | no_source | error
    source_id: Optional[str] = None
    source_name: str = ""
    licence: str = ""
    quality_ceiling: str = ""
    detail: str = ""
    download: Optional[DownloadResult] = None
    enrichment: Optional[EnrichmentResult] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)


def play_via_registry(
    query: str,
    *,
    search_fn: Optional[Callable[[str], Any]] = None,
    download_fn: Optional[Callable[[MusicSource, str], DownloadResult]] = None,
    enrich_chain: Optional[EnrichmentChain] = None,
    source_id: Optional[str] = None,
) -> PlayResult:
    """Resolve ``query`` through the registry to a playable/downloadable result."""
    q = (query or "").strip()
    if not q:
        return PlayResult(query=query, status="error", detail="empty query")
    # 1. Source selection: explicit id wins, else search, else metadata fallback.
    source: Optional[MusicSource] = None
    candidates: List[Dict[str, Any]] = []
    if source_id:
        source = by_id(source_id)
        if source is None:
            return PlayResult(query=q, status="no_source",
                              detail=f"unknown music source {source_id!r}")
    elif search_fn is not None:
        try:
            found = search_fn(q)
        except Exception as exc:  # noqa: BLE001 — search failure is data
            return PlayResult(query=q, status="error",
                              detail=f"search failed: {type(exc).__name__}: {exc}")
        if isinstance(found, dict):
            candidates = [found]
        elif isinstance(found, list):
            candidates = [c for c in found if isinstance(c, dict)]
        for candidate in candidates:
            named = candidate.get("source_id") or candidate.get("source")
            if isinstance(named, str) and by_id(named) is not None:
                source = by_id(named)
                break
        if source is None and candidates:
            # Candidate names no registry source — fall back to the
            # first metadata source so the licence ladder still applies.
            metas = by_class("metadata")
            source = metas[0] if metas else None
    else:
        return PlayResult(
            query=q, status="no_search",
            detail=(
                "no metadata search wired — browse with "
                "`nm music sources` / `/music sources`, then "
                "`/play registry <query> --source <id>`."
            ),
        )
    if source is None:
        return PlayResult(query=q, status="no_source",
                          detail="no registry source resolved for this query")
    # 2. Download (plan or fetch) via the per-class downloader.
    if download_fn is not None:
        try:
            dl = download_fn(source, q)
        except Exception as exc:  # noqa: BLE001 — download failure is data
            return PlayResult(query=q, status="error", source_id=source.id,
                              source_name=source.name, licence=source.licence,
                              quality_ceiling=source.quality_ceiling,
                              detail=f"download failed: {type(exc).__name__}: {exc}",
                              candidates=candidates)
    else:
        dl = _download(source, query=q)
    # 3. Enrichment (optional chain; missing chain = recorded, not fatal).
    enrichment: Optional[EnrichmentResult] = None
    if enrich_chain is not None:
        try:
            enrichment = enrich_chain.resolve(q)
        except Exception:  # noqa: BLE001 — enrichment never sinks playback
            enrichment = None
    if dl.action == "checkout":
        status = "checkout"
    elif dl.downloaded:
        status = "ready"
    else:
        status = "plan_only"
    return PlayResult(
        query=q, status=status, source_id=source.id, source_name=source.name,
        licence=source.licence, quality_ceiling=source.quality_ceiling,
        detail=dl.detail, download=dl, enrichment=enrichment,
        candidates=candidates,
    )


def registry_play_text(result: PlayResult) -> str:
    """Render a PlayResult for chat/CLI — licence always shown."""
    if result.status in ("error", "no_search", "no_source"):
        return f"🎵 {result.detail}"
    lines = [
        f"🎵 {result.query}",
        f"  source: {result.source_name} ({result.source_id})",
        f"  licence: {result.licence}",
    ]
    if result.quality_ceiling:
        lines.append(f"  quality: up to {result.quality_ceiling}")
    if result.status == "checkout":
        lines.append("  action: buy in the store — checkout plan, never auto-downloaded.")
    elif result.status == "ready":
        lines.append("  action: downloaded via wired fetcher — ready to play.")
    else:
        lines.append("  action: plan only — no fetcher wired, nothing downloaded.")
    if result.detail:
        lines.append(f"  detail: {result.detail}")
    if result.enrichment is not None:
        missing = ", ".join(result.enrichment.missing) or "none"
        lines.append(f"  enrichment: missing steps: {missing}")
    return "\n".join(lines)
