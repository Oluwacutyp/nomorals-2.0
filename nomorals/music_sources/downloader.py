"""Downloader plans per registry class — legal sources only.

A download plan is honest about what automation can and cannot do:

* ``purchase`` → action ``checkout``: the file is bought, never
  auto-downloaded. Steps tell the owner to buy in the store and keep
  the receipt. No store URL is fabricated — the plan names the store
  and the query, never a made-up link.
* ``pool`` → action ``membership``: download via the member library
  under the pool's terms; honour rate limits.
* ``open`` / ``artist_enabled`` / ``dataset`` → action ``fetch``:
  bytes come from an injected ``fetch_fn``. With no fetcher wired,
  the plan is returned with ``downloaded=False`` and the honest
  reason — never a claimed download.
* ``metadata`` / ``preview`` → action ``metadata_only``: no file
  download exists for this class; metadata/previews only.

Every plan and every result carries the source's licence label.
Licence is never hidden, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .registry import MusicSource, by_id

__all__ = ["DownloadPlan", "DownloadResult", "plan_download", "download"]


@dataclass
class DownloadPlan:
    source_id: str
    source_name: str
    action: str  # checkout | membership | fetch | metadata_only
    licence: str
    quality_ceiling: str = ""
    steps: List[str] = field(default_factory=list)
    url: Optional[str] = None  # always None: no URL is ever fabricated
    detail: str = ""


@dataclass
class DownloadResult:
    source_id: str
    licence: str
    downloaded: bool
    action: str
    detail: str = ""
    payload: Any = None
    plan: Optional[DownloadPlan] = None


def plan_download(source: MusicSource, *, query: str = "") -> DownloadPlan:
    """Build the per-class plan for ``source``. No network, no fetch."""
    q = (query or "").strip()
    base = DownloadPlan(
        source_id=source.id, source_name=source.name,
        action="metadata_only", licence=source.licence,
        quality_ceiling=source.quality_ceiling,
    )
    if source.source_class == "purchase":
        base.action = "checkout"
        base.steps = [
            f"Open {source.name} and search for {q!r}." if q else f"Open {source.name}.",
            "Buy the track/album — DRM-free purchase, keep the receipt.",
            f"Download the file ({', '.join(source.formats) or 'store formats'}).",
            "No URL is stored here — open the store; never trust a fabricated link.",
        ]
        base.detail = "Purchase sources are bought, never auto-downloaded."
    elif source.source_class == "pool":
        base.action = "membership"
        base.steps = [
            f"Membership required: join {source.name} under its terms.",
            f"Download via the member library (search {q!r})." if q else "Download via the member library.",
            "Honour the pool's rate limits and terms; extended edits / intros / acapellas live here.",
        ]
        base.detail = "Pool downloads go through the member library."
    elif source.source_class in ("open", "artist_enabled", "dataset"):
        base.action = "fetch"
        base.steps = [
            f"Fetch {q!r} from {source.name} via its API / artist-enabled download." if q
            else f"Fetch from {source.name} via its API / artist-enabled download.",
            f"Etiquette: {source.etiquette}" if source.etiquette else "Fetch politely.",
        ]
        base.detail = "Fetch needs a wired fetcher; without one this is a plan only."
    else:  # metadata / preview
        base.action = "metadata_only"
        base.steps = [
            f"Use {source.name} for metadata / previews ({q!r})." if q
            else f"Use {source.name} for metadata / previews.",
            "No file download exists for this class — metadata and previews only.",
        ]
        base.detail = "Metadata/preview sources do not serve files."
    return base


def download(
    source: MusicSource,
    *,
    query: str = "",
    fetch_fn: Optional[Callable[[MusicSource, str], Any]] = None,
) -> DownloadResult:
    """Execute a download plan.

    With no ``fetch_fn`` wired, returns the plan with
    ``downloaded=False`` — never a claimed download. With a fetcher,
    the fetcher's return value is carried as ``payload`` and the
    licence label travels with it.
    """
    plan = plan_download(source, query=query)
    if plan.action in ("checkout", "membership"):
        return DownloadResult(
            source_id=source.id, licence=source.licence, downloaded=False,
            action=plan.action,
            detail=f"{plan.detail} Steps: {' '.join(plan.steps)}",
            plan=plan,
        )
    if plan.action == "metadata_only":
        return DownloadResult(
            source_id=source.id, licence=source.licence, downloaded=False,
            action=plan.action, detail=plan.detail, plan=plan,
        )
    # fetch class
    if fetch_fn is None:
        return DownloadResult(
            source_id=source.id, licence=source.licence, downloaded=False,
            action="fetch",
            detail="no fetcher wired — plan only, nothing downloaded",
            plan=plan,
        )
    try:
        payload = fetch_fn(source, query)
    except Exception as exc:  # noqa: BLE001 — fetch failure is data
        return DownloadResult(
            source_id=source.id, licence=source.licence, downloaded=False,
            action="fetch", detail=f"fetch failed: {type(exc).__name__}: {exc}",
            plan=plan,
        )
    if payload is None:
        return DownloadResult(
            source_id=source.id, licence=source.licence, downloaded=False,
            action="fetch", detail="fetcher returned nothing", plan=plan,
        )
    return DownloadResult(
        source_id=source.id, licence=source.licence, downloaded=True,
        action="fetch", detail="fetched via wired fetcher", payload=payload,
        plan=plan,
    )


def plan_for(source_id: str, *, query: str = "") -> DownloadPlan:
    """Plan for a registry id, or raise KeyError on an unknown id."""
    source = by_id(source_id)
    if source is None:
        raise KeyError(f"unknown music source {source_id!r}")
    return plan_download(source, query=query)
