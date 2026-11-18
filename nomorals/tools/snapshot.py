"""Snapshot tool: the Archive.ph rung for the brain.

"Snapshot this page" over the snapshot organ — not a standalone
fetcher. The organ (:mod:`nomorals.browser.snapshot`) owns the
capture, the seed sources, the hashes, the store and the prune;
this module is only the surface between the brain and the organ.

The receipt is the point: every capture answers with its content
hash and render hash, and ``action=cite`` hands the brain the
snapshot's text plus its "as captured on <date>" line, so a
research answer can cite her own capture when the live page has
since changed — the snapshot text registers into a
:class:`~nomorals.research.citations.CitationManager` exactly
like any other source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..browser.snapshot import (
    SnapshotStore,
    capture,
    cite_line,
    format_snapshot_reply,
    prune,
    seed,
    set_retention,
    snapshot_text,
    verify,
)
from ..core.policy import Capability

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store_of(context: Any) -> SnapshotStore:
    home = _home_of(context)
    return SnapshotStore(home) if home else SnapshotStore()


def _find(store: SnapshotStore, ref: str) -> Any | None:
    ref = (ref or "").strip()
    if not ref:
        return None
    snap = store.get(ref)
    if snap is not None:
        return snap
    if ref.startswith(("http://", "https://")):
        return store.find_by_url(ref)
    for candidate in store.list():
        if candidate.id.startswith(ref):
            return candidate
    return None


def register(registry: Any) -> None:
    """Attach the ``snapshot`` tool."""
    context = registry.context

    @registry.register(
        "snapshot",
        description=(
            "Snapshot a web page as it rendered and keep the receipt: "
            "action=capture takes url=<page> and saves the rendered DOM "
            "+ asset bundle + screenshot into her own store, answering "
            "with the content hash and render hash. action=seed takes "
            "url=<page> and source=wayback | archive.ph and records a "
            "page she did not capture herself as a reference copy "
            "(labelled, never sole evidence). action=show takes "
            "id=<snapshot id or URL> and returns the receipt; "
            "action=cite takes id=<snapshot id> and returns the "
            "snapshot text plus its 'as captured on <date>' line for "
            "research citations. action=list lists saved snapshots; "
            "action=verify takes id=<snapshot id> and re-hashes the "
            "stored files; action=prune drops snapshots older than "
            "days=<n> (default: her retention setting); action=status "
            "reports how many snapshots she holds, how much disk they "
            "use, and the retention setting. action=retention takes "
            "days=<n> and sets how long snapshots are kept."
        ),
        capability=Capability.NET_BROWSER,
        result_protocol="dict_ok",
    )
    def _snapshot_tool(**kwargs: Any) -> dict[str, Any]:
        action = str(kwargs.get("action", "capture") or "capture").lower()
        store = _store_of(context)
        if action == "capture":
            url = str(kwargs.get("url", "") or "").strip()
            if not url:
                return {"ok": False,
                        "error": "capture needs url=<page>"}
            try:
                snap = capture(url, store=store)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "snapshot": snap.to_dict(),
                    "receipt": format_snapshot_reply(snap),
                    "cite": cite_line(snap)}
        if action == "seed":
            url = str(kwargs.get("url", "") or "").strip()
            source = str(kwargs.get("source", "") or "").strip()
            if not url or not source:
                return {"ok": False,
                        "error": "seed needs url=<page> and "
                                 "source=wayback | archive.ph"}
            try:
                snap = seed(url, source=source, store=store)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "snapshot": snap.to_dict(),
                    "receipt": format_snapshot_reply(snap),
                    "cite": cite_line(snap)}
        if action == "show":
            snap = _find(store, str(kwargs.get("id", "") or ""))
            if snap is None:
                return {"ok": False,
                        "error": "no snapshot matches that id or URL"}
            return {"ok": True, "snapshot": snap.to_dict(),
                    "receipt": format_snapshot_reply(snap),
                    "cite": cite_line(snap)}
        if action == "cite":
            snap = _find(store, str(kwargs.get("id", "") or ""))
            if snap is None:
                return {"ok": False,
                        "error": "no snapshot matches that id or URL"}
            dom_path = Path(snap.dom_path)
            text = (snapshot_text(dom_path.read_text(
                        encoding="utf-8", errors="replace"))
                    if dom_path.is_file() else "")
            return {"ok": True, "id": snap.id, "url": snap.url,
                    "cite": cite_line(snap), "text": text}
        if action == "list":
            rows = []
            for snap in store.list():
                rows.append({
                    "id": snap.id, "url": snap.url,
                    "captured_iso": snap.captured_iso,
                    "kind": snap.kind, "source": snap.source,
                    "rendered": snap.rendered,
                    "content_hash": snap.content_hash[:16] + "…",
                    "assets": sum(1 for a in snap.assets
                                  if a.get("ok")),
                    "assets_missing": sum(
                        1 for a in snap.assets if not a.get("ok")),
                })
            return {"ok": True, "snapshots": rows,
                    "count": len(rows),
                    "retention_days": store.retention_days}
        if action == "verify":
            return verify(store, str(kwargs.get("id", "") or ""))
        if action == "prune":
            raw_days = kwargs.get("days")
            days = None
            if raw_days is not None:
                try:
                    days = int(raw_days)
                except (TypeError, ValueError):
                    return {"ok": False,
                            "error": "prune days=<n> must be a number"}
            return prune(store, older_than_days=days)
        if action == "retention":
            raw_days = kwargs.get("days")
            try:
                days = int(raw_days)
            except (TypeError, ValueError):
                return {"ok": False,
                        "error": "retention needs days=<n>"}
            try:
                return set_retention(store, days)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        if action == "status":
            total_bytes = 0
            for snap in store.list():
                snap_dir = store.snapshot_dir(snap.id)
                if snap_dir.is_dir():
                    total_bytes += sum(
                        p.stat().st_size for p in snap_dir.rglob("*")
                        if p.is_file())
            return {"ok": True, "count": len(store.list()),
                    "bytes": total_bytes,
                    "retention_days": store.retention_days,
                    "store": str(store.path)}
        return {"ok": False,
                "error": "unknown action — capture | seed | show | "
                         "cite | list | verify | prune | retention | "
                         "status"}
