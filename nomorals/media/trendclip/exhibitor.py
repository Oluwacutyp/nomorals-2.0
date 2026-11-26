"""TrendClip museum exhibitor — rendered clips land as museum artifacts
with provenance, following ``downloads_exhibitor``'s shape.

One card per export package (``<trendclip root>/exports/*/
provenance.json``): the clip's title, its source URL, its rights label,
its moment and both rungs' picks. A missing root exhibits nothing, never
an error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["trendclip_exhibitor"]

_HASH_MAX_BYTES = 64 * 1024 * 1024


def trendclip_exhibitor(root: Path | str) -> list[dict[str, Any]]:
    """Cards for every rendered trendclip export package under ``root``."""
    from ...museum import museum_card

    base = Path(root)
    if not base.is_dir():
        return []
    exports = base / "exports"
    if not exports.is_dir():
        return []
    cards: list[dict[str, Any]] = []
    for prov_path in sorted(exports.glob("*/provenance.json")):
        try:
            provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        package_dir = prov_path.parent
        clip_path = base / "clips" / (package_dir.name + ".mp4")
        if not clip_path.is_file():
            found = list(package_dir.glob("*.mp4"))
            clip_path = found[0] if found else None
        if clip_path is None or not clip_path.is_file():
            continue
        try:
            size = clip_path.stat().st_size
        except OSError:
            size = 0
        digest = ""
        if 0 < size <= _HASH_MAX_BYTES:
            try:
                digest = hashlib.sha256(
                    clip_path.read_bytes()).hexdigest()[:16]
            except OSError:
                digest = ""
        moment = provenance.get("moment") or {}
        topic = str(provenance.get("topic") or "trendclip")
        title = f"{topic} — clip #{moment.get('n', '?')}"
        tags = ["trendclip", f"topic:{topic}"]
        rights = str(provenance.get("rights") or "")
        if rights:
            tags.append(f"rights:{rights.split(' — ')[0]}")
        cards.append(museum_card(
            id=f"trendclip:{package_dir.name}",
            title=title,
            kind="clip",
            shelf="trendclip",
            path=str(clip_path),
            hash=digest,
            size=size,
            provenance={
                "type": "trendclip-clip",
                "ref": package_dir.name,
                "tool": "trendclip",
                "job_id": str(provenance.get("job_id") or ""),
                "source_url": str(provenance.get("source_url") or ""),
                "rights": rights,
                "moment": moment,
                "destinations": list(provenance.get("destinations") or []),
            },
            tags=tags))
    return cards
