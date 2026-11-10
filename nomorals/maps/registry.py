"""The Atlas release registry: what packs this host holds (doc 21 §3).

Doc 21 superseded doc 19's per-source TileStore URL registry
(S2) with a *release* registry: Devon-built packs, pinned by
digest, with their signature state graded honestly.  The local
form is one JSON file in the packs directory — ``releases.json`` —
mapping pack_id → {digest, signature_state, built_at, source,
source_sha256, tile_format, zooms}.  M2 writes entries at install
time; the signature state comes from the M1 manifest verdict, so
an unsigned pack reads ``"unsigned"`` here forever — this file
never upgrades a state the manifest did not earn.  (A signing
mechanism does not exist in the repo today; when M-later adds
one, it writes through this same registry.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["REGISTRY_NAME", "ReleaseEntry", "ReleaseRegistry"]

#: The registry filename inside the packs directory.
REGISTRY_NAME = "releases.json"


@dataclass(frozen=True)
class ReleaseEntry:
    """One installed release, as the registry records it."""

    pack_id: str
    digest: str
    signature_state: str
    built_at: str
    source: str
    source_sha256: str = ""
    tile_format: str = ""
    zoom_min: int = 0
    zoom_max: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"pack_id": self.pack_id, "digest": self.digest,
                "signature_state": self.signature_state,
                "built_at": self.built_at, "source": self.source,
                "source_sha256": self.source_sha256,
                "tile_format": self.tile_format,
                "zoom_min": int(self.zoom_min),
                "zoom_max": int(self.zoom_max)}

    @classmethod
    def from_dict(cls, data: dict) -> ReleaseEntry:
        return cls(
            pack_id=str(data.get("pack_id", "")),
            digest=str(data.get("digest", "")),
            signature_state=str(data.get("signature_state", "unsigned")),
            built_at=str(data.get("built_at", "")),
            source=str(data.get("source", "")),
            source_sha256=str(data.get("source_sha256", "")),
            tile_format=str(data.get("tile_format", "")),
            zoom_min=int(data.get("zoom_min", 0)),
            zoom_max=int(data.get("zoom_max", 0)))


class ReleaseRegistry:
    """Read/write the packs directory's ``releases.json``.

    Writes are atomic (tmp file + replace) and the file is
    re-read on every access, so two processes never act on a
    stale in-memory copy.
    """

    def __init__(self, packs_dir: Path | str) -> None:
        self.path = Path(packs_dir) / REGISTRY_NAME

    def _read(self) -> dict[str, ReleaseEntry]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, ReleaseEntry] = {}
        for pack_id, payload in data.items():
            if isinstance(payload, dict):
                out[str(pack_id)] = ReleaseEntry.from_dict(payload)
        return out

    def get(self, pack_id: str) -> ReleaseEntry | None:
        """The entry for one pack, or None when not installed."""
        return self._read().get(pack_id)

    def all(self) -> tuple[ReleaseEntry, ...]:
        """Every installed release, ordered by pack_id."""
        entries = self._read()
        return tuple(entries[key] for key in sorted(entries))

    def record(self, entry: ReleaseEntry) -> ReleaseEntry:
        """Insert or replace one pack's entry; returns the entry."""
        entries = self._read()
        entries[entry.pack_id] = entry
        payload = {key: value.to_dict()
                   for key, value in entries.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(tmp, self.path)
        return entry
