"""Attribution DB — the SHA-256 provenance record.

Port of the *design* of the pdfcraft/wordcraft attribution DBs
(lanes 99/100): one entry per asset, each with its source,
licence, and the SHA-256 of the file as read.
``verify_attribution`` recomputes the hashes against the upstream
clone on disk, so provenance is checkable at runtime, not a
comment. Entries come in two kinds:

* ``upstream`` — a file in the storytold/vectorcraft clone this
  port's design derives from; ``sha256`` is pinned and verified.
* ``devon`` — a file this lane created in Devon's tree; it is not
  hash-pinned (it is ours and it changes), but it must name the
  upstream units it derives from in ``derives_from`` — the check
  is that the lineage is written down, not that a hash matches.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from .errors import VectorError

__all__ = ["DEFAULT_ATTRIBUTION_PATH", "DEFAULT_CLONE_ROOT",
           "load_attribution", "verify_attribution"]

DEFAULT_ATTRIBUTION_PATH = Path(__file__).resolve().parent.parent \
    / "vector-attribution.toml"
DEFAULT_CLONE_ROOT = Path.home() / "workspace" / "echelon-mining" \
    / "repos" / "storytold" / "vectorcraft"


def load_attribution(path: str | Path | None = None
                     ) -> dict[str, Any]:
    """Load an attribution TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_ATTRIBUTION_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise VectorError(
            f"cannot read attribution TOML {path}: {exc}",
            kind="io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise VectorError(f"bad attribution TOML {path}: {exc}",
                          kind="parse") from exc
    if not isinstance(data.get("assets"), list) or not data["assets"]:
        raise VectorError(
            f"attribution TOML {path} has no [[assets]] rows",
            kind="parse")
    return data


def verify_attribution(path: str | Path | None = None, *,
                       clone_root: str | Path | None = None
                       ) -> dict[str, Any]:
    """Verify every [[assets]] entry: upstream entries are
    hash-checked against the clone; devon entries are checked for
    a written ``derives_from`` lineage. Returns {checked, failed,
    skipped, entries}. When the clone is not on disk, upstream
    entries report ``skipped`` — never a silent pass."""
    data = load_attribution(path)
    root = Path(clone_root) if clone_root is not None \
        else DEFAULT_CLONE_ROOT
    clone_present = root.is_dir()
    entries: list[dict[str, Any]] = []
    for asset in data["assets"]:
        kind = str(asset.get("kind", "upstream"))
        asset_path = str(asset.get("path", "?"))
        if kind == "devon":
            derives = asset.get("derives_from", [])
            if derives:
                entries.append({"path": asset_path, "kind": kind,
                                "status": "noted",
                                "detail": "lineage written: "
                                          + ", ".join(derives)})
            else:
                entries.append({"path": asset_path, "kind": kind,
                                "status": "fail",
                                "detail": "no derives_from lineage"})
            continue
        if not clone_present:
            entries.append({"path": asset_path, "kind": kind,
                            "status": "skipped",
                            "detail": "clone not on disk"})
            continue
        target = root / asset_path
        if not target.exists():
            entries.append({"path": asset_path, "kind": kind,
                            "status": "fail",
                            "detail": "file missing from clone"})
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == str(asset.get("sha256", "")).lower():
            entries.append({"path": asset_path, "kind": kind,
                            "status": "verified",
                            "detail": f"sha256 {digest[:12]}…"})
        else:
            entries.append({"path": asset_path, "kind": kind,
                            "status": "fail",
                            "detail": f"sha256 mismatch: got "
                                      f"{digest[:12]}…, pinned "
                                      f"{str(asset.get('sha256'))[:12]}…"})
    checked = sum(1 for e in entries if e["status"] == "verified")
    failed = sum(1 for e in entries if e["status"] == "fail")
    skipped = sum(1 for e in entries if e["status"] == "skipped")
    noted = sum(1 for e in entries if e["status"] == "noted")
    return {"checked": checked, "failed": failed,
            "skipped": skipped, "noted": noted, "entries": entries}
