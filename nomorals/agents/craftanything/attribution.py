"""Attribution DB — the SHA-256 provenance record.

Same habit as the cadcraft/materialcraft/texturecraft organs: one entry
per asset, each with its source, licence, and the SHA-256 of the file as
read. ``verify_attribution`` recomputes the hashes against the upstream
clone on disk, so provenance is checkable at runtime, not a comment.
Entries come in two kinds:

* ``upstream`` — a file in the storytold/craft-anything clone this
  port's design derives from; ``sha256`` is pinned and verified.
* ``devon`` — a file this lane created in Devon's tree; it is not
  hash-pinned (it is ours and it changes), but it must name the
  upstream units it derives from in ``derives_from`` — the check is
  that the lineage is written down, not that a hash matches.

Return shape: ``{"checked": n, "failed": [paths], "skipped": [paths],
"noted": n, "entries": [...]}`` — ``failed`` and ``skipped`` are lists
of asset paths, so a caller can name what broke.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_ATTRIBUTION_PATH", "DEFAULT_CLONE_ROOT",
           "load_attribution", "verify_attribution"]

DEFAULT_ATTRIBUTION_PATH = Path(__file__).resolve().parent.parent \
    / "craftanything-attribution.toml"
DEFAULT_CLONE_ROOT = Path.home() / "workspace" / "echelon-mining" \
    / "repos" / "storytold" / "craft-anything"


def load_attribution(path: str | Path | None = None
                     ) -> dict[str, Any]:
    """Load an attribution TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_ATTRIBUTION_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ValueError(
            f"cannot read attribution TOML {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"bad attribution TOML {path}: {exc}") from exc
    if not isinstance(data.get("assets"), list) or not data["assets"]:
        raise ValueError(
            f"attribution TOML {path} has no [[assets]] rows")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_attribution(path: str | Path | None = None, *,
                       clone_root: str | Path | None = None
                       ) -> dict[str, Any]:
    """Verify every [[assets]] entry: upstream entries are hash-checked
    against the clone; devon entries are checked for lineage."""
    data = load_attribution(path)
    root = Path(clone_root) if clone_root is not None else DEFAULT_CLONE_ROOT
    checked = 0
    failed: list[str] = []
    skipped: list[str] = []
    noted = 0
    entries: list[dict[str, Any]] = []
    for asset in data["assets"]:
        source = asset.get("source", "?")
        kind = asset.get("kind", "upstream")
        entry: dict[str, Any] = {"source": source, "kind": kind}
        if kind == "devon":
            if not asset.get("derives_from"):
                failed.append(source)
                entry["status"] = "failed: no derives_from"
            else:
                noted += 1
                entry["status"] = "noted"
            entries.append(entry)
            continue
        target = root / source
        if not target.is_file():
            skipped.append(source)
            entry["status"] = "skipped: not in clone"
            entries.append(entry)
            continue
        checked += 1
        actual = _sha256(target)
        if actual != asset.get("sha256"):
            failed.append(source)
            entry["status"] = (f"failed: sha256 mismatch "
                               f"(expected {asset.get('sha256')}, "
                               f"got {actual})")
        else:
            entry["status"] = "verified"
        entries.append(entry)
    return {"checked": checked, "failed": failed, "skipped": skipped,
            "noted": noted, "entries": entries}
