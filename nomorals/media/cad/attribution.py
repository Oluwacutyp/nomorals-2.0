"""Attribution DB — the SHA-256 provenance record.

Port of the design of the vectorcraft/layout attribution DBs (lanes
101/104): one entry per asset, each with its source, licence, and
the SHA-256 of the file as read. ``verify_attribution`` recomputes
the hashes against the upstream clones on disk, so provenance is
checkable at runtime, not a comment. Entries come in two kinds:

* ``upstream`` — a file in the storytold/cadcraft clone (or the
  cadcraft-corpus clone, named per asset in ``repo``) this port's
  design derives from; ``sha256`` is pinned and verified.
* ``devon`` — a file this lane created in Devon's tree; it is not
  hash-pinned (it is ours and it changes), but it must name the
  upstream units it derives from in ``derives_from`` — the check
  is that the lineage is written down, not that a hash matches.

Return shape: ``{"checked": n, "failed": [paths], "skipped":
[paths], "noted": n, "entries": [...]}`` — ``failed`` and
``skipped`` are lists of asset paths, so a caller can name what
broke.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from .errors import CadError

__all__ = ["DEFAULT_ATTRIBUTION_PATH", "DEFAULT_CLONE_ROOT",
           "DEFAULT_CORPUS_ROOT", "load_attribution",
           "verify_attribution"]

DEFAULT_ATTRIBUTION_PATH = Path(__file__).resolve().parent.parent \
    / "cad-attribution.toml"
DEFAULT_CLONE_ROOT = Path.home() / "workspace" / "echelon-mining" \
    / "repos" / "storytold" / "cadcraft"
DEFAULT_CORPUS_ROOT = Path.home() / "workspace" / "echelon-mining" \
    / "repos" / "storytold" / "cadcraft-corpus"


def load_attribution(path: str | Path | None = None
                     ) -> dict[str, Any]:
    """Load an attribution TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_ATTRIBUTION_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise CadError(
            f"cannot read attribution TOML {path}: {exc}",
            kind="io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CadError(f"bad attribution TOML {path}: {exc}",
                       kind="parse") from exc
    if not isinstance(data.get("assets"), list) or not data["assets"]:
        raise CadError(
            f"attribution TOML {path} has no [[assets]] rows",
            kind="parse")
    return data


def verify_attribution(path: str | Path | None = None, *,
                       clone_root: str | Path | None = None,
                       corpus_root: str | Path | None = None
                       ) -> dict[str, Any]:
    """Verify every [[assets]] entry: upstream entries are
    hash-checked against the clone; devon entries are checked for
    a written ``derives_from`` lineage. When a clone is not on
    disk, its entries report ``skipped`` — never a silent pass."""
    data = load_attribution(path)
    root = Path(clone_root) if clone_root is not None \
        else DEFAULT_CLONE_ROOT
    corpus = Path(corpus_root) if corpus_root is not None \
        else DEFAULT_CORPUS_ROOT
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
        base = corpus if str(asset.get("repo", "")) \
            .endswith("cadcraft-corpus") else root
        if not base.is_dir():
            entries.append({"path": asset_path, "kind": kind,
                            "status": "skipped",
                            "detail": "clone not on disk"})
            continue
        target = base / asset_path
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
    failed = [e["path"] for e in entries if e["status"] == "fail"]
    skipped = [e["path"] for e in entries if e["status"] == "skipped"]
    noted = sum(1 for e in entries if e["status"] == "noted")
    return {"checked": checked, "failed": failed,
            "skipped": skipped, "noted": noted, "entries": entries}
