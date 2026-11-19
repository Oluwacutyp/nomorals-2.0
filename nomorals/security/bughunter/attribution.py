"""Attribution DB — the SHA-256 provenance record for bughunter.

One entry per vendored file, each with its upstream path, its
vendored path in Devon's tree, its licence, and the SHA-256 of the
file as read. ``verify_attribution`` recomputes the hashes against
BOTH the upstream clone on disk and Devon's vendored copy, so
provenance is checkable at runtime, not a comment. Entries come in
two kinds:

* ``upstream`` — a file vendored from elementalsouls/Claude-BugHunter
  (commit 0e70a02). ``sha256`` is pinned and verified against the
  clone file (``file``) and the vendored copy (``vendored``).
  An entry reformatted without behaviour change (ruff --fix only)
  additionally carries ``upstream_sha256``: the clone file is then
  checked against ``upstream_sha256`` (the byte-identical upstream
  bytes) and the vendored copy against ``sha256``.
* ``devon`` — a path this lane created in Devon's tree; it is not
  hash-pinned (it is ours and it changes), but it must exist and
  name the upstream units it derives from in ``derives_from``.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_ATTRIBUTION_PATH", "DEFAULT_CLONE_ROOT",
           "load_attribution", "verify_attribution",
           "verify_against_clone"]

DEFAULT_ATTRIBUTION_PATH = Path(__file__).resolve().parent / "bughunter-attribution.toml"
DEFAULT_CLONE_ROOT = Path.home() / "workspace" / "bughunter-scout" / "repo"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_attribution(path: str | Path | None = None) -> dict[str, Any]:
    """Load an attribution TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_ATTRIBUTION_PATH
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data.get("upstream"), list) or not data["upstream"]:
        raise ValueError(
            f"attribution TOML {path} has no [[upstream]] rows")
    return data


def verify_attribution(path: str | Path | None = None, *,
                       clone_root: str | Path | None = None
                       ) -> dict[str, Any]:
    """Verify every [[upstream]] and [[devon]] entry.

    Upstream entries are hash-checked against the clone AND against
    Devon's vendored copy (both must equal the pinned SHA-256);
    devon entries are checked to exist. An upstream entry carrying
    ``upstream_sha256`` is adapted (reformatting only): the clone
    file is checked against ``upstream_sha256`` and the vendored
    copy against ``sha256``. Returns {verified, failed, skipped}.
    """
    toml_path = Path(path) if path is not None else DEFAULT_ATTRIBUTION_PATH
    if not toml_path.exists():
        return {"verified": 0,
                "failed": [{"id": "<toml>",
                            "reason": f"missing {toml_path}"}],
                "skipped": []}
    data = load_attribution(toml_path)
    clones = Path(clone_root) if clone_root is not None \
        else DEFAULT_CLONE_ROOT
    repo_root = Path(__file__).resolve().parents[3]
    verified = 0
    failed: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for entry in data["upstream"]:
        source = clones / entry["file"]
        if not source.exists():
            skipped.append({"id": entry["id"],
                            "reason": f"missing upstream file {source}"})
            continue
        clone_sha = entry.get("upstream_sha256", entry["sha256"])
        if _sha256(source) != clone_sha:
            failed.append({"id": entry["id"],
                           "reason": f"hash mismatch for {entry['file']}"})
            continue
        vendored = repo_root / entry["vendored"]
        if not vendored.exists():
            failed.append({"id": entry["id"],
                           "reason": f"missing vendored file {vendored}"})
            continue
        if _sha256(vendored) != entry["sha256"]:
            failed.append({"id": entry["id"],
                           "reason": "hash mismatch for vendored "
                                     f"{entry['vendored']}"})
            continue
        verified += 1
    for entry in data.get("devon", []):
        target = repo_root / entry["file"]
        if not target.exists():
            failed.append({"id": entry["id"],
                           "reason": f"missing Devon file {target}"})
            continue
        verified += 1
    return {"verified": verified, "failed": failed, "skipped": skipped}


def verify_against_clone(clone_root: str | Path | None = None,
                         path: str | Path | None = None
                         ) -> dict[str, Any]:
    """Cross-check the vendored tree against a fresh clone root.

    Same as ``verify_attribution`` but reports the clone path used,
    so the proof script can show the cross-check was against the
    clone on disk at ``~/workspace/bughunter-scout/repo``.
    """
    result = verify_attribution(path, clone_root=clone_root)
    result["clone_root"] = str(
        Path(clone_root) if clone_root is not None
        else DEFAULT_CLONE_ROOT)
    return result
