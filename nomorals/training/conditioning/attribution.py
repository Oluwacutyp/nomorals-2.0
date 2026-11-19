"""Attribution DB — the SHA-256 provenance record for conditioning.

One entry per asset, each with its source, licence, and the SHA-256 of
the file as read. ``verify_attribution`` recomputes the hashes against
the upstream clone on disk, so provenance is checkable at runtime, not
a comment. Entries come in two kinds:

* ``upstream`` — a file in the storytold/FineTrainers-Conditioning
  clone this port's design derives from; ``sha256`` is pinned and
  verified. The clone is Apache-2.0 at commit 9f0fdb5 — a pristine
  snapshot of upstream aarnphm/FineTrainers by its own author, with no
  storytold-authored commits (``git log`` shows a single commit by
  Aryan <aryan@huggingface.co>), so "upstream" here means the HF
  FineTrainers project, not storytold code.
* ``devon`` — a file this lane created in Devon's tree; it is not
  hash-pinned (it is ours and it changes), but it must exist and name
  the upstream units it derives from in ``derives_from``.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_ATTRIBUTION_PATH", "DEFAULT_CLONE_ROOT",
           "load_attribution", "verify_attribution"]

DEFAULT_ATTRIBUTION_PATH = Path(__file__).resolve().parent.parent.parent.parent / "training" / "conditioning-attribution.toml"
DEFAULT_CLONE_ROOT = Path.home() / "workspace" / "echelon-mining" / "repos" / "storytold" / "FineTrainers-Conditioning"


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
    """Verify every [[upstream]] and [[devon]] entry: upstream entries
    are hash-checked against the clone; devon entries are checked to
    exist. Returns {verified, failed, skipped}."""
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
        if _sha256(source) != entry["sha256"]:
            failed.append({"id": entry["id"],
                           "reason": f"hash mismatch for {entry['file']}"})
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
