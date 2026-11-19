"""SHA-256 attribution for the materials organ.

Every ported unit is pinned to the upstream file it derives from,
with the SHA-256 of the source as read from the clone at the pinned
upstream commit. ``verify_attribution`` re-hashes the clone files
and reports verified / failed / skipped counts; Devon-authored
modules are listed with their lineage, not as upstream assets.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

TOML_PATH = Path(__file__).resolve().parent.parent / "material-attribution.toml"
CLONE = Path.home() / "workspace" / "echelon-mining" / "repos" / "storytold" / "materialcraft"


def load_attribution(toml_path: Path | str | None = None) -> dict:
    path = Path(toml_path) if toml_path else TOML_PATH
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def verify_attribution(toml_path: Path | str | None = None,
                       clone: Path | str | None = None) -> dict:
    """Re-hash every upstream asset against the clone on disk."""
    data = load_attribution(toml_path)
    root = Path(clone) if clone else CLONE
    verified = failed = skipped = 0
    failures: list[dict] = []
    noted = 0
    for asset in data.get("assets", []):
        if asset.get("kind") != "upstream":
            noted += 1
            continue
        source = root / asset["path"]
        if not source.is_file():
            skipped += 1
            failures.append({"path": asset["path"], "reason": "not on disk"})
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest == asset.get("sha256"):
            verified += 1
        else:
            failed += 1
            failures.append({"path": asset["path"],
                             "reason": "sha256 mismatch"})
    return {"verified": verified, "failed": failed, "skipped": skipped,
            "noted": noted, "failures": failures,
            "upstream_commit": data.get("upstream_commit", ""),
            "upstream_repo": data.get("upstream_repo", "")}
