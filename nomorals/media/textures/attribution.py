"""TextureCraft attribution — SHA-256 record of the port (lane 108).

``nomorals/media/textures-attribution.toml`` maps every ported unit
to its upstream file at commit ``37ad2112`` plus the SHA-256 of the
source read, and every Devon-authored module to its lineage.
``verify_attribution`` recomputes the upstream hash when the
upstream clone is present on disk (it is on the build box); when
the clone is absent the asset is reported skipped, never silently
passed.
"""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

__all__ = ["verify_attribution"]

_UPSTREAM_COMMIT = "37ad2112"
_UPSTREAM_REPO = "storytold/texturecraft"
_DEFAULT_PATH = (Path(__file__).resolve().parents[2] / "media"
                 / "textures-attribution.toml")
_CLONE = (Path.home() / "workspace" / "echelon-mining" / "repos"
          / "storytold" / "texturecraft")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_attribution(path: str | Path | None = None) -> dict[str, Any]:
    toml_path = Path(path) if path is not None else _DEFAULT_PATH
    if not toml_path.exists():
        return {"verified": 0, "failed": [{"id": "<toml>",
                                           "reason": f"missing {toml_path}"}],
                "skipped": []}
    with toml_path.open("rb") as handle:
        data = tomllib.load(handle)
    verified = 0
    failed: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    repo_root = Path(__file__).resolve().parents[3]
    for asset in data.get("assets", []):
        asset_id = str(asset.get("id", "<unnamed>"))
        upstream = str(asset.get("upstream", ""))
        want = str(asset.get("sha256", ""))
        source = _CLONE / upstream
        if not source.exists():
            skipped.append({"id": asset_id,
                            "reason": f"upstream clone lacks {upstream}"})
            continue
        got = _sha256(source)
        if got != want:
            failed.append({"id": asset_id,
                           "reason": f"sha256 mismatch: got {got[:12]}, "
                                     f"want {want[:12]}"})
        else:
            verified += 1
    for entry in data.get("devon", []):
        entry_id = str(entry.get("id", "<unnamed>"))
        devon_path = repo_root / str(entry.get("path", ""))
        if not devon_path.exists():
            failed.append({"id": entry_id,
                           "reason": f"devon file missing: {entry.get('path')}"})
    return {"verified": verified, "failed": failed, "skipped": skipped}
