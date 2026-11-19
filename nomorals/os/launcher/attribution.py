"""SHA-256 attribution: every ported unit traced to its upstream file.

Upstream entries are hash-checked against the craft-launcher clone;
Devon entries are checked for a lineage note naming the upstream
file they derive from. The ``craft_update`` crate (upstream's
self-update engine) is NOT in the launcher clone — it arrives as a
git dependency — so it is recorded as noted, never hash-faked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = ["verify_attribution"]

_CLONE = Path.home() / "workspace" / "echelon-mining" / "repos" / \
    "storytold" / "craft-launcher"


def _toml_path() -> Path:
    return Path(__file__).resolve().parent.parent / "launcher-attribution.toml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_attribution(toml_path: str | Path | None = None, *,
                       clone_dir: str | Path | None = None) -> dict[str, Any]:
    """Check every attribution entry. Returns counts + the entries."""
    path = Path(toml_path) if toml_path is not None else _toml_path()
    clone = Path(clone_dir) if clone_dir is not None else _CLONE
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    checked = failed = skipped = noted = 0
    entries: list[dict[str, Any]] = []
    for entry in data.get("upstream", []):
        source = clone / str(entry["file"])
        if not source.is_file():
            skipped += 1
            entries.append({**entry, "status": "skipped",
                            "reason": "not in the local clone"})
            continue
        checked += 1
        ok = _sha(source) == str(entry["sha256"])
        if not ok:
            failed += 1
        entries.append({**entry, "status": "ok" if ok else "mismatch"})
    for entry in data.get("devon", []):
        target = Path(__file__).resolve().parent / str(entry["file"])
        checked += 1
        ok = target.is_file() and bool(entry.get("derives_from"))
        if not ok:
            failed += 1
        entries.append({**entry, "status": "ok" if ok else "mismatch"})
    for entry in data.get("noted", []):
        noted += 1
        entries.append({**entry, "status": "noted"})
    return {"checked": checked, "failed": failed, "skipped": skipped,
            "noted": noted, "entries": entries}
