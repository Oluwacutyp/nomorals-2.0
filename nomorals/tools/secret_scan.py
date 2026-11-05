"""Committed-secret scan as a real gate (Echelon Wave 8).

Until this wave the secret scan was advisory by design — a CI job with
``continue-on-error`` and an explicit ``exit 0``.  The echelon lesson
(INTEGRATION_PLAN §5: their repos carried production secrets for years)
is that a scan that cannot fail is a scan nobody reads.  This module is
the gate: it runs the repo's one secret-pattern battery
(:mod:`nomorals.security.audit_patterns` — the same battery the
security-audit organs use; a pattern here is a *shape*, and this gate is
the third consumer, not a second pattern list) over every tracked file
and exits non-zero on any hit that is not allowlisted.

The allowlist (``.secret-scan-allowlist.json`` at the repo root) is the
disclosure playbook's first step made data: each entry names one exact
hit — path, pattern, and the SHA-256 of the matched value — plus the
written reason it is not a live secret (documentation example, test
fixture, planted canary, format marker).  A *different* value at the
same path still fails the gate; only the adjudicated value is excused.
If a real secret is ever found: allowlisting is NOT the response —
rotate it, purge it, and publish the explanation (docs/CI_GATES.md).

Redaction law (the audit class's law): reports carry only the battery's
redacted form of a value, never the value itself.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..security.audit_patterns import entropy_hits, scan_text_for_secrets

_log = get_logger(__name__)

__all__ = [
    "ALLOWLIST_FILENAME",
    "MAX_FILE_BYTES",
    "tracked_files",
    "load_allowlist",
    "scan_repo",
    "register",
    "main",
]

#: Allowlist data file, at the repo root.
ALLOWLIST_FILENAME = ".secret-scan-allowlist.json"

#: Files larger than this are skipped (and the skip is reported as
#: data): the battery is a text scan, and multi-MB blobs are the
#: oversized-blob problem verify-then-install guards, not this gate's.
MAX_FILE_BYTES = 2_000_000

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ruff_cache"}


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parents[2]


def allowlist_path(repo: str | Path | None = None) -> Path:
    root = Path(repo).resolve() if repo else repo_root()
    return root / ALLOWLIST_FILENAME


def tracked_files(root: Path) -> list[str]:
    """Tracked files (git ls-files); falls back to a filesystem walk
    on an exported tree where git cannot answer."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0:
        names = proc.stdout.decode("utf-8", "replace").split("\0")
        return sorted(n for n in names if n)
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        found.append(rel.as_posix())
    return found


def load_allowlist(repo: str | Path | None = None) -> list[dict[str, Any]]:
    """The allowlist entries ([] when the file is absent).

    A malformed allowlist raises: a gate whose exceptions file cannot
    be parsed must fail loudly, never silently scan with no exceptions
    (or worse, with all of them).
    """
    path = allowlist_path(repo)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    data = json.loads(raw)
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected an object with an 'entries' list")
    return [e for e in entries if isinstance(e, dict)]


def _value_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def scan_repo(repo: str | Path | None = None) -> dict[str, Any]:
    """Scan every tracked file under ``repo`` with the secret battery.

    Returns comparable data: unallowlisted hits (path, line, pattern,
    redacted value — never the raw value), the count of allowlisted
    hits, allowlist entries that no longer match any hit (stale — the
    hygiene signal for the next allowlist review), and skip counts.
    ``ok`` is True exactly when there are no unallowlisted hits.
    """
    root = Path(repo).resolve() if repo else repo_root()
    entries = load_allowlist(root)
    allowed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (str(entry.get("path", "")), str(entry.get("pattern", "")),
               str(entry.get("value_sha256", "")))
        allowed[key] = entry

    hits: list[dict[str, Any]] = []
    allowlisted = 0
    matched_entries: set[tuple[str, str, str]] = set()
    files_scanned = 0
    skipped_binary = 0
    skipped_large = 0
    for rel in tracked_files(root):
        path = root / rel
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if len(data) > MAX_FILE_BYTES:
            skipped_large += 1
            continue
        if _is_binary(data):
            skipped_binary += 1
            continue
        files_scanned += 1
        text = data.decode("utf-8", errors="replace")
        found = scan_text_for_secrets(text) + entropy_hits(text)
        seen: set[tuple[str, int, str]] = set()
        for hit in found:
            fingerprint = _value_fingerprint(hit.value)
            dedupe = (hit.pattern, hit.offset, fingerprint)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            key = (rel, hit.pattern, fingerprint)
            if key in allowed:
                allowlisted += 1
                matched_entries.add(key)
                continue
            hits.append({
                "path": rel,
                "line": text.count("\n", 0, hit.offset) + 1,
                "pattern": hit.pattern,
                "value_redacted": hit.redacted,
                "value_sha256": fingerprint,
            })
    stale = [
        {"path": k[0], "pattern": k[1], "value_sha256": k[2],
         "reason": str(allowed[k].get("reason", ""))}
        for k in sorted(allowed) if k not in matched_entries
    ]
    hits.sort(key=lambda h: (h["path"], h["line"], h["pattern"]))
    return {
        "ok": not hits,
        "repo": str(root),
        "files_scanned": files_scanned,
        "skipped_binary": skipped_binary,
        "skipped_large": skipped_large,
        "hits": hits,
        "hit_count": len(hits),
        "allowlisted_hits": allowlisted,
        "stale_allowlist_entries": stale,
    }


def register(registry: Any) -> None:
    """Attach the secret-scan gate to a registry."""
    from ..core.policy import Capability

    @registry.register(
        "secret_scan",
        description=("Committed-secret gate: runs the security battery "
                     "over every tracked file. Returns {ok, hit_count, "
                     "hits (redacted values only), allowlisted_hits, "
                     "stale_allowlist_entries}. ok=False names every "
                     "unallowlisted secret-shaped hit."),
        capability=Capability.FS_READ,
    )
    def _secret_scan(repo: str | None = None) -> dict[str, Any]:
        return scan_repo(repo=repo)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="nomorals.tools.secret_scan",
        description="Committed-secret gate (fails on unallowlisted hits).")
    parser.add_argument("--repo", default=None,
                        help="repo root (default: this checkout)")
    args = parser.parse_args(argv)
    result = scan_repo(repo=args.repo)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
