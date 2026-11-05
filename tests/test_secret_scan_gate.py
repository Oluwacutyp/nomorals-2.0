"""Committed-secret gate (Echelon Wave 8).

Tier: unit. Pins the gate on synthetic git repos: a clean tree passes;
a planted full-format secret — assembled at runtime from concatenated
parts, never a literal in this file (the standing secaudit rule) —
fails the gate; the allowlist excuses exactly one (path, pattern,
value-hash) hit and nothing else; gate output never carries the raw
secret value. One slow test runs the gate over the REAL repo tree and
pins ok + a fully-exercised allowlist (no stale entries).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nomorals.tools import secret_scan

_GIT = shutil.which("git")

# A full-format AWS-style key, assembled from parts: the literal must
# never appear in this source file (GitHub secret-scanning blocks blobs
# carrying full-format literals, and the gate scans this file too).
_PROBE = "AKIA" + "IOSFODNN7" + "EXAMPLE"
_PROBE_ALT = "AKIA" + "IOSFODNN7" + "EXAMPLF"


def _repo(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@unittest.skipUnless(_GIT, "git not installed")
class TestSecretGate(unittest.TestCase):
    def test_clean_tree_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(root, {"app.py": "VALUE = 1\n"})
            result = secret_scan.scan_repo(repo=root)
            self.assertTrue(result["ok"], result["hits"])
            self.assertEqual(result["hit_count"], 0)

    def test_planted_secret_fails_and_is_never_printed_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(root, {"config.py": f'AWS_KEY = "{_PROBE}"\n'})
            result = secret_scan.scan_repo(repo=root)
            self.assertFalse(result["ok"])
            self.assertEqual(result["hit_count"], 1)
            hit = result["hits"][0]
            self.assertEqual(hit["pattern"], "aws_access_key")
            self.assertNotIn(_PROBE, json.dumps(result),
                             "the gate must never emit the raw value")

    def test_allowlist_excuses_exactly_one_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(root, {"config.py": f'AWS_KEY = "{_PROBE}"\n'})
            allowlist = {"entries": [{
                "path": "config.py",
                "pattern": "aws_access_key",
                "value_sha256": _fingerprint(_PROBE),
                "reason": "documentation example value (test)",
            }]}
            (root / secret_scan.ALLOWLIST_FILENAME).write_text(
                json.dumps(allowlist), encoding="utf-8")
            result = secret_scan.scan_repo(repo=root)
            self.assertTrue(result["ok"], result["hits"])
            self.assertEqual(result["allowlisted_hits"], 1)
            # A DIFFERENT value at the same path is not excused.
            (root / "config.py").write_text(
                f'AWS_KEY = "{_PROBE_ALT}"\n', encoding="utf-8")
            result = secret_scan.scan_repo(repo=root)
            self.assertFalse(result["ok"])
            self.assertEqual(result["hit_count"], 1)

    def test_stale_allowlist_entries_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(root, {"app.py": "VALUE = 1\n"})
            allowlist = {"entries": [{
                "path": "config.py",
                "pattern": "aws_access_key",
                "value_sha256": _fingerprint(_PROBE),
                "reason": "no longer present (test)",
            }]}
            (root / secret_scan.ALLOWLIST_FILENAME).write_text(
                json.dumps(allowlist), encoding="utf-8")
            result = secret_scan.scan_repo(repo=root)
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["stale_allowlist_entries"]), 1)


class TestRealTreeGate(unittest.TestCase):
    def test_real_tree_passes_with_no_stale_allowlist(self) -> None:
        result = secret_scan.scan_repo()
        self.assertTrue(result["ok"],
                        f"unallowlisted hits: {result['hits'][:5]}")
        self.assertEqual(result["stale_allowlist_entries"], [],
                         "every allowlist entry must match a live hit; "
                         "prune or re-adjudicate stale entries")


if __name__ == "__main__":
    unittest.main()
