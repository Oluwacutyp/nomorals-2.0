"""Pass #19 — backup + restore audit fixes.

Five verified bugs, each with a failing-first test:

A. A relocated generation could not be verified or drilled: manifest
   part paths are absolute (recorded at protect time), and both
   ``verify_manifest_files`` and the drill used them verbatim — so a
   from-zero restore on a clean machine reported every blob "missing"
   even with the blobs sitting in the given blobs dir.
B. Retention pruned generations still queued in the outbox: with the
   network down long enough, ``_prune_local`` deleted a queued
   generation's only copy; the outbox entry then pointed at files that
   no longer exist and could never ship.
C. A corrupt ``outbox.json`` silently emptied the queue: ``_load``
   swallowed JSON errors and returned [], so every queued backup
   vanished with no error anywhere — the exact silent skip the outbox
   exists to prevent.
D. A partial upload poisoned a generation forever: the release-reuse
   path (422 on create) re-uploaded every asset, and GitHub rejects a
   duplicate asset name with 422 — so a retry after dying mid-upload
   could never finish shipping.
E. ``join_parts`` sorted part names lexicographically: past 99 parts
   (``.part100of128`` sorts before ``.part99of128``) the reassembly
   order is wrong and the joined ciphertext is garbage.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.backup import build_manifest
from nomorals.backup.coordinator import BackupCoordinator
from nomorals.backup.errors import BackupError
from nomorals.backup.manifest import verify_manifest_files, write_manifest
from nomorals.backup.protect import (
    generate_backup_key,
    join_parts,
    protect_snapshot,
    split_encrypted,
)
from nomorals.backup.ship import Outbox, ship_snapshot
from nomorals.backup.snapshot import take_snapshot
from nomorals.backup.verify import drill


def _make_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("CREATE TABLE schema_migrations (version INTEGER)")
        conn.execute("INSERT INTO schema_migrations VALUES (86)")
        conn.execute("INSERT INTO items VALUES (1, 'x')")
        conn.commit()
    finally:
        conn.close()


def _make_generation(tmp: Path, key: bytes):
    root = tmp / "root"
    _make_db(root / "data" / "nomorals.db")
    (root / "config").mkdir(parents=True)
    (root / "config" / "settings.json").write_text('{"a": 1}')
    work = tmp / "work"
    snap = take_snapshot(root, work)
    out_dir = work / snap.backup_id / "protected"
    protected = protect_snapshot(snap, out_dir, key)
    manifest = build_manifest(backup_id=snap.backup_id,
                              created_at=snap.created_at,
                              protected=protected, plan=snap.plan)
    write_manifest(work / snap.backup_id, manifest)
    return work, snap.backup_id, manifest


class TestRelocatedGeneration(unittest.TestCase):
    """Bug A: blobs resolve by name under the given dir, not only by
    the absolute path recorded at protect time."""

    def test_verify_and_drill_pass_after_relocation(self):
        key = generate_backup_key()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            work, backup_id, manifest = _make_generation(tmp, key)
            gen = work / backup_id
            in_place = drill(gen / "manifest.json", gen / "protected", key)
            self.assertTrue(in_place.ok, in_place.summary())
            # From-zero restore: the tree now lives somewhere else and
            # the original location is gone.
            moved = tmp / "elsewhere" / backup_id
            shutil.copytree(gen, moved)
            shutil.rmtree(work)
            problems = verify_manifest_files(moved / "protected", manifest)
            self.assertEqual(problems, [],
                             f"relocated blobs reported: {problems}")
            report = drill(moved / "manifest.json", moved / "protected",
                           key)
            self.assertTrue(report.ok, report.summary())


class TestPruneProtectsQueued(unittest.TestCase):
    """Bug B: retention must never delete a generation the outbox
    still owes GitHub — its local blobs are the only copy."""

    def test_prune_local_spares_outbox_queued_generation(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "work"
            work.mkdir()
            now = time.time()
            names = []
            for i in range(9):  # same-day burst: retention prunes the tail
                gen = work / f"backup-20260101-0{i}0000"
                gen.mkdir()
                (gen / "manifest.json").write_text(json.dumps(
                    {"backup_id": gen.name,
                     "created_at": now - (9 - i) * 3600}))
                (gen / "protected").mkdir()
                (gen / "protected" / "x.gz.enc").write_bytes(b"cipher")
                names.append(gen.name)
            outbox = Outbox(work / "outbox")
            blob = str(work / names[0] / "protected" / "x.gz.enc")
            outbox.enqueue({"backup_id": names[0], "manifest": {},
                            "blob_files": [blob], "error": "network down"})
            coord = BackupCoordinator(root=Path(td), work_dir=work)
            result = coord._prune_local()
            self.assertNotIn(names[0], result["pruned"])
            self.assertTrue((work / names[0]).is_dir(),
                            "queued generation was pruned")
            self.assertTrue(Path(blob).is_file(),
                            "queued generation's blob was deleted")
            # Unqueued tail generations still prune normally.
            self.assertIn(names[1], result["pruned"])


class TestOutboxCorruption(unittest.TestCase):
    """Bug C: a corrupt outbox file is a loud BackupError, never a
    silently empty queue."""

    def test_corrupt_outbox_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ob = Outbox(Path(td) / "outbox")
            ob.enqueue({"backup_id": "backup-20260101-000000",
                        "manifest": {}, "blob_files": [],
                        "error": "network"})
            (Path(td) / "outbox" / "outbox.json").write_text("{not json!!")
            with self.assertRaises(BackupError):
                ob.pending()
            with self.assertRaises(BackupError):
                ob.depth()

    def test_nonlist_outbox_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ob_dir = Path(td) / "outbox"
            ob = Outbox(ob_dir)
            (ob_dir / "outbox.json").write_text('{"backup_id": "x"}')
            with self.assertRaises(BackupError):
                ob.pending()


class _Resp:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self.body = body
        self.headers: dict = {}
        self.text = body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class _AssetHttp:
    """Uploads endpoint double: GitHub rejects duplicate asset names
    with 422, exactly like the real Releases API."""

    def __init__(self, connector: "_AssetGitHub") -> None:
        self.connector = connector

    def request(self, method: str, url: str, **kwargs):
        if method == "POST" and "uploads.github.com" in url:
            name = url.split("name=")[-1]
            release = self.connector.releases[-1]
            if any(a["name"] == name for a in release["assets"]):
                return _Resp(422, json.dumps(
                    {"message": "Validation Failed: already exists"}
                ).encode())
            if (self.connector.fail_on == name):
                raise ConnectionError("Connection reset mid-upload")
            data = kwargs.get("data", b"")
            asset = {"id": self.connector.next_asset_id, "name": name,
                     "size": len(data)}
            self.connector.next_asset_id += 1
            release["assets"].append(asset)
            return _Resp(201, json.dumps(asset).encode())
        raise AssertionError(f"unexpected http {method} {url}")


class _AssetGitHub:
    """Connector double with a real per-release asset store."""

    def __init__(self) -> None:
        self.releases: list[dict] = []
        self.next_asset_id = 100
        self.fail_on = ""
        self.http = _AssetHttp(self)
        self.credential = SimpleNamespace(
            username="owner", password="tok", metadata={})

    def _require_credential(self):
        return self.credential

    def _load_credential(self):
        return self.credential

    def _headers(self, token: str):
        return {"Authorization": f"Bearer {token}"}

    def create_release(self, full_name: str, tag: str, *, name="",
                       body="", draft=False, prerelease=False):
        for rel in self.releases:
            if rel["tag_name"] == tag:
                raise RuntimeError("already_exists") from None
        release = {"id": len(self.releases) + 1, "tag_name": tag,
                   "upload_url": "https://uploads.github.com/repos/o/n"
                   "/releases/1/assets{?name,label}", "assets": []}
        self.releases.append(release)
        return release

    def _api(self, method: str, path: str, payload=None, *, params=None):
        if method == "GET" and "/releases/tags/" in path:
            tag = path.rsplit("/", 1)[-1]
            for rel in self.releases:
                if rel["tag_name"] == tag:
                    return rel
            raise RuntimeError("not found")
        if method == "GET" and path.endswith("/assets"):
            rid = int(path.split("/")[-2])
            for rel in self.releases:
                if rel["id"] == rid:
                    return list(rel["assets"])
            raise RuntimeError("no such release")
        if method == "DELETE" and "/releases/assets/" in path:
            aid = int(path.rsplit("/", 1)[-1])
            for rel in self.releases:
                rel["assets"] = [a for a in rel["assets"]
                                 if a["id"] != aid]
            return {}
        raise AssertionError(f"unexpected _api {method} {path}")


class TestShipResumesPartialUpload(unittest.TestCase):
    """Bug D: after dying mid-upload, the retry must finish the
    generation — assets already on the release (same name + size) are
    kept, not re-uploaded into a 422."""

    def test_retry_after_partial_upload_ships(self):
        gh = _AssetGitHub()
        manifest = build_manifest(backup_id="backup-20260101-000000",
                                  protected=[])
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a = td / "a.gz.enc"
            a.write_bytes(b"cipher-a")
            b = td / "b.gz.enc"
            b.write_bytes(b"cipher-b")
            gh.fail_on = "b.gz.enc"  # die on the second asset
            with self.assertRaises(BackupError):
                ship_snapshot(gh, "owner/nomorals-backup", manifest,
                              [a, b])
            self.assertEqual(
                [x["name"] for x in gh.releases[0]["assets"]],
                ["a.gz.enc"])
            gh.fail_on = ""  # network is back
            result = ship_snapshot(gh, "owner/nomorals-backup", manifest,
                                   [a, b])
            self.assertTrue(result["shipped"])
            names = sorted(x["name"] for x in gh.releases[0]["assets"])
            self.assertEqual(names, ["a.gz.enc", "b.gz.enc"])


class TestJoinPartsOrder(unittest.TestCase):
    """Bug E: part order is numeric, not lexicographic."""

    def test_join_over_99_parts_round_trips(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            blob = td / "big.gz.enc"
            data = bytes(range(256)) * 8
            blob.write_bytes(data)
            parts = split_encrypted(blob, max_bytes=16)
            self.assertGreater(len(parts), 99)
            joined = td / "joined.enc"
            join_parts(parts, joined)
            self.assertEqual(joined.read_bytes(), data)

    def test_join_two_digit_parts_still_ordered(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            blob = td / "small.gz.enc"
            data = bytes(range(64))
            blob.write_bytes(data)
            parts = split_encrypted(blob, max_bytes=16)
            self.assertEqual(len(parts), 4)
            joined = td / "joined.enc"
            join_parts(parts, joined)
            self.assertEqual(joined.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
