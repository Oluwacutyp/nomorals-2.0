"""WS1-P5 — one bundle seam for every bundle family.

Before P5, four bundle implementations shared no code: artifact
export (storage/artifacts), database-backup blob sidecars
(storage/backup), OS snapshot export (os/snapshots) and backup
generations (backup/) each hand-rolled a tar/zip writer and its
own manifest shape.  The seam lives in ``nomorals.archives``:
``create_bundle`` / ``open_bundle`` / ``list_bundle`` /
``verify_bundle`` / ``extract_bundle`` / ``read_bundle_members``
over ONE versioned manifest (``bundle.manifest.json``: names,
sizes, sha256, created-by, created-at, format version).

Pins here:
* the manifest shape itself (the cross-family contract);
* a round trip per family THROUGH the seam (each family's own
  producer writes, the shared reader reads);
* backward compatibility — bundles in the legacy shapes (built
  exactly the way the pre-seam code built them) still open,
  extract and import;
* verify catches a tampered member BY NAME;
* a traversal member refuses the whole bundle;
* a bundle from one family opens through the shared reader with
  its manifest visible.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from nomorals.archives import (
    BUNDLE_FORMAT,
    BUNDLE_FORMAT_VERSION,
    BUNDLE_MANIFEST_NAME,
    bundle_manifest,
    create_bundle,
    extract_bundle,
    list_bundle,
    open_bundle,
    read_bundle_members,
    verify_bundle,
)
from nomorals.core.errors import ToolError


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── the manifest contract ─────────────────────────────────────────────

def test_manifest_shape_is_the_contract():
    man = bundle_manifest(
        [{"name": "a.txt", "size": 3, "sha256": _sha(b"abc")}],
        created_by="unit-test", kind="demo", created_at=1700000000.0)
    assert man["format"] == BUNDLE_FORMAT == "nomorals-bundle"
    assert man["version"] == BUNDLE_FORMAT_VERSION == 1
    assert man["created_by"] == "unit-test"
    assert man["created_at"] == 1700000000.0
    assert man["kind"] == "demo"
    assert man["members"] == [
        {"name": "a.txt", "size": 3, "sha256": _sha(b"abc"),
         "is_dir": False}]
    # extra keys ride along but never override the core
    man2 = bundle_manifest([], created_by="x", extra={"version": 99,
                                                       "note": "hi"})
    assert man2["version"] == 1 and man2["note"] == "hi"


@pytest.mark.parametrize("container,name", [
    ("tar.gz", "b.tar.gz"), ("tar", "b.tar"), ("tar.bz2", "b.tar.bz2"),
    ("tar.xz", "b.tar.xz"), ("zip", "b.zip"),
])
def test_create_open_verify_extract_round_trip(tmp_path, container, name):
    src = tmp_path / "src.txt"
    src.write_text("seam round trip")
    dest = tmp_path / name
    man = create_bundle(
        dest, [("src.txt", src), ("blob.bin", b"\x00\x01\x02")],
        created_by="unit-test", kind="demo", container=container,
        created_at=1700000000.0)
    assert man["members"][0]["sha256"] == _sha(b"seam round trip")
    info = open_bundle(dest)
    assert info["legacy"] is False
    assert info["manifest"]["created_by"] == "unit-test"
    assert [m["name"] for m in info["members"]] == ["src.txt", "blob.bin"]
    verdict = verify_bundle(dest)
    assert verdict["ok"] and verdict["checked"] == 2
    out = tmp_path / ("out-" + name)
    result = extract_bundle(dest, out)
    assert result["extracted"] == 3  # payload + bundle manifest member
    assert (out / "src.txt").read_text() == "seam round trip"
    assert (out / "blob.bin").read_bytes() == b"\x00\x01\x02"
    assert read_bundle_members(dest)["blob.bin"] == b"\x00\x01\x02"


def test_create_rejects_reserved_duplicate_and_unsafe_names(tmp_path):
    with pytest.raises(ToolError, match="reserved"):
        create_bundle(tmp_path / "x.tar.gz",
                      [(BUNDLE_MANIFEST_NAME, b"nope")],
                      created_by="t")
    with pytest.raises(ToolError, match="duplicate"):
        create_bundle(tmp_path / "x.tar.gz",
                      [("a.txt", b"1"), ("a.txt", b"2")],
                      created_by="t")
    with pytest.raises(ToolError, match="traversal"):
        create_bundle(tmp_path / "x.tar.gz", [("../evil", b"1")],
                      created_by="t")


def test_verify_catches_tampered_member_by_name(tmp_path):
    dest = tmp_path / "t.tar.gz"
    man = create_bundle(dest, [("a.txt", b"same-size!")],
                        created_by="t")
    # Rewrite the payload with SAME-LENGTH different bytes, keeping
    # the original bundle manifest: size passes, sha256 must name it.
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tf:
        data = b"tampered!!"
        assert len(data) == len(b"same-size!")
        info = tarfile.TarInfo("a.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        man_bytes = json.dumps(man).encode()
        info = tarfile.TarInfo(BUNDLE_MANIFEST_NAME)
        info.size = len(man_bytes)
        tf.addfile(info, io.BytesIO(man_bytes))
    verdict = verify_bundle(evil)
    assert not verdict["ok"]
    assert verdict["problems"] == ["sha256 mismatch: a.txt"]


def test_verify_names_missing_and_unmanifested_members(tmp_path):
    dest = tmp_path / "t.zip"
    create_bundle(dest, [("a.txt", b"1")], created_by="t",
                  container="zip")
    with zipfile.ZipFile(dest, "a") as zf:
        zf.writestr("sneaky.txt", "unrecorded")
    verdict = verify_bundle(dest)
    assert "unmanifested member: sneaky.txt" in verdict["problems"]


def test_extract_refuses_traversal_member(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../escape.txt", "x")
        zf.writestr("ok.txt", "y")
    with pytest.raises(ToolError, match="unsafe member"):
        extract_bundle(bad, tmp_path / "out")
    assert not (tmp_path.parent / "escape.txt").exists()
    assert not (tmp_path / "out" / "ok.txt").exists()  # refuse-WHOLE


def test_extract_refuses_tar_link_member(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ToolError, match="not a regular file"):
        extract_bundle(bad, tmp_path / "out")


def test_extract_bomb_budget_refuses(tmp_path):
    dest = tmp_path / "big.tar.gz"
    create_bundle(dest, [("a.bin", b"x" * 4096)], created_by="t")
    with pytest.raises(ToolError, match="budget"):
        extract_bundle(dest, tmp_path / "out", max_bytes=1024)


# ── legacy shapes: bundles written by the OLD code still open ─────────

def _legacy_artifact_bundle(path: Path, payload: bytes) -> None:
    """Exactly the pre-seam ArtifactStore.export_bundle shape:
    ``<id>.bin`` members + a ``manifest.json`` LIST, tar.gz, no
    bundle manifest."""
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("art1.bin")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        man = json.dumps([{"id": "art1", "type": "report"}]).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(man)
        tar.addfile(info, io.BytesIO(man))


def test_legacy_artifact_bundle_opens_through_seam(tmp_path):
    bundle = tmp_path / "legacy.tar.gz"
    _legacy_artifact_bundle(bundle, b"payload-bytes")
    info = open_bundle(bundle)
    assert info["legacy"] is True and info["manifest"] is None
    assert info["family_manifest"] == [{"id": "art1", "type": "report"}]
    assert {m["name"] for m in info["members"]} == {
        "art1.bin", "manifest.json"}
    assert verify_bundle(bundle)["ok"]
    out = tmp_path / "out"
    extract_bundle(bundle, out)
    assert (out / "art1.bin").read_bytes() == b"payload-bytes"


def test_legacy_blob_sidecar_opens_and_extracts(tmp_path):
    """Pre-seam storage/backup sidecar: plain tar.gz of a blob dir,
    no manifest member at all."""
    sidecar = tmp_path / "snap-blobs.tar.gz"
    with tarfile.open(sidecar, "w:gz") as tar:
        data = b"blob-bytes"
        info = tarfile.TarInfo("ab/cd/blob1")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    info = open_bundle(sidecar)
    assert info["legacy"] is True
    out = tmp_path / "out"
    extract_bundle(sidecar, out)
    assert (out / "ab" / "cd" / "blob1").read_bytes() == b"blob-bytes"


def test_legacy_snapshot_tar_opens_through_seam(tmp_path):
    """Pre-seam os/snapshots export: ``tar.add(dir, arcname=id)`` —
    directory entries included, no bundle manifest."""
    snap_dir = tmp_path / "snapX"
    (snap_dir / "blobs").mkdir(parents=True)
    (snap_dir / "db.sqlite").write_bytes(b"db")
    (snap_dir / "manifest.json").write_text(json.dumps(
        {"id": "snapX", "files": {"db.sqlite": {
            "sha256": _sha(b"db"), "bytes": 2}}}))
    tarball = tmp_path / "snap.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(snap_dir, arcname="snapX")
    info = open_bundle(tarball)
    assert info["legacy"] is True
    tops = {m["name"].split("/")[0] for m in info["members"]}
    assert tops == {"snapX"}
    out = tmp_path / "out"
    extract_bundle(tarball, out)
    assert (out / "snapX" / "db.sqlite").read_bytes() == b"db"
    assert (out / "snapX" / "blobs").is_dir()


def test_legacy_directory_bundles_open_through_seam(tmp_path):
    # Snapshot directory shape (manifest.json with a files dict).
    snap = tmp_path / "snapY"
    snap.mkdir()
    (snap / "db.sqlite").write_bytes(b"db2")
    (snap / "manifest.json").write_text(json.dumps(
        {"id": "snapY", "files": {"db.sqlite": {
            "sha256": _sha(b"db2"), "bytes": 3}}}))
    info = open_bundle(snap)
    assert info["format"] == "directory" and info["legacy"] is True
    assert info["members"][0]["name"] == "db.sqlite"
    assert verify_bundle(snap)["ok"]
    # Tamper with the recorded file: the family hash names it.
    (snap / "db.sqlite").write_bytes(b"bad")
    verdict = verify_bundle(snap)
    assert verdict["problems"] == ["sha256 mismatch: db.sqlite"]


# ── one family through the seam, end to end ───────────────────────────

def test_artifact_store_round_trip_through_seam(tmp_path):
    from nomorals.storage.artifacts import ArtifactStore
    from nomorals.storage.blob import BlobStore
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    store = ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))
    art = store.put_text("bundle me", type="report", creator="me",
                         metadata={"k": "v"})
    dest = tmp_path / "arts.tar.gz"
    bundle = store.export_bundle([art.id, "missing-id"], dest)
    # The bundle carries BOTH manifests: the family's artifact
    # manifest (visible through the seam as family_manifest) and
    # the standard bundle manifest.
    info = open_bundle(bundle)
    assert info["legacy"] is False
    assert info["manifest"]["kind"] == "artifacts"
    assert info["manifest"]["created_by"] == "artifact-store"
    assert info["family_manifest"][0]["id"] == art.id
    assert verify_bundle(bundle)["ok"]
    # Import into a fresh store through the migrated reader.
    db2 = Database(":memory:")
    db2.migrate()
    other = ArtifactStore(db2, BlobStore(db2, tmp_path / "blobs2"))
    imported = other.import_bundle(bundle)
    assert len(imported) == 1
    assert other.read_text(imported[0].id) == "bundle me"
    # And a LEGACY-shaped bundle (old writer, simulated above)
    # imports through the new reader too.
    legacy = tmp_path / "legacy.tar.gz"
    _legacy_artifact_bundle(legacy, b"old-payload")
    db3 = Database(":memory:")
    db3.migrate()
    third = ArtifactStore(db3, BlobStore(db3, tmp_path / "blobs3"))
    imported = third.import_bundle(legacy)
    assert len(imported) == 1
    assert third.read_text(imported[0].id) == "old-payload"


def test_backup_blob_sidecar_round_trip_through_seam(tmp_path):
    from nomorals.storage.backup import BackupManager
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    blob_dir = tmp_path / "blobsrc"
    (blob_dir / "aa").mkdir(parents=True)
    (blob_dir / "aa" / "b1").write_bytes(b"blob-one")
    (blob_dir / "b2").write_bytes(b"blob-two")
    mgr = BackupManager(db, tmp_path / "backups", include_blobs=True,
                        blob_dir=blob_dir)
    info = mgr.create(label="seam")
    assert info.blob_name and info.blob_count == 2
    sidecar = Path(info.path).parent / info.blob_name
    # The sidecar is a seam bundle; its manifest members are the
    # same per-file digests the backup manifest records.
    opened = open_bundle(sidecar)
    assert opened["legacy"] is False
    assert opened["manifest"]["kind"] == "backup-blobs"
    seam_files = {m["name"]: m["sha256"]
                  for m in opened["manifest"]["members"]}
    assert seam_files == info.blob_files
    assert verify_bundle(sidecar)["ok"]
    # The manager's own verify (which extracts via the seam) is clean.
    assert mgr.verify(info) == []


def test_snapshot_export_round_trip_through_seam(tmp_path):
    from nomorals.os.snapshots import SnapshotManager

    home = tmp_path / "home"
    home.mkdir()
    db_path = home / "state.db"
    import sqlite3
    sqlite3.connect(str(db_path)).close()
    mgr = SnapshotManager(home, db_path)
    snap = mgr.create(label="seam")
    tarball = tmp_path / "snap.tar.gz"
    mgr.export_tar(snap.id, tarball)
    opened = open_bundle(tarball)
    assert opened["legacy"] is False
    assert opened["manifest"]["kind"] == "snapshot"
    assert opened["manifest"]["created_by"] == "os.snapshots"
    assert verify_bundle(tarball)["ok"]
    mgr.delete(snap.id)
    imported = mgr.import_tar(tarball)
    assert imported.id == snap.id
    # The bundle manifest must not pollute the imported snapshot.
    assert not (imported.path / BUNDLE_MANIFEST_NAME).exists()
    assert not (mgr.root / BUNDLE_MANIFEST_NAME).exists()


def test_backup_generation_manifest_carries_bundle_block(tmp_path):
    from nomorals.backup.manifest import (
        build_manifest,
        load_manifest,
        verify_manifest_files,
        write_manifest,
    )

    protected = [{
        "rel": "db/main.db", "encrypted_path": "main.db.gz.enc",
        "sha256": _sha(b"ciphertext"), "size": len(b"ciphertext"),
        "parts": [], "kind": "database", "schema_version": 1,
        "tables": 3,
    }]
    manifest = build_manifest(backup_id="backup-1", protected=protected,
                              created_at=1700000000.0)
    block = manifest["bundle"]
    assert block["format"] == BUNDLE_FORMAT
    assert block["version"] == BUNDLE_FORMAT_VERSION
    assert block["created_by"] == "backup.package"
    assert block["kind"] == "backup-generation"
    assert block["members"] == [
        {"name": "db/main.db", "size": len(b"ciphertext"),
         "sha256": _sha(b"ciphertext"), "is_dir": False}]
    # The generation dir opens through the seam reader and the
    # block is its manifest; the family's own verify still passes.
    gen = tmp_path / "backup-1"
    gen.mkdir()
    (gen / "main.db.gz.enc").write_bytes(b"ciphertext")
    write_manifest(gen, manifest)
    loaded = load_manifest(gen / "manifest.json")
    assert loaded["bundle"]["kind"] == "backup-generation"
    assert verify_manifest_files(gen, loaded) == []
    info = open_bundle(gen)
    assert info["legacy"] is False
    assert info["manifest"]["kind"] == "backup-generation"
    assert list_bundle(gen)[0]["name"] == "db/main.db"
