"""Media-cache discipline — the 2026-11-08 EC2 disk incident, pinned.

What happened live: the WhatsApp bridges' inbound-media
download cache (``<home>/workspace/data/whatsapp_sessions/
wa-<number>-media/``) reached 11.07 GB and filled the owner's
50 GB disk, breaking /update at its snapshot step.  The same
76.3 MB channel video sat in the cache in ~145 copies from one
morning.  Three failures compounded:

1. No dedup: both bridges named every download
   ``<download-time-millis>-<id fragment>`` and never checked
   the cache, so the channel sweep — which re-serves the newest
   posts every housekeeping cycle — fetched the same media
   again every cycle.  (Bridge side pinned in
   bridge/gobridge/media_cache_test.go; the Baileys source is
   pinned by shape below.)
2. No retention: nothing ever evicted the cache.
3. The DM cache held raw cache paths for delete/edit
   recovery, so recovery records pointed into a volatile
   directory; and the disk report accounted only the snapshot
   store, so the biggest writer on the disk was invisible to
   /doctor and the /update preflight alike.

The repairs pinned here: the eviction primitive + the
housekeeping sweep with its journal row, the vault-copy rule
at capture (DM originals preserved into the vault; no vault
record points into the cache), and the whole-tree disk
accounting (areas + the largest-area note on the preflight).

Everything runs against temp dirs — never the real disk,
never a live install.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from nomorals.core import diskspace
from nomorals.social.pairing import AccountRecord, PairingRegistry
from nomorals.social.personal import PersonalSurfaces, PersonalVault

NOW = 1_800_000_000.0
REC_ID = "whatsapp:2348012345678"


def _mkfile(path: Path, size: int, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def _surfaces(tmp_path: Path, *, creds_dir: str = "",
              toggles: dict | None = None, **surface_kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id=REC_ID, platform="whatsapp", identity="2348012345678",
        creds_dir=creds_dir, status="linked", created_at=1.0))
    if toggles:
        registry.set_toggles(rec.id, **toggles)
    vault = PersonalVault(tmp_path / "personal")
    surfaces = PersonalSurfaces(registry, vault,
                                clock=lambda: NOW, **surface_kw)
    return surfaces, vault, registry.get(rec.id)


# ── eviction primitive ────────────────────────────────────────

def test_evict_oldest_first_to_cap(tmp_path):
    d = tmp_path / "cache"
    for i in range(5):
        _mkfile(d / f"f{i}.bin", 100, NOW - 5000 + i * 100)
    res = diskspace.evict_cache_dir(d, cap_bytes=250, now=NOW)
    assert res["files_evicted"] == 3
    assert res["bytes_freed"] == 300
    assert res["bytes_remaining"] == 200
    assert sorted(p.name for p in d.iterdir()) == ["f3.bin", "f4.bin"]


def test_evict_grace_spares_young_while_old_can_pay(tmp_path):
    d = tmp_path / "cache"
    _mkfile(d / "old.bin", 100, NOW - 7200)
    _mkfile(d / "young1.bin", 100, NOW - 10)
    _mkfile(d / "young2.bin", 100, NOW - 5)
    res = diskspace.evict_cache_dir(
        d, cap_bytes=200, min_age_seconds=3600.0, now=NOW)
    assert res["files_evicted"] == 1
    assert sorted(p.name for p in d.iterdir()) == [
        "young1.bin", "young2.bin"]


def test_evict_cap_is_hard_even_inside_grace(tmp_path):
    d = tmp_path / "cache"
    for i in range(4):
        _mkfile(d / f"y{i}.bin", 100, NOW - 60 + i)
    res = diskspace.evict_cache_dir(
        d, cap_bytes=150, min_age_seconds=3600.0, now=NOW)
    assert res["bytes_remaining"] <= 150
    assert res["files_evicted"] == 3


def test_evict_under_cap_is_a_noop(tmp_path):
    d = tmp_path / "cache"
    _mkfile(d / "a.bin", 50, NOW - 10)
    res = diskspace.evict_cache_dir(d, cap_bytes=250, now=NOW)
    assert res == {"files_evicted": 0, "bytes_freed": 0,
                   "bytes_remaining": 50, "errors": 0}
    assert (d / "a.bin").exists()


# ── the housekeeping sweep + its journal row ──────────────────

def test_cycle_media_cache_evicts_and_journals_once(tmp_path):
    sessions = tmp_path / "workspace" / "data" / "whatsapp_sessions"
    creds = sessions / "wa-2348012345678"
    media = sessions / "wa-2348012345678-media"
    creds.mkdir(parents=True)
    for i in range(5):
        _mkfile(media / f"v{i}.mp4", 100, NOW - 5000 + i * 100)
    surfaces, vault, rec = _surfaces(
        tmp_path, creds_dir=str(creds),
        media_cache_max_bytes=250, media_cache_min_age_seconds=0.0)
    out = surfaces.cycle_media_cache()
    assert len(out) == 1
    row = out[0]
    assert row["kind"] == "media_cache_evicted"
    assert row["account"] == rec.id
    assert row["files_evicted"] == 3
    assert row["bytes_freed"] == 300
    assert row["bytes_remaining"] == 200
    rows = [r for r in vault.journal(rec.id)
            if r["kind"] == "media_cache_evicted"]
    assert len(rows) == 1
    # Under the cap now: a second sweep writes nothing.
    assert surfaces.cycle_media_cache() == []
    assert len([r for r in vault.journal(rec.id)
                if r["kind"] == "media_cache_evicted"]) == 1


def test_cycle_media_cache_ignores_telegram_and_missing(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id="telegram:777", platform="telegram", identity="777",
        creds_dir=str(tmp_path / "tgcreds"), status="linked",
        created_at=1.0))
    vault = PersonalVault(tmp_path / "personal")
    surfaces = PersonalSurfaces(registry, vault, clock=lambda: NOW)
    assert surfaces.cycle_media_cache() == []


# ── the vault-copy rule ───────────────────────────────────────

def _cache_file(tmp_path: Path) -> Path:
    return _mkfile(
        tmp_path / "workspace" / "data" / "whatsapp_sessions"
        / "wa-2348012345678-media" / "1794122349750-A50946F5F931.mp4",
        64, NOW - 60)


def test_antidelete_original_is_preserved_into_the_vault(tmp_path):
    cache = _cache_file(tmp_path)
    surfaces, vault, rec = _surfaces(
        tmp_path, toggles={"antidelete": True})
    surfaces.observe_message(
        "whatsapp", chat_key="dm1", message_id="m1",
        sender_id="111@s.whatsapp.net", sender_name="Ada",
        text="watch this",
        media=[{"path": str(cache), "kind": "video"}],
        account_id=rec.id, ts=NOW - 30)
    cached = vault.get_message(rec.id, "dm1", "m1")
    assert cached is not None
    held = cached["media"]
    assert len(held) == 1
    held_path = Path(held[0])
    # The cache holds a VAULT path — never the download cache's.
    assert held_path != cache
    assert vault.root.resolve() in held_path.resolve().parents
    assert held_path.exists()
    # The cache file dies (eviction); the recovery still lands,
    # pointing at the vault copy.
    cache.unlink()
    out = surfaces.observe_delete(
        "whatsapp", chat_key="dm1", deleted_ids=["m1"],
        account_id=rec.id, ts=NOW)
    assert len(out) == 1
    assert out[0]["recovered"] is True
    assert out[0]["original_media"] == held
    assert Path(out[0]["original_media"][0]).exists()


def test_no_vault_record_points_into_the_cache(tmp_path):
    cache = _cache_file(tmp_path)
    surfaces, vault, rec = _surfaces(
        tmp_path, toggles={"antidelete": True,
                           "status_saving": True,
                           "view_once": True})
    surfaces.observe_message(
        "whatsapp", chat_key="dm1", message_id="m1",
        sender_id="111@s.whatsapp.net", text="",
        media=[{"path": str(cache), "kind": "video"}],
        view_once=True, account_id=rec.id, ts=NOW - 30)
    surfaces.observe_status_post(
        "whatsapp", poster_id="222@s.whatsapp.net",
        poster_name="Bo", text="",
        media=[{"path": str(cache), "kind": "video"}],
        source_id="s1", account_id=rec.id, ts=NOW - 20)
    surfaces.observe_delete(
        "whatsapp", chat_key="dm1", deleted_ids=["m1"],
        account_id=rec.id, ts=NOW)
    cache_dir = str(cache.parent)
    for name in ("journal.jsonl", "recoveries.jsonl",
                 "dm_cache.json"):
        blob = (vault.root / name).read_text(encoding="utf-8")
        assert cache_dir not in blob, f"{name} references the cache"
    # The arrival ledger names the media KIND, never its address.
    arrivals = [r for r in vault.journal(rec.id)
                if r["kind"] == "status_arrival"]
    assert arrivals and all(
        not m.get("path") for m in arrivals[0]["media"])


def test_view_once_capture_survives_cache_deletion(tmp_path):
    cache = _cache_file(tmp_path)
    surfaces, vault, rec = _surfaces(
        tmp_path, toggles={"view_once": True})
    surfaces.observe_message(
        "whatsapp", chat_key="dm1", message_id="m9",
        sender_id="111@s.whatsapp.net", text="",
        media=[{"path": str(cache), "kind": "video"}],
        view_once=True, account_id=rec.id, ts=NOW - 30)
    cache.unlink()
    saved = [r for r in vault.saved_entries(rec.id)
             if r["kind"] == "view_once"]
    assert len(saved) == 1
    assert Path(saved[0]["path"]).exists()


def test_preserve_original_dedups_and_passes_vault_paths(tmp_path):
    surfaces, vault, rec = _surfaces(tmp_path)
    cache = _cache_file(tmp_path)
    first = vault.preserve_original(rec.id, str(cache))
    second = vault.preserve_original(rec.id, str(cache))
    assert first == second
    originals = list(vault.originals_dir(rec.id).iterdir())
    assert len(originals) == 1
    # A path already inside the vault is returned unchanged.
    assert vault.preserve_original(rec.id, first) == first
    try:
        vault.preserve_original(rec.id, str(tmp_path / "gone.jpg"))
        raise AssertionError("a gone source must raise")
    except FileNotFoundError:
        pass


# ── whole-tree disk accounting ────────────────────────────────

def _home_with_cache(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    _mkfile(home / "data" / "nomorals.db", 100, NOW)
    cache = (home / "workspace" / "data" / "whatsapp_sessions"
             / "wa-2348012345678-media")
    _mkfile(cache / "a.mp4", 6000, NOW)
    _mkfile(cache / "b.mp4", 4000, NOW)
    _mkfile(home / "snapshots" / "s1" / "db.copy", 300, NOW)
    (home / "backups").mkdir(parents=True, exist_ok=True)
    return home


def test_disk_report_accounts_the_whole_tree(tmp_path):
    home = _home_with_cache(tmp_path)
    rep = diskspace.disk_report(home, home / "data" / "nomorals.db")
    assert rep["media_cache_bytes"] == 10000
    assert rep["largest_area"]["name"] == "workspace"
    assert rep["largest_area"]["bytes"] == 10000
    names = [a["name"] for a in rep["areas"]]
    assert names[0] == "workspace"
    assert "snapshots" in names and "data" in names
    note = diskspace.largest_area_note(rep)
    assert "workspace" in note
    assert "whatsapp media cache" in note


def test_largest_area_note_quiet_when_snapshot_dominates():
    rep = {"largest_area": {"name": "data", "bytes": 100},
           "snapshot_bytes": 5000, "media_cache_bytes": 0}
    assert diskspace.largest_area_note(rep) == ""
    assert diskspace.largest_area_note(
        {"largest_area": None, "snapshot_bytes": 1}) == ""


def test_preflight_names_the_biggest_area(tmp_path):
    from nomorals.os.snapshots import SnapshotManager
    from nomorals.os.update import UpdateManager

    home = _home_with_cache(tmp_path)
    snapshots = SnapshotManager(home, home / "data" / "nomorals.db")
    mgr = UpdateManager(str(home), snapshots,
                        health_checks=[lambda: (True, "fine")])
    ok, detail = mgr._preflight_disk()
    assert ok is True
    assert "biggest area: workspace" in detail
    with mock.patch("nomorals.core.diskspace.free_bytes",
                    lambda p: 0):
        ok, detail = mgr._preflight_disk()
    assert ok is False
    assert "Biggest consumer on disk: biggest area: workspace" \
        in detail


# ── Baileys source pins (the Node rung's dedup shape) ─────────

def test_baileys_bridge_dedups_by_media_identity():
    mjs = (Path(__file__).resolve().parents[1] / "bridge"
           / "whatsapp-bridge.mjs").read_text(encoding="utf-8")
    # The identity-keyed cache exists on this rung too…
    assert "mediaCacheName" in mjs
    assert "mediaInflight" in mjs
    assert "cachedMediaRow" in mjs
    # …and the old every-download-a-new-file grammar is gone.
    assert "Date.now()}-${(m.key.id" not in mjs
    assert "slice(0, 12)}${ext}" not in mjs
