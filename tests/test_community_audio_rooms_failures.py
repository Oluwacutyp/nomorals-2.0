"""Pass #171 pins: the community failure ledger (NEW) + audio_rooms conversions.

``nomorals/community/failures.py`` (namespace ``community``) is new:
no community-area ledger existed on this base. The 17 durability tests
below mirror the #141+ shape exactly; the 17 site pins prove each
conversion records without changing flow (fallback value unchanged AND
a ledger entry under its site name); the 2 exclusion pins prove the
excluded probes record nothing. No pin hardcodes scan totals or
audio_rooms counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written. No bare swallow blocks anywhere in this
file (pins-rule from #169): every handler binds ``as exc`` and uses the
bound name.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.community import failures as cf
from nomorals.community import audio_rooms as A


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    cf.clear_failures()
    yield
    cf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in cf.recent_failures()]


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("community")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("community").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("community")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("community").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("community")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("community").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("community").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("community").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("community")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("community").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("community").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("community").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("community").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("community")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 - test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("community").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        cf.record_failure("no-home", ValueError("boom"))
        assert cf.failure_count() == 1
        assert [f.site for f in cf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []
    assert cf.format_failures() == "no community failures recorded."
    cf.record_failure("community.site-a", ValueError("boom"))
    cf.record_failure("community.site-b", RuntimeError("bang"))
    assert cf.failure_count() == 2
    assert [f.site for f in cf.recent_failures()] == ["community.site-b", "community.site-a"]
    assert [f.site for f in cf.recent_failures(1)] == ["community.site-b"]
    text = cf.format_failures()
    assert "2 total" in text and "community.site-a" in text and "community.site-b" in text
    cf.clear_failures()
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    cf.record_failure("community.disk", ValueError("boom"))
    entries, total = fs.Store("community").load()
    assert total == 1 and entries[0]["site"] == "community.disk"
    cf.clear_failures()
    assert fs.Store("community").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    cf.record_failure("community.mem", ValueError("boom"))
    assert cf.failure_count() == 1
    assert [f.site for f in cf.recent_failures()] == ["community.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                cf.record_failure(f"community.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 - test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cf.failure_count() == 400
    assert len(cf.recent_failures(1000)) == 200
    entries, total = cf._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.community import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.community import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.community import failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("community").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Site pins ------------------------------------------------------------

def _boom(*a, **k):
    raise RuntimeError("boom")


class _BoomPlatform:
    def set_muted(self, room, user_id, muted):
        raise RuntimeError("platform down")


class _BoomGroup:
    def get(self, cid):
        raise RuntimeError("groups down")


class _OkGroup:
    def get(self, cid):
        return object()


class _RaisingListStore(A.RoomStore):
    def _load_index(self):
        raise RuntimeError("index exploded")


class _RaisingDueStore(A.RoomStore):
    def list(self, community_id="", live_only=True):
        raise RuntimeError("list exploded")


class _RaisingGetStore(A.RoomStore):
    def get(self, room_id):
        raise RuntimeError("get exploded")


class _RaisingPutStore(A.RoomStore):
    def put(self, room):
        raise RuntimeError("put exploded")


def _store(tmp_path):
    return A.RoomStore(data_dir=str(tmp_path / "rooms"))


def test_site_default_dir_records_and_returns_dir(monkeypatch):
    monkeypatch.setattr(Path, "mkdir", _boom)
    d = A._default_dir()
    assert d.name == "audio_rooms"
    assert "audio_rooms.default_dir" in _sites()


def test_site_store_init_records_and_falls_back_to_tempdir(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    st = A.RoomStore(data_dir=str(blocker / "sub"))
    assert "devon_audio_rooms" in str(st._dir)
    assert "audio_rooms.store_init" in _sites()


def test_site_load_index_records_and_returns_empty(tmp_path):
    store = _store(tmp_path)
    store._index_path().write_text("{not json", encoding="utf-8")
    assert store._load_index() == []
    assert "audio_rooms.load_index" in _sites()


def test_site_save_index_records_and_returns_none(tmp_path):
    store = _store(tmp_path)
    store._index_path().mkdir()
    assert store._save_index(["x"]) is None
    assert "audio_rooms.save_index" in _sites()


def test_site_put_records_and_returns_false(tmp_path):
    store = _store(tmp_path)
    room = A.Room(room_id="r1", community_id="c1", title="t", host_id="h")
    (tmp_path / "rooms" / "r1.json").mkdir(parents=True)
    assert store.put(room) is False
    assert "audio_rooms.put" in _sites()


def test_site_get_records_and_returns_none(tmp_path):
    store = _store(tmp_path)
    (tmp_path / "rooms").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rooms" / "r1.json").write_text("{bad json", encoding="utf-8")
    assert store.get("r1") is None
    assert "audio_rooms.get" in _sites()


def test_site_list_records_and_returns_empty(tmp_path):
    store = _RaisingListStore(data_dir=str(tmp_path / "rooms"))
    assert store.list() == []
    assert "audio_rooms.list" in _sites()


def test_site_remove_records_and_returns_false(tmp_path):
    store = _store(tmp_path)
    (tmp_path / "rooms").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rooms" / "r1.json").mkdir()
    assert store.remove("r1") is False
    assert "audio_rooms.remove" in _sites()


def test_site_platform_mute_records_and_returns_false():
    room = A.Room(room_id="r1")
    assert A._platform_mute(_BoomPlatform(), room, "u", True) is False
    assert "audio_rooms.platform_mute" in _sites()


def test_site_platform_mute_records_even_though_mute_user_reports_ok(tmp_path):
    # Safety pin: mute_user discards the platform result and reports
    # ok=True; the ledger entry is what makes the failure visible.
    store = _store(tmp_path)
    res = A.create_room("c1", "title", "host1", "Host",
                        store=store, group_store=_OkGroup())
    rid = res["room"].room_id
    A.join_room(rid, "spk", "Spk", store=store)
    res = A.mute_user(rid, "spk", "host1", True,
                      store=store, platform=_BoomPlatform())
    assert res.get("ok") is True
    assert "audio_rooms.platform_mute" in _sites()


def test_site_community_exists_records_and_returns_false():
    assert A._community_exists("c1", group_store=_BoomGroup()) is False
    assert "audio_rooms.community_exists" in _sites()


def test_site_due_starts_records_and_returns_empty(tmp_path):
    store = _RaisingDueStore(data_dir=str(tmp_path / "rooms"))
    assert A.due_starts(store=store) == []
    assert "audio_rooms.due_starts" in _sites()


def test_site_get_room_records_and_returns_none(tmp_path):
    store = _RaisingGetStore(data_dir=str(tmp_path / "rooms"))
    assert A.get_room("x", store=store) is None
    assert "audio_rooms.get_room" in _sites()


def test_site_save_records_and_does_not_raise(tmp_path):
    store = _RaisingPutStore(data_dir=str(tmp_path / "rooms"))
    assert A._save(store, A.Room(room_id="r1")) is None
    assert "audio_rooms.save" in _sites()


def test_site_expire_hands_records_and_returns_zero():
    room = A.Room(room_id="r1")
    room.hand_raise_queue = None  # type: ignore[assignment]
    assert A._expire_hands(room) == 0
    assert "audio_rooms.expire_hands" in _sites()


def test_site_vtt_ts_records_and_returns_epoch_string():
    assert A._vtt_ts(1e20) == "00:00:00.000"
    assert "audio_rooms.vtt_ts" in _sites()


def test_site_fmt_naira_records_and_returns_zero_string():
    assert A._fmt_naira(float("nan")) == "₦0"
    assert "audio_rooms.fmt_naira" in _sites()


def test_site_render_room_records_and_returns_failure_string():
    room = A.Room(room_id="r1", title="t")
    room.reactions = ["x"]  # type: ignore[assignment]
    assert A.render_room(room) == "room card failed to render"
    assert "audio_rooms.render_room" in _sites()


# ---- Exclusion pins (NOT-recording) ---------------------------------------

def test_exclusion_transcribe_records_nothing(tmp_path):
    store = _store(tmp_path)
    room = A.Room(room_id="r1", community_id="c1", title="t",
                  host_id="host1", state="live")
    room.caption_log = ["hello", "world"]
    assert store.put(room) is True
    # Snapshot AFTER setup: the fresh store's first put legitimately
    # records one audio_rooms.load_index FileNotFoundError (no index
    # file exists yet on a brand-new store — see SURVEY.md note). The
    # exclusion claim is about the transcribe call itself: it must add
    # no entry.
    before = _sites()
    res = A.end_room("r1", "host1", transcriber=_boom, store=store)
    assert res["room"].transcript == "hello\nworld"
    assert "transcript from live captions" in (res.get("transcript_note") or "")
    assert _sites() == before


def test_exclusion_tip_parse_records_nothing(tmp_path):
    store = _store(tmp_path)
    res = A.create_room("c1", "title", "host1", "Host",
                        store=store, group_store=_OkGroup())
    rid = res["room"].room_id
    before = _sites()
    out = A.control_room(f"tip {rid} spk abc", sender="Host",
                         sender_id="host1", store=store)
    assert out == "tip amount must be a number (naira)"
    assert _sites() == before
