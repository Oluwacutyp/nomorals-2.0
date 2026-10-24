"""Pass #142 pins: the social/chat failure ledger + durability wiring.

Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through the new ``nomorals.social.chat.failures`` ledger; restart
tests use real subprocesses (a fresh interpreter is the honest
simulated restart — no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.social.chat import failures as chat


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    chat.clear_failures()
    yield
    chat.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("social.chat")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("social.chat").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("social.chat")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("social.chat").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("social.chat")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("social.chat").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("social.chat").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("social.chat").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("social.chat")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("social.chat").load() == ([], 0)
    assert store.save(_entries(1, "after"), 1) is True
    entries, total = fs.Store("social.chat").load()
    assert [e["site"] for e in entries] == ["after0"]
    assert total == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("social.chat").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    # The #141 fix-up shape, pinned for the new ledger too: save()
    # must resolve its path INSIDE the try, so a home-resolution
    # failure degrades to a counted write failure, never raises.
    # NM_HOME must be unset: an absolute NM_HOME never calls
    # expanduser, which is how scratch-home batches mask this.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("social.chat").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        chat.record_failure("no-home", ValueError("boom"))
        assert chat.failure_count() == 1
        assert [f.site for f in chat.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("social.chat")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("social.chat").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_api_record_recent_count_clear_format():
    assert chat.failure_count() == 0
    assert chat.recent_failures() == []
    assert chat.format_failures() == "no social.chat failures recorded."
    chat.record_failure("chat.site-a", ValueError("boom"))
    chat.record_failure("chat.site-b", RuntimeError("bang"))
    assert chat.failure_count() == 2
    assert [f.site for f in chat.recent_failures()] == ["chat.site-b", "chat.site-a"]
    assert [f.site for f in chat.recent_failures(1)] == ["chat.site-b"]
    text = chat.format_failures()
    assert "2 total" in text and "chat.site-a" in text and "chat.site-b" in text
    chat.clear_failures()
    assert chat.failure_count() == 0
    assert chat.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    chat.record_failure("disk-a", ValueError("boom"))
    chat.record_failure("disk-b", RuntimeError("bang"))
    assert chat.failure_count() == 2
    assert chat._store.path().exists()
    assert chat._store.path().name == "social.chat.json"
    chat.clear_failures()
    assert chat.failure_count() == 0
    assert not chat._store.path().exists()


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    chat.record_failure("blocked", ValueError("boom"))
    assert chat.failure_count() == 1
    assert [f.site for f in chat.recent_failures()] == ["blocked"]
    assert fs.write_failures() > before


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                chat.record_failure(f"w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert chat.failure_count() == 400
    assert len(chat.recent_failures(1000)) == 200
    entries, total = chat._store.load()
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
        "from nomorals.social.chat import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.social.chat import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.social.chat import failures as f\n"
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
    store_path = home / "failures" / "social.chat.json"
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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

class _BrokenStr:
    def __str__(self):
        raise RuntimeError("broken __str__")


class _BrokenExc(Exception):
    @property
    def seconds(self):
        raise TypeError("no seconds")

    @property
    def status(self):
        raise TypeError("no status")


def test_converted_detect_platform_records_and_falls_back():
    from nomorals.social.chat.platforms import detect_platform
    assert detect_platform(_BrokenStr()) == "telegram"
    sites = [f.site for f in chat.recent_failures()]
    assert "platforms.detect_platform" in sites


def test_converted_style_kv_records_and_falls_back():
    from nomorals.social.chat.style import kv
    assert kv([(_BrokenStr(), "v")]) == ""
    sites = [f.site for f in chat.recent_failures()]
    assert "style.kv" in sites


def test_converted_sms_encoding_records_and_falls_back():
    from nomorals.social.chat.sms import sms_encoding
    assert sms_encoding(_BrokenStr()) == "ucs2"
    sites = [f.site for f in chat.recent_failures()]
    assert "sms.sms_encoding" in sites


def test_converted_doc_from_json_records_and_falls_back():
    from nomorals.social.chat.present import doc_from_json
    # A JSON string that parses but breaks construction: sections must
    # iterate, and 123 does not — the handler records and returns None.
    # (A non-string input returns None by design BEFORE the handler;
    # that early return is not a swallow and is not pinned here.)
    assert doc_from_json('{"v": 1, "sections": 123}') is None
    sites = [f.site for f in chat.recent_failures()]
    assert "present.doc_from_json" in sites


def test_converted_flood_wait_records_and_falls_back():
    from nomorals.social.chat.telegram import _flood_wait_seconds
    assert _flood_wait_seconds(_BrokenExc()) == 0
    sites = [f.site for f in chat.recent_failures()]
    assert "telegram._flood_wait_seconds" in sites
