"""Pass #203 durability + NOT-recording pins for nomorals/games/whatsapp_format.py.

The games swallow ledger (nomorals/games/failures.py, namespace
``games``, cap 200) is REUSED here. All 13 failure sites are
exclusions, so this file carries no converted-site pins: each excluded
site must stay silent when its failure path is driven. The durability
half mirrors the #141+ shape: store-level round-trip, cap, clear,
namespace isolation, corrupt recovery, save-failure counting,
concurrent saves, plus ledger-level memory degradation, disk hits,
concurrent records, and subprocess restart persistence.

Docstrings in this file avoid spelling out the bare-swallow handler
form, because the error scanner counts test files too.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from nomorals.games import failures as gf
from nomorals.games import whatsapp_format as wf
from nomorals.core import failure_store as fs
from nomorals.social.chat.platforms import TELEGRAM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    gf.clear_failures()
    yield
    gf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in gf.recent_failures()]


class _Boom(Exception):
    pass


class _BadStr:
    def __bool__(self):
        return True

    def __str__(self):
        raise _Boom("no str")


class _IterRaise:
    def __iter__(self):
        raise _Boom("no iter")


class _GetRaise(dict):
    def get(self, *a, **k):
        raise _Boom("no get")


def _raise(*a, **k):
    raise _Boom("down")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("games")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("games").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("games")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("games").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("games")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("games").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("games").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("games").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("games")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("games").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("games").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("games").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(
        monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    import unittest.mock as mock
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("games").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("games")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("games").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(
        monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    import unittest.mock as mock
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        gf.record_failure("no-home", ValueError("boom"))
        assert gf.failure_count() == 1
        assert [f.site for f in gf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []
    assert gf.format_failures() == "no games failures recorded."
    gf.record_failure("whatsapp_format.site-a", ValueError("boom"))
    gf.record_failure("whatsapp_format.site-b", RuntimeError("bang"))
    assert gf.failure_count() == 2
    assert [f.site for f in gf.recent_failures()] == [
        "whatsapp_format.site-b", "whatsapp_format.site-a"]
    assert [f.site for f in gf.recent_failures(1)] == [
        "whatsapp_format.site-b"]
    text = gf.format_failures()
    assert ("2 total" in text and "whatsapp_format.site-a" in text
            and "whatsapp_format.site-b" in text)
    gf.clear_failures()
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    gf.record_failure("whatsapp_format.disk", ValueError("boom"))
    entries, total = fs.Store("games").load()
    assert total == 1 and entries[0]["site"] == "whatsapp_format.disk"
    gf.clear_failures()
    assert fs.Store("games").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    gf.record_failure("whatsapp_format.mem", ValueError("boom"))
    assert gf.failure_count() == 1
    assert [f.site for f in gf.recent_failures()] == ["whatsapp_format.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                gf.record_failure(f"whatsapp_format.w{k}-{i}",
                                  ValueError("boom"))
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert gf.failure_count() == 400
    assert len(gf.recent_failures(1000)) == 200
    entries, total = gf._store.load()
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
        "from nomorals.games import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.games import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.games import failures as f\n"
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
    store_path = fs.Store("games").path()
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


# ---- NOT-recording pins (one per excluded site) --------------------------
# Each drives the site's real failure path and asserts the games ledger
# stays empty: the exclusion holds, no record is written.

def test_notrec_is_whatsapp_false_on_unreadable_key():
    assert wf.is_whatsapp(_BadStr()) is False
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_platform_of_telegram_on_unreadable_key():
    assert wf.platform_of(_BadStr()) == TELEGRAM
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_format_game_text_untouched_on_convert_failure(monkeypatch):
    # Both callees never raise on real inputs (their own handlers catch
    # internally and the inner failure is traced in the social.chat
    # ledger), so this second-level handler is driven via a patched
    # dependency — see SURVEY.md dead-handler observation.
    monkeypatch.setattr(wf, "format_for_platform", _raise)
    assert wf.format_game_text("<b>hi</b>", "whatsapp:1") == "<b>hi</b>"
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_wa_chunks_single_on_chunk_failure(monkeypatch):
    monkeypatch.setattr(wf, "chunk_text", _raise)
    assert wf.wa_chunks("hello", limit=100) == ["hello"]
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_wa_bar_empty_on_bad_numbers():
    assert wf.wa_bar("x", "y") == ""
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_render_stats_outer_unavailable_on_bad_gear():
    assert wf.render_stats_wa("Ada", {}, [1]) == "*Ada* — stats unavailable"
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_get_inner_zero_on_unreadable_stat():
    out = wf.render_stats_wa("Ada", _GetRaise())
    assert "*Strength* 0" in out
    assert "Unspent points: *0*" in out
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_render_battle_outer_title_on_bad_rounds():
    assert wf.render_battle_wa("Duel", _IterRaise()) == "*⚔️ Duel*"
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_render_leaderboard_outer_title_on_bad_entry():
    assert wf.render_leaderboard_wa("Top", [42]) == "*🏆 Top*"
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_render_achievements_outer_name_on_bad_unlocks():
    assert wf.render_achievements_wa("Ada", _IterRaise()) == "*🏅 Ada*"
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_render_status_outer_game_on_bad_lines():
    assert wf.render_game_status_wa("chess", _IterRaise()) == "*🎮 chess*"
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_render_menu_outer_header_on_bad_entry():
    assert wf.render_game_menu_wa([42]) == "*🎮 Games*"
    assert gf.failure_count() == 0
    assert _sites() == []


def test_notrec_render_coins_outer_name_on_bad_items():
    assert wf.render_coins_wa("Ada", 250, _IterRaise()) == "*🪙 Ada*"
    assert gf.failure_count() == 0
    assert _sites() == []
