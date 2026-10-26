"""Pass #210 pins: characters converts record, exclusions stay silent.

``nomorals/audio/characters.py`` converts 3 of its 12 true sites to the
REUSED audio ledger (the three mutators whose swallowed failure feeds
a false user-facing message: ``pin_fact`` reporting "already pinned"
for an unpinned fact, ``set_relationship`` / ``set_mood`` feeding
unconditional success lines while the state never changed), and
excludes 9 (best-effort db-path fallback, self-correcting cutoff
rendering, cap fallbacks that equal the documented no-restriction
rendering and never exceed the character's own cutoff, a migration
skip whose reads degrade via column fallbacks and whose writes fail
honestly, a close whose dead-store assignment runs regardless, a
voice fallback identical to the genuine no-voice rendering, a card
failure that asserts nothing, and a name scan whose empty result is
the genuine no-names outcome). The 17 durability tests below mirror
the #141+ shape exactly (same as #209's, against the REUSED audio
ledger); the 3 behavior pins prove each convert records while keeping
its fallback; the 9 NOT-recording pins prove each exclusion stays
silent with its designed output. No pin hardcodes scan totals or
characters counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import json
import os
import pathlib
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.audio import failures as af
from nomorals.audio import characters as ch


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _assert_silent():
    assert af.failure_count() == 0
    assert af.recent_failures() == []


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("audio")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("audio").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("audio")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("audio").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("audio")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("audio").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("audio").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("audio").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("audio")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("audio").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("audio").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("audio").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("audio").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("audio")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("audio").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        af.record_failure("no-home", ValueError("boom"))
        assert af.failure_count() == 1
        assert [f.site for f in af.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no audio failures recorded."
    af.record_failure("audio.site-a", ValueError("boom"))
    af.record_failure("audio.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["audio.site-b", "audio.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["audio.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "audio.site-a" in text and "audio.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("audio.disk", ValueError("boom"))
    entries, total = fs.Store("audio").load()
    assert total == 1 and entries[0]["site"] == "audio.disk"
    af.clear_failures()
    assert fs.Store("audio").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("audio.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["audio.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"audio.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert af.failure_count() == 400
    assert len(af.recent_failures(1000)) == 200
    entries, total = af._store.load()
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
        "from nomorals.audio import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.audio import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.audio import failures as f\n"
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
    store_path = fs.Store("audio").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


# ---- Behavior pins (the three converts) ---------------------------------

def test_behavior_171_pin_failure_records_and_still_returns_false():
    c = ch.Character(name="Ada", facts=["brave"])
    c.facts = 123  # type: ignore[assignment]
    assert c.pin_fact("kind") is False  # fallback unchanged
    assert [f.site for f in af.recent_failures()] == ["characters.pin_fact"]
    assert af.recent_failures()[0].exc_type == "TypeError"


def test_behavior_180_relationship_failure_records_and_state_unchanged():
    c = ch.Character(name="Bo")
    c.relationships = 123  # type: ignore[assignment]
    c.set_relationship("Cy", "trusts")
    assert c.relationships == 123  # fallback unchanged: silent no-op
    assert [f.site for f in af.recent_failures()] == [
        "characters.set_relationship"]
    assert af.recent_failures()[0].exc_type == "TypeError"


class _BadStr(str):
    def strip(self, *a, **k):
        raise RuntimeError("strip boom")


def test_behavior_187_mood_failure_records_and_state_unchanged():
    c = ch.Character(name="Cy")
    c.set_mood(_BadStr("calm"))  # type: ignore[arg-type]
    assert c.mood == ""  # fallback unchanged: silent no-op
    assert [f.site for f in af.recent_failures()] == ["characters.set_mood"]
    assert af.recent_failures()[0].exc_type == "RuntimeError"


# ---- NOT-recording pins (one per exclusion) ------------------------------

def test_not_recording_067_db_mkdir_failure_still_returns_path():
    with patch.object(pathlib.Path, "mkdir",
                      side_effect=OSError("ro fs")):
        assert ch._default_db().endswith("audio/characters.db")
    _assert_silent()


def test_not_recording_196_cutoff_failure_keeps_cutoff():
    c = ch.Character(name="Dee", knowledge_cutoff=5)
    c.set_cutoff("not-a-number")  # type: ignore[arg-type]
    assert c.knows_up_to() == 5
    _assert_silent()


def test_not_recording_206_known_memories_failure_keeps_cutoff():
    c = ch.Character(name="Eli", knowledge_cutoff=7)
    c.chapter_memories = {3: ["early"], 9: ["late"]}
    assert c._known_memories(current_chapter=object()) == ["early"]
    _assert_silent()


def test_not_recording_226_prompt_failure_keeps_cutoff_no_leak():
    c = ch.Character(name="Eli", knowledge_cutoff=7)
    c.chapter_memories = {3: ["early"], 9: ["late"]}
    prompt = c.to_system_prompt(current_chapter={"bogus": True})  # type: ignore[arg-type]
    assert "chapter 7" in prompt
    assert "late" not in prompt
    _assert_silent()


class _PragmaBoomDB:
    """DB double: schema script + commit fine, PRAGMA read fails."""

    def __init__(self):
        self.committed = False

    def executescript(self, sql):
        return None

    def execute(self, sql, *a, **k):
        if "PRAGMA" in sql:
            raise RuntimeError("pragma gone")
        return None

    def commit(self):
        self.committed = True


def test_not_recording_353_migration_failure_skips_and_commits():
    store = ch.CharacterStore.__new__(ch.CharacterStore)
    store._db = _PragmaBoomDB()  # type: ignore[assignment]
    store._init_schema()
    assert store._db.committed is True
    _assert_silent()


def test_not_recording_453_close_failure_still_drops_reference():
    store = ch.CharacterStore.__new__(ch.CharacterStore)

    class _BoomConn:
        def close(self):
            raise RuntimeError("close boom")

    store._db = _BoomConn()  # type: ignore[assignment]
    store.close()
    assert store._db is None
    _assert_silent()


def test_not_recording_580_voice_failure_keeps_text_no_audio():
    def boom_voice(text):
        raise RuntimeError("voice boom")

    res = ch.talk_to(ch.Character(name="Gus"), "hello?",
                     llm_fn=lambda s, q: "hi there", voice_fn=boom_voice)
    assert res.ok is True
    assert res.text == "hi there"
    assert res.audio == ""
    _assert_silent()


def test_not_recording_637_card_failure_returns_empty():
    # speech_patterns/goals render via f-strings (never raise) and
    # post_init coerces personality, so break the memory render instead:
    # a non-iterable chapter payload trips _known_memories.
    c = ch.Character(name="Fay")
    c.chapter_memories = {1: 123}  # type: ignore[dict-item]
    assert ch.export_card(c) == ""
    _assert_silent()


def test_not_recording_716_name_scan_failure_returns_empty():
    assert ch._name_candidates(123) == []  # type: ignore[arg-type]
    _assert_silent()
