"""Pass #161 pins: the games failure ledger (NEW) + trivia_forge conversions.

``nomorals/games/failures.py`` (namespace ``games``) is new in this
pass: the pre-existing ``engine_failures.py`` is engine-scoped
(namespace ``games.engine``, imported only by ``engine.py``) and is NOT
reused for trivia. The 17 durability tests below mirror the #141+
shape exactly; the 16 site pins prove each conversion records without
changing flow (fallback value unchanged AND a ledger entry under its
site name); the 7 exclusion pins prove the designed probe outcome AND
no ledger entry. No pin hardcodes scan totals or trivia_forge counts.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Restart tests use real
subprocesses. Every trigger was probed live before its pin was written.
Forge/bank objects under test are built with ``db=None`` (``_kv`` None)
or explicit fake KV objects — never the real-home default store.
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
from nomorals.games import failures as gf
from nomorals.games import trivia_forge as tf


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


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


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


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("games").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("games")
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
    entries, _ = fs.Store("games").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        gf.record_failure("no-home", ValueError("boom"))
        assert gf.failure_count() == 1
        assert [f.site for f in gf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []
    assert gf.format_failures() == "no games failures recorded."
    gf.record_failure("games.site-a", ValueError("boom"))
    gf.record_failure("games.site-b", RuntimeError("bang"))
    assert gf.failure_count() == 2
    assert [f.site for f in gf.recent_failures()] == ["games.site-b", "games.site-a"]
    assert [f.site for f in gf.recent_failures(1)] == ["games.site-b"]
    text = gf.format_failures()
    assert "2 total" in text and "games.site-a" in text and "games.site-b" in text
    gf.clear_failures()
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    gf.record_failure("games.disk", ValueError("boom"))
    entries, total = fs.Store("games").load()
    assert total == 1 and entries[0]["site"] == "games.disk"
    gf.clear_failures()
    assert fs.Store("games").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    gf.record_failure("games.mem", ValueError("boom"))
    assert gf.failure_count() == 1
    assert [f.site for f in gf.recent_failures()] == ["games.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                gf.record_failure(f"games.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
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


# ---- Site pins ------------------------------------------------------------

class _RaisingKV:
    def get(self, *args, **kwargs):
        raise RuntimeError("kv boom")

    def set(self, *args, **kwargs):
        raise RuntimeError("kv boom")


class _StubLedger:
    def __init__(self, themes=None, served=None, cooldown_raises=False):
        self._themes = themes if themes is not None else []
        self._served = served if served is not None else []
        self._cooldown_raises = cooldown_raises

    def eligible_themes(self, now=None):
        raise RuntimeError("ledger boom")

    def served_items(self, since=None):
        return self._served

    def theme_in_cooldown(self, theme):
        if self._cooldown_raises:
            raise RuntimeError("ledger boom")
        return False


class _OkLedger(_StubLedger):
    def eligible_themes(self, now=None):
        return self._themes


def _forge(suggest=None):
    return tf.TriviaForge(suggest=suggest, db=None, seed=1)


def test_pin_independent_solve_suggest():
    out = tf._independent_solve("2 + 2?", _boom)
    assert out == (None, "independent solve failed", "")
    assert "trivia_forge.independent_solve_suggest" in _sites()


def test_pin_maths_distractors_transpose(monkeypatch):
    real_float = float

    def guarded_float(arg, *args, **kwargs):
        # Only the digit-transposition candidate is a str; every other
        # float() call in maths_distractors takes a number.
        if isinstance(arg, str) and arg == "24":
            raise ValueError("boom")
        return real_float(arg, *args, **kwargs)

    monkeypatch.setattr("builtins.float", guarded_float)
    out = tf.maths_distractors(42.0, "42", __import__("random").Random(1))
    assert len(out) == 3
    assert "trivia_forge.maths_distractors_transpose" in _sites()


def test_pin_load_seen():
    forge = _forge()
    forge._kv = _RaisingKV()
    forge._load_seen()
    assert forge._seen == []
    assert "trivia_forge.load_seen" in _sites()


def test_pin_save_seen():
    forge = _forge()
    forge._kv = _RaisingKV()
    forge._seen = ["abc"]
    forge._save_seen()  # must not raise
    assert "trivia_forge.save_seen" in _sites()


def test_pin_ask_model_suggest():
    forge = _forge(suggest=_boom)
    assert forge._ask_model("science", "normal", 2) == []
    assert "trivia_forge.ask_model_suggest" in _sites()


def test_pin_blocked_themes():
    forge = _forge()
    forge._context = object()
    forge._ledger_obj = _StubLedger(cooldown_raises=True)
    assert forge._blocked_themes([{"theme": "mythology"}]) == set()
    assert "trivia_forge.blocked_themes" in _sites()


def test_pin_load_bank():
    forge = _forge()
    forge._kv = _RaisingKV()
    assert forge.bank._load_bank() == ([], [], 0)
    assert "trivia_forge.load_bank" in _sites()


def test_pin_save_bank():
    forge = _forge()
    forge._kv = _RaisingKV()
    forge.bank._save_bank([], [], 0)  # must not raise
    assert "trivia_forge.save_bank" in _sites()


def test_pin_load_stats():
    forge = _forge()
    forge._kv = _RaisingKV()
    stats = forge.bank._load_stats()
    assert stats == {"candidates": 0, "admitted": 0, "discarded": 0,
                     "reasons": {}, "ticks": 0, "last_tick": 0.0}
    assert "trivia_forge.load_stats" in _sites()


def test_pin_save_stats():
    forge = _forge()
    forge._kv = _RaisingKV()
    forge.bank._save_stats({"candidates": 1})  # must not raise
    assert "trivia_forge.save_stats" in _sites()


def _tick_harness(monkeypatch, ledger, candidates):
    """Forge+bank wired for _tick_library with stubbed ledger/sources."""
    forge = _forge(suggest=_boom)
    forge._context = object()
    forge._ledger_obj = ledger
    bank = forge.bank
    monkeypatch.setattr("nomorals.agents.teachings.library_candidates",
                        lambda ctx, themes: list(candidates))
    bank._research_items = lambda ctx: []
    return forge, bank


def test_pin_tick_library_themes(monkeypatch):
    forge, bank = _tick_harness(monkeypatch, _StubLedger(), [])
    # eligible_themes raises on the stub; served_items returns [].
    bank._tick_library({"notes": [], "errors": []}, lambda *a, **k: None,
                       max_passages=2, now=1000.0)
    assert "trivia_forge.tick_library_themes" in _sites()


def test_pin_tick_library_served(monkeypatch):
    forge, bank = _tick_harness(monkeypatch, _OkLedger(themes=["t"]), [])
    # eligible_themes fine; served_items on _OkLedger returns [] here —
    # swap in a raising served_items to trip the served site.
    class _RaisingServed(_OkLedger):
        def served_items(self, since=None):
            raise RuntimeError("ledger boom")

    forge._ledger_obj = _RaisingServed(themes=["t"])
    bank._tick_library({"notes": [], "errors": []}, lambda *a, **k: None,
                       max_passages=2, now=1000.0)
    assert "trivia_forge.tick_library_served" in _sites()


def test_pin_tick_library_suggest(monkeypatch):
    passage = "P" * 200
    item = {"source": {"type": "book", "ref": "r1", "label": "L",
                       "passage": passage}}
    forge, bank = _tick_harness(monkeypatch, _OkLedger(themes=["t"]), [item])
    counts = []
    bank._tick_library({"notes": [], "errors": []},
                       lambda ok, reason, src: counts.append((ok, reason, src)),
                       max_passages=2, now=1000.0)
    assert (False, "unparseable", "library") in counts
    assert "trivia_forge.tick_library_suggest" in _sites()


def test_pin_research_items_kg(monkeypatch):
    class _BoomKG:
        def __init__(self, db):
            raise RuntimeError("kg boom")

    monkeypatch.setattr("nomorals.agents.kg.KnowledgeGraph", _BoomKG)
    bank = _forge().bank
    assert bank._research_items(object()) == []
    assert "trivia_forge.research_items_kg" in _sites()


def test_pin_research_items_node(monkeypatch):
    class _NodeKG:
        def __init__(self, db):
            pass

        def top(self, ntype=None, n=None):
            return [123]  # not a dict: .get raises AttributeError

    monkeypatch.setattr("nomorals.agents.kg.KnowledgeGraph", _NodeKG)
    bank = _forge().bank
    assert bank._research_items(object()) == []
    assert "trivia_forge.research_items_node" in _sites()


def test_pin_tick_maths_llm_suggest():
    forge = _forge(suggest=_boom)
    counts = []
    forge.bank._tick_maths_llm(
        {"notes": [], "errors": []},
        lambda ok, reason, src: counts.append((ok, reason, src)),
        max_items=1, now=1000.0)
    assert (False, "unparseable", "maths_llm") in counts
    assert "trivia_forge.tick_maths_llm_suggest" in _sites()


# ---- Exclusion pins (designed outcome AND no record) ----------------------

def test_excl_safe_eval_arith_gate():
    assert tf.safe_eval_arith("import os") is None  # disallowed node
    assert tf.safe_eval_arith("2 +") is None  # unparseable
    assert _sites() == []


def test_excl_independent_solve_json():
    out = tf._independent_solve("2 + 2?", lambda p: "not json at all")
    assert out == (None, "independent solve unparseable", "")
    assert _sites() == []


def test_excl_ask_model_json():
    forge = _forge(suggest=lambda p: "[{oops]")
    assert forge._ask_model("science", "normal", 2) == []
    assert _sites() == []


def test_excl_interest_profile(monkeypatch):
    monkeypatch.setattr("nomorals.agents.arena.activity.interest_profile",
                        _boom)
    forge = _forge()
    forge._kv = type("KV", (), {"_db": object()})()
    assert forge._interest_profile() is None
    assert _sites() == []


def test_excl_ledger_enhancement(monkeypatch):
    class _BoomLedger:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ledger boom")

    monkeypatch.setattr("nomorals.agents.teachings.TeachingsLedger",
                        _BoomLedger)
    forge = _forge()
    forge._context = object()
    assert forge._ledger() is None
    assert _sites() == []


def test_excl_tick_library_json(monkeypatch):
    passage = "P" * 200
    item = {"source": {"type": "book", "ref": "r1", "label": "L",
                       "passage": passage}}
    forge = _forge(suggest=lambda p: '{"question": broken}')
    forge._context = object()
    forge._ledger_obj = _OkLedger(themes=["t"])
    bank = forge.bank
    monkeypatch.setattr("nomorals.agents.teachings.library_candidates",
                        lambda ctx, themes: [item])
    bank._research_items = lambda ctx: []
    counts = []
    bank._tick_library({"notes": [], "errors": []},
                       lambda ok, reason, src: counts.append((ok, reason, src)),
                       max_passages=2, now=1000.0)
    assert (False, "unparseable", "library") in counts
    assert _sites() == []


def test_excl_tick_maths_llm_json():
    forge = _forge(suggest=lambda p: '{"question": broken}')
    counts = []
    forge.bank._tick_maths_llm(
        {"notes": [], "errors": []},
        lambda ok, reason, src: counts.append((ok, reason, src)),
        max_items=1, now=1000.0)
    assert (False, "unparseable", "maths_llm") in counts
    assert _sites() == []
