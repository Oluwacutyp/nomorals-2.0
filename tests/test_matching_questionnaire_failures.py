"""Pass #158 pins: the matching failure ledger (NEW) + questionnaire conversions.

No matching ledger existed before this pass: ``nomorals/matching/failures.py``
(namespace ``matching``) is created here, durability-wired through the shared
``nomorals/core/failure_store.py`` exactly like the #141–#157 ledgers. The
17 durability tests below mirror #157's shape exactly; the 25 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name). There are 0
exclusions. No pin hardcodes scan totals or questionnaire counts.
Every test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree), and every Questionnaire under test
gets an explicit tmp ``db_path`` (the module default lives under the real
home). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written.
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
from nomorals.matching import failures as mf
from nomorals.matching.questionnaire import DealBreaker, Question, Questionnaire


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mf.clear_failures()
    yield
    mf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in mf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _mkq(tmp_path, name="q"):
    return Questionnaire(owner="t", db_path=str(tmp_path / f"{name}.db"))


class _BrokenDb:
    """execute/commit always raise — trips every DB-touching handler."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("boom")

    def commit(self):
        raise RuntimeError("boom")


def _broken(q):
    q._db = _BrokenDb()
    return q


# ---- Store-level (durability mirror of #157) -----------------------------

def test_store_round_trip():
    store = fs.Store("matching")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("matching").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("matching")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("matching").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("matching")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("matching").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("matching").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("matching").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("matching")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("matching").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("matching").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("matching").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("matching").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("matching")
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
    entries, _ = fs.Store("matching").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        mf.record_failure("no-home", ValueError("boom"))
        assert mf.failure_count() == 1
        assert [f.site for f in mf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []
    assert mf.format_failures() == "no matching failures recorded."
    mf.record_failure("matching.site-a", ValueError("boom"))
    mf.record_failure("matching.site-b", RuntimeError("bang"))
    assert mf.failure_count() == 2
    assert [f.site for f in mf.recent_failures()] == ["matching.site-b", "matching.site-a"]
    assert [f.site for f in mf.recent_failures(1)] == ["matching.site-b"]
    text = mf.format_failures()
    assert "2 total" in text and "matching.site-a" in text and "matching.site-b" in text
    mf.clear_failures()
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    mf.record_failure("matching.disk", ValueError("boom"))
    entries, total = fs.Store("matching").load()
    assert total == 1 and entries[0]["site"] == "matching.disk"
    mf.clear_failures()
    assert fs.Store("matching").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mf.record_failure("matching.mem", ValueError("boom"))
    assert mf.failure_count() == 1
    assert [f.site for f in mf.recent_failures()] == ["matching.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mf.record_failure(f"matching.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert mf.failure_count() == 400
    assert len(mf.recent_failures(1000)) == 200
    entries, total = mf._store.load()
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
        "from nomorals.matching import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.matching import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.matching import failures as f\n"
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
    store_path = fs.Store("matching").path()
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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def test_converted_passes_records_and_fails_open(tmp_path):
    # Bad predicate value: float("xx") raises inside the le-branch, the
    # handler returns True — a broken predicate passes the candidate.
    breaker = DealBreaker(name="b", attribute="budget", predicate="le",
                          value="xx")
    assert breaker.passes({"budget": 100}) is True
    assert _sites() == ["questionnaire.passes"]


def test_converted_ensure_starters_records_and_returns_empty(tmp_path, monkeypatch):
    # A broken add_question is the only honest trigger: with a broken DB
    # alone, add_question catches internally and starters just filter out.
    monkeypatch.setattr(Questionnaire, "add_question", _boom)
    q = _mkq(tmp_path)
    assert q.ensure_starters() == []
    assert _sites() == ["questionnaire.ensure_starters"]


def test_converted_add_question_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.add_question("x", "text") == Question()
    assert _sites() == ["questionnaire.add_question"]


def test_converted_row_acceptable_records_and_returns_empty_list(tmp_path):
    q = _mkq(tmp_path)
    q.add_question("q1", "text")
    q._db.execute("UPDATE answers SET acceptable='{\"bad' WHERE question_id='q1'")
    q._db.commit()
    got = q.get("q1")
    assert got is not None and got.acceptable == []
    assert _sites() == ["questionnaire.row_acceptable"]


def test_converted_get_records_and_returns_none(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.get("anything") is None
    assert _sites() == ["questionnaire.get"]


def test_converted_answer_records_and_returns_false(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.answer("q", 3, "v") is False
    assert _sites() == ["questionnaire.answer"]


def test_converted_set_acceptable_records_and_returns_false(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.set_acceptable("q", ["a"]) is False
    assert _sites() == ["questionnaire.set_acceptable"]


def test_converted_unanswered_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.unanswered() == []
    assert _sites() == ["questionnaire.unanswered"]


def test_converted_answered_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.answered() == []
    assert _sites() == ["questionnaire.answered"]


def test_converted_weights_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.weights() == {}
    assert _sites() == ["questionnaire.weights"]


def test_converted_okcupid_weights_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.okcupid_weights() == {}
    assert _sites() == ["questionnaire.okcupid_weights"]


def test_converted_set_dealbreaker_records_and_returns_false(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.set_dealbreaker("b", "budget", "le", 100) is False
    assert _sites() == ["questionnaire.set_dealbreaker"]


def test_converted_remove_dealbreaker_records_and_returns_false(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.remove_dealbreaker("b") is False
    assert _sites() == ["questionnaire.remove_dealbreaker"]


def test_converted_dealbreakers_value_records_and_returns_none(tmp_path):
    q = _mkq(tmp_path)
    q._db.execute(
        "INSERT OR REPLACE INTO dealbreakers (owner, name, attribute,"
        " predicate, value) VALUES (?, 'b', 'budget', 'le', '{\"bad')",
        (q.owner,))
    q._db.commit()
    out = q.dealbreakers()
    assert len(out) == 1 and out[0].value is None
    assert _sites() == ["questionnaire.dealbreakers_value"]


def test_converted_dealbreakers_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.dealbreakers() == []
    assert _sites() == ["questionnaire.dealbreakers"]


def test_converted_boundary_questions_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.boundary_questions() == []
    assert _sites() == ["questionnaire.boundary_questions"]


def test_converted_set_ranking_records_and_returns_false(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.set_ranking("w", "proposer", ["a"]) is False
    assert _sites() == ["questionnaire.set_ranking"]


def test_converted_rankings_list_records_and_returns_empty_list(tmp_path):
    q = _mkq(tmp_path)
    q._db.execute(
        "INSERT OR REPLACE INTO ranks (owner, who, side, ranking,"
        " updated_at) VALUES (?, 'w', 'proposer', '{\"bad', 0.0)",
        (q.owner,))
    q._db.commit()
    assert q.rankings("proposer") == {"w": []}
    assert _sites() == ["questionnaire.rankings_list"]


def test_converted_rankings_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.rankings("proposer") == {}
    assert _sites() == ["questionnaire.rankings"]


def test_converted_set_capacity_records_and_returns_false(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.set_capacity("r", 2) is False
    assert _sites() == ["questionnaire.set_capacity"]


def test_converted_capacities_records_and_returns_empty(tmp_path):
    q = _broken(_mkq(tmp_path))
    assert q.capacities() == {}
    assert _sites() == ["questionnaire.capacities"]


def test_converted_filter_dealbreakers_records_and_returns_unfiltered(tmp_path):
    # Malformed candidate: int has no .get, so the comprehension raises
    # inside the filter body — the handler returns every candidate
    # unfiltered. A broken DB alone would NOT trip this site (dealbreakers
    # catches internally and returns []), so the malformed candidate is
    # the honest trigger.
    q = _mkq(tmp_path)
    assert q.set_dealbreaker("b", "budget", "le", 100) is True
    assert q.filter_dealbreakers([42]) == [42]
    assert _sites() == ["questionnaire.filter_dealbreakers"]


def test_converted_match_breakdown_records_and_returns_partial(tmp_path):
    # Non-dict attrs: attrs.get raises inside the scoring loop; the
    # handler returns the partial breakdown built so far (here: none).
    q = _mkq(tmp_path)
    q.add_question("q1", "text", attribute="remote")
    assert q.answer("q1", 3, "yes") is True
    assert q.match_breakdown(42) == []
    assert _sites() == ["questionnaire.match_breakdown"]


def test_converted_satisfaction_records_and_returns_neutral(tmp_path, monkeypatch):
    # match_breakdown never raises on its own (its own handler), so the
    # honest trigger is a broken breakdown underneath satisfaction.
    monkeypatch.setattr(Questionnaire, "match_breakdown", _boom)
    q = _mkq(tmp_path)
    assert q.satisfaction({}) == 0.5
    assert _sites() == ["questionnaire.satisfaction"]


def test_converted_score_records_and_returns_neutral(tmp_path, monkeypatch):
    monkeypatch.setattr(Questionnaire, "match_breakdown", _boom)
    q = _mkq(tmp_path)
    assert q.score({}) == 0.5
    assert _sites() == ["questionnaire.score"]
