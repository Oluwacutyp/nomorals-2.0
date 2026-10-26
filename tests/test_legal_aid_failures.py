"""Pass #212 pins: aid billing failures are recorded, everything else stays silent.

``nomorals/legal/aid.py`` converts 1 of its 12 true sites and excludes
11: the corpus/citation helpers map an unavailable index to the same
value a missing corpus or a genuine miss produces, the rendering
fallbacks degrade to the disclaimer alone (an omission, never a claim),
triage maps failure to the genuine no-match dict, and SMS chunking
keeps the full text. The one convert is 914 ``bill_answer``: 0 is never
a genuine success value (rates are 1400/8400 kobo), so a failed bill
reporting 0 is a false cost claim on a money path — it is recorded in
the REUSED legal ledger under ``aid.bill_answer`` while the fallback
(still 0, still never raises otherwise) is unchanged.

The 17 durability tests below mirror the #141+ shape exactly (same as
#209/#210/#211, against the REUSED legal ledger); the behavior pin
proves the convert records with its designed output; the 11
NOT-recording pins prove each exclusion stays silent with its designed
output. No pin hardcodes scan totals or aid counts. Every test
redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.legal import aid
from nomorals.legal import failures as lf


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    lf.clear_failures()
    aid._index = None
    yield
    lf.clear_failures()
    aid._index = None


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"}
            for i in range(n)]


def _assert_silent():
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("legal")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("legal").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("legal")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("legal").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("legal")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("legal").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("legal").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("legal").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("legal")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("legal").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("legal").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("legal").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("legal").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("legal")
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
    entries, _ = fs.Store("legal").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        lf.record_failure("no-home", ValueError("boom"))
        assert lf.failure_count() == 1
        assert [f.site for f in lf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []
    assert lf.format_failures() == "no legal failures recorded."
    lf.record_failure("aid.site-a", ValueError("boom"))
    lf.record_failure("aid.site-b", RuntimeError("bang"))
    assert lf.failure_count() == 2
    assert [f.site for f in lf.recent_failures()] == ["aid.site-b",
                                                      "aid.site-a"]
    assert [f.site for f in lf.recent_failures(1)] == ["aid.site-b"]
    text = lf.format_failures()
    assert "2 total" in text and "aid.site-a" in text \
        and "aid.site-b" in text
    lf.clear_failures()
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    lf.record_failure("aid.disk", ValueError("boom"))
    entries, total = fs.Store("legal").load()
    assert total == 1 and entries[0]["site"] == "aid.disk"
    lf.clear_failures()
    assert fs.Store("legal").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    lf.record_failure("aid.mem", ValueError("boom"))
    assert lf.failure_count() == 1
    assert [f.site for f in lf.recent_failures()] == ["aid.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                lf.record_failure(f"aid.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert lf.failure_count() == 400
    assert len(lf.recent_failures(1000)) == 200
    entries, total = lf._store.load()
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
        "from nomorals.legal import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.legal import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.legal import failures as f\n"
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
    store_path = fs.Store("legal").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


# ---- Behavior pin (the one convert) --------------------------------------

def test_behavior_914_bill_failure_records_but_still_returns_zero(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    # Tracker construction fails (parent is a file) → fallback 0, recorded.
    assert aid.bill_answer("2348000000000",
                           db_path=str(blocker / "cost.db")) == 0
    assert lf.failure_count() == 1
    entry = lf.recent_failures()[0]
    assert entry.site == "aid.bill_answer"
    assert entry.exc_type == "OperationalError"
    # Healthy billing stays silent and reports the real rate.
    lf.clear_failures()
    assert aid.bill_answer("2348000000000",
                           db_path=str(tmp_path / "cost.db")) == 1400
    _assert_silent()
    # Caller bugs still surface, never recorded as a billing failure.
    with pytest.raises(ValueError):
        aid.bill_answer("2348000000000", category="bogus")
    _assert_silent()


# ---- NOT-recording pins (one per exclusion) ------------------------------

def test_not_recording_470_bad_doc_skipped_index_still_builds(monkeypatch):
    class _BoomIndex:
        def __init__(self):
            self.docs = []

        def add(self, doc):
            raise RuntimeError("doc down")

        def __len__(self):
            return len(self.docs)

    monkeypatch.setattr("nomorals.documents.index.DocumentIndex", _BoomIndex)
    assert aid._load_index() is None
    _assert_silent()


def test_not_recording_474_index_construction_failure_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("index down")

    monkeypatch.setattr("nomorals.documents.index.DocumentIndex", _boom)
    assert aid._load_index() is None
    _assert_silent()


def test_not_recording_486_search_failure_returns_empty(monkeypatch):
    class _BoomSearch:
        def search(self, *a, **k):
            raise RuntimeError("search down")

    monkeypatch.setattr(aid, "_load_index", lambda: _BoomSearch())
    assert aid.corpus_search("tenancy") == []
    _assert_silent()


def test_not_recording_552_malformed_hit_keeps_cites_so_far(monkeypatch):
    scenario = aid._Scenario(
        id="t", patterns=(), corpus_docs=("lagos_tenancy_law.md",),
        escalate=True, core={"en": "core"})
    # First doc's hit is malformed (no .get) → 552 fires, cites stay [].
    monkeypatch.setattr(aid, "corpus_search", lambda *a, **k: [object()])
    assert aid._citations_for(scenario, "my landlord locked me out") == []
    _assert_silent()


def test_not_recording_600_researcher_failure_falls_back_honestly():
    def _boom(q):
        raise RuntimeError("research down")

    ans = aid.answer_legal_question(
        "what is the capital of France",
        researcher=SimpleNamespace(research=_boom))
    assert ans.escalate is True
    assert "couldn't match" in ans.text
    _assert_silent()


def test_not_recording_641_render_failure_returns_disclaimer_only():
    class _BoomAnswer:
        def render(self):
            raise RuntimeError("render down")

    from nomorals.legal.contracts import DISCLAIMER
    assert aid.format_answer(_BoomAnswer()) == DISCLAIMER
    _assert_silent()


def test_not_recording_650_sms_render_failure_returns_disclaimer_chunk():
    class _BoomAnswer:
        def render(self):
            raise RuntimeError("render down")

    from nomorals.legal.contracts import DISCLAIMER
    assert aid.format_answer_sms(_BoomAnswer()) == [DISCLAIMER]
    _assert_silent()


def test_not_recording_725_triage_failure_returns_empty_match(monkeypatch):
    def _boom(q):
        raise RuntimeError("match down")

    monkeypatch.setattr(aid, "_match_scenario", _boom)
    assert aid.triage("my landlord locked me out") == {
        "match": None, "candidates": [], "questions": []}
    _assert_silent()


def test_not_recording_772_referral_failure_returns_disclaimer(monkeypatch):
    monkeypatch.setattr(aid, "_LEGAL_AID_DIRECTORY", (None,))
    from nomorals.legal.contracts import DISCLAIMER
    assert aid.refer_lawyer("tenancy") == DISCLAIMER
    _assert_silent()


def test_not_recording_858_letter_template_failure_returns_disclaimer(
        monkeypatch):
    monkeypatch.setattr(aid, "_LETTERS", {
        "x": {"title": "t", "fields": (), "template": "{missing}"}})
    from nomorals.legal.contracts import DISCLAIMER
    assert aid.render_letter("x") == DISCLAIMER
    _assert_silent()


def test_not_recording_888_chunk_failure_returns_full_text_unchunked():
    class _BoomText:
        def split(self):
            raise RuntimeError("split down")

    boom = _BoomText()
    assert aid.chunk_for_sms(boom) == [boom]
    _assert_silent()
