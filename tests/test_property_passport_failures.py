"""Pass #162 pins: the property failure ledger (NEW) + passport conversions.

``nomorals/property/failures.py`` (namespace ``property``) is new in
this pass: no property ledger existed. The 17 durability tests below
mirror the #141+ shape exactly; the 20 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger
entry under its site name); the 3 exclusion pins prove the designed
probe outcome AND no ledger entry. No pin hardcodes scan totals or
passport counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Passport stores under test use
explicit tmp ``db_path`` values — never the real-home default store.
Every trigger was probed live before its pin was written.
"""
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.property import failures as pf
from nomorals.property import passport as pp


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    pf.clear_failures()
    yield
    pf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in pf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("property")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("property").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("property")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("property").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("property")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("property").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("property").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("property").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("property")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("property").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("property").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("property").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
        before = fs.write_failures()
        assert fs.Store("property").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("property")
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
    entries, _ = fs.Store("property").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
        before = fs.write_failures()
        pf.record_failure("no-home", ValueError("boom"))
        assert pf.failure_count() == 1
        assert [f.site for f in pf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []
    assert pf.format_failures() == "no property failures recorded."
    pf.record_failure("property.site-a", ValueError("boom"))
    pf.record_failure("property.site-b", RuntimeError("bang"))
    assert pf.failure_count() == 2
    assert [f.site for f in pf.recent_failures()] == ["property.site-b", "property.site-a"]
    assert [f.site for f in pf.recent_failures(1)] == ["property.site-b"]
    text = pf.format_failures()
    assert "2 total" in text and "property.site-a" in text and "property.site-b" in text
    pf.clear_failures()
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    pf.record_failure("property.disk", ValueError("boom"))
    entries, total = fs.Store("property").load()
    assert total == 1 and entries[0]["site"] == "property.disk"
    pf.clear_failures()
    assert fs.Store("property").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    pf.record_failure("property.mem", ValueError("boom"))
    assert pf.failure_count() == 1
    assert [f.site for f in pf.recent_failures()] == ["property.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                pf.record_failure(f"property.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert pf.failure_count() == 400
    assert len(pf.recent_failures(1000)) == 200
    entries, total = pf._store.load()
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
        "from nomorals.property import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.property import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.property import failures as f\n"
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
    store_path = fs.Store("property").path()
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

class _RaisingDB:
    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("boom")

    def commit(self):
        raise sqlite3.DatabaseError("boom")


def _store(tmp_path):
    return pp.PassportStore(str(tmp_path / "p.db"))


def test_pin_is_fresh():
    p = pp.RentalPassport(attested_at="garbage")
    assert p.is_fresh() is False
    assert "passport.is_fresh" in _sites()


def test_pin_freshness_note():
    p = pp.RentalPassport(attested_at="garbage")
    assert p.freshness_note() == "attestation: unknown"
    assert "passport.freshness_note" in _sites()


def test_pin_get(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.get("pp_x") is None
    assert "passport.get" in _sites()


def test_pin_list(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.list() == []
    assert "passport.list" in _sites()


def test_pin_set_income_band(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.set_income_band("pp_x", "100k-250k") is False
    assert "passport.set_income_band" in _sites()


def test_pin_set_kyc(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.set_kyc("pp_x", "summary") is False
    assert "passport.set_kyc" in _sites()


def _patched_get_store(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(pp.PassportStore, "get",
                        lambda self, pid: pp.RentalPassport(passport_id=pid))
    store._db = _RaisingDB()
    return store


def test_pin_add_history(tmp_path, monkeypatch):
    store = _patched_get_store(tmp_path, monkeypatch)
    assert store.add_history("pp_x", "12 Allen Ave") is False
    assert "passport.add_history" in _sites()


def test_pin_add_reference(tmp_path, monkeypatch):
    store = _patched_get_store(tmp_path, monkeypatch)
    assert store.add_reference("pp_x", "Ada") is False
    assert "passport.add_reference" in _sites()


def test_pin_add_doc_ref(tmp_path, monkeypatch):
    store = _patched_get_store(tmp_path, monkeypatch)
    assert store.add_doc_ref("pp_x", "vault:doc1") is False
    assert "passport.add_doc_ref" in _sites()


def test_pin_reattest(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.reattest("pp_x") is False
    assert "passport.reattest" in _sites()


class _BoomIter:
    def __iter__(self):
        raise RuntimeError("boom")


def test_pin_missing_kyc_docs():
    p = pp.RentalPassport(doc_refs=_BoomIter())
    assert pp.missing_kyc_docs(p) == list(pp.NIGERIA_KYC_DOCS)
    assert "passport.missing_kyc_docs" in _sites()


class _BoomLen:
    def __len__(self):
        raise RuntimeError("boom")


def test_pin_referencing_readiness():
    p = SimpleNamespace(income_band="100k-250k", kyc_summary="x",
                        history=_BoomLen(), references=[], doc_refs=[])
    out = pp.referencing_readiness(p)
    assert out == {"ok": False, "score": 0, "components": {},
                   "missing": ["no passport"], "verdict": "no passport"}
    assert "passport.referencing_readiness" in _sites()


def test_pin_affordability():
    out = pp.affordability("x", "100k-250k")
    assert out == {"ok": False, "reason": "couldn't assess affordability"}
    assert "passport.affordability" in _sites()


def test_pin_can_afford(monkeypatch):
    monkeypatch.setattr(pp, "affordability", _boom)
    assert pp.can_afford(250000000, "100k-250k") is False
    assert "passport.can_afford" in _sites()


def test_pin_required_monthly_income():
    assert pp.required_monthly_income("x") == 0
    assert "passport.required_monthly_income" in _sites()


def test_pin_required_income_band(monkeypatch):
    monkeypatch.setattr(pp, "can_afford", _boom)
    assert pp.required_income_band(250000000) == ""
    assert "passport.required_income_band" in _sites()


def test_pin_can_guarantee():
    assert pp.can_guarantee("x", "100k-250k") is False
    assert "passport.can_guarantee" in _sites()


def test_pin_combined_affordability():
    out = pp.combined_affordability("x", ["100k-250k"])
    assert out == {"ok": False, "reason": "couldn't assess combined "
                                          "affordability"}
    assert "passport.combined_affordability" in _sites()


class _FlakyMem:
    def __init__(self):
        self._data = {"owner_name": "Ada"}

    def get(self, key):
        if key == "income_band":
            raise RuntimeError("boom")
        return self._data.get(key)


def test_pin_memory_facts_key():
    ctx = SimpleNamespace(memory=_FlakyMem())
    out = pp._memory_facts(ctx)
    assert out == {"owner_name": "Ada"}
    assert "passport.memory_facts_key" in _sites()


class _BoomCtx:
    @property
    def memory(self):
        raise RuntimeError("boom")


def test_pin_memory_facts_outer():
    assert pp._memory_facts(_BoomCtx()) == {}
    assert "passport.memory_facts" in _sites()


# ---- Exclusion pins (designed outcome AND no record) ----------------------

def test_excl_fee_kobo_percent_after():
    # First %-strategy ("agency fee 10%"): the ValueError guard cannot
    # fire (float() on a regex-matched digit string never fails).
    assert pp._fee_kobo("agency fee 10% of rent", ["agency fee"],
                        25000000) == (2500000, True)
    assert _sites() == []


def test_excl_fee_kobo_percent_before():
    # Second %-strategy ("10% agency fee"): same dead-guard argument.
    assert pp._fee_kobo("10% agency fee", ["agency fee"],
                        25000000) == (2500000, True)
    assert _sites() == []


def test_excl_alter_idempotency(tmp_path):
    db = str(tmp_path / "p.db")
    pp.PassportStore(db)  # first init: ALTER adds attested_at
    pf.clear_failures()
    store = pp.PassportStore(db)  # second init: ALTER hits duplicate column
    assert store.get("nope") is None
    assert _sites() == []
