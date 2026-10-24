"""Pass #149 pins: the NEW legal failure ledger + conversions.

The legal ledger did not exist before this pass; it is created
here, durability-wired through the shared failure_store helper (the
#141 shape: Store.save() resolves its path INSIDE the try).
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through ``nomorals.legal.failures``; restart tests use real
subprocesses (a fresh interpreter is the honest simulated restart —
no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name. The excluded handlers (pure-parse
probes returning the designed None, the user-input months probe whose
default is printed in the reply) are pinned as NOT recording.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.legal import failures as lf
from nomorals.legal import portfolio as portfolio_mod
from nomorals.legal.portfolio import ContractPortfolio
from nomorals.legal.contracts import review_contract

TENANCY = """TENANCY AGREEMENT
This tenancy agreement is between the landlord and the tenant for the
premises at 12 Adeola Odeku Street, Victoria Island, Lagos.
1. TERM: The tenancy shall commence on 1st January 2026 and run for a
period of 12 months.
2. RENT: Annual rent of ₦2,400,000 payable yearly.
3. RENEWAL: This agreement shall automatically renew for another 12
months unless either party gives notice in writing before 1st November 2026.
4. AGENCY FEE: 25% agency fee payable on signing.
"""


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    lf.clear_failures()
    yield
    lf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in lf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _portfolio():
    return ContractPortfolio(
        db_path=os.path.join(tempfile.mkdtemp(), "portfolio.db"))


class _BrokenDb:
    """execute() always raises; everything else is inert."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("db gone")

    def commit(self):
        return None


class _FakeRow:
    """sqlite-Row double: keys() honest, __getitem__ fails on demand."""

    def __init__(self, fail_on=()):
        self._fail = set(fail_on)
        self._vals = {"id": "c1", "name": "N", "contract_type": "tenancy",
                      "parties": "A", "grade": "A", "score": 90,
                      "findings_count": 0, "created_at": 0.0,
                      "auto_renew": 1, "annual_value": 100.0,
                      "supersedes": "old"}

    def keys(self):
        return list(self._vals)

    def __getitem__(self, key):
        if key in self._fail:
            raise RuntimeError("boom")
        return self._vals[key]


# ---- Store-level -------------------------------------------------------

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
    # The #141 fix-up shape: save() must resolve its path INSIDE the
    # try, so a home-resolution failure degrades to a counted write
    # failure, never raises. NM_HOME must be unset: an absolute
    # NM_HOME never calls expanduser, which is how scratch-home
    # batches mask this.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("legal").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        lf.record_failure("no-home", ValueError("boom"))
        assert lf.failure_count() == 1
        assert [f.site for f in lf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("legal")
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
    entries, _ = fs.Store("legal").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_api_record_recent_count_clear_format():
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []
    assert lf.format_failures() == "no legal failures recorded."
    lf.record_failure("legal.site-a", ValueError("boom"))
    lf.record_failure("legal.site-b", RuntimeError("bang"))
    assert lf.failure_count() == 2
    assert [f.site for f in lf.recent_failures()] == ["legal.site-b", "legal.site-a"]
    assert [f.site for f in lf.recent_failures(1)] == ["legal.site-b"]
    text = lf.format_failures()
    assert "2 total" in text and "legal.site-a" in text and "legal.site-b" in text
    lf.clear_failures()
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    lf.record_failure("legal.disk", ValueError("boom"))
    entries, total = fs.Store("legal").load()
    assert total == 1 and entries[0]["site"] == "legal.disk"
    lf.clear_failures()
    assert fs.Store("legal").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    lf.record_failure("legal.mem", ValueError("boom"))
    assert lf.failure_count() == 1
    assert [f.site for f in lf.recent_failures()] == ["legal.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                lf.record_failure(f"legal.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
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
    # path() resolves against the test process env, not the
    # subprocess home — resolve it under the subprocess home instead.
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

def test_converted_parse_dates_records_and_returns_empty(monkeypatch):
    monkeypatch.setattr(portfolio_mod, "_to_ts", _boom)
    assert portfolio_mod._parse_dates("1st January 2026") == []
    assert _sites() == ["portfolio._parse_dates"]


def test_converted_add_contract_money_records_and_zeroes_value(monkeypatch):
    p = _portfolio()
    review = review_contract(TENANCY, "tenancy")
    monkeypatch.setattr("nomorals.legal.contracts._money", _boom)
    c = p.add_contract(review, doc_text=TENANCY, name="Lekki Flat",
                       parties="Landlord / Tenant")
    assert c is not None
    assert c.annual_value == 0.0
    assert _sites() == ["portfolio.ContractPortfolio.add_contract.money"]


def test_converted_add_raw_review_records_and_returns_none(monkeypatch):
    p = _portfolio()
    monkeypatch.setattr(portfolio_mod, "review_contract", _boom)
    assert p.add_raw(TENANCY, name="Lekki Flat") is None
    assert _sites() == ["portfolio.ContractPortfolio.add_raw.review"]


def test_converted_row_auto_renew_records_and_returns_false():
    c = ContractPortfolio._row_to_contract(_FakeRow(fail_on={"auto_renew"}))
    assert c.auto_renew is False
    assert _sites() == ["portfolio.ContractPortfolio._row_to_contract.auto_renew"]


def test_converted_row_annual_value_records_and_returns_zero():
    c = ContractPortfolio._row_to_contract(_FakeRow(fail_on={"annual_value"}))
    assert c.annual_value == 0.0
    assert _sites() == ["portfolio.ContractPortfolio._row_to_contract.annual_value"]


def test_converted_row_supersedes_records_and_returns_empty():
    c = ContractPortfolio._row_to_contract(_FakeRow(fail_on={"supersedes"}))
    assert c.supersedes == ""
    assert _sites() == ["portfolio.ContractPortfolio._row_to_contract.supersedes"]


def test_converted_list_contracts_records_and_returns_empty():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.list_contracts() == []
    assert _sites() == ["portfolio.ContractPortfolio.list_contracts"]


def test_converted_get_contract_records_and_returns_none():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.get_contract("c1") is None
    assert _sites() == ["portfolio.ContractPortfolio.get_contract"]


def test_converted_obligations_confirmed_records_and_returns_false():
    p = _portfolio()

    # FakeRow lacks the full obligation column set; fill the rest.
    row = _FakeRow(fail_on={"confirmed"})
    row._vals.update({"kind": "renewal", "description": "d",
                      "due_at": 9999999999.0, "done": 0, "done_at": 0.0,
                      "source": "extracted", "contract_id": "c1",
                      "confirmed": 0})

    class _RowsDb2:
        def execute(self, *args, **kwargs):
            return self

        def fetchall(self):
            return [row]

    p._db = _RowsDb2()
    out = p.obligations()
    assert len(out) == 1
    assert out[0].confirmed is False
    assert _sites() == ["portfolio.ContractPortfolio.obligations.confirmed"]


def test_converted_obligations_records_and_returns_empty():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.obligations() == []
    assert _sites() == ["portfolio.ContractPortfolio.obligations"]


def test_converted_timeline_records_and_returns_empty(monkeypatch):
    p = _portfolio()
    monkeypatch.setattr(p, "list_contracts", lambda: [])
    monkeypatch.setattr(p, "obligations", lambda **k: [
        SimpleNamespace(due_at=None, contract_id="", kind="renewal",
                        description="d", id="o1")])
    assert p.timeline() == []
    assert _sites() == ["portfolio.ContractPortfolio.timeline"]


def test_converted_needs_attention_records_and_returns_empty(monkeypatch):
    p = _portfolio()
    monkeypatch.setattr(p, "list_contracts", lambda: [])
    monkeypatch.setattr(p, "obligations", lambda **k: [
        SimpleNamespace(due_at=None, contract_id="", kind="renewal",
                        description="d", id="o1")])
    assert p.needs_attention() == []
    assert _sites() == ["portfolio.ContractPortfolio.needs_attention"]


def test_converted_track_obligation_records_and_returns_none(monkeypatch):
    p = _portfolio()
    monkeypatch.setattr(p, "get_contract", lambda cid: SimpleNamespace(id="x"))
    p._db = _BrokenDb()
    assert p.track_obligation("c1", "Pay rent", 9999999999.0) is None
    assert _sites() == ["portfolio.ContractPortfolio.track_obligation"]


def test_converted_mark_done_records_and_returns_false():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.mark_done("o1") is False
    assert _sites() == ["portfolio.ContractPortfolio.mark_done"]


def test_converted_confirm_obligation_records_and_returns_false():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.confirm_obligation("o1") is False
    assert _sites() == ["portfolio.ContractPortfolio.confirm_obligation"]


def test_converted_correct_obligation_records_and_returns_false():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.correct_obligation("o1", description="fixed") is False
    assert _sites() == ["portfolio.ContractPortfolio.correct_obligation"]


def test_converted_remove_obligation_records_and_returns_false():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.remove_obligation("o1") is False
    assert _sites() == ["portfolio.ContractPortfolio.remove_obligation"]


def test_converted_unconfirmed_records_and_returns_empty():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.unconfirmed_obligations() == []
    assert _sites() == ["portfolio.ContractPortfolio.unconfirmed_obligations"]


def test_converted_search_records_and_returns_empty():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.search_contracts("rent") == []
    assert _sites() == ["portfolio.ContractPortfolio.search_contracts"]


def test_converted_renewal_pipeline_records_and_returns_empty_buckets(monkeypatch):
    p = _portfolio()
    monkeypatch.setattr(p, "list_contracts",
                        lambda: [SimpleNamespace(id="c1")])
    monkeypatch.setattr(p, "obligations", lambda **k: [
        SimpleNamespace(kind="renewal", contract_id="c1", due_at=None,
                        id="o1")])
    out = p.renewal_pipeline()
    assert out == {"overdue": [], "30": [], "60": [], "90": [], "later": []}
    assert _sites() == ["portfolio.ContractPortfolio.renewal_pipeline"]


def test_converted_roll_forward_records_and_returns_zero():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.roll_forward("c1", 12.0) == 0
    assert _sites() == ["portfolio.ContractPortfolio.roll_forward"]


def test_converted_portfolio_value_records_and_returns_zero():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.portfolio_value() == 0.0
    assert _sites() == ["portfolio.ContractPortfolio.portfolio_value"]


def test_converted_amend_records_and_returns_none(monkeypatch):
    p = _portfolio()
    old = SimpleNamespace(id="old1", name="Old", contract_type="tenancy",
                          parties="A / B")
    new = SimpleNamespace(id="new1", supersedes=None)
    monkeypatch.setattr(p, "get_contract", lambda cid: old)
    monkeypatch.setattr(p, "add_raw", lambda *a, **k: (new, SimpleNamespace()))
    p._db = _BrokenDb()
    assert p.amend("old1", "x" * 100) is None
    assert _sites() == ["portfolio.ContractPortfolio.amend"]


def test_converted_history_records_and_returns_empty(monkeypatch):
    p = _portfolio()
    monkeypatch.setattr(p, "get_contract",
                        lambda cid: SimpleNamespace(id=["unhashable"],
                                                    supersedes=""))
    assert p.history("c1") == []
    assert _sites() == ["portfolio.ContractPortfolio.history"]


def test_converted_track_sla_records_and_returns_false():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.track_sla("c1", "uptime", "99%") is False
    assert _sites() == ["portfolio.ContractPortfolio.track_sla"]


def test_converted_report_sla_breach_records_and_returns_false():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.report_sla_breach("c1", "uptime") is False
    assert _sites() == ["portfolio.ContractPortfolio.report_sla_breach"]


def test_converted_sla_status_records_and_returns_empty():
    p = _portfolio()
    p._db = _BrokenDb()
    assert p.sla_status() == []
    assert _sites() == ["portfolio.ContractPortfolio.sla_status"]


def test_converted_ensure_schedule_records_and_still_schedules():
    class _Sched:
        def __init__(self):
            self.scheduled = []

        def list_jobs(self):
            raise RuntimeError("list gone")

        async def schedule_cron(self, **kw):
            self.scheduled.append(kw)
            return True

    sched = _Sched()
    assert portfolio_mod.ensure_schedule(sched) is True
    assert len(sched.scheduled) == 1
    assert sched.scheduled[0]["task_id"] == "contracts-portfolio-daily"
    assert _sites() == ["portfolio.ensure_schedule._ensure"]


# ---- Excluded handlers: proven NOT recording ----------------------------

def test_excluded_to_ts_parse_probe_records_nothing():
    assert portfolio_mod._to_ts(99999, 1, 1) is None
    assert portfolio_mod._to_ts(2026, 2, 30) is None
    assert lf.failure_count() == 0


def test_excluded_parse_date_arg_probe_records_nothing():
    assert portfolio_mod._parse_date_arg("no dates here at all") is None
    assert portfolio_mod._parse_date_arg("garbage") is None
    assert lf.failure_count() == 0


def test_excluded_roll_months_probe_records_nothing(monkeypatch):
    calls = []

    class _FakePortfolio:
        def roll_forward(self, cid, months):
            calls.append((cid, months))
            return 0

    monkeypatch.setattr(portfolio_mod, "ContractPortfolio", _FakePortfolio)
    out = portfolio_mod.control_contracts("roll abc123 xyz")
    assert "No open renewal/expiry dates to roll for abc123." in out
    assert calls == [("abc123", 12.0)]
    assert lf.failure_count() == 0
