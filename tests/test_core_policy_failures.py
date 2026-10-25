"""Pass #198 pins: the reused core failure ledger + policy.py conversion.

The core ledger (``nomorals/core/failures.py``, namespace ``core``) was
created in #197 and is REUSED here unchanged — this pass adds no ledger
file. The 17 durability tests below mirror the #141+ shape for this
ledger; the 1 site pin proves the conversion records without changing
flow (fallback behavior unchanged AND a ledger entry under its site
name); the 13 NOT-recording pins prove each exclusion stays silent on
its failure path. No pin hardcodes scan totals or policy.py counts.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Restart tests use real
subprocesses. Every trigger was probed live before its pin was written.
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

from nomorals.core import failures as cf
from nomorals.core import failure_store as fs
from nomorals.core.policy import (
    PermissionGradient,
    Policy,
    approve_with_biometric,
    is_explicit_instruction,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    cf.clear_failures()
    yield
    cf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("core")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("core").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("core")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("core").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("core")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("core").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("core").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("core").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("core")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("core").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("core").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("core").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("core").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("core")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("core").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) -------------------------------------------

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
    assert cf.format_failures() == "no core failures recorded."
    cf.record_failure("core.site-a", ValueError("boom"))
    cf.record_failure("core.site-b", RuntimeError("bang"))
    assert cf.failure_count() == 2
    assert [f.site for f in cf.recent_failures()] == ["core.site-b", "core.site-a"]
    assert [f.site for f in cf.recent_failures(1)] == ["core.site-b"]
    text = cf.format_failures()
    assert "2 total" in text and "core.site-a" in text and "core.site-b" in text
    cf.clear_failures()
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    cf.record_failure("core.disk", ValueError("boom"))
    entries, total = fs.Store("core").load()
    assert total == 1 and entries[0]["site"] == "core.disk"
    cf.clear_failures()
    assert fs.Store("core").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    cf.record_failure("core.mem", ValueError("boom"))
    assert cf.failure_count() == 1
    assert [f.site for f in cf.recent_failures()] == ["core.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                cf.record_failure(f"core.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
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


def test_ledger_ring_cap_drops_oldest_silently():
    for i in range(250):
        cf.record_failure(f"core.ring{i}", ValueError("boom"))
    assert cf.failure_count() == 250
    ring = cf.recent_failures(1000)
    assert len(ring) == 200
    assert ring[0].site == "core.ring249"
    assert ring[-1].site == "core.ring50"


# ---- Restart (subprocess) -------------------------------------------------

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
        "from nomorals.core import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.core import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.core import failures as f\n"
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


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    corrupt = (
        "from pathlib import Path\n"
        "from nomorals.core import failure_store as fs\n"
        "fs.Store('core').path().write_text('{not json', encoding='utf-8')\n"
        "print('corrupted')\n")
    proc = _run_subprocess(home, corrupt)
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0 and data["sites"] == []
    proc = _run_subprocess(home, _record_code(1))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1 and data["sites"] == ["site0"]


# ---- Site pin (convert records, fallback behavior unchanged) --------------

class _FailingSaveStore:
    def save_proposal(self, proposal):
        raise RuntimeError("store down")

    def get_proposal(self, proposal_id):
        return None

    def pending_proposals(self):
        return []

    def update_proposal(self, proposal_id, updates):
        return False


def test_pin_propose_save_records():
    gradient = PermissionGradient(store=_FailingSaveStore())
    proposal = gradient.propose("fs.write", actor="t", draft="d")
    assert proposal["proposal_id"].startswith("prp_")
    assert proposal["status"] == "pending"
    recs = cf.recent_failures()
    assert len(recs) == 1
    assert recs[0].site == "policy.propose_save"
    assert recs[0].exc_type == "RuntimeError"


# ---- NOT-recording pins (exclusions stay silent on their failure path) ---

class _RaisingDb:
    """Fake policy db: execute is a no-op, query behavior set per test."""

    def __init__(self, query):
        self._query = query
        self.calls = 0

    def execute(self, *args, **kwargs):
        return None

    def query(self, *args, **kwargs):
        self.calls += 1
        return self._query(self.calls)


def test_pin_no_record_load_rules_query():
    pol = Policy()
    db = _RaisingDb(lambda n: (_ for _ in ()).throw(RuntimeError("db down")))
    assert pol.load_rules(db) == 0
    assert pol.describe()["rules"] == 0
    assert cf.recent_failures() == []


def test_pin_no_record_load_grants_query():
    pol = Policy()
    pol.grant("fs.read")

    def query(n):
        if n == 1:
            return []
        raise RuntimeError("grants down")

    db = _RaisingDb(query)
    assert pol.load_rules(db) == 0
    assert pol.list_grants() == []
    assert cf.recent_failures() == []


class _BoomCapability:
    @property
    def capability(self):
        raise RuntimeError("rule unreadable")

    effect = "allow"
    note = ""
    priority = 10
    when = None


def test_pin_no_record_requires_biometric():
    pol = Policy()
    pol._rules = [_BoomCapability()]
    assert pol.requires_biometric("fs.write") is True
    assert cf.recent_failures() == []


def test_pin_no_record_progression_membership():
    pol = Policy(progression=lambda: 123)
    assert pol._progression_grants("fs.write") is False
    assert cf.recent_failures() == []


def test_pin_no_record_biometric_available():
    pol = Policy()
    token = approve_with_biometric(
        pol, "fs.delete", biometric=(_boom, lambda *a, **k: True))
    assert token is None
    assert cf.recent_failures() == []


def test_pin_no_record_biometric_prompt():
    pol = Policy()
    token = approve_with_biometric(
        pol, "fs.delete",
        biometric=(lambda: (True, ""), _boom))
    assert token is None
    assert cf.recent_failures() == []


def test_pin_no_record_biometric_mint(monkeypatch):
    pol = Policy()
    monkeypatch.setattr(pol, "issue_confirmation", _boom)
    token = approve_with_biometric(
        pol, "fs.delete",
        biometric=(lambda: (True, ""), lambda *a, **k: True))
    assert token is None
    assert cf.recent_failures() == []


class _BadText:
    def strip(self):
        raise RuntimeError("unreadable text")

    def lower(self):
        return ""


def test_pin_no_record_is_explicit_instruction():
    assert is_explicit_instruction(_BadText()) is False
    assert cf.recent_failures() == []


def test_pin_no_record_level_for():
    gradient = PermissionGradient()
    gradient.policy._rules = [_BoomCapability()]
    assert gradient.level_for("fs.write") is None
    assert cf.recent_failures() == []


class _RaisingStore:
    def save_proposal(self, proposal):
        return None

    def get_proposal(self, proposal_id):
        raise RuntimeError("store down")

    def pending_proposals(self):
        raise RuntimeError("store down")

    def update_proposal(self, proposal_id, updates):
        raise RuntimeError("store down")


def test_pin_no_record_get_proposal():
    gradient = PermissionGradient(store=_RaisingStore())
    assert gradient.get_proposal("prp_missing") is None
    assert cf.recent_failures() == []


def test_pin_no_record_pending_proposals():
    gradient = PermissionGradient(store=_RaisingStore())
    assert gradient.pending_proposals() == []
    assert cf.recent_failures() == []


def test_pin_no_record_approve_proposal():
    gradient = PermissionGradient(store=_RaisingStore())
    assert gradient.approve_proposal("prp_missing") is None
    assert cf.recent_failures() == []


def test_pin_no_record_reject_proposal():
    gradient = PermissionGradient(store=_RaisingStore())
    assert gradient.reject_proposal("prp_missing") is False
    assert cf.recent_failures() == []
