"""Pass #145 pins: the agents/trial failure ledger + durability wiring.

Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through the new ``nomorals.agents.trial.failures`` ledger; restart
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
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents.trial import failures as trial


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    trial.clear_failures()
    yield
    trial.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("agents.trial")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents.trial").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents.trial")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents.trial").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents.trial")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents.trial").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents.trial").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents.trial").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents.trial")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents.trial").load() == ([], 0)
    assert store.save(_entries(1, "after"), 1) is True
    entries, total = fs.Store("agents.trial").load()
    assert [e["site"] for e in entries] == ["after0"]
    assert total == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents.trial").save(_entries(1), 1) is False
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
        assert fs.Store("agents.trial").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        trial.record_failure("no-home", ValueError("boom"))
        assert trial.failure_count() == 1
        assert [f.site for f in trial.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents.trial")
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
    entries, _ = fs.Store("agents.trial").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_api_record_recent_count_clear_format():
    assert trial.failure_count() == 0
    assert trial.recent_failures() == []
    assert trial.format_failures() == "no agents.trial failures recorded."
    trial.record_failure("trial.site-a", ValueError("boom"))
    trial.record_failure("trial.site-b", RuntimeError("bang"))
    assert trial.failure_count() == 2
    assert [f.site for f in trial.recent_failures()] == ["trial.site-b", "trial.site-a"]
    assert [f.site for f in trial.recent_failures(1)] == ["trial.site-b"]
    text = trial.format_failures()
    assert "2 total" in text and "trial.site-a" in text and "trial.site-b" in text
    trial.clear_failures()
    assert trial.failure_count() == 0
    assert trial.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    trial.record_failure("disk-a", ValueError("boom"))
    trial.record_failure("disk-b", RuntimeError("bang"))
    assert trial.failure_count() == 2
    assert trial._store.path().exists()
    assert trial._store.path().name == "agents.trial.json"
    trial.clear_failures()
    assert trial.failure_count() == 0
    assert not trial._store.path().exists()


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    trial.record_failure("blocked", ValueError("boom"))
    assert trial.failure_count() == 1
    assert [f.site for f in trial.recent_failures()] == ["blocked"]
    assert fs.write_failures() > before


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                trial.record_failure(f"w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert trial.failure_count() == 400
    assert len(trial.recent_failures(1000)) == 200
    entries, total = trial._store.load()
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
        "from nomorals.agents.trial import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents.trial import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents.trial import failures as f\n"
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
    store_path = home / "failures" / "agents.trial.json"
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

def _sites():
    return [f.site for f in trial.recent_failures()]


def _bare_flow():
    from nomorals.agents.trial.flow import TrialFlow
    flow = object.__new__(TrialFlow)
    flow.db = object()
    return flow


class _RaisingKVStore:
    def __init__(self, db):
        pass

    def get(self, *a, **k):
        raise RuntimeError("kv down")

    def set(self, *a, **k):
        raise RuntimeError("kv down")


def test_converted_load_plan_records_and_returns_empty(monkeypatch):
    from nomorals.agents.trial import flow as flow_mod
    monkeypatch.setattr(flow_mod, "KVStore", _RaisingKVStore)
    assert _bare_flow()._load_plan("telegram") == {}
    assert "flow._load_plan" in _sites()


def test_converted_stash_plan_records_and_never_raises(monkeypatch):
    from nomorals.agents.trial import flow as flow_mod
    monkeypatch.setattr(flow_mod, "KVStore", _RaisingKVStore)
    _bare_flow()._stash_plan("telegram", "plan text")
    assert "flow._stash_plan" in _sites()


def test_converted_load_temp_number_records_and_returns_empty(monkeypatch):
    from nomorals.agents.trial import flow as flow_mod
    monkeypatch.setattr(flow_mod, "KVStore", _RaisingKVStore)
    assert _bare_flow()._load_temp_number() == {}
    assert "flow._load_temp_number" in _sites()


def test_converted_send_records_and_returns_false():
    from nomorals.agents.trial.flow import TrialFlow

    class _RaisingGateway:
        def send(self, *a, **k):
            raise RuntimeError("gateway down")

    flow = object.__new__(TrialFlow)
    flow._sender = None
    assert flow._send("telegram", "telegram:123", "hi",
                      gateway=_RaisingGateway()) is False
    assert "flow._send" in _sites()


def test_converted_existing_account_trial_vault_records_and_unknown():
    from nomorals.agents.trial.flow import TrialFlow

    class _RaisingVault:
        def get(self, *a, **k):
            raise RuntimeError("vault down")

    flow = object.__new__(TrialFlow)
    flow.vault = _RaisingVault()
    flow.db = None
    assert flow._existing_account("telegram") == ("unknown", None)
    assert "flow._existing_account.trial_vault" in _sites()


def test_converted_audit_write_records_and_never_raises(tmp_path):
    from nomorals.agents.trial.flow import TrialFlow
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    flow = object.__new__(TrialFlow)
    flow.vault = SimpleNamespace(home=blocker / "sub")
    flow._audit("save", "telegram", "stored")
    assert "flow._audit" in _sites()


def test_converted_audit_log_read_records_and_reports(tmp_path):
    from nomorals.agents.trial.flow import TrialFlow
    home = tmp_path / "home"
    (home / "trial_audit.jsonl").parent.mkdir(parents=True)
    (home / "trial_audit.jsonl").mkdir()  # a dir, not a file: read fails
    flow = object.__new__(TrialFlow)
    flow.vault = SimpleNamespace(home=home)
    assert flow.audit_log() == "could not read the trial audit log."
    assert "flow.audit_log.read" in _sites()


def test_converted_audit_log_line_records_and_skips(tmp_path):
    from nomorals.agents.trial.flow import TrialFlow
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "trial_audit.jsonl").write_text(
        '{"ts": 1, "action": "save", "platform": "telegram"}\n'
        "{corrupt line}\n"
        '{"ts": 2, "action": "list", "platform": "telegram"}\n',
        encoding="utf-8")
    flow = object.__new__(TrialFlow)
    flow.vault = SimpleNamespace(home=home)
    text = flow.audit_log()
    assert "save" in text and "list" in text
    assert "could not read" not in text
    assert "flow.audit_log.line" in _sites()


def test_converted_vault_read_all_records_and_starts_fresh(tmp_path):
    from nomorals.agents.trial.vault import TrialVault
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "trial_accounts.json").write_text("{not json", encoding="utf-8")
    vault = TrialVault(home)
    assert vault._read_all() == {}
    assert (home / "trial_accounts.corrupt").exists()
    assert "vault._read_all" in _sites()


def test_converted_vault_audit_records_and_never_raises(tmp_path):
    from nomorals.agents.trial.vault import TrialVault
    home = tmp_path / "home"
    home.mkdir(parents=True)
    vault = TrialVault(home)
    # Replace the home dir with a file: the audit append can no longer
    # open its path, and must degrade to a ledger entry, never raise.
    import shutil
    shutil.rmtree(home)
    home.write_text("a file, not a dir", encoding="utf-8")
    vault._audit("get", "telegram", ok=False)
    assert "vault._audit" in _sites()


def test_converted_vault_get_unreadable_records_and_marks(tmp_path):
    from nomorals.agents.trial.vault import TrialVault
    home = tmp_path / "home"
    vault = TrialVault(home)
    vault.store("telegram", "login1", "s3cret-pw", "note")
    data = json.loads((home / "trial_accounts.json").read_text())
    data["telegram"]["secret"] = "zz"  # invalid hex: decryption must fail
    (home / "trial_accounts.json").write_text(json.dumps(data))
    entry = TrialVault(home).get("telegram")
    assert entry is not None and entry.get("unreadable") is True
    assert entry.get("secret") is None
    assert "vault.get" in _sites()


def test_converted_vault_audit_trail_line_records_and_skips(tmp_path):
    from nomorals.agents.trial.vault import TrialVault
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "trial_access.jsonl").write_text(
        '{"ts": 1, "action": "get", "platform": "telegram"}\n'
        "{corrupt line}\n"
        '{"ts": 2, "action": "list", "platform": ""}\n',
        encoding="utf-8")
    rows = TrialVault(home).audit_trail()
    assert [r["action"] for r in rows] == ["list", "get"]
    assert "vault.audit_trail.line" in _sites()
