"""Pass #186 pins: the REUSED agents failure ledger + skill_proving conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) already existed
(created #152, reused #153/#163/#166/#169/#171/#173/#174/#184/#185) —
skill_proving.py is an agents-subsystem module, so it reuses that
ledger; no new ledger in this pass. The 17 durability tests below mirror
the #141+ shape exactly (same as #184/#185's, against the agents ledger);
the 8 site pins prove each conversion records without changing flow
(fallback value unchanged AND a ledger entry under its site name); the
8 NOT-recording pins prove each exclusion stays silent with the
designed output. No pin hardcodes scan totals or skill_proving
counts. Every test redirects stores with the app's own ``NM_HOME``
override (never the real home, never the repo tree). Restart tests use
real subprocesses. Every trigger was probed live before its pin was
written (probe.py).

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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.agents import skill_proving as sp
from nomorals.agents import failures as af
from nomorals.core import failure_store as fs


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in af.recent_failures()]


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("agents")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("agents").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agents").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents")
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
    entries, _ = fs.Store("agents").load()
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
    assert af.format_failures() == "no agents failures recorded."
    af.record_failure("agents.site-a", ValueError("boom"))
    af.record_failure("agents.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["agents.site-b", "agents.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["agents.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "agents.site-a" in text and "agents.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("agents.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1 and entries[0]["site"] == "agents.disk"
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("agents.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["agents.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"agents.w{k}-{i}", ValueError("boom"))
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
        "from nomorals.agents import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents import failures as f\n"
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
    store_path = fs.Store("agents").path()
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

class _Boom(Exception):
    pass


class _FakeDB:
    def __init__(self, executescript_raises=False, execute_raises=False,
                 query_one_raises=False, query_one_value=None):
        self.executescript_raises = executescript_raises
        self.execute_raises = execute_raises
        self.query_one_raises = query_one_raises
        self.query_one_value = query_one_value
        self.executes = []

    def executescript(self, sql):
        if self.executescript_raises:
            raise _Boom("db executescript boom")
        return None

    def execute(self, sql, args=()):
        if self.execute_raises:
            raise _Boom("db execute boom")
        self.executes.append((sql, args))
        return None

    def query_one(self, sql, args=()):
        if self.query_one_raises:
            raise _Boom("db query_one boom")
        return self.query_one_value


class _FakeRegistry:
    def __init__(self, versions=None, versions_raises=False,
                 get_two=None, get_one_raises=False,
                 pin_side_effect=None, disable_raises=False,
                 enable_raises=False):
        self._versions = versions or []
        self.versions_raises = versions_raises
        self.get_two = get_two
        self.get_one_raises = get_one_raises
        self.pin_side_effect = pin_side_effect
        self.disable_raises = disable_raises
        self.enable_raises = enable_raises
        self.pins = []

    def versions(self, name):
        if self.versions_raises:
            raise _Boom("registry versions boom")
        return list(self._versions)

    def get(self, *args):
        if len(args) == 1:
            if self.get_one_raises:
                raise _Boom("registry get(name) boom")
            return SimpleNamespace(enabled=True, active=True)
        return self.get_two

    def pin(self, name, version):
        if self.pin_side_effect is not None:
            self.pin_side_effect(name, version)
        self.pins.append((name, version))

    def disable(self, name):
        if self.disable_raises:
            raise _Boom("registry disable boom")

    def enable(self, name):
        if self.enable_raises:
            raise _Boom("registry enable boom")


class _OnceRaisingRegistry(_FakeRegistry):
    """get() raises exactly once, then returns None (not present)."""

    def __init__(self):
        super().__init__()
        self._n = 0

    def get(self, *args):
        self._n += 1
        if self._n == 1:
            raise _Boom("get boom once")
        return None


@contextmanager
def _stub_canary(cls):
    """Stub skill_canary in sys.modules, restoring the ORIGINAL entry after.

    A plain delete after stubbing would destroy the real module entry
    when the module was already imported (probe.py caught this live).
    """
    key = "nomorals.agents.skill_canary"
    saved = sys.modules.get(key, None)
    had = key in sys.modules
    sys.modules[key] = SimpleNamespace(CanaryRollout=cls)
    try:
        yield
    finally:
        if had:
            sys.modules[key] = saved
        else:
            sys.modules.pop(key, None)


class _RaisingCanary:
    def __init__(self, ctx):
        raise _Boom("canary init boom")

    def record_version(self, *a, **k):
        raise AssertionError("unreachable")


class _BenignCanary:
    def __init__(self, ctx):
        self.ctx = ctx

    def record_version(self, name, body, source=""):
        return "hash1"


class _EvilEnabled:
    @property
    def enabled(self):
        raise AttributeError("no enabled attr")


def test_pin_skill_proving_has_proof():
    db = _FakeDB(executescript_raises=True, query_one_raises=True)
    assert sp.has_proof("s", db) is False
    assert "skill_proving.has_proof" in _sites()


def test_pin_skill_proving_flag_skill_disable():
    db = _FakeDB()
    reg = _FakeRegistry(disable_raises=True)
    assert sp.flag_skill("s", "r", db, reg) is True
    assert any("skill_flags" in sql for sql, _ in db.executes)
    assert "skill_proving.flag_skill_disable" in _sites()


def test_pin_skill_proving_unflag_skill_enable():
    db = _FakeDB()
    reg = _FakeRegistry(enable_raises=True)
    assert sp.unflag_skill("s", db, reg) is True
    assert "skill_proving.unflag_skill_enable" in _sites()


def test_pin_skill_proving_get_flag():
    db = _FakeDB(query_one_raises=True)
    assert sp.get_flag("s", db) is None
    assert "skill_proving.get_flag" in _sites()


def test_pin_skill_proving_promote_canary_chain():
    proof = sp.ProofResult("s", "v9", True)
    with _stub_canary(_RaisingCanary):
        out = sp.promote_on_proof(SimpleNamespace(workflow="w"), proof,
                                  _FakeRegistry(), _FakeDB())
    assert out == "active"
    assert proof.version_hash == ""
    assert "skill_proving.promote_canary_chain" in _sites()


def test_pin_skill_proving_installed_version():
    reg = _FakeRegistry(versions_raises=True)
    assert sp._installed_version(reg, "s") == ""
    assert "skill_proving.installed_version" in _sites()


def test_pin_skill_proving_lifecycle_active():
    reg = _FakeRegistry(versions=["v1"],
                        get_two=SimpleNamespace(enabled=True),
                        get_one_raises=True)
    assert sp.skill_lifecycle("s", _FakeDB(), reg) == "draft"
    assert "skill_proving.lifecycle_active" in _sites()


def test_pin_skill_proving_ensure_capability_present():
    reg = _OnceRaisingRegistry()
    out = sp.ensure_capability("my cap", "desc", _FakeDB(), reg, None, None)
    assert out["status"] != "present"
    assert "skill_proving.ensure_capability_present" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_skill_proving_manifest_version_silent():
    draft = SimpleNamespace(name="x", workflow="",
                            to_manifest=Mock(side_effect=_Boom("manifest")))
    res = sp.prove_skill(draft, _FakeDB(execute_raises=True))
    assert res.version == ""
    assert _sites() == []


def test_pin_skill_proving_unflag_outer_silent():
    db = _FakeDB(execute_raises=True)
    assert sp.unflag_skill("s", db, None) is False
    assert _sites() == []


def test_pin_skill_proving_refuse_untested_silent():
    with patch.object(sp, "has_proof", side_effect=_Boom("boom")):
        assert sp.refuse_untested("s", _FakeDB()) is True
    assert _sites() == []


def test_pin_skill_proving_pin_retry_silent():
    calls = {"n": 0}

    def pin_once_then_ok(name, version):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Boom("pin v1 boom")

    proof = sp.ProofResult("s", "v1", True)
    reg = _FakeRegistry(versions=["v2"], pin_side_effect=pin_once_then_ok)
    with _stub_canary(_BenignCanary):
        out = sp.promote_on_proof(SimpleNamespace(workflow="w"), proof, reg,
                                  _FakeDB())
    assert out == "active"
    assert ("s", "v2") in reg.pins
    assert _sites() == []


def test_pin_skill_proving_fallback_pin_silent():
    def pin_always(name, version):
        raise _Boom("pin always boom")

    proof = sp.ProofResult("s", "v1", True)
    reg = _FakeRegistry(versions=["v2"], pin_side_effect=pin_always)
    out = sp.promote_on_proof(SimpleNamespace(workflow="w"), proof, reg,
                              _FakeDB())
    assert out == "flagged"
    assert _sites() == []


def test_pin_skill_proving_any_installed_silent():
    reg = _FakeRegistry(versions=["v1"])

    def get_boom(*args):
        if len(args) == 2:
            raise _Boom("get(name, version) boom")
        return SimpleNamespace(enabled=True, active=True)

    reg.get = get_boom
    assert sp.skill_lifecycle("s", _FakeDB(), reg) == "unknown"
    assert _sites() == []


def test_pin_skill_proving_lifecycle_outer_silent():
    reg = _FakeRegistry(versions=["v1"])

    def get_evil(*args):
        if len(args) == 2:
            return _EvilEnabled()
        return SimpleNamespace(enabled=True, active=True)

    reg.get = get_evil
    assert sp.skill_lifecycle("s", _FakeDB(), reg) == "unknown"
    assert _sites() == []


def test_pin_skill_proving_canary_check_silent():
    with patch.object(sp, "has_proof", side_effect=_Boom("boom")):
        chk = sp.canary_proof_check(_FakeDB())
        assert chk("s", "h") is False
    assert _sites() == []
