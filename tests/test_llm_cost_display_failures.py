"""Pass #191 pins: the REUSED llm failure ledger + cost_display conversions.

``nomorals/llm/swallow_failures.py`` (namespace ``llm``, from #165)
is reused — cost_display.py is an llm-subsystem module, so no new
ledger file. The 17 durability tests below mirror the #141+ shape
exactly (same as #181's, against the same Store backend); the 2 site
pins prove each conversion records without changing flow (fallback
value unchanged AND a ledger entry under its site name); the 13
NOT-recording pins prove each exclusion stays silent with the
designed output. No pin hardcodes scan totals or cost_display counts.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Restart tests use real
subprocesses. Every trigger was probed live before its pin was
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
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.llm import swallow_failures as lf
from nomorals.llm import cost_display as cd


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


class StubDisplay:
    """Minimal CostDisplay stand-in for control_cost/maybe_cost_footer."""

    def __init__(self, enabled=False, spend=0.0):
        self._enabled = enabled
        self._spend = spend
        self.toggles = []

    def is_enabled(self, user_id="owner"):
        return self._enabled

    def set_enabled(self, on, user_id="owner"):
        self.toggles.append((on, user_id))
        return True

    def today_spend(self, cost_path=None):
        return self._spend

    def footer_for_turn(self, since_ts, user_id="owner", cost_path=None):
        if not self._enabled:
            return ""
        return f"\n\n_this answer cost {cd.format_cost(self._spend)}_"


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("llm")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("llm").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("llm")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("llm").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("llm")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("llm").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("llm").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("llm").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("llm")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("llm").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("llm").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("llm").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("llm").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("llm")
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
    entries, _ = fs.Store("llm").load()
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
    assert lf.format_failures() == "no llm failures recorded."
    lf.record_failure("llm.site-a", ValueError("boom"))
    lf.record_failure("llm.site-b", RuntimeError("bang"))
    assert lf.failure_count() == 2
    assert [f.site for f in lf.recent_failures()] == ["llm.site-b", "llm.site-a"]
    assert [f.site for f in lf.recent_failures(1)] == ["llm.site-b"]
    text = lf.format_failures()
    assert "2 total" in text and "llm.site-a" in text and "llm.site-b" in text
    lf.clear_failures()
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    lf.record_failure("llm.disk", ValueError("boom"))
    entries, total = fs.Store("llm").load()
    assert total == 1 and entries[0]["site"] == "llm.disk"
    lf.clear_failures()
    assert fs.Store("llm").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    lf.record_failure("llm.mem", ValueError("boom"))
    assert lf.failure_count() == 1
    assert [f.site for f in lf.recent_failures()] == ["llm.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                lf.record_failure(f"llm.w{k}-{i}", ValueError("boom"))
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
        "from nomorals.llm import swallow_failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.llm import swallow_failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.llm import swallow_failures as f\n"
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
    store_path = fs.Store("llm").path()
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


# ---- Site pins (converts) -------------------------------------------------

def test_pin_cost_display_set_enabled():
    disp = cd.CostDisplay.__new__(cd.CostDisplay)
    db = Mock()
    db.execute.side_effect = RuntimeError("disk down")
    disp._db = db
    assert disp.set_enabled(True) is False
    assert _sites() == ["cost_display.set_enabled"]


def test_pin_cost_display_budget_set():
    ctx = Mock()
    ctx.set_research_budget.side_effect = RuntimeError("ctx down")
    ctx.user_id = "owner"
    out = cd.control_cost("budget spend $0.50", context=ctx,
                          display=StubDisplay())
    assert "research budget set" in out
    assert _sites() == ["cost_display.budget_set"]


# ---- NOT-recording pins (exclusions) --------------------------------------

def test_pin_format_cost_not_recorded():
    assert cd.format_cost({"not": "a number"}) == "free"
    assert _sites() == []


def test_pin_parse_budget_nl_not_recorded():
    assert cd.parse_budget_nl(12345) is None
    assert _sites() == []


def test_pin_format_cost_table_not_recorded():
    bad = {"total": {}, "by_operation": {"op": {"cost_usd": {"x": 1}}},
           "by_provider": {}}
    assert cd.format_cost_table(bad) == "💰 spend unavailable"
    assert _sites() == []


def test_pin_budget_alert_line_not_recorded():
    assert cd.budget_alert_line({"x": 1}, 1.0) == "💰 budget status unavailable"
    assert _sites() == []


def test_pin_default_db_not_recorded():
    with patch.object(Path, "mkdir", side_effect=OSError("denied")):
        p = cd._default_db()
    assert p.endswith("cost_display.db")
    assert _sites() == []


def test_pin_is_enabled_not_recorded():
    disp = cd.CostDisplay.__new__(cd.CostDisplay)
    db = Mock()
    db.execute.side_effect = RuntimeError("db down")
    disp._db = db
    assert disp.is_enabled() is False
    assert _sites() == []


def test_pin_turn_cost_not_recorded():
    key = "nomorals.llm.router"
    had = key in sys.modules
    old = sys.modules.get(key)
    sys.modules[key] = None
    try:
        disp = cd.CostDisplay.__new__(cd.CostDisplay)
        disp._db = None
        assert disp.turn_cost(0.0) == 0.0
    finally:
        if had:
            sys.modules[key] = old
        else:
            del sys.modules[key]
    assert _sites() == []


def test_pin_today_spend_not_recorded():
    disp = cd.CostDisplay.__new__(cd.CostDisplay)
    disp._db = None
    with patch.object(cd.CostDisplay, "turn_cost",
                      side_effect=RuntimeError("boom")):
        assert disp.today_spend() == 0.0
    assert _sites() == []


def test_pin_footer_for_turn_not_recorded():
    disp = cd.CostDisplay.__new__(cd.CostDisplay)
    disp._db = None
    with patch.object(cd.CostDisplay, "is_enabled",
                      side_effect=RuntimeError("boom")):
        assert disp.footer_for_turn(0.0) == ""
    assert _sites() == []


def test_pin_maybe_cost_footer_not_recorded():
    bad = Mock()
    bad.footer_for_turn.side_effect = RuntimeError("boom")
    assert cd.maybe_cost_footer("hello", display=bad) == "hello"
    assert _sites() == []


class _RaisingUserId:
    @property
    def user_id(self):
        raise RuntimeError("prop down")


def test_pin_user_id_inner_not_recorded():
    out = cd.control_cost("", context=_RaisingUserId(),
                          display=StubDisplay())
    assert "cost display: off" in out
    assert _sites() == []


def test_pin_breakdown_inner_not_recorded():
    key = "nomorals.llm.router"
    had = key in sys.modules
    old = sys.modules.get(key)
    sys.modules[key] = None
    try:
        out = cd.control_cost("breakdown", display=StubDisplay())
    finally:
        if had:
            sys.modules[key] = old
        else:
            del sys.modules[key]
    assert out == "💰 breakdown unavailable."
    assert _sites() == []


def test_pin_outer_not_recorded():
    bad = Mock()
    bad.today_spend.side_effect = RuntimeError("boom")
    bad.is_enabled.return_value = False
    out = cd.control_cost("", display=bad)
    assert out.startswith("💰 /cost")
    assert _sites() == []
