"""Pass #165 pins: the llm failure ledger (NEW) + router conversions.

``nomorals/llm/swallow_failures.py`` (namespace ``llm``) is NEW in this
pass: ``nomorals/llm/failures.py`` already exists as the provider-failure
taxonomy (``FailureClass``/``classify_failure``) and must NOT be reused
for census recording — see SURVEY.md. The 17 durability tests mirror
the #141+ shape exactly; the 20 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger
entry under its site name); the 2 exclusion pins prove the duck-typed
annotation probe and the dead tagging handler record nothing. No pin
hardcodes scan totals or router counts. Every test redirects stores
with the app's own ``NM_HOME`` override (never the real home, never
the repo tree). Restart tests use real subprocesses. Every trigger was
probed live before its pin was written.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.llm import swallow_failures as lf
from nomorals.llm.base import LLMResponse
from nomorals.llm.router import (
    CostLog,
    LLMRouter,
    ProviderHealth,
    _FLAT_LLM_CALL_USD,
    estimate_cost,
    log_llm_call,
)


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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
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


# ---- Site pins ------------------------------------------------------------

class _RaisingBus:
    def emit(self, *a, **k):
        raise RuntimeError("bus down")


def _raising_hook(*a, **k):
    raise RuntimeError("hook down")


def _write_cost_row(path: Path, **over):
    row = {"ts": time.time(), "provider": "a", "operation": "chat",
           "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.001,
           "latency_ms": 10.0, "cached_tokens": 0, "reasoning_tokens": 0}
    row.update(over)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_pin_estimate_cost():
    # "groq" is in _TOKEN_PRICES, so the math runs and the bad token
    # count raises; an unknown provider would return the flat rate
    # early WITHOUT raising, so it cannot trigger this handler.
    assert estimate_cost("groq", "y", "bad", 0) == _FLAT_LLM_CALL_USD  # type: ignore[arg-type]
    assert _sites() == ["router.estimate_cost"]


def test_pin_entries_missing(tmp_path):
    assert CostLog(tmp_path / "missing.jsonl").entries() == []
    assert _sites() == ["router.entries_missing"]


def test_pin_total_spend(tmp_path):
    # Valid ts but junk cost_usd: entries() returns the row (ts parses),
    # total_spend's sum() raises on the junk value -> 0.0 + record.
    p = tmp_path / "cost.jsonl"
    _write_cost_row(p, cost_usd="junk")
    assert CostLog(p).total_spend() == 0.0
    assert _sites() == ["router.total_spend"]


def test_pin_breakdown_total_cast(tmp_path):
    # cached_tokens is read ONLY by the total loop, so only it fires.
    p = tmp_path / "cost.jsonl"
    _write_cost_row(p, cached_tokens="junk")
    bd = CostLog(p).breakdown()
    assert bd["total"]["cost_usd"] == 0.001
    assert _sites() == ["router.breakdown_total_cast"]


def test_pin_breakdown_slot_cast(tmp_path):
    # cost_usd is read by both loops; the slot site must appear (the
    # total site firing too is covered by its own pin).
    p = tmp_path / "cost.jsonl"
    _write_cost_row(p, cost_usd="junk")
    CostLog(p).breakdown()
    assert "router.breakdown_slot_cast" in _sites()


def test_pin_budget_report(tmp_path):
    # Pre-created empty log so total_spend reads [] without recording;
    # only the bad-budget float() raises.
    p = tmp_path / "cost.jsonl"
    p.write_text("", encoding="utf-8")
    assert CostLog(p).budget_report("junk") == {  # type: ignore[arg-type]
        "spent_usd": 0.0, "budget_usd": 0.0, "alert": "ok", "pct": 0.0}
    assert _sites() == ["router.budget_report"]


def test_pin_p95_latency():
    h = ProviderHealth(name="h")
    h.latencies.append("x")  # type: ignore[arg-type]
    assert h.p95_latency_ms == 0.0
    assert _sites() == ["router.p95_latency"]


def test_pin_adaptive_block_emit():
    r = LLMRouter(bus=_RaisingBus())
    verdict = SimpleNamespace(deferred=False, blocked=True, action="block",
                              note="over", utilisation=0.9)
    resp = r._adaptive_block_response(verdict, "chat")
    assert resp.error == "over"
    assert _sites() == ["router.adaptive_block_emit"]


def test_pin_adaptive_apply_emit():
    r = LLMRouter(bus=_RaisingBus())
    r._adaptive_apply(SimpleNamespace(action="truncate", note="n",
                                      utilisation=0.5), "chat")
    assert r.stats["budget_adaptations"] == 1
    assert _sites() == ["router.adaptive_apply_emit"]


def test_pin_route_price():
    r = LLMRouter()

    class _RaisingBroker:
        def cards(self):
            raise RuntimeError("broker down")

    r._broker = _RaisingBroker()
    assert r._route_price("any") == 0.0
    assert _sites() == ["router.route_price"]


def test_pin_max_price_cap():
    r = LLMRouter()
    out = r._apply_route([SimpleNamespace(name="a"), SimpleNamespace(name="b")],
                         {"max_price": "junk"})
    assert [p.name for p in out] == ["a", "b"]
    assert _sites() == ["router.max_price_cap"]


def test_pin_budget_warning_emit():
    r = LLMRouter(bus=_RaisingBus(), daily_budget_usd=10.0)
    r._budget_checked_at = time.time()
    r._budget_cached_spent = 9.0
    assert r._budget_block("chat") is None
    assert _sites() == ["router.budget_warning_emit"]


def _failing_router(bus):
    r = LLMRouter(bus=bus)
    stub = SimpleNamespace(name="stub", model_id="m", capabilities={"chat"})
    r._providers = [stub]
    r._by_name = {"stub": stub}
    r._health = {"stub": ProviderHealth(name="stub")}
    return r


def test_pin_brain_down_emit():
    r = _failing_router(_RaisingBus())

    def boom(p):
        raise RuntimeError("provider boom")

    last = r._dispatch("chat", boom)
    assert last.error != "" and "boom" in last.error
    assert "router.brain_down_emit" in _sites()


def test_pin_note_success_heartbeat(monkeypatch):
    import nomorals.core.error_system as es
    monkeypatch.setattr(es, "heartbeat", _boom)
    r = LLMRouter()
    r._health = {"s": ProviderHealth(name="s")}
    r._note_success("s", 5.0)
    assert _sites() == ["router.note_success_heartbeat"]


def test_pin_note_failure_heartbeat(monkeypatch):
    import nomorals.core.error_system as es
    monkeypatch.setattr(es, "heartbeat", _boom)
    r = LLMRouter()
    r._health = {"s": ProviderHealth(name="s")}
    r._note_failure("s", "err")
    assert _sites() == ["router.note_failure_heartbeat"]


def test_pin_repair_hook_dict(monkeypatch):
    import nomorals.core.error_system as es
    monkeypatch.setattr(es, "heartbeat", lambda *a, **k: None)
    r = LLMRouter(repair_hooks={"h": _raising_hook})
    r._health = {"s": ProviderHealth(name="s")}
    r._note_failure("s", "err")
    assert r.stats["repairs"] == 0
    assert _sites() == ["router.repair_hook_dict"]


def test_pin_repair_hook_list(monkeypatch):
    import nomorals.core.error_system as es
    monkeypatch.setattr(es, "heartbeat", lambda *a, **k: None)
    r = LLMRouter(repair_hooks=[_raising_hook])
    r._health = {"s": ProviderHealth(name="s")}
    r._note_failure("s", "err")
    assert r.stats["repairs"] == 0
    assert _sites() == ["router.repair_hook_list"]


def test_pin_probe_health():
    bad = SimpleNamespace(name="bad", model_id="m",
                          health=lambda: (_ for _ in ()).throw(
                              RuntimeError("health boom")))
    r = LLMRouter()
    r._providers = [bad]
    r._by_name = {"bad": bad}
    assert r.probe() == {"bad": False}
    assert _sites() == ["router.probe_health"]


def test_pin_snapshot_budget():
    r = LLMRouter(daily_budget_usd=5.0)
    r._cost_log = lambda: SimpleNamespace(
        budget_report=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("log boom")))
    snap = r.stats_snapshot()
    assert "budget" not in snap
    assert _sites() == ["router.snapshot_budget"]


def test_pin_snapshot_adaptive():
    r = LLMRouter()
    r.adaptive_budget = SimpleNamespace(
        status=lambda: (_ for _ in ()).throw(RuntimeError("status boom")))
    snap = r.stats_snapshot()
    assert "adaptive_budget" not in snap
    assert _sites() == ["router.snapshot_adaptive"]


# ---- Exclusion pins -------------------------------------------------------

def test_excl_duck_annotation(tmp_path):
    # Frozen response rejects the cost_usd annotation; the cost is still
    # returned AND recorded in the shared log, and nothing is recorded
    # for the (excluded) annotation site.
    class _FrozenResp:
        __slots__ = ("usage",)

        def __init__(self):
            self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                         cached_tokens=0, reasoning_tokens=0)

    fr = _FrozenResp()
    lp = tmp_path / "log.jsonl"
    cost = log_llm_call("prov", "model", fr, path=lp)
    assert cost > 0 and not hasattr(fr, "cost_usd") and lp.exists()
    assert _sites() == []


def test_excl_tagging_dead_handler(monkeypatch):
    # Real dispatch tail: tagging runs (failure_class set) and the bus
    # is None so no emit site fires; the dead tagging handler records
    # nothing. (Heartbeat is stubbed to a no-op — it has its own pins.)
    import nomorals.core.error_system as es
    monkeypatch.setattr(es, "heartbeat", lambda *a, **k: None)
    r = _failing_router(None)

    def boom(p):
        raise RuntimeError("provider boom")

    last = r._dispatch("chat", boom)
    assert last.failure_class != ""
    assert _sites() == []
