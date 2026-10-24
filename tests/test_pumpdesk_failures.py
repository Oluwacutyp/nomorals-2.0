"""Pass #148 pins: the NEW agents/pumpdesk failure ledger + conversions.

The pumpdesk ledger did not exist before this pass; it is created
here, durability-wired through the shared failure_store helper (the
#141 shape: Store.save() resolves its path INSIDE the try).
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through ``nomorals.agents.pumpdesk.failures``; restart tests use
real subprocesses (a fresh interpreter is the honest simulated
restart — no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name. The excluded handlers (parse
probes, teardown, surfaced-on-record mirrors, same-value defaults,
advisory sort key, named dossier gap) are pinned as NOT recording.
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
from nomorals.agents.pumpdesk import failures as pd
from nomorals.agents.pumpdesk import attention as attention_mod
from nomorals.agents.pumpdesk import calls as calls_mod
from nomorals.agents.pumpdesk import chain as chain_mod
from nomorals.agents.pumpdesk import desk as desk_mod
from nomorals.agents.pumpdesk import dossier as dossier_mod
from nomorals.agents.pumpdesk import shelf as shelf_mod
from nomorals.agents.pumpdesk import sources as sources_mod
from nomorals.agents.pumpdesk import store as store_mod
from nomorals.agents.pumpdesk import wsclient as ws_mod


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    pd.clear_failures()
    yield
    pd.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in pd.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


class _StubStore:
    """PumpStore double: attribute-driven, no disk, no network."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def watched_devs(self):
        return self.__dict__.get("watched_devs", set())

    def dossiers(self):
        return self.__dict__.get("dossiers_val", [])

    def launch(self, mint):
        fn = self.__dict__.get("launch_fn")
        if fn is not None:
            return fn(mint)
        return self.__dict__.get("launch_val", {})

    def launches(self):
        return self.__dict__.get("launches_val", [])

    def health_events(self):
        fn = self.__dict__.get("health_events_fn")
        if fn is not None:
            return fn()
        return []

    def record_launch(self, payload):
        fn = self.__dict__.get("record_launch_fn")
        if fn is not None:
            return fn(payload)
        return True

    def update_launch(self, mint, **fields):
        return None

    def trades(self, mint):
        return self.__dict__.get("trades_val", [])

    def dev_profiles(self):
        return {}

    def graph(self):
        return self.__dict__.get("graph_val", {})

    def watched_smart(self):
        return set()

    def record_health_event(self, event):
        return None

    def save_dossier(self, dossier):
        return None

    def observations(self, mint, since=None, until=None):
        return []

    def record_observation(self, *args, **kwargs):
        return None


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("agents.pumpdesk")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents.pumpdesk").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents.pumpdesk")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents.pumpdesk").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents.pumpdesk")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents.pumpdesk").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents.pumpdesk").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents.pumpdesk").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents.pumpdesk")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents.pumpdesk").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("agents.pumpdesk").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents.pumpdesk").save(_entries(1), 1) is False
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
        assert fs.Store("agents.pumpdesk").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        pd.record_failure("no-home", ValueError("boom"))
        assert pd.failure_count() == 1
        assert [f.site for f in pd.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents.pumpdesk")
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
    entries, _ = fs.Store("agents.pumpdesk").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_api_record_recent_count_clear_format():
    assert pd.failure_count() == 0
    assert pd.recent_failures() == []
    assert pd.format_failures() == "no agents.pumpdesk failures recorded."
    pd.record_failure("pd.site-a", ValueError("boom"))
    pd.record_failure("pd.site-b", RuntimeError("bang"))
    assert pd.failure_count() == 2
    assert [f.site for f in pd.recent_failures()] == ["pd.site-b", "pd.site-a"]
    assert [f.site for f in pd.recent_failures(1)] == ["pd.site-b"]
    text = pd.format_failures()
    assert "2 total" in text and "pd.site-a" in text and "pd.site-b" in text
    pd.clear_failures()
    assert pd.failure_count() == 0
    assert pd.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    pd.record_failure("pd.disk", ValueError("boom"))
    entries, total = fs.Store("agents.pumpdesk").load()
    assert total == 1 and entries[0]["site"] == "pd.disk"
    pd.clear_failures()
    assert fs.Store("agents.pumpdesk").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    pd.record_failure("pd.mem", ValueError("boom"))
    assert pd.failure_count() == 1
    assert [f.site for f in pd.recent_failures()] == ["pd.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                pd.record_failure(f"pd.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert pd.failure_count() == 400
    assert len(pd.recent_failures(1000)) == 200
    entries, total = pd._store.load()
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
        "from nomorals.agents.pumpdesk import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents.pumpdesk import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents.pumpdesk import failures as f\n"
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
    store_path = fs.Store("agents.pumpdesk").path()
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

def test_converted_search_fn_records_and_returns_empty(monkeypatch):
    # Force the federated import itself to fail: the leg must yield
    # [] (its old value) AND record the failure.
    monkeypatch.setitem(sys.modules, "nomorals.search.federated", None)
    search = attention_mod.default_search_fn(context=None)
    assert search("query") == []
    assert _sites() == ["attention.default_search_fn._search"]


def test_converted_platform_fn_records_and_returns_none():
    platform = attention_mod.default_platform_fn(http_get=_boom)
    assert platform("mint") is None
    assert _sites() == ["attention.default_platform_fn._platform"]


def test_converted_dex_fn_records_and_returns_none(monkeypatch):
    monkeypatch.setattr(
        "nomorals.agents.pumpdesk.outcomes.token_attention", _boom)
    dex = attention_mod.default_dex_fn()
    assert dex("mint") is None
    assert _sites() == ["attention.default_dex_fn._dex"]


def test_converted_verify_launch_records_and_continues():
    store = _StubStore(launch_fn=_boom, launches_val=[])
    out = attention_mod.verify_attention(
        {"mint": "m", "symbol": "S", "trades": {}}, store=store)
    assert isinstance(out, dict)
    assert _sites() == ["attention.verify_attention.launch"]


def test_converted_grade_records_and_scores_miss_no_data(tmp_path):
    engine = calls_mod.CallEngine(str(tmp_path))
    call = calls_mod.PumpCall(mint="m")
    store = _StubStore()
    engine._grade(call, store, _boom)
    assert call.graded_via == "none"
    assert call.scored is False
    assert call.outcome == "miss-no-data"
    assert _sites() == ["calls.CallEngine._grade.snapshot"]


def test_converted_report_records_and_never_raises():
    rpc = chain_mod.SolanaRPC(on_outcome=_boom)
    assert rpc._report(True, time.time()) is None
    assert _sites() == ["chain.SolanaRPC._report"]


def test_converted_desk_init_records_and_leaves_notifier_none(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.agents.notifier", None)
    desk = desk_mod.PumpDesk(context=object(), desk_dir=str(tmp_path),
                             store=_StubStore())
    assert desk.notifier is None
    assert _sites() == ["desk.PumpDesk.__init__.notifier"]


def test_converted_record_source_records_and_never_raises(tmp_path):
    desk = desk_mod.PumpDesk(context=None, desk_dir=str(tmp_path),
                             store=_StubStore(),
                             registry=SimpleNamespace(for_domain=_boom))
    assert desk._record_source("example.com", True) is None
    assert _sites() == ["desk.PumpDesk._record_source"]


def test_converted_journal_state_records_and_never_raises(tmp_path):
    store = _StubStore(health_events_fn=_boom)
    desk = desk_mod.PumpDesk(context=None, desk_dir=str(tmp_path),
                             store=store)
    assert desk._journal_desk_state({"state": "degraded", "sources": []},
                                    time.time()) is None
    assert _sites() == ["desk.PumpDesk._journal_desk_state"]


def test_converted_firehose_watch_records_and_keeps_desk(tmp_path):
    store = _StubStore(dossiers_val=["!!!not-base58!!!"])
    desk = desk_mod.PumpDesk(context=None, desk_dir=str(tmp_path),
                             store=store, registry=SimpleNamespace(
                                 by_kind=lambda kind: []))
    fh = desk.firehose
    assert fh is not None
    assert _sites() == ["desk.PumpDesk.firehose.watch"]


def test_converted_ingest_records_and_never_raises(tmp_path):
    store = _StubStore(record_launch_fn=_boom)
    desk = desk_mod.PumpDesk(context=None, desk_dir=str(tmp_path),
                             store=store)
    assert desk._ingest("launch", {"mint": "x"}) is None
    assert _sites() == ["desk.PumpDesk._ingest"]


def test_converted_refresh_slot_records_and_returns_dossier(
        tmp_path, monkeypatch):
    monkeypatch.setattr(desk_mod._dossier, "build_dossier",
                        lambda *a, **k: {})
    monkeypatch.setattr(desk_mod._watch, "smart_confluence",
                        lambda *a, **k: None)
    store = _StubStore(launch_val={"signature": "sig", "slot": 0})
    rpc = SimpleNamespace(get_transaction=_boom)
    desk = desk_mod.PumpDesk(context=None, desk_dir=str(tmp_path),
                             store=store, rpc=rpc)
    assert desk.refresh_dossier("m") == {}
    assert _sites() == ["desk.PumpDesk.refresh_dossier.slot"]


def test_converted_backfill_setup_records_and_returns_zero(tmp_path):
    rpc = SimpleNamespace(get_signatures=_boom)
    desk = desk_mod.PumpDesk(context=None, desk_dir=str(tmp_path),
                             store=_StubStore(), rpc=rpc)
    mint = "So11111111111111111111111111111111111111112"
    assert desk._backfill_slot_trades(mint, 7) == 0
    assert _sites() == ["desk.PumpDesk._backfill_slot_trades.setup"]


def test_converted_backfill_row_records_and_skips_row(tmp_path):
    rows = [{"slot": 7, "signature": "s", "err": None}]
    rpc = SimpleNamespace(get_signatures=lambda *a, **k: rows,
                          get_transaction=_boom)
    desk = desk_mod.PumpDesk(context=None, desk_dir=str(tmp_path),
                             store=_StubStore(), rpc=rpc)
    mint = "So11111111111111111111111111111111111111112"
    assert desk._backfill_slot_trades(mint, 7) == 0
    assert _sites() == ["desk.PumpDesk._backfill_slot_trades.row"]


def _health_desk(tmp_path, states):
    desk = desk_mod.PumpDesk(
        context=None, desk_dir=str(tmp_path), store=_StubStore(),
        registry=SimpleNamespace(by_kind=lambda kind: []),
        notifier=SimpleNamespace(publish=_boom))
    return desk, states


def test_converted_health_standdown_records_and_reports_none(tmp_path):
    states = [{"domain": "a", "name": "A", "health": "down",
               "state": "down", "reason": "x"}]
    desk, _ = _health_desk(tmp_path, states)
    out = desk._health_report(states)
    assert out["alerted"] == []
    assert _sites() == ["desk.PumpDesk._health_report.standdown"]


def test_converted_health_feed_records_and_reports_none(tmp_path):
    states = [{"domain": "a", "name": "A", "health": "down",
               "state": "down", "reason": "x"},
              {"domain": "b", "name": "B", "health": "ok",
               "state": "ok", "reason": ""}]
    desk, _ = _health_desk(tmp_path, states)
    out = desk._health_report(states)
    assert out["alerted"] == []
    assert _sites() == ["desk.PumpDesk._health_report.feed"]


def test_converted_pool_exists_records_and_returns_false():
    state = SimpleNamespace(
        creator="So11111111111111111111111111111111111111112",
        quote_mint="So11111111111111111111111111111111111111112")
    rpc = SimpleNamespace(get_account=_boom)
    mint = "So11111111111111111111111111111111111111112"
    assert dossier_mod._pool_exists(rpc, state, mint) is False
    assert _sites() == ["dossier._pool_exists"]


def test_converted_shelve_notable_records_and_shelves_nothing(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.agents.pumpdesk.calls", None)
    store = _StubStore(dossiers_val={}, graph_val={})
    # _StubStore lacks shelve_knowledge; shelve_notable only calls it
    # when something earns shelving — with empty dossiers/graph it
    # must not reach it.
    out = shelf_mod.shelve_notable(store)
    assert out == {"dossiers": 0, "wallets": 0}
    assert _sites() == ["shelf.shelve_notable.calls"]


class _FakeDesc:
    def __init__(self, kinds, auth_needed):
        self.kinds = kinds
        self.auth_needed = auth_needed
        self.key = "k"


def test_converted_ensure_update_records_and_reports_updated(monkeypatch):
    class Reg:
        def __init__(self):
            self._by_key = {}

        def for_domain(self, domain):
            return _FakeDesc(("other",), False)

        def _persist(self):
            raise IOError("disk gone")

    out = sources_mod.ensure_pump_sources(Reg())
    assert out and all(e.get("updated") for e in out)
    assert _sites() and all(
        s == "sources.ensure_pump_sources.update" for s in _sites())


def test_converted_ensure_register_records_and_reports_registered():
    class Reg:
        def __init__(self):
            self._by_key = {}

        def for_domain(self, domain):
            return None

        def register(self, desc):
            return True

        def _persist(self):
            raise IOError("disk gone")

    out = sources_mod.ensure_pump_sources(Reg())
    assert out and all(e.get("registered") for e in out)
    assert _sites() and all(
        s == "sources.ensure_pump_sources.register" for s in _sites())


def test_converted_registry_bind_records_and_returns_registry(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.storage.kv", None)
    context = SimpleNamespace(db=object())
    reg = sources_mod.pump_registry_for_context(context)
    assert reg is not None
    assert _sites() == ["sources.pump_registry_for_context.bind"]


def test_converted_load_json_records_and_returns_empty(tmp_path):
    store = store_mod.PumpStore(str(tmp_path))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert store._load_json(str(bad)) == {}
    assert _sites() == ["store.PumpStore._load_json"]
    good = tmp_path / "good.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert store._load_json(str(good)) == {"a": 1}


# ---- Excluded handlers: proven NOT recording ----------------------------

def test_debug_honest_bad_log_line_records_nothing():
    out = chain_mod.decode_log_lines(["Program data: a"])
    assert out == []
    assert pd.failure_count() == 0


def test_debug_honest_ws_close_records_nothing():
    client = ws_mod.WebSocketClient("wss://example.invalid")

    class Sock:
        def sendall(self, data):
            raise OSError("send gone")

        def close(self):
            raise OSError("close gone")

    client._sock = Sock()
    client.close()
    assert client._sock is None
    assert pd.failure_count() == 0


def test_debug_honest_kg_mirrors_record_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.agents.kg", None)
    context = SimpleNamespace(db=object())
    assert shelf_mod._kg_upsert(context, {"mint": "m"}, None) is False
    assert shelf_mod._kg_upsert_wallet(context, {"wallet": "w"}) is False
    assert pd.failure_count() == 0


def test_debug_honest_tools_settings_records_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.core.config", None)
    assert sources_mod._tools_settings() is None
    assert pd.failure_count() == 0


def test_debug_honest_tier_probe_records_nothing(monkeypatch):
    monkeypatch.delenv("HELIUS_API_KEY", raising=False)
    reg = SimpleNamespace(for_domain=_boom)
    routes = sources_mod.rpc_routes(reg)
    assert len(routes) == len(sources_mod.PUBLIC_RPC_ENDPOINTS)
    assert pd.failure_count() == 0


def test_debug_honest_mint_gap_records_nothing():
    store = _StubStore(launch_val={})
    rpc = SimpleNamespace(get_curve=lambda mint: None,
                          get_mint_info=_boom)
    d = dossier_mod.build_dossier(store, rpc, "m")
    assert "mint account unreadable" in d["gaps"]
    assert pd.failure_count() == 0
