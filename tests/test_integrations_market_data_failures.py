"""Pass #204 pins: core-ledger durability (reused by market_data) +
the market_data convert and the NOT-recording exclusions.

The market_data convert (site 349) records through the REUSED core
ledger (``nomorals.core.failures``, namespace ``core``) — see
SURVEY.md for the layering reason (market_data is pinned at L2, so a
dedicated integrations ledger at L5 would add an upward import).
Site name: ``market_data.record_quote_outcome``.

Every test runs with NM_HOME redirected at a scratch dir and the
ledger's in-memory state reset, so no test touches the real home.
No test in this file uses a bare handler: there are no ``except``
statements at all — probes assert on return values.
"""

import json
import os
import subprocess
import sys
import threading

import pytest

from nomorals.core import failures as led
from nomorals.core import failure_store as store_mod
from nomorals.integrations import market_data as md


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Redirect NM_HOME and reset ledger memory before every test."""
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    led._ledger.clear()
    led._total = 0
    led._loaded = False
    yield
    led._ledger.clear()
    led._total = 0
    led._loaded = False


def _store(ns="core"):
    return store_mod.Store(ns)


# ── durability (17, mirroring #141+) ─────────────────────────────────


def test_round_trip():
    led.record_failure("market_data.probe", ValueError("boom"))
    recs = led.recent_failures(5)
    assert len(recs) == 1
    assert recs[0].site == "market_data.probe"
    assert recs[0].exc_type == "ValueError"
    assert recs[0].message == "boom"
    assert led.failure_count() == 1


def test_store_cap_250_to_200_total_preserved():
    st = _store("cap-probe")
    rows = [{"ts": float(i), "site": f"s{i}", "exc_type": "E",
             "message": "m"} for i in range(250)]
    assert st.save(rows, 250) is True
    entries, total = st.load()
    assert len(entries) == 200
    assert total == 250
    assert entries[0]["site"] == "s50"
    assert entries[-1]["site"] == "s249"


def test_ledger_cap_250_200_total_preserved():
    for i in range(250):
        led.record_failure(f"market_data.bulk{i}", RuntimeError("x"))
    assert len(led.recent_failures(500)) == 200
    assert led.failure_count() == 250


def test_clear_gone():
    led.record_failure("market_data.probe", ValueError("x"))
    assert led.failure_count() == 1
    led.clear_failures()
    assert led.recent_failures() == []
    assert led.failure_count() == 0
    assert not _store("core").path().exists()


def test_namespace_isolation():
    _store("ns-a").save([{"ts": 1.0, "site": "a", "exc_type": "E",
                          "message": "m"}], 1)
    entries, total = _store("ns-b").load()
    assert entries == [] and total == 0
    assert _store("ns-a").path() != _store("ns-b").path()


def test_corrupt_recovery_store_level():
    st = _store("corrupt")
    st.path().parent.mkdir(parents=True, exist_ok=True)
    st.path().write_text("{not json", encoding="utf-8")
    entries, total = st.load()
    assert entries == [] and total == 0


def test_save_failure_ancestor_is_file(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir", encoding="utf-8")
    st = store_mod.Store("x")
    before = store_mod.write_failures()
    # Point the store under the file: mkdir must fail.
    import nomorals.core.failure_store as fsm
    old = os.environ.get("NM_HOME")
    os.environ["NM_HOME"] = str(blocker / "sub")
    try:
        assert st.save([{"ts": 1.0, "site": "s", "exc_type": "E",
                         "message": "m"}], 1) is False
    finally:
        if old is None:
            del os.environ["NM_HOME"]
        else:
            os.environ["NM_HOME"] = old
    assert store_mod.write_failures() == before + 1


def test_save_failure_failures_dir_is_file(tmp_path, monkeypatch):
    (tmp_path / "failures").write_text("file, not dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    st = _store("x")
    assert st.save([{"ts": 1.0, "site": "s", "exc_type": "E",
                     "message": "m"}], 1) is False


def test_save_failure_counter_bumps(tmp_path, monkeypatch):
    (tmp_path / "failures").write_text("file, not dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    before = store_mod.write_failures()
    assert _store("x").save([{"ts": 1.0, "site": "s", "exc_type": "E",
                              "message": "m"}], 1) is False
    assert store_mod.write_failures() == before + 1
    assert store_mod.last_error() is not None


def test_concurrent_saves():
    st = _store("conc")
    errors = []

    def worker(n):
        ok = st.save([{"ts": float(n), "site": f"w{n}",
                       "exc_type": "E", "message": "m"}], n + 1)
        if not ok:
            errors.append(n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, total = st.load()
    assert isinstance(entries, list) and total >= 1
    for row in entries:
        assert set(row) == {"ts", "site", "exc_type", "message"}


def test_disk_hit():
    led.record_failure("market_data.disk", ValueError("on-disk"))
    text = _store("core").path().read_text(encoding="utf-8")
    assert "market_data.disk" in text
    assert "on-disk" in text


def test_concurrent_records():
    errors = []

    def worker(n):
        for i in range(25):
            led.record_failure(f"market_data.t{n}-{i}", ValueError("c"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert led.failure_count() == 200


def test_subprocess_restart(tmp_path):
    code_write = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.core.failures import record_failure;"
        "record_failure('market_data.sub', ValueError('persisted'));"
        "print('wrote')"
    )
    code_read = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.core.failures import recent_failures, failure_count;"
        "recs = recent_failures(5);"
        "print(recs[0].site if recs else 'EMPTY');"
        "print(failure_count())"
    )
    env = dict(os.environ, NM_HOME=str(tmp_path))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # The tree root is the repo checkout's parent of tests/; fall back
    # to the cwd the suite runs from when that guess is wrong.
    cwd = tree if os.path.isdir(os.path.join(tree, "nomorals")) else "."
    w = subprocess.run([sys.executable, "-c", code_write], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert w.returncode == 0, w.stderr
    r = subprocess.run([sys.executable, "-c", code_read], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "market_data.sub"
    assert lines[1] == "1"


def test_broken_home_ledger_level(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker / "sub"))
    # Disk writes fail, but the in-memory ledger still records.
    led.record_failure("market_data.mem", ValueError("mem-only"))
    assert led.failure_count() == 1
    assert led.recent_failures(1)[0].site == "market_data.mem"


def test_format_failures():
    assert led.format_failures() == "no core failures recorded."
    led.record_failure("market_data.fmt", ValueError("hello"))
    out = led.format_failures()
    assert "market_data.fmt" in out
    assert "ValueError" in out and "hello" in out


def test_clear_gone_on_reload_subprocess(tmp_path):
    code_write = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.core.failures import record_failure, clear_failures;"
        "record_failure('market_data.tmp', ValueError('gone'));"
        "clear_failures(); print('cleared')"
    )
    code_read = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.core.failures import recent_failures, failure_count;"
        "print(len(recent_failures())); print(failure_count())"
    )
    env = dict(os.environ, NM_HOME=str(tmp_path))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = tree if os.path.isdir(os.path.join(tree, "nomorals")) else "."
    w = subprocess.run([sys.executable, "-c", code_write], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert w.returncode == 0, w.stderr
    r = subprocess.run([sys.executable, "-c", code_read], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines() == ["0", "0"]


def test_total_preserved_across_restart(tmp_path):
    code_write = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.core.failures import record_failure;"
        "[record_failure(f'market_data.t{i}', ValueError('v'))"
        " for i in range(3)]; print('wrote3')"
    )
    code_read = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.core.failures import failure_count;"
        "print(failure_count())"
    )
    env = dict(os.environ, NM_HOME=str(tmp_path))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = tree if os.path.isdir(os.path.join(tree, "nomorals")) else "."
    w = subprocess.run([sys.executable, "-c", code_write], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert w.returncode == 0, w.stderr
    r = subprocess.run([sys.executable, "-c", code_read], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "3"


# ── behavior pin for the one convert ─────────────────────────────────


def test_convert_349_records_and_never_raises():
    h = md._blank_health()
    h["fails"] = "corrupt"
    md._source_health["probe-349"] = h
    try:
        md.record_quote_outcome("probe-349", "failure")
    finally:
        md._source_health.pop("probe-349", None)
    recs = led.recent_failures(5)
    assert any(r.site == "market_data.record_quote_outcome"
               for r in recs)
    assert any(r.exc_type == "TypeError" for r in recs)


# ── NOT-recording pins (one per exclusion) ───────────────────────────


def test_excl_447_chain_as_is_no_record(monkeypatch):
    def _boom(src):
        raise RuntimeError("stats down")
    monkeypatch.setattr(md, "price_source_stats", _boom)
    pool = ["binance", "coingecko"]
    assert md.ordered_price_chain("crypto", pool) == pool
    assert led.recent_failures() == []


def test_excl_508_bind_load_failure_no_record(monkeypatch):
    old_store = md._price_store

    def _boom():
        raise RuntimeError("load down")
    monkeypatch.setattr(md, "_load_price_state", _boom)
    try:
        assert md.bind_price_source_store(object()) is False
    finally:
        md._price_store = old_store
    assert led.recent_failures() == []


def test_excl_570_clear_delete_failure_no_record(monkeypatch):
    class StubStore:
        def get(self, key):
            return None

        def delete(self, key):
            raise RuntimeError("delete down")

    old_store = md._price_store
    try:
        assert md.bind_price_source_store(StubStore()) is True
        assert md.clear_price_source_state() is False
    finally:
        md._price_store = old_store
    assert led.recent_failures() == []


def test_excl_635_frame_ts_none_no_record():
    class BoomIndex:
        @property
        def index(self):
            raise RuntimeError("index down")

    assert md._frame_last_ts(BoomIndex()) is None
    assert led.recent_failures() == []


def test_excl_1072_iso_parse_none_no_record():
    assert md._parse_iso_ts("not-a-date") is None
    assert md._parse_iso_ts("") is None
    assert led.recent_failures() == []


def test_excl_1086_date_parse_none_no_record():
    assert md._parse_date_ts("99/99/9999") is None
    assert md._parse_date_ts("") is None
    assert led.recent_failures() == []


def test_excl_1159_open_er_no_timestamp_no_record(monkeypatch):
    def _fake_json(url, params=None):
        return {"rates": {"USD": 1.2}}
    monkeypatch.setattr(md, "_json", _fake_json)
    monkeypatch.setattr(md, "_live_chain",
                        lambda market, chain: ["open_er_api"])
    result = md.quote("EURUSD", market="forex")
    assert result["price"] == 1.2
    # _record_quote_result always stamps the key: the genuine
    # no-timestamp outcome is quote_ts None, not a missing key.
    assert result["quote_ts"] is None
    assert result["freshness"] == "unknown"
    assert led.recent_failures() == []


def test_excl_1418_attrs_surface_no_record(monkeypatch):
    class BoomAttrs(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("attrs down")

    class StubFrame:
        @property
        def index(self):
            raise RuntimeError("index down")

        @property
        def attrs(self):
            return BoomAttrs()

        def __len__(self):
            return 1

    monkeypatch.setitem(md._FETCHERS, "stubsrc",
                        lambda sym, market, timeframe, bars: StubFrame())
    try:
        df = md.get_ohlcv("BTCUSDT", market="crypto", source="stubsrc")
    finally:
        md._FETCHERS.pop("stubsrc", None)
    assert isinstance(df, StubFrame)
    assert led.recent_failures() == []


def test_excl_1581_stooq_time_none_no_record(monkeypatch):
    csv_text = ("Date,Time,Close\n2026-10-24,not-a-time,123.45\n")

    def _fake_fetch(url, params=None):
        return csv_text.encode("utf-8")
    monkeypatch.setattr(md, "_fetch", _fake_fetch)
    monkeypatch.setattr(md, "_live_chain",
                        lambda market, chain: ["stooq"])
    result = md.quote("AAPL", market="stocks")
    assert result["price"] == 123.45
    import calendar
    import time as _time
    assert result["quote_ts"] == float(
        calendar.timegm(_time.strptime("2026-10-24", "%Y-%m-%d")))
    assert led.recent_failures() == []


def test_excl_1759_status_degraded_no_record(monkeypatch):
    def _boom(src):
        raise RuntimeError("stats down")
    monkeypatch.setattr(md, "price_source_stats", _boom)
    out = md.price_source_status()
    assert out["sources"] == [] and out["chains"] == {}
    assert led.recent_failures() == []


def test_excl_2146_closes_stdlib_no_record(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert md._closes([[1, 2, 3, 4, 5, 6]]) == [5.0]
    assert led.recent_failures() == []


def test_excl_2192_indicators_stdlib_no_record(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    rows = [[1000 + i, 10.0, 11.0, 9.0, 10.5 + i * 0.1, 100.0]
            for i in range(40)]
    out = md.indicators(rows)
    assert isinstance(out, dict)
    assert led.recent_failures() == []
