"""Trading-pipeline timeout fixes (2026-11-14 incident).

The ``trading pipeline`` scheduled job timed out after 300s on EC2.
The alert named three dwells, each traced to a root cause and fixed
here — a test per root cause, failing-first on pristine ec8c6c38:

(a) TLS context build: ``HttpClient.__init__`` built a fresh
    ``ssl.create_default_context()`` on EVERY init, and the market-data
    layer builds a fresh HttpClient on EVERY fetch — 7+ store
    enumerations per tick on a host without certifi. Fixed by caching
    one verifying context per CA bundle in ``core/http.py``.

(b) DB wait: ``record_quote_outcome`` persisted the whole learned
    price state to the KV store (a DB write through the writer
    actor) on EVERY fetch outcome — 7+ writes per tick, each a
    chance to queue behind another job's open transaction. Fixed
    by debouncing the persist to at most once a minute in
    ``integrations/market_data.py``.

(c) JSON parse: the pipeline's ``_tick_seen`` / ``control_state`` /
    ``pipeline_status`` parsed the ENTIRE unbounded trading journal
    with ``read_events(limit=0, ...)`` on every tick, just to find
    the newest record of one kind. Fixed by a tail-read
    (``latest_event``) in ``finance/trading_journal.py``.
"""

from __future__ import annotations

import json
import ssl
from unittest import mock

import pytest

from nomorals.agents import trade_pipeline as pipeline
from nomorals.core import http as http_mod
from nomorals.core.http import HttpClient
from nomorals.finance import trading_journal
from nomorals.integrations import market_data as md

# ── (a) the TLS context is built once per CA bundle ──────────────────


def test_tls_context_cached_per_cafile():
    """Same cafile -> the identical context; a different cafile ->
    a different context. Before the fix every call built a new one."""
    first = http_mod._build_tls_context(None)
    assert http_mod._build_tls_context(None) is first
    assert http_mod._build_tls_context(None) is first
    # A second, real bundle: copy the resolved bundle to a new path
    # so the cache key differs while the content stays valid.
    bundle = http_mod.tls_cafile()
    if bundle is None:
        pytest.skip("no CA bundle resolvable on this box")
    import shutil
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "ca.pem"
        shutil.copyfile(bundle, copy)
        other = http_mod._build_tls_context(str(copy))
        assert other is not first
        assert http_mod._build_tls_context(str(copy)) is other


def test_two_http_clients_share_one_tls_context():
    """The regression as the pipeline pays it: a fresh HttpClient per
    fetch used to mean a fresh TLS context per fetch."""
    c1 = HttpClient(timeout=20.0)
    c2 = HttpClient(timeout=20.0)
    assert c1._tls_context is c2._tls_context
    assert isinstance(c1._tls_context, ssl.SSLContext)


# ── (c) the journal tail-read ─────────────────────────────────────────


def _write_journal(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _counting_loads(calls):
    orig = json.loads

    def counting(s, *a, **kw):
        calls.append(s)
        return orig(s, *a, **kw)

    return counting


def test_latest_event_returns_newest_match(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        {"event": "pipeline_tick", "tick_id": "t1"},
        {"event": "market_read", "symbol": "XAUUSD"},
        {"event": "pipeline_tick", "tick_id": "t2"},
        {"event": "pipeline_tick", "tick_id": "t3"},
    ])
    rec = trading_journal.latest_event("pipeline_tick", path=journal)
    assert rec is not None and rec["tick_id"] == "t3"
    assert trading_journal.latest_event("market_read",
                                        path=journal)["symbol"] == "XAUUSD"


def test_latest_event_missing_empty_and_no_match(tmp_path):
    assert trading_journal.latest_event(
        "pipeline_tick", path=tmp_path / "nope.jsonl") is None
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert trading_journal.latest_event("pipeline_tick",
                                        path=empty) is None
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [{"event": "market_read"}])
    assert trading_journal.latest_event("pipeline_tick",
                                        path=journal) is None


def test_latest_event_parses_only_the_tail(tmp_path):
    """200 filler records + 1 target at the end: the tail-read parses
    exactly the last line. The old ``read_events(limit=0)`` path
    parsed all 201."""
    journal = tmp_path / "j.jsonl"
    records = [{"event": "market_read", "n": i} for i in range(200)]
    records.append({"event": "pipeline_tick", "tick_id": "t-last"})
    _write_journal(journal, records)
    calls: list = []
    with mock.patch.object(json, "loads", _counting_loads(calls)):
        rec = trading_journal.latest_event("pipeline_tick", path=journal)
    assert rec is not None and rec["tick_id"] == "t-last"
    assert len(calls) == 1, f"parsed {len(calls)} lines, expected 1"


def test_latest_event_finds_record_larger_than_one_chunk(tmp_path):
    """A record bigger than the 64 KiB read chunk is reassembled
    across chunk boundaries instead of being missed."""
    journal = tmp_path / "j.jsonl"
    records = [{"event": "market_read", "n": i} for i in range(10)]
    big = {"event": "pipeline_tick", "tick_id": "t-big",
           "pad": "x" * 200_000}
    records.append(big)
    _write_journal(journal, records)
    rec = trading_journal.latest_event("pipeline_tick", path=journal)
    assert rec is not None and rec["tick_id"] == "t-big"
    assert rec["pad"] == "x" * 200_000


def test_tick_seen_reads_only_the_tail(tmp_path):
    """The pipeline's already-ran check on a 200-record journal:
    before the fix ``_tick_seen`` parsed every line; now it reads
    the newest pipeline_tick record only."""
    journal = tmp_path / "j.jsonl"
    records = [{"event": "market_read", "n": i} for i in range(200)]
    records.append({"event": "pipeline_tick", "tick_id": "T1"})
    _write_journal(journal, records)
    calls: list = []
    with mock.patch.object(json, "loads", _counting_loads(calls)):
        seen = pipeline._tick_seen("T1", settings=None,
                                   journal_path=journal)
        missing = pipeline._tick_seen("T-other", settings=None,
                                      journal_path=journal)
    assert seen is not None and seen["tick_id"] == "T1"
    assert missing is None
    # One parse for the matching record; the miss re-reads the same
    # newest record once more — still a tail read, never the file.
    assert len(calls) <= 2, f"parsed {len(calls)} lines, expected <= 2"


def test_control_state_reads_only_the_tail(tmp_path):
    """``control_state`` runs on every tick too — it must not parse
    the whole journal either."""
    journal = tmp_path / "j.jsonl"
    records = [{"event": "market_read", "n": i} for i in range(200)]
    records.append({"event": "pipeline_control", "state": "stopped",
                    "actor": "brain", "reason": "rest"})
    _write_journal(journal, records)
    calls: list = []
    with mock.patch.object(json, "loads", _counting_loads(calls)):
        state = pipeline.control_state(journal_path=journal)
    assert state["state"] == "stopped" and state["actor"] == "brain"
    assert len(calls) == 1, f"parsed {len(calls)} lines, expected 1"


# ── (b) the price-state persist is debounced ──────────────────────────


class _CountingStore:
    """A KV store that counts writes; ``get`` returns nothing."""

    def __init__(self):
        self.writes: list = []

    def get(self, key):
        return None

    def set_raw(self, key, raw, fmt):
        self.writes.append((key, raw, fmt))

    def set(self, key, value):
        self.writes.append((key, value, None))


def test_price_state_persist_debounced():
    """Five fetch outcomes in a row -> ONE store write. Before the
    fix every outcome wrote the whole state (a DB write each)."""
    store = _CountingStore()
    old_store = md._price_store
    old_clock = getattr(md, "_last_price_persist", 0.0)
    try:
        assert md.bind_price_source_store(store) is True
        for _ in range(5):
            md.record_quote_outcome("binance", "fresh")
        assert len(store.writes) == 1, (
            f"{len(store.writes)} writes for 5 outcomes, expected 1")
        # After the interval (clock reset here instead of waiting
        # 60s) the next outcome persists again.
        if hasattr(md, "_last_price_persist"):
            md._last_price_persist = 0.0
        md.record_quote_outcome("binance", "fresh")
        assert len(store.writes) == 2
    finally:
        md._price_store = old_store
        if hasattr(md, "_last_price_persist"):
            md._last_price_persist = old_clock
        md._source_health.pop("binance", None)
