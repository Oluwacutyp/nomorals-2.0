"""Tail-103 pins: nomorals/finance/alerts.py census sites.

All four sites are held as exclusions:

162 — ``AlertStore._load``: corrupt or non-dict JSON ->
``{}``. This was the batch's most consequential site: the
store's mutators (add/update/remove) all load-then-rewrite,
so a corrupt alerts.json was silently emptied by the next
mutation. CONVERTED by the data-loss targeted fix (Pass 1):
the load still reads as empty, but the corrupt bytes are
quarantined to ``alerts.json.corrupt``, the failure is
recorded in the core failures ledger, and a mutation on the
poisoned instance raises ``StoreUnreadableError`` instead of
rewriting over the loss. A fresh instance starts clean.

192 — ``AlertStore.get``: a stored record whose
``created_at``/``fire_count`` cannot be coerced makes
``Alert.from_dict`` raise -> the record reads as absent
(None), the same answer a genuinely missing id gets.
Per-record parse guard; nothing is fabricated.

200 — ``AlertStore.list``: same guard, loop form — the
malformed record is skipped, healthy records still list.

279 — ``_default_fx_fn``: the FX source raising -> None,
which is the surface's documented "no data" value:
``evaluate_alerts`` leaves the alert unchecked ("a broken
source marks the alert unchecked, not dead"). It never
fires on a fabricated rate. Shelved sketch: log at debug
like the sibling ``_default_price_fn`` does.

NOT-recording is pinned with the snapshot method for sites
192/200/279: both the agency and agents ledgers must be
unchanged across each trigger. Site 162 now records into
the CORE ledger (via core/storeguard) — the agency/agents
snapshot still holds for it.
"""
from __future__ import annotations

from unittest import mock

import pytest

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.core.failures as core_fail
import nomorals.integrations.naija_shopping as naija_shopping
from nomorals.core.storeguard import StoreUnreadableError
from nomorals.finance import alerts as A


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ── 162: corrupt store file loads as empty ──────────────────────

def test_162_corrupt_store_loads_empty(tmp_path):
    # CONVERTED (data-loss Pass 1): the corrupt load still reads
    # as empty, but the bytes are quarantined + ledgered and the
    # poisoned instance refuses to rewrite over them.
    before = _snap()
    path = tmp_path / "alerts.json"
    store = A.AlertStore(path)
    path.write_text("{corrupt json", encoding="utf-8")
    assert store._load() == {}
    assert (tmp_path / "alerts.json.corrupt").read_text(
        encoding="utf-8") == "{corrupt json"
    # valid JSON, wrong shape -> also empty, never a crash
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert store._load() == {}
    assert store.list() == []
    # the mutation that used to silently empty the registry refuses
    with pytest.raises(StoreUnreadableError):
        A.add_alert("price_above", "BTC", 100.0, store=store)
    assert "finance.alerts._load" in [
        f.site for f in core_fail.recent_failures(limit=200)]
    # control: a fresh store still works end to end afterwards
    fresh = A.AlertStore(path)
    alert = A.add_alert("price_above", "BTC", 100.0, store=fresh)
    assert fresh.get(alert.id) is not None
    assert [a.id for a in fresh.list()] == [alert.id]
    assert _snap() == before


# ── 192: unparseable record reads as absent ─────────────────────

def _store_with_bad_record(tmp_path):
    store = A.AlertStore(tmp_path / "alerts.json")
    good = A.add_alert("price_above", "BTC", 100.0, store=store)
    data = store._load()
    data["bad"] = {"id": "bad", "kind": "price_above", "target": "ETH",
                   "threshold": 5.0, "created_at": "not-a-number"}
    store._save(data)
    return store, good


def test_192_bad_record_get_returns_none(tmp_path):
    before = _snap()
    store, good = _store_with_bad_record(tmp_path)
    assert store.get("bad") is None
    # control: the healthy record still resolves
    got = store.get(good.id)
    assert got is not None and got.target == "BTC"
    assert _snap() == before


# ── 200: list skips the unparseable record ──────────────────────

def test_200_bad_record_skipped_in_list(tmp_path):
    before = _snap()
    store, good = _store_with_bad_record(tmp_path)
    assert [a.id for a in store.list()] == [good.id]
    assert _snap() == before


# ── 279: dead FX source is "no data", never a rate ──────────────

def test_279_fx_failure_returns_none():
    before = _snap()
    with mock.patch.object(naija_shopping, "get_fx_rate", _raise):
        assert A._default_fx_fn("USD") is None
    # control: a healthy source's rate flows through untouched
    with mock.patch.object(naija_shopping, "get_fx_rate",
                           lambda base, quote: 1600.0):
        assert A._default_fx_fn("USD") == 1600.0
    assert _snap() == before
