"""Tail-114 pins: nomorals/integrations/naija_deals.py census sites
(319, 633, 862). All three are held as exclusions:

319 — ``_ensure_schema`` additive-ALTER loop: re-running init on a
     database that already has the sweep columns raises duplicate-
     column inside the guard and is swallowed (in-code comment:
     "column already there"). Same family as missions/idempotency
     178 (tail-109). The schema lands exactly once and the store
     constructs.
633 — ``_parse_deals`` discount guard: ``float(discounts[i])``
     raising ValueError -> discount stays 0.0. REACHABILITY: the
     only source of discount tokens is the module's own
     ``(-?\\d+)%`` regex, whose matches are float-parseable by
     construction (arbitrary digit strings parse to inf at worst —
     probed); the guard is defense for a future pattern change.
     The forced-raise half below proves the fallback shape: 0.0 is
     filtered by ``min_discount`` like any no-discount deal, and
     with min_discount=0 the deal survives with the module's own
     ``price * 1.2`` no-discount original.
862 — ``_fx_rate``: fetch crash -> the stale cached rate, or 0.0
     when nothing was ever fetched. Docstring prints "best-effort";
     the value feeds only the USD display toggle.
"""

from __future__ import annotations

import builtins
import json
import urllib.request
from unittest.mock import MagicMock

from nomorals.integrations.naija_deals import NaijaDealHunter
from nomorals.storage.db import Database

HTML = '<div data-price="10000"></div><span>50% off</span>'


# ── 319: additive ALTERs are idempotent ──────────────────────────────

def test_319_reinit_swallows_duplicate_column_and_schema_lands_once():
    db = Database(":memory:")
    NaijaDealHunter(MagicMock(), db)
    cols = [r["name"] for r in db.query("PRAGMA table_info(price_history)")]
    assert "in_stock" in cols and "coupon" in cols
    # second construction re-runs both ALTERs into the guard
    NaijaDealHunter(MagicMock(), db)
    cols2 = [r["name"] for r in db.query("PRAGMA table_info(price_history)")]
    assert cols2 == cols


# ── 633: unparseable discount treated as 0 ───────────────────────────

def _hunter() -> NaijaDealHunter:
    return NaijaDealHunter.__new__(NaijaDealHunter)


def test_633_forced_discount_failure_is_filtered_like_no_discount(monkeypatch):
    real_float = builtins.float

    def picky(x):
        if x == "50":
            raise ValueError("forced")
        return real_float(x)

    monkeypatch.setattr(builtins, "float", picky)
    deals = _hunter()._parse_deals(HTML, "jumia", 999999.0, 60.0)
    assert deals == []


def test_633_forced_discount_failure_fallback_shape(monkeypatch):
    real_float = builtins.float

    def picky(x):
        if x == "50":
            raise ValueError("forced")
        return real_float(x)

    monkeypatch.setattr(builtins, "float", picky)
    deals = _hunter()._parse_deals(HTML, "jumia", 999999.0, 0.0)
    assert len(deals) == 1
    assert deals[0].discount_percent == 0.0
    assert deals[0].original_price == 10000.0 * 1.2


def test_633_control_parseable_discount_flows_through():
    deals = _hunter()._parse_deals(HTML, "jumia", 999999.0, 10.0)
    assert len(deals) == 1
    assert deals[0].discount_percent == 50.0
    assert deals[0].original_price == 20000.0


# ── 862: FX rate best-effort fallback ────────────────────────────────

def test_862_fetch_crash_returns_stale_cache(monkeypatch):
    hunter = _hunter()
    hunter._fx = 1500.0
    hunter._fx_at = 0.0

    def _down(*a, **k):
        raise RuntimeError("fx down")

    monkeypatch.setattr(urllib.request, "urlopen", _down)
    assert hunter._fx_rate() == 1500.0


def test_862_fetch_crash_without_cache_returns_zero(monkeypatch):
    hunter = _hunter()

    def _down(*a, **k):
        raise RuntimeError("fx down")

    monkeypatch.setattr(urllib.request, "urlopen", _down)
    assert hunter._fx_rate() == 0.0


def test_862_control_successful_fetch_returns_live_rate(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"rate": 1550.25}).encode()

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp())
    hunter = _hunter()
    assert hunter._fx_rate() == 1550.25
