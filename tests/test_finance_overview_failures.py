"""Tail-113 pins: nomorals/finance/overview.py census sites (95, 107, 115).

All three are per-rail fail-soft probes whose crash answer is None —
the module's documented "no data" value; the overview renders the
rail as unpriced/unavailable and never invents a number:

95  — ``_crypto_to_ngn``: the injected price_fn raises -> None.
107 — ``_default_fx``: the FX source raises -> None (sibling shape
      of finance/alerts.py's ``_default_fx_fn``, tail-103).
115 — ``_default_price``: the market-data quote raises -> None.
"""

from __future__ import annotations

from unittest import mock

import nomorals.integrations.market_data as market_data
import nomorals.integrations.naija_shopping as naija_shopping
from nomorals.finance import overview


def _boom(*args, **kwargs):
    raise RuntimeError("source down")


def test_crypto_price_crash_returns_none() -> None:
    assert overview._crypto_to_ngn(
        "BTC", 10, price_fn=_boom, fx_fn=lambda cur: 100.0) is None


def test_crypto_price_control_computes() -> None:
    value = overview._crypto_to_ngn(
        "BTC", 10,
        price_fn=lambda sym, market: {"price": 2.0},
        fx_fn=lambda cur: 100.0)
    assert value == 2000.0


def test_default_fx_crash_returns_none() -> None:
    with mock.patch.object(naija_shopping, "get_fx_rate", side_effect=_boom):
        assert overview._default_fx("USD") is None


def test_default_fx_control_passes_rate_through() -> None:
    with mock.patch.object(naija_shopping, "get_fx_rate", return_value=1500.0):
        assert overview._default_fx("USD") == 1500.0


def test_default_price_crash_returns_none() -> None:
    with mock.patch.object(market_data, "quote", side_effect=_boom):
        assert overview._default_price("BTC", "crypto") is None


def test_default_price_control_passes_quote_through() -> None:
    with mock.patch.object(market_data, "quote", return_value={"price": 3.0}):
        assert overview._default_price("BTC", "crypto") == {"price": 3.0}
