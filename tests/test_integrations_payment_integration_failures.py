"""Tail-75 pins for nomorals/integrations/payment_integration.py —
0 converts / 6 exclusions (1017, 1088, 1093, 1117, 1128, 1137).

6 tests for 6 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED (healthy fake returns a distinctive value and the
surface shows it), then the failure half asserting the pinned
degradation. The file contains no exception handlers, so it
adds zero census sites.

These six guard the module's DISPLAY surfaces (tx status card,
wallet card, receive QR). The money-moving paths are not here:
send/approve raise PaymentError loudly (the class docstring:
"always a loud failure — never a fake success"), and none of
these four methods has an in-tree caller that decides on its
output — format_wallet consumes fiat_value for display only.

Verdict notes (full grounds in the tail-75 SURVEY.md):
- 1017: ETH receipt lookup failure -> status "pending",
  byte-identical to a genuinely unmined tx (receipt None).
  Tension stated plainly; held because "pending" never
  overclaims (it is the not-confirmed state, true in both
  cases) and web3 raises for the normal unmined case, which
  is what the guard exists for. The BTC branch of the same
  method raises PaymentError instead — the asymmetry is
  surveyed in the SURVEY.
- 1088 / 1093: qrcode ImportError -> the documented ASCII
  fallback (the block labels itself "NOT a real QR" and the
  address text rides below it). 1093 is the lint-calm
  re-import probe on the same path.
- 1117: price-feed failure -> fiat_value 0.0, so the wallet
  card prints "≈ 0.00" beside a real balance and the total
  understates. Weakest exclusion of the batch: failure is
  byte-identical to a zero price on a user-facing figure,
  but no caller decides on it and the direction understates
  holdings rather than inflating them. NOT-recording pinned:
  the failure writes no log record (module logger has _log
  and uses it elsewhere). Observability sketch shelved in
  the SURVEY (follow-up feature family).
- 1128: balance-read failure -> the currency is skipped,
  byte-identical to a zero balance; the card total
  understates, never overstates. NOT-recording pinned.
- 1137: address lookup failure -> the balance line prints
  without the short address; cosmetic degrade only.
"""

import asyncio
import logging
import sys
import types
from unittest import mock

from nomorals.integrations import market_data
from nomorals.integrations.payment_integration import PaymentIntegration

PAY_LOG = "nomorals.integrations.payment_integration"
ADDRESS = "bc1qexampleaddress00000000000000000000000"


def _pi() -> PaymentIntegration:
    return PaymentIntegration(None, None)


class _FakeEth:
    def __init__(self, mode):
        self.mode = mode

    def get_transaction_receipt(self, txid):
        if self.mode == "raise":
            raise RuntimeError("rpc down")
        if self.mode == "none":
            return None
        return {"status": 1, "blockNumber": 42, "gasUsed": 21000}


class _FakeW3:
    def __init__(self, mode):
        self.eth = _FakeEth(mode)


def _fake_web3(mode):
    module = types.ModuleType("web3")
    provider = types.SimpleNamespace()
    module.Web3 = lambda *a, **k: _FakeW3(mode)
    module.Web3.HTTPProvider = staticmethod(lambda *a, **k: provider)
    return module


def test_track_tx_eth_receipt_failure_reads_pending():
    pi = _pi()
    # Control: a real receipt confirms.
    with mock.patch.dict(sys.modules, {"web3": _fake_web3("ok")}):
        ok = asyncio.run(pi.track_tx("0xabc", "ETH"))
    assert ok["status"] == "confirmed" and ok["confirmations"] == 12
    # Genuine unmined (receipt None) reads pending.
    with mock.patch.dict(sys.modules, {"web3": _fake_web3("none")}):
        unmined = asyncio.run(pi.track_tx("0xabc", "ETH"))
    assert unmined["status"] == "pending" and unmined["confirmations"] == 0
    # Failure half (site 1017): the lookup raising reads exactly
    # the same pending — never confirmed, never failed.
    with mock.patch.dict(sys.modules, {"web3": _fake_web3("raise")}):
        failed = asyncio.run(pi.track_tx("0xabc", "ETH"))
    assert failed["status"] == unmined["status"] == "pending"
    assert failed["confirmations"] == 0


async def _receive_ok(self, currency, wallet="main"):
    return ADDRESS


def test_receive_qr_missing_qrcode_falls_back_to_ascii():
    pi = _pi()
    with mock.patch.object(PaymentIntegration, "receive_crypto",
                           _receive_ok), \
         mock.patch.dict(sys.modules, {"qrcode": None}):
        out = asyncio.run(pi.receive_qr("BTC"))
    # Site 1088: ImportError swallowed -> ASCII fallback, and the
    # result is NOT the PNG markdown form.
    assert not out.startswith("![receive QR]")
    assert ADDRESS in out


def test_receive_qr_second_import_guard_also_falls_back():
    pi = _pi()
    with mock.patch.object(PaymentIntegration, "receive_crypto",
                           _receive_ok), \
         mock.patch.dict(sys.modules, {"qrcode": None}):
        out = asyncio.run(pi.receive_qr("BTC"))
    # Site 1093: flow passes the second (re-import) guard and
    # lands in _ascii_qr — proven by its honest not-a-QR label.
    assert "install `qrcode` for a real QR" in out
    assert ADDRESS in out


def test_fiat_value_feed_failure_returns_zero(caplog):
    pi = _pi()
    # Control: a healthy quote prices the amount.
    with mock.patch.object(market_data, "quote",
                           return_value={"price": 100.0}):
        assert asyncio.run(pi.fiat_value("BTC", 2.0)) == 200.0
    # Failure half (site 1117): feed down -> 0.0, and NOT recorded.
    with mock.patch.object(market_data, "quote",
                           side_effect=RuntimeError("feed down")):
        with caplog.at_level(logging.DEBUG, logger=PAY_LOG):
            caplog.clear()
            value = asyncio.run(pi.fiat_value("BTC", 2.0))
    assert value == 0.0
    assert [r for r in caplog.records if r.name == PAY_LOG] == []


async def _bal_raise(self, wallet, currency):
    raise RuntimeError("node down")


async def _bal_some(self, wallet, currency):
    return 1.5 if currency == "BTC" else 0.0


async def _fiat_ten(self, currency, amount, fiat="USD"):
    return 10.0


def test_format_wallet_balance_failure_skips_currency(caplog):
    pi = _pi()
    # Failure half (site 1128): every balance read fails -> the
    # card is header + zero total, no currency lines, no record.
    with mock.patch.object(PaymentIntegration, "get_balance",
                           _bal_raise):
        with caplog.at_level(logging.DEBUG, logger=PAY_LOG):
            caplog.clear()
            card = asyncio.run(pi.format_wallet())
    assert "**total ≈ 0.00 USD**" in card
    assert "•" not in card
    assert [r for r in caplog.records if r.name == PAY_LOG] == []
    # Control: the skip is per-currency — BTC reads fine and its
    # line appears while a failing ETH read is the one skipped.
    async def bal_mixed(self, wallet, currency):
        if currency == "ETH":
            raise RuntimeError("eth node down")
        return 1.5 if currency == "BTC" else 0.0

    async def recv_fail(self, currency, wallet="main"):
        raise RuntimeError("no wallet")

    with mock.patch.object(PaymentIntegration, "get_balance",
                           bal_mixed), \
         mock.patch.object(PaymentIntegration, "fiat_value",
                           _fiat_ten), \
         mock.patch.object(PaymentIntegration, "receive_crypto",
                           recv_fail):
        card = asyncio.run(pi.format_wallet())
    assert "**1.50000000 BTC**" in card
    assert "**total ≈ 10.00 USD**" in card


async def _recv_raise(self, currency, wallet="main"):
    raise RuntimeError("no wallet")


def test_format_wallet_address_failure_omits_address_only():
    pi = _pi()
    # Control: address known -> short form appended in backticks.
    with mock.patch.object(PaymentIntegration, "get_balance",
                           _bal_some), \
         mock.patch.object(PaymentIntegration, "fiat_value",
                           _fiat_ten), \
         mock.patch.object(PaymentIntegration, "receive_crypto",
                           _receive_ok):
        card = asyncio.run(pi.format_wallet())
    assert "`bc1qexampl…000000`" in card
    # Failure half (site 1137): address lookup fails -> the
    # balance line still prints, just without the address.
    with mock.patch.object(PaymentIntegration, "get_balance",
                           _bal_some), \
         mock.patch.object(PaymentIntegration, "fiat_value",
                           _fiat_ten), \
         mock.patch.object(PaymentIntegration, "receive_crypto",
                           _recv_raise):
        card = asyncio.run(pi.format_wallet())
    assert "**1.50000000 BTC** ≈ 10.00 USD" in card
    assert "…" not in card
