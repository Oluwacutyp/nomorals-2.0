"""Pass 37 — CLI wiring for ``nm finance`` and ``nm cards``.

The fakes below deliberately mirror the REAL connector method sets
(MonoConnector / PlaidConnector / VirtualCardsConnector). That is the
point: the handlers were written against imagined method names, so a
fake that grows the imagined methods would mask the defects.

Pre-fix behaviour pinned by these tests (all reproduced on HEAD):
* ``finance accounts`` called ``conn.accounts()`` — neither connector
  has ``list_accounts``/``accounts`` — AttributeError for both providers.
* ``finance link`` / ``finance balance`` were in the argparse choices
  but fell through to "unknown action" (exit 1) — never wired, despite
  both connectors exposing begin_link / balance reads.
* ``finance transactions`` passed the account id positionally: for
  Plaid that lands in ``start_date`` (and ``end_date`` is missing);
  for Mono it returned a dict the handler then tried to slice.
  ``--days`` was parsed and ignored.
* A connector init failure emitted prose ``""`` — terminal users saw a
  blank line with exit 1 and no cause (JSON users saw the error).
* ALL THREE connectors were constructed with a ``db=`` kwarg their
  real ``__init__(vault, http=None)`` does not accept — ``nm finance``
  died at init (exit 1) and ``nm cards`` crashed with TypeError on
  EVERY invocation, status included. The fakes therefore mirror the
  real constructor signature too; a ``db=`` call turns these tests red.
* ``cards pause`` / ``cards close`` were in the argparse choices but
  fell through to "unknown action" — the connector's block_card /
  terminate_card were never called.
"""

from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
from unittest import mock

import nomorals.connectors.mono as mono_mod
import nomorals.connectors.plaid as plaid_mod
import nomorals.connectors.virtualcards as vc_mod
from nomorals.cmdline.commands.cards import _cmd_cards
from nomorals.cmdline.commands.finance import _cmd_finance
from nomorals.cmdline.commands.project import _cmd_project


class _Status:
    def to_dict(self):
        return {"connected": True}


class FakeMono:
    """Real MonoConnector surface (no list_accounts / accounts)."""

    def __init__(self, vault=None, http=None):
        self.link_calls = []
        self.tx_calls = []

    def status(self):
        return _Status()

    def begin_link(self, db=None, context=None):
        self.link_calls.append((db, context))
        return {"linked": False, "checkpoint_id": "chk-1",
                "next": "complete the widget, then resolve chk-1"}

    def link_account(self, code):
        return {"account_id": "m1"}

    def linked_accounts(self):
        return [{"account_id": "m1", "name": "GTBank"}]

    def get_account(self, account_id=""):
        return {"account": {"balance": 5000, "currency": "NGN",
                             "name": "GTBank"}}

    def get_transactions(self, account_id, start="", end="", **kw):
        self.tx_calls.append((account_id, start, end))
        return {"account_id": account_id,
                "transactions": [{"id": "t1"}, {"id": "t2"}], "meta": {}}


class FakePlaid:
    """Real PlaidConnector surface (date-window transactions)."""

    def __init__(self, vault=None, http=None):
        self.link_calls = []
        self.tx_calls = []

    def status(self):
        return _Status()

    def begin_link(self, db=None, context=None, **kw):
        self.link_calls.append((db, context))
        return {"linked": False, "checkpoint_id": "chk-2",
                "next": "finish Plaid Link, then resolve chk-2"}

    def get_accounts(self, access_token=None):
        return [{"institution_name": "Bank",
                 "accounts": [{"account_id": "p1", "name": "Checking"}]}]

    def get_balances(self, access_token=None):
        return [{"institution_name": "Bank", "accounts": [
            {"name": "Checking", "available": 10, "current": 12,
             "currency": "USD"}]}]

    def get_transactions(self, start_date, end_date, account_ids=None,
                         **kw):
        self.tx_calls.append((start_date, end_date, account_ids))
        return [{"institution_name": "Bank",
                 "transactions": [{"id": "t9"}], "total": 1}]


class FakeCards:
    """Real VirtualCardsConnector surface."""

    def __init__(self, vault=None, http=None):
        self.blocked = []
        self.terminated = []

    def status(self):
        return _Status()

    def list_cards(self):
        return [{"id": "c1"}]

    def create_card(self, **kw):
        return {"id": "c2"}

    def block_card(self, card_id):
        self.blocked.append(card_id)
        return {"id": card_id, "blocked": True}

    def terminate_card(self, card_id):
        self.terminated.append(card_id)
        return {"id": card_id, "terminated": True}


def _ctx():
    ctx = mock.Mock()
    ctx.db = None
    return ctx


def _run(fn, **kw):
    args = argparse.Namespace(json=False, **kw)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = fn(args, _ctx())
    return rc, out.getvalue() + err.getvalue()


class FinanceWiringTests(unittest.TestCase):
    def setUp(self):
        self.mono = FakeMono()
        self.plaid = FakePlaid()
        patchers = [
            mock.patch.object(mono_mod, "MonoConnector",
                              lambda vault=None, http=None: self.mono),
            mock.patch.object(plaid_mod, "PlaidConnector",
                              lambda vault=None, http=None: self.plaid),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def fin(self, **kw):
        kw.setdefault("provider", "mono")
        kw.setdefault("account_id", "")
        kw.setdefault("days", 30)
        return _run(_cmd_finance, **kw)

    def test_accounts_mono_uses_linked_accounts(self):
        rc, text = self.fin(action="accounts")
        self.assertEqual(rc, 0, text)
        self.assertIn("1 account(s)", text)

    def test_accounts_plaid_flattens_items(self):
        rc, text = self.fin(action="accounts", provider="plaid")
        self.assertEqual(rc, 0, text)
        self.assertIn("1 account(s)", text)

    def test_link_mono_starts_flow(self):
        rc, text = self.fin(action="link")
        self.assertEqual(rc, 0, text)
        self.assertEqual(len(self.mono.link_calls), 1)
        self.assertIn("chk-1", text)

    def test_link_plaid_starts_flow(self):
        rc, text = self.fin(action="link", provider="plaid")
        self.assertEqual(rc, 0, text)
        self.assertEqual(len(self.plaid.link_calls), 1)

    def test_balance_plaid_reads_balances(self):
        rc, text = self.fin(action="balance", provider="plaid")
        self.assertEqual(rc, 0, text)
        self.assertIn("Checking", text)

    def test_balance_mono_reads_linked_account_balance(self):
        rc, text = self.fin(action="balance")
        self.assertEqual(rc, 0, text)
        self.assertIn("GTBank", text)
        self.assertIn("5000", text)

    def test_transactions_mono_unwraps_dict_and_passes_window(self):
        rc, text = self.fin(action="transactions", account_id="m1", days=7)
        self.assertEqual(rc, 0, text)
        self.assertIn("2 transaction(s)", text)
        account_id, start, end = self.mono.tx_calls[0]
        self.assertEqual(account_id, "m1")
        self.assertEqual(end, date.today().isoformat())
        self.assertEqual(start, (date.today() - timedelta(days=7)).isoformat())

    def test_transactions_plaid_passes_dates_not_account_id(self):
        rc, text = self.fin(action="transactions", provider="plaid",
                            account_id="p1", days=7)
        self.assertEqual(rc, 0, text)
        self.assertIn("1 transaction(s)", text)
        start, end, account_ids = self.plaid.tx_calls[0]
        self.assertEqual(end, date.today().isoformat())
        self.assertEqual(start, (date.today() - timedelta(days=7)).isoformat())
        self.assertEqual(account_ids, ["p1"])

    def test_init_failure_prose_names_the_cause(self):
        def boom(vault=None, db=None):
            raise RuntimeError("vault locked")

        with mock.patch.object(mono_mod, "MonoConnector", boom):
            rc, text = _run(_cmd_finance, action="status", provider="mono",
                            account_id="", days=30)
        self.assertEqual(rc, 1)
        self.assertIn("mono init failed", text)
        self.assertIn("vault locked", text)

    def test_status_still_works(self):
        rc, text = self.fin(action="status")
        self.assertEqual(rc, 0, text)
        self.assertIn("Mono: connected", text)


class CardsWiringTests(unittest.TestCase):
    def setUp(self):
        self.cards = FakeCards()
        p = mock.patch.object(vc_mod, "VirtualCardsConnector",
                              lambda vault=None, http=None: self.cards)
        p.start()
        self.addCleanup(p.stop)

    def card(self, **kw):
        kw.setdefault("token", None)
        kw.setdefault("type", "UNLOCKED")
        kw.setdefault("limit", None)
        kw.setdefault("merchant", None)
        return _run(_cmd_cards, **kw)

    def test_pause_blocks_the_card(self):
        rc, text = self.card(action="pause", token="c1")
        self.assertEqual(rc, 0, text)
        self.assertEqual(self.cards.blocked, ["c1"])

    def test_close_terminates_the_card(self):
        rc, text = self.card(action="close", token="c1")
        self.assertEqual(rc, 0, text)
        self.assertEqual(self.cards.terminated, ["c1"])

    def test_pause_without_token_is_a_usage_error(self):
        rc, text = self.card(action="pause", token=None)
        self.assertEqual(rc, 1)
        self.assertIn("--token", text)

    def test_list_still_works(self):
        rc, text = self.card(action="list")
        self.assertEqual(rc, 0, text)
        self.assertIn("1 card(s)", text)


class ProjectHonestyTests(unittest.TestCase):
    """``nm project update``/``delete`` are parser-advertised, but
    ProjectManager has no such operations. Pre-fix they printed
    "unknown action" with exit 2 (the usage-error code) — claiming a
    valid verb did not exist. They must now say "not supported" with
    the command-failed code 1."""

    def proj(self, action):
        args = argparse.Namespace(json=False, action=action, title="p1",
                                  description="", id="")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = _cmd_project(args, mock.Mock())
        return rc, out.getvalue() + err.getvalue()

    def test_update_reports_not_supported(self):
        rc, text = self.proj("update")
        self.assertEqual(rc, 1, text)
        self.assertIn("not supported", text)

    def test_delete_reports_not_supported(self):
        rc, text = self.proj("delete")
        self.assertEqual(rc, 1, text)
        self.assertIn("not supported", text)


if __name__ == "__main__":
    unittest.main()
