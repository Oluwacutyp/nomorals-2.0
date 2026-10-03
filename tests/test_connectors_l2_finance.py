"""Mono + Plaid connector tests. HTTP is fully mocked — no network."""

from __future__ import annotations

import json
import os
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.checkpoints import (
    CheckpointKind,
    CheckpointState,
    CheckpointStore,
    HumanCheckpointPending,
)
from nomorals.connectors.mono import MonoConnector, MonoError
from nomorals.connectors.plaid import PlaidConnector, PlaidError
from nomorals.connectors.registry import get_connector
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


# ── fake HTTP ────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        raw: str = "",
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._payload = payload
        self._raw = raw

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        if self._raw:
            return self._raw
        return json.dumps(self._payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON here")
        return self._payload

    def raise_for_status(self) -> "FakeResponse":
        return self


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(
        self, method: str, url: str, payload: Any = None, **kw: Any
    ) -> FakeResponse:
        self.calls.append((method.upper(), url, payload, kw.get("headers")))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"message": "not mocked"})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            # mirror HttpClient: query params are encoded into the URL
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, None, **kw)

    def last_post_body(self) -> Any:
        for method, _url, payload, _headers in reversed(self.calls):
            if method == "POST":
                return payload
        return None


# ═══════════════════════════════════════════════════════════════════════
# Mono
# ═══════════════════════════════════════════════════════════════════════

MONO_ACCOUNT = {
    "status": "successful",
    "data": {
        "account": {
            "_id": "acc_123",
            "name": "Adaeze Obi",
            "accountNumber": "0123456789",
            "currency": "NGN",
            "balance": 1250000,
            "type": "SAVINGS_ACCOUNT",
            "bvn": "22123456789",
            "institution": {
                "name": "GTBank",
                "bankCode": "058",
                "type": "PERSONAL_BANKING",
            },
        },
        "meta": {"data_status": "AVAILABLE"},
    },
}


def _mono(http: FakeHttp | None = None) -> tuple[MonoConnector, FakeHttp]:
    http = http or FakeHttp()
    return MonoConnector(_vault(), http=http), http


def _mono_connected(
    http: FakeHttp | None = None, *, linked: bool = False
) -> tuple[MonoConnector, FakeHttp]:
    conn, http = _mono(http)
    http.route("GET", "/institutions",
               FakeResponse(200, {"status": "ok", "data": []}))
    if linked:
        http.route("POST", "/accounts/auth",
                   FakeResponse(200, {"id": "acc_123"}))
        http.route("GET", "/accounts/acc_123", FakeResponse(200, MONO_ACCOUNT))
        conn.connect(secret_key="test_sk_123", code="widget-code-1")
    else:
        conn.connect(secret_key="test_sk_123")
    return conn, http


class MonoRegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("mono"), MonoConnector)

    def test_metadata(self) -> None:
        self.assertEqual(MonoConnector.id, "mono")
        self.assertIn("mono-sec-key", MonoConnector.description)


class MonoConnectTests(unittest.TestCase):
    def test_connect_stores_secret_and_validates(self) -> None:
        conn, http = _mono()
        http.route("GET", "/institutions",
                   FakeResponse(200, {"status": "ok", "data": []}))
        result = conn.connect(secret_key="test_sk_123")
        self.assertTrue(result.ok)
        cred = conn.vault.get("connector:mono", "mono")
        self.assertEqual(cred.password, "test_sk_123")
        # validation call carried the secret in the header, never in the URL
        _m, url, _p, headers = http.calls[0]
        self.assertIn("api.withmono.com/v2/institutions", url)
        self.assertEqual(headers["mono-sec-key"], "test_sk_123")
        self.assertNotIn("test_sk_123", url)

    def test_connect_reads_env_var(self) -> None:
        conn, http = _mono()
        http.route("GET", "/institutions",
                   FakeResponse(200, {"status": "ok", "data": []}))
        with mock.patch.dict(os.environ, {"MONO_SECRET_KEY": "test_sk_env"}):
            conn.connect()
        self.assertEqual(
            conn.vault.get("connector:mono", "mono").password, "test_sk_env"
        )

    def test_connect_with_code_links_account(self) -> None:
        conn, http = _mono_connected()
        http.route("POST", "/accounts/auth",
                   FakeResponse(200, {"id": "acc_123"}))
        http.route("GET", "/accounts/acc_123", FakeResponse(200, MONO_ACCOUNT))
        result = conn.link_account("widget-code-1")
        self.assertEqual(result["account_id"], "acc_123")
        self.assertIn("Adaeze Obi", result["label"])
        linked = conn.vault.get("connector:mono", "mono").metadata[
            "linked_accounts"]
        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["account_id"], "acc_123")

    def test_connect_with_code_envelope_shape(self) -> None:
        conn, http = _mono()
        http.route("GET", "/institutions",
                   FakeResponse(200, {"status": "ok", "data": []}))
        http.route("POST", "/accounts/auth", FakeResponse(
            200, {"status": "successful", "data": {"id": "acc_999"}}))
        http.route("GET", "/accounts/acc_999", FakeResponse(200, MONO_ACCOUNT))
        result = conn.connect(secret_key="test_sk_123", code="c")
        self.assertTrue(result.ok)
        self.assertIn("Adaeze Obi", result.account)

    def test_connect_rejects_second_key(self) -> None:
        conn, _http = _mono_connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(secret_key="other")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_invalid_key_fails_fast(self) -> None:
        conn, http = _mono()
        http.route("GET", "/institutions", FakeResponse(401, {
            "status": "failed", "message": "invalid key"}))
        with self.assertRaises(MonoError) as ctx:
            conn.connect(secret_key="bad_key")
        self.assertIn("401", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_no_tty_no_env_fails_fast(self) -> None:
        conn, _http = _mono()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MONO_SECRET_KEY", None)
            with mock.patch("sys.stdin.isatty", return_value=False):
                with self.assertRaises(ConnectorError) as ctx:
                    conn.connect()
        self.assertIn("MONO_SECRET_KEY", str(ctx.exception))


class MonoStatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _mono()
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("not connected", st.detail)

    def test_status_connected_no_accounts(self) -> None:
        conn, http = _mono_connected()
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("no accounts linked", st.account)

    def test_status_connected_with_account(self) -> None:
        conn, http = _mono()
        http.route("GET", "/institutions",
                   FakeResponse(200, {"status": "ok", "data": []}))
        http.route("POST", "/accounts/auth",
                   FakeResponse(200, {"id": "acc_123"}))
        http.route("GET", "/accounts/acc_123", FakeResponse(200, MONO_ACCOUNT))
        conn.connect(secret_key="test_sk_123", code="c")
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("Adaeze Obi", st.account)

    def test_status_rejected_key(self) -> None:
        conn, http = _mono_connected()
        http.routes.clear()
        http.route("GET", "/institutions", FakeResponse(401, {}))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)

    def test_test_connection(self) -> None:
        conn, http = _mono_connected()
        self.assertTrue(conn.test_connection())
        http.routes.clear()
        http.route("GET", "/institutions", FakeResponse(401, {}))
        self.assertFalse(conn.test_connection())

    def test_test_connection_not_connected(self) -> None:
        conn, _http = _mono()
        self.assertFalse(conn.test_connection())

    def test_disconnect_idempotent(self) -> None:
        conn, _http = _mono_connected()
        conn.disconnect()
        conn.disconnect()
        self.assertIsNone(conn._load_credential())


class MonoReadTests(unittest.TestCase):
    def test_get_account(self) -> None:
        conn, http = _mono_connected(linked=True)
        info = conn.get_account("acc_123")
        self.assertEqual(info["account"]["name"], "Adaeze Obi")

    def test_get_account_defaults_to_single_linked(self) -> None:
        conn, http = _mono_connected(linked=True)
        info = conn.get_account()
        self.assertEqual(info["account_id"], "acc_123")

    def test_get_account_no_linked_raises(self) -> None:
        conn, _http = _mono_connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_account()
        self.assertIn("no bank account linked", str(ctx.exception))

    def test_get_transactions_params(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/transactions", FakeResponse(200, {
            "status": "successful",
            "data": [{"_id": "t1", "type": "debit", "amount": 150000,
                      "narration": "POS ShopRite", "date": "2026-09-01",
                      "currency": "NGN"}],
            "meta": {"total": 1, "page": 1},
        }))
        result = conn.get_transactions(
            "acc_123", start="2026-09-01", end="2026-09-30",
            type="debit", narration="shop", limit=10, page=2)
        self.assertEqual(len(result["transactions"]), 1)
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("/accounts/acc_123/transactions", url)
        self.assertIn("start=2026-09-01", url)
        self.assertIn("type=debit", url)
        self.assertIn("narration=shop", url)
        self.assertIn("page=2", url)

    def test_get_transactions_bad_type(self) -> None:
        conn, _http = _mono_connected(linked=True)
        with self.assertRaises(ConnectorError):
            conn.get_transactions("acc_123", type="sideways")

    def test_get_credits_debits(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/credits", FakeResponse(200, {
            "status": "successful", "data": [{"amount": 5}], "meta": {}}))
        http.route("GET", "/accounts/acc_123/debits", FakeResponse(200, {
            "status": "successful", "data": [{"amount": 6}], "meta": {}}))
        self.assertEqual(len(conn.get_credits("acc_123")["credits"]), 1)
        self.assertEqual(len(conn.get_debits("acc_123")["debits"]), 1)

    def test_get_identity(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/identity", FakeResponse(200, {
            "status": "successful",
            "data": {"fullName": "Adaeze Obi", "bvn": "22123456789"}}))
        ident = conn.get_identity("acc_123")
        self.assertEqual(ident["identity"]["fullName"], "Adaeze Obi")

    def test_get_income(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/income", FakeResponse(200, {
            "status": "successful",
            "data": {"type": "salary", "amount": 25000000,
                     "employer": "Acme Ltd" }}))
        income = conn.get_income("acc_123")
        self.assertEqual(income["income"]["employer"], "Acme Ltd")

    def test_get_statement_json(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/statement", FakeResponse(200, {
            "status": "successful", "data": {"entries": []}}))
        stmt = conn.get_statement("acc_123", period="last6months")
        self.assertEqual(stmt["output"], "json")
        _m, url, _p, _h = http.calls[-1]
        self.assertIn("output=json", url)
        self.assertIn("period=last6months", url)

    def test_get_statement_bad_output(self) -> None:
        conn, _http = _mono_connected(linked=True)
        with self.assertRaises(ConnectorError):
            conn.get_statement("acc_123", output="xml")

    def test_poll_statement_pdf(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/statement/job_1", FakeResponse(200, {
            "status": "successful",
            "data": {"status": "completed", "url": "https://x/y.pdf"}}))
        result = conn.poll_statement_pdf("acc_123", "job_1")
        self.assertEqual(result["status"]["status"], "completed")

    def test_sync_data(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("POST", "/sync", FakeResponse(200, {
            "status": "successful", "data": {"synced": True}}))
        result = conn.sync_data("acc_123")
        self.assertTrue(result["result"]["synced"])

    def test_reauthorise(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("POST", "/reauthorise", FakeResponse(200, {
            "status": "successful", "data": {"token": "reauth-tok"}}))
        result = conn.reauthorise("acc_123")
        self.assertEqual(result["token"], "reauth-tok")

    def test_unlink_account(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("POST", "/unlink", FakeResponse(200, {
            "status": "successful", "data": {"unlinked": True}}))
        result = conn.unlink_account("acc_123")
        self.assertTrue(result["unlinked"])
        linked = conn.vault.get("connector:mono", "mono").metadata[
            "linked_accounts"]
        self.assertEqual(linked, [])

    def test_get_institutions(self) -> None:
        conn, http = _mono_connected()
        http.routes.clear()
        http.route("GET", "/institutions", FakeResponse(200, {
            "status": "successful",
            "data": [{"name": "GTBank", "bankCode": "058"}]}))
        insts = conn.get_institutions()
        self.assertEqual(insts[0]["name"], "GTBank")


class MonoSummaryTests(unittest.TestCase):
    def test_summarize_account_converts_and_masks(self) -> None:
        conn, http = _mono_connected(linked=True)
        summary = conn.summarize_account("acc_123")
        self.assertEqual(summary["balance"], 12500.0)  # kobo -> naira
        self.assertEqual(summary["currency"], "NGN")
        self.assertEqual(summary["number"], "…6789")
        self.assertEqual(summary["institution"], "GTBank")
        self.assertNotIn("0123456789", json.dumps(summary))

    def test_summarize_identity_masks_bvn(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/identity", FakeResponse(200, {
            "status": "successful",
            "data": {"fullName": "Adaeze Obi", "bvn": "22123456789",
                     "accountNumber": "0123456789"}}))
        summary = conn.summarize_identity("acc_123")
        self.assertEqual(summary["identity"]["bvn"], "…6789")
        self.assertEqual(summary["identity"]["accountNumber"], "…6789")
        self.assertEqual(summary["identity"]["fullName"], "Adaeze Obi")

    def test_summarize_transactions_major_units(self) -> None:
        conn, http = _mono_connected(linked=True)
        http.route("GET", "/accounts/acc_123/transactions", FakeResponse(200, {
            "status": "successful",
            "data": [{"type": "debit", "amount": 150000,
                      "narration": "POS ShopRite", "date": "2026-09-01",
                      "currency": "NGN", "category": "groceries",
                      "balance": 1100000}],
            "meta": {}}))
        txns = conn.summarize_transactions("acc_123")
        self.assertEqual(txns[0]["amount"], 1500.0)
        self.assertEqual(txns[0]["balance_after"], 11000.0)
        self.assertEqual(txns[0]["narration"], "POS ShopRite")

    def test_mask_edge_cases(self) -> None:
        self.assertEqual(MonoConnector._mask(""), "")
        self.assertEqual(MonoConnector._mask("123"), "…123")
        self.assertEqual(MonoConnector._mask("0123456789"), "…6789")
        self.assertEqual(MonoConnector._major("nope", "NGN"), 0.0)
        self.assertEqual(MonoConnector._major(250, "USD"), 2.5)


class MonoErrorTests(unittest.TestCase):
    def test_401_message(self) -> None:
        conn, http = _mono_connected()
        http.routes.clear()
        http.route("GET", "/institutions", FakeResponse(401, {}))
        with self.assertRaises(MonoError) as ctx:
            conn.get_institutions()
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("secret key", str(ctx.exception))

    def test_429_message(self) -> None:
        conn, http = _mono_connected()
        http.routes.clear()
        http.route("GET", "/institutions", FakeResponse(429, {}))
        with self.assertRaises(MonoError) as ctx:
            conn.get_institutions()
        self.assertIn("rate limit", str(ctx.exception))

    def test_500_with_mono_message(self) -> None:
        conn, http = _mono_connected()
        http.routes.clear()
        http.route("GET", "/institutions", FakeResponse(500, {
            "status": "failed", "message": "boom", "code": "E1"}))
        with self.assertRaises(MonoError) as ctx:
            conn.get_institutions()
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("boom", str(ctx.exception))

    def test_invalid_json(self) -> None:
        conn, http = _mono_connected()
        http.routes.clear()
        http.route("GET", "/institutions",
                   FakeResponse(200, None, raw="<html>nope</html>"))
        with self.assertRaises(MonoError) as ctx:
            conn.get_institutions()
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_exchange_empty_code(self) -> None:
        conn, _http = _mono_connected()
        with self.assertRaises(ConnectorError):
            conn.link_account("  ")

    def test_exchange_no_id_returned(self) -> None:
        conn, http = _mono_connected()
        http.route("POST", "/accounts/auth",
                   FakeResponse(200, {"status": "successful"}))
        with self.assertRaises(MonoError) as ctx:
            conn.link_account("stale-code")
        self.assertIn("expired", str(ctx.exception))

    def test_multiple_linked_require_explicit_id(self) -> None:
        conn, http = _mono_connected(linked=True)
        meta = conn.vault.get("connector:mono", "mono").metadata
        meta["linked_accounts"].append({
            "account_id": "acc_2", "label": "second", "linked_at": 0})
        conn._save_linked(conn.vault.get("connector:mono", "mono"),
                          meta["linked_accounts"])
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_account()
        self.assertIn("explicitly", str(ctx.exception))


class MonoLinkFlowTests(unittest.TestCase):
    def _db(self) -> Database:
        return Database(":memory:")

    def test_begin_link_without_db_prints_guide(self) -> None:
        conn, _http = _mono_connected()
        with mock.patch("builtins.print") as fake_print:
            result = conn.begin_link()
        self.assertFalse(result["linked"])
        self.assertIn("link_account(code=", result["next"])
        printed = " ".join(c.args[0] for c in fake_print.call_args_list)
        self.assertIn("connect.mono.co", printed)
        self.assertIn("never", printed)  # Devon never sees bank credentials

    def test_begin_link_with_db_returns_pending(self) -> None:
        conn, _http = _mono_connected()
        db = self._db()
        with mock.patch("builtins.print"):
            result = conn.begin_link(db=db)
        self.assertFalse(result["linked"])
        cp_id = result["checkpoint_id"]
        store = CheckpointStore(db)
        cp = store.get(cp_id)
        self.assertEqual(cp.connector_id, "mono")
        self.assertEqual(cp.kind, CheckpointKind.MANUAL_STEP)
        self.assertEqual(cp.state, CheckpointState.PENDING)

    def test_resume_checkpoint_links_account(self) -> None:
        conn, http = _mono_connected()
        db = self._db()
        with mock.patch("builtins.print"):
            cp_id = conn.begin_link(db=db)["checkpoint_id"]
        http.route("POST", "/accounts/auth",
                   FakeResponse(200, {"id": "acc_123"}))
        http.route("GET", "/accounts/acc_123", FakeResponse(200, MONO_ACCOUNT))
        store = CheckpointStore(db)
        store.resolve(cp_id, note="code=widget-code-9")
        result = conn.resume_checkpoint(store.get(cp_id), db=db)
        self.assertEqual(result["account_id"], "acc_123")

    def test_resume_unresolved_checkpoint_raises(self) -> None:
        conn, _http = _mono_connected()
        db = self._db()
        store = CheckpointStore(db)
        cp = store.create("mono", CheckpointKind.MANUAL_STEP, "t", "i",
                          resume_state={"stage": "link_account"})
        with self.assertRaises(ConnectorError) as ctx:
            conn.resume_checkpoint(cp, db=db)
        self.assertIn("not resolved", str(ctx.exception))

    def test_resume_wrong_stage_raises(self) -> None:
        conn, _http = _mono_connected()
        db = self._db()
        store = CheckpointStore(db)
        cp = store.create("mono", CheckpointKind.MANUAL_STEP, "t", "i",
                          resume_state={"stage": "other"})
        store.resolve(cp.id, note="code=x")
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(store.get(cp.id), db=db)

    def test_resume_without_code_raises(self) -> None:
        conn, _http = _mono_connected()
        db = self._db()
        store = CheckpointStore(db)
        cp = store.create("mono", CheckpointKind.MANUAL_STEP, "t", "i",
                          resume_state={"stage": "link_account"})
        store.resolve(cp.id, note="done, thanks")
        with self.assertRaises(ConnectorError) as ctx:
            conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertIn("no authorization code", str(ctx.exception))

    def test_code_from_note_variants(self) -> None:
        parse = MonoConnector._code_from_note
        self.assertEqual(parse("code=abc123"), "abc123")
        self.assertEqual(parse("code: abc123"), "abc123")
        self.assertEqual(parse("abc123"), "abc123")
        self.assertEqual(parse("done, code is abc123"), "abc123")
        self.assertEqual(parse(""), "")
        self.assertEqual(parse("code"), "")

    def test_link_instructions_mentions_widget(self) -> None:
        conn, _http = _mono()
        text = conn.link_instructions()
        self.assertIn("https://app.withmono.com", text)
        self.assertIn("https://docs.mono.co", text)


# ═══════════════════════════════════════════════════════════════════════
# Plaid
# ═══════════════════════════════════════════════════════════════════════

PLAID_ACCOUNTS = {
    "accounts": [{
        "account_id": "acc_1",
        "name": "Plaid Checking",
        "mask": "0000",
        "type": "depository",
        "subtype": "checking",
        "balances": {
            "available": 1200.50, "current": 1250.00, "limit": None,
            "iso_currency_code": "USD",
        },
    }],
    "item": {"item_id": "item_1", "institution_id": "ins_1"},
}


def _plaid(http: FakeHttp | None = None) -> tuple[PlaidConnector, FakeHttp]:
    http = http or FakeHttp()
    return PlaidConnector(_vault(), http=http), http


def _route_institutions_ok(http: FakeHttp) -> None:
    http.route("POST", "/institutions/get",
               FakeResponse(200, {"institutions": [], "total": 0}))


def _plaid_connected(
    http: FakeHttp | None = None,
    *,
    with_item: bool = False,
    env: str = "sandbox",
) -> tuple[PlaidConnector, FakeHttp]:
    conn, http = _plaid(http)
    _route_institutions_ok(http)
    if with_item:
        http.route("POST", "/item/public_token/exchange", FakeResponse(200, {
            "access_token": "access-sandbox-xyz",
            "item_id": "item_1",
            "request_id": "r1",
        }))
        http.route("POST", "/accounts/get", FakeResponse(200, PLAID_ACCOUNTS))
        http.route("POST", "/institutions/get_by_id", FakeResponse(200, {
            "institution": {"institution_id": "ins_1", "name": "First Bank"}}))
        conn.connect(client_id="cid", secret="sec", env=env,
                     access_token="public-sandbox-abc")
    else:
        conn.connect(client_id="cid", secret="sec", env=env)
    return conn, http


class PlaidRegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("plaid"), PlaidConnector)

    def test_read_only_docstring(self) -> None:
        import nomorals.connectors.plaid as plaid_mod
        self.assertIn("READ-ONLY", plaid_mod.__doc__)
        self.assertIn("Read-only", PlaidConnector.__doc__)
        # no money-movement surface at all
        for name in ("transfer", "pay", "payment", "move_money",
                     "initiate_payment"):
            self.assertFalse(hasattr(PlaidConnector, name), name)


class PlaidConnectTests(unittest.TestCase):
    def test_connect_stores_client_pair(self) -> None:
        conn, http = _plaid()
        _route_institutions_ok(http)
        result = conn.connect(client_id="cid", secret="sec")
        self.assertTrue(result.ok)
        self.assertIn("no banks linked", result.account)
        cred = conn.vault.get("connector:plaid", "__client__")
        pair = json.loads(cred.password)
        self.assertEqual(pair, {"client_id": "cid", "secret": "sec"})
        # validation call carried the pair in the body
        _m, url, payload, _h = http.calls[0]
        self.assertEqual(url, "https://sandbox.plaid.com/institutions/get")
        self.assertEqual(payload["client_id"], "cid")
        self.assertEqual(payload["secret"], "sec")

    def test_connect_env_vars(self) -> None:
        conn, http = _plaid()
        _route_institutions_ok(http)
        with mock.patch.dict(os.environ, {
            "PLAID_CLIENT_ID": "env_cid", "PLAID_SECRET": "env_sec",
            "PLAID_ENV": "production",
        }):
            result = conn.connect()
        self.assertTrue(result.ok)
        _m, url, _p, _h = http.calls[0]
        self.assertTrue(url.startswith("https://production.plaid.com"))

    def test_connect_bad_env_rejected(self) -> None:
        conn, _http = _plaid()
        with self.assertRaises(ConnectorError):
            conn.connect(client_id="c", secret="s", env="moon")

    def test_connect_invalid_keys_fails_fast(self) -> None:
        conn, http = _plaid()
        http.route("POST", "/institutions/get", FakeResponse(400, {
            "error_type": "INVALID_INPUT",
            "error_code": "INVALID_API_KEYS",
            "error_message": "bad keys",
            "display_message": "Check your keys.",
        }))
        with self.assertRaises(PlaidError) as ctx:
            conn.connect(client_id="c", secret="s")
        self.assertEqual(ctx.exception.error_code, "INVALID_API_KEYS")
        self.assertIn("check the pair", str(ctx.exception).lower())
        self.assertIsNone(conn._client_credential())

    def test_connect_rejects_second_client(self) -> None:
        conn, _http = _plaid_connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(client_id="c2", secret="s2")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_with_access_token_links_item(self) -> None:
        conn, http = _plaid_connected(with_item=True)
        result_items = conn._items()
        self.assertEqual(len(result_items), 1)
        self.assertEqual(result_items[0]["institution_name"], "First Bank")
        cred = conn.vault.get("connector:plaid", "item:item_1")
        self.assertEqual(cred.password, "access-sandbox-xyz")

    def test_connect_no_tty_no_env_fails_fast(self) -> None:
        conn, _http = _plaid()
        with mock.patch.dict(os.environ, {}, clear=False):
            for var in ("PLAID_CLIENT_ID", "PLAID_SECRET"):
                os.environ.pop(var, None)
            with mock.patch("sys.stdin.isatty", return_value=False):
                with self.assertRaises(ConnectorError) as ctx:
                    conn.connect()
        self.assertIn("PLAID_CLIENT_ID", str(ctx.exception))


class PlaidStatusTests(unittest.TestCase):
    def test_status_not_connected(self) -> None:
        conn, _http = _plaid()
        st = conn.status()
        self.assertFalse(st.connected)

    def test_status_connected(self) -> None:
        conn, http = _plaid_connected(with_item=True)
        _route_institutions_ok(http)
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("First Bank", st.account)

    def test_status_rejected_client(self) -> None:
        conn, http = _plaid_connected()
        http.routes.clear()
        http.route("POST", "/institutions/get", FakeResponse(401, {
            "error_code": "INVALID_API_KEYS",
            "error_message": "bad",
        }))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)

    def test_test_connection(self) -> None:
        conn, http = _plaid_connected()
        self.assertTrue(conn.test_connection())
        http.routes.clear()
        http.route("POST", "/institutions/get",
                   FakeResponse(401, {"error_code": "x"}))
        self.assertFalse(conn.test_connection())
        conn2, _h2 = _plaid()
        self.assertFalse(conn2.test_connection())

    def test_disconnect_idempotent(self) -> None:
        conn, _http = _plaid_connected(with_item=True)
        conn.disconnect()
        conn.disconnect()
        self.assertIsNone(conn._client_credential())


class PlaidLinkFlowTests(unittest.TestCase):
    def test_create_link_token(self) -> None:
        conn, http = _plaid_connected()
        http.route("POST", "/link/token/create", FakeResponse(200, {
            "link_token": "link-sandbox-123", "expiration": "2026-10-03"}))
        token = conn.create_link_token()
        self.assertEqual(token["link_token"], "link-sandbox-123")
        body = http.last_post_body()
        self.assertEqual(body["client_name"], "Devon")
        self.assertEqual(body["products"], ["transactions"])
        self.assertEqual(body["user"], {"client_user_id": "devon-owner"})

    def test_link_page_writes_opener(self) -> None:
        conn, _http = _plaid()
        path = conn.link_page("link-sandbox-123")
        try:
            html = open(path, encoding="utf-8").read()
            self.assertIn("link-sandbox-123", html)
            self.assertIn("cdn.plaid.com", html)
            self.assertIn("public_token", html)
        finally:
            os.unlink(path)

    def test_begin_link_without_db(self) -> None:
        conn, http = _plaid_connected()
        http.route("POST", "/link/token/create", FakeResponse(200, {
            "link_token": "link-sandbox-123"}))
        with mock.patch("builtins.print"):
            result = conn.begin_link()
        self.assertFalse(result["linked"])
        self.assertEqual(result["link_token"], "link-sandbox-123")
        self.assertTrue(result["page"].endswith(".html"))
        os.unlink(result["page"])

    def test_begin_link_with_db_returns_pending(self) -> None:
        conn, http = _plaid_connected()
        http.route("POST", "/link/token/create", FakeResponse(200, {
            "link_token": "link-sandbox-123"}))
        db = Database(":memory:")
        with mock.patch("builtins.print"):
            result = conn.begin_link(db=db)
        self.assertFalse(result["linked"])
        cp_id = result["checkpoint_id"]
        store = CheckpointStore(db)
        cp = store.get(cp_id)
        self.assertEqual(cp.connector_id, "plaid")
        self.assertEqual(cp.state, CheckpointState.PENDING)

    def test_link_bank_empty_token_raises(self) -> None:
        conn, _http = _plaid_connected()
        with self.assertRaises(ConnectorError):
            conn.link_bank("  ")

    def test_resume_checkpoint_links_bank(self) -> None:
        conn, http = _plaid_connected()
        http.route("POST", "/link/token/create", FakeResponse(200, {
            "link_token": "link-sandbox-123"}))
        db = Database(":memory:")
        with mock.patch("builtins.print"):
            cp_id = conn.begin_link(db=db)["checkpoint_id"]
        http.route("POST", "/item/public_token/exchange", FakeResponse(200, {
            "access_token": "access-sandbox-xyz", "item_id": "item_1"}))
        http.route("POST", "/accounts/get", FakeResponse(200, PLAID_ACCOUNTS))
        http.route("POST", "/institutions/get_by_id", FakeResponse(200, {
            "institution": {"name": "First Bank"}}))
        store = CheckpointStore(db)
        store.resolve(cp_id, note="public_token=public-sandbox-abc")
        result = conn.resume_checkpoint(store.get(cp_id), db=db)
        self.assertEqual(result["institution_name"], "First Bank")

    def test_resume_unresolved_raises(self) -> None:
        conn, _http = _plaid_connected()
        db = Database(":memory:")
        store = CheckpointStore(db)
        cp = store.create("plaid", CheckpointKind.MANUAL_STEP, "t", "i",
                          resume_state={"stage": "link_bank"})
        with self.assertRaises(ConnectorError):
            conn.resume_checkpoint(cp, db=db)

    def test_resume_without_token_raises(self) -> None:
        conn, _http = _plaid_connected()
        db = Database(":memory:")
        store = CheckpointStore(db)
        cp = store.create("plaid", CheckpointKind.MANUAL_STEP, "t", "i",
                          resume_state={"stage": "link_bank"})
        store.resolve(cp.id, note="all done, no token here")
        with self.assertRaises(ConnectorError) as ctx:
            conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertIn("no public_token", str(ctx.exception))

    def test_token_from_note_variants(self) -> None:
        parse = PlaidConnector._token_from_note
        self.assertEqual(parse("public_token=abc"), "abc")
        self.assertEqual(parse("public-token: abc"), "abc")
        self.assertEqual(parse("abc"), "abc")
        self.assertEqual(parse("here it is: public_token abc"), "abc")
        self.assertEqual(parse(""), "")

    def test_remove_item(self) -> None:
        conn, http = _plaid_connected(with_item=True)
        http.route("POST", "/item/remove",
                   FakeResponse(200, {"removed": True}))
        result = conn.remove_item(item_id="item_1")
        self.assertTrue(result["removed"])
        self.assertEqual(result["institution_name"], "First Bank")
        with self.assertRaises(ConnectorError):
            conn._items()
        body = http.last_post_body()
        self.assertEqual(body["access_token"], "access-sandbox-xyz")

    def test_remove_item_no_match(self) -> None:
        conn, _http = _plaid_connected(with_item=True)
        with self.assertRaises(ConnectorError):
            conn.remove_item(item_id="nope")


class PlaidReadTests(unittest.TestCase):
    def _itemed(self) -> tuple[PlaidConnector, FakeHttp]:
        conn, http = _plaid_connected(with_item=True)
        http.route("POST", "/accounts/get", FakeResponse(200, PLAID_ACCOUNTS))
        return conn, http

    def test_get_accounts(self) -> None:
        conn, _http = self._itemed()
        entries = conn.get_accounts()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["institution_name"], "First Bank")
        self.assertEqual(entries[0]["accounts"][0]["mask"], "0000")

    def test_get_balances(self) -> None:
        conn, _http = self._itemed()
        balances = conn.get_balances()
        acct = balances[0]["accounts"][0]
        self.assertEqual(acct["available"], 1200.50)
        self.assertEqual(acct["current"], 1250.00)
        self.assertEqual(acct["currency"], "USD")

    def test_get_transactions_pages(self) -> None:
        conn, http = self._itemed()
        txn = {"transaction_id": "t1", "account_id": "acc_1",
               "amount": 42.5, "date": "2026-09-15", "name": "ShopRite",
               "merchant_name": "ShopRite", "pending": False,
               "iso_currency_code": "USD"}
        http.route("POST", "/transactions/get", FakeResponse(200, {
            "accounts": [], "transactions": [txn], "has_more": False,
            "total_transactions": 1}))
        entries = conn.get_transactions("2026-09-01", "2026-09-30")
        self.assertEqual(len(entries[0]["transactions"]), 1)
        body = http.last_post_body()
        self.assertEqual(body["start_date"], "2026-09-01")
        self.assertEqual(body["options"]["count"], 100)

    def test_get_transactions_bad_dates(self) -> None:
        conn, _http = self._itemed()
        with self.assertRaises(ConnectorError):
            conn.get_transactions("09/01/2026", "2026-09-30")
        with self.assertRaises(ConnectorError):
            conn.get_transactions("2026-09-30", "2026-09-01")

    def test_sync_transactions_persists_cursor(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/transactions/sync", FakeResponse(200, {
            "added": [{"transaction_id": "t1"}],
            "modified": [], "removed": [],
            "has_more": False, "next_cursor": "cursor-abc"}))
        entries = conn.sync_transactions()
        self.assertEqual(len(entries[0]["added"]), 1)
        self.assertEqual(entries[0]["cursor"], "cursor-abc")
        meta = conn.vault.get("connector:plaid", "item:item_1").metadata
        self.assertEqual(meta["cursor"], "cursor-abc")
        # second sync reuses the stored cursor
        http.route("POST", "/transactions/sync", FakeResponse(200, {
            "added": [], "modified": [], "removed": [],
            "has_more": False, "next_cursor": "cursor-abc"}))
        conn.sync_transactions()
        self.assertEqual(http.last_post_body()["cursor"], "cursor-abc")

    def test_get_recurring(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/transactions/recurring/get", FakeResponse(200, {
            "inflow_streams": [{"description": "Paycheck", "is_active": True}],
            "outflow_streams": [{"description": "Netflix",
                                 "is_active": True}]}))
        entries = conn.get_recurring()
        self.assertEqual(len(entries[0]["outflow_streams"]), 1)

    def test_get_liabilities(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/liabilities/get", FakeResponse(200, {
            "liabilities": {"credit": [{
                "last_statement_balance": 321.0,
                "next_payment_due_date": "2026-10-15"}]},
            "accounts": []}))
        entries = conn.get_liabilities()
        cards = entries[0]["liabilities"]["credit"]
        self.assertEqual(cards[0]["last_statement_balance"], 321.0)

    def test_get_investment_holdings(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/investments/holdings/get", FakeResponse(200, {
            "accounts": [], "holdings": [{"quantity": 10}],
            "securities": [{"name": "ETF"}]}))
        entries = conn.get_investment_holdings()
        self.assertEqual(len(entries[0]["holdings"]), 1)
        self.assertEqual(entries[0]["securities"][0]["name"], "ETF")

    def test_get_investment_transactions_pages_to_total(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/investments/transactions/get",
                   FakeResponse(200, {
                       "investment_transactions": [{"name": "BUY"}],
                       "total_investment_transactions": 1}))
        entries = conn.get_investment_transactions("2026-01-01",
                                                   "2026-09-30")
        self.assertEqual(len(entries[0]["investment_transactions"]), 1)

    def test_reads_require_linked_bank(self) -> None:
        conn, _http = _plaid_connected()
        with self.assertRaises(ConnectorError) as ctx:
            conn.get_accounts()
        self.assertIn("no banks linked", str(ctx.exception))

    def test_reads_require_connection(self) -> None:
        conn, _http = _plaid()
        with self.assertRaises(ConnectorError):
            conn.get_accounts()


class PlaidSummaryTests(unittest.TestCase):
    def _itemed(self) -> tuple[PlaidConnector, FakeHttp]:
        conn, http = _plaid_connected(with_item=True)
        http.route("POST", "/accounts/get", FakeResponse(200, PLAID_ACCOUNTS))
        return conn, http

    def test_summarize_accounts_plain_english(self) -> None:
        conn, _http = self._itemed()
        summary = conn.summarize_accounts()
        acct = summary[0]["accounts"][0]
        self.assertEqual(acct["account"], "Plaid Checking (…0000)")
        self.assertEqual(acct["type"], "depository (checking)")
        self.assertNotIn("acc_1", json.dumps(summary))
        self.assertNotIn("item_1", json.dumps(summary))

    def test_summarize_transactions_sign_convention(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/transactions/get", FakeResponse(200, {
            "accounts": [{"account_id": "acc_1", "name": "Plaid Checking"}],
            "transactions": [
                {"account_id": "acc_1", "amount": 42.5,
                 "date": "2026-09-15", "name": "ShopRite",
                 "merchant_name": "ShopRite", "pending": False,
                 "iso_currency_code": "USD"},
                {"account_id": "acc_1", "amount": -10.0,
                 "date": "2026-09-16", "name": "Refund", "merchant_name": "",
                 "pending": True, "iso_currency_code": "USD"},
            ],
            "has_more": False, "total_transactions": 2}))
        summary = conn.summarize_transactions("2026-09-01", "2026-09-30")
        txns = summary[0]["transactions"]
        self.assertEqual(txns[0]["direction"], "out")
        self.assertEqual(txns[1]["direction"], "in")
        self.assertTrue(txns[1]["pending"])
        self.assertEqual(txns[0]["account"], "Plaid Checking")

    def test_summarize_recurring_only_active(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/transactions/recurring/get", FakeResponse(200, {
            "inflow_streams": [],
            "outflow_streams": [
                {"description": "Netflix", "average_amount": 15.99,
                 "iso_currency_code": "USD", "frequency": "MONTHLY",
                 "last_date": "2026-09-01", "is_active": True},
                {"description": "Old gym", "average_amount": 30.0,
                 "iso_currency_code": "USD", "frequency": "MONTHLY",
                 "last_date": "2025-01-01", "is_active": False},
            ]}))
        summary = conn.summarize_recurring()
        out = summary[0]["money_out"]
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["description"], "Netflix")

    def test_summarize_liabilities(self) -> None:
        conn, http = self._itemed()
        http.route("POST", "/liabilities/get", FakeResponse(200, {
            "liabilities": {"credit": [{
                "last_statement_balance": 321.0,
                "last_statement_issue_date": "2026-09-20",
                "minimum_payment_amount": 25.0,
                "next_payment_due_date": "2026-10-15",
                "is_overdue": False}]},
            "accounts": []}))
        summary = conn.summarize_liabilities()
        card = summary[0]["credit_cards"][0]
        self.assertEqual(card["last_statement_balance"], 321.0)
        self.assertEqual(card["next_payment_due_date"], "2026-10-15")


class PlaidErrorTests(unittest.TestCase):
    def _failing(self, payload: dict, status: int = 400
                 ) -> tuple[PlaidConnector, FakeHttp]:
        conn, http = _plaid()
        http.route("POST", "/institutions/get",
                   FakeResponse(status, payload))
        return conn, http

    def test_invalid_access_token_guidance(self) -> None:
        conn, http = self._failing({
            "error_type": "ITEM_ERROR",
            "error_code": "INVALID_ACCESS_TOKEN",
            "error_message": "token bad",
            "display_message": "Reconnect.",
        })
        with self.assertRaises(PlaidError) as ctx:
            conn.connect(client_id="c", secret="s")
        self.assertEqual(ctx.exception.error_code, "INVALID_ACCESS_TOKEN")
        self.assertIn("re-link", str(ctx.exception))

    def test_item_login_required_guidance(self) -> None:
        conn, http = self._failing({
            "error_code": "ITEM_LOGIN_REQUIRED",
            "error_message": "login needed",
        })
        with self.assertRaises(PlaidError) as ctx:
            conn.connect(client_id="c", secret="s")
        self.assertIn("sign in again", str(ctx.exception))

    def test_rate_limit_guidance(self) -> None:
        conn, http = self._failing({
            "error_code": "RATE_LIMIT_EXCEEDED",
            "error_message": "slow down",
        })
        with self.assertRaises(PlaidError) as ctx:
            conn.connect(client_id="c", secret="s")
        self.assertIn("rate limit", str(ctx.exception))

    def test_non_json_error_body(self) -> None:
        conn, http = _plaid()
        http.route("POST", "/institutions/get",
                   FakeResponse(500, None, raw="gateway exploded"))
        with self.assertRaises(PlaidError) as ctx:
            conn.connect(client_id="c", secret="s")
        self.assertIn("gateway exploded", str(ctx.exception))

    def test_network_failure_wrapped(self) -> None:
        conn, _http = _plaid()

        class Boom:
            def post_json(self, *a: Any, **k: Any) -> Any:
                raise OSError("dns down")

        conn.http = Boom()  # type: ignore[assignment]
        with self.assertRaises(PlaidError) as ctx:
            conn.connect(client_id="c", secret="s")
        self.assertIn("request failed", str(ctx.exception))

    def test_sandbox_token_in_production_refused(self) -> None:
        conn, http = _plaid_connected(env="sandbox")
        # flip the stored env to production without touching the network
        cred = conn.vault.get("connector:plaid", "__client__")
        meta = dict(cred.metadata or {})
        meta["env"] = "production"
        conn._store_credential("__client__", cred.password,
                              credential_type="api_key", metadata=meta)
        with self.assertRaises(ConnectorError) as ctx:
            conn.sandbox_public_token()
        self.assertIn("sandbox", str(ctx.exception))

    def test_sandbox_public_token(self) -> None:
        conn, http = _plaid_connected()
        http.route("POST", "/sandbox/public_token/create",
                   FakeResponse(200, {
                       "public_token": "public-sandbox-1",
                       "request_id": "r"}))
        result = conn.sandbox_public_token()
        self.assertEqual(result["public_token"], "public-sandbox-1")
        body = http.last_post_body()
        self.assertEqual(body["institution_id"], "ins_109508")


if __name__ == "__main__":
    unittest.main()
