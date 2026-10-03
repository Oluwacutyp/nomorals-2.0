"""Wave L2: Virtual Cards connector (Flutterwave primary).

Offline by design — the Flutterwave API is faked at the HttpClient
boundary, credentials live in an in-memory vault, checkpoints use an
in-memory database. No network, no secrets.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import (
    AuthMethod,
    CheckpointKind,
    CheckpointStore,
    ConnectorError,
    HumanCheckpointPending,
    create_connector,
)
from nomorals.connectors.virtualcards import (
    FlutterwaveProvider,
    VirtualCardProvider,
    VirtualCardsConnector,
    VirtualCardsError,
    mask_card,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _db() -> Database:
    return Database(":memory:")


# ── fake HTTP ────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._payload = payload if payload is not None else {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return json.dumps(self._payload)

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. Records calls, returns routes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: dict[tuple[str, str], FakeResponse] = {}

    def route(
        self, method: str, path: str, response: FakeResponse
    ) -> None:
        self.routes[(method.upper(), path)] = response

    def _call(self, method: str, url: str, payload: Any, kw: Any):
        self.calls.append((method, url, payload, kw))
        path = url.split("api.flutterwave.com")[1].split("?")[0]
        key = (method, path)
        if key not in self.routes:
            raise AssertionError(f"no route for {method} {path}")
        return self.routes[key]

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._call("GET", url, None, kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._call("POST", url, payload, kw)

    def put_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._call("PUT", url, payload, kw)

    def request(self, *a: Any, **kw: Any) -> FakeResponse:
        raise AssertionError("unexpected raw request")


def _flw_envelope(data: Any, message: str = "ok") -> FakeResponse:
    return FakeResponse(
        200, {"status": "success", "message": message, "data": data}
    )


_CARD_ID = "c6d7f40b-f772-47b7-8136-81256d2f87a2"
_FULL_PAN = "5531886652142950"


def _card_obj() -> dict[str, Any]:
    # Flutterwave's real card shape (per the official SDK sample
    # responses): PAN in card_pan, expiry as expiration "YYYY-MM".
    return {
        "id": _CARD_ID,
        "card_pan": _FULL_PAN,
        "masked_pan": "5531********2950",
        "cvv": "564",
        "expiration": "2029-09",
        "card_type": "MASTERCARD",
        "name_on_card": "Jermaine Graham",
        "currency": "USD",
        "amount": 200,
        "is_active": True,
    }


def _flw() -> FakeHttp:
    """FakeHttp pre-routed for the happy-path card lifecycle."""
    http = FakeHttp()
    http.route("GET", "/v3/virtual-cards", _flw_envelope([]))
    http.route(
        "POST", "/v3/virtual-cards", _flw_envelope(_card_obj(), "created")
    )
    http.route(
        "GET", f"/v3/virtual-cards/{_CARD_ID}", _flw_envelope(_card_obj())
    )
    http.route(
        "POST",
        f"/v3/virtual-cards/{_CARD_ID}/fund",
        _flw_envelope({"balance": 4000}, "funded"),
    )
    http.route(
        "POST",
        f"/v3/virtual-cards/{_CARD_ID}/withdraw",
        _flw_envelope({"balance": 3000}, "withdrawn"),
    )
    http.route(
        "PUT",
        f"/v3/virtual-cards/{_CARD_ID}/status/block",
        _flw_envelope({"is_active": False}, "blocked"),
    )
    http.route(
        "PUT",
        f"/v3/virtual-cards/{_CARD_ID}/status/unblock",
        _flw_envelope({"is_active": True}, "unblocked"),
    )
    http.route(
        "PUT",
        f"/v3/virtual-cards/{_CARD_ID}/terminate",
        _flw_envelope({"is_active": False}, "terminated"),
    )
    http.route(
        "GET",
        f"/v3/virtual-cards/{_CARD_ID}/transactions",
        _flw_envelope(
            {"transactions": [{"id": "tx1", "amount": 50}]}, "fetched"
        ),
    )
    return http


def _vcards(http: FakeHttp | None = None) -> VirtualCardsConnector:
    http = http or FakeHttp()
    return VirtualCardsConnector(_vault(), http=http)


_CREATE_KWARGS = {
    "currency": "USD",
    "amount": 200,
    "debit_currency": "NGN",
    "billing_name": "Jermaine Graham",
    "billing_address": "2014 Forest Hills Drive",
    "billing_city": "Lagos",
    "billing_state": "Lagos",
    "billing_postal_code": "100001",
    "billing_country": "NG",
    "first_name": "Jermaine",
    "last_name": "Graham",
    "date_of_birth": "1990/01/15",
    "email": "jermaine@example.com",
    "phone": "+2348012345678",
    "title": "Mr",
    "gender": "M",
}


# ── registration / declaration ───────────────────────────────────────


class RegistrationTests(unittest.TestCase):
    def test_registered_under_id(self) -> None:
        conn = create_connector("virtualcards", _vault())
        self.assertIsInstance(conn, VirtualCardsConnector)

    def test_declared_shape(self) -> None:
        self.assertEqual(VirtualCardsConnector.id, "virtualcards")
        self.assertEqual(
            VirtualCardsConnector.auth_methods, (AuthMethod.API_KEY,)
        )
        self.assertEqual(VirtualCardsConnector.PROVISIONABLE, ("virtual_card",))
        self.assertTrue(VirtualCardsConnector(_vault()).can_provision(
            "virtual_card"))
        self.assertFalse(
            VirtualCardsConnector(_vault()).can_provision("repo")
        )


# ── masking ──────────────────────────────────────────────────────────


class MaskingTests(unittest.TestCase):
    def test_full_pan_reduced_to_last4(self) -> None:
        masked = mask_card({"card_pan": _FULL_PAN, "cvv": "564"})
        self.assertEqual(masked["card_pan"], "**** **** **** 2950")
        self.assertNotIn(_FULL_PAN, masked["card_pan"])
        self.assertEqual(masked["cvv"], "***")

    def test_legacy_card_number_field_still_masked(self) -> None:
        masked = mask_card({"card_number": _FULL_PAN})
        self.assertEqual(masked["card_number"], "**** **** **** 2950")

    def test_masked_pan_stays_masked(self) -> None:
        masked = mask_card({"card_pan": "5531********2950"})
        self.assertIn("2950", masked["card_pan"])
        self.assertNotIn("55318866", masked["card_pan"])

    def test_non_card_shapes_pass_through(self) -> None:
        card = {"id": "abc", "currency": "USD", "is_active": True}
        self.assertEqual(mask_card(card), card)

    def test_missing_fields_ok(self) -> None:
        self.assertEqual(mask_card({}), {})
        self.assertEqual(mask_card({"card_pan": None})["card_pan"], None)

    def test_input_not_mutated(self) -> None:
        card = {"card_pan": _FULL_PAN, "cvv": "564"}
        mask_card(card)
        self.assertEqual(card["card_pan"], _FULL_PAN)


# ── connect lifecycle ────────────────────────────────────────────────


class ConnectTests(unittest.TestCase):
    def test_connect_validates_and_vault_stores(self) -> None:
        http = _flw()
        conn = _vcards(http)
        with mock.patch.dict(os.environ, {"FLW_SECRET_KEY": "FLWSECK_TEST-x"}):
            result = conn.connect()
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "flutterwave")
        cred = conn.vault.get("connector:virtualcards", "flutterwave")
        self.assertEqual(cred.password, "FLWSECK_TEST-x")
        # Authorization header carried the key
        method, url, _, kw = http.calls[0]
        self.assertEqual(method, "GET")
        self.assertIn("Bearer FLWSECK_TEST-x",
                      kw["headers"]["Authorization"])

    def test_connect_explicit_arg_beats_env(self) -> None:
        conn = _vcards(_flw())
        with mock.patch.dict(os.environ, {"FLW_SECRET_KEY": "env-key"}):
            conn.connect(api_key="FLWSECK_TEST-arg")
        cred = conn.vault.get("connector:virtualcards", "flutterwave")
        self.assertEqual(cred.password, "FLWSECK_TEST-arg")

    def test_connect_rejects_bad_key_fast(self) -> None:
        http = FakeHttp()
        http.route(
            "GET",
            "/v3/virtual-cards",
            FakeResponse(
                401,
                {"status": "error", "message": "Invalid API key"},
            ),
        )
        conn = _vcards(http)
        with self.assertRaises(VirtualCardsError) as ctx:
            conn.connect(api_key="FLWSECK_TEST-bad")
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(
            conn.vault.list_all(service="connector:virtualcards"), []
        )

    def test_connect_fails_fast_without_key_non_tty(self) -> None:
        conn = _vcards(FakeHttp())
        env = {k: v for k, v in os.environ.items() if k != "FLW_SECRET_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConnectorError):
                conn.connect()

    def test_disconnect_is_idempotent(self) -> None:
        conn = _vcards(_flw())
        conn.connect(api_key="FLWSECK_TEST-x")
        conn.disconnect()
        conn.disconnect()  # no raise
        self.assertFalse(conn.test_connection())

    def test_status_connected_and_detail(self) -> None:
        conn = _vcards(_flw())
        conn.connect(api_key="FLWSECK_TEST-x")
        status = conn.status()
        self.assertTrue(status.connected)
        self.assertEqual(status.account, "flutterwave")
        self.assertIn("valid", status.detail)
        self.assertTrue(conn.test_connection())

    def test_status_not_connected(self) -> None:
        status = _vcards(FakeHttp()).status()
        self.assertFalse(status.connected)
        self.assertIn("not connected", status.detail)

    def test_status_key_rejected(self) -> None:
        http = _flw()
        conn = _vcards(http)
        conn.connect(api_key="FLWSECK_TEST-x")
        http.route(
            "GET",
            "/v3/virtual-cards",
            FakeResponse(
                401, {"status": "error", "message": "revoked"}
            ),
        )
        status = conn.status()
        self.assertFalse(status.connected)
        self.assertIn("rejected", status.detail)

    def test_unknown_provider_refused(self) -> None:
        conn = _vcards(_flw())
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(api_key="k", provider="nope")
        self.assertIn("unknown virtual-card provider", str(ctx.exception))


# ── card operations ──────────────────────────────────────────────────


class CardOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.http = _flw()
        self.conn = _vcards(self.http)
        self.conn.connect(api_key="FLWSECK_TEST-x")

    def test_create_card_returns_masked_and_vaults_secrets(self) -> None:
        masked = self.conn.create_card(**_CREATE_KWARGS)
        self.assertEqual(masked["card_pan"], "**** **** **** 2950")
        self.assertEqual(masked["cvv"], "***")
        self.assertNotIn(_FULL_PAN, json.dumps(masked))

        # Full secrets live in the vault under card:<id>
        cred = self.conn.vault.get(
            "connector:virtualcards", f"card:{_CARD_ID}"
        )
        details = json.loads(cred.password)
        self.assertEqual(details["card_pan"], _FULL_PAN)
        self.assertEqual(details["cvv"], "564")
        self.assertEqual(details["expiration"], "2029-09")

        # Payload went to POST /v3/virtual-cards with create fields
        method, url, payload, _ = self.http.calls[1]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/v3/virtual-cards"))
        self.assertEqual(payload["currency"], "USD")
        self.assertEqual(payload["amount"], 200)
        # debit_currency is a real create field (official SDK payload)
        self.assertEqual(payload["debit_currency"], "NGN")

    def test_create_card_strips_unknown_fields(self) -> None:
        kwargs = dict(_CREATE_KWARGS, evil="drop me")
        self.conn.create_card(**kwargs)
        _, _, payload, _ = self.http.calls[1]
        self.assertNotIn("evil", payload)

    def test_create_card_validates_inputs(self) -> None:
        for bad in (
            dict(_CREATE_KWARGS, currency=""),
            dict(_CREATE_KWARGS, amount=0),
            dict(_CREATE_KWARGS, billing_name=""),
        ):
            with self.assertRaises(ConnectorError):
                self.conn.create_card(**bad)

    def test_list_cards_masked(self) -> None:
        self.http.route(
            "GET",
            "/v3/virtual-cards",
            _flw_envelope([_card_obj(), _card_obj()]),
        )
        cards = self.conn.list_cards(per_page=5)
        self.assertEqual(len(cards), 2)
        for card in cards:
            self.assertEqual(card["card_pan"], "**** **** **** 2950")
            self.assertNotIn(_FULL_PAN, json.dumps(card))

    def test_get_card_masked(self) -> None:
        card = self.conn.get_card(_CARD_ID)
        self.assertEqual(card["card_pan"], "**** **** **** 2950")
        self.assertEqual(card["cvv"], "***")
        self.assertNotIn(_FULL_PAN, json.dumps(card))

    def test_fund_card_payload(self) -> None:
        result = self.conn.fund_card(_CARD_ID, 4000, debit_currency="NGN")
        self.assertEqual(result["balance"], 4000)
        method, url, payload, _ = [
            c for c in self.http.calls if c[0] == "POST"
            and c[1].endswith("/fund")
        ][0]
        self.assertEqual(payload, {"debit_currency": "NGN", "amount": 4000.0})

    def test_withdraw_card_payload(self) -> None:
        result = self.conn.withdraw_card(_CARD_ID, 1000)
        self.assertEqual(result["balance"], 3000)
        _, url, payload, _ = [
            c for c in self.http.calls if c[1].endswith("/withdraw")
        ][0]
        self.assertEqual(payload, {"amount": 1000.0})

    def test_amounts_must_be_positive_numbers(self) -> None:
        for amount in (0, -5, "lots"):
            with self.assertRaises(ConnectorError):
                self.conn.fund_card(_CARD_ID, amount)
            with self.assertRaises(ConnectorError):
                self.conn.withdraw_card(_CARD_ID, amount)

    def test_block_and_unblock(self) -> None:
        self.conn.block_card(_CARD_ID)
        _, url, payload, _ = [
            c for c in self.http.calls if "/status/block" in c[1]
        ][0]
        self.assertTrue(url.endswith(f"/virtual-cards/{_CARD_ID}/status/block"))
        self.assertEqual(payload, {"status_action": "block"})
        self.conn.unblock_card(_CARD_ID)
        _, url2, payload2, _ = [
            c for c in self.http.calls if "/status/unblock" in c[1]
        ][0]
        self.assertTrue(
            url2.endswith(f"/virtual-cards/{_CARD_ID}/status/unblock"))
        self.assertEqual(payload2, {"status_action": "unblock"})
        with self.assertRaises(VirtualCardsError):
            self.conn._require_provider().block_card(_CARD_ID, "freeze")

    def test_terminate_drops_vault_secrets(self) -> None:
        self.conn.create_card(**_CREATE_KWARGS)
        self.assertTrue(
            self.conn.vault.list_all(service="connector:virtualcards")
        )
        self.conn.terminate_card(_CARD_ID)
        with self.assertRaises(Exception):
            self.conn.vault.get(
                "connector:virtualcards", f"card:{_CARD_ID}"
            )
        with self.assertRaises(ConnectorError):
            self.conn.reveal_card(_CARD_ID)

    def test_list_transactions(self) -> None:
        result = self.conn.list_transactions(
            _CARD_ID, from_date="2026-01-01", to_date="2026-10-01",
            index=0, size=5,
        )
        self.assertEqual(result["transactions"][0]["id"], "tx1")
        method, url, _, kw = [
            c for c in self.http.calls if "/transactions" in c[1]
        ][0]
        params = kw.get("params", {})
        self.assertEqual(params["from"], "2026-01-01")
        self.assertEqual(params["to"], "2026-10-01")
        self.assertEqual(params["size"], 5)

    def test_reveal_card_from_vault(self) -> None:
        self.conn.create_card(**_CREATE_KWARGS)
        details = self.conn.reveal_card(_CARD_ID)
        self.assertEqual(details["card_pan"], _FULL_PAN)
        self.assertEqual(details["cvv"], "564")

    def test_reveal_card_unknown_fails_fast(self) -> None:
        with self.assertRaises(ConnectorError) as ctx:
            self.conn.reveal_card("no-such-card")
        self.assertIn("not created through this connector",
                      str(ctx.exception))

    def test_ops_require_connection(self) -> None:
        conn = _vcards(_flw())
        with self.assertRaises(ConnectorError):
            conn.list_cards()

    def test_error_envelope_surfaces_message(self) -> None:
        http = FakeHttp()
        http.route("GET", "/v3/virtual-cards", _flw_envelope([]))
        http.route(
            "POST",
            "/v3/virtual-cards",
            FakeResponse(
                400,
                {"status": "error", "message": "amount below minimum"},
            ),
        )
        conn = _vcards(http)
        conn.connect(api_key="FLWSECK_TEST-x")
        with self.assertRaises(VirtualCardsError) as ctx:
            conn.create_card(**_CREATE_KWARGS)
        self.assertIn("amount below minimum", str(ctx.exception))


# ── provisioning + human-in-the-loop ─────────────────────────────────


class ProvisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.http = _flw()
        self.conn = _vcards(self.http)
        self.conn.connect(api_key="FLWSECK_TEST-x")

    def test_provision_virtual_card_routes_to_create(self) -> None:
        masked = self.conn.provision("virtual_card", **_CREATE_KWARGS)
        self.assertEqual(masked["card_pan"], "**** **** **** 2950")

    def test_provision_unknown_kind_refused(self) -> None:
        with self.assertRaises(ConnectorError) as ctx:
            self.conn.provision("bank_account")
        self.assertIn("cannot provision", str(ctx.exception))

    def test_confirm_without_db_fails_fast(self) -> None:
        with self.assertRaises(ConnectorError) as ctx:
            self.conn.provision("virtual_card", confirm=True, **_CREATE_KWARGS)
        self.assertIn("needs a database", str(ctx.exception))
        with self.assertRaises(ConnectorError):
            self.conn.fund_card(_CARD_ID, 100, confirm=True)

    def test_confirm_pauses_at_human_checkpoint(self) -> None:
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            self.conn.provision(
                "virtual_card", confirm=True, db=db, **_CREATE_KWARGS
            )
        cp = ctx.exception.checkpoint
        self.assertIn("USD 200", cp.title)
        # Nothing was created while paused
        self.assertEqual(
            [c for c in self.http.calls if c[0] == "POST"
             and c[1].endswith("/v3/virtual-cards")],
            [],
        )
        # Owner approves -> resume completes the create
        store = CheckpointStore(db)
        store.resolve(cp.id, note="owner approved")
        masked = self.conn.resume_checkpoint(
            store.get(cp.id), db=db, context=None
        )
        self.assertEqual(masked["card_pan"], "**** **** **** 2950")

    def test_fund_confirm_pauses_then_resumes(self) -> None:
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            self.conn.fund_card(
                _CARD_ID, 4000, debit_currency="NGN",
                confirm=True, db=db,
            )
        cp = ctx.exception.checkpoint
        self.assertIn("NGN 4000", cp.title)
        store = CheckpointStore(db)
        store.resolve(cp.id, note="approved")
        result = self.conn.resume_checkpoint(
            store.get(cp.id), db=db, context=None
        )
        self.assertEqual(result["balance"], 4000)

    def test_resume_unresolved_checkpoint_refused(self) -> None:
        db = _db()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            self.conn.provision(
                "virtual_card", confirm=True, db=db, **_CREATE_KWARGS
            )
        store = CheckpointStore(db)
        with self.assertRaises(ConnectorError) as err:
            self.conn.resume_checkpoint(
                store.get(ctx.exception.checkpoint.id), db=db, context=None
            )
        self.assertIn("not resolved", str(err.exception))

    def test_resume_unknown_intent_refused(self) -> None:
        db = _db()
        store = CheckpointStore(db)
        cp = store.create(
            "virtualcards", CheckpointKind.MANUAL_STEP, "t", "i",
            resume_state={"intent": "nope"},
        )
        store.resolve(cp.id, note="approved")
        with self.assertRaises(ConnectorError) as ctx:
            self.conn.resume_checkpoint(
                store.get(cp.id), db=db, context=None
            )
        self.assertIn("cannot resume checkpoint intent", str(ctx.exception))


# ── provider ABC slot-in ─────────────────────────────────────────────


class ProviderAbcTests(unittest.TestCase):
    def test_abc_requires_name(self) -> None:
        class Nameless(VirtualCardProvider):
            name = ""

            def create_card(self, payload): ...
            def list_cards(self, *, per_page=20): ...
            def get_card(self, card_id): ...
            def fund_card(self, card_id, amount, *, debit_currency): ...
            def withdraw_card(self, card_id, amount): ...
            def terminate_card(self, card_id): ...
            def block_card(self, card_id, action): ...
            def list_transactions(self, card_id, **kw): ...

        with self.assertRaises(VirtualCardsError):
            Nameless(FakeHttp(), "key")

    def test_second_provider_slots_into_registry(self) -> None:
        class DemoProvider(FlutterwaveProvider):
            name = "demo"

        original = dict(VirtualCardsConnector._PROVIDERS)
        VirtualCardsConnector._PROVIDERS["demo"] = DemoProvider
        try:
            self.assertIs(
                VirtualCardsConnector._PROVIDERS["demo"], DemoProvider
            )
            conn = _vcards(_flw())
            res = conn.connect(api_key="k", provider="demo")
            self.assertEqual(res.account, "demo")
        finally:
            VirtualCardsConnector._PROVIDERS.clear()
            VirtualCardsConnector._PROVIDERS.update(original)

    def test_provider_needs_api_key(self) -> None:
        with self.assertRaises(VirtualCardsError):
            FlutterwaveProvider(FakeHttp(), "")


if __name__ == "__main__":
    unittest.main()
