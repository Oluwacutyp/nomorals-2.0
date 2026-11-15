"""Representation ledger coverage: Gmail sends + Duffel book/cancel.

The ledger's one-line hook already covers gig applications, Twilio SMS
and social publishes. These tests pin the next three live sites: a Gmail
send, a Duffel flight booking, and a Duffel cancellation. Each must log
exactly once, with the extended action class in metadata, and a ledger
crash must never break the action it observes.

All offline — scripted HTTP, tmp mandate store, patched hook.
"""

from __future__ import annotations

import tempfile
import urllib.parse
from typing import Any
from unittest import mock

import pytest

import nomorals.cognition.representation as rep
from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.duffel import DuffelConnector
from nomorals.connectors.gmail import GmailConnector
from nomorals.finance.mandate import (
    SCOPE_TRAVEL,
    MandateStore,
    issue_mandate,
)
from nomorals.storage.db import Database

# ── fakes (same shape as test_gmail_connector / test_duffel) ───────────


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None) -> None:
        self.status = status
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        # Mirrors HttpResponse.text (core/http.py): a property, not a method.
        return str(self._payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON here")
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str,
                  payload: Any = None, **kw: Any) -> FakeResponse:
        self.calls.append((method.upper(), url, payload, kw.get("headers")))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"error": {"message": "not mocked"}})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, None, **kw)


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


TOKENS = {
    "access_token": "ya29.acc",
    "refresh_token": "1//ref",
    "expires_in": 3600,
    "token_type": "Bearer",
}
PROFILE = {"emailAddress": "owner@example.com", "messagesTotal": 42}


def _gmail_connected() -> tuple[GmailConnector, FakeHttp]:
    http = FakeHttp()
    conn = GmailConnector(_vault(), http=http)
    http.route("POST", "oauth2.googleapis.com/token",
               FakeResponse(200, dict(TOKENS)))
    http.route("GET", "/users/me/profile", FakeResponse(200, dict(PROFILE)))
    result = conn.connect(
        client_id="cid123", client_secret="<redacted>", code="auth-code-1"
    )
    assert result.ok
    return conn, http


def _duffel_connected() -> tuple[DuffelConnector, FakeHttp]:
    http = FakeHttp()
    conn = DuffelConnector(_vault(), http=http)
    http.route("GET", "/air/airlines", FakeResponse(200, {"data": []}))
    result = conn.connect(secret_key="duffel_test_abc")
    assert result.ok
    return conn, http


def _mandates() -> MandateStore:
    return MandateStore(path=tempfile.mktemp(suffix=".json"))


OFFER_JSON = {
    "id": "off_0000A3vU9nqa2l5r1cJ1",
    "total_amount": "1520.00",
    "total_currency": "USD",
    "cabin_class": "economy",
    "expires_at": "2026-12-01T12:00:00Z",
    "owner": {"name": "British Airways"},
    "passengers": [{"id": "pas_0000A3vU9nqa2l5r1cJ1"}],
    "slices": [{
        "segments": [{
            "origin": {"iata_code": "LOS"},
            "destination": {"iata_code": "LHR"},
            "departing_at": "2026-12-10T09:00:00",
            "arriving_at": "2026-12-10T14:30:00",
        }],
    }],
}

ORDER_JSON = {
    "id": "ord_0000A3vU9nqa2l5r1cJ1",
    "booking_reference": "ABC123",
    "total_amount": "1520.00",
    "total_currency": "USD",
    "created_at": "2026-11-01T10:00:00Z",
    "slices": [{
        "segments": [{
            "origin": {"iata_code": "LOS"},
            "destination": {"iata_code": "LHR"},
            "departing_at": "2026-12-10T09:00:00",
            "arriving_at": "2026-12-10T14:30:00",
        }],
    }],
}

PAX = [{
    "id": "pas_0000A3vU9nqa2l5r1cJ1",
    "given_name": "Ada",
    "family_name": "Obi",
    "born_on": "1990-05-01",
    "email": "ada@example.com",
    "phone_number": "+2348012345678",
}]


def _travel_mandate() -> MandateStore:
    store = _mandates()
    issue_mandate(store, scope=SCOPE_TRAVEL, cap_per_txn=500_000_00,
                  cap_per_day=1_000_000_00)
    return store


def _raise(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("ledger is down")


# ── Gmail send ─────────────────────────────────────────────────────────


def test_gmail_send_logs_once() -> None:
    conn, http = _gmail_connected()
    http.route("POST", "/users/me/messages/send", FakeResponse(200, {
        "id": "sent1", "threadId": "t9", "labelIds": ["SENT"],
    }))
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    with mock.patch.object(
            rep, "log_representation_action",
            lambda *a, **k: calls.append((a, k))):
        result = conn.send_message(
            "a@b.c", "Hello", "body text", confirmed=True)
    assert result["id"] == "sent1"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "message"
    assert "a@b.c" in args[1] and "Hello" in args[1]
    assert "body text" not in args[1]  # no mail body in the ledger
    assert kwargs["outcome"] == "success"
    assert kwargs["metadata"]["action_class"] == "send"
    assert kwargs["metadata"]["message_id"] == "sent1"
    assert kwargs["metadata"]["to"] == "a@b.c"


def test_gmail_send_logs_real_record(tmp_path) -> None:
    """End to end through the real hook: the row lands in the ledger DB."""
    from nomorals.cognition.ledger_ext import ExtendedRepresentationLedger
    from nomorals.cognition.representation import RepresentationLedger

    conn, http = _gmail_connected()
    http.route("POST", "/users/me/messages/send", FakeResponse(200, {
        "id": "sent1", "threadId": "t9", "labelIds": ["SENT"],
    }))
    db_path = tmp_path / "representation.db"
    ledger = ExtendedRepresentationLedger(RepresentationLedger(db_path))
    with mock.patch.object(rep, "_ledger_for", lambda settings=None: ledger):
        result = conn.send_message(
            "a@b.c", "Hello", "body text", confirmed=True)
    assert result["id"] == "sent1"
    rows = ledger.recent(action_type="message", limit=5)
    assert len(rows) == 1
    assert rows[0].outcome == "success"
    assert rows[0].metadata.get("action_class") == "send"
    assert rows[0].metadata.get("message_id") == "sent1"


def test_gmail_ledger_crash_never_breaks_send() -> None:
    conn, http = _gmail_connected()
    http.route("POST", "/users/me/messages/send", FakeResponse(200, {
        "id": "sent1", "threadId": "t9", "labelIds": ["SENT"],
    }))
    with mock.patch.object(rep, "log_representation_action", _raise):
        result = conn.send_message(
            "a@b.c", "Hello", "body text", confirmed=True)
    assert result["id"] == "sent1"


def test_gmail_failed_send_logs_nothing() -> None:
    conn, http = _gmail_connected()
    http.route("POST", "/users/me/messages/send",
               FakeResponse(500, {"error": {"message": "boom"}}))
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    with mock.patch.object(
            rep, "log_representation_action",
            lambda *a, **k: calls.append((a, k))), \
            pytest.raises(ConnectorError):
        conn.send_message("a@b.c", "Hello", "body text", confirmed=True)
    assert calls == []


# ── Duffel book ────────────────────────────────────────────────────────


def test_duffel_book_logs_once() -> None:
    c, http = _duffel_connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    http.route("POST", "/air/orders",
               FakeResponse(200, {"data": ORDER_JSON}))
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    with mock.patch.object(
            rep, "log_representation_action",
            lambda *a, **k: calls.append((a, k))):
        order = c.create_order("off_1", PAX, confirmed=True,
                               mandate_store=_travel_mandate())
    assert order.booking_reference == "ABC123"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "purchase"
    assert "LOS" in args[1] and "LHR" in args[1]
    assert "ABC123" in args[1]
    assert kwargs["outcome"] == "success"
    assert kwargs["metadata"]["action_class"] == "book"
    assert kwargs["metadata"]["order_id"] == "ord_0000A3vU9nqa2l5r1cJ1"
    assert kwargs["metadata"]["total"] == "1520.00"


def test_duffel_book_logs_real_record(tmp_path) -> None:
    """End to end through the real hook: the row lands in the ledger DB."""
    from nomorals.cognition.ledger_ext import ExtendedRepresentationLedger
    from nomorals.cognition.representation import RepresentationLedger

    c, http = _duffel_connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    http.route("POST", "/air/orders",
               FakeResponse(200, {"data": ORDER_JSON}))
    db_path = tmp_path / "representation.db"
    ledger = ExtendedRepresentationLedger(RepresentationLedger(db_path))
    with mock.patch.object(rep, "_ledger_for", lambda settings=None: ledger):
        order = c.create_order("off_1", PAX, confirmed=True,
                               mandate_store=_travel_mandate())
    assert order.booking_reference == "ABC123"
    rows = ledger.recent(action_type="purchase", limit=5)
    assert len(rows) == 1
    assert rows[0].outcome == "success"
    assert rows[0].metadata.get("action_class") == "book"


def test_duffel_ledger_crash_never_breaks_booking() -> None:
    c, http = _duffel_connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    http.route("POST", "/air/orders",
               FakeResponse(200, {"data": ORDER_JSON}))
    with mock.patch.object(rep, "log_representation_action", _raise):
        order = c.create_order("off_1", PAX, confirmed=True,
                               mandate_store=_travel_mandate())
    assert order.booking_reference == "ABC123"


def test_duffel_blocked_booking_logs_nothing() -> None:
    c, http = _duffel_connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    with mock.patch.object(
            rep, "log_representation_action",
            lambda *a, **k: calls.append((a, k))), \
            pytest.raises(ConnectorError):
        # No mandate issued: the mandate gate refuses before ticketing.
        c.create_order("off_1", PAX, confirmed=True,
                       mandate_store=_mandates())
    assert calls == []


# ── Duffel cancel ──────────────────────────────────────────────────────


def test_duffel_cancel_logs_once() -> None:
    c, http = _duffel_connected()
    http.route("POST", "/air/order_cancellations",
               FakeResponse(200, {"data": {"id": "oec_1"}}))
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    with mock.patch.object(
            rep, "log_representation_action",
            lambda *a, **k: calls.append((a, k))):
        result = c.cancel_order("ord_1", confirmed=True)
    assert result["id"] == "oec_1"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "purchase"
    assert "ord_1" in args[1]
    assert kwargs["outcome"] == "success"
    assert kwargs["metadata"]["action_class"] == "cancel"
    assert kwargs["metadata"]["order_id"] == "ord_1"


def test_duffel_cancel_ledger_crash_never_breaks_cancel() -> None:
    c, http = _duffel_connected()
    http.route("POST", "/air/order_cancellations",
               FakeResponse(200, {"data": {"id": "oec_1"}}))
    with mock.patch.object(rep, "log_representation_action", _raise):
        result = c.cancel_order("ord_1", confirmed=True)
    assert result["id"] == "oec_1"
