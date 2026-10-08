"""Tests for the Duffel flight connector. All offline — scripted HTTP."""

from __future__ import annotations

import json
import tempfile
from typing import Any

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.duffel import (
    DuffelConnector,
    DuffelError,
    Offer,
    Order,
)
from nomorals.connectors.registry import get_connector
from nomorals.finance.mandate import (
    MandateStore,
    issue_mandate,
    SCOPE_TRAVEL,
)
from nomorals.storage.db import Database


# ── fakes ──────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None) -> None:
        self.status = status
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON here")
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str,
                  payload: Any = None, **kw: Any) -> FakeResponse:
        self.calls.append((method.upper(), url, payload))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"errors": [{"message": "not mocked"}]})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _duffel(http: FakeHttp | None = None) -> tuple[DuffelConnector, FakeHttp]:
    http = http or FakeHttp()
    return DuffelConnector(_vault(), http=http), http


def _connected(http: FakeHttp | None = None,
               key: str = "duffel_test_abc") -> tuple[DuffelConnector, FakeHttp]:
    conn, http = _duffel(http)
    http.route("GET", "/air/airlines", FakeResponse(200, {"data": []}))
    result = conn.connect(secret_key=key)
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


# ── registration ───────────────────────────────────────────────────────


def test_registered():
    assert get_connector("duffel") is DuffelConnector


# ── search ─────────────────────────────────────────────────────────────


def test_search_offers_sorted_by_price():
    c, http = _connected()
    cheap = dict(OFFER_JSON, id="off_cheap", total_amount="900.00",
                 owner={"name": "Virgin"})
    pricey = dict(OFFER_JSON, id="off_pricey", total_amount="1500.00")
    http.route("POST", "/air/offer_requests", FakeResponse(200, {
        "data": {"offers": [pricey, cheap]}}))
    offers = c.search_offers("LOS", "LHR", "2026-12-10")
    assert [o.id for o in offers] == ["off_cheap", "off_pricey"]
    assert offers[0].origin == "LOS" and offers[0].destination == "LHR"
    assert offers[0].airline == "Virgin"
    assert offers[0].total == "USD 900.00"
    # Duffel-Version header is required
    assert any("duffel.com/air/offer_requests" in url
               for _, url, _ in http.calls)


def test_search_offers_bad_date():
    c, _ = _duffel()
    try:
        c.search_offers("LOS", "LHR", "10-12-2026")
        assert False, "should raise"
    except DuffelError as e:
        assert "YYYY-MM-DD" in str(e)


def test_search_offers_api_error():
    c, http = _connected()
    http.route("POST", "/air/offer_requests", FakeResponse(422, {
        "errors": [{"message": "origin is not a valid IATA code"}]}))
    try:
        c.search_offers("XXX", "YYY", "2026-12-10")
        assert False, "should raise"
    except DuffelError as e:
        assert "not a valid IATA code" in str(e)


def test_get_offer_reprice():
    c, http = _connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    offer = c.get_offer("off_1")
    assert isinstance(offer, Offer)
    assert offer.total == "USD 1520.00"
    assert offer.passenger_ids == ["pas_0000A3vU9nqa2l5r1cJ1"]


# ── booking: confirmation gate ─────────────────────────────────────────


def test_create_order_needs_confirmation():
    c, http = _connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    try:
        c.create_order("off_1", PAX)  # not confirmed, no db
        assert False, "should raise"
    except ConnectorError as e:
        assert "explicit owner confirmation" in str(e)
    # No order POST happened
    assert not any(url.endswith("/air/orders")
                   for _, url, _ in http.calls)


def test_create_order_confirmed():
    c, http = _connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    http.route("POST", "/air/orders",
               FakeResponse(200, {"data": ORDER_JSON}))
    store = _mandates()
    issue_mandate(store, scope=SCOPE_TRAVEL, cap_per_txn=500_000_00,
                  cap_per_day=1_000_000_00)
    order = c.create_order("off_1", PAX, confirmed=True,
                           mandate_store=store)
    assert isinstance(order, Order)
    assert order.booking_reference == "ABC123"
    # Payment is balance, amount matches the offer exactly
    posts = [p for m, u, p in http.calls
             if m == "POST" and u.endswith("/air/orders")]
    assert len(posts) == 1
    payments = posts[0]["data"]["payments"]
    assert payments == [{"type": "balance", "amount": "1520.00",
                         "currency": "USD"}]
    assert posts[0]["data"]["type"] == "instant"


def test_create_order_reprices_before_booking():
    """A stale price is never booked — get_offer runs first."""
    c, http = _connected()
    repriced = dict(OFFER_JSON, total_amount="1600.00")
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": repriced}))
    http.route("POST", "/air/orders",
               FakeResponse(200, {"data": ORDER_JSON}))
    store = _mandates()
    issue_mandate(store, scope=SCOPE_TRAVEL, cap_per_txn=500_000_00,
                  cap_per_day=1_000_000_00)
    c.create_order("off_1", PAX, confirmed=True, mandate_store=store)
    posts = [p for m, u, p in http.calls
             if m == "POST" and u.endswith("/air/orders")]
    assert posts[0]["data"]["payments"][0]["amount"] == "1600.00"


# ── booking: mandate gate ──────────────────────────────────────────────


def test_create_order_blocked_without_mandate():
    c, http = _connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    store = _mandates()  # empty — no mandate
    try:
        c.create_order("off_1", PAX, confirmed=True, mandate_store=store)
        assert False, "should raise"
    except ConnectorError as e:
        assert "mandate" in str(e).lower()
    assert not any(url.endswith("/air/orders")
                   for _, url, _ in http.calls)


def test_create_order_blocked_over_cap():
    c, http = _connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    store = _mandates()
    issue_mandate(store, scope=SCOPE_TRAVEL, cap_per_txn=100_00,  # $1 cap
                  cap_per_day=1_000_000_00)
    try:
        c.create_order("off_1", PAX, confirmed=True, mandate_store=store)
        assert False, "should raise"
    except ConnectorError as e:
        assert "mandate" in str(e).lower()


def test_transfer_mandate_does_not_cover_travel():
    """A transfer-scope mandate must not authorize flight bookings."""
    c, http = _connected()
    http.route("GET", "/air/offers/off_1",
               FakeResponse(200, {"data": OFFER_JSON}))
    store = _mandates()
    issue_mandate(store, scope="transfer", cap_per_txn=500_000_00,
                  cap_per_day=1_000_000_00)
    try:
        c.create_order("off_1", PAX, confirmed=True, mandate_store=store)
        assert False, "should raise"
    except ConnectorError as e:
        assert "mandate" in str(e).lower()


# ── itinerary checkpoint text ──────────────────────────────────────────


def test_format_itinerary():
    c, _ = _duffel()
    text = c.format_itinerary(_parse_offer_for_test())
    assert "LOS → LHR" in text
    assert "USD 1520.00" in text
    assert "Duffel organisation balance" in text


def _parse_offer_for_test() -> Offer:
    from nomorals.connectors.duffel import _parse_offer
    return _parse_offer(OFFER_JSON)


# ── cancel / change ────────────────────────────────────────────────────


def test_cancel_order_needs_confirmation():
    c, http = _duffel()
    try:
        c.cancel_order("ord_1")
        assert False, "should raise"
    except ConnectorError as e:
        assert "explicit owner confirmation" in str(e)
    assert not any("order_cancellations" in u for _, u, _ in http.calls)


def test_cancel_order_confirmed():
    c, http = _connected()
    http.route("POST", "/air/order_cancellations",
               FakeResponse(200, {"data": {"id": "oec_1"}}))
    result = c.cancel_order("ord_1", confirmed=True)
    assert result["id"] == "oec_1"


def test_request_order_change_confirmed():
    c, http = _connected()
    http.route("POST", "/air/order_changes",
               FakeResponse(200, {"data": {"id": "och_1"}}))
    result = c.request_order_change(
        "ord_1", [{"origin": "LOS", "destination": "JFK",
                   "departure_date": "2026-12-11"}],
        confirmed=True)
    assert result["id"] == "och_1"


# ── two-path dispatch ──────────────────────────────────────────────────


def test_book_flight_returns_duffel_offers():
    c, http = _connected()
    http.route("POST", "/air/offer_requests", FakeResponse(200, {
        "data": {"offers": [OFFER_JSON]}}))
    result = c.book_flight("LOS", "LHR", "2026-12-10", PAX)
    assert result["path"] == "duffel"
    assert len(result["offers"]) == 1
    assert result["offers"][0]["offer_id"].startswith("off_")
    assert "create_order" in result["next"]


def test_book_flight_falls_back_when_no_offers():
    c, http = _connected()
    http.route("POST", "/air/offer_requests",
               FakeResponse(200, {"data": {"offers": []}}))
    result = c.book_flight("LOS", "XYZ", "2026-12-10", PAX)
    assert result["path"] == "browser"
    assert "not in Duffel" in result["reason"]
    assert result["task"]["kind"] == "flight_booking"
    assert result["task"]["origin"] == "LOS"
    # The fallback never enters payment on its own
    assert "payment" in result["task"]["instructions"].lower()


def test_book_flight_falls_back_when_api_down():
    c, http = _duffel()
    http.route("POST", "/air/offer_requests",
               FakeResponse(500, {"errors": [{"message": "boom"}]}))
    result = c.book_flight("LOS", "LHR", "2026-12-10", PAX)
    assert result["path"] == "browser"
    assert "Duffel error" in result["reason"]


# ── fail-closed ────────────────────────────────────────────────────────


def test_fail_closed_without_connection():
    c, _ = _duffel()
    try:
        c.search_offers("LOS", "LHR", "2026-12-10")
        assert False, "should raise"
    except DuffelError as e:
        assert "not connected" in str(e)


def test_401_is_clear():
    c, http = _duffel()
    # connect a key first, then have the API reject it
    http.route("GET", "/air/airlines", FakeResponse(401, {
        "errors": [{"message": "Invalid API key"}]}))
    c._store_credential("duffel", "duffel_test_bad",
                        credential_type="api_key")
    assert c.test_connection() is False
    status = c.status()
    assert status.connected is False
    assert "rejected" in status.detail


def test_connect_rejects_bad_key():
    c, http = _duffel()
    http.route("GET", "/air/airlines", FakeResponse(401, {
        "errors": [{"message": "Invalid API key"}]}))
    try:
        c.connect(secret_key="duffel_test_bad")
        assert False, "should raise"
    except ConnectorError as e:
        assert "401" in str(e)
