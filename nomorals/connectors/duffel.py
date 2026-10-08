"""Duffel connector — flight search, booking, and ticketing.

Docs: https://duffel.com/docs/api/v2/offer-requests

Auth: one API key (dashboard: https://app.duffel.com -> Settings ->
Access tokens; ``duffel_test_*`` for the sandbox, ``duffel_live_*`` for
production) in the ``Authorization: Bearer`` header plus
``Duffel-Version: v2``, ``AuthMethod.API_KEY``.

Two-path booking model:

* **API path (Duffel)** — one API, NDC/GDS/LCC inventory, PNR creation.
  ``search_offers`` → ``get_offer`` (live re-price) → ``create_order``
  (PNR + ticket). This is the fast path and is tried first.
* **Browser path (airline-direct)** — when Duffel has no inventory for a
  route, or the API is unreachable, ``book_flight`` returns an honest
  ``browser_fallback`` spec describing exactly what the parent should do
  in the live browser. Nothing is faked; the connector never pretends a
  browser booking happened.

Money safety (mirrors the Paystack architecture):

* ``create_order`` moves real money and refuses to run without explicit
  owner confirmation (``confirmed=True`` after the owner approved the
  exact itinerary + price, or a human checkpoint when ``db`` is given).
  The confirmation screen always shows the itinerary and total first —
  "permission checkpoints before payment".
* The agent's payment mandate (#69) is checked in the dispatch path —
  no active mandate (or expired/revoked/over cap) → ``ConnectorError``,
  no ticket is issued. Structural, not advisory.
* The explicit-override rule holds: an explicit "book it" (after the
  itinerary was shown) executes via ``confirmed=True``; the gradient
  governs autonomous/ambiguous action.

Payment: orders are created with ``type: "instant"`` and a
``"balance"`` payment drawn from the Duffel organisation balance —
the org must be funded (dashboard -> Payments) or order creation
fails with a Duffel payments error. This is stated plainly in the
confirmation text so the owner is never surprised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..core.logging_setup import get_logger
from ..finance.mandate import MandateError, require_mandate, SCOPE_TRAVEL
from ._confirm import confirm_or_checkpoint
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["DuffelConnector", "DuffelError", "Offer", "Order"]

_log = get_logger(__name__)

API_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"
SECRET_ENV = "DUFFEL_API_KEY"
DOCS_URL = "https://duffel.com/docs/api/v2/offer-requests"


class DuffelError(ConnectorError):
    """A Duffel API call failed."""


@dataclass
class Offer:
    """One bookable flight offer from Duffel."""

    id: str
    total_amount: str  # e.g. "1520.00" — always a decimal string
    total_currency: str  # e.g. "USD", "NGN"
    airline: str = ""
    origin: str = ""
    destination: str = ""
    departure: str = ""  # ISO local time of first segment
    arrival: str = ""  # ISO local time of last segment
    stops: int = 0
    cabin_class: str = ""
    expires_at: str = ""
    passenger_ids: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> str:
        return f"{self.total_currency} {self.total_amount}"


@dataclass
class Order:
    """A ticketed order (PNR) from Duffel."""

    id: str
    booking_reference: str = ""
    total_amount: str = ""
    total_currency: str = ""
    origin: str = ""
    destination: str = ""
    created_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> str:
        return f"{self.total_currency} {self.total_amount}"


def _parse_offer(data: dict[str, Any]) -> Offer:
    slices = data.get("slices") or []
    first, last = {}, {}
    stops = 0
    if slices:
        segs0 = slices[0].get("segments") or []
        if segs0:
            first = segs0[0]
            stops += max(0, len(segs0) - 1)
        segs_last = (slices[-1].get("segments") or [])
        if segs_last:
            last = segs_last[-1]
            stops += max(0, len(segs_last) - 1) if len(slices) > 1 else 0
    owner = data.get("owner") or {}
    return Offer(
        id=data.get("id", ""),
        total_amount=str(data.get("total_amount", "0")),
        total_currency=str(data.get("total_currency", "")),
        airline=str(owner.get("name", "")),
        origin=str((first.get("origin") or {}).get("iata_code", "")),
        destination=str((last.get("destination") or {}).get("iata_code", "")),
        departure=str(first.get("departing_at", "")),
        arrival=str(last.get("arriving_at", "")),
        stops=stops,
        cabin_class=str(data.get("cabin_class", "")),
        expires_at=str(data.get("expires_at", "")),
        passenger_ids=[str(p.get("id", "")) for p in (data.get("passengers") or [])],
        raw=data,
    )


def _parse_order(data: dict[str, Any]) -> Order:
    slices = data.get("slices") or []
    segs0 = (slices[0].get("segments") or []) if slices else []
    segs_last = (slices[-1].get("segments") or []) if slices else []
    first, last = (segs0[0] if segs0 else {}), (segs_last[-1] if segs_last else {})
    return Order(
        id=data.get("id", ""),
        booking_reference=str(data.get("booking_reference", "")),
        total_amount=str(data.get("total_amount", "0")),
        total_currency=str(data.get("total_currency", "")),
        origin=str((first.get("origin") or {}).get("iata_code", "")),
        destination=str((last.get("destination") or {}).get("iata_code", "")),
        created_at=str(data.get("created_at", "")),
        raw=data,
    )


@register_connector
class DuffelConnector(Connector):
    """Devon's Duffel adapter: flight search, booking, changes, cancels."""

    id = "duffel"
    name = "Duffel"
    description = (
        "Flight booking: search offers across 300+ airlines, live re-price, "
        "create orders (PNR/ticketing), changes and cancellations. Order "
        "creation is confirmation-gated and mandate-checked; the two-path "
        "model falls back to airline-direct browser booking when Duffel has "
        "no inventory. Authenticates with a Duffel API key (Bearer <redacted>)."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        secret_key: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate an API key and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "duffel is already connected — one account per service. "
                "Disconnect first to switch API keys."
            )
        secret = (secret_key or "").strip() or prompt_secret(
            "Duffel API key", env_var=SECRET_ENV
        )
        if not secret:
            raise ConnectorError("empty API key: nothing to connect with")
        mode = "live" if secret.startswith("duffel_live_") else "test"
        self._validate_key(secret)
        self._store_credential(
            "duffel",
            secret,
            credential_type="api_key",
            scopes=["offer_requests", "offers", "orders"],
            metadata={"mode": mode},
        )
        _log.info("duffel connected (%s mode)", mode)
        return ConnectResult(
            ok=True,
            account=f"duffel ({mode})",
            scopes=["offer_requests", "offers", "orders"],
            message=(
                f"connected to Duffel in {mode} mode. The API key is in the "
                "encrypted vault. search_offers() finds flights; "
                "create_order() needs explicit owner confirmation every time "
                "plus an active travel mandate, and charges the Duffel "
                "organisation balance."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name duffel`",
            )
        try:
            self._api("GET", "/air/airlines", params={"limit": 1})
        except DuffelError as exc:
            return ConnectorStatus(
                connected=False,
                account="duffel",
                scopes=list((cred.metadata or {}).get("scopes", [])),
                detail=f"API key rejected ({exc}): reconnect with a fresh "
                       "key from the Duffel dashboard",
            )
        mode = (cred.metadata or {}).get("mode", "?")
        return ConnectorStatus(
            connected=True,
            account=f"duffel ({mode})",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            detail="API key valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/air/airlines", params={"limit": 1})
            return True
        except ConnectorError:
            return False

    # ── flight search ──────────────────────────────────────────

    def search_offers(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        *,
        return_date: str | None = None,
        passengers: int = 1,
        cabin_class: str = "economy",
        max_connections: int | None = None,
    ) -> list[Offer]:
        """Search flights. Returns offers sorted by total price.

        ``departure_date``/``return_date`` are ``YYYY-MM-DD``. Raises
        :class:`DuffelError` when the API refuses or is unreachable —
        the caller decides whether to fall back to the browser path.
        """
        origin = (origin or "").strip().upper()
        destination = (destination or "").strip().upper()
        if not origin or not destination:
            raise DuffelError("origin and destination IATA codes are required")
        for label, value in (("departure_date", departure_date),
                             ("return_date", return_date)):
            if value:
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    raise DuffelError(
                        f"{label} {value!r} must be YYYY-MM-DD"
                    ) from None
        if passengers < 1:
            raise DuffelError("passengers must be at least 1")
        slices = [{"origin": origin, "destination": destination,
                   "departure_date": departure_date}]
        if return_date:
            slices.append({"origin": destination, "destination": origin,
                           "departure_date": return_date})
        payload: dict[str, Any] = {
            "data": {
                "slices": slices,
                "passengers": [{"type": "adult"} for _ in range(passengers)],
                "cabin_class": cabin_class,
                "return_offers": True,
            }
        }
        if max_connections is not None:
            payload["data"]["max_connections"] = max_connections
        body = self._api("POST", "/air/offer_requests", payload=payload)
        offers = [ _parse_offer(o) for o in (body.get("data", {}).get("offers") or []) ]
        offers.sort(key=lambda o: float(o.total_amount or 0))
        return offers

    def get_offer(self, offer_id: str) -> Offer:
        """Live re-price of one offer. Always call right before booking —
        prices move and offers expire (~30 min)."""
        offer_id = (offer_id or "").strip()
        if not offer_id:
            raise DuffelError("empty offer_id")
        body = self._api("GET", f"/air/offers/{offer_id}")
        return _parse_offer(body.get("data", {}))

    # ── booking (money moves — confirmation-gated) ─────────────

    def format_itinerary(self, offer: Offer) -> str:
        """The permission-checkpoint text: itinerary + price, shown to the
        owner BEFORE any confirmation is requested."""
        lines = [
            "✈️ Flight booking — review before confirming:",
            f"  {offer.origin} → {offer.destination}"
            + (f" ({offer.airline})" if offer.airline else ""),
            f"  Departs: {offer.departure or '?'}",
            f"  Arrives: {offer.arrival or '?'}",
            f"  Stops: {offer.stops}   Cabin: {offer.cabin_class or 'economy'}",
            f"  **Total: {offer.total}**",
        ]
        if offer.expires_at:
            lines.append(f"  Offer expires: {offer.expires_at}")
        lines.append(
            "Payment comes from the Duffel organisation balance "
            "(fund it in the Duffel dashboard first)."
        )
        return "\n".join(lines)

    def create_order(
        self,
        offer_id: str,
        passengers: list[dict[str, Any]],
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
        mandate_store: Any = None,
        principal: str = "owner",
        ledger: Any = None,
    ) -> Order:
        """Create an order (PNR) and ticket it. Moves real money.

        ``passengers`` must carry the ``id`` from the selected offer's
        passengers plus ``given_name``, ``family_name``, ``born_on``
        (YYYY-MM-DD), ``email``, ``phone_number``. Same confirmation
        architecture as Paystack: pass ``confirmed=True`` only after the
        owner approved the exact itinerary + price (an explicit "book it"),
        or pass ``db`` to park the exact booking on a human checkpoint.

        The agent's travel mandate (#69) is checked in the dispatch path —
        no mandate → ``MandateError``, no ticket.
        """
        offer_id = (offer_id or "").strip()
        if not offer_id:
            raise ConnectorError("empty offer_id")
        if not passengers:
            raise ConnectorError("at least one passenger is required")
        # Live re-price first: never book a stale price.
        offer = self.get_offer(offer_id)
        try:
            amount = float(offer.total_amount)
        except (TypeError, ValueError):
            raise ConnectorError(
                f"could not parse offer total {offer.total_amount!r} — "
                "re-run search_offers and try again"
            ) from None
        if amount <= 0:
            raise ConnectorError(f"offer total {offer.total!r} is not payable")
        payload: dict[str, Any] = {
            "offer_id": offer_id,
            "passengers": passengers,
            "total": offer.total,
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="booking",
            title=f"Book flight {offer.origin} → {offer.destination} "
                  f"({offer.total})",
            instructions="\n".join([
                "Devon wants to book and ticket a flight via Duffel.",
                "This moves real money from the Duffel organisation balance.",
                self.format_itinerary(offer),
                "Approve ONLY if the itinerary and price above are correct.",
            ]),
            resume_state={"payload": payload},
        )
        try:
            require_mandate(mandate_store, principal, SCOPE_TRAVEL,
                            int(amount * 100), ledger=ledger)
        except MandateError as exc:
            raise ConnectorError(f"mandate blocked: {exc}") from exc
        body = self._api("POST", "/air/orders", payload={
            "data": {
                "selected_offers": [offer_id],
                "passengers": passengers,
                "type": "instant",
                "payments": [{
                    "type": "balance",
                    "amount": offer.total_amount,
                    "currency": offer.total_currency,
                }],
            }
        })
        return _parse_order(body.get("data", {}))

    def cancel_order(
        self,
        order_id: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Cancel an order. Confirmation-gated (refunds/fees are money)."""
        order_id = (order_id or "").strip()
        if not order_id:
            raise ConnectorError("empty order_id")
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="cancel",
            title=f"Cancel flight order {order_id}",
            instructions="\n".join([
                "Devon wants to cancel a flight order via Duffel.",
                f"Order: {order_id}",
                "Cancellation terms (fees, refundability) come from the "
                "fare rules — review them before approving.",
            ]),
            resume_state={"order_id": order_id},
        )
        body = self._api("POST", "/air/order_cancellations",
                         payload={"data": {"order_id": order_id}})
        return body.get("data", {})

    def request_order_change(
        self,
        order_id: str,
        slices: list[dict[str, str]],
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Request a change quote for an order. Confirmation-gated."""
        order_id = (order_id or "").strip()
        if not order_id:
            raise ConnectorError("empty order_id")
        if not slices:
            raise ConnectorError("at least one slice is required")
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="change",
            title=f"Change flight order {order_id}",
            instructions="\n".join([
                "Devon wants to change a flight order via Duffel.",
                f"Order: {order_id}",
                f"New slices: {slices}",
                "Changes may add fees — review the quote before approving.",
            ]),
            resume_state={"order_id": order_id, "slices": slices},
        )
        body = self._api("POST", "/air/order_changes",
                         payload={"data": {"order_id": order_id,
                                           "slices": slices}})
        return body.get("data", {})

    # ── two-path dispatch ──────────────────────────────────────

    def book_flight(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        passengers: list[dict[str, Any]],
        *,
        return_date: str | None = None,
        cabin_class: str = "economy",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
        mandate_store: Any = None,
        principal: str = "owner",
        ledger: Any = None,
    ) -> dict[str, Any]:
        """Two-path booking: Duffel API first, airline-direct fallback.

        Tries the Duffel API; when it has no inventory for the route or
        is unreachable, returns an honest ``browser_fallback`` spec that
        the caller hands to the live browser — never a fake booking.
        When Duffel HAS offers, returns them for the owner to pick from;
        the actual ticketing still goes through :meth:`create_order` with
        its confirmation gate and mandate check.
        """
        search: dict[str, Any] = {
            "origin": (origin or "").strip().upper(),
            "destination": (destination or "").strip().upper(),
            "departure_date": departure_date,
            "return_date": return_date,
            "cabin_class": cabin_class,
            "passengers": passengers,
        }
        try:
            offers = self.search_offers(
                search["origin"], search["destination"], departure_date,
                return_date=return_date, passengers=max(1, len(passengers)),
                cabin_class=cabin_class,
            )
        except DuffelError as exc:
            _log.info("duffel search failed (%s) — browser fallback", exc)
            return self._browser_fallback(search, f"Duffel error: {exc}")
        if not offers:
            return self._browser_fallback(
                search, "Duffel returned no offers for this route")
        return {
            "path": "duffel",
            "offers": [
                {
                    "offer_id": o.id,
                    "airline": o.airline,
                    "route": f"{o.origin} → {o.destination}",
                    "departure": o.departure,
                    "arrival": o.arrival,
                    "stops": o.stops,
                    "cabin_class": o.cabin_class,
                    "total": o.total,
                    "expires_at": o.expires_at,
                }
                for o in offers[:10]
            ],
            "next": ("pick an offer_id and call create_order() with the "
                     "passenger details — confirmation gate + travel mandate "
                     "apply"),
        }

    def _browser_fallback(
        self, search: dict[str, Any], reason: str
    ) -> dict[str, Any]:
        """Honest fallback spec — the parent delegates this to the live
        browser (airline-direct booking). Never fakes a booking."""
        return {
            "path": "browser",
            "reason": f"not in Duffel — trying airline-direct ({reason})",
            "task": {
                "kind": "flight_booking",
                "origin": search["origin"],
                "destination": search["destination"],
                "departure_date": search["departure_date"],
                "return_date": search.get("return_date"),
                "cabin_class": search.get("cabin_class", "economy"),
                "passenger_count": len(search.get("passengers") or []),
                "instructions": (
                    "Book this flight directly on an airline's website in "
                    "the live browser. Search the route and dates, pick the "
                    "best option, and STOP at the payment step — never enter "
                    "payment details without the owner's explicit approval. "
                    "Report the itinerary and price back for confirmation."
                ),
            },
        }

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
        mandate_store: Any = None,
        principal: str = "owner",
        ledger: Any = None,
    ) -> dict[str, Any]:
        """Continue a booking flow after its human checkpoint resolved."""
        from .checkpoints import CheckpointState
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must approve first"
            )
        state = dict(checkpoint.resume_state or {})
        stage = state.get("stage")
        if stage == "booking":
            payload = state.get("payload", {})
            return self.create_order(
                payload["offer_id"], payload["passengers"],
                confirmed=True, mandate_store=mandate_store,
                principal=principal, ledger=ledger,
            ).raw
        if stage == "cancel":
            return self.cancel_order(
                state["order_id"], confirmed=True)
        if stage == "change":
            return self.request_order_change(
                state["order_id"], state["slices"], confirmed=True)
        raise ConnectorError(
            f"duffel cannot resume checkpoint stage {stage!r}"
        )

    # ── transport ──────────────────────────────────────────────

    def _require_credential(self):  # type: ignore[no-untyped-def]
        cred = self._load_credential()
        if cred is None:
            raise DuffelError(
                "duffel is not connected — run `nm connectors connect "
                "--name duffel` first"
            )
        return cred

    def _validate_key(self, secret: str) -> None:
        try:
            resp = self.http.get(
                f"{API_BASE}/air/airlines",
                headers=self._headers(secret),
                params={"limit": 1},
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise ConnectorError(f"duffel key validation failed: {exc}") from exc
        if resp.status == 401:
            raise ConnectorError(
                "duffel rejected the API key (401): it is invalid or "
                "revoked — copy a fresh one from the Duffel dashboard "
                "(Settings -> Access tokens)"
            )
        if not resp.ok:
            raise ConnectorError(
                f"duffel key validation failed ({resp.status})"
            )

    @staticmethod
    def _headers(secret: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {secret}",
            "Duffel-Version": DUFFEL_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        secret: str | None = None,
    ) -> dict[str, Any]:
        """One Duffel API call; failures become DuffelError."""
        key = secret or self._require_credential().password
        url = f"{API_BASE}{path}"
        headers = self._headers(key)
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise DuffelError(f"duffel request failed: {exc}") from exc
        if resp.status == 401:
            raise DuffelError(
                "duffel rejected the API key (401): it is invalid or "
                "revoked — copy a fresh one from the Duffel dashboard",
            )
        if resp.status == 429:
            raise DuffelError(
                "duffel rate limit exceeded (429) — wait and retry"
            )
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - invalid JSON is an error
            raise DuffelError(
                f"duffel {method} {path} returned invalid JSON"
            ) from None
        if not resp.ok:
            raise DuffelError(
                f"duffel {method} {path} failed ({resp.status}): "
                f"{self._error_text(body)}"
            )
        if not isinstance(body, dict):
            raise DuffelError(
                f"duffel {method} {path} returned an unexpected body"
            )
        return body

    @staticmethod
    def _error_text(body: Any) -> str:
        if isinstance(body, dict):
            errors = body.get("errors")
            if isinstance(errors, list) and errors:
                first = errors[0]
                if isinstance(first, dict):
                    return str(first.get("message") or first.get("title")
                               or first)
                return str(first)
            return str(body.get("message") or body)[:300]
        return str(body)[:300]
