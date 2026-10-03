"""Binance connector — crypto spot trading.

Docs: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md

Auth: API key + secret (``AuthMethod.API_KEY``), HMAC-SHA256 request
signing. Every signed call carries ``timestamp`` (ms) + ``recvWindow``
(5000); the signature covers the full query string and rides in the
``X-MBX-APIKEY`` header + ``signature`` param. Market data (ticker price)
is public and needs no key.

REAL MONEY: ``place_order`` and ``cancel_order`` never run on implied
consent — they require ``confirmed=True`` (owner approved the exact
order) or a human checkpoint when ``db`` is given. ``testnet=True`` on
connect points at https://testnet.binance.vision for dry runs — use it
before trusting any order flow.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from typing import Any

from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .checkpoints import CheckpointState
from .registry import register_connector

__all__ = ["BinanceConnector", "BinanceError"]

_log = get_logger(__name__)

API_BASE = "https://api.binance.com"
TESTNET_BASE = "https://testnet.binance.vision"
API_KEY_ENV = "BINANCE_API_KEY"
API_SECRET_ENV = "BINANCE_API_SECRET"

_RECV_WINDOW = 5000

_ORDER_SIDES = ("BUY", "SELL")
_ORDER_TYPES = (
    "LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT",
    "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER",
)


class BinanceError(ConnectorError):
    """A Binance API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        binance_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.binance_code = binance_code


@register_connector
class BinanceConnector(Connector):
    """Devon's Binance spot adapter: prices, balances, orders."""

    id = "binance"
    name = "Binance"
    description = (
        "Binance spot trading: public ticker prices, account balances, "
        "open orders, and placing/cancelling orders (HMAC-SHA256 signed; "
        "orders require explicit owner confirmation). Testnet supported."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a key pair against /api/v3/account and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "binance is already connected — one account per service. "
                "Disconnect first to switch API keys."
            )
        key = (api_key or "").strip() or prompt_secret(
            "Binance API key", env_var=API_KEY_ENV
        )
        secret = (api_secret or "").strip() or prompt_secret(
            "Binance API secret", env_var=API_SECRET_ENV
        )
        if not key or not secret:
            raise ConnectorError(
                "empty API key/secret: nothing to connect with"
            )
        base = TESTNET_BASE if testnet else API_BASE
        account = self._signed(
            "GET", "/api/v3/account", {}, key=key, secret=secret,
            base_url=base,
        )
        can_trade = bool(account.get("canTrade", False))
        self._store_credential(
            "binance",
            secret,
            credential_type="api_key",
            scopes=["spot:read", "spot:trade"],
            metadata={
                "api_key": key,
                "base_url": base,
                "testnet": testnet,
                "can_trade": can_trade,
            },
        )
        _log.info(
            "binance connected (%s, canTrade=%s)",
            "testnet" if testnet else "live", can_trade,
        )
        return ConnectResult(
            ok=True,
            account=f"binance ({'testnet' if testnet else 'live'})",
            scopes=["spot:read", "spot:trade"],
            message=(
                f"connected to Binance {'testnet' if testnet else 'LIVE'} "
                f"(spot trading permission: {can_trade}). The secret is in "
                "the encrypted vault. Every order still needs your explicit "
                "confirmation at call time."
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
                       "--name binance`",
            )
        meta = cred.metadata or {}
        try:
            self._signed("GET", "/api/v3/account", {})
        except BinanceError as exc:
            return ConnectorStatus(
                connected=False,
                account="binance",
                scopes=list(meta.get("scopes", [])),
                last_checked=time.time(),
                detail=f"key rejected ({exc}): check IP restrictions and "
                       "reconnect with a fresh pair",
            )
        mode = "testnet" if meta.get("testnet") else "live"
        return ConnectorStatus(
            connected=True,
            account=f"binance ({mode})",
            scopes=list(meta.get("scopes", [])),
            last_checked=time.time(),
            detail=f"keys valid; spot trading enabled: "
                   f"{meta.get('can_trade', '?')}",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._signed("GET", "/api/v3/account", {})
            return True
        except ConnectorError:
            return False

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Execute a confirmed order after the owner resolved it."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must approve the order first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        payload = dict((checkpoint.resume_state or {}).get("payload", {}))
        if stage == "place_order":
            if not payload.get("symbol"):
                raise ConnectorError(
                    "the resolved checkpoint has no order payload — "
                    "it cannot place the order"
                )
            return self._place_now(payload)
        if stage == "cancel_order":
            if not payload.get("symbol"):
                raise ConnectorError(
                    "the resolved checkpoint has no cancel payload"
                )
            return self._cancel_now(payload)
        raise ConnectorError(
            f"binance cannot resume checkpoint stage {stage!r}"
        )

    # ── market data (public) ─────────────────────────────────────

    def get_price(self, symbol: str = "") -> Any:
        """Latest price(s) (``GET /api/v3/ticker/price``).

        No key needed. ``symbol="BTCUSDT"`` for one price; empty for all.
        """
        params = {"symbol": symbol.upper()} if symbol else {}
        data = self._public("GET", "/api/v3/ticker/price", params=params)
        return data

    def get_ticker_24h(self, symbol: str = "") -> Any:
        """24h rolling stats (``GET /api/v3/ticker/24hr``). Public."""
        params = {"symbol": symbol.upper()} if symbol else {}
        return self._public("GET", "/api/v3/ticker/24hr", params=params)

    # ── account ──────────────────────────────────────────────────

    def get_balances(self, *, nonzero: bool = True) -> list[dict[str, Any]]:
        """Spot balances (``GET /api/v3/account``).

        ``nonzero`` drops dust (free + locked == 0).
        """
        account = self._signed("GET", "/api/v3/account", {})
        balances = account.get("balances", [])
        out = []
        for entry in balances:
            free = float(entry.get("free", 0) or 0)
            locked = float(entry.get("locked", 0) or 0)
            if nonzero and free + locked <= 0:
                continue
            out.append({
                "asset": entry.get("asset", ""),
                "free": free,
                "locked": locked,
                "total": free + locked,
            })
        return out

    def list_open_orders(self, symbol: str = "") -> list[dict[str, Any]]:
        """Open orders (``GET /api/v3/openOrders``), optionally per symbol."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = self._signed("GET", "/api/v3/openOrders", params)
        return data if isinstance(data, list) else []

    def get_order(
        self, symbol: str, *, order_id: int = 0, client_order_id: str = ""
    ) -> dict[str, Any]:
        """One order's status (``GET /api/v3/order``)."""
        if not symbol:
            raise ConnectorError("symbol is required")
        if not order_id and not client_order_id:
            raise ConnectorError(
                "pass order_id or client_order_id to identify the order"
            )
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        data = self._signed("GET", "/api/v3/order", params)
        return data if isinstance(data, dict) else {}

    # ── orders (confirmation-gated real money) ────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        *,
        price: float = 0.0,
        time_in_force: str = "GTC",
        stop_price: float = 0.0,
        client_order_id: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Place a spot order (``POST /api/v3/order``). REAL MONEY.

        Never runs on implied consent: pass ``confirmed=True`` only after
        the owner approved the exact symbol/side/type/quantity/price — or
        pass ``db`` to park the exact order on a human checkpoint.
        LIMIT orders need ``price``; STOP_LOSS/TAKE_PROFIT need
        ``stop_price``.
        """
        symbol = (symbol or "").upper()
        side = (side or "").upper()
        order_type = (order_type or "").upper()
        if not symbol:
            raise ConnectorError("symbol is required (e.g. BTCUSDT)")
        if side not in _ORDER_SIDES:
            raise ConnectorError(
                f"invalid side {side!r}: use BUY or SELL"
            )
        if order_type not in _ORDER_TYPES:
            raise ConnectorError(
                f"invalid order type {order_type!r}: use one of "
                f"{', '.join(_ORDER_TYPES)}"
            )
        if quantity <= 0:
            raise ConnectorError(
                f"invalid quantity {quantity}: must be positive"
            )
        if order_type in ("LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT",
                         "LIMIT_MAKER") and price <= 0:
            raise ConnectorError(
                f"{order_type} needs a positive price"
            )
        if order_type in ("STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT",
                          "TAKE_PROFIT_LIMIT") and stop_price <= 0:
            raise ConnectorError(
                f"{order_type} needs a positive stop_price"
            )
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price,
            "time_in_force": time_in_force,
            "stop_price": stop_price,
            "client_order_id": client_order_id,
        }
        price_str = f"{price:g}" if price else "MARKET"
        summary = f"{side} {quantity:g} {symbol} @ {price_str} ({order_type})"
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="place_order",
            title=f"Place Binance order: {summary}",
            instructions="\n".join([
                "Devon wants to place this REAL-MONEY spot order.",
                "Review it — once placed it can fill immediately.",
                f"Order: {summary}",
                f"Stop price: {stop_price:g}" if stop_price else "",
                f"Client order id: {client_order_id}" if client_order_id else "",
                f"Account: {self._mode_label()}",
            ]),
            resume_state={"payload": payload},
        )
        return self._place_now(payload)

    def _place_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": payload["symbol"],
            "side": payload["side"],
            "type": payload["type"],
            "quantity": repr(float(payload["quantity"])),
        }
        order_type = payload["type"]
        if order_type in ("LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT",
                          "LIMIT_MAKER"):
            params["price"] = repr(float(payload["price"]))
            if order_type != "LIMIT_MAKER":
                params["timeInForce"] = payload.get("time_in_force", "GTC")
        if payload.get("stop_price"):
            params["stopPrice"] = repr(float(payload["stop_price"]))
        if payload.get("client_order_id"):
            params["newClientOrderId"] = payload["client_order_id"]
        data = self._signed("POST", "/api/v3/order", params)
        result = data if isinstance(data, dict) else {}
        _log.info(
            "binance order placed: %s %s %s (id %s, status %s)",
            payload["side"], payload["quantity"], payload["symbol"],
            result.get("orderId", "?"), result.get("status", "?"),
        )
        return result

    def cancel_order(
        self,
        symbol: str,
        *,
        order_id: int = 0,
        client_order_id: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Cancel an open order (``DELETE /api/v3/order``).

        Confirmation-gated like placement: cancelling changes a live
        position's protection, so the owner approves each one.
        """
        symbol = (symbol or "").upper()
        if not symbol:
            raise ConnectorError("symbol is required")
        if not order_id and not client_order_id:
            raise ConnectorError(
                "pass order_id or client_order_id to identify the order"
            )
        payload = {
            "symbol": symbol,
            "order_id": order_id,
            "client_order_id": client_order_id,
        }
        ident = f"order {order_id}" if order_id else client_order_id
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="cancel_order",
            title=f"Cancel Binance order {ident} on {symbol}",
            instructions="\n".join([
                "Devon wants to cancel this open order.",
                f"Symbol: {symbol}",
                f"Order: {ident}",
                f"Account: {self._mode_label()}",
            ]),
            resume_state={"payload": payload},
        )
        return self._cancel_now(payload)

    def _cancel_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": payload["symbol"]}
        if payload.get("order_id"):
            params["orderId"] = payload["order_id"]
        if payload.get("client_order_id"):
            params["origClientOrderId"] = payload["client_order_id"]
        data = self._signed("DELETE", "/api/v3/order", params)
        result = data if isinstance(data, dict) else {}
        _log.info(
            "binance order cancelled: %s (status %s)",
            result.get("orderId", "?"), result.get("status", "?"),
        )
        return result

    # ── signing + HTTP ───────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "binance is not connected — run "
                "`nm connectors connect --name binance` first"
            )
        return cred

    def _mode_label(self) -> str:
        cred = self._load_credential()
        meta = (cred.metadata or {}) if cred else {}
        return "testnet" if meta.get("testnet") else "live"

    def _sign(self, secret: str, query: str) -> str:
        return hmac.new(
            secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _signed(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
        *,
        key: str | None = None,
        secret: str | None = None,
        base_url: str | None = None,
    ) -> Any:
        """A signed Binance call: timestamp + recvWindow + HMAC signature."""
        if key and secret and base_url:
            # Explicit credentials (the connect() validation call): the
            # vault has nothing stored yet, so don't require it.
            api_key, api_secret, base = key, secret, base_url
        else:
            cred = self._require_credential()
            meta = cred.metadata or {}
            api_key = key or str(meta.get("api_key", ""))
            api_secret = secret or cred.password
            base = base_url or str(meta.get("base_url", API_BASE))
        all_params = dict(params)
        all_params["recvWindow"] = _RECV_WINDOW
        all_params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(all_params)
        signature = self._sign(api_secret, query)
        url = f"{base}{path}?{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": api_key}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers)
            elif method == "POST":
                resp = self.http.request("POST", url, headers=headers)
            elif method == "DELETE":
                resp = self.http.request("DELETE", url, headers=headers)
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise BinanceError(f"binance request failed: {exc}") from exc
        return self._result(method, path, resp)

    def _public(
        self, method: str, path: str, params: dict[str, Any]
    ) -> Any:
        """An unsigned (public market-data) call — no key needed."""
        cred = self._load_credential()
        base = str((cred.metadata or {}).get("base_url", API_BASE)) \
            if cred else API_BASE
        url = f"{base}{path}"
        try:
            if method == "GET":
                resp = self.http.get(url, params=params)
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise BinanceError(f"binance request failed: {exc}") from exc
        return self._result(method, path, resp)

    def _result(self, method: str, path: str, resp: Any) -> Any:
        if resp.status == 401:
            raise BinanceError(
                "binance rejected the API key (401): it is invalid or has "
                "IP restrictions — check the API management page and "
                "reconnect",
                status_code=401,
            )
        if resp.status == 429 or resp.status == 418:
            raise BinanceError(
                f"binance rate limited ({resp.status}) — back off; "
                "repeated 418s mean a temporary IP ban",
                status_code=resp.status,
            )
        if not resp.ok:
            code, msg = self._error_detail(resp)
            raise BinanceError(
                f"binance {method} {path} failed ({resp.status}): {msg}",
                status_code=resp.status,
                binance_code=code,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise BinanceError(
                f"binance {method} {path} returned invalid JSON"
            ) from exc

    @staticmethod
    def _error_detail(resp: Any) -> tuple[int, str]:
        try:
            body = resp.json()
            if isinstance(body, dict):
                try:
                    code = int(body.get("code", 0))
                except (TypeError, ValueError):
                    code = 0
                return code, str(body.get("msg", body))[:200]
        except Exception:  # noqa: BLE001 - fall back to raw text
            pass
        return 0, resp.text[:200]
