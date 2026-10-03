"""Coinbase connector — Advanced Trade API (spot trading).

Docs: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis

Auth: CDP API key (``AuthMethod.API_KEY``) — the key name (a UUID or
``organizations/.../apiKeys/...``) plus the EC private key PEM Coinbase
hands you at creation time. Every request carries a short-lived (120s)
ES256 JWT in the ``Authorization`` header. Claims per the Advanced Trade
scheme::

    header:  {"alg": "ES256", "kid": <key name>, "nonce": <random hex>}
    payload: {"sub": <key name>, "iss": "coinbase-cloud",
              "nbf": now, "exp": now + 120,
              "aud": ["retail_rest_api_proxy"],
              "uri": "METHOD api.coinbase.com/path"}

The JWT is signed with a hand-rolled, stdlib-only P-256 ECDSA
implementation (``_es256.py`` math lives inline here) — no third-party
crypto dependency, which keeps the connector runnable on the phone/Termux
and minimal CI images. The PEM may be SEC1 (``BEGIN EC PRIVATE KEY``) or
PKCS8 (``BEGIN PRIVATE KEY``). A malformed key fails fast at connect
time with a clear message; ``connect()`` also sign-checks the key
against its own public point before the first API call.

REAL MONEY: ``place_order`` and ``cancel_order`` never run on implied
consent — they require ``confirmed=True`` (owner approved the exact
order) or a human checkpoint when ``db`` is given.

Sandbox: ``https://api-sandbox.coinbase.com`` exists but does not
enforce auth and only mocks Accounts/Orders with static responses — it
is not a real dry-run venue, so this connector talks to production only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
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

__all__ = ["CoinbaseConnector", "CoinbaseError"]

_log = get_logger(__name__)

API_BASE = "https://api.coinbase.com"
API_HOST = "api.coinbase.com"
KEY_NAME_ENV = "COINBASE_API_KEY_NAME"
PRIVATE_KEY_ENV = "COINBASE_API_PRIVATE_KEY"

_JWT_TTL = 120  # seconds, per Coinbase's scheme

# ── secp256r1 (P-256) domain parameters (SEC2) ────────────────────

_P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
_A = _P - 3
_B = 0x5AC635D8AA3A93E7B3EBBD557769886BC651D06B0CC53B0F63BCE3C3E27D2604B
_GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
_GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


class CoinbaseError(ConnectorError):
    """A Coinbase API call (or the JWT auth around it) failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, error_code: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


# ── stdlib-only P-256 ECDSA (ES256 for the JWT) ──────────────────


class _Point:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def _point_add(p1: _Point | None, p2: _Point | None) -> _Point | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    if p1.x == p2.x:
        if (p1.y + p2.y) % _P == 0:
            return None  # inverse points -> point at infinity
        lam = (3 * p1.x * p1.x + _A) * pow(2 * p1.y, -1, _P) % _P
    else:
        lam = (p2.y - p1.y) * pow(p2.x - p1.x, -1, _P) % _P
    x3 = (lam * lam - p1.x - p2.x) % _P
    y3 = (lam * (p1.x - x3) - p1.y) % _P
    return _Point(x3, y3)


def _scalar_mul(k: int, point: _Point) -> _Point | None:
    result: _Point | None = None
    addend: _Point | None = point
    while k:
        if k & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return result


_G = _Point(_GX, _GY)


def _der_items(data: bytes) -> list[tuple[int, bytes]]:
    """Split DER bytes into (tag, value) items (short + long form)."""
    out: list[tuple[int, bytes]] = []
    i = 0
    while i < len(data):
        if i + 2 > len(data):
            raise CoinbaseError(
                "malformed EC private key: truncated DER"
            )
        tag = data[i]
        length = data[i + 1]
        i += 2
        if length & 0x80:
            nbytes = length & 0x7F
            if not nbytes or i + nbytes > len(data):
                raise CoinbaseError(
                    "malformed EC private key: bad DER length"
                )
            length = int.from_bytes(data[i:i + nbytes], "big")
            i += nbytes
        if i + length > len(data):
            raise CoinbaseError(
                "malformed EC private key: DER overruns buffer"
            )
        out.append((tag, data[i:i + length]))
        i += length
    return out


def _sec1_private_scalar(der: bytes) -> int:
    """Extract the 32-byte private scalar from SEC1 or PKCS8 DER."""
    if not der or der[0] != 0x30:
        raise CoinbaseError(
            "malformed EC private key: expected a DER SEQUENCE"
        )
    items = _der_items(der)
    # Skip the outer SEQUENCE wrapper (tag 0x30) when present.
    if len(items) == 1 and items[0][0] == 0x30:
        items = _der_items(items[0][1])
    for tag, value in items:
        if tag != 0x04:  # OCTET STRING
            continue
        if value[:1] == b"\x30":
            # PKCS8: the octet string wraps the inner SEC1 structure.
            return _sec1_private_scalar(value)
        if len(value) == 32:
            scalar = int.from_bytes(value, "big")
            if 1 <= scalar < _N:
                return scalar
    raise CoinbaseError(
        "malformed EC private key: no P-256 private scalar found — "
        "the key must be a SEC1 (BEGIN EC PRIVATE KEY) or PKCS8 "
        "(BEGIN PRIVATE KEY) P-256 key"
    )


def _load_private_scalar(pem: str) -> int:
    """Parse a PEM EC private key into the scalar ``d``."""
    text = (pem or "").strip()
    if "PRIVATE KEY" not in text:
        raise CoinbaseError(
            "the Coinbase API private key must be the PEM text Coinbase "
            "showed at key creation (-----BEGIN EC PRIVATE KEY----- ...)"
        )
    b64 = "".join(
        line.strip()
        for line in text.splitlines()
        if line and not line.startswith("-----")
    )
    try:
        der = base64.b64decode(b64)
    except Exception as exc:
        raise CoinbaseError(
            "the Coinbase API private key is not valid base64 PEM"
        ) from exc
    return _sec1_private_scalar(der)


def _ecdsa_sign_p256(scalar: int, message: bytes) -> bytes:
    """Sign with P-256; returns the raw 64-byte (r || s) signature."""
    digest = int.from_bytes(hashlib.sha256(message).digest(), "big")
    while True:
        k = secrets.randbelow(_N - 1) + 1
        r_point = _scalar_mul(k, _G)
        if r_point is None:  # pragma: no cover - k == 0 mod n
            continue
        r = r_point.x % _N
        if r == 0:
            continue
        s = (pow(k, -1, _N) * (digest + r * scalar)) % _N
        if s == 0:
            continue
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _ecdsa_verify_p256(
    pub_x: int, pub_y: int, message: bytes, signature: bytes
) -> bool:
    """Verify a raw 64-byte P-256 signature (independent check path)."""
    if len(signature) != 64:
        return False
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:], "big")
    if not (1 <= r < _N and 1 <= s < _N):
        return False
    digest = int.from_bytes(hashlib.sha256(message).digest(), "big")
    w = pow(s, -1, _N)
    u1 = (digest * w) % _N
    u2 = (r * w) % _N
    point = _point_add(
        _scalar_mul(u1, _G), _scalar_mul(u2, _Point(pub_x, pub_y))
    )
    return point is not None and (point.x % _N) == r


def _public_point(scalar: int) -> _Point:
    point = _scalar_mul(scalar, _G)
    if point is None:  # pragma: no cover - scalar == 0 mod n
        raise CoinbaseError("invalid EC private key (zero scalar)")
    return point


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _build_jwt(
    method: str, path: str, key_name: str, pem: str
) -> str:
    """Mint one request-scoped ES256 JWT for the Advanced Trade API."""
    scalar = _load_private_scalar(pem)
    # Fail fast on a key that cannot possibly authenticate: sign a probe
    # and verify it against the key's own public point.
    pub = _public_point(scalar)
    probe = b"coinbase-connector-key-check"
    if not _ecdsa_verify_p256(
        pub.x, pub.y, probe, _ecdsa_sign_p256(scalar, probe)
    ):
        raise CoinbaseError(
            "the EC private key failed its own signature check — "
            "it is corrupt or not a P-256 key"
        )
    now = int(time.time())
    header = {"alg": "ES256", "kid": key_name,
              "nonce": secrets.token_hex(16)}
    payload = {
        "sub": key_name,
        "iss": "coinbase-cloud",
        "nbf": now,
        "exp": now + _JWT_TTL,
        "aud": ["retail_rest_api_proxy"],
        "uri": f"{method.upper()} {API_HOST}{path}",
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    ).encode("ascii")
    signature = _ecdsa_sign_p256(scalar, signing_input)
    return f"{signing_input.decode('ascii')}.{_b64url(signature)}"


def _num(value: float) -> str:
    """A plain decimal string for the Coinbase API (no 3000.0, no 1e-05)."""
    text = repr(float(value))
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


# ── connector ────────────────────────────────────────────────────

_ORDER_SIDES = ("BUY", "SELL")
_ORDER_TYPES = ("MARKET", "LIMIT")


@register_connector
class CoinbaseConnector(Connector):
    """Devon's Coinbase Advanced Trade adapter: accounts, products, orders."""

    id = "coinbase"
    name = "Coinbase"
    description = (
        "Coinbase Advanced Trade: list brokerage accounts, fetch product "
        "details, and place/cancel spot orders. CDP API key auth via "
        "short-lived ES256 JWTs (stdlib-only signer; SEC1 or PKCS8 PEM). "
        "Orders require explicit owner confirmation."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        key_name: str | None = None,
        private_key: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a CDP API key against /brokerage/accounts and vault it.

        ``key_name`` is the key's name/UUID from the CDP portal;
        ``private_key`` is the full PEM text (or a path to the PEM file).
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "coinbase is already connected — one account per service. "
                "Disconnect first to switch API keys."
            )
        name = (key_name or "").strip() or prompt_secret(
            "Coinbase API key name", env_var=KEY_NAME_ENV
        )
        pem = (private_key or "").strip() or prompt_secret(
            "Coinbase API private key (PEM text or path)",
            env_var=PRIVATE_KEY_ENV,
        )
        if not name or not pem:
            raise ConnectorError(
                "empty API key name/private key: nothing to connect with"
            )
        pem = self._read_pem(pem)
        accounts = self._api(
            "GET", "/api/v3/brokerage/accounts", key_name=name, pem=pem
        )
        n_accounts = len(accounts.get("accounts", []))
        self._store_credential(
            "coinbase",
            pem,
            credential_type="api_key",
            scopes=["trade:read", "trade:write"],
            metadata={"key_name": name},
        )
        _log.info("coinbase connected (key %s...)", name[:8])
        return ConnectResult(
            ok=True,
            account="coinbase",
            scopes=["trade:read", "trade:write"],
            message=(
                f"connected to Coinbase Advanced Trade ({n_accounts} "
                "brokerage account(s) visible). The EC private key is in "
                "the encrypted vault. Every order still needs your "
                "explicit confirmation at call time."
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
                       "--name coinbase`",
            )
        try:
            accounts = self._api("GET", "/api/v3/brokerage/accounts")
        except CoinbaseError as exc:
            return ConnectorStatus(
                connected=False,
                account="coinbase",
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"key rejected ({exc}): rotate it in the CDP "
                       "portal and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account="coinbase",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"JWT accepted; "
                   f"{len(accounts.get('accounts', []))} account(s) visible",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/api/v3/brokerage/accounts")
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
            if not payload.get("product_id"):
                raise CoinbaseError(
                    "the resolved checkpoint has no order payload — "
                    "it cannot place the order"
                )
            return self._place_now(payload)
        if stage == "cancel_order":
            if not payload.get("order_ids"):
                raise CoinbaseError(
                    "the resolved checkpoint has no cancel payload"
                )
            return self._cancel_now(payload)
        raise CoinbaseError(
            f"coinbase cannot resume checkpoint stage {stage!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def list_accounts(self) -> list[dict[str, Any]]:
        """Brokerage accounts (``GET /api/v3/brokerage/accounts``)."""
        data = self._api("GET", "/api/v3/brokerage/accounts")
        accounts = data.get("accounts", [])
        return accounts if isinstance(accounts, list) else []

    def get_product(self, product_id: str) -> dict[str, Any]:
        """One product's details (``GET /api/v3/brokerage/products/{id}``).

        ``product_id`` looks like ``BTC-USD``.
        """
        product_id = (product_id or "").strip().upper()
        if not product_id:
            raise ConnectorError("product_id is required (e.g. BTC-USD)")
        data = self._api(
            "GET", f"/api/v3/brokerage/products/{product_id}"
        )
        return data if isinstance(data, dict) else {}

    # ── orders (confirmation-gated real money) ────────────────────

    def place_order(
        self,
        product_id: str,
        side: str,
        order_type: str,
        *,
        size: float = 0.0,
        price: float = 0.0,
        quote_size: float = 0.0,
        client_order_id: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Place a spot order (``POST /api/v3/brokerage/orders``).

        REAL MONEY — never runs on implied consent. MARKET orders need
        either ``size`` (base currency, e.g. 0.01 BTC) or ``quote_size``
        (quote currency spend, e.g. 50 USD — BUY only). LIMIT orders need
        ``size`` + ``price``. ``client_order_id`` defaults to a random
        UUID (Coinbase requires uniqueness per order).
        """
        product_id = (product_id or "").strip().upper()
        side = (side or "").upper()
        order_type = (order_type or "").upper()
        if not product_id:
            raise ConnectorError("product_id is required (e.g. BTC-USD)")
        if side not in _ORDER_SIDES:
            raise ConnectorError(f"invalid side {side!r}: use BUY or SELL")
        if order_type not in _ORDER_TYPES:
            raise ConnectorError(
                f"invalid order type {order_type!r}: use MARKET or LIMIT"
            )
        if order_type == "MARKET":
            if size <= 0 and quote_size <= 0:
                raise ConnectorError(
                    "MARKET orders need size (base) or quote_size (quote)"
                )
            if quote_size > 0 and side != "BUY":
                raise ConnectorError(
                    "quote_size market orders are BUY-only on Coinbase"
                )
        else:  # LIMIT
            if size <= 0:
                raise ConnectorError("LIMIT orders need a positive size")
            if price <= 0:
                raise ConnectorError("LIMIT orders need a positive price")
        payload: dict[str, Any] = {
            "product_id": product_id,
            "side": side,
            "order_type": order_type,
            "size": size,
            "price": price,
            "quote_size": quote_size,
            "client_order_id": client_order_id or secrets.token_hex(16),
        }
        sizing = (
            f"{size:g} base @ {price:g}"
            if order_type == "LIMIT"
            else (f"{size:g} base" if size > 0 else f"{quote_size:g} quote")
        )
        summary = f"{side} {sizing} {product_id} ({order_type})"
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="place_order",
            title=f"Place Coinbase order: {summary}",
            instructions="\n".join([
                "Devon wants to place this REAL-MONEY order.",
                "Review it — a market order can fill immediately.",
                f"Order: {summary}",
                f"Client order id: {payload['client_order_id']}",
            ]),
            resume_state={"payload": payload},
        )
        return self._place_now(payload)

    def _place_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload["order_type"] == "MARKET":
            if payload.get("quote_size"):
                config = {
                    "market_market_ioc": {
                        "quote_size": _num(payload["quote_size"])
                    }
                }
            else:
                config = {
                    "market_market_ioc": {
                        "base_size": _num(payload["size"])
                    }
                }
        else:
            config = {
                "limit_limit_gtc": {
                    "base_size": _num(payload["size"]),
                    "limit_price": _num(payload["price"]),
                    "post_only": False,
                }
            }
        body = {
            "client_order_id": payload["client_order_id"],
            "product_id": payload["product_id"],
            "side": payload["side"],
            "order_configuration": config,
        }
        data = self._api("POST", "/api/v3/brokerage/orders", body=body)
        result = data.get("success_response", {}) if isinstance(
            data, dict) else {}
        _log.info(
            "coinbase order placed: %s %s (id %s)",
            payload["side"], payload["product_id"],
            result.get("order_id", "?"),
        )
        return data if isinstance(data, dict) else {}

    def cancel_order(
        self,
        order_id: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Cancel an open order (``POST .../orders/batch_cancel``).

        Confirmation-gated like placement: cancelling changes live
        protection, so the owner approves each one.
        """
        order_id = (order_id or "").strip()
        if not order_id:
            raise ConnectorError("order_id is required")
        payload = {"order_ids": [order_id]}
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="cancel_order",
            title=f"Cancel Coinbase order {order_id}",
            instructions="\n".join([
                "Devon wants to cancel this open order.",
                f"Order id: {order_id}",
            ]),
            resume_state={"payload": payload},
        )
        return self._cancel_now(payload)

    def _cancel_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._api(
            "POST",
            "/api/v3/brokerage/orders/batch_cancel",
            body={"order_ids": payload["order_ids"]},
        )
        result = data if isinstance(data, dict) else {}
        _log.info(
            "coinbase cancel requested for %s",
            ",".join(payload["order_ids"]),
        )
        return result

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "coinbase is not connected — run "
                "`nm connectors connect --name coinbase` first"
            )
        return cred

    @staticmethod
    def _read_pem(pem: str) -> str:
        """Accept PEM text or a path to a PEM file."""
        import os

        text = (pem or "").strip()
        if "PRIVATE KEY" in text:
            return text
        if os.path.isfile(text):
            with open(text, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        return text  # let _load_private_scalar fail with a clear message

    def _auth_header(
        self, method: str, path: str, *, key_name: str = "",
        pem: str = ""
    ) -> str:
        if not key_name or not pem:
            cred = self._require_credential()
            meta = cred.metadata or {}
            key_name = key_name or str(meta.get("key_name", ""))
            pem = pem or cred.password
        if not key_name:
            raise CoinbaseError("no API key name stored — reconnect")
        token = _build_jwt(method, path, key_name, pem)
        return f"Bearer {token}"

    def _api(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        key_name: str = "",
        pem: str = "",
    ) -> dict[str, Any]:
        """One Advanced Trade call. Failures become CoinbaseError."""
        url = f"{API_BASE}{path}"
        headers = {
            "Authorization": self._auth_header(
                method, path, key_name=key_name, pem=pem
            )
        }
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=headers, params=params or None
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, body or {}, headers=headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise CoinbaseError(f"coinbase request failed: {exc}") from exc
        if resp.status == 401:
            raise CoinbaseError(
                "coinbase rejected the JWT (401): the key name/PEM pair is "
                "wrong or the key was revoked — rotate it in the CDP "
                "portal and reconnect",
                status_code=401,
            )
        if resp.status == 403:
            raise CoinbaseError(
                "coinbase refused (403): the key lacks permission for "
                "this endpoint — check the key's scopes in the CDP portal",
                status_code=403,
            )
        if resp.status == 429:
            raise CoinbaseError(
                "coinbase rate limit hit (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            code, msg = self._error_detail(resp)
            raise CoinbaseError(
                f"coinbase {method} {path} failed ({resp.status}): {msg}",
                status_code=resp.status,
                error_code=code,
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise CoinbaseError(
                f"coinbase {method} {path} returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, str]:
        try:
            body = resp.json()
            if isinstance(body, dict):
                errors = body.get("errors", [])
                if errors:
                    first = errors[0]
                    return (
                        str(first.get("error", "")),
                        str(first.get("message", body))[:200],
                    )
                return "", str(
                    body.get("message", body.get("error", body))
                )[:200]
        except Exception:  # noqa: BLE001 - fall back to raw text
            pass
        return "", resp.text[:200]
