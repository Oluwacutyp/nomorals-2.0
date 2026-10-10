"""Adapter registry: one lookup for every connector Devon knows."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..accounts.vault import CredentialVault
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .base import Connector, ConnectorError

__all__ = [
    "connector_categories",
    "create_connector",
    "describe_connector",
    "get_connector",
    "health_snapshot",
    "list_connectors",
    "register_connector",
    "search_connectors",
]

_log = get_logger(__name__)

_REGISTRY: dict[str, type[Connector]] = {}

#: Fallback grouping for fleet views (n8n node groups / HA device classes
#: teach this pattern). A connector's own ``CATEGORY`` class attribute wins
#: over this map.
_CATEGORY_MAP: dict[str, str] = {
    "github": "dev",
    "jumia": "commerce", "konga": "commerce", "jiji": "commerce",
    "mono": "banking", "plaid": "banking",
    "paystack": "payments", "stripe": "payments", "wise": "payments",
    "virtualcards": "payments",
    "binance": "trading", "coinbase": "trading", "exness": "trading",
    "telegram": "messaging", "discord": "messaging", "slack": "messaging",
    "twilio": "messaging",
    "gmail": "productivity", "gdrive": "productivity",
    "gcalendar": "productivity", "notion": "productivity",
    "trello": "productivity",
    "x": "social", "instagram": "social", "linkedin": "social",
    "youtube": "media", "spotify": "media", "soundcloud": "media",
    "audd": "media", "leonardo": "media", "nano_banana": "media",
    "stability_ai": "media", "google_flow": "media",
    "aws": "cloud", "dropbox": "cloud",
    "duffel": "travel",
    "proxypool": "network",
}


def register_connector(cls: type[Connector]) -> type[Connector]:
    """Class decorator registering a connector adapter by its id."""
    cid = getattr(cls, "id", "")
    if not cid:
        raise ConnectorError(
            f"cannot register {cls.__name__}: empty connector id"
        )
    if cid in _REGISTRY:
        raise ConnectorError(f"connector {cid!r} is already registered")
    _REGISTRY[cid] = cls
    return cls


def get_connector(connector_id: str) -> type[Connector]:
    """The adapter class for ``connector_id``; raises on unknown ids."""
    try:
        return _REGISTRY[connector_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ConnectorError(
            f"unknown connector {connector_id!r}; known: {known}"
        ) from None


def category_of(connector_id: str) -> str:
    """The fleet-view category for a connector id."""
    cls = _REGISTRY.get(connector_id)
    declared = (getattr(cls, "CATEGORY", "") or "").strip() if cls else ""
    if declared:
        return declared
    return _CATEGORY_MAP.get(connector_id, "")


def list_connectors() -> list[dict[str, Any]]:
    """Every registered connector: id, name, description, auth methods."""
    infos = []
    for cid in sorted(_REGISTRY):
        cls = _REGISTRY[cid]
        infos.append(
            {
                "id": cid,
                "name": cls.name,
                "description": cls.description,
                "auth_methods": [m.value for m in cls.auth_methods],
                "provisionable": list(cls.PROVISIONABLE),
                "category": category_of(cid),
            }
        )
    return infos


def search_connectors(query: str) -> list[dict[str, Any]]:
    """Filter connectors by keyword (n8n's node-list query).

    Matches against id, name, description, and category; empty query
    returns everything. Case-insensitive.
    """
    q = (query or "").strip().lower()
    if not q:
        return list_connectors()
    return [
        info for info in list_connectors()
        if q in info["id"].lower()
        or q in (info["name"] or "").lower()
        or q in (info["description"] or "").lower()
        or q in (info["category"] or "").lower()
    ]


def describe_connector(connector_id: str) -> dict[str, Any]:
    """Full manifest for one connector: identity, auth, capabilities.

    Merges the class-level metadata with the live ``capabilities()``
    contract. Instantiation-free where possible — capabilities that need a
    vault are reported from the class shape instead.
    """
    cls = get_connector(connector_id)
    caps: dict[str, Any] = {}
    probe = getattr(cls, "capabilities", None)
    if callable(probe):
        try:
            # Probe on the class without instantiating: the base contract
            # only reads class attributes, so passing the class as
            # ``self`` works; ad-hoc overrides needing instance state
            # fall through to the class-level inventory below.
            maybe = probe(cls)  # type: ignore[arg-type]
            if isinstance(maybe, dict):
                caps = maybe
        except Exception:  # noqa: BLE001 - capability probe is best-effort
            caps = {}
    # Class-level feature inventory without a vault: public methods minus
    # the lifecycle set, mirroring Connector.capabilities().
    lifecycle = {
        "connect", "disconnect", "status", "test_connection",
        "connect_url", "can_provision", "provision",
        "request_human", "resume_checkpoint", "capabilities",
    }
    features = sorted(
        name for name in dir(cls)
        if not name.startswith("_")
        and name not in lifecycle
        and callable(getattr(cls, name, None))
    )
    return {
        "id": connector_id,
        "name": cls.name,
        "description": cls.description,
        "category": category_of(connector_id),
        "auth_methods": [m.value for m in cls.auth_methods],
        "provisionable": list(cls.PROVISIONABLE),
        "features": features,
        "capabilities": caps,
    }


def connector_categories() -> dict[str, list[str]]:
    """Category → connector ids, for fleet/grouped views."""
    groups: dict[str, list[str]] = {}
    for cid in sorted(_REGISTRY):
        cat = category_of(cid) or "other"
        groups.setdefault(cat, []).append(cid)
    return groups


def create_connector(
    connector_id: str,
    vault: CredentialVault,
    http: HttpClient | None = None,
) -> Connector:
    """Instantiate the adapter for ``connector_id``."""
    return get_connector(connector_id)(vault, http=http)


def _check_one(
    connector_id: str,
    vault: Any,
    timeout: float,
    factory: Any,
) -> dict[str, Any]:
    started = time.time()
    try:
        connector = factory(connector_id, vault)
        st = connector.status()
        return {
            "id": connector_id,
            "name": connector.name,
            "connected": bool(st.connected),
            "account": st.account,
            "detail": st.detail,
            "latency_ms": round((time.time() - started) * 1000, 1),
        }
    except Exception as exc:  # noqa: BLE001 - one bad connector can't sink the fleet
        _log.debug("health check %s failed: %s", connector_id, exc)
        return {
            "id": connector_id,
            "name": connector_id,
            "connected": False,
            "account": None,
            "detail": f"check failed: {exc}",
            "latency_ms": round((time.time() - started) * 1000, 1),
        }


def health_snapshot(
    vault: Any = None,
    *,
    ids: list[str] | None = None,
    timeout: float = 15.0,
    factory: Any = None,
) -> dict[str, Any]:
    """Fleet-wide connector health — the dashboard primitive.

    Runs ``status()`` for every registered connector (or ``ids``) with a
    per-connector timeout, in threads. One connector hanging or crashing
    never sinks the snapshot (HA's device-health view teaches this: mark
    unavailable, keep going).

    ``factory`` defaults to :func:`create_connector`; tests inject fakes.
    """
    targets = list(ids) if ids is not None else sorted(_REGISTRY)
    make = factory or create_connector
    details: list[dict[str, Any]] = []
    if not targets:
        return {
            "total": 0, "connected": 0, "disconnected": 0,
            "checked_at": time.time(), "details": [],
        }
    with ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
        futures = {
            pool.submit(_check_one, cid, vault, timeout, make): cid
            for cid in targets
        }
        for future in futures:
            try:
                details.append(future.result(timeout=timeout + 5))
            except Exception as exc:  # noqa: BLE001 - never sink the fleet
                cid = futures[future]
                details.append({
                    "id": cid, "name": cid, "connected": False,
                    "account": None,
                    "detail": f"check timed out/crashed: {exc}",
                    "latency_ms": 0.0,
                })
    details.sort(key=lambda d: d["id"])
    connected = sum(1 for d in details if d["connected"])
    return {
        "total": len(details),
        "connected": connected,
        "disconnected": len(details) - connected,
        "checked_at": time.time(),
        "details": details,
    }
