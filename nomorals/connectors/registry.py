"""Adapter registry: one lookup for every connector Devon knows."""

from __future__ import annotations

from typing import Any

from ..accounts.vault import CredentialVault
from ..core.http import HttpClient
from .base import Connector, ConnectorError

__all__ = [
    "create_connector",
    "get_connector",
    "list_connectors",
    "register_connector",
]

_REGISTRY: dict[str, type[Connector]] = {}


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
            }
        )
    return infos


def create_connector(
    connector_id: str,
    vault: CredentialVault,
    http: HttpClient | None = None,
) -> Connector:
    """Instantiate the adapter for ``connector_id``."""
    return get_connector(connector_id)(vault, http=http)
