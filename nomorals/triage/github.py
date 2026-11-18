"""The GitHub client the triage engine files through.

One construction path, shared by the brain tool and the /triage verb:
the vault-backed :class:`nomorals.connectors.github.GitHubConnector`
(the same construction :mod:`nomorals.relay.github` uses for the
ticket relay). Without a stored GitHub credential the client is None
and the caller answers honestly — run ``nm connectors connect --name
github`` first — instead of failing every triage run with a bare
traceback.
"""
from __future__ import annotations

from typing import Any

__all__ = ["github_client", "github_not_connected_reason"]


def github_not_connected_reason() -> str:
    return ("github is not connected — run "
            "`nm connectors connect --name github` first")


def github_client(context: Any = None, *,
                  vault: Any = None) -> Any | None:
    """A GitHubConnector with a credential, or None when unconnected.

    A client passed in on the context (``context._triage_github``)
    wins — the test seam, same shape as the relay's
    ``_ticket_github``. Otherwise the vault is opened through the one
    unlock path and the connector is built from it; a connector with
    no stored credential comes back as None.
    """
    injected = getattr(context, "_triage_github", None) if context else None
    if injected is not None:
        return injected
    try:
        from ..accounts.vault import open_vault
        from ..connectors.registry import create_connector
    except Exception:  # noqa: BLE001 - imports are the caller's problem
        return None
    try:
        if vault is None:
            vault = open_vault(
                getattr(context, "db", None) if context else None,
                context=context, actor="triage")
        connector = create_connector("github", vault)
    except Exception:  # noqa: BLE001 - a locked vault is "not connected"
        return None
    try:
        if connector._load_credential() is None:  # noqa: SLF001
            return None
    except Exception:  # noqa: BLE001 - no credential readable
        return None
    return connector
