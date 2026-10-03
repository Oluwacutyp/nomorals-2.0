"""Devon connectors: the bridge from advice to action on external services.

A connector owns one service's auth flow, credential lifecycle (encrypted
vault — never chat, logs, or argv), live status, and API-legitimate
provisioning (repos, webhooks, keys, releases, ...).

Adding a service: subclass :class:`Connector`, decorate with
:func:`register_connector`, and it appears in ``nm connectors list``.
Queued next: finance (Mono/Plaid), virtual cards, proxy pool, Nigerian
commerce — the framework is deliberately service-agnostic so those slot in.
"""

from __future__ import annotations

from .auth import device_flow_token, pick_scopes, prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .checkpoints import (
    CheckpointKind,
    CheckpointState,
    CheckpointStore,
    HumanCheckpoint,
    HumanCheckpointPending,
    request_human_action,
)
from .github import GitHubConnector, GitHubError
from .registry import (
    create_connector,
    get_connector,
    list_connectors,
    register_connector,
)

__all__ = [
    "AuthMethod",
    "CheckpointKind",
    "CheckpointState",
    "CheckpointStore",
    "Connector",
    "ConnectorError",
    "ConnectorStatus",
    "ConnectResult",
    "GitHubConnector",
    "GitHubError",
    "HumanCheckpoint",
    "HumanCheckpointPending",
    "create_connector",
    "device_flow_token",
    "get_connector",
    "list_connectors",
    "pick_scopes",
    "prompt_secret",
    "register_connector",
    "request_human_action",
]
