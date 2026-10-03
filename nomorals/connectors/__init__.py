"""Devon connectors: the bridge from advice to action on external services.

A connector owns one service's auth flow, credential lifecycle (encrypted
vault — never chat, logs, or argv), live status, and API-legitimate
provisioning (repos, webhooks, keys, releases, ...).

Adding a service: subclass :class:`Connector`, decorate with
:func:`register_connector`, and it appears in ``nm connectors list``.
Shipped: GitHub, Mono (NG bank data), Plaid (US/EU bank data), virtual
cards (Flutterwave), proxy pool, Jumia seller API, Konga buyer browse,
Telegram Bot API, Discord Bot API, Gmail, Google Drive, Paystack (NG
payments), Binance spot, AWS (EC2/S3), Dropbox, YouTube, Spotify — the
framework is deliberately service-agnostic so more slot in.
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
from .jiji import JijiConnector, JijiError
from .jumia import JumiaConnector
from .konga import KongaConnector, KongaError
from .mono import MonoConnector
from .plaid import PlaidConnector
from .proxypool import ProxyPoolConnector
from .registry import (
    create_connector,
    get_connector,
    list_connectors,
    register_connector,
)
from .virtualcards import VirtualCardsConnector
from .aws import AWSConnector, AWSError
from .binance import BinanceConnector, BinanceError
from .discord import DiscordConnector, DiscordError
from .dropbox import DropboxConnector, DropboxError
from .drive import DriveConnector, DriveError
from .gmail import GmailConnector, GmailError
from .paystack import PaystackConnector, PaystackError
from .spotify import SpotifyConnector, SpotifyError
from .telegram import TelegramConnector, TelegramError
from .youtube import YouTubeConnector, YouTubeError

__all__ = [
    "AWSConnector",
    "AWSError",
    "AuthMethod",
    "BinanceConnector",
    "BinanceError",
    "CheckpointKind",
    "CheckpointState",
    "CheckpointStore",
    "Connector",
    "ConnectorError",
    "ConnectorStatus",
    "ConnectResult",
    "DiscordConnector",
    "DiscordError",
    "DriveConnector",
    "DriveError",
    "DropboxConnector",
    "DropboxError",
    "GitHubConnector",
    "GitHubError",
    "GmailConnector",
    "GmailError",
    "HumanCheckpoint",
    "HumanCheckpointPending",
    "JijiConnector",
    "JijiError",
    "JumiaConnector",
    "KongaConnector",
    "KongaError",
    "MonoConnector",
    "PaystackConnector",
    "PaystackError",
    "PlaidConnector",
    "ProxyPoolConnector",
    "SpotifyConnector",
    "SpotifyError",
    "TelegramConnector",
    "TelegramError",
    "VirtualCardsConnector",
    "YouTubeConnector",
    "YouTubeError",
    "create_connector",
    "device_flow_token",
    "get_connector",
    "list_connectors",
    "pick_scopes",
    "prompt_secret",
    "register_connector",
    "request_human_action",
]
