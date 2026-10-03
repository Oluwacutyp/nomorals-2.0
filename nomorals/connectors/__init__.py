"""Devon connectors: the bridge from advice to action on external services.

A connector owns one service's auth flow, credential lifecycle (encrypted
vault — never chat, logs, or argv), live status, and API-legitimate
provisioning (repos, webhooks, keys, releases, ...).

Adding a service: subclass :class:`Connector`, decorate with
:func:`register_connector`, and it appears in ``nm connectors list``.
Shipped: GitHub, Mono (NG bank data), Plaid (US/EU bank data), virtual
cards (Flutterwave), proxy pool, Jumia seller API, Konga buyer browse,
Jiji listings, Telegram Bot API, Discord Bot API, Gmail, Google Drive,
Paystack (NG payments), Binance spot, Coinbase Advanced Trade, Wise
transfers, Notion, Google Calendar, Trello, X/Twitter, Instagram,
LinkedIn, Slack, Twilio, AWS, Dropbox, YouTube, Spotify — the framework
is deliberately service-agnostic so more slot in.
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
from .binance import BinanceConnector, BinanceError
from .coinbase import CoinbaseConnector, CoinbaseError
from .discord import DiscordConnector, DiscordError
from .drive import DriveConnector, DriveError
from .gcalendar import GCalendarConnector, GCalendarError
from .gmail import GmailConnector, GmailError
from .notion import NotionConnector, NotionError
from .paystack import PaystackConnector, PaystackError
from .telegram import TelegramConnector, TelegramError
from .trello import TrelloConnector, TrelloError
from .wise import WiseConnector, WiseError
from .x import XConnector, XError
from .instagram import InstagramConnector, InstagramError
from .linkedin import LinkedInConnector, LinkedInError
from .slack import SlackConnector, SlackError
from .twilio import TwilioConnector, TwilioError

from .aws import AWSConnector, AWSError
from .dropbox import DropboxConnector, DropboxError
from .spotify import SpotifyConnector, SpotifyError
from .soundcloud import SoundCloudConnector, SoundCloudError
from .youtube import YouTubeConnector, YouTubeError

from .googleflow import GoogleFlowConnector, GoogleFlowError
from .leonardo import LeonardoConnector, LeonardoError
from .nanobanana import NanoBananaConnector, NanoBananaError
from .stabilityai import StabilityAIConnector, StabilityAIError

__all__ = [
    "AuthMethod",
    "AWSConnector",
    "AWSError",
    "BinanceConnector",
    "BinanceError",
    "CheckpointKind",
    "CheckpointState",
    "CheckpointStore",
    "CoinbaseConnector",
    "CoinbaseError",
    "Connector",
    "ConnectorError",
    "ConnectorStatus",
    "ConnectResult",
    "DiscordConnector",
    "DiscordError",
    "DropboxConnector",
    "DropboxError",
    "DriveConnector",
    "DriveError",
    "GCalendarConnector",
    "GCalendarError",
    "GitHubConnector",
    "GitHubError",
    "GmailConnector",
    "GmailError",
    "GoogleFlowConnector",
    "GoogleFlowError",
    "HumanCheckpoint",
    "HumanCheckpointPending",
    "InstagramConnector",
    "InstagramError",
    "JijiConnector",
    "JijiError",
    "JumiaConnector",
    "LinkedInConnector",
    "LinkedInError",
    "KongaConnector",
    "KongaError",
    "LeonardoConnector",
    "LeonardoError",
    "MonoConnector",
    "NanoBananaConnector",
    "NanoBananaError",
    "NotionConnector",
    "NotionError",
    "PaystackConnector",
    "PaystackError",
    "PlaidConnector",
    "ProxyPoolConnector",
    "SlackConnector",
    "SlackError",
    "SoundCloudConnector",
    "SoundCloudError",
    "SpotifyConnector",
    "SpotifyError",
    "StabilityAIConnector",
    "StabilityAIError",
    "TelegramConnector",
    "TelegramError",
    "TrelloConnector",
    "TrelloError",
    "TwilioConnector",
    "TwilioError",
    "VirtualCardsConnector",
    "WiseConnector",
    "WiseError",
    "YouTubeConnector",
    "YouTubeError",
    "XConnector",
    "XError",
    "create_connector",
    "device_flow_token",
    "get_connector",
    "list_connectors",
    "pick_scopes",
    "prompt_secret",
    "register_connector",
    "request_human_action",
]
