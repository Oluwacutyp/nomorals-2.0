"""Service integrations for the bot.

This package provides integrations with various external services:
- Email (Gmail, IMAP/SMTP, webmail)
- Calendar (Google Calendar, CalDAV)
- Shopping (product search, price comparison, Naija marketplaces)
- Smart home (Home Assistant, direct device APIs, MQTT/Zigbee2MQTT)
- Digital twin (persistent queryable model of the home)
- Routines (natural-language → validated HA automations)
- Market data (keyless-first OHLCV/quotes, ccxt upgrades)
- Trading validation (backtest, paper trade, risk guards)
- Social media (posting, monitoring)
- Payments (crypto wallets, payment processors, virtual cards)
- Voice (TTS/STT, voice messages)

Each integration supports multiple methods where possible:
- API-based (fastest, most reliable)
- Browser automation (most flexible, works with any web interface)
- CLI tools (when available)

The original six classes stay eagerly importable (they're light). The rest
of the sweep surface is exposed lazily via ``__getattr__`` so that
``import nomorals.integrations`` never drags in heavy optional deps
(numpy/pandas/ccxt/faster-whisper) — see test_market_data.py
``LazyImportTests``.
"""

from .email_integration import EmailIntegration, EmailMessage, format_digest
from .calendar_integration import (
    CalendarIntegration,
    CalendarEvent,
    format_agenda,
    format_event,
)
from .shopping_integration import (
    ShoppingIntegration,
    Product,
    PriceComparison,
)
from .naija_deals import NaijaDealHunter, Deal, PriceAlert
from .naija_shopping import NaijaShoppingEngine, Steal
from .smarthome_integration import (
    SmartHomeIntegration,
    HAWebSocket,
    Device,
    DeviceState,
    Scene,
)
from .mqtt_client import MQTTBridge
from .digital_twin import HomeTwin
from . import routines
from . import market_data
from .payment_integration import PaymentIntegration, Transaction
from .voice_integration import VoiceIntegration, TTSEngine

__all__ = [
    # eager (light)
    "EmailIntegration",
    "EmailMessage",
    "format_digest",
    "CalendarIntegration",
    "CalendarEvent",
    "format_agenda",
    "format_event",
    "ShoppingIntegration",
    "Product",
    "PriceComparison",
    "NaijaDealHunter",
    "Deal",
    "PriceAlert",
    "NaijaShoppingEngine",
    "Steal",
    "SmartHomeIntegration",
    "HAWebSocket",
    "Device",
    "DeviceState",
    "Scene",
    "MQTTBridge",
    "HomeTwin",
    "routines",
    "market_data",
    "PaymentIntegration",
    "Transaction",
    "VoiceIntegration",
    "TTSEngine",
    # lazy (heavy optional deps)
    "sentinel_bridge",
    "SpeechToText",
    "TranscriptionResult",
]

# module-name → real module, attr → attribute inside it. Resolved on first
# access so the package import stays light.
_LAZY = {
    "sentinel_bridge": (".sentinel_bridge", None),  # the module itself
    "SpeechToText": (".stt", "SpeechToText"),
    "TranscriptionResult": (".stt", "TranscriptionResult"),
}

_LAZY_CACHE: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """PEP 562 lazy attribute loading for heavy integrations."""
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = module if attr is None else getattr(module, attr)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
