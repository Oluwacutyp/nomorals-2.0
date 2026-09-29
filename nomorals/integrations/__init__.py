"""Service integrations for the bot.

This package provides integrations with various external services:
- Email (Gmail, IMAP/SMTP, webmail)
- Calendar (Google Calendar, CalDAV)
- Shopping (product search, price comparison)
- Smart home (Home Assistant, direct device APIs)
- Social media (posting, monitoring)
- Payments (crypto wallets, payment processors)

Each integration supports multiple methods where possible:
- API-based (fastest, most reliable)
- Browser automation (most flexible, works with any web interface)
- CLI tools (when available)
"""

from .email_integration import EmailIntegration
from .calendar_integration import CalendarIntegration
from .shopping_integration import ShoppingIntegration
from .smarthome_integration import SmartHomeIntegration
from .voice_integration import VoiceIntegration
from .payment_integration import PaymentIntegration

__all__ = [
    "EmailIntegration",
    "CalendarIntegration",
    "ShoppingIntegration",
    "SmartHomeIntegration",
    "VoiceIntegration",
    "PaymentIntegration",
]
