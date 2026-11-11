"""Mock screens: FakeDetail-style platform screen generators.

A transport-free L4 organ beside ``makers``.  A screen is a
typed model first (``schema``), drawn second (``render``)
through platform designs held as data (``designs``) — the
makers doctrine: designs are data, one layout engine per
screen class, typography through the typography service,
avatars through the makers mask/placeholder machinery, and a
drawn-region record returned with the pixels so tests prove
structure, not vibes.

Phase 0+1 (this build): the schema, the design registry, the
Pillow renderer foundation, and the flagship chats —
WhatsApp 1:1, WhatsApp group and Instagram DM, each in
iPhone and Android chrome.  Lists/status/calls, social
profiles, animation, bulk data and NL authoring are later
phases of the same organ.

The organ never claims a render is a real captured
screenshot: it is a mock screen, generated from data the
caller supplied.
"""

from __future__ import annotations

from .designs import CHAT_DESIGNS
from .schema import (
    DEVICE_STYLES,
    MESSAGE_KINDS,
    TICK_STATES,
    DeviceChrome,
    Message,
    Participant,
    Screen,
)
from .service import MockScreensError, MockScreenService

#: The /mock command kinds this organ answers (the runtime
#: dispatch set and this tuple are pinned equal by tests,
#: the makers idiom).
MOCK_COMMANDS: tuple[str, ...] = ("mock",)

__all__ = [
    "CHAT_DESIGNS",
    "DEVICE_STYLES",
    "MESSAGE_KINDS",
    "MOCK_COMMANDS",
    "TICK_STATES",
    "DeviceChrome",
    "Message",
    "MockScreenService",
    "MockScreensError",
    "Participant",
    "Screen",
]
