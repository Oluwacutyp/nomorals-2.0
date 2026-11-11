"""Mock screens: FakeDetail-style platform screen generators.

A transport-free L4 organ beside ``makers``.  A screen is a
typed model first (``schema``), drawn second (``render``)
through platform designs held as data (``designs``) — the
makers doctrine: designs are data, one layout engine per
screen class, typography through the typography service,
avatars through the makers mask/placeholder machinery, and a
drawn-region record returned with the pixels so tests prove
structure, not vibes.

Phase 0+1: the schema, the design registry, the Pillow
renderer foundation, and the flagship chats — WhatsApp 1:1,
WhatsApp group and Instagram DM, each in iPhone and Android
chrome.  Phase 2 (this build): the remaining chat families
enter the same registry — Telegram, Messenger, Snapchat,
iMessage (incl. the green-SMS Apple-to-Android variant)
and Android Messages, 1:1 and group — each family's
choreography carried as design data (where times and ticks
live, bubble tails, run avatars, read-state lines, input
bars).  Chat tools the census names but does not inspect
to flagship depth stay named remainders (see the census
§3.10 note in the tests) rather than guessed designs.
The media hand-off: image and voice messages carry REAL
media — a generated image or any image bytes compose
into the photo frame, a real voice note is drawn with
its measured waveform and true duration (``media``
probes in :mod:`.media`) — and the placeholder scene /
synthetic waveform remain only for messages carrying no
media.  Phase 2b (this build): the lists / status / calls
classes — WhatsApp, Instagram, Telegram and Snapchat
chat lists, the WhatsApp call list and status (Updates)
list, the WhatsApp Status screen and its Views sheet,
the generic + named-platform call screens, and the
iPhone Call History — each a typed model in ``schema``,
designs as data in ``designs`` (``LIST_DESIGNS`` /
``STATUS_DESIGNS`` / ``CALL_DESIGNS``), and its own
layout engine in ``render_screens`` under the same
media hand-off contract.  Social profiles, animation,
bulk data and NL authoring are later phases of the same
organ.

The organ never claims a render is a real captured
screenshot: it is a mock screen, generated from data the
caller supplied.
"""

from __future__ import annotations

from .designs import (
    CALL_DESIGNS,
    CHAT_DESIGNS,
    LIST_DESIGNS,
    STATUS_BG_COLORS,
    STATUS_DESIGNS,
)
from .schema import (
    CALL_CONTROLS,
    CALL_STATES,
    DEVICE_STYLES,
    LIST_FLAVORS,
    LIST_PREVIEW_KINDS,
    LIST_ROW_STATES,
    LIST_ROW_TYPES,
    MESSAGE_KINDS,
    RING_STATES,
    STATUS_KINDS,
    TICK_STATES,
    CallHistoryEntry,
    CallHistoryScreen,
    CallScreen,
    DeviceChrome,
    FolderTab,
    ListRow,
    ListScreen,
    Message,
    Participant,
    Screen,
    StatusScreen,
    StatusViewer,
    StatusViewsScreen,
    StoryItem,
)
from .service import MockScreensError, MockScreenService

#: The /mock command kinds this organ answers (the runtime
#: dispatch set and this tuple are pinned equal by tests,
#: the makers idiom).
MOCK_COMMANDS: tuple[str, ...] = ("mock",)

__all__ = [
    "CALL_CONTROLS",
    "CALL_DESIGNS",
    "CALL_STATES",
    "CHAT_DESIGNS",
    "DEVICE_STYLES",
    "LIST_DESIGNS",
    "LIST_FLAVORS",
    "LIST_PREVIEW_KINDS",
    "LIST_ROW_STATES",
    "LIST_ROW_TYPES",
    "MESSAGE_KINDS",
    "MOCK_COMMANDS",
    "RING_STATES",
    "STATUS_BG_COLORS",
    "STATUS_DESIGNS",
    "STATUS_KINDS",
    "TICK_STATES",
    "CallHistoryEntry",
    "CallHistoryScreen",
    "CallScreen",
    "DeviceChrome",
    "FolderTab",
    "ListRow",
    "ListScreen",
    "Message",
    "MockScreenService",
    "MockScreensError",
    "Participant",
    "Screen",
    "StatusScreen",
    "StatusViewer",
    "StatusViewsScreen",
    "StoryItem",
]
