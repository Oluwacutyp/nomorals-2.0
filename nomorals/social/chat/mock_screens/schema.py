"""The mock-screen schema: plain data, no rendering.

A :class:`Screen` is everything a chat screen shows, as
data: who is in it (:class:`Participant`), what was said
(:class:`Message` — text and media kinds, ticks, reply-to,
reactions, seen state), and the phone it is shown on
(:class:`DeviceChrome`).  The schema round-trips through
plain dicts (``to_dict`` / ``from_dict``) so a screen can be
authored by a command parser today and by an NL front end
later without the renderer changing.

States are strings from small closed vocabularies, validated
at construction — a screen that cannot exist in the real app
should not be describable here either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Message element kinds the chat engine draws.  ``divider``
#: is a day chip (Today / Yesterday / a date), ``system`` a
#: centred admin/notice row (encryption notice, "X added Y").
MESSAGE_KINDS: tuple[str, ...] = (
    "text", "image", "voice", "divider", "system")

#: Delivery choreography for an outgoing message:
#: ``sending`` (clock), ``sent`` (one tick), ``delivered``
#: (two ticks), ``read`` (two blue ticks on WhatsApp).
TICK_STATES: tuple[str, ...] = ("sending", "sent", "delivered", "read")

#: Device chrome styles — the same conversation wears either.
DEVICE_STYLES: tuple[str, ...] = ("iphone", "android")


@dataclass
class Participant:
    """One person on the screen.  ``id`` is the stable key
    messages address (``"me"`` is the phone's owner);
    ``color`` is the group name colour as ``#rrggbb`` (empty
    = the design assigns one deterministically);
    ``avatar`` is a reference — a local path the renderer
    loads when it exists, else the initials placeholder
    stands in."""

    id: str
    name: str
    color: str = ""
    avatar: str = ""
    admin: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name,
                "color": self.color, "avatar": self.avatar,
                "admin": self.admin}

    @classmethod
    def from_dict(cls, data: dict) -> Participant:
        return cls(id=str(data.get("id") or ""),
                   name=str(data.get("name") or ""),
                   color=str(data.get("color") or ""),
                   avatar=str(data.get("avatar") or ""),
                   admin=bool(data.get("admin")))


@dataclass
class Message:
    """One element of the conversation.  ``sender`` is a
    participant id (``"me"`` for the phone owner).  ``ticks``
    only means anything on outgoing messages; ``seen`` is
    the Instagram-style read state (and the "Seen" line under
    the last own message); ``reply_to`` names the ``id`` of
    the quoted message; ``reactions`` are emoji strings worn
    on the bubble.  ``caption`` rides an image, ``duration``
    a voice note."""

    id: str
    sender: str
    kind: str = "text"
    text: str = ""
    time: str = ""
    ticks: str = ""
    reply_to: str = ""
    reactions: list = field(default_factory=list)
    seen: bool = False
    caption: str = ""
    duration: str = ""

    def __post_init__(self) -> None:
        if self.kind not in MESSAGE_KINDS:
            raise ValueError(
                "message kind " + repr(self.kind)
                + " not in " + repr(MESSAGE_KINDS))
        if self.ticks and self.ticks not in TICK_STATES:
            raise ValueError(
                "ticks " + repr(self.ticks)
                + " not in " + repr(TICK_STATES))
        self.reactions = [str(r) for r in (self.reactions or [])]

    def to_dict(self) -> dict:
        return {"id": self.id, "sender": self.sender,
                "kind": self.kind, "text": self.text,
                "time": self.time, "ticks": self.ticks,
                "reply_to": self.reply_to,
                "reactions": list(self.reactions),
                "seen": self.seen, "caption": self.caption,
                "duration": self.duration}

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(id=str(data.get("id") or ""),
                   sender=str(data.get("sender") or ""),
                   kind=str(data.get("kind") or "text"),
                   text=str(data.get("text") or ""),
                   time=str(data.get("time") or ""),
                   ticks=str(data.get("ticks") or ""),
                   reply_to=str(data.get("reply_to") or ""),
                   reactions=list(data.get("reactions") or []),
                   seen=bool(data.get("seen")),
                   caption=str(data.get("caption") or ""),
                   duration=str(data.get("duration") or ""))


@dataclass
class DeviceChrome:
    """The phone around the chat: which handset language
    (``style``), the status-bar clock, battery and network,
    and the count badge the iPhone back button carries."""

    style: str = "android"
    clock: str = "9:41"
    battery: int = 84
    network: str = "4G"
    unread: int = 0

    def __post_init__(self) -> None:
        if self.style not in DEVICE_STYLES:
            raise ValueError(
                "device style " + repr(self.style)
                + " not in " + repr(DEVICE_STYLES))
        self.battery = max(0, min(100, int(self.battery)))
        self.unread = max(0, int(self.unread))

    def to_dict(self) -> dict:
        return {"style": self.style, "clock": self.clock,
                "battery": self.battery,
                "network": self.network, "unread": self.unread}

    @classmethod
    def from_dict(cls, data: dict) -> DeviceChrome:
        return cls(style=str(data.get("style") or "android"),
                   clock=str(data.get("clock") or "9:41"),
                   battery=int(data.get("battery") or 0),
                   network=str(data.get("network") or "4G"),
                   unread=int(data.get("unread") or 0))


@dataclass
class Screen:
    """One whole chat screen.  ``platform`` names a design
    in the registry (``whatsapp`` / ``instagram``); ``group``
    turns on the group choreography (member name colours,
    admin rows) for designs that carry one; ``title`` /
    ``subtitle`` are the header lines (contact name /
    presence, group name / member list)."""

    platform: str
    title: str
    participants: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    chrome: DeviceChrome = field(default_factory=DeviceChrome)
    subtitle: str = ""
    group: bool = False
    dark: bool = False

    def participant(self, pid: str) -> Participant | None:
        for person in self.participants:
            if person.id == pid:
                return person
        return None

    def message_by_id(self, mid: str) -> Message | None:
        for msg in self.messages:
            if msg.id == mid:
                return msg
        return None

    def to_dict(self) -> dict:
        return {"platform": self.platform,
                "title": self.title,
                "subtitle": self.subtitle,
                "group": self.group,
                "dark": self.dark,
                "participants": [p.to_dict()
                                  for p in self.participants],
                "messages": [m.to_dict() for m in self.messages],
                "chrome": self.chrome.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> Screen:
        return cls(
            platform=str(data.get("platform") or ""),
            title=str(data.get("title") or ""),
            subtitle=str(data.get("subtitle") or ""),
            group=bool(data.get("group")),
            dark=bool(data.get("dark")),
            participants=[Participant.from_dict(p)
                          for p in data.get("participants") or []],
            messages=[Message.from_dict(m)
                      for m in data.get("messages") or []],
            chrome=DeviceChrome.from_dict(
                data.get("chrome") or {}))
