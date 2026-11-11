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
    a voice note.  ``media`` is the real-media hand-off for
    an image/voice message: a local file path whose pixels
    (image) or samples (voice — the duration and waveform
    are MEASURED from it at render) compose into the
    screen.  Empty means the explicit no-media case, drawn
    as the placeholder scene / synthetic waveform.  The
    command form's ``attach:``/``last:`` tokens live only
    between the parser and the runtime, which resolves
    them to paths before render; the animation phase
    consumes this same field, so a screen built with real
    media today moves tomorrow with no schema break."""

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
    media: str = ""

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
                "duration": self.duration, "media": self.media}

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
                   duration=str(data.get("duration") or ""),
                   media=str(data.get("media") or ""))


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
    in the registry (``whatsapp`` / ``instagram`` /
    ``telegram`` / ``messenger`` / ``snapchat`` /
    ``imessage`` / ``sms``); ``group`` turns on the group
    choreography (member name colours, admin rows) for
    designs that carry one; ``title`` / ``subtitle`` are
    the header lines (contact name / presence, group name
    / member list).  ``variant`` names a design-level
    layout variant (iMessage: ``"sms"`` — the green-bubble
    Apple-to-Android form the census records); designs
    that carry no variant ignore it."""

    platform: str
    title: str
    participants: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    chrome: DeviceChrome = field(default_factory=DeviceChrome)
    subtitle: str = ""
    group: bool = False
    dark: bool = False
    variant: str = ""

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
                "variant": self.variant,
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
            variant=str(data.get("variant") or ""),
            participants=[Participant.from_dict(p)
                          for p in data.get("participants") or []],
            messages=[Message.from_dict(m)
                      for m in data.get("messages") or []],
            chrome=DeviceChrome.from_dict(
                data.get("chrome") or {}))


# ── Phase 2b: lists, status and calls (census §26) ───────
#
# The chat screen class above is one of four; the census's
# Phase 2 families are three more screen classes with their
# own models — a list is rows, a status is one full-screen
# medium with a progress model, a call is a caller with a
# state and a control cluster.  They share the organ's
# Participant / DeviceChrome data and the designs-as-data
# doctrine, never the chat renderer.

#: A list row's story-ring state (``""`` = no ring).
RING_STATES: tuple[str, ...] = ("seen", "unseen")

#: What a list row IS (census §4.7's Telegram row types;
#: the other lists use person/group).
LIST_ROW_TYPES: tuple[str, ...] = (
    "person", "group", "channel", "bot", "secret", "saved",
    "archived")

#: The last-message preview vocabulary of the list tools
#: (census §4.4's WhatsApp set + §4.6's Instagram set).
#: ``draft`` renders the red "Draft:" prefix; ``typing``
#: renders the typing line; the ``call_*`` kinds carry a
#: call's direction, medium and missed state in the kind
#: itself so a missed call is distinguishable without
#: relying on colour alone (census §26 exit proof).
LIST_PREVIEW_KINDS: tuple[str, ...] = (
    "text", "photo", "video", "audio", "voice", "gif",
    "sticker", "document", "location", "contact",
    "disappearing", "deleted", "status_reaction",
    "story_mention", "story_reaction", "story_reply",
    "reel_reaction", "typing", "draft",
    "call_in_voice", "call_out_voice", "call_missed_voice",
    "call_in_video", "call_out_video", "call_missed_video")

#: A list row's own-message state as the inbox prints it
#: (Instagram's Seen / Sent preview states, census §4.6).
LIST_ROW_STATES: tuple[str, ...] = ("sent", "delivered", "seen")

#: A list screen's flavour: a chat inbox, a call log, or
#: the WhatsApp Updates (status list) screen.
LIST_FLAVORS: tuple[str, ...] = ("chats", "calls", "status")


@dataclass
class FolderTab:
    """One chat-folder tab (census §4.7: up to eight
    folders with unread counts, one selected)."""

    label: str
    unread: int = 0

    def to_dict(self) -> dict:
        return {"label": self.label, "unread": self.unread}

    @classmethod
    def from_dict(cls, data: dict) -> FolderTab:
        return cls(label=str(data.get("label") or ""),
                   unread=int(data.get("unread") or 0))


@dataclass
class StoryItem:
    """One entry of a stories / notes row above a list:
    a Telegram story ring (census §4.7) or an Instagram
    note bubble over an avatar (census §4.6).  ``note`` is
    the note text; empty draws the ring alone."""

    participant_id: str
    label: str = ""
    ring: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.ring and self.ring not in RING_STATES:
            raise ValueError(
                "ring " + repr(self.ring)
                + " not in " + repr(RING_STATES))

    def to_dict(self) -> dict:
        return {"participant_id": self.participant_id,
                "label": self.label, "ring": self.ring,
                "note": self.note}

    @classmethod
    def from_dict(cls, data: dict) -> StoryItem:
        return cls(
            participant_id=str(data.get("participant_id") or ""),
            label=str(data.get("label") or ""),
            ring=str(data.get("ring") or ""),
            note=str(data.get("note") or ""))


@dataclass
class ListRow:
    """One row of a list screen.  ``participant_id`` keys
    the avatar (resolved against the screen's
    participants); ``name`` is the row's own title (a
    channel or Saved Messages wears no participant).
    ``sender`` is a group row's last-message sender
    (Telegram prints it before the preview).  ``segments``
    is a status row's ring segment count.  ``call_type``
    marks a call-log row's medium (voice / video)."""

    id: str
    name: str
    participant_id: str = ""
    preview: str = ""
    preview_kind: str = "text"
    time: str = ""
    unread: int = 0
    pinned: bool = False
    muted: bool = False
    ring: str = ""
    ticks: str = ""
    status: str = ""
    row_type: str = "person"
    sender: str = ""
    verified: bool = False
    online: bool = False
    mention: bool = False
    draft: bool = False
    section: str = ""
    segments: int = 1
    call_type: str = ""

    def __post_init__(self) -> None:
        if self.preview_kind not in LIST_PREVIEW_KINDS:
            raise ValueError(
                "preview kind " + repr(self.preview_kind)
                + " not in " + repr(LIST_PREVIEW_KINDS))
        if self.ring and self.ring not in RING_STATES:
            raise ValueError(
                "ring " + repr(self.ring)
                + " not in " + repr(RING_STATES))
        if self.ticks and self.ticks not in TICK_STATES:
            raise ValueError(
                "ticks " + repr(self.ticks)
                + " not in " + repr(TICK_STATES))
        if self.status and self.status not in LIST_ROW_STATES:
            raise ValueError(
                "row status " + repr(self.status)
                + " not in " + repr(LIST_ROW_STATES))
        if self.row_type not in LIST_ROW_TYPES:
            raise ValueError(
                "row type " + repr(self.row_type)
                + " not in " + repr(LIST_ROW_TYPES))
        if self.call_type and self.call_type not in (
                "voice", "video"):
            raise ValueError(
                "call type " + repr(self.call_type)
                + " not in ('voice', 'video')")
        self.unread = max(0, int(self.unread))
        self.segments = max(1, int(self.segments))

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name,
                "participant_id": self.participant_id,
                "preview": self.preview,
                "preview_kind": self.preview_kind,
                "time": self.time, "unread": self.unread,
                "pinned": self.pinned, "muted": self.muted,
                "ring": self.ring, "ticks": self.ticks,
                "status": self.status,
                "row_type": self.row_type,
                "sender": self.sender,
                "verified": self.verified,
                "online": self.online,
                "mention": self.mention, "draft": self.draft,
                "section": self.section,
                "segments": self.segments,
                "call_type": self.call_type}

    @classmethod
    def from_dict(cls, data: dict) -> ListRow:
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            participant_id=str(data.get("participant_id") or ""),
            preview=str(data.get("preview") or ""),
            preview_kind=str(data.get("preview_kind") or "text"),
            time=str(data.get("time") or ""),
            unread=int(data.get("unread") or 0),
            pinned=bool(data.get("pinned")),
            muted=bool(data.get("muted")),
            ring=str(data.get("ring") or ""),
            ticks=str(data.get("ticks") or ""),
            status=str(data.get("status") or ""),
            row_type=str(data.get("row_type") or "person"),
            sender=str(data.get("sender") or ""),
            verified=bool(data.get("verified")),
            online=bool(data.get("online")),
            mention=bool(data.get("mention")),
            draft=bool(data.get("draft")),
            section=str(data.get("section") or ""),
            segments=int(data.get("segments") or 1),
            call_type=str(data.get("call_type") or ""))


@dataclass
class ListScreen:
    """One whole list screen: a chat inbox, a call log or
    the status (Updates) list for a platform design in
    ``LIST_DESIGNS``.  ``tabs`` are folder/filter tabs;
    ``stories`` the stories-or-notes row above the rows;
    ``archived`` is the Archived Chats row's unread count
    (-1 = no archived row, census §4.7); ``nav_badge`` is
    the bottom-navigation badge (WhatsApp's missed-call
    count, census §4.4)."""

    platform: str
    title: str
    rows: list = field(default_factory=list)
    participants: list = field(default_factory=list)
    chrome: DeviceChrome = field(default_factory=DeviceChrome)
    subtitle: str = ""
    flavor: str = "chats"
    dark: bool = False
    tabs: list = field(default_factory=list)
    tab_selected: int = 0
    stories: list = field(default_factory=list)
    archived: int = -1
    nav_badge: int = 0
    search: str = ""

    def __post_init__(self) -> None:
        if self.flavor not in LIST_FLAVORS:
            raise ValueError(
                "list flavor " + repr(self.flavor)
                + " not in " + repr(LIST_FLAVORS))
        self.nav_badge = max(0, int(self.nav_badge))

    def participant(self, pid: str) -> Participant | None:
        for person in self.participants:
            if person.id == pid:
                return person
        return None

    def to_dict(self) -> dict:
        return {"platform": self.platform,
                "title": self.title,
                "subtitle": self.subtitle,
                "flavor": self.flavor, "dark": self.dark,
                "rows": [r.to_dict() for r in self.rows],
                "participants": [p.to_dict()
                                 for p in self.participants],
                "tabs": [t.to_dict() for t in self.tabs],
                "tab_selected": self.tab_selected,
                "stories": [s.to_dict() for s in self.stories],
                "archived": self.archived,
                "nav_badge": self.nav_badge,
                "search": self.search,
                "chrome": self.chrome.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> ListScreen:
        return cls(
            platform=str(data.get("platform") or ""),
            title=str(data.get("title") or ""),
            subtitle=str(data.get("subtitle") or ""),
            flavor=str(data.get("flavor") or "chats"),
            dark=bool(data.get("dark")),
            rows=[ListRow.from_dict(r)
                  for r in data.get("rows") or []],
            participants=[Participant.from_dict(p)
                          for p in data.get("participants") or []],
            tabs=[FolderTab.from_dict(t)
                  for t in data.get("tabs") or []],
            tab_selected=int(data.get("tab_selected") or 0),
            stories=[StoryItem.from_dict(s)
                     for s in data.get("stories") or []],
            archived=int(data.get("archived", -1)
                         if data.get("archived") is not None
                         else -1),
            nav_badge=int(data.get("nav_badge") or 0),
            search=str(data.get("search") or ""),
            chrome=DeviceChrome.from_dict(
                data.get("chrome") or {}))


def _hex_color(value: str, field_name: str) -> str:
    """Validate a ``#rrggbb`` colour string (schema data
    the renderer parses without a guard otherwise)."""
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
            return text
        except ValueError:
            pass
    raise ValueError(
        field_name + " must be #rrggbb, got " + repr(value))


#: A status screen's medium (census §4.2: a photo status
#: or a text status on a colour).
STATUS_KINDS: tuple[str, ...] = ("photo", "text")


@dataclass
class StatusScreen:
    """One WhatsApp Status screen (census §4.2): the
    owner's status full-screen — a photo (``media`` is the
    real-picture hand-off, composed through the media
    module like a chat photo) or a text on a colour
    (``bg`` one of the design's stocked colours or a
    custom ``#rrggbb``; ``text_size`` in the census's
    14–60 px range) — with the progress model (``segments``
    updates, ``index`` playing, ``played`` the fraction of
    it shown), the time line (``when`` + optional
    ``label``) and the reply bar (quick ``reactions``, the
    like button, ``liked`` wearing the green heart).
    ``fit`` is the photo's fit-or-fill; ``surround`` is
    what an iPhone's space around a fitted photo takes —
    the photo's colours or black."""

    owner: Participant
    kind: str = "photo"
    media: str = ""
    text: str = ""
    caption: str = ""
    bg: str = ""
    fit: str = "fill"
    surround: str = "black"
    text_size: float = 24.0
    when: str = ""
    label: str = ""
    segments: int = 1
    index: int = 0
    played: float = 0.0
    reply_bar: bool = True
    reply_text: str = ""
    reactions: list = field(default_factory=list)
    like: bool = True
    liked: bool = False
    reshare: bool = False
    reply_style: str = ""
    chrome: DeviceChrome = field(default_factory=DeviceChrome)
    dark: bool = False
    platform: str = "whatsapp"

    def __post_init__(self) -> None:
        if self.kind not in STATUS_KINDS:
            raise ValueError(
                "status kind " + repr(self.kind)
                + " not in " + repr(STATUS_KINDS))
        if self.fit not in ("fit", "fill"):
            raise ValueError(
                "status fit " + repr(self.fit)
                + " not in ('fit', 'fill')")
        if self.surround not in ("black", "photo"):
            raise ValueError(
                "status surround " + repr(self.surround)
                + " not in ('black', 'photo')")
        if self.reply_style not in ("", "glass", "classic"):
            raise ValueError(
                "reply style " + repr(self.reply_style)
                + " not in ('', 'glass', 'classic')")
        self.bg = _hex_color(self.bg, "status bg")
        self.segments = max(1, int(self.segments))
        self.index = max(0, min(self.segments - 1,
                                int(self.index)))
        self.played = max(0.0, min(1.0, float(self.played)))
        self.text_size = max(14.0, min(60.0,
                                       float(self.text_size)))
        self.reactions = [str(r) for r in (self.reactions or [])]

    def to_dict(self) -> dict:
        return {"owner": self.owner.to_dict(),
                "kind": self.kind, "media": self.media,
                "text": self.text, "caption": self.caption,
                "bg": self.bg, "fit": self.fit,
                "surround": self.surround,
                "text_size": self.text_size,
                "when": self.when, "label": self.label,
                "segments": self.segments,
                "index": self.index, "played": self.played,
                "reply_bar": self.reply_bar,
                "reply_text": self.reply_text,
                "reactions": list(self.reactions),
                "like": self.like, "liked": self.liked,
                "reshare": self.reshare,
                "reply_style": self.reply_style,
                "dark": self.dark, "platform": self.platform,
                "chrome": self.chrome.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> StatusScreen:
        return cls(
            owner=Participant.from_dict(data.get("owner") or {}),
            kind=str(data.get("kind") or "photo"),
            media=str(data.get("media") or ""),
            text=str(data.get("text") or ""),
            caption=str(data.get("caption") or ""),
            bg=str(data.get("bg") or ""),
            fit=str(data.get("fit") or "fill"),
            surround=str(data.get("surround") or "black"),
            text_size=float(data.get("text_size") or 24.0),
            when=str(data.get("when") or ""),
            label=str(data.get("label") or ""),
            segments=int(data.get("segments") or 1),
            index=int(data.get("index") or 0),
            played=float(data.get("played") or 0.0),
            reply_bar=bool(data.get("reply_bar", True)),
            reply_text=str(data.get("reply_text") or ""),
            reactions=list(data.get("reactions") or []),
            like=bool(data.get("like", True)),
            liked=bool(data.get("liked")),
            reshare=bool(data.get("reshare")),
            reply_style=str(data.get("reply_style") or ""),
            dark=bool(data.get("dark")),
            platform=str(data.get("platform") or "whatsapp"),
            chrome=DeviceChrome.from_dict(
                data.get("chrome") or {}))


@dataclass
class StatusViewer:
    """One viewer on a status views sheet (census §4.1:
    the owner's own status, its view count and viewers).
    ``participant_id`` keys the avatar against the
    screen's participants."""

    name: str
    participant_id: str = ""
    when: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name,
                "participant_id": self.participant_id,
                "when": self.when}

    @classmethod
    def from_dict(cls, data: dict) -> StatusViewer:
        return cls(
            name=str(data.get("name") or ""),
            participant_id=str(data.get("participant_id") or ""),
            when=str(data.get("when") or ""))


@dataclass
class StatusViewsScreen:
    """The WhatsApp Status Views screen (census §4.1): my
    status at the top, the views sheet below it — the
    view count and one row per viewer."""

    owner: Participant
    when: str = ""
    viewers: list = field(default_factory=list)
    participants: list = field(default_factory=list)
    chrome: DeviceChrome = field(default_factory=DeviceChrome)
    dark: bool = False
    platform: str = "whatsapp"

    def participant(self, pid: str) -> Participant | None:
        for person in self.participants:
            if person.id == pid:
                return person
        return None

    def to_dict(self) -> dict:
        return {"owner": self.owner.to_dict(),
                "when": self.when,
                "viewers": [v.to_dict() for v in self.viewers],
                "participants": [p.to_dict()
                                 for p in self.participants],
                "dark": self.dark, "platform": self.platform,
                "chrome": self.chrome.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> StatusViewsScreen:
        return cls(
            owner=Participant.from_dict(data.get("owner") or {}),
            when=str(data.get("when") or ""),
            viewers=[StatusViewer.from_dict(v)
                     for v in data.get("viewers") or []],
            participants=[Participant.from_dict(p)
                          for p in data.get("participants") or []],
            dark=bool(data.get("dark")),
            platform=str(data.get("platform") or "whatsapp"),
            chrome=DeviceChrome.from_dict(
                data.get("chrome") or {}))


#: A call screen's state (census §6.2: incoming or
#: ongoing; the status line carries the timer / ringing
#: text).
CALL_STATES: tuple[str, ...] = ("incoming", "ongoing")

#: The in-call control vocabulary (census §6.2's "in-call
#: buttons"; each design's data names its default cluster
#: per state, a screen may name its own from this set).
CALL_CONTROLS: tuple[str, ...] = (
    "mute", "keypad", "speaker", "video", "message", "add",
    "remind", "decline", "accept", "end")


@dataclass
class CallScreen:
    """One call screen (census §6): a caller, a state, a
    status line (a timer while ongoing, the ringing line
    while incoming) and a control cluster.  ``media`` is
    the caller-picture hand-off — the Android picture and
    a video call's feed (composed through the media
    module); no picture wears the letter avatar in the
    caller's colour (``caller.color`` or the design's
    deterministic pick).  ``sim`` is the iPhone SIM label.
    ``controls`` empty = the design's default cluster for
    the state."""

    platform: str = "generic"
    caller: Participant = field(
        default_factory=lambda: Participant("", ""))
    number: str = ""
    state: str = "ongoing"
    status: str = ""
    video: bool = False
    media: str = ""
    sim: str = ""
    controls: list = field(default_factory=list)
    chrome: DeviceChrome = field(default_factory=DeviceChrome)
    dark: bool = False

    def __post_init__(self) -> None:
        if self.state not in CALL_STATES:
            raise ValueError(
                "call state " + repr(self.state)
                + " not in " + repr(CALL_STATES))
        for control in self.controls or []:
            if control not in CALL_CONTROLS:
                raise ValueError(
                    "call control " + repr(control)
                    + " not in " + repr(CALL_CONTROLS))
        self.controls = [str(c) for c in (self.controls or [])]

    def display_name(self) -> str:
        return self.caller.name or self.number or "Unknown"

    def to_dict(self) -> dict:
        return {"platform": self.platform,
                "caller": self.caller.to_dict(),
                "number": self.number, "state": self.state,
                "status": self.status, "video": self.video,
                "media": self.media, "sim": self.sim,
                "controls": list(self.controls),
                "dark": self.dark,
                "chrome": self.chrome.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> CallScreen:
        return cls(
            platform=str(data.get("platform") or "generic"),
            caller=Participant.from_dict(
                data.get("caller") or {}),
            number=str(data.get("number") or ""),
            state=str(data.get("state") or "ongoing"),
            status=str(data.get("status") or ""),
            video=bool(data.get("video")),
            media=str(data.get("media") or ""),
            sim=str(data.get("sim") or ""),
            controls=list(data.get("controls") or []),
            dark=bool(data.get("dark")),
            chrome=DeviceChrome.from_dict(
                data.get("chrome") or {}))


#: An iPhone call-history entry's direction/type (census
#: §6.3; a missed entry is the red one — and the record
#: carries the type so it never relies on colour alone).
HISTORY_TYPES: tuple[str, ...] = ("incoming", "outgoing", "missed")

#: How a history entry's avatar is worn (census §6.3:
#: default, letter or picture).
HISTORY_AVATAR_MODES: tuple[str, ...] = (
    "default", "letter", "picture")


@dataclass
class CallHistoryEntry:
    """One Recents row (census §6.3): the caller, the
    call type, the avatar mode, the SIM, the label under
    the name (mobile / home / FaceTime Audio), the number
    of calls and the time label (a clock time, a date,
    Yesterday or a weekday)."""

    id: str
    name: str
    participant_id: str = ""
    type: str = "incoming"
    avatar_mode: str = "letter"
    sim: str = ""
    label: str = ""
    count: int = 1
    when: str = ""

    def __post_init__(self) -> None:
        if self.type not in HISTORY_TYPES:
            raise ValueError(
                "history type " + repr(self.type)
                + " not in " + repr(HISTORY_TYPES))
        if self.avatar_mode not in HISTORY_AVATAR_MODES:
            raise ValueError(
                "avatar mode " + repr(self.avatar_mode)
                + " not in " + repr(HISTORY_AVATAR_MODES))
        self.count = max(1, int(self.count))

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name,
                "participant_id": self.participant_id,
                "type": self.type,
                "avatar_mode": self.avatar_mode,
                "sim": self.sim, "label": self.label,
                "count": self.count, "when": self.when}

    @classmethod
    def from_dict(cls, data: dict) -> CallHistoryEntry:
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            participant_id=str(data.get("participant_id") or ""),
            type=str(data.get("type") or "incoming"),
            avatar_mode=str(data.get("avatar_mode") or "letter"),
            sim=str(data.get("sim") or ""),
            label=str(data.get("label") or ""),
            count=int(data.get("count") or 1),
            when=str(data.get("when") or ""))


@dataclass
class CallHistoryScreen:
    """The iPhone Call History (Recents) screen (census
    §6.3): editable header texts (Edit / All / Missed /
    the title), a search field, the entries, and the
    footer tab bar (Favourites / Calls / Contacts /
    Keypad / Voicemail) with the voicemail count worn as
    its badge.  iPhone chrome only — the tool is the
    iPhone generator."""

    entries: list = field(default_factory=list)
    participants: list = field(default_factory=list)
    chrome: DeviceChrome = field(
        default_factory=lambda: DeviceChrome(style="iphone"))
    edit_text: str = "Edit"
    all_text: str = "All"
    missed_text: str = "Missed"
    title_text: str = "Recents"
    search_text: str = ""
    tabs: tuple = ("Favourites", "Calls", "Contacts",
                   "Keypad", "Voicemail")
    active_tab: str = "Calls"
    voicemail_count: int = 0
    dark: bool = False
    platform: str = "iphone_history"

    def __post_init__(self) -> None:
        if self.chrome.style != "iphone":
            raise ValueError(
                "call history wears iPhone chrome, got "
                + repr(self.chrome.style))
        self.tabs = tuple(str(t) for t in (self.tabs or ()))
        self.voicemail_count = max(0, int(self.voicemail_count))

    def participant(self, pid: str) -> Participant | None:
        for person in self.participants:
            if person.id == pid:
                return person
        return None

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries],
                "participants": [p.to_dict()
                                 for p in self.participants],
                "edit_text": self.edit_text,
                "all_text": self.all_text,
                "missed_text": self.missed_text,
                "title_text": self.title_text,
                "search_text": self.search_text,
                "tabs": list(self.tabs),
                "active_tab": self.active_tab,
                "voicemail_count": self.voicemail_count,
                "dark": self.dark, "platform": self.platform,
                "chrome": self.chrome.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> CallHistoryScreen:
        return cls(
            entries=[CallHistoryEntry.from_dict(e)
                     for e in data.get("entries") or []],
            participants=[Participant.from_dict(p)
                          for p in data.get("participants") or []],
            edit_text=str(data.get("edit_text") or "Edit"),
            all_text=str(data.get("all_text") or "All"),
            missed_text=str(data.get("missed_text") or "Missed"),
            title_text=str(data.get("title_text") or "Recents"),
            search_text=str(data.get("search_text") or ""),
            tabs=tuple(data.get("tabs") or (
                "Favourites", "Calls", "Contacts", "Keypad",
                "Voicemail")),
            active_tab=str(data.get("active_tab") or "Calls"),
            voicemail_count=int(data.get("voicemail_count") or 0),
            dark=bool(data.get("dark")),
            platform=str(data.get("platform") or "iphone_history"),
            chrome=DeviceChrome.from_dict(
                data.get("chrome") or {}))
