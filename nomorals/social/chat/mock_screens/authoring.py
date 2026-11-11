"""Authoring for mock screens: built-in demos + the
compact structured command form.

NL authoring is a later phase; today a screen is authored
two structured ways:

- **Demos** — :func:`demo_screen` builds a small real-shape
  script per flagship family (the same scripts the visual
  proofs render).
- **The compact form** the ``/mock`` command parses
  (:func:`parse_mock_tail`)::

      /mock <family> [iphone|android] [dark] [with <Name>]
            <segment> ;; <segment> ;; …

  Families: whatsapp, telegram, messenger, snapchat,
  imessage, sms, instagram — each with a ``-group``
  spelling for the group form.

  Segments: ``me: words`` is outgoing; ``<Name>: words``
  is incoming from that participant; ``! words`` is a
  system row; ``# Today`` is a day divider.  Suffixes on a
  segment: ``[read]`` / ``[sent]`` / ``[delivered]`` set an
  outgoing message's ticks, ``[react ❤️ 🔥]`` hangs
  reactions on it, and a leading ``^`` makes the message a
  reply to the previous one.  Times are assigned on a
  steady clock from 9:41 AM unless a segment opens with a
  time (``9:58 Adaeze: words``).

  Media segments carry REAL media into the screen: a
  segment whose words open with a media marker —
  ``me: [photo]`` (the image attached to the command
  message itself), ``me: [photo <path>]`` (a file on disk,
  e.g. the ``saved:`` path an image generation reported),
  ``me: [photo last]`` (the newest generated image), or
  ``Adaeze: [voice]`` / ``[voice <path>]`` (an attached or
  on-disk voice note, drawn with its measured waveform
  and true duration).  Words after a ``[photo]`` marker
  are its caption; a voice segment carries no words.
  The parser stores the reference (or the ``attach:`` /
  ``last:`` token) on the message; the runtime resolves
  tokens to real paths before render.
"""

from __future__ import annotations

import re

from .demo_data import (
    COUNTRIES,
    CREATIVE_FAMILIES,
    MAIL_PROVIDERS,
    BulkGenerator,
    DemoDataError,
    bulk_call_history,
    bulk_chat_screen,
    bulk_list_screen,
    bulk_status_views,
    export_records,
)
from .render import MockScreensError
from .schema import (
    CallHistoryEntry,
    CallHistoryScreen,
    CallScreen,
    CommentScreen,
    DeviceChrome,
    FolderTab,
    ListRow,
    ListScreen,
    Message,
    Participant,
    PostScreen,
    ProfileCount,
    ProfileItem,
    ProfileScreen,
    Screen,
    SocialComment,
    SocialProfile,
    StatusScreen,
    StatusViewer,
    StatusViewsScreen,
    StoryItem,
    StoryScreen,
    StorySticker,
    UpNextItem,
    VideoScreen,
)

USAGE = (
    "usage: /mock demo [whatsapp|group|instagram|telegram|"
    "telegram-group|messenger|messenger-group|snapchat|"
    "snapchat-group|imessage|imessage-sms|sms] "
    "[iphone|android] — screen demos: /mock demo "
    "whatsapp-list · whatsapp-calls · whatsapp-status-list "
    "· instagram-list · telegram-list · snapchat-list · "
    "whatsapp-status · whatsapp-status-text · "
    "whatsapp-status-views · call · call-incoming · "
    "whatsapp-call · facebook-call · instagram-call · "
    "hangouts-call · iphone-call-history · facebook-post · "
    "instagram-post · threads-post · x-tweet · tiktok-post "
    "· facebook-profile · instagram-profile · x-profile · "
    "tiktok-profile · youtube-channel · instagram-story · "
    "instagram-story-text · tiktok-comment · youtube-video "
    "· youtube-video-comments · youtube-video-description "
    "— bulk: /mock demo bulk-list [whatsapp|instagram|"
    "telegram|snapchat] [count] [seed N] [flavor chats|"
    "calls|status] [iphone|android] [dark] · bulk-calls · "
    "bulk-status-list · bulk-history [count] [seed N] · "
    "bulk-chat [platform] [count] [seed N] [group] · "
    "bulk-views [count] [seed N] — data: /mock data "
    "[identity|name|username|password|email|phone|"
    "address|company|imei|creative <family>] [count] "
    "[seed N] [json|csv|xml|text]"
    " — or build one: "
    "/mock whatsapp "
    "[iphone|android] [with <Name>] me: hi ;; Adaeze: hello "
    ";; me: on my way [read] — families: whatsapp · "
    "telegram · messenger · snapchat · imessage · sms · "
    "instagram (add -group for the group form) — media: "
    "me: [photo] uses an image attached to your message, "
    "[photo <path>] a file on disk, [photo last] your "
    "newest generated image; [voice] / [voice <path>] a "
    "voice note with its real waveform and duration; "
    "words after [photo] are its caption — record one as "
    "video: /mock record demo whatsapp · /mock record "
    "whatsapp me: hi ;; Adaeze: hello")

_PLATFORMS = {
    "whatsapp": ("whatsapp", False),
    "wa": ("whatsapp", False),
    "group": ("whatsapp", True),
    "whatsapp-group": ("whatsapp", True),
    "wag": ("whatsapp", True),
    "instagram": ("instagram", False),
    "ig": ("instagram", False),
    "instagram-group": ("instagram", True),
    "igg": ("instagram", True),
    "telegram": ("telegram", False),
    "tg": ("telegram", False),
    "telegram-group": ("telegram", True),
    "tgg": ("telegram", True),
    "messenger": ("messenger", False),
    "msgr": ("messenger", False),
    "messenger-group": ("messenger", True),
    "msgg": ("messenger", True),
    "facebook": ("messenger", True),
    "snapchat": ("snapchat", False),
    "snap": ("snapchat", False),
    "snapchat-group": ("snapchat", True),
    "snapg": ("snapchat", True),
    "imessage": ("imessage", False),
    "imsg": ("imessage", False),
    "imessage-group": ("imessage", True),
    "sms": ("sms", False),
    "android-sms": ("sms", False),
    "sms-group": ("sms", True),
}

#: demo-only names selecting a design variant (the
#: green-bubble Apple-to-Android iMessage form)
_DEMO_VARIANTS = {"imessage-sms": "sms", "imessage-green": "sms"}

_TICK_SUFFIX = re.compile(r"\[(read|sent|delivered|sending)\]\s*$")
_REACT_SUFFIX = re.compile(r"\[react\s+([^\]]*)\]\s*$")
_TIME_PREFIX = re.compile(
    r"^(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\s+(.*)$", re.DOTALL)
#: a media marker opening a segment's words: [photo],
#: [photo <path>], [photo last], [voice], [voice <path>]
#: ([image] / [audio] are the same markers by other names)
_MEDIA_MARKER = re.compile(
    r"\[(photo|image|voice|audio)(?:\s+([^\]]+))?\]\s*",
    re.IGNORECASE)


def demo_screen(name: str = "whatsapp", style: str = "android",
                *, dark: bool = False) -> Screen:
    """One built-in demo script per flagship family."""
    key = str(name or "whatsapp").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41",
                          battery=84, network="4G",
                          unread=3 if style == "iphone" else 0)
    if key in ("group", "whatsapp-group", "wag"):
        return Screen(
            platform="whatsapp", group=True, dark=dark,
            title="Think Creativity Think Peace",
            subtitle="Adaeze, Bola, Chiamaka, you",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze", admin=True),
                Participant("bola", "Bola"),
                Participant("chiamaka", "Chiamaka"),
            ],
            messages=[
                Message("g1", "system", kind="system",
                        text="Adaeze created the group"),
                Message("g2", "system", kind="system",
                        text="Bola added Chiamaka"),
                Message("g3", "adaeze",
                        text="Morning everyone. The studio is "
                             "booked for Saturday, 10 AM.",
                        time="9:12 AM"),
                Message("g4", "bola",
                        text="Perfect. I am bringing the new "
                             "backdrops.",
                        time="9:15 AM", reactions=["🔥"]),
                Message("g5", "chiamaka",
                        text="I will handle the guest list and "
                             "the drinks.",
                        time="9:16 AM"),
                Message("g6", "me",
                        text="Love it. I will post the flyer "
                             "here once it is ready.",
                        time="9:20 AM", ticks="read"),
                Message("g7", "adaeze",
                        text="Flyer goes out by noon, no "
                             "extensions.",
                        time="9:21 AM"),
            ])
    if key in ("instagram", "ig"):
        return Screen(
            platform="instagram", dark=dark,
            title="adaeze.creates",
            subtitle="Active now",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze"),
            ],
            messages=[
                Message("i1", "adaeze",
                        text="Your last post came out so clean",
                        time="9:02 AM"),
                Message("i2", "adaeze",
                        text="What did you grade it with?",
                        time="9:02 AM", reactions=["😍"]),
                Message("i3", "me",
                        text="Just the phone and patience, "
                             "honestly",
                        time="9:05 AM"),
                Message("i4", "me", kind="image",
                        caption="the final frame",
                        time="9:06 AM"),
                Message("i5", "adaeze",
                        text="Okay this is the one. Post it.",
                        time="9:08 AM"),
                Message("i6", "me",
                        text="Posting tonight at 7",
                        time="9:09 AM", seen=True),
            ])
    if key in ("instagram-group", "igg"):
        return Screen(
            platform="instagram", group=True, dark=dark,
            title="studio.crew",
            subtitle="3 members",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze"),
                Participant("bola", "Bola"),
            ],
            messages=[
                Message("ig1", "adaeze",
                        text="Studio is booked for Saturday",
                        time="9:12 AM"),
                Message("ig2", "bola",
                        text="Bringing the backdrops",
                        time="9:15 AM", reactions=["🔥"]),
                Message("ig3", "me",
                        text="Flyer drops at noon",
                        time="9:20 AM", seen=True),
            ])
    if key in ("telegram-group", "tgg"):
        return Screen(
            platform="telegram", group=True, dark=dark,
            title="Think Creativity Think Peace",
            subtitle="24 members, 5 online",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze", admin=True),
                Participant("bola", "Bola"),
                Participant("chiamaka", "Chiamaka"),
            ],
            messages=[
                Message("tg0", "system", kind="system",
                        text="Adaeze created the group"),
                Message("tgd", "system", kind="divider",
                        text="Today"),
                Message("tg1", "adaeze",
                        text="Studio is booked for Saturday, "
                             "10 AM sharp",
                        time="9:12 AM"),
                Message("tg2", "bola",
                        text="Bringing the new backdrops and "
                             "the ring light",
                        time="9:15 AM", reactions=["👍"]),
                Message("tg3", "me", reply_to="tg1",
                        text="I will post the flyer here once "
                             "it is ready",
                        time="9:20 AM", ticks="read"),
                Message("tg4", "chiamaka",
                        text="Guest list and drinks are on me",
                        time="9:21 AM"),
                Message("tg5", "adaeze",
                        text="Flyer goes out by noon. "
                             "No extensions.",
                        time="9:22 AM"),
            ])
    if key in ("telegram", "tg"):
        return Screen(
            platform="telegram", dark=dark,
            title="Adaeze Nwosu",
            subtitle="online",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze Nwosu"),
            ],
            messages=[
                Message("td", "system", kind="divider",
                        text="Today"),
                Message("t1", "adaeze",
                        text="Morning! Did the renders finish?",
                        time="9:38 AM"),
                Message("t2", "me",
                        text="They did. Sending the export now",
                        time="9:40 AM", ticks="read"),
                Message("t3", "me", kind="image",
                        caption="final_export.png",
                        time="9:41 AM", ticks="read"),
                Message("t4", "adaeze", reply_to="t2",
                        text="This is exactly the look 🔥",
                        time="9:42 AM", reactions=["❤️"]),
                Message("t5", "me",
                        text="Told you the teal would win",
                        time="9:43 AM", ticks="delivered"),
            ])
    if key in ("messenger-group", "msgg", "facebook"):
        return Screen(
            platform="messenger", group=True, dark=dark,
            title="Studio Crew",
            subtitle="Active now",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze"),
                Participant("bola", "Bola"),
                Participant("chiamaka", "Chiamaka"),
            ],
            messages=[
                Message("mg1", "adaeze",
                        text="Studio is booked for Saturday",
                        time="9:12 AM"),
                Message("mg2", "bola",
                        text="Bringing the backdrops",
                        time="9:15 AM", reactions=["❤️"]),
                Message("mg3", "me",
                        text="Flyer drops at noon",
                        time="9:20 AM", seen=True),
                Message("mg4", "chiamaka",
                        text="Guest list is on me",
                        time="9:21 AM"),
            ])
    if key in ("messenger", "msgr"):
        return Screen(
            platform="messenger", dark=dark,
            title="Adaeze Nwosu",
            subtitle="Active now",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze Nwosu"),
            ],
            messages=[
                Message("mm1", "adaeze",
                        text="Your last post came out so clean 😍",
                        time="9:02 AM"),
                Message("mm2", "me",
                        text="Just the phone and patience, "
                             "honestly",
                        time="9:05 AM"),
                Message("mm3", "me", kind="image",
                        caption="the final frame",
                        time="9:06 AM"),
                Message("mm4", "adaeze",
                        text="Okay this is the one. Post it!",
                        time="9:08 AM"),
                Message("mm5", "me",
                        text="Posting tonight at 7",
                        time="9:09 AM", seen=True),
            ])
    if key in ("snapchat-group", "snapg"):
        return Screen(
            platform="snapchat", group=True, dark=dark,
            title="Studio Crew 🎬",
            subtitle="Adaeze, Bola, Chiamaka",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze"),
                Participant("bola", "Bola"),
                Participant("chiamaka", "Chiamaka"),
            ],
            messages=[
                Message("scd", "system", kind="divider",
                        text="Today"),
                Message("sc1", "adaeze",
                        text="studio day lets gooo"),
                Message("sc2", "bola", kind="image",
                        caption="the setup so far"),
                Message("sc3", "chiamaka",
                        text="omw with the drinks 🧃"),
                Message("sc4", "me",
                        text="flyer drops at noon, be ready",
                        ticks="read"),
            ])
    if key in ("snapchat", "snap"):
        return Screen(
            platform="snapchat", dark=dark,
            title="Adaeze",
            subtitle="@adaeze.creates",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze"),
            ],
            messages=[
                Message("snd", "system", kind="divider",
                        text="Today"),
                Message("sn1", "adaeze",
                        text="morning!! did you see the snaps "
                             "from last night"),
                Message("sn2", "adaeze", kind="image"),
                Message("sn3", "me",
                        text="the rooftop ones?? unreal 😭",
                        ticks="delivered"),
                Message("sn4", "me",
                        text="saving that one forever",
                        ticks="read"),
            ])
    if key in ("imessage-group",):
        return Screen(
            platform="imessage", group=True, dark=dark,
            title="Studio Crew",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze"),
                Participant("bola", "Bola"),
            ],
            messages=[
                Message("gi1", "adaeze",
                        text="Studio is booked for Saturday",
                        time="Today 9:12 AM"),
                Message("gi2", "bola",
                        text="Bringing the backdrops",
                        time="Today 9:15 AM"),
                Message("gi3", "me",
                        text="Flyer drops at noon",
                        time="Today 9:20 AM", ticks="read"),
            ])
    if key in _DEMO_VARIANTS or key in ("imessage", "imsg"):
        return Screen(
            platform="imessage", dark=dark,
            variant=_DEMO_VARIANTS.get(key, ""),
            title="Adaeze Nwosu",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze Nwosu"),
            ],
            messages=[
                Message("im1", "adaeze",
                        text="Morning! Are we still on for "
                             "the shoot?",
                        time="Today 9:41 AM"),
                Message("im2", "me",
                        text="Yes — studio by 11. Do not be "
                             "late this time.",
                        time="Today 9:41 AM", ticks="delivered"),
                Message("im3", "adaeze",
                        text="I was late ONCE 😂",
                        time="Today 9:44 AM", reactions=["😂"]),
                Message("im4", "me",
                        text="And the client still tells "
                             "that story",
                        time="Today 9:44 AM", ticks="read"),
                Message("im5", "adaeze",
                        text="Fine. One story per meeting, "
                             "that is the deal",
                        time="Today 9:46 AM"),
            ])
    if key in ("sms-group",):
        return Screen(
            platform="sms", group=True, dark=dark,
            title="Studio Crew",
            subtitle="3 people",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze"),
                Participant("bola", "Bola"),
            ],
            messages=[
                Message("sg1", "adaeze",
                        text="Studio Saturday, 10 AM",
                        time="9:12 AM"),
                Message("sg2", "bola",
                        text="Backdrops on me",
                        time="9:15 AM"),
                Message("sg3", "me",
                        text="Flyer at noon",
                        time="9:20 AM"),
            ])
    if key in ("sms", "android-sms"):
        return Screen(
            platform="sms", dark=dark,
            title="Adaeze Nwosu",
            chrome=chrome,
            participants=[
                Participant("me", "You"),
                Participant("adaeze", "Adaeze Nwosu"),
            ],
            messages=[
                Message("sd", "system", kind="divider",
                        text="Today"),
                Message("s1", "adaeze",
                        text="Good morning! Are we still on "
                             "for the shoot today?",
                        time="9:41 AM"),
                Message("s2", "me",
                        text="Yes — studio by 11, do not "
                             "be late",
                        time="9:43 AM"),
                Message("s3", "adaeze", kind="image",
                        caption="the backdrop from last time",
                        time="9:47 AM"),
                Message("s4", "me",
                        text="Perfect. See you there!",
                        time="9:49 AM"),
            ])
    return Screen(
        platform="whatsapp", dark=dark,
        title="Adaeze Nwosu",
        subtitle="online",
        chrome=chrome,
        participants=[
            Participant("me", "You"),
            Participant("adaeze", "Adaeze Nwosu"),
        ],
        messages=[
            Message("m0", "system", kind="system",
                    text="Messages are end-to-end encrypted"),
            Message("m1", "system", kind="divider",
                    text="Today"),
            Message("m2", "adaeze",
                    text="Good morning! Are we still on for "
                         "the shoot today?",
                    time="9:41 AM"),
            Message("m3", "me",
                    text="Morning Adaeze. Yes — studio by 11, "
                         "do not be late this time.",
                    time="9:43 AM", ticks="read"),
            Message("m4", "adaeze",
                    text="I was late once. Once!",
                    time="9:44 AM", reactions=["😂"]),
            Message("m5", "me", reply_to="m4",
                    text="And the client still tells that "
                         "story at every meeting.",
                    time="9:45 AM", ticks="read"),
            Message("m6", "adaeze", kind="image",
                    caption="the backdrop from last time",
                    time="9:47 AM"),
            Message("m7", "adaeze", kind="voice",
                    duration="0:42", time="9:48 AM"),
            Message("m8", "me",
                    text="Perfect. See you there!",
                    time="9:49 AM", ticks="read"),
        ])


# ── Phase 2b demos: lists, status, calls ─────────────────
#
# One demo script per Phase 2b family, in the flagship
# demos' idiom: real-shape data exercising the states the
# census names for the family (the §26 exit proof: every
# row state has a fixture).

_LIST_DEMOS = (
    "whatsapp-list", "whatsapp-calls", "whatsapp-status-list",
    "instagram-list", "telegram-list", "snapchat-list")
_STATUS_DEMOS = ("whatsapp-status", "whatsapp-status-text")
_VIEWS_DEMOS = ("whatsapp-status-views",)
_CALL_DEMOS = (
    "call", "call-incoming", "whatsapp-call", "facebook-call",
    "instagram-call", "hangouts-call")
_HISTORY_DEMOS = ("iphone-call-history", "call-history")

#: Phase 3 demo names (census §27 build list).
_POST_DEMOS = (
    "facebook-post", "instagram-post", "threads-post",
    "x-tweet", "tiktok-post")
_PROFILE_DEMOS = (
    "facebook-profile", "instagram-profile", "x-profile",
    "tiktok-profile", "youtube-channel")
_STORY_DEMOS = ("instagram-story", "instagram-story-text")
_COMMENT_DEMOS = ("tiktok-comment",)
_VIDEO_DEMOS = (
    "youtube-video", "youtube-video-comments",
    "youtube-video-description")

#: Phase 5 bulk demo names (census §29): one seed +
#: count populates a whole screen.
_BULK_DEMOS = (
    "bulk-list", "bulk-calls", "bulk-status-list",
    "bulk-history", "bulk-chat", "bulk-views")

#: Which service door each bulk demo rides (the same
#: doors as the Phase 2b/3 screen classes).
_BULK_MODES = {
    "bulk-list": "demo-list",
    "bulk-calls": "demo-list",
    "bulk-status-list": "demo-list",
    "bulk-history": "demo-history",
    "bulk-chat": "demo",
    "bulk-views": "demo-views",
}

#: /mock data kinds (census §7 + §29).
_DATA_KINDS = (
    "identity", "name", "username", "password", "email",
    "phone", "address", "company", "imei", "creative")

#: The /mock data form answers in chat text, so its
#: count stays readable — volume belongs to the bulk
#: screen forms (and to export_records for code callers).
MAX_DATA_TEXT = 50


def _bulk_opts(tokens: list) -> dict:
    """Parse a bulk demo's option tokens: a bare integer
    is the count, ``seed N`` the seed, ``flavor F`` the
    list flavour, a platform name the platform, and
    iphone / android / dark / group the usual switches.
    Anything else is the usage error."""
    opts = {"platform": "whatsapp", "count": 25, "seed": 0,
            "flavor": "chats", "style": "android",
            "dark": False, "group": False}
    idx = 0
    while idx < len(tokens):
        tok = str(tokens[idx]).strip().lower()
        if tok == "iphone" or tok == "android":
            opts["style"] = tok
        elif tok == "dark":
            opts["dark"] = True
        elif tok == "group":
            opts["group"] = True
        elif tok == "seed":
            if idx + 1 >= len(tokens):
                raise MockScreensError(
                    "seed needs a number — " + USAGE)
            try:
                opts["seed"] = int(tokens[idx + 1])
            except ValueError:
                raise MockScreensError(
                    "seed needs a whole number, got "
                    + repr(tokens[idx + 1])) from None
            idx += 1
        elif tok == "flavor":
            if idx + 1 >= len(tokens):
                raise MockScreensError(
                    "flavor needs chats · calls · status — "
                    + USAGE)
            flavor = str(tokens[idx + 1]).strip().lower()
            if flavor not in ("chats", "calls", "status"):
                raise MockScreensError(
                    "flavor " + repr(flavor) + " is not one "
                    "of chats · calls · status")
            opts["flavor"] = flavor
            idx += 1
        elif tok in ("whatsapp", "instagram", "telegram",
                     "messenger", "snapchat", "imessage",
                     "sms"):
            opts["platform"] = tok
        elif tok.isdigit():
            opts["count"] = int(tok)
        else:
            raise MockScreensError(USAGE)
        idx += 1
    return opts


def _parse_bulk_demo(name: str, tokens: list):
    """One bulk demo name + its option tokens -> the
    populated screen (the renderer's own doors draw it).
    DemoDataError becomes the organ's usage error."""
    opts = _bulk_opts(tokens)
    try:
        if name == "bulk-history":
            return bulk_call_history(
                count=opts["count"], seed=opts["seed"])
        if name == "bulk-views":
            return bulk_status_views(
                count=opts["count"], seed=opts["seed"],
                style=opts["style"])
        if name == "bulk-chat":
            return bulk_chat_screen(
                opts["platform"], count=opts["count"],
                seed=opts["seed"], style=opts["style"],
                dark=opts["dark"], group=opts["group"])
        flavor = {"bulk-list": opts["flavor"],
                  "bulk-calls": "calls",
                  "bulk-status-list": "status"}[name]
        return bulk_list_screen(
            opts["platform"], count=opts["count"],
            seed=opts["seed"], flavor=flavor,
            style=opts["style"], dark=opts["dark"])
    except DemoDataError as exc:
        raise MockScreensError(str(exc)) from exc


def _parse_data_tail(text: str) -> str:
    """``/mock data <kind> [count] [seed N] [json|csv|
    xml|text] [country]`` -> the export text.  The chat
    form caps at 50 records (a reply must stay
    readable); volume is the bulk screens' job."""
    tokens = str(text or "").split()
    if not tokens:
        raise MockScreensError(USAGE)
    kind = tokens[0].strip().lower()
    if kind not in _DATA_KINDS:
        raise MockScreensError(
            "data kinds: identity · name · username · "
            "password · email · phone · address · company "
            "· imei · creative <family> — " + USAGE)
    count = 5
    seed = 0
    fmt = "json"
    country = "ng"
    provider = "gmail"
    family = ""
    idx = 1
    while idx < len(tokens):
        tok = str(tokens[idx]).strip().lower()
        if tok == "seed":
            if idx + 1 >= len(tokens):
                raise MockScreensError(
                    "seed needs a number — " + USAGE)
            try:
                seed = int(tokens[idx + 1])
            except ValueError:
                raise MockScreensError(
                    "seed needs a whole number, got "
                    + repr(tokens[idx + 1])) from None
            idx += 1
        elif tok in ("json", "csv", "xml", "text"):
            fmt = tok
        elif tok in COUNTRIES:
            country = tok
        elif tok in MAIL_PROVIDERS:
            provider = tok
        elif tok in CREATIVE_FAMILIES:
            family = tok
        elif tok.isdigit():
            count = int(tok)
        else:
            raise MockScreensError(USAGE)
        idx += 1
    if count < 1 or count > MAX_DATA_TEXT:
        raise MockScreensError(
            "the /mock data form carries 1–"
            + str(MAX_DATA_TEXT) + " records (a chat reply "
            "has to stay readable) — got " + str(count)
            + "; volume is the bulk screen demos' job")
    if kind == "creative" and not family:
        raise MockScreensError(
            "creative needs a family: "
            + ", ".join(CREATIVE_FAMILIES) + " — " + USAGE)
    gen = BulkGenerator(seed)
    try:
        if kind == "identity":
            records = gen.identities(count, country=country)
        elif kind == "name":
            records = [{"name": gen.name()}
                       for _ in range(count)]
        elif kind == "username":
            records = [{"username": gen.username()}
                       for _ in range(count)]
        elif kind == "password":
            records = [{"password": gen.password()}
                       for _ in range(count)]
        elif kind == "email":
            records = [gen.email(provider)
                       for _ in range(count)]
        elif kind == "phone":
            records = [gen.phone(country)
                       for _ in range(count)]
        elif kind == "address":
            records = [gen.address(country)
                       for _ in range(count)]
        elif kind == "company":
            records = [{"company": gen.company(country)}
                       for _ in range(count)]
        elif kind == "imei":
            records = [gen.imei() for _ in range(count)]
        else:  # creative
            records = [{"family": family,
                        "name": gen.creative_name(family)}
                       for _ in range(count)]
        return export_records(records, fmt)
    except DemoDataError as exc:
        raise MockScreensError(str(exc)) from exc


def _demo_social_author() -> SocialProfile:
    return SocialProfile(
        "adaeze", "Adaeze Nwosu", handle="adaeze.creates",
        verified=True)


def demo_post_screen(name: str, style: str = "android",
                     *, dark: bool = False):
    """One built-in demo per post family (census §5.2,
    §5.7, §5.10)."""
    key = str(name or "").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G")
    author = _demo_social_author()
    if key == "instagram-post":
        return PostScreen(
            platform="instagram", author=author, dark=dark,
            text="The studio, finally finished. Doors open "
                 "Saturday at 10 — bring your ideas.",
            time="2h", likes=2540, comments=128, shares=64,
            chrome=chrome)
    if key == "threads-post":
        return PostScreen(
            platform="threads", author=author, dark=dark,
            text="Building in public is the whole point. "
                 "Watch this space.",
            time="41m", likes=312, comments=27, shares=9,
            chrome=chrome)
    if key == "x-tweet":
        return PostScreen(
            platform="x", author=author, dark=dark,
            text="Gold holding above the zone. The setup "
                 "is the setup — no chasing.",
            time="9:41 AM", likes=12400, comments=342,
            shares=1200, chrome=chrome)
    if key == "tiktok-post":
        return PostScreen(
            platform="tiktok", author=author, dark=dark,
            text="POV: the render finally finishes 🎬",
            time="1h", likes=45200, comments=1204,
            shares=3300, video=True, chrome=chrome)
    return PostScreen(
        platform="facebook", author=author, dark=dark,
        text="Studio day! Doors open at 10 — bring your "
             "ideas, we bring the backdrops.",
        time="9:12 AM", visibility="public", likes=1240,
        comments=87, shares=32, chrome=chrome)


def demo_profile_screen(name: str, style: str = "android",
                        *, dark: bool = False):
    """One built-in demo per profile family (census §5.3,
    §5.6, §5.8, §5.10)."""
    key = str(name or "").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G")
    ident = _demo_social_author()
    if key == "instagram-profile":
        return ProfileScreen(
            platform="instagram", identity=ident, dark=dark,
            bio="Studio · Lagos · building in public",
            link="peacethefirst.studio",
            counts=[ProfileCount("posts", 128),
                    ProfileCount("followers", 25400),
                    ProfileCount("following", 312)],
            buttons=["Follow", "Message"],
            tabs=["Posts", "Reels", "Tagged"],
            items=[ProfileItem("ig" + str(i), "photo",
                               caption="studio " + str(i))
                   for i in range(1, 7)],
            chrome=chrome)
    if key == "x-profile":
        return ProfileScreen(
            platform="x", identity=ident, dark=dark,
            bio="Building Devon and GoldenSwing. "
                 "No consumers, only builders.",
            link="nomorals.ai",
            counts=[ProfileCount("following", 412),
                    ProfileCount("followers", 25400)],
            buttons=["Follow", "Message"],
            tabs=["Posts", "Replies", "Media", "Likes"],
            items=[
                ProfileItem(
                    "xp1", "post",
                    title="Gold holding above the zone. "
                          "The setup is the setup."),
                ProfileItem(
                    "xp2", "post",
                    title="The render finally finished. "
                          "Studio opens Saturday."),
            ],
            chrome=chrome)
    if key == "tiktok-profile":
        return ProfileScreen(
            platform="tiktok", identity=ident, dark=dark,
            bio="Studio days 🎬 · Lagos",
            link="peacethefirst.studio",
            counts=[ProfileCount("following", 89),
                    ProfileCount("followers", 1200000),
                    ProfileCount("likes", 4500000)],
            buttons=["Follow", "Message"],
            tabs=["Videos", "Liked"],
            items=[
                ProfileItem("tk1", "video", views=45200,
                            duration="0:24",
                            caption="the render",
                            pinned=True),
                ProfileItem("tk2", "video", views=38100,
                            duration="0:31",
                            caption="studio tour",
                            pinned=True),
                ProfileItem("tk3", "video", views=12900,
                            duration="0:18",
                            caption="backdrops"),
                ProfileItem("tk4", "video", views=8400,
                            duration="0:27",
                            caption="guest list"),
                ProfileItem("tk5", "video", views=5200,
                            duration="0:15",
                            caption="drinks"),
                ProfileItem("tk6", "video", views=3100,
                            duration="0:22",
                            caption="doors"),
            ],
            chrome=chrome)
    if key == "youtube-channel":
        return ProfileScreen(
            platform="youtube",
            identity=SocialProfile(
                "gs", "GoldenSwing Signals",
                handle="@GoldenSwingSignals", verified=True),
            dark=dark,
            bio="XAUUSD reads, setups and the journal — "
                 "demo only, no real-money routing.",
            counts=[ProfileCount("subscribers", 1200000),
                    ProfileCount("videos", 342)],
            buttons=["Subscribe", "Join"],
            tabs=["Videos", "Shorts", "Live", "Playlists"],
            items=[
                ProfileItem("yt1", "video",
                            title="Gold above the zone — "
                                  "the morning read",
                            views=124000, duration="12:04"),
                ProfileItem("yt2", "video",
                            title="Why the setup is the setup",
                            views=45200, duration="8:31"),
                ProfileItem("yt3", "video",
                            title="Journal: the week in R",
                            views=18900, duration="15:47"),
                ProfileItem("ys1", "short",
                            title="60 seconds on friction",
                            views=98000),
                ProfileItem("ys2", "short",
                            title="The magnet room",
                            views=41000),
            ],
            chrome=chrome)
    return ProfileScreen(
        platform="facebook", identity=ident, dark=dark,
        bio="Studio · Lagos",
        counts=[ProfileCount("friends", 1240),
                ProfileCount("posts", 86)],
        buttons=["Friends", "Message"],
        tabs=["Posts", "About", "Photos", "Friends"],
        items=[
            ProfileItem(
                "fb1", "post",
                title="Studio day! Doors open at 10 — "
                      "bring your ideas."),
            ProfileItem(
                "fb2", "post",
                title="The flyer goes out by noon, "
                      "no extensions."),
        ],
        chrome=chrome)


def demo_story_screen(name: str, style: str = "android",
                      *, dark: bool = False):
    """The Instagram Story demos (census §5.4): a photo
    story with tray stickers at their positions, and a
    text slide."""
    key = str(name or "").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G")
    owner = _demo_social_author()
    if key == "instagram-story-text":
        return StoryScreen(
            owner=owner, kind="text", dark=dark,
            text="Doors open at 10 — bring your ideas.",
            bg="#7656C9", segments=2, index=0, played=0.3,
            when="2h", chrome=chrome)
    return StoryScreen(
        owner=owner, kind="photo", dark=dark,
        segments=3, index=1, played=0.4, when="41m",
        stickers=[
            StorySticker("mention", "adaeze.creates",
                         x=0.5, y=0.32),
            StorySticker("poll", "Studio Saturday?",
                         x=0.5, y=0.58,
                         options=["Yes", "No"]),
            StorySticker("music", "Midnight City",
                         x=0.5, y=0.82),
        ],
        chrome=chrome)


def demo_comment_screen(style: str = "android",
                        *, dark: bool = False):
    """The TikTok comment sheet demo (census §5.5): a
    creator comment with its replies and a Liked-by-
    Creator mark, plus a comment with more replies than
    the sheet shows."""
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G")
    creator = _demo_social_author()
    replies = [SocialComment(
        "r" + str(i),
        author=SocialProfile("u" + str(i), "Viewer " + str(i)),
        text="This is the one.", time="2h", likes=12 + i)
        for i in range(1, 13)]
    return CommentScreen(
        total=128, dark=dark,
        comments=[
            SocialComment(
                "c1", author=creator,
                text="Full breakdown is on the channel — "
                     "link in bio.",
                time="3h", likes=452, creator=True,
                liked_by_creator=False,
                replies=[
                    SocialComment(
                        "c1r1",
                        author=SocialProfile(
                            "bola", "Bola"),
                        text="Watching it now 🔥", time="2h",
                        likes=89),
                ]),
            SocialComment(
                "c2",
                author=SocialProfile("chiamaka", "Chiamaka"),
                text="The render finally finished 😭",
                time="2h", likes=318, liked_by_creator=True,
                replies=replies),
            SocialComment(
                "c3",
                author=SocialProfile("mum", "Mum"),
                text="So proud of you.", time="1h", likes=96),
        ],
        chrome=chrome)


def demo_video_screen(name: str, style: str = "android",
                      *, dark: bool = False):
    """The YouTube video page demos (census §5.9): the
    watch page with up-next, the comments layout with
    its top comment, and the description panel."""
    key = str(name or "").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G")
    channel = SocialProfile(
        "gs", "GoldenSwing Signals",
        handle="@GoldenSwingSignals", verified=True)
    description = (
        "The morning read on gold: structure first, "
        "then the trigger. The zone held, the setup "
        "printed, and the journal carries the rest. "
        "Demo only — no real-money routing.")
    if key == "youtube-video-comments":
        return VideoScreen(
            title="Gold above the zone — the morning read",
            views=1240000, likes=45000, date="12 Nov 2026",
            hashtags=["#xauusd", "#gold"],
            description=description, channel=channel,
            subscribers=1200000, layout="comments",
            top_comment=SocialComment(
                "vc1",
                author=SocialProfile("bola", "Bola"),
                text="The zone held exactly as drawn.",
                time="2h", likes=1204),
            dark=dark, chrome=chrome)
    if key == "youtube-video-description":
        return VideoScreen(
            title="Gold above the zone — the morning read",
            views=1240000, likes=45000, date="12 Nov 2026",
            hashtags=["#xauusd", "#gold"],
            description=description, channel=channel,
            subscribers=1200000, layout="description",
            dark=dark, chrome=chrome)
    return VideoScreen(
        title="Gold above the zone — the morning read",
        views=1240000, likes=45000, date="12 Nov 2026",
        hashtags=["#xauusd", "#gold"],
        description=description, channel=channel,
        subscribers=1200000, layout="watch",
        up_next=[
            UpNextItem("un1", title="Why the setup is "
                                    "the setup",
                       channel="GoldenSwing Signals",
                       duration="8:31", views=45200),
            UpNextItem("un2", title="Journal: the week "
                                    "in R",
                       channel="GoldenSwing Signals",
                       duration="15:47", views=18900),
            UpNextItem("un3", title="Friction: the gate "
                                    "that saves accounts",
                       channel="GoldenSwing Signals",
                       duration="6:12", views=9800),
        ],
        dark=dark, chrome=chrome)


def _demo_people() -> list:
    return [
        Participant("me", "You"),
        Participant("adaeze", "Adaeze Nwosu"),
        Participant("bola", "Bola"),
        Participant("chiamaka", "Chiamaka"),
        Participant("mum", "Mum"),
    ]


def demo_list_screen(name: str, style: str = "android",
                     *, dark: bool = False):
    """One built-in demo per list family (census §4)."""
    key = str(name or "").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G",
                 unread=3 if style == "iphone" else 0)
    people = _demo_people()
    if key == "whatsapp-calls":
        return ListScreen(
            platform="whatsapp", flavor="calls", dark=dark,
            title="Calls", chrome=chrome, participants=people,
            rows=[
                ListRow("c1", "Adaeze Nwosu",
                        participant_id="adaeze",
                        preview_kind="call_in_voice",
                        time="Today, 9:41 AM"),
                ListRow("c2", "Bola", participant_id="bola",
                        preview_kind="call_out_video",
                        time="Today, 8:15 AM"),
                ListRow("c3", "Mum", participant_id="mum",
                        preview_kind="call_missed_voice",
                        preview="(2)", time="Yesterday, 9:02 PM"),
                ListRow("c4", "Chiamaka",
                        participant_id="chiamaka",
                        preview_kind="call_missed_video",
                        time="Monday"),
            ])
    if key == "whatsapp-status-list":
        return ListScreen(
            platform="whatsapp", flavor="status", dark=dark,
            title="Updates", chrome=chrome,
            participants=people,
            rows=[
                ListRow("s0", "My Status",
                        participant_id="me",
                        preview="Tap to add status update",
                        time="", row_type="person"),
                ListRow("s1", "Adaeze Nwosu",
                        participant_id="adaeze", ring="unseen",
                        segments=3, section="Recent updates",
                        time="Today, 9:12 AM"),
                ListRow("s2", "Bola", participant_id="bola",
                        ring="unseen", segments=2,
                        section="Recent updates",
                        time="Today, 8:47 AM"),
                ListRow("s3", "Chiamaka",
                        participant_id="chiamaka", ring="seen",
                        section="Viewed updates",
                        time="Today, 7:58 AM"),
                ListRow("s4", "Mum", participant_id="mum",
                        ring="seen", section="Muted updates",
                        muted=True, time="Yesterday, 9:31 PM"),
                ListRow("s5", "GoldenSwing Signals",
                        preview="1,240 followers",
                        section="Channels", row_type="channel",
                        time="Today, 9:00 AM"),
            ])
    if key == "instagram-list":
        return ListScreen(
            platform="instagram", dark=dark,
            title="peacethefirst", chrome=chrome,
            participants=people,
            stories=[
                StoryItem("adaeze", note="studio day 🎬"),
                StoryItem("bola", note="new drops friday"),
                StoryItem("chiamaka"),
            ],
            rows=[
                ListRow("i1", "Adaeze Nwosu",
                        participant_id="adaeze",
                        preview="Okay this is the one. Post it.",
                        time="2m", unread=1, ring="unseen"),
                ListRow("i2", "Bola", participant_id="bola",
                        preview_kind="typing", time="now"),
                ListRow("i3", "Chiamaka",
                        participant_id="chiamaka",
                        preview="Sent", status="sent",
                        time="1h"),
                ListRow("i4", "Mum", participant_id="mum",
                        preview="Seen", status="seen",
                        time="3h"),
                ListRow("i5", "Studio Crew",
                        preview_kind="story_reaction",
                        preview="Bola reacted to your story",
                        time="5h", row_type="group"),
                ListRow("i6", "Adaeze Nwosu",
                        participant_id="adaeze",
                        preview_kind="call_missed_video",
                        time="1d"),
            ])
    if key == "telegram-list":
        return ListScreen(
            platform="telegram", dark=dark, title="Chats",
            chrome=chrome, participants=people,
            tabs=[FolderTab("All Chats", 4), FolderTab("Work"),
                  FolderTab("Channels", 2)],
            stories=[StoryItem("adaeze", ring="unseen"),
                     StoryItem("bola", ring="seen")]
            if style == "iphone" else [],
            archived=2,
            rows=[
                ListRow("t1", "Saved Messages",
                        row_type="saved",
                        preview="the packing list draft",
                        time="9:40 AM"),
                ListRow("t2", "GoldenSwing Signals",
                        row_type="channel", verified=True,
                        preview="XAUUSD holding above the zone",
                        time="9:12 AM", unread=5, mention=True),
                ListRow("t3", "Adaeze Nwosu",
                        participant_id="adaeze", online=True,
                        preview="the renders finished",
                        time="8:58 AM", ticks="read"),
                ListRow("t4", "Studio Crew", row_type="group",
                        sender="Bola",
                        preview="bringing the backdrops",
                        time="8:31 AM", unread=2, pinned=True),
                ListRow("t5", "Bola", participant_id="bola",
                        preview="invoice for the backdrop",
                        time="Yesterday", draft=True),
                ListRow("t6", "Secret Chat",
                        row_type="secret",
                        preview_kind="typing", time="Tuesday"),
                ListRow("t7", "Devon Bot", row_type="bot",
                        preview="your morning briefing is ready",
                        time="Monday", muted=True),
            ])
    if key == "snapchat-list":
        return ListScreen(
            platform="snapchat", dark=dark, title="Chats",
            chrome=chrome, participants=people,
            rows=[
                ListRow("n1", "Adaeze",
                        participant_id="adaeze",
                        preview="morning!! did you see the "
                                "snaps from last night",
                        time="2m"),
                ListRow("n2", "Bola", participant_id="bola",
                        preview="the setup so far",
                        time="1h"),
                ListRow("n3", "Chiamaka",
                        participant_id="chiamaka",
                        preview="omw with the drinks",
                        time="3h"),
                ListRow("n4", "Mum", participant_id="mum",
                        preview="call me when you land",
                        time="Yesterday"),
            ])
    return ListScreen(
        platform="whatsapp", dark=dark, title="WhatsApp",
        chrome=chrome, participants=people, nav_badge=1,
        rows=[
            ListRow("w1", "Adaeze Nwosu",
                    participant_id="adaeze", ring="unseen",
                    preview="the flyer is ready 🔥",
                    time="9:41 AM", unread=2),
            ListRow("w2", "Studio Crew", row_type="group",
                    preview_kind="photo", pinned=True,
                    preview="Bola: the setup so far",
                    time="9:15 AM"),
            ListRow("w3", "Bola", participant_id="bola",
                    preview="on my way", time="9:02 AM",
                    ticks="read"),
            ListRow("w4", "Mum", participant_id="mum",
                    preview="half of the list, still typing",
                    time="8:47 AM", draft=True),
            ListRow("w5", "Chiamaka",
                    participant_id="chiamaka",
                    preview_kind="voice", time="8:30 AM",
                    unread=1),
            ListRow("w6", "GoldenSwing Signals",
                    row_type="channel", muted=True,
                    preview_kind="call_missed_voice",
                    time="Yesterday"),
            ListRow("w7", "Bola", participant_id="bola",
                    preview_kind="typing", time="Yesterday"),
        ])


def demo_status_screen(name: str, style: str = "android",
                       *, dark: bool = False):
    """The WhatsApp Status demos (census §4.2): a photo
    status (the no-media case wears the placeholder scene,
    recorded as such) and a text status on a stocked
    colour."""
    key = str(name or "").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G")
    when = "9:12" if style == "iphone" else "21 minutes ago"
    owner = Participant("adaeze", "Adaeze Nwosu")
    if key == "whatsapp-status-text":
        return StatusScreen(
            owner=owner, kind="text", dark=dark,
            text="Studio day. Doors open at 10 — "
                 "bring your ideas.",
            bg="#7656C9", text_size=30, when=when,
            segments=2, index=0, played=0.3,
            reactions=["❤️", "🔥", "😮"], chrome=chrome)
    return StatusScreen(
        owner=owner, kind="photo", dark=dark,
        caption="the backdrop from last time",
        when=when, label="Close friends", segments=4,
        index=1, played=0.45, reshare=style == "iphone",
        reply_style="glass" if style == "iphone" else "",
        reactions=["❤️", "🔥", "😮"], liked=True,
        chrome=chrome)


def demo_status_views(style: str = "android"):
    """The Status Views demo (census §4.1)."""
    return StatusViewsScreen(
        owner=Participant("me", "You"),
        when="Today, 9:12 AM",
        participants=_demo_people(),
        viewers=[
            StatusViewer("Adaeze Nwosu", "adaeze",
                         "Today, 9:31 AM"),
            StatusViewer("Bola", "bola", "Today, 9:40 AM"),
            StatusViewer("Chiamaka", "chiamaka",
                         "Today, 10:02 AM"),
        ],
        chrome=DeviceChrome(style=style, clock="9:41", battery=84,
                   network="4G"))


def demo_call_screen(name: str, style: str = "android",
                     *, dark: bool = False):
    """One demo per call family (census §6.1 + §6.2)."""
    key = str(name or "").strip().lower()
    chrome = DeviceChrome(style=style, clock="9:41", battery=84,
                 network="4G")
    adaeze = Participant("adaeze", "Adaeze Nwosu")
    if key == "call-incoming":
        return CallScreen(
            platform="generic", dark=dark,
            caller=Participant("caller", "",
                               color="#E53935"),
            number="+234 800 111 2222", state="incoming",
            sim="SIM 1" if style == "iphone" else "",
            chrome=chrome)
    if key == "whatsapp-call":
        return CallScreen(
            platform="whatsapp", caller=adaeze,
            number="+234 800 000 0000", state="ongoing",
            status="04:32", video=True, chrome=chrome,
            dark=dark)
    if key == "facebook-call":
        return CallScreen(
            platform="facebook", caller=adaeze,
            state="ongoing", status="12:08", chrome=chrome,
            dark=dark)
    if key == "instagram-call":
        return CallScreen(
            platform="instagram",
            caller=Participant("adaeze", "adaeze.creates"),
            state="incoming", video=True, chrome=chrome,
            dark=dark)
    if key == "hangouts-call":
        return CallScreen(
            platform="hangouts", caller=adaeze,
            state="ongoing", status="01:47", chrome=chrome,
            dark=dark)
    return CallScreen(
        platform="generic", caller=adaeze,
        number="+234 800 000 0000", state="ongoing",
        status="02:15", chrome=chrome, dark=dark)


def demo_call_history(style: str = "iphone"):
    """The iPhone Call History demo (census §6.3): every
    entry type and avatar mode has a row."""
    return CallHistoryScreen(
        participants=_demo_people(),
        entries=[
            CallHistoryEntry(
                "h1", "Adaeze Nwosu", "adaeze", type="missed",
                avatar_mode="picture", label="mobile",
                count=2, when="9:41 AM"),
            CallHistoryEntry(
                "h2", "+234 800 111 2222", type="outgoing",
                avatar_mode="default", label="FaceTime Audio",
                when="Yesterday"),
            CallHistoryEntry(
                "h3", "Bola", "bola", type="incoming",
                avatar_mode="letter", sim="SIM 2",
                label="home", when="Tuesday"),
            CallHistoryEntry(
                "h4", "Mum", "mum", type="incoming",
                avatar_mode="picture", label="mobile",
                when="12/05/2026"),
            CallHistoryEntry(
                "h5", "Chiamaka", "chiamaka", type="missed",
                avatar_mode="letter", label="work",
                when="Monday"),
        ],
        voicemail_count=2,
        chrome=DeviceChrome(style="iphone", clock="9:41", battery=84,
                   network="4G"))


def _clock(minutes: int) -> str:
    total = 9 * 60 + 41 + minutes
    hh, mm = divmod(total, 60)
    suffix = "AM" if hh < 12 else "PM"
    return str(hh if hh <= 12 else hh - 12) + ":" + format(mm, "02d") \
        + " " + suffix


def parse_mock_tail(tail: str) -> tuple:
    """Parse the compact command form into a Screen.
    Raises MockScreensError with the usage line on a shape
    the form cannot carry."""
    text = str(tail or "").strip()
    if not text:
        raise MockScreensError(USAGE)
    words = text.split(None, 1)
    head = words[0].lower()
    if head == "data":
        return _parse_data_tail(
            words[1] if len(words) > 1 else ""), "data"
    if head == "record":
        # /mock record <demo … | built form …> — the
        # remainder parses exactly as a still would, and
        # the mode comes back with the record- prefix the
        # runtime dispatches on (Phase 4).
        rest = words[1].strip() if len(words) > 1 else ""
        if not rest or rest.split(None, 1)[0].lower() \
                == "record":
            raise MockScreensError(USAGE)
        screen, mode = parse_mock_tail(rest)
        return screen, "record-" + mode
    if head == "demo":
        rest = words[1].split() if len(words) > 1 else []
        name = rest[0] if rest else "whatsapp"
        style = "iphone" if "iphone" in rest else "android"
        dark = "dark" in rest
        if name in _BULK_DEMOS:
            return _parse_bulk_demo(name, rest[1:]), \
                _BULK_MODES[name]
        if name in _LIST_DEMOS:
            return demo_list_screen(name, style,
                                    dark=dark), "demo-list"
        if name in _STATUS_DEMOS:
            return demo_status_screen(name, style,
                                      dark=dark), "demo-status"
        if name in _VIEWS_DEMOS:
            return demo_status_views(style), "demo-views"
        if name in _CALL_DEMOS:
            return demo_call_screen(name, style,
                                    dark=dark), "demo-call"
        if name in _HISTORY_DEMOS:
            return demo_call_history(style), "demo-history"
        if name in _POST_DEMOS:
            return demo_post_screen(name, style,
                                    dark=dark), "demo-post"
        if name in _PROFILE_DEMOS:
            return demo_profile_screen(name, style,
                                       dark=dark), "demo-profile"
        if name in _STORY_DEMOS:
            return demo_story_screen(name, style,
                                     dark=dark), "demo-story"
        if name in _COMMENT_DEMOS:
            return demo_comment_screen(style,
                                       dark=dark), "demo-comment"
        if name in _VIDEO_DEMOS:
            return demo_video_screen(name, style,
                                     dark=dark), "demo-video"
        if name not in _PLATFORMS and name not in _DEMO_VARIANTS:
            raise MockScreensError(
                "demo families: whatsapp · group · instagram "
                "· telegram · telegram-group · messenger · "
                "messenger-group · snapchat · snapchat-group "
                "· imessage · imessage-sms · sms — screen "
                "demos: whatsapp-list · whatsapp-calls · "
                "whatsapp-status-list · instagram-list · "
                "telegram-list · snapchat-list · "
                "whatsapp-status · whatsapp-status-text · "
                "whatsapp-status-views · call · "
                "call-incoming · whatsapp-call · "
                "facebook-call · instagram-call · "
                "hangouts-call · iphone-call-history · "
                "facebook-post · instagram-post · "
                "threads-post · x-tweet · tiktok-post · "
                "facebook-profile · instagram-profile · "
                "x-profile · tiktok-profile · "
                "youtube-channel · instagram-story · "
                "instagram-story-text · tiktok-comment · "
                "youtube-video · youtube-video-comments · "
                "youtube-video-description · bulk-list · "
                "bulk-calls · bulk-status-list · bulk-history "
                "· bulk-chat · bulk-views")
        return demo_screen(name, style, dark=dark), "demo"
    if head not in _PLATFORMS:
        raise MockScreensError(USAGE)
    platform, group = _PLATFORMS[head]
    rest = words[1] if len(words) > 1 else ""
    style = "android"
    dark = False
    title = ""
    segments_text = rest
    # leading option words, consumed until the first ":"
    while True:
        stripped = segments_text.lstrip()
        low = stripped.lower()
        if low.startswith("iphone"):
            style = "iphone"
            segments_text = stripped[len("iphone"):]
        elif low.startswith("android"):
            style = "android"
            segments_text = stripped[len("android"):]
        elif low.startswith("dark"):
            dark = True
            segments_text = stripped[len("dark"):]
        elif low.startswith("with "):
            m = re.match(
                r"with\s+(.+?)(?=\s*;;|\s+\S+:\s|\s*$)",
                stripped, re.IGNORECASE)
            if m is None or not m.group(1).strip():
                break
            title = m.group(1).strip()
            segments_text = stripped[m.end():]
        else:
            break
    segments = [s.strip() for s in segments_text.split(";;")]
    segments = [s for s in segments if s]
    if not segments:
        raise MockScreensError(USAGE)

    participants: dict = {"me": Participant("me", "You")}
    messages: list = []
    clock_min = 0

    def person(name: str) -> str:
        pid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") \
            or "them"
        if pid not in participants:
            participants[pid] = Participant(pid, name)
        return pid

    if group and not title:
        title = "Group Chat"
    for idx, seg in enumerate(segments):
        seg = seg.strip()
        if seg.startswith("!"):
            messages.append(Message(
                "e" + str(idx), "system", kind="system",
                text=seg[1:].strip()))
            continue
        if seg.startswith("#"):
            messages.append(Message(
                "e" + str(idx), "system", kind="divider",
                text=seg[1:].strip() or "Today"))
            continue
        reply = seg.startswith("^")
        if reply:
            seg = seg[1:].strip()
        ticks = ""
        m = _TICK_SUFFIX.search(seg)
        if m is not None:
            ticks = m.group(1)
            seg = seg[:m.start()].strip()
        reactions: list = []
        m = _REACT_SUFFIX.search(seg)
        if m is not None:
            reactions = m.group(1).split()
            seg = seg[:m.start()].strip()
        when = ""
        m = _TIME_PREFIX.match(seg)
        if m is not None and m.group(1).strip()[-2:].upper() \
                in ("AM", "PM"):
            when = m.group(1).strip()
            seg = m.group(2).strip()
        if ":" not in seg:
            raise MockScreensError(
                "each message segment is 'Name: words' — " + USAGE)
        name, words_text = seg.split(":", 1)
        name = name.strip()
        words_text = words_text.strip()
        if words_text.startswith("^"):
            reply = True
            words_text = words_text[1:].strip()
        if not when:
            clock_min += 2
            when = _clock(clock_min)
        if name.lower() in ("me", "you"):
            pid = "me"
            if not ticks:
                ticks = "read"
        else:
            pid = person(name)
            if not title and not group:
                title = name
        shared = {
            "time": when,
            "ticks": ticks if pid == "me" else "",
            "reactions": reactions,
            "reply_to": (messages[-1].id
                         if reply and messages else ""),
        }
        marker = _MEDIA_MARKER.match(words_text)
        if marker is not None:
            word = marker.group(1).lower()
            arg = (marker.group(2) or "").strip()
            rest = words_text[marker.end():].strip()
            if word in ("photo", "image"):
                ref = "attach:image" if not arg else (
                    "last:image" if arg.lower() == "last"
                    else arg)
                messages.append(Message(
                    "e" + str(idx), pid, kind="image",
                    caption=rest, media=ref, **shared))
            else:
                if rest:
                    raise MockScreensError(
                        "a voice segment carries no words — "
                        "put them in their own segment: "
                        + USAGE)
                ref = arg if arg else "attach:voice"
                messages.append(Message(
                    "e" + str(idx), pid, kind="voice",
                    media=ref, **shared))
            continue
        messages.append(Message(
            "e" + str(idx), pid, text=words_text, **shared))
    if not messages:
        raise MockScreensError(USAGE)
    if not any(m.kind == "divider" for m in messages):
        messages.insert(0, Message(
            "e-day", "system", kind="divider", text="Today"))
    others = [p for p in participants.values() if p.id != "me"]
    if not title:
        title = others[0].name if others else "Chat"
    subtitle = ""
    if group:
        subtitle = ", ".join(
            [p.name for p in participants.values()
             if p.id != "me"] + ["you"])
    elif platform in ("whatsapp", "telegram"):
        subtitle = "online"
    elif platform == "snapchat":
        subtitle = "🔥 21"
    elif platform in ("imessage", "sms"):
        subtitle = ""
    else:
        subtitle = "Active now"
    screen = Screen(
        platform=platform, group=group, dark=dark, title=title,
        subtitle=subtitle,
        participants=list(participants.values()),
        messages=messages,
        chrome=DeviceChrome(
            style=style, clock="9:41", battery=84, network="4G",
            unread=3 if style == "iphone" else 0))
    return screen, "built"
