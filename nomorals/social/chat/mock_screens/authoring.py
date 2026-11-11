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

from .render import MockScreensError
from .schema import DeviceChrome, Message, Participant, Screen

USAGE = (
    "usage: /mock demo [whatsapp|group|instagram|telegram|"
    "telegram-group|messenger|messenger-group|snapchat|"
    "snapchat-group|imessage|imessage-sms|sms] "
    "[iphone|android] — or build one: /mock whatsapp "
    "[iphone|android] [with <Name>] me: hi ;; Adaeze: hello "
    ";; me: on my way [read] — families: whatsapp · "
    "telegram · messenger · snapchat · imessage · sms · "
    "instagram (add -group for the group form) — media: "
    "me: [photo] uses an image attached to your message, "
    "[photo <path>] a file on disk, [photo last] your "
    "newest generated image; [voice] / [voice <path>] a "
    "voice note with its real waveform and duration; "
    "words after [photo] are its caption")

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
    if head == "demo":
        rest = words[1].split() if len(words) > 1 else []
        name = rest[0] if rest else "whatsapp"
        if name not in _PLATFORMS and name not in _DEMO_VARIANTS:
            raise MockScreensError(
                "demo families: whatsapp · group · instagram "
                "· telegram · telegram-group · messenger · "
                "messenger-group · snapchat · snapchat-group "
                "· imessage · imessage-sms · sms")
        style = "iphone" if "iphone" in rest else "android"
        return demo_screen(name, style,
                           dark="dark" in rest), "demo"
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
