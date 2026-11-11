"""Authoring for mock screens: built-in demos + the
compact structured command form.

NL authoring is a later phase; today a screen is authored
two structured ways:

- **Demos** — :func:`demo_screen` builds a small real-shape
  script per flagship family (the same scripts the visual
  proofs render).
- **The compact form** the ``/mock`` command parses
  (:func:`parse_mock_tail`)::

      /mock whatsapp [iphone|android] [dark] [with <Name>]
            <segment> ;; <segment> ;; …

  Segments: ``me: words`` is outgoing; ``<Name>: words``
  is incoming from that participant; ``! words`` is a
  system row; ``# Today`` is a day divider.  Suffixes on a
  segment: ``[read]`` / ``[sent]`` / ``[delivered]`` set an
  outgoing message's ticks, ``[react ❤️ 🔥]`` hangs
  reactions on it, and a leading ``^`` makes the message a
  reply to the previous one.  Times are assigned on a
  steady clock from 9:41 AM unless a segment opens with a
  time (``9:58 Adaeze: words``).
"""

from __future__ import annotations

import re

from .render import MockScreensError
from .schema import DeviceChrome, Message, Participant, Screen

USAGE = (
    "usage: /mock demo [whatsapp|group|instagram] "
    "[iphone|android] — or build one: /mock whatsapp "
    "[iphone|android] [with <Name>] me: hi ;; Adaeze: hello "
    ";; me: on my way [read]")

_PLATFORMS = {
    "whatsapp": ("whatsapp", False),
    "wa": ("whatsapp", False),
    "group": ("whatsapp", True),
    "whatsapp-group": ("whatsapp", True),
    "wag": ("whatsapp", True),
    "instagram": ("instagram", False),
    "ig": ("instagram", False),
}

_TICK_SUFFIX = re.compile(r"\[(read|sent|delivered|sending)\]\s*$")
_REACT_SUFFIX = re.compile(r"\[react\s+([^\]]*)\]\s*$")
_TIME_PREFIX = re.compile(
    r"^(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\s+(.*)$", re.DOTALL)


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
        if name not in _PLATFORMS and name != "whatsapp":
            raise MockScreensError(
                "demo families: whatsapp · group · instagram")
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
        messages.append(Message(
            "e" + str(idx), pid, text=words_text, time=when,
            ticks=ticks if pid == "me" else "",
            reactions=reactions,
            reply_to=(messages[-1].id
                      if reply and messages else "")))
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
    elif platform == "whatsapp":
        subtitle = "online"
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
