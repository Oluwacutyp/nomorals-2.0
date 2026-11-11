"""Natural-language authoring for mock screens (FakeDetail
Phase 6, census §30).

A plain-language description becomes a visible structured
:model:`Screen` before anything renders — the census's
exit proof.  The parser is a DETERMINISTIC code rung over
a named, bounded grammar, not a free-form model call: a
model rung may rewrite free text INTO this grammar first,
but the grammar is the only thing that builds a screen.

The grammar (NL_GRAMMAR):

    make a <platform> chat [group] with <Name> [and <Name>]
        [iphone|android] [dark|light]
        <Speaker>: "words" [read|sent|delivered] [react X]
            [photo|voice] ;; <Speaker>: "words" ...

- platform: whatsapp · instagram · telegram · messenger ·
  snapchat · imessage · sms (aliases wa / ig / tg / msgr /
  snap / imsg, as in the compact form).  Two platforms
  named = ambiguity, reported as candidates, never
  guessed between.
- "group" turns on the group choreography; without it the
  chat is 1:1 and the title is the other participant.
- a message with no quoted words is never invented: a
  description that says "three messages" but gives none
  (or fewer than it claims) comes back as QUESTIONS, and
  the same for a chat with no one named.  Missing facts
  are asked for; only visible defaults are applied, and
  the model summary NAMES them (times from 9:41 two
  minutes apart, your messages ticked read, the device
  chrome).
- media rides the compact form's tokens: ``[photo]`` /
  ``[photo last]`` / ``[voice]`` inside a segment, or a
  whole segment that IS "a photo" / "a voice note"; a
  trailing "last one is a voice note" / "last message is
  a photo" re-kinds the last message.  The runtime
  resolves the tokens exactly as it does for the compact
  form.
- follow-up revision (``revise_screen``) addresses
  elements by the number the model summary shows them
  with — "change message 2 to ...", "add Ada: ...",
  "rename Ada to Adaeze", "delete message 1", "make it
  dark" — and touches ONLY the addressed element: every
  other message's dict is byte-identical after a revision
  (pinned by the Phase 6 suite).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import (
    DeviceChrome,
    Message,
    Participant,
    Screen,
)

__all__ = [
    "MAKE_USAGE",
    "NL_GRAMMAR",
    "NLAuthoringError",
    "NLParseResult",
    "PROJECT_USAGE",
    "model_summary",
    "parse_nl_description",
    "revise_screen",
]

#: The platform words the grammar recognises (aliases as
#: in the compact form's _PLATFORMS).
_PLATFORMS = {
    "whatsapp": "whatsapp", "wa": "whatsapp",
    "instagram": "instagram", "ig": "instagram",
    "telegram": "telegram", "tg": "telegram",
    "messenger": "messenger", "msgr": "messenger",
    "facebook": "messenger",
    "snapchat": "snapchat", "snap": "snapchat",
    "imessage": "imessage", "imsg": "imessage",
    "sms": "sms",
}
_PLATFORM_WORDS = tuple(sorted(_PLATFORMS, key=len, reverse=True))

NL_GRAMMAR = (
    "make a <platform> chat [group] with <Name> [and <Name>] "
    "[iphone|android] [dark|light] — then each message as "
    "Name: \"words\" separated by ;; — e.g. /mock make a "
    "WhatsApp chat with Ada, me: \"hey\" ;; Ada: \"hi\" [read] "
    "— media: a message that IS \"a photo\" or \"a voice "
    "note\", or [photo] / [photo last] / [voice] in it; "
    "\"last one is a voice note\" re-kinds the last message.")

MAKE_USAGE = (
    "usage: /mock make <description> [save <name>] — "
    + NL_GRAMMAR)

PROJECT_USAGE = (
    "usage: /mock project list — /mock project show <name> "
    "— /mock project edit <name> <instruction> — /mock "
    "project delete <name> — a project is a saved screen "
    "spec (never baked pixels): reopen it, revise one "
    "message, re-render it.")


class NLAuthoringError(Exception):
    """The description (or revision) cannot become a
    screen as written — the message names what is missing
    or what the grammar cannot carry."""


@dataclass
class NLParseResult:
    """What the NL front end understood.  Exactly one of
    ``screen`` / ``questions`` / ``candidates`` carries
    the outcome: a screen to show and render, facts to
    ask for (never invented), or ambiguity to choose
    between (never guessed).  ``defaults`` names every
    default the screen carries, so the shown model is
    honest about what the description did not say."""

    screen: Screen | None = None
    questions: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    defaults: list = field(default_factory=list)
    summary: str = ""


# ── small helpers ────────────────────────────────────────

_CLOCK_BASE = 9 * 60 + 41


def _clock(minutes: int) -> str:
    total = _CLOCK_BASE + minutes
    hh, mm = divmod(total, 60)
    suffix = "AM" if hh < 12 else "PM"
    return (str(hh if hh <= 12 else hh - 12) + ":"
            + format(mm, "02d") + " " + suffix)


def _clock_to_minutes(when: str) -> int:
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)",
                 str(when or "").strip(), re.IGNORECASE)
    if m is None:
        return _CLOCK_BASE
    hh, mm = int(m.group(1)), int(m.group(2))
    if m.group(3).upper() == "PM" and hh != 12:
        hh += 12
    if m.group(3).upper() == "AM" and hh == 12:
        hh = 0
    return hh * 60 + mm


def _subtitle(platform: str, group: bool, others: list) -> str:
    if group:
        return ", ".join([p.name for p in others] + ["you"])
    if platform in ("whatsapp", "telegram"):
        return "online"
    if platform == "snapchat":
        return "🔥 21"
    if platform in ("imessage", "sms"):
        return ""
    return "Active now"


def _person_id(name: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", name.lower())
            .strip("-") or "them")


_QUOTED = re.compile(
    r"([A-Za-z][A-Za-z0-9 _.'-]*?)\s*(?::|says)\s*\"([^\"]*)\"")
_TICK_SUFFIX = re.compile(r"\[(read|sent|delivered|sending)\]\s*$")
_REACT_SUFFIX = re.compile(r"\[react\s+([^\]]*)\]\s*$")
_MEDIA_MARKER = re.compile(
    r"\[(photo|image|voice|audio)(?:\s+([^\]]+))?\]\s*",
    re.IGNORECASE)
_LAST_KIND = re.compile(
    r"\blast\s+(?:message|one)\s+(?:is\s+)?a\s+"
    r"(photo|picture|voice note|voice)\b", re.IGNORECASE)
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+messages?\b", re.IGNORECASE)


def _count_claimed(text: str) -> int | None:
    m = _COUNT_RE.search(text)
    if m is None:
        return None
    raw = m.group(1).lower()
    return int(raw) if raw.isdigit() else _COUNT_WORDS[raw]


# ── the parser ───────────────────────────────────────────

def parse_nl_description(text: str) -> NLParseResult:
    """Parse a plain-language description into a screen,
    or into questions / candidates.  Never invents a
    participant, a message, or a time."""
    raw = str(text or "").strip()
    if not raw:
        return NLParseResult(
            questions=["describe the screen — " + NL_GRAMMAR])
    low = raw.lower()

    # platform: exactly one, or ambiguity / a question.
    found: list[str] = []
    for word in _PLATFORM_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", low):
            canon = _PLATFORMS[word]
            if canon not in found:
                found.append(canon)
    if len(found) > 1:
        return NLParseResult(
            candidates=found,
            summary=("which platform — " + " or ".join(found)
                     + "? say one of them and I will build it."))
    if not found:
        return NLParseResult(
            questions=["which platform — whatsapp, instagram, "
                       "telegram, messenger, snapchat, imessage "
                       "or sms?"],
            candidates=["whatsapp", "instagram", "telegram",
                        "messenger", "snapchat", "imessage",
                        "sms"],
            summary="which platform?")

    platform = found[0]
    group = "group" in low
    style = "iphone" if re.search(r"\biphone\b", low) else "android"
    dark = bool(re.search(r"\bdark\b", low))
    if re.search(r"\blight mode\b", low):
        dark = False

    # participants from the "with ..." clause — the chunk
    # ends at the first message, comma, or style word, so
    # "with Ada dark mode" names Ada, not "Ada dark mode".
    names: list[str] = []
    m = re.search(r"\bwith\s+([A-Za-z][A-Za-z0-9 ,.'-]*)",
                  raw, re.IGNORECASE)
    if m is not None:
        chunk = m.group(1)
        chunk = re.split(
            r"\s*[,;]\s*"
            r"|(?=\s+(?:dark|light|iphone|android)\b)"
            r"|(?=\s+[A-Za-z][A-Za-z0-9 _.'-]*\s*:)",
            chunk)[0]
        for part in re.split(r"\s+and\s+|\s*,\s*", chunk):
            name = part.strip().strip(",")
            if name and name.lower() not in ("me", "you"):
                names.append(name)

    # messages: Speaker: "words" (or Speaker says "words").
    messages: list[Message] = []
    participants: dict = {"me": Participant("me", "You")}
    clock_min = 0

    def person(name: str) -> str:
        pid = _person_id(name)
        if pid not in participants:
            participants[pid] = Participant(pid, name.strip())
        return pid

    for match in _QUOTED.finditer(raw):
        name = match.group(1).strip()
        words_text = match.group(2).strip()
        ticks = ""
        tm = _TICK_SUFFIX.search(words_text)
        if tm is not None:
            ticks = tm.group(1)
            words_text = words_text[:tm.start()].strip()
        reactions: list = []
        rm = _REACT_SUFFIX.search(words_text)
        if rm is not None:
            reactions = rm.group(1).split()
            words_text = words_text[:rm.start()].strip()
        # markers after the closing quote, too — the
        # natural NL form is me: "hello" [sent], not a
        # marker inside the quote.
        after = raw[match.end():match.end() + 48]
        tm2 = re.match(
            r"\s*\[(read|sent|delivered|sending)\]", after)
        if tm2 is not None:
            ticks = tm2.group(1)
        rm2 = re.match(r"\s*\[react\s+([^\]]*)\]", after)
        if rm2 is not None:
            reactions = rm2.group(1).split()
        if name.lower() in ("me", "you"):
            pid = "me"
            if not ticks:
                ticks = "read"
        else:
            pid = person(name)
            if name not in names:
                names.append(name)
        clock_min += 2
        when = _clock(clock_min)
        shared = {"time": when,
                  "ticks": ticks if pid == "me" else "",
                  "reactions": reactions, "reply_to": ""}
        mid = "n" + str(len(messages) + 1)
        marker = _MEDIA_MARKER.match(words_text)
        bare = words_text.lower()
        if marker is not None:
            word = marker.group(1).lower()
            arg = (marker.group(2) or "").strip()
            rest = words_text[marker.end():].strip()
            if word in ("photo", "image"):
                ref = "attach:image" if not arg else (
                    "last:image" if arg.lower() == "last" else arg)
                messages.append(Message(
                    mid, pid, kind="image", caption=rest,
                    media=ref, **shared))
            else:
                if rest:
                    raise NLAuthoringError(
                        "a voice segment carries no words — "
                        "put them in their own message")
                ref = arg if arg else "attach:voice"
                messages.append(Message(
                    mid, pid, kind="voice", media=ref, **shared))
            continue
        if bare in ("a photo", "a picture", "a photo message"):
            messages.append(Message(
                mid, pid, kind="image", caption="",
                media="attach:image", **shared))
            continue
        if bare in ("a voice note", "a voice message"):
            messages.append(Message(
                mid, pid, kind="voice", media="attach:voice",
                **shared))
            continue
        messages.append(Message(
            mid, pid, text=words_text, **shared))

    # a trailing "last one is a voice note / a photo".
    lm = _LAST_KIND.search(raw)
    if lm is not None and messages:
        word = lm.group(1).lower()
        last = messages[-1]
        if word in ("photo", "picture"):
            last.kind = "image"
            last.media = "attach:image"
            last.caption = last.text
            last.text = ""
        else:
            if last.text:
                raise NLAuthoringError(
                    "the last message carries words, so it "
                    "cannot also be a voice note — give the "
                    "voice note its own message")
            last.kind = "voice"
            last.media = "attach:voice"

    # missing facts are asked for, never invented.
    questions: list[str] = []
    if not names and not messages:
        questions.append(
            "who is the chat with? name them — "
            "\"with Ada\" — and give each message as "
            "Name: \"words\"")
    elif not messages:
        questions.append(
            "what do they say? give each message as "
            "Name: \"words\" — I will not invent the "
            "conversation")
    claimed = _count_claimed(raw)
    if claimed is not None and messages \
            and claimed != len(messages):
        questions.append(
            "you said " + str(claimed) + " messages but I "
            "have " + str(len(messages)) + " quoted — "
            "which are the missing ones?")
    if questions:
        return NLParseResult(questions=questions)

    others = [p for p in participants.values()
              if p.id != "me"]
    if group:
        title = ", ".join(p.name for p in others) \
            if others else "Group Chat"
    else:
        title = others[0].name if others else "Chat"
    if not any(msg.kind == "divider" for msg in messages):
        messages.insert(0, Message(
            "n-day", "system", kind="divider", text="Today"))
    defaults = [
        "times from 9:41, two minutes apart",
        "your messages ticked read",
        "device " + style + " · 84% battery · 4G",
    ]
    screen = Screen(
        platform=platform, group=group, dark=dark,
        title=title, subtitle=_subtitle(platform, group, others),
        participants=list(participants.values()),
        messages=messages,
        chrome=DeviceChrome(style=style, clock="9:41",
                            battery=84, network="4G",
                            unread=3 if style == "iphone" else 0))
    return NLParseResult(
        screen=screen, defaults=defaults,
        summary=model_summary(screen, defaults))


def model_summary(screen: Screen, defaults: list | None = None
                  ) -> str:
    """The structured model, as text — what the screen IS
    before it renders: platform, participants with stable
    ids, messages numbered as the revision grammar
    addresses them, style, and the defaults applied."""
    lines = [
        screen.platform + (" group" if screen.group else "")
        + " chat — " + screen.title,
        "device: " + screen.chrome.style
        + (" · dark" if screen.dark else " · light"),
        "people: " + ", ".join(
            p.name + " (" + p.id + ")"
            for p in screen.participants),
    ]
    numbered = [m for m in screen.messages
                if m.kind not in ("divider", "system")]
    for idx, msg in enumerate(numbered, 1):
        who = "you" if msg.sender == "me" else (
            screen.participant(msg.sender).name
            if screen.participant(msg.sender) else msg.sender)
        if msg.kind == "voice":
            body = "voice note"
        elif msg.kind == "image":
            body = "photo" + (
                ": " + msg.caption if msg.caption else "")
        else:
            body = msg.text
        lines.append(str(idx) + ". [" + msg.id + "] " + who
                     + ": " + body + " (" + msg.time + ")"
                     + (" [" + msg.ticks + "]"
                        if msg.ticks else ""))
    if defaults:
        lines.append("defaults: " + "; ".join(defaults))
    return "\n".join(lines)


# ── follow-up revision ───────────────────────────────────

_CHANGE_RE = re.compile(
    r"(?:change|rewrite|edit)\s+message\s+(\S+)\s+to\s+\"([^\"]*)\"",
    re.IGNORECASE)
_MESSAGE_SET_RE = re.compile(
    r"message\s+(\S+)\s*:\s*\"([^\"]*)\"", re.IGNORECASE)
_ADD_RE = re.compile(
    r"add\s+(?:a\s+message\s+from\s+)?"
    r"([A-Za-z][A-Za-z0-9 _.'-]*?)\s*:\s*\"([^\"]*)\"",
    re.IGNORECASE)
_DELETE_RE = re.compile(
    r"(?:delete|remove)\s+message\s+(\S+)", re.IGNORECASE)
_RENAME_RE = re.compile(
    r"rename\s+([A-Za-z][A-Za-z0-9 _.'-]*?)\s+to\s+"
    r"([A-Za-z][A-Za-z0-9 _.'-]*)", re.IGNORECASE)
_KIND_RE = re.compile(
    r"(?:make\s+)?message\s+(\S+)\s+(?:is\s+)?a\s+"
    r"(photo|picture|voice note|voice)", re.IGNORECASE)
_REACT_RE = re.compile(
    r"(?:add\s+)?react(?:ion)?\s+(\S+)\s+to\s+message\s+(\S+)",
    re.IGNORECASE)
_TITLE_RE = re.compile(
    r"(?:change\s+the\s+)?title\s+(?:to\s+)?\"([^\"]*)\"",
    re.IGNORECASE)


def _content_messages(screen: Screen) -> list:
    return [m for m in screen.messages
            if m.kind not in ("divider", "system")]


def _address(screen: Screen, token: str) -> Message | None:
    token = str(token or "").strip().lower()
    for msg in _content_messages(screen):
        if msg.id.lower() == token:
            return msg
    if token.isdigit():
        numbered = _content_messages(screen)
        idx = int(token)
        if 1 <= idx <= len(numbered):
            return numbered[idx - 1]
    return None


def revise_screen(screen: Screen, instruction: str
                  ) -> tuple[Screen, str]:
    """Apply one follow-up instruction to a screen and
    return (new screen, what changed).  Only the
    addressed element changes — every other message's
    dict is identical afterwards.  Raises
    NLAuthoringError on an instruction the grammar
    cannot carry."""
    text = str(instruction or "").strip()
    if not text:
        raise NLAuthoringError(
            "say what to change — e.g. change message 2 to "
            "\"see you then\"")
    low = text.lower()
    new = Screen.from_dict(screen.to_dict())
    content = _content_messages(new)

    m = _CHANGE_RE.search(text) or _MESSAGE_SET_RE.search(text)
    if m is not None:
        target = _address(new, m.group(1))
        if target is None:
            raise NLAuthoringError(
                "no message " + m.group(1) + " — the screen "
                "has " + str(len(content)) + " messages")
        if target.kind == "voice":
            raise NLAuthoringError(
                "message " + target.id + " is a voice note — "
                "it carries no words to change")
        if target.kind == "image":
            target.caption = m.group(2)
        else:
            target.text = m.group(2)
        return new, ("message " + target.id + " now reads "
                     + repr(m.group(2)))

    m = _ADD_RE.search(text)
    if m is not None:
        name = m.group(1).strip()
        words_text = m.group(2).strip()
        pid = "me" if name.lower() in ("me", "you") \
            else _person_id(name)
        if pid != "me" and new.participant(pid) is None:
            new.participants.append(Participant(pid, name))
            if new.group:
                others = [p for p in new.participants
                          if p.id != "me"]
                new.subtitle = ", ".join(
                    [p.name for p in others] + ["you"])
        last_min = _clock_to_minutes(
            content[-1].time) if content else _CLOCK_BASE
        when = _clock(last_min - _CLOCK_BASE + 2)
        nxt = 1
        for msg in new.messages:
            if msg.id.startswith("n") \
                    and msg.id[1:].isdigit():
                nxt = max(nxt, int(msg.id[1:]) + 1)
        ticks = "read" if pid == "me" else ""
        new.messages.append(Message(
            "n" + str(nxt), pid, text=words_text, time=when,
            ticks=ticks))
        return new, ("added message n" + str(nxt) + " from "
                     + ("you" if pid == "me" else name))

    m = _DELETE_RE.search(text)
    if m is not None:
        target = _address(new, m.group(1))
        if target is None:
            raise NLAuthoringError(
                "no message " + m.group(1) + " — the screen "
                "has " + str(len(content)) + " messages")
        new.messages = [msg for msg in new.messages
                        if msg is not target]
        return new, "deleted message " + target.id

    m = _RENAME_RE.search(text)
    if m is not None:
        old, new_name = m.group(1).strip(), m.group(2).strip()
        pid = _person_id(old)
        person = new.participant(pid)
        if person is None:
            raise NLAuthoringError(
                "no one called " + old + " on this screen")
        person.name = new_name
        if new.title == old:
            new.title = new_name
        if new.group:
            others = [p for p in new.participants
                      if p.id != "me"]
            new.subtitle = ", ".join(
                [p.name for p in others] + ["you"])
        return new, "renamed " + old + " to " + new_name

    m = _KIND_RE.search(text)
    if m is not None:
        target = _address(new, m.group(1))
        if target is None:
            raise NLAuthoringError(
                "no message " + m.group(1) + " — the screen "
                "has " + str(len(content)) + " messages")
        word = m.group(2).lower()
        if word in ("photo", "picture"):
            target.kind = "image"
            target.media = "attach:image"
            target.caption = target.text
            target.text = ""
            return new, "message " + target.id + " is a photo now"
        if target.text:
            raise NLAuthoringError(
                "message " + target.id + " carries words, so "
                "it cannot also be a voice note — give the "
                "voice note its own message")
        target.kind = "voice"
        target.media = "attach:voice"
        return new, ("message " + target.id
                     + " is a voice note now")

    m = _REACT_RE.search(text)
    if m is not None:
        target = _address(new, m.group(2))
        if target is None:
            raise NLAuthoringError(
                "no message " + m.group(2) + " — the screen "
                "has " + str(len(content)) + " messages")
        target.reactions = list(target.reactions) + [m.group(1)]
        return new, ("added " + m.group(1) + " to message "
                     + target.id)

    m = _TITLE_RE.search(text)
    if m is not None:
        new.title = m.group(1)
        return new, "title is now " + repr(m.group(1))

    if "dark" in low:
        new.dark = True
        return new, "dark mode on"
    if "light mode" in low or low.strip() == "light":
        new.dark = False
        return new, "light mode on"
    if re.search(r"\biphone\b", low):
        new.chrome = DeviceChrome(
            style="iphone", clock=new.chrome.clock,
            battery=new.chrome.battery,
            network=new.chrome.network, unread=3)
        return new, "iphone chrome"
    if re.search(r"\bandroid\b", low):
        new.chrome = DeviceChrome(
            style="android", clock=new.chrome.clock,
            battery=new.chrome.battery,
            network=new.chrome.network, unread=0)
        return new, "android chrome"

    raise NLAuthoringError(
        "I cannot do that yet — revisions I understand: "
        "change message N to \"...\" · add Name: \"...\" · "
        "delete message N · rename A to B · message N is a "
        "photo / a voice note · react X to message N · "
        "title \"...\" · dark / light mode · iphone / android")
