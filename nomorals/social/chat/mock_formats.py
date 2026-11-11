"""Mock screens as a CONTENT format, not a standalone toy.

The mock-screens organ (:mod:`nomorals.social.chat.mock_screens`)
renders authored chats; this module is the consumer side the
content classes call: it stages a class's material — a
motivation piece, a quote, a meme's lines — as a conversation
(:class:`~.mock_screens.schema.Screen`) and hands it to the
organ's service.  The classes keep their own delivery seams,
journals and fallback ladders; the screen is one more FORMAT
rung on them, chosen the way their other formats are chosen.

Registry-driven by construction: the platform is validated
against the organ's LIVE design registry
(``mock_screens.designs.CHAT_DESIGNS``) at call time, so a
family the organ gains becomes selectable here with no new
wiring — the builders below speak only the schema.
"""

from __future__ import annotations

import re
from typing import Any

from .mock_screens import designs as _designs
from .mock_screens.schema import (
    DeviceChrome,
    Message,
    Participant,
    Screen,
)

__all__ = [
    "chat_platforms",
    "require_platform",
    "split_bubbles",
    "entries_screen",
    "quote_screen",
    "meme_screen",
    "motivation_screen",
    "render_screen_image",
]


def chat_platforms() -> tuple[str, ...]:
    """The chat families selectable right now — the organ's
    live registry keys, read at call time."""
    return tuple(_designs.CHAT_DESIGNS)


def require_platform(platform: str) -> str:
    """Validate a platform name against the live registry.

    Raises ValueError naming what exists — a caller whose
    pick isn't real falls back to its other formats, never
    renders a screen the organ can't draw."""
    name = str(platform or "").strip().lower()
    if name in _designs.CHAT_DESIGNS:
        return name
    raise ValueError(
        "no chat design called " + repr(platform or "")
        + " — designs: " + ", ".join(sorted(_designs.CHAT_DESIGNS)))


def _slug(name: str, taken: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    base = base or "someone"
    pid, n = base, 2
    while pid in taken or pid == "me":
        pid = f"{base}-{n}"
        n += 1
    taken.add(pid)
    return pid


def _clock(index: int) -> str:
    """Deterministic bubble times from 09:41, a minute apart."""
    total = 9 * 60 + 41 + max(0, int(index))
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def split_bubbles(text: str, *, max_bubbles: int = 4,
                  cap: int = 420) -> list[str]:
    """Split long text into chat-sized bubbles: paragraphs
    first, then sentences, then word-wrapped chunks — the
    segments are PACKED into bubbles of at most ``cap``
    chars, so a 200-word piece becomes a short run of
    even messages, never one wall inside a bubble and
    never a confetti of one-line bubbles either."""
    raw = str(text or "").strip()
    if not raw:
        return []
    segments: list[str] = []
    for para in re.split(r"\n\s*\n", raw):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= cap:
            segments.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= cap:
                segments.append(sent)
                continue
            words, cur = sent.split(), ""
            for word in words:
                nxt = (cur + " " + word).strip()
                if len(nxt) > cap and cur:
                    segments.append(cur)
                    cur = word
                else:
                    cur = nxt
            if cur:
                segments.append(cur)
    bubbles: list[str] = []
    for seg in segments:
        if bubbles and len(bubbles[-1]) + 1 + len(seg) <= cap:
            bubbles[-1] += " " + seg
        else:
            bubbles.append(seg)
    if len(bubbles) <= max_bubbles:
        return bubbles
    head = bubbles[: max_bubbles - 1]
    head.append(" ".join(bubbles[max_bubbles - 1:]))
    return head


def entries_screen(entries: list[dict], *,
                   platform: str = "whatsapp",
                   device: str = "android",
                   dark: bool = False,
                   reaction: str = "",
                   me_reply: str = "") -> Screen:
    """Stage card entries (``{name, text}`` — the makers
    entry shape) as a chat exchange: each distinct name is a
    participant, the first name carries the header, every
    entry is that person's bubble(s), split chat-sized.
    ``reaction`` lands on the first person's last bubble;
    ``me_reply`` is the phone owner's closing bubble."""
    platform = require_platform(platform)
    people: dict[str, Participant] = {"me": Participant("me", "You")}
    order: list[str] = []
    taken: set = {"me"}
    for entry in entries:
        name = str(entry.get("name") or "Someone").strip() or "Someone"
        if name not in order:
            pid = _slug(name, taken)
            people[name] = Participant(pid, name)
            order.append(name)
    messages: list[Message] = [
        Message("d0", "system", kind="divider", text="Today")]
    tick = 0
    first_pid = people[order[0]].id if order else "me"
    first_last_mid = ""
    for idx, entry in enumerate(entries):
        name = str(entry.get("name") or "Someone").strip() or "Someone"
        pid = people[name].id
        for j, bubble in enumerate(
                split_bubbles(str(entry.get("text") or ""))):
            mid = f"m{idx}-{j}"
            messages.append(Message(
                mid, pid, text=bubble, time=_clock(tick),
                ticks="read" if pid == "me" else ""))
            tick += 1
            if pid == first_pid:
                first_last_mid = mid
    if reaction and first_last_mid:
        for msg in messages:
            if msg.id == first_last_mid:
                msg.reactions = [reaction]
    if me_reply:
        messages.append(Message(
            "me-last", "me", text=me_reply, time=_clock(tick),
            ticks="read"))
    title = order[0] if order else "Chat"
    return Screen(
        platform=platform, title=title, subtitle="online",
        participants=list(people.values()), messages=messages,
        chrome=DeviceChrome(style=device), dark=dark)


def quote_screen(quote_text: str, author: str, *,
                 platform: str = "whatsapp",
                 device: str = "android",
                 dark: bool = False) -> Screen:
    """A quote as a chat exchange: the author sends the line
    as their message; the owner reacts and answers — the
    quote carried by a conversation, not a card."""
    return entries_screen(
        [{"name": author or "Unknown",
          "text": str(quote_text or "")}],
        platform=platform, device=device, dark=dark,
        reaction="❤️", me_reply="needed that.")


def meme_screen(entries: list[dict], *,
                platform: str = "whatsapp",
                device: str = "android",
                dark: bool = False) -> Screen:
    """A meme whose joke is the exchange itself: the named
    voice drops the line(s), the owner dies laughing — the
    chat IS the meme's delivery shape."""
    return entries_screen(
        entries, platform=platform, device=device, dark=dark,
        reaction="😂", me_reply="😂😂😂")


def motivation_screen(piece: str, *, theme: str = "",
                      window: str = "morning",
                      platform: str = "whatsapp",
                      device: str = "android") -> Screen:
    """A motivation piece staged as a conversation with
    Devon: the owner asks (the theme, or the day-part's
    ask), Devon answers in chat-sized bubbles, the owner
    receives it — a two-voice exchange carrying the lesson."""
    platform = require_platform(platform)
    theme = str(theme or "").strip()
    if theme:
        ask = f"Talk to me about {theme}."
    elif window == "evening":
        ask = "Close the day with me — what should I carry into tonight?"
    else:
        ask = "Give me something to carry into today."
    close = ("Resting on that tonight." if window == "evening"
             else "Carrying that with me today.")
    messages: list[Message] = [
        Message("d0", "system", kind="divider", text="Today"),
        Message("ask", "me", text=ask, time=_clock(0), ticks="read"),
    ]
    bubbles = split_bubbles(piece)
    for j, bubble in enumerate(bubbles):
        msg = Message(f"p{j}", "devon", text=bubble,
                      time=_clock(1 + j))
        if j == len(bubbles) - 1:
            msg.reactions = ["🙏"]
        messages.append(msg)
    messages.append(Message(
        "close", "me", text=close, time=_clock(1 + len(bubbles)),
        ticks="read"))
    return Screen(
        platform=platform, title="Devon", subtitle="online",
        participants=[Participant("me", "You"),
                      Participant("devon", "Devon")],
        messages=messages, chrome=DeviceChrome(style=device))


def render_screen_image(screen: Screen, *, media_dir: Any = ""
                        ) -> dict[str, Any]:
    """Render a staged screen to PNG bytes through the
    organ's service (no send — the caller delivers through
    its own seam).  Never raises: ``ok`` False carries the
    organ's own reason."""
    from .mock_screens import MockScreenService

    try:
        out = MockScreenService(media_dir=media_dir or "").render(screen)
    except Exception as exc:  # noqa: BLE001 - a format rung, never a wall
        return {"ok": False, "reason": str(exc)[:160]}
    if out.get("data"):
        return {"ok": True, "data": out["data"],
                "drawn": out.get("drawn") or {}}
    return {"ok": False,
            "reason": str(out.get("reason")
                          or "the screen render failed")}
