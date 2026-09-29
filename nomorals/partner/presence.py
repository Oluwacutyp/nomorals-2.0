"""Human presence: typing pace and the busy/distracted gaps.

A bot replies in a fixed time with a fixed typing indicator. A person:

* **types at a rate their thumb and mood allow** — a 300-character message
  shows ~a minute of typing, a "k" shows a second; tired = slower, excited
  = faster, and every duration carries jitter.
* **is sometimes busy** — a substantive message can get a reply minutes
  later (she was mid-call, mid-shift, mid-argument with the router), and a
  low-stakes "k"/"lol" can be read and left without a reply at all.

Everything here is pure and takes an injected ``random.Random`` so the
probabilities are testable and reproducible.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "Presence",
    "decide_presence",
    "human_typing_seconds",
    "is_low_content",
    "thinking_typing_seconds",
]

#: A message that costs a human zero emotional processing: acknowledgment,
#: laugh, filler. These — and only these — may be read and left unanswered.
_LOW_CONTENT = re.compile(
    r"^(ok|okay|kk?|mhm|hmm+|lol+|lmao|haha+|hah|sure|fine|yeah+|yep|yup|no|nope|"
    r"yes|yess?|uh+|um+|cool|nice|thx|thanks|ty|👍|🤷|😂)\.?[\s!?\-…]*$",
    re.IGNORECASE,
)

#: Mood labels where silence costs nothing — she is already closed off,
#: curt, or drained, so a gap reads as personality, not as a fight.
_CLOSED_OFF = {"tired", "annoyed", "irritated", "cold", "distant", "sad", "vulnerable"}
#: Labels where an eager person replies fast — low chance of a busy gap.
_EAGER = {"happy", "excited", "affectionate", "playful", "proud"}

#: (min, max) seconds for a "busy" gap. Log-uniform inside: most gaps are
#: 1–4 minutes, the long tail reaches half an hour.
DELAY_RANGE = (60.0, 900.0)
LONG_TAIL_CHANCE = 0.1
LONG_TAIL_RANGE = (900.0, 1800.0)


@dataclass(frozen=True)
class Presence:
    """The presence decision for one inbound message."""

    reply: bool
    delay_seconds: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "delay_seconds": round(self.delay_seconds, 1),
            "reason": self.reason,
        }


def is_low_content(text: str) -> bool:
    """True for the 'k' / 'lol' / '👍' class of messages."""
    return bool(_LOW_CONTENT.match((text or "").strip()))


def human_typing_seconds(
    text: str,
    *,
    mood: Mapping[str, float] | None = None,
    rng: random.Random | None = None,
    minimum: float = 2.0,
    cap: float = 45.0,
) -> float:
    """How long a human takes to type ``text`` right now.

    Model: a short settle-in pause, then chars-per-second drawn from a band
    the mood widens or narrows (drained = slow thumb; excited = fast
    thumbs), with per-message jitter. Durations are anchored to how people
    actually text on a phone — a quick "yeah" takes a second or two, a
    three-line message takes ten to twenty, a wall of text takes most of a
    minute. The result is the typing indicator duration: deliberately the
    *whole* interval, because that is what a phone shows. Longer messages
    always type longer; nothing is instant, nothing is absurd.
    """
    rng = rng or random
    mood = mood or {}
    energy = float(mood.get("energy", 60))
    happiness = float(mood.get("happiness", 60))

    # chars/second band by mood (phone-thumb territory, not typist territory)
    if energy < 30:
        cps_lo, cps_hi = 4.0, 9.0
    elif energy > 78 and happiness > 70:
        cps_lo, cps_hi = 14.0, 26.0
    else:
        cps_lo, cps_hi = 8.0, 18.0
    cps = rng.uniform(cps_lo, cps_hi)

    chars = max(0, len(text or ""))
    settle_pause = rng.uniform(0.4, 1.4) + min(1.5, chars / 150.0)
    duration = settle_pause + chars / cps
    duration *= rng.uniform(0.85, 1.25)  # humans are not metronomes
    return max(minimum, min(cap, duration))


def thinking_typing_seconds(
    text: str,
    *,
    rng: random.Random | None = None,
    minimum: float = 1.5,
    cap: float = 8.0,
) -> float:
    """How long she stays in "typing" while she READS + THINKS.

    Shown before the reply is ready (the model call is in flight). A person
    glancing at two words starts composing almost at once; a long incoming
    paragraph gets a longer read before the thumbs move. Kept short on
    purpose — this is the lead-in, and the real per-message typing follows
    when the reply actually goes out.
    """
    rng = rng or random
    chars = max(0, len(text or ""))
    duration = rng.uniform(1.0, 2.5) + min(4.0, chars / 150.0)
    duration *= rng.uniform(0.8, 1.2)
    return max(minimum, min(cap, duration))


def decide_presence(
    mood: Any,
    text: str,
    *,
    chat_kind: str = "dm",
    is_owner: bool = False,
    rng: random.Random | None = None,
) -> Presence:
    """Will she answer right now, later, or not at all?

    ``mood`` is a :class:`~nomorals.partner.mood.MoodState` (anything with
    ``.label`` and ``.values``). Rules, in order:

    * low-stakes messages (``k``, ``lol``) may be **read and left** — the
      chance rises when she is closed off and falls when she is eager;
    * substantive messages are **never ignored**, but may be **delayed**
      (busy/distracted) — a 1–15 minute gap, with a long tail to 30;
    * the **owner** gets half the chances: you can be distracted around
      your person, but you don't leave them on read on a schedule.
    """
    rng = rng or random
    label = getattr(mood, "label", "") or ""

    if is_low_content(text):
        if label in _CLOSED_OFF:
            ignore_chance = 0.25
        elif label in _EAGER:
            ignore_chance = 0.03
        else:
            ignore_chance = 0.10
        if is_owner:
            ignore_chance *= 0.5
        if rng.random() < ignore_chance:
            return Presence(reply=False, reason="read, left — low stakes")

    # Busy gap: she is mid-something. Distraction is more human than
    # instant, but it must stay a rarity — mostly.
    if label in _CLOSED_OFF:
        busy_chance = 0.15
    elif label in _EAGER:
        busy_chance = 0.03
    else:
        busy_chance = 0.07
    if is_owner:
        busy_chance *= 0.5

    if rng.random() < busy_chance:
        if rng.random() < LONG_TAIL_CHANCE:
            delay = rng.uniform(*LONG_TAIL_RANGE)
            reason = "really busy — long gap"
        else:
            delay = rng.uniform(*DELAY_RANGE)
            reason = "busy / distracted"
        return Presence(reply=True, delay_seconds=delay, reason=reason)

    return Presence(reply=True, reason="present")
