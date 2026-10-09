"""Freestyle — real-time lyrical improvisation over a beat.

Not a verse bank. The brain generates bars live, bar by bar, matching flow
and cadence to the beat's BPM and energy — the way a human freestyler rides
a beat: listening, catching the pocket, occasionally stumbling and flipping
it into something better.

- :class:`FreestyleSession` — holds the beat, the topic seed, and every bar
  spat so far. :meth:`spit` generates the next bars with full context.
- Syllable budgeting keeps bars in the pocket: the bar's syllable target is
  derived from BPM so flow matches the beat instead of floating over it.
- Human texture is *allowed, never scheduled*: the prompt permits an
  occasional stumble-and-recover, ad-libs, and breath marks — the model
  decides when, or never.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

_log = logging.getLogger(__name__)

# syllables per beat at a natural rap/sung cadence — the pocket
_SYL_PER_BEAT = 2.4


def bar_syllable_target(bpm: int, beats_per_bar: int = 4) -> tuple[int, int]:
    """Syllable budget for one bar at this tempo (min, max)."""
    seconds_per_bar = 60.0 / max(40, bpm) * beats_per_bar
    # a comfortable delivery rate: ~3.2 syllables/sec, ±25% pocket
    mid = int(seconds_per_bar * 3.2)
    lo = max(4, int(mid * 0.75))
    hi = int(mid * 1.25) + 1
    return lo, hi


_FREESTYLE_PROMPT = """You are freestyling — live, off the top, over a beat. \
This is real improvisation, not reciting.

The beat: {bpm} BPM, {beats_per_bar}/4, energy {energy:.1f}/1.0, feel: {feel}.
The topic/seed in your head: {seed!r}

Bars so far (keep the flow, rhyme, and story going — or flip it if the \
moment calls for it):
{history}

Now spit exactly {n_bars} NEW bars. Rules:
- Each bar ~{syl_lo}-{syl_hi} syllables so it sits IN THE POCKET of the beat.
- Rhyme like you mean it — internal rhymes, multis, the works. The last \
word of each bar should lock with the rhyme you established (or deliberately \
break it for effect, then come back).
- Stay on the seed's world, but let your mind wander the way freestylers \
do — wordplay, boasts, confessions, observations. Be specific and human. \
No greeting-card lines.
- Ad-libs in (parentheses) where they'd naturally fall. Breath marks as \
"..." only where a real MC would breathe.
- ONCE IN A WHILE — only when it feels genuinely natural, never forced, \
never every verse — you may stumble mid-bar and catch yourself, flipping \
the stumble into a harder line. Like: "I been... hold up, let me run it \
back — I been running since the..." If it doesn't feel natural, DON'T.
- Respond with ONLY the bars, one per line. No numbering, no commentary.
"""


@dataclass
class FreestyleSession:
    """A live freestyle session over a beat."""
    bpm: int = 92
    energy: float = 0.6
    feel: str = "laid-back but hungry"
    seed: str = ""
    beats_per_bar: int = 4
    bars: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def spit(self, n_bars: int = 4, context: Any = None) -> list[str]:
        """Generate the next ``n_bars`` bars, live. Appends to history."""
        from ..llm.base import Message, SamplingParams
        from ..llm.brain import brain_for

        n_bars = max(1, min(16, int(n_bars or 4)))
        if context is None:
            raise ValueError("an LLM context is required — freestyle is "
                             "improvised by the model, never banked")
        syl_lo, syl_hi = bar_syllable_target(self.bpm, self.beats_per_bar)
        history = "\n".join(self.bars[-16:]) or "(fresh — open with something that grabs the beat)"
        prompt = _FREESTYLE_PROMPT.format(
            bpm=self.bpm, beats_per_bar=self.beats_per_bar,
            energy=self.energy, feel=self.feel, seed=self.seed or "whatever's on your mind",
            history=history, n_bars=n_bars, syl_lo=syl_lo, syl_hi=syl_hi)
        resp = brain_for(context).chat(
            [Message.user(prompt)],
            SamplingParams(temperature=1.0, max_tokens=1024),
            task_kind="creative")
        text = (getattr(resp, "text", "") or "").strip()
        new_bars = [ln.strip(" \t-•0123456789.)") for ln in text.splitlines()
                    if ln.strip()]
        new_bars = new_bars[:n_bars]
        if not new_bars:
            raise ValueError("the model went silent mid-freestyle")
        self.bars.extend(new_bars)
        _log.info("freestyle: %d bars @ %d bpm (total %d)",
                  len(new_bars), self.bpm, len(self.bars))
        return new_bars

    def stream(self, total_bars: int = 16, chunk: int = 4,
               context: Any = None) -> Iterator[list[str]]:
        """Yield bars in chunks — a live session feel."""
        remaining = max(1, int(total_bars or 16))
        while remaining > 0:
            n = min(chunk, remaining)
            yield self.spit(n, context=context)
            remaining -= n

    def transcript(self) -> str:
        """The full session as text."""
        head = (f"# 🎤 freestyle — {self.bpm} BPM, {self.feel}\n"
                f"*seed: {self.seed or '(open)'} · {len(self.bars)} bars*\n")
        return head + "\n".join(self.bars) + "\n"


def start_session(bpm: int = 92, energy: float = 0.6, feel: str = "",
                  seed: str = "") -> FreestyleSession:
    """Open a new freestyle session."""
    return FreestyleSession(bpm=int(bpm or 92),
                            energy=max(0.0, min(1.0, float(energy or 0.6))),
                            feel=feel or "laid-back but hungry",
                            seed=(seed or "").strip())
