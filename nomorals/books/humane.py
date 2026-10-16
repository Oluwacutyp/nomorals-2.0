"""Humane structure — the book is FOR someone, the characters push back,
the narrator knows less than the author, and silence is a technique.

* ``for`` — the dedication field on the book model (see
  :mod:`nomorals.books.model` ``Book.for_whom`` and
  ``StoryState.for_whom``): who the book is for — a name, a
  relationship, an occasion.  It steers register (intimate vs formal),
  the dedication line, and what the prose explains vs assumes: a book
  for a partner assumes shared history; a book for a stranger explains
  it.  Nothing here is prescriptive about the story itself — the
  ``for`` field is context the prose is measured against, not a
  template the story must follow.

* Characters push back — :func:`character_pushback`, wired into
  :meth:`FictionWriter.validate` via ``GenreEngine.character_pushback``
  (see ``fiction.py``).  A character refuses plot convenience, and a
  character with nothing to say in a scene is a signal the scene is
  probably wrong.  Pushback is advisory (``blocked: False``) — the
  owner overrules it — but it is reported in the same breath as the
  genre validation, not buried in a side channel.

* The narrator knows less than the author — :func:`narrator_knows_less`
  flags omniscient tells (the narrator stating a hidden motive as
  fact, "little did X know" framing) so the narration stays inside
  what the POV could actually know.

* Silence as technique — :func:`silence_analysis` counts off-page
  events, evasion in dialogue, and scenes that end early, so silence
  shows up as a measured choice instead of an absence the editor has
  to guess at.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .prose import split_paragraphs

__all__ = [
    "DedicatedTo", "register_guidance", "character_pushback",
    "narrator_knows_less", "silence_analysis", "humane_report",
]

# ── for: ─────────────────────────────────────────────────────────────────────


@dataclass
class DedicatedTo:
    """Who the book is for: name, relationship, occasion.

    Stored on the book as ``for`` (JSON key ``"for"``; the Python
    attribute is ``for_whom`` because ``for`` is a keyword).
    """

    name: str = ""
    relationship: str = ""      # partner | friend | parent | child |
                                # mentor | colleague | reader | ...
    occasion: str = ""          # birthday | wedding | grief | ...
    note: str = ""              # one line the owner wants carried

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DedicatedTo":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def dedication_line(self) -> str:
        """The dedication page, from the ``for`` field."""
        if not self.name:
            return ""
        line = f"For {self.name}"
        if self.relationship:
            line += f", {self.relationship}"
        if self.occasion:
            line += f" — {self.occasion}"
        return line + "."

    def register(self) -> dict[str, Any]:
        """Register + explained-vs-assumed guidance from the ``for`` field."""
        return register_guidance(self)


_CLOSE_RELATIONSHIPS = {
    "partner", "spouse", "wife", "husband", "girlfriend", "boyfriend",
    "best friend", "best-friend", "sibling", "brother", "sister",
    "child", "daughter", "son", "parent", "mother", "father",
    "mentor", "friend",
}
_FORMAL_RELATIONSHIPS = {
    "colleague", "teacher", "professor", "client", "reader", "readers",
    "public", "stranger", "audience",
}


def register_guidance(for_whom: DedicatedTo | dict[str, Any] | None
                       ) -> dict[str, Any]:
    """Register + explained-vs-assumed steering from a ``for`` field.

    Close relationships → intimate register, shared history ASSUMED
    (inside jokes land, backstory stays off-page).  Formal / public
    relationships → formal register, context EXPLAINED (nothing
    assumed, the stranger is oriented).  Unknown → balanced.  This
    steers the dedication and the prose's assumptions; it never
    dictates plot.
    """
    if isinstance(for_whom, dict):
        for_whom = DedicatedTo.from_dict(for_whom)
    if for_whom is None:
        for_whom = DedicatedTo()
    rel = (for_whom.relationship or "").strip().lower()
    if rel in _CLOSE_RELATIONSHIPS:
        mode = "intimate"
        explained = "assumed"
        steering = [
            "shared history is assumed — inside references land without setup",
            "backstory stays off-page; the reader is trusted to know",
            "register: warm, direct, private shorthand allowed",
        ]
    elif rel in _FORMAL_RELATIONSHIPS:
        mode = "formal"
        explained = "explained"
        steering = [
            "nothing assumed — orient the stranger: who, where, why it matters",
            "context is explained on first mention, then trusted",
            "register: clear, courteous, public-facing",
        ]
    else:
        mode = "balanced"
        explained = "balanced"
        steering = [
            "explain on first mention, assume afterwards",
            "register: plain and warm, neither private shorthand nor formality",
        ]
    if for_whom.occasion:
        steering.append(
            f"occasion ({for_whom.occasion}) sets the dedication's tone, "
            "not the story's plot")
    return {
        "for": for_whom.to_dict(),
        "dedication": for_whom.dedication_line(),
        "register": mode,
        "explained_vs_assumed": explained,
        "steering": steering,
        "blocked": False,
    }


# ── characters push back ─────────────────────────────────────────────────────


_CONVENIENCE_MARKERS = (
    "suddenly knew", "just happened to", "conveniently", "as luck would have it",
    "out of nowhere", "by coincidence", "luckily", "fortunately",
    "somehow knew", "for some reason", "without knowing why",
)
_DECISION_VERBS = (
    "decided", "chose", "refused", "agreed", "promised", "vowed",
    "threatened", "confessed", "lied", "hid", "revealed", "asked",
)
_DIALOGUE_RE = re.compile(r'"[^"]{2,}"|“[^”]{2,}”')


def _character_mentions(text: str, name: str) -> list[int]:
    """Offsets where ``name`` (or first name) appears."""
    first = name.split()[0]
    return [m.start() for m in re.finditer(
        r"\b" + re.escape(first) + r"\b", text)]


def character_pushback(text: str, characters: list[dict[str, Any]] | None,
                        brief: Any = None) -> list[dict[str, Any]]:
    """Characters push back against the scene.

    For each character in the cast:

    * **nothing to say** — the character never appears in the scene:
      the scene is probably wrong (or the character doesn't belong in
      it).  Reported, not blocking.
    * **silent** — the character appears but never speaks, thinks, or
      decides: present as furniture.
    * **plot convenience** — the character's action rides a
      convenience marker ("suddenly knew", "just happened to"): the
      character refuses the convenience — the scene must earn the turn.
    * **out of voice** — the character speaks but none of their
      recorded speech tics / vocab band shows up (only when a voice
      profile exists for them).

    Returns a list of pushback notes, each with ``blocked: False``.
    The owner overrules any of them.
    """
    notes: list[dict[str, Any]] = []
    if not characters:
        return notes
    text = text or ""
    low = text.lower()
    for char in characters:
        if isinstance(char, dict):
            name = str(char.get("name", "") or "")
            tics = [str(t) for t in char.get("speech_tics", []) or []]
            want = str(char.get("want", "") or "")
            lie = str(char.get("lie", "") or "")
        else:
            name = str(getattr(char, "name", "") or "")
            tics = [str(t) for t in getattr(char, "speech_tics", []) or []]
            want = str(getattr(char, "want", "") or "")
            lie = str(getattr(char, "lie", "") or "")
        if not name:
            continue
        mentions = _character_mentions(text, name)
        if not mentions:
            notes.append({
                "character": name, "kind": "nothing_to_say",
                "note": (f"{name} has nothing to say in this scene — "
                         "they never appear. If the scene is theirs, it "
                         "is probably the wrong scene; if it isn't, "
                         "they don't belong in it."),
                "blocked": False,
            })
            continue
        # dialogue / interiority / decision presence
        window = " ".join(
            text[max(0, m - 200):m + 400] for m in mentions[:6])
        window_low = window.lower()
        speaks = bool(_DIALOGUE_RE.search(window)) and \
            any(v in window_low for v in (
                "said", "asked", "replied", "whispered", "shouted"))
        decides = any(v in window_low for v in _DECISION_VERBS)
        thinks = any(m in window_low for m in (
            "thought", "wondered", "knew", "felt", "remembered"))
        if not (speaks or decides or thinks):
            notes.append({
                "character": name, "kind": "silent",
                "note": (f"{name} appears but never speaks, thinks, or "
                         "decides — present as furniture. Give them a "
                         "line, a thought, or a refusal."),
                "blocked": False,
            })
        # plot convenience near this character
        for marker in _CONVENIENCE_MARKERS:
            if marker in low:
                # only blame the character if the marker is near a mention
                idx = low.find(marker)
                if any(abs(idx - m) < 600 for m in mentions):
                    notes.append({
                        "character": name, "kind": "plot_convenience",
                        "note": (f"{name} refuses the convenience "
                                 f"({marker!r}): the turn is unearned. "
                                 "Earn it — plant the knowledge, the "
                                 "tool, or the motive earlier."),
                        "blocked": False,
                    })
                    break
        # out-of-voice: tics recorded, dialogue present, no tic anywhere
        if tics and speaks:
            if not any(t.lower() in low for t in tics):
                notes.append({
                    "character": name, "kind": "out_of_voice",
                    "note": (f"{name} speaks but none of their recorded "
                             f"voice ({', '.join(tics[:3])}) shows up — "
                             "the line could be anyone's."),
                    "blocked": False,
                })
        # want/lie tension: a character acting directly against their
        # recorded want with no acknowledgement is worth a look
        _ = (want, lie)
    return notes


# ── narrator knows less than author ──────────────────────────────────────────


_OMNISCIENT_MARKERS = (
    r"little did (he|she|they|\w+) know",
    r"(he|she|they) would never know",
    r"what (he|she|they) didn't know was",
    r"unknown to (him|her|them)",
    r"in truth,? (he|she|they)",
    r"the truth was",
    r"deep down,? (he|she|they)",
    r"unbeknownst to",
    r"years later,? (he|she|they) would",
    r"it would be (his|her|their) undoing",
)


def narrator_knows_less(text: str) -> dict[str, Any]:
    """Flag omniscient tells: the narrator claiming more than the POV knows.

    Reports ``{omniscient_claims, blocked: False}``.  The fix is
    never to delete knowledge from the story — it is to move it into
    a head that could hold it, or into a scene that shows it.
    """
    claims: list[dict[str, Any]] = []
    low = text or ""
    for pat in _OMNISCIENT_MARKERS:
        for m in re.finditer(pat, low, re.IGNORECASE):
            claims.append({
                "pattern": pat,
                "excerpt": " ".join(
                    low[max(0, m.start() - 60):m.end() + 60].split()),
            })
    return {
        "omniscient_claims": claims,
        "count": len(claims),
        "blocked": False,
        "note": ("The narrator knows less than the author. Move each "
                 "claim into a head that could hold it, or a scene "
                 "that shows it."),
    }


# ── silence as technique ─────────────────────────────────────────────────────


_EVASION_MARKERS = (
    "never mind", "forget it", "it's nothing", "it's fine",
    "don't worry about it", "let's not", "drop it", "not now",
    "i can't talk about", "we don't talk about",
)
_OFF_PAGE_MARKERS = (
    "never spoke of", "never mentioned", "was never spoken of",
    "what happened next", "behind closed doors", "off-screen",
    "later, they would", "much later", "the rest is silence",
)


def silence_analysis(text: str) -> dict[str, Any]:
    """Measure silence as a technique: off-page, evasion, ending early."""
    paras = split_paragraphs(text or "")
    low = (text or "").lower()
    evasion = sum(low.count(m) for m in _EVASION_MARKERS)
    off_page = sum(low.count(m) for m in _OFF_PAGE_MARKERS)
    trailing = low.count("…") + low.count("...")
    dash_cut = len(re.findall(r"[—–-]\s*$", text or "", re.MULTILINE))
    # scenes ending early: a paragraph ending on ellipsis / dash
    ends_early = sum(1 for p in paras
                     if p.strip().endswith(("…", "...", "—", "–", "-")))
    return {
        "evasion_count": evasion,
        "off_page_count": off_page,
        "trailing_off_count": trailing,
        "cut_off_count": dash_cut,
        "paragraphs_ending_early": ends_early,
        "paragraphs": len(paras),
        "silence_present": bool(evasion or off_page or ends_early),
        "blocked": False,
        "note": ("Silence is a technique: off-page events, evasion, and "
                 "scenes that end early are choices, measured here so "
                 "they read as deliberate instead of accidental."),
    }


def humane_report(text: str, characters: list[dict[str, Any]] | None = None,
                  for_whom: DedicatedTo | dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    """One advisory humane-structure report for a passage."""
    pushback = character_pushback(text, characters)
    narrator = narrator_knows_less(text)
    silence = silence_analysis(text)
    return {
        "for": register_guidance(for_whom),
        "character_pushback": pushback,
        "narrator_knows_less": narrator,
        "silence": silence,
        "blocked": False,
        "note": "Humane structure reports; it never blocks.",
    }
