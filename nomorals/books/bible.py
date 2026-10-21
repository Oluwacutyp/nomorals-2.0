"""Story bibles — the machine-readable soul of a story.

A bible is auto-built from chapters Devon has actually read (fetched via
the reader or pasted by the owner): the cast, the open plot threads, the
world's rules, and the story's voice (POV, tense, tone, rhythm).  The
continuation engine and the fiction writer both steer by it, so a story
continued in chapter 400 still sounds like itself and never forgets who
is alive.

Two build paths, model-first like the rest of BookForge:

* model path — the live model extracts a structured bible as JSON;
* heuristic floor — a real NLP-lite pipeline (name mining, dialogue
  attribution, goal-statement mining, tense/dialogue/rhythm profiling)
  that produces a genuinely useful bible with no model at all.

Bibles persist as ``workspace/books/bibles/<story-slug>.json`` and are
updated incrementally as new chapters are digested.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .model import slugify
from .voiceprint import Voiceprint, fingerprint

_log = get_logger(__name__)

__all__ = [
    "StoryBible", "BibleCharacter", "PlotThread",
    "BibleBuilder", "build_bible", "bible_needs_redigest",
    "LorebookEntry", "Progression", "ArcRecap", "BibleLocation",
    "DIGESTION_VERSION",
]

#: Digestion generations.  0 = legacy digestion (pre-grounding): it
#: mined cast, threads and rules and claimed every chapter digested,
#: but it never produced the grounding fields — character states, arc
#: recaps, locations, current situation — so a bible from that era can
#: claim chapter 2,545 digested while its memory is the story's
#: opening.  1 = the grounding fold (:meth:`BibleBuilder._fold_digest`).
#: build/update stamp the generation when their fold completes;
#: bibles stamped current are never re-digested (see
#: :func:`bible_needs_redigest`).
DIGESTION_VERSION = 1


# ── lorebook (NovelAI-style keyed entries) ────────────────────────────────────
#
# The bible is the full record; the LOREBOOK is the working memory injected
# into generation prompts.  Mined from NovelAI's lorebook + AI Dungeon's
# story cards + 1667's facts:
# - an entry surfaces only when one of its keys appears in recent text
#   (or when it is always_on / enabled);
# - keys are case-insensitive, ``/regex/`` for regex, ``a & b`` = AND;
# - chain activation: an active entry's text can trigger other entries;
# - always_on facts are trimmed by budget (stale always-on steers wrong).


_LORE_KINDS = ("character", "location", "item", "concept", "rule",
               "faction", "event", "style")


def _key_matches(key: str, text: str) -> bool:
    """One lorebook key against text.

    ``/regex/`` → regex (case-sensitive).  ``a & b`` → all parts must
    match.  Otherwise case-insensitive whole-word match.
    """
    key = key.strip()
    if not key:
        return False
    if "&" in key and not (key.startswith("/") and key.endswith("/")):
        return all(_key_matches(part, text) for part in key.split("&"))
    if len(key) >= 2 and key.startswith("/") and key.endswith("/"):
        try:
            return re.search(key[1:-1], text) is not None
        except re.error:
            return False
    return re.search(r"\b" + re.escape(key) + r"\b", text,
                     flags=re.IGNORECASE) is not None


@dataclass
class LorebookEntry:
    """A NovelAI-style keyed lore entry: injected only when relevant."""
    name: str
    text: str = ""
    kind: str = "concept"          # character|location|item|concept|rule|…
    keys: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    always_on: bool = False
    enabled: bool = True
    hidden: bool = False          # concealed from readers, not the model
    order: int = 100              # injection order (lower = earlier)

    def __post_init__(self) -> None:
        if self.kind not in _LORE_KINDS:
            self.kind = "concept"

    def matches(self, text: str) -> bool:
        if not self.enabled or not text:
            return False
        hay = text
        forms = [self.name, *self.aliases]
        if any(_key_matches(f, hay) for f in forms):
            return True
        return any(_key_matches(k, hay) for k in self.keys)

    def render(self) -> str:
        label = self.name if not self.hidden else "⟦hidden⟧"
        return f"[{self.kind}] {label}: {self.text}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LorebookEntry":
        return cls(
            name=str(d.get("name", "")),
            text=str(d.get("text", "")),
            kind=str(d.get("kind", "concept")),
            keys=[str(k) for k in d.get("keys", [])],
            aliases=[str(a) for a in d.get("aliases", [])],
            always_on=bool(d.get("always_on", False)),
            enabled=bool(d.get("enabled", True)),
            hidden=bool(d.get("hidden", False)),
            order=int(d.get("order", 100)),
        )


@dataclass
class Progression:
    """A timeline-tracked detail change (Novelcrafter Progressions).

    Outdated lore is overwritten by newer progressions, keeping the bible
    accurate to the current story moment instead of the first draft.
    """
    name: str
    detail: str
    chapter: int = 0
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Progression":
        return cls(name=str(d.get("name", "")),
                   detail=str(d.get("detail", "")),
                   chapter=int(d.get("chapter", 0)),
                   recorded_at=str(d.get("recorded_at", "")))


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class BibleCharacter:
    name: str
    role: str = "supporting"          # protagonist | antagonist | supporting
    description: str = ""
    traits: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    first_seen: int = 0
    mentions: int = 0
    #: CURRENT state as of the last digested chapter — the latest
    #: state-bearing sentence about this character (injury, possession,
    #: whereabouts, status change).  Empty until digestion finds one.
    #: This is what stops a continuation from writing chapter 2,545's
    #: hero as his chapter-1 self.
    state: str = ""
    state_chapter: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArcRecap:
    """One digested segment of the story: chapters ``start``–``end``.

    Built incrementally as chapters are digested (never just from the
    opening), so the bible remembers the LATEST arcs, not only how the
    story began.
    """
    start: int = 0
    end: int = 0
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArcRecap":
        return cls(start=int(d.get("start", 0)), end=int(d.get("end", 0)),
                   text=str(d.get("text", "")))


@dataclass
class BibleLocation:
    """A place the story actually visits, with its latest context."""
    name: str = ""
    note: str = ""                    # latest sentence placing action there
    last_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BibleLocation":
        return cls(name=str(d.get("name", "")),
                   note=str(d.get("note", "")),
                   last_seen=int(d.get("last_seen", 0)))


@dataclass
class PlotThread:
    id: str
    summary: str
    status: str = "open"              # open | resolved
    introduced: int = 0
    last_seen: int = 0
    heat: float = 0.0                 # narrative charge: mentions × recency

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoryBible:
    story_slug: str
    title: str = ""
    pov: str = "third"                # first | third | omniscient
    tense: str = "past"               # past | present
    tone: list[str] = field(default_factory=list)
    voice_notes: str = ""
    characters: list[BibleCharacter] = field(default_factory=list)
    threads: list[PlotThread] = field(default_factory=list)
    world_rules: list[str] = field(default_factory=list)
    arc_summary: str = ""
    chapters_digested: int = 0
    #: per-segment recaps from digestion, in chapter order — the
    #: bible's memory of the WHOLE story, latest arcs included
    recaps: list[ArcRecap] = field(default_factory=list)
    #: places the story visits, with latest context per place
    locations: list[BibleLocation] = field(default_factory=list)
    #: where the story STOPPED at the last digested chapter — the
    #: closing beat of the newest chapter, refreshed by every digest
    current_situation: str = ""
    #: digestion generation that produced this bible's grounding
    #: (see DIGESTION_VERSION); absent/0 in bible files written
    #: before the grounding fold existed
    digestion_version: int = 0
    #: highest chapter the grounding fold has covered, in order.
    #: Equals chapters_digested whenever grounding is current; it
    #: lags behind only while a legacy re-digest is in flight, and
    #: tells that re-digest exactly where to resume.
    grounding_through: int = 0
    built_at: str = ""
    #: NovelAI-style keyed entries (working memory for generation)
    lore: list[LorebookEntry] = field(default_factory=list)
    #: timeline-tracked detail changes (Novelcrafter Progressions)
    progressions: list[Progression] = field(default_factory=list)
    #: Measured voiceprint of the digested chapters — the VOICE TARGET
    #: new chapters are compared against (see :mod:`nomorals.books.voiceprint`).
    #: Stored as a plain dict (``Voiceprint.to_dict()``); empty ``{}``
    #: means "no voice measured yet".  Failures measuring it leave it
    #: empty — they never break a build.
    voiceprint: dict[str, Any] = field(default_factory=dict)

    @property
    def voice(self) -> Voiceprint | None:
        """The bible's voice target as a :class:`Voiceprint`, or None."""
        if not self.voiceprint:
            return None
        vp = Voiceprint.from_dict(self.voiceprint)
        return vp if vp.words > 0 else None

    def voice_target(self) -> list[str]:
        """Prompt lines steering generation toward the bible's voice.

        Empty when no voiceprint has been measured yet — a missing
        target steers nothing, it never fabricates one.
        """
        vp = self.voice
        if vp is None:
            return []
        return [
            "VOICE TARGET — write in the story's measured voice:",
            f"  {vp.describe()}",
            "  Match this fingerprint (sentence length, dialogue ratio, "
            "register, sensory balance, filtering). Drift is reported, "
            "never blocking.",
        ]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoryBible":
        chars = [BibleCharacter(**{k: v for k, v in c.items()
                                   if k in BibleCharacter.__dataclass_fields__})
                 for c in d.get("characters", [])]
        threads = [PlotThread(**t) for t in d.get("threads", [])]
        lore = [LorebookEntry.from_dict(e) for e in d.get("lore", [])]
        progs = [Progression.from_dict(p) for p in d.get("progressions", [])]
        recaps = [ArcRecap.from_dict(r) for r in d.get("recaps", [])]
        locations = [BibleLocation.from_dict(x)
                     for x in d.get("locations", [])]
        return cls(
            story_slug=d.get("story_slug", ""), title=d.get("title", ""),
            pov=d.get("pov", "third"), tense=d.get("tense", "past"),
            tone=list(d.get("tone", [])), voice_notes=d.get("voice_notes", ""),
            characters=chars, threads=threads,
            world_rules=list(d.get("world_rules", [])),
            arc_summary=d.get("arc_summary", ""),
            chapters_digested=int(d.get("chapters_digested", 0)),
            built_at=d.get("built_at", ""), lore=lore, progressions=progs,
            recaps=recaps, locations=locations,
            current_situation=str(d.get("current_situation", "")),
            digestion_version=int(d.get("digestion_version", 0) or 0),
            grounding_through=int(d.get("grounding_through", 0) or 0),
            voiceprint=dict(d.get("voiceprint") or {}))

    def character(self, name: str) -> BibleCharacter | None:
        low = name.lower()
        for c in self.characters:
            if c.name.lower() == low:
                return c
        return None

    def open_threads(self) -> list[PlotThread]:
        return [t for t in self.threads if t.status == "open"]

    def grounding_reaches_end(self) -> bool:
        """True when the recorded grounding demonstrably covers the
        last digested chapter: the latest recap ends there and the
        current situation is that chapter's closing beat.  Only the
        grounding fold produces either, so a bible passing this check
        was grounded by it — the data backstop for bibles written
        before the digestion_version stamp existed."""
        if self.chapters_digested < 1 or not self.recaps:
            return False
        if self.recaps[-1].end < self.chapters_digested:
            return False
        return self.current_situation.startswith(
            f"End of chapter {self.chapters_digested}:")

    def brief(self) -> str:
        """Compact steering text for generation prompts."""
        lines = [f"STORY: {self.title} (POV: {self.pov}, tense: {self.tense})"]
        if self.tone:
            lines.append("Tone: " + ", ".join(self.tone))
        if self.voice_notes:
            lines.append("Voice: " + self.voice_notes)
        lines.extend(self.voice_target())
        if self.world_rules:
            lines.append("World rules:")
            lines += [f"  - {r}" for r in self.world_rules[:12]]
        if self.characters:
            lines.append("Cast:")
            for c in self.characters[:12]:
                rel = f" ({'; '.join(c.relationships[:3])})" if c.relationships else ""
                lines.append(f"  - {c.name} [{c.role}]{rel}: "
                             f"{c.description[:160]}")
                if c.state:
                    lines.append(f"    now (ch.{c.state_chapter}): "
                                 f"{c.state[:180]}")
        open_t = self.open_threads()
        if open_t:
            lines.append("Open plot threads:")
            lines += [f"  - {t.summary[:160]}" for t in open_t[:10]]
        if self.locations:
            recent = sorted(self.locations, key=lambda x: x.last_seen,
                            reverse=True)[:6]
            lines.append("Places: " + "; ".join(
                f"{x.name} (ch.{x.last_seen})" for x in recent))
        if self.arc_summary:
            lines.append("So far: " + self.arc_summary[:800])
        if self.recaps:
            lines.append("Latest arcs:")
            lines += [f"  - ch.{r.start}–{r.end}: {r.text[:200]}"
                      for r in self.recaps[-3:]]
        if self.current_situation:
            lines.append("Where the story stopped: "
                         + self.current_situation[:400])
        return "\n".join(lines)

    # ── lorebook ──────────────────────────────────────────────────────
    def _lore(self) -> list["LorebookEntry"]:
        return list(getattr(self, "lore", []) or [])

    def inject(self, recent_text: str, max_chars: int = 2500,
               scan_chars: int = 4000) -> str:
        """Assemble the keyed lorebook injection for a generation prompt.

        AI-Dungeon canonical order: always-on facts first, then entries
        whose keys appear in the recent text (chain activation included),
        trimmed to ``max_chars``.  Returns the prompt-ready block (empty
        string when nothing qualifies).
        """
        entries = [e for e in self._lore() if e.enabled]
        if not entries:
            return ""
        window = (recent_text or "")[-scan_chars:]
        active: list[LorebookEntry] = []
        seen: set[str] = set()

        def _fire(e: LorebookEntry) -> None:
            if e.name in seen:
                return
            seen.add(e.name)
            active.append(e)

        # pass 1: always-on
        for e in entries:
            if e.always_on:
                _fire(e)
        # pass 2: keyed scan with TRUE chain activation — the text of an
        # active entry can trigger other entries (NovelAI/1667 behavior)
        frontier = [e.text for e in active]
        for _ in range(3):
            progressed = False
            scan_texts = [window] + frontier
            for e in entries:
                if e.name in seen:
                    continue
                if any(e.matches(t) for t in scan_texts):
                    _fire(e)
                    frontier.append(e.text)
                    progressed = True
            if not progressed:
                break
        if not active:
            return ""
        lines = ["LORE:"]
        budget = max_chars
        for e in sorted(active, key=lambda x: x.order):
            chunk = e.render()
            if len(chunk) > budget:
                chunk = chunk[:budget].rsplit(" ", 1)[0] + "…"
            lines.append(chunk)
            budget -= len(chunk)
            if budget <= 200:
                break
        return "\n".join(lines).strip()

    def mention_index(self, chapters: list[tuple[int, str, str]]) -> dict[str, list[int]]:
        """Novelcrafter-style global mapping: name → chapter numbers.

        Uses characters + lorebook entries (with aliases) as the index
        vocabulary.
        """
        vocab: dict[str, str] = {}  # surface form → canonical name
        for c in self.characters:
            vocab[c.name.lower()] = c.name
        for e in self._lore():
            vocab[e.name.lower()] = e.name
            for a in e.aliases:
                vocab[a.lower()] = e.name
        index: dict[str, set[int]] = {}
        for n, _t, text in chapters:
            low = text.lower()
            for surface, canon in vocab.items():
                if re.search(r"\b" + re.escape(surface) + r"\b", low):
                    index.setdefault(canon, set()).add(n)
        return {k: sorted(v) for k, v in index.items()}

    def add_progression(self, name: str, detail: str,
                        chapter: int = 0) -> "Progression":
        """Record a timeline-tracked detail change (Novelcrafter Progressions)."""
        progs = getattr(self, "progressions", None)
        if progs is None:
            self.progressions = progs = []
        p = Progression(name=name, detail=detail, chapter=chapter,
                        recorded_at=datetime.now(timezone.utc).isoformat(
                            timespec="seconds"))
        progs.append(p)
        return p

    def progressions_for(self, name: str) -> list["Progression"]:
        return [p for p in (getattr(self, "progressions", []) or [])
                if p.name.lower() == name.lower()]

    def consistency_check(self) -> list[str]:
        """Flag bible rot: always-on bloat, keyless entries, stale threads."""
        flags: list[str] = []
        entries = self._lore()
        always = [e for e in entries if e.always_on and e.enabled]
        if len(always) > 8:
            flags.append(f"always-on bloat: {len(always)} entries always "
                         "injected — trim to the essentials or the model "
                         "will steer off them")
        for e in entries:
            if e.enabled and not e.always_on and not e.keys:
                flags.append(f"entry {e.name!r} is enabled but has no keys "
                             "and is not always-on — it will never fire")
        for t in self.threads:
            if t.status == "open" and t.heat <= 0 and t.last_seen:
                flags.append(f"thread {t.id} open but cold "
                             f"(last seen ch.{t.last_seen}) — resolve or heat it")
        return flags

    def export_lorebook(self) -> dict[str, Any]:
        """Portable lorebook (NovelAI-style import/export)."""
        return {
            "story": self.story_slug,
            "title": self.title,
            "exported_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "entries": [e.to_dict() for e in self._lore()],
        }

    def import_lorebook(self, data: dict[str, Any]) -> int:
        """Merge an exported lorebook. Returns entries added."""
        lore = getattr(self, "lore", None)
        if lore is None:
            self.lore = lore = []
        have = {e.name.lower() for e in lore}
        added = 0
        for d in data.get("entries", []):
            e = LorebookEntry.from_dict(d)
            if e.name.lower() not in have:
                lore.append(e)
                have.add(e.name.lower())
                added += 1
        return added


def bible_needs_redigest(bible: StoryBible) -> bool:
    """True for a LEGACY bible: it claims chapters digested, but its
    grounding predates the grounding fold — character states, recaps,
    locations and the current situation stand empty or frozen behind
    the claimed end, so continuations ground on the story's opening.

    Never misfires on a grounded bible: one stamped with the current
    DIGESTION_VERSION is current by construction, and an unstamped
    one whose grounding demonstrably reaches its last digested
    chapter (written by the fold before the stamp existed) is
    recognised from its data and only gets stamped, never refolded.
    """
    if bible.digestion_version >= DIGESTION_VERSION:
        return False
    if bible.chapters_digested < 1:
        return False
    return not bible.grounding_reaches_end()


# ── heuristic floor ──────────────────────────────────────────────────────────
#
# Real extraction without a model: name mining with dialogue-attribution
# evidence, goal-statement thread mining, rule-pattern mining, and a
# style profiler.  Deterministic, test-covered, genuinely useful.

_STOP_NAMES = frozenset("""
    The And But For With From That This When Then There Here What Where Who
    Why How His Her Their Our Your Its Chapter Volume Part Book One Two
    Three Four Five Six Seven Eight Nine Ten Lord Lady King Queen Sir
    Master Young Miss Mister Missus Doctor Captain General System Host
    Ding Congratulations Quest Reward Skill Level Status Heaven Earth
    Divine Ancient Great Little Old Big New First Last Next Previous
    After Before During While Until Since Because Although Though However
    Meanwhile Later Soon Now Today Tonight Morning Evening Night Still Just
    Even Only Also Well Oh Yes No So Suddenly They Them Theirs We You Your
    Him Her She He It Itself Himself Herself Themselves Someone Something
    Everyone Everybody Nobody Nothing Anything Everything Each Every
    Another Other Such Same Very Much Many More Most Less Least Own
    Against Between Among Through During Without Within Upon Into Out
    Over Under Again Once Twice Thrice
""".split())

#: honorifics stripped from the front of mined names ("Elder Sable" → "Sable")
_TITLE_WORDS = frozenset("""
    Elder Lord Lady King Queen Sir Master Doctor Captain General Old Young
    Miss Mister Mistress Saint Father Mother Brother Sister
""".split())

_DIALOGUE_VERBS = (
    "said", "asked", "replied", "shouted", "whispered", "muttered",
    "growled", "laughed", "sighed", "exclaimed", "demanded", "answered",
    "snarled", "hissed", "yelled", "murmured", "declared", "warned",
)

_CONFLICT_WORDS = frozenset("""
    kill killed death died battle fight fought war enemy enemies rival
    revenge betray betrayed murder assassin duel clash destroy destroyed
    hunt hunted trap ambush poison curse
""".split())

_RESOLVE_WORDS = frozenset("""
    defeated victory won succeeded finally avenged resolved ended over
    triumph peace restored
""".split())

_PAST_VERBS = frozenset("""
    was were had did said went came saw took made knew thought felt
    looked turned walked stood sat lay began became brought built
    caught chose fought found gave grew heard kept left lost met
    paid ran rose spoke stood understood wrote
""".split())

_PRESENT_VERBS = frozenset("""
    is are has have does says goes comes sees takes makes knows thinks
    feels looks turns walks stands sits lies begins becomes brings
    builds catches chooses fights finds gives grows hears keeps leaves
    loses meets pays runs rises speaks understands writes am
""".split())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", text)
    return [p.strip() for p in parts if len(p.strip()) > 12]


def _name_candidates(text: str) -> Counter:
    """Capitalized name sequences, weighted by dialogue-attribution evidence.

    A candidate scores +1 per mid-sentence occurrence and +2 when it
    appears next to a dialogue verb ("X said", "said X") — the strongest
    signal a token is a character, not a place or title word.
    """
    def _clean(raw: str) -> str:
        parts = raw.split()
        while parts and parts[0] in _TITLE_WORDS:
            parts.pop(0)
        name = " ".join(parts)
        if not name or name in _STOP_NAMES:
            return ""
        if all(w in _STOP_NAMES for w in parts):
            return ""
        return name

    # name-internal separator: spaces/tabs only -- never newlines, so a
    # chapter heading can't fuse with the next paragraph's first name
    _SEP = r"[ \t]+"
    scores: Counter = Counter()
    # dialogue attribution: "X said" / "said X" / "X," she said
    for verb in _DIALOGUE_VERBS:
        for m in re.finditer(
                rf"\b([A-Z][a-z]+(?:{_SEP}[A-Z][a-z]+){{0,2}})\b{_SEP}{verb}\b",
                text):
            name = _clean(m.group(1))
            if name:
                scores[name] += 2
        for m in re.finditer(
                rf"\b{verb}\b{_SEP}([A-Z][a-z]+(?:{_SEP}[A-Z][a-z]+){{0,2}})\b",
                text):
            name = _clean(m.group(1))
            if name:
                scores[name] += 2
    # mid-sentence capitalized sequences (not sentence- or
    # paragraph-initial -- \n is deliberately excluded from the lookbehind)
    for m in re.finditer(
            r"(?<=[a-z,;: \u201c\u201d])\b([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,}){0,2})\b",
            text):
        name = _clean(m.group(1))
        if name:
            scores[name] += 1
    return scores


def _mine_characters(chapters: list[tuple[int, str, str]]) -> list[BibleCharacter]:
    """chapters: (number, title, text)."""
    per_chapter: list[Counter] = []
    first_seen: dict[str, int] = {}
    for number, _title, text in chapters:
        counts = _name_candidates(text)
        per_chapter.append(counts)
        for name in counts:
            first_seen.setdefault(name, number)
    total: Counter = Counter()
    for counts in per_chapter:
        total.update(counts)
    # keep names with real evidence; drop single-occurrence noise
    names = [n for n, s in total.most_common(40) if s >= 3]
    if not names:
        names = [n for n, s in total.most_common(8) if s >= 2]
    # description: best sentences mentioning the name
    char_sents: dict[str, list[str]] = {n: [] for n in names}
    for _number, _title, text in chapters:
        for sent in _sentences(text):
            for name in names:
                if name in sent and len(char_sents[name]) < 6:
                    char_sents[name].append(sent)
    ordered = sorted(names, key=lambda n: total[n], reverse=True)
    characters: list[BibleCharacter] = []
    for i, name in enumerate(ordered[:24]):
        sents = char_sents[name]
        desc = " ".join(sents[:2])[:300]
        # role: most-mentioned + early = protagonist; conflict-adjacent = antagonist
        conflict_hits = sum(
            1 for s in sents
            if _CONFLICT_WORDS & set(re.findall(r"[a-z]+", s.lower())))
        if i == 0 and first_seen.get(name, 99) <= 2:
            role = "protagonist"
        elif conflict_hits >= 2 and i > 2:
            role = "antagonist"
        else:
            role = "supporting"
        # relationships: other names co-occurring in the same sentences
        rels: Counter = Counter()
        for sent in sents:
            for other in ordered:
                if other != name and other in sent:
                    rels[other] += 1
        relationships = [f"{other}" for other, _ in rels.most_common(4)]
        # traits: adjectives near the name
        traits: list[str] = []
        for sent in sents[:4]:
            for m in re.finditer(
                    rf"\b([a-z]+)\s+{re.escape(name.split()[0])}\b", sent):
                if m.group(1) not in {"the", "a", "an", "his", "her", "their"}:
                    traits.append(m.group(1))
        characters.append(BibleCharacter(
            name=name, role=role, description=desc,
            traits=list(dict.fromkeys(traits))[:5],
            relationships=relationships,
            first_seen=first_seen.get(name, 0), mentions=total[name]))
    return characters


_GOAL_RES: list[re.Pattern] = []

_GOAL_PATTERNS = (
    r"\b[Ii] must\s+([^.!?]{8,120})",
    r"\b[Ww]e must\s+([^.!?]{8,120})",
    r"\b[Ii] will\s+([^.!?]{8,120})",
    r"\b[Ww]e need to\s+([^.!?]{8,120})",
    r"\b[Ii] need to\s+([^.!?]{8,120})",
    r"\b[Ii] have to\s+([^.!?]{8,120})",
    r"\b[Ww]e have to\s+([^.!?]{8,120})",
    r"\b[Tt]he (?:quest|mission|plan|goal) (?:is|was) to\s+([^.!?]{8,120})",
    r"\b[Ss]wore (?:to|that)\s+([^.!?]{8,120})",
    r"\bvow(?:ed)?\s+([^.!?]{8,120})",
)

_GOAL_RES = [re.compile(p) for p in _GOAL_PATTERNS]


def _mine_threads(chapters: list[tuple[int, str, str]]) -> list[PlotThread]:
    found: dict[str, PlotThread] = {}
    for number, _title, text in chapters:
        for pat in _GOAL_PATTERNS:
            for m in re.finditer(pat, text):
                goal = re.sub(r"\s+", " ", m.group(1)).strip(" .")
                # strip trailing dialogue attribution: '...," Quinn said'
                goal = re.sub(
                    r',?\s*"?\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[a-z]+)'
                    r"\s+(?:said|asked|replied|whispered|shouted|muttered|"
                    r"vowed|promised|swore|declared|warned|answered)\.?$",
                    "", goal).strip(" .")
                key = " ".join(sorted(set(
                    w.lower() for w in re.findall(r"[a-z]{4,}", goal))))[:80]
                if not key or len(goal) < 10:
                    continue
                thread = found.get(key)
                if thread is None:
                    tid = f"t{len(found) + 1}"
                    found[key] = PlotThread(
                        id=tid, summary=goal[:220], introduced=number,
                        last_seen=number, heat=1.0)
                else:
                    thread.last_seen = number
                    thread.heat += 1.0
        # resolution evidence near resolve words
        for sent in _sentences(text):
            low = sent.lower()
            if _RESOLVE_WORDS & set(re.findall(r"[a-z]+", low)):
                for thread in found.values():
                    words = set(re.findall(r"[a-z]{4,}", thread.summary.lower()))
                    if len(words & set(re.findall(r"[a-z]+", low))) >= 2:
                        thread.status = "resolved"
                        thread.last_seen = number
    threads = sorted(found.values(), key=lambda t: t.heat, reverse=True)
    # re-id in heat order
    for i, t in enumerate(threads):
        t.id = f"t{i + 1}"
    return threads[:20]


_RULE_PATTERNS = (
    r"([^.!?]{10,140}\bcan(?:not|'t)\b[^.!?]{4,80}[.!?])",
    r"([^.!?]{10,140}\bcosts?\b[^.!?]{4,80}[.!?])",
    r"(\[[^[\]]{4,80}\][^.!?]{0,80}[.!?])",          # [System] notifications
    r"([^.!?]{10,120}\bLevel\s+\d+[^.!?]{0,60}[.!?])",
)


def _mine_rules(chapters: list[tuple[int, str, str]]) -> list[str]:
    rules: list[str] = []
    seen: set[str] = set()
    for _number, _title, text in chapters:
        for pat in _RULE_PATTERNS:
            for m in re.finditer(pat, text):
                rule = re.sub(r"\s+", " ", m.group(1)).strip()
                key = rule.lower()[:60]
                if key not in seen and len(rule) > 14:
                    seen.add(key)
                    rules.append(rule[:220])
        if len(rules) >= 24:
            break
    return rules


# ── digest fold: state, places, recaps, situation ─────────────────────────────
#
# Digestion must ADVANCE the bible's memory, not just accumulate names.
# These miners run on every digested batch (build and update alike) and
# keep four things current: each character's latest state, the places
# the story has been, a recap of the segment just digested, and the
# closing beat of the newest chapter.  Without them a bible built from
# chapter 1-6 sample describes the story's OPENING forever, and a
# continuation written from it re-stages the beginning.

_STATE_MARKERS = (
    "injur", "wound", "scar", "crush", "broken", "bleed", "dead",
    "died", "alive", "captur", "fled", "escap", "arriv", "reach",
    "carries", "carrying", "wield", "holding", "took the", "took a",
    "gained", "lost his", "lost her", "lost the", "became", "blind",
    "deaf", "poison", "curs", "exil", "banish", "missing", "now stood",
    "now sat", "now lay", "no longer", "swore", "vowed", "promot",
    "level", "rank",
)

#: markers strong enough to be timeline facts (Progressions)
_STRONG_STATE_MARKERS = (
    "injur", "wound", "scar", "crush", "broken", "dead", "died",
    "captur", "took the", "gained", "lost his", "lost her", "blind",
    "poison", "exil", "missing",
)


def _name_forms(name: str) -> list[str]:
    forms = [name]
    first = name.split()[0] if name.split() else ""
    if first and len(first) > 2 and first != name:
        forms.append(first)
    return forms


def _mentions(sentence: str, name: str) -> bool:
    return any(re.search(r"\b" + re.escape(f) + r"\b", sentence)
               for f in _name_forms(name))


def _mine_character_states(
        chapters: list[tuple[int, str, str]],
        names: list[str]) -> dict[str, tuple[str, int, bool]]:
    """Latest state-bearing sentence per character in this batch.

    Returns ``{name: (sentence, chapter, strong)}`` — only characters
    with a genuinely state-bearing sentence (injury, possession,
    whereabouts, status change) appear; a batch that says nothing
    about a character's state leaves their recorded state untouched.
    """
    out: dict[str, tuple[str, int, bool]] = {}
    for number, _title, text in chapters:
        for sent in _sentences(text):
            low = sent.lower()
            marker = next((m for m in _STATE_MARKERS if m in low), None)
            if marker is None:
                continue
            strong = any(m in low for m in _STRONG_STATE_MARKERS)
            for name in names:
                if _mentions(sent, name):
                    out[name] = (re.sub(r"\s+", " ", sent).strip()[:240],
                                 number, strong)
    return out


_LOCATION_PREPS = (
    "in", "at", "to", "into", "inside", "outside", "beneath", "across",
    "toward", "towards", "reached", "arrived at", "entered", "left",
)

# prepositions match lower- or sentence-initial Capitalized form; the
# PLACE itself must be Capitalized (case-sensitive classes on purpose)
_LOCATION_RE = re.compile(
    r"\b(?:" + "|".join(
        re.escape(v) for w in _LOCATION_PREPS
        for v in (w, w.capitalize())) + r")"
    r"\s+(?:the\s+|a\s+|an\s+)?"
    r"([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,}){0,2})")


def _mine_locations(
        chapters: list[tuple[int, str, str]],
        known_names: set[str]) -> list[BibleLocation]:
    """Capitalized places action happens in/at, latest context kept."""
    counts: Counter = Counter()
    latest: dict[str, tuple[str, int]] = {}
    chapters_seen: dict[str, set[int]] = {}
    for number, _title, text in chapters:
        for sent in _sentences(text):
            for m in _LOCATION_RE.finditer(sent):
                place = m.group(1).strip()
                words = place.split()
                if not place or place in known_names:
                    continue
                if any(w in _STOP_NAMES or w in _TITLE_WORDS
                       for w in words):
                    continue
                if any(place == n or place in _name_forms(n)
                       for n in known_names):
                    continue
                counts[place] += 1
                chapters_seen.setdefault(place, set()).add(number)
                latest[place] = (re.sub(r"\s+", " ", sent).strip()[:200],
                                 number)
    out = []
    for place, count in counts.most_common(24):
        if count < 2 and len(chapters_seen.get(place, set())) < 2:
            continue
        note, seen = latest[place]
        out.append(BibleLocation(name=place, note=note, last_seen=seen))
    return out[:12]


def _batch_recap(chapters: list[tuple[int, str, str]],
                 names: list[str]) -> str:
    """Extractive recap of one digested segment: the sentences that
    carry cast, goals, conflict and resolution, in story order."""
    scored: list[tuple[int, int, str]] = []
    seq = 0
    for _number, _title, text in chapters:
        for sent in _sentences(text):
            low = sent.lower()
            score = 0
            if any(_mentions(sent, n) for n in names):
                score += 2
            if _CONFLICT_WORDS & set(re.findall(r"[a-z]+", low)):
                score += 2
            if _RESOLVE_WORDS & set(re.findall(r"[a-z]+", low)):
                score += 1
            if any(p.search(sent) for p in _GOAL_RES):
                score += 2
            if 40 <= len(sent) <= 220:
                score += 1
            scored.append((seq, score, re.sub(r"\s+", " ", sent).strip()))
            seq += 1
    if not scored:
        return ""
    top = sorted(scored, key=lambda s: s[1], reverse=True)[:3]
    if all(s[1] == 0 for s in top):
        first = scored[0][2]
        last = scored[-1][2]
        return (first + " " + last)[:340].strip()
    top.sort(key=lambda s: s[0])
    return " ".join(s[2] for s in top)[:340].strip()


def _closing_situation(text: str, number: int) -> str:
    """Where a chapter STOPPED: its closing beat, verbatim-trimmed."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "")
             if p.strip() and not p.strip().startswith("#")]
    if not paras:
        return ""
    last = re.sub(r"\s+", " ", paras[-1]).strip()
    sents = [s.strip() for s in
             re.split(r"(?<=[.!?])\s+", last) if len(s.strip()) > 8]
    beat = " ".join(sents[-3:]) if sents else last
    return f"End of chapter {number}: {beat[:420]}"


def _profile_style(chapters: list[tuple[int, str, str]]) -> dict[str, Any]:
    full = "\n".join(t for _, _, t in chapters)
    words = re.findall(r"[A-Za-z']+", full.lower())
    past = sum(1 for w in words if w in _PAST_VERBS)
    present = sum(1 for w in words if w in _PRESENT_VERBS)
    tense = "past" if past >= present else "present"
    quoted = len(re.findall(r"\"[^\"]{4,}\"|“[^”]{4,}”", full))
    sents = _sentences(full)
    dialogue_ratio = quoted / max(1, len(sents))
    avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
    first_person = len(re.findall(r"\b(I|me|my|we|us|our)\b", full))
    third_person = len(re.findall(r"\b(he|she|they|him|her|them|his|theirs)\b",
                                  full, re.I))
    pov = "first" if first_person > third_person * 0.6 else "third"
    tone: list[str] = []
    if dialogue_ratio > 0.35:
        tone.append("dialogue-driven")
    if avg_len < 12:
        tone.append("punchy, short sentences")
    elif avg_len > 24:
        tone.append("long, flowing sentences")
    low = full.lower()
    if sum(low.count(w) for w in ("blood", "kill", "death", "dark")) > len(words) / 400:
        tone.append("dark")
    if sum(low.count(w) for w in ("laugh", "grin", "joke", "funny")) > len(words) / 500:
        tone.append("humorous")
    if "system" in low and low.count("system") > len(words) / 800:
        tone.append("system/litrpg texture")
    voice_notes = (
        f"{pov}-person, {tense} tense; "
        f"~{dialogue_ratio:.0%} dialogue; avg sentence {avg_len:.0f} words.")
    return {"pov": pov, "tense": tense, "tone": tone,
            "voice_notes": voice_notes}


# ── model path ───────────────────────────────────────────────────────────────


_BIBLE_SCHEMA_HINT = """{
  "title": "story title",
  "pov": "first|third|omniscient",
  "tense": "past|present",
  "tone": ["dark", "humorous"],
  "voice_notes": "how the prose sounds in 2 sentences",
  "characters": [{"name": "...", "role": "protagonist|antagonist|supporting",
                  "description": "...", "traits": ["..."],
                  "relationships": ["name: relation"]}],
  "threads": [{"summary": "...", "status": "open|resolved"}],
  "world_rules": ["..."],
  "arc_summary": "what has happened so far, 5-8 sentences"
}"""


def _model_bible(title: str, sample: str, context: Any) -> dict[str, Any] | None:
    try:
        from ..llm.base import Message, SamplingParams
        from ..llm.brain import brain_for
        from .write import model_available
    except Exception:  # noqa: BLE001
        return None
    if not model_available(context):
        return None
    try:
        brain = brain_for(context)
        data, response = brain.chat_json(
            [Message.system(
                "You are a story analyst building a series bible for a "
                "fiction continuation engine. Reply with ONLY the JSON "
                "object matching this schema:\n" + _BIBLE_SCHEMA_HINT),
             Message.user(
                 f"Story: {title}\n\nChapters sample:\n{sample[:12000]}\n\n"
                 "Build the bible. Track every named character, every open "
                 "plot thread, every world/system rule, and the voice.")],
            SamplingParams(temperature=0.2, max_tokens=4000),
            task_kind="extract")
        if data and isinstance(data, dict) and getattr(response, "ok", False):
            return data
    except Exception as exc:  # noqa: BLE001
        _log.debug("model bible build failed: %s", exc)
    return None


# ── builder ────────────────────────────────────────────────────────────────


class BibleBuilder:
    """Build and incrementally update story bibles."""

    def __init__(self, context: Any) -> None:
        self.context = context

    def _workspace(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = getattr(settings, "workspace_dir", None) if settings else None
        return Path(root) if root else Path.cwd() / "workspace"

    def bible_dir(self) -> Path:
        d = self._workspace() / "books" / "bibles"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bible_path(self, story_slug: str) -> Path:
        return self.bible_dir() / f"{slugify(story_slug)}.json"

    def load(self, story_slug: str) -> StoryBible | None:
        path = self.bible_path(story_slug)
        if not path.exists():
            return None
        try:
            return StoryBible.from_dict(
                json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            return None

    def save(self, bible: StoryBible) -> Path:
        path = self.bible_path(bible.story_slug)
        bible.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(bible.to_dict(), indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def build(self, story_slug: str, title: str,
              chapters: list[tuple[int, str, str]]) -> StoryBible:
        """chapters: (number, title, text) in reading order."""
        chapters = sorted(chapters, key=lambda c: c[0])
        sample = "\n\n".join(
            f"### Chapter {n}: {t}\n{txt[:4000]}"
            for n, t, txt in chapters[:6])
        data = _model_bible(title, sample, self.context)
        if data:
            bible = self._from_model(story_slug, title, data,
                                     len(chapters))
        else:
            bible = self._from_heuristics(story_slug, title, chapters)
        self._fold_digest(bible, chapters)
        bible.digestion_version = DIGESTION_VERSION
        bible.grounding_through = max(
            bible.grounding_through, max(n for n, _, _ in chapters))
        self._sync_lorebook(bible)
        self.save(bible)
        return bible

    def _fold_digest(self, bible: StoryBible,
                     chapters: list[tuple[int, str, str]]) -> None:
        """Advance the bible's memory with one digested batch.

        Runs on build AND on every incremental update, so the bible's
        picture of the story is always as of the LAST digested
        chapter — never frozen at the opening sample:

        * each character's CURRENT state (latest state-bearing
          sentence; strong changes also become Progressions — the
          timeline machinery finally wired into digestion);
        * locations, refreshed with their latest context;
        * an :class:`ArcRecap` for this segment;
        * ``current_situation`` — the closing beat of the newest
          chapter in the batch;
        * the voiceprint, re-measured from THIS batch, so the voice
          target tracks the story's current voice, not its first
          chapters forever.
        """
        if not chapters:
            return
        chapters = sorted(chapters, key=lambda c: c[0])
        names = [c.name for c in bible.characters if c.name]
        if names:
            states = _mine_character_states(chapters, names)
            by_name = {c.name: c for c in bible.characters}
            have_prog = {(p.name.lower(), p.detail[:60].lower())
                         for p in bible.progressions}
            for name, (sentence, number, strong) in states.items():
                char = by_name.get(name)
                if char is None:
                    continue
                if number >= char.state_chapter:
                    char.state = sentence
                    char.state_chapter = number
                if strong:
                    key = (name.lower(), sentence[:60].lower())
                    if key not in have_prog:
                        bible.add_progression(name, sentence,
                                              chapter=number)
                        have_prog.add(key)
        known = {c.name for c in bible.characters}
        by_place = {x.name.lower(): x for x in bible.locations}
        for loc in _mine_locations(chapters, known):
            existing = by_place.get(loc.name.lower())
            if existing is None:
                bible.locations.append(loc)
                by_place[loc.name.lower()] = loc
            elif loc.last_seen >= existing.last_seen:
                existing.note = loc.note
                existing.last_seen = loc.last_seen
        recap = _batch_recap(chapters, names)
        if recap:
            # a segment's recap REPLACES itself on a re-fold, so
            # refolding digested chapters (the legacy re-digest)
            # never duplicates the bible's memory of a segment
            seg = (chapters[0][0], chapters[-1][0])
            bible.recaps = [r for r in bible.recaps
                            if (r.start, r.end) != seg]
            bible.recaps.append(ArcRecap(
                start=seg[0], end=seg[1], text=recap))
            bible.recaps.sort(key=lambda r: (r.start, r.end))
        last_n, _t, last_text = chapters[-1]
        situation = _closing_situation(last_text, last_n)
        if situation:
            bible.current_situation = situation
        self._measure_voiceprint(bible, chapters)

    @staticmethod
    def _measure_voiceprint(bible: StoryBible,
                            chapters: list[tuple[int, str, str]]) -> None:
        """Measure the voiceprint of ``chapters`` into ``bible.voiceprint``.

        Failures leave it empty — a voiceprint failure must never break
        a bible build.
        """
        try:
            texts = [(t or "") for _, _, t in chapters if (t or "").strip()]
            if not texts:
                return
            names = [c.name for c in bible.characters if c.name]
            vp = fingerprint("\n\n".join(texts), names)
            if vp.words > 0:
                bible.voiceprint = vp.to_dict()
        except Exception as exc:  # noqa: BLE001 - voiceprint is advisory
            _log.debug("bible voiceprint measurement failed: %s", exc)

    def update(self, story_slug: str,
               new_chapters: list[tuple[int, str, str]]) -> StoryBible:
        """Digest more chapters into the existing bible (or build fresh)."""
        bible = self.load(story_slug)
        if bible is None:
            raise ValueError(f"no bible for {story_slug!r} — build one first")
        if not new_chapters:
            return bible
        # heuristic merge: mine the new chapters, fold into the bible
        chars = _mine_characters(new_chapters)
        by_name = {c.name.lower(): c for c in bible.characters}
        for c in chars:
            existing = by_name.get(c.name.lower())
            if existing:
                existing.mentions += c.mentions
                for rel in c.relationships:
                    if rel not in existing.relationships:
                        existing.relationships.append(rel)
                if len(existing.description) < 400 and c.description:
                    existing.description += " " + c.description[:200]
            else:
                bible.characters.append(c)
        threads = _mine_threads(new_chapters)
        by_summary = {t.summary[:60].lower(): t for t in bible.threads}
        for t in threads:
            key = t.summary[:60].lower()
            existing = by_summary.get(key)
            if existing:
                existing.last_seen = max(existing.last_seen, t.last_seen)
                existing.heat += t.heat
                if t.status == "resolved":
                    existing.status = "resolved"
            else:
                t.id = f"t{len(bible.threads) + 1}"
                bible.threads.append(t)
        for rule in _mine_rules(new_chapters):
            if rule not in bible.world_rules:
                bible.world_rules.append(rule)
        bible.chapters_digested = max(
            bible.chapters_digested,
            max(n for n, _, _ in new_chapters))
        # digestion advances the bible's memory (states, places,
        # recaps, current situation, current voice) — not just names
        self._fold_digest(bible, new_chapters)
        # grounding coverage advances only across a CONTIGUOUS fold,
        # and the version stamp asserts the fold covers everything
        # the bible claims digested — a tail-only fold over a legacy
        # bible must not certify chapters it never grounded (those
        # are the re-digest's job, via StoryContinuer.prepare)
        if min(n for n, _, _ in new_chapters) <= bible.grounding_through + 1:
            bible.grounding_through = max(
                bible.grounding_through,
                max(n for n, _, _ in new_chapters))
        if bible.grounding_through >= bible.chapters_digested:
            bible.digestion_version = DIGESTION_VERSION
        self._sync_lorebook(bible)
        self.save(bible)
        return bible

    def refold(self, story_slug: str,
               chapters: list[tuple[int, str, str]]) -> StoryBible:
        """Re-run ONLY the grounding fold over chapters a bible
        already claims digested — the legacy re-digest.

        Unlike :meth:`update` this mines nothing new (no double-
        counted mentions, no duplicated threads): the fold is
        idempotent over a segment — character states key on chapter,
        progressions dedupe on content, locations refresh by last
        sighting, a segment's recap replaces itself, and the current
        situation + voiceprint track the newest chapter folded.
        ``grounding_through`` advances and is saved per call, so an
        interrupted re-digest resumes where it stopped; stamping
        ``digestion_version`` is the caller's job, once the whole
        claimed range has been folded.
        """
        bible = self.load(story_slug)
        if bible is None:
            raise ValueError(
                f"no bible for {story_slug!r} — build one first")
        if not chapters:
            return bible
        self._fold_digest(bible, chapters)
        bible.grounding_through = max(
            bible.grounding_through, max(n for n, _, _ in chapters))
        self._sync_lorebook(bible)
        self.save(bible)
        return bible

    def _sync_lorebook(self, bible: StoryBible) -> None:
        """Derive/refresh keyed lorebook entries from the bible body.

        Characters become ``kind=character`` entries keyed by name +
        aliases (first/last name split); world rules become
        ``kind=rule`` always_on entries (capped — see consistency_check).
        """
        if not hasattr(bible, "lore") or bible.lore is None:
            bible.lore = []
        have = {e.name.lower() for e in bible.lore}
        order = 100 + len(bible.lore)
        for c in bible.characters:
            if c.name.lower() in have:
                continue
            aliases = []
            parts = c.name.split()
            if len(parts) > 1:
                aliases = [p for p in (parts[0], parts[-1]) if len(p) > 2]
            keys = [k for k in [c.name, *aliases]]
            text_bits = [c.description]
            if c.traits:
                text_bits.append("Traits: " + ", ".join(c.traits))
            if c.relationships:
                text_bits.append("Bonds: " + "; ".join(c.relationships[:4]))
            bible.lore.append(LorebookEntry(
                name=c.name, kind="character",
                text=" ".join(b for b in text_bits if b)[:600],
                keys=keys, aliases=aliases,
                always_on=(c.role == "protagonist"), order=order))
            order += 1
            have.add(c.name.lower())
        # rules: always_on but ONLY the first few (trimmed by budget)
        rule_entries = [e for e in bible.lore if e.kind == "rule"]
        for i, rule in enumerate(bible.world_rules[:6]):
            key = f"rule:{rule[:40]}"
            if key.lower() in {e.name.lower() for e in bible.lore}:
                continue
            bible.lore.append(LorebookEntry(
                name=key, kind="rule", text=rule[:300],
                keys=[w for w in rule.split()[:4] if len(w) > 4][:3],
                always_on=True, order=10 + i))

    # -- construction ---------------------------------------------------------
    def _from_model(self, story_slug: str, title: str,
                    data: dict[str, Any], n_chapters: int) -> StoryBible:
        characters = []
        for c in data.get("characters", [])[:30]:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            role = str(c.get("role", "supporting")).lower()
            if role not in ("protagonist", "antagonist", "supporting"):
                role = "supporting"
            characters.append(BibleCharacter(
                name=str(c["name"])[:80], role=role,
                description=str(c.get("description", ""))[:500],
                traits=[str(t)[:60] for t in c.get("traits", [])][:8],
                relationships=[str(r)[:120]
                               for r in c.get("relationships", [])][:8]))
        threads = []
        for i, t in enumerate(data.get("threads", [])[:20]):
            if not isinstance(t, dict) or not t.get("summary"):
                continue
            status = str(t.get("status", "open")).lower()
            threads.append(PlotThread(
                id=f"t{i + 1}", summary=str(t["summary"])[:300],
                status="resolved" if status == "resolved" else "open",
                heat=5.0))
        style = _profile_style([])  # model supplies tone directly
        return StoryBible(
            story_slug=story_slug, title=title,
            pov=str(data.get("pov", "third")).lower(),
            tense=str(data.get("tense", "past")).lower(),
            tone=[str(x)[:60] for x in data.get("tone", [])][:8],
            voice_notes=str(data.get("voice_notes", ""))[:600],
            characters=characters, threads=threads,
            world_rules=[str(r)[:220] for r in
                         data.get("world_rules", [])][:24],
            arc_summary=str(data.get("arc_summary", ""))[:1500],
            chapters_digested=n_chapters)

    def _from_heuristics(self, story_slug: str, title: str,
                         chapters: list[tuple[int, str, str]]) -> StoryBible:
        characters = _mine_characters(chapters)
        threads = _mine_threads(chapters)
        rules = _mine_rules(chapters)
        style = _profile_style(chapters)
        # arc summary: first-sentence sketch of each chapter's thrust
        bits = []
        for n, t, text in chapters[:8]:
            sents = _sentences(text)
            bits.append(f"Ch{n} ({t}): "
                        f"{sents[0][:140] if sents else ''}")
        return StoryBible(
            story_slug=story_slug, title=title, pov=style["pov"],
            tense=style["tense"], tone=style["tone"],
            voice_notes=style["voice_notes"], characters=characters,
            threads=threads, world_rules=rules,
            arc_summary=" ".join(bits)[:1500],
            chapters_digested=max(n for n, _, _ in chapters))


def build_bible(story_slug: str, title: str,
                chapters: list[tuple[int, str, str]],
                context: Any = None) -> StoryBible:
    """Build a bible without a builder instance (tests, one-shots)."""
    return BibleBuilder(context).build(story_slug, title, chapters)
