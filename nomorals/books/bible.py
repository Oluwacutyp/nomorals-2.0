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

_log = get_logger(__name__)

__all__ = [
    "StoryBible", "BibleCharacter", "PlotThread",
    "BibleBuilder", "build_bible",
]


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    built_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoryBible":
        chars = [BibleCharacter(**c) for c in d.get("characters", [])]
        threads = [PlotThread(**t) for t in d.get("threads", [])]
        return cls(
            story_slug=d.get("story_slug", ""), title=d.get("title", ""),
            pov=d.get("pov", "third"), tense=d.get("tense", "past"),
            tone=list(d.get("tone", [])), voice_notes=d.get("voice_notes", ""),
            characters=chars, threads=threads,
            world_rules=list(d.get("world_rules", [])),
            arc_summary=d.get("arc_summary", ""),
            chapters_digested=int(d.get("chapters_digested", 0)),
            built_at=d.get("built_at", ""))

    def character(self, name: str) -> BibleCharacter | None:
        low = name.lower()
        for c in self.characters:
            if c.name.lower() == low:
                return c
        return None

    def open_threads(self) -> list[PlotThread]:
        return [t for t in self.threads if t.status == "open"]

    def brief(self) -> str:
        """Compact steering text for generation prompts."""
        lines = [f"STORY: {self.title} (POV: {self.pov}, tense: {self.tense})"]
        if self.tone:
            lines.append("Tone: " + ", ".join(self.tone))
        if self.voice_notes:
            lines.append("Voice: " + self.voice_notes)
        if self.world_rules:
            lines.append("World rules:")
            lines += [f"  - {r}" for r in self.world_rules[:12]]
        if self.characters:
            lines.append("Cast:")
            for c in self.characters[:12]:
                rel = f" ({'; '.join(c.relationships[:3])})" if c.relationships else ""
                lines.append(f"  - {c.name} [{c.role}]{rel}: "
                             f"{c.description[:160]}")
        open_t = self.open_threads()
        if open_t:
            lines.append("Open plot threads:")
            lines += [f"  - {t.summary[:160]}" for t in open_t[:10]]
        if self.arc_summary:
            lines.append("So far: " + self.arc_summary[:800])
        return "\n".join(lines)


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
        self.save(bible)
        return bible

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
        self.save(bible)
        return bible

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
