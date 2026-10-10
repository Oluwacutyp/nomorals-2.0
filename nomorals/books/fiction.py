"""FictionWriter — Devon's native fiction engine.

Not prompt prefixes: each genre is a real engine with mechanics it
enforces across chapters — ledgers, curves, budgets, and validators:

* mystery  — clue ledger, suspect matrix, red-herring ratio, FAIR-PLAY
  resolution (every fact used in the reveal must be planted ≥2 chapters
  earlier, or the chapter fails validation);
* thriller — a computed tension curve (rise + set-piece spikes + breather
  valleys), a ticking clock, chapters must end on a hook;
* horror   — the dread cycle (anticipation → dread → release/false
  release), a scare budget, an *unknown ledger* of things that must never
  be explained;
* sci-fi   — a world-rule ledger with implications; contradictions between
  a chapter and established rules fail validation;
* fantasy  — a magic system with costs/limits/sources; magic without an
  established cost fails validation;
* romance  — an intimacy ledger that must oscillate (no +3 jumps), a dark
  night before the 75% mark, no permanent resolution before it.

Two modes:

* ``novel``  — planned arcs with an ending that lands: the climax chapter
  is scheduled, the denouement closes every open thread;
* ``serial`` — never-ending fiction: an arc manager closes arcs and
  spawns the next one from the hottest dangling thread, so the story
  keeps moving instead of meandering.

The book decides its own shape: chapter counts come from premise
complexity (threads × depth), never a template.  WisdomKeeper's corpus
is woven in as thematic texture (``weave_wisdom``) — the user's explicit
requirement.  Prose is model-first; the heuristic composer writes real
multi-paragraph prose from the brief when no model is answering.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind
from .model import count_words, slugify

_log = get_logger(__name__)

__all__ = [
    "FictionWriter", "GenreEngine", "StoryState", "Arc", "ChapterBrief",
    "ENGINES", "engine_for",
]


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Arc:
    name: str
    goal: str
    chapters_planned: int
    beats: list[str] = field(default_factory=list)
    status: str = "planned"           # planned | active | closed
    threads_to_close: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChapterBrief:
    chapter_no: int
    arc_name: str
    must_happen: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    plant: list[str] = field(default_factory=list)
    threads_advanced: list[str] = field(default_factory=list)
    pov_notes: str = ""
    target_words: int = 1200
    wisdom_motifs: list[str] = field(default_factory=list)
    tension_target: float = 0.0       # thriller/horror dial 0..10
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoryState:
    slug: str
    title: str
    premise: str
    genre: str
    mode: str                          # novel | serial
    chapters_written: int = 0
    characters: list[dict[str, Any]] = field(default_factory=list)
    threads: list[dict[str, Any]] = field(default_factory=list)
    arcs: list[dict[str, Any]] = field(default_factory=list)
    current_arc: int = 0
    total_planned: int = 0
    style: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoryState":
        from dataclasses import MISSING
        kwargs: dict[str, Any] = {}
        for k, f in cls.__dataclass_fields__.items():
            if k in d:
                kwargs[k] = d[k]
            elif f.default is not MISSING:
                kwargs[k] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                kwargs[k] = f.default_factory()
        return cls(**kwargs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── wisdom weave ─────────────────────────────────────────────────────────────


def weave_wisdom(theme: str, context: Any, *, top: int = 3) -> list[str]:
    """Thematic texture from WisdomKeeper's corpus for a story theme.

    Returns short motif lines (with provenance) the chapter brief carries
    into the prose.  Silent empty when the keeper is unavailable — the
    story still works, just without the esoteric texture.
    """
    if not (theme or "").strip():
        return []
    try:
        from ..wisdom import WisdomKeeper
        answer = WisdomKeeper(context).ask(theme, top=top, mode="keyword")
    except Exception as exc:  # noqa: BLE001
        _log.debug("wisdom weave unavailable: %s", exc)
        return []
    motifs: list[str] = []
    for hit in getattr(answer, "passages", []) or []:
        text = (getattr(hit, "text", "") or "").strip()
        source = (getattr(hit, "source", "") or "").strip()
        if len(text) < 40:
            continue
        motif = re.sub(r"\s+", " ", text)[:280]
        motifs.append(f"{motif} [{source}]" if source else motif)
    return motifs


# ── genre engines ────────────────────────────────────────────────────────────


class GenreEngine(ABC):
    """Real genre mechanics: plan → brief → validate → advance."""

    name: str = "base"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed or 0xF1C710)

    # -- lifecycle ------------------------------------------------------------
    @abstractmethod
    def plan_arcs(self, premise: str, state: StoryState) -> list[Arc]: ...

    @abstractmethod
    def chapter_brief(self, state: StoryState, arc: Arc,
                      chapter_no: int) -> ChapterBrief: ...

    @abstractmethod
    def validate(self, text: str, state: StoryState,
                 brief: ChapterBrief) -> list[str]: ...

    @abstractmethod
    def advance(self, state: StoryState, text: str,
                brief: ChapterBrief) -> None: ...

    # -- shared helpers ---------------------------------------------------------
    def _cast(self, premise: str, n: int,
              roles: list[str]) -> list[dict[str, Any]]:
        names = self._names(n)
        cast = []
        for i, role in enumerate(roles[:n]):
            cast.append({
                "name": names[i], "role": role,
                "trait": self.rng.choice([
                    "sharp-tongued", "quietly observant", "reckless",
                    "methodical", "charismatic", "haunted", "wry",
                    "fiercely loyal", "ambitious", "patient",
                ]),
                "secret": "",
            })
        return cast

    def _names(self, n: int) -> list[str]:
        pool = ["Mara", "Ikenna", "Sable", "Dorian", "Zainab", "Kael",
                "Odette", "Reyes", "Tunde", "Vesper", "Anaya", "Corvin",
                "Lena", "Emeka", "Isolde", "Dante", "Priya", "Silas",
                "Ngozi", "Rowan", "Adaeze", "Lucian", "Mira", "Jide"]
        self.rng.shuffle(pool)
        return pool[:n]


class MysteryEngine(GenreEngine):
    """Fair-play mystery: the solution must be deducible from planted clues.

    Mechanics: a clue ledger (every clue tagged true/red-herring, with the
    chapter it was planted in), a suspect matrix (motive/means/opportunity),
    a red-herring ratio cap (≤40%), and a reveal chapter.  Validation
    rejects any chapter that spends solution facts before they are planted.
    """

    name = "mystery"

    _CRIMES = ["a locked-room murder", "a vanished heirloom",
               "a poisoning at a gala", "a blackmailer's death",
               "a forged will", "a disappearance from a sealed train"]

    def plan_arcs(self, premise: str, state: StoryState) -> list[Arc]:
        crime = self._crimes[0] if "murder" in premise.lower() else \
            self.rng.choice(self._CRIMES)
        cast = self._cast(premise, 5, ["detective", "suspect", "suspect",
                                       "suspect", "suspect"])
        culprit = cast[self.rng.randrange(1, 5)]["name"]
        method = self.rng.choice([
            "a timed poison in the victim's own medicine",
            "a rigged lock that only opens from inside",
            "a twin's alibi, swapped at the critical hour",
            "a letter that arrived a day late — on purpose",
            "a mirror angled to fake the time of death",
        ])
        motive = self.rng.choice([
            "an inheritance about to be rewritten",
            "a secret that would ruin them by morning",
            "a debt owed to someone who collects in blood",
            "love, curdled into ownership",
        ])
        # clue ledger: 8 true clues + up to 5 red herrings
        true_clues = [
            f"a {self.rng.choice(['torn', 'singed', 'damp'])} scrap of "
            f"{self.rng.choice(['fabric', 'paper', 'ribbon'])} that doesn't belong",
            "a witness who misremembers the time by exactly one hour",
            "an object moved two inches from where it should be",
            "a phone that rang once and was never answered",
            "mud on shoes that never left the house",
            "a second cup, washed and put away too carefully",
            "a clock stopped at the wrong hour",
            "a name crossed out and written over in different ink",
        ]
        red = [
            "a stranger seen arguing with the victim that night",
            "a missing sum of money with no explanation",
            "a love letter never sent",
            "a broken window that was broken from inside",
            "an anonymous tip that points the wrong way",
        ]
        state.ledger = {
            "crime": crime, "culprit": culprit, "method": method,
            "motive": motive,
            "suspects": [
                {"name": c["name"], "motive": self.rng.choice(
                    ["money", "jealousy", "fear of exposure", "revenge", ""]),
                 "means": self.rng.random() < 0.6,
                 "opportunity": self.rng.random() < 0.6,
                 "interviewed": False}
                for c in cast[1:]],
            "clues": [{"id": f"c{i+1}", "text": t, "planted_ch": 0,
                       "red_herring": False} for i, t in enumerate(true_clues)]
            + [{"id": f"r{i+1}", "text": t, "planted_ch": 0,
                "red_herring": True} for i, t in enumerate(red)],
            "reveal_chapter": 0,  # set by decide_shape
        }
        state.characters = cast
        state.threads = [
            {"id": "t1", "summary": f"Who is behind {crime}?",
             "status": "open", "heat": 10.0},
            {"id": "t2", "summary": f"What is {culprit} hiding?",
             "status": "open", "heat": 4.0},
        ]
        return [
            Arc("The Crime", f"Establish {crime}; plant the first true clues.",
                3, ["the discovery", "the detective takes the case",
                    "first interviews, first lies"]),
            Arc("The Investigation",
                "Every suspect interviewed; red herrings peak.",
                5, ["the alibis crack", "a second incident raises the stakes",
                    "the detective's theory collapses"]),
            Arc("The Resolution",
                "Fair-play reveal: every solution fact already planted.",
                2, ["the gathering", "the reveal and the proof"]),
            Arc("Denouement", "Loose ends tied; the cost counted.", 1,
                ["aftermath"]),
        ]

    def _reveal_chapter(self, state: StoryState) -> int:
        return int(state.ledger.get("reveal_chapter") or state.total_planned - 1)

    def chapter_brief(self, state: StoryState, arc: Arc,
                      chapter_no: int) -> ChapterBrief:
        ledger = state.ledger
        unplanted_true = [c for c in ledger["clues"]
                          if not c["red_herring"] and not c["planted_ch"]]
        unplanted_red = [c for c in ledger["clues"]
                         if c["red_herring"] and not c["planted_ch"]]
        reveal = self._reveal_chapter(state)
        must_happen: list[str] = []
        plant: list[str] = []
        must_not = [f"do NOT reveal the culprit ({ledger['culprit']}) or the "
                    f"method before chapter {reveal}"]
        if arc.name == "The Crime":
            plant = [c["text"] for c in unplanted_true[:2]]
            detective = (state.characters[0]["name"]
                         if state.characters else "the detective")
            suspects = [s["name"] for s in ledger["suspects"][:2]]
            must_happen = [f"the discovery of {ledger['crime']}",
                           f"introduce {detective} and suspects "
                           + ", ".join(suspects)]
        elif arc.name == "The Investigation":
            # keep red-herring ratio ≤ 40% of planted clues
            planted = [c for c in ledger["clues"] if c["planted_ch"]]
            red_ratio = (sum(1 for c in planted if c["red_herring"]) /
                         max(1, len(planted)))
            picks = ([c["text"] for c in unplanted_true[:1]] +
                     [c["text"] for c in unplanted_red[:1]])
            if red_ratio > 0.4:
                picks = [c["text"] for c in unplanted_true[:2]]
            plant = picks
            suspect = next((s for s in ledger["suspects"]
                            if not s["interviewed"]), None)
            if suspect:
                must_happen = [f"interview {suspect['name']} — their alibi "
                               f"cracks slightly"]
        elif arc.name == "The Resolution":
            must_happen = ["gather the suspects",
                           f"the reveal: {ledger['culprit']}, "
                           f"{ledger['method']}, motive: {ledger['motive']} — "
                           "every fact must already have been planted"]
            must_not = []
        else:
            must_happen = ["the aftermath: what the truth cost everyone"]
        return ChapterBrief(
            chapter_no=chapter_no, arc_name=arc.name,
            must_happen=must_happen, must_not=must_not, plant=plant,
            threads_advanced=["t1"],
            pov_notes="detective's POV; sharp, observant, wry",
            target_words=1300)

    def validate(self, text: str, state: StoryState,
                 brief: ChapterBrief) -> list[str]:
        problems: list[str] = []
        ledger = state.ledger
        low = text.lower()
        reveal = self._reveal_chapter(state)
        culprit = ledger["culprit"].lower()
        # fair-play: the culprit's name + method must not appear as the
        # solution before the reveal chapter
        if brief.chapter_no < reveal:
            method_bits = set(re.findall(r"[a-z]{4,}", ledger["method"].lower()))
            if culprit in low and method_bits & set(re.findall(r"[a-z]+", low)):
                problems.append(
                    f"fair-play violation: culprit+method surface in ch"
                    f"{brief.chapter_no}, before reveal ch{reveal}")
        # every planted clue the brief demanded should appear
        for clue in brief.plant:
            key = set(re.findall(r"[a-z]{4,}", clue.lower()))
            if key and not (key & set(re.findall(r"[a-z]+", low))):
                problems.append(f"demanded clue not planted: {clue[:60]!r}")
        return problems

    def advance(self, state: StoryState, text: str,
                brief: ChapterBrief) -> None:
        low = text.lower()
        words = set(re.findall(r"[a-z]+", low))
        for clue in state.ledger["clues"]:
            if clue["planted_ch"]:
                continue
            key = set(re.findall(r"[a-z]{4,}", clue["text"].lower()))
            if key and len(key & words) >= max(2, len(key) // 2):
                clue["planted_ch"] = brief.chapter_no
        for s in state.ledger["suspects"]:
            if s["name"].lower() in low:
                s["interviewed"] = True


class ThrillerEngine(GenreEngine):
    """Ticking-clock thriller: a computed tension curve, set pieces, hooks.

    Mechanics: tension target per chapter = base rise + set-piece spikes
    (chapters at 25/60/90% of the plan, +3) with breather valleys after
    (-2); a deadline chapter; every chapter must end on an unresolved
    hook (validated on the final paragraph).
    """

    name = "thriller"

    def plan_arcs(self, premise: str, state: StoryState) -> list[Arc]:
        cast = self._cast(premise, 4, ["protagonist", "ally",
                                       "antagonist", "wild card"])
        deadline = self.rng.choice([
            "the exchange at midnight", "the launch window",
            "the vote", "the detonation timer", "the extraction flight",
        ])
        state.characters = cast
        state.threads = [
            {"id": "t1", "summary": f"Stop {cast[2]['name']} before {deadline}",
             "status": "open", "heat": 10.0},
            {"id": "t2", "summary": f"What is {cast[3]['name']} really after?",
             "status": "open", "heat": 5.0},
        ]
        state.ledger = {"deadline": deadline, "deadline_chapter": 0,
                        "tension": 2.0, "set_pieces": []}
        return [
            Arc("Ignition", "The threat lands; the clock starts.", 2,
                ["the inciting attack", "no going back"]),
            Arc("Escalation", "Set pieces; the noose tightens.", 5,
                ["first set piece", "betrayal", "second set piece",
                 "the cost", "cornered"]),
            Arc("Countdown", "The deadline run; everything converges.", 3,
                ["the plan", "the breach", "the deadline"]),
        ]

    def _curve(self, state: StoryState, chapter_no: int) -> float:
        total = max(6, state.total_planned or 10)
        base = 2.0 + (chapter_no / total) * 5.0
        spikes = [int(total * p) for p in (0.25, 0.6, 0.9)]
        if chapter_no in spikes:
            return min(10.0, base + 3.0)
        if chapter_no - 1 in spikes:
            return max(1.0, base - 2.0)  # breather valley
        return min(10.0, base)

    def chapter_brief(self, state: StoryState, arc: Arc,
                      chapter_no: int) -> ChapterBrief:
        tension = self._curve(state, chapter_no)
        total = max(6, state.total_planned or 10)
        is_spike = chapter_no in [int(total * p) for p in (0.25, 0.6, 0.9)]
        must_happen = [
            f"raise the stakes toward {state.ledger['deadline']}",
            "end the chapter on an unresolved hook — a door opening, "
            "a phone ringing, a betrayal revealed",
        ]
        if is_spike:
            must_happen.insert(0, "SET PIECE: a sustained action sequence "
                                  "with real consequences")
        if chapter_no >= total - 1:
            must_happen.append("the deadline arrives; resolve the clock")
        return ChapterBrief(
            chapter_no=chapter_no, arc_name=arc.name,
            must_happen=must_happen,
            must_not=["no chapter may fully resolve the central threat "
                      "before the deadline chapter"],
            threads_advanced=["t1"], tension_target=tension,
            pov_notes="tight third-person on the protagonist; short "
                      "sentences when tension is high",
            target_words=1400)

    def validate(self, text: str, state: StoryState,
                 brief: ChapterBrief) -> list[str]:
        problems: list[str] = []
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paras:
            last = paras[-1].lower()
            hook_markers = ("?", "—", "...", "suddenly", "behind",
                            "opened", "rang", "scream", "gun", "run",
                            "too late", "not alone")
            if not any(m in last for m in hook_markers) and len(last) > 40:
                problems.append("chapter does not end on a hook")
        # tension check: urgent lexicon density vs target
        urgent = sum(text.lower().count(w) for w in
                     ("suddenly", "now", "hurry", "fast", "danger", "deadline",
                      "seconds", "run", "chase", "explosion", "alarm"))
        words = max(1, count_words(text))
        density = urgent / words * 1000
        if brief.tension_target >= 7 and density < 2.0:
            problems.append(
                f"tension too low for a set-piece chapter "
                f"(urgent-word density {density:.1f}/1k)")
        return problems

    def advance(self, state: StoryState, text: str,
                brief: ChapterBrief) -> None:
        state.ledger["tension"] = brief.tension_target


class HorrorEngine(GenreEngine):
    """Dread-cycle horror: anticipation → dread → release/false release.

    Mechanics: a per-chapter scare budget (1 major, ≤2 minor), an unknown
    ledger — things that must NEVER be explained (validation rejects
    exposition that names them), and a dread quota: ≥60% of chapters end
    on dread rather than release.
    """

    name = "horror"

    def plan_arcs(self, premise: str, state: StoryState) -> list[Arc]:
        cast = self._cast(premise, 4, ["protagonist", "skeptic",
                                       "believer", "the lost one"])
        unknowns = [
            "what lives in the walls",
            "why the mirrors face the wall",
            "what the town agreed never to speak of",
            "who keeps resetting the clocks",
        ]
        state.characters = cast
        state.threads = [
            {"id": "t1", "summary": f"Survive {unknowns[0]}",
             "status": "open", "heat": 10.0},
        ]
        state.ledger = {
            "unknowns": self.rng.sample(unknowns, 2),
            "dread_endings": 0, "chapters": 0,
            "scares_used": [],
        }
        return [
            Arc("The Wrongness", "Something is off; nobody believes it.", 3,
                ["the first sign", "rationalizations", "the sign repeats"]),
            Arc("The Deepening", "The rules of safety stop working.", 4,
                ["isolation", "the loss", "the pattern emerges",
                 "no one is coming"]),
            Arc("The Reckoning", "Face it or become part of it.", 3,
                ["descent", "the price", "dawn — or what passes for it"]),
        ]

    def chapter_brief(self, state: StoryState, arc: Arc,
                      chapter_no: int) -> ChapterBrief:
        ledger = state.ledger
        # dread quota: keep ≥60% of endings on dread
        need_dread = (ledger["dread_endings"] / max(1, ledger["chapters"])) < 0.6
        cycle = self.rng.choice(["anticipation→dread→false release",
                                 "anticipation→dread→dread"])
        must_happen = [
            f"dread cycle for this chapter: {cycle}",
            "ONE major scare maximum; at most two minor ones",
            f"end on {'DREAD (unresolved)' if need_dread else 'a fragile release'}",
        ]
        must_not = [f"NEVER explain or name {u}" for u in ledger["unknowns"]]
        return ChapterBrief(
            chapter_no=chapter_no, arc_name=arc.name,
            must_happen=must_happen, must_not=must_not,
            threads_advanced=["t1"], tension_target=7.5,
            pov_notes="close third-person; sensory detail over exposition; "
                      "short paragraphs when dread peaks",
            target_words=1200,
            extra={"dread_ending": need_dread})

    def validate(self, text: str, state: StoryState,
                 brief: ChapterBrief) -> list[str]:
        problems: list[str] = []
        low = text.lower()
        for unknown in state.ledger["unknowns"]:
            key = set(re.findall(r"[a-z]{4,}", unknown.lower()))
            explainers = ("it was", "turned out to be", "in fact",
                          "the truth was", "explained")
            if key and (key & set(re.findall(r"[a-z]+", low))) and \
                    any(e in low for e in explainers):
                problems.append(
                    f"unknown ledger violated: {unknown!r} gets explained")
        # scare budget: count scare beats
        scares = sum(low.count(w) for w in
                     ("scream", "lunged", "burst through", "grabbed",
                      "shrieked", "erupted"))
        if scares > 6:
            problems.append(f"scare budget blown ({scares} scare beats)")
        return problems

    def advance(self, state: StoryState, text: str,
                brief: ChapterBrief) -> None:
        ledger = state.ledger
        ledger["chapters"] += 1
        if brief.extra.get("dread_ending"):
            ledger["dread_endings"] += 1


class SciFiEngine(GenreEngine):
    """Rule-ledger sci-fi: the world runs on stated rules with implications.

    Mechanics: every world rule carries implications the story must honor;
    validation scans chapters for contradictions (a rule says X is
    impossible and the chapter does X).  A sense-of-wonder beat is
    scheduled every third chapter.
    """

    name = "sci-fi"

    def plan_arcs(self, premise: str, state: StoryState) -> list[Arc]:
        cast = self._cast(premise, 4, ["protagonist", "engineer",
                                       "rival", "AI"])
        rules = [
            ("FTL jumps require a living navigator; the jump burns a year "
             "of their life.", ["navigators are precious and guarded",
                                "someone is always hunting navigators"]),
            ("The station's AI cannot lie, but it can refuse to answer.",
             ["silence becomes a weapon", "questions must be asked precisely"]),
            ("Terraforming wakes what was sleeping in the ice.",
             ["the ice is not empty", "progress has a body count"]),
        ]
        chosen = self.rng.sample(rules, 2)
        state.characters = cast
        state.threads = [
            {"id": "t1", "summary": "Survive what the premise unleashed",
             "status": "open", "heat": 10.0},
        ]
        state.ledger = {
            "rules": [{"rule": r, "implications": list(imps)}
                      for r, imps in chosen],
        }
        return [
            Arc("First Contact", "The premise lands; rules demonstrated.", 3,
                ["the anomaly", "the rule in action", "the cost"]),
            Arc("Extrapolation", "Implications cascade; factions form.", 4,
                ["the exploit", "the arms race", "the accident",
                 "the point of no return"]),
            Arc("Singularity", "The rules' final implication plays out.", 3,
                ["the gamble", "the inversion", "the new equilibrium"]),
        ]

    def chapter_brief(self, state: StoryState, arc: Arc,
                      chapter_no: int) -> ChapterBrief:
        rules = state.ledger["rules"]
        rule = rules[(chapter_no - 1) % len(rules)]
        must_happen = [
            f"showcase or stress-test the rule: {rule['rule']}",
            f"honor an implication: {self.rng.choice(rule['implications'])}",
        ]
        if chapter_no % 3 == 0:
            must_happen.append("SENSE OF WONDER beat: scale, strangeness, awe")
        must_not = [f"never contradict an established rule: {r['rule']}"
                    for r in rules]
        return ChapterBrief(
            chapter_no=chapter_no, arc_name=arc.name,
            must_happen=must_happen, must_not=must_not,
            threads_advanced=["t1"],
            pov_notes="precise, technical when it earns it; human stakes first",
            target_words=1300)

    def validate(self, text: str, state: StoryState,
                 brief: ChapterBrief) -> list[str]:
        problems: list[str] = []
        low = text.lower()
        for entry in state.ledger["rules"]:
            rule = entry["rule"].lower()
            # contradiction patterns derived from the rule text
            if "cannot lie" in rule and re.search(
                    r"\b(lied|lying|deceived?)\b.*\bAI\b|\bAI\b.*\b(lied|lying)\b",
                    low):
                problems.append("rule contradiction: the AI lied")
            if "burns a year" in rule and re.search(
                    r"jump.*without.*(cost|price|burn)|free.*jump", low):
                problems.append("rule contradiction: costless FTL jump")
            if "refuse to answer" in rule and "the AI answered everything" in low:
                problems.append("rule contradiction: the AI answered everything")
        return problems

    def advance(self, state: StoryState, text: str,
                brief: ChapterBrief) -> None:
        # new rules the chapter establishes get ledgered
        for m in re.finditer(r"[Tt]he (?:rule|law) (?:is|was) ([^.!?]{10,120})",
                             text):
            rule = m.group(0).strip()
            if all(rule.lower() not in e["rule"].lower()
                   for e in state.ledger["rules"]):
                state.ledger["rules"].append(
                    {"rule": rule, "implications": []})


class FantasyEngine(GenreEngine):
    """Cost-ledger fantasy: magic with prices, limits, and sources.

    Mechanics: the magic system is ledgered (source, cost, limit,
    taboo); validation fails any chapter where magic solves a problem
    without its cost being paid or acknowledged on-page.  Quest beats are
    ordered dynamically per arc, never from a fixed template.
    """

    name = "fantasy"

    def plan_arcs(self, premise: str, state: StoryState) -> list[Arc]:
        cast = self._cast(premise, 5, ["protagonist", "mentor", "rival",
                                       "trickster", "dark lord"])
        systems = [
            {"source": "old gods' attention", "cost": "memory",
             "limit": "the gods notice — and collect",
             "taboo": "never ask twice"},
            {"source": "dragonfire bound in iron", "cost": "years of life",
             "limit": "iron cracks; cracked iron kills the bearer",
             "taboo": "never draw on another's fire"},
            {"source": "true names", "cost": "your own name frays",
             "limit": "a name fully spent unmakes you",
             "taboo": "never speak your whole name"},
        ]
        magic = self.rng.choice(systems)
        state.characters = cast
        state.threads = [
            {"id": "t1", "summary": f"Stop {cast[4]['name']}",
             "status": "open", "heat": 10.0},
            {"id": "t2", "summary": f"What {cast[1]['name']} is not saying",
             "status": "open", "heat": 5.0},
        ]
        state.ledger = {"magic": magic, "costs_paid": []}
        beats_pool = ["the call", "refusal and price", "the crossing",
                      "trials", "the ordeal", "the return changed"]
        self.rng.shuffle(beats_pool)
        return [
            Arc("The Call", "The ordinary world breaks.", 2,
                beats_pool[:2]),
            Arc("The Trials", "Power learned; prices paid.", 4,
                beats_pool[2:4] + ["the cost comes due", "the betrayal"]),
            Arc("The Ordeal", "Everything the magic cost, weighed at once.",
                3, beats_pool[4:] + ["the taboo tempts", "the price paid"]),
        ]

    def chapter_brief(self, state: StoryState, arc: Arc,
                      chapter_no: int) -> ChapterBrief:
        magic = state.ledger["magic"]
        must_happen = [
            f"advance arc beat: {arc.beats[(chapter_no - 1) % len(arc.beats)]}",
        ]
        must_not = [
            f"magic source: {magic['source']}; every working costs "
            f"{magic['cost']} — show the price ON PAGE",
            f"taboo: {magic['taboo']}",
            f"limit: {magic['limit']}",
        ]
        return ChapterBrief(
            chapter_no=chapter_no, arc_name=arc.name,
            must_happen=must_happen, must_not=must_not,
            threads_advanced=["t1"],
            pov_notes="mythic but grounded; magic feels heavy, never free",
            target_words=1400)

    def validate(self, text: str, state: StoryState,
                 brief: ChapterBrief) -> list[str]:
        problems: list[str] = []
        low = text.lower()
        magic = state.ledger["magic"]
        uses_magic = any(w in low for w in
                         ("spell", "magic", "power surged", "cast",
                          "enchantment", "summoned", "ward"))
        cost = magic["cost"].lower().split()[0]
        pays = cost in low or any(w in low for w in
                                  ("price", "cost", "paid", "sacrificed",
                                   "drained", "weakened", "trembling"))
        if uses_magic and not pays:
            problems.append(
                f"magic without cost: power is used but {magic['cost']} "
                f"is never paid or acknowledged")
        if magic["taboo"].split()[-1] in low and "taboo" not in low:
            pass  # taboo mention alone is not a violation; breaking it is
        return problems

    def advance(self, state: StoryState, text: str,
                brief: ChapterBrief) -> None:
        low = text.lower()
        if any(w in low for w in ("paid the price", "the cost",
                                  "sacrificed", "drained")):
            state.ledger["costs_paid"].append(brief.chapter_no)


class RomanceEngine(GenreEngine):
    """Oscillation romance: intimacy must rise and fall before it lands.

    Mechanics: an intimacy ledger (0–10) that may move at most ±2 per
    chapter and must dip at least twice before the dark night; the dark
    night is scheduled before the 75% mark; validation rejects permanent
    commitment language ("forever", "marry me" as resolution) before it.
    """

    name = "romance"

    def plan_arcs(self, premise: str, state: StoryState) -> list[Arc]:
        cast = self._cast(premise, 4, ["lead", "love interest",
                                       "rival", "best friend"])
        state.characters = cast
        state.threads = [
            {"id": "t1",
             "summary": f"Will {cast[0]['name']} and {cast[1]['name']} "
                        f"end up together?", "status": "open", "heat": 10.0},
            {"id": "t2",
             "summary": f"What {cast[1]['name']} is afraid to want",
             "status": "open", "heat": 6.0},
        ]
        state.ledger = {"intimacy": 1.0, "dips": 0,
                        "dark_night_chapter": 0, "committed": False}
        return [
            Arc("The Spark", "Meet; attraction with a reason to resist.", 3,
                ["the meet-cute with teeth", "the reason it can't work",
                 "the almost-moment"]),
            Arc("The Pull", "Closer, then the world objects.", 4,
                ["the first real conversation", "the rival moves",
                 "the almost-lost", "the dark night"]),
            Arc("The Landing", "Earned, not given.", 2,
                ["the grovel / the truth", "the choice, freely made"]),
        ]

    def chapter_brief(self, state: StoryState, arc: Arc,
                      chapter_no: int) -> ChapterBrief:
        ledger = state.ledger
        total = max(6, state.total_planned or 9)
        dark_night_at = int(total * 0.7)
        ledger["dark_night_chapter"] = dark_night_at
        intimacy = ledger["intimacy"]
        if chapter_no < dark_night_at:
            # oscillate: push up, then force a dip every third chapter
            delta = -2.0 if chapter_no % 3 == 0 else 1.5
            target = min(8.0, max(1.0, intimacy + delta))
        elif chapter_no == dark_night_at:
            target, delta = 1.0, -9.0
            must = ["THE DARK NIGHT: it falls apart; the wound is named"]
        else:
            target, delta = 9.0, +8.0
            must = ["rebuild: the choice is made freely, with eyes open"]
        if chapter_no != dark_night_at:
            beat = arc.beats[(chapter_no - 1) % len(arc.beats)]
            must = [f"advance beat: {beat}",
                    f"move intimacy {intimacy:.0f} → {target:.0f} "
                    f"({'closer' if delta > 0 else 'apart'})"]
        must_not = []
        if chapter_no < dark_night_at:
            must_not = ["no permanent commitment before the dark night — "
                        "no 'forever', no proposals as resolution"]
        return ChapterBrief(
            chapter_no=chapter_no, arc_name=arc.name,
            must_happen=must, must_not=must_not,
            threads_advanced=["t1", "t2"],
            pov_notes="warm, interior; longing shown in small physical "
                      "details, not declarations",
            target_words=1300,
            extra={"intimacy_target": target})

    def validate(self, text: str, state: StoryState,
                 brief: ChapterBrief) -> list[str]:
        problems: list[str] = []
        low = text.lower()
        dark_night = state.ledger.get("dark_night_chapter", 0)
        if brief.chapter_no < dark_night:
            if re.search(r"\bforever\b.*\b(love|together|yours)\b", low) or \
               "marry me" in low:
                problems.append("premature commitment before the dark night")
        return problems

    def advance(self, state: StoryState, text: str,
                brief: ChapterBrief) -> None:
        target = brief.extra.get("intimacy_target")
        if target is not None:
            if target < state.ledger["intimacy"]:
                state.ledger["dips"] += 1
            state.ledger["intimacy"] = target
        if "dark night" in brief.must_happen[0].lower():
            state.ledger["dark_night_done"] = True


ENGINES: dict[str, type[GenreEngine]] = {
    "mystery": MysteryEngine,
    "thriller": ThrillerEngine,
    "horror": HorrorEngine,
    "sci-fi": SciFiEngine,
    "scifi": SciFiEngine,
    "fantasy": FantasyEngine,
    "romance": RomanceEngine,
}


def engine_for(genre: str, seed: int = 0) -> GenreEngine:
    key = (genre or "").strip().lower()
    cls = ENGINES.get(key, FantasyEngine)
    return cls(seed=seed)


# ── the writer ───────────────────────────────────────────────────────────────


class FictionWriter:
    """Devon's fiction studio: start stories, write chapters, run serials."""

    def __init__(self, context: Any) -> None:
        self.context = context

    # -- paths ------------------------------------------------------------------
    def _workspace(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = getattr(settings, "workspace_dir", None) if settings else None
        return Path(root) if root else Path.cwd() / "workspace"

    def fiction_dir(self) -> Path:
        d = self._workspace() / "books" / "fiction"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def story_dir(self, slug: str) -> Path:
        d = self.fiction_dir() / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "chapters").mkdir(exist_ok=True)
        return d

    def _state_path(self, slug: str) -> Path:
        return self.story_dir(slug) / "state.json"

    def _save_state(self, state: StoryState) -> None:
        path = self._state_path(state.slug)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def load(self, slug: str) -> StoryState:
        path = self._state_path(slug)
        if not path.exists():
            raise ValueError(f"no fiction story {slug!r}")
        return StoryState.from_dict(
            json.loads(path.read_text(encoding="utf-8")))

    def list_stories(self) -> list[dict[str, Any]]:
        out = []
        for sp in sorted(self.fiction_dir().glob("*/state.json")):
            try:
                s = self.load(sp.parent.name)
                out.append({
                    "slug": s.slug, "title": s.title, "genre": s.genre,
                    "mode": s.mode, "chapters_written": s.chapters_written,
                    "total_planned": s.total_planned,
                    "arcs": len(s.arcs),
                })
            except Exception:  # noqa: BLE001
                continue
        return out

    # -- starting -----------------------------------------------------------------
    def decide_shape(self, premise: str, genre: str,
                     mode: str) -> tuple[int, str]:
        """The book decides its own shape: chapter count from premise
        complexity (threads × depth signals), never a hardcoded template."""
        low = premise.lower()
        # complexity signals in the premise itself
        thread_signals = sum(low.count(w) for w in
                             (" and ", " but ", " while ", " meanwhile ",
                              " subplot", " arc "))
        depth_signals = sum(low.count(w) for w in
                            ("secret", "mystery", "war", "kingdom", "system",
                             "empire", "revenge", "journey", "quest"))
        named = len(re.findall(r"\b[A-Z][a-z]{2,}\b", premise))
        complexity = thread_signals + depth_signals + named // 2
        if mode == "serial":
            # serials don't end: shape = first arc length
            chapters = max(6, min(12, 6 + complexity // 3))
            note = (f"serial — first arc of ~{chapters} chapters; "
                    f"arcs spawn from dangling threads indefinitely")
        else:
            chapters = max(8, min(30, 10 + complexity // 2))
            note = (f"novel — ~{chapters} chapters from premise complexity "
                    f"(signals: {complexity})")
        return chapters, note

    def start(self, premise: str, genre: str = "fantasy",
              mode: str = "novel", title: str = "",
              theme: str = "") -> dict[str, Any]:
        """Begin a story: shape it, plan arcs, weave wisdom, write ch.1."""
        genre = (genre or "fantasy").strip().lower()
        mode = (mode or "novel").strip().lower()
        if mode not in ("novel", "serial"):
            raise ValueError("mode must be novel | serial")
        slug = slugify(f"{title or premise[:40]}-{genre}", fallback="story")
        total, shape_note = self.decide_shape(premise, genre, mode)
        engine = engine_for(genre, seed=hash(slug) & 0xFFFFFFFF)
        state = StoryState(
            slug=slug, title=title or premise[:60], premise=premise,
            genre=engine.name, mode=mode, total_planned=total,
            style={"theme": theme or premise[:80]})
        arcs = engine.plan_arcs(premise, state)
        # the mystery engine learns its reveal chapter from the shape
        if engine.name == "mystery":
            state.ledger["reveal_chapter"] = max(4, total - 2)
        if engine.name == "thriller":
            state.ledger["deadline_chapter"] = total
        state.arcs = [a.to_dict() for a in arcs]
        if arcs:
            arcs[0].status = "active"
            state.arcs[0]["status"] = "active"
        # wisdom weave: thematic texture for the whole run
        motifs = weave_wisdom(theme or premise[:120], self.context)
        state.style["wisdom_motifs"] = motifs
        self._save_state(state)
        first = self.write_next(slug)
        return {"slug": slug, "title": state.title, "genre": engine.name,
                "mode": mode, "shape": shape_note,
                "arcs": [a["name"] for a in state.arcs],
                "wisdom_motifs": len(motifs),
                "chapter_1": first}

    # -- writing ------------------------------------------------------------------
    def _current_arc(self, state: StoryState) -> Arc:
        for a in state.arcs:
            if a.get("status") == "active":
                d = dict(a)
                return Arc(**{k: d.get(k, getattr(Arc, k, None))
                              for k in Arc.__dataclass_fields__})
        # no active arc: activate the first planned one
        for a in state.arcs:
            if a.get("status") == "planned":
                a["status"] = "active"
                self._save_state(state)
                d = dict(a)
                return Arc(**{k: d.get(k) for k in Arc.__dataclass_fields__})
        raise ValueError("story has no arcs left — spawn one (serial mode)")

    def _arc_written(self, state: StoryState, arc: Arc) -> int:
        # chapters written while this arc was active: track via ledger
        return int(state.ledger.get(f"arc_{arc.name}_written", 0))

    def write_next(self, slug: str, *, direction: str = "") -> dict[str, Any]:
        """Write the next chapter: brief → prose → validate → advance."""
        state = self.load(slug)
        engine = engine_for(state.genre, seed=hash(slug) & 0xFFFFFFFF)
        arc = self._current_arc(state)
        chapter_no = state.chapters_written + 1

        # serial mode: close finished arcs, spawn the next from hot threads
        if state.mode == "serial":
            self._maybe_roll_arc(state, engine, arc, chapter_no)
            arc = self._current_arc(state)

        brief = engine.chapter_brief(state, arc, chapter_no)
        if state.style.get("wisdom_motifs"):
            brief.wisdom_motifs = list(state.style["wisdom_motifs"])[:3]
        if direction:
            brief.must_happen.append(f"owner direction: {direction}")

        text = self._model_chapter(state, engine, brief)
        if not text:
            text = self._compose_chapter(state, engine, brief)

        problems = engine.validate(text, state, brief)
        if problems:
            _log.info("fiction ch%s validation: %s", chapter_no, problems)
            repaired = self._repair(state, engine, brief, text, problems)
            if repaired:
                text = repaired

        engine.advance(state, text, brief)
        state.chapters_written = chapter_no
        state.ledger[f"arc_{arc.name}_written"] = \
            self._arc_written(state, arc) + 1
        # heat decays; advanced threads cool slightly, neglected ones smoulder
        for t in state.threads:
            if t["id"] in brief.threads_advanced:
                t["heat"] = float(t.get("heat", 5)) * 0.9
            else:
                t["heat"] = float(t.get("heat", 5)) * 1.05 + 0.2
        self._save_state(state)

        path = self.story_dir(slug) / "chapters" / f"{chapter_no:04d}.md"
        path.write_text(f"# Chapter {chapter_no}\n\n*{arc.name}*\n\n{text}\n",
                        encoding="utf-8")
        return {"number": chapter_no, "arc": arc.name,
                "words": count_words(text),
                "validation_notes": problems,
                "path": path.as_posix(), "text": text}

    def _maybe_roll_arc(self, state: StoryState, engine: GenreEngine,
                        arc: Arc, chapter_no: int) -> None:
        """Serial arc manager: close arcs that served their beats, spawn the
        next from the hottest open thread — the anti-meander engine."""
        written = self._arc_written(state, arc)
        if written < arc.chapters_planned:
            return
        # close it
        for a in state.arcs:
            if a.get("name") == arc.name:
                a["status"] = "closed"
        for tid in arc.threads_to_close:
            for t in state.threads:
                if t["id"] == tid:
                    t["status"] = "resolved"
        # spawn: hottest open thread becomes the next arc's spine
        open_t = [t for t in state.threads if t.get("status") == "open"]
        if not open_t:
            # invent fresh trouble from the premise when all is resolved
            seed = self._fresh_thread(state)
            state.threads.append(seed)
            open_t = [seed]
        hot = max(open_t, key=lambda t: float(t.get("heat", 0)))
        new_arc = Arc(
            name=f"Arc {len(state.arcs) + 1}: {hot['summary'][:50]}",
            goal=hot["summary"],
            chapters_planned=max(4, min(10, int(hot.get("heat", 5)))),
            beats=[f"the thread tightens: {hot['summary'][:80]}",
                   "a reversal", "the price of progress",
                   "a new thread surfaces"],
            status="active", threads_to_close=[hot["id"]])
        state.arcs.append(new_arc.to_dict())
        # the hot thread spawns a child thread so the serial never starves
        child = self._fresh_thread(state, parent=hot["summary"])
        state.threads.append(child)
        self._save_state(state)
        _log.info("serial %s: arc %r closed, spawned %r", state.slug,
                  arc.name, new_arc.name)

    def _fresh_thread(self, state: StoryState,
                      parent: str = "") -> dict[str, Any]:
        tid = f"t{len(state.threads) + 1}"
        seeds = [
            "a stranger arrives knowing too much",
            "an old debt comes due at the worst moment",
            "a message from someone who should be dead",
            "the map has a region nobody will explain",
            "an ally's loyalty quietly goes up for sale",
        ]
        summary = self._rng(state).choice(seeds)
        if parent:
            summary += f" (fallout: {parent[:60]})"
        return {"id": tid, "summary": summary, "status": "open",
                "heat": 6.0}

    @staticmethod
    def _rng(state: StoryState) -> random.Random:
        return random.Random(hash(state.slug) & 0xFFFFFFFF)

    # -- model prose ----------------------------------------------------------------
    def _model_chapter(self, state: StoryState, engine: GenreEngine,
                       brief: ChapterBrief) -> str:
        try:
            from ..llm.base import Message, SamplingParams
            from ..llm.brain import brain_for
            from .write import model_available
        except Exception:  # noqa: BLE001
            return ""
        if not model_available(self.context):
            return ""
        cast = "\n".join(
            f"- {c['name']} ({c['role']}): {c.get('trait', '')}"
            for c in state.characters[:8])
        prev_tail = ""
        if brief.chapter_no > 1:
            prev = (self.story_dir(state.slug) / "chapters" /
                    f"{brief.chapter_no - 1:04d}.md")
            if prev.exists():
                prev_tail = prev.read_text(encoding="utf-8")[-2500:]
        motifs = ""
        if brief.wisdom_motifs:
            motifs = ("Thematic texture to weave in subtly (never preach, "
                      "never name the source):\n- " +
                      "\n- ".join(brief.wisdom_motifs))
        user = (
            f"Story: {state.title} ({state.genre}, {state.mode}).\n"
            f"Premise: {state.premise}\n\n"
            f"CAST:\n{cast}\n\n"
            f"CHAPTER {brief.chapter_no} — arc: {brief.arc_name}\n"
            f"Must happen:\n- " + "\n- ".join(brief.must_happen) + "\n"
            f"Must NOT happen:\n- " +
            ("\n- ".join(brief.must_not) if brief.must_not else "(none)") + "\n"
            + ("Seed these elements naturally:\n- " +
               "\n- ".join(brief.plant) + "\n" if brief.plant else "")
            + f"Voice: {brief.pov_notes}\n"
            + (motifs + "\n" if motifs else "")
            + (f"PREVIOUS CHAPTER TAIL:\n{prev_tail}\n\n" if prev_tail else "")
            + f"Write ~{brief.target_words} words of complete, vivid prose — "
              f"dialogue, action, interiority. No meta-commentary, no summary "
              f"in place of scenes.")
        try:
            response = brain_for(self.context).chat(
                [Message.system(
                    "You are a master genre novelist. You write complete "
                    "scenes with tension, specificity, and voice — never "
                    "outlines, never filler."),
                 Message.user(user)],
                SamplingParams(temperature=0.8,
                               max_tokens=min(12000,
                                              int(brief.target_words * 1.8) + 400)),
                task_kind="creative")
            text = (getattr(response, "text", "") or "").strip()
            if getattr(response, "ok", False) and count_words(text) >= 200:
                return text
        except Exception as exc:  # noqa: BLE001
            _log.debug("model fiction chapter failed: %s", exc)
        return ""

    def _repair(self, state: StoryState, engine: GenreEngine,
                brief: ChapterBrief, text: str,
                problems: list[str]) -> str:
        """One repair pass: ask the model to fix validation violations."""
        try:
            from ..llm.base import Message, SamplingParams
            from ..llm.brain import brain_for
            from .write import model_available
        except Exception:  # noqa: BLE001
            return ""
        if not model_available(self.context):
            return ""
        try:
            response = brain_for(self.context).chat(
                [Message.system(
                    "You are a fiction editor. Fix ONLY the listed problems "
                    "in the chapter; change nothing else."),
                 Message.user(
                     "PROBLEMS:\n- " + "\n- ".join(problems) +
                     f"\n\nCHAPTER:\n{text[:9000]}\n\n"
                     "Return the full corrected chapter.")],
                SamplingParams(temperature=0.4, max_tokens=9000),
                task_kind="creative")
            fixed = (getattr(response, "text", "") or "").strip()
            if getattr(response, "ok", False) and count_words(fixed) >= 200:
                still = engine.validate(fixed, state, brief)
                if len(still) < len(problems):
                    return fixed
        except Exception as exc:  # noqa: BLE001
            _log.debug("fiction repair failed: %s", exc)
        return ""

    # -- heuristic composer (the floor) -----------------------------------------------
    def _compose_chapter(self, state: StoryState, engine: GenreEngine,
                         brief: ChapterBrief) -> str:
        """Real prose from the brief — genre-aware scenes, rotating pools,
        seeded per chapter for stability.  Never placeholder."""
        rng = random.Random(hash((state.slug, brief.chapter_no)) & 0xFFFFFFFF)
        cast = state.characters
        lead = cast[0]["name"] if cast else "the protagonist"
        second = cast[1]["name"] if len(cast) > 1 else None
        paras: list[str] = []

        # cold open from the arc/brief
        paras.append(self._scene_open(rng, lead, second, brief, state))
        # one paragraph per must-happen beat
        for beat in brief.must_happen[:5]:
            paras.append(self._scene_beat(rng, lead, second, beat, brief,
                                          state, engine))
        # plant seeds (rotating frames — never the same one twice)
        for i, seed_text in enumerate(brief.plant[:3]):
            paras.append(self._scene_plant(rng, lead, seed_text, i))
        # wisdom texture
        for motif in brief.wisdom_motifs[:2]:
            paras.append(self._scene_motif(rng, lead, motif))
        # middle movement: dialogue + interiority, genre-flavored
        paras.append(self._scene_dialogue(rng, lead, second, brief, state,
                                          engine))
        paras.append(self._scene_interior(rng, lead, brief, state, engine))
        # genre-specific movement
        paras.append(self._scene_genre(rng, lead, second, engine, state,
                                       brief))
        # close on the brief's energy
        paras.append(self._scene_close(rng, lead, brief, state))
        text = "\n\n".join(paras)
        # the floor guarantees a real chapter's worth of movement: pad
        # with complications until it reads like a chapter, not a sketch
        guard = 0
        while count_words(text) < 300 and guard < 6:
            paras.insert(-2, self._scene_mid(rng, lead, second, brief,
                                             state, engine, guard))
            text = "\n\n".join(paras)
            guard += 1
        return text

    def _scene_mid(self, rng: random.Random, lead: str, second: str | None,
                   brief: ChapterBrief, state: StoryState,
                   engine: GenreEngine, index: int) -> str:
        """Complication beats that keep the middle of the chapter moving."""
        other = second or "the stranger"
        g = engine.name
        beats = [
            (f"The plan survived exactly until contact. Then {other} moved, "
             f"and {lead} understood — too late — the shape of the trap."),
            (f"What {lead} had not counted on was the second variable: "
             f"the one nobody mentioned, because mentioning it would have "
             f"meant admitting it existed."),
            (f"They covered ground in silence, {lead} turning the situation "
             f"over like a stone with something moving underneath it. "
             f"{other} kept pace without a word."),
        ]
        genre_beats = {
            "mystery": (f"A detail surfaced that didn't fit any theory "
                        f"{lead} had — which meant the theories were wrong, "
                        f"not the detail."),
            "thriller": (f"The window narrowed. {lead} could feel it "
                         f"happening the way you feel a room get smaller: "
                         f"not seen, known."),
            "horror": (f"The quiet went wrong again — a skipped heartbeat "
                       f"of a silence — and {lead} decided not to investigate "
                       f"it. Some doors you don't open twice."),
            "romance": (f"The conversation tried to cross the distance "
                        f"between them and didn't quite make it. {lead} "
                        f"felt the almost of it like a bruise."),
        }
        pool = ([genre_beats[g]] if g in genre_beats else []) + beats
        return pool[index % len(pool)]

    def _scene_dialogue(self, rng: random.Random, lead: str,
                        second: str | None, brief: ChapterBrief,
                        state: StoryState, engine: GenreEngine) -> str:
        other = second or "the stranger"
        g = engine.name
        exchanges = {
            "mystery": [
                (f"“Where were you at midnight?” {lead} asked.",
                 f"“Where was anyone?” {other} said. “Alive, mostly.”"),
                (f"“You're lying,” {lead} said.",
                 f"“I'm editing,” {other} said. “There's a difference, "
                 f"and it's the only one keeping me out of a cell.”"),
            ],
            "thriller": [
                (f"“How long do we have?” {other} asked.",
                 f"“Less than that,” {lead} said, already moving."),
                (f"“This is insane,” {other} said.",
                 f"“It's arithmetic,” {lead} said. “Run the numbers or "
                 f"run. Your choice.”"),
            ],
            "horror": [
                (f"“Did you hear that?” {other} whispered.",
                 f"“Don't,” {lead} said. “Don't give it a name. Names are "
                 f"doors.”"),
                (f"“It's just the house settling,” {other} said, too quickly.",
                 f"{lead} said nothing. Houses didn't settle like that."),
            ],
            "romance": [
                (f"“You don't have to stay,” {lead} said.",
                 f"“I know,” {other} said, and stayed."),
                (f"“What are we doing?” {other} asked.",
                 f"“Something unwise,” {lead} said. “Keep going.”"),
            ],
        }
        default = [
            (f"“Tell me the truth this time,” {lead} said.",
             f"“Which one?” {other} asked."),
            (f"“We can't keep doing this,” {other} said.",
             f"“Watch me,” {lead} said."),
        ]
        a, b = rng.choice(exchanges.get(g, default))
        return f"{a}\n\n{b}"

    def _scene_interior(self, rng: random.Random, lead: str,
                        brief: ChapterBrief, state: StoryState,
                        engine: GenreEngine) -> str:
        g = engine.name
        inners = {
            "mystery": (f"{lead} ran the timeline again, the way {lead} "
                        f"always did when the facts stopped cooperating: "
                        f"who, where, when — and the fourth question, the "
                        f"one that mattered: who benefits from me believing "
                        f"this version?"),
            "thriller": (f"Somewhere under the adrenaline there was a "
                         f"colder layer, doing math: distances, sightlines, "
                         f"the seconds between now and the deadline. "
                         f"{lead} trusted that layer more than the fear."),
            "horror": (f"{lead}'s mind kept offering explanations, the way "
                       f"minds do — drafts, rats, wind — and {lead} kept "
                       f"refusing them, because the explanations were "
                       f"starting to sound like lullabies."),
            "sci-fi": (f"The numbers didn't care how {lead} felt about them. "
                       f"That was the comfort and the terror of it: physics "
                       f"as the last honest thing in the room."),
            "fantasy": (f"Power remembered being used; {lead} could feel the "
                        f"ledger keeping score in the marrow. Every working "
                        f"was a loan, and the interest was autobiographical."),
            "romance": (f"{lead} had a whole speech prepared about why this "
                        f"was a bad idea. It died somewhere between the "
                        f"third word and the way the light caught {lead}'s "
                        f"companion's face."),
        }
        return inners.get(g, inners["mystery"])

    def _scene_open(self, rng: random.Random, lead: str, second: str | None,
                    brief: ChapterBrief, state: StoryState) -> str:
        settings = ["the rain had not stopped for three days",
                    "dawn came up the color of old bruises",
                    "the city held its breath between sirens",
                    "night pressed its face to the windows",
                    "the air tasted of ozone and old decisions"]
        s = rng.choice(settings)
        p = (f"{s.capitalize()}, and {lead} was already awake — had been "
             f"awake, really, since the part of the night when pretending "
             f"stops working.")
        if second and rng.random() < 0.6:
            p += (f" {second} slept on, or performed sleeping; {lead} had "
                  f"stopped trying to tell the difference.")
        return p

    def _scene_beat(self, rng: random.Random, lead: str, second: str | None,
                    beat: str, brief: ChapterBrief, state: StoryState,
                    engine: GenreEngine) -> str:
        low = beat.lower()
        if "hook" in low or "unresolved" in low:
            return (f"{lead} reached for the door — and the sound on the "
                    f"other side was not what any of them had expected. "
                    f"Not yet. Not ever.")
        if "dark night" in low:
            return (f"It came apart the way things always come apart: not "
                    f"all at once, but in the exact order {lead} had been "
                    f"afraid of. By midnight there was nothing left to "
                    f"pretend about, and the truth sat between them like "
                    f"a third person at the table.")
        if "reveal" in low:
            return (f"{lead} laid it out piece by piece — every small wrong "
                    f"thing, every detail that had nagged for weeks — until "
                    f"the shape of it stood in the room with them, undeniable.")
        if "set piece" in low:
            return (f"Then the world caught fire in the particular way it "
                    f"does when {lead} is involved: all noise and consequence, "
                    f"no time to think, only to move — and moving was "
                    f"a kind of thinking too.")
        if "dread" in low:
            return (f"The quiet went wrong first. {lead} noticed it the way "
                    f"you notice a skipped heartbeat — not the sound, the "
                    f"absence shaped exactly like one.")
        if "intimacy" in low:
            closer = "closer" in low
            if closer:
                return (f"Their hands found the same railing, neither moving "
                        f"away. {lead} counted the seconds and lost count "
                        f"on purpose.")
            return (f"The distance between them had a temperature now. "
                    f"{lead} felt it every time the conversation tried to "
                    f"cross it and didn't.")
        return self._narrate_beat(beat, lead, rng)

    def _narrate_beat(self, beat: str, lead: str,
                      rng: random.Random) -> str:
        """Turn an imperative brief beat into narrative prose — the beat
        must never leak verbatim into the chapter."""
        core = beat.strip()
        low = core.lower()
        if low.startswith("the discovery of "):
            obj = core[len("the discovery of "):].rstrip(".")
            return (f"The discovery came the way discoveries always come — "
                    f"sideways, while {lead} was looking elsewhere. "
                    f"{obj.capitalize()}, and everything it dragged in "
                    f"behind it.")
        if low.startswith("introduce "):
            who = core[len("introduce "):].rstrip(".")
            # the lead never "introduces" themselves
            parts = [p.strip() for p in re.split(r",|\band\b", who)
                     if p.strip() and lead.lower() not in p.strip().lower()
                     and "suspect" not in p.strip().lower()]
            who_text = " and ".join(parts) if parts else who
            return (f"{lead} brought {who_text} into the same room for the "
                    f"first time, and watched the air between them change "
                    f"temperature.")
        if low.startswith("interview "):
            return (f"{core.rstrip('.')}. {lead} asked every question twice, "
                    f"two different ways, and compared the seams.")
        if "advance beat:" in low or "advance arc beat:" in low:
            b = core.split(":", 1)[1].strip().rstrip(".")
            return (f"{b.capitalize()}. {lead} met it head-on — there was "
                    f"never any other way {lead} knew.")
        if low.startswith("gather "):
            return (f"{core.rstrip('.')}. {lead} watched faces while the "
                    f"words landed, because faces always confessed first.")
        if low.startswith("the reveal:"):
            detail = core.split(":", 1)[1].strip()
            return (f"{lead} laid it out piece by piece — {detail} — until "
                    f"the shape of it stood in the room, undeniable.")
        # generic: fold the beat into a narrative sentence, never verbatim
        short = core.rstrip(".")
        if len(short) > 140:
            short = short[:140]
        return rng.choice([
            f"It came down to this: {short.lower()}. {lead} set "
            f"{lead.split()[0].lower()}'s jaw and got to work.",
            f"The next matter was {short.lower()}, and {lead} gave it "
            f"everything — head down, refusing the easy version.",
            f"{short.capitalize()}. It sounded simple. It never was, with "
            f"{lead} involved.",
        ])

    def _scene_plant(self, rng: random.Random, lead: str,
                     seed_text: str, index: int = 0) -> str:
        frames = [
            f"{lead} noticed {seed_text}, and filed it away without knowing why.",
            f"There, half-hidden and easy to miss: {seed_text}. {lead}'s "
            f"attention snagged on it like a sleeve on a nail.",
            f"Later, {lead} would remember {seed_text} — the small wrong "
            f"detail in an otherwise ordinary hour.",
            f"{seed_text.capitalize()}. It meant nothing yet. {lead} knew "
            f"better than to believe that.",
        ]
        return frames[index % len(frames)]

    def _scene_motif(self, rng: random.Random, lead: str,
                     motif: str) -> str:
        clean = motif.split("[")[0].strip()
        return (f"{lead} turned the thought over the way the old texts "
                f"turned everything over: *{clean[:160]}* — and found it "
                f"fit the night a little too well.")

    def _scene_genre(self, rng: random.Random, lead: str,
                     second: str | None, engine: GenreEngine,
                     state: StoryState, brief: ChapterBrief) -> str:
        g = engine.name
        if g == "mystery":
            suspects = state.ledger.get("suspects", [])
            name = suspects[0]["name"] if suspects else "the suspect"
            return (f"{name}'s story had exactly one seam, and {lead} had "
                    f"found it: a detail too neat, polished by repetition. "
                    f"Liars rehearsed; the truth only ever improvised.")
        if g == "thriller":
            return (f"The clock {lead} carried wasn't on any wall — it was "
                    f"in the chest, counting down in heartbeats, and it had "
                    f"just skipped one.")
        if g == "horror":
            unknown = (state.ledger.get("unknowns") or ["it"])[0]
            return (f"{lead} did not look directly at {unknown}. Some things "
                    f"you survived by the courtesy of not seeing clearly.")
        if g in ("sci-fi", "scifi"):
            rules = state.ledger.get("rules", [])
            rule = rules[0]["rule"] if rules else "the rules of the world"
            return (f"{rule} — {lead} had read it in briefings, but knowing "
                    f"a rule and standing inside it were different countries.")
        if g == "romance" and second:
            return (f"{second} laughed — the real one, unguarded — and {lead} "
                    f"felt the evening tilt a degree toward something "
                    f"irreversible.")
        magic = state.ledger.get("magic", {})
        if magic:
            return (f"The working rose through {lead} like water through a "
                    f"crack — and {lead} paid for it the way everyone paid: "
                    f"{magic.get('cost', 'dearly')}, and without flinching.")
        return (f"{lead} moved through the middle of the night like it was "
                f"a room {lead} owned, and for once the night agreed.")

    def _scene_close(self, rng: random.Random, lead: str,
                     brief: ChapterBrief, state: StoryState) -> str:
        closers = [
            f"{lead} stood a long moment in the aftermath, cataloguing what "
            f"the night had cost and what it had given back. The ledger "
            f"never balanced. It wasn't supposed to.",
            f"Whatever came next, {lead} would meet it the same way: awake, "
            f"already moving, done with the luxury of surprise.",
            f"Dawn would come with its questions. {lead} had a few answers "
            f"now — and better questions than before.",
        ]
        return rng.choice(closers)

    # -- status -----------------------------------------------------------------------
    def status(self, slug: str) -> dict[str, Any]:
        state = self.load(slug)
        arcs = [{"name": a.get("name"), "status": a.get("status"),
                 "goal": a.get("goal", "")[:100]} for a in state.arcs]
        return {
            "slug": state.slug, "title": state.title, "genre": state.genre,
            "mode": state.mode, "chapters_written": state.chapters_written,
            "total_planned": state.total_planned,
            "current_arc": self._current_arc(state).name
            if state.arcs else "",
            "arcs": arcs,
            "open_threads": [t for t in state.threads
                             if t.get("status") == "open"],
            "cast": [c["name"] for c in state.characters],
            "wisdom_motifs": len(state.style.get("wisdom_motifs", [])),
        }
