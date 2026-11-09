"""Semantic selection — intent in, a deterministic face out.

Callers never pick a font file. They state an intent (a poster
title, a deed's text face, a caption) plus hard needs (a script
or the actual text, a weight) and get a :class:`Selection`: the
primary face, an ordered fallback chain of other faces for the
glyphs the primary lacks, whether the need is actually covered,
and notes that say plainly when it is not. No roulette: the sort
is total and stable, so the same catalogue state answers the
same intent with the same face every time.

Trait classification is grounded in the parsed metadata, never
in taste: serif/sans/script/decorative come from the OS/2 IBM
family class, mono from the fixed-pitch flag and name, compact
from the OS/2 width class, weights from the weight class.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalogue import FontCatalogue
from .fontparse import FontFace

__all__ = ["Selection", "INTENTS", "select", "face_traits"]

#: The intent vocabulary (doc 8 P2) plus the plain roles.
INTENTS = (
    "text", "display", "caption", "scoreboard", "editorial",
    "playful", "technical", "cinematic", "compact", "handwritten",
    "elegant", "mono",
)

_SERIF_CLASSES = {1, 2, 3, 4, 5, 7}      # oldstyle→slab serif
_SCRIPT_CLASSES = {10}
_MONO_TOKENS = ("mono", "courier", "console", "code", "typewriter")
_SCRIPT_TOKENS = ("script", "hand", "handwriting", "cursive", "brush")
_DISPLAY_TOKENS = ("display", "poster", "headline", "titling")
_ROUNDED_TOKENS = ("rounded", "round", "soft", "comic")


@dataclass(frozen=True)
class Traits:
    serif: bool
    sans: bool
    script: bool
    mono: bool
    decorative: bool
    condensed: bool
    rounded: bool


def _name_tokens(face: FontFace) -> set[str]:
    import re
    raw = f"{face.family} {face.subfamily}".lower()
    return set(re.split(r"[^a-z0-9]+", raw)) - {""}


def face_traits(face: FontFace) -> Traits:
    # Whole-token matching only: "Inscriptional" must not read as
    # the "script" trait, "Sans" must not hide inside other words.
    tokens = _name_tokens(face)
    cls = face.family_class
    serif = cls in _SERIF_CLASSES or (
        "serif" in tokens and "sans" not in tokens)
    script = cls in _SCRIPT_CLASSES or \
        any(tok in tokens for tok in _SCRIPT_TOKENS)
    mono = face.fixed_pitch or \
        any(tok in tokens for tok in _MONO_TOKENS)
    sans = (cls == 8 or "sans" in tokens) and not (serif or script or mono)
    decorative = bool(face.color) or \
        any(tok in tokens for tok in _DISPLAY_TOKENS) or cls == 12
    condensed = face.width_class <= 4 or \
        any(tok in tokens for tok in ("condensed", "narrow", "compact"))
    rounded = any(tok in tokens for tok in _ROUNDED_TOKENS)
    return Traits(serif=serif, sans=sans, script=script, mono=mono,
                  decorative=decorative, condensed=condensed,
                  rounded=rounded)


#: intent → (trait points, preferred weight). Points are additive
#: per matching trait; weight distance penalizes separately.
_INTENT_RULES: dict[str, tuple[dict[str, int], int]] = {
    "text": ({"sans": 3, "serif": 2}, 400),
    "caption": ({"sans": 4, "serif": 1}, 400),
    "display": ({"decorative": 3, "sans": 2, "serif": 2}, 700),
    "scoreboard": ({"mono": 4, "sans": 2}, 700),
    "editorial": ({"serif": 4, "sans": 0}, 400),
    "playful": ({"rounded": 4, "decorative": 3, "script": 1}, 600),
    "technical": ({"mono": 4, "sans": 2}, 400),
    "cinematic": ({"decorative": 3, "condensed": 2, "serif": 2}, 800),
    "compact": ({"condensed": 4, "sans": 2}, 400),
    "handwritten": ({"script": 5}, 400),
    "elegant": ({"serif": 4, "sans": 1}, 400),
    "mono": ({"mono": 5, "sans": 1}, 400),
}


@dataclass
class Selection:
    face: FontFace | None
    chain: tuple[FontFace, ...] = ()
    covered: bool = False
    intent: str = "text"
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.face is not None


def _score(face: FontFace, intent: str, want_weight: int) -> float:
    rules, default_weight = _INTENT_RULES.get(
        intent, _INTENT_RULES["text"])
    traits = face_traits(face)
    points = sum(pts for trait, pts in rules.items()
                 if getattr(traits, trait, False))
    target = want_weight
    return points * 10.0 - abs(face.weight - target) / 50.0


def select(catalogue: FontCatalogue, intent: str = "text", *,
           script: str | None = None, text: str | None = None,
           weight: int | None = None,
           family: str | None = None) -> Selection:
    intent = intent if intent in _INTENT_RULES else "text"
    notes: list[str] = []
    pool = catalogue.faces()
    if family:
        fam_faces = catalogue.faces(family=family)
        if fam_faces:
            pool = fam_faces
        else:
            notes.append(f"family {family!r} not in the catalogue; "
                         "selecting across all faces")
    if not pool:
        return Selection(face=None, intent=intent,
                         notes=("the catalogue is empty — no font "
                                "files were found on this host",))
    want_weight = weight if weight is not None else \
        _INTENT_RULES[intent][1]

    def covers_need(face: FontFace) -> bool:
        if text:
            return face.covers_text(text)
        if script:
            return face.covers(script)
        return True

    need = bool(text or script)
    covering = [f for f in pool if covers_need(f)] if need else []
    covered = bool(covering) if need else True
    ranked_pool = covering if covering else pool
    if need and not covering:
        notes.append(
            f"no face in the catalogue covers "
            f"{script or 'the requested text'}; returning the best "
            "face anyway — missing glyphs will fall back or box")

    def sort_key(face: FontFace):
        return (-_score(face, intent, want_weight),
                face.family.lower(), abs(face.weight - want_weight),
                face.path, face.face_index)

    ranked = sorted(ranked_pool, key=sort_key)
    primary = ranked[0]
    if need and covering and not covers_need(primary):
        covered = False
    # Fallback chain: other families first (glyph diversity),
    # coverage-bearing faces ahead of decorative ones.
    chain_faces: list[FontFace] = [primary]
    seen_families = {primary.family}
    for face in sorted(pool, key=sort_key):
        if len(chain_faces) >= 4:
            break
        if face is primary or face.family in seen_families:
            continue
        if need and covering and not covers_need(face):
            continue
        chain_faces.append(face)
        seen_families.add(face.family)
    for face in sorted(pool, key=sort_key):
        if len(chain_faces) >= 4:
            break
        if face not in chain_faces:
            chain_faces.append(face)
    rules, _dw = _INTENT_RULES[intent]
    primary_traits = face_traits(primary)
    if rules and not any(getattr(primary_traits, trait, False)
                         for trait, pts in rules.items() if pts > 0):
        notes.append(
            f"no face with the {intent!r} character exists in the "
            "catalogue; the closest available face was returned")
    if primary.italic and intent in ("text", "caption"):
        notes.append("selected face is italic")
    return Selection(face=primary, chain=tuple(chain_faces),
                     covered=covered, intent=intent,
                     notes=tuple(notes))
