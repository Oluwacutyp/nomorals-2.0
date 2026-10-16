"""Two mechanical tests — read-aloud and delete-the-first-paragraph.

Both are measurements, not verdicts.  Neither blocks.  The owner
overrules.

1. Read aloud — :func:`read_aloud_check` measures what a voice would
   stumble on: sentences over 40 words, consonant pileups, and
   sentences that stack 3+ clauses before the verb arrives.  With
   ``try_audio=True`` it really synthesises the passage through
   :mod:`nomorals.voice.tts` (``UniversalTTS`` — the same engine the
   voice stack uses) and reports the backend + output path, or the
   honest failure.  Audio synthesis is opt-in: the text measurement
   runs offline and never needs a backend.

2. Delete the first paragraph — :func:`delete_first_paragraph_test`:
   if the chapter reads better without its first paragraph, the pass
   should have caught it.  The test scores the first paragraph for
   throat-clearing markers (weather opening, "for as long as X could
   remember", backstory dump, waking up, mirror description) and for
   doing nothing (no character, no decision, no dialogue) while the
   second paragraph carries action.  The same test runs per scene
   (:func:`delete_first_paragraph_per_scene`) and on the last paragraph
   (:func:`delete_last_paragraph_test` — the moralising-last-line twin
   of the craft ending pass).
"""

from __future__ import annotations

import re
from typing import Any

from .prose import split_paragraphs, split_sentences

__all__ = [
    "read_aloud_check", "synthesize_for_read_aloud",
    "delete_first_paragraph_test", "delete_first_paragraph_per_scene",
    "delete_last_paragraph_test", "split_scenes",
]

_VERB_HINTS = frozenset("""
    was were is are had has have did do does went came saw took made
    knew thought felt looked turned walked stood sat lay began became
    brought built caught chose fought found gave grew heard kept left
    lost met paid ran rose spoke stood understood wrote asked replied
    moved opened closed waited listened watched remembered decided
    wanted needed tried called fell hit broke reached pulled pushed
    said says say asked asks ask told tells tell came comes come
""".split())

_THROAT_CLEARING = (
    r"for as long as .* could remember",
    r"it was a (dark|stormy|cold|quiet|bright|grey|gray).* night",
    r"the (rain|sun|wind|morning) (had|was)",
    r"woke up|woken up|alarm (clock )?rang|alarm went off",
    r"looked (at (him|her|them)self|in the mirror)",
    r"in the beginning|once upon a time|it all started",
    r"little did|in that moment|the story (begins|starts)",
    r"had always (been|wanted|known|lived|dreamed)",
    r"ever since (he|she|they) (was|were|had)",
    r"the (town|village|city|kingdom) of .* (was|had|lay)",
)
_DIALOGUE_RE = re.compile(r'"[^"]{2,}"|“[^”]{2,}”')
_MORALISING_END = (
    "in the end", "what mattered", "the lesson", "learned that",
    "the meaning", "and that was", "after all", "the truth was",
    "what it means to", "the real meaning", "from now on",
    "never again", "forever",
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or "")


def _consonant_pileups(sentence: str) -> list[str]:
    """Words with 4+ consecutive consonants — tongue-twisters on read-aloud."""
    hits: list[str] = []
    for w in _words(sentence):
        if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", w.lower()):
            hits.append(w)
    return hits


_AUXILIARIES = frozenset("""
    have has had be been being is are was were am do does did
    will would can could should may might must
""".split())


def _clauses_before_verb(sentence: str) -> int:
    """Comma-clauses before the main verb arrives.

    The main verb is the LAST non-auxiliary verb hint in the sentence:
    auxiliaries ("have", "had", "was") open subordinate clauses, so
    counting to the first auxiliary would call nearly every complex
    sentence late.  A sentence whose main verb arrives after 3+
    comma-clauses is a read-aloud stumble.
    """
    toks = [w.lower().strip(",.;:") for w in sentence.split()]
    main = [i for i, t in enumerate(toks)
            if t in _VERB_HINTS and t not in _AUXILIARIES]
    if not main:
        main = [i for i, t in enumerate(toks) if t in _VERB_HINTS]
    if not main:
        return 0
    verb_idx = main[-1]
    head = sentence[:sum(len(t) + 1 for t in toks[:verb_idx])]
    return head.count(",")


def read_aloud_check(text: str, *, try_audio: bool = False,
                     tts: Any = None) -> dict[str, Any]:
    """Measure what a voice would stumble on when reading ``text`` aloud.

    ``try_audio=True`` additionally synthesises through
    ``nomorals.voice.tts.UniversalTTS`` (see
    :func:`synthesize_for_read_aloud`); the text measurement always
    runs, audio or not.  Never raises for a missing backend — the
    audio block reports the failure honestly.
    """
    sents = split_sentences(text or "")
    over_40: list[dict[str, Any]] = []
    pileups: list[dict[str, Any]] = []
    late_verbs: list[dict[str, Any]] = []
    for i, sent in enumerate(sents, 1):
        n = len(_words(sent))
        if n > 40:
            over_40.append({"sentence": i, "words": n,
                            "excerpt": sent[:180]})
        piles = _consonant_pileups(sent)
        if piles:
            pileups.append({"sentence": i, "words": piles[:5],
                            "excerpt": sent[:180]})
        clauses = _clauses_before_verb(sent)
        if clauses >= 3:
            late_verbs.append({"sentence": i,
                               "clauses_before_verb": clauses,
                               "excerpt": sent[:180]})
    stumbles = len(over_40) + len(pileups) + len(late_verbs)
    total_sents = max(1, len(sents))
    audio = None
    if try_audio:
        audio = synthesize_for_read_aloud(text, tts=tts)
    return {
        "sentences": len(sents),
        "sentences_over_40_words": over_40,
        "consonant_pileups": pileups,
        "clauses_before_verb_3plus": late_verbs,
        "stumbles": stumbles,
        "stumble_rate": round(stumbles / total_sents, 3),
        "readable_aloud": stumbles == 0,
        "audio": audio,
        "blocked": False,
        "note": ("Read-aloud stumbles are reported, not blocking. "
                 "If a sentence can't be spoken in one breath, "
                 "rewrite it — or keep it, if the breathlessness "
                 "is the point."),
    }


def synthesize_for_read_aloud(text: str, *, tts: Any = None,
                              out_path: str = "") -> dict[str, Any]:
    """Synthesise ``text`` through ``nomorals.voice.tts.UniversalTTS``.

    Returns ``{ok, backend, path, error}`` — never raises.  ``ok=False``
    with an error string is an honest "no backend available" report,
    not a failure of the read-aloud test: the text measurement in
    :func:`read_aloud_check` already ran.
    """
    try:
        from ..voice.tts import UniversalTTS, available_backends
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "backend": None, "path": None,
                "error": f"voice.tts import failed: {exc}"}
    try:
        backends = available_backends()
    except Exception:  # noqa: BLE001
        backends = []
    if not backends:
        return {"ok": False, "backend": None, "path": None,
                "error": "no TTS backend installed (see nomorals/voice/tts.py)"}
    engine = tts or UniversalTTS()
    try:
        result = engine.speak(text[:3000], out_path=out_path or "")
        return {"ok": True, "backend": result.get("backend"),
                "path": result.get("path"),
                "segments": result.get("segments"),
                "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "backend": None, "path": None,
                "error": f"TTS synthesis failed: {exc}"}


def split_scenes(text: str) -> list[str]:
    """Split a chapter into scenes on horizontal rules / headings / gaps.

    Scene breaks are ``---`` / ``***`` / ``#`` headings / 3+ blank
    lines.  Without any break the whole chapter is one scene.
    """
    chunks = re.split(r"\n\s*(?:---|\*\*\*|#{1,3}\s+[^\n]+)\s*\n|\n{3,}",
                      text or "")
    scenes = [c.strip() for c in chunks if c.strip()]
    return scenes or ([(text or "").strip()] if (text or "").strip() else [])


def _throat_clearing_score(first: str) -> list[str]:
    low = first.lower()
    return [pat for pat in _THROAT_CLEARING if re.search(pat, low)]


def _does_nothing(first: str) -> bool:
    """A first paragraph doing nothing: no character, decision, dialogue."""
    low = first.lower()
    has_dialogue = bool(_DIALOGUE_RE.search(first))
    has_decision = any(v in low for v in (
        "decided", "chose", "refused", "promised", "asked", "ran",
        "opened", "grabbed", "turned", "left", "arrived"))
    has_name = bool(re.search(r"\b[A-Z][a-z]{2,}\b", first))
    return not (has_dialogue or has_decision or has_name)


def delete_first_paragraph_test(text: str) -> dict[str, Any]:
    """Would the chapter read better without its first paragraph?

    Verdict ``likely_better_without`` when the first paragraph is
    throat-clearing (weather opening, waking up, mirror, backstory
    dump, "for as long as X could remember") or does nothing while
    the second paragraph carries the scene.  ``keep`` otherwise, with
    the signals listed either way so the owner can overrule.
    """
    paras = split_paragraphs(text or "")
    if len(paras) < 2:
        return {"verdict": "keep", "reason": "fewer than two paragraphs",
                "signals": [], "blocked": False}
    first, second = paras[0], paras[1]
    markers = _throat_clearing_score(first)
    nothing = _does_nothing(first)
    second_moves = bool(_DIALOGUE_RE.search(second)) or any(
        v in second.lower() for v in (
            "decided", "chose", "ran", "opened", "grabbed", "asked",
            "arrived", "shouted", "whispered"))
    signals: list[str] = []
    if markers:
        signals.append(f"throat-clearing: {len(markers)} marker(s)")
    if nothing:
        signals.append("first paragraph does nothing (no character, "
                       "decision, or dialogue)")
    if nothing and second_moves:
        signals.append("second paragraph carries the scene — the first "
                       "is a runway, not a launch")
    better = bool(markers) or (nothing and second_moves)
    return {
        "verdict": "likely_better_without" if better else "keep",
        "reason": ("; ".join(signals) if signals else
                   "first paragraph earns its place"),
        "signals": signals,
        "throat_clearing_markers": markers,
        "first_paragraph": first[:300],
        "blocked": False,
        "note": ("If the chapter is better without its first paragraph, "
                 "the pass should have caught it — this test is that "
                 "catch, reported not blocking."),
    }


def delete_first_paragraph_per_scene(text: str) -> list[dict[str, Any]]:
    """The same delete-first-paragraph test, per scene."""
    out = []
    for i, scene in enumerate(split_scenes(text), 1):
        result = delete_first_paragraph_test(scene)
        result["scene"] = i
        out.append(result)
    return out


def delete_last_paragraph_test(text: str) -> dict[str, Any]:
    """Would the chapter read better without its last paragraph?

    The twin of the craft ending pass: a last paragraph that
    moralises, asks a rhetorical question, or closes on 'hope' is
    usually a runway after the landing.
    """
    paras = split_paragraphs(text or "")
    if len(paras) < 2:
        return {"verdict": "keep", "reason": "fewer than two paragraphs",
                "signals": [], "blocked": False}
    last = paras[-1]
    low = last.lower()
    signals: list[str] = []
    moralising = [p for p in _MORALISING_END if p in low]
    if moralising:
        signals.append("moralising last line: " + ", ".join(moralising))
    if last.strip().endswith("?"):
        signals.append("ends on a rhetorical question")
    if re.search(r"\bhope\b", low):
        signals.append("closes on the word 'hope'")
    better = bool(signals)
    return {
        "verdict": "likely_better_without" if better else "keep",
        "reason": ("; ".join(signals) if signals else
                   "last paragraph earns its place"),
        "signals": signals,
        "last_paragraph": last[:300],
        "blocked": False,
    }
