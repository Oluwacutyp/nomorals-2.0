"""Eight craft passes — each reports, none blocks.

Every pass returns a measurable report::

    {"name", "score" (0..1, higher = stronger craft), "metrics": {...},
     "findings": [...], "suggestions": [...], "blocked": False}

``score`` is a diagnostic, not a grade and never a gate.  The owner
overrules any pass.  Wired into :meth:`FictionWriter.validate` (see
``fiction.py``), which aggregates the eight passes + the AI-tell scan +
the voiceprint drift check into one advisory report — the genre
engines' own ``validate()`` keeps its blocking behaviour for real
mechanics (fair-play, rule ledgers, cost ledgers), the craft passes
never block.

Passes:

1. specificity — proper nouns, numbers, measurements per 1k; vague
   words counted.
2. restraint — cut 10–15%: cut candidates measured, a darlings pass
   (longest / most ornate sentences), readability before vs after a
   mechanical trim; readability must improve.
3. subtext — dialogue that states its emotion outright vs implies it.
4. rhythm — sentence-length mean / variance / CV, short-long runs.
5. sensory — density per sense per 1k; non-visual coverage per scene.
6. interiority — one unspoken thought per scene; scenes missing one.
7. de-filter — filtering verbs per 1k, target ~0.
8. ending — delete-the-moralising-last-line test; end on image/action.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from .prose import split_paragraphs, split_sentences
from .voiceprint import FILTERING_VERBS, SENSORY_WORDS, fingerprint

__all__ = [
    "run_craft_passes", "CRAFT_PASSES",
    "pass_specificity", "pass_restraint", "pass_subtext", "pass_rhythm",
    "pass_sensory", "pass_interiority", "pass_de_filter", "pass_ending",
    "readability",
]

CRAFT_PASSES = ("specificity", "restraint", "subtext", "rhythm",
                "sensory", "interiority", "de_filter", "ending")

_VAGUE_WORDS = (
    "thing", "things", "stuff", "a lot", "lots", "very", "really",
    "quite", "somewhat", "somehow", "somewhere", "someone", "something",
    "amazing", "incredible", "beautiful", "wonderful", "terrible",
    "nice", "good", "bad", "big", "small", "huge",
)
_FILLER_PHRASES = (
    "in order to", "due to the fact that", "at this point in time",
    "for all intents and purposes", "it goes without saying",
    "needless to say", "as a matter of fact", "in the process of",
    "a lot of", "kind of", "sort of", "just then", "of course",
)
_ON_THE_NOSE = (
    r"\bi feel (angry|sad|happy|scared|afraid|guilty|ashamed|jealous)\b",
    r"\bi am (angry|sad|happy|scared|afraid|guilty|ashamed|jealous|furious)\b",
    r"\bi'm (angry|sad|happy|scared|afraid|guilty|ashamed|jealous|furious)\b",
    r"\bi love you\b", r"\bi hate you\b", r"\bi don't trust you\b",
    r"\byou betrayed me\b", r"\bi'm leaving\b",
)
_MORALISING_END = (
    "in the end", "what mattered", "the lesson", "learned that",
    "the meaning", "and that was", "after all", "the truth was",
    "what it means to", "the real meaning", "from now on",
    "never again", "forever",
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or "")


def _syllables(word: str) -> int:
    word = word.lower().strip("'")
    if len(word) <= 3:
        return 1
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e"):
        groups = max(1, groups - 1)
    return max(1, groups)


def readability(text: str) -> float:
    """Flesch Reading Ease (0..100, higher = easier). 0.0 for empty text."""
    sents = split_sentences(text)
    words = _words(text)
    if not sents or not words:
        return 0.0
    asl = len(words) / len(sents)
    asw = sum(_syllables(w) for w in words) / len(words)
    return round(max(0.0, min(100.0, 206.835 - 1.015 * asl - 84.6 * asw)), 1)


def _report(name: str, score: float, metrics: dict[str, Any],
            findings: list[str], suggestions: list[str]) -> dict[str, Any]:
    return {"name": name, "score": round(max(0.0, min(1.0, score)), 3),
            "metrics": metrics, "findings": findings,
            "suggestions": suggestions, "blocked": False}


def pass_specificity(text: str) -> dict[str, Any]:
    """Specific nouns / numbers / measurements per 1k; vague words counted."""
    vp = fingerprint(text)
    low = text.lower()
    vague = {w: len(re.findall(r"\b" + re.escape(w) + r"\b", low))
             for w in _VAGUE_WORDS}
    vague = {w: c for w, c in vague.items() if c}
    vague_total = sum(vague.values())
    words = max(1, vp.words)
    metrics = {
        "specificity_per_1k": vp.specificity_per_1k,
        "proper_nouns_per_1k": vp.proper_nouns_per_1k,
        "numbers_per_1k": vp.numbers_per_1k,
        "measurements_per_1k": vp.measurements_per_1k,
        "vague_words": vague_total,
        "vague_per_1k": round(vague_total / words * 1000, 2),
        "vague_top": sorted(vague.items(), key=lambda kv: -kv[1])[:5],
    }
    findings, suggestions = [], []
    if vp.specificity_per_1k < 15:
        findings.append(
            f"low specificity ({vp.specificity_per_1k}/1k): the scene "
            "runs on generic nouns — name the object, the number, the place")
        suggestions.append("swap one generic noun per paragraph for a "
                           "named, measured, or branded one")
    if vague_total:
        findings.append(f"{vague_total} vague words "
                         f"({metrics['vague_per_1k']}/1k)")
        suggestions.append("replace vague words with the concrete thing "
                           "they stand in for")
    score = 0.5
    score += min(0.3, vp.specificity_per_1k / 60.0)
    score -= min(0.3, metrics["vague_per_1k"] / 40.0)
    return _report("specificity", score, metrics, findings, suggestions)


def pass_restraint(text: str) -> dict[str, Any]:
    """Cut 10–15%, darlings pass, readability must improve.

    The trim is mechanical and measurable: filler phrases + -ly
    adverbs are removed from a COPY, word count and readability are
    re-measured on the copy, and the report states whether readability
    improved.  Nothing in the original is rewritten by this pass.
    """
    words = _words(text)
    n = len(words)
    low = text.lower()
    filler_hits = sum(low.count(p) for p in _FILLER_PHRASES)
    adverbs = [w for w in words
               if w.lower().endswith("ly") and len(w) > 4
               and w.lower() not in {"only", "early", "really", "finally",
                                     "family", "likely", "nearly"}]
    cut_candidates = filler_hits + len(adverbs)
    # darlings: the 3 longest sentences + the 3 most adverb-dense
    sents = split_sentences(text)
    by_len = sorted(sents, key=lambda s: -len(_words(s)))[:3]
    darlings = [s[:160] for s in by_len]
    trimmed = text
    for phrase in _FILLER_PHRASES:
        trimmed = re.sub(re.escape(phrase), "", trimmed, flags=re.IGNORECASE)
    trimmed_words = len(_words(trimmed))
    cut_pct = (n - trimmed_words) / max(1, n) * 100
    before = readability(text)
    after = readability(trimmed)
    metrics = {
        "words": n,
        "cut_candidates": cut_candidates,
        "cut_target_words": round(n * 0.10),
        "cut_target_words_max": round(n * 0.15),
        "cut_target_pct": "10-15%",
        "trimmed_words": trimmed_words,
        "mechanical_cut_pct": round(cut_pct, 1),
        "readability_before": before,
        "readability_after_trim": after,
        "readability_improved": after >= before,
        "darlings": darlings,
    }
    findings, suggestions = [], []
    if cut_pct < 10:
        findings.append(
            f"mechanical trim only removes {cut_pct:.1f}% — the 10–15% "
            "cut target needs real line edits, not just filler removal")
        suggestions.append("cut one sentence per paragraph that repeats "
                           "what another already says")
    if after < before:
        findings.append(
            f"readability would DROP after the mechanical trim "
            f"({before} → {after}) — the trim is cutting load-bearing "
            "words; cut sentences, not glue")
    else:
        findings.append(
            f"readability improves or holds after trim "
            f"({before} → {after})")
    suggestions.append("darlings pass: read the 3 longest sentences aloud; "
                       "cut the one you love most if the scene survives it")
    score = 0.5
    score += 0.2 if after >= before else -0.2
    score += min(0.2, cut_pct / 50.0)
    score -= 0.1 if n > 0 and len(by_len) and len(_words(by_len[0])) > 45 else 0.0
    return _report("restraint", score, metrics, findings, suggestions)


def pass_subtext(text: str) -> dict[str, Any]:
    """Dialogue that states its emotion vs dialogue that implies it."""
    low = text.lower()
    quotes = re.findall(r'"[^"]{2,}"|“[^”]{2,}”', text)
    on_nose = 0
    on_nose_lines: list[str] = []
    for q in quotes:
        ql = q.lower()
        if any(re.search(p, ql) for p in _ON_THE_NOSE):
            on_nose += 1
            on_nose_lines.append(q[:120])
    # subtext signals: trailing off, evasion, contradiction, questions
    # answered with questions, silence beats
    subtext_signals = sum(low.count(m) for m in (
        "…", "...", "never mind", "forget it", "it's nothing",
        "don't worry about it", "maybe", "i don't know",
    ))
    metrics = {
        "dialogue_lines": len(quotes),
        "on_the_nose_lines": on_nose,
        "on_the_nose_pct": round(on_nose / max(1, len(quotes)) * 100, 1),
        "subtext_signals": subtext_signals,
        "on_the_nose_examples": on_nose_lines[:3],
    }
    findings, suggestions = [], []
    if on_nose:
        findings.append(f"{on_nose} dialogue line(s) state the emotion "
                        "outright instead of implying it")
        suggestions.append("let the line imply the feeling; give the "
                           "emotion to a physical beat, not a declaration")
    if len(quotes) and subtext_signals == 0:
        findings.append("no subtext signals — everyone says exactly what "
                        "they mean, which no one does")
        suggestions.append("let one character answer a question with a "
                           "question, or not answer at all")
    score = 0.7
    score -= min(0.4, metrics["on_the_nose_pct"] / 100.0)
    score += 0.1 if subtext_signals else -0.1
    return _report("subtext", score, metrics, findings, suggestions)


def pass_rhythm(text: str) -> dict[str, Any]:
    """Sentence-length distribution + short/long runs."""
    sents = split_sentences(text)
    lens = [len(_words(s)) for s in sents]
    mean = statistics.fmean(lens) if lens else 0.0
    std = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    cv = std / mean if mean else 0.0
    # runs: 3+ consecutive short (<8) or long (>25) sentences
    runs: list[str] = []
    run, run_kind = 1, None
    for x in lens:
        kind = "short" if x < 8 else ("long" if x > 25 else "mid")
        if kind == run_kind and kind != "mid":
            run += 1
        else:
            if run >= 3 and run_kind:
                runs.append(f"{run}× {run_kind} in a row")
            run, run_kind = 1, kind
    if run >= 3 and run_kind:
        runs.append(f"{run}× {run_kind} in a row")
    metrics = {
        "sentence_len_mean": round(mean, 2),
        "sentence_len_std": round(std, 2),
        "sentence_len_cv": round(cv, 3),
        "pct_under_8": round(sum(1 for x in lens if x < 8) / max(1, len(lens)), 3),
        "pct_over_25": round(sum(1 for x in lens if x > 25) / max(1, len(lens)), 3),
        "longest_sentence": max(lens) if lens else 0,
        "runs": runs,
    }
    findings, suggestions = [], []
    if cv < 0.45 and len(lens) >= 8:
        findings.append(f"monotone rhythm (CV {cv:.2f}) — sentence "
                        "length never breathes")
        suggestions.append("follow every 25+ word sentence with one under 8")
    if runs:
        findings.append("rhythm runs: " + "; ".join(runs))
        suggestions.append("break a run with a sentence of the opposite length")
    score = 0.5 + min(0.3, cv) - (0.15 if runs else 0.0)
    return _report("rhythm", score, metrics, findings, suggestions)


def pass_sensory(text: str) -> dict[str, Any]:
    """Sensory density by sense per 1k; non-visual coverage per scene."""
    vp = fingerprint(text)
    paras = split_paragraphs(text)
    per_1k = 1000.0 / max(1, vp.words)
    non_visual = {"sound", "smell", "touch", "taste"}
    scenes_missing: list[int] = []
    for i, para in enumerate(paras, 1):
        low = para.lower()
        if not any(w in low
                   for sense in non_visual
                   for w in SENSORY_WORDS[sense]):
            scenes_missing.append(i)
    metrics = {
        "sensory_per_1k": dict(vp.sensory),
        "non_visual_per_1k": round(
            sum(vp.sensory.get(s, 0.0) for s in non_visual), 2),
        "visual_per_1k": round(vp.sensory.get("sight", 0.0), 2),
        "scenes": len(paras),
        "scenes_missing_non_visual": scenes_missing,
    }
    findings, suggestions = [], []
    if metrics["non_visual_per_1k"] < 3:
        findings.append(
            f"non-visual senses nearly absent "
            f"({metrics['non_visual_per_1k']}/1k) — the scene is all eyes")
        suggestions.append("give one paragraph a smell, a texture, or a "
                           "sound the camera could never catch")
    if metrics["visual_per_1k"] > 0 and \
            metrics["non_visual_per_1k"] == 0 and metrics["visual_per_1k"] > 5:
        findings.append("sight-only scene — every sensory hit is visual")
    if scenes_missing:
        findings.append(f"{len(scenes_missing)}/{len(paras)} paragraphs "
                        "carry no non-visual sense")
    score = 0.5 + min(0.3, metrics["non_visual_per_1k"] / 15.0)
    score -= 0.15 if metrics["non_visual_per_1k"] == 0 else 0.0
    return _report("sensory", score, metrics, findings, suggestions)


def pass_interiority(text: str) -> dict[str, Any]:
    """One unspoken thought per scene; scenes missing one are listed."""
    paras = split_paragraphs(text)
    vp = fingerprint(text)
    markers = (
        "thought", "wondered", "realized", "realised", "knew",
        "told herself", "told himself", "of course", "after all",
        "somehow", "maybe", "perhaps",
    )
    missing: list[int] = []
    for i, para in enumerate(paras, 1):
        low = para.lower()
        quoted = re.sub(r'"[^"]*"|“[^”]*”', " ", low)
        has_thought = any(m in quoted for m in markers) or "?" in quoted
        if not has_thought:
            missing.append(i)
    metrics = {
        "interiority_per_1k": vp.interiority_per_1k,
        "scenes": len(paras),
        "scenes_missing_interiority": missing,
        "scenes_with_interiority": len(paras) - len(missing),
    }
    findings, suggestions = [], []
    if missing:
        findings.append(f"{len(missing)}/{len(paras)} paragraphs carry no "
                        "unspoken thought")
        suggestions.append("one unspoken thought per scene — what the "
                           "character does NOT say, in their own head")
    if vp.interiority_per_1k == 0:
        findings.append("no interiority markers at all — the POV head is empty")
    score = 0.6
    score += 0.2 * (metrics["scenes_with_interiority"] / max(1, len(paras)))
    score -= 0.2 if vp.interiority_per_1k == 0 else 0.0
    return _report("interiority", score, metrics, findings, suggestions)


def pass_de_filter(text: str) -> dict[str, Any]:
    """Filtering verbs per 1k — target ~0."""
    vp = fingerprint(text)
    low = text.lower()
    hits = {v: len(re.findall(r"\b" + re.escape(v) + r"\b", low))
            for v in FILTERING_VERBS if " " not in v}
    hits = {v: c for v, c in hits.items() if c}
    metrics = {
        "filtering_per_1k": vp.filtering_per_1k,
        "filter_hits": hits,
        "filter_total": sum(hits.values()),
        "target_per_1k": 0.0,
    }
    findings, suggestions = [], []
    if vp.filtering_per_1k > 4:
        findings.append(
            f"filtering verbs at {vp.filtering_per_1k}/1k — perception "
            "kept behind a verb instead of rendered")
        suggestions.append("delete the filter verb; keep the thing seen. "
                           "'She saw the door stand open' → 'The door "
                           "stood open.'")
    elif vp.filtering_per_1k > 0:
        findings.append(f"{metrics['filter_total']} filtering verb(s) "
                        f"({vp.filtering_per_1k}/1k) — near the ~0 target")
    score = 0.9 - min(0.6, vp.filtering_per_1k / 20.0)
    return _report("de_filter", score, metrics, findings, suggestions)


def pass_ending(text: str) -> dict[str, Any]:
    """Delete-the-moralising-last-line test; end on image / action."""
    paras = split_paragraphs(text)
    last = paras[-1] if paras else ""
    low = last.lower()
    moralising = [p for p in _MORALISING_END if p in low]
    rhetorical = last.strip().endswith("?")
    hope_word = bool(re.search(r"\bhope\b", low))
    # image/action ending: concrete noun or motion verb in the last sentence
    image_markers = ("hand", "door", "light", "road", "face", "breath",
                     "step", "walked", "ran", "stood", "opened", "closed",
                     "turned", "looked", "stone", "water", "fire", "night")
    ends_image = any(w in low for w in image_markers)
    metrics = {
        "last_paragraph": last[-300:],
        "moralising_phrases": moralising,
        "ends_rhetorical_question": rhetorical,
        "ends_on_hope": hope_word,
        "ends_on_image_or_action": ends_image,
        "delete_last_line_better": bool(moralising or rhetorical or hope_word),
    }
    findings, suggestions = [], []
    if moralising:
        findings.append("last line moralises: " + ", ".join(moralising))
        suggestions.append("delete the moralising last line — end on the "
                           "image or action before it")
    if rhetorical:
        findings.append("chapter ends on a rhetorical question")
        suggestions.append("end on an image or an action, not a question "
                           "the reader already knows the answer to")
    if hope_word:
        findings.append("chapter closes on the word 'hope'")
        suggestions.append("cut 'hope'; let the last image carry it")
    if not ends_image and not (moralising or rhetorical):
        findings.append("last line is neither image, action, nor theme — "
                        "it just stops")
        suggestions.append("rewrite the last line as one concrete image "
                           "or one action taken")
    score = 0.7
    score -= 0.25 if moralising else 0.0
    score -= 0.15 if rhetorical else 0.0
    score -= 0.1 if hope_word else 0.0
    score += 0.1 if ends_image else 0.0
    return _report("ending", score, metrics, findings, suggestions)


def run_craft_passes(text: str) -> dict[str, Any]:
    """Run all eight craft passes. Reports, never blocks."""
    passes = {
        "specificity": pass_specificity(text),
        "restraint": pass_restraint(text),
        "subtext": pass_subtext(text),
        "rhythm": pass_rhythm(text),
        "sensory": pass_sensory(text),
        "interiority": pass_interiority(text),
        "de_filter": pass_de_filter(text),
        "ending": pass_ending(text),
    }
    scores = [p["score"] for p in passes.values()]
    return {
        "passes": passes,
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "weakest": sorted(passes, key=lambda k: passes[k]["score"])[:2],
        "blocked": False,
        "note": "Craft passes report; they never block. The owner overrules.",
    }
