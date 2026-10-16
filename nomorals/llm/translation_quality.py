"""Translation quality bridge: translation + back-translation + judge.

A bare translation is unverifiable — the caller cannot tell a good
rendering from a fluent mistranslation.  The bridge closes that gap
with two independent checks, both cheap and model-agnostic:

1. BACK-TRANSLATION — translate target -> source and measure overlap
   with the original (token F1).  A faithful translation round-trips;
   a mistranslation loses content on the way back.
2. JUDGE — the ``llm/adjudicate.py`` judge (or any injected
   ``judge_fn``) scores the translation for meaning preservation.

The bridge is translator-agnostic: ``translate_fn`` is injected, so
any engine (local, cloud, rule-based) can sit behind it.  No new
provider is added here — this wraps whatever translator the caller
already uses.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "TranslationResult",
    "back_translation_f1",
    "token_f1",
    "translate_with_quality",
]

_TOKEN_RE = re.compile(r"[^\s]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def token_f1(reference: str, hypothesis: str) -> float:
    """Token-level F1 between two strings (multiset overlap, 0..1)."""
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    from collections import Counter

    ref_counts, hyp_counts = Counter(ref), Counter(hyp)
    overlap = sum((ref_counts & hyp_counts).values())
    precision = overlap / len(hyp)
    recall = overlap / len(ref)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def back_translation_f1(original: str, back_translated: str) -> float:
    """Round-trip fidelity: F1(original, back-translated)."""
    return token_f1(original, back_translated)


@dataclass
class TranslationResult:
    """One bridged translation with both quality signals."""

    source_text: str
    translated: str
    source_lang: str = ""
    target_lang: str = ""
    back_translated: str = ""
    back_translation_f1: float = 0.0
    judge_score: float | None = None
    judge_reason: str = ""
    quality: float = 0.0
    passed: bool = False
    threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text, "translated": self.translated,
            "source_lang": self.source_lang, "target_lang": self.target_lang,
            "back_translated": self.back_translated,
            "back_translation_f1": round(self.back_translation_f1, 4),
            "judge_score": self.judge_score,
            "judge_reason": self.judge_reason,
            "quality": round(self.quality, 4),
            "passed": self.passed, "threshold": self.threshold,
        }


def translate_with_quality(
    text: str,
    *,
    source_lang: str,
    target_lang: str,
    translate_fn: Callable[[str, str, str], str],
    judge_fn: Callable[[str, str], tuple[float, str]] | None = None,
    threshold: float = 0.5,
) -> TranslationResult:
    """Translate with back-translation + optional judge verification.

    ``translate_fn(text, source_lang, target_lang) -> str`` does the
    actual translation (any engine).  Quality = mean of the
    back-translation F1 and the judge score when a judge is supplied,
    else the F1 alone.  ``passed`` is quality >= ``threshold``.  A
    failed round-trip (empty back-translation) scores 0 — never a
    silent pass.
    """
    text = (text or "").strip()
    if not text:
        return TranslationResult(source_text="", translated="",
                                 source_lang=source_lang,
                                 target_lang=target_lang, passed=False,
                                 threshold=threshold,
                                 judge_reason="empty source text")
    try:
        translated = str(translate_fn(text, source_lang, target_lang) or "").strip()
    except Exception as exc:  # noqa: BLE001 — translator failure is a result
        return TranslationResult(source_text=text, translated="",
                                 source_lang=source_lang,
                                 target_lang=target_lang, passed=False,
                                 threshold=threshold,
                                 judge_reason=f"translate failed: {exc}")
    if not translated:
        return TranslationResult(source_text=text, translated="",
                                 source_lang=source_lang,
                                 target_lang=target_lang, passed=False,
                                 threshold=threshold,
                                 judge_reason="translator returned empty")
    try:
        back = str(translate_fn(translated, target_lang, source_lang)
                   or "").strip()
    except Exception as exc:  # noqa: BLE001
        back = ""
        _log.debug("back-translation failed: %s", exc)
    f1 = back_translation_f1(text, back) if back else 0.0
    judge_score: float | None = None
    judge_reason = ""
    if judge_fn is not None:
        try:
            judge_score, judge_reason = judge_fn(text, translated)
            judge_score = min(1.0, max(0.0, float(judge_score)))
        except Exception as exc:  # noqa: BLE001
            judge_reason = f"judge failed: {exc}"
            judge_score = None
    quality = (f1 + judge_score) / 2.0 if judge_score is not None else f1
    return TranslationResult(
        source_text=text, translated=translated,
        source_lang=source_lang, target_lang=target_lang,
        back_translated=back, back_translation_f1=f1,
        judge_score=judge_score, judge_reason=judge_reason,
        quality=quality, passed=bool(quality >= threshold),
        threshold=threshold)


def judge_with_llm_judge(
    ask: Any,
    *,
    source_lang: str = "",
    target_lang: str = "",
) -> Callable[[str, str], tuple[float, str]]:
    """Build a ``judge_fn`` on the ``llm/adjudicate.py`` judge.

    The judge picks between the translation and a deliberately
    degraded variant (word order shuffled); a judge that cannot tell
    them apart scores 0.5.  This keeps the bridge honest without a
    separate scoring model: the judge's own preference is the signal.
    """
    from ..llm.adjudicate import Judge

    def _judge(original: str, translated: str) -> tuple[float, str]:
        words = translated.split()
        degraded = " ".join(reversed(words)) if len(words) > 1 else translated + " ???"
        verdict = Judge(ask).pairwise(
            f"Which translation best preserves the meaning of this "
            f"{source_lang or 'source'} text? Original: {original}",
            translated, degraded)
        if not verdict.ok:
            return 0.5, f"judge failed: {verdict.error}"
        if verdict.winner == 0:
            return (0.95 if verdict.consistent else 0.7,
                    "judge preferred the translation"
                    + ("" if verdict.consistent else " (split verdict)"))
        if verdict.winner == 1:
            return 0.2, "judge preferred the degraded variant — suspect translation"
        return 0.5, "judge tied — no preference"

    return _judge
