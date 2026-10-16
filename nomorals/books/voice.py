"""Voice claims, measured: style vector per story/author, similarity
per chapter *reported*, never assumed.

A style vector is a small deterministic feature map over a passage:
sentence length, dialogue ratio, word length, type-token ratio,
punctuation rates, pronoun ratios, paragraph length. Cosine over the
shared feature set gives the similarity number the continuity engine
reports (``continuity.drift``) and the per-chapter table here.

Two voices can share a 0.91 and still differ in content — similarity is
a *style* claim, and the report says so. Empty text yields the zero
vector and similarity 0.0, never a fabricated 'same voice'.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .prose import split_paragraphs, split_sentences

__all__ = ["StyleVector", "vectorize", "cosine_features", "voice_similarity",
           "VoiceClaim", "FEATURES"]

FEATURES = (
    "sentence_len_mean", "sentence_len_std", "dialogue_ratio",
    "avg_word_len", "type_token_ratio", "comma_rate", "exclam_rate",
    "question_rate", "first_person_rate", "third_person_rate",
    "paragraph_len_mean", "quoted_ratio", "ellipsis_rate", "dash_rate",
    "short_sentence_ratio",
)
_FIRST = frozenset("i me my mine we us our ours".split())
_THIRD = frozenset("he him his she her hers they them their theirs it its".split())


@dataclass
class StyleVector:
    """Measured style features for one passage (or a whole work)."""

    features: dict[str, float]
    words: int = 0
    sentences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"features": {k: round(v, 4) for k, v in
                             self.features.items()},
                "words": self.words, "sentences": self.sentences}


def vectorize(text: str) -> StyleVector:
    """Deterministic style features for ``text``."""
    text = text or ""
    words = re.findall(r"[A-Za-z']+", text)
    n_words = len(words)
    sentences = split_sentences(text)
    paras = split_paragraphs(text)
    sent_lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]
    mean = (sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0
    var = (sum((x - mean) ** 2 for x in sent_lens) / len(sent_lens)
           ) if sent_lens else 0.0
    low_words = [w.lower() for w in words]
    quoted = sum(len(m) for m in re.findall(r'"[^"]*"|“[^”]*”', text))
    feats: dict[str, float] = {
        "sentence_len_mean": mean,
        "sentence_len_std": math.sqrt(var),
        "dialogue_ratio": quoted / max(1, len(text)),
        "avg_word_len": (sum(len(w) for w in words) / n_words
                         ) if n_words else 0.0,
        "type_token_ratio": (len(set(low_words)) / n_words
                             ) if n_words else 0.0,
        "comma_rate": text.count(",") / max(1, n_words),
        "exclam_rate": text.count("!") / max(1, len(sentences)),
        "question_rate": text.count("?") / max(1, len(sentences)),
        "first_person_rate": sum(1 for w in low_words if w in _FIRST
                                 ) / max(1, n_words),
        "third_person_rate": sum(1 for w in low_words if w in _THIRD
                                 ) / max(1, n_words),
        "paragraph_len_mean": (sum(len(re.findall(r"[A-Za-z']+", p))
                                   for p in paras) / max(1, len(paras))),
        "quoted_ratio": quoted / max(1, len(text)),
        "ellipsis_rate": text.count("...") / max(1, len(sentences)),
        "dash_rate": (text.count("—") + text.count("--")
                      ) / max(1, len(sentences)),
        "short_sentence_ratio": (sum(1 for x in sent_lens if x <= 5)
                                 / max(1, len(sent_lens))),
    }
    return StyleVector(features=feats, words=n_words,
                       sentences=len(sentences))


def cosine_features(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine over the shared FEATURES set. 0.0 when either is empty."""
    keys = [k for k in FEATURES if a.get(k) is not None
            and b.get(k) is not None]
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(a[k] ** 2 for k in keys))
    nb = math.sqrt(sum(b[k] ** 2 for k in keys))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def voice_similarity(text_a: str, text_b: str) -> float:
    """Style similarity between two passages (0..1)."""
    return round(cosine_features(vectorize(text_a).features,
                                 vectorize(text_b).features), 4)


class VoiceClaim:
    """Build a reference voice from early chapters; score every chapter.

    ``build_reference`` averages the feature vectors of the given
    chapters (the claim 'this is the voice' is the mean of measured
    chapters, stated as such). ``per_chapter`` reports each chapter's
    similarity to that reference — drift is a number in a table, not a
    verdict.
    """

    def __init__(self, *, drift_threshold: float = 0.75) -> None:
        self.drift_threshold = drift_threshold

    def build_reference(self, chapters: Sequence[tuple[int, str]]
                        ) -> StyleVector:
        """Mean style vector over ``chapters`` (chapter_no, text)."""
        vecs = [vectorize(t) for _, t in chapters if (t or "").strip()]
        if not vecs:
            raise ValueError("build_reference needs at least one "
                             "non-empty chapter")
        feats = {k: sum(v.features[k] for v in vecs) / len(vecs)
                 for k in FEATURES}
        return StyleVector(features=feats,
                           words=sum(v.words for v in vecs),
                           sentences=sum(v.sentences for v in vecs))

    def per_chapter(self, chapters: Sequence[tuple[int, str]],
                    reference: StyleVector) -> list[dict[str, Any]]:
        out = []
        for number, text in chapters:
            sim = cosine_features(vectorize(text).features,
                                  reference.features)
            out.append({"chapter": number, "similarity": round(sim, 4),
                        "drifted": sim < self.drift_threshold})
        return out

    def report(self, chapters: Sequence[tuple[int, str]],
               *, reference_chapters: Sequence[tuple[int, str]] | None = None
               ) -> dict[str, Any]:
        """Reference from ``reference_chapters`` (default: first 3), then
        a per-chapter similarity table over all chapters."""
        ref_src = list(reference_chapters) if reference_chapters else \
            list(chapters)[:3]
        reference = self.build_reference(ref_src)
        table = self.per_chapter(chapters, reference)
        sims = [row["similarity"] for row in table]
        return {
            "reference_chapters": [n for n, _ in ref_src],
            "reference": reference.to_dict(),
            "per_chapter": table,
            "mean_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0,
            "min_similarity": round(min(sims), 4) if sims else 0.0,
            "drifted_chapters": [r["chapter"] for r in table
                                 if r["drifted"]],
            "threshold": self.drift_threshold,
            "note": "similarity is a STYLE claim (sentence shape, "
                    "dialogue, pronouns), not a content claim",
        }
