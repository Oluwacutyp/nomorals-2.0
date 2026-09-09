"""Corpus cleaning, deduplication, and splitting.

The order matters. Dedup before filtering, so near-identical rows do not each
consume a quality slot. Split last, so the train/eval boundary is drawn on the
cleaned set and eval cannot leak a training example.

Dedup is by simhash rather than exact hash on purpose: distillation corpora are
full of rows that differ by one trailing newline, and exact-hash dedup keeps all
of them.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..core.logging_setup import get_logger
from .dataset import Example

__all__ = ["CleanStats", "clean_text", "simhash", "dedupe", "quality_filter", "split"]

_log = get_logger(__name__)

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str, *, min_chars: int = 1) -> str:
    """Normalize whitespace and strip control characters."""
    if not text:
        return ""
    text = _CONTROL.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def simhash(text: str, *, bits: int = 64) -> int:
    """A 64-bit locality-sensitive hash. Similar texts differ in few bits."""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        for index in range(bits):
            vector[index] += 1 if (digest >> index) & 1 else -1
    fingerprint = 0
    for index in range(bits):
        if vector[index] > 0:
            fingerprint |= 1 << index
    return fingerprint


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def dedupe(examples: Sequence[Example], *, threshold: int = 3) -> list[Example]:
    """Drop near-duplicates. ``threshold`` is max differing bits out of 64."""
    kept: list[Example] = []
    seen: list[int] = []
    for example in examples:
        text = example.prompt + "\n" + example.completion
        fingerprint = simhash(clean_text(text))
        if any(hamming(fingerprint, other) <= threshold for other in seen):
            continue
        seen.append(fingerprint)
        kept.append(example)
    _log.info("deduped %d -> %d examples", len(examples), len(kept))
    return kept


def quality_filter(
    examples: Sequence[Example],
    *,
    min_prompt_chars: int = 8,
    min_completion_chars: int = 4,
    max_chars: int = 32_000,
    max_repetition: float = 0.55,
) -> list[Example]:
    """Drop examples that would teach the model nothing, or teach it to repeat."""
    kept: list[Example] = []
    for example in examples:
        prompt = clean_text(example.prompt)
        completion = clean_text(example.completion)
        if len(prompt) < min_prompt_chars or len(completion) < min_completion_chars:
            continue
        if len(prompt) + len(completion) > max_chars:
            continue
        if _repetition_ratio(completion) > max_repetition:
            continue
        kept.append(example)
    _log.info("quality filter kept %d/%d", len(kept), len(examples))
    return kept


def _repetition_ratio(text: str) -> float:
    """Fraction of the text made of the most repeated token. Catches degenerate rows."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 8:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return max(counts.values()) / len(tokens)


@dataclass
class CleanStats:
    input: int = 0
    cleaned: int = 0
    deduped: int = 0
    filtered: int = 0
    train: int = 0
    eval: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input, "cleaned": self.cleaned, "deduped": self.deduped,
            "filtered": self.filtered, "train": self.train, "eval": self.eval,
        }


def split(
    examples: Sequence[Example], *, eval_fraction: float = 0.1, seed: int = 1234
) -> tuple[list[Example], list[Example]]:
    """Deterministic shuffled split. Eval must not leak training examples."""
    if not examples:
        return [], []
    ordered = list(examples)
    random.Random(seed).shuffle(ordered)
    cut = max(1, int(len(ordered) * eval_fraction)) if len(ordered) > 1 else 0
    if cut == 0:
        return ordered, []
    return ordered[cut:], ordered[:cut]


def prepare(
    examples: Iterable[Example],
    *,
    dedupe_threshold: int = 3,
    eval_fraction: float = 0.1,
    seed: int = 1234,
    **filter_kwargs: Any,
) -> tuple[list[Example], list[Example], CleanStats]:
    """Full pipeline: clean -> dedupe -> filter -> split."""
    materialized = list(examples)
    stats = CleanStats(input=len(materialized))

    for example in materialized:
        for turn in example.turns:
            turn.content = clean_text(turn.content)
    materialized = [e for e in materialized if any(t.content for t in e.turns)]
    stats.cleaned = len(materialized)

    deduped = dedupe(materialized, threshold=dedupe_threshold)
    stats.deduped = len(deduped)

    filtered = quality_filter(deduped, **filter_kwargs)
    stats.filtered = len(filtered)

    train, evaluation = split(filtered, eval_fraction=eval_fraction, seed=seed)
    stats.train, stats.eval = len(train), len(evaluation)
    return train, evaluation, stats
