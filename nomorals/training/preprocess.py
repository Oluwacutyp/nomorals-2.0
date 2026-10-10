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

__all__ = [
    "CleanStats", "clean_text", "simhash", "hamming", "dedupe",
    "quality_filter", "refined_quality_signals", "check_leakage", "split",
    "prepare",
]

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


def _example_text(example: Example) -> str:
    return clean_text(example.prompt + "\n" + example.completion)


def dedupe(
    examples: Sequence[Example],
    *,
    threshold: int = 3,
    exact_first: bool = True,
) -> list[Example]:
    """Drop duplicates: exact matches first (cheap), then near-duplicates.

    The RedPajama pipeline order — a byte-identical row is the most common
    duplicate in real corpora and a sha256 pre-pass is O(n) versus the
    O(n²) simhash scan, so exact-first is both faster and stricter.
    ``threshold`` is max differing bits out of 64 for the fuzzy pass.
    """
    exact_dropped = 0
    candidates = list(examples)
    if exact_first:
        seen_exact: set[str] = set()
        unique: list[Example] = []
        for example in candidates:
            digest = hashlib.sha256(_example_text(example).encode("utf-8")).hexdigest()
            if digest in seen_exact:
                exact_dropped += 1
                continue
            seen_exact.add(digest)
            unique.append(example)
        candidates = unique
    kept: list[Example] = []
    seen: list[int] = []
    for example in candidates:
        fingerprint = simhash(_example_text(example))
        if any(hamming(fingerprint, other) <= threshold for other in seen):
            continue
        seen.append(fingerprint)
        kept.append(example)
    _log.info("deduped %d -> %d examples (%d exact, %d fuzzy dropped)",
              len(examples), len(kept), exact_dropped,
              len(candidates) - len(kept))
    return kept


def refined_quality_signals(text: str) -> dict[str, float]:
    """RefinedWeb/C4-style quality signals for one text.

    Returns the raw signals (callers decide thresholds): alphabetic ratio,
    terminal-punctuation presence, symbol-to-word ratio, mean line length,
    and word count.  Degenerate crawl rows fail these long before a human
    would notice.
    """
    words = re.findall(r"\w+", text.lower())
    chars = len(text)
    alpha = sum(1 for ch in text if ch.isalpha())
    lines = [line for line in text.split("\n") if line.strip()]
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return {
        "alpha_ratio": alpha / max(1, chars),
        "word_count": float(len(words)),
        "symbol_ratio": symbols / max(1, len(words)),
        "mean_line_len": sum(len(line) for line in lines) / max(1, len(lines)),
        "ends_with_terminal": 1.0 if text.rstrip().endswith((".", "?", "!")) else 0.0,
    }


def quality_filter(
    examples: Sequence[Example],
    *,
    min_prompt_chars: int = 8,
    min_completion_chars: int = 4,
    max_chars: int = 32_000,
    max_repetition: float = 0.55,
    strict: bool = False,
) -> list[Example]:
    """Drop examples that would teach the model nothing, or teach it to repeat.

    ``strict=True`` adds the RefinedWeb-style signals: the completion must
    be mostly alphabetic, end with terminal punctuation, and not be
    symbol-dense.  Strict mode is for web-mined rows; the default stays
    lenient for curated/chat rows.
    """
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
        if strict:
            signals = refined_quality_signals(completion)
            if signals["alpha_ratio"] < 0.5:
                continue
            if signals["symbol_ratio"] > 0.4:
                continue
            if signals["ends_with_terminal"] < 1.0 and signals["word_count"] > 12:
                continue
        kept.append(example)
    _log.info("quality filter kept %d/%d (strict=%s)", len(kept), len(examples), strict)
    return kept


def check_leakage(
    train: Sequence[Example],
    evaluation: Sequence[Example],
    *,
    threshold: int = 3,
) -> dict[str, Any]:
    """Find eval rows that near-duplicate a train row (SlimPajama lesson).

    A split drawn *after* dedupe can still leak when the dedupe ran per-file;
    eval must be deduplicated against train GLOBALLY.  Returns the count and
    the offending eval indices — the caller decides whether to drop them.
    """
    train_prints = [simhash(_example_text(e)) for e in train]
    leaked: list[int] = []
    for index, example in enumerate(evaluation):
        fingerprint = simhash(_example_text(example))
        if any(hamming(fingerprint, prior) <= threshold for prior in train_prints):
            leaked.append(index)
    return {
        "eval_rows": len(evaluation),
        "leaked": len(leaked),
        "leaked_indices": leaked,
        "leak_rate": round(len(leaked) / max(1, len(evaluation)), 4),
    }


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
