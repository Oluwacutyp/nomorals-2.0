"""Text primitives: tokenization, chunking, normalization, dedup.

Shared by memory, training-data curation, retrieval, and the context builder.
Everything here is pure stdlib and deterministic — the same input always yields
the same output, which is what makes dataset builds reproducible.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

__all__ = [
    "Chunk",
    "SimHash",
    "approx_token_count",
    "chunk_text",
    "compact_whitespace",
    "dedupe_by_simhash",
    "detokenize_bytes",
    "levenshtein",
    "ngrams",
    "normalize_text",
    "sentences",
    "shingle",
    "similarity",
    "slugify",
    "tokenize_bpe_bytes",
    "truncate",
    "truncate_to_tokens",
    "word_frequencies",
]

# ── Normalization ──────────────────────────────────────────────────────────────

_WS_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str, *, fold_unicode: bool = True, lower: bool = False) -> str:
    """Canonical form used before hashing, dedup, and comparison."""
    if not text:
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    if fold_unicode:
        out = unicodedata.normalize("NFKC", out)
    out = _WS_RE.sub(" ", out)
    out = _BLANK_LINES_RE.sub("\n\n", out)
    out = out.strip()
    return out.lower() if lower else out


def compact_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    """Trim ``text`` to ``limit`` characters, appending ``suffix`` when cut.

    Canonical char-based truncation for the whole system: text at or under
    the limit is returned unchanged, longer text is cut at exactly
    ``limit`` chars plus the suffix.  (For token-budget trimming use
    :func:`truncate_to_tokens`.)
    """
    return text if len(text) <= limit else text[:limit] + suffix


# ── Tokenization ───────────────────────────────────────────────────────────────

# A pragmatic tokenizer for budgeting. Real token counts need the model's actual
# tokenizer; this approximation is accurate to a few percent on English prose and
# never needs a 2 MB vocab file to be correct about "will this fit in context".
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def approx_token_count(text: str) -> int:
    """Estimate BPE token count. ~1.3 tokens per whitespace-delimited word."""
    if not text:
        return 0
    words = text.split()
    if not words:
        return 0
    long_words = sum(max(0, (len(w) // 6)) for w in words)
    return len(words) + long_words


def truncate_to_tokens(text: str, max_tokens: int, *, suffix: str = "…") -> str:
    """Trim ``text`` so its estimated token count fits ``max_tokens``."""
    if max_tokens <= 0:
        return ""
    if approx_token_count(text) <= max_tokens:
        return text
    # Binary search on character count; the estimator is monotonic enough.
    low, high = 0, len(text)
    budget = max(0, max_tokens - approx_token_count(suffix))
    while low < high:
        mid = (low + high + 1) // 2
        if approx_token_count(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + suffix


def tokenize_bpe_bytes(text: str, chunk: int = 3) -> list[bytes]:
    """Fixed-width byte chunks: the fallback tokenizer when ``transformers`` is absent.

    Not a learned BPE — a deterministic byte n-gram split. It returns ``bytes``
    rather than ``str`` on purpose: a fixed byte width can cut a multi-byte UTF-8
    sequence in half, and decoding such a fragment lossily would make the token
    stream impossible to invert. Join the pieces with :func:`detokenize_bytes` to
    recover the original text exactly.

    Stable across runs and platforms, which is the property the training pipeline
    actually needs from a fallback tokenizer.
    """
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    raw = text.encode("utf-8")
    return [raw[i : i + chunk] for i in range(0, len(raw), chunk)]


def detokenize_bytes(pieces: Iterable[bytes]) -> str:
    """Inverse of :func:`tokenize_bpe_bytes`."""
    return b"".join(pieces).decode("utf-8")


def ngrams(tokens: Sequence[str], n: int) -> Iterator[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def shingle(text: str, k: int = 5) -> set[str]:
    """Set of k-word shingles, used for Jaccard near-duplicate detection."""
    words = normalize_text(text, lower=True).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def levenshtein(a: str, b: str) -> int:
    """Edit distance, O(min(len(a), len(b))) memory."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """Normalized similarity in [0, 1] based on edit distance."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


def word_frequencies(text: str, *, stopwords: bool = True) -> Counter[str]:
    counts = Counter(w for w in normalize_text(text, lower=True).split() if len(w) > 1)
    if stopwords:
        for word in _STOPWORDS:
            counts.pop(word, None)
    return counts


_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
below between both but by can cannot could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
you your yours yourself yourselves""".split()
)


# ── Sentence splitting ─────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"'“”‘’(\[]?[A-Z0-9])")
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig", "no"}
)


def sentences(text: str) -> list[str]:
    """Split prose into sentences, respecting common abbreviations."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    raw = _SENTENCE_END.split(normalized)
    out: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if out:
            previous = out[-1]
            tail = previous.rstrip(".!?…").split()[-1].lower() if previous.split() else ""
            if tail in _ABBREVIATIONS and len(piece) > 0:
                out[-1] = f"{previous} {piece}"
                continue
        out.append(piece)
    return out


# ── Chunking ───────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A span of a larger document, with enough metadata to cite it."""

    text: str
    index: int
    start: int
    end: int
    source: str = ""
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return approx_token_count(self.text)

    def __len__(self) -> int:
        return len(self.text)


def chunk_text(
    text: str,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    source: str = "",
    respect_sentences: bool = True,
) -> list[Chunk]:
    """Split text into overlapping chunks that respect sentence boundaries.

    Overlap matters: a fact straddling a chunk boundary is otherwise unretrievable
    by either chunk. 64 tokens of overlap is a good default for prose.
    """
    if not text.strip():
        return []
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    units: list[tuple[int, int, str]] = []
    if respect_sentences:
        cursor = 0
        for sentence in sentences(text):
            start = text.find(sentence, cursor)
            if start < 0:  # normalization shifted offsets; fall back to sequential
                start = cursor
            end = start + len(sentence)
            units.append((start, end, sentence))
            cursor = end
    else:
        units = [(0, len(text), text)]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_start = 0
    buffer_tokens = 0

    def flush(end: int) -> None:
        nonlocal buffer, buffer_tokens, buffer_start
        if not buffer:
            return
        body = " ".join(buffer).strip()
        if body:
            chunks.append(
                Chunk(
                    text=body,
                    index=len(chunks),
                    start=buffer_start,
                    end=end,
                    source=source,
                    meta={"sentences": len(buffer)},
                )
            )
        buffer = []
        buffer_tokens = 0

    for start, end, sentence in units:
        sent_tokens = approx_token_count(sentence)
        if sent_tokens > max_tokens:
            # A single oversized sentence: hard-split it on word boundaries.
            flush(end)
            words = sentence.split()
            piece: list[str] = []
            piece_tokens = 0
            for word in words:
                wt = approx_token_count(word) + 1
                if piece_tokens + wt > max_tokens and piece:
                    body = " ".join(piece)
                    chunks.append(
                        Chunk(
                            text=body,
                            index=len(chunks),
                            start=start,
                            end=end,
                            source=source,
                            meta={"hard_split": True},
                        )
                    )
                    piece = piece[-max(1, overlap_tokens // 2) :]
                    piece_tokens = sum(approx_token_count(w) + 1 for w in piece)
                piece.append(word)
                piece_tokens += wt
            if piece:
                chunks.append(
                    Chunk(
                        text=" ".join(piece),
                        index=len(chunks),
                        start=start,
                        end=end,
                        source=source,
                        meta={"hard_split": True},
                    )
                )
            buffer_start = end
            continue

        if not buffer:
            buffer_start = start
        if buffer_tokens + sent_tokens > max_tokens:
            flush(end)
            buffer_start = start
        buffer.append(sentence)
        buffer_tokens += sent_tokens

        if buffer_tokens >= max_tokens - overlap_tokens and overlap_tokens > 0:
            # Slide the window: keep the tail as the head of the next chunk.
            flush(end)
            keep: list[str] = []
            keep_tokens = 0
            for sentence_back in reversed(chunks[-1].text.split(". ")):
                t = approx_token_count(sentence_back)
                if keep_tokens + t > overlap_tokens:
                    break
                keep.insert(0, sentence_back)
                keep_tokens += t
            buffer = keep
            buffer_tokens = keep_tokens

    if buffer:
        flush(units[-1][1] if units else len(text))
    return chunks


# ── Near-duplicate detection ───────────────────────────────────────────────────


class SimHash:
    """64-bit SimHash for near-duplicate detection over large corpora.

    Charikar's construction: weight each feature, sum signed bit vectors, take the
    sign. Documents differing in a small fraction of features get hashes with small
    Hamming distance — so you can find near-dupes in a million-document corpus by
    comparing 64-bit integers instead of shingle sets.
    """

    __slots__ = ("value",)
    BITS = 64

    def __init__(self, value: int) -> None:
        self.value = value & ((1 << self.BITS) - 1)

    @classmethod
    def from_text(cls, text: str, *, ngram: int = 3) -> "SimHash":
        tokens = normalize_text(text, lower=True).split()
        if not tokens:
            return cls(0)
        if len(tokens) < ngram:
            features = tokens
        else:
            features = [" ".join(g) for g in ngrams(tokens, ngram)]
        if not features:
            return cls(0)
        vector = [0] * cls.BITS
        for feature, weight in Counter(features).items():
            digest = int.from_bytes(
                hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
            )
            for bit in range(cls.BITS):
                vector[bit] += weight if (digest >> bit) & 1 else -weight
        value = 0
        for bit in range(cls.BITS):
            if vector[bit] > 0:
                value |= 1 << bit
        return cls(value)

    def hamming(self, other: "SimHash") -> int:
        return bin(self.value ^ other.value).count("1")

    def similarity(self, other: "SimHash") -> float:
        return 1.0 - self.hamming(other) / self.BITS

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SimHash) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"SimHash({self.value:016x})"

    def __int__(self) -> int:
        return self.value


@dataclass
class DedupResult:
    kept: list[int]
    dropped: list[int]
    groups: list[list[int]]

    @property
    def dropped_ratio(self) -> float:
        total = len(self.kept) + len(self.dropped)
        return len(self.dropped) / total if total else 0.0


def dedupe_by_simhash(
    texts: Iterable[str], *, threshold: int = 3, min_length: int = 0
) -> DedupResult:
    """Remove near-duplicate documents.

    ``threshold`` is the maximum Hamming distance treated as a duplicate. 3 of 64
    bits is a good default: it catches copy-paste with minor edits while keeping
    genuinely distinct documents that share phrasing.

    Buckets by 16-bit bands so the comparison set stays small — O(n) for corpora
    with modest duplication rather than O(n²).
    """
    kept: list[int] = []
    dropped: list[int] = []
    groups: list[list[int]] = []
    bands: dict[tuple[int, int], list[int]] = {}
    hashes: list[SimHash] = []

    for index, text in enumerate(texts):
        if len(text.strip()) < min_length:
            dropped.append(index)
            continue
        digest = SimHash.from_text(text)
        hashes.append(digest)
        candidate_keys = {(band, (digest.value >> (band * 16)) & 0xFFFF) for band in range(4)}
        matches: list[int] = []
        for key in candidate_keys:
            matches.extend(bands.get(key, []))
        duplicate_of = -1
        for candidate in set(matches):
            if digest.hamming(hashes[candidate]) <= threshold:
                duplicate_of = candidate
                break
        if duplicate_of >= 0:
            dropped.append(index)
            groups.append([duplicate_of, index])
        else:
            kept.append(index)
            for key in candidate_keys:
                bands.setdefault(key, []).append(len(hashes) - 1)

    return DedupResult(kept=kept, dropped=dropped, groups=groups)


def cosine_counts(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between two term-frequency vectors."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def summarize(text: str, *, max_sentences: int = 3) -> str:
    """Extractive summarizer: pick the highest-scoring sentences in reading order.

    No model required — used for memory consolidation when no LLM is available and
    as the baseline the LLM summarizer is compared against.
    """
    parts = sentences(text)
    if len(parts) <= max_sentences:
        return " ".join(parts)
    freq = word_frequencies(text)
    if not freq:
        return " ".join(parts[:max_sentences])
    top = max(freq.values())
    scores = []
    for position, sentence in enumerate(parts):
        words = [w for w in normalize_text(sentence, lower=True).split() if w in freq]
        if not words:
            scores.append((0.0, position))
            continue
        score = sum(freq[w] / top for w in words) / math.sqrt(len(words))
        # Mild preference for earlier sentences: leads usually carry the point.
        score *= 1.0 / (1.0 + 0.05 * position)
        scores.append((score, position))
    ranked = sorted(scores, key=lambda s: (-s[0], s[1]))[:max_sentences]
    chosen = sorted(position for _, position in ranked)
    return " ".join(parts[i] for i in chosen)


# ── Slugs ──────────────────────────────────────────────────────────────────


def slugify(
    text: str | None,
    *,
    limit: int | None = None,
    fallback: str = "",
    separator: str = "-",
    keep_case: bool = False,
    extra: str = "",
    strip: str | None = None,
    strip_after_limit: bool = False,
) -> str:
    """Turn free text into a URL/filename-safe slug.

    Runs of non-word characters collapse to ``separator``; leading/trailing
    separators are stripped; the result is truncated to ``limit`` and falls
    back to ``fallback`` when empty.

    Canonical unification of the six ``_slug``/``_slugify`` copies across
    agents, media, tools, and builders.  Parameters cover every old variant:

    - ``limit`` — max length (``None`` = unlimited, like app_builder).
    - ``fallback`` — returned when the slug is empty.
    - ``separator`` — ``"-"`` everywhere except fanout, which used ``"_"``.
    - ``keep_case`` / ``extra`` — filesend kept case and allowed ``._-``.
    - ``strip`` — chars stripped from both ends (default: the separator);
      filesend stripped ``"-."``.
    - ``strip_after_limit`` — reflection stripped a trailing separator left
      behind by truncation.
    """
    raw = text or ""
    if not keep_case:
        raw = raw.lower()
    if keep_case:
        pattern = r"[^A-Za-z0-9" + re.escape(extra) + r"]+"
    else:
        pattern = r"[^a-z0-9" + re.escape(extra) + r"]+"
    slug = re.sub(pattern, separator, raw).strip(strip if strip is not None else separator)
    if limit is not None:
        slug = slug[:limit]
        if strip_after_limit:
            slug = slug.strip(strip if strip is not None else separator)
    return slug or fallback
