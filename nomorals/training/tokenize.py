"""Byte-pair encoding, trained from scratch with the standard library.

A real BPE trainer, not a wrapper. Training merges the most frequent adjacent
pair until the target vocabulary size is reached; encoding applies those merges
in learned order. It is slower than a Rust tokenizer and needs a bigger corpus
for the same vocabulary quality, but it means the system can build a tokenizer
for its own personal model with nothing installed.

When ``transformers`` is present, ``HFTokenizer`` wraps it instead. Both satisfy
the same three-method protocol, so the trainer never knows which one it has.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..compat import load_optional
from ..core.errors import ParseError, ValidationError
from ..core.logging_setup import get_logger

__all__ = [
    "BPETokenizer", "HFTokenizer", "build_tokenizer", "SPECIAL_TOKENS",
    "encode_batch", "tokenizer_corpus_stats",
]

_log = get_logger(__name__)

SPECIAL_TOKENS = (
    "<" + "|endoftext|>",
    "<" + "|im_start|>",
    "<" + "|im_end|>",
    "<" + "|pad|>",
    "<" + "|user|>",
    "<" + "|assistant|>",
)
UNK = "<" + "|unk|>"


def _pairs(symbols: list[str]) -> Counter:
    """Count adjacent pairs. The inner loop of BPE training."""
    counts: Counter = Counter()
    previous = symbols[0] if symbols else None
    for symbol in symbols[1:]:
        counts[(previous, symbol)] += 1
        previous = symbol
    return counts


class BPETokenizer:
    """Byte-level BPE. Trains from a plain-text corpus."""

    def __init__(
        self,
        *,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        specials: tuple[str, ...] = SPECIAL_TOKENS,
    ) -> None:
        self.specials = tuple(specials)
        self.merges: list[tuple[str, str]] = list(merges or [])
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[str, int] = dict(vocab or {})
        if not self.vocab:
            # Byte-level base: every byte is a token, so nothing is ever unencodable.
            self.vocab = {bytes([b]).decode("latin-1"): b for b in range(256)}
            for index, token in enumerate(self.specials, start=256):
                self.vocab[token] = index
            self.vocab[UNK] = len(self.vocab)
        self._inverse = {i: t for t, i in self.vocab.items()}
        self.unk_id = self.vocab.get(UNK, 0)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ── training ─────────────────────────────────────────────────────────────

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 2048,
        specials: tuple[str, ...] = SPECIAL_TOKENS,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn merges from a corpus. Deterministic for a given input."""
        if vocab_size < 300:
            raise ValidationError("vocab_size must be at least 300", field="vocab_size")

        words: Counter = Counter()
        for text in texts:
            for word in _pretokenize(text):
                words[word] += 1
        if not words:
            raise ValidationError("cannot train a tokenizer on an empty corpus")

        splits = {w: list(w) for w in words}
        merges: list[tuple[str, str]] = []
        target_merges = vocab_size - 256 - len(specials) - 1

        while len(merges) < target_merges:
            counts: Counter = Counter()
            for word, frequency in words.items():
                symbols = splits[word]
                for index in range(len(symbols) - 1):
                    counts[(symbols[index], symbols[index + 1])] += frequency
            if not counts:
                break
            best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            pair, frequency = best
            if frequency < min_frequency:
                break
            merges.append(pair)

            merged = pair[0] + pair[1]
            for word in list(splits):
                symbols = splits[word]
                if pair[0] not in symbols:
                    continue
                out: list[str] = []
                index = 0
                while index < len(symbols):
                    if (
                        index < len(symbols) - 1
                        and symbols[index] == pair[0]
                        and symbols[index + 1] == pair[1]
                    ):
                        out.append(merged)
                        index += 2
                    else:
                        out.append(symbols[index])
                        index += 1
                splits[word] = out

        vocab = {bytes([b]).decode("latin-1"): b for b in range(256)}
        for token in specials:
            vocab[token] = len(vocab)
        for left, right in merges:
            combined = left + right
            if combined not in vocab:
                vocab[combined] = len(vocab)
        vocab[UNK] = len(vocab)
        _log.info("trained BPE: %d merges, vocab %d", len(merges), len(vocab))
        return cls(vocab=vocab, merges=merges, specials=specials)

    # ── encode / decode ──────────────────────────────────────────────────────

    def encode(self, text: str, *, add_specials: bool = False) -> list[int]:
        """Tokenize text. Special tokens in the input become their own ids."""
        if not text:
            return []
        ids: list[int] = []
        for chunk in _split_specials(text, self.specials):
            if chunk in self.specials:
                ids.append(self.vocab.get(chunk, self.unk_id))
                continue
            for word in _pretokenize(chunk):
                ids.extend(self._encode_word(word))
        if add_specials:
            ids.append(self.vocab.get(SPECIAL_TOKENS[0], self.unk_id))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        symbols = list(word)
        while len(symbols) > 1:
            best_index = -1
            best_rank = None
            for index in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_index < 0:
                break
            symbols = (
                symbols[:best_index]
                + [symbols[best_index] + symbols[best_index + 1]]
                + symbols[best_index + 2 :]
            )
        return [self.vocab.get(s, self.unk_id) for s in symbols]

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[str] = []
        for token_id in ids:
            token = self._inverse.get(token_id)
            if token is None or token == UNK:
                continue
            parts.append(token)
        return _latin1_to_utf8("".join(parts))

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "kind": "bpe",
                    "vocab": self.vocab,
                    "merges": [list(m) for m in self.merges],
                    "specials": list(self.specials),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "BPETokenizer":
        target = Path(path).expanduser()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ParseError(f"could not load tokenizer {target}: {exc}") from exc
        return cls(
            vocab={k: int(v) for k, v in (data.get("vocab") or {}).items()},
            merges=[(m[0], m[1]) for m in data.get("merges") or [] if len(m) == 2],
            specials=tuple(data.get("specials") or SPECIAL_TOKENS),
        )


def _pretokenize(text: str) -> list[str]:
    """Split into runs of word characters, whitespace, or single punctuation.

    BPE works on pre-tokenized words; without this, merges would happily join
    across a sentence boundary and the vocabulary fills with nonsense.
    """
    import re

    return re.findall(r"\w+|\s+|[^\w\s]", text, flags=re.UNICODE)


def _split_specials(text: str, specials: tuple[str, ...]) -> list[str]:
    """Split text around special tokens so they tokenize to their own ids."""
    import re

    if not specials:
        return [text]
    pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
    return [chunk for chunk in re.split(pattern, text) if chunk]


def _latin1_to_utf8(text: str) -> str:
    """Byte-level BPE decodes to latin-1 code points; re-decode as UTF-8."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class HFTokenizer:
    """Wraps a Hugging Face tokenizer behind the same three-method protocol."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    @property
    def vocab_size(self) -> int:
        return int(self.inner.vocab_size)

    def encode(self, text: str, *, add_specials: bool = False) -> list[int]:
        return list(self.inner.encode(text, add_special_tokens=add_specials))

    def decode(self, ids: Iterable[int]) -> str:
        return str(self.inner.decode(list(ids), skip_special_tokens=True))

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        self.inner.save_pretrained(str(target))
        return target


def build_tokenizer(model: str = "", *, prefer: str = "auto") -> Any:
    """Return an HF tokenizer when available, else an empty BPE to be trained."""
    if prefer in {"auto", "hf"}:
        transformers = load_optional("transformers")
        if transformers is not None and model:
            try:
                return HFTokenizer(transformers.AutoTokenizer.from_pretrained(model))
            except Exception as exc:  # noqa: BLE001 - fall back to the built-in BPE
                _log.warning("could not load HF tokenizer %s: %s", model, exc)
    if prefer == "hf":
        raise ValidationError("transformers is not available or the model did not load")
    return BPETokenizer()


def encode_batch(
    tokenizer: Any, texts: Iterable[str], *, add_specials: bool = False
) -> list[list[int]]:
    """Encode many texts with one call — the batch path tokenizers are
    expected to offer (HF's ``__call__`` takes lists; the BPE never did)."""
    return [tokenizer.encode(text, add_specials=add_specials) for text in texts]


def tokenizer_corpus_stats(
    tokenizer: Any, texts: Iterable[str], *, sample: int = 2000
) -> dict[str, Any]:
    """Token-length distribution over a corpus (capped sample).

    This is what sizes ``max_seq_len`` honestly: the p95 token count tells
    you what sequence length fits the data instead of guessing 512.  Also
    reports the compression ratio (chars per token) as a tokenizer-quality
    sniff test — byte-level BPE on chat text should land around 3-5.
    """
    lengths: list[int] = []
    chars = 0
    total = 0
    for index, text in enumerate(texts):
        if index >= sample:
            break
        total += 1
        ids = tokenizer.encode(text)
        lengths.append(len(ids))
        chars += len(text)
    if not lengths:
        return {"rows": 0, "sampled": 0}
    lengths.sort()
    tokens = sum(lengths)
    return {
        "rows": total,
        "sampled": len(lengths),
        "tokens_total": tokens,
        "tokens_median": lengths[len(lengths) // 2],
        "tokens_p95": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
        "tokens_max": lengths[-1],
        "chars_per_token": round(chars / max(1, tokens), 2),
    }
