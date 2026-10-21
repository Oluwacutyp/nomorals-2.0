"""Real CLIP BPE tokenizer for the imggen organ.

Devon's SD pipelines condition on CLIP token ids. Until this module,
:mod:`.sdcompat` tokenized by *hashing* words into the vocab — stable,
but not CLIP BPE, so prompts were understood only approximately. This
module implements the actual byte-pair encoding algorithm CLIP uses:

- byte-level alphabet (``bytes_to_unicode``) so any UTF-8 text maps
  onto printable token characters with no unknown-token failure;
- merge ranks from ``merges.txt`` applied greedily by rank;
- ``<|startoftext|>`` (49406) / ``<|endoftext|>`` (49407) framing and
  the 77-token context CLIP was trained with (pad = EOT, matching
  what the encoder's consumers expect).

Vocab/merges are *data*, shipped with models, never hardcoded here.
:func:`load_clip_tokenizer` searches, in order:

1. the checkpoint's own directory and its ``tokenizer/`` subdir
   (single-file layout, and diffusers layout respectively);
2. ``$DEVON_IMGGEN_DIR/tokenizer/``;
3. ``~/.devon-models/tokenizer/``.

In each directory it accepts either ``tokenizer.json`` (fast-tokenizer
format: ``model.vocab`` + ``model.merges``) or the classic pair
``vocab.json`` + ``merges.txt``. When nothing is found it returns
``None`` and the caller decides how to disclose the fallback — this
module never silently pretends hashing is BPE.
"""

from __future__ import annotations

import html
import json
import os
import re
from functools import lru_cache

from . import ImgGenError, checkpoint_dir

__all__ = [
    "CLIPBPETokenizer",
    "bytes_to_unicode",
    "load_clip_tokenizer",
    "tokenizer_search_dirs",
]

SOT_ID = 49406
EOT_ID = 49407
CONTEXT_LENGTH = 77


# ---------------------------------------------------------------------------
# Byte alphabet (OpenAI CLIP's bytes_to_unicode)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a unique printable unicode char.

    Printable bytes map to themselves; the rest (whitespace, control)
    are shifted into the 256+ range so tokens never contain literal
    whitespace — merges can then be stored as plain text lines.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# CLIP's pre-tokenization pattern, approximated with stdlib ``re``:
# contractions first, then letter runs, digit runs, symbol runs, and
# whitespace (kept when not trailing). Unicode-aware via ``\w``/``\d``
# classes; the BPE core below is exact regardless of this split.
_TOKEN_RE = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""",
    re.IGNORECASE)


def _clean(text: str) -> str:
    """CLIP's basic_clean + whitespace_clean (ftfy-free variant)."""
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text)


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class CLIPBPETokenizer:
    """Byte-pair encoding over a CLIP vocab + merge table."""

    def __init__(self, vocab: dict[str, int],
                 bpe_ranks: dict[tuple[str, str], int],
                 source: str = "") -> None:
        if not vocab:
            raise ImgGenError("CLIP tokenizer: empty vocab")
        self.encoder = dict(vocab)
        self.decoder = {i: t for t, i in vocab.items()}
        self.bpe_ranks = dict(bpe_ranks)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.source = source
        self.sot_id = self.encoder.get("<|startoftext|>", SOT_ID)
        self.eot_id = self.encoder.get("<|endoftext|>", EOT_ID)
        self._bpe_cache: dict[str, str] = {}

    # -- construction -------------------------------------------------
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str) -> "CLIPBPETokenizer":
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        merges = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#version"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ImgGenError(
                    f"CLIP tokenizer: malformed merges line in "
                    f"{merges_path}: {line!r}")
            merges.append((parts[0], parts[1]))
        ranks = {pair: i for i, pair in enumerate(merges)}
        return cls(vocab, ranks,
                   source=f"{vocab_path} + {merges_path}")

    @classmethod
    def from_tokenizer_json(cls, path: str) -> "CLIPBPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model") or {}
        vocab = model.get("vocab")
        if not isinstance(vocab, dict) or not vocab:
            raise ImgGenError(
                f"CLIP tokenizer: {path} has no model.vocab")
        merges = []
        for item in model.get("merges", []):
            if isinstance(item, str):
                parts = item.split()
            else:  # newer format: {"type": ..., "string": "A B"}-ish
                parts = str(item.get("string", "")).split()
            if len(parts) != 2:
                raise ImgGenError(
                    f"CLIP tokenizer: malformed merge in {path}: "
                    f"{item!r}")
            merges.append((parts[0], parts[1]))
        ranks = {pair: i for i, pair in enumerate(merges)}
        return cls(vocab, ranks, source=path)

    # -- BPE core -------------------------------------------------------
    def _bpe(self, token: str) -> list[str]:
        """Apply merge ranks to one pre-tokenized chunk."""
        if token in self._bpe_cache:
            return self._bpe_cache[token].split(" ")
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if len(word) == 1:
            result = list(word)
            self._bpe_cache[token] = " ".join(result)
            return result
        pairs = _get_pairs(word)
        while True:
            bigram = min(pairs,
                         key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 \
                        and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = list(word)
        self._bpe_cache[token] = " ".join(result)
        return result

    # -- public API -----------------------------------------------------
    def encode(self, text: str,
               max_tokens: int = CONTEXT_LENGTH) -> list[int]:
        """Text → exactly ``max_tokens`` ids, SOT/EOT framed, EOT-padded."""
        text = _clean(text).lower()
        ids: list[int] = []
        for chunk in _TOKEN_RE.findall(text):
            if chunk == "<|startoftext|>":
                ids.append(self.sot_id)
                continue
            if chunk == "<|endoftext|>":
                ids.append(self.eot_id)
                continue
            encoded = "".join(self.byte_encoder[b]
                              for b in chunk.encode("utf-8"))
            for piece in self._bpe(encoded):
                if piece not in self.encoder:
                    raise ImgGenError(
                        f"CLIP tokenizer: piece {piece!r} (from "
                        f"{chunk!r}) is not in the vocab from "
                        f"{self.source or 'unknown source'} — the "
                        f"vocab/merges pair is incomplete or mismatched")
                ids.append(self.encoder[piece])
        ids = ids[:max_tokens - 2]
        out = [self.sot_id, *ids, self.eot_id]
        out.extend([self.eot_id] * (max_tokens - len(out)))
        return out[:max_tokens]


# ---------------------------------------------------------------------------
# Vocab discovery
# ---------------------------------------------------------------------------

def tokenizer_search_dirs(checkpoint_path: str = "") -> list[str]:
    """Directories searched for tokenizer data, in priority order."""
    dirs: list[str] = []
    if checkpoint_path:
        base = checkpoint_path if os.path.isdir(checkpoint_path) \
            else os.path.dirname(checkpoint_path)
        if base:
            dirs.append(base)
            dirs.append(os.path.join(base, "tokenizer"))
    dirs.append(os.path.join(checkpoint_dir(), "tokenizer"))
    dirs.append(os.path.expanduser("~/.devon-models/tokenizer"))
    seen: set[str] = set()
    out = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def load_clip_tokenizer(checkpoint_path: str = "") -> CLIPBPETokenizer | None:
    """Find and load CLIP BPE data near a checkpoint; None if absent.

    A directory holding a *partial* pair (vocab without merges, say)
    is a hard error, not a skip — half a tokenizer silently ignored
    would be exactly the silent fidelity loss this module exists to
    end. Absence everywhere is fine and returns None.
    """
    for d in tokenizer_search_dirs(checkpoint_path):
        tj = os.path.join(d, "tokenizer.json")
        if os.path.exists(tj):
            return CLIPBPETokenizer.from_tokenizer_json(tj)
        vj = os.path.join(d, "vocab.json")
        mt = os.path.join(d, "merges.txt")
        if os.path.exists(vj) and os.path.exists(mt):
            return CLIPBPETokenizer.from_files(vj, mt)
        if os.path.exists(vj) != os.path.exists(mt):
            raise ImgGenError(
                f"CLIP tokenizer: incomplete tokenizer data in {d} — "
                f"vocab.json and merges.txt must be provided together")
    return None
