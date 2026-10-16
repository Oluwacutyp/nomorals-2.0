"""Lossless chunking + media routing, driven by :class:`PlatformLimits`.

Split priority: paragraph → sentence → hard boundary. **Never truncate**:
``"".join(chunk_message(text)) == text`` for every input, including a
single pathological word, empty text, and emoji. When a message is split,
:func:`label_chunks` tags each part ``(i/n)`` so the reader can reassemble
it; the label is metadata, not content — the content chunks themselves
concatenate to the original exactly.

Media over the platform's inline cap is sent as a document, never as a
failed inline attempt. A caption over the caption cap is chunked: first
chunk as the caption, the rest as follow-up text.
"""

from __future__ import annotations

import re
from typing import Literal

from .platform_limits import PlatformLimits, for_platform

__all__ = [
    "chunk_message",
    "chunk_for_platform",
    "label_chunks",
    "strip_label",
    "media_plan",
    "media_kind_for",
    "split_caption",
]

#: Paragraph boundary: a blank line (a newline, optional spaces, a newline).
_PARAGRAPH_SEP = re.compile(r"(\n[ \t]*\n)")
#: Sentence boundary: sentence-ending punctuation + following whitespace.
#: The whitespace is captured so it stays attached to its sentence.
_SENTENCE_SEP = re.compile(r"(?<=[.!?…])(\s+)")
#: A trailing " (i/n)" label, for stripping labels back off.
_LABEL_RE = re.compile(r"\s*\(\d+/\d+\)\s*$")


def _split_keep(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split on ``pattern``, keeping each separator attached to the piece
    before it. ``"".join(result) == text`` always; empty pieces dropped
    (they contribute nothing to the join)."""
    parts = pattern.split(text)
    atoms: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:  # captured separator — end of this atom
            atoms.append(current)
            current = ""
    if current:
        atoms.append(current)
    return [a for a in atoms if a]


def _hard_split(atom: str, budget: int) -> list[str]:
    """Split ``atom`` at ``budget`` chars, preferring a space boundary.
    The space stays at the end of the left chunk, so the join is exact."""
    out: list[str] = []
    rest = atom
    while len(rest) > budget:
        cut = rest.rfind(" ", 0, budget)
        if cut <= budget // 2:
            cut = budget
        else:
            cut = cut + 1  # keep the space — join stays exact
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        out.append(rest)
    return out


def _pack(atoms: list[str], budget: int) -> list[str]:
    """Greedily pack atoms (each <= budget) into chunks <= budget."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for atom in atoms:
        if current and current_len + len(atom) > budget:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(atom)
        current_len += len(atom)
    if current:
        chunks.append("".join(current))
    return chunks or [""]


def _chunk_with_budget(text: str, budget: int) -> list[str]:
    if len(text) <= budget:
        return [text]
    atoms: list[str] = []
    for para in _split_keep(text, _PARAGRAPH_SEP):
        if len(para) <= budget:
            atoms.append(para)
            continue
        for sent in _split_keep(para, _SENTENCE_SEP):
            if len(sent) <= budget:
                atoms.append(sent)
                continue
            atoms.extend(_hard_split(sent, budget))
    return _pack(atoms, budget)


def chunk_message(text: str | None, *, max_text: int) -> list[str]:
    """Split ``text`` into content chunks of at most ``max_text`` chars.

    Content chunks carry no labels (see :func:`label_chunks`). The join
    of the returned chunks is exactly ``text`` — never truncated, never
    reordered, separators preserved.
    """
    text = str(text or "")
    budget = max(1, int(max_text))
    if len(text) <= budget:
        return [text]
    chunks = _chunk_with_budget(text, budget)
    if len(chunks) == 1:
        return chunks
    # Reserve room for the " (i/n)" labels: the label length depends on
    # the chunk count, which depends on the label length. Iterate until
    # stable (digit-width changes converge in at most a couple of passes).
    for _ in range(4):
        suffix_len = len(f" ({len(chunks)}/{len(chunks)})")
        budget2 = budget - suffix_len
        if budget2 < 1:
            return chunks
        new = _chunk_with_budget(text, budget2)
        if len(new) == len(chunks):
            return new
        chunks = new
    return chunks


def label_chunks(chunks: list[str]) -> list[str]:
    """Tag each chunk ``(i/n)``. A single chunk goes out unlabelled."""
    chunks = list(chunks)
    if len(chunks) <= 1:
        return chunks
    n = len(chunks)
    return [f"{c} ({i + 1}/{n})" for i, c in enumerate(chunks)]


def strip_label(chunk: str) -> str:
    """Best-effort removal of a trailing ``(i/n)`` label. Only use when
    the chunk is known to be labelled — content that naturally ends in
    ``(2/3)`` is indistinguishable from a label."""
    return _LABEL_RE.sub("", str(chunk or ""))


def chunk_for_platform(text: str | None, platform: str) -> list[str]:
    """Chunk ``text`` for ``platform``. Platforms whose adapter owns
    segmentation (SMS) return the text untouched as a single chunk."""
    limits = for_platform(platform)
    if not limits.gateway_chunks:
        return [str(text or "")]
    return chunk_message(text, max_text=limits.max_text)


def media_plan(size_bytes: int, limits: PlatformLimits) -> Literal["inline", "document"]:
    """``"inline"`` when the media fits the platform's inline cap, else
    ``"document"``. A cap of 0 means the platform has no cap."""
    cap_mb = float(limits.max_file_mb or 0)
    if cap_mb <= 0:
        return "inline"
    return "inline" if float(size_bytes) <= cap_mb * 1024 * 1024 else "document"


def media_kind_for(size_bytes: int, requested_kind: str, limits: PlatformLimits) -> str:
    """The kind to send: ``"document"`` when over the inline cap, else
    the requested kind unchanged."""
    if media_plan(size_bytes, limits) == "document":
        return "document"
    return requested_kind


def split_caption(caption: str | None, *, max_caption: int) -> list[str]:
    """Split a caption against the caption cap. First chunk is the caption;
    the rest go out as follow-up text. Never truncates."""
    caption = str(caption or "")
    if max_caption <= 0 or len(caption) <= max_caption:
        return [caption]
    return chunk_message(caption, max_text=max_caption)
