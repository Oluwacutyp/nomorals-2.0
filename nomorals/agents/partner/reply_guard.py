"""Send-boundary guard for conversational replies.

A model reply is guilty until proven innocent. Small models answer a
chat prompt with their reasoning scaffold instead of a reply — the
incident this guard exists for (owner's WhatsApp group, 2026-11-04):
the model's per-message analysis

    analyze the user's input:
    "u dey smoke" - this is a nigerian pidgin/english phrase
    meaning "you are smoking," often used playfully…
    context: group chat with sucidial and ekpo. the vibe is
    neutral, pace is buzzing

posted verbatim, AS THE OWNER, on his personal account. The draft
pipeline (``partner/responder.py``) post-processes model text for
FORMATTING only — markdown, emoji caps, texting voice, budget — and
nothing between the draft and ``gateway.send`` ever asked whether
the text was a reply at all. The games narration path learned this
lesson first (``games/gamemaster``'s narration sanitizer: scaffold
rejected, journaled, template-forge fallback); this module is the
same discipline for the conversational path, in this path's own
module, applied at the one seam every conversational send crosses:
:meth:`PartnerRuntime._send_reply`. Command/control replies and
bot-system output (games, economy) travel other seams and are NOT
this guard's business.

Semantics: detection is at the BODY level, not line-stripping. The
analysis prose between the header and the context line is itself
the leak, so a part that matches the scaffold shape is rejected
WHOLE — never trimmed into a "reply". Surviving clean parts are
sent untouched and in order (a composed reply downstream always
wins). If every non-blank part is rejected, the turn resolves to
one short in-character reaction — she was addressed, typing showed,
silence would read as a glitch — drawn from a static pool (never
the lexicon's degraded "brain's buffering" stall phrases, never
meta-commentary about the failure). Every rejection is journaled
to the partner failure ledger with the scaffold's hash and length
only: the raw text never enters any chat, and the ledger keeps no
copy worth quoting either.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from ...social.chat.base import ChatKind
from .failures import record_failure

__all__ = [
    "scaffold_reason",
    "is_scaffold",
    "fallback_reaction",
    "guard_reply_parts",
]

#: The template header the incident wore, and its close variants:
#: the model's analysis is ABOUT the message it was handed.
_HEADER_RE = re.compile(
    r"^\s*analy[sz](?:e|ing|is)\s+(?:of\s+)?the\s+(?:user'?s\s+)?"
    r"(?:input|message|request|text)\b",
    re.IGNORECASE,
)
#: A first line that is just a scaffold label: "analysis:", "reasoning —".
_LABEL_RE = re.compile(
    r"^\s*(?:analysis|reasoning|thought\s+process|chain\s+of\s+thought)"
    r"\s*[:\-—]",
    re.IGNORECASE,
)
#: Reasoning preambles: the model narrating its own process as the
#: opening of the "reply".
_PREAMBLE_RE = re.compile(
    r"^\s*(?:let\s+me\s+(?:analyze|analyse|think\s+about|consider|break\s+down)"
    r"|here(?:'s|\s+is)\s+my\s+analysis"
    r"|my\s+analysis\s+of\s+(?:the|your|this)\b"
    r"|i\s+need\s+to\s+analy[sz]e\b)",
    re.IGNORECASE,
)
#: The chat-profile scaffold line: a "context:" summary carrying the
#: profile's vibe/pace readout — the profile's own context lines fed
#: to the prompt, echoed back as a verdict about the room.
_CONTEXT_LINE_RE = re.compile(r"^\s*context\s*[:\-]", re.IGNORECASE)
_VIBE_PACE_RE = re.compile(
    r"\bthe\s+vibe\s+is\b[^.\n]*\b(?:the\s+)?pace\s+is\b", re.IGNORECASE)
#: Phrases that betray scaffold/meta anywhere in a body (mirrors the
#: narration guard's meta set, trimmed to what a chat reply could
#: plausibly carry).
_META_PHRASES = (
    "analyze the user's input", "analyze the user’s input",
    "constraint checklist", "confidence score",
    "as an ai", "language model", "the prompt", "my instructions",
    "the user asked", "the user's input is", "the user’s input is",
)

#: In-character last-resort reactions. She is a person in the chat,
#: not a system reporting an error: no apology, no explanation, no
#: promise of a follow-up that isn't coming.
_GROUP_FALLBACKS = (
    "😂", "lol 😅", "😅", "abeg 😂", "😂😂", "ehn 😅",
)
_DM_FALLBACKS = (
    "😅", "lol 😅", "hmm 😅", "😂",
)


def scaffold_reason(text: str) -> str:
    """Why ``text`` is analysis scaffold, or "" when it reads as a reply.

    Body-level signatures, checked strongest-first. Ordinary speech —
    including pidgin, including a reply that quotes the triggering
    phrase — matches none of these: every rule needs the model's
    own scaffolding furniture (the analyze-header, a context/vibe
    verdict line, a reasoning preamble, or a meta phrase), never
    just the chat's content words.
    """
    body = str(text or "")
    if not body.strip():
        return ""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    if _HEADER_RE.match(first):
        return "analysis header"
    for line in lines:
        if _VIBE_PACE_RE.search(line):
            return "vibe/pace scaffold line"
        if _CONTEXT_LINE_RE.match(line) and "vibe" in line.lower():
            return "context/vibe scaffold line"
    if _LABEL_RE.match(first) or _PREAMBLE_RE.match(first):
        return "reasoning preamble"
    low = body.lower()
    for phrase in _META_PHRASES:
        if phrase in low:
            return f"meta phrase {phrase!r}"
    return ""


def is_scaffold(text: str) -> bool:
    return bool(scaffold_reason(text))


def fallback_reaction(kind: Any, rng: Any = None) -> str:
    """One short in-character reaction for a fully-rejected turn."""
    pool = _GROUP_FALLBACKS if kind == ChatKind.GROUP else _DM_FALLBACKS
    pick = rng.choice if rng is not None else random.choice
    return pick(pool)


def _journal_rejection(reason: str, raw: str) -> None:
    """Held ledger entry for a rejected reply part (never posted).

    Hash + length only — the scaffold's text is not worth keeping
    anywhere, and the ledger is readable via /status.
    """
    try:
        digest = hashlib.sha256(str(raw or "").encode("utf-8")).hexdigest()
        record_failure(
            "runtime.PartnerRuntime._send_reply:scaffold_guard",
            ValueError(
                f"reply scaffold rejected ({reason}): "
                f"sha256={digest[:16]} len={len(str(raw or ''))}"),
        )
    except Exception:  # noqa: BLE001 - journaling must never break a reply
        pass


def guard_reply_parts(parts: list[str], *, kind: Any,
                      rng: Any = None) -> list[str]:
    """Filter composed reply parts at the send boundary.

    Clean parts pass through untouched and in order; scaffold parts
    are dropped and journaled. When the turn had content but every
    part was scaffold, one in-character fallback reaction replaces
    it, so an addressed turn never resolves to silence. Blank parts
    are not the guard's concern and pass through as they were.
    """
    kept: list[str] = []
    had_content = False
    for part in parts or []:
        text = str(part or "")
        if not text.strip():
            kept.append(part)
            continue
        had_content = True
        reason = scaffold_reason(text)
        if reason:
            _journal_rejection(reason, text)
            continue
        kept.append(part)
    if had_content and not any(str(p or "").strip() for p in kept):
        kept = [p for p in kept if str(p or "").strip()] \
            + [fallback_reaction(kind, rng)]
    return kept
