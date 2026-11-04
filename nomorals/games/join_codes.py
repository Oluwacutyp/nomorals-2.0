"""Short shareable join codes for duels.

One code system for every duel surface (games-engine pass, DESIGN
§13/§34 follow-up): when a member opens a SPECIFIC duel — a relay
invite, an open math duel — the post carries a short code designed
for people, never the internal session/duel id. Six characters from
an unambiguous alphabet (no 0/O, 1/I/L — nothing that survives a
screenshot or a read-aloud badly), minted with ``secrets`` and
checked against the caller's live set, so a code is unique among
the duels it could be confused with. Codes are handles, not
credentials: they expire with the duel, die on first use, and
failed guesses are rate-limited by the surfaces that accept them.
"""
from __future__ import annotations

import secrets
from typing import Iterable

__all__ = ["JOIN_CODE_ALPHABET", "JOIN_CODE_LENGTH", "new_join_code",
           "looks_like_join_code"]

#: 31 glyphs, each visually distinct in chat fonts and unambiguous
#: when read aloud. No vowels-by-accident profanity filters needed —
#: this is a game invite, not a password.
JOIN_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

JOIN_CODE_LENGTH = 6


def new_join_code(taken: Iterable[str] = ()) -> str:
    """Mint a fresh code not in ``taken`` (compared case-insensitively)."""
    used = {str(c).strip().upper() for c in taken}
    for _ in range(100):
        code = "".join(secrets.choice(JOIN_CODE_ALPHABET)
                       for _ in range(JOIN_CODE_LENGTH))
        if code not in used:
            return code
    # Astronomically unlikely with 31^6 codes; lengthen rather than
    # ever hand out a duplicate.
    code = "".join(secrets.choice(JOIN_CODE_ALPHABET)
                   for _ in range(JOIN_CODE_LENGTH + 2))
    return code


def looks_like_join_code(text: str) -> bool:
    """True when ``text`` is shaped like a join code (right length,
    alphabet only). Shape only — never proof a code exists."""
    t = (text or "").strip().upper()
    return (len(t) == JOIN_CODE_LENGTH
            and all(ch in JOIN_CODE_ALPHABET for ch in t))
