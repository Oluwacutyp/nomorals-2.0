"""Challenge packs for the arena: tasks with verifiable outcomes.

The core topic bank (``topics.py``) holds research *prompts*. A
*challenge* is a step harder: a concrete task — build it, measure it,
prove it — plus a one-line verifiable acceptance criterion
(``verify``) and a ``kind`` (``"code"`` | ``"research"`` | ``"build"``).

Challenge entries are plain topic-pack entries with two extra keys,
so they flow through the existing pack registry, sampler, anti-repeat
window, and knowledge store unchanged:

    challenge(
        "Implement a lock-free MPMC ring buffer in pure Python",
        3,
        "pytest passes: 4 producers x 4 consumers, 100k items, no loss",
        "code", "concurrency", "lock-free",
    )
    # -> {"t": ..., "d": 3, "tags": (...), "verify": ..., "kind": "code"}

The packs themselves live in ``challenge_packs/`` (one module per
domain); each module registers itself on import via
``register_topic_pack``. This module holds the constructor and the
validation the tests enforce.
"""

from __future__ import annotations

from typing import Any

__all__ = ["KINDS", "challenge", "validate_pack", "is_challenge"]

#: Allowed challenge kinds.
KINDS = ("code", "research", "build")


def challenge(text: str, difficulty: int, verify: str,
              kind: str = "code", *tags: str) -> dict[str, Any]:
    """Build one challenge entry.

    ``verify`` is mandatory and must name a checkable outcome — a
    challenge without an acceptance criterion is an essay prompt, and
    essay prompts are rejected.
    """
    text = str(text or "").strip()
    if not text:
        raise ValueError("challenge needs non-empty text")
    verify = str(verify or "").strip()
    if not verify:
        raise ValueError("challenge needs a verifiable acceptance criterion "
                         f"(got none for {text[:60]!r})")
    try:
        d = min(3, max(1, int(difficulty)))
    except (TypeError, ValueError):
        raise ValueError(f"challenge difficulty must be 1-3, got {difficulty!r}")
    kind = str(kind or "code").strip().lower()
    if kind not in KINDS:
        raise ValueError(f"challenge kind must be one of {KINDS}, got {kind!r}")
    clean_tags = tuple(str(t).strip().lower() for t in (tags or ())
                       if str(t).strip())
    return {"t": text, "d": d, "tags": clean_tags,
            "verify": verify, "kind": kind}


def is_challenge(entry: dict[str, Any]) -> bool:
    """True when a bank entry carries a verifiable acceptance criterion."""
    return bool(entry.get("verify"))


def validate_pack(pack_name: str,
                  topics: dict[str, list[dict[str, Any]]],
                  min_per_category: int = 10) -> list[str]:
    """Validate a challenge pack. Returns a list of error strings (empty = ok).

    Enforced: non-empty text, difficulty 1-3, non-empty ``verify``,
    known ``kind``, at least ``min_per_category`` challenges per
    category, a difficulty spread (not everything grade 3), and no
    duplicate challenge texts within the pack.
    """
    errors: list[str] = []
    if not topics:
        return [f"pack {pack_name!r}: no categories"]
    seen: set[str] = set()
    for cat, entries in topics.items():
        if not entries or len(entries) < min_per_category:
            errors.append(
                f"pack {pack_name!r} category {cat!r}: "
                f"{len(entries or [])} challenges, need >= {min_per_category}")
        spread = {1: 0, 2: 0, 3: 0}
        for i, e in enumerate(entries or []):
            where = f"pack {pack_name!r} {cat!r}[{i}]"
            text = str(e.get("t", "")).strip()
            if not text:
                errors.append(f"{where}: empty text")
                continue
            if text in seen:
                errors.append(f"{where}: duplicate challenge text")
            seen.add(text)
            d = e.get("d")
            if d not in (1, 2, 3):
                errors.append(f"{where}: difficulty {d!r} not in 1-3")
            else:
                spread[int(d)] += 1
            if not str(e.get("verify", "")).strip():
                errors.append(f"{where}: missing verifiable acceptance criterion")
            if str(e.get("kind", "code")) not in KINDS:
                errors.append(f"{where}: unknown kind {e.get('kind')!r}")
        if entries and spread[1] == 0 and spread[2] == 0:
            errors.append(
                f"pack {pack_name!r} category {cat!r}: no easy/medium "
                f"challenges — the sampler needs a difficulty ramp")
    return errors
