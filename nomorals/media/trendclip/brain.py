"""The brain rung — Devon's brain picks moments too.

The deterministic rung (``scoring.select_moments``) scores every second
and ranks by evidence. The brain rung reads the same transcript evidence
and picks by meaning: which moments land as a short clip, why, in the
brain's own words, grounded in what was actually said. Both run, both are
recorded in the job receipt; the profile runtime sorts which runs where.

The organ never imports the brain: the caller supplies a ``pick_fn`` —
a callable taking a prompt string and returning the brain's parsed JSON
(or None when the brain is unavailable). The /clip handler supplies
``brain_for(context).chat_json`` wrapped around that shape; a tool call
from the brain itself runs the deterministic rung only (the brain is
already driving — it reads the candidates and picks). A failed or
unavailable brain is an honest note in the receipt, never a fake pick.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

__all__ = [
    "brain_pick_moments",
    "brain_platform_copy",
    "build_copy_prompt",
    "build_pick_prompt",
    "validate_picks",
]

MAX_PICKS = 3


def build_pick_prompt(*, topic: str, transcript: str,
                      candidates: list[dict[str, Any]]) -> str:
    """The moment-pick prompt: the brain reads the transcript, digests it
    (never dumps a raw line), and picks the moments that land as a clip."""
    lines = [
        "You are picking the best moments to cut into a vertical short "
        "clip. Read the transcript below and the scored candidates, then "
        "choose up to 3 moments that would land as a short video — a clear "
        "hook in the first seconds, something said that stands on its own, "
        "a payoff worth watching. Explain each pick in your own words, "
        "grounded in what was actually said. Never invent words that are "
        "not in the transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Transcript (with seconds):",
        transcript[:6000],
        "",
        "Scored candidates (n, seconds, quality 0-10, why):",
    ]
    for cand in candidates:
        lines.append(
            f"  #{cand.get('n')} {cand.get('start', 0):.0f}s-"
            f"{cand.get('end', 0):.0f}s quality "
            f"{cand.get('quality_score', '?')}/10 — "
            + "; ".join(cand.get("why") or ["no signal"]))
    lines.append("")
    lines.append("Respond with ONLY JSON:")
    lines.append('{"picks": [{"n": <candidate number>, '
                 '"reason": "<why this moment lands, in your own words>"}, '
                 "...]}")
    return "\n".join(lines)


def validate_picks(data: Any,
                   candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the brain's picks: every ``n`` must name a real candidate,
    every reason must be non-empty, duplicates collapse. Anything else is
    dropped, never repaired into a fake pick."""
    if not isinstance(data, dict):
        return []
    known = {int(c.get("n")): c for c in candidates
             if isinstance(c.get("n"), int)}
    picks: list[dict[str, Any]] = []
    seen: set[int] = set()
    for pick in data.get("picks") or []:
        if not isinstance(pick, dict):
            continue
        try:
            n = int(pick.get("n"))
        except (TypeError, ValueError):
            continue
        if n not in known or n in seen:
            continue
        reason = str(pick.get("reason") or "").strip()
        if not reason:
            continue
        seen.add(n)
        picks.append({"n": n, "reason": reason,
                      "candidate": known[n], "source": "brain"})
    return picks[:MAX_PICKS]


def brain_pick_moments(pick_fn: Callable[[str], Any], *,
                       topic: str, transcript: str,
                       candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask the brain to pick moments. Returns ``{"picks": [...],
    "error": str | None}`` — an empty pick list with a named error when
    the brain is unavailable or its answer fails validation."""
    if not candidates:
        return {"picks": [], "error": "no candidates to pick from"}
    prompt = build_pick_prompt(topic=topic, transcript=transcript,
                               candidates=candidates)
    try:
        data = pick_fn(prompt)
    except Exception as exc:  # noqa: BLE001 - a dead brain is a note, not a crash
        return {"picks": [],
                "error": f"brain pick failed: {exc}"}
    if data is None:
        return {"picks": [],
                "error": "no brain available — deterministic picks only"}
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return {"picks": [],
                    "error": "brain returned unparseable JSON"}
    picks = validate_picks(data, candidates)
    if not picks:
        return {"picks": [],
                "error": "brain returned no valid picks"}
    return {"picks": picks, "error": None}


def build_copy_prompt(*, topic: str, title: str,
                      transcript: str) -> str:
    """The platform-copy prompt: short titles, one-line descriptions and
    hashtags for YouTube Shorts, Facebook Reels and TikTok, written from
    the transcript's meaning — never a dump of a raw line."""
    return (
        "Write upload copy for a vertical short clip. Keep every title "
        "under 60 characters, every description to one sentence, and give "
        "up to 5 hashtags per platform. Ground everything in what was "
        "actually said; never invent quotes.\n\n"
        f"Topic: {topic}\nTitle: {title}\n\n"
        "Transcript excerpt:\n"
        f"{transcript[:3000]}\n\n"
        "Respond with ONLY JSON:\n"
        '{"youtube": {"title": "...", "description": "...", '
        '"hashtags": ["..."]}, "facebook": {...}, "tiktok": {...}}'
    )


def brain_platform_copy(copy_fn: Callable[[str], Any], *,
                        topic: str, title: str,
                        transcript: str) -> dict[str, Any]:
    """Ask the brain for per-platform upload copy. Returns a dict keyed by
    platform; an empty dict when the brain is unavailable (the caller
    falls back to deterministic copy)."""
    try:
        data = copy_fn(build_copy_prompt(topic=topic, title=title,
                                         transcript=transcript))
    except Exception:  # noqa: BLE001
        return {}
    if data is None:
        return {}
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for platform, entry in data.items():
        if not isinstance(entry, dict):
            continue
        out[str(platform)] = {
            "title": str(entry.get("title") or "")[:100],
            "description": str(entry.get("description") or "")[:500],
            "hashtags": [str(h)[:40] for h in (entry.get("hashtags") or [])][:8],
        }
    return out
