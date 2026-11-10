"""The /worldmodel board — the world model, rendered for its owner.

The render reads the model's RECORDED state only: every line traces
to a fact the WorldModel holds, with the fact's own provenance kind,
its source, its version and its age.  An area with no facts does not
appear (it is not rendered as a zero-state claim about the world);
a fact older than a day is labelled stale; an invalidated fact shows
its tombstone.  Internal ids never print — work facts render by the
goal and step their values carry, never by run or contract id.
"""

from __future__ import annotations

import time

from .worldmodel import FactSnapshot, WorldModel

__all__ = ["render_world", "render_topic", "STALE_AFTER_S"]

#: A fact not re-asserted within a day is labelled stale on the board
#: (a presentation label over the recorded age, stated in the header).
STALE_AFTER_S = 86400.0

_KIND_TAGS = {
    "observed": "observed",
    "inferred": "inferred",
    "proposed": "proposed",
    "unverified": "unverified",
}


def _age_text(updated_at: float, now: float) -> str:
    age = max(0.0, now - float(updated_at or now))
    if age < 60:
        return "just now"
    if age < 3600:
        return f"{int(age // 60)}m ago"
    if age < 86400:
        return f"{int(age // 3600)}h ago"
    return f"{int(age // 86400)}d ago"


def _is_stale(fact: FactSnapshot, now: float) -> bool:
    return (now - float(fact.updated_at or now)) > STALE_AFTER_S


def _subject(fact: FactSnapshot) -> str:
    """The owner-facing name of a fact: work facts by their goal and
    step (from the value — never the run/contract-keyed id), anything
    else by its key's subject part."""
    value = fact.value
    if isinstance(value, dict) and value.get("goal"):
        goal = str(value["goal"])
        step = str(value.get("step") or "")
        return f"{goal} — {step}" if step else goal
    key = fact.key
    return key.split(":", 1)[1] if ":" in key else key


def _summary(fact: FactSnapshot) -> str:
    value = fact.value
    if isinstance(value, dict):
        outcome = str(value.get("outcome") or "")
        parts: list[str] = []
        if outcome:
            parts.append(outcome)
        reasons = value.get("reasons")
        if isinstance(reasons, (list, tuple)) and reasons:
            parts.append("; ".join(str(r) for r in reasons))
        detail = str(value.get("detail") or "")
        if detail:
            parts.append(detail)
        if parts:
            return " — ".join(parts)[:300]
        pairs = [f"{k}={v}" for k, v in list(value.items())[:4]]
        return ", ".join(pairs)[:300]
    if value is None:
        return "(no value recorded)"
    return str(value)[:300]


def _fact_line(fact: FactSnapshot, now: float) -> str:
    kind = _KIND_TAGS.get(fact.kind, fact.kind)
    line = (f"• {_subject(fact)} [{kind}] — {_summary(fact)}"
            f" · v{fact.version} · {_age_text(fact.updated_at, now)}")
    if _is_stale(fact, now):
        line += " (stale)"
    if fact.source:
        line += f" · from {fact.source}"
    if fact.invalidated:
        reason = fact.invalidate_reason or "no reason recorded"
        line += f" · INVALIDATED: {reason}"
    return line


def render_world(world: WorldModel, *, now: float | None = None) -> str:
    """The whole map: header (version, size), then per-area sections
    in key order.  An empty model renders the honest empty picture —
    an empty map is an absence of record, never a claim that nothing
    is true."""
    now = time.time() if now is None else now
    facts = world.facts()
    if not facts:
        return ("🌍 WORLD MODEL — empty\n"
                "The world model holds no facts yet (map version 0). "
                "Facts land here as work settles through the plan "
                "seam and other writers assert with provenance. An "
                "empty map is an absence of record — it says nothing "
                "about the world itself.")
    areas: dict[str, list[FactSnapshot]] = {}
    for fact in facts:
        area = fact.key.split(":", 1)[0] if ":" in fact.key else "general"
        areas.setdefault(area, []).append(fact)
    lines = [
        f"🌍 WORLD MODEL — map version {world.version()} · "
        f"{len(facts)} fact(s) across {len(areas)} area(s)",
        "Kinds: observed = directly seen · inferred = derived · "
        "proposed = intent, not yet reality · unverified = asserted, "
        "not yet checked. A fact unrefreshed for a day is labelled "
        "stale.",
    ]
    for area in sorted(areas):
        lines.append(f"\n[{area}]")
        for fact in areas[area]:
            lines.append(_fact_line(fact, now))
    return "\n".join(lines)


def render_topic(world: WorldModel, topic: str, *,
                 now: float | None = None) -> str:
    """/worldmodel <topic> — every fact under one key prefix, in full:
    value fields, provenance, version history depth.  A prefix with
    no facts says so in words — absence of record, not a claim."""
    now = time.time() if now is None else now
    topic = (topic or "").strip()
    facts = world.facts(prefix=topic)
    if not facts:
        return (f"🌍 WORLD — {topic}\n"
                "No facts recorded under that name. The map holds "
                "nothing there — an absence of record, not a claim "
                "about the world.")
    lines = [f"🌍 WORLD — {topic} · {len(facts)} fact(s)"]
    for fact in facts:
        lines.append("")
        lines.append(_fact_line(fact, now))
        if isinstance(fact.value, dict):
            for k, v in fact.value.items():
                lines.append(f"   {k}: {v}")
        history = world.history(fact.key)
        lines.append(f"   versions on record: {len(history)} · "
                     f"asserted by {fact.author or 'worldmodel'}")
    return "\n".join(lines)
