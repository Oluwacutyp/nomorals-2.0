"""Pre-visit layer — navigation, NOT diagnosis.

Covers the full journey nobody owns: what to do next (triage routing),
what it costs (Nigeria price transparency), what to bring (visit prep),
and what the doctor said (post-visit recap).

CRITICAL POSITIONING, encoded in the code:
- This is NAVIGATION, not diagnosis. Every output says so.
- Conservative taxonomy: OVER-route, never under-route.
  Uncertain → bump a level UP, never down.
- Crisis resources are on EVERY route output. Never buried.
- Devon names NO conditions. Ever. "You should see someone this
  week" — never "this sounds like X."
- Owner-scoped only, like the health timeline.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .timeline import BANNED_PHRASES, HealthTimeline

_log = logging.getLogger(__name__)

__all__ = [
    "ROUTE_LEVELS",
    "NAVIGATION_DISCLAIMER",
    "CRISIS_RESOURCES",
    "Route",
    "VisitRecap",
    "triage_route",
    "format_route",
    "consultation_costs",
    "format_costs",
    "prepare_visit",
    "summarize_visit",
    "PREVISIT_BANNED_PHRASES",
]

#: Every route output carries this framing. Non-negotiable.
NAVIGATION_DISCLAIMER = (
    "This is navigation, not diagnosis — only a clinician can diagnose. "
    "If you're worried, see a doctor; this router errs on the side of "
    "sending you in."
)

#: Route levels, weakest → strongest. Uncertain cases move UP.
ROUTE_LEVELS = ("self_care", "routine_care", "urgent", "emergency")

#: Extra banned phrases for the pre-visit layer (condition-naming,
#: medication advice). Tested in test_previsit.py.
PREVISIT_BANNED_PHRASES = (
    "sounds like",
    "this could be",
    "you might have",
    "you probably have",
    "likely to be",
    "diagnos",
    "prescrib",
    "you should take",
    "i recommend you take",
    "try taking",
    "it's just",
    "nothing to worry about",
    "don't worry about it",
)

#: Crisis resources, always visible on every route output.
#: Verified 2026-10-08 against public Nigerian emergency listings:
#: 112 = national emergency (fire/police/medical, toll-free, nationwide),
#: 199 = national police, 767 = Lagos LASEMA, 122 = FRSC road safety.
CRISIS_RESOURCES = (
    "🚨 Emergency (national, toll-free): 112",
    "🚔 Police emergency: 199",
    "🏙️ Lagos emergency (LASEMA): 767 or 112",
    "🛣️ Road accidents (FRSC): 122",
    "💚 Mental health crisis — Lagos Lifeline (free, confidential, "
    "24/7): 070 0000 6463 · Suicide prevention lines: "
    "08062106493, 08092106493, 09080217555",
)


# ── red-flag detection ────────────────────────────────────────────────────
# Each entry: (compiled pattern, reason string). A hit → emergency.
# Patterns are deliberately broad: false positives route up, which is the
# safe direction.

_RED_FLAGS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(chest\s+(pain|pressure|tightness|ache)|"
                r"pressure\s+in\s+(my\s+|the\s+)?chest)\b", re.I),
     "chest pain or pressure"),
    (re.compile(r"\b(can'?t|cannot|couldn'?t|difficulty|trouble|hard)\s+"
                r"(breathe|breathing)\b|\bshort(ness)?\s+of\s+breath\b|"
                r"\bgasping\b", re.I),
     "difficulty breathing"),
    (re.compile(r"\b(bleeding|blood)\b.*\b(won'?t\s+stop|severe|heavy|"
                r"a lot)\b|\bsevere\s+bleed", re.I),
     "severe or unstoppable bleeding"),
    (re.compile(r"\bsuicid\w*\b|\bkill\s+myself\b|want\s+to\s+die|"
                r"\bend(ing)?\s+my\s+life\b|"
                r"self[-\s]?harm|cutting\s+myself\b", re.I),
     "thoughts of self-harm — please reach out now"),
    (re.compile(r"\bface\s+droop|\barm\s+(weak|numb)|\bslurred\s+speech\b|"
                r"\bsudden\s+(weakness|numbness|confusion|vision\s+loss)\b",
                re.I),
     "possible stroke signs (face/arm/speech — FAST)"),
    (re.compile(r"\b(unconscious|fainted|collapsed|blacked\s+out|seizure|"
                r"convulsion)\b", re.I),
     "loss of consciousness or seizure"),
    (re.compile(r"\bworst\s+headache\b|\bsudden\s+severe\s+headache\b", re.I),
     "sudden worst-ever headache"),
    (re.compile(r"\bthroat\s+(swell|closing|tight)|anaphylaxis|"
                r"\bswollen\s+(tongue|lips|throat)\b", re.I),
     "possible severe allergic reaction"),
    (re.compile(r"\bbaby|infant|newborn\b.*\bfever\b|\bfever\b.*"
                r"\b(baby|infant|newborn)\b", re.I),
     "fever in an infant"),
    (re.compile(r"\bsudden\s+severe\s+(abdominal|stomach|belly)\s+pain\b",
                re.I),
     "sudden severe abdominal pain"),
)

#: Urgent-but-not-emergency patterns.
_URGENT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bfever\b.*\b(39|40|41|high)\b|\bhigh\s+fever\b|"
                r"\bfever\s+of\s+(39|40|41)", re.I),
     "high fever"),
    (re.compile(r"\b(vomit|vomiting)\b.*\b(can'?t\s+keep|repeated|"
                r"persistent)\b|\bdehydrat", re.I),
     "persistent vomiting / dehydration risk"),
    (re.compile(r"\bsevere\s+(pain|ache)\b|\bpain\b.*\b(8|9|10)\s*/\s*10\b|"
                r"\b(8|9|10)\s*/\s*10\s+pain\b", re.I),
     "severe pain"),
    (re.compile(r"\bworsening\b|\bgetting\s+worse\b", re.I),
     "symptoms getting worse"),
)

#: Self-care patterns: mild, short, low severity. Narrow on purpose —
#: anything else defaults UP.
_SELF_CARE_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bmild\s+(headache|cold|cough|sore\s+throat)\b", re.I),
     "mild, common symptom"),
    (re.compile(r"\bslight\s+(headache|pain|ache|discomfort)\b", re.I),
     "slight discomfort"),
)


@dataclass
class Route:
    """The result of triage_route: where to go next, and how urgently."""
    level: str                     # one of ROUTE_LEVELS
    reasons: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    crisis_resources: tuple[str, ...] = CRISIS_RESOURCES
    disclaimer: str = NAVIGATION_DISCLAIMER
    symptoms: list[str] = field(default_factory=list)


_NEXT_STEPS = {
    "emergency": [
        "Call 112 now (or 199 for police) — don't drive yourself if "
        "you feel faint.",
        "If someone is with you, tell them what's happening.",
        "Bring ID, your HMO/insurance card if you have one, and any "
        "meds you're taking.",
    ],
    "urgent": [
        "See a doctor within 24 hours — urgent-care clinic, hospital "
        "casualty/A&E, or your GP if they can fit you in today.",
        "If anything gets worse while you wait, call 112.",
        "Write down when it started and what's changed — your /health "
        "timeline log helps here.",
    ],
    "routine_care": [
        "Book a GP appointment this week — it doesn't sound like an "
        "emergency, but a clinician should look at it.",
        "Log the symptom in /health so you have a timeline to show "
        "the doctor.",
        "If anything changes or worsens, re-run /health route.",
    ],
    "self_care": [
        "Rest, fluids, and monitor — this looks manageable at home "
        "for now.",
        "See a doctor if it lasts more than a few days, worsens, or "
        "worries you — you're never wrong to get checked.",
    ],
}


def _bump_up(level: str) -> str:
    """Move one level toward emergency. Conservative direction only."""
    idx = ROUTE_LEVELS.index(level)
    return ROUTE_LEVELS[min(idx + 1, len(ROUTE_LEVELS) - 1)]


def triage_route(symptoms: list[str], history: Any = None) -> Route:
    """Route symptoms to a care level. Navigation, not diagnosis.

    Conservative: red flags → emergency immediately; anything uncertain
    bumps UP a level, never down. Crisis resources on every output.
    Never names a condition.
    """
    joined = " ".join(symptoms or [])
    text = (joined or "").strip()
    route = Route(level="self_care", symptoms=list(symptoms or []))

    if not text:
        route.level = "routine_care"
        route.reasons.append(
            "no symptoms described — when in doubt, a routine check "
            "is the safe default")
        route.next_steps = list(_NEXT_STEPS["routine_care"])
        return route

    # 1. Red flags → emergency, immediately. No further deliberation.
    for pattern, reason in _RED_FLAGS:
        if pattern.search(text):
            route.level = "emergency"
            route.reasons.append(f"red flag: {reason}")
            # keep scanning so ALL red flags are named, not just the first
    if route.level == "emergency":
        route.next_steps = list(_NEXT_STEPS["emergency"])
        return route

    # 2. Urgent patterns.
    for pattern, reason in _URGENT_PATTERNS:
        if pattern.search(text):
            route.level = "urgent"
            route.reasons.append(f"urgent signal: {reason}")

    # 3. Severity from history (timeline events carry severity 1-5).
    max_sev: int | None = None
    try:
        if history is not None:
            events = history() if callable(history) else history
            for ev in events or []:
                sev = getattr(ev, "severity", None)
                if isinstance(sev, int):
                    max_sev = sev if max_sev is None else max(max_sev, sev)
    except Exception:  # noqa: BLE001 — history is advisory only
        _log.debug("triage history read failed", exc_info=True)
    if max_sev is not None and max_sev >= 4:
        if route.level in ("self_care", "routine_care"):
            route.level = "urgent"
            route.reasons.append(
                f"you rated this {max_sev}/5 — high severity routes up")

    # 4. Self-care only if something mild matched AND nothing else did.
    if route.level == "self_care":
        matched = False
        for pattern, reason in _SELF_CARE_PATTERNS:
            if pattern.search(text):
                route.reasons.append(f"mild signal: {reason}")
                matched = True
                break
        if not matched:
            # Uncertain → bump UP to routine_care. Never stay low on a guess.
            route.level = "routine_care"
            route.reasons.append(
                "couldn't confidently classify as mild — routing up to "
                "routine care (conservative default)")

    route.next_steps = list(_NEXT_STEPS[route.level])
    return route


def format_route(route: Route) -> str:
    """Render a Route for chat. Crisis resources always visible."""
    emoji = {"self_care": "🟢", "routine_care": "🟡",
             "urgent": "🟠", "emergency": "🔴"}.get(route.level, "⚪")
    label = route.level.replace("_", " ").upper()
    lines = [f"{emoji} route: **{label}**"]
    if route.reasons:
        lines.append("why:")
        for r in route.reasons:
            lines.append(f"  • {r}")
    lines.append("next steps:")
    for s in route.next_steps:
        lines.append(f"  • {s}")
    lines.append("")
    lines.append("crisis resources (always here, never buried):")
    for c in route.crisis_resources:
        lines.append(f"  • {c}")
    lines.append("")
    lines.append(route.disclaimer)
    return "\n".join(lines)


# ── price transparency (Nigeria) ──────────────────────────────────────────

@dataclass
class CostEstimate:
    label: str
    low_naira: int
    high_naira: int
    note: str = ""


def consultation_costs() -> list[CostEstimate]:
    """Typical Nigerian private-hospital costs. APPROXIMATE — verify.

    Ballpark ranges for planning, not quotes. Public hospitals and HMO
    cover are usually cheaper; Lagos/Abuja trend higher.
    """
    return [
        CostEstimate("GP consultation (private hospital)",
                     5_000, 15_000,
                     "public hospital often ₦1,000–₦3,000; HMO may cover"),
        CostEstimate("Specialist consultation",
                     15_000, 40_000,
                     "cardiology, neurology etc. trend higher"),
        CostEstimate("Basic labs (malaria test, FBC, urinalysis)",
                     3_000, 12_000,
                     "panel tests cost more; ask for an itemized list"),
        CostEstimate("Casualty/A&E visit (private)",
                     20_000, 60_000,
                     "before treatment costs; HMO changes this a lot"),
        CostEstimate("Chest X-ray", 8_000, 20_000, ""),
        CostEstimate("Ultrasound scan", 10_000, 30_000,
                     "depends on type and facility"),
    ]


def format_costs(costs: list[CostEstimate] | None = None) -> str:
    costs = costs if costs is not None else consultation_costs()
    lines = ["💰 typical private-hospital costs (Nigeria) — "
             "**approximate, verify with the facility:**"]
    for c in costs:
        lines.append(f"• {c.label}: ₦{c.low_naira:,}–₦{c.high_naira:,}"
                     + (f" — {c.note}" if c.note else ""))
    lines.append("")
    lines.append("these are ballparks for planning, not quotes. Public "
                 "hospitals are cheaper; HMO cover changes everything. "
                 "Always confirm before you go.")
    lines.append(NAVIGATION_DISCLAIMER)
    return "\n".join(lines)


# ── visit prep ────────────────────────────────────────────────────────────

def prepare_visit(symptoms: list[str] | None = None, *,
                  timeline: HealthTimeline | None = None,
                  days: int = 30) -> str:
    """Build the pre-visit packet: timeline + questions + what to bring."""
    tl = timeline
    own = False
    if tl is None:
        try:
            tl = HealthTimeline()
            own = True
        except Exception:  # noqa: BLE001
            tl = None
    try:
        if tl is not None:
            recap = tl.summary(days=days)
        else:
            recap = "timeline unavailable."
    finally:
        if own and tl is not None:
            tl.close()

    lines = ["📋 **for the doctor** — your pre-visit packet:"]
    lines.append("")
    lines.append("your recent log:")
    lines.append(recap)
    lines.append("")
    if symptoms:
        lines.append("what you're going in about:")
        for s in symptoms:
            lines.append(f"  • {s}")
        lines.append("")
    lines.append("questions to ask:")
    lines.append("  • What do you think is going on, and what else could "
                 "it be?")
    lines.append("  • What tests do I need, and what will they tell us?")
    lines.append("  • What should I watch for that means 'come back "
                 "sooner'?")
    lines.append("  • Are there side effects I should know about for "
                 "any medication?")
    lines.append("")
    lines.append("bring:")
    lines.append("  • ID and your HMO/insurance card (if you have one)")
    lines.append("  • list of current medications and doses")
    lines.append("  • this timeline summary (screenshot or /health summary)")
    lines.append("")
    lines.append(NAVIGATION_DISCLAIMER)
    return "\n".join(lines)


# ── post-visit recap ──────────────────────────────────────────────────────

@dataclass
class VisitRecap:
    """What the doctor said, as the USER reported it.

    Every field quotes the user's own notes. Devon adds no medical
    interpretation — it structures, it never diagnoses.
    """
    raw_notes: str
    doctor_said: list[str] = field(default_factory=list)
    meds_reported: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    logged_event_id: str = ""


_DOCTOR_SAID_RE = re.compile(
    r"(?:doctor|dr\.?|she|he)\s+(?:said|told\s+me|mentioned|explained|"
    r"diagnosed|thinks?)\s+(.+?)(?:\.|$)", re.I)

_MED_RE = re.compile(
    r"(?:prescribed|gave\s+me|put\s+me\s+on|take)\s+(.+?)(?:\.|$)", re.I)

_FOLLOWUP_RE = re.compile(
    r"(?:come\s+back|return|follow[\s-]?up|review|see\s+(?:me|him|her))\s+"
    r"(?:in\s+|after\s+|on\s+)?(.+?)(?:\.|$)", re.I)


def _clean_sentences(matches: list[str]) -> list[str]:
    out: list[str] = []
    for m in matches:
        s = (m or "").strip().rstrip(".")
        if s and s not in out:
            out.append(s)
    return out


def summarize_visit(notes: str, *,
                    timeline: HealthTimeline | None = None) -> VisitRecap:
    """Structure the user's visit notes. Quotes the user — never diagnoses.

    The recap reports what the DOCTOR said as the user wrote it down.
    Logged to the health timeline as a ``visit`` event.
    """
    raw = (notes or "").strip()
    recap = VisitRecap(raw_notes=raw)
    if not raw:
        return recap

    # Split into sentences for extraction; always quote, never interpret.
    sentences = re.split(r"(?<=[.!?])\s+", raw)
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        dm = _DOCTOR_SAID_RE.search(s)
        if dm:
            recap.doctor_said.extend(_clean_sentences([dm.group(1)]))
        mm = _MED_RE.search(s)
        if mm:
            recap.meds_reported.extend(_clean_sentences([mm.group(1)]))
        fm = _FOLLOWUP_RE.search(s)
        if fm:
            recap.follow_ups.extend(_clean_sentences([fm.group(0).strip()]))

    # Log to the timeline as a visit event (owner-scoped, like everything).
    tl = timeline
    own = False
    if tl is None:
        try:
            tl = HealthTimeline()
            own = True
        except Exception:  # noqa: BLE001
            tl = None
    try:
        if tl is not None:
            ev = tl.log("visit", raw, source="chat",
                        extra={"recap": True})
            recap.logged_event_id = ev.id
    finally:
        if own and tl is not None:
            tl.close()
    return recap


def format_recap(recap: VisitRecap) -> str:
    """Render a VisitRecap for chat — the doctor's words, structured."""
    if not recap.raw_notes:
        return "usage: /health visited <what the doctor said>"
    lines = ["🩺 **visit recap** — what you told me the doctor said:"]
    if recap.doctor_said:
        lines.append("the doctor said (your words):")
        for d in recap.doctor_said:
            lines.append(f"  • \"{d}\"")
    else:
        lines.append("(no doctor statements picked out — your full notes "
                     "are saved below)")
    if recap.meds_reported:
        lines.append("meds mentioned (as you reported — this is not "
                     "medical advice):")
        for m in recap.meds_reported:
            lines.append(f"  • \"{m}\"")
    if recap.follow_ups:
        lines.append("follow-ups:")
        for f in recap.follow_ups:
            lines.append(f"  • \"{f}\"")
    lines.append("")
    lines.append(f"full notes saved to your timeline "
                 f"{'(id ' + recap.logged_event_id + ')' if recap.logged_event_id else ''}.")
    lines.append("this is your record of what the doctor said — only "
                 "your doctor can interpret it.")
    return "\n".join(lines)


# ── safety: scan our own outputs ──────────────────────────────────────────

def _check_banned(text: str) -> list[str]:
    """Return banned phrases found in text (for tests + self-check).

    The required navigation disclaimer legitimately contains the word
    "diagnosis" ("only a clinician can diagnose") — it is stripped before
    scanning. Double-quoted regions are the USER's verbatim words (e.g.
    what they reported the doctor said/prescribed) — the ban targets
    Devon's voice, so quotes are stripped too.
    """
    hay = (text or "").lower()
    hay = hay.replace(NAVIGATION_DISCLAIMER.lower(), "")
    hay = re.sub(r'"[^"]*"', "", hay)
    hits: list[str] = []
    for phrase in BANNED_PHRASES + PREVISIT_BANNED_PHRASES:
        if phrase in hay:
            hits.append(phrase)
    return hits
