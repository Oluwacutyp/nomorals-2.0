"""Beat resolution for exploration sweeps.

A *beat* is one field a sweep watches: a topic phrase plus the
weight the owner's interest profile gives it. Beats are resolved,
in order, from:

1. an **injected profile** — the real ``NewsAgent.interest_profile()``
   dict (``{beats, pins, beats_text}``), when a caller above the
   media layer supplies it;
2. the **persisted news profile** read from the context db — the
   same state the NewsAgent itself reads (its saved profile wins;
   absent a saved row, the seed beats are the profile);
3. the **seed beats** alone, when there is no database at all.

``agents.news`` is L5, above media's L4, so the seed tuple and the
profile loader are mirrored here — the exact pattern
``goals/providers.py`` established for the same boundary (its
``_SEED_BEATS`` / ``_load_news_profile``). The WS4 test suite pins
this mirror against the real tuple so it can never drift silently.

On top of the profile, two more seed sources fold in — sources are
seeds, never a closed list:

* the owner's **stated briefing topics** (``settings.briefing.topics``
  — the same list the NewsAgent folds into its effective profile),
  origin ``briefing``;
* the **WS2 interests-class watches**: topics declared by active
  interests-class goals (``watch:`` tags, else the goal's title —
  the ``goals.providers.interest_topics`` rule, read straight from
  the goals table), origin ``watch``.

Every beat carries its ``origin`` so a sweep report can say where
each field it watched came from.

**The trading beat pack (WS4 P6).** The census's §8(a) obligation:
exploration carries a trading beat — XAUUSD / gold-trading video
topics are first-class seeds, so the trend/niche findings cover the
field the owner's trading life runs on (GoldenSwing gains no video
system; the beat is Devon-side data). :data:`TRADING_SEED_BEATS`
is exploration's OWN seed data — no NewsAgent counterpart exists
(the news seeds are news-interest seeds), so no mirror pin applies;
the P6 suite pins the tuple itself, so edits are deliberate. The
pack folds in under origin ``trading`` exactly when the beat base
is the shipped seed profile (no database, or a persisted profile
with no saved beat list in force): a saved profile replaces the
shipped seed profile wholesale — trading pack included — exactly
as it replaces the news seeds (the P1 rule, extended). An injected
profile is likewise taken verbatim. The pack rides the sweep
machinery unchanged: same caps, same instruments, no engine
special-casing.
"""

from __future__ import annotations

import json
from typing import Any

from ...core.logging_setup import get_logger
from ..failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "EXPLORE_SEED_BEATS",
    "TRADING_SEED_BEATS",
    "beat_origin",
    "beats_from_profile",
    "collect_watch_topics",
    "load_beats",
    "load_news_profile",
]

#: Mirror of ``agents.news.INTEREST_SEED_BEATS`` — the fallback
#: beats when no profile was ever saved. agents/ is L5, above
#: media's L4, so the constant cannot be imported here; the WS4 P1
#: test suite pins this mirror against the real tuple.
EXPLORE_SEED_BEATS: tuple[tuple[str, float], ...] = (
    ("XAUUSD and gold trading", 1.0),
    ("AI and model work", 0.9),
    ("Nigerian tech and Lagos", 0.9),
    ("Devon herself — the owner's own agent and her systems", 1.0),
    ("USA money opportunities, including loans", 0.8),
)

#: The trading beat pack (WS4 P6, census §8(a)): durable VIDEO
#: search topics for the field the owner's trading life runs on —
#: gold/XAUUSD price-action education, strategy and smart-money
#: teaching, the macro forces that move gold, and session/weekly
#: market recaps. Queries chosen to still make sense in a year:
#: methods, instruments and institutions, never this week's
#: ticker or headline. Exploration's own seed data (no L5 mirror
#: exists); folded in by :func:`load_beats` under origin
#: ``trading`` on the seed-fallback paths only — see the module
#: docstring for the replacement rule.
TRADING_SEED_BEATS: tuple[tuple[str, float], ...] = (
    ("XAUUSD price action analysis", 1.0),
    ("gold trading strategies and setups", 0.9),
    ("smart money concepts ICT gold trading", 0.9),
    ("central banks, interest rates and the gold price", 0.8),
    ("gold trading session recaps and weekly outlook", 0.8),
)

_TRADING_TOPICS = frozenset(
    t.casefold() for t, _w in TRADING_SEED_BEATS)


def beat_origin(topic: Any) -> str:
    """The shipped-pack origin a beat topic belongs to:
    ``"trading"`` when the topic is one of the trading pack's
    topics, else ``""`` (no pack claims it).

    Claims record their beat (subject/scope), not the origin the
    beat resolved under at sweep time, so a claim's beat origin is
    re-derived from the pack data — deterministically, from the
    same constant the sweep folds — never guessed. The WS4 P5
    feed adapter uses this to keep a trading-beat claim
    identifiable as one all the way into a draft's metadata."""
    key = str(topic or "").strip().casefold()
    return "trading" if key in _TRADING_TOPICS else ""


def beats_from_profile(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalise a NewsAgent-shaped profile dict into beat rows:
    ``[{topic, weight, origin}]`` — profile beats first (origin
    ``profile``), then pins (origin ``pin``, weight 1.0 — the
    NewsAgent renders pins as always-high interest). Deduped
    case-insensitively, first occurrence wins."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(topic: Any, weight: Any, origin: str) -> None:
        topic = str(topic or "").strip()
        if not topic or topic.lower() in seen:
            return
        seen.add(topic.lower())
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 0.5
        out.append({"topic": topic, "weight": w, "origin": origin})

    if isinstance(profile, dict):
        for beat in profile.get("beats") or []:
            if isinstance(beat, dict):
                _add(beat.get("topic"), beat.get("weight", 0.5), "profile")
            elif isinstance(beat, str):
                _add(beat, 0.5, "profile")
        for pin in profile.get("pins") or []:
            _add(pin, 1.0, "pin")
    return out


def load_news_profile(db: Any) -> dict[str, Any] | None:
    """The news interest profile ``{beats, pins}``, NewsAgent-shaped.

    Mirrors ``NewsAgent._load_profile`` (via the same mirror
    ``goals.providers._load_news_profile`` keeps): the saved row
    wins; absent a row, the seeds are the profile. ``None`` when
    there is no database at all. A corrupt saved value falls back
    to the seeds, exactly as the NewsAgent falls back."""
    profile, _saved_beats = _load_news_profile_state(db)
    return profile


def _load_news_profile_state(
        db: Any) -> tuple[dict[str, Any] | None, bool]:
    """:func:`load_news_profile` plus one fact: whether the
    profile's BEAT LIST came from a saved row (``True``) or is
    still the shipped seeds (``False`` — no row, a corrupt row, or
    a saved row that named no beats). :func:`load_beats` needs
    the distinction: the trading pack folds only while the
    shipped seed beats are the beat base."""
    if db is None:
        return None, False
    profile: dict[str, Any] = {
        "beats": [{"topic": t, "weight": w} for t, w in EXPLORE_SEED_BEATS],
        "pins": [],
    }
    saved_beats = False
    raw: Any = None
    try:
        from ...storage.rowstore import attached_rows
        profile_t = attached_rows(db, "news_interest")
    except Exception:  # noqa: BLE001 — fall through to the SQL rung
        profile_t = None
    if profile_t is not None:
        row = profile_t.get("profile")
        raw = (row or {}).get("value")
    else:
        try:
            row = db.query_one(
                "SELECT value FROM news_interest WHERE key = ?",
                ("profile",))
        except Exception as exc:  # noqa: BLE001 — table may not exist
            if "no such table" not in str(exc).lower():
                record_failure("explore.load_news_profile", exc)
            row = None
        raw = (row or {}).get("value")
    if raw:
        try:
            saved = json.loads(raw)
        except (TypeError, ValueError):
            saved = None
        if isinstance(saved, dict):
            beats = saved.get("beats")
            if isinstance(beats, list) and beats:
                profile["beats"] = [
                    {"topic": str(b.get("topic") or ""),
                     "weight": float(b.get("weight") or 0.5)}
                    for b in beats
                    if isinstance(b, dict) and b.get("topic")]
                saved_beats = bool(profile["beats"])
            pins = saved.get("pins")
            if isinstance(pins, list):
                profile["pins"] = [str(p) for p in pins if str(p).strip()]
    return profile, saved_beats


def _briefing_topics(settings: Any = None) -> list[str]:
    """The owner's stated briefing topics — the same list the
    NewsAgent folds into its effective profile."""
    if settings is None:
        try:
            from ...core.config import get_settings
            settings = get_settings()
        except Exception:  # noqa: BLE001 — settings are an enhancement
            return []
    briefing = getattr(settings, "briefing", None)
    return [str(t).strip() for t in
            (getattr(briefing, "topics", None) or []) if str(t).strip()]


def collect_watch_topics(db: Any) -> list[str]:
    """Watch topics declared by active interests-class goals.

    The rule mirrors ``goals.providers.interest_topics``: a goal's
    ``watch:`` tags name its topics; with none, its title is its
    topic. Read straight from the goals table (the tracker is async
    and schema-owning; a sweep seed read must be neither). Any
    failure — no table yet, no class column yet — yields no watches:
    watches deepen the seed set, they never gate it. Never raises."""
    if db is None:
        return []
    try:
        rows = db.query(
            "SELECT title, tags FROM goals "
            "WHERE goal_class = ? AND status = ?",
            ("interests", "active")) or []
    except Exception:  # noqa: BLE001 — goals organ absent/older schema
        return []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        topics: list[str] = []
        try:
            tags = json.loads((row or {}).get("tags") or "[]")
        except (TypeError, ValueError):
            tags = []
        if isinstance(tags, list):
            topics = [str(t)[len("watch:"):].strip() for t in tags
                      if isinstance(t, str)
                      and t.lower().startswith("watch:")
                      and t[len("watch:"):].strip()]
        if not topics:
            title = str((row or {}).get("title") or "").strip()
            if title:
                topics = [title]
        for topic in topics:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                out.append(topic)
    return out


def load_beats(db: Any = None, settings: Any = None, *,
               profile: dict[str, Any] | None = None,
               watches: list[str] | None = None) -> list[dict[str, Any]]:
    """Resolve the sweep beat list: ``[{topic, weight, origin}]``.

    ``profile`` (an injected NewsAgent profile dict) wins over the
    persisted profile, which wins over the bare seeds. While the
    beat base is the shipped seed profile (no injected profile;
    no saved beat list in force), the trading pack folds in under
    origin ``trading`` — a saved profile replaces the shipped
    seeds wholesale, pack included. Briefing topics and interests
    watches fold in after, each under its own origin; explicit
    ``watches`` (a list of topic strings) replaces the db watch
    read when given (tests, L5 callers)."""
    if profile is not None:
        beats = beats_from_profile(profile)
        seed_based = False
    else:
        saved, saved_beats = _load_news_profile_state(db)
        if saved is not None:
            beats = beats_from_profile(saved)
            seed_based = not saved_beats
        else:
            beats = [{"topic": t, "weight": w, "origin": "seed"}
                     for t, w in EXPLORE_SEED_BEATS]
            seed_based = True
    seen = {b["topic"].lower() for b in beats}
    if seed_based:
        for topic, weight in TRADING_SEED_BEATS:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                beats.append({"topic": topic, "weight": weight,
                              "origin": "trading"})

    def _fold(topic: Any, origin: str) -> None:
        topic = str(topic or "").strip()
        if topic and topic.lower() not in seen:
            seen.add(topic.lower())
            beats.append({"topic": topic, "weight": 1.0, "origin": origin})

    for topic in _briefing_topics(settings):
        _fold(topic, "briefing")
    if watches is None:
        watches = collect_watch_topics(db)
    for topic in watches:
        _fold(topic, "watch")
    return beats
