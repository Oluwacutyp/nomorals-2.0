"""Topic engine for the always-on research loop (Wave E, phase 2026-10-21).

The loop was starving: owner seeds capped at 12, 4 evergreen fallbacks,
at most 4 KG contradicted follow-ups, and a hard last-3-runs
anti-repeat wall — a finite pool that empties, and once empty the loop
skips with "no fresh topics". This organ is the third topic source,
feeding :func:`research_loop.pick_topics` beside owner seeds and
contradicted follow-ups:

* (a) open questions from findings — the research organ's durable
  ``knowledge_gaps`` (open questions noted during research runs and
  via the ``research`` tool), plus question sentences carried by KG
  claims;
* (b) KG gaps and staleness — claims older than ``_STALE_AFTER_DAYS``
  with no re-confirmation become "re-check" topics (contradicted
  claims stay with the loop's own follow-ups, never duplicated here);
* (c) world signals — the enabled ``research_watches`` (money-watch +
  builder-watch and any hand-added watch): their queries are topics
  the loop already sees, kept live instead of re-seeded by hand;
* (d) work-generated questions — sites that keep failing in the
  failure ledger (one failure is noise, a repeat is a question) and
  upgrade proposals the owner denied or ignored past the ignore
  window (a denied idea is still a question worth re-researching).

Every harvester is fail-soft: a missing table or a raising database
returns ``[]`` for that source, never a crash in the loop's path.

The staleness curve (owner order 2026-10-21, built 2026-11-17) lives
here too: :func:`staleness_eligibility` maps a topic's age since last
coverage to an eligibility in ``[_STALENESS_FLOOR, 1.0]`` — fresh
topics win, a just-covered topic ranks last, nothing is ever hard
excluded. Coverage is read from ``research_loop_runs`` (the loop's
own run history), so the curve needs no second record.

The durable topic ledger (kv ``topic_engine.topics``) records every
topic the engine surfaces, with provenance (where it came from) and
dedupe: a normalized-exact or token-Jaccard (>= 0.65, the same
threshold as the loop's proposal dedupe) near-duplicate never enters
twice. The ledger is bounded (``LEDGER_CAP``); eviction drops the
lowest-priority, oldest entries first. ``suggest()`` ranks by
priority, then least-suggested, then oldest — ``note_picked()``
records exactly the topics the loop actually picked, so a topic the
engine offered but the loop skipped is not penalised for it.

Stdlib only. Nothing here approves or applies anything — the engine
only supplies topics; the loop still files tickets into the upgrade
queue for the owner's word.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from ..core.logging_setup import get_logger
from ..storage.kv import KVStore

_log = get_logger(__name__)

__all__ = [
    "LEDGER_CAP",
    "STALENESS_FLOOR",
    "STALENESS_FULL_DAYS",
    "TopicEngine",
    "harvest",
    "harvest_failures",
    "harvest_gaps",
    "harvest_kg_stale",
    "harvest_proposals",
    "harvest_watches",
    "load_ledger",
    "normalize_topic",
    "save_ledger",
    "staleness_eligibility",
    "topic_ages",
    "topics_duplicate",
]

#: kv_store key holding the engine's durable topic ledger (JSON list).
_STORE_KEY = "topic_engine.topics"

#: Bounded ledger: enough to remember a season of suggestions, small
#: enough to stay a cheap kv read on every tick.
LEDGER_CAP = 200

#: Token overlap that marks two topics the same question.
_DEDUP_JACCARD = 0.65

#: A KG claim with no re-confirmation after this many days is stale.
_STALE_AFTER_DAYS = 30.0

#: An ignored/denied proposal becomes a follow-up question after this
#: many days (the loop's own ignore window, mirrored here so the two
#: never disagree by more than a day).
_IGNORE_AFTER_DAYS = 7.0

#: Staleness curve: a topic is fully eligible again after this many
#: days since last coverage...
STALENESS_FULL_DAYS = 21.0
#: ...and never below this eligibility, so the pool can never empty.
STALENESS_FLOOR = 0.15

#: Per-source harvest caps (bounded per tick so one source can't flood
#: the ledger).
_HARVEST_LIMITS = {"gap": 20, "kg_stale": 20, "watch": 20,
                   "failure": 10, "proposal": 10}

#: A repeated failure site is a question only from this many hits —
#: one failure is noise.
_FAILURE_MIN_REPEATS = 2


# ── normalization / dedupe ───────────────────────────────────────────────

def normalize_topic(text: str) -> str:
    """Lowercased, whitespace-collapsed, punctuation-stripped topic key."""
    clean = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return " ".join(clean.split())


def _topic_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", str(text or "").lower()))


def topics_duplicate(a: str, b: str,
                     threshold: float = _DEDUP_JACCARD) -> bool:
    """True when two topic wordings ask the same question: identical
    once normalized, or token overlap at/above the threshold."""
    na, nb = normalize_topic(a), normalize_topic(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = _topic_tokens(a), _topic_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


# ── staleness curve ──────────────────────────────────────────────────────

def staleness_eligibility(age_days: float | None) -> float:
    """Eligibility of a topic last covered ``age_days`` ago (``None`` =
    never covered). 1.0 fresh, decaying linearly to
    ``STALENESS_FLOOR`` at age 0 — never zero, never a wall."""
    if age_days is None:
        return 1.0
    frac = min(1.0, max(0.0, float(age_days)) / STALENESS_FULL_DAYS)
    return STALENESS_FLOOR + (1.0 - STALENESS_FLOOR) * frac


def topic_ages(context: Any, now: float | None = None) -> dict[str, float]:
    """Days since each topic was last covered, from the loop's own run
    history (``research_loop_runs``). Never-covered topics are absent.
    ``{}`` when there is no database or no history — never raises."""
    db = getattr(context, "db", None)
    if db is None:
        return {}
    moment = time.time() if now is None else float(now)
    try:
        rows = db.query(
            "SELECT topics, started_at FROM research_loop_runs "
            "ORDER BY started_at DESC LIMIT 200")
    except Exception:  # noqa: BLE001 - migration not applied yet
        return {}
    ages: dict[str, float] = {}
    for row in rows or []:
        try:
            started = float(row.get("started_at") or 0.0)
            topics = json.loads(row.get("topics") or "[]")
        except (TypeError, ValueError):
            continue
        age = max(0.0, (moment - started) / 86400.0)
        for topic in topics or []:
            key = normalize_topic(topic)
            if key and (key not in ages or age < ages[key]):
                ages[key] = age
    return ages


# ── durable ledger ───────────────────────────────────────────────────────

def _kv_get(context: Any, key: str) -> Any:
    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        return KVStore(db).get(key)
    except Exception:  # noqa: BLE001 - kv is best-effort
        return None


def _kv_put(context: Any, key: str, value: Any) -> bool:
    db = getattr(context, "db", None)
    if db is None:
        return False
    try:
        KVStore(db).set(key, value)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("kv put failed (%s): %s", key, exc)
        return False


def load_ledger(context: Any) -> list[dict[str, Any]]:
    """The engine's topic ledger (``[]`` when empty or unreadable)."""
    stored = _kv_get(context, _STORE_KEY)
    if not isinstance(stored, list):
        return []
    out: list[dict[str, Any]] = []
    for row in stored:
        if not isinstance(row, dict) or not str(row.get("text") or "").strip():
            continue
        out.append({
            "text": str(row["text"]),
            "source": str(row.get("source") or "unknown"),
            "provenance": str(row.get("provenance") or ""),
            "priority": float(row.get("priority") or 0.5),
            "first_seen_at": float(row.get("first_seen_at") or 0.0),
            "last_suggested_at": float(row.get("last_suggested_at") or 0.0),
            "times_suggested": int(row.get("times_suggested") or 0),
        })
    return out


def save_ledger(context: Any, entries: list[dict[str, Any]]) -> bool:
    """Persist the ledger (bounded to ``LEDGER_CAP``)."""
    return _kv_put(context, _STORE_KEY, list(entries or [])[:LEDGER_CAP])


def _evict(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the lowest-priority, oldest entries past the cap."""
    if len(entries) <= LEDGER_CAP:
        return entries
    ranked = sorted(entries,
                    key=lambda e: (e["priority"], e["first_seen_at"]))
    return ranked[len(ranked) - LEDGER_CAP:]


# ── harvesters (each fail-soft) ──────────────────────────────────────────

def harvest_gaps(context: Any) -> list[dict[str, Any]]:
    """(a) Open questions: the research organ's ``knowledge_gaps``."""
    db = getattr(context, "db", None)
    if db is None:
        return []
    try:
        rows = db.query(
            "SELECT id, question, priority FROM knowledge_gaps "
            "WHERE status = 'open' ORDER BY priority DESC, ts ASC LIMIT ?",
            (_HARVEST_LIMITS["gap"],))
    except Exception:  # noqa: BLE001 - table may not exist yet
        return []
    out: list[dict[str, Any]] = []
    for row in rows or []:
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        out.append({"text": question, "source": "gap",
                    "provenance": f"knowledge_gap#{row.get('id')}",
                    "priority": float(row.get("priority") or 0.5)})
    return out


def harvest_kg_stale(context: Any,
                     now: float | None = None) -> list[dict[str, Any]]:
    """(b) KG gaps/staleness: claims with no re-confirmation after
    ``_STALE_AFTER_DAYS``. Contradicted claims are skipped — the loop's
    own follow-ups already chase those."""
    db = getattr(context, "db", None)
    if db is None:
        return []
    moment = time.time() if now is None else float(now)
    try:
        rows = db.query(
            "SELECT id, properties, created_at FROM kg_nodes "
            "WHERE type = 'claim' ORDER BY created_at ASC LIMIT 200")
    except Exception:  # noqa: BLE001 - table may not exist yet
        return []
    out: list[dict[str, Any]] = []
    for row in rows or []:
        try:
            props = json.loads(row.get("properties") or "{}")
        except (TypeError, ValueError):
            continue
        if str(props.get("status", "")) == "contradicted":
            continue
        created = float(row.get("created_at") or 0.0)
        if (moment - created) / 86400.0 < _STALE_AFTER_DAYS:
            continue
        query = str(props.get("query") or "").strip()
        if query:
            text = query
        else:
            claim = str(props.get("claim") or "").strip()
            if not claim:
                continue
            text = f"re-check KG claim: {claim[:120]}"
        out.append({"text": text, "source": "kg_stale",
                    "provenance": f"kg_claim:{row.get('id')}",
                    "priority": 0.7})
        if len(out) >= _HARVEST_LIMITS["kg_stale"]:
            break
    return out


def harvest_watches(context: Any) -> list[dict[str, Any]]:
    """(c) World signals: queries of the enabled research watches."""
    db = getattr(context, "db", None)
    if db is None:
        return []
    try:
        rows = db.query(
            "SELECT id, queries FROM research_watches WHERE enabled = 1 "
            "ORDER BY created_at DESC LIMIT 20")
    except Exception:  # noqa: BLE001 - table may not exist yet
        return []
    out: list[dict[str, Any]] = []
    for row in rows or []:
        try:
            queries = json.loads(row.get("queries") or "[]")
        except (TypeError, ValueError):
            continue
        for query in queries or []:
            text = str(query or "").strip()
            if not text:
                continue
            out.append({"text": text, "source": "watch",
                        "provenance": f"research_watch:{row.get('id')}",
                        "priority": 0.6})
            if len(out) >= _HARVEST_LIMITS["watch"]:
                return out
    return out


def harvest_failures(context: Any) -> list[dict[str, Any]]:
    """(d) Work-generated questions: failure-ledger sites that keep
    failing. One failure is noise; a repeat is a research question."""
    try:
        from .failures import recent_failures
    except Exception:  # noqa: BLE001 - ledger import must never break harvest
        return []
    try:
        items = recent_failures(20)
    except Exception:  # noqa: BLE001 - ledger read is best-effort
        return []
    by_site: dict[str, list[Any]] = {}
    for item in items or []:
        by_site.setdefault(str(item.site), []).append(item)
    out: list[dict[str, Any]] = []
    for site, hits in sorted(by_site.items()):
        if len(hits) < _FAILURE_MIN_REPEATS:
            continue
        latest = hits[0]
        message = str(getattr(latest, "message", "") or "").strip()
        suffix = f" — {message[:80]}" if message else ""
        out.append({
            "text": (f"recurring failure at {site}: "
                     f"{getattr(latest, 'exc_type', '?')}{suffix}"),
            "source": "failure",
            "provenance": f"failure_site:{site}",
            "priority": min(0.9, 0.5 + 0.1 * len(hits)),
        })
        if len(out) >= _HARVEST_LIMITS["failure"]:
            break
    return out


def harvest_proposals(context: Any,
                      now: float | None = None) -> list[dict[str, Any]]:
    """(d) Work-generated questions: upgrade proposals the owner denied
    or ignored past the ignore window — a denied idea is still a
    question worth re-researching."""
    db = getattr(context, "db", None)
    if db is None:
        return []
    moment = time.time() if now is None else float(now)
    cutoff = moment - _IGNORE_AFTER_DAYS * 86400.0
    try:
        rows = db.query(
            "SELECT id, title FROM upgrade_proposals "
            "WHERE status IN ('denied', 'proposed') AND created_at < ? "
            "ORDER BY created_at ASC LIMIT ?",
            (cutoff, _HARVEST_LIMITS["proposal"]))
    except Exception:  # noqa: BLE001 - table may not exist yet
        return []
    out: list[dict[str, Any]] = []
    for row in rows or []:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        out.append({"text": title, "source": "proposal",
                    "provenance": f"upgrade_proposal:{row.get('id')}",
                    "priority": 0.5})
    return out


def harvest(context: Any, now: float | None = None) -> list[dict[str, Any]]:
    """All five sources, merged: priority desc, source order breaks
    ties, near-duplicates dropped (first wins). Never raises."""
    moment = time.time() if now is None else float(now)
    found: list[dict[str, Any]] = []
    for fn in (harvest_gaps, harvest_watches, harvest_failures):
        try:
            found.extend(fn(context))
        except Exception:  # noqa: BLE001 - one dead source, not a dead harvest
            _log.warning("topic engine harvest %s failed", fn.__name__,
                         exc_info=True)
    for fn in (harvest_kg_stale, harvest_proposals):
        try:
            found.extend(fn(context, now=moment))
        except Exception:  # noqa: BLE001 - one dead source, not a dead harvest
            _log.warning("topic engine harvest %s failed", fn.__name__,
                         exc_info=True)
    found.sort(key=lambda c: -c["priority"])
    out: list[dict[str, Any]] = []
    for cand in found:
        if any(topics_duplicate(cand["text"], kept["text"])
               for kept in out):
            continue
        out.append(cand)
    return out


# ── the engine ───────────────────────────────────────────────────────────

class TopicEngine:
    """Suggests research topics from the harvested sources + the
    durable ledger. ``suggest()`` ranks; ``note_picked()`` records
    exactly what the loop picked."""

    def __init__(self, context: Any) -> None:
        self.context = context

    def load_ledger(self) -> list[dict[str, Any]]:
        return load_ledger(self.context)

    def save_ledger(self, entries: list[dict[str, Any]]) -> bool:
        return save_ledger(self.context, entries)

    def suggest(self, limit: int = 10,
                now: float | None = None) -> list[str]:
        """Top ``limit`` topics: fresh harvest merged into the ledger,
        ranked by priority desc, then least-suggested, then oldest.
        Never raises — a dead engine returns ``[]`` and the loop falls
        back to owner seeds + contradicted follow-ups."""
        moment = time.time() if now is None else float(now)
        try:
            fresh = harvest(self.context, now=moment)
        except Exception:  # noqa: BLE001 - harvest is fail-soft already
            fresh = []
        try:
            ledger = load_ledger(self.context)
        except Exception:  # noqa: BLE001
            ledger = []
        changed = False
        for cand in fresh:
            if any(topics_duplicate(cand["text"], e["text"])
                   for e in ledger):
                continue
            ledger.append({
                "text": cand["text"], "source": cand["source"],
                "provenance": cand["provenance"],
                "priority": cand["priority"],
                "first_seen_at": moment, "last_suggested_at": 0.0,
                "times_suggested": 0,
            })
            changed = True
        if changed:
            ledger = _evict(ledger)
            save_ledger(self.context, ledger)
        ranked = sorted(
            ledger,
            key=lambda e: (-e["priority"], e["times_suggested"],
                           e["first_seen_at"]))
        return [e["text"] for e in ranked[:max(1, int(limit))]]

    def note_picked(self, topics: list[str],
                    now: float | None = None) -> None:
        """Record exactly the topics the loop picked (bump their
        suggestion counts). Topics the engine offered but the loop
        skipped are not penalised. Never raises."""
        if not topics:
            return
        moment = time.time() if now is None else float(now)
        try:
            ledger = load_ledger(self.context)
        except Exception:  # noqa: BLE001
            return
        changed = False
        for topic in topics:
            for entry in ledger:
                if topics_duplicate(topic, entry["text"]):
                    entry["last_suggested_at"] = moment
                    entry["times_suggested"] += 1
                    changed = True
                    break
        if changed:
            save_ledger(self.context, ledger)

    def stats(self) -> dict[str, Any]:
        """Honest counts for status surfaces. Zeros, never a raise."""
        try:
            ledger = load_ledger(self.context)
        except Exception:  # noqa: BLE001
            ledger = []
        by_source: dict[str, int] = {}
        for entry in ledger:
            by_source[entry["source"]] = by_source.get(entry["source"], 0) + 1
        return {
            "topics": len(ledger),
            "by_source": by_source,
            "suggested_total": sum(e["times_suggested"] for e in ledger),
        }
