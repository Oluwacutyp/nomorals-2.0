"""Proactive triggers: the worker NOTICES and INITIATES.

A loop that ticks and waits to be poked is not enough. Each system
proactively notices a trigger — an event, change, deadline, silence,
or opportunity — and initiates work through the initiative lifecycle
+ rotation (never bypassing them). The rotation still governs
capacity/fairness: proactivity means the system CREATES initiatives
from triggers, not that it runs unthrottled.

Detectors never spend / post / send / submit; authority is consulted
at execution time. Dedupe is two-layer: ``dedupe_key`` (merged by
``InitiativeStore.create``) + caller-owned ``state`` seen-sets, so a
completed trigger never re-fires for the same observation. State is
durable (``agency_proactive_state`` table) so a restart never
re-fires either.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .initiatives import Initiative, InitiativeStore
from .failures import record_failure

__all__ = [
    "Trigger", "fire_triggers", "detect_triggers", "scan_proactive",
    "load_proactive_state", "save_proactive_state",
    "LEAD_QUIET_S",
]

#: A lead in an active conversation stage with no movement this long
#: is "gone quiet" and earns a proactive follow-up.
LEAD_QUIET_S = 48 * 3600

#: Lead stages where silence means something (waiting on the lead).
_LEAD_QUIET_STAGES = frozenset(
    {"contacted", "replied", "scoped", "proposal"})


@dataclass
class Trigger:
    system: str
    kind: str
    dedupe_key: str
    evidence: str
    priority: float = 0.5
    next_action: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


def fire_triggers(store: InitiativeStore, triggers: list[Trigger],
                  now: float) -> list[Initiative]:
    """Turn triggers into initiatives (dedupe via store merge)."""
    out = []
    for t in triggers:
        out.append(store.create(Initiative(
            outcome=f"{t.system}: {t.kind} — {t.evidence[:200]}",
            kind="evidence_opportunity",
            objective=t.system,
            evidence=t.evidence,
            next_action=t.next_action or f"work {t.system} trigger {t.kind}",
            next_review_at=now + 3600,
            dedupe_key=t.dedupe_key,
            priority=t.priority)))
    return out


def scan_proactive(systems: Any, *, now: float | None = None) -> list[Trigger]:
    """Convenience: run ``systems.scan_proactive`` (ActiveSystems)."""
    return systems.scan_proactive(now=now)


# ── durable state ────────────────────────────────────────────────

def _encode(value: Any) -> Any:
    if isinstance(value, set):
        return {"__set__": sorted(value, key=repr)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__set__" in value:
            return set(value["__set__"])
        if "__tuple__" in value:
            return tuple(_decode(v) for v in value["__tuple__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _fresh_state() -> dict[str, Any]:
    return {
        "deals": {"seen_alert_ids": set()},
        "weather": {"baselines": {}},
        "news": {"seen_cluster_keys": set()},
        "social": {"prev": {}},
        "leads": {"seen_lead_ids": set(), "last_seen": {}},
        "gigs": {"seen_opp_ids": set(), "seen_followups": set()},
        "research": {"seen_gap_ids": set(), "seen_watch_ids": set()},
        "wisdom": {"flagged": False},
        "impossible": {"seen_claim_ids": set()},
        "digestion": {"last_count": 0},
        "planner": {"seen_overdue": set()},
    }


def load_proactive_state(db: Any) -> dict[str, Any]:
    """Load durable detector state. Mock-safe: any failure → fresh."""
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS agency_proactive_state ("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')")
        rows = db.query("SELECT key, value FROM agency_proactive_state") or []
        state = _fresh_state()
        for row in rows:
            try:
                decoded = _decode(json.loads(row.get("value", "") or "{}"))
            except Exception as exc:  # noqa: BLE001 - corrupt row: skip
                record_failure("proactive.load_row_decode", exc)
                continue
            state[row.get("key", "")] = decoded
        # Fill any missing per-system buckets (fresh keys on upgrade).
        for key, bucket in _fresh_state().items():
            state.setdefault(key, bucket)
        return state
    except Exception as exc:  # noqa: BLE001 - no table/db = fresh state
        record_failure("proactive.load_state", exc)
        return _fresh_state()


def save_proactive_state(db: Any, state: dict[str, Any]) -> None:
    """Persist detector state. Mock-safe: failures are swallowed."""
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS agency_proactive_state ("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')")
        for key, bucket in state.items():
            db.execute(
                "INSERT OR REPLACE INTO agency_proactive_state "
                "(key, value) VALUES (?, ?)",
                (key, json.dumps(_encode(bucket), default=str)))
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        record_failure("proactive.save_state", exc)


# ── detectors ────────────────────────────────────────────────────

def _sig_weather(snapshot: dict[str, Any]) -> str:
    days = [(d.get("date", ""), str(d.get("summary", d))) for d in
            (snapshot.get("days") or [])]
    alerts = sorted(a.get("text", "") for a in (snapshot.get("alerts") or []))
    return json.dumps({"days": days, "alerts": alerts}, sort_keys=True,
                      default=str)


def detect_triggers(system: str, source: Any, state: dict[str, Any], *,
                    now: float | None = None,
                    places: list[str] | None = None) -> list[Trigger]:
    """Detect new triggers for one system. Never side-effects."""
    now = time.time() if now is None else now
    fn = _DETECTORS.get(system)
    if fn is None:
        return []
    try:
        return fn(source, state, now, places or [])
    except Exception as exc:  # noqa: BLE001 - detector failure = no triggers
        record_failure("proactive.detect_dispatch", exc)
        return []


def _detect_deals(source: Any, state: dict, now: float,
                  places: list[str]) -> list[Trigger]:
    if source is None:
        return []
    try:
        alerts = source.check_alerts() or []
    except Exception as exc:  # noqa: BLE001
        record_failure("proactive.deals_alerts", exc)
        return []
    bucket = state.setdefault("deals", {})
    seen = bucket.setdefault("seen_alert_ids", set())
    out = []
    for alert in alerts:
        aid = str(alert.get("id", alert.get("product", alert)))
        if aid in seen:
            continue
        seen.add(aid)
        out.append(Trigger(
            system="deals", kind="price_alert",
            dedupe_key=f"deals:alert:{aid}",
            evidence=f"price alert: {alert.get('product', aid)} — "
                     f"{alert.get('reason', alert.get('summary', ''))}",
            priority=0.7,
            next_action=f"re-verify deal for {alert.get('product', aid)}",
            payload={"alert": dict(alert) if isinstance(alert, dict)
                     else {"value": str(alert)}}))
    return out


def _detect_weather(source: Any, state: dict, now: float,
                    places: list[str]) -> list[Trigger]:
    if source is None:
        return []
    bucket = state.setdefault("weather", {})
    baselines = bucket.setdefault("baselines", {})
    out = []
    for place in places:
        try:
            snap = source.snapshot(place) or {}
        except Exception as exc:  # noqa: BLE001
            record_failure("proactive.weather_snapshot", exc)
            continue
        sig = _sig_weather(snap)
        prev = baselines.get(place)
        baselines[place] = sig
        if prev is None:
            # First sight: baseline deposited, nothing to compare yet.
            continue
        if sig == prev:
            continue
        official = any(a.get("category") == "official_warning"
                       for a in (snap.get("alerts") or []))
        out.append(Trigger(
            system="weather",
            kind="official_warning" if official else "forecast_change",
            dedupe_key=f"weather:{place}:{abs(hash(sig)) % 10**8}",
            evidence=f"weather changed for {place}"
                     + (" — official warning present" if official else ""),
            priority=0.9 if official else 0.6,
            next_action=f"brief weather change for {place}",
            payload={"place": place, "snapshot": snap}))
    return out


def _detect_news(source: Any, state: dict, now: float,
                 places: list[str]) -> list[Trigger]:
    if source is None:
        return []
    try:
        clusters = source.clusters() or []
    except Exception as exc:  # noqa: BLE001
        record_failure("proactive.news_clusters", exc)
        return []
    bucket = state.setdefault("news", {})
    seen = bucket.setdefault("seen_cluster_keys", set())
    out = []
    for c in clusters:
        key = str(c.get("key", c.get("title", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(Trigger(
            system="news", kind="new_cluster",
            dedupe_key=f"news:cluster:{abs(hash(key)) % 10**8}",
            evidence=f"new story cluster: {c.get('title', key)}",
            priority=0.5,
            next_action="digest new story cluster",
            payload={"cluster": c}))
    return out


def _detect_social(source: Any, state: dict, now: float,
                   places: list[str]) -> list[Trigger]:
    if source is None:
        return []
    try:
        backlog = source.backlog() or {}
    except Exception as exc:  # noqa: BLE001
        record_failure("proactive.social_backlog", exc)
        return []
    if backlog.get("stop_signal"):
        return []  # honour opt-out: no proactive contact
    if not backlog.get("cadence_ok", True):
        return []  # cadence window closed: silence is correct
    bucket = state.setdefault("social", {})
    prev = bucket.setdefault("prev", {})
    flags = {
        "promised_reply": int(backlog.get("promised_replies", 0) or 0) > 0,
        "unanswered_question": int(backlog.get("unanswered", 0) or 0) > 0,
        "milestone": int(backlog.get("milestones", 0) or 0) > 0,
    }
    out = []
    for kind, is_on in flags.items():
        was = prev.get(kind, False)
        prev[kind] = is_on
        if is_on and not was:
            out.append(Trigger(
                system="social", kind=kind,
                dedupe_key=f"social:{kind}:{int(now)}",
                evidence=f"social: {kind.replace('_', ' ')} needs action",
                priority=0.7,
                next_action=f"handle social {kind}",
                payload={"backlog": dict(backlog)}))
    return out


def _detect_leads(source: Any, state: dict, now: float,
                  places: list[str]) -> list[Trigger]:
    if source is None:
        return []
    try:
        leads = source.live_leads() or []
    except Exception as exc:  # noqa: BLE001
        record_failure("proactive.leads_live", exc)
        return []
    bucket = state.setdefault("leads", {})
    seen = bucket.setdefault("seen_lead_ids", set())
    last_seen = bucket.setdefault("last_seen", {})
    out = []
    for lead in leads:
        lid = str(lead.get("lead_id", lead.get("id", "")))
        if not lid:
            continue
        stage = str(lead.get("stage", lead.get("status", "discovered")))
        if lid not in seen:
            seen.add(lid)
            last_seen[lid] = now
            out.append(Trigger(
                system="leads", kind="new_lead",
                dedupe_key=f"leads:new:{lid}",
                evidence=f"new lead: {lead.get('name', lid)} ({stage})",
                priority=0.6,
                next_action=f"qualify lead {lid}",
                payload={"lead": dict(lead)}))
            continue
        # Quiet lead: active stage, no movement for LEAD_QUIET_S.
        prev_seen = last_seen.get(lid, now)
        if stage in _LEAD_QUIET_STAGES and now - prev_seen >= LEAD_QUIET_S:
            last_seen[lid] = now  # one follow-up trigger per quiet span
            out.append(Trigger(
                system="leads", kind="quiet_lead",
                dedupe_key=f"leads:quiet:{lid}:{int(now)}",
                evidence=f"lead {lead.get('name', lid)} quiet in {stage} "
                         f"for {int((now - prev_seen) // 3600)}h",
                priority=0.65,
                next_action=f"follow up lead {lid}",
                payload={"lead": dict(lead)}))
        else:
            last_seen[lid] = now
    return out


def _detect_gigs(source: Any, state: dict, now: float,
                 places: list[str]) -> list[Trigger]:
    if source is None:
        return []
    out = []
    try:
        opps = source.scan() or []
    except Exception as exc:  # noqa: BLE001
        record_failure("proactive.gigs_scan", exc)
        opps = []
    bucket = state.setdefault("gigs", {})
    seen = bucket.setdefault("seen_opp_ids", set())
    for opp in opps:
        oid = str(opp.get("id", opp.get("url", opp.get("title", ""))))
        if not oid or oid in seen:
            continue
        seen.add(oid)
        out.append(Trigger(
            system="gigs", kind="new_opportunity",
            dedupe_key=f"gigs:opp:{oid}",
            evidence=f"new opportunity: {opp.get('title', oid)}",
            priority=0.6,
            next_action=f"score + draft opportunity {oid}",
            payload={"opp": dict(opp)}))
    try:
        due = source.follow_ups_due() or []
    except Exception as exc:  # noqa: BLE001
        record_failure("proactive.gigs_followups", exc)
        due = []
    seen_fu = bucket.setdefault("seen_followups", set())
    for app in due:
        gid = str(app.get("gig_id", app.get("id", "")))
        if not gid or gid in seen_fu:
            continue
        seen_fu.add(gid)
        out.append(Trigger(
            system="gigs", kind="follow_up_due",
            dedupe_key=f"gigs:followup:{gid}",
            evidence=f"follow-up due for gig {gid}",
            priority=0.65,
            next_action=f"draft follow-up for gig {gid}",
            payload={"gig_id": gid}))
    return out


def _detect_research(source: Any, state: dict, now: float,
                     places: list[str]) -> list[Trigger]:
    out = []
    gaps: list[Any] = []
    watches: list[Any] = []
    if isinstance(source, dict):
        gaps = list(source.get("gaps") or [])
        watches = list(source.get("watches") or [])
    elif source is not None:
        try:
            gaps = list(source.open_gaps(limit=50) or [])
        except Exception as exc:  # noqa: BLE001
            record_failure("proactive.research_gaps", exc)
            gaps = []
        try:
            watches = list(source.due_watches(now=now) or [])
        except Exception as exc:  # noqa: BLE001
            record_failure("proactive.research_watches", exc)
            watches = []
    bucket = state.setdefault("research", {})
    seen_g = bucket.setdefault("seen_gap_ids", set())
    for gap in gaps:
        gid = str(gap.get("id", gap) if isinstance(gap, dict) else gap)
        if not gid or gid in seen_g:
            continue
        seen_g.add(gid)
        out.append(Trigger(
            system="research", kind="new_gap",
            dedupe_key=f"research:gap:{gid}",
            evidence=f"new knowledge gap: {gap.get('question', gid) if isinstance(gap, dict) else gid}",
            priority=0.6,
            next_action=f"investigate gap {gid}",
            payload={"gap": dict(gap) if isinstance(gap, dict)
                     else {"id": gid}}))
    seen_w = bucket.setdefault("seen_watch_ids", set())
    for w in watches:
        wid = str(w.get("id", w) if isinstance(w, dict) else w)
        if not wid or wid in seen_w:
            continue
        seen_w.add(wid)
        out.append(Trigger(
            system="research", kind="watch_due",
            dedupe_key=f"research:watch:{wid}",
            evidence=f"research watch due: {wid}",
            priority=0.6,
            next_action=f"check research watch {wid}",
            payload={"watch": dict(w) if isinstance(w, dict)
                     else {"id": wid}}))
    return out


def _detect_wisdom(source: Any, state: dict, now: float,
                   places: list[str]) -> list[Trigger]:
    queued = 0
    if isinstance(source, dict):
        queued = int(source.get("queued", 0) or 0)
    elif source is not None:
        try:
            queued = len(source._pending_manifest() or [])
        except Exception as exc:  # noqa: BLE001
            record_failure("proactive.wisdom_queue", exc)
            queued = 0
    bucket = state.setdefault("wisdom", {})
    flagged = bucket.get("flagged", False)
    if queued > 0 and not flagged:
        bucket["flagged"] = True
        return [Trigger(
            system="wisdom", kind="ingest_queue",
            dedupe_key=f"wisdom:ingest:{int(now)}",
            evidence=f"wisdom ingest queue has {queued} texts",
            priority=0.5,
            next_action="ingest wisdom queue",
            payload={"queued": queued})]
    if queued == 0:
        bucket["flagged"] = False
    return []


def _detect_impossible(source: Any, state: dict, now: float,
                       places: list[str]) -> list[Trigger]:
    queued: list[Any] = []
    if isinstance(source, dict):
        queued = list(source.get("queued") or [])
    elif source is not None:
        try:
            queued = list(source.queue_list(status="queued") or [])
        except Exception as exc:  # noqa: BLE001
            record_failure("proactive.impossible_queue", exc)
            queued = []
    bucket = state.setdefault("impossible", {})
    seen = bucket.setdefault("seen_claim_ids", set())
    out = []
    for claim in queued:
        cid = str(claim.get("id", claim) if isinstance(claim, dict)
                  else getattr(claim, "id", claim))
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(Trigger(
            system="impossible", kind="new_claim",
            dedupe_key=f"impossible:claim:{cid}",
            evidence=f"new impossible claim queued: {cid}",
            priority=0.6,
            next_action=f"attempt claim {cid}",
            payload={"claim_id": cid}))
    return out


def _detect_digestion(source: Any, state: dict, now: float,
                      places: list[str]) -> list[Trigger]:
    count = 0
    if isinstance(source, dict):
        count = int(source.get("knowledge_count", 0) or 0)
    elif source is not None:
        try:
            count = int(source.count() or 0)
        except Exception as exc:  # noqa: BLE001
            record_failure("proactive.digestion_count", exc)
            count = 0
    bucket = state.setdefault("digestion", {})
    last = int(bucket.get("last_count", 0) or 0)
    if count > last:
        bucket["last_count"] = count
        return [Trigger(
            system="digestion", kind="knowledge_grew",
            dedupe_key=f"digestion:grew:{count}:{int(now)}",
            evidence=f"knowledge base grew {last} → {count} items",
            priority=0.4,
            next_action="digest new knowledge",
            payload={"from": last, "to": count})]
    return []


def _detect_planner(source: Any, state: dict, now: float,
                    places: list[str]) -> list[Trigger]:
    initiatives: list[Any] = []
    if isinstance(source, dict):
        initiatives = list(source.get("initiatives") or [])
    elif isinstance(source, list):
        initiatives = source
    elif source is not None:
        try:
            initiatives = list(source.list_active() or [])
        except Exception as exc:  # noqa: BLE001
            record_failure("proactive.planner_list", exc)
            initiatives = []
    bucket = state.setdefault("planner", {})
    seen = bucket.setdefault("seen_overdue", set())
    out = []
    for it in initiatives:
        if isinstance(it, dict):
            priority = float(it.get("priority", 0) or 0)
            nxt = float(it.get("next_review_at", 0) or 0)
            iid = str(it.get("id", it.get("dedupe_key", "")))
        else:
            priority = float(getattr(it, "priority", 0) or 0)
            nxt = float(getattr(it, "next_review_at", 0) or 0)
            iid = str(getattr(it, "id", "") or getattr(it, "dedupe_key", ""))
        if priority >= 0.8 and nxt and now - nxt > 3600 and iid not in seen:
            seen.add(iid)
            out.append(Trigger(
                system="planner", kind="overdue_critical",
                dedupe_key=f"planner:overdue:{iid}",
                evidence=f"overdue critical initiative {iid}",
                priority=0.9,
                next_action=f"plan around overdue initiative {iid}",
                payload={"initiative_id": iid}))
    return out


_DETECTORS = {
    "deals": _detect_deals,
    "weather": _detect_weather,
    "news": _detect_news,
    "social": _detect_social,
    "leads": _detect_leads,
    "gigs": _detect_gigs,
    "research": _detect_research,
    "wisdom": _detect_wisdom,
    "impossible": _detect_impossible,
    "digestion": _detect_digestion,
    "planner": _detect_planner,
}
