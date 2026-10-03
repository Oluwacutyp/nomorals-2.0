"""Ops alerts (wave 82 + wave 83 escalation).

The operations surfaces that should page the operator instead of
waiting for a human to run --health.  Three conditions are scanned:

* **stuck missions** — RUNNING with no checkpoint progress for
  STUCK_AFTER_SECONDS (the worker died mid-step; the mission is
  resumable but no one has picked it up)
* **chronic routing roles** — a ``route:<role>`` skill whose real
  outcome record has crossed the chronic threshold (the same priors the
  router pays, now surfaced as an alert)
* **unrecovered pivots** — a course-corrected step that pivoted and
  STILL failed (the approach was changed and the change did not work)

Findings are notified through the standard notifier (durable queue +
multi-channel delivery + dedupe) with a per-finding cooldown so a
persistently stuck mission pages once, not every ten minutes.

Wave 83 escalation: a stuck mission is NOT paged on first sight — the
system tries ONE self-heal run (bounded, corrective) and pages the
operator only if the mission is still stuck when the next scan looks.
The attempt is recorded per mission (kv) so it happens exactly once.
"""
from __future__ import annotations

import time
from typing import Any

#: how long a finding must stay quiet before it may page again
COOLDOWN_SECONDS = 6 * 3600.0
#: pivots older than this are history, not an alert
PIVOT_WINDOW_SECONDS = 24 * 3600.0


def _kv_get(db: Any, key: str) -> float:
    try:
        row = db.query_one("SELECT value FROM kv_store WHERE key=?", (key,))
        return float(row["value"]) if row else 0.0
    except Exception:  # noqa: BLE001 - cooldown is best-effort
        return 0.0


def _kv_set(db: Any, key: str, value: float) -> None:
    try:
        db.execute(
            "INSERT INTO kv_store (key, value, kind, updated_at) "
            "VALUES (?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (key, str(value), "json", time.time()))
    except Exception:  # noqa: BLE001
        pass


def _stuck_findings(context: Any, *, escalate: bool,
                    heal_background: bool,
                    healed_out: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stuck-mission findings.  With ``escalate=True`` a stuck mission
    gets one self-heal attempt (recorded per mission) before it is
    reported; already-healed missions report immediately."""
    from ..missions import wired_runner
    runner = wired_runner(context)
    db = getattr(context, "db", None)
    findings: list[dict[str, Any]] = []
    for entry in runner.health(limit=50).get("active", []):
        if not entry.get("stuck"):
            continue
        key = f"stuck:{entry['id']}"
        already_healed = _kv_get(db, f"opsselfheal:{key}") > 0
        if escalate and not already_healed:
            try:
                heal = runner.self_heal(entry["id"],
                                        background=heal_background)
            except Exception:  # noqa: BLE001 - healing is best-effort
                heal = {"attempted": False}
            if heal.get("attempted"):
                _kv_set(db, f"opsselfheal:{key}", time.time())
                healed_out.append({
                    "key": key,
                    "mission": entry["id"][:12],
                    "detail": str(heal.get("status", "queued")),
                })
                # gave it one chance — the NEXT scan decides whether
                # the operator gets paged
                continue
        findings.append({
            "kind": "stuck_mission",
            "key": key,
            "title": (f"Mission stuck: {entry['id'][:12]} — "
                      f"{str(entry.get('goal', ''))[:60]}"
                      + (" (still stuck after self-heal)"
                         if already_healed else "")),
            "body": (f"{entry.get('iterations', 0)} iteration(s), "
                     f"{len(entry.get('completed') or [])} step(s) done, "
                     f"last error: {str(entry.get('last_error', ''))[:120] or 'none'} "
                     f"— resumable: nm missions --resume {entry['id'][:12]}"),
            "data": entry,
        })
    return findings


def scan_ops_escalated(context: Any, *, escalate: bool = True,
                       heal_background: bool = True
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full ops scan.  Returns ``(findings, self_healed)``."""
    findings: list[dict[str, Any]] = []
    self_healed: list[dict[str, Any]] = []
    now = time.time()

    try:
        findings.extend(_stuck_findings(
            context, escalate=escalate, heal_background=heal_background,
            healed_out=self_healed))
    except Exception:  # noqa: BLE001 - one dead surface must not kill the scan
        pass

    # 2) chronic routing roles
    try:
        from .skills import SkillLibrary
        for skill in SkillLibrary(getattr(context, "db", None)).list(
                kind="routing", limit=200):
            if not skill.name.startswith("route:"):
                continue
            total = skill.success_count + skill.failure_count
            if total < 3:
                continue
            if skill.success_rate > 0.34:
                continue
            role = skill.name.split(":", 1)[1]
            findings.append({
                "kind": "chronic_role",
                "key": f"chronic:{skill.name}",
                "title": (f"Routing role '{role}' chronically failing — "
                          f"{skill.success_rate:.0%} over {total} runs"),
                "body": (f"The router is discounting this role (-0.15 prior). "
                         f"Check the agent behind it or retrain the goal "
                         f"wording. Last used: "
                         f"{int(now - (skill.last_used or now))}s ago"),
                "data": skill.to_dict(),
            })
    except Exception:  # noqa: BLE001
        pass

    # 3) pivots that did not recover
    try:
        from .orchestrator import MasterOrchestrator
        for run in MasterOrchestrator.runs(context, limit=40):
            if not run.get("ts") or now - run["ts"] > PIVOT_WINDOW_SECONDS:
                continue
            for piv in run.get("pivots", []):
                if piv.get("recovered"):
                    continue
                findings.append({
                    "kind": "unrecovered_pivot",
                    "key": f"pivot:{run.get('goal', '')[:32]}:{piv.get('task', '')}",
                    "title": (f"Pivoted step '{piv.get('task', '?')}' did not "
                              f"recover — {str(run.get('goal', ''))[:60]}"),
                    "body": (f"The approach was changed ({str(piv.get('pivot', ''))[:100]}) "
                             f"and the step still failed. Needs new information, "
                             f"not another retry."),
                    "data": piv,
                })
    except Exception:  # noqa: BLE001
        pass

    return findings, self_healed


def scan_ops(context: Any) -> list[dict[str, Any]]:
    """Pure scan (no self-heal side effects) — every condition that
    currently warrants an operator's attention.  Each finding carries a
    stable ``key`` (for the per-finding cooldown), a human
    ``title``/``body``, and the raw ``data`` for --json."""
    return scan_ops_escalated(context, escalate=False)[0]


def ops_alerts(context: Any, *, force: bool = False,
               heal_background: bool = True) -> dict[str, Any]:
    """Scan the ops surfaces and page the operator on new findings.

    A finding pages at most once per COOLDOWN_SECONDS (kv-tracked), so a
    stuck mission alerts, then goes quiet until it is fixed or the
    cooldown lapses.  ``force`` bypasses the cooldown AND the
    self-heal step (the operator asked for the raw truth).  Returns
    ``{"findings", "sent", "cooldown", "self_healed", "pending"}``.
    """
    from .notifier import Notifier, notify

    findings, self_healed = scan_ops_escalated(
        context, escalate=not force, heal_background=heal_background)
    sent: list[dict[str, Any]] = []
    cooled: list[str] = []
    now = time.time()
    db = getattr(context, "db", None)
    for f in findings:
        key = f["key"]
        if not force:
            last = _kv_get(db, f"opsalert:{key}") if db is not None else 0.0
            if last and now - last < COOLDOWN_SECONDS:
                cooled.append(key)
                continue
        if db is not None:
            _kv_set(db, f"opsalert:{key}", now)
        try:
            notify(context, "ops", f["title"], f["body"])
        except Exception:  # noqa: BLE001 - alerting must never raise
            pass
        sent.append(f)
    pending = 0
    try:
        pending = len(Notifier(context).pending())
    except Exception:  # noqa: BLE001
        pass
    return {"findings": findings, "sent": sent,
            "cooldown": cooled, "self_healed": self_healed,
            "pending": pending}
