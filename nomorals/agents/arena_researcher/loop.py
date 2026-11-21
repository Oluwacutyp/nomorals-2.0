"""Arena researcher: the organ whose subject is Devon herself.

A cadenced organ beside the research loop (owner phase 2026-10-21 item
2): every ``NM_ARENA_RESEARCHER_HOURS`` (default 24) it surveys the
live tree — organs, wiring, dead/decorative paths, weak spots — with
the module-gap audit's evidence discipline (every finding carries a
``file:line`` anchor, importers, callers), mines external state of the
art per organ through the existing research pipeline
(:class:`research_digest.ResearchPipeline` — extended by use, never
duplicated), and files RANKED proposals with evidence into the upgrade
queue (source ``"arena_researcher"``) for the owner's word.

Never auto-approved, never auto-applied: the owner decides, the organ
proposes. A tick never raises: gating, survey, SOTA, filing, notify
and recording each degrade independently and the run record says what
happened.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any

from .rank import DEFAULT_MAX_PROPOSALS, finding_to_ticket, rank_findings
from .store import ArenaResearcherStore, default_arena_researcher_path
from .survey import survey_repo, verify_anchor

#: Durable scheduler job name (also the schedule key).
ARENA_RESEARCHER_JOB = "arena researcher"

#: Filing source recorded on every proposal this organ files.
SOURCE = "arena_researcher"

#: Default cadence (hours) between arena-researcher ticks.
DEFAULT_HOURS = 24.0
MIN_HOURS = 6.0
MAX_HOURS = 168.0

#: Default dedupe window (days): a finding filed within this window is
#: not refiled.
DEFAULT_DEDUPE_DAYS = 14

#: Organs mined for state of the art per cycle (top findings' organs).
DEFAULT_SOTA_ORGANS = 2


def arena_hours() -> float:
    """Cadence in hours from NM_ARENA_RESEARCHER_HOURS (default 24)."""
    try:
        hours = float(os.environ.get("NM_ARENA_RESEARCHER_HOURS", "") or 0)
    except ValueError:
        hours = 0.0
    if hours <= 0:
        return DEFAULT_HOURS
    return min(MAX_HOURS, max(MIN_HOURS, hours))


def arena_max_proposals() -> int:
    """Proposals filed per cycle from NM_ARENA_RESEARCHER_MAX (default 5)."""
    try:
        value = int(os.environ.get("NM_ARENA_RESEARCHER_MAX", "") or 0)
    except ValueError:
        value = 0
    if value <= 0:
        return DEFAULT_MAX_PROPOSALS
    return min(20, value)


def arena_dedupe_days() -> int:
    """Dedupe window in days from NM_ARENA_RESEARCHER_DEDUPE_DAYS."""
    try:
        value = int(os.environ.get(
            "NM_ARENA_RESEARCHER_DEDUPE_DAYS", "") or 0)
    except ValueError:
        value = 0
    if value <= 0:
        return DEFAULT_DEDUPE_DAYS
    return min(90, value)


#: Tick wall budget (seconds): the whole cycle — survey + SOTA +
#: filing — must finish inside the scheduler's 300s job wall, with
#: room for the filing and the notify.  A tick with no budget wedged
#: past the wall on 2026-11-21 (the survey alone reads ~2,900 files).
DEFAULT_TICK_BUDGET_S = 240.0
#: Survey share of the tick budget (seconds).
DEFAULT_SURVEY_BUDGET_S = 120.0


def arena_tick_budget() -> float:
    """Tick wall budget in seconds from NM_ARENA_TICK_BUDGET_S."""
    try:
        value = float(os.environ.get("NM_ARENA_TICK_BUDGET_S", "") or 0)
    except ValueError:
        value = 0.0
    if value <= 0:
        return DEFAULT_TICK_BUDGET_S
    return min(290.0, max(60.0, value))


def arena_survey_budget() -> float:
    """Survey share of the tick budget, NM_ARENA_SURVEY_BUDGET_S."""
    try:
        value = float(os.environ.get("NM_ARENA_SURVEY_BUDGET_S", "") or 0)
    except ValueError:
        value = 0.0
    if value <= 0:
        return DEFAULT_SURVEY_BUDGET_S
    return min(240.0, max(10.0, value))


def _store_for(context: Any) -> ArenaResearcherStore:
    home = None
    try:
        home = getattr(getattr(context, "settings", None), "home", None)
    except Exception:
        home = None
    return ArenaResearcherStore(path=default_arena_researcher_path(home))


def _recent_proposal_keys(context: Any, days: int) -> set[str]:
    """Claim ids this organ filed in the last *days* (for dedupe)."""
    from ..upgrade_queue import UpgradeQueue

    cutoff = time.time() - days * 86400.0
    keys: set[str] = set()
    try:
        for row in UpgradeQueue(context).list(status="proposed"):
            if row.get("source") != SOURCE:
                continue
            if float(row.get("created_at") or 0) < cutoff:
                continue
            for cid in row.get("claim_ids") or []:
                keys.add(str(cid))
    except Exception:
        pass
    return keys


def _source_signals(context: Any) -> dict[str, int]:
    """Owner-visible usefulness of this organ's proposals, by status."""
    from ..upgrade_queue import UpgradeQueue

    signals = {"proposed": 0, "approved": 0, "denied": 0, "ignored": 0,
               "pending": 0, "applied": 0, "tests_passed": 0,
               "tests_failed": 0}
    try:
        for row in UpgradeQueue(context).list():
            if row.get("source") != SOURCE:
                continue
            status = str(row.get("status") or "")
            if status == "proposed":
                signals["pending"] += 1
                signals["proposed"] += 1
            elif status in signals:
                signals[status] += 1
    except Exception:
        pass
    return signals


class ArenaResearcher:
    """One cadenced arena-researcher cycle: survey -> rank -> file."""

    def __init__(self, context: Any, *,
                 max_proposals: int | None = None,
                 sota_organs: int = DEFAULT_SOTA_ORGANS,
                 repo_root: Any = None):
        self.context = context
        self.max_proposals = (max_proposals if max_proposals is not None
                              else arena_max_proposals())
        self.sota_organs = max(0, int(sota_organs))
        self.repo_root = repo_root
        self._store = _store_for(context)

    # ── gates ────────────────────────────────────────────────────────
    def _gates(self) -> tuple[bool, str]:
        from ..research_loop import loop_gates

        return loop_gates(self.context)

    # ── SOTA mining (existing pipeline, never a second one) ─────────
    def _mine_sota(self, organs: list[str], *,
                   deadline: float | None = None
                   ) -> tuple[bool, int, dict[str, str]]:
        """Mine state of the art for *organs* via ResearchPipeline.

        Returns (mined_ok, claim_count, organ -> best claim text).
        A pipeline failure degrades to no SOTA, never a failed cycle.
        ``deadline`` (a ``time.monotonic()`` stamp) stops starting new
        organs once the tick's budget is spent — one organ's research
        is unbounded network + model work, and two of them already
        filled the tick's wall on 2026-11-21.
        """
        from ..research_digest import ResearchPipeline

        claims_by_organ: dict[str, str] = {}
        total = 0
        for organ in organs[:self.sota_organs]:
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                result = ResearchPipeline.run(
                    f"state of the art in {organ} 2026",
                    context=self.context, specialists=1,
                    min_confidence=0.4, notify=False,
                    promote_claims=False)
            except Exception:
                continue
            for claim in result.get("claims") or []:
                text = str(claim.get("claim") or claim.get("text") or "")
                if not text:
                    continue
                total += 1
                if organ not in claims_by_organ:
                    claims_by_organ[organ] = text
        return bool(total), total, claims_by_organ

    # ── the cycle ────────────────────────────────────────────────────
    def tick(self) -> dict[str, Any]:
        """Run one arena-researcher cycle. Never raises."""
        started = time.time()
        record: dict[str, Any] = {
            "id": "ar-run-" + uuid.uuid4().hex[:12],
            "started_at": started,
            "finished_at": started,
            "ok": False,
            "skipped_reason": "",
            "organs_surveyed": 0,
            "findings_count": 0,
            "proposals_created": 0,
            "proposal_ids": [],
            "proposals_deduped": 0,
            "sota_mined": False,
            "sota_claims": 0,
            "notified": False,
            "partial": False,
            "survey_partial": False,
            "surveyed_files": 0,
            "total_files": 0,
            "error": "",
        }
        gates_ok, reason = self._gates()
        if not gates_ok:
            record["skipped_reason"] = reason or "arena researcher disabled"
            record["finished_at"] = time.time()
            self._store.record_run(record)
            return record
        # The tick's own wall budget: the scheduler abandons the
        # worker at 300s and the ghost keeps reading while the next
        # tick starts a second survey.  Survey + SOTA + filing must
        # finish inside the budget, partial rather than wedged.
        tick_deadline = time.monotonic() + arena_tick_budget()
        try:
            from ..research_digest import repo_root

            root = self.repo_root or repo_root()
            findings = survey_repo(root,
                                   deadline_s=arena_survey_budget())
            record["survey_partial"] = bool(
                getattr(findings, "partial", False))
            record["partial"] = record["survey_partial"]
            record["surveyed_files"] = int(
                getattr(findings, "surveyed_files", 0))
            record["total_files"] = int(
                getattr(findings, "total_files", 0))
            findings = [f for f in findings
                        if verify_anchor(root, f["path"], f["line"])]
            ranked = rank_findings(findings)
            record["findings_count"] = len(ranked)
            record["organs_surveyed"] = len({f.get("organ", "")
                                             for f in ranked})
            organs = []
            for f in ranked:
                if f.get("organ") not in organs:
                    organs.append(str(f.get("organ") or "general"))
            sota_ok, sota_count, sota_text = self._mine_sota(
                organs, deadline=tick_deadline)
            record["sota_mined"] = sota_ok
            record["sota_claims"] = sota_count
            from ..upgrade_queue import UpgradePipeline

            pipeline = UpgradePipeline(self.context)
            recent = _recent_proposal_keys(
                self.context, arena_dedupe_days())
            created: list[str] = []
            deduped = 0
            for finding in ranked[:self.max_proposals]:
                if str(finding.get("id") or "") in recent:
                    deduped += 1
                    continue
                ticket = finding_to_ticket(finding, run_id=record["id"])
                sota = sota_text.get(str(finding.get("organ") or ""))
                if sota:
                    ticket["rationale"] = (
                        "{} External state of the art (2026): {}".format(
                            ticket["rationale"], sota))
                pid = pipeline.propose_from_ticket(ticket, source=SOURCE)
                if pid:
                    created.append(pid)
            record["proposals_created"] = len(created)
            record["proposal_ids"] = created
            record["proposals_deduped"] = deduped
            if created:
                from ..notifier import Notifier

                body = (
                    f"Arena researcher surveyed the live system and filed "
                    f"{len(created)} ranked proposal(s) into the upgrade queue "
                    f"({len(ranked)} finding(s) total, {deduped} deduped as already filed). "
                    f"Top: {ranked[0]['kind']} — {ranked[0]['path']}. "
                    "Nothing is applied without your word."
                    if ranked else
                    f"Arena researcher surveyed the live system and filed "
                    f"{len(created)} ranked proposal(s).")
                Notifier(self.context).publish(
                    "arena_researcher",
                    title=f"Arena researcher: {len(created)} ranked proposal(s)",
                    body=body,
                    alert_hints={"notice": True,
                                 "subject": "arena researcher"})
                record["notified"] = True
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001 - a tick never raises
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished_at"] = time.time()
        self._store.record_run(record)
        return record


def ensure_arena_researcher_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``arena researcher`` job (idempotent).

    Runs ``every NM_ARENA_RESEARCHER_HOURS`` (default 24h) via the
    ``arena_researcher`` tool's ``tick`` action. Safe to call on every
    boot: if the owner changed the interval (or env), the stale job is
    replaced; the loop itself re-checks the feature flag, the proactive
    master switch, and quiet hours before burning budget. Mirrors
    :func:`research_loop.ensure_research_job`.
    """
    from ...core.failures import record_failure
    from ..scheduler import Scheduler

    sched = Scheduler(context)
    hours = arena_hours()
    want = f"every {int(hours * 3600)}s"
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == ARENA_RESEARCHER_JOB]
    except Exception:  # noqa: BLE001 — scheduler table may not exist yet
        have = []
    if have and have[0].get("spec") == want:
        _reconcile_overlap_skip(sched, have[0])
        return {"name": ARENA_RESEARCHER_JOB, "already_scheduled": True,
                "job_id": have[0].get("id"), "interval_hours": hours}
    for j in have:  # stale interval -> replace
        try:
            sched.remove(j["id"])
        except Exception as exc:  # noqa: BLE001
            record_failure("arena_researcher.ensure_arena_researcher_job",
                           exc)
    job = sched.add(ARENA_RESEARCHER_JOB, f"every {hours:g}h", "tool",
                    {"tool": "arena_researcher",
                     "args": {"action": "tick"}},
                    overlap_policy="skip")
    # add() is idempotent by exact replay: an existing row at the same
    # spec comes back deduped, carrying its OLD policy — reconcile it
    # so the skip lands on persisted rows too, not just fresh ones.
    _reconcile_overlap_skip(sched, job)
    return {"name": ARENA_RESEARCHER_JOB, "scheduled": True,
            "job_id": job.get("id"), "interval_hours": hours}


def _reconcile_overlap_skip(sched: Any, row: dict[str, Any]) -> None:
    """Force a persisted job row onto ``overlap_policy=skip``.

    ``add()`` only sets the policy on a FRESH row, so a job registered
    months ago as ``concurrent`` keeps stacking a second worker behind
    a wedged first one forever (2026-11-19 stall: the news poll kept
    ``concurrent`` and went 1 → 2 abandoned workers).  Best-effort:
    never raises — a stale policy is a finding, not a crash.
    """
    try:
        if (row.get("overlap_policy") or "concurrent") != "skip":
            sched.set_overlap_policy(row["id"], "skip")
    except Exception:  # noqa: BLE001 - reconciliation is best-effort
        pass


def status(context: Any) -> dict[str, Any]:
    """Status surface: job health, gates, last run, pending proposals.

    Mirrors :func:`research_loop.status` — same gates dict, same job
    shape — so the two loops read identically from ``nm status``.
    """
    from ..features import feature_enabled
    from ..research_loop import _in_quiet_hours_now
    from ..scheduler import Scheduler
    from ..upgrade_queue import UpgradeQueue

    out: dict[str, Any] = {
        "job": {"scheduled": False, "enabled": False, "next_run": None,
                "next_run_iso": None, "interval_hours": arena_hours()},
        "gates": {"feature_research": feature_enabled(context, "research"),
                  "proactive_master": True, "quiet_hours": False},
        "last_run": None,
        "pending_proposals": 0,
        "errors": {},
    }
    partner = getattr(getattr(context, "settings", None), "partner", None)
    if partner is not None:
        out["gates"]["proactive_master"] = bool(
            getattr(partner, "proactive_enabled", True))
    try:
        out["gates"]["quiet_hours"] = bool(_in_quiet_hours_now(context))
    except Exception as exc:  # noqa: BLE001
        out["errors"]["quiet_hours"] = f"{type(exc).__name__}: {exc}"
    try:
        sched = Scheduler(context)
        jobs = [j for j in sched.list_jobs()
                if j.get("name") == ARENA_RESEARCHER_JOB]
        if jobs:
            j = jobs[0]
            out["job"].update({
                "scheduled": True, "enabled": bool(j.get("enabled")),
                "next_run": j.get("next_run"),
                "next_run_iso": j.get("next_run_iso"),
                "interval_hours": arena_hours(),
            })
    except Exception as exc:  # noqa: BLE001
        out["errors"]["scheduler"] = f"{type(exc).__name__}: {exc}"
    try:
        for row in UpgradeQueue(context).list(status="proposed"):
            if row.get("source") == SOURCE:
                out["pending_proposals"] += 1
    except Exception as exc:  # noqa: BLE001
        out["errors"]["upgrade_queue"] = f"{type(exc).__name__}: {exc}"
    store = _store_for(context)
    out["last_run"] = store.last_run()
    out["signals"] = _source_signals(context)
    out["degraded_reason"] = store.degraded_reason
    return out


def register(registry: Any) -> None:
    """Register the ``arena_researcher`` tool beside ``research_loop``."""
    from ...core.policy import Capability

    context = registry.context

    @registry.register(
        "arena_researcher",
        description=(
            "arena researcher (Wave E): Devon researching herself — "
            "survey the live system for dead paths, stubs, unwired "
            "tools and weak spots, mine state of the art per organ "
            "through the research pipeline, and file RANKED proposals "
            "into the upgrade queue for the owner's word (nothing is "
            "auto-applied). Actions: tick | status | survey | ensure | "
            "enable | disable | proposals."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — tick | status | survey | ensure | enable | "
                      "disable | proposals (default status)",
            "max_proposals": "int (optional, default 5) — proposals "
                             "filed per tick",
        },
    )
    def arena_researcher(
        action: str = "status",
        max_proposals: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        verb = (action or "status").strip().lower()
        if verb == "status":
            return {"ok": True, **status(context)}
        if verb == "survey":
            from ..research_digest import repo_root

            root = repo_root()
            findings = [f for f in survey_repo(root)
                        if verify_anchor(root, f["path"], f["line"])]
            return {"ok": True, "findings": rank_findings(findings)}
        if verb == "ensure":
            return {"ok": True, **ensure_arena_researcher_job(context)}
        if verb in ("enable", "disable"):
            from ..scheduler import Scheduler

            sched = Scheduler(context)
            jobs = [j for j in sched.list_jobs()
                    if j.get("name") == ARENA_RESEARCHER_JOB]
            if not jobs:
                info = ensure_arena_researcher_job(context)
                jobs = [j for j in sched.list_jobs()
                        if j.get("name") == ARENA_RESEARCHER_JOB]
                if not jobs:
                    return {"ok": False, "error": "job registration failed",
                            **info}
            row = sched.set_enabled(jobs[0]["id"], verb == "enable")
            return {"ok": True, "enabled": verb == "enable", "job": row}
        if verb == "proposals":
            from ..upgrade_queue import UpgradeQueue

            rows = [r for r in UpgradeQueue(context).list()
                    if r.get("source") == SOURCE]
            return {"ok": True, "proposals": rows}
        if verb == "tick":
            loop = ArenaResearcher(
                context, max_proposals=max_proposals or None)
            return {"ok": True, **loop.tick()}
        raise ValueError(f"unknown action: {action!r}")
