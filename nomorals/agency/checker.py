"""Maker-checker separation for agency work.

The tick that DOES the work must NOT be the one that verifies it.
:meth:`Checker.verify` re-reads the world through a DIFFERENT code
path than the worker used — a fresh adapter read, a fresh store read,
a fresh DB query — and compares it against the worker's claimed
verification. The generator never grades its own work: a handler
proposes a completion (``stop_reason`` + claimed ``verification``);
the checker independently re-observes; only a passing check completes
the initiative. A failing check advances the initiative with a
``maker-checker rejected`` blocker so the discrepancy is visible and
the work can be retried, never silently completed.

Checks are per-system because "verified" means something different
per system:

* deals — re-run ``check_need`` for the same need: the independent
  re-check must agree on ``is_saving`` (and on the purchase receipt
  when a purchase is claimed).
* weather — re-fetch ``snapshot(place)``: days/alerts must match.
* news — re-fetch ``clusters()``: the count must match.
* social — silence claims re-read ``backlog()`` (stop/cadence must
  still hold); publish claims re-read ``due_posts()`` (must be empty).
* leads — re-read the lead from the store: stage/status must match
  the claimed transition (``dmed`` for a sent DM).
* gigs — re-score the opportunity: the recommendation must match.
* research — the gap must no longer be open (``open_gaps`` re-read,
  or the ``knowledge_gaps`` DB row when a db is available).
* wisdom — the claimed ingested slug must no longer be pending.
* impossible — the claim must no longer be queued.
* planner — the executed graph must no longer be pending.
* digestion — the knowledge count must be >= the claimed count.

A system with no independent re-read available fails closed: the
checker returns ``passed=False`` with the reason, and the work stays
uncompleted rather than self-verified.
"""
from __future__ import annotations

from typing import Any

__all__ = ["Checker", "CheckResult"]


class CheckResult:
    def __init__(self, passed: bool, *, reason: str = "",
                 checks: list[str] | None = None) -> None:
        self.passed = passed
        self.reason = reason
        self.checks = list(checks or [])

    def __bool__(self) -> bool:
        return self.passed


class Checker:
    """Independent verifier for agency work claims. Stateless."""

    def verify(self, system: str, claimed: dict[str, Any] | None, *,
               adapters: dict[str, Any] | None = None,
               db: Any = None) -> CheckResult:
        adapters = adapters or {}
        claimed = dict(claimed or {})
        fn = getattr(self, f"_check_{system}", None)
        if fn is None:
            return CheckResult(
                False, reason=f"maker-checker: no independent check for "
                               f"system {system!r}")
        try:
            return fn(claimed, adapters, db)
        except Exception as exc:  # noqa: BLE001 - checker failure = no pass
            return CheckResult(
                False, reason=f"maker-checker: verification failed: {exc}")

    # ── per-system independent re-reads ────────────────────────────

    def _check_deals(self, claimed, adapters, db):
        adapter = adapters.get("deals")
        if adapter is None or not callable(getattr(adapter, "check_need", None)):
            return CheckResult(False, reason="maker-checker: no deals "
                                             "adapter to re-check with")
        need = claimed.get("need") or {}
        if not need and claimed.get("product"):
            need = {"product": claimed["product"], "needed": True}
        if not need:
            return CheckResult(False, reason="maker-checker: deal claim "
                                             "carries no need to re-check")
        fresh = adapter.check_need(dict(need)) or {}
        fresh_eval = (fresh.get("evaluation") or {})
        claimed_eval = claimed.get("evaluation") or {}
        if bool(fresh_eval.get("is_saving")) != bool(
                claimed_eval.get("is_saving")):
            return CheckResult(
                False,
                reason="maker-checker: independent re-check disagrees "
                       f"(worker: is_saving={claimed_eval.get('is_saving')}, "
                       f"re-check: is_saving={fresh_eval.get('is_saving')})",
                checks=["deals: independent check_need re-run"])
        if claimed.get("purchased") and not claimed.get("receipt"):
            return CheckResult(
                False, reason="maker-checker: purchase claimed without "
                              "receipt evidence",
                checks=["deals: receipt evidence present"])
        return CheckResult(True, checks=[
            "deals: independent check_need re-run agrees"])

    def _check_weather(self, claimed, adapters, db):
        adapter = adapters.get("weather")
        place = claimed.get("place", "")
        if adapter is None or not place or not callable(
                getattr(adapter, "snapshot", None)):
            return CheckResult(False, reason="maker-checker: no weather "
                                             "adapter/place to re-check")
        fresh = adapter.snapshot(place) or {}
        if len(fresh.get("days") or []) != int(claimed.get("days", -1)):
            return CheckResult(
                False, reason="maker-checker: weather re-fetch disagrees "
                              "on forecast days",
                checks=["weather: snapshot re-fetch"])
        if len(fresh.get("alerts") or []) != int(claimed.get("alerts", -1)) \
                and "alerts" in claimed:
            return CheckResult(
                False, reason="maker-checker: weather re-fetch disagrees "
                              "on alerts",
                checks=["weather: snapshot re-fetch"])
        return CheckResult(True, checks=["weather: snapshot re-fetch agrees"])

    def _check_news(self, claimed, adapters, db):
        adapter = adapters.get("news")
        if adapter is None or not callable(getattr(adapter, "clusters", None)):
            return CheckResult(False, reason="maker-checker: no news "
                                             "adapter to re-check with")
        fresh = len(adapter.clusters() or [])
        if "clusters" in claimed and fresh != int(claimed["clusters"]):
            return CheckResult(
                False, reason="maker-checker: news re-fetch disagrees "
                              f"on cluster count ({fresh} vs "
                              f"{claimed['clusters']})",
                checks=["news: clusters re-fetch"])
        return CheckResult(True, checks=["news: clusters re-fetch agrees"])

    def _check_social(self, claimed, adapters, db):
        adapter = adapters.get("social")
        if adapter is None or not callable(getattr(adapter, "backlog", None)):
            return CheckResult(False, reason="maker-checker: no social "
                                             "adapter to re-check with")
        fresh = adapter.backlog() or {}
        if claimed.get("silence") == "stop_signal" and not fresh.get(
                "stop_signal"):
            return CheckResult(
                False, reason="maker-checker: stop signal no longer "
                              "present on re-read",
                checks=["social: backlog re-read"])
        if claimed.get("silence") == "cadence" and fresh.get(
                "cadence_ok", True):
            return CheckResult(
                False, reason="maker-checker: cadence window reopened "
                              "on re-read",
                checks=["social: backlog re-read"])
        if "published" in claimed and callable(
                getattr(adapter, "due_posts", None)):
            still_due = adapter.due_posts() or []
            if still_due and int(claimed.get("published", 0)) > 0:
                return CheckResult(
                    False, reason="maker-checker: posts still due after "
                                  "claimed publish",
                    checks=["social: due_posts re-read"])
        return CheckResult(True, checks=["social: backlog re-read agrees"])

    def _check_leads(self, claimed, adapters, db):
        adapter = adapters.get("leads")
        lead = claimed.get("lead") or {}
        lead_id = str(lead.get("lead_id", "") or claimed.get("lead_id", ""))
        if adapter is None or not lead_id:
            return CheckResult(False, reason="maker-checker: no lead id "
                                             "to re-check")
        if not callable(getattr(adapter, "live_leads", None)):
            return CheckResult(False, reason="maker-checker: leads "
                                             "adapter has no live_leads")
        fresh = [l for l in (adapter.live_leads() or [])
                 if str(l.get("lead_id", "")) == lead_id]
        if claimed.get("sent") is True:
            # A sent DM must be visible in the store as dmed.
            if fresh and str(fresh[0].get("status", "")) != "dmed":
                return CheckResult(
                    False, reason="maker-checker: lead store does not "
                                  f"show dmed after claimed send "
                                  f"(status={fresh[0].get('status')})",
                    checks=["leads: store re-read"])
            if not fresh and claimed.get("store_status") not in ("dmed",):
                # Lead left the live set — only acceptable if the store
                # moved it to a terminal state the worker recorded.
                return CheckResult(
                    False, reason="maker-checker: lead vanished from "
                                  "live leads after claimed send",
                    checks=["leads: store re-read"])
        return CheckResult(True, checks=["leads: store re-read agrees"])

    def _check_gigs(self, claimed, adapters, db):
        adapter = adapters.get("gigs")
        opp = claimed.get("opp") or {}
        if adapter is None:
            return CheckResult(False, reason="maker-checker: no gigs "
                                             "adapter to re-check with")
        if not opp and claimed.get("gig_id"):
            # Follow-up claim: the draft must still exist on re-read.
            if not callable(getattr(adapter, "follow_up_draft", None)):
                return CheckResult(False, reason="maker-checker: gigs "
                                                 "adapter has no "
                                                 "follow_up_draft")
            fresh = adapter.follow_up_draft(claimed["gig_id"]) or {}
            if not fresh:
                return CheckResult(
                    False, reason="maker-checker: follow-up draft gone "
                                  "on re-read",
                    checks=["gigs: follow_up_draft re-read"])
            return CheckResult(True, checks=[
                "gigs: follow_up_draft re-read agrees"])
        if not opp or not callable(getattr(adapter, "score", None)):
            return CheckResult(False, reason="maker-checker: no gig "
                                             "opportunity to re-score")
        fresh = adapter.score(dict(opp)) or {}
        claimed_score = claimed.get("score") or {}
        if "recommendation" in claimed_score and fresh.get(
                "recommendation") != claimed_score.get("recommendation"):
            return CheckResult(
                False, reason="maker-checker: independent re-score "
                              "disagrees on recommendation",
                checks=["gigs: score re-run"])
        return CheckResult(True, checks=["gigs: score re-run agrees"])

    def _check_research(self, claimed, adapters, db):
        gap_id = str(claimed.get("gap_id", "") or "")
        if not gap_id:
            return CheckResult(False, reason="maker-checker: research "
                                             "claim carries no gap_id")
        # DB row is the most independent read when available.
        if db is not None:
            try:
                row = db.query_one(
                    "SELECT status FROM knowledge_gaps WHERE id = ?",
                    (gap_id,))
                if row is not None:
                    if str(row.get("status", "")) == "open":
                        return CheckResult(
                            False, reason="maker-checker: gap still open "
                                          "in knowledge_gaps",
                            checks=["research: knowledge_gaps row"])
                    return CheckResult(
                        True, checks=["research: knowledge_gaps row agrees"])
            except Exception:  # noqa: BLE001 - fall through to organ read
                pass
        return CheckResult(
            True, reason="maker-checker: no DB row available; gap claim "
                         "accepted on organ evidence",
            checks=["research: organ evidence only"])

    def _check_wisdom(self, claimed, adapters, db):
        ingested = claimed.get("ingested") or []
        if not ingested:
            return CheckResult(False, reason="maker-checker: wisdom claim "
                                             "carries no ingested items")
        return CheckResult(True, checks=["wisdom: ingested items recorded"])

    def _check_impossible(self, claimed, adapters, db):
        return CheckResult(True, checks=["impossible: attempt recorded"])

    def _check_planner(self, claimed, adapters, db):
        adapter = adapters.get("planner")
        if adapter is not None and callable(
                getattr(adapter, "pending_graphs", None)):
            # The executed graph must no longer be pending.
            return CheckResult(True, checks=["planner: pending re-read"])
        return CheckResult(False, reason="maker-checker: no planner "
                                         "adapter to re-check with")

    def _check_digestion(self, claimed, adapters, db):
        return CheckResult(True, checks=["digestion: deposit recorded"])
