"""Weekly paid-task hunt — the standing weekly layer over the hunter.

``agents/opportunities.py`` (the :class:`OpportunityHunter`) is the
finder: it hunts money-making opportunities of every kind, ranks them,
dedupes them and remembers what it has seen.  It has no weekly rhythm
and no Nigeria vetting — a scan result's ``regions`` are *inferred from
snippet text*, which is a guess, and a guess never reaches the owner
as a vetted pick.

This module is the layer the owner asked for (2026-10-22): once a
week, ahead of the morning briefing, the hunter's paid-task finds are
put through a fail-closed verification, and only the survivors land
in the briefing as a short ranked shortlist:

* **Nigerian signup eligibility** — verified from the source page
  itself: the fetched page must carry positive availability evidence
  (worldwide / work-from-anywhere / Nigeria named / …) and no hard
  exclusion (US-only, must-reside-in, …).  No evidence → dropped.
* **A payout method that works from Nigeria** — the page must name at
  least one qualifying rail (Payoneer, Wise, Grey, Deel, bank/wire
  transfer, crypto, …).  PayPal alone never qualifies: a Nigerian
  personal PayPal cannot receive payments, so a PayPal-only platform
  is a dead end dressed as an opportunity.

Everything else about the week is remembered in the module's own
state file (``weekly_hunt.json`` beside the hunter's seen-store):
which tasks were already shown (never re-surfaced), per-platform
verification history (platforms that keep failing the Nigeria checks
are deprioritized and only re-probed monthly — and a platform that
starts passing after failing is flagged as newly accepting), and the
running hit rate.  A week that verifies nothing stores an honest
empty shortlist — the briefing says the scan ran and found nothing
rather than padding the list.

Surfaces: the ``opportunity_hunt`` tool (run | status | latest), the
``opportunity hunt`` scheduler job (``agents/opportunity_jobs.py``,
Mondays 05:30 — ahead of the 07:00 briefing), and the
``PaidTasksProvider`` section in ``agents/morning_briefing.py``.
Delivery is the briefing's own: owner-only, no new surface.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

__all__ = [
    "SHORTLIST_MAX",
    "VERIFY_BUDGET",
    "FRESH_S",
    "WeeklyOpportunityHunt",
    "verify_candidate",
    "platform_of",
    "run_weekly_hunt",
    "register",
]

_log = logging.getLogger(__name__)

#: Entries in one week's shortlist — a briefing section, not a dump.
SHORTLIST_MAX = 5
#: Verification fetches per run, in hunter rank order.  Bounded so a
#: fat scan week cannot turn the job into an unbounded crawl.
VERIFY_BUDGET = 12
#: Consecutive failed verifications before a platform is deprioritized.
DEPRIORITIZE_STREAK = 3
#: A deprioritized platform is re-probed at most this often — platforms
#: do start accepting Nigerians, and the hunt should notice (the run
#: flags them ``newly_accepting`` when they do).
REPROBE_AFTER_S = 28 * 86400.0
#: The briefing shows the shortlist only while it is this week's.
FRESH_S = 8 * 86400.0
#: A source page yielding less text than this cannot carry evidence —
#: it is unverifiable by construction and fails closed.
MIN_EVIDENCE_CHARS = 200
#: Run records kept in state (one year of weeks).
MAX_RUNS_KEPT = 52

STATE_FILE = "weekly_hunt.json"

# ── verification evidence ────────────────────────────────────────────
# Everything below is matched against the FETCHED SOURCE PAGE text.
# Snippets and search-result blurbs never count as evidence.

#: Payout rails that actually pay out to Nigeria.  PayPal is
#: deliberately absent (see module docstring).
PAYOUT_RAILS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bpayoneer\b", re.I), "Payoneer"),
    (re.compile(r"\btransferwise\b|\bwise\.com\b"
                r"|\bpaid (?:via|through|with) wise\b", re.I), "Wise"),
    (re.compile(r"\bgrey\.co\b", re.I), "Grey"),
    (re.compile(r"\bdeel\b", re.I), "Deel"),
    (re.compile(r"\bcleva\b|\braenest\b", re.I), "Cleva/Raenest"),
    (re.compile(r"\busdt\b|\busdc\b|\bbitcoin\b|\bcryptocurrenc", re.I),
     "crypto"),
    (re.compile(r"international (?:bank|wire) transfer"
                r"|\bwire transfer\b|\bswift transfer\b"
                r"|\bbank transfer\b", re.I), "bank transfer"),
)

#: Positive availability evidence: the page says the work is open
#: beyond one country's residents.
_ELIGIBILITY_PATTERNS: tuple[re.Pattern, ...] = tuple(re.compile(p, re.I)
                                                     for p in (
    r"\bworldwide\b",
    r"work from anywhere",
    r"anywhere in the world",
    r"\ball countries\b",
    r"no location restrictions?",
    r"\bnigeria\b",
    r"\bglobally remote\b",
    r"remote[^.]{0,20}\bglobal\b",
    r"open to (?:applicants|candidates|freelancers|contributors)"
    r"[^.]{0,40}(?:worldwide|globally|anywhere)",
))

#: Hard exclusions: the page restricts who may sign up.  Any match
#: settles eligibility negatively, whatever else the page says.
_EXCLUSION_PATTERNS: tuple[re.Pattern, ...] = tuple(re.compile(p, re.I)
                                                    for p in (
    r"\b(?:u\.?s\.?a?\.?|united states|u\.?k\.?|united kingdom|canada"
    r"|australia|european union|eu|eea)(?: residents?)? only\b",
    r"must (?:be )?(?:located|resid(?:e|ing)|based) in (?:the )?"
    r"(?:u\.?s\.?a?\.?|united states|u\.?k\.?|united kingdom|canada"
    r"|australia)",
    r"only (?:open|available) to (?:residents of )?(?:the )?"
    r"(?:u\.?s\.?a?\.?|united states|u\.?k\.?|united kingdom|europe)",
    r"not (?:currently )?available in your (?:country|region)",
))


def platform_of(url: str) -> str:
    """The platform identity of a candidate: its bare host."""
    try:
        host = urlsplit(url or "").netloc.lower()
    except Exception:  # noqa: BLE001
        return ""
    return host[4:] if host.startswith("www.") else host


def _evidence(text: str, match: re.Match) -> str:
    """The matched phrase with a little context — the audit trail."""
    s = max(0, match.start() - 70)
    e = min(len(text), match.end() + 70)
    return re.sub(r"\s+", " ", text[s:e]).strip()


def _default_fetch(url: str) -> str:
    """Fetch + readability-extract one source page.  "" on ANY
    failure — walls, dead hosts and robots all read the same:
    no evidence, no verdict, dropped."""
    try:
        from ..tools.web import web_extract

        out = web_extract(url, max_chars=20000)
        return str((out or {}).get("text") or "").strip()
    except Exception:  # noqa: BLE001 - fail-closed by design
        _log.debug("opportunity hunt: extract failed for %s", url,
                   exc_info=True)
        return ""


def verify_candidate(opp: Any,
                     fetch_text: Callable[[str], str]) -> dict[str, Any]:
    """Vet ONE candidate against its own source page.

    Returns a verdict dict.  ``ok`` is True only when BOTH checks
    pass on evidence found in the fetched page:

    * ``eligibility`` — positive availability evidence and no hard
      exclusion;
    * ``payout`` — at least one Nigeria-workable payout rail named.

    Failure kinds: ``unverifiable`` (page unreadable/thin, or a check
    found no evidence either way) and ``excluded`` (the page itself
    restricts signup).  Both fail closed — the candidate is dropped.
    """
    verdict: dict[str, Any] = {
        "ok": False, "kind": "unverifiable", "reason": "",
        "eligibility_evidence": "", "payout_rail": "",
        "payout_evidence": "",
    }
    text = ""
    try:
        text = fetch_text(opp.url) or ""
    except Exception:  # noqa: BLE001 - a broken fetch is no evidence
        text = ""
    if len(text.strip()) < MIN_EVIDENCE_CHARS:
        verdict["reason"] = "source page unreadable or too thin to vet"
        return verdict
    for pat in _EXCLUSION_PATTERNS:
        m = pat.search(text)
        if m:
            verdict["kind"] = "excluded"
            verdict["reason"] = "page restricts who may sign up"
            verdict["eligibility_evidence"] = _evidence(text, m)
            return verdict
    m = None
    for pat in _ELIGIBILITY_PATTERNS:
        m = pat.search(text)
        if m:
            break
    if m is None:
        verdict["reason"] = "no availability evidence on the page"
        return verdict
    verdict["eligibility_evidence"] = _evidence(text, m)
    for pat, label in PAYOUT_RAILS:
        pm = pat.search(text)
        if pm:
            verdict["ok"] = True
            verdict["kind"] = "verified"
            verdict["payout_rail"] = label
            verdict["payout_evidence"] = _evidence(text, pm)
            return verdict
    verdict["reason"] = ("no Nigeria-workable payout method named "
                          "on the page")
    return verdict


# ── the weekly hunt ──────────────────────────────────────────────────

def _empty_state() -> dict[str, Any]:
    return {"version": 1, "shown": {}, "platforms": {},
            "runs": [], "latest": None,
            "stats": {"checks": 0, "passed": 0}}


class WeeklyOpportunityHunt:
    """One weekly pass: scan → verify → shortlist → remember.

    The hunter itself is used exactly as it ships (injected or built
    from settings); this class adds the vetting, the cross-run memory
    and the stored shortlist the briefing renders.  ``fetch_text``
    and ``now_fn`` are seams for tests.
    """

    def __init__(self, settings: Any = None, hunter: Any = None,
                 fetch_text: Optional[Callable[[str], str]] = None,
                 now_fn: Callable[[], float] = time.time) -> None:
        self.settings = settings
        if hunter is None:
            from .opportunities import OpportunityHunter

            hunter = OpportunityHunter(settings=settings)
        self.hunter = hunter
        self._fetch = fetch_text or _default_fetch
        self._now = now_fn

    @classmethod
    def from_context(cls, context: Any) -> "WeeklyOpportunityHunt":
        """Build from a runtime context: the hunter gets the context's
        web_search tool as its search seam (the same construction
        ``/money`` uses), so a scheduled run hunts live, not just the
        curated pool."""
        from .opportunities import OpportunityHunter

        settings = getattr(context, "settings", None)
        if settings is None:
            from ..core.config import get_settings

            settings = get_settings()
        search_fn = None
        tools = getattr(context, "tools", None) if context is not None \
            else None
        if tools is not None:
            def search_fn(q: str, n: int = 6):  # noqa: E306
                outcome = tools.call("web_search", query=q,
                                     max_results=n)
                if not outcome.ok:
                    err = outcome.error.message if outcome.error \
                        else "unknown"
                    raise RuntimeError(f"search failed: {err}")
                return list(outcome.unwrap().get("results") or [])

        hunter = OpportunityHunter(settings=settings,
                                   search_fn=search_fn)
        return cls(settings=settings, hunter=hunter)

    # -- state ---------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return Path(self.hunter.data_dir) / STATE_FILE

    def load_state(self) -> dict[str, Any]:
        p = self.state_path
        if p.exists():
            try:
                state = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(state, dict):
                    base = _empty_state()
                    base.update(state)
                    return base
            except Exception:  # noqa: BLE001 - corrupt state restarts
                _log.warning("opportunity hunt: state unreadable; "
                             "starting fresh")
        return _empty_state()

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(
            json.dumps(state, indent=1), encoding="utf-8")

    def latest(self) -> Optional[dict[str, Any]]:
        return self.load_state().get("latest")

    def status(self) -> dict[str, Any]:
        state = self.load_state()
        stats = state.get("stats") or {}
        checks = int(stats.get("checks") or 0)
        passed = int(stats.get("passed") or 0)
        deprioritized = sorted(
            host for host, ps in (state.get("platforms") or {}).items()
            if int(ps.get("fail_streak") or 0) >= DEPRIORITIZE_STREAK)
        latest = state.get("latest") or {}
        return {
            "runs": len(state.get("runs") or []),
            "shown_total": len(state.get("shown") or {}),
            "checks": checks, "passed": passed,
            "hit_rate": (round(passed / checks, 3) if checks else None),
            "deprioritized_platforms": deprioritized,
            "last_run_at": latest.get("run_at"),
            "last_run_entries": len(latest.get("entries") or []),
        }

    # -- the run --------------------------------------------------------
    def run(self, now: Optional[float] = None) -> dict[str, Any]:
        """Execute one weekly hunt and store its shortlist.

        Never raises: a scan failure is recorded as a failed run
        (the briefing reports it honestly) and returned as
        ``{"ok": False, ...}``.
        """
        now = float(now if now is not None else self._now())
        state = self.load_state()
        record: dict[str, Any] = {
            "at": now, "candidates": 0, "checks": 0, "verified": 0,
            "dropped_unverifiable": 0, "dropped_excluded": 0,
            "skipped_shown": 0, "skipped_deprioritized": 0,
            "hit_rate": None,
        }
        try:
            candidates = self.hunter.scan(kinds=["paid_task"])
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            _log.warning("opportunity hunt: scan failed: %s", exc)
            record["error"] = f"{type(exc).__name__}: {exc}"
            state["runs"] = (state.get("runs") or [])[-MAX_RUNS_KEPT:] \
                + [record]
            state["latest"] = {"run_at": now, "entries": [],
                               "error": record["error"],
                               "newly_accepting": []}
            self._save_state(state)
            return {"ok": False, "run": record, "entries": 0}

        record["candidates"] = len(candidates)
        shown = state.setdefault("shown", {})
        platforms = state.setdefault("platforms", {})
        entries: list[dict[str, Any]] = []
        newly: list[str] = []
        checks = passes = 0
        for opp in candidates:
            if len(entries) >= SHORTLIST_MAX \
                    or checks >= VERIFY_BUDGET:
                break
            if opp.id in shown:
                record["skipped_shown"] += 1
                continue
            host = platform_of(opp.url)
            ps = platforms.setdefault(host, {
                "checks": 0, "passed": 0, "fail_streak": 0,
                "last_checked": 0.0, "last_verdict": "",
                "ever_failed": False})
            if int(ps.get("fail_streak") or 0) >= DEPRIORITIZE_STREAK \
                    and now - float(ps.get("last_checked") or 0) \
                    < REPROBE_AFTER_S:
                record["skipped_deprioritized"] += 1
                continue
            verdict = verify_candidate(opp, self._fetch)
            checks += 1
            ps["checks"] = int(ps.get("checks") or 0) + 1
            ps["last_checked"] = now
            if verdict["ok"]:
                passes += 1
                ps["passed"] = int(ps.get("passed") or 0) + 1
                if ps.get("ever_failed") and host not in newly:
                    newly.append(host)
                ps["fail_streak"] = 0
                ps["last_verdict"] = "verified"
                entries.append({
                    "id": opp.id, "title": opp.title, "url": opp.url,
                    "platform": host,
                    "payout_text": opp.payout_text or "",
                    "payout_usd": opp.payout_usd,
                    "score": opp.score,
                    "rail": verdict["payout_rail"],
                    "eligibility_evidence":
                        verdict["eligibility_evidence"],
                    "payout_evidence": verdict["payout_evidence"],
                })
            else:
                ps["fail_streak"] = int(ps.get("fail_streak") or 0) + 1
                ps["ever_failed"] = True
                ps["last_verdict"] = verdict["kind"]
                if verdict["kind"] == "excluded":
                    record["dropped_excluded"] += 1
                else:
                    record["dropped_unverifiable"] += 1
        record["checks"] = checks
        record["verified"] = passes
        record["hit_rate"] = (round(passes / checks, 3)
                              if checks else None)
        stats = state.setdefault("stats", {"checks": 0, "passed": 0})
        stats["checks"] = int(stats.get("checks") or 0) + checks
        stats["passed"] = int(stats.get("passed") or 0) + passes
        for e in entries:
            shown[e["id"]] = {"at": now, "platform": e["platform"],
                              "title": e["title"]}
        state["runs"] = (state.get("runs") or [])[-MAX_RUNS_KEPT:] \
            + [record]
        state["latest"] = {"run_at": now, "entries": entries,
                            "error": "", "newly_accepting": newly}
        self._save_state(state)
        _log.info("opportunity hunt: %d candidates, %d checked, "
                  "%d verified, %d shortlisted",
                  record["candidates"], checks, passes, len(entries))
        return {"ok": True, "run": record, "entries": len(entries),
                "newly_accepting": newly}


def run_weekly_hunt(context: Any) -> dict[str, Any]:
    """The scheduler/tool entry: one hunt on the context's organs."""
    return WeeklyOpportunityHunt.from_context(context).run()


def register(registry: Any) -> None:
    """Register the ``opportunity_hunt`` tool (run | status | latest).

    The weekly scheduler job binds ``action=run``; ``status`` reports
    the run history, hit rate and deprioritized platforms; ``latest``
    returns the stored shortlist the briefing renders.
    """
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "opportunity_hunt",
        description=(
            "Weekly paid-task hunt over the OpportunityHunter: scans "
            "global paid tasks, verifies each against its source page "
            "for Nigerian signup eligibility AND a payout method that "
            "works from Nigeria (unverifiable entries are dropped, "
            "never guessed), and stores a short ranked shortlist that "
            "lands in the morning briefing. action=run (the weekly "
            "job) | status (runs, hit rate, deprioritized platforms) "
            "| latest (the stored shortlist)."),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — run|status|latest",
        },
    )
    def opportunity_hunt(*, action: str = "status") -> dict[str, Any]:
        act = (action or "status").strip().lower()
        hunt = WeeklyOpportunityHunt.from_context(context)
        if act == "run":
            return hunt.run()
        if act == "latest":
            return {"ok": True, "latest": hunt.latest()}
        if act == "status":
            return {"ok": True, **hunt.status()}
        return {"ok": False, "error": f"unknown action {act!r}"}
