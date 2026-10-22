"""Weekly paid-task hunt (agents/opportunity_hunt.py) — offline tests.

The network lives behind two seams and both are faked here: the
hunter (a stub returning hand-built candidates, plus one pass over
the REAL hunter's curated pool) and ``fetch_text`` (a dict of canned
source pages).  What is pinned:

* the eligibility filter is fail-closed — a candidate with no
  availability evidence, a hard exclusion on its own page, a thin /
  unreadable page, or only a Nigeria-unworkable payout method
  (PayPal alone) is DROPPED, never guessed into the shortlist;
* a verified candidate lands with its evidence + payout rail;
* shown tasks are never re-surfaced across runs;
* platform learning: repeated failures deprioritize a platform
  (its candidates stop being fetched), and a platform that starts
  passing after failing is flagged newly-accepting;
* an all-dropped week stores an honest EMPTY shortlist, and the
  briefing provider renders the "scan ran, found nothing" line —
  while a stale or never-run week renders no section at all;
* the scheduler job registers with the exact spec + payload, and
  the ``opportunity_hunt`` tool binding binds against the real
  registry (the pass-#75 contract shape).
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents import opportunities as O
from nomorals.agents.opportunity_hunt import (
    WeeklyOpportunityHunt, platform_of, verify_candidate)
from nomorals.agents.opportunity_jobs import ensure_opportunity_jobs

GOOD_PAGE = (
    "Join our contributor program — open worldwide. Work from "
    "anywhere in the world, on your own schedule, testing websites "
    "and apps for clients everywhere. New contributors sign up in "
    "minutes. Contributors are paid via Payoneer every Friday for "
    "the previous week's completed tasks, no minimum threshold."
)
EXCLUDED_PAGE = (
    "This panel is open to United States residents only. You must "
    "reside in the United States and hold a US bank account to take "
    "part in paid studies on this platform. Panel members are paid "
    "via Payoneer at the end of each month for completed studies."
)
NO_EVIDENCE_PAGE = (
    "A long page about the history of bread making, the chemistry "
    "of fermentation, regional grain varieties, and the social role "
    "of bakeries in nineteenth century towns, with many paragraphs "
    "of detail that say nothing about jobs, signups, eligibility, "
    "or how anybody on this site could ever possibly be paid."
)
PAYPAL_ONLY_PAGE = (
    "Testers wanted worldwide! Work from anywhere testing mobile "
    "apps and games for studios around the globe. Sign up free and "
    "start your first paid test today. All payments are sent via "
    "PayPal within seven days of completing each testing cycle."
)
THIN_PAGE = "Sign up now."


def _opp(url: str, title: str = "Paid testing gig", **kw) -> O.Opportunity:
    base = dict(kind="paid_task", title=title, source=platform_of(url),
                url=url, payout_text="earn $25 per test",
                effort="low", regions=["global"])
    base.update(kw)
    return O.Opportunity(**base)


class _StubHunter:
    """The hunter's exact surface the weekly layer uses."""

    def __init__(self, data_dir: Path, opps: list) -> None:
        self._data_dir = Path(data_dir)
        self._opps = opps

    @property
    def data_dir(self) -> Path:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir

    def scan(self, kinds=None):
        assert kinds == ["paid_task"]
        return list(self._opps)


class _Fetch:
    def __init__(self, pages: dict) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        return self.pages.get(url, "")


def _hunt(tmp_path, opps, pages, **kw):
    hunter = _StubHunter(tmp_path / "data", opps)
    fetch = _Fetch(pages)
    hunt = WeeklyOpportunityHunt(hunter=hunter, fetch_text=fetch, **kw)
    return hunt, fetch


# ── the verifier, candidate by candidate ─────────────────────────────

def test_verify_good_page_passes_with_evidence():
    opp = _opp("https://good.example/apply")
    v = verify_candidate(opp, lambda url: GOOD_PAGE)
    assert v["ok"] is True
    assert v["payout_rail"] == "Payoneer"
    assert "worldwide" in v["eligibility_evidence"].lower()
    assert "payoneer" in v["payout_evidence"].lower()


def test_verify_excluded_page_dropped_as_excluded():
    opp = _opp("https://usonly.example/join")
    v = verify_candidate(opp, lambda url: EXCLUDED_PAGE)
    assert v["ok"] is False
    assert v["kind"] == "excluded"


def test_verify_no_evidence_dropped_unverifiable():
    opp = _opp("https://bread.example/about")
    v = verify_candidate(opp, lambda url: NO_EVIDENCE_PAGE)
    assert v["ok"] is False
    assert v["kind"] == "unverifiable"


def test_verify_paypal_only_dropped():
    # Worldwide availability is not enough: PayPal alone cannot pay
    # out to a Nigerian personal account, so the rail check fails.
    opp = _opp("https://paypalonly.example/test")
    v = verify_candidate(opp, lambda url: PAYPAL_ONLY_PAGE)
    assert v["ok"] is False
    assert "payout" in v["reason"]


def test_verify_thin_or_dead_page_fails_closed():
    opp = _opp("https://thin.example/")
    assert verify_candidate(opp, lambda url: THIN_PAGE)["ok"] is False
    assert verify_candidate(opp, lambda url: "")["ok"] is False

    def boom(url):
        raise RuntimeError("network down")

    assert verify_candidate(opp, boom)["ok"] is False


# ── the weekly run ───────────────────────────────────────────────────

def test_run_shortlists_only_verified(tmp_path):
    good = _opp("https://good.example/apply", title="Good gig")
    bad = _opp("https://bread.example/about", title="No evidence gig")
    usonly = _opp("https://usonly.example/join", title="US only gig")
    pages = {good.url: GOOD_PAGE, bad.url: NO_EVIDENCE_PAGE,
             usonly.url: EXCLUDED_PAGE}
    hunt, _ = _hunt(tmp_path, [good, bad, usonly], pages)
    out = hunt.run(now=1000.0)
    assert out["ok"] is True
    assert out["entries"] == 1
    rec = out["run"]
    assert rec["candidates"] == 3
    assert rec["verified"] == 1
    assert rec["dropped_unverifiable"] == 1
    assert rec["dropped_excluded"] == 1
    assert rec["hit_rate"] == round(1 / 3, 3)
    latest = hunt.latest()
    assert [e["title"] for e in latest["entries"]] == ["Good gig"]
    assert latest["entries"][0]["rail"] == "Payoneer"


def test_shown_tasks_never_resurface(tmp_path):
    good = _opp("https://good.example/apply")
    hunt, _ = _hunt(tmp_path, [good], {good.url: GOOD_PAGE})
    first = hunt.run(now=1000.0)
    assert first["entries"] == 1
    second = hunt.run(now=2000.0)
    assert second["entries"] == 0
    assert second["run"]["skipped_shown"] == 1
    assert hunt.latest()["entries"] == []
    assert hunt.status()["shown_total"] == 1


def test_empty_week_is_stored_honestly(tmp_path):
    bad = _opp("https://bread.example/about")
    hunt, _ = _hunt(tmp_path, [bad], {bad.url: NO_EVIDENCE_PAGE})
    out = hunt.run(now=1000.0)
    assert out["ok"] is True
    assert out["entries"] == 0
    latest = hunt.latest()
    assert latest["entries"] == []
    assert latest["error"] == ""


def test_scan_failure_is_a_recorded_failed_run(tmp_path):
    class BoomHunter(_StubHunter):
        def scan(self, kinds=None):
            raise RuntimeError("search backend down")

    hunter = BoomHunter(tmp_path / "data", [])
    hunt = WeeklyOpportunityHunt(hunter=hunter, fetch_text=_Fetch({}))
    out = hunt.run(now=1000.0)
    assert out["ok"] is False
    latest = hunt.latest()
    assert latest["entries"] == []
    assert "search backend down" in latest["error"]


# ── platform learning ────────────────────────────────────────────────

def test_repeated_failures_deprioritize_platform(tmp_path):
    opps = [_opp(f"https://flaky.example/gig{i}", title=f"Gig {i}")
            for i in range(3)]
    pages = {o.url: NO_EVIDENCE_PAGE for o in opps}
    hunt, fetch = _hunt(tmp_path, opps, pages)
    hunt.run(now=1000.0)
    state = hunt.load_state()
    assert state["platforms"]["flaky.example"]["fail_streak"] == 3
    # Next week a fresh gig from the same platform is not even fetched.
    fresh = _opp("https://flaky.example/gig9", title="Gig 9")
    hunt2, fetch2 = _hunt(tmp_path, [fresh],
                          {fresh.url: GOOD_PAGE})
    hunt2.hunter._data_dir = hunt.hunter._data_dir
    out = hunt2.run(now=2000.0)
    assert out["run"]["skipped_deprioritized"] == 1
    assert fetch2.calls == []


def test_platform_that_starts_passing_is_flagged(tmp_path):
    opp = _opp("https://late.example/apply")
    hunt, fetch = _hunt(tmp_path, [opp], {opp.url: NO_EVIDENCE_PAGE})
    hunt.run(now=1000.0)
    fetch.pages[opp.url] = GOOD_PAGE
    out = hunt.run(now=2000.0)
    assert out["entries"] == 1
    assert out["newly_accepting"] == ["late.example"]
    assert hunt.latest()["newly_accepting"] == ["late.example"]


def test_real_hunter_curated_pool_end_to_end(tmp_path):
    # The REAL OpportunityHunter, offline (no search_fn → curated
    # pool only): outlier.ai pages vet clean, mindrift's canned page
    # carries no payout evidence and is dropped.
    hunter = O.OpportunityHunter()
    (tmp_path / "data").mkdir(parents=True)
    hunter._data_dir = tmp_path / "data"
    pages = {
        "https://outlier.ai/": GOOD_PAGE,
        "https://outlier.ai/coding-expertise": GOOD_PAGE,
        "https://mindrift.ai/": NO_EVIDENCE_PAGE,
    }
    hunt = WeeklyOpportunityHunt(hunter=hunter, fetch_text=_Fetch(pages))
    out = hunt.run(now=1000.0)
    assert out["ok"] is True
    assert out["run"]["candidates"] == 3
    assert out["entries"] == 2
    hosts = {e["platform"] for e in hunt.latest()["entries"]}
    assert hosts == {"outlier.ai"}


# ── the briefing provider ────────────────────────────────────────────

def _provider_ctx(tmp_path):
    data_root = tmp_path / "root"
    settings = SimpleNamespace(
        resolve=lambda rel: data_root / rel,
        home_path=tmp_path, workspace_dir=str(tmp_path))
    return SimpleNamespace(settings=settings, tools=None, db=None,
                           extras={}), data_root


def test_provider_renders_shortlist(tmp_path):
    ctx, data_root = _provider_ctx(tmp_path)
    good = _opp("https://good.example/apply", title="Good gig")
    hunter = _StubHunter(data_root / "data" / "opportunities", [good])
    hunt = WeeklyOpportunityHunt(hunter=hunter,
                                 fetch_text=_Fetch({good.url: GOOD_PAGE}))
    hunt.run()
    from nomorals.agents.morning_briefing import PaidTasksProvider

    sec = PaidTasksProvider().collect(ctx, 0.0)
    assert sec is not None
    assert sec.name == "paid_tasks"
    text = "\n".join(sec.lines)
    assert "Good gig" in text
    assert "Payoneer" in text
    assert "https://good.example/apply" in text


def test_provider_renders_honest_empty_note(tmp_path):
    ctx, data_root = _provider_ctx(tmp_path)
    bad = _opp("https://bread.example/about")
    hunter = _StubHunter(data_root / "data" / "opportunities", [bad])
    hunt = WeeklyOpportunityHunt(
        hunter=hunter, fetch_text=_Fetch({bad.url: NO_EVIDENCE_PAGE}))
    hunt.run()
    from nomorals.agents.morning_briefing import PaidTasksProvider

    sec = PaidTasksProvider().collect(ctx, 0.0)
    assert sec is not None
    text = "\n".join(sec.lines)
    assert "nothing" in text and "passed the Nigeria" in text


def test_provider_silent_without_a_fresh_run(tmp_path):
    ctx, _ = _provider_ctx(tmp_path)
    from nomorals.agents.morning_briefing import PaidTasksProvider

    # No state at all → no section (the scan simply hasn't run).
    assert PaidTasksProvider().collect(ctx, 0.0) is None


def test_provider_silent_when_shortlist_is_stale(tmp_path):
    ctx, data_root = _provider_ctx(tmp_path)
    good = _opp("https://good.example/apply")
    hunter = _StubHunter(data_root / "data" / "opportunities", [good])
    hunt = WeeklyOpportunityHunt(hunter=hunter,
                                 fetch_text=_Fetch({good.url: GOOD_PAGE}))
    hunt.run(now=time.time() - 9 * 86400)  # nine days ago
    from nomorals.agents.morning_briefing import PaidTasksProvider

    assert PaidTasksProvider().collect(ctx, 0.0) is None


def test_provider_is_in_the_composer(tmp_path):
    from nomorals.agents.morning_briefing import BriefingComposer

    names = {p.name for p in BriefingComposer().providers}
    assert "paid_tasks" in names


# ── job + tool wiring ────────────────────────────────────────────────

class _SchedNotifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True, "channel": "test"}


def _sched_ctx(tmp_path):
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    settings = SimpleNamespace(
        resolve=lambda rel: tmp_path / rel,
        home_path=tmp_path, workspace_dir=str(tmp_path))
    return SimpleNamespace(db=db, settings=settings, extras={},
                           tools=None)


def test_job_registers_exactly_once_with_exact_payload(tmp_path):
    from nomorals.agents.scheduler import Scheduler

    ctx = _sched_ctx(tmp_path)
    sched = Scheduler(ctx)
    sched.notifier = _SchedNotifier()
    out = ensure_opportunity_jobs(sched)
    assert [o["name"] for o in out] == ["opportunity hunt"]
    rows = [j for j in sched.list_jobs() if j["name"] == "opportunity hunt"]
    assert len(rows) == 1
    assert rows[0]["spec"] == "cron 30 5 * * MON"
    row = ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE name = ?",
        ("opportunity hunt",))
    assert json.loads(row["payload"]) == {
        "tool": "opportunity_hunt", "args": {"action": "run"}}
    # Idempotent: a second registration changes nothing.
    ensure_opportunity_jobs(sched)
    assert len([j for j in sched.list_jobs()
                if j["name"] == "opportunity hunt"]) == 1


def test_tool_registered_and_binding_binds(tmp_path):
    from nomorals.tools.registry import ToolRegistry

    ctx = _sched_ctx(tmp_path)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    # The hunter's own tools ride the same bridge registration.
    assert "money_scan" in registry._tools
    spec = registry._tools.get("opportunity_hunt")
    assert spec is not None, "opportunity_hunt tool not registered"
    # The scheduler's exact call shape binds against the real signature.
    inspect.signature(spec.fn).bind(action="run")
    outcome = registry.call("opportunity_hunt", action="status")
    assert outcome.ok, getattr(outcome.error, "message", outcome.error)
    body = outcome.unwrap()
    assert body["ok"] is True
    assert body["runs"] == 0


def test_job_runs_end_to_end_through_scheduler(tmp_path, monkeypatch):
    from nomorals.agents import opportunity_hunt as hunt_mod
    from nomorals.tools.registry import ToolRegistry

    monkeypatch.setattr(
        hunt_mod, "_default_fetch",
        lambda url: GOOD_PAGE if "outlier" in url else NO_EVIDENCE_PAGE)
    ctx = _sched_ctx(tmp_path)
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    # Offline + deterministic: the hunt's web_search seam answers
    # empty, so the scan runs on the hunter's curated pool only.
    monkeypatch.setattr(
        registry._tools["web_search"], "fn",
        lambda query="", max_results=6, **kw: {"results": []})
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(ctx)
    sched.notifier = _SchedNotifier()
    ensure_opportunity_jobs(sched)
    row = [j for j in sched.list_jobs()
           if j["name"] == "opportunity hunt"][0]
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    stored = ctx.db.query_one(
        "SELECT last_result FROM schedule_jobs WHERE id = ?",
        (row["id"],))
    assert not (stored["last_result"] or "").startswith("job failed"), \
        stored["last_result"]
    # The real hunter's curated pool was vetted and stored: the two
    # outlier.ai entries passed, mindrift was dropped unverifiable.
    state = json.loads(
        (tmp_path / "data" / "opportunities" / "weekly_hunt.json")
        .read_text())
    assert len(state["latest"]["entries"]) == 2
    assert state["runs"][0]["dropped_unverifiable"] == 1
