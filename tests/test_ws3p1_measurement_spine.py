"""WS3 P1 — the measurement spine.

Pins, end to end:
  (a) the refresh job registers via the ensure_* pattern, idempotently,
      as a routine tool job — and the weekly-content ensure next to it
      is untouched;
  (b) a posted row's metrics are refreshed from its adapter and written
      back WITH their fetch time;
  (c) an adapter that cannot produce metrics is accounted
      ``unavailable`` by platform name — its rows are never zeroed;
  (d) ``optimal_windows`` provably reads measured data: seeded publish
      metrics rank one slot first; after a refresh the measured numbers
      move the top window to the other slot;
  (e) the briefs outcome loop gets its first production caller: a
      post's first fair-age refresh feeds exactly one outcome into
      ``BriefStore.record_outcome`` when the post links to exactly one
      brief draft — and refuses to guess on zero/several candidates.

Plus the honesty edges: an empty fetch never overwrites stored
metrics, a raising adapter is counted and named, a just-refreshed row
is skipped until the next cycle, and a too-young post is measured but
not yet fed as an outcome.

No network: every adapter is a fake; the shipped adapters appear only
in the capability-shape pin (constructor + MRO, no calls).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from nomorals.core.ids import new_id
from nomorals.social.base import PlatformAdapter, PostResult
from nomorals.social.manager import SocialManager
from nomorals.social.metrics import adapter_supports_metrics, refresh_post_metrics

NOW_DT = datetime(2026, 11, 9, 12, 0)  # a Monday
NOW = NOW_DT.timestamp()
HOUR = 3600.0
DAY = 86400.0


# ── fakes + harness ──────────────────────────────────────────────────


class MetricsAdapter(PlatformAdapter):
    """Capable adapter: serves canned metrics per external id."""

    def __init__(self, name: str, results: dict | None = None):
        self.name = name
        self.results = dict(results or {})
        self.calls: list[str] = []

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="unused")

    def metrics(self, account, external_id):
        self.calls.append(external_id)
        return dict(self.results.get(external_id, {}))


class NoMetricsAdapter(PlatformAdapter):
    """Incapable adapter: inherits the base metrics ({} for everyone)."""

    def __init__(self, name: str):
        self.name = name

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="unused")


class EmptyAdapter(PlatformAdapter):
    """Capable on paper, answers no data (the shipped best-effort {})."""

    def __init__(self, name: str):
        self.name = name

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="unused")

    def metrics(self, account, external_id):
        return {}


class BoomAdapter(PlatformAdapter):
    """Capable adapter whose fetch explodes."""

    def __init__(self, name: str):
        self.name = name

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="unused")

    def metrics(self, account, external_id):
        raise RuntimeError("platform exploded")


def _db(path=None):
    from nomorals.storage.db import Database

    db = Database(path or ":memory:")
    db.migrate()
    return db


def _manager(db, adapters) -> SocialManager:
    ctx = SimpleNamespace(db=db, tools=None, extras={})
    return SocialManager(ctx, adapters=adapters, enforce_policy=False)


def _seed_post(manager, account, *, content="hello world",
               external_id="ext-1", posted_at=NOW - 2 * HOUR,
               metrics=None, platform=None, account_id=None):
    row = manager.posts.create({
        "id": new_id(), "platform": platform or account.platform,
        "account_id": account_id or account.id,
        "content": content, "media_paths": [], "status": "posted",
        "external_id": external_id, "posted_at": posted_at,
        "created_at": posted_at, "metrics": metrics or {},
    })
    return row["id"]


def _row(manager, post_id):
    return manager.posts.get(post_id)


# ── (a) job registration ─────────────────────────────────────────────


def _sched_ctx(tmp_path, db):
    from nomorals.tools.registry import ToolRegistry

    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(workspace_dir=str(tmp_path),
                                 home_path=tmp_path),
        extras={}, router=None, tools=None)
    registry = ToolRegistry(ctx)
    ctx.tools = registry
    return ctx, registry


class TestEnsureJobs:
    def test_metrics_job_registers_routine_and_idempotent(self, tmp_path):
        from nomorals.agents.scheduler import Scheduler
        from nomorals.social.jobs import ensure_social_metrics_jobs

        ctx, _registry = _sched_ctx(tmp_path, _db())
        sched = Scheduler(ctx)
        out = ensure_social_metrics_jobs(sched)
        assert out == [{"name": "social metrics refresh",
                        "scheduled": True, "job_id": out[0]["job_id"]}]
        rows = [j for j in sched.list_jobs()
                if j.get("name") == "social metrics refresh"]
        assert len(rows) == 1
        payload_row = ctx.db.query_one(
            "SELECT payload, routine FROM schedule_jobs WHERE name = ?",
            ("social metrics refresh",))
        import json

        assert json.loads(payload_row["payload"]) == {
            "tool": "social_metrics_refresh", "args": {}}
        assert payload_row["routine"]  # a chore, not a report
        again = ensure_social_metrics_jobs(sched)
        assert again == [{"name": "social metrics refresh",
                          "already_scheduled": True}]
        assert len([j for j in sched.list_jobs()
                    if j.get("name") == "social metrics refresh"]) == 1

    def test_weekly_content_ensure_untouched(self, tmp_path):
        from nomorals.agents.scheduler import Scheduler
        from nomorals.social.jobs import ensure_social_metrics_jobs, ensure_weekly_content_jobs

        ctx, _registry = _sched_ctx(tmp_path, _db())
        sched = Scheduler(ctx)
        ensure_social_metrics_jobs(sched)
        out = ensure_weekly_content_jobs(sched)
        assert out == [{"name": "weekly content", "scheduled": True,
                        "job_id": out[0]["job_id"]}]


class TestJobEndToEnd:
    def test_tool_registered_and_directly_callable(self, tmp_path):
        _ctx, registry = _sched_ctx(tmp_path, _db())
        registry.register_builtins()
        assert registry.get("social_metrics_refresh") is not None
        failed = {f["module"] for f in registry._failed_modules}
        assert "nomorals.tools.social" not in failed
        out = registry.call("social_metrics_refresh")
        assert out.ok, out
        assert out.value["ok"] is True
        assert out.value["considered"] == 0

    def test_scheduled_job_runs_the_spine(self, tmp_path):
        from nomorals.agents.scheduler import Scheduler
        from nomorals.social.jobs import ensure_social_metrics_jobs

        # File-backed: Database connections are per-thread and the
        # scheduler runs tool payloads on a worker thread — a :memory:
        # db would be a different (empty) database over there. The
        # production database is a file, like this one.
        ctx, registry = _sched_ctx(tmp_path, _db(str(tmp_path / "ctx.db")))
        registry.register_builtins()
        sched = Scheduler(ctx)
        sched.notifier = SimpleNamespace(
            publish=lambda *a, **kw: {"delivered": True})
        ensure_social_metrics_jobs(sched)
        row = [j for j in sched.list_jobs()
               if j.get("name") == "social metrics refresh"][0]
        out = sched.run_now(row["id"])
        assert out["ok"] is True, out
        runs = ctx.db.query(
            "SELECT ok FROM schedule_runs WHERE job_id = ?", (row["id"],))
        assert runs and all(r["ok"] for r in runs)


# ── capability shape ─────────────────────────────────────────────────


class TestCapabilityShape:
    def test_base_adapter_contract(self):
        assert NoMetricsAdapter("x").metrics(None, "e") == {}
        assert adapter_supports_metrics(NoMetricsAdapter("x")) is False
        assert adapter_supports_metrics(MetricsAdapter("x")) is True
        assert adapter_supports_metrics(EmptyAdapter("x")) is True

    def test_shipped_adapters(self):
        from nomorals.social.adapters import bluesky, mastodon, postiz

        assert adapter_supports_metrics(mastodon.Adapter()) is True
        assert adapter_supports_metrics(bluesky.Adapter()) is False
        assert adapter_supports_metrics(postiz.Adapter()) is False


# ── (b) + honesty edges: the refresh itself ──────────────────────────


class TestRefresh:
    def test_refresh_writes_metrics_with_fetch_time(self):
        adapter = MetricsAdapter(
            "mastodon", {"ext-1": {"likes": 41, "replies": 3}})
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        post_id = _seed_post(manager, account)

        report = refresh_post_metrics(manager, now=NOW)
        assert report["considered"] == 1
        assert report["refreshed"] == 1
        assert adapter.calls == ["ext-1"]
        row = _row(manager, post_id)
        assert row["metrics"] == {"likes": 41, "replies": 3}
        assert row["metadata"]["metrics_fetched_at"] == NOW

    def test_recently_refreshed_row_is_skipped_until_next_cycle(self):
        adapter = MetricsAdapter("mastodon", {"ext-1": {"likes": 1}})
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        _seed_post(manager, account)

        refresh_post_metrics(manager, now=NOW)
        second = refresh_post_metrics(manager, now=NOW + HOUR)
        assert second["skipped_recent"] == 1
        assert second["refreshed"] == 0
        assert adapter.calls == ["ext-1"]  # not re-fetched
        third = refresh_post_metrics(manager, now=NOW + 7 * HOUR)
        assert third["refreshed"] == 1
        assert adapter.calls == ["ext-1", "ext-1"]

    def test_unavailable_platform_named_and_never_zeroed(self):
        adapter = NoMetricsAdapter("quietnet")
        manager = _manager(_db(), {"quietnet": adapter})
        account = manager.connect("quietnet", "@me")
        post_id = _seed_post(manager, account, metrics={"likes": 7})

        report = refresh_post_metrics(manager, now=NOW)
        assert report["unavailable"] == 1
        assert report["unavailable_platforms"] == ["quietnet"]
        assert report["refreshed"] == 0
        row = _row(manager, post_id)
        assert row["metrics"] == {"likes": 7}  # untouched, not zeroed
        assert "metrics_fetched_at" not in (row["metadata"] or {})

    def test_empty_fetch_leaves_stored_metrics_untouched(self):
        adapter = EmptyAdapter("mastodon")
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        post_id = _seed_post(manager, account, metrics={"likes": 9})

        report = refresh_post_metrics(manager, now=NOW)
        assert report["empty"] == 1
        assert report["refreshed"] == 0
        assert _row(manager, post_id)["metrics"] == {"likes": 9}

    def test_raising_adapter_counted_and_named(self):
        adapter = BoomAdapter("mastodon")
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        post_id = _seed_post(manager, account, metrics={"likes": 5})

        report = refresh_post_metrics(manager, now=NOW)
        assert report["failed"] == 1
        assert report["failures"][0]["platform"] == "mastodon"
        assert report["failures"][0]["external_id"] == "ext-1"
        assert "platform exploded" in report["failures"][0]["error"]
        assert _row(manager, post_id)["metrics"] == {"likes": 5}

    def test_missing_account_and_adapter_counted(self):
        adapter = MetricsAdapter("mastodon", {})
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        orphan = manager.connect("orphan", "@me")  # no adapter for it
        _seed_post(manager, account, external_id="ghost",
                   account_id="no-such-account")
        _seed_post(manager, orphan, external_id="no-adapter")

        report = refresh_post_metrics(manager, now=NOW)
        assert report["considered"] == 2
        assert report["skipped_no_account"] == 1
        assert report["skipped_no_adapter"] == 1

    def test_window_excludes_too_young_and_too_old(self):
        adapter = MetricsAdapter("mastodon", {"a": {"likes": 1},
                                              "b": {"likes": 1}})
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        _seed_post(manager, account, external_id="a",
                   posted_at=NOW - 600)  # 10 minutes old
        _seed_post(manager, account, external_id="b",
                   posted_at=NOW - 100 * DAY)  # beyond the 90d window
        report = refresh_post_metrics(manager, now=NOW)
        assert report["considered"] == 0
        assert adapter.calls == []


# ── (d) optimal windows read measured data ───────────────────────────


class TestOptimalWindowsMove:
    def test_windows_move_after_refresh(self):
        from nomorals.social.content_pipeline import optimal_windows

        adapter = MetricsAdapter("mastodon", {})
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        # Slot A: Tuesdays 09:00, small publish-time engagement.
        # Slot B: Thursdays 21:00, no publish-time engagement — but the
        # platform's real numbers for B are large.
        for day in (13, 20, 27):
            ts = datetime(2026, 10, day, 9, 0).timestamp()
            _seed_post(manager, account, external_id=f"a-{day}",
                       posted_at=ts, metrics={"likes": 2})
        ts = datetime(2026, 11, 3, 9, 0).timestamp()
        _seed_post(manager, account, external_id="a-3",
                   posted_at=ts, metrics={"likes": 2})
        for day in (15, 22, 29):
            ts = datetime(2026, 10, day, 21, 0).timestamp()
            _seed_post(manager, account, external_id=f"b-{day}",
                       posted_at=ts, metrics={})
        ts = datetime(2026, 11, 5, 21, 0).timestamp()
        _seed_post(manager, account, external_id="b-5",
                   posted_at=ts, metrics={})
        adapter.results.update(
            {f"a-{d}": {"likes": 2} for d in (13, 20, 27)})
        adapter.results["a-3"] = {"likes": 2}
        adapter.results.update(
            {f"b-{d}": {"likes": 100} for d in (15, 22, 29)})
        adapter.results["b-5"] = {"likes": 100}

        before, from_data = optimal_windows(
            manager.db, days=90, n=1, now=NOW_DT)
        assert from_data is True
        assert before == [datetime(2026, 11, 10, 9, 0)]  # next Tue 09:00

        from nomorals.marketing.briefs import BriefStore

        store = BriefStore(db_path=":memory:")
        report = refresh_post_metrics(manager, now=NOW, brief_store=store)
        assert report["refreshed"] == 8
        assert report["outcomes_unlinked"] == 8  # no brief drafts exist

        after, from_data = optimal_windows(
            manager.db, days=90, n=1, now=NOW_DT)
        assert from_data is True
        assert after == [datetime(2026, 11, 12, 21, 0)]  # next Thu 21:00
        assert after != before


# ── (e) the briefs outcome feed ──────────────────────────────────────

MOCK_RESULTS = [
    {"title": "Lagos rents explained",
     "snippet": "How much is rent in Lagos? 2 bedroom flats in Yaba "
                "cost around 2m per year. What should you look for "
                "when renting? Always inspect before paying."},
    {"title": "Yaba rental guide",
     "snippet": "Why are Yaba rents rising? The area is close to tech "
                "hubs. How do you negotiate with Lagos landlords? Get "
                "everything in writing."},
]


def _brief_store(tmp_path):
    from nomorals.marketing.briefs import BriefStore

    return BriefStore(db_path=str(tmp_path / "briefs.db"))


def _brief_draft(store):
    brief = store.build_brief("Lagos rents",
                              search_fn=lambda topic: MOCK_RESULTS)
    draft = store.draft_from_brief(brief)
    assert draft is not None
    return draft


class TestBriefsFeed:
    def test_first_outcome_feeds_briefs_exactly_once(self, tmp_path):
        store = _brief_store(tmp_path)
        draft = _brief_draft(store)
        adapter = MetricsAdapter(
            "x", {"ext-9": {"likes": 30, "replies": 2}})
        manager = _manager(_db(), {"x": adapter})
        account = manager.connect("x", "@me")
        post_id = _seed_post(
            manager, account, content=draft.content, external_id="ext-9",
            posted_at=NOW - 2 * DAY)

        report = refresh_post_metrics(manager, now=NOW, brief_store=store)
        assert report["refreshed"] == 1
        assert report["outcomes_fed"] == 1
        fed = store.get_draft(draft.draft_id)
        assert fed.outcome_metrics["likes"] == 30
        assert fed.outcome_metrics["engagement_score"] == 32.0
        assert _row(manager, post_id)["metadata"]["metrics_outcome"] \
            == "fed"
        feedback = store._db.execute(
            "SELECT COUNT(*) AS n FROM score_feedback").fetchone()
        assert feedback["n"] == 1

        # A later cycle re-measures but never re-feeds the outcome.
        later = refresh_post_metrics(
            manager, now=NOW + 7 * HOUR, brief_store=store)
        assert later["refreshed"] == 1
        assert later["outcomes_fed"] == 0
        feedback = store._db.execute(
            "SELECT COUNT(*) AS n FROM score_feedback").fetchone()
        assert feedback["n"] == 1

    def test_unlinked_post_stamps_unlinked(self, tmp_path):
        store = _brief_store(tmp_path)  # no drafts at all
        adapter = MetricsAdapter("x", {"ext-1": {"likes": 3}})
        manager = _manager(_db(), {"x": adapter})
        account = manager.connect("x", "@me")
        post_id = _seed_post(manager, account, posted_at=NOW - 2 * DAY)

        report = refresh_post_metrics(manager, now=NOW, brief_store=store)
        assert report["outcomes_unlinked"] == 1
        assert _row(manager, post_id)["metadata"]["metrics_outcome"] \
            == "unlinked"

    def test_ambiguous_link_is_never_fed(self, tmp_path):
        store = _brief_store(tmp_path)
        first = _brief_draft(store)
        second = _brief_draft(store)  # same brief → identical content
        assert first.draft_id != second.draft_id
        assert first.content == second.content
        adapter = MetricsAdapter("x", {"ext-1": {"likes": 12}})
        manager = _manager(_db(), {"x": adapter})
        account = manager.connect("x", "@me")
        _seed_post(manager, account, content=first.content,
                   posted_at=NOW - 2 * DAY)

        report = refresh_post_metrics(manager, now=NOW, brief_store=store)
        assert report["outcomes_ambiguous"] == 1
        assert report["outcomes_fed"] == 0
        assert store.get_draft(first.draft_id).outcome_metrics == {}
        assert store.get_draft(second.draft_id).outcome_metrics == {}

    def test_young_post_is_measured_but_not_fed(self, tmp_path):
        store = _brief_store(tmp_path)
        draft = _brief_draft(store)
        adapter = MetricsAdapter("x", {"ext-1": {"likes": 4}})
        manager = _manager(_db(), {"x": adapter})
        account = manager.connect("x", "@me")
        post_id = _seed_post(
            manager, account, content=draft.content,
            posted_at=NOW - 2 * HOUR)

        report = refresh_post_metrics(manager, now=NOW, brief_store=store)
        assert report["refreshed"] == 1
        assert report["outcomes_fed"] == 0
        meta = _row(manager, post_id)["metadata"] or {}
        assert "metrics_outcome" not in meta  # verdict waits for 24h
        assert store.get_draft(draft.draft_id).outcome_metrics == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
