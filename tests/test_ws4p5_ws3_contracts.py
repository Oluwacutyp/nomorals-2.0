"""WS4 P5 — the WS3 contracts, end to end.

Pins, per the censuses (WS3 §9 / WS4 §5) and the pass brief:

(a) contract 1 — a current trend claim arrives in a pipeline run
    as a finding carrying its claim identity + basis; a fresh
    spike and a fresh unverified claim arrive LABELLED as what
    they are; a stale claim does not feed at all (the rule the
    adapter implements, stated in media/explore/feed.py: fresh
    claims feed, stale claims are excluded — decayed evidence
    does not linger into drafting);
(b) contract 2 — an adopted overlay niche is visible to drafting
    context (the adapter read, the plugin resolver, the weekly
    cell) and floor niches resolve beside it; a merely-proposed
    niche is invisible to all three;
(c) contract 3 — a finding with artifact-backed media produces a
    draft whose metadata artifact citation resolves through the
    WS1 artifact store to the same bytes; unverified or absent
    artifact ids cite nothing;
(d) contract 4 — a measured post descending from a claim records
    the outcome against the claim (claim key, post ref, measured
    engagement, at); an ambiguous join records ambiguous and
    feeds nothing; a claim with no descended post stays
    outcome-less; a too-young post is measured but not yet fed;
(e) production wiring — build_context installs the findings feed
    and the pipeline's REAL entry point receives it (the P6
    install-line proof shape), and the metrics refresh counts the
    claim feed through its production kwargs.

Real stores on both sides throughout; no network.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from nomorals.media.explore import ExploreStore
from nomorals.media.explore.feed import (
    claims_as_findings,
    drafting_niches,
    drafting_plugin_resolver,
    make_claims_feed,
    niche_drafting_context,
)
from nomorals.social import content_pipeline as cp_module
from nomorals.social.content_pipeline import set_findings_feed, weekly_content
from nomorals.social.drafts import DraftQueue

HOUR = 3600.0
DAY = 86400.0

CLEAN_COPY = ("Shipping agents that finish real work — here is what"
              " changed this week and why it matters.")


# ── helpers ──────────────────────────────────────────────────────


def _claim(store, *, key, kind="trend", at, text=None, basis=None,
           subject_kind="beat", subject=None, backing=None,
           freshness_s=None):
    store.record_claim({
        "claim_id": f"claim:{key}",
        "claim_key": key,
        "kind": kind,
        "subject": subject if subject is not None else key,
        "subject_kind": subject_kind,
        "text": text or f"{key}: sustained movement in the field",
        "source": "explore-trends",
        "observed_at": at,
        "scope": key,
        "confidence_basis": basis or "seen in 2 sweeps; classes video + web",
        "basis": {"sweeps": ["sweep-1", "sweep-2"],
                  "source_classes": ["video", "web"]},
        "freshness_s": freshness_s if freshness_s is not None else {
            "trend": 3 * DAY, "spike": 12 * HOUR,
            "unverified": DAY}[kind],
        "published_at": None,
        "contradicts": "",
        "entities": [],
        "backing": backing or {"observations": [],
                               "candidate_key": ""},
    })


@pytest.fixture()
def queue(tmp_path):
    q = DraftQueue(db_path=str(tmp_path / "drafts.db"))
    yield q
    q.close()


def _note(store, kind, name, *, beat, sweep_id, url, now,
          instrument="youtube_search"):
    return store.note_candidate(
        kind, name, via=f"sweep:{instrument}",
        platform="youtube" if kind == "channel" else "",
        beat=beat,
        evidence={"title": f"{name} — a video", "url": url,
                  "rank": 1, "view_count": 9000,
                  "instrument": instrument, "beat": beat},
        sweep_id=sweep_id, now=now)


def _gold_cluster(store, t0):
    """A corroborated cluster of the P4 shape: channel + topic
    under one beat, sharing URLs, sighted across two sweeps, with
    both candidates promoted on evidence."""
    for i, sweep in enumerate(["sweep-1", "sweep-2"]):
        t = t0 + i * 3600
        for url in ("https://youtu.be/g1", "https://youtu.be/g2"):
            _note(store, "channel", "NG Gold Room",
                  beat="gold trading", sweep_id=sweep, url=url, now=t)
            _note(store, "topic", "xauusd level breakdown",
                  beat="gold trading", sweep_id=sweep, url=url, now=t)
    for cand in store.candidates():
        store.update_candidate_verdict(
            cand["key"], status="promoted",
            standing={"sightings": 4, "score": 9})
    _claim(store, key="gold trading", at=t0 + 4000)
    return store


# ── (a) contract 1: claims feed the pipeline as findings ────────


class TestClaimsFeed:
    def test_fresh_trend_claim_projects_with_identity_and_basis(self):
        now = time.time()
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600,
               basis="seen in 2 sweeps; velocity 4x baseline")
        findings = claims_as_findings(store, now=now)
        assert len(findings) == 1
        f = findings[0]
        # the flat shape the pipeline's consumers read
        assert f["title"] == "gold trading: sustained movement in the field"
        assert "velocity 4x baseline" in f["snippet"]
        assert f["url"] == ""  # no exemplars recorded for this claim
        # the claim identity + basis the contracts ride on
        assert f["claim_id"] == "claim:gold trading"
        assert f["claim_key"] == "gold trading"
        assert f["claim_kind"] == "trend"
        assert f["standing"] == "fresh"
        assert f["confidence_basis"] == \
            "seen in 2 sweeps; velocity 4x baseline"
        assert f["source"] == "explore"

    def test_spike_and_unverified_feed_labelled(self):
        now = time.time()
        store = ExploreStore()
        _claim(store, key="burst topic", kind="spike", at=now - 300,
               text="burst topic: one huge sweep")
        _claim(store, key=" faint signal ", kind="unverified",
               at=now - 300, text="faint signal: seen once")
        findings = {f["claim_kind"]: f
                    for f in claims_as_findings(store, now=now)}
        assert findings["spike"]["title"].startswith("Spike: ")
        assert "spike" in findings["spike"]["snippet"]
        assert findings["unverified"]["title"].startswith("Unverified: ")
        assert "unverified" in findings["unverified"]["snippet"]

    def test_stale_claims_do_not_feed(self):
        now = time.time()
        store = ExploreStore()
        _claim(store, key="old trend", at=now - 4 * DAY)  # trend: 3d
        _claim(store, key="old spike", kind="spike",
               at=now - 13 * HOUR)  # spike: 12h
        _claim(store, key="live trend", at=now - 600)
        findings = claims_as_findings(store, now=now)
        assert [f["claim_key"] for f in findings] == ["live trend"]

    def test_pipeline_run_receives_the_feed(self, queue):
        now = time.time()
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600)
        set_findings_feed(make_claims_feed(store))
        try:
            result = weekly_content(
                "general", queue=queue,
                llm_fn=lambda prompt: CLEAN_COPY,
                platforms=("linkedin",), min_drafts=1, max_drafts=2)
        finally:
            set_findings_feed(None)
        fed = [f for f in result.findings if f.get("claim_key")]
        assert len(fed) == 1
        assert fed[0]["claim_key"] == "gold trading"
        assert result.drafts
        assert result.drafts[0].finding_title == fed[0]["title"]
        # the fan-out stamped the claim identity onto the queued
        # draft — the metadata contract 4 joins on
        stamped = [d for d in queue.pending()
                   if (d.metadata or {}).get("claim_key")]
        assert stamped
        assert stamped[0].metadata["claim_id"] == "claim:gold trading"
        assert stamped[0].metadata["claim_kind"] == "trend"

    def test_explicit_findings_bypass_the_feed(self, queue):
        now = time.time()
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600)
        set_findings_feed(make_claims_feed(store))
        try:
            result = weekly_content(
                "general", findings=[{
                    "title": "Injected finding",
                    "snippet": "caller-controlled evidence",
                    "url": "https://example.com/x"}],
                queue=queue, llm_fn=lambda prompt: CLEAN_COPY,
                platforms=("linkedin",), min_drafts=1, max_drafts=2)
        finally:
            set_findings_feed(None)
        assert [f["title"] for f in result.findings] == \
            ["Injected finding"]

    def test_feed_appends_after_research_never_displaces(
            self, queue, monkeypatch):
        now = time.time()
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600)
        research_finding = {
            "title": "Research result", "snippet": "from the run's"
            " own bounded research", "url": "https://example.com/r"}
        monkeypatch.setattr(
            cp_module, "_default_research",
            lambda niche, rctx, budget_usd=0.0: [research_finding])
        set_findings_feed(make_claims_feed(store))
        try:
            result = weekly_content(
                "general", queue=queue, rctx=object(),
                llm_fn=lambda prompt: CLEAN_COPY,
                platforms=("linkedin",), min_drafts=1, max_drafts=2)
        finally:
            set_findings_feed(None)
        # research first, feed appended after — the run's own
        # evidence is never displaced by exploration's
        assert result.findings[0]["title"] == "Research result"
        assert result.findings[1]["claim_key"] == "gold trading"

    def test_feed_dedupes_against_research(self, queue, monkeypatch):
        now = time.time()
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600,
               text="gold trading: sustained movement in the field")
        # research already found the very item the claim projects
        # (same title, same url — the claim's exemplar url is "")
        dupe = {"title": "gold trading: sustained movement in the"
                " field", "snippet": "same item via research",
                "url": ""}
        monkeypatch.setattr(
            cp_module, "_default_research",
            lambda niche, rctx, budget_usd=0.0: [dupe])
        set_findings_feed(make_claims_feed(store))
        try:
            result = weekly_content(
                "general", queue=queue, rctx=object(),
                llm_fn=lambda prompt: CLEAN_COPY,
                platforms=("linkedin",), min_drafts=1, max_drafts=2)
        finally:
            set_findings_feed(None)
        assert len(result.findings) == 1
        assert result.findings[0]["snippet"] == \
            "same item via research"


# ── (b) contract 2: the shared niche registry ───────────────────


class TestNicheRegistry:
    def _proposed_store(self, t0):
        from nomorals.media.explore import discover_niches

        store = _gold_cluster(ExploreStore(), t0)
        discover_niches(store, now=t0 + 5000)
        return store

    def test_proposed_invisible_adopted_visible(self, tmp_path):
        from nomorals.media.explore import adopt_niche

        t0 = time.time() - 100000
        store = self._proposed_store(t0)
        proposed = {r["slug"] for r in drafting_niches(store)}
        assert "gold_trading" not in proposed
        assert "finance_facts" in proposed  # the floor is context
        assert niche_drafting_context(store, "gold_trading") is None

        result = adopt_niche(store, "#1", dest_dir=tmp_path / "drafts",
                             now=t0 + 6000)
        assert result["ok"], result
        adopted = {r["slug"]: r for r in drafting_niches(store)}
        assert adopted["gold_trading"]["origin"] == "overlay"
        assert adopted["gold_trading"]["status"] == "adopted"
        ctx = niche_drafting_context(store, "gold_trading")
        assert ctx is not None
        assert ctx["thesis"] == store.niche_proposal(
            "gold_trading")["thesis"]
        # the free-text form resolves too (slugified match)
        assert niche_drafting_context(
            store, "Gold Trading")["slug"] == "gold_trading"

    def test_plugin_resolver_floor_overlay_default(self, tmp_path):
        from nomorals.media.contentops.niches import registry
        from nomorals.media.explore import adopt_niche

        t0 = time.time() - 100000
        store = self._proposed_store(t0)
        thesis = store.niche_proposal("gold_trading")["thesis"]
        resolve = drafting_plugin_resolver(store)
        # proposed: resolves through the shipped default, NEVER to
        # a plugin carrying the unadopted thesis
        assert getattr(resolve("gold_trading"), "thesis", None) \
            != thesis
        # floor: the code registry's own plugin
        assert resolve("finance_facts").thesis == \
            registry.get("finance_facts").thesis

        adopt_niche(store, "#1", dest_dir=tmp_path / "drafts",
                    now=t0 + 6000)
        plugin = resolve("gold_trading")
        assert plugin.name == "gold_trading"
        assert plugin.thesis == thesis

    def test_weekly_cell_carries_adopted_context(self, tmp_path):
        from nomorals.agents.content_cell import make_weekly_cell
        from nomorals.media.explore import adopt_niche, discover_niches
        from nomorals.storage.db import Database
        from nomorals.storage.kv import KVStore

        t0 = time.time() - 100000
        db = Database(":memory:")
        db.migrate()
        store = _gold_cluster(ExploreStore(KVStore(db)), t0)
        discover_niches(store, now=t0 + 5000)

        proposed_cell = make_weekly_cell(db, "gold_trading")
        assert proposed_cell is not None
        assert proposed_cell.niche_context is None

        adopt_niche(store, "#1", dest_dir=tmp_path / "drafts",
                    now=t0 + 6000)
        cell = make_weekly_cell(db, "gold_trading")
        assert cell.niche_context is not None
        assert cell.niche_context["origin"] == "overlay"
        thesis = store.niche_proposal("gold_trading")["thesis"]
        cell.begin()
        assert thesis in cell.contract.goal
        # a floor niche frames its run the same way
        floor_cell = make_weekly_cell(db, "finance_facts")
        assert floor_cell.niche_context["origin"] == "floor"


# ── (c) contract 3: artifact-cited media ────────────────────────


class TestArtifactCitations:
    def _artifact_store(self, tmp_path):
        from nomorals.storage.artifacts import ArtifactStore
        from nomorals.storage.blob import BlobStore
        from nomorals.storage.db import Database

        db = Database(":memory:")
        db.migrate()
        return ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))

    def _channel_claim(self, store, *, artifact_id, now):
        store.note_candidate(
            "channel", "Clip Channel", via="sweep:youtube_search",
            platform="youtube", beat="clips",
            evidence={"title": "Clip One",
                      "url": "https://youtu.be/c1",
                      "instrument": "youtube_search",
                      "artifact_id": artifact_id},
            sweep_id="sweep-1", now=now - 700)
        key = store.candidates()[0]["key"]
        _claim(store, key="clip channel rising", at=now - 600,
               subject=key, subject_kind="channel",
               backing={"candidate_key": key, "observations": []})
        return key

    def test_citation_flows_to_draft_metadata_and_resolves(
            self, tmp_path, queue):
        artifacts = self._artifact_store(tmp_path)
        blob = b"fake-video-bytes-for-a-clip"
        art = artifacts.put(blob, type="video", creator="test",
                            metadata={"title": "Clip One"})
        now = time.time()
        store = ExploreStore()
        self._channel_claim(store, artifact_id=art.id, now=now)

        findings = claims_as_findings(store, now=now, artifacts=artifacts)
        assert len(findings) == 1
        f = findings[0]
        assert f["artifact_id"] == art.id
        assert f["artifact_uri"] == f"artifact://{art.id}"
        assert f["url"] == "https://youtu.be/c1"
        assert f["exemplars"][0]["artifact_id"] == art.id

        result = weekly_content(
            "clips", findings=findings, queue=queue,
            llm_fn=lambda prompt: CLEAN_COPY,
            platforms=("linkedin",), min_drafts=1, max_drafts=2)
        assert result.drafts
        cited = [d for d in queue.pending()
                 if (d.metadata or {}).get("artifact_id")]
        assert cited, "the claim's draft must carry the citation"
        assert cited[0].metadata["artifact_uri"] == \
            f"artifact://{art.id}"
        # the citation resolves through the artifact store to the
        # same bytes — media by reference, never copied
        assert artifacts.read(cited[0].metadata["artifact_id"]) == blob

    def test_unverified_artifact_id_cites_nothing(self, tmp_path):
        artifacts = self._artifact_store(tmp_path)
        now = time.time()
        store = ExploreStore()
        self._channel_claim(store, artifact_id="no-such-artifact",
                            now=now)
        findings = claims_as_findings(store, now=now,
                                      artifacts=artifacts)
        assert len(findings) == 1
        assert "artifact_id" not in findings[0]
        assert "artifact_id" not in findings[0]["exemplars"][0]
        # and with no store injected, no citation is issued either
        findings = claims_as_findings(store, now=now)
        assert "artifact_id" not in findings[0]


# ── (d) contract 4: outcomes flow back to the claim ─────────────


class TestClaimOutcomes:
    NOW = time.time()

    def _manager(self, db, results):
        from nomorals.social.base import PlatformAdapter, PostResult
        from nomorals.social.manager import SocialManager

        class MetricsAdapter(PlatformAdapter):
            def __init__(self, name, served):
                self.name = name
                self.served = served

            def post(self, account, content, **kwargs):
                return PostResult(platform=self.name, ok=True,
                                  external_id="unused")

            def metrics(self, account, external_id):
                return dict(self.served.get(external_id, {}))

        ctx = SimpleNamespace(db=db, tools=None, extras={})
        return SocialManager(
            ctx, adapters={"mastodon": MetricsAdapter("mastodon",
                                                      results)},
            enforce_policy=False)

    def _db(self):
        from nomorals.storage.db import Database

        db = Database(":memory:")
        db.migrate()
        return db

    def _posted_draft(self, queue, content, *, claim_key=""):
        meta = {}
        if claim_key:
            meta = {"claim_key": claim_key,
                    "claim_id": f"claim:{claim_key}",
                    "claim_kind": "trend"}
        draft = queue.create_draft(content, platforms=["mastodon"],
                                   metadata=meta)
        queue.propose(draft.id)
        queue.approve(draft.id)
        queue.mark_posted(draft.id)
        return draft

    def _seed_post(self, manager, account, *, content,
                   external_id="ext-9", posted_at):
        from nomorals.core.ids import new_id

        row = manager.posts.create({
            "id": new_id(), "platform": "mastodon",
            "account_id": account.id, "content": content,
            "media_paths": [], "status": "posted",
            "external_id": external_id, "posted_at": posted_at,
            "created_at": posted_at, "metrics": {}})
        return row["id"]

    def _refresh(self, manager, queue, store, tmp_path, *, now):
        from nomorals.marketing.briefs import BriefStore
        from nomorals.social.metrics import refresh_post_metrics

        return refresh_post_metrics(
            manager, now=now, draft_queue=queue, explore_store=store,
            brief_store=BriefStore(str(tmp_path / "briefs.db")))

    def test_measured_post_records_outcome_against_claim(
            self, tmp_path, queue):
        from nomorals.social.content_pipeline import _engagement_of

        now = self.NOW
        db = self._db()
        manager = self._manager(db, {"ext-9": {"likes": 10,
                                               "replies": 5}})
        account = manager.connect("mastodon", "@me@example.social")
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600)
        content = "Gold room levels held all week — here is the map."
        self._posted_draft(queue, content, claim_key="gold trading")
        self._seed_post(manager, account, content=content,
                        posted_at=now - 2 * DAY)

        report = self._refresh(manager, queue, store, tmp_path,
                               now=now)
        assert report["refreshed"] == 1
        assert report["claims_fed"] == 1
        outcomes = store.claim_outcomes(claim_key="gold trading")
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome["post_ref"] == "ext-9"
        assert outcome["platform"] == "mastodon"
        assert outcome["handle"] == account.handle
        assert outcome["engagement"] == _engagement_of(
            {"likes": 10, "replies": 5})
        assert outcome["at"] == now

        # the next cycle re-measures but never records twice
        second = self._refresh(manager, queue, store, tmp_path,
                               now=now + 7 * HOUR)
        assert second["claims_already_recorded"] == 1
        assert len(store.claim_outcomes(claim_key="gold trading")) == 1

    def test_ambiguous_join_records_ambiguous_and_feeds_nothing(
            self, tmp_path, queue):
        now = self.NOW
        db = self._db()
        manager = self._manager(db, {"ext-9": {"likes": 3}})
        account = manager.connect("mastodon", "@me@example.social")
        store = ExploreStore()
        content = "Same words, two claims — no guessing."
        self._posted_draft(queue, content, claim_key="claim one")
        self._posted_draft(queue, content, claim_key="claim two")
        self._seed_post(manager, account, content=content,
                        posted_at=now - 2 * DAY)

        report = self._refresh(manager, queue, store, tmp_path,
                               now=now)
        assert report["claims_ambiguous"] == 1
        assert report["claims_fed"] == 0
        assert store.claim_outcomes() == []

    def test_unlinked_post_and_outcomeless_claim(self, tmp_path,
                                                 queue):
        now = self.NOW
        db = self._db()
        manager = self._manager(db, {"ext-9": {"likes": 3}})
        account = manager.connect("mastodon", "@me@example.social")
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600)
        content = "A post that descends from no claim at all."
        self._posted_draft(queue, content)  # no claim metadata
        self._seed_post(manager, account, content=content,
                        posted_at=now - 2 * DAY)

        report = self._refresh(manager, queue, store, tmp_path,
                               now=now)
        assert report["claims_unlinked"] == 1
        assert store.claim_outcomes(claim_key="gold trading") == []

    def test_young_post_is_measured_but_not_yet_fed(
            self, tmp_path, queue):
        now = self.NOW
        db = self._db()
        manager = self._manager(db, {"ext-9": {"likes": 3}})
        account = manager.connect("mastodon", "@me@example.social")
        store = ExploreStore()
        content = "Fresh post, numbers not yet a fair read."
        self._posted_draft(queue, content, claim_key="gold trading")
        self._seed_post(manager, account, content=content,
                        posted_at=now - 2 * HOUR)

        report = self._refresh(manager, queue, store, tmp_path,
                               now=now)
        assert report["refreshed"] == 1
        assert report["claims_fed"] == 0
        assert store.claim_outcomes() == []

    def test_queue_find_by_content_reads_posted_drafts(self, queue):
        draft = self._posted_draft(queue, "Exact words here.",
                                   claim_key="k")
        found = queue.find_by_content("Exact words here.",
                                      platform="mastodon")
        assert [d.id for d in found] == [draft.id]
        assert queue.find_by_content(
            "Exact words here.", platform="x") == []
        assert queue.find_by_content("other words") == []

    def test_outcomes_persist_and_dedupe_in_the_store(self, tmp_path):
        from nomorals.storage.db import Database
        from nomorals.storage.kv import KVStore

        db = Database(":memory:")
        db.migrate()
        kv = KVStore(db)
        store = ExploreStore(kv)
        record = {"claim_key": "gold trading",
                  "claim_id": "claim:gold trading",
                  "platform": "mastodon", "post_ref": "ext-1",
                  "handle": "@me", "engagement": 12.0,
                  "metrics": {"likes": 12}, "at": self.NOW}
        assert store.record_claim_outcome(record) is True
        assert store.record_claim_outcome(record) is False
        reloaded = ExploreStore(kv)
        rows = reloaded.claim_outcomes(claim_key="gold trading")
        assert len(rows) == 1
        assert rows[0]["engagement"] == 12.0


# ── (e) production wiring ───────────────────────────────────────


class TestProductionWiring:
    def test_build_context_installs_the_feed(self, tmp_path):
        from nomorals.agents.context import build_context
        from nomorals.core.config import load_settings
        from nomorals.media.explore.engine import explore_for_context

        settings = load_settings(overrides={
            "home": str(tmp_path),
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        queue = DraftQueue(db_path=str(tmp_path / "drafts.db"))
        try:
            assert cp_module._FINDINGS_FEED is not None
            store, _engine, _beats = explore_for_context(context)
            _claim(store, key="gold trading", at=time.time() - 300)
            # the pipeline's real entry point, no findings=
            # injection: the installed feed reaches the run
            result = weekly_content(
                "general", queue=queue,
                llm_fn=lambda prompt: CLEAN_COPY,
                platforms=("linkedin",), min_drafts=1, max_drafts=2)
            assert any(f.get("claim_key") == "gold trading"
                       for f in result.findings)
        finally:
            set_findings_feed(None)
            queue.close()
            context.close()
