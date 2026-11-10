"""WS4 Pass 6 — the trading beat + the interests read-side.

The census (``~/workspace/ws4-video-census/CENSUS.md`` §8 + §10
P6): GoldenSwing gains no video system; "both" resolves to
Devon-side obligations. (a) Exploration carries a TRADING BEAT —
XAUUSD/gold video topics as first-class seeds (data, not code
paths), visibly their own origin, riding the shipped sweep
machinery unchanged. (b) The WS2 interests provider gains the
read side over the exploration store (the write direction —
interests goals filing beats — shipped with WS4 P1; whichever
landed second did the wiring, and that is this pass). (c) A
trading-beat claim stays identifiable as one through the WS4 P5
findings feed, into a queued draft's metadata.

All instruments are faked — no network. Pins:

(a) the pack's exact contents; the fold rule (in while the beat
    base is the shipped seed profile — no db, no saved row, a
    pins-only saved row; OUT when a saved beat list or an
    injected profile is in force, wholesale like the news
    seeds); origin ``trading`` on every pack beat;
(b) a sweep over the resolved beats records trading-origin
    observations + candidates under the same caps as any beat
    (a trading beat past the cap defers like any other);
(c) the interests provider's real read path: a goal whose topic
    has a fresh trend claim shows it labelled kind + standing +
    source "exploration"; a spike-only topic reads "spike",
    never "trend"; a stale claim and an unrelated claim surface
    nothing; no findings → no exploration evidence at all (the
    percent stays the news store's measured number);
(d) claims_as_findings carries beat + beat_origin; the weekly
    fan-out stamps both onto the draft — and a non-pack claim
    carries its beat with NO origin (unlabelled, not guessed).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

DAY = 86400.0
HOUR = 3600.0
NOW = 1_800_000_000.0
SINCE = NOW - 7 * DAY

TRADING_TOPICS = [
    "XAUUSD price action analysis",
    "gold trading strategies and setups",
    "smart money concepts ICT gold trading",
    "central banks, interest rates and the gold price",
    "gold trading session recaps and weekly outlook",
]

NO_BRIEFING = SimpleNamespace(briefing=None)


# ── helpers ────────────────────────────────────────────────────


def _kv_db():
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    return db


def _save_profile_row(db, payload: dict) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS news_interest "
        "(key TEXT PRIMARY KEY, value TEXT)")
    db.execute(
        "INSERT OR REPLACE INTO news_interest (key, value) "
        "VALUES (?, ?)", ("profile", json.dumps(payload)))


def _row(rank, title, url, *, channel=None, source="youtube",
         view_count=None, duration_s=None):
    return {"rank": rank, "title": title, "url": url,
            "source": source, "channel": channel,
            "view_count": view_count, "duration_s": duration_s}


class FakeInstrument:
    """Answers from a script: {query: [rows]} or {query: Exception}."""

    def __init__(self, name, script):
        self.name = name
        self.script = script
        self.calls = []

    def search(self, query, *, limit=10):
        self.calls.append((query, limit))
        answer = self.script.get(query, [])
        if isinstance(answer, Exception):
            raise answer
        return [dict(r) for r in answer[:limit]]


def _claim(store, *, key, kind="trend", at, text=None,
           subject_kind="beat", freshness_s=None):
    """A claim row in the exact shape P2's evaluator authors."""
    store.record_claim({
        "claim_id": f"claim:{key}",
        "claim_key": key,
        "kind": kind,
        "subject": key,
        "subject_kind": subject_kind,
        "text": text or f"{key}: sustained movement in the field",
        "source": "explore-trends",
        "observed_at": at,
        "scope": key,
        "confidence_basis": "seen in 2 sweeps; classes video + web",
        "basis": {"sweeps": ["sweep-1", "sweep-2"],
                  "source_classes": ["video", "web"]},
        "freshness_s": freshness_s if freshness_s is not None else {
            "trend": 3 * DAY, "spike": 12 * HOUR,
            "unverified": DAY}[kind],
        "published_at": None,
        "contradicts": "",
        "entities": [],
        "backing": {"observations": [], "candidate_key": ""},
    })


# ── (a) the trading pack: contents + the fold rule ─────────────


class TestTradingPack:
    def test_pack_contents_pinned(self):
        from nomorals.media.explore.beats import TRADING_SEED_BEATS

        assert [t for t, _w in TRADING_SEED_BEATS] == TRADING_TOPICS
        assert all(0 < w <= 1.0 for _t, w in TRADING_SEED_BEATS)

    def test_folds_with_trading_origin_when_no_db(self):
        from nomorals.media.explore.beats import EXPLORE_SEED_BEATS, TRADING_SEED_BEATS, load_beats

        beats = load_beats(None, NO_BRIEFING, watches=[])
        by_topic = {b["topic"]: b for b in beats}
        for topic, weight in TRADING_SEED_BEATS:
            assert by_topic[topic]["origin"] == "trading"
            assert by_topic[topic]["weight"] == weight
        for topic, _w in EXPLORE_SEED_BEATS:
            assert topic in by_topic

    def test_folds_when_persisted_profile_is_the_seeds(self):
        from nomorals.media.explore.beats import load_beats

        beats = load_beats(_kv_db(), NO_BRIEFING, watches=[])
        by_topic = {b["topic"]: b for b in beats}
        for topic in TRADING_TOPICS:
            assert by_topic[topic]["origin"] == "trading"
        # the news seeds ride the profile path, as P1 pinned
        assert by_topic["XAUUSD and gold trading"]["origin"] == \
            "profile"

    def test_saved_profile_replaces_the_pack_wholesale(self):
        from nomorals.media.explore.beats import load_beats

        db = _kv_db()
        _save_profile_row(db, {
            "beats": [{"topic": "amapiano dance challenges",
                       "weight": 0.7}],
            "pins": ["Lagos street food"],
        })
        beats = load_beats(db, NO_BRIEFING, watches=[])
        topics = {b["topic"] for b in beats}
        assert "amapiano dance challenges" in topics
        # the pack does NOT leak into a profile that replaced the
        # shipped seeds — and neither do the news seeds (P1's pin)
        for topic in TRADING_TOPICS:
            assert topic not in topics
        assert "XAUUSD and gold trading" not in topics

    def test_pins_only_saved_row_keeps_the_pack(self):
        """A saved row that names no beats leaves the shipped
        seed beats in force — the pack, being part of that seed
        profile, stays too."""
        from nomorals.media.explore.beats import load_beats

        db = _kv_db()
        _save_profile_row(db, {"beats": [], "pins": ["owner pin"]})
        beats = load_beats(db, NO_BRIEFING, watches=[])
        by_topic = {b["topic"]: b for b in beats}
        assert by_topic["owner pin"]["origin"] == "pin"
        for topic in TRADING_TOPICS:
            assert by_topic[topic]["origin"] == "trading"

    def test_injected_profile_is_verbatim(self):
        from nomorals.media.explore.beats import load_beats

        beats = load_beats(
            None, NO_BRIEFING,
            profile={"beats": [{"topic": "his own field",
                                "weight": 0.6}], "pins": []},
            watches=[])
        topics = {b["topic"] for b in beats}
        assert topics == {"his own field"}

    def test_beat_origin_classifier(self):
        from nomorals.media.explore.beats import beat_origin

        assert beat_origin("XAUUSD price action analysis") == \
            "trading"
        assert beat_origin("xauusd price action analysis") == \
            "trading"
        assert beat_origin("AI and model work") == ""
        assert beat_origin("") == ""
        assert beat_origin(None) == ""


# ── (b) the pack rides the sweep machinery unchanged ───────────


class TestTradingSweep:
    def test_sweep_records_trading_origin_and_candidates(self):
        from nomorals.media.explore import ExploreStore, SweepEngine
        from nomorals.media.explore.beats import load_beats

        beats = load_beats(None, NO_BRIEFING, watches=[])
        assert len(beats) <= 12  # seeds + pack fit the per-sweep cap
        fake = FakeInstrument("fake_youtube", {
            "XAUUSD price action analysis": [
                _row(1, "XAUUSD Price Action — London Session",
                     "https://youtu.be/x1", channel="Gold Desk Live",
                     view_count=4200),
                _row(2, "Gold Levels Into New York",
                     "https://youtu.be/x2", channel="Gold Desk Live"),
            ],
        })
        store = ExploreStore()
        engine = SweepEngine(store, [fake])
        report = engine.run_sweep(beats, trigger="test", now=1000.0)

        assert report["status"] == "complete"
        assert report["beats_deferred"] == 0
        per_beat = {row["beat"]: row for row in report["beats"]}
        trading = per_beat["XAUUSD price action analysis"]
        assert trading["origin"] == "trading"
        assert trading["results"] == 2
        obs = store.observations(
            beat="XAUUSD price action analysis")
        assert obs and obs[0]["status"] == "ok"
        assert obs[0]["result_count"] == 2
        candidates = {c["name"]: c for c in store.candidates()}
        assert candidates["Gold Desk Live"]["kind"] == "channel"
        assert "XAUUSD price action analysis" in \
            candidates["Gold Desk Live"]["beats"]

    def test_a_trading_beat_past_the_cap_defers_like_any_beat(self):
        from nomorals.media.explore import ExploreStore, SweepEngine
        from nomorals.media.explore.engine import MAX_BEATS_PER_SWEEP

        filler = [{"topic": f"field {i}", "weight": 0.5,
                   "origin": "profile"}
                  for i in range(MAX_BEATS_PER_SWEEP)]
        beats = filler + [{"topic": TRADING_TOPICS[0],
                           "weight": 1.0, "origin": "trading"}]
        store = ExploreStore()
        engine = SweepEngine(store, [FakeInstrument("fake", {})])
        report = engine.run_sweep(beats, trigger="test", now=1000.0)
        assert report["beats_deferred"] == 1
        swept = {row["beat"] for row in report["beats"]}
        assert TRADING_TOPICS[0] not in swept


# ── the board names the trading field ──────────────────────────


def test_board_names_the_trading_field():
    from nomorals.media.explore import ExploreStore
    from nomorals.media.explore.board import render_explore_board

    context = SimpleNamespace(db=None, settings=NO_BRIEFING)
    text = render_explore_board(context, store=ExploreStore())
    assert "fields watched:" in text
    assert "trading" in text
    assert "trading field: XAUUSD price action analysis" in text


# ── (c) the interests read side ────────────────────────────────


@pytest.fixture()
def goals_env(monkeypatch, tmp_path):
    from nomorals.goals import GoalTracker, providers
    from nomorals.storage import Database

    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    settings = SimpleNamespace(
        home_path=home, briefing=SimpleNamespace(topics=[]))
    monkeypatch.setattr(providers, "get_settings", lambda: settings)
    db = Database(":memory:")
    tracker = GoalTracker(db)
    # The explore store persists through kv_store — present on the
    # production context db (migrations 0001 + 86); the P5 harness
    # deliberately leaves this db unmigrated (the news read's
    # hand-shaped tables), so the one table is stood up here with
    # its current shape.
    db.execute(
        "CREATE TABLE IF NOT EXISTS kv_store ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'json', "
        "updated_at REAL NOT NULL, expires_at REAL)")
    return tracker


def _goal(title, *, tags=(), created=SINCE):
    return SimpleNamespace(
        goal_id="g-test", title=title, tags=list(tags),
        created_at=created, goal_class="interests", progress=0.0)


def _explore_store_over(db):
    from nomorals.media.explore import ExploreStore
    from nomorals.storage.kv import KVStore

    store = ExploreStore()
    store.bind_store(KVStore(db))
    return store


class TestInterestsReadSide:
    def test_trend_claim_and_candidate_surface_labelled(
            self, goals_env):
        from nomorals.goals import providers

        tracker = goals_env
        goal = _goal("Quantum sensors")
        store = _explore_store_over(tracker.db)
        _claim(
            store, key="Quantum sensors", at=SINCE + 100,
            text="Quantum sensors: sustained search presence "
                 "across sweeps (a trend, not a one-off)")
        store.note_candidate(
            "channel", "Quantum Sensors Lab",
            via="sweep:youtube_search", platform="youtube",
            beat="quantum sensors",
            evidence={"title": "Quantum Sensors Lab — a video",
                      "url": "https://youtu.be/q1", "rank": 1,
                      "view_count": 900,
                      "instrument": "youtube_search",
                      "beat": "quantum sensors"},
            sweep_id="sweep-1", now=SINCE + 200)

        reading = providers.interests_progress(
            tracker.db, goal, now=SINCE + 1000)
        assert reading is not None
        # the percent is still the news store's measured number —
        # exploration evidence never inflates it
        assert reading.source == "news watch"
        assert reading.percent == 0.0
        exploration = reading.evidence["exploration"]
        assert exploration["source"] == "exploration"
        assert exploration["claims"][0]["kind"] == "trend"
        assert exploration["claims"][0]["standing"] == "fresh"
        assert exploration["candidates"][0]["name"] == \
            "Quantum Sensors Lab"
        assert exploration["candidates"][0]["status"] == "watching"
        assert "trend claim (fresh)" in reading.headline
        assert "Exploration:" in reading.headline

    def test_spike_only_topic_reads_spike_never_trend(
            self, goals_env):
        from nomorals.goals import providers

        tracker = goals_env
        goal = _goal("Nollywood trailers")
        store = _explore_store_over(tracker.db)
        _claim(
            store, key="Nollywood trailers", kind="spike",
            at=SINCE + 700,
            text="Nollywood trailers: spike — a burst of search "
                 "results in a single sweep, uncorroborated so far")

        reading = providers.interests_progress(
            tracker.db, goal, now=SINCE + 1000)
        assert reading is not None
        exploration = reading.evidence["exploration"]
        assert exploration["claims"][0]["kind"] == "spike"
        assert "spike claim (fresh)" in reading.headline
        assert "trend claim" not in reading.headline

    def test_no_findings_no_exploration_evidence(self, goals_env):
        from nomorals.goals import providers

        tracker = goals_env
        goal = _goal("Quantum sensors")
        store = _explore_store_over(tracker.db)
        # a claim on an unrelated field must not leak into the
        # reading, and an unwatched store adds nothing at all
        _claim(store, key="gold trading", at=SINCE + 100)

        reading = providers.interests_progress(
            tracker.db, goal, now=SINCE + 1000)
        assert reading is not None
        assert "exploration" not in reading.evidence
        assert "Exploration:" not in reading.headline

    def test_stale_claim_does_not_surface(self, goals_env):
        from nomorals.goals import providers

        tracker = goals_env
        goal = _goal("Quantum sensors")
        store = _explore_store_over(tracker.db)
        _claim(store, key="Quantum sensors", at=SINCE + 100)

        reading = providers.interests_progress(
            tracker.db, goal, now=SINCE + 4 * DAY)
        assert reading is not None
        assert "exploration" not in reading.evidence

    def test_candidate_only_headline(self, goals_env):
        from nomorals.goals import providers

        tracker = goals_env
        goal = _goal("Amapiano dance")
        store = _explore_store_over(tracker.db)
        store.note_candidate(
            "channel", "Amapiano Dance TV",
            via="sweep:youtube_search", platform="youtube",
            beat="amapiano dance challenges",
            evidence={"title": "Amapiano Dance TV — a video",
                      "url": "https://youtu.be/a1", "rank": 1,
                      "view_count": 300,
                      "instrument": "youtube_search",
                      "beat": "amapiano dance challenges"},
            sweep_id="sweep-1", now=SINCE + 300)

        reading = providers.interests_progress(
            tracker.db, goal, now=SINCE + 1000)
        assert reading is not None
        exploration = reading.evidence["exploration"]
        assert exploration["claims"] == []
        assert exploration["candidates"][0]["name"] == \
            "Amapiano Dance TV"
        assert "1 candidate(s) under watch" in reading.headline


# ── (d) beat origin survives the feed into draft metadata ──────

CLEAN_COPY = ("Shipping agents that finish real work — here is "
              "what changed this week and why it matters.")


@pytest.fixture()
def queue(tmp_path):
    from nomorals.social.drafts import DraftQueue

    q = DraftQueue(db_path=str(tmp_path / "drafts.db"))
    yield q
    q.close()


class TestBeatOriginThroughFeed:
    def test_trading_beat_claim_keeps_origin_into_draft_metadata(
            self, queue):
        from nomorals.media.explore import ExploreStore
        from nomorals.media.explore.feed import claims_as_findings, make_claims_feed
        from nomorals.social.content_pipeline import set_findings_feed, weekly_content

        now = time.time()
        store = ExploreStore()
        _claim(store, key="XAUUSD price action analysis",
               at=now - 600)

        findings = claims_as_findings(store, now=now)
        assert len(findings) == 1
        assert findings[0]["beat"] == \
            "XAUUSD price action analysis"
        assert findings[0]["beat_origin"] == "trading"

        set_findings_feed(make_claims_feed(store))
        try:
            weekly_content(
                "general", queue=queue,
                llm_fn=lambda prompt: CLEAN_COPY,
                platforms=("linkedin",), min_drafts=1, max_drafts=2)
        finally:
            set_findings_feed(None)
        stamped = [d for d in queue.pending()
                   if (d.metadata or {}).get("claim_key")]
        assert stamped
        assert stamped[0].metadata["beat"] == \
            "XAUUSD price action analysis"
        assert stamped[0].metadata["beat_origin"] == "trading"

    def test_non_pack_claim_carries_beat_with_no_origin(
            self, queue):
        from nomorals.media.explore import ExploreStore
        from nomorals.media.explore.feed import claims_as_findings, make_claims_feed
        from nomorals.social.content_pipeline import set_findings_feed, weekly_content

        now = time.time()
        store = ExploreStore()
        _claim(store, key="gold trading", at=now - 600)
        findings = claims_as_findings(store, now=now)
        assert findings[0]["beat"] == "gold trading"
        assert findings[0]["beat_origin"] == ""

        set_findings_feed(make_claims_feed(store))
        try:
            weekly_content(
                "general", queue=queue,
                llm_fn=lambda prompt: CLEAN_COPY,
                platforms=("linkedin",), min_drafts=1, max_drafts=2)
        finally:
            set_findings_feed(None)
        stamped = [d for d in queue.pending()
                   if (d.metadata or {}).get("claim_key")]
        assert stamped
        assert stamped[0].metadata["beat"] == "gold trading"
        assert "beat_origin" not in stamped[0].metadata

    def test_candidate_claim_under_a_trading_beat(self):
        from nomorals.media.explore import ExploreStore
        from nomorals.media.explore.feed import claims_as_findings

        now = time.time()
        store = ExploreStore()
        store.note_candidate(
            "topic", "xauusd level breakdown",
            via="sweep:youtube_search",
            beat="XAUUSD price action analysis",
            evidence={"title": "XAUUSD Level Breakdown — a video",
                      "url": "https://youtu.be/l1", "rank": 1,
                      "view_count": 500,
                      "instrument": "youtube_search",
                      "beat": "XAUUSD price action analysis"},
            sweep_id="sweep-1", now=now - 500)
        cand_key = store.candidates()[0]["key"]
        store.record_claim({
            "claim_id": "claim:explore-candidate-xauusd",
            "claim_key": "explore candidate xauusd",
            "kind": "trend",
            "subject": cand_key,
            "subject_kind": "topic",
            "text": "xauusd level breakdown (topic): recurring "
                    "across sweeps — a trend in the field",
            "source": "explore-trends",
            "observed_at": now - 400,
            "scope": "XAUUSD price action analysis",
            "confidence_basis": "seen in 2 sweeps",
            "basis": {"sweeps": ["sweep-1", "sweep-2"],
                      "source_classes": ["video", "web"]},
            "freshness_s": 3 * DAY,
            "published_at": None,
            "contradicts": "",
            "entities": ["xauusd level breakdown"],
            "backing": {"observations": [],
                        "candidate_key": cand_key},
        })
        findings = claims_as_findings(store, now=now)
        assert len(findings) == 1
        assert findings[0]["beats"] == \
            ["XAUUSD price action analysis"]
        assert findings[0]["beat_origin"] == "trading"
