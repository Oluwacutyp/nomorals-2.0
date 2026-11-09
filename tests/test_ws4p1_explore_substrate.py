"""WS4 Pass 1 — the exploration substrate.

The census (``~/workspace/ws4-video-census/CENSUS.md`` §3/§6): the
fetch half of the video system is finished; the exploration half
does not exist. P1 builds its foundation — an ``ExploreStore``
(append-only observations with provenance, candidate intake,
per-beat baselines + delivery log, KV persistence on the source
registry's pattern), a sweep engine over the EXISTING search organs
(manual trigger; P3 adds the scheduler), and the ``video_explore``
tool. Trend claims are P2: nothing in this suite — or the code —
calls one observation a trend.

All instruments are faked — no network. Pins:

(a) a sweep records observations with instrument/query/timestamp
    provenance per beat;
(b) beats seed from the NewsAgent interest profile (a profile topic
    becomes a beat; the seed mirror is pinned against the real
    tuple; nothing is hardcoded in the engine);
(c) instrument failure records a FAILED observation, distinct from
    an empty result;
(d) caps hold (beats per sweep, results per beat, new candidates);
(e) history is append-only — a second sweep never rewrites the
    first's rows, and state survives a store round-trip;
(f) the read surface returns what was recorded, provenance attached;
(+) candidate intake mirrors note_candidate (watching, unpromoted,
    shed re-intake window) and fires from the owner's real download
    path; the tool registers in the production registry.
"""

from __future__ import annotations

import json
from contextlib import suppress
from types import SimpleNamespace

# ── helpers ────────────────────────────────────────────────────────


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _row(rank, title, url, *, channel=None, source="youtube",
         view_count=None, duration_s=None):
    return {"rank": rank, "title": title, "url": url, "source": source,
            "channel": channel, "view_count": view_count,
            "duration_s": duration_s}


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


def _beats(*topics):
    return [{"topic": t, "weight": 1.0, "origin": "test"} for t in topics]


# ── (a) provenance per beat ────────────────────────────────────────


def test_sweep_records_observations_with_provenance():
    from nomorals.media.explore import ExploreStore, SweepEngine

    store = ExploreStore()
    fake = FakeInstrument("fake_web", {
        "gold trading": [
            _row(1, "Gold Levels Today", "https://youtu.be/aaa",
                 channel="Gold House", view_count=1200, duration_s=61),
            _row(2, "XAUUSD Morning Brief", "https://youtu.be/bbb",
                 channel="Gold House"),
        ],
        "ai models": [],
    })
    engine = SweepEngine(store, [fake])
    report = engine.run_sweep(_beats("gold trading", "ai models"),
                              trigger="test", now=1000.0)

    assert report["status"] == "complete"
    assert report["observations"] == 2
    assert report["results"] == 2
    assert report["empty"] == 1

    obs = store.observations(beat="gold trading")
    assert len(obs) == 1
    row = obs[0]
    assert row["instrument"] == "fake_web"
    assert row["query"] == "gold trading"
    assert row["at"] > 0
    assert row["status"] == "ok"
    assert row["result_count"] == 2
    first = row["results"][0]
    assert first["rank"] == 1 and first["channel"] == "Gold House"
    assert first["view_count"] == 1200 and first["duration_s"] == 61

    empty = store.observations(beat="ai models")
    assert empty[0]["status"] == "empty"
    assert empty[0]["result_count"] == 0

    sweep = store.latest_sweep()
    assert sweep["trigger"] == "test"
    assert sweep["beats"] == ["gold trading", "ai models"]
    assert sweep["results"] == 2 and sweep["empty"] == 1


# ── (b) beats seed from the NewsAgent profile ──────────────────────


def test_seed_mirror_matches_newsagent_tuple():
    from nomorals.agents.news import INTEREST_SEED_BEATS
    from nomorals.media.explore.beats import EXPLORE_SEED_BEATS

    assert EXPLORE_SEED_BEATS == INTEREST_SEED_BEATS


def test_beats_come_from_the_saved_profile_not_hardcoded(tmp_path):
    from nomorals.media.explore.beats import load_beats, load_news_profile

    _kvstore, db = _kv(tmp_path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS news_interest "
        "(key TEXT PRIMARY KEY, value TEXT)")
    db.execute(
        "INSERT OR REPLACE INTO news_interest (key, value) VALUES (?, ?)",
        ("profile", json.dumps({
            "beats": [{"topic": "amapiano dance challenges",
                       "weight": 0.7}],
            "pins": ["Lagos street food"],
        })))
    profile = load_news_profile(db)
    assert profile["pins"] == ["Lagos street food"]

    beats = load_beats(db, SimpleNamespace(briefing=None), watches=[])
    topics = {b["topic"]: b for b in beats}
    assert topics["amapiano dance challenges"]["origin"] == "profile"
    assert topics["amapiano dance challenges"]["weight"] == 0.7
    assert topics["Lagos street food"]["origin"] == "pin"
    # the shipped seeds are NOT the beat list once a profile is saved
    assert "XAUUSD and gold trading" not in topics


def test_beats_fall_back_to_seeds_and_fold_watches(tmp_path):
    from nomorals.media.explore.beats import EXPLORE_SEED_BEATS, load_beats

    beats = load_beats(None, SimpleNamespace(briefing=None),
                       watches=["short-form comedy skits"])
    topics = [b["topic"] for b in beats]
    for topic, _w in EXPLORE_SEED_BEATS:
        assert topic in topics
    watch = beats[-1]
    assert watch == {"topic": "short-form comedy skits", "weight": 1.0,
                     "origin": "watch"}


def test_watch_topics_read_from_interests_goals(tmp_path):
    from nomorals.media.explore.beats import collect_watch_topics

    _kvstore, db = _kv(tmp_path)
    # migrate() already created the real goals table; the class
    # column arrives by the tracker's defensive ALTER — same here.
    with suppress(Exception):
        db.execute(
            "ALTER TABLE goals ADD COLUMN goal_class TEXT NOT NULL "
            "DEFAULT 'general'")

    def _goal(gid, title, tags, gclass, status):
        db.execute(
            "INSERT INTO goals (goal_id, user_id, title, description, "
            "status, progress, priority, created_at, updated_at, tags, "
            "goal_class) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (gid, "owner", title, "", status, 0.0, 0, 1.0, 1.0,
             json.dumps(tags), gclass))

    _goal("g1", "Follow Nollywood trailers", ["watch:nollywood"],
          "interests", "active")
    _goal("g2", "Cooking channels", [], "interests", "active")
    _goal("g3", "Old watch", [], "interests", "completed")
    _goal("g4", "Money goal", [], "finance", "active")
    assert collect_watch_topics(db) == ["nollywood", "Cooking channels"]
    assert collect_watch_topics(None) == []


def test_real_newsagent_profile_drives_beats(tmp_path):
    """The injected-profile path: a REAL NewsAgent's profile (the
    L5 caller's seam) becomes the beat list verbatim."""
    from nomorals.agents.news import NewsAgent
    from nomorals.media.explore.beats import beats_from_profile

    _kvstore, db = _kv(tmp_path)
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(briefing=None), extras={},
        tools=None)
    agent = NewsAgent(ctx)
    agent.pin("afrobeats video premieres")
    profile = agent.interest_profile()
    beats = beats_from_profile(profile)
    topics = {b["topic"]: b for b in beats}
    assert topics["afrobeats video premieres"]["origin"] == "pin"
    assert topics["XAUUSD and gold trading"]["origin"] == "profile"


# ── (c) failure is not empty ───────────────────────────────────────


def test_instrument_failure_records_failed_not_empty():
    from nomorals.media.explore import ExploreStore, SweepEngine
    from nomorals.media.explore.instruments import InstrumentError

    store = ExploreStore()
    broken = FakeInstrument("broken", {
        "gold trading": InstrumentError("yt-dlp missing"),
    })
    empty = FakeInstrument("quiet", {"gold trading": []})
    engine = SweepEngine(store, [broken, empty])
    report = engine.run_sweep(_beats("gold trading"), now=1000.0)

    assert report["failed"] == 1 and report["empty"] == 1
    failed = store.observations(status="failed")
    assert len(failed) == 1
    assert failed[0]["instrument"] == "broken"
    assert "yt-dlp missing" in failed[0]["error"]
    assert failed[0]["result_count"] == 0
    # the baseline series keeps the failure OUT of the result counts
    base = store.beat_baselines()["gold trading"]
    assert base["failed"] == 1 and base["result_counts"] == [0]


def test_sweep_completes_when_every_instrument_fails():
    from nomorals.media.explore import ExploreStore, SweepEngine
    from nomorals.media.explore.instruments import InstrumentError

    store = ExploreStore()
    engine = SweepEngine(store, [FakeInstrument("x", {
        "beat one": InstrumentError("down"),
        "beat two": InstrumentError("down"),
    })])
    report = engine.run_sweep(_beats("beat one", "beat two"), now=5.0)
    assert report["status"] == "complete"
    assert report["failed"] == 2 and report["results"] == 0


# ── (d) caps hold ──────────────────────────────────────────────────


def test_caps_hold_for_beats_results_and_candidates():
    from nomorals.media.explore import ExploreStore, SweepEngine
    from nomorals.media.explore import engine as engine_mod

    store = ExploreStore()
    fake = FakeInstrument("fake", {
        f"beat {i}": [
            _row(r + 1, f"Video {i} Number {r}", f"https://youtu.be/v{i}-{r}",
                 channel=f"Chan {i}-{r}")
            for r in range(30)]
        for i in range(20)})
    engine = SweepEngine(store, [fake])
    report = engine.run_sweep(
        _beats(*[f"beat {i}" for i in range(20)]), now=1000.0)

    assert len(report["beats"]) == engine_mod.MAX_BEATS_PER_SWEEP
    assert report["beats_deferred"] == 20 - engine_mod.MAX_BEATS_PER_SWEEP
    for obs in store.observations(limit=100):
        assert obs["result_count"] <= engine_mod.MAX_RESULTS_PER_BEAT
    assert report["new_candidates"] <= engine_mod.MAX_CANDIDATES_PER_SWEEP
    assert report["candidates_deferred"] > 0
    assert report["exemplar_fetches"] == 0


# ── (e) append-only history + persistence ──────────────────────────


def test_history_is_append_only_across_sweeps(tmp_path):
    from nomorals.media.explore import ExploreStore, SweepEngine

    kv, _db = _kv(tmp_path)
    store = ExploreStore(kv)
    fake = FakeInstrument("fake", {
        "gold trading": [_row(1, "First Cut", "https://youtu.be/one")],
    })
    engine = SweepEngine(store, [fake])
    engine.run_sweep(_beats("gold trading"), now=1000.0)
    first_obs = store.observations()
    assert len(first_obs) == 1
    snapshot = json.loads(json.dumps(first_obs[0]))

    fake.script["gold trading"] = [
        _row(1, "Second Cut", "https://youtu.be/two")]
    engine.run_sweep(_beats("gold trading"), now=2000.0)

    obs = store.observations()
    assert len(obs) == 2
    # the first sweep's row is byte-identical after the second sweep
    assert obs[1] == snapshot
    assert obs[0]["results"][0]["title"] == "Second Cut"
    assert len(store.sweeps()) == 2
    base = store.beat_baselines()["gold trading"]
    assert base["sweep_count"] == 2 and base["result_counts"] == [1, 1]

    # and it all survives a fresh store over the same KV
    reloaded = ExploreStore(kv)
    assert len(reloaded.observations()) == 2
    assert len(reloaded.sweeps()) == 2
    assert reloaded.beat_baselines()["gold trading"]["sweep_count"] == 2


def test_failed_sweep_still_records(tmp_path):
    from nomorals.media.explore import ExploreStore

    kv, _db = _kv(tmp_path)
    store = ExploreStore(kv)
    sid = store.begin_sweep(trigger="test", beats=["x"], at=10.0)
    store.finish_sweep(sid, at=11.0, status="failed")
    assert store.latest_sweep()["status"] == "failed"
    assert ExploreStore(kv).latest_sweep()["status"] == "failed"


# ── (f) read surface ───────────────────────────────────────────────


def test_read_surface_returns_provenance():
    from nomorals.media.explore import ExploreStore, SweepEngine, explore_status

    store = ExploreStore()
    fake = FakeInstrument("fake", {
        "gold trading": [
            _row(1, "Gold Levels Today", "https://youtu.be/aaa",
                 channel="Gold House", view_count=42)],
    })
    SweepEngine(store, [fake]).run_sweep(_beats("gold trading"), now=100.0)

    data = explore_status(store)
    assert data["summary"]["sweeps"] == 1
    assert data["summary"]["observations"] == 1
    recent = data["recent_observations"]
    assert recent[0]["instrument"] == "fake"
    assert recent[0]["query"] == "gold trading"
    assert recent[0]["results"][0]["view_count"] == 42
    assert data["baselines"]["gold trading"]["observations"] == 1
    names = {c["name"] for c in data["candidates"]}
    assert "Gold House" in names


def test_delivery_log_records_per_beat():
    from nomorals.media.explore import ExploreStore, SweepEngine

    store = ExploreStore()
    fake = FakeInstrument("fake", {
        "gold trading": [_row(1, "A", "https://youtu.be/a"),
                         _row(2, "B", "https://youtu.be/b")],
        "ai models": [],
    })
    SweepEngine(store, [fake]).run_sweep(
        _beats("gold trading", "ai models"), now=100.0)
    deliveries = store.deliveries()
    by_beat = {d["beat"]: d["delivered"] for d in deliveries}
    assert by_beat == {"gold trading": 2, "ai models": 0}
    assert all(d["channel"] == "explore-store" for d in deliveries)


# ── candidate intake ───────────────────────────────────────────────


def test_candidate_intake_watching_unpromoted_and_shed_window(tmp_path):
    import time as _time

    from nomorals.media.explore import ExploreStore

    kv, _db = _kv(tmp_path)
    store = ExploreStore(kv)
    now = _time.time()
    assert store.note_candidate(
        "channel", "Gold House", via="sweep:fake", platform="youtube",
        beat="gold trading", sweep_id="sweep-1", now=now) is True
    cand = store.candidates()[0]
    assert cand["status"] == "watching"
    assert cand["sightings"] == 1

    # a repeat sighting bumps, never duplicates or promotes
    assert store.note_candidate(
        "channel", "Gold House", via="sweep:fake", platform="youtube",
        now=now + 10) is False
    cand = store.candidates()[0]
    assert cand["sightings"] == 2 and cand["status"] == "watching"

    # shed → the re-intake window refuses a flap straight back in
    assert store.shed_candidate(cand["key"], reason="noise", now=now + 20)
    assert store.candidates() == []
    assert store.note_candidate(
        "channel", "Gold House", via="sweep:fake", platform="youtube",
        now=now + 30) is False
    # after the window it may be noted again (fresh record)
    assert store.note_candidate(
        "channel", "Gold House", via="sweep:fake", platform="youtube",
        now=now + 8 * 86400.0) is True


def test_owner_download_path_notes_candidates(tmp_path):
    """The downloader's success seam is wired: a landed download
    notes its content into the SAME store a sweep reads."""
    from nomorals.media.downloader import DownloadReport, MediaDownloader
    from nomorals.media.explore import default_explore_store

    _kvstore, db = _kv(tmp_path)
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(home_path=tmp_path,
                                        workspace_dir=str(tmp_path)),
        extras={}, tools=None)
    downloader = MediaDownloader(ctx)
    report = DownloadReport(
        ok=True, url="https://youtu.be/xyz", title="Gold Levels Today",
        extractor="youtube")
    downloader._intake_exploration(report)

    store = default_explore_store()
    cands = store.candidates()
    assert cands, "the download seam noted nothing"
    # the report carries no channel, so the title lands as a TOPIC
    # candidate — never dressed up as a channel it didn't evidence
    assert all(c["kind"] == "topic" for c in cands)
    assert {c["name"] for c in cands} == {"Gold Levels Today"}
    assert {c["via"] for c in cands} == {"download"}


def test_note_owner_activity_channel_and_topics():
    from nomorals.media.explore import ExploreStore, note_owner_activity

    store = ExploreStore()
    added = note_owner_activity(
        store, title="Amapiano Mix Session", channel="DJ Maphorisa TV",
        url="https://youtu.be/zzz", via="download", now=100.0)
    assert added >= 1
    by_kind = {c["kind"]: c for c in store.candidates()}
    assert by_kind["channel"]["name"] == "DJ Maphorisa TV"
    assert by_kind["channel"]["via"] == "download"
    assert by_kind["channel"]["evidence"][0]["url"] == "https://youtu.be/zzz"


# ── the tool ───────────────────────────────────────────────────────


def _tool_registry(tmp_path, monkeypatch, store, instruments):
    from nomorals.media.explore import engine as engine_mod
    from nomorals.media.explore import tool as tool_mod
    from nomorals.tools.registry import ToolRegistry

    _kvstore, db = _kv(tmp_path)
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(
            home_path=tmp_path, workspace_dir=str(tmp_path),
            briefing=None),
        extras={}, tools=None)
    monkeypatch.setattr(
        engine_mod, "default_instruments", lambda context: instruments)
    registry = ToolRegistry(ctx)
    tool_mod.register(registry)
    return registry


def test_tool_run_and_status(tmp_path, monkeypatch):
    from nomorals.media.explore import ExploreStore

    store = ExploreStore()
    fake = FakeInstrument("fake", {
        "XAUUSD and gold trading": [
            _row(1, "Gold Levels Today", "https://youtu.be/aaa",
                 channel="Gold House")],
    })
    registry = _tool_registry(tmp_path, monkeypatch, store, [fake])
    # route the tool at our store via the default-store singleton
    from nomorals.media.explore import engine as engine_mod
    monkeypatch.setattr(engine_mod, "_DEFAULT_STORE", store)

    out = registry.call("video_explore", action="run")
    assert out.ok, out.error if not out.ok else ""
    payload = out.unwrap()
    assert payload["report"]["results"] == 1
    assert "sweep-1" not in payload["text"]  # no internal ids in text
    assert "Gold Levels" not in payload["text"]

    out = registry.call("video_explore", action="status")
    assert out.ok
    payload = out.unwrap()
    assert payload["summary"]["sweeps"] == 1
    assert "Gold House" in payload["text"]
    assert "obs-" not in payload["text"]

    out = registry.call("video_explore", action="candidates")
    assert out.unwrap()["count"] >= 1
    out = registry.call("video_explore", action="beats")
    assert out.unwrap()["count"] >= 1


def test_tool_registers_in_production_registry():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry().register_builtins()
    assert "video_explore" in registry.names()
    failed = {m["module"] for m in registry.failed_modules()}
    assert "nomorals.media.explore.tool" not in failed
    out = registry.call("video_explore", action="beats")
    assert out.ok
    topics = {b["topic"] for b in out.unwrap()["beats"]}
    assert "XAUUSD and gold trading" in topics


# ── cross-instrument corroboration ─────────────────────────────────


def test_corroboration_counts_shared_results():
    from nomorals.media.explore import ExploreStore, SweepEngine

    store = ExploreStore()
    shared = _row(1, "Gold Levels Today", "https://youtu.be/aaa")
    one = FakeInstrument("one", {"gold trading": [shared]})
    two = FakeInstrument("two", {"gold trading": [
        _row(1, "Gold Levels Today", "https://youtu.be/aaa"),
        _row(2, "Other", "https://youtu.be/other")]})
    engine = SweepEngine(store, [one, two])
    report = engine.run_sweep(_beats("gold trading"), now=1.0)
    assert report["beats"][0]["corroborated"] == 1
