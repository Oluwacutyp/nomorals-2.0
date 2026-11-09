"""WS4 Pass 2 — trend identification over the exploration substrate.

The census (``~/workspace/ws4-video-census/CENSUS.md`` §3.2/§6):
one sweep = an observation; a TREND claim requires ≥2 observations
across ≥2 sweeps OR cross-source-class corroboration in one sweep;
anything less is labelled spike or unverified; claims carry a
mandatory confidence_basis naming their legs; freshness decays and
a stale claim reads stale. Candidate verdicts mirror the source
registry's evidence thresholds (promote on evidence, shed after 3
consecutive missed sweeps). All state is hand-built in the store —
no network, no instruments run.

Pins:

(a) one sweep's burst is a spike / plain sighting unverified —
    never a trend, even at extreme velocity;
(b) the same signal across two sweeps (or cross-class
    corroboration in one sweep) earns a trend whose
    confidence_basis names the sweeps + instruments;
(c) velocity vs the beat's own baseline is named in the basis;
(d) a claim read back after its freshness window reads stale;
(e) evaluator re-run is idempotent — same claim set, no new rows;
(f) candidate standing strengthens with corroborated sightings,
    promotes on the evidence bar, decays toward shed across
    missed sweeps, and sheds at the threshold;
(g) the tool's trends read renders kind/basis/freshness with no
    internal ids in the text.
"""

from __future__ import annotations

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


def _rows(n, tag):
    return [_row(i + 1, f"{tag} video {i + 1}",
                 f"https://youtu.be/{tag}{i}")
            for i in range(n)]


def _sweep(store, beats, at, dur=10.0):
    sweep_id = store.begin_sweep(trigger="test", beats=beats, at=at)
    store.finish_sweep(sweep_id, at=at + dur)
    return sweep_id


def _obs(store, sweep_id, beat, instrument, rows, at, status=None):
    status = status or ("ok" if rows else "empty")
    return store.record_observation({
        "sweep_id": sweep_id, "beat": beat, "instrument": instrument,
        "query": beat, "at": at, "status": status,
        "error": "instrument down" if status == "failed" else "",
        "result_count": len(rows), "results": rows})


def _claim_for(claims, subject):
    for claim in claims:
        if claim.get("subject") == subject:
            return claim
    return None


# ── (a) one sweep is never a trend ─────────────────────────────────


def test_single_sweep_burst_is_a_spike_never_a_trend():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["retro gaming"], at=100.0)
    _obs(store, s1, "retro gaming", "web_search", [], at=101.0)
    s2 = _sweep(store, ["retro gaming"], at=200.0)
    _obs(store, s2, "retro gaming", "web_search",
         _rows(6, "retro"), at=201.0)

    report = evaluate_trends(store, now=300.0)
    claim = _claim_for(report["claims_current"], "retro gaming")
    assert claim is not None
    assert claim["kind"] == "spike"
    assert claim["basis"]["velocity"]["from_zero"] is True
    # single class only: the basis must carry the seed-signal flag
    assert claim["basis"]["single_source"] is True
    assert "single-source" in claim["confidence_basis"]


def test_single_sweep_plain_sighting_is_unverified():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["home baking"], at=100.0)
    _obs(store, s1, "home baking", "youtube_search",
         _rows(2, "bake"), at=101.0)

    report = evaluate_trends(store, now=200.0)
    claim = _claim_for(report["claims_current"], "home baking")
    assert claim is not None
    assert claim["kind"] == "unverified"
    assert claim["freshness_s"] == 86400.0


def test_velocity_alone_never_upgrades_past_the_bar():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["quiet games"], at=100.0)
    _obs(store, s1, "quiet games", "web_search", [], at=101.0)
    s2 = _sweep(store, ["quiet games"], at=200.0)
    _obs(store, s2, "quiet games", "web_search",
         _rows(20, "loud"), at=201.0)

    report = evaluate_trends(store, now=300.0)
    claim = _claim_for(report["claims_current"], "quiet games")
    assert claim["kind"] == "spike"  # 20x from zero — still one sweep


# ── (b) the trend bar ──────────────────────────────────────────────


def test_two_sweeps_make_a_trend_with_named_basis():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["gold trading"], at=1000.0)
    _obs(store, s1, "gold trading", "youtube_search",
         _rows(3, "g1"), at=1001.0)
    _obs(store, s1, "gold trading", "video_finder",
         _rows(2, "g2"), at=1002.0)
    s2 = _sweep(store, ["gold trading"], at=2000.0)
    _obs(store, s2, "gold trading", "youtube_search",
         _rows(4, "g3"), at=2001.0)
    _obs(store, s2, "gold trading", "web_search",
         _rows(2, "g4"), at=2002.0)

    report = evaluate_trends(store, now=2100.0)
    claim = _claim_for(report["claims_current"], "gold trading")
    assert claim is not None
    assert claim["kind"] == "trend"
    assert claim["basis"]["sweeps"] == [s1, s2]
    assert set(claim["basis"]["source_classes"]) == {"video", "web"}
    assert claim["basis"]["single_source"] is False
    # the mandatory basis names the evidence, in words
    assert s1 in claim["confidence_basis"]
    assert s2 in claim["confidence_basis"]
    assert "youtube_search" in claim["confidence_basis"]
    # backing observation ids ride the claim (census §3.4)
    assert len(claim["backing"]["observations"]) == 4
    assert claim["freshness_s"] == 3 * 86400.0


def test_cross_class_corroboration_in_one_sweep_makes_a_trend():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    shared = "https://youtu.be/shared"
    s1 = _sweep(store, ["gold trading"], at=1000.0)
    _obs(store, s1, "gold trading", "video_finder",
         [_row(1, "Levels", shared)], at=1001.0)
    _obs(store, s1, "gold trading", "web_search",
         [_row(1, "Levels", shared, source="web")], at=1002.0)

    report = evaluate_trends(store, now=1100.0)
    claim = _claim_for(report["claims_current"], "gold trading")
    assert claim["kind"] == "trend"
    assert claim["basis"]["corroborated_urls"] == 1
    assert "corroborated" in claim["confidence_basis"]


def test_same_class_agreement_is_not_corroboration():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    shared = "https://youtu.be/shared"
    s1 = _sweep(store, ["gold trading"], at=1000.0)
    # two VIDEO-class organs surfacing the same URL in one sweep:
    # one class agreeing with itself is not a second leg
    _obs(store, s1, "gold trading", "video_finder",
         [_row(1, "Levels", shared)], at=1001.0)
    _obs(store, s1, "gold trading", "youtube_search",
         [_row(1, "Levels", shared)], at=1002.0)

    report = evaluate_trends(store, now=1100.0)
    claim = _claim_for(report["claims_current"], "gold trading")
    assert claim["kind"] == "unverified"
    assert claim["basis"]["corroborated_urls"] == 0
    assert claim["basis"]["single_source"] is True


# ── (c) velocity ───────────────────────────────────────────────────


def test_velocity_rise_is_named_in_the_basis():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    for i, count in enumerate((2, 2, 9)):
        sid = _sweep(store, ["ai models"], at=1000.0 * (i + 1))
        _obs(store, sid, "ai models", "web_search",
             _rows(count, f"ai{i}"), at=1000.0 * (i + 1) + 1)

    report = evaluate_trends(store, now=3100.0)
    claim = _claim_for(report["claims_current"], "ai models")
    assert claim["kind"] == "trend"
    velocity = claim["basis"]["velocity"]
    assert velocity["latest"] == 9
    assert velocity["baseline_mean"] == 2.0
    assert velocity["ratio"] == 4.5
    assert velocity["rising"] is True
    assert "velocity 4.5x" in claim["confidence_basis"]
    assert "rising" in claim["text"]


def test_failed_observations_never_enter_the_velocity_series():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["gold trading"], at=1000.0)
    _obs(store, s1, "gold trading", "web_search",
         _rows(4, "a"), at=1001.0)
    s2 = _sweep(store, ["gold trading"], at=2000.0)
    _obs(store, s2, "gold trading", "web_search", [], at=2001.0,
         status="failed")
    s3 = _sweep(store, ["gold trading"], at=3000.0)
    _obs(store, s3, "gold trading", "web_search",
         _rows(4, "b"), at=3001.0)

    report = evaluate_trends(store, now=3100.0)
    claim = _claim_for(report["claims_current"], "gold trading")
    assert claim["kind"] == "trend"
    # series is [4, 4] — the failed sweep is not a zero
    assert claim["basis"]["velocity"]["prior_sweeps"] == 1
    assert claim["basis"]["velocity"]["baseline_mean"] == 4.0


# ── (d) freshness / decay ──────────────────────────────────────────


def _trend_store(base=10000.0):
    from nomorals.media.explore import ExploreStore

    store = ExploreStore()
    s1 = _sweep(store, ["gold trading"], at=base)
    _obs(store, s1, "gold trading", "youtube_search",
         _rows(3, "x"), at=base + 1)
    s2 = _sweep(store, ["gold trading"], at=base + 10000.0)
    _obs(store, s2, "gold trading", "youtube_search",
         _rows(3, "y"), at=base + 10001.0)
    return store


def test_claim_reads_stale_after_its_freshness_window():
    from nomorals.media.explore import (
        claim_standing, current_claims, evaluate_trends)

    store = _trend_store()
    evaluate_trends(store, now=20100.0)
    newest_evidence = 20001.0  # base 10000 + 10001 (see _trend_store)

    fresh = _claim_for(current_claims(store, now=newest_evidence
                                      + 2 * 86400),
                       "gold trading")
    assert fresh["standing"] == "fresh"
    stale = _claim_for(current_claims(store, now=newest_evidence
                                      + 4 * 86400),
                       "gold trading")
    assert stale["standing"] == "stale"
    # standing is computed, never stored: the raw row has none
    raw = store.claims()[0]
    assert "standing" not in raw
    assert claim_standing(raw, now=0.0) == "fresh"  # no clock, no stale


# ── (e) idempotency ────────────────────────────────────────────────


def test_evaluation_is_idempotent():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = _trend_store()
    first = evaluate_trends(store, now=20100.0)
    assert first["claims_written"] >= 1
    rows_after_first = len(store.claims(limit=10000))

    second = evaluate_trends(store, now=20500.0)
    assert second["claims_written"] == 0
    assert len(store.claims(limit=10000)) == rows_after_first
    assert [c["claim_id"] for c in second["claims_current"]] == \
        [c["claim_id"] for c in first["claims_current"]]
    assert [c["kind"] for c in second["claims_current"]] == \
        [c["kind"] for c in first["claims_current"]]


def test_new_evidence_supersedes_the_old_claim_row():
    from nomorals.media.explore import evaluate_trends

    store = _trend_store()
    evaluate_trends(store, now=20100.0)
    s3 = _sweep(store, ["gold trading"], at=30000.0)
    _obs(store, s3, "gold trading", "web_search",
         _rows(5, "z"), at=30001.0)
    report = evaluate_trends(store, now=30100.0)
    assert report["claims_written"] == 1
    claim = _claim_for(report["claims_current"], "gold trading")
    assert claim["observed_at"] == 30001.0
    assert set(claim["basis"]["source_classes"]) == {"video", "web"}
    # history is kept; the fold shows only the newest row
    assert len(store.claims(limit=10000)) == 2
    assert len(report["claims_current"]) == 1


# ── declined: no manufactured certainty ────────────────────────────


def test_declined_subjects_are_named_with_reasons():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    for at in (100.0, 200.0):
        sid = _sweep(store, ["silent topic", "broken topic"], at=at)
        _obs(store, sid, "silent topic", "web_search", [], at=at + 1)
        _obs(store, sid, "broken topic", "web_search", [], at=at + 1,
             status="failed")

    report = evaluate_trends(store, now=300.0)
    assert "error" not in report
    declined = {row["subject"]: row["reason"]
                for row in report["declined"]}
    assert "no results" in declined["silent topic"]
    assert "every instrument failed" in declined["broken topic"]
    assert _claim_for(report["claims_current"], "silent topic") is None
    assert _claim_for(report["claims_current"], "broken topic") is None


# ── (f) candidate verdicts ─────────────────────────────────────────


def _note(store, kind, name, *, instrument, url, sweep_id, beat, now,
          platform="youtube"):
    return store.note_candidate(
        kind, name, via=f"sweep:{instrument}", platform=platform,
        beat=beat, sweep_id=sweep_id, now=now,
        evidence={"title": "Levels", "url": url,
                  "instrument": instrument, "beat": beat})


def test_candidate_promotes_on_cross_class_corroborated_sighting():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["gold trading"], at=100.0)
    _note(store, "channel", "Gold House", instrument="youtube_search",
          url="https://youtu.be/aaa", sweep_id=s1, beat="gold trading",
          now=101.0)
    _note(store, "channel", "Gold House", instrument="web_search",
          url="https://youtu.be/aaa", sweep_id=s1, beat="gold trading",
          now=102.0)

    report = evaluate_trends(store, now=200.0)
    key = ExploreStore.candidate_key("channel", "Gold House", "youtube")
    candidate = store.candidate(key)
    assert candidate["status"] == "promoted"
    standing = candidate["standing"]
    assert standing["corroborated_sightings"] == 1
    assert standing["sightings"] == 2
    assert standing["score"] == 4.0  # 2 sightings + 2×1 corroborated
    assert key in report["candidates"]["promoted"]
    claim = _claim_for(report["claims_current"], key)
    assert claim["kind"] == "trend"
    assert claim["basis"]["single_source"] is False


def test_candidate_promotes_across_two_sweeps_and_absence_resets():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    s1 = _sweep(store, ["gold trading"], at=100.0)
    _note(store, "channel", "Silver Desk", instrument="youtube_search",
          url="https://youtu.be/s1", sweep_id=s1, beat="gold trading",
          now=101.0)
    key = ExploreStore.candidate_key("channel", "Silver Desk", "youtube")

    # two sweeps cover its beat without sighting it: decay, no shed
    _sweep(store, ["gold trading"], at=200.0)
    _sweep(store, ["gold trading"], at=300.0)
    evaluate_trends(store, now=400.0)
    candidate = store.candidate(key)
    assert candidate["status"] == "watching"
    assert candidate["standing"]["missed_sweeps"] == 2
    assert candidate["standing"]["score"] < candidate["sightings"]

    # a new sighting in a second sweep resets the absence streak
    # and clears the promotion bar (2 distinct sweeps)
    s4 = _sweep(store, ["gold trading"], at=500.0)
    _note(store, "channel", "Silver Desk", instrument="youtube_search",
          url="https://youtu.be/s2", sweep_id=s4, beat="gold trading",
          now=501.0)
    evaluate_trends(store, now=600.0)
    candidate = store.candidate(key)
    assert candidate["status"] == "promoted"
    assert candidate["standing"]["missed_sweeps"] == 0
    assert candidate["standing"]["sweeps_seen"] == 2


def test_candidate_sheds_after_three_missed_sweeps():
    from nomorals.media.explore import (
        ExploreStore, current_claims, evaluate_trends)

    store = ExploreStore()
    s1 = _sweep(store, ["gold trading"], at=100.0)
    _note(store, "topic", "London Open Levels",
          instrument="youtube_search", url="https://youtu.be/l1",
          sweep_id=s1, beat="gold trading", now=101.0)
    _note(store, "topic", "Sunday Roast Timing",
          instrument="youtube_search", url="https://youtu.be/r1",
          sweep_id=s1, beat="cooking", now=101.0)
    gold_key = ExploreStore.candidate_key("topic", "London Open Levels")
    cook_key = ExploreStore.candidate_key("topic", "Sunday Roast Timing")

    evaluate_trends(store, now=150.0)
    assert _claim_for(current_claims(store), gold_key) is not None

    for at in (200.0, 300.0, 400.0):
        _sweep(store, ["gold trading"], at=at)
    report = evaluate_trends(store, now=500.0)

    # absent from 3 consecutive sweeps covering its beat → shed,
    # and its claim stops being current the moment it leaves
    assert store.candidate(gold_key) is None
    assert gold_key in report["candidates"]["shed"]
    assert _claim_for(report["claims_current"], gold_key) is None
    # the re-intake window holds (registry discipline)
    assert store.note_candidate(
        "topic", "London Open Levels", via="sweep:youtube_search",
        now=600.0) is False
    # a candidate whose beat was never swept is NOT shed —
    # Devon not looking is not the candidate vanishing
    assert store.candidate(cook_key) is not None


def test_owner_gravity_is_labelled_never_dressed_as_field_evidence():
    from nomorals.media.explore import ExploreStore, evaluate_trends

    store = ExploreStore()
    store.note_candidate(
        "channel", "DJ Maphorisa TV", via="download",
        evidence={"title": "Mix", "url": "https://youtu.be/zzz",
                  "instrument": "download", "beat": ""}, now=100.0)
    report = evaluate_trends(store, now=200.0)
    key = ExploreStore.candidate_key("channel", "DJ Maphorisa TV")
    candidate = store.candidate(key)
    # one owner download: watched, unpromoted, honestly labelled
    assert candidate["status"] == "watching"
    claim = _claim_for(report["claims_current"], key)
    assert claim["kind"] == "unverified"
    assert claim["basis"]["owner_gravity"] is True
    assert "owner's own" in claim["confidence_basis"]


# ── (g) the tool's trends read ─────────────────────────────────────


def _tool_registry(tmp_path, monkeypatch, store):
    from nomorals.media.explore import engine as engine_mod
    from nomorals.media.explore import tool as tool_mod
    from nomorals.tools.registry import ToolRegistry

    _kvstore, db = _kv(tmp_path)
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(
            home_path=tmp_path, workspace_dir=str(tmp_path),
            briefing=None),
        extras={}, tools=None)
    monkeypatch.setattr(engine_mod, "_DEFAULT_STORE", store)
    registry = ToolRegistry(ctx)
    tool_mod.register(registry)
    return registry


def test_tool_trends_renders_claims_without_internal_ids(
        tmp_path, monkeypatch):
    import time

    from nomorals.media.explore import ExploreStore

    # the tool evaluates against the wall clock, so the evidence
    # is dated recently (inside the trend freshness window)
    store = _trend_store(base=time.time() - 20000.0)
    registry = _tool_registry(tmp_path, monkeypatch, store)

    out = registry.call("video_explore", action="trends")
    assert out.ok, out.error if not out.ok else ""
    payload = out.unwrap()
    assert payload["count"] == 1
    claim = payload["claims"][0]
    assert claim["kind"] == "trend"
    assert claim["standing"] == "fresh"
    text = payload["text"]
    assert "[trend]" in text
    assert "gold trading" in text
    # no internal ids in rendered text — ids ride the structured
    # result only (house rule; P1's pin, extended to claims)
    assert "sweep-" not in text
    assert "obs-" not in text
    assert "claim:" not in text
    assert claim["claim_id"] in str(payload["claims"])

    # refresh=false renders the stored set without re-evaluating
    out = registry.call("video_explore", action="trends", refresh=False)
    assert out.ok
    assert out.unwrap()["evaluation"] is None
    assert out.unwrap()["count"] == 1
