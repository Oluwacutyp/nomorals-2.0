"""WS4 Pass 4 — niche discovery + the shared niche overlay.

Pins, end to end:

(a) Corroborated field evidence produces a niche proposal with
    its evidence pack (members, backing claims, exemplars, score
    WITH inputs); thin evidence produces none + a declined reason.
(b) A proposed niche changes NOTHING in the effective niche set
    or the code registry — proposal is not adoption.
(c) Adoption (the owner's act) flips the overlay record AND
    scaffold-generates a draft plugin in the plugins' shape at
    the drafts location; the effective set gains the niche via
    the OVERLAY — the draft file never silently loads, the code
    registry never changes, and the generic materialisation
    serves the adopted record.
(d) Dismissal sticks: the same evidence does not re-propose;
    genuinely new evidence (a fresh sighting) does.
(e) The 7 floor plugins are in the effective set before and
    after every overlay operation.
(f) The board/tool surfaces show proposals AS proposals, and
    the adopt/dismiss verbs work through the owner-only
    /explore surface with the same resolution the board prints.
(g) A discovery failure inside a sweep cycle is recorded on
    the cycle and the sweep still completes; a healthy cycle
    ends with discovery counts.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from nomorals.media.explore import (
    NICHE_ADOPTED,
    NICHE_DISMISSED,
    NICHE_PROPOSED,
    ExploreStore,
    adopt_niche,
    discover_niches,
    dismiss_niche,
    effective_niches,
    floor_niches,
    overlay_plugin,
)
from nomorals.media.explore import engine as engine_mod
from nomorals.media.explore import niches as niches_mod
from nomorals.media.explore.board import (
    render_explore_board,
    render_explore_niches,
)
from nomorals.media.explore.jobs import run_sweep_cycle

T0 = 1_700_000_000.0
FLOOR_SLUGS = {
    "anime_edits", "did_you_know", "finance_facts",
    "horror_stories", "motivation", "reddit_stories",
    "sports_edits",
}


# ── seeding helpers (the shipped seams, no back doors) ──────────


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


def _promote(store, key, score=9):
    return store.update_candidate_verdict(
        key, status="promoted",
        standing={"sightings": 4, "score": score})


def _gold_cluster(store, *, promote=True):
    """A corroborated cluster: channel + topic under one beat,
    sharing result URLs, sighted across two sweeps."""
    for i, sweep in enumerate(["sweep-1", "sweep-2"]):
        t = T0 + i * 3600
        for url in ("https://youtu.be/g1", "https://youtu.be/g2"):
            _note(store, "channel", "NG Gold Room",
                  beat="gold trading", sweep_id=sweep, url=url,
                  now=t)
            _note(store, "topic", "xauusd level breakdown",
                  beat="gold trading", sweep_id=sweep, url=url,
                  now=t)
    if promote:
        for cand in store.candidates():
            _promote(store, cand["key"])
    return store


def _trend_claim(store, *, subject="gold trading", scope=None,
                 at=T0 + 4000):
    store.record_claim({
        "claim_id": f"claim:{subject}", "claim_key": subject,
        "kind": "trend", "subject": subject,
        "subject_kind": "beat", "text": f"{subject}: sustained",
        "source": "explore-trends", "observed_at": at,
        "scope": scope if scope is not None else subject,
        "confidence_basis": "2 sweeps; classes video + web",
        "basis": {
            "sweeps": ["sweep-1", "sweep-2"],
            "instruments": ["youtube_search", "web_search"],
            "source_classes": ["video", "web"],
            "corroborated_urls": 1,
            "velocity": {"latest": 8, "baseline_mean": 2.0,
                         "prior_sweeps": 1, "ratio": 4.0,
                         "from_zero": False, "rising": True},
            "single_source": False},
        "freshness_s": 3 * 86400.0,
        "backing": {"observations": ["obs-1"], "candidate_key": ""},
        "entities": [subject]})
    return store


def _code_registry_slugs():
    from nomorals.media.contentops.niches import registry

    return set(registry.list_niches())


# ── (a) discovery: the bar, the pack, the declined list ─────────


def test_corroborated_cluster_becomes_a_proposal_with_pack():
    store = _trend_claim(_gold_cluster(ExploreStore()))
    report = discover_niches(store, now=T0 + 5000)
    assert report["ok"]
    assert [p["slug"] for p in report["proposed"]] == ["gold_trading"]
    rec = store.niche_proposal("gold_trading")
    assert rec["status"] == NICHE_PROPOSED
    # members with their sighting inputs
    members = {m["name"]: m for m in rec["members"]}
    assert set(members) == {"NG Gold Room", "xauusd level breakdown"}
    assert members["NG Gold Room"]["sightings"] == 4
    assert members["NG Gold Room"]["sweeps"] == ["sweep-1", "sweep-2"]
    # backing claim cited; exemplars cited from stored evidence
    assert [c["kind"] for c in rec["claims"]] == ["trend"]
    pack = rec["evidence_pack"]
    assert pack["sightings"] == 8
    assert pack["sweeps"] == ["sweep-1", "sweep-2"]
    assert set(pack["source_classes"]) == {"video", "web"}
    assert pack["exemplars"]
    assert {e["url"] for e in pack["exemplars"]} <= {
        "https://youtu.be/g1", "https://youtu.be/g2"}
    # the score WITH its inputs, and the why-not-existing note
    assert pack["score"] > 0
    assert pack["score_inputs"]["members"] == 2
    assert pack["score_inputs"]["trend_claims"] == 1
    assert pack["score_inputs"]["max_velocity_ratio"] == 4.0
    assert "gold trading" in pack["velocity"]
    assert pack["why_not_existing"]


def test_discovery_is_idempotent_refresh_not_duplication():
    store = _trend_claim(_gold_cluster(ExploreStore()))
    discover_niches(store, now=T0 + 5000)
    again = discover_niches(store, now=T0 + 6000)
    assert again["proposed"] == []
    assert [p["slug"] for p in again["refreshed"]] == ["gold_trading"]
    assert len(store.niche_proposals()) == 1
    rec = store.niche_proposal("gold_trading")
    assert rec["created_at"] == T0 + 5000
    assert rec["updated_at"] == T0 + 6000


def test_thin_evidence_declines_with_reasons():
    # one watching candidate, one sighting, one sweep: no cluster
    store = ExploreStore()
    _note(store, "topic", "lonely topic", beat="obscure beat",
          sweep_id="sweep-1", url="https://youtu.be/x", now=T0)
    report = discover_niches(store, now=T0 + 100)
    assert report["proposed"] == []
    assert store.niche_proposals() == []

    # a promoted LONE candidate is evaluated — and declined as a
    # single member
    store2 = ExploreStore()
    _note(store2, "channel", "Solo Channel", beat="solo beat",
          sweep_id="sweep-1", url="https://youtu.be/y", now=T0)
    _promote(store2, store2.candidates()[0]["key"])
    report2 = discover_niches(store2, now=T0 + 100)
    assert report2["proposed"] == []
    reasons = [d["reason"] for d in report2["declined"]]
    assert any("single member" in r for r in reasons)

    # a two-member cluster with no promotion and no claim
    store3 = _gold_cluster(ExploreStore(), promote=False)
    report3 = discover_niches(store3, now=T0 + 5000)
    assert report3["proposed"] == []
    reasons3 = [d["reason"] for d in report3["declined"]]
    assert any("watching evidence only" in r for r in reasons3)

    # a cluster sighted in ONE sweep only cannot corroborate
    store4 = ExploreStore()
    for url in ("https://youtu.be/a", "https://youtu.be/b",
                "https://youtu.be/c"):
        _note(store4, "channel", "One Sweep TV", beat="one beat",
              sweep_id="sweep-1", url=url, now=T0)
        _note(store4, "topic", "one beat topic", beat="one beat",
              sweep_id="sweep-1", url=url, now=T0)
    for cand in store4.candidates():
        _promote(store4, cand["key"])
    report4 = discover_niches(store4, now=T0 + 5000)
    assert report4["proposed"] == []
    reasons4 = [d["reason"] for d in report4["declined"]]
    assert any("single sweep" in r for r in reasons4)


def test_cluster_that_is_an_existing_niche_is_declined():
    store = ExploreStore()
    for i, sweep in enumerate(["sweep-1", "sweep-2"]):
        t = T0 + i * 3600
        for url in ("https://youtu.be/f1", "https://youtu.be/f2"):
            _note(store, "channel", "Money Facts Daily",
                  beat="finance facts", sweep_id=sweep, url=url,
                  now=t)
            _note(store, "topic", "money fact of the day",
                  beat="finance facts", sweep_id=sweep, url=url,
                  now=t)
    for cand in store.candidates():
        _promote(store, cand["key"])
    report = discover_niches(store, now=T0 + 5000)
    assert report["proposed"] == []
    reasons = [d["reason"] for d in report["declined"]]
    assert any("already covered by the existing niche "
               "finance_facts" in r for r in reasons)


# ── (b) + (e) proposal changes nothing; the floor always holds ──


def test_proposal_changes_nothing_effective():
    store = _trend_claim(_gold_cluster(ExploreStore()))
    before = {row["slug"] for row in effective_niches(store)}
    assert before == FLOOR_SLUGS
    discover_niches(store, now=T0 + 5000)
    after = effective_niches(store)
    assert {row["slug"] for row in after} == FLOOR_SLUGS
    assert all(row["origin"] == "floor" for row in after)
    # and the code registry itself is untouched
    assert _code_registry_slugs() == FLOOR_SLUGS
    # the floor reads are exactly the seven shipped plugins
    assert {row["slug"] for row in floor_niches()} == FLOOR_SLUGS


def test_proposals_persist_through_the_kv_store(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    kv = KVStore(db)
    store = _trend_claim(_gold_cluster(ExploreStore(kv)))
    discover_niches(store, now=T0 + 5000)
    reloaded = ExploreStore(kv)
    rec = reloaded.niche_proposal("gold_trading")
    assert rec is not None
    assert rec["status"] == NICHE_PROPOSED
    assert rec["evidence_pack"]["sightings"] == 8


# ── (c) adoption: overlay flip + draft scaffold, never silent ───


def test_adoption_flips_overlay_and_scaffolds_a_draft(tmp_path):
    store = _trend_claim(_gold_cluster(ExploreStore()))
    discover_niches(store, now=T0 + 5000)
    drafts = tmp_path / "drafts"
    result = adopt_niche(store, "#1", dest_dir=drafts, now=T0 + 6000)
    assert result["ok"], result
    rec = store.niche_proposal("gold_trading")
    assert rec["status"] == NICHE_ADOPTED
    assert rec["adopted_at"] == T0 + 6000
    # the draft exists, in the plugins' own scaffold shape, at
    # the drafts location
    draft = drafts / "gold_trading.py"
    assert result["draft_path"] == str(draft)
    text = draft.read_text(encoding="utf-8")
    assert "class GoldTradingNiche(NichePlugin)" in text
    assert "register(GoldTradingNiche())" in text
    assert "scaffold" in text  # generated-by header
    # the effective set gains the niche via the OVERLAY
    effective = {row["slug"]: row for row in effective_niches(store)}
    assert effective["gold_trading"]["origin"] == "overlay"
    assert set(effective) >= FLOOR_SLUGS
    # NOT via the draft silently loading: the code registry is
    # unchanged and the draft is not an importable package module
    assert _code_registry_slugs() == FLOOR_SLUGS
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            "nomorals.media.contentops.niches.gold_trading")
    # the generic materialisation serves the adopted record and
    # passes the registry's own shape validation
    plugin = overlay_plugin("gold_trading", store)
    assert plugin is not None and plugin.name == "gold_trading"
    from nomorals.media.contentops.niches.registry import validate

    assert validate(plugin) == []
    assert len(plugin.script_prompt("xauusd levels")) >= 200
    plan = plugin.visual_strategy("hook\n\npayoff")
    assert plan.scenes and plan.style_lock


def test_adoption_requires_the_proposed_state(tmp_path):
    store = _trend_claim(_gold_cluster(ExploreStore()))
    # nothing proposed yet — the act cannot invent a niche
    result = adopt_niche(store, "gold_trading",
                         dest_dir=tmp_path, now=T0)
    assert not result["ok"]
    discover_niches(store, now=T0 + 5000)
    assert adopt_niche(store, "gold_trading", dest_dir=tmp_path,
                       now=T0 + 6000)["ok"]
    # and it cannot be adopted twice
    again = adopt_niche(store, "gold_trading", dest_dir=tmp_path,
                        now=T0 + 7000)
    assert not again["ok"]
    assert "only a proposed niche" in again["reason"]
    assert overlay_plugin("motivation", store) is None  # floor
    assert overlay_plugin("nope", store) is None


def test_adoption_failure_leaves_the_proposal_proposed(
        tmp_path, monkeypatch):
    store = _trend_claim(_gold_cluster(ExploreStore()))
    discover_niches(store, now=T0 + 5000)

    def _boom(*args, **kwargs):
        raise RuntimeError("disk is full")

    import importlib

    scaffold_mod = importlib.import_module(
        "nomorals.media.contentops.niches.scaffold")
    monkeypatch.setattr(scaffold_mod, "scaffold", _boom)
    result = adopt_niche(store, "gold_trading",
                         dest_dir=tmp_path / "d", now=T0 + 6000)
    assert not result["ok"]
    assert "stays proposed" in result["reason"]
    assert store.niche_proposal("gold_trading")["status"] \
        == NICHE_PROPOSED


# ── (d) dismissal sticks until the evidence moves ───────────────


def test_dismissal_sticks_on_the_same_evidence():
    store = _trend_claim(_gold_cluster(ExploreStore()))
    discover_niches(store, now=T0 + 5000)
    result = dismiss_niche(store, "gold_trading", now=T0 + 6000)
    assert result["ok"]
    assert store.niche_proposal("gold_trading")["status"] \
        == NICHE_DISMISSED
    report = discover_niches(store, now=T0 + 7000)
    assert report["proposed"] == []
    assert store.niche_proposal("gold_trading")["status"] \
        == NICHE_DISMISSED
    reasons = [d["reason"] for d in report["declined"]]
    assert any("dismissed by the owner" in r for r in reasons)
    # a fresh sighting after the dismissal is new evidence
    _note(store, "channel", "NG Gold Room", beat="gold trading",
          sweep_id="sweep-3", url="https://youtu.be/g3",
          now=T0 + 8000)
    revived = discover_niches(store, now=T0 + 9000)
    assert [p["slug"] for p in revived["proposed"]] == ["gold_trading"]
    assert store.niche_proposal("gold_trading")["status"] \
        == NICHE_PROPOSED


# ── (f) surfaces: proposals as proposals; owner-act verbs ───────


def _ctx(tmp_path=None):
    return SimpleNamespace(
        db=None, settings=SimpleNamespace(partner=None),
        extras={}, tools=None, home_path=str(tmp_path or ""),
        workspace_dir=str(tmp_path or ""))


def test_board_renders_proposals_as_proposals():
    store = _trend_claim(_gold_cluster(ExploreStore()))
    discover_niches(store, now=T0 + 5000)
    text = render_explore_niches(store=store)
    assert "gold_trading" in text
    assert "NOT live" in text
    assert "production floor" in text
    for slug in FLOOR_SLUGS:
        assert slug in text
    assert "candidate:" not in text  # no internal record ids
    board = render_explore_board(store=store)
    assert "niche proposals: 1 awaiting your tap" in board


def test_control_explore_adopt_and_dismiss_verbs(
        tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_media import (
        RuntimeMediaMixin,
    )

    store = _trend_claim(_gold_cluster(ExploreStore()))
    discover_niches(store, now=T0 + 5000)
    monkeypatch.setattr(engine_mod, "_DEFAULT_STORE", store)
    monkeypatch.setenv("NM_NICHE_DRAFTS_DIR",
                       str(tmp_path / "drafts"))
    surface = RuntimeMediaMixin()
    surface.context = _ctx(tmp_path)
    surface._scheduler = None

    niches_text = surface._control_explore("niches")
    assert "gold_trading" in niches_text
    assert "NOT live" in niches_text

    adopted = surface._control_explore("adopt #1")
    assert "adopted into the overlay" in adopted
    assert "not a production plugin" in adopted
    assert store.niche_proposal("gold_trading")["status"] \
        == NICHE_ADOPTED
    assert (tmp_path / "drafts" / "gold_trading.py").exists()

    dismissed = surface._control_explore("dismiss gold_trading")
    assert "dismissed" in dismissed
    assert store.niche_proposal("gold_trading")["status"] \
        == NICHE_DISMISSED

    bogus = surface._control_explore("adopt #9")
    assert "couldn't adopt" in bogus
    no_ref = surface._control_explore("adopt")
    assert no_ref.startswith("usage: /explore adopt")
    usage = surface._control_explore("bogus")
    assert "niches" in usage and "adopt <ref>" in usage


def test_tool_niches_adopt_dismiss_actions(tmp_path, monkeypatch):
    from nomorals.media.explore import tool as tool_mod
    from nomorals.tools.registry import ToolRegistry

    store = _trend_claim(_gold_cluster(ExploreStore()))
    discover_niches(store, now=T0 + 5000)
    monkeypatch.setattr(engine_mod, "_DEFAULT_STORE", store)
    monkeypatch.setenv("NM_NICHE_DRAFTS_DIR",
                       str(tmp_path / "drafts"))
    ctx = SimpleNamespace(
        db=None, settings=SimpleNamespace(partner=None),
        extras={}, tools=None, home_path=str(tmp_path),
        workspace_dir=str(tmp_path))
    registry = ToolRegistry(ctx)
    tool_mod.register(registry)

    out = registry.call("video_explore", action="niches")
    assert out.ok, out.error if not out.ok else ""
    payload = out.unwrap()
    assert [p["slug"] for p in payload["proposals"]] \
        == ["gold_trading"]
    assert {row["slug"] for row in payload["effective"]} \
        == FLOOR_SLUGS
    assert "NOT live" in payload["text"]

    adopted = registry.call("video_explore", action="adopt",
                            slug="gold_trading")
    assert adopted.ok
    assert adopted.unwrap()["ok"] is True
    assert {row["slug"] for row in
            registry.call("video_explore", action="niches")
            .unwrap()["effective"]} == FLOOR_SLUGS | {"gold_trading"}

    dismissed = registry.call("video_explore", action="dismiss",
                              slug="#1")
    # #1 no longer resolves (nothing proposed) — honest refusal
    assert dismissed.unwrap()["ok"] is False
    dismissed2 = registry.call("video_explore", action="dismiss",
                               slug="gold_trading")
    assert dismissed2.unwrap()["ok"] is True
    assert store.niche_proposal("gold_trading")["status"] \
        == NICHE_DISMISSED


# ── (g) the sweep cycle: discovery rides, never fails it ────────


class _FakeInstrument:
    def __init__(self, name, rows=None):
        self.name = name
        self.rows = rows if rows is not None else []
        self.calls = []

    def search(self, query, *, limit=10):
        self.calls.append(query)
        return [dict(r) for r in self.rows[:limit]]


def _beats(topic="gold trading"):
    return [{"topic": topic, "queries": ["gold trading tutorial"],
             "origins": ["test"]}]


def test_cycle_ends_with_discovery_counts(tmp_path):
    store = _trend_claim(_gold_cluster(ExploreStore()))
    cycle = run_sweep_cycle(
        None, store=store, beats=_beats(),
        instruments=[_FakeInstrument("fake_web")], manual=True,
        now=time.time())
    assert cycle["ok"] and cycle["sweep"] is not None
    assert cycle["niches"]["proposed"] == 1
    assert store.niche_proposal("gold_trading") is not None


def test_discovery_failure_never_fails_the_sweep(
        tmp_path, monkeypatch):
    store = ExploreStore()

    def _boom(store_, **kwargs):
        raise RuntimeError("clustering exploded")

    monkeypatch.setattr(niches_mod, "discover_niches", _boom)
    cycle = run_sweep_cycle(
        None, store=store, beats=_beats(),
        instruments=[_FakeInstrument("fake_web")], manual=True,
        now=time.time())
    assert cycle["ok"] and cycle["sweep"] is not None
    assert "RuntimeError" in cycle["niches"]["error"]
    from nomorals.media.explore.board import render_sweep_cycle

    text = render_sweep_cycle(cycle)
    assert "niche discovery hit a problem" in text
