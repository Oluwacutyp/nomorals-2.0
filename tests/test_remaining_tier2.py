"""Remaining Tier 2 invariants: graph missions, vision critique bridge,
property cross-listing, travel visa honesty, TA evidence decay.

Every test pins an invariant from the build spec — no test merely
checks that a function exists.
"""

import io
import json
import time

import pytest

from nomorals.planning.graph_missions import (
    PlanGraph, plan_authoring, compiled, execute_graph,
)
from nomorals.vision.critique_bridge import (
    compare_images, critique_from_vision, describe_screen,
)
from nomorals.property.cross_listing import (
    Listing, PropertyPassportStore, cross_listing_score, group_listings,
)
from nomorals.travel import visa as visa_mod
from nomorals.ta.evidence import (
    EvidenceStore, decay_score, is_stale, monitor,
)


def _png_bytes(color_a, color_b=None, size=(16, 16)):
    """Deterministic PNG bytes: solid color, or left/right halves."""
    from PIL import Image
    img = Image.new("RGB", size, color_a)
    if color_b is not None:
        right = Image.new("RGB", (size[0] // 2, size[1]), color_b)
        img.paste(right, (size[0] // 2, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── graph missions ──────────────────────────────────────────────

def test_graph_mission_finds_alternative_on_failure():
    """A failed node with an alternative reroutes: 'found another way',
    not 'failed at step N'."""
    spec = {
        "goal": "ship the report",
        "tasks": [
            {"id": "collect", "label": "collect facts", "action": "collect",
             "depends_on": [],
             "alternatives": [
                 {"id": "collect-cache", "label": "use cached facts",
                  "action": "collect_cache"}]},
            {"id": "draft", "label": "draft report", "action": "draft",
             "depends_on": ["collect"]},
        ],
    }
    graph = plan_authoring(spec)
    assert graph.nodes, "authoring must produce nodes"

    def boom(node, ctx):
        raise RuntimeError("primary source down")

    def ok(node, ctx):
        return True

    result = execute_graph(graph, {"collect": boom,
                                   "collect_cache": ok,
                                   "draft": ok})
    assert result.ok is True
    assert result.reroutes, "failure must record a reroute"
    assert result.reroutes[0]["failed"] == "collect facts"
    assert result.reroutes[0]["alternative"] == "use cached facts"
    assert "another way" in result.message
    assert "failed at step" not in result.message


def test_graph_mission_fails_honestly_with_no_working_alternative():
    spec = {"goal": "g", "tasks": [
        {"id": "a", "label": "do a", "action": "a", "depends_on": []}]}
    graph = plan_authoring(spec)
    result = execute_graph(graph, {"a": lambda n, c: False})
    assert result.ok is False
    assert result.failed == ["do a"]
    assert "another way" not in result.message


def test_graph_mission_round_trips_json_and_compiles():
    spec = {"goal": "g", "tasks": [
        {"id": "a", "label": "do a", "action": "a", "depends_on": []},
        {"id": "b", "label": "do b", "action": "b", "depends_on": ["a"]}]}
    graph = plan_authoring(spec)
    restored = PlanGraph.from_json(graph.to_json())
    assert restored.goal == "g"
    assert {n.label for n in restored.nodes.values()} == {
        n.label for n in graph.nodes.values()}
    plan = compiled(restored)
    assert plan.ok is True
    labels = [restored.nodes[nid].label for nid in plan.order]
    assert labels.index("do a") < labels.index("do b")


def test_graph_mission_persists_to_file(tmp_path):
    spec = {"goal": "g", "tasks": [
        {"id": "a", "label": "do a", "action": "a", "depends_on": []}]}
    graph = plan_authoring(spec)
    path = str(tmp_path / "plan.json")
    assert graph.save(path) is True
    loaded = PlanGraph.load(path)
    assert loaded is not None and loaded.goal == "g"
    assert json.loads(open(path).read())["goal"] == "g"


# ── vision -> critique bridge ───────────────────────────────────

def test_vision_diff_deterministic():
    a = _png_bytes((10, 20, 30))
    b = _png_bytes((200, 100, 50))
    first = compare_images(a, b)
    second = compare_images(a, b)
    assert first == second, "same byte inputs must give identical output"
    assert first["ok"] is True
    assert first["hash_a"] and first["hash_b"]
    assert first["diff_score"] > 0
    assert first["regions"], "a real change must report a region"


def test_vision_diff_identical_is_zero():
    a = _png_bytes((10, 20, 30))
    result = compare_images(a, a)
    assert result["identical"] is True
    assert result["diff_score"] == 0.0
    assert result["hash_distance"] == 0
    assert result["regions"] == []


def test_vision_critique_checklist_from_diff():
    a = _png_bytes((10, 20, 30))
    b = _png_bytes((200, 100, 50))
    report = critique_from_vision(a, b)
    assert report["verdict"] in ("ship", "revise", "rewrite")
    checks = {c["check"] for c in report["checklist"]}
    assert "pixel-identical" in checks
    assert "perceptual near-duplicate" in checks
    identical = critique_from_vision(a, a)
    assert identical["score"] == 1.0
    assert identical["verdict"] == "ship"


def test_describe_screen_never_hallucinates():
    empty = describe_screen("abc123", [])
    assert empty["elements"] == []
    assert empty["counts"] == {}
    assert "none inferred" in empty["note"]
    supplied = describe_screen("abc123", [
        {"kind": "button", "label": "Buy now"},
        {"kind": "input", "label": "Search"},
        {"kind": "text", "label": "Price: $10"},
        {"kind": "widget", "label": "Mystery"},  # unknown kind: dropped
        {"kind": "button", "label": ""},          # no label: dropped
    ])
    labels = [e["label"] for e in supplied["elements"]]
    assert labels == ["Buy now", "Search", "Price: $10"]
    assert supplied["dropped"] == 2
    assert supplied["counts"] == {"button": 1, "input": 1, "text": 1}
    assert supplied["source"] == "supplied-elements-only"


# ── property cross-listing ──────────────────────────────────────

def test_cross_listing_dedupes_same_property():
    listings = [
        Listing(source="jumia", address="12, Allen Avenue, Ikeja",
                price=2_500_000, photo_hashes=["ph1", "ph2"]),
        Listing(source="propertypro", address="12 Allen Avenue Ikeja",
                price=2_500_000, photo_hashes=["ph2", "ph3"]),
        Listing(source="agent", address="45 Opebi Road, Ikeja",
                price=2_500_000, photo_hashes=["ph1"]),
    ]
    groups = group_listings(listings)
    assert len(groups) == 2, "same property merges; different address stays apart"
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


def test_cross_listing_score_counts_sources():
    group = [
        Listing(source="jumia", address="12 Allen Avenue, Ikeja",
                price=2_500_000, photo_hashes=["ph1"]),
        Listing(source="propertypro", address="12 Allen Avenue, Ikeja",
                price=2_500_000, photo_hashes=["ph1"]),
    ]
    score = cross_listing_score(group, total_sources=3)
    assert score["sources"] == 2
    assert score["score"] == pytest.approx(2 / 3, abs=0.0001)
    single = cross_listing_score(group[:1], total_sources=3)
    assert single["score"] == pytest.approx(1 / 3, abs=0.0001)


def test_property_passport_versioned_and_persisted(tmp_path):
    store = PropertyPassportStore(str(tmp_path / "passports.json"))
    passport = store.create("12 Allen Avenue, Ikeja")
    assert passport is not None
    assert passport.version == 1
    assert store.add_event(passport.passport_id, "listed",
                           "seen on jumia", source="jumia") is True
    assert store.add_document(passport.passport_id, "vault:doc1",
                              "Survey") is True
    assert store.add_viewing(passport.passport_id, time.time(),
                             outcome="scheduled") is True
    reloaded = PropertyPassportStore(str(tmp_path / "passports.json"))
    same = reloaded.by_address("12, Allen Avenue, Ikeja")
    assert same is not None
    assert same.version == 4  # create + 3 mutations
    assert len(same.history) == 1
    assert len(same.documents) == 1
    assert len(same.viewings) == 1
    # Same property gets ONE passport, not a duplicate.
    assert reloaded.create("12 Allen Avenue Ikeja").passport_id == \
        passport.passport_id


# ── travel visa ─────────────────────────────────────────────────

def test_visa_known_route():
    result = visa_mod.visa_needed("NG", "GH", "NG")
    assert result["known"] is True
    assert result["status"] == "visa_free"
    assert result["needed"] is False
    docs = visa_mod.documents_required("NG", "GB", "NG")
    assert docs["known"] is True
    assert "valid passport" in docs["documents"]


def test_visa_unknown_is_honest():
    """A route not in the table is unknown — never an invented rule."""
    result = visa_mod.visa_needed("NG", "JP", "NG")
    assert result["known"] is False
    assert result["status"] == "unknown"
    assert result["needed"] is None, "unknown must not fabricate yes/no"
    docs = visa_mod.documents_required("NG", "JP", "NG")
    assert docs["known"] is False
    assert docs["documents"] == [], "unknown must not invent a document list"
    full = visa_mod.lookup("NG", "JP", "NG")
    assert full["known"] is False
    assert "never guesses" in full["notes"]


def test_visa_table_register_extends_without_inventing():
    table = visa_mod.VisaTable()
    assert table.lookup("NG", "JP", "NG") is None
    rule = visa_mod.VisaRequirement(
        origin="NG", destination="JP", passport="NG",
        status="visa_required", documents=("valid passport",),
        notes="registered by caller")
    assert table.register(rule) is True
    assert table.lookup("NG", "JP", "NG") is not None
    assert table.lookup("NG", "KR", "NG") is None  # still unknown


# ── TA evidence decay ───────────────────────────────────────────

def test_ta_decay_decreases_with_age():
    assert decay_score(0) == 1.0
    assert decay_score(0) > decay_score(30) > decay_score(90) > decay_score(365)
    assert decay_score(30) == pytest.approx(0.5)
    assert decay_score(-5) == 1.0  # negative age clamps to fresh
    assert 0.0 <= decay_score(10_000) < 0.001


def test_ta_monitor_flags_stale_and_keeps_it(tmp_path):
    store = EvidenceStore(str(tmp_path / "evidence.json"))
    now = time.time()
    fresh = store.record("sweep+FVG", "H1", result="win", timestamp=now - 5 * 86400)
    stale = store.record("sweep+FVG", "H1", result="win",
                         timestamp=now - 200 * 86400)
    assert fresh is not None and stale is not None
    report = monitor(store, now=now)
    assert report["total"] == 2
    assert report["fresh"] == 1
    assert report["stale_count"] == 1
    assert report["stale"][0]["record_id"] == stale.record_id
    assert report["stale"][0]["age_days"] == pytest.approx(200.0, abs=0.5)
    assert "sweep+FVG" in report["stale_setups"]
    # Stale evidence is flagged, NOT silently dropped.
    assert store.get(stale.record_id) is not None
    assert is_stale(200) is True
    assert is_stale(5) is False


def test_ta_weighted_results_favour_fresh_evidence(tmp_path):
    store = EvidenceStore(str(tmp_path / "evidence.json"))
    now = time.time()
    store.record("qm-m5", "M5", result="win", timestamp=now - 1 * 86400)
    store.record("qm-m5", "M5", result="loss", timestamp=now - 300 * 86400)
    summary = store.weighted_results("qm-m5", now=now)
    assert summary["records"] == 2
    # The fresh win outweighs the ancient loss after decay.
    assert summary["weights"]["win"] > summary["weights"]["loss"]
    assert summary["decayed_win_rate"] is not None
    assert summary["decayed_win_rate"] > 0.5


def test_ta_record_rejects_empty_setup(tmp_path):
    store = EvidenceStore(str(tmp_path / "evidence.json"))
    assert store.record("", "H1") is None
    assert store.record("sweep", "") is None
    assert store.list() == []
