"""Tests for MEDIA upgrade tiers (nomorals/media/upgrade/).

One invariant test per capability. No ffmpeg, no model, no network:
images are synthetic PIL images, generation is a fake callable.
"""

from __future__ import annotations

import pytest
from PIL import Image

from nomorals.media.edit_engine import EditSpec
from nomorals.media.edit_engine.timeline import Clip
from nomorals.media.upgrade import (
    CONSISTENCY_TARGET,
    AdjustmentStack,
    Asset,
    AssetLibrary,
    BudgetLedger,
    CameraMove,
    Mask,
    ModelRoute,
    OpStack,
    EditOp,
    ProjectRoom,
    RecipeHistory,
    Reference,
    ReviewRoom,
    RouteTable,
    SceneBible,
    TasteModel,
    TextComposite,
    apply_stack_to_spec,
    auto_highlights,
    beat_times,
    brief_to_checklist,
    camera_plan,
    composite_text,
    conditioning_pack,
    contact_sheet,
    critique_render,
    derive_mask,
    detect_failures,
    drift_report,
    face_similarity,
    geometric_mask,
    inpaint_spec,
    keep_or_discard,
    mask_from_prompt,
    nigerian_pack,
    nigerian_prompt,
    outpaint_spec,
    palette_drift,
    passes_floor,
    recipe_from_spec,
    regional_prompt,
    route_for_task,
    run_pipeline,
    segment_needs_rerender,
    silence_spans,
    thumbnail_ab,
)
from nomorals.media.upgrade.failures import remedy_for


def _img(color=(200, 120, 60), size=(64, 64)):
    return Image.new("RGB", size, color)


def _spec():
    return EditSpec(clips=[Clip(path="a.mp4", end=10.0), Clip(path="b.mp4", end=8.0)])


# --- Tier 0: critique ---
def test_critique_floor_refuses_garbage():
    crit = critique_render("a red car", scores={a: 0.1 for a in
        ("prompt_adherence", "technical", "composition", "consistency", "aesthetic")})
    assert crit.verdict == "reject" and not passes_floor(crit)


def test_critique_claims_from_brief():
    claims = brief_to_checklist("A woman wearing red, standing in a Lagos market at sunset.")
    assert claims and claims[0].kind in ("setting", "framing", "subject")
    assert "market" in claims[0].keywords or "sunset" in claims[0].keywords


def test_critique_measure_image_real_features():
    crit = critique_render("a red car", image=_img())
    assert 0.0 <= crit.scores["technical"] <= 1.0


# --- Tier 0: masks ---
def test_geometric_mask_contains():
    m = geometric_mask("rect", x0=0.0, y0=0.0, x1=0.5, y1=1.0)
    assert m.contains(0.25, 0.5) and not m.contains(0.75, 0.5)
    assert m.to_dict()["kind"] == "geometric:rect"


def test_prompt_mask_needs_resolver_or_stays_unresolved():
    m = mask_from_prompt("the gele")
    assert m.kind == "prompt"
    with pytest.raises(ValueError):
        mask_from_prompt("   ")
    resolved = mask_from_prompt("the gele", resolver=lambda p: geometric_mask("ellipse", cx=0.5, cy=0.5, rx=0.3, ry=0.3))
    assert resolved.kind == "geometric:ellipse"


def test_derived_masks_are_grids():
    for kind in ("subject", "depth", "edges", "skin", "sky"):
        grid = derive_mask(_img(), kind)
        assert len(grid) == 64 and len(grid[0]) == 64
        assert all(v in (0.0, 1.0) for row in grid for v in row)
    with pytest.raises(ValueError):
        derive_mask(_img(), "nope")


# --- Tier 0: conditioning ---
def test_conditioning_pack_normalises_and_fingerprints():
    refs = [Reference(kind="style", source="s.png"), Reference(kind="character", source="c.png")]
    pack = conditioning_pack(refs)
    assert abs(sum(r.weight for r in pack.refs) - 1.0) < 1e-9
    assert conditioning_pack(refs).fingerprint == pack.fingerprint
    with pytest.raises(ValueError):
        conditioning_pack([])


# --- Tier 0: op stack ---
def test_opstack_undo_branch_explain_and_apply():
    stack = OpStack().push(EditOp("grade", {"saturation": 0.1})).push(EditOp("text", {"text": "hi"}))
    assert len(stack) == 2 and len(stack.explain()) == 2
    branched = stack.branch()
    assert stack.undo().name == "text" and len(stack) == 1 and len(branched) == 2
    out = apply_stack_to_spec(_spec(), stack)
    assert out.clips[0].effects and out.clips[0].effects[0].name == "grade"
    assert _spec().clips[0].effects == ()  # base never mutated
    with pytest.raises(ValueError):
        EditOp("teleport", {})
    # determinism: same stack + same base = same spec dict
    s2 = OpStack().push(EditOp("grade", {"saturation": 0.1}))
    assert apply_stack_to_spec(_spec(), s2).to_dict() == out.to_dict()


# --- Tier 1: pipeline ---
def test_pipeline_seven_stages_and_floor():
    res = run_pipeline("a red car in Lagos", rung="preview")
    assert res.stage_log and res.shipped  # default fake scores pass floor
    assert len(res.refined) <= 2  # preview keeps 2


def test_pipeline_cull_ships_nothing_below_floor():
    def bad_gen(brief, count, seed):
        return [{"id": str(i), "scores": {a: 0.05 for a in
                ("prompt_adherence", "technical", "composition", "consistency", "aesthetic")}}
                for i in range(count)]
    res = run_pipeline("a red car", rung="preview", generate=bad_gen)
    assert res.shipped == [] and res.kept == []


def test_contact_sheet_real_grid():
    sheet = contact_sheet([{"image": _img((255, 0, 0))}, {"image": _img((0, 255, 0))}], cols=2, thumb=32)
    assert sheet.size == (64, 32)


# --- Tier 1: bible / consistency ---
def test_consistency_measured_not_assumed():
    bible = SceneBible(title="t", palette=["#f5c396"])
    rep = drift_report(bible, _img(), _img())
    assert rep.meets_target is (rep.overall >= CONSISTENCY_TARGET)
    assert rep.method == "feature-proxy"
    assert face_similarity(_img(), _img()) == pytest.approx(1.0)
    assert palette_drift(["#ffffff"], ["#ffffff"]) == 0.0
    assert palette_drift([], ["#ffffff"]) == 1.0


def test_segment_rerender_flags_only_bad_segments():
    bad = segment_needs_rerender({"id": "s1"}, identity_drift=0.9)
    assert bad["needs_rerender"] and bad["duration_s"] == 2.0
    good = segment_needs_rerender({"id": "s2"}, flow_jump=0.1)
    assert not good["needs_rerender"]


# --- Tier 1: recipes ---
def test_recipe_reproducible_fingerprint_and_branching():
    r1 = recipe_from_spec("brief", _spec(), models={"img": "m1"}, seeds={"img": 7})
    r2 = recipe_from_spec("brief", _spec(), models={"img": "m1"}, seeds={"img": 7})
    assert r1.fingerprint == r2.fingerprint
    hist = RecipeHistory()
    hist.commit(r1)
    hist.branch("alt")
    assert hist.branches() == ["alt", "main"] and hist.head("alt") == r1


# --- Tier 1: typography ---
def test_text_composited_never_generated():
    src = _img()
    out = composite_text(src, TextComposite(text="OWAMBE", position="bottom"))
    assert out is not src and out.size == src.size
    # compositing changed pixels (text drawn) but source untouched
    assert list(src.getdata()) != list(out.getdata())


# --- Tier 1: camera ---
def test_camera_plan_first_last_intermediate():
    plan = camera_plan(CameraMove(kind="push-in", duration_s=1.0), fps=10)
    kfs = plan["keyframes"]
    assert kfs[0]["progress"] == 0.0 and kfs[-1]["progress"] == 1.0
    assert kfs[-1]["zoom"] > kfs[0]["zoom"]
    with pytest.raises(ValueError):
        CameraMove(kind="teleport")


# --- Tier 1: regional ---
def test_regional_inpaint_outpaint_adjustments_deterministic():
    m = geometric_mask("rect", x0=0, y0=0, x1=0.5, y1=0.5)
    rp = regional_prompt("fix the gele", m)
    assert rp.mask is m
    assert inpaint_spec("a.png", m, "fix hands")["op"] == "inpaint"
    assert outpaint_spec("a.png", direction="left", extra_px=128)["extra_px"] == 128
    stack = AdjustmentStack({"brightness": 0.1, "contrast": 0.1})
    assert list(stack.apply(_img()).getdata()) == list(stack.apply(_img()).getdata())
    assert stack.apply(_img()).size == (64, 64)


# --- Tier 2: taste ---
def test_taste_model_abstains_then_learns():
    model = TasteModel()
    assert keep_or_discard(model, {"warmth": 0.9}) == "undecided"
    for _ in range(60):
        model.learn({"warmth": 0.9, "shadow_warmth": 0.9, "centeredness": 0.1}, kept=True)
        model.learn({"warmth": 0.1, "shadow_warmth": 0.1, "centeredness": 0.9}, kept=False)
    assert model.examples >= 100 and model.learned
    assert keep_or_discard(model, {"warmth": 0.9, "shadow_warmth": 0.9, "centeredness": 0.1}) == "keep"
    assert any("warm" in r for r in model.rules())


# --- Tier 2: failures ---
def test_failure_library_detector_plus_remedy():
    assert "typography" in remedy_for("text").lower() or "composite" in remedy_for("text").lower()
    findings = detect_failures(_img(), text_zones=[{"x": 0, "y": 0}], prompt_mentions=["person"])
    assert any(f.mode == "text" for f in findings)
    with pytest.raises(ValueError):
        remedy_for("nope")


# --- Tier 2: routing ---
def test_routing_health_scored_fallback():
    table = RouteTable([ModelRoute(task="photoreal", model="a", health=0.9, quality=0.8),
                        ModelRoute(task="photoreal", model="b", health=0.9, quality=0.5)])
    assert route_for_task(table, "photoreal").model == "a"
    route_a = table.fallback_chain("photoreal")[0]
    assert route_a.model == "a"
    for _ in range(4):
        route_a.observe(success=False)
    # after 4 failures health 0.9-1.0 -> 0, falls back to b
    assert route_for_task(table, "photoreal").model == "b"
    assert route_for_task(table, "anime") is None


# --- Tier 2: nigerian packs ---
def test_nigerian_packs():
    pack = nigerian_pack("owambe")
    assert pack.palette and pack.critique_notes
    assert "Lagos" in nigerian_prompt("lagos_street", "danfo at rush hour")
    assert "Owambe" in nigerian_prompt("owambe", "gele and aso-oke")
    with pytest.raises(ValueError):
        nigerian_pack("nope")


# --- Extras: projects ---
def test_project_room_snapshot_and_budget_cap():
    room = ProjectRoom(name="p1", palette=["#f5c396"], ledger=BudgetLedger(cap=10.0))
    assert room.ledger.charge(6.0, what="draft") is True
    assert room.ledger.charge(6.0, what="max") is False  # over cap blocked, not silently spent
    assert room.ledger.spent == 6.0
    assert room.snapshot()["spent"] == 6.0


def test_asset_library_token_search():
    lib = AssetLibrary()
    lib.add(Asset(id="1", kind="image", path="a.png", tags=["owambe", "gele"], caption="party"))
    lib.add(Asset(id="2", kind="image", path="b.png", tags=["street"], caption="danfo"))
    assert lib.search("owambe gele")[0].id == "1"


def test_highlights_silence_beats():
    assert auto_highlights([0.9, 0.1], [1.0, 0.0], [0.5, 0.0], top_n=1)[0]["second"] == 0
    assert silence_spans([0.5, 0.01, 0.01, 0.01, 0.5]) == [{"start": 1, "end": 4, "length": 3}]
    assert beat_times([0.1, 0.9, 0.1, 0.8, 0.1]) == [1, 3]


def test_thumbnail_ab_and_review_room_floor():
    good = {"id": "g", "scores": {a: 0.8 for a in
            ("prompt_adherence", "technical", "composition", "consistency", "aesthetic")}}
    bad = {"id": "b", "scores": {a: 0.1 for a in
           ("prompt_adherence", "technical", "composition", "consistency", "aesthetic")}}
    assert thumbnail_ab([bad, good])["id"] == "g"
    room = ReviewRoom()
    room.submit("g", good)
    assert room.approve("g") is True
    room.submit("b", bad)
    assert room.approve("b") is False  # floor invariant: cannot approve garbage
