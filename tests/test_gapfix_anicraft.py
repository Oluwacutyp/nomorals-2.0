"""AniCraft port tests (census 3A.17) — 2D animation editor organ.

Failing-first: written before the organ existed. Every case is measured
against a real fixture or a real upstream ground-truth value (AniCraft
``crates/doc/src/easing_tests.rs``); a missing fixture is a failure,
never a silent skip.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from nomorals.media.animation import (
    AiClient,
    AnimationDocument,
    AnimationEditor,
    AnimationError,
    ConsentLedger,
    DisplayList,
    Easing,
    Element,
    Role,
    Transform,
    build_preset,
    evaluate,
    list_easings,
    list_presets,
    publish,
    render_frame,
    search_presets,
)
from nomorals.media.animation import run_parity as run_animation_parity
from nomorals.media.animation import (
    verify_attribution as verify_animation_attribution,
)
from nomorals.media.animation.xfl import export_xfl, import_xfl

FIXTURES = Path(__file__).parent / "fixtures" / "anicraft"


def _fixture(name: str) -> Path:
    path = FIXTURES / name
    assert path.exists(), f"fixture missing: {name}"
    return path


def _simple_doc() -> AnimationDocument:
    doc = AnimationDocument.create("simple", width=200, height=120, fps=24)
    scene = doc.scenes[0]
    layer = scene.timeline.add_layer("ball")
    kf0 = layer.add_keyframe(0, 12)
    ball = Element.ellipse(20.0, 60.0, 15.0, 15.0, fill="#ff0000")
    ball.transform = Transform(x=0.0, y=0.0)
    kf0.add_element(ball)
    kf1 = layer.add_keyframe(12, 12)
    ball2 = Element.ellipse(20.0, 60.0, 15.0, 15.0, fill="#ff0000")
    ball2.id = ball.id
    ball2.transform = Transform(x=160.0, y=0.0)
    kf1.add_element(ball2)
    kf0.tween_to(kf1, easing=Easing.named("linear"))
    return doc


# ---------------------------------------------------------------------------
# Easing — ground truth from upstream easing_tests.rs (22 cases)
# ---------------------------------------------------------------------------

EASING_CASES = [
    ("linear", 0.3, 0.3),
    ("quad-in", 0.5, 0.25),
    ("quad-out", 0.5, 0.75),
    ("quad-in-out", 0.25, 0.125),
    ("cubic-in", 0.5, 0.125),
    ("cubic-out", 0.5, 0.875),
    ("quart-in", 0.5, 0.0625),
    ("quart-in-out", 0.75, 0.96875),
    ("quint-in", 0.5, 0.03125),
    ("quint-out", 0.5, 0.96875),
    ("sine-in", 0.5, 1.0 - math.sqrt(0.5)),
    ("sine-out", 1.0 / 3.0, 0.5),
    ("sine-in-out", 0.25, 0.5 * (1.0 - math.sqrt(0.5))),
    ("back-in", 0.5, -0.0876975),
    ("back-out", 0.5, 1.0876975),
    ("circ-in", 0.6, 0.2),
    ("circ-out", 0.5, 0.8660254),
    ("bounce-out", 0.5, 0.765625),
    ("bounce-in", 0.5, 0.234375),
    ("bounce-out", 1.0 / 2.75, 1.0),
    ("elastic-in", 0.5, -0.015625),
    ("elastic-out", 0.5, 1.015625),
]


@pytest.mark.parametrize("name,t,want", EASING_CASES)
def test_easing_ground_truth(name, t, want):
    got = Easing.named(name).apply(t)
    assert abs(got - want) < 1e-5, f"{name} at {t}: {got} vs {want}"


def test_easing_endpoints_fixed():
    for name in list_easings():
        e = Easing.named(name)
        assert abs(e.apply(0.0)) < 1e-4, name
        assert abs(e.apply(1.0) - 1.0) < 1e-4, name


def test_easing_classic_slider():
    assert abs(Easing.classic(-100.0).apply(0.5) - 0.25) < 1e-6
    assert abs(Easing.classic(100.0).apply(0.5) - 0.75) < 1e-6
    assert abs(Easing.classic(0.0).apply(0.5) - 0.5) < 1e-6


def test_easing_out_mirrors_in():
    for family in ("quad", "cubic", "quart", "quint", "sine", "back",
                   "circ", "bounce", "elastic"):
        inn = Easing.named(f"{family}-in")
        out = Easing.named(f"{family}-out")
        io = Easing.named(f"{family}-in-out")
        for k in range(0, 41):
            t = k / 40.0
            assert abs(out.apply(t) - (1.0 - inn.apply(1.0 - t))) < 1e-5
            assert abs(io.apply(t) + io.apply(1.0 - t) - 1.0) < 1e-5
            assert abs(io.apply(t * 0.5) - 0.5 * inn.apply(t)) < 1e-5


def test_easing_bezier_and_custom():
    css = Easing.bezier(0.25, 0.1, 0.25, 1.0)
    assert abs(css.apply(0.0)) < 1e-6
    assert abs(css.apply(1.0) - 1.0) < 1e-6
    assert 0.0 < css.apply(0.5) < 1.0
    wave = Easing.custom([(0.0, 0.0), (0.2, 1.3), (0.5, -0.2),
                          (0.7, 1.0), (1.0, 1.0)])
    assert abs(wave.apply(0.0)) < 1e-6
    assert abs(wave.apply(1.0) - 1.0) < 1e-6
    assert wave.apply(0.2) > 1.0  # overshoot is real, not clamped away


def test_easing_sampled():
    table = [0.0, 0.25, 0.5, 0.75, 1.0]
    e = Easing.sampled(table)
    assert abs(e.apply(0.0)) < 1e-9
    assert abs(e.apply(0.5) - 0.5) < 1e-9
    assert abs(e.apply(1.0) - 1.0) < 1e-9
    assert abs(e.apply(0.25) - 0.25) < 1e-9


def test_easing_unknown_name_refused():
    with pytest.raises(AnimationError):
        Easing.named("no-such-ease")


# ---------------------------------------------------------------------------
# Document model + codec + fidelity
# ---------------------------------------------------------------------------

def test_create_document_defaults():
    doc = AnimationDocument.create("demo")
    assert doc.name == "demo"
    assert doc.fps == 24
    assert len(doc.scenes) == 1
    assert doc.scenes[0].stage.width == 640
    assert doc.scenes[0].stage.height == 360


def test_codec_round_trip_byte_identical(tmp_path):
    doc = _simple_doc()
    raw = doc.to_bytes()
    doc2 = AnimationDocument.parse(raw)
    assert doc2.to_bytes() == raw
    assert doc2.content_hash() == doc.content_hash()
    report = doc.fidelity_report(doc2)
    assert report["changed_layers"] == []
    assert report["changed_keyframes"] == []
    assert report["changed_elements"] == []


def test_fidelity_names_only_the_edited_keyframe():
    doc = _simple_doc()
    doc2 = AnimationDocument.parse(doc.to_bytes())
    layer = doc2.scenes[0].timeline.layers[0]
    kf1 = layer.keyframes[1]
    el = kf1.elements[0]
    el.transform = Transform(x=100.0, y=0.0)
    report = doc.fidelity_report(doc2)
    assert report["changed_keyframes"] == [kf1.start]
    assert report["changed_elements"] == [el.id]
    assert report["unchanged_elements"] == 1


def test_codec_rejects_corrupt_and_bad_json():
    with pytest.raises(AnimationError):
        AnimationDocument.parse(_fixture("corrupt.acraft").read_bytes())
    with pytest.raises(AnimationError):
        AnimationDocument.parse(_fixture("invalid.acraft").read_bytes())


def test_fixture_documents_open():
    doc = AnimationDocument.parse(_fixture("simple.acraft").read_bytes())
    assert len(doc.scenes[0].timeline.layers) == 1
    assert doc.scenes[0].timeline.layers[0].keyframes[0].elements
    doc = AnimationDocument.parse(_fixture("bounce.acraft").read_bytes())
    assert doc.scenes[0].timeline.frame_count() >= 24


def test_keyframe_span_rules():
    doc = AnimationDocument.create("span")
    layer = doc.scenes[0].timeline.add_layer("l")
    kf = layer.add_keyframe(0, 5)
    assert kf.duration == 5
    assert kf.end() == 5
    assert kf.covers(4) and not kf.covers(5)
    with pytest.raises(AnimationError):
        layer.add_keyframe(3, 4)  # overlaps the first span
    with pytest.raises(AnimationError):
        layer.add_keyframe(10, 0)  # zero-length span


def test_layer_kinds():
    doc = AnimationDocument.create("kinds")
    tl = doc.scenes[0].timeline
    for kind in ("normal", "guide", "mask", "folder", "camera"):
        tl.add_layer(kind, kind=kind)
    assert [ly.kind for ly in tl.layers] == [
        "normal", "guide", "mask", "folder", "camera"]
    with pytest.raises(AnimationError):
        tl.add_layer("bad", kind="armature")  # not ported yet, refused


# ---------------------------------------------------------------------------
# Evaluation — the playhead produces real values
# ---------------------------------------------------------------------------

def test_eval_linear_tween_midpoint():
    doc = _simple_doc()
    dl = evaluate(doc, 6)
    assert isinstance(dl, DisplayList)
    assert len(dl.items) == 1
    # ball moves 0 -> 160 over frames 0..12; frame 6 is the midpoint
    assert abs(dl.items[0].transform.x - 80.0) < 0.5


def test_eval_hold_after_last_keyframe():
    doc = _simple_doc()
    dl = evaluate(doc, 20)
    assert abs(dl.items[0].transform.x - 160.0) < 0.5


def test_eval_eased_tween_is_not_linear():
    doc = _simple_doc()
    layer = doc.scenes[0].timeline.layers[0]
    layer.keyframes[0].tween_to(
        layer.keyframes[1], easing=Easing.named("quad-in"))
    dl = evaluate(doc, 6)
    # quad-in at t=0.5 eases to 0.25, so x is 40, not 80
    assert abs(dl.items[0].transform.x - 40.0) < 0.5


def test_eval_per_group_easing():
    doc = _simple_doc()
    layer = doc.scenes[0].timeline.layers[0]
    kf0, kf1 = layer.keyframes
    kf0.tween_to(kf1, easing=Easing.named("linear"),
                 groups={"position": Easing.named("quad-in"),
                         "rotation": Easing.named("quad-out")})
    kf0.elements[0].transform = Transform(x=0.0, y=0.0, rotation=0.0)
    kf1.elements[0].transform = Transform(x=160.0, y=0.0, rotation=90.0)
    dl = evaluate(doc, 6)
    assert abs(dl.items[0].transform.x - 40.0) < 0.5
    assert abs(dl.items[0].transform.rotation - 67.5) < 0.5


def test_eval_shape_tween_morphs_matching_paths():
    doc = AnimationDocument.create("morph")
    layer = doc.scenes[0].timeline.add_layer("m")
    kf0 = layer.add_keyframe(0, 10)
    sq = Element.path("M0,0 L40,0 L40,40 L0,40 Z", fill="#00ff00")
    kf0.add_element(sq)
    kf1 = layer.add_keyframe(10, 10)
    sq2 = Element.path("M0,0 L80,0 L80,20 L0,20 Z", fill="#00ff00")
    sq2.id = sq.id
    kf1.add_element(sq2)
    kf0.tween_to(kf1, easing=Easing.named("linear"), kind="shape")
    dl = evaluate(doc, 5)
    pts = dl.items[0].points
    # midpoint: x2 40->80 gives 60, y3 40->20 gives 30
    xs = [x for contour in pts for x, _ in contour]
    ys = [y for contour in pts for _, y in contour]
    assert max(xs) == pytest.approx(60.0, abs=0.5)
    assert max(ys) == pytest.approx(30.0, abs=0.5)


def test_eval_shape_tween_mismatch_is_honest():
    doc = AnimationDocument.create("morph-bad")
    layer = doc.scenes[0].timeline.add_layer("m")
    kf0 = layer.add_keyframe(0, 10)
    tri = Element.path("M0,0 L40,0 L20,40 Z", fill="#00ff00")
    kf0.add_element(tri)
    kf1 = layer.add_keyframe(10, 10)
    sq = Element.path("M0,0 L40,0 L40,40 L0,40 Z", fill="#00ff00")
    sq.id = tri.id
    kf1.add_element(sq)
    kf0.tween_to(kf1, easing=Easing.named("linear"), kind="shape")
    dl = evaluate(doc, 5)
    assert len(dl.items) == 1
    assert any("shape" in n.lower() and "morph" in n.lower()
               for n in dl.notes)


def test_eval_guide_layer_never_in_output():
    doc = AnimationDocument.create("guide")
    layer = doc.scenes[0].timeline.add_layer("g", kind="guide")
    kf = layer.add_keyframe(0, 10)
    kf.add_element(Element.rect(0, 0, 50, 50, fill="#0000ff"))
    dl = evaluate(doc, 3)
    assert dl.items == []


def test_eval_mask_coverage_clips():
    doc = AnimationDocument.create("mask")
    tl = doc.scenes[0].timeline
    mask = tl.add_layer("mask", kind="mask")
    mkf = mask.add_keyframe(0, 10)
    mkf.add_element(Element.rect(0, 0, 40, 40, fill="#ffffff"))
    below = tl.add_layer("below")
    bkf = below.add_keyframe(0, 10)
    bkf.add_element(Element.rect(0, 0, 100, 100, fill="#ff0000"))
    dl = evaluate(doc, 3)
    assert len(dl.items) == 1
    assert dl.items[0].clip is not None


def test_eval_camera_pans_the_stage():
    doc = AnimationDocument.create("cam")
    tl = doc.scenes[0].timeline
    cam = tl.add_layer("cam", kind="camera")
    ckf0 = cam.add_keyframe(0, 12)
    ckf0.camera = {"x": 0.0, "y": 0.0, "zoom": 1.0, "rotation": 0.0}
    ckf1 = cam.add_keyframe(12, 12)
    ckf1.camera = {"x": 100.0, "y": 0.0, "zoom": 1.0, "rotation": 0.0}
    ckf0.tween_to(ckf1, easing=Easing.named("linear"))
    layer = tl.add_layer("ball")
    kf = layer.add_keyframe(0, 24)
    kf.add_element(Element.ellipse(100.0, 60.0, 10.0, 10.0,
                                   fill="#ff0000"))
    dl = evaluate(doc, 6)
    # camera has panned 50 right, so the ball reads 50 left on stage
    pts = [p for contour in dl.items[0].points for p in contour]
    cx = sum(p[0] for p in pts) / len(pts)
    assert abs(cx - 50.0) < 0.5


def test_eval_symbol_instance_loops():
    doc = AnimationDocument.create("sym")
    sym = doc.add_symbol("spinner", kind="graphic")
    sl = sym.timeline.add_layer("s")
    skf0 = sl.add_keyframe(0, 6)
    skf0.add_element(Element.rect(0, 0, 20, 20, fill="#00ff00"))
    skf1 = sl.add_keyframe(6, 6)
    r2 = Element.rect(0, 0, 20, 20, fill="#00ff00")
    r2.id = skf0.elements[0].id
    r2.transform = Transform(rotation=90.0)
    skf1.add_element(r2)
    skf0.tween_to(skf1, easing=Easing.named("linear"))
    layer = doc.scenes[0].timeline.add_layer("main")
    kf = layer.add_keyframe(0, 24)
    inst = Element.symbol(sym.id, Transform(x=50.0, y=50.0))
    kf.add_element(inst)
    dl = evaluate(doc, 18)  # 18 % 12 = 6 -> second keyframe, rotation 90
    assert len(dl.items) == 1
    assert abs(dl.items[0].transform.rotation - 90.0) < 0.5


def test_eval_motion_path_and_orient():
    doc = AnimationDocument.create("path")
    layer = doc.scenes[0].timeline.add_layer("m")
    kf0 = layer.add_keyframe(0, 12)
    ball = Element.ellipse(0.0, 0.0, 8.0, 8.0, fill="#ff0000")
    kf0.add_element(ball)
    kf1 = layer.add_keyframe(12, 12)
    ball2 = Element.ellipse(0.0, 0.0, 8.0, 8.0, fill="#ff0000")
    ball2.id = ball.id
    kf1.add_element(ball2)
    kf0.tween_motion(kf1, path=[(0.0, 0.0), (60.0, -40.0),
                                (120.0, 0.0)], orient=True)
    dl = evaluate(doc, 6)
    assert abs(dl.items[0].transform.x - 60.0) < 1.0
    assert abs(dl.items[0].transform.y - (-40.0)) < 1.0


# ---------------------------------------------------------------------------
# Rendering — real pixels
# ---------------------------------------------------------------------------

def test_render_frame_draws_the_ball(tmp_path):
    doc = _simple_doc()
    img, report = render_frame(doc, 6)
    assert img.size == (200, 120)
    # the ball's centre is path (20, 60) + transform.x 80 = (100, 60)
    px = img.convert("RGB").getpixel((100, 60))
    assert px[0] > 200 and px[1] < 60 and px[2] < 60
    assert report["unrendered"] == []


def test_render_onion_skin_ghosts_neighbours():
    doc = _simple_doc()
    img, _ = render_frame(doc, 6, onion=1)
    assert img.size == (200, 120)
    # ghost of frame 5 and frame 7 also present: red pixels left and right
    rgb = img.convert("RGB")
    left = rgb.getpixel((70, 60))
    right = rgb.getpixel((90, 60))
    assert left[0] > 100 or right[0] > 100


def test_render_names_what_it_does_not_draw():
    doc = AnimationDocument.create("text")
    layer = doc.scenes[0].timeline.add_layer("t")
    kf = layer.add_keyframe(0, 10)
    kf.add_element(Element.bitmap("missing.png", 0, 0, 40, 40))
    img, report = render_frame(doc, 3)
    assert img.size == (640, 360)
    assert any("bitmap" in u.lower() for u in report["unrendered"])


# ---------------------------------------------------------------------------
# Editing — undo/redo, retime, trim, copy/paste
# ---------------------------------------------------------------------------

def test_editor_undo_redo_restores_keyframes():
    doc = _simple_doc()
    ed = AnimationEditor(doc)
    layer = doc.scenes[0].timeline.layers[0]
    assert len(layer.keyframes) == 2
    ed.add_keyframe(layer.id, 24, 12)
    assert len(layer.keyframes) == 3
    ed.undo()
    # Undo restores the document's contents; re-fetch the layer, since
    # the restored tree holds new layer objects.
    layer = ed.doc.scenes[0].timeline.layers[0]
    assert len(layer.keyframes) == 2
    ed.redo()
    layer = ed.doc.scenes[0].timeline.layers[0]
    assert len(layer.keyframes) == 3


def test_editor_retime_and_trim():
    doc = _simple_doc()
    ed = AnimationEditor(doc)
    layer = doc.scenes[0].timeline.layers[0]
    ed.retime_keyframe(layer.id, 12, 18)
    assert layer.keyframes[1].start == 18
    ed.trim_keyframe(layer.id, 18, 6)
    assert layer.keyframes[1].duration == 6
    assert layer.keyframes[1].end() == 24


def test_editor_copy_paste_keyframes():
    doc = _simple_doc()
    ed = AnimationEditor(doc)
    layer = doc.scenes[0].timeline.layers[0]
    ed.copy_keyframes(layer.id, [0])
    ed.paste_keyframes(layer.id, 24)
    assert len(layer.keyframes) == 3
    assert layer.keyframes[2].start == 24
    assert layer.keyframes[2].elements[0].id != \
        layer.keyframes[0].elements[0].id


def test_editor_atomic_edit_restores_on_failure():
    doc = _simple_doc()
    ed = AnimationEditor(doc)
    layer = doc.scenes[0].timeline.layers[0]
    before = doc.to_bytes()
    with pytest.raises(AnimationError), ed.edit():
        ed.add_keyframe(layer.id, 24, 12)
        raise AnimationError("boom")
    assert doc.to_bytes() == before


def test_editor_move_keyframe_refuses_overlap():
    doc = _simple_doc()
    ed = AnimationEditor(doc)
    layer = doc.scenes[0].timeline.layers[0]
    with pytest.raises(AnimationError):
        ed.move_keyframe(layer.id, 12, 6)


# ---------------------------------------------------------------------------
# Library presets
# ---------------------------------------------------------------------------

PRESETS = ("walk-cycle", "bounce", "fade-in", "slide-in", "pulse",
           "typewriter")


@pytest.mark.parametrize("name", PRESETS)
def test_preset_is_a_real_timeline(name):
    doc = build_preset(name)
    assert doc.scenes[0].timeline.frame_count() >= 12
    img, report = render_frame(doc, 0)
    assert img.size[0] > 0
    assert report["unrendered"] == []


def test_preset_search():
    assert "bounce" in search_presets("bounce")
    assert search_presets("walk") == ["walk-cycle"]
    assert list_presets() == sorted(PRESETS)


def test_preset_bounce_squashes_at_the_bottom():
    doc = build_preset("bounce")
    dl = evaluate(doc, 12)
    assert len(dl.items) == 1
    # squash: wider than tall at the bottom of the bounce
    assert dl.items[0].transform.scale_x > 1.0
    assert dl.items[0].transform.scale_y < 1.0


# ---------------------------------------------------------------------------
# Publish — real animated artifacts
# ---------------------------------------------------------------------------

def test_publish_gif_frame_count(tmp_path):
    doc = _simple_doc()
    out = tmp_path / "simple.gif"
    report = publish(doc, "gif", out)
    assert report["frame_count"] == 24
    assert out.is_file() and out.stat().st_size > 0
    from PIL import Image
    with Image.open(out) as im:
        assert im.n_frames > 1  # PIL merges identical held frames


def test_publish_mp4_is_real_video(tmp_path):
    doc = _simple_doc()
    out = tmp_path / "simple.mp4"
    report = publish(doc, "mp4", out)
    assert report["frame_count"] == 24
    assert out.is_file() and out.stat().st_size > 1000
    assert out.read_bytes()[4:8] == b"ftyp"


def test_publish_png_sequence_and_spritesheet(tmp_path):
    doc = _simple_doc()
    seq = publish(doc, "pngseq", tmp_path / "seq")
    assert seq["frame_count"] == 24
    assert len(list((tmp_path / "seq").glob("*.png"))) == 24
    sheet = publish(doc, "spritesheet", tmp_path / "sheet.png")
    assert sheet["frame_count"] == 24
    assert (tmp_path / "sheet.png").is_file()


def test_publish_svg_is_real_vector(tmp_path):
    doc = _simple_doc()
    out = tmp_path / "frame.svg"
    report = publish(doc, "svg", out, frame=6)
    assert report["frame_count"] == 1
    text = out.read_text()
    assert "<svg" in text and "<path" in text


def test_publish_unsupported_format_is_honest(tmp_path):
    doc = _simple_doc()
    with pytest.raises(AnimationError):
        publish(doc, "lottie", tmp_path / "out.json")


# ---------------------------------------------------------------------------
# XFL import / export
# ---------------------------------------------------------------------------

def test_xfl_import_sample():
    doc = import_xfl(_fixture("sample.xfl"))
    assert len(doc.scenes) == 1
    tl = doc.scenes[0].timeline
    assert len(tl.layers) == 2
    assert tl.layers[0].name == "Layer 1"
    assert len(tl.layers[0].keyframes) == 2
    assert tl.layers[0].keyframes[0].elements[0].kind == "shape"


def test_xfl_round_trip_keeps_timeline(tmp_path):
    doc = import_xfl(_fixture("sample.xfl"))
    doc2 = import_xfl(export_xfl(doc, tmp_path / "roundtrip.xfl"))
    tl1 = doc.scenes[0].timeline
    tl2 = doc2.scenes[0].timeline
    assert len(tl2.layers) == len(tl1.layers)
    assert [len(ly.keyframes) for ly in tl2.layers] == \
        [len(ly.keyframes) for ly in tl1.layers]


def test_xfl_import_rejects_bad_xml(tmp_path):
    bad = tmp_path / "bad.xfl"
    bad.mkdir()
    (bad / "DOMDocument.xml").write_text("<broken")
    with pytest.raises(AnimationError):
        import_xfl(bad)


# ---------------------------------------------------------------------------
# AI-client seam — the shape of craft-ai-integrations' ai-client
# ---------------------------------------------------------------------------

def test_ai_client_local_motion_suggestion():
    client = AiClient.standard("anicraft")
    role = Role.suggest_motion()
    status = client.status(role)
    assert status["available"] is True
    assert status["engine"] == "local"
    request = {"prompt": "a ball bouncing happily"}
    plan = client.plan(role, request)
    assert plan["capability"] == "motion.suggest"
    ledger = ConsentLedger()
    assert plan["needs_consent"] is False  # local engine, no disclosure
    job = client.start(plan, ledger)
    result = job.poll()
    assert result["status"] == "done"
    assert result["result"]["preset"] == "bounce"
    assert "quad" in result["result"]["easing"] or \
        "bounce" in result["result"]["easing"]


def test_ai_client_external_provider_is_honest():
    client = AiClient.standard("anicraft")
    role = Role(id="rewrite", label="Rewrite",
                capability="text.complete", inputs=["prompt"])
    status = client.status(role)
    assert status["available"] is False
    assert "provider" in status["reason"].lower()


def test_ai_client_consent_blocks_without_grant():
    client = AiClient.standard("anicraft")
    role = Role(id="cloud-motion", label="Cloud motion",
                capability="motion.suggest-cloud", inputs=["prompt"])
    plan = client.plan(role, {"prompt": "spin"})
    assert plan["needs_consent"] is True
    ledger = ConsentLedger()
    with pytest.raises(AnimationError):
        client.start(plan, ledger)
    ledger.grant(plan["disclosure"])
    job = client.start(plan, ledger)
    assert job.poll()["status"] in ("done", "unavailable")


# ---------------------------------------------------------------------------
# Parity + attribution
# ---------------------------------------------------------------------------

def test_parity_runs_all_cases():
    result = run_animation_parity()
    assert result["passed"] == result["total"]
    assert result["failed"] == []


def test_attribution_verifies():
    result = verify_animation_attribution()
    assert result["failed"] == 0
    assert result["verified"] >= 10


# ---------------------------------------------------------------------------
# Tool-count pin (parent fixes the number; the test names the reason)
# ---------------------------------------------------------------------------

def test_registry_tool_count(self=None):
    from types import SimpleNamespace

    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert registry.get("anicraft") is not None
    # 728 after the materialcraft port (lane 109), 729 after the
    # texturecraft port (lane 108), 730 after the anicraft port
    # (lane 110) added the anicraft tool, 731 after the craft-anything
    # port (lane 111) added the craftanything tool, 732 after the
    # craftlauncher port (lane 112) added the craftlauncher tool,
    # 733 after the craft-libs port (lane 113) added the craftlibs tool,
    # 734 after the craft-fonts port (lane 114) added the craftfonts tool.
    # 735 after the indd-utils port (lane 117) added the inddutils tool.
    # 748 after the auth-plugin wiring added 5 tools (session_forge, session_verify, payload_png_bomb, payload_svg_bomb, payload_keylogger) on top of the bughunter port (lane 121). 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist.
    assert len(registry._tools) == 805
