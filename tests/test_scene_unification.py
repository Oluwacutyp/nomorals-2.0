"""Echelon Wave 1 · §1.5a acceptance: one canonical Scene.

The four legacy scene vocabularies (edit_engine spec dicts,
contentops SceneVisual, videogen chaining dicts, director Shot)
converge on :class:`nomorals.media.edit_engine.Scene`:

* conversion tests per vocabulary — every legacy field has a named
  destination (canonical field, design, or extras); nothing is
  silently dropped;
* round-trips per vocabulary where the vocabulary is complete;
* one fixture production expressed in all four vocabularies renders
  to the same EditSpec clips.
"""

import pytest

from nomorals.media.contentops.niches.base import SceneVisual
from nomorals.media.director.breakdown import Shot
from nomorals.media.director.design import CameraSpec, DesignSpec
from nomorals.media.edit_engine import Scene, scenes_to_edit_spec
from nomorals.media.edit_engine.spec import EditSpec
from nomorals.media_edit import MediaEditError


def _clip_sig(spec):
    """The render-relevant signature of a spec's clips."""
    return [(c.path, c.end,
             [(e.name, e.params) for e in c.effects])
            for c in spec.clips]


# ── vocabulary 1: edit_engine spec scene dicts ────────────────────────

def test_spec_scene_round_trip():
    d = {"image": "a.png", "duration": 3.5, "effect": "kenburns",
         "zoom_direction": "out"}
    s = Scene.from_spec_scene(d)
    assert s.kind == "still" and s.source == "a.png"
    assert s.duration_s == 3.5
    assert s.camera["move"] == "zoom_out"
    assert s.to_spec_scene() == d


def test_spec_scene_non_zoom_effect_is_design_data_not_dropped():
    d = {"image": "a.png", "duration": 2.0, "effect": "static",
         "zoom_direction": "in"}
    s = Scene.from_spec_scene(d)
    assert s.design["effect"] == "static"
    assert s.to_spec_scene() == d
    # and the effect still lands on the rendered clip
    clip = s.to_clip()
    assert [(e.name, e.params) for e in clip.effects] == [("static", {})]


def test_spec_scenes_still_build_specs_like_before():
    spec = EditSpec(scenes=[{"image": "a.png", "duration": 3.0,
                             "effect": "kenburns",
                             "zoom_direction": "in"}])
    assert _clip_sig(spec) == [
        ("a.png", 3.0, [("ken_burns", {"direction": "in"})])]


# ── vocabulary 2: contentops SceneVisual ──────────────────────────────

def test_scene_visual_round_trip():
    sv = SceneVisual(index=3, imggen_prompt="a lighthouse at dusk",
                     motion="zoom-out-slow", duration_s=6.0,
                     transition="crossfade", caption_hook="dusk")
    s = sv.scene()
    assert s.kind == "generated"           # a plan precedes generation
    assert s.prompt == "a lighthouse at dusk"
    assert s.camera["move"] == "zoom_out"
    assert s.transition == {"kind": "crossfade"}
    assert s.extras["index"] == 3
    assert s.extras["caption_hook"] == "dusk"
    back = s.to_scene_visual()
    assert back == sv


def test_scene_visual_unknown_motion_kept_verbatim():
    sv = SceneVisual(index=0, imggen_prompt="x", motion="dolly-left-fast")
    s = Scene.from_scene_visual(sv)
    assert s.design["motion"] == "dolly-left-fast"
    assert s.to_scene_visual().motion == "dolly-left-fast"


# ── vocabulary 3: videogen chaining dicts ─────────────────────────────

def test_chain_scene_round_trip_with_selection():
    d = {"prompt": "rain over Lagos", "mode": "i2v",
         "image": "seed.png", "duration_s": 7.0, "seed": 42,
         "backend": "ltx", "model": "ltx-2", "lora": [{"name": "n"}]}
    s = Scene.from_chain_scene(d, index=1)
    assert s.kind == "generated"
    assert s.source == "seed.png" and s.seed == 42
    # selection provenance rides in extras, restored on the way back
    assert s.extras["chain_backend"] == "ltx"
    assert s.to_chain_scene() == d


def test_chain_scene_text_and_bare_string():
    s = Scene.from_chain_scene({"mode": "text", "text": "CHAPTER ONE",
                                "duration_s": 3.0})
    assert s.kind == "text"
    assert s.to_chain_scene()["text"] == "CHAPTER ONE"
    bare = Scene.from_chain_scene("a prompt")
    assert bare.kind == "generated" and bare.duration_s == 5.0
    assert bare.to_chain_scene()["mode"] == "t2v"


def test_chain_scenes_accepts_canonical_scenes():
    # chain_scenes projects Scene inputs at the boundary — the
    # projection is what this asserts (the chain itself needs real
    # backends and is covered by test_videogen).
    s = Scene(kind="still", source="a.png", duration_s=4.0,
              prompt="a still")
    assert s.to_chain_scene() == {"prompt": "a still", "mode": "image",
                                   "duration_s": 4.0, "image": "a.png"}


# ── vocabulary 4: director Shot ───────────────────────────────────────

def _shot(**kw):
    base = dict(id="2.3", scene_number=2, index=3, kind="dialogue",
                description="Ada turns to the window",
                characters=["Ada"], speaker="Ada",
                dialogue="We leave at dawn.",
                performance_note="(quietly)", duration_s=5.0,
                design=DesignSpec(camera=CameraSpec(
                    shot_size="close-up", angle="low", movement="push-in",
                    lens="85mm")))
    base.update(kw)
    return Shot(**base)


def test_shot_maps_to_scene_with_production_provenance():
    shot = _shot(status="generated", clip_path="clips/2_3.mp4",
                 engine="ltx", qa={"sharpness": 0.9}, regen_count=1)
    s = shot.scene()
    assert s.kind == "clip" and s.source == "clips/2_3.mp4"
    assert s.prompt == "Ada turns to the window"
    assert s.duration_s == 5.0
    # the design spec's camera becomes the Scene camera
    assert s.camera["shot_size"] == "close-up"
    assert s.camera["angle"] == "low"
    assert s.camera["movement"] == "push-in"
    assert s.camera["move"] == "zoom_in"   # push-in, in engine terms
    assert s.camera["lens"] == "85mm"
    # the full design rides as the design ref
    assert s.design["design_spec"]["camera"]["shot_size"] == "close-up"
    # production state: provenance in extras, NOT scene structure
    prov = s.extras["shot"]
    assert prov["id"] == "2.3" and prov["status"] == "generated"
    assert prov["dialogue"] == "We leave at dawn."
    assert prov["speaker"] == "Ada"
    assert prov["qa"] == {"sharpness": 0.9}
    assert prov["regen_count"] == 1
    assert not hasattr(s, "status") and not hasattr(s, "qa")


def test_unshot_shot_is_a_generated_scene():
    s = _shot().scene()
    assert s.kind == "generated" and s.source == ""


# ── the acceptance fixture: one production, four vocabularies ─────────

def _render_sig(spec):
    """Clip signature at RENDER semantics: a ken_burns effect's
    effective direction is its param or the engine's default ("in" —
    effects._fx_ken_burns), so a vocabulary that leaves direction
    unspecified and one that states it render identically."""
    out = []
    for c in spec.clips:
        fx = [(e.name, e.params.get("direction", "in")
               if e.name == "ken_burns" else e.params)
              for e in c.effects]
        out.append((c.path, c.end, fx))
    return out


def test_one_production_four_vocabularies_same_edit_spec():
    # one still-with-zoom beat of a production, expressed four ways
    spec_dict = {"image": "frames/lighthouse.png", "duration": 4.0,
                 "effect": "kenburns", "zoom_direction": "in"}
    from_spec = Scene.from_spec_scene(spec_dict)

    visual = SceneVisual(index=0, imggen_prompt="lighthouse",
                         motion="zoom-in-slow", duration_s=4.0)
    from_visual = visual.scene()
    from_visual.source = "frames/lighthouse.png"  # the generated still

    chained = Scene.from_chain_scene(
        {"prompt": "lighthouse", "mode": "image",
         "image": "frames/lighthouse.png", "duration_s": 4.0})
    # the chaining vocabulary carries no direction — the chain renders
    # image scenes through kenburns "auto"; the Scene records the
    # family + the chain directive, inventing no direction.
    assert chained.camera["move"] == "ken_burns"
    assert chained.design["chain_move"] == "auto"

    shot = _shot(description="lighthouse", duration_s=4.0,
                 clip_path="frames/lighthouse.png",
                 design=DesignSpec(camera=CameraSpec(movement="zoom-in")))
    from_shot = shot.scene()
    assert from_shot.camera["move"] == "zoom_in"  # movement translated

    expected = [("frames/lighthouse.png", 4.0, [("ken_burns", "in")])]
    for scene in (from_spec, from_visual, chained, from_shot):
        assert _render_sig(scenes_to_edit_spec([scene])) == expected
    # and the engine's own legacy path agrees
    assert _render_sig(EditSpec(scenes=[spec_dict])) == expected


def test_scenes_to_edit_spec_sequences_and_text_layers():
    scenes = [
        Scene(kind="still", source="a.png", duration_s=2.0),
        Scene(kind="text", prompt="CHAPTER ONE", duration_s=3.0),
        Scene(kind="clip", source="b.mp4", duration_s=4.0),
    ]
    spec = scenes_to_edit_spec(scenes)
    assert [(c.path, c.at, c.end) for c in spec.clips] == [
        ("a.png", 0.0, 2.0), ("b.mp4", 5.0, 4.0)]
    assert [(t.text, t.start, t.end) for t in spec.text_layers] == [
        ("CHAPTER ONE", 2.0, 5.0)]


def test_scene_dict_round_trip_and_validation():
    s = Scene(kind="generated", prompt="p", duration_s=3.0, seed=7,
              camera={"move": "zoom_in"}, design={"effect": "glow"},
              extras={"origin": "test"})
    assert Scene.from_dict(s.to_dict()) == s
    with pytest.raises(MediaEditError, match="kind"):
        Scene(kind="hologram", duration_s=1.0)
    with pytest.raises(MediaEditError, match="duration"):
        Scene(kind="clip", source="x.mp4", duration_s=0)
