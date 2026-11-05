"""Scene3D spine — plan §1.5e (owner decision 2026-11-05: both 3D
engine rungs are built behind this registry; the profile runtime
picks). Three layers under test:

* the Scene3D document — D1 idiom: frozen records, JSON round-trip,
  seal/dirty, loud validation (unknown parents, cycles, dangling
  material refs, bad tracks);
* the deterministic frame function — scene state at frame f is a
  pure function of the document (analytic pins: parenting, lookat,
  perspective, eased sampling) plus file goldens
  (``tests/golden/scene3d_goldens.json`` — state, not pixels);
* the renderer registry + wiring — the §1.1e honesty rules applied
  to 3D: capabilities as data, probe with reasons, refusals named.
  With the registry empty (its shipped state) every historical
  refusal fires verbatim; a fake rung proves the serving paths —
  formats, effects classification, MotionProject composition —
  without pretending to be an engine.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nomorals.media.edit_engine.scene3d import (
    ANIMATED_PROPS,
    AnimKey,
    AnimTrack,
    Camera,
    Light,
    Material,
    Mesh,
    RendererCapabilities,
    RendererProbe,
    Rotation,
    Scene3D,
    Scene3DError,
    SceneNode,
    Transform,
    capability_gaps,
    get_renderer,
    register_renderer,
    registered_renderers,
    render_scene,
    renderer_status,
    reset_renderers,
    resolve_renderer,
    scene_requirements,
    scene_state_at,
    scene_state_dict,
    unregister_renderer,
)

GOLDENS = (Path(__file__).parent / "golden" / "scene3d_goldens.json")


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_renderers()
    yield
    reset_renderers()


def sample_scene() -> Scene3D:
    return Scene3D(
        title="sample",
        nodes=[
            SceneNode(name="root", transform=Transform(position=(10, 0, 0))),
            SceneNode(name="crate", parent="root",
                      transform=Transform(position=(1, 2, 3),
                                          rotation=Rotation.from_euler(0, 0, 90)),
                      mesh=Mesh(kind="primitive", primitive="cube",
                                params={"size": 2.0}),
                      material="paint"),
            SceneNode(name="sun", transform=Transform(
                position=(3, 5, 0), rotation=Rotation.look_at((0, 0, 0))),
                light=Light(kind="directional", intensity=2.0)),
        ],
        cameras={"main": Camera(transform=Transform(
            position=(0, 0, 5), rotation=Rotation.look_at((0, 0, 0))))},
        materials={"paint": Material(kind="lambert", color=(0.2, 0.4, 0.9))},
        tracks=[AnimTrack(target="nodes/crate/position",
                          keys=[AnimKey(0, (1, 2, 3)),
                                AnimKey(30, (4, 2, 3), "linear")])],
    )


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_round_trip_dict_and_json(self):
        scene = sample_scene()
        d = scene.to_dict()
        assert d["kind"] == "devon-scene3d" and d["version"] == 1
        assert Scene3D.from_dict(d) == scene
        via_json = Scene3D.from_dict(json.loads(json.dumps(d)))
        assert via_json == scene

    def test_hash_seal_dirty_cycle(self):
        scene = sample_scene()
        # the D1 idiom: dirty means a *recorded* hash no longer
        # matches the content — a fresh document has none recorded
        assert not scene.is_dirty()
        sealed = scene.seal()
        assert sealed.doc_hash == scene.compute_hash()
        assert not sealed.is_dirty()
        assert Scene3D.from_dict(sealed.to_dict()).is_dirty() is False
        import dataclasses
        edited = dataclasses.replace(sealed, title="other")
        assert edited.is_dirty()

    def test_save_load(self, tmp_path):
        scene = sample_scene()
        p = tmp_path / "scene.json"
        scene.save(p)
        assert Scene3D.load(p) == scene
        bad = tmp_path / "bad.json"
        bad.write_text('{"kind": "devon-scene", "version": 1}',
                       encoding="utf-8")
        with pytest.raises(Scene3DError, match="not a scene3d document"):
            Scene3D.load(bad)

    def test_from_dict_rejects_foreign(self):
        with pytest.raises(Scene3DError, match="not a scene3d document"):
            Scene3D.from_dict({"kind": "devon-motion", "version": 1})
        with pytest.raises(Scene3DError, match="version"):
            Scene3D.from_dict({"kind": "devon-scene3d", "version": 99})

    def test_validation_battery(self):
        base = dict(cameras={"c": Camera()})
        with pytest.raises(Scene3DError, match="duplicate scene node"):
            Scene3D(nodes=[SceneNode(name="a"), SceneNode(name="a")], **base)
        with pytest.raises(Scene3DError, match="unknown node"):
            Scene3D(nodes=[SceneNode(name="a", parent="ghost")], **base)
        with pytest.raises(Scene3DError, match="scene graph cycle"):
            Scene3D(nodes=[SceneNode(name="a", parent="b"),
                           SceneNode(name="b", parent="a")], **base)
        with pytest.raises(Scene3DError, match="scene graph cycle"):
            Scene3D(nodes=[SceneNode(name="a", parent="a")], **base)
        with pytest.raises(Scene3DError, match="unknown material"):
            Scene3D(nodes=[SceneNode(
                name="a", mesh=Mesh(kind="primitive", primitive="cube"),
                material="ghost")], **base)
        with pytest.raises(Scene3DError, match="active camera"):
            Scene3D(cameras={"c": Camera()}, active_camera="ghost")
        with pytest.raises(Scene3DError, match="both a mesh and a light"):
            Scene3D(nodes=[SceneNode(
                name="a", mesh=Mesh(kind="primitive", primitive="cube"),
                light=Light(kind="ambient"))], **base)
        with pytest.raises(Scene3DError, match="carries no mesh"):
            Scene3D(nodes=[SceneNode(name="a", material="m")],
                    materials={"m": Material()}, **base)

    def test_component_validation(self):
        with pytest.raises(Scene3DError, match="fov"):
            Camera(fov=190.0)
        with pytest.raises(Scene3DError, match="near < far"):
            Camera(near=10.0, far=5.0)
        with pytest.raises(Scene3DError, match="look-at target"):
            Rotation.look_at((1, 1, 1)).to_quat((1, 1, 1))
        with pytest.raises(Scene3DError, match="zero-length"):
            Rotation(quat=(0, 0, 0, 0))
        with pytest.raises(Scene3DError, match="opacity"):
            Material(opacity=1.5)
        with pytest.raises(Scene3DError, match="spot_angle"):
            Light(kind="spot", spot_angle=200.0)
        with pytest.raises(Scene3DError, match="primitive must be one of"):
            Mesh(kind="primitive", primitive="torus")
        with pytest.raises(Scene3DError, match="out of range"):
            Mesh(kind="explicit", vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                 faces=[(0, 1, 7)])
        with pytest.raises(Scene3DError, match="normals"):
            Mesh(kind="explicit",
                 vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                 faces=[(0, 1, 2)],
                 normals=[(0, 0, 1), (0, 1, 0)])

    def test_track_validation(self):
        cam = {"c": Camera()}
        with pytest.raises(Scene3DError, match="not animatable"):
            Scene3D(cameras=cam, tracks=[AnimTrack(
                target="cameras/c/zoom",
                keys=[AnimKey(0, (1.0,))])])
        with pytest.raises(Scene3DError, match="no node"):
            Scene3D(cameras=cam, tracks=[AnimTrack(
                target="nodes/ghost/position",
                keys=[AnimKey(0, (0, 0, 0))])])
        with pytest.raises(Scene3DError, match="strictly increasing"):
            Scene3D(cameras=cam, tracks=[AnimTrack(
                target="cameras/c/fov",
                keys=[AnimKey(5, (50.0,)), AnimKey(5, (60.0,))])])
        with pytest.raises(Scene3DError, match="keys carry"):
            Scene3D(cameras=cam, tracks=[AnimTrack(
                target="cameras/c/fov", keys=[AnimKey(0, (1.0, 2.0))])])
        with pytest.raises(Scene3DError, match="duplicate animation track"):
            Scene3D(cameras=cam, tracks=[
                AnimTrack(target="cameras/c/fov",
                          keys=[AnimKey(0, (50.0,))]),
                AnimTrack(target="cameras/c/fov",
                          keys=[AnimKey(0, (60.0,))])])
        with pytest.raises(Scene3DError, match="euler degrees"):
            Scene3D(nodes=[SceneNode(
                name="q", transform=Transform(
                    rotation=Rotation.from_quaternion(
                        (0, 0.70710678, 0, 0.70710678))))],
                cameras=cam, tracks=[AnimTrack(
                    target="nodes/q/rotation",
                    keys=[AnimKey(0, (0, 0, 0))])])
        with pytest.raises(Scene3DError, match="carries no light"):
            Scene3D(nodes=[SceneNode(
                name="m", mesh=Mesh(kind="primitive", primitive="cube"))],
                cameras=cam, tracks=[AnimTrack(
                    target="nodes/m/intensity",
                    keys=[AnimKey(0, (1.0,))])])
        with pytest.raises(Scene3DError, match="unknown easing"):
            AnimKey(0, (1.0,), easing="wobble")


# ---------------------------------------------------------------------------
# the frame function
# ---------------------------------------------------------------------------

def _approx(a, b, tol=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_approx(a[k], b[k], tol)
                                            for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_approx(x, y, tol)
                                        for x, y in zip(a, b, strict=True))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)
    return a == b


class TestFrameFunction:
    def test_parenting_and_animation(self):
        scene = sample_scene()
        state = scene_state_at(scene, 15)
        node = state.nodes["crate"]
        # root at x=10; crate lerps x 1→4 over 0..30 → 2.5 at 15
        assert node["world_position"] == pytest.approx((12.5, 2.0, 3.0))
        assert node["kind"] == "mesh"
        assert node["material"]["kind"] == "lambert"
        assert node["visible"] is True
        root = state.nodes["root"]
        assert root["world_position"] == pytest.approx((10.0, 0.0, 0.0))

    def test_lookat_camera_view(self):
        scene = sample_scene()
        state = scene_state_at(scene, 0)
        view = state.camera["view_matrix"]  # flat row-major 4x4
        # the world origin sits 5 units down the camera's -Z:
        # the view matrix's translation column
        assert (view[3], view[7], view[11]) == \
            pytest.approx((0.0, 0.0, -5.0))
        assert state.camera["name"] == "main"
        assert state.camera["aspect"] == pytest.approx(1920 / 1080)

    def test_perspective_near_far(self):
        cam = Camera(fov=90.0, near=1.0, far=11.0, aspect=1.0)
        proj = scene_state_at(Scene3D(cameras={"c": cam}), 0
                              ).camera["projection_matrix"]  # flat 4x4
        # OpenGL-style: view z=-near → NDC -1; z=-far → NDC +1
        def ndc_z(z_view):
            clip_z = proj[10] * z_view + proj[11]
            clip_w = proj[14] * z_view
            return clip_z / clip_w
        assert ndc_z(-1.0) == pytest.approx(-1.0)
        assert ndc_z(-11.0) == pytest.approx(1.0)
        # fov 90° → focal 1/tan(45°) = 1
        assert proj[0] == pytest.approx(1.0)
        assert proj[5] == pytest.approx(1.0)

    def test_directional_light_direction(self):
        state = scene_state_at(sample_scene(), 0)
        sun = state.nodes["sun"]["light"]
        # from (3,5,0) aiming at the origin, normalized -(pos)
        n = math.sqrt(34.0)
        assert sun["direction"] == pytest.approx((-3 / n, -5 / n, 0.0))
        assert sun["intensity"] == 2.0

    def test_eased_sampling_and_holds(self):
        scene = Scene3D(
            cameras={"c": Camera()},
            tracks=[AnimTrack(target="cameras/c/fov", keys=[
                AnimKey(10, (40.0,)), AnimKey(20, (80.0,), "smooth")])])
        assert scene_state_at(scene, 0).camera["fov"] == 40.0
        assert scene_state_at(scene, 99).camera["fov"] == 80.0
        mid = scene_state_at(scene, 15).camera["fov"]
        assert mid == pytest.approx(60.0)  # smooth(0.5) = 0.5

    def test_visibility_inherits(self):
        scene = Scene3D(
            nodes=[SceneNode(name="p", visible=False),
                   SceneNode(name="c", parent="p",
                             mesh=Mesh(kind="primitive",
                                       primitive="plane"))],
            cameras={"cam": Camera()})
        state = scene_state_at(scene, 0)
        assert state.nodes["c"]["visible"] is False  # inherited

    def test_seconds_ride_the_frame_grid(self):
        scene = sample_scene()
        state = scene_state_at(scene, 45)
        assert state.seconds == pytest.approx(1.5)  # 30 fps default
        assert state.frame == 45

    def test_determinism_double_evaluation(self):
        scene = sample_scene()
        a = scene_state_at(scene, 17)
        b = scene_state_at(scene, 17)
        c = scene_state_at(Scene3D.from_dict(scene.to_dict()), 17)
        assert a == b == c
        assert scene_state_dict(scene, 17) == a.to_dict()
        from nomorals.media.edit_engine.scene3d import _state_hash
        assert _state_hash(a) == _state_hash(b) == _state_hash(c)

    def test_goldens(self):
        data = json.loads(GOLDENS.read_text(encoding="utf-8"))
        assert sorted(data["scenes"]) == ["nested_doll", "orbit_crate",
                                          "still_minimal"]
        for name, entry in data["scenes"].items():
            scene = Scene3D.from_dict(entry["document"])
            for frame_s, expected in entry["states"].items():
                got = scene_state_at(scene, int(frame_s)).to_dict()
                assert _approx(got, expected), (
                    f"golden drift: {name} @ frame {frame_s}")


# ---------------------------------------------------------------------------
# the renderer registry
# ---------------------------------------------------------------------------

def full_caps(**overrides) -> RendererCapabilities:
    caps = dict(
        surfaces=("scene3d", "motion_camera", "format3d"),
        node_kinds=("group", "mesh", "light"),
        mesh_kinds=("primitive", "explicit"),
        primitives=("cube", "sphere", "plane", "cylinder", "cone"),
        material_kinds=("flat", "lambert", "phong"),
        light_kinds=("ambient", "directional", "point", "spot"),
        animated_props=("position", "rotation", "scale", "intensity",
                        "fov", "near", "far", "color", "opacity"),
        max_width=7680, max_height=4320, deterministic=True)
    caps.update(overrides)
    return RendererCapabilities(**caps)


class FakeRenderer:
    """A registry citizen, not an engine: writes placeholder frame
    files in the pipeline's image-sequence shape so the contract —
    resolution, records, state hashes — is exercised end to end."""

    def __init__(self, name="fake3d", caps=None, *,
                 available=True, reason=""):
        self._name = name
        self._caps = caps if caps is not None else full_caps()
        self._probe = RendererProbe(available, reason)
        self.probed = False
        self.rendered: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> RendererCapabilities:
        return self._caps

    def probe(self) -> RendererProbe:
        self.probed = True
        return self._probe

    def render_frames(self, scene, *, frames, workdir):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in frames:
            p = workdir / f"f_{f:04d}.png"
            p.write_bytes(b"fake-frame")
            paths.append(str(p))
            self.rendered.append(f)
        return {"frames_dir": str(workdir), "pattern": "f_%04d.png",
                "framerate": float(scene.output.get("fps", 30)),
                "frames": paths,
                "width": scene.output["width"],
                "height": scene.output["height"]}


class TestRegistry:
    def test_empty_registry_refuses_by_name(self):
        scene = sample_scene()
        with pytest.raises(Scene3DError, match="no 3D renderer is "
                                               "registered"):
            resolve_renderer(scene)
        with pytest.raises(Scene3DError, match="no 3D renderer is "
                                               "registered"):
            render_scene(scene, frames=[0])
        assert renderer_status() == []

    def test_requirements_read_the_document(self):
        req = scene_requirements(sample_scene())
        assert "scene3d" in req["surfaces"]
        assert set(req["node_kinds"]) == {"group", "mesh", "light"}
        assert set(req["primitives"]) == {"cube"}
        assert set(req["material_kinds"]) == {"lambert"}
        assert set(req["animated_props"]) == {"position"}
        assert req["width"] == 1920 and req["height"] == 1080

    def test_resolution_and_probe_refusals_recorded(self):
        down = FakeRenderer("downrung", available=False,
                            reason="binary missing")
        register_renderer(down)
        register_renderer(FakeRenderer())
        renderer, record = resolve_renderer(sample_scene())
        assert renderer.name == "fake3d"
        assert down.probed is True
        assert record["refused"] == [
            {"renderer": "downrung",
             "probe": {"available": False, "reason": "binary missing"},
             "reason": "binary missing"}]
        assert [e["name"] for e in renderer_status()] == [
            "downrung", "fake3d"]

    def test_capability_gaps_named(self):
        weak = FakeRenderer("weak", full_caps(primitives=("sphere",),
                                              animated_props=()))
        gaps = capability_gaps(weak.capabilities(),
                               scene_requirements(sample_scene()))
        assert any(g.startswith("primitives:") and "cube" in g
                   for g in gaps)
        assert any(g.startswith("animated_props:") and "position" in g
                   for g in gaps)
        register_renderer(weak)
        with pytest.raises(Scene3DError) as exc:
            resolve_renderer(sample_scene())
        assert "weak" in str(exc.value) and "cube" in str(exc.value)

    def test_named_resolution_and_registration_rules(self):
        register_renderer(FakeRenderer())
        renderer, _rec = resolve_renderer(sample_scene(), name="fake3d")
        assert renderer.name == "fake3d"
        with pytest.raises(Scene3DError, match="no scene renderer"):
            resolve_renderer(sample_scene(), name="blenderish")
        with pytest.raises(Scene3DError, match="already registered"):
            register_renderer(FakeRenderer())
        assert get_renderer("fake3d") is not None
        assert unregister_renderer("fake3d") is True
        assert unregister_renderer("fake3d") is False
        assert registered_renderers() == []

    def test_render_scene_record_and_state_hashes(self, tmp_path):
        fake = FakeRenderer()
        register_renderer(fake)
        scene = sample_scene()
        record = render_scene(scene, frames=[0, 15], workdir=tmp_path)
        assert record["renderer"] == "fake3d"
        assert record["pattern"] == "f_%04d.png"
        assert len(record["frames"]) == 2
        assert fake.rendered == [0, 15]
        from nomorals.media.edit_engine.scene3d import _state_hash
        expect = {str(f): _state_hash(scene_state_at(scene, f))
                  for f in (0, 15)}
        assert record["state_hashes"] == expect
        assert record["scene_hash"] == scene.compute_hash()
        assert record["resolution"]
        with pytest.raises(Scene3DError, match="needs at least one frame"):
            render_scene(scene, frames=[])

    def test_render_contract_violation_refused(self, tmp_path):
        class Broken(FakeRenderer):
            def render_frames(self, scene, *, frames, workdir):
                return {"note": "no frames here"}

        register_renderer(Broken("broken"))
        with pytest.raises(Scene3DError, match="record without"):
            render_scene(sample_scene(), frames=[0], workdir=tmp_path)

    def test_animated_props_contract_matches_frame_function(self):
        # the registry's vocabulary is the frame function's vocabulary
        assert set(ANIMATED_PROPS) == {"nodes", "cameras", "materials"}
        assert ANIMATED_PROPS["cameras"] == {
            "position": 3, "fov": 1, "near": 1, "far": 1}


# ---------------------------------------------------------------------------
# wiring — the three historical refusal sites, registry-resolved
# ---------------------------------------------------------------------------

class TestFormatWiring:
    FORMAT_REFUSAL = (
        "format 'camera3d' declares renderer 'camera3d', which is "
        "unavailable: no 3D rung has been chosen (§1.5e/§6.3 — "
        "Blender-class external engine vs a Rust rung is the owner's "
        "decision). The format resolves as data; it does not render.")

    def _spec(self):
        from nomorals.media.edit_engine.spec import EditSpec
        return EditSpec(clips=[{"path": "a.mp4", "duration": 1.0}])

    def test_no_rung_refusal_verbatim(self):
        from nomorals.media.edit_engine import FormatError, apply_format
        with pytest.raises(FormatError) as exc:
            apply_format(self._spec(), "camera3d")
        assert str(exc.value) == self.FORMAT_REFUSAL

    def test_serving_rung_applies_with_record(self):
        from nomorals.media.edit_engine import apply_format
        register_renderer(FakeRenderer())
        spec, record = apply_format(self._spec(), "camera3d")
        assert record["renderer3d"]["renderer"] == "fake3d"
        assert record["canvas"] == {"width": 1920, "height": 1080,
                                    "fps": 30}

    def test_non_serving_rung_refuses_naming_the_gap(self):
        from nomorals.media.edit_engine import FormatError, apply_format
        register_renderer(FakeRenderer(
            "scenersonly", full_caps(surfaces=("scene3d",))))
        with pytest.raises(FormatError) as exc:
            apply_format(self._spec(), "camera3d")
        assert "scenersonly" in str(exc.value)
        assert "format3d" in str(exc.value)


class TestEffectsWiring:
    def test_no_rung_reason_verbatim(self):
        from nomorals.media.edit_engine.effects import (
            DEFERRED_EFFECTS,
            _effect_filter,
            deferred_effect_reason,
        )
        assert deferred_effect_reason("camera3d") == \
            DEFERRED_EFFECTS["camera3d"]
        assert "the 3D rung is the owner's unmade choice" in \
            DEFERRED_EFFECTS["camera3d"]
        with pytest.raises(Exception, match="deferred"):
            _effect_filter("camera3d", {}, {})

    def test_serving_rung_reason_names_the_rung(self):
        from nomorals.media.edit_engine.effects import _effect_filter, deferred_effect_reason
        from nomorals.media_edit.videos import MediaEditError
        register_renderer(FakeRenderer())
        reason = deferred_effect_reason("camera3d")
        assert "fake3d" in reason and "not a 2D chain effect" in reason
        with pytest.raises(MediaEditError, match="fake3d"):
            _effect_filter("camera3d", {}, {})

    def test_classify_track_reason_follows_the_registry(self):
        from nomorals.media.edit_engine.keyframes import Keyframe, KeyframeTrack, classify_track
        track = KeyframeTrack("c1", "camera3d", "zoom",
                              [Keyframe(0, 1.0)])
        with pytest.raises(ValueError, match="unmade choice"):
            classify_track(track)
        register_renderer(FakeRenderer())
        with pytest.raises(ValueError, match="fake3d"):
            classify_track(track)


def _motion_project_with_camera3d(tmp_path, props):
    from PIL import Image

    from nomorals.media.motion_studio.project import MotionProject, MotionSegment
    img = tmp_path / "still.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img)
    seg = MotionSegment(uid="hero", kind="image", src=str(img),
                        start=0.0, duration_s=2.0,
                        properties=dict(props))
    return MotionProject(title="cam test", segments=[seg],
                         output={"width": 320, "height": 240, "fps": 30})


class TestMotionWiring:
    VERBATIM_TAIL = ("— the 3D rung is not chosen (plan §1.5e: the "
                     "owner picks the engine; the document carries "
                     "these paths but no renderer fakes them)")

    def test_no_rung_refusal_verbatim(self, tmp_path):
        from nomorals.media.motion_studio.project import MotionStudioError
        proj = _motion_project_with_camera3d(
            tmp_path, {"camera3d/fov": 42.0})
        with pytest.raises(MotionStudioError) as exc:
            proj.to_document()
        assert str(exc.value) == \
            f"segment @hero: camera3d/fov {self.VERBATIM_TAIL}"

    def test_serving_rung_composes_the_camera(self, tmp_path):
        from nomorals.media.motion_studio.project import PropertyTrack
        register_renderer(FakeRenderer())
        proj = _motion_project_with_camera3d(tmp_path, {
            "camera3d/fov": PropertyTrack(
                "camera3d/fov",
                keys=[{"frame": 0, "value": 40.0},
                      {"frame": 60, "value": 70.0}]),
            "camera3d/position/x": 1.0,
            "camera3d/position/z": 6.0,
            "camera3d/lookat/y": 0.5,
        })
        _doc, record = proj.to_document()
        rec = record["segments"][0]
        assert "scene3d" in rec
        comp = rec["scene3d"]
        assert comp["rung"] == "fake3d"
        assert comp["camera"] == "hero-camera"
        scene = Scene3D.from_dict(comp["document"])
        cam = scene.cameras["hero-camera"]
        assert cam.transform.position == (1.0, 0.0, 6.0)
        assert cam.transform.rotation.kind == "lookat"
        assert cam.transform.rotation.target == (1.0, 0.5, 5.0)
        fov_track = [t for t in scene.tracks
                     if t.target == "cameras/hero-camera/fov"]
        assert len(fov_track) == 1
        assert scene_state_at(scene, 30).camera["fov"] == \
            pytest.approx(55.0)
        stages = [r["stage"] for r in record["rungs"]]
        assert "motion:hero:camera3d" in stages

    def test_unserved_prop_refuses_verbatim(self, tmp_path):
        from nomorals.media.motion_studio.project import MotionStudioError, PropertyTrack
        register_renderer(FakeRenderer(
            caps=full_caps(animated_props=("position",))))  # no fov
        proj = _motion_project_with_camera3d(tmp_path, {
            "camera3d/fov": PropertyTrack(
                "camera3d/fov",
                keys=[{"frame": 0, "value": 40.0}])})
        with pytest.raises(MotionStudioError, match="3D rung is not "
                                                    "chosen"):
            proj.to_document()

    def test_unmapped_path_named_even_with_a_rung(self, tmp_path):
        from nomorals.media.motion_studio.project import MotionStudioError
        register_renderer(FakeRenderer())
        proj = _motion_project_with_camera3d(
            tmp_path, {"camera3d/bokeh": 1.0})
        with pytest.raises(MotionStudioError, match="no composition "
                                                    "maps"):
            proj.to_document()
