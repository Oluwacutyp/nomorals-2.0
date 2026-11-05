"""The native Rust 3D rung (§1.5e engine 1 of 2) — contract side.

Three layers under test:

* the provisioning ladder (Devon's self-provisioning doctrine):
  binary present → available; absent + no toolchain + provisioning
  disabled → unavailable *by name*; the real box state → available;
* the registry contract: capabilities as data covering the core's
  vocabularies, resolution through `resolve_renderer`, the
  `render_scene` record (state hashes tie pixels to the frame
  function);
* golden pixels — analytically checkable expectations, not
  snapshots: exact flat colors, occlusion order, a Lambert shade
  computed by hand, the sRGB background, a camera move shifting
  the projection where the matrices predict, a Phong highlight
  out-brightening its Lambert twin, animation moving a cube, and
  the determinism contract (same scene, same bytes, twice).

The Rust crate's own unit tests (cargo test) pin the encoder, the
JSON reader, tessellation and the same Lambert/occlusion math at
the binary level; these tests pin the Python↔binary seam.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from nomorals.media.edit_engine import render3d_rust
from nomorals.media.edit_engine.scene3d import (
    AnimKey,
    AnimTrack,
    Camera,
    Light,
    Material,
    Mesh,
    Rotation,
    Scene3D,
    SceneNode,
    Transform,
    capability_gaps,
    registered_renderers,
    render_scene,
    renderer_status,
    reset_renderers,
    resolve_renderer,
    scene_requirements,
    scene_state_at,
    unregister_renderer,
)


@pytest.fixture(autouse=True)
def _rust_registered():
    render3d_rust.register_rust_renderer()
    yield
    # leave the registry as the package import left it
    reset_renderers()
    render3d_rust.register_rust_renderer()


def _flat(color, **kw):
    return Material(kind="flat", color=color, **kw)


def _camera(pos=(0, 0, 5), target=(0, 0, 0), **kw):
    kw.setdefault("fov", 90.0)
    kw.setdefault("near", 1.0)
    kw.setdefault("far", 11.0)
    kw.setdefault("aspect", 1.0)
    return Camera(
        transform=Transform(position=pos, rotation=Rotation.look_at(target)),
        **kw)


def _scene(nodes, *, materials=None, tracks=None, size=64,
           background=(0.0, 0.0, 0.0), cameras=None):
    return Scene3D(
        nodes=nodes,
        cameras=cameras or {"main": _camera()},
        active_camera="main",
        materials=materials or {},
        tracks=tracks or [],
        background=background,
        output={"width": size, "height": size, "fps": 30},
    )


def _render(scene, frames, workdir):
    renderer, _record = resolve_renderer(scene, name="rust3d")
    return renderer.render_frames(scene, frames=frames, workdir=workdir)


def _px(path, x, y):
    with Image.open(path) as img:
        return img.convert("RGB").getpixel((x, y))


# ---------------------------------------------------------------------------
# provisioning ladder
# ---------------------------------------------------------------------------

class TestProvisioning:
    def test_real_binary_probes_available(self):
        probe = render3d_rust.RustSceneRenderer().probe()
        assert probe.available, probe.reason
        status = render3d_rust.provisioning_status()
        assert status["binary_present"] is True
        assert status["crate_present"] is True

    def test_missing_source_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NM_RENDER3D_BIN", str(tmp_path / "no-bin"))
        monkeypatch.setenv("NM_RENDER3D_TOOLS_DIR", str(tmp_path / "tools"))
        monkeypatch.setenv("NM_RENDER3D_SOURCE_DIR", str(tmp_path / "no-src"))
        monkeypatch.setenv("NM_RENDER3D_NO_PROVISION", "1")
        record = render3d_rust.ensure_binary()
        assert record["status"] == "unavailable"
        assert "crate source not found" in record["reason"]

    def test_missing_toolchain_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NM_RENDER3D_BIN", str(tmp_path / "no-bin"))
        monkeypatch.setenv("NM_RENDER3D_TOOLS_DIR", str(tmp_path / "tools"))
        monkeypatch.setenv("NM_RENDER3D_NO_PROVISION", "1")
        monkeypatch.delenv("CARGO_HOME", raising=False)
        monkeypatch.delenv("RUSTUP_HOME", raising=False)
        record = render3d_rust.ensure_binary()
        assert record["status"] == "unavailable"
        assert "cargo is absent" in record["reason"]
        probe = render3d_rust.RustSceneRenderer().probe()
        assert probe.available is False
        assert "cargo is absent" in probe.reason


# ---------------------------------------------------------------------------
# registry contract
# ---------------------------------------------------------------------------

class TestRegistryContract:
    def test_capabilities_cover_the_core_vocabularies(self):
        from nomorals.media.edit_engine.scene3d import (
            LIGHT_KINDS,
            MATERIAL_KINDS,
            NODE_KINDS,
            PRIMITIVES,
        )
        caps = render3d_rust.RustSceneRenderer().capabilities()
        assert caps.deterministic is True
        assert set(caps.surfaces) == {"scene3d", "motion_camera", "format3d"}
        assert set(caps.node_kinds) == set(NODE_KINDS)
        assert set(caps.primitives) == set(PRIMITIVES)
        assert set(caps.material_kinds) == set(MATERIAL_KINDS)
        assert set(caps.light_kinds) == set(LIGHT_KINDS)
        assert caps.max_width > 0 and caps.max_height > 0

    def test_no_capability_gaps_for_a_full_feature_scene(self):
        scene = _scene(
            [SceneNode(name="ball", mesh=Mesh(
                kind="primitive", primitive="sphere"), material="m"),
             SceneNode(name="lamp", light=Light(kind="spot"))],
            materials={"m": Material(kind="phong")},
            tracks=[AnimTrack(target="nodes/ball/position",
                              keys=[AnimKey(0, (0, 0, 0))])])
        caps = render3d_rust.RustSceneRenderer().capabilities()
        assert capability_gaps(caps, scene_requirements(scene)) == []

    def test_resolution_and_status(self):
        scene = _scene([SceneNode(
            name="c", mesh=Mesh(kind="primitive", primitive="cube"),
            material="m")], materials={"m": _flat((1, 0, 0))})
        renderer, record = resolve_renderer(scene)
        assert renderer.name == "rust3d"
        assert record["capabilities"]["deterministic"] is True
        entry = [e for e in renderer_status() if e["name"] == "rust3d"]
        assert entry and entry[0]["probe"]["available"] is True

    def test_render_scene_record_ties_frames_to_states(self, tmp_path):
        scene = _scene([SceneNode(
            name="c", mesh=Mesh(kind="primitive", primitive="cube"),
            material="m")], materials={"m": _flat((0, 1, 0))})
        record = render_scene(scene, frames=[0], workdir=tmp_path)
        assert record["renderer"] == "rust3d"
        assert record["pattern"] == "f_%04d.png"
        assert len(record["frames"]) == 1
        from nomorals.media.edit_engine.scene3d import _state_hash
        expect = _state_hash(scene_state_at(scene, 0))
        assert record["state_hashes"] == {"0": expect}


# ---------------------------------------------------------------------------
# golden pixels
# ---------------------------------------------------------------------------

class TestGoldenPixels:
    def test_background_encodes_exactly(self, tmp_path):
        scene = _scene([], background=(0.05, 0.07, 0.12))
        record = _render(scene, [0], tmp_path)
        # linear (0.05, 0.07, 0.12) -> sRGB bytes (63, 75, 97)
        assert _px(record["frames"][0], 3, 3) == (63, 75, 97)
        assert _px(record["frames"][0], 60, 60) == (63, 75, 97)

    def test_occlusion_order(self, tmp_path):
        scene = _scene(
            [SceneNode(name="blue", transform=Transform(position=(0, 0, -4)),
                       mesh=Mesh(kind="primitive", primitive="cube",
                                 params={"size": 6.0}), material="b"),
             SceneNode(name="red", mesh=Mesh(
                 kind="primitive", primitive="cube", params={"size": 2.0}),
                 material="r")],
            materials={"b": _flat((0, 0, 1)), "r": _flat((1, 0, 0))})
        record = _render(scene, [0], tmp_path)
        frame = record["frames"][0]
        assert _px(frame, 32, 32) == (255, 0, 0)   # near cube wins center
        assert _px(frame, 40, 32) == (0, 0, 255)   # far cube alone at edge
        assert _px(frame, 2, 2) == (0, 0, 0)       # background corner

    def test_lambert_facing_shade_is_analytic(self, tmp_path):
        # plane facing +Z, directional light shining straight down
        # -Z: N·L = 1, so the pixel is the material color, encoded.
        scene = _scene(
            [SceneNode(name="quad", mesh=Mesh(
                kind="primitive", primitive="plane",
                params={"size": 4.0}), material="m"),
             SceneNode(name="sun", transform=Transform(
                 position=(0, 0, 5), rotation=Rotation.look_at((0, 0, 0))),
                 light=Light(kind="directional", intensity=1.0))],
            materials={"m": Material(kind="lambert", color=(0.5, 0.25, 1.0))})
        record = _render(scene, [0], tmp_path)
        px = _px(record["frames"][0], 32, 32)
        for got, want in zip(px, (188, 137, 255), strict=True):
            assert abs(got - want) <= 1

    def test_ambient_and_emissive_terms(self, tmp_path):
        # no directional light: ambient 0.4 * color 0.5 = 0.2 linear
        # -> sRGB 124; plus emissive 0.1 -> 0.3 linear -> 149.
        scene = _scene(
            [SceneNode(name="quad", mesh=Mesh(
                kind="primitive", primitive="plane",
                params={"size": 4.0}), material="m"),
             SceneNode(name="amb", light=Light(kind="ambient",
                                               intensity=0.4))],
            materials={"m": Material(kind="lambert", color=(0.5, 0.5, 0.5),
                                     emissive=(0.1, 0.1, 0.1))})
        record = _render(scene, [0], tmp_path)
        px = _px(record["frames"][0], 32, 32)
        assert all(abs(got - 149) <= 1 for got in px)

    def _project(self, scene, point, size=100):
        state = scene_state_at(scene, 0)
        view = state.camera["view_matrix"]
        proj = state.camera["projection_matrix"]

        def apply(m, v):
            return tuple(
                m[i * 4] * v[0] + m[i * 4 + 1] * v[1]
                + m[i * 4 + 2] * v[2] + m[i * 4 + 3] * v[3]
                for i in range(4))

        clip = apply(proj, apply(view, (*point, 1.0)))
        ndc = (clip[0] / clip[3], clip[1] / clip[3])
        return ((ndc[0] + 1) / 2 * size, (0.5 - ndc[1] / 2) * size)

    def test_camera_move_shifts_projection_as_predicted(self, tmp_path):
        def make(cam_pos):
            return _scene(
                [SceneNode(name="c", mesh=Mesh(
                    kind="primitive", primitive="cube",
                    params={"size": 2.0}), material="m")],
                materials={"m": _flat((1, 1, 1))},
                cameras={"main": Camera(
                    transform=Transform(position=cam_pos),
                    fov=90.0, near=1.0, far=11.0, aspect=1.0)},
                size=100)

        scene_a = make((0, 0, 5))
        scene_b = make((-3, 0, 5))  # same aim (-Z), camera slid left
        rec_a = _render(scene_a, [0], tmp_path / "a")
        rec_b = _render(scene_b, [0], tmp_path / "b")
        xa, ya = self._project(scene_a, (0, 0, 0))
        xb, yb = self._project(scene_b, (0, 0, 0))
        assert xb > xa + 20  # cube slides right in frame
        assert _px(rec_a["frames"][0], round(xa), round(ya)) == (255, 255, 255)
        assert _px(rec_b["frames"][0], round(xb), round(yb)) == (255, 255, 255)
        # and where the cube was in frame A is background in frame B
        assert _px(rec_b["frames"][0], round(xa), round(ya)) == (0, 0, 0)

    def test_phong_highlight_outbrightens_lambert(self, tmp_path):
        def make(kind):
            return _scene(
                [SceneNode(name="ball", mesh=Mesh(
                    kind="primitive", primitive="sphere",
                    params={"radius": 1.0, "segments": 24.0}),
                    material="m"),
                 SceneNode(name="sun", transform=Transform(
                     position=(0, 0, 5),
                     rotation=Rotation.look_at((0, 0, 0))),
                     light=Light(kind="directional", intensity=1.0))],
                materials={"m": Material(kind=kind, color=(0.3, 0.3, 0.3),
                                         shininess=32.0)})

        lam = _render(make("lambert"), [0], tmp_path / "lam")
        pho = _render(make("phong"), [0], tmp_path / "pho")
        lam_px = _px(lam["frames"][0], 32, 32)
        pho_px = _px(pho["frames"][0], 32, 32)
        # center fragment: N == V == L, so specular adds ~light color
        assert all(p > base + 40
                   for p, base in zip(pho_px, lam_px, strict=True))

    def test_point_light_fades_with_distance(self, tmp_path):
        def make(light_z):
            return _scene(
                [SceneNode(name="quad", mesh=Mesh(
                    kind="primitive", primitive="plane",
                    params={"size": 4.0}), material="m"),
                 SceneNode(name="bulb", transform=Transform(
                     position=(0, 0, light_z)),
                     light=Light(kind="point", intensity=2.0))],
                materials={"m": Material(kind="lambert",
                                         color=(1.0, 1.0, 1.0))})

        near = _render(make(1.0), [0], tmp_path / "near")
        far = _render(make(4.0), [0], tmp_path / "far")
        near_px = _px(near["frames"][0], 32, 32)
        far_px = _px(far["frames"][0], 32, 32)
        assert all(n > f for n, f in zip(near_px, far_px, strict=True))

    def test_spot_cone_lights_center_not_corner(self, tmp_path):
        scene = _scene(
            [SceneNode(name="quad", mesh=Mesh(
                kind="primitive", primitive="plane",
                params={"size": 6.0}), material="m"),
             SceneNode(name="spot", transform=Transform(
                 position=(0, 0, 5), rotation=Rotation.look_at((0, 0, 0))),
                 light=Light(kind="spot", intensity=6.0,
                             spot_angle=40.0, spot_penumbra=0.0))],
            materials={"m": Material(kind="lambert", color=(1.0, 1.0, 1.0))})
        record = _render(scene, [0], tmp_path)
        center = _px(record["frames"][0], 32, 32)
        corner = _px(record["frames"][0], 4, 4)
        assert center[0] > 110
        assert corner == (0, 0, 0)

    def test_translucent_blends_over_background(self, tmp_path):
        scene = _scene(
            [SceneNode(name="quad", mesh=Mesh(
                kind="primitive", primitive="plane",
                params={"size": 4.0}), material="m")],
            materials={"m": _flat((1, 1, 1), opacity=0.5)})
        record = _render(scene, [0], tmp_path)
        px = _px(record["frames"][0], 32, 32)
        # 0.5 * white over black = 0.5 linear -> sRGB 188
        assert all(abs(got - 188) <= 1 for got in px)

    def test_explicit_mesh_with_normals_renders(self, tmp_path):
        mesh = Mesh(kind="explicit",
                    vertices=[(-1, -1, 0), (1, -1, 0), (1, 1, 0),
                              (-1, 1, 0)],
                    faces=[(0, 1, 2, 3)],
                    normals=[(0, 0, 1)] * 4)
        scene = _scene(
            [SceneNode(name="quad", mesh=mesh, material="m")],
            materials={"m": _flat((0, 1, 0))})
        record = _render(scene, [0], tmp_path)
        assert _px(record["frames"][0], 32, 32) == (0, 255, 0)

    def test_animation_moves_the_cube(self, tmp_path):
        scene = _scene(
            [SceneNode(name="c", mesh=Mesh(
                kind="primitive", primitive="cube",
                params={"size": 2.0}), material="m")],
            materials={"m": _flat((1, 0, 0))},
            tracks=[AnimTrack(target="nodes/c/position", keys=[
                AnimKey(0, (0, 0, 0)), AnimKey(10, (2.5, 0, 0), "linear")])])
        record = _render(scene, [0, 10], tmp_path)
        assert [Path(p).name for p in record["frames"]] == [
            "f_0000.png", "f_0010.png"]
        assert record["framerate"] == 30.0
        assert record["width"] == 64 and record["height"] == 64

        def red_centroid(path):
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                xs = [x for x in range(64) for y in range(64)
                      if rgb.getpixel((x, y)) == (255, 0, 0)]
            return sum(xs) / len(xs)

        assert red_centroid(record["frames"][1]) > \
            red_centroid(record["frames"][0]) + 5

    def test_byte_identical_double_render(self, tmp_path):
        scene = _scene(
            [SceneNode(name="ball", transform=Transform(
                position=(0.5, 0.25, -1)),
                mesh=Mesh(kind="primitive", primitive="sphere",
                          params={"radius": 1.2, "segments": 16.0}),
                material="m"),
             SceneNode(name="sun", transform=Transform(
                 position=(3, 4, 5), rotation=Rotation.look_at((0, 0, 0))),
                 light=Light(kind="directional", intensity=1.1)),
             SceneNode(name="amb", light=Light(kind="ambient",
                                               intensity=0.2))],
            materials={"m": Material(kind="phong", color=(0.4, 0.5, 0.9),
                                     shininess=24.0)})
        first = _render(scene, [0], tmp_path / "one")
        second = _render(scene, [0], tmp_path / "two")

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        assert digest(first["frames"][0]) == digest(second["frames"][0])

    def test_still_scene_through_core_render_scene(self, tmp_path):
        # the full pipeline shape: a still (no tracks) renders its
        # single frame through the registry via render_scene
        scene = _scene(
            [SceneNode(name="c", mesh=Mesh(
                kind="primitive", primitive="cone",
                params={"radius": 1.0, "height": 2.0, "segments": 12.0}),
                material="m"),
             SceneNode(name="sun", transform=Transform(
                 position=(2, 3, 4), rotation=Rotation.look_at((0, 0, 0))),
                 light=Light(kind="directional"))],
            materials={"m": Material(kind="lambert",
                                     color=(0.9, 0.6, 0.2))})
        record = render_scene(scene, workdir=tmp_path)
        assert len(record["frames"]) == 1
        assert Path(record["frames"][0]).exists()
        assert json.loads(
            (tmp_path / "render3d_job.json").read_text())["frames"]


# ---------------------------------------------------------------------------
# registration hygiene
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_is_idempotent(self):
        render3d_rust.register_rust_renderer()
        render3d_rust.register_rust_renderer()
        assert registered_renderers().count("rust3d") == 1

    def test_unregister_and_reregister(self):
        assert unregister_renderer("rust3d") is True
        assert "rust3d" not in registered_renderers()
        render3d_rust.register_rust_renderer()
        assert "rust3d" in registered_renderers()
