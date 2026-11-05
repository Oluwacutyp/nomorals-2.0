"""The Blender-class 3D rung (§1.5e engine 2 of 2).

Three levels, in the build's honesty order:

* script generation — a fixed Scene3D produces a deterministic
  bpy script whose statements cover every mapping table entry,
  with the evaluated frame states baked as data (never re-derived);
* the driver against a protocol-faithful fake ``blender`` binary
  (honors ``--version`` and ``--background --python s -- manifest``,
  writes the manifest's PNGs, prints the completion marker) — this
  proves Devon's side end to end;
* a real-Blender smoke, only when a real binary is findable on the
  host (the provisioning route may have landed one).
"""

from __future__ import annotations

import json
import math
import stat
import tarfile
from pathlib import Path

import pytest

from nomorals.media.edit_engine import render3d_blender as rb
from nomorals.media.edit_engine.scene3d import (
    LIGHT_KINDS,
    MATERIAL_KINDS,
    PRIMITIVES,
    AnimKey,
    AnimTrack,
    Camera,
    Light,
    Material,
    Mesh,
    Rotation,
    Scene3D,
    Scene3DError,
    SceneNode,
    Transform,
    capability_gaps,
    register_renderer,
    registered_renderers,
    render_scene,
    reset_renderers,
    resolve_renderer,
    scene_requirements,
    unregister_renderer,
)

FAKE_BLENDER = """\
#!/usr/bin/env python3
import json, os, sys

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452")

FULL_PNG = PNG + bytes(64)

args = sys.argv[1:]
if "--version" in args:
    print("Blender 4.2.3 (fake-blender)")
    sys.exit(0)
if os.environ.get("FAKE_BLENDER_FAIL"):
    print("fake blender: deliberate failure", file=sys.stderr)
    sys.exit(3)
if "--python" in args:
    manifest = json.load(open(args[args.index("--") + 1]))
    os.makedirs(manifest["frames_dir"], exist_ok=True)
    for f in manifest["frames"]:
        path = os.path.join(manifest["frames_dir"],
                            "f_%04d.png" % f)
        with open(path, "wb") as fh:
            fh.write(FULL_PNG)
        print("DEVON3D_FRAME", f)
    print("DEVON3D_DONE " + json.dumps(
        {"blender_version": "4.2.3 (fake-blender)",
         "frames": len(manifest["frames"])}))
    sys.exit(0)
sys.exit(2)
"""


def make_fake_blender(path: Path) -> Path:
    path.write_text(FAKE_BLENDER, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
               | stat.S_IXOTH)
    return path


def rich_scene() -> Scene3D:
    return Scene3D(
        title="rich",
        nodes=[
            SceneNode(name="root",
                      transform=Transform(position=(10, 0, 0))),
            SceneNode(name="crate", parent="root",
                      transform=Transform(position=(1, 2, 3)),
                      mesh=Mesh(kind="primitive", primitive="cube",
                                params={"size": 2.0}),
                      material="paint"),
            SceneNode(name="ball",
                      mesh=Mesh(kind="primitive", primitive="sphere",
                                params={"radius": 0.5, "segments": 24,
                                        "ring_count": 12}),
                      material="shiny"),
            SceneNode(name="floor",
                      mesh=Mesh(kind="primitive", primitive="plane",
                                params={"size": 20.0}),
                      material="paint"),
            SceneNode(name="pipe",
                      mesh=Mesh(kind="primitive",
                                primitive="cylinder",
                                params={"radius": 0.25, "height": 3.0,
                                        "segments": 16}),
                      material="shiny"),
            SceneNode(name="cone",
                      mesh=Mesh(kind="primitive", primitive="cone",
                                params={"radius": 1.0, "height": 2.0}),
                      material="flatm"),
            SceneNode(name="tri",
                      mesh=Mesh(kind="explicit",
                                vertices=[(0, 0, 0), (1, 0, 0),
                                          (0, 1, 0)],
                                faces=[(0, 1, 2)],
                                normals=[(0, 0, 1)] * 3),
                      material="paint"),
            SceneNode(name="sun",
                      transform=Transform(
                          position=(3, 5, 0),
                          rotation=Rotation.look_at((0, 0, 0))),
                      light=Light(kind="directional", intensity=2.0)),
            SceneNode(name="lamp",
                      transform=Transform(position=(0, 3, 0)),
                      light=Light(kind="point", intensity=5.0)),
            SceneNode(name="spot",
                      transform=Transform(position=(0, 4, 2)),
                      light=Light(kind="spot", intensity=8.0,
                                  spot_angle=30.0,
                                  spot_penumbra=0.25)),
            SceneNode(name="amb",
                      light=Light(kind="ambient", intensity=0.3,
                                  color=(0.2, 0.2, 0.3))),
        ],
        cameras={"main": Camera(transform=Transform(
            position=(0, 0, 6),
            rotation=Rotation.look_at((0, 0, 0))), fov=50.0)},
        active_camera="main",
        materials={
            "paint": Material(kind="lambert", color=(0.2, 0.4, 0.9)),
            "shiny": Material(kind="phong", color=(0.9, 0.8, 0.2),
                              shininess=32.0),
            "flatm": Material(kind="flat", color=(1.0, 0.0, 0.0),
                              opacity=0.5),
        },
        tracks=[
            AnimTrack(target="nodes/crate/position",
                      keys=[AnimKey(0, (1, 2, 3)),
                            AnimKey(30, (4, 2, 3), "linear")]),
            AnimTrack(target="nodes/spot/intensity",
                      keys=[AnimKey(0, (8.0,)),
                            AnimKey(30, (2.0,), "linear")]),
            AnimTrack(target="materials/shiny/color",
                      keys=[AnimKey(0, (0.9, 0.8, 0.2)),
                            AnimKey(30, (0.1, 0.2, 0.9), "linear")]),
            AnimTrack(target="cameras/main/fov",
                      keys=[AnimKey(0, (50.0,)),
                            AnimKey(30, (35.0,), "linear")]),
        ],
        output={"width": 320, "height": 240, "fps": 30},
    )


@pytest.fixture()
def fake_locator(tmp_path):
    fake = make_fake_blender(tmp_path / "fake-blender")
    return rb.BlenderLocator(
        tools_dir=tmp_path / "tools",
        environ={"NM_BLENDER_BIN": str(fake), "PATH": ""})


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_renderers()
    yield
    reset_renderers()


# ---------------------------------------------------------------------------
# script generation
# ---------------------------------------------------------------------------

class TestScriptGeneration:
    def test_deterministic_byte_for_byte(self):
        scene = rich_scene()
        a = rb.generate_bpy_script(scene, [0, 15, 30])
        b = rb.generate_bpy_script(scene, [30, 0, 15])
        assert a == b

    def test_material_insertion_order_does_not_matter(self):
        scene = rich_scene()
        data = scene.to_dict()
        mats = dict(reversed(list(data["materials"].items())))
        scene2 = Scene3D.from_dict({**data, "materials": mats})
        assert rb.generate_bpy_script(scene, [0, 30]) == \
            rb.generate_bpy_script(scene2, [0, 30])

    def test_all_primitives_emitted(self):
        script = rb.generate_bpy_script(rich_scene(), [0])
        for primitive in PRIMITIVES:
            assert rb.PRIMITIVE_MAP[primitive]["op"] in script
        # param translation landed: cylinder height -> depth
        assert "depth=3.0" in script
        assert "radius=0.5" in script and "ring_count=12" in script

    def test_explicit_mesh_emitted(self):
        script = rb.generate_bpy_script(rich_scene(), [0])
        assert "from_pydata" in script
        assert "normals_split_custom_set_from_vertices" in script

    def test_material_constructions(self):
        script = rb.generate_bpy_script(rich_scene(), [0])
        assert "ShaderNodeEmission" in script          # flat
        assert "ShaderNodeBsdfPrincipled" in script   # lambert/phong
        assert "ShaderNodeBsdfTransparent" in script  # flat opacity
        rough = math.sqrt(2.0 / (32.0 + 2.0))
        assert f"BS.inputs['Roughness'].default_value = {rough!r}" \
            in script  # the named Blinn-Phong -> GGX conversion

    def test_light_constructions(self):
        script = rb.generate_bpy_script(rich_scene(), [0])
        for btype in ("'SUN'", "'POINT'", "'SPOT'"):
            assert "bpy.data.lights.new" in script
            assert btype in script
        assert "spot_size" in script and "spot_blend" in script
        # ambient rides the world background, not a light object
        assert "'Background'" in script

    def test_animation_is_baked_state_not_rederived(self):
        scene = rich_scene()
        script = rb.generate_bpy_script(scene, [0, 30])
        table = rb._frame_table(scene, [0, 30])
        embedded = json.dumps(table, sort_keys=True,
                              separators=(",", ":"))
        assert embedded in script
        assert "keyframe_insert('location'" in script
        assert "keyframe_insert('rotation_quaternion'" in script
        assert "keyframe_insert('energy'" in script
        # the fov at the last frame is the core's sampled 35.0,
        # inside the baked table — not an fcurve Blender computes
        assert table["30"]["cam"]["fov"] == 35.0
        assert table["30"]["nodes"]["spot"]["e"] == 2.0

    def test_camera_and_render_settings(self):
        script = rb.generate_bpy_script(rich_scene(), [0])
        assert "S.render.resolution_x = 320" in script
        assert "S.render.fps = 30" in script
        lens0 = 12.0 / math.tan(math.radians(50.0) / 2.0)
        assert f"CD.lens = {lens0!r}" in script
        assert "CD.keyframe_insert('lens'" in script
        assert "S.camera = CAM" in script
        assert "'f_%04d' % _f_no" in script
        assert "DEVON3D_DONE" in script

    def test_empty_frames_refuse(self):
        with pytest.raises(Scene3DError):
            rb.generate_bpy_script(rich_scene(), [])


# ---------------------------------------------------------------------------
# capabilities — declared from the same tables the generator emits
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_caps_derive_from_mapping_tables(self):
        caps = rb.BlenderRenderer().capabilities()
        assert set(caps.primitives) == set(rb.PRIMITIVE_MAP) \
            == set(PRIMITIVES)
        assert set(caps.material_kinds) == set(rb.MATERIAL_MAP) \
            == set(MATERIAL_KINDS)
        assert set(caps.light_kinds) == set(rb.LIGHT_MAP) \
            == set(LIGHT_KINDS)
        assert caps.surfaces == ("scene3d",)
        assert caps.deterministic is False
        assert "unitless" in caps.notes  # approximations named

    def test_resolution_serves_rich_scene(self, fake_locator):
        renderer = rb.BlenderRenderer(locator=fake_locator)
        register_renderer(renderer)
        scene = rich_scene()
        assert capability_gaps(renderer.capabilities(),
                               scene_requirements(scene)) == []
        found, record = resolve_renderer(scene)
        assert found is renderer
        assert record["renderer"] == "blender"


# ---------------------------------------------------------------------------
# probe + provisioning
# ---------------------------------------------------------------------------

class TestProbe:
    def test_fake_binary_probes_available(self, fake_locator):
        probe = fake_locator.probe()
        assert probe.available
        assert "4.2.3" in probe.reason

    def test_explicit_bad_bin_path_is_a_fact(self, tmp_path):
        loc = rb.BlenderLocator(
            tools_dir=tmp_path / "tools",
            environ={"NM_BLENDER_BIN": str(tmp_path / "nope"),
                     "PATH": ""})
        probe = loc.probe()
        assert not probe.available
        assert "NM_BLENDER_BIN" in probe.reason

    def test_failed_provision_is_named_and_not_retried(self, tmp_path):
        calls = []

        def downloader(url, dest):
            calls.append(url)
            raise RuntimeError("network down")

        loc = rb.BlenderLocator(
            tools_dir=tmp_path / "tools",
            environ={"PATH": ""}, downloader=downloader)
        probe = loc.probe()
        assert not probe.available
        assert "network down" in probe.reason
        marker = json.loads(
            (tmp_path / "tools" / ".provision.json").read_text())
        assert marker["status"] == "failed"
        probe2 = loc.probe()
        assert not probe2.available
        assert "already failed" in probe2.reason
        assert len(calls) == 1  # marker-gated: no silent retry

    def test_no_provision_env_skips_attempt(self, tmp_path):
        def downloader(url, dest):  # pragma: no cover - must not run
            raise AssertionError("download attempted")

        loc = rb.BlenderLocator(
            tools_dir=tmp_path / "tools",
            environ={"PATH": "", "NM_BLENDER_NO_PROVISION": "1"},
            downloader=downloader)
        probe = loc.probe()
        assert not probe.available
        assert "provisioning is disabled" in probe.reason

    def test_provision_route_lands_a_binary(self, tmp_path):
        # a tiny tar.xz carrying a fake `blender`, built like the
        # official layout: blender-<ver>-linux-x64/blender
        fake_src = make_fake_blender(tmp_path / "src-blender")
        arcname = ("blender-4.2.3-linux-x64/blender")

        def downloader(url, dest):
            assert url == rb.BLENDER_TARBALL_URL
            with tarfile.open(dest, "w:xz") as tar:
                tar.add(fake_src, arcname=arcname)

        loc = rb.BlenderLocator(
            tools_dir=tmp_path / "tools",
            environ={"PATH": ""}, downloader=downloader)
        probe = loc.probe()
        assert probe.available, probe.reason
        assert "self-provisioned" in probe.reason
        binary = loc.find()
        assert binary is not None
        assert binary.name == "blender"
        assert loc.version(binary).startswith("Blender 4.2.3")


# ---------------------------------------------------------------------------
# the driver, against the fake binary
# ---------------------------------------------------------------------------

class TestDriverFake:
    def test_render_frames_end_to_end(self, fake_locator, tmp_path):
        renderer = rb.BlenderRenderer(locator=fake_locator)
        record = renderer.render_frames(
            rich_scene(), frames=[0, 15, 30], workdir=tmp_path / "out")
        assert record["pattern"] == "f_%04d.png"
        assert record["frames"] == 3
        assert record["framerate"] == 30
        assert record["width"] == 320 and record["height"] == 240
        assert record["files"] == ["f_0000.png", "f_0015.png",
                                   "f_0030.png"]
        assert "fake-blender" in record["blender_version"]
        frames_dir = Path(record["frames_dir"])
        for name in record["files"]:
            data = (frames_dir / name).read_bytes()
            assert data.startswith(b"\x89PNG")
        assert Path(record["script"]).is_file()
        manifest = json.loads(Path(record["manifest"]).read_text())
        assert manifest["frames"] == [0, 15, 30]

    def test_render_scene_through_registry(self, fake_locator,
                                            tmp_path):
        register_renderer(rb.BlenderRenderer(locator=fake_locator))
        record = render_scene(rich_scene(), frames=[0, 30],
                              workdir=tmp_path / "out")
        assert record["renderer"] == "blender"
        assert record["resolution"]["renderer"] == "blender"
        assert set(record["state_hashes"]) == {"0", "30"}
        assert record["frames"] == 2

    def test_failing_blender_raises_with_tail(self, fake_locator,
                                              tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_BLENDER_FAIL", "1")
        renderer = rb.BlenderRenderer(locator=fake_locator)
        with pytest.raises(Scene3DError) as exc:
            renderer.render_frames(rich_scene(), frames=[0],
                                   workdir=tmp_path / "out")
        assert "exited 3" in str(exc.value)

    def test_unavailable_renderer_refuses(self, tmp_path):
        loc = rb.BlenderLocator(
            tools_dir=tmp_path / "tools",
            environ={"PATH": "", "NM_BLENDER_NO_PROVISION": "1"})
        renderer = rb.BlenderRenderer(locator=loc)
        with pytest.raises(Scene3DError) as exc:
            renderer.render_frames(rich_scene(), frames=[0],
                                   workdir=tmp_path / "out")
        assert "unavailable" in str(exc.value)

    def test_module_registration_is_live(self):
        # importing the module registered the rung (import-time
        # wiring); the fixture reset it — re-register and see it.
        rb.register_blender_renderer()
        assert "blender" in registered_renderers()
        assert unregister_renderer("blender")


# ---------------------------------------------------------------------------
# the real thing, when the host has it
# ---------------------------------------------------------------------------

@pytest.mark.skipif(rb.find_blender() is None,
                    reason="no real blender binary on this host")
class TestRealBlenderSmoke:
    def test_one_frame_render(self, tmp_path):
        scene = Scene3D(
            title="smoke",
            nodes=[SceneNode(
                name="crate",
                mesh=Mesh(kind="primitive", primitive="cube",
                          params={"size": 2.0}),
                material="paint"),
                   SceneNode(
                       name="sun",
                       transform=Transform(position=(3, 5, 0)),
                       light=Light(kind="directional",
                                   intensity=2.0))],
            cameras={"main": Camera(transform=Transform(
                position=(0, 0, 6),
                rotation=Rotation.look_at((0, 0, 0))))},
            active_camera="main",
            materials={"paint": Material(kind="lambert",
                                         color=(0.2, 0.4, 0.9))},
            output={"width": 64, "height": 64, "fps": 30})
        renderer = rb.BlenderRenderer(samples=1)
        record = renderer.render_frames(scene, frames=[0],
                                        workdir=tmp_path / "out")
        png = Path(record["frames_dir"]) / "f_0000.png"
        data = png.read_bytes()
        assert data.startswith(b"\x89PNG")
        assert len(data) > 200  # a real render, not an empty frame
        # bpy.app.version_string shape: "4.2.3 LTS"
        assert "4.2" in record["blender_version"]
