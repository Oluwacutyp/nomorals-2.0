"""Failing-first tests for the ArtCraft port (lane 92).

ArtCraft (storytold/artcraft, dual MIT/Apache-2.0 — LICENSE-MIT +
LICENSE-APACHE verified in the clone on disk) is the IDE for artists:
text-to-image across models, prompted + canvas image editing, 2D
compositing, 3D compositing, image-to-location, character posing +
camera before generating, scene blocking with kitbashing, character
identity transfer, background removal, image-to-3D-mesh, image-to-video.

The portable core — the part Devon does not already have — is the
SCENE-BEFORE-GENERATION organ: build the scene (2D layers, 3D staging
with depth + camera, characters with pose + identity, one consistent
location across shots), render the staged frame, then generate through
her own rungs. NOT ported: ArtCraft's paid provider router (fal,
Midjourney, Sora, Grok, WorldLabs, GPT Image, Nano Banana, Seedream —
Devon calls no paid service, owner order 2026-11-18), its cloud 3D
scene projects, its cost estimation in dollars (Devon reports
capability, her rungs cost $0), the desktop egui app itself (Devon
answers in chat).

What Devon already has and this organ wires (never rebuilds):
imggen Studio (text-to-image / img2img / inpaint / upscale),
imggen bgremove + classical upscale, videogen pipeline.generate
(LTX previz / Wan finals / motion fallback) + chain_scenes
(multi-shot films), motion-studio kenburns + montage assemble,
photo_editor PhotoDocument layers, media_edit avatar.talking_head
(lip-sync), /models provision. The Wan/LTX neural rungs are
import-gated and need a GPU this box does not have — the engine
names that wall honestly and the motion/classical chain is the real
proof path here; neural orchestration is proven with injected rungs.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from PIL import Image  # noqa: E402

from nomorals.media.artist import compose as artist_compose  # noqa: E402
from nomorals.media.artist import generate as artist_generate  # noqa: E402
from nomorals.media.artist import scene as artist_scene  # noqa: E402
from nomorals.media.artist.store import ArtistStore, artist_path  # noqa: E402

UPSTREAM = "storytold/artcraft"
LICENCE = "MIT OR Apache-2.0"


def _tmp_home(test: unittest.TestCase) -> Path:
    tmp = tempfile.mkdtemp(prefix="artist-test-")
    test.addCleanup(__import__("shutil").rmtree, tmp, True)
    return Path(tmp)


def _solid_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> str:
    Image.new("RGB", size, color).save(path)
    return str(path)


def _scene(**over: Any) -> artist_scene.Scene:
    kw: dict[str, Any] = {"title": "Test scene"}
    kw.update(over)
    return artist_scene.Scene(**kw)


class SceneModelTests(unittest.TestCase):
    def test_create_defaults(self) -> None:
        s = _scene()
        self.assertEqual((s.width, s.height), (1280, 720))
        self.assertEqual(s.layers, [])
        self.assertEqual(s.shots, [])
        self.assertEqual(s.upstream, UPSTREAM)
        self.assertEqual(s.licence, LICENCE)
        self.assertTrue(s.id)

    def test_bad_size_refused(self) -> None:
        with self.assertRaises(artist_scene.ArtistError):
            _scene(width=0)
        with self.assertRaises(artist_scene.ArtistError):
            _scene(width=33, height=720)  # odd widths break encoders
        with self.assertRaises(artist_scene.ArtistError):
            _scene(width=1280, height=720, title="")

    def test_layer_add_and_validation(self) -> None:
        s = _scene()
        layer = s.add_layer(kind="solid", name="bg", color="#112233",
                            x=0, y=0, width=1280, height=720)
        self.assertEqual(layer.kind, "solid")
        self.assertEqual(len(s.layers), 1)
        with self.assertRaises(artist_scene.ArtistError):
            s.add_layer(kind="hologram", name="x")
        with self.assertRaises(artist_scene.ArtistError):
            s.add_layer(kind="solid", name="x", opacity=1.5)
        with self.assertRaises(artist_scene.ArtistError):
            s.add_layer(kind="solid", name="x", opacity=-0.1)
        with self.assertRaises(artist_scene.ArtistError):
            s.add_layer(kind="text", name="t")  # text layer needs text
        with self.assertRaises(artist_scene.ArtistError):
            s.add_layer(kind="image", name="i")  # image layer needs path
        with self.assertRaises(artist_scene.ArtistError):
            s.add_layer(kind="solid", name="bg")  # duplicate name

    def test_layer_move_and_remove(self) -> None:
        s = _scene()
        s.add_layer(kind="solid", name="a", color="#000000")
        s.add_layer(kind="solid", name="b", color="#ffffff")
        s.move_layer("a", 1)
        self.assertEqual([layer.name for layer in s.layers], ["b", "a"])
        removed = s.remove_layer("b")
        self.assertEqual(removed.name, "b")
        self.assertEqual([layer.name for layer in s.layers], ["a"])
        with self.assertRaises(artist_scene.ArtistError):
            s.remove_layer("nope")

    def test_character_pose(self) -> None:
        s = _scene()
        ch = s.add_character("Mara", ref_image="/tmp/mara.png",
                             lora="mara.gguf")
        self.assertEqual(ch.pose, "neutral")
        s.pose_character("Mara", "wave", joints={"arm_r": 45.0})
        self.assertEqual(s.character("Mara").pose, "wave")
        self.assertEqual(s.character("Mara").joints["arm_r"], 45.0)
        with self.assertRaises(artist_scene.ArtistError):
            s.pose_character("Ghost", "wave")
        with self.assertRaises(artist_scene.ArtistError):
            s.add_character("Mara")

    def test_camera_and_location(self) -> None:
        s = _scene()
        s.set_camera(x=0.0, y=0.0, z=0.0, fov=50.0)
        self.assertEqual(s.camera.fov, 50.0)
        with self.assertRaises(artist_scene.ArtistError):
            s.set_camera(fov=5.0)
        with self.assertRaises(artist_scene.ArtistError):
            s.set_camera(fov=200.0)
        s.set_location("Roof at dusk", backdrop="/tmp/roof.png",
                       notes="golden hour")
        self.assertEqual(s.location.name, "Roof at dusk")
        self.assertEqual(s.location.backdrop, "/tmp/roof.png")

    def test_shot_add_and_validation(self) -> None:
        s = _scene()
        shot = s.add_shot("Opening", prompt="a rooftop at dusk",
                          mode="i2v", duration_s=6.0, lines="Hello there.")
        self.assertEqual(shot.mode, "i2v")
        with self.assertRaises(artist_scene.ArtistError):
            s.add_shot("Bad", prompt="x", mode="teleport")
        with self.assertRaises(artist_scene.ArtistError):
            s.add_shot("Bad", prompt="x", duration_s=0.0)
        with self.assertRaises(artist_scene.ArtistError):
            s.add_shot("Bad", prompt="x", duration_s=9999.0)
        with self.assertRaises(artist_scene.ArtistError):
            s.add_shot("Opening", prompt="dup")
        with self.assertRaises(artist_scene.ArtistError):
            s.add_shot("Talk", prompt="x", mode="talk",
                       character="Nobody")

    def test_staging_items(self) -> None:
        s = _scene()
        item = s.stage_item("Lantern", image="/tmp/lantern.png",
                            depth=2.0, x=100.0, y=50.0)
        self.assertEqual(item.depth, 2.0)
        with self.assertRaises(artist_scene.ArtistError):
            s.stage_item("Bad", depth=-1.0)
        with self.assertRaises(artist_scene.ArtistError):
            s.stage_item("Lantern", depth=1.0)  # duplicate name

    def test_round_trip(self) -> None:
        s = _scene(title="Round trip")
        s.add_layer(kind="solid", name="bg", color="#112233")
        s.add_character("Mara", pose="wave")
        s.set_camera(fov=60.0)
        s.set_location("Studio")
        s.add_shot("One", prompt="a studio", mode="t2v", duration_s=5.0)
        s.stage_item("Prop", depth=1.5, x=10.0, y=20.0)
        data = s.to_dict()
        back = artist_scene.Scene.from_dict(data)
        self.assertEqual(back.to_dict(), data)
        self.assertEqual(back.title, "Round trip")
        self.assertEqual(back.layers[0].color, "#112233")
        self.assertEqual(back.character("Mara").pose, "wave")
        self.assertEqual(back.shots[0].mode, "t2v")
        with self.assertRaises(artist_scene.ArtistError):
            artist_scene.Scene.from_dict({"title": "No id"})
        with self.assertRaises(artist_scene.ArtistError):
            artist_scene.Scene.from_dict(
                {"id": "x", "title": "t", "width": 10, "height": 10,
                 "layers": [{"kind": "nope", "name": "l"}]})


class ComposeTests(unittest.TestCase):
    def test_flatten_layer_order(self) -> None:
        s = _scene(width=100, height=100)
        s.add_layer(kind="solid", name="bottom", color="#ff0000",
                    x=0, y=0, width=100, height=100)
        s.add_layer(kind="solid", name="top", color="#0000ff",
                    x=50, y=50, width=50, height=50)
        img = artist_compose.flatten_layers(s)
        self.assertEqual(img.size, (100, 100))
        self.assertEqual(img.getpixel((10, 10))[:3], (255, 0, 0))
        self.assertEqual(img.getpixel((75, 75))[:3], (0, 0, 255))

    def test_flatten_opacity_blends(self) -> None:
        s = _scene(width=16, height=16)
        s.add_layer(kind="solid", name="bottom", color="#ffffff",
                    x=0, y=0, width=16, height=16)
        s.add_layer(kind="solid", name="half", color="#000000",
                    x=0, y=0, width=16, height=16, opacity=0.5)
        img = artist_compose.flatten_layers(s)
        px = img.getpixel((8, 8))[:3]
        self.assertTrue(all(100 <= v <= 160 for v in px),
                        f"half-black over white should be grey, got {px}")

    def test_text_layer_renders(self) -> None:
        s = _scene(width=200, height=100)
        s.add_layer(kind="solid", name="bg", color="#000000",
                    x=0, y=0, width=200, height=100)
        s.add_layer(kind="text", name="title", text="HELLO",
                    x=10, y=10, width=180, height=40,
                    font_size=24, color="#ffffff")
        img = artist_compose.flatten_layers(s)
        colors = img.convert("RGB").getcolors(maxcolors=1000000) or []
        self.assertTrue(any(c[1] != (0, 0, 0) for c in colors),
                        "text layer rendered nothing")

    def test_missing_image_layer_is_named(self) -> None:
        s = _scene(width=100, height=100)
        s.add_layer(kind="image", name="pic", path="/tmp/does-not-exist.png",
                    x=0, y=0, width=100, height=100)
        with self.assertRaises(artist_scene.ArtistError) as ctx:
            artist_compose.flatten_layers(s)
        self.assertIn("/tmp/does-not-exist.png", str(ctx.exception))

    def test_staged_frame_depth_order(self) -> None:
        home = _tmp_home(self)
        far = _solid_png(home / "far.png", (40, 40), (255, 0, 0))
        near = _solid_png(home / "near.png", (40, 40), (0, 0, 255))
        s = _scene(width=200, height=200)
        s.set_location("Room")
        s.stage_item("far", image=far, depth=5.0, x=0.0, y=0.0)
        s.stage_item("near", image=near, depth=0.5, x=0.0, y=0.0)
        img = artist_compose.render_staged_frame(s)
        self.assertEqual(img.size, (200, 200))
        # Both items sit at the centre; the nearer one must win.
        self.assertEqual(img.getpixel((100, 100))[:3], (0, 0, 255))

    def test_staged_frame_camera_pan_moves_content(self) -> None:
        home = _tmp_home(self)
        dot = _solid_png(home / "dot.png", (20, 20), (0, 255, 0))
        s = _scene(width=200, height=200)
        s.set_location("Room")
        s.stage_item("dot", image=dot, depth=1.0, x=0.0, y=0.0)
        base = artist_compose.render_staged_frame(s)
        s.set_camera(x=40.0, y=0.0, z=0.0, fov=50.0)
        moved = artist_compose.render_staged_frame(s)
        self.assertNotEqual(list(base.getdata()), list(moved.getdata()))

    def test_character_without_ref_renders_mannequin(self) -> None:
        s = _scene(width=200, height=200)
        s.set_location("Room")
        s.add_character("Mara")
        s.stage_item("Mara", character="Mara", depth=1.0, x=0.0, y=0.0)
        img = artist_compose.render_staged_frame(s)
        colors = img.convert("RGB").getcolors(maxcolors=1000000) or []
        # A mannequin block must be visible against the backdrop.
        self.assertTrue(len(colors) > 1)


class StoreTests(unittest.TestCase):
    def _store(self, home: Path) -> ArtistStore:
        return ArtistStore(artist_path(
            SimpleNamespace(home_path=home)))

    def test_crud_and_list(self) -> None:
        home = _tmp_home(self)
        store = self._store(home)
        scene = _scene(title="Alpha")
        store.save(scene)
        got = store.get(scene.id)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.title, "Alpha")
        self.assertEqual([row["id"] for row in store.list()], [scene.id])
        scene.title = "Alpha 2"
        store.save(scene)
        self.assertEqual(store.get(scene.id).title, "Alpha 2")  # type: ignore[union-attr]
        self.assertTrue(store.delete(scene.id))
        self.assertIsNone(store.get(scene.id))
        self.assertFalse(store.delete(scene.id))
        self.assertIsNone(store.get("missing"))

    def test_path_honours_settings_home(self) -> None:
        home = _tmp_home(self)
        self.assertEqual(
            artist_path(SimpleNamespace(home_path=home)),
            home / "artist" / "scenes.json")

    def test_corrupt_store_quarantined(self) -> None:
        home = _tmp_home(self)
        path = artist_path(SimpleNamespace(home_path=home))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        store = self._store(home)
        self.assertEqual(store.list(), [])
        quarantined = sorted(path.parent.glob("scenes.json.corrupt.*"))
        self.assertEqual(len(quarantined), 1)


class FakeStudio:
    """Stand-in for imggen Studio: records calls, writes a real PNG."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.calls: list[dict[str, Any]] = []

    def _out(self, save_to: str | None) -> str:
        out = Path(save_to) if save_to else (
            self.home / f"fake-{len(self.calls)}.png")
        Image.new("RGB", (64, 64), (10, 20, 30)).save(out)
        return str(out)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"op": "generate", "prompt": prompt,
                           "kwargs": kwargs})
        return self._out(kwargs.get("save_to"))

    def img2img(self, image_path: str, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"op": "img2img", "image": image_path,
                           "prompt": prompt, "kwargs": kwargs})
        return self._out(kwargs.get("save_to"))

    def inpaint(self, image_path: str, mask_path: str, prompt: str,
                **kwargs: Any) -> str:
        self.calls.append({"op": "inpaint", "image": image_path,
                           "mask": mask_path, "prompt": prompt,
                           "kwargs": kwargs})
        return self._out(kwargs.get("save_to"))

    def upscale(self, image_path: str, scale: float = 2.0,
                save_to: str | None = None, **kwargs: Any) -> list[str]:
        self.calls.append({"op": "upscale", "image": image_path,
                           "scale": scale})
        out = Path(save_to) if save_to else self.home / "fake-up.png"
        Image.new("RGB", (128, 128), (40, 50, 60)).save(out)
        return [str(out)]


class FakeVideo:
    """Stand-in for the videogen chain + montage rungs."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.chains: list[list[dict[str, Any]]] = []
        self.assembles: list[dict[str, Any]] = []
        self.talks: list[dict[str, Any]] = []

    def chain(self, scenes: list[dict[str, Any]], **kwargs: Any):
        self.chains.append(scenes)
        out = self.home / "fake-chain.mp4"
        out.write_bytes(b"FAKEVIDEO")
        report = SimpleNamespace(
            final_path=str(out),
            scenes=[SimpleNamespace(index=i, path=str(out), engine="motion",
                                    prompt=s.get("prompt", ""),
                                    model="", loras=[])
                    for i, s in enumerate(scenes)],
            neural_scenes=0, motion_scenes=len(scenes),
            grade_preset="", format="", note="", consistency={})
        return report

    def assemble(self, timeline, out, **kwargs: Any) -> str:
        self.assembles.append({"timeline": timeline, "out": str(out)})
        Path(out).write_bytes(b"FAKEVIDEO")
        return str(out)

    def talking_head(self, photo: str, lines: str, **kwargs: Any) -> str:
        self.talks.append({"photo": photo, "lines": lines})
        out = self.home / "fake-talk.mp4"
        out.write_bytes(b"FAKETALK")
        return str(out)


def _engine(home: Path, studio: Any = None, video: FakeVideo | None = None,
            **over: Any) -> artist_generate.ArtistEngine:
    store = ArtistStore(artist_path(SimpleNamespace(home_path=home)))
    return artist_generate.ArtistEngine(
        store, home=home, studio=studio,
        chain_scenes=(video.chain if video else None),
        assemble=(video.assemble if video else None),
        talking_head=(video.talking_head if video else None),
        **over)


class EngineTests(unittest.TestCase):
    def test_compose_preview_writes_real_png(self) -> None:
        home = _tmp_home(self)
        engine = _engine(home)
        scene = _scene(title="Preview")
        scene.add_layer(kind="solid", name="bg", color="#203040")
        engine.store.save(scene)
        out = engine.compose_preview(scene.id)
        self.assertTrue(out["ok"])
        img = Image.open(out["path"])
        self.assertEqual(img.size, (1280, 720))
        self.assertEqual(engine.store.get(scene.id).artifacts["preview"]["path"],  # type: ignore[union-attr]
                         out["path"])

    def test_generate_text_calls_studio_generate(self) -> None:
        home = _tmp_home(self)
        studio = FakeStudio(home)
        engine = _engine(home, studio=studio)
        scene = _scene(title="Gen")
        engine.store.save(scene)
        out = engine.generate_image(scene.id, prompt="a red door",
                                    mode="text")
        self.assertTrue(out["ok"], out)
        self.assertEqual(studio.calls[0]["op"], "generate")
        self.assertEqual(studio.calls[0]["prompt"], "a red door")
        self.assertTrue(Path(out["path"]).exists())
        saved = engine.store.get(scene.id)
        assert saved is not None
        self.assertEqual(saved.artifacts["image"]["rung"], "imggen-studio")
        self.assertEqual(saved.artifacts["image"]["mode"], "text")

    def test_generate_img2img_uses_staged_frame(self) -> None:
        home = _tmp_home(self)
        studio = FakeStudio(home)
        engine = _engine(home, studio=studio)
        scene = _scene(title="Ref")
        scene.add_layer(kind="solid", name="bg", color="#112233")
        scene.set_location("Room")
        engine.store.save(scene)
        out = engine.generate_image(scene.id, prompt="a moody room")
        self.assertTrue(out["ok"], out)
        self.assertEqual(studio.calls[0]["op"], "img2img")
        frame = studio.calls[0]["image"]
        self.assertTrue(Path(frame).exists())
        self.assertTrue(frame.endswith(".png"))
        saved = engine.store.get(scene.id)
        assert saved is not None
        self.assertEqual(saved.artifacts["image"]["mode"], "img2img")

    def test_generate_without_studio_is_honest(self) -> None:
        home = _tmp_home(self)
        engine = _engine(home, studio=None)
        scene = _scene(title="NoStudio")
        engine.store.save(scene)
        out = engine.generate_image(scene.id, prompt="x", mode="text")
        self.assertFalse(out["ok"])
        self.assertIn("error", out)
        self.assertNotIn("path", out)
        self.assertTrue(
            "torch" in out["error"].lower()
            or "not installed" in out["error"].lower()
            or "unavailable" in out["error"].lower(),
            out["error"])

    def test_edit_routes_inpaint_with_mask(self) -> None:
        home = _tmp_home(self)
        studio = FakeStudio(home)
        engine = _engine(home, studio=studio)
        scene = _scene(title="Edit")
        engine.store.save(scene)
        src = _solid_png(home / "src.png", (64, 64), (1, 2, 3))
        mask = _solid_png(home / "mask.png", (64, 64), (255, 255, 255))
        out = engine.edit_image(scene.id, src, "add a cat", mask=mask)
        self.assertTrue(out["ok"], out)
        self.assertEqual(studio.calls[0]["op"], "inpaint")
        out2 = engine.edit_image(scene.id, src, "make it dusk")
        self.assertTrue(out2["ok"], out2)
        self.assertEqual(studio.calls[1]["op"], "img2img")

    def test_cutout_and_upscale_use_injected_rungs(self) -> None:
        home = _tmp_home(self)
        cut_src = _solid_png(home / "cut.png", (32, 32), (9, 9, 9))
        cut_out = home / "cut-out.png"
        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(cut_out)
        calls: list[dict[str, Any]] = []

        def _bgremove(src: str, dst: str) -> str:
            calls.append({"op": "bgremove", "src": src})
            return str(cut_out)

        def _upscale(src: str, scale: float, dst: str) -> str:
            calls.append({"op": "upscale", "src": src, "scale": scale})
            Image.new("RGB", (64, 64), (5, 5, 5)).save(dst)
            return dst

        engine = _engine(home, bgremove=_bgremove, upscale=_upscale)
        scene = _scene(title="Cut")
        engine.store.save(scene)
        out = engine.cutout(scene.id, cut_src)
        self.assertTrue(out["ok"], out)
        self.assertEqual(calls[0]["op"], "bgremove")
        out = engine.upscale(scene.id, cut_src, scale=2.0)
        self.assertTrue(out["ok"], out)
        self.assertEqual(calls[1]["scale"], 2.0)
        self.assertTrue(Path(out["path"]).exists())

    def test_video_chains_shots_and_records_provenance(self) -> None:
        home = _tmp_home(self)
        video = FakeVideo(home)
        engine = _engine(home, video=video)
        scene = _scene(title="Film")
        scene.set_location("Roof")
        scene.add_shot("One", prompt="dawn over the roof", mode="i2v",
                       duration_s=4.0)
        scene.add_shot("Two", prompt="the door opens", mode="motion",
                       duration_s=3.0)
        engine.store.save(scene)
        out = engine.video(scene.id)
        self.assertTrue(out["ok"], out)
        self.assertEqual(len(video.chains), 1)
        self.assertEqual(len(video.chains[0]), 2)
        self.assertEqual(video.chains[0][0]["prompt"], "dawn over the roof")
        self.assertEqual(video.chains[0][0]["duration_s"], 4.0)
        self.assertTrue(Path(out["path"]).exists())
        saved = engine.store.get(scene.id)
        assert saved is not None
        self.assertIn("video", saved.artifacts)
        self.assertEqual(saved.artifacts["video"]["shots"], 2)

    def test_video_talk_shot_uses_talking_head(self) -> None:
        home = _tmp_home(self)
        video = FakeVideo(home)
        engine = _engine(home, video=video)
        scene = _scene(title="Talk")
        scene.set_location("Studio")
        photo = _solid_png(home / "face.png", (64, 64), (200, 150, 100))
        scene.add_character("Mara", ref_image=photo)
        scene.add_shot("Say", prompt="Mara speaks", mode="talk",
                       duration_s=5.0, lines="Welcome to the studio.",
                       character="Mara")
        engine.store.save(scene)
        out = engine.video(scene.id)
        self.assertTrue(out["ok"], out)
        self.assertEqual(len(video.talks), 1)
        self.assertEqual(video.talks[0]["lines"], "Welcome to the studio.")
        self.assertEqual(video.talks[0]["photo"], photo)

    def test_video_talk_without_rung_names_the_wall(self) -> None:
        home = _tmp_home(self)
        engine = _engine(home)
        scene = _scene(title="TalkNoRung")
        scene.set_location("Studio")
        scene.add_shot("Say", prompt="x", mode="talk", duration_s=5.0,
                       lines="Hello.")
        engine.store.save(scene)
        out = engine.video(scene.id)
        self.assertFalse(out["ok"])
        self.assertIn("talk", out["error"].lower())
        self.assertIn("Say", out["error"])

    def test_video_no_shots_refused(self) -> None:
        home = _tmp_home(self)
        video = FakeVideo(home)
        engine = _engine(home, video=video)
        scene = _scene(title="Empty")
        engine.store.save(scene)
        out = engine.video(scene.id)
        self.assertFalse(out["ok"])
        self.assertIn("shot", out["error"].lower())

    def test_missing_scene_errors(self) -> None:
        home = _tmp_home(self)
        engine = _engine(home)
        for fn in (engine.compose_preview, engine.video):
            out = fn("no-such-id")
            self.assertFalse(out["ok"])
            self.assertIn("no-such-id", out["error"])
        out = engine.generate_image("no-such-id", prompt="x")
        self.assertFalse(out["ok"])


class CapabilityTests(unittest.TestCase):
    def test_capability_report_is_honest(self) -> None:
        home = _tmp_home(self)
        engine = _engine(home)
        report = engine.capability_report()
        self.assertIn("imggen", report)
        self.assertIn("videogen", report)
        self.assertIn("talking_head", report)
        self.assertIn("ffmpeg", report)
        for key in ("imggen", "videogen", "talking_head", "ffmpeg"):
            self.assertIn("available", report[key])
            self.assertIn("reason", report[key])
            self.assertIsInstance(report[key]["available"], bool)
            self.assertTrue(report[key]["reason"])
        # This box has no torch and no GPU: the neural rungs must say so,
        # never claim to be available.
        if not report["imggen"]["available"]:
            self.assertIn("torch", report["imggen"]["reason"].lower())


class ToolTests(unittest.TestCase):
    def _registry(self) -> Any:
        from nomorals.tools import artist as artist_tool
        from nomorals.tools import registry as tool_registry
        home = _tmp_home(self)
        reg = tool_registry.ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=home),
            db=None, router=None))
        artist_tool.register(reg)
        return reg

    def _call(self, reg: Any, **kw: Any) -> dict[str, Any]:
        return reg._tools["artist"].fn(**kw)

    def test_tool_registers(self) -> None:
        reg = self._registry()
        spec = reg.get("artist")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.capability, "owner")

    def test_tool_create_show_list_compose(self) -> None:
        reg = self._registry()
        created = self._call(reg, action="create", title="Tool scene")
        self.assertTrue(created["ok"], created)
        sid = created["id"]
        shown = self._call(reg, action="show", id=sid)
        self.assertTrue(shown["ok"])
        self.assertEqual(shown["scene"]["title"], "Tool scene")
        listed = self._call(reg, action="list")
        self.assertTrue(listed["ok"])
        self.assertEqual(len(listed["scenes"]), 1)
        composed = self._call(reg, action="compose", id=sid)
        self.assertTrue(composed["ok"], composed)
        self.assertTrue(Path(composed["path"]).exists())

    def test_tool_unknown_action_and_missing_scene(self) -> None:
        reg = self._registry()
        out = self._call(reg, action="teleport")
        self.assertFalse(out["ok"])
        out = self._call(reg, action="show", id="nope")
        self.assertFalse(out["ok"])
        self.assertIn("nope", out["error"])

    def test_tool_status_names_rungs(self) -> None:
        reg = self._registry()
        out = self._call(reg, action="status")
        self.assertTrue(out["ok"])
        self.assertIn("imggen", out["capabilities"])
        self.assertIn("videogen", out["capabilities"])

    def test_tool_generate_without_neural_is_honest(self) -> None:
        reg = self._registry()
        created = self._call(reg, action="create", title="NoNeural")
        out = self._call(reg, action="generate", id=created["id"],
                         prompt="a door", mode="text")
        self.assertFalse(out["ok"])
        self.assertIn("error", out)
        self.assertNotIn("path", out)


if __name__ == "__main__":
    unittest.main()
