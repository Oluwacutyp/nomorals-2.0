"""EditStudio tests: pro session layer + generative instruction edits.

Covers:
- studio image ops (grade/filter/text/layers/collage/smart-crop/letterbox)
- EditStudio: op stack, undo/redo, project save/load, determinism,
  batch, compare exports, templates
- generative backends: interface contract (fake backend), no-backend
  error, HF param pass-through (stubbed client), masked region edits
- new NL intents: studio mechanical + generative (kept distinct)
- tool layer: studio_* tools, sandboxing
- `nm studio` CLI surface
- video assembly: filter-graph compilation assertions (no ffmpeg run),
  ffmpeg-gated end-to-end render
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, ImageDraw

from nomorals.media_edit import (
    EditStudio, build_template, list_templates, studio_presets,
    GenerativeBackend, GenerativeEditError, get_backend, backend_status,
    validate_ops,
)
from nomorals.media_edit import images
from nomorals.media_edit.images import MediaEditError
from nomorals.media_edit import intent as intent_mod
from nomorals.media_edit.intent import (
    AmbiguousInstructionError, parse_instruction)
from nomorals.media_edit import studio as studio_mod
from nomorals.media_edit import generate as generate_mod
from nomorals.tools.registry import ToolRegistry


# Absolute repo root (tests/ lives one level below it). Used as subprocess
# cwd / PYTHONPATH so the suite works from any checkout location.
REPO_ROOT = str(Path(__file__).resolve().parents[1])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_image(path, size=(1600, 1200), color=(60, 120, 200)):
    img = Image.new("RGB", size, color)
    # a bright blob so saliency-based smart crop has something to find
    d = ImageDraw.Draw(img)
    d.ellipse([size[0] * 0.6, size[1] * 0.6,
               size[0] * 0.9, size[1] * 0.9], fill=(230, 230, 240))
    img.save(path, quality=90)
    return path


def _make_clip(path, duration=2, size="320x240"):
    if not shutil.which("ffmpeg"):
        raise unittest.SkipTest("ffmpeg not available")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path), "-y"],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"fixture clip failed: {proc.stderr[:300]}")
    return path


class FakeBackend(GenerativeBackend):
    """Deterministic stand-in: records the call, marks the image."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def edit(self, image, instruction, *, mask=None, strength=0.75,
             seed=None, **kwargs):
        self.calls.append({"instruction": instruction, "mask": mask,
                           "strength": strength, "seed": seed,
                           "size": image.size, "kwargs": kwargs})
        out = image.convert("RGB").copy()
        ImageDraw.Draw(out).rectangle([0, 0, 15, 15], fill=(255, 0, 0))
        return out


# ---------------------------------------------------------------------------
# studio image ops
# ---------------------------------------------------------------------------

class StudioOpTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="studio_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.src = _make_image(self.root / "photo.jpg")

    def _render(self, ops):
        st = EditStudio(self.src)
        for op in ops:
            op = dict(op)
            st.op(op.pop("op"), **op)
        return st.render(out_dir=self.root / "out")

    def test_filter_preset_deterministic(self):
        a = self._render([{"op": "filter", "preset": "cinematic"}])
        b = self._render([{"op": "filter", "preset": "cinematic"}])
        self.assertEqual(Path(a["output"]).read_bytes(),
                         Path(b["output"]).read_bytes())
        # and it actually changes the image
        plain = EditStudio(self.src).render(out_dir=self.root / "out2")
        self.assertNotEqual(Path(a["output"]).read_bytes(),
                            Path(plain["output"]).read_bytes())

    def test_unknown_filter_preset(self):
        with self.assertRaises(MediaEditError):
            self._render([{"op": "filter", "preset": "not-a-preset"}])

    def test_grade_warms(self):
        img = images.load_image(self.src)
        try:
            warm = studio_mod.op_grade(img, temperature=1500)
            import statistics
            mr0 = statistics.mean(img.convert("RGB").split()[0].getdata())
            mr1 = statistics.mean(warm.convert("RGB").split()[0].getdata())
            mb0 = statistics.mean(img.convert("RGB").split()[2].getdata())
            mb1 = statistics.mean(warm.convert("RGB").split()[2].getdata())
            self.assertGreater(mr1, mr0)   # red channel up
            self.assertLess(mb1, mb0)      # blue channel down
        finally:
            img.close()

    def test_letterbox(self):
        r = self._render([{"op": "letterbox", "aspect": "21:9"}])
        info = images.image_probe(r["output"])
        self.assertEqual((info["width"], info["height"]), (1600, 686))

    def test_smart_crop_aspect(self):
        r = self._render([{"op": "smart_crop", "aspect": "1:1",
                           "mode": "saliency"}])
        info = images.image_probe(r["output"])
        self.assertEqual(info["width"], info["height"])

    def test_text_layer_renders(self):
        r = self._render([{"op": "text_layer", "text": "Hello Studio",
                           "position": "bottom", "size": 64}])
        self.assertNotEqual(Path(r["output"]).read_bytes(),
                            Path(self.src).read_bytes())

    def test_layer_composite_pixel(self):
        blue = self.root / "blue.png"
        Image.new("RGB", (200, 200), (0, 0, 255)).save(blue)
        st = EditStudio(self.src)
        st.layers([{"type": "image", "path": str(blue),
                    "position": "top-left", "scale": 1.0, "margin": 0}])
        r = st.render(out_dir=self.root / "out")
        out = Image.open(r["output"]).convert("RGB")
        try:
            r_, g_, b_ = out.getpixel((5, 5))
            self.assertGreater(b_, 200)  # blue layer (JPEG)
            self.assertLess(r_, 60)
            r2, g2, b2 = out.getpixel((1500, 1100))
            self.assertFalse(b2 > 200 and r2 < 60)
        finally:
            out.close()

    def test_blend_mode_multiply_darkens(self):
        white = self.root / "white.png"
        Image.new("RGB", (1600, 1200), (255, 255, 255)).save(white)
        st = EditStudio(self.src)
        st.layers([{"type": "image", "path": str(white),
                    "position": "top-left", "scale": 1.0, "margin": 0,
                    "blend": "multiply"}])
        r = st.render(out_dir=self.root / "out")
        # multiply with white == identity → same as source pixels
        a = Image.open(r["output"]).convert("RGB")
        b = Image.open(self.src).convert("RGB")
        try:
            self.assertEqual(list(a.getdata())[:100], list(b.getdata())[:100])
        finally:
            a.close()
            b.close()

    def test_collage_grid(self):
        imgs = [str(_make_image(self.root / f"c{i}.jpg")) for i in range(3)]
        r = self._render([{"op": "collage", "images": imgs,
                           "template": "grid", "cols": 2, "gap": 8}])
        info = images.image_probe(r["output"])
        self.assertGreater(info["width"], 0)

    def test_compare_modes(self):
        st = EditStudio(self.src)
        st.filter("noir")
        for mode in ("side-by-side", "split", "stacked"):
            r = st.compare(mode=mode, out_dir=self.root / "cmp")
            self.assertTrue(Path(r["output"]).exists(), mode)
        r = st.compare(mode="html", out_dir=self.root / "cmp")
        html = Path(r["output"]).read_text()
        self.assertIn("range", html)

    def test_render_never_overwrites_source(self):
        before = Path(self.src).read_bytes()
        st = EditStudio(self.src)
        st.filter("vintage").letterbox()
        st.render(out_dir=self.root / "out")
        self.assertEqual(Path(self.src).read_bytes(), before)


# ---------------------------------------------------------------------------
# EditStudio session mechanics
# ---------------------------------------------------------------------------

class StudioSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sess_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.src = _make_image(self.root / "photo.jpg")

    def test_undo_redo(self):
        st = EditStudio(self.src, name="t")
        st.filter("cinematic").letterbox("21:9")
        self.assertEqual(len(st.ops), 2)
        undone = st.undo()
        self.assertEqual(undone["op"], "letterbox")
        self.assertEqual(len(st.ops), 1)
        redone = st.redo()
        self.assertEqual(redone["op"], "letterbox")
        self.assertEqual(len(st.ops), 2)
        self.assertIsNone(EditStudio(self.src).undo())
        self.assertIsNone(EditStudio(self.src).redo())

    def test_new_op_clears_redo(self):
        st = EditStudio(self.src)
        st.filter("a").filter("b")
        st.undo()
        st.filter("c")
        self.assertIsNone(st.redo())
        self.assertEqual([o["op"] for o in st.ops],
                         ["filter", "filter"])
        self.assertEqual(st.ops[1]["params"]["preset"], "c")

    def test_project_save_load_roundtrip(self):
        st = EditStudio(self.src, name="proj")
        st.filter("cinematic", strength=0.8)
        st.generative_edit("make it sunset", strength=0.6, seed=3,
                           mask=[10, 10, 100, 100])
        st.text("title", position="top")
        proj = self.root / "proj.studio.json"
        st.save_project(proj)
        data = json.loads(proj.read_text())
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["source"], str(self.src))
        gen = [o for o in data["ops"] if o["op"] == "generative_edit"][0]
        self.assertEqual(gen["params"]["instruction"], "make it sunset")
        self.assertEqual(gen["params"]["mask"], [10, 10, 100, 100])
        back = EditStudio.load_project(proj)
        self.assertEqual(back.ops, st.ops)
        self.assertEqual(back.name, "proj")

    def test_load_bad_project(self):
        bad = self.root / "bad.json"
        bad.write_text("{not json")
        with self.assertRaises(MediaEditError):
            EditStudio.load_project(bad)
        old = self.root / "old.json"
        old.write_text(json.dumps({"version": 99, "ops": []}))
        with self.assertRaises(MediaEditError):
            EditStudio.load_project(old)

    def test_generative_edit_roundtrip_with_fake_backend(self):
        fake = FakeBackend()
        st = EditStudio(self.src)
        st.generative_edit("make it sunset", strength=0.5, seed=7)
        proj = self.root / "g.studio.json"
        st.save_project(proj)
        back = EditStudio.load_project(proj)
        with mock.patch.object(generate_mod, "get_backend",
                               return_value=fake):
            r = back.render(out_dir=self.root / "out")
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["instruction"], "make it sunset")
        self.assertEqual(call["strength"], 0.5)
        self.assertEqual(call["seed"], 7)
        self.assertIsNone(call["mask"])
        out = Image.open(r["output"]).convert("RGB")
        try:
            r_, g_, b_ = out.getpixel((2, 2))
            self.assertGreater(r_, 200)  # fake backend's red mark (JPEG)
            self.assertLess(g_, 80)
        finally:
            out.close()

    def test_generative_edit_mask_box_passthrough(self):
        fake = FakeBackend()
        st = EditStudio(self.src)
        st.generative_edit("add a bird", mask=[10, 20, 300, 400])
        with mock.patch.object(generate_mod, "get_backend",
                               return_value=fake):
            st.render(out_dir=self.root / "out")
        mask = fake.calls[0]["mask"]
        self.assertIsNotNone(mask)  # normalized to an L image by the op
        self.assertEqual(mask.size, (1600, 1200))
        self.assertEqual(mask.mode, "L")

    def test_undo_redo_generative_op(self):
        st = EditStudio(self.src)
        st.filter("noir").generative_edit("make it sunset")
        st.undo()
        self.assertEqual([o["op"] for o in st.ops], ["filter"])
        st.redo()
        self.assertEqual([o["op"] for o in st.ops],
                         ["filter", "generative_edit"])


# ---------------------------------------------------------------------------
# generative backends
# ---------------------------------------------------------------------------

class GenerativeBackendTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gen_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.img = Image.new("RGB", (64, 48), (10, 20, 30))

    def tearDown(self):
        self.img.close()

    def test_fake_backend_contract(self):
        fake = FakeBackend()
        out = fake.edit(self.img, "make it sunset", mask=None,
                        strength=0.5, seed=42)
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["instruction"], "make it sunset")
        self.assertEqual(call["strength"], 0.5)
        self.assertEqual(call["seed"], 42)
        self.assertIsNone(call["mask"])
        self.assertEqual(out.size, (64, 48))

    def test_off_backend_error_names_env(self):
        with mock.patch.dict(os.environ, {"MEDIA_GEN_BACKEND": "off"}):
            with self.assertRaises(GenerativeEditError) as cm:
                get_backend()
        self.assertIn("MEDIA_GEN_BACKEND", str(cm.exception))

    def test_unknown_backend_error(self):
        with mock.patch.dict(os.environ, {"MEDIA_GEN_BACKEND": "nope"}):
            with self.assertRaises(GenerativeEditError):
                get_backend()

    def test_auto_with_nothing_installed_errors_honestly(self):
        # hide both optional deps: auto must raise, never fake an edit
        with mock.patch.dict(os.environ, {"MEDIA_GEN_BACKEND": "auto"}), \
             mock.patch.dict("sys.modules",
                             {"huggingface_hub": None,
                              "diffusers": None, "torch": None}), \
             mock.patch.object(generate_mod.DiffusersBackend,
                               "available", staticmethod(lambda: False)):
            with self.assertRaises(GenerativeEditError) as cm:
                get_backend()
        self.assertIn("HF_TOKEN", str(cm.exception))

    def test_hf_backend_missing_package(self):
        with mock.patch.dict("sys.modules", {"huggingface_hub": None}):
            be = generate_mod.HFInferenceBackend(model="m/x")
            with self.assertRaises(GenerativeEditError) as cm:
                be.edit(self.img, "make it sunset")
        self.assertIn("huggingface_hub", str(cm.exception))

    def test_hf_backend_passes_params(self):
        seen = {}

        class StubClient:
            def __init__(self, **kwargs):
                seen["client_kwargs"] = kwargs

            def image_to_image(self, image, prompt=None, model=None,
                               **kwargs):
                seen["prompt"] = prompt
                seen["model"] = model
                seen["params"] = kwargs
                return Image.new("RGB", image.size, (1, 2, 3))

        stub = SimpleNamespace(InferenceClient=StubClient)
        with mock.patch.dict("sys.modules", {"huggingface_hub": stub}):
            be = generate_mod.HFInferenceBackend(model="black-forest-labs/x",
                                                provider="fal-ai",
                                                token="hf_test")
            out = be.edit(self.img, "make it sunset", strength=0.6, seed=9)
        self.assertEqual(seen["prompt"], "make it sunset")
        self.assertEqual(seen["model"], "black-forest-labs/x")
        self.assertEqual(seen["params"]["strength"], 0.6)
        self.assertEqual(seen["params"]["seed"], 9)
        self.assertEqual(seen["client_kwargs"]["provider"], "fal-ai")
        self.assertEqual(seen["client_kwargs"]["api_key"], "hf_test")
        self.assertEqual(out.size, (64, 48))

    def test_hf_backend_mask_composites_region(self):
        class StubClient:
            def __init__(self, **kwargs):
                pass

            def image_to_image(self, image, prompt=None, model=None,
                               **kwargs):
                return Image.new("RGB", image.size, (200, 0, 0))

        stub = SimpleNamespace(InferenceClient=StubClient)
        with mock.patch.dict("sys.modules", {"huggingface_hub": stub}):
            be = generate_mod.HFInferenceBackend(model="m/x", token="t")
            out = be.edit(self.img, "do it", mask=[0, 0, 32, 48])
        # left half regenerated (red), right half original
        self.assertEqual(out.convert("RGB").getpixel((4, 4))[0], 200)
        self.assertEqual(out.convert("RGB").getpixel((60, 4)), (10, 20, 30))

    def test_hf_backend_empty_instruction(self):
        be = generate_mod.HFInferenceBackend(model="m/x")
        with self.assertRaises(GenerativeEditError):
            be.edit(self.img, "   ")

    def test_diffusers_unavailable_here(self):
        if generate_mod.DiffusersBackend.available():
            self.skipTest("diffusers installed here")
        with mock.patch.dict(os.environ, {"MEDIA_GEN_BACKEND": "diffusers"}):
            with self.assertRaises(GenerativeEditError):
                get_backend()

    def test_backend_status_makes_no_calls(self):
        status = backend_status()
        for key in ("hf_installed", "hf_token_set", "hf_model",
                    "diffusers_available", "selected"):
            self.assertIn(key, status, key)

    def test_op_registered_in_allowlist(self):
        import nomorals.media_edit.studio  # noqa: F401 (registers ops)
        from nomorals.media_edit.images import OP_ALLOWLIST
        self.assertIn("generative_edit", OP_ALLOWLIST)
        ops = validate_ops([{"op": "generative_edit",
                             "instruction": "make it sunset"}])
        self.assertEqual(ops[0]["op"], "generative_edit")


# ---------------------------------------------------------------------------
# intents: studio mechanical + generative
# ---------------------------------------------------------------------------

class StudioIntentTests(unittest.TestCase):
    def test_generative_intents(self):
        for text in ("make a bird sit on the tree",
                     "put him in a grand room",
                     "make it sunset",
                     "change the car to red",
                     "turn day into night",
                     "remove the trash can",
                     "add a lighthouse on the cliff"):
            r = parse_instruction(text, kind="image")
            self.assertEqual(r.kind, "image", text)
            self.assertEqual(r.ops[0]["op"], "generative_edit", text)
            self.assertEqual(r.ops[0]["instruction"], text, text)

    def test_mechanical_intents_not_stolen(self):
        self.assertEqual(parse_instruction(
            "make it square", kind="image").ops[0]["op"], "crop")
        self.assertEqual(parse_instruction(
            "make a thumbnail", kind="image").ops[0]["op"], "thumbnail")
        r = parse_instruction('add text "hello"', kind="image")
        self.assertNotEqual(r.ops[0]["op"], "generative_edit")
        r = parse_instruction("make a gif", kind="video")
        self.assertEqual(r.kind, "video")

    def test_studio_mechanical_intents(self):
        r = parse_instruction("give it a cinematic look", kind="image")
        self.assertEqual(r.ops[0]["op"], "filter")
        self.assertEqual(r.ops[0]["preset"], "cinematic")
        r = parse_instruction("make it warmer", kind="image")
        self.assertEqual(r.ops[0]["op"], "grade")
        self.assertGreater(r.ops[0]["temperature"], 0)
        r = parse_instruction("add a letterbox", kind="image")
        self.assertEqual(r.ops[0]["op"], "letterbox")
        r = parse_instruction("smart crop to 4:5", kind="image")
        self.assertEqual(r.ops[0]["op"], "smart_crop")
        r = parse_instruction("crop to 16:9", kind="image")
        self.assertEqual(r.ops[0]["op"], "crop")  # mechanical wins
        r = parse_instruction("title: My Day", kind="image")
        self.assertEqual(r.ops[0]["op"], "text_layer")

    def test_gibberish_still_ambiguous(self):
        with self.assertRaises(AmbiguousInstructionError):
            parse_instruction("blorple the wumpus", kind="image")

    def test_parse_auto_video_wins(self):
        r = parse_instruction("make a gif", kind="auto")
        self.assertEqual(r.kind, "video")


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="tmpl_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.bg = _make_image(self.root / "bg.jpg")
        self.p1 = _make_image(self.root / "p1.jpg", color=(200, 60, 60))
        self.p2 = _make_image(self.root / "p2.jpg", color=(60, 200, 60))

    def test_list_templates(self):
        tmpls = list_templates()
        for name in ("podcast-clip", "quote-card", "product-showcase",
                     "meme", "slideshow"):
            self.assertIn(name, tmpls)
            self.assertIn("params", tmpls[name])

    def test_build_meme(self):
        st = build_template("meme", image=str(self.bg), top="TOP",
                            bottom="BOTTOM")
        self.assertEqual(st.ops[0]["op"], "meme")
        r = st.render(out_dir=self.root / "out")
        self.assertTrue(Path(r["output"]).exists())

    def test_build_quote_card(self):
        st = build_template("quote-card", background=str(self.bg),
                            quote="Be the change", author="Someone")
        kinds = [o["op"] for o in st.ops]
        self.assertIn("filter", kinds)
        self.assertIn("text_layer", kinds)
        r = st.render(out_dir=self.root / "out")
        self.assertTrue(Path(r["output"]).exists())

    def test_build_product_showcase(self):
        st = build_template("product-showcase",
                            images=[str(self.bg), str(self.p1), str(self.p2)],
                            title="Widget", price="$9.99")
        kinds = [o["op"] for o in st.ops]
        self.assertIn("collage", kinds)
        self.assertIn("text_layer", kinds)
        r = st.render(out_dir=self.root / "out")
        self.assertTrue(Path(r["output"]).exists())

    def test_build_podcast_clip(self):
        st = build_template("podcast-clip", source="clip.mp4",
                            title="Ep 42", show="My Show")
        kinds = [o["op"] for o in st.ops]
        self.assertIn("v_lower_third", kinds)
        self.assertIn("v_title", kinds)
        self.assertIn("v_export", kinds)
        self.assertEqual(st.ops[-1]["params"]["preset"], "social-vertical")

    def test_build_slideshow(self):
        st = build_template("slideshow",
                            images=[str(self.bg), str(self.p1)],
                            title="Trip", bgm="music.mp3")
        self.assertEqual(st._resolve_kind(), "video")
        parts = st._collect_video_ops()
        self.assertEqual(len(parts["segments"]), 2)
        self.assertEqual(len(parts["transitions"]), 2)  # normalized
        self.assertIsNotNone(parts["title"])

    def test_unknown_template(self):
        with self.assertRaises(MediaEditError):
            build_template("nope", image="x.jpg")

    def test_missing_params(self):
        with self.assertRaises(MediaEditError):
            build_template("meme")

    def test_studio_presets_listing(self):
        p = studio_presets()
        self.assertIn("cinematic", p["filters"])
        self.assertIn("fade", p["transitions"])
        self.assertIn("social-vertical", p["export_presets"])
        self.assertIn("meme", p["templates"])


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

class BatchTests(unittest.TestCase):
    def test_batch_applies_project(self):
        root = Path(tempfile.mkdtemp(prefix="batch_"))
        self.addCleanup(shutil.rmtree, root, True)
        for i in range(3):
            _make_image(root / f"img{i}.jpg")
        st = EditStudio(str(root / "img0.jpg"), name="b")
        st.filter("noir")
        proj = root / "b.studio.json"
        st.save_project(proj)
        results = EditStudio.batch(root, proj, pattern="img*.jpg",
                                   out_dir=root / "out")
        self.assertEqual(len(results), 3)
        self.assertTrue(all("output" in r for r in results))
        self.assertTrue(all(Path(r["output"]).exists() for r in results))


# ---------------------------------------------------------------------------
# video assembly (graph compile — no ffmpeg run; e2e gated on ffmpeg)
# ---------------------------------------------------------------------------

class VideoCompileTests(unittest.TestCase):
    def setUp(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not available")
        self.root = Path(tempfile.mkdtemp(prefix="vcomp_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.c1 = _make_clip(self.root / "a.mp4")
        self.c2 = _make_clip(self.root / "b.mp4")

    def test_graph_compilation(self):
        st = EditStudio(str(self.c1), kind="video", name="v")
        st.add_clip(str(self.c2))
        st.transition("fade", duration=0.5)
        st.title_card("Hello", duration=2.0)
        st.lower_third("Guest", start=3.0, duration=2.0)
        st.speed(2.0, start=3.0, end=4.5)
        st.chapter("intro", at=0.0, end=2.0)
        st.export("social-vertical")
        parts = st._collect_video_ops()
        self.assertEqual(len(parts["segments"]), 2)
        self.assertEqual(len(parts["transitions"]), 2)  # title + 1 → 2 xfades
        work = self.root / "work"
        work.mkdir()
        compiled = studio_mod.compile_video(
            parts["segments"], transitions=parts["transitions"],
            title=parts["title"], lower_thirds=parts["lower_thirds"],
            speed_ramps=parts["speed_ramps"], chapters=parts["chapters"],
            export=parts["export"], work_dir=work)
        fc = compiled["filter_complex"]
        self.assertIn("xfade", fc)
        self.assertIn("acrossfade", fc)
        # 2s clips, 0.5s fade → offsets 1.5 then (2+2-0.5)-0.5=3.0... first:
        self.assertIn("offset=1.5", fc)
        self.assertIn("scale=1080:1920", fc)  # social-vertical
        self.assertIn("overlay=", fc)  # lower-third PNG composited
        self.assertIn("between(t,3.00,5.00)", fc)  # lower-third timing
        self.assertIn("atempo=2.0", fc)  # speed ramp
        # chapters file
        chap = Path([i for i in compiled["inputs"]
                     if str(i).endswith(".txt")][0])
        self.assertIn("intro", chap.read_text())
        self.assertIn("-map_chapters", compiled["maps"])
        # audio duck absent → amix absent
        self.assertNotIn("sidechaincompress", fc)

    def test_duck_adds_sidechain(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not available")
        bgm = self.root / "bgm.mp3"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=220:duration=4", "-c:a", "libmp3lame",
             str(bgm), "-y"], check=True, timeout=60)
        st = EditStudio(str(self.c1), kind="video")
        st.duck(str(bgm), amount_db=12.0)
        parts = st._collect_video_ops()
        work = self.root / "work2"
        work.mkdir()
        compiled = studio_mod.compile_video(
            parts["segments"], duck=parts["duck"], export="web-optimized",
            work_dir=work)
        self.assertIn("sidechaincompress", compiled["filter_complex"])

    def test_render_end_to_end(self):
        st = EditStudio(str(self.c1), kind="video", name="e2e")
        st.add_clip(str(self.c2))
        st.transition("fade", duration=0.5)
        st.title_card("E2E", duration=1.5)
        st.export("web-optimized")
        r = st.render(out_dir=self.root / "out", wait=True, timeout=300)
        self.assertEqual(r["status"], "done", r)
        out = Path(r["output_ref"])
        self.assertTrue(out.exists())
        from nomorals.media_edit import videos as videos_mod
        info = videos_mod.media_probe_any(out)
        self.assertEqual(info["kind"], "video")
        self.assertGreater(info["duration"], 3.0)

    def test_render_speed_ramp_end_to_end(self):
        # R22 live check: the timeline speed-ramp filter graph (split →
        # retime → concat, R20's "output survival" fix) must survive an
        # actual ffmpeg run, not just graph compilation. Timeline: 1.0s
        # title + 2×2s clips with 0.5s fade = 4.5s; 2× ramp over [1.5, 3.0]
        # shrinks 1.5s → 0.75s, so output lands at ~3.75s.
        st = EditStudio(str(self.c1), kind="video", name="ramp-e2e")
        st.add_clip(str(self.c2))
        st.transition("fade", duration=0.5)
        st.title_card("Ramp", duration=1.0)
        st.speed(2.0, start=1.5, end=3.0)
        st.export("web-optimized")
        r = st.render(out_dir=self.root / "out-ramp", wait=True, timeout=300)
        self.assertEqual(r["status"], "done", r)
        out = Path(r["output_ref"])
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)
        from nomorals.media_edit import videos as videos_mod
        info = videos_mod.media_probe_any(out)
        self.assertEqual(info["kind"], "video")
        types = {s.get("type") for s in info.get("streams", [])}
        self.assertIn("video", types)
        self.assertIn("audio", types)  # atempo chain kept audio alive
        self.assertGreater(info["duration"], 2.0)
        self.assertLess(info["duration"], 4.5)  # ramp shortened the timeline
        self.assertEqual((info.get("width"), info.get("height")),
                         (1280, 720))  # web-optimized preset

    def test_render_async_job(self):
        st = EditStudio(str(self.c1), kind="video", name="async")
        st.export("web-optimized")
        r = st.render(out_dir=self.root / "out2")
        self.assertIn("job_id", r)
        from nomorals.media_edit.jobs import get_manager
        done = get_manager().wait(r["job_id"], timeout=300)
        self.assertEqual(done["status"], "done", done)


# ---------------------------------------------------------------------------
# tool layer
# ---------------------------------------------------------------------------

class StudioToolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="stool_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        _make_image(self.root / "photo.jpg")
        context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(self.root)),
            db=None, router=None)
        self.registry = ToolRegistry(context)
        self.registry.register_builtins()

    def _call(self, name, **kwargs):
        outcome = self.registry.call(name, actor="test", **kwargs)
        self.assertTrue(outcome.ok, f"{name} failed: {outcome.error}")
        return outcome.value

    def test_studio_presets(self):
        p = self._call("studio_presets")
        self.assertIn("cinematic", p["filters"])
        self.assertIn("meme", p["templates"])

    def test_studio_run_filter(self):
        r = self._call("studio_run", source="photo.jpg",
                       ops=[{"op": "filter", "preset": "noir"}])
        self.assertTrue(Path(r["output"]).exists())
        self.assertNotIn(str(self.root / "photo.jpg"), r["output"])

    def test_studio_run_generative_off_errors(self):
        with mock.patch.dict(os.environ, {"MEDIA_GEN_BACKEND": "off"}):
            outcome = self.registry.call(
                "studio_run", actor="test", source="photo.jpg",
                ops=[{"op": "generative_edit",
                      "instruction": "make it sunset"}])
        self.assertFalse(outcome.ok)
        self.assertIn("MEDIA_GEN_BACKEND", str(outcome.error))

    def test_studio_template_meme(self):
        r = self._call("studio_template", template="meme",
                       params={"image": "photo.jpg", "top": "HI"})
        self.assertIn("meme", r["template"])
        self.assertTrue(Path(r["output"]).exists())

    def test_studio_project_cycle(self):
        saved = self._call("studio_project", action="save",
                           source="photo.jpg",
                           project_path="p.studio.json",
                           ops=[{"op": "filter", "preset": "vintage"}])
        self.assertTrue(Path(saved["saved"]).exists())
        desc = self._call("studio_project", action="describe",
                          project_path="p.studio.json")
        self.assertIn("vintage", desc["describe"])
        r = self._call("studio_project", action="render",
                       project_path="p.studio.json")
        self.assertTrue(Path(r["output"]).exists())

    def test_studio_compare(self):
        r = self._call("studio_compare", source="photo.jpg",
                       ops=[{"op": "filter", "preset": "noir"}],
                       mode="side-by-side")
        info = images.image_probe(r["output"])
        self.assertEqual(info["width"], 3204)  # 1600 + 1600 + 2×2px divider
        r2 = self._call("studio_compare", source="photo.jpg",
                        ops=[{"op": "filter", "preset": "noir"}],
                        mode="html")
        self.assertTrue(r2["output"].endswith(".html"))

    def test_studio_gen_status(self):
        s = self._call("studio_gen_status")
        self.assertIn("selected", s)

    def test_studio_run_sandbox_escape(self):
        outcome = self.registry.call(
            "studio_run", actor="test", source="/etc/passwd",
            ops=[{"op": "filter", "preset": "noir"}])
        self.assertFalse(outcome.ok)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

class StudioCLITests(unittest.TestCase):
    def setUp(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not available")
        self.home = Path(tempfile.mkdtemp(prefix="scli_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        ws = self.home / ".nomorals" / "workspace"
        ws.mkdir(parents=True)
        _make_image(ws / "photo.jpg")
        self.env = dict(os.environ, HOME=str(self.home),
                        PYTHONPATH=REPO_ROOT)

    def _nm(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "nomorals", "studio", *args],
            capture_output=True, text=True, timeout=180,
            cwd=REPO_ROOT, env=self.env)

    def test_cli_presets(self):
        proc = self._nm("presets")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("cinematic", proc.stdout)

    def test_cli_filter(self):
        proc = self._nm("filter", "photo.jpg", "noir")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("wrote", proc.stdout)
        out = list((self.home / ".nomorals" / "workspace" / "edited")
                   .glob("photo-*"))
        self.assertTrue(out, "edited artifact missing")

    def test_cli_gen_status(self):
        proc = self._nm("gen-status")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("backend", proc.stdout)

    def test_cli_ai_off_errors_cleanly(self):
        env = dict(self.env, MEDIA_GEN_BACKEND="off")
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals", "studio", "ai", "photo.jpg",
             "make it sunset"],
            capture_output=True, text=True, timeout=180,
            cwd=REPO_ROOT, env=env)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("MEDIA_GEN_BACKEND", proc.stderr)


if __name__ == "__main__":
    unittest.main()
