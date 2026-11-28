"""Gapfix 164 — wire the stocked font library into the renderers
and trim the pack.

Every visible surface wears its policy family instead of the old
DejaVu door; measurement uses the face that is drawn; and the
pruner removes only unreferenced families from Devon's own stocked
pack — never a system font path — with a dry-run that lists exactly
what would go. Tests are failing-first: each one names the door
the render path must ask for.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import ImageFont

_DEJAVU = Path("/usr/share/fonts/truetype/dejavu")


def _family(font) -> str:
    """The family name a PIL font reports."""
    return str(font.getname()[0])


def _core_font_path(name: str) -> Path:
    """The vendored core-subset file for ``name``."""
    from nomorals.media.typography import fontlib
    from nomorals.media.typography.fontlib import FontLibrary

    lib = FontLibrary(core_dir=fontlib.core_fonts_dir())
    return Path(lib.resolve(name))


@contextmanager
def _recording(module, attr: str):
    """Patch ``module.attr`` with a wrapper that records every
    call (args + kwargs) and delegates to the real function."""
    original = getattr(module, attr)
    calls: list[tuple[tuple, dict]] = []

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    with mock.patch.object(module, attr, wrapper):
        yield calls


# ── makers: the card door ──────────────────────────────────────────


class TestMakersPolicyDoors(unittest.TestCase):
    def test_contexts_resolve_to_their_stocked_policy_family(self):
        from nomorals.social.chat import makers

        cases = {
            "quote": "Playfair Display",
            "quote_script": "Great Vibes",
            "motivation": "Montserrat",
            "meme": "Anton",
            "card": "Inter",
            "card_title": "Montserrat",
            "caption": "Inter",
            "poster": "Anton",
            "cad": "JetBrains Mono",
            "cover": "Anton",
        }
        for context, family in cases.items():
            with self.subTest(context=context):
                font = makers._font(36, context=context)
                self.assertEqual(_family(font), family)

    def test_bare_intent_still_resolves_as_before(self):
        # The legacy bare-intent door is byte-stable for old
        # callers (pinned by test_font_wiring): bare "text"
        # stays DejaVu Sans, bare "display" stays Noto Sans
        # Display Condensed.
        from nomorals.social.chat import makers

        self.assertEqual(_family(makers._font(36)), "DejaVu Sans")
        self.assertEqual(_family(makers._font(36, intent="display")),
                         "Noto Sans Display")


class TestMakersLegacyRenderer(unittest.TestCase):
    def _entries(self, text="Be the light you want to see."):
        return [{"name": "Ada", "text": text}]

    def test_quote_card_wears_quote_and_quote_script(self):
        from nomorals.social.chat import makers

        with _recording(makers, "_font") as calls:
            makers.render_card_png("quote", self._entries())
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("quote", contexts)
        self.assertIn("quote_script", contexts)
        self.assertNotIn("", contexts)

    def test_meme_card_wears_meme_and_caption(self):
        from nomorals.social.chat import makers

        with _recording(makers, "_font") as calls:
            makers.render_card_png("meme", self._entries())
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("meme", contexts)
        self.assertIn("caption", contexts)
        self.assertNotIn("", contexts)

    def test_duo_and_chat_skins_wear_card(self):
        from nomorals.social.chat import makers

        for style in ("duo", "ttc", "chat", "ff"):
            entries = self._entries()
            if style == "duo":
                entries = entries + [{"name": "Bo", "text": "And you."}]
            with self.subTest(style=style), \
                    _recording(makers, "_font") as calls:
                makers.render_card_png(style, entries)
            contexts = [kw.get("context", "") for _a, kw in calls]
            self.assertIn("card", contexts)
            self.assertNotIn("", contexts)

    def test_medallion_initials_wear_card_title(self):
        from nomorals.social.chat import makers

        with _recording(makers, "_font") as calls:
            makers.render_card_png("quote", self._entries())
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("card_title", contexts)


class TestMakersPlatformAndFeature(unittest.TestCase):
    def test_platform_header_labels_wear_card_title(self):
        from nomorals.social.chat import makers

        entries = [{"name": "Ada", "text": "Hello world",
                    "handle": "@ada", "meta": "2h", "likes": 5},
                   {"name": "Bo", "text": "Agreed.", "likes": 1}]
        with _recording(makers, "_font") as calls:
            makers.render_card_png("facebook", entries)
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("card_title", contexts)
        self.assertNotIn("", contexts)

    def test_platform_rail_and_counts_wear_card(self):
        from nomorals.social.chat import makers

        entries = [{"name": "Ada", "text": "Hello world",
                    "handle": "@ada", "meta": "2h", "likes": 5}]
        with _recording(makers, "_font") as calls:
            makers.render_card_png("tiktok", entries)
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("card", contexts)
        self.assertNotIn("", contexts)

    def test_feature_title_and_author_wear_caption(self):
        from nomorals.social.chat import makers

        entry = {"name": "Ada", "text": "Be the light.",
                 "title": "Onward"}
        with _recording(makers, "_font") as calls:
            makers.render_card_png("bigquote", [entry])
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("caption", contexts)
        self.assertNotIn("", contexts)

    def test_motivation_body_wears_the_motivation_policy(self):
        from nomorals.social.chat import makers

        entry = {"name": "Ada", "text": "Keep going."}
        with _recording(makers, "_font") as calls:
            makers.render_card_png("motivation", [entry])
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("motivation", contexts)


# ── layout organ: measure with the face you draw with ─────────────


class TestLayoutOrgan(unittest.TestCase):
    def test_default_still_measures_with_dejavu(self):
        # Pinned by the craftfonts lane: no context means the
        # layout organ's DejaVu door, widths unchanged.
        from nomorals.media.layout.compose import font_for, measure_text

        direct = ImageFont.truetype(
            str(_DEJAVU / "DejaVuSans.ttf"), 16)
        self.assertEqual(measure_text("Hello world", 16.0),
                         direct.getlength("Hello world"))
        self.assertEqual(_family(font_for(16.0)), "DejaVu Sans")

    def test_context_measures_with_the_policy_face(self):
        from nomorals.media.layout.compose import font_for, measure_text

        playfair = ImageFont.truetype(
            str(_core_font_path("Playfair Display")), 16)
        self.assertEqual(_family(font_for(16.0, context="quote")),
                         "Playfair Display")
        self.assertEqual(measure_text("Hello world", 16.0,
                                      context="quote"),
                         playfair.getlength("Hello world"))
        self.assertNotEqual(
            measure_text("Hello world", 16.0, context="quote"),
            measure_text("Hello world", 16.0))

    def test_compose_story_threads_the_context(self):
        from nomorals.media.layout import compose
        from nomorals.media.layout.model import CharRun, ParaFormat, Story

        story = Story(id="s1", paras=[ParaFormat(runs=[CharRun(
            "The quick brown fox jumps over the lazy dog")])])
        with _recording(compose, "measure_text") as calls:
            compose.compose_story(story, 200.0, 400.0, font_size=12.0,
                                  context="quote")
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("quote", contexts)

    def test_text_flows_threads_the_context(self):
        from nomorals.media.layout import render as layout_render
        from nomorals.media.layout.model import CharRun, ParaFormat, Story

        story = Story(id="s1", paras=[ParaFormat(runs=[CharRun(
            "hello world")])])
        item = SimpleNamespace(
            id="i1", story_id="s1",
            content_type=lambda: "text",
            text_options=lambda: SimpleNamespace(
                next_frame_id=None, columns=1, gutter=0.0),
            local_bounds=lambda: (0.0, 0.0, 200.0, 100.0))
        doc = SimpleNamespace(
            all_items=lambda: [item], stories={"s1": story})
        with _recording(layout_render, "compose_story") as calls:
            flows = layout_render._text_flows(
                doc, font_context="quote")
        self.assertIn("i1", flows)
        contexts = [kw.get("context", "")
                    for _a, kw in calls]
        self.assertIn("quote", contexts)

    def test_draw_text_threads_the_context(self):
        from PIL import Image, ImageDraw

        from nomorals.media.layout import render as layout_render
        from nomorals.media.layout.compose import ComposedLine
        from nomorals.media.layout.model import CharRun, ParaFormat, Story
        from nomorals.media.vector.geom import Affine

        story = Story(id="s1", paras=[ParaFormat(runs=[CharRun(
            "hello world", font_size=12.0)])])
        item = SimpleNamespace(
            id="i1", story_id="s1",
            content_type=lambda: "text",
            text_options=lambda: SimpleNamespace(
                next_frame_id=None, columns=1, gutter=0.0),
            local_bounds=lambda: (0.0, 0.0, 200.0, 100.0))
        doc = SimpleNamespace(
            all_items=lambda: [item], stories={"s1": story})
        image = Image.new("RGB", (200, 100), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        lines = [ComposedLine(text="hello world", y=12.0, height=14.0)]
        with _recording(layout_render, "font_for") as calls:
            layout_render._draw_text(draw, doc, item, Affine(),
                                     1.0, lines, font_context="quote")
        contexts = [kw.get("context", "") for _a, kw in calls]
        self.assertIn("quote", contexts)


# ── design organ: the node's named family ─────────────────────────


class TestDesignOrgan(unittest.TestCase):
    def test_measure_uses_the_named_family(self):
        from nomorals.media.design import layout as design_layout
        from nomorals.media.typography import get_service, pil_font_for

        # Inter is a variable face: the reference width comes
        # from the typography door itself (which pins the axes),
        # not a raw truetype load (which uses the default
        # instance and measures slightly wider).
        door = pil_font_for(get_service(), "text", 16, family="Inter",
                            weight=400)
        width, _height = design_layout.measure_text(
            "Hello world", font_family="Inter", font_size=16.0)
        self.assertEqual(width, door.getlength("Hello world"))
        dejavu = ImageFont.truetype(
            str(_DEJAVU / "DejaVuSans.ttf"), 16)
        self.assertNotEqual(width, dejavu.getlength("Hello world"))

    def test_measure_with_an_unknown_family_names_the_miss(self):
        from nomorals.media.design import layout as design_layout

        with self.assertLogs(
                "nomorals.media.design.layout", level="WARNING") as logs:
            width, _height = design_layout.measure_text(
                "Hello world", font_family="NoSuchFamily123",
                font_size=16.0)
        self.assertGreater(width, 0.0)
        self.assertTrue(any("NoSuchFamily123" in line
                            for line in logs.output))

    def test_draw_text_reads_the_nodes_font_family(self):
        from PIL import Image

        from nomorals.media.design import render as design_render

        node = {"type": "text", "characters": "Hello",
                "style": {"font_family": "Montserrat",
                          "font_size": 16.0, "font_style": "",
                          "alignment": "left", "line_height": {}},
                "fills": []}
        layer = Image.new("RGBA", (400, 60), (255, 255, 255, 255))
        # _draw_text touches only the node (via _draw_strokes),
        # never the document/layout the constructor wants.
        renderer = object.__new__(design_render._Renderer)
        with _recording(design_render, "_font") as calls:
            renderer._draw_text(layer, node, 400, 60)
        families = [kw.get("family") for _a, kw in calls]
        self.assertIn("Montserrat", families)


# ── animation, artist, cad, board: one context each ──────────────


class TestAnimationOrgan(unittest.TestCase):
    def test_unnamed_text_wears_the_poster_context(self):
        from nomorals.media.animation import render as anim_render

        self.assertEqual(_family(anim_render._font(24.0)), "Anton")

    def test_named_text_wears_its_named_family(self):
        from nomorals.media.animation import render as anim_render

        self.assertEqual(_family(anim_render._font(24.0, family="Inter")),
                         "Inter")

    def test_draw_shape_passes_the_items_font_through(self):
        from PIL import Image

        from nomorals.media.animation import render as anim_render
        from nomorals.media.animation.eval import DisplayItem

        item = DisplayItem(kind="text", text="hello", font_size=24.0,
                           font="Inter", text_color="#000000",
                           points=[[(0.0, 0.0)]])
        base = Image.new("RGBA", (200, 60), (255, 255, 255, 255))
        with _recording(anim_render, "_font") as calls:
            anim_render._draw_shape(base, item, None)
        families = [kw.get("family") for _a, kw in calls]
        self.assertIn("Inter", families)


class TestArtistOrgan(unittest.TestCase):
    def test_artist_labels_wear_the_cover_context(self):
        from nomorals.media.artist import compose as artist_compose

        self.assertEqual(_family(artist_compose._font(24)), "Anton")


class TestCadOrgan(unittest.TestCase):
    def test_cad_text_wears_the_mono_context(self):
        from nomorals.media.cad import render as cad_render

        self.assertEqual(_family(cad_render._font(12.0)),
                         "JetBrains Mono")


class TestBoardOrgan(unittest.TestCase):
    def test_board_text_wears_the_card_context(self):
        from nomorals.media.edit_engine import board

        self.assertEqual(_family(board._load_font(16)), "Inter")


# ── decks: the run's own typeface ─────────────────────────────────


class TestDecksOrgan(unittest.TestCase):
    def test_named_run_font_resolves(self):
        from nomorals.documents.decks import render as deck_render

        self.assertEqual(
            _family(deck_render._font(36, family="Inter")), "Inter")

    def test_unnamed_run_wears_the_card_context(self):
        from nomorals.documents.decks import render as deck_render

        self.assertEqual(_family(deck_render._font(36)), "Inter")

    def test_unknown_run_font_names_the_miss_and_still_draws(self):
        from nomorals.documents.decks import render as deck_render

        with self.assertLogs(
                "nomorals.documents.decks.render",
                level="WARNING") as logs:
            font = deck_render._font(36, family="Calibri")
        self.assertIsNotNone(font)
        self.assertGreater(font.getlength("Hello"), 0)
        self.assertTrue(any("Calibri" in line for line in logs.output))

    def test_render_passes_the_runs_font_to_the_door(self):
        from pathlib import Path

        from nomorals.documents.decks import render as deck_render
        from nomorals.documents.decks.model import DeckDocument

        fixtures = Path(__file__).resolve().parent / "fixtures" / "deckcraft"
        doc = DeckDocument.from_bytes(
            (fixtures / "simple.pptx").read_bytes())
        run = doc.slides[0].shapes[0].paragraphs[0].runs[0]
        run.font = "Inter"
        with _recording(deck_render, "_font") as calls:
            deck_render.render_slide(doc, 0)
        families = [kw.get("family") for _a, kw in calls]
        self.assertIn("Inter", families)

    def test_wrap_and_draw_measure_with_the_same_face(self):
        # The wrap pass and the draw pass must ask for the same
        # family, or the laid-out lines do not match the drawn
        # ones.
        from pathlib import Path

        from nomorals.documents.decks import render as deck_render
        from nomorals.documents.decks.model import DeckDocument

        fixtures = Path(__file__).resolve().parent / "fixtures" / "deckcraft"
        doc = DeckDocument.from_bytes(
            (fixtures / "simple.pptx").read_bytes())
        for slide in doc.slides:
            for shape in slide.shapes:
                for para in shape.paragraphs:
                    for run in para.runs:
                        run.font = "Montserrat"
        with _recording(deck_render, "_font") as calls:
            deck_render.render_slide(doc, 0)
        families = [kw.get("family") for _a, kw in calls]
        self.assertTrue(all(f == "Montserrat" for f in families),
                        families)


# ── corpus: the library first ─────────────────────────────────────


class TestCorpusOrgan(unittest.TestCase):
    def test_embedded_font_comes_from_the_stocked_library(self):
        from nomorals.documents import corpus
        from nomorals.media.typography.fontparse import parse_font_file

        path = corpus.find_embedded_font()
        self.assertIsNotNone(path)
        faces = parse_font_file(Path(path))
        self.assertTrue(faces)
        self.assertEqual(faces[0].family, "Inter")

    def test_env_override_still_wins(self):
        import os
        from unittest import mock

        from nomorals.documents import corpus

        dejavu = str(_DEJAVU / "DejaVuSans.ttf")
        with mock.patch.dict(os.environ, {"NM_CORPUS_FONT": dejavu}):
            self.assertEqual(corpus.find_embedded_font(), dejavu)

    def test_parity_doc_records_a_repo_relative_path(self):
        # The parity doc is committed: a vendored font must be
        # recorded relative to the repo root (stable across
        # checkouts), never as the absolute path of the tree
        # that happened to generate it.
        from nomorals.documents import corpus

        vendored = str(
            Path(corpus.__file__).resolve().parents[2]
            / "nomorals" / "media" / "typography" / "fonts"
            / "ofl" / "inter" / "Inter[opsz,wght].ttf")
        self.assertEqual(
            corpus._display_font_path(vendored),
            "nomorals/media/typography/fonts/ofl/inter/"
            "Inter[opsz,wght].ttf")
        host = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.assertEqual(corpus._display_font_path(host), host)


# ── the pruner: only unreferenced pack families go ──────────────


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_named_pack(root: Path, families, *, kind: str = "full") -> Path:
    """A synthetic pack whose families are named by the caller —
    the file bytes are host DejaVu faces, the manifest is
    truthful (names, rel paths, hashes, sizes)."""
    root.mkdir(parents=True, exist_ok=True)
    faces = []

    def add_family(name, category):
        slug = name.lower().replace(" ", "")
        rel = f"ofl/{slug}/{name.replace(' ', '')}-Regular.ttf"
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_DEJAVU / "DejaVuSans.ttf", dest)
        record = {"file": rel, "weight": 400, "style": "normal",
                  "variable": False, "sha256": _sha(dest),
                  "bytes": dest.stat().st_size, "scripts": ["Latin"]}
        license_rel = str(Path(rel).parent / "OFL.txt")
        (root / license_rel).write_text(
            "SIL Open Font License 1.1\n", encoding="utf-8")
        faces.append({
            "name": name, "category": category,
            "license": "OFL-1.1", "license_file": license_rel,
            "reserved_font_name": False,
            "dir": str(Path(rel).parent), "subsets": ["latin"],
            "faces": [record],
        })

    for name, category in families:
        add_family(name, category)
    manifest = {
        "format": 1, "kind": kind,
        "source": {"repo": "google/fonts", "commit": "testcommit"},
        "family_count": len(faces),
        "file_count": sum(len(f["faces"]) for f in faces),
        "total_bytes": sum(fa["bytes"] for f in faces
                           for fa in f["faces"]),
        "pending_families": 0 if kind == "full" else 1,
        "families": faces,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return root


def _stock_pack_into_home(tmp_path, families, *, kind: str = "full"):
    """Write the named pack straight into a fake home's
    library dir (the pruner's own root) and return the home."""
    home = tmp_path / "home"
    lib = home / "fonts" / "library"
    _write_named_pack(lib, families, kind=kind)
    return home


class TestFontPruner(unittest.TestCase):
    _FAMILIES = [("Inter", "Sans Serif"),
                 ("Playfair Display", "Serif"),
                 ("DejaVu Sans", "Sans Serif"),
                 ("Zzz Unused", "Sans Serif"),
                 ("Yyy Unused", "Display")]

    def test_referenced_set_covers_policy_pins_and_dejavu(self):
        from nomorals.media.typography.prune import referenced_families

        refs = referenced_families()
        for name in ("Inter", "Playfair Display", "Anton",
                     "Montserrat", "JetBrains Mono", "DejaVu Sans",
                     "Great Vibes", "Roboto"):
            self.assertIn(name, refs)
        self.assertNotIn("Zzz Unused", refs)

    def test_dry_run_lists_and_removes_nothing(self):
        import tempfile

        from nomorals.media.typography.prune import prune_library

        with tempfile.TemporaryDirectory() as tmp:
            home = _stock_pack_into_home(Path(tmp), self._FAMILIES)
            before = sorted(p.name for p in
                            (home / "fonts" / "library").rglob("*")
                            if p.is_file())
            report = prune_library(home=home, dry_run=True)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["removed_families"],
                             ["Yyy Unused", "Zzz Unused"])
            self.assertGreater(report["prunable_bytes"], 0)
            self.assertEqual(report["bytes_freed"], 0)
            after = sorted(p.name for p in
                           (home / "fonts" / "library").rglob("*")
                           if p.is_file())
            self.assertEqual(before, after)

    def test_prune_keeps_referenced_removes_the_rest(self):
        import tempfile

        from nomorals.media.typography.prune import prune_library

        with tempfile.TemporaryDirectory() as tmp:
            home = _stock_pack_into_home(Path(tmp), self._FAMILIES)
            lib = home / "fonts" / "library"
            report = prune_library(home=home)
            self.assertEqual(report["removed_families"],
                             ["Yyy Unused", "Zzz Unused"])
            self.assertGreater(report["bytes_freed"], 0)
            self.assertFalse((lib / "ofl" / "zzzunused").exists())
            self.assertFalse((lib / "ofl" / "yyyunused").exists())
            self.assertTrue((lib / "ofl" / "inter").is_dir())
            self.assertTrue((lib / "ofl" / "playfairdisplay").is_dir())
            manifest = json.loads(
                (lib / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["family_count"], 3)
            names = {f["name"] for f in manifest["families"]}
            self.assertEqual(names, {"Inter", "Playfair Display",
                                     "DejaVu Sans"})
            self.assertEqual(manifest["total_bytes"],
                             report["pack_bytes"])

    def test_pinned_family_is_referenced_and_kept(self):
        import tempfile

        from nomorals.core.style_pins import PinStore, set_pin_store
        from nomorals.media.typography.prune import prune_library, referenced_families

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = PinStore(path=tmp / "pins.json")
            store.set("card", font_family="Zzz Unused")
            old = set_pin_store(store)
            try:
                self.assertIn("Zzz Unused", referenced_families())
                home = _stock_pack_into_home(tmp, self._FAMILIES)
                report = prune_library(home=home)
                self.assertEqual(report["removed_families"],
                                 ["Yyy Unused"])
                lib = home / "fonts" / "library"
                self.assertTrue((lib / "ofl" / "zzzunused").is_dir())
            finally:
                set_pin_store(old)

    def test_never_touches_a_path_outside_the_pack(self):
        import tempfile

        from nomorals.media.typography.prune import prune_library

        with tempfile.TemporaryDirectory() as tmp:
            home = _stock_pack_into_home(Path(tmp), self._FAMILIES)
            system = Path(tmp) / "system-fonts"
            system.mkdir()
            (system / "arial.ttf").write_bytes(b"fake")
            report = prune_library(home=home)
            self.assertTrue((system / "arial.ttf").is_file())
            for name in report["removed_families"]:
                self.assertNotIn("arial", name.lower())

    def test_build_dir_goes_only_when_the_pack_is_whole(self):
        import tempfile

        from nomorals.media.typography.prune import prune_library

        with tempfile.TemporaryDirectory() as tmp:
            home = _stock_pack_into_home(Path(tmp), self._FAMILIES)
            build = home / "fonts" / "library.build"
            build.mkdir(parents=True)
            (build / "stock-state.json").write_text(
                json.dumps({"done": ["Inter"], "commit": "x"}),
                encoding="utf-8")
            (build / "ofl").mkdir()
            (build / "ofl" / "stale.ttf").write_bytes(b"stale")
            report = prune_library(home=home)
            self.assertTrue(report["build_dir_removed"])
            self.assertGreater(report["build_bytes_freed"], 0)
            self.assertFalse(build.exists())

    def test_build_dir_kept_while_a_build_is_pending(self):
        import tempfile

        from nomorals.media.typography.prune import prune_library

        with tempfile.TemporaryDirectory() as tmp:
            home = _stock_pack_into_home(
                Path(tmp), self._FAMILIES, kind="partial")
            build = home / "fonts" / "library.build"
            build.mkdir(parents=True)
            (build / "stock-state.json").write_text(
                json.dumps({"done": [], "commit": "x"}),
                encoding="utf-8")
            report = prune_library(home=home)
            self.assertFalse(report["build_dir_removed"])
            self.assertTrue(build.exists())

    def test_no_pack_manifest_is_an_error_not_a_crash(self):
        import tempfile

        from nomorals.media.typography.prune import prune_library

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            (home / "fonts" / "library").mkdir(parents=True)
            report = prune_library(home=home)
            self.assertTrue(report["error"])
            self.assertEqual(report["removed_families"], [])


class TestFontsPruneVerb(unittest.TestCase):
    def _runtime(self, operator=True):
        from nomorals.agents.partner.runtime_fonts import (
            RuntimeFontsMixin,
        )

        class _Stub(RuntimeFontsMixin):
            def _is_operator(self, message):
                return operator

        return _Stub()

    def test_prune_dry_run_lists_what_would_go(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = _stock_pack_into_home(
                Path(tmp), TestFontPruner._FAMILIES)
            with mock.patch.dict("os.environ", {
                    "NM_HOME": str(home), "NM_FONT_DIRS": ""}):
                runtime = self._runtime()
                reply = runtime._control_fonts("fonts", "prune", None)
            self.assertIn("Zzz Unused", reply)
            self.assertIn("Yyy Unused", reply)
            self.assertIn("Inter", reply)  # kept, named as kept
            lib = home / "fonts" / "library"
            self.assertTrue((lib / "ofl" / "zzzunused").is_dir())

    def test_prune_go_removes_the_unreferenced(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = _stock_pack_into_home(
                Path(tmp), TestFontPruner._FAMILIES)
            with mock.patch.dict("os.environ", {
                    "NM_HOME": str(home), "NM_FONT_DIRS": ""}):
                runtime = self._runtime()
                reply = runtime._control_fonts(
                    "fonts", "prune go", None)
            self.assertIn("Zzz Unused", reply)
            lib = home / "fonts" / "library"
            self.assertFalse((lib / "ofl" / "zzzunused").exists())
            self.assertTrue((lib / "ofl" / "inter").is_dir())

    def test_prune_is_owner_only(self):
        runtime = self._runtime(operator=False)
        reply = runtime._control_fonts(
            "fonts", "prune", SimpleNamespace())
        self.assertEqual(reply, "that one's just for the owner.")
