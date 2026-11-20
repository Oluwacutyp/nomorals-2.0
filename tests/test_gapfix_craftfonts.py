"""Craft-Fonts port (census §3A.21) — storytold/craft-fonts in Devon.

Devon already has a typography organ (nomorals/media/typography/):
a catalogue parsed from the real font files on the host, a stocked
library (pack / core / local tiers), semantic selection and a
service facade. What it lacks is the craft-fonts half: the OFL
font catalog for the scripts Devon's stocked core does not cover
(Japanese, Simplified Chinese, Korean, Arabic, Thai, Georgian,
Hebrew, UI symbols), the 8-field manifest format with its checkout
loading (CRAFT_FONTS_DIR), an install path from a manifest into
the stocked library, and a build/embed payload. This test pins
that port — and the layout organ's font door unified onto the
typography service instead of its private DejaVu ladder.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "craftfonts"
_HEBREW_TTF = _FIXTURES / "noto-sans-hebrew" / "NotoSansHebrew-Regular.ttf"
_HEBREW_OFL = _FIXTURES / "noto-sans-hebrew" / "OFL.txt"
_CALADEA_TTF = _FIXTURES / "caladea" / "Caladea-Regular.ttf"
_CALADEA_OFL = _FIXTURES / "caladea" / "OFL.txt"
_HEBREW_SHA = "671951828bd5c95db818e5bb12dcea2d0c0dda00311888522be061ee6835125e"
_CALADEA_SHA = "f1e899278b7b4491aba5b6a8253c4b04c050cc59b21865be5c37559a775153cd"

_MANIFEST_TWO_ROWS = (
    "Noto Sans Hebrew | Regular | fonts/noto-sans-hebrew/"
    "NotoSansHebrew-Regular.ttf | Hebr | OFL-1.1 | "
    "fonts/noto-sans-hebrew/OFL.txt | " + _HEBREW_SHA + " | "
    "https://github.com/notofonts/hebrew/releases/tag/NotoSansHebrew-v3.001\n"
    "Caladea | Regular | fonts/caladea/Caladea-Regular.ttf | Latn | "
    "OFL-1.1 | fonts/caladea/OFL.txt | " + _CALADEA_SHA + " | "
    "https://github.com/google/fonts/tree/7085eb89a950e85db5b166b7a58d414544b4140c/ofl/caladea\n"
)


def _fixture_checkout(directory: Path) -> Path:
    """A craft-fonts checkout in ``directory`` from the real fixtures."""
    root = Path(directory)
    (root / "fonts" / "noto-sans-hebrew").mkdir(parents=True)
    (root / "fonts" / "caladea").mkdir(parents=True)
    (root / "fonts" / "noto-sans-hebrew" / "NotoSansHebrew-Regular.ttf").write_bytes(
        _HEBREW_TTF.read_bytes())
    (root / "fonts" / "noto-sans-hebrew" / "OFL.txt").write_bytes(
        _HEBREW_OFL.read_bytes())
    (root / "fonts" / "caladea" / "Caladea-Regular.ttf").write_bytes(
        _CALADEA_TTF.read_bytes())
    (root / "fonts" / "caladea" / "OFL.txt").write_bytes(
        _CALADEA_OFL.read_bytes())
    (root / "fonts" / "manifest.txt").write_text(
        "# Fixture manifest: two real faces, real hashes.\n"
        + _MANIFEST_TWO_ROWS, encoding="utf-8")
    return root


def _registry():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry


def _call(registry, name, **kwargs):
    # registry.call returns an Outcome, not the dict — unwrap .value
    # the same way the craftlauncher test does.
    outcome = registry.call(name, **kwargs)
    if hasattr(outcome, "value"):
        return outcome.value
    return outcome


class ManifestParseTest(unittest.TestCase):
    def test_valid_rows_parse_with_all_eight_fields(self):
        from nomorals.media.typography.craftfonts import parse_manifest

        entries = parse_manifest(_MANIFEST_TWO_ROWS)
        self.assertEqual(len(entries), 2)
        hebrew, caladea = entries
        self.assertEqual(hebrew.family, "Noto Sans Hebrew")
        self.assertEqual(hebrew.style, "Regular")
        self.assertEqual(hebrew.file,
                         "fonts/noto-sans-hebrew/NotoSansHebrew-Regular.ttf")
        self.assertEqual(hebrew.scripts, ("Hebr",))
        self.assertEqual(hebrew.licence, "OFL-1.1")
        self.assertEqual(hebrew.licence_file,
                         "fonts/noto-sans-hebrew/OFL.txt")
        self.assertEqual(hebrew.sha256, _HEBREW_SHA)
        self.assertIn("notofonts/hebrew", hebrew.source)
        self.assertEqual(caladea.family, "Caladea")
        self.assertEqual(caladea.scripts, ("Latn",))

    def test_short_line_is_a_field_count_error_not_a_panic(self):
        from nomorals.media.typography.craftfonts import ManifestError, parse_manifest

        with self.assertRaises(ManifestError) as caught:
            parse_manifest("A | B\n")
        self.assertEqual(caught.exception.kind, "field_count")
        self.assertEqual(caught.exception.line, 1)
        self.assertEqual(caught.exception.found, 2)

    def test_blank_style_is_an_empty_field_error(self):
        from nomorals.media.typography.craftfonts import ManifestError, parse_manifest

        blank = ("A |  | f | Jpan | OFL-1.1 | l | " + "0" * 64 + " | s\n")
        with self.assertRaises(ManifestError) as caught:
            parse_manifest(blank)
        self.assertEqual(caught.exception.kind, "empty_field")
        self.assertEqual(caught.exception.field, "style")

    def test_non_hex_sha_is_a_bad_sha_error(self):
        from nomorals.media.typography.craftfonts import ManifestError, parse_manifest

        with self.assertRaises(ManifestError) as caught:
            parse_manifest("A | B | f | Jpan | OFL-1.1 | l | XYZ | s\n")
        self.assertEqual(caught.exception.kind, "bad_sha256")
        self.assertEqual(caught.exception.line, 1)

    def test_comments_and_blank_lines_are_skipped(self):
        from nomorals.media.typography.craftfonts import parse_manifest

        self.assertEqual(parse_manifest("# comment\n\n"), [])


class ShippedCatalogTest(unittest.TestCase):
    def test_default_table_is_thirteen_faces(self):
        from nomorals.media.typography.craftfonts import default_catalog

        catalog = default_catalog()
        self.assertEqual(len(catalog.default_faces), 13)
        self.assertTrue(all(f.licence == "OFL-1.1"
                            for f in catalog.default_faces))

    def test_latin_table_is_fifty_one_faces(self):
        from nomorals.media.typography.craftfonts import default_catalog

        catalog = default_catalog()
        self.assertEqual(len(catalog.latin_faces), 51)
        self.assertEqual(len({f.family for f in catalog.latin_faces}), 16)

    def test_simplified_chinese_table_is_one_face(self):
        from nomorals.media.typography.craftfonts import default_catalog

        catalog = default_catalog()
        self.assertEqual(len(catalog.chinese_faces), 1)
        self.assertEqual(catalog.chinese_faces[0].family,
                         "LXGW WenKai Lite")

    def test_japanese_faces_are_the_four_documented_rows(self):
        from nomorals.media.typography.craftfonts import default_catalog

        catalog = default_catalog()
        japanese = catalog.for_script("Jpan")
        self.assertEqual(len(japanese), 4)
        self.assertEqual(
            {(f.family, f.style) for f in japanese},
            {("BIZ UDPGothic", "Regular"), ("BIZ UDPGothic", "Bold"),
             ("BIZ UDMincho", "Regular"), ("Shippori Mincho", "Regular")})

    def test_default_and_latin_tables_are_disjoint(self):
        from nomorals.media.typography.craftfonts import default_catalog

        catalog = default_catalog()
        default_files = {f.file for f in catalog.default_faces}
        latin_files = {f.file for f in catalog.latin_faces}
        self.assertEqual(default_files & latin_files, set())

    def test_shipped_manifests_are_byte_identical_to_upstream(self):
        import hashlib

        from nomorals.media.typography import craftfonts

        data_dir = Path(craftfonts.__file__).resolve().parent / "craftfonts"
        expect = {
            "manifest.txt":
                "e825709cd46ff7abe213c141e981c0c11e098195a134951e95dada85eaea778a",
            "latin-manifest.txt":
                "ed0271fc566c46642ad4280bafd2fd4125b3df5ebe582ca9f85e14372a503542",
            "simplified-chinese-manifest.txt":
                "5e42723fe7148934bff0ddf28191e59cc41b021cf82c21803a5477031a24b7c1",
        }
        for name, sha in expect.items():
            got = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            self.assertEqual(got, sha, name)


class CheckoutLoadTest(unittest.TestCase):
    def test_load_reads_every_listed_file_and_verifies_its_hash(self):
        from nomorals.media.typography.craftfonts import load

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            loaded = load(root)
        self.assertEqual(len(loaded), 2)
        by_family = {entry.family: data for entry, data in loaded}
        self.assertEqual(
            hashlib.sha256(by_family["Noto Sans Hebrew"]).hexdigest(),
            _HEBREW_SHA)
        self.assertEqual(
            hashlib.sha256(by_family["Caladea"]).hexdigest(), _CALADEA_SHA)

    def test_tampered_file_is_refused_and_named(self):
        from nomorals.media.typography.craftfonts import LoadError, load

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            target = (root / "fonts" / "caladea" / "Caladea-Regular.ttf")
            data = bytearray(target.read_bytes())
            data[100] ^= 0xFF
            target.write_bytes(bytes(data))
            with self.assertRaises(LoadError) as caught:
                load(root)
        self.assertIn("Caladea-Regular.ttf", str(caught.exception))

    def test_missing_file_is_an_io_error_not_a_panic(self):
        from nomorals.media.typography.craftfonts import LoadError, load

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            (root / "fonts" / "noto-sans-hebrew"
             / "NotoSansHebrew-Regular.ttf").unlink()
            with self.assertRaises(LoadError) as caught:
                load(root)
        self.assertEqual(caught.exception.kind, "io")

    def test_dir_from_env_names_a_checkout_and_nothing_else(self):
        from nomorals.media.typography.craftfonts import dir_from_env

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            os.environ["CRAFT_FONTS_DIR"] = str(root)
            try:
                self.assertEqual(dir_from_env(), root)
            finally:
                del os.environ["CRAFT_FONTS_DIR"]
            os.environ["CRAFT_FONTS_DIR"] = str(Path(tmp) / "nope")
            try:
                self.assertIsNone(dir_from_env())
            finally:
                del os.environ["CRAFT_FONTS_DIR"]
        self.assertIsNone(dir_from_env())


class InstallTest(unittest.TestCase):
    def test_install_verifies_then_copies_fonts_and_licences(self):
        from nomorals.media.typography.craftfonts import install_checkout

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            store = Path(tmp) / "store"
            installed = install_checkout(root, store)
            self.assertEqual(len(installed), 2)
            for entry in installed:
                target = store / entry.file
                self.assertTrue(target.is_file(), entry.file)
                self.assertEqual(
                    hashlib.sha256(target.read_bytes()).hexdigest(),
                    entry.sha256)
                licence = store / entry.licence_file
                self.assertTrue(licence.is_file(), entry.licence_file)
                self.assertIn("SIL OPEN FONT LICENSE",
                              licence.read_text(encoding="utf-8"))
            record = json.loads((store / "install-record.json").read_text(
                encoding="utf-8"))
            self.assertEqual(len(record["installed"]), 2)
            self.assertEqual(record["installed"][0]["licence"], "OFL-1.1")
            self.assertEqual(record["installed"][0]["sha256"], _HEBREW_SHA)

    def test_tampered_checkout_installs_nothing(self):
        from nomorals.media.typography.craftfonts import LoadError, install_checkout

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            store = Path(tmp) / "store"
            target = (root / "fonts" / "caladea" / "Caladea-Regular.ttf")
            data = bytearray(target.read_bytes())
            data[100] ^= 0xFF
            target.write_bytes(bytes(data))
            with self.assertRaises(LoadError):
                install_checkout(root, store)
            self.assertFalse((store / "fonts").exists())


class BundleTest(unittest.TestCase):
    def test_bundle_carries_exactly_the_allowlisted_faces(self):
        from nomorals.media.typography.craftfonts import build_embed, load

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            loaded = load(root)
            payload = build_embed(
                loaded, root=root,
                allowlist=[("Noto Sans Hebrew", "Regular"),
                           ("Caladea", "Regular")])
            out = Path(tmp) / "bundle.zip"
            out.write_bytes(payload)
            with zipfile.ZipFile(out) as bundle:
                names = sorted(bundle.namelist())
                provenance = json.loads(
                    bundle.read("provenance.json").decode("utf-8"))
                manifest = bundle.read("manifest.txt").decode("utf-8")
            self.assertEqual(names, [
                "fonts/caladea/Caladea-Regular.ttf",
                "fonts/caladea/OFL.txt",
                "fonts/noto-sans-hebrew/NotoSansHebrew-Regular.ttf",
                "fonts/noto-sans-hebrew/OFL.txt",
                "manifest.txt",
                "provenance.json",
            ])
            self.assertIn("Noto Sans Hebrew", manifest)
            self.assertIn("Caladea", manifest)
            self.assertEqual(provenance["faces"][0]["sha256"], _HEBREW_SHA)
            self.assertEqual(provenance["faces"][1]["sha256"], _CALADEA_SHA)
            self.assertEqual(provenance["licence"], "OFL-1.1")

    def test_bundle_names_a_missing_allowlisted_face(self):
        from nomorals.media.typography.craftfonts import build_embed, load

        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            loaded = load(root)
            with self.assertRaises(ValueError) as caught:
                build_embed(loaded, root=root,
                            allowlist=[("No Such Face", "Regular")])
        self.assertIn("No Such Face", str(caught.exception))


class NameTableTest(unittest.TestCase):
    def test_fixture_name_table_matches_the_manifest(self):
        from nomorals.media.typography.fontparse import parse_font_file

        faces = parse_font_file(_HEBREW_TTF)
        self.assertEqual(len(faces), 1)
        face = faces[0]
        self.assertEqual(face.family, "Noto Sans Hebrew")
        self.assertEqual(face.subfamily, "Regular")

    def test_hebrew_face_covers_hebrew_text_and_not_japanese(self):
        from nomorals.media.typography.fontparse import parse_font_file

        face = parse_font_file(_HEBREW_TTF)[0]
        self.assertTrue(face.covers_text("שלום"))
        self.assertFalse(face.covers_text("日本語"))


class LayoutUnificationTest(unittest.TestCase):
    def test_layout_font_door_measures_with_the_service_face(self):
        from PIL import ImageFont

        from nomorals.media.layout.compose import font_for, measure_text

        direct = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        self.assertEqual(measure_text("Hello world", 16.0),
                         direct.getlength("Hello world"))
        bold_direct = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        self.assertEqual(measure_text("Hello world", 16.0, bold=True),
                         bold_direct.getlength("Hello world"))
        font = font_for(16.0)
        self.assertEqual(
            getattr(font, "path", ""),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


class ToolCountTest(unittest.TestCase):
    def test_tool_count_is_783_after_craftfonts(self):
        # 732 after the craft-launcher port (lane 112), 733 after the
        # craft-libs port (lane 113) added the craftlibs tool, 734 after
        # the craft-fonts port (lane 114) added the craftfonts tool,
        # 735 after the indd-utils port (lane 117) added the
        # inddutils tool, 748 after the auth-plugin wiring (5 new tools) on top of the bughunter port (lane 121)
        # added 8 bughunter tools. 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome.
        registry = _registry()
        self.assertIn("craftfonts", registry._tools)
        self.assertEqual(len(registry._tools), 783)


class ToolSurfaceTest(unittest.TestCase):
    def test_catalog_action_lists_all_three_tables(self):
        registry = _registry()
        result = _call(registry, "craftfonts", action="catalog")
        self.assertTrue(result["ok"])
        self.assertEqual(result["default_faces"], 13)
        self.assertEqual(result["latin_faces"], 51)
        self.assertEqual(result["chinese_faces"], 1)
        self.assertEqual(result["default_families"], 12)

    def test_for_script_action_names_the_japanese_faces(self):
        registry = _registry()
        result = _call(registry, "craftfonts", action="for_script",
                               value="Jpan")
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["faces"]), 4)

    def test_check_action_verifies_a_fixture_checkout(self):
        registry = _registry()
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            result = _call(registry, "craftfonts", action="check",
                                   sources=[str(root)])
        self.assertTrue(result["ok"])
        self.assertEqual(result["faces"], 2)
        self.assertEqual(result["verified"], 2)

    def test_install_and_render_actions_run_end_to_end(self):
        registry = _registry()
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_checkout(Path(tmp))
            home = Path(tmp) / "home"
            installed = _call(
                registry, "craftfonts", action="install", sources=[str(root)],
                params={"home": str(home)})
            self.assertTrue(installed["ok"], installed)
            self.assertEqual(installed["installed"], 2)
            rendered = _call(
                registry, "craftfonts", action="render",
                value="שלום Hello",
                params={"family": "Noto Sans Hebrew", "size": 32,
                        "home": str(home)})
            self.assertTrue(rendered["ok"], rendered)
            self.assertEqual(rendered["family"], "Noto Sans Hebrew")
            self.assertGreater(rendered["png_bytes"], 0)
            other = _call(
                registry, "craftfonts", action="render", value="Hello",
                params={"family": "Noto Sans Hebrew", "size": 32,
                        "home": str(home)})
            self.assertNotEqual(rendered["png_sha256"],
                                other["png_sha256"])

    def test_parity_and_attribution_actions_pass(self):
        registry = _registry()
        parity = _call(registry, "craftfonts", action="parity")
        self.assertTrue(parity["ok"], parity)
        self.assertEqual(parity["failed"], 0)
        self.assertEqual(parity["passed"], 16)
        attribution = _call(registry, "craftfonts", action="attribution")
        self.assertTrue(attribution["ok"], attribution)
        self.assertEqual(attribution["failed"], 0)


class ParityTest(unittest.TestCase):
    def test_parity_toml_runs_fourteen_cases(self):
        from nomorals.media.typography.craftfonts import run_parity

        result = run_parity()
        self.assertEqual(result["failed"], 0, result)
        self.assertEqual(result["passed"], 16)
        self.assertEqual(result["total"], 16)


class AttributionTest(unittest.TestCase):
    def test_attribution_hashes_match_the_clone(self):
        from nomorals.media.typography.craftfonts import verify_attribution

        result = verify_attribution()
        self.assertEqual(result["failed"], 0, result)
        self.assertEqual(result["skipped"], 0, result)
        self.assertGreaterEqual(result["checked"], 12)


if __name__ == "__main__":
    unittest.main()
