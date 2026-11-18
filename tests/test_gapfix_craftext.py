"""Studio extension store — the craft-extensions contract, ported.

The upstream contract (storytold/craft-extensions, dual MIT/Apache):
a package manifest (``craft-extension.json``, schema 1), validated
ids, a theme palette with a WCAG contrast check, trust tiers the
host assigns (never the package), and an on-disk store with bounded,
hash-verified unpacking, atomic state, enable-per-app and one active
theme per app. Devon's Studio wears the themes and renders the
panels; the photo page is the first page that does.

These tests fail first: there is no ``nomorals.studio`` at base.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "photocraft"
RGB8 = FIXTURES / "rgb8-baseline-layer-copy.psd"

DARK_THEME = {
    "schema": 1,
    "palette": {
        "dark": True,
        "background": "#111111",
        "surface": "#1a1a1a",
        "text": "#f2f2f2",
        "muted_text": "#c8c8c8",
        "line": "#333333",
        "accent": "#7fb3ff",
        "good": "#5fd35f",
        "warn": "#ffcf5c",
        "bad": "#ff7a7a",
    },
}

LIGHT_THEME = {
    "schema": 1,
    "palette": {
        "dark": False,
        "background": "#ffffff",
        "surface": "#f2f2f2",
        "text": "#111111",
        "muted_text": "#444444",
        "line": "#cccccc",
        "accent": "#1a5fb4",
        "good": "#1e7d32",
        "warn": "#8a6100",
        "bad": "#b3261e",
    },
}

PANEL = {
    "schema": 1,
    "page": "photo",
    "slot": "below",
    "title": "Photo notes",
    "blocks": [
        {"kind": "text", "text": "A panel added by an extension."},
        {"kind": "kv", "label": "Source", "value": "org.example.photo-notes"},
        {"kind": "link", "label": "Craft docs",
         "href": "https://example.org/docs"},
    ],
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(ext_id: str, kind: str, files: dict[str, bytes],
              *, version: str = "1.0.0", apps=("studio",),
              points: list[str] | None = None,
              interface: int | None = None,
              permissions: list[str] | None = None,
              provenance: dict | None = None) -> dict:
    compat: dict[str, dict] = {}
    for app in apps:
        entry: dict = {}
        if kind == "plugin":
            entry["interface"] = 1 if interface is None else interface
        compat[app] = entry
    body: dict = {
        "schema": 1,
        "id": ext_id,
        "name": "Sample " + kind,
        "version": version,
        "kind": kind,
        "publisher": {"name": "Example"},
        "description": "A sample package for the tests.",
        "compat": compat,
        "files": [
            {"path": name, "sha256": _sha(data), "size": len(data)}
            for name, data in sorted(files.items())
        ],
        "permissions": [] if permissions is None else permissions,
    }
    if points is not None:
        body["extension_points"] = points
    if provenance is not None:
        body["provenance"] = provenance
    return body


def _package_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(files.items()):
            zf.writestr(name, data)
    return buf.getvalue()


def _theme_package(ext_id: str = "org.example.midnight",
                   theme: dict | None = None,
                   **kw) -> bytes:
    payload = json.dumps(theme if theme is not None else DARK_THEME
                         ).encode("utf-8")
    files = {"theme.json": payload}
    manifest = _manifest(ext_id, "theme", files,
                         points=["studio.theme"], **kw)
    files["craft-extension.json"] = json.dumps(manifest).encode("utf-8")
    return _package_zip(files)


def _plugin_package(ext_id: str = "org.example.photo-notes",
                    panel: dict | None = None, **kw) -> bytes:
    payload = json.dumps(panel if panel is not None else PANEL
                         ).encode("utf-8")
    files = {"panel.json": payload}
    manifest = _manifest(ext_id, "plugin", files,
                         points=["photo.panel"], **kw)
    files["craft-extension.json"] = json.dumps(manifest).encode("utf-8")
    return _package_zip(files)


def _resource_package(ext_id: str = "org.example.textures",
                      **kw) -> bytes:
    files = {"assets/grain.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 64}
    manifest = _manifest(ext_id, "resource", files,
                         points=["studio.resource"], **kw)
    files["craft-extension.json"] = json.dumps(manifest).encode("utf-8")
    return _package_zip(files)


class VersionTests(unittest.TestCase):
    def test_parses_and_orders(self) -> None:
        from nomorals.studio.version import Version

        self.assertEqual(Version.parse("1.2.3"), Version(1, 2, 3))
        self.assertLess(Version.parse("1.9.9"), Version.parse("1.10.0"))
        self.assertLess(Version.parse("1.99.99"), Version.parse("2.0.0"))
        self.assertEqual(str(Version(3, 1, 4)), "3.1.4")

    def test_refuses_everything_else(self) -> None:
        from nomorals.studio.version import Version, VersionError

        for bad in ["", "1", "1.2", "1.2.3.4", "01.2.3", "1.02.3",
                    "v1.2.3", "1.2.3-beta", "1.2.3+x", "-1.2.3",
                    "1. 2.3", "1.2.x", "4294967296.0.0"]:
            with self.assertRaises(VersionError, msg=bad):
                Version.parse(bad)


class IdTests(unittest.TestCase):
    def test_extension_ids(self) -> None:
        from nomorals.studio.ids import ExtensionId, IdError, IdErrorKind

        for ok in ["org.example.tone", "io.github.someone.dark-ocean",
                   "a1.b2"]:
            self.assertEqual(ExtensionId.new(ok), ok)
        for bad, why in [("", IdErrorKind.EMPTY),
                         ("single", IdErrorKind.BAD_SHAPE),
                         ("Org.example", IdErrorKind.BAD_CHARACTER),
                         ("org.exämple", IdErrorKind.BAD_CHARACTER),
                         (".leading", IdErrorKind.BAD_SHAPE),
                         ("trailing.", IdErrorKind.BAD_SHAPE),
                         ("org.-dash", IdErrorKind.BAD_SHAPE)]:
            with self.assertRaises(IdError) as caught:
                ExtensionId.new(bad)
            self.assertEqual(caught.exception.kind, why, bad)

    def test_app_ids(self) -> None:
        from nomorals.studio.ids import AppId, IdError

        self.assertEqual(AppId.new("studio"), "studio")
        self.assertEqual(AppId.new("photo-craft2"), "photo-craft2")
        for bad in ["", "Photo", "photo craft", "-photo", "photo.craft",
                    "1photo"]:
            with self.assertRaises(IdError, msg=bad):
                AppId.new(bad)

    def test_first_party_namespaces(self) -> None:
        from nomorals.studio.ids import ExtensionId

        for prefix in ["ai.devon.", "org.devon.",
                       "io.github.oluwacutyp."]:
            ext = ExtensionId.new(prefix + "midnight")
            self.assertTrue(ext.in_first_party_namespace(), prefix)
        plain = ExtensionId.new("org.example.midnight")
        self.assertFalse(plain.in_first_party_namespace())
        lookalike = ExtensionId.new("org.devonx.midnight")
        self.assertFalse(lookalike.in_first_party_namespace())


class TextTests(unittest.TestCase):
    def test_accepts_ordinary_text(self) -> None:
        from nomorals.studio.text import Lines, validate_text

        for ok in ["Dark Ocean", "Café", "深色海洋"]:
            self.assertIsNone(validate_text(ok, 64, Lines.ONE), ok)

    def test_refuses_hostile_text(self) -> None:
        from nomorals.studio.text import Lines, TextError, validate_text

        cases = [("", TextError.EMPTY), ("   ", TextError.NOTHING_VISIBLE),
                 ("a\0b", TextError.CONTROL),
                 ("a\u202eb", TextError.INVISIBLE),
                 ("a\u200bb", TextError.INVISIBLE),
                 ("a\nb", TextError.NEWLINE)]
        for text, why in cases:
            err = validate_text(text, 64, Lines.ONE)
            self.assertEqual(err, why, repr(text))
        self.assertIsNone(validate_text("a\nb", 64, Lines.MANY))


class PathTests(unittest.TestCase):
    def test_accepts_plain_relative_paths(self) -> None:
        from nomorals.studio.paths import validate_path

        for ok in ["theme.json", "assets/icon.png", "a/b/c/d.txt"]:
            self.assertIsNone(validate_path(ok), ok)

    def test_refuses_escapes_and_reserved_names(self) -> None:
        from nomorals.studio.paths import PathError, validate_path

        cases = [("../x", PathError.DOT_SEGMENT),
                 ("/etc/passwd", PathError.EMPTY_SEGMENT),
                 ("a//b", PathError.EMPTY_SEGMENT),
                 ("a\\b", PathError.BAD_CHARACTER),
                 ("con", PathError.RESERVED),
                 ("nul.txt", PathError.RESERVED),
                 ("dir/aux", PathError.RESERVED),
                 ("a/b/c/d/e/f/g/h/i", PathError.TOO_DEEP)]
        for path, why in cases:
            self.assertEqual(validate_path(path), why, path)

    def test_whole_list_refuses_duplicates_and_case_collisions(self) -> None:
        from nomorals.studio.paths import PathError, validate_all

        self.assertIsNone(validate_all(["a.txt", "b.txt"]))
        self.assertEqual(validate_all(["a.txt", "a.txt"]),
                         PathError.DUPLICATE)
        self.assertEqual(validate_all(["Readme.md", "README.md"]),
                         PathError.CASE_COLLISION)
        self.assertEqual(validate_all(["a", "a/b"]),
                         PathError.CASE_COLLISION)


class ManifestTests(unittest.TestCase):
    def _parse(self, body: dict) -> object:
        from nomorals.studio.manifest import Manifest

        return Manifest.parse(json.dumps(body).encode("utf-8"))

    def test_well_formed_theme_parses(self) -> None:
        manifest = self._parse(_manifest(
            "org.example.midnight", "theme",
            {"theme.json": b"{}"}, points=["studio.theme"]))
        self.assertEqual(str(manifest.id), "org.example.midnight")
        self.assertEqual(str(manifest.kind), "theme")

    def test_schema_two_is_refused(self) -> None:
        from nomorals.studio.manifest import ManifestError

        body = _manifest("org.example.midnight", "theme",
                         {"theme.json": b"{}"}, points=["studio.theme"])
        body["schema"] = 2
        with self.assertRaises(ManifestError) as caught:
            self._parse(body)
        self.assertIn("format version", str(caught.exception))

    def test_permissions_must_be_empty(self) -> None:
        from nomorals.studio.manifest import ManifestError

        body = _manifest("org.example.midnight", "theme",
                         {"theme.json": b"{}"}, points=["studio.theme"],
                         permissions=["files.read"])
        with self.assertRaises(ManifestError) as caught:
            self._parse(body)
        self.assertIn("permissions", str(caught.exception))

    def test_plugin_needs_an_interface_and_theme_must_not_have_one(
            self) -> None:
        from nomorals.studio.manifest import ManifestError

        body = _manifest("org.example.notes", "plugin",
                         {"panel.json": b"{}"}, points=["photo.panel"],
                         interface=None)
        body["compat"]["studio"] = {}
        with self.assertRaises(ManifestError):
            self._parse(body)
        body = _manifest("org.example.midnight", "theme",
                         {"theme.json": b"{}"}, points=["studio.theme"])
        body["compat"]["studio"] = {"interface": 1}
        with self.assertRaises(ManifestError):
            self._parse(body)

    def test_required_file_and_bad_hash(self) -> None:
        from nomorals.studio.manifest import ManifestError

        body = _manifest("org.example.midnight", "theme",
                         {"other.json": b"{}"}, points=["studio.theme"])
        with self.assertRaises(ManifestError) as caught:
            self._parse(body)
        self.assertIn("missing", str(caught.exception))
        body = _manifest("org.example.midnight", "theme",
                         {"theme.json": b"{}"}, points=["studio.theme"])
        body["files"][0]["sha256"] = "zz" * 32
        with self.assertRaises(ManifestError):
            self._parse(body)

    def test_extension_points_are_checked(self) -> None:
        from nomorals.studio.manifest import ManifestError

        body = _manifest("org.example.midnight", "theme",
                         {"theme.json": b"{}"},
                         points=["photo.panel"])
        with self.assertRaises(ManifestError) as caught:
            self._parse(body)
        self.assertIn("extension point", str(caught.exception))
        body = _manifest("org.example.midnight", "theme",
                         {"theme.json": b"{}"},
                         points=["studio.nope"])
        with self.assertRaises(ManifestError):
            self._parse(body)

    def test_provenance_is_checked(self) -> None:
        from nomorals.studio.manifest import ManifestError

        good = {"source": "https://github.com/example/midnight",
                "commit": "a" * 40}
        self._parse(_manifest("org.example.midnight", "theme",
                              {"theme.json": b"{}"},
                              points=["studio.theme"], provenance=good))
        bad = {"source": "http://github.com/example/midnight",
               "commit": "a" * 40}
        with self.assertRaises(ManifestError):
            self._parse(_manifest(
                "org.example.midnight", "theme", {"theme.json": b"{}"},
                points=["studio.theme"], provenance=bad))

    def test_manifest_too_large(self) -> None:
        from nomorals.studio.manifest import ManifestError

        body = _manifest("org.example.midnight", "theme",
                         {"theme.json": b"{}"}, points=["studio.theme"])
        body["description"] = "x" * (64 * 1024)
        with self.assertRaises(ManifestError):
            self._parse(body)


class PaletteTests(unittest.TestCase):
    def test_dark_and_light_themes_parse(self) -> None:
        from nomorals.studio.palette import ThemeFile

        for theme in (DARK_THEME, LIGHT_THEME):
            parsed = ThemeFile.parse(json.dumps(theme).encode("utf-8"))
            self.assertEqual(str(parsed.palette.background),
                             theme["palette"]["background"])

    def test_low_contrast_is_refused_and_named(self) -> None:
        from nomorals.studio.palette import ThemeError, ThemeFile

        bad = json.loads(json.dumps(DARK_THEME))
        bad["palette"]["text"] = "#333333"
        with self.assertRaises(ThemeError) as caught:
            ThemeFile.parse(json.dumps(bad).encode("utf-8"))
        self.assertIn("text and background", str(caught.exception))

    def test_dark_flag_must_agree_with_the_colours(self) -> None:
        from nomorals.studio.palette import ThemeError, ThemeFile

        bad = json.loads(json.dumps(DARK_THEME))
        bad["palette"]["dark"] = False
        with self.assertRaises(ThemeError):
            ThemeFile.parse(json.dumps(bad).encode("utf-8"))

    def test_bad_hex_and_bad_token_names(self) -> None:
        from nomorals.studio.palette import ThemeError, ThemeFile

        bad = json.loads(json.dumps(DARK_THEME))
        bad["palette"]["accent"] = "blue"
        with self.assertRaises(ThemeError):
            ThemeFile.parse(json.dumps(bad).encode("utf-8"))
        bad = json.loads(json.dumps(DARK_THEME))
        bad["apps"] = {"studio": {"Bad Name": "#111111"}}
        with self.assertRaises(ThemeError):
            ThemeFile.parse(json.dumps(bad).encode("utf-8"))

    def test_contrast_ratio(self) -> None:
        from nomorals.studio.palette import Rgb, contrast_ratio

        self.assertAlmostEqual(
            contrast_ratio(Rgb(0, 0, 0), Rgb(255, 255, 255)), 21.0,
            places=1)
        self.assertAlmostEqual(
            contrast_ratio(Rgb(17, 17, 17), Rgb(17, 17, 17)), 1.0,
            places=6)


class TierTests(unittest.TestCase):
    def test_policy(self) -> None:
        from nomorals.studio.tier import Policy, Tier, TierError

        policy = Policy.standard()
        self.assertIsNone(policy.admit(Tier.FIRST_PARTY))
        self.assertIsNone(policy.admit(Tier.FROM_FILE))
        self.assertEqual(policy.admit(Tier.REVIEWED_THIRD_PARTY),
                         TierError.THIRD_PARTY_CLOSED)

    def test_only_first_party_may_use_first_party_names(self) -> None:
        from nomorals.studio.ids import ExtensionId
        from nomorals.studio.tier import Policy, Tier, TierError

        policy = Policy.standard()
        reserved = ExtensionId.new("ai.devon.midnight")
        self.assertIsNone(policy.check_id(Tier.FIRST_PARTY, reserved))
        self.assertEqual(policy.check_id(Tier.FROM_FILE, reserved),
                         TierError.RESERVED_NAMESPACE)
        plain = ExtensionId.new("org.example.midnight")
        self.assertIsNone(policy.check_id(Tier.FROM_FILE, plain))


class PanelTests(unittest.TestCase):
    def test_well_formed_panel_parses(self) -> None:
        from nomorals.studio.panel import PanelFile

        panel = PanelFile.parse(json.dumps(PANEL).encode("utf-8"))
        self.assertEqual(panel.page, "photo")
        self.assertEqual(panel.slot, "below")
        self.assertEqual(len(panel.blocks), 3)

    def test_unknown_page_slot_and_scheme_are_refused(self) -> None:
        from nomorals.studio.panel import PanelError, PanelFile

        for tweak in ({"page": "board"}, {"slot": "sidebar-top"},
                      {"blocks": [{"kind": "link", "label": "x",
                                   "href": "javascript:alert(1)"}]},
                      {"blocks": [{"kind": "text",
                                   "text": "x" * 2001}]}):
            bad = json.loads(json.dumps(PANEL))
            bad.update(tweak)
            with self.assertRaises(PanelError, msg=str(tweak)):
                PanelFile.parse(json.dumps(bad).encode("utf-8"))


class StoreTests(unittest.TestCase):
    def _store(self, tmp: str):
        from nomorals.studio.store import StudioStore

        return StudioStore(root=str(Path(tmp) / "extensions"))

    def test_install_theme_then_wear_it(self) -> None:
        from nomorals.studio.ids import AppId

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            rec = store.install_zip(_theme_package(), tier="from-file")
            self.assertEqual(rec.id, "org.example.midnight")
            self.assertEqual(rec.version, "1.0.0")
            self.assertEqual(store.state().generation, 1)
            app = AppId.new("studio")
            self.assertIsNone(store.active_theme(app))
            store.set_active_theme(app, rec.id)
            active = store.active_theme(app)
            self.assertIsNotNone(active)
            assert active is not None
            self.assertEqual(active.name, "Sample theme")
            self.assertTrue(active.palette.dark)
            self.assertEqual(str(active.palette.background), "#111111")
            self.assertEqual(
                [f.installed.id for f in
                 store.installed_for(app, "theme")],
                ["org.example.midnight"])

    def test_install_is_all_or_nothing(self) -> None:
        from nomorals.studio.store import StoreError

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            # The manifest promises one theme payload; the zip carries
            # different bytes, so the hash check fails and nothing lands.
            payload = json.dumps(DARK_THEME).encode("utf-8")
            manifest = _manifest("org.example.midnight", "theme",
                                 {"theme.json": payload},
                                 points=["studio.theme"])
            files = {"theme.json": b"not the theme",
                     "craft-extension.json":
                         json.dumps(manifest).encode("utf-8")}
            with self.assertRaises(StoreError):
                store.install_zip(_package_zip(files), tier="from-file")
            self.assertEqual(store.state().installed, [])
            self.assertEqual(list(Path(tmp).rglob("theme.json")), [])

    def test_same_version_downgrade_and_first_party(self) -> None:
        from nomorals.studio.store import StoreError

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.install_zip(_theme_package(), tier="from-file")
            with self.assertRaises(StoreError) as caught:
                store.install_zip(_theme_package(), tier="from-file")
            self.assertIn("already installed", str(caught.exception))
            with self.assertRaises(StoreError) as caught:
                store.install_zip(_theme_package(version="0.9.0"),
                                  tier="from-file")
            self.assertIn("newer version", str(caught.exception))
            with self.assertRaises(StoreError) as caught:
                store.install_zip(
                    _theme_package("ai.devon.midnight"),
                    tier="from-file")
            self.assertIn("only Devon", str(caught.exception))

    def test_uninstall_removes_files_state_and_choice(self) -> None:
        from nomorals.studio.ids import AppId

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            rec = store.install_zip(_theme_package(), tier="from-file")
            app = AppId.new("studio")
            store.set_active_theme(app, rec.id)
            store.uninstall(rec.id)
            self.assertEqual(store.state().installed, [])
            self.assertIsNone(store.active_theme(app))
            self.assertFalse((Path(tmp) / "extensions" / "packages"
                              / "org.example.midnight").exists())

    def test_switching_off_hides_it_and_clears_the_theme(self) -> None:
        from nomorals.studio.ids import AppId

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            rec = store.install_zip(_theme_package(), tier="from-file")
            app = AppId.new("studio")
            store.set_active_theme(app, rec.id)
            gen = store.state().generation
            store.set_enabled(app, rec.id, False)
            self.assertEqual(store.installed_for(app, "theme"), [])
            self.assertIsNone(store.active_theme(app))
            self.assertGreater(store.state().generation, gen)

    def test_tampered_theme_on_disk_counts_as_absent(self) -> None:
        from nomorals.studio.ids import AppId

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            rec = store.install_zip(_theme_package(), tier="from-file")
            app = AppId.new("studio")
            store.set_active_theme(app, rec.id)
            theme_path = (Path(tmp) / "extensions" / "packages"
                          / "org.example.midnight" / "1.0.0" / "theme.json")
            theme_path.write_bytes(b"garbage")
            self.assertIsNone(store.active_theme(app))
            low = json.loads(json.dumps(DARK_THEME))
            low["palette"]["text"] = "#333333"
            theme_path.write_text(json.dumps(low))
            self.assertIsNone(store.active_theme(app))

    def test_damaged_state_reads_as_absent_and_never_repairs(self) -> None:
        from nomorals.studio.ids import AppId
        from nomorals.studio.store import StoreError

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.install_zip(_theme_package(), tier="from-file")
            state_path = Path(tmp) / "extensions" / "state.json"
            state_path.write_bytes(b"{ broken")
            self.assertEqual(store.reader().generation(), 0)
            self.assertEqual(store.reader().installed_for(
                AppId.new("studio"), "theme"), [])
            self.assertEqual(state_path.read_bytes(), b"{ broken")
            with self.assertRaises(StoreError):
                store.install_zip(_theme_package("org.example.other"),
                                  tier="from-file")

    def test_package_for_another_app_is_refused(self) -> None:
        from nomorals.studio.store import StoreError

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            with self.assertRaises(StoreError) as caught:
                store.install_zip(
                    _theme_package(apps=("photocraft",)),
                    tier="from-file")
            self.assertIn("not made for this app", str(caught.exception))

    def test_install_from_url_pins_the_digest(self) -> None:
        from nomorals.studio.store import StoreError

        payload = _theme_package()

        class _FakeClient:
            def __init__(self, body: bytes, status: int = 200) -> None:
                self._body = body
                self._status = status

            def get(self, url: str):
                return SimpleNamespace(ok=self._status == 200,
                                       status=self._status,
                                       body=self._body, text="")

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            rec = store.install_url(
                "https://example.org/midnight.zip",
                sha256=_sha(payload), size=len(payload),
                client=_FakeClient(payload))
            self.assertEqual(rec.id, "org.example.midnight")
            with self.assertRaises(StoreError) as caught:
                store.install_url(
                    "https://example.org/midnight.zip",
                    sha256="0" * 64, size=len(payload),
                    client=_FakeClient(payload))
            self.assertIn("digest", str(caught.exception))
            with self.assertRaises(StoreError):
                store.install_url(
                    "https://example.org/missing.zip",
                    sha256=_sha(payload), size=len(payload),
                    client=_FakeClient(payload, status=404))


class UnpackTests(unittest.TestCase):
    def _unpack(self, payload: bytes, tmp: str):
        from nomorals.studio.store import unpack_into

        return unpack_into(payload, Path(tmp) / "staged")

    def test_unlisted_file_is_refused(self) -> None:
        from nomorals.studio.store import UnpackError

        files = {"theme.json": json.dumps(DARK_THEME).encode("utf-8"),
                 "stowaway.txt": b"not in the manifest"}
        manifest = _manifest("org.example.midnight", "theme",
                             {"theme.json": files["theme.json"]},
                             points=["studio.theme"])
        files["craft-extension.json"] = json.dumps(manifest).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnpackError) as caught:
                self._unpack(_package_zip(files), tmp)
            self.assertIn("does not mention", str(caught.exception))

    def test_missing_listed_file_is_refused(self) -> None:
        from nomorals.studio.store import UnpackError

        manifest = _manifest("org.example.midnight", "theme",
                             {"theme.json": b"{}", "ghost.json": b"{}"},
                             points=["studio.theme"])
        files = {"theme.json": b"{}",
                 "craft-extension.json":
                     json.dumps(manifest).encode("utf-8")}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(UnpackError):
            self._unpack(_package_zip(files), tmp)

    def test_wrong_size_and_wrong_hash_are_refused(self) -> None:
        from nomorals.studio.store import UnpackError

        manifest = _manifest("org.example.midnight", "theme",
                             {"theme.json": b"{}"},
                             points=["studio.theme"])
        manifest["files"][0]["size"] = 99
        files = {"theme.json": b"{}",
                 "craft-extension.json":
                     json.dumps(manifest).encode("utf-8")}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnpackError) as caught:
                self._unpack(_package_zip(files), tmp)
            self.assertIn("size", str(caught.exception))
        manifest["files"][0]["size"] = 2
        manifest["files"][0]["sha256"] = "0" * 64
        files["craft-extension.json"] = json.dumps(manifest).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnpackError) as caught:
                self._unpack(_package_zip(files), tmp)
            self.assertIn("changed or damaged", str(caught.exception))

    def test_escape_entry_and_symlink_are_refused(self) -> None:
        from nomorals.studio.store import UnpackError

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", b"x")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(UnpackError):
            self._unpack(buf.getvalue(), tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("link.txt")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, b"target")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnpackError) as caught:
                self._unpack(buf.getvalue(), tmp)
            self.assertIn("shortcut", str(caught.exception))

    def test_no_manifest_is_refused(self) -> None:
        from nomorals.studio.store import UnpackError

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnpackError) as caught:
                self._unpack(_package_zip({"theme.json": b"{}"}), tmp)
            self.assertIn("no description", str(caught.exception))


class ToolAndPageTests(unittest.TestCase):
    def _context(self, tmp: str):
        return SimpleNamespace(settings=SimpleNamespace(home=tmp))

    def _call(self, registry, action: str, **kw):
        from nomorals.core.policy import CapabilitySet

        outcome = registry.call("studio_extensions", actor="test",
                                capabilities=CapabilitySet.all(),
                                action=action, **kw)
        self.assertTrue(outcome.ok, outcome)
        return outcome.unwrap()

    def test_tool_end_to_end(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(self._context(tmp))
            registry.register_builtins()
            names = {s["name"] for s in registry.schemas()}
            self.assertIn("studio_extensions", names)
            pkg = Path(tmp) / "midnight.zip"
            pkg.write_bytes(_theme_package())
            installed = self._call(registry, "install", path=str(pkg))
            self.assertTrue(installed["ok"], installed)
            self.assertEqual(installed["id"], "org.example.midnight")
            listed = self._call(registry, "list")
            self.assertEqual(len(listed["installed"]), 1)
            themed = self._call(registry, "theme",
                                ext_id="org.example.midnight")
            self.assertTrue(themed["ok"], themed)
            shown = self._call(registry, "list")
            self.assertEqual(shown["active_theme"], "org.example.midnight")
            plugin = Path(tmp) / "notes.zip"
            plugin.write_bytes(_plugin_package())
            self._call(registry, "install", path=str(plugin))
            panels = self._call(registry, "panels", page="photo")
            self.assertEqual(len(panels["panels"]), 1)
            self.assertEqual(panels["panels"][0]["title"], "Photo notes")

    def test_photo_page_wears_the_theme_and_the_panel(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.page import photo_page_html
        from nomorals.studio.palette import ThemeFile

        doc = PhotoDocument.from_psd_bytes(RGB8.read_bytes())
        plain = photo_page_html(doc, doc_id="abc")
        self.assertNotIn("bgcolor", plain)
        theme = ThemeFile.parse(json.dumps(DARK_THEME).encode("utf-8"))
        themed = photo_page_html(doc, doc_id="abc", theme=theme.palette,
                                 theme_name="Midnight")
        self.assertIn('bgcolor="#111111"', themed)
        self.assertIn('text="#f2f2f2"', themed)
        self.assertIn('link="#7fb3ff"', themed)
        self.assertIn("Midnight", themed)
        from nomorals.studio.panel import PanelFile

        panel = PanelFile.parse(json.dumps(PANEL).encode("utf-8"))
        with_panel = photo_page_html(doc, doc_id="abc", panels=[panel])
        self.assertIn("Photo notes", with_panel)
        self.assertIn("A panel added by an extension.", with_panel)
        self.assertIn("https://example.org/docs", with_panel)

    def test_panel_text_is_escaped_on_the_page(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.page import photo_page_html
        from nomorals.studio.panel import PanelFile

        nasty = json.loads(json.dumps(PANEL))
        nasty["blocks"] = [{"kind": "text",
                            "text": "<script>alert(1)</script>"}]
        panel = PanelFile.parse(json.dumps(nasty).encode("utf-8"))
        doc = PhotoDocument.from_psd_bytes(RGB8.read_bytes())
        html = photo_page_html(doc, doc_id="abc", panels=[panel])
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


class _LiveServer:
    """Real HTTP server on loopback; http.client for full control."""

    def __init__(self, api: Any) -> None:
        import http.client as _http
        import threading as _threading
        from http.server import ThreadingHTTPServer as _Server

        from nomorals.api.server import _make_handler

        self._http = _http
        self.httpd = _Server(("127.0.0.1", 0), _make_handler(api))
        self.port = self.httpd.server_address[1]
        self.thread = _threading.Thread(
            target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def get(self, path: str, token: str | None = None):
        conn = self._http.HTTPConnection(
            "127.0.0.1", self.port, timeout=10)
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn.request("GET", path, headers=headers)
        return conn.getresponse()


class HttpRouteTests(unittest.TestCase):
    def _server(self, tmp: str):
        from nomorals.api.server import APIServer
        from nomorals.storage.db import Database
        from nomorals.tools.registry import ToolRegistry

        settings = SimpleNamespace(
            home=tmp,
            api=SimpleNamespace(token="", max_body_mb=0),
            backup_dir=tmp,
            workspace_dir=tmp,
        )
        context = SimpleNamespace(
            db=Database(":memory:"), settings=settings,
            tools=ToolRegistry(), bus=None, memory=None,
        )
        api = APIServer(context, token="<redacted>",
                        max_body_bytes=64 * 1024)
        return _LiveServer(api)

    def test_extensions_board_route(self) -> None:
        from nomorals.studio.store import StudioStore

        with tempfile.TemporaryDirectory() as tmp:
            store = StudioStore(home=tmp)
            store.install_zip(_theme_package(), tier="from-file")
            store.install_zip(_plugin_package(), tier="from-file")
            store.set_active_theme("studio", "org.example.midnight")
            server = self._server(tmp)
            try:
                no_auth = server.get("/studio/extensions")
                self.assertEqual(no_auth.status, 401)
                no_auth.read()
                resp = server.get("/studio/extensions",
                                  token="<redacted>")
                payload = resp.read()
                self.assertEqual(resp.status, 200, payload)
                body = payload.decode("utf-8")
                self.assertIn("org.example.midnight", body)
                self.assertIn("org.example.photo-notes", body)
                self.assertIn("Sample theme", body)
            finally:
                server.close()

    def test_photo_page_wears_theme_and_panel_over_http(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.store import PhotoStore
        from nomorals.studio.store import StudioStore

        with tempfile.TemporaryDirectory() as tmp:
            store = StudioStore(home=tmp)
            store.install_zip(_theme_package(), tier="from-file")
            store.install_zip(_plugin_package(), tier="from-file")
            store.set_active_theme("studio", "org.example.midnight")
            photo_store = PhotoStore(home=tmp)
            doc = PhotoDocument.from_psd_bytes(RGB8.read_bytes())
            doc_id = photo_store.save(doc, title="baseline")
            server = self._server(tmp)
            try:
                resp = server.get(f"/studio/photo/{doc_id}",
                                  token="<redacted>")
                payload = resp.read()
                self.assertEqual(resp.status, 200, payload)
                body = payload.decode("utf-8")
                self.assertIn("#111111", body)
                self.assertIn("#f2f2f2", body)
                self.assertIn("Photo notes", body)
                self.assertIn("Sample theme", body)
            finally:
                server.close()


class WiringTests(unittest.TestCase):
    def test_extensions_verb_registered(self) -> None:
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
        )

        # /extensions is the Studio verb. /studio stays the AniList
        # studio lookup — the two must never collide again.
        self.assertIn("extensions", CONTROL_COMMANDS)
        details = COMMAND_DETAILS["extensions"]
        self.assertTrue(details["what"])
        self.assertTrue(details["usage"])
        self.assertTrue(details["example"])
        self.assertIn("extensions", LIST_ONELINERS)
        slots = [group for group, kinds in LIST_GROUPS
                 if "extensions" in kinds]
        self.assertEqual(len(slots), 1,
                         f"extensions needs exactly one LIST_GROUPS "
                         f"slot, found {slots}")
        # owner-only like /photo: absent from PUBLIC_COMMANDS.
        self.assertNotIn("extensions", PUBLIC_COMMANDS)
        self.assertIn("studio", CONTROL_COMMANDS)
        self.assertIn("AniList",
                      COMMAND_DETAILS["studio"]["what"])

    def test_runtime_dispatches_extensions(self) -> None:
        from pathlib import Path as _Path

        src = (_Path(__file__).resolve().parent.parent
               / "nomorals" / "agents" / "partner" / "runtime.py"
               ).read_text()
        self.assertIn('kind == "extensions"', src)
        meta = (_Path(__file__).resolve().parent.parent
                / "nomorals" / "agents" / "partner" / "runtime_meta.py"
                ).read_text()
        self.assertIn("def _control_extensions", meta)

    def test_layering_names_studio(self) -> None:
        from pathlib import Path as _Path

        src = (_Path(__file__).resolve().parent.parent
               / "tests" / "test_layering.py").read_text()
        self.assertIn('"studio": 4', src)

    def test_extensions_board_renders(self) -> None:
        from nomorals.studio.page import extensions_board_html
        from nomorals.studio.store import StudioStore

        with tempfile.TemporaryDirectory() as tmp:
            store = StudioStore(home=tmp)
            store.install_zip(_theme_package(), tier="from-file")
            store.install_zip(_plugin_package(), tier="from-file")
            html = extensions_board_html(store)
            self.assertIn("org.example.midnight", html)
            self.assertIn("org.example.photo-notes", html)
            self.assertIn("theme", html)
            self.assertIn("plugin", html)


if __name__ == "__main__":
    unittest.main()
