"""Pins for the FontLibrary programme (font census S1–S3):
the stocked library, the registry, provisioning, the
typography service's registry door, and the /fonts surface.

Fixtures are small synthetic packs built over host DejaVu
faces — the suite never depends on the real ~2.5 GB pack.
The vendored core subset, though, is real and in the repo:
its pins read the shipped manifest itself.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
from pathlib import Path

import pytest

from nomorals.media.typography import fontlib, fontpack
from nomorals.media.typography.fontlib import (
    FontLibrary,
    FontNotStocked,
)

_DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
pytestmark = pytest.mark.skipif(
    not (_DEJAVU / "DejaVuSans.ttf").is_file(),
    reason="host DejaVu faces are the fixture source")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pack(root: Path, *, kind: str = "full") -> Path:
    """A two-family synthetic pack with a truthful manifest."""
    root.mkdir(parents=True, exist_ok=True)
    faces = []

    def add_family(name, category, files):
        fam_faces = []
        for rel, src, weight, style, variable, wrange in files:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            record = {
                "file": rel, "weight": weight, "style": style,
                "variable": variable,
                "sha256": _sha(dest),
                "bytes": dest.stat().st_size, "scripts": ["Latin"],
            }
            if wrange:
                record["weight_range"] = list(wrange)
            fam_faces.append(record)
        license_rel = str(Path(rel).parent / "OFL.txt")
        (root / license_rel).write_text(
            "SIL Open Font License 1.1\nReserved Font Name Test\n",
            encoding="utf-8")
        faces.append({
            "name": name, "category": category,
            "license": "OFL-1.1", "license_file": license_rel,
            "reserved_font_name": True,
            "dir": str(Path(rel).parent), "subsets": ["latin"],
            "faces": fam_faces,
        })

    add_family("DejaVu Sans", "Sans Serif", [
        ("ofl/testsans/TestSans-Regular.ttf",
         _DEJAVU / "DejaVuSans.ttf", 400, "normal", False, None),
        ("ofl/testsans/TestSans-Bold.ttf",
         _DEJAVU / "DejaVuSans-Bold.ttf", 700, "normal", False, None),
    ])
    add_family("DejaVu Serif", "Serif", [
        ("ofl/testserif/TestSerif-Regular.ttf",
         _DEJAVU / "DejaVuSerif.ttf", 400, "normal", False, None),
    ])
    add_family("DejaVu Sans Mono", "Monospace", [
        ("ofl/testvar/TestVar[wght].ttf",
         _DEJAVU / "DejaVuSans.ttf", 400, "normal", True, (100, 900)),
    ])
    manifest = {
        "format": 1, "kind": kind,
        "source": {"repo": "google/fonts", "commit": "testcommit"},
        "family_count": len(faces),
        "file_count": sum(len(f["faces"]) for f in faces),
        "total_bytes": sum(fa["bytes"] for f in faces
                           for fa in f["faces"]),
        "families": faces,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return root


@pytest.fixture()
def pack_dir(tmp_path):
    return _write_pack(tmp_path / "pack")


@pytest.fixture()
def core_dir(tmp_path):
    root = tmp_path / "corefonts"
    root.mkdir(parents=True)
    dest = root / "CoreMono.ttf"
    shutil.copyfile(_DEJAVU / "DejaVuSansMono.ttf", dest)
    manifest = {
        "format": 1, "kind": "core",
        "source": {"repo": "google/fonts", "commit": "testcommit"},
        "family_count": 1, "file_count": 1,
        "total_bytes": dest.stat().st_size,
        "families": [{
            "name": "DejaVu Sans", "category": "Sans Serif",
            "license": "OFL-1.1", "license_file": "",
            "reserved_font_name": False, "dir": ".",
            "subsets": ["latin"],
            "faces": [{"file": "CoreMono.ttf", "weight": 400,
                       "style": "normal", "variable": False,
                       "sha256": _sha(dest),
                       "bytes": dest.stat().st_size,
                       "scripts": ["Latin"]}],
        }],
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return root


# ── registry resolution ──────────────────────────────────────

class TestResolve:
    def test_resolve_from_pack_tier(self, pack_dir):
        lib = FontLibrary(pack_dir=pack_dir)
        path = lib.resolve("DejaVu Sans")
        assert path.is_file()
        assert path.name == "TestSans-Regular.ttf"
        assert lib.family("dejavu sans").tier == "pack"

    def test_resolve_weight_picks_closest_face(self, pack_dir):
        lib = FontLibrary(pack_dir=pack_dir)
        assert lib.resolve("DejaVu Sans", weight=700).name == \
            "TestSans-Bold.ttf"
        assert lib.resolve("DejaVu Sans", weight=600).name == \
            "TestSans-Bold.ttf"
        assert lib.resolve("DejaVu Sans", weight=300).name == \
            "TestSans-Regular.ttf"

    def test_resolve_variable_covers_any_weight(self, pack_dir):
        lib = FontLibrary(pack_dir=pack_dir)
        path = lib.resolve("DejaVu Sans Mono", weight=250)
        assert path.name == "TestVar[wght].ttf"

    def test_resolve_squashed_name(self, pack_dir):
        lib = FontLibrary(pack_dir=pack_dir)
        assert lib.resolve("dejavusans").name == \
            "TestSans-Regular.ttf"

    def test_tier_precedence_pack_over_core(self, pack_dir, core_dir):
        lib = FontLibrary(pack_dir=pack_dir, core_dir=core_dir)
        assert lib.resolve("DejaVu Sans").name == \
            "TestSans-Regular.ttf"
        core_only = FontLibrary(core_dir=core_dir)
        assert core_only.resolve("DejaVu Sans").name == "CoreMono.ttf"

    def test_named_miss(self, pack_dir):
        lib = FontLibrary(pack_dir=pack_dir)
        with pytest.raises(FontNotStocked) as caught:
            lib.resolve("Comic Papyrus Deluxe")
        exc = caught.value
        assert exc.family == "Comic Papyrus Deluxe"
        text = str(exc)
        assert "not in the stocked library" in text
        assert "pack" in text
        assert not lib.stocked("Comic Papyrus Deluxe")

    def test_search_and_categories(self, pack_dir):
        lib = FontLibrary(pack_dir=pack_dir)
        assert [f.name for f in lib.search("dejavu s")] == \
            ["DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif"]
        cats = lib.categories()
        assert cats["Sans Serif"] == 1
        assert cats["Monospace"] == 1
        assert cats["Serif"] == 1


class TestLocalTier:
    def test_owner_dropped_font_resolves(self, tmp_path):
        drop = tmp_path / "dropped"
        drop.mkdir()
        shutil.copyfile(_DEJAVU / "DejaVuSans.ttf",
                        drop / "whatever.ttf")
        lib = FontLibrary(local_dirs=[drop])
        path = lib.resolve("DejaVu Sans")
        assert path.name == "whatever.ttf"
        stats = lib.stats()
        local = [t for t in stats["tiers"] if t["tier"] == "local"]
        assert local and local[0]["families"] == 1


# ── manifest verification + stocking ─────────────────────────

class TestVerifyAndStock:
    def test_verify_counts_and_sampled_hash(self, pack_dir):
        report = fontpack.verify_pack(pack_dir, sample=2)
        assert report["ok"] is True
        assert report["families"] == 3
        assert report["files"] == 4
        assert report["checked"] >= 2

    def test_verify_full_catches_tamper(self, tmp_path):
        pack = _write_pack(tmp_path / "pack2")
        target = pack / "ofl/testserif/TestSerif-Regular.ttf"
        blob = bytearray(target.read_bytes())
        blob[100] ^= 0xFF
        target.write_bytes(bytes(blob))
        report = fontpack.verify_pack(pack, full=True)
        assert report["ok"] is False
        assert any("DejaVu Serif" in bad for bad in report["bad"])

    def test_stock_into_home_and_discover(self, pack_dir, tmp_path,
                                          monkeypatch):
        home = tmp_path / "home"
        report = fontpack.stock_pack(pack_dir, home=home)
        assert report["families"] == 3
        dest = fontlib.estate_library_dir(home)
        assert (dest / "MANIFEST.json").is_file()
        monkeypatch.setenv("NM_HOME", str(home))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        lib = FontLibrary.discover()
        assert lib.family("DejaVu Sans").tier == "pack"
        assert lib.resolve("DejaVu Serif").is_file()

    def test_stock_refuses_tampered_pack(self, tmp_path):
        pack = _write_pack(tmp_path / "pack3")
        target = pack / "ofl/testsans/TestSans-Bold.ttf"
        blob = bytearray(target.read_bytes())
        blob[50] ^= 0xFF
        target.write_bytes(bytes(blob))
        with pytest.raises(fontpack.FontPackError) as caught:
            fontpack.stock_pack(pack, home=tmp_path / "home3")
        assert "refusing to stock" in str(caught.value)


# ── the typography service door ──────────────────────────────

class TestServiceDoor:
    def test_named_miss_raises(self, tmp_path):
        from nomorals.media.typography import TypographyService

        empty = tmp_path / "empty"
        empty.mkdir()
        service = TypographyService(
            roots=[empty], store_dir=tmp_path / "store")
        with pytest.raises(FontNotStocked):
            service.select(family="Zzz No Such Family")

    def test_offline_render_from_stock(self, pack_dir, tmp_path,
                                       monkeypatch):
        from nomorals.media.typography import TypographyService

        def _blocked(*args, **kwargs):
            raise AssertionError("network touched during render")

        monkeypatch.setattr(socket, "socket", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)
        service = TypographyService(
            roots=[pack_dir], store_dir=tmp_path / "store")
        font = service.pil_font(family="DejaVu Sans", size=40)
        assert font is not None
        out = service.render_text("Stocked and offline",
                                  family="DejaVu Sans", size=32)
        assert out.is_file()
        assert out.suffix == ".png"

    def test_core_subset_alone_renders(self, tmp_path):
        from nomorals.media.typography import TypographyService

        core = fontlib.core_fonts_dir()
        assert (core / "MANIFEST.json").is_file()
        service = TypographyService(
            roots=[core], store_dir=tmp_path / "store")
        selection = service.select("text")
        assert selection.face is not None
        assert str(selection.face.path).startswith(str(core))
        assert service.pil_font("text", 36) is not None


class TestVendoredCore:
    def test_core_tier_is_real_stock(self):
        lib = FontLibrary(core_dir=fontlib.core_fonts_dir())
        stats = lib.stats()
        assert stats["families"] >= 20
        cats = lib.categories()
        assert "Monospace" in cats
        assert "Handwriting" in cats
        assert "Display" in cats

    def test_core_resolves_spread(self):
        lib = FontLibrary(core_dir=fontlib.core_fonts_dir())
        for name in ("Inter", "Playfair Display",
                     "JetBrains Mono", "Dancing Script"):
            path = lib.resolve(name)
            assert path.is_file(), name


# ── the /fonts surface + catalogue ───────────────────────────

class TestFontsSurface:
    def _runtime(self):
        from nomorals.agents.partner.runtime_fonts import (
            RuntimeFontsMixin,
        )

        class _Stub(RuntimeFontsMixin):
            def _is_operator(self, message):
                return True

        return _Stub()

    def test_surface_answers_from_real_manifest(
            self, pack_dir, tmp_path, monkeypatch):
        fontpack.stock_pack(pack_dir, home=tmp_path / "home")
        monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        runtime = self._runtime()
        overview = runtime._control_fonts("fonts", "", None)
        assert "Font library" in overview
        hits = runtime._control_fonts("fonts", "search DejaVu", None)
        assert "DejaVu Sans" in hits
        assert "DejaVu Serif" in hits
        info = runtime._control_fonts(
            "fonts", "info DejaVu Sans", None)
        assert "OFL-1.1" in info
        assert "TestSans-Regular.ttf" in info
        missing = runtime._control_fonts(
            "fonts", "info Zzz Absent", None)
        assert "not in the stocked" in missing

    def test_catalogue_registration(self):
        from nomorals.social.chat import control

        assert "fonts" in control.CONTROL_COMMANDS
        assert "fonts" in control.COMMAND_DETAILS
        assert "fonts" in control.LIST_ONELINERS
        assert any("fonts" in kinds
                   for _title, kinds in control.LIST_GROUPS)
        assert "fonts" not in control.PUBLIC_COMMANDS

    def test_cli_parser_knows_fonts(self):
        from nomorals.cmdline import _parser

        args = _parser().parse_args(
            ["fonts", "verify", "--pack", "/tmp/pack"])
        assert args.fonts_action == "verify"
        assert args.pack_dir == "/tmp/pack"
        args = _parser().parse_args(
            ["fonts", "stock", "--pack", "/tmp/pack", "--build"])
        assert args.fonts_action == "stock"
        assert args.build is True
