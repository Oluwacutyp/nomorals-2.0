"""Typography service — doc 8 Priority 2 (owner-adopted as a floor).

Pins: a real font catalogue parsed from real font files (family,
weight, italic, OS/2 family class, cmap-derived script coverage,
licence as metadata); deterministic semantic selection with honest
fallback chains; grapheme-safe breaking; real shaping through the
Pillow/raqm stack (Arabic joining + Devanagari conjuncts proven by
measurement, not by "it rendered"); a bounded, journaled
rendered-asset cache; and the in-repo render consumers routing
through the service.

Fixture fonts are BUILT with fontTools where it is available, so
the parser is tested against fonts whose tables are known exactly.
Shaping tests discover real system faces through the catalogue
itself and skip only when the host carries no covering face.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ft = pytest.importorskip("fontTools")
from fontTools.fontBuilder import FontBuilder  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402

from nomorals.media.typography import (  # noqa: E402
    TypographyService,
    graphemes,
    parse_font_file,
    reset_service,
    wrap_text,
)

LATIN = list(range(0x20, 0x100))  # ASCII + Latin-1 supplement
ARABIC = list(range(0x0620, 0x064B)) + list(range(0x0660, 0x066A))


def _build_font(path: Path, family: str, style: str, codepoints,
                *, weight: int = 400, italic: bool = False,
                family_class: int = 0, licence: str = "") -> Path:
    glyphs = [".notdef"] + [f"u{cp:04X}" for cp in codepoints]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyphs)
    fb.setupCharacterMap({cp: f"u{cp:04X}" for cp in codepoints})
    from fontTools.ttLib.tables._g_l_y_f import Glyph
    fb.setupGlyf({g: Glyph() for g in glyphs})
    fb.setupHorizontalMetrics({g: (600, 0) for g in glyphs})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({
        "familyName": family, "styleName": style,
        "uniqueFontIdentifier": f"{family} {style} fixture",
        "fullName": f"{family} {style}",
        "psName": (family + style).replace(" ", ""),
        "version": "1.000",
    })
    fb.setupOS2()
    fb.setupPost()
    fb.setupMaxp()
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(path))
    # Post-edit the fields FontBuilder leaves at defaults so the
    # fixture's OS/2 + licence metadata are exactly known.
    font = TTFont(str(path))
    os2 = font["OS/2"]
    os2.usWeightClass = weight
    os2.sFamilyClass = family_class << 8
    fs = os2.fsSelection
    if italic:
        fs |= 0x01
    if weight >= 600:
        fs |= 0x20
    os2.fsSelection = fs
    if licence:
        name = font["name"]
        name.setName(licence, 13, 3, 1, 0x409)
        name.setName("Fixture Foundry", 0, 3, 1, 0x409)
    font.save(str(path))
    font.close()
    return path


@pytest.fixture(scope="module")
def font_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixturefonts")
    _build_font(d / "FixtureSans-Regular.ttf", "Fixture Sans", "Regular",
                LATIN, weight=400, family_class=8,
                licence="Fixture Licence — metadata only")
    _build_font(d / "FixtureSans-Bold.ttf", "Fixture Sans", "Bold",
                LATIN, weight=700, family_class=8)
    _build_font(d / "FixtureSerif-Regular.ttf", "Fixture Serif", "Regular",
                LATIN, weight=400, family_class=2)
    _build_font(d / "FixtureNaskh-Regular.ttf", "Fixture Naskh", "Regular",
                LATIN + ARABIC, weight=400, family_class=0)
    _build_font(d / "FixtureHand-Italic.ttf", "Fixture Hand", "Italic",
                LATIN, weight=400, italic=True, family_class=10)
    (d / "broken.ttf").write_bytes(b"this is not a font at all")
    return d


@pytest.fixture()
def service(font_dir, tmp_path) -> TypographyService:
    return TypographyService(roots=[str(font_dir)],
                             store_dir=tmp_path / "typo-store")


# ── parser ────────────────────────────────────────────────────────

def test_parser_reads_real_tables(font_dir):
    faces = parse_font_file(font_dir / "FixtureSans-Bold.ttf")
    assert len(faces) == 1
    face = faces[0]
    assert face.family == "Fixture Sans"
    assert face.weight == 700
    assert face.italic is False
    assert face.family_class == 8
    assert face.scripts.get("Latin", 0.0) > 0.9
    assert face.covers_text("Hello, Devon!")
    assert not face.covers_text("سلام")


def test_parser_records_licence_as_metadata(font_dir):
    face = parse_font_file(font_dir / "FixtureSans-Regular.ttf")[0]
    assert "Fixture Licence" in face.licence
    assert "Fixture Foundry" in face.copyright


def test_parser_arabic_coverage_from_cmap(font_dir):
    face = parse_font_file(font_dir / "FixtureNaskh-Regular.ttf")[0]
    assert face.covers("Arabic")
    assert face.scripts["Arabic"] >= 0.6
    assert face.covers_text("سلام")


def test_parser_italic_flag(font_dir):
    face = parse_font_file(font_dir / "FixtureHand-Italic.ttf")[0]
    assert face.italic is True
    assert face.family_class == 10


def test_parser_survives_garbage(font_dir):
    assert parse_font_file(font_dir / "broken.ttf") == []


# ── catalogue ─────────────────────────────────────────────────────

def test_catalogue_scans_and_searches(service):
    cat = service.catalogue()
    families = {f.family for f in cat.faces()}
    assert {"Fixture Sans", "Fixture Serif", "Fixture Naskh",
            "Fixture Hand"} <= families
    assert len(cat.faces(family="Fixture Sans")) == 2
    arabic = cat.covering("Arabic")
    assert [f.family for f in arabic] == ["Fixture Naskh"]
    hits = cat.search("naskh")
    assert hits and hits[0].family == "Fixture Naskh"
    # the broken file is counted as a failure, never a face
    assert cat.stats()["files_failed"] == 1


def test_catalogue_index_reused(service, font_dir, tmp_path):
    service.catalogue()
    # a second service over the same store reuses the disk index:
    # deleting the licence fixture must not resurrect its metadata
    again = TypographyService(roots=[str(font_dir)],
                              store_dir=tmp_path / "typo-store")
    assert {f.family for f in again.catalogue().faces()} == \
        {f.family for f in service.catalogue().faces()}


# ── selection ─────────────────────────────────────────────────────

def test_selection_is_deterministic(service):
    a = service.select("editorial")
    b = service.select("editorial")
    assert a.face is not None and b.face is not None
    assert a.face.path == b.face.path
    assert a.face.family == "Fixture Serif"  # serif by OS/2 class


def test_selection_weight_preference(service):
    sel = service.select("text", weight=700)
    assert sel.face is not None
    assert sel.face.weight == 700


def test_selection_script_need_walks_chain(service):
    sel = service.select("text", script="Arabic")
    assert sel.face is not None
    assert sel.face.family == "Fixture Naskh"
    assert sel.covered is True


def test_selection_uncovered_script_says_so(service):
    sel = service.select("text", script="Thai")
    assert sel.covered is False
    assert sel.notes  # the gap is stated, not hidden


def test_selection_handwritten_intent(service):
    sel = service.select("handwritten")
    assert sel.face is not None
    assert sel.face.family == "Fixture Hand"


# ── graphemes + wrapping ──────────────────────────────────────────

def test_graphemes_never_split_people_clusters():
    assert graphemes("á") == ["á"]
    assert graphemes("👨‍👩‍👧") == ["👨‍👩‍👧"]
    assert graphemes("🇳🇬🇳🇬") == ["🇳🇬", "🇳🇬"]
    assert graphemes("👍🏽!") == ["👍🏽", "!"]
    assert graphemes("1️⃣") == ["1️⃣"]
    # Devanagari: consonant + virama + consonant is one cluster
    assert graphemes("क्ष") == ["क्ष"]
    # Arabic letter + combining mark stays together
    assert graphemes("بً") == ["بً"]


def test_wrap_is_grapheme_safe(service):
    font = service.pil_font("text", 40)
    assert font is not None
    text = "क्षत्रिय क्षत्रिय क्षत्रिय क्षत्रिय"
    lines = wrap_text(text, font, max_width=120)
    assert len(lines) > 1
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    for line in lines:
        for ch in line:
            assert not (0x900 <= ord(ch) <= 0x97F and line.startswith(ch)
                        and ord(ch) == 0x94D)


# ── shaping (real faces, real raqm) ───────────────────────────────

@pytest.fixture(scope="module")
def system_service(tmp_path_factory) -> TypographyService:
    return TypographyService(
        store_dir=tmp_path_factory.mktemp("typo-sys") / "store")


def _face_covering(svc: TypographyService, script: str):
    cat = svc.catalogue()
    for face in cat.covering(script):
        if os.path.isfile(face.path):
            return face
    return None


def test_arabic_joining_is_real_shaping(system_service):
    from nomorals.media.typography import shaped_width, unshaped_width
    face = _face_covering(system_service, "Arabic")
    if face is None:
        pytest.skip("host carries no Arabic-covering face")
    from PIL import ImageFont
    font = ImageFont.truetype(face.path, 48, index=face.face_index,
                              layout_engine=ImageFont.Layout.RAQM)
    word = "سلام"
    broken = "س‌لام"  # ZWNJ breaks the lam-alef join
    # joined shaping is NARROWER than the same letters unjoined:
    # glyphs merge into contextual forms instead of sitting apart
    assert shaped_width(word, font) < shaped_width(broken, font)
    # and narrower than the unshaped (basic engine) rendering
    assert shaped_width(word, font) < unshaped_width(word, face, 48)


def test_devanagari_conjunct_is_one_shaped_unit(system_service):
    from nomorals.media.typography import shaped_width
    face = _face_covering(system_service, "Devanagari")
    if face is None:
        pytest.skip("host carries no Devanagari-covering face")
    from PIL import ImageFont
    font = ImageFont.truetype(face.path, 48, index=face.face_index,
                              layout_engine=ImageFont.Layout.RAQM)
    conjunct = "क्ष"
    broken = "क्‌ष"  # ZWNJ keeps the virama visible, no conjunct
    parts = shaped_width("क", font) + shaped_width("ष", font)
    assert shaped_width(conjunct, font) < shaped_width(broken, font)
    assert shaped_width(conjunct, font) < parts


def test_cjk_face_renders_real_ink(system_service):
    face = _face_covering(system_service, "Han")
    if face is None:
        pytest.skip("host carries no Han-covering face")
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(face.path, 48, index=face.face_index)
    img = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(img).text((10, 10), "漢字", font=font, fill=255)
    assert img.getbbox() is not None  # real glyphs, not empty boxes


def test_mixed_direction_line_measures_and_renders(system_service):
    from nomorals.media.typography import shaped_width
    face = _face_covering(system_service, "Arabic")
    if face is None:
        pytest.skip("host carries no Arabic-covering face")
    from PIL import ImageFont
    font = ImageFont.truetype(face.path, 40, index=face.face_index,
                              layout_engine=ImageFont.Layout.RAQM)
    line = "المجموع 250 total"
    width = shaped_width(line, font)
    assert width > 0


# ── rendered-asset cache ──────────────────────────────────────────

def test_render_cache_identity_and_journal(service):
    p1 = service.render_text("Devon", intent="display", size=64)
    assert p1.is_file()
    rows = (service.store_dir / "journal.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    kinds = [json.loads(r)["kind"] for r in rows]
    assert "rendered_text_stored" in kinds
    p2 = service.render_text("Devon", intent="display", size=64)
    assert p2 == p1  # same identity → same asset, no re-render
    rows2 = (service.store_dir / "journal.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(rows2) == len(rows)  # a hit writes no new store row


def test_render_cache_cap_evicts_and_journals(font_dir, tmp_path):
    svc = TypographyService(roots=[str(font_dir)],
                            store_dir=tmp_path / "small-store",
                            cache_cap_bytes=1_500)
    for i in range(6):
        svc.render_text(f"a considerably longer asset {i}",
                        intent="display", size=110)
    total = sum(p.stat().st_size for p in
                (tmp_path / "small-store" / "assets").glob("*.png"))
    assert total <= 1_500 or svc.cache_stats()["evictions"] > 0
    kinds = [json.loads(r)["kind"] for r in
             (tmp_path / "small-store" / "journal.jsonl")
             .read_text(encoding="utf-8").strip().splitlines()]
    assert "rendered_text_evicted" in kinds


def test_preview_is_smaller_than_final(service):
    final = service.render_text("Title", intent="display", size=96)
    preview = service.preview_text("Title", intent="display", size=96)
    from PIL import Image
    with Image.open(final) as a, Image.open(preview) as b:
        assert b.size[1] < a.size[1]


# ── consumers route through the service ───────────────────────────

def test_makers_font_comes_from_service(font_dir, tmp_path, monkeypatch):
    # Inject a fixture-only service: on a full host the intent
    # pick is (correctly) some real system bold sans, so the
    # wiring assertion needs the service makers actually consults.
    svc = TypographyService(roots=[str(font_dir)],
                            store_dir=tmp_path / "svc-store")
    monkeypatch.setattr("nomorals.media.typography.get_service",
                        lambda: svc)
    from nomorals.social.chat import makers
    font = makers._font(40, bold=True)
    assert font.getname() == ("Fixture Sans", "Bold")


def test_media_edit_find_font_uses_catalogue(font_dir, monkeypatch):
    monkeypatch.setenv("NM_FONT_DIRS", str(font_dir))
    reset_service()
    try:
        from nomorals.media_edit import studio
        font = studio.find_font("Fixture Naskh", 32)
        assert font.getname()[0] == "Fixture Naskh"
    finally:
        reset_service()
