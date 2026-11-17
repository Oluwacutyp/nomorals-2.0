"""Failing-first tests: AlternativeTo software-alternatives source.

Site #4 in the 52-websites verdict table (alternativeto.net): a
crowd-ranked software-alternatives directory. Devon's native organ is
``nomorals/sources/software.py`` — a software catalogue
(SoftwareRecord + AlternativeEdge) seeded from the live site's real
shape, grown by live fetch, queryable by software name/category with
platform/license filters, and registered as a source the brain can use.

Fixtures below are reconstructed from the LIVE pages fetched
2026-11-17 (alternatives to Adobe Photoshop + search for
"photoshop"): same DOM order, same fields (likes, description,
Cost/License, Application type, Since, Platforms, comments).
"""

import json

from nomorals.sources.software import (
    AlternativeToSource,
    SoftwareCatalog,
    SoftwareRecord,
    format_alternatives_reply,
    parse_alternatives_page,
    parse_search_page,
    resolve_slug,
)

# ── fixtures: live-page shape, 2026-11-17 ──────────────────────────────────

ALT_PHOTOSHOP_HTML = """<html><body>
<ol class="alternatives">
<li>
<h3><a href="/software/gimp/">GIMP</a></h3>
<div class="likes">4888 likes</div>
<p>Free open-source suite offering advanced photo retouching, image
composition, original artwork creation, batch processing, plugin
support, custom scripting, mass image format conversion, and
integration with free publishing tools for comprehensive editing
solutions.</p>
<a href="/software/gimp/">238 GIMP alternatives</a>
<h4>Cost / License</h4>
<ul><li>Free</li><li>Open Source (<a href="https://choosealicense.com/licenses/gpl-3.0/">GPL-3.0</a>)</li></ul>
<h4>Application type</h4>
<ul><li>Image Editor</li><li>Digital Painting</li></ul>
<h4>Since</h4>
<ul><li>Since 2009</li></ul>
<h4>Platforms</h4>
<ul><li>Windows</li><li>Mac</li><li>Linux</li><li>BSD</li><li>Flathub</li><li>Snapcraft</li><li>Flatpak</li><li>AppImage</li></ul>
<p>Good alternative? Is <b>GIMP</b> a good alternative to Adobe Photoshop?</p>
<p>116 comments</p>
</li>
<li>
<h3><a href="/software/krita/">Krita</a></h3>
<div class="likes">1520 likes</div>
<p>Free, open-source painting software designed by artists for artists.
Offers customizable UI, 9 brush engines, color management, tablet
support, non-destructive editing, and privacy features.</p>
<a href="/software/krita/">175 Krita alternatives</a>
<h4>Cost / License</h4>
<ul><li>Free</li><li>Open Source</li></ul>
<h4>Application type</h4>
<ul><li>Digital Painting</li></ul>
<h4>Since</h4>
<ul><li>Since 2009</li></ul>
<h4>Platforms</h4>
<ul><li>Windows</li><li>Mac</li><li>Linux</li><li>Android Tablet</li><li>F-Droid</li><li>AppImage</li></ul>
<p>Good alternative? Is <b>Krita</b> a good alternative to Adobe Photoshop?</p>
<p>49 comments</p>
</li>
<li>
<h3><a href="/software/affinity/">Affinity</a></h3>
<div class="likes">99 likes</div>
<p>Unified design solution merging advanced photo, vector, and layout
tools for seamless workflow, non-destructive editing, RAW support,
batch processing, AI enhancements, GPU acceleration.</p>
<a href="/software/affinity/">122 Affinity alternatives</a>
<h4>Cost / License</h4>
<ul><li>Freemium</li><li>Proprietary</li></ul>
<h4>Application type</h4>
<ul><li>Image Editor</li><li>Vector Graphic</li></ul>
<h4>Since</h4>
<ul><li>Since 2025</li></ul>
<h4>Platforms</h4>
<ul><li>Windows</li><li>Mac</li><li>iPad</li></ul>
<p>Good alternative? Is <b>Affinity</b> a good alternative to Adobe Photoshop?</p>
<p>6 comments</p>
</li>
<li>
<h3><a href="/software/photocraft/">PhotoCraft</a></h3>
<div class="likes">66 likes</div>
<p>Layers, masks, adjustment layers, layer styles, type, vectors and
brushes in a native app written entirely in Rust. Open source, offline,
and yours.</p>
<a href="/software/photocraft/">16 PhotoCraft alternatives</a>
<p>Good alternative? Is <b>PhotoCraft</b> a good alternative to Adobe Photoshop?</p>
<p>1 comment</p>
</li>
</ol>
<p>12 of 277 Adobe Photoshop alternatives</p>
</body></html>
"""

SEARCH_PHOTOSHOP_HTML = """<html><body>
<ol class="results">
<li>
<h3><a href="/software/adobe-photoshop/">Adobe Photoshop</a></h3>
<div class="likes">2469 likes</div>
<p>Adobe Photoshop is a proprietary raster graphics editor developed
and published by Adobe for Windows and macOS.</p>
<h4>Cost / License</h4>
<ul><li>Paid</li><li>Proprietary</li></ul>
<h4>Application type</h4>
<ul><li>Image Editor</li><li>Digital Painting Tool</li></ul>
<h4>Since</h4>
<ul><li>Since 1990</li></ul>
<h4>Platforms</h4>
<ul><li>Windows</li><li>Mac</li><li>Online</li><li>Android</li><li>iPhone</li><li>iPad</li></ul>
</li>
<li>
<h3><a href="/software/adobe-photoshop-touch/">Adobe Photoshop Touch</a></h3>
<div class="likes">29 likes</div>
<p>Desktop-based photo editing software with advanced controls,
offering tools for enhancing, organizing, and sharing photos.</p>
<h4>Cost / License</h4>
<ul><li>Paid</li><li>Proprietary</li></ul>
<h4>Alerts</h4>
<ul><li>Discontinued</li></ul>
<h4>Platforms</h4>
<ul><li>Android</li><li>iPhone</li><li>iPad</li></ul>
</li>
</ol>
</body></html>
"""


def _by_name(records, name):
    for rec in records:
        if rec.name == name:
            return rec
    raise AssertionError(f"{name!r} not in {[r.name for r in records]}")


# ── parsers: the live shape ────────────────────────────────────────────────

def test_parse_alternatives_page_photoshop_shape():
    records, total = parse_alternatives_page(ALT_PHOTOSHOP_HTML)
    assert total == 277
    assert [r.name for r in records] == [
        "GIMP", "Krita", "Affinity", "PhotoCraft"]
    gimp = _by_name(records, "GIMP")
    assert gimp.slug == "gimp"
    assert gimp.likes == 4888
    assert gimp.license_tier == "Free"
    assert gimp.open_source is True
    assert gimp.license_id == "GPL-3.0"
    assert gimp.categories == ["Image Editor", "Digital Painting"]
    assert gimp.since_year == 2009
    assert "Linux" in gimp.platforms and "Flathub" in gimp.platforms
    assert gimp.comments == 116
    assert gimp.alternatives_count == 238
    assert gimp.url == "https://alternativeto.net/software/gimp/"
    assert "retouching" in gimp.description
    krita = _by_name(records, "Krita")
    assert krita.likes == 1520 and krita.open_source is True
    affinity = _by_name(records, "Affinity")
    assert affinity.license_tier == "Freemium"
    assert affinity.open_source is False
    assert affinity.license_id == ""
    assert affinity.since_year == 2025
    # Sparse entry: no Cost/License/Platforms sections at all — the
    # parser must not die, it records what the page carries.
    craft = _by_name(records, "PhotoCraft")
    assert craft.likes == 66
    assert craft.license_tier == ""
    assert craft.platforms == []
    assert craft.comments == 1


def test_parse_search_page_photoshop_shape():
    records = parse_search_page(SEARCH_PHOTOSHOP_HTML)
    assert [r.name for r in records] == [
        "Adobe Photoshop", "Adobe Photoshop Touch"]
    ps = records[0]
    assert ps.slug == "adobe-photoshop"
    assert ps.likes == 2469
    assert ps.license_tier == "Paid"
    assert ps.open_source is False
    assert ps.since_year == 1990
    assert ps.discontinued is False
    touch = records[1]
    assert touch.discontinued is True
    assert touch.likes == 29


def test_parse_empty_and_garbage_never_raise():
    assert parse_alternatives_page("") == ([], 0)
    assert parse_alternatives_page("<html><body>no apps here</body></html>") == ([], 0)
    assert parse_search_page("") == []
    assert parse_search_page("<html></html>") == []


# ── slug resolution ────────────────────────────────────────────────────────

def test_resolve_slug_seed_map():
    assert resolve_slug("Photoshop") == "adobe-photoshop"
    assert resolve_slug("adobe photoshop") == "adobe-photoshop"
    assert resolve_slug("GIMP") == "gimp"
    assert resolve_slug("Microsoft Office") == "microsoft-office"
    assert resolve_slug("notion") == "notion"
    assert resolve_slug("slack") == "slack"
    assert resolve_slug("zoom") == "zoom"


def test_resolve_slug_search_fallback_uses_live_search():
    def fake_fetch(url):
        assert "browse/search" in url and "q=foobar" in url
        return SEARCH_PHOTOSHOP_HTML

    assert resolve_slug("foobar", fetcher=fake_fetch) == "adobe-photoshop"


def test_resolve_slug_unknown_without_network_is_none():
    assert resolve_slug("definitely-not-a-real-app-xyz") is None


# ── seed catalogue: the query the owner asks ───────────────────────────────

def test_seed_catalogue_photoshop_linux_free(tmp_path):
    cat = SoftwareCatalog(tmp_path)
    results = cat.alternatives_for(
        "adobe-photoshop", platform="linux", license_tier="free")
    assert results, "seed catalogue must answer Photoshop at all"
    assert results[0].name == "GIMP"
    assert all("Linux" in r.platforms for r in results)
    assert all(r.license_tier == "Free" for r in results)
    assert "Linux" in results[0].reason
    assert "Free" in results[0].reason
    assert "GPL-3.0" in results[0].reason


def test_seed_catalogue_open_source_only(tmp_path):
    cat = SoftwareCatalog(tmp_path)
    results = cat.alternatives_for("adobe-photoshop", open_source=True)
    assert results
    assert all(r.open_source for r in results)
    assert "GIMP" in [r.name for r in results]
    assert "Affinity" not in [r.name for r in results]


def test_seed_catalogue_devon_first(tmp_path):
    # Devon herself is an alternative to Photoshop: she generates and
    # edits images. She goes first, labelled as herself — the porting
    # doctrine (the reference is the floor) plus the owner's rule that
    # she has every skill herself.
    cat = SoftwareCatalog(tmp_path)
    results = cat.alternatives_for("adobe-photoshop")
    assert results[0].is_devon
    assert results[0].name == "Devon image studio"
    assert "herself" in results[0].reason


def test_seed_catalogue_word_has_devon_and_libreoffice(tmp_path):
    cat = SoftwareCatalog(tmp_path)
    results = cat.alternatives_for("microsoft-office")
    names = [r.name for r in results]
    assert names[0] == "Devon documents"
    assert "LibreOffice" in names
    libre = _by_name(results, "LibreOffice")
    assert libre.open_source is True
    assert "Linux" in libre.platforms


def test_seed_catalogue_unknown_software_is_empty(tmp_path):
    cat = SoftwareCatalog(tmp_path)
    assert cat.alternatives_for("definitely-not-a-real-app-xyz") == []


def test_catalog_persists_and_grows(tmp_path):
    cat = SoftwareCatalog(tmp_path)
    assert (tmp_path / "software_catalog.json").exists()
    cat.upsert(SoftwareRecord(
        name="TestApp", slug="testapp", license_tier="Free",
        open_source=True, platforms=["Linux"]))
    cat.add_edge("adobe-photoshop", "testapp", "same task: image editing")
    again = SoftwareCatalog(tmp_path)
    names = [r.name for r in again.alternatives_for(
        "adobe-photoshop", limit=20)]
    assert "TestApp" in names
    stored = json.loads((tmp_path / "software_catalog.json").read_text())
    assert "testapp" in stored["records"]


# ── the source: live fetch merges into the catalogue ───────────────────────

def test_source_alternatives_live_fetch(tmp_path):
    def fake_fetch(url):
        assert url == "https://alternativeto.net/software/adobe-photoshop/"
        return ALT_PHOTOSHOP_HTML

    src = AlternativeToSource(catalog=SoftwareCatalog(tmp_path),
                              fetcher=fake_fetch)
    results = src.alternatives("Photoshop")
    names = [r.name for r in results]
    assert "GIMP" in names and "Krita" in names
    assert results[0].is_devon  # Devon first, live crowd rank after her
    gimp = _by_name(results, "GIMP")
    assert gimp.likes == 4888
    assert gimp.source == "alternativeto"
    # The live fetch grew the durable catalogue.
    again = SoftwareCatalog(tmp_path)
    assert "Krita" in [r.name for r in
                        again.alternatives_for("adobe-photoshop")]


def test_source_search_never_raises(tmp_path):
    def boom(url):
        raise ConnectionError("network down")

    src = AlternativeToSource(catalog=SoftwareCatalog(tmp_path),
                              fetcher=boom)
    assert src.search("photoshop") == []
    assert src.alternatives("Photoshop") != []  # seeds still answer


def test_source_urls():
    src = AlternativeToSource()
    assert src.search_url("Photoshop") == (
        "https://alternativeto.net/browse/search/?q=Photoshop")
    assert src.alternatives_url("adobe-photoshop") == (
        "https://alternativeto.net/software/adobe-photoshop/")


# ── reply rendering ────────────────────────────────────────────────────────

def test_format_alternatives_reply(tmp_path):
    cat = SoftwareCatalog(tmp_path)
    results = cat.alternatives_for(
        "adobe-photoshop", platform="linux", license_tier="free", limit=2)
    text = format_alternatives_reply("Adobe Photoshop", results)
    assert "Alternatives to Adobe Photoshop" in text
    assert "1. GIMP" in text
    assert "4,888 likes" in text
    assert "alternativeto.net/software/gimp/" in text
    assert "AlternativeTo" in text


def test_format_alternatives_reply_empty():
    text = format_alternatives_reply("NoSuchApp", [])
    assert "NoSuchApp" in text
    assert "no alternatives" in text.lower()


# ── registration: source family + brain tool + chat command ────────────────

def test_registry_collects_software_family():
    from nomorals.agency.sources_registry import collect, spec_by_key

    specs, errors = collect()
    assert errors == []
    cats = {s.category for s in specs}
    assert "software" in cats
    spec = spec_by_key(specs, "software:alternativeto")
    assert spec is not None
    assert spec.url.startswith("https://alternativeto.net/browse/search/")
    assert spec.kind == "html-alternatives"


def test_brain_tool_registered():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "software_alternatives" in reg._tools
    outcome = reg.call("software_alternatives", actor="test",
                       software="Adobe Photoshop",
                       license_tier="free", limit=3)
    assert outcome.ok, outcome.error
    result = outcome.value
    assert result["ok"] is True
    names = [r["name"] for r in result["results"]]
    assert "GIMP" in names
    assert result["results"][0]["is_devon"] is True


def test_chat_command_registered():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
        list_catalog,
    )

    assert "alternatives" in CONTROL_COMMANDS
    assert CONTROL_COMMANDS["alternatives"] == (1, None)
    assert "alternatives" in PUBLIC_COMMANDS
    assert len(PUBLIC_COMMANDS) == 275
    assert "alternatives" in COMMAND_DETAILS
    assert "alternatives" in LIST_ONELINERS
    assert "alternatives" in [c for _, kinds in LIST_GROUPS
                              for c in kinds]
    catalog = list_catalog()
    assert "alternatives" in catalog
