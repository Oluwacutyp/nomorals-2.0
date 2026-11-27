"""Lane 116 — Craft-I18n port (census §3A.23).

The failing-first spec for the craft-i18n runtime port.  Every
fixture row below is a real upstream row (checked against the
clone at ~/workspace/echelon-mining/repos/storytold/craft-i18n)
or an upstream test vector from its checker crate; nothing here
is invented.  The API under test is Devon's real one —
``nomorals.i18n.catalog.Entry/Catalog/parse_catalog`` and
``nomorals.i18n.runtime.Translator`` — extended in place, never
replaced: the upstream three-column format loads into the same
``Catalog``/``Entry`` shapes, and the upstream lookup methods
(``t``/``tc``/``get``/``tid``/``trn``/…) sit on the same
``Translator`` beside ``translate``/``translate_detailed``.

Run from the tree root (the module reads the vendored catalogs
under ``nomorals/i18n/catalogs/upstream/``).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

from nomorals.i18n import (
    Translator,
    check_upstream_tree,
    dir_from_env,
    load_upstream_tree,
    parse_upstream_catalog,
    upstream_coverage,
)
from nomorals.i18n.bidi import (
    fmt,
    fmt_plain,
    has_bidi_control,
    has_rtl,
    isolate,
    ltr,
    strip_isolates,
)
from nomorals.i18n.fold import Folded, Folder
from nomorals.i18n.plural import plural_categories, plural_category
from nomorals.i18n.scripts import (
    Direction,
    Han,
    Script,
    base_direction,
    direction,
    han,
    region_script,
    scripts,
    usual_script,
)
from nomorals.i18n.upstream import UpstreamCatalogSet

VENDORED = (Path(__file__).resolve().parent.parent
            / "nomorals" / "i18n" / "catalogs" / "upstream")

#: The header upstream's checker prescribes (tree.rs: copy the
#: header from docs/catalog-format.md).  The shipped upstream
#: files predate it — the real-tree test below says so plainly —
#: but every fixture here carries it, so the fixture tree is
#: checker-clean.
_HEADER = ("# Clean room: translations written from the plain "
           "meaning of the English.\n")

#: Real rows from the clone's fr/common.tsv (verified 2026-11-19).
_FR_COMMON = (
    _HEADER
    + "\tOpen\tOuvrir\n"
    + "\tSave\tEnregistrer\n"
    + "\tAdjust\tRégler\n"
    + "theme\tAppearance\tApparence\n"
    + "@plural\t{n} byte|{n} bytes\t{n} octet|{n} octets\n"
    + "@plural\t{n} commit|{n} commits\t{n} commit|{n} commits\n"
    + "@id\tfile.close\tFermer\n"
    + "@id\tlanguage.name\tFrançais\n"
    + "@msg\tCouldn't open {name}: {e}\t"
      "Impossible d\u2019ouvrir {name} : {e}\n"
    + "\tA\\nB\tC\\tD\\\\\n"
)

#: Real rows from the clone's fr/wordcraft.tsv: the Adjust
#: override (common says Régler).
_FR_WORDCRAFT = _HEADER + "\tAdjust\tAjuster\n"

#: Real rows from the clone's de/common.tsv.
_DE_COMMON = (
    _HEADER
    + "\tOpen\tÖffnen\n"
    + "\tSave\tSpeichern\n"
    + "@plural\t{n} commit|{n} commits\t{n} Commit|{n} Commits\n"
    + "@id\tlanguage.name\tDeutsch\n"
)


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "locales" / "fr").mkdir(parents=True)
    (root / "locales" / "de").mkdir(parents=True)
    (root / "locales" / "fr" / "common.tsv").write_text(
        _FR_COMMON, encoding="utf-8")
    (root / "locales" / "fr" / "wordcraft.tsv").write_text(
        _FR_WORDCRAFT, encoding="utf-8")
    (root / "locales" / "de" / "common.tsv").write_text(
        _DE_COMMON, encoding="utf-8")
    return root


def _translator(tmp_path: Path) -> Translator:
    return Translator.from_upstream(_tree(tmp_path))


def _bad_tree(tmp_path: Path) -> Path:
    """A tree with one bad file per checker kind (and no Clean
    room header anywhere, so the header kind fires too)."""
    root = tmp_path / "bad"
    fr = root / "locales" / "fr"
    fr.mkdir(parents=True)
    (fr / "common.tsv").write_text(
        "\tOpen\tOuvrir\n"
        "\tSave\tEnregistrer\n"
        "@id\tlanguage.name\tFrançais\n"
        "\tSave\t\n"
        "\tSave {file}\tEnregistrer\n"
        "@plural\t{n} byte|{n} bytes\t{n} octet\n"
        "Dup\tSame\tUn\n"
        "Dup\tSame\tDeux\n"
        "@msg\t{x}\t{x}\n",
        encoding="utf-8")
    (fr / "wordcraft.tsv").write_text(
        "\tSave\tEnregistrer\n", encoding="utf-8")
    (fr / "bad name.tsv").write_text(
        "\tOpen\tOuvrir\n", encoding="utf-8")
    fr_ca = root / "locales" / "fr-CA"
    fr_ca.mkdir(parents=True)
    (fr_ca / "common.tsv").write_text(
        "\tOpen\tOuvrir\n"
        "@id\tlanguage.name\tFrançais (Canada)\n",
        encoding="utf-8")
    de = root / "locales" / "de"
    de.mkdir(parents=True)
    (de / "common.tsv").write_text(
        "\tOpen\tÖffnen\n", encoding="utf-8")
    # A valid regional pair: pt-br's Open row is identical to
    # pt's, so removing it changes nothing (regional_noop).
    pt = root / "locales" / "pt"
    pt.mkdir(parents=True)
    (pt / "common.tsv").write_text(
        "\tOpen\tAbrir\n"
        "@id\tlanguage.name\tPortuguês\n", encoding="utf-8")
    pt_br = root / "locales" / "pt-br"
    pt_br.mkdir(parents=True)
    (pt_br / "common.tsv").write_text(
        "\tOpen\tAbrir\n"
        "@id\tlanguage.name\tPortuguês (Brasil)\n",
        encoding="utf-8")
    xx = root / "locales" / "XX"
    xx.mkdir(parents=True)
    (xx / "common.tsv").write_text(
        "\tOpen\tOpen\n", encoding="utf-8")
    return root


# ── plural table: the counts whole numbers 0..1000 reach ───────

def test_plural_category_sets_corrected():
    # Russian/Polish/Ukrainian/Belarusian: three forms, not four —
    # "other" is a decimals-only category there.
    assert plural_categories("ru") == ("one", "few", "many")
    assert plural_categories("uk") == ("one", "few", "many")
    assert plural_categories("be") == ("one", "few", "many")
    assert plural_categories("pl") == ("one", "few", "many")
    assert plural_categories("tr") == ("one", "other")
    assert plural_categories("sr") == ("one", "few", "other")
    assert plural_categories("hr") == ("one", "few", "other")
    assert plural_categories("bs") == ("one", "few", "other")
    assert plural_categories("fa") == ("one", "other")
    assert plural_categories("ja") == ("other",)


def test_plural_category_region_aware_portuguese():
    # CLDR's pt is Brazilian (0 is singular); any other pt takes
    # pt-PT's rule (1 only).
    assert plural_category("pt-BR", 0) == "one"
    assert plural_category("pt-BR", 2) == "other"
    assert plural_category("pt-PT", 0) == "other"
    assert plural_category("pt-PT", 1) == "one"
    assert plural_category("pt", 0) == "other"
    assert plural_category("pt", 1) == "one"


def test_plural_category_serbian():
    assert plural_category("sr", 1) == "one"
    assert plural_category("sr", 3) == "few"
    assert plural_category("sr", 5) == "other"
    assert plural_category("sr", 21) == "one"
    assert plural_category("sr", 22) == "few"


# ── scripts (craft-libs scripts.rs) ────────────────────────────

def test_scripts_tables():
    assert scripts("ja") == (Script.HANI, Script.HIRA, Script.KANA)
    assert scripts("ko") == (Script.HANG, Script.HANI)
    assert scripts("sr") == (Script.CYRL,)
    assert scripts("sr-latn") == (Script.LATN,)
    assert scripts("sr_RS.UTF-8@latin") == (Script.LATN,)
    assert scripts("zh-TW") == (Script.HANI,)
    assert scripts("pa-PK") == (Script.ARAB,)
    assert scripts("ar") == (Script.ARAB,)
    assert scripts("he") == (Script.HEBR,)
    assert scripts("xx") == (Script.LATN,)
    assert Script.ARAB.rtl and Script.HEBR.rtl
    assert not Script.LATN.rtl
    assert Script.LATN.code == "Latn"


def test_direction_and_region_script():
    assert direction("ar") == Direction.RTL
    assert direction("he") == Direction.RTL
    assert direction("fr") == Direction.LTR
    assert direction("sr-latn") == Direction.LTR
    assert region_script("pa", "pk") == "arab"
    assert region_script("zh", "tw") == "hant"
    assert region_script("pt", "br") is None
    assert usual_script("zh") == "hans"
    assert usual_script("sr") == "cyrl"
    assert usual_script("en") is None


def test_han_forms():
    assert han("zh-CN") == Han.SIMPLIFIED
    assert han("zh-Hans") == Han.SIMPLIFIED
    assert han("zh") == Han.SIMPLIFIED
    assert han("zh-TW") == Han.TRADITIONAL
    assert han("zh-Hant") == Han.TRADITIONAL
    assert han("zh-HK") == Han.HONG_KONG
    assert han("zh-Hant-HK") == Han.HONG_KONG
    assert han("yue") == Han.HONG_KONG
    assert han("yue-Hant-TW") == Han.TRADITIONAL
    assert han("ja") == Han.JAPANESE
    assert han("ko") == Han.KOREAN
    assert han("fr") is None


def test_base_direction():
    assert base_direction("Hello") == Direction.LTR
    assert base_direction("مرحبا") == Direction.RTL
    assert base_direction("123") is None
    assert base_direction("") is None
    # Isolate contents are skipped (UAX #9 P2/P3).
    assert base_direction("\u2067abc") is None
    assert base_direction("\u2067a\u2069b") == Direction.LTR
    # The search stops at the first paragraph separator.
    assert base_direction("Hello\u2029مرحبا") == Direction.LTR


# ── bidi-safe filling (craft-libs lib.rs fmt/isolate/ltr) ─────

def test_bidi_isolate_and_ltr():
    assert isolate("abc") == "\u2068abc\u2069"
    assert isolate("a\u2069b") == "\u2068ab\u2069"
    assert isolate("a\u2067b") == "\u2068a\u2067b\u2069\u2069"
    assert isolate("\u2067a\u202eb\u2069c") == \
        "\u2068\u2067a\u202eb\u202c\u2069c\u2069"
    assert ltr("/x/y") == "\u2066/x/y\u2069"
    assert strip_isolates("\u2068abc\u2069") == "abc"
    assert strip_isolates("a\u2066b\u2069c") == "abc"


def test_bidi_fmt_isolates_only_when_needed():
    # No RTL anywhere: no marks at all.
    assert fmt("Open {name}", [("name", "file.txt")]) == \
        "Open file.txt"
    # An RTL template isolates each value.
    assert fmt("افتح {name}", [("name", "file.txt")]) == \
        "افتح \u2068file.txt\u2069"
    # An RTL value isolates itself in an LTR template.
    assert fmt("Open {name}", [("name", "ملف")]) == \
        "Open \u2068ملف\u2069"
    # A value that already is one isolate keeps it when the text
    # needs isolates — and has it taken off when it does not.
    assert fmt("افتح {name}", [("name", ltr("1.5"))]) == \
        "افتح \u20661.5\u2069"
    assert fmt("Open {name}", [("name", ltr("1.5"))]) == "Open 1.5"
    # Unknown placeholders stay; values are never templates.
    assert fmt("{a} {b}", [("a", "x")]) == "x {b}"
    assert fmt("Open {name}", [("name", "{n}")]) == "Open {n}"
    assert fmt_plain("افتح {name}", [("name", "file.txt")]) == \
        "افتح file.txt"
    assert fmt_plain("Open {name}", [("name", "ملف")]) == "Open ملف"


def test_bidi_controls_and_rtl():
    assert has_rtl("مرحبا")
    assert not has_rtl("hello")
    assert not has_rtl("123")
    assert has_bidi_control("\u202eabc")
    # Upstream unwraps exactly one isolate before scanning
    # (lib.rs has_bidi_control): a clean isolate carries no
    # control; the control must sit inside the wrapped text.
    assert not has_bidi_control(ltr("abc"))
    assert has_bidi_control(ltr("\u202eabc"))
    assert not has_bidi_control("abc")
    assert not has_bidi_control("\u2068abc\u2069")


# ── folding (craft-libs fold.rs) ───────────────────────────────

def test_fold_keys():
    assert Folder("en").key("Straße") == "strasse"
    assert Folder("en").key("ﬁsh") == "fish"
    assert Folder("en").key("ＡＢＣ") == "abc"
    assert Folder("en").key("１２３") == "123"
    assert Folder("en").key("١٢٣") == "123"
    assert Folder("en").key("É") == "e"
    assert Folder("en").key("Œuvre") == "oeuvre"
    assert Folder("en").key("ΟΣ") == "οσ"
    assert Folder("en").key("ΟΣΟΣ") == "οσοσ"
    assert Folder("en").key("ك") == "ك"
    assert Folder("en").key("ک") == "ك"
    assert Folder("en").key("ی") == "ي"
    assert Folder("en").key("ة") == "ه"
    assert Folder("en").key("क") == "क"
    assert Folder("en").key("soft\u00adware") == "software"
    assert Folder("en").key("a\u200bb") == "ab"
    assert Folder("ja").key("ＡＢＣ１２３　ｶﾞ") == \
        unicodedata.normalize("NFD", "abc123 ガ")


def test_fold_turkish_i():
    # Ignoring accents, all four i's fold to "i" — even in
    # Turkish.  The dotless-i mapping only applies when accents
    # count (upstream fold.rs: the turkic arm needs accents).
    assert Folder("tr").key("I") == "i"
    assert Folder("tr").keep_accents().key("I") == "ı"
    assert Folder("tr").key("İ") == "i"
    assert Folder("en").key("ı") == "i"
    assert Folder("en").key("I") == "i"
    assert Folder("tr").key("İSTANBUL") == "istanbul"
    assert Folder("tr").key("IRMAK") == "irmak"
    assert Folder("tr").key("ırmak") == "irmak"


def test_fold_keep_accents():
    assert Folder("en").keep_accents().key("É") == \
        unicodedata.normalize("NFD", "é")
    assert Folder("en").key("É") == "e"


def test_folded_find_maps_back():
    folded = Folder("en").fold("Straße")
    assert folded.find(Folder("en").fold("strasse")) == slice(0, 6)
    assert Folder("en").fold("ﬁsh").find(
        Folder("en").fold("fi")) == slice(0, 1)
    assert Folder("en").fold("aéb").find(
        Folder("en").fold("aeb")) == slice(0, 3)
    assert Folder("en").fold("Café").find(
        Folder("en").fold("cafe")) == slice(0, 4)
    assert Folder("en").fold("hello").find(
        Folder("en").fold("x")) is None
    assert Folder("en").fold("hello").find(
        Folder("en").fold("")) == slice(0, 0)
    # Accents counting: "re" must not match "ré".
    folded = Folder("en").keep_accents().fold("ré")
    assert folded.find(
        Folder("en").keep_accents().fold("re")) is None
    # A Hangul syllable and its jamo spell the same thing.
    folded = Folder("en").fold("각")
    assert folded.find(Folder("en").fold("각")) == slice(0, 3)
    assert Folder("en").fold("가각").find(
        Folder("en").fold("각")) == slice(1, 2)
    assert Folder("en").contains("Große Straße", "strasse")
    assert not Folder("en").contains("hello", "x")
    assert isinstance(Folder("en").fold("abc"), Folded)

# ── upstream catalog parsing (craft-i18n-tests catalog.rs) ───

def test_parse_upstream_catalog_good_rows():
    cat = parse_upstream_catalog(_FR_COMMON, "fr")
    assert cat.issues == []
    assert cat.get("Open").text == "Ouvrir"
    assert cat.get("Save").text == "Enregistrer"
    assert cat.get("Adjust").text == "Régler"
    assert cat.get("Appearance", context="theme").text == "Apparence"
    assert cat.get("Appearance") is None  # context rows need context
    assert cat.get("file.close", context="@id").text == "Fermer"
    assert cat.get("language.name", context="@id").text == "Français"
    entry = cat.get("{n} byte|{n} bytes", context="@plural")
    assert entry is not None and entry.forms == (
        "{n} octet", "{n} octets")
    assert entry.is_plural
    # The escapes row: source "A\nB", translation "C\tD\".
    esc = cat.get("A\nB")
    assert esc is not None and esc.text == "C\tD\\"
    # The @msg row parses as a template.
    msg = cat.get("Couldn't open {name}: {e}", context="@msg")
    assert msg is not None
    assert "{name}" in msg.text and "{e}" in msg.text


@pytest.mark.parametrize("text,line", [
    ("Only\tTwo\n", 1),                       # 2 columns, not 3
    ("\tNope\t\n", 1),                        # empty translation
    ("\t\tVide\n", 1),                        # empty source
    ("Dup\tSame\tUn\nDup\tSame\tDeux\n", 2),  # duplicate key
    ("@plural\t{n} byte|{n} bytes\t{n} octet\n", 1),  # 1 of 2 forms
    ("@plural\tpage\tpage|pages\n", 1),       # not one|other
    ("@nope\tA\tB\n", 1),                     # unknown @context
    ("\tSave {file}\tEnregistrer\n", 1),      # placeholder lost
    ("\tOpen…\tOuvrir\n", 1),                 # trailing … lost
    ("\tA\\qB\tA\n", 1),                      # unknown escape
    ("\tA\\\tA\n", 1),                        # trailing backslash
    ("@msg\t{x}\t{x}\n", 1),                  # no literal text
    ("@msg\tOpen {a}{b}\tOuvrir {a}{b}\n", 1),  # two in a row
    ("@msg\tOpen {a\tOuvrir {a\n", 1),        # lone {
    ("\tSave\tEn\rreg\n", 1),                 # control char in row
    ("\tA\tB\n\ufeff\tC\tD\n", 2),            # BOM past line 1
])
def test_parse_upstream_catalog_bad_row_is_an_issue(text, line):
    cat = parse_upstream_catalog(text, "fr")
    assert len(cat.issues) == 1, cat.issues
    issue = cat.issues[0]
    assert issue.line == line
    assert issue.locale == "fr"


def test_parse_upstream_catalog_bad_rows_do_not_lose_good_ones():
    cat = parse_upstream_catalog(
        "\tOpen\tOuvrir\n\tSave\t\n\tUndo\tAnnuler\n", "fr")
    assert len(cat.issues) == 1
    assert cat.get("Open").text == "Ouvrir"
    assert cat.get("Undo").text == "Annuler"
    assert cat.get("Save") is None


def test_parse_upstream_catalog_one_form_locale():
    # Japanese lists one plural form; one form is enough there.
    cat = parse_upstream_catalog(
        "@plural\t{n} a|{n} b\t{n} x\n", "ja")
    assert cat.issues == []
    entry = cat.get("{n} a|{n} b", context="@plural")
    assert entry is not None and entry.forms == ("{n} x",)


def test_parse_upstream_catalog_crlf_and_bom():
    cat = parse_upstream_catalog(
        "\ufeff\tSave\tEnregistrer\r\n\tUndo\tAnnuler\r\n", "fr")
    assert cat.issues == []
    assert cat.get("Save").text == "Enregistrer"
    assert cat.get("Undo").text == "Annuler"


def test_parse_upstream_catalog_placeholders_may_reorder():
    cat = parse_upstream_catalog("\t{a} to {b}\t{b} vers {a}\n", "fr")
    assert cat.issues == []


# ── upstream tree loading ─────────────────────────────────────

def test_upstream_tree_loads(tmp_path):
    root = _tree(tmp_path)
    loaded = load_upstream_tree(root)
    assert sorted(loaded) == ["de", "fr"]
    assert loaded["fr"].get("Open").text == "Ouvrir"
    assert loaded["fr"].get("Adjust").text == "Régler"
    assert loaded["fr"].get("Adjust", app="wordcraft").text == "Ajuster"
    assert loaded["fr"].get("Adjust", app="pdfcraft") is None
    assert loaded["de"].get("Open").text == "Öffnen"
    assert loaded["de"].get("language.name",
                            context="@id").text == "Deutsch"


def test_dir_from_env(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setenv("CRAFT_I18N_DIR", str(root))
    assert dir_from_env() == root
    monkeypatch.delenv("CRAFT_I18N_DIR")
    assert dir_from_env() is None


def test_upstream_set_from_text_and_dir(tmp_path):
    texts = {"fr": _FR_COMMON, "de": _DE_COMMON}
    from_text = UpstreamCatalogSet.from_text(texts)
    assert from_text.locales() == ["de", "fr"]
    assert from_text.get("Open", locale="fr") == "Ouvrir"
    from_dir = UpstreamCatalogSet.from_dir(_tree(tmp_path))
    assert from_dir.locales() == ["de", "fr"]
    assert from_dir.get("Adjust", app="wordcraft",
                        locale="fr") == "Ajuster"


def test_upstream_set_language_name(tmp_path):
    ups = UpstreamCatalogSet.from_dir(_tree(tmp_path))
    assert ups.language_name("fr") == "Français"
    assert ups.language_name("de") == "Deutsch"
    assert ups.language_name("xx") is None


# ── the upstream runtime (craft-libs lib.rs I18n) ─────────────

def test_t_tc_get_tid(tmp_path):
    tr = _translator(tmp_path)
    assert tr.t("Open", locale="fr") == "Ouvrir"
    assert tr.t("Missing", locale="fr") == "Missing"
    assert tr.tc("Appearance", "theme", locale="fr") == "Apparence"
    assert tr.tc("Appearance", "other", locale="fr") == "Appearance"
    # The app layer beats the shared row; without it, shared wins.
    assert tr.tc("Adjust", "", locale="fr",
                 app="wordcraft") == "Ajuster"
    assert tr.tc("Adjust", "", locale="fr") == "Régler"
    # get() asks one (context, source) pair only: no plain
    # fallback, no source fallback.
    assert tr.get("Open", locale="fr") == "Ouvrir"
    assert tr.get("Open", context="menu", locale="fr") is None
    assert tr.get("Missing", locale="fr") is None
    # tid(): the @id row, then the plain row of the label, then
    # the label itself.
    assert tr.tid("file.close", "Close", locale="fr") == "Fermer"
    assert tr.tid("missing.id", "Do it", locale="fr") == "Do it"


def test_trn_plural_forms(tmp_path):
    tr = _translator(tmp_path)
    assert tr.trn("{n} byte|{n} bytes", "fr", 2) == "2 octets"
    assert tr.trn("{n} byte|{n} bytes", "fr", 1) == "1 octet"
    assert tr.trn("{n} byte|{n} bytes", "fr", 0) == "0 octet"
    assert tr.trn("{n} commit|{n} commits", "de", 2) == "2 Commits"
    assert tr.trn("{n} commit|{n} commits", "de", 1) == "1 Commit"
    assert tr.trn_plain("{n} byte|{n} bytes", "fr", 2) == "2 octets"
    # No row anywhere: the English forms serve.
    assert tr.trn("{n} file|{n} files", "fr", 2) == "2 files"
    assert tr.trn("{n} file|{n} files", "fr", 1) == "1 file"


def test_trn_decimal_formats_count(tmp_path):
    tr = _translator(tmp_path)
    assert tr.trn_decimal("{n} byte|{n} bytes", "fr", 1.5, 1) == \
        "1.5 octets"
    from nomorals.craftlibs.i18n import Format
    tr.set_format(Format.from_tag("fr"))
    assert tr.trn_decimal("{n} byte|{n} bytes", "fr", 1.5, 1) == \
        "1,5 octets"
    tr.reset_format()
    assert tr.trn_decimal("{n} byte|{n} bytes", "fr", 1.5, 1) == \
        "1.5 octets"


def test_upstream_chain_rules(tmp_path):
    tr = _translator(tmp_path)
    # Only a region falls back, and only when the parent locale
    # is loaded; a script never crosses.  Codes are lowercase
    # BCP 47, as upstream's code() writes them.
    assert tr.chain("pt-BR") == ["pt-br", "en"]
    assert tr.chain("sr-Latn") == ["sr-latn", "en"]
    assert tr.chain("zh-Hant-HK") == ["zh-hant-hk", "en"]
    assert tr.chain("fr") == ["fr", "en"]
    assert tr.chain("fr-CA") == ["fr-ca", "fr", "en"]
    assert tr.chain("xx") == ["xx", "en"]


def test_code_from_code_languages_name_rtl(tmp_path):
    tr = _translator(tmp_path)
    assert tr.code("fr") == "fr"
    assert tr.code("xx") == "en"
    assert tr.from_code("fr") == "fr"
    assert tr.from_code("FR") == "fr"
    assert tr.from_code("xx") is None
    assert tr.from_code("pseudo") == "pseudo"
    assert tr.languages() == ["en", "de", "fr"]
    assert tr.name("fr") == "Français"
    assert tr.name("de") == "Deutsch"
    assert tr.name("en") == "English"
    assert tr.rtl("ar") is True
    assert tr.rtl("he") is True
    assert tr.rtl("fr") is False


def test_match_tag_and_tags(tmp_path):
    tr = _translator(tmp_path)
    assert tr.match_tag("fr_CA.UTF-8") == "fr"
    assert tr.match_tag("de-DE") == "de"
    assert tr.match_tag("C") == "en"
    assert tr.match_tag("POSIX") == "en"
    assert tr.match_tag("xx") is None
    assert tr.match_tag("pt-BR") is None  # pt is not loaded here
    assert tr.match_tags(["xx", "de-DE"]) == "de"
    assert tr.match_tags(["xx", "yy"]) == "en"


def test_system_lang(tmp_path, monkeypatch):
    tr = _translator(tmp_path)
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert tr.system_lang() == "de"
    monkeypatch.setenv("LANG", "C")
    assert tr.system_lang() == "en"
    monkeypatch.setenv("LANGUAGE", "fr:de")
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert tr.system_lang() == "fr"


def test_pseudo_locale_marks_and_records(tmp_path):
    tr = _translator(tmp_path)
    assert tr.t("Open", locale="pseudo") == "⟦Open⟧"
    assert tr.t("Missing", locale="pseudo") == "⟦Missing⟧"
    assert tr.t("Line1\nLine2", locale="pseudo") == \
        "⟦Line1⟧\n⟦Line2⟧"
    assert tr.looked_up() == {"Open", "Missing", "Line1\nLine2"}
    # Plurals are marked too, but not recorded.
    assert tr.trn("{n} byte|{n} bytes", "pseudo", 2) == \
        "⟦2 bytes⟧"
    assert "bytes" not in " ".join(tr.looked_up())


def test_translate_serves_upstream_rows(tmp_path):
    tr = _translator(tmp_path)
    assert tr.translate("Open", "fr") == "Ouvrir"
    # count fills both {count} (Devon) and {n} (upstream).
    assert tr.translate("{n} byte|{n} bytes", "fr", count=2) == \
        "2 octets"
    detailed = tr.translate_detailed("Open", "fr")
    assert detailed.served_locale == "fr"
    assert detailed.source == "catalog"
    # A missing key still falls back to the source string.
    assert tr.translate("Nobody catalogued this.", "fr") == \
        "Nobody catalogued this."


def test_check_all_clean_and_bad(tmp_path):
    tr = _translator(tmp_path)
    assert tr.check_all() == []
    bad = Translator.from_upstream(_bad_tree(tmp_path))
    problems = bad.check_all()
    assert problems
    assert any("fr/common.tsv" in p for p in problems)

# ── the checker crate (craft-i18n-tests tree.rs) ─────────────

def test_check_upstream_tree_clean(tmp_path):
    assert check_upstream_tree(_tree(tmp_path)) == []


def test_check_upstream_tree_no_locales_dir(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert check_upstream_tree(root) == []


def test_check_upstream_tree_bad_tree(tmp_path):
    issues = check_upstream_tree(_bad_tree(tmp_path))
    kinds = {i["kind"] for i in issues}
    for kind in ("empty_translation", "placeholder_mismatch",
                 "plural_form_count", "duplicate_key", "header",
                 "regional_noop", "app_override_identical",
                 "missing_language_name", "file_naming",
                 "bad_locale_name", "bad_template"):
        assert kind in kinds, f"missing kind {kind}: {issues}"
    for issue in issues:
        assert issue["path"].startswith("locales/")
        assert issue["problem"]


def test_check_upstream_tree_real_tree_is_clean():
    # The shipped files carry the clean-room header and pass
    # every other rule too: all 102 files, no findings.  (An
    # earlier draft of this test assumed the shipped files
    # predated the header rule and carried redundant sr-latn
    # rows; both were wrong — every vendored file has a
    # "# Clean room:" line, and sr-latn holds 1148 rows against
    # sr's 421, so almost nothing in it is redundant.)
    assert check_upstream_tree(VENDORED) == []


def test_upstream_coverage_fixture_tree(tmp_path):
    lines = upstream_coverage(_tree(tmp_path))
    assert set(lines) == {
        "de/common.tsv: 4/10",
        "fr/common.tsv: 10/10",
        "fr/wordcraft.tsv: 1/1",
    }


def test_upstream_coverage_real_tree():
    lines = upstream_coverage(VENDORED)
    assert len(lines) == 102
    assert all(re.match(r"^[^/]+/[^:]+: \d+/\d+$", line)
               for line in lines)
    assert any(line.startswith("fr/common.tsv: 1290/")
               for line in lines)


# ── the craftlibs tool's i18n action, extended in place ──────

def _registry():
    from types import SimpleNamespace

    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry


def _call(registry, name, **kwargs):
    outcome = registry.call(name, **kwargs)
    if hasattr(outcome, "value"):
        return outcome.value
    return outcome


def test_tool_i18n_translate():
    registry = _registry()
    out = _call(registry, "craftlibs", action="i18n",
                op="translate", key="Open", locale="fr")
    assert out["text"] == "Ouvrir"
    assert out["locale"] == "fr"


def test_tool_i18n_context_and_app():
    registry = _registry()
    out = _call(registry, "craftlibs", action="i18n",
                op="translate", key="Adjust", locale="fr",
                app="wordcraft")
    assert out["text"] == "Ajuster"
    out = _call(registry, "craftlibs", action="i18n",
                op="translate", key="Adjust", locale="fr")
    assert out["text"] == "Régler"
    out = _call(registry, "craftlibs", action="i18n",
                op="translate", key="Appearance", locale="fr",
                context="theme")
    assert out["text"] == "Apparence"


def test_tool_i18n_plural():
    registry = _registry()
    out = _call(registry, "craftlibs", action="i18n",
                op="plural", source="{n} byte|{n} bytes",
                locale="fr", count=2)
    assert out["text"] == "2 octets"
    out = _call(registry, "craftlibs", action="i18n",
                op="plural", source="{n} byte|{n} bytes",
                locale="fr", count=1)
    assert out["text"] == "1 octet"


def test_tool_i18n_check_and_languages():
    registry = _registry()
    out = _call(registry, "craftlibs", action="i18n", op="check")
    assert out["files"] == 102
    # The vendored tree passes its own checker: no findings.
    assert out["issues"] == 0
    assert out["kinds"] == []
    out = _call(registry, "craftlibs", action="i18n",
                op="languages")
    assert "fr" in out["languages"]
    assert "de" in out["languages"]
    assert out["names"]["fr"] == "Français"
    assert out["names"]["en"] == "English"


def test_tool_i18n_old_ops_still_work():
    registry = _registry()
    out = _call(registry, "craftlibs", action="i18n",
                format=1234.5, locale="fr", decimals=1)
    assert out["formatted"] == "1\u202f234,5"
    out = _call(registry, "craftlibs", action="i18n",
                parse="1 234,5", locale="fr")
    assert out["value"] == 1234.5
    out = _call(registry, "craftlibs", action="i18n",
                file_size=1536, locale="en")
    assert out["size"] == "1.5 KB"


# ── parity (craftlibs parity runner + the i18n TOML) ──────────

def _tool_caller():
    """A tool caller for the parity ``tool_call`` op: the
    registry lives a layer up from the organ, so the test (not
    the organ) builds it and injects it."""
    registry = _registry()

    def call(args):
        return _call(registry, "craftlibs", **args)

    return call


def test_parity_run_case_ops(tmp_path):
    from nomorals.i18n.parity import run_case

    tr = _translator(tmp_path)
    assert run_case({"op": "lookup", "locale": "fr",
                     "source": "Open", "expect": "Ouvrir"}, tr)
    assert run_case({"op": "fallback", "locale": "fr-CA",
                     "source": "Open", "expect": "Ouvrir"}, tr)
    assert run_case({"op": "plural", "locale": "fr",
                     "source": "{n} byte|{n} bytes", "count": 2,
                     "expect": "2 octets"}, tr)
    assert run_case({"op": "plural_fallback", "locale": "fr-CA",
                     "source": "{n} byte|{n} bytes", "count": 1,
                     "expect": "1 octet"}, tr)
    assert run_case({"op": "placeholder_mismatch",
                     "locale": "fr",
                     "source": "Couldn't open {name}: {e}"}, tr)
    assert run_case({"op": "checker_finding",
                     "locale": "fr",
                     "text": "\tSave\t\n",
                     "kind": "empty_translation"}, tr)
    assert run_case({"op": "tool_call",
                     "args": {"action": "i18n", "op": "translate",
                              "key": "Open", "locale": "fr"},
                     "expect_contains": "Ouvrir"}, tr,
                    tool_caller=_tool_caller())
    with pytest.raises(AssertionError):
        run_case({"op": "lookup", "locale": "fr",
                  "source": "Open", "expect": "Wrong"}, tr)
    # A tool_call case without an injected caller fails naming
    # that, never silently.
    with pytest.raises(AssertionError, match="tool_caller"):
        run_case({"op": "tool_call",
                  "args": {"action": "i18n", "op": "translate",
                           "key": "Open", "locale": "fr"},
                  "expect_contains": "Ouvrir"}, tr)


def test_parity_toml_runs():
    from nomorals.craftlibs.parity import run_parity
    from nomorals.i18n.parity import run_case

    toml_path = (Path(__file__).resolve().parent.parent
                 / "nomorals" / "i18n" / "crafti18n-parity.toml")
    assert toml_path.exists()
    caller = _tool_caller()
    result = run_parity(
        toml_path,
        run_case=lambda case: run_case(case,
                                       tool_caller=caller))
    assert result["failed"] == [], result["failures"]
    assert result["total"] > 0
    assert result["passed"] == result["total"]


def test_parity_organ_runner_runs():
    # The organ's own entry point runs the same TOML with the
    # same result shape (its loop lives here because L3 cannot
    # import the L4 shared runner).
    from nomorals.i18n.parity import run_i18n_parity

    result = run_i18n_parity(tool_caller=_tool_caller())
    assert result["failed"] == [], result["failures"]
    assert result["total"] == 7
    assert result["passed"] == 7
    missing = run_i18n_parity("/nonexistent/crafti18n.toml")
    assert missing["failed"] == ["<toml>"]


# ── attribution (craftlibs attribution runner) ────────────────

def test_attribution_verifies():
    from nomorals.craftlibs.attribution import verify_attribution

    toml_path = (Path(__file__).resolve().parent.parent
                 / "nomorals" / "i18n" / "crafti18n-attribution.toml")
    assert toml_path.exists()
    result = verify_attribution(toml_path)
    assert result["failed"] == [], result["failed"]
    assert result["skipped"] == [], result["skipped"]
    assert result["verified"] > 100  # code ports + 102 catalogs


# ── the Devon path is unchanged ───────────────────────────────

def test_devon_runtime_still_green():
    from nomorals.i18n import load_catalogs

    tr = Translator(load_catalogs())
    assert tr.translate("Welcome to **{group}**!", "es",
                        group="Lagos Devs") == \
        "¡Bienvenido a **Lagos Devs**!"
    # Russian grammar through the runtime: 1/2/5/21 forms, now
    # from a three-form row (the fourth form was a duplicate).
    assert tr.translate("{count} member", "ru", count=1) == \
        "1 участник"
    assert tr.translate("{count} member", "ru", count=2) == \
        "2 участника"
    assert tr.translate("{count} member", "ru", count=5) == \
        "5 участников"
    assert tr.translate("{count} member", "ru", count=21) == \
        "21 участник"
    # A decimal count in Russian takes the "few" form (upstream's
    # category → few → last fallback), not a crash.
    assert tr.translate("{count} member", "ru", count=1.5) == \
        "1.5 участника"
    # The shipped catalogs still pass the Devon checker.
    from nomorals.i18n import check_catalogs

    report = check_catalogs()
    assert report.ok, report.issues


def test_tool_count_unchanged():
    # This lane adds no tool: the craftlibs tool's i18n action
    # grows in place.  (Lane 115's count pins at 786; the parent
    # re-bases this pin if lane 115 ships first — rebased here: 789
    # after D5 (walabel) added whatsapp_account_label.)
    registry = _registry()
    assert len(registry._tools) == 811
    assert "craftlibs" in registry._tools
