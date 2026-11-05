"""Echelon Wave 7 — i18n runtime, catalogs, plurals, bridge, surface.

All offline.  Temp catalog dirs live under the pytest temp tree
(disk-backed basetemp in the wave's runs); the shipped catalogs are
exercised read-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nomorals import i18n
from nomorals.i18n import bridge
from nomorals.i18n.catalog import load_catalogs, parse_catalog
from nomorals.i18n.locale import fallback_chain, language_of, normalize_locale
from nomorals.i18n.plural import plural_categories, plural_category
from nomorals.i18n.runtime import I18nError, Translator

WELCOME = "Welcome to **{group}**!"
MEMBER = "{count} member"


# ── locale tags ──────────────────────────────────────────────────

def test_normalize_locale():
    assert normalize_locale("pt_br") == "pt-BR"
    assert normalize_locale("PT-BR") == "pt-BR"
    assert normalize_locale("ZH-hans-cn") == "zh-Hans-CN"
    assert normalize_locale("en") == "en"
    assert normalize_locale("es-419") == "es-419"
    assert normalize_locale("") == ""
    assert normalize_locale(None) == ""
    assert normalize_locale("!!") == ""
    assert language_of("pt-BR") == "pt"


def test_fallback_chain():
    assert fallback_chain("pt-BR") == ["pt-BR", "pt", "en"]
    assert fallback_chain("zh-Hans-CN") == ["zh-Hans-CN", "zh-Hans", "zh", "en"]
    assert fallback_chain("en") == ["en"]
    assert fallback_chain("") == ["en"]
    assert fallback_chain("fr") == ["fr", "en"]


# ── plural rules ─────────────────────────────────────────────────

@pytest.mark.parametrize("locale,count,category", [
    ("en", 1, "one"), ("en", 0, "other"), ("en", 2, "other"),
    ("fr", 0, "one"), ("fr", 1, "one"), ("fr", 2, "other"),
    ("ru", 1, "one"), ("ru", 2, "few"), ("ru", 5, "many"),
    ("ru", 21, "one"), ("ru", 22, "few"), ("ru", 25, "many"),
    ("ru", 11, "many"), ("ru", 1.5, "other"),
    ("ar", 0, "zero"), ("ar", 1, "one"), ("ar", 2, "two"),
    ("ar", 5, "few"), ("ar", 23, "many"), ("ar", 100, "other"),
    ("ja", 1, "other"), ("ja", 100, "other"),
    ("cs", 1, "one"), ("cs", 3, "few"), ("cs", 5, "other"),
    ("pl", 1, "one"), ("pl", 3, "few"), ("pl", 12, "many"),
    ("lv", 0, "zero"), ("lv", 21, "one"), ("lv", 11, "zero"),
    ("lv", 5, "other"),
    ("sl", 101, "one"), ("sl", 102, "two"), ("sl", 103, "few"),
    ("yo", 1, "other"), ("yo", 7, "other"),
    ("pt-BR", 0, "one"), ("pt-BR", 2, "other"),  # region → language rule
])
def test_plural_category(locale, count, category):
    assert plural_category(locale, count) == category


def test_plural_category_sets():
    assert plural_categories("ar") == ("zero", "one", "two", "few", "many", "other")
    assert plural_categories("ru") == ("one", "few", "many", "other")
    assert plural_categories("ja") == ("other",)
    assert plural_categories("qq") == ("other",)  # unlisted → visible default


# ── catalog parsing & the checker ────────────────────────────────

def test_parse_catalog_issues():
    bad = parse_catalog("en", (
        "Hello\tHello\n"
        "Hello\tAgain\n"                       # duplicate
        "Empty\t\n"                            # empty text
        "{count} thing\t@plural\tonly-one\n"   # en needs 2 forms
        "Broken row with no tab\n"
    ))
    problems = [i.problem for i in bad.issues]
    assert any("duplicate" in p for p in problems)
    assert any("empty translation" in p for p in problems)
    assert any("@plural" in p for p in problems)
    assert any("expected 2 cells" in p for p in problems)
    assert bad.entries["Hello"].text == "Hello"  # first definition kept


def _write_catalog(directory: Path, locale: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{locale}.tsv").write_text(body, encoding="utf-8")


def test_data_only_locale_addition(tmp_path):
    """Adding a locale is adding a data file — no code changes."""
    _write_catalog(tmp_path, "en",
                   f"{WELCOME}\t{WELCOME}\n"
                   f"{MEMBER}\t@plural\t{{count}} member\t{{count}} members\n")
    _write_catalog(tmp_path, "qu",  # Quechua: no plural rule entry, 1 of 2 keys
                   f"{WELCOME}\tAllillanchu **{{group}}**!\n")
    catalogs = load_catalogs([tmp_path])
    tr = Translator(catalogs)
    hit = tr.translate_detailed(WELCOME, "qu", group="Lagos Devs")
    assert hit.text == "Allillanchu **Lagos Devs**!"
    assert hit.exact and hit.served_locale == "qu"
    miss = tr.translate_detailed(MEMBER, "qu", count=3)
    assert miss.text == "3 members"
    assert miss.served_locale == "en" and miss.fell_back
    report = i18n.check_catalogs(catalogs)
    assert report.completeness["qu"] == {
        "translated": 1, "total": 2, "percent": 50.0}


def test_env_catalog_dirs_extend_not_override(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    _write_catalog(extra, "qu", f"{WELCOME}\tAllillanchu **{{group}}**!\n")
    monkeypatch.setenv("NM_I18N_CATALOG_DIRS", str(extra))
    catalogs = load_catalogs()
    assert "qu" in catalogs.locales()
    assert "es" in catalogs.locales()  # shipped set still loaded


def test_checker_catches_placeholder_loss(tmp_path):
    _write_catalog(tmp_path, "en", f"{WELCOME}\t{WELCOME}\n")
    _write_catalog(tmp_path, "es", f"{WELCOME}\t¡Bienvenido!\n")  # dropped {group}
    report = i18n.check_catalogs(load_catalogs([tmp_path]))
    assert not report.ok
    assert any("placeholders" in i["problem"] for i in report.issues)


# ── runtime honesty ──────────────────────────────────────────────

def test_shipped_catalog_serving():
    tr = i18n.default_translator()
    assert tr.translate(WELCOME, "es", group="Lagos Devs") == \
        "¡Bienvenido a **Lagos Devs**!"
    assert tr.translate(MEMBER, "en", count=1) == "1 member"
    assert tr.translate(MEMBER, "en", count=2) == "2 members"
    # Russian grammar through the runtime: 1/2/5/21 forms.
    assert tr.translate(MEMBER, "ru", count=1) == "1 участник"
    assert tr.translate(MEMBER, "ru", count=2) == "2 участника"
    assert tr.translate(MEMBER, "ru", count=5) == "5 участников"
    assert tr.translate(MEMBER, "ru", count=21) == "21 участник"
    # Arabic spelled-out forms carry no {count} and still render.
    assert tr.translate(MEMBER, "ar", count=0) == "لا أعضاء"
    assert tr.translate(MEMBER, "ar", count=2) == "عضوان"
    assert tr.translate(MEMBER, "ar", count=5) == "5 أعضاء"
    # French treats 0 as the singular form.
    assert tr.translate(MEMBER, "fr", count=0) == "0 membre"


def test_fallback_records_itself():
    tr = i18n.default_translator()
    regional = tr.translate_detailed(WELCOME, "pt-BR", group="X")
    assert regional.served_locale == "pt"
    assert regional.fell_back and not regional.exact
    assert regional.source == "catalog"
    unknown_locale = tr.translate_detailed("Left the group.", "uk")
    assert unknown_locale.served_locale == "en"
    assert unknown_locale.source == "catalog"
    unknown_key = tr.translate_detailed("A string nobody catalogued.", "es")
    assert unknown_key.text == "A string nobody catalogued."
    assert unknown_key.source == "source-string"
    assert unknown_key.served_locale == "en"


def test_missing_parameter_fails_fast():
    tr = i18n.default_translator()
    with pytest.raises(I18nError) as excinfo:
        tr.translate(WELCOME, "es")
    assert excinfo.value.details["kind"] == "missing_parameter"
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.code == "i18n.error"


def test_plural_without_count_still_needs_count_param():
    # The {count} placeholder is a real parameter: without it the
    # render refuses rather than printing a literal "{count}".
    tr = i18n.default_translator()
    with pytest.raises(I18nError):
        tr.translate(MEMBER, "en")


# ── bridge: incumbents mapped, not replaced ──────────────────────

def test_bridge_family_packs():
    assert bridge.pack_primary_locale("ng") == "en"
    locales = bridge.pack_locales("ng")
    assert locales == ["en", "pcm", "yo", "ha", "ig"]
    assert bridge.pack_locales("gb") == ["en"]


def test_bridge_mt_codes():
    assert bridge.mt_language_code("pt-BR") == "pt"
    assert bridge.mt_language_code("yo") == "yo"
    # The ladder's table carries no "pcm" — a recorded absence.
    assert bridge.mt_language_code("pcm") == ""
    assert bridge.mt_language_code("qq") == ""


def test_bridge_score_contribution():
    lexicon = {"Welcome": "Bienvenue"}

    def fake_translate(text: str, src: str, dst: str) -> str:
        if src == "en" and dst == "fr":
            return lexicon.get(text, text)
        if src == "fr" and dst == "en":
            return {v: k for k, v in lexicon.items()}.get(text, text)
        return text

    good = bridge.score_contribution(
        "Welcome", "Bienvenue", target_locale="fr",
        translate_fn=fake_translate)
    assert good.passed and good.back_translation_f1 == 1.0
    bad = bridge.score_contribution(
        "Welcome", "Zqxjkv nonsense", target_locale="fr",
        translate_fn=fake_translate)
    assert not bad.passed


# ── surface: community groups render from catalogs ───────────────

@pytest.fixture()
def cgroup(tmp_path, monkeypatch):
    from nomorals.community import groups as gmod

    monkeypatch.setattr(gmod, "_DATA_DIR", tmp_path / "cgroup")
    gmod.control_cgroup("new Lagos Devs | build things", sender="Ada")
    gid = gmod.GroupStore().list_groups()[0].id
    return gmod, gid


def test_surface_join_two_locales(cgroup):
    gmod, gid = cgroup
    en = gmod.control_cgroup(f"join {gid}", sender="Bob")
    assert en.startswith("Welcome to **Lagos Devs**! (2 members)")
    gmod.control_cgroup(f"lang {gid} es", sender="Ada")
    es = gmod.control_cgroup(f"join {gid}", sender="Caro")
    assert es.startswith("¡Bienvenido a **Lagos Devs**! (3 miembros)")
    gmod.control_cgroup(f"lang {gid} ru", sender="Ada")
    ru_few = gmod.control_cgroup(f"join {gid}", sender="Dima")
    assert ru_few.startswith("Добро пожаловать в **Lagos Devs**! (4 участника)")
    ru_many = gmod.control_cgroup(f"join {gid}", sender="Elena")
    assert ru_many.startswith("Добро пожаловать в **Lagos Devs**! (5 участников)")


def test_surface_membership_flow_localized(cgroup):
    gmod, gid = cgroup
    gmod.control_cgroup(f"join {gid}", sender="Bob")
    gmod.control_cgroup(f"lang {gid} es", sender="Ada")
    # The incumbent handler splits the tail into 3 segments
    # (group, _, answer) — segs[2] is the recorded answer.
    assert gmod.control_cgroup(f"answer {gid} a si", sender="Bob") == \
        "Respuesta registrada. 👍"
    assert gmod.control_cgroup(f"leave {gid}", sender="Bob") == \
        "Saliste del grupo."
    assert gmod.control_cgroup(f"leave {gid}", sender="Bob") == \
        "No se pudo salir del grupo."


def test_surface_language_setting_normalizes_and_persists(cgroup):
    gmod, gid = cgroup
    out = gmod.control_cgroup(f"lang {gid} pt_br", sender="Ada")
    assert out == "Group language set to pt-BR."
    g = gmod.GroupStore().get(gid)
    assert g is not None and g.language == "pt-BR"
    # Old-shape JSON (no language field) still loads, defaulting "".
    from nomorals.community.groups import ThemedGroup
    legacy = ThemedGroup.from_dict({"id": "g1", "name": "Old", "topic": ""})
    assert legacy.language == ""
