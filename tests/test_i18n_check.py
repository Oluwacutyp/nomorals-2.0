"""The i18n checker in CI-gate form (the plan's ``check_i18n``).

The shipped catalogs must be valid, every fallback chain must
resolve, and completeness must be reported per locale — and the
gate must be able to FAIL, proven with a deliberately broken
catalog set so a green run here means something.
"""

from __future__ import annotations

from pathlib import Path

from nomorals.i18n import check_catalogs, load_catalogs, render_parity

SHIPPED_LOCALES = {
    "ar", "de", "en", "es", "fr", "ha", "hi", "ig", "it", "ja",
    "nl", "pcm", "pt", "ru", "sw", "yo",
}


def test_shipped_catalogs_are_valid():
    report = check_catalogs()
    assert report.ok, f"catalog issues: {report.issues}"
    assert report.issues == []


def test_shipped_completeness_reported_per_locale():
    report = check_catalogs()
    assert set(report.locales) == SHIPPED_LOCALES
    for locale in SHIPPED_LOCALES:
        row = report.completeness[locale]
        assert row["total"] == 6  # the wired surface's key set
        assert row["percent"] == 100.0
    table = render_parity(report)
    for locale in SHIPPED_LOCALES:
        assert f"| {locale} |" in table


def test_gate_can_fail(tmp_path):
    (tmp_path / "en.tsv").write_text(
        "Welcome to **{group}**!\tWelcome to **{group}**!\n",
        encoding="utf-8")
    (tmp_path / "es.tsv").write_text(
        "Welcome to **{group}**!\t¡Bienvenido!\n"      # placeholder lost
        "Ghost key\tFantasma\n",                        # not in source
        encoding="utf-8")
    report = check_catalogs(load_catalogs([tmp_path]))
    assert not report.ok
    problems = [i["problem"] for i in report.issues]
    assert any("placeholders" in p for p in problems)
    assert any("not present in the source" in p for p in problems)


def test_gate_fails_without_source_catalog(tmp_path):
    (tmp_path / "es.tsv").write_text("Hola\tHola\n", encoding="utf-8")
    report = check_catalogs(load_catalogs([tmp_path]))
    assert not report.ok
    assert any("no source catalog" in i["problem"] for i in report.issues)
