"""The catalog checker — craft-i18n's checker crate, Devon-shaped.

Validates a whole catalog set and reports, per locale, how complete
it is.  Three layers of checks:

1. **Row validity** — the parse issues collected by
   :mod:`nomorals.i18n.catalog` (duplicates, empty texts, wrong
   ``@plural`` form counts for the locale's categories).
2. **Source parity** — every translated key exists in the source
   catalog; a plain translation carries exactly the source's
   placeholders (a dropped ``{group}`` is a bug, not a style
   choice); plural entries are checked collectively (no invented
   placeholders, none lost across all forms — CLDR zero/one/two
   forms may spell the number out); plural keys stay plural and
   plain keys stay plain across locales.
3. **Chain resolution** — every shipped locale's fallback chain
   terminates at a catalog that exists (the source catalog), so a
   lookup can never walk off the end of the world.

The report is data (:class:`CheckReport`); ``render_parity`` turns
it into the markdown table that lives in ``i18n/PARITY.md``, and
``tests/test_i18n_check.py`` is the CI-gate form: zero errors,
chains resolve, completeness reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import CatalogSet, load_catalogs, placeholders
from .locale import DEFAULT_LOCALE, fallback_chain

__all__ = ["CheckReport", "check_catalogs", "render_parity"]


@dataclass
class CheckReport:
    ok: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    completeness: dict[str, dict[str, Any]] = field(default_factory=dict)
    locales: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": self.issues,
                "completeness": self.completeness,
                "locales": self.locales}


def check_catalogs(catalogs: CatalogSet | None = None,
                   dirs: list[Path] | None = None) -> CheckReport:
    """Run every check over ``catalogs`` (default: shipped set)."""
    if catalogs is None:
        catalogs = load_catalogs(dirs)
    issues: list[dict[str, Any]] = [i.to_dict()
                                    for i in catalogs.all_issues()]
    source = catalogs.source_catalog()
    if source is None:
        issues.append({"locale": DEFAULT_LOCALE, "line": 0, "key": "",
                       "problem": "no source catalog "
                                  f"({DEFAULT_LOCALE}.tsv) loaded"})
        return CheckReport(ok=False, issues=issues,
                           completeness=catalogs.completeness(),
                           locales=catalogs.locales())
    for locale in catalogs.locales():
        catalog = catalogs.catalog(locale)
        assert catalog is not None  # locales() comes from the set
        # Chain resolution: the walk must reach a loaded catalog
        # that can serve (at minimum the source catalog).
        chain = fallback_chain(locale)
        if not any(catalogs.catalog(link) is not None for link in chain):
            issues.append({"locale": locale, "line": 0, "key": "",
                           "problem": f"fallback chain {chain} reaches "
                                      "no loaded catalog"})
        if locale == DEFAULT_LOCALE:
            continue
        for key, entry in catalog.entries.items():
            src = source.entries.get(key)
            if src is None:
                issues.append({"locale": locale, "line": 0, "key": key,
                               "problem": "key not present in the "
                                          "source catalog"})
                continue
            if entry.is_plural != src.is_plural:
                issues.append({"locale": locale, "line": 0, "key": key,
                               "problem": "plural/plain shape differs "
                                          "from the source entry"})
                continue
            if not entry.is_plural:
                if set(placeholders(entry.text)) != set(placeholders(src.text)):
                    issues.append({
                        "locale": locale, "line": 0, "key": key,
                        "problem": f"placeholders {placeholders(entry.text)} "
                                   f"differ from source "
                                   f"{placeholders(src.text)}"})
                continue
            # Plural parity is collective, not per-form: CLDR zero/
            # one/two forms legitimately spell the number out
            # (Arabic "عضوان" — "two members" — carries no {count}).
            # The guarded bug class is a placeholder invented out of
            # nowhere, or one the source needs that NO form carries.
            src_fields = set(placeholders(src.texts()[0]))
            form_fields = [set(placeholders(f)) for f in entry.texts()]
            carried = set().union(*form_fields) if form_fields else set()
            invented = carried - src_fields
            lost = src_fields - carried
            if invented:
                issues.append({
                    "locale": locale, "line": 0, "key": key,
                    "problem": f"plural forms introduce placeholder(s) "
                               f"{sorted(invented)} absent from the source"})
            if lost:
                issues.append({
                    "locale": locale, "line": 0, "key": key,
                    "problem": f"placeholder(s) {sorted(lost)} of the "
                               f"source appear in no plural form"})
    return CheckReport(ok=not issues, issues=issues,
                       completeness=catalogs.completeness(),
                       locales=catalogs.locales())


def render_parity(report: CheckReport) -> str:
    """The per-locale completeness table for ``i18n/PARITY.md``."""
    lines = [
        "| locale | translated | total | completeness |",
        "|---|---|---|---|",
    ]
    for locale in report.locales:
        row = report.completeness.get(locale, {})
        lines.append(f"| {locale} | {row.get('translated', 0)} "
                     f"| {row.get('total', 0)} "
                     f"| {row.get('percent', 0.0)}% |")
    return "\n".join(lines)
