"""Message catalogs: TSV data files, parsed and validated.

The craft-i18n shape, ported: one TSV file per locale in a
catalog directory, rows keyed by the **English source string**
(the key is the message id *and* the default-locale text, so a
missing translation can always fall back to something truthful).
Catalogs are data — a pinned in-repo input, never a runtime
dependency; adding a locale is adding a file.

Row grammar (tab-separated, ``#`` lines are comments)::

    key<TAB>text                          a plain message
    key<TAB>@plural<TAB>form<TAB>form…    one form per plural
                                          category of the locale,
                                          in CLDR category order

Placeholders are ``{name}`` fields, identical in every locale's
rendering of a key — the checker enforces that against the source
catalog, because a translation that drops ``{group}`` is a bug,
not a style choice.

Parsing never raises on a bad row: problems are collected as
:class:`CatalogIssue` records so the checker can report every one
of them; lookups work with whatever rows parsed cleanly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .locale import DEFAULT_LOCALE, normalize_locale
from .plural import plural_categories

__all__ = [
    "CatalogIssue",
    "Entry",
    "Catalog",
    "CatalogSet",
    "placeholders",
    "default_catalog_dir",
    "catalog_dirs_from_env",
    "load_catalogs",
]

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Env var naming extra catalog directories (os.pathsep-separated),
#: loaded after the in-repo directory so deployments can extend —
#: never silently override — the shipped catalogs.
ENV_CATALOG_DIRS = "NM_I18N_CATALOG_DIRS"


def placeholders(text: str) -> tuple[str, ...]:
    """The ``{name}`` fields in a message, in order, de-duplicated."""
    out: list[str] = []
    for name in _PLACEHOLDER_RE.findall(text or ""):
        if name not in out:
            out.append(name)
    return tuple(out)


@dataclass(frozen=True)
class CatalogIssue:
    """One catalog problem, located precisely enough to fix."""

    locale: str
    line: int
    key: str
    problem: str

    def to_dict(self) -> dict[str, Any]:
        return {"locale": self.locale, "line": self.line,
                "key": self.key, "problem": self.problem}


@dataclass(frozen=True)
class Entry:
    """One catalog row: a plain text, or plural forms per category."""

    key: str
    text: str = ""
    forms: tuple[str, ...] = ()

    @property
    def is_plural(self) -> bool:
        return bool(self.forms)

    def texts(self) -> tuple[str, ...]:
        return self.forms if self.is_plural else (self.text,)


@dataclass
class Catalog:
    """The parsed rows of one locale."""

    locale: str
    entries: dict[str, Entry] = field(default_factory=dict)
    issues: list[CatalogIssue] = field(default_factory=list)
    source: str = ""

    def get(self, key: str) -> Entry | None:
        return self.entries.get(key)


def parse_catalog(locale: str, text: str, source: str = "") -> Catalog:
    """Parse one locale's TSV text into a :class:`Catalog`."""
    locale = normalize_locale(locale) or locale
    catalog = Catalog(locale=locale, source=source)
    categories = plural_categories(locale)
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cells = raw.split("\t")
        key = cells[0]
        if not key:
            catalog.issues.append(CatalogIssue(
                locale, lineno, "", "row with an empty key"))
            continue
        if key in catalog.entries:
            catalog.issues.append(CatalogIssue(
                locale, lineno, key, "duplicate key"))
            continue
        if len(cells) >= 2 and cells[1] == "@plural":
            forms = tuple(cells[2:])
            if not forms or any(f == "" for f in forms):
                catalog.issues.append(CatalogIssue(
                    locale, lineno, key, "@plural row with an empty form"))
                continue
            if len(forms) != len(categories):
                catalog.issues.append(CatalogIssue(
                    locale, lineno, key,
                    f"@plural row has {len(forms)} form(s) but locale "
                    f"{locale} has {len(categories)} categories "
                    f"{list(categories)}"))
                continue
            catalog.entries[key] = Entry(key=key, forms=forms)
            continue
        if len(cells) != 2:
            catalog.issues.append(CatalogIssue(
                locale, lineno, key,
                f"expected 2 cells (key, text), found {len(cells)}"))
            continue
        if cells[1] == "":
            catalog.issues.append(CatalogIssue(
                locale, lineno, key, "empty translation text"))
            continue
        catalog.entries[key] = Entry(key=key, text=cells[1])
    return catalog


def default_catalog_dir() -> Path:
    """The in-repo catalog directory shipped with the package."""
    return Path(__file__).resolve().parent / "catalogs"


def catalog_dirs_from_env() -> list[Path]:
    raw = os.environ.get(ENV_CATALOG_DIRS, "")
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


@dataclass
class CatalogSet:
    """Every loaded catalog, keyed by normalized locale."""

    catalogs: dict[str, Catalog] = field(default_factory=dict)
    dirs: list[Path] = field(default_factory=list)

    def locales(self) -> list[str]:
        return sorted(self.catalogs)

    def catalog(self, locale: str) -> Catalog | None:
        return self.catalogs.get(normalize_locale(locale))

    def source_catalog(self) -> Catalog | None:
        return self.catalogs.get(DEFAULT_LOCALE)

    def all_issues(self) -> list[CatalogIssue]:
        out: list[CatalogIssue] = []
        for locale in self.locales():
            out.extend(self.catalogs[locale].issues)
        return out

    def completeness(self) -> dict[str, dict[str, Any]]:
        """Per-locale coverage against the source catalog's keys.

        A key counts translated when the locale carries its own row
        for it (plain or plural).  The source locale reports 100%
        by definition; a locale with no catalog is absent here —
        absence is its own fact, recorded by callers via the
        fallback machinery, not dressed up as 0% of a catalog that
        does not exist.
        """
        source = self.source_catalog()
        total = len(source.entries) if source else 0
        report: dict[str, dict[str, Any]] = {}
        for locale in self.locales():
            if locale == DEFAULT_LOCALE:
                report[locale] = {"translated": total, "total": total,
                                  "percent": 100.0}
                continue
            have = sum(1 for key in (source.entries if source else ())
                       if key in self.catalogs[locale].entries)
            report[locale] = {
                "translated": have, "total": total,
                "percent": round(100.0 * have / total, 1) if total else 0.0,
            }
        return report


def load_catalogs(dirs: list[Path] | None = None) -> CatalogSet:
    """Load every ``<locale>.tsv`` in ``dirs`` (default: in-repo +
    ``NM_I18N_CATALOG_DIRS``).  Later directories add locales and
    keys but never override a key an earlier directory defined —
    extension is additive, so a deployment catalog cannot silently
    rewrite a shipped string; a collision is recorded as an issue.
    """
    if dirs is None:
        dirs = [default_catalog_dir(), *catalog_dirs_from_env()]
    result = CatalogSet(dirs=list(dirs))
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.tsv")):
            locale = normalize_locale(path.stem)
            if not locale:
                continue
            parsed = parse_catalog(
                locale, path.read_text(encoding="utf-8"),
                source=str(path))
            existing = result.catalogs.get(locale)
            if existing is None:
                result.catalogs[locale] = parsed
                continue
            for key, entry in parsed.entries.items():
                if key in existing.entries:
                    existing.issues.append(CatalogIssue(
                        locale, 0, key,
                        f"key also defined in {existing.source}; "
                        "first definition kept"))
                else:
                    existing.entries[key] = entry
            existing.issues.extend(parsed.issues)
    return result
