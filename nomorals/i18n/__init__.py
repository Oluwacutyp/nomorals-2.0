"""Devon's internationalization system (Echelon Wave 7).

Greenfield adoption of craft-i18n's shape, at L3 beside ``llm``:

* **Catalogs are data** — one TSV per locale under ``catalogs/``,
  rows keyed by the English source string, ``@plural`` rows for
  count-shaped messages.  Adding a locale is adding a file;
  catalogs are a pinned in-repo input, never a runtime service.
* **Fallback is a recorded fact** — lookups walk
  ``pt-BR → pt → en`` and the returned :class:`Translation` names
  the locale that actually served.  A missing translation never
  blocks and is never dressed up as the requested locale.
* **Plurals are per-locale** — CLDR integer cardinal rules, so a
  Russian member count picks among four forms and a Japanese one
  among one, by rule rather than by English habit.
* **The checker is the gate** — every row valid, placeholders at
  parity with the source, chains resolving, completeness reported
  per locale (``i18n/PARITY.md``; ``tests/test_i18n_check.py``).

Incumbents are mapped, not replaced: ``family/locales.py`` region
packs and the ``llm`` translation ladder keep their jobs;
:mod:`nomorals.i18n.bridge` is the explicit seam, and
``llm/translation_quality.py`` remains the quality bridge for
machine-translated catalog contributions.

First consumer (the plan's deliberate pick): the community group
surfaces — ``community/groups.py`` membership messages render from
these catalogs.  CLI surfaces follow in a later pass.

Honest limit, recorded: Python's stdlib carries no ICU, so number
and date *formatting* is plain interpolation (``{count}`` renders
the integer as given, no locale grouping) until a formatting rung
earns its place; message selection, plurals and fallback — the
parts that change meaning — are fully rule-driven here.
"""

from __future__ import annotations

from .catalog import (
    Catalog,
    CatalogIssue,
    CatalogSet,
    Entry,
    load_catalogs,
    parse_catalog,
    placeholders,
)
from .check import CheckReport, check_catalogs, render_parity
from .locale import DEFAULT_LOCALE, fallback_chain, language_of, normalize_locale
from .plural import plural_categories, plural_category
from .runtime import (
    I18nError,
    Translation,
    Translator,
    default_translator,
    reset_default_translator,
    translate,
    translate_detailed,
)

__all__ = [
    "Catalog",
    "CatalogIssue",
    "CatalogSet",
    "Entry",
    "load_catalogs",
    "parse_catalog",
    "placeholders",
    "CheckReport",
    "check_catalogs",
    "render_parity",
    "DEFAULT_LOCALE",
    "fallback_chain",
    "language_of",
    "normalize_locale",
    "plural_categories",
    "plural_category",
    "I18nError",
    "Translation",
    "Translator",
    "default_translator",
    "reset_default_translator",
    "translate",
    "translate_detailed",
]
