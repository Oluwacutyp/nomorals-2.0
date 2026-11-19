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

__all__ = ["CheckReport", "check_catalogs", "render_parity",
           "check_upstream_tree", "check_upstream_text",
           "upstream_coverage"]


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


# ── the upstream checker (craft-i18n-tests tree.rs) ──────────

#: The shared catalog's file name.
COMMON = "common.tsv"

#: What a row problem means, for the machine-readable ``kind``.
#: The ``problem`` text keeps upstream's wording.
_KIND_BY_MARKER = (
    ("empty translation", "empty_translation"),
    ("empty source text", "empty_source"),
    ("defined twice", "duplicate_key"),
    ("columns; expected 3", "bad_row"),
    ("invisible character", "invisible_character"),
    ("unknown escape", "bad_escape"),
    ("trailing backslash", "bad_escape"),
    ("unknown reserved context", "bad_context"),
    ("plural source text must be one|other", "plural_form_count"),
    ("plural source forms must be nonempty", "plural_form_count"),
    ("expected ", "plural_form_count"),
    ("plural form placeholders differ", "plural_form_count"),
    ("placeholders differ", "placeholder_mismatch"),
    ("trailing … differs", "trailing_ellipsis"),
    ("@msg template", "bad_template"),
)


def _kind_of(problem: str) -> str:
    for marker, kind in _KIND_BY_MARKER:
        if marker in problem:
            return kind
    return "bad_row"


def _check_locale_name(name: str) -> str | None:
    """A locale directory name's problem, or ``None``: a BCP 47
    tag written in lowercase (``pt-br``, ``zh-hans``).  The
    stdlib carries no BCP 47 parser, so the shape is the tag
    grammar the checker needs — language, optional script,
    optional region — all lowercase."""
    import re as _re

    if not _re.fullmatch(
            r"[a-z]{2,3}(-[a-z]{4})?(-([a-z]{2}|[0-9]{3}))?", name):
        return f"{name!r} is not a BCP 47 locale written in " \
            "lowercase"
    return None


def _plural_forms(locale: str) -> int:
    """How many plural forms ``locale``'s ``@plural`` rows list
    (the CLDR categories whole numbers 0..1000 reach).  Raises
    for the source language (``en`` has no catalog) and for a
    language with no plural rules."""
    from .locale import language_of
    from .plural import has_plural_rules, plural_categories

    if language_of(locale) == "en":
        raise ValueError(
            f"{locale!r}: the source language; no catalog")
    if not has_plural_rules(locale):
        raise ValueError(
            f"{locale!r}: no plural rules for this locale")
    return len(plural_categories(locale))


def _is_catalog_name(file: str) -> bool:
    """``common.tsv``, or ``<app>.tsv`` with ``<app>`` in
    ``[a-z0-9]+(-[a-z0-9]+)*``."""
    if not file.endswith(".tsv"):
        return False
    app = file[:-4]
    return bool(app) and all(
        part and all(char.islower() or char.isdigit()
                     for char in part)
        for part in app.split("-"))


def _entries(directory: Path) -> list[str]:
    """The names in ``directory``, sorted, without dotfiles.  A
    missing directory has none."""
    if not directory.exists():
        return []
    return sorted(name for name in
                  (entry.name for entry in directory.iterdir())
                  if not name.startswith("."))


def _row_key(context: str, source: str) -> tuple[str, str]:
    return (context, source)


def check_upstream_text(text: str, locale: str) -> list[dict[str, Any]]:
    """The row problems in one upstream catalog text, as issue
    dicts (``path`` empty here — the caller names the file)."""
    from .catalog import parse_upstream_catalog

    catalog = parse_upstream_catalog(text, locale)
    return [{"path": "", "line": issue.line,
             "kind": _kind_of(issue.problem),
             "problem": issue.problem}
            for issue in catalog.issues]


def check_upstream_tree(root: str | Path) -> list[dict[str, Any]]:
    """Every problem under ``root/locales``, each with its path
    relative to ``root`` (``locales/<locale>/<file>``), its line
    and its kind.  No ``locales/`` directory means no locales,
    and no problems.  A faithful port of craft-i18n-tests'
    ``check_tree``: row problems, file naming, locale names,
    app overrides identical to the shared row, a missing
    ``@id language.name`` row, and regional rows that removing
    would not change."""
    from .catalog import load_upstream_tree, parse_upstream_catalog

    root = Path(root)
    issues: list[dict[str, Any]] = []
    locales_dir = root / "locales"
    if not locales_dir.exists():
        return []
    if not locales_dir.is_dir():
        return [{"path": "locales", "line": 0,
                 "kind": "bad_row",
                 "problem": "locales: not a directory"}]
    catalogs: dict[str, dict[str, Any]] = {}
    loaded_tree = load_upstream_tree(root)
    for locale in _entries(locales_dir):
        locale_dir = locales_dir / locale
        if not locale_dir.is_dir():
            issues.append({"path": f"locales/{locale}", "line": 0,
                           "kind": "bad_locale_name",
                           "problem": f"locales/{locale}: not a "
                                      "locale directory"})
            continue
        name_problem = _check_locale_name(locale)
        if name_problem is not None:
            issues.append({"path": f"locales/{locale}", "line": 0,
                           "kind": "bad_locale_name",
                           "problem": f"locales/{locale}: "
                                      f"{name_problem}"})
            continue
        try:
            _plural_forms(locale)
        except ValueError as exc:
            issues.append({"path": f"locales/{locale}", "line": 0,
                           "kind": "no_plural_rules",
                           "problem": f"locales/{locale}: {exc}"})
        files = _entries(locale_dir)
        loaded = loaded_tree.get(locale)
        if loaded is None:
            continue
        shared_rows: set[tuple[str, str, str]] = {
            (key[0], key[1], entry.text if not entry.is_plural
             else "|".join(entry.forms))
            for key, entry in loaded.context_entries.items()}
        catalogs[locale] = {"common": {
            key: (entry.text if not entry.is_plural
                  else "|".join(entry.forms))
            for key, entry in loaded.context_entries.items()},
            "apps": {
                app: {key: (entry.text if not entry.is_plural
                            else "|".join(entry.forms))
                      for key, entry in rows.items()}
                for app, rows in loaded.app_entries.items()}}
        for file in files:
            path = f"locales/{locale}/{file}"
            if not _is_catalog_name(file):
                issues.append({"path": path, "line": 0,
                               "kind": "file_naming",
                               "problem": f"{path}: not a catalog; "
                                          f"name it {COMMON} or "
                                          "<app>.tsv (lowercase, "
                                          "digits, -)"})
                continue
            file_path = locale_dir / file
            try:
                text = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                issues.append({"path": path, "line": 0,
                               "kind": "bad_row",
                               "problem": f"{path}: {exc}"})
                continue
            parsed = parse_upstream_catalog(text, locale)
            if not any(line.startswith("# Clean room:")
                       for line in text.splitlines()):
                issues.append({"path": path, "line": 0,
                               "kind": "header",
                               "problem": f"{path}: no "
                                          "\"# Clean room:\" line; "
                                          "copy the header from "
                                          "docs/catalog-format.md"})
            for issue in parsed.issues:
                issues.append({"path": path, "line": issue.line,
                               "kind": _kind_of(issue.problem),
                               "problem": f"{path}: "
                                          f"{issue.problem}"})
            if file != COMMON:
                for (context, source), entry in \
                        loaded.app_entries.get(
                            file[:-4], {}).items():
                    row_text = entry.text if not entry.is_plural \
                        else "|".join(entry.forms)
                    if (context, source, row_text) in shared_rows:
                        # The line the row sits on, for the report.
                        line_no = 0
                        for lineno, raw in enumerate(
                                text.splitlines(), 1):
                            cells = raw.split("\t")
                            if len(cells) == 3 and cells[0] == context \
                                    and cells[1] == source:
                                line_no = lineno
                                break
                        issues.append({
                            "path": path, "line": line_no,
                            "kind": "app_override_identical",
                            "problem": f"{path}: line {line_no}: "
                                       f"same translation as "
                                       f"{COMMON}; remove the "
                                       "override"})
    for locale, loaded in catalogs.items():
        key = ("@id", "language.name")
        has_name = key in loaded["common"] or any(
            key in rows for rows in loaded["apps"].values())
        if not has_name:
            issues.append({"path": f"locales/{locale}", "line": 0,
                           "kind": "missing_language_name",
                           "problem": f"locales/{locale}: no "
                                      "`@id<TAB>language.name<TAB>"
                                      "<name>` row in common.tsv "
                                      "or an app catalog; the "
                                      "language picker would show "
                                      "the code"})
    issues.extend(_redundant_regional_rows(catalogs))
    return issues


def _base_locale(locale: str,
                 catalogs: dict[str, Any]) -> str | None:
    """The locale ``locale`` falls back to: only a trailing
    region subtag (two letters or three digits) is dropped, and
    only when that locale is in the tree too."""
    if "-" not in locale:
        return None
    parent, region = locale.rsplit("-", 1)
    is_region = ((len(region) == 2
                  and region.islower() and region.isalpha())
                 or (len(region) == 3 and region.isdigit()))
    if is_region and parent in catalogs:
        return parent
    return None


def _lookup_keys(context: str, source: str) -> list[tuple[str, str]]:
    """The keys an app tries for a row, in order: its own, then
    (for a label context) the plain row of the same source text.
    ``@id`` and ``@plural`` rows have no plain fallback."""
    own = (context, source)
    if not context or context in ("@id", "@plural"):
        return [own]
    return [own, ("", source)]


def _resolve(app: str | None, keys: list[tuple[str, str]],
             locale: dict[str, Any], base: dict[str, Any],
             hidden: tuple[str | None, tuple[str, str]] | None
             ) -> str | None:
    """What ``app`` shows for ``keys`` in ``locale``, then in
    ``base``: per locale, the first key found, the app's rows
    over the shared ones.  ``hidden`` is a row of ``locale`` to
    leave out."""

    def in_locale(place: dict[str, Any],
                  hidden_here: tuple[str | None,
                                     tuple[str, str]] | None
                  ) -> str | None:
        for key in keys:
            from_app = None
            if app is not None and hidden_here != (app, key):
                from_app = place["apps"].get(app, {}).get(key)
            if from_app is not None:
                return from_app
            shared = place["common"].get(key)
            if shared is not None and hidden_here != (None, key):
                return shared
        return None

    found = in_locale(locale, hidden)
    if found is not None:
        return found
    return in_locale(base, None)


def _redundant_regional_rows(
        catalogs: dict[str, Any]) -> list[dict[str, Any]]:
    """Every regional row that removing would not change: for
    each app that shows it, the lookup gives the same text
    through the base language."""
    problems: list[dict[str, Any]] = []
    for name, locale in catalogs.items():
        base_name = _base_locale(name, catalogs)
        if base_name is None:
            continue
        base = catalogs[base_name]
        apps = list(locale["apps"]) + list(base["apps"])
        files: list[tuple[str | None, dict[tuple[str, str], str]]] = [
            (None, locale["common"])]
        files.extend((app, rows)
                     for app, rows in locale["apps"].items())
        for file, rows in files:
            shown_in: list[str | None] = (
                [file] if file is not None
                else [None] + apps)
            for key in rows:
                keys = _lookup_keys(key[0], key[1])
                redundant = all(
                    (kept := _resolve(app, keys, locale, base,
                                      None)) is not None
                    and kept == _resolve(app, keys, locale, base,
                                         (file, key))
                    for app in shown_in)
                if redundant:
                    file_name = COMMON if file is None \
                        else f"{file}.tsv"
                    problems.append({
                        "path": f"locales/{name}/{file_name}",
                        "line": 0, "kind": "regional_noop",
                        "problem": f"locales/{name}/{file_name}: "
                                   f"{key[0]!r} {key[1]!r}: same "
                                   f"translation as {base_name}; "
                                   "remove the row"})
    problems.sort(key=lambda issue: (issue["path"],
                                     issue["problem"]))
    return problems


def upstream_coverage(root: str | Path) -> list[str]:
    """One line per catalog, ``<locale>/<file>: <rows>/<max>``,
    where ``<max>`` is the most rows any locale's catalog of
    that name has.  A report only: unreadable files count as
    empty."""
    from .catalog import parse_upstream_catalog

    root = Path(root)
    counts: list[tuple[str, str, int]] = []
    for locale in _entries(root / "locales"):
        locale_dir = root / "locales" / locale
        if not locale_dir.is_dir():
            continue
        for file in _entries(locale_dir):
            if not _is_catalog_name(file):
                continue
            try:
                text = (locale_dir / file).read_text(
                    encoding="utf-8")
            except OSError:
                counts.append((locale, file, 0))
                continue
            parsed = parse_upstream_catalog(text, locale)
            rows = len(parsed.context_entries) + sum(
                len(layer)
                for layer in parsed.app_entries.values())
            counts.append((locale, file, rows))
    maxima: dict[str, int] = {}
    for _, file, rows in counts:
        maxima[file] = max(maxima.get(file, 0), rows)
    return [f"{locale}/{file}: {rows}/{maxima[file]}"
            for locale, file, rows in counts]
