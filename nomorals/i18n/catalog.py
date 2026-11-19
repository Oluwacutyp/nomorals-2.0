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
import unicodedata
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
    "parse_upstream_catalog",
    "load_upstream_tree",
    "dir_from_env",
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
    """One catalog row: a plain text, or plural forms per category.

    ``context`` is the upstream context (``""`` for a plain row;
    ``"@plural"``, ``"@id"`` or ``"@msg"`` for a marked row; any
    other word tells apart two meanings of the same English) and
    ``app`` the app layer the row came from (``None`` for the
    shared catalog).  Devon's own rows carry ``""``/``None``.
    """

    key: str
    text: str = ""
    forms: tuple[str, ...] = ()
    context: str = ""
    app: str | None = None

    @property
    def is_plural(self) -> bool:
        return bool(self.forms)

    def texts(self) -> tuple[str, ...]:
        return self.forms if self.is_plural else (self.text,)


@dataclass
class Catalog:
    """The parsed rows of one locale.

    ``entries`` is the plain (context ``""``) map, kept for
    backward compatibility.  ``context_entries`` is every shared
    row by (context, key); ``app_entries`` is one map per app
    layer, an app's rows overriding the shared ones of the same
    (context, key) — the craft-i18n shape, where each app file
    is a layer over ``common.tsv``.
    """

    locale: str
    entries: dict[str, Entry] = field(default_factory=dict)
    issues: list[CatalogIssue] = field(default_factory=list)
    source: str = ""
    context_entries: dict[tuple[str, str], Entry] = field(
        default_factory=dict)
    app_entries: dict[str, dict[tuple[str, str], Entry]] = field(
        default_factory=dict)

    def get(self, key: str, context: str = "",
            app: str | None = None) -> Entry | None:
        """The entry for ``key`` under ``context`` (``""`` for a
        plain row), from ``app``'s layer when given, else the
        shared rows.  An app row overrides the shared row of the
        same (context, key); a context row is found only under
        its own context.  A named app with no loaded layer has
        no rows at all — the lookup finds nothing rather than
        borrowing the shared row and passing it off as the app's
        wording."""
        if app is not None:
            layer = self.app_entries.get(app)
            if layer is None:
                return None
            entry = layer.get((context, key))
            if entry is not None:
                return entry
        entry = self.context_entries.get((context, key))
        if entry is not None:
            return entry
        if context == "":
            # A catalog built by hand with entries= only (no
            # context map): the plain row still answers.
            return self.entries.get(key)
        return None

    def find(self, keys: list[tuple[str, str]],
             app: str | None = None) -> Entry | None:
        """The first of ``keys`` (context, key) found, each looked
        up in the merged view (app layer over shared) — the
        lookup order ``tc`` uses: in one locale, the context row
        first, then the plain one."""
        for context, key in keys:
            entry = self.get(key, context=context, app=app)
            if entry is not None:
                return entry
        return None


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
            catalog.context_entries[("", key)] = \
                catalog.entries[key]
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
        catalog.context_entries[("", key)] = catalog.entries[key]
    return catalog


#: The env var naming an upstream catalog checkout (a directory
#: with ``locales/<locale>/{common,<app>}.tsv``).  A named
#: checkout replaces the vendored tree wholesale when set and
#: valid.
ENV_UPSTREAM_DIR = "CRAFT_I18N_DIR"

#: The contexts the apps give a meaning to (craft-i18n-tests
#: ``catalog.rs`` RESERVED).  Any other context starting with
#: ``@`` is an error; contexts without ``@`` are free labels.
RESERVED_CONTEXTS = ("@plural", "@id", "@msg")


class CatalogError(ValueError):
    """A catalog problem raised while parsing one row (used
    internally by the upstream parser; :func:`parse_catalog` and
    :func:`parse_upstream_catalog` report problems as
    :class:`CatalogIssue` records instead, never raising)."""


def _unescape(cell: str) -> str:
    """A catalog cell with upstream's escapes undone: ``\\n`` is
    a newline, ``\\t`` a tab, ``\\\\`` a backslash.  Read left to
    right, so ``\\\\n`` is a backslash and then ``n``.  Raises
    :class:`CatalogError` on an unknown escape or a trailing
    backslash."""
    out: list[str] = []
    idx = 0
    while idx < len(cell):
        char = cell[idx]
        if char == "\\" and idx + 1 < len(cell):
            nxt = cell[idx + 1]
            if nxt == "n":
                out.append("\n")
                idx += 2
                continue
            if nxt == "t":
                out.append("\t")
                idx += 2
                continue
            if nxt == "\\":
                out.append("\\")
                idx += 2
                continue
            raise CatalogError(
                f"unknown escape \\{nxt}; use \\n, \\t or \\\\")
        if char == "\\":
            raise CatalogError(
                "trailing backslash; use \\\\ for a backslash")
        out.append(char)
        idx += 1
    return "".join(out)


def _placeholders_sorted(text: str) -> list[str]:
    """The ``{name}`` placeholders in ``text``, sorted, duplicates
    kept: a translation that drops one of two ``{n}``s differs
    from its source, even though the set matches."""
    names: list[str] = []
    rest = text
    while "{" in rest:
        _, after = rest.split("{", 1)
        if "}" not in after:
            break
        name, rest = after.split("}", 1)
        names.append(name)
    names.sort()
    return names


def _check_template(source: str) -> str | None:
    """An ``@msg`` source text's template problem, or ``None``:
    ``{`` with no closing ``}``; two placeholders in a row;
    no literal text at all."""
    literal = 0
    after_placeholder = False
    rest = source
    while "{" in rest:
        before, after_open = rest.split("{", 1)
        if "}" not in after_open:
            return ("@msg template has { with no closing }; "
                    "it can never match")
        _, after = after_open.split("}", 1)
        if not before and after_placeholder:
            return ("@msg template has two placeholders in a row; "
                    "it can never match")
        literal += len(before)
        after_placeholder = True
        rest = after
    if after_placeholder and literal + len(rest) == 0:
        return ("@msg template has no literal text; "
                "it would match any message")
    return None


def _parse_upstream_row(line: str, lineno: int, locale: str,
                        forms: int) -> tuple[Entry | None, str | None]:
    """One upstream row: the entry when it passes every check,
    else the problem (``line N: …``).  A bad row is left out;
    parsing never raises."""
    parts = line.split("\t")
    if len(parts) != 3:
        return None, f"line {lineno}: {len(parts)} columns; " \
            "expected 3"
    if any(char != "\t" and (unicodedata.category(char) == "Cc"
                             or char == "\ufeff")
           for char in line):
        bad = next(char for char in line
                   if char != "\t"
                   and (unicodedata.category(char) == "Cc"
                        or char == "\ufeff"))
        return None, f"line {lineno}: invisible character " \
            f"{bad!r} in the row; write \\n or \\t, or delete it"
    context, source, translation = parts
    if not source:
        return None, f"line {lineno}: empty source text"
    try:
        context = _unescape(context)
        source = _unescape(source)
        translation = _unescape(translation)
    except CatalogError as exc:
        return None, f"line {lineno}: {exc}"
    if not translation:
        return None, f"line {lineno}: empty translation; " \
            "leave the row out instead"
    if context == "@id":
        pass  # a command id, not a template: no further checks
    elif context == "@plural":
        english = source.split("|")
        if len(english) != 2:
            return None, f"line {lineno}: plural source text " \
                "must be one|other"
        one, other = english
        if (not one or not other
                or _placeholders_sorted(one)
                != _placeholders_sorted(other)):
            return None, f"line {lineno}: plural source forms " \
                "must be nonempty with the same placeholders"
        translated = translation.split("|")
        if (len(translated) != forms
                or any(not form for form in translated)):
            return None, f"line {lineno}: expected {forms} " \
                "nonempty plural forms"
        if any(_placeholders_sorted(form)
               != _placeholders_sorted(other)
               for form in translated):
            return None, f"line {lineno}: plural form " \
                "placeholders differ from the source text"
    elif context.startswith("@") \
            and context not in RESERVED_CONTEXTS:
        return None, f"line {lineno}: unknown reserved context " \
            f"{context!r}; use one of {list(RESERVED_CONTEXTS)!r}"
    else:
        if (_placeholders_sorted(source)
                != _placeholders_sorted(translation)):
            return None, f"line {lineno}: placeholders differ " \
                "from the source text"
        if source.endswith("…") != translation.endswith("…"):
            return None, f"line {lineno}: trailing … differs " \
                "from the source text"
        if context == "@msg":
            problem = _check_template(source)
            if problem is not None:
                return None, f"line {lineno}: {problem}"
    if context == "@plural":
        return Entry(key=source, forms=tuple(translation.split("|")),
                     context="@plural"), None
    return Entry(key=source, text=translation,
                 context=context), None


def _parse_upstream_into(text: str, catalog: Catalog, locale: str,
                         app: str | None, path: Path | None) -> None:
    """Parse upstream three-column text into ``catalog``: the
    shared layer when ``app`` is ``None``, else that app's layer.
    Bad rows become :class:`CatalogIssue` records; good rows load.
    A leading byte-order mark and ``\\r\\n`` line endings are
    ignored; a ``\\r`` inside a row is an invisible character."""
    text = text.removeprefix("\ufeff")
    # The locale's plural form count; an unknown locale falls
    # back to one form (the runtime's visible default).
    forms = len(plural_categories(locale))
    seen: set[tuple[str, str]] = set()
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line.strip() or line.startswith("#"):
            continue
        entry, problem = _parse_upstream_row(line, lineno, locale,
                                             forms)
        if problem is not None:
            catalog.issues.append(CatalogIssue(
                locale, lineno, "", problem))
            continue
        assert entry is not None
        if (entry.context, entry.key) in seen:
            catalog.issues.append(CatalogIssue(
                locale, lineno, entry.key,
                f"line {lineno}: defined twice (same context and "
                "source text)"))
            continue
        seen.add((entry.context, entry.key))
        entry = Entry(key=entry.key, text=entry.text,
                      forms=entry.forms, context=entry.context,
                      app=app)
        if app is None:
            catalog.context_entries[(entry.context, entry.key)] = \
                entry
            if entry.context == "":
                catalog.entries[entry.key] = entry
        else:
            catalog.app_entries.setdefault(app, {})[
                (entry.context, entry.key)] = entry


def parse_upstream_catalog(text: str, locale: str,
                           source: str = "") -> Catalog:
    """Parse one upstream three-column catalog file (usually a
    locale's ``common.tsv``) into a :class:`Catalog`.  Problems
    are collected as :class:`CatalogIssue` records — parsing
    never raises — and the good rows load."""
    locale = normalize_locale(locale) or locale
    catalog = Catalog(locale=locale, source=source)
    _parse_upstream_into(text, catalog, locale, None,
                         Path(source) if source else None)
    return catalog


def load_upstream_tree(root: str | Path) -> dict[str, Catalog]:
    """Load an upstream catalog tree — a directory with
    ``locales/<locale>/{common,<app>}.tsv`` — into one
    :class:`Catalog` per locale.  Each locale's ``common.tsv``
    is the shared layer; each app file is that app's layer, its
    rows overriding the shared ones.  A locale with no
    ``common.tsv`` still loads from its app files; a directory
    with no ``locales/`` loads nothing."""
    root = Path(root)
    locales = root / "locales"
    out: dict[str, Catalog] = {}
    if not locales.is_dir():
        return out
    for locale_dir in sorted(locales.iterdir()):
        if not locale_dir.is_dir():
            continue
        locale = locale_dir.name
        catalog = Catalog(locale=locale, source=str(locale_dir))
        common = locale_dir / "common.tsv"
        if common.is_file():
            _parse_upstream_into(
                common.read_text(encoding="utf-8"), catalog,
                locale, None, common)
        for app_file in sorted(locale_dir.glob("*.tsv")):
            if app_file.name == "common.tsv":
                continue
            _parse_upstream_into(
                app_file.read_text(encoding="utf-8"), catalog,
                locale, app_file.stem, app_file)
        out[locale] = catalog
    return out


def dir_from_env() -> Path | None:
    """The upstream catalog checkout ``CRAFT_I18N_DIR`` names, or
    ``None`` when it is unset.  A set variable that names no
    catalog tree raises :class:`ValueError`: the checkout is the
    source of truth when it is named, never a silent fall-back."""
    raw = os.environ.get(ENV_UPSTREAM_DIR, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_dir() or not (path / "locales").is_dir():
        raise ValueError(
            f"{ENV_UPSTREAM_DIR}={raw}: no locales/ directory there")
    return path


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
