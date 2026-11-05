"""The translation runtime: chain lookup, plurals, interpolation.

``Translator.translate_detailed`` is the honest core: it walks the
fallback chain (``pt-BR → pt → en``), serves the first catalog that
carries the key, and returns a :class:`Translation` recording what
was *requested* and what actually *served*.  A missing translation
never blocks and never lies — when no catalog carries the key, the
English source string itself is the text (keys *are* source
strings), served by the default locale and recorded as
``source="source-string"``.

Plurals: when ``count`` is given and the served entry is a
``@plural`` row, the form is chosen by the *served* locale's CLDR
rule — a Russian fallback serves Russian grammar, not English
grammar with Russian words.  Without a count, the ``other`` form
is the one selected (the generic form in most languages) — and if
that form still references ``{count}``, the render refuses like
any other missing parameter rather than printing a literal.

Interpolation is strict in one direction only: a placeholder the
chosen text needs but the caller did not supply raises
:class:`I18nError` naming it (fail fast — a literal ``{group}``
in a member's welcome message is worse than an error in a log);
extra parameters are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..core.errors import NoMoralsError
from .catalog import CatalogSet, Entry, load_catalogs, placeholders
from .locale import DEFAULT_LOCALE, fallback_chain, normalize_locale
from .plural import plural_categories, plural_category

__all__ = [
    "I18nError",
    "I18N_ERROR_KINDS",
    "Translation",
    "Translator",
    "default_translator",
    "translate",
    "translate_detailed",
    "reset_default_translator",
]

_FIELD_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


#: The closed taxonomy of i18n failure kinds, mirrored into
#: ``details["kind"]`` (the Echelon standard: machine-readable
#: kinds separate from human messages).
I18N_ERROR_KINDS = frozenset({
    "error",              # unclassified
    "missing_parameter",  # a placeholder had no supplied value
    "invalid_locale",     # a locale tag that cannot be normalized
    "catalog",            # a catalog problem surfaced at runtime
})


class I18nError(NoMoralsError, ValueError):
    """A translation that cannot be rendered truthfully.

    Carries a stable ``code`` (``i18n.error``) and a ``kind`` from
    :data:`I18N_ERROR_KINDS`; a ``ValueError`` so plain
    ``except ValueError`` handlers keep working.
    """

    code = "i18n.error"

    def __init__(self, message: str = "", *, kind: str = "error",
                 code: str | None = None,
                 details: dict[str, Any] | None = None,
                 retryable: bool | None = None) -> None:
        if kind not in I18N_ERROR_KINDS:
            raise ValueError(f"unknown i18n error kind: {kind!r}")
        merged = {**(details or {}), "kind": kind}
        super().__init__(message, code=code, details=merged,
                         retryable=retryable)


@dataclass(frozen=True)
class Translation:
    """One rendered message and the record of how it was served."""

    key: str
    text: str
    requested_locale: str
    served_locale: str
    #: "catalog" — a catalog row served; "source-string" — no
    #: catalog carried the key and the source string itself served.
    source: str

    @property
    def exact(self) -> bool:
        """True when the requested locale itself served the text."""
        return self.served_locale == self.requested_locale

    @property
    def fell_back(self) -> bool:
        return not self.exact

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "text": self.text,
                "requested_locale": self.requested_locale,
                "served_locale": self.served_locale,
                "source": self.source, "exact": self.exact}


def _pick_form(entry: Entry, locale: str, count: float | None) -> str:
    categories = plural_categories(locale)
    category = "other" if count is None else plural_category(locale, count)
    if category not in categories:
        category = "other"
    return entry.forms[categories.index(category)]


def _interpolate(text: str, values: dict[str, Any], key: str) -> str:
    missing = [name for name in placeholders(text) if name not in values]
    if missing:
        raise I18nError(
            f"translation of {key!r} needs parameter(s) "
            f"{', '.join(missing)} which were not supplied",
            kind="missing_parameter")
    return _FIELD_RE.sub(lambda m: str(values[m.group(1)]), text)


class Translator:
    """Renders catalog keys for a locale, with the record kept."""

    def __init__(self, catalogs: CatalogSet) -> None:
        self.catalogs = catalogs

    def translate_detailed(self, key: str, locale: str | None = None,
                           *, count: float | None = None,
                           **params: Any) -> Translation:
        requested = normalize_locale(locale) or DEFAULT_LOCALE
        values = dict(params)
        if count is not None:
            values.setdefault("count", count)
        for served in fallback_chain(requested):
            catalog = self.catalogs.catalog(served)
            entry = catalog.get(key) if catalog else None
            if entry is None:
                continue
            text = (_pick_form(entry, served, count)
                    if entry.is_plural else entry.text)
            return Translation(
                key=key, text=_interpolate(text, values, key),
                requested_locale=requested, served_locale=served,
                source="catalog")
        return Translation(
            key=key, text=_interpolate(key, values, key),
            requested_locale=requested, served_locale=DEFAULT_LOCALE,
            source="source-string")

    def translate(self, key: str, locale: str | None = None,
                  *, count: float | None = None, **params: Any) -> str:
        return self.translate_detailed(
            key, locale, count=count, **params).text


_default: Translator | None = None


def default_translator() -> Translator:
    """The process-wide translator over the shipped catalogs."""
    global _default
    if _default is None:
        _default = Translator(load_catalogs())
    return _default


def reset_default_translator() -> None:
    """Drop the cached default translator (tests, catalog reloads)."""
    global _default
    _default = None


def translate(key: str, locale: str | None = None,
              *, count: float | None = None, **params: Any) -> str:
    return default_translator().translate(key, locale, count=count, **params)


def translate_detailed(key: str, locale: str | None = None,
                       *, count: float | None = None,
                       **params: Any) -> Translation:
    return default_translator().translate_detailed(
        key, locale, count=count, **params)
