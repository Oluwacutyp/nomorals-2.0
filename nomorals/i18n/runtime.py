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

import math
import os
import re
from dataclasses import dataclass
from typing import Any

from ..core.errors import NoMoralsError
from .bidi import fmt as _bidi_fmt
from .bidi import ltr as _bidi_ltr
from .bidi import strip_isolates as _strip_isolates
from .catalog import CatalogSet, Entry, load_catalogs, placeholders
from .locale import DEFAULT_LOCALE, fallback_chain, normalize_locale
from .plural import plural_categories, plural_category
from .scripts import Direction as _Direction
from .scripts import direction as _script_direction

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
        # A category whole numbers never reach has no form of
        # its own (upstream's rule): Russian and Ukrainian
        # decimals ("other") take the "few" form; any other takes
        # the last.
        category = "few" if "few" in categories else categories[-1]
    return entry.forms[categories.index(category)]


def _interpolate(text: str, values: dict[str, Any], key: str) -> str:
    missing = [name for name in placeholders(text) if name not in values]
    if missing:
        raise I18nError(
            f"translation of {key!r} needs parameter(s) "
            f"{', '.join(missing)} which were not supplied",
            kind="missing_parameter")
    return _FIELD_RE.sub(lambda m: str(values[m.group(1)]), text)


class _PlainFormat:
    """Numbers as plain ASCII, until a locale formatter is
    injected: ``integer`` writes a whole number, ``number``
    writes a decimal with ``decimals`` digits.  Infinities and
    NaN are ``∞``, ``-∞`` and ``NaN`` (the craftlibs shapes)."""

    def integer(self, number: int) -> str:
        return str(int(number))

    def number(self, value: float, decimals: int) -> str:
        if math.isnan(float(value)):
            return "NaN"
        if math.isinf(float(value)):
            return "∞" if float(value) > 0 else "-∞"
        return f"{float(value):.{decimals}f}"


def _strip_zeros(rendered: str) -> str:
    """A decimal rendering without trailing zeros or a trailing
    point: ``"1.0"`` → ``"1"``, ``"1.50"`` → ``"1.5"``."""
    if "." not in rendered:
        return rendered
    return rendered.rstrip("0").rstrip(".")


class Translator:
    """Renders catalog keys for a locale, with the record kept.

    Built over Devon's catalogs, and optionally over an
    upstream set (``from_upstream``): the Devon ``translate`` /
    ``translate_detailed`` consult the upstream catalogs first
    (upstream chain), then Devon's (Devon chain), then the
    source string.  The ``t`` / ``tc`` / ``get`` / ``tid`` /
    ``trn`` / ``trn_plain`` / ``trn_decimal`` methods are the
    craft-libs ``I18n`` API on the same object: ``t`` is a plain
    lookup (the source text when missing), ``tc`` tries the
    context row then the plain one, ``get`` asks one
    (context, source) pair only, ``tid`` resolves a command id,
    and the ``trn`` family renders a plural count.  ``t`` and
    friends take no strict parameters — extra values fill
    through the bidi-safe ``fmt`` (unknown placeholders stay);
    ``translate`` keeps its strict interpolation (a missing
    parameter raises).
    """

    def __init__(self, catalogs: CatalogSet,
                 upstream: Any | None = None,
                 formatter: Any | None = None,
                 default: str = DEFAULT_LOCALE) -> None:
        self.catalogs = catalogs
        self.upstream = upstream
        self._formatter = formatter if formatter is not None \
            else _PlainFormat()
        self._looked_up: set[str] = set()
        # The language the source strings are written in: the
        # fallback a missing translation serves as.
        self.default = normalize_locale(default) or DEFAULT_LOCALE

    @classmethod
    def from_upstream(cls, source: Any,
                      default: str = DEFAULT_LOCALE,
                      formatter: Any | None = None) -> Translator:
        """A translator over an upstream catalog tree (a directory
        path) or ``{locale: common.tsv text}`` (a dict).  Devon's
        own catalogs are empty here; the upstream set serves."""
        from .upstream import UpstreamCatalogSet

        if isinstance(source, dict):
            upstream = UpstreamCatalogSet.from_text(source)
        else:
            upstream = UpstreamCatalogSet.from_dir(source)
        return cls(CatalogSet(), upstream=upstream,
                   formatter=formatter, default=default)

    def translate_detailed(self, key: str, locale: str | None = None,
                           *, count: float | None = None,
                           **params: Any) -> Translation:
        requested = normalize_locale(locale) or DEFAULT_LOCALE
        values = dict(params)
        if count is not None:
            values.setdefault("count", count)
            # Upstream plural rows spell the count "{n}"; a count
            # fills both spellings.
            values.setdefault("n", count)
        if self.upstream is not None:
            found = self.upstream.find(
                [("", key), ("@plural", key)], locale=requested)
            if found is not None:
                entry, served = found
                text = (_pick_form(entry, served, count)
                        if entry.is_plural else entry.text)
                return Translation(
                    key=key, text=_interpolate(text, values, key),
                    requested_locale=requested, served_locale=served,
                    source="catalog")
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
            requested_locale=requested, served_locale=self.default,
            source="source-string")

    def translate(self, key: str, locale: str | None = None,
                  *, count: float | None = None, **params: Any) -> str:
        return self.translate_detailed(
            key, locale, count=count, **params).text

    # ── upstream API (craft-libs lib.rs I18n) ───────────────

    def _find(self, keys: list[tuple[str, str]],
              locale: str | None,
              app: str | None) -> tuple[Entry, str] | None:
        """The first of ``keys`` found: the upstream set along
        its chain when present, else Devon's catalogs along
        their chain."""
        if self.upstream is not None:
            return self.upstream.find(keys, app=app, locale=locale)
        for served in fallback_chain(locale or DEFAULT_LOCALE):
            catalog = self.catalogs.catalog(served)
            if catalog is None:
                continue
            entry = catalog.find(keys, app=app)
            if entry is not None:
                return entry, served
        return None

    @staticmethod
    def _entry_text(entry: Entry) -> str:
        """An entry as one string: a plural row's forms,
        "|"-joined, as upstream stores them."""
        return entry.text if not entry.is_plural \
            else "|".join(entry.forms)

    @staticmethod
    def _mark(source: str) -> str:
        """``source`` marked for the pseudo locale, each line on
        its own (``⟦text⟧``)."""
        return "⟦" + source.replace("\n", "⟧\n⟦") + "⟧"

    def _is_pseudo(self, locale: str | None) -> bool:
        # The pseudo locale is a real language upstream
        # (Lang::PSEUDO, code "pseudo"), but Devon's
        # normalize_locale only accepts 2–3 letter languages, so
        # it is recognised on the raw tag, not the normalised one.
        return str(locale or "").strip().lower() == "pseudo"

    def _fill(self, text: str, params: dict[str, Any]) -> str:
        if not params:
            return text
        return _bidi_fmt(text, [(key, str(value))
                                for key, value in params.items()])

    def t(self, source: str, locale: str | None = None,
          app: str | None = None, **params: Any) -> str:
        """``source`` in ``locale``, or ``source`` itself."""
        if self._is_pseudo(locale):
            self._looked_up.add(source)
            return self._mark(source)
        found = self._find([("", source)], locale, app)
        text = self._entry_text(found[0]) \
            if found is not None else source
        return self._fill(text, params)

    def tc(self, source: str, context: str,
           locale: str | None = None, app: str | None = None,
           **params: Any) -> str:
        """``source`` with a disambiguating ``context``: in each
        locale of the chain the context row, then the plain one,
        so a region's plain row beats its parent's context row."""
        if self._is_pseudo(locale):
            self._looked_up.add(source)
            return self._mark(source)
        found = self._find([(context, source), ("", source)],
                           locale, app)
        text = self._entry_text(found[0]) \
            if found is not None else source
        return self._fill(text, params)

    def get(self, source: str, context: str = "",
            app: str | None = None,
            locale: str | None = None) -> str | None:
        """The translation of (``context``, ``source``) along the
        chain, with no fallback to the plain row or the English:
        for strings that must not borrow another meaning's
        wording, or to ask whether a translation exists."""
        if self._is_pseudo(locale):
            self._looked_up.add(source)
            return self._mark(source)
        if self.upstream is not None:
            return self.upstream.get(source, context=context,
                                     app=app, locale=locale)
        found = self._find([(context, source)], locale, app)
        return self._entry_text(found[0]) \
            if found is not None else None

    def tid(self, id: str, label: str, locale: str | None = None,
            app: str | None = None, **params: Any) -> str:
        """A command's label by its ``id`` (``@id`` rows); in each
        locale of the chain the ``@id`` row, then the plain row
        of the English ``label``."""
        if self._is_pseudo(locale):
            self._looked_up.add(label)
            return self._mark(label)
        found = self._find([("@id", id), ("", label)], locale, app)
        text = self._entry_text(found[0]) \
            if found is not None else label
        return self._fill(text, params)

    def _plural(self, source: str, locale: str | None,
                count: float, n_arg: str, english_one: bool,
                params: dict[str, Any]) -> str:
        """A count's wording: the ``@plural`` row for the
        ``one|other`` key along the chain, the form by the served
        locale's rule (category → few → last), ``{n}`` filled
        with ``n_arg``.  No row anywhere: the English forms."""
        if "|" in source:
            one, other = source.split("|", 1)
        else:
            one, other = source, source
        text: str | None = None
        if self.upstream is not None and not self._is_pseudo(locale):
            for candidate in self.upstream.chain(
                    locale or DEFAULT_LOCALE):
                if candidate == DEFAULT_LOCALE:
                    continue
                catalog = self.upstream.catalogs.get(candidate)
                if catalog is None:
                    continue
                entry = catalog.get(source, context="@plural")
                if entry is None or not entry.is_plural:
                    continue
                categories = plural_categories(candidate)
                category = plural_category(candidate, count)
                if category not in categories:
                    category = "few" if "few" in categories \
                        else categories[-1]
                text = entry.forms[categories.index(category)]
                break
        if text is None:
            text = one if english_one else other
        args = [("n", n_arg)]
        args.extend((key, str(value))
                    for key, value in params.items())
        return _bidi_fmt(text, args)

    def trn(self, source: str, locale: str | None = None,
            count: int = 1, **params: Any) -> str:
        """A count's wording, ``{n}`` filled as the formatter
        writes numbers: grouped and in the locale's digits, as a
        left-to-right isolate when the wording has a right-to-left
        letter."""
        text = self._plural(
            source, locale, float(count),
            _bidi_ltr(self._formatter.integer(count)),
            count == 1, params)
        if self._is_pseudo(locale):
            return self._mark(text)
        return text

    def trn_plain(self, source: str, locale: str | None = None,
                  count: int = 1, **params: Any) -> str:
        """``trn`` with ``{n}`` in ``0``–``9`` without grouping
        and no isolate marks, for text that goes into documents,
        file names or anything read back later."""
        text = self._plural(source, locale, float(count),
                            str(count), count == 1, params)
        if self._is_pseudo(locale):
            text = self._mark(text)
        return _strip_isolates(text)

    def trn_decimal(self, source: str, locale: str | None = None,
                    count: float = 1, decimals: int = 2,
                    **params: Any) -> str:
        """``trn`` for a count with ``decimals`` digits after the
        decimal separator.  The plural form follows the digits
        shown, so ``1.0`` takes English ``other``; infinities and
        NaN take ``other`` too."""
        rendered = self._formatter.number(count, decimals)
        plain = f"{float(count):.{decimals}f}"
        english_one = _strip_zeros(plain) == "1"
        text = self._plural(
            source, locale, float(count),
            _bidi_ltr(rendered), english_one, params)
        if self._is_pseudo(locale):
            return self._mark(text)
        return text

    def chain(self, locale: str | None) -> list[str]:
        """The resolution order for ``locale`` (upstream chain
        when an upstream set is present, else the Devon chain)."""
        if self.upstream is not None:
            return self.upstream.chain(locale or DEFAULT_LOCALE)
        return fallback_chain(locale or DEFAULT_LOCALE)

    def code(self, locale: str | None) -> str:
        """The locale's code, lowercase BCP 47 (``pt-br``); an
        unknown locale reads as English."""
        if self._is_pseudo(locale):
            return "pseudo"
        norm = normalize_locale(locale).lower()
        if not norm:
            return DEFAULT_LOCALE
        if self.upstream is not None:
            if norm in self.upstream.catalogs \
                    or norm in ("en", "pseudo"):
                return norm
            return DEFAULT_LOCALE
        return norm

    def from_code(self, code: str) -> str | None:
        """The language with this code (any case): ``en``, an
        offered locale, or ``pseudo``.  ``None`` when no offered
        language fits."""
        if str(code or "").strip().lower() == "pseudo":
            return "pseudo"
        norm = normalize_locale(code).lower()
        if norm == "en":
            return norm
        if self.upstream is not None:
            return norm if norm in self.upstream.catalogs else None
        return norm if norm in self.catalogs.catalogs else None

    def languages(self) -> list[str]:
        """The languages to offer: English first, then the
        offered locales by code.  Never the pseudo locale."""
        if self.upstream is not None:
            offered = sorted(self.upstream.catalogs)
        else:
            offered = sorted(self.catalogs.catalogs)
        return [DEFAULT_LOCALE] + [code for code in offered
                                   if code != DEFAULT_LOCALE]

    def name(self, locale: str | None) -> str:
        """The language's name in itself, for a language picker:
        its ``@id language.name`` row (the app's catalog over
        the shared one), or the code when there is none."""
        norm = normalize_locale(locale).lower()
        if norm == "en":
            return "English"
        if self.upstream is not None:
            found = self.upstream.language_name(norm)
            if found is not None:
                return found
        return norm or DEFAULT_LOCALE

    def rtl(self, locale: str | None) -> bool:
        """Whether the language is written right to left."""
        return _script_direction(
            normalize_locale(locale) or DEFAULT_LOCALE) == _Direction.RTL

    def match_tag(self, tag: str) -> str | None:
        """The language for a system locale tag (``fr_CA.UTF-8``,
        ``zh-Hant-MO``); ``C`` and ``POSIX`` are English.
        ``None`` when no offered language fits."""
        from .upstream import candidates

        found_candidates = candidates(tag)
        if found_candidates and found_candidates[0] in ("c", "posix"):
            return "en"
        for candidate in found_candidates:
            if candidate == "pseudo":
                continue
            found = self.from_code(candidate)
            if found is not None:
                return found
        return None

    def match_tags(self, tags: Any) -> str:
        """The first of ``tags`` (most preferred first) that
        :meth:`match_tag` knows; English if none."""
        for tag in tags:
            found = self.match_tag(tag)
            if found is not None:
                return found
        return DEFAULT_LOCALE

    def system_lang(self) -> str:
        """The language the system asks for (``LANGUAGE`` and
        friends, then ``LC_ALL``, ``LC_MESSAGES``, ``LANG``);
        English if none fits.  Upstream reads once and caches;
        this port reads each call so a long-lived process (and
        its tests) see the environment change."""
        tags: list[str] = []
        language = os.environ.get("LANGUAGE", "")
        tags.extend(part for part in language.split(":") if part)
        for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
            value = os.environ.get(var, "")
            if value:
                tags.append(value)
        return self.match_tags(tags)

    def looked_up(self) -> set[str]:
        """Every source text looked up in the pseudo locale so
        far, for an audit to check against a complete catalog.
        Plurals aren't recorded."""
        return set(self._looked_up)

    def check_all(self) -> list[str]:
        """Every problem in the upstream catalogs (``[]`` without
        an upstream set)."""
        if self.upstream is None:
            return []
        return self.upstream.check_all()

    def set_format(self, formatter: Any) -> None:
        """Write numbers as ``formatter`` does from now on: the
        user's region setting, or Latin digits."""
        self._formatter = formatter

    def reset_format(self) -> None:
        """Write numbers as the plain default does again, undoing
        :meth:`set_format`."""
        self._formatter = _PlainFormat()


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
