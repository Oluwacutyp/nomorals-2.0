"""The upstream catalog set: the craft-i18n catalogs as one
lookup object, with the craft-libs ``I18n`` lookup rules.

One set holds every loaded locale's shared rows plus its app
layers.  Lookups walk the upstream chain — only a trailing
region subtag falls back, and only when the parent locale is
loaded; a script never crosses (``sr-latn`` never falls to
``sr``) — then the English source text itself, which is the
fallback for a missing translation, never an empty slot.

A set built from a checkout directory (``CRAFT_I18N_DIR``) or
from the vendored tree replaces that tree wholesale; a named
checkout that names no tree is an error, never a silent
fall-back to the vendored files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .catalog import (
    Catalog,
    dir_from_env,
    load_upstream_tree,
    parse_upstream_catalog,
)
from .locale import DEFAULT_LOCALE, normalize_locale
from .scripts import region_script, usual_script

__all__ = ["UpstreamCatalogSet", "candidates"]


def candidates(tag: str) -> list[str]:
    """The locale tags a system tag could mean, most specific
    first (craft-libs ``lib.rs`` ``candidates``): the tag itself,
    its prefixes, a script implied by its region, ``zh-hans`` /
    ``zh-hant`` for plain ``zh``, and ``pt-br`` as the last
    resort for ``pt``."""
    lowered = tag.strip().lower()
    base = lowered.split(".", 1)[0].split("@", 1)[0].replace("_", "-")
    parts = [p for p in base.split("-") if p]
    if not parts:
        return []
    primary = parts[0]
    modifier = None
    if "@" in lowered:
        after = lowered.split("@", 1)[1]
        if after == "latin":
            modifier = "latn"
        elif after == "cyrillic":
            modifier = "cyrl"
    if modifier is not None:
        parts.insert(1, modifier)
    script = (parts[1] if len(parts) > 1 and len(parts[1]) == 4
              and parts[1].isalpha() and parts[1].isascii() else None)
    if (script is None and primary != "zh" and len(parts) > 1
            and _is_region(parts[1])):
        implied = region_script(primary, parts[1])
        if implied is not None:
            out = ["-".join(parts)]
            with_script = parts[:1] + [implied] + parts[1:]
            out.extend(candidates("-".join(with_script)))
            return out
    out = ["-".join(parts[:n])
           for n in range(len(parts), 0, -1)]
    if script is not None:
        usual = usual_script(primary)
        if usual is not None and usual != script:
            out = [c for c in out
                   if script in c.split("-")]
    if primary == "zh" and not any(
            p in ("hans", "hant") for p in parts):
        hant = any(region_script("zh", p) == "hant"
                   for p in parts[1:])
        insert = "zh-hant" if hant else "zh-hans"
        out.insert(max(0, len(out) - 1), insert)
    if primary == "pt" and "pt-br" not in out:
        out.append("pt-br")
    return out


def _is_region(subtag: str) -> bool:
    return ((len(subtag) == 2 and subtag.isalpha()
             and subtag.isascii())
            or (len(subtag) == 3 and subtag.isdigit()))


class UpstreamCatalogSet:
    """Every loaded upstream locale, by lowercase code."""

    def __init__(self, catalogs: dict[str, Catalog],
                 root: Path | None = None) -> None:
        self.catalogs = catalogs
        self.root = root

    @classmethod
    def from_text(cls, texts: dict[str, str]) -> UpstreamCatalogSet:
        """Build from ``{locale: common.tsv text}`` (no app
        layers, no tree on disk)."""
        return cls({locale: parse_upstream_catalog(text, locale)
                    for locale, text in texts.items()})

    @classmethod
    def from_dir(cls, root: str | Path) -> UpstreamCatalogSet:
        """Build from a catalog tree directory."""
        root = Path(root)
        return cls(load_upstream_tree(root), root=root)

    @classmethod
    def vendored(cls) -> UpstreamCatalogSet:
        """Build from the tree shipped in the repo."""
        from .runtime import I18nError

        root = (Path(__file__).resolve().parent
                / "catalogs" / "upstream")
        if not (root / "locales").is_dir():
            raise I18nError(
                f"vendored catalog tree missing at {root}",
                kind="catalog")
        return cls(load_upstream_tree(root), root=root)

    @classmethod
    def from_env(cls) -> UpstreamCatalogSet:
        """Build from ``CRAFT_I18N_DIR`` when it names a tree,
        else the vendored tree.  ``CRAFT_I18N_REQUIRED=1`` makes
        a missing tree an error instead of an empty set."""
        from .runtime import I18nError

        try:
            named = dir_from_env()
        except ValueError as exc:
            raise I18nError(str(exc), kind="catalog") from exc
        if named is not None:
            return cls.from_dir(named)
        if os.environ.get("CRAFT_I18N_REQUIRED", "").strip() == "1":
            try:
                return cls.vendored()
            except I18nError as exc:
                raise I18nError(
                    "CRAFT_I18N_REQUIRED=1 but no catalog tree "
                    "was found", kind="catalog") from exc
        return cls.vendored()

    def locales(self) -> list[str]:
        """The loaded locale codes, sorted."""
        return sorted(self.catalogs)

    def catalog(self, locale: str) -> Catalog | None:
        return self.catalogs.get(normalize_locale(locale).lower()
                                 or locale.lower())

    def chain(self, locale: str) -> list[str]:
        """The resolution order for ``locale``: itself, then the
        parent locale when the last subtag is a region AND that
        parent is loaded, then English (the source text).  A
        script subtag never falls back."""
        norm = normalize_locale(locale).lower()
        chain: list[str] = []
        if norm:
            chain.append(norm)
            parts = norm.split("-")
            if len(parts) >= 2 and _is_region(parts[-1]):
                parent = "-".join(parts[:-1])
                if parent in self.catalogs:
                    chain.append(parent)
        if DEFAULT_LOCALE not in chain:
            chain.append(DEFAULT_LOCALE)
        return chain

    def get(self, source: str, context: str = "",
            app: str | None = None,
            locale: str | None = None) -> str | None:
        """The translation of (``context``, ``source``) along the
        chain for ``locale`` (default), or ``None``.  No plain
        fallback, no source fallback: for strings that must not
        borrow another meaning's wording, or to ask whether a
        translation exists."""
        for candidate in self.chain(locale or DEFAULT_LOCALE):
            if candidate == DEFAULT_LOCALE:
                continue
            catalog = self.catalogs.get(candidate)
            if catalog is None:
                continue
            entry = catalog.get(source, context=context, app=app)
            if entry is not None:
                # A plural row's translation is its forms,
                # "|"-joined, as upstream stores it.
                return entry.text if not entry.is_plural \
                    else "|".join(entry.forms)
        return None

    def find(self, keys: list[tuple[str, str]],
             app: str | None = None,
             locale: str | None = None
             ) -> tuple[Any, str] | None:
        """The first of ``keys`` (context, source) found along
        the chain, as (entry, served locale).  ``None`` when no
        catalog on the chain carries any of them."""
        for candidate in self.chain(locale or DEFAULT_LOCALE):
            if candidate == DEFAULT_LOCALE:
                continue
            catalog = self.catalogs.get(candidate)
            if catalog is None:
                continue
            entry = catalog.find(keys, app=app)
            if entry is not None:
                return entry, candidate
        return None

    def language_name(self, locale: str) -> str | None:
        """The locale's name in itself (its ``@id language.name``
        row, app layer over shared), or ``None``."""
        catalog = self.catalogs.get(
            normalize_locale(locale).lower())
        if catalog is None:
            return None
        entry = catalog.get("language.name", context="@id")
        return entry.text if entry is not None else None

    def check_all(self) -> list[str]:
        """Every problem in the loaded catalogs, as
        ``<path>: <problem>`` strings.  From a directory, the
        tree check (which reports each row problem with its
        file); from bare texts, the parse issues."""
        if self.root is not None:
            from .check import check_upstream_tree

            return [f"{issue['path']}: {issue['problem']}"
                    for issue in check_upstream_tree(self.root)]
        problems: list[str] = []
        for locale in self.locales():
            catalog = self.catalogs[locale]
            for issue in catalog.issues:
                problems.append(
                    f"locales/{locale}: {issue.problem}")
        return problems
