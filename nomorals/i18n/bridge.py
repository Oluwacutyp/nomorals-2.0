"""Bridges to the incumbent language machinery — mapped, not replaced.

Three incumbents predate this package and stay exactly where they
are; this module is the explicit mapping onto them:

* ``family/locales.py`` — region packs (``ng``, ``gh``, …) whose
  ``languages`` lists are the household's served languages.  The
  packs are region data, not message catalogs; :func:`pack_locales`
  reads a pack's languages as i18n locales so a family surface can
  ask the catalogs for the household's first language.
* ``llm/languages.py`` — the translator's language-code table.
  :func:`mt_language_code` maps an i18n locale to the code the
  translation ladder speaks (``pt-BR`` → ``pt``), or ``""`` when
  the ladder does not know the language — a recorded absence, not
  a guess.
* ``llm/translation_quality.py`` — the quality bridge for
  machine-translated catalog contributions, exactly as the plan
  keeps it: a drafted catalog row is scored by back-translation
  against its English source key before it is trusted into a
  catalog file.  :func:`score_contribution` is that scoring path.

The ``llm`` imports are lazy (function-level): the catalog runtime
must stay importable without the provider stack, and the bridge is
only touched by contribution tooling and family surfaces.
"""

from __future__ import annotations

from typing import Any, Callable

from ..family.locales import get_locale_pack
from .locale import language_of, normalize_locale

__all__ = [
    "pack_locales",
    "pack_primary_locale",
    "mt_language_code",
    "score_contribution",
]


def pack_locales(pack_code: str | None) -> list[str]:
    """The family pack's languages as normalized i18n locales."""
    pack = get_locale_pack(pack_code)
    out: list[str] = []
    for lang in pack.get("languages", []):
        norm = normalize_locale(str(lang))
        if norm and norm not in out:
            out.append(norm)
    return out


def pack_primary_locale(pack_code: str | None) -> str:
    """The first language of a family pack (``""`` if it has none)."""
    locales = pack_locales(pack_code)
    return locales[0] if locales else ""


def mt_language_code(locale: str | None) -> str:
    """The translation ladder's code for a locale, or ``""``.

    ``pt-BR`` → ``pt``; ``pcm`` → ``pcm``; a language the ladder's
    table does not carry → ``""`` (the caller records the absence;
    machine translation for that locale is simply not offered).
    """
    language = language_of(locale or "")
    if not language:
        return ""
    from ..llm.languages import normalize_code

    return normalize_code(language) or ""


def score_contribution(source_text: str, translated_text: str, *,
                       target_locale: str,
                       translate_fn: Callable[[str, str, str], str],
                       judge_fn: Callable[[str, str],
                                          tuple[float, str]] | None = None,
                       threshold: float = 0.5) -> Any:
    """Score one machine-translated catalog row before it lands.

    Wraps ``llm.translation_quality.translate_with_quality``: the
    drafted row is back-translated to English and measured against
    the source key (token F1), with the optional judge as the
    second signal.  ``translate_fn`` is injected — the caller's
    translation ladder (or a test double); this function adds no
    provider of its own.  Returns the bridge's
    ``TranslationResult`` (``passed`` is the gate a contribution
    must clear to be written into a catalog).
    """
    from ..llm.translation_quality import translate_with_quality

    target = mt_language_code(target_locale) or language_of(target_locale)

    def _fn(text: str, src: str, dst: str) -> str:
        # The forward leg returns the draft under test verbatim;
        # the back-translation leg uses the caller's real engine.
        if translated_text and src == "en" and dst == target:
            return translated_text
        return translate_fn(text, src, dst)

    return translate_with_quality(
        source_text, source_lang="en", target_lang=target,
        translate_fn=_fn, judge_fn=judge_fn, threshold=threshold)
