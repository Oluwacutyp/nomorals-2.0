"""Plural categories per locale (CLDR cardinals, integer form).

English-shaped plural logic — "add s unless the count is 1" — is
wrong for most of the world: Russian has three integer forms,
Arabic six, Japanese none.  This module carries the category *set*
and the selection rule per locale so catalogs can store one form
per category and the runtime picks the right one for a count.

Scope, stated plainly: these are the CLDR **integer cardinal**
rules.  Locales whose CLDR rules grow extra categories only for
decimal values (Czech ``many``, Lithuanian ``many``) list the
integer categories here; a non-integral count lands in ``other``
for every locale except where the integer rule itself says
otherwise.  A locale with no entry uses the visible default —
categories ``("other",)``, every count ``other`` — which is the
true rule for Japanese, Yoruba, Hausa, Igbo and Pidgin anyway, and
an honest approximation (recorded in PARITY.md) for any locale
not yet given a rule.

Category order follows CLDR: zero, one, two, few, many, other —
catalog ``@plural`` rows store forms in exactly this order.
"""

from __future__ import annotations

from typing import Callable

__all__ = [
    "CATEGORY_ORDER",
    "plural_categories",
    "plural_category",
    "known_locales",
]

CATEGORY_ORDER = ("zero", "one", "two", "few", "many", "other")

Rule = Callable[[float], str]


def _integral(n: float) -> bool:
    return float(n) == int(float(n))


def _one_other(n: float) -> str:
    return "one" if _integral(n) and int(n) == 1 else "other"


def _zero_one_other(n: float) -> str:
    """French/Portuguese/Hindi shape: 0 and 1 share the first form."""
    return "one" if _integral(n) and int(n) in (0, 1) else "other"


def _other(_n: float) -> str:
    return "other"


def _ru(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        return "one"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "few"
    if n % 10 == 0 or n % 10 in (5, 6, 7, 8, 9) or n % 100 in (11, 12, 13, 14):
        return "many"
    return "other"


def _pl(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n == 1:
        return "one"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "few"
    if n % 10 in (0, 1) or n % 10 in (5, 6, 7, 8, 9) or n % 100 in (12, 13, 14):
        return "many"
    return "other"


def _cs(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n == 1:
        return "one"
    if 2 <= n <= 4:
        return "few"
    return "other"


def _ar(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n == 0:
        return "zero"
    if n == 1:
        return "one"
    if n == 2:
        return "two"
    if 3 <= n % 100 <= 10:
        return "few"
    if 11 <= n % 100 <= 99:
        return "many"
    return "other"


def _lv(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n % 10 == 0 or 11 <= n % 100 <= 19:
        return "zero"
    if n % 10 == 1 and n % 100 != 11:
        return "one"
    return "other"


def _lt(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n % 10 == 1 and not 11 <= n % 100 <= 19:
        return "one"
    if 2 <= n % 10 <= 9 and not 11 <= n % 100 <= 19:
        return "few"
    return "other"


def _sl(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n) % 100
    if n == 1:
        return "one"
    if n == 2:
        return "two"
    if n in (3, 4):
        return "few"
    return "other"


def _ro(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n == 1:
        return "one"
    if n == 0 or 1 <= n % 100 <= 19:
        return "few"
    return "other"


def _he(n: float) -> str:
    if not _integral(n):
        return "other"
    n = int(n)
    if n == 1:
        return "one"
    if n == 2:
        return "two"
    return "other"


#: language → (categories in CLDR order, integer cardinal rule).
_RULES: dict[str, tuple[tuple[str, ...], Rule]] = {
    "en": (("one", "other"), _one_other),
    "de": (("one", "other"), _one_other),
    "nl": (("one", "other"), _one_other),
    "sv": (("one", "other"), _one_other),
    "da": (("one", "other"), _one_other),
    "nb": (("one", "other"), _one_other),
    "fi": (("one", "other"), _one_other),
    "et": (("one", "other"), _one_other),
    "af": (("one", "other"), _one_other),
    "es": (("one", "other"), _one_other),
    "it": (("one", "other"), _one_other),
    "el": (("one", "other"), _one_other),
    "sw": (("one", "other"), _one_other),
    "zu": (("one", "other"), _one_other),
    "xh": (("one", "other"), _one_other),
    "fr": (("one", "other"), _zero_one_other),
    "pt": (("one", "other"), _zero_one_other),
    "hi": (("one", "other"), _zero_one_other),
    "ru": (("one", "few", "many", "other"), _ru),
    "uk": (("one", "few", "many", "other"), _ru),
    "be": (("one", "few", "many", "other"), _ru),
    "pl": (("one", "few", "many", "other"), _pl),
    "cs": (("one", "few", "other"), _cs),
    "sk": (("one", "few", "other"), _cs),
    "ar": (("zero", "one", "two", "few", "many", "other"), _ar),
    "lv": (("zero", "one", "other"), _lv),
    "lt": (("one", "few", "other"), _lt),
    "sl": (("one", "two", "few", "other"), _sl),
    "ro": (("one", "few", "other"), _ro),
    "he": (("one", "two", "other"), _he),
    # True other-only rules (CLDR): no category split at any count.
    "ja": (("other",), _other),
    "zh": (("other",), _other),
    "ko": (("other",), _other),
    "vi": (("other",), _other),
    "th": (("other",), _other),
    "id": (("other",), _other),
    "ms": (("other",), _other),
    "tr": (("other",), _other),
    "yo": (("other",), _other),
    "ha": (("other",), _other),
    "ig": (("other",), _other),
    "pcm": (("other",), _other),
    "tw": (("other",), _other),
    "ee": (("other",), _other),
}


def _entry(locale: str) -> tuple[tuple[str, ...], Rule]:
    from .locale import language_of

    return _RULES.get(language_of(locale), (("other",), _other))


def plural_categories(locale: str) -> tuple[str, ...]:
    """The plural categories ``locale`` uses, in CLDR order."""
    return _entry(locale)[0]


def plural_category(locale: str, count: float) -> str:
    """The category ``count`` falls in for ``locale``."""
    return _entry(locale)[1](count)


def known_locales() -> list[str]:
    """Languages with an explicit rule entry (sorted)."""
    return sorted(_RULES)
