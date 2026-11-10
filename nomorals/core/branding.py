"""One brand-mark resolution for every file artifact (core, L1).

The typography law gives presented output the Peacethefirst stamp;
file artifacts carry it as a DESIGNED mark, not a text line: a
translucent diagonal watermark on every PDF page, a fixed
translucent layer behind HTML edition content, a visible line on
the EPUB title page, a small corner mark burned into public-bound
images.  Every one of those renderers resolves its mark HERE, so
the policy semantics can never drift between artifact kinds.

Policy follows the manifest's
:class:`~nomorals.core.longdoc.PresentationHints` and mirrors
``social.chat.typography.resolve_style`` exactly — the constants
are mirrored here because core cannot import social, and a test
pins them equal so the two homes cannot drift:

* ``default`` (and any unknown value — an unknown policy is data,
  not an error) → the house mark;
* ``off`` → no mark at all (``None``);
* ``custom`` → the caller's custom text, falling back to the house
  mark when none is supplied, the same way a group ``stamp_text``
  of "" resolves to the default.

The mark is for PUBLIC-bound artifacts — files and images that
leave to other people.  Work made for the owner himself (his own
song scores, his personal image generations) is not marked; that
classification lives with the producers/senders, this module only
resolves what a mark says once one is owed.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BRAND_NAME", "BRAND_TEXT", "BrandMark", "resolve_brand_mark"]

#: The brand, mirrored from social.chat.typography (whose
#: DEFAULT_STAMP_TEXT is the canonical chat form) — core cannot
#: import social, and a test pins the two equal.
BRAND_NAME = "Peacethefirst"
BRAND_TEXT = "Powered by peacethefirst"


@dataclass(frozen=True)
class BrandMark:
    """A resolved brand mark: the text every renderer draws.

    ``opacity`` is the PDF watermark's ExtGState alpha (and the
    design anchor the other renderers scale from): strong enough
    to read as a deliberate mark on the page, weak enough to stay
    subordinate to the content it sits under.
    """

    text: str
    opacity: float = 0.15


def resolve_brand_mark(brand_policy: str = "",
                       brand_text: str = "") -> BrandMark | None:
    """Resolve a brand policy to the mark it owes (None = no mark).

    ``brand_policy`` follows PresentationHints semantics:
    ``default``/unknown → the house mark; ``off`` → None;
    ``custom`` → ``brand_text`` when supplied, else the house
    mark.  Callers with a manifest pass
    ``manifest.presentation.brand_policy`` (or their own explicit
    policy parameter first, the manifest as fallback — the same
    precedence core.edition has always used).
    """
    policy = (brand_policy or "default").strip().lower()
    if policy == "off":
        return None
    if policy == "custom":
        custom = (brand_text or "").strip()
        if custom:
            return BrandMark(text=custom)
    return BrandMark(text=BRAND_TEXT)
