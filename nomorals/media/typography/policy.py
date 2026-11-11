"""Context font policy — what face fits where, as DATA.

The font census (§6/§7 W3) named the wiring shape: content
classes declare what fits their context in a policy table —
an elegant serif for a quote, a playful display face for a
game, the platform's real UI face for a mock screen — and
every surface resolves through the one registry door
(:mod:`fontlib` via :class:`~.service.TypographyService`).
Changing what a quote "sounds like" typographically is a
data edit to this table, not a code edit in a renderer.

Resolution order inside :func:`pil_font_for`:

1. an explicit ``family=`` — the registry door.  A family
   held nowhere (neither a stocked tier nor the host
   catalogue) raises :class:`FontNotStocked` naming the
   family and every tier searched: the named outcome, never
   a silent slide to an unrelated face.  A family the host
   alone holds still resolves, exactly as the service's own
   ``select`` semantics (F1) define.
2. an owner style pin's family for the caller's
   ``surface_family`` (the /stylepin seam outranks policy,
   census W3) — resolved through the service, which applies
   the pin; an unresolvable pin family is likewise named.
3. the context's policy families, in order — the first one
   the stocked library holds wins.
4. the context's intent through the service's semantic
   selection (host catalogue included) — the taste fallback
   when no policy family is stocked on this host.
5. nothing anywhere → :class:`FontNotStocked` naming the
   context.  Callers whose contract is never-raise keep
   Pillow's bitmap default reachable ONLY behind this raise,
   logged as the registry miss it is — never silent.

Because every policy family below is stocked in the vendored
core subset, steps 1–3 succeed on a bare clone with no pack
stocked and no host fonts at all; the pack (and any
owner-local drop-in) only widens the choice.  A family added
to the registry is selectable by every surface with zero new
plumbing: policy families are looked up by name at render
time, and an explicit ``family=`` needs no policy entry at
all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from . import shaping
from .fontlib import FontNotStocked

_log = get_logger(__name__)

__all__ = [
    "CONTEXT_POLICIES",
    "EMOJI_FAMILIES",
    "ContextFontPolicy",
    "context_policy",
    "emoji_font_path",
    "pil_font_for",
]


@dataclass(frozen=True)
class ContextFontPolicy:
    """One context's taste: stocked families to prefer, in
    order, over a semantic intent; the weight/style the
    context wears unless a caller overrides them."""

    families: tuple[str, ...] = ()
    intent: str = "text"
    weight: int = 400
    italic: bool = False


#: The policy table.  Keys are context names surfaces ask for;
#: values are data.  Every family named here is stocked in the
#: vendored core subset (the bare-clone floor) unless a row
#: says otherwise in its comment.
CONTEXT_POLICIES: dict[str, ContextFontPolicy] = {
    # Quotes wear an elegant serif; the script row is the
    # accent voice (an author line, a signature flourish).
    "quote": ContextFontPolicy(
        families=("Playfair Display", "Lora", "Merriweather",
                  "PT Serif"),
        intent="editorial"),
    "quote_script": ContextFontPolicy(
        families=("Great Vibes", "Dancing Script", "Caveat"),
        intent="handwritten"),
    # Motivation: a confident display sans over the gradient.
    "motivation": ContextFontPolicy(
        families=("Montserrat", "Poppins", "Playfair Display"),
        intent="display"),
    # Posters and maker cards: bold display first.
    "poster": ContextFontPolicy(
        families=("Anton", "Bebas Neue", "Montserrat"),
        intent="display", weight=700),
    "meme": ContextFontPolicy(
        families=("Anton", "Bebas Neue", "Oswald"),
        intent="display", weight=700),
    # Games: playful display for titles, rounded sans for
    # play text, mono for the board's numbers.
    "game": ContextFontPolicy(
        families=("Poppins", "Montserrat"),
        intent="playful"),
    "game_title": ContextFontPolicy(
        families=("Bebas Neue", "Anton", "Poppins"),
        intent="display", weight=700),
    "scoreboard": ContextFontPolicy(
        families=("JetBrains Mono", "Space Mono", "Roboto Mono"),
        intent="scoreboard"),
    # Cards and menus: the house text face, a display title.
    "card": ContextFontPolicy(
        families=("Inter", "Roboto", "Open Sans"),
        intent="text"),
    "card_title": ContextFontPolicy(
        families=("Montserrat", "Inter", "Poppins"),
        intent="display", weight=700),
    # The brand mark ("Powered by peacethefirst"): a
    # guaranteed face instead of a host lottery.
    "brandmark": ContextFontPolicy(
        families=("Montserrat", "Inter", "Poppins"),
        intent="text", weight=700),
    "caption": ContextFontPolicy(
        families=("Inter", "Roboto", "Open Sans"),
        intent="caption"),
    "mono": ContextFontPolicy(
        families=("JetBrains Mono", "Roboto Mono", "Space Mono",
                  "IBM Plex Mono"),
        intent="mono"),
    # Cover art titles.
    "cover": ContextFontPolicy(
        families=("Anton", "Montserrat", "Bebas Neue"),
        intent="display", weight=700),
    # Mock screens wear the platforms' real UI faces where
    # they are stockable: Roboto is the Android face behind
    # WhatsApp / Telegram / Google Messages.  iMessage's San
    # Francisco and Snapchat's Helvetica-class face are not
    # stockable (Apple / proprietary licences), so Inter —
    # the stocked neo-grotesque in the same class — stands in,
    # named here rather than silently substituted.
    "screen": ContextFontPolicy(
        families=("Roboto", "Inter", "Noto Sans"),
        intent="text"),
    "screen_whatsapp": ContextFontPolicy(
        families=("Roboto", "Inter"), intent="text"),
    "screen_telegram": ContextFontPolicy(
        families=("Roboto", "Inter"), intent="text"),
    "screen_sms": ContextFontPolicy(
        families=("Roboto", "Inter"), intent="text"),
    "screen_instagram": ContextFontPolicy(
        families=("Inter", "Roboto"), intent="text"),
    "screen_messenger": ContextFontPolicy(
        families=("Inter", "Roboto"), intent="text"),
    "screen_snapchat": ContextFontPolicy(
        families=("Inter", "Roboto"), intent="text"),
    "screen_imessage": ContextFontPolicy(
        families=("Inter", "Roboto"), intent="text"),
}

_DEFAULT_POLICY = ContextFontPolicy(intent="text")

#: Colour-emoji families, best first: the pack stocks Noto
#: Color Emoji; Windows hosts catalogue Segoe UI Emoji; macOS
#: hosts Apple Color Emoji.  One door (registry, then
#: catalogue) replaces the old hardcoded host-path pair.
EMOJI_FAMILIES = ("Noto Color Emoji", "Segoe UI Emoji",
                  "Apple Color Emoji")


def context_policy(context: Any) -> ContextFontPolicy:
    """The policy row for ``context`` (the plain text row for
    an unknown context)."""
    return CONTEXT_POLICIES.get(str(context or ""),
                                 _DEFAULT_POLICY)


def _load(path: Path, size: int, weight: int | None = None):
    """Load a stocked face the service way; on a VARIABLE
    face, also set the variation axes — the Weight axis to
    the requested weight (clamped to the axis range) and an
    Optical size axis to the render size, other axes at
    their defaults — so a stocked variable file renders the
    weight that was asked for, not silently its default
    instance.  Any variation failure keeps the default
    instance (the service's long-standing behaviour)."""
    font = shaping.load_font(str(path), size)
    if weight:
        try:
            axes = font.get_variation_axes()
        except Exception:  # noqa: BLE001 - not a variable face
            axes = []
        if axes:
            values = []
            for axis in axes:
                name = axis.get("name", b"")
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                name = name.strip().lower()
                lo, hi = int(axis["minimum"]), int(axis["maximum"])
                if name == "weight":
                    values.append(max(lo, min(hi, int(weight))))
                elif "optical" in name:
                    values.append(max(lo, min(hi, int(size))))
                else:
                    values.append(int(axis["default"]))
            try:
                font.set_variation_by_axes(values)
            except Exception as exc:  # noqa: BLE001 - default
                _log.debug("typography policy: variation on "
                           "%s failed: %s", path, exc)
    return font


def pil_font_for(service, context: str, size: int, *,
                 family: str | None = None,
                 intent: str | None = None,
                 weight: int | None = None,
                 italic: bool | None = None,
                 text: str | None = None,
                 surface_family: str | None = None):
    """A Pillow font for ``context`` through the registry
    door (see the module docstring for the resolution order).

    A caller may pass ``intent`` instead of a context to ask
    for the intent's own semantics (the policy families are
    the context's taste and do not override a named intent);
    the intent then drives the service door exactly as a
    direct ``pil_font`` call would, minus the private
    ladders that used to sit underneath it.

    Raises :class:`FontNotStocked` when a requested family —
    or, at the end of the chain, any face at all — cannot be
    resolved: the caller decides what its error seam does
    with a named miss, but the miss is never a silent slide.
    """
    policy = context_policy(context)
    eff_intent = intent or policy.intent
    want_weight = policy.weight if weight is None \
        else int(weight)
    want_italic = policy.italic if italic is None \
        else bool(italic)
    library = service.font_library()
    if family:
        try:
            path = library.resolve(family, weight=want_weight,
                                   italic=want_italic)
        except FontNotStocked as miss:
            if service.catalogue().faces(family=family):
                picked = service.pil_font(
                    eff_intent, size, weight=want_weight,
                    family=family, text=text,
                    surface_family=surface_family)
                if picked is not None:
                    return picked
            raise miss
        return _load(path, size, want_weight)
    # The owner's pin (the surface family's, or the global
    # pin when no family is named — the service's own _pin_for
    # semantics) outranks the policy table, census W3.
    pin = service._pin_for(surface_family)
    if pin is not None and pin.font_family:
        picked = service.pil_font(
            eff_intent, size, weight=want_weight,
            text=text, surface_family=surface_family)
        if picked is not None:
            return picked
        raise FontNotStocked(pin.font_family,
                             library.searched())
    for name in (() if intent else policy.families):
        try:
            path = library.resolve(name, weight=want_weight,
                                   italic=want_italic)
        except FontNotStocked:
            continue
        try:
            return _load(path, size, want_weight)
        except Exception as exc:  # noqa: BLE001 - next family
            _log.debug("typography policy: stocked face %s "
                       "failed to load: %s", path, exc)
            continue
    picked = service.pil_font(eff_intent, size,
                              weight=want_weight, text=text,
                              surface_family=surface_family)
    if picked is not None:
        return picked
    raise FontNotStocked(
        f"any face for context {context!r}",
        library.searched())


def emoji_font_path(service) -> Path | None:
    """The host's colour-emoji face through the one door:
    the stocked registry first (the pack carries Noto Color
    Emoji), then the catalogue's host faces under their real
    family names.  None when no colour-emoji face exists —
    callers keep their text-glyph fallback for that case."""
    library = service.font_library()
    for name in EMOJI_FAMILIES:
        try:
            return library.resolve(name)
        except FontNotStocked:
            continue
    catalogue = service.catalogue()
    for name in EMOJI_FAMILIES:
        hits = catalogue.faces(family=name)
        if hits:
            return Path(hits[0].path)
    return None
