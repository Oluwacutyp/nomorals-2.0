"""The shared theme: a small palette every Studio page can map
onto its own tokens, plus optional per-app overrides in that app's
own token names.

A port of the rules in ``craft-extensions`` (``crates/ext-core/src/
palette.rs``, dual MIT/Apache): a theme is data only. The palette
must keep text readable (WCAG contrast, the same 4.5:1 rule
PhotoCraft already tests) so a theme cannot be installed that hides
the interface. Overrides use names only the app understands, so the
app re-checks its own pairs after applying them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = [
    "MAX_THEME_BYTES",
    "Palette",
    "Rgb",
    "ThemeError",
    "ThemeFile",
    "contrast_ratio",
]

MAX_THEME_BYTES = 64 * 1024
SCHEMA = 1
MAX_OVERRIDE_APPS = 32
MAX_TOKENS_PER_APP = 64

#: Minimum contrast: 4.5:1 for text, 3:1 for colours that are
#: shapes or status marks, as WCAG asks.
TEXT_RATIO = 4.5
MARK_RATIO = 3.0


class ThemeError(ValueError):
    """A theme file was refused, with the reason as the message."""


@dataclass(frozen=True)
class Rgb:
    """An opaque colour written as ``#rrggbb``."""

    red: int
    green: int
    blue: int

    @classmethod
    def parse(cls, text: str) -> Rgb | None:
        """Parse ``#rrggbb`` or return None."""
        if not isinstance(text, str) or not text.startswith("#"):
            return None
        digits = text[1:]
        if len(digits) != 6 or not all(
                c in "0123456789abcdefABCDEF" for c in digits):
            return None
        return cls(int(digits[0:2], 16), int(digits[2:4], 16),
                   int(digits[4:6], 16))

    def __str__(self) -> str:
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"

    def luminance(self) -> float:
        """Relative luminance as WCAG defines it, 0.0 to 1.0."""
        def linear(value: int) -> float:
            c = value / 255.0
            if c <= 0.04045:
                return c / 12.92
            return ((c + 0.055) / 1.055) ** 2.4

        return (0.2126 * linear(self.red)
                + 0.7152 * linear(self.green)
                + 0.0722 * linear(self.blue))


def contrast_ratio(a: Rgb, b: Rgb) -> float:
    """WCAG contrast ratio between two colours, 1.0 to 21.0."""
    la, lb = a.luminance(), b.luminance()
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


@dataclass(frozen=True)
class Palette:
    """The ten colours a Studio page is built from."""

    dark: bool
    background: Rgb
    surface: Rgb
    text: Rgb
    muted_text: Rgb
    line: Rgb
    accent: Rgb
    good: Rgb
    warn: Rgb
    bad: Rgb

    def check(self) -> None:
        """Refuse a palette that hides the interface."""
        if self.dark != (self.background.luminance()
                         < self.text.luminance()):
            raise ThemeError(
                "the theme says it is dark or light but its colours "
                "say otherwise")
        rows = [
            (self.text, self.background, TEXT_RATIO,
             "text and background"),
            (self.text, self.surface, TEXT_RATIO,
             "text and panel colours"),
            (self.muted_text, self.background, TEXT_RATIO,
             "secondary text and background"),
            (self.muted_text, self.surface, TEXT_RATIO,
             "secondary text and panel colours"),
            (self.accent, self.background, MARK_RATIO,
             "highlight and background"),
            (self.good, self.background, MARK_RATIO,
             "success colour and background"),
            (self.warn, self.background, MARK_RATIO,
             "warning colour and background"),
            (self.bad, self.background, MARK_RATIO,
             "error colour and background"),
        ]
        for front, back, need, pair in rows:
            if contrast_ratio(front, back) < need:
                raise ThemeError(
                    f"the theme's {pair} are too close in colour "
                    "to read")


def _valid_token_name(name: str) -> bool:
    return (isinstance(name, str) and len(name) <= 32
            and name[:1].islower()
            and all(c.isascii() and (c.islower() or c.isdigit()
                                     or c == "_") for c in name))


@dataclass(frozen=True)
class ThemeFile:
    """A parsed ``theme.json``: palette plus per-app overrides."""

    schema: int
    palette: Palette
    apps: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, data: bytes) -> ThemeFile:
        """Parse and check a theme file, or raise ThemeError."""
        if len(data) > MAX_THEME_BYTES:
            raise ThemeError("the theme file is too large")
        try:
            raw = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ThemeError("the theme file cannot be read") from None
        if not isinstance(raw, dict):
            raise ThemeError("the theme file cannot be read")
        schema = raw.get("schema")
        if schema != SCHEMA:
            raise ThemeError(
                "the theme uses a format version "
                f"({schema}) this app does not know")
        palette_raw = raw.get("palette") or {}
        if not isinstance(palette_raw, dict):
            raise ThemeError("the theme file cannot be read")
        palette = _parse_palette(palette_raw)
        apps_raw = raw.get("apps") or {}
        if not isinstance(apps_raw, dict):
            raise ThemeError("the theme file cannot be read")
        if len(apps_raw) > MAX_OVERRIDE_APPS:
            raise ThemeError("the theme tunes too many apps")
        apps: dict = {}
        for app, tokens in apps_raw.items():
            if not isinstance(tokens, dict):
                raise ThemeError("the theme file cannot be read")
            if len(tokens) > MAX_TOKENS_PER_APP:
                raise ThemeError(
                    "the theme tunes too many colours for one app")
            if not all(_valid_token_name(name) for name in tokens):
                raise ThemeError(
                    "the theme names a colour in a way that is not "
                    "allowed")
            parsed: dict = {}
            for name, value in tokens.items():
                colour = Rgb.parse(str(value))
                if colour is None:
                    raise ThemeError("the theme file cannot be read")
                parsed[name] = colour
            apps[str(app)] = parsed
        palette.check()
        return cls(schema=SCHEMA, palette=palette, apps=apps)

    def overrides_for(self, app: str) -> dict | None:
        """This app's own overrides, or None when it has none."""
        return self.apps.get(app)


def _parse_palette(raw: dict) -> Palette:
    fields = ["dark", "background", "surface", "text", "muted_text",
              "line", "accent", "good", "warn", "bad"]
    values: dict = {}
    for name in fields:
        if name not in raw:
            raise ThemeError("the theme file cannot be read")
        if name == "dark":
            if not isinstance(raw[name], bool):
                raise ThemeError("the theme file cannot be read")
            values[name] = raw[name]
            continue
        colour = Rgb.parse(str(raw[name]))
        if colour is None:
            raise ThemeError("the theme file cannot be read")
        values[name] = colour
    return Palette(**values)
