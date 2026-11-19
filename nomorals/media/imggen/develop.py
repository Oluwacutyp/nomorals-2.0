"""The develop stage — LightCraft's full non-destructive develop
engine as pure, deterministic tone math (the LightCraft port, lane
98: ``crates/develop`` settings schema + ``crates/pipeline`` stage
order + ``crates/engine/src/errors.rs`` typed ErrorKind, reimplemented
in Python from the dual MIT/Apache-2.0 reference).

Two entry points share one engine:

* the legacy 10-slider vocabulary (``develop_image``,
  ``DEVELOP_FIELDS``) — byte-identical behaviour, kept for the
  ImageDocument develop stage, the photo_editor develop op and the
  existing tests;
* the full parametric model (``DevelopSettings`` + ``render_develop``)
  — Light, white balance in Kelvin with named presets, parametric +
  point tone curves, 8-band HSL mixer, colour grading wheels, B&W
  mix, texture/clarity/dehaze, vignette, seeded grain, detail
  (sharpen + noise reduction), optics (distortion/CA/defringe/
  vignetting), geometry (crop/rotate/flips), calibration, HDR with
  an SDR rendition, local-adjustment masks, spot healing, profiles,
  process versions (V1 frozen, V2 current) and a typed
  ``DevelopErrorKind`` — callers decide on the kind, never on the
  message text.

Non-destructive by construction: settings are parametric data
(``DevelopSettings`` round-trips through JSON, unknown fields are
ignored, missing fields take defaults, a damaged ``process`` reads
as V1); the source pixels are never touched — every render is
re-derived from the source plus the settings (``DevelopStack`` keeps
the source bytes and a named edit stack). A render is deterministic:
the same source + settings renders to the same bytes on every
machine, which is what lets the image document promise
byte-identical renders. No neural rung, no hidden state: the rung
label is always ``develop (numpy, deterministic)``.

Control order is the pipeline's stage order, and part of the
contract:

1. geometry — crop/rotate/flips, lens corrections, defringe →
2. scene-linear — white balance, exposure, dehaze, local tone
   (highlights/shadows), texture, clarity, local masks →
3. tone map — contrast / whites / blacks filmic curve on luminance,
   highlight desaturation →
4. colour — vibrance, saturation, colour mixer, colour grading,
   B&W, calibration →
5. display — parametric + point tone curves, vignette, grain.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from . import ImgGenError
from .edit import NUMPY_AVAILABLE, PIL_AVAILABLE

__all__ = [
    "DEVELOP_FIELDS",
    "develop_image",
    "develop_defaults",
]

#: The develop vocabulary, in application order.
DEVELOP_FIELDS = (
    "exposure", "temperature", "tint",
    "brightness", "contrast",
    "shadows", "highlights",
    "saturation", "vibrance",
    "sharpness",
)

_RANGES = {
    "exposure": (-2.0, 2.0),
    **{f: (-1.0, 1.0) for f in DEVELOP_FIELDS if f != "exposure"},
}


def develop_defaults() -> dict:
    """The identity develop settings (every slider at zero)."""
    return {f: 0.0 for f in DEVELOP_FIELDS}


def _validate(settings: dict) -> dict:
    out = develop_defaults()
    for k, v in settings.items():
        if k not in DEVELOP_FIELDS:
            raise ImgGenError(
                f"unknown develop control {k!r}; "
                f"use one of {list(DEVELOP_FIELDS)}")
        lo, hi = _RANGES[k]
        fv = float(v)
        if not lo <= fv <= hi:
            raise ImgGenError(
                f"develop {k} must be in [{lo}, {hi}], got {fv}")
        out[k] = fv
    return out


def develop_image(img, **settings):
    """Apply develop controls to a PIL image. Pure: returns a new
    image, never mutates the input. All-zero settings return a copy
    of the input (bit-identical pixels)."""
    if not PIL_AVAILABLE:
        raise ImgGenError(
            "develop needs Pillow (pip install Pillow)")
    if not NUMPY_AVAILABLE:
        raise ImgGenError(
            "develop needs numpy (pip install numpy)")
    s = _validate(settings)
    from PIL import Image

    if all(v == 0.0 for v in s.values()):
        return img.copy()

    import numpy as np

    x = np.asarray(img.convert("RGB")).astype(np.float64) / 255.0

    # 1. exposure — EV stops are a plain gain in linear-ish space.
    if s["exposure"]:
        x = x * (2.0 ** s["exposure"])

    # 2. white balance — temperature warms (R up, B down), tint
    # shifts the green↔magenta axis.
    if s["temperature"] or s["tint"]:
        t = s["temperature"]
        x[..., 0] *= 1.0 + 0.18 * t
        x[..., 2] *= 1.0 - 0.18 * t
        g = s["tint"]
        x[..., 1] *= 1.0 - 0.14 * g
        x[..., 0] *= 1.0 + 0.05 * g
        x[..., 2] *= 1.0 + 0.05 * g

    # 3. brightness / contrast around middle gray.
    if s["brightness"]:
        x = x + 0.22 * s["brightness"]
    if s["contrast"]:
        c = s["contrast"]
        factor = (1.0 + c) if c >= 0 else 1.0 / (1.0 - c)
        x = (x - 0.5) * factor + 0.5

    # 4. shadows / highlights — luminance-weighted so the lift
    # lands where the control's name says and leaves the opposite
    # end (and the hue) alone.
    if s["shadows"] or s["highlights"]:
        lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1]
               + 0.0722 * x[..., 2])
        if s["shadows"]:
            w = (1.0 - np.clip(lum, 0.0, 1.0)) ** 2
            x = x + (0.30 * s["shadows"] * w)[..., None]
        if s["highlights"]:
            lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1]
                   + 0.0722 * x[..., 2])
            w = np.clip(lum, 0.0, 1.0) ** 2
            x = x + (0.30 * s["highlights"] * w)[..., None]

    # 5. saturation / vibrance — vibrance scales by how unsaturated
    # a pixel already is, so skin-tone-ish muted colors move and
    # already-vivid colors stay put.
    if s["saturation"] or s["vibrance"]:
        lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1]
               + 0.0722 * x[..., 2])[..., None]
        if s["saturation"]:
            x = lum + (x - lum) * (1.0 + s["saturation"])
        if s["vibrance"]:
            sat = (x.max(axis=2, keepdims=True)
                   - x.min(axis=2, keepdims=True))
            gain = 1.0 + s["vibrance"] * (1.0 - np.clip(sat, 0.0, 1.0))
            x = lum + (x - lum) * gain

    out = Image.fromarray(
        np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")

    # 6. sharpness — deterministic unsharp via PIL's enhancer.
    if s["sharpness"]:
        from PIL import ImageEnhance

        out = ImageEnhance.Sharpness(out).enhance(
            1.0 + s["sharpness"])
    return out


# ======================================================================
# Full develop engine — the LightCraft port (lane 98)
# ======================================================================
#
# Attribution: the settings schema, stage order, white-balance
# presets, mixer band centres, process-version rules and the typed
# ErrorKind follow storytold/lightcraft (dual MIT/Apache-2.0),
# crates/develop/src/settings.rs, crates/pipeline/src/lib.rs and
# crates/engine/src/errors.rs. The Python rendering is Devon's own
# numpy implementation of the same documented behaviour — no Rust
# code, no Adobe data, no copied assets.
#
# What is deliberately NOT ported: raw decoding (no raw reader in
# Devon — ``open_develop_source`` refuses raw extensions honestly),
# the lens-profile database (no LCP data vendored), AI masks
# (Subject/Sky/Object need a segmentation model — reported as
# ``model_missing``), lens blur, red-eye auto-detection and
# super-resolution (``Enhance``). Every one of those reports its
# absence with a typed error instead of silently doing nothing.

__all__ = [
    "DEVELOP_FIELDS",
    "develop_image",
    "develop_defaults",
    "DevelopErrorKind",
    "DevelopError",
    "ProcessVersion",
    "WB_PRESETS",
    "WB_MODES",
    "MIXER_BANDS",
    "MIXER_HUES",
    "BUILTIN_PROFILES",
    "BUILTIN_PRESETS",
    "Profile",
    "DevelopSettings",
    "DevelopStack",
    "render_develop",
    "develop_settings_from_legacy",
    "develop_settings_from_params",
    "open_develop_source",
    "RAW_EXTENSIONS",
]


class DevelopErrorKind(StrEnum):
    """Typed failure kinds for the develop engine (LightCraft's
    ``ErrorKind`` rule: code decides on the kind, never on the
    message text — messages are for people and may be reworded)."""
    bad_params = "bad_params"
    bad_file = "bad_file"
    unsupported_format = "unsupported_format"
    missing_profile = "missing_profile"
    model_missing = "model_missing"
    render_failed = "render_failed"
    cancelled = "cancelled"
    other = "other"


class DevelopError(ImgGenError):
    """A develop failure with a machine-readable ``kind``.

    Subclasses ``ImgGenError`` so every existing caller that catches
    the imggen error still catches develop failures; new callers
    match on ``err.kind`` instead of the message.
    """

    def __init__(self, kind, message: str, *, hint: str = "") -> None:
        self.kind = (kind if isinstance(kind, DevelopErrorKind)
                     else DevelopErrorKind(kind))
        text = f"develop [{self.kind.value}]: {message}"
        if hint:
            text += f" — {hint}"
        super().__init__(text)
        self.hint = hint

    @property
    def kind_name(self) -> str:
        return self.kind.value


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(
        value, bool) and math.isfinite(float(value))


def _check_num(value, name: str, lo: float, hi: float) -> float:
    if not _is_number(value):
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"{name} must be a number, got {value!r}")
    fv = float(value)
    if not lo <= fv <= hi:
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"{name} must be in [{lo}, {hi}], got {fv}")
    return fv


def _check_choice(value, name: str, choices) -> str:
    if not isinstance(value, str) or value not in choices:
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"{name} must be one of {sorted(choices)}, got "
            f"{value!r}")
    return value


# ----------------------------------------------------------------------
# Process versions (settings.rs: ProcessVersion / Process)
# ----------------------------------------------------------------------

class ProcessVersion(int):
    """The rendering process a settings dict is interpreted by.

    V1 is frozen (the rendering those older files were made with);
    V2 is current (highlights/shadows keep local contrast via an
    edge-aware base). A change that would alter how existing settings
    render ships as a new process, so a photo keeps its look until
    someone updates it. Reading is lenient, exactly like upstream:
    a whole number is that process; anything else (missing,
    negative, fractional, too large, text, null, a list) reads as
    V1, so a damaged value never costs the rest of the settings.
    """

    V1 = 1
    V2 = 2
    LATEST = 2

    def __new__(cls, value=1):
        return int.__new__(cls, int(value))

    @classmethod
    def from_value(cls, value) -> ProcessVersion:
        if isinstance(value, bool):
            return cls(cls.V1)
        if isinstance(value, int) and 0 <= value <= 2 ** 32 - 1:
            return cls(value)
        if isinstance(value, float) and value.is_integer() \
                and 0.0 <= value <= float(2 ** 32 - 1):
            return cls(int(value))
        return cls(cls.V1)

    @classmethod
    def legacy(cls) -> ProcessVersion:
        return cls(cls.V1)

    def is_legacy(self) -> bool:
        return int(self) == self.V1

    def is_known(self) -> bool:
        return int(self) in (self.V1, self.V2)

    def is_outdated(self) -> bool:
        return int(self) < self.LATEST

    def process(self) -> int:
        """The process this build renders with: the newest it knows
        that isn't newer than this number (a number from a newer
        build renders with LATEST; below V1, with V1)."""
        if int(self) >= self.V2:
            return self.V2
        return self.V1


# The version constants are ProcessVersion instances (not plain
# ints) so callers can ask ``ProcessVersion.V1.is_outdated()``.
ProcessVersion.V1 = ProcessVersion(1)
ProcessVersion.V2 = ProcessVersion(2)
ProcessVersion.LATEST = ProcessVersion(2)


# ----------------------------------------------------------------------
# White balance presets (settings.rs: WbMode::preset)
# ----------------------------------------------------------------------

#: WB modes.
WB_MODES = ("as_shot", "auto", "daylight", "cloudy", "shade",
            "tungsten", "fluorescent", "flash", "custom")

#: Preset temperature/tint (Kelvin, tint) — the CIE daylight series
#: and illuminant A, as LightCraft documents them.
WB_PRESETS = {
    "daylight": (5500.0, 10.0),
    "cloudy": (6500.0, 10.0),
    "shade": (7500.0, 10.0),
    "tungsten": (2850.0, 0.0),
    "fluorescent": (3800.0, 21.0),
    "flash": (5500.0, 0.0),
}

#: Raw extensions Devon cannot decode (no raw reader built).
RAW_EXTENSIONS = frozenset({
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2",
    ".raf", ".pef", ".srw", ".x3f", ".iiq", ".3fr",
})

#: The 8 colour-mixer bands, Lightroom order, and their centre hues.
MIXER_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue",
               "purple", "magenta")
MIXER_HUES = (0.0, 30.0, 60.0, 120.0, 180.0, 225.0, 270.0, 315.0)


# ----------------------------------------------------------------------
# Section dataclasses — one per panel of the settings schema.
# Every field carries its upstream range; from_dict validates.
# ----------------------------------------------------------------------

@dataclass
class Light:
    exposure: float = 0.0      # EV stops, -5..5
    contrast: float = 0.0      # -100..100
    highlights: float = 0.0    # -100..100
    shadows: float = 0.0       # -100..100
    whites: float = 0.0        # -100..100
    blacks: float = 0.0        # -100..100

    _RANGES = {"exposure": (-5.0, 5.0), "contrast": (-100.0, 100.0),
               "highlights": (-100.0, 100.0),
               "shadows": (-100.0, 100.0), "whites": (-100.0, 100.0),
               "blacks": (-100.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> Light:
        return _section_from_dict(cls, raw or {}, "light")


@dataclass
class WhiteBalance:
    mode: str = "as_shot"
    temp: float = 6500.0       # Kelvin, 1500..15000
    tint: float = 0.0          # -150..150

    @classmethod
    def from_dict(cls, raw) -> WhiteBalance:
        raw = raw or {}
        _reject_unknown(raw, {"mode", "temp", "tint"}, "wb")
        mode = raw.get("mode", "as_shot")
        mode = _check_choice(mode, "wb.mode", WB_MODES)
        temp = _check_num(raw.get("temp", 6500.0), "wb.temp",
                          1500.0, 15000.0)
        tint = _check_num(raw.get("tint", 0.0), "wb.tint",
                          -150.0, 150.0)
        if mode in WB_PRESETS and "temp" not in raw \
                and "tint" not in raw:
            temp, tint = WB_PRESETS[mode]
        return cls(mode=mode, temp=temp, tint=tint)


@dataclass
class ToneCurveSettings:
    highlights: float = 0.0
    lights: float = 0.0
    darks: float = 0.0
    shadows: float = 0.0
    split_shadows: float = 25.0
    split_mid: float = 50.0
    split_highlights: float = 75.0
    master: list = field(default_factory=list)
    red: list = field(default_factory=list)
    green: list = field(default_factory=list)
    blue: list = field(default_factory=list)
    refine_saturation: float = 100.0

    _RANGES = {"highlights": (-100.0, 100.0),
               "lights": (-100.0, 100.0), "darks": (-100.0, 100.0),
               "shadows": (-100.0, 100.0),
               "split_shadows": (0.0, 100.0), "split_mid": (0.0, 100.0),
               "split_highlights": (0.0, 100.0),
               "refine_saturation": (0.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> ToneCurveSettings:
        raw = raw or {}
        _reject_unknown(
            raw,
            {"highlights", "lights", "darks", "shadows",
             "split_shadows", "split_mid", "split_highlights",
             "master", "red", "green", "blue", "refine_saturation"},
            "curve")
        out = _section_from_dict(
            cls, {k: v for k, v in raw.items()
                  if k not in ("master", "red", "green", "blue")},
            "curve")
        for name in ("master", "red", "green", "blue"):
            pts = raw.get(name, [])
            if pts is None:
                pts = []
            if not isinstance(pts, list):
                raise DevelopError(
                    DevelopErrorKind.bad_params,
                    f"curve.{name} must be a list of [x, y] points")
            checked = []
            prev_x = None
            for p in pts:
                if (not isinstance(p, (list, tuple)) or len(p) != 2
                        or not _is_number(p[0])
                        or not _is_number(p[1])):
                    raise DevelopError(
                        DevelopErrorKind.bad_params,
                        f"curve.{name} points must be [x, y] numbers")
                x, y = float(p[0]), float(p[1])
                if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                    raise DevelopError(
                        DevelopErrorKind.bad_params,
                        f"curve.{name} points must be in [0, 1]")
                if prev_x is not None and x <= prev_x:
                    raise DevelopError(
                        DevelopErrorKind.bad_params,
                        f"curve.{name} x values must increase")
                prev_x = x
                checked.append([x, y])
            setattr(out, name, checked)
        return out


@dataclass
class ColorSettings:
    vibrance: float = 0.0
    saturation: float = 0.0

    _RANGES = {"vibrance": (-100.0, 100.0),
               "saturation": (-100.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> ColorSettings:
        return _section_from_dict(cls, raw or {}, "color")


@dataclass
class HslBand:
    hue: float = 0.0
    sat: float = 0.0
    lum: float = 0.0


@dataclass
class Mixer:
    red: HslBand = field(default_factory=HslBand)
    orange: HslBand = field(default_factory=HslBand)
    yellow: HslBand = field(default_factory=HslBand)
    green: HslBand = field(default_factory=HslBand)
    aqua: HslBand = field(default_factory=HslBand)
    blue: HslBand = field(default_factory=HslBand)
    purple: HslBand = field(default_factory=HslBand)
    magenta: HslBand = field(default_factory=HslBand)

    @classmethod
    def from_dict(cls, raw) -> Mixer:
        raw = raw or {}
        _reject_unknown(raw, set(MIXER_BANDS), "mixer")
        bands = {}
        for name in MIXER_BANDS:
            band = raw.get(name, {}) or {}
            _reject_unknown(band, {"hue", "sat", "lum"},
                            f"mixer.{name}")
            bands[name] = HslBand(
                hue=_check_num(band.get("hue", 0.0),
                               f"mixer.{name}.hue", -100.0, 100.0),
                sat=_check_num(band.get("sat", 0.0),
                               f"mixer.{name}.sat", -100.0, 100.0),
                lum=_check_num(band.get("lum", 0.0),
                               f"mixer.{name}.lum", -100.0, 100.0))
        return cls(**bands)

    def bands(self):
        return [getattr(self, name) for name in MIXER_BANDS]

    def is_neutral(self) -> bool:
        return all(b.hue == 0.0 and b.sat == 0.0 and b.lum == 0.0
                   for b in self.bands())


@dataclass
class Wheel:
    hue: float = 0.0
    sat: float = 0.0
    lum: float = 0.0


@dataclass
class ColorGrading:
    shadows: Wheel = field(default_factory=Wheel)
    midtones: Wheel = field(default_factory=Wheel)
    highlights: Wheel = field(default_factory=Wheel)
    global_: Wheel = field(default_factory=Wheel)
    blending: float = 50.0
    balance: float = 0.0

    @classmethod
    def from_dict(cls, raw) -> ColorGrading:
        raw = raw or {}
        _reject_unknown(raw, {"shadows", "midtones", "highlights",
                              "global", "blending", "balance"},
                        "grading")
        wheels = {}
        for name in ("shadows", "midtones", "highlights", "global"):
            w = raw.get(name, {}) or {}
            _reject_unknown(w, {"hue", "sat", "lum"},
                            f"grading.{name}")
            wheels[name] = Wheel(
                hue=_check_num(w.get("hue", 0.0),
                               f"grading.{name}.hue", 0.0, 360.0),
                sat=_check_num(w.get("sat", 0.0),
                               f"grading.{name}.sat", 0.0, 100.0),
                lum=_check_num(w.get("lum", 0.0),
                               f"grading.{name}.lum", -100.0, 100.0))
        return cls(
            shadows=wheels["shadows"], midtones=wheels["midtones"],
            highlights=wheels["highlights"],
            global_=wheels["global"],
            blending=_check_num(raw.get("blending", 50.0),
                                "grading.blending", 0.0, 100.0),
            balance=_check_num(raw.get("balance", 0.0),
                               "grading.balance", -100.0, 100.0))

    def to_dict(self) -> dict:
        return {"shadows": asdict(self.shadows),
                "midtones": asdict(self.midtones),
                "highlights": asdict(self.highlights),
                "global": asdict(self.global_),
                "blending": self.blending,
                "balance": self.balance}

    def is_neutral(self) -> bool:
        return all(w.sat == 0.0 and w.lum == 0.0 for w in (
            self.shadows, self.midtones, self.highlights,
            self.global_))


@dataclass
class BwMix:
    red: float = 0.0
    orange: float = 0.0
    yellow: float = 0.0
    green: float = 0.0
    aqua: float = 0.0
    blue: float = 0.0
    purple: float = 0.0
    magenta: float = 0.0

    _RANGES = {name: (-100.0, 100.0) for name in MIXER_BANDS}

    @classmethod
    def from_dict(cls, raw) -> BwMix:
        return _section_from_dict(cls, raw or {}, "bw_mix")

    def bands(self):
        return [getattr(self, name) for name in MIXER_BANDS]


@dataclass
class Effects:
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0

    _RANGES = {"texture": (-100.0, 100.0),
               "clarity": (-100.0, 100.0), "dehaze": (-100.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> Effects:
        return _section_from_dict(cls, raw or {}, "effects")


@dataclass
class Vignette:
    amount: float = 0.0
    midpoint: float = 50.0
    roundness: float = 0.0
    feather: float = 50.0
    highlights: float = 0.0

    _RANGES = {"amount": (-100.0, 100.0),
               "midpoint": (0.0, 100.0), "roundness": (-100.0, 100.0),
               "feather": (0.0, 100.0), "highlights": (0.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> Vignette:
        return _section_from_dict(cls, raw or {}, "vignette")


@dataclass
class Grain:
    amount: float = 0.0
    size: float = 25.0
    roughness: float = 50.0
    seed: int = 0

    @classmethod
    def from_dict(cls, raw) -> Grain:
        raw = raw or {}
        _reject_unknown(raw, {"amount", "size", "roughness", "seed"},
                        "grain")
        seed = raw.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                f"grain.seed must be an integer, got {seed!r}")
        return cls(
            amount=_check_num(raw.get("amount", 0.0),
                              "grain.amount", 0.0, 100.0),
            size=_check_num(raw.get("size", 25.0), "grain.size",
                            0.0, 100.0),
            roughness=_check_num(raw.get("roughness", 50.0),
                                 "grain.roughness", 0.0, 100.0),
            seed=seed)


@dataclass
class Detail:
    sharpen_amount: float = 0.0
    sharpen_radius: float = 1.0
    sharpen_detail: float = 25.0
    sharpen_masking: float = 0.0
    nr_luminance: float = 0.0
    nr_detail: float = 50.0
    nr_contrast: float = 0.0
    nr_color: float = 0.0
    nr_color_detail: float = 50.0
    nr_color_smoothness: float = 50.0

    _RANGES = {"sharpen_amount": (0.0, 150.0),
               "sharpen_radius": (0.5, 3.0),
               "sharpen_detail": (0.0, 100.0),
               "sharpen_masking": (0.0, 100.0),
               "nr_luminance": (0.0, 100.0), "nr_detail": (0.0, 100.0),
               "nr_contrast": (0.0, 100.0), "nr_color": (0.0, 100.0),
               "nr_color_detail": (0.0, 100.0),
               "nr_color_smoothness": (0.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> Detail:
        return _section_from_dict(cls, raw or {}, "detail")


@dataclass
class Optics:
    remove_ca: bool = False
    lens_profile: bool = False
    profile_distortion: float = 100.0
    profile_vignetting: float = 100.0
    defringe_purple_amount: float = 0.0
    defringe_purple_hue_lo: float = 30.0
    defringe_purple_hue_hi: float = 70.0
    defringe_green_amount: float = 0.0
    defringe_green_hue_lo: float = 40.0
    defringe_green_hue_hi: float = 60.0
    distortion: float = 0.0
    vignetting: float = 0.0
    vignetting_midpoint: float = 50.0
    ca_red: float = 0.0
    ca_blue: float = 0.0

    _RANGES = {"profile_distortion": (0.0, 200.0),
               "profile_vignetting": (0.0, 200.0),
               "defringe_purple_amount": (0.0, 100.0),
               "defringe_purple_hue_lo": (0.0, 100.0),
               "defringe_purple_hue_hi": (0.0, 100.0),
               "defringe_green_amount": (0.0, 100.0),
               "defringe_green_hue_lo": (0.0, 100.0),
               "defringe_green_hue_hi": (0.0, 100.0),
               "distortion": (-100.0, 100.0),
               "vignetting": (-100.0, 100.0),
               "vignetting_midpoint": (0.0, 100.0),
               "ca_red": (-100.0, 100.0), "ca_blue": (-100.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> Optics:
        raw = raw or {}
        _reject_unknown(raw, set(cls._RANGES) | {"remove_ca",
                                                 "lens_profile"},
                        "optics")
        out = _section_from_dict(
            cls, {k: v for k, v in raw.items()
                  if k in cls._RANGES}, "optics")
        for flag in ("remove_ca", "lens_profile"):
            if flag in raw:
                if not isinstance(raw[flag], bool):
                    raise DevelopError(
                        DevelopErrorKind.bad_params,
                        f"optics.{flag} must be true/false")
                setattr(out, flag, raw[flag])
        return out


@dataclass
class Geometry:
    upright: str = "off"
    vertical: float = 0.0
    horizontal: float = 0.0
    rotate: float = 0.0
    aspect: float = 0.0
    scale: float = 100.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    crop: dict = field(default_factory=dict)

    _RANGES = {"vertical": (-100.0, 100.0),
               "horizontal": (-100.0, 100.0),
               "rotate": (-180.0, 180.0), "aspect": (-100.0, 100.0),
               "scale": (1.0, 400.0), "offset_x": (-100.0, 100.0),
               "offset_y": (-100.0, 100.0)}
    _UPRIGHT = ("off", "auto", "guided", "level", "vertical",
                "full")

    @classmethod
    def from_dict(cls, raw) -> Geometry:
        raw = raw or {}
        _reject_unknown(raw, set(cls._RANGES) | {"upright", "crop"},
                        "geometry")
        out = _section_from_dict(
            cls, {k: v for k, v in raw.items()
                  if k in cls._RANGES}, "geometry")
        upright = raw.get("upright", "off")
        out.upright = _check_choice(upright, "geometry.upright",
                                    cls._UPRIGHT)
        crop = raw.get("crop", {}) or {}
        if not isinstance(crop, dict):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "geometry.crop must be an object")
        _reject_unknown(crop, {"x", "y", "w", "h", "aspect",
                               "flip_h", "flip_v"}, "geometry.crop")
        clean_crop = {}
        for key in ("x", "y", "w", "h"):
            if key in crop:
                clean_crop[key] = _check_num(
                    crop[key], f"geometry.crop.{key}", 0.0, 1.0)
        if "aspect" in crop:
            aspect = crop["aspect"]
            if aspect is not None:
                if (not isinstance(aspect, (list, tuple))
                        or len(aspect) != 2
                        or not all(_is_number(v) and v > 0
                                   for v in aspect)):
                    raise DevelopError(
                        DevelopErrorKind.bad_params,
                        "geometry.crop.aspect must be [w, h] > 0")
                clean_crop["aspect"] = [float(aspect[0]),
                                        float(aspect[1])]
        for flag in ("flip_h", "flip_v"):
            if flag in crop:
                if not isinstance(crop[flag], bool):
                    raise DevelopError(
                        DevelopErrorKind.bad_params,
                        f"geometry.crop.{flag} must be true/false")
                clean_crop[flag] = crop[flag]
        out.crop = clean_crop
        return out

    def is_identity(self) -> bool:
        return (self.upright == "off" and self.vertical == 0.0
                and self.horizontal == 0.0 and self.rotate == 0.0
                and self.aspect == 0.0 and self.scale == 100.0
                and self.offset_x == 0.0 and self.offset_y == 0.0
                and not self.crop)


@dataclass
class Calibration:
    shadows_tint: float = 0.0
    red_hue: float = 0.0
    red_sat: float = 0.0
    green_hue: float = 0.0
    green_sat: float = 0.0
    blue_hue: float = 0.0
    blue_sat: float = 0.0

    _RANGES = {"shadows_tint": (-100.0, 100.0),
               "red_hue": (-100.0, 100.0), "red_sat": (-100.0, 100.0),
               "green_hue": (-100.0, 100.0),
               "green_sat": (-100.0, 100.0),
               "blue_hue": (-100.0, 100.0),
               "blue_sat": (-100.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> Calibration:
        return _section_from_dict(cls, raw or {}, "calibration")

    def is_neutral(self) -> bool:
        return all(getattr(self, f) == 0.0 for f in self._RANGES)


@dataclass
class Hdr:
    enabled: bool = False
    max_ev: float = 3.0
    sdr_brightness: float = 0.0
    sdr_contrast: float = 0.0
    sdr_highlights: float = 0.0
    sdr_shadows: float = 0.0
    sdr_whites: float = 0.0
    sdr_clarity: float = 0.0

    _RANGES = {"max_ev": (0.0, 5.0),
               "sdr_brightness": (-100.0, 100.0),
               "sdr_contrast": (-100.0, 100.0),
               "sdr_highlights": (-100.0, 100.0),
               "sdr_shadows": (-100.0, 100.0),
               "sdr_whites": (-100.0, 100.0),
               "sdr_clarity": (-100.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> Hdr:
        raw = raw or {}
        _reject_unknown(raw, set(cls._RANGES) | {"enabled"}, "hdr")
        out = _section_from_dict(
            cls, {k: v for k, v in raw.items()
                  if k in cls._RANGES}, "hdr")
        if "enabled" in raw:
            if not isinstance(raw["enabled"], bool):
                raise DevelopError(
                    DevelopErrorKind.bad_params,
                    "hdr.enabled must be true/false")
            out.enabled = raw["enabled"]
        return out

    def is_default(self) -> bool:
        return self == Hdr()

    def peak(self) -> float:
        """Peak linear value of an HDR render relative to SDR
        white (1.0 when HDR is off)."""
        if not self.enabled:
            return 1.0
        ev = self.max_ev if math.isfinite(self.max_ev) else 3.0
        return 2.0 ** max(0.0, min(5.0, ev))


@dataclass
class LocalAdjustments:
    temp: float = 0.0
    tint: float = 0.0
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    hue: float = 0.0
    saturation: float = 0.0
    sharpness: float = 0.0
    noise: float = 0.0
    moire: float = 0.0
    defringe: float = 0.0
    color_hue: float = 0.0
    color_sat: float = 0.0
    amount: float = 100.0

    _RANGES = {"temp": (-100.0, 100.0), "tint": (-100.0, 100.0),
               "exposure": (-5.0, 5.0), "contrast": (-100.0, 100.0),
               "highlights": (-100.0, 100.0),
               "shadows": (-100.0, 100.0), "whites": (-100.0, 100.0),
               "blacks": (-100.0, 100.0), "texture": (-100.0, 100.0),
               "clarity": (-100.0, 100.0), "dehaze": (-100.0, 100.0),
               "hue": (-100.0, 100.0), "saturation": (-100.0, 100.0),
               "sharpness": (0.0, 150.0), "noise": (-100.0, 100.0),
               "moire": (0.0, 100.0), "defringe": (0.0, 100.0),
               "color_hue": (0.0, 360.0), "color_sat": (0.0, 100.0),
               "amount": (0.0, 100.0)}

    @classmethod
    def from_dict(cls, raw) -> LocalAdjustments:
        return _section_from_dict(cls, raw or {}, "adjust")

    def is_neutral(self) -> bool:
        default = LocalAdjustments()
        return asdict(self) == asdict(default)


#: Mask shape kinds the engine can evaluate. AI kinds (subject,
#: sky, background, object, prompt, people, landscape) need a
#: segmentation model Devon does not ship — rendering one raises
#: DevelopError(kind=model_missing), never a silent pass-through.
MASK_SHAPES = ("brush", "linear", "radial", "color_range",
               "luminance", "luminance_range", "depth_range",
               "subject", "sky", "background", "object", "prompt",
               "people", "landscape")
AI_MASK_SHAPES = ("subject", "sky", "background", "object",
                  "prompt", "people", "landscape")
LOCAL_MASK_SHAPES = ("brush", "linear", "radial", "luminance",
                     "luminance_range")


@dataclass
class DevelopMask:
    shape: dict = field(default_factory=dict)
    adjust: LocalAdjustments = field(default_factory=LocalAdjustments)
    name: str = ""
    visible: bool = True
    invert: bool = False
    refine: float = 0.0

    @classmethod
    def from_dict(cls, raw) -> DevelopMask:
        if not isinstance(raw, dict):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "a mask must be an object")
        _reject_unknown(raw, {"shape", "adjust", "name", "visible",
                              "invert", "refine"}, "mask")
        shape = raw.get("shape", {}) or {}
        if not isinstance(shape, dict):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "mask.shape must be an object")
        kind = shape.get("kind", "")
        kind = _check_choice(kind, "mask.shape.kind", MASK_SHAPES)
        _validate_mask_shape(kind, shape)
        adjust = LocalAdjustments.from_dict(raw.get("adjust", {}))
        name = raw.get("name", "")
        if not isinstance(name, str):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "mask.name must be a string")
        for flag in ("visible", "invert"):
            if flag in raw and not isinstance(raw[flag], bool):
                raise DevelopError(
                    DevelopErrorKind.bad_params,
                    f"mask.{flag} must be true/false")
        refine = _check_num(raw.get("refine", 0.0), "mask.refine",
                            0.0, 100.0)
        return cls(shape=dict(shape), adjust=adjust, name=name,
                   visible=raw.get("visible", True),
                   invert=raw.get("invert", False), refine=refine)

    def to_dict(self) -> dict:
        return {"shape": dict(self.shape),
                "adjust": asdict(self.adjust), "name": self.name,
                "visible": self.visible, "invert": self.invert,
                "refine": self.refine}


def _validate_mask_shape(kind: str, shape: dict) -> None:
    if kind == "linear":
        _reject_unknown(shape, {"kind", "start", "end"}, "mask.shape")
        for key in ("start", "end"):
            _check_point(shape.get(key), f"mask.shape.{key}")
    elif kind == "radial":
        _reject_unknown(shape, {"kind", "center", "rx", "ry",
                                "angle", "feather", "invert"},
                        "mask.shape")
        _check_point(shape.get("center"), "mask.shape.center")
        for key in ("rx", "ry"):
            _check_num(shape.get(key, 0.2), f"mask.shape.{key}",
                       0.0, 2.0)
        _check_num(shape.get("angle", 0.0), "mask.shape.angle",
                   -180.0, 180.0)
        _check_num(shape.get("feather", 50.0), "mask.shape.feather",
                   0.0, 100.0)
    elif kind in ("luminance", "luminance_range"):
        _reject_unknown(shape, {"kind", "lo", "hi", "lo_feather",
                                "hi_feather"}, "mask.shape")
        for key in ("lo", "hi", "lo_feather", "hi_feather"):
            _check_num(shape.get(key, 0.0), f"mask.shape.{key}",
                       0.0, 1.0)
    elif kind == "brush":
        _reject_unknown(shape, {"kind", "strokes"}, "mask.shape")
        strokes = shape.get("strokes", [])
        if not isinstance(strokes, list):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "mask.shape.strokes must be a list")
        for stroke in strokes:
            if not isinstance(stroke, dict):
                raise DevelopError(
                    DevelopErrorKind.bad_params,
                    "a brush stroke must be an object")
            _reject_unknown(stroke, {"points", "size", "feather",
                                     "flow", "density", "erase"},
                            "mask.shape.strokes")
            points = stroke.get("points", [])
            if not isinstance(points, list):
                raise DevelopError(
                    DevelopErrorKind.bad_params,
                    "brush points must be a list")
            for p in points:
                _check_point(p, "brush point")
    elif kind == "color_range":
        raise DevelopError(
            DevelopErrorKind.bad_params,
            "mask.shape color_range needs sampled colours — "
            "pass samples as [[r, g, b], ...] in OkLab; the "
            "sampler that picks them from the image is not built",
            hint="use a luminance_range mask instead")
    elif kind == "depth_range":
        raise DevelopError(
            DevelopErrorKind.bad_params,
            "mask.shape depth_range needs a depth map — the "
            "source carries none",
            hint="use a luminance_range mask instead")
    # AI shapes validate here; the model check happens at render.


def _check_point(value, name: str) -> tuple:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not _is_number(value[0]) or not _is_number(value[1])
            or not 0.0 <= float(value[0]) <= 1.0
            or not 0.0 <= float(value[1]) <= 1.0):
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"{name} must be [x, y] in [0, 1]")
    return (float(value[0]), float(value[1]))


#: Spot modes.
SPOT_MODES = ("remove", "heal", "clone", "fill")


@dataclass
class Spot:
    mode: str = "remove"
    points: list = field(default_factory=list)
    size: float = 0.02
    feather: float = 50.0
    opacity: float = 100.0
    source_offset: tuple | None = None

    @classmethod
    def from_dict(cls, raw) -> Spot:
        if not isinstance(raw, dict):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "a spot must be an object")
        _reject_unknown(raw, {"mode", "points", "size", "feather",
                              "opacity", "source_offset"}, "spot")
        mode = _check_choice(raw.get("mode", "remove"),
                             "spot.mode", SPOT_MODES)
        points = raw.get("points", [])
        if not isinstance(points, list):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "spot.points must be a list")
        checked = [_check_point(p, "spot point") for p in points]
        offset = raw.get("source_offset")
        if offset is not None:
            # A shift RELATIVE to the spot point ([dx, dy] in
            # [-1, 1]): clone copies from point + offset, so an
            # offset of [-0.5, -0.5] samples half an image up-left
            # of the spot.
            if (not isinstance(offset, (list, tuple))
                    or len(offset) != 2
                    or not all(_is_number(v) for v in offset)
                    or not all(-1.0 <= float(v) <= 1.0
                               for v in offset)):
                raise DevelopError(
                    DevelopErrorKind.bad_params,
                    "spot.source_offset must be [dx, dy] in "
                    "[-1, 1] — a shift relative to the spot point")
            offset = (float(offset[0]), float(offset[1]))
        return cls(
            mode=mode, points=checked,
            size=_check_num(raw.get("size", 0.02), "spot.size",
                            0.0, 1.0),
            feather=_check_num(raw.get("feather", 50.0),
                               "spot.feather", 0.0, 100.0),
            opacity=_check_num(raw.get("opacity", 100.0),
                               "spot.opacity", 0.0, 100.0),
            source_offset=offset)

    def to_dict(self) -> dict:
        out = {"mode": self.mode, "points": [list(p)
                                             for p in self.points],
               "size": self.size, "feather": self.feather,
               "opacity": self.opacity}
        if self.source_offset is not None:
            out["source_offset"] = list(self.source_offset)
        return out


def _reject_unknown(raw: dict, allowed, where: str) -> None:
    """Lenient read, upstream's serde rule with one guard: unknown
    keys riding along with recognised controls are ignored (a file
    from a newer build keeps working); a level carrying ONLY
    unknown keys is rejected, because the caller asked for nothing
    the engine understands — that is almost always a typo."""
    unknown = sorted(set(raw) - set(allowed))
    known = set(raw) & set(allowed)
    if unknown and not known:
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"unknown develop control(s) in {where}: {unknown}; "
            f"use {sorted(allowed)}")


def _section_from_dict(cls, raw, where: str):
    if not isinstance(raw, dict):
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"{where} must be an object")
    ranges = getattr(cls, "_RANGES", {})
    _reject_unknown(raw, set(ranges), where)
    kwargs = {}
    for name, (lo, hi) in ranges.items():
        if name in raw:
            kwargs[name] = _check_num(raw[name], f"{where}.{name}",
                                      lo, hi)
    return cls(**kwargs)


# ----------------------------------------------------------------------
# Profiles + presets
# ----------------------------------------------------------------------

@dataclass
class Profile:
    """A creative profile reference: ``id`` names the profile,
    ``amount`` scales it (0 = off, 100 = full, 200 = double)."""
    id: str = "lc.color"
    amount: float = 100.0


#: Built-in creative profiles (id → section deltas at amount 100).
#: ``lc.color`` is the identity; ``lc.mono`` switches treatment.
BUILTIN_PROFILES = {
    "lc.color": {},
    "lc.vivid": {"color": {"saturation": 20.0, "vibrance": 15.0},
                 "light": {"contrast": 10.0}},
    "lc.mono": {"treatment": "bw"},
    "lc.warm": {"wb": {"mode": "custom", "temp": 7300.0,
                       "tint": 8.0},
                "color": {"saturation": 5.0}},
    "lc.cool": {"wb": {"mode": "custom", "temp": 5700.0,
                       "tint": -5.0},
                "color": {"saturation": 5.0}},
}

#: Built-in named presets (partial settings, applied at an amount).
BUILTIN_PRESETS = {
    "vivid": {"profile": {"id": "lc.vivid", "amount": 100.0}},
    "mono": {"treatment": "bw"},
    "warm": {"wb": {"mode": "custom", "temp": 7500.0, "tint": 10.0},
             "color": {"saturation": 10.0}},
    "cool": {"wb": {"mode": "custom", "temp": 5500.0, "tint": -8.0},
             "color": {"saturation": 10.0}},
    "fade": {"light": {"contrast": -20.0, "blacks": 20.0},
             "color": {"saturation": -15.0},
             "grain": {"amount": 15.0, "seed": 11}},
}


# ----------------------------------------------------------------------
# DevelopSettings — the whole schema
# ----------------------------------------------------------------------

@dataclass
class DevelopSettings:
    version: int = 1
    process: ProcessVersion = field(
        default_factory=lambda: ProcessVersion(ProcessVersion.LATEST))
    profile_id: str = "lc.color"
    profile_amount: float = 100.0
    treatment: str = "color"
    wb: WhiteBalance = field(default_factory=WhiteBalance)
    light: Light = field(default_factory=Light)
    hdr: Hdr = field(default_factory=Hdr)
    curve: ToneCurveSettings = field(default_factory=ToneCurveSettings)
    color: ColorSettings = field(default_factory=ColorSettings)
    mixer: Mixer = field(default_factory=Mixer)
    grading: ColorGrading = field(default_factory=ColorGrading)
    bw_mix: BwMix = field(default_factory=BwMix)
    effects: Effects = field(default_factory=Effects)
    vignette: Vignette = field(default_factory=Vignette)
    grain: Grain = field(default_factory=Grain)
    detail: Detail = field(default_factory=Detail)
    optics: Optics = field(default_factory=Optics)
    geometry: Geometry = field(default_factory=Geometry)
    calibration: Calibration = field(default_factory=Calibration)
    masks: list = field(default_factory=list)
    spots: list = field(default_factory=list)
    disabled_sections: list = field(default_factory=list)

    _SECTIONS = ("wb", "light", "hdr", "curve", "color", "mixer",
                 "grading", "bw_mix", "effects", "vignette", "grain",
                 "detail", "optics", "geometry", "calibration")
    _TOP = _SECTIONS + ("version", "process", "profile", "treatment",
                        "masks", "spots", "disabled_sections")

    @classmethod
    def from_dict(cls, raw) -> DevelopSettings:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "develop settings must be an object")
        _reject_unknown(raw, set(cls._TOP), "settings")
        version = raw.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                f"settings.version must be an integer, got "
                f"{version!r}")
        process = ProcessVersion.from_value(raw.get("process"))
        profile = raw.get("profile", {}) or {}
        if not isinstance(profile, dict):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "settings.profile must be an object")
        _reject_unknown(profile, {"id", "amount", "camera_local_tone"},
                        "profile")
        profile_id = profile.get("id", "lc.color")
        if not isinstance(profile_id, str):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "profile.id must be a string")
        profile_amount = _check_num(
            profile.get("amount", 100.0), "profile.amount", 0.0, 200.0)
        treatment = _check_choice(raw.get("treatment", "color"),
                                  "treatment", ("color", "bw"))
        section_cls = {
            "wb": WhiteBalance, "light": Light, "hdr": Hdr,
            "curve": ToneCurveSettings, "color": ColorSettings,
            "mixer": Mixer, "grading": ColorGrading, "bw_mix": BwMix,
            "effects": Effects, "vignette": Vignette, "grain": Grain,
            "detail": Detail, "optics": Optics,
            "geometry": Geometry, "calibration": Calibration,
        }
        kwargs = {"version": version, "process": process,
                  "profile_id": profile_id,
                  "profile_amount": profile_amount,
                  "treatment": treatment}
        for name, klass in section_cls.items():
            kwargs[name] = klass.from_dict(raw.get(name, {}))
        masks = raw.get("masks", []) or []
        if not isinstance(masks, list):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "settings.masks must be a list")
        kwargs["masks"] = [DevelopMask.from_dict(m) for m in masks]
        spots = raw.get("spots", []) or []
        if not isinstance(spots, list):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "settings.spots must be a list")
        kwargs["spots"] = [Spot.from_dict(s) for s in spots]
        disabled = raw.get("disabled_sections", []) or []
        if (not isinstance(disabled, list)
                or not all(isinstance(v, str) for v in disabled)):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "settings.disabled_sections must be a list of names")
        unknown_sections = sorted(set(disabled) - set(cls._SECTIONS))
        if unknown_sections:
            raise DevelopError(
                DevelopErrorKind.bad_params,
                f"unknown disabled section(s): {unknown_sections}")
        kwargs["disabled_sections"] = list(disabled)
        return cls(**kwargs)

    @property
    def profile(self) -> Profile:
        """The profile as one object (``settings.profile.id``)."""
        return Profile(id=self.profile_id,
                       amount=self.profile_amount)

    def to_dict(self) -> dict:
        out = {"version": self.version}
        if not self.process.is_legacy():
            out["process"] = int(self.process)
        out["profile"] = {"id": self.profile_id,
                          "amount": self.profile_amount}
        out["treatment"] = self.treatment
        for name in self._SECTIONS:
            value = getattr(self, name)
            if name == "grading":
                out[name] = value.to_dict()
            elif name == "wb":
                out[name] = {"mode": value.mode, "temp": value.temp,
                             "tint": value.tint}
            else:
                out[name] = asdict(value)
        if self.masks:
            out["masks"] = [m.to_dict() for m in self.masks]
        if self.spots:
            out["spots"] = [s.to_dict() for s in self.spots]
        if self.disabled_sections:
            out["disabled_sections"] = list(self.disabled_sections)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def hash64(self) -> int:
        """Stable hash of these settings (cache key, like the
        pipeline's stage keys)."""
        digest = hashlib.sha256(self.to_json().encode("utf-8"))
        return int.from_bytes(digest.digest()[:8], "big")

    def is_identity(self) -> bool:
        """True when these settings render exactly like the
        defaults of the same process (same process, identity
        profile at full amount, no masks, no spots)."""
        return self == DevelopSettings(process=self.process) \
            and self.profile_id in ("lc.color",) \
            and self.profile_amount in (100.0,) \
            and not self.masks and not self.spots

    def effective(self) -> DevelopSettings:
        """These settings with disabled sections (their panel's eye
        switch) reset to defaults — upstream's ``effective()``."""
        if not self.disabled_sections:
            return self
        data = self.to_dict()
        defaults = DevelopSettings()
        for name in self.disabled_sections:
            data[name] = asdict(getattr(defaults, name)) \
                if name != "grading" \
                else defaults.grading.to_dict()
        return DevelopSettings.from_dict(data)

    def sdr_rendition(self) -> DevelopSettings:
        """The settings an SDR render uses (upstream's
        ``sdr_rendition``): unchanged when HDR is off, otherwise HDR
        off and the SDR offsets applied to the matching sliders."""
        if not self.hdr.enabled:
            return self
        h = self.hdr
        data = self.to_dict()
        data["hdr"] = asdict(Hdr())
        light = data["light"]
        light["exposure"] = max(
            -5.0, min(5.0, light["exposure"]
                      + h.sdr_brightness / 100.0))
        for key, off in (("contrast", h.sdr_contrast),
                         ("highlights", h.sdr_highlights),
                         ("shadows", h.sdr_shadows),
                         ("whites", h.sdr_whites)):
            light[key] = max(-100.0, min(100.0, light[key] + off))
        data["effects"]["clarity"] = max(
            -100.0, min(100.0,
                        data["effects"]["clarity"] + h.sdr_clarity))
        out = DevelopSettings.from_dict(data)
        return out

    def before_settings(self) -> DevelopSettings:
        """The 'before' of a before/after pair (upstream's
        ``before_settings``): defaults, keeping the crop, the
        orientation and the rendering process so framing matches."""
        out = DevelopSettings(process=self.process)
        out.geometry.crop = dict(self.geometry.crop)
        return out

    def apply_preset(self, name: str, amount: float = 100.0
                     ) -> DevelopSettings:
        """A named built-in preset at ``amount`` percent (100 =
        full). Returns new settings; this one is unchanged."""
        if name not in BUILTIN_PRESETS:
            raise KeyError(f"unknown develop preset {name!r}; use "
                           f"{sorted(BUILTIN_PRESETS)}")
        amount = _check_num(amount, "preset amount", 0.0, 200.0)
        data = self.to_dict()
        patch = json.loads(json.dumps(BUILTIN_PRESETS[name]))
        _deep_merge_scaled(data, patch, amount / 100.0)
        return DevelopSettings.from_dict(data)

    def render(self, img, **_kw):
        """Render ``img`` (a PIL image) with these settings."""
        return render_develop(img, self)


def _deep_merge_scaled(base: dict, patch: dict, scale: float) -> None:
    """Merge ``patch`` into ``base`` with numeric leaves scaled by
    ``scale`` (preset amount); dicts recurse, anything else (mode
    names, treatment, profile ids) replaces wholesale when the
    amount is non-zero."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(
                base.get(key), dict):
            _deep_merge_scaled(base[key], value, scale)
        elif _is_number(value) and _is_number(base.get(key)):
            base[key] = float(base[key]) + float(value) * scale
        elif scale:
            base[key] = value


def develop_settings_from_legacy(**sliders) -> DevelopSettings:
    """The 10 legacy sliders as a full ``DevelopSettings`` (an
    approximation, documented as such: the legacy model has a
    ``brightness`` slider upstream's schema lacks, so it folds into
    exposure at half weight, and its temperature/tint are relative
    −1..1, mapped onto the Kelvin scale around as-shot 6500 K)."""
    legacy = _validate(sliders)
    data: dict = {"light": {}, "wb": {}, "color": {},
                  "detail": {}}
    if legacy["exposure"]:
        data["light"]["exposure"] = legacy["exposure"]
    if legacy["brightness"]:
        data["light"]["exposure"] = data["light"].get(
            "exposure", 0.0) + legacy["brightness"] * 0.5
    if legacy["contrast"]:
        data["light"]["contrast"] = legacy["contrast"] * 100.0
    if legacy["shadows"]:
        data["light"]["shadows"] = legacy["shadows"] * 100.0
    if legacy["highlights"]:
        data["light"]["highlights"] = legacy["highlights"] * 100.0
    if legacy["temperature"] or legacy["tint"]:
        data["wb"]["mode"] = "custom"
        data["wb"]["temp"] = 6500.0 + legacy["temperature"] * 1500.0
        data["wb"]["tint"] = legacy["tint"] * 100.0
    if legacy["saturation"]:
        data["color"]["saturation"] = legacy["saturation"] * 100.0
    if legacy["vibrance"]:
        data["color"]["vibrance"] = legacy["vibrance"] * 100.0
    if legacy["sharpness"]:
        data["detail"]["sharpen_amount"] = legacy["sharpness"] * 100.0
    return DevelopSettings.from_dict(data)


#: Dotted keys the new engine understands in a flat params dict:
#: ``section.field`` and ``section.band.field`` (mixer / bw_mix /
#: grading wheels), plus ``settings`` for a whole settings dict.
def _dotted_to_settings(params: dict) -> DevelopSettings:
    data: dict = {}
    legacy: dict = {}
    for key, value in params.items():
        if key == "settings":
            continue
        if "." not in key:
            legacy[key] = value
            continue
        parts = key.split(".")
        if parts[0] not in DevelopSettings._SECTIONS \
                and parts[0] not in ("profile",):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                f"unknown develop control {key!r}")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise DevelopError(
                    DevelopErrorKind.bad_params,
                    f"develop control {key!r} conflicts with an "
                    f"earlier value")
        node[parts[-1]] = value
    base = DevelopSettings.from_dict(params["settings"]) \
        if "settings" in params \
        else DevelopSettings()
    if legacy:
        base = _merge_settings(base, develop_settings_from_legacy(
            **legacy))
    if data:
        merged = _merge_settings(base, DevelopSettings.from_dict(data))
        return merged
    return base


def _merge_settings(base: DevelopSettings,
                    patch: DevelopSettings) -> DevelopSettings:
    """Field-level merge: every field of ``patch`` that differs from
    its default overrides the same field of ``base``."""
    defaults = DevelopSettings()
    base_data = base.to_dict()
    patch_data = patch.to_dict()
    default_data = defaults.to_dict()
    for section in DevelopSettings._SECTIONS:
        if section not in patch_data:
            continue
        base_sec = base_data.setdefault(section, {})
        patch_sec = patch_data[section]
        default_sec = default_data.get(section, {})
        _merge_section(base_sec, patch_sec, default_sec)
    if patch_data.get("treatment") != default_data.get("treatment"):
        base_data["treatment"] = patch_data["treatment"]
    if patch_data.get("profile", {}).get("id") != "lc.color" \
            or patch_data.get("profile", {}).get("amount") != 100.0:
        base_data["profile"] = patch_data["profile"]
    if patch.masks:
        base_data["masks"] = patch_data["masks"]
    if patch.spots:
        base_data["spots"] = patch_data["spots"]
    if patch.disabled_sections:
        base_data["disabled_sections"] = patch_data["disabled_sections"]
    if int(patch.process) != int(ProcessVersion.V1):
        base_data["process"] = int(patch.process)
    return DevelopSettings.from_dict(base_data)


def _merge_section(base: dict, patch: dict, default: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(
                base.get(key), dict) and isinstance(
                default.get(key), dict):
            _merge_section(base[key], value, default[key])
        elif value != default.get(key):
            base[key] = value


def develop_settings_from_params(params: dict) -> DevelopSettings:
    """A flat develop params dict (legacy sliders, dotted new
    controls, and/or a whole ``settings`` dict) as one
    ``DevelopSettings``. Unknown keys raise ``DevelopError``
    (``bad_params``) naming the key."""
    if not isinstance(params, dict):
        raise DevelopError(
            DevelopErrorKind.bad_params,
            "develop params must be an object")
    if any("." in k or k == "settings" for k in params):
        return _dotted_to_settings(params)
    unknown = sorted(set(params) - set(DEVELOP_FIELDS))
    if unknown:
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"develop op got unknown controls {unknown}; use "
            f"{list(DEVELOP_FIELDS)} or dotted controls like "
            f"'light.exposure'")
    return develop_settings_from_legacy(**params)


def open_develop_source(source):
    """Open a develop source: a path, bytes, or a file-like object.

    Raw extensions (``.dng``/``.cr2``/…) raise
    ``unsupported_format`` — Devon has no raw reader, and the failure
    says so instead of producing a black image. Corrupt or missing
    data raises ``bad_file``. Anything else Pillow can open loads.
    """
    from PIL import Image

    if isinstance(source, (bytes, bytearray)):
        stream = io.BytesIO(bytes(source))
        path = None
    elif hasattr(source, "read"):
        stream = source
        path = None
    elif isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        ext = os.path.splitext(path)[1].lower()
        if ext in RAW_EXTENSIONS:
            raise DevelopError(
                DevelopErrorKind.unsupported_format,
                f"raw file {path!r} needs a raw decoder — Devon "
                f"has none built",
                hint="convert it to JPEG/PNG/TIFF first")
        if not os.path.exists(path):
            raise DevelopError(
                DevelopErrorKind.bad_file,
                f"develop source {path!r} does not exist")
        stream = path
    else:
        raise DevelopError(
            DevelopErrorKind.bad_file,
            f"develop source must be a path or image bytes, got "
            f"{type(source).__name__}")
    try:
        img = Image.open(stream)
        img.load()
    except DevelopError:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow errors vary
        raise DevelopError(
            DevelopErrorKind.bad_file,
            f"develop source could not be read as an image: {exc}"
        ) from exc
    if img.mode not in ("RGB", "RGBA", "L", "LA", "P", "CMYK", "I",
                        "I;16"):
        raise DevelopError(
            DevelopErrorKind.unsupported_format,
            f"image mode {img.mode!r} is not a develop source")
    return img.convert("RGB")


# ----------------------------------------------------------------------
# Rendering — crates/pipeline stage order, in numpy on float64.
#
#   1. geometry — crop/rotate/flips, distortion, optics, defringe
#   2. scene-linear — white balance, exposure, dehaze, local tone,
#      texture, clarity, local masks
#   3. tone map — contrast / whites / blacks filmic curve on
#      luminance, highlight desaturation
#   4. colour — vibrance, saturation, mixer, grading, B&W,
#      calibration
#   5. display — parametric + point tone curves, detail (sharpen /
#      noise reduction), vignette, grain
#
# The input image is never mutated; every stage returns a new array.
# All-zero settings render pixel-identical (each stage is the
# identity at its defaults, so the round-trip is exact).
# ----------------------------------------------------------------------

_LUM = (0.2126, 0.7152, 0.0722)  # Rec. 709 luminance weights


def _smoothstep(edge0: float, edge1: float, x):
    import numpy as np

    if edge1 <= edge0:
        return np.where(x > edge0, 1.0, 0.0).astype(np.float64)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance(x):
    return (x[..., 0] * _LUM[0] + x[..., 1] * _LUM[1]
            + x[..., 2] * _LUM[2])


def _box_blur(a, radius: int):
    """Separable box blur (edge-padded). Radius 0 is the identity."""
    import numpy as np

    if radius <= 0:
        return a.copy()
    k = 2 * radius + 1

    def blur_axis(arr, axis):
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (radius, radius)
        p = np.pad(arr, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float64)
        c = np.concatenate(
            [np.zeros_like(np.take(c, [0], axis=axis)),
             c], axis=axis)
        size = p.shape[axis]
        idx_hi = np.arange(k, size + 1)
        idx_lo = idx_hi - k
        hi = np.take(c, idx_hi, axis=axis)
        lo = np.take(c, idx_lo, axis=axis)
        return (hi - lo) / float(k)

    out = blur_axis(a, 0)
    if out.ndim == 3:
        out = blur_axis(out, 1)
    return out


def _median3(a):
    """3x3 median (edge-padded) of a 2-D array."""
    import numpy as np

    p = np.pad(a, 1, mode="edge")
    stack = [p[i:i + a.shape[0], j:j + a.shape[1]]
             for i in range(3) for j in range(3)]
    return np.median(np.stack(stack, axis=-1), axis=-1)


def _bilinear(x, xs, ys):
    """Sample array ``x`` (h, w[, c]) at float source coords."""
    import numpy as np

    h, w = x.shape[0], x.shape[1]
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    if x.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = (x[y0c, x0c] * (1 - fx) + x[y0c, x1c] * fx)
    bot = (x[y1c, x0c] * (1 - fx) + x[y1c, x1c] * fx)
    return top * (1 - fy) + bot * fy


def _rgb_to_hsv(x):
    import numpy as np

    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    c = mx - mn
    h = np.zeros_like(mx)
    safe = np.where(c > 0, c, 1.0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    is_r = (mx == r) & (c > 0)
    is_g = (mx == g) & (c > 0) & ~is_r
    is_b = (mx == b) & (c > 0) & ~is_r & ~is_g
    h = np.where(is_r, ((g - b) / safe) % 6.0, h)
    h = np.where(is_g, (b - r) / safe + 2.0, h)
    h = np.where(is_b, (r - g) / safe + 4.0, h)
    s = np.where(mx > 0, c / np.where(mx > 0, mx, 1.0), 0.0)
    return h * 60.0, s, mx


def _hsv_to_rgb(h, s, v):
    import numpy as np

    h = np.mod(h, 360.0) / 60.0
    c = v * s
    x = c * (1.0 - np.abs(h % 2.0 - 1.0))
    m = v - c
    z = np.zeros_like(h)
    conds = [h < 1, (h >= 1) & (h < 2), (h >= 2) & (h < 3),
             (h >= 3) & (h < 4), (h >= 4) & (h < 5), h >= 5]
    r = np.select(conds, [c, x, z, z, x, c])
    g = np.select(conds, [x, c, c, x, z, z])
    b = np.select(conds, [z, z, x, c, c, x])
    return np.stack([r + m, g + m, b + m], axis=-1)


def _kelvin_to_rgb(kelvin: float):
    """Blackbody colour (Tanner Helland's approximation), 0..255."""
    t = max(1000.0, min(40000.0, kelvin)) / 100.0
    r = 255.0 if t <= 66.0 else 329.698727446 * (
        (t - 60.0) ** -0.1332047592)
    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    return (max(0.0, min(255.0, r)), max(0.0, min(255.0, g)),
            max(0.0, min(255.0, b)))


def _wb_gains(temp: float, tint: float):
    """Per-channel gains for a white-balance setting, normalised
    at 6500 K: a higher temperature warms the image (R up, B down),
    matching the Lightroom slider direction and the legacy
    ``develop_image`` temperature control."""
    ref = _kelvin_to_rgb(6500.0)
    cur = _kelvin_to_rgb(temp)
    gains = [ref[i] / cur[i] for i in range(3)]
    g = gains[1] or 1.0
    gains = [v / g for v in gains]
    t = tint / 150.0
    gains[0] *= 1.0 + 0.15 * t
    gains[1] *= 1.0 - 0.15 * t
    gains[2] *= 1.0 + 0.15 * t
    return gains


def _apply_wb(x, temp: float, tint: float):
    if temp == 6500.0 and tint == 0.0:
        return x
    gains = _wb_gains(temp, tint)
    return x * gains


def _apply_exposure(x, ev: float):
    if ev == 0.0:
        return x
    return x * (2.0 ** ev)


def _tone_windows(light: Light, lum, process: int):
    """Highlights/shadows windows. V1 weights by the pixel's own
    luminance; V2 weights by an edge-aware base (3x3 median of
    log-luminance), so a lift never bleeds across a hard edge and
    a small dark detail inside a bright field keeps its contrast."""
    import numpy as np

    if process >= 2:
        base = _median3(np.log(lum + 1e-3))
        base_lum = np.exp(base) - 1e-3
        base_lum = np.clip(base_lum, 0.0, 1.0)
    else:
        base_lum = lum
    hi_w = _smoothstep(0.5, 1.0, base_lum)
    sh_w = 1.0 - _smoothstep(0.0, 0.5, base_lum)
    return hi_w, sh_w


def _apply_light(x, light: Light, process: int):
    import numpy as np

    if (light.highlights == 0.0 and light.shadows == 0.0):
        return x
    lum = _luminance(x)
    hi_w, sh_w = _tone_windows(light, lum, process)
    new_lum = lum.copy()
    if light.highlights:
        v = light.highlights / 100.0
        if v > 0:
            new_lum = new_lum + v * hi_w * (1.0 - new_lum) * 0.5
        else:
            new_lum = new_lum + v * hi_w * new_lum * 0.5
    if light.shadows:
        v = light.shadows / 100.0
        if v > 0:
            new_lum = new_lum + v * sh_w * (1.0 - new_lum) * 0.5
        else:
            new_lum = new_lum + v * sh_w * new_lum * 0.5
    new_lum = np.clip(new_lum, 0.0, 1.0)
    ratio = np.where(lum > 1e-6, new_lum / np.where(lum > 1e-6,
                                                    lum, 1.0),
                     1.0)
    return x * ratio[..., None]


def _apply_dehaze(x, amount: float):
    """Dark-channel dehaze: transmission from the per-pixel dark
    channel against a per-channel airlight (95th percentile)."""
    import numpy as np

    if amount == 0.0:
        return x
    dark = x.min(axis=-1)
    air = np.array([np.percentile(x[..., c], 95.0)
                    for c in range(3)])
    air_dark = max(float(air.min()), 1e-3)
    omega = (amount / 100.0) * 0.95
    t = 1.0 - omega * np.clip(dark / air_dark, 0.0, 1.0)
    t = np.clip(t, 0.1, 2.0)
    return (x - air) / t[..., None] + air


def _local_contrast(x, amount: float, radius: int, gain: float):
    if amount == 0.0:
        return x
    blur = _box_blur(x, radius)
    detail = x - blur
    return x + (amount / 100.0) * gain * detail


def _apply_tone_map(x, light: Light):
    """Stage 3: contrast / whites / blacks as an S-curve in the
    gamma-2.2 domain on luminance, with a shoulder above 0.95 and
    highlight desaturation — ``ToneMap::display`` semantics."""
    import numpy as np

    if (light.contrast == 0.0 and light.whites == 0.0
            and light.blacks == 0.0):
        return x
    lum = _luminance(x)
    g = np.power(np.clip(lum, 0.0, 1.0), 1.0 / 2.2)
    if light.contrast:
        c = light.contrast / 100.0
        g = np.clip((g - 0.5) * (1.0 + c) + 0.5, 0.0, 1.0)
    new_lum = np.power(g, 2.2)
    if light.whites:
        v = light.whites / 100.0
        w = _smoothstep(0.7, 1.0, new_lum)
        new_lum = (new_lum + v * w * (1.0 - new_lum) if v > 0
                   else new_lum + v * w * new_lum)
    if light.blacks:
        v = light.blacks / 100.0
        w = 1.0 - _smoothstep(0.0, 0.3, new_lum)
        new_lum = (new_lum + v * w * (1.0 - new_lum) if v > 0
                   else new_lum + v * w * new_lum)
    new_lum = np.clip(new_lum, 0.0, 1.0)
    # Shoulder: roll off above 0.95 instead of hard-clipping.
    over = new_lum > 0.95
    new_lum = np.where(over, 0.95 + (new_lum - 0.95) * 0.5,
                       new_lum)
    ratio = np.where(lum > 1e-6,
                     new_lum / np.where(lum > 1e-6, lum, 1.0), 1.0)
    out = x * ratio[..., None]
    # Highlight desaturation: the brightest tones lose colour.
    lum2 = _luminance(out)
    f = np.clip((lum2 - 0.8) / 0.2, 0.0, 1.0) * 0.5
    out = out * (1.0 - f[..., None]) + lum2[..., None] * f[..., None]
    return out


def _apply_saturation(x, color: ColorSettings):
    import numpy as np

    if color.vibrance == 0.0 and color.saturation == 0.0:
        return x
    lum = _luminance(x)[..., None]
    if color.saturation:
        x = lum + (x - lum) * (1.0 + color.saturation / 100.0)
    if color.vibrance:
        mx = x.max(axis=-1)
        mn = x.min(axis=-1)
        sat = np.where(mx > 1e-6, (mx - mn) / np.where(mx > 1e-6,
                                                       mx, 1.0),
                       0.0)
        factor = 1.0 + (color.vibrance / 100.0) * (1.0 - sat)
        x = lum + (x - lum) * factor[..., None]
    return x


def _band_weight(hue_deg, centre: float):
    """Triangular mixer weight: 1 at the band centre, 0 sixty
    degrees away (half the distance to a non-adjacent band)."""
    import numpy as np

    d = np.abs((hue_deg - centre + 180.0) % 360.0 - 180.0)
    return np.clip(1.0 - d / 60.0, 0.0, 1.0)


def _apply_mixer(x, mixer: Mixer):
    import numpy as np

    if mixer.is_neutral():
        return x
    h, s, v = _rgb_to_hsv(np.clip(x, 0.0, 1.0))
    dh = np.zeros_like(h)
    ds = np.zeros_like(h)
    dv = np.zeros_like(h)
    for band, centre in zip(mixer.bands(), MIXER_HUES, strict=True):
        if band.hue == 0.0 and band.sat == 0.0 and band.lum == 0.0:
            continue
        w = _band_weight(h, centre)
        dh = dh + w * (band.hue / 100.0) * 30.0
        ds = ds + w * (band.sat / 100.0)
        dv = dv + w * (band.lum / 100.0)
    h2 = h + dh
    s2 = np.clip(s * (1.0 + ds), 0.0, 1.0)
    v2 = np.clip(v * (1.0 + dv), 0.0, 1.0)
    return _hsv_to_rgb(h2, s2, v2)


def _apply_grading(x, grading: ColorGrading):
    import numpy as np

    if grading.is_neutral():
        return x
    lum = _luminance(x)
    lz = np.clip(lum - grading.balance * 0.002, 0.0, 1.0)
    # Blending widens the zone transitions: at 0 the zones meet
    # at the midtone; at 100 they overlap generously. It never
    # pulls a zone's weight towards the mean, so a shadows tint
    # stays out of the highlights.
    span = 0.25 + (grading.blending / 100.0) * 0.25
    sh_w = 1.0 - _smoothstep(0.5 - span, 0.5 + span, lz)
    mid_w = 1.0 - np.abs(2.0 * lz - 1.0)
    hi_w = _smoothstep(0.5 - span, 0.5 + span, lz)
    weights = {"shadows": sh_w, "midtones": mid_w,
               "highlights": hi_w}
    wheels = {"shadows": grading.shadows,
              "midtones": grading.midtones,
              "highlights": grading.highlights,
              "global": grading.global_}
    out = x.copy()
    for name, wheel in wheels.items():
        if wheel.sat == 0.0 and wheel.lum == 0.0:
            continue
        w = weights[name] if name != "global" \
            else np.ones_like(lum)
        if wheel.sat:
            tint = _hsv_to_rgb(
                np.full_like(lum, wheel.hue),
                np.full_like(lum, wheel.sat / 100.0),
                np.full_like(lum, 0.5))
            strength = w * (wheel.sat / 100.0) * 0.5
            delta = (tint - 0.5) * 2.0 * strength[..., None]
            out = out + delta
        if wheel.lum:
            out = out + (wheel.lum / 100.0) * 0.5 * w[..., None]
    return out


def _apply_bw(x, bw_mix: BwMix):
    """B&W treatment: per-band mix of the luminance, so a red
    filter brightens reds and darkens blues."""
    import numpy as np

    lum = _luminance(x)
    if all(v == 0.0 for v in bw_mix.bands()):
        gray = lum
    else:
        h, _, _ = _rgb_to_hsv(np.clip(x, 0.0, 1.0))
        factor = np.ones_like(lum)
        for value, centre in zip(bw_mix.bands(), MIXER_HUES,
                                 strict=True):
            if value == 0.0:
                continue
            factor = factor + _band_weight(h, centre) * (value / 100.0)
        gray = np.clip(lum * factor, 0.0, 1.0)
    return np.stack([gray, gray, gray], axis=-1)


def _apply_calibration(x, calibration: Calibration):
    import numpy as np

    if calibration.is_neutral():
        return x
    out = x.copy()
    if calibration.shadows_tint:
        lum = _luminance(x)
        w = 1.0 - _smoothstep(0.25, 0.75, lum)
        delta = (calibration.shadows_tint / 100.0) * 0.1 * w
        out[..., 0] = out[..., 0] + delta
        out[..., 2] = out[..., 2] - delta
    primaries = (("red", 0.0, calibration.red_hue,
                  calibration.red_sat),
                 ("green", 120.0, calibration.green_hue,
                  calibration.green_sat),
                 ("blue", 240.0, calibration.blue_hue,
                  calibration.blue_sat))
    if any(p[2] != 0.0 or p[3] != 0.0 for p in primaries):
        h, s, v = _rgb_to_hsv(np.clip(out, 0.0, 1.0))
        dh = np.zeros_like(h)
        ds = np.zeros_like(h)
        for _, centre, hue_adj, sat_adj in primaries:
            if hue_adj == 0.0 and sat_adj == 0.0:
                continue
            w = _band_weight(h, centre)
            dh = dh + w * (hue_adj / 100.0) * 30.0
            ds = ds + w * (sat_adj / 100.0)
        out = _hsv_to_rgb(h + dh, np.clip(s * (1.0 + ds), 0.0, 1.0),
                          v)
    return out


def _curve_lut(points):
    """A 256-entry LUT for a point curve ([[x, y], ...] in
    [0, 1]); empty means identity."""
    import numpy as np

    xs = np.arange(256, dtype=np.float64) / 255.0
    if not points:
        return xs
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    if px[0] > 0.0:
        px = [0.0] + px
        py = [py[0]] + py
    if px[-1] < 1.0:
        px = px + [1.0]
        py = py + [py[-1]]
    return np.interp(xs, px, py)


def _apply_curves(x, curve: ToneCurveSettings):
    import numpy as np

    if (curve.highlights == 0.0 and curve.lights == 0.0
            and curve.darks == 0.0 and curve.shadows == 0.0
            and not curve.master and not curve.red
            and not curve.green and not curve.blue
            and curve.refine_saturation == 100.0):
        return x
    lum = _luminance(x)
    # Parametric: four smooth bumps over the split regions.
    s1 = curve.split_shadows / 100.0
    s2 = curve.split_mid / 100.0
    s3 = curve.split_highlights / 100.0
    out = np.clip(x, 0.0, 1.0)
    if (curve.highlights or curve.lights or curve.darks
            or curve.shadows):
        y = lum.copy()
        bumps = (
            (curve.shadows, 0.0, s1, s1 / 2.0 if s1 else 0.125),
            (curve.darks, s1, s2, (s1 + s2) / 2.0),
            (curve.lights, s2, s3, (s2 + s3) / 2.0),
            (curve.highlights, s3, 1.0,
             (s3 + 1.0) / 2.0 if s3 < 1.0 else 0.875),
        )
        for amount, lo, hi, centre in bumps:
            if amount == 0.0:
                continue
            width = max(hi - lo, 0.05)
            w = np.clip(1.0 - np.abs(lum - centre) / width, 0.0, 1.0)
            y = y + (amount / 100.0) * 0.2 * w
        y = np.clip(y, 0.0, 1.0)
        ratio = np.where(lum > 1e-6,
                         y / np.where(lum > 1e-6, lum, 1.0), 1.0)
        out = out * ratio[..., None]
    # Point curves: master first, then per-channel.
    idx = (np.clip(out, 0.0, 1.0) * 255.0).round().astype(int)
    if curve.master:
        lut = _curve_lut(curve.master)
        out = lut[idx]
    for channel, points in ((0, curve.red), (1, curve.green),
                            (2, curve.blue)):
        if points:
            lut = _curve_lut(points)
            out[..., channel] = lut[idx[..., channel]]
    if curve.refine_saturation != 100.0:
        lum2 = _luminance(out)[..., None]
        out = lum2 + (out - lum2) * (curve.refine_saturation / 100.0)
    return out


def _apply_detail(x, detail: Detail):
    import numpy as np

    if (detail.sharpen_amount == 0.0 and detail.nr_luminance == 0.0
            and detail.nr_color == 0.0):
        return x
    out = x
    if detail.nr_luminance:
        radius = 1
        blur = _box_blur(out, radius)
        a = (detail.nr_luminance / 100.0) * 0.85
        if detail.nr_contrast:
            edge = np.abs(out - blur).max(axis=-1)
            protect = np.clip(edge * 4.0, 0.0, 1.0)
            a_eff = a * (1.0 - (detail.nr_contrast / 100.0) * protect)
            out = out * (1.0 - a_eff[..., None]) \
                + blur * a_eff[..., None]
        else:
            out = out * (1.0 - a) + blur * a
    if detail.nr_color:
        lum = _luminance(out)[..., None]
        chroma = out - lum
        blur_c = _box_blur(chroma, 1)
        a = (detail.nr_color / 100.0) * 0.85
        out = lum + chroma * (1.0 - a) + blur_c * a
    if detail.sharpen_amount:
        radius = max(1, min(3, int(round(detail.sharpen_radius))))
        blur = _box_blur(out, radius)
        delta = out - blur
        gain = (detail.sharpen_amount / 100.0) \
            * (0.5 + detail.sharpen_detail / 100.0)
        if detail.sharpen_masking:
            edge = np.abs(delta).max(axis=-1)
            emask = np.clip(edge * 4.0, 0.0, 1.0)
            blend = (1.0 - detail.sharpen_masking / 100.0) \
                + (detail.sharpen_masking / 100.0) * emask
            out = out + gain * delta * blend[..., None]
        else:
            out = out + gain * delta
    return out


def _apply_vignette(x, vignette: Vignette):
    import numpy as np

    if vignette.amount == 0.0:
        return x
    h, w = x.shape[0], x.shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    # Roundness 0 follows the frame's aspect; 100 is a circle.
    rx = max((w - 1) / 2.0, 1.0)
    ry = max((h - 1) / 2.0, 1.0)
    r = vignette.roundness / 100.0
    rx = rx * (1.0 - r) + min(rx, ry) * r
    ry = ry * (1.0 - r) + min(rx, ry) * r
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    mid = vignette.midpoint / 100.0
    span = max(vignette.feather / 100.0, 0.05)
    f = np.clip((d - mid) / span, 0.0, 1.0)
    lum = _luminance(x)
    protect = 1.0 - (vignette.highlights / 100.0) * lum
    factor = 1.0 + (vignette.amount / 100.0) * f * protect
    return x * factor[..., None]


def _apply_grain(x, grain: Grain):
    import numpy as np

    if grain.amount == 0.0:
        return x
    rng = np.random.default_rng(grain.seed)
    h, w = x.shape[0], x.shape[1]
    fine = rng.standard_normal((h, w, 1))
    rough = grain.roughness / 100.0
    if rough > 0.0:
        coarse = _box_blur(fine, 1)
        noise = fine * (1.0 - rough) + coarse * rough
    else:
        noise = fine
    sigma = (grain.amount / 100.0) * 0.06 \
        * (0.5 + grain.size / 50.0)
    return x + noise * sigma


# ----------------------------------------------------------------------
# Geometry (stage 1, PIL) + optics remaps (numpy)
# ----------------------------------------------------------------------

def _apply_geometry_pil(img, geometry: Geometry):
    from PIL import Image

    out = img
    crop = geometry.crop
    if crop:
        w, h = out.size
        x0 = int(round(crop.get("x", 0.0) * w))
        y0 = int(round(crop.get("y", 0.0) * h))
        cw = int(round(crop.get("w", 1.0) * w))
        ch = int(round(crop.get("h", 1.0) * h))
        aspect = crop.get("aspect")
        if aspect:
            target = aspect[0] / aspect[1]
            cur = cw / max(ch, 1)
            if cur > target:
                cw = int(round(ch * target))
            else:
                ch = int(round(cw / target))
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        cw = max(1, min(cw, w - x0))
        ch = max(1, min(ch, h - y0))
        out = out.crop((x0, y0, x0 + cw, y0 + ch))
    if crop.get("flip_h"):
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if crop.get("flip_v"):
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    if geometry.rotate:
        # Lightroom rotate is clockwise-positive; PIL is not.
        out = out.rotate(-geometry.rotate, expand=True,
                         resample=Image.BICUBIC)
    if geometry.scale != 100.0:
        w, h = out.size
        nw = max(1, int(round(w * geometry.scale / 100.0)))
        nh = max(1, int(round(h * geometry.scale / 100.0)))
        out = out.resize((nw, nh), Image.BICUBIC)
    if geometry.vertical or geometry.horizontal:
        # A shear stands in for true perspective keystone
        # correction (no homography solver in this rung); the
        # direction matches: positive vertical tips the top away.
        w, h = out.size
        shy = geometry.vertical / 100.0 * 0.2
        shx = geometry.horizontal / 100.0 * 0.2
        out = out.transform(
            (w, h), Image.AFFINE, (1, shx, -shx * h / 2.0,
                                   shy, 1, -shy * w / 2.0),
            resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    if geometry.offset_x or geometry.offset_y:
        w, h = out.size
        dx = geometry.offset_x / 100.0 * w
        dy = geometry.offset_y / 100.0 * h
        out = out.transform(
            (w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
            resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    return out


def _apply_distortion(x, amount: float):
    """Radial distortion: sample the source at
    ``p * (1 + k r^2)``. The centre pixel never moves."""
    import numpy as np

    if amount == 0.0:
        return x
    h, w = x.shape[0], x.shape[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / max((w - 1) / 2.0, 1.0)
    ny = (yy - cy) / max((h - 1) / 2.0, 1.0)
    r2 = nx ** 2 + ny ** 2
    k = (amount / 100.0) * 0.3
    factor = 1.0 + k * r2
    sx = cx + (xx - cx) * factor
    sy = cy + (yy - cy) * factor
    return _bilinear(x, sx, sy)


def _apply_optics_vignetting(x, optics: Optics):
    import numpy as np

    amount = optics.vignetting * (optics.profile_vignetting / 100.0) \
        if optics.lens_profile else optics.vignetting
    if amount == 0.0:
        return x
    h, w = x.shape[0], x.shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rx = max((w - 1) / 2.0, 1.0)
    ry = max((h - 1) / 2.0, 1.0)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    mid = optics.vignetting_midpoint / 100.0
    f = np.clip((d - mid) / max(1.0 - mid, 0.05), 0.0, 1.0)
    # Positive vignetting brightens the corners (it corrects the
    # lens's own falloff); negative darkens them.
    return x * (1.0 + (amount / 100.0) * f)[..., None]


def _apply_ca(x, optics: Optics):
    """Lateral chromatic aberration: shift R/B radially by the
    calibrated amounts (scaled by the profile amount when a lens
    profile is on)."""
    import numpy as np

    red = optics.ca_red
    blue = optics.ca_blue
    if optics.lens_profile:
        red = red * (optics.profile_distortion / 100.0)
        blue = blue * (optics.profile_distortion / 100.0)
    if red == 0.0 and blue == 0.0:
        return x
    h, w = x.shape[0], x.shape[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    out = x.copy()
    for channel, shift in ((0, red), (2, blue)):
        if shift == 0.0:
            continue
        k = (shift / 100.0) * 0.02
        nx = (xx - cx) / max((w - 1) / 2.0, 1.0)
        ny = (yy - cy) / max((h - 1) / 2.0, 1.0)
        r2 = nx ** 2 + ny ** 2
        factor = 1.0 + k * r2
        sx = cx + (xx - cx) * factor
        sy = cy + (yy - cy) * factor
        out[..., channel] = _bilinear(x[..., channel], sx, sy)
    return out


def _apply_defringe(x, optics: Optics):
    """Desaturate purple/green fringe hues by the defringe
    amounts (no-op when both are zero)."""
    import numpy as np

    if (optics.defringe_purple_amount == 0.0
            and optics.defringe_green_amount == 0.0):
        return x
    h, s, v = _rgb_to_hsv(np.clip(x, 0.0, 1.0))
    lum = _luminance(x)[..., None]
    out = x.copy()
    for amount, lo, hi in (
            (optics.defringe_purple_amount,
             optics.defringe_purple_hue_lo,
             optics.defringe_purple_hue_hi),
            (optics.defringe_green_amount,
             optics.defringe_green_hue_lo,
             optics.defringe_green_hue_hi)):
        if amount == 0.0:
            continue
        # Hue is 0..360 here; the stored range is 0..100.
        hue_lo = lo / 100.0 * 360.0
        hue_hi = hi / 100.0 * 360.0
        if hue_lo <= hue_hi:
            in_range = (h >= hue_lo) & (h <= hue_hi)
        else:
            in_range = (h >= hue_lo) | (h <= hue_hi)
        w = np.where(in_range & (s > 0.05),
                     amount / 100.0, 0.0)
        out = lum + (out - lum) * (1.0 - w[..., None])
    return out


# ----------------------------------------------------------------------
# Masks + spots
# ----------------------------------------------------------------------

def _mask_alpha(mask: DevelopMask, shape, lum):
    """The mask's alpha plane (h, w), 0..1, before ``invert``."""
    import numpy as np

    h, w = shape[0], shape[1]
    kind = mask.shape.get("kind", "")
    if kind in AI_MASK_SHAPES:
        raise DevelopError(
            DevelopErrorKind.model_missing,
            f"mask shape {kind!r} needs a segmentation model — "
            f"Devon ships none",
            hint="use a linear, radial, luminance_range or brush "
                 "mask instead")
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    nx = xx / max(w - 1, 1)
    ny = yy / max(h - 1, 1)
    if kind == "linear":
        sx, sy = mask.shape.get("start", [0.0, 0.0])
        ex, ey = mask.shape.get("end", [1.0, 1.0])
        ax, ay = ex - sx, ey - sy
        denom = ax * ax + ay * ay
        if denom <= 0:
            alpha = np.zeros((h, w))
        else:
            t = ((nx - sx) * ax + (ny - sy) * ay) / denom
            alpha = np.clip(t, 0.0, 1.0)
    elif kind == "radial":
        cx, cy = mask.shape.get("center", [0.5, 0.5])
        rx = max(mask.shape.get("rx", 0.2), 1e-3)
        ry = max(mask.shape.get("ry", 0.2), 1e-3)
        angle = math.radians(mask.shape.get("angle", 0.0))
        ca, sa = math.cos(angle), math.sin(angle)
        u = (nx - cx) * ca + (ny - cy) * sa
        v = -(nx - cx) * sa + (ny - cy) * ca
        d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        feather = mask.shape.get("feather", 50.0) / 100.0
        inner = max(1.0 - feather, 0.0)
        if inner >= 1.0:
            alpha = np.where(d <= 1.0, 1.0, 0.0)
        else:
            alpha = np.clip((1.0 - d) / (1.0 - inner), 0.0, 1.0)
    elif kind in ("luminance", "luminance_range"):
        lo = mask.shape.get("lo", 0.0)
        hi = mask.shape.get("hi", 1.0)
        lo_f = mask.shape.get("lo_feather", 0.0)
        hi_f = mask.shape.get("hi_feather", 0.0)
        alpha = _smoothstep(lo, lo + lo_f, lum) \
            * (1.0 - _smoothstep(hi - hi_f, hi, lum))
    elif kind == "brush":
        alpha = np.zeros((h, w))
        size_px = max(min(h, w), 1)
        for stroke in mask.shape.get("strokes", []):
            pts = stroke.get("points", [])
            size = stroke.get("size", 0.05) * size_px
            feather = stroke.get("feather", 50.0) / 100.0
            flow = stroke.get("flow", 100.0) / 100.0
            density = stroke.get("density", 100.0) / 100.0
            erase = stroke.get("erase", False)
            radius = max(size, 1.0)
            inner = max(radius * (1.0 - feather), 0.0)
            for px, py in pts:
                cx = px * (w - 1)
                cy = py * (h - 1)
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                if inner >= radius:
                    disc = np.where(d <= radius, 1.0, 0.0)
                else:
                    disc = np.clip((radius - d)
                                   / max(radius - inner, 1e-6),
                                   0.0, 1.0)
                disc = disc * flow * density
                if erase:
                    alpha = np.clip(alpha - disc, 0.0, 1.0)
                else:
                    alpha = np.clip(alpha + disc, 0.0, 1.0)
    else:  # pragma: no cover - from_dict rejects other kinds
        alpha = np.zeros((h, w))
    if mask.invert:
        alpha = 1.0 - alpha
    if mask.refine:
        alpha = _box_blur(alpha, max(
            1, int(round(mask.refine / 100.0 * 3.0))))
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def _apply_local_adjustments(x, adj: LocalAdjustments, alpha):
    """A mask's local adjustments as a mini-develop, blended by
    ``alpha * amount``. Only the adjustments the local panel
    actually carries are applied."""
    import numpy as np

    if adj.is_neutral() or not np.any(alpha > 0):
        return x
    y = np.clip(x, 0.0, 1.0)
    if adj.temp or adj.tint:
        y = _apply_wb(y, 6500.0 + adj.temp * 15.0, adj.tint)
    if adj.exposure:
        y = _apply_exposure(y, adj.exposure)
    light = Light(exposure=0.0, contrast=adj.contrast,
                  highlights=adj.highlights, shadows=adj.shadows,
                  whites=adj.whites, blacks=adj.blacks)
    y = _apply_light(y, light, 1)
    y = _apply_tone_map(y, light)
    if adj.texture:
        y = _local_contrast(y, adj.texture, 1, 0.5)
    if adj.clarity:
        y = _local_contrast(y, adj.clarity, 2, 1.0)
    if adj.dehaze:
        y = _apply_dehaze(y, adj.dehaze)
    if adj.hue or adj.saturation:
        h, s, v = _rgb_to_hsv(np.clip(y, 0.0, 1.0))
        h = h + adj.hue
        s = np.clip(s * (1.0 + adj.saturation / 100.0), 0.0, 1.0)
        y = _hsv_to_rgb(h, s, v)
    if adj.color_sat:
        tint = _hsv_to_rgb(
            np.full(y.shape[:2], adj.color_hue),
            np.full(y.shape[:2], adj.color_sat / 100.0),
            np.full(y.shape[:2], 0.5))
        a2 = (adj.color_sat / 100.0) * 0.5
        y = y * (1.0 - a2) + tint * a2
    if adj.sharpness:
        y = _apply_detail(y, Detail(
            sharpen_amount=adj.sharpness, sharpen_radius=1.0,
            sharpen_detail=25.0))
    if adj.noise:
        blur = _box_blur(y, 1)
        a = (adj.noise / 100.0) * 0.85
        y = y * (1.0 - a) + blur * a
    m = np.clip(alpha * (adj.amount / 100.0), 0.0, 1.0)
    return x * (1.0 - m[..., None]) + y * m[..., None]


def _apply_spots(x, spots):
    """Spot removal / healing / cloning. Remove, heal and fill
    sample the surrounding ring's median colour; clone copies
    from the given source point. All blends honour feather and
    opacity, and the centre of a removed spot matches its
    surroundings instead of going black."""
    import numpy as np

    if not spots:
        return x
    h, w = x.shape[0], x.shape[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = x.copy()
    for spot in spots:
        if not spot.points:
            continue
        radius = max(spot.size * min(h, w), 1.0)
        feather = spot.feather / 100.0
        opacity = spot.opacity / 100.0
        for px, py in spot.points:
            cx = px * (w - 1)
            cy = py * (h - 1)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            inner = max(radius * (1.0 - feather), 0.0)
            if inner >= radius:
                weight = np.where(d <= radius, 1.0, 0.0)
            else:
                weight = np.clip((radius - d)
                                 / max(radius - inner, 1e-6),
                                 0.0, 1.0)
            weight = weight * opacity
            if not np.any(weight > 0):
                continue
            if spot.mode == "clone" and spot.source_offset:
                sx, sy = spot.source_offset
                scx = (px + sx) * (w - 1)
                scy = (py + sy) * (h - 1)
                sxx = xx - cx + scx
                syy = yy - cy + scy
                fill = _bilinear(out, sxx, syy)
            else:
                ring = (d > radius) & (d <= radius * 2.0 + 1.0)
                if not np.any(ring):
                    ring = d > radius
                fill = np.array([
                    np.median(out[..., c][ring])
                    if np.any(ring) else 0.0 for c in range(3)])
            out = out * (1.0 - weight[..., None]) \
                + fill * weight[..., None]
    return out


# ----------------------------------------------------------------------
# Profile application + the render entry point
# ----------------------------------------------------------------------

def _apply_profile(settings: DevelopSettings) -> DevelopSettings:
    """Fold a built-in profile into the settings at its amount
    (0 = no-op, 100 = full, 200 = double). Unknown profile ids
    raise ``missing_profile`` here, at render — not when the
    settings were read, so a file naming a profile Devon has
    never seen still loads."""
    if settings.profile_id in ("lc.color",) \
            or settings.profile_amount == 0.0:
        return settings
    patch = BUILTIN_PROFILES.get(settings.profile_id)
    if patch is None:
        raise DevelopError(
            DevelopErrorKind.missing_profile,
            f"develop profile {settings.profile_id!r} is not "
            f"installed — built-ins are "
            f"{sorted(BUILTIN_PROFILES)}",
            hint="use profile lc.color, or install the profile "
                 "first")
    scale = settings.profile_amount / 100.0
    data = settings.to_dict()
    for section, fields in patch.items():
        if section == "treatment":
            if scale:
                data["treatment"] = fields
            continue
        if section == "wb":
            for key, value in fields.items():
                if key == "mode":
                    if scale:
                        data["wb"]["mode"] = value
                else:
                    base = data["wb"].get(key, 6500.0 if key == "temp"
                                          else 0.0)
                    data["wb"][key] = base + (value - base) * scale
            continue
        _deep_merge_scaled(data.setdefault(section, {}), fields,
                           scale)
    return DevelopSettings.from_dict(data)


def render_develop(img, settings=None):
    """Render ``img`` (a PIL image) through the full develop
    pipeline. ``settings`` is a ``DevelopSettings`` or a plain
    dict (``None`` = defaults). Returns a new PIL RGB image; the
    input is never mutated. HDR settings render through their SDR
    rendition — an 8-bit PIL image has no HDR headroom, and the
    offsets land on the matching sliders instead of vanishing.
    Raises ``DevelopError`` with a typed ``kind`` on every
    failure; unexpected internal errors surface as
    ``render_failed``, never as a bare traceback."""
    if settings is None:
        settings = DevelopSettings()
    elif isinstance(settings, dict):
        settings = DevelopSettings.from_dict(settings)
    elif not isinstance(settings, DevelopSettings):
        raise DevelopError(
            DevelopErrorKind.bad_params,
            "render_develop settings must be a DevelopSettings or "
            f"a dict, got {type(settings).__name__}")
    if not hasattr(img, "convert") or not hasattr(img, "size"):
        raise DevelopError(
            DevelopErrorKind.bad_file,
            "render_develop needs a PIL image — pass the result "
            "of open_develop_source() or Image.open()")
    try:
        return _render_develop_inner(img, settings)
    except DevelopError:
        raise
    except Exception as exc:  # noqa: BLE001 - render must not leak
        raise DevelopError(
            DevelopErrorKind.render_failed,
            f"develop render failed: {exc}",
            hint="try the same settings on a smaller image, or "
                 "reset to defaults and re-apply one section at a "
                 "time") from exc


def _render_develop_inner(img, settings: DevelopSettings):
    import numpy as np
    from PIL import Image

    if settings.hdr.enabled:
        settings = settings.sdr_rendition()
    settings = settings.effective()
    settings = _apply_profile(settings)
    if settings.geometry.upright in ("auto", "guided", "vertical",
                                     "full"):
        raise DevelopError(
            DevelopErrorKind.bad_params,
            f"geometry.upright {settings.geometry.upright!r} "
            f"needs scene analysis — Devon has no horizon/line "
            f"detector built",
            hint="use geometry.rotate for a manual correction, or "
                 "upright 'level' / 'off'")
    if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
        raise DevelopError(
            DevelopErrorKind.render_failed,
            "develop render needs Pillow and numpy installed")
    work = img.convert("RGB")

    # -- Stage 1: geometry (PIL) then optics remaps (numpy). --
    work = _apply_geometry_pil(work, settings.geometry)
    x = np.asarray(work).astype(np.float64) / 255.0
    if settings.optics.distortion:
        distortion = settings.optics.distortion
        if settings.optics.lens_profile:
            distortion = distortion \
                * (settings.optics.profile_distortion / 100.0)
        x = _apply_distortion(x, distortion)
    x = _apply_optics_vignetting(x, settings.optics)
    x = _apply_ca(x, settings.optics)
    x = _apply_defringe(x, settings.optics)
    x = np.clip(x, 0.0, 1.0)

    # -- Spots (normalised coords of the cropped image). --
    x = _apply_spots(x, settings.spots)

    # -- Stage 2: scene-linear. --
    if settings.wb.mode == "auto":
        # No auto-WB scene analysis is built; auto renders at the
        # as-shot defaults (6500 K, tint 0) and says so.
        x = _apply_wb(x, 6500.0, 0.0)
    else:
        x = _apply_wb(x, settings.wb.temp, settings.wb.tint)
    x = _apply_exposure(x, settings.light.exposure)
    x = _apply_dehaze(x, settings.effects.dehaze)
    process = settings.process.process()
    x = _apply_light(x, settings.light, process)
    if settings.effects.texture:
        x = _local_contrast(x, settings.effects.texture, 1, 0.5)
    if settings.effects.clarity:
        radius = max(1, min(x.shape[0], x.shape[1]) // 16)
        x = _local_contrast(x, settings.effects.clarity, radius,
                            1.0)
    lum = _luminance(x)
    for mask in settings.masks:
        if not mask.visible:
            continue
        alpha = _mask_alpha(mask, x.shape, lum)
        x = _apply_local_adjustments(x, mask.adjust, alpha)
    x = np.clip(x, 0.0, 1.0)

    # -- Stage 3: tone map. --
    x = _apply_tone_map(x, settings.light)
    x = np.clip(x, 0.0, 1.0)

    # -- Stage 4: colour. --
    x = _apply_saturation(x, settings.color)
    x = _apply_mixer(x, settings.mixer)
    x = _apply_grading(x, settings.grading)
    if settings.treatment == "bw":
        x = _apply_bw(x, settings.bw_mix)
    x = _apply_calibration(x, settings.calibration)
    x = np.clip(x, 0.0, 1.0)

    # -- Stage 5: display. --
    x = _apply_curves(x, settings.curve)
    x = _apply_detail(x, settings.detail)
    x = _apply_vignette(x, settings.vignette)
    x = _apply_grain(x, settings.grain)
    x = np.clip(x, 0.0, 1.0)

    out = Image.fromarray((np.round(x * 255.0)).astype("uint8"),
                          mode="RGB")
    return out


# ----------------------------------------------------------------------
# DevelopStack — the non-destructive edit stack
# ----------------------------------------------------------------------

#: How many rendered results the stack caches (upstream's
#: StageCache capacity is 2; the same bound keeps memory honest).
_STACK_CACHE_CAPACITY = 2


class DevelopStack:
    """A named, non-destructive edit stack over one source image.

    The source bytes are stored once and never mutated; every
    render re-derives from them plus the current settings, so
    applying a second edit never re-encodes an intermediate —
    there is no quality loss from re-rendering. A render cache
    (capacity 2, keyed by source hash + settings hash) makes
    re-rendering unchanged settings byte-identical without
    re-running the pipeline.
    """

    def __init__(self, source, name: str = "") -> None:
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        elif isinstance(source, (str, os.PathLike)):
            path = os.fspath(source)
            if not os.path.exists(path):
                raise DevelopError(
                    DevelopErrorKind.bad_file,
                    f"develop source {path!r} does not exist")
            with open(path, "rb") as fh:
                raw = fh.read()
        elif hasattr(source, "save"):
            stream = io.BytesIO()
            source.save(stream, format="PNG")
            raw = stream.getvalue()
        else:
            raise DevelopError(
                DevelopErrorKind.bad_file,
                "a develop stack needs image bytes, a path, or a "
                f"PIL image, got {type(source).__name__}")
        # Validate once, up front: a corrupt source fails here,
        # not on the first render.
        open_develop_source(raw)
        self._source_bytes = raw
        self._name = name
        self._settings = DevelopSettings()
        self._edits: list = []
        self._cache: dict = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_bytes(self) -> bytes:
        """The source bytes, exactly as given — never mutated."""
        return self._source_bytes

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self._source_bytes).hexdigest()

    @property
    def settings(self) -> DevelopSettings:
        return self._settings

    @property
    def edits(self) -> list:
        """The named edits, in order, as (name, settings) pairs —
        the stack's edit history, so a round-trip keeps it."""
        return list(self._edits)

    def add_edit(self, name: str, params=None, **kwargs):
        """Apply a NAMED edit on top of the current settings and
        record it in the history: ``add_edit("lift", {"light":
        {"exposure": 1.0}})``. ``params`` is a nested settings
        dict, a flat params dict (legacy sliders / dotted new
        controls), or a ``DevelopSettings``. Returns ``self`` so
        edits chain."""
        if not isinstance(name, str):
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "a stack edit name must be a string")
        if params is None and kwargs:
            params = kwargs
        if params is None:
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "add_edit needs settings — a dict or a "
                "DevelopSettings")
        if isinstance(params, DevelopSettings):
            patch = params
        elif isinstance(params, dict):
            if any(k in DevelopSettings._TOP for k in params):
                patch = DevelopSettings.from_dict(params)
            else:
                patch = develop_settings_from_params(params)
        else:
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "a stack edit takes a DevelopSettings or a dict "
                f"of develop params, got {type(params).__name__}")
        self._settings = _merge_settings(self._settings, patch)
        self._edits.append((name, patch))
        return self

    def edit(self, params=None, **kwargs):
        """Apply an edit on top of the current settings: a
        ``DevelopSettings`` (field-level merge — only its
        non-default fields override), a dict of flat develop
        params (legacy sliders, dotted new controls, or a whole
        ``settings`` dict), or keyword params. Returns ``self``
        so edits chain. The cache is left alone — a changed
        settings hash simply misses it."""
        if params is None and kwargs:
            params = kwargs
        if params is None:
            return self
        if isinstance(params, DevelopSettings):
            patch = params
        elif isinstance(params, dict):
            patch = develop_settings_from_params(params)
        else:
            raise DevelopError(
                DevelopErrorKind.bad_params,
                "a stack edit takes a DevelopSettings or a dict "
                f"of develop params, got {type(params).__name__}")
        self._settings = _merge_settings(self._settings, patch)
        return self

    def apply_preset(self, name: str, amount: float = 100.0):
        self._settings = self._settings.apply_preset(name, amount)
        return self

    def reset(self):
        self._settings = DevelopSettings()
        return self

    def before_settings(self) -> DevelopSettings:
        return self._settings.before_settings()

    def _render_png(self, settings: DevelopSettings) -> bytes:
        key = (self.source_sha256, settings.hash64())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        img = open_develop_source(self._source_bytes)
        rendered = render_develop(img, settings)
        stream = io.BytesIO()
        rendered.save(stream, format="PNG")
        data = stream.getvalue()
        self._cache[key] = data
        while len(self._cache) > _STACK_CACHE_CAPACITY:
            self._cache.pop(next(iter(self._cache)))
        return data

    def render(self):
        """The current render as a PIL RGB image."""
        from PIL import Image

        return Image.open(
            io.BytesIO(self._render_png(self._settings))).convert(
                "RGB")

    def render_png(self) -> bytes:
        """The current render as PNG bytes — byte-identical for
        the same source + settings, every time."""
        return self._render_png(self._settings)

    def render_before(self):
        """The 'before' render: defaults, keeping crop,
        orientation and process so the framing matches."""
        from PIL import Image

        return Image.open(io.BytesIO(
            self._render_png(self.before_settings()))).convert("RGB")

    def to_dict(self) -> dict:
        return {"name": self._name,
                "source_sha256": self.source_sha256,
                "settings": self._settings.to_dict(),
                "edits": [{"name": name,
                           "settings": patch.to_dict()}
                          for name, patch in self._edits]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text, source):
        """Rebuild a stack from ``to_json()`` output plus the
        source bytes. A source whose hash does not match the
        stored hash raises ``bad_file`` — the settings belong to
        a different photo."""
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise DevelopError(
                DevelopErrorKind.bad_file,
                f"develop stack JSON could not be read: {exc}"
            ) from exc
        if not isinstance(data, dict) or "settings" not in data:
            raise DevelopError(
                DevelopErrorKind.bad_file,
                "develop stack JSON must be an object with a "
                "'settings' key")
        stack = cls(source, name=data.get("name", ""))
        expected = data.get("source_sha256")
        if expected is not None and expected != stack.source_sha256:
            raise DevelopError(
                DevelopErrorKind.bad_file,
                "develop stack source does not match the stored "
                "source hash — these settings belong to a "
                "different photo",
                hint="pass the original source bytes, or start a "
                     "new stack")
        stack._settings = DevelopSettings.from_dict(data["settings"])
        edits = data.get("edits", []) or []
        if not isinstance(edits, list):
            raise DevelopError(
                DevelopErrorKind.bad_file,
                "develop stack JSON 'edits' must be a list")
        for entry in edits:
            if (not isinstance(entry, dict)
                    or "name" not in entry
                    or "settings" not in entry):
                raise DevelopError(
                    DevelopErrorKind.bad_file,
                    "a develop stack edit must be an object with "
                    "'name' and 'settings'")
            stack._edits.append(
                (entry["name"],
                 DevelopSettings.from_dict(entry["settings"])))
        return stack
