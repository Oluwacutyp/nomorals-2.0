"""Format registry — one registry for Devon's video formats (§1.5b).

A *format* is a named parameter set over the Scene engine: canvas
(dimensions + fps), safe areas, caption defaults, a grade pack (the
:mod:`pure_render` pack shape), an audio target (platform loudness via
:func:`audio_ops.loudness_spec_for`), and — for formats that steer
directed work — a director design pack. Formats are *data*
(:class:`FormatSpec` is a frozen dataclass with a JSON form), registered
in the `contentops/niches/registry.py` idiom (register / get /
validate / list — copied pattern, P7), extended by
:func:`scaffold_format` + :func:`load_format_file` so the owner can add
a format as a data file, never as code.

The three partial format systems that predated the registry are
*mapped onto it*, not rewritten:

* ``contentops.styles`` (phonk / documentary / vlog / minimal) — each
  style's entry is **probed** from the style's own ``build_spec``
  output at bootstrap (canvas, caption style/position/size and audio
  target read off the spec it actually builds), so the registry can
  never drift from the incumbent. ``contentops.styles.build_spec``
  in turn consumes the registry via its ``format=`` argument.
* ``pure_render`` packs — a format's ``grade_pack`` *is* a pack
  (``{"style", "spec_overrides"}``), and :func:`apply_pack` accepts
  ``pack={"format": <name>}``, resolving through
  :func:`pack_for_format`.
* ``director.design.STYLE_PACKS`` (neutral / naturalist / noir / neon /
  golden) — registered as design-only formats (``design_pack`` group);
  they steer shot design, not the render canvas, and say so.
* ``motion_studio.grading.FORMAT_SIZES`` aspects (9:16 / 16:9 / 1:1 /
  4:5) — registered as the canvas formats ``portrait`` / ``landscape``
  / ``square`` / ``portrait45`` those aspect names map to.

Resolution: :func:`resolve_format` walks the entry → ``extends``
chain → registry defaults (which mirror the engine's own defaults:
EditSpec 1080×1920@30, TextLayer karaoke/bottom/72, LoudnessSpec
-14 LUFS) and returns a total :class:`ResolvedFormat` in which *every*
parameter names its provenance (``format`` / ``base:<name>`` /
``default``). Broken chains (unknown base, cycle) and invalid declared
values fail loudly; ``require=(...)`` makes resolution fail loudly
when a needed parameter group is one the format itself does not speak
(defaults do not count). The 3D seam (§1.5e): the ``camera3d`` format
declares ``renderer="camera3d"`` — it resolves as data, and
:func:`apply_format` resolves it through the scene3d renderer
registry (:func:`_resolve_format_renderer`): a rung serving the
``format3d`` surface lets the apply proceed with the rung named in
the record; otherwise it refuses with the reason. No stub renderer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...media_edit.videos import MediaEditError

__all__ = [
    "FormatError",
    "FormatSpec",
    "ResolvedFormat",
    "validate_format",
    "register_format",
    "get_format",
    "list_formats",
    "resolve_format",
    "apply_format",
    "pack_for_format",
    "formats_data",
    "scaffold_format",
    "load_format_file",
    "bootstrap_issues",
]


class FormatError(MediaEditError):
    """Raised when a format is malformed, unknown, or unresolvable."""


#: parameter groups a format can speak (declared by itself or a base).
GROUPS = ("canvas", "safe", "captions", "grade", "audio", "design")

#: registry-wide defaults — the engine's own defaults, named as such in
#: every provenance record (EditSpec / TextLayer / LoudnessSpec).
_DEFAULTS: dict[str, Any] = {
    "width": 1080, "height": 1920, "fps": 30,
    "safe": {"left": 0.05, "right": 0.05, "top": 0.05, "bottom": 0.10},
    "caption_style": "karaoke", "caption_position": "bottom",
    "caption_size": 72, "caption_wpm": 150.0,
    "grade_pack": None, "design_pack": None,
    "audio_platform": "", "target_lufs": None,
}

_SAFE_KEYS = ("left", "right", "top", "bottom")
_RENDERERS = ("engine", "camera3d")
_NAME_RE = re.compile(r"^[a-z0-9_]+$")


# ---------------------------------------------------------------------------
# The format itself — data, JSON-round-trippable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatSpec:
    """One format as data. Every parameter group is optional at
    declaration; :func:`resolve_format` makes the set total."""

    name: str
    label: str = ""
    extends: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    safe: dict[str, float] | None = None
    caption_style: str | None = None
    caption_position: str | None = None
    caption_size: int | None = None
    caption_wpm: float | None = None
    grade_pack: dict[str, Any] | None = None
    audio_platform: str | None = None
    target_lufs: float | None = None
    design_pack: str | None = None
    renderer: str = "engine"
    source: str = ""        # where a mapped entry came from (audit)

    def groups(self) -> list[str]:
        """The parameter groups this entry itself declares."""
        out: list[str] = []
        if self.width is not None or self.height is not None \
                or self.fps is not None:
            out.append("canvas")
        if self.safe is not None:
            out.append("safe")
        if any(v is not None for v in (self.caption_style,
                                       self.caption_position,
                                       self.caption_size,
                                       self.caption_wpm)):
            out.append("captions")
        if self.grade_pack is not None:
            out.append("grade")
        if self.audio_platform is not None or self.target_lufs is not None:
            out.append("audio")
        if self.design_pack is not None:
            out.append("design")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "label": self.label,
            "extends": self.extends,
            "width": self.width, "height": self.height, "fps": self.fps,
            "safe": dict(self.safe) if self.safe is not None else None,
            "caption_style": self.caption_style,
            "caption_position": self.caption_position,
            "caption_size": self.caption_size,
            "caption_wpm": self.caption_wpm,
            "grade_pack": (dict(self.grade_pack)
                           if self.grade_pack is not None else None),
            "audio_platform": self.audio_platform,
            "target_lufs": self.target_lufs,
            "design_pack": self.design_pack,
            "renderer": self.renderer, "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FormatSpec":
        """Build from a data dict (e.g. a scaffolded JSON file).

        Keys starting with ``_`` are comments and skipped; any other
        unknown key is an issue — data discipline means a typo'd key
        fails loudly at load, never silently drops a parameter.
        """
        if not isinstance(data, dict):
            raise FormatError(f"format must be a dict, got {type(data).__name__}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(k for k in data
                         if k not in known and not str(k).startswith("_"))
        if unknown:
            raise FormatError(
                f"unknown format keys {unknown}; use: {sorted(known)}")
        safe = data.get("safe")
        return cls(
            name=str(data.get("name", "")),
            label=str(data.get("label", "")),
            extends=data.get("extends") or None,
            width=data.get("width"), height=data.get("height"),
            fps=data.get("fps"),
            safe=dict(safe) if isinstance(safe, dict) else safe,
            caption_style=data.get("caption_style") or None,
            caption_position=data.get("caption_position") or None,
            caption_size=data.get("caption_size"),
            caption_wpm=data.get("caption_wpm"),
            grade_pack=data.get("grade_pack"),
            audio_platform=data.get("audio_platform") or None,
            target_lufs=data.get("target_lufs"),
            design_pack=data.get("design_pack") or None,
            renderer=str(data.get("renderer", "engine")),
            source=str(data.get("source", "")),
        )


@dataclass(frozen=True)
class ResolvedFormat:
    """A format with every parameter resolved + provenance named."""

    name: str
    label: str
    source: str
    renderer: str
    groups: list[str]           # groups spoken by the entry or a base
    width: int
    height: int
    fps: int
    safe: dict[str, float]
    caption_style: str
    caption_position: str
    caption_size: int
    caption_wpm: float
    grade_pack: dict[str, Any] | None
    design_pack: str | None
    loudness: dict[str, Any]    # LoudnessSpec as data
    provenance: dict[str, str]  # param → "format" | "base:<name>" | "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "label": self.label, "source": self.source,
            "renderer": self.renderer, "groups": list(self.groups),
            "canvas": {"width": self.width, "height": self.height,
                       "fps": self.fps},
            "safe": dict(self.safe),
            "captions": {"style": self.caption_style,
                         "position": self.caption_position,
                         "size": self.caption_size, "wpm": self.caption_wpm},
            "grade_pack": (dict(self.grade_pack)
                           if self.grade_pack is not None else None),
            "design_pack": self.design_pack,
            "loudness": dict(self.loudness),
            "provenance": dict(self.provenance),
        }


# ---------------------------------------------------------------------------
# Validation — the niches idiom: issues list, register raises
# ---------------------------------------------------------------------------

def validate_format(spec: FormatSpec) -> list[str]:
    """Return a list of problems with ``spec``; empty means valid."""
    issues: list[str] = []
    if not isinstance(spec, FormatSpec):
        return [f"not a FormatSpec instance: {type(spec).__name__}"]
    if not spec.name or not _NAME_RE.match(spec.name):
        issues.append(f"name {spec.name!r} must be a lowercase slug (a-z0-9_)")
    if not (spec.label or "").strip():
        issues.append("label must be non-empty")
    if spec.extends is not None and spec.extends == spec.name:
        issues.append("a format may not extend itself")
    if spec.renderer not in _RENDERERS:
        issues.append(f"renderer {spec.renderer!r}; use: {list(_RENDERERS)}")
    if not spec.groups() and spec.renderer == "engine":
        issues.append(
            "format declares no parameter group — a format must speak "
            "at least one of: " + ", ".join(GROUPS))
    # canvas: both dimensions or neither, all positive
    if (spec.width is None) != (spec.height is None):
        issues.append("width and height must be declared together")
    for attr in ("width", "height", "fps"):
        v = getattr(spec, attr)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool)
                              or v <= 0):
            issues.append(f"{attr} must be a positive int, got {v!r}")
    # safe areas: known keys, fractions in [0, 0.5), opposite sides < 1
    if spec.safe is not None:
        if not isinstance(spec.safe, dict):
            issues.append("safe must be a dict of side → fraction")
        else:
            unknown = sorted(set(spec.safe) - set(_SAFE_KEYS))
            if unknown:
                issues.append(f"safe keys {unknown}; use: {list(_SAFE_KEYS)}")
            for side, v in spec.safe.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool) \
                        or not 0.0 <= float(v) < 0.5:
                    issues.append(
                        f"safe[{side!r}] must be a fraction in [0, 0.5), "
                        f"got {v!r}")
            if isinstance(spec.safe.get("left"), (int, float)) \
                    and isinstance(spec.safe.get("right"), (int, float)) \
                    and spec.safe["left"] + spec.safe["right"] >= 1.0:
                issues.append("safe left + right must be < 1")
            if isinstance(spec.safe.get("top"), (int, float)) \
                    and isinstance(spec.safe.get("bottom"), (int, float)) \
                    and spec.safe["top"] + spec.safe["bottom"] >= 1.0:
                issues.append("safe top + bottom must be < 1")
    # captions: style/position checked against the real vocabularies
    if spec.caption_style is not None:
        from ...media_edit.captions import caption_styles
        if spec.caption_style not in caption_styles():
            issues.append(
                f"caption_style {spec.caption_style!r}; use: "
                f"{caption_styles()}")
    if spec.caption_position is not None:
        from .text import _POSITIONS
        if spec.caption_position not in _POSITIONS:
            issues.append(
                f"caption_position {spec.caption_position!r}; use: "
                f"{sorted(_POSITIONS)}")
    if spec.caption_size is not None and (
            not isinstance(spec.caption_size, int)
            or isinstance(spec.caption_size, bool) or spec.caption_size <= 0):
        issues.append(f"caption_size must be a positive int, "
                      f"got {spec.caption_size!r}")
    if spec.caption_wpm is not None and (
            not isinstance(spec.caption_wpm, (int, float))
            or isinstance(spec.caption_wpm, bool) or spec.caption_wpm <= 0):
        issues.append(f"caption_wpm must be > 0, got {spec.caption_wpm!r}")
    # grade pack: must be a pack apply_pack would accept
    if spec.grade_pack is not None:
        if not isinstance(spec.grade_pack, dict):
            issues.append("grade_pack must be a pack dict "
                          '{"style": ..., "spec_overrides": {...}}')
        else:
            from ..contentops.styles import list_styles
            from .pure_render import _PACK_OVERRIDE_KEYS
            style = spec.grade_pack.get("style")
            if style is not None and str(style).strip().lower() \
                    not in list_styles():
                issues.append(
                    f"grade_pack style {style!r}; use: {list_styles()}")
            overrides = spec.grade_pack.get("spec_overrides") or {}
            if not isinstance(overrides, dict):
                issues.append("grade_pack['spec_overrides'] must be a dict")
            else:
                bad = sorted(set(overrides) - _PACK_OVERRIDE_KEYS)
                if bad:
                    issues.append(
                        f"grade_pack may not override {bad} — a pack swaps "
                        f"the look, never the cut (allowed: "
                        f"{sorted(_PACK_OVERRIDE_KEYS)})")
    # audio: platform must be a published loudness norm
    if spec.audio_platform is not None:
        from ..deliverables import PLATFORM_LOUDNESS
        if spec.audio_platform.strip().lower() not in PLATFORM_LOUDNESS:
            issues.append(
                f"audio_platform {spec.audio_platform!r}; use: "
                f"{sorted(PLATFORM_LOUDNESS)}")
    if spec.target_lufs is not None and (
            not isinstance(spec.target_lufs, (int, float))
            or isinstance(spec.target_lufs, bool)
            or not -70.0 <= float(spec.target_lufs) <= 0.0):
        issues.append(f"target_lufs must be in [-70, 0], "
                      f"got {spec.target_lufs!r}")
    # design: pack must exist in the director vocabulary
    if spec.design_pack is not None:
        from ..director.design import STYLE_PACKS
        if spec.design_pack.strip().lower() not in STYLE_PACKS:
            issues.append(
                f"design_pack {spec.design_pack!r}; use: "
                f"{sorted(STYLE_PACKS)}")
    return issues


# ---------------------------------------------------------------------------
# Registry — niches idiom (register / get / list), bootstrapped shipped set
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, FormatSpec] = {}
_BOOTSTRAPPED = False
_BOOTSTRAP_ISSUES: list[str] = []


def register_format(spec: FormatSpec) -> FormatSpec:
    """Register a format. Raises :class:`FormatError` if malformed."""
    issues = validate_format(spec)
    if issues:
        raise FormatError(
            f"invalid format {getattr(spec, 'name', '?')!r}: "
            + "; ".join(issues))
    name = spec.name
    if name in _REGISTRY and _REGISTRY[name] is not spec \
            and _REGISTRY[name] != spec:
        raise FormatError(f"format {name!r} is already registered")
    _REGISTRY[name] = spec
    return spec


def _register_quiet(spec: FormatSpec, origin: str) -> None:
    try:
        register_format(spec)
    except FormatError as exc:
        _BOOTSTRAP_ISSUES.append(f"{origin}: {exc}")


def _style_entries() -> list[FormatSpec]:
    """contentops styles, probed from their own build_spec output.

    The registry reads the incumbent's real product — canvas, caption
    style/position/size and audio target off a spec the style actually
    builds, caption pace off its signature — so a style that changes
    its look changes its format entry with it. No copied constants.
    """
    import inspect

    from ..contentops import styles as _styles

    entries: list[FormatSpec] = []
    for name in _styles.list_styles():
        module = _styles.get_style(name)
        probe = _styles.build_spec(
            name, captions="format registry probe words",
            music="format-probe.mp3")
        layer = probe.text_layers[0] if probe.text_layers else None
        sig = inspect.signature(module.build_spec).parameters
        wpm = sig["caption_wpm"].default if "caption_wpm" in sig else None
        entries.append(FormatSpec(
            name=name,
            label=f"{name} — contentops edit style, as a format",
            width=int(probe.width), height=int(probe.height),
            fps=int(probe.fps),
            caption_style=layer.style if layer else None,
            caption_position=layer.position if layer else None,
            caption_size=int(layer.size) if layer else None,
            caption_wpm=float(wpm) if wpm is not None else None,
            grade_pack={"style": name},
            target_lufs=(float(probe.audio.target_lufs)
                         if probe.audio is not None else None),
            source=f"contentops.styles.{name} (probed build_spec)",
        ))
    return entries


def _design_entries() -> list[FormatSpec]:
    """director STYLE_PACKS as design-only formats."""
    from ..director.design import STYLE_PACKS

    return [
        FormatSpec(
            name=name,
            label=str(pack.get("label") or name),
            design_pack=name,
            source="director.design.STYLE_PACKS",
        )
        for name, pack in sorted(STYLE_PACKS.items())
    ]


def _aspect_entries() -> list[FormatSpec]:
    """motion FORMAT_SIZES aspects as canvas formats.

    FORMAT_SIZES maps aspect → profile canvas key; the registry entry
    takes that key as its name and the standard delivery canvas for
    the aspect as its dimensions.
    """
    from ..motion_studio.grading import FORMAT_SIZES

    canvases = {
        "portrait": (1080, 1920),     # 9:16
        "landscape": (1920, 1080),    # 16:9
        "square": (1080, 1080),       # 1:1
        "portrait45": (1080, 1350),   # 4:5
    }
    aspects = {v: k for k, v in FORMAT_SIZES.items()}
    entries: list[FormatSpec] = []
    for key in sorted(set(FORMAT_SIZES.values())):
        w, h = canvases[key]
        entries.append(FormatSpec(
            name=key,
            label=f"{aspects.get(key, key)} canvas "
                  f"({w}×{h}) — motion FORMAT_SIZES",
            width=w, height=h,
            source="motion_studio.grading.FORMAT_SIZES",
        ))
    return entries


def _shipped_entries() -> list[FormatSpec]:
    """Formats that are this registry's own data (not incumbent maps)."""
    return [
        FormatSpec(
            name="tiktok_short",
            label="TikTok / Shorts vertical — platform-UI safe zones, "
                  "karaoke captions, TikTok loudness",
            extends="portrait",
            safe={"left": 0.06, "right": 0.14, "top": 0.10,
                  "bottom": 0.22},
            caption_style="karaoke", caption_position="bottom",
            audio_platform="tiktok",
            source="format registry (shipped data)",
        ),
        FormatSpec(
            name="youtube_shorts",
            label="YouTube Shorts vertical — bold captions, "
                  "platform-UI safe zones, YouTube loudness",
            extends="portrait",
            safe={"left": 0.05, "right": 0.12, "top": 0.08,
                  "bottom": 0.18},
            caption_style="hormozi", caption_position="bottom",
            audio_platform="youtube",
            source="format registry (shipped data)",
        ),
        FormatSpec(
            name="youtube_video",
            label="YouTube landscape — minimal captions, "
                  "YouTube loudness",
            extends="landscape",
            caption_style="minimal", caption_position="bottom",
            audio_platform="youtube",
            source="format registry (shipped data)",
        ),
        FormatSpec(
            name="instagram_square",
            label="Instagram square feed — Instagram loudness",
            extends="square",
            audio_platform="instagram",
            source="format registry (shipped data)",
        ),
        FormatSpec(
            name="instagram_portrait",
            label="Instagram portrait feed (4:5) — minimal captions, "
                  "Instagram loudness",
            extends="portrait45",
            caption_style="minimal", caption_position="bottom",
            audio_platform="instagram",
            source="format registry (shipped data)",
        ),
        FormatSpec(
            name="camera3d",
            label="3D camera format — resolves as data; at apply it "
                  "resolves through the scene3d renderer registry "
                  "(§1.5e) and refuses, naming why, while no rung "
                  "serves the format3d surface",
            extends="landscape",
            renderer="camera3d",
            source="format registry (shipped data; §1.5e seam)",
        ),
    ]


def _bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True
    for origin, builder in (("styles", _style_entries),
                            ("design packs", _design_entries),
                            ("aspect sizes", _aspect_entries)):
        try:
            for spec in builder():
                _register_quiet(spec, origin)
        except Exception as exc:  # noqa: BLE001 — a broken incumbent
            # mapping must not take the registry down; it is recorded
            # and surfaced by bootstrap_issues() instead.
            _BOOTSTRAP_ISSUES.append(f"{origin}: {exc}")
    for spec in _shipped_entries():
        _register_quiet(spec, "shipped formats")
    # Owner-extended formats: data files, validated like everything
    # else. Activated only when NM_FORMAT_DIRS names directories.
    for directory in os.environ.get("NM_FORMAT_DIRS", "").split(os.pathsep):
        if not directory.strip():
            continue
        d = Path(directory)
        if not d.is_dir():
            _BOOTSTRAP_ISSUES.append(f"NM_FORMAT_DIRS: not a dir: {d}")
            continue
        for path in sorted(d.glob("*.format.json")):
            try:
                load_format_file(path)
            except Exception as exc:  # noqa: BLE001 — recorded, not fatal
                _BOOTSTRAP_ISSUES.append(f"{path}: {exc}")


def bootstrap_issues() -> list[str]:
    """Problems hit while bootstrapping incumbent mappings / user
    format files — recorded honestly instead of failing the registry."""
    _bootstrap()
    return list(_BOOTSTRAP_ISSUES)


def get_format(name: str) -> FormatSpec:
    """The registered format entry. Raises on unknown."""
    _bootstrap()
    key = str(name or "").strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise FormatError(
            f"unknown format {name!r}; use: {list_formats()}") from None


def list_formats() -> list[str]:
    """Sorted list of registered format names."""
    _bootstrap()
    return sorted(_REGISTRY)


def formats_data() -> dict[str, Any]:
    """The registry as comparable data (brain tool / CLI list)."""
    _bootstrap()
    return {
        "formats": [
            {"name": s.name, "label": s.label, "groups": s.groups(),
             "extends": s.extends, "renderer": s.renderer,
             "source": s.source}
            for s in (_REGISTRY[n] for n in sorted(_REGISTRY))
        ],
        "count": len(_REGISTRY),
        "bootstrap_issues": list(_BOOTSTRAP_ISSUES),
    }


# ---------------------------------------------------------------------------
# Resolution — entry → extends chain → defaults, provenance per parameter
# ---------------------------------------------------------------------------

_PARAMS = ("width", "height", "fps", "safe", "caption_style",
           "caption_position", "caption_size", "caption_wpm",
           "grade_pack", "design_pack", "audio_platform", "target_lufs")

_PARAM_GROUP = {
    "width": "canvas", "height": "canvas", "fps": "canvas",
    "safe": "safe",
    "caption_style": "captions", "caption_position": "captions",
    "caption_size": "captions", "caption_wpm": "captions",
    "grade_pack": "grade", "design_pack": "design",
    "audio_platform": "audio", "target_lufs": "audio",
}


def _chain(name: str) -> list[FormatSpec]:
    """[entry, base, base's base, …] — cycles/unknown bases raise."""
    chain: list[FormatSpec] = []
    seen: set[str] = set()
    key = str(name or "").strip().lower()
    while True:
        if key in seen:
            raise FormatError(
                f"format extends cycle at {key!r}: "
                + " → ".join(s.name for s in chain) + f" → {key}")
        seen.add(key)
        entry = get_format(key)
        chain.append(entry)
        if entry.extends is None:
            return chain
        key = str(entry.extends).strip().lower()


def resolve_format(name: str, *,
                   require: Any = ()) -> ResolvedFormat:
    """Resolve ``name`` to a total parameter set with provenance.

    Every parameter comes from the entry, an ``extends`` ancestor, or
    the registry defaults (the engine's own defaults), and the
    provenance record says which. ``require`` names parameter groups
    the *format itself* (entry or ancestors) must speak — a required
    group covered only by defaults fails loudly, naming the group.
    """
    chain = _chain(name)
    entry = chain[0]
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for param in _PARAMS:
        for depth, link in enumerate(chain):
            v = getattr(link, param)
            if v is not None:
                values[param] = v
                provenance[param] = ("format" if depth == 0
                                     else f"base:{link.name}")
                break
        else:
            values[param] = _DEFAULTS[param]
            provenance[param] = "default"
    groups: list[str] = []
    for link in chain:
        for g in link.groups():
            if g not in groups:
                groups.append(g)
    missing = [g for g in require if g not in groups]
    if missing:
        raise FormatError(
            f"format {entry.name!r} does not declare required group(s) "
            f"{missing} (it speaks: {sorted(groups) or 'none'}; defaults "
            f"do not count)")
    from .audio_ops import loudness_spec_for
    loudness = loudness_spec_for(
        str(values["audio_platform"] or ""),
        target_lufs=(float(values["target_lufs"])
                     if values["target_lufs"] is not None else None),
    ).to_dict()
    return ResolvedFormat(
        name=entry.name, label=entry.label, source=entry.source,
        renderer=entry.renderer, groups=sorted(groups),
        width=int(values["width"]), height=int(values["height"]),
        fps=int(values["fps"]),
        safe={k: float(values["safe"].get(k, _DEFAULTS["safe"][k]))
              for k in _SAFE_KEYS},
        caption_style=str(values["caption_style"]),
        caption_position=str(values["caption_position"]),
        caption_size=int(values["caption_size"]),
        caption_wpm=float(values["caption_wpm"]),
        grade_pack=(dict(values["grade_pack"])
                    if values["grade_pack"] is not None else None),
        design_pack=values["design_pack"],
        loudness=loudness, provenance=provenance,
    )


def pack_for_format(name: str) -> dict[str, Any]:
    """The format's grade pack, in :func:`pure_render.apply_pack` shape.

    ``{}`` when the format declares no grade — a pack swaps the look,
    and a format without a grade has no look to swap.
    """
    resolved = resolve_format(name)
    return dict(resolved.grade_pack) if resolved.grade_pack else {}


# ---------------------------------------------------------------------------
# Application — a format steers a spec
# ---------------------------------------------------------------------------

def _resolve_format_renderer(resolved: ResolvedFormat) -> dict[str, Any]:
    """The §1.5e seam, resolved through the scene3d registry.

    A format declaring a non-engine renderer (the shipped
    ``camera3d``) asks :func:`scene3d.resolve_renderer` for a rung
    serving the ``format3d`` surface at the format's canvas:

    * **no rung registered** — the historical refusal, verbatim:
      the seam predates the registry and its wording is pinned;
    * **rungs registered, none serving** — a refusal naming each
      rung's recorded reason (probe failure or capability gap);
    * **a rung serves** — its resolution record, which
      :func:`apply_format` carries in the apply record's
      ``renderer3d`` entry while the format steers the spec as data.
    """
    from .scene3d import Scene3DError, registered_renderers, resolve_renderer

    if not registered_renderers():
        raise FormatError(
            f"format {resolved.name!r} declares renderer "
            f"{resolved.renderer!r}, which is unavailable: no 3D rung "
            f"has been chosen (§1.5e/§6.3 — Blender-class external "
            f"engine vs a Rust rung is the owner's decision). The "
            f"format resolves as data; it does not render.")
    try:
        _renderer, record = resolve_renderer(
            surface="format3d",
            requirements={"width": resolved.width,
                          "height": resolved.height})
    except Scene3DError as exc:
        raise FormatError(
            f"format {resolved.name!r} declares renderer "
            f"{resolved.renderer!r}: {exc} The format resolves as "
            f"data; it does not render.") from exc
    return record


def apply_format(spec: Any, name: str, *,
                 require: Any = ()) -> tuple[Any, dict[str, Any]]:
    """Apply format ``name`` to an EditSpec (or spec dict).

    Returns ``(new_spec, record)``; the input is not mutated. Canvas
    (width/height/fps) is set from the format. Caption defaults fill
    text layers whose fields still sit at the TextLayer defaults —
    a layer that chose its own style/position/size keeps it — and a
    format-declared safe area raises caption ``margin_v`` to clear
    the bottom safe zone. The record names the format, the resolved
    provenance, the grade pack (for the render path to consume via
    :func:`pure_render.render_pure`), the loudness contract, and how
    many layers the caption defaults touched. A format whose renderer
    is unavailable (the §1.5e camera3d seam) refuses, naming why; the
    seam resolves through the scene3d renderer registry — see
    :func:`_resolve_format_renderer`. When a rung serves, the apply
    proceeds and the record's ``renderer3d`` entry names the rung
    and its capabilities.
    """
    from .spec import EditSpec

    resolved = resolve_format(name, require=require)
    renderer3d_record = None
    if resolved.renderer != "engine":
        renderer3d_record = _resolve_format_renderer(resolved)
    was_spec = isinstance(spec, EditSpec)
    data = spec.to_dict() if was_spec else dict(spec)
    data["width"] = resolved.width
    data["height"] = resolved.height
    data["fps"] = resolved.fps
    from .text import TextLayer
    defaults = {f: getattr(TextLayer.__dataclass_fields__[f], "default")
                for f in ("style", "position", "size", "margin_v")}
    safe_declared = resolved.provenance.get("safe") != "default"
    touched = 0
    for layer in data.get("text_layers") or []:
        if not isinstance(layer, dict):
            continue
        changed = False
        for field_name, value in (("style", resolved.caption_style),
                                  ("position", resolved.caption_position),
                                  ("size", resolved.caption_size)):
            if resolved.provenance.get(
                    {"style": "caption_style",
                     "position": "caption_position",
                     "size": "caption_size"}[field_name]) != "default" \
                    and layer.get(field_name) == defaults[field_name] \
                    and layer.get(field_name) != value:
                layer[field_name] = value
                changed = True
        if safe_declared:
            margin = int(round(resolved.safe["bottom"] * resolved.height))
            if margin > int(layer.get("margin_v", defaults["margin_v"])):
                layer["margin_v"] = margin
                changed = True
        touched += 1 if changed else 0
    record = {
        "format": resolved.name,
        "canvas": {"width": resolved.width, "height": resolved.height,
                   "fps": resolved.fps},
        "captions_applied": touched,
        "grade_pack": resolved.grade_pack,
        "loudness": resolved.loudness,
        "design_pack": resolved.design_pack,
        "provenance": resolved.provenance,
    }
    if renderer3d_record is not None:
        record["renderer3d"] = renderer3d_record
    out = EditSpec.from_dict(data) if was_spec else data
    return out, record


# ---------------------------------------------------------------------------
# Extension — formats are data the owner can add (niches scaffold idiom)
# ---------------------------------------------------------------------------

_SCAFFOLD_HELP = {
    "name": "lowercase slug (a-z0-9_) — required",
    "label": "one line saying what this format is for — required",
    "extends": "another format to inherit from, or null",
    "width/height/fps": "canvas; width and height together, or null",
    "safe": '{"left": 0.05, "right": 0.05, "top": 0.05, '
            '"bottom": 0.10} — fractions of the frame, or null',
    "caption_style": "hormozi | karaoke | minimal | mrbeast, or null",
    "caption_position/size/wpm": "caption defaults, or null",
    "grade_pack": '{"style": "phonk", "spec_overrides": {}} '
                  "— a pure_render pack, or null",
    "audio_platform": "a platform loudness norm (youtube, tiktok, "
                      "instagram, …), or null",
    "target_lufs": "explicit loudness target in [-70, 0], or null",
    "design_pack": "a director style pack (neutral, naturalist, noir, "
                   "neon, golden), or null",
    "renderer": "engine (camera3d declares the §1.5e seam: resolves "
                "through the scene3d registry at apply, refuses "
                "while no rung serves)",
}


def scaffold_format(name: str, *, dest: str | Path | None = None,
                    label: str = "",
                    overwrite: bool = False) -> str:
    """Write a ``<name>.format.json`` skeleton the owner can fill in.

    The skeleton is data: edit the TODOs, then load it with
    :func:`load_format_file` (or point ``NM_FORMAT_DIRS`` at its
    directory). Validation runs at load — a half-filled file fails
    loudly there, naming each missing piece.
    """
    key = str(name or "").strip().lower()
    if not _NAME_RE.match(key):
        raise FormatError(f"format name must be a lowercase slug, "
                          f"got {name!r}")
    path = Path(dest) if dest else Path(f"{key}.format.json")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists "
                              f"(pass overwrite=True)")
    data: dict[str, Any] = {
        "_help": _SCAFFOLD_HELP,
        "name": key,
        "label": label or f"TODO: what {key} is for",
        "extends": "portrait",
        "width": None, "height": None, "fps": None,
        "safe": None,
        "caption_style": None, "caption_position": None,
        "caption_size": None, "caption_wpm": None,
        "grade_pack": None,
        "audio_platform": None, "target_lufs": None,
        "design_pack": None,
        "renderer": "engine",
        "source": "owner format file",
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(path)


def load_format_file(path: str | Path) -> FormatSpec:
    """Load + register a format from a JSON data file."""
    p = Path(path)
    try:
        data = json.loads(p.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormatError(f"could not read format file {p}: {exc}") from exc
    return register_format(FormatSpec.from_dict(data))
