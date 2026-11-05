"""MotionProject — the motion document (Echelon Wave 4 · plan §2.3).

EffectCraft's property tree over Devon's procedural frames. Until
this module, a motion-studio production was a *call* —
``list[montage.Segment]`` into ``assemble()``, moves as function
arguments — nothing that could be saved, diffed, resumed, or handed
to the brain as data. A :class:`MotionProject` is that missing
document, in the D1 idiom Wave 1 set (the canonical
:class:`~nomorals.media.edit_engine.document.Document`'s shape:
JSON-pure dicts, content hash, ``seal()``, save/load like
``EditStudio`` projects):

* **segments** with stable ``@uid`` identity (EffectCraft's ``@uid``);
* **path-addressed properties** (``transform/zoom``,
  ``effects/blur`` — EffectCraft's ``transform/position``,
  ``effects/#1/blurriness`` shape, Devon-flattened) whose values are
  scalars or :class:`PropertyTrack` keyframes **on the frame grid**
  (1.2a discipline: keys are integer frames at the project's fps;
  seconds exist only where the engine's API demands them);
* **expressions** = the keyframe/easing layer (plan §2.3b): property
  values are sampled through ``directed/motion_score``'s easing
  library — the same single source of truth the pose compiler and
  ``_core.ease`` already use. **Scripting** = the brain driving
  ToolSpecs (P1); no embedded JS runtime is ported (the §2.3b
  decision row, recorded in ``nomorals/media/PARITY.md``).

Compilation (``to_document``) is total — every authored property
lands on exactly one rung, named in the compile record:

* **engine paths** (``transform/zoom``, ``opacity``) become keyframe
  tracks on the canonical :class:`Document` and composite inside
  ``render_document`` (1.2c). Key times are FrameGrid-exact.
  Easings the engine's sampler does not speak are *baked* —
  subdivided into linear keys sampled from the true curve — never
  silently re-eased to smooth.
* **procedural paths** (``transform/cx``/``cy``/``rotation``,
  ``effects/blur``, the motion brush) have no engine rung, so the
  segment materializes through this package's CPU tier (the
  always-rung), sampling exactly those paths per frame; the result
  is an ordinary clip in the Document.
* **text segments** materialize through :mod:`.typography`, whose
  words come from :func:`typography.to_words` — measured timings
  (e.g. :func:`edit_engine.text.captions_from_audio` word dicts)
  land on their measured frames with provenance recorded;
  plain text is estimated and *labelled* estimated (§2.4).
* **motion brush**: a segment's :class:`MaskRef` anchors
  ``brush/zoom`` — the masked region of the picture takes the brush
  zoom while the rest follows the segment transform; the compile
  record names the mask ladder rung that produced the mask (1.1d).

Deferred, named (never silently shrunk): ``camera3d/*`` property
paths are *carried* by the document but refuse at compile — the 3D
rung is the owner's open choice (plan §1.5e/§6.3), and this module
will not fake a renderer for it. Procedural paths on video/text/
visualizer segments and the code-driven explainer family (§1.5d)
likewise refuse with their reason named in the error.

Rendering (``render``) goes through
:func:`edit_engine.render.render_document`; the result carries the
engine's per-stage ``rungs`` honesty record with the compile's
per-segment stages prepended, and the on-disk ledger entry gains
the **project hash** (plan §2.3a acceptance).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment without numpy
    np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageFilter
except ImportError:  # pragma: no cover - environment without Pillow
    Image = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]

from ._core import (
    MotionStudioError,
    encode_frames,
    profile_defaults,
    probe_duration,
    record_ledger,
)
from .kenburns import MOVES, _crop_view, _load_cover, _pick_move
from .typography import timing_sources, to_words
from ..directed.motion_score import EASE_FUNCS, ease_value
from ..edit_engine.audio import AudioLayer, AudioMix
from ..edit_engine.document import Document, DocTrack
from ..edit_engine.frametime import FrameGrid
from ..edit_engine.keyframes import list_easings as engine_easings
from ..edit_engine.render import render_document
from ..edit_engine.timeline import Clip, Effect
from ...core.logging_setup import get_logger
from ...media_edit.videos import MediaEditError

_log = get_logger(__name__)

__all__ = [
    "MOTION_PROPERTIES",
    "MOTION_TRANSITIONS",
    "SEGMENT_KINDS",
    "PropKey",
    "PropertyTrack",
    "MotionSegment",
    "MotionProject",
    "lyric_project",
    "slideshow_project",
    "trailer_project",
    "visualizer_project",
]

_DOC_KIND = "devon-motion-project"
_DOC_VERSION = 1

SEGMENT_KINDS = ("video", "image", "text", "procedural")


# ---------------------------------------------------------------------------
# the property registry (1.3a: every animatable thing is a Property
# at a path; a path outside the registry is a build error, never a
# silent no-op — the KEYFRAMABLE_PARAMS doctrine, motion-shaped)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PropSpec:
    default: float
    lo: float
    hi: float
    engine: tuple[str, str] | None    # (effect, param) on the Document
    procedural: bool                  # the CPU materializer can sample it
    doc: str


MOTION_PROPERTIES: dict[str, _PropSpec] = {
    "transform/zoom": _PropSpec(
        1.0, 0.05, 20.0, ("zoom", "z"), True,
        "view zoom factor (1.0 = cover-fit); engine: zoom.z keyframes, "
        "or sampled by the materializer about the animated centre "
        "when the segment materializes"),
    "transform/cx": _PropSpec(
        0.5, 0.0, 1.0, None, True,
        "view centre x, 0..1 of the source (pan); procedural only — "
        "the engine's zoom template crops about the centre"),
    "transform/cy": _PropSpec(
        0.5, 0.0, 1.0, None, True,
        "view centre y, 0..1 of the source (tilt); procedural only"),
    "transform/rotation": _PropSpec(
        0.0, -3600.0, 3600.0, None, True,
        "view rotation in degrees, counter-clockwise; procedural only"),
    "opacity": _PropSpec(
        1.0, 0.0, 1.0, ("opacity", "alpha"), False,
        "segment opacity; engine: opacity.alpha keyframes (alpha "
        "over the program's black base, applied by the renderer)"),
    "effects/blur": _PropSpec(
        0.0, 0.0, 100.0, None, True,
        "gaussian blur radius in px (EffectCraft's "
        "effects/#1/blurriness); procedural only"),
    "brush/zoom": _PropSpec(
        1.0, 0.05, 20.0, None, True,
        "motion-brush zoom: applies INSIDE the segment's MaskRef "
        "region only; requires a mask; procedural only"),
}

#: property-path prefix carried but never rendered in this wave —
#: the 3D rung is the owner's open choice (plan §1.5e/§6.3).
DEFERRED_PREFIX = "camera3d/"

#: segment transition name → the session-directive type
#: :func:`render_document` interprets (``None`` = no engine rung —
#: a compile error, named, never an approximation).
MOTION_TRANSITIONS: dict[str, str | None] = {
    "cut": "cut",
    "crossfade": "fade",
    "dissolve": "dissolve",
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "wipeleft": None,
    "circleopen": None,
}


def _check_path(path: str) -> _PropSpec | None:
    """Registry lookup; ``None`` for a deferred (carry-only) path."""
    if path.startswith(DEFERRED_PREFIX):
        return None
    spec = MOTION_PROPERTIES.get(path)
    if spec is None:
        raise MotionStudioError(
            f"unknown motion property {path!r} — registry: "
            f"{', '.join(sorted(MOTION_PROPERTIES))} "
            f"(+ {DEFERRED_PREFIX}* carried, deferred)")
    return spec


def _check_value(path: str, value: float) -> float:
    spec = _check_path(path)
    v = float(value)
    if spec is not None and not (spec.lo <= v <= spec.hi):
        raise MotionStudioError(
            f"property {path!r} value {v} out of range "
            f"[{spec.lo}, {spec.hi}]")
    return v


# ---------------------------------------------------------------------------
# keyframes on the frame grid
# ---------------------------------------------------------------------------

@dataclass
class PropKey:
    """One keyframe: at integer ``frame``, the value is ``value``.

    ``easing`` names the curve INTO this key from the previous one
    (the KeyframeTrack convention), drawn from
    ``directed/motion_score``'s easing library — the expression
    layer's vocabulary (plan §2.3b).
    """
    frame: int
    value: float
    easing: str = "smooth"

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or int(self.frame) != self.frame:
            raise MotionStudioError(
                f"keyframe frame must be an integer frame count, "
                f"got {self.frame!r}")
        self.frame = int(self.frame)
        if self.frame < 0:
            raise MotionStudioError("keyframe frame must be >= 0")
        self.value = float(self.value)
        self.easing = str(self.easing or "smooth").lower()
        if self.easing not in EASE_FUNCS:
            raise MotionStudioError(
                f"unknown easing {self.easing!r} — the expression "
                f"layer speaks: {', '.join(sorted(EASE_FUNCS))}")

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "value": self.value,
                "easing": self.easing}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PropKey":
        return cls(frame=d["frame"], value=d["value"],
                   easing=str(d.get("easing", "smooth")))


@dataclass
class PropertyTrack:
    """Keyframes for one property path, addressed in frames.

    Sampling is the expression layer evaluating the track: between
    keys, progress is eased by the destination key's easing through
    :func:`directed.motion_score.ease_value`; outside the key range
    the end values hold (the KeyframeTrack.sample contract).
    """
    path: str
    keys: list[PropKey] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_path(self.path)
        self.keys = [k if isinstance(k, PropKey) else PropKey.from_dict(k)
                     for k in self.keys]
        if not self.keys:
            raise MotionStudioError(
                f"property track {self.path!r} has no keyframes")
        for k in self.keys:
            _check_value(self.path, k.value)
        self.keys.sort(key=lambda k: k.frame)

    def sample(self, frame: float) -> float:
        """The expression's value at ``frame`` (floats interpolate)."""
        ks = self.keys
        f = float(frame)
        if f <= ks[0].frame:
            return ks[0].value
        if f >= ks[-1].frame:
            return ks[-1].value
        for prev, nxt in zip(ks, ks[1:]):
            if prev.frame <= f <= nxt.frame:
                span = nxt.frame - prev.frame
                u = 0.0 if span <= 0 else (f - prev.frame) / span
                e = ease_value(nxt.easing, u)
                return prev.value + (nxt.value - prev.value) * e
        return ks[-1].value

    def sample_at(self, grid: FrameGrid, frame: int) -> float:
        """Value at an exact grid frame — the value the renderer
        sees for that frame (sampling is frame-space, so this is
        :meth:`sample` at the integer frame; the grid pins *which*
        frames exist)."""
        return self.sample(float(frame))

    def to_dict(self) -> dict[str, Any]:
        return {"keys": [k.to_dict() for k in self.keys]}

    @classmethod
    def from_dict(cls, path: str, d: dict[str, Any]) -> "PropertyTrack":
        return cls(path=path,
                   keys=[PropKey.from_dict(k) for k in d.get("keys", [])])


def _prop_from_value(path: str, value: Any) -> float | PropertyTrack:
    """Authored property value → float | PropertyTrack (validated)."""
    if isinstance(value, PropertyTrack):
        if value.path != path:
            raise MotionStudioError(
                f"track for {value.path!r} stored under {path!r}")
        return value
    if isinstance(value, dict):
        return PropertyTrack.from_dict(path, value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _check_value(path, float(value))
    raise MotionStudioError(
        f"property {path!r}: value must be a number or a keyframe "
        f"track dict, got {type(value).__name__}")


def _prop_to_value(value: float | PropertyTrack) -> Any:
    if isinstance(value, PropertyTrack):
        return value.to_dict()
    return float(value)


def _prop_sample(value: float | PropertyTrack, frame: float,
                 default: float) -> float:
    if isinstance(value, PropertyTrack):
        return value.sample(frame)
    return float(value)


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------

@dataclass
class MotionSegment:
    """One timed span of a MotionProject (the canonical Scene's
    motion-shaped sibling: a source, a duration, property paths).

    ``kind``: ``video`` | ``image`` | ``text`` | ``procedural``.
    Text segments carry ``text`` (plain lyrics — timings estimated,
    labelled) or ``words`` (timed word dicts, segment-local seconds —
    measured when they come from a transcriber; label them via
    ``timing_source`` or per-word ``timing_source`` keys), plus a
    typography ``preset``. Procedural segments name their engine in
    ``design["engine"]`` (``"visualizer"`` ships; the explainer
    family is §1.5d, deferred and named at compile).
    """
    uid: str
    kind: str
    duration_s: float
    src: str = ""
    text: str = ""
    start: float = 0.0
    transition: str = ""
    transition_duration: float = 0.6
    preset: str = ""
    properties: dict[str, float | PropertyTrack] = field(
        default_factory=dict)
    mask: Any = None                    # MaskRef | dict | None
    words: list[dict[str, Any]] = field(default_factory=list)
    audio: str = ""
    timing_source: str = ""
    design: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SEGMENT_KINDS:
            raise MotionStudioError(
                f"segment kind must be one of {SEGMENT_KINDS}, "
                f"got {self.kind!r}")
        self.duration_s = float(self.duration_s)
        if self.duration_s <= 0:
            raise MotionStudioError("segment duration_s must be > 0")
        self.start = max(0.0, float(self.start))
        if self.transition and self.transition not in MOTION_TRANSITIONS:
            raise MotionStudioError(
                f"unknown transition {self.transition!r} — pick "
                f"from: {', '.join(sorted(MOTION_TRANSITIONS))}")
        props: dict[str, float | PropertyTrack] = {}
        for path, value in (self.properties or {}).items():
            props[str(path)] = _prop_from_value(str(path), value)
        self.properties = props
        if self.mask is not None and not isinstance(self.mask, dict):
            to_dict = getattr(self.mask, "to_dict", None)
            if to_dict is None:
                raise MotionStudioError(
                    "segment mask must be a MaskRef (or its dict)")
            self.mask = to_dict()
        self.words = [dict(w) for w in self.words]
        self.design = copy.deepcopy(self.design or {})
        self.extras = copy.deepcopy(self.extras or {})

    # -- properties ----------------------------------------------------
    def get(self, path: str) -> float | PropertyTrack | None:
        return self.properties.get(path)

    def sample(self, path: str, frame: float) -> float:
        """Property value at ``frame`` (registry default when unset)."""
        spec = _check_path(path)
        if spec is None:
            raise MotionStudioError(
                f"property {path!r} is deferred (camera3d) — it is "
                f"carried, never sampled")
        value = self.properties.get(path)
        if value is None:
            return spec.default
        return _prop_sample(value, frame, spec.default)

    def animated_paths(self) -> list[str]:
        return sorted(p for p, v in self.properties.items()
                      if isinstance(v, PropertyTrack))

    # -- identity --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid, "kind": self.kind, "src": self.src,
            "text": self.text, "duration_s": self.duration_s,
            "start": self.start, "transition": self.transition,
            "transition_duration": self.transition_duration,
            "preset": self.preset,
            "properties": {p: _prop_to_value(v)
                           for p, v in self.properties.items()},
            "mask": copy.deepcopy(self.mask),
            "words": copy.deepcopy(self.words),
            "audio": self.audio, "timing_source": self.timing_source,
            "design": copy.deepcopy(self.design),
            "extras": copy.deepcopy(self.extras),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MotionSegment":
        return cls(
            uid=str(d.get("uid", "")), kind=str(d.get("kind", "video")),
            duration_s=float(d.get("duration_s", 0.0)),
            src=str(d.get("src", "")), text=str(d.get("text", "")),
            start=float(d.get("start", 0.0)),
            transition=str(d.get("transition", "")),
            transition_duration=float(d.get("transition_duration", 0.6)),
            preset=str(d.get("preset", "")),
            properties=dict(d.get("properties") or {}),
            mask=copy.deepcopy(d.get("mask")),
            words=[dict(w) for w in d.get("words") or []],
            audio=str(d.get("audio", "")),
            timing_source=str(d.get("timing_source", "")),
            design=copy.deepcopy(d.get("design") or {}),
            extras=copy.deepcopy(d.get("extras") or {}))


# ---------------------------------------------------------------------------
# the project document
# ---------------------------------------------------------------------------

def _canonical(value: Any) -> Any:
    """Document._canonical's discipline, mirrored (sorted keys,
    JSON primitives, to_dict-able values) for the project hash."""
    if isinstance(value, dict):
        return {str(k): _canonical(value[k])
                for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    raise MotionStudioError(
        f"not JSON-serializable in MotionProject: "
        f"{type(value).__name__}")


@dataclass
class MotionProject:
    """A motion production as a document: renders, diffs, resumes.

    The D1 idiom, mirrored from :class:`edit_engine.Document` and
    :class:`imggen.ImageDocument`: pure-JSON ``to_dict``/``from_dict``
    round-trip, ``compute_hash``/``seal()`` identity (EffectCraft's
    snapshot identity: a saved project's hash *is* its modified-ness
    test — :meth:`is_dirty`), save/load like ``EditStudio``
    projects.
    """
    title: str = ""
    segments: list[MotionSegment] = field(default_factory=list)
    audio: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    seed: int = 21
    version: int = 1
    doc_hash: str = ""

    def __post_init__(self) -> None:
        self.segments = [s if isinstance(s, MotionSegment)
                         else MotionSegment.from_dict(dict(s))
                         for s in self.segments]
        seen: set[str] = set()
        for i, seg in enumerate(self.segments):
            if not seg.uid:
                seg.uid = f"s{i + 1}"
            if seg.uid in seen:
                raise MotionStudioError(
                    f"duplicate segment uid {seg.uid!r} — @uid "
                    f"identity must be unique in a project")
            seen.add(seg.uid)
        self.output = dict(self.output or {})
        defaults = profile_defaults()
        self.output.setdefault("width", int(defaults["size"][0]))
        self.output.setdefault("height", int(defaults["size"][1]))
        self.output.setdefault("fps", defaults["fps"])
        self.output.setdefault("profile", "")
        self.version = int(self.version)

    # -- identity ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"kind": _DOC_KIND, "version": _DOC_VERSION,
                "title": self.title,
                "segments": [s.to_dict() for s in self.segments],
                "audio": self.audio, "output": dict(self.output),
                "seed": self.seed, "doc_version": self.version,
                "doc_hash": self.doc_hash}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionProject":
        if not isinstance(data, dict):
            raise MotionStudioError("motion project must be a dict")
        if data.get("kind") not in (None, _DOC_KIND):
            raise MotionStudioError(
                f"not a motion project (kind={data.get('kind')!r})")
        if int(data.get("version", _DOC_VERSION)) != _DOC_VERSION:
            raise MotionStudioError(
                f"unsupported motion project version "
                f"{data.get('version')}")
        return cls(title=str(data.get("title", "")),
                   segments=[MotionSegment.from_dict(s)
                             for s in data.get("segments") or []],
                   audio=str(data.get("audio", "")),
                   output=dict(data.get("output") or {}),
                   seed=int(data.get("seed", 21)),
                   version=int(data.get("doc_version", 1)),
                   doc_hash=str(data.get("doc_hash", "")))

    def compute_hash(self) -> str:
        """SHA-256 of the canonical project (hash field excluded)."""
        raw = self.to_dict()
        raw.pop("doc_hash", None)
        blob = json.dumps(_canonical(raw), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def seal(self) -> "MotionProject":
        """Bump version + stamp hash. Call after any edit."""
        self.version = int(self.version) + 1
        self.doc_hash = self.compute_hash()
        return self

    def is_dirty(self) -> bool:
        """True when the content differs from the sealed snapshot
        (a never-sealed project with content is dirty by definition
        only once it carries a hash; a fresh unsealed project is
        simply unsealed, reported False until first seal/save)."""
        if not self.doc_hash:
            return False
        return self.compute_hash() != self.doc_hash

    def copy(self) -> "MotionProject":
        return MotionProject.from_dict(self.to_dict())

    def save(self, path: str | os.PathLike) -> str:
        """Save like EditStudio projects: one JSON file. Seals first
        so the file carries the snapshot identity it was saved at."""
        self.seal()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2),
                     encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "MotionProject":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise MotionStudioError(
                f"could not load motion project {path}: {exc}") from exc
        return cls.from_dict(data)

    # -- property tree ---------------------------------------------------
    def segment(self, uid: str) -> MotionSegment:
        for seg in self.segments:
            if seg.uid == uid:
                return seg
        raise MotionStudioError(
            f"no segment @{uid} — project has: "
            f"{', '.join(s.uid for s in self.segments)}")

    def property_paths(self) -> list[str]:
        """Every addressed property: ``segments/<uid>/<path>``."""
        out: list[str] = []
        for seg in self.segments:
            for path in sorted(seg.properties):
                out.append(f"segments/{seg.uid}/{path}")
        return out

    def get_property(self, uid: str, path: str) -> Any:
        _check_path(path)
        return self.segment(uid).properties.get(path)

    def set_property(self, uid: str, path: str,
                     value: Any) -> "MotionProject":
        """Set a scalar or keyframe track on a segment's property.

        Validated against the registry (unknown path / out-of-range
        value / unknown easing all raise here, at authoring time —
        not at render). Seal afterwards to stamp the new snapshot.
        """
        seg = self.segment(uid)
        seg.properties[path] = _prop_from_value(path, value)
        return self

    @property
    def duration(self) -> float:
        return sum(s.duration_s for s in self.segments)

    @property
    def grid(self) -> FrameGrid:
        return FrameGrid(self.output.get("fps", 30))

    # -- compilation -----------------------------------------------------
    def to_document(self, *, workdir: str | os.PathLike | None = None
                    ) -> tuple[Document, dict[str, Any]]:
        """Compile to the canonical :class:`Document` + the record.

        The record states, per segment, which tier each property
        landed on and why — the honesty surface ``render`` merges
        into the engine's rungs. Materialized segments are written
        under ``workdir`` (created when omitted).
        """
        if not self.segments:
            raise MotionStudioError("empty project — nothing to compile")
        tmp = Path(workdir) if workdir else Path(
            tempfile.mkdtemp(prefix="motionproj-"))
        tmp.mkdir(parents=True, exist_ok=True)
        grid = self.grid
        width = int(self.output.get("width", 1080))
        height = int(self.output.get("height", 1920))
        size = (width, height)
        fps = grid.fps_float

        clips: list[Clip] = []
        keyframes: list[dict[str, Any]] = []
        seg_records: list[dict[str, Any]] = []
        rungs: list[dict[str, Any]] = []
        at = 0.0
        # transitions ride render_document's session directives:
        # declared transitions map onto clip boundaries IN ORDER, so
        # once any segment declares a non-cut transition, every
        # following clip carries its directive (cut = explicit cut).
        any_transition = any(
            s.transition and s.transition != "cut"
            for s in self.segments[1:])

        for idx, seg in enumerate(self.segments):
            deferred = [p for p in seg.properties
                        if p.startswith(DEFERRED_PREFIX)]
            if deferred:
                raise MotionStudioError(
                    f"segment @{seg.uid}: {', '.join(sorted(deferred))} "
                    f"— the 3D rung is not chosen (plan §1.5e: the "
                    f"owner picks the engine; the document carries "
                    f"these paths but no renderer fakes them)")
            rec, clip, seg_kf = _compile_segment(
                seg, grid=grid, size=size, fps=fps, tmp=tmp,
                seed=self.seed, project_audio=self.audio)
            clip.at = round(at, 6)
            if idx > 0 and any_transition:
                tname = seg.transition or "cut"
                ttype = MOTION_TRANSITIONS[tname]
                if ttype is None:
                    raise MotionStudioError(
                        f"segment @{seg.uid}: transition {tname!r} "
                        f"has no engine rung (session directives "
                        f"interpret cut/fade/dissolve/flash/dip/"
                        f"whip_pan; wipe/circle kinds are not among "
                        f"them) — pick a mapped transition")
                clip.effects = tuple(clip.effects) + (Effect(
                    name="v_transition",
                    params={"type": ttype,
                            "duration": float(seg.transition_duration)}),)
                rec["transition"] = (
                    f"{tname} → session {ttype} "
                    f"{float(seg.transition_duration):.2f}s")
            clips.append(clip)
            keyframes.extend(seg_kf)
            rec["at"] = round(at, 6)
            seg_records.append(rec)
            rungs.append({
                "stage": f"motion:{seg.uid}",
                "rung": rec["rung"],
                "note": rec["note"]})
            at += seg.duration_s

        audio_mix = None
        if self.audio:
            audio_mix = AudioMix(layers=[AudioLayer(path=self.audio)])
        doc = Document(
            tracks=[DocTrack(kind="video", clips=clips)],
            audio_mix=audio_mix,
            keyframes=keyframes,
            output={"width": width, "height": height,
                    "fps": self.output.get("fps", 30),
                    "profile": str(self.output.get("profile") or "")
                    or "workstation"},
            title=self.title or "motion project")
        doc.seal()
        record = {
            "project_hash": self.doc_hash or self.compute_hash(),
            "title": self.title,
            "segments": seg_records,
            "rungs": rungs,
            "workdir": str(tmp),
            "document_hash": doc.doc_hash,
        }
        return doc, record

    # -- rendering ---------------------------------------------------------
    def render(self, out: str | os.PathLike | None = None, *,
               host: Any = None,
               workdir: str | os.PathLike | None = None) -> dict[str, Any]:
        """Compile + render through ``render_document``.

        The result is the engine's report enriched: ``rungs`` =
        the compile's per-segment stages followed by the engine's
        own per-stage record; ``motion_project`` = the compile
        record (project hash, per-segment tiers). The on-disk
        ledger entry carries the project hash (plan §2.3a).
        """
        tmp = Path(workdir) if workdir else Path(
            tempfile.mkdtemp(prefix="motionproj-render-"))
        tmp.mkdir(parents=True, exist_ok=True)
        doc, record = self.to_document(workdir=tmp / "compile")
        res = render_document(doc, out=out, host=host,
                              workdir=tmp / "render")
        res["rungs"] = record["rungs"] + list(res.get("rungs", []))
        res["motion_project"] = record
        record_ledger({"kind": "motion_project",
                       "path": res["output"],
                       "project_hash": record["project_hash"],
                       "title": self.title,
                       "segments": len(self.segments),
                       "duration": res.get("duration")})
        return res


# ---------------------------------------------------------------------------
# segment compilation
# ---------------------------------------------------------------------------

def _engine_keyframes(seg: MotionSegment, path: str,
                      grid: FrameGrid) -> dict[str, Any] | None:
    """One engine-mapped property → a Document keyframe dict.

    Times are FrameGrid-exact (``float(grid.seconds(frame))``).
    Only easings whose curve is IDENTICAL in both vocabularies pass
    through (``linear``, ``smooth``/``smoothstep`` — smoothstep in
    both). The rest are BAKED — the span is subdivided into linear
    keys sampled from the true curve — including ``ease_in`` /
    ``ease_out`` / ``ease_in_out``: the engine's keyframe sampler
    gives those names QUADRATIC curves while the expression layer
    (``motion_score``) means CUBIC ones, so passing the name through
    would silently re-ease the animation. Baking makes the rendered
    curve the authored curve (the engine's own ``_subdivided``
    precedent, generalized).
    """
    spec = MOTION_PROPERTIES[path]
    assert spec.engine is not None
    value = seg.properties.get(path)
    if value is None:
        return None
    effect, param = spec.engine
    if not isinstance(value, PropertyTrack):
        if float(value) == spec.default:
            return None  # identity value — no track needed
        return {"clip_id": seg.uid, "effect": effect, "param": param,
                "keys": [{"time": 0.0, "value": float(value),
                          "easing": "linear"}]}
    # pass-through only where the CURVE is the same in the engine's
    # table (verified: linear, smooth — not merely the same name)
    native = {"linear", "smooth"}
    assert native <= set(engine_easings())
    keys: list[dict[str, Any]] = []
    ks = value.keys

    def _emit(frame: Fraction | int | float, val: float,
              easing: str) -> None:
        t = float(Fraction(frame) / grid.fps)
        if keys and abs(keys[-1]["time"] - t) < 1e-9:
            keys[-1] = {"time": t, "value": float(val), "easing": easing}
        else:
            keys.append({"time": t, "value": float(val),
                         "easing": easing})

    _emit(ks[0].frame, ks[0].value, "linear")
    for prev, nxt in zip(ks, ks[1:]):
        easing = nxt.easing
        mapped = "smooth" if easing == "smoothstep" else easing
        if mapped in native:
            _emit(nxt.frame, nxt.value, mapped)
            continue
        steps = 8
        for j in range(1, steps + 1):
            f = Fraction(prev.frame) + Fraction(
                nxt.frame - prev.frame) * Fraction(j, steps)
            _emit(f, value.sample(float(f)), "linear")
    return {"clip_id": seg.uid, "effect": effect, "param": param,
            "keys": keys}


def _compile_segment(seg: MotionSegment, *, grid: FrameGrid,
                     size: tuple[int, int], fps: float, tmp: Path,
                     seed: int, project_audio: str
                     ) -> tuple[dict[str, Any], Clip, list[dict[str, Any]]]:
    """Compile one segment → (record, clip, engine keyframe dicts)."""
    rec: dict[str, Any] = {"uid": seg.uid, "kind": seg.kind,
                           "engine_paths": [], "procedural_paths": [],
                           "tier": "engine", "note": ""}
    # Identity values need no rung: a scalar sitting at its registry
    # default is the un-animated value, whichever tier renders the
    # segment — it neither forces materialization nor errors.
    props = {p: v for p, v in seg.properties.items()
             if p not in MOTION_PROPERTIES
             or isinstance(v, PropertyTrack)
             or float(v) != MOTION_PROPERTIES[p].default}
    procedural = [p for p in props
                    if p in MOTION_PROPERTIES
                    and MOTION_PROPERTIES[p].procedural
                    and p not in ("transform/zoom",)]
    # transform/zoom is dual-rung: the materializer samples it (about
    # the animated centre) when the segment materializes anyway;
    # otherwise it rides the engine. opacity is engine-only.
    engine_paths = [p for p in props
                    if p in MOTION_PROPERTIES
                    and MOTION_PROPERTIES[p].engine is not None]

    if seg.kind == "video":
        if procedural:
            raise MotionStudioError(
                f"segment @{seg.uid}: {', '.join(sorted(procedural))} "
                f"have no rung on video segments in this wave — the "
                f"procedural tier materializes stills/text, while "
                f"moving footage animates through the engine's "
                f"keyframe registry (transform/zoom, opacity). "
                f"Named remainder, not a silent drop.")
        if not seg.src or not Path(seg.src).exists():
            raise MotionStudioError(
                f"segment @{seg.uid}: video source not found: "
                f"{seg.src!r}")
        clip = Clip(path=seg.src, start=seg.start,
                    end=seg.start + seg.duration_s,
                    duration=seg.duration_s, id=seg.uid)
        rec["rung"] = "source clip — edit_engine.render_document"
        rec["note"] = "video segment; properties on engine keyframes"

    elif seg.kind == "image":
        if not seg.src or not Path(seg.src).exists():
            raise MotionStudioError(
                f"segment @{seg.uid}: image source not found: "
                f"{seg.src!r}")
        wants_brush = "brush/zoom" in props
        if wants_brush and seg.mask is None:
            raise MotionStudioError(
                f"segment @{seg.uid}: brush/zoom needs a mask — "
                f"the motion brush is anchored by a MaskRef")
        if procedural or wants_brush:
            out = tmp / f"seg-{seg.uid}.mp4"
            mat = _materialize_image_segment(
                seg, out=out, grid=grid, size=size, fps=fps, seed=seed)
            rec.update(mat)
            rec["tier"] = "procedural"
            rec["rung"] = ("procedural frames — "
                           "motion_studio.project materializer "
                           "(CPU tier)")
            clip = Clip(path=str(out), duration=seg.duration_s,
                        id=seg.uid)
            # zoom was sampled by the materializer (about the
            # animated centre); it must not ALSO ride the engine.
            engine_paths = [p for p in engine_paths
                            if p != "transform/zoom"]
            sampled = set(procedural)
            if "transform/zoom" in props:
                sampled.add("transform/zoom")
            rec["procedural_paths"] = sorted(sampled)
        else:
            clip = Clip(path=seg.src, duration=seg.duration_s,
                        id=seg.uid)
            rec["rung"] = ("still clip — edit_engine "
                           "render_document (stills materialize in "
                           "the engine)")
            rec["note"] = "image segment; properties on engine keyframes"

    elif seg.kind == "text":
        bad = [p for p in procedural]
        if bad:
            raise MotionStudioError(
                f"segment @{seg.uid}: {', '.join(sorted(bad))} have "
                f"no rung on text segments in this wave — kinetic "
                f"type materializes through typography, which samples "
                f"no transform paths; engine paths (transform/zoom, "
                f"opacity) still apply to the rendered clip.")
        words = _segment_words(seg, project_audio)
        from .typography import render_lyrics
        out = tmp / f"seg-{seg.uid}.mp4"
        render_lyrics(words, out, duration=seg.duration_s,
                      preset=seg.preset or "neon_pop",
                      size=size, fps=fps, seed=seed)
        srcs = timing_sources(words)
        rec["tier"] = "procedural"
        rec["rung"] = ("kinetic typography — "
                       "motion_studio.typography.render_lyrics")
        rec["timing_sources"] = srcs
        rec["note"] = ("text segment; word timing: " + ", ".join(
            f"{k}×{v}" for k, v in sorted(srcs.items())))
        clip = Clip(path=str(out), duration=seg.duration_s, id=seg.uid)

    elif seg.kind == "procedural":
        bad = [p for p in procedural]
        if bad:
            raise MotionStudioError(
                f"segment @{seg.uid}: {', '.join(sorted(bad))} have "
                f"no rung on procedural segments — the procedural "
                f"engines render their own frames; segment property "
                f"paths do not reach inside them in this wave.")
        engine_name = str(seg.design.get("engine", ""))
        if engine_name != "visualizer":
            raise MotionStudioError(
                f"segment @{seg.uid}: procedural engine "
                f"{engine_name!r} is not built — 'visualizer' ships "
                f"in this wave; the code-driven explainer family "
                f"(diagrams, equations-as-text, chart builds) is "
                f"plan §1.5d, a named deferral, not a stub")
        audio = seg.src or project_audio
        if not audio or not Path(audio).exists():
            raise MotionStudioError(
                f"segment @{seg.uid}: visualizer needs an audio "
                f"source (segment src or project audio)")
        from .visualizer import render_visualizer
        out = tmp / f"seg-{seg.uid}.mp4"
        render_visualizer(audio, out,
                          preset=seg.preset or "phonk",
                          images=[str(i) for i in
                                  seg.design.get("images", [])],
                          duration=seg.duration_s, size=size, fps=fps)
        rec["tier"] = "procedural"
        rec["rung"] = ("audio-reactive visualizer — "
                       "motion_studio.visualizer.render_visualizer")
        rec["note"] = "procedural segment (visualizer engine)"
        clip = Clip(path=str(out), duration=seg.duration_s, id=seg.uid)

    else:  # pragma: no cover - __post_init__ already enforces kinds
        raise MotionStudioError(f"unknown segment kind {seg.kind!r}")

    keyframes: list[dict[str, Any]] = []
    for path in sorted(engine_paths):
        kf = _engine_keyframes(seg, path, grid)
        if kf is not None:
            keyframes.append(kf)
            rec["engine_paths"].append(path)
    if seg.kind == "image" and rec["tier"] == "engine":
        rec["note"] = "image segment; properties on engine keyframes"
    return rec, clip, keyframes


def _segment_words(seg: MotionSegment, project_audio: str):
    """A text segment's timed words, provenance intact.

    Precedence: authored ``words`` (timed dicts — measured when a
    transcriber produced them and they say so) → ``text`` + audio
    (estimated, beat-snapped) → ``text`` alone (estimated). Segment
    times are segment-local.
    """
    if seg.words:
        return to_words(seg.words,
                        timing_source=seg.timing_source or "")
    if seg.text:
        return to_words(seg.text,
                        audio=seg.audio or project_audio or None,
                        timing_source=seg.timing_source or "")
    raise MotionStudioError(
        f"segment @{seg.uid}: text segment carries neither words "
        f"nor text — nothing to typeset")


def _materialize_image_segment(seg: MotionSegment, *, out: Path,
                               grid: FrameGrid, size: tuple[int, int],
                               fps: float, seed: int) -> dict[str, Any]:
    """Render an image segment's procedural paths to a clip.

    Per frame (FrameGrid-exact count): sample cx/cy/zoom/rotation/
    blur + the brush; the view is kenburns' cover/crop math; the
    brush composites a second, brush-zoomed view through the
    segment's resolved mask. Deterministic: same segment + seed →
    identical frames.
    """
    if Image is None or np is None:
        raise MotionStudioError(
            "image materialization needs numpy + Pillow")
    defaults = profile_defaults()
    supersample = int(defaults.get("supersample", 1) or 1)
    cover = _load_cover(seg.src, size, supersample)

    mask_img = None
    mask_note = ""
    if seg.mask is not None:
        from ..imggen.masks import MaskRef
        ref = MaskRef.from_dict(copy.deepcopy(seg.mask))
        native = Image.open(seg.src).convert("RGB")
        result = ref.resolve(native)
        stamped = ref.resolved(result)
        seg.mask = stamped.to_dict()
        m = result.mask
        tw, th = size[0] * supersample, size[1] * supersample
        scale = max(tw / m.width, th / m.height)
        mask_img = m.resize(
            (math.ceil(m.width * scale), math.ceil(m.height * scale)),
            Image.BILINEAR)
        mask_note = (f"motion brush mask via {result.tier} "
                     f"({result.detail})")

    n_frames = max(1, grid.frames(seg.duration_s))
    w, h = size

    def frame_fn(i: int, t: float) -> np.ndarray:
        f = float(i)
        cx = seg.sample("transform/cx", f)
        cy = seg.sample("transform/cy", f)
        zoom = seg.sample("transform/zoom", f)
        view = _crop_view(cover, cx, cy, zoom, size, supersample)
        if "brush/zoom" in seg.properties and mask_img is not None:
            bz = seg.sample("brush/zoom", f)
            inner = _crop_view(cover, cx, cy, bz, size, supersample)
            mview = _crop_view(mask_img, cx, cy, zoom, size,
                               supersample)
            view = Image.composite(inner, view, mview)
        rot = seg.sample("transform/rotation", f)
        if rot:
            view = view.rotate(rot, resample=Image.BICUBIC,
                               center=(w // 2, h // 2))
        blur = seg.sample("effects/blur", f)
        if blur > 0:
            view = view.filter(ImageFilter.GaussianBlur(blur))
        return np.asarray(view.convert("RGB"))

    encode_frames((frame_fn(i, i / fps) for i in range(n_frames)),
                  out, w, h, fps,
                  crf=int(defaults.get("crf", 20)),
                  preset=str(defaults.get("preset", "veryfast")))
    return {"frames": n_frames, "mask": mask_note,
            "note": "image segment materialized: "
                    + ", ".join(sorted(
                        p for p in seg.properties
                        if MOTION_PROPERTIES[p].procedural))
                    + (f"; {mask_note}" if mask_note else "")}


# ---------------------------------------------------------------------------
# product constructors — the four studio products as documents
# (studio.make_* remain the one-call surface; these are their
# saveable/diffable/resumable form)
# ---------------------------------------------------------------------------

def _out(size: tuple[int, int] | None, fps: float | None,
         profile: str = "") -> dict[str, Any]:
    defaults = profile_defaults()
    size = size or defaults["size"]  # type: ignore[assignment]
    return {"width": int(size[0]), "height": int(size[1]),
            "fps": fps or defaults["fps"], "profile": profile}


def lyric_project(audio: str | os.PathLike, lyrics: Any, *,
                  preset: str = "neon_pop", title: str = "",
                  words: Sequence[dict[str, Any]] | None = None,
                  timing_source: str = "",
                  size: tuple[int, int] | None = None,
                  fps: float | None = None, seed: int = 21
                  ) -> MotionProject:
    """A lyric video as a project: one text segment over the song.

    ``words`` (measured word dicts, e.g. from
    :func:`edit_engine.text.captions_from_audio`) take precedence
    over ``lyrics`` text and keep their provenance; plain text is
    estimated at compile, labelled as such.
    """
    audio_s = str(audio)
    if words:
        wlist = [dict(w) for w in words]
        duration = max(float(w["end"]) for w in wlist) + 1.0
        seg = MotionSegment(uid="lyrics", kind="text",
                            duration_s=duration, preset=preset,
                            words=wlist, audio=audio_s,
                            timing_source=timing_source)
    else:
        duration = probe_duration(audio_s)
        if duration <= 0:
            raise MotionStudioError(
                "lyric project needs a probeable audio duration "
                "(or measured words to size the segment)")
        seg = MotionSegment(uid="lyrics", kind="text",
                            duration_s=duration,
                            text=str(lyrics), preset=preset,
                            audio=audio_s,
                            timing_source=timing_source)
    return MotionProject(title=title or "lyric video",
                         segments=[seg], audio=audio_s,
                         output=_out(size, fps), seed=seed)


def slideshow_project(images: Sequence[str | os.PathLike], *,
                      per_image: float = 4.0, move: str = "auto",
                      transition: str = "crossfade",
                      audio: str | os.PathLike | None = None,
                      title: str = "",
                      size: tuple[int, int] | None = None,
                      fps: float | None = None, seed: int = 7
                      ) -> MotionProject:
    """A Ken Burns slideshow as a project: image segments whose
    camera moves are property tracks (the MOVES table compiled to
    keyframes — start/end centre+zoom over the segment's frames)."""
    if not images:
        raise MotionStudioError("no images — nothing to slide")
    if move != "auto" and move != "random" and move not in MOVES:
        raise MotionStudioError(
            f"unknown move {move!r} — pick from: "
            f"{', '.join(sorted(MOVES))}, auto, random")
    out = _out(size, fps)
    grid = FrameGrid(out["fps"])
    n = max(2, grid.frames(per_image))
    segments: list[MotionSegment] = []
    for i, img in enumerate(images):
        mv = _pick_move(move, seed, i)
        props: dict[str, Any] = {}
        for path, a, b in (("transform/cx", mv.start[0], mv.end[0]),
                           ("transform/cy", mv.start[1], mv.end[1]),
                           ("transform/zoom", mv.start[2], mv.end[2])):
            if a != b:
                props[path] = {"keys": [
                    {"frame": 0, "value": a, "easing": "smooth"},
                    {"frame": n, "value": b, "easing": "smooth"}]}
            else:
                props[path] = a
        segments.append(MotionSegment(
            uid=f"img{i + 1}", kind="image", src=str(img),
            duration_s=per_image, transition=transition,
            properties=props))
    return MotionProject(title=title or "slideshow",
                         segments=segments,
                         audio=str(audio) if audio else "",
                         output=out, seed=seed)


def trailer_project(clips: Sequence[str | os.PathLike], *,
                    title: str = "", tagline: str = "",
                    audio: str | os.PathLike | None = None,
                    clip_len: float = 2.2, transition: str = "cut",
                    size: tuple[int, int] | None = None,
                    fps: float | None = None, seed: int = 21
                    ) -> MotionProject:
    """A trailer cut as a project: title card → clips → tagline."""

    def _card(uid: str, text: str, dur: float,
              preset: str) -> MotionSegment:
        words = [{"text": w, "start": round(i * 0.45, 3),
                  "end": round(i * 0.45 + 0.6, 3),
                  "timing_source": "authored"}
                 for i, w in enumerate(text.split())]
        return MotionSegment(uid=uid, kind="text", duration_s=dur,
                             words=words, preset=preset,
                             transition="fadeblack",
                             transition_duration=0.5)

    if not clips:
        raise MotionStudioError("no clips — nothing to cut")
    segments: list[MotionSegment] = []
    if title:
        segments.append(_card("title", title, 2.4, "bold_statement"))
    for i, c in enumerate(clips):
        segments.append(MotionSegment(
            uid=f"clip{i + 1}", kind="video", src=str(c),
            duration_s=clip_len, transition=transition,
            transition_duration=0.35))
    if tagline:
        segments.append(_card("tagline", tagline, 2.6, "neon_pop"))
    return MotionProject(title=title or "trailer", segments=segments,
                         audio=str(audio) if audio else "",
                         output=_out(size, fps), seed=seed)


def visualizer_project(audio: str | os.PathLike, *,
                       preset: str = "phonk",
                       duration: float | None = None,
                       images: Sequence[str | os.PathLike] = (),
                       title: str = "",
                       size: tuple[int, int] | None = None,
                       fps: float | None = None, seed: int = 21
                       ) -> MotionProject:
    """A music visualizer as a project: one procedural segment."""
    dur = duration or probe_duration(audio)
    if dur <= 0:
        raise MotionStudioError(
            "visualizer project needs a duration (or probeable audio)")
    seg = MotionSegment(
        uid="visual", kind="procedural", src=str(audio),
        duration_s=float(dur), preset=preset,
        design={"engine": "visualizer",
                "images": [str(i) for i in images]})
    return MotionProject(title=title or "visualizer", segments=[seg],
                         output=_out(size, fps), seed=seed)
