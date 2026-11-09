"""Procedural creation (Living Chat P3 · doc 6, 2026-11-08).

Devon creates images and motion WITHOUT a generation model:

    brief → executable description → renderer executes it →
    the system INSPECTS the real output → repairs the
    description → delivers the verified artifact.

The description is data in the repo's own shapes, never a
prompt sent to an image/video model:

* ``studio`` — an emoji-studio composition (the design dict
  :func:`emoji_studio.parse_design` composes and
  :func:`emoji_studio.render_master` draws);
* ``scene3d`` — a Scene3D document, executed by the registered
  Scene3D renderer rungs (Rust / Blender) through
  :func:`scene3d.render_scene`;
* ``motion`` — a motion-studio explainer spec (diagram /
  equation / chart scenes), executed by
  :func:`motion_studio.studio.make_explainer`.

Inspection probes the REAL artifact — decodability, format,
dimensions, non-blankness (opaque coverage + distinct colours)
for images; stream/dimension/duration facts via ffprobe for
video. Repairs are deterministic description edits, one per
finding class, bounded by ``max_attempts``; every attempt —
its findings AND the changes they caused — is journaled (the
runtime-created ``procedural_creations`` table, Batch B
pattern; ``storage/migrations.py`` is never touched) and
carried on the result.

STRICT MODE IS THE ONLY MODE: this module never imports or
calls a generation backend — there is no ladder here that
quietly falls back to one. The mixed ladder (generation rung
first, studio second) lives in the expression organ and is
deliberately NOT used: a procedural request whose description
cannot be made to render fails honestly, with the inspection
evidence attached, rather than succeeding by a different
mechanism than the one that was asked for.
"""

from __future__ import annotations

import copy
import json
import logging
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import emoji_studio

_log = logging.getLogger(__name__)

__all__ = [
    "KIND_STUDIO", "KIND_SCENE3D", "KIND_MOTION", "CREATION_KINDS",
    "Finding", "AttemptRecord", "CreationSpec", "CreationResult",
    "author_studio_spec", "author_scene_spec", "author_motion_spec",
    "validate_spec", "inspect_output", "create_procedural",
    "MAX_ATTEMPTS",
]

KIND_STUDIO = "studio"
KIND_SCENE3D = "scene3d"
KIND_MOTION = "motion"
CREATION_KINDS = (KIND_STUDIO, KIND_SCENE3D, KIND_MOTION)

#: The repair budget: a description gets at most this many
#: render attempts before the failure is reported as-is.
MAX_ATTEMPTS = 3

_SIZE_MIN, _SIZE_MAX = 16, 1024

_STUDIO_FORMS = ("png", "sticker", "emoji")
_FORM_FORMATS = {"png": "png", "sticker": "webp", "emoji": "png"}


# ── the records ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """One inspection/validation verdict: a stable code plus
    the concrete evidence (never a bare "failed")."""

    code: str
    detail: str = ""


@dataclass
class AttemptRecord:
    """One attempt, journaled: what was wrong, what changed."""

    attempt: int
    findings: tuple[Finding, ...] = ()
    changes: tuple[str, ...] = ()
    artifact: str = ""


@dataclass
class CreationSpec:
    """The executable description: a kind, the kind's payload
    (the description itself), and the acceptance expectations
    the rendered artifact is inspected against."""

    kind: str
    title: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    expect: dict[str, Any] = field(default_factory=dict)


@dataclass
class CreationResult:
    ok: bool
    path: Path | None
    spec: CreationSpec
    attempts: tuple[AttemptRecord, ...] = ()
    creation_id: str = ""

    @property
    def findings(self) -> tuple[Finding, ...]:
        """The final attempt's findings (empty on success)."""
        return self.attempts[-1].findings if self.attempts else ()

    @property
    def repairs(self) -> tuple[str, ...]:
        """Every repair applied across all attempts, in order."""
        return tuple(c for a in self.attempts for c in a.changes)

    def summary_rows(self) -> tuple[tuple[str, str], ...]:
        """Card rows: the creation's auditable shape."""
        rows = [("kind", self.spec.kind),
                ("attempts", str(len(self.attempts)))]
        if self.repairs:
            rows.append(("repairs", "; ".join(self.repairs)))
        if self.ok and self.path is not None:
            rows.append(("artifact", self.path.name))
        else:
            rows.append(("outcome", "not delivered — see findings"))
        return tuple(rows)

    def describe(self) -> str:
        """The outcome in plain words, evidence included."""
        if self.ok:
            base = (f"created procedurally ({self.spec.kind}): "
                    f"{self.path.name if self.path else ''}")
            if self.repairs:
                base += (f" — the first description needed repair "
                         f"({'; '.join(self.repairs)}); the "
                         f"delivered render passed inspection")
            return base
        evidence = "; ".join(
            f"{f.code}: {f.detail}" if f.detail else f.code
            for f in self.findings) or "no findings recorded"
        return (f"procedural creation ({self.spec.kind}) could "
                f"not be made to render after "
                f"{len(self.attempts)} attempt(s) — {evidence}. "
                f"No generation model was used as a fallback; "
                f"nothing was delivered.")


# ── authoring: brief → executable description ───────────────────


def author_studio_spec(description: str, *,
                       form: str = "sticker") -> CreationSpec:
    """A studio composition from plain words — the studio's own
    deterministic composer (parse_design) authors the design;
    the pipeline owns only the delivery form + expectations."""
    form = form if form in _STUDIO_FORMS else "sticker"
    design = emoji_studio.parse_design(description)
    size = 100 if form == "emoji" else 512
    return CreationSpec(
        kind=KIND_STUDIO,
        title=" ".join(str(description or "").split()),
        payload={"design": design, "size": size, "form": form},
        expect={"format": _FORM_FORMATS[form], "width": size,
                "height": size, "min_opaque": 0.02, "min_colors": 2})


_PRIMITIVE_WORDS = {
    "cube": "cube", "box": "cube", "sphere": "sphere",
    "ball": "sphere", "orb": "sphere", "cone": "cone",
    "cylinder": "cylinder", "tube": "cylinder",
    "plane": "plane", "ground": "plane",
}


def _hex_rgb(value: str) -> tuple[float, float, float]:
    raw = str(value).lstrip("#")
    return tuple(int(raw[i:i + 2], 16) / 255.0
                 for i in (0, 2, 4))  # type: ignore[return-value]


def author_scene_spec(description: str) -> CreationSpec:
    """A Scene3D document from plain words: the first primitive
    word names the subject's geometry, the first palette word
    its material colour; camera and light are the house still
    setup. The payload IS a Scene3D document (to_dict shape)."""
    from ...media.edit_engine.scene3d import (
        Camera,
        Light,
        Material,
        Mesh,
        Rotation,
        Scene3D,
        SceneNode,
        Transform,
    )

    text = " ".join(str(description or "").lower().split())
    words = text.split()
    primitive = next((_PRIMITIVE_WORDS[w] for w in words
                      if w in _PRIMITIVE_WORDS), "cube")
    palette = next((name for name in emoji_studio.PALETTES
                    if name in words), "gold")
    color = _hex_rgb(emoji_studio.PALETTES[palette][0])
    scene = Scene3D(
        title=text or "procedural scene",
        nodes=[
            SceneNode(
                name="subject",
                transform=Transform(position=(0.0, 0.0, 0.0)),
                mesh=Mesh(kind="primitive", primitive=primitive,
                          params={"size": 2.0}),
                material="paint"),
            SceneNode(
                name="sun",
                transform=Transform(
                    position=(3.0, 5.0, 2.0),
                    rotation=Rotation.look_at((0.0, 0.0, 0.0))),
                light=Light(kind="directional", intensity=2.0)),
        ],
        cameras={"main": Camera(transform=Transform(
            position=(0.0, 1.5, 6.0),
            rotation=Rotation.look_at((0.0, 0.0, 0.0))))},
        active_camera="main",
        materials={"paint": Material(kind="lambert", color=color)},
        output={"width": 320, "height": 320, "fps": 30})
    return CreationSpec(
        kind=KIND_SCENE3D, title=text,
        payload={"document": scene.to_dict(), "renderer": ""},
        expect={"format": "png", "width": 320, "height": 320,
                "min_opaque": 0.5, "min_colors": 2})


def author_motion_spec(description: str) -> CreationSpec:
    """A motion explainer spec from plain words: one diagram
    scene whose single node carries the brief — the smallest
    honest explainer, rendered by the motion studio."""
    title = " ".join(str(description or "").split()) or "explainer"
    scene = {"engine": "diagram", "title": title,
             "nodes": [{"id": "core", "label": title,
                        "x": 0.5, "y": 0.5}],
             "edges": [], "duration": 1.6, "text": ""}
    return CreationSpec(
        kind=KIND_MOTION, title=title,
        payload={"scenes": [scene], "per_scene": 1.6,
                 "size": [320, 180], "fps": 10},
        expect={"format": "mp4", "width": 320, "height": 180,
                "min_duration": 1.0})


# ── validation: description defects, found before rendering ───


def _valid_shapes() -> set[str]:
    return {name for _words, name in emoji_studio.SHAPES} \
        | {"face", "letter"}


def _size_findings(what: str, width: Any, height: Any,
                   ) -> list[Finding]:
    out = []
    for label, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) \
                or not (_SIZE_MIN <= value <= _SIZE_MAX):
            out.append(Finding(
                "size-out-of-range",
                f"{what} {label} {value!r} is outside "
                f"{_SIZE_MIN}–{_SIZE_MAX}"))
    return out


def validate_spec(spec: CreationSpec) -> list[Finding]:
    """Structural defects in the DESCRIPTION (not the render):
    names the renderers cannot honor, sizes they cannot take,
    documents that do not parse."""
    findings: list[Finding] = []
    payload = spec.payload or {}
    if spec.kind == KIND_STUDIO:
        design = payload.get("design") or {}
        shape = str(design.get("shape") or "")
        if shape not in _valid_shapes():
            findings.append(Finding(
                "unknown-shape",
                f"the studio has no shape {shape!r}"))
        palette = str(design.get("palette") or "")
        if palette not in emoji_studio.PALETTES:
            findings.append(Finding(
                "unknown-palette",
                f"the studio has no palette {palette!r}"))
        if shape == "letter" and not str(design.get("letter") or ""):
            findings.append(Finding(
                "missing-letter",
                "a letter design names no letter"))
        form = str(payload.get("form") or "")
        if form not in _STUDIO_FORMS:
            findings.append(Finding(
                "unknown-form", f"unknown delivery form {form!r}"))
        elif form != "emoji":
            size = payload.get("size")
            findings += _size_findings("render", size, size)
    elif spec.kind == KIND_SCENE3D:
        from ...media.edit_engine.scene3d import Scene3D, Scene3DError
        try:
            scene = Scene3D.from_dict(payload.get("document") or {})
        except Scene3DError as exc:
            findings.append(Finding("scene-invalid", str(exc)))
        except Exception as exc:  # noqa: BLE001 - a bad doc is a finding
            findings.append(Finding(
                "scene-invalid", f"{type(exc).__name__}: {exc}"))
        else:
            findings += _size_findings(
                "scene output", scene.output.get("width"),
                scene.output.get("height"))
    elif spec.kind == KIND_MOTION:
        from ...media.motion_studio.explainers import is_explainer_engine
        scenes = payload.get("scenes") or []
        if not scenes:
            findings.append(Finding(
                "empty-scenes", "a motion spec needs at least one scene"))
        for i, scene in enumerate(scenes):
            engine = str((scene or {}).get("engine") or "")
            if not is_explainer_engine(engine):
                findings.append(Finding(
                    "unknown-engine",
                    f"scene {i + 1} names engine {engine!r}, which "
                    f"the motion studio does not have"))
            duration = (scene or {}).get(
                "duration", payload.get("per_scene", 0))
            try:
                bad_duration = float(duration) <= 0
            except (TypeError, ValueError):
                bad_duration = True
            if bad_duration:
                findings.append(Finding(
                    "bad-duration",
                    f"scene {i + 1} duration {duration!r} "
                    f"is not positive"))
        size = payload.get("size") or []
        if len(size) == 2:
            findings += _size_findings("motion canvas", size[0], size[1])
    else:
        findings.append(Finding(
            "unknown-kind", f"no renderer for kind {spec.kind!r}"))
    return findings


# ── render adapters: the existing renderers execute ───────────


def _render_studio(spec: CreationSpec, out_dir: Path,
                   name: str) -> Path:
    design = dict(spec.payload.get("design") or {})
    form = str(spec.payload.get("form") or "sticker")
    if form == "emoji":
        return emoji_studio.emoji_png(design, out_dir, name=name)
    if form == "sticker":
        master = emoji_studio.sticker_master_png(
            design, out_dir, name=name)
        from .media_convert import image_to_sticker
        return image_to_sticker(master, out_dir=out_dir)
    size = int(spec.payload.get("size") or 512)
    img = emoji_studio.render_master(design, size=size)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    img.save(str(path))
    return path


def _render_scene(spec: CreationSpec, out_dir: Path,
                  name: str) -> Path:
    from ...media.edit_engine import scene3d
    try:
        from ...media.edit_engine import render3d_blender, render3d_rust  # noqa: F401 - registration
    except Exception as exc:  # noqa: BLE001 - rungs may be absent
        _log.debug("scene rung import failed: %s", exc)
    scene = scene3d.Scene3D.from_dict(spec.payload.get("document")
                                      or {})
    frames_dir = out_dir / f"{name}_frames"
    record = scene3d.render_scene(
        scene, frames=[0], workdir=frames_dir,
        name=str(spec.payload.get("renderer") or "") or None)
    frames = record.get("frames") or []
    if not frames:
        raise RuntimeError("the scene renderer returned no frames")
    return Path(str(frames[0]))


def _render_motion(spec: CreationSpec, out_dir: Path,
                   name: str) -> Path:
    from ...media.motion_studio.studio import make_explainer
    payload = spec.payload
    size = payload.get("size") or [320, 180]
    out = out_dir / f"{name}.mp4"
    made = make_explainer(
        payload.get("scenes") or [], out=str(out),
        title=spec.title,
        per_scene=float(payload.get("per_scene") or 1.6),
        size=(int(size[0]), int(size[1])),
        fps=float(payload.get("fps") or 10))
    return Path(str(made))


def _render(spec: CreationSpec, out_dir: Path, name: str) -> Path:
    if spec.kind == KIND_STUDIO:
        return _render_studio(spec, out_dir, name)
    if spec.kind == KIND_SCENE3D:
        return _render_scene(spec, out_dir, name)
    if spec.kind == KIND_MOTION:
        return _render_motion(spec, out_dir, name)
    raise RuntimeError(f"no renderer for kind {spec.kind!r}")


# ── inspection: probe the real artifact ─────────────────────────


def _inspect_image(path: Path, expect: dict[str, Any],
                   ) -> list[Finding]:
    findings: list[Finding] = []
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.load()
            fmt = str(img.format or "").lower()
            width, height = img.size
            rgba = img.convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - undecodable IS the finding
        return [Finding("undecodable",
                        f"{type(exc).__name__}: {exc}")]
    want_format = str(expect.get("format") or "")
    if want_format and fmt != want_format:
        findings.append(Finding(
            "format-mismatch",
            f"rendered {fmt}, the description expects {want_format}"))
    want_w, want_h = expect.get("width"), expect.get("height")
    if (want_w and width != want_w) or (want_h and height != want_h):
        findings.append(Finding(
            "dimension-mismatch",
            f"rendered {width}×{height}, the description expects "
            f"{want_w}×{want_h}"))
    small = rgba.resize((64, 64))
    alpha = small.split()[3]
    hist = alpha.histogram()
    opaque = sum(hist[9:]) / float(64 * 64)
    min_opaque = float(expect.get("min_opaque", 0.02))
    if opaque < min_opaque:
        findings.append(Finding(
            "blank-render",
            f"only {opaque:.1%} of the frame is opaque "
            f"(floor {min_opaque:.0%}) — the design drew nothing "
            f"visible"))
    colors = small.convert("RGB").getcolors(64 * 64) or []
    min_colors = int(expect.get("min_colors", 2))
    if len(colors) < min_colors:
        findings.append(Finding(
            "flat-render",
            f"the frame carries {len(colors)} distinct colour(s) "
            f"(floor {min_colors})"))
    return findings


def _inspect_video(path: Path, expect: dict[str, Any],
                   ) -> list[Finding]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return [Finding(
            "probe-unavailable",
            "ffprobe is not on this host — the render cannot be "
            "verified, so it is not delivered as verified")]
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60)
        data = json.loads(proc.stdout or "{}")
    except Exception as exc:  # noqa: BLE001 - unprobeable is a finding
        return [Finding("undecodable",
                        f"ffprobe failed: {type(exc).__name__}: {exc}")]
    findings: list[Finding] = []
    streams = data.get("streams") or []
    video = [s for s in streams if s.get("codec_type") == "video"]
    if not video:
        return [Finding("no-video-stream",
                        "the rendered file carries no video stream")]
    stream = video[0]
    want_w, want_h = expect.get("width"), expect.get("height")
    if (want_w and stream.get("width") != want_w) \
            or (want_h and stream.get("height") != want_h):
        findings.append(Finding(
            "dimension-mismatch",
            f"rendered {stream.get('width')}×{stream.get('height')}, "
            f"the description expects {want_w}×{want_h}"))
    duration = float((data.get("format") or {}).get("duration") or 0)
    min_duration = float(expect.get("min_duration", 0))
    if min_duration and duration < min_duration:
        findings.append(Finding(
            "short-duration",
            f"the render runs {duration:.2f}s, below the "
            f"description's {min_duration:.2f}s floor"))
    return findings


def inspect_output(path: Path, expect: dict[str, Any],
                   ) -> list[Finding]:
    """Probe one rendered artifact against the description's
    expectations. An artifact that cannot be probed is a
    finding, never a pass."""
    if not path.exists() or path.stat().st_size <= 0:
        return [Finding("no-output",
                        "the renderer produced no file (or an "
                        "empty one)")]
    if str(expect.get("format") or "") in ("mp4", "webm", "gif"):
        return _inspect_video(path, expect)
    return _inspect_image(path, expect)


# ── repairs: deterministic description edits, one per finding ──


def _clamp_size(value: Any) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = 512
    return max(_SIZE_MIN, min(_SIZE_MAX, number))


def _apply_repairs(spec: CreationSpec,
                   findings: list[Finding]) -> list[str]:
    """Mutate ``spec`` toward renderable, returning the changes
    in plain words. A finding class with no honest description
    edit (a structurally invalid scene, an unknown engine, a
    renderer that errored) yields NO change — the caller then
    fails honestly instead of looping. Each rule fires at most
    once per creation (tracked in the payload), so the loop
    cannot chase its own tail."""
    payload = spec.payload
    done: list[str] = payload.setdefault("_repaired", [])
    changes: list[str] = []
    codes = {f.code for f in findings}

    def fire(code: str, change: str) -> None:
        if code not in done:
            done.append(code)
            changes.append(change)

    if spec.kind == KIND_STUDIO:
        design = payload.setdefault("design", {})
        if "unknown-palette" in codes:
            old = str(design.get("palette") or "")
            design["palette"] = "gold"
            fire("unknown-palette",
                 f"palette {old!r} is not a studio palette — "
                 f"set to 'gold'")
        if "unknown-shape" in codes:
            old = str(design.get("shape") or "")
            design["shape"] = "face"
            design["face"] = str(design.get("face") or "happy")
            fire("unknown-shape",
                 f"shape {old!r} is not a studio shape — "
                 f"set to a happy face")
        if "missing-letter" in codes:
            letter = next((c for c in spec.title if c.isalpha()), "")
            if letter:
                design["letter"] = letter.lower()
                fire("missing-letter",
                     f"the letter design named no letter — "
                     f"took {letter.lower()!r} from the brief")
            else:
                design["shape"] = "face"
                design["face"] = "happy"
                fire("missing-letter",
                     "the letter design named no letter and the "
                     "brief has none — set to a happy face")
        if "unknown-form" in codes:
            payload["form"] = "png"
            fire("unknown-form",
                 "the delivery form was unknown — set to 'png'")
        if "size-out-of-range" in codes:
            old = payload.get("size")
            payload["size"] = _clamp_size(old)
            fire("size-out-of-range",
                 f"render size {old!r} outside "
                 f"{_SIZE_MIN}–{_SIZE_MAX} — set to "
                 f"{payload['size']}")
        if "dimension-mismatch" in codes:
            want = spec.expect.get("width")
            if want:
                payload["size"] = int(want)
                fire("dimension-mismatch",
                     f"render size set to the expected {want}")
        if "format-mismatch" in codes:
            want = str(spec.expect.get("format") or "")
            form = {"png": "png", "webp": "sticker"}.get(want)
            if form and payload.get("form") != form:
                payload["form"] = form
                fire("format-mismatch",
                     f"delivery form set to {form!r} so the "
                     f"render is {want}")
        if "blank-render" in codes and "blank-render" not in done:
            design["bg"] = "disc"
            design["sparkles"] = True
            fire("blank-render",
                 "the render was blank — the design now draws "
                 "on a badge disc with sparkles")
        if "flat-render" in codes and not design.get("sparkles"):
            design["sparkles"] = True
            fire("flat-render",
                 "the render was flat — sparkles added for "
                 "colour range")
    elif spec.kind == KIND_SCENE3D:
        doc = payload.setdefault("document", {})
        if "size-out-of-range" in codes:
            output = dict(doc.get("output") or {})
            output["width"] = _clamp_size(output.get("width"))
            output["height"] = _clamp_size(output.get("height"))
            doc["output"] = output
            fire("size-out-of-range",
                 f"scene output clamped to {output['width']}×"
                 f"{output['height']}")
        if "dimension-mismatch" in codes:
            output = dict(doc.get("output") or {})
            output["width"] = int(spec.expect.get("width")
                                  or output.get("width") or 320)
            output["height"] = int(spec.expect.get("height")
                                   or output.get("height") or 320)
            doc["output"] = output
            fire("dimension-mismatch",
                 f"scene output set to the expected "
                 f"{output['width']}×{output['height']}")
    elif spec.kind == KIND_MOTION:
        if "bad-duration" in codes:
            for scene in payload.get("scenes") or []:
                try:
                    bad = float(scene.get(
                        "duration", payload.get("per_scene", 0))) <= 0
                except (TypeError, ValueError):
                    bad = True
                if bad:
                    scene["duration"] = 2.0
            fire("bad-duration",
                 "non-positive scene durations set to 2.0s")
        if "size-out-of-range" in codes:
            size = list(payload.get("size") or [320, 180])
            size = [_clamp_size(size[0]), _clamp_size(size[1])]
            payload["size"] = size
            fire("size-out-of-range",
                 f"motion canvas clamped to {size[0]}×{size[1]}")
        if "dimension-mismatch" in codes:
            payload["size"] = [int(spec.expect.get("width") or 320),
                               int(spec.expect.get("height") or 180)]
            fire("dimension-mismatch",
                 f"motion canvas set to the expected "
                 f"{payload['size'][0]}×{payload['size'][1]}")
    return changes


# ── the journal (Batch B pattern: a runtime-created table) ─────

_JOURNAL_SQL = (
    "CREATE TABLE IF NOT EXISTS procedural_creations ("
    "creation_id TEXT NOT NULL, kind TEXT NOT NULL DEFAULT '', "
    "attempt INTEGER NOT NULL DEFAULT 0, "
    "findings TEXT NOT NULL DEFAULT '[]', "
    "changes TEXT NOT NULL DEFAULT '[]', "
    "artifact TEXT NOT NULL DEFAULT '', "
    "ok INTEGER NOT NULL DEFAULT 0, at REAL NOT NULL DEFAULT 0)")


def _journal(db: Any, creation_id: str, kind: str,
             record: AttemptRecord, *, ok: bool) -> None:
    """One attempt → one journal row: findings, changes,
    artifact, verdict. Never raises — a creation must not die
    because its journal did."""
    if db is None:
        return
    try:
        db.execute(_JOURNAL_SQL)
        db.execute(
            "INSERT INTO procedural_creations (creation_id, kind, "
            "attempt, findings, changes, artifact, ok, at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (creation_id, kind, record.attempt,
             json.dumps([{"code": f.code, "detail": f.detail}
                         for f in record.findings]),
             json.dumps(list(record.changes)),
             record.artifact, 1 if ok else 0, time.time()))
    except Exception as exc:  # noqa: BLE001 - journal is evidence
        _log.debug("procedural creation journal failed: %s", exc)


# ── the pipeline ────────────────────────────────────────────────


def create_procedural(spec: CreationSpec, out_dir: Path | str, *,
                      db: Any = None,
                      max_attempts: int = MAX_ATTEMPTS,
                      name: str = "") -> CreationResult:
    """Execute one procedural creation end to end.

    validate → render → inspect → repair, bounded by
    ``max_attempts``. The delivered artifact is the one that
    PASSED inspection; intermediate renders stay on disk as
    evidence but are never reported as the result. When no
    repair applies or the budget ends, the result is an honest
    failure carrying the final findings.
    """
    creation_id = secrets.token_hex(4)
    work = CreationSpec(
        kind=spec.kind, title=spec.title,
        payload=copy.deepcopy(spec.payload or {}),
        expect=dict(spec.expect or {}))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(name or f"proc_{creation_id}")
    attempts: list[AttemptRecord] = []
    artifact: Path | None = None
    budget = max(1, int(max_attempts))
    for n in range(1, budget + 1):
        findings = validate_spec(work)
        rendered: Path | None = None
        if not findings:
            try:
                rendered = _render(work, out, stem)
            except Exception as exc:  # noqa: BLE001 - evidence, not death
                findings = [Finding(
                    "renderer-error",
                    f"{type(exc).__name__}: {exc}")]
            if rendered is not None:
                findings = inspect_output(rendered, work.expect)
        record = AttemptRecord(
            attempt=n, findings=tuple(findings),
            artifact=str(rendered) if rendered is not None else "")
        succeeded = rendered is not None and not findings
        if not succeeded:
            record.changes = tuple(_apply_repairs(work, findings))
        attempts.append(record)
        _journal(db, creation_id, work.kind, record, ok=succeeded)
        if succeeded:
            artifact = rendered
            break
        if not record.changes:
            break  # nothing honest left to change in the description
    return CreationResult(
        ok=artifact is not None, path=artifact, spec=work,
        attempts=tuple(attempts), creation_id=creation_id)
