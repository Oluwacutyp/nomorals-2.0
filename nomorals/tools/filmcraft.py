"""FilmCraft + EffectCraft tools — the brain's handles on the
video-editor and motion-compositor organs.

Two tools, action-dispatched, over :mod:`nomorals.media.film` and
:mod:`nomorals.media.effects`; the organs own the timeline, the
compositor and the render — this module is only the surface.

``filmcraft`` — action=create makes a project
(name=, width=, height=, fps=); action=list lists projects;
action=show shows one (name=); action=add_clip adds a clip
(name=, track=, source=, at=, duration=, kind=, start=, gain= —
the track is created when missing); action=insert ripple-inserts
one; action=split splits a clip (clip=, at=); action=trim trims
(clip=, edge=in|out, delta=, ripple=); action=lift /
action=extract / action=ripple_delete remove a clip (clip=);
action=transition joins two adjacent clips (from=, to=, kind=,
duration=); action=title adds a title (track=, text=, at=,
duration=); action=caption adds a caption clip (track=, words=
JSON list of {text, start, end}); action=marker adds a named
marker (time=, label=); action=grade grades a clip (clip=,
lift=, gamma=, gain=, saturation=); action=duck sets the bed
ducking (duck_gain=); action=render renders the finished video
(name=, out=, narration_wav=, bed_wav=, profile=); action=delete
drops a project (name=).

``effectcraft`` — action=create makes a composition
(name=, width=, height=, fps=, duration=); action=list lists
compositions; action=show shows one (name=); action=add_layer
adds a layer (name=, kind=solid|text|image|shape, color=,
text=, image_path=, size=, blend=, opacity=); action=keyframe
animates a layer property (name=, layer= index, property=,
time=, value=, easing=); action=expression drives a property
from a safe expression (name=, layer=, property=, expr=);
action=effect applies a raster effect to a layer (name=, layer=,
effect=, params= JSON); action=camera sets the composition's 3D
camera (name=, position=, rotation=, fov_y=); action=render
renders the composition (name=, out=, audio=); action=list_effects
lists every effect with a real implementation; action=delete drops
a composition (name=).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..media.effects import effects as fx_effects
from ..media.effects.camera import Camera3D
from ..media.effects.composition import Composition, Layer, render_composition_video
from ..media.effects.keyframes import PropertyTrack
from ..media.effects.store import CompositionStore, effects_path
from ..media.film.project import FilmError, FilmProject
from ..media.film.render import render_project
from ..media.film.store import FilmStore, film_path

__all__ = ["register"]


def parse_filmcraft_request(
        text: str, known: list[str]
        ) -> tuple[str, str] | None:
    """Natural phrasing -> (verb, ref) for the film editor, or None.

    Matches only when the text says "film" and names a known film
    project (``show`` / ``render``), or asks for the project list
    with the words "film project" — a bare "edit this video" is not
    a match, so the hook never steals a video-generation question,
    and VOD wording ("analyze this match video") never matches
    either (the VOD film study owns that phrasing).
    """
    low = (text or "").lower()
    if "film" not in low:
        return None
    for name in known:
        if name and name.lower() in low:
            if ("render" in low or "export" in low
                    or "finish" in low):
                return ("render", name)
            return ("show", name)
    if "film project" in low and (
            "list" in low or "projects" in low):
        return ("list", "")
    return None


def parse_effect_request(
        text: str, known: list[str]
        ) -> tuple[str, str] | None:
    """Natural phrasing -> (verb, ref) for the compositor, or None.

    Matches only when the text says "effect", "composition" or
    "motion" and names a known composition (``show`` / ``render``),
    asks what effects exist, or asks for the composition list — a
    bare "great effects" with nothing to act on is not a match, so
    the hook never steals an unrelated question.
    """
    low = (text or "").lower()
    if ("effect" not in low and "composition" not in low
            and "motion" not in low):
        return None
    for name in known:
        if name and name.lower() in low:
            if "render" in low or "export" in low:
                return ("render", name)
            return ("show", name)
    if ("what effects" in low or "list effects" in low
            or "show effects" in low):
        return ("effects", "")
    if "list" in low and ("composition" in low or "effect" in low):
        return ("list", "")
    return None


def _film_store(context: Any) -> FilmStore:
    settings = getattr(context, "settings", None)
    return FilmStore(film_path(settings))


def _film_workdir(context: Any, name: str) -> Path:
    settings = getattr(context, "settings", None)
    root = film_path(settings).parent / "work" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _comp_store(context: Any) -> CompositionStore:
    settings = getattr(context, "settings", None)
    return CompositionStore(effects_path(settings))


def _comp_workdir(context: Any, name: str) -> Path:
    settings = getattr(context, "settings", None)
    root = effects_path(settings).parent / "work" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_arg(value: Any, what: str) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} must be JSON: {exc}") from exc


def _num(value: Any, what: str, default: float | None = None) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError(f"{what} is required")
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a number, got {value!r}") from exc


def _bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _project_summary(proj: FilmProject) -> dict[str, Any]:
    return {"name": proj.name, "width": proj.width,
            "height": proj.height, "fps": proj.fps,
            "duration": proj.duration,
            "tracks": [{"id": t.id, "kind": t.kind, "name": t.name,
                        "clips": len(t.items),
                        "sync_locked": t.sync_locked}
                       for t in proj.tracks],
            "transitions": len(proj.transitions),
            "markers": [{"time": m.time, "name": m.name}
                        for m in proj.markers],
            "content_hash": proj.content_hash()}


def _composition_summary(comp: Composition) -> dict[str, Any]:
    return {"width": comp.width, "height": comp.height,
            "fps": comp.fps, "duration": comp.duration,
            "layers": [{"kind": layer.kind, "blend": layer.blend,
                        "effects": [e["name"] for e in layer.effects]}
                       for layer in comp.layers],
            "camera": comp.camera.to_dict() if comp.camera else None}


def register(registry: Any) -> None:
    """Attach the ``filmcraft`` and ``effectcraft`` tools."""
    context = registry.context

    @registry.register(
        "filmcraft",
        description=(
            "Devon's own non-linear video editor (FilmCraft port): "
            "multi-track timeline, Premiere edit algebra, transitions, "
            "titles, captions, colour grade, audio ducking, loudness, "
            "and a render queue that produces finished videos of any "
            "length. action=create makes a project (name=, width=, "
            "height=, fps=); action=list lists projects; action=show "
            "shows one (name=); action=add_clip adds a clip (name=, "
            "track=, source=, at=, duration=); action=insert "
            "ripple-inserts one; action=split splits a clip (clip=, "
            "at=); action=trim trims (clip=, edge=in|out, delta=); "
            "action=lift / extract / ripple_delete remove a clip "
            "(clip=); action=transition joins adjacent clips "
            "(from=, to=, kind=, duration=); action=title adds a "
            "title (track=, text=, at=, duration=); action=caption "
            "adds a caption clip (track=, words= JSON); action=marker "
            "adds a named marker (time=, label=); action=grade grades "
            "a clip (clip=, lift=, gamma=, gain=, saturation=); "
            "action=duck sets bed ducking (duck_gain=); action=render "
            "renders the finished video (name=, out=, narration_wav=, "
            "bed_wav=); action=delete drops a project (name=)."
        ),
        capability="owner",
    )
    def filmcraft_tool(
        action: str = "list",
        name: str = "",
        track: str = "",
        source: str = "",
        at: Any = None,
        duration: Any = None,
        kind: str = "",
        start: Any = None,
        gain: Any = None,
        clip: str = "",
        edge: str = "out",
        delta: Any = None,
        ripple: Any = None,
        from_id: str = "",
        to_id: str = "",
        text: str = "",
        words: Any = None,
        time: Any = None,
        label: str = "",
        lift: Any = None,
        gamma: Any = None,
        saturation: Any = None,
        duck_gain: Any = None,
        out: str = "",
        narration_wav: str = "",
        bed_wav: str = "",
        profile: str = "workstation",
        allow_full_res: Any = None,
        width: Any = None,
        height: Any = None,
        fps: Any = None,
    ) -> dict[str, Any]:
        store = _film_store(context)
        action = (action or "list").strip().lower()
        try:
            if action == "list":
                return {"ok": True,
                        "projects": [_project_summary(store.load(n))
                                     for n in store.list_names()]}
            if action == "create":
                if not name.strip():
                    return {"ok": False,
                            "error": "give a project name: "
                                     "action=create name=demo"}
                if name in store.list_names():
                    return {"ok": False,
                            "error": f"project {name!r} already exists"}
                proj = FilmProject(
                    name.strip(),
                    width=int(_num(width, "width", 1920)),
                    height=int(_num(height, "height", 1080)),
                    fps=_num(fps, "fps", 24.0))
                store.save(proj)
                return {"ok": True, "project": _project_summary(proj)}
            if action == "show":
                return {"ok": True,
                        "project": _project_summary(store.load(name))}
            if action == "delete":
                if store.delete(name):
                    return {"ok": True, "deleted": name}
                return {"ok": False, "error": f"no project {name!r}"}
            if action in ("add_clip", "insert"):
                proj = store.load(name)
                if not track.strip():
                    return {"ok": False,
                            "error": "give a track name: track=V1"}
                if not source:
                    return {"ok": False,
                            "error": "give a source path: source=clip.mp4"}
                try:
                    tr = proj.find_track(track.strip())
                except FilmError:
                    # create the track: audio when the source is
                    # audio (or kind says so), video otherwise
                    track_kind = "audio" if (
                        kind == "audio"
                        or Path(source).suffix.lower() in
                        (".wav", ".mp3", ".flac", ".ogg", ".m4a",
                         ".aac", ".opus")) else "video"
                    tr = proj.add_track(track_kind, track.strip())
                fn = proj.insert_clip if action == "insert" \
                    else proj.add_clip
                made = fn(tr, source, at=_num(at, "at"),
                          duration=_num(duration, "duration"),
                          start=_num(start, "start", 0.0),
                          kind=kind or None,
                          gain=_num(gain, "gain", 1.0))
                store.save(proj)
                return {"ok": True, "clip": made.to_dict(),
                        "project": _project_summary(proj)}
            if action == "split":
                proj = store.load(name)
                first, second = proj.split_clip(
                    clip, at_time=_num(at, "at"))
                store.save(proj)
                return {"ok": True, "first": first.to_dict(),
                        "second": second.to_dict()}
            if action == "trim":
                proj = store.load(name)
                made = proj.trim_clip(clip, edge,
                                      _num(delta, "delta"),
                                      ripple=_bool(ripple))
                store.save(proj)
                return {"ok": True, "clip": made.to_dict()}
            if action in ("lift", "extract", "ripple_delete"):
                proj = store.load(name)
                fn = {"lift": proj.lift, "extract": proj.extract,
                      "ripple_delete": proj.ripple_delete}[action]
                removed = fn(clip)
                store.save(proj)
                return {"ok": True, "removed": removed.to_dict()}
            if action == "transition":
                proj = store.load(name)
                tr = proj.add_transition(
                    from_id, to_id, kind or "cross_dissolve",
                    _num(duration, "duration"))
                store.save(proj)
                return {"ok": True, "transition": tr.to_dict()}
            if action == "title":
                proj = store.load(name)
                try:
                    tr = proj.find_track(track.strip())
                except FilmError:
                    tr = proj.add_track("title", track.strip())
                made = proj.add_title(tr, text, at=_num(at, "at"),
                                      duration=_num(duration, "duration"))
                store.save(proj)
                return {"ok": True, "clip": made.to_dict()}
            if action == "caption":
                proj = store.load(name)
                parsed = _json_arg(words, "words")
                if not isinstance(parsed, list) or not parsed:
                    return {"ok": False,
                            "error": "words must be a JSON list of "
                                     "{text, start, end}"}
                try:
                    tr = proj.find_track(track.strip())
                except FilmError:
                    tr = proj.add_track("caption", track.strip())
                made = proj.add_caption(tr, parsed)
                store.save(proj)
                return {"ok": True, "clip": made.to_dict()}
            if action == "marker":
                proj = store.load(name)
                made = proj.add_marker(_num(time, "time"),
                                       label or "marker")
                store.save(proj)
                return {"ok": True, "marker": made.to_dict()}
            if action == "grade":
                proj = store.load(name)
                made = proj.grade_clip(
                    clip,
                    lift=_num_or_none(lift),
                    gamma=_num_or_none(gamma),
                    gain=_num_or_none(gain),
                    saturation=_num_or_none(saturation))
                store.save(proj)
                return {"ok": True, "clip": made.to_dict()}
            if action == "duck":
                proj = store.load(name)
                proj.set_ducking(enabled=True,
                                 duck_gain=_num(duck_gain, "duck_gain",
                                               0.25))
                store.save(proj)
                return {"ok": True, "ducking": proj.ducking}
            if action == "render":
                proj = store.load(name)
                if not out:
                    return {"ok": False,
                            "error": "give an output path: "
                                     "action=render out=film.mp4"}
                return render_project(
                    proj, out, workdir=_film_workdir(context, name),
                    narration_wav=narration_wav or None,
                    bed_wav=bed_wav or None,
                    profile=profile or "workstation",
                    allow_full_res=_bool(allow_full_res))
            return {"ok": False,
                    "error": f"unknown filmcraft action {action!r}"}
        except (FilmError, ValueError, KeyError) as exc:
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "effectcraft",
        description=(
            "Devon's own motion graphics compositor (EffectCraft "
            "port): compositions, layers, keyframes, raster effects, "
            "safe value expressions, and a 3D plate camera. "
            "action=create makes a composition (name=, width=, "
            "height=, fps=, duration=); action=list lists "
            "compositions; action=show shows one (name=); "
            "action=add_layer adds a layer (name=, kind=solid|text|"
            "image|shape, color=, text=, image_path=, size=, blend=, "
            "opacity=); action=keyframe animates a layer property "
            "(name=, layer= index, property=, time=, value=, "
            "easing=); action=expression drives a property from a "
            "safe expression (name=, layer=, property=, expr=); "
            "action=effect applies a raster effect to a layer "
            "(name=, layer=, effect=, params= JSON); action=camera "
            "sets the composition's 3D camera (name=, position=, "
            "rotation=, fov_y=); action=render renders the "
            "composition (name=, out=, audio=); action=list_effects "
            "lists every effect with a real implementation; "
            "action=delete drops a composition (name=)."
        ),
        capability="owner",
    )
    def effectcraft_tool(
        action: str = "list_effects",
        name: str = "",
        layer: Any = None,
        kind: str = "",
        color: Any = None,
        text: str = "",
        image_path: str = "",
        size: Any = None,
        blend: str = "normal",
        opacity: Any = None,
        property: str = "",
        time: Any = None,
        value: Any = None,
        easing: str = "smooth",
        expr: str = "",
        effect: str = "",
        params: Any = None,
        position: Any = None,
        rotation: Any = None,
        fov_y: Any = None,
        out: str = "",
        audio: str = "",
        width: Any = None,
        height: Any = None,
        fps: Any = None,
        duration: Any = None,
    ) -> dict[str, Any]:
        store = _comp_store(context)
        action = (action or "list_effects").strip().lower()
        try:
            if action == "list_effects":
                return {"ok": True, "effects": [
                    {"name": n, "params": fx_effects.effect_params(n)}
                    for n in fx_effects.list_effects()]}
            if action == "list":
                return {"ok": True,
                        "compositions": [
                            {"name": n,
                             **_composition_summary(store.load(n))}
                            for n in store.list_names()]}
            if action == "create":
                if not name.strip():
                    return {"ok": False,
                            "error": "give a composition name: "
                                     "action=create name=comp1"}
                if name in store.list_names():
                    return {"ok": False,
                            "error": f"composition {name!r} already exists"}
                comp = Composition(
                    width=int(_num(width, "width", 1920)),
                    height=int(_num(height, "height", 1080)),
                    fps=_num(fps, "fps", 24.0),
                    duration=_num(duration, "duration", 1.0))
                store.save(name.strip(), comp)
                return {"ok": True, "name": name.strip(),
                        "composition": _composition_summary(comp)}
            if action == "show":
                comp = store.load(name)
                return {"ok": True, "name": name,
                        "composition": _composition_summary(comp)}
            if action == "delete":
                if store.delete(name):
                    return {"ok": True, "deleted": name}
                return {"ok": False, "error": f"no composition {name!r}"}
            if action == "add_layer":
                comp = store.load(name)
                layer_obj = _layer_from_args(
                    kind=kind or "solid", color=color, text=text,
                    image_path=image_path, size=size, blend=blend,
                    opacity=opacity)
                comp.add_layer(layer_obj)
                store.save(name, comp)
                return {"ok": True, "name": name,
                        "layers": len(comp.layers)}
            if action in ("keyframe", "expression", "effect"):
                comp = store.load(name)
                layer_obj = _layer_at(comp, layer)
                if action == "keyframe":
                    _set_keyframe(layer_obj, property,
                                  _num(time, "time"),
                                  _num(value, "value"),
                                  easing or "smooth")
                elif action == "expression":
                    _set_expression(layer_obj, property, expr)
                else:
                    params_dict = dict(_json_arg(params, "params") or {})
                    # validate now, not at render time: apply the
                    # effect to a 1x1 image with these params
                    from PIL import Image as _Image
                    fx_effects.apply_effect(
                        effect, _Image.new("RGB", (1, 1)), params_dict)
                    layer_obj.effects.append({
                        "name": effect, "params": params_dict})
                store.save(name, comp)
                return {"ok": True, "name": name, "action": action}
            if action == "camera":
                comp = store.load(name)
                pos = _json_arg(position, "position") or (0.0, 0.0, 500.0)
                rot = _json_arg(rotation, "rotation") or (0.0, 0.0, 0.0)
                comp.camera = Camera3D(
                    position=tuple(float(v) for v in pos),
                    rotation=tuple(float(v) for v in rot),
                    fov_y=_num(fov_y, "fov_y", 40.0))
                store.save(name, comp)
                return {"ok": True, "name": name,
                        "camera": comp.camera.to_dict()}
            if action == "render":
                comp = store.load(name)
                if not out:
                    return {"ok": False,
                            "error": "give an output path: "
                                     "action=render out=comp.mp4"}
                report = render_composition_video(
                    comp, out, workdir=_comp_workdir(context, name),
                    audio=audio or None)
                return report
            return {"ok": False,
                    "error": f"unknown effectcraft action {action!r}"}
        except (ValueError, KeyError) as exc:
            return {"ok": False, "error": str(exc)}


def _num_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _layer_at(comp: Composition, index: Any) -> Layer:
    if index is None or index == "":
        raise ValueError("give a layer index: layer=0")
    i = int(index)
    if not 0 <= i < len(comp.layers):
        raise ValueError(
            f"layer {i} out of range (composition has "
            f"{len(comp.layers)} layer(s))")
    return comp.layers[i]


def _layer_from_args(*, kind: str, color: Any, text: str,
                     image_path: str, size: Any, blend: str,
                     opacity: Any) -> Layer:
    rgb = (0, 0, 0)
    if color is not None and color != "":
        parsed = _json_arg(color, "color")
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            rgb = tuple(int(c) for c in parsed)
        else:
            raise ValueError("color must be [r, g, b]")
    layer_size = None
    if size is not None and size != "":
        parsed = _json_arg(size, "size")
        if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
            layer_size = (int(parsed[0]), int(parsed[1]))
        else:
            raise ValueError("size must be [width, height]")
    op: float | PropertyTrack = 1.0
    if opacity is not None and opacity != "":
        op = _num(opacity, "opacity")
    return Layer(kind=kind, color=rgb, text=text,
                 image_path=image_path or None, size=layer_size,
                 blend=blend or "normal", opacity=op)


def _set_keyframe(layer_obj: Layer, prop: str, t: float, value: float,
                  easing: str) -> None:
    if prop == "opacity":
        track = layer_obj.opacity
        if not isinstance(track, PropertyTrack):
            track = PropertyTrack("opacity")
            layer_obj.opacity = track
        track.add(t, value, easing)
        return
    track = layer_obj.transform.get(prop)
    if not isinstance(track, PropertyTrack):
        track = PropertyTrack(prop)
        layer_obj.transform[prop] = track
    track.add(t, value, easing)


def _set_expression(layer_obj: Layer, prop: str, expr: str) -> None:
    if not expr.strip():
        raise ValueError("give an expression: expr='time * 10'")
    if prop == "opacity":
        track = layer_obj.opacity
        if not isinstance(track, PropertyTrack):
            track = PropertyTrack("opacity")
            layer_obj.opacity = track
        track.set_expression(expr)
        return
    track = layer_obj.transform.get(prop)
    if not isinstance(track, PropertyTrack):
        track = PropertyTrack(prop)
        layer_obj.transform[prop] = track
    track.set_expression(expr)
