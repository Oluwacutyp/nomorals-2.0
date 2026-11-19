"""AniCraft: design, play and publish 2D vector animations (AniCraft
port of storytold/anicraft — an open-source clean-room 2D animation
editor in Rust).

A thin surface over :mod:`nomorals.media.animation` — the organ owns
the ``.acraft`` codec, the document model, the Penner easings, the
timeline evaluator, the PIL renderer, the publish formats and the
preset library; the tool only routes arguments and shapes the reply.
Pure local work, stdlib + numpy + PIL, no upload, no retention.
Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.animation import (
    AiClient,
    AnimationDocument,
    AnimationEditor,
    AnimationError,
    ConsentLedger,
    Easing,
    Element,
    Role,
    build_preset,
    evaluate,
    list_presets,
    publish,
    render_frame,
    search_presets,
)
from ..media.animation import run_parity as run_animation_parity
from ..media.animation import (
    verify_attribution as verify_animation_attribution,
)
from ..media.animation.xfl import export_xfl, import_xfl

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "tween", "set", "play",
            "render", "publish", "library", "preset", "suggest",
            "import", "export", "parity", "attribution")

_LEDGER = ConsentLedger()


def _load(sources: list[str]) -> tuple[AnimationDocument, Path]:
    if not sources:
        raise ValueError("anicraft needs a source file")
    path = Path(sources[0])
    if not path.is_file():
        raise ValueError(f"anicraft source not found: {path}")
    try:
        return AnimationDocument.parse(path.read_bytes()), path
    except AnimationError as exc:
        raise ValueError(str(exc)) from exc


def _save(doc: AnimationDocument, sources: list[str],
          output: str) -> Path:
    target = Path(output) if output else Path(sources[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(doc.to_bytes())
    return target


def _parse_params(params: dict | None) -> dict:
    return dict(params or {})


def register(registry: Any) -> None:
    """Attach the anicraft family to a registry."""
    @registry.register(
        "anicraft",
        description=(
            "Design, play and publish 2D vector animations, local and "
            "uncapped — inspect animations, timelines and layers; add "
            "layers, keyframes and shapes (rect, ellipse, path, text, "
            "symbol instances); tween with the 27 Penner easings "
            "(Quad/Cubic/Quart/Quint/Sine/Back/Circ/Bounce/Elastic "
            "in/out/in-out, Classic slider, cubic Bezier, custom and "
            "sampled curves) plus per-group eases, shape tweens and "
            "motion paths; play the timeline to real frames; render "
            "frames with onion skinning; publish GIF, APNG, WebP, MP4, "
            "PNG sequences, sprite sheets, SVG and HTML; use the preset "
            "library (walk-cycle, bounce, fade-in, slide-in, pulse, "
            "typewriter); ask the local AI for a motion suggestion; "
            "import/export Adobe Animate XFL. AniCraft actions: "
            "inspect=<read the animation, sources=[path]>; create=<blank "
            "animation, output=<path>, title=<t>>; add=<add a layer, "
            "keyframe or shape, kind=<layer|keyframe|rect|ellipse|text>, "
            "node=<layer id>, x=<n>, y=<n>, value=<text|label>, "
            "params=<dict>>; tween=<tween a keyframe, node=<layer id>, "
            "value=<start frame>, kind=<classic|shape>, command=<easing "
            "name>>; set=<set an ease or a layer property, node=<layer "
            "id>, kind=<ease|opacity|visible>, value=<v>>; play=<evaluate "
            "a frame, value=<frame>>; render=<render frames to PNG, "
            "output=<path>, value=<frame>, params=<{onion: n}>>; publish=<"
            "publish a format, output=<path>, kind=<gif|mp4|pngseq|"
            "spritesheet|svg|png|html|webp|apng>, params=<{start, end}>>; "
            "library=<list/search presets, value=<query>>; preset=<build "
            "a preset animation, value=<id>, output=<path>>; suggest=<ask "
            "the local AI for a motion plan, value=<prompt>>; import=<"
            "import an XFL folder, sources=[path.xfl], output=<path>>; "
            "export=<export an XFL folder, sources=[path], output=<path."
            "xfl>>; parity=<run the machine-checked parity TOML>; "
            "attribution=<verify the SHA-256 attribution DB>. action=<"
            "inspect|create|add|tween|set|play|render|publish|library|"
            "preset|suggest|import|export|parity|attribution>; sources=<"
            "list of .acraft paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def anicraft_tool(action: str = "inspect",
                      sources: list[str] | None = None,
                      output: str = "", kind: str = "",
                      node: str = "", nodes: str | list[str] | None = None,
                      x: float = 0.0, y: float = 0.0,
                      value: Any = None, command: str = "",
                      params: dict | None = None,
                      title: str = "",
                      ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown anicraft action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if action == "parity":
            result = run_animation_parity()
            result["reply"] = (
                f"anicraft parity: {result['passed']} passed, "
                f"{result['failed']} failed.")
            return result
        if action == "attribution":
            result = verify_animation_attribution()
            result["reply"] = (
                f"anicraft attribution: {result['verified']} verified, "
                f"{result['failed']} failed, {result['skipped']} skipped, "
                f"{result['noted']} noted.")
            return result
        if action == "library":
            query = str(value or "")
            presets = search_presets(query) if query else list_presets()
            return {"presets": presets,
                    "reply": f"{len(presets)} presets"
                             + (f" matching {query!r}." if query else ".")}
        if action == "preset":
            preset_id = str(value or "")
            if not preset_id:
                raise ValueError("anicraft preset needs value=<id>")
            if not output:
                raise ValueError("anicraft preset needs output=<path>")
            doc = build_preset(preset_id)
            target = _save(doc, sources, output)
            return {"preset": preset_id,
                    "frames": doc.scenes[0].timeline.frame_count(),
                    "output": str(target),
                    "reply": f"built preset {preset_id} at {target}."}
        if action == "create":
            if not output:
                raise ValueError("anicraft create needs output=<path>")
            doc = AnimationDocument.create(title or "Untitled")
            target = _save(doc, sources, output)
            return {"output": str(target),
                    "reply": f"created blank animation at {target}."}
        if action == "suggest":
            prompt = str(value or "")
            if not prompt:
                raise ValueError("anicraft suggest needs value=<prompt>")
            client = AiClient.standard("anicraft")
            role = Role.suggest_motion()
            status = client.status(role)
            if not status["available"]:
                return {"available": False, "reason": status["reason"],
                        "reply": f"motion suggestion unavailable: "
                                 f"{status['reason']}"}
            plan = client.plan(role, {"prompt": prompt})
            job = client.start(plan, _LEDGER)
            snap = job.poll()
            result = snap.get("result", {})
            return {"suggestion": result, "reply": (
                f"suggested {result.get('preset')} with "
                f"{result.get('easing')} easing "
                f"({result.get('note', '')})")}
        if action == "import":
            if not sources:
                raise ValueError("anicraft import needs sources=[path.xfl]")
            if not output:
                raise ValueError("anicraft import needs output=<path>")
            try:
                doc = import_xfl(sources[0])
            except AnimationError as exc:
                raise ValueError(str(exc)) from exc
            target = _save(doc, sources, output)
            skipped = getattr(doc, "_xfl_skipped", [])
            return {"output": str(target), "skipped": skipped,
                    "reply": f"imported {sources[0]} as {target}"
                             + (f" ({len(skipped)} elements skipped)."
                                if skipped else ".")}
        doc, path = _load(sources)
        editor = AnimationEditor(doc)
        if action == "inspect":
            scene = doc.scenes[0]
            layers = [{"name": ly.name, "kind": ly.kind,
                       "keyframes": len(ly.keyframes)}
                      for ly in scene.timeline.layers]
            return {"title": doc.name, "fps": doc.fps,
                    "scenes": [s.name for s in doc.scenes],
                    "symbols": [s.name for s in doc.symbols],
                    "layers": layers,
                    "frames": scene.timeline.frame_count(),
                    "reply": (f"{path}: animation {doc.name!r}, "
                              f"{len(scene.timeline.layers)} layers, "
                              f"{scene.timeline.frame_count()} frames.")}
        if action == "add":
            p = _parse_params(params)
            if kind in ("layer", ""):
                layer = editor.add_layer(str(value or "Layer"),
                                         kind=p.get("layer_kind", "normal"))
                _save(doc, sources, output)
                return {"layer_id": layer.id, "layer": layer.name,
                        "reply": f"added layer {layer.name!r}."}
            layer_id = int(node)
            if kind == "keyframe":
                kf = editor.add_keyframe(layer_id, int(x),
                                         int(p.get("duration", 12)),
                                         label=str(value or ""))
                _save(doc, sources, output)
                return {"start": kf.start, "duration": kf.duration,
                        "reply": f"added keyframe at {kf.start}."}
            if kind in ("rect", "ellipse", "text"):
                layer = doc.scene().timeline.layer_by_id(layer_id)
                kf = layer.keyframes[0] if layer.keyframes else \
                    layer.add_keyframe(0, 12)
                if kind == "rect":
                    el = Element.rect(x, y, float(p.get("w", 40.0)),
                                      float(p.get("h", 40.0)),
                                      fill=str(p.get("fill", "#000000")))
                elif kind == "ellipse":
                    el = Element.ellipse(x, y, float(p.get("rx", 20.0)),
                                         float(p.get("ry", 20.0)),
                                         fill=str(p.get("fill", "#000000")))
                else:
                    el = Element.text(str(value or ""), x, y,
                                      font_size=float(
                                          p.get("font_size", 24.0)))
                kf.add_element(el)
                _save(doc, sources, output)
                return {"element_id": el.id, "kind": kind,
                        "reply": f"added {kind} element {el.id}."}
            raise ValueError(f"anicraft add needs kind=<layer|keyframe|"
                             f"rect|ellipse|text>, got {kind!r}")
        if action == "tween":
            layer_id = int(node)
            start = int(value) if value is not None else 0
            layer = doc.scene().timeline.layer_by_id(layer_id)
            kf = next((k for k in layer.keyframes if k.start == start),
                      None)
            if kf is None:
                raise ValueError(
                    f"no keyframe starting at {start} on layer "
                    f"{layer.name!r}")
            easing = Easing.named(command or "linear")
            kf.tween_to(layer.keyframes[-1], easing=easing,
                        kind=kind or "classic")
            _save(doc, sources, output)
            return {"easing": easing.key(), "kind": kind or "classic",
                    "reply": f"tweened keyframe {start} with "
                             f"{easing.label()}."}
        if action == "set":
            layer_id = int(node)
            layer = doc.scene().timeline.layer_by_id(layer_id)
            if kind == "opacity":
                layer.opacity = max(0.0, min(1.0, float(value)))
            elif kind == "visible":
                layer.visible = bool(value)
            elif kind == "ease":
                start = int(_parse_params(params).get("start", 0))
                editor.set_ease(layer_id, start,
                                Easing.named(str(value or "linear")))
            else:
                raise ValueError(
                    "anicraft set needs kind=<ease|opacity|visible>")
            _save(doc, sources, output)
            return {"layer": layer.name,
                    "reply": f"set {kind} on layer {layer.name!r}."}
        if action == "play":
            frame = int(value) if value is not None else 0
            dl = evaluate(doc, frame)
            return {"frame": frame, "items": len(dl.items),
                    "notes": dl.notes,
                    "reply": f"frame {frame}: {len(dl.items)} items."}
        if action == "render":
            p = _parse_params(params)
            frame = int(value) if value is not None else 0
            if not output:
                raise ValueError("anicraft render needs output=<path>")
            img, report = render_frame(doc, frame,
                                       onion=int(p.get("onion", 0)))
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            img.save(output)
            return {"output": output, "frame": frame,
                    "unrendered": report["unrendered"],
                    "reply": f"rendered frame {frame} to {output}."}
        if action == "publish":
            if not output:
                raise ValueError("anicraft publish needs output=<path>")
            p = _parse_params(params)
            report = publish(doc, kind or "gif", output,
                             start=int(p.get("start", 0)),
                             end=p.get("end"))
            return {"format": kind or "gif",
                    "paths": report["paths"],
                    "frame_count": report["frame_count"],
                    "reply": (f"published {kind or 'gif'}: "
                              f"{report['frame_count']} frames to "
                              f"{report['paths'][0]}.")}
        if action == "export":
            if not output:
                raise ValueError("anicraft export needs output=<path.xfl>")
            target = export_xfl(doc, output)
            return {"output": str(target),
                    "reply": f"exported XFL to {target}."}
        raise ValueError(f"unhandled anicraft action {action!r}")
