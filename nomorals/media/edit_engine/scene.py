"""One canonical Scene (Echelon Wave 1, plan §1.5a).

Devon grew four incompatible scene vocabularies:

* ``edit_engine`` spec scene dicts — ``{"image", "duration", "effect",
  "zoom_direction"}`` (see :func:`spec._scene_to_clip`);
* ``contentops.niches`` :class:`SceneVisual` — a visual *plan* entry
  (imggen prompt + ken-burns motion directive);
* ``videogen.chaining`` scene dicts — ``{"prompt", "mode", "image",
  "text", "duration_s", "seed", ...}`` for chained generation;
* ``director.breakdown`` :class:`Shot` — a scene.shot with a design
  spec plus production state (status, qa, regen_count).

:class:`Scene` is the one type they converge on: a timed span with a
source, camera/motion properties, and a design ref. The legacy types
become *constructors and projections* of it — converters live here, at
the boundary, so producers and consumers keep working unchanged:

* :meth:`Scene.from_spec_scene` / :meth:`Scene.to_spec_scene` —
  the edit_engine spec vocabulary;
* :meth:`Scene.from_scene_visual` / :meth:`Scene.to_scene_visual` —
  the contentops visual-plan vocabulary;
* :meth:`Scene.from_chain_scene` / :meth:`Scene.to_chain_scene` —
  the videogen chaining vocabulary;
* :meth:`Scene.from_shot` — the director vocabulary. One-way by
  design (plan §1.5a): a Shot is a Scene *plus production state*;
  status/qa/regen_count stay in ``director`` and are NOT Scene fields.
  The Shot's identity and its full production record ride in
  :attr:`Scene.extras` so the mapping is lossless to audit, and
  ``director`` remains the producer of Scenes from Shots.

Losslessness rule: a field with a canonical home lands there; a field
without one lands in :attr:`Scene.design` (look/motion intent) or
:attr:`Scene.extras` (provenance/production data) under a namespaced
key. Nothing is silently dropped — the conversion tests pin every
legacy field's destination.

Scene kinds — what the source *is*:

* ``clip`` — an existing video clip (path in :attr:`source`);
* ``still`` — an existing still image, materialised at render;
* ``generated`` — a shot to be generated from :attr:`prompt`
  (text-to-video / image-to-video / a planned still);
* ``text`` — a typography scene (the text is the source);
* ``procedural`` — a code-rendered segment (motion-studio class).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ...media_edit.videos import MediaEditError
from .spec import EditSpec
from .text import TextLayer
from .timeline import Clip, Effect, canonical_effect

__all__ = [
    "SCENE_KINDS",
    "Scene",
    "scenes_to_edit_spec",
]

SCENE_KINDS = ("clip", "still", "generated", "text", "procedural")

#: camera moves with a first-class meaning in the engine's effects.
_ZOOM_MOVES = ("zoom_in", "zoom_out")


@dataclass
class Scene:
    """A timed span with a source, camera properties, and a design ref.

    ``duration_s`` is the span's timeline length (always > 0).
    ``at`` is the span's timeline position in seconds (sequences are
    built by cumulative placement; a lone scene sits at 0).
    ``camera`` keys: ``move`` (``zoom_in`` | ``zoom_out`` | ``static``
    | a producer directive string), ``shot_size``, ``angle``, ``lens``,
    ``movement`` — the director CameraSpec vocabulary, as data.
    ``transition``: ``{"kind": ..., "duration": ...}`` for the boundary
    *into* this scene, or ``None`` (hard cut / chain default).
    ``design``: look/motion intent that is data, not structure
    (design-spec dicts, style/motion directives). ``extras``:
    namespaced provenance (origin vocabulary, production records).
    """
    kind: str
    duration_s: float
    source: str = ""
    prompt: str = ""
    at: float = 0.0
    seed: int | None = None
    camera: dict[str, Any] = field(default_factory=dict)
    transition: dict[str, Any] | None = None
    design: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SCENE_KINDS:
            raise MediaEditError(
                f"scene kind must be one of {SCENE_KINDS}, "
                f"got {self.kind!r}")
        self.duration_s = float(self.duration_s)
        if self.duration_s <= 0:
            raise MediaEditError("scene duration_s must be > 0")
        self.at = max(0.0, float(self.at))
        self.camera = dict(self.camera or {})
        self.design = copy.deepcopy(self.design or {})
        self.extras = copy.deepcopy(self.extras or {})
        if self.transition is not None:
            self.transition = dict(self.transition)

    # -- identity ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source,
                "prompt": self.prompt, "duration_s": self.duration_s,
                "at": self.at, "seed": self.seed,
                "camera": dict(self.camera),
                "transition": (dict(self.transition)
                               if self.transition else None),
                "design": copy.deepcopy(self.design),
                "extras": copy.deepcopy(self.extras)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(kind=str(data.get("kind", "clip")),
                   source=str(data.get("source", "")),
                   prompt=str(data.get("prompt", "")),
                   duration_s=float(data.get("duration_s", 0) or 0),
                   at=float(data.get("at", 0.0) or 0.0),
                   seed=data.get("seed"),
                   camera=dict(data.get("camera") or {}),
                   transition=data.get("transition"),
                   design=dict(data.get("design") or {}),
                   extras=dict(data.get("extras") or {}))

    def copy(self) -> "Scene":
        return Scene.from_dict(self.to_dict())

    # -- engine projections ---------------------------------------------
    def to_clip(self) -> Clip:
        """Scene → engine Clip (the renderable projection).

        Mirrors :func:`spec._scene_to_clip` semantics: a still's span
        is its source window (``end=duration``); a zoom camera move
        becomes the ``ken_burns`` effect with a direction. Text scenes
        have no clip — use :meth:`to_text_layer`.
        """
        if self.kind == "text":
            raise MediaEditError(
                "text scenes project to TextLayer, not Clip — "
                "use to_text_layer()")
        effects: tuple[Effect, ...] = ()
        move = str(self.camera.get("move", "") or "")
        if move in _ZOOM_MOVES:
            effects = (Effect(name="ken_burns",
                              params={"direction":
                                      "out" if move == "zoom_out"
                                      else "in"}),)
        elif move == "ken_burns":
            # ken-burns family, direction unspecified by the source
            # vocabulary — no param: the engine's own default ("in")
            # applies at render, and the Scene invents nothing.
            effects = (Effect(name="ken_burns", params={}),)
        elif move and move != "static":
            effects = (Effect(name=move, params={}),)
        elif self.design.get("effect"):
            # a non-zoom effect carried as design data (the spec
            # vocabulary's shape) still lands on the clip
            effects = (Effect(name=str(self.design["effect"]),
                              params={}),)
        return Clip(path=self.source, end=float(self.duration_s),
                    at=float(self.at), effects=effects)

    def to_text_layer(self) -> TextLayer:
        """Text scene → engine TextLayer over the scene's span."""
        if self.kind != "text":
            raise MediaEditError(
                f"only text scenes project to TextLayer, got {self.kind!r}")
        return TextLayer(text=self.prompt or self.source,
                         start=float(self.at),
                         end=float(self.at) + float(self.duration_s))

    # -- vocabulary 1: edit_engine spec scene dicts ----------------------
    @classmethod
    def from_spec_scene(cls, scene: Any) -> "Scene":
        """Spec dict ``{"image", "duration", "effect", "zoom_direction"}``
        (or a duck-typed object with those attributes) → Scene."""
        if isinstance(scene, dict):
            get = scene.get
        else:
            get = lambda k, d=None: getattr(scene, k, d)  # noqa: E731
        image = str(get("image", "") or "")
        duration = get("duration")
        effect = canonical_effect(str(get("effect", "") or ""))
        zoom_dir = str(get("zoom_direction", "in") or "in").lower()
        camera: dict[str, Any] = {}
        design: dict[str, Any] = {}
        if effect in ("kenburns", "ken_burns"):
            camera["move"] = "zoom_out" if zoom_dir == "out" else "zoom_in"
        elif effect:
            # a non-zoom effect is look data on the scene, kept verbatim
            design["effect"] = effect
        return cls(kind="still", source=image,
                   duration_s=float(duration) if duration else 4.0,
                   at=float(get("at", 0.0) or 0.0),
                   camera=camera, design=design,
                   extras={"origin": "spec_scene"})

    def to_spec_scene(self) -> dict[str, Any]:
        """Scene → spec scene dict (the ``EditSpec(scenes=...)`` shape).

        Only still scenes project losslessly — the spec vocabulary is
        stills-only by construction (``_scene_to_clip`` materialises
        stills). Anything else raises rather than mislabelling itself.
        """
        if self.kind != "still":
            raise MediaEditError(
                f"spec scene dicts are stills-only; a {self.kind!r} "
                f"scene has no spec-scene projection")
        move = str(self.camera.get("move", "") or "")
        if move in _ZOOM_MOVES:
            effect, zoom = "kenburns", ("out" if move == "zoom_out"
                                        else "in")
        elif self.design.get("effect"):
            effect, zoom = str(self.design["effect"]), "in"
        else:
            effect, zoom = "", "in"
        return {"image": self.source, "duration": float(self.duration_s),
                "effect": effect, "zoom_direction": zoom}

    # -- vocabulary 2: contentops SceneVisual -----------------------------
    @classmethod
    def from_scene_visual(cls, sv: Any) -> "Scene":
        """``contentops.niches.SceneVisual`` → Scene.

        A SceneVisual is a *plan*: the image does not exist yet, so the
        scene kind is ``generated`` and the imggen prompt is the source
        of truth (named decision — the plan precedes generation). The
        motion directive maps by the same rule the contentops pipeline
        already uses (``extract_scene_motions``): ``zoom-in-*`` →
        zoom in, ``zoom-out-*`` → zoom out, ``static`` → static;
        anything else is kept verbatim as the camera move AND in
        ``design["motion"]``, so no directive is normalised away.
        """
        motion = str(getattr(sv, "motion", "") or "")
        m = motion.lower()
        if m.startswith("zoom-out"):
            move = "zoom_out"
        elif m.startswith("zoom-in"):
            move = "zoom_in"
        elif "static" in m:
            move = "static"
        else:
            move = motion
        transition = str(getattr(sv, "transition", "cut") or "cut")
        return cls(
            kind="generated",
            prompt=str(getattr(sv, "imggen_prompt", "") or ""),
            duration_s=float(getattr(sv, "duration_s", 4.0) or 4.0),
            camera={"move": move} if move else {},
            transition={"kind": transition},
            design={"motion": motion},
            extras={"origin": "scene_visual",
                    "index": getattr(sv, "index", None),
                    "caption_hook": str(
                        getattr(sv, "caption_hook", "") or "")})

    def to_scene_visual(self, index: int | None = None) -> Any:
        """Scene → ``contentops.niches.SceneVisual``.

        Only ``generated``/``still`` scenes are visual-plan entries.
        The motion directive is rebuilt from the camera move
        (``zoom-in-slow`` / ``zoom-out-slow`` / ``static``); a verbatim
        directive preserved in ``design["motion"]`` wins, so a
        SceneVisual that came in through :meth:`from_scene_visual`
        round-trips exactly.
        """
        from ..contentops.niches.base import SceneVisual

        if self.kind not in ("generated", "still"):
            raise MediaEditError(
                f"a {self.kind!r} scene is not a visual-plan entry")
        move = str(self.camera.get("move", "") or "")
        motion = str(self.design.get("motion", "") or "")
        if not motion:
            motion = {"zoom_in": "zoom-in-slow",
                      "zoom_out": "zoom-out-slow",
                      "static": "static"}.get(move, move)
        return SceneVisual(
            index=(int(self.extras.get("index", 0) or 0)
                   if index is None else int(index)),
            imggen_prompt=self.prompt,
            motion=motion or "zoom-in-slow",
            duration_s=float(self.duration_s),
            transition=str((self.transition or {}).get("kind", "cut")),
            caption_hook=str(self.extras.get("caption_hook", "") or ""))

    # -- vocabulary 3: videogen chaining scene dicts ----------------------
    @classmethod
    def from_chain_scene(cls, scene: Any, *, index: int = 0) -> "Scene":
        """Chaining scene dict (or bare prompt string) → Scene.

        Modes map to kinds: ``image`` → still, ``text`` → text,
        ``t2v``/``i2v`` → generated, ``extend`` → generated (the
        extension intent is kept in ``design``). Per-scene generation
        selection (``backend``/``model``/``lora``) is generation
        provenance, not scene structure — it rides in ``extras``
        verbatim so the chain can be rebuilt exactly.
        """
        if isinstance(scene, str):
            scene = {"prompt": scene}
        d = dict(scene)
        mode = str(d.get("mode", "t2v") or "t2v").lower()
        kind = {"image": "still", "text": "text"}.get(mode, "generated")
        design: dict[str, Any] = {}
        camera: dict[str, Any] = {}
        if mode == "extend":
            design["chain_mode"] = "extend"
        if mode == "image":
            # the chain renders image scenes through kenburns
            # move="auto" — a seed-resolved move from the ken-burns
            # family, with NO direction in the vocabulary. The Scene
            # records the family and the chain's own directive; it
            # does not invent a direction.
            camera["move"] = "ken_burns"
            design["chain_move"] = "auto"
        extras: dict[str, Any] = {"origin": "chain_scene", "index": index,
                                  "mode": mode}
        for key in ("backend", "model", "lora"):
            if d.get(key) is not None:
                extras[f"chain_{key}"] = copy.deepcopy(d[key])
        prompt = str(d.get("prompt", "") or "")
        if mode == "text" and not prompt:
            prompt = str(d.get("text", "") or "")
        return cls(
            kind=kind,
            source=str(d.get("image", "") or ""),
            prompt=prompt,
            duration_s=float(d.get("duration_s", 5.0) or 5.0),
            seed=(int(d["seed"]) if d.get("seed") is not None else None),
            camera=camera, design=design, extras=extras)

    def to_chain_scene(self) -> dict[str, Any]:
        """Scene → chaining scene dict (the ``chain_scenes`` input shape).

        Mode is recovered from kind + the recorded origin mode:
        generated → its original mode (``t2v``/``i2v``/``extend``);
        a generated scene with an image source and no recorded mode is
        ``i2v``. Selection overrides recorded in ``extras`` are
        restored to the dict's top level, where the chain reads them.
        """
        mode = str(self.extras.get("mode", "") or "")
        if self.kind == "still":
            mode = "image"
        elif self.kind == "text":
            mode = "text"
        elif not mode:
            mode = "i2v" if self.source else "t2v"
        out: dict[str, Any] = {"prompt": self.prompt, "mode": mode,
                               "duration_s": float(self.duration_s)}
        if self.source:
            out["image"] = self.source
        if self.kind == "text":
            out["text"] = self.prompt or self.source
        if self.seed is not None:
            out["seed"] = int(self.seed)
        for key in ("backend", "model", "lora"):
            if f"chain_{key}" in self.extras:
                out[key] = copy.deepcopy(self.extras[f"chain_{key}"])
        return out

    # -- vocabulary 4: director Shot ---------------------------------------
    @classmethod
    def from_shot(cls, shot: Any) -> "Scene":
        """``director.breakdown.Shot`` → Scene (plan §1.5a mapping).

        The Shot's *scene* content maps canonically: description is the
        generation prompt, the design spec's camera becomes the Scene
        camera, the design itself rides as the design ref (a dict —
        Scene stays serializable), and a generated clip, once the shoot
        produces one, becomes the source. Production state (status,
        qa, regen_count, engine) stays director's — it is recorded in
        ``extras["shot"]`` for provenance, never promoted to Scene
        structure, and there is deliberately no ``to_shot``: building
        a Shot is the director's job.
        """
        design = getattr(shot, "design", None)
        design_dict = design.to_dict() if design is not None else {}
        camera: dict[str, Any] = {}
        cam = design_dict.get("camera") or {}
        for key in ("shot_size", "angle", "lens"):
            if cam.get(key):
                camera[key] = cam[key]
        if cam.get("movement"):
            camera["movement"] = cam["movement"]
            # the director's movement vocabulary → the engine's move
            # vocabulary, for the moves the engine has: push-in is a
            # zoom-in, pull-back a zoom-out; anything else stays as
            # the verbatim movement (and in the design ref).
            mv = str(cam["movement"]).lower()
            if "zoom-in" in mv or "push-in" in mv or mv == "push in":
                camera["move"] = "zoom_in"
            elif ("zoom-out" in mv or "pull-back" in mv
                    or "pull back" in mv):
                camera["move"] = "zoom_out"
            elif "static" in mv:
                camera["move"] = "static"
        clip_path = str(getattr(shot, "clip_path", "") or "")
        return cls(
            kind="clip" if clip_path else "generated",
            source=clip_path,
            prompt=str(getattr(shot, "description", "") or ""),
            duration_s=float(getattr(shot, "duration_s", 4.0) or 4.0),
            camera=camera,
            design={"design_spec": design_dict} if design_dict else {},
            extras={"origin": "director_shot",
                    "shot": {
                        "id": getattr(shot, "id", ""),
                        "scene_number": getattr(shot, "scene_number",
                                                None),
                        "index": getattr(shot, "index", None),
                        "kind": getattr(shot, "kind", ""),
                        "characters": list(
                            getattr(shot, "characters", []) or []),
                        "speaker": getattr(shot, "speaker", ""),
                        "dialogue": getattr(shot, "dialogue", ""),
                        "performance_note": getattr(
                            shot, "performance_note", ""),
                        "status": getattr(shot, "status", ""),
                        "engine": getattr(shot, "engine", ""),
                        "qa": copy.deepcopy(
                            getattr(shot, "qa", {}) or {}),
                        "regen_count": getattr(shot, "regen_count", 0),
                    }})


def scenes_to_edit_spec(scenes: list[Scene], **spec_kw: Any) -> EditSpec:
    """A scene sequence → one EditSpec (the shared render target).

    Non-text scenes become clips in order (timeline positions are the
    cumulative spans, so the spec's cut matches the sequence); text
    scenes become text layers over their spans. Any per-scene
    transition recorded on a scene after the first becomes the spec's
    boundary transition when they all agree; mixed transitions are the
    Timeline layer's business and raise here rather than being
    silently flattened to one.

    ``format="<name>"`` (§1.5b) applies a format registry entry to the
    built spec — the format's canvas and caption defaults steer the
    render; the application record rides on the returned spec as
    ``format_application``. Explicit ``spec_kw`` canvas values are
    overridden by the format (naming a format is the decision).
    """
    clips: list[Clip] = []
    layers: list[TextLayer] = []
    transitions: list[str] = []
    at = 0.0
    for i, s in enumerate(scenes):
        placed = s.copy()
        placed.at = at
        at += float(s.duration_s)
        if s.kind == "text":
            layers.append(placed.to_text_layer())
        else:
            clips.append(placed.to_clip())
        if i > 0 and s.transition:
            transitions.append(str(s.transition.get("kind", "cut")))
    if transitions:
        kinds = set(transitions)
        if len(kinds) > 1:
            raise MediaEditError(
                f"mixed scene transitions {sorted(kinds)} need the "
                f"Timeline layer — one EditSpec carries one transition")
        spec_kw.setdefault("transition", transitions[0])
    fmt = spec_kw.pop("format", None)
    spec = EditSpec(clips=clips, text_layers=layers, **spec_kw)
    if fmt:
        from .formats import apply_format
        spec, record = apply_format(spec, str(fmt))
        spec.format_application = record  # type: ignore[attr-defined]
    return spec
