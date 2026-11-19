"""Artist-IDE tool — Devon's own ArtCraft port (storytold/artcraft,
dual MIT/Apache-2.0), owner-only.

Build the scene before generating: layers, 3D staging with depth +
camera, posed characters with identity references, one consistent
location across shots — then generate through her own rungs (imggen
Studio, videogen chain, motion studio, talking-head). No paid
service, no per-clip billing, no limits.

Actions (all return ``{"ok": bool, ...}`` dicts, never raise):
create / show / list / delete — scene CRUD;
layer — add a 2D layer (kind=image|text|shape|solid);
pose — pose a character (name=, pose=, joints=);
camera — place the scene camera (fov=, x=, y=, z=);
location — set the consistent location (name=, backdrop=);
shot — add a shot (name=, prompt=, mode=t2v|i2v|motion|talk,
duration_s=, lines=, character=);
stage — stage a 3D item (name=, image=, depth=, x=, y=, character=);
compose — render the staged frame;
generate — generate an image from the scene (prompt=, mode=text|img2img|auto);
edit — img2img / inpaint an image (image=, prompt=, mask=);
cutout — remove the background of an image (image=);
upscale — upscale an image (image=, scale=);
video — cut the scene's shots into one video (backend=, narrate=);
status — which rungs this host can actually run, each wall named.
"""
from __future__ import annotations

from typing import Any

from ..media.artist.generate import ArtistEngine
from ..media.artist.scene import ArtistError, Scene
from ..media.artist.store import ArtistStore, artist_dir, artist_path


def _store(context: Any) -> ArtistStore:
    settings = getattr(context, "settings", None)
    return ArtistStore(artist_path(settings))


def _engine(context: Any) -> ArtistEngine:
    settings = getattr(context, "settings", None)
    return ArtistEngine(_store(context), home=artist_dir(settings))


def _get(store: ArtistStore, scene_id: str) -> Scene | None:
    if not scene_id:
        return None
    return store.get(scene_id)


def _missing(scene_id: str) -> dict[str, Any]:
    return {"ok": False, "error": f"no scene with id {scene_id!r}"}


def register(registry: Any) -> None:
    """Attach the ``artist`` tool."""
    context = registry.context

    @registry.register(
        "artist",
        description=(
            "Artist IDE (ArtCraft port): build the scene before "
            "generating — 2D layers, 3D staging with depth + camera, "
            "posed characters, one consistent location — then generate "
            "through Devon's own image/video rungs. action=create "
            "starts a scene (title=); action=show/list/delete manage "
            "scenes (id=); action=layer adds a 2D layer; action=pose "
            "poses a character; action=camera places the camera; "
            "action=location sets the location; action=shot adds a "
            "shot; action=stage stages a 3D item; action=compose "
            "renders the staged frame; action=generate generates an "
            "image (prompt=, mode=text|img2img|auto); action=edit "
            "edits an image (image=, prompt=, mask=); action=cutout "
            "removes a background (image=); action=upscale upscales "
            "(image=, scale=); action=video cuts the shots into one "
            "video (backend=, narrate=); action=status reports which "
            "rungs this host can run. No paid service, no limits."
        ),
        capability="owner",
    )
    def artist_tool(
        action: str = "list",
        id: str = "",
        title: str = "",
        width: int = 0,
        height: int = 0,
        kind: str = "",
        name: str = "",
        path: str = "",
        text: str = "",
        color: str = "",
        opacity: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        shape: str = "",
        font_size: int = 0,
        pose: str = "",
        joints: dict[str, float] | None = None,
        fov: float = 0.0,
        z: float = 0.0,
        backdrop: str = "",
        notes: str = "",
        prompt: str = "",
        mode: str = "",
        duration_s: float = 0.0,
        lines: str = "",
        character: str = "",
        depth: float = 1.0,
        scale: float = 0.0,
        image: str = "",
        mask: str = "",
        backend: str = "",
        narrate: bool = False,
        strength: float = 0.6,
        seed: int = 0,
    ) -> dict[str, Any]:
        store = _store(context)
        action = (action or "list").strip().lower()
        try:
            if action == "create":
                if not (title or "").strip():
                    return {"ok": False,
                            "error": "give a title: action=create title=..."}
                kwargs: dict[str, Any] = {}
                if width:
                    kwargs["width"] = width
                if height:
                    kwargs["height"] = height
                scene = Scene(title=title.strip(), **kwargs)
                store.save(scene)
                return {"ok": True, "id": scene.id, "title": scene.title,
                        "width": scene.width, "height": scene.height}
            if action == "list":
                return {"ok": True, "scenes": store.list()}
            if action == "show":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                return {"ok": True, "scene": scene.to_dict()}
            if action == "delete":
                if store.delete(id):
                    return {"ok": True, "id": id, "deleted": True}
                return _missing(id)
            if action == "status":
                engine = _engine(context)
                return {"ok": True, "capabilities": engine.capability_report()}
            if action == "layer":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                layer_kw: dict[str, Any] = {"x": x, "y": y, "opacity": opacity}
                if path:
                    layer_kw["path"] = path
                if text:
                    layer_kw["text"] = text
                if color:
                    layer_kw["color"] = color
                if shape:
                    layer_kw["shape"] = shape
                if font_size:
                    layer_kw["font_size"] = font_size
                layer = scene.add_layer(kind=kind or "solid", name=name,
                                        **layer_kw)
                store.save(scene)
                return {"ok": True, "id": scene.id, "layer": layer.to_dict()}
            if action == "pose":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                if not name:
                    return {"ok": False,
                            "error": "give a character: action=pose name=..."}
                if not pose:
                    return {"ok": False,
                            "error": "give a pose: action=pose pose=..."}
                character_obj = scene.pose_character(name, pose,
                                                     joints=joints)
                store.save(scene)
                return {"ok": True, "id": scene.id,
                        "character": character_obj.to_dict()}
            if action == "camera":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                camera_kw: dict[str, Any] = {"x": x, "y": y, "z": z}
                if fov:
                    camera_kw["fov"] = fov
                camera = scene.set_camera(**camera_kw)
                store.save(scene)
                return {"ok": True, "id": scene.id,
                        "camera": camera.to_dict()}
            if action == "location":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                location = scene.set_location(name or "Untitled",
                                              backdrop=backdrop, notes=notes)
                store.save(scene)
                return {"ok": True, "id": scene.id,
                        "location": location.to_dict()}
            if action == "stage":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                item = scene.stage_item(name, image=image, depth=depth,
                                        x=x, y=y,
                                        character=character)
                if scale:
                    item.scale = scale
                    item.validate()
                store.save(scene)
                return {"ok": True, "id": scene.id, "item": item.to_dict()}
            if action == "shot":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                shot_kw: dict[str, Any] = {}
                if mode:
                    shot_kw["mode"] = mode
                if duration_s:
                    shot_kw["duration_s"] = duration_s
                if lines:
                    shot_kw["lines"] = lines
                if character:
                    shot_kw["character"] = character
                if image:
                    shot_kw["image"] = image
                shot = scene.add_shot(name, prompt=prompt, **shot_kw)
                store.save(scene)
                return {"ok": True, "id": scene.id, "shot": shot.to_dict()}
            if action == "compose":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                return _engine(context).compose_preview(scene.id)
            if action == "generate":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                return _engine(context).generate_image(
                    scene.id, prompt=prompt, mode=mode or "auto",
                    seed=seed, strength=strength)
            if action == "edit":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                return _engine(context).edit_image(
                    scene.id, image, prompt, mask=mask or None,
                    strength=strength)
            if action == "cutout":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                return _engine(context).cutout(scene.id, image)
            if action == "upscale":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                return _engine(context).upscale(
                    scene.id, image, scale=scale or 2.0)
            if action == "video":
                scene = _get(store, id)
                if scene is None:
                    return _missing(id)
                return _engine(context).video(
                    scene.id, backend=backend or "auto", narrate=narrate)
            return {"ok": False, "error": f"unknown action {action!r} — "
                    "one of create, show, list, delete, status, layer, "
                    "pose, camera, location, stage, shot, compose, "
                    "generate, edit, cutout, upscale, video"}
        except ArtistError as exc:
            return {"ok": False, "error": str(exc)}
