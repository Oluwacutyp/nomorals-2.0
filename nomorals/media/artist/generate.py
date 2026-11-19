"""The artist-IDE engine — scene → staged frame → her own rungs.

The orchestration layer of the ArtCraft port. A scene is rendered to
a staged frame (:func:`nomorals.media.artist.compose.render_staged_frame`)
and then generated through Devon's own rungs — never a paid service:

* image: imggen :class:`Studio` (text-to-image, img2img, inpaint,
  upscale) over the staged frame as the control reference;
* cutout: imggen bgremove (the background-removal ladder);
* video: videogen :func:`chain_scenes` (neural when the host can run
  it, motion studio otherwise) across the scene's shots, with
  ``talk`` shots lip-synced through media_edit
  :func:`avatar.talking_head`, and the cut assembled by motion-studio
  :func:`montage.assemble`;
* narration: the voice engine (UniversalTTS) mixed under the cut when
  asked — a missing TTS backend is reported as ``narration: missing``,
  never faked.

Every rung is injectable through the constructor, so tests drive the
orchestration with fake rungs while production resolves the real ones
lazily at call time. A rung that is missing or fails returns
``{"ok": False, "error": <the wall, named>}`` — never a fake file,
never a silent substitution. The neural rungs are import-gated and
need a GPU this box does not have; the motion/classical chain is the
real proof path here.
"""
from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compose import render_staged_frame
from .scene import ArtistError, Scene
from .store import ArtistStore

_RENDERS = "renders"


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _as_path(result: Any) -> str:
    """Normalise a rung's return value to an output path string."""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        return str(result[0]) if result else ""
    if isinstance(result, dict):
        for key in ("path", "output", "final_path"):
            if result.get(key):
                return str(result[key])
        return ""
    path = getattr(result, "path", None) or getattr(result, "final_path", None)
    return str(path) if path else ""


class ArtistEngine:
    """Scene orchestration over Devon's own image/video rungs."""

    def __init__(self, store: ArtistStore, *, home: str | Path | None = None,
                 studio: Any = None, video_generate: Any = None,
                 chain_scenes: Any = None, assemble: Any = None,
                 kenburns: Any = None, talking_head: Any = None,
                 bgremove: Any = None, upscale: Any = None,
                 speak: Any = None) -> None:
        self.store = store
        self.home = Path(home) if home is not None else store.path.parent
        self._studio = studio
        self._video_generate = video_generate
        self._chain_scenes = chain_scenes
        self._assemble = assemble
        self._kenburns = kenburns
        self._talking_head = talking_head
        self._bgremove = bgremove
        self._upscale = upscale
        self._speak = speak

    # -- rung resolution (lazy: the real rungs import at call time) --
    def _resolve_studio(self) -> Any:
        if self._studio is not None:
            return self._studio
        try:
            from ..imggen.studio import Studio
        except ImportError as exc:
            raise ArtistError(
                "image generation is unavailable: the imggen organ "
                f"cannot be imported ({exc})") from exc
        return Studio()

    def _resolve_chain(self) -> Any:
        if self._chain_scenes is not None:
            return self._chain_scenes
        try:
            from ..videogen.chaining import chain_scenes
        except ImportError as exc:
            raise ArtistError(
                "video chaining is unavailable: the videogen organ "
                f"cannot be imported ({exc})") from exc
        return chain_scenes

    def _resolve_assemble(self) -> Any:
        if self._assemble is not None:
            return self._assemble
        try:
            from ..motion_studio.montage import assemble
        except ImportError as exc:
            raise ArtistError(
                "video assembly is unavailable: the motion-studio organ "
                f"cannot be imported ({exc})") from exc
        return assemble

    def _resolve_kenburns(self) -> Any:
        if self._kenburns is not None:
            return self._kenburns
        try:
            from ..motion_studio.kenburns import kenburns
        except ImportError as exc:
            raise ArtistError(
                "motion fallback is unavailable: the motion-studio organ "
                f"cannot be imported ({exc})") from exc
        return kenburns

    def _resolve_talking_head(self) -> Any:
        if self._talking_head is not None:
            return self._talking_head
        try:
            from ...media_edit.avatar import (
                avatar_available,
                talking_head,
                wav2lip_available,
            )
        except ImportError as exc:
            raise ArtistError(
                "talking-head is unavailable: the avatar organ cannot "
                f"be imported ({exc})") from exc
        latent_ok, latent_reason = avatar_available()
        wave_ok, wave_reason = wav2lip_available()
        if not latent_ok and not wave_ok:
            raise ArtistError(
                "talking-head is unavailable: "
                f"LatentSync ({latent_reason}); Wav2Lip ({wave_reason})")
        return talking_head

    def _resolve_bgremove(self) -> Any:
        if self._bgremove is not None:
            return self._bgremove

        def _real_bgremove(src: str, dst: str) -> str:
            from ..imggen.bgremove import remove_background_file
            record = remove_background_file(src, dst)
            return str(record.get("output") or dst)

        return _real_bgremove

    def _resolve_upscale(self) -> Any:
        if self._upscale is not None:
            return self._upscale

        def _real_upscale(src: str, scale: float, dst: str) -> str:
            studio = self._resolve_studio()
            result = studio.upscale(src, scale=scale, save_to=dst)
            return _as_path(result)

        return _real_upscale

    def _resolve_speak(self) -> Any:
        if self._speak is not None:
            return self._speak

        def _real_speak(text: str, out_path: str) -> str:
            from ...voice.tts import UniversalTTS
            engine = UniversalTTS(backend="auto")
            result = engine.speak(text, out_path=out_path)
            if isinstance(result, dict):
                if result.get("ok") is False:
                    raise ArtistError(
                        "narration failed: "
                        + str(result.get("reason") or result.get("error") or "unknown"))
                path = result.get("path") or result.get("output") or out_path
                return str(path)
            return out_path

        return _real_speak

    # -- helpers -----------------------------------------------------
    def _renders_dir(self) -> Path:
        directory = self.home / _RENDERS
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _scene(self, scene_id: str) -> Scene:
        scene = self.store.get(scene_id)
        if scene is None:
            raise ArtistError(f"no scene with id {scene_id!r}")
        return scene

    def _record(self, scene: Scene, kind: str, record: dict[str, Any]) -> None:
        scene.artifacts[kind] = record
        self.store.save(scene)

    def _staged_frame(self, scene: Scene, out_name: str) -> str:
        frame = render_staged_frame(scene)
        path = self._renders_dir() / out_name
        frame.save(path, format="PNG")
        return str(path)

    # -- capability --------------------------------------------------
    def capability_report(self) -> dict[str, Any]:
        """Which rungs this host can actually run, each wall named."""
        try:
            from ..imggen import TORCH_AVAILABLE
        except ImportError:
            TORCH_AVAILABLE = False
        image_models: list[str] = []
        if TORCH_AVAILABLE:
            try:
                from ..imggen.registry import list_models
                image_models = [str(m) for m in list_models()]
            except Exception:  # noqa: BLE001 — capability is best-effort
                image_models = []
        imggen = {
            "available": bool(TORCH_AVAILABLE),
            "reason": (
                "torch is installed"
                + (f" — {len(image_models)} local image model(s)"
                   if image_models else " — no local image checkpoint found")
                if TORCH_AVAILABLE
                else "torch is not installed — neural image rungs "
                     "unavailable on this host"),
            "models": image_models,
        }
        try:
            from ..videogen.capabilities import neural_capability
            cap = neural_capability("auto")
            videogen = {
                "available": bool(cap.available),
                "reason": f"{cap.reason} (backend: {cap.backend})",
                "backend": cap.backend,
            }
        except Exception as exc:  # noqa: BLE001 — capability is best-effort
            videogen = {"available": False,
                        "reason": f"videogen capability probe failed: {exc}",
                        "backend": "none"}
        try:
            from ...media_edit.avatar import (
                avatar_available,
                wav2lip_available,
            )
            latent_ok, latent_reason = avatar_available()
            wave_ok, wave_reason = wav2lip_available()
            talking = {
                "available": bool(latent_ok or wave_ok),
                "reason": (
                    "LatentSync ready" if latent_ok else
                    "Wav2Lip ready" if wave_ok else
                    f"LatentSync ({latent_reason}); "
                    f"Wav2Lip ({wave_reason})"),
            }
        except Exception as exc:  # noqa: BLE001 — capability is best-effort
            talking = {"available": False,
                       "reason": f"avatar probe failed: {exc}"}
        ffmpeg = shutil.which("ffmpeg")
        return {
            "imggen": imggen,
            "videogen": videogen,
            "talking_head": talking,
            "ffmpeg": {
                "available": bool(ffmpeg),
                "reason": (f"ffmpeg found at {ffmpeg}" if ffmpeg
                           else "ffmpeg is not installed — motion video "
                                "cannot be encoded on this host"),
            },
            "models": {"image": image_models},
        }

    # -- preview -----------------------------------------------------
    def compose_preview(self, scene_id: str, *,
                        out: str | None = None) -> dict[str, Any]:
        try:
            scene = self._scene(scene_id)
            path = out or str(self._renders_dir()
                              / f"{scene.id}-staged-{_stamp()}.png")
            frame = render_staged_frame(scene)
            frame.save(path, format="PNG")
            self._record(scene, "preview", {"path": path, "kind": "preview"})
            return {"ok": True, "path": path, "id": scene.id}
        except ArtistError as exc:
            return {"ok": False, "error": str(exc)}

    # -- image -------------------------------------------------------
    def generate_image(self, scene_id: str, *, prompt: str = "",
                       mode: str = "auto", shot: str | None = None,
                       seed: int = 0, strength: float = 0.6,
                       aspect_ratio: str | None = None) -> dict[str, Any]:
        try:
            scene = self._scene(scene_id)
            if shot:
                prompt = prompt or scene.shot(shot).prompt
            if not prompt:
                return {"ok": False,
                        "error": "a prompt is required to generate an image"}
            if mode not in ("auto", "text", "img2img"):
                return {"ok": False,
                        "error": f"unknown generate mode {mode!r} — "
                                 "one of auto, text, img2img"}
            studio = self._resolve_studio()
            out = str(self._renders_dir()
                      / f"{scene.id}-gen-{_stamp()}.png")
            kwargs: dict[str, Any] = {}
            if seed:
                kwargs["seed"] = seed
            if aspect_ratio:
                kwargs["aspect_ratio"] = aspect_ratio
            if mode == "text":
                resolved = "text"
                result = studio.generate(prompt, save_to=out, **kwargs)
            else:
                resolved = "img2img"
                frame = self._staged_frame(
                    scene, f"{scene.id}-staged-{_stamp()}.png")
                result = studio.img2img(frame, prompt, strength=strength,
                                        save_to=out, **kwargs)
            path = _as_path(result)
            if not path or not Path(path).exists():
                return {"ok": False,
                        "error": "the image rung returned no file — "
                                 "generation did not produce an image"}
            self._record(scene, "image", {
                "path": path, "rung": "imggen-studio", "mode": resolved,
                "prompt": prompt, "seed": seed})
            return {"ok": True, "path": path, "id": scene.id,
                    "mode": resolved, "rung": "imggen-studio"}
        except ArtistError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a rung failure is a dict
            return {"ok": False,
                    "error": f"image generation failed: {exc}"}

    def edit_image(self, scene_id: str, image: str, prompt: str, *,
                   mask: str | None = None,
                   strength: float = 0.6) -> dict[str, Any]:
        try:
            scene = self._scene(scene_id)
            if not Path(image).exists():
                return {"ok": False, "error": f"no such image: {image}"}
            if not prompt:
                return {"ok": False,
                        "error": "a prompt is required to edit an image"}
            studio = self._resolve_studio()
            out = str(self._renders_dir()
                      / f"{scene.id}-edit-{_stamp()}.png")
            if mask:
                if not Path(mask).exists():
                    return {"ok": False, "error": f"no such mask: {mask}"}
                result = studio.inpaint(image, mask, prompt, save_to=out)
                resolved = "inpaint"
            else:
                result = studio.img2img(image, prompt, strength=strength,
                                        save_to=out)
                resolved = "img2img"
            path = _as_path(result)
            if not path or not Path(path).exists():
                return {"ok": False,
                        "error": "the image rung returned no file — "
                                 "the edit did not produce an image"}
            self._record(scene, "edit", {
                "path": path, "rung": "imggen-studio", "mode": resolved,
                "prompt": prompt, "source": image})
            return {"ok": True, "path": path, "id": scene.id,
                    "mode": resolved, "rung": "imggen-studio"}
        except ArtistError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a rung failure is a dict
            return {"ok": False, "error": f"image edit failed: {exc}"}

    def cutout(self, scene_id: str, image: str) -> dict[str, Any]:
        try:
            scene = self._scene(scene_id)
            if not Path(image).exists():
                return {"ok": False, "error": f"no such image: {image}"}
            bgremove = self._resolve_bgremove()
            out = str(self._renders_dir()
                      / f"{scene.id}-cutout-{_stamp()}.png")
            path = bgremove(image, out)
            path = _as_path(path) or out
            if not Path(path).exists():
                return {"ok": False,
                        "error": "the cutout rung returned no file"}
            self._record(scene, "cutout", {
                "path": path, "rung": "imggen-bgremove", "source": image})
            return {"ok": True, "path": path, "id": scene.id,
                    "rung": "imggen-bgremove"}
        except ArtistError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a rung failure is a dict
            return {"ok": False, "error": f"cutout failed: {exc}"}

    def upscale(self, scene_id: str, image: str,
                scale: float = 2.0) -> dict[str, Any]:
        try:
            scene = self._scene(scene_id)
            if not Path(image).exists():
                return {"ok": False, "error": f"no such image: {image}"}
            if scale <= 0:
                return {"ok": False,
                        "error": f"upscale scale must be positive, got {scale}"}
            upscale = self._resolve_upscale()
            out = str(self._renders_dir()
                      / f"{scene.id}-upscale-{_stamp()}.png")
            path = _as_path(upscale(image, scale, out)) or out
            if not Path(path).exists():
                return {"ok": False,
                        "error": "the upscale rung returned no file"}
            self._record(scene, "upscale", {
                "path": path, "rung": "imggen-upscale", "scale": scale,
                "source": image})
            return {"ok": True, "path": path, "id": scene.id,
                    "scale": scale, "rung": "imggen-upscale"}
        except ArtistError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a rung failure is a dict
            return {"ok": False, "error": f"upscale failed: {exc}"}

    # -- video -------------------------------------------------------
    def video(self, scene_id: str, *, backend: str = "auto",
              narrate: bool = False, out: str | None = None) -> dict[str, Any]:
        """Cut the scene's shots into one video.

        Non-talk shots go through the videogen chain (neural when the
        host can run it, motion studio otherwise); a chain failure
        falls back to per-shot kenburns + assembly, and the record
        says so. ``talk`` shots are lip-synced first, then spliced
        into the cut in shot order — chain shots re-enter the merged
        timeline through their start/duration window in the chained
        film, so interleaved talk shots land in the right place.
        """
        try:
            scene = self._scene(scene_id)
            if not scene.shots:
                return {"ok": False,
                        "error": f"scene {scene.title!r} has no shots — "
                                 "add a shot before making a video"}
            frame = self._staged_frame(
                scene, f"{scene.id}-staged-{_stamp()}.png")
            clips: dict[int, dict[str, Any]] = {}
            chain_specs: list[dict[str, Any]] = []
            chain_shot_index: list[int] = []
            for index, shot in enumerate(scene.shots):
                if shot.mode == "talk":
                    photo = shot.image
                    if not photo and shot.character:
                        ref = scene.character(shot.character).ref_image
                        photo = ref or frame
                    if not photo:
                        photo = frame
                    try:
                        talking_head = self._resolve_talking_head()
                    except ArtistError as exc:
                        return {"ok": False,
                                "error": f"talk shot {shot.name!r} cannot "
                                         f"be lip-synced: {exc}"}
                    clip = str(self._renders_dir()
                               / f"{scene.id}-talk-{index}-{_stamp()}.mp4")
                    try:
                        result = talking_head(photo, shot.lines,
                                              out_path=clip)
                    except Exception as exc:  # noqa: BLE001 — named below
                        return {"ok": False,
                                "error": f"talk shot {shot.name!r} failed: "
                                         f"{exc}"}
                    clip_path = _as_path(result) or clip
                    if not Path(clip_path).exists():
                        return {"ok": False,
                                "error": f"talk shot {shot.name!r} "
                                         "produced no video"}
                    clips[index] = {"kind": "video", "src": clip_path,
                                    "duration": shot.duration_s}
                    continue
                shot_frame = shot.image or frame
                if shot.camera:
                    shot_frame = _render_with_camera(scene, shot.camera, self)
                mode = shot.mode if shot.mode in ("t2v", "i2v", "motion") \
                    else "i2v"
                chain_specs.append({
                    "prompt": shot.prompt,
                    "mode": mode,
                    "image": shot_frame if mode in ("i2v", "motion") else None,
                    "duration_s": shot.duration_s,
                })
                chain_shot_index.append(index)
            final = ""
            rung = ""
            chain_error: Exception | None = None
            if chain_specs:
                chain = None
                try:
                    chain = self._resolve_chain()
                except ArtistError as exc:
                    chain_error = exc
                if chain is not None:
                    try:
                        report = chain(chain_specs, backend=backend)
                        final = _as_path(report)
                    except Exception as exc:  # noqa: BLE001 — motion fallback
                        chain_error = exc
                        final = ""
                if final and Path(final).exists():
                    rung = "videogen-chain"
                    if clips:
                        # Splice the talk clips in shot order; each
                        # chain shot re-enters through its window.
                        assemble = self._resolve_assemble()
                        merged: list[dict[str, Any]] = []
                        offset = 0.0
                        chain_pos = 0
                        for index in range(len(scene.shots)):
                            if index in clips:
                                merged.append(clips[index])
                            else:
                                spec = chain_specs[chain_pos]
                                merged.append({
                                    "kind": "video", "src": final,
                                    "start": offset,
                                    "duration": spec["duration_s"]})
                                offset += spec["duration_s"]
                                chain_pos += 1
                        out_path = out or str(self._renders_dir()
                                              / f"{scene.id}-film-{_stamp()}.mp4")
                        final = assemble(merged, out_path,
                                    size=(scene.width, scene.height))
                        rung = "videogen-chain+assemble"
                else:
                    # The chain is down — motion studio per shot, then
                    # assemble the whole cut in shot order.
                    kenburns = self._resolve_kenburns()
                    for spec, scene_index in zip(chain_specs,
                                                 chain_shot_index,
                                                 strict=True):
                        image = spec.get("image") or frame
                        clip = kenburns(image, duration=spec["duration_s"])
                        clips[scene_index] = {
                            "kind": "video", "src": str(clip),
                            "duration": spec["duration_s"]}
                    assemble = self._resolve_assemble()
                    merged = [clips[i] for i in range(len(scene.shots))]
                    out_path = out or str(self._renders_dir()
                                          / f"{scene.id}-film-{_stamp()}.mp4")
                    final = assemble(merged, out_path,
                                    size=(scene.width, scene.height))
                    rung = "motion-studio"
                    if chain_error is not None:
                        rung += f" (chain failed: {chain_error})"
            else:
                assemble = self._resolve_assemble()
                merged = [clips[i] for i in range(len(scene.shots))]
                out_path = out or str(self._renders_dir()
                                      / f"{scene.id}-film-{_stamp()}.mp4")
                final = assemble(merged, out_path,
                                    size=(scene.width, scene.height))
                rung = "talking-head+assemble"
            if not final or not Path(final).exists():
                return {"ok": False,
                        "error": "the video rungs produced no file"}
            narration = "off"
            if narrate:
                narrated = self._narrate(scene, final)
                if narrated.startswith("missing"):
                    narration = narrated
                else:
                    final = narrated
                    narration = "mixed"
            self._record(scene, "video", {
                "path": final, "rung": rung, "shots": len(scene.shots),
                "narration": narration})
            return {"ok": True, "path": final, "id": scene.id,
                    "shots": len(scene.shots), "rung": rung,
                    "narration": narration}
        except ArtistError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a rung failure is a dict
            return {"ok": False, "error": f"video failed: {exc}"}

    def _narrate(self, scene: Scene, final: str) -> str:
        """Mix a spoken narration under ``final``; returns the new path,
        or a ``missing: <reason>`` string when no TTS backend can speak.
        """
        lines = [shot.lines or shot.prompt for shot in scene.shots]
        text = " ".join(line for line in lines if line)
        if not text:
            return "missing: no lines or prompts to narrate"
        try:
            speak = self._resolve_speak()
        except ArtistError as exc:
            return f"missing: {exc}"
        wav = str(self._renders_dir() / f"{scene.id}-narr-{_stamp()}.wav")
        try:
            spoken = speak(text, wav)
            wav_path = _as_path(spoken) or wav
            if not Path(wav_path).exists():
                return "missing: the TTS rung returned no audio"
            assemble = self._resolve_assemble()
            mixed = str(self._renders_dir()
                        / f"{scene.id}-film-narr-{_stamp()}.mp4")
            return assemble([{"kind": "video", "src": final}],
                            mixed, audio=wav_path,
                            size=(scene.width, scene.height))
        except ArtistError as exc:
            return f"missing: {exc}"
        except Exception as exc:  # noqa: BLE001 — narration is optional
            return f"missing: narration failed: {exc}"


def _render_with_camera(scene: Scene, camera: dict[str, Any],
                        engine: ArtistEngine) -> str:
    """Render a staged frame with a shot's camera override."""
    from .compose import render_staged_frame
    from .scene import Camera
    frame = render_staged_frame(scene, camera=Camera.from_dict(camera))
    path = engine._renders_dir() / f"{scene.id}-staged-{_stamp()}.png"
    frame.save(path, format="PNG")
    return str(path)
