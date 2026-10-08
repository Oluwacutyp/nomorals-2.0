"""ComfyUI backend for generative image work.

Drives a running ComfyUI server over its HTTP API (stdlib ``urllib`` only —
no new dependencies). Workflows live in :mod:`nomorals.media_edit.workflows`
as ComfyUI API-format JSON and are parameterized in Python (load the dict,
walk nodes, set fields — no string templating).

GPU locking: one module-level :class:`threading.Semaphore` serializes
generations because a single GPU can only run one diffusion job at a time.
Pass ``parallel=N`` for multi-GPU hosts (one slot per GPU). This is
process-scoped: two Devon processes on the same box would not see each
other's lock — run one media worker per GPU host.

Select with ``get_backend("comfy")`` or ``MEDIA_GEN_BACKEND=comfy``.
Host/port from ``COMFYUI_HOST`` / ``COMFYUI_PORT`` (defaults 127.0.0.1:8188).
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind
from .generate import GenerativeBackend, GenerativeEditError, _as_mask
from .images import load_image

_log = get_logger(__name__)

_WORKFLOWS_DIR = Path(__file__).with_name("workflows")

#: Process-scoped GPU lock: one diffusion job at a time per process.
_GPU_SEMAPHORE = threading.Semaphore(1)

_DEFAULT_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
_DEFAULT_PORT = int(os.environ.get("COMFYUI_PORT", "8188") or 8188)

_POLL_INTERVAL = 1.0
_AVAIL_TIMEOUT = 3.0


# ---------------------------------------------------------------------------
# workflow templates
# ---------------------------------------------------------------------------

def _load_workflow(name: str) -> dict[str, Any]:
    path = _WORKFLOWS_DIR / f"{name}.json"
    try:
        with open(path, encoding="utf-8") as fh:
            wf = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerativeEditError(
            f"comfy workflow template {name!r} unreadable: {exc}") from exc
    if not isinstance(wf, dict) or not wf:
        raise GenerativeEditError(
            f"comfy workflow template {name!r} is not a valid node dict")
    return wf


def _nodes(wf: dict[str, Any], class_type: str) -> list[dict[str, Any]]:
    """Nodes of ``class_type`` in document order."""
    return [n for n in wf.values()
            if isinstance(n, dict) and n.get("class_type") == class_type]


def _single(wf: dict[str, Any], class_type: str) -> dict[str, Any]:
    found = _nodes(wf, class_type)
    if len(found) != 1:
        raise GenerativeEditError(
            f"workflow template expects exactly one {class_type}, "
            f"found {len(found)}")
    return found[0]


def _prompt_nodes(wf: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(positive, negative) CLIPTextEncode nodes — positive is first in the
    template document, negative second."""
    encoders = _nodes(wf, "CLIPTextEncode")
    if len(encoders) < 2:
        raise GenerativeEditError(
            "workflow template needs two CLIPTextEncode nodes "
            "(positive, negative)")
    return encoders[0], encoders[1]


def _apply_common(wf: dict[str, Any], *, prompt: str,
                  negative_prompt: str | None,
                  seed: int | None, steps: int | None,
                  guidance_scale: float | None,
                  checkpoint: str | None) -> None:
    positive, negative = _prompt_nodes(wf)
    positive["inputs"]["text"] = prompt
    negative["inputs"]["text"] = negative_prompt or ""
    sampler = _single(wf, "KSampler")
    inputs = sampler["inputs"]
    inputs["seed"] = int(seed) if seed is not None else _random_seed()
    if steps:
        inputs["steps"] = int(steps)
    if guidance_scale:
        inputs["cfg"] = float(guidance_scale)
    if checkpoint:
        _single(wf, "CheckpointLoaderSimple")["inputs"]["ckpt_name"] = checkpoint


def _random_seed() -> int:
    import random
    return random.randint(0, 2**31 - 1)


def _txt2img_workflow(*, prompt: str, negative_prompt: str | None,
                      seed: int | None, steps: int | None,
                      guidance_scale: float | None,
                      width: int | None, height: int | None,
                      n: int, checkpoint: str | None) -> dict[str, Any]:
    wf = _load_workflow("txt2img")
    _apply_common(wf, prompt=prompt, negative_prompt=negative_prompt,
                  seed=seed, steps=steps, guidance_scale=guidance_scale,
                  checkpoint=checkpoint)
    latent = _single(wf, "EmptyLatentImage")["inputs"]
    latent["width"] = int(width or 512)
    latent["height"] = int(height or 512)
    latent["batch_size"] = max(1, int(n))
    return wf


def _img2img_workflow(*, image_name: str, prompt: str,
                      negative_prompt: str | None, strength: float,
                      seed: int | None, steps: int | None,
                      guidance_scale: float | None,
                      checkpoint: str | None) -> dict[str, Any]:
    if not 0 < strength <= 1:
        raise GenerativeEditError(
            f"img2img strength must be in (0, 1], got {strength}")
    wf = _load_workflow("img2img")
    _apply_common(wf, prompt=prompt, negative_prompt=negative_prompt,
                  seed=seed, steps=steps, guidance_scale=guidance_scale,
                  checkpoint=checkpoint)
    _single(wf, "LoadImage")["inputs"]["image"] = image_name
    _single(wf, "KSampler")["inputs"]["denoise"] = float(strength)
    return wf


def _inpaint_workflow(*, image_name: str, mask_name: str, prompt: str,
                      negative_prompt: str | None,
                      seed: int | None, steps: int | None,
                      guidance_scale: float | None,
                      checkpoint: str | None) -> dict[str, Any]:
    wf = _load_workflow("inpaint")
    _apply_common(wf, prompt=prompt, negative_prompt=negative_prompt,
                  seed=seed, steps=steps, guidance_scale=guidance_scale,
                  checkpoint=checkpoint)
    loaders = _nodes(wf, "LoadImage")
    if len(loaders) < 2:
        raise GenerativeEditError(
            "inpaint template needs two LoadImage nodes (image, mask)")
    loaders[0]["inputs"]["image"] = image_name
    loaders[1]["inputs"]["image"] = mask_name
    return wf


def _upscale_workflow(*, image_name: str,
                      scale: float) -> dict[str, Any]:
    if scale <= 0:
        raise GenerativeEditError(f"upscale scale must be > 0, got {scale}")
    wf = _load_workflow("upscale")
    _single(wf, "LoadImage")["inputs"]["image"] = image_name
    _single(wf, "UpscaleImageBy")["inputs"]["scale_by"] = float(scale)
    return wf


#: Node types the Wan 2.2 template needs. ComfyUI reports unknown node
#: types in /prompt's ``node_errors``; when that happens we name the
#: custom-node packs to install instead of failing cryptically.
_WAN_WORKFLOW_NOTE = (
    "the wan22_t2v workflow needs ComfyUI-WanVideoWrapper "
    "(WanVideoModelLoader, WanVideoVAELoader, WanVideoTextEncode, "
    "WanVideoEmptyEmbeds, WanVideoEmptyLatent, WanVideoSampler, "
    "WanVideoDecode) and ComfyUI-VideoHelperSuite (VHS_VideoCombine). "
    "Install both custom-node packs in ComfyUI Manager, restart, "
    "and place the Wan 2.2 + VAE weights in ComfyUI/models/diffusion_models "
    "and ComfyUI/models/vae."
)


def _wan22_t2v_workflow(*, prompt: str, negative_prompt: str | None,
                        width: int | None, height: int | None,
                        num_frames: int, steps: int | None,
                        seed: int | None, cfg: float | None,
                        model_file: str | None,
                        vae_file: str | None) -> dict[str, Any]:
    """Parameterize the wan22_t2v template. Node types are walked by
    class_type (same discipline as the image templates); unknown node
    types fail at submit time with _WAN_WORKFLOW_NOTE."""
    if num_frames < 1:
        raise GenerativeEditError(
            f"num_frames must be >= 1, got {num_frames}")
    wf = _load_workflow("wan22_t2v")
    if model_file:
        _single(wf, "WanVideoModelLoader")["inputs"]["model"] = model_file
    if vae_file:
        _single(wf, "WanVideoVAELoader")["inputs"]["model"] = vae_file
    enc = _single(wf, "WanVideoTextEncode")["inputs"]
    enc["positive_prompt"] = prompt
    enc["negative_prompt"] = negative_prompt or ""
    w, h = int(width or 1280), int(height or 720)
    for ct in ("WanVideoEmptyEmbeds", "WanVideoEmptyLatent"):
        node = _single(wf, ct)["inputs"]
        node["width"] = w
        node["height"] = h
        node["num_frames"] = int(num_frames)
    sampler = _single(wf, "WanVideoSampler")["inputs"]
    sampler["seed"] = int(seed) if seed is not None else _random_seed()
    if steps:
        sampler["steps"] = int(steps)
    if cfg:
        sampler["cfg"] = float(cfg)
    sampler["num_frames"] = int(num_frames)
    return wf


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------

class ComfyUIBackend(GenerativeBackend):
    """Generative backend driving a ComfyUI server over HTTP.

    ``name = "comfy"``. Lazy: ``__init__`` never connects; the first real
    call raises :class:`GenerativeEditError` with the server's own error
    text when ComfyUI is unreachable or the workflow fails.
    """

    name = "comfy"

    def __init__(self, host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT,
                 *, timeout_s: float = 600.0,
                 checkpoint: str | None = None,
                 parallel: int = 1,
                 queue_timeout_s: float = 300.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.checkpoint = checkpoint or os.environ.get("COMFYUI_CHECKPOINT")
        self.queue_timeout_s = float(queue_timeout_s)
        # parallel > 1: multi-GPU host, one slot per GPU. parallel == 1 uses
        # the shared process-wide semaphore so separately constructed
        # backends still serialize.
        self._sem = _GPU_SEMAPHORE if parallel <= 1 else threading.Semaphore(parallel)
        self._client_id = uuid.uuid4().hex

    # -- HTTP ---------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def _request(self, method: str, path: str, *,
                 payload: bytes | None = None,
                 content_type: str | None = None,
                 timeout: float | None = None) -> tuple[int, bytes, str]:
        req = urllib.request.Request(self._url(path), data=payload,
                                     method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(
                    req, timeout=self.timeout_s if timeout is None else timeout
            ) as resp:
                return resp.status, resp.read(), resp.headers.get_content_type()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:2000]
            raise GenerativeEditError(
                f"ComfyUI {method} {path} -> HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise GenerativeEditError(
                f"ComfyUI unreachable at {self.host}:{self.port} "
                f"({method} {path}): {exc}") from exc

    def _get_json(self, path: str, *,
                  timeout: float | None = None) -> Any:
        _status, body, _ctype = self._request("GET", path, timeout=timeout)
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise GenerativeEditError(
                f"ComfyUI {path} returned non-JSON: "
                f"{body[:200].decode('utf-8', 'replace')}") from exc

    def available(self) -> bool:
        """True when a ComfyUI server answers. Never raises."""
        try:
            self._get_json("/system_stats", timeout=_AVAIL_TIMEOUT)
            return True
        except Exception:  # noqa: BLE001 - availability probe, False is data
            return False

    # -- uploads -------------------------------------------------------------
    def _upload_image(self, image: Any, filename: str) -> str:
        """Upload a PIL image (or path) to ComfyUI's input dir.

        Returns the server-side filename to reference in LoadImage nodes.
        """
        png = _pil_to_png_bytes(image)
        boundary = uuid.uuid4().hex
        parts: list[bytes] = []
        def _field(name: str, value: str) -> None:
            parts.append(f'--{boundary}\r\n'.encode())
            parts.append(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            parts.append(f'{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\n'.encode())
        parts.append(
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{filename}"\r\n'.encode())
        parts.append(b'Content-Type: image/png\r\n\r\n')
        parts.append(png)
        parts.append(b'\r\n')
        _field("overwrite", "true")
        _field("type", "input")
        parts.append(f'--{boundary}--\r\n'.encode())
        body = b"".join(parts)
        _status, resp_body, _ctype = self._request(
            "POST", "/upload/image", payload=body,
            content_type=f"multipart/form-data; boundary={boundary}")
        try:
            info = json.loads(resp_body.decode("utf-8"))
            return str(info["name"])
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            raise GenerativeEditError(
                "ComfyUI /upload/image returned an unexpected response: "
                f"{resp_body[:200]!r}") from exc

    # -- prompt lifecycle ----------------------------------------------------
    def _submit(self, workflow: dict[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": workflow, "client_id": self._client_id}).encode()
        _status, body, _ctype = self._request(
            "POST", "/prompt", payload=payload,
            content_type="application/json")
        try:
            info = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise GenerativeEditError(
                f"ComfyUI /prompt returned non-JSON: {body[:200]!r}") from exc
        node_errors = info.get("node_errors") or {}
        if node_errors:
            details = "; ".join(
                f"{nid}: {err.get('message', err)}"
                for nid, err_list in node_errors.items()
                for err in (err_list if isinstance(err_list, list) else [err_list]))
            raise GenerativeEditError(
                f"ComfyUI rejected the workflow: {details}")
        prompt_id = info.get("prompt_id")
        if not prompt_id:
            raise GenerativeEditError(
                f"ComfyUI /prompt gave no prompt_id: {body[:200]!r}")
        return str(prompt_id)

    def _wait(self, prompt_id: str,
              progress_cb: Callable[[float], None] | None = None) -> dict[str, Any]:
        """Poll /history until the prompt completes. Returns the history
        entry. Raises GenerativeEditError with the node's own error text."""
        deadline = time.time() + self.timeout_s
        if progress_cb is not None:
            progress_cb(0.05)
        while True:
            history = self._get_json(f"/history/{prompt_id}")
            entry = history.get(prompt_id) if isinstance(history, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("completed"):
                    if progress_cb is not None:
                        progress_cb(0.95)
                    if status.get("status_str") == "error":
                        raise GenerativeEditError(
                            "ComfyUI workflow failed: "
                            + _format_execution_error(status))
                    return entry
            if time.time() >= deadline:
                raise GenerativeEditError(
                    f"ComfyUI generation timed out after {self.timeout_s:.0f}s "
                    f"(prompt {prompt_id})")
            time.sleep(_POLL_INTERVAL)

    def _download_images(self, entry: dict[str, Any]) -> list[Any]:
        outputs = entry.get("outputs") or {}
        images: list[Any] = []
        for _node_id, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for img_info in node_out.get("images") or []:
                query = urllib.parse.urlencode({
                    "filename": img_info.get("filename", ""),
                    "subfolder": img_info.get("subfolder", ""),
                    "type": img_info.get("type", "output"),
                })
                _status, body, _ctype = self._request("GET", f"/view?{query}")
                images.append(_png_bytes_to_pil(body))
        if not images:
            raise GenerativeEditError(
                "ComfyUI finished but produced no images")
        return images

    def _download_videos(self, entry: dict[str, Any],
                         out_dir: str | Path | None = None) -> list[Path]:
        """Download rendered videos (VHS_VideoCombine puts them under
        the ``gifs`` key). Saves to ``out_dir`` (default: temp dir) and
        returns the local paths. Never fakes: empty output raises."""
        outputs = entry.get("outputs") or {}
        paths: list[Path] = []
        dest = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
        dest.mkdir(parents=True, exist_ok=True)
        for _node_id, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for vid_info in node_out.get("gifs") or []:
                query = urllib.parse.urlencode({
                    "filename": vid_info.get("filename", ""),
                    "subfolder": vid_info.get("subfolder", ""),
                    "type": vid_info.get("type", "output"),
                })
                _status, body, _ctype = self._request(
                    "GET", f"/view?{query}")
                name = str(vid_info.get("filename") or
                           f"devon_video_{uuid.uuid4().hex}.mp4")
                if not name.lower().endswith((".mp4", ".webm", ".mov")):
                    name += ".mp4"
                path = dest / name
                path.write_bytes(body)
                paths.append(path)
        if not paths:
            raise GenerativeEditError(
                "ComfyUI finished but produced no video files")
        return paths

    def _run_workflow(self, workflow: dict[str, Any],
                      progress_cb: Callable[[float], None] | None = None
                      ) -> list[Any]:
        """Submit → poll → download, holding the GPU lock. Never fakes: a
        failed or empty generation raises with the real reason."""
        sem = self._sem
        if not sem.acquire(timeout=self.queue_timeout_s):
            raise GenerativeEditError(
                "timed out waiting for the ComfyUI GPU lock after "
                f"{self.queue_timeout_s:.0f}s — another generation is running")
        try:
            prompt_id = self._submit(workflow)
            _log.info("comfy prompt %s submitted", prompt_id)
            entry = self._wait(prompt_id, progress_cb)
            return self._download_images(entry)
        finally:
            sem.release()

    # -- GenerativeBackend API ----------------------------------------------
    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1,
                 progress_cb: Callable[[float], None] | None = None) -> list[Any]:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-image needs a prompt")
        if n < 1:
            raise GenerativeEditError("n must be >= 1")
        wf = _txt2img_workflow(
            prompt=prompt, negative_prompt=negative_prompt, seed=seed,
            steps=steps, guidance_scale=guidance_scale, width=width,
            height=height, n=n, checkpoint=self.checkpoint)
        _log.info("comfy txt2img n=%d prompt=%.60r", n, prompt)
        return self._run_workflow(wf, progress_cb)

    def img2img(self, image: Any, prompt: str, *,
                strength: float = 0.6,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None,
                progress_cb: Callable[[float], None] | None = None) -> Any:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("img2img needs a prompt")
        name = self._upload_image(image, f"devon_in_{uuid.uuid4().hex}.png")
        wf = _img2img_workflow(
            image_name=name, prompt=prompt, negative_prompt=negative_prompt,
            strength=strength, seed=seed, steps=steps,
            guidance_scale=guidance_scale, checkpoint=self.checkpoint)
        _log.info("comfy img2img strength=%.2f prompt=%.60r", strength, prompt)
        return self._run_workflow(wf, progress_cb)[0]

    def inpaint(self, image: Any, mask: Any, prompt: str, *,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None,
                progress_cb: Callable[[float], None] | None = None) -> Any:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("inpaint needs a prompt")
        pil = _coerce_pil(image)
        mask_img = _as_mask(pil.size, mask)
        img_name = self._upload_image(pil, f"devon_in_{uuid.uuid4().hex}.png")
        mask_name = self._upload_image(
            mask_img, f"devon_mask_{uuid.uuid4().hex}.png")
        wf = _inpaint_workflow(
            image_name=img_name, mask_name=mask_name, prompt=prompt,
            negative_prompt=negative_prompt, seed=seed, steps=steps,
            guidance_scale=guidance_scale, checkpoint=self.checkpoint)
        _log.info("comfy inpaint prompt=%.60r", prompt)
        return self._run_workflow(wf, progress_cb)[0]

    def edit(self, image: Any, instruction: str, *,
             mask: Any | None = None,
             strength: float = 0.75,
             seed: int | None = None,
             negative_prompt: str | None = None,
             steps: int | None = None,
             guidance_scale: float | None = None,
             width: int | None = None,
             height: int | None = None,
             progress_cb: Callable[[float], None] | None = None) -> Any:
        """Instruction edit = img2img at high strength with the instruction
        as the prompt (ComfyUI has no instruction-editing model by default;
        a masked region goes through inpaint instead)."""
        if mask is not None:
            return self.inpaint(image, mask, instruction, seed=seed,
                                negative_prompt=negative_prompt, steps=steps,
                                guidance_scale=guidance_scale,
                                progress_cb=progress_cb)
        return self.img2img(image, instruction, strength=strength, seed=seed,
                            negative_prompt=negative_prompt, steps=steps,
                            guidance_scale=guidance_scale,
                            progress_cb=progress_cb)

    def upscale(self, image: Any, scale: float = 2.0,
                progress_cb: Callable[[float], None] | None = None) -> Any:
        """Model-free upscale on the ComfyUI host (Lanczos). For learned
        super-resolution, point the server at an upscale model workflow."""
        name = self._upload_image(image, f"devon_up_{uuid.uuid4().hex}.png")
        wf = _upscale_workflow(image_name=name, scale=scale)
        _log.info("comfy upscale scale=%.2f", scale)
        return self._run_workflow(wf, progress_cb)[0]

    def generate_video(self, prompt: str, *, duration_s: int = 5,
                       fps: int = 16, seed: int | None = None,
                       negative_prompt: str | None = None,
                       steps: int | None = None,
                       cfg: float | None = None,
                       width: int | None = None, height: int | None = None,
                       model_file: str | None = None,
                       vae_file: str | None = None,
                       out_dir: str | Path | None = None,
                       progress_cb: Callable[[float], None] | None = None
                       ) -> Path:
        """Text-to-video via the wan22_t2v workflow (Wan 2.2).

        Returns the local mp4 path. Holds the GPU lock for the whole
        render. A server that rejects the Wan node types raises with
        the install note instead of a cryptic error.
        """
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-video needs a prompt")
        duration_s = int(duration_s)
        if duration_s < 1:
            raise GenerativeEditError(
                f"duration_s must be >= 1, got {duration_s}")
        num_frames = max(1, duration_s * int(fps))
        wf = _wan22_t2v_workflow(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            steps=steps, seed=seed, cfg=cfg,
            model_file=model_file, vae_file=vae_file)
        _log.info("comfy t2v %ds@%dfps (%d frames) prompt=%.60r",
                  duration_s, fps, num_frames, prompt)
        sem = self._sem
        if not sem.acquire(timeout=self.queue_timeout_s):
            raise GenerativeEditError(
                "timed out waiting for the ComfyUI GPU lock after "
                f"{self.queue_timeout_s:.0f}s — another generation is running")
        try:
            try:
                prompt_id = self._submit(wf)
            except GenerativeEditError as exc:
                if "rejected the workflow" in str(exc):
                    raise GenerativeEditError(
                        f"{exc}\n{_WAN_WORKFLOW_NOTE}") from exc
                raise
            _log.info("comfy video prompt %s submitted", prompt_id)
            entry = self._wait(prompt_id, progress_cb)
            return self._download_videos(entry, out_dir)[0]
        finally:
            sem.release()

    def describe(self) -> str:
        return (f"generative backend 'comfy' "
                f"({self.host}:{self.port}, "
                f"checkpoint={self.checkpoint or 'server default'})")


# ---------------------------------------------------------------------------
# availability / profile gating
# ---------------------------------------------------------------------------

def comfy_available(host: str | None = None,
                    port: int | None = None) -> tuple[bool, str]:
    """(True, "") when ComfyUI is usable on this machine.

    Honest gating: Termux has no GPU host to talk to, so it is a flat no
    (use the HF backend). Otherwise reachability is the real test — GPU
    presence is not directly detectable from the profile, so we probe the
    server instead of guessing. Never raises.
    """
    if get_profile_kind() == "termux":
        return False, ("ComfyUI needs a GPU host; use the HF backend "
                       "(MEDIA_GEN_BACKEND=hf)")
    be = ComfyUIBackend(host=host or _DEFAULT_HOST,
                        port=port if port is not None else _DEFAULT_PORT)
    try:
        if be.available():
            return True, ""
    except Exception:  # noqa: BLE001 - fail closed, reason below
        pass
    return False, (f"ComfyUI not reachable at {be.host}:{be.port} — start it "
                   "there or set COMFYUI_HOST/COMFYUI_PORT")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _format_execution_error(status: dict[str, Any]) -> str:
    msgs = status.get("messages") or []
    parts: list[str] = []
    for msg in msgs:
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            continue
        kind, data = msg[0], msg[1]
        if kind == "execution_error" and isinstance(data, dict):
            parts.append(
                f"node {data.get('node_id')}: "
                f"{data.get('exception_message') or data.get('exception_type')}")
        elif kind == "execution_interrupted" and isinstance(data, dict):
            parts.append(f"interrupted: {data.get('node_id')}")
    detail = "; ".join(parts) or json.dumps(msgs)[:500]
    return detail


def _coerce_pil(image: Any) -> Any:
    from .images import _require_pillow
    Image = _require_pillow()
    if isinstance(image, Image.Image):
        return image
    return load_image(image)


def _pil_to_png_bytes(image: Any) -> bytes:
    pil = _coerce_pil(image)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _png_bytes_to_pil(data: bytes) -> Any:
    from .images import _require_pillow
    Image = _require_pillow()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise GenerativeEditError(
            f"ComfyUI returned undecodable image bytes "
            f"({len(data)} bytes): {exc}") from exc
    return img.convert("RGB")
