"""Text-to-image on the attested forwarded GPU — remote render routing.

A box that cannot render locally (no torch, or no loadable checkpoint)
sends the job to the owner's GPU host instead of failing: the prompt
goes up as JSON, the rendered PNG comes back base64 in the answer.
The server side runs the exact production path — ``Studio.generate``
with remote routing disabled — so there is one implementation of the
render, not two.

Trust model (same standing rule as ``GET /gpu/attest``): the endpoint
is the owner's own Devon on a GPU host, and there is no shared secret
between the two Devons yet, so ``POST /gpu/imggen`` is public like
``/gpu/attest``. The guards are (a) the endpoint is only ever routed
to after it *attests* its VRAM (``remote_available`` — an unattested
endpoint clears no floor), (b) one render at a time on the server
(the GPU is a bounded resource; a second render gets 429), and (c)
the server binds loopback by default — do not expose it to the open
internet without bearer <redacted> in front of it. A token-authenticated
variant is the follow-up if the endpoint is ever internet-facing.

Wire contract (``POST {url}/gpu/imggen``)::

    {"prompt": "...", "checkpoint": "...", "seed": 42, "steps": 30,
     "guidance_scale": 7.5, "width": 512, "height": 512,
     "negative_prompt": "...", "sampler": "ddim"}
    →
    {"ok": true, "image_b64": "<base64 PNG>", "format": "png",
     "bytes": 123456, "model": "<checkpoint name>", "engine": "imggen"}

Only single-image renders route (``batch_size`` stays 1): a batch
would need a list of images on the wire, and no caller routes a
batch today. A render the endpoint cannot run answers 400 with the
reason; a render already in flight answers 429.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from ...core.errors import NoMoralsError
from ...core.gpu import detect_gpu, gpu_up, gpu_up_reason
from ...core.http import HttpClient
from ...core.logging_setup import get_logger
from . import ImgGenError

_log = get_logger(__name__)

__all__ = [
    "REMOTE_FAMILY",
    "REMOTE_TIMEOUT_S",
    "MAX_IMAGE_BYTES",
    "remote_available",
    "remote_generate",
    "serve_render",
    "validate_render_params",
]

#: The GPU family a remote render must clear. "generating" (16 GB) is
#: the shared bar — an endpoint that cannot clear it is not routed to,
#: even though SD1.5 alone would run on less.
REMOTE_FAMILY = "generating"

#: A render plus a cold model load on the GPU host takes tens of
#: seconds; 600 s is the generous bound. Past it the client times out
#: and reports the remote failure honestly.
REMOTE_TIMEOUT_S = 600.0

#: A PNG over this size is not a render — refuse it rather than
#: letting a bad endpoint OOM the client. A 2048x2048 PNG is an order
#: of magnitude smaller than this.
MAX_IMAGE_BYTES = 64 * 1024 * 1024

#: PNG file magic — the client verifies the returned bytes really are
#: a PNG before handing them to the image pipeline.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Sampling knobs the wire carries (the PipelineConfig fields a
#: single-image render takes). Anything else is rejected by
#: :func:`validate_render_params`.
REMOTE_PARAMS = (
    "seed",
    "steps",
    "guidance_scale",
    "width",
    "height",
    "negative_prompt",
    "sampler",
)


def remote_available() -> tuple[bool, str]:
    """Is there an attested forwarded GPU clearing the generating floor?

    Returns ``(True, "")`` when the job may route, else ``(False,
    reason)``. An unattested endpoint, a too-small GPU, and a box with
    no forwarded GPU configured all answer False — the caller falls
    through to the local path (and its honest error) instead.
    """
    try:
        info = detect_gpu()
    except Exception as exc:  # noqa: BLE001 - probe, never raises
        return False, f"GPU detection failed: {exc}"
    if info.path != "forwarded":
        return False, "no forwarded GPU configured (NM_GPU_URL)"
    if not info.attested:
        return False, (
            "forwarded GPU at "
            f"{info.forward_url or 'the configured endpoint'} "
            "was never attested — its VRAM is unproven"
        )
    if not gpu_up(REMOTE_FAMILY, info):
        return False, gpu_up_reason(REMOTE_FAMILY, info)
    return True, ""


def _default_post(url: str, payload: dict[str, Any]):  # type: ignore[no-untyped-def]
    """POST JSON to the endpoint. ``allow_private_ips`` because the GPU
    host is the owner's own box — a LAN address by definition, which
    the SSRF guard would otherwise refuse (same rule as attestation).
    """
    client = HttpClient(allow_private_ips=True, timeout=REMOTE_TIMEOUT_S)
    return client.post_json(url, payload)


def remote_generate(
    prompt: str,
    *,
    checkpoint: str = "",
    seed: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    negative_prompt: str | None = None,
    sampler: str | None = None,
    http_post: Any | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Render ``prompt`` on the attested forwarded GPU.

    Returns ``(png_bytes, info)`` where ``info`` carries ``format``,
    ``bytes``, ``model`` and ``endpoint``. Raises :class:`ImgGenError`
    on any failure — the endpoint unreachable, an error status, bad
    JSON, invalid base64, non-PNG bytes, or an absurd size. Never
    returns a faked image. ``http_post`` is the seam tests inject
    (``(url, payload) -> response`` with ``.ok`` / ``.status`` /
    ``.text`` / ``.json()`` like :class:`HttpResponse`).
    """
    ok, reason = remote_available()
    if not ok:
        raise ImgGenError(f"remote image generation unavailable: {reason}")
    info = detect_gpu()  # cached: the same attested answer as above
    url = (info.forward_url or "").rstrip("/")
    payload: dict[str, Any] = {"prompt": prompt}
    if checkpoint:
        payload["checkpoint"] = checkpoint
    for key, value in (
        ("seed", seed),
        ("steps", steps),
        ("guidance_scale", guidance_scale),
        ("width", width),
        ("height", height),
        ("negative_prompt", negative_prompt),
        ("sampler", sampler),
    ):
        if value is not None:
            payload[key] = value
    post = http_post or _default_post
    try:
        resp = post(f"{url}/gpu/imggen", payload)
    except Exception as exc:  # noqa: BLE001 - every failure is ImgGenError
        raise ImgGenError(f"remote render request failed: {exc}") from exc
    if not resp.ok:
        raise ImgGenError(
            f"remote render failed (HTTP {resp.status}): "
            f"{resp.text[:200]}"
        )
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - invalid JSON is a failure
        raise ImgGenError(
            f"remote render returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or not data.get("ok"):
        detail = ""
        if isinstance(data, dict):
            detail = str(data.get("error") or "")[:200]
        raise ImgGenError(
            f"remote render failed: {detail or 'endpoint answered ok:false'}")
    try:
        raw = base64.b64decode(str(data.get("image_b64") or ""),
                               validate=True)
    except Exception as exc:  # noqa: BLE001 - invalid base64 is a failure
        raise ImgGenError(
            "remote render returned invalid image bytes") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImgGenError(
            f"remote render returned {len(raw)} bytes, over the "
            f"{MAX_IMAGE_BYTES} cap")
    if not raw.startswith(PNG_MAGIC):
        raise ImgGenError("remote render returned non-PNG bytes")
    return raw, {
        "format": str(data.get("format") or "png"),
        "bytes": len(raw),
        "model": str(data.get("model") or ""),
        "endpoint": url,
    }


def validate_render_params(body: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a ``POST /gpu/imggen`` body into the render params.

    Returns ``{"prompt": ..., <known params>}``. Raises
    :class:`NoMoralsError` (→ 400) on a missing/empty prompt, a wrong
    type, an out-of-range value, or an unknown key — the endpoint
    never invents a rule the organ does not have, and it never
    silently drops a param the caller thought it sent.
    """
    if not isinstance(body, Mapping):
        raise NoMoralsError("render body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise NoMoralsError("prompt is required (non-empty string)")
    if len(prompt) > 4096:
        raise NoMoralsError("prompt is too long (max 4096 characters)")
    known = {"prompt", "checkpoint", *REMOTE_PARAMS}
    unknown = sorted(set(body) - known)
    if unknown:
        raise NoMoralsError(
            f"unknown render param(s): {', '.join(unknown)}")
    out: dict[str, Any] = {"prompt": prompt.strip()}

    def _int_param(name: str, lo: int, hi: int) -> None:
        value = body.get(name)
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise NoMoralsError(f"{name} must be an integer")
        if not lo <= value <= hi:
            raise NoMoralsError(f"{name} must be between {lo} and {hi}")
        out[name] = value

    def _float_param(name: str, lo: float, hi: float) -> None:
        value = body.get(name)
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NoMoralsError(f"{name} must be a number")
        if not lo <= float(value) <= hi:
            raise NoMoralsError(f"{name} must be between {lo:g} and {hi:g}")
        out[name] = value

    def _str_param(name: str, max_len: int) -> None:
        value = body.get(name)
        if value is None:
            return
        if not isinstance(value, str):
            raise NoMoralsError(f"{name} must be a string")
        if len(value) > max_len:
            raise NoMoralsError(f"{name} is too long (max {max_len} characters)")
        out[name] = value

    _int_param("seed", -(2**63), 2**63 - 1)
    _int_param("steps", 1, 150)
    _int_param("width", 64, 2048)
    _int_param("height", 64, 2048)
    _float_param("guidance_scale", 0.0, 30.0)
    _str_param("negative_prompt", 4096)
    _str_param("checkpoint", 256)
    sampler = body.get("sampler")
    if sampler is not None:
        if not isinstance(sampler, str) or sampler not in ("ddim", "ddpm"):
            raise NoMoralsError("sampler must be 'ddim' or 'ddpm'")
        out["sampler"] = sampler
    return out


def serve_render(prompt: str, **params: Any) -> dict[str, Any]:
    """Endpoint side of ``POST /gpu/imggen``: render locally and answer.

    Runs the exact production path — ``Studio.generate`` with remote
    routing disabled, so a forwarded request is never forwarded on
    (no A→B→A loop). Returns the wire dict (base64 PNG + metadata).
    Raises :class:`ImgGenError` when the render cannot run here; the
    API handler translates it to a 400 with the message.
    """
    from .studio import Studio, StudioConfig

    checkpoint = str(params.pop("checkpoint", "") or "")
    studio = (Studio(StudioConfig(checkpoint=checkpoint))
              if checkpoint else Studio())
    paths = studio.generate(prompt, allow_remote=False, **params)
    if not paths:
        raise ImgGenError("remote render produced no image")
    with open(paths[0], "rb") as handle:
        raw = handle.read()
    return {
        "ok": True,
        "image_b64": base64.b64encode(raw).decode("ascii"),
        "format": "png",
        "bytes": len(raw),
        "model": checkpoint,
        "engine": "imggen",
    }
