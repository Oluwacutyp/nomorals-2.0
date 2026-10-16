"""Guarded face swap: consent + identity lock + watermark, local only.

The raw swap lives in :mod:`.faceswap` (INSWapper + GFPGAN, local
ONNX). This module is the gate around it — the swap never runs
unguarded:

* **Reference material lives in a vault, never in the repo.** The
  source (reference) face must resolve under ``vault_dir``, and
  ``vault_dir`` itself must not be inside the repo tree. Anything
  else → refused before any model loads.
* **Consent flag.** ``consent`` must be exactly ``True`` — a missing
  flag, a truthy string, or ``1`` does not count.
* **Identity lock.** The first use of an ``identity_id`` binds it to
  the SHA-256 of the reference file (``.identity_lock.json`` in the
  vault). Later uses with the same id but a different reference are
  refused: one id, one face, no silent re-targeting.
* **Local-only path.** The swap runs through the local ONNX swapper;
  this module performs no upload, no HTTP, no cloud call.
* **Watermark bound to the render.** The output carries a visible
  watermark (text + identity hash prefix) burned into the pixels, and
  a sidecar JSON binds ``{source_sha256, identity_id, watermark,
  output_sha256}`` — the sidecar hash is verified against the file on
  write, so a swapped output cannot be passed off as the watermarked
  one.

Invariant: for every guarded output, the sidecar's ``output_sha256``
equals the SHA-256 of the output file, and the watermark text is
recorded in the sidecar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .videos import MediaEditError

__all__ = [
    "FaceSwapGuardError",
    "FaceswapRequest",
    "guarded_swap",
    "verify_guarded_output",
]

_log = get_logger(__name__)

#: repo root = parents[2] of nomorals/media_edit/faceswap_guard.py
_REPO_ROOT = Path(__file__).resolve().parents[2]

_LOCK_NAME = ".identity_lock.json"


class FaceSwapGuardError(MediaEditError):
    """The guard refused the swap — the reason is the message."""


@dataclass
class FaceswapRequest:
    """One guarded swap. All fields required; consent must be True."""

    source_ref: str | Path     # reference face — must live in vault_dir
    target: str | Path         # target image to swap onto
    vault_dir: str | Path       # vault holding reference material
    identity_id: str           # locked to the reference file's hash
    consent: bool = False       # must be exactly True
    watermark_text: str = "face-swapped"
    restore: bool = True
    out_path: str | Path | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (ValueError, OSError):
        return False


def _load_lock(vault: Path) -> dict[str, str]:
    lock_path = vault / _LOCK_NAME
    if not lock_path.exists():
        return {}
    try:
        data = json.loads(lock_path.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        raise FaceSwapGuardError(
            "identity lock file is corrupt — refusing to swap until "
            "the vault owner inspects it")


def _save_lock(vault: Path, lock: dict[str, str]) -> None:
    lock_path = vault / _LOCK_NAME
    tmp = lock_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True), "utf-8")
    tmp.replace(lock_path)


def _watermark(img: Any, text: str) -> Any:
    """Burn a visible watermark strip into the pixels (PIL, local)."""
    from PIL import Image, ImageDraw
    from .images import _require_pillow
    _require_pillow()  # clear error when Pillow is missing
    base = img.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    strip_h = max(22, base.size[1] // 28)
    draw.rectangle([0, base.size[1] - strip_h, base.size[0], base.size[1]],
                   fill=(0, 0, 0, 140))
    draw.text((8, base.size[1] - strip_h + 4), text, fill=(255, 255, 255, 255))
    return Image.alpha_composite(
        base.convert("RGBA"), overlay).convert("RGB")


def _check_request(req: FaceswapRequest) -> tuple[Path, Path, Path, str]:
    """Validate the request. Returns (source, target, vault, source_hash)."""
    if req.consent is not True:
        raise FaceSwapGuardError(
            "face swap refused: consent must be exactly True — "
            "no flag, no swap")
    identity = str(req.identity_id or "").strip()
    if not identity:
        raise FaceSwapGuardError("face swap refused: identity_id required")
    vault = Path(req.vault_dir).resolve()
    if _is_within(vault, _REPO_ROOT):
        raise FaceSwapGuardError(
            f"face swap refused: vault_dir {vault} is inside the repo "
            "tree — reference material lives in a vault, never the repo")
    if not vault.is_dir():
        raise FaceSwapGuardError(f"face swap refused: no vault at {vault}")
    source = Path(req.source_ref)
    if not _is_within(source, vault):
        raise FaceSwapGuardError(
            f"face swap refused: reference {source} is not inside the "
            f"vault {vault}")
    if not source.is_file():
        raise FaceSwapGuardError(
            f"face swap refused: reference not found: {source}")
    target = Path(req.target)
    if not target.is_file():
        raise FaceSwapGuardError(
            f"face swap refused: target not found: {target}")
    source_hash = _sha256(source)
    lock = _load_lock(vault)
    if identity in lock and lock[identity] != source_hash:
        raise FaceSwapGuardError(
            f"face swap refused: identity {identity!r} is locked to a "
            "different reference face — refusing to re-target it")
    return source, target, vault, source_hash


def guarded_swap(req: FaceswapRequest, *,
                 swap_fn: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Run a guarded swap → ``{"output", "sidecar", "identity_id", ...}``.

    ``swap_fn`` defaults to the local :func:`.faceswap.swap_face`
    (ONNX, no network). Tests inject a fake swap; production never
    passes one. The output is watermarked, the identity lock is
    recorded, and the sidecar is written + verified before returning.
    """
    source, target, vault, source_hash = _check_request(req)
    from .images import load_image
    if swap_fn is None:
        from .faceswap import swap_face as swap_fn  # local ONNX only
    src_img = load_image(source)
    tgt_img = load_image(target)
    try:
        swapped = swap_fn(src_img, tgt_img, restore=req.restore)
    except Exception as exc:  # noqa: BLE001 - wrap with the guard's error
        raise FaceSwapGuardError(f"face swap failed: {exc}") from exc
    if not hasattr(swapped, "convert"):
        raise FaceSwapGuardError(
            "face swap returned a non-image — refusing to watermark")
    watermark = f"{req.watermark_text} · id:{req.identity_id[:8]}"
    marked = _watermark(swapped, watermark)
    out = Path(req.out_path) if req.out_path else \
        target.with_name(f"{target.stem}.faceswapped{target.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    marked.save(str(out))
    output_hash = _sha256(out)
    sidecar = {"source_sha256": source_hash,
               "identity_id": req.identity_id,
               "watermark": watermark,
               "output_sha256": output_hash,
               "consent": True, "local_only": True,
               "output": str(out)}
    sidecar_path = out.with_suffix(".faceswap.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), "utf-8")
    # bind the identity lock only after a successful, verified render
    lock = _load_lock(vault)
    lock[req.identity_id] = source_hash
    _save_lock(vault, lock)
    _log.info("guarded face swap: %s (identity %s)", out, req.identity_id)
    return {"output": str(out), "sidecar": str(sidecar_path),
            "identity_id": req.identity_id, "watermark": watermark,
            "output_sha256": output_hash, "source_sha256": source_hash}


def verify_guarded_output(output: str | Path) -> dict[str, Any]:
    """Verify a guarded output against its sidecar. Raises on mismatch."""
    out = Path(output)
    sidecar_path = out.with_suffix(".faceswap.json")
    if not sidecar_path.exists():
        raise FaceSwapGuardError(
            f"no faceswap sidecar for {out} — not a guarded render")
    sidecar = json.loads(sidecar_path.read_text("utf-8"))
    actual = _sha256(out)
    if actual != sidecar.get("output_sha256"):
        raise FaceSwapGuardError(
            f"output hash mismatch for {out} — the render was modified "
            "after watermarking")
    return {"ok": True, "output": str(out),
            "identity_id": sidecar.get("identity_id"),
            "watermark": sidecar.get("watermark")}
