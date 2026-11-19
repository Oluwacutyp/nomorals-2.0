"""ImageDocument — the non-destructive image document (Echelon
Wave 2 · plan §2.1a/§2.0-D2).

PhotoCraft's document model in Devon's idiom: an image edit is
**data** — a source reference plus an ordered list of ops, each op
a stable id + JSON-able params (the D2 op-spec discipline). The
document never mutates its source; :meth:`ImageDocument.render`
replays the op list from the source every time, so:

- undo = pop the op list, redo = re-push (the EditStudio stack
  mechanics, mirrored exactly);
- branch = copy the document;
- the same document renders byte-identical output on every replay
  (all deterministic rungs; neural rungs are seed-pinned through
  their PipelineConfig params);
- the document compiles to the canonical
  :class:`nomorals.media.edit_engine.Document` (``to_document``)
  and rebuilds from one (``from_document``) — the same
  façade↔document agreement Wave 1 pinned for EditStudio;
- the shared PIL vocabulary also bridges to
  :class:`nomorals.media_edit.EditStudio` sessions
  (``to_edit_studio`` / ``from_edit_studio``); ops a bridge cannot
  express raise — never a silent drop.

One vocabulary, two families (plan §2.1a: "the media_edit/images.py
ops and imggen neural ops as one vocabulary"):

- the PIL ops of :mod:`nomorals.media_edit.images`, executed by
  that module's own ``apply_chain`` (one canonical implementation);
- the imggen ops: ``develop`` (the §2.1b develop stage —
  LightCraft controls as an op, masks included), ``img2img``
  (SDEdit), ``inpaint`` / ``outpaint`` (neural or CPU tiers, the
  tier chosen in the open and labelled), ``upscale``.

Per-rung honesty (the GenShot rule): every op's render result
records **which rung produced it** in the render provenance —
``neural-blended-diffusion``, ``cv2-telea …``, ``numpy-voronoi-
diffuse (classical, non-neural quality)`` — and an op with no
available rung **raises**; an ImageDocument render never passes an
image through unstyled and calls it edited.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ImgGenError, TORCH_AVAILABLE
from .masks import MaskRef

__all__ = [
    "ImageDocument",
    "ImageOp",
    "RenderedImage",
    "IMAGE_OPS",
    "CLASSIC_OPS",
    "op_spec",
]

_DOC_KIND = "devon-image-document"
_DOC_VERSION = 1

#: The media_edit/images.py PIL vocabulary, as ImageDocument ops.
CLASSIC_OPS = (
    "resize", "crop", "rotate", "flip", "blur", "border",
    "enhance", "thumbnail", "annotate_text", "annotate_shape",
    "duotone", "round_corners", "gradient_overlay", "meme",
    "stack", "grid", "watermark",
)

#: imggen-native ops (neural + develop + upscale).
IMGGEN_OPS = ("develop", "img2img", "inpaint", "outpaint", "upscale",
              "remove_background")


# ---------------------------------------------------------------------------
# Op values (D2: operations are data with a stable id)
# ---------------------------------------------------------------------------

@dataclass
class ImageOp:
    """One edit, as data: stable id + JSON-able params + label."""

    op: str
    params: dict = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict:
        return {"op": self.op, "params": self.params,
                "label": self.label or self.op}

    @classmethod
    def from_dict(cls, d: dict) -> "ImageOp":
        if not isinstance(d, dict) or "op" not in d:
            raise ImgGenError(f"bad image op entry {d!r}: needs an 'op' key")
        return cls(op=str(d["op"]), params=dict(d.get("params") or {}),
                   label=str(d.get("label") or ""))


def _jsonable(value: Any) -> Any:
    """Coerce params to pure JSON values (EditStudio's discipline,
    extended: MaskRef → its dict, Path → str)."""
    if isinstance(value, MaskRef):
        return value.to_dict()
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _check_jsonable(op_id: str, params: dict) -> dict:
    clean = _jsonable(params)
    try:
        json.dumps(clean)
    except (TypeError, ValueError) as exc:
        raise ImgGenError(
            f"op {op_id!r} params must be JSON-able data "
            f"(documents are data): {exc}") from exc
    return clean


# ---------------------------------------------------------------------------
# Render context + result
# ---------------------------------------------------------------------------

@dataclass
class _RenderContext:
    pipeline: Any = None            # imggen pipeline for neural ops
    box_predictor: Any = None       # (image, text) -> box, for masks


@dataclass
class RenderedImage:
    """A rendered document: the pixels plus per-op provenance.

    ``provenance`` is a list of ``{"op", "rung", "note"}`` dicts in
    replay order — the record of which rung produced each stage.
    ``notes`` collects document-level disclosures (e.g. a neural
    pipeline that could not be built, with CPU tiers used instead).
    """

    image: Any
    provenance: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def rungs(self) -> list:
        return [p["rung"] for p in self.provenance]


# ---------------------------------------------------------------------------
# Op executors — each returns (image, rung_label, note)
# ---------------------------------------------------------------------------

def _run_classic(img, op_id: str, params: dict, ctx: _RenderContext):
    from ...media_edit.images import apply_chain, op_watermark, load_image

    if op_id == "watermark":
        p = dict(params)
        logo_ref = p.pop("logo", None)
        if logo_ref is None:
            raise ImgGenError("watermark op needs a 'logo' image path")
        logo = load_image(logo_ref)
        try:
            out = op_watermark(img, logo, **p)
        finally:
            try:
                logo.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        return out, "pil (media_edit.images)", ""
    out = apply_chain(img, [{"op": op_id, **params}])
    return out, "pil (media_edit.images)", ""


def _mask_from_params(params: dict, op_id: str) -> MaskRef:
    m = params.get("mask")
    if m is None:
        raise ImgGenError(
            f"op {op_id!r} needs a mask — pass a MaskRef (or its "
            "dict) as the 'mask' param")
    if isinstance(m, MaskRef):
        return m
    return MaskRef.from_dict(m)


def _run_develop(img, params: dict, ctx: _RenderContext):
    from .develop import DEVELOP_FIELDS, develop_image, develop_settings_from_params, render_develop

    p = dict(params)
    mask_ref = None
    if p.get("mask") is not None:
        mask_ref = _mask_from_params(p, "develop")
        p.pop("mask")
    if any("." in k or k == "settings" for k in p):
        # The full LightCraft engine: dotted controls
        # (light.exposure, wb.temp, curve.master, ...) or a whole
        # settings dict — merged over the legacy sliders.
        settings = develop_settings_from_params(p)
        developed = render_develop(img, settings)
    else:
        unknown = sorted(set(p) - set(DEVELOP_FIELDS))
        if unknown:
            raise ImgGenError(
                f"develop op got unknown controls {unknown}; "
                f"use {list(DEVELOP_FIELDS)}")
        developed = develop_image(img, **p)
    if mask_ref is None:
        return developed, "develop (numpy, deterministic)", ""
    res = mask_ref.resolve(img, box_predictor=ctx.box_predictor)
    import numpy as np

    a = (np.asarray(res.mask.convert("L")).astype(np.float64)
         / 255.0)[..., None]
    base = np.asarray(img.convert("RGB")).astype(np.float64)
    over = np.asarray(developed.convert("RGB")).astype(np.float64)
    out = over * a + base * (1.0 - a)
    from PIL import Image

    blended = Image.fromarray(
        np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")
    return (blended, "develop (numpy, deterministic)",
            f"regional: mask rung {res.tier} ({res.detail})")


def _run_img2img(img, params: dict, ctx: _RenderContext):
    if ctx.pipeline is None:
        raise ImgGenError(
            "op 'img2img' needs a neural pipeline (Studio.pipeline()); "
            "SDEdit has no CPU rung — refusing to return the image "
            "unstyled and call it edited")
    from .edit import img2img

    p = dict(params)
    prompt = str(p.pop("prompt", ""))
    strength = p.pop("strength", 0.6)
    out = img2img(ctx.pipeline, img, prompt, strength=strength, **p)
    return out[0], "neural img2img (SDEdit, imggen pipeline)", ""


_INPAINT_GEN_KEYS = ("seed", "steps", "sampler", "guidance_scale",
                     "negative_prompt", "width", "height", "eta")


def _run_inpaint(img, params: dict, ctx: _RenderContext):
    from .edit import inpaint, inpaint_auto

    p = dict(params)
    mask_ref = _mask_from_params(p, "inpaint")
    p.pop("mask", None)
    prompt = str(p.pop("prompt", ""))
    feather = int(p.pop("feather", 8))
    method = str(p.pop("method", "auto"))
    preserve = bool(p.pop("preserve_outside", False))
    gen = {k: p.pop(k) for k in list(p) if k in _INPAINT_GEN_KEYS}
    if p:
        raise ImgGenError(
            f"inpaint op got unknown params {sorted(p)}")
    res = mask_ref.resolve(img, box_predictor=ctx.box_predictor)
    note = f"mask rung {res.tier} ({res.detail})"
    if method == "neural":
        if ctx.pipeline is None:
            raise ImgGenError(
                "inpaint method='neural' needs a pipeline "
                "(Studio.pipeline()); use method='auto' for the "
                "CPU tiers")
        out = inpaint(ctx.pipeline, img, res.mask, prompt,
                      feather=feather, **gen)[0]
        label = "neural-blended-diffusion"
    else:
        if method not in ("auto", "telea", "ns", "numpy"):
            raise ImgGenError(
                f"unknown inpaint method {method!r}; "
                "use auto|neural|telea|ns|numpy")
        out, label = inpaint_auto(
            img, res.mask, prompt,
            pipeline=ctx.pipeline if method == "auto" else None,
            method=method, feather=feather, **gen)
    if preserve:
        # Byte-identity outside the mask, enforced in the op itself:
        # the numpy rung resizes internally and the neural rung
        # re-noises the known region, so neither guarantees untouched
        # pixels by itself. Hard threshold, not the soft alpha: the
        # rung already feather-blends its own output, and applying
        # the soft alpha again would square it (a^2) and pull the
        # original back into the feather band. Inside the hard mask
        # the rung's output stands as delivered; outside it the
        # source pixel stands, exactly (same blend shape as
        # _run_develop's regional develop).
        import numpy as np
        from PIL import Image as _Image

        a = (np.asarray(res.mask.convert("L")) > 127
             ).astype(np.float64)[..., None]
        base = np.asarray(img.convert("RGB")).astype(np.float64)
        over = np.asarray(out.convert("RGB")).astype(np.float64)
        out = _Image.fromarray(
            np.clip(over * a + base * (1.0 - a) + 0.5, 0, 255
                    ).astype(np.uint8), "RGB")
        note += "; outside-mask pixels byte-identical to the source"
    return out, label, note


def _run_outpaint(img, params: dict, ctx: _RenderContext):
    from .edit import outpaint, outpaint_cpu

    p = dict(params)
    prompt = str(p.pop("prompt", ""))
    tier = str(p.pop("tier", "auto"))
    feather = int(p.pop("feather", 12))
    ext = {}
    for side in ("left", "right", "top", "bottom"):
        ext[side] = int(p.pop(side, 0))
    gen = {k: p.pop(k) for k in list(p) if k in _INPAINT_GEN_KEYS}
    if p:
        raise ImgGenError(
            f"outpaint op got unknown params {sorted(p)}")
    if tier not in ("auto", "neural", "cpu"):
        raise ImgGenError(
            f"unknown outpaint tier {tier!r}; use auto|neural|cpu")
    if tier == "neural" and ctx.pipeline is None:
        raise ImgGenError(
            "outpaint tier='neural' needs a pipeline "
            "(Studio.pipeline()); use tier='auto' for the CPU tier")
    if tier == "cpu" or (tier == "auto" and not (
            ctx.pipeline is not None and TORCH_AVAILABLE)):
        out, label = outpaint_cpu(img, prompt, **ext)
        return out, label, ""
    out = outpaint(ctx.pipeline, img, prompt, feather=feather,
                   **ext, **gen)
    return out[0], "neural outpaint (blended diffusion)", ""


def _run_remove_background(img, params: dict, ctx: _RenderContext):
    from .bgremove import remove_background

    p = dict(params)
    tier = str(p.pop("tier", "auto"))
    model = str(p.pop("model", ""))
    margin = float(p.pop("margin", 0.06))
    feather = int(p.pop("feather", 2))
    if p:
        raise ImgGenError(
            f"remove_background op got unknown params {sorted(p)}")
    res = remove_background(img, tier=tier, model=model,
                            margin=margin, feather=feather)
    note = res.detail
    if res.notes:
        note += "; " + "; ".join(res.notes)
    return res.image, f"bgremove ({res.rung})", note


def _run_optimize(img, params: dict, ctx: _RenderContext):
    # An encoding decision, not a pixel transform: the op records
    # which codec/quality the ladder chose (and its measured size
    # and PSNR); render_to / Studio.render_document apply it at
    # save time. Replaying the document without saving re-records
    # the decision in the provenance — the pixels pass through
    # untouched.
    from .optimize import CODECS, TARGETS

    p = dict(params)
    codec = str(p.pop("codec", "webp"))
    quality = p.pop("quality", 80)
    target = str(p.pop("target", ""))
    max_bytes = int(p.pop("max_bytes", 0))
    size_bytes = int(p.pop("size_bytes", 0))
    psnr = p.pop("psnr_db", None)
    trials = int(p.pop("trials", 0))
    if p:
        raise ImgGenError(
            f"optimize op got unknown params {sorted(p)}")
    if codec not in CODECS:
        raise ImgGenError(
            f"optimize op got unknown codec {codec!r}; "
            f"want one of {list(CODECS)}")
    if quality is not None and not (1 <= int(quality) <= 100):
        raise ImgGenError(
            f"optimize op quality {quality!r} out of range 1..100")
    if target and target not in TARGETS:
        raise ImgGenError(
            f"optimize op got unknown target {target!r}; "
            f"want one of {sorted(TARGETS)}")
    q = "lossless" if quality is None else f"q{quality}"
    detail = f"{codec} {q}"
    if size_bytes:
        detail += f" · {size_bytes:,} bytes"
    if max_bytes:
        detail += f" · budget {max_bytes:,} bytes"
    if psnr is None and size_bytes:
        detail += " · identical pixels"
    elif psnr is not None:
        detail += f" · psnr {psnr:.1f} dB"
    if target:
        detail += f" · target {target}"
    if trials:
        detail += f" ({trials} trials)"
    return img, "optimize (encoding decision)", detail


def _run_upscale(img, params: dict, ctx: _RenderContext):
    from .upscale import upscale_classical, upscale_diffusion

    p = dict(params)
    scale = float(p.pop("scale", 2.0))
    diffusion = bool(p.pop("diffusion", False))
    prompt = str(p.pop("prompt", ""))
    creativity = p.pop("creativity", None)
    if p:
        raise ImgGenError(
            f"upscale op got unknown params {sorted(p)}")
    if diffusion:
        if ctx.pipeline is None:
            raise ImgGenError(
                "upscale diffusion=True needs a pipeline "
                "(Studio.pipeline()); use diffusion=False for the "
                "classical rung")
        out = upscale_diffusion(ctx.pipeline, img, prompt,
                                scale=scale, creativity=creativity)
        return out, "diffusion upscale (imggen.upscale)", ""
    if creativity is not None:
        raise ImgGenError(
            "creativity needs diffusion=True (classical "
            "upscale is always structure-preserving)")
    out = upscale_classical(img, scale=scale)
    return out, "classical lanczos+unsharp (non-neural)", ""


@dataclass(frozen=True)
class _OpSpec:
    summary: str
    run: Any  # (img, params, ctx) -> (img, rung, note)


def _classic_spec(op_id: str) -> _OpSpec:
    return _OpSpec(
        summary=f"media_edit.images {op_id} (PIL, deterministic)",
        run=lambda img, params, ctx, _id=op_id:
            _run_classic(img, _id, params, ctx))


#: The one image-op vocabulary (D2): stable id → spec.
IMAGE_OPS: dict = {
    **{name: _classic_spec(name) for name in CLASSIC_OPS},
    "develop": _OpSpec(
        "LightCraft develop stage (numpy, deterministic; "
        "optional MaskRef for regional develop)",
        _run_develop),
    "img2img": _OpSpec(
        "SDEdit restyle via the imggen pipeline (neural only)",
        _run_img2img),
    "inpaint": _OpSpec(
        "masked repaint: neural blended diffusion or CPU tiers, "
        "mask carried as a MaskRef",
        _run_inpaint),
    "outpaint": _OpSpec(
        "canvas extension: neural inpaint of the new region or "
        "the CPU edge-extend tier",
        _run_outpaint),
    "upscale": _OpSpec(
        "classical Lanczos upscale or diffusion detail synthesis",
        _run_upscale),
    "remove_background": _OpSpec(
        "one-shot background removal: the rembg model rung "
        "where provisioned, then the classical mask ladder; "
        "the cutout is a transparent PNG and the rung is "
        "recorded in the provenance",
        _run_remove_background),
    "optimize": _OpSpec(
        "image optimisation: the codec-ladder decision (codec + "
        "quality + target + measured size/PSNR) recorded as data; "
        "the pixels pass through untouched and render_to / "
        "Studio.render_document apply the encoding at save time",
        _run_optimize),
}


def op_spec(op_id: str) -> _OpSpec:
    """The spec for an op id — raises naming the vocabulary."""
    try:
        return IMAGE_OPS[op_id]
    except KeyError:
        raise ImgGenError(
            f"unknown image op {op_id!r}; "
            f"vocabulary: {sorted(IMAGE_OPS)}") from None


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

class ImageDocument:
    """A non-destructive image edit: source ref + ordered op list.

    Chainable session façade in the EditStudio idiom — every
    builder returns ``self``; :meth:`render` replays from the
    source, which is never modified.
    """

    def __init__(self, source: str | os.PathLike[str] = "", *,
                 title: str = "",
                 source_prompt: str = "",
                 source_rung: str = "",
                 notes: list | None = None) -> None:
        self.source = str(source) if source else ""
        self.title = title
        #: How the source itself was made, when it was generated
        #: (prompt + producing rung) — provenance, never implied.
        self.source_prompt = source_prompt
        self.source_rung = source_rung
        self.notes: list = list(notes or [])
        self.ops: list = []
        self._undone: list = []

    # -- stack mechanics (EditStudio mechanics, mirrored) -------------
    def push(self, op_id: str, label: str = "", **params: Any
             ) -> "ImageDocument":
        """Append an op by id. Params must be JSON-able data; a
        :class:`MaskRef` param is stored as its dict."""
        op_spec(op_id)  # validates the id against the vocabulary
        clean = _check_jsonable(op_id, params)
        self.ops.append(ImageOp(op=op_id, params=clean,
                                label=label or op_id))
        self._undone.clear()
        return self

    def undo(self) -> ImageOp | None:
        if not self.ops:
            return None
        op = self.ops.pop()
        self._undone.append(op)
        return op

    def redo(self) -> ImageOp | None:
        if not self._undone:
            return None
        op = self._undone.pop()
        self.ops.append(op)
        return op

    def clear(self) -> "ImageDocument":
        self.ops.clear()
        self._undone.clear()
        return self

    def branch(self) -> "ImageDocument":
        """An independent copy — edit either without the other."""
        other = ImageDocument(
            self.source, title=self.title,
            source_prompt=self.source_prompt,
            source_rung=self.source_rung, notes=list(self.notes))
        other.ops = copy.deepcopy(self.ops)
        other._undone = copy.deepcopy(self._undone)
        return other

    def describe(self) -> str:
        lines = [f"image document '{self.title or 'untitled'}' "
                 f"({len(self.ops)} ops)"]
        if self.source:
            lines.append(f"  source: {self.source}")
        if self.source_rung:
            lines.append(f"  source rung: {self.source_rung}")
        for i, op in enumerate(self.ops, 1):
            params = ", ".join(
                f"{k}={v}" for k, v in op.params.items())
            lines.append(f"  {i}. {op.op}({params})")
        if self._undone:
            lines.append(f"  ({len(self._undone)} undone)")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    # -- typed builders (one per op family) -----------------------------
    def resize(self, width: int, height: int, **kw) -> "ImageDocument":
        return self.push("resize", width=width, height=height, **kw)

    def crop(self, **kw) -> "ImageDocument":
        return self.push("crop", **kw)

    def rotate(self, angle: float, **kw) -> "ImageDocument":
        return self.push("rotate", angle=angle, **kw)

    def flip(self, direction: str = "horizontal") -> "ImageDocument":
        return self.push("flip", direction=direction)

    def blur(self, radius: float = 2.0) -> "ImageDocument":
        return self.push("blur", radius=radius)

    def enhance(self, **kw) -> "ImageDocument":
        return self.push("enhance", **kw)

    def develop(self, mask: MaskRef | dict | None = None, **controls
                ) -> "ImageDocument":
        """The develop stage (§2.1b): LightCraft controls as an op.
        ``mask`` (a MaskRef) makes it a *regional* develop."""
        params = dict(controls)
        if mask is not None:
            params["mask"] = mask
        return self.push("develop", **params)

    def img2img(self, prompt: str, strength: float | str = 0.6,
                **gen) -> "ImageDocument":
        return self.push("img2img", prompt=prompt,
                         strength=strength, **gen)

    def inpaint(self, prompt: str = "", *,
                mask: MaskRef | dict | None = None,
                method: str = "auto", feather: int = 8,
                **gen) -> "ImageDocument":
        """Masked repaint. ``mask`` is a MaskRef (or its dict);
        ``method``: auto|neural|telea|ns|numpy."""
        params: dict = {"prompt": prompt, "method": method,
                        "feather": feather, **gen}
        if mask is not None:
            params["mask"] = mask
        return self.push("inpaint", **params)

    def remove_object(self, description: str = "", *,
                      mask: MaskRef | dict | str | None = None,
                      prompt: str = "", method: str = "auto",
                      feather: int = 8, box=None,
                      point=None) -> ImageDocument:
        """A remove-object edit: an inpaint op whose mask is a MaskRef
        recipe — ``description`` + located ``box``/``point`` (the
        Cleanup.pictures loop), or an explicit ``mask`` (file path,
        MaskRef, or its dict). Same op id as :meth:`inpaint`, so it
        undoes and replays like every image-document op; ``method``
        is validated here so a bad method fails at build, not render.
        The op carries ``preserve_outside``: the executor composites
        the source pixels back outside the mask, so the delivered
        image is byte-identical to the source everywhere except the
        removed region.
        """
        if method not in ("auto", "neural", "telea", "ns", "numpy"):
            raise ImgGenError(
                f"unknown inpaint method {method!r}; "
                "use auto|neural|telea|ns|numpy")
        if mask is not None:
            if isinstance(mask, MaskRef):
                ref = mask
            elif isinstance(mask, dict):
                ref = MaskRef.from_dict(mask)
            else:
                # A file mask is authoritative — no feather.
                ref = MaskRef.file(str(mask))
        elif description:
            ref = MaskRef.auto(text=description, box=box, point=point,
                               feather=feather)
        else:
            raise ImgGenError(
                "remove_object needs a description or a mask")
        return self.push("inpaint", prompt=prompt, mask=ref,
                         method=method, feather=feather,
                         preserve_outside=True)

    def outpaint(self, prompt: str = "", *, tier: str = "auto",
                 feather: int = 12, **ext) -> "ImageDocument":
        """Canvas extension: extents as left/right/top/bottom px."""
        return self.push("outpaint", prompt=prompt, tier=tier,
                         feather=feather, **ext)

    def upscale(self, scale: float = 2.0, *, diffusion: bool = False,
                prompt: str = "",
                creativity: float | None = None) -> "ImageDocument":
        return self.push("upscale", scale=scale, diffusion=diffusion,
                         prompt=prompt, creativity=creativity)

    def remove_background(self, *, tier: str = "auto", model: str = "",
                          margin: float = 0.06,
                          feather: int = 2) -> ImageDocument:
        """One-shot background removal as an op: the rembg model
        rung where provisioned, then the classical mask ladder.
        ``tier``: auto|rembg|sam|grabcut|flood|box. The cutout is
        a transparent PNG; the rung that served it is recorded
        in the render provenance."""
        return self.push("remove_background", tier=tier, model=model,
                         margin=margin, feather=feather)

    def optimize(self, *, codec: str = "webp",
                 quality: int | None = 80, target: str = "",
                 max_bytes: int = 0, size_bytes: int = 0,
                 psnr_db: float | None = None,
                 trials: int = 0) -> ImageDocument:
        """An optimisation decision as an op (Squoosh): the codec
        ladder's winning encoding, recorded as data. ``codec`` is
        one of jpeg|webp|png|avif; ``quality`` None means lossless;
        ``size_bytes``/``psnr_db``/``trials`` are the ladder's
        measured values for the winning trial. Validated here so a
        bad codec or quality fails at build, not render. The op
        changes no pixels — render_to / Studio.render_document
        apply the encoding when the document is saved."""
        from .optimize import CODECS, TARGETS

        if codec not in CODECS:
            raise ImgGenError(
                f"unknown optimize codec {codec!r}; "
                f"want one of {list(CODECS)}")
        if quality is not None and not (1 <= int(quality) <= 100):
            raise ImgGenError(
                f"optimize quality {quality!r} out of range 1..100")
        if target and target not in TARGETS:
            raise ImgGenError(
                f"unknown optimize target {target!r}; "
                f"want one of {sorted(TARGETS)}")
        return self.push("optimize", codec=codec, quality=quality,
                         target=target, max_bytes=max_bytes,
                         size_bytes=size_bytes, psnr_db=psnr_db,
                         trials=trials)

    # -- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "kind": _DOC_KIND,
            "version": _DOC_VERSION,
            "title": self.title,
            "source": self.source,
            "source_prompt": self.source_prompt,
            "source_rung": self.source_rung,
            "notes": list(self.notes),
            "ops": [o.to_dict() for o in self.ops],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageDocument":
        if not isinstance(d, dict) or d.get("kind") != _DOC_KIND:
            raise ImgGenError(
                f"not an image document (kind={d.get('kind')!r}); "
                f"expected {_DOC_KIND!r}")
        if d.get("version") != _DOC_VERSION:
            raise ImgGenError(
                f"image document version {d.get('version')!r} "
                f"unsupported; this build reads {_DOC_VERSION}")
        doc = cls(d.get("source", ""), title=d.get("title", ""),
                  source_prompt=d.get("source_prompt", ""),
                  source_rung=d.get("source_rung", ""),
                  notes=list(d.get("notes") or []))
        for entry in d.get("ops") or []:
            op = ImageOp.from_dict(entry)
            op_spec(op.op)  # vocabulary check on load, too
            doc.ops.append(op)
        return doc

    def content_hash(self) -> str:
        """Content address of the document exactly as serialized —
        same document, same hash, on every machine."""
        blob = json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def save(self, path: str | os.PathLike[str]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1),
                       encoding="utf-8")
        os.replace(tmp, p)
        return str(p)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ImageDocument":
        p = Path(path)
        if not p.exists():
            raise ImgGenError(f"no such image document: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ImgGenError(
                f"image document {p} is not valid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)

    # -- canonical Document bridge (D1) ---------------------------------
    def to_document(self) -> Any:
        """Compile into the canonical edit_engine ``Document`` —
        one image clip, every op an Effect verbatim, sealed."""
        from ...media.edit_engine.document import image_document
        from ...media.edit_engine.timeline import Effect

        if not self.source:
            raise ImgGenError(
                "image document needs a source file to compile "
                "to a Document")
        doc = image_document(self.source,
                             title=self.title or "untitled")
        doc.tracks[0].clips[0].effects = tuple(
            Effect(name=o.op, params=copy.deepcopy(o.params))
            for o in self.ops)
        return doc.seal()

    @classmethod
    def from_document(cls, doc: Any) -> "ImageDocument":
        """Rebuild from a canonical Document. Effects outside the
        image vocabulary raise — never a silent drop."""
        from ...media.edit_engine.document import Document

        if not isinstance(doc, Document):
            raise ImgGenError(
                f"from_document takes a Document, "
                f"got {type(doc).__name__}")
        image = doc.track("image")
        if image is None or len(image.clips) != 1:
            raise ImgGenError(
                "an image document holds exactly one image clip; "
                f"this document has "
                f"{0 if image is None else len(image.clips)}")
        clip = image.clips[0]
        if clip.at or clip.gain != 1.0 or clip.start \
                or clip.end is not None:
            raise ImgGenError(
                "clip placement/gain/source-window have no "
                "ImageDocument equivalent")
        out = cls(clip.path, title=doc.title or "")
        for e in clip.effects:
            if e.name not in IMAGE_OPS:
                raise ImgGenError(
                    f"effect {e.name!r} is not in the image op "
                    f"vocabulary {sorted(IMAGE_OPS)} — refusing "
                    "to drop it silently")
            out.ops.append(ImageOp(op=e.name,
                                   params=copy.deepcopy(dict(e.params)),
                                   label=e.name))
        return out

    # -- EditStudio bridge (shared PIL vocabulary) ------------------------
    def to_edit_studio(self) -> Any:
        """Open this document as an EditStudio image session.

        Only the shared PIL vocabulary crosses; a document using
        imggen ops (develop/img2img/inpaint/outpaint/upscale)
        raises, naming them — the session engine cannot express
        them and dropping them would falsify the edit.
        """
        from ...media_edit import MediaEditError
        from ...media_edit.studio import EditStudio

        foreign = [o.op for o in self.ops if o.op not in CLASSIC_OPS]
        if foreign:
            raise MediaEditError(
                f"ops {sorted(set(foreign))} are imggen ops with no "
                "EditStudio session equivalent; render the "
                "ImageDocument itself for those")
        st = EditStudio(self.source, kind="image",
                        name=self.title or "untitled")
        for o in self.ops:
            st.op(o.op, **copy.deepcopy(o.params))
        return st

    @classmethod
    def from_edit_studio(cls, st: Any) -> "ImageDocument":
        """Import an EditStudio image session's ops. Ops outside
        the image vocabulary raise — never a silent drop."""
        if st._resolve_kind() != "image":
            raise ImgGenError(
                "only image EditStudio sessions import into an "
                "ImageDocument")
        out = cls(st.source or "", title=st.name)
        for o in st.ops:
            if o["op"] not in IMAGE_OPS:
                raise ImgGenError(
                    f"EditStudio op {o['op']!r} is not in the image "
                    f"op vocabulary {sorted(IMAGE_OPS)} — refusing "
                    "to drop it silently")
            out.ops.append(ImageOp(
                op=o["op"],
                params=copy.deepcopy(dict(o["params"])),
                label=o.get("label") or o["op"]))
        return out

    # -- render -----------------------------------------------------------
    def _load_source(self):
        if not self.source:
            raise ImgGenError(
                "image document has no source — nothing to replay from")
        src = Path(self.source)
        if not src.exists():
            raise ImgGenError(f"no such source image: {self.source}")
        from ...media_edit.images import load_image

        return load_image(src)

    def render(self, pipeline: Any = None, *,
               box_predictor: Any = None) -> RenderedImage:
        """Replay the op list from the source. The source file is
        opened read-only and never written. ``pipeline`` is the
        imggen pipeline neural ops need; without it, neural ops
        raise and tiered ops (inpaint/outpaint ``auto``) use their
        CPU rungs — labelled as such in the provenance.
        """
        ctx = _RenderContext(pipeline=pipeline,
                             box_predictor=box_predictor)
        first = self._load_source()
        img = first
        provenance: list = []
        try:
            for op in self.ops:
                spec = op_spec(op.op)
                img, rung, note = spec.run(
                    img, copy.deepcopy(op.params), ctx)
                provenance.append({"op": op.op, "rung": rung,
                                   "note": note})
        except Exception:
            if img is not first:
                try:
                    first.close()
                except Exception:  # noqa: BLE001 - best effort
                    pass
            raise
        if img is not first:
            try:
                first.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        return RenderedImage(image=img, provenance=provenance,
                             notes=list(self.notes))

    def render_to(self, path: str | os.PathLike[str] | None = None,
                  pipeline: Any = None, *,
                  box_predictor: Any = None,
                  fmt: str | None = None,
                  quality: int | None = None) -> dict:
        """Render and save. The output path can never equal the
        source (the EditStudio guarantee, kept). Returns the
        EditStudio-shaped render record plus the rung provenance.

        An ``optimize`` op in the document decides the encoding
        when the caller does not: its codec becomes the output
        format (and the default file extension) and its quality
        the save quality — a lossless op saves lossless. An
        explicit ``fmt``/``quality`` still wins over the op. An
        explicit output path keeps its stem but takes the
        encoding's extension, so the file never claims a format
        it is not.
        """
        if not self.source:
            raise ImgGenError(
                "image document has no source — nothing to replay from")
        from ...media_edit.images import save_image

        opt = next((o.params for o in reversed(self.ops)
                    if o.op == "optimize"), None)
        if fmt is None and opt is not None:
            from .optimize import CODEC_FORMAT

            fmt = CODEC_FORMAT[opt["codec"]]
        lossless = bool(opt is not None
                        and opt.get("quality") is None
                        and opt["codec"] in ("webp", "png"))
        if quality is None:
            quality = (opt["quality"] if opt is not None
                       and opt.get("quality") is not None else 90)
        rendered = self.render(pipeline=pipeline,
                               box_predictor=box_predictor)
        src = Path(self.source)
        if path is not None:
            target = Path(path)
            if fmt is not None:
                # A .png path must not end up holding WebP bytes:
                # the extension follows the encoding that won.
                want = {"JPEG": ".jpg", "WEBP": ".webp",
                        "PNG": ".png", "AVIF": ".avif"}.get(fmt)
                if want is not None and target.suffix.lower() != want:
                    target = target.with_suffix(want)
            if target.resolve() == src.resolve():
                # checked AFTER the extension rewrite: the rewrite
                # itself could redirect the target onto the source
                raise ImgGenError(
                    "render output must not overwrite the source")
        else:
            out_dir = src.parent / "edited"
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = src.suffix.lower()
            if opt is not None:
                from .optimize import CODEC_EXT

                ext = CODEC_EXT[opt["codec"]]
            target = out_dir / f"{src.stem}-imgdoc{ext}"
            n = 1
            while target.exists():
                # never clobber the source or an earlier render
                target = out_dir / f"{src.stem}-imgdoc-{n}{ext}"
                n += 1
                if n > 999:
                    raise ImgGenError(
                        f"too many renders beside {src}; "
                        "pass an explicit output path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "AVIF":
            from .optimize import save_optimized

            save_optimized(rendered.image, target, codec="avif",
                           quality=quality)
        else:
            save_image(rendered.image, target, fmt=fmt,
                       quality=quality, lossless=lossless)
        return {
            "input": str(src),
            "output": str(target),
            "ops": [o.op for o in self.ops],
            "bytes": target.stat().st_size,
            "kind": "image",
            "rungs": rendered.rungs,
            "provenance": rendered.provenance,
            "notes": rendered.notes,
        }
