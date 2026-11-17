"""Dtype / quantization / profile-aware defaults for the image studio.

Pass C of the imggen ladder (A1 SD1.5 → A2 SDXL → B FLUX → C this
pass). Before this module the loaders ran everything in fp32 — the
dtype the weights happened to arrive in — with no way to ask for
less, no way to know what a render would cost, and no profile-aware
choice. This module is the one place those decisions live:

- names: ``fp32`` / ``fp16`` / ``bf16`` (with the ``float32`` /
  ``float16`` / ``half`` / ``bfloat16`` aliases) and quant ``none`` /
  ``int8`` (dynamic int8, the one real post-training quant path
  torch ships);
- memory: weights-only estimates from the published parameter
  counts, so a load can be refused *before* CUDA runs out — never
  an OOM mid-render;
- the honest refusals, each traced to the real API limit:
  fp16 needs CUDA (torch autocast on CPU supports bf16 only;
  ``torch.nn.functional.linear`` on fp32-CPU tensors has no fp16
  kernel), and dynamic int8 is CPU-only (PyTorch ships no
  quantized operators on CUDA — docs.pytorch.org quantization
  guide);
- the profile default: bf16 on CUDA, fp32 on CPU. No fixed
  ceiling anywhere — an explicit request is always honored when
  the hardware supports it, and power mode has nothing to lift
  here because nothing is capped.

Everything in this module is pure python except
:func:`torch_dtype`, :func:`cast_module` and
:func:`quantize_linear_int8`, which need torch and raise an honest
:class:`ImgGenError` without it. The estimator's parameter counts
are the published ones (SD1.5: 860M UNet + 123M CLIP-L + 84M VAE;
SDXL: 2.6B UNet + 123M CLIP-L + 694M CLIP-G + 84M VAE; FLUX.1-dev:
12B transformer + 4.7B T5-XXL + 123M CLIP-L + 84M VAE) and
reconcile with the published 16-bit file totals (SDXL 3.5B →
6.94 GB; FLUX 16.9B → 34.17 GB). The Linear shares are documented
approximations (±) for the quant estimate only — the estimator is
a fit decision, not a guarantee, and the load itself is the proof.
"""

from __future__ import annotations

from . import TORCH_AVAILABLE, ImgGenError

__all__ = [
    "DTYPE_NAMES",
    "QUANT_NAMES",
    "normalize_dtype",
    "normalize_quant",
    "dtype_bytes",
    "torch_dtype",
    "estimate_bytes",
    "resolve_request",
    "validate_request",
    "check_fit",
    "default_dtype",
    "describe_load",
    "cast_module",
    "quantize_linear_int8",
]

#: Canonical dtype names, in the order Studio prefers them.
DTYPE_NAMES: tuple[str, ...] = ("fp32", "fp16", "bf16")

_DTYPE_ALIASES = {
    "float32": "fp32",
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
}

#: Bytes per parameter for each dtype (weights only).
_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

#: Canonical quant names. ``int8`` is torch dynamic quantization
#: (weights in int8, activations quantized at compute time) — the
#: only post-training quant path torch ships for nn.Linear.
QUANT_NAMES: tuple[str, ...] = ("none", "int8")

_QUANT_ALIASES = {"dynamic-int8": "int8"}

#: Weights-only parameter counts per model class, per component.
#: Published figures (see the module docstring); the VAE is the
#: 84M LDM AutoencoderKL in all three families.
MODEL_PARAMS: dict[str, dict[str, int]] = {
    "sd15": {"unet": 860_000_000, "text_encoder": 123_000_000,
             "vae": 84_000_000},
    "sdxl": {"unet": 2_600_000_000, "text_encoder_l": 123_000_000,
             "text_encoder_g": 694_000_000, "vae": 84_000_000},
    "flux": {"transformer": 12_000_000_000, "t5": 4_700_000_000,
             "clip": 123_000_000, "vae": 84_000_000},
}

#: Fraction of each component's parameters living in nn.Linear
#: modules — the part dynamic int8 can quantize (convs stay at the
#: chosen dtype). Documented approximations: transformer stacks are
#: almost all Linear; UNet resnets are convs; the VAE is conv-only.
LINEAR_SHARE: dict[str, dict[str, float]] = {
    "sd15": {"unet": 0.20, "text_encoder": 0.90, "vae": 0.0},
    "sdxl": {"unet": 0.30, "text_encoder_l": 0.90,
             "text_encoder_g": 0.90, "vae": 0.0},
    "flux": {"transformer": 0.95, "t5": 0.90, "clip": 0.90,
             "vae": 0.0},
}

#: Activation + CUDA-context reserve applied on top of the weights
#: when deciding whether a model fits a card. A single-image render
#: at the model's native size holds a few GB of activations beside
#: the weights; 15% is the documented approximation, not a promise.
_FIT_RESERVE = 1.15


def normalize_dtype(name: str | None) -> str | None:
    """Canonical dtype name, or None for \"not specified\".

    Raises ImgGenError naming the valid names for anything else —
    never silently coerces an unknown name to a default.
    """
    if name is None:
        return None
    key = str(name).strip().lower()
    if not key:
        return None
    if key in DTYPE_NAMES:
        return key
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    raise ImgGenError(
        f"unknown dtype {name!r} — use one of: "
        + ", ".join(DTYPE_NAMES))


def normalize_quant(name: str | None) -> str:
    """Canonical quant name; \"\" / None means \"none\".

    Raises ImgGenError naming the valid names for anything else.
    """
    if name is None:
        return "none"
    key = str(name).strip().lower()
    if not key or key == "none":
        return "none"
    if key in QUANT_NAMES:
        return key
    if key in _QUANT_ALIASES:
        return _QUANT_ALIASES[key]
    raise ImgGenError(
        f"unknown quantization {name!r} — use one of: "
        + ", ".join(QUANT_NAMES))


def dtype_bytes(name: str) -> int:
    """Bytes per parameter for a canonical dtype name."""
    canonical = normalize_dtype(name)
    if canonical is None:
        raise ImgGenError(
            "no dtype given — use one of: " + ", ".join(DTYPE_NAMES))
    return _DTYPE_BYTES[canonical]


def torch_dtype(name: str):
    """The torch dtype for a canonical dtype name.

    Raises ImgGenError (never a bare ImportError) when torch is
    not installed.
    """
    canonical = normalize_dtype(name)
    if canonical is None:
        raise ImgGenError(
            "no dtype given — use one of: " + ", ".join(DTYPE_NAMES))
    if not TORCH_AVAILABLE:
        raise ImgGenError(
            "torch is not installed — dtype selection needs torch: "
            "pip install torch")
    import torch

    return {"fp32": torch.float32, "fp16": torch.float16,
            "bf16": torch.bfloat16}[canonical]


def estimate_bytes(model_class: str, dtype: str = "fp32",
                   quant: str = "none") -> int:
    """Weights-only bytes for a (model class, dtype, quant) triple.

    ``int8`` quantizes only the Linear share to 1 byte/param (the
    dynamic-quant scope); convs and embeddings stay at the chosen
    dtype's bytes. Raises ImgGenError for an unknown class/dtype/
    quant — the table is the source of truth, never a guess.
    """
    if model_class not in MODEL_PARAMS:
        raise ImgGenError(
            f"cannot estimate weights for model class "
            f"{model_class!r} — known classes: "
            + ", ".join(sorted(MODEL_PARAMS)))
    canonical = normalize_dtype(dtype)
    if canonical is None:
        canonical = "fp32"
    quant_name = normalize_quant(quant)
    total = 0
    for component, params in MODEL_PARAMS[model_class].items():
        per_param = _DTYPE_BYTES[canonical]
        if quant_name == "int8":
            share = LINEAR_SHARE[model_class][component]
            per_param = share * 1.0 + (1.0 - share) * per_param
        total += int(params * per_param)
    return total


def resolve_request(dtype: str | None, quant: str | None,
                    *, device: str = "") -> tuple[str, str, str]:
    """(dtype, quant, device) with defaults filled and rules checked.

    ``dtype`` "" / None → the profile default for the resolved
    device (bf16 on CUDA, fp32 on CPU); ``quant`` "" / None →
    "none"; ``device`` "" → "cuda" when torch sees a GPU, else
    "cpu". Raises ImgGenError on any invalid combination (unknown
    name, fp16 on CPU, int8 on CUDA) — before any weight is
    touched, so callers validate first and load second.
    """
    want_dtype = normalize_dtype(dtype)
    want_quant = normalize_quant(quant)
    if device:
        resolved_device = str(device)
    elif TORCH_AVAILABLE:
        import torch

        resolved_device = ("cuda" if torch.cuda.is_available()
                           else "cpu")
    else:
        resolved_device = "cpu"
    if want_dtype is None:
        want_dtype = default_dtype(
            has_cuda=resolved_device.startswith("cuda"))
    validate_request(want_dtype, want_quant,
                     device=resolved_device)
    return want_dtype, want_quant, resolved_device


def _is_cuda(device: str) -> bool:
    return str(device or "").strip().lower().startswith("cuda")


def validate_request(dtype: str | None, quant: str,
                     *, device: str) -> None:
    """Refuse a (dtype, quant, device) combination torch cannot run.

    - fp16 on CPU: torch has no fp16 Linear/conv kernels on CPU
      (CPU autocast supports bf16 only) — use bf16 on CPU or run
      on CUDA.
    - int8 on CUDA: PyTorch ships no quantized operators on CUDA
      (dynamic quantization is CPU-only) — run on CPU or drop the
      quant.
    Anything else passes. ``dtype=None`` means \"not specified\" and
    imposes no rule; ``quant`` is normalized first.
    """
    canonical = normalize_dtype(dtype)
    quant_name = normalize_quant(quant)
    if canonical == "fp16" and not _is_cuda(device):
        raise ImgGenError(
            "fp16 needs CUDA — torch has no fp16 Linear/conv "
            "support on CPU (CPU mixed precision is bf16 only); "
            "use bf16 on this box or run on a CUDA device")
    if quant_name == "int8" and _is_cuda(device):
        raise ImgGenError(
            "dynamic int8 quantization is CPU-only — PyTorch "
            "ships no quantized operators on CUDA; run on CPU or "
            "drop the quant (--quant none)")


def check_fit(model_class: str, dtype: str = "fp32",
              quant: str = "none",
              *, vram_gb: float | None) -> tuple[bool, str]:
    """Does the model fit a card with ``vram_gb`` of VRAM?

    Unknown VRAM (None or <= 0) cannot be verified — the load is
    allowed and the reason says so, never an invented number.
    Otherwise the weights estimate plus the activation reserve must
    fit; the reason names both numbers either way.
    """
    need = estimate_bytes(model_class, dtype, quant)
    need_gb = need / 1e9
    if vram_gb is None or vram_gb <= 0:
        return True, (
            f"{model_class} at {dtype} needs ~{need_gb:.1f} GB of "
            f"weights; VRAM unknown — fit not verified")
    have = float(vram_gb)
    required = need_gb * _FIT_RESERVE
    if required <= have:
        return True, (
            f"{model_class} at {dtype} needs ~{need_gb:.1f} GB of "
            f"weights (~{required:.1f} GB with activations) and "
            f"the card has {have:.1f} GB")
    return False, (
        f"{model_class} at {dtype} needs ~{need_gb:.1f} GB of "
        f"weights (~{required:.1f} GB with activations) but the "
        f"card has only {have:.1f} GB — pick a smaller dtype "
        f"(bf16 halves it), quant int8, or a bigger card")


def default_dtype(*, has_cuda: bool) -> str:
    """The profile default dtype: bf16 on CUDA, fp32 on CPU.

    bf16 is the standard inference dtype for SDXL/FLUX (same range
    as fp32, half the bytes); fp32 is the only dtype that runs
    everywhere on CPU. There is no fixed ceiling here — an explicit
    request is always honored when the hardware supports it.
    """
    return "bf16" if has_cuda else "fp32"


def describe_load(model_class: str, dtype: str, quant: str,
                  device: str) -> str:
    """One line naming how a model is loaded: class, dtype, quant,
    device. Used by the readiness message and the image sidecar —
    the choice is always visible, never silent."""
    return (f"{model_class} at {dtype}, quant {quant}, "
            f"on {device}")


def cast_module(module, dtype_name: str):
    """Cast a module's parameters to ``dtype_name`` in place.

    The cast happens *before* weights are loaded into the module,
    so ``load_state_dict`` copies the file's tensors down to the
    target dtype instead of building a full fp32 copy first (for
    FLUX that fp32 copy would be ~67 GB before the cast).
    Raises ImgGenError (never a bare ImportError) without torch.
    """
    module.to(torch_dtype(dtype_name))
    return module


def quantize_linear_int8(module):
    """Dynamic int8 quantization of a module's nn.Linear layers.

    Real torch API: ``torch.quantization.quantize_dynamic(module,
    {nn.Linear}, dtype=torch.qint8)`` — Linear weights go to int8,
    activations stay floating point and are quantized at compute
    time. Convs are untouched (dynamic quantization does not cover
    them). CPU-only by the API's own limit: refuses a module whose
    parameters already sit on CUDA, naming the reason. Returns the
    quantized module (a new object); the caller rebinds it.
    """
    if not TORCH_AVAILABLE:
        raise ImgGenError(
            "torch is not installed — int8 quantization needs "
            "torch: pip install torch")
    import torch
    import torch.nn as nn

    on_cuda = any(p.is_cuda for p in module.parameters())
    if on_cuda:
        raise ImgGenError(
            "dynamic int8 quantization is CPU-only — this module "
            "already sits on CUDA; move it to CPU first or drop "
            "the quant (--quant none)")
    quantized = torch.quantization.quantize_dynamic(
        module, {nn.Linear}, dtype=torch.qint8)
    quantized.eval()
    return quantized
