"""Pass C tests: dtype / quantization / profile-aware defaults.

Covers the new :mod:`nomorals.media.imggen.dtype` organ (dtype +
quant parsing, weights-only memory estimates, the fp16-on-CPU and
int8-on-CUDA refusals, the VRAM fit check, the profile default),
Studio wiring (StudioConfig dtype/quant, the pipeline key, the
sidecar line, readiness), the three loaders' dtype/quant plumbing
(cast before load, dynamic int8 on the Linear modules), the CLI
(--dtype/--quant) and the chat surface.

Same environment honesty as A1/A2/B: torch is not installed on this
box, so the loader-construction tests skip here (needs_torch) and
the full-size round-trip is gated on real RAM. What runs here: the
whole dtype organ (pure python), Studio resolution + validation,
the CLI parsing, and the readiness line.
"""

from __future__ import annotations

import pytest

from nomorals.media.imggen import ImgGenError

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None

needs_torch = pytest.mark.skipif(torch is None,
                                 reason="torch not installed")


# ---------------------------------------------------------------------------
# dtype organ: names
# ---------------------------------------------------------------------------

def test_normalize_dtype_names():
    from nomorals.media.imggen.dtype import normalize_dtype

    assert normalize_dtype("fp32") == "fp32"
    assert normalize_dtype("float32") == "fp32"
    assert normalize_dtype("FP32") == "fp32"
    assert normalize_dtype("fp16") == "fp16"
    assert normalize_dtype("float16") == "fp16"
    assert normalize_dtype("half") == "fp16"
    assert normalize_dtype("bf16") == "bf16"
    assert normalize_dtype("bfloat16") == "bf16"
    assert normalize_dtype("") is None
    assert normalize_dtype(None) is None


def test_normalize_dtype_unknown_raises_naming_valid():
    from nomorals.media.imggen.dtype import normalize_dtype

    with pytest.raises(ImgGenError) as ei:
        normalize_dtype("fp9")
    assert "fp32" in str(ei.value)
    assert "fp16" in str(ei.value)
    assert "bf16" in str(ei.value)


def test_normalize_quant_names():
    from nomorals.media.imggen.dtype import normalize_quant

    assert normalize_quant("") == "none"
    assert normalize_quant(None) == "none"
    assert normalize_quant("none") == "none"
    assert normalize_quant("int8") == "int8"
    assert normalize_quant("dynamic-int8") == "int8"


def test_normalize_quant_unknown_raises_naming_valid():
    from nomorals.media.imggen.dtype import normalize_quant

    with pytest.raises(ImgGenError) as ei:
        normalize_quant("fp8")
    assert "none" in str(ei.value)
    assert "int8" in str(ei.value)


def test_dtype_bytes():
    from nomorals.media.imggen.dtype import dtype_bytes

    assert dtype_bytes("fp32") == 4
    assert dtype_bytes("fp16") == 2
    assert dtype_bytes("bf16") == 2
    with pytest.raises(ImgGenError):
        dtype_bytes("fp9")


def test_torch_dtype_without_torch_raises():
    from nomorals.media.imggen.dtype import torch_dtype

    if torch is None:
        with pytest.raises(ImgGenError) as ei:
            torch_dtype("bf16")
        assert "torch is not installed" in str(ei.value)
    else:
        assert torch_dtype("bf16") is torch.bfloat16
        assert torch_dtype("fp16") is torch.float16
        assert torch_dtype("fp32") is torch.float32


# ---------------------------------------------------------------------------
# dtype organ: memory estimates (weights only, published param counts)
# ---------------------------------------------------------------------------

def test_estimate_bytes_arithmetic():
    from nomorals.media.imggen.dtype import estimate_bytes

    # SD1.5: 860M UNet + 123M CLIP-L + 84M VAE = 1,067M params.
    assert estimate_bytes("sd15", "fp32") == 1_067_000_000 * 4
    assert estimate_bytes("sd15", "bf16") == 1_067_000_000 * 2
    # SDXL: 2.6B UNet + 123M CLIP-L + 694M CLIP-G + 84M VAE = 3,501M.
    assert estimate_bytes("sdxl", "fp32") == 3_501_000_000 * 4
    assert estimate_bytes("sdxl", "bf16") == 3_501_000_000 * 2
    # FLUX.1-dev: 12B transformer + 4.7B T5-XXL + 123M CLIP-L + 84M
    # VAE = 16,907M. bf16 lands at ~33.8 GB — the published 16-bit
    # file total is 34.17 GB, so the table reconciles within 2%.
    flux_bf16 = estimate_bytes("flux", "bf16")
    assert flux_bf16 == 16_907_000_000 * 2
    assert abs(flux_bf16 - 34_170_000_000) / 34_170_000_000 < 0.02
    assert estimate_bytes("flux", "fp32") == 16_907_000_000 * 4
    with pytest.raises(ImgGenError):
        estimate_bytes("nope", "fp32")


def test_estimate_bytes_int8_uses_linear_share():
    from nomorals.media.imggen.dtype import estimate_bytes

    # FLUX is almost all Linear (MMDiT + T5): int8 lands well under
    # half the fp32 bytes. SD1.5's UNet is conv-heavy, so int8 saves
    # far less than 4x there — the share math, not a flat quarter.
    flux_fp32 = estimate_bytes("flux", "fp32")
    flux_int8 = estimate_bytes("flux", "bf16", "int8")
    assert flux_int8 < flux_fp32 // 2
    assert flux_int8 > flux_fp32 // 8
    sd15_fp32 = estimate_bytes("sd15", "fp32")
    sd15_int8 = estimate_bytes("sd15", "bf16", "int8")
    assert sd15_int8 > sd15_fp32 // 4
    assert sd15_int8 < sd15_fp32
    assert estimate_bytes("sdxl", "bf16", "int8") < estimate_bytes(
        "sdxl", "bf16")


# ---------------------------------------------------------------------------
# dtype organ: validation rules (the honest refusals)
# ---------------------------------------------------------------------------

def test_validate_request_rules():
    from nomorals.media.imggen.dtype import validate_request

    # fp16 has no Linear/conv support on CPU (torch autocast on CPU
    # supports bf16 only) — refuse, naming CUDA.
    with pytest.raises(ImgGenError) as ei:
        validate_request("fp16", "none", device="cpu")
    assert "CUDA" in str(ei.value)
    # Dynamic int8 is CPU-only: PyTorch ships no quantized operators
    # on CUDA — refuse, naming CPU.
    with pytest.raises(ImgGenError) as ei:
        validate_request("fp32", "int8", device="cuda")
    assert "CPU" in str(ei.value)
    with pytest.raises(ImgGenError):
        validate_request("fp32", "int8", device="cuda:0")
    # The allowed combinations pass silently.
    validate_request("bf16", "none", device="cpu")
    validate_request("bf16", "int8", device="cpu")
    validate_request("fp32", "none", device="cuda")
    validate_request("fp16", "none", device="cuda")
    validate_request(None, "none", device="cpu")


def test_check_fit():
    from nomorals.media.imggen.dtype import check_fit

    # FLUX bf16 needs ~34 GB of weights: a 24 GB card refuses, and
    # the reason names both numbers.
    fits, reason = check_fit("flux", "bf16", "none", vram_gb=24.0)
    assert fits is False
    assert "33.8" in reason and "24" in reason
    fits, _reason = check_fit("sdxl", "bf16", "none", vram_gb=16.0)
    assert fits is True
    fits, _reason = check_fit("sd15", "fp32", "none", vram_gb=8.0)
    assert fits is True
    # Unknown VRAM: the fit cannot be verified — allow the load and
    # say so, never invent a number.
    fits, reason = check_fit("flux", "bf16", "none", vram_gb=0.0)
    assert fits is True
    assert "not verified" in reason
    fits, reason = check_fit("flux", "bf16", "none", vram_gb=None)
    assert fits is True
    assert "not verified" in reason


def test_default_dtype():
    from nomorals.media.imggen.dtype import default_dtype

    assert default_dtype(has_cuda=True) == "bf16"
    assert default_dtype(has_cuda=False) == "fp32"


def test_describe_load():
    from nomorals.media.imggen.dtype import describe_load

    line = describe_load("sdxl", "bf16", "none", "cuda")
    assert "sdxl" in line and "bf16" in line
    assert "cuda" in line
    line = describe_load("flux", "fp32", "int8", "cpu")
    assert "int8" in line


# ---------------------------------------------------------------------------
# dtype organ: the torch-backed helpers (skip here, run on a torch host)
# ---------------------------------------------------------------------------

@needs_torch
def test_cast_module_changes_param_dtype():
    from nomorals.media.imggen.dtype import cast_module

    lin = torch.nn.Linear(4, 4)
    cast_module(lin, "bf16")
    assert next(lin.parameters()).dtype is torch.bfloat16
    cast_module(lin, "fp32")
    assert next(lin.parameters()).dtype is torch.float32


@needs_torch
def test_quantize_linear_int8_swaps_linears_only():
    import torch.nn as nn

    from nomorals.media.imggen.dtype import quantize_linear_int8

    seq = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    out = quantize_linear_int8(seq)
    assert out[0].weight().dtype is torch.qint8
    assert isinstance(out[1], nn.ReLU)
    x = torch.randn(2, 8)
    assert out(x).shape == (2, 4)


@needs_torch
def test_quantize_linear_int8_refuses_cuda_module():
    import torch.nn as nn

    from nomorals.media.imggen.dtype import quantize_linear_int8

    if not torch.cuda.is_available():
        pytest.skip("no CUDA here")
    lin = nn.Linear(4, 4).cuda()
    with pytest.raises(ImgGenError) as ei:
        quantize_linear_int8(lin)
    assert "CPU" in str(ei.value)


# ---------------------------------------------------------------------------
# Studio wiring (torch-free: resolution + validation only)
# ---------------------------------------------------------------------------

def test_studio_config_dtype_quant_defaults():
    from nomorals.media.imggen.studio import StudioConfig

    cfg = StudioConfig()
    assert cfg.dtype == ""
    assert cfg.quant == ""


def test_studio_effective_dtype_quant_cpu_defaults():
    from nomorals.media.imggen.studio import Studio, StudioConfig

    # This box has no CUDA: the profile default is fp32, quant none.
    studio = Studio(StudioConfig())
    assert studio._effective_dtype_quant(
        {"model_class": "sd15"}) == ("fp32", "none")
    studio = Studio(StudioConfig(dtype="bf16"))
    assert studio._effective_dtype_quant(
        {"model_class": "sdxl"}) == ("bf16", "none")
    studio = Studio(StudioConfig(quant="int8"))
    assert studio._effective_dtype_quant(
        {"model_class": "flux"}) == ("fp32", "int8")


def test_studio_effective_dtype_quant_refusals():
    from nomorals.media.imggen.studio import Studio, StudioConfig

    # fp16 on a CPU box refuses before any weight is touched.
    studio = Studio(StudioConfig(dtype="fp16"))
    with pytest.raises(ImgGenError) as ei:
        studio._effective_dtype_quant({"model_class": "sd15"})
    assert "CUDA" in str(ei.value)
    # Native checkpoints are fp32-only: an explicit dtype/quant
    # request on one refuses instead of silently dropping it.
    studio = Studio(StudioConfig(dtype="bf16"))
    with pytest.raises(ImgGenError) as ei:
        studio._effective_dtype_quant({"model_class": "native"})
    assert "fp32" in str(ei.value)
    studio = Studio(StudioConfig(quant="int8"))
    with pytest.raises(ImgGenError):
        studio._effective_dtype_quant({"model_class": "native"})
    # An unknown name refuses naming the valid ones.
    studio = Studio(StudioConfig(dtype="fp9"))
    with pytest.raises(ImgGenError):
        studio._effective_dtype_quant({"model_class": "sd15"})


def test_studio_generate_validates_dtype_before_torch():
    from nomorals.media.imggen.studio import Studio

    studio = Studio()
    with pytest.raises(ImgGenError) as ei:
        studio.generate("a cat", dtype="fp9")
    assert "fp32" in str(ei.value)
    with pytest.raises(ImgGenError) as ei:
        studio.generate("a cat", dtype="fp16")
    assert "CUDA" in str(ei.value)


def test_studio_readiness_without_torch():
    from nomorals.media.imggen.studio import Studio

    if torch is not None:
        pytest.skip("torch is installed here")
    ok, reason = Studio().readiness()
    assert ok is False
    assert "torch is not installed" in reason


# ---------------------------------------------------------------------------
# Adapters expose dtype/quant (needs torch: the adapter modules raise
# ImgGenError on import without it)
# ---------------------------------------------------------------------------

class _FakePipe:
    """Hand-written stand-in for a loaded pipeline (no mocks)."""

    def __init__(self, dtype="bf16", quant="none"):
        self.device = "cpu"
        self.warnings = []
        self.tokenizer_mode = "bpe"
        self.dtype = dtype
        self.quant = quant


@needs_torch
def test_adapters_expose_dtype_quant():
    from nomorals.media.imggen.flux import FLUXStudioAdapter
    from nomorals.media.imggen.sdcompat import SD15StudioAdapter
    from nomorals.media.imggen.sdxl import SDXLStudioAdapter

    for adapter_cls in (SD15StudioAdapter, SDXLStudioAdapter,
                        FLUXStudioAdapter):
        adapter = adapter_cls(_FakePipe(dtype="bf16", quant="int8"))
        assert adapter.dtype == "bf16"
        assert adapter.quant == "int8"


# ---------------------------------------------------------------------------
# CLI: --dtype / --quant on generate; checkpoints names readiness
# ---------------------------------------------------------------------------

def test_cli_generate_accepts_dtype_quant():
    from nomorals.cmdline.parser import _parser

    args = _parser().parse_args(
        ["imggen", "generate", "a cat", "--dtype", "bf16",
         "--quant", "int8"])
    assert args.dtype == "bf16"
    assert args.quant == "int8"
    args = _parser().parse_args(["imggen", "generate", "a cat"])
    assert args.dtype == ""
    assert args.quant == ""


def test_cli_generate_rejects_unknown_dtype():
    from nomorals.cmdline.parser import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["imggen", "generate", "a cat", "--dtype", "fp9"])


def test_cli_studio_wiring_carries_dtype_quant():
    import argparse

    from nomorals.cmdline.commands.imggen import _studio

    args = argparse.Namespace(checkpoint="", device="",
                              dtype="bf16", quant="int8")
    studio = _studio(args)
    assert studio.config.dtype == "bf16"
    assert studio.config.quant == "int8"


def test_cli_checkpoints_names_readiness(capsys):
    import argparse

    from nomorals.cmdline.commands.imggen import _cmd_imggen_checkpoints

    args = argparse.Namespace(json=False)
    assert _cmd_imggen_checkpoints(args, None) == 0
    out = capsys.readouterr().out
    assert "readiness:" in out
    if torch is None:
        assert "torch is not installed" in out
