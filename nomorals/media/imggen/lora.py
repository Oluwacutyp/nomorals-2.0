"""LoRA (Low-Rank Adaptation, Hu et al. 2021, arXiv:2106.09685) for UNets.

A frozen weight W (d_out × d_in) is adapted as:

    W' = W + (alpha / r) * B @ A

where B is (d_out × r) zero-initialized, A is (r × d_in)
kaiming-initialized, r << min(d_out, d_in) is the rank. Only A and B
train — typically <1% of the base parameters.

Injection targets: the attention projections (Q/K/V/O) where style and
content binding actually live, plus optionally the cross-attention
K/V that carry text conditioning. Conv 1×1 projections are handled as
their channel matrix.
"""

from __future__ import annotations

import math

from . import ImgGenError, TORCH_AVAILABLE

if not TORCH_AVAILABLE:  # pragma: no cover - torch missing
    raise ImgGenError(
        "nomorals.media.imggen.lora needs PyTorch: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoRALinear",
    "LoRAConv2d",
    "inject_lora",
    "lora_parameters",
    "merge_lora",
    "DEFAULT_TARGET_SUFFIXES",
]

#: module name suffixes that get LoRA adapters (attention projections).
DEFAULT_TARGET_SUFFIXES = ("to_q", "to_k", "to_v", "qkv", "proj")


class LoRALinear(nn.Module):
    """nn.Linear + trainable low-rank update. Frozen base weight."""

    def __init__(self, base: nn.Linear, r: int = 8,
                 alpha: float = 16.0) -> None:
        super().__init__()
        if r < 1:
            raise ImgGenError("LoRA rank must be >= 1")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_a = nn.Parameter(
            torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # (alpha/r) * (x @ A^T @ B^T)
        update = (x @ self.lora_a.t()) @ self.lora_b.t()
        return out + self.scaling * update

    def merged_weight(self) -> torch.Tensor:
        """W + (alpha/r) B A — for baking into the base at export."""
        return (self.base.weight.data
                + self.scaling * (self.lora_b @ self.lora_a))


class LoRAConv2d(nn.Module):
    """1×1 nn.Conv2d + trainable low-rank update on the channel matrix.

    Only valid for kernel_size == 1 (the attention projections); a 3×3
    conv's spatial structure doesn't factor this way.
    """

    def __init__(self, base: nn.Conv2d, r: int = 8,
                 alpha: float = 16.0) -> None:
        super().__init__()
        if base.kernel_size != (1, 1):
            raise ImgGenError(
                "LoRAConv2d only supports 1x1 convolutions "
                f"(got {base.kernel_size})")
        if r < 1:
            raise ImgGenError("LoRA rank must be >= 1")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_channels))
        self.lora_b = nn.Parameter(torch.zeros(base.out_channels, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # Apply (alpha/r) * B A as a 1x1 conv on the channel axis.
        w = (self.scaling * (self.lora_b @ self.lora_a)
             ).unsqueeze(-1).unsqueeze(-1)
        return out + F.conv2d(x, w.to(x.dtype))

    def merged_weight(self) -> torch.Tensor:
        w = self.base.weight.data.squeeze(-1).squeeze(-1)
        merged = w + self.scaling * (self.lora_b @ self.lora_a)
        return merged.unsqueeze(-1).unsqueeze(-1)


def _wrap_module(mod: nn.Module, r: int, alpha: float) -> nn.Module | None:
    if isinstance(mod, nn.Linear):
        return LoRALinear(mod, r=r, alpha=alpha)
    if isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
        return LoRAConv2d(mod, r=r, alpha=alpha)
    return None


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0,
                target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
                ) -> list[str]:
    """Wrap matching submodules of ``model`` with LoRA adapters in place.

    Returns the dotted names that were wrapped. Base weights are frozen;
    only the A/B matrices train. Idempotent per module name — calling
    twice on the same model wraps nothing new.
    """
    wrapped: list[str] = []
    for name, mod in list(model.named_modules()):
        if not name or not name.endswith(target_suffixes):
            continue
        # Don't double-wrap.
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        existing = getattr(parent, leaf)
        if isinstance(existing, (LoRALinear, LoRAConv2d)):
            continue
        new = _wrap_module(existing, r, alpha)
        if new is None:
            continue
        setattr(parent, leaf, new)
        wrapped.append(name)
    return wrapped


def lora_parameters(model: nn.Module):
    """Yield only the LoRA A/B parameters (what the optimizer trains)."""
    for mod in model.modules():
        if isinstance(mod, (LoRALinear, LoRAConv2d)):
            yield mod.lora_a
            yield mod.lora_b


def merge_lora(model: nn.Module) -> None:
    """Bake LoRA updates into base weights in place (for export)."""
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.base.weight.data.copy_(mod.merged_weight())
            mod.lora_a.data.zero_()
            mod.lora_b.data.zero_()
        elif isinstance(mod, LoRAConv2d):
            mod.base.weight.data.copy_(mod.merged_weight())
            mod.lora_a.data.zero_()
            mod.lora_b.data.zero_()
