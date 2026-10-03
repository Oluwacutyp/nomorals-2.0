"""Liger-Kernel: free fused Triton kernels for transformer training.

Liger-Kernel (Apache-2.0, LinkedIn) replaces the wasteful parts of a
transformer training step — RMSNorm, RoPE, SwiGLU, and above all the
cross-entropy — with fused Triton kernels.  The numbers are measured,
not marketing: +20% multi-GPU throughput, **-60% memory usage**
(LLaMA 3-8B, bf16, FSDP1 on 8xA100; plain HF OOMs at 4K context where
HF + Liger scales to 16K).  The biggest single win is the fused linear
cross-entropy: the ``(batch, seq, vocab)`` logits tensor is the largest
allocation in a training step and pure waste — Liger chunks it away.

This module is an *accelerator*, not a backend: it has no train() of
its own.  It answers one question — "is Liger usable here?" — and
hands the verified config fragments to the backends that train real
transformers:

* :mod:`nomorals.training.backends.axolotl` — the ``LigerPlugin`` block
  in the generated YAML (keys copied from the axolotl docs,
  ``docs/custom_integrations.html#liger-kernels``);
* the Colab script in :mod:`nomorals.training.finetune` — ``trl``
  accepts ``SFTConfig(use_liger_kernel=True)`` since trl 0.9.

Deliberately NOT wired into the Unsloth backend or the Unsloth Colab
notebook: Unsloth patches its own fused cross-entropy and attention
kernels into the model at load time, and TRL's liger flag targets HF
model classes — stacking the two risks double-patching the same
modules.  Unsloth already ships the fused-CE win; Liger's marginal
gain there is nil and the conflict risk is real.

Import-safe: ``liger_kernel`` is probed with ``importlib.util``, never
imported at module level.
"""

from __future__ import annotations

import importlib.util

__all__ = [
    "AXOLOTL_LIGER_PLUGIN",
    "axolotl_liger_block",
    "liger_available",
    "pip_hint",
    "trl_sft_kwargs",
]


def _spec_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def liger_available() -> tuple[bool, str]:
    """(is Liger-Kernel importable here, human-readable reason)."""
    if _spec_exists("liger_kernel"):
        return True, "liger_kernel installed — fused kernels enabled"
    return (
        False,
        "liger_kernel not installed (pip install liger-kernel — "
        "optional; training works without it, just hungrier)",
    )


def trl_sft_kwargs() -> dict[str, bool]:
    """``use_liger_kernel=True`` when the package is importable, else {}.

    Passed straight into ``trl.SFTConfig`` by callers.  Callers must
    still guard the SFTConfig signature (old trl raises TypeError on an
    unknown kwarg) — see the generated Colab script for the probe.
    """
    ok, _ = liger_available()
    return {"use_liger_kernel": True} if ok else {}


def pip_hint() -> str:
    return "pip install liger-kernel  # free: fused Triton kernels, -60% VRAM"


#: The axolotl plugin entry — verified against the axolotl docs
#: (Custom Integrations → Liger Kernels).  The per-kernel toggles are
#: appended by :func:`axolotl_liger_block`.
AXOLOTL_LIGER_PLUGIN = "axolotl.integrations.liger.LigerPlugin"


def axolotl_liger_block() -> list[str]:
    """YAML lines enabling Liger inside an axolotl config.

    ``liger_fused_linear_cross_entropy`` is the money line (the logits
    tensor never materializes — the single biggest VRAM allocation in
    the step disappears); the norm/activation/rope lines fuse the rest.
    ``liger_use_token_scaling`` is deliberately left OFF — it changes
    the loss semantics (token-averaged CE), not just the kernels, and
    must stay an explicit user choice.
    """
    return [
        "plugins:",
        f"  - {AXOLOTL_LIGER_PLUGIN}",
        "",
        "liger_rope: true",
        "liger_rms_norm: true",
        "liger_glu_activation: true",
        "liger_layer_norm: true",
        "liger_fused_linear_cross_entropy: true",
    ]
