"""Speculative decoding + KV-cache config for the local GGUF server.

Two separate speed levers for ``llm/local_server.py`` (which already
handles KV-cache quantization via ``--ctk``/``--ctv``):

1. SPECULATIVE DECODING — a small draft model proposes ``draft_max``
   tokens; the target model verifies them in one forward pass and
   keeps the accepted prefix.  Expected tokens per target step for
   acceptance rate ``alpha`` and ``gamma`` draft tokens is the standard
   speculative-sampling expectation ``(1 - alpha^(gamma+1)) / (1 - alpha)``.
   The draft model MUST differ from the target — drafting from the
   target is pure overhead, so :meth:`SpeculativeConfig.validate`
   rejects that configuration instead of silently wasting a pass.
2. KV CACHE — the knobs are owned by ``GGUFServerManager``; this module
   exposes :func:`kv_cache_args` (same ``--ctk``/``--ctv`` flags) so
   callers build one consistent argv from one place.

Server flags used are llama.cpp's: ``--draft-model`` (draft GGUF path),
``--draft-max`` (draft tokens per step), ``--draft-min`` / ``--draft-p-min``
(acceptance floor).  When no draft model is configured, the config is
a no-op — speculative decoding is opt-in, never a default that slows
a plain server down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "SpeculativeConfig",
    "kv_cache_args",
    "speculative_args",
    "expected_tokens_per_step",
    "estimate_speedup",
]


def expected_tokens_per_step(acceptance_rate: float, draft_tokens: int) -> float:
    """Expected accepted tokens per target forward pass.

    Standard speculative-sampling expectation: with per-token acceptance
    ``alpha`` and ``gamma`` draft tokens, the run of accepted tokens is
    geometric — ``E = (1 - alpha^(gamma+1)) / (1 - alpha)``.  At
    ``alpha = 0`` this is exactly 1 (the bonus token); at ``alpha -> 1``
    it tends to ``gamma + 1``.
    """
    alpha = min(1.0, max(0.0, float(acceptance_rate)))
    gamma = max(0, int(draft_tokens))
    if gamma == 0:
        return 1.0
    if alpha >= 1.0:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def estimate_speedup(acceptance_rate: float, draft_tokens: int, *,
                     draft_cost_ratio: float = 0.15) -> dict[str, Any]:
    """Expected speedup of speculative decoding over plain autoregressive.

    ``draft_cost_ratio`` is the draft model's per-token cost as a
    fraction of the target's (a ~10x smaller draft is ~0.10–0.20).
    Speedup ≈ expected tokens per step / (1 + gamma * cost_ratio).
    Below 1.0 the configuration LOSES — the caller should not enable it.
    """
    gamma = max(0, int(draft_tokens))
    expected = expected_tokens_per_step(acceptance_rate, gamma)
    cost = 1.0 + gamma * max(0.0, float(draft_cost_ratio))
    speedup = expected / cost if cost else expected
    return {
        "expected_tokens_per_step": round(expected, 3),
        "draft_tokens": gamma,
        "acceptance_rate": float(acceptance_rate),
        "draft_cost_ratio": float(draft_cost_ratio),
        "estimated_speedup": round(speedup, 3),
        "worth_enabling": bool(speedup > 1.0),
    }


def kv_cache_args(kv_cache_quant: str | None) -> list[str]:
    """KV-cache quantization argv (``--ctk``/``--ctv``) — same as local_server."""
    if not kv_cache_quant:
        return []
    return ["--ctk", str(kv_cache_quant), "--ctv", str(kv_cache_quant)]


def speculative_args(config: "SpeculativeConfig") -> list[str]:
    """llama-server argv for speculative decoding (empty when disabled)."""
    if not config.enabled or not config.draft_model:
        return []
    return [
        "--draft-model", str(config.draft_model),
        "--draft-max", str(int(config.draft_max)),
        "--draft-min", str(int(config.draft_min)),
        "--draft-p-min", str(float(config.draft_p_min)),
    ]


@dataclass
class SpeculativeConfig:
    """Speculative-decoding configuration for one GGUF server."""

    enabled: bool = False
    draft_model: str = ""
    target_model: str = ""
    draft_max: int = 4
    draft_min: int = 1
    draft_p_min: float = 0.9
    kv_cache_quant: str | None = None
    draft_cost_ratio: float = 0.15
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[bool, str]:
        """True when the config is safe to pass to llama-server."""
        if not self.enabled:
            return True, "speculative decoding disabled"
        if not self.draft_model:
            return False, "speculative decoding needs a draft_model path"
        if self.draft_model == self.target_model and self.target_model:
            return False, ("draft_model == target_model: drafting from the "
                           "target is pure overhead, never a speedup")
        if self.draft_max < 1:
            return False, "draft_max must be >= 1"
        if not 0.0 < self.draft_p_min <= 1.0:
            return False, "draft_p_min must be in (0, 1]"
        return True, "ok"

    def to_server_args(self) -> list[str]:
        """Full argv fragment: speculative flags + KV-cache flags."""
        ok, reason = self.validate()
        if not ok:
            _log.warning("speculative config invalid (%s) — no flags emitted",
                         reason)
            return list(kv_cache_args(self.kv_cache_quant))
        return speculative_args(self) + kv_cache_args(self.kv_cache_quant)

    def estimate(self, acceptance_rate: float) -> dict[str, Any]:
        """Expected speedup at a measured acceptance rate."""
        return estimate_speedup(
            acceptance_rate, self.draft_max,
            draft_cost_ratio=self.draft_cost_ratio)
