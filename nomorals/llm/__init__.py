"""L3 model plane: providers, routing, registry, downloads."""

from __future__ import annotations

from .adjudicate import Judge, Judgment, adjudicate, fan_out
from .base import LLMProvider, LLMResponse, Message, SamplingParams, Usage
from .benchmarks import BenchmarkDB, BenchmarkSample, benchmark_model, seed_synthetic
from .brain import Brain, brain_for, explain_failure, get_brain, reset_brain
from .broker import BrokerConstraints, ModelBroker, NoCandidate
from .capabilities import Capability, ModelCard, capability_from, provider_capabilities
from .context_fit import (
    DEFAULT_CONTEXT_TOKENS,
    TASK_FIT_CHAINS,
    Compact,
    ContextStrategy,
    FitResult,
    SummarizeMiddle,
    TruncateOldest,
    fit_messages,
    fit_prompt,
    strategy_for,
)
from .failures import (
    FailureClass,
    FailureInfo,
    RecoveryPolicy,
    RECOVERY,
    classify_failure,
    parse_retry_after,
)
from .lifecycle import (
    LifecycleError,
    LocalGGUFProvisioner,
    ManagedModel,
    ModelLifecycle,
    ModelProvisioner,
    STAGES,
)
from .router import LLMRouter, ProviderHealth
from .prompts import render_prompt, system_prompt_for
from .cost_display import (
    CostDisplay,
    control_cost,
    format_cost,
    get_display,
    maybe_cost_footer,
    parse_budget_nl,
)

__all__ = [
    "BenchmarkDB",
    "BenchmarkSample",
    "Brain",
    "BrokerConstraints",
    "Capability",
    "Compact",
    "ContextStrategy",
    "CostDisplay",
    "DEFAULT_CONTEXT_TOKENS",
    "FailureClass",
    "FailureInfo",
    "FitResult",
    "Judge",
    "Judgment",
    "LifecycleError",
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "LocalGGUFProvisioner",
    "ManagedModel",
    "Message",
    "ModelBroker",
    "ModelCard",
    "ModelLifecycle",
    "ModelProvisioner",
    "NoCandidate",
    "ProviderHealth",
    "RECOVERY",
    "RecoveryPolicy",
    "SamplingParams",
    "STAGES",
    "SummarizeMiddle",
    "TASK_FIT_CHAINS",
    "TruncateOldest",
    "Usage",
    "adjudicate",
    "benchmark_model",
    "brain_for",
    "capability_from",
    "classify_failure",
    "control_cost",
    "explain_failure",
    "fan_out",
    "fit_messages",
    "fit_prompt",
    "format_cost",
    "get_brain",
    "get_display",
    "maybe_cost_footer",
    "parse_budget_nl",
    "parse_retry_after",
    "provider_capabilities",
    "reset_brain",
    "render_prompt",
    "seed_synthetic",
    "strategy_for",
    "system_prompt_for",
]
