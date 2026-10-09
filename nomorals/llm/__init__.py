"""L3 model plane: providers, routing, registry, downloads."""

from __future__ import annotations

from .base import LLMProvider, LLMResponse, Message, SamplingParams, Usage
from .benchmarks import BenchmarkDB, BenchmarkSample, benchmark_model, seed_synthetic
from .brain import Brain, explain_failure, get_brain, reset_brain
from .broker import BrokerConstraints, ModelBroker, NoCandidate
from .capabilities import Capability, ModelCard, capability_from, provider_capabilities
from .lifecycle import (
    LifecycleError,
    LocalGGUFProvisioner,
    ManagedModel,
    ModelLifecycle,
    ModelProvisioner,
    STAGES,
)
from .router import LLMRouter, ProviderHealth
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
    "CostDisplay",
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
    "SamplingParams",
    "STAGES",
    "Usage",
    "benchmark_model",
    "capability_from",
    "control_cost",
    "explain_failure",
    "format_cost",
    "get_brain",
    "get_display",
    "maybe_cost_footer",
    "parse_budget_nl",
    "provider_capabilities",
    "reset_brain",
    "seed_synthetic",
]
