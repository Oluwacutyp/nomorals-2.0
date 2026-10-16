"""Cross-domain taste — descriptors as data, weights learned from signals.

See :mod:`nomorals.preferences.store` (descriptors + storage),
:mod:`nomorals.preferences.adapters` (domain -> descriptor passthroughs),
:mod:`nomorals.preferences.feedback` (signals in, ranking out).
"""

from __future__ import annotations

from .adapters import (
    Adapter,
    DescriptorAdapter,
    PassthroughAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    reset_adapters,
)
from .feedback import (
    APPROVE_WEIGHT,
    DISAPPROVE_WEIGHT,
    learn_from_representation,
    record_signal,
    score_item,
    suggest,
    suggest_ranked,
)
from .store import DOMAIN_DESCRIPTORS, PreferenceStore, extract_axis_values

__all__ = [
    "DOMAIN_DESCRIPTORS",
    "PreferenceStore",
    "extract_axis_values",
    "Adapter",
    "DescriptorAdapter",
    "PassthroughAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "reset_adapters",
    "APPROVE_WEIGHT",
    "DISAPPROVE_WEIGHT",
    "record_signal",
    "suggest",
    "suggest_ranked",
    "score_item",
    "learn_from_representation",
]
