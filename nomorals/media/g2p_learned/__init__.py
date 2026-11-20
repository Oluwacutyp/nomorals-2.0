"""Learned English G2P — the Arpabet prediction model (see model.py)."""

from __future__ import annotations

from .model import (
    GRAPHEMES,
    PHONEME_TABLE,
    G2PLearnedError,
    Model,
    get_model,
    model_status,
    predict_word,
)

__all__ = [
    "GRAPHEMES",
    "PHONEME_TABLE",
    "G2PLearnedError",
    "Model",
    "get_model",
    "model_status",
    "predict_word",
]
