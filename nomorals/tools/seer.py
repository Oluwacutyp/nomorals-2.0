"""Seer tool registration: vision as a sensor for the agentic orchestrator.

Thin wrapper — the real implementation lives in ``nomorals.vision``.
"""

from __future__ import annotations

from typing import Any


def register(registry: Any) -> None:
    """Register the ``see`` tool."""
    from ..vision import register as _register_vision
    _register_vision(registry)
