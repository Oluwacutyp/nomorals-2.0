"""``build_app`` tool: the ten-stack :class:`AppBuilder` as a registry tool.

The real implementation lives in :mod:`nomorals.builders.app_builder`
(the builders organ, L7).  This module is the thin bridge that the tool
registry's ``register_builtins()`` expects under the ``build_app`` name,
so ``nm apps`` and agents can call the builder through the normal tool
surface instead of importing the organ directly.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(registry: Any) -> None:
    """Attach the builders organ's ``build_app`` tool to ``registry``."""
    from ..builders.app_builder import register as _register

    _register(registry)
