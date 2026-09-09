"""NoMorals Core (NMC).

A self-hosted, self-improving multi-agent personal AI substrate.

The package is layered; see ARCHITECTURE.md. Layers may only import downward::

    L1 core       -> kernel primitives
    L2 storage    -> persistence
    L3 cognition  -> memory, llm, training
    L4 capability -> tools, social
    L5 agents     -> orchestration
    L6 missions   -> autonomy
    L7 surface    -> cli, api

The invariant is machine-checked by ``tests/test_layering.py``.
"""

from __future__ import annotations

from .version import __version__, version_info

__all__ = ["__version__", "version_info"]
