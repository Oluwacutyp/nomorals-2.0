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

# Tier 0 (additive): persisted organ bus + promotion rollback registry.
from .organs_bus import PersistedBus
from .tier0_rollback import (
    promotion_status,
    promotions,
    record_promotion,
    rollback,
    rollback_all,
)

__all__ += [
    "PersistedBus",
    "record_promotion",
    "rollback",
    "rollback_all",
    "promotions",
    "promotion_status",
]

# MORE_TO_ADD gaps 4-6 (additive): family mode, location layer, offline
# mode. Stdlib + sibling imports only, so this stays import-safe.
from . import family as family, location as location, offline as offline

__all__ += ["family", "location", "offline"]
