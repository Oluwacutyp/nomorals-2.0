"""Marketing automation, agent-native (build-map Phase 23).

Modules here run as the owner's marketing team: AEO/GEO visibility
tracking (share of answer, not rank), self-hosted send layers, etc.
"""

from .aeo import AEOTracker, VisibilityReport

__all__ = ["AEOTracker", "VisibilityReport"]
