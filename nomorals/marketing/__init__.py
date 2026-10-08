"""Marketing automation, agent-native (build-map Phase 23).

Modules here run as the owner's marketing team: AEO/GEO visibility
tracking (share of answer, not rank), self-hosted send layers, etc.
"""

from .aeo import AEOTracker, VisibilityReport
from .send import SendEngine, control_send, get_engine, render_template

__all__ = ["AEOTracker", "VisibilityReport", "SendEngine", "control_send", "get_engine", "render_template"]
