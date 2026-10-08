"""Marketing automation, agent-native (build-map Phase 23).

Modules here run as the owner's marketing team: AEO/GEO visibility
tracking (share of answer, not rank), self-hosted send layers, etc.
"""

from .aeo import AEOTracker, VisibilityReport
from .send import SendEngine, control_send, get_engine, render_template
from .briefs import BriefStore, ContentBrief, ContentScore, BriefDraft, control_brief, control_content
from .guardrails import GuardrailStore, Rule, FiredAction, control_guardrails, evaluate, execute_override, parse_rule

__all__ = ["AEOTracker", "VisibilityReport", "SendEngine", "control_send", "get_engine", "render_template",
           "BriefStore", "ContentBrief", "ContentScore", "BriefDraft", "control_brief", "control_content",
           "GuardrailStore", "Rule", "FiredAction", "control_guardrails", "evaluate", "execute_override",
           "parse_rule"]
