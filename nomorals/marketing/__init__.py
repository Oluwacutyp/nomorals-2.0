"""Marketing automation, agent-native (build-map Phase 23).

Modules here run as the owner's marketing team: AEO/GEO visibility
tracking (share of answer, not rank), self-hosted send layers, etc.
"""

from .aeo import (
    AEOTracker, VisibilityReport, Mention, EngineResponse,
    classify_positioning, sentiment_score, expand_prompts,
    site_readiness_check, citation_quality, sparkline,
    AI_CRAWLER_TOKENS, parse_citations, find_mentions, estimate_cost,
)
from .send import (
    SendEngine, control_send, get_engine, render_template,
    spam_score, warmup_plan, deliverability_check, parse_dsn,
)
from .briefs import (
    BriefStore, ContentBrief, ContentScore, BriefDraft, control_brief, control_content,
    letter_grade, brief_quality, geo_score,
)
from .guardrails import (
    GuardrailStore, Rule, FiredAction, control_guardrails, evaluate,
    execute_override, parse_rule, RULE_PRESETS, add_preset, preview,
    record_metrics, fatigue_signal, format_naira, parse_naira_kobo,
)
from .competitor import (
    CompetitorStore, CompetitorReport, Post, Pillar, control_competitor, aeo_compare,
    analyze_pillars, analyze_cadence, analyze_engagement,
    top_posts, viral_posts, best_times, analyze_hashtags, key_insights,
    benchmark, content_gaps, to_briefs,
)

__all__ = [
    "AEOTracker", "VisibilityReport", "Mention", "EngineResponse",
    "classify_positioning", "sentiment_score", "expand_prompts",
    "site_readiness_check", "citation_quality", "sparkline",
    "AI_CRAWLER_TOKENS", "parse_citations", "find_mentions", "estimate_cost",
    "SendEngine", "control_send", "get_engine", "render_template",
    "spam_score", "warmup_plan", "deliverability_check", "parse_dsn",
    "BriefStore", "ContentBrief", "ContentScore", "BriefDraft",
    "control_brief", "control_content",
    "letter_grade", "brief_quality", "geo_score",
    "GuardrailStore", "Rule", "FiredAction", "control_guardrails", "evaluate",
    "execute_override", "parse_rule", "RULE_PRESETS", "add_preset", "preview",
    "record_metrics", "fatigue_signal", "format_naira", "parse_naira_kobo",
    "CompetitorStore", "CompetitorReport", "Post", "Pillar",
    "control_competitor", "aeo_compare",
    "analyze_pillars", "analyze_cadence", "analyze_engagement",
    "top_posts", "viral_posts", "best_times", "analyze_hashtags",
    "key_insights", "benchmark", "content_gaps", "to_briefs",
]
