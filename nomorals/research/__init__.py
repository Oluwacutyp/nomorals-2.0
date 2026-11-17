"""Autonomous background research: scheduled web research, worth-interrupting
judgment, and delivery to the owner's DMs.

The pipeline is: ResearchJob -> run_job() -> assess_worth() -> deliver().
The scheduler in scheduler.py ticks the jobs. The core rule is conservative:
better to miss something than to spam the owner.
"""

from .citations import Conflict, corroboration_map, find_conflicts
from .costs import COST_TABLE
from .extraction import (
    Cell,
    EvidenceReviewStore,
    EvidenceTable,
    ExtractionColumn,
    ExtractionSchema,
    Paper,
    PaperSet,
    answer_question,
    default_schema,
    export_csv,
    export_ris,
    extract_cells,
    format_evidence_reply,
    open_review_store,
    retrieve_papers,
)
from .pipeline import (
    Assessment,
    DeepReport,
    Refinement,
    ResearchBudget,
    ResearchContext,
    ResearchFinding,
    ResearchJob,
    assess_worth,
    clarify,
    decompose,
    deliver,
    deliver_digest,
    ensure_schema,
    execute_job,
    research_deep,
    run_job,
    synthesize,
    synthesize_with_stats,
)
from .scheduler import ResearchScheduler, default_jobs
from .screening import InclusionCriteria, ScreenDecision, screen_papers

__all__ = [
    "Assessment",
    "COST_TABLE",
    "Cell",
    "Conflict",
    "DeepReport",
    "EvidenceReviewStore",
    "EvidenceTable",
    "ExtractionColumn",
    "ExtractionSchema",
    "InclusionCriteria",
    "Paper",
    "PaperSet",
    "Refinement",
    "ResearchBudget",
    "ResearchContext",
    "ResearchFinding",
    "ResearchJob",
    "ResearchScheduler",
    "ScreenDecision",
    "answer_question",
    "assess_worth",
    "clarify",
    "corroboration_map",
    "decompose",
    "default_jobs",
    "default_schema",
    "deliver",
    "deliver_digest",
    "ensure_schema",
    "execute_job",
    "export_csv",
    "export_ris",
    "extract_cells",
    "find_conflicts",
    "format_evidence_reply",
    "open_review_store",
    "research_deep",
    "retrieve_papers",
    "run_job",
    "screen_papers",
    "synthesize",
    "synthesize_with_stats",
]
