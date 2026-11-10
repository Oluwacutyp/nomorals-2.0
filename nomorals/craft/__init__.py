"""The craft cooperative foundation (L5) — docs 22+23, Stage A.

Four pieces, each a projection or schema over organs that already
run — no parallel systems:

* :mod:`nomorals.craft.graph` — the CraftGraph: a read-computed
  projection over the goals registry, the taxonomy seed, the
  specialist manifests, delegation contracts, run tickets,
  artifacts and the skill filebook;
* :mod:`nomorals.craft.receipts` — the universal receipt schema
  (QualityReceipt / HandoffReceipt) with read-side adapters over
  the five shipped receipt stores, grades derived through
  ``core/proof_grade``;
* :mod:`nomorals.craft.levels` — capability levels L0–L5 as
  derived labels mapped onto the one G0–G6 proof ladder;
* :mod:`nomorals.craft.seed` — the taxonomy seed data the graph's
  occupational skeleton resolves to.

The pack form lives with the filebook it extends
(:mod:`nomorals.skills.packs`); role contracts live with the role
machinery they complete (:mod:`nomorals.agents.role_contracts`).
"""

from __future__ import annotations

from .levels import (
    LEVEL_GRADE_MAPPINGS,
    LEVEL_INFO,
    LEVELS,
    LevelError,
    cap_claim,
    grade_for_level,
    is_handoff_level,
    parse_level,
)
from .receipts import (
    HandoffReceipt,
    QualityReceipt,
    adapt_bill_receipt,
    adapt_case_receipt,
    adapt_content_receipt,
    adapt_delivery_receipt,
    adapt_police_receipt,
    handoff_for,
    render_receipt_chain,
)
from .seed import SEED_DOMAINS, seed_domain, seed_occupation

__all__ = [
    # levels
    "LEVELS", "LEVEL_INFO", "LEVEL_GRADE_MAPPINGS", "LevelError",
    "parse_level", "grade_for_level", "is_handoff_level",
    "cap_claim",
    # receipts
    "QualityReceipt", "HandoffReceipt", "handoff_for",
    "adapt_content_receipt", "adapt_case_receipt",
    "adapt_police_receipt", "adapt_delivery_receipt",
    "adapt_bill_receipt", "render_receipt_chain",
    # seed
    "SEED_DOMAINS", "seed_domain", "seed_occupation",
    # graph (lazy — it composes the record organs)
    "CraftGraph", "GraphNode", "GraphEdge", "project_graph",
    "NODE_KINDS",
]


def __getattr__(name: str):  # PEP 562 lazy graph exports
    if name in ("CraftGraph", "GraphNode", "GraphEdge",
                "project_graph", "NODE_KINDS"):
        from . import graph
        return getattr(graph, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
