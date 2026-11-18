"""Software sources: external-software catalogues Devon can query.

First organ: :mod:`nomorals.sources.software` — the AlternativeTo
software-alternatives catalogue (52-websites site #4).

Second organ: :mod:`nomorals.sources.watchtower` — the outward
outage watch (52-websites site #45, Downdetector).
"""

from __future__ import annotations

from .software import (
    AlternativeEdge,
    AlternativeToSource,
    SoftwareCatalog,
    SoftwareRecord,
)
from .watchtower import (
    Incident,
    ProbeBaseline,
    ProbeSample,
    Verdict,
    WatchTarget,
    WatchtowerStore,
)

__all__ = [
    "AlternativeEdge",
    "AlternativeToSource",
    "SoftwareCatalog",
    "SoftwareRecord",
    "Incident",
    "ProbeBaseline",
    "ProbeSample",
    "Verdict",
    "WatchTarget",
    "WatchtowerStore",
]
