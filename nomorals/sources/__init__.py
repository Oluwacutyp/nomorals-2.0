"""Software sources: external-software catalogues Devon can query.

First organ: :mod:`nomorals.sources.software` — the AlternativeTo
software-alternatives catalogue (52-websites site #4).

Second organ: :mod:`nomorals.sources.watchtower` — the outward
outage watch (52-websites site #45, Downdetector).

Third organ: :mod:`nomorals.sources.learning` — the free-learning
collection (52-websites site #12, OpenCulture).
"""

from __future__ import annotations

from .learning import (
    CollectionEntry,
    LearningCollection,
    LiveLearningSearch,
    check_entry,
    format_learning_reply,
    learn_free,
    refresh_health,
)
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
    "CollectionEntry",
    "LearningCollection",
    "LiveLearningSearch",
    "check_entry",
    "format_learning_reply",
    "learn_free",
    "refresh_health",
    "Incident",
    "ProbeBaseline",
    "ProbeSample",
    "Verdict",
    "WatchTarget",
    "WatchtowerStore",
]
