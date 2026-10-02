"""WisdomKeeper — the single entry point for the wisdom organ.

Thin façade: it owns one CanonCorpus, one HistoryEngine, one
PracticeGuide, and routes. It does not implement.
"""
from __future__ import annotations

from typing import Any

from .corpus import Answer, CanonCorpus


class WisdomKeeper:
    """Study companion and practice timer for the esoteric corpus."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.corpus = CanonCorpus(context)
        self._history: Any = None
        self._practice: Any = None

    @property
    def history(self) -> Any:
        if self._history is None:
            from .history import HistoryEngine
            self._history = HistoryEngine(self.context)
        return self._history

    @property
    def practice(self) -> Any:
        if self._practice is None:
            from .practice import PracticeGuide
            self._practice = PracticeGuide(self.context)
        return self._practice

    # ── ask ───────────────────────────────────────────────────────────
    def ask(self, query: str, *, top: int = 5,
            tradition: str = "") -> Answer:
        """Ask the corpus. Every returned passage carries provenance."""
        return self.corpus.ask(query, top=top, tradition=tradition)

    def status(self) -> dict[str, Any]:
        """Corpus + organ status for `nm wisdom status`."""
        return {
            "corpus": self.corpus.status(),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.status()
