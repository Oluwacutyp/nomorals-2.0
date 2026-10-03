"""WisdomKeeper error hierarchy.

Fail fast across the organ's boundary: every failure carries what was
being attempted and why it failed. The CLI translates these to exit codes.
"""
from __future__ import annotations


class WisdomError(Exception):
    """Base for all WisdomKeeper failures."""


class CorpusError(WisdomError):
    """The corpus manifest or library substrate is unusable."""


class IngestError(WisdomError):
    """Fetching, parsing, or indexing a text failed."""

    def __init__(self, url: str = "", reason: str = "") -> None:
        self.url = url
        self.reason = reason
        super().__init__(url, reason)

    def __str__(self) -> str:
        # str(Exception) on two args renders the tuple repr
        # "('url', 'reason')" — unreadable in CLI output. Render the
        # failure as a sentence instead.
        if self.url and self.reason:
            return f"{self.url}: {self.reason}"
        return self.reason or self.url or super().__str__()


class PracticeError(WisdomError):
    """A guided practice session could not start or run."""


class HistoryError(WisdomError):
    """The timeline dataset is invalid or a query is malformed."""


class EmbeddingError(WisdomError):
    """An embedding backend is unavailable or failed to encode text."""


class VectorStoreError(WisdomError):
    """The semantic vector index is corrupt, stale, or unusable."""
