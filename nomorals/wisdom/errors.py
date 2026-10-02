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


class PracticeError(WisdomError):
    """A guided practice session could not start or run."""


class HistoryError(WisdomError):
    """The timeline dataset is invalid or a query is malformed."""
