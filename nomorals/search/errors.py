"""Federated-search error types. Fail fast, never silent."""

from __future__ import annotations


class SearchError(Exception):
    """Base error for the federated search layer."""


class UnknownSourceError(SearchError):
    """Raised when a requested source name is not a known search source."""

    def __init__(self, name: str, valid: list[str]) -> None:
        self.name = name
        self.valid = list(valid)
        super().__init__(
            f"unknown search source: {name!r}. valid sources: {', '.join(valid)}"
        )


class UnknownTypeError(SearchError):
    """Raised when a --type filter names no known result type."""

    def __init__(self, name: str, valid: list[str]) -> None:
        self.name = name
        self.valid = list(valid)
        super().__init__(
            f"unknown result type: {name!r}. valid types: {', '.join(valid)}"
        )


class InvalidDateError(SearchError):
    """Raised when a --since/--before value cannot be parsed as a date."""
