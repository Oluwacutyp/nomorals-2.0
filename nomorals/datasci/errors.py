"""Errors for the data-science workspace."""

from __future__ import annotations

__all__ = [
    "DataSciError",
    "DatasetNotFound",
    "DatasetExists",
    "LoadError",
    "QueryError",
    "PlotError",
]


class DataSciError(Exception):
    """Base error for the data-science workspace."""


class DatasetNotFound(DataSciError):
    """No dataset with that name is loaded."""


class DatasetExists(DataSciError):
    """A dataset with that name is already loaded (use drop first)."""


class LoadError(DataSciError):
    """A file could not be loaded as a dataset."""


class QueryError(DataSciError):
    """A pandas query/expression failed."""


class PlotError(DataSciError):
    """A plot could not be generated."""
