"""Data-science workspace: in-agent pandas analysis with provenance."""

from __future__ import annotations

from .errors import (
    DataSciError,
    DatasetExists,
    DatasetNotFound,
    LoadError,
    PlotError,
    QueryError,
)
from .plots import PLOT_KINDS, render_plot
from .workspace import SUPPORTED_SUFFIXES, Dataset, DataWorkspace

__all__ = [
    "DataSciError",
    "DatasetExists",
    "DatasetNotFound",
    "LoadError",
    "PlotError",
    "QueryError",
    "PLOT_KINDS",
    "SUPPORTED_SUFFIXES",
    "Dataset",
    "DataWorkspace",
    "render_plot",
]
