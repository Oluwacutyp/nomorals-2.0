"""Data-science workspace: in-agent pandas analysis with provenance."""

from __future__ import annotations

from .errors import (
    DataSciError,
    DatasetExists,
    DatasetNotFound,
    ExportError,
    LoadError,
    PlotError,
    QueryError,
)
from .plots import PLOT_KINDS, THEMES, render_plot
from .workspace import SUPPORTED_SUFFIXES, EXPORT_FORMATS, Dataset, DataWorkspace

__all__ = [
    "DataSciError",
    "DatasetExists",
    "DatasetNotFound",
    "ExportError",
    "LoadError",
    "PlotError",
    "QueryError",
    "PLOT_KINDS",
    "THEMES",
    "SUPPORTED_SUFFIXES",
    "EXPORT_FORMATS",
    "Dataset",
    "DataWorkspace",
    "render_plot",
]
