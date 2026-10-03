"""Universal federated search (L5): one query across everything Devon knows.

Fan-out over per-subsystem adapters (see ``sources.py`` for the reuse
survey), per-source score normalization, content-hash dedupe, and
deterministic ranking. A federated layer — not a new index.
"""

from __future__ import annotations

from .errors import (
    InvalidDateError,
    SearchError,
    UnknownSourceError,
    UnknownTypeError,
)
from .federated import federated_search, list_sources, parse_date
from .model import SearchResponse, SearchResult
from .sources import (
    SOURCE_SPECS,
    SourceAdapter,
    build_adapters,
    default_doc_index_path,
    valid_source_names,
    valid_types,
)

__all__ = [
    "SearchError",
    "UnknownSourceError",
    "UnknownTypeError",
    "InvalidDateError",
    "SearchResult",
    "SearchResponse",
    "SourceAdapter",
    "SOURCE_SPECS",
    "federated_search",
    "list_sources",
    "parse_date",
    "build_adapters",
    "default_doc_index_path",
    "valid_source_names",
    "valid_types",
]
