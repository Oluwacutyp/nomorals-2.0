"""Universal federated search (L5): one query across everything Devon knows.

Fan-out over per-subsystem adapters (see ``sources.py`` for the reuse
survey), per-source score normalization, content-hash dedupe, and
deterministic ranking. A federated layer — not a new index.

The universal upgrade adds the live web: six backends behind the same
adapter interface (``web.py`` — SearXNG keyless metasearch first, keyed
free-tier APIs after), BM25 snippet re-ranking (``rerank.py``), and
reciprocal rank fusion across sources (``model.reciprocal_rank_fusion``,
wired as ``federated_search(fusion="rrf")``).
"""

from __future__ import annotations

from .base import SourceAdapter
from .errors import (
    InvalidDateError,
    SearchError,
    UnknownSourceError,
    UnknownTypeError,
)
from .federated import federated_search, list_sources, parse_date
from .model import (
    SearchResponse,
    SearchResult,
    reciprocal_rank_fusion,
)
from .rerank import bm25_rerank, bm25_scores, tokenize
from .sources import (
    SOURCE_SPECS,
    build_adapters,
    default_doc_index_path,
    valid_source_names,
    valid_types,
)
from .web import (
    WEB_SPECS,
    BraveWebSource,
    DdgsWebSource,
    ExaWebSource,
    SerperWebSource,
    SearXNGWebSource,
    TavilyWebSource,
    WebBackendError,
    WebSearchSource,
    web_backends_configured,
    web_source_names,
)

__all__ = [
    "SearchError",
    "UnknownSourceError",
    "UnknownTypeError",
    "InvalidDateError",
    "WebBackendError",
    "SearchResult",
    "SearchResponse",
    "SourceAdapter",
    "WebSearchSource",
    "SearXNGWebSource",
    "DdgsWebSource",
    "TavilyWebSource",
    "SerperWebSource",
    "ExaWebSource",
    "BraveWebSource",
    "WEB_SPECS",
    "web_source_names",
    "web_backends_configured",
    "SOURCE_SPECS",
    "federated_search",
    "list_sources",
    "parse_date",
    "build_adapters",
    "default_doc_index_path",
    "valid_source_names",
    "valid_types",
    "reciprocal_rank_fusion",
    "bm25_rerank",
    "bm25_scores",
    "tokenize",
]
