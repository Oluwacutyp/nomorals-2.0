"""WisdomKeeper — the esoteric study companion and practice organ (L5)."""
from __future__ import annotations

from .corpus import Answer, CanonCorpus, ManifestEntry, ProvenanceHit
from .embeddings import (BACKENDS, EmbeddingBackend, FastEmbedBackend,
                         HashEmbedBackend, OllamaBackend,
                         SentenceTransformersBackend, auto_backend,
                         available_backends, embed_texts)
from .errors import (CorpusError, EmbeddingError, HistoryError, IngestError,
                     PracticeError, VectorStoreError, WisdomError)
from .hybrid import fuse_hits, reciprocal_rank_fusion
from .vectorstore import VectorIndex, open_index
from .ingestor import ArchiveIngestor
from .history import HistoryEngine
from .keeper import WisdomKeeper
from .practice import SAFETY_TEXT, PracticeGuide
from .chat_session import ChatPracticeSession, WisdomChatManager

__all__ = [
    "Answer",
    "ArchiveIngestor",
    "BACKENDS",
    "CanonCorpus",
    "CorpusError",
    "EmbeddingBackend",
    "EmbeddingError",
    "FastEmbedBackend",
    "HashEmbedBackend",
    "HistoryError",
    "HistoryEngine",
    "IngestError",
    "ManifestEntry",
    "OllamaBackend",
    "PracticeError",
    "ProvenanceHit",
    "SAFETY_TEXT",
    "SentenceTransformersBackend",
    "VectorIndex",
    "VectorStoreError",
    "ChatPracticeSession",
    "PracticeGuide",
    "WisdomChatManager",
    "WisdomError",
    "WisdomKeeper",
    "auto_backend",
    "available_backends",
    "embed_texts",
    "fuse_hits",
    "open_index",
    "reciprocal_rank_fusion",
]
