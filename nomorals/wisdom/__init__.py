"""WisdomKeeper — the esoteric study companion and practice organ (L5)."""
from __future__ import annotations

from .corpus import Answer, CanonCorpus, ManifestEntry, ProvenanceHit
from .embeddings import (BACKENDS, EmbeddingBackend, FastEmbedBackend,
                         HashEmbedBackend, OllamaBackend,
                         SentenceTransformersBackend, auto_backend,
                         available_backends, embed_texts, truncate_dim)
from .errors import (CorpusError, EmbeddingError, HistoryError, IngestError,
                     PracticeError, VectorStoreError, WisdomError)
from .hybrid import fuse_hits, reciprocal_rank_fusion, weighted_rrf
from .vectorstore import VectorIndex, open_index
from .ingestor import ArchiveIngestor
from .autonomy import WisdomOrgan, mmr_order, textrank
from .history import ERAS, HistoryEngine, era_of
from .reading import (PAGE_CHARS, SEEDS, TRADITION_INFO, dehyphenate,
                      normalize_tradition, paginate_text, parse_read_args,
                      parse_timeline_args, sentence_passage,
                      tradition_description, tradition_label,
                      trim_to_sentences)
from .keeper import WisdomKeeper
from .practice import SAFETY_TEXT, PracticeGuide
from .chat_session import ChatPracticeSession, WisdomChatManager

__all__ = [
    "Answer",
    "ArchiveIngestor",
    "BACKENDS",
    "ERAS",
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
    "WisdomOrgan",
    "auto_backend",
    "available_backends",
    "dehyphenate",
    "embed_texts",
    "era_of",
    "fuse_hits",
    "mmr_order",
    "normalize_tradition",
    "open_index",
    "paginate_text",
    "parse_read_args",
    "parse_timeline_args",
    "reciprocal_rank_fusion",
    "sentence_passage",
    "textrank",
    "tradition_description",
    "tradition_label",
    "trim_to_sentences",
    "truncate_dim",
    "weighted_rrf",
    "PAGE_CHARS",
    "SEEDS",
    "TRADITION_INFO",
]
