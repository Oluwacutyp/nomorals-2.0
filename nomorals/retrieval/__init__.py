"""One retrieval spine (L3) — public package surface."""

from .spine import (Hit, HashEmbedder, RetrievalSpine, adapt_corpus,
                    adapt_library, cosine_sim, mmr_rerank, rrf_fuse,
                    tokenize)

__all__ = [
    "Hit", "HashEmbedder", "RetrievalSpine", "adapt_corpus",
    "adapt_library", "cosine_sim", "mmr_rerank", "rrf_fuse", "tokenize",
]
