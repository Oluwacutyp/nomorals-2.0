"""Embedding-based error->fix recall (audit Phase C).

Wave 67 matched the current sandbox error against distilled fix skills with
raw string matching — one renamed variable and the lesson missed.  This
module keeps a small embedding index over the ``_distill_session`` outputs:
the error signature/trail of each self-corrected session is embedded (via
``nomorals.memory.embeddings.Embedder`` — deterministic feature hashing by
default, a semantic provider when the router offers one) and recall is a
cosine-similarity search.  A renamed-variable variant of a seen error still
scores high because almost all stem/bigram features overlap.

The index persists as JSON (best-effort) so lessons survive restarts; every
failure path degrades to "no recall", never to a broken build.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".nomorals" / "error_recall.json"

#: Minimum cosine similarity for a recall hit.  The hashing embedder is
#: L2-normalized, so this is a true cosine.  0.30 is deliberately generous:
#: recall is advisory (it only adds context to a fix prompt), so a near
#: miss costs nothing while a hard miss costs a re-derived fix.
MIN_SCORE = 0.30

_TOP_K = 3


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class ErrorRecallIndex:
    """Small persistent embedding index: error text -> skill id."""

    def __init__(self, embedder: Any | None = None,
                 store_path: str | Path | None = None) -> None:
        if embedder is None:
            from ..memory.embeddings import Embedder

            embedder = Embedder()  # hashing provider: deterministic, no model
        self._embedder = embedder
        self._path = Path(store_path) if store_path else _DEFAULT_PATH
        # skill_id -> {"error": str, "vector": [...], "indexed_at": float}
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    # ── write ────────────────────────────────────────────────────────────
    def index(self, skill_id: str, error_text: str) -> None:
        """Embed an error signature/trail and link it to a skill id."""
        text = " ".join(str(error_text or "").split())
        if not text or not skill_id:
            return
        try:
            vector = self._embedder.embed(text)
        except Exception as exc:  # noqa: BLE001 — recall is best-effort
            _log.debug("error-recall embed failed: %s", exc)
            return
        self._records[skill_id] = {
            "error": text[:2000],
            "vector": [float(v) for v in vector],
            "indexed_at": time.time(),
        }
        self._save()

    def drop(self, skill_id: str) -> None:
        if skill_id in self._records:
            del self._records[skill_id]
            self._save()

    # ── read ─────────────────────────────────────────────────────────────
    def recall(self, error_text: str, top_k: int = _TOP_K,
               min_score: float = MIN_SCORE) -> list[dict[str, Any]]:
        """Top-k ``{skill_id, score, error}`` hits by cosine similarity."""
        text = " ".join(str(error_text or "").split())
        if not text or not self._records:
            return []
        try:
            query = self._embedder.embed(text)
        except Exception as exc:  # noqa: BLE001
            _log.debug("error-recall query embed failed: %s", exc)
            return []
        scored = []
        for skill_id, rec in self._records.items():
            score = _cosine(query, rec["vector"])
            if score >= min_score:
                scored.append({"skill_id": skill_id, "score": round(score, 3),
                               "error": rec["error"]})
        scored.sort(key=lambda h: -h["score"])
        return scored[:max(1, top_k)]

    def __len__(self) -> int:
        return len(self._records)

    # ── persistence (best-effort) ────────────────────────────────────────
    def _load(self) -> None:
        try:
            if not self._path.is_file():
                return
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._records = {
                    str(k): v for k, v in data.items()
                    if isinstance(v, dict) and isinstance(v.get("vector"), list)
                }
        except Exception as exc:  # noqa: BLE001
            _log.debug("error-recall load failed: %s", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._records), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            _log.debug("error-recall save failed: %s", exc)
