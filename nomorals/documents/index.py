"""Full-text search over parsed documents: a small inverted index.

Tokenizes on lowercased word tokens, scores with raw term frequency, and
returns a ~120-character snippet around the first query-term hit.  Persists
as JSON via :meth:`DocumentIndex.save` / :meth:`DocumentIndex.load`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import DocumentError
from .model import Document, Section, full_text

__all__ = ["DocumentIndex"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SNIPPET_RADIUS_BEFORE = 40
_SNIPPET_RADIUS_AFTER = 80


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= 2]


class DocumentIndex:
    """In-memory inverted index over :class:`Document` full text."""

    def __init__(self) -> None:
        # doc_id -> {"title": str, "text": str}
        self._docs: dict[str, dict[str, str]] = {}
        # term -> {doc_id: term frequency}
        self._index: dict[str, dict[str, int]] = {}

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc: Document) -> None:
        """Index ``doc``.  Re-adding an existing id replaces the entry."""
        if not doc.id:
            raise DocumentError("cannot index a document without an id")
        text = full_text(doc)
        if not text.strip():
            raise DocumentError(f"document {doc.id} has no indexable text")
        self.remove(doc.id)
        self._docs[doc.id] = {"title": doc.title, "text": text}
        counts: dict[str, int] = {}
        for token in _tokenize(f"{doc.title} {text}"):
            counts[token] = counts.get(token, 0) + 1
        for token, freq in counts.items():
            self._index.setdefault(token, {})[doc.id] = freq

    def remove(self, doc_id: str) -> bool:
        """Drop ``doc_id`` from the index.  Returns True when it was present."""
        if doc_id not in self._docs:
            return False
        del self._docs[doc_id]
        for postings in self._index.values():
            postings.pop(doc_id, None)
        # prune terms that no longer point anywhere
        self._index = {t: p for t, p in self._index.items() if p}
        return True

    def _snippet(self, text: str, terms: list[str]) -> str:
        lowered = text.lower()
        hit = -1
        for term in terms:
            pos = lowered.find(term)
            if pos >= 0 and (hit < 0 or pos < hit):
                hit = pos
        if hit < 0:
            window = text[:_SNIPPET_RADIUS_BEFORE + _SNIPPET_RADIUS_AFTER]
            return window + ("…" if len(text) > len(window) else "")
        start = max(0, hit - _SNIPPET_RADIUS_BEFORE)
        end = min(len(text), hit + _SNIPPET_RADIUS_AFTER)
        snippet = text[start:end]
        if start > 0:
            snippet = "…" + snippet.lstrip()
        if end < len(text):
            snippet = snippet.rstrip() + "…"
        return " ".join(snippet.split())

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search the index; each hit is {doc_id, title, score, snippet}.

        Score is the sum of query-term frequencies in the document (raw TF).
        Ties break on doc id so ordering is deterministic.
        """
        terms = _tokenize(query or "")
        if not terms:
            raise DocumentError("search query has no indexable terms")
        if limit <= 0:
            raise DocumentError(f"limit must be positive, got {limit}")
        scores: dict[str, int] = {}
        for term in terms:
            for doc_id, freq in self._index.get(term, {}).items():
                scores[doc_id] = scores.get(doc_id, 0) + freq
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        results = []
        for doc_id, score in ranked[:limit]:
            record = self._docs[doc_id]
            results.append({
                "doc_id": doc_id,
                "title": record["title"],
                "score": score,
                "snippet": self._snippet(record["text"], terms),
            })
        return results

    def save(self, path: str | Path) -> Path:
        """Persist the index as JSON (documents' id/title/text; index rebuilt)."""
        file_path = Path(path)
        payload = {
            "version": 1,
            "docs": [
                {"id": doc_id, "title": rec["title"], "text": rec["text"]}
                for doc_id, rec in self._docs.items()
            ],
        }
        try:
            file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        except OSError as exc:
            raise DocumentError(f"cannot write index to {file_path}: {exc}") from exc
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> DocumentIndex:
        """Load an index saved with :meth:`save`."""
        file_path = Path(path)
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DocumentError(f"cannot load index from {file_path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise DocumentError(f"not a document index file: {file_path}")
        index = cls()
        for entry in payload.get("docs", []):
            doc = Document(id=str(entry.get("id", "")),
                           title=str(entry.get("title", "")))
            # Rebuild via add() so postings stay consistent; the stored text
            # is injected as a single section because only text was persisted.
            doc.sections = [Section(level=1, heading="", text=str(entry.get("text", "")))]
            index.add(doc)
        return index
