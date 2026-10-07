"""Source-grounded research mode (NotebookLM pattern).

Answer ONLY from user-provided sources, with inline citations to exact
passages.  Refuse when the sources don't support an answer — never
confabulate.

The citation rule that matters: the model emits ``[Label]`` markers and
CODE maps them to deterministic numbers + a Sources section.  The model
never numbers citations itself (known failure point — models renumber,
skip, and hallucinate citation indices).

Flow::

    session = GroundedSession()
    session.add_file(path)          # or add_bytes(data, filename=...)
    answer = session.ask("...", llm_fn=...)
    print(answer.text)              # "... [1] ...\n\nSources:\n[1] title — snippet"
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..documents.index import DocumentIndex
from ..documents.parsers import parse_bytes, parse_path

_log = get_logger(__name__)

__all__ = ["GroundedSession", "GroundedAnswer", "GroundedError"]


class GroundedError(Exception):
    """Grounded session failure (ingest, retrieval, or answering)."""


@dataclass
class Source:
    doc_id: str
    title: str
    snippet: str
    score: float = 0.0


@dataclass
class GroundedAnswer:
    text: str                    # answer with [n] citations
    sources: list[Source] = field(default_factory=list)
    refused: bool = False
    query: str = ""

    def render(self) -> str:
        """Full answer text + Sources section."""
        if self.refused:
            return self.text
        lines = [self.text, "", "Sources:"]
        for i, s in enumerate(self.sources, 1):
            lines.append(f"[{i}] {s.title} — {s.snippet[:160]}")
        return "\n".join(lines)


#: the model cites with [Label]; code assigns the numbers
_LABEL_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9 _-]{0,40})\]")

_REFUSAL = ("I can't answer that from your documents — none of the "
            "sources you provided cover it. Add a relevant document and "
            "ask again.")

_ANSWER_PROMPT = """Answer the question using ONLY the sources below. Rules:
- Every factual claim must cite a source with [Label] (use the exact label shown).
- If the sources don't contain the answer, reply with exactly: CANNOT_ANSWER
- Do not use any knowledge outside these sources.
- Keep it concise.

QUESTION: {question}

SOURCES:
{sources}
"""


def _chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for retrieval."""
    text = (text or "").strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # break on a sentence boundary when convenient
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
        if start >= len(text) - overlap // 2:
            break
    return [c for c in chunks if c]


class GroundedSession:
    """A question-answering session bound to a fixed set of documents."""

    def __init__(self, *, chunk_size: int = 1200) -> None:
        self.index = DocumentIndex()
        self.chunk_size = chunk_size
        self.doc_ids: list[str] = []
        self.created_at = time.time()

    # ── ingest ───────────────────────────────────────────────────

    def add_file(self, path: str | Path) -> str:
        """Ingest a file.  Returns the doc id."""
        try:
            doc = parse_path(str(path))
        except Exception as exc:  # noqa: BLE001
            raise GroundedError(f"could not parse {path}: {exc}") from exc
        return self._index_doc(doc)

    def add_bytes(self, data: bytes, *, filename: str = "",
                  mime: str = "") -> str:
        """Ingest raw bytes.  Returns the doc id."""
        try:
            doc = parse_bytes(data, filename=filename, mime=mime)
        except Exception as exc:  # noqa: BLE001
            raise GroundedError(f"could not parse {filename!r}: {exc}") from exc
        return self._index_doc(doc)

    def add_text(self, text: str, *, title: str = "pasted text") -> str:
        """Ingest plain text directly."""
        return self.add_bytes(text.encode("utf-8"),
                              filename=f"{title}.txt", mime="text/plain")

    def _index_doc(self, doc: Any) -> str:
        from ..documents.model import Document, Section, full_text
        chunks = _chunk(full_text(doc), self.chunk_size)
        if not chunks:
            raise GroundedError(
                f"document {getattr(doc, 'id', '?')} has no indexable text")
        base_id = str(getattr(doc, "id", "") or f"doc-{len(self.doc_ids)}")
        title = str(getattr(doc, "title", "") or base_id)
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                id=f"{base_id}#c{i}",
                title=f"{title} (part {i + 1})",
                sections=[Section(heading="", text=chunk_text)],
            )
            self.index.add(chunk_doc)
        self.doc_ids.append(base_id)
        _log.info("grounded session: indexed %s (%d chunks)",
                  base_id, len(chunks))
        return base_id

    # ── ask ──────────────────────────────────────────────────────

    def ask(self, question: str,
            llm_fn: Callable[[str], str] | None = None,
            context: Any = None,
            top_k: int = 5) -> GroundedAnswer:
        """Answer ``question`` from the indexed documents only."""
        question = (question or "").strip()
        if not question:
            raise GroundedError("empty question")
        if not self.doc_ids:
            raise GroundedError("no documents ingested yet")
        hits = self.index.search(question, limit=top_k)
        if not hits:
            return GroundedAnswer(text=_REFUSAL, refused=True, query=question)

        sources = [
            Source(doc_id=h["doc_id"], title=h.get("title", h["doc_id"]),
                   snippet=h.get("snippet", "")[:300],
                   score=float(h.get("score", 0) or 0))
            for h in hits
        ]
        # label each source for the model: [S1], [S2], ...
        labeled = []
        for i, s in enumerate(sources, 1):
            labeled.append(f"[S{i}] {s.title}:\n{s.snippet}")
        prompt = _ANSWER_PROMPT.format(
            question=question,
            sources="\n\n".join(labeled),
        )
        try:
            raw = self._complete(prompt, llm_fn, context)
        except Exception as exc:  # noqa: BLE001
            raise GroundedError(f"answer generation failed: {exc}") from exc

        raw = (raw or "").strip()
        if "CANNOT_ANSWER" in raw.upper():
            return GroundedAnswer(text=_REFUSAL, refused=True, query=question)
        return self._number_citations(raw, sources, question)

    def _complete(self, prompt: str,
                  llm_fn: Callable[[str], str] | None,
                  context: Any) -> str:
        if llm_fn is not None:
            return llm_fn(prompt)
        router = getattr(context, "router", None) if context else None
        if router is None:
            raise GroundedError("no LLM available (pass llm_fn or context)")
        resp = router.complete(prompt)
        return resp.text if hasattr(resp, "text") else str(resp)

    def _number_citations(self, raw: str, sources: list[Source],
                          question: str) -> GroundedAnswer:
        """Map the model's [Label] markers to deterministic [n] numbers.

        Labels the model invented (not matching any source) are stripped —
        a citation to nothing is worse than no citation.
        """
        label_to_num: dict[str, int] = {}
        used: list[Source] = []

        def label_key(label: str) -> str:
            return label.strip().upper()

        # the model was told to use [S1], [S2], ... — map those
        def replace(m: re.Match) -> str:
            label = label_key(m.group(1))
            idx = None
            if label.startswith("S") and label[1:].isdigit():
                n = int(label[1:])
                if 1 <= n <= len(sources):
                    idx = n - 1
            if idx is None:
                # maybe it cited by a title fragment — fuzzy match
                for i, s in enumerate(sources):
                    if label in s.title.upper():
                        idx = i
                        break
            if idx is None:
                return ""  # invented citation — strip it
            key = f"S{idx + 1}"
            if key not in label_to_num:
                label_to_num[key] = len(used) + 1
                used.append(sources[idx])
            return f"[{label_to_num[key]}]"

        text = _LABEL_RE.sub(replace, raw)
        # claim-verification pass: if NO citation survived but the answer
        # makes factual claims, refuse rather than serve uncited claims
        if not used and len(text.split()) > 12:
            _log.warning("grounded answer had no valid citations — refusing")
            return GroundedAnswer(text=_REFUSAL, refused=True, query=question)
        return GroundedAnswer(text=text, sources=used, query=question)
