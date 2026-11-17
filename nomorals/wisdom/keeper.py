"""WisdomKeeper — the single entry point for the wisdom organ.

Thin façade: it owns one CanonCorpus, one HistoryEngine, one
PracticeGuide, and routes. It does not implement.
"""
from __future__ import annotations

from typing import Any

from .corpus import Answer, CanonCorpus, ProvenanceHit


class WisdomKeeper:
    """Study companion and practice timer for the esoteric corpus."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.corpus = CanonCorpus(context)
        self._history: Any = None
        self._practice: Any = None
        self._oral: Any = None
        self._practice_memory: Any = None

    @property
    def history(self) -> Any:
        if self._history is None:
            from .history import HistoryEngine
            self._history = HistoryEngine(self.context)
        return self._history

    @property
    def practice(self) -> Any:
        if self._practice is None:
            from .practice import PracticeGuide
            self._practice = PracticeGuide(self.context)
        return self._practice

    @property
    def practice_memory(self) -> Any:
        """Durable recall memory (SM-2, source-grounded) for practice.

        Lives at ``<wisdom root>/practice_memory.json`` — the same
        durable root as the practice log and the oral store. Lazily
        built like history/practice/oral.
        """
        if self._practice_memory is None:
            from .practice_memory import PracticeMemory
            self._practice_memory = PracticeMemory(
                self.corpus._wisdom_root() / "practice_memory.json")
        return self._practice_memory

    def disagreement_topics(self) -> list[str]:
        """Topics the disagreement map can compare (never averaged)."""
        from .disagreements import DisagreementMap
        return DisagreementMap().topics()

    def compare_positions(self, topic: str) -> Any:
        """Cross-tradition positions on ``topic``, mapped not averaged.

        Returns a :class:`Disagreement` (positions side by side +
        convergences/divergences), grounded in corpus passages when
        the corpus can supply them. Unknown topic raises ValueError
        naming the known topics.
        """
        from .disagreements import DisagreementMap
        return DisagreementMap().compare_with_corpus(topic,
                                                     corpus=self.corpus)

    def lineage(self, query: str) -> list[dict[str, Any]]:
        """Lineage for ``query``: graph nodes matching the name plus
        their edges + neighbours, built from the real timeline.

        The graph is derived from the timeline events (52 events):
        ``tradition_sequence`` edges between consecutive events of the
        same tradition. Every edge carries its source.
        """
        from .lineage import LineageGraph
        graph = LineageGraph.from_timeline(
            self.history.timeline())
        return graph.lineage_for(query)

    def lineage_nodes(self) -> list[dict[str, Any]]:
        """Every node of the timeline-derived lineage graph."""
        from .lineage import LineageGraph
        graph = LineageGraph.from_timeline(
            self.history.timeline())
        return [n.to_dict() for n in graph.nodes.values()]

    @property
    def oral(self) -> Any:
        """The oral-tradition ingest pipeline (draft → owner verifies
        → ingest), rooted at the wisdom store next to the corpus
        manifest. Lazily built like history/practice."""
        if self._oral is None:
            from .oral import OralIngest
            self._oral = OralIngest(self.corpus._wisdom_root() / "oral")
        return self._oral

    # ── ask ───────────────────────────────────────────────────────────
    def ask(self, query: str, *, top: int = 5,
            tradition: str = "", mode: str = "keyword",
            include_oral: bool = True, **kwargs: Any) -> Answer:
        """Ask the corpus. Every returned passage carries provenance.

        ``mode``: ``"keyword"`` (FTS5/BM25), ``"semantic"`` (vector KNN),
        or ``"hybrid"`` (reciprocal-rank fusion of both). Extra keyword
        arguments (``expand``, ``min_score``, ``diversify``,
        ``mmr_lambda``, ``weights``) forward to
        :meth:`CanonCorpus.ask`.

        Ingested oral-tradition records are searched too (keyword match
        over their turns) and fill any slots the corpus leaves under
        ``top`` — an oral passage never displaces a corpus passage.
        ``include_oral=False`` opts out.
        """
        ans = self.corpus.ask(query, top=top, tradition=tradition,
                              mode=mode, **kwargs)
        if include_oral:
            found = self._oral_hits(query, limit=top - len(ans.passages))
            if found:
                for f in found:
                    lang = f" ({f['language']})" if f["language"] else ""
                    ans.passages.append(ProvenanceHit(
                        work="oral tradition",
                        translator=f.get("contributor", ""),
                        section=f"draft {f['draft_id']} · {f['speaker']}{lang}",
                        url="", snippet=f["text"],
                        score=float(f["score"]),
                        canon_status="oral-tradition"))
                ans.confidence = CanonCorpus._confidence(ans.passages)
                ids = sorted({f["draft_id"] for f in found})
                ans.synthesis += (
                    "\n\nOral-tradition passage(s) above — read the full "
                    "draft with /wisdom oral read " + " / ".join(ids) + ".")
        return ans

    def _oral_hits(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        """Ingested oral turns matching the query (keyword, durable).

        Never breaks ask: a missing/unreadable oral store yields no
        hits, not an error.
        """
        if limit <= 0:
            return []
        try:
            return self.oral.search(query, top=limit)
        except Exception:  # noqa: BLE001 - oral store is best-effort here
            return []

    def read_oral(self, draft_id: str) -> dict[str, Any]:
        """The full ingested oral record for one draft (or OralError)."""
        return self.oral.read_ingested(draft_id)

    def oral_status(self) -> dict[str, Any]:
        """Durable oral counts for status surfaces."""
        try:
            ingested = self.oral.list_ingested()
        except Exception:  # noqa: BLE001 - store missing reads as empty
            ingested = []
        return {"ingested": len(ingested),
                "store": str(self.oral.store_dir)}

    def build_semantic_index(self, backend_name: str = "", *,
                             rebuild: bool = False,
                             progress: Any = None) -> dict[str, Any]:
        """Embed every corpus passage and build the vector index."""
        return self.corpus.build_semantic_index(
            backend_name, rebuild=rebuild, progress=progress)

    def semantic_index_status(self) -> dict[str, Any]:
        """Backend/engine/vector counts for the semantic index."""
        return self.corpus.semantic_index_status()

    def status(self) -> dict[str, Any]:
        """Corpus + organ status for `nm wisdom status`."""
        from .completeness import wisdom_completeness_report
        return {
            "corpus": self.corpus.status(),
            "practice": self.practice.stats(),
            "semantic": self.corpus.semantic_index_status(),
            "oral": self.oral_status(),
            "completeness": wisdom_completeness_report(),
        }

    # ── readability: read / traditions / seeds ────────────────────────
    def read_work(self, query: str, *, chapter: Any = None, page: int = 1,
                  page_chars: int = 3500) -> dict[str, Any]:
        """Read the actual full text of an ingested work, paginated."""
        return self.corpus.read_work(query, chapter=chapter, page=page,
                                     page_chars=page_chars)

    def traditions(self) -> list[dict[str, Any]]:
        """Traditions with labels, explanations, and their works."""
        return self.corpus.traditions_detail()

    def seeds(self) -> list[dict[str, Any]]:
        """Every seed source + which works came from it."""
        return self.corpus.seeds()

    # -- autonomy (WisdomOrgan, lazy) ---------------------------------
    def _organ(self) -> Any:
        from .autonomy import WisdomOrgan
        return WisdomOrgan(self.context)

    def synthesize(self, query: str, top: int = 5) -> dict[str, Any]:
        """Cross-tradition synthesis for one question (brain entry)."""
        return self._organ().synthesize(query, top=top)

    def digest(self, slug: str) -> dict[str, Any] | None:
        """The extractive digest for a work (builds it when missing)."""
        organ = self._organ()
        got = organ.get_digest(slug)
        return got if got is not None else organ.digest_work(slug)

    def gaps(self) -> dict[str, Any]:
        """Where the corpus is thin + live hunt candidates."""
        return self._organ().gap_analysis()

    def answer_card(self, query: str, **ask_kwargs: Any) -> str:
        """Ask the corpus and render a chat-ready answer card."""
        return self.ask(query, **ask_kwargs).render("chat")

    def to_dict(self) -> dict[str, Any]:
        return self.status()
