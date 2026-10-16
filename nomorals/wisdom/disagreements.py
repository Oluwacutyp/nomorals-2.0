"""Map disagreements, never average them.

Wisdom gap fix: cross-tradition comparison used to flatten positions
into one synthesized view. This module keeps each tradition's position
intact — claim + basis + sources, per tradition — and reports
convergences and divergences *between* positions without resolving
them into an average.

Rules:
- Every position needs a ``basis`` (the reasoning/textual ground) and
  at least one ``source`` pointer — a position without a basis is
  rejected, not stored.
- No position is ever merged into another: ``compare`` returns the
  positions side by side plus explicit convergence/divergence notes.
- Corpus grounding is opt-in: ``compare_with_corpus`` attaches
  retrieved passages per tradition (each with its own provenance) —
  the map never invents quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Position", "Disagreement", "DisagreementMap",
           "BUILTIN_TOPICS"]


@dataclass
class Position:
    """One tradition's position on a topic: claim + basis + sources."""

    tradition: str
    claim: str
    basis: str  # the reasoning / textual ground for the claim
    sources: list[str] = field(default_factory=list)
    provenance_note: str = ""

    def validate(self) -> None:
        if not self.tradition:
            raise ValueError("position needs a tradition")
        if not self.claim:
            raise ValueError(f"position ({self.tradition}) needs a claim")
        if not self.basis:
            raise ValueError(
                f"position ({self.tradition}) needs a basis — a claim "
                "without its ground is not storable")
        if not self.sources:
            raise ValueError(
                f"position ({self.tradition}) needs at least one source "
                "pointer")

    def to_dict(self) -> dict[str, Any]:
        return {"tradition": self.tradition, "claim": self.claim,
                "basis": self.basis, "sources": self.sources,
                "provenance_note": self.provenance_note}


@dataclass
class Disagreement:
    """A mapped disagreement: positions side by side, never averaged."""

    topic: str
    question: str
    positions: list[Position]
    convergences: list[str] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)
    passages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic, "question": self.question,
            "positions": [p.to_dict() for p in self.positions],
            "convergences": self.convergences,
            "divergences": self.divergences,
            "passages": self.passages,
            "note": "Positions are mapped, not averaged: no tradition's "
                    "claim was merged into another's.",
        }

    def render(self) -> str:
        lines = [f"## {self.topic}: {self.question}", ""]
        for p in self.positions:
            lines.append(f"### {p.tradition}")
            lines.append(f"Claim: {p.claim}")
            lines.append(f"Basis: {p.basis}")
            lines.append(f"Sources: {'; '.join(p.sources)}")
            lines.append("")
        if self.convergences:
            lines.append("Convergences:")
            lines += [f"- {c}" for c in self.convergences]
            lines.append("")
        if self.divergences:
            lines.append("Divergences (not resolved):")
            lines += [f"- {d}" for d in self.divergences]
            lines.append("")
        lines.append("_Positions mapped, not averaged._")
        return "\n".join(lines)


def _builtin_positions() -> dict[str, tuple[str, list[Position]]]:
    """Built-in topic maps. Claims are doctrine-level summaries with
    named textual bases — no invented quotes, no verse text."""
    return {
        "self": (
            "What is the self?",
            [
                Position(
                    tradition="Advaita Vedanta",
                    claim="Atman is Brahman: the true self is non-dual "
                          "awareness, not the body-mind.",
                    basis="Upanishadic mahavakyas and Shankara's "
                          "commentaries: the self is realized by "
                          "discriminating the witness from the witnessed.",
                    sources=["Upanishads (public-domain translations)",
                             "Shankara — Brahma Sutra commentary"],
                    provenance_note="Doctrine-level summary; see corpus "
                                    "passages for textual ground."),
                Position(
                    tradition="Theravada Buddhism",
                    claim="Anatta: there is no permanent, unchanging self; "
                          "what is called self is five aggregates in flux.",
                    basis="The Buddha's anatta-lakkhana teaching: every "
                          "aggregate is impermanent and not-self, so "
                          "clinging to a permanent self is the root error.",
                    sources=["Dhammapada", "Sutta Nipata",
                             "Anatta-lakkhana Sutta (discourse)"],
                    provenance_note="Doctrine-level summary."),
                Position(
                    tradition="Ifa / Yoruba",
                    claim="Ori (inner head) is the seat of personhood and "
                          "destiny: a person is their ori plus their "
                          "character (iwa) and their chosen path.",
                    basis="Yoruba thought as recorded in public "
                          "scholarship: ori is consulted, fed, and "
                          "appeased; character determines whether "
                          "destiny is fulfilled.",
                    sources=["Public scholarship on Yoruba thought",
                             "Nigeria tradition pack (descriptor-only)"],
                    provenance_note="Public-scholarship summary; "
                                    "initiated odu verses are NOT quoted."),
            ],
        ),
        "liberation": (
            "What is liberation, and how is it reached?",
            [
                Position(
                    tradition="Advaita Vedanta",
                    claim="Moksha: liberation is knowledge — recognizing "
                          "oneself as Brahman ends bondage.",
                    basis="Jnana (knowledge) as the direct path; "
                          "practice prepares the mind for recognition.",
                    sources=["Upanishads", "Shankara commentaries"]),
                Position(
                    tradition="Theravada Buddhism",
                    claim="Nibbana: liberation is the extinguishing of "
                          "greed, hatred, and delusion via the Eightfold Path.",
                    basis="Four Noble Truths: suffering has a cause "
                          "(craving) and a path that ends it.",
                    sources=["Dhammapada", "Sutta Nipata"]),
                Position(
                    tradition="Sufism",
                    claim="Fana: liberation is annihilation of the ego "
                          "in the Divine, through love and remembrance.",
                    basis="The path of love (ishq) and dhikr: the self "
                          "is effaced so only the Beloved remains.",
                    sources=["Masnavi of Rumi (abridged)",
                             "Bezels of Wisdom (Ibn Arabi)"]),
            ],
        ),
        "divine": (
            "Is the Divine personal, impersonal, or beyond both?",
            [
                Position(
                    tradition="Advaita Vedanta",
                    claim="Brahman is impersonal non-dual reality; "
                          "personal gods (Ishvara) are true at the "
                          "relative level.",
                    basis="Two-truths framing: saguna (with qualities) "
                          "and nirguna (without qualities) Brahman.",
                    sources=["Upanishads"]),
                Position(
                    tradition="Christianity",
                    claim="God is personal and triune — Father, Son, "
                          "Holy Spirit — known through revelation.",
                    basis="Scripture and creedal tradition: God acts, "
                          "speaks, and enters history.",
                    sources=["King James Bible", "Philokalia (selections)"]),
                Position(
                    tradition="Taoism",
                    claim="The Tao that can be named is not the eternal "
                          "Tao: the ultimate is beyond personal/impersonal.",
                    basis="Tao Te Ching ch. 1: naming already limits; "
                          "the sage aligns with the unnamed way.",
                    sources=["Tao Te Ching"]),
            ],
        ),
    }


BUILTIN_TOPICS: tuple[str, ...] = tuple(_builtin_positions())


class DisagreementMap:
    """Registry + comparison of tradition positions, never averaged."""

    def __init__(self) -> None:
        self._topics: dict[str, tuple[str, list[Position]]] = {}
        for topic, (question, positions) in _builtin_positions().items():
            self._topics[topic] = (question, list(positions))

    def topics(self) -> list[str]:
        return sorted(self._topics)

    def register(self, topic: str, question: str,
                 positions: list[Position]) -> None:
        """Register a new topic. Every position is validated (basis +
        sources required); the topic is stored as given, never merged
        with an existing one."""
        if not topic or not question:
            raise ValueError("register needs topic + question")
        if len(positions) < 2:
            raise ValueError(
                "a disagreement needs >= 2 positions — one position is "
                "a monologue, not a map")
        for p in positions:
            p.validate()
        self._topics[topic] = (question, list(positions))

    def compare(self, topic: str, *,
                convergences: list[str] | None = None,
                divergences: list[str] | None = None) -> Disagreement:
        if topic not in self._topics:
            raise ValueError(
                f"unknown topic {topic!r}; known: {self.topics()}")
        question, positions = self._topics[topic]
        return Disagreement(
            topic=topic, question=question,
            positions=[Position(p.tradition, p.claim, p.basis,
                                list(p.sources), p.provenance_note)
                       for p in positions],
            convergences=list(convergences or []),
            divergences=list(divergences or self._default_divergences(
                positions)))

    @staticmethod
    def _default_divergences(positions: list[Position]) -> list[str]:
        trads = [p.tradition for p in positions]
        return [f"{trads[i]} vs {trads[j]}: claims differ on the "
                f"topic's core — see each position's basis, not a "
                "synthesis"
                for i in range(len(trads))
                for j in range(i + 1, len(trads))]

    def compare_with_corpus(self, topic: str, corpus: Any, *,
                            top: int = 3) -> Disagreement:
        """Attach retrieved passages per tradition (with provenance).

        Passages are grouped by the tradition the corpus reports — the
        map still does not average them.
        """
        disagreement = self.compare(topic)
        if corpus is None:
            return disagreement
        try:
            answer = corpus.ask(topic, top=top * 3, mode="keyword")
            passages = getattr(answer, "passages", [])
        except Exception:  # noqa: BLE001 - corpus failure ≠ map failure
            return disagreement
        manifest = getattr(getattr(corpus, "corpus", corpus),
                           "_manifest", {}) or {}
        for hit in passages:
            d = hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
            tradition = d.get("tradition", "")
            if not tradition:
                for slug, entry in manifest.items():
                    if slug in (d.get("work") or ""):
                        tradition = getattr(entry, "tradition", "")
                        break
            disagreement.passages.setdefault(tradition or "unknown",
                                             []).append(d)
        return disagreement
