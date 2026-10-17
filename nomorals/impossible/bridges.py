"""Bridges — Impossible <-> WisdomKeeper <-> BookForge, hand in hand.

Owner rule: "all knowledge should be able to work hand in hand when need be."

Findings (verified in code, not assumed):
- Impossible (science/engineering): digestion sources registry
  (sources.py) + prior art + feasibility gate + falsifier.
- WisdomKeeper (esoteric / philosophical / ethics): HAS sources +
  digest — ArchiveIngestor with 4 search backends (Gutendex,
  Wikisource, archive.org, web fallback), explicit ingest
  (search -> ingest_entry, sha-verified), 47-seed canon corpus
  (canon/apocrypha/pseudepigrapha/gnostic/DSS/eastern/esoteric),
  FTS5 + semantic + hybrid ask with provenance. Digest is EXPLICIT
  (operator drives search -> ingest), not an always-on auto-tick.
- BookForge / books (narrative / story): HAS sources + digest —
  descriptor adapters (Gutenberg, Scribble Hub, AO3, Webnovel,
  Wattpad) + aggregator adapters (FreeWebNovel, NovelFull, RoyalRoad,
  PandaNovel unverified, Generic self-tuning), whole-story download
  with resume/progress, StoryBible auto-built from read chapters
  (cast, threads, voiceprint), continuation in the story's voice.

This module does NOT rebuild any of that. It is the wiring between
them, with lazy imports (no heavy import at module load, no circular
imports) and honest "unavailable" whenever a side has no corpus /
library wired — never fake cross-system results.

- Impossible digest draws on WisdomKeeper when a problem touches
  philosophy / ethics / meaning / consciousness (keyword gate).
- Impossible digest draws on BookForge when a problem touches
  narrative / story / character / framing.
- WisdomKeeper claims can be grounded by Impossible's feasibility
  gate (physics check on an esoteric claim — reported as a gate
  verdict, not as a dismissal).
- BookForge story bibles can feed narrative framing into an
  Impossible dossier.
"""

from __future__ import annotations

# Keyword gates for "does this problem touch the other system's turf?"
WISDOM_KEYWORDS = (
    "meaning", "purpose", "ethics", "ethical", "moral", "consciousness",
    "soul", "spirit", "karma", "meditation", "enlightenment", "afterlife",
    "reincarnation", "astral", "kundalini", "chakra", "prayer", "ritual",
    "myth", "archetype", "wisdom", "philosophy", "existential", "divine",
    "sacred", "prophecy", "oracle", "dream", "void", "akashic",
)

NARRATIVE_KEYWORDS = (
    "story", "novel", "character", "narrative", "plot", "fiction",
    "screenplay", "dialogue", "protagonist", "worldbuilding", "lore",
    "serial", "chapter", "book", "tale", "fable", "mythology",
)


def touches_wisdom_domains(problem: str) -> bool:
    text = (problem or "").lower()
    return any(k in text for k in WISDOM_KEYWORDS)


def touches_narrative(problem: str) -> bool:
    text = (problem or "").lower()
    return any(k in text for k in NARRATIVE_KEYWORDS)


def query_wisdom(problem: str, corpus=None) -> dict:
    """Ask WisdomKeeper's corpus. ``corpus`` is an injected CanonCorpus
    (duck-typed: needs .ask). Without one, honest unavailable — the
    bridge never constructs a corpus behind the caller's back."""
    if corpus is None:
        return {"system": "wisdomkeeper", "status": "unavailable",
                "reason": "No WisdomKeeper corpus wired — pass a CanonCorpus to cross_query().",
                "passages": []}
    try:
        answer = corpus.ask(problem, top=3)
        passages = [p.to_dict() if hasattr(p, "to_dict") else dict(p)
                    for p in (getattr(answer, "passages", None) or [])]
        return {"system": "wisdomkeeper", "status": "found" if passages else "none",
                "passages": passages,
                "synthesis": getattr(answer, "synthesis", ""),
                "confidence": getattr(answer, "confidence", "")}
    except Exception as exc:  # noqa: BLE001 — a dead side never kills the bridge
        return {"system": "wisdomkeeper", "status": "unavailable",
                "reason": f"WisdomKeeper ask failed: {exc}. No results invented.",
                "passages": []}


def query_books(problem: str, library=None) -> dict:
    """Search BookForge's library. ``library`` is an injected
    books.Library (duck-typed: needs .search). Without one, honest
    unavailable."""
    if library is None:
        return {"system": "bookforge", "status": "unavailable",
                "reason": "No BookForge library wired — pass a books.Library to cross_query().",
                "hits": []}
    try:
        hits = library.search(problem, top=3)
        out = []
        for h in hits or []:
            out.append({"title": getattr(h, "title", getattr(h, "book", "")),
                        "passage": getattr(h, "passage", ""),
                        "chapter": getattr(h, "chapter", ""),
                        "score": getattr(h, "score", 0.0)})
        return {"system": "bookforge", "status": "found" if out else "none", "hits": out}
    except Exception as exc:  # noqa: BLE001
        return {"system": "bookforge", "status": "unavailable",
                "reason": f"BookForge search failed: {exc}. No results invented.",
                "hits": []}


def ground_claim_with_science(claim: str) -> dict:
    """Ground any claim (e.g. a WisdomKeeper claim) with Impossible's
    feasibility gate. Returns the gate verdict + reframed achievable
    version. A gate verdict is physics evidence, not a dismissal —
    the caller decides what it means for the claim."""
    from . import feasible as _feasible
    from . import formulate as _formulate
    formulation = _formulate.formulate(claim)
    if formulation.underspecified:
        return {"system": "impossible", "status": "needs_clarification",
                "clarifications": formulation.clarifications,
                "note": "Claim too underspecified to gate — not guessing."}
    gate = _feasible.gate(formulation)
    return {"system": "impossible", "status": "gated",
            "gate_verdict": gate.get("verdict"), "why": gate.get("why"),
            "failed": gate.get("failed", [])}


def narrative_frame(problem: str, bible=None) -> dict:
    """Feed a BookForge story bible into an Impossible dossier as
    narrative framing. ``bible`` is an injected StoryBible
    (duck-typed). Without one, honest unavailable."""
    if bible is None:
        return {"system": "bookforge", "status": "unavailable",
                "reason": "No story bible supplied — pass a StoryBible to cross_query().",
                "frame": ""}
    try:
        cast = [getattr(c, "name", str(c)) for c in (getattr(bible, "characters", None) or [])][:8]
        threads = [getattr(t, "title", str(t)) for t in (getattr(bible, "threads", None) or [])][:8]
        frame = (f"Narrative frame for '{problem}': cast={cast or 'unknown'}; "
                 f"open threads={threads or 'unknown'}.")
        return {"system": "bookforge", "status": "found", "frame": frame,
                "cast": cast, "threads": threads}
    except Exception as exc:  # noqa: BLE001
        return {"system": "bookforge", "status": "unavailable",
                "reason": f"Story bible read failed: {exc}.", "frame": ""}


def cross_query(problem: str, *, wisdom=None, books=None, bible=None) -> dict:
    """One problem, three knowledge systems, hand in hand.

    Always asks Impossible's own gate (the science side). Asks
    WisdomKeeper when the problem touches philosophy/ethics/meaning
    (or whenever a corpus is injected); asks BookForge when it
    touches narrative (or whenever a library is injected). Every
    side reports its own status honestly.
    """
    from . import feasible as _feasible
    from . import formulate as _formulate
    formulation = _formulate.formulate(problem)
    gate = None
    if not formulation.underspecified:
        gate = _feasible.gate(formulation)
    result: dict = {
        "problem": problem,
        "touches_wisdom": touches_wisdom_domains(problem),
        "touches_narrative": touches_narrative(problem),
        "impossible": ({"status": "gated", "gate_verdict": gate.get("verdict"),
                        "why": gate.get("why")} if gate else
                       {"status": "needs_clarification",
                        "clarifications": formulation.clarifications}),
    }
    if result["touches_wisdom"] or wisdom is not None:
        result["wisdomkeeper"] = query_wisdom(problem, corpus=wisdom)
    if result["touches_narrative"] or books is not None or bible is not None:
        result["bookforge"] = query_books(problem, library=books)
        if bible is not None:
            result["bookforge"]["narrative_frame"] = narrative_frame(problem, bible=bible)
    return result
