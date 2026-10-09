"""wisdom — agent tool for WisdomKeeper.

Lets the brain answer esoteric/wisdom questions from natural language
("when was the earth created?") without the user needing /wisdom.
Uses dynamic import to respect layering.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(registry: Any) -> None:
    # Closure pattern: context comes from the registry, never as a tool
    # parameter. (The old signature took `context` positionally, which
    # broke every call through registry.call — the spine couldn't reach
    # these tools at all.)
    context = registry.context

    @registry.register(
        "wisdom_ask",
        description=(
            "Ask the WisdomKeeper corpus: esoteric, metaphysical, religious "
            "and wisdom texts (scriptures, hermetic, gnostic, apocrypha, "
            "philosophy). USE THIS when the user asks about spiritual / "
            "esoteric / religious topics, the nature of reality, ancient "
            "texts, or wisdom traditions. Returns passages with provenance. "
            "Args: query (the question, plain language)."
        ),
        capability="network",
        parameters={
            "query": "str — the wisdom/esoteric question",
        },
    )
    def wisdom_ask(query: str) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.keeper")
        keeper = mod.WisdomKeeper(context)
        try:
            st = keeper.status()["corpus"]
        except Exception:  # noqa: BLE001
            st = {}
        if not st.get("ingested"):
            return {"ok": False,
                    "error": "corpus empty — run /wisdom seed first",
                    "query": query}
        ans = keeper.ask(query, top=5)
        return {"ok": True,
                "query": query,
                "synthesis": ans.synthesis,
                "passages": [{"work": p.work, "section": p.section,
                              "snippet": p.snippet.strip(),
                              "url": p.url or "",
                              "canon": p.canon_status}
                             for p in ans.passages[:5]]}

    @registry.register(
        "wisdom_status",
        description=(
            "Check the WisdomKeeper corpus: how many texts are ingested, "
            "by tradition. Use to answer 'is wisdom ready?' questions."
        ),
        capability="network",
        parameters={},
    )
    def wisdom_status() -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.keeper")
        keeper = mod.WisdomKeeper(context)
        return {"ok": True, **keeper.status()["corpus"]}

    @registry.register(
        "wisdom_ingest",
        description=(
            "Queue a text for the WisdomKeeper corpus: the autonomous "
            "wisdom organ ingests it (fetch, parse, index) on its own tick. "
            "Use when the owner shares a sacred/philosophical text, or when "
            "research turns up something worth keeping. Args: url, title, "
            "tradition (optional)."
        ),
        capability="network",
        parameters={
            "url": "str — the text's URL",
            "title": "str — human title",
            "tradition": "str (optional) — e.g. christianity, sufism, philosophy",
        },
    )
    def wisdom_ingest(url: str, title: str,
                      tradition: str = "") -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.autonomy")
        organ = mod.WisdomOrgan(context)
        qid = organ.queue_ingest(url, title, tradition)
        return {"ok": True, "queued_id": qid, "title": title,
                "note": "the wisdom organ ingests it on its own tick"}

    @registry.register(
        "wisdom_digest",
        description=(
            "Digest an ingested wisdom text into its key passages "
            "(extractive, no model needed). Use to brief the owner on what "
            "a text actually contains. Args: slug (from wisdom_status)."
        ),
        capability="network",
        parameters={"slug": "str — the manifest slug, e.g. bible-kjv"},
    )
    def wisdom_digest(slug: str) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.autonomy")
        organ = mod.WisdomOrgan(context)
        digest = organ.get_digest(slug) or organ.digest_work(slug)
        return {"ok": True, **digest}

    @registry.register(
        "wisdom_synthesize",
        description=(
            "Cross-tradition wisdom synthesis: asks the corpus across all "
            "traditions and fuses passages with shared-concept notes from "
            "the link graph. USE THIS for 'what do traditions say about X' "
            "questions. Args: query."
        ),
        capability="network",
        parameters={"query": "str — the question, plain language"},
    )
    def wisdom_synthesize(query: str) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.autonomy")
        organ = mod.WisdomOrgan(context)
        return {"ok": True, **organ.synthesize(query)}

    @registry.register(
        "wisdom_brief",
        description=(
            "What the wisdom organ has been doing on its own: recent "
            "ingestions, digests, and cross-tradition links. Use to answer "
            "'what has wisdom been learning?'"
        ),
        capability="network",
        parameters={},
    )
    def wisdom_brief() -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.autonomy")
        organ = mod.WisdomOrgan(context)
        keeper_mod = importlib.import_module("nomorals.wisdom.keeper")
        keeper = keeper_mod.WisdomKeeper(context)
        return {"ok": True,
                "corpus": keeper.status()["corpus"],
                "recent_digests": organ.recent_digests(),
                "ingest_queue": organ.db.query(
                    "SELECT COUNT(*) AS n FROM wisdom_ingest_queue"
                    " WHERE status = 'queued'")[0]["n"]}

    @registry.register(
        "wisdom_organ_tick",
        description=(
            "Run one autonomous wisdom-organ cycle: ingest pending texts, "
            "digest new works, cross-link traditions. This is what the "
            "scheduler runs — the owner never calls it directly."
        ),
        capability="network",
        parameters={},
    )
    def wisdom_organ_tick() -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.autonomy")
        organ = mod.WisdomOrgan(context)
        return {"ok": True, **organ.tick()}
