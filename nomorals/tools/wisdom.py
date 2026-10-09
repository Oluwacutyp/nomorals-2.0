"""wisdom — agent tool for WisdomKeeper.

Lets the brain answer esoteric/wisdom questions from natural language
("when was the earth created?") without the user needing /wisdom.
Uses dynamic import to respect layering.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(registry: Any) -> None:
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
    def wisdom_ask(context: Any, query: str) -> dict[str, Any]:
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
    def wisdom_status(context: Any) -> dict[str, Any]:
        import importlib
        mod = importlib.import_module("nomorals.wisdom.keeper")
        keeper = mod.WisdomKeeper(context)
        return {"ok": True, **keeper.status()["corpus"]}
