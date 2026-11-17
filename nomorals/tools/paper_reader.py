"""Paper-reader tool: ``paper_reader`` — read a paper, explain it.

Devon hands over a paper (DOI, title, pasted text, or a local file)
and gets a plain-language explanation: what it asks, what it did,
what it found — every finding quoted verbatim from the paper — why
it matters, and the hard parts (equations, tables) unpacked into
named symbols and compared columns. She can ask questions against
the paper's own sentences, cite it in APA / MLA / Chicago / BibTeX,
and paraphrase a span more simply without losing a number or a
name. The tool is a thin surface over
:mod:`nomorals.research.paper_reader`, the SciSpace extension
beside the evidence, consensus, paper-graph and paper-record
organs — one research family, no second corpus. An unknown paper,
or a question the paper does not cover, is refused honestly —
nothing here is ever invented.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _llm_from_context(context: Any):
    """An LLM callable from the context router, or None.

    The extractive reading never needs a model; when the context
    carries a router, its complete() backs the plain-language
    rewording rung (still guarded by the number/name invariance
    check in the organ).
    """
    router = getattr(context, "router", None) if context else None
    if router is None:
        return None

    def _call(prompt: str) -> str:
        resp = router.complete(prompt)
        return getattr(resp, "text", "") or str(resp)

    return _call


def register(registry: Any) -> None:
    """Attach the ``paper_reader`` tool to a registry."""
    context = registry.context

    @registry.register(
        "paper_reader",
        description=(
            "Paper reader — read a paper and explain it in plain "
            "language (SciSpace core). Give it a paper (DOI, title, "
            "pasted text, or a local file path) and get what it asks, "
            "what it did, what it found — every finding quoted "
            "verbatim from the paper — why it matters, and the hard "
            "parts (equations, tables) unpacked. action=<read|explain|"
            "ask|cite|paraphrase|list|show>; read takes query=<DOI, "
            "title, pasted text, or file path>; explain takes "
            "reading_id=<id from a previous read> or query=<paper>; "
            "ask takes reading_id=<id> and question=<question>; cite "
            "takes reading_id=<id> and style=<apa|mla|chicago|bibtex>; "
            "paraphrase takes text=<span to simplify>; show takes "
            "reading_id=<id>. An unknown paper, or a question the "
            "paper does not cover, is refused honestly — no finding, "
            "quote, or answer is ever invented. Returns ok, "
            "reading_id, explanation, reply, or error."
        ),
        capability=Capability.NET_OUT,
    )
    def paper_reader_tool(
        query: str = "",
        action: str = "read",
        reading_id: str = "",
        question: str = "",
        style: str = "apa",
        text: str = "",
    ) -> dict[str, Any]:
        from ..research.paper_reader import (
            ask_paper,
            explain_span,
            format_citation,
            format_reading_reply,
            open_reading_store,
            paraphrase_span,
            read_paper,
        )

        home = _home_from_context(context)
        store = open_reading_store(home)
        llm_fn = _llm_from_context(context)
        act = (action or "read").strip().lower()
        if act == "list":
            return {"ok": True, "readings": store.list()}
        if act == "show":
            row = store.get((reading_id or query or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored reading "
                                 f"'{reading_id or query}'"}
            shown = dict(row)
            shown["ok"] = True
            shown["reply"] = format_reading_reply(
                {**row, "ok": True,
                 "reading_id": row["paper"]["paper_id"]})
            return shown
        if act == "paraphrase":
            span = (text or query or "").strip()
            return paraphrase_span(span, llm_fn=llm_fn)
        if act == "cite":
            row = store.get((reading_id or query or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored reading "
                                 f"'{reading_id or query}' — read the "
                                 "paper first"}
            from ..research.paper_records import PaperRecord

            record = PaperRecord.from_dict(row["paper"])
            try:
                return {"ok": True,
                        "citation": format_citation(record, style)}
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        if act == "explain":
            row = store.get((reading_id or query or "").strip())
            if row is not None:
                from ..research.paper_reader import (
                    PaperExplanation,
                    PaperText,
                )

                paper = PaperText.from_dict(row["paper"])
                explanation = PaperExplanation.from_dict(
                    row["explanation"])
                return {"ok": True, "reading_id": paper.paper_id,
                        "paper": paper.to_dict(),
                        "explanation": explanation.to_dict(),
                        "reply": format_reading_reply(
                            {"ok": True, "reading_id": paper.paper_id,
                             "paper": paper.to_dict(),
                             "explanation": explanation.to_dict()})}
            if not (query or "").strip():
                return {"ok": False,
                        "error": "explain needs reading_id=<id> or "
                                 "query=<DOI or title>"}
            return read_paper(query, home=home, llm_fn=llm_fn)
        if act == "ask":
            row = store.get((reading_id or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored reading '{reading_id}' — "
                                 "read the paper first"}
            from ..research.paper_reader import PaperText

            paper = PaperText.from_dict(row["paper"])
            result = ask_paper(paper, question, llm_fn=llm_fn)
            if result.get("ok"):
                store.log_question(paper.paper_id, question,
                                   result["answer"])
            return result
        if act == "explain_span":
            row = store.get((reading_id or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored reading '{reading_id}' — "
                                 "read the paper first"}
            from ..research.paper_reader import PaperText

            paper = PaperText.from_dict(row["paper"])
            return explain_span(paper, query, llm_fn=llm_fn)
        if act != "read":
            raise ValueError(
                f"unknown paper_reader action “{act}” — "
                "read|explain|ask|cite|paraphrase|list|show")
        query = (query or "").strip()
        if not query:
            return {"ok": False,
                    "error": "read needs a paper DOI, title, pasted "
                             "text, or file path"}
        return read_paper(query, home=home, llm_fn=llm_fn)
