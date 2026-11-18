"""Brain tool: requirements analysis (dynavio REA core, 52-websites #51).

"pull every requirement out of this spec" / "where does this document
contradict itself?" — feed it a specification (pasted text, a file
path, or several files as one corpus) and get every requirement
statement extracted, categorized, and scored against the classic
requirement smells, the relations between them as a graph, and the
brain rung's contradiction judgments — each labelled as a judgment
with its reason, open to owner override.

The tool is a thin shell over :mod:`nomorals.documents.requirements`:
the organ owns extraction, similarity, and the store. The code rung
is fully offline and never invents anything — every requirement names
its exact span in the source document. The brain rung (the context
router's complete()) only ever judges what the code rung extracted,
and anything it claims without a real id or a verbatim span is
dropped.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from .paper_reader import _home_from_context, _llm_from_context


def register(registry: Any) -> None:
    """Attach the ``analyze_requirements`` tool to a registry."""
    context = registry.context

    @registry.register(
        "analyze_requirements",
        description=(
            "Requirements analysis (dynavio REA core) — pull every "
            "requirement out of a spec, flag the ambiguous ones, and "
            "find where a document contradicts itself. Give it a "
            "specification as pasted text, a file path, or several "
            "file paths as one corpus, and get each requirement "
            "statement extracted with its exact span in the source, "
            "categorized (functional / quality / constraint), scored "
            "against the requirement smells (ambiguous, weak verb, "
            "unverifiable, incomplete, compound — each with its "
            "evidence word), and linked into a graph (similar, "
            "depends). The brain rung adds contradiction judgments — "
            "labelled as judgments with reasons, open to override. "
            "Ask 'pull every requirement out of this spec' or 'where "
            "does this document contradict itself?'. action=<analyze|"
            "corpus|list|show|report>; analyze takes text=<pasted "
            "spec> or path=<file path>; corpus takes paths=<comma-"
            "separated file paths> to analyze several documents as "
            "one pooled graph; list shows past analyses; show takes "
            "analysis_id=<id>; report takes analysis_id=<id> and "
            "returns the markdown report with its file path. A "
            "missing file, or an analysis with no input, is refused "
            "honestly — no requirement, span, or link is ever "
            "invented. Returns ok, analysis_id, requirements, "
            "summary, reply, or error."
        ),
        capability=Capability.MEM_WRITE,
    )
    def analyze_requirements_tool(
        text: str = "",
        path: str = "",
        paths: str = "",
        action: str = "analyze",
        analysis_id: str = "",
    ) -> dict[str, Any]:
        from ..documents.requirements import (
            analyze_corpus,
            analyze_document,
            format_requirements_reply,
            open_requirements_store,
            render_report_markdown,
            write_report,
        )

        home = _home_from_context(context)
        store = open_requirements_store(home)
        llm_fn = _llm_from_context(context)
        act = (action or "analyze").strip().lower()
        if act == "list":
            return {"ok": True, "analyses": store.list()}
        if act == "show":
            row = store.get((analysis_id or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored analysis "
                                 f"'{analysis_id}'"}
            out = row.to_dict()
            out["ok"] = True
            out["reply"] = format_requirements_reply(row)
            return out
        if act == "report":
            row = store.get((analysis_id or "").strip())
            if row is None:
                return {"ok": False,
                        "error": f"no stored analysis "
                                 f"'{analysis_id}'"}
            report_path = write_report(row, home)
            return {"ok": True, "analysis_id": row.analysis_id,
                    "report_path": str(report_path),
                    "report": render_report_markdown(row)}
        if act == "corpus":
            wanted = [p.strip() for p in (paths or "").split(",")
                      if p.strip()]
            if not wanted:
                return {"ok": False,
                        "error": "corpus needs paths=<comma-separated "
                                 "file paths>"}
            from ..documents.parsers import parse_path

            docs = []
            for wanted_path in wanted:
                try:
                    docs.append(parse_path(wanted_path))
                except Exception as exc:  # noqa: BLE001 — named, kept
                    return {"ok": False,
                            "error": f"cannot read {wanted_path}: {exc}"}
            analysis = analyze_corpus(docs, llm_fn=llm_fn, store=store)
            out = analysis.to_dict()
            out["reply"] = format_requirements_reply(analysis)
            return out
        analysis = analyze_document(
            text=(text or "") or None,
            path=(path or "") or None,
            llm_fn=llm_fn, store=store)
        out = analysis.to_dict()
        out["reply"] = format_requirements_reply(analysis)
        return out
