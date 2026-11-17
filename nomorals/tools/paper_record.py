"""Paper-record tool: ``paper_record`` — the full Semantic Scholar record.

Devon hands over a paper and gets its full record: authors, venue,
year, citation counts, the influential citations named, a one-line
TLDR, fields of study, and the open-access PDF when one exists. The
tool is a thin surface over :mod:`nomorals.research.paper_records`,
the Semantic Scholar extension beside the evidence, consensus and
paper-graph organs — one retrieval path, no second corpus. The TLDR
is Semantic Scholar's own when the API carries one, otherwise one
generated locally from the abstract and stored — generated once,
reused. Topic feeds watch a subject and surface new papers. A paper
with no record is refused honestly, never invented.
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


def register(registry: Any) -> None:
    """Attach the ``paper_record`` tool to a registry."""
    context = registry.context

    @registry.register(
        "paper_record",
        description=(
            "Paper record — the full Semantic Scholar record of a paper. "
            "Give it a paper (title or DOI) and get authors, venue, year, "
            "citation counts, the influential citations named with their "
            "intents, a one-line TLDR (Semantic Scholar's own, or one "
            "generated locally from the abstract when the API carries "
            "none), fields of study, and the open-access PDF when one "
            "exists — or an honest 'no open-access PDF' when it does not. "
            "action=<record|search|list|show|watch|feeds|unwatch>; record "
            "takes query=<title or DOI>; search takes query=<topic>; show "
            "takes record_id=<id from a previous record>; watch/unwatch "
            "take topic=<subject>. A paper with no record is refused "
            "honestly — no paper, author, citation or TLDR is ever "
            "invented. Returns ok, record, reply, or error."
        ),
        capability=Capability.NET_OUT,
    )
    def paper_record_tool(
        query: str = "",
        action: str = "record",
        record_id: str = "",
        topic: str = "",
    ) -> dict[str, Any]:
        from ..research.paper_records import (
            check_feeds,
            ensure_tldr,
            fetch_influential_citations,
            fetch_paper_record,
            format_record_reply,
            open_record_store,
            search_records,
        )

        home = _home_from_context(context)
        store = open_record_store(home)
        act = (action or "record").strip().lower()
        if act == "list":
            return {"ok": True, "records": store.list()}
        if act == "feeds":
            return {"ok": True, "feeds": store.feeds()}
        if act == "watch":
            topic = (topic or query or "").strip()
            if not topic:
                return {"ok": False, "error": "watch needs a topic"}
            store.watch(topic)
            return {"ok": True, "watched": topic}
        if act == "unwatch":
            topic = (topic or query or "").strip()
            if not store.unwatch(topic):
                return {"ok": False,
                        "error": f"no feed watched for '{topic}'"}
            return {"ok": True, "unwatched": topic}
        if act == "check_feeds":
            found = check_feeds(store)
            return {"ok": True, "new": found}
        if act == "show":
            record = store.get((record_id or query or "").strip())
            if record is None:
                return {"ok": False,
                        "error": f"no stored record '{record_id or query}'"}
            return {"ok": True, "record": record.to_dict(),
                    "reply": format_record_reply(
                        {"ok": True, "record": record})}
        if act == "search":
            rows, note = search_records(query)
            return {"ok": bool(rows), "rows": rows, "note": note}
        query = (query or "").strip()
        if not query:
            return {"ok": False,
                    "error": "record needs a paper title or DOI"}
        stored = None
        record, note = fetch_paper_record(query)
        if record is None:
            return {"ok": False, "error": note}
        if record.paper_id:
            stored = store.get(record.paper_id)
        if stored is not None and stored.tldr:
            # The stored record already carries its TLDR — reuse it,
            # never regenerate per read.
            record = stored
        else:
            record = ensure_tldr(record)
            store.save(record)
        if record.paper_id:
            record.influential = fetch_influential_citations(record.paper_id)
            store.save(record)
        return {"ok": True, "record": record.to_dict(),
                "reply": format_record_reply({"ok": True, "record": record}),
                "note": note}
