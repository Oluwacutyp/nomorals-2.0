"""``nm scholar`` — the full Semantic Scholar record of a paper.

    nm scholar <title or DOI> [--json]
    nm scholar list [--json]
    nm scholar show <id> [--json]
    nm scholar watch <topic> [--json]
    nm scholar feeds [--json]
    nm scholar unwatch <topic> [--json]

The same paper-record organ chat /scholar uses: the full record —
authors, venue, year, citation counts, the influential citations
named with their intents, a one-line TLDR (Semantic Scholar's own,
or one generated locally from the abstract when the API carries
none), fields of study, and the open-access PDF when one exists.
A paper with no record prints the honest reason and exits 1 —
never an invented record.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...research.paper_records import (
    check_feeds,
    ensure_tldr,
    fetch_influential_citations,
    fetch_paper_record,
    format_record_reply,
    open_record_store,
    search_records,
)

__all__ = ["_cmd_scholar"]


def _home(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _print(answer: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer, indent=2, default=str))
    else:
        print(answer.get("reply") or answer.get("error") or "")


def _cmd_scholar(args: argparse.Namespace, context: Any) -> int:
    home = _home(context)
    store = open_record_store(home)
    words = list(getattr(args, "scholar_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    verb = (words[0].lower() if words else "")
    rest = " ".join(words[1:]).strip() if len(words) > 1 else ""
    if not words:
        print("usage: nm scholar <title or DOI> | list | show <id> | "
              "watch <topic> | feeds | unwatch <topic>")
        return 2
    if verb == "list":
        rows = store.list()
        if as_json:
            print(json.dumps({"ok": True, "records": rows}, indent=2))
        elif not rows:
            print("no paper records yet — fetch one with "
                  "nm scholar <title or DOI>")
        else:
            print("paper records:")
            for row in rows[:10]:
                print(f"  {row['paper_id']} — {row['title']} "
                      f"({row.get('year') or '?'})")
        return 0
    if verb == "feeds":
        feeds = store.feeds()
        if as_json:
            print(json.dumps({"ok": True, "feeds": feeds}, indent=2))
        elif not feeds:
            print("no topic feeds watched — nm scholar watch <topic>")
        else:
            print("watched topics:")
            for topic in feeds:
                print(f"  {topic}")
        return 0
    if verb == "watch":
        if not rest:
            print("usage: nm scholar watch <topic>")
            return 2
        store.watch(rest)
        print(f"watching '{rest}' — new papers surface on the daily tick")
        return 0
    if verb == "unwatch":
        if not rest:
            print("usage: nm scholar unwatch <topic>")
            return 2
        if not store.unwatch(rest):
            print(f"no feed watched for '{rest}'")
            return 1
        print(f"stopped watching '{rest}'")
        return 0
    if verb == "check_feeds":
        found = check_feeds(store)
        if as_json:
            print(json.dumps({"ok": True, "new": found}, indent=2,
                             default=str))
        elif not found:
            print("no new papers on watched topics")
        else:
            for topic, rows in found:
                print(f"new on '{topic}':")
                for row in rows[:5]:
                    print(f"  {row['title']} ({row.get('year') or '?'})")
        return 0
    if verb == "show":
        if not rest:
            print("usage: nm scholar show <id>")
            return 2
        record = store.get(rest)
        if record is None:
            print(f"no stored record '{rest}'")
            return 1
        _print({"ok": True, "record": record.to_dict(),
                "reply": format_record_reply({"ok": True, "record": record})},
               as_json)
        return 0
    if verb == "search":
        if not rest:
            print("usage: nm scholar search <topic>")
            return 2
        rows, note = search_records(rest)
        if as_json:
            print(json.dumps({"ok": bool(rows), "rows": rows, "note": note},
                             indent=2))
        elif not rows:
            print(note)
        else:
            print(note)
            for row in rows:
                print(f"  {row['title']} ({row.get('year') or '?'})")
        return 0 if rows else 1
    query = " ".join(words).strip()
    record, note = fetch_paper_record(query)
    if record is None:
        print(note)
        return 1
    stored = store.get(record.paper_id) if record.paper_id else None
    if stored is not None and stored.tldr:
        record = stored
    else:
        record = ensure_tldr(record)
        store.save(record)
    if record.paper_id:
        record.influential = fetch_influential_citations(record.paper_id)
        store.save(record)
    _print({"ok": True, "record": record.to_dict(),
            "reply": format_record_reply({"ok": True, "record": record}),
            "note": note}, as_json)
    return 0
