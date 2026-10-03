"""``nm search <query>`` — universal federated search across Devon's stores."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any


def _cmd_search(args: Any, context: Any) -> int:
    """Route ``nm search``. Returns a process exit code."""
    from ...search import (
        SearchError,
        federated_search,
        valid_source_names,
        valid_types,
    )

    words = list(getattr(args, "task", None) or [])
    if not words:
        print(
            "usage: nm search <query> [--source NAME]... [--type TYPE]...\n"
            "                  [--since DATE] [--before DATE] [--limit N]\n"
            "                  [--dir DIR] [--json]\n"
            f"sources: {', '.join(valid_source_names())}\n"
            f"types:   {', '.join(valid_types())}",
            file=sys.stderr,
        )
        return 2
    query = " ".join(words)

    # The timeline store lives in os/ (L6); this command module is L7, so
    # it may wire the injected Timeline the search layer needs.
    from ...os.timeline import Timeline

    db_path = getattr(getattr(context, "db", None), "path", None)
    tl = Timeline(db_path)
    try:
        try:
            response = federated_search(
                query,
                context=context,
                sources=getattr(args, "source", None) or None,
                limit=getattr(args, "limit", 10) or 10,
                types=getattr(args, "type", None) or None,
                since=getattr(args, "since", None),
                before=getattr(args, "before", None),
                doc_dir=getattr(args, "dir", "") or None,
                timeline=tl,
            )
        except SearchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    finally:
        tl.close()

    if getattr(args, "json", False):
        print(json.dumps(response.to_dict(), indent=2, default=str,
                         ensure_ascii=False))
        return 0

    if response.sources_skipped:
        for name, note in response.sources_skipped.items():
            print(f"[{name} skipped: {note}]")
    if not response.hits:
        searched = ", ".join(response.sources_searched) or "(none)"
        print(f"no matches (searched: {searched})")
        return 0

    for hit in response.hits:
        ts = ""
        if hit.timestamp:
            ts = datetime.fromtimestamp(
                hit.timestamp, tz=timezone.utc).strftime("%Y-%m-%d") + " "
        print(f"[{hit.source}/{hit.type} {hit.score:.2f}] {ts}{hit.title}")
        if hit.snippet:
            print(f"    {hit.snippet[:220]}")
    if response.deduped:
        print(f"({response.deduped} duplicate(s) merged)")
    return 0
