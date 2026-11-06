"""``nm organs`` — the dormant-organ inventory, served (AA-48/AA-49).

Two meta organs existed with no production consumer:
``meta/dormant_inventory.py`` mechanically scans the source tree for
modules nothing references, and ``meta/expose_dormant.py`` keeps a
hand-verified catalogue of dormant-but-working organs (name, module,
what it does, how to invoke it).  Nothing in production ever ran the
scan or queried the catalogue, so both were write-only knowledge.

This command is the consumer.  The scan is a source-tree AST walk
over ``nomorals/`` + ``tests/`` — far too heavy for startup — so it
runs ONLY on demand (``nm organs scan``), never at boot and never in
``nm doctor``'s default path (doctor carries an exit-code contract
on the owner's EC2 deploys).  The catalogue itself is cheap (built
at import from on-disk existence checks) and is served directly.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any


def _render_entry(entry: Any, status: str = "") -> str:
    head = f"• {entry.name} — {entry.module}"
    if status:
        head += f" [{status}]"
    return f"{head}\n    {entry.what}\n    invoke: {entry.invoke}"


def _cmd_organs(args: argparse.Namespace, settings: Any) -> int:
    from ...meta import dormant_inventory, expose_dormant

    action = (getattr(args, "action", None) or "list").strip().lower()
    as_json = bool(getattr(args, "json", False))

    if action == "find":
        keyword = getattr(args, "keyword", "") or ""
        found = expose_dormant.find(keyword)
        if as_json:
            print(json.dumps(
                {"keyword": keyword,
                 "found": [{"name": e.name, "module": e.module,
                            "what": e.what, "invoke": e.invoke}
                           for e in found]},
                indent=2))
        else:
            print(f"organs matching {keyword!r}: {len(found)}")
            for e in found:
                print(_render_entry(e))
        return 0

    if action == "scan":
        started = time.time()
        result = dormant_inventory.scan()
        seconds = round(time.time() - started, 2)
        dormant_mods = {d.module for d in result.dormant}
        catalogue = expose_dormant.entries()
        still = [e for e in catalogue if e.module in dormant_mods]
        wired = [e for e in catalogue if e.module not in dormant_mods]
        if as_json:
            print(json.dumps({
                "scanned": result.scanned,
                "referenced": result.referenced_count,
                "dormant_count": len(result.dormant),
                "dormant": [d.to_dict() for d in result.dormant],
                "catalogue": {"total": len(catalogue),
                              "still_dormant": [e.module for e in still],
                              "now_wired": [e.module for e in wired]},
                "scan_seconds": seconds,
            }, indent=2))
            return 0
        print(f"dormant-organ scan ({seconds:.2f}s, on demand):")
        print(f"  modules scanned: {result.scanned} · "
              f"referenced: {result.referenced_count} · "
              f"dormant: {len(result.dormant)}")
        for d in result.dormant:
            print(f"  • {d.module} — {d.reason}")
        print(f"catalogue cross-check: {len(catalogue)} organs — "
              f"{len(still)} still dormant, {len(wired)} now wired")
        for e in still:
            print(_render_entry(e, "still dormant"))
        for e in wired:
            print(_render_entry(e, "now wired"))
        return 0

    entries = expose_dormant.entries()
    if as_json:
        print(json.dumps(
            {"organs": [{"name": e.name, "module": e.module,
                         "what": e.what, "invoke": e.invoke}
                        for e in entries]},
            indent=2))
    else:
        print(f"dormant-organ catalogue: {len(entries)} verified organs")
        print("(run `nm organs scan` for the live source-tree "
              "inventory and which of these are still dormant)")
        for e in entries:
            print(_render_entry(e))
    return 0
