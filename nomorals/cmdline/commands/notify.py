"""``nm notify`` — notification decision inspection (doc 15 P6).

Parity surface for chat /notify decisions | why | coverage: the
same renders from agents/inspection, over the same journals.
``why <n>`` takes a DECISION number (from ``nm notify decisions``)
— the CLI has no stored-notification board of its own, and the
decision journal is the view that includes suppressed and held
alerts.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _cmd_notify(args: argparse.Namespace, context: Any) -> int:
    from ...agents import inspection
    from ...agents.alert_policy import AlertPolicy

    as_json = getattr(args, "json", False)
    action = getattr(args, "notify_action", None) or "decisions"
    db = context.db

    if action == "coverage":
        coverage = AlertPolicy.candidate_coverage(db)
        if as_json:
            print(json.dumps(coverage, indent=2, default=str))
        else:
            print(inspection.render_coverage(coverage))
        return 0

    if action == "decisions":
        entries = inspection.decision_entries(db, limit=int(args.n))
        if as_json:
            print(json.dumps(entries, indent=2, default=str))
        else:
            print(inspection.render_decisions(entries))
        return 0

    if action == "why":
        entries = inspection.decision_entries(db, limit=50)
        n = int(args.n)
        if not 1 <= n <= len(entries):
            print(f"decision #{n} isn't on the board — "
                  f"{len(entries)} decision(s) recorded")
            return 1
        row = inspection.candidate_row_for(
            db, entries[n - 1].get("event_id", ""))
        if row is None:
            print("that decision's evidence record has aged out of "
                  "the journal — the decision itself is on "
                  "nm notify decisions")
            return 1
        if as_json:
            print(json.dumps(row, indent=2, default=str))
        else:
            print(inspection.render_why(row))
        return 0

    print(f"unknown notify action {action!r}")
    return 2
