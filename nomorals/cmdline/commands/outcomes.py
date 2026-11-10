"""``nm outcomes`` — outcome stewardship at the CLI (doc 15 P6).

Parity surface for chat /cases: the same board ordering
(:func:`case_board`), the same renders, the same acts — status,
list, show, verify (the steward's real investigation cycle for the
case), ack (a recorded acknowledgement; never a resolution).
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _cmd_outcomes(args: argparse.Namespace, context: Any) -> int:
    from ...agents.delegation import DelegationStore
    from ...agents.outcome_cases import OutcomeCaseStore
    from ...agents.outcome_steward import (
        OutcomeSteward,
        adapter_health,
        case_board,
        inspect_case,
        render_case_board,
        render_case_detail,
    )

    as_json = getattr(args, "json", False)
    action = args.outcomes_action
    db = context.db
    store = OutcomeCaseStore(db)

    if action == "status":
        payload = {"counts": store.counts(),
                   "adapters": adapter_health()}
        if as_json:
            print(json.dumps(payload, indent=2, default=str))
        else:
            counts = payload["counts"]
            print("outcome cases: "
                  + " · ".join(f"{k} {v}" for k, v in
                               sorted(counts.items())))
            for row in payload["adapters"]:
                state = "working" if row.get("status") == "working" \
                    else "unavailable"
                line = f"  {row['domain']}: {state}"
                if row.get("note"):
                    line += f" — {row['note']}"
                print(line)
        return 0

    if action == "list":
        if as_json:
            payload = [inspect_case(store, DelegationStore(db), c.id)
                       for c in case_board(store)]
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(render_case_board(store))
        return 0

    board = case_board(store)
    n = int(args.n)
    if not 1 <= n <= len(board):
        print(f"case #{n} isn't on the board — {len(board)} case(s) "
              "listed")
        return 1
    case = board[n - 1]

    if action == "show":
        if as_json:
            print(json.dumps(
                inspect_case(store, DelegationStore(db), case.id),
                indent=2, default=str))
        else:
            contract = None
            if case.contract_id:
                record = DelegationStore(db).get_contract(
                    case.contract_id)
                if record is not None:
                    contract = DelegationStore(db).summary(record)
            print(render_case_detail(case, store.receipts(case.id),
                                     contract))
        return 0

    if action == "verify":
        if case.state in ("resolved", "closed-unknown"):
            print(f"that case is already closed ({case.state}) — a "
                  "closed-unknown case reopens on its own when the "
                  "work is seen again")
            return 0
        result = OutcomeSteward(db).run_case(case.id)
        if as_json:
            print(json.dumps(result, indent=2, default=str))
        else:
            outcome = result.get("outcome", "")
            if outcome == "resolved":
                print(f"re-checked — resolved: "
                      f"{result.get('verdict', '')} on the cited "
                      "evidence")
            elif outcome == "closed-unknown":
                print("re-checked — the outcome is still honestly "
                      "unknown; the checks are recorded on the case")
            else:
                shown = outcome or "incomplete"
                print(f"re-check ran — outcome: {shown}; the case "
                      "stays open")
        return 0

    if action == "ack":
        before = (case.state, case.verdict)
        store.acknowledge(case.id, actor="owner")
        after = store.get(case.id)
        assert (after.state, after.verdict) == before
        if as_json:
            print(json.dumps({"acknowledged": after.id,
                              "state": after.state,
                              "verdict": after.verdict},
                             indent=2, default=str))
        else:
            print(f"acknowledged — {after.subject} recorded as seen "
                  f"({after.state}). Acknowledgement does not "
                  "resolve it.")
        return 0

    print(f"unknown outcomes action {action!r}")
    return 2
