"""``nm why`` — explain autonomous firings; run the unified scheduler."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..emit import _emit


def _cmd_why(args: argparse.Namespace, context: Any) -> int:
    """``nm why [firing-id]`` / ``nm why run``.

    The autonomy cluster's owner surface (AA-35): the stack is
    assembled by :mod:`nomorals.autonomy.wiring` under the nomorals
    home — the REAL OwnerRhythm / BudgetGovernor / WhyStore /
    PreemptionLog / UsageTracker around the UnifiedScheduler.

    * ``nm why`` — why the last autonomous firing happened.
    * ``nm why <id>`` — the explanation for one firing.
    * ``nm why run`` — run every due unified mission/tick once
      (rhythm + budget gates apply; every firing explains itself into
      the why store, so a later ``nm why`` can answer for it).
    """
    from ...autonomy import wiring
    from ...autonomy.why import why_command

    home = getattr(getattr(context, "settings", None), "home", None)
    if home is None:
        print("why: no nomorals home on this context", file=sys.stderr)
        return 1
    stack = wiring.build_stack(home)
    words = [str(w) for w in (getattr(args, "firing", None) or [])]

    if words and words[0] == "run":
        results = stack.scheduler.run_due()
        if not results:
            _emit(args, {"results": []},
                  "unified scheduler: nothing due")
            return 0
        lines = [f"unified scheduler: {len(results)} firing(s)"]
        for r in results:
            label = r.get("name") or r.get("id") or "?"
            reason = f" — {r['reason']}" if r.get("reason") else ""
            err = f" — {r['error']}" if r.get("error") else ""
            lines.append(f"  {r.get('kind', '?')} {label}: "
                         f"{r.get('status', '?')}{reason}{err}")
        _emit(args, {"results": results}, "\n".join(lines))
        return 0

    firing_id = words[0] if words else None
    text = why_command(stack.why_store, firing_id)
    record = (stack.why_store.why_for(firing_id) if firing_id
              else stack.why_store.why_last())
    _emit(args, record, text)
    return 0
