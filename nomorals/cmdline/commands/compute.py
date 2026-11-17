"""``nm compute`` — computed answers from the console.

    nm compute 2 + 3 * 4
    nm compute "integrate x^2" [--rung sympy]
    nm compute "5 miles to km" [--json]
    nm compute "population of Lagos in 1990"

The same ladder chat /compute uses: WolframAlpha when an AppID is
configured (NM_WOLFRAM_APPID), then the sympy CAS, then exact
stdlib arithmetic/units/dates. An unanswerable question prints the
honest reason and exits 1 — never a guessed number.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_compute"]


def _appid(context: Any) -> str:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return ""
    wolfram = getattr(settings, "wolfram", None)
    if wolfram is None:
        return ""
    return str(getattr(wolfram, "appid", "") or "")


def _cmd_compute(args: argparse.Namespace, context: Any) -> int:
    from ...codews.compute import compute_answer, format_compute_reply

    question = " ".join(args.question or []).strip()
    if not question:
        print("compute: give me a question — "
              "nm compute \"integrate x^2\"", file=sys.stderr)
        return 2
    try:
        comp = compute_answer(
            question, appid=_appid(context) or None,
            rung=getattr(args, "rung", "auto") or "auto")
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        print(f"compute: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps({
            "ok": comp.ok,
            "question": comp.query,
            "result": comp.result,
            "rung": comp.rung,
            "steps": list(comp.steps),
            "source": comp.source,
            "pods": comp.pods,
            "assumptions": list(comp.assumptions),
            "error": comp.error,
        }, indent=2))
    else:
        print(format_compute_reply(comp))
    return 0 if comp.ok else 1
