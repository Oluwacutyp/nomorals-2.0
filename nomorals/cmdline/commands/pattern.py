"""``nm pattern`` — pattern study: plain-words market patterns proven
against history, watched live (analysis only)."""

from __future__ import annotations

from typing import Any

from ..emit import _emit


def _cmd_pattern(args: Any, context: Any) -> int:
    """Route `nm pattern` to the pattern-study organ."""
    from ...agents.marketdesk.patterns import (
        check_watches,
        list_patterns,
        pattern_status,
        remove_pattern,
        render_check,
        render_pattern_list,
        render_status,
        render_study,
        set_watching,
        study_rule,
    )

    settings = getattr(context, "settings", None)
    action = args.pattern_action
    if action == "study":
        rule = " ".join(getattr(args, "rule", []) or [])
        rep = study_rule(rule, settings=settings)
        _emit(args, rep, render_study(rep))
        return 0 if rep.get("ok") else 1
    if action == "list":
        rep = list_patterns(settings=settings)
        _emit(args, rep, render_pattern_list(rep))
        return 0
    if action in ("watch", "unwatch"):
        rep = set_watching(args.pattern_id, action == "watch",
                           settings=settings)
        if not rep.get("ok"):
            _emit(args, rep, f"pattern {action} failed: {rep.get('error')}")
            return 1
        p = rep["pattern"]
        _emit(args, rep,
              f"pattern {p['id']} — watching "
              f"{'on' if p['watching'] else 'off'} ({p['name']})")
        return 0
    if action == "check":
        rep = check_watches(settings=settings)
        _emit(args, rep, render_check(rep))
        return 0
    if action == "status":
        pid = (getattr(args, "pattern_id", "") or "").strip()
        if not pid:
            rep = list_patterns(settings=settings)
            _emit(args, rep, render_pattern_list(rep))
            return 0
        rep = pattern_status(pid, settings=settings)
        _emit(args, rep, render_status(rep))
        return 0 if rep.get("ok") else 1
    if action == "remove":
        rep = remove_pattern(args.pattern_id, settings=settings)
        if not rep.get("ok"):
            _emit(args, rep, f"pattern remove failed: {rep.get('error')}")
            return 1
        _emit(args, rep, f"forgot pattern {rep['removed']}")
        return 0
    return 1
