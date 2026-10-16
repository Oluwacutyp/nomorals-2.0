"""Focus interruption wiring — focus sessions meet the interruption policy.

``nomorals.focus.sessions.FocusTimer`` owns focus sessions;
``nomorals.autonomy.interruption.assess_worth`` owns the single
interruption policy. This module is the wiring between them: it asks
the policy whether an interruption is worth surfacing, then applies
the focus rule on top.

Focus rule:

* Policy says ``interrupt`` → worth interrupting even during focus
  (safety / owner-direct / genuinely urgent always get through).
* Policy says ``defer`` → worth surfacing when idle, NOT worth
  interrupting a focus session in progress — held for the digest.
* Policy says ``drop`` → never worth it.

"Focus session in progress" is resolved from, in order:

1. an explicit ``focus_active`` keyword,
2. a ``timer`` keyword (a :class:`FocusTimer` — only its public
   ``active()`` API is used),
3. ``context`` carrying ``focus_active`` / ``in_focus`` /
   ``session_active`` / ``timer``,
4. a non-empty ``session_label`` — naming the session that would be
   interrupted means one is in progress.

No policy is imported here: the caller injects ``assess_fn`` (normally
``autonomy.interruption.assess_worth``). No focus-session internals.
"""

from __future__ import annotations

from typing import Any

__all__ = ["should_interrupt"]


def _in_focus(session_label: str, context: Any, timer: Any,
              focus_active: bool | None) -> bool:
    if focus_active is not None:
        return bool(focus_active)
    if timer is not None:
        active = timer.active()
        return active is not None
    if isinstance(context, dict):
        for key in ("focus_active", "in_focus", "session_active"):
            if key in context:
                return bool(context[key])
        ctx_timer = context.get("timer")
        if ctx_timer is not None:
            return ctx_timer.active() is not None
    return bool((session_label or "").strip())


def _as_interruption(interruption: Any) -> Any:
    if isinstance(interruption, str):
        return {"title": interruption, "priority": "normal"}
    return interruption if interruption is not None else {}


def should_interrupt(session_label: str, interruption: Any,
                     context: Any = None, *,
                     timer: Any = None,
                     focus_active: bool | None = None,
                     assess_fn: Any = None) -> dict[str, Any]:
    """Decide whether ``interruption`` should interrupt a focus session.

    The worth judgement comes from the injected ``assess_fn`` — normally
    ``autonomy.interruption.assess_worth``, passed by the caller. This
    module deliberately does NOT import the policy itself: ``focus`` is
    L0 and the policy lives at L5, so a static import would be an upward
    layer violation. Callers without a policy get a safe default
    (``drop`` / not worth it) and a reason saying so.

    Returns ``{worth_it, reason, decision, score, in_focus}``.
    """
    in_focus = _in_focus(session_label, context, timer, focus_active)
    if assess_fn is None:
        return {
            "worth_it": False,
            "reason": ("no interruption policy supplied "
                       "(pass assess_fn=autonomy.interruption.assess_worth)"),
            "decision": "drop",
            "score": 0.0,
            "in_focus": in_focus,
            "policy_reasons": [],
        }
    result = assess_fn(_as_interruption(interruption), context)
    decision = result.get("decision", "drop")
    score = result.get("score", 0.0)
    base_reasons = list(result.get("reasons", []))
    if decision == "interrupt":
        worth_it = True
        reason = ("worth interrupting: policy says interrupt"
                  + (" even during focus" if in_focus else ""))
    elif decision == "defer":
        if in_focus:
            worth_it = False
            reason = ("not worth interrupting during focus: policy says "
                      "defer — held for the digest")
        else:
            worth_it = True
            reason = "worth surfacing (no focus session in progress): policy says defer"
    else:  # drop
        worth_it = False
        reason = "not worth interrupting: policy says drop"
    return {
        "worth_it": worth_it,
        "reason": reason,
        "decision": decision,
        "score": score,
        "in_focus": in_focus,
        "policy_reasons": base_reasons,
    }
