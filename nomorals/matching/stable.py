"""Gale-Shapley stable matching (#83) — two-sided, ~50 lines of real logic.

"One best pairing per day" beats infinite browsing for quality
perception: freelancer ↔ client, mentor ↔ mentee, roommate ↔ roommate.

Proposer-optimal: proposers get their best achievable stable match.
A match is STABLE when no (proposer, reviewer) pair both prefer each
other over their assigned partners (no blocking pairs) —
``verify_stable()`` checks this so tests and callers can prove it.

Unequal sets are fine: the smaller side's leftovers stay unmatched
(``None``). Unknown ids in preference lists are ignored. Never raises.
"""

from __future__ import annotations

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MatchResult",
    "stable_match",
    "verify_stable",
]


class MatchResult(dict):
    """``{proposer_id: reviewer_id | None}`` — dict with a summary."""

    def summary(self) -> str:
        pairs = [f"{p} ↔ {r}" for p, r in self.items() if r]
        unmatched = [p for p, r in self.items() if not r]
        lines = ["🤝 stable matches (%d):" % len(pairs)]
        lines += ["  " + p for p in pairs] or []
        if unmatched:
            lines.append("unmatched: " + ", ".join(unmatched))
        return "\n".join(lines)


def stable_match(
    proposers: list[str],
    reviewers: list[str],
    proposer_prefs: dict[str, list[str]],
    reviewer_prefs: dict[str, list[str]],
) -> MatchResult:
    """Gale-Shapley (proposer-optimal). Pure; never raises.

    Each preference list is most-preferred first; ids not in the other
    set are ignored. Returns ``{proposer: reviewer | None}``.
    """
    try:
        props = [p for p in (proposers or []) if p]
        revs = [r for r in (reviewers or []) if r]
        rev_set = set(revs)

        # Cleaned proposer preference lists (most-preferred first).
        # A proposer with no stated preferences is indifferent: he will
        # propose to everyone rather than sit out.
        prefs: dict[str, list[str]] = {}
        for p in props:
            seen: list[str] = []
            for r in (proposer_prefs or {}).get(p) or []:
                if r in rev_set and r not in seen:
                    seen.append(r)
            if not seen:
                seen = list(revs)
            prefs[p] = seen

        # Reviewer rank: lower = more preferred (unlisted = worst).
        rank: dict[str, dict[str, int]] = {}
        for r in revs:
            order = (reviewer_prefs or {}).get(r) or []
            rank[r] = {p: i for i, p in enumerate(order) if p in set(props)}

        free = list(props)              # proposers still looking
        next_i = {p: 0 for p in props}  # next reviewer index to propose to
        engaged: dict[str, str] = {}    # reviewer -> proposer

        while free:
            p = free.pop(0)
            if next_i[p] >= len(prefs[p]):
                continue  # exhausted his list — stays unmatched
            r = prefs[p][next_i[p]]
            next_i[p] += 1
            current = engaged.get(r)
            if current is None:
                engaged[r] = p
            elif rank[r].get(p, 10**9) < rank[r].get(current, 10**9):
                engaged[r] = p
                free.append(current)  # dumped proposer proposes again
            else:
                free.append(p)  # rejected — proposes to the next

        matched_rev = {p: r for r, p in engaged.items()}
        return MatchResult({p: matched_rev.get(p) for p in props})
    except Exception:  # noqa: BLE001 — matching must never break the chat
        _log.warning("matching.stable: stable_match failed", exc_info=True)
        return MatchResult({p: None for p in (proposers or []) if p})


def verify_stable(
    result: MatchResult,
    proposer_prefs: dict[str, list[str]],
    reviewer_prefs: dict[str, list[str]],
) -> tuple[bool, list[tuple[str, str]]]:
    """Check for blocking pairs. Returns (is_stable, blockers).

    A blocking pair is (proposer, reviewer) who both strictly prefer
    each other over their assigned partners. Pure; never raises.
    """
    try:
        assigned_rev = {p: r for p, r in result.items() if r}
        assigned_prop = {r: p for p, r in assigned_rev.items()}

        def rank_of(prefs: dict[str, list[str]], who: str, other: str) -> int:
            order = prefs.get(who) or []
            try:
                return order.index(other)
            except ValueError:
                return 10**9  # unlisted = least preferred

        blockers: list[tuple[str, str]] = []
        for p, r in result.items():
            for r2 in (reviewer_prefs or {}):
                if r2 == r:
                    continue
                # Does p prefer r2 over his current match?
                p_prefers = (
                    r is None
                    or rank_of(proposer_prefs, p, r2) < rank_of(proposer_prefs, p, r)
                )
                if not p_prefers:
                    continue
                # Does r2 prefer p over her current match?
                current = assigned_prop.get(r2)
                r_prefers = (
                    current is None
                    or rank_of(reviewer_prefs, r2, p)
                    < rank_of(reviewer_prefs, r2, current)
                )
                if r_prefers:
                    blockers.append((p, r2))
        return (not blockers, blockers)
    except Exception:  # noqa: BLE001
        _log.warning("matching.stable: verify_stable failed", exc_info=True)
        return (False, [])
