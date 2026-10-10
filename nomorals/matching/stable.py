"""Stable matching mechanisms (#83) — two-sided, one-pool, and allocation.

"One best pairing per day" beats infinite browsing for quality perception:
freelancer ↔ client, mentor ↔ mentee, roommate ↔ roommate.

Mechanisms (every practical one sacrifices something — pick deliberately):

- ``stable_match`` — Gale–Shapley deferred acceptance, 1:1, proposer-optimal.
  Stable: no blocking pairs (``verify_stable`` proves it). This is the
  mechanism behind Hinge's "Most Compatible" (Gale–Shapley + learned
  preferences, refreshed daily, 8x dates in trials).
- ``stable_match_capacities`` — Hospital–Resident deferred acceptance: one
  reviewer takes up to ``capacity`` proposers (a mentor takes N mentees).
  Reference API: daffidwilde/matching's ``HospitalResident``.
- ``roommate_match`` — Irving (1985) two-phase algorithm for a SINGLE pool
  (member ↔ member community pairing). Unlike SM, a stable pairing may not
  exist: the result says so honestly instead of fabricating one.
- ``top_trading_cycles`` — Shapley–Scarf (1974) housing market: allocates
  indivisible goods (who gets which gig). Strategy-proof + Pareto efficient,
  but NOT stable — the efficiency extreme of the tradeoff.
- ``serial_dictatorship`` — agents pick in a fixed (or random) order.
  Strategy-proof, Pareto efficient, dead simple; the order is the fairness
  question (RSD = random order).
- ``optimal_assignment`` — Hungarian (Kuhn–Munkres) max-total assignment
  for one-sided score matrices. Pure Python, no dependencies.

Never raises. Pure functions; the chat layer owns persistence.
"""

from __future__ import annotations

from collections import deque
from math import inf

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MatchResult",
    "CapacityMatchResult",
    "RoommateResult",
    "stable_match",
    "stable_match_capacities",
    "verify_stable",
    "roommate_match",
    "verify_roommate_stable",
    "top_trading_cycles",
    "serial_dictatorship",
    "optimal_assignment",
]


class MatchResult(dict):
    """``{proposer_id: reviewer_id | None}`` — dict with a summary."""

    def summary(self) -> str:
        pairs = [(p, r) for p, r in self.items() if r]
        unmatched = [p for p, r in self.items() if not r]
        lines = ["🤝 stable matches (%d):" % len(pairs)]
        lines += ["  %d. %s ↔ %s" % (i, p, r)
                  for i, (p, r) in enumerate(pairs, 1)]
        if unmatched:
            lines.append("unmatched: " + ", ".join(unmatched))
        return "\n".join(lines)


class CapacityMatchResult(dict):
    """``{proposer_id: reviewer_id | None}`` with per-reviewer rosters."""

    def __init__(self, *args, capacities: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacities = dict(capacities or {})

    def rosters(self) -> dict[str, list[str]]:
        """``{reviewer: [proposers]}`` for matched reviewers."""
        out: dict[str, list[str]] = {}
        for p, r in self.items():
            if r:
                out.setdefault(r, []).append(p)
        return out

    def summary(self) -> str:
        rosters = self.rosters()
        total = sum(len(v) for v in rosters.values())
        lines = ["🤝 stable matches (%d across %d reviewers):"
                 % (total, len(rosters))]
        for r in sorted(rosters):
            cap = self.capacities.get(r)
            fill = (" (%d/%d filled)" % (len(rosters[r]), cap)
                    if cap else "")
            lines.append("  %s%s: %s" % (r, fill, ", ".join(rosters[r])))
        unmatched = [p for p, r in self.items() if not r]
        if unmatched:
            lines.append("unmatched: " + ", ".join(unmatched))
        return "\n".join(lines)


class RoommateResult(dict):
    """``{person: partner | None}`` — ``.stable`` tells the truth.

    Irving (1985) proved some single-pool instances admit NO stable
    pairing. When that happens the dict holds ``None`` values and
    ``stable`` is False — an honest report, never a fabricated pairing.
    """

    def __init__(self, *args, stable: bool = True,
                 sat_out: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stable = stable
        self.sat_out = list(sat_out or [])

    def pairs(self) -> list[tuple[str, str]]:
        seen: set[str] = set()
        out = []
        for p, q in self.items():
            if q and p not in seen and q not in seen:
                out.append((p, q))
                seen.add(p)
                seen.add(q)
        return out

    def summary(self) -> str:
        if not self.stable:
            return ("🙅 no stable pairing exists for this pool — "
                    "Irving's algorithm eliminated every option honestly.")
        lines = ["🤝 roommate pairs (%d):" % len(self.pairs())]
        lines += ["  %d. %s ↔ %s" % (i, p, q)
                  for i, (p, q) in enumerate(self.pairs(), 1)]
        unmatched = [p for p, r in self.items() if not r]
        if unmatched:
            lines.append("unmatched: " + ", ".join(unmatched))
        if self.sat_out:
            lines.append("sat out (odd pool): " + ", ".join(self.sat_out))
        return "\n".join(lines)


def _clean_prefs(ids: list[str], prefs: dict | None,
                 valid: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for i in ids:
        seen: list[str] = []
        for other in (prefs or {}).get(i) or []:
            if other in valid and other not in seen:
                seen.append(other)
        out[i] = seen
    return out


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
        prefs = _clean_prefs(props, proposer_prefs, rev_set)
        for p in props:
            if not prefs[p]:
                prefs[p] = list(revs)

        # Reviewer rank: lower = more preferred (unlisted = worst).
        prop_set = set(props)
        rank: dict[str, dict[str, int]] = {}
        for r in revs:
            order = (reviewer_prefs or {}).get(r) or []
            rank[r] = {p: i for i, p in enumerate(order) if p in prop_set}

        free: deque[str] = deque(props)  # proposers still looking
        next_i = {p: 0 for p in props}   # next reviewer index to propose to
        engaged: dict[str, str] = {}     # reviewer -> proposer

        while free:
            p = free.popleft()
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


def stable_match_capacities(
    proposers: list[str],
    reviewers: list[str],
    proposer_prefs: dict[str, list[str]],
    reviewer_prefs: dict[str, list[str]],
    capacities: dict[str, int] | None = None,
) -> CapacityMatchResult:
    """Hospital–Resident deferred acceptance (proposer-proposing).

    Each reviewer accepts up to ``capacities[reviewer]`` proposers
    (default 1 — a mentor with room for 3 mentees sets 3). Still
    proposer-optimal and stable among filled slots. Pure; never raises.
    """
    try:
        props = [p for p in (proposers or []) if p]
        revs = [r for r in (reviewers or []) if r]
        rev_set = set(revs)
        caps = {r: max(1, int((capacities or {}).get(r, 1))) for r in revs}

        prefs = _clean_prefs(props, proposer_prefs, rev_set)
        for p in props:
            if not prefs[p]:
                prefs[p] = list(revs)

        prop_set = set(props)
        rank: dict[str, dict[str, int]] = {}
        for r in revs:
            order = (reviewer_prefs or {}).get(r) or []
            rank[r] = {p: i for i, p in enumerate(order) if p in prop_set}

        free: deque[str] = deque(props)
        next_i = {p: 0 for p in props}
        held: dict[str, list[str]] = {r: [] for r in revs}  # reviewer -> ps

        while free:
            p = free.popleft()
            if next_i[p] >= len(prefs[p]):
                continue
            r = prefs[p][next_i[p]]
            next_i[p] += 1
            roster = held[r]
            if len(roster) < caps[r]:
                roster.append(p)
            else:
                # Keep the best `cap` by the reviewer's ranking.
                contenders = roster + [p]
                contenders.sort(key=lambda x: rank[r].get(x, 10**9))
                kept, bumped = contenders[: caps[r]], contenders[caps[r]:]
                held[r] = kept
                free.extend(bumped)

        matched = {p: r for r, ps in held.items() for p in ps}
        return CapacityMatchResult(
            {p: matched.get(p) for p in props}, capacities=caps)
    except Exception:  # noqa: BLE001
        _log.warning("matching.stable: stable_match_capacities failed",
                     exc_info=True)
        return CapacityMatchResult(
            {p: None for p in (proposers or []) if p},
            capacities=dict(capacities or {}))


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


# ── Stable roommates (Irving 1985) ──────────────────────────────────
#
# Structure follows the reference implementation in daffidwilde/matching
# (MIT; JOSS 10.21105/joss.02169): phase 1 does the table reduction INLINE
# during proposals — a new proposer always outranks whoever the recipient
# currently holds (anyone ranked lower was already deleted as a
# successor), so acceptance is unconditional. Phase 2 eliminates
# all-or-nothing cycles with the GI89 deletion rule (Gusfield & Irving
# 1989, §4.2.3): for rotation (x_i, y_i), delete from y_i's list every
# successor of x_{i-1} — deleting only the cycle pairs themselves is
# insufficient and can strand conflicting pairs.


def _delete_pair(lists: dict[str, list[str]], a: str, b: str) -> None:
    if b in lists.get(a, []):
        lists[a].remove(b)
    if a in lists.get(b, []):
        lists[b].remove(a)


def _find_rotation(
    lists: dict[str, list[str]], start: str
) -> list[tuple[str, str]] | None:
    """Locate an all-or-nothing cycle (rotation) from ``start``.

    Traversal: y = second(x), w = last(y), repeat. Returns pairs
    (x_i, y_i) with y_i = second(x_{i-1}) and x_i = last(y_i).
    """
    lasts = [start]
    seconds: list[str] = []
    x = start
    while True:
        if len(lists.get(x, [])) < 2:
            return None
        second_best = lists[x][1]
        if not lists.get(second_best, []):
            return None
        their_worst = lists[second_best][-1]
        seconds.append(second_best)
        lasts.append(their_worst)
        x = their_worst
        if lasts.count(x) > 1:
            break
    idx = lasts.index(x)
    return list(zip(lasts[idx + 1:], seconds[idx:]))


def _rotation_deletions(
    lists: dict[str, list[str]], cycle: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """GI89 deletion set for a rotation: for each (x_i, y_i), delete from
    y_i's list all successors of x_{i-1} (indices mod r)."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    r = len(cycle)
    for i, (_, right) in enumerate(cycle):
        left = cycle[(i - 1) % r][0]
        if left not in lists.get(right, []):
            continue
        for s in lists[right][lists[right].index(left) + 1:]:
            key = tuple(sorted((right, s)))
            if key not in seen:
                seen.add(key)
                pairs.append((right, s))
    return pairs


def roommate_match(
    people: list[str],
    prefs: dict[str, list[str]],
    *,
    allow_odd: bool = True,
) -> RoommateResult:
    """Irving's stable-roommates algorithm on ONE pool. Pure; never raises.

    ``prefs`` maps each person to the others most-preferred first
    (incomplete lists allowed — unlisted people are unacceptable).
    Returns ``RoommateResult``: ``{person: partner | None}`` with
    ``.stable`` False when no stable pairing exists (Irving proved some
    instances are unsolvable — we report it, not fabricate it).

    Odd pools: with ``allow_odd`` the least-approved person sits out
    (documented, deterministic); otherwise the run reports no stable
    pairing.
    """
    try:
        folks = [p for p in (people or []) if p]
        folks = list(dict.fromkeys(folks))
        sat_out: list[str] = []
        if len(folks) % 2 == 1:
            if not allow_odd or len(folks) < 3:
                return RoommateResult({p: None for p in folks}, stable=False,
                                      sat_out=sat_out)
            # Least-approved sits out: worst mean rank given by others.
            n = len(folks)
            rank_of: dict[str, dict[str, int]] = {}
            for p in folks:
                order = (prefs or {}).get(p) or []
                rank_of[p] = {q: i for i, q in enumerate(order) if q != p}
            def approval(q: str) -> float:
                total, cnt = 0, 0
                for p in folks:
                    if p != q:
                        total += rank_of[p].get(q, n)
                        cnt += 1
                return total / max(1, cnt)
            drop = max(folks, key=lambda q: (approval(q), folks.index(q)))
            sat_out = [drop]
            folks = [p for p in folks if p != drop]

        # Mutual pruning: x lists y only if y lists x.
        lists: dict[str, list[str]] = {}
        for p in folks:
            seen = []
            for q in (prefs or {}).get(p) or []:
                if q in folks and q != p and q not in seen:
                    seen.append(q)
            lists[p] = seen
        for p in folks:
            lists[p] = [q for q in lists[p]
                        if p in ((prefs or {}).get(q) or [])]

        # ── Phase 1: proposals with INLINE reduction ──
        # Invariant: when x proposes to y = first(x's list), any current
        # holder h of y is ranked below x (h's successors on y's list
        # were deleted when h proposed), so y always takes x.
        holder: dict[str, str | None] = {p: None for p in folks}
        free: deque[str] = deque(folks)
        in_free = set(folks)

        def fail() -> RoommateResult:
            return RoommateResult({p: None for p in folks}, stable=False,
                                  sat_out=sat_out)

        while free:
            x = free.popleft()
            in_free.discard(x)
            if not lists[x]:
                return fail()  # rejected by everyone: unsolvable
            y = lists[x][0]
            h = holder[y]
            if h is not None and h != x:
                holder[y] = None
                if h not in in_free:
                    free.append(h)
                    in_free.add(h)
            holder[y] = x
            for z in list(lists[y][lists[y].index(x) + 1:]):
                _delete_pair(lists, z, y)
                if not lists[z] and z in in_free:
                    in_free.discard(z)
                    try:
                        free.remove(z)
                    except ValueError:
                        pass
        if any(not lists[p] for p in folks):
            return fail()

        # ── Phase 2: rotation elimination ──
        while True:
            start = next((p for p in folks if len(lists[p]) > 1), None)
            if start is None:
                break
            cycle = _find_rotation(lists, start)
            if not cycle:
                return fail()
            for a, b in _rotation_deletions(lists, cycle):
                _delete_pair(lists, a, b)
            if any(not lists[p] for p in folks):
                return fail()

        if any(len(lists[p]) != 1 for p in folks):
            return fail()
        pairs = {p: lists[p][0] for p in folks}
        # Mutuality guard (stable-table invariant should guarantee it).
        if any(pairs.get(pairs[p]) != p for p in folks):
            return fail()
        return RoommateResult(pairs, stable=True, sat_out=sat_out)
    except Exception:  # noqa: BLE001
        _log.warning("matching.stable: roommate_match failed", exc_info=True)
        return RoommateResult({p: None for p in (people or []) if p},
                              stable=False)


def verify_roommate_stable(
    pairs: dict[str, str | None],
    prefs: dict[str, list[str]],
) -> tuple[bool, list[tuple[str, str]]]:
    """Blocking-pair check for a single-pool pairing. Pure; never raises."""
    try:
        partner = {p: q for p, q in (pairs or {}).items() if q}

        def rank_of(who: str, other: str) -> float:
            order = (prefs or {}).get(who) or []
            try:
                return order.index(other)
            except ValueError:
                return inf  # unacceptable

        folks = list((pairs or {}).keys())
        blockers = []
        for i, a in enumerate(folks):
            for b in folks[i + 1:]:
                if partner.get(a) == b:
                    continue
                pa, pb = partner.get(a), partner.get(b)
                a_wants = (rank_of(a, b) < rank_of(a, pa)
                           if pa else rank_of(a, b) < inf)
                b_wants = (rank_of(b, a) < rank_of(b, pb)
                           if pb else rank_of(b, a) < inf)
                if a_wants and b_wants:
                    blockers.append((a, b))
        return (not blockers, blockers)
    except Exception:  # noqa: BLE001
        _log.warning("matching.stable: verify_roommate_stable failed",
                     exc_info=True)
        return (False, [])


# ── Allocation mechanisms ───────────────────────────────────────────


def top_trading_cycles(
    agents: list[str],
    prefs: dict[str, list[str]],
    endowments: dict[str, str],
) -> dict[str, str | None]:
    """Shapley–Scarf (1974) Top Trading Cycles. Pure; never raises.

    Each agent owns one item (``endowments``); agents point to the owner
    of their most-preferred remaining item; cycles trade; repeat. The
    result is the unique core allocation: strategy-proof and Pareto
    efficient, but NOT stable (no blocking-pair concept here — this is
    the efficiency extreme). Use for allocating indivisible goods:
    who gets which gig / learning slot.

    An agent with no acceptable remaining item keeps their endowment.
    """
    try:
        remaining = [a for a in (agents or [])
                     if a and a in (endowments or {})]
        remaining = list(dict.fromkeys(remaining))
        owner = {item: a for a, item in endowments.items()
                 if a in remaining}
        items_left = set(owner)
        result: dict[str, str | None] = {}
        while remaining:
            # Agents with no acceptable remaining item sit out and keep
            # their endowment — recomputed each round so `points` below
            # never references a removed agent.
            tradable: list[str] = []
            for a in remaining:
                top = None
                for item in (prefs or {}).get(a) or []:
                    if item in items_left:
                        top = item
                        break
                if top is None:
                    result[a] = endowments[a]
                    items_left.discard(endowments[a])
                else:
                    tradable.append(a)
            remaining = tradable
            if not remaining:
                break
            points = {a: owner[next(
                item for item in (prefs or {}).get(a) or []
                if item in items_left)] for a in remaining}
            # Find a cycle in the functional graph `points` (one exists).
            start = remaining[0]
            path: list[str] = []
            seen: dict[str, int] = {}
            cur: str | None = start
            while cur is not None and cur not in seen:
                seen[cur] = len(path)
                path.append(cur)
                cur = points[cur]
            cycle = path[seen[cur]:] if cur in seen else [start]
            for a in cycle:
                item = next(
                    i for i in (prefs or {}).get(a) or []
                    if i in items_left and owner[i] == points[a])
                result[a] = item
                items_left.discard(item)
                remaining.remove(a)
        return result
    except Exception:  # noqa: BLE001
        _log.warning("matching.stable: top_trading_cycles failed",
                     exc_info=True)
        return {a: None for a in (agents or []) if a}


def serial_dictatorship(
    agents: list[str],
    prefs: dict[str, list[str]],
    items: list[str],
    order: list[str] | None = None,
    seed: int | None = None,
) -> dict[str, str | None]:
    """Agents pick their top remaining item in ``order``.

    Strategy-proof and Pareto efficient; the ORDER is the fairness
    question — pass ``seed`` for Random Serial Dictatorship.
    An agent with no stated preferences takes the first remaining item;
    agents past the items run out get None. Pure; never raises.
    """
    try:
        import random as _random

        folks = [a for a in (agents or []) if a]
        folks = list(dict.fromkeys(folks))
        stock = [i for i in (items or []) if i]
        stock = list(dict.fromkeys(stock))
        if order:
            seq = [a for a in order if a in folks]
            seq += [a for a in folks if a not in seq]
        else:
            rng = _random.Random(seed)
            seq = folks[:]
            rng.shuffle(seq)
        result: dict[str, str | None] = {}
        for a in seq:
            pick = None
            for item in (prefs or {}).get(a) or []:
                if item in stock:
                    pick = item
                    break
            if pick is None and stock:
                pick = stock[0]  # indifferent — first remaining
            result[a] = pick
            if pick in stock:
                stock.remove(pick)
        for a in folks:
            result.setdefault(a, None)
        return result
    except Exception:  # noqa: BLE001
        _log.warning("matching.stable: serial_dictatorship failed",
                     exc_info=True)
        return {a: None for a in (agents or []) if a}


def _hungarian_min(cost: list[list[float]]) -> list[int]:
    """O(n³) Hungarian for min-cost, n rows ≤ m cols. Returns col per row."""
    n, m = len(cost), len(cost[0])
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assign[p[j] - 1] = j - 1
    return assign


def optimal_assignment(
    scores: dict[str, dict[str, float]],
) -> dict[str, str | None]:
    """Max-total-score one-sided assignment (Hungarian, Kuhn–Munkres).

    ``scores[row][col]`` → ``{row: col | None}`` maximizing the summed
    score. Rectangular matrices are padded with zero-score dummies;
    unmatched rows get None. This is the EFFICIENCY extreme — unlike
    ``stable_match`` it can produce blocking pairs; say so when you
    surface it. Pure Python, no dependencies. Pure; never raises.
    """
    try:
        rows = [r for r in (scores or {}) if r]
        cols: list[str] = []
        for r in rows:
            for c in (scores[r] or {}):
                if c and c not in cols:
                    cols.append(c)
        if not rows or not cols:
            return {r: None for r in rows}
        n, m = len(rows), len(cols)
        # Hungarian needs n ≤ m; transpose the problem if not.
        transposed = n > m
        rr, cc = (cols, rows) if transposed else (rows, cols)
        size = max(len(rr), len(cc))
        cost = [[0.0] * size for _ in range(len(rr))]
        for i, a in enumerate(rr):
            for j, b in enumerate(cc):
                try:
                    s = float((scores[a] or {}).get(b, 0.0)
                              if not transposed
                              else (scores[b] or {}).get(a, 0.0))
                except (TypeError, ValueError):
                    s = 0.0
                cost[i][j] = -s  # negate: min-cost solver, max-score goal
        assign = _hungarian_min(cost)
        out: dict[str, str | None] = {}
        if transposed:
            # assign[j] = index into cc (= real rows) for rr[j] (= real col j)
            row_to_col: dict[str, str] = {}
            for j, i in enumerate(assign):
                if 0 <= i < n and j < m:
                    row_to_col[cc[i]] = rr[j]
            for r in rows:
                out[r] = row_to_col.get(r)
        else:
            for i, r in enumerate(rows):
                j = assign[i]
                out[r] = cols[j] if 0 <= j < m else None
        return out
    except Exception:  # noqa: BLE001
        _log.warning("matching.stable: optimal_assignment failed",
                     exc_info=True)
        return {r: None for r in (scores or {}) if r}
