"""Churn / growth / topic-cluster analytics for group health.

Pure functions over caller-supplied data — no I/O, no store, no
owner data. The caller passes the member ledger and the recorded
messages it already holds; this module only aggregates them.

* Churn + growth — joins / leaves / net change / churn rate /
  growth rate over the current window AND the window before it,
  so the trend is a comparison of two measured windows, never a
  vibe. Rates are over the members present at the window start
  (current members, plus the ones who left inside the window,
  minus the ones who joined inside it) — a number recomputable
  from the ledger. No baseline (nobody was a member at the
  window start) → the rate is None, never zero — "no data"
  differs from "nobody left".
* Topic clusters — what the group actually talks about, clustered
  from the recorded message text. Terms that appear in at least
  ``min_messages`` messages are significant; terms that co-occur
  in at least two messages link into one topic; messages are
  assigned to the topic they share the most terms with. Empty
  input → no clusters; a message with no significant terms is
  unclustered, never forced into a topic. Deterministic: the same
  messages always give the same clusters, in the same order.

Inputs accept dicts (``{"joined_ts", "left_ts", "is_bot",
"deleted"}`` for members; ``{"sender", "text", "ts"}`` for
messages) or objects with the same attributes, so callers can pass
their own group shapes without this module importing them.

Isolation: stdlib only. No owner-data imports.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

__all__ = [
    "WINDOW_DAYS",
    "churn_and_growth",
    "topic_clusters",
]

#: The churn/activity window, in days — the same window the
#: /ghealth report already uses.
WINDOW_DAYS = 7

#: A term must appear in this many messages to count as a topic
#: term at all (a word said once is not a topic).
MIN_TERM_MESSAGES = 2
#: Two terms link into one topic only when they co-occur in at
#: least this many messages — one bridge message never merges
#: two topics.
MIN_CO_OCCURRENCE = 2
#: Clusters shown, most messages first.
CLUSTER_LIMIT = 5

_STOPWORDS = frozenset([
    "a", "about", "above", "after", "again", "against", "all", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "like", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
    "hello", "hi", "hey", "good", "morning", "afternoon", "evening",
    "everyone", "everybody", "guys", "please", "thanks", "thank",
    "lol", "haha", "yeah", "yes", "okay", "ok",
])

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']{2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _ts(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _present_at(member: Any, ts: float) -> bool:
    """Was this member in the room at ``ts``? A missing join stamp
    reads as "joined before the records begin" — the member is in
    the ledger, so they joined at some point; a missing leave
    stamp reads as "still in"."""
    joined = _ts(_field(member, "joined_ts", 0.0))
    left = _ts(_field(member, "left_ts", 0.0))
    return joined <= ts and (left <= 0.0 or left > ts)


def _window_stats(members: list[Any], start: float, end: float) -> dict[str, Any]:
    joins = 0
    leaves = 0
    at_start = 0
    for member in members:
        if _field(member, "is_bot", False) or _field(member, "deleted", False):
            continue
        joined = _ts(_field(member, "joined_ts", 0.0))
        left = _ts(_field(member, "left_ts", 0.0))
        if start < joined <= end:
            joins += 1
        if start < left <= end:
            leaves += 1
        if _present_at(member, start):
            at_start += 1
    net = joins - leaves
    return {
        "joins": joins,
        "leaves": leaves,
        "members_at_start": at_start,
        "net": net,
        "churn_rate": (leaves / at_start) if at_start else None,
        "growth_rate": (net / at_start) if at_start else None,
    }


def churn_and_growth(
    members: Iterable[Any],
    *,
    window_days: int = WINDOW_DAYS,
    now: float | None = None,
) -> dict[str, Any]:
    """Churn + growth over the current window and the one before it.

    Returns ``{"window_days", "current": {...}, "prior": {...},
    "churn_trend", "growth_trend", "direction"}`` where each window
    carries joins / leaves / members_at_start / net / churn_rate /
    growth_rate. ``direction`` is "improving" / "worsening" /
    "steady" by the churn trend, or "no data" when either window
    has no baseline to rate against."""
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    now = now if now is not None else time.time()
    span = window_days * 86400.0
    member_list = list(members)
    current = _window_stats(member_list, now - span, now)
    prior = _window_stats(member_list, now - 2 * span, now - span)
    churn_trend = (
        current["churn_rate"] - prior["churn_rate"]
        if current["churn_rate"] is not None
        and prior["churn_rate"] is not None
        else None
    )
    growth_trend = (
        current["growth_rate"] - prior["growth_rate"]
        if current["growth_rate"] is not None
        and prior["growth_rate"] is not None
        else None
    )
    if churn_trend is None:
        direction = "no data"
    elif churn_trend < 0:
        direction = "improving"
    elif churn_trend > 0:
        direction = "worsening"
    else:
        direction = "steady"
    return {
        "window_days": window_days,
        "current": current,
        "prior": prior,
        "churn_trend": churn_trend,
        "growth_trend": growth_trend,
        "direction": direction,
    }


def _terms(text: str) -> set[str]:
    """Significant terms of one message: 3+ letter words, no URLs,
    no @mentions, no stopwords."""
    clean = _URL_RE.sub(" ", str(text or "").lower())
    clean = _MENTION_RE.sub(" ", clean)
    return {
        token
        for token in _TOKEN_RE.findall(clean)
        if token not in _STOPWORDS
    }


def topic_clusters(
    messages: Iterable[Any],
    *,
    limit: int = CLUSTER_LIMIT,
    min_messages: int = MIN_TERM_MESSAGES,
) -> dict[str, Any]:
    """Cluster supplied messages into topics by shared terms.

    Returns ``{"clusters": [...], "messages_scanned": n,
    "unclustered": n}``. Each cluster carries ``label`` (top terms),
    ``terms``, ``messages`` (count), ``share`` (of clustered
    messages), ``senders`` ([(sender, count)] most first) and one
    ``example`` message. A message joins the topic it shares the
    most terms with; a message with no significant term is
    unclustered."""
    if min_messages < 1:
        raise ValueError("min_messages must be at least 1")
    if limit < 0:
        raise ValueError("limit cannot be negative")
    rows: list[dict[str, Any]] = []
    for message in messages:
        text = str(_field(message, "text", "") or "")
        rows.append({
            "sender": str(_field(message, "sender", "") or "?"),
            "text": text,
            "ts": _ts(_field(message, "ts", 0.0)),
            "terms": _terms(text),
        })
    doc_freq: Counter[str] = Counter()
    for row in rows:
        for term in row["terms"]:
            doc_freq[term] += 1
    significant = {t for t, n in doc_freq.items() if n >= min_messages}
    for row in rows:
        row["terms"] = row["terms"] & significant
    # Term graph: two significant terms link when they co-occur in
    # at least MIN_CO_OCCURRENCE messages. Connected components are
    # the topics — deterministic, no sampling, no randomness.
    co_occurrence: Counter[tuple[str, str]] = Counter()
    for row in rows:
        terms = sorted(row["terms"])
        for i, first in enumerate(terms):
            for second in terms[i + 1:]:
                co_occurrence[(first, second)] += 1
    neighbours: dict[str, set[str]] = {term: set() for term in significant}
    for (first, second), count in co_occurrence.items():
        if count >= MIN_CO_OCCURRENCE:
            neighbours[first].add(second)
            neighbours[second].add(first)
    seen: set[str] = set()
    groups: list[set[str]] = []
    for term in sorted(significant, key=lambda t: (-doc_freq[t], t)):
        if term in seen:
            continue
        stack = [term]
        group: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            group.add(current)
            stack.extend(sorted(neighbours.get(current, ())))
        groups.append(group)
    # Assign each message to the topic it shares the most terms
    # with (ties → the earlier-ranked topic); no shared term → out.
    assignment: dict[int, list[int]] = {i: [] for i in range(len(groups))}
    unclustered = 0
    for row_index, row in enumerate(rows):
        if not row["terms"]:
            unclustered += 1
            continue
        best: int | None = None
        best_overlap = 0
        for index, group in enumerate(groups):
            overlap = len(row["terms"] & group)
            if overlap > best_overlap:
                best = index
                best_overlap = overlap
        if best is None:
            unclustered += 1
        else:
            assignment[best].append(row_index)
    clusters: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        member_rows = [rows[i] for i in assignment[index]]
        if len(member_rows) < min_messages:
            unclustered += len(member_rows)
            continue
        label = sorted(group, key=lambda t: (-doc_freq[t], t))[:3]
        senders = Counter(row["sender"] for row in member_rows)
        # Example: the message sharing the most label terms,
        # earliest first — the same input always picks the same one.
        label_set = set(label)
        example = sorted(
            member_rows,
            key=lambda r: (
                -len(r["terms"] & label_set),
                r["ts"],
                r["sender"],
                r["text"],
            ),
        )[0]
        clusters.append({
            "label": label,
            "terms": sorted(group),
            "messages": len(member_rows),
            "senders": sorted(senders.items(),
                              key=lambda kv: (-kv[1], kv[0]))[:3],
            "example": example["text"],
        })
    clusters.sort(key=lambda c: (-c["messages"], c["label"]))
    # Messages in clusters dropped by the limit are unclustered —
    # the report never claims more than it shows.
    dropped = clusters[limit:] if limit else clusters
    unclustered += sum(c["messages"] for c in dropped)
    clusters = clusters[:limit] if limit else []
    clustered = sum(c["messages"] for c in clusters)
    for cluster in clusters:
        cluster["share"] = (
            cluster["messages"] / clustered if clustered else 0.0
        )
    return {
        "clusters": clusters,
        "messages_scanned": len(rows),
        "unclustered": unclustered,
    }
