"""Issue urgency — what should be fixed first (pattern port).

The design is craft-repo-status-app's ``craft-core/src/urgency.rs``:
a transparent, deterministic points table over open issues. Every
issue earns points for what it says (crash / hang / data loss beats
"doesn't work" beats everything else), for who filed it (a user
report outranks the team's own automated findings), for how many
people are piling on (reactions, comments), for naming the latest
build, and for being fresh. Feature requests, questions and docs
nits are pushed down, and wontfix / duplicate / invalid / not-planned
issues drop out entirely. Each contribution is recorded as a short
reason so the UI can show *why*.

Pattern only: the source repo declares MIT in its manifest with no
licence file, so no upstream code or text is vendored or copied here
— the table (signals, weights, thresholds) is re-implemented as
Devon's own code. The reason strings are the table's observable
output vocabulary, kept so the renders can show *why* an issue
scores what it does.

There is deliberately no threshold band turning points into named
attention levels in the source — the attention mechanism IS the
ranking: the top issues first, the repo urgency (the sum of the top
five scores), and the critical flag. No band is invented here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "IssueInput",
    "Scored",
    "UrgentIssue",
    "needs_info",
    "score_issue",
    "rank_issues",
]

#: Phrases that mean the app is unusable or user data is at risk:
#: (phrase, category). One category counts once, however many of
#: its phrases hit. +40 when any category hits.
CRITICAL_PHRASES: tuple[tuple[str, str], ...] = (
    ("crash", "crash"), ("crashes", "crash"), ("crashed", "crash"),
    ("crashing", "crash"), ("panic", "panic"), ("panics", "panic"),
    ("segfault", "crash"), ("data loss", "data loss"),
    ("lost my", "data loss"), ("lost work", "data loss"),
    ("corrupt", "corruption"), ("corrupted", "corruption"),
    ("corrupts", "corruption"), ("freeze", "freeze"),
    ("freezes", "freeze"), ("frozen", "freeze"), ("hang", "hang"),
    ("hangs", "hang"), ("unresponsive", "hang"),
    ("not responding", "hang"), ("won't open", "won't open"),
    ("wont open", "won't open"), ("can't open", "won't open"),
    ("cannot open", "won't open"), ("won't launch", "won't launch"),
    ("wont launch", "won't launch"), ("won't start", "won't launch"),
    ("doesn't start", "won't launch"), ("does not start", "won't launch"),
    ("fails to launch", "won't launch"), ("fails to start", "won't launch"),
    ("not launching", "won't launch"), ("black screen", "blank screen"),
    ("blank screen", "blank screen"), ("white screen", "blank screen"),
    ("security", "security"), ("vulnerability", "security"),
    ("malware", "security"), ("virus", "security"),
    ("trojan", "security"),
)

#: Phrases that mean something is broken, but not catastrophically.
#: +15 when any hits and no critical phrase did.
BROKEN_PHRASES: tuple[str, ...] = (
    "bug", "broken", "regression", "doesn't work", "does not work",
    "not working", "doesn't", "does not", "don't", "fails", "failed",
    "failure", "error", "can't", "cannot", "unable", "wrong", "lost",
    "install", "installer", "slow", "lag", "glitch",
)

#: Labels that mark an issue a request rather than a defect.
FEATURE_LABELS: tuple[str, ...] = (
    "enhancement", "feature", "feature request", "question",
    "documentation", "docs",
)
#: Labels that drop an issue from the ranking entirely.
EXCLUDE_LABELS: tuple[str, ...] = (
    "wontfix", "duplicate", "invalid", "not planned",
)
#: Title prefixes that mark an issue a request rather than a defect.
FEATURE_PREFIXES: tuple[str, ...] = (
    "[feature", "feature request", "feature:", "request:", "[request",
    "suggestion", "[suggestion", "idea:", "add ", "support ",
    "please add", "fix(docs)", "docs:", "docs(", "[docs", "question",
    "[question",
)
#: Labels saying the reporter owes more detail before anyone can act.
NEEDS_INFO_LABELS: tuple[str, ...] = (
    "needs info", "needs-info", "need info", "needs more info",
)

CRITICAL_POINTS = 40.0
BROKEN_POINTS = 15.0
BUG_LABEL_POINTS = 15.0
RELIABILITY_LABEL_POINTS = 6.0
TRIAGE_CRITICAL_POINTS = 25.0
TRIAGE_HIGH_POINTS = 10.0
TRIAGE_LOW_POINTS = -15.0
NEEDS_INFO_POINTS = -15.0
FEATURE_POINTS = -25.0
USER_REPORT_POINTS = 10.0
LATEST_BUILD_POINTS = 10.0
FRESH_DAY_POINTS = 8.0
FRESH_3DAY_POINTS = 4.0
STALE_POINTS = -5.0
ENGAGEMENT_CAP = 40.0


@dataclass
class IssueInput:
    """One issue, as the scorer reads it.

    ``created_at`` / ``updated_at`` take an ISO-8601 string, a
    datetime, or None. ``reactions`` is the total reaction count,
    ``author_association`` is GitHub's authorAssociation
    (``NONE``, ``CONTRIBUTOR``, ``MEMBER`` …).
    """
    number: int = 0
    title: str = ""
    url: str = ""
    created_at: Any = None
    updated_at: Any = None
    comments: int = 0
    reactions: int = 0
    labels: list[str] = field(default_factory=list)
    author_association: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> IssueInput:
        return cls(
            number=int(row.get("number") or 0),
            title=str(row.get("title") or ""),
            url=str(row.get("url") or ""),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            comments=int(row.get("comments") or 0),
            reactions=int(row.get("reactions") or 0),
            labels=[str(label) for label in (row.get("labels") or [])],
            author_association=str(
                row.get("author_association") or ""),
        )


@dataclass
class Scored:
    score: float
    critical: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class UrgentIssue:
    number: int
    title: str
    url: str
    created_at: Any
    updated_at: Any
    comments: int
    reactions: int
    labels: list[str]
    score: int
    critical: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "comments": self.comments,
            "reactions": self.reactions,
            "labels": list(self.labels),
            "score": self.score,
            "critical": self.critical,
            "reasons": list(self.reasons),
        }


def needs_info(labels: list[str]) -> bool:
    """Whether an issue is waiting on its reporter, by its labels."""
    lowered = [str(label).lower() for label in labels]
    return any(label in NEEDS_INFO_LABELS for label in lowered)


def _triage_level(labels: list[str]) -> str | None:
    """The triage level a label states, if any.

    Reads labels such as ``priority: critical`` and ``priority:P1``.
    The first label that names a known level wins; ``p2`` names no
    level and is skipped, like any other unknown value.
    """
    for label in labels:
        if not label.startswith("priority:"):
            continue
        value = label[len("priority:"):].strip()
        if value in ("critical", "p0"):
            return "critical"
        if value in ("high", "p1"):
            return "high"
        if value in ("low", "p3", "p4"):
            return "low"
    return None


def _normalize(text: str) -> str:
    """Lowercase, normalize apostrophes, punctuation to spaces,
    padded so ``" phrase "`` matches whole words only."""
    out = [" "]
    last_space = True
    for ch in text.lower():
        if ch in ("\u2019", "`"):
            ch = "'"
        if ch.isalnum() or ch == "'":
            out.append(ch)
            last_space = False
        elif not last_space:
            out.append(" ")
            last_space = True
    if not last_space:
        out.append(" ")
    return "".join(out)


def _has_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in text


def _major_minor(tag: str) -> str | None:
    """``"v0.5.0"`` → ``"0.5"``: issues name the version loosely
    ("0.5", "0.50", "v0.5.0")."""
    stripped = tag.lstrip("vV")
    parts = stripped.replace("-", ".").split(".")
    if len(parts) < 2:
        return None
    major, minor = parts[0], parts[1]
    if (major and minor and major.isdigit() and minor.isdigit()):
        return f"{major}.{minor}"
    return None


def _mentions_version(title: str, mm: str) -> bool:
    """Whether a title names version ``mm``.

    ``"0.5"`` matches ``"0.5"``, ``"0.5.0"`` and ``"0.50"`` (a
    common typo for 0.5.0) but not ``"0.51"`` or ``"v10.5"``.
    """
    start = 0
    while True:
        idx = title.find(mm, start)
        if idx < 0:
            return False
        before_ok = idx == 0 or not title[idx - 1].isdigit()
        after = title[idx + len(mm):idx + len(mm) + 1]
        if not after or not after.isdigit():
            after_ok = True
        elif after == "0":
            nxt = title[idx + len(mm) + 1:idx + len(mm) + 2]
            after_ok = not (nxt and nxt.isdigit())
        else:
            after_ok = False
        if before_ok and after_ok:
            return True
        start = idx + 1


def _parse_when(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        when = value
    else:
        try:
            when = datetime.fromisoformat(
                str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when


def _round_half_up(value: float) -> int:
    """Whole-number score: the source rounds halves away from zero;
    scores are clamped at zero first, so floor(x + 0.5) is exact."""
    return int(math.floor(value + 0.5))


def score_issue(issue: IssueInput, now: datetime | None = None,
                latest_tag: str | None = None) -> Scored | None:
    """Score one issue. ``None`` when it is closed-in-spirit
    (wontfix, duplicate, invalid, not planned)."""
    labels = [str(label).lower() for label in issue.labels]
    if any(label in EXCLUDE_LABELS for label in labels):
        return None
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    title_lc = issue.title.lower()
    text = _normalize(f"{issue.title} {' '.join(issue.labels)}")

    score = 0.0
    reasons: list[str] = []

    feature_like = (
        any(label in FEATURE_LABELS for label in labels)
        or any(title_lc.strip().startswith(p)
               for p in FEATURE_PREFIXES)
    )

    critical_hits: list[str] = []
    for phrase, category in CRITICAL_PHRASES:
        if _has_phrase(text, phrase) and category not in critical_hits:
            critical_hits.append(category)
    triaged = _triage_level(labels)
    # Someone who triaged it knows better than a word in its title.
    if triaged == "critical":
        critical = True
    elif triaged == "low":
        critical = False
    else:
        critical = bool(critical_hits) and not feature_like
    if critical_hits:
        score += CRITICAL_POINTS
        reasons.append(", ".join(critical_hits))
    elif any(_has_phrase(text, p) for p in BROKEN_PHRASES):
        score += BROKEN_POINTS
        reasons.append("broken")

    if any(label in ("bug", "regression") for label in labels):
        score += BUG_LABEL_POINTS
        reasons.append("bug label")
    if any(label in ("reliability", "file-compat", "performance")
           for label in labels):
        score += RELIABILITY_LABEL_POINTS
    if triaged == "critical":
        score += TRIAGE_CRITICAL_POINTS
        reasons.append("priority: critical")
    elif triaged == "high":
        score += TRIAGE_HIGH_POINTS
        reasons.append("priority: high")
    elif triaged == "low":
        score += TRIAGE_LOW_POINTS
        reasons.append("priority: low")
    if needs_info(labels):
        score += NEEDS_INFO_POINTS
        reasons.append("needs info")
    if feature_like:
        score += FEATURE_POINTS
        reasons.append("request/docs")

    if issue.author_association in (
            "NONE", "FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR"):
        score += USER_REPORT_POINTS
        reasons.append("user report")

    engagement = 3.0 * issue.reactions + 2.0 * issue.comments
    if engagement > 0:
        score += min(10.0 * math.log(1.0 + engagement),
                     ENGAGEMENT_CAP)
        if issue.reactions > 0:
            reasons.append(f"{issue.reactions} 👍")
        if issue.comments > 0:
            reasons.append(f"{issue.comments} 💬")

    if latest_tag:
        mm = _major_minor(latest_tag)
        if mm and _mentions_version(title_lc, mm):
            score += LATEST_BUILD_POINTS
            reasons.append("latest build")

    created = _parse_when(issue.created_at)
    if created is not None:
        age_hours = (moment - created).total_seconds() / 3600.0
        if age_hours < 24:
            score += FRESH_DAY_POINTS
            reasons.append("new")
        elif age_hours < 72:
            score += FRESH_3DAY_POINTS
    updated = _parse_when(issue.updated_at)
    if updated is not None and (moment - updated).days > 30:
        score += STALE_POINTS

    return Scored(score=max(0.0, score), critical=critical,
                  reasons=reasons)


def rank_issues(issues: list[IssueInput],
                now: datetime | None = None,
                latest_tag: str | None = None,
                keep: int = 10) -> tuple[list[UrgentIssue], float, int]:
    """Rank issues by urgency.

    Returns the top ``keep`` issues (score > 0, best first, ties →
    newest), the repo urgency (the sum of the top five scores) and
    the count of critical issues. Duplicate numbers keep their first
    row; closed-in-spirit issues drop out before counting.
    """
    seen: set[int] = set()
    ranked: list[UrgentIssue] = []
    for issue in issues:
        if issue.number in seen:
            continue
        seen.add(issue.number)
        scored = score_issue(issue, now, latest_tag)
        if scored is None:
            continue
        ranked.append(UrgentIssue(
            number=issue.number,
            title=issue.title,
            url=issue.url,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            comments=issue.comments,
            reactions=issue.reactions,
            labels=list(issue.labels),
            score=_round_half_up(scored.score),
            critical=scored.critical,
            reasons=list(scored.reasons),
        ))
    critical = sum(1 for row in ranked if row.critical)
    ranked = [row for row in ranked if row.score > 0]

    def sort_key(row: UrgentIssue) -> tuple[int, int, float]:
        when = _parse_when(row.created_at)
        if when is None:
            return (row.score, 0, 0.0)
        return (row.score, 1, when.timestamp())

    ranked.sort(key=sort_key, reverse=True)
    urgency = float(sum(row.score for row in ranked[:5]))
    return ranked[:keep], urgency, critical
