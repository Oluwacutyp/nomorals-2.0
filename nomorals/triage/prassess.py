"""PR landability — the landing heuristics (pattern port).

The design is craft-pr-review-tool's ``crates/pr-core/src/assess.rs``
(the headline truth table is identical in craft-repo-status-app's
embedded copy): a PR has independent flags (approved, conflicted,
CI failed …) and one headline status worked out from them by the
truth table in :func:`headline` — the first matching row wins.
Around it sit the quality score (baseline 60, moved by CI, reviews,
description, tests and CodeRabbit's findings), the ease score
(size, files, spread, conflicts, flags), the author tiers, and the
focus sort key that orders the review queue.

Pattern only: both source repos declare MIT in their manifests with
no licence file, so no upstream code or text is vendored or copied
here — the tables (signals, weights, thresholds) are re-implemented
as Devon's own code. The state names, reason strings and point
values are the design's observable vocabulary, kept so the verdict
a caller sees matches the table the design promises.

Two rules from the source are structural here. First, an outside
review (the source calls it Claude) can only make a PR look harder
and mark it interesting — it can never improve the headline or the
quality, so :func:`assess_pr` takes such a review as caller-supplied
``analysis`` data and applies only those two effects. Second, an
approval counts for Landable only when it is on the latest commit;
an approval of an older commit shows as stale and does not land.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "PrConfig",
    "PrDetail",
    "PrInput",
    "AuthorFacts",
    "Signals",
    "headline",
    "ci_state",
    "author_tier",
    "assess_pr",
    "PR_STATES",
    "STATE_FOCUS",
]

#: The nine headline states, best first for the review queue.
PR_STATES: tuple[str, ...] = (
    "draft", "needs_work", "conflicted", "not_inspected", "checking",
    "ci_not_run", "landable", "promising", "reviewable",
)
#: Focus sort-key base per state; quality, ease and author only
#: break ties within a status.
STATE_FOCUS: dict[str, int] = {
    "landable": 8000,
    "promising": 7000,
    "reviewable": 6000,
    "ci_not_run": 5000,
    "checking": 4000,
    "not_inspected": 3000,
    "conflicted": 2000,
    "needs_work": 1000,
    "draft": 0,
}

HIGH_QUALITY_DEFAULT = 75
FREQUENT_OVER_DEFAULT = 30

#: Paths whose change by someone without write access asks the
#: reviewer to look closer (it never counts against the PR).
SENSITIVE_PATHS_DEFAULT: tuple[str, ...] = (
    ".github/workflows/",
    ".github/actions/",
    ".github/CODEOWNERS",
    "CODEOWNERS",
    "build.rs",
    ".cargo/config",
    "deny.toml",
    ".coderabbit.yaml",
    "sonar-project.properties",
    "Dockerfile",
    "tauri.conf.json",
    "/capabilities/",
)
#: Check names that mark a check security-relevant.
SECURITY_CHECK_WORDS: tuple[str, ...] = (
    "secur", "audit", "cargo-deny", "cargo deny", "codeql",
    "gitleaks", "semgrep", "osv", "supply chain", "git dependencies",
)
#: Check states that count as failed. Cancelled runs say nothing
#: about the code, usually just that a newer one started.
FAILED_CHECK_STATES: tuple[str, ...] = (
    "FAILURE", "ERROR", "TIMED_OUT", "STARTUP_FAILURE",
    "ACTION_REQUIRED",
)
#: Check states that count as still running.
RUNNING_CHECK_STATES: tuple[str, ...] = (
    "PENDING", "IN_PROGRESS", "QUEUED", "WAITING", "REQUESTED",
    "EXPECTED",
)
#: File names that make a change lockfile / generated, not real code.
LOCK_FILENAMES: tuple[str, ...] = (
    "Cargo.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "contributors.json",
)
#: Code extensions, for "changes code without tests".
CODE_EXTENSIONS: tuple[str, ...] = (
    "rs", "ts", "tsx", "js", "py", "wgsl", "c", "cpp", "h", "swift",
    "kt", "go",
)

#: Ease bands: (upper line bound, points, label).
_EASE_LINES: tuple[tuple[int, int, str], ...] = (
    (50, 0, "Tiny"),
    (200, 1, "Small"),
    (600, 2, "Moderate"),
    (2000, 4, "Large"),
)


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


def _when_gte(a: Any, b: Any) -> bool:
    """``a >= b`` with missing times sorting below every time —
    the source's Option ordering, where None is less than Some."""
    pa, pb = _parse_when(a), _parse_when(b)
    if pa is None:
        return pb is None
    if pb is None:
        return True
    return pa >= pb


@dataclass
class Signals:
    """The facts the headline depends on."""
    draft: bool = False
    #: A failed check, requested changes, a CodeRabbit flag or an
    #: open conversation.
    problem: bool = False
    conflicted: bool = False
    inspected: bool = False
    #: CI or CodeRabbit is running, or GitHub hasn't worked out
    #: mergeability.
    running: bool = False
    #: CI ran and passed, as opposed to not running at all. Failing
    #: is a problem and running is ``running``, so neither gets this
    #: far.
    ci_passed: bool = False
    #: Approved on the latest commit.
    approved: bool = False
    high_quality: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Signals:
        def flag(name: str) -> bool:
            value = row.get(name)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "y")
            return bool(value)

        return cls(
            draft=flag("draft"),
            problem=flag("problem"),
            conflicted=flag("conflicted"),
            inspected=flag("inspected"),
            running=flag("running"),
            ci_passed=flag("ci_passed"),
            approved=flag("approved"),
            high_quality=flag("high_quality"),
        )


def headline(signals: Signals) -> str:
    """The truth table: the first row that matches wins.

    draft → Draft; problem → Needs work; conflicted → Conflicted;
    not inspected → Not inspected; running → Checking; CI not run →
    CI not run; approved → Landable; high quality → Promising;
    otherwise Reviewable. The nine rows cover all 256 combinations
    exactly once — see the tests.
    """
    if signals.draft:
        return "draft"
    if signals.problem:
        return "needs_work"
    if signals.conflicted:
        return "conflicted"
    if not signals.inspected:
        return "not_inspected"
    if signals.running:
        return "checking"
    if not signals.ci_passed:
        return "ci_not_run"
    if signals.approved:
        return "landable"
    if signals.high_quality:
        return "promising"
    return "reviewable"


@dataclass
class PrConfig:
    """Tunable thresholds, with the source's defaults."""
    high_quality: int = HIGH_QUALITY_DEFAULT
    frequent_over: int = FREQUENT_OVER_DEFAULT
    people_toml_is_core: bool = True
    sensitive_paths: tuple[str, ...] = SENSITIVE_PATHS_DEFAULT

    @classmethod
    def from_dict(cls, row: dict[str, Any] | None) -> PrConfig:
        row = row or {}
        paths = row.get("sensitive_paths")
        return cls(
            high_quality=int(row.get("high_quality",
                                     HIGH_QUALITY_DEFAULT)),
            frequent_over=int(row.get("frequent_over",
                                      FREQUENT_OVER_DEFAULT)),
            people_toml_is_core=bool(
                row.get("people_toml_is_core", True)),
            sensitive_paths=(tuple(str(p) for p in paths)
                             if paths is not None
                             else SENSITIVE_PATHS_DEFAULT),
        )


@dataclass
class AuthorFacts:
    """What is known about a PR's author, for the tier call."""
    login: str = ""
    #: Write access or better, if the roster says so.
    permission: str | None = None
    #: In the repo's contributors/people.toml.
    listed: bool = False
    landed_here: int = 0
    #: Merged PRs across the watched repos.
    landed_total: int = 0
    #: Whether every repo's merge history has been walked, so the
    #: counts are final.
    counts_complete: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> AuthorFacts:
        return cls(
            login=str(row.get("login") or ""),
            permission=row.get("permission"),
            listed=bool(row.get("listed")),
            landed_here=int(row.get("landed_here") or 0),
            landed_total=int(row.get("landed_total") or 0),
            counts_complete=bool(row.get("counts_complete")),
        )


@dataclass
class PrDetail:
    """The inspected detail of a PR (CodeRabbit, files, tests)."""
    fetched_at: Any = None
    for_updated_at: Any = None
    for_head: str = ""
    body: str = ""
    files: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    rabbit: str = "none"
    findings: list[dict[str, Any]] = field(default_factory=list)
    human_threads_open: int = 0
    linked_issues: list[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any] | None) -> PrDetail | None:
        if not row:
            return None
        return cls(
            fetched_at=row.get("fetched_at"),
            for_updated_at=row.get("for_updated_at"),
            for_head=str(row.get("for_head") or ""),
            body=str(row.get("body") or ""),
            files=[dict(f) for f in (row.get("files") or [])],
            checks=[dict(c) for c in (row.get("checks") or [])],
            rabbit=str(row.get("rabbit") or "none").lower(),
            findings=[dict(f) for f in (row.get("findings") or [])],
            human_threads_open=int(
                row.get("human_threads_open") or 0),
            linked_issues=list(row.get("linked_issues") or []),
        )


@dataclass
class PrInput:
    """One PR, as the evaluator reads it."""
    number: int = 0
    author: str = ""
    association: str = ""
    author_is_bot: bool = False
    is_draft: bool = False
    mergeable: str = ""
    merge_state: str = ""
    review_decision: str | None = None
    approved_on_head: list[str] = field(default_factory=list)
    ci_rollup: str | None = None
    reactions: int = 0
    comments_total: int = 0
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    listed_at: Any = None
    updated_at: Any = None
    head_oid: str = ""
    detail: PrDetail | None = None
    #: Caller-supplied outside review (a model, a human's notes):
    #: ``{"ease": "easy"|"medium"|"hard", "interesting": bool}``.
    analysis: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> PrInput:
        return cls(
            number=int(row.get("number") or 0),
            author=str(row.get("author") or ""),
            association=str(row.get("association") or ""),
            author_is_bot=bool(row.get("author_is_bot")),
            is_draft=bool(row.get("is_draft")),
            mergeable=str(row.get("mergeable") or ""),
            merge_state=str(row.get("merge_state") or ""),
            review_decision=row.get("review_decision"),
            approved_on_head=[str(a) for a in
                              (row.get("approved_on_head") or [])],
            ci_rollup=row.get("ci_rollup"),
            reactions=int(row.get("reactions") or 0),
            comments_total=int(row.get("comments_total") or 0),
            additions=int(row.get("additions") or 0),
            deletions=int(row.get("deletions") or 0),
            changed_files=int(row.get("changed_files") or 0),
            listed_at=row.get("listed_at"),
            updated_at=row.get("updated_at"),
            head_oid=str(row.get("head_oid") or ""),
            detail=PrDetail.from_dict(row.get("detail")),
            analysis=row.get("analysis"),
        )

    def __post_init__(self) -> None:
        # Tolerate a raw detail dict at construction time (JSON-shaped
        # callers); the typed path stays PrDetail.
        if isinstance(self.detail, dict):
            self.detail = PrDetail.from_dict(self.detail)

    def current_detail(self) -> PrDetail | None:
        """The detail for the current head, at least as fresh as the
        PR's last update — older detail does not count."""
        detail = self.detail
        if detail is None:
            return None
        if detail.for_head != self.head_oid:
            return None
        if not _when_gte(detail.for_updated_at, self.updated_at):
            return None
        return detail


def _is_rabbit_check(name: str) -> bool:
    return name.lower() == "coderabbit"


def _is_security_check(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in SECURITY_CHECK_WORDS)


def _check_failed(state: str) -> bool:
    return state in FAILED_CHECK_STATES


def _check_running(state: str) -> bool:
    return state in RUNNING_CHECK_STATES


def _check_counts(check: dict[str, Any]) -> bool:
    """Whether a check says anything about this commit: CodeRabbit
    is judged by its findings, and cancelled or skipped runs say
    nothing."""
    name = str(check.get("name") or "")
    state = str(check.get("state") or "")
    return (not _is_rabbit_check(name)
            and state not in ("CANCELLED", "SKIPPED"))


def ci_state(pr: PrInput) -> str:
    """``failure``, ``pending``, ``success`` or ``none``.

    The checks in the detail leave out CodeRabbit, which is judged
    by its findings, and cancelled or skipped runs — but they may be
    older than the last listing's rollup, so a detail fetched before
    the PR was last listed falls back to the rollup. The rollup
    cannot tell cancelled runs from failed ones, so a PR not yet
    inspected may read as failing until it is.
    """
    detail = pr.current_detail()
    if detail is not None and _when_gte(detail.fetched_at,
                                        pr.listed_at):
        checks = [c for c in detail.checks if _check_counts(c)]
        if any(_check_failed(str(c.get("state") or ""))
               for c in checks):
            return "failure"
        if any(_check_running(str(c.get("state") or ""))
               for c in checks):
            return "pending"
        if not checks:
            return "none"
        return "success"
    rollup = pr.ci_rollup
    if rollup == "SUCCESS":
        return "success"
    if rollup in ("FAILURE", "ERROR"):
        return "failure"
    if rollup is not None:
        return "pending"
    return "none"


def author_tier(pr: PrInput, author: AuthorFacts,
                config: PrConfig) -> str:
    """core / frequent / new / regular / bot.

    Core has write access or better, or is listed in the repo's
    people file (when that file counts as core). Frequent has more
    than ``frequent_over`` PRs merged across the watched repos.
    New is GitHub's first-time contributor.
    """
    if pr.author_is_bot or pr.author.endswith("[bot]"):
        return "bot"
    can_write = author.permission in ("WRITE", "MAINTAIN", "ADMIN")
    member = pr.association in ("OWNER", "MEMBER", "COLLABORATOR")
    if can_write or member or (author.listed
                               and config.people_toml_is_core):
        return "core"
    if author.landed_total > config.frequent_over:
        return "frequent"
    if pr.association in ("FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"):
        return "new"
    return "regular"


def _is_lock_or_generated(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (name in LOCK_FILENAMES
            or "/generated/" in path
            or path.startswith("vendor/"))


def _is_test(path: str) -> bool:
    lowered = path.lower()
    return ("test" in lowered or "/spec" in lowered
            or "fixture" in lowered or "corpus" in lowered)


def _is_code(path: str) -> bool:
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    return ext in CODE_EXTENSIONS


def _reason(text: str, points: int) -> dict[str, Any]:
    return {"text": text, "points": points}


def assess_pr(pr: PrInput, author: AuthorFacts,
              config: PrConfig | None = None) -> dict[str, Any]:
    """Run the heuristics over a PR. Pure arithmetic — no network,
    no model calls; ``pr.analysis`` is caller-supplied data."""
    cfg = config or PrConfig()
    tier = author_tier(pr, author, cfg)
    state_ci = ci_state(pr)
    detail = pr.current_detail() or pr.detail
    inspected = pr.current_detail() is not None

    conflicted = pr.mergeable == "CONFLICTING" or pr.merge_state == "DIRTY"
    changes_requested = pr.review_decision == "CHANGES_REQUESTED"
    approved = pr.review_decision == "APPROVED"

    critical = major = minor = trivial = heavy = 0
    human_open = 0
    security_check_failed: list[str] = []
    rabbit = "none"
    has_body = long_body = linked = False
    real_lines = pr.additions + pr.deletions
    tops: set[str] = set()
    tests_touched = deps_changed = False
    sensitive_paths: list[str] = []
    security_flags = security_nits = 0
    if detail is not None:
        for finding in detail.findings:
            if finding.get("resolved") or finding.get("outdated"):
                continue
            severity = str(finding.get("severity") or "minor").lower()
            if severity == "critical":
                critical += 1
            elif severity == "major":
                major += 1
            elif severity == "trivial":
                trivial += 1
            else:
                minor += 1
            if str(finding.get("effort") or "").lower() == "heavy lift":
                heavy += 1
            is_flag = severity in ("critical", "major")
            if is_flag and finding.get("security"):
                security_flags += 1
            elif (not is_flag) and finding.get("security"):
                security_nits += 1
        rabbit_flags = critical + major
        rabbit_nits = minor + trivial
        human_open = detail.human_threads_open
        rabbit = detail.rabbit
        security_check_failed = [
            str(c.get("name") or "")
            for c in detail.checks
            if _is_security_check(str(c.get("name") or ""))
            and _check_failed(str(c.get("state") or ""))
        ]
        body_len = len(detail.body.strip())
        has_body = body_len >= 40
        long_body = body_len >= 200
        linked = bool(detail.linked_issues)
        tests_touched = any(_is_test(str(f.get("path") or ""))
                            for f in detail.files)
        deps_changed = any(
            str(f.get("path") or "").rsplit("/", 1)[-1]
            in ("Cargo.lock", "package-lock.json")
            for f in detail.files)
        generated = sum(
            int(f.get("additions") or 0) + int(f.get("deletions") or 0)
            for f in detail.files
            if _is_lock_or_generated(str(f.get("path") or "")))
        real_lines = max(0, real_lines - generated)
        for f in detail.files:
            path = str(f.get("path") or "")
            if _is_lock_or_generated(path):
                continue
            tops.add("/".join(path.split("/")[:2]))
        can_write = (
            tier == "bot"
            or author.permission in ("WRITE", "MAINTAIN", "ADMIN")
            or pr.association in ("OWNER", "MEMBER", "COLLABORATOR"))
        if not can_write:
            sensitive_paths = [
                str(f.get("path") or "")
                for f in detail.files
                if any(s in str(f.get("path") or "")
                       for s in cfg.sensitive_paths)
            ]
    else:
        rabbit_flags = rabbit_nits = 0

    security_reasons: list[str] = []
    if security_flags > 0:
        security_reasons.append(
            f"{security_flags} CodeRabbit security finding(s)")
    for name in security_check_failed:
        security_flags += 1
        security_reasons.append(f"{name} failed")
    if sensitive_paths:
        security_reasons.append(
            f"Changes {len(sensitive_paths)} sensitive path(s) "
            "without write access")

    # Flags.
    approval_stale = approved and not pr.approved_on_head
    problems: list[str] = []
    if state_ci == "failure":
        problems.append("CI is failing")
    if changes_requested:
        problems.append("A reviewer requested changes")
    if critical + major > 0:
        problems.append(
            f"{critical + major} unresolved CodeRabbit flag(s)")
    if security_check_failed:
        problems.append(
            "Security check failed: " + ", ".join(security_check_failed))
    if human_open > 0:
        problems.append(
            f"{human_open} unresolved review conversation(s)")
    running: list[str] = []
    if state_ci == "pending":
        running.append("CI is still running")
    if rabbit == "inprogress":
        running.append("CodeRabbit is still reviewing")
    if pr.mergeable == "UNKNOWN" or not pr.mergeable:
        running.append("GitHub hasn't worked out mergeability yet")

    # Ease.
    ease: list[dict[str, Any]] = []
    line_points, line_label = 6, "Huge"
    for bound, points, label in _EASE_LINES:
        if real_lines < bound:
            line_points, line_label = points, label
            break
    ease.append(_reason(
        f"{line_label} change: {real_lines} lines", line_points))
    files = pr.changed_files
    if files <= 3:
        file_points = 0
    elif files <= 10:
        file_points = 1
    elif files <= 30:
        file_points = 2
    else:
        file_points = 4
    if file_points > 0:
        ease.append(_reason(f"{files} files", file_points))
    if len(tops) > 4:
        ease.append(_reason(
            f"Spreads across {len(tops)} areas", 1))
    if conflicted:
        ease.append(_reason("Needs a rebase", 2))
    if state_ci == "failure":
        ease.append(_reason("CI has to be fixed first", 1))
    if critical + major > 0:
        ease.append(_reason(
            f"{critical + major} CodeRabbit flag(s) to address",
            min(critical + major, 3)))
    if heavy > 0:
        ease.append(_reason(f"{heavy} heavy-lift finding(s)", 1))
    if sensitive_paths:
        ease.append(_reason("Touches sensitive paths", 2))
    if tier in ("core", "frequent"):
        ease.append(_reason("Author knows the codebase", -1))
    elif tier == "new":
        ease.append(_reason("First-time contributor", 1))
    ease_score = sum(r["points"] for r in ease)
    if ease_score <= 2:
        ease_label = "easy"
    elif ease_score <= 5:
        ease_label = "medium"
    else:
        ease_label = "hard"
    ease_raised = False
    analysis = pr.analysis or {}
    outside_ease = str(analysis.get("ease") or "").lower()
    if outside_ease in ("easy", "medium", "hard"):
        order = {"easy": 0, "medium": 1, "hard": 2}
        if order[outside_ease] > order[ease_label]:
            ease_label = outside_ease
            ease_raised = True

    # Quality.
    quality: list[dict[str, Any]] = [_reason("Baseline", 60)]
    if state_ci == "success":
        quality.append(_reason("CI passes", 10))
    elif state_ci == "failure":
        quality.append(_reason("CI fails", -20))
    if conflicted:
        quality.append(_reason("Merge conflicts", -15))
    if changes_requested:
        quality.append(_reason("Changes requested", -15))
    if approved:
        quality.append(_reason("Approved", 10))
    if real_lines > 10_000:
        quality.append(_reason("Too large to review well", -20))
    elif real_lines > 2_000:
        quality.append(_reason("Very large", -10))
    if detail is not None:
        if long_body:
            quality.append(_reason("Explains itself", 8))
        elif not has_body:
            quality.append(_reason("Little or no description", -15))
        if linked:
            quality.append(_reason("Closes an issue", 6))
        touches_code = any(_is_code(str(f.get("path") or ""))
                           for f in detail.files)
        if tests_touched:
            quality.append(_reason("Includes tests", 12))
        elif touches_code:
            quality.append(_reason("Changes code without tests", -8))
        if critical > 0:
            quality.append(_reason(
                f"{critical} critical finding(s)",
                max(-20 * critical, -40)))
        if major > 0:
            quality.append(_reason(
                f"{major} major finding(s)",
                max(-10 * major, -30)))
        if minor > 0:
            quality.append(_reason(
                f"{minor} minor finding(s)",
                max(-3 * minor, -12)))
        if trivial > 0:
            quality.append(_reason(
                f"{trivial} trivial finding(s)",
                max(-trivial, -4)))
        if (rabbit == "reviewed"
                and critical + major + minor + trivial == 0):
            quality.append(_reason("CodeRabbit had nothing to add", 8))
        if security_flags > 0:
            quality.append(_reason("Security flags", -15))
    if tier in ("core", "frequent"):
        quality.append(_reason("Track record", 4))
    quality_score = max(0, min(100, sum(
        r["points"] for r in quality)))
    high_quality = inspected and quality_score >= cfg.high_quality

    # Headline.
    state = headline(Signals(
        draft=pr.is_draft,
        problem=bool(problems),
        conflicted=conflicted,
        inspected=inspected,
        running=bool(running),
        ci_passed=state_ci == "success",
        approved=approved and not approval_stale,
        high_quality=high_quality,
    ))
    stale_note = ("An approval predates the latest commit, so it "
                  "doesn't count")
    if state == "draft":
        state_reasons = ["Marked as a draft"]
    elif state == "needs_work":
        state_reasons = list(problems)
    elif state == "conflicted":
        state_reasons = [
            "Has merge conflicts, and nothing else known to be wrong"]
        if not inspected:
            state_reasons.append(
                "Not inspected yet, so CodeRabbit and conversations "
                "are unknown")
    elif state == "not_inspected":
        if pr.detail is not None:
            state_reasons = ["Changed since it was last inspected"]
        else:
            state_reasons = [
                "Not inspected yet, so CodeRabbit, files and "
                "conversations are unknown"]
    elif state == "checking":
        state_reasons = list(running)
    elif state == "ci_not_run":
        state_reasons = [
            "CI hasn't run: its workflows may need approving"]
    elif state == "landable":
        state_reasons = [
            "Approved on the latest commit, CI passed, nothing in "
            "the way"]
    else:
        if state == "promising":
            state_reasons = [
                "Passes every automated check and scores well, but "
                "no person has approved it"]
        else:
            state_reasons = [
                "CI passed and nothing is in the way, but no person "
                "has approved it"]
        if approval_stale:
            state_reasons.append(stale_note)

    notable = pr.reactions >= 3 or pr.comments_total >= 8
    interesting = notable or bool(analysis.get("interesting"))

    focus = STATE_FOCUS[state] + quality_score
    focus += {"easy": 60, "medium": 30, "hard": 0}[ease_label]
    focus += {"core": 40, "frequent": 30}.get(tier, 0)
    if interesting:
        focus += 50
    if pr.review_decision == "APPROVED":
        focus += 50

    return {
        "state": state,
        "state_reasons": state_reasons,
        "ease": ease_label,
        "ease_score": ease_score,
        "ease_reasons": ease,
        "ease_raised_by_review": ease_raised,
        "quality": quality_score,
        "quality_reasons": quality,
        "high_quality": high_quality,
        "conflicted": conflicted,
        "approved": approved,
        "approval_stale": approval_stale,
        "changes_requested": changes_requested,
        "inspected": inspected,
        "ci": state_ci,
        "rabbit_flags": rabbit_flags,
        "rabbit_nits": rabbit_nits,
        "security_flags": security_flags,
        "security_nits": security_nits,
        "sensitive_paths": sensitive_paths,
        "security_reasons": security_reasons,
        "deps_changed": deps_changed,
        "tests_touched": tests_touched,
        "interesting": interesting,
        "focus": focus,
        "author_tier": tier,
        "real_lines": real_lines,
    }
