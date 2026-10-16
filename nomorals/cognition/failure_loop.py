"""Close the failure loop: cluster -> fix -> sandbox -> proposal.

The failure KB can already surface recurring clusters; this module
turns a cluster into an *actionable, gated* proposal:

1. **Cluster** — :func:`cluster_failures` groups recent failures
   (via :class:`FailureKB` triage or the raw
   :class:`TrajectoryStore` clusters), decorated with cluster key,
   priority and notes.
2. **Fix** — :func:`propose_fix` maps a cluster's failure class to a
   concrete suggested fix (template + the KB's first-response
   guidance).  The proposal is data, not code — it never edits
   anything.
3. **Sandbox** — :func:`sandbox_check` validates a proposal without
   applying it: the fix text must be non-empty, must not contain
   destructive patterns, an optional ``code_text`` must compile, and
   an optional caller-supplied ``check_fn`` must return truthy.  A
   failed check means the proposal stays unapplied.
4. **Proposal** — :func:`build_proposal` assembles the proposal dict
   (``applied`` is always ``False`` on build), and
   :func:`apply_proposal` is the *only* path that flips ``applied``
   to ``True`` — and only when the caller passes ``approve=True`` AND
   a sandbox pass is on record.  No sandbox pass → no apply, ever.
   ``apply_proposal`` itself never executes fix code: it only marks
   the proposal authorized-for-application; the caller (owner /
   operator tooling) performs the actual change.

Invariants: a proposal never auto-applies; a sandbox failure blocks
application even with approval; approval without a sandbox pass
blocks application.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .failure_kb import FailureKB, guidance_for
from .trajectories import TrajectoryStore, cluster_key_for

__all__ = [
    "FIX_TEMPLATES",
    "DANGEROUS_PATTERNS",
    "cluster_failures",
    "propose_fix",
    "sandbox_check",
    "build_proposal",
    "apply_proposal",
    "FailureLoop",
]

#: Failure class -> (fix_kind, suggested fix).  Templates are starting
#: points for the operator, not auto-applied patches.
FIX_TEMPLATES = {
    "rate_limited": (
        "backoff_retry",
        "Add exponential backoff with jitter before retrying; spread load"
        " across providers instead of hammering one.",
    ),
    "auth": (
        "refresh_credentials",
        "Refresh / re-issue credentials and verify token expiry before"
        " retrying — this class never heals on its own.",
    ),
    "context_overflow": (
        "shrink_context",
        "Shrink the context (summarize, drop old history) before retrying;"
        " do not resend the same payload.",
    ),
    "timeout": (
        "retry_with_backoff",
        "Retry with a longer timeout and backoff; treat repeats as"
        " provider degradation, not a flake.",
    ),
    "network": (
        "failover_provider",
        "Retry once, then fail over to an alternate provider or endpoint.",
    ),
    "not_found": (
        "verify_reference",
        "Verify the referenced id / path exists; cache the lookup instead"
        " of retrying blindly.",
    ),
    "invalid_request": (
        "fix_payload",
        "Fix the request payload — the request itself is malformed, so do"
        " not retry it unchanged.",
    ),
}

#: Substrings that fail the sandbox's static scan.  A suggested fix is
#: prose guidance, never shell — anything that looks like a destructive
#: command or code execution has no business in a proposal.
DANGEROUS_PATTERNS = (
    "rm -rf",
    "rm -fr",
    "DROP TABLE",
    "DELETE FROM",
    "os.system",
    "subprocess",
    "eval(",
    "exec(",
    "__import__",
    "shutil.rmtree",
    "format(",
)


def _suggested_fix(cluster: dict[str, Any]) -> tuple[str, str, str]:
    """(fix_kind, suggested_fix, guidance) for a cluster."""
    failure_class = (cluster.get("failure_class") or "").lower()
    for key, (kind, text) in FIX_TEMPLATES.items():
        if key in failure_class:
            return kind, text, guidance_for(failure_class, cluster.get("error_signature") or "")
    guidance = guidance_for(failure_class, cluster.get("error_signature") or "")
    return (
        "investigate",
        "Investigate the cluster's example failures and write a targeted"
        " fix — no template covers this failure class, so do not apply a"
        " generic retry.",
        guidance,
    )


def _decorate_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    """Ensure cluster_key + priority are present on a raw cluster."""
    out = dict(cluster)
    if not out.get("cluster_key"):
        out["cluster_key"] = cluster_key_for(
            out.get("task_kind") or "", out.get("error_signature") or "")
    if out.get("priority") is None:
        out["priority"] = TrajectoryStore.cluster_priority(out)
    return out


def cluster_failures(
    source: Any | None = None,
    task_kind: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Failure clusters, most urgent first.

    ``source`` may be a :class:`FailureKB` (triage view: open clusters,
    priority-ordered, with notes), a :class:`TrajectoryStore` (raw
    clusters), or ``None`` (empty — no store, no invented clusters).
    """
    if source is None:
        return []
    if isinstance(source, FailureKB):
        return source.triage(limit=limit)
    if isinstance(source, TrajectoryStore):
        clusters = source.failure_clusters(task_kind, limit=10_000)
        decorated = [_decorate_cluster(c) for c in clusters]
        decorated.sort(key=lambda c: -c["priority"])
        return decorated[: max(0, int(limit))]
    # Duck-typed store with failure_clusters().
    clusters = source.failure_clusters(task_kind, limit=10_000)  # type: ignore[attr-defined]
    decorated = [_decorate_cluster(c) for c in clusters]
    decorated.sort(key=lambda c: -c.get("priority", 0))
    return decorated[: max(0, int(limit))]


def propose_fix(cluster: dict[str, Any]) -> dict[str, Any]:
    """Turn a cluster into a proposal dict.  Never applies anything.

    The returned dict always has ``applied: False``,
    ``sandbox_required: True``, ``sandbox_passed: False`` and
    ``requires_owner_approval: True`` — a proposal is born unapplied.
    """
    if not isinstance(cluster, dict) or not cluster.get("error_signature"):
        raise ValueError("propose_fix needs a cluster dict with error_signature")
    cluster = _decorate_cluster(cluster)
    fix_kind, suggested_fix, guidance = _suggested_fix(cluster)
    return {
        "cluster_key": cluster.get("cluster_key") or "",
        "task_kind": cluster.get("task_kind") or "",
        "failure_class": cluster.get("failure_class") or "",
        "error_signature": cluster.get("error_signature") or "",
        "exception_class": cluster.get("exception_class") or "",
        "count": int(cluster.get("count") or 0),
        "priority": cluster.get("priority"),
        "fix_kind": fix_kind,
        "suggested_fix": suggested_fix,
        "guidance": guidance,
        "notes": cluster.get("notes") or [],
        "sandbox_required": True,
        "sandbox_passed": False,
        "requires_owner_approval": True,
        "applied": False,
        "status": "proposed",
        "created_at": time.time(),
    }


def sandbox_check(
    proposal: dict[str, Any],
    *,
    check_fn: Callable[[dict[str, Any]], Any] | None = None,
    code_text: str | None = None,
) -> dict[str, Any]:
    """Validate a proposal *without applying it*.

    Checks, in order: the proposal has a non-empty suggested fix; the
    fix text contains no :data:`DANGEROUS_PATTERNS`; an optional
    ``code_text`` compiles (syntax only — it is never executed); an
    optional ``check_fn(proposal)`` returns truthy (its exception
    fails the check).  Returns ``{"passed", "checks"}`` where each
    check is ``{"name", "passed", "detail"}``.  Never mutates the
    proposal and never applies the fix.
    """
    checks: list[dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    fix_text = (proposal or {}).get("suggested_fix") or ""
    _check("fix_present", bool(str(fix_text).strip()),
           "suggested_fix is empty" if not str(fix_text).strip() else "")
    hits = [p for p in DANGEROUS_PATTERNS if p in str(fix_text)]
    _check("no_dangerous_patterns", not hits,
           f"dangerous pattern(s): {', '.join(hits)}" if hits else "")
    if code_text is not None:
        try:
            compile(str(code_text), "<sandbox-proposal>", "exec")
        except SyntaxError as exc:
            _check("code_compiles", False, f"SyntaxError: {exc}")
        else:
            _check("code_compiles", True)
    if check_fn is not None:
        try:
            result = check_fn(proposal)
        except Exception as exc:  # noqa: BLE001 — a crashing check fails closed
            _check("custom_check", False, f"check_fn raised: {exc!r}")
        else:
            _check("custom_check", bool(result),
                   "" if result else "check_fn returned falsy")
    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "checks": checks}


def build_proposal(
    cluster: dict[str, Any],
    *,
    sandbox_result: dict[str, Any] | None = None,
    check_fn: Callable[[dict[str, Any]], Any] | None = None,
    code_text: str | None = None,
) -> dict[str, Any]:
    """Build a proposal and optionally run its sandbox check.

    ``applied`` is *always* ``False`` here — building never applies.
    When a sandbox result/check is supplied, ``sandbox_passed`` and
    ``status`` reflect it (``"sandbox_passed"`` vs ``"proposed"``);
    without one the proposal stays ``"proposed"`` with
    ``sandbox_passed: False``.
    """
    proposal = propose_fix(cluster)
    if sandbox_result is not None:
        proposal["sandbox_passed"] = bool(sandbox_result.get("passed"))
        proposal["sandbox_checks"] = sandbox_result.get("checks", [])
    elif check_fn is not None or code_text is not None:
        result = sandbox_check(proposal, check_fn=check_fn, code_text=code_text)
        proposal["sandbox_passed"] = bool(result["passed"])
        proposal["sandbox_checks"] = result["checks"]
    proposal["status"] = (
        "sandbox_passed" if proposal.get("sandbox_passed") else "proposed")
    proposal["applied"] = False
    return proposal


def apply_proposal(
    proposal: dict[str, Any],
    *,
    approve: bool = False,
    sandbox_result: dict[str, Any] | None = None,
    check_fn: Callable[[dict[str, Any]], Any] | None = None,
    code_text: str | None = None,
) -> dict[str, Any]:
    """Authorize a proposal for application — the only path to applied.

    Returns a *new* dict; the input proposal is never mutated.  Rules:

    - ``approve=False`` → ``applied: False``, status
      ``"awaiting_approval"``.  Approval is explicit, never implied.
    - No sandbox pass (neither on the proposal, nor in
      ``sandbox_result``, nor from a fresh ``check_fn``/``code_text``
      run) → ``applied: False``, status ``"sandbox_required"`` — even
      with approval.
    - ``approve=True`` + sandbox pass → ``applied: True``, status
      ``"applied"``.  This marks the proposal authorized; it does NOT
      execute any fix code — the caller performs the change.

    A proposal that was built without a sandbox pass can never be
    applied by accident: every path through this function re-checks
    the sandbox gate.
    """
    out = dict(proposal or {})
    if not approve:
        out["applied"] = False
        out["status"] = "awaiting_approval"
        out["apply_blocked_reason"] = "owner approval required (approve=True)"
        return out
    sandbox_passed = bool(out.get("sandbox_passed"))
    if sandbox_result is not None:
        sandbox_passed = bool(sandbox_result.get("passed"))
        out["sandbox_checks"] = sandbox_result.get("checks", [])
    elif check_fn is not None or code_text is not None:
        result = sandbox_check(out, check_fn=check_fn, code_text=code_text)
        sandbox_passed = bool(result["passed"])
        out["sandbox_checks"] = result["checks"]
    out["sandbox_passed"] = sandbox_passed
    if not sandbox_passed:
        out["applied"] = False
        out["status"] = "sandbox_required"
        out["apply_blocked_reason"] = (
            "sandbox check has not passed — fix must pass sandbox first")
        return out
    out["applied"] = True
    out["status"] = "applied"
    out["applied_at"] = time.time()
    out.pop("apply_blocked_reason", None)
    return out


class FailureLoop:
    """Object wrapper over the cluster -> fix -> sandbox -> proposal flow."""

    def __init__(
        self,
        store: TrajectoryStore | None = None,
        kb: FailureKB | None = None,
    ) -> None:
        self.store = store if store is not None else TrajectoryStore(":memory:")
        self.kb = kb if kb is not None else FailureKB(store=self.store)

    def cluster_failures(self, task_kind: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Open clusters via the KB triage (see :func:`cluster_failures`)."""
        return cluster_failures(self.kb, task_kind=task_kind, limit=limit)

    def propose_fix(self, cluster: dict[str, Any]) -> dict[str, Any]:
        return propose_fix(cluster)

    def sandbox_check(
        self,
        proposal: dict[str, Any],
        *,
        check_fn: Callable[[dict[str, Any]], Any] | None = None,
        code_text: str | None = None,
    ) -> dict[str, Any]:
        return sandbox_check(proposal, check_fn=check_fn, code_text=code_text)

    def build_proposal(self, cluster: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return build_proposal(cluster, **kwargs)

    def apply_proposal(self, proposal: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return apply_proposal(proposal, **kwargs)

    def report(self, limit: int = 10) -> str:
        """Owner-readable loop status: clusters + their proposals."""
        clusters = self.cluster_failures(limit=limit)
        lines = ["🔁 Failure loop"]
        if not clusters:
            lines.append("No open failure clusters — nothing to propose.")
            return "\n".join(lines)
        for cluster in clusters:
            proposal = propose_fix(cluster)
            lines.append(
                f"• {proposal['task_kind']}::{proposal['error_signature']}"
                f" — {proposal['count']}×, priority {proposal['priority']}"
                f" → fix [{proposal['fix_kind']}]: {proposal['suggested_fix'][:90]}"
                f" (status: {proposal['status']}, applied: False)")
        return "\n".join(lines)
