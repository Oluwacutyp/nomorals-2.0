"""The ship gate: arena builds become real repo upgrades.

The arena's review loop ends at ``approve`` (which only stages files under
``~/.nomorals/arena/approved`` — they never touch the repo).  This module
closes the gap: a pending arena build is *promoted* into an
:class:`~nomorals.agents.evolution.EvolutionProposal` whose edits land
under ``<repo>/arena_builds/<name>/`` — deliberately NOT under
``nomorals/``, so the import tree and the layering map stay untouched —
and the proposal then rides the exact same verify → apply → commit
machinery every other evolution uses.

Pipeline::

    /arena promote <build-id>   → promote_build()  (build → proposal, status "promoted")
    /arena ship                 → ship_queue()     (pending arena proposals)
    /arena apply <proposal-id>  → approve_ship()   (fast gate + EvolutionAgent.apply)
    /arena reject <id> <reason> → deny_ship()      (proposal rejected)

Everything here is local (sqlite + git + the compiler + error_scan) —
no function in this module touches the network.  Bad input fails fast
with :class:`~nomorals.core.errors.ToolError`.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from ...core.errors import ToolError
from ...core.logging_setup import get_logger
from ...tools.error_scan import scan
from ..evolution import EvolutionAgent, EvolutionProposal

_log = get_logger(__name__)

__all__ = [
    "promote_build",
    "verify_proposal_files",
    "ship_queue",
    "approve_ship",
    "deny_ship",
]

# Repo-relative landing zone for shipped arena builds.  Kept outside
# nomorals/ on purpose: shipped modules are untrusted candidate code
# until they earn their place, so they must not join the import tree or
# the layering map just by landing.
SHIP_DIR = "arena_builds"


def _arena_row(db: Any, build_id: str) -> dict[str, Any] | None:
    try:
        return db.query_one("SELECT * FROM arena_builds WHERE id = ?",
                            (build_id,))
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"cannot read arena build {build_id!r}: {exc}") from exc


def promote_build(db: Any, context: Any, build_id: str,
                  repo_root: str | Path) -> str:
    """Turn a pending arena build into an evolution proposal.

    Loads the ``arena_builds`` row (must be ``pending`` with syntax
    ``"ok"``), reads every file under its build dir, and creates an
    :class:`EvolutionProposal` with one new-file edit per build file at
    ``arena_builds/<name>/<rel>``.  The build row flips to ``"promoted"``.
    Returns the proposal id.
    """
    build_id = (build_id or "").strip()
    if not build_id:
        raise ToolError("promote needs a build id")
    if db is None:
        raise ToolError("promote needs a database")
    row = _arena_row(db, build_id)
    if row is None:
        raise ToolError(f"no arena build {build_id!r}")
    status = row.get("status") or ""
    if status != "pending":
        raise ToolError(
            f"build {build_id} is {status!r}, not 'pending' — only pending "
            "builds can be promoted")
    syntax = row.get("syntax") or ""
    if syntax != "ok":
        raise ToolError(
            f"build {build_id} failed the syntax gate: {syntax}")
    src = Path(row.get("dir") or "")
    if not src.is_dir():
        raise ToolError(
            f"build {build_id} files are missing on disk ({src})")
    name = (str(row.get("name") or "").strip() or "module")

    edits: list[dict[str, str]] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if ".." in rel.split("/"):
            raise ToolError(f"build file escapes its directory: {rel}")
        content = path.read_text(encoding="utf-8", errors="replace")
        edits.append({
            "path": f"{SHIP_DIR}/{name}/{rel}",
            "old": "",
            "new": content,
        })
    if not edits:
        raise ToolError(f"build {build_id} contains no files")

    proposal_id = f"evo-arena-{build_id[:8]}-{int(time.time() * 1000)}"
    purpose = (str(row.get("purpose") or "").strip() or name)
    topic = (str(row.get("topic") or "").strip() or "?")
    proposal = EvolutionProposal(
        id=proposal_id,
        instruction=f"[arena] {purpose} (topic: {topic})",
        edits=edits,
        rationale=f"arena build {build_id} · {topic}",
    )
    EvolutionAgent(context)._save(proposal)
    try:
        with db.transaction():
            db.execute(
                "UPDATE arena_builds SET status = ?, decided_at = ? "
                "WHERE id = ?",
                ("promoted", time.time(), build_id),
            )
    except Exception as exc:  # noqa: BLE001
        raise ToolError(
            f"proposal {proposal_id} was created but the build row could "
            f"not be marked promoted: {exc}") from exc
    _log.info("promoted arena build %s → proposal %s (%d files)",
              build_id, proposal_id, len(edits))
    return proposal_id


def verify_proposal_files(edits: list[dict[str, str]] | None,
                          repo_root: str | Path) -> tuple[bool, str]:
    """Fast lint gate over a proposal's new-file edits.

    Every new-file edit (``old == ""``) with a ``.py`` path is compiled
    with the :func:`compile` builtin (no disk needed); then
    :func:`~nomorals.tools.error_scan.scan` runs programmatically over
    the proposed contents and any *error*-severity finding fails the
    gate.  Replacement edits (``old != ""``) are fragments of existing
    files, not whole modules — neither gate applies to them.

    Returns ``(ok, report)``.
    """
    root = Path(repo_root).resolve()
    edits = [e for e in (edits or []) if isinstance(e, dict)]
    new_files = [e for e in edits if not str(e.get("old") or "")]
    problems: list[str] = []
    lintable: list[tuple[str, str]] = []  # (rel path, content)

    for edit in new_files:
        rel = str(edit.get("path") or "").strip().lstrip("/")
        if not rel or ".." in rel.split("/"):
            problems.append(f"{rel or '(empty path)'}: path escapes the repo")
            continue
        try:
            (root / rel).resolve().relative_to(root)
        except ValueError:
            problems.append(f"{rel}: path escapes the repo")
            continue
        if not rel.endswith(".py"):
            continue  # no syntax gate exists for non-python content
        content = str(edit.get("new") or "")
        try:
            compile(content, rel, "exec")
        except SyntaxError as exc:
            problems.append(f"{rel}: syntax error: {exc}")
            continue
        lintable.append((rel, content))

    if lintable and not problems:
        # error_scan reads from disk — stage the proposed contents in a
        # throwaway temp dir so the public scan() API sees real files.
        with tempfile.TemporaryDirectory(prefix="arena-ship-") as tmp:
            tmpdir = Path(tmp)
            for rel, content in lintable:
                dest = tmpdir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            report = scan([str(tmpdir)])
            for finding in report.errors:
                try:
                    shown = str(Path(finding.file).relative_to(tmpdir))
                except ValueError:
                    shown = finding.file
                problems.append(
                    f"{shown}:{finding.line} "
                    f"[{finding.rule}/{finding.severity}] {finding.message}")

    if problems:
        return False, ("ship gate failed:\n"
                       + "\n".join(f"  - {p}" for p in problems))
    return True, (f"ship gate passed: {len(new_files)} new file(s), "
                  f"{len(lintable)} python, syntax ok, error_scan clean")


def ship_queue(db: Any, context: Any) -> list[dict[str, Any]]:
    """Arena proposals awaiting an owner decision.

    :class:`EvolutionAgent.list` filtered to ``planned``/``verifying``
    proposals that came from the arena (``evo-arena-`` id prefix or the
    ``[arena]`` instruction marker).  Returns one dict per proposal:
    ``{id, instruction, status, edits:[paths], created_at}``.
    """
    out: list[dict[str, Any]] = []
    for proposal in EvolutionAgent(context).list(200):
        if proposal.status not in ("planned", "verifying"):
            continue
        if not (proposal.id.startswith("evo-arena-")
                or "[arena]" in proposal.instruction):
            continue
        out.append({
            "id": proposal.id,
            "instruction": proposal.instruction,
            "status": proposal.status,
            "edits": [str(e.get("path", "")) for e in proposal.edits
                      if isinstance(e, dict)],
            "created_at": proposal.created_at,
        })
    return out


def approve_ship(db: Any, context: Any, proposal_id: str,
                 repo_root: str | Path, *, commit: bool = True,
                 full_suite: bool = False) -> dict[str, Any]:
    """Ship a pending arena proposal into the repo.

    Runs :func:`verify_proposal_files` first (a failure rejects the
    proposal and is reported, nothing is written); then delegates to
    :meth:`EvolutionAgent.apply`, whose result dict is returned as-is.
    ``EvolutionAgent.apply`` requires a clean git tree and raises
    :class:`ToolError` when it isn't — surfaced honestly, never
    swallowed.
    """
    proposal_id = (proposal_id or "").strip()
    if not proposal_id:
        raise ToolError("approve_ship needs a proposal id")
    agent = EvolutionAgent(context, repo_root=Path(repo_root))
    proposal = agent._load(proposal_id)
    if proposal is None:
        raise ToolError(f"no evolution proposal {proposal_id!r}")
    if proposal.status not in ("planned", "verifying"):
        raise ToolError(
            f"proposal {proposal_id} is {proposal.status!r}, not pending")
    ok, report = verify_proposal_files(proposal.edits, repo_root)
    if not ok:
        proposal.status = "rejected"
        proposal.verify_result = report
        agent._save(proposal)
        return {"applied": False, "proposal": proposal_id,
                "status": "rejected", "reason": "ship gate failed",
                "report": report}
    return agent.apply(proposal_id, verify=full_suite, commit=commit)


def deny_ship(db: Any, context: Any, proposal_id: str,
              reason: str = "") -> bool:
    """Reject a pending arena proposal.  Returns False when missing."""
    proposal_id = (proposal_id or "").strip()
    if not proposal_id:
        return False
    agent = EvolutionAgent(context)
    proposal = agent._load(proposal_id)
    if proposal is None:
        return False
    proposal.status = "rejected"
    proposal.verify_result = (
        f"denied by owner: {(reason or '').strip() or 'no reason given'}")
    agent._save(proposal)
    return True
