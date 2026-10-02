"""Skill self-rewrite loop — rewrite failing skills, safely, through a gate.

Why a separate loop instead of extending ``ImprovementLoop``: the improvement
loop works in *benchmark dimensions* (reasoning, planning, tool_use,
self_correction) and edits repo files via ``EvolutionAgent`` proposals. The
skill loop works in *failure clusters attributed to a skill* and edits either
a skill file under ``nomorals/skills/`` or a skill body stored in the
``SkillLibrary`` database. Different trigger (failures, not benchmark
scores), different target (skill text, not lever files), different gate
(skill tests + static sanity + benchmark verify, not dimension re-measure).
Sharing the mode gating (``settings.improvement.mode``) and the
propose->gate->apply->revert shape keeps the two loops conceptually aligned
without forcing one abstraction to serve both.

One cycle:

  1. **detect** — group recent failures by skill attribution; a skill
     implicated in >= N failures (default 3) inside the window (default 7
     days) becomes a candidate.
  2. **propose** — ask the model (or an injected proposer) for a *surgical*
     unified diff against the current skill text, seeded with the failing
     cases. Diffs are capped (default <= 60 changed lines); larger rewrites
     are staged for approval even in autonomous mode.
  3. **gate** — static sanity (file parses / frontmatter valid), the skill's
     own tests (run against a scratch copy, repo restored afterwards), and a
     benchmark baseline capture. ALL must pass before anything is committed.
  4. **apply** — ``off`` refuses, ``approval`` stages the diff for the owner,
     ``autonomous`` commits it through the gate, then verifies the benchmark
     did not regress; a regression reverts to the previous hash.
  5. **record** — every proposal lands in ``skill_edits`` with before/after
     hashes, the diff, triggering failures, gate results, mode, and status.
     Reverted diffs are fingerprinted so the loop never proposes the same
     edit twice.

The loop NEVER touches credentials/vault material, connector tokens,
settings secrets, or anything under ``nomorals/connectors/`` — see
``EXCLUDED_PATTERNS``, enforced before any write.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.diff import (
    _FilePatch,
    _apply_parsed_diff,
    _parse_unified_diff,
    DiffApplyError,
)
from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .failure import FailureAnalyzer
from .skills import SkillLibrary

_log = get_logger(__name__)

__all__ = [
    "SkillEvolutionLoop", "SkillEditRecord", "SkillTarget",
    "EXCLUDED_PATTERNS", "apply_unified_diff", "diff_fingerprint",
    "SkillEvolutionError", "ensure_improvement_schedule", "register",
]

# ── safety: paths/content the loop must never edit ──────────────────────────
EXCLUDED_PATTERNS = (
    "nomorals/connectors/",   # connector auth flows + tokens
    "connectors/",            # any connector auth path, wherever rooted
    "vault",                  # secrets vault material
    "credential",
    ".env",
    "api_key",
    "apikey",
    "nomorals/core/config.py",  # settings incl. secrets
)

MAX_DIFF_LINES = 60            # changed (+/-) lines per proposal
DEFAULT_MIN_FAILURES = 3       # failures implicating one skill to trigger
DEFAULT_WINDOW_DAYS = 7.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILLS_DIR = _REPO_ROOT / "nomorals" / "skills"


class SkillEvolutionError(Exception):
    """Raised when the loop cannot proceed safely (excluded target, no
    proposer, oversized diff, ...)."""


# ── diff utilities ──────────────────────────────────────────────────────────
def diff_fingerprint(diff_text: str) -> str:
    """Stable fingerprint of a diff so reverted edits are never re-proposed."""
    norm = "\n".join(
        l.rstrip() for l in (diff_text or "").splitlines()
        if l.strip() not in ("", "\\ No newline at end of file"))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def count_changed_lines(diff_text: str) -> int:
    n = 0
    for line in (diff_text or "").splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            n += 1
    return n


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# Pseudo target for the single-text adapter below: the canonical engine is
# file-oriented, while this contract applies every hunk to one text.
_ADAPTER_TARGET = "skill_text"


def _bare_hunks(diff_text: str) -> bool:
    """True when the text carries ``@@`` hunks but no ``---``/``+++`` file
    headers — the old hand-rolled parser accepted those bare-hunk diffs."""
    has_hunk = False
    for raw in (diff_text or "").splitlines():
        if raw.startswith(("--- ", "+++ ")):
            return False
        if raw.startswith("@@") and _HUNK_RE.match(raw):
            has_hunk = True
    return has_hunk


def apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply a unified diff to ``original`` text.

    Thin adapter over the canonical engine in :mod:`nomorals.core.diff`
    (differential-fuzzed 500/500 against GNU ``patch``: count-driven hunk
    parsing, patch-style offset tolerance, ``\\ No newline`` handling).
    The contract is unchanged: returns the patched text, raises
    :class:`SkillEvolutionError` when no hunks are found or a hunk does
    not apply cleanly.

    Adapter semantics (kept from the previous hand-rolled parser):
    file headers are validated but the target path is ignored — every
    hunk in the diff is applied to ``original`` in order.  Bare ``@@``
    hunks without ``---``/``+++`` headers are still accepted.

    Two deliberate improvements from the canonical engine: a diff
    ``\\ No newline at end of file`` marker is now authoritative for the
    result's trailing newline (previously silently ignored), and hunks
    apply with GNU patch offset semantics instead of failing on
    overlaps.
    """
    text = diff_text or ""
    try:
        patches = _parse_unified_diff(text)
        if not patches and _bare_hunks(text):
            patches = _parse_unified_diff(
                f"--- a/{_ADAPTER_TARGET}\n+++ b/{_ADAPTER_TARGET}\n{text}")
    except DiffApplyError as exc:
        raise SkillEvolutionError(str(exc)) from exc
    hunks = [h for fp in patches for h in fp.hunks]
    if not hunks:
        raise SkillEvolutionError("no hunks found in diff")
    pseudo = _FilePatch(old_path=_ADAPTER_TARGET, new_path=_ADAPTER_TARGET,
                        hunks=hunks)
    try:
        result = _apply_parsed_diff([pseudo], {_ADAPTER_TARGET: original})
    except DiffApplyError as exc:
        raise SkillEvolutionError(str(exc)) from exc
    return result[_ADAPTER_TARGET]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── records ─────────────────────────────────────────────────────────────────
@dataclass
class SkillEditRecord:
    id: str
    skill_name: str
    target_kind: str          # db | file
    target_ref: str           # skill id or repo-relative path
    before_hash: str
    after_hash: str
    diff: str
    triggering_failures: list[dict[str, Any]] = field(default_factory=list)
    gate_results: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    status: str = "proposed"  # proposed|staged|applied|reverted|regressed|refused
    fingerprint: str = ""
    created_at: float = 0.0
    decided_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "skill_name": self.skill_name,
            "target_kind": self.target_kind, "target_ref": self.target_ref,
            "before_hash": self.before_hash, "after_hash": self.after_hash,
            "diff": self.diff, "triggering_failures": self.triggering_failures,
            "gate_results": self.gate_results, "mode": self.mode,
            "status": self.status, "fingerprint": self.fingerprint,
            "created_at": self.created_at, "decided_at": self.decided_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SkillEditRecord":
        try:
            failures = json.loads(row.get("triggering_failures") or "[]")
        except Exception:  # noqa: BLE001
            failures = []
        try:
            gate = json.loads(row.get("gate_results") or "{}")
        except Exception:  # noqa: BLE001
            gate = {}
        return cls(
            id=row["id"], skill_name=row.get("skill_name", ""),
            target_kind=row.get("target_kind", ""),
            target_ref=row.get("target_ref", ""),
            before_hash=row.get("before_hash", ""),
            after_hash=row.get("after_hash", ""),
            diff=row.get("diff", ""), triggering_failures=failures,
            gate_results=gate, mode=row.get("mode", ""),
            status=row.get("status", "proposed"),
            fingerprint=row.get("fingerprint", ""),
            created_at=float(row.get("created_at", 0)),
            decided_at=float(row.get("decided_at", 0)))


@dataclass
class SkillTarget:
    """Where a skill's editable text lives."""
    kind: str          # db | file
    ref: str           # skill id (db) or repo-relative path (file)
    name: str
    text: str

    @property
    def test_slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")

    def test_path(self) -> Path:
        return _REPO_ROOT / "tests" / f"test_skill_{self.test_slug}.py"


# ── the loop ────────────────────────────────────────────────────────────────
class SkillEvolutionLoop:
    def __init__(
        self,
        context: Any,
        *,
        proposer: Callable[[str, str, list[dict[str, Any]]], str] | None = None,
        test_runner: Callable[[list[str]], tuple[int, str]] | None = None,
        benchmark_fn: Callable[[str, list[dict[str, Any]]], dict[str, Any]]
        | None = None,
        min_failures: int = DEFAULT_MIN_FAILURES,
        window_days: float = DEFAULT_WINDOW_DAYS,
        max_diff_lines: int = MAX_DIFF_LINES,
    ) -> None:
        self.context = context
        self.db = context.db
        self.skills = SkillLibrary(context.db)
        self._proposer = proposer or self._default_proposer
        self._test_runner = test_runner or self._default_test_runner
        self._benchmark_fn = benchmark_fn or self._default_benchmark
        self.min_failures = min_failures
        self.window_days = window_days
        self.max_diff_lines = max_diff_lines

    @property
    def settings(self):
        return self.context.settings.improvement

    @property
    def mode(self) -> str:
        return (self.settings.mode or "off").strip().lower()

    # ── 1. detect ───────────────────────────────────────────────────────────
    def detect(self) -> list[dict[str, Any]]:
        """Group recent failures by skill attribution. Returns candidates
        ``[{skill, count, failures}]`` for skills at/above the threshold."""
        cutoff = time.time() - self.window_days * 86400.0
        try:
            rows = self.db.query(
                "SELECT * FROM failures WHERE ts > ? AND skill<>'' "
                "ORDER BY ts DESC", (cutoff,))
        except Exception:  # noqa: BLE001 — pre-migration-52 databases
            rows = []
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r["skill"], []).append(dict(r))
        return [
            {"skill": skill, "count": len(failures), "failures": failures}
            for skill, failures in sorted(
                groups.items(), key=lambda kv: len(kv[1]), reverse=True)
            if len(failures) >= self.min_failures
        ]

    # ── target resolution ───────────────────────────────────────────────────
    def resolve_target_by_id(self, skill_id: str) -> SkillTarget | None:
        """Resolve a SkillLibrary row by id (used for lesson promotion)."""
        try:
            skill = self.skills.get(skill_id)
        except Exception:  # noqa: BLE001
            skill = None
        if skill is None:
            return None
        return SkillTarget(kind="db", ref=skill.id,
                           name=skill.name, text=skill.body or "")

    def resolve_target(self, skill_name: str) -> SkillTarget | None:
        """Map a skill name to its editable text: a file under
        ``nomorals/skills/`` when one matches, else the SkillLibrary row."""
        slug = re.sub(r"[^a-z0-9]+", "_", (skill_name or "").lower()).strip("_")
        if _SKILLS_DIR.is_dir():
            for path in sorted(_SKILLS_DIR.glob("*.py")):
                if path.stem == slug or slug in path.stem or path.stem in slug:
                    try:
                        text = path.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    return SkillTarget(
                        kind="file",
                        ref=str(path.relative_to(_REPO_ROOT)),
                        name=skill_name, text=text)
        try:
            skill = self.skills.get_by_name(skill_name)
        except Exception:  # noqa: BLE001
            skill = None
        if skill is not None:
            return SkillTarget(kind="db", ref=skill.id,
                               name=skill.name, text=skill.body or "")
        return None

    # ── 2. propose ──────────────────────────────────────────────────────────
    def propose(self, skill_name: str,
                failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Propose a surgical diff for a skill. Returns the proposal dict;
        raises ``SkillEvolutionError`` on exclusion or oversized diff."""
        target = self.resolve_target(skill_name)
        if target is None:
            raise SkillEvolutionError(f"no editable target for skill {skill_name!r}")
        self._check_excluded(target, "")
        if failures is None:
            failures = [f for c in self.detect() if c["skill"] == skill_name
                        for f in c["failures"]]
        diff = (self._proposer(skill_name, target.text, failures[:8]) or "").strip()
        diff = self._strip_fences(diff)
        changed = count_changed_lines(diff)
        if changed > self.max_diff_lines and self.mode != "approval":
            raise SkillEvolutionError(
                f"diff touches {changed} lines (cap {self.max_diff_lines}); "
                "larger rewrites require approval mode even in autonomous")
        new_text = apply_unified_diff(target.text, diff)
        self._check_excluded(target, diff + "\n" + new_text)
        return {
            "skill_name": skill_name, "target_kind": target.kind,
            "target_ref": target.ref, "diff": diff,
            "changed_lines": changed, "before_hash": _sha(target.text),
            "after_hash": _sha(new_text), "new_text": new_text,
            "failures": failures[:8],
            "fingerprint": diff_fingerprint(diff),
        }

    @staticmethod
    def _strip_fences(diff: str) -> str:
        lines = diff.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _default_proposer(self, skill_name: str, current_text: str,
                          failures: list[dict[str, Any]]) -> str:
        router = getattr(self.context, "router", None)
        if router is None:
            raise SkillEvolutionError(
                "no model router available for skill proposals; "
                "pass proposer= explicitly")
        from ..llm.base import Message, SamplingParams
        evidence = "\n".join(
            f"- [{f.get('source', '')}] {(f.get('summary', '') or '')[:120]}: "
            f"{(f.get('error', '') or '')[:300]}"
            for f in failures[:5]) or "(no failure details)"
        prompt = (
            f"You are improving the Devon skill '{skill_name}'. Make the "
            f"SMALLEST surgical edit that addresses these failures:\n"
            f"{evidence}\n\nCurrent skill text:\n{current_text[:6000]}\n\n"
            "Respond with ONLY a unified diff (--- / +++ headers, @@ hunks) "
            f"against the current text, at most {self.max_diff_lines} changed "
            "lines. No explanation, no code fences.")
        try:
            resp = router.chat([Message.user(prompt)],
                               SamplingParams(temperature=0.2, max_tokens=2000))
            if not getattr(resp, "ok", False):
                raise SkillEvolutionError("model proposal failed")
            return resp.text or ""
        except SkillEvolutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SkillEvolutionError(f"model proposal failed: {exc}") from exc

    # ── exclusion enforcement ───────────────────────────────────────────────
    def _check_excluded(self, target: SkillTarget, text: str) -> None:
        low_ref = f"{target.ref} {target.name}".lower()
        for pat in EXCLUDED_PATTERNS:
            if pat.lower() in low_ref:
                raise SkillEvolutionError(
                    f"refused: target {target.ref!r} matches excluded "
                    f"pattern {pat!r}")
        low_text = (text or "").lower()
        for pat in ("connectors/", "vault", "credential", ".env", "api_key",
                    "apikey"):
            if pat in low_text:
                raise SkillEvolutionError(
                    f"refused: proposed edit touches excluded material "
                    f"({pat!r})")

    # ── 3. gate ─────────────────────────────────────────────────────────────
    def gate(self, proposal: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Run the pre-apply gate. Returns (passed, results). ALL phases must
        pass before the edit is committed."""
        target = SkillTarget(kind=proposal["target_kind"],
                             ref=proposal["target_ref"],
                             name=proposal["skill_name"],
                             text="")  # text unused by the gate
        results: dict[str, Any] = {}

        # phase 1 — static sanity: the edited text parses
        ok, detail = self._gate_static(proposal)
        results["static"] = {"ok": ok, "detail": detail}
        if not ok:
            return False, results

        # phase 2 — the skill's own tests, against a scratch copy so the
        # repo is restored afterwards no matter what
        ok, detail = self._gate_tests(target, proposal)
        results["tests"] = {"ok": ok, "detail": detail}
        if not ok:
            return False, results

        # phase 3 — benchmark baseline capture; the no-regression check runs
        # post-commit with auto-revert (same shape as ImprovementLoop)
        try:
            results["benchmark"] = self._benchmark_fn(
                proposal["skill_name"], proposal.get("failures", []))
        except Exception as exc:  # noqa: BLE001
            results["benchmark"] = {"ok": True, "skipped": True,
                                    "detail": f"baseline failed: {exc}"[:200]}
        return True, results

    def _gate_static(self, proposal: dict[str, Any]) -> tuple[bool, str]:
        new_text = proposal["new_text"]
        ref = proposal["target_ref"]
        try:
            if ref.endswith(".py"):
                compile(new_text, ref or "<skill>", "exec")
                return True, "python compiles"
            if ref.endswith(".md") or proposal["target_kind"] == "db":
                # SkillLibrary bodies are text/Markdown, not Python
                self._check_frontmatter(new_text)
                return True, "skill text valid"
            return True, "no static check for this target type"
        except (SyntaxError, ValueError) as exc:
            return False, f"static check failed: {exc}"[:300]

    @staticmethod
    def _check_frontmatter(text: str) -> None:
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end < 0:
                raise ValueError("unterminated frontmatter")
            try:
                import yaml  # local import: only needed for md skills
            except ImportError:
                return  # PyYAML is optional; the delimiter check above stands
            yaml.safe_load(text[3:end])

    def _gate_tests(self, target: SkillTarget,
                    proposal: dict[str, Any]) -> tuple[bool, str]:
        test_path = target.test_path()
        if not test_path.is_file():
            return True, f"skipped: no skill test at {test_path.name}"
        backup: bytes | None = None
        applied = False
        try:
            if target.kind == "file":
                full = _REPO_ROOT / target.ref
                backup = full.read_bytes()
                full.write_text(proposal["new_text"], encoding="utf-8")
                applied = True
            # db skills: the registered body is what the test file exercises;
            # the test run below validates the suite is green pre-commit
            code, out = self._test_runner(
                ["-m", "pytest", str(test_path), "-q"])
            if code == 0:
                return True, f"skill tests pass ({test_path.name})"
            return False, f"skill tests failed:\n{out[-1500:]}"
        except Exception as exc:  # noqa: BLE001
            return False, f"test harness error: {exc}"[:300]
        finally:
            if applied and backup is not None:
                try:
                    (_REPO_ROOT / target.ref).write_bytes(backup)
                except OSError:
                    _log.error("failed to restore scratch file %s", target.ref)

    @staticmethod
    def _default_test_runner(cmd: list[str]) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                ["python3", *cmd], capture_output=True, text=True,
                timeout=300, cwd=str(_REPO_ROOT))
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except Exception as exc:  # noqa: BLE001
            return 1, f"test runner failed: {exc}"[:500]

    def _default_benchmark(self, skill_name: str,
                           failures: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from .benchmark import measurable, run_benchmark
        except Exception:  # noqa: BLE001
            return {"ok": True, "skipped": True,
                    "detail": "no benchmark module"}
        if not measurable(self.context):
            return {"ok": True, "skipped": True,
                    "detail": "provider not measurable"}
        try:
            report = run_benchmark(self.context, limit=1)
            scores = {k: v.score for k, v in report.scores.items()
                      if v.score is not None}
            return {"ok": True, "skipped": False, "baseline": scores,
                    "detail": f"baseline captured: {scores}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": True, "skipped": True,
                    "detail": f"baseline failed: {exc}"[:200]}

    # ── 4. apply / revert ───────────────────────────────────────────────────
    def apply_proposal(self, proposal: dict[str, Any], *,
                       mode: str = "") -> SkillEditRecord:
        """Commit a proposal through the gate per ``mode``. In autonomous
        mode the benchmark no-regression check runs post-commit with
        auto-revert, so a shipped edit always passed the full gate."""
        mode = (mode or self.mode or "off").strip().lower()
        rec = SkillEditRecord(
            id=new_short_id("skedit"), skill_name=proposal["skill_name"],
            target_kind=proposal["target_kind"],
            target_ref=proposal["target_ref"],
            before_hash=proposal["before_hash"],
            after_hash=proposal["after_hash"], diff=proposal["diff"],
            triggering_failures=[
                {"id": f.get("id"), "source": f.get("source"),
                 "summary": (f.get("summary") or "")[:160],
                 "family": f.get("family", "")}
                for f in proposal.get("failures", [])],
            mode=mode, status="proposed",
            fingerprint=proposal["fingerprint"], created_at=time.time())
        if mode == "off":
            rec.status = "refused"
            rec.gate_results = {"refused": "improvement is off"}
            self._store(rec)
            return rec
        if self._known_regressed(rec.fingerprint):
            rec.status = "refused"
            rec.gate_results = {"refused": "diff fingerprint already regressed"}
            self._store(rec)
            return rec

        passed, results = self.gate(proposal)
        rec.gate_results = results
        if not passed:
            rec.status = "reverted"
            rec.decided_at = time.time()
            self._store(rec)
            return rec

        if mode == "approval":
            # stage the diff, then file it into the owner's upgrade queue
            # — the queue (not this table) is the approve/deny surface.
            rec.status = "staged"
            rec.decided_at = time.time()
            self._store(rec)
            rec.gate_results["upgrade_proposal_id"] = self._file_for_approval(
                proposal, rec)
            self._store(rec)
            return rec

        # autonomous: canary first when applicable (default ON); otherwise
        # commit, then verify no benchmark regression
        canary_run = self._start_canary_if_applicable(proposal)
        if canary_run is not None:
            rec.status = "canary"
            rec.gate_results["canary"] = canary_run
            # settle_canaries() matches the edit by canary hash — record
            # the run's real version hash, not the plain content sha.
            rec.after_hash = canary_run["canary_hash"]
            rec.decided_at = time.time()
            self._store(rec)
            return rec

        self._commit(proposal)
        verify = self._verify_no_regression(proposal, results)
        rec.gate_results["verify"] = verify
        if not verify.get("ok", True):
            self._restore(proposal)
            rec.status = "regressed"
        else:
            rec.status = "applied"
        rec.decided_at = time.time()
        self._store(rec)
        return rec

    def _file_for_approval(self, proposal: dict[str, Any],
                           rec: SkillEditRecord) -> str:
        """File a staged skill edit into the owner's upgrade queue.

        Returns the queue proposal id, or "" when the queue is
        unavailable (the staged row is still reviewable via
        ``skill_evolve history``).  Never raises.
        """
        try:
            from .upgrade_queue import UpgradeQueue

            failures = "; ".join(
                str(f.get("summary") or f.get("error") or "")[:80]
                for f in (proposal.get("failures") or [])[:3])
            queue = UpgradeQueue(self.context)
            return queue.propose(
                title=f"skill edit: {rec.skill_name}",
                rationale=(
                    f"Skill '{rec.skill_name}' implicated in repeated "
                    f"failures; proposed surgical fix ({rec.fingerprint}). "
                    f"Triggering failures: {failures or 'n/a'}"[:500]),
                patch_plan={
                    "source": "skill_evolution",
                    "skill_edit_id": rec.id,
                    "skill_name": rec.skill_name,
                    "target_kind": rec.target_kind,
                    "target_ref": rec.target_ref,
                    "changed_lines": proposal.get("changed_lines", 0),
                },
                files=[rec.target_ref] if rec.target_kind == "file" else [],
                tests=[f"skill tests for {rec.skill_name}",
                       "benchmark no-regression check"],
                source="skill_evolution",
            )
        except Exception as exc:  # noqa: BLE001 - filing is best-effort
            _log.warning("could not file skill edit %s to the upgrade "
                         "queue: %s", rec.id, exc)
            return ""

    def _commit(self, proposal: dict[str, Any]) -> None:
        if proposal["target_kind"] == "file":
            path = _REPO_ROOT / proposal["target_ref"]
            path.write_text(proposal["new_text"], encoding="utf-8")
        else:
            skill = self.skills.get_by_name(proposal["skill_name"])
            if skill is None:
                raise SkillEvolutionError("skill vanished before commit")
            self.skills.save(skill.name, kind=skill.kind,
                             body=proposal["new_text"],
                             description=skill.description,
                             tags=list(skill.tags), source=skill.source,
                             id=skill.id)

    def _restore(self, proposal: dict[str, Any]) -> None:
        """Restore the pre-edit text (revert)."""
        before = self._text_before(proposal)
        if proposal["target_kind"] == "file":
            (_REPO_ROOT / proposal["target_ref"]).write_text(
                before, encoding="utf-8")
        else:
            skill = self.skills.get_by_name(proposal["skill_name"])
            if skill is not None:
                self.skills.save(skill.name, kind=skill.kind, body=before,
                                 description=skill.description,
                                 tags=list(skill.tags), source=skill.source,
                                 id=skill.id)

    def _text_before(self, proposal: dict[str, Any]) -> str:
        target = self.resolve_target(proposal["skill_name"])
        if target is None:
            raise SkillEvolutionError("cannot restore: target vanished")
        # the target currently holds the NEW text; reverse-apply the diff
        reverse = _reverse_diff(proposal["diff"])
        return apply_unified_diff(target.text, reverse)

    def _verify_no_regression(self, proposal: dict[str, Any],
                              gate_results: dict[str, Any]) -> dict[str, Any]:
        bench = gate_results.get("benchmark") or {}
        baseline = bench.get("baseline")
        if not baseline:
            return {"ok": True, "skipped": True,
                    "detail": "no benchmark baseline to compare"}
        try:
            after = self._benchmark_fn(proposal["skill_name"],
                                       proposal.get("failures", []))
        except Exception as exc:  # noqa: BLE001
            return {"ok": True, "skipped": True,
                    "detail": f"verify failed: {exc}"[:200]}
        after_scores = after.get("baseline") or {}
        regressions = {k: (baseline[k], after_scores.get(k)) for k in baseline
                       if k in after_scores
                       and after_scores[k] < baseline[k] - 1e-9}
        if regressions:
            return {"ok": False, "regressions": regressions,
                    "detail": f"benchmark regressed: {regressions}"}
        return {"ok": True, "detail": "no benchmark regression"}

    def _start_canary_if_applicable(
            self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Start a canary rollout for a DB skill edit in autonomous mode.

        The live skill keeps serving the baseline body; ``fraction`` of
        tasks see the new version until the canary is settled. File targets
        cannot canary (one repo file); they apply directly. Returns the
        canary run dict, or None when canary does not apply.
        """
        if proposal["target_kind"] != "db":
            return None
        try:
            from .skill_canary import CanaryRollout
        except Exception:  # noqa: BLE001
            return None
        try:
            target = self.resolve_target(proposal["skill_name"])
            baseline_body = target.text if target else ""
            rollout = CanaryRollout(self.context)
            # no baseline_hash override: start() records the baseline body
            # as a real skill_versions row, so settle_canaries can fetch it
            # back with get_body() for promotion commits.
            run = rollout.start(
                proposal["skill_name"], proposal["new_text"],
                baseline_body=baseline_body)
            if not run.get("ok"):
                return None
            return {"run_id": run["id"], "fraction": run["fraction"],
                    "canary_hash": run["canary_hash"],
                    "baseline_hash": run["baseline_hash"]}
        except Exception as exc:  # noqa: BLE001
            _log.debug("canary start failed: %s", exc)
            return None

    def settle_canaries(self) -> list[dict[str, Any]]:
        """Evaluate every running canary: promote (commit the canary body,
        mark the edit applied) or revert (mark the edit regressed)."""
        try:
            from .skill_canary import CanaryRollout
        except Exception:  # noqa: BLE001
            return []
        rollout = CanaryRollout(self.context)
        settled: list[dict[str, Any]] = []
        try:
            running = [r for r in rollout.history(limit=50)
                       if r.status == "running"]
        except Exception:  # noqa: BLE001
            return []
        for run in running:
            try:
                outcome = rollout.evaluate(run.skill_name)
            except Exception:  # noqa: BLE001
                continue
            if outcome.get("decision") == "waiting":
                continue
            if outcome.get("decision") == "inconclusive":
                continue  # still collecting data; settle next time
            edit = self._find_canary_edit(run.skill_name, run.canary_hash)
            if outcome.get("decision") == "promote":
                body = rollout.get_body(run.skill_name, run.canary_hash)
                if body is not None:
                    self._commit_body(run.skill_name, body)
                if edit:
                    self.db.execute(
                        "UPDATE skill_edits SET status='applied', "
                        "decided_at=? WHERE id=?", (time.time(), edit["id"]))
            else:  # revert
                if edit:
                    self.db.execute(
                        "UPDATE skill_edits SET status='regressed', "
                        "decided_at=? WHERE id=?", (time.time(), edit["id"]))
            settled.append({"skill": run.skill_name,
                            "decision": outcome.get("decision"),
                            "detail": {k: v for k, v in outcome.items()
                                       if k in ("canary", "baseline")}})
        return settled

    def _find_canary_edit(self, skill_name: str,
                          canary_hash: str) -> dict[str, Any] | None:
        try:
            return self.db.query_one(
                "SELECT * FROM skill_edits WHERE skill_name=? AND "
                "after_hash=? AND status='canary' ORDER BY created_at DESC "
                "LIMIT 1", (skill_name, canary_hash))
        except Exception:  # noqa: BLE001
            return None

    def _commit_body(self, skill_name: str, body: str) -> None:
        skill = self.skills.get_by_name(skill_name)
        if skill is None:
            raise SkillEvolutionError("skill vanished before canary promote")
        self.skills.save(skill.name, kind=skill.kind, body=body,
                         description=skill.description,
                         tags=list(skill.tags), source=skill.source,
                         id=skill.id)

    def _known_regressed(self, fingerprint: str) -> bool:
        row = self.db.query_one(
            "SELECT 1 FROM skill_edits WHERE fingerprint=? AND status IN "
            "('regressed','reverted') LIMIT 1", (fingerprint,))
        return row is not None

    def revert_edit(self, edit_id: str) -> dict[str, Any]:
        """Manually revert an applied edit to its before-hash."""
        row = self.db.query_one("SELECT * FROM skill_edits WHERE id=?",
                                (edit_id,))
        if not row:
            return {"ok": False, "error": "no such edit"}
        rec = SkillEditRecord.from_row(row)
        if rec.status != "applied":
            return {"ok": False,
                    "error": f"edit is {rec.status}, not applied"}
        self._restore({"skill_name": rec.skill_name,
                       "target_kind": rec.target_kind,
                       "target_ref": rec.target_ref, "diff": rec.diff})
        self.db.execute(
            "UPDATE skill_edits SET status='reverted', decided_at=? "
            "WHERE id=?", (time.time(), edit_id))
        return {"ok": True, "edit": edit_id}

    def apply_staged_edit(self, edit_id: str) -> dict[str, Any]:
        """Commit a staged (approval-mode) edit: re-run the gate, then
        commit exactly like the autonomous path.  Called by the upgrade
        queue when the owner approves the filed proposal."""
        row = self.db.query_one("SELECT * FROM skill_edits WHERE id=?",
                                (edit_id,))
        if not row:
            return {"ok": False, "error": "no such edit"}
        rec = SkillEditRecord.from_row(row)
        if rec.status != "staged":
            return {"ok": False,
                    "error": f"edit is {rec.status}, not staged"}
        target = self.resolve_target(rec.skill_name)
        if target is None:
            return {"ok": False, "error": "skill target vanished"}
        try:
            new_text = apply_unified_diff(target.text, rec.diff)
        except SkillEvolutionError as exc:
            return {"ok": False, "edit": edit_id,
                    "error": f"stale diff, no longer applies: {exc}"}
        proposal = {
            "skill_name": rec.skill_name,
            "target_kind": rec.target_kind,
            "target_ref": rec.target_ref,
            "diff": rec.diff,
            "new_text": new_text,
            "failures": rec.triggering_failures,
        }
        passed, results = self.gate(proposal)
        if not passed:
            self.db.execute(
                "UPDATE skill_edits SET status='reverted', "
                "gate_results=?, decided_at=? WHERE id=?",
                (json.dumps(results, default=str), time.time(), edit_id))
            return {"ok": False, "edit": edit_id,
                    "error": "gate failed on re-check", "gate": results}
        try:
            self._commit(proposal)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "edit": edit_id,
                    "error": f"commit failed: {exc}"}
        verify = self._verify_no_regression(proposal, results)
        if not verify.get("ok", True):
            self._restore(proposal)
            self.db.execute(
                "UPDATE skill_edits SET status='regressed', decided_at=? "
                "WHERE id=?", (time.time(), edit_id))
            return {"ok": False, "edit": edit_id,
                    "error": "benchmark regressed — reverted",
                    "verify": verify}
        self.db.execute(
            "UPDATE skill_edits SET status='applied', decided_at=? "
            "WHERE id=?", (time.time(), edit_id))
        return {"ok": True, "edit": edit_id, "status": "applied"}

    def deny_staged_edit(self, edit_id: str) -> dict[str, Any]:
        """Mark a staged edit denied (owner denied it in the upgrade
        queue).  Nothing is written to the skill; the fingerprint is
        kept so the loop never re-proposes the same edit."""
        row = self.db.query_one("SELECT * FROM skill_edits WHERE id=?",
                                (edit_id,))
        if not row:
            return {"ok": False, "error": "no such edit"}
        rec = SkillEditRecord.from_row(row)
        if rec.status not in {"staged", "proposed"}:
            return {"ok": False,
                    "error": f"edit is {rec.status}, not pending"}
        self.db.execute(
            "UPDATE skill_edits SET status='denied', decided_at=? "
            "WHERE id=?", (time.time(), edit_id))
        return {"ok": True, "edit": edit_id, "status": "denied"}

    # ── records ─────────────────────────────────────────────────────────────
    def _store(self, rec: SkillEditRecord) -> None:
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO skill_edits (id, skill_name, "
                "target_kind, target_ref, before_hash, after_hash, diff, "
                "triggering_failures, gate_results, mode, status, "
                "fingerprint, created_at, decided_at) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.id, rec.skill_name, rec.target_kind, rec.target_ref,
                 rec.before_hash, rec.after_hash, rec.diff,
                 json.dumps(rec.triggering_failures),
                 json.dumps(rec.gate_results, default=str), rec.mode,
                 rec.status, rec.fingerprint, rec.created_at,
                 rec.decided_at))
        except Exception:  # noqa: BLE001 — pre-migration-52 databases
            _log.debug("skill_edits store failed", exc_info=True)

    def history(self, *, limit: int = 20,
                skill: str = "") -> list[SkillEditRecord]:
        try:
            q = "SELECT * FROM skill_edits"
            args: tuple = ()
            if skill:
                q += " WHERE skill_name=?"
                args = (skill,)
            q += " ORDER BY created_at DESC LIMIT ?"
            rows = self.db.query(q, (*args, limit))
        except Exception:  # noqa: BLE001
            return []
        return [SkillEditRecord.from_row(r) for r in rows]

    def status(self) -> dict[str, Any]:
        recent = self.history(limit=10)
        by_status: dict[str, int] = {}
        for r in recent:
            by_status[r.status] = by_status.get(r.status, 0) + 1
        analyzer = FailureAnalyzer(self.context)
        return {
            "mode": self.mode,
            "min_failures": self.min_failures,
            "window_days": self.window_days,
            "max_diff_lines": self.max_diff_lines,
            "candidates": [
                {"skill": c["skill"], "count": c["count"]}
                for c in self.detect()],
            "recent_edits": [r.to_dict() for r in recent],
            "by_status": by_status,
            "lessons": analyzer.stats().get("usefulness", {}),
        }

    # ── continuous ──────────────────────────────────────────────────────────
    def tick(self) -> list[SkillEditRecord]:
        """One loop pass: settle running canaries, score lesson usefulness,
        then detect candidates, propose, gate, apply per mode. No-op when
        improvement is off."""
        if self.mode == "off":
            return []
        try:
            settled = self.settle_canaries()
            if settled:
                _log.info("skill_evolution: settled %d canaries", len(settled))
        except Exception:  # noqa: BLE001
            pass
        try:
            stats = FailureAnalyzer(self.context).evaluate_usefulness()
            if stats.get("evaluated") or stats.get("demoted") \
                    or stats.get("archived"):
                _log.info("skill_evolution: usefulness %s", stats)
        except Exception:  # noqa: BLE001
            pass
        out: list[SkillEditRecord] = []
        # high-usefulness lessons get folded back into their governing skill
        try:
            for lesson in FailureAnalyzer(self.context).promotion_candidates():
                if not lesson.skill_id:
                    continue
                target = self.resolve_target_by_id(lesson.skill_id)
                if target is None:
                    continue
                try:
                    proposal = self.propose(target.name)
                except SkillEvolutionError:
                    continue
                out.append(self.apply_proposal(proposal))
        except Exception:  # noqa: BLE001
            pass
        for cand in self.detect():
            try:
                proposal = self.propose(cand["skill"], cand["failures"])
            except SkillEvolutionError as exc:
                _log.info("skill_evolution: no proposal for %s: %s",
                          cand["skill"], exc)
                continue
            out.append(self.apply_proposal(proposal))
        return out


def ensure_improvement_schedule(context: Any) -> list[dict[str, Any]]:
    """Register the prompt-01 recurring jobs on the agent scheduler
    (idempotent by name): daily skill-evolution tick, weekly synthesis scan.
    Both are no-ops while ``settings.improvement.mode`` is ``off``."""
    from .scheduler import Scheduler
    sched = Scheduler(context)
    jobs = [
        ("improve-evolution-tick", "every 24h", "skill_evolve",
         {"action": "tick"}),
        ("improve-synthesis-scan", "every 7d", "skill_synthesize",
         {"action": "scan"}),
    ]
    out = []
    for name, spec, tool, args in jobs:
        try:
            existing = context.db.query_one(
                "SELECT id FROM schedule_jobs WHERE name=? LIMIT 1", (name,))
        except Exception:  # noqa: BLE001
            existing = None
        if existing:
            out.append({"name": name, "already_scheduled": True})
            continue
        try:
            job = sched.add(name, spec, "tool", {"tool": tool, "args": args})
            out.append({"name": name, "scheduled": True,
                        "job_id": job.get("id")})
        except Exception as exc:  # noqa: BLE001
            out.append({"name": name, "error": str(exc)[:200]})
    return out


def _reverse_diff(diff_text: str) -> str:
    """Flip a unified diff so it undoes itself.

    The ``---``/``+++`` header pair is swapped (paths exchanged) so the
    result stays a valid unified diff — ``---`` names the "before" file
    and still comes first.  (The old marker-flip-only version emitted
    ``+++`` before ``---``, which the lenient hand-rolled applier
    tolerated but the canonical engine rightly rejects.)
    """
    out: list[str] = []
    lines = (diff_text or "").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        if (raw.startswith("--- ") and i + 1 < len(lines)
                and lines[i + 1].startswith("+++ ")):
            out.append("--- " + lines[i + 1][4:])
            out.append("+++ " + raw[4:])
            i += 2
            continue
        if raw.startswith("--- "):
            out.append("+++ " + raw[4:])
        elif raw.startswith("+++ "):
            out.append("--- " + raw[4:])
        elif raw.startswith("@@"):
            m = _HUNK_RE.match(raw)
            if m:
                a, ac, b, bc = m.groups()
                out.append(f"@@ -{b},{bc or 1} +{a},{ac or 1} @@")
            else:
                out.append(raw)
        elif raw.startswith("+") and not raw.startswith("+++"):
            out.append("-" + raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
            out.append("+" + raw[1:])
        else:
            out.append(raw)
        i += 1
    return "\n".join(out)


# ── registry ────────────────────────────────────────────────────────────────
def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "skill_evolve",
        description=(
            "Skill self-rewrite loop: detect skills implicated in repeated "
            "failures, propose a surgical diff, gate it (static + skill "
            "tests + benchmark), apply per improvement mode. action=tick | "
            "status | detect | propose <skill> | history | revert <edit_id>. "
            "mode: off|approval|autonomous."
        ),
        capability="model.call",
        parameters={
            "action": "str — tick|status|detect|propose|history|revert",
            "skill": "str — skill name, for propose",
            "edit_id": "str — for revert",
            "limit": "int — for history",
        },
    )
    def skill_evolve(
        action: str = "status", *, skill: str = "", edit_id: str = "",
        limit: str = "10",
    ) -> dict[str, Any]:
        loop = SkillEvolutionLoop(context)
        action = (action or "status").strip().lower()
        if action == "tick":
            recs = loop.tick()
            return {"ok": True,
                    "edits": [r.to_dict() for r in recs]}
        if action == "detect":
            return {"candidates": loop.detect()}
        if action == "propose":
            try:
                proposal = loop.propose(skill)
            except SkillEvolutionError as exc:
                return {"ok": False, "error": str(exc)}
            passed, results = loop.gate(proposal)
            return {"ok": True, "proposal": {
                "skill_name": proposal["skill_name"],
                "changed_lines": proposal["changed_lines"],
                "before_hash": proposal["before_hash"],
                "after_hash": proposal["after_hash"],
                "fingerprint": proposal["fingerprint"]},
                "gate_passed": passed, "gate": results}
        if action == "history":
            try:
                n = int(limit or 10)
            except ValueError:
                n = 10
            return {"edits": [r.to_dict()
                              for r in loop.history(limit=n, skill=skill)]}
        if action == "revert":
            return loop.revert_edit(edit_id)
        return loop.status()
