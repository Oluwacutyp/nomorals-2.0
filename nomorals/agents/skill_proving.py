"""H.O.T-Jarvis test-proof skill evolution — no activation without proof.

Extension of skill_distillation (#2): every generated skill goes through
``skill_canary.py`` with an AUTO-GENERATED test.

Lifecycle::

    draft -> test -> pass -> active      (proven skills activate)
    draft -> test -> fail -> flagged     (failed skills are disabled + flagged)

Untested skills are flagged and refused — a skill with no passing proof
test never activates, even when a canary run's numbers look good.  The
proof gate is wired into ``CanaryRollout.evaluate`` via ``proof_check``:
promotion of a version without a passing proof is refused.

EloPhanto pattern ("writes its own tools when missing"):
``ensure_capability()`` drafts the missing tool + an auto-generated test,
proves it, and activates only on a passing test.

All entry points never raise; failures are returned as data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "ProofResult",
    "SKILL_PROVING_DDL",
    "canary_proof_check",
    "ensure_capability",
    "ensure_tables",
    "flag_skill",
    "generate_test",
    "get_flag",
    "has_proof",
    "promote_on_proof",
    "prove_skill",
    "record_proof",
    "refuse_untested",
    "run_proof_test",
    "skill_lifecycle",
    "unflag_skill",
]

#: lifecycle states
DRAFT, TESTING, ACTIVE, FLAGGED, INACTIVE, UNKNOWN = (
    "draft", "testing", "active", "flagged", "inactive", "unknown")

#: cap generated test size so a runaway LLM can't bloat the run
MAX_TEST_CHARS = 8000

#: sandbox timeout for one proof test
PROOF_TIMEOUT_S = 90

SKILL_PROVING_DDL = """
CREATE TABLE IF NOT EXISTS skill_proofs (
    skill_name   TEXT NOT NULL,
    version      TEXT NOT NULL,
    version_hash TEXT NOT NULL DEFAULT '',
    passed       INTEGER NOT NULL DEFAULT 0,
    test_output  TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    PRIMARY KEY (skill_name, version)
);
CREATE TABLE IF NOT EXISTS skill_flags (
    name       TEXT PRIMARY KEY,
    reason     TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class ProofResult:
    """Outcome of proving one skill draft."""
    skill_name: str
    version: str
    passed: bool
    test_code: str = ""
    test_output: str = ""
    reason: str = ""
    version_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name, "version": self.version,
            "passed": self.passed, "test_output": self.test_output[:2000],
            "reason": self.reason, "version_hash": self.version_hash,
        }


def ensure_tables(db: Any) -> bool:
    """Create the proof/flag tables.  Never raises."""
    try:
        db.executescript(SKILL_PROVING_DDL)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("skill_proving tables failed: %s", exc)
        return False


# ── test generation ──────────────────────────────────────────────────────

_PROOF_TEST_PROMPT = """You are writing a PROOF TEST for an AI agent skill.
The skill must prove itself: write a self-contained pytest file that the
skill's workflow actually works.  Offline only — no network, no API keys.

SKILL NAME: {name}
DESCRIPTION: {description}
TOOLS (in order): {tools}
WORKFLOW:
{workflow}

Write a pytest file (plain `def test_*` functions, stdlib + pytest only)
that proves the workflow.  Rules:
1. Embed the skill as a SKILL dict at the top (name, tools, workflow steps).
2. Mock every tool call — record calls on a FakeTools object, return canned data.
3. test_workflow_steps_grounded: every numbered workflow step must name at
   least one of the declared TOOLS (case-insensitive substring).
4. test_dry_run: walk the workflow steps in order against FakeTools; assert
   each step's tool exists and was called; assert data flows (step N's output
   feeds step N+1).
5. test_manifest_shape: name is snake_case, description non-empty, tools list
   non-empty.
6. No imports beyond stdlib and pytest.  No fixtures, no network, no sleeps.

Reply with ONLY the python code, no markdown fences.
"""


def generate_test(draft: Any, llm_fn: Any = None) -> Optional[str]:
    """Auto-generate a proof test for a skill draft.

    Uses the LLM when available; falls back to a deterministic structural
    test so proving works offline.  Returns None only when the draft is
    unusable.  Never raises.
    """
    try:
        name = getattr(draft, "name", "") or "unnamed_skill"
        description = getattr(draft, "description", "") or ""
        tools = list(getattr(draft, "tools", []) or [])
        workflow = getattr(draft, "workflow", "") or ""
        if not name or not workflow.strip():
            return None
        code = None
        if llm_fn is not None:
            try:
                prompt = _PROOF_TEST_PROMPT.format(
                    name=name, description=description,
                    tools=", ".join(tools) or "(none declared)",
                    workflow=workflow[:3000])
                text = llm_fn(prompt)
                code = _strip_fences(text)
            except Exception as exc:  # noqa: BLE001
                _log.warning("proof test LLM generation failed: %s", exc)
                code = None
        if not code or "def test_" not in code:
            code = _structural_test(name, description, tools, workflow)
        return code[:MAX_TEST_CHARS]
    except Exception as exc:  # noqa: BLE001
        _log.warning("generate_test failed: %s", exc)
        return None


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop opening fence (```python or ```) and closing fence
        start = 1 if lines and lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines and lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return text.strip()


def _structural_test(name: str, description: str, tools: list[str],
                     workflow: str) -> str:
    """Deterministic fallback proof test — no LLM needed.

    Proves: manifest shape, workflow steps grounded in declared tools, and a
    dry-run walk of the workflow against mocked tools.
    """
    steps = [ln.strip() for ln in workflow.splitlines() if ln.strip()][:12]
    payload = {
        "name": name, "description": description,
        "tools": tools, "steps": steps,
    }
    blob = json.dumps(payload, ensure_ascii=False)
    return f'''"""Auto-generated structural proof test for skill {name}."""
import json
import re

SKILL = json.loads({blob!r})


class FakeTools:
    def __init__(self):
        self.calls = []
        self._known = set(SKILL["tools"])

    def call(self, tool, **kwargs):
        assert tool in self._known, f"unknown tool: {{tool}}"
        self.calls.append((tool, kwargs))
        return {{"ok": True, "tool": tool, "echo": kwargs}}


def _step_tools(step):
    low = step.lower()
    return [t for t in SKILL["tools"] if t.lower() in low]


def test_manifest_shape():
    assert re.fullmatch(r"[a-z0-9_]+", SKILL["name"]), "name must be snake_case"
    assert SKILL["description"].strip(), "description must be non-empty"
    assert SKILL["tools"], "tools must be non-empty"


def test_workflow_steps_grounded():
    assert SKILL["steps"], "workflow must have steps"
    for step in SKILL["steps"]:
        found = _step_tools(step)
        assert found, f"step not grounded in any declared tool: {{step!r}}"


def test_dry_run():
    tools = FakeTools()
    prev_out = None
    for step in SKILL["steps"]:
        step_tools = _step_tools(step)
        assert step_tools, f"no tool for step: {{step!r}}"
        out = tools.call(step_tools[0], previous=prev_out, step=step[:60])
        assert out["ok"], f"tool call failed for step: {{step!r}}"
        prev_out = out  # data flows step -> step
    assert len(tools.calls) == len(SKILL["steps"])
'''


# ── sandboxed test execution ─────────────────────────────────────────────

def run_proof_test(test_code: str,
                   timeout_s: int = PROOF_TIMEOUT_S) -> tuple[bool, str]:
    """Run a proof test in a sandboxed subprocess.  Never raises.

    Returns (passed, output).  Offline by construction: temp cwd, no
    network-dependent imports allowed to fail the run silently — a
    collection error counts as a failed proof.
    """
    try:
        if not (test_code or "").strip() or "def test_" not in test_code:
            return False, "no test functions found in generated test"
        with tempfile.TemporaryDirectory(prefix="skill_proof_") as tmp:
            path = Path(tmp) / "test_skill_proof.py"
            path.write_text(test_code, encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get(
                "PYTHONPATH", "")
            # belt-and-braces: no proxy leakage into the sandbox
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                env.pop(k, None)
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(path), "-q",
                 "-p", "no:cacheprovider", "--no-header"],
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=timeout_s)
            out = (proc.stdout or "") + (proc.stderr or "")
            passed = proc.returncode == 0
            return passed, out[-4000:]
    except subprocess.TimeoutExpired:
        return False, f"proof test timed out after {timeout_s}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"proof runner failed: {exc}"


# ── proving + lifecycle ──────────────────────────────────────────────────

def prove_skill(draft: Any, db: Any, llm_fn: Any = None,
                test_code: str | None = None) -> ProofResult:
    """Generate (or take) a proof test, run it, record the outcome.

    Returns a ProofResult; the row is recorded in ``skill_proofs``.
    Never raises.
    """
    name = str(getattr(draft, "name", "") or "unnamed_skill")
    version = ""
    try:
        manifest = draft.to_manifest() if hasattr(draft, "to_manifest") else {}
        version = str(manifest.get("version", "")) if isinstance(
            manifest, dict) else ""
    except Exception:  # noqa: BLE001
        version = ""
    try:
        ensure_tables(db)
        code = test_code or generate_test(draft, llm_fn=llm_fn)
        if not code:
            result = ProofResult(name, version, False,
                                 reason="could not generate a proof test")
        else:
            passed, output = run_proof_test(code)
            result = ProofResult(
                name, version, passed, test_code=code, test_output=output,
                reason="" if passed else "proof test failed")
        record_proof(db, result)
        return result
    except Exception as exc:  # noqa: BLE001
        _log.warning("prove_skill failed: %s", exc)
        return ProofResult(name, version, False, reason=f"prover error: {exc}")


def record_proof(db: Any, proof: ProofResult) -> bool:
    """Persist a proof outcome.  Never raises."""
    try:
        ensure_tables(db)
        db.execute(
            "INSERT OR REPLACE INTO skill_proofs "
            "(skill_name, version, version_hash, passed, test_output, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (proof.skill_name, proof.version, proof.version_hash,
             1 if proof.passed else 0, proof.test_output[:4000],
             time.time()))
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("record_proof failed: %s", exc)
        return False


def has_proof(skill_name: str, db: Any, version: str | None = None,
              version_hash: str | None = None) -> bool:
    """True when a passing proof exists for this skill (optionally pinned
    to one version / version hash).  Never raises."""
    try:
        ensure_tables(db)
        q = ("SELECT 1 FROM skill_proofs WHERE skill_name=? AND passed=1")
        args: list[Any] = [skill_name]
        if version:
            q += " AND version=?"
            args.append(version)
        if version_hash:
            q += " AND version_hash=?"
            args.append(version_hash)
        q += " LIMIT 1"
        return db.query_one(q, tuple(args)) is not None
    except Exception:  # noqa: BLE001
        return False


def flag_skill(name: str, reason: str, db: Any,
               registry: Any = None) -> bool:
    """Flag a skill as unproven/failed: record the flag and disable it so
    the runner refuses to execute it.  Never raises."""
    try:
        ensure_tables(db)
        db.execute(
            "INSERT OR REPLACE INTO skill_flags (name, reason, created_at) "
            "VALUES (?,?,?)", (name, (reason or "")[:500], time.time()))
        if registry is not None:
            try:
                registry.disable(name)
            except Exception:  # noqa: BLE001
                pass
        _log.info("skill flagged: %s (%s)", name, reason)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("flag_skill failed: %s", exc)
        return False


def unflag_skill(name: str, db: Any, registry: Any = None) -> bool:
    """Clear a flag (human-review path).  Re-enables the skill but does NOT
    activate it — activation still needs a passing proof.  Never raises."""
    try:
        ensure_tables(db)
        db.execute("DELETE FROM skill_flags WHERE name=?", (name,))
        if registry is not None:
            try:
                registry.enable(name)
            except Exception:  # noqa: BLE001
                pass
        return True
    except Exception:  # noqa: BLE001
        return False


def get_flag(name: str, db: Any) -> dict[str, Any] | None:
    """Return the flag record for a skill, or None.  Never raises."""
    try:
        ensure_tables(db)
        row = db.query_one("SELECT * FROM skill_flags WHERE name=?", (name,))
        return dict(row) if row else None
    except Exception:  # noqa: BLE001
        return None


def refuse_untested(name: str, db: Any) -> bool:
    """True when a skill must be refused: no passing proof test on record.
    This is the enforcement behind 'untested skills are flagged and
    refused'.  Never raises."""
    try:
        return not has_proof(name, db)
    except Exception:  # noqa: BLE001
        return True


def promote_on_proof(draft: Any, proof: ProofResult, registry: Any,
                     db: Any) -> str:
    """Apply the lifecycle transition after proving.

    pass  -> pin the version (activate) and record the proof hash on the
             canary version chain for auditability.
    fail  -> flag + disable (refused by the runner).
    Returns "active" or "flagged".  Never raises.
    """
    name = proof.skill_name
    try:
        if proof.passed:
            version = proof.version or _installed_version(registry, name)
            try:
                registry.pin(name, version)
            except Exception:  # noqa: BLE001
                # proof.version may be stale (draft re-manifested); retry
                # against the actually installed version before giving up.
                fallback = _installed_version(registry, name)
                if fallback and fallback != version:
                    try:
                        registry.pin(name, fallback)
                        version = fallback
                    except Exception:  # noqa: BLE001
                        version = ""
                else:
                    version = ""
            if not version:
                _log.warning("promote pin failed for %s", name)
                flag_skill(name, "proof passed but pin failed", db, registry)
                return "flagged"
            # attach the proof to the canary version chain for auditability
            try:
                from .skill_canary import CanaryRollout
                ctx = _Ctx(db)
                rollout = CanaryRollout(ctx)
                body = getattr(draft, "workflow", "") or name
                vhash = rollout.record_version(
                    name, body, source="proof-pass")
                proof.version_hash = vhash
                record_proof(db, proof)
            except Exception:  # noqa: BLE001
                pass
            _log.info("skill proven and activated: %s %s", name,
                      proof.version)
            return "active"
        flag_skill(name, proof.reason or "proof test failed", db, registry)
        return "flagged"
    except Exception as exc:  # noqa: BLE001
        _log.warning("promote_on_proof failed: %s", exc)
        return "flagged"


def _installed_version(registry: Any, name: str) -> str:
    """Latest installed version for a skill, or ''.  Never raises."""
    try:
        versions = registry.versions(name) or []
        return versions[-1] if versions else ""
    except Exception:  # noqa: BLE001
        return ""


def _any_installed(registry: Any, name: str) -> Any | None:
    """Fetch any installed version row (unlike registry.get(name), which
    only resolves the active pin).  Never raises."""
    try:
        version = _installed_version(registry, name)
        if not version:
            return None
        return registry.get(name, version)
    except Exception:  # noqa: BLE001
        return None


class _Ctx:
    """Minimal context shim for CanaryRollout (needs .db)."""

    def __init__(self, db: Any) -> None:
        self.db = db


def skill_lifecycle(name: str, db: Any, registry: Any = None) -> str:
    """Current lifecycle state of a skill: unknown | draft | testing |
    active | flagged | inactive.  Never raises."""
    try:
        if get_flag(name, db):
            return FLAGGED
        if registry is None:
            return UNKNOWN
        installed = _any_installed(registry, name)
        if installed is None:
            return UNKNOWN
        if not installed.enabled:
            return FLAGGED if get_flag(name, db) else INACTIVE
        # enabled: active pin set?
        try:
            active = registry.get(name)
        except Exception:  # noqa: BLE001
            active = None
        if active is not None and active.active:
            return ACTIVE if has_proof(name, db) else INACTIVE
        return DRAFT
    except Exception:  # noqa: BLE001
        return UNKNOWN


# ── EloPhanto: write the missing tool, then prove it ─────────────────────

_CAPABILITY_PROMPT = """An AI agent is missing a capability and must write its
own tool for it (then prove it with a test before use).

MISSING CAPABILITY: {name}
DESCRIPTION: {description}

Write the capability as a SKILL DRAFT in exactly this format:
NAME: <short_snake_case_name>
DESCRIPTION: <one line: when to use this skill>
TOOLS: <tool1>, <tool2>   (existing tool names this skill composes)
WORKFLOW:
1. <step naming its tool>
2. <step naming its tool>
3. ...

Rules: 3-6 numbered steps, each step names at least one listed tool,
no task-specific details (no URLs, no names, no dates).
Reply in exactly that format, nothing else.
"""


def ensure_capability(name: str, description: str, db: Any,
                      registry: Any = None, llm_fn: Any = None,
                      context: Any = None) -> dict[str, Any]:
    """EloPhanto pattern: when a capability is missing, draft the tool +
    an auto-generated test, prove it, and activate only on a passing test.

    Returns {"status": ...} where status is one of:
      present          — an active, proven skill already covers it
      active           — drafted, proven, and activated
      flagged          — drafted but the proof failed (disabled + flagged)
      failed           — could not draft (no LLM / parse failure)
    Never raises.
    """
    try:
        ensure_tables(db)
        name = (name or "").strip().lower().replace(" ", "_")
        if not name:
            return {"status": "failed", "reason": "empty capability name"}
        if registry is not None:
            try:
                # the drafted skill is stored as distilled_<name>; accept
                # either spelling when checking for an existing capability
                for candidate in (name, f"distilled_{name}"):
                    installed = registry.get(candidate)
                    if (installed is not None and installed.enabled
                            and installed.active
                            and has_proof(candidate, db)):
                        return {"status": "present", "skill": candidate}
            except Exception:  # noqa: BLE001
                pass
        # draft the missing tool
        draft = _draft_capability(name, description, llm_fn, context)
        if draft is None:
            return {"status": "failed",
                    "reason": "could not draft the capability"}
        if registry is not None:
            try:
                registry.install(draft.to_manifest())
                registry.deactivate(draft.name)
            except Exception as exc:  # noqa: BLE001
                return {"status": "failed",
                        "reason": f"install failed: {exc}"}
        proof = prove_skill(draft, db, llm_fn=llm_fn)
        if registry is not None:
            outcome = promote_on_proof(draft, proof, registry, db)
        else:
            outcome = "active" if proof.passed else "flagged"
            if not proof.passed:
                flag_skill(draft.name, proof.reason, db, None)
        return {"status": outcome, "skill": draft.name,
                "proof_passed": proof.passed,
                "reason": proof.reason}
    except Exception as exc:  # noqa: BLE001
        _log.warning("ensure_capability failed: %s", exc)
        return {"status": "failed", "reason": str(exc)[:200]}


def _draft_capability(name: str, description: str, llm_fn: Any,
                      context: Any) -> Any | None:
    """Draft a skill for a missing capability.  Never raises."""
    try:
        from .skill_distillation import _parse_draft
        prompt = _CAPABILITY_PROMPT.format(
            name=name, description=(description or name)[:500])
        text = None
        if llm_fn is not None:
            text = llm_fn(prompt)
        else:
            router = getattr(context, "router", None) if context else None
            if router is None:
                _log.warning("capability draft skipped: no LLM available")
                return None
            resp = router.complete(prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
        draft = _parse_draft(text)
        if draft is None:
            return None
        # keep the requested name when the LLM renamed it
        if not draft.name.endswith(name):
            draft.name = f"distilled_{name}"
        return draft
    except Exception as exc:  # noqa: BLE001
        _log.warning("capability draft failed: %s", exc)
        return None


# ── canary promotion gate ────────────────────────────────────────────────

def canary_proof_check(db: Any) -> Callable[[str, str], bool]:
    """Build a ``proof_check(skill_name, version_hash)`` callable for
    ``CanaryRollout.evaluate(..., proof_check=...)``.

    Promotion of a canary version is refused unless a passing proof test
    is recorded for that version hash.  This is how generated skills go
    through skill_canary.py with an auto-generated test: the numbers can
    only promote what the tests have already proven.
    """
    def check(skill_name: str, version_hash: str) -> bool:
        try:
            return has_proof(skill_name, db, version_hash=version_hash)
        except Exception:  # noqa: BLE001
            return False

    return check
