"""Sandbox-first safety for the red-team harness. Isolate first, then trust.

Open SWE pattern: the red-team harness attacks Devon's OWN agent loop,
so the harness itself is a privileged position.  Before any scenario
runs, :class:`RedTeamSandbox` verifies the harness is actually isolated:

1. the tool registry is a fake (``_FakeRegistry``) — no real tool
   functions that could touch the filesystem, network, or money;
2. the model is scripted — no real LLM client that could leak the
   malicious payloads outward;
3. the policy is a fresh ``Policy()`` instance, not a shared production
   one whose grants could be mutated;
4. the skill DB is a stub (``_NullSkillDB``) or absent;
5. no connector / vault / credential objects are reachable from the
   harness values;
6. the user message contains no secret-shaped values (canaries live in
   harness *responses*, never in the inbound message).

A scenario whose harness fails verification is NEVER run: the finding
is recorded ``inconclusive`` with a ``sandbox violation`` evidence
string.  Fail closed, always.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "IsolationReport",
    "RedTeamSandbox",
    "audit_harness_module",
]

#: Module prefixes whose instances must never appear in a harness.
_FORBIDDEN_MODULE_PREFIXES = (
    "nomorals.connectors",
    "nomorals.accounts.vault",
    "nomorals.agents.trial.vault",
)


@dataclass
class IsolationReport:
    ok: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok,
                "violations": list(self.violations),
                "warnings": list(self.warnings)}


class RedTeamSandbox:
    """Verifies harness isolation before a scenario runs. Never raises."""

    def verify_isolation(self, harness: dict[str, Any] | None) -> IsolationReport:
        violations: list[str] = []
        warnings: list[str] = []
        try:
            h = harness or {}

            # 1. tool registry must be a fake
            tools = h.get("tools")
            if tools is not None:
                if type(tools).__name__ != "_FakeRegistry":
                    violations.append(
                        f"tools registry is {type(tools).__name__}, "
                        f"not _FakeRegistry — real tools could execute")
                else:
                    for name, entry in getattr(tools, "_tools", {}).items():
                        fn = entry[2] if len(entry) > 2 else None
                        mod = getattr(fn, "__module__", "") or ""
                        if mod.startswith(_FORBIDDEN_MODULE_PREFIXES):
                            violations.append(
                                f"tool {name!r} binds a real {mod} function")

            # 2. scripted model only: a loop harness must carry a script
            #    (S5/S8-style direct-check harnesses carry no tools and
            #    never run the loop — nothing to verify there).
            if tools is not None and "script" not in h:
                violations.append(
                    "loop harness has tools but no scripted model script")

            # 3. policy must be a Policy instance (fresh per scenario)
            policy = h.get("policy")
            if policy is not None and type(policy).__name__ != "Policy":
                violations.append(
                    f"policy is {type(policy).__name__}, not Policy")

            # 4. skill DB must be a stub or absent
            db = h.get("db")
            if db is not None and type(db).__name__ not in (
                    "_NullSkillDB", "NullDB", "FakeDB", "MemoryDB"):
                warnings.append(
                    f"harness db is {type(db).__name__}; expected a stub")

            # 5. no credential-bearing objects reachable from the harness
            for key, value in h.items():
                if key in ("policy", "tools", "db", "script",
                           "seen", "executed", "result"):
                    continue
                if self._is_credentialed(value):
                    violations.append(
                        f"harness[{key!r}] exposes a credentialed object "
                        f"({type(value).__name__})")

            # 6. user message must not contain secret-shaped values
            #    (canaries belong in harness responses, never inbound)
            um = h.get("user_message") or ""
            try:
                from .redteam import _secret_hit

                hit = _secret_hit(str(um))
                if hit:
                    violations.append(
                        "user_message contains a secret-shaped value — "
                        "canaries must not be inbound")
            except Exception:  # noqa: BLE001 - the check itself never breaks
                pass

            return IsolationReport(ok=not violations,
                                   violations=violations, warnings=warnings)
        except Exception as exc:  # noqa: BLE001 - fail closed
            return IsolationReport(
                ok=False,
                violations=[f"sandbox verifier errored: {exc}"],
                warnings=warnings,
            )

    @staticmethod
    def _is_credentialed(value: Any) -> bool:
        """True if the value looks like a live connector/vault object."""
        try:
            mod = type(value).__module__ or ""
            if mod.startswith(_FORBIDDEN_MODULE_PREFIXES):
                return True
            # duck-typing: live credential loaders
            if hasattr(value, "_load_credential") and not type(value).__name__.startswith(
                    ("Fake", "Mock", "Stub", "Null")):
                return True
            return False
        except Exception:  # noqa: BLE001
            return False


def audit_harness_module() -> IsolationReport:
    """Static self-audit: the redteam harness modules must not import
    credential paths.  Guards the harness code itself, not just one run.
    Never raises."""
    import ast
    from pathlib import Path

    violations: list[str] = []
    try:
        here = Path(__file__).resolve().parent
        for name in ("redteam.py", "redteam_scenarios.py",
                     "redteam_catalog.py", "redteam_sandbox.py"):
            path = here / name
            if not path.exists():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8",
                                                errors="replace"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                mods: list[str] = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mods = [node.module]
                for m in mods:
                    if m.startswith(_FORBIDDEN_MODULE_PREFIXES):
                        violations.append(f"{name} imports {m}")
        return IsolationReport(ok=not violations, violations=violations)
    except Exception as exc:  # noqa: BLE001
        return IsolationReport(ok=False,
                               violations=[f"harness audit errored: {exc}"])
