"""Isolation enforcement for nomorals.community.

1. AST walk: no module under nomorals/community/ may import
   nomorals.memory, nomorals.accounts, any vault module, or any private
   connector (nomorals.connectors.*). This is the CI gate for the
   architectural isolation boundary.
2. Registry allowlist: every tool in community_registry() must carry a
   capability inside COMMUNITY_CAPABILITIES.
3. Policy check: community_policy() must deny a non-community capability
   and allow a community one.
"""

import ast
from pathlib import Path

from nomorals.community.policy import (
    COMMUNITY_CAPABILITIES,
    community_policy,
    is_community_capability,
)
from nomorals.community.registry import COMMUNITY_TOOL_NAMES, community_registry
from nomorals.core.policy import CapabilitySet

COMMUNITY_DIR = Path(__file__).resolve().parent.parent / "nomorals" / "community"

FORBIDDEN_PREFIXES = (
    "nomorals.memory",
    "nomorals.accounts",
    "nomorals.connectors",
)
FORBIDDEN_SUBSTRINGS = ("vault",)


def _imported_modules(tree: ast.AST) -> list[str]:
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
    return mods


def test_no_forbidden_imports():
    violations: list[str] = []
    for path in sorted(COMMUNITY_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for mod in _imported_modules(tree):
            if mod.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{path.name}: imports {mod}")
            if any(s in mod for s in FORBIDDEN_SUBSTRINGS):
                violations.append(f"{path.name}: imports {mod} (vault?)")
    assert not violations, "isolation breach:\n" + "\n".join(violations)


def test_registry_allowlist_capabilities():
    reg = community_registry()
    assert set(reg._tools) == set(COMMUNITY_TOOL_NAMES)
    for name, spec in reg._tools.items():
        assert is_community_capability(spec.capability), (
            f"tool {name} has capability {spec.capability!r} outside the allowlist"
        )
        assert spec.capability in COMMUNITY_CAPABILITIES


def test_registry_has_no_dangerous_tools():
    reg = community_registry()
    dangerous = {"memory", "vault", "account", "exec", "shell", "post", "dm",
                 "connector", "browser", "download"}
    for name in reg._tools:
        assert not any(d in name for d in dangerous), f"suspicious tool {name}"


def test_community_policy_allows_and_denies():
    policy = community_policy()
    ok = policy.check("community.miniapp", grant=CapabilitySet(COMMUNITY_CAPABILITIES))
    assert ok.allowed
    denied = policy.check("mem.read", grant=CapabilitySet(COMMUNITY_CAPABILITIES))
    assert not denied.allowed
    denied2 = policy.check("exec.shell", grant=CapabilitySet(COMMUNITY_CAPABILITIES))
    assert not denied2.allowed


def test_registry_tools_pass_community_policy():
    policy = community_policy()
    reg = community_registry()
    grant = CapabilitySet(COMMUNITY_CAPABILITIES)
    for name, spec in reg._tools.items():
        assert policy.check(spec.capability, grant=grant).allowed, name
