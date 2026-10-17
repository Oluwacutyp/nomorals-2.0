"""Argument-bound confirmations + oversight-tamper gate (arxiv 2507.19672v1).

Distilled spec (proposal #1, Alignment & Safety):
Devon's policy used to check ONLY the capability string. A confirmation
token minted for ``fs.delete`` of one file authorized ANY ``fs.delete``,
and an explicit-override phrase ("do it") skipped the confirmation gate
entirely — including for writes that target the oversight machinery
itself (``core/policy.py``, ``partner/social_gate.py``, audit logging,
the vault). Paper grounding: §2.3 Hierarchical Policy Stacking (the
safety check must see the actual action, not just its category) and
§3.1.4 / §12 corrigibility (the agent must not be able to undermine
its own oversight).

Behavior under test:
1. Confirmation tokens can bind to the exact argument digest; a token
   bound to args A does not authorize the same capability with args B,
   and is not consumed by the mismatched attempt.
2. An oversight-targeting write/delete/exec requires a confirmation
   token bound to its exact arguments — an explicit-override bypass and
   an unbound (generic) token are both insufficient.
3. Non-oversight behavior is unchanged: unbound tokens still authorize
   ordinary confirmable actions, and reads of oversight files stay
   allowed (transparency).
4. The registry passes the argument digest + oversight flag into the
   policy on every call, so the gate holds at the real choke point.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.core.oversight import args_digest, is_oversight_target
from nomorals.core.policy import Capability, CapabilitySet, Policy


class ArgsDigestTests(unittest.TestCase):
    def test_digest_is_stable_and_argument_sensitive(self) -> None:
        a = args_digest({"path": "x.txt", "content": "1"})
        self.assertEqual(a, args_digest({"content": "1", "path": "x.txt"}))
        self.assertNotEqual(a, args_digest({"path": "x.txt", "content": "2"}))
        self.assertNotEqual(a, args_digest({"path": "y.txt", "content": "1"}))


class OversightClassifierTests(unittest.TestCase):
    def test_write_to_policy_file_is_oversight(self) -> None:
        self.assertTrue(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": "nomorals/core/policy.py", "content": "x"}))
        self.assertTrue(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": "core/policy.py", "content": "x"}))
        self.assertTrue(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": "/home/o/nomorals-2.0/nomorals/partner/social_gate.py",
             "content": "x"}))

    def test_write_to_ordinary_file_is_not_oversight(self) -> None:
        self.assertFalse(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": "notes/a.txt", "content": "x"}))
        # A bare basename is not enough: other projects have a policy.py.
        self.assertFalse(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": "policy.py", "content": "x"}))

    def test_reads_are_never_oversight(self) -> None:
        self.assertFalse(is_oversight_target(
            "fs_read", Capability.FS_READ, {"path": "nomorals/core/policy.py"}))
        self.assertFalse(is_oversight_target(
            "fs_list", Capability.FS_READ, {"path": "nomorals/core"}))

    def test_delete_and_copy_destination_are_oversight(self) -> None:
        self.assertTrue(is_oversight_target(
            "fs_delete", Capability.FS_DELETE, {"path": "nomorals/core/policy.py"}))
        self.assertTrue(is_oversight_target(
            "fs_copy", Capability.FS_WRITE,
            {"source": "notes/a.txt",
             "destination": "nomorals/core/policy.py"}))
        # Copying an oversight file OUT is read-equivalent, not tampering.
        self.assertFalse(is_oversight_target(
            "fs_copy", Capability.FS_WRITE,
            {"source": "nomorals/core/policy.py",
             "destination": "notes/copy.py"}))

    def test_secret_material_is_oversight(self) -> None:
        self.assertTrue(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": ".ssh/id_rsa", "content": "x"}))
        self.assertTrue(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": "somewhere/.devon-secrets", "content": "x"}))
        self.assertTrue(is_oversight_target(
            "fs_write", Capability.FS_WRITE,
            {"path": "certs/server.pem", "content": "x"}))

    def test_shell_tripwire_needs_a_write_indicator(self) -> None:
        self.assertTrue(is_oversight_target(
            "shell_run", Capability.EXEC_SHELL,
            {"command": "echo hacked > nomorals/core/policy.py"}))
        self.assertFalse(is_oversight_target(
            "shell_run", Capability.EXEC_SHELL,
            {"command": "grep -rn policy nomorals/core/policy.py"}))
        self.assertFalse(is_oversight_target(
            "shell_run", Capability.EXEC_SHELL, {"command": "echo hello"}))


class ArgumentBoundTokenTests(unittest.TestCase):
    def _policy(self) -> Policy:
        return Policy(default_grant=CapabilitySet.of(
            Capability.FS_DELETE, Capability.FS_WRITE))

    def test_bound_token_authorizes_matching_args(self) -> None:
        policy = self._policy()
        digest = args_digest({"path": "d.txt"})
        token = policy.issue_confirmation(
            Capability.FS_DELETE, args_digest=digest)
        decision = policy.check(
            Capability.FS_DELETE, actor="a", confirmation=token,
            context={"args_digest": digest})
        self.assertTrue(decision.allowed)

    def test_bound_token_rejects_redirected_args_and_survives(self) -> None:
        policy = self._policy()
        digest_a = args_digest({"path": "d.txt"})
        digest_b = args_digest({"path": "nomorals/core/policy.py"})
        token = policy.issue_confirmation(
            Capability.FS_DELETE, args_digest=digest_a)
        denied = policy.check(
            Capability.FS_DELETE, actor="a", confirmation=token,
            context={"args_digest": digest_b})
        self.assertFalse(denied.allowed)
        # The mismatched attempt must not burn the token: the owner's
        # original approval is still good for the args they approved.
        allowed = policy.check(
            Capability.FS_DELETE, actor="a", confirmation=token,
            context={"args_digest": digest_a})
        self.assertTrue(allowed.allowed)

    def test_unbound_token_still_authorizes_ordinary_actions(self) -> None:
        policy = self._policy()
        token = policy.issue_confirmation(Capability.FS_DELETE)
        decision = policy.check(
            Capability.FS_DELETE, actor="a", confirmation=token,
            context={"args_digest": args_digest({"path": "d.txt"})})
        self.assertTrue(decision.allowed)


class OversightGateTests(unittest.TestCase):
    def _policy(self) -> Policy:
        return Policy(default_grant=CapabilitySet.of(
            Capability.FS_DELETE, Capability.FS_WRITE))

    def _oversight_ctx(self, **kwargs) -> dict:
        return {
            "args_digest": args_digest(kwargs),
            "oversight": True,
            "tool": "fs_write",
        }

    def test_explicit_override_does_not_bypass_oversight(self) -> None:
        policy = self._policy()
        decision = policy.check(
            Capability.FS_WRITE, actor="a", explicit_override=True,
            context=self._oversight_ctx(path="nomorals/core/policy.py",
                                        content="x"))
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.needs_confirmation)

    def test_unbound_token_does_not_authorize_oversight(self) -> None:
        policy = self._policy()
        token = policy.issue_confirmation(Capability.FS_WRITE)
        decision = policy.check(
            Capability.FS_WRITE, actor="a", confirmation=token,
            context=self._oversight_ctx(path="nomorals/core/policy.py",
                                        content="x"))
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.needs_confirmation)

    def test_bound_token_authorizes_exact_oversight_action(self) -> None:
        policy = self._policy()
        kwargs = {"path": "nomorals/core/policy.py", "content": "x"}
        token = policy.issue_confirmation(
            Capability.FS_WRITE, args_digest=args_digest(kwargs))
        decision = policy.check(
            Capability.FS_WRITE, actor="a", confirmation=token,
            context=self._oversight_ctx(**kwargs))
        self.assertTrue(decision.allowed)

    def test_bound_token_does_not_authorize_redirected_oversight(self) -> None:
        policy = self._policy()
        approved = {"path": "nomorals/core/policy.py", "content": "x"}
        token = policy.issue_confirmation(
            Capability.FS_WRITE, args_digest=args_digest(approved))
        redirected = {"path": "nomorals/partner/social_gate.py",
                      "content": "x"}
        decision = policy.check(
            Capability.FS_WRITE, actor="a", confirmation=token,
            context=self._oversight_ctx(**redirected))
        self.assertFalse(decision.allowed)

    def test_explicit_override_still_bypasses_ordinary_actions(self) -> None:
        policy = self._policy()
        decision = policy.check(
            Capability.FS_WRITE, actor="a", explicit_override=True,
            context={"args_digest": args_digest(
                {"path": "notes/a.txt", "content": "x"})})
        self.assertTrue(decision.allowed)


class RegistryOversightTests(unittest.TestCase):
    """The gate holds at ToolRegistry.call, the real choke point."""

    def setUp(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.core.config import Settings

        self.home = tempfile.mkdtemp(prefix="nm-argbound-")
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.tools = self.context.tools.register_builtins()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        shutil.rmtree(self.home, ignore_errors=True)

    def test_registry_blocks_oversight_write_without_bound_token(self) -> None:
        denied = self.tools.call(
            "fs_write", actor="t", capabilities=CapabilitySet.all(),
            path="nomorals/core/policy.py", content="x")
        self.assertFalse(denied.ok)
        token = self.context.policy.issue_confirmation(Capability.FS_WRITE)
        still_denied = self.tools.call(
            "fs_write", actor="t", capabilities=CapabilitySet.all(),
            confirmation=token,
            path="nomorals/core/policy.py", content="x")
        self.assertFalse(still_denied.ok)

    def test_registry_allows_oversight_write_with_bound_token(self) -> None:
        kwargs = {"path": "nomorals/core/policy.py", "content": "x"}
        token = self.context.policy.issue_confirmation(
            Capability.FS_WRITE, args_digest=args_digest(kwargs))
        allowed = self.tools.call(
            "fs_write", actor="t", capabilities=CapabilitySet.all(),
            confirmation=token, **kwargs)
        self.assertTrue(allowed.ok, allowed.error)

    def test_registry_read_of_oversight_file_stays_allowed(self) -> None:
        outcome = self.tools.call(
            "fs_read", actor="t", capabilities=CapabilitySet.all(),
            path="nomorals/core/policy.py")
        # Either the file exists in the test workspace (allowed) or it
        # does not (NotFound) — but never a policy denial.
        if not outcome.ok:
            self.assertNotIn("confirmation", str(outcome.error).lower())

    def test_registry_ordinary_delete_with_unbound_token_unchanged(self) -> None:
        self.tools.call("fs_write", actor="t",
                        capabilities=CapabilitySet.all(),
                        path="d.txt", content="x")
        token = self.context.policy.issue_confirmation(Capability.FS_DELETE)
        allowed = self.tools.call(
            "fs_delete", actor="t", capabilities=CapabilitySet.all(),
            confirmation=token, path="d.txt")
        self.assertTrue(allowed.ok, allowed.error)


if __name__ == "__main__":
    unittest.main()
