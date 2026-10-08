"""Tests for biometric (fingerprint) approval — all offline, no device needed."""

import subprocess
import unittest
from unittest.mock import patch

from nomorals.core.policy import (
    AUDIT_BIOMETRIC,
    Capability,
    CapabilitySet,
    Policy,
    approve_with_biometric,
)
from nomorals.native import biometric as bio_mod
from nomorals.tools.registry import ToolRegistry


def _open_policy(**kw):
    return Policy(default_grant=CapabilitySet.all(), **kw)


class AvailabilityTests(unittest.TestCase):
    def _avail(self, profile, which):
        with patch("nomorals.core.profiles.get_profile_kind", return_value=profile), \
             patch("shutil.which", return_value=which):
            return bio_mod.biometric_available()

    def test_termux_with_binary(self):
        ok, reason = self._avail("termux", "/data/data/com.termux/files/usr/bin/termux-fingerprint")
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_termux_without_binary(self):
        ok, reason = self._avail("termux", None)
        self.assertFalse(ok)
        self.assertIn("termux-fingerprint", reason)

    def test_non_termux_profile(self):
        ok, reason = self._avail("laptop", "/usr/bin/termux-fingerprint")
        self.assertFalse(ok)
        self.assertIn("Termux", reason)

    def test_profile_detection_failure_fails_closed(self):
        with patch("nomorals.core.profiles.get_profile_kind", side_effect=RuntimeError("boom")):
            ok, reason = bio_mod.biometric_available()
        self.assertFalse(ok)
        self.assertTrue(reason)


class RequestBiometricTests(unittest.TestCase):
    def _run(self, **kw):
        run_kwargs = {"returncode": 0}
        run_kwargs.update(kw)
        fake = subprocess.CompletedProcess(args=["termux-fingerprint"], **run_kwargs)
        with patch("subprocess.run", return_value=fake) as m:
            result = bio_mod.request_biometric("test title")
        return result, m

    def test_exit_zero_approves(self):
        ok, m = self._run(returncode=0)
        self.assertTrue(ok)
        args, kwargs = m.call_args
        self.assertEqual(args[0], ["termux-fingerprint"])
        self.assertEqual(kwargs["timeout"], 60.0)
        self.assertTrue(kwargs["capture_output"])

    def test_nonzero_exit_denies(self):
        ok, _ = self._run(returncode=1)
        self.assertFalse(ok)

    def test_timeout_denies(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
            self.assertFalse(bio_mod.request_biometric("t", timeout_s=1))

    def test_missing_binary_denies(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("nope")):
            self.assertFalse(bio_mod.request_biometric("t"))

    def test_custom_timeout_passed_through(self):
        fake = subprocess.CompletedProcess(args=["x"], returncode=0)
        with patch("subprocess.run", return_value=fake) as m:
            bio_mod.request_biometric("t", timeout_s=5.0)
        self.assertEqual(m.call_args[1]["timeout"], 5.0)


class PolicyBiometricTests(unittest.TestCase):
    def test_biometric_capability_denial_sets_both_flags(self):
        p = _open_policy()
        d = p.check(Capability.FS_DELETE, actor="agent")
        self.assertFalse(d.allowed)
        self.assertTrue(d.needs_confirmation)
        self.assertTrue(d.needs_biometric)
        self.assertEqual(p.audit_log(limit=1)[0]["kind"], AUDIT_BIOMETRIC)

    def test_non_biometric_confirmable_sets_only_confirmation(self):
        p = _open_policy()
        d = p.check(Capability.EXEC_INSTALL, actor="agent")
        self.assertFalse(d.allowed)
        self.assertTrue(d.needs_confirmation)
        self.assertFalse(d.needs_biometric)
        self.assertEqual(p.audit_log(limit=1)[0]["kind"], "confirm")

    def test_biometric_rule_priority_over_confirm(self):
        p = _open_policy()
        p.confirm("net.*", priority=50)
        p.biometric("net.out", priority=60)
        d = p.check(Capability.NET_OUT, actor="agent")
        self.assertTrue(d.needs_biometric)
        # and the lower-priority confirm rule does not win for another cap
        d2 = p.check(Capability.NET_DOWNLOAD, actor="agent")
        self.assertTrue(d2.needs_confirmation)
        self.assertFalse(d2.needs_biometric)

    def test_biometric_token_round_trip(self):
        p = _open_policy()
        token = p.issue_confirmation(Capability.FS_DELETE)
        d = p.check(Capability.FS_DELETE, actor="agent", confirmation=token)
        self.assertTrue(d.allowed)
        self.assertFalse(d.needs_biometric)
        # single-use: the same token cannot be replayed
        d2 = p.check(Capability.FS_DELETE, actor="agent", confirmation=token)
        self.assertFalse(d2.allowed)
        self.assertTrue(d2.needs_biometric)

    def test_token_bound_to_capability(self):
        p = _open_policy()
        token = p.issue_confirmation(Capability.DB_ADMIN)
        d = p.check(Capability.FS_DELETE, actor="agent", confirmation=token)
        self.assertFalse(d.allowed)

    def test_verify_decision_biometric(self):
        p = _open_policy()
        d = p.check(Capability.SYS_SHUTDOWN, actor="agent")
        self.assertTrue(p.verify_decision(d))
        import copy
        # flipping only `allowed` keeps needs_confirmation=True: the decision
        # is honestly still "pending confirmation", so it verifies.
        tampered = copy.copy(d)
        tampered.allowed = True
        self.assertTrue(p.verify_decision(tampered))
        # a real tamper claims a clean grant: allowed=True AND the
        # confirmation flag cleared — that must NOT verify.
        tampered2 = copy.copy(d)
        tampered2.allowed = True
        tampered2.needs_confirmation = False
        self.assertFalse(p.verify_decision(tampered2))

    def test_requires_biometric_helper(self):
        p = _open_policy()
        self.assertTrue(p.requires_biometric(Capability.FS_DELETE))
        self.assertTrue(p.requires_biometric(Capability.DB_ADMIN))
        self.assertFalse(p.requires_biometric(Capability.EXEC_INSTALL))
        self.assertFalse(p.requires_biometric(Capability.FS_READ))
        p.biometric("net.*")
        self.assertTrue(p.requires_biometric(Capability.NET_OUT))
        p.deny("net.out")
        self.assertFalse(p.requires_biometric(Capability.NET_OUT))

    def test_decision_to_dict_includes_flag(self):
        p = _open_policy()
        d = p.check(Capability.FS_DELETE, actor="agent")
        self.assertTrue(d.to_dict()["needs_biometric"])


class ApproveWithBiometricTests(unittest.TestCase):
    def _stubs(self, available=True, approved=True):
        p1 = patch("nomorals.native.biometric.biometric_available",
                   return_value=(available, "" if available else "nope"))
        p2 = patch("nomorals.native.biometric.request_biometric",
                   return_value=approved)
        return p1, p2

    def test_full_round_trip_deny_approve_check(self):
        p = _open_policy()
        denied = p.check(Capability.FS_DELETE, actor="agent")
        self.assertTrue(denied.needs_biometric)
        p1, p2 = self._stubs(available=True, approved=True)
        with p1, p2:
            token = approve_with_biometric(p, Capability.FS_DELETE, title="delete it?")
        self.assertIsNotNone(token)
        allowed = p.check(Capability.FS_DELETE, actor="agent", confirmation=token)
        self.assertTrue(allowed.allowed)

    def test_unavailable_returns_none(self):
        p = _open_policy()
        p1, p2 = self._stubs(available=False)
        with p1, p2:
            self.assertIsNone(approve_with_biometric(p, Capability.FS_DELETE))

    def test_prompt_denied_returns_none(self):
        p = _open_policy()
        p1, p2 = self._stubs(available=True, approved=False)
        with p1, p2:
            self.assertIsNone(approve_with_biometric(p, Capability.FS_DELETE))

    def test_prompt_exception_returns_none(self):
        p = _open_policy()
        p1 = patch("nomorals.native.biometric.biometric_available",
                   return_value=(True, ""))
        with p1, patch("nomorals.native.biometric.request_biometric",
                       side_effect=RuntimeError("boom")):
            self.assertIsNone(approve_with_biometric(p, Capability.FS_DELETE))

    def test_missing_biometric_module_returns_none(self):
        p = _open_policy()
        import sys
        with patch.dict(sys.modules, {"nomorals.native.biometric": None}):
            # `from X import Y` with None in sys.modules raises ImportError;
            # the helper must swallow it and return None.
            self.assertIsNone(approve_with_biometric(p, Capability.FS_DELETE))


class RegistryRequestApprovalTests(unittest.TestCase):
    def _registry(self, policy=None):
        ctx = type("Ctx", (), {"policy": policy})()
        return ToolRegistry(context=ctx)

    def _stub_biometric(self, approved=True):
        p1 = patch("nomorals.native.biometric.biometric_available",
                   return_value=(True, ""))
        p2 = patch("nomorals.native.biometric.request_biometric",
                   return_value=approved)
        return p1, p2

    def test_unknown_tool_returns_none(self):
        reg = self._registry(_open_policy())
        self.assertIsNone(reg.request_approval("nope"))

    def test_biometric_spec_level_mints_token(self):
        reg = self._registry(_open_policy())

        @reg.register("wipe", capability=Capability.FS_DELETE, confirm="biometric")
        def wipe():
            return "wiped"

        p1, p2 = self._stub_biometric(approved=True)
        with p1, p2:
            token = reg.request_approval("wipe", actor="agent")
        self.assertIsNotNone(token)
        # the token actually authorises the capability
        policy = reg.context.policy
        d = policy.check(Capability.FS_DELETE, actor="agent", confirmation=token)
        self.assertTrue(d.allowed)

    def test_biometric_via_policy_set_without_spec_flag(self):
        # spec says nothing, but the capability is in BIOMETRIC → biometric path
        reg = self._registry(_open_policy())

        @reg.register("dropdb", capability=Capability.DB_ADMIN)
        def dropdb():
            return "dropped"

        p1, p2 = self._stub_biometric(approved=True)
        with p1, p2:
            token = reg.request_approval("dropdb", actor="agent")
        self.assertIsNotNone(token)

    def test_plain_confirm_returns_none(self):
        reg = self._registry(_open_policy())

        @reg.register("mild", capability=Capability.FS_WRITE, confirm=True)
        def mild():
            return "ok"

        self.assertIsNone(reg.request_approval("mild", actor="agent"))

    def test_no_policy_returns_none(self):
        reg = self._registry(policy=None)

        @reg.register("wipe2", capability=Capability.FS_DELETE, confirm="biometric")
        def wipe2():
            return "wiped"

        self.assertIsNone(reg.request_approval("wipe2"))

    def test_biometric_unavailable_returns_none(self):
        reg = self._registry(_open_policy())

        @reg.register("wipe3", capability=Capability.FS_DELETE, confirm="biometric")
        def wipe3():
            return "wiped"

        p1 = patch("nomorals.native.biometric.biometric_available",
                   return_value=(False, "nope"))
        with p1:
            self.assertIsNone(reg.request_approval("wipe3"))

    def test_never_raises(self):
        reg = self._registry(_open_policy())
        # broken context: policy raises on requires_biometric
        class BadPolicy:
            def requires_biometric(self, cap):
                raise RuntimeError("boom")
        reg.context.policy = BadPolicy()

        @reg.register("x", capability=Capability.FS_DELETE, confirm="biometric")
        def x():
            return 1

        # spec-level flag still goes through approve path; stub it unavailable
        p1 = patch("nomorals.native.biometric.biometric_available",
                   return_value=(False, "nope"))
        with p1:
            self.assertIsNone(reg.request_approval("x"))


class ConfirmLevelsTests(unittest.TestCase):
    def test_spec_accepts_string_level(self):
        reg = ToolRegistry()
        reg.register("a", capability=Capability.FS_DELETE, confirm="biometric")(lambda: 1)
        reg.register("b", capability=Capability.FS_WRITE, confirm=True)(lambda: 1)
        reg.register("c", capability=Capability.FS_READ)(lambda: 1)
        self.assertEqual(reg._tools["a"].confirm, "biometric")
        self.assertTrue(reg._tools["b"].confirm)
        self.assertFalse(reg._tools["c"].confirm)


if __name__ == "__main__":
    unittest.main()
