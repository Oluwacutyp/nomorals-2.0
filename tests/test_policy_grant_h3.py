"""Wave H3 audit tests for ``Policy.grant_for_role``.

The old implementation was
``CapabilitySet.role(role).intersect(self.default_grant.union(CapabilitySet.all()))``.
Because ``union`` with ``CapabilitySet.all()`` always injects the ``"*"``
wildcard and ``intersect`` returns the other side when either side is
unrestricted, ``default_grant`` was dead code: every role got its full preset
no matter what ceiling was configured. These tests pin the fixed semantics:

* no ceiling configured  -> role preset passes through unchanged (backward compat)
* non-empty ceiling      -> role grant is narrowed to ``preset ∩ ceiling``
* owner (wildcard preset) -> full power, never narrowed by a ceiling
* unknown role           -> KeyError
"""

import unittest

from nomorals.core.policy import (
    GRADIENT_ACT_SILENT,
    GRADIENT_ACT_WITH_APPROVAL,
    GRADIENT_OBSERVE,
    GRADIENT_PROPOSE,
    ROLE_PRESETS,
    Capability,
    CapabilitySet,
    PermissionGradient,
    Policy,
    is_explicit_instruction,
)


class TestGrantForRole(unittest.TestCase):
    # (a) backward compatibility: default Policy has no ceiling, so the role
    # preset must come back untouched.
    def test_no_ceiling_returns_full_role_preset(self) -> None:
        policy = Policy()
        grant = policy.grant_for_role("coding")
        self.assertEqual(grant.patterns, CapabilitySet.role("coding").patterns)
        self.assertTrue(grant.grants(Capability.EXEC_SHELL))
        self.assertFalse(grant.grants(Capability.SOCIAL_POST))

    def test_explicit_empty_ceiling_returns_full_role_preset(self) -> None:
        policy = Policy(default_grant=CapabilitySet.none())
        grant = policy.grant_for_role("research")
        self.assertEqual(grant.patterns, CapabilitySet.role("research").patterns)

    # (b) a configured ceiling must actually narrow the role grant.
    def test_ceiling_narrows_role_grant(self) -> None:
        ceiling = CapabilitySet.of(Capability.FS_READ, Capability.FS_WRITE)
        policy = Policy(default_grant=ceiling)
        grant = policy.grant_for_role("coding")
        # The ceiling now matters: coding's shell/code/model/db powers are gone.
        self.assertTrue(grant.grants(Capability.FS_READ))
        self.assertTrue(grant.grants(Capability.FS_WRITE))
        self.assertFalse(grant.grants(Capability.EXEC_SHELL))
        self.assertFalse(grant.grants(Capability.MODEL_CALL))
        self.assertFalse(grant.grants(Capability.DB_READ))
        self.assertEqual(grant.patterns, frozenset({"fs.read", "fs.write"}))

    def test_ceiling_applies_to_every_known_role(self) -> None:
        ceiling = CapabilitySet.of(Capability.FS_READ)
        policy = Policy(default_grant=ceiling)
        for role in ROLE_PRESETS:
            grant = policy.grant_for_role(role)
            if "*" in CapabilitySet.role(role).patterns:
                continue  # owner: pinned full power, checked separately
            self.assertFalse(
                grant.grants(Capability.EXEC_SHELL),
                f"role {role!r} escaped the ceiling",
            )

    def test_wildcard_ceiling_is_still_a_no_op(self) -> None:
        # A ceiling of "*" is vacuous: the preset must pass through unchanged.
        policy = Policy(default_grant=CapabilitySet.all())
        grant = policy.grant_for_role("coding")
        self.assertEqual(grant.patterns, CapabilitySet.role("coding").patterns)

    # (c) the owner role must keep full power even under a narrow ceiling.
    def test_owner_role_is_pinned_full_power(self) -> None:
        self.assertIn("owner", ROLE_PRESETS)
        self.assertIn("*", CapabilitySet.role("owner").patterns)
        policy = Policy(
            default_grant=CapabilitySet.of(Capability.FS_READ)  # tight ceiling
        )
        grant = policy.grant_for_role("owner")
        for cap in Capability.ALL:
            self.assertTrue(grant.grants(cap), f"owner lost {cap!r}")
        # Wildcard covers extension capabilities not in Capability.ALL too.
        self.assertTrue(grant.grants("future.new.capability"))

    def test_owner_full_power_without_ceiling(self) -> None:
        grant = Policy().grant_for_role("owner")
        self.assertTrue(grant.grants(Capability.SYS_SHUTDOWN))
        self.assertTrue(grant.grants(Capability.EXEC_SHELL))

    # (d) unknown roles still raise KeyError, ceiling or not.
    def test_unknown_role_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            Policy().grant_for_role("wizard")
        with self.assertRaises(KeyError):
            Policy(default_grant=CapabilitySet.of(Capability.FS_READ)).grant_for_role(
                "wizard"
            )


class TestPermissionGradient(unittest.TestCase):
    """Build-map extension #8: Dots-pattern permission gradient."""

    def setUp(self) -> None:
        self.gradient = PermissionGradient()
        self.owner = CapabilitySet.all()

    # -- level classification -------------------------------------------------
    def test_read_capabilities_are_observe(self) -> None:
        for cap in (
            Capability.FS_READ,
            Capability.MEM_READ,
            Capability.DB_READ,
            Capability.SOCIAL_READ,
            Capability.MODEL_CALL,
            Capability.NET_OUT,
        ):
            self.assertEqual(
                self.gradient.level_for(cap), GRADIENT_OBSERVE, cap
            )

    def test_write_capabilities_default_to_propose(self) -> None:
        self.assertEqual(self.gradient.level_for(Capability.FS_WRITE), GRADIENT_PROPOSE)
        self.assertEqual(self.gradient.level_for(Capability.SOCIAL_POST), GRADIENT_PROPOSE)
        self.assertEqual(self.gradient.level_for(Capability.EXEC_SHELL), GRADIENT_PROPOSE)

    def test_confirmable_capabilities_are_act_with_approval(self) -> None:
        self.assertEqual(
            self.gradient.level_for(Capability.FS_DELETE), GRADIENT_ACT_WITH_APPROVAL
        )
        self.assertEqual(
            self.gradient.level_for(Capability.SOCIAL_DM), GRADIENT_ACT_WITH_APPROVAL
        )

    def test_deny_rule_returns_none(self) -> None:
        self.gradient.policy.deny(Capability.NET_OUT)
        self.assertIsNone(self.gradient.level_for(Capability.NET_OUT))

    def test_allow_rule_is_act_silent(self) -> None:
        self.gradient.policy.allow(Capability.SOCIAL_POST)
        self.assertEqual(
            self.gradient.level_for(Capability.SOCIAL_POST), GRADIENT_ACT_SILENT
        )

    # -- autonomous evaluation ------------------------------------------------
    def test_autonomous_observe_allowed_silently(self) -> None:
        d = self.gradient.check(Capability.MEM_READ, actor="agent-1", grant=self.owner)
        self.assertTrue(d.allowed)
        self.assertEqual(d.gradient, GRADIENT_OBSERVE)

    def test_autonomous_propose_records_proposal(self) -> None:
        d = self.gradient.check(
            Capability.SOCIAL_POST, actor="agent-1", grant=self.owner, draft="Draft post"
        )
        self.assertFalse(d.allowed)
        self.assertTrue(d.needs_proposal)
        self.assertEqual(d.gradient, GRADIENT_PROPOSE)
        self.assertTrue(d.proposal_id)
        pending = self.gradient.pending_proposals()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["draft"], "Draft post")

    def test_proposal_approval_mints_token_and_executes(self) -> None:
        d = self.gradient.check(
            Capability.SOCIAL_POST, actor="agent-1", grant=self.owner
        )
        token = self.gradient.approve_proposal(d.proposal_id)
        self.assertTrue(token)
        d2 = self.gradient.policy.check(
            Capability.SOCIAL_POST, actor="agent-1", grant=self.owner, confirmation=token
        )
        self.assertTrue(d2.allowed)

    def test_rejected_proposal_cannot_be_approved(self) -> None:
        d = self.gradient.check(
            Capability.SOCIAL_POST, actor="agent-1", grant=self.owner
        )
        self.assertTrue(self.gradient.reject_proposal(d.proposal_id, note="nope"))
        self.assertIsNone(self.gradient.approve_proposal(d.proposal_id))

    def test_non_autonomous_falls_back_to_plain_check(self) -> None:
        d = self.gradient.check(
            Capability.FS_WRITE, actor="agent-1", grant=self.owner, autonomous=False
        )
        self.assertTrue(d.allowed)
        self.assertEqual(d.gradient, GRADIENT_PROPOSE)

    # -- explicit override ----------------------------------------------------
    def test_explicit_override_bypasses_confirmation_gate(self) -> None:
        # social.dm is confirmable: normally needs a token.
        plain = self.gradient.policy.check(
            Capability.SOCIAL_DM, actor="agent-1", grant=self.owner
        )
        self.assertFalse(plain.allowed)
        self.assertTrue(plain.needs_confirmation)
        d = self.gradient.check(
            Capability.SOCIAL_DM, actor="agent-1", grant=self.owner, explicit_override=True
        )
        self.assertTrue(d.allowed)
        self.assertTrue(d.explicit_override)
        self.assertEqual(d.gradient, GRADIENT_ACT_SILENT)

    def test_explicit_override_keeps_biometric_structural(self) -> None:
        d = self.gradient.check(
            Capability.FS_DELETE, actor="agent-1", grant=self.owner, explicit_override=True
        )
        self.assertFalse(d.allowed)
        self.assertTrue(d.needs_biometric)

    def test_explicit_override_still_honors_deny(self) -> None:
        self.gradient.policy.deny(Capability.SOCIAL_DM)
        d = self.gradient.check(
            Capability.SOCIAL_DM, actor="agent-1", grant=self.owner, explicit_override=True
        )
        self.assertFalse(d.allowed)

    # -- explicit-instruction detection ---------------------------------------
    def test_is_explicit_instruction_positives(self) -> None:
        for text in (
            "post it",
            "send it",
            "do it now",
            "just do it",
            "go ahead",
            "Post it at 6 for engagement",
            "book it",
            "apply now",
            "confirmed",
        ):
            self.assertTrue(is_explicit_instruction(text), text)

    def test_is_explicit_instruction_negatives(self) -> None:
        for text in (
            "",
            "should I post it?",
            "don't post it",
            "do not send it",
            "never delete it",
            "thinking about posting later",
        ):
            self.assertFalse(is_explicit_instruction(text), text)

    # -- audit ----------------------------------------------------------------
    def test_observe_records_observe_audit_kind(self) -> None:
        self.gradient.check(Capability.DB_READ, actor="agent-1", grant=self.owner)
        entries = self.gradient.policy.audit_log(kind="observe")
        self.assertTrue(entries)
        self.assertEqual(entries[-1]["capability"], Capability.DB_READ)

    def test_propose_decision_verifies_against_audit(self) -> None:
        d = self.gradient.check(
            Capability.SOCIAL_POST, actor="agent-1", grant=self.owner
        )
        self.assertTrue(self.gradient.policy.verify_decision(d))

    def test_to_dict_carries_gradient_fields(self) -> None:
        d = self.gradient.check(Capability.MEM_READ, actor="agent-1", grant=self.owner)
        payload = d.to_dict()
        self.assertEqual(payload["gradient"], GRADIENT_OBSERVE)
        self.assertFalse(payload["needs_proposal"])
        self.assertFalse(payload["explicit_override"])

    # -- robustness -----------------------------------------------------------
    def test_gradient_never_raises_on_garbage(self) -> None:
        d = self.gradient.check(None, actor="x")  # type: ignore[arg-type]
        self.assertFalse(d.allowed)
        self.assertFalse(is_explicit_instruction(None))  # type: ignore[arg-type]
        self.assertIsNone(self.gradient.approve_proposal("nope"))
        self.assertFalse(self.gradient.reject_proposal("nope"))


if __name__ == "__main__":
    unittest.main()
