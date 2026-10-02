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
    ROLE_PRESETS,
    Capability,
    CapabilitySet,
    Policy,
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


if __name__ == "__main__":
    unittest.main()
