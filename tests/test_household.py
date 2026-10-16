"""Tests for nomorals.household.household (shared layer over family)."""

import tempfile
import unittest

from nomorals.family.registry import family_registry
from nomorals.household.household import ROLE_RANK, Household


def _household():
    tmp = tempfile.TemporaryDirectory()
    registry = family_registry(tmp.name)
    # Bootstrap: first member must be an owner.
    registry.call("family_person_add", actor_id="owner1", actor_role="owner",
                  person_id="owner1", display_name="Owner", role="owner")
    registry.call("family_person_add", actor_id="owner1", actor_role="owner",
                  person_id="adult1", display_name="Adult", role="adult")
    registry.call("family_person_add", actor_id="owner1", actor_role="owner",
                  person_id="teen1", display_name="Teen", role="teen")
    household = Household(registry=registry)
    return tmp, household, registry


class TestHousehold(unittest.TestCase):
    def setUp(self):
        self._tmp, self.household, self.registry = _household()

    def tearDown(self):
        self.household.close()
        self._tmp.cleanup()

    def test_role_rank_hierarchy(self):
        self.assertGreater(ROLE_RANK["owner"], ROLE_RANK["adult"])
        self.assertGreater(ROLE_RANK["adult"], ROLE_RANK["teen"])
        self.assertGreater(ROLE_RANK["teen"], ROLE_RANK["child"])

    def test_shared_commitment_visible_to_members(self):
        entry = self.household.add_shared_commitment(
            "owner1", "Pay the rent", owner_person="adult1", due=1_800_000_000.0)
        self.assertEqual(entry["what"], "Pay the rent")
        self.assertEqual(entry["owner_person"], "adult1")
        for member in ("owner1", "adult1", "teen1"):
            visible = self.household.list_shared_commitments(member)
            self.assertEqual([c["id"] for c in visible], [entry["id"]])

    def test_shared_calendar_and_assets_visible_to_members(self):
        cal = self.household.add_shared_calendar(
            "adult1", "Dentist", 1_800_000_100.0, person_id="teen1")
        asset = self.household.add_shared_asset(
            "owner1", "Family car", owner_person="owner1")
        self.assertEqual(cal["person_id"], "teen1")
        self.assertEqual(asset["name"], "Family car")
        self.assertEqual(
            [c["id"] for c in self.household.list_shared_calendar("teen1")],
            [cal["id"]])
        self.assertEqual(
            [a["id"] for a in self.household.list_shared_assets("teen1")],
            [asset["id"]])

    def test_non_member_cannot_see_shared(self):
        self.household.add_shared_commitment("owner1", "Pay the rent")
        with self.assertRaises(PermissionError):
            self.household.list_shared_commitments("stranger")
        with self.assertRaises(PermissionError):
            self.household.add_shared_commitment("stranger", "Sneaky")

    def test_private_not_visible_to_other_members(self):
        # Adult stores one private + one shared item.
        self.registry.call(
            "family_item_put", actor_id="adult1", actor_role="adult",
            person_id="adult1", item_id="priv-1", kind="note",
            title="My private diary", body="private thoughts",
            visibility="private")
        self.registry.call(
            "family_item_put", actor_id="adult1", actor_role="adult",
            person_id="adult1", item_id="shared-1", kind="note",
            title="Shopping list", body="milk and bread",
            visibility="shared")
        own = self.household.get_visible("adult1", "adult1")
        self.assertEqual({i["id"] for i in own}, {"priv-1", "shared-1"})
        # Household owner sees the private item too (audit).
        owner_view = self.household.get_visible("owner1", "adult1")
        self.assertEqual({i["id"] for i in owner_view}, {"priv-1", "shared-1"})
        # Another member sees only the shared item — never the private one.
        teen_view = self.household.get_visible("teen1", "adult1")
        self.assertEqual({i["id"] for i in teen_view}, {"shared-1"})

    def test_audit_records_actions_and_is_owner_only(self):
        self.household.add_shared_commitment("adult1", "Book flights")
        self.household.add_shared_asset("teen1", "Board game")
        log = self.household.audit_log("owner1")
        actions = [e["action"] for e in log]
        self.assertIn("shared_commitment_add", actions)
        self.assertIn("shared_asset_add", actions)
        # Append-only: no update/delete helpers on the audit log.
        self.assertFalse(hasattr(self.household, "delete_audit_entry"))
        self.assertFalse(hasattr(self.household, "update_audit_entry"))
        with self.assertRaises(PermissionError):
            self.household.audit_log("adult1")
        with self.assertRaises(PermissionError):
            self.household.audit_log("teen1")

    def test_revoke_delegated_not_duplicated(self):
        # Revoke goes through the family registry's semantics.
        self.household.revoke_member(
            actor_id="owner1", actor_role="owner",
            person_id="teen1", reason="test")
        with self.assertRaises(PermissionError):
            self.household.list_shared_commitments("teen1")
        with self.assertRaises(PermissionError):
            self.household.get_visible("teen1", "adult1")
        # Last owner cannot be revoked — family rule, surfaced here.
        with self.assertRaises(ValueError):
            self.household.revoke_member(
                actor_id="owner1", actor_role="owner", person_id="owner1")

    def test_household_from_root_path(self):
        with tempfile.TemporaryDirectory() as root:
            other = Household(root=root)
            try:
                other.add_member(
                    actor_id="boss", actor_role="owner", person_id="boss",
                    display_name="Boss", role="owner")
                entry = other.add_shared_commitment("boss", "Water the plants")
                self.assertEqual(
                    [c["id"] for c in other.list_shared_commitments("boss")],
                    [entry["id"]])
            finally:
                other.close()


if __name__ == "__main__":
    unittest.main()
