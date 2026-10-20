"""Pass #48 — family surface: roster-derived authority, reserved ids,
the registry delete tool, and write durability.

Every test here pins a defect reproduced failing-first on the pristine
tree (see the pass report):

* A — registry/members trusted the CALLER-CLAIMED ``actor_role`` and
  never consulted the roster record for ``actor_id``: any active
  member (and for revoke/set_role/audit, any caller at all) could
  claim "owner" and manage members, read the audit log, read other
  members' private items, share them, or write into their stores.
* B — a person_id colliding with a household file name
  (``members.json`` / ``audit.jsonl`` / ``household_shared.db``) was
  saved into the roster and THEN failed, half-added with no audit.
* C — ``delete_own`` existed in roles + store but no registry tool
  could reach it.
* D/E — whole-file JSON was written non-atomically with no locking:
  concurrent writers lost items and readers could hit torn files.

Authority contract pinned here: the roster record for ``actor_id`` is
the ONLY source of a caller's role; the claimed role is advisory.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from nomorals.family.members import Household as FamHousehold
from nomorals.family.registry import FAMILY_TOOL_NAMES, family_registry
from nomorals.family.store import FamilyItem, PerPersonStore


def _reg(tmp_path):
    reg = family_registry(tmp_path / "fam")
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="owner1", display_name="Owner", role="owner")
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="adult1", display_name="Adult", role="adult")
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="kid1", display_name="Kid", role="child")
    reg.call("family_item_put", actor_id="adult1", actor_role="adult",
             person_id="adult1", item_id="diary", kind="note",
             title="adult diary", body="private thoughts",
             visibility="private")
    return reg


# ── A: claimed roles confer nothing ──────────────────────────────────

def test_child_cannot_revoke_by_claiming_owner(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(PermissionError):
        reg.call("family_person_revoke", actor_id="kid1",
                 actor_role="owner", person_id="adult1")
    assert reg.household.is_active("adult1")


def test_stranger_cannot_revoke_or_set_role(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(PermissionError):
        reg.call("family_person_revoke", actor_id="ghost",
                 actor_role="owner", person_id="adult1")
    with pytest.raises(PermissionError):
        reg.household.set_role(actor_id="ghost", actor_role="owner",
                               person_id="kid1", role="adult")
    assert reg.household.get_member("kid1")["role"] == "child"


def test_child_cannot_add_member_by_claiming_owner(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(PermissionError):
        reg.call("family_person_add", actor_id="kid1", actor_role="owner",
                 person_id="kid2", display_name="Kid2", role="child")
    assert reg.household.get_member("kid2") is None


def test_revoked_member_cannot_manage_by_claiming_owner(tmp_path):
    reg = _reg(tmp_path)
    reg.call("family_person_revoke", actor_id="owner1", actor_role="owner",
             person_id="adult1")
    with pytest.raises(PermissionError):
        reg.call("family_person_add", actor_id="adult1", actor_role="owner",
                 person_id="kid2", display_name="Kid2", role="child")


def test_audit_list_requires_a_real_owner(tmp_path):
    reg = _reg(tmp_path)
    # The old call shape (claimed role only, no actor) is refused.
    with pytest.raises(TypeError):
        reg.call("family_audit_list", actor_role="owner")
    # A non-owner member with an actor id is refused.
    with pytest.raises(PermissionError):
        reg.call("family_audit_list", actor_id="adult1", actor_role="owner")
    # A stranger is refused.
    with pytest.raises(PermissionError):
        reg.call("family_audit_list", actor_id="ghost", actor_role="owner")
    # The real owner reads it.
    entries = reg.call("family_audit_list", actor_id="owner1",
                       actor_role="owner")
    assert any(e["action"] == "person_add" for e in entries)


def test_private_item_not_leaked_by_claimed_owner_role(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(PermissionError):
        reg.call("family_item_get", actor_id="kid1", actor_role="owner",
                 person_id="adult1", item_id="diary")
    listed = reg.call("family_item_list", actor_id="kid1",
                      actor_role="owner", person_id="adult1")
    assert [i["id"] for i in listed] == []


def test_revoked_member_cannot_share_others_item(tmp_path):
    reg = _reg(tmp_path)
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="teen1", display_name="Teen", role="teen")
    reg.call("family_person_revoke", actor_id="owner1", actor_role="owner",
             person_id="teen1")
    with pytest.raises(PermissionError):
        reg.call("family_item_share", actor_id="teen1", actor_role="owner",
                 person_id="adult1", item_id="diary")
    # Still private afterwards.
    got = reg.call("family_item_get", actor_id="owner1", actor_role="owner",
                   person_id="adult1", item_id="diary")
    assert got["visibility"] == "private"


def test_stranger_cannot_write_into_a_members_store(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(PermissionError):
        reg.call("family_item_put", actor_id="ghost", actor_role="owner",
                 person_id="adult1", item_id="planted", kind="note",
                 title="planted", body="by a stranger", visibility="shared")


def test_peer_cannot_write_into_a_members_store_by_claiming_owner(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(PermissionError):
        reg.call("family_item_put", actor_id="kid1", actor_role="owner",
                 person_id="adult1", item_id="planted", kind="note",
                 title="planted", body="by a peer", visibility="shared")


def test_authority_is_the_roster_record_not_the_claim(tmp_path):
    reg = _reg(tmp_path)
    # The owner claiming a LOWER role keeps owner authority (the claim
    # is advisory); a member claiming a HIGHER role gains nothing
    # (pinned by the tests above). Both directions = roster wins.
    got = reg.call("family_item_get", actor_id="owner1",
                   actor_role="child", person_id="adult1", item_id="diary")
    assert got["title"] == "adult diary"


def test_legit_owner_flows_still_work(tmp_path):
    reg = _reg(tmp_path)
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="teen1", display_name="Teen", role="teen")
    reg.household.set_role(actor_id="owner1", actor_role="owner",
                           person_id="teen1", role="adult")
    assert reg.household.get_member("teen1")["role"] == "adult"
    # Last-owner protections unchanged.
    with pytest.raises(ValueError):
        reg.call("family_person_revoke", actor_id="owner1",
                 actor_role="owner", person_id="owner1")


def test_household_wrapper_inherits_roster_authority(tmp_path):
    from nomorals.household.household import Household
    reg = _reg(tmp_path)
    hh = Household(registry=reg)
    try:
        with pytest.raises(PermissionError):
            hh.add_member(actor_id="kid1", actor_role="owner",
                          person_id="kid2", display_name="Kid2", role="child")
        with pytest.raises(PermissionError):
            hh.revoke_member(actor_id="ghost", actor_role="owner",
                             person_id="adult1")
    finally:
        hh.close()


def test_kid_safe_rules_use_the_real_role(tmp_path):
    reg = _reg(tmp_path)
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="teen1", display_name="Teen", role="teen")
    # A teen claiming "adult" still gets teen masking, not a free pass.
    stored = reg.call("family_item_put", actor_id="teen1",
                      actor_role="adult", person_id="teen1", item_id="t1",
                      kind="note", title="casino night", body="fun",
                      visibility="private")
    assert "casino" not in stored["title"] and "***" in stored["title"]
    # A child claiming "adult" is still refused flagged content.
    with pytest.raises(ValueError):
        reg.call("family_item_put", actor_id="kid1", actor_role="adult",
                 person_id="kid1", item_id="k1", kind="note",
                 title="casino night", body="fun", visibility="private")


# ── B: reserved person ids refuse cleanly, before any state change ───

@pytest.mark.parametrize("bad_id", ["members.json", "audit.jsonl",
                                    "household_shared.db", "MEMBERS.JSON"])
def test_reserved_person_id_refused_without_partial_add(tmp_path, bad_id):
    h = FamHousehold(tmp_path / "fam")
    h.add_person(actor_id="o", actor_role="owner", person_id="o",
                 display_name="O", role="owner")
    with pytest.raises(ValueError):
        h.add_person(actor_id="o", actor_role="owner", person_id=bad_id,
                     display_name="Evil", role="adult")
    roster = json.loads((tmp_path / "fam" / "members.json")
                        .read_text(encoding="utf-8"))
    assert list(roster) == ["o"]
    entries = h.audit_log(actor_role="owner")
    assert [e["action"] for e in entries] == ["person_add"]
    # The household is fully usable afterwards.
    h.add_person(actor_id="o", actor_role="owner", person_id="ok1",
                 display_name="Ok", role="adult")
    assert h.is_active("ok1")


# ── C: the delete tool exists and enforces own-or-owner ──────────────

def test_item_delete_tool_end_to_end(tmp_path):
    reg = _reg(tmp_path)
    assert "family_item_delete" in FAMILY_TOOL_NAMES
    assert "family_item_delete" in reg.tool_names
    # A peer cannot delete the adult's item, even claiming owner.
    with pytest.raises(PermissionError):
        reg.call("family_item_delete", actor_id="kid1", actor_role="owner",
                 person_id="adult1", item_id="diary")
    # The owner of the item deletes it; the deletion is audit-logged.
    out = reg.call("family_item_delete", actor_id="adult1",
                   actor_role="adult", person_id="adult1", item_id="diary")
    assert out == {"deleted": True, "person_id": "adult1",
                   "item_id": "diary"}
    with pytest.raises(KeyError):
        reg.call("family_item_get", actor_id="adult1", actor_role="adult",
                 person_id="adult1", item_id="diary")
    entries = reg.call("family_audit_list", actor_id="owner1",
                       actor_role="owner")
    assert any(e["action"] == "item_delete" and e["target"] == "diary"
               for e in entries)
    # Deleting a missing item is an honest False, not an error.
    out = reg.call("family_item_delete", actor_id="adult1",
                   actor_role="adult", person_id="adult1", item_id="diary")
    assert out["deleted"] is False


def test_owner_can_delete_another_members_item(tmp_path):
    reg = _reg(tmp_path)
    out = reg.call("family_item_delete", actor_id="owner1",
                   actor_role="owner", person_id="adult1", item_id="diary")
    assert out["deleted"] is True


# ── D/E: durable, atomic, race-free writes ───────────────────────────

def test_concurrent_puts_from_two_stores_lose_nothing(tmp_path):
    root = tmp_path / "fam"
    h = FamHousehold(root)
    h.add_person(actor_id="o", actor_role="owner", person_id="o",
                 display_name="O", role="owner")
    s1, s2 = PerPersonStore(root), PerPersonStore(root)
    errors: list[BaseException] = []

    def put(store: PerPersonStore, tag: str) -> None:
        try:
            for i in range(60):
                store.put_item(FamilyItem(id=f"{tag}-{i}", owner_id="o",
                                          kind="note", title=f"t{tag}{i}"),
                               requester_role="owner")
        except BaseException as exc:  # noqa: BLE001 — collected + asserted
            errors.append(exc)

    threads = [threading.Thread(target=put, args=(s1, "a")),
               threading.Thread(target=put, args=(s2, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    got = s1.list_items("o", requester_id="o", requester_role="owner")
    assert len(got) == 120


def test_concurrent_member_adds_from_two_households_lose_nothing(tmp_path):
    root = tmp_path / "fam"
    h1, h2 = FamHousehold(root), FamHousehold(root)
    h1.add_person(actor_id="o", actor_role="owner", person_id="o",
                  display_name="O", role="owner")
    errors: list[BaseException] = []

    def add(hh: FamHousehold, tag: str) -> None:
        try:
            for i in range(25):
                hh.add_person(actor_id="o", actor_role="owner",
                              person_id=f"{tag}{i}", display_name=tag,
                              role="adult")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(h1, "a")),
               threading.Thread(target=add, args=(h2, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    members = h1.list_members(actor_role="owner")
    assert len(members) == 51


def test_interrupted_write_never_tears_the_items_file(tmp_path, monkeypatch):
    root = tmp_path / "fam"
    store = PerPersonStore(root)
    store.put_item(FamilyItem(id="kept", owner_id="o", kind="note",
                              title="kept"), requester_role="owner")
    real_write_text = Path.write_text

    def torn_write(self, data, *args, **kwargs):  # noqa: ANN001
        real_write_text(self, str(data)[:5], *args, **kwargs)
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(Path, "write_text", torn_write)
    with pytest.raises(OSError):
        store.put_item(FamilyItem(id="new", owner_id="o", kind="note",
                                  title="new"), requester_role="owner")
    monkeypatch.undo()
    # The previous complete state is intact and loadable.
    got = store.list_items("o", requester_id="o", requester_role="owner")
    assert [i.id for i in got] == ["kept"]


def test_state_survives_reopen_byte_for_byte(tmp_path):
    root = tmp_path / "fam"
    reg = family_registry(root)
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="owner1", display_name="Owner", role="owner")
    reg.call("family_person_add", actor_id="owner1", actor_role="owner",
             person_id="adult1", display_name="Adult", role="adult")
    reg.call("family_item_put", actor_id="adult1", actor_role="adult",
             person_id="adult1", item_id="diary", kind="note",
             title="adult diary", body="private thoughts",
             visibility="private")
    before_members = json.loads((root / "members.json")
                                .read_text(encoding="utf-8"))
    before_items = json.loads((root / "adult1" / "items.json")
                              .read_text(encoding="utf-8"))
    before_profile = json.loads((root / "adult1" / "profile.json")
                                .read_text(encoding="utf-8"))
    reg2 = family_registry(root)
    after_members = reg2.household.list_members(actor_role="owner")
    assert {m["person_id"]: m for m in after_members} == before_members
    items = reg2.call("family_item_list", actor_id="adult1",
                      actor_role="adult", person_id="adult1")
    assert items == before_items
    assert reg2.household.store.get_profile("adult1") == before_profile
