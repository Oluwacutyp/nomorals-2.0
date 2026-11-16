"""Family mode, wired into production.

MORE_TO_ADD gap 4: the domain (nomorals/family/ — members, policy,
roles, safety, store, registry) existed and was unit-pinned
(tests/test_more_gaps456.py) but nothing in production reached it —
no durable household, no chat command, no dispatch. Every test here
drives the PRODUCTION wiring: the durable Household under the home's
family/ directory and the /family control — never the dormant organ
in isolation.

Teeth: on the pristine base each test fails for the missing wiring
(no _control_family, no "family" command kind).

The chat surface speaks as the household owner (actor "owner"):
authority always resolves from the roster, never from a claimed
role, and the first household member must be an owner. One exception:
item writes act as the item's owner (actor = person_id), so the
organ's per-role kid-safe rules apply to the member the item belongs
to — a child's flagged write is refused and a teen's is masked, the
way the registry's own end-to-end test drives it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.family.members import Household
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    detailed_help,
    help_text,
)
from nomorals.social.chat.personal_commands import PERSONAL_OWNER_COMMANDS


def _reach_host(tmp_path):
    """Production control host on a throwaway home (assets harness)."""
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _bootstrap(host):
    """Owner + adult + child, through the real /family verbs."""
    out = host._control_family("add owner Owner")
    assert out.startswith("added: owner — Owner (owner"), out
    out = host._control_family("add ada Adaeze | adult")
    assert out.startswith("added: ada — Adaeze (adult"), out
    out = host._control_family("add kid Kid | child")
    assert out.startswith("added: kid — Kid (child"), out


# ── durable store ───────────────────────────────────────────────────


def test_household_persists_across_instances(tmp_path):
    root = tmp_path / "fam"
    h = Household(root)
    h.add_person(actor_id="owner", actor_role="owner", person_id="owner",
                 display_name="Owner", role="owner")
    h.add_person(actor_id="owner", actor_role="owner", person_id="ada",
                 display_name="Adaeze", role="adult")
    from nomorals.family.store import FamilyItem
    h.store.put_item(FamilyItem(id="note-1", owner_id="ada", kind="note",
                                title="grocery list", body="yam, rice",
                                visibility="shared"),
                     requester_role="adult")

    fresh = Household(root)
    members = fresh.list_members(actor_role="owner")
    assert [m["person_id"] for m in members] == ["ada", "owner"]
    items = fresh.store.list_items("ada", requester_id="owner",
                                   requester_role="owner")
    assert [i.id for i in items] == ["note-1"]
    assert items[0].title == "grocery list"
    # Roster + items + audit all live on disk under the same root.
    assert (root / "members.json").exists()
    assert (root / "ada" / "items.json").exists()
    assert (root / "audit.jsonl").exists()
    assert any(e["action"] == "person_add"
               for e in fresh.audit_log(actor_role="owner",
                                       actor_id="owner"))


# ── /family control: members ────────────────────────────────────────


def test_family_empty_household_answers(tmp_path):
    host = _reach_host(tmp_path)
    assert "no household yet" in host._control_family("")
    assert "no members yet" in host._control_family("list")


def test_family_bootstrap_first_member_must_be_owner(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_family("add kid Kid | child")
    assert "first household member must be an owner" in out
    # No role given on an empty roster defaults to owner: bootstrap.
    out = host._control_family("add owner Owner")
    assert out.startswith("added: owner — Owner (owner"), out
    assert "Owner (owner" in host._control_family("list")


def test_family_add_list_revoke_role(tmp_path):
    host = _reach_host(tmp_path)
    _bootstrap(host)
    roster = host._control_family("list")
    assert "owner — Owner (owner" in roster
    assert "ada — Adaeze (adult" in roster
    assert "kid — Kid (child" in roster
    # Duplicate add refuses; bad role refuses; bad id refuses.
    assert "already a member" in host._control_family(
        "add ada Adaeze | adult")
    assert "unknown family role" in host._control_family(
        "add zed Zed | superuser")
    assert "invalid person_id" in host._control_family(
        "add ../evil Evil | adult")
    # Role change: adult -> teen, then back.
    out = host._control_family("role ada teen")
    assert out == "role: ada is now teen (was adult)", out
    assert "ada — Adaeze (teen" in host._control_family("list")
    host._control_family("role ada adult")
    # Revoke cuts the member from the active roster.
    out = host._control_family("revoke kid grounded")
    assert out.startswith("revoked: kid — Kid (child)"), out
    assert "kid — Kid" not in host._control_family("list")
    assert "no active member" in host._control_family("revoke kid")
    # The last owner cannot be revoked.
    assert "cannot revoke the last owner" in host._control_family(
        "revoke owner")


def test_family_bare_shows_roster_and_verbs(tmp_path):
    host = _reach_host(tmp_path)
    _bootstrap(host)
    out = host._control_family("")
    assert "owner — Owner (owner" in out
    assert "ada — Adaeze (adult" in out
    assert "add" in out and "audit" in out
    assert "usage:" in host._control_family("frobnicate")


# ── /family control: items ──────────────────────────────────────────


def _add_item(host, person="ada", kind="note", title="grocery list",
              rest=""):
    out = host._control_family(f"item {person} {kind} {title}{rest}")
    assert out.startswith("stored: "), out
    return out.split(":")[1].strip().split()[0]


def test_family_items_end_to_end(tmp_path):
    host = _reach_host(tmp_path)
    _bootstrap(host)
    item_id = _add_item(host, rest=" | yam, rice | shared")
    listed = host._control_family("items ada")
    assert "grocery list" in listed and item_id in listed
    assert "shared" in listed
    shown = host._control_family(f"show ada {item_id}")
    assert "grocery list" in shown and "yam, rice" in shown
    assert "shared" in shown and "kid-safe: yes" in shown
    # Share flips a private item to shared.
    private_id = _add_item(host, title="diary")
    assert "private" in host._control_family(f"show ada {private_id}")
    out = host._control_family(f"share ada {private_id}")
    assert out.startswith(f"shared: {private_id} — diary"), out
    assert "shared" in host._control_family(f"show ada {private_id}")
    # Delete removes it; a second show says so plainly.
    assert host._control_family(
        f"delete ada {private_id}") == f"deleted: {private_id}"
    assert "no item" in host._control_family(f"show ada {private_id}")
    assert "no item" in host._control_family("delete ada nope")
    assert "no items for zed" in host._control_family("items zed")


def test_family_kid_safe_through_chat(tmp_path):
    host = _reach_host(tmp_path)
    _bootstrap(host)
    host._control_family("add teen Teen | teen")
    # A child storing flagged content is refused outright.
    out = host._control_family("item kid note casino night")
    assert "refused" in out
    assert "casino night" not in host._control_family("items kid")
    # A teen storing flagged content gets it masked, not refused.
    teen_id = _add_item(host, person="teen", title="porn video")
    shown = host._control_family(f"show teen {teen_id}")
    assert "***" in shown and "porn" not in shown
    # A clean child item stores fine.
    kid_id = _add_item(host, person="kid", title="homework")
    assert kid_id
    # An adult storing flagged content passes, flags reported.
    adult_id = _add_item(host, title="porn video")
    shown = host._control_family(f"show ada {adult_id}")
    assert "porn video" in shown and "flagged: porn" in shown


# ── /family control: audit, safety, policy ─────────────────────────


def test_family_audit_records_changes(tmp_path):
    host = _reach_host(tmp_path)
    _bootstrap(host)
    host._control_family("revoke kid grounded")
    audit = host._control_family("audit")
    assert "person_add" in audit and "person_revoke" in audit
    assert "kid" in audit and "owner" in audit
    # Bare audit on a fresh household is empty, not an error.
    fresh = _reach_host(tmp_path / "other")
    assert "empty" in fresh._control_family("audit")


def test_family_check_and_policy(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_family("check child porn video")
    assert out.startswith("refused for child"), out
    assert "porn" in out
    out = host._control_family("check teen porn video")
    assert out.startswith("masked for teen: *** video"), out
    out = host._control_family("check adult porn video")
    assert out.startswith("passes for adult"), out
    assert "porn" in out
    out = host._control_family("check adult hello world")
    assert "clean" in out
    assert "unknown family role" in host._control_family(
        "check superuser hello")
    policy = host._control_family("policy")
    assert "family.item" in policy and "family.audit" in policy
    assert "family.share" in policy and "family.safety" in policy
    assert "denied" in policy and "ng" in policy
    assert "usage:" in host._control_family("check adult")


# ── registry ────────────────────────────────────────────────────────


def test_family_registered():
    assert "family" in CONTROL_COMMANDS
    slots = [title for title, kinds in LIST_GROUPS if "family" in kinds]
    assert slots == ["memory & thinking"]
    assert "family" in LIST_ONELINERS and "/family" in LIST_ONELINERS["family"]
    assert "/family" in help_text()
    page = detailed_help("family")
    assert "try:" in page and "related:" in page
    assert "/family add" in page


def test_family_owner_only_off_menu():
    # Owner-only: absent from PUBLIC_COMMANDS, so caller_may_run denies
    # non-owners before the handler runs (assets/legacy pattern).
    assert "family" not in PUBLIC_COMMANDS
    # Off the BotFather menu by design: the 100-command cap is full
    # (48 owner pins) — registered, runnable, /list-ed, like the
    # reachability wave's off-menu owner commands.
    assert "family" not in PERSONAL_OWNER_COMMANDS
