"""MORE_TO_ADD gaps 4-6: family mode, location layer, offline mode.

* Family isolation: zero forbidden imports + per-person state never
  crosses + kid-safe enforced + owner revoke works.
* Location: refuses without consent + geofence enter/exit correct.
* Offline: queue survives restart + sync merges + matrix covers features.
"""

import ast
from pathlib import Path

import pytest

from nomorals.family import (
    FAMILY_CAPABILITIES,
    FamilyItem,
    Household,
    PerPersonStore,
    filter_for_role,
    is_kid_safe,
    scan_text,
)
from nomorals.family.locales import get_locale_pack
from nomorals.family.policy import is_family_capability
from nomorals.family.registry import FAMILY_TOOL_NAMES, family_registry
from nomorals.family.roles import can
from nomorals.location import (
    ConsentError,
    ConsentStore,
    FenceState,
    LatLon,
    LocationReminder,
    check_reminders,
    circle_fence,
    evaluate_update,
    polygon_fence,
)
from nomorals.offline import (
    OFFLINE_MATRIX,
    OfflineQueue,
    SyncEngine,
    features_at,
    matrix_table,
    offline_level,
)

ROOT = Path(__file__).resolve().parent.parent
FAMILY_DIR = ROOT / "nomorals" / "family"

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


# ── Gap 4: family ──────────────────────────────────────────────────

def test_family_zero_forbidden_imports():
    violations: list[str] = []
    for path in sorted(FAMILY_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for mod in _imported_modules(tree):
            if mod.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{path.name}: imports {mod}")
            if any(s in mod for s in FORBIDDEN_SUBSTRINGS):
                violations.append(f"{path.name}: imports {mod} (vault?)")
            # No absolute nomorals.* imports at all: stdlib + siblings only.
            if mod.startswith("nomorals."):
                violations.append(f"{path.name}: absolute import {mod}")
    assert not violations, "family isolation breach:\n" + "\n".join(violations)


def test_family_registry_is_closed_allowlist():
    reg = family_registry()
    assert reg.tool_names == sorted(FAMILY_TOOL_NAMES)
    for name in reg.tool_names:
        assert name.startswith("family_")
    with pytest.raises(LookupError):
        reg.call("exec_shell", actor_id="x", actor_role="owner")
    with pytest.raises(LookupError):
        reg.call("memory_read", actor_id="x", actor_role="owner")


def test_family_policy_allows_and_denies():
    assert is_family_capability("family.item")
    assert is_family_capability("family.audit")
    assert not is_family_capability("mem.read")
    assert not is_family_capability("exec.shell")
    assert not is_family_capability("vault.read")
    assert set(FAMILY_CAPABILITIES) <= {
        "family.profile", "family.item", "family.share",
        "family.audit", "family.safety"}


def test_family_roles_permissions():
    assert can("owner", "manage_members")
    assert can("owner", "view_audit")
    assert can("owner", "read_private_other")
    assert not can("adult", "manage_members")
    assert not can("adult", "view_audit")
    assert not can("adult", "read_private_other")
    assert not can("teen", "share_item")
    assert not can("child", "share_item")
    assert can("child", "read_shared")
    assert can("child", "write_shared")
    with pytest.raises(ValueError):
        can("superuser", "read_shared")


def test_family_locale_packs_nigeria_default_never_boundary():
    ng = get_locale_pack("ng")
    assert ng["country"] == "Nigeria"
    assert ng["default_city"] == "Lagos"
    assert get_locale_pack(None)["code"] == "ng"
    # Unknown pack: generic behaviour, never a rejection.
    xx = get_locale_pack("xx")
    assert xx["code"] == "xx"
    assert is_kid_safe("hello world", "xx")
    assert not is_kid_safe("porn", "xx")


def test_family_per_person_state_never_crosses(tmp_path):
    h = Household(tmp_path / "fam")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="owner1",
                 display_name="Owner", role="owner")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="adult1",
                 display_name="Adult", role="adult")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="kid1",
                 display_name="Kid", role="child")
    store: PerPersonStore = h.store
    store.put_item(FamilyItem(id="a-secret", owner_id="adult1", kind="note",
                              title="adult diary", body="private thoughts",
                              visibility="private"),
                   requester_role="adult")
    store.put_item(FamilyItem(id="a-shared", owner_id="adult1", kind="note",
                              title="grocery list", body="yam, rice",
                              visibility="shared"),
                   requester_role="adult")
    # A peer cannot read the private item, in either direction.
    with pytest.raises(PermissionError):
        store.get_item("adult1", "a-secret", requester_id="kid1",
                       requester_role="child")
    # ...but CAN read the shared one.
    assert store.get_item("adult1", "a-shared", requester_id="kid1",
                          requester_role="child").title == "grocery list"
    # The owner can audit-read the private item.
    assert store.get_item("adult1", "a-secret", requester_id="owner1",
                          requester_role="owner").title == "adult diary"
    # Physically separate directories.
    assert (tmp_path / "fam" / "adult1" / "items.json").exists()
    assert (tmp_path / "fam" / "kid1").exists()
    kid_items = store.list_items("adult1", requester_id="kid1",
                                 requester_role="child")
    assert [i.id for i in kid_items] == ["a-shared"]


def test_family_kid_safe_enforced(tmp_path):
    h = Household(tmp_path / "fam")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="owner1",
                 display_name="Owner", role="owner")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="kid1",
                 display_name="Kid", role="child")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="teen1",
                 display_name="Teen", role="teen")
    # Child: flagged content refused at store level...
    with pytest.raises(ValueError):
        h.store.put_item(FamilyItem(id="bad", owner_id="kid1", kind="note",
                                    title="casino night", body="betting"),
                         requester_role="child")
    # ...and at filter level.
    assert not is_kid_safe("let's go to the casino", "ng")
    assert scan_text("porn", "ng") == ["porn"]
    refused = filter_for_role("porn video", "child", "ng")
    assert refused["allowed"] is False and refused["text"] == ""
    # Teen: masked, not refused.
    masked = filter_for_role("porn video", "teen", "ng")
    assert masked["allowed"] is True
    assert "porn" not in masked["text"] and "***" in masked["text"]
    # Adult: passes unchanged, flags reported.
    adult = filter_for_role("porn video", "adult", "ng")
    assert adult["allowed"] is True and adult["text"] == "porn video"
    assert adult["flags"] == ["porn"]
    # Nigeria pack extras apply: Pidgin term flagged in ng, not in us.
    assert not is_kid_safe("ashawo runs", "ng")
    assert is_kid_safe("ashawo runs", "us")
    # Kid-safe flag on stored items: clean item is kid_safe.
    item = h.store.put_item(FamilyItem(id="ok", owner_id="kid1", kind="note",
                                       title="homework", body="maths"),
                            requester_role="child")
    assert item.kid_safe is True


def test_family_owner_revoke_works(tmp_path):
    h = Household(tmp_path / "fam")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="owner1",
                 display_name="Owner", role="owner")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="adult1",
                 display_name="Adult", role="adult")
    h.add_person(actor_id="owner1", actor_role="owner", person_id="kid1",
                 display_name="Kid", role="child")
    h.store.put_item(FamilyItem(id="k-note", owner_id="kid1", kind="note",
                                title="kid secret", body="shh",
                                visibility="private"),
                     requester_role="child")
    # Non-owner cannot revoke.
    with pytest.raises(PermissionError):
        h.revoke_person(actor_id="adult1", actor_role="adult",
                        person_id="kid1")
    # Owner revokes: access cut immediately...
    revoked = h.revoke_person(actor_id="owner1", actor_role="owner",
                              person_id="kid1", reason="grounded")
    assert revoked["status"] == "revoked"
    assert h.is_revoked("kid1")
    with pytest.raises(PermissionError):
        h.store.get_item("kid1", "k-note", requester_id="kid1",
                         requester_role="child", requester_revoked=True)
    # ...but the owner's audit read still works (data kept, not deleted).
    assert h.store.get_item("kid1", "k-note", requester_id="owner1",
                            requester_role="owner",
                            owner_revoked=True).title == "kid secret"
    # Peers lose access to the revoked member's items entirely.
    assert h.store.list_items("kid1", requester_id="adult1",
                              requester_role="adult",
                              owner_revoked=True) == []
    # Revocation is audit-logged; audit readable only by owner.
    entries = h.audit_log(actor_role="owner")
    assert any(e["action"] == "person_revoke" and e["target"] == "kid1"
               for e in entries)
    with pytest.raises(PermissionError):
        h.audit_log(actor_role="adult")
    # Last owner cannot be revoked or demoted.
    with pytest.raises(ValueError):
        h.revoke_person(actor_id="owner1", actor_role="owner",
                        person_id="owner1")
    with pytest.raises(ValueError):
        h.set_role(actor_id="owner1", actor_role="owner",
                   person_id="owner1", role="adult")
    # Bootstrap: first member must be an owner.
    h2 = Household(tmp_path / "fam2")
    with pytest.raises(ValueError):
        h2.add_person(actor_id="x", actor_role="owner", person_id="x",
                      display_name="X", role="child")


def test_family_registry_end_to_end(tmp_path):
    reg = family_registry(tmp_path / "fam")
    reg.call("family_person_add", actor_id="o", actor_role="owner",
             person_id="o", display_name="O", role="owner")
    reg.call("family_person_add", actor_id="o", actor_role="owner",
             person_id="k", display_name="K", role="child")
    stored = reg.call("family_item_put", actor_id="k", actor_role="child",
                      person_id="k", item_id="hw", kind="note",
                      title="homework", body="maths", visibility="shared")
    assert stored["kid_safe"] is True
    got = reg.call("family_item_get", actor_id="o", actor_role="owner",
                   person_id="k", item_id="hw")
    assert got["title"] == "homework"
    # Kid-safe check tool works through the registry.
    assert reg.call("family_kid_check", text="porn", role="child",
                    locale="ng")["allowed"] is False


# ── Gap 5: location ────────────────────────────────────────────────

def test_location_refuses_without_consent():
    fence = circle_fence("home", "Home", LatLon(6.5244, 3.3792), 500.0)
    states: dict[str, FenceState] = {}
    with pytest.raises(ConsentError):
        evaluate_update(person_id="owner", loc=LatLon(6.5244, 3.3792),
                        fences=[fence], states=states, consent=False)
    with pytest.raises(ConsentError):
        check_reminders([], [], consent=False)
    # Consent store: opt-in default is NO, grant then revoke.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cs = ConsentStore(td)
        assert not cs.has_consent("owner", "geofence")
        with pytest.raises(ConsentError):
            cs.check("owner", "geofence")
        cs.grant("owner", "geofence")
        assert cs.has_consent("owner", "geofence")
        cs.revoke("owner", "geofence")
        assert not cs.has_consent("owner", "geofence")
    # One scope's grant never implies another's.
    with pytest.raises(ConsentError):
        check_reminders([], [], consent=False)


def test_location_geofence_enter_exit_circle():
    home = circle_fence("home", "Home", LatLon(6.5244, 3.3792), 500.0)
    outside = LatLon(6.6000, 3.3792)  # ~8.4 km north: outside
    inside = LatLon(6.5244, 3.3792)   # centre: inside
    assert not home.contains(outside)
    assert home.contains(inside)
    states: dict[str, FenceState] = {}
    # First update establishes baseline: NO event.
    assert evaluate_update(person_id="o", loc=outside, fences=[home],
                            states=states, consent=True) == []
    # Still outside: no event.
    assert evaluate_update(person_id="o", loc=outside, fences=[home],
                            states=states, consent=True) == []
    # Arrive: entered.
    events = evaluate_update(person_id="o", loc=inside, fences=[home],
                             states=states, consent=True)
    assert [e["event"] for e in events] == ["entered"]
    assert events[0]["fence_id"] == "home"
    # Leave: exited.
    events = evaluate_update(person_id="o", loc=outside, fences=[home],
                             states=states, consent=True)
    assert [e["event"] for e in events] == ["exited"]
    # State round-trips.
    st = states["home"]
    assert FenceState.from_dict(st.to_dict()).inside is False


def test_location_geofence_polygon():
    square = polygon_fence("sq", "Square", [
        LatLon(0.0, 0.0), LatLon(0.0, 1.0),
        LatLon(1.0, 1.0), LatLon(1.0, 0.0)])
    assert square.contains(LatLon(0.5, 0.5))
    assert not square.contains(LatLon(2.0, 0.5))
    assert square.contains(LatLon(0.0, 0.5))  # boundary counts as inside
    with pytest.raises(ValueError):
        polygon_fence("bad", "Bad", [LatLon(0, 0), LatLon(1, 1)])
    with pytest.raises(ValueError):
        LatLon(91.0, 0.0)


def test_location_reminders_fire_on_enter_and_exit():
    home = circle_fence("home", "Home", LatLon(6.5244, 3.3792), 500.0)
    outside, inside = LatLon(6.6000, 3.3792), LatLon(6.5244, 3.3792)
    arrive = LocationReminder(id="r1", person_id="o", fence_id="home",
                              message="Buy yam", fire_on="enter")
    leave = LocationReminder(id="r2", person_id="o", fence_id="home",
                             message="Lock up", fire_on="exit")
    states: dict[str, FenceState] = {}
    evaluate_update(person_id="o", loc=outside, fences=[home],
                    states=states, consent=True)
    events = evaluate_update(person_id="o", loc=inside, fences=[home],
                             states=states, consent=True)
    fired = check_reminders([arrive, leave], events, consent=True)
    assert [r.id for r in fired] == ["r1"]
    assert arrive.fired and not leave.fired
    # Fired once: does not fire again on the same event.
    assert check_reminders([arrive, leave], events, consent=True) == []
    # Exit fires the leave reminder.
    events = evaluate_update(person_id="o", loc=outside, fences=[home],
                             states=states, consent=True)
    fired = check_reminders([arrive, leave], events, consent=True)
    assert [r.id for r in fired] == ["r2"]


def test_location_nigeria_default_not_boundary():
    from nomorals.location import default_place
    assert default_place().name == "Lagos"
    assert default_place("xx").name == "Lagos"  # unknown locale → default
    assert default_place("us").name == "New York"
    # Any coordinate on Earth works.
    tokyo = circle_fence("t", "Tokyo", LatLon(35.6762, 139.6503), 1000.0)
    assert tokyo.contains(LatLon(35.6762, 139.6503))


# ── Gap 6: offline ─────────────────────────────────────────────────

def test_offline_queue_survives_restart(tmp_path):
    q1 = OfflineQueue(tmp_path / "off")
    a1 = q1.enqueue("note_write", {"text": "hello"}, entity="note:1",
                    base_version=0)
    q1.enqueue("email_send", {"to": "a@b.c"}, entity="email:1")
    assert q1.count_pending() == 2
    # New instance over the same root = "restart": queue intact.
    q2 = OfflineQueue(tmp_path / "off")
    assert q2.count_pending() == 2
    assert q2.get(a1.id) is not None
    assert q2.get(a1.id).payload == {"text": "hello"}
    assert [a.kind for a in q2.pending()] == ["note_write", "email_send"]


def test_offline_sync_merges_in_order(tmp_path):
    q = OfflineQueue(tmp_path / "off")
    applied: list[str] = []
    q.enqueue("step", {"n": 1}, entity="doc:1", base_version=0)
    q.enqueue("step", {"n": 2}, entity="doc:1", base_version=1)
    q.enqueue("step", {"n": 3}, entity="doc:1", base_version=2)
    engine = SyncEngine(q, lambda a: applied.append(a.payload["n"]),
                        versions={"doc:1": 0})
    result = engine.sync()
    assert result.ok
    assert applied == [1, 2, 3]
    assert q.count_pending() == 0
    assert engine.versions["doc:1"] == 3


def test_offline_sync_conflict_flagged_not_clobbered(tmp_path):
    q = OfflineQueue(tmp_path / "off")
    applied: list[str] = []
    q.enqueue("edit", {"text": "stale"}, entity="doc:1", base_version=0)
    q.enqueue("edit", {"text": "fresh-ish"}, entity="doc:2", base_version=0)
    # doc:1 moved on while offline (current 5 != base 0) → conflict.
    engine = SyncEngine(q, lambda a: applied.append(a.payload["text"]),
                        versions={"doc:1": 5, "doc:2": 0})
    result = engine.sync()
    assert not result.ok
    assert len(result.conflicts) == 1
    assert result.conflicts[0].entity == "doc:1"
    assert result.conflicts[0].current_version == 5
    # The stale action was NOT applied; the good one was.
    assert applied == ["fresh-ish"]
    # Conflict persisted + visible; nothing silently dropped.
    assert len(engine.conflicts()) == 1
    assert engine.conflicts()[0].action_id == result.conflicts[0].action_id
    # Failed applier: recorded, sync continues, nothing hidden.
    q.enqueue("boom", {}, entity="doc:3", base_version=0)

    def _applier(a):
        if a.kind == "boom":
            raise RuntimeError("kaboom")
        applied.append(a.kind)

    engine2 = SyncEngine(q, _applier, versions={"doc:3": 0})
    result2 = engine2.sync()
    assert result2.failed and result2.failed[0]["action_id"]
    assert q.get(result2.failed[0]["action_id"]).status == "failed"


def test_offline_matrix_covers_features():
    assert len(OFFLINE_MATRIX) >= 20
    for feature, entry in OFFLINE_MATRIX.items():
        assert entry["level"] in ("full", "degraded", "unavailable"), feature
        assert entry["notes"], feature
    # Safety-critical: money never moves offline.
    assert offline_level("trading_execution") == "unavailable"
    assert offline_level("payments") == "unavailable"
    # Local-first features genuinely work offline.
    assert offline_level("chat_reply") == "full"
    assert offline_level("location_triggers") == "full"
    assert offline_level("family_items") == "full"
    # Queued features degrade, never fake success.
    assert offline_level("email_send") == "degraded"
    assert offline_level("social_post") == "degraded"
    # Unknown feature: fail-closed.
    assert offline_level("teleport") == "unavailable"
    # Table + level buckets are consistent.
    rows = matrix_table()
    assert len(rows) == len(OFFLINE_MATRIX)
    assert all(r["feature"] in OFFLINE_MATRIX for r in rows)
    assert set(features_at("full")) | set(features_at("degraded")) \
        | set(features_at("unavailable")) == set(OFFLINE_MATRIX)
    with pytest.raises(ValueError):
        features_at("sometimes")
