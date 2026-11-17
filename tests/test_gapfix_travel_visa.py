"""Gapfix R3-67: wire travel visa + group-trip share into the live paths.

- /trip visa <origin> <destination> <passport> — advisory visa answer
  from the curated table via _control_trip (the live /trip surface).
  Unknown routes stay unknown; nothing is invented, nothing is gated.
- /gtrip share [id] — the trip's share card posted to the group via
  CommunityGroupTripBridge over the group's own store.
- /gtrip rsvp <going|maybe|declined> [id] + /gtrip rsvps [id] — RSVPs
  recorded durably in the group trip store (not the bridge's in-memory
  dict), so a later call in a fresh process sees them.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.travel.community_impl import CommunityGroupTripBridge
from nomorals.travel.groups import GroupTripStore, control_gtrip


def _group_chat():
    return SimpleNamespace(kind="group", platform="test", id="g1")


def _ctx(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(community_dir=str(tmp_path)))


def _make_trip(tmp_path, name="Lagos Trip"):
    out = control_gtrip(f"new {name}", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "Group trip created" in out
    out = control_gtrip("destination Lekki", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "Lekki" in out
    return out


# ── /trip visa ────────────────────────────────────────────────────────────

def test_trip_visa_known_route():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    out = mixin._control_trip("visa NG GH NG")
    assert "visa_free" in out
    assert "NG" in out and "GH" in out
    assert "valid passport" in out


def test_trip_visa_unknown_is_honest():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    out = mixin._control_trip("visa NG JP NG")
    assert "unknown" in out.lower()
    assert "visa_free" not in out
    assert "visa_required" not in out


def test_trip_visa_needs_three_codes():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    out = mixin._control_trip("visa NG GB")
    assert "usage" in out.lower()
    assert "/trip visa" in out


def test_trip_visa_required_names_documents():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    out = mixin._control_trip("visa NG GB NG")
    assert "visa_required" in out
    assert "DS-160" in out or "visa application" in out


# ── /gtrip share ──────────────────────────────────────────────────────────

def test_gtrip_share_posts_card_to_group(tmp_path):
    _make_trip(tmp_path)
    control_gtrip("expense 5k dinner", context=_ctx(tmp_path),
                  chat=_group_chat(), sender="Ada")
    out = control_gtrip("share", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "Lekki" in out
    assert "RSVP:" in out
    assert "Ada" in out


def test_gtrip_share_without_trip(tmp_path):
    out = control_gtrip("share", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "Create a trip first" in out


# ── /gtrip rsvp (durable) ─────────────────────────────────────────────────

def test_gtrip_rsvp_persists_across_calls(tmp_path):
    _make_trip(tmp_path)
    out = control_gtrip("rsvp going", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "going" in out.lower()
    # a fresh call (fresh store, fresh bridge) must still see it
    out = control_gtrip("rsvps", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Mama")
    assert "going" in out.lower()
    assert "Ada" in out


def test_gtrip_rsvp_rejects_bad_status(tmp_path):
    _make_trip(tmp_path)
    out = control_gtrip("rsvp perhaps", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "going" in out and "maybe" in out and "declined" in out


def test_gtrip_rsvp_without_trip(tmp_path):
    out = control_gtrip("rsvp going", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "Create a trip first" in out


def test_gtrip_help_lists_share_and_rsvp(tmp_path):
    out = control_gtrip("help", context=_ctx(tmp_path),
                        chat=_group_chat(), sender="Ada")
    assert "share" in out
    assert "rsvp" in out


# ── store-level RSVP durability ───────────────────────────────────────────

def test_store_rsvp_roundtrip(tmp_path):
    store = GroupTripStore("test:rsvp1", db_path=str(tmp_path / "r.db"))
    trip = store.create_trip("Beach", ["Ada", "Mama"], created_by="Ada")
    store.record_rsvp(trip.id, "Ada", "going")
    store.record_rsvp(trip.id, "Mama", "maybe")
    summary = store.rsvp_summary(trip.id)
    assert summary["counts"] == {"going": 1, "maybe": 1,
                                 "declined": 0, "no_reply": 0}
    assert summary["going"] == ["Ada"]
    assert summary["maybe"] == ["Mama"]


def test_store_rsvp_rejects_bad_status(tmp_path):
    store = GroupTripStore("test:rsvp2", db_path=str(tmp_path / "r2.db"))
    trip = store.create_trip("Beach", ["Ada"], created_by="Ada")
    with pytest.raises(ValueError):
        store.record_rsvp(trip.id, "Ada", "perhaps")
    with pytest.raises(ValueError):
        store.record_rsvp("no-such-trip", "Ada", "going")


def test_bridge_rsvp_delegates_to_store(tmp_path):
    tmp = tempfile.mkdtemp(prefix="gapfix67-bridge-")
    store = GroupTripStore("test:bridge",
                           db_path=str(Path(tmp) / "trips.db"))
    bridge = CommunityGroupTripBridge(store, post=None)
    trip = bridge.create_group_trip("Lagos weekend", ["Ada", "Chidi"],
                                    destination="Lekki")
    bridge.rsvp(trip.id, "Ada", "going")
    # a NEW bridge over the SAME store must see the RSVP (durable, not
    # the first bridge's in-memory dict)
    bridge2 = CommunityGroupTripBridge(store, post=None)
    summary = bridge2.rsvp_summary(trip.id)
    assert summary["counts"]["going"] == 1
    assert summary["going"] == ["Ada"]
    assert summary["no_reply"] == ["Chidi"]
