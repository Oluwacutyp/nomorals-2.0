"""Chat-verb gaps: /cgroup rituals add + /map fence edit (scout 7 top pick #1).

Two one-verb gaps on organs that already support them — no new store
or job:

* ``/cgroup rituals add`` — the rituals verb served list / run /
  enable / disable, but a group could never create a ritual of its
  own (only enable/disable the five defaults). The store
  (load_rituals / save_rituals) and the tick already read persisted
  rituals, so an added ritual fires on its own clock.
* ``/map fence edit`` — /map fence served add / list / remove /
  remind / reminders / unremind / rearm with no edit verb (the R2-28
  remainder). FenceStore.load_all / save_all / resolve_fence support
  load-mutate-save over the Geofence fields.

Seams copied from the neighbouring lanes: the rituals harness from
test_gapfix_group_rituals.py (_setup over a tmp _DATA_DIR) and the
fence rig from test_gapfix_location_fences.py (bare PartnerRuntime
over a tmp home, NM_MAPS_ENABLED=1).
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.community import digest as dmod
from nomorals.community import groups as gmod
from nomorals.community.digest import DEFAULT_RITUALS, Ritual
from nomorals.community.groups import GroupStore, control_cgroup
from nomorals.location.primitives import LatLon

FRIDAY_1600 = datetime(2026, 10, 16, 16, 0, tzinfo=UTC)
HOME = LatLon(6.52441, 3.3792)
INSIDE = LatLon(6.52450, 3.37930)


def _setup_group(tmp_path, monkeypatch):
    """One themed group (creator Ada, member Bob) under a tmp data dir."""
    monkeypatch.setattr(gmod, "_DATA_DIR", tmp_path / "themed")
    created = control_cgroup("new Testers | testing", context=None,
                             chat=None, sender="Ada", sender_id="Ada")
    m = re.search(r"`(grp_[^`]+)`", created)
    assert m, created
    gid = m.group(1)
    joined = control_cgroup(f"join {gid}", context=None, chat=None,
                            sender="Bob", sender_id="Bob")
    assert "Welcome" in joined, joined
    return gid


def _seed_last_delivered(gid, when):
    store = GroupStore()
    rituals = [Ritual(name=r.name, kind=r.kind, rrule=r.rrule,
                      template=r.template, group_id=gid, enabled=True,
                      last_delivered=when.timestamp())
               for r in DEFAULT_RITUALS]
    store.save_rituals(gid, rituals)
    return rituals


def _rituals(gid, sender="Ada"):
    return control_cgroup(f"rituals {gid}", context=None, chat=None,
                          sender=sender, sender_id=sender)


# ── /cgroup rituals add ────────────────────────────────────────────────


def test_add_daily_ritual_persists_and_lists(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    out = control_cgroup(
        f"rituals {gid} add Standup | 09:30 | Morning standup — {{date}}",
        context=None, chat=None, sender="Ada", sender_id="Ada")
    assert "Standup" in out, out
    assert "09:30" in out, out
    listed = _rituals(gid)
    assert "Standup" in listed, listed
    rituals = GroupStore().load_rituals(gid)
    added = [r for r in rituals if r.name == "Standup"]
    assert len(added) == 1
    assert added[0].kind == "custom"
    assert added[0].rrule == "FREQ=DAILY;BYHOUR=9;BYMINUTE=30"
    assert added[0].template == "Morning standup — {date}"
    assert added[0].group_id == gid
    assert added[0].enabled is True


def test_add_weekly_ritual_fires_on_tick(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    _seed_last_delivered(gid, FRIDAY_1600)
    out = control_cgroup(
        f"rituals {gid} add Friday demo | FR 16:00 | Demo time — {{date}}",
        context=None, chat=None, sender="Ada", sender_id="Ada")
    assert "Friday demo" in out, out
    assert "FR" in out or "Friday" in out, out
    result = dmod.run_group_ritual_tick_once(store=GroupStore(),
                                             now=FRIDAY_1600)
    assert result["ok"] is True
    assert [d["ritual"] for d in result["delivered"]] == ["Friday demo"], \
        result
    feed = GroupStore().feed(gid, limit=5)
    assert feed and feed[0].author == "Devon"
    assert "Demo time" in feed[0].text
    # A second tick at the same instant: delivered once, never re-fired.
    result2 = dmod.run_group_ritual_tick_once(store=GroupStore(),
                                              now=FRIDAY_1600)
    assert result2["delivered"] == []
    assert len(GroupStore().feed(gid, limit=10)) == 1


def test_add_duplicate_name_refused(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    first = control_cgroup(
        f"rituals {gid} add Standup | 09:30 | Morning standup",
        context=None, chat=None, sender="Ada", sender_id="Ada")
    assert "Standup" in first
    second = control_cgroup(
        f"rituals {gid} add standup | 10:00 | Another standup",
        context=None, chat=None, sender="Ada", sender_id="Ada")
    assert "already" in second.lower(), second
    assert "Standup" in second, second
    rituals = GroupStore().load_rituals(gid)
    assert len([r for r in rituals if r.name.lower() == "standup"]) == 1


def test_add_bad_time_refused(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    out = control_cgroup(
        f"rituals {gid} add Standup | 25:00 | Morning standup",
        context=None, chat=None, sender="Ada", sender_id="Ada")
    assert "25:00" in out or "hour" in out.lower(), out
    assert len([r for r in GroupStore().load_rituals(gid)
                if r.name == "Standup"]) == 0


def test_add_bad_weekday_refused(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    out = control_cgroup(
        f"rituals {gid} add Standup | XX 09:30 | Morning standup",
        context=None, chat=None, sender="Ada", sender_id="Ada")
    assert "XX" in out or "weekday" in out.lower(), out
    assert len([r for r in GroupStore().load_rituals(gid)
                if r.name == "Standup"]) == 0


def test_add_nonadmin_refused(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    out = control_cgroup(
        f"rituals {gid} add Standup | 09:30 | Morning standup",
        context=None, chat=None, sender="Bob", sender_id="Bob")
    assert "only" in out.lower() and "admin" in out.lower(), out
    assert len([r for r in GroupStore().load_rituals(gid)
                if r.name == "Standup"]) == 0


def test_add_nonmember_refused(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    out = control_cgroup(
        f"rituals {gid} add Standup | 09:30 | Morning standup",
        context=None, chat=None, sender="Carol", sender_id="Carol")
    assert "join" in out.lower(), out
    assert len([r for r in GroupStore().load_rituals(gid)
                if r.name == "Standup"]) == 0


def test_add_missing_parts_refused(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    out = control_cgroup(f"rituals {gid} add Standup",
                         context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    assert "Usage" in out, out
    assert len([r for r in GroupStore().load_rituals(gid)
                if r.name == "Standup"]) == 0


def test_add_empty_message_refused(tmp_path, monkeypatch):
    gid = _setup_group(tmp_path, monkeypatch)
    out = control_cgroup(f"rituals {gid} add Standup | 09:30 | ",
                         context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    assert "message" in out.lower() or "Usage" in out, out
    assert len([r for r in GroupStore().load_rituals(gid)
                if r.name == "Standup"]) == 0


# ── /map fence edit ────────────────────────────────────────────────────


class _FakeAdapter:
    def send(self, chat, text, **kw):
        return SimpleNamespace(ok=True)

    def send_location(self, chat, latitude, longitude, **kw):
        return SimpleNamespace(ok=True)


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters

    def send_file(self, platform, chat, path, **kw):  # pragma: no cover
        return SimpleNamespace(ok=True)


def _ctx(tmp: str):
    from nomorals.agents.context import build_context
    from nomorals.core.config import load_settings

    return build_context(
        load_settings(overrides={
            "home": tmp,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        }),
        with_executor=False, with_tools=True, with_router=False)


def _msg(key="telegram:1"):
    return SimpleNamespace(chat=SimpleNamespace(key=key), text="",
                           meta={}, sender_id="")


class _FenceRig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gapfix-chatverbs-")
        self.home = Path(self.tmp.name)
        (self.home / "packs").mkdir(parents=True)
        self.adapter = _FakeAdapter()
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = _ctx(str(self.home))
        rt._owner_chats = {"telegram:1"}
        rt._owner_sender_ids = set()
        rt.gateway = _Gateway({"telegram": self.adapter})
        self.rt = rt
        self.env = mock.patch.dict(os.environ, {
            "NM_MAPS_ENABLED": "1",
            "NM_MAPS_PACKS_DIR": str(self.home / "packs"),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _add_home(self, radius: str = "150"):
        out = self.rt._control_map(
            f"fence add circle 6.52441,3.3792 {radius} Home",
            message=_msg())
        assert "saved" in out, out
        return out

    def _store(self):
        from nomorals.location.fences import FenceStore

        return FenceStore(self.home / "location")

    def _fence(self, name="Home"):
        fences = self._store().fences()
        found = [f for f in fences if f.name == name]
        assert len(found) == 1, fences
        return found[0]


class TestFenceEdit(_FenceRig):
    def test_edit_radius(self):
        self._add_home()
        out = self.rt._control_map("fence edit Home radius 300",
                                   message=_msg())
        assert "300" in out, out
        assert self._fence().radius_m == 300.0
        listed = self.rt._control_map("fence list", message=_msg())
        assert "300 m" in listed, listed

    def test_edit_center(self):
        self._add_home()
        out = self.rt._control_map(
            "fence edit Home center 6.53000,3.3900", message=_msg())
        assert "6.5300" in out and "3.3900" in out, out
        fence = self._fence()
        assert fence.center is not None
        assert fence.center.lat == 6.53
        assert fence.center.lon == 3.39
        listed = self.rt._control_map("fence list", message=_msg())
        assert "6.5300,3.3900" in listed, listed

    def test_edit_name(self):
        self._add_home()
        before = self._fence().id
        out = self.rt._control_map("fence edit Home name HQ",
                                   message=_msg())
        assert "HQ" in out, out
        listed = self.rt._control_map("fence list", message=_msg())
        assert "HQ" in listed, listed
        assert "Home" not in listed, listed
        # A rename keeps the fence (and its reminders): same id.
        assert self._fence("HQ").id == before

    def test_edit_polygon_points(self):
        out = self.rt._control_map(
            "fence add polygon 6.52441,3.3792;6.52441,3.3802;"
            "6.52541,3.3802;6.52541,3.3792 Market",
            message=_msg())
        assert "saved" in out, out
        out = self.rt._control_map(
            "fence edit Market points 6.52441,3.3792;6.52441,3.3812;"
            "6.52641,3.3812",
            message=_msg())
        assert "3 vertices" in out, out
        fence = self._fence("Market")
        assert len(fence.vertices) == 3
        listed = self.rt._control_map("fence list", message=_msg())
        assert "polygon, 3 vertices" in listed, listed

    def test_edit_locale(self):
        self._add_home()
        out = self.rt._control_map("fence edit Home locale yo",
                                   message=_msg())
        assert "yo" in out, out
        assert self._fence().locale == "yo"

    def test_edit_unknown_fence_names_real_ones(self):
        self._add_home()
        out = self.rt._control_map("fence edit Nowhere radius 10",
                                   message=_msg())
        assert "no fence" in out.lower(), out
        assert "Home" in out, out
        assert self._fence().radius_m == 150.0

    def test_edit_wrong_kind_field_refused(self):
        self.rt._control_map(
            "fence add polygon 6.52441,3.3792;6.52441,3.3802;"
            "6.52541,3.3802;6.52541,3.3792 Market",
            message=_msg())
        out = self.rt._control_map("fence edit Market radius 300",
                                   message=_msg())
        assert "circle" in out.lower(), out
        assert len(self._fence("Market").vertices) == 4
        self._add_home()
        out = self.rt._control_map(
            "fence edit Home points 6.52441,3.3792;6.52441,3.3802;"
            "6.52541,3.3802",
            message=_msg())
        assert "polygon" in out.lower(), out
        assert self._fence().kind == "circle"

    def test_edit_bad_radius_refused(self):
        self._add_home()
        for bad in ("0", "nope"):
            out = self.rt._control_map(f"fence edit Home radius {bad}",
                                       message=_msg())
            assert "saved" not in out and "updated" not in out, out
            assert self._fence().radius_m == 150.0

    def test_edit_bad_center_refused(self):
        self._add_home()
        out = self.rt._control_map("fence edit Home center 91,3.3792",
                                   message=_msg())
        assert "updated" not in out, out
        assert self._fence().center == HOME

    def test_edit_unknown_field_refused(self):
        self._add_home()
        out = self.rt._control_map("fence edit Home colour red",
                                   message=_msg())
        assert "unknown" in out.lower(), out
        assert "radius" in out and "center" in out, out
        assert self._fence().radius_m == 150.0

    def test_edit_geometry_resets_live_state(self):
        from nomorals.location.fences import evaluate_fence_update
        from nomorals.location.privacy import ConsentStore

        self._add_home()
        store = self._store()
        result = evaluate_fence_update(
            store, ConsentStore(self.home / "location"),
            person_id="owner", loc=INSIDE)
        assert result["evaluated"] is True, result
        _, states_before, _ = store.load_all()
        assert states_before[self._fence().id].inside is True
        # Move the fence far away: the old inside/outside baseline
        # must not survive the move (no spurious exit on next update).
        out = self.rt._control_map(
            "fence edit Home center 6.60000,3.4000", message=_msg())
        assert "updated" in out, out
        _, states_after, _ = self._store().load_all()
        assert self._fence().id not in states_after

    def test_edit_stranger_refused_owner_only(self):
        self._add_home()
        out = self.rt._control_map(
            "fence edit Home radius 300", message=_msg("telegram:999"))
        assert "owner-only" in out, out
        assert self._fence().radius_m == 150.0

    def test_edit_then_remove_round_trip(self):
        self._add_home()
        out = self.rt._control_map("fence edit Home radius 300",
                                   message=_msg())
        assert "updated" in out, out
        out = self.rt._control_map("fence remove Home", message=_msg())
        assert "removed" in out, out
        assert "Home" not in self.rt._control_map("fence list",
                                                  message=_msg())
