"""Gapfix R1-45 — location geofence / reminders / triggers wired live.

Pins (failing-first: ``nomorals.location.fences`` does not exist yet,
``/map fence`` is an unknown verb):
* FenceStore: add/list/remove round-trips durably; corrupt file
  degrades to empty with a named reason; bad rows are skipped.
* evaluate_fence_update: no consent -> nothing evaluated, no state
  stored; consent + outside baseline -> inside fires exactly one
  enter; a second inside update fires nothing (stateful); leaving
  fires exactly one exit.
* Reminders: fire-once (no refire on the next update), re-armable,
  enter vs exit direction, no consent -> never fire.
* The runtime hook: owner-only (a stranger's location never touches
  the owner's fences), Atlas-gated, never raises, alerts go out
  through the notifier (never a reply in the room).
* /map fence add|list|remove|remind|reminders|unremind|rearm: circle
  + polygon, bad geometry refused with the reason named, the add
  command IS the geofence opt-in grant (like /map share), stranger
  refused owner-only.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.location.geofence import circle_fence
from nomorals.location.primitives import LatLon
from nomorals.location.privacy import ConsentStore

HOME = LatLon(6.52441, 3.3792)      # Lagos centre (fence centre)
OUTSIDE = LatLon(6.60000, 3.4000)   # ~8 km away, outside any test fence
INSIDE = LatLon(6.52450, 3.37930)   # ~10 m from centre


# ── FenceStore ────────────────────────────────────────────────────


class TestFenceStore:
    def test_add_list_remove_round_trip_durable(self, tmp_path):
        from nomorals.location.fences import FenceStore

        store = FenceStore(tmp_path)
        fence = circle_fence("f-home", "Home", HOME, 150.0)
        store.add_fence(fence)
        assert [f.id for f in store.fences()] == ["f-home"]
        assert store.get_fence("f-home").name == "Home"
        # a second instance over the same root sees it (durable)
        assert [f.id for f in FenceStore(tmp_path).fences()] == ["f-home"]
        assert store.remove_fence("f-home") is True
        assert store.fences() == []
        assert FenceStore(tmp_path).fences() == []
        assert store.remove_fence("f-home") is False

    def test_corrupt_file_degrades_to_empty_with_a_reason(self, tmp_path):
        from nomorals.location.fences import FenceStore

        (tmp_path / "fences.json").write_text("{not json",
                                              encoding="utf-8")
        store = FenceStore(tmp_path)
        assert store.fences() == []
        assert store.degraded_reason() != ""
        # the corrupt file is quarantined, not left to poison reads
        assert not (tmp_path / "fences.json").exists()
        assert list(tmp_path.glob("fences.json.corrupt-*"))

    def test_bad_rows_are_skipped_not_fatal(self, tmp_path):
        from nomorals.location.fences import FenceStore

        (tmp_path / "fences.json").write_text(json.dumps({
            "fences": {
                "f-ok": circle_fence("f-ok", "Ok", HOME, 100.0).to_dict(),
                "f-bad": {"id": "f-bad", "name": "Bad", "kind": "circle",
                          "center": None, "radius_m": -5},
                "f-junk": "not a dict",
            },
            "states": {},
            "reminders": {},
        }), encoding="utf-8")
        store = FenceStore(tmp_path)
        assert [f.id for f in store.fences()] == ["f-ok"]

    def test_resolve_by_id_prefix_or_label(self, tmp_path):
        from nomorals.location.fences import FenceStore

        store = FenceStore(tmp_path)
        store.add_fence(circle_fence("f-abcdef123456", "Home", HOME, 100.0))
        store.add_fence(circle_fence("f-abcdef999999", "Work", HOME, 100.0))
        assert store.resolve_fence("f-abcdef123456").name == "Home"
        assert store.resolve_fence("f-abcdef999999").name == "Work"
        # ambiguous prefix resolves to nothing (honest, not first-match)
        assert store.resolve_fence("f-abcdef") is None
        assert store.resolve_fence("Home").name == "Home"
        assert store.resolve_fence("Nope") is None


# ── pure evaluation ───────────────────────────────────────────────


def _rig(tmp_path):
    from nomorals.location.fences import FenceStore

    store = FenceStore(tmp_path)
    consent = ConsentStore(tmp_path)
    return store, consent


def _grant(consent, scope="geofence"):
    consent.grant("owner", scope)


class TestEvaluateFenceUpdate:
    def test_no_consent_evaluates_nothing_and_stores_nothing(self,
                                                              tmp_path):
        from nomorals.location.fences import evaluate_fence_update

        store, consent = _rig(tmp_path)
        store.add_fence(circle_fence("f-home", "Home", HOME, 150.0))
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert out["evaluated"] is False
        assert out["events"] == []
        assert out["fired"] == []
        # no state was stored: a later consented baseline still starts
        # from "no baseline", it does not inherit a silent sighting
        _, states, _ = store.load_all()
        assert states == {}

    def test_enter_fires_once_then_stateful_silence(self, tmp_path):
        from nomorals.location.fences import evaluate_fence_update

        store, consent = _rig(tmp_path)
        _grant(consent)
        store.add_fence(circle_fence("f-home", "Home", HOME, 150.0))
        # baseline: outside first — no event (arriving is not entering
        # until we already knew you were outside)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=OUTSIDE)
        assert out["events"] == []
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert len(out["events"]) == 1
        assert out["events"][0]["event"] == "entered"
        assert out["events"][0]["fence_id"] == "f-home"
        # second inside update: no second enter (stateful)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert out["events"] == []
        # and the state survived a fresh store instance (durable)
        _, states, _ = store.load_all()
        assert states["f-home"].inside is True

    def test_exit_fires_once_on_leaving(self, tmp_path):
        from nomorals.location.fences import evaluate_fence_update

        store, consent = _rig(tmp_path)
        _grant(consent)
        store.add_fence(circle_fence("f-home", "Home", HOME, 150.0))
        evaluate_fence_update(store, consent, person_id="owner",
                              loc=OUTSIDE)
        evaluate_fence_update(store, consent, person_id="owner",
                              loc=INSIDE)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=OUTSIDE)
        assert len(out["events"]) == 1
        assert out["events"][0]["event"] == "exited"
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=OUTSIDE)
        assert out["events"] == []

    def test_first_sighting_sets_baseline_without_event(self, tmp_path):
        from nomorals.location.fences import evaluate_fence_update

        store, consent = _rig(tmp_path)
        _grant(consent)
        store.add_fence(circle_fence("f-home", "Home", HOME, 150.0))
        # already inside on first sighting: no enter (no baseline yet)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert out["events"] == []
        # leaving now fires the exit (the baseline was inside)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=OUTSIDE)
        assert [e["event"] for e in out["events"]] == ["exited"]

    def test_reminder_fires_once_then_rearms(self, tmp_path):
        from nomorals.location.fences import evaluate_fence_update
        from nomorals.location.reminders import LocationReminder

        store, consent = _rig(tmp_path)
        _grant(consent)
        _grant(consent, "reminders")
        store.add_fence(circle_fence("f-home", "Home", HOME, 150.0))
        store.add_reminder(LocationReminder(
            id="r-1", person_id="owner", fence_id="f-home",
            message="buy milk", fire_on="enter"))
        evaluate_fence_update(store, consent, person_id="owner",
                              loc=OUTSIDE)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert [r.id for r in out["fired"]] == ["r-1"]
        # a second enter/exit cycle does NOT refire (fire-once)
        evaluate_fence_update(store, consent, person_id="owner",
                              loc=OUTSIDE)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert out["fired"] == []
        # re-arm, then the next enter fires again
        assert store.rearm_reminder("r-1") is True
        evaluate_fence_update(store, consent, person_id="owner",
                              loc=OUTSIDE)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert [r.id for r in out["fired"]] == ["r-1"]

    def test_reminder_needs_its_own_consent(self, tmp_path):
        from nomorals.location.fences import evaluate_fence_update
        from nomorals.location.reminders import LocationReminder

        store, consent = _rig(tmp_path)
        _grant(consent)  # geofence only — no "reminders" grant
        store.add_fence(circle_fence("f-home", "Home", HOME, 150.0))
        store.add_reminder(LocationReminder(
            id="r-1", person_id="owner", fence_id="f-home",
            message="buy milk", fire_on="enter"))
        evaluate_fence_update(store, consent, person_id="owner",
                              loc=OUTSIDE)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert len(out["events"]) == 1  # the fence event still fires
        assert out["fired"] == []       # the reminder does not

    def test_exit_reminder_matches_exit_only(self, tmp_path):
        from nomorals.location.fences import evaluate_fence_update
        from nomorals.location.reminders import LocationReminder

        store, consent = _rig(tmp_path)
        _grant(consent)
        _grant(consent, "reminders")
        store.add_fence(circle_fence("f-home", "Home", HOME, 150.0))
        store.add_reminder(LocationReminder(
            id="r-1", person_id="owner", fence_id="f-home",
            message="lock up", fire_on="exit"))
        evaluate_fence_update(store, consent, person_id="owner",
                              loc=OUTSIDE)
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=INSIDE)
        assert out["fired"] == []
        out = evaluate_fence_update(store, consent, person_id="owner",
                                    loc=OUTSIDE)
        assert [r.id for r in out["fired"]] == ["r-1"]


# ── runtime hook + /map fence surface (M6 harness shape) ──────────


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


def _location_message(*, owner: bool, loc: LatLon):
    import time as _time

    from nomorals.social.chat.base import ChatMessage, ChatRef

    meta = {"location": {"lat": loc.lat, "lon": loc.lon, "live": True}}
    if owner:
        meta["is_owner"] = True
    return ChatMessage(
        chat=ChatRef(platform="telegram",
                     chat_id="1" if owner else "999"),
        incoming=True, meta=meta, ts=_time.time())


class _FenceRig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gapfix18-fence-")
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


class TestFenceHook(_FenceRig):
    def test_owner_enter_alerts_once_via_notifier(self):
        self._add_home()
        outside = _location_message(owner=True, loc=OUTSIDE)
        inside = _location_message(owner=True, loc=INSIDE)
        with mock.patch("nomorals.agents.notifier.notify") as sent:
            self.rt._maybe_fence_update(outside)
            self.assertEqual(sent.call_count, 0)  # baseline, no event
            self.rt._maybe_fence_update(inside)
            self.assertEqual(sent.call_count, 1)
            args = sent.call_args[0]
            self.assertEqual(args[1], "geofence")
            self.assertIn("entered", args[2].lower())
            self.assertIn("Home", args[2])
            # second inside update: no second alert (stateful)
            self.rt._maybe_fence_update(inside)
            self.assertEqual(sent.call_count, 1)

    def test_owner_exit_alerts(self):
        self._add_home()
        with mock.patch("nomorals.agents.notifier.notify") as sent:
            self.rt._maybe_fence_update(
                _location_message(owner=True, loc=OUTSIDE))
            self.rt._maybe_fence_update(
                _location_message(owner=True, loc=INSIDE))
            self.rt._maybe_fence_update(
                _location_message(owner=True, loc=OUTSIDE))
            self.assertEqual(sent.call_count, 2)
            self.assertIn("exited",
                          sent.call_args[0][2].lower())

    def test_stranger_location_never_touches_owner_fences(self):
        self._add_home()
        with mock.patch("nomorals.agents.notifier.notify") as sent:
            self.rt._maybe_fence_update(
                _location_message(owner=False, loc=INSIDE))
            self.rt._maybe_fence_update(
                _location_message(owner=False, loc=OUTSIDE))
            self.assertEqual(sent.call_count, 0)
        # and no state was stored for the stranger's fixes
        from nomorals.location.fences import FenceStore

        _, states, _ = FenceStore(self.home / "location").load_all()
        assert states == {}

    def test_hook_dark_when_atlas_off(self):
        self._add_home()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_MAPS_ENABLED", None)
            with mock.patch("nomorals.agents.notifier.notify") as sent:
                self.rt._maybe_fence_update(
                    _location_message(owner=True, loc=INSIDE))
                self.assertEqual(sent.call_count, 0)

    def test_hook_never_raises_on_a_corrupt_store(self):
        self._add_home()
        (self.home / "location" / "fences.json").write_text(
            "{broken", encoding="utf-8")
        # must not raise — a corrupt store never eats chat
        self.rt._maybe_fence_update(
            _location_message(owner=True, loc=INSIDE))

    def test_fired_reminder_alert_carries_its_message(self):
        self.rt._control_map(
            "fence add circle 6.52441,3.3792 150 Home", message=_msg())
        out = self.rt._control_map(
            "fence remind Home enter buy milk on the way in",
            message=_msg())
        assert "saved" in out, out
        with mock.patch("nomorals.agents.notifier.notify") as sent:
            self.rt._maybe_fence_update(
                _location_message(owner=True, loc=OUTSIDE))
            self.rt._maybe_fence_update(
                _location_message(owner=True, loc=INSIDE))
            bodies = [c[0][3] for c in sent.call_args_list]
            self.assertTrue(
                any("buy milk" in b for b in bodies),
                f"reminder message never delivered: {bodies}")


class TestMapFenceSurface(_FenceRig):
    def test_add_circle_grants_geofence_consent(self):
        out = self._add_home()
        assert "Home" in out
        assert "150" in out
        consent = ConsentStore(self.home / "location")
        assert consent.has_consent("owner", "geofence")

    def test_add_polygon_and_list_shows_state(self):
        out = self.rt._control_map(
            "fence add polygon 6.52441,3.3792;6.52441,3.3802;"
            "6.52541,3.3802;6.52541,3.3792 Market",
            message=_msg())
        assert "saved" in out, out
        listed = self.rt._control_map("fence list", message=_msg())
        assert "Market" in listed
        assert "polygon" in listed
        assert "unknown" in listed  # no baseline yet

    def test_add_bad_radius_refused_with_reason(self):
        out = self.rt._control_map(
            "fence add circle 6.52441,3.3792 0 Home", message=_msg())
        assert "saved" not in out
        assert "radius" in out.lower()
        out = self.rt._control_map(
            "fence add circle 6.52441,3.3792 nope Home", message=_msg())
        assert "saved" not in out

    def test_add_bad_coords_refused(self):
        out = self.rt._control_map(
            "fence add circle 91,3.3792 150 Home", message=_msg())
        assert "saved" not in out
        out = self.rt._control_map(
            "fence add polygon 6.52441,3.3792;6.52541,3.3792 Yard",
            message=_msg())
        assert "saved" not in out  # only 2 vertices

    def test_remove_by_label(self):
        self._add_home()
        out = self.rt._control_map("fence remove Home", message=_msg())
        assert "removed" in out
        assert "Home" not in self.rt._control_map(
            "fence list", message=_msg())
        out = self.rt._control_map("fence remove Home", message=_msg())
        assert "no fence" in out

    def test_list_empty_names_the_add_verb(self):
        out = self.rt._control_map("fence list", message=_msg())
        assert "no fences" in out
        assert "fence add" in out

    def test_remind_verbs_round_trip(self):
        self._add_home()
        out = self.rt._control_map(
            "fence remind Home enter buy milk", message=_msg())
        assert "saved" in out, out
        consent = ConsentStore(self.home / "location")
        assert consent.has_consent("owner", "reminders")
        listed = self.rt._control_map("fence reminders", message=_msg())
        assert "buy milk" in listed
        assert "enter" in listed
        out = self.rt._control_map(
            "fence remind Home sideways nope", message=_msg())
        assert "saved" not in out
        out = self.rt._control_map(
            "fence remind Nowhere enter hello", message=_msg())
        assert "saved" not in out

    def test_stranger_refused_owner_only(self):
        out = self.rt._control_map(
            "fence add circle 6.52441,3.3792 150 Home",
            message=_msg("telegram:999"))
        assert "owner-only" in out
        consent = ConsentStore(self.home / "location")
        assert not consent.has_consent("owner", "geofence")

    def test_unknown_fence_verb_named(self):
        out = self.rt._control_map("fence explode", message=_msg())
        assert "unknown /map fence verb" in out
