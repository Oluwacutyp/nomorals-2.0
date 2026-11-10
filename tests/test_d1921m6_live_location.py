"""DOCS 19+21 M6 — the LiveLocationBroker on ConsentStore.

Pins:
* a session exists ONLY on a live "live-share" consent grant
  (no grant → the store's typed refusal; another scope's grant
  does not substitute);
* supplied updates forward to exactly the session recipients
  while the share is live — at the session's precision, never
  finer — and every evaluation (forwarded or withheld) leaves a
  durable decision record with its reason;
* expiry stops forwarding on its own clock (injected), a consent
  revoke lands on the very next update, a stale fix is withheld
  as stale, and multiple sessions evaluate independently;
* terminal states are sticky across broker re-instantiation
  (restart): ended/revoked/expired never resurrect;
* the never-track pin: the broker module imports no network /
  process / threading machinery, has no sleep, no polling loop;
* the /map share verbs (start card + status + stop) over a fake
  gateway, the NM_MAPS_LIVE_SHARE second gate, the runtime's
  supplied-update hook (owner-only, settings-gated), and the
  Telegram adapter surfacing a supplied location into meta.
"""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.location.live import (
    SCOPE_LIVE_SHARE,
    LiveLocationBroker,
    LiveLocationError,
    coarsen,
)
from nomorals.location.primitives import LatLon
from nomorals.location.privacy import ConsentError, ConsentStore

LAGOS = LatLon(6.52441, 3.37921)


class _Clock:
    def __init__(self, now: float = 1_700_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Sender:
    def __init__(self, fail_for: tuple[str, ...] = ()):
        self.calls: list[tuple[str, LatLon]] = []
        self.fail_for = set(fail_for)

    def __call__(self, chat_key: str, loc: LatLon) -> bool:
        self.calls.append((chat_key, loc))
        return chat_key not in self.fail_for


class BrokerRig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m6-live-")
        self.root = Path(self.tmp.name)
        self.clock = _Clock()
        self.sender = _Sender()
        self.consent = ConsentStore(self.root)
        self.broker = LiveLocationBroker(
            self.root, consent=self.consent, clock=self.clock,
            sender=self.sender)

    def tearDown(self):
        self.tmp.cleanup()

    def grant(self, subject: str = "owner") -> None:
        self.consent.grant(subject, SCOPE_LIVE_SHARE)

    def start(self, recipients=("whatsapp:234801",), **kw):
        self.grant()
        return self.broker.start_session(
            "owner", recipients, ttl_s=kw.pop("ttl_s", 1800.0), **kw)


class TestConsentGate(BrokerRig):
    def test_no_grant_no_session_typed_refusal(self):
        with self.assertRaises(ConsentError) as caught:
            self.broker.start_session(
                "owner", ("whatsapp:234801",), ttl_s=600.0)
        self.assertIn("live-share", str(caught.exception))
        self.assertEqual(self.broker.sessions("owner"), [])

    def test_other_scope_grant_does_not_substitute(self):
        self.consent.grant("owner", "geofence")
        with self.assertRaises(ConsentError):
            self.broker.start_session(
                "owner", ("whatsapp:234801",), ttl_s=600.0)

    def test_bad_shapes_refuse_typed(self):
        self.grant()
        for kw in ({"recipients": (), "ttl_s": 60.0},
                   {"recipients": ("telegram:9",), "ttl_s": 0.0},
                   {"recipients": ("telegram:9",), "ttl_s": 90_000.0},
                   {"recipients": ("telegram:9",), "ttl_s": 60.0,
                    "precision": "house"}):
            recips = kw.pop("recipients")
            with self.assertRaises(LiveLocationError, msg=str(kw)):
                self.broker.start_session("owner", recips, **kw)


class TestForwarding(BrokerRig):
    def test_forward_while_live_exact_recipients(self):
        session = self.start(("whatsapp:234801", "telegram:55"))
        decisions = self.broker.evaluate("owner", LAGOS)
        self.assertEqual(len(decisions), 1)
        rec = decisions[0]
        self.assertEqual(rec["decision"], "forwarded")
        self.assertEqual(rec["reason"], "grant-live")
        self.assertEqual(rec["session_id"], session.id)
        self.assertEqual(
            sorted(k for k, _ in self.sender.calls),
            ["telegram:55", "whatsapp:234801"])
        # the record is durable — the broker's own read-back has it
        stored = self.broker.recent_decisions(subject="owner")
        self.assertEqual(stored[-1]["reason"], "grant-live")

    def test_precision_coarsens_payload_and_record(self):
        self.start(precision="neighborhood")
        decisions = self.broker.evaluate("owner", LAGOS)
        sent_loc = self.sender.calls[0][1]
        self.assertEqual(sent_loc, LatLon(6.52, 3.38))
        self.assertEqual(decisions[0]["lat"], 6.52)
        self.assertEqual(decisions[0]["lon"], 3.38)
        self.assertEqual(coarsen(LAGOS, "city"), LatLon(6.5, 3.4))
        self.assertEqual(coarsen(LAGOS, "exact"), LAGOS)

    def test_no_session_records_withheld_without_coordinates(self):
        decisions = self.broker.evaluate("owner", LAGOS)
        self.assertEqual(len(decisions), 1)
        rec = decisions[0]
        self.assertEqual((rec["decision"], rec["reason"]),
                         ("withheld", "no-session"))
        self.assertIsNone(rec["lat"])
        self.assertIsNone(rec["lon"])
        self.assertEqual(self.sender.calls, [])

    def test_expiry_stops_forwarding_on_its_own(self):
        self.start(ttl_s=600.0)
        self.broker.evaluate("owner", LAGOS)
        self.assertEqual(len(self.sender.calls), 1)
        self.clock.advance(601.0)
        decisions = self.broker.evaluate("owner", LAGOS)
        self.assertEqual(decisions[0]["reason"], "expired")
        self.assertEqual(len(self.sender.calls), 1)  # no second send
        self.assertEqual(self.broker.sessions("owner"), [])
        states = [s.state for s in self.broker.sessions(
            "owner", include_terminal=True)]
        self.assertEqual(states, ["expired"])

    def test_consent_revoke_lands_on_next_update(self):
        self.start()
        self.broker.evaluate("owner", LAGOS)
        self.assertEqual(len(self.sender.calls), 1)
        self.consent.revoke("owner", SCOPE_LIVE_SHARE)
        decisions = self.broker.evaluate("owner", LAGOS)
        self.assertEqual(decisions[0]["reason"], "consent-denied")
        self.assertEqual(len(self.sender.calls), 1)

    def test_stale_fix_is_withheld_as_stale(self):
        self.start()
        stale_ts = self.clock.now - 901.0
        decisions = self.broker.evaluate("owner", LAGOS, ts=stale_ts)
        self.assertEqual(decisions[0]["reason"], "stale-update")
        self.assertGreater(decisions[0]["update_age_s"], 900.0)
        self.assertEqual(self.sender.calls, [])
        fresh = self.broker.evaluate(
            "owner", LAGOS, ts=self.clock.now - 60.0)
        self.assertEqual(fresh[0]["decision"], "forwarded")

    def test_send_failure_is_recorded_not_claimed(self):
        self.sender.fail_for.add("whatsapp:234801")
        self.start(("whatsapp:234801", "telegram:55"))
        decisions = self.broker.evaluate("owner", LAGOS)
        rec = decisions[0]
        self.assertEqual(rec["decision"], "forwarded")  # one landed
        self.assertEqual(rec["failed_recipients"],
                         ["whatsapp:234801"])
        only_bad = _Sender(fail_for=("telegram:55", "whatsapp:234801"))
        broker2 = LiveLocationBroker(
            self.root, consent=self.consent, clock=self.clock,
            sender=only_bad)
        broker2.evaluate("owner", LAGOS)
        rec2 = broker2.recent_decisions(subject="owner")[-1]
        self.assertEqual((rec2["decision"], rec2["reason"]),
                         ("withheld", "send-failed"))

    def test_sessions_evaluate_independently(self):
        self.start(("telegram:55",), ttl_s=1800.0)
        short = self.start(("whatsapp:234801",), ttl_s=300.0)
        self.assertNotEqual(short.id, "")
        self.clock.advance(301.0)
        decisions = self.broker.evaluate("owner", LAGOS)
        by_reason = {d["reason"] for d in decisions}
        self.assertEqual(by_reason, {"grant-live", "expired"})
        sent_to = [k for k, _ in self.sender.calls]
        self.assertEqual(sent_to, ["telegram:55"])


class TestStickyStop(BrokerRig):
    def test_stop_then_restart_stays_ended(self):
        session = self.start()
        self.broker.stop_session(session.id)
        revived = LiveLocationBroker(
            self.root, consent=ConsentStore(self.root),
            clock=self.clock, sender=self.sender)
        self.assertEqual(revived.sessions("owner"), [])
        states = [s.state for s in revived.sessions(
            "owner", include_terminal=True)]
        self.assertEqual(states, ["ended"])
        decisions = revived.evaluate("owner", LAGOS)
        self.assertEqual(decisions[0]["reason"], "ended")
        self.assertEqual(self.sender.calls, [])

    def test_revoke_subject_then_restart_stays_revoked(self):
        self.start()
        self.start(("telegram:55",))
        flipped = self.broker.revoke_subject("owner")
        self.assertEqual(len(flipped), 2)
        revived = LiveLocationBroker(
            self.root, consent=ConsentStore(self.root),
            clock=self.clock, sender=self.sender)
        decisions = revived.evaluate("owner", LAGOS)
        self.assertEqual({d["reason"] for d in decisions},
                         {"revoked"})
        self.assertEqual(self.sender.calls, [])

    def test_expired_never_resurrects_and_start_is_fresh(self):
        first = self.start(ttl_s=60.0)
        self.clock.advance(120.0)
        revived = LiveLocationBroker(
            self.root, consent=ConsentStore(self.root),
            clock=self.clock, sender=self.sender)
        second = revived.start_session(
            "owner", ("whatsapp:234801",), ttl_s=60.0)
        self.assertNotEqual(first.id, second.id)
        states = sorted(s.state for s in revived.sessions(
            "owner", include_terminal=True))
        self.assertEqual(states, ["active", "expired"])


class TestNeverTrackPin(unittest.TestCase):
    def test_broker_module_has_no_tracking_machinery(self):
        path = (Path(__file__).resolve().parent.parent
                / "nomorals" / "location" / "live.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = {"socket", "urllib", "requests", "http.client",
                  "http", "subprocess", "threading",
                  "multiprocessing", "asyncio", "ctypes"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        offenders = [m for m in imported
                     if any(m == b or m.startswith(b + ".")
                            for b in banned)]
        self.assertEqual(offenders, [], f"live.py: {offenders}")
        # no sleep (a poll loop's heartbeat) and no open-ended loop
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (func.attr if isinstance(func, ast.Attribute)
                        else getattr(func, "id", ""))
                self.assertNotEqual(name, "sleep")
            if isinstance(node, ast.While):
                self.assertNotIsInstance(node.test, ast.Constant)


# ── the /map share surface ────────────────────────────────────────


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
    def __init__(self):
        self.texts: list[tuple[str, str]] = []
        self.pins: list[tuple[str, float, float]] = []

    def send(self, chat, text, **kw):
        self.texts.append((chat.key, text))
        return SimpleNamespace(ok=True)

    def send_location(self, chat, latitude, longitude, **kw):
        self.pins.append((chat.key, latitude, longitude))
        return SimpleNamespace(ok=True)


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent: list[dict] = []

    def send_file(self, platform, chat, path, **kw):  # pragma: no cover
        return SimpleNamespace(ok=True)


class TestShareSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m6-share-")
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
            "NM_MAPS_LIVE_SHARE": "1",
            "NM_MAPS_PACKS_DIR": str(self.home / "packs"),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_start_grants_starts_and_cards_the_recipient(self):
        out = self.rt._control_map(
            "share telegram:55 30 neighborhood", message=_msg())
        self.assertIn("live share started", out)
        self.assertIn("neighborhood", out)
        # the grant the session rides is on record
        consent = ConsentStore(self.home / "location")
        self.assertTrue(consent.has_consent("owner", SCOPE_LIVE_SHARE))
        # exactly one card, to the recipient, naming precision + TTL
        self.assertEqual(len(self.adapter.texts), 1)
        card_key, card = self.adapter.texts[0]
        self.assertEqual(card_key, "telegram:55")
        self.assertIn("neighborhood", card)
        self.assertIn("30 min", card)
        # no session id leaks into the owner reply
        broker = self.rt._live_broker()
        sid = broker.sessions("owner")[0].id
        self.assertNotIn(sid, out)

    def test_supplied_update_forwards_coarsened_pin(self):
        self.rt._control_map("share telegram:55 30 neighborhood",
                             message=_msg())
        broker = self.rt._live_broker()
        decisions = broker.evaluate("owner", LAGOS)
        self.assertEqual(decisions[0]["decision"], "forwarded")
        self.assertEqual(self.adapter.pins,
                         [("telegram:55", 6.52, 3.38)])

    def test_status_and_stop_round_trip(self):
        self.rt._control_map("share telegram:55 30", message=_msg())
        status = self.rt._control_map("share status", message=_msg())
        self.assertIn("telegram:55", status)
        self.assertIn("left", status)
        out = self.rt._control_map("share stop", message=_msg())
        self.assertIn("stopped", out)
        status = self.rt._control_map("share status", message=_msg())
        self.assertIn("no live share is running", status)
        # and the ended share withholds from here on
        broker = self.rt._live_broker()
        decisions = broker.evaluate("owner", LAGOS)
        self.assertEqual(decisions[0]["reason"], "ended")

    def test_share_gate_second_flag(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_MAPS_LIVE_SHARE", None)
            out = self.rt._control_map("share telegram:55 30",
                                       message=_msg())
        self.assertIn("NM_MAPS_LIVE_SHARE=1", out)
        self.assertEqual(self.adapter.texts, [])

    def test_stranger_refused_before_any_grant(self):
        out = self.rt._control_map(
            "share telegram:55 30", message=_msg("telegram:999"))
        self.assertIn("owner-only", out)
        consent = ConsentStore(self.home / "location")
        self.assertFalse(consent.has_consent("owner",
                                             SCOPE_LIVE_SHARE))


class TestSuppliedUpdateHook(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m6-hook-")
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
            "NM_MAPS_LIVE_SHARE": "1",
            "NM_MAPS_PACKS_DIR": str(self.home / "packs"),
        })
        self.env.start()
        self.rt._control_map("share telegram:55 30", message=_msg())

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _location_message(self, *, owner: bool):
        import time as _time

        from nomorals.social.chat.base import ChatMessage, ChatRef

        meta = {"location": {"lat": 6.52441, "lon": 3.37921,
                             "live": True}}
        if owner:
            meta["is_owner"] = True
        return ChatMessage(
            chat=ChatRef(platform="telegram",
                         chat_id="1" if owner else "999"),
            incoming=True, meta=meta, ts=_time.time())

    def test_owner_supplied_location_reaches_recipient(self):
        self.rt._maybe_live_share_update(
            self._location_message(owner=True))
        self.assertEqual(self.adapter.pins,
                         [("telegram:55", 6.52441, 3.37921)])

    def test_stranger_location_is_never_brokered(self):
        self.rt._maybe_live_share_update(
            self._location_message(owner=False))
        self.assertEqual(self.adapter.pins, [])
        broker = self.rt._live_broker()
        self.assertEqual(broker.recent_decisions(subject="owner"),
                         [])

    def test_hook_dark_when_share_flag_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_MAPS_LIVE_SHARE", None)
            self.rt._maybe_live_share_update(
                self._location_message(owner=True))
        self.assertEqual(self.adapter.pins, [])


class TestTelegramSuppliedLocation(unittest.TestCase):
    def _adapter(self):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        return TelegramBotAdapter(token="123456:fixture-token")

    def _update(self, extra: dict) -> dict:
        msg = {"message_id": 7, "from": {"id": 42},
               "chat": {"id": 42, "type": "private"}}
        msg.update(extra)
        return {"update_id": 3, "message": msg}

    def test_location_share_surfaces_in_meta(self):
        msg = self._adapter()._convert(self._update({
            "location": {"latitude": 6.5244, "longitude": 3.3792,
                         "live_period": 3600}}))
        self.assertIsNotNone(msg)
        self.assertEqual(msg.meta["location"]["lat"], 6.5244)
        self.assertEqual(msg.meta["location"]["lon"], 3.3792)
        self.assertTrue(msg.meta["location"]["live"])

    def test_venue_location_surfaces_static(self):
        msg = self._adapter()._convert(self._update({
            "venue": {"location": {"latitude": 7.6211,
                                   "longitude": 5.2214},
                      "title": "Ilawe"}}))
        self.assertIsNotNone(msg)
        self.assertEqual(msg.meta["location"]["lat"], 7.6211)
        self.assertFalse(msg.meta["location"]["live"])

    def test_plain_text_carries_no_location(self):
        msg = self._adapter()._convert(self._update({"text": "hi"}))
        self.assertIsNotNone(msg)
        self.assertNotIn("location", msg.meta)


if __name__ == "__main__":
    unittest.main()
