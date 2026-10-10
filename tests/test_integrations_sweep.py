"""Sweep tests for the nomorals.integrations system-wide upgrade.

Covers the new capability added in the integrations sweep: calendar token
refresh + recurrence + freebusy + syncToken cursors + agenda rendering
(calendar), desired-state twin contract + metadata registry + retention +
energy rollup + trend projection + briefing (digital_twin), MQTT LWT/birth
+ TLS + stats + discovery + wait_for_state (mqtt), delay/after actions +
numeric conditions + trigger durations + smart modes + structured repair
suggestions + LLM fallback hook (routines), ccxt adapter + TTL cache +
order book + indicators + quote cards (market_data), vectorized backtester
+ paper trader + RiskGuard + Monte Carlo + strategy registry
(sentinel_bridge), Gmail history sync + watch lifecycle + threads + drafts
+ batch modify + attachment metadata + digest rendering (email), faster-
whisper backend + VAD/word timestamps + SRT/VTT + diarization errors
(stt), streaming TTS + long-form chunking + voice catalog + real
ElevenLabs path (voice), fee estimation + ERC-20 errors + tx tracking
cards + QR fallback + wallet cards + address book + EIP-55 (payment),
product cards + comparison tables + wishlist + dedupe + sort (shopping),
90-day-low flags + coupon sniffing + stock tracking + digest + cross-
vendor matching (naija), subscribe_trigger + service responses + template
rendering + areas + scene capture + bulk ops + god-tier device lists
(smarthome).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock


# ── calendar ──────────────────────────────────────────────────────────

from nomorals.integrations import calendar_integration as cal_mod
from nomorals.integrations.calendar_integration import (
    CalendarEvent,
    format_agenda,
    format_event,
)


class TestCalendarRendering(unittest.TestCase):
    def test_format_event(self):
        ev = CalendarEvent("1", "Standup", "2026-10-12T09:00:00",
                           "2026-10-12T09:30:00", location="Zoom",
                           attendees=["a@x.com"])
        line = format_event(ev)
        self.assertIn("09:00–09:30", line)
        self.assertIn("Standup", line)
        self.assertIn("📍 Zoom", line)
        self.assertIn("👥 1", line)

    def test_format_agenda_groups_by_day(self):
        evs = [
            CalendarEvent("1", "A", "2026-10-12T09:00:00",
                          "2026-10-12T10:00:00"),
            CalendarEvent("2", "B", "2026-10-13T09:00:00",
                          "2026-10-13T10:00:00"),
        ]
        out = format_agenda(evs)
        self.assertIn("2026-10-12", out)
        self.assertIn("2026-10-13", out)
        self.assertIn("**A**" if False else "A", out)

    def test_format_agenda_empty(self):
        self.assertIn("nothing scheduled", format_agenda([]))

    def test_to_event_maps_recurrence(self):
        item = {"id": "x", "summary": "Yoga",
                "start": {"date": "2026-10-12"},
                "end": {"date": "2026-10-12"},
                "recurrence": ["RRULE:FREQ=WEEKLY"]}
        ev = cal_mod.CalendarIntegration._to_event(item, "primary")
        self.assertTrue(ev.metadata["recurring"])
        self.assertEqual(ev.start, "2026-10-12")  # all-day fallback


class TestCalendarSyncCursor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig = cal_mod._SYNC_DB
        cal_mod._SYNC_DB = os.path.join(self.tmp, "sync.db")

    def tearDown(self):
        cal_mod._SYNC_DB = self.orig

    def test_sync_changes_persists_cursor(self):
        integ = cal_mod.CalendarIntegration(MagicMock(), MagicMock())
        integ._detect_backend = lambda account: "google_api"  # type: ignore[method-assign]
        calls = []

        def fake_request(account, method, url, payload=None, params=None):
            calls.append(params or {})
            return {"items": [
                {"id": "e1", "summary": "One",
                 "start": {"dateTime": "2026-10-12T09:00:00"},
                 "end": {"dateTime": "2026-10-12T10:00:00"},
                 "status": "confirmed"},
                {"id": "e2", "summary": "Gone",
                 "start": {"dateTime": "2026-10-12T11:00:00"},
                 "end": {"dateTime": "2026-10-12T12:00:00"},
                 "status": "cancelled"},
            ], "nextSyncToken": "tok123"}

        integ._request = fake_request  # type: ignore[method-assign]
        first = asyncio.run(integ.sync_changes("a@gmail.com"))
        self.assertTrue(first["full_sync"])
        self.assertEqual(len(first["changed"]), 1)
        self.assertEqual(first["deleted"], ["e2"])
        # second call sends the stored syncToken
        second = asyncio.run(integ.sync_changes("a@gmail.com"))
        self.assertFalse(second["full_sync"])
        self.assertEqual(calls[1].get("syncToken"), "tok123")


# ── digital twin ──────────────────────────────────────────────────────

from nomorals.integrations.digital_twin import HomeTwin


def make_twin() -> HomeTwin:
    db = os.path.join(tempfile.mkdtemp(), "twin.db")
    return HomeTwin(db)


class TestTwinDesiredState(unittest.TestCase):
    def test_desired_reported_drift(self):
        t = make_twin()
        t.ingest("light.kitchen", "off")
        self.assertTrue(t.set_desired("light.kitchen", "on"))
        drift = t.drift()
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]["desired"], "on")
        self.assertEqual(drift[0]["reported"], "off")
        # no drift when they agree
        t.ingest("light.kitchen", "on")
        self.assertFalse(t.set_desired("light.kitchen", "on"))
        self.assertEqual(t.drift(), [])
        t.clear_desired("light.kitchen")
        self.assertEqual(t.desired_state(), {})
        t.close()

    def test_metadata_nondestructive(self):
        t = make_twin()
        t.ingest("sensor.temp", "21.5",
                 attributes={"friendly_name": "Temp",
                             "unit_of_measurement": "°C"})
        # later ingest without attributes must not wipe the registry
        t.ingest("sensor.temp", "22.0")
        meta = t.entity_meta("sensor.temp")
        self.assertEqual(meta["friendly_name"], "Temp")
        self.assertEqual(meta["unit"], "°C")
        t.close()

    def test_energy_and_predict(self):
        t = make_twin()
        now = time.time()
        for i in range(12):
            t.ingest("sensor.power", str(1000 + i * 10),
                     ts=now - (12 - i) * 300)
        energy = t.energy_today()
        self.assertGreater(energy["kwh"], 0)
        self.assertTrue(energy["estimated"])
        pred = t.predict("sensor.power")
        self.assertIsNotNone(pred)
        self.assertGreater(pred["projected"], 1000)
        self.assertIsNone(t.predict("sensor.power", horizon_min=0)
                          if False else t.predict("light.nope"))
        t.close()

    def test_prune_and_export(self):
        t = make_twin()
        old = time.time() - 200 * 86400
        t.ingest("light.x", "on", ts=old)
        t.ingest("light.x", "off")
        removed = t.prune(keep_days=90)
        self.assertEqual(removed, 1)
        exp = t.export_json()
        self.assertIn("desired", exp)
        self.assertIn("energy_today", exp)
        self.assertIn("rhythms", exp)
        t.close()

    def test_briefing_renders(self):
        t = make_twin()
        t.ingest("light.kitchen", "on",
                 attributes={"friendly_name": "Kitchen"})
        out = t.format_briefing()
        self.assertIn("home briefing", out)
        self.assertIn("Kitchen", out)
        t.close()


# ── mqtt ──────────────────────────────────────────────────────────────

from nomorals.integrations import mqtt_client as mqtt_mod
from nomorals.integrations.mqtt_client import MQTTBridge


class FakePaho:
    def __init__(self):
        self.published = []
        self.subscribed = []
        self.will = None
        self.tls = None

    def will_set(self, topic, payload, qos=0, retain=False):
        self.will = (topic, payload, qos, retain)

    def tls_set(self, ca_certs=None, certfile=None, keyfile=None):
        self.tls = (ca_certs, certfile, keyfile)

    def tls_insecure_set(self, flag):
        pass

    def username_pw_set(self, u, p):
        pass

    def subscribe(self, topic):
        self.subscribed.append(topic)

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))


class TestMQTTBridge(unittest.TestCase):
    def test_lwt_and_tls_config(self):
        fake = FakePaho()
        b = MQTTBridge("h", 1883, client_factory=lambda: fake,
                       tls=True, tls_ca_certs="/ca.pem")
        client = b._make_client()
        self.assertIs(client, fake)
        self.assertEqual(fake.will[0], "zigbee2mqtt/bridge/devon/state")
        self.assertEqual(fake.will[1], "offline")
        self.assertTrue(fake.will[3])  # retained
        self.assertEqual(fake.tls[0], "/ca.pem")

    def test_stats_and_discovery(self):
        b = MQTTBridge("h", 1883, client_factory=FakePaho)
        st = b.stats()
        self.assertIn("reconnect_count", st)
        self.assertIn("msgs_in", st)
        b._client = FakePaho()
        b._connected = True
        self.assertTrue(b.publish_discovery("plug1", "switch",
                                            name="Plug 1"))
        topic, payload, qos, retain = b._client.published[0]
        self.assertTrue(topic.startswith("homeassistant/switch/"))
        self.assertTrue(retain)
        body = json.loads(payload)
        self.assertEqual(body["command_topic"], "zigbee2mqtt/plug1/set")

    def test_wait_for_state(self):
        b = MQTTBridge("h", 1883, client_factory=FakePaho,
                       clock=time.time)
        b._handle_message("zigbee2mqtt/lamp",
                          json.dumps({"state": "ON"}).encode())
        self.assertTrue(b.wait_for_state("lamp", "ON", timeout=1))
        self.assertFalse(b.wait_for_state("lamp", "OFF", timeout=0.3))

    def test_topic_wildcards(self):
        self.assertTrue(mqtt_mod.MQTTBridge._topic_matches("a/+", "a/b"))
        self.assertTrue(mqtt_mod.MQTTBridge._topic_matches("a/#", "a/b/c"))
        self.assertFalse(mqtt_mod.MQTTBridge._topic_matches("a/b", "a/c"))


# ── routines ──────────────────────────────────────────────────────────

from nomorals.integrations import routines as R


DEVS = [
    {"entity_id": "light.kitchen", "friendly_name": "Kitchen Light"},
    {"entity_id": "binary_sensor.motion", "friendly_name": "Motion"},
    {"entity_id": "sensor.temp", "friendly_name": "Temp"},
]


class TestRoutinesSweep(unittest.TestCase):
    def test_delay_order_preserved(self):
        d = R.build_routine(
            "at 7 turn on the kitchen light then wait 10 minutes",
            devices=DEVS)
        svcs = [a.service for a in d.actions]
        self.assertEqual(svcs, ["turn_on", "delay"])
        self.assertEqual(d.actions[1].params["minutes"], 10)

    def test_delay_first(self):
        d = R.build_routine("at 7 wait 5 min then turn on the kitchen light",
                            devices=DEVS)
        svcs = [a.service for a in d.actions]
        self.assertEqual(svcs[0], "delay")

    def test_after_pattern(self):
        d = R.build_routine(
            "when motion turns on turn on the kitchen light after 2 minutes",
            devices=DEVS)
        svcs = [a.service for a in d.actions]
        self.assertIn("turn_on", svcs)
        self.assertIn("delay", svcs)

    def test_no_motion_starter(self):
        d = R.build_routine("when no motion for 5 min turn off the "
                            "kitchen light", devices=DEVS)
        self.assertEqual(d.starter.kind, "device")
        self.assertEqual(d.starter.to_state, "off")
        self.assertEqual(d.starter.for_min, 5)
        r = R.confirm(d)
        trigger, _, _ = R.to_ha_automation(r)
        self.assertEqual(trigger["for"], {"minutes": 5})

    def test_numeric_condition(self):
        d = R.build_routine("at 7 turn on the kitchen light only if temp "
                            "above 28", devices=DEVS)
        self.assertEqual(len(d.conditions), 1)
        c = d.conditions[0]
        self.assertEqual(c.kind, "numeric")
        self.assertEqual(c.above, 28.0)
        r = R.confirm(d)
        _, _, conds = R.to_ha_automation(r)
        self.assertEqual(conds[0]["condition"], "numeric_state")
        self.assertEqual(conds[0]["above"], 28.0)

    def test_smart_mode(self):
        d = R.build_routine("when motion turns on turn on the kitchen "
                            "light", devices=DEVS)
        self.assertEqual(R.confirm(d).mode, "restart")
        d2 = R.build_routine("at 7 notify me hi", devices=DEVS)
        self.assertEqual(R.confirm(d2).mode, "parallel")

    def test_describe_steps(self):
        d = R.build_routine("at 7 turn on the kitchen light", devices=DEVS)
        out = R.describe(d)
        self.assertIn("▶️ **when:**", out)
        self.assertIn("⚡ **then:**", out)
        self.assertIn("1. turn on light.kitchen", out)

    def test_suggest_fix(self):
        d = R.build_routine("at 7 turn on the blarg light", devices=DEVS)
        fixes = R.suggest_fix(d, DEVS)
        self.assertTrue(fixes)
        self.assertEqual(fixes[0]["code"], "unknown_device")
        self.assertTrue(fixes[0]["suggestions"])

    def test_llm_fallback_hook(self):
        def fake_llm(chunk, devices):
            if "gizmo" in chunk:
                return {"entity_id": "light.kitchen", "service": "turn_on"}
            return None

        d = R.build_routine("at 7 activate the gizmo", devices=DEVS,
                            llm_parse=fake_llm)
        self.assertTrue(any(a.entity_id == "light.kitchen"
                            for a in d.actions))

    def test_delay_in_ha(self):
        d = R.build_routine("at 7 turn on the kitchen light then wait "
                            "3 minutes", devices=DEVS)
        r = R.confirm(d)
        _, actions, _ = R.to_ha_automation(r)
        self.assertEqual(actions[1], {"delay": {"minutes": 3}})


# ── market_data ───────────────────────────────────────────────────────

from nomorals.integrations import market_data as md


def synth_bars(n=100, seed=1):
    import random
    rng = random.Random(seed)
    bars, p = [], 100.0
    for i in range(n):
        o = p
        h = o * (1 + rng.random() * 0.01)
        l = o * (1 - rng.random() * 0.01)
        c = l + rng.random() * (h - l)
        p = c
        bars.append([i * 60000, o, h, l, c, 1000.0])
    return bars


class TestMarketDataSweep(unittest.TestCase):
    def test_indicators_list_input(self):
        ind = md.indicators(synth_bars())
        self.assertIn("rsi", ind)
        self.assertIn("macd", ind)
        self.assertIn("bb_upper", ind)
        self.assertIn("atr", ind)
        self.assertEqual(ind["n_bars"], 100)
        self.assertTrue(0 <= ind["rsi"] <= 100)

    def test_indicators_dict_input(self):
        rows = [{"open": b[1], "high": b[2], "low": b[3], "close": b[4]}
                for b in synth_bars()]
        ind = md.indicators(rows)
        self.assertIsNotNone(ind["ema_fast"])

    def test_indicators_need_bars(self):
        with self.assertRaises(md.MarketDataError):
            md.indicators([[0, 1, 1, 1, 1, 1]])

    def test_format_quote(self):
        q = {"symbol": "BTC/USDT", "price": 67000.0, "change_pct_24h": 2.5,
             "currency": "USDT", "bid": 66990.0, "ask": 67010.0,
             "high": 69000.0, "low": 66000.0, "spread_bps": 3.0,
             "source": "binance"}
        out = md.format_quote(q)
        self.assertIn("BTC/USDT", out)
        self.assertIn("67,000", out)
        self.assertIn("🟢", out)
        neg = dict(q, change_pct_24h=-1.0)
        self.assertIn("🔴", md.format_quote(neg))

    def test_cache_roundtrip(self):
        md.clear_cache()
        calls = []

        def loader():
            calls.append(1)
            return {"x": 1}

        v1 = md._cached(("k",), 60, loader)
        v2 = md._cached(("k",), 60, loader)
        self.assertEqual(v1, v2)
        self.assertEqual(len(calls), 1)
        md.clear_cache()

    def test_order_book_unsupported_market(self):
        with self.assertRaises(md.MarketDataError):
            md.order_book("AAPL", market="stocks")

    def test_ccxt_unavailable_is_false_without_pkg(self):
        # ccxt isn't installed in the test env → graceful False
        self.assertFalse(md.ccxt_available())

    def test_batch_quotes_skips_failures(self):
        orig = md.quote
        md.quote = lambda s, market="crypto": (_ for _ in ()).throw(
            md.MarketDataError("nope"))  # type: ignore[assignment]
        try:
            self.assertEqual(md.batch_quotes(["BTC", "ETH"]), [])
        finally:
            md.quote = orig  # type: ignore[assignment]


# ── sentinel_bridge validation layer ──────────────────────────────────

from nomorals.integrations import sentinel_bridge as sb


class TestSentinelValidation(unittest.TestCase):
    def test_backtest_runs(self):
        rep = sb.backtest(sb.sma_cross_strategy(), synth_bars(600),
                          symbol="BTC/USDT")
        self.assertGreaterEqual(rep.n_trades, 0)
        self.assertIn("backtest", rep.format_report())
        # equity starts after the indicator warmup window
        self.assertTrue(0 < len(rep.equity_curve) <= 600)
        self.assertLessEqual(rep.max_drawdown_pct, 0)

    def test_backtest_needs_bars(self):
        with self.assertRaises(sb.SentinelError):
            sb.backtest(sb.sma_cross_strategy(), synth_bars(10))

    def test_backtest_signal_shape_checked(self):
        with self.assertRaises(sb.SentinelError):
            sb.backtest(lambda data: [1, 0], synth_bars(100))

    def test_rsi_strategy(self):
        rep = sb.backtest(sb.rsi_mean_reversion_strategy(),
                          synth_bars(400), symbol="ETH/USDT")
        self.assertIsInstance(rep.n_trades, int)

    def test_risk_guard(self):
        g = sb.RiskGuard(daily_stop_loss_pct=3.0,
                         max_consecutive_losses=3)
        self.assertEqual(g.check(), (True, "ok"))
        for _ in range(3):
            g.record_trade(-1.0)
        self.assertTrue(g.halted)
        allowed, reason = g.check()
        self.assertFalse(allowed)
        self.assertIn("halted", reason)

    def test_risk_guard_daily_stop(self):
        g = sb.RiskGuard(daily_stop_loss_pct=2.0)
        g.record_trade(-2.5)
        allowed, _ = g.check()
        self.assertFalse(allowed)

    def test_paper_trader(self):
        pt = sb.PaperTrader(sb.sma_cross_strategy())
        status = {"status": "warming_up"}
        for b in synth_bars(120):
            status = pt.on_bar(b)
        self.assertEqual(status["status"], "live")
        self.assertIn("equity", status)

    def test_monte_carlo(self):
        rep = sb.backtest(sb.sma_cross_strategy(), synth_bars(2000))
        mc = sb.monte_carlo(rep.trades)
        if len(rep.trades) >= 10:
            self.assertIn("p_profit", mc)
            self.assertTrue(0 <= mc["p_profit"] <= 1)
        else:
            self.assertIn("note", mc)

    def test_position_size(self):
        size = sb.position_size(10000, 1.0, 67000, 65000)
        self.assertAlmostEqual(size, 0.05)

    def test_graduated_capital(self):
        stages = sb.graduated_capital(10000)
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[0][1], 1000.0)
        self.assertEqual(stages[-1][1], 10000.0)

    def test_strategy_registry(self):
        names = [s["name"] for s in sb.list_registered_strategies()]
        self.assertIn("sma_cross_10_30", names)
        fn = sb.get_strategy("sma_cross_10_30")
        self.assertTrue(callable(fn))
        with self.assertRaises(sb.SentinelError):
            sb.get_strategy("nope")


# ── email ─────────────────────────────────────────────────────────────

from nomorals.integrations.email_integration import (
    EmailMessage,
    format_digest,
    _html_to_text,
    _extract_body,
)


class TestEmailSweep(unittest.TestCase):
    def test_html_to_text(self):
        out = _html_to_text("<p>Hello<br><b>world</b></p>")
        self.assertIn("Hello", out)
        self.assertIn("world", out)
        self.assertNotIn("<p>", out)

    def test_extract_body_prefers_plain(self):
        import base64
        payload = {"mimeType": "multipart/alternative", "parts": [
            {"mimeType": "text/plain", "filename": "",
             "body": {"data": base64.urlsafe_b64encode(b"plain").decode()}},
            {"mimeType": "text/html", "filename": "",
             "body": {"data": base64.urlsafe_b64encode(
                 b"<p>html</p>").decode()}},
            {"mimeType": "application/pdf", "filename": "d.pdf",
             "body": {"attachmentId": "a1", "size": 9}},
        ]}
        body, atts = _extract_body(payload)
        self.assertEqual(body, "plain")
        self.assertEqual(atts[0]["filename"], "d.pdf")
        self.assertEqual(atts[0]["attachmentId"], "a1")

    def test_extract_body_html_fallback(self):
        import base64
        payload = {"mimeType": "text/html", "filename": "",
                   "body": {"data": base64.urlsafe_b64encode(
                       b"<p>Hi <b>there</b></p>").decode()}}
        body, _ = _extract_body(payload)
        self.assertIn("Hi", body)
        self.assertIn("there", body)

    def test_format_digest(self):
        msgs = [
            EmailMessage("1", "Alice <a@x>", ["me"], "Subj", "body text",
                         time.time(), is_read=False, snippet="snip"),
            EmailMessage("2", "Bob <b@x>", ["me"], "Old", "x",
                         time.time(), is_read=True),
        ]
        out = format_digest(msgs)
        self.assertIn("1 unread", out)
        self.assertIn("🔵", out)
        self.assertIn("⚪", out)
        self.assertIn("Alice", out)
        self.assertEqual(format_digest([]), "📬 inbox\n_all clear — nothing here._")

    def test_short_from(self):
        m = EmailMessage("1", '"Alice Smith" <a@x>', [], "s", "b", 0)
        self.assertEqual(m.short_from(), "Alice Smith")


# ── stt ───────────────────────────────────────────────────────────────

from nomorals.integrations.stt import (
    SpeechToText,
    STTEngine,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


class TestSTTSweep(unittest.TestCase):
    def _result(self):
        return TranscriptionResult(
            text="hello world", language="en", segments=[
                TranscriptionSegment(0.5, 1.5, "hello",
                                     words=[TranscriptionWord("hello", 0.5,
                                                              1.0, 0.9)],
                                     speaker="SPEAKER_00"),
                TranscriptionSegment(1.6, 2.5, "world"),
            ])

    def test_to_srt(self):
        srt = self._result().to_srt()
        self.assertIn("00:00:00,500 --> 00:00:01,500", srt)
        self.assertIn("[SPEAKER_00] hello", srt)

    def test_to_vtt(self):
        vtt = self._result().to_vtt()
        self.assertIn("WEBVTT", vtt)
        self.assertIn("00:00:00.500 --> 00:00:01.500", vtt)

    def test_word_dict(self):
        d = TranscriptionWord("hi", 0.1, 0.2, 0.8).to_dict()
        self.assertEqual(d["word"], "hi")

    def test_fw_device_ladder_cpu(self):
        s = SpeechToText(engine=STTEngine.FASTER_WHISPER)
        dev, ct = s._fw_device()
        self.assertIn(dev, ("cpu", "cuda"))
        self.assertIn(ct, ("int8", "float16"))

    def test_diarize_missing_file(self):
        s = SpeechToText()
        with self.assertRaises(FileNotFoundError):
            asyncio.run(s.transcribe_diarized("/tmp/does-not-exist.wav"))

    def test_missing_audio_raises(self):
        s = SpeechToText()
        with self.assertRaises(FileNotFoundError):
            asyncio.run(s.transcribe_full("/tmp/does-not-exist.wav"))


# ── voice tts ─────────────────────────────────────────────────────────

from nomorals.integrations.voice_integration import TTSEngine


class TestTTSSweep(unittest.TestCase):
    def test_chunking(self):
        text = "Hello world. " * 400
        chunks = TTSEngine._chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 1800 for c in chunks))
        # no sentence is cut mid-word badly: rejoins to original-ish
        self.assertIn("Hello world.", chunks[0])

    def test_pick_voice(self):
        t = TTSEngine()
        self.assertEqual(t.pick_voice(style="news"), "news")
        self.assertEqual(t.pick_voice(language="en-NG"), "nigerian")
        self.assertEqual(t.pick_voice(style="nope"), "default")
        self.assertEqual(t.pick_voice(gender="female"), "female")

    def test_format_voices(self):
        out = TTSEngine().format_voices()
        self.assertIn("nigerian", out)
        self.assertIn("en-NG-EzinneNeural", out)

    def test_list_voices(self):
        self.assertIn("nigerian", TTSEngine().list_voices())


# ── payment ───────────────────────────────────────────────────────────

from nomorals.integrations.payment_integration import (
    PaymentIntegration,
    PaymentApproval,
    Transaction,
)


class TestPaymentSweep(unittest.TestCase):
    def setUp(self):
        self.p = PaymentIntegration(MagicMock(), MagicMock())
        self.tmp = tempfile.mkdtemp()
        self.orig = self.p._address_book_path
        path = os.path.join(self.tmp, "book.json")
        self.p._address_book_path = lambda: path  # type: ignore[method-assign]

    def tearDown(self):
        self.p._address_book_path = self.orig  # type: ignore[method-assign]

    def test_eip55(self):
        self.assertTrue(self.p._validate_address(
            "ETH", "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"))
        self.assertTrue(self.p._validate_address(
            "ETH", "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"))
        self.assertFalse(self.p._validate_address("ETH", "0x123"))
        self.assertTrue(self.p._validate_address(
            "BTC", "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"))

    def test_address_book(self):
        self.p.save_address("Exchange", "BTC",
                            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4")
        got = self.p.get_address("exchange")
        self.assertEqual(got["currency"], "BTC")
        self.assertEqual(len(self.p.list_addresses()), 1)
        self.assertTrue(self.p.delete_address("Exchange"))
        self.assertIsNone(self.p.get_address("exchange"))
        with self.assertRaises(Exception):
            self.p.save_address("Bad", "BTC", "not-an-address")

    def test_transaction_format(self):
        t = Transaction("abc", "send", 0.001, "BTC", "confirmed")
        out = t.format()
        self.assertIn("📤", out)
        self.assertIn("✅", out)
        self.assertIn("0.00100000 BTC", out)

    def test_approval_message_fee_fiat(self):
        a = PaymentApproval("x", "crypto_send", 0.5, "ETH", "0xabc",
                            "test", fee_estimate="~$1.20",
                            fiat_amount="$1,500.00")
        msg = a.to_message()
        self.assertIn("~$1.20", msg)
        self.assertIn("$1,500.00", msg)

    def test_explorer_url(self):
        self.assertEqual(
            self.p._explorer_url("BTC", txid="abc"),
            "https://mempool.space/tx/abc")
        self.assertIn("etherscan.io",
                      self.p._explorer_url("ETH", address="0x1"))

    def test_ascii_qr_honest(self):
        out = self.p._ascii_qr("bc1qtest")
        self.assertIn("bc1qtest", out)
        self.assertIn("qrcode", out)  # honest about being a placeholder

    def test_fee_estimate_static_fallback(self):
        est = asyncio.run(self.p.estimate_fee("SOL"))
        self.assertEqual(est["currency"], "SOL")
        self.assertIn("source", est)


# ── shopping ──────────────────────────────────────────────────────────

from nomorals.integrations.shopping_integration import (
    ShoppingIntegration,
    Product,
    PriceComparison,
)


class TestShoppingSweep(unittest.TestCase):
    def test_stars(self):
        self.assertTrue(Product("x", 1, rating=4.7).stars().startswith("★★★★"))
        self.assertEqual(Product("x", 1).stars(), "no ratings yet")

    def test_to_card(self):
        p = Product("Widget", 19.99, "USD", "amazon", rating=4.5,
                    review_count=100,
                    metadata={"shipping": "free 2-day"})
        card = p.to_card()
        self.assertIn("Widget", card)
        self.assertIn("19.99", card)
        self.assertIn("free 2-day", card)

    def test_comparison_format(self):
        prods = [Product("A", 10.0, "USD", "amazon"),
                 Product("A", 12.0, "USD", "walmart")]
        out = PriceComparison(query="a", products=prods,
                              best_price=prods[0]).format()
        self.assertIn("🏆 **best**", out)
        self.assertIn("spread", out)
        self.assertIn("no results", PriceComparison(query="z").format())

    def test_dedupe(self):
        prods = [Product("Sony WH-1000XM5", 399, "USD", "amazon"),
                 Product("WH-1000XM5 Sony", 379, "USD", "bestbuy"),
                 Product("Bose QC45", 299, "USD", "amazon")]
        deduped = ShoppingIntegration._dedupe_products(prods)
        self.assertEqual(len(deduped), 2)
        sony = next(p for p in deduped if "Sony" in p.title)
        self.assertEqual(sony.price, 379)  # cheaper kept

    def test_wishlist(self):
        s = ShoppingIntegration(MagicMock(), MagicMock())
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "w.json")
        s._wishlist_path = lambda: path  # type: ignore[method-assign]
        p = Product("W", 10.0, "USD", "amazon", url="http://x")
        s.add_to_wishlist(p, target_price=8.0)
        items = s.get_wishlist()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["target_price"], 8.0)
        self.assertIn("wishlist", s.format_wishlist())
        self.assertTrue(s.remove_from_wishlist("http://x"))
        self.assertEqual(s.get_wishlist(), [])


# ── naija ─────────────────────────────────────────────────────────────

from nomorals.integrations.naija_deals import Deal, NaijaDealHunter
from nomorals.integrations.naija_shopping import Steal


class TestNaijaSweep(unittest.TestCase):
    def test_deal_card(self):
        d = Deal("1", "Oraimo FreePods", "jumia", 25000, 40000, 37.5,
                 "https://x", deal_score=88, is_flash_sale=True,
                 is_lowest_90d=True, coupons=["JUMIA10"])
        msg = d.to_message()
        self.assertIn("GOD-TIER", msg)
        self.assertIn("lowest in 90 days", msg)
        self.assertIn("JUMIA10", msg)
        self.assertIn("FLASH SALE", msg)

    def test_score_badges(self):
        self.assertIn("HOT", Deal("1", "x", "jumia", 1, 2, 50, "u",
                                  deal_score=75).to_message())
        self.assertIn("MEH", Deal("1", "x", "jumia", 1, 2, 50, "u",
                                  deal_score=10).to_message())

    def test_match_products(self):
        ds = [Deal("1", "Oraimo FreePods Pro", "jumia", 25000, 40000,
                   37.5, "u1", deal_score=88),
              Deal("2", "FreePods Pro Oraimo", "konga", 27000, 40000,
                   32.5, "u2", deal_score=75),
              Deal("3", "Samsung A55", "jumia", 350000, 400000, 12.5,
                   "u3", deal_score=55)]
        groups = NaijaDealHunter.match_products(ds)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["vendors"], 2)
        self.assertEqual(groups[0]["best"].vendor, "jumia")
        out = NaijaDealHunter.format_matches(groups)
        self.assertIn("best: jumia", out)
        self.assertIn("**Samsung A55**", out)

    def test_steal_card(self):
        s = Steal("u", "jumia", "Oraimo", 25000, 40000, 24000, 92,
                  37.5, -4.2)
        msg = s.to_message()
        self.assertIn("GOD-TIER", msg)
        self.assertIn("cheaper elsewhere", msg)

    def test_is_lowest_90d(self):
        from nomorals.storage.db import Database
        db = Database(":memory:")
        h = NaijaDealHunter(MagicMock(), db)
        now = time.time()
        with db.transaction():
            for i, price in enumerate([30000, 28000, 26000]):
                db.execute(
                    "INSERT OR REPLACE INTO price_history "
                    "(product_url, vendor, price, timestamp, in_stock, "
                    "coupon) VALUES (?, ?, ?, ?, 1, '')",
                    ("http://p", "jumia", price, now - (3 - i) * 86400))
        self.assertTrue(h._is_lowest_90d("http://p", 25000))
        self.assertFalse(h._is_lowest_90d("http://p", 29000))


# ── smarthome ─────────────────────────────────────────────────────────

from nomorals.integrations.smarthome_integration import (
    SmartHomeIntegration,
    HAWebSocket,
    Device,
)


class TestSmartHomeSweep(unittest.TestCase):
    def test_format_devices(self):
        s = SmartHomeIntegration(MagicMock(), MagicMock())
        devs = [
            Device("light.kitchen", "Kitchen", "light", room="Kitchen",
                   state={"state": "on"}, available=True),
            Device("lock.front", "Front", "lock", room="Hall",
                   state={"state": "unlocked"}, available=True),
            Device("sensor.t", "T", "sensor", room="Kitchen",
                   state={"state": "24"}, available=False),
        ]
        out = s.format_devices(devs)
        self.assertIn("**Kitchen**", out)
        self.assertIn("🟢 Kitchen — on", out)
        self.assertIn("🔓 Front — unlocked", out)
        self.assertIn("unavailable", out)
        self.assertIn("no devices", s.format_devices([]))

    def test_state_text(self):
        d = Device("light.x", "X", "light", state={"state": "on"})
        self.assertEqual(d.state_text, "on")

    def test_subscribe_trigger_registration(self):
        ws = HAWebSocket("ws://localhost:8123/api/websocket", "tok")
        received = []

        async def go():
            sub_id = await ws.subscribe_trigger(
                {"platform": "state", "entity_id": "light.kitchen",
                 "to": "on"},
                callback=received.append)
            self.assertIn(sub_id, ws._trigger_subs)
            # simulate an incoming trigger event
            ws._handle_message({"id": sub_id, "type": "event",
                                "event": {"entity_id": "light.kitchen"}})
            self.assertEqual(len(received), 1)
            self.assertTrue(ws.unsubscribe_trigger(sub_id))
            self.assertFalse(ws.unsubscribe_trigger(sub_id))

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
