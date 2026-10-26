"""Devon Hub: hub screens, stack navigation, cooldowns, BIN lookup,
general-purpose progress, and the adapter's navigation edit.

All offline: BIN via injected fetchers, proxy via a fake lab,
adapter via a fake Bot API session (same pattern as
tests/test_telegram_callbacks.py).
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.social.chat import hub as hub_mod
from nomorals.social.chat.base import ChatRef
from nomorals.social.chat.hub import (
    BIN_HEADER,
    HUB_HEADER,
    PROXY_COOLDOWN_S,
    PROXY_HEADER,
    CooldownStore,
    bin_usage_text,
    format_bin_result,
    hub_help_text,
    hub_keyboard,
    hub_text,
    is_hub_screen,
    keyboard_for_screen,
    lookup_bin,
    proxy_intro_text,
)
from nomorals.social.chat.progress import ProgressReporter, render_bar
from nomorals.social.chat.progress_throttle import ProgressThrottler
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.chat.tgbot_buttons import (
    buttons_for_text,
    check_callback_data,
    personal_buttons_for,
)


# ── progress ─────────────────────────────────────────────────────────────

class ProgressTests(unittest.TestCase):
    def test_bar_from_real_counts(self):
        self.assertEqual(render_bar(5, 10), "▓▓▓▓▓░░░░░")
        self.assertEqual(render_bar(10, 10), "▓" * 10)
        self.assertEqual(render_bar(0, 10), "░" * 10)
        self.assertEqual(render_bar(0, 0), "")  # no counts → no bar
        self.assertEqual(render_bar(99, 10), "▓" * 10)  # clamped

    def test_throttle_and_finish(self):
        clock = [0.0]
        seen: list[str] = []
        rep = ProgressReporter(seen.append,
                               throttler=ProgressThrottler(
                                   min_interval_s=3.0,
                                   clock=lambda: clock[0]))
        rep.milestone("step one")
        self.assertEqual(seen, ["step one"])
        rep.milestone("step one")  # unchanged → suppressed
        self.assertEqual(seen, ["step one"])
        rep.milestone("step two")  # changed but too soon → suppressed
        self.assertEqual(seen, ["step one"])
        clock[0] = 3.5
        rep.milestone("step two")
        self.assertEqual(seen, ["step one", "step two"])
        rep.count(2, 4, "testing")
        clock[0] = 7.0
        rep.count(2, 4, "testing")  # same text → suppressed even later
        rep.finish("✅ done")  # finish is never throttled away
        self.assertEqual(seen[-1], "✅ done")
        rep.milestone("after finish")  # finished reporters stay quiet
        self.assertEqual(seen[-1], "✅ done")

    def test_count_line_shape(self):
        seen: list[str] = []
        rep = ProgressReporter(seen.append,
                               throttler=ProgressThrottler(min_interval_s=0.0))
        rep.count(3, 12, "testing")
        self.assertEqual(seen, ["testing: 3/12 ▓▓▓░░░░░░░"])
        rep.count(1, 0, "scraping")  # no total → label only, no bar
        self.assertEqual(seen[-1], "scraping")

    def test_thread_safe_counts(self):
        seen: list[str] = []
        lock = threading.Lock()

        def _emit(text):
            with lock:
                seen.append(text)

        rep = ProgressReporter(_emit,
                               throttler=ProgressThrottler(min_interval_s=0.0))
        threads = [threading.Thread(target=rep.count, args=(i, 50, "t"))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(seen), 20)  # no lost/duplicated crash


# ── screens + keyboards ──────────────────────────────────────────────────

def _all_payloads(keyboard):
    return [data for row in keyboard for _label, data in row]


class ScreenTests(unittest.TestCase):
    def test_every_keyboard_within_budget_and_routable(self):
        screens = ["hub", "proxy", "proxy_result", "proxy_cooldown",
                   "bin", "bin_result", "help"]
        for screen in screens:
            kb = keyboard_for_screen(screen)
            self.assertTrue(kb, screen)
            for data in _all_payloads(kb):
                check_callback_data(data)  # raises when over 47 bytes
                # every payload is a real toolhub command
                self.assertTrue(data == "toolhub"
                                or data.startswith("toolhub "),
                                (screen, data))

    def test_back_walks_exactly_one_level(self):
        # hub is the root (no Back); tool screens Back to hub;
        # result screens Back to their tool screen (not the hub).
        self.assertNotIn("toolhub", _all_payloads(hub_keyboard()))
        self.assertIn("toolhub", _all_payloads(keyboard_for_screen("proxy")))
        self.assertIn("toolhub", _all_payloads(keyboard_for_screen("bin")))
        self.assertIn("toolhub", _all_payloads(keyboard_for_screen("help")))
        self.assertIn("toolhub proxy",
                      _all_payloads(keyboard_for_screen("proxy_result")))
        self.assertNotIn("toolhub",
                         [d for d in _all_payloads(
                             keyboard_for_screen("proxy_result"))
                          if d == "toolhub"])
        self.assertIn("toolhub proxy",
                      _all_payloads(keyboard_for_screen("proxy_cooldown")))
        self.assertIn("toolhub bin",
                      _all_payloads(keyboard_for_screen("bin_result")))
        # Back labels name the parent screen
        labels = [label for row in keyboard_for_screen("proxy_result")
                  for label, _ in row]
        self.assertIn("« Back to Proxy", labels)
        labels = [label for row in keyboard_for_screen("bin_result")
                  for label, _ in row]
        self.assertIn("« Back to BIN", labels)

    def test_is_hub_screen(self):
        self.assertTrue(is_hub_screen(hub_text()))
        self.assertTrue(is_hub_screen(hub_help_text()))
        self.assertTrue(is_hub_screen(proxy_intro_text()))
        self.assertTrue(is_hub_screen(bin_usage_text()))
        self.assertFalse(is_hub_screen("usage: /toolhub | /hub proxy [run]"))
        self.assertFalse(is_hub_screen("hello"))
        self.assertFalse(is_hub_screen(""))

    def test_text_derivation_all_screens(self):
        self.assertEqual(buttons_for_text(hub_text()), hub_keyboard())
        self.assertEqual(buttons_for_text(hub_help_text()),
                         keyboard_for_screen("help"))
        self.assertEqual(buttons_for_text(proxy_intro_text()),
                         keyboard_for_screen("proxy"))
        self.assertEqual(
            buttons_for_text("🌐 Proxy lab — 3 working proxies\ntested: 10"),
            keyboard_for_screen("proxy_result"))
        self.assertEqual(
            buttons_for_text("🌐 Proxy lab — no working proxies right now"),
            keyboard_for_screen("proxy_result"))
        self.assertEqual(
            buttons_for_text("🌐 Proxy lab — refresh failed: boom"),
            keyboard_for_screen("proxy_result"))
        self.assertEqual(
            buttons_for_text("🌐 Proxy lab\n\n⏳ cooling down — try again "
                             "in 42m."),
            keyboard_for_screen("proxy_cooldown"))
        self.assertEqual(buttons_for_text(bin_usage_text()),
                         keyboard_for_screen("bin"))
        self.assertEqual(buttons_for_text("💳 BIN lookup — 457173\n\nscheme: visa"),
                         keyboard_for_screen("bin_result"))
        self.assertIsNone(buttons_for_text("hello world"))
        # the hub card MENTIONS the tools but keeps the hub keyboard
        card = hub_text()
        self.assertIn("💳 BIN lookup", card)
        self.assertEqual(buttons_for_text(card), hub_keyboard())

    def test_origin_derivation_all_screens(self):
        self.assertEqual(personal_buttons_for("x", "/toolhub"), hub_keyboard())
        self.assertEqual(personal_buttons_for("x", "/toolhub proxy"),
                         keyboard_for_screen("proxy"))
        self.assertEqual(personal_buttons_for("🌐 Proxy lab — 3 working",
                                              "/toolhub proxy run"),
                         keyboard_for_screen("proxy_result"))
        self.assertEqual(personal_buttons_for(
            "🌐 Proxy lab\n\n⏳ cooling down — try again in 5m.",
            "/toolhub proxy run"), keyboard_for_screen("proxy_cooldown"))
        self.assertEqual(personal_buttons_for("x", "/toolhub bin"),
                         keyboard_for_screen("bin"))
        self.assertEqual(personal_buttons_for("x", "/toolhub bin 457173"),
                         keyboard_for_screen("bin_result"))
        self.assertEqual(personal_buttons_for("x", "/toolhub help"),
                         keyboard_for_screen("help"))
        self.assertIsNone(personal_buttons_for("x", "/toolhub frobnicate"))


# ── cooldowns ────────────────────────────────────────────────────────────

class CooldownTests(unittest.TestCase):
    def test_enforce_expiry_and_remaining_text(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/hub/cooldowns.json"
            store = CooldownStore(path)
            self.assertEqual(store.remaining("tg:1", "proxy",
                                             PROXY_COOLDOWN_S), 0.0)
            store.mark("tg:1", "proxy", now=1000.0)
            left = store.remaining("tg:1", "proxy", PROXY_COOLDOWN_S,
                                   now=1000.0 + 3600 - 120)
            self.assertAlmostEqual(left, 120.0)
            self.assertEqual(CooldownStore.remaining_text(120.0), "2m")
            self.assertEqual(CooldownStore.remaining_text(45.0), "45s")
            self.assertEqual(CooldownStore.remaining_text(3600.0 + 300),
                             "1h 5m")
            self.assertEqual(store.remaining("tg:1", "proxy",
                                             PROXY_COOLDOWN_S,
                                             now=1000.0 + 3601), 0.0)
            # other user / other tool unaffected
            self.assertEqual(store.remaining("tg:2", "proxy",
                                             PROXY_COOLDOWN_S,
                                             now=1000.0), 0.0)

    def test_survives_restart_and_corrupt_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/cooldowns.json"
            store = CooldownStore(path)
            store.mark("tg:1", "proxy", now=5000.0)
            reloaded = CooldownStore(path)
            self.assertGreater(reloaded.remaining("tg:1", "proxy",
                                                  PROXY_COOLDOWN_S,
                                                  now=5000.0), 0.0)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            broken = CooldownStore(path)  # corrupt → start clean, no raise
            self.assertEqual(broken.remaining("tg:1", "proxy",
                                              PROXY_COOLDOWN_S), 0.0)


# ── BIN lookup ───────────────────────────────────────────────────────────

def _bin_ok(url, headers, timeout):
    assert headers.get("Accept-Version") == "3"
    assert timeout == 10.0
    return 200, {
        "scheme": "visa", "type": "debit", "brand": "Visa",
        "prepaid": False,
        "country": {"name": "Denmark", "alpha2": "DK",
                    "emoji": "🇩🇰", "currency": "DKK"},
        "bank": {"name": "Jyske Bank", "city": "Silkeborg",
                 "url": "www.jyskebank.dk", "phone": "+4589898989"},
    }


class BinTests(unittest.TestCase):
    def test_known_bin(self):
        res = lookup_bin("457173", fetcher=_bin_ok)
        self.assertTrue(res["ok"])
        self.assertEqual(res["scheme"], "visa")
        self.assertEqual(res["type"], "debit")
        self.assertEqual(res["country"]["name"], "Denmark")
        self.assertEqual(res["bank"]["name"], "Jyske Bank")
        text = format_bin_result(res)
        self.assertIn("visa", text)
        self.assertIn("Denmark", text)
        self.assertIn("Jyske Bank", text)

    def test_unknown_rate_limited_unavailable(self):
        res = lookup_bin("999999", fetcher=lambda *a: (404, None))
        self.assertEqual((res["ok"], res["error"]), (False, "unknown"))
        self.assertIn("Unknown BIN", format_bin_result(res))
        res = lookup_bin("999999", fetcher=lambda *a: (429, None))
        self.assertEqual((res["ok"], res["error"]), (False, "rate_limited"))
        self.assertIn("Rate-limited", format_bin_result(res))

        def _boom(*args):
            raise ConnectionError("down")

        res = lookup_bin("999999", fetcher=_boom)
        self.assertEqual((res["ok"], res["error"]), (False, "unavailable"))
        self.assertIn("unavailable", format_bin_result(res))

    def test_invalid_input_never_calls(self):
        calls: list[str] = []

        def _spy(*args):
            calls.append("called")
            return 200, {}

        for bad in ("abc", "12345", "123456789", ""):
            res = lookup_bin(bad, fetcher=_spy)
            self.assertEqual((res["ok"], res["error"]), (False, "usage"))
        self.assertEqual(calls, [])
        # spaces/dashes are stripped, then validated
        res = lookup_bin("4571 73", fetcher=_bin_ok)
        self.assertTrue(res["ok"])


# ── runtime dispatch ─────────────────────────────────────────────────────

class _FakeAdapter:
    def __init__(self):
        self.edits: list[str] = []

    def edit_progress(self, chat, message_id, text):
        self.edits.append(text)


class _FakeGateway:
    def __init__(self, *, file_ok=True):
        self.file_ok = file_ok
        self.sent: list[str] = []
        self.adapter = _FakeAdapter()

    def send(self, platform, chat, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id="77", ok=True)

    def _adapter_for(self, platform):
        return self.adapter

    def send_file(self, platform, chat, path, caption=""):
        return SimpleNamespace(ok=self.file_ok)


class _FakeLab:
    constructed: list = []

    def __init__(self, context):
        _FakeLab.constructed.append(context)
        self.context = context

    def status(self):
        return {"working_total": 4, "fresh_24h": 3}

    def refresh(self, *, on_scrape_source=None, on_test_progress=None):
        if on_scrape_source is not None:
            on_scrape_source("src1", 5, "")
            on_scrape_source("src2", 0, "timeout")
        if on_test_progress is not None:
            on_test_progress(1, 2, None)
            on_test_progress(2, 2, None)
        return {"tested": 10, "working": 3,
                "by_scheme": {"http": 3},
                "best": [{"country": "US"}, {"country": "DE"}]}

    def active_file(self):
        return {"written": True, "count": 3, "path": "/tmp/x.txt",
                "fastest": "http://1.2.3.4:8080"}


def _runtime(tmp_home, gateway):
    rt = object.__new__(RuntimeSystemMixin)
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(home=tmp_home))
    rt.gateway = gateway
    rt._ref_from_key = lambda key: SimpleNamespace(platform="telegram",
                                                   chat_id="42")
    return rt


class ControlHubTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_lab = None
        import nomorals.tools.proxylab as proxylab_mod

        self._proxylab_mod = proxylab_mod
        self._orig_lab = proxylab_mod.ProxyLab
        _FakeLab.constructed = []

    def tearDown(self):
        self._proxylab_mod.ProxyLab = self._orig_lab

    def _patch_lab(self):
        self._proxylab_mod.ProxyLab = _FakeLab

    def test_bare_card_carries_pool_line(self):
        self._patch_lab()
        rt = _runtime(self._tmp.name, _FakeGateway())
        text = rt._control_toolhub("", "tg:1")
        self.assertTrue(text.startswith(HUB_HEADER))
        self.assertIn("4 working (3 fresh <24h)", text)

    def test_help_bin_usage_and_unknown(self):
        rt = _runtime(self._tmp.name, _FakeGateway())
        self.assertTrue(rt._control_toolhub("help", "tg:1").startswith(HUB_HEADER))
        self.assertTrue(rt._control_toolhub("bin", "tg:1").startswith(BIN_HEADER))
        self.assertIn("unknown proxy action",
                      rt._control_toolhub("proxy frobnicate", "tg:1"))
        self.assertIn("usage: /toolhub",
                      rt._control_toolhub("frobnicate", "tg:1"))

    def test_bin_result_uses_lookup(self):
        rt = _runtime(self._tmp.name, _FakeGateway())
        orig = hub_mod.lookup_bin
        hub_mod.lookup_bin = lambda digits: {
            "ok": True, "bin": digits, "scheme": "visa", "type": "credit",
            "brand": "Visa", "prepaid": False,
            "country": {"name": "Denmark", "alpha2": "DK", "emoji": "🇩🇰",
                        "currency": "DKK"},
            "bank": {"name": "Jyske Bank", "city": "", "url": "",
                     "phone": ""}}
        try:
            text = rt._control_toolhub("bin 457173", "tg:1")
        finally:
            hub_mod.lookup_bin = orig
        self.assertIn("457173", text)
        self.assertIn("Denmark", text)

    def test_proxy_tool_screen_is_not_a_run(self):
        # /hub proxy shows the tool screen; nothing is constructed,
        # no file is sent, no cooldown is marked.
        self._patch_lab()
        gw = _FakeGateway()
        rt = _runtime(self._tmp.name, gw)
        text = rt._control_toolhub("proxy", "tg:1")
        self.assertTrue(text.startswith(PROXY_HEADER))
        self.assertIn("Tap Refresh", text)
        self.assertEqual(_FakeLab.constructed, [])
        self.assertEqual(gw.sent, [])

    def test_proxy_run_flow_with_progress_and_file(self):
        self._patch_lab()
        # Zero-interval throttle: the fake lab finishes instantly, so
        # the real 3s throttle would (correctly) suppress rapid
        # milestones; the test pins the milestone WIRING, not the
        # throttle (which ProgressTests covers separately).
        import nomorals.social.chat.progress as progress_mod
        from nomorals.social.chat.progress_throttle import ProgressThrottler

        orig_throttler = progress_mod.ProgressThrottler
        progress_mod.ProgressThrottler = (
            lambda: ProgressThrottler(min_interval_s=0.0))
        try:
            gw = _FakeGateway()
            rt = _runtime(self._tmp.name, gw)
            text = rt._control_toolhub("proxy run", "tg:1")
        finally:
            progress_mod.ProgressThrottler = orig_throttler
        self.assertIn("3 working proxies", text)
        self.assertIn("tested: 10 candidates", text)
        self.assertIn("http×3", text)
        self.assertIn("US, DE", text)
        self.assertIn("rot fast", text)
        self.assertIn("File sent above", text)
        # progress: intro sent, scrape milestones + test counts edited,
        # finish edit last
        self.assertEqual(len(gw.sent), 1)
        self.assertTrue(gw.sent[0].startswith(PROXY_HEADER))
        edits = gw.adapter.edits
        self.assertIn("🌐 Proxy lab — scraping free lists…", edits)
        self.assertTrue(any("scraped src1: 5 candidates" in e for e in edits))
        self.assertTrue(any("src2 failed" in e for e in edits))
        self.assertTrue(any("testing: 2/2" in e for e in edits))
        self.assertEqual(edits[-1], "🌐 Proxy lab — ✅ done")

    def test_proxy_run_cooldown_returns_wait_not_run(self):
        self._patch_lab()
        gw = _FakeGateway()
        rt = _runtime(self._tmp.name, gw)
        store = CooldownStore(f"{self._tmp.name}/hub/cooldowns.json")
        store.mark("tg:1", "proxy")
        text = rt._control_toolhub("proxy run", "tg:1")
        self.assertIn("⏳ cooling down", text)
        self.assertIn("try again in", text)
        self.assertEqual(_FakeLab.constructed, [])  # no wasted scrape
        self.assertEqual(gw.sent, [])

    def test_proxy_run_file_fallback_when_adapter_plain(self):
        self._patch_lab()
        gw = _FakeGateway(file_ok=False)
        rt = _runtime(self._tmp.name, gw)
        text = rt._control_toolhub("proxy run", "tg:1")
        self.assertIn("file: /tmp/x.txt", text)
        self.assertNotIn("File sent above", text)


# ── adapter navigation edit ──────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload=None):
        self._payload = payload or {"ok": True, "result": True}

    def json(self):
        return self._payload


class _FakeSession:
    """Records Bot API calls; never touches the network."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        if "sendMessage" in url:
            # send() reads result["message_id"] off the API result
            return _FakeResp({"ok": True, "result": {"message_id": 99}})
        return _FakeResp()

    def get(self, url, timeout=None):
        self.calls.append((url, {}))
        return _FakeResp()


class _FailingEditSession(_FakeSession):
    def post_json(self, url, params, timeout=None):
        if "editMessageText" in url:
            raise ValueError("message is not modified")
        return super().post_json(url, params, timeout)


def _personal_adapter(session=None):
    return TelegramBotAdapter(token="test-token-123",
                              session=session or _FakeSession(),
                              owner_only=True, owner_sender_ids=("7",),
                              adapter_name="telegram-personal")


def _query(adapter, data, chat_id="42", message_id=555):
    return {
        "id": "q1",
        "from": {"id": 7, "is_bot": False, "first_name": "Owner"},
        "message": {"chat": {"id": int(chat_id), "type": "private"},
                    "message_id": message_id},
        "data": data,
    }


class AdapterNavTests(unittest.TestCase):
    def test_callback_records_tapped_message(self):
        a = _personal_adapter()
        signed = a._sign_callback("toolhub proxy")
        a._handle_callback(_query(a, signed), lambda m: None)
        self.assertEqual(a._nav_edit["42"][0], "555")
        self.assertEqual(a._nav_edit["42"][1], "/toolhub proxy")

    def test_nav_send_edits_tapped_message_once(self):
        a = _personal_adapter()
        sess = a._session
        a._last_command["42"] = "/toolhub proxy"
        a._last_inbound_was_command["42"] = True
        a._nav_edit["42"] = ("555", "/toolhub proxy", time.time())
        chat = ChatRef(platform="telegram-personal", chat_id="42")
        res = a.send(chat, proxy_intro_text())
        self.assertTrue(res.ok)
        self.assertEqual(res.message_id, "555")
        edits = [c for c in sess.calls if "editMessageText" in c[0]]
        sends = [c for c in sess.calls if "sendMessage" in c[0]]
        self.assertEqual(len(edits), 1)
        self.assertEqual(len(sends), 0)
        params = edits[0][1]
        self.assertEqual(params["message_id"], 555)
        keyboard = params["reply_markup"]["inline_keyboard"]
        payloads = [btn["callback_data"] for row in keyboard for btn in row]
        # signed payloads verify back to the proxy tool screen's buttons
        unsigned = [a._verify_callback(p) for p in payloads]
        self.assertIn("toolhub proxy run", unsigned)
        self.assertIn("toolhub", unsigned)
        # marker consumed: the next send goes out new
        res2 = a.send(chat, hub_text())
        sends2 = [c for c in sess.calls if "sendMessage" in c[0]]
        self.assertEqual(len(sends2), 1)
        self.assertTrue(res2.ok)

    def test_non_hub_text_consumes_marker_and_sends_new(self):
        a = _personal_adapter()
        sess = a._session
        a._last_command["42"] = "/toolhub proxy"
        a._last_inbound_was_command["42"] = True
        a._nav_edit["42"] = ("555", "/toolhub proxy", time.time())
        chat = ChatRef(platform="telegram-personal", chat_id="42")
        a.send(chat, "hello, not a hub screen")
        self.assertEqual([c for c in sess.calls if "editMessageText" in c[0]],
                         [])
        self.assertEqual(len([c for c in sess.calls
                              if "sendMessage" in c[0]]), 1)
        self.assertNotIn("42", a._nav_edit)

    def test_expired_marker_sends_new(self):
        a = _personal_adapter()
        sess = a._session
        a._last_command["42"] = "/toolhub proxy"
        a._last_inbound_was_command["42"] = True
        a._nav_edit["42"] = ("555", "/toolhub proxy", time.time() - 400)
        chat = ChatRef(platform="telegram-personal", chat_id="42")
        a.send(chat, proxy_intro_text())
        self.assertEqual([c for c in sess.calls if "editMessageText" in c[0]],
                         [])
        self.assertEqual(len([c for c in sess.calls
                              if "sendMessage" in c[0]]), 1)

    def test_edit_failure_falls_back_to_new_message(self):
        a = _personal_adapter(session=_FailingEditSession())
        sess = a._session
        a._last_command["42"] = "/toolhub proxy"
        a._last_inbound_was_command["42"] = True
        a._nav_edit["42"] = ("555", "/toolhub proxy", time.time())
        chat = ChatRef(platform="telegram-personal", chat_id="42")
        res = a.send(chat, proxy_intro_text())
        self.assertTrue(res.ok)
        self.assertEqual(len([c for c in sess.calls
                              if "sendMessage" in c[0]]), 1)

    def test_gated_adapter_edits_via_content_fallback(self):
        # Gated bot: no personal keyboards, but the adaptive fallback
        # (buttons_for_text) still yields the hub card's keyboard.
        a = TelegramBotAdapter(token="test-token-123",
                               session=_FakeSession(),
                               adapter_name="telegram-bot")
        sess = a._session
        a._last_command["42"] = "/toolhub"
        a._last_inbound_was_command["42"] = True
        a._nav_edit["42"] = ("555", "/toolhub", time.time())
        chat = ChatRef(platform="telegram-bot", chat_id="42")
        res = a.send(chat, hub_text())
        self.assertTrue(res.ok)
        edits = [c for c in sess.calls if "editMessageText" in c[0]]
        self.assertEqual(len(edits), 1)
        payloads = [btn["callback_data"] for row in
                    edits[0][1]["reply_markup"]["inline_keyboard"]
                    for btn in row]
        unsigned = [a._verify_callback(p) for p in payloads]
        self.assertIn("toolhub proxy", unsigned)


if __name__ == "__main__":
    unittest.main()
