"""Navigation audit — Devon half (held register §3.8).

Executes the audit's own R1–R5 hub rules against the CURRENT hub
(SURVEY.md, gs-nav-audit: "no self-button; path back to hub; no
dead buttons; honest Back labels; one markup per message") and the
per-command behaviour pass over the full catalog (511 kinds):
exactly-once claiming, §33 gate shapes, the stamp boundary, honest
failure, catalog truth.

Findings fixed here (each pin red on the pre-fix tree):
- /file crashed on every arg'd call — dispatch dropped chat_key.
- /webshot and /console crashed on message-less dispatch
  (handle_control(message=None) is a designed shape: tour probes,
  self-heal and meta re-dispatch all call it that way).
- Self-sending handlers (the _send_long_checked discipline: ""
  means delivered) were followed by the #32 wrapper's false
  "bug on my end" fallback — /vision, /py proven here.
- The tour probe reported ActionableMessage replies (/game,
  /toolhub) as broken — it died on .strip().
- /doctor's registry check reported all 43 games undispatchable —
  its static scan read only literal branches, not the named
  membership sets the dispatch routes through.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.social.chat import control as C
from nomorals.social.chat import hub as H
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.tgbot_buttons import buttons_for_text

_TMP = os.environ.get("NM_TEST_TMPDIR") or None

STAMP = "Powered by peacethefirst"
FALLBACK = "ran but produced no reply"


# ── hub R1–R5 ─────────────────────────────────────────────────────

#: callback data → the screen it navigates to (run = an ACTION,
#: not navigation — the GS audit's re-run precedent).
NAV_TARGET = {
    "toolhub": "hub",
    "toolhub proxy": "proxy",
    "toolhub bin": "bin",
    "toolhub help": "help",
}
ACTION_CALLBACKS = {"toolhub proxy run"}
PARENT = {"proxy": "hub", "proxy_result": "proxy",
          "proxy_cooldown": "proxy", "bin": "hub",
          "bin_result": "bin", "help": "hub"}
SCREENS = ["hub", "proxy", "proxy_result", "proxy_cooldown",
           "bin", "bin_result", "help"]


class HubRuleTests(unittest.TestCase):
    def test_r3_every_button_parses_to_a_registered_command(self):
        for screen in SCREENS:
            for row in H.keyboard_for_screen(screen):
                for _label, data in row:
                    self.assertIn(data, set(NAV_TARGET) | ACTION_CALLBACKS,
                                  f"{screen}: unmapped callback {data}")
                    parsed = C.parse_control("/" + data)
                    self.assertIsNotNone(parsed, f"{screen}: {data}")
                    self.assertEqual(parsed.kind, "toolhub")

    def test_r1_no_screen_buttons_to_itself(self):
        for screen in SCREENS:
            for row in H.keyboard_for_screen(screen):
                for _label, data in row:
                    if data in NAV_TARGET:
                        self.assertNotEqual(
                            NAV_TARGET[data], screen,
                            f"{screen} has a self-button: {data}")

    def test_r2_r4_back_buttons_land_on_the_parent_they_name(self):
        for screen, parent in PARENT.items():
            backs = [(label, data)
                     for row in H.keyboard_for_screen(screen)
                     for (label, data) in row
                     if label.startswith("« Back")]
            self.assertEqual(len(backs), 1, f"{screen}: {backs}")
            label, data = backs[0]
            self.assertEqual(NAV_TARGET[data], parent,
                             f"{screen}: Back goes to {NAV_TARGET[data]}")
            named = label.replace("« Back to ", "").strip().lower()
            self.assertEqual(
                named, "hub" if parent == "hub" else parent,
                f"{screen}: label {label!r} names {named}, "
                f"target is {parent}")
        # the root carries no Back at all
        hub_labels = [label for row in H.keyboard_for_screen("hub")
                      for (label, _d) in row]
        self.assertFalse(any("Back" in x for x in hub_labels))

    def test_r5_screen_text_derives_exactly_its_own_keyboard(self):
        """One markup source: buttons_for_text(screen text) is the
        screen's keyboard_for_screen entry — nothing else attaches."""
        texts = {
            "hub": H.hub_text(),
            "proxy": H.proxy_intro_text(),
            "bin": H.bin_usage_text(),
            "help": H.hub_help_text(),
            "proxy_result": (f"{H.PROXY_HEADER}\n\n"
                             "12 working proxies saved"),
            "proxy_cooldown": (f"{H.PROXY_HEADER}\n\n"
                               "⏳ cooling down — try again in 30m."),
            "bin_result": H.format_bin_result({
                "ok": True, "bin": "45717300", "scheme": "visa",
                "type": "debit", "brand": "Visa", "prepaid": False,
                "country": {"name": "Denmark", "alpha2": "DK",
                            "emoji": "", "currency": "DKK"},
                "bank": {"name": "Test Bank", "city": ""}}),
        }
        for screen, text in texts.items():
            self.assertEqual(buttons_for_text(text),
                             H.keyboard_for_screen(screen), screen)


class _FakeResp:
    def __init__(self, payload=None):
        self._payload = payload or {"ok": True, "result": True}

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls: list = []

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        if "sendMessage" in url:
            return _FakeResp({"ok": True, "result": {"message_id": 99}})
        return _FakeResp()

    def get(self, url, timeout=None):
        self.calls.append((url, {}))
        return _FakeResp()


class AdapterMarkupTests(unittest.TestCase):
    """R5, adapter half: a message carries ONE markup; an empty
    reply_keyboard is the remove verb (the abolished grid's
    clearing carrier)."""

    def _adapter(self, session):
        from nomorals.social.chat.telegram import TelegramBotAdapter
        return TelegramBotAdapter(
            token="0" * 8 + ":" + "x" * 30, session=session,
            owner_only=True, owner_sender_ids=("7",),
            adapter_name="telegram-personal")

    def test_both_markups_reply_keyboard_wins(self):
        import json as _json
        session = _FakeSession()
        adapter = self._adapter(session)
        chat = ChatRef(platform="telegram-personal", chat_id="42",
                       kind=ChatKind.DM)
        result = adapter.send(
            chat, "hello",
            buttons=[[("Tap", "toolhub")]],
            reply_keyboard=[["/status"]])
        self.assertTrue(result.ok)
        params = next(p for u, p in session.calls if "sendMessage" in u)
        markup = params["reply_markup"]
        if isinstance(markup, str):
            markup = _json.loads(markup)
        # exactly one markup, and it is the reply keyboard
        self.assertIn("keyboard", markup)
        self.assertNotIn("inline_keyboard", markup)

    def test_empty_reply_keyboard_is_the_remove_verb(self):
        import json as _json
        session = _FakeSession()
        adapter = self._adapter(session)
        chat = ChatRef(platform="telegram-personal", chat_id="42",
                       kind=ChatKind.DM)
        result = adapter.send(chat, "clearing", reply_keyboard=[])
        self.assertTrue(result.ok)
        params = next(p for u, p in session.calls if "sendMessage" in u)
        markup = params["reply_markup"]
        if isinstance(markup, str):
            markup = _json.loads(markup)
        self.assertEqual(markup, {"remove_keyboard": True})


# ── catalog truth (law e) ─────────────────────────────────────────

class CatalogTruthTests(unittest.TestCase):
    def test_every_kind_documented_grouped_and_parseable(self):
        groups = {}
        for _name, members in C.LIST_GROUPS:
            for m in members:
                groups[m] = groups.get(m, 0) + 1
        for kind in C.CONTROL_COMMANDS:
            self.assertIn(kind, C.COMMAND_DETAILS, kind)
            self.assertIn(kind, C.LIST_ONELINERS, kind)
            self.assertEqual(groups.get(kind), 1, kind)
            # parse with the kind's minimum arity met: a bare call
            # to a min-args kind parses to the arity-error shape by
            # design (its usage answer), not to the kind itself.
            filler = " ".join(["x"] * C.CONTROL_COMMANDS[kind][0])
            parsed = C.parse_control(f"/{kind} {filler}".strip())
            self.assertIsNotNone(parsed, kind)
            self.assertEqual(parsed.kind, kind)


# ── runtime harness ───────────────────────────────────────────────

class _FastRouter:
    def providers(self):
        return []

    def chat(self, *a, **k):
        from nomorals.llm.base import LLMResponse
        return LLMResponse(text="canned answer", model="fake")


class _StubGateway:
    adapters: dict = {}
    dry_run = False

    def __init__(self):
        self.sent: list = []

    def status(self):
        return {"local": {"running_in_session": True}}

    def set_rate_limit(self, *a, **k):
        return None

    def typing(self, *a, **k):
        return True

    def send(self, platform, chat, text, **kw):
        self.sent.append(text if isinstance(text, str) else str(text))
        return SimpleNamespace(ok=True)

    def start_one(self, name, handler, adapter=None):
        return {"platform": name, "running": False}

    def stop_one(self, name):
        return {"platform": name, "running": False}

    def stop(self):
        return None


def _make_runtime(router=None):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="nm-navaudit-", dir=_TMP)
    settings = load_settings(overrides={
        "home": tmp, "partner.platforms": "local",
        "chat.local_enabled": "true", "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=True,
                            with_tools=True)
    context.router = router or _FastRouter()
    gateway = _StubGateway()
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime._typing_enabled = lambda chat: False
    return runtime, gateway


class DispatchFixTests(unittest.TestCase):
    """The audit's fix pins — each red on the pre-fix tree."""

    @classmethod
    def setUpClass(cls):
        cls.runtime, cls.gateway = _make_runtime()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.runtime.stop()
        except Exception as exc:
            assert isinstance(exc, Exception)

    def call(self, text):
        return self.runtime.handle_control(text, "telegram:111")

    # F1 — /file dispatch dropped chat_key (TypeError on any call
    # that passed the arity gate).
    def test_file_with_an_arg_answers_usage_not_typeerror(self):
        out = self.call("/file x")
        self.assertIn("usage: /file", out)

    def test_file_full_shape_fails_honestly(self):
        out = self.call("/file telegram 111 /no/such/file.bin")
        self.assertIn("file send failed", out)

    # F2/F3 — message-less dispatch is a designed shape.
    def test_webshot_without_a_message_answers(self):
        out = self.call("/webshot")
        self.assertIn("webshot answers with a photo", out)

    def test_console_without_a_message_answers(self):
        out = self.call("/console")
        self.assertIn("the console lives in my DM", out)

    # F6 — a handler that answered by sending returns "" and the
    # wrapper must NOT follow it with the false fallback.
    def test_vision_no_arg_delivers_by_send_without_fallback(self):
        before = len(self.gateway.sent)
        out = self.call("/vision")
        self.assertEqual(out, "")
        new = self.gateway.sent[before:]
        self.assertTrue(any("vision" in s.lower() for s in new), new)
        self.assertFalse(any(FALLBACK in s for s in new))

    def test_py_output_delivers_by_send_without_fallback(self):
        before = len(self.gateway.sent)
        out = self.call("/py print(40 + 2)")
        self.assertEqual(out, "")
        new = self.gateway.sent[before:]
        self.assertTrue(any("42" in s for s in new), new)
        self.assertFalse(any(FALLBACK in s for s in new))

    def test_genuine_silence_still_gets_the_fallback(self):
        # The #32 law itself is untouched: a dispatch that produced
        # nothing and delivered nothing still answers honestly.
        inner = self.runtime._handle_control_inner
        self.runtime._handle_control_inner = (
            lambda text, chat_key, message=None, _no_stack=False: "")
        try:
            out = self.call("/ping")
        finally:
            self.runtime._handle_control_inner = inner
        self.assertIn(FALLBACK, out)

    # F4 — the tour probe reads ActionableMessage replies as text.
    def test_tour_probe_toolhub_and_game_are_ok(self):
        for kind in ("toolhub", "game"):
            res = self.runtime._tour_probe(kind, "telegram:111")
            self.assertEqual(res["verdict"], "ok",
                             f"{kind}: {res.get('error')}")

    # adjudication pin (passes pre- and post-fix): /providers was a
    # harness artifact in the audit sweep — the real Router exposes
    # providers() as a method, and with one the command answers.
    def test_providers_answers_with_a_method_router(self):
        out = self.call("/providers")
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())


class DoctorRegistryTests(unittest.TestCase):
    """F5 — /doctor's registry check must not report the games as
    undispatchable: they route through named membership sets."""

    def test_doctor_registry_line_is_ok(self):
        runtime, _gw = _make_runtime()
        try:
            out = runtime.handle_control("/doctor", "telegram:111")
        finally:
            try:
                runtime.stop()
            except Exception as exc:
                assert isinstance(exc, Exception)
        registry_lines = [ln for ln in out.splitlines()
                          if "registry:" in ln]
        self.assertTrue(registry_lines, out[:400])
        self.assertFalse(any("FAIL" in ln for ln in registry_lines),
                         registry_lines)


# ── pipeline laws (a) exactly-once, (b) §33, (c) stamp ───────────

def _msg(text, platform, kind, owner=False):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id="777", kind=kind),
        incoming=True, text=text,
        sender="owner" if owner else "member",
        sender_id="1" if owner else "42",
        meta={"is_owner": True} if owner else {})


class _ExplodingRouter:
    def chat(self, *a, **k):
        raise AssertionError("brain called: the command path owns it")


class PipelineLawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime, cls.gateway = _make_runtime(
            router=_ExplodingRouter())

    @classmethod
    def tearDownClass(cls):
        try:
            cls.runtime.stop()
        except Exception as exc:
            assert isinstance(exc, Exception)

    def drive(self, m):
        before = len(self.gateway.sent)
        self.runtime._process_inner(m)
        return self.gateway.sent[before:]

    def test_owner_dm_command_exactly_once(self):
        sends = self.drive(_msg("/ping", "telegram", ChatKind.DM,
                                owner=True))
        self.assertEqual(len(sends), 1)
        self.assertIn("pong", sends[0])

    def test_group_public_command_exactly_once(self):
        sends = self.drive(_msg("/ping", "telegram", ChatKind.GROUP))
        self.assertEqual(len(sends), 1)
        self.assertIn("pong", sends[0])

    def test_channel_command_is_silent(self):
        sends = self.drive(_msg("/help", "telegram", ChatKind.CHANNEL,
                                owner=True))
        self.assertEqual(sends, [])

    def test_presented_command_reply_carries_the_stamp(self):
        sends = self.drive(_msg("/joke", "telegram", ChatKind.DM,
                                owner=True))
        self.assertEqual(len(sends), 1)
        self.assertIn(STAMP, sends[0])


class ConversationalStampTests(unittest.TestCase):
    """The ratified boundary: free conversation is NEVER stamped."""

    def test_plain_conversation_carries_no_stamp(self):
        runtime, gateway = _make_runtime(router=_FastRouter())
        try:
            before = len(gateway.sent)
            runtime._process_inner(_msg(
                "hello there, how are you", "telegram", ChatKind.DM,
                owner=True))
            sends = gateway.sent[before:]
        finally:
            try:
                runtime.stop()
            except Exception as exc:
                assert isinstance(exc, Exception)
        self.assertEqual(len(sends), 1)
        self.assertIn("canned answer", sends[0])
        self.assertNotIn(STAMP, sends[0])


if __name__ == "__main__":
    unittest.main()
