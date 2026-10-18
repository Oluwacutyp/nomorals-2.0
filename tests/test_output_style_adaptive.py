"""Outputs get style + adaptive buttons; normal chats do not get restyled.

Owner correction (2026-10-17): style was meant for OUTPUTS — the
results that are not normal chats. These tests pin:

* ``core.output_style.render_output`` — header + divider + structured
  body, never a raw JSON blob, clean truncation;
* ``social.chat.adaptive_buttons`` — buttons derived from the output
  (kind + content), never the stale Retry/Expand/Next set, never a
  button for an unrelated command, never a dead button (every
  callback resolves in CONTROL_COMMANDS and fits the 47-byte budget);
* the scheduler / tool-reply / api paths render human-readable
  output (no ``{"...": ...}`` blobs);
* the Notifier attaches adaptive buttons (or explicit none) to pushes;
* the Telegram adapter never inherits buttons from a stale command.
"""

import unittest


class TestOutputStyle(unittest.TestCase):
    def test_dict_payload_structured_no_raw_json(self):
        from nomorals.core.output_style import render_output

        out = render_output(kind="tool", title="memory",
                            payload={"mode": "additive", "episodes": 1,
                                     "batches": 0, "ok": True})
        self.assertIn("🔧 memory", out)
        self.assertIn("mode: additive", out)
        self.assertIn("episodes: 1", out)
        self.assertNotIn('{"mode"', out)
        self.assertNotIn('"mode":', out)

    def test_key_json_string_parsed(self):
        from nomorals.core.output_style import render_output

        out = render_output(
            kind="scheduled", title="scheduled: memory-consolidation",
            payload='memory: {"mode": "additive", "episodes": 1, '
                    '"batches": 0, "summaries": 0, "forgotten": 0, '
                    '"skipped_distilled": 0, "seconds": 0.008, '
                    '"remaining": 2, "ok": true, "run_number": 15}')
        self.assertIn("memory:", out)
        self.assertIn("mode: additive", out)
        self.assertIn("run_number: 15", out)
        self.assertNotIn('{"mode"', out)
        self.assertNotIn('"mode":', out)

    def test_bare_json_string_parsed(self):
        from nomorals.core.output_style import render_output

        out = render_output(kind="tool", title="probe",
                            payload='{"ok": true, "count": 3}')
        self.assertIn("ok: yes", out)
        self.assertIn("count: 3", out)
        self.assertNotIn('{"ok"', out)

    def test_plain_string_gets_header(self):
        from nomorals.core.output_style import render_output

        out = render_output(kind="search", title="search: cats",
                            payload="found 3 results")
        self.assertTrue(out.startswith("🔍 search: cats"))
        self.assertIn("found 3 results", out)

    def test_long_output_truncates_cleanly(self):
        from nomorals.core.output_style import render_output

        big = {"field_%02d" % i: ("x" * 200) for i in range(40)}
        out = render_output(kind="tool", title="big", payload=big,
                            max_chars=500)
        self.assertLessEqual(len(out), 500)
        self.assertNotIn('{"field_00"', out)

    def test_never_raises(self):
        from nomorals.core.output_style import render_output

        self.assertIsInstance(render_output(kind="tool", title="t",
                                           payload=object()), str)
        self.assertIsInstance(render_output(kind="tool", title="t",
                                           payload=None), str)

    def test_parse_json_payload(self):
        from nomorals.core.output_style import parse_json_payload

        key, parsed = parse_json_payload('memory: {"a": 1}')
        self.assertEqual(key, "memory")
        self.assertEqual(parsed, {"a": 1})
        key, parsed = parse_json_payload('{"a": 1}')
        self.assertIsNone(key)
        self.assertEqual(parsed, {"a": 1})
        self.assertEqual(parse_json_payload("just text"), (None, None))
        self.assertEqual(parse_json_payload('memory: {broken'),
                         (None, None))


class TestAdaptiveButtons(unittest.TestCase):
    def _all_callbacks_valid(self, kb):
        from nomorals.social.chat.control import CONTROL_COMMANDS

        for row in kb or []:
            for _label, data in row:
                kind = (data or "").lstrip("/").split(None, 1)[0].lower()
                self.assertIn(kind, CONTROL_COMMANDS,
                              f"dead button callback: {data!r}")
                self.assertLessEqual(len(data.encode("utf-8")), 47)

    def test_scheduled_memory_consolidation(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(
            output_kind="scheduled",
            payload={"job": "memory-consolidation", "ok": True})
        flat = {label: data for row in kb for label, data in row}
        # Memory-system actions (NOT /memory — that is a game).
        self.assertEqual(flat["🧠 Recall"], "recall")
        self.assertEqual(flat["📋 Memories"], "memories")
        self.assertEqual(flat["📋 Jobs"], "schedule list")
        self.assertEqual(flat["🔁 Re-run"],
                         "schedule run memory-consolidation")
        self.assertNotIn("⏭️ Next", flat)
        self.assertNotIn("🔍 Expand", flat)
        self._all_callbacks_valid(kb)

    def test_scheduled_other_job(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(
            output_kind="scheduled", payload={"job": "nightly-backup"})
        flat = {label: data for row in kb for label, data in row}
        self.assertEqual(flat["🔁 Re-run"], "schedule run nightly-backup")
        self.assertEqual(flat["📋 Jobs"], "schedule list")
        self._all_callbacks_valid(kb)

    def test_search_buttons_use_the_query(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(output_kind="search",
                                  payload={"query": "cats"})
        flat = {label: data for row in kb for label, data in row}
        self.assertEqual(flat["💾 Save query"], "remember cats")
        self.assertEqual(flat["🔬 Deep research"], "searchdeep cats")
        self.assertEqual(flat["🔁 Search again"], "search cats")
        self._all_callbacks_valid(kb)

    def test_search_without_query_gets_none(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        self.assertIsNone(adaptive_buttons_for(output_kind="search",
                                               payload={}))

    def test_game_button_label_names_the_game(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(output_kind="game",
                                  payload={"game": "trivia"})
        flat = {label: data for row in kb for label, data in row}
        self.assertEqual(flat["🔁 Rematch trivia"], "game rematch")
        self.assertEqual(flat["⏭️ Next"], "game list")
        self._all_callbacks_valid(kb)

    def test_command_result_gets_retry_only(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(output_kind="command",
                                  origin_command="/game trivia")
        flat = [b for row in kb for b in row]
        self.assertEqual(flat, [("🔁 Retry", "/game trivia")])
        self._all_callbacks_valid(kb)

    def test_tool_result_without_origin_gets_none(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        self.assertIsNone(adaptive_buttons_for(output_kind="tool"))
        self.assertIsNone(adaptive_buttons_for(
            output_kind="tool", origin_command=""))

    def test_text_game_result_via_patterns(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(
            text="The House wins — points and coins credited")
        flat = {label: data for row in kb for label, data in row}
        self.assertEqual(flat["🔁 Rematch"], "game rematch")
        self._all_callbacks_valid(kb)

    def test_plain_text_gets_none(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        self.assertIsNone(adaptive_buttons_for(text="just chatting"))
        self.assertIsNone(adaptive_buttons_for(text="✅ done"))
        self.assertIsNone(adaptive_buttons_for(text=""))

    def test_over_budget_button_omitted_not_truncated(self):
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for

        long_query = "x" * 60  # "remember <query>" exceeds 47 bytes
        kb = adaptive_buttons_for(output_kind="search",
                                  payload={"query": long_query})
        flat = {label: data for row in kb for label, data in row} if kb else {}
        # Over-budget entries are dropped; the keyboard never carries
        # a truncated query that would run the wrong query.
        for data in flat.values():
            self.assertLessEqual(len(data.encode("utf-8")), 47)
            self.assertNotIn("xxxx", data)
        self._all_callbacks_valid(kb)


class TestRenderedOutputs(unittest.TestCase):
    def test_scheduler_run_tool_no_raw_json(self):
        from unittest.mock import MagicMock

        from nomorals.agents.scheduler import Scheduler

        sched = Scheduler.__new__(Scheduler)
        ctx = MagicMock()
        outcome = MagicMock()
        outcome.ok = True
        outcome.value = {"mode": "additive", "episodes": 1, "ok": True}
        ctx.tools.call.return_value = outcome
        sched.context = ctx
        out = sched._run_tool({"tool": "memory", "args": {}})
        self.assertIn("mode: additive", out)
        self.assertNotIn('{"mode"', out)
        self.assertNotIn('"mode":', out)

    def test_tool_reply_no_raw_json(self):
        from unittest.mock import MagicMock

        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        ctx = MagicMock()
        outcome = MagicMock()
        outcome.ok = True
        outcome.value = {"count": 3, "items": ["a", "b"]}
        ctx.tools.call.return_value = outcome
        rt.context = ctx
        rt._ref_from_key = MagicMock(return_value=MagicMock(platform="telegram"))
        rt._send_long_checked = MagicMock(
            side_effect=lambda _p, _c, text: text)
        out = rt._tool_reply("probe", "telegram:1")
        self.assertIn("count: 3", out)
        self.assertNotIn('{"count"', out)


class TestNotifierAdaptiveButtons(unittest.TestCase):
    def _notifier(self, gateway):
        from unittest.mock import MagicMock

        from nomorals.agents.notifier import Notifier

        ctx = MagicMock()
        ctx.db = None
        n = Notifier.__new__(Notifier)
        n.context = ctx
        n.db = None
        n.gateway = gateway
        n.settings = MagicMock()
        n.settings.partner.owner_chats = "telegram-bot:123"
        return n

    def test_scheduled_push_gets_adaptive_buttons(self):
        from unittest.mock import MagicMock

        sent = {}

        class Gw:
            def status(self):
                return {"telegram-bot": {"running_in_session": True}}

            def send(self, platform, chat, text, **kw):
                sent.update(kw)
                m = MagicMock()
                m.ok = True
                return m

        n = self._notifier(Gw())
        # _deliver is exercised directly: publish() would hit the DB.
        out = n._deliver("schedule", "✅ scheduled: memory-consolidation",
                         "memory:\nmode: additive",
                         output_kind="scheduled",
                         output_meta={"job": "memory-consolidation",
                                      "ok": True})
        self.assertTrue(out.delivered)
        flat = {label: data for row in sent["buttons"]
                for label, data in row}
        self.assertEqual(flat["🧠 Recall"], "recall")
        self.assertEqual(flat["🔁 Re-run"],
                         "schedule run memory-consolidation")

    def test_push_without_kind_gets_explicit_no_buttons(self):
        from unittest.mock import MagicMock

        sent = {}

        class Gw:
            def status(self):
                return {"telegram-bot": {"running_in_session": True}}

            def send(self, platform, chat, text, **kw):
                sent.update(kw)
                m = MagicMock()
                m.ok = True
                return m

        n = self._notifier(Gw())
        n._deliver("briefing", "🔔 Morning briefing", "news here")
        # Explicit [] — the adapter must NOT derive buttons from the
        # last inbound command for a push.
        self.assertEqual(sent.get("buttons"), [])

    def test_redelivery_derives_job_from_title(self):
        from unittest.mock import MagicMock

        sent = {}

        class Gw:
            def status(self):
                return {"telegram-bot": {"running_in_session": True}}

            def send(self, platform, chat, text, **kw):
                sent.update(kw)
                m = MagicMock()
                m.ok = True
                return m

        n = self._notifier(Gw())
        # No output_kind (stored row) — the title names the job.
        n._deliver("schedule", "✅ scheduled: memory-consolidation",
                   "memory:\nmode: additive")
        flat = {label: data for row in sent["buttons"]
                for label, data in row}
        self.assertEqual(flat["🔁 Re-run"],
                         "schedule run memory-consolidation")


class TestAdapterNoStaleButtons(unittest.TestCase):
    def _adapter(self, owner_only=True):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        a = TelegramBotAdapter.__new__(TelegramBotAdapter)
        a.owner_only = owner_only
        a.token = "123:fake"
        a.stats = {"sent": 0, "send_errors": 0}
        a._last_command = {}
        a._last_inbound_was_command = {}
        a._api = MagicMockApi()
        return a

    def test_explicit_empty_buttons_means_no_keyboard(self):
        a = self._adapter()
        a.send(MagicChat("1"), "✅ scheduled: memory-consolidation",
               buttons=[])
        self.assertNotIn("reply_markup", a._api.last_params)

    def test_chat_reply_gets_no_origin_buttons(self):
        a = self._adapter()
        # Owner typed /game trivia earlier, then chatted. The chat
        # reply must NOT inherit Retry for the old command.
        a._last_command["1"] = "/game trivia"
        a._last_inbound_was_command["1"] = False
        a.send(MagicChat("1"), "✅ got it, thanks!")
        self.assertNotIn("reply_markup", a._api.last_params)

    def test_command_result_gets_retry_only(self):
        a = self._adapter()
        a._last_command["1"] = "/game trivia"
        a._last_inbound_was_command["1"] = True
        a.send(MagicChat("1"), "✅ done — Result: 3 points")
        params = a._api.last_params
        self.assertIn("reply_markup", params)
        labels = [b["text"] for row in
                  params["reply_markup"]["inline_keyboard"] for b in row]
        self.assertEqual(labels, ["🔁 Retry"])
        self.assertNotIn("🔍 Expand", labels)
        self.assertNotIn("⏭️ Next", labels)

    def test_scheduled_push_text_with_stale_command_gets_no_buttons(self):
        # The exact screenshot regression: a scheduled
        # memory-consolidation push arriving after the owner last
        # typed an unrelated command must NOT carry that command's
        # Retry/Expand/Next.
        a = self._adapter()
        a._last_command["1"] = "/game trivia"
        a._last_inbound_was_command["1"] = False
        a.send(MagicChat("1"),
               "✅ scheduled: memory-consolidation\n"
               "memory:\nmode: additive\nepisodes: 1")
        self.assertNotIn("reply_markup", a._api.last_params)


class MagicChat:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.thread_id = None


class MagicApi:
    def __init__(self):
        self.last_params = {}

    def __call__(self, method, text=None, **params):
        self.last_params = params
        return {"message_id": "1"}


def MagicMockApi():
    return MagicApi()


class TestResponderNoOutputStyle(unittest.TestCase):
    def test_conversational_draft_not_restyled(self):
        import random

        from nomorals.partner.mood import MoodEngine
        from nomorals.partner.persona import default_persona
        from nomorals.partner.relationship import Relationship
        from nomorals.partner.responder import PartnerResponder

        class FakeRouter:
            def __init__(self, replies):
                self.replies = list(replies)

            def chat(self, messages, params=None, **kw):
                text = self.replies.pop(0) if self.replies else "mhm"
                from nomorals.partner.responder import LLMResponse

                return LLMResponse(text=text, model="fake")

        # format_reply would DROP the "----" divider line for a group
        # chat (sections off) and strip the 🎮 marker. The responder
        # must NOT do that to a normal chat: style is for outputs.
        draft = "🎮 GAMES\n----\nwe play mafia tonight"
        r = PartnerResponder(
            FakeRouter([draft]), default_persona(),
            MoodEngine(default_persona().baselines),
            Relationship(), None, None, rng=random.Random(0))
        bundle = r.respond(chat_platform="telegram",
                           user_text="what games?",
                           gate_mode="group")
        text = " ".join(bundle.parts)
        self.assertIn("mafia", text)
        self.assertIn("----", text)


if __name__ == "__main__":
    unittest.main()
