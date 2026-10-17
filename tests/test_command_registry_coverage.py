"""Commands + outputs audit: every registered command must have help,
a handler, a /list slot, and a non-empty result.

Regression coverage for the infrastructure-round commands audit:

* No silent duplicate keys in CONTROL_COMMANDS / COMMAND_DETAILS /
  LIST_ONELINERS. (Python dict literals silently keep the LAST duplicate:
  /say and /research each had two entries, so the help page described a
  different feature than the runtime ran, and the rich-voice /say branch
  was unreachable dead code.)
* Every CONTROL_COMMANDS kind has: a COMMAND_DETAILS page (what/usage/
  example, all non-empty), a LIST_ONELINERS entry, exactly one LIST_GROUPS
  slot, a mention in help_text() and list_catalog(), and a dispatch path
  in handle_control (explicit branch, set branch, or the generic
  GAME_START_COMMANDS game branch).
* handle_control has exactly ONE /say branch, routing all three /say
  forms (gateway send, named-voice TTS, rich voice).
* The tour probe and the heal probe treat "unknown command /<kind>" and
  an empty reply as BROKEN, not OK — a registered command answering
  "unknown command" is a dispatch gap, and a command returning nothing
  returns nothing.
* Game-start commands are never probed by tour/heal (probing one would
  START a live game in the chat); they are skipped honestly.

The registry checks parse source / import the light control module; the
probe/routing checks drive a real PartnerRuntime with a fake gateway
(the same harness as test_slash_help_routing).
"""

from __future__ import annotations

import ast
import collections
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.core.self_heal import _probe_inner as heal_probe_inner
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    GAME_COMMANDS,
    GAME_START_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    detailed_help,
    help_text,
    list_catalog,
)
from nomorals.social.chat.gateway import ChatGateway

REPO = Path(__file__).resolve().parent.parent
CONTROL_SRC = REPO / "nomorals" / "social" / "chat" / "control.py"
RUNTIME_SRC = REPO / "nomorals" / "agents" / "partner" / "runtime.py"

#: GAME_COMMANDS kinds with their own explicit handle_control branch
#: (not routed through the generic GAME_START_COMMANDS game branch).
GAME_EXPLICIT_BRANCHES = {
    "math", "inventory", "equip", "unequip", "repair", "level",
    "skill", "title", "stats", "daily", "mastery", "gift",
}


def _literal_keys(src: Path, name: str) -> list[str]:
    """Literal dict keys of one AnnAssign in source (duplicates kept)."""
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name):
            return [ast.literal_eval(k) for k in node.value.keys]
    raise AssertionError(f"{name} not found in {src}")


def _handle_control_source() -> str:
    src = RUNTIME_SRC.read_text()
    start = src.index("    def handle_control(")
    rest = src[start:]
    nxt = re.search(r"\n    def ", rest[1:])
    return rest[: nxt.start() + 1] if nxt else rest


def _handled_kinds(body: str) -> set[str]:
    kinds = set(re.findall(r'if kind == "([^"]+)"', body))
    for m in re.finditer(r"if kind in \{([^}]+)\}", body):
        kinds.update(re.findall(r'"([^"]+)"', m.group(1)))
    for m in re.finditer(r"if kind in \(([^)]+)\)", body):
        kinds.update(re.findall(r'"([^"]+)"', m.group(1)))
    return kinds


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []

    def run(self, handler) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons=None, **kwargs) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


class FakeGatewaySend:
    """Stand-in for ChatGateway.send: records the (platform, chat, text)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def send(self, platform, chat, text, **kwargs):
        self.calls.append((platform, chat, text))
        return SimpleNamespace(ok=True, error="")


def _make_runtime():
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-cmd-coverage-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": "5478650254",
        "partner.owner_sender_ids": "5478650254",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeAdapter("telegram")
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, tmp


class RegistryCoverageTests(unittest.TestCase):
    def test_no_duplicate_literal_keys(self):
        # A duplicate key in a dict literal silently discards every
        # earlier entry — the exact failure that shadowed /say and
        # /research. The literal key list (duplicates kept) must equal
        # the deduped set.
        for name in ("CONTROL_COMMANDS", "COMMAND_DETAILS",
                     "LIST_ONELINERS"):
            keys = _literal_keys(CONTROL_SRC, name)
            dupes = sorted({k for k, c in collections.Counter(keys).items()
                            if c > 1})
            self.assertEqual(dupes, [],
                             f"{name} has duplicate literal keys: {dupes}")

    def test_every_command_has_help_page(self):
        for kind in CONTROL_COMMANDS:
            with self.subTest(kind=kind):
                page = COMMAND_DETAILS.get(kind)
                self.assertIsNotNone(page, f"/{kind} has no COMMAND_DETAILS")
                for field in ("what", "usage", "example"):
                    self.assertTrue(str(page.get(field, "")).strip(),
                                    f"/{kind} DETAILS.{field} is empty")
                self.assertNotIn("run /" + kind + ".",
                                 page["what"],
                                 f"/{kind} has a placeholder help page")

    def test_every_command_has_oneliner(self):
        for kind in CONTROL_COMMANDS:
            with self.subTest(kind=kind):
                one = LIST_ONELINERS.get(kind, "")
                self.assertTrue(str(one).strip(),
                                f"/{kind} has no LIST_ONELINERS entry")

    def test_every_command_in_exactly_one_list_group(self):
        grouped: list[str] = []
        for _title, kinds in LIST_GROUPS:
            grouped.extend(kinds)
        counts = collections.Counter(grouped)
        dupes = sorted(k for k, c in counts.items() if c > 1)
        self.assertEqual(dupes, [],
                         f"commands in more than one /list group: {dupes}")
        missing = sorted(set(CONTROL_COMMANDS) - set(grouped))
        self.assertEqual(missing, [],
                         f"commands missing from every /list group: {missing}")
        extra = sorted(set(grouped) - set(CONTROL_COMMANDS))
        self.assertEqual(extra, [],
                         f"/list groups name unregistered commands: {extra}")

    def test_list_catalog_covers_every_command(self):
        catalog = list_catalog()
        for kind in CONTROL_COMMANDS:
            with self.subTest(kind=kind):
                self.assertIn(f"/{kind}", catalog,
                              f"/{kind} missing from list_catalog()")
        m = re.search(r"(\d+) commands", catalog)
        self.assertIsNotNone(m, "list_catalog() has no command count")
        self.assertEqual(int(m.group(1)), len(CONTROL_COMMANDS),
                         "list_catalog() count != registered commands")

    def test_list_catalog_lines_are_scannable(self):
        # Every command line is `  <code>/cmd</code> — what it does`
        # (headers via style.section, markers via style.cmd) — no bare
        # command name without its one-liner.
        catalog = list_catalog()
        for line in catalog.splitlines():
            # Command rows start with two spaces; the header/footer
            # lines mention <code>/command</code> generically.
            if line.startswith("  <code>/") and "&lt;" not in line:
                self.assertRegex(
                    line, r"<code>/[a-z0-9_]+</code> — .+",
                    f"unscannable /list line: {line!r}")

    def test_help_text_covers_every_command(self):
        text = help_text()
        for kind in CONTROL_COMMANDS:
            with self.subTest(kind=kind):
                self.assertIn(f"/{kind}", text,
                              f"/{kind} missing from help_text()")

    def test_detailed_help_resolves_every_command(self):
        for kind in CONTROL_COMMANDS:
            with self.subTest(kind=kind):
                page = detailed_help(kind)
                self.assertTrue(page.strip())
                self.assertNotIn("no help page", page.lower(),
                                 f"/help {kind} has no page")
                self.assertIn("<code>", page,
                              f"/help {kind} has no usage block")
                self.assertIn("try:", page,
                              f"/help {kind} has no example")
                self.assertIn("related:", page,
                              f"/help {kind} has no related links")

    def test_game_start_commands_partition(self):
        self.assertTrue(GAME_START_COMMANDS)
        self.assertLess(GAME_START_COMMANDS, set(GAME_COMMANDS))
        self.assertEqual(
            set(GAME_START_COMMANDS) | GAME_EXPLICIT_BRANCHES,
            set(GAME_COMMANDS),
            "GAME_START_COMMANDS + explicit branches must equal GAME_COMMANDS")
        self.assertEqual(
            set(GAME_START_COMMANDS) & GAME_EXPLICIT_BRANCHES, set(),
            "a game kind must not be both generic and explicit")

    def test_handle_control_dispatch_is_total(self):
        body = _handle_control_source()
        handled = _handled_kinds(body)
        # The generic game branch covers GAME_START_COMMANDS.
        self.assertIn("GAME_START_COMMANDS", body)
        handled |= set(GAME_START_COMMANDS)
        missing = sorted(set(CONTROL_COMMANDS) - handled - {"error"})
        self.assertEqual(
            missing, [],
            f"registered commands with no handle_control dispatch: {missing}")

    def test_handle_control_has_exactly_one_say_branch(self):
        body = _handle_control_source()
        # Code lines only: comments mention the branch text too.
        branches = re.findall(r'(?m)^\s*if kind == "say":', body)
        self.assertEqual(len(branches), 1,
                         "a second /say branch would be unreachable dead code")
        self.assertIn("_control_say(", body,
                      "/say must route rich-voice forms to _control_say")

    def test_say_and_research_help_match_runtime(self):
        # The two commands whose help used to describe a different
        # feature than the runtime ran: the pages must now name the
        # real behaviour.
        say = COMMAND_DETAILS["say"]
        self.assertIn("platform:chat", say["usage"])
        self.assertIn("[direction]", say["usage"])
        research = COMMAND_DETAILS["research"]
        self.assertIn("web research", research["what"])
        self.assertNotIn("scheduled research runs", research["what"])


class ProbeHonestyTests(unittest.TestCase):
    def test_heal_probe_flags_unknown_command_reply(self):
        report = heal_probe_inner(
            lambda text, chat_key, message=None: "unknown command /status",
            "status")
        self.assertTrue(report.broken)
        self.assertIn("dispatch gap", report.err)

    def test_heal_probe_flags_empty_reply(self):
        report = heal_probe_inner(
            lambda text, chat_key, message=None: "", "status")
        self.assertTrue(report.broken)
        self.assertIn("empty reply", report.err)

    def test_heal_probe_accepts_real_reply(self):
        report = heal_probe_inner(
            lambda text, chat_key, message=None: "all good", "status")
        self.assertFalse(report.broken)

    def test_heal_probe_accepts_none_reply(self):
        # Non-string replies (legacy handlers returning None) are not
        # reply-level failures the probe can judge — no false positive.
        report = heal_probe_inner(
            lambda text, chat_key, message=None: None, "status")
        self.assertFalse(report.broken)


class RuntimeProbeRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime, cls._tmp = _make_runtime()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_tour_probe_flags_unknown_command_reply(self):
        with mock.patch.object(
                self.runtime, "handle_control",
                return_value="unknown command /status"):
            res = self.runtime._tour_probe("status", "telegram:1")
        self.assertEqual(res["verdict"], "broken")
        self.assertIn("dispatch gap", res.get("error", ""))

    def test_tour_probe_flags_empty_reply(self):
        with mock.patch.object(
                self.runtime, "handle_control", return_value=""):
            res = self.runtime._tour_probe("status", "telegram:1")
        self.assertEqual(res["verdict"], "broken")
        self.assertIn("empty reply", res.get("error", ""))

    def test_tour_probe_accepts_real_reply(self):
        with mock.patch.object(
                self.runtime, "handle_control",
                return_value="status: all good"):
            res = self.runtime._tour_probe("status", "telegram:1")
        self.assertEqual(res["verdict"], "ok")

    def test_tour_probe_skips_game_start_without_dispatching(self):
        seen: list[str] = []

        def _never(text, chat_key, message=None):
            seen.append(text)
            return "started a game"

        with mock.patch.object(self.runtime, "handle_control",
                                side_effect=_never):
            res = self.runtime._tour_probe("hangman", "telegram:1")
        self.assertEqual(res["verdict"], "skip")
        self.assertIn("interactive game", res.get("reason", ""))
        self.assertEqual(seen, [],
                         "tour must never start a game while probing it")
        self.assertTrue(res.get("help_ok"),
                        "/hangman must still have a help page")

    def test_heal_skips_game_start_without_dispatching(self):
        with mock.patch.object(self.runtime, "handle_control",
                                side_effect=AssertionError("must not probe")):
            reply = self.runtime._control_heal("hangman", "telegram:1")
        self.assertIn("interactive game", reply)

    def test_say_routes_gateway_send(self):
        fake = FakeGatewaySend()
        with mock.patch.object(self.runtime, "gateway", fake):
            reply = self.runtime.handle_control(
                "/say telegram:123 hi", "telegram:1", message=None)
        self.assertEqual(reply, "sent.")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0][0], "telegram")
        self.assertEqual(fake.calls[0][2], "hi")

    def test_say_routes_named_voice_tts(self):
        with mock.patch.object(
                self.runtime, "_control_say_tts",
                return_value="spoken") as tts:
            reply = self.runtime.handle_control(
                "/say trump hello there", "telegram:1", message=None)
        self.assertEqual(reply, "spoken")
        tts.assert_called_once_with("trump", "hello there", "telegram:1")

    def test_say_routes_rich_voice_bracket(self):
        with mock.patch.object(
                self.runtime, "_control_say",
                return_value="rich") as rich:
            reply = self.runtime.handle_control(
                "/say [whispers fearfully] don't go in there",
                "telegram:1", message=None)
        self.assertEqual(reply, "rich")
        rich.assert_called_once()

    def test_say_routes_rich_voice_options(self):
        with mock.patch.object(
                self.runtime, "_control_say",
                return_value="rich") as rich:
            reply = self.runtime.handle_control(
                "/say accent=british hello", "telegram:1", message=None)
        self.assertEqual(reply, "rich")
        rich.assert_called_once()


if __name__ == "__main__":
    unittest.main()
