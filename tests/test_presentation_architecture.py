"""Presentation as ARCHITECTURE (owner correction, 2026-11-06).

The owner's words: "it should be a real architecture not you
applying to some places only" — and the riddle is the type case:
``/riddle`` printed ``(/riddle answer reveals it)`` as raw text
while other replies carried real buttons ("some has it some
don't"), and the reveal posted a SECOND message ("the answer
was: …") instead of editing the riddle message.

What this file pins:

(a) **The hint-text guard.** A mechanical AST scan of the whole
    control-reply funnel (``nomorals/agents/partner/runtime*.py``)
    for action-hint shapes — a command affordance printed as
    text where a button belongs. Every remaining hit must be a
    named exception in ``HINT_EXCEPTIONS`` with its reason, and
    every exception must still match a live literal (no silent
    grandfathering, no registry rot). A planted bypass in a
    scratch copy fails the scan (teeth, below).

(b) **Declared actions.** The converted high-traffic handlers
    return ``ActionableMessage`` — buttons declared with the
    text, the hint text gone.

(c) **Live sessions.** The riddle end-to-end per surface shape:
    post once → reveal tap EDITS the same message (zero follow-up
    posts) → Another tap edits it again; a typed fresh /riddle
    posts new; WhatsApp's text-only edit verb is called in its
    own grammar; a refused edit falls back to an honest fresh
    send; a no-edit platform never edits.

(d) **Tap plumbing.** A Telegram bot callback stamps the tapped
    message id into the synthetic message's meta; both WhatsApp
    bridges extract the button response's context stanza so a
    tap reports the message it landed on.

Context homes honour NM_TEST_TMPDIR (see test_surface_laws).
"""

from __future__ import annotations

import ast
import os
import random
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.live_session import (
    LiveSessionRegistry,
    edit_session_message,
    edit_target,
    tapped_message_id,
)
from nomorals.social.chat.tgbot_buttons import ActionableMessage

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

_REPO = Path(__file__).resolve().parents[1]


# ── (a) the hint-text guard ──────────────────────────────────────

#: Action-hint shapes: a command affordance printed as text where
#: a declared button belongs. These are the census patterns.
HINT_PATTERNS: dict[str, re.Pattern] = {
    "paren_hint": re.compile(r"\s\(/[a-z][a-z0-9_]+( [^)]*)?\)"),
    "imperative": re.compile(
        r"\b(type|send|run|tap|press|hit)\s+/[a-z][a-z0-9_]+"),
    "dash_hint": re.compile(
        r"[—–-]\s*/[a-z][a-z0-9_]+ [a-z]+ "
        r"(to|for|reveals|starts|opens)\b"),
    "colon_cmd": re.compile(
        r"\b(approve|deny|reveal|answer|next|retry|confirm|cancel)"
        r":\s*/[a-z]"),
    "backtick_tap": re.compile(r"(?i)tap\s+`/[a-z]"),
}

#: The funnel: every module whose handlers produce control
#: replies. (social/chat modules are organs, catalog data and
#: transports — their strings are data the funnel renders, or
#: documentation; the catalog is /help's job by design.)
FUNNEL_GLOB = "nomorals/agents/partner/runtime*.py"

#: Named exceptions: (module basename, literal key, reason). An
#: exception is legitimate ONLY when a button genuinely cannot
#: carry the action — the argument is the user's own data, the
#: action belongs to another surface/room, the recipient cannot
#: run the command, the copy is documentation/provenance, or the
#: reply never crosses the control funnel. Anything else is a
#: bypass and fails the guard.
HINT_EXCEPTIONS: list[tuple[str, str, str]] = [
    ("runtime.py", "buttons don't exist on this surface",
     "the presentation layer's OWN no-button-surface fallback "
     "copy (numbered-pick rewrite lives beside it); /help is the "
     "content, not an affordance"),
    ("runtime.py", "will not retry on its own",
     "update-recovery owner NOTICE on the alert seam (title/body "
     "tuple), not a control reply; /update after a crash is the "
     "owner's deliberate manual step"),
    ("runtime_chatops.py", "run /gcstatus inside the group",
     "room-scoped action: a button here would execute it in the "
     "DM the reply is read in — the wrong room by construction"),
    ("runtime_games.py", "duel queue spot expired",
     "§34 held notice: timer-driven render composed into notice "
     "boards, not a standalone funnel reply"),
    ("runtime_games.py", "promote a build first",
     "argument-carrying: the build id is the user's own data — "
     "no button can invent it"),
    ("runtime_group.py", "in the group log for the admins",
     "provenance in a receipt to the REPORTER, who cannot run "
     "the admin log; the record's location is the content"),
    ("runtime_group.py", "audits run in DM",
     "surface-scoped action: a button would run the audit in "
     "the group the redirect is read in — audits are DM-only "
     "disclosure material"),
    ("runtime_group.py", "the console lives in my DM",
     "surface-scoped action: the console opens in DM; a button "
     "here would run it in the group"),
    ("runtime_group.py", "tap Claim or send /claim",
     "buttons ride this same reply (the Claim button is "
     "authored in the _card call beside the note); the note is "
     "the typed equivalent for button-degraded surfaces"),
    ("runtime_group.py", "crime, geared",
     "primer/documentation copy listing argument-carrying "
     "crime commands (@member, <district>) in a help board"),
    ("runtime_group.py", "my group systems aren't running",
     "§33 gate explanation: the designation path is owner "
     "console configuration; the line is the refusal's "
     "provenance"),
    ("runtime_group.py", "I only take stickers I'm handed here",
     "target-message argument: /steal acts on a replied-to "
     "sticker — a button cannot supply the target message"),
    ("runtime_group.py", "raw activity still counts every message",
     "board annotation explaining two boards' difference — a "
     "cross-reference, not the reply's action"),
    ("runtime_group.py", "the §16 kit:",
     "syntax documentation for an argument-carrying command "
     "(/tag @member <words>)"),
    ("runtime_group.py", "type /console g1",
     "buttons ride this same reply (the console home's group "
     "buttons are in its _card call); the note names the typed "
     "equivalent"),
    ("runtime_group.py", "sensitive replies (/balance)",
     "settings explanation citing an example command — a "
     "cross-reference inside a composite settings card"),
    ("runtime_group.py", "cleanup auto on|off",
     "settings REPORT documentation: the knobs and their "
     "current values, inside the composite cleanup card"),
    ("runtime_group.py", "auto-named forum topics get a proper name",
     "settings board documentation prose (feature "
     "descriptions, example commands) in a composite card"),
    ("runtime_group.py", "into your loadout (/gear)",
     "descriptive cross-reference in a result sentence, not "
     "the reply's action"),
    ("runtime_group.py", "(/flag reviews them)",
     "one status line inside the composite group-settings "
     "board; the review queue is /flag's own surface"),
    ("runtime_group.py", "in-group war (/war declare)",
     "routing explanation + argument-carrying: declaring "
     "needs the rival gang named by the user"),
    ("runtime_group.py", "declare a side first",
     "argument-carrying: the side is the user's own gang "
     "choice (<gang>)"),
    ("runtime_live.py", "grounded session is empty",
     "argument-carrying: the action's argument is the user's "
     "own documents to attach"),
    ("runtime_media.py", "run /dj long from a chat",
     "no-chat invocation notice: the reply has no surface to "
     "hold a button and the mix must land in a chat the user "
     "picks"),
    ("runtime_media.py", "run /dj mix from a chat",
     "no-chat invocation notice (as /dj long)"),
    ("runtime_media.py", "run /dj live from a chat",
     "no-chat invocation notice (as /dj long)"),
    ("runtime_media.py", "usage: /play remove",
     "usage documentation: the syntax line IS the content "
     "(/help's shape), including its cross-reference"),
    ("runtime_media.py", "couldn't run /musicnote",
     "error REPORT naming the command that failed — a record, "
     "not a next-action affordance"),
    ("runtime_media.py", "went quiet while downloading",
     "async job-outcome push on the direct gateway seam; the "
     "re-queue action needs the original query, which this "
     "seam no longer holds"),
    ("runtime_meta.py", "/game quit to close it",
     "reply FRAGMENT concatenated into a larger toggle reply "
     "by the chat-modes path — not a standalone reply"),
    ("runtime_provision.py", "run /provision with a service name",
     "argument-carrying: the service name is the user's own "
     "data (the line was rewritten bracket-free in the "
     "delivery-shape pass; the exception follows the literal)"),
    ("runtime_provision.py", "toggle is on (/pair toggle)",
     "explanatory line: the toggle takes account + toggle "
     "arguments; the sentence explains when saves land"),
    ("runtime_system.py", "/file <platform> <chat> <path>",
     "argument-carrying next step (platform, chat, path are "
     "the user's own data), on a data-export report"),
    ("runtime_system.py", "set the key in the bot's environment",
     "out-of-chat precondition: the key is set in the bot's "
     "environment first; a button would re-run into the same "
     "wall"),
    ("runtime_system.py", "is not connected — run /connectors connect",
     "raised as an exception message through connector "
     "plumbing and surfaced verbatim by many callers — no "
     "single reply site exists to declare at"),
    ("runtime_voice.py", "auto-tagging: ON by default for /tts",
     "capability documentation listing argument forms "
     "(/tts auto=off <text>) in a TTS report"),
    ("runtime_wisdom.py", "the corpus is empty — run /wisdom seed",
     "rides the wisdom ASK path's own checked long-send seam "
     "(_send_long_checked), which takes text only — the reply "
     "never crosses the control-reply funnel"),
]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    docs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.body \
                and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            docs.add(id(node.body[0].value))
    return docs


def _literal_text(node: ast.Constant) -> str:
    return str(node.value)


def _joined_text(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        else:
            parts.append("{}")
    return "".join(parts)


def scan_hints(root: Path) -> list[tuple[str, int, str, str]]:
    """Every action-hint literal in the funnel under ``root``:
    (module basename, line, pattern name, literal text)."""
    hits: list[tuple[str, int, str, str]] = []
    for path in sorted(root.glob(FUNNEL_GLOB)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstring_nodes(tree)
        inner: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                for value in node.values:
                    inner.add(id(value))
        for node in ast.walk(tree):
            text = ""
            line = 0
            if isinstance(node, ast.Constant) \
                    and isinstance(node.value, str):
                if id(node) in docs or id(node) in inner:
                    continue
                text = _literal_text(node)
                line = node.lineno
            elif isinstance(node, ast.JoinedStr):
                text = _joined_text(node)
                line = node.lineno
            else:
                continue
            for name, rx in HINT_PATTERNS.items():
                if rx.search(text):
                    hits.append((path.name, line, name,
                                 text.replace("\n", " ")))
                    break
    return hits


class HintGuardTests(unittest.TestCase):
    def test_funnel_carries_no_unexcepted_hint_text(self) -> None:
        hits = scan_hints(_REPO)
        bad = []
        for module, line, pat, excerpt in hits:
            if not any(module == mod and key in excerpt
                       for mod, key, _why in HINT_EXCEPTIONS):
                bad.append((module, line, pat, excerpt))
        self.assertEqual(bad, [])

    def test_exceptions_are_not_stale(self) -> None:
        """Every registered exception still matches a live
        literal — the registry cannot quietly rot into a
        blanket grandfather clause."""
        texts: dict[str, list[str]] = {}
        for path in sorted(_REPO.glob(FUNNEL_GLOB)):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docs = _docstring_nodes(tree)
            bucket = texts.setdefault(path.name, [])
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) \
                        and isinstance(node.value, str) \
                        and id(node) not in docs:
                    bucket.append(str(node.value))
                elif isinstance(node, ast.JoinedStr):
                    bucket.append(_joined_text(node))
        for module, key, why in HINT_EXCEPTIONS:
            self.assertTrue(why.strip(), f"empty reason: {key}")
            self.assertTrue(
                any(key in t for t in texts.get(module, [])),
                f"stale exception: {module} :: {key}")

    def test_guard_has_teeth_on_a_planted_bypass(self) -> None:
        """A scratch copy of a funnel module plus one planted
        hint literal: the scan must name the plant."""
        with tempfile.TemporaryDirectory(
                prefix="nm-guard-", dir=_TEST_TMPDIR) as td:
            root = Path(td)
            funnel = root / "nomorals" / "agents" / "partner"
            funnel.mkdir(parents=True)
            src = (_REPO / "nomorals" / "agents" / "partner"
                   / "runtime_fun.py").read_text(encoding="utf-8")
            clean = funnel / "runtime_fun.py"
            clean.write_text(src, encoding="utf-8")
            self.assertEqual(scan_hints(root), [])
            clean.write_text(
                src + "\n\ndef _planted():\n"
                "    return 'the thing is hidden ' \
                '(/zap answer reveals it)'\n",
                encoding="utf-8")
            hits = scan_hints(root)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0][2], "paren_hint")


# ── shared runtime harness (modeled on test_surface_laws) ───────

class BotEditAdapter(ChatAdapter):
    """Telegram-shaped fake: send records; edit_message takes
    buttons + parse_mode and returns the API dict (no ``ok``)."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, Any]] = []
        self.edits: list[tuple[str, str, dict]] = []

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def edit_message(self, chat: ChatRef, message_id: str,
                     text: str, *, origin_command: str = "",
                     buttons: Any = None,
                     parse_mode: str = "") -> dict:
        self.edits.append((message_id, text,
                           {"buttons": buttons,
                            "parse_mode": parse_mode}))
        return {"message_id": int(message_id.lstrip("m") or 0)}


class WaEditAdapter(ChatAdapter):
    """WhatsApp-shaped fake: the bridge edit verb is TEXT-ONLY —
    its signature takes no buttons and no parse mode."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, Any]] = []
        self.edits: list[tuple[str, str]] = []

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def edit_message(self, chat: ChatRef, message_id: str,
                     text: str) -> dict:
        self.edits.append((message_id, text))
        return {"ok": True}


class RefuseEditAdapter(BotEditAdapter):
    """Edit verb present but refusing (expired window, message
    gone): the Bot API shape returns None on any failure."""

    def edit_message(self, chat: ChatRef, message_id: str,
                     text: str, **kw: Any) -> None:
        self.edits.append((message_id, text, dict(kw)))
        return None


class PlainAdapter(ChatAdapter):
    """No edit verb at all (the SMS shape)."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, Any]] = []

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_context() -> tuple[Any, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-",
                                      dir=_TEST_TMPDIR)
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.enabled": "false",
        }
    )
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return context, tmp


class _ArchCase(unittest.TestCase):
    ADAPTERS: dict[str, ChatAdapter] = {}

    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        self.adapters = {
            "telegram-bot": BotEditAdapter("telegram-bot"),
            "telegram": BotEditAdapter("telegram"),
            "whatsapp": WaEditAdapter("whatsapp"),
            "local": PlainAdapter("local"),
        }
        self.gateway = ChatGateway(self.adapters,
                                   db=self.context.db)
        self.runtime = PartnerRuntime(self.context,
                                      gateway=self.gateway)
        self.runtime._typing_enabled = lambda chat: False
        # Deterministic, offline riddle service: no providers,
        # so the shipped bank of classics is the floor.
        from nomorals.social.chat.expression import ProviderHealth
        from nomorals.social.chat.fun import FunService
        self.runtime._fun_service_obj = FunService(
            media_dir=Path(self.tmp.name) / "fun_media", chains={},
            health=ProviderHealth(), rng=random.Random(0))

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.context.close()
        self.tmp.cleanup()

    def _msg(self, platform: str, *, kind: ChatKind = ChatKind.DM,
             chat_id: str = "42", text: str = "",
             reply_to: str = "",
             meta: dict | None = None) -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id=chat_id,
                       kind=kind, title="room", peer=chat_id)
        msg = ChatMessage(chat=chat, incoming=True, text=text,
                          sender="someone", reply_to=reply_to)
        if meta:
            msg.meta.update(meta)
        return msg

    def _drive(self, msg: ChatMessage):
        reply = self.runtime.handle_control(
            msg.text, msg.chat.key, message=msg)
        return self.runtime._send_control_reply(msg, reply), reply


# ── (c) live sessions: the riddle, end to end ────────────────────

class RiddleLiveEditTests(_ArchCase):
    def _open_riddle(self, platform: str = "telegram-bot",
                     chat_id: str = "42"):
        msg = self._msg(platform, chat_id=chat_id, text="/riddle")
        result, reply = self._drive(msg)
        adapter = self.adapters[platform]
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        return adapter, result.message_id

    def test_riddle_posts_once_with_declared_buttons(self) -> None:
        adapter, mid = self._open_riddle()
        self.assertEqual(len(adapter.sent), 1)
        text, buttons = adapter.sent[0]
        self.assertIn("🧩", text)
        self.assertNotIn("reveals it", text)
        self.assertNotIn("(/riddle", text)
        flat = [data for row in (buttons or [])
                for _label, data in row]
        self.assertIn("riddle answer", flat)
        self.assertIn("riddle", flat)
        registry = self.runtime._live_session_registry()
        key = ChatRef(platform="telegram-bot", chat_id="42",
                      kind=ChatKind.DM).key
        self.assertEqual(registry.lookup(key, "riddle"), mid)

    def test_reveal_tap_edits_the_same_message(self) -> None:
        adapter, mid = self._open_riddle()
        question_text = adapter.sent[0][0]
        question = question_text.split("\n")[0].removeprefix("🧩 ")
        tap = self._msg(
            "telegram-bot", text="/riddle answer",
            meta={"callback_query": True, "tapped_message_id": mid})
        result, _reply = self._drive(tap)
        self.assertTrue(result.ok)
        self.assertEqual(result.message_id, mid)
        self.assertEqual(len(adapter.sent), 1,
                         "the reveal posted a follow-up message")
        self.assertEqual(len(adapter.edits), 1)
        edit_mid, edit_text, edit_kw = adapter.edits[0]
        self.assertEqual(edit_mid, mid)
        self.assertIn(question, edit_text)
        self.assertIn("the answer:", edit_text)
        flat = [data for row in (edit_kw["buttons"] or [])
                for _label, data in row]
        self.assertEqual(flat, ["riddle"])

    def test_another_tap_edits_again_no_new_post(self) -> None:
        adapter, mid = self._open_riddle()
        tap = self._msg(
            "telegram-bot", text="/riddle answer",
            meta={"callback_query": True, "tapped_message_id": mid})
        self._drive(tap)
        tap2 = self._msg(
            "telegram-bot", text="/riddle",
            meta={"callback_query": True, "tapped_message_id": mid})
        result, _ = self._drive(tap2)
        self.assertTrue(result.ok)
        self.assertEqual(len(adapter.sent), 1)
        self.assertEqual(len(adapter.edits), 2)
        self.assertIn("🧩", adapter.edits[1][1])
        self.assertNotIn("the answer:", adapter.edits[1][1])

    def test_typed_fresh_riddle_posts_a_new_message(self) -> None:
        adapter, mid = self._open_riddle()
        msg = self._msg("telegram-bot", text="/riddle")
        result, _ = self._drive(msg)
        self.assertTrue(result.ok)
        self.assertEqual(len(adapter.sent), 2)
        self.assertNotEqual(result.message_id, mid)
        self.assertEqual(adapter.edits, [])

    def test_typed_answer_edits_the_live_message(self) -> None:
        adapter, mid = self._open_riddle()
        msg = self._msg("telegram-bot", text="/riddle answer")
        result, _ = self._drive(msg)
        self.assertTrue(result.ok)
        self.assertEqual(len(adapter.sent), 1)
        self.assertEqual(len(adapter.edits), 1)
        self.assertEqual(adapter.edits[0][0], mid)

    def test_whatsapp_tap_edits_text_only_via_reply_to(self) -> None:
        adapter, mid = self._open_riddle(
            platform="whatsapp", chat_id="15551234567")
        tap = self._msg("whatsapp", chat_id="15551234567",
                        text="/riddle answer", reply_to=mid)
        result, _ = self._drive(tap)
        self.assertTrue(result.ok)
        self.assertEqual(len(adapter.sent), 1)
        self.assertEqual(len(adapter.edits), 1)
        edit_mid, edit_text = adapter.edits[0]
        self.assertEqual(edit_mid, mid)
        self.assertIn("the answer:", edit_text)

    def test_refused_edit_falls_back_to_a_fresh_send(self) -> None:
        self.adapters["telegram-bot"] = RefuseEditAdapter(
            "telegram-bot")
        self.gateway = ChatGateway(self.adapters,
                                   db=self.context.db)
        self.runtime.gateway = self.gateway
        adapter, mid = self._open_riddle()
        tap = self._msg(
            "telegram-bot", text="/riddle answer",
            meta={"callback_query": True, "tapped_message_id": mid})
        result, _ = self._drive(tap)
        self.assertTrue(result.ok)
        self.assertEqual(len(adapter.edits), 1)   # attempted
        self.assertEqual(len(adapter.sent), 2)    # fell back
        self.assertIn("the answer:", adapter.sent[1][0])
        self.assertNotEqual(result.message_id, mid)

    def test_no_edit_platform_never_edits(self) -> None:
        msg = self._msg("local", chat_id="u1", text="/riddle")
        result, reply = self._drive(msg)
        self.assertTrue(result.ok)
        adapter = self.adapters["local"]
        self.assertEqual(len(adapter.sent), 1)
        msg2 = self._msg("local", chat_id="u1",
                         text="/riddle answer")
        result2, _ = self._drive(msg2)
        self.assertTrue(result2.ok)
        self.assertEqual(len(adapter.sent), 2)
        self.assertIn("the answer:", adapter.sent[1][0])

    def test_riddle_answer_with_none_open_declares_new(self) -> None:
        msg = self._msg("telegram-bot", text="/riddle answer")
        _result, reply = self._drive(msg)
        self.assertIsInstance(reply, ActionableMessage)
        self.assertIn("no riddle is open", reply.text)
        flat = [data for row in reply.buttons
                for _label, data in row]
        self.assertEqual(flat, ["riddle"])


# ── (b) declared actions on converted handlers ───────────────────

class DeclaredActionTests(_ArchCase):
    def _reply(self, platform: str, text: str, chat_id: str = "42"):
        # is_owner is the gateway-stamped owner verdict the
        # runtime's operator test reads (owner tools are
        # owner-only by law).
        msg = self._msg(platform, chat_id=chat_id, text=text,
                        meta={"is_owner": True})
        return self.runtime.handle_control(
            text, msg.chat.key, message=msg)

    # Driven on the WhatsApp personal surface: the public bot
    # gates owner tools for strangers by design (register law),
    # and the DECLARED reply object is surface-independent.

    def _assert_declared(self, reply, want_data: str,
                         banned: str) -> None:
        self.assertIsInstance(reply, ActionableMessage)
        flat = [data for row in reply.buttons
                for _label, data in row]
        self.assertIn(want_data, flat)
        self.assertNotIn(banned, reply.text)

    def test_mission_not_found(self) -> None:
        reply = self._reply("whatsapp",
                            "/mission status zzz-no-match")
        self._assert_declared(reply, "mission list", "/mission")

    def test_monitor_not_found(self) -> None:
        reply = self._reply("whatsapp", "/monitor rm zzz")
        self._assert_declared(reply, "monitor list", "/monitor")

    def test_schedule_not_found(self) -> None:
        reply = self._reply("whatsapp", "/schedule rm zzz")
        self._assert_declared(reply, "schedule list", "/schedule")

    def test_course_unknown(self) -> None:
        reply = self._reply("whatsapp", "/course teach zzz")
        self._assert_declared(reply, "course list", "/course")

    def test_news_empty(self) -> None:
        reply = self._reply("whatsapp", "/news status")
        self._assert_declared(reply, "news run", "/news")

    def test_unknown_game(self) -> None:
        reply = self._reply("whatsapp", "/game zzznope")
        self._assert_declared(reply, "game list", "/game")

    def test_confirm_prompt_declares_its_buttons(self) -> None:
        reply = self._reply("whatsapp",
                            "/forget some old memory")
        self.assertIsInstance(reply, ActionableMessage)
        flat = [data for row in reply.buttons
                for _label, data in row]
        self.assertIn("forget confirm", flat)
        self.assertIn("forget cancel", flat)
        self.assertTrue(reply.text.startswith("⚠️ confirm: /forget"))
        self.assertNotIn("Send /forget confirm", reply.text)


# ── live_session units ───────────────────────────────────────────

class LiveSessionUnitTests(unittest.TestCase):
    def test_registry_roundtrip(self) -> None:
        reg = LiveSessionRegistry()
        reg.record("telegram-bot:42", "riddle", "m1")
        self.assertEqual(reg.lookup("telegram-bot:42", "riddle"),
                         "m1")
        reg.record("telegram-bot:42", "riddle", "m2")
        self.assertEqual(reg.lookup("telegram-bot:42", "riddle"),
                         "m2")
        reg.forget("telegram-bot:42", "riddle")
        self.assertEqual(reg.lookup("telegram-bot:42", "riddle"),
                         "")
        reg.record("", "riddle", "m1")
        self.assertEqual(reg.lookup("", "riddle"), "")

    def _fake_msg(self, *, meta=None, reply_to=""):
        chat = ChatRef(platform="telegram-bot", chat_id="42",
                       kind=ChatKind.DM)
        msg = ChatMessage(chat=chat, incoming=True, text="/x",
                          sender="s", reply_to=reply_to)
        if meta:
            msg.meta.update(meta)
        return msg

    def test_edit_target_prefers_the_tap(self) -> None:
        reg = LiveSessionRegistry()
        reg.record("telegram-bot:42", "riddle", "m1")
        msg = self._fake_msg(meta={"tapped_message_id": "m9"})
        self.assertEqual(
            edit_target(reg, msg, "riddle", session_edit=False),
            "m9")

    def test_edit_target_whatsapp_reply_to_match(self) -> None:
        reg = LiveSessionRegistry()
        reg.record("whatsapp:1", "riddle", "m1")
        chat = ChatRef(platform="whatsapp", chat_id="1",
                       kind=ChatKind.DM)
        msg = ChatMessage(chat=chat, incoming=True, text="/x",
                          sender="s", reply_to="m1")
        self.assertEqual(
            edit_target(reg, msg, "riddle", session_edit=False),
            "m1")
        msg2 = ChatMessage(chat=chat, incoming=True, text="/x",
                           sender="s", reply_to="other")
        self.assertEqual(
            edit_target(reg, msg2, "riddle", session_edit=False),
            "")

    def test_edit_target_continuation_uses_record(self) -> None:
        reg = LiveSessionRegistry()
        reg.record("telegram-bot:42", "riddle", "m1")
        msg = self._fake_msg()
        self.assertEqual(
            edit_target(reg, msg, "riddle", session_edit=True),
            "m1")
        self.assertEqual(
            edit_target(reg, msg, "riddle", session_edit=False),
            "")

    def test_tapped_message_id_reads_meta(self) -> None:
        msg = self._fake_msg(meta={"tapped_message_id": "m7"})
        self.assertEqual(tapped_message_id(msg), "m7")
        self.assertEqual(tapped_message_id(self._fake_msg()), "")

    def test_edit_dispatch_shapes(self) -> None:
        chat = ChatRef(platform="whatsapp", chat_id="1",
                       kind=ChatKind.DM)

        class _WaVerb:
            def __init__(self):
                self.calls = []

            def edit_message(self, chat_, mid, text):
                self.calls.append((mid, text))
                return {"ok": True}

        wa = _WaVerb()
        self.assertTrue(edit_session_message(
            wa, chat, "m1", "hello", buttons=[[("A", "a")]],
            parse_mode="HTML"))
        self.assertEqual(wa.calls, [("m1", "hello")])

        class _BotVerb:
            def __init__(self):
                self.calls = []

            def edit_message(self, chat_, mid, text, *,
                             buttons=None, parse_mode=""):
                self.calls.append((mid, text, buttons, parse_mode))
                return {"message_id": 5}

        bot = _BotVerb()
        self.assertTrue(edit_session_message(
            bot, chat, "m2", "hi", buttons=[[("A", "a")]],
            parse_mode="HTML"))
        self.assertEqual(bot.calls[0][2], [[("A", "a")]])
        self.assertEqual(bot.calls[0][3], "HTML")

        class _Refusing:
            def edit_message(self, chat_, mid, text):
                return {"ok": False}

        self.assertFalse(edit_session_message(
            _Refusing(), chat, "m1", "x"))
        self.assertFalse(edit_session_message(
            object(), chat, "m1", "x"))
        self.assertFalse(edit_session_message(
            wa, chat, "", "x"))


# ── (d) tap plumbing ─────────────────────────────────────────────

class TapPlumbingTests(unittest.TestCase):
    def test_bot_callback_stamps_tapped_message_id(self) -> None:
        from nomorals.social.chat.telegram import TelegramBotAdapter

        class _Resp:
            def json(self):
                return {"ok": True, "result": True}

        class _Session:
            def post_json(self, url, params, timeout=None):
                return _Resp()

            def get(self, url, timeout=None):
                return _Resp()

        adapter = TelegramBotAdapter(token="t0ken",
                                     session=_Session())
        query = {
            "id": "q1",
            "from": {"id": 7, "is_bot": False,
                     "first_name": "Owner"},
            "message": {"chat": {"id": 42, "type": "private"},
                        "message_id": 555},
            "data": adapter._sign_callback("riddle answer"),
        }
        received: list = []
        adapter._handle_callback(query, received.append)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "/riddle answer")
        self.assertEqual(
            received[0].meta.get("tapped_message_id"), "555")
        self.assertTrue(received[0].meta.get("callback_query"))

    def test_userbot_callback_stamps_tapped_message_id(self) -> None:
        import asyncio

        from nomorals.social.chat.telegram import TelegramAdapter

        class _Event:
            data = b"riddle answer"
            chat_id = "42"
            sender_id = "7"
            message_id = 777

            async def answer(self):
                return None

            async def get_chat(self):
                return None

            async def get_sender(self):
                return None

        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.partner_context = False
        adapter.stats = {"received": 0, "sent": 0,
                         "send_errors": 0, "started_at": 0.0}
        received: list = []
        asyncio.run(adapter._handle_callback_query(
            _Event(), received.append))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "/riddle answer")
        self.assertEqual(
            received[0].meta.get("tapped_message_id"), "777")

    def test_bridges_extract_button_response_context(self) -> None:
        go_src = (_REPO / "bridge" / "gobridge"
                  / "messages.go").read_text(encoding="utf-8")
        for needle in (
                "GetButtonsResponseMessage().GetContextInfo()",
                "GetTemplateButtonReplyMessage().GetContextInfo()",
                "GetListResponseMessage().GetContextInfo()",
                "GetInteractiveResponseMessage().GetContextInfo()"):
            self.assertIn(needle, go_src)
        mjs = (_REPO / "bridge"
               / "whatsapp-bridge.mjs").read_text(encoding="utf-8")
        for needle in (
                "buttonsResponseMessage?.contextInfo",
                "templateButtonReplyMessage?.contextInfo",
                "listResponseMessage?.contextInfo",
                "interactiveResponseMessage?.contextInfo"):
            self.assertIn(needle, mjs)


# ── (e) markup integrity: the paged-presentation leak ────────────
#
# Owner screenshot, 2026-11-06: Telegram bot DM, help page 2 of 2
# printed ``<i>Powered by peacethefirst</i> (2/2)`` as literal
# text. The chain: help prose carries literal ``<command>`` /
# ``<topic>`` placeholders Telegram HTML rejects → the API refuses
# the entity parse → the bot adapter's plain fallback dropped the
# parse mode but shipped the chunk's text UNCHANGED, tags and all.
# The layer's answers, pinned here: repair at the gateway (markup
# valid before chunking, one discipline for every page) and a
# fallback that CONVERTS (strips) instead of merely un-moding.


def _tag_violations(text: str) -> list[str]:
    """format_conformance violations from the HTML-tag family."""
    from nomorals.social.format_conformance import check_message

    return [v for v in check_message("telegram-bot", text)
            if "html-tag" in v or "href" in v]


class RepairTelegramHtmlTests(unittest.TestCase):
    def test_unsupported_placeholders_become_literal_text(self):
        from nomorals.social.format_conformance import repair_telegram_html

        out = repair_telegram_html(
            "run `nm <command>` — `nm help <topic>` shows the pages, "
            "<b>bold</b> stays bold")
        self.assertIn("&lt;command&gt;", out)
        self.assertIn("&lt;topic&gt;", out)
        self.assertIn("<b>bold</b>", out)
        self.assertEqual(_tag_violations(out), [])

    def test_noop_on_valid_html(self):
        from nomorals.social.format_conformance import repair_telegram_html

        for text in (
                "<i>Powered by peacethefirst</i>",
                "<b>already</b> formatted &amp; ready",
                "**raw** stays raw",
                '<a href="https://x.test/?a=1">link</a>',
                "<pre>code &lt; sample</pre>",
                "plain prose, no markup at all"):
            self.assertEqual(repair_telegram_html(text), text, text)

    def test_balance_restored_and_strays_escaped(self):
        from nomorals.social.format_conformance import repair_telegram_html

        self.assertEqual(repair_telegram_html("<b>bold"),
                         "<b>bold</b>")
        self.assertEqual(repair_telegram_html("<b><i>x</b>"),
                         "<b><i>x</i></b>")
        self.assertEqual(repair_telegram_html("a </i> b"),
                         "a &lt;/i&gt; b")
        self.assertEqual(repair_telegram_html("1 < 2 & 3"),
                         "1 &lt; 2 &amp; 3")
        self.assertEqual(
            repair_telegram_html("<a>nohref</a>"),
            "&lt;a&gt;nohref&lt;/a&gt;")

    def test_repair_is_idempotent(self):
        from nomorals.social.format_conformance import repair_telegram_html

        text = ("nm <command> & <b>bold <i>both</b> tail "
                "<i>Powered by peacethefirst</i>")
        once = repair_telegram_html(text)
        self.assertEqual(repair_telegram_html(once), once)
        self.assertEqual(_tag_violations(once), [])


class _RecordingAdapter(ChatAdapter):
    """Gateway-level fake capturing (text, parse_mode) per send."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.calls.append((text, parse_mode))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.calls)}")


class PagedHelpMarkupTests(unittest.TestCase):
    """The evidence, end to end at the gateway: the REAL detailed
    help overview + the layer's stamp, sent as ONE paged HTML
    presentation — every page carries the mode, every page is
    valid for the surface, the placeholders display as text and
    the stamp renders as markup exactly once."""

    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        self.adapter = _RecordingAdapter("telegram-bot")
        self.gateway = ChatGateway(
            {"telegram-bot": self.adapter}, db=self.context.db)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_paged_help_every_page_one_discipline(self):
        from nomorals.social.chat.control import _detailed_overview
        from nomorals.social.chat.typography import apply_stamp

        text = apply_stamp(_detailed_overview())
        self.assertIn("<command>", text)   # the offending prose
        # Presentation Phase 3: the overview is a menu-class Card
        # and stamps itself in the canonical dialect — apply_stamp
        # is a no-op on it by design (never double-stamps).  The
        # input is house dialect now, not authored HTML, so no
        # parse mode is declared here — the gateway converts it,
        # the same path _send_control_reply takes for this text
        # (no <b>/<code>/<i> in it → parse_mode "").  The HTML
        # form of the stamp appears only after that conversion,
        # asserted on the joined pages below.
        self.assertIn("_Powered by peacethefirst_", text)
        chat = ChatRef(platform="telegram-bot", chat_id="42",
                       kind=ChatKind.DM)
        result = self.gateway.send("telegram-bot", chat, text,
                                   fresh=True)
        self.assertTrue(result.ok)
        calls = self.adapter.calls
        self.assertGreaterEqual(len(calls), 2,
                                "the overview must actually page")
        for chunk, mode in calls:
            self.assertEqual(mode, "HTML",
                             "every page carries the parse mode")
            self.assertEqual(_tag_violations(chunk), [],
                             f"page invalid for the surface: {chunk[:80]}")
            self.assertNotIn("<command>", chunk)
            self.assertNotIn("<topic>", chunk)
        joined = "".join(chunk for chunk, _mode in calls)
        self.assertIn("&lt;command&gt;", joined,
                      "the placeholder displays as literal text")
        self.assertEqual(
            joined.count("<i>Powered by peacethefirst</i>"), 1,
            "the stamp renders as markup, once")


class BotAdapterPlainFallbackTests(unittest.TestCase):
    """When the API still rejects a chunk's entities, the bot
    adapter's fallback CONVERTS the text (tags stripped) — for
    the rejected chunk and every chunk after it. Shipping the
    raw chunk with only the mode dropped is the leak."""

    def _adapter(self):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        ad = TelegramBotAdapter(token="<redacted>", session=object())
        calls: list[dict] = []

        def fake_api(method, **params):
            calls.append(dict(params))
            if params.get("parse_mode"):
                raise Exception(
                    "Bad Request: can't parse entities: Unsupported "
                    "start tag \"command\" at byte offset 5")
            return {"message_id": 100 + len(calls)}

        ad._api = fake_api  # type: ignore[method-assign]
        return ad, calls

    def test_entity_rejection_converts_every_chunk(self):
        ad, calls = self._adapter()
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        text = ("run `nm <command>` for <b>bold</b> detail. "
                + "filler words go here. " * 260
                + "the end. <i>Powered by peacethefirst</i>")
        result = ad.send(chat, text, parse_mode="HTML")
        self.assertTrue(result.ok)
        self.assertGreaterEqual(len(calls), 3,
                                "reject + converted retry + chunk 2")
        # The first call is the formatted attempt; every call after
        # it is plain AND tag-free.
        self.assertEqual(calls[0].get("parse_mode"), "HTML")
        for params in calls[1:]:
            self.assertNotIn("parse_mode", params)
            body = params.get("text", "")
            for tag in ("<b>", "</b>", "<i>", "</i>", "<command>"):
                self.assertNotIn(tag, body)
        self.assertIn("Powered by peacethefirst", calls[-1]["text"],
                      "the stamp survives as readable plain text")


class PresentedFamilyModeGuardTests(unittest.TestCase):
    """The presented-output family in both modes: a stamped Card
    renders as HTML with the mode ON for the Telegram family and
    as tag-free plain text (mode off) for a plain surface."""

    def test_card_renders_both_modes(self):
        from nomorals.social.chat.typography import Card, render_card

        card = Card("menu", title="Games")
        card.line("pick a table to play")
        tg = render_card(card, platform="telegram-bot")
        self.assertEqual(tg.parse_mode, "HTML")
        self.assertIn("<i>Powered by peacethefirst</i>", tg.text)
        self.assertEqual(_tag_violations(tg.text), [])
        plain = render_card(card, platform="local")
        self.assertEqual(plain.parse_mode, "")
        self.assertIn("Powered by peacethefirst", plain.text)
        self.assertNotIn("<i>", plain.text)
        self.assertNotIn("<b>", plain.text)


# ── (e2) the menu family: one structure, every platform ─────────
#
# Owner, 2026-11-06: "The formatting for the menus across all
# platforms even." The catalog menu existed as three systems:
# WhatsApp's XMD frame over the registry, Telegram's hand-parsed
# _HELP_TEXT render through a separate style path, and the paged
# /help prose that leaked. Now: ONE declared structure
# (control.menu_sections — the registry, ordered/titled once),
# rendered through the shared typography layer per platform.
# WhatsApp's rendered menu is byte-identical to before (its
# structure became the canonical one); Telegram converged.


class MenuFamilyGuardTests(unittest.TestCase):
    def test_sections_parity_across_platforms(self):
        from nomorals.social.chat.control import WHATSAPP_UNSUPPORTED_KINDS, menu_sections

        tg = menu_sections("telegram")
        wa = menu_sections("whatsapp")
        # WhatsApp's one platform-truth extra leads: the
        # plain-words group capabilities exist only on its bridge.
        self.assertEqual(wa[0][1], "groups — in plain words")
        # Same sections, same order, same titles after that.
        self.assertEqual([t for _e, t, _i in tg],
                         [t for _e, t, _i in wa[1:]])
        # Item parity modulo exactly the platform-impossible kinds.
        tg_cmds = {c for _e, _t, items in tg for c, _d in items}
        wa_cmds = {c for _e, _t, items in wa[1:] for c, _d in items}
        self.assertEqual(
            tg_cmds - wa_cmds,
            {f"/{k}" for k in WHATSAPP_UNSUPPORTED_KINDS})
        self.assertEqual(wa_cmds - tg_cmds, set())

    def test_telegram_menu_is_a_layer_render(self):
        from nomorals.social.chat.control import help_menu

        out = help_menu("telegram")
        # The layer's stamp, styled — not a literal, not missing.
        self.assertIn("<i>Powered by peacethefirst</i>", out)
        self.assertEqual(_tag_violations(out), [])
        # Every declared section renders, emoji-headed, in order.
        from nomorals.social.chat.control import menu_sections

        pos = -1
        for emoji, title, _items in menu_sections("telegram"):
            head = f"<b>{emoji} {title}</b>".replace(
                "&", "&amp;")
            nxt = out.find(head)
            self.assertGreater(nxt, pos,
                               f"section head missing/out of order: {head}")
            pos = nxt

    def test_whatsapp_menu_plain_stamped_framed(self):
        from nomorals.social.chat.control import menu_sections, whatsapp_menu
        from nomorals.social.chat.style import small_caps

        out = whatsapp_menu()
        for tag in ("<b>", "</b>", "<i>", "</i>", "<code>",
                    "</code>", "<pre>", "</pre>"):
            self.assertNotIn(tag, out,
                             "no HTML survives into the WhatsApp render")
        self.assertTrue(
            out.rstrip().endswith("_Powered by peacethefirst_"),
            "the stamp rides the WhatsApp render, styled")
        self.assertEqual(out.count("╭━━〔"),
                         out.count("╰━━━━━━━━━━━━┈⊷"),
                         "frames balanced")
        for _emoji, title, _items in menu_sections("whatsapp"):
            self.assertIn(small_caps(title), out,
                          f"section missing from WA render: {title}")

    def test_public_menu_is_markup_free(self):
        from nomorals.social.chat.control import public_menu_text

        out = public_menu_text()
        for tag in ("<b>", "</b>", "<i>", "</i>", "<code>"):
            self.assertNotIn(tag, out)


class PagedTelegramMenuTests(unittest.TestCase):
    """The unified Telegram catalog menu, paged through the real
    gateway: at 55K it spans many pages — every page carries the
    HTML discipline and stays valid for the surface."""

    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        self.adapter = _RecordingAdapter("telegram-bot")
        self.gateway = ChatGateway(
            {"telegram-bot": self.adapter}, db=self.context.db)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_menu_pages_carry_the_mode(self):
        from nomorals.social.chat.control import help_menu

        text = help_menu("telegram")
        chat = ChatRef(platform="telegram-bot", chat_id="42",
                       kind=ChatKind.DM)
        result = self.gateway.send("telegram-bot", chat, text,
                                   parse_mode="HTML", fresh=True)
        self.assertTrue(result.ok)
        calls = self.adapter.calls
        self.assertGreaterEqual(len(calls), 4,
                                "the catalog menu must actually page")
        for chunk, mode in calls:
            self.assertEqual(mode, "HTML")
            self.assertEqual(_tag_violations(chunk), [],
                             f"page invalid for the surface: {chunk[:80]}")
        joined = "".join(chunk for chunk, _mode in calls)
        self.assertEqual(
            joined.count("<i>Powered by peacethefirst</i>"), 1,
            "the stamp renders once across all pages")


# ── (f) games output: the settle composition, fixed as a class ───
#
# Owner screenshot, 2026-11-06: two /slots 20 plays in a group,
# each a fresh wall printing the provably-fair commitment hash
# TWICE, a dotted-glyph XP bar, and "/game rematch" as text.
# The five corrections, pinned: no hash in any chat render ·
# the fairness note at most once per result · rematch/fairness
# are declared buttons · the bar is the typography layer's bar ·
# slots rides the live-session registry (post once, edits after).


def _engine_rig():
    from nomorals.games.engine import GameEngine
    from nomorals.storage.db import Database

    class Ctx:
        def __init__(self, db):
            self.db = db

    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


class SettleCompositionClassTests(unittest.TestCase):
    """The shared builder (engine start + _finish_inner + the
    final card) carries the corrections for EVERY game in the
    fairness class — slots, blackjack and roulette played through,
    poker and craps through the same settle path (quit)."""

    def setUp(self) -> None:
        from nomorals.games.players import Player

        self.engine, self.db, self.sent = _engine_rig()
        self.ada = Player.from_sender("telegram", "111", "Ada")
        self.engine.store.add_coins(self.ada, 500,
                                    reason="test funding")

    def tearDown(self) -> None:
        self.engine.shutdown()

    def _assert_clean_composition(self, start_msgs, settle_msgs,
                                  commit: str) -> None:
        start = "\n".join(start_msgs)
        settle = "\n".join(settle_msgs)
        for body in (start, settle):
            self.assertNotIn(commit, body, "the hash never prints")
            self.assertNotIn(commit[:12], body,
                             "not even a hash prefix prints")
            self.assertNotIn("commitment `", body)
            self.assertNotIn("/game rematch", body)
            self.assertNotIn("run it back", body)
        self.assertEqual(start.count("provably fair"), 1,
                         "the note rides the table open, once")
        self.assertEqual(settle.count("provably fair"), 1,
                         "the note rides the settle, once")

    def test_slots_played_through(self):
        engine, ada = self.engine, self.ada
        room, msgs = engine.start("test:sl", "slots", ada, kind="dm")
        commit = room.state["fair"]["commit"]
        engine.move("test:sl", "bet 20", ada, kind="dm")
        out = engine.move("test:sl", "spin", ada, kind="dm")
        self._assert_clean_composition(msgs, out, commit)
        bar = re.search(r"`[█░]{10}` \d+% \d+/\d+ \(lvl \d+\)",
                        "\n".join(out))
        self.assertIsNotNone(bar,
                             "the settle's XP bar is the typography bar")

    def test_blackjack_played_through(self):
        engine, ada = self.engine, self.ada
        room, msgs = engine.start("test:bj", "blackjack", ada,
                                  kind="dm")
        commit = room.state["fair"]["commit"]
        out = engine.move("test:bj", "stand", ada, kind="dm")
        self.assertIsNone(engine.live("test:bj"))
        self._assert_clean_composition(msgs, out, commit)

    def test_roulette_played_through(self):
        engine, ada = self.engine, self.ada
        room, msgs = engine.start("test:rl", "roulette", ada,
                                  kind="dm")
        commit = room.state["fair"]["commit"]
        engine.move("test:rl", "bet red", ada, kind="dm")
        out = engine.move("test:rl", "spin", ada, kind="dm")
        self.assertIsNone(engine.live("test:rl"))
        self._assert_clean_composition(msgs, out, commit)

    def test_poker_and_craps_share_the_settle(self):
        engine, ada = self.engine, self.ada
        for game in ("poker", "craps"):
            room, msgs = engine.start(f"test:{game}", game, ada,
                                      kind="dm")
            commit = room.state["fair"]["commit"]
            out = engine.quit(f"test:{game}")
            self._assert_clean_composition(msgs, out, commit)

    def test_nonfair_card_has_no_text_rematch(self):
        from nomorals.games.players import Player

        engine = self.engine
        p = Player(key="local:tt", platform="local", name="Tt")
        room, _ = engine.start("test:ttt", "ttt", p)
        room.state["done"] = True
        out = engine._finish(room)
        joined = "\n".join(out)
        self.assertIn("🏁", joined)
        self.assertNotIn("/game rematch", joined)
        self.assertNotIn("run it back", joined)


class SlotsLiveSessionTests(_ArchCase):
    """Slots end to end on Telegram group shapes through the real
    intake (_process_inner -> the any-chat game dispatch -> the
    funnel): the first play posts the session message with
    declared actions; a replay EDITS it; a rematch tap edits it
    again into the new table; typed /game rematch still works;
    WhatsApp's reply_to-shaped edit grammar falls back honestly.
    """

    def _gmsg(self, platform: str, *, chat_id: str, text: str,
              meta: dict | None = None) -> ChatMessage:
        """A group message with a REAL sender shape: group players
        resolve from the numeric sender_id (a name-only group
        sender is refused a profile by design)."""
        msg = self._msg(platform, kind=ChatKind.GROUP,
                        chat_id=chat_id, text=text, meta=meta)
        msg.sender_id = "111"
        return msg

    def _fund(self, msg: ChatMessage, coins: int = 1000):
        """Fund BOTH purses a slots table can stake: the games
        wallet (DM rooms) and the group estate (group rooms stake
        the estate — member id is the sender id)."""
        engine = self.runtime._game_engine()
        player = self.runtime._game_player(msg)
        engine.store.add_coins(player, coins, reason="test funding")
        try:
            self.runtime._group_surface().game_wager_credit(
                msg.chat.key, "111", coins, "test funding")
        except Exception as exc:  # noqa: BLE001 - DM keys have no estate
            assert isinstance(exc, Exception)
        return engine

    def _play(self, msg: ChatMessage) -> None:
        self.runtime._process_inner(msg)

    def _commit_for(self, engine, chat_key: str) -> str:
        reveal = engine.last_fair_reveal(chat_key)
        self.assertIsNotNone(reveal)
        m = re.search(r"[0-9a-f]{64}", reveal or "")
        self.assertIsNotNone(m, "the reveal keeps the commitment")
        return m.group(0)  # type: ignore[union-attr]

    def _assert_result_text(self, text: str, commit: str) -> None:
        self.assertIn("🏁", text)
        self.assertNotIn(commit, text)
        self.assertNotIn(commit[:12], text)
        self.assertNotIn("commitment `", text)
        self.assertEqual(text.count("provably fair"), 1,
                         "one fairness note per result")
        self.assertNotIn("/game rematch", text)
        self.assertNotIn("run it back", text)
        self.assertIsNotNone(
            re.search(r"`[█░]{10}` \d+% \d+/\d+ \(lvl \d+\)", text),
            "the XP bar is the typography layer's bar")

    def test_first_play_posts_session_with_declared_actions(self):
        msg = self._gmsg("telegram-bot", chat_id="777",
                         text="/game slots 20")
        engine = self._fund(msg)
        self._play(msg)
        adapter = self.adapters["telegram-bot"]
        self.assertEqual(len(adapter.sent), 1)
        self.assertEqual(adapter.edits, [])
        text, buttons = adapter.sent[0]
        commit = self._commit_for(engine, msg.chat.key)
        self._assert_result_text(text, commit)
        flat = [data for row in (buttons or [])
                for _label, data in row]
        self.assertIn("game rematch", flat)
        self.assertIn("game fair", flat)
        self.assertIn("game leaderboard", flat)
        registry = self.runtime._live_session_registry()
        self.assertEqual(registry.lookup(msg.chat.key, "slots"),
                         "m1")

    def test_replay_edits_the_session_message(self):
        msg = self._gmsg("telegram-bot", chat_id="777",
                         text="/game slots 20")
        engine = self._fund(msg)
        self._play(msg)
        adapter = self.adapters["telegram-bot"]
        self.assertEqual(len(adapter.sent), 1)
        self._play(self._gmsg("telegram-bot", chat_id="777",
                              text="/game slots 20"))
        self.assertEqual(len(adapter.sent), 1,
                         "a replay must not post a second wall")
        self.assertEqual(len(adapter.edits), 1)
        edit_mid, edit_text, _kw = adapter.edits[0]
        self.assertEqual(edit_mid, "m1")
        commit = self._commit_for(engine, msg.chat.key)
        self._assert_result_text(edit_text, commit)

    def test_rematch_tap_edits_into_the_new_table(self):
        msg = self._gmsg("telegram-bot", chat_id="777",
                         text="/game slots 20")
        engine = self._fund(msg)
        self._play(msg)
        adapter = self.adapters["telegram-bot"]
        self.assertEqual(len(adapter.sent), 1)
        tap = self._gmsg(
            "telegram-bot", chat_id="777", text="/game rematch",
            meta={"callback_query": True, "tapped_message_id": "m1"})
        result, _reply = self._drive(tap)
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.message_id, "m1")
        self.assertEqual(len(adapter.sent), 1,
                         "the rematch tap must not post fresh")
        self.assertEqual(len(adapter.edits), 1)
        edit_mid, edit_text, _kw = adapter.edits[0]
        self.assertEqual(edit_mid, "m1")
        self.assertIn("provably fair", edit_text)
        self.assertNotIn("commitment `", edit_text)
        room = engine.live(msg.chat.key)
        self.assertIsNotNone(room)
        self.assertEqual(room.game, "slots")

    def test_typed_rematch_still_works(self):
        msg = self._gmsg("telegram-bot", chat_id="888",
                         text="/game slots 20")
        engine = self._fund(msg)
        self._play(msg)
        self.assertIsNone(engine.live(msg.chat.key))
        typed = self._gmsg("telegram-bot", chat_id="888",
                           text="/game rematch")
        result, _reply = self._drive(typed)
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        room = engine.live(msg.chat.key)
        self.assertIsNotNone(room,
                             "typed /game rematch opens the table")
        self.assertEqual(room.game, "slots")

    def test_dm_bare_slots_command_is_the_same_session(self):
        # The canonical entry: "/slots 20" in a DM (no group key,
        # so no group-economy intercept) reaches the same
        # quickstart and declares the same session + actions.
        msg = self._msg("telegram-bot", chat_id="42",
                        text="/slots 20")
        msg.sender_id = "111"
        engine = self._fund(msg)
        self._play(msg)
        adapter = self.adapters["telegram-bot"]
        self.assertEqual(len(adapter.sent), 1)
        text, buttons = adapter.sent[0]
        commit = self._commit_for(engine, msg.chat.key)
        self._assert_result_text(text, commit)
        flat = [data for row in (buttons or [])
                for _label, data in row]
        self.assertIn("game rematch", flat)
        self.assertIn("game fair", flat)
        registry = self.runtime._live_session_registry()
        self.assertEqual(registry.lookup(msg.chat.key, "slots"),
                         "m1")

    def test_whatsapp_replay_edits_text_only(self):
        msg = self._msg("whatsapp", chat_id="15551234567",
                        text="/slots 20")
        msg.sender_id = "15551234567"
        engine = self._fund(msg)
        self._play(msg)
        adapter = self.adapters["whatsapp"]
        self.assertEqual(len(adapter.sent), 1)
        text = adapter.sent[0][0]
        self.assertIn("🏁", text)
        msg2 = self._msg("whatsapp", chat_id="15551234567",
                         text="/slots 20")
        msg2.sender_id = "15551234567"
        self._play(msg2)
        # The session law holds on WhatsApp too, in its own
        # grammar: the bridge edit verb is text-only, so the
        # replay EDITS the live message's text (the original
        # keyboard — Rematch / Fairness / Leaderboard — stays).
        self.assertEqual(len(adapter.sent), 1)
        self.assertEqual(len(adapter.edits), 1)
        edit_mid, edit_text = adapter.edits[0]
        self.assertEqual(edit_mid, "m1")
        commit = self._commit_for(engine, msg.chat.key)
        self._assert_result_text(edit_text, commit)


if __name__ == "__main__":
    unittest.main()
