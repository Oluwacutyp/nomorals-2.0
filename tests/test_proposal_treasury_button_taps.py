"""Button-fix lane 3 — proposals Deny + treasury "Second it",
tap-tested BY EFFECT.

The button-effect audit (~/workspace/button-audit/AUDIT.md S2/S3)
proved two action buttons that could not do their job:

* **S2 — Proposals Deny.** The Telegram /proposals list's Deny
  button carries ``deny <id>`` and nothing else — a callback
  payload has no room for a reason — but the upgrade queue's
  denial REQUIRES one, so the tap could never deny an upgrade
  proposal (it answered "deny failed: upgrade deny needs a
  reason" and the ticket stayed proposed). The fix follows the
  draft Schedule/Edit shape (lane 1): the tap ARMS a reason ask
  for the chat, and the owner's next message completes the
  denial through ``PartnerRuntime.deny()`` itself — the same
  handler the typed path uses. Proactive proposals deny
  reasonless, so their Deny tap still denies in one tap.

* **S3 — Treasury "Second it".** The /treasury card emitted a
  bare ``approve`` with no proposal id; a tap landed in
  ``_control_approve`` with an empty tail and opened the
  open-proposals LIST instead of seconding the motion the card
  rendered under. The card now emits one "Second it" per open
  proposal, each carrying ``approve <pid>`` — the tap seconds
  THAT motion, through the same admin gating the typed path
  enforces (a second admin, never the proposer, never a
  non-admin).

This suite is the §5 harness for both: harvest the REAL emitted
payloads, drive the REAL callback entry points (Bot API signed
callback, MTProto callback query, the runtime's synthetic-text
re-entry), and assert STATE — pairwise against the typed path.
Reply text alone never counts as proof.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.agents.upgrade_queue import UpgradeQueue
from nomorals.core.config import load_settings
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.social.chat.telegram import TelegramAdapter, TelegramBotAdapter

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402

_OWNER_ID = "5478650254"


# ── rig: a full PartnerRuntime over a tmp home ─────────────────


class _RuntimeRig(unittest.TestCase):
    """The lane-1 rig shape: real context, real dispatch, real
    queues; only the model router and the tool seam are doubled."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix3-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": _OWNER_ID,
            "partner.owner_sender_ids": _OWNER_ID,
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    # — seeding / reading state ────────────────────────────────
    def _upgrade(self, title: str = "cache compiled regexes") -> str:
        return UpgradeQueue(self.context).propose(
            title=title, rationale="rationale " * 10, patch_plan={},
            files=[], tests=[], claim_ids=[], source="research_loop")

    def _upgrade_row(self, pid: str) -> dict:
        row = UpgradeQueue(self.context).get(pid)
        assert row is not None
        return row

    def _proactive(self, pid: str = "prop-1",
                   content: str = "held: a note") -> str:
        self.context.db.execute(
            "INSERT INTO proactive_log (id, kind, platform, chat_id,"
            " content, status, reason, decided_at, acted_at)"
            " VALUES (?, 'dm', 'telegram', '123', ?, 'pending',"
            " '', 0, 0)", (pid, content))
        return pid

    def _proactive_status(self, pid: str) -> str:
        row = self.context.db.query_one(
            "SELECT status FROM proactive_log WHERE id = ?", (pid,))
        assert row is not None
        return str(row["status"])

    def _run(self, text: str, *, chat_key: str = "telegram:123") -> Any:
        return self.runtime.handle_control(
            text, chat_key,
            message=_msg(text, chat_key=chat_key, owner=True))

    @staticmethod
    def _text(reply: Any) -> str:
        return reply.text if hasattr(reply, "text") else str(reply)

    def _list_payloads(self, chat_key: str = "telegram:123") -> dict:
        """Harvest: the /proposals list's real emitted buttons,
        keyed by (row id, verb) from the payload itself."""
        out = self._run("/proposals", chat_key=chat_key)
        self.assertTrue(hasattr(out, "buttons"),
                        f"/proposals carried no buttons: {out!r}")
        found: dict[tuple[str, str], str] = {}
        for row in out.buttons:
            for _label, data in row:
                verb, _, pid = str(data).partition(" ")
                found[(pid, verb)] = str(data)
        return found


# ── S2 harvest ─────────────────────────────────────────────────


class ProposalsHarvestTests(_RuntimeRig):
    def test_list_emits_a_deny_button_per_proposal(self) -> None:
        pid = self._upgrade()
        self._proactive("prop-1")
        payloads = self._list_payloads()
        self.assertEqual(payloads[(pid, "deny")], f"deny {pid}")
        self.assertEqual(payloads[("prop-1", "deny")], "deny prop-1")
        # Inside the signed-callback budget, like every emitter.
        self.assertLessEqual(len(f"deny {pid}".encode()), 47)


# ── S2: the tap arms, the answer denies ────────────────────────


class _FakeBotSession:
    """Records Bot API calls; never touches the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        return SimpleNamespace(
            json=lambda: {"ok": True, "result": True})

    def get(self, url, timeout=None):
        self.calls.append((url, {}))
        return SimpleNamespace(
            json=lambda: {"ok": True, "result": True})


def _personal_bot() -> TelegramBotAdapter:
    return TelegramBotAdapter(
        token="<redacted>", session=_FakeBotSession(),
        chat_allow="42", adapter_name="telegram-personal",
        owner_only=True, owner_sender_ids=(_OWNER_ID,))


def _bot_query(adapter: TelegramBotAdapter, data: str,
               chat_id: str = "42") -> dict:
    return {
        "id": "q1",
        "from": {"id": int(_OWNER_ID), "is_bot": False,
                 "first_name": "Owner"},
        "message": {"chat": {"id": int(chat_id), "type": "private"},
                    "message_id": 5},
        "data": adapter._sign_callback(data),
    }


class ProposalsDenyTapTests(_RuntimeRig):
    def _tap(self, adapter: TelegramBotAdapter, data: str):
        received: list = []
        replies: list = []

        def handler(msg: Any) -> None:
            received.append(msg)
            replies.append(self.runtime.handle_control(
                msg.text, msg.chat.key, message=msg))

        adapter._handle_callback(_bot_query(adapter, data), handler)
        return received, replies

    def test_botapi_deny_tap_arms_then_the_answer_denies(self) -> None:
        pid = self._upgrade()
        payload = self._list_payloads()[(pid, "deny")]
        adapter = _personal_bot()
        received, replies = self._tap(adapter, payload)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, f"/deny {pid}")
        prompt = self._text(replies[0])
        # The prompt is specific: it names the ticket and says
        # the denial needs the owner's reason. Nothing moved.
        self.assertIn(pid, prompt)
        self.assertIn("needs a reason", prompt)
        self.assertEqual(self._upgrade_row(pid)["status"], "proposed")
        chat_key = received[0].chat.key
        ask = self.runtime._proposal_ask_store().get(chat_key)
        self.assertIsNotNone(ask)
        assert ask is not None
        self.assertEqual(ask["proposal_id"], pid)

        # The owner's next message in that chat IS the reason —
        # driven through the real message pipeline.
        self.runtime._process_inner(
            _msg("not worth the risk", chat_key=chat_key))
        sent = [text for _platform, text in self.runtime.gateway.sent]
        self.assertTrue(
            any(f"denied {pid}" in line for line in sent), sent)
        row = self._upgrade_row(pid)
        self.assertEqual(row["status"], "denied")
        self.assertEqual(row["reason"], "not worth the risk")
        self.assertNotIn(chat_key, self.runtime._proposal_ask_store())

    def test_typed_bare_deny_arms_the_same_ask(self) -> None:
        # Pairwise with the tap: the typed path and the tap path
        # are the same handler, so both arm and both complete.
        pid = self._upgrade("typed twin ticket")
        reply = self._run(f"/proposals deny {pid}")
        self.assertIn("needs a reason", self._text(reply))
        self.assertEqual(self._upgrade_row(pid)["status"], "proposed")
        answer = self.runtime._proposal_ask_answer(
            _msg("superseded by the router rework",
                 chat_key="telegram:123"))
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn(f"denied {pid}", answer)
        row = self._upgrade_row(pid)
        self.assertEqual(row["status"], "denied")
        self.assertEqual(row["reason"],
                         "superseded by the router rework")

    def test_typed_deny_with_reason_lands_immediately(self) -> None:
        pid = self._upgrade()
        reply = self._run(f"/proposals deny {pid} stale approach")
        self.assertIn(pid, self._text(reply))
        row = self._upgrade_row(pid)
        self.assertEqual(row["status"], "denied")
        self.assertEqual(row["reason"], "stale approach")
        # No ask was armed — the next message is ordinary chat.
        self.assertEqual(self.runtime._proposal_ask_store(), {})
        self.assertIsNone(self.runtime._proposal_ask_answer(
            _msg("anything else", chat_key="telegram:123")))

    def test_proactive_deny_tap_denies_in_one_tap(self) -> None:
        # Reasonless kinds deny directly — the tap completes.
        self._proactive("prop-1")
        payload = self._list_payloads()[("prop-1", "deny")]
        reply = self._run(f"/{payload}")
        self.assertIn("denied", self._text(reply))
        self.assertEqual(self._proactive_status("prop-1"), "denied")
        self.assertEqual(self.runtime._proposal_ask_store(), {})

    def test_cancel_drops_the_ask_and_keeps_the_proposal(self) -> None:
        pid = self._upgrade()
        self._run(f"/deny {pid}")
        answer = self.runtime._proposal_ask_answer(
            _msg("cancel", chat_key="telegram:123"))
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("stays pending", answer)
        self.assertEqual(self._upgrade_row(pid)["status"], "proposed")
        # Disarmed: a later message is NOT captured as a reason.
        self.assertIsNone(self.runtime._proposal_ask_answer(
            _msg("some later thought", chat_key="telegram:123")))
        self.assertEqual(self._upgrade_row(pid)["status"], "proposed")

    def test_answer_in_another_chat_does_not_complete(self) -> None:
        pid = self._upgrade()
        self._run(f"/deny {pid}", chat_key="telegram:123")
        self.assertIsNone(self.runtime._proposal_ask_answer(
            _msg("wrong room reason", chat_key="telegram:777")))
        self.assertEqual(self._upgrade_row(pid)["status"], "proposed")
        self.assertIn("telegram:123",
                      self.runtime._proposal_ask_store())

    def test_late_answer_after_a_decision_fails_honestly(self) -> None:
        pid = self._upgrade()
        self._run(f"/deny {pid}")
        # Decided by the full typed form while the ask was armed.
        self._run(f"/proposals deny {pid} decided meanwhile")
        self.assertEqual(self._upgrade_row(pid)["status"], "denied")
        answer = self.runtime._proposal_ask_answer(
            _msg("a late second reason", chat_key="telegram:123"))
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("couldn't deny", answer)
        row = self._upgrade_row(pid)
        # The first decision stands — never re-denied, never
        # re-reasoned by a stale ask.
        self.assertEqual(row["status"], "denied")
        self.assertEqual(row["reason"], "decided meanwhile")
        self.assertEqual(self.runtime._proposal_ask_store(), {})


class _FakeCallbackEvent:
    def __init__(self, data: bytes, chat_id: str = "42",
                 sender_id: str = _OWNER_ID) -> None:
        self.data = data
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.message_id = 7

    async def answer(self) -> None:
        return None

    async def get_chat(self) -> None:
        return None

    async def get_sender(self) -> None:
        return None


class ProposalsDenyMtprotoTests(_RuntimeRig):
    def test_mtproto_deny_tap_arms_the_ask(self) -> None:
        pid = self._upgrade()
        adapter = TelegramAdapter(api_id=123, api_hash="x",
                                  session_path=":memory:")
        received: list = []
        replies: list = []

        def handler(msg: Any) -> None:
            received.append(msg)
            replies.append(self.runtime.handle_control(
                msg.text, msg.chat.key, message=msg))

        event = _FakeCallbackEvent(f"deny {pid}".encode())
        asyncio.run(adapter._handle_callback_query(event, handler))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, f"/deny {pid}")
        self.assertIn("needs a reason", self._text(replies[0]))
        self.assertEqual(self._upgrade_row(pid)["status"], "proposed")
        self.assertIn(received[0].chat.key,
                      self.runtime._proposal_ask_store())


# ── S3: treasury "Second it" ───────────────────────────────────

_KEY = "telegram-bot:-1001234567"
_ADMINS = {"admin1", "admin2"}


class _GroupAdapter:
    def admin_member(self, chat, user_id):
        if str(user_id) in _ADMINS:
            return {"status": "administrator"}
        return {"status": "member"}


def _group_rig(tmp: str):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _GroupAdapter()

        def _is_operator(self, message):  # type: ignore[no-untyped-def]
            return False

    rt = _RT()
    surface = GroupSurface(GroupStore(Path(tmp) / "g.json").load())
    rt._group_surface_obj = surface
    return rt, surface


def _gmsg(*, sender_id: str = "admin2", sender: str = "Admin"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=_KEY, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender, sender_username="boss",
        text="", meta={}, message_id="1")


def _seed_member(surface, amount: int = 900, *,
                 user_id: str = "55") -> None:
    """Presence + real earnings through the surface's own
    faucets (the group-economy test idiom)."""
    surface.handle_message(_KEY, user_id, username="bola",
                           name="Bola", text="hello group")
    base = time.time()
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(190, remaining)
        surface.game_award(_KEY, user_id, chunk, "ttt",
                           now=base - step * 8 * 86400)
        remaining -= chunk
        step += 1


def _fund_and_propose(surface, *, amount: int = 600,
                      by: str = "admin1") -> str:
    _seed_member(surface, 900)
    assert surface.economy.contribute_treasury(_KEY, "55", 900)
    proposal = surface.propose_spend(
        _KEY, kind="grant",
        payload={"recipient": "55", "amount": amount,
                 "reason": "top contributor"},
        by=by)
    return str(proposal["id"])


def _flat_buttons(reply: Any) -> list[str]:
    buttons = getattr(reply, "buttons", None) or []
    return [str(data) for row in buttons for _label, data in row]


def _drive_payload(rt, payload: str, message):
    """The surface re-entry, faithfully: a tap comes back as
    "/<payload>" text and dispatch splits it into kind + tail.
    Returns the handler reply for the approve kind."""
    kind, _, tail = str(payload).partition(" ")
    assert kind == "approve", payload
    return rt._control_approve(tail, message)


class TreasurySecondItTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix3-t-")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_card_button_carries_the_proposal_id(self) -> None:
        rt, surface = _group_rig(self.tmp.name)
        pid = _fund_and_propose(surface)
        card = rt._control_treasury(_gmsg())
        # Exactly the motion's own button — and no bare
        # "approve" payload anywhere on the card.
        self.assertEqual(_flat_buttons(card), [f"approve {pid}"])
        labels = [label for row in card.buttons
                  for label, _data in row]
        self.assertEqual(labels, ["🗳️ Second it"])

    def test_tap_seconds_the_motion_like_the_typed_command(self):
        # Tap rig and typed rig are twins: same seed, same
        # motion; one is decided by driving the harvested
        # payload, the other by the typed handler call.
        outcomes = []
        for mode in ("tap", "typed"):
            with tempfile.TemporaryDirectory(
                    prefix="nm-btnfix3-twin-") as twin:
                rt, surface = _group_rig(twin)
                pid = _fund_and_propose(surface)
                wallet0 = surface.economy.balance(_KEY, "55")
                treasury0 = surface.economy.treasury_balance(_KEY)
                if mode == "tap":
                    card = rt._control_treasury(_gmsg())
                    payload = _flat_buttons(card)[0]
                    reply = _drive_payload(rt, payload, _gmsg())
                else:
                    reply = rt._control_approve(pid, _gmsg())
                outcomes.append((
                    "approved and executed" in str(reply),
                    surface.open_proposals(_KEY),
                    surface.economy.balance(_KEY, "55") - wallet0,
                    surface.economy.treasury_balance(_KEY)
                    - treasury0))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], (True, {}, 600, -600))

    def test_proposer_cannot_second_their_own_motion(self) -> None:
        rt, surface = _group_rig(self.tmp.name)
        pid = _fund_and_propose(surface, by="admin1")
        card = rt._control_treasury(_gmsg(sender_id="admin1"))
        payload = _flat_buttons(card)[0]
        reply = _drive_payload(rt, payload,
                               _gmsg(sender_id="admin1"))
        self.assertIn("can't approve their own", str(reply))
        self.assertIn(pid, surface.open_proposals(_KEY))

    def test_non_admin_tap_is_refused_and_nothing_moves(self):
        rt, surface = _group_rig(self.tmp.name)
        pid = _fund_and_propose(surface)
        card = rt._control_treasury(_gmsg())
        payload = _flat_buttons(card)[0]
        reply = _drive_payload(
            rt, payload, _gmsg(sender_id="member1", sender="Member"))
        self.assertEqual(str(reply), "approvals are for admins.")
        self.assertIn(pid, surface.open_proposals(_KEY))

    def test_two_motions_each_button_seconds_its_own(self) -> None:
        rt, surface = _group_rig(self.tmp.name)
        pid1 = _fund_and_propose(surface, amount=600)
        proposal2 = surface.propose_spend(
            _KEY, kind="grant",
            payload={"recipient": "55", "amount": 100,
                     "reason": "second motion"},
            by="admin1")
        pid2 = str(proposal2["id"])
        card = rt._control_treasury(_gmsg())
        payloads = _flat_buttons(card)
        self.assertEqual(set(payloads),
                         {f"approve {pid1}", f"approve {pid2}"})
        labels = [label for row in card.buttons
                  for label, _data in row]
        self.assertTrue(
            all(label.startswith("🗳️ Second it — grant")
                for label in labels), labels)
        # Second ONLY the second motion.
        wallet0 = surface.economy.balance(_KEY, "55")
        reply = _drive_payload(
            rt, f"approve {pid2}", _gmsg())
        self.assertIn("approved and executed", str(reply))
        self.assertEqual(set(surface.open_proposals(_KEY)), {pid1})
        self.assertEqual(
            surface.economy.balance(_KEY, "55") - wallet0, 100)


if __name__ == "__main__":
    unittest.main()
