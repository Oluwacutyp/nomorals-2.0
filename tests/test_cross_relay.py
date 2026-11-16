"""Cross-platform relay with attribution (census §6 candidate #2).

One command forwards one existing message — with its forwarded mark
intact — to WhatsApp groups + Telegram channels + her channels at
once. Same-platform targets ride the native forward verb (the
platform's own attribution); cross-platform targets get the message
re-posted with an attribution header carrying the source platform
and chat, text verbatim, media carried when the source row has it
on local disk. Preview-first like plan_crosspost, arm-gated and
dry-run by default like the §14 mass tools and /obroadcast, paced
3s between targets, every run journaled, one dead target never
kills the rest.

Discipline under test: a relay that cannot resolve its source or
any target refuses before anything sends; a live run refuses until
ARMED; a standing owner stop halts that platform's targets; an
empty target set refuses; per-target verdicts name the mode
(forwarded / reposted) and the failure.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import (
    ChatKind,
    ChatMessage,
    ChatRef,
    MediaRef,
    SendResult,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.db import Database
from tests.frozen_clock import MIDDAY_TS

OWNER_ID = "111"
BOT_ID = "999888777"
WA_GROUP = "whatsapp:120363111111@g.us"
WA_GROUP_B = "whatsapp:120363222222@g.us"
TG_GROUP = "telegram-bot:-100111"
TG_CHANNEL = "telegram-bot:-100333"
FOREIGN = "discord:444555"
NOW = MIDDAY_TS

SRC_TEXT = "gold is up — XAUUSD broke 2680"


def _ref(platform: str, chat_id: str, kind: str, title: str = "") -> ChatRef:
    return ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                   title=title, peer=chat_id)


def _src(platform: str = "whatsapp", text: str = SRC_TEXT,
         media: tuple = (), message_id: str = "m42") -> SimpleNamespace:
    from nomorals.social.cross_relay import RelaySource

    return RelaySource(
        platform=platform, chat_key=WA_GROUP, chat_title="Gold Signals",
        message_id=message_id, text=text, media=list(media))


class FakeAdapter:
    """One adapter per platform: records forward/send/media calls."""

    def __init__(self, name: str, *, fail_keys=()):
        self.name = name
        self.forwards: list = []
        self.sends: list = []
        self.media_sends: list = []
        self._history: dict = {}
        self._fail = set(fail_keys)

    def set_history(self, chat_key: str, rows: list) -> None:
        self._history[chat_key] = list(rows)

    def history(self, chat, limit=50):
        return list(self._history.get(str(getattr(chat, "key", chat)), []))

    def forward_message(self, chat, message_id, dest):
        self.forwards.append((str(getattr(chat, "key", chat)),
                              str(message_id),
                              str(getattr(dest, "key", dest))))
        if str(getattr(dest, "key", dest)) in self._fail:
            return {"ok": False, "error": "bridge refused"}
        return {"ok": True, "message_id": str(message_id)}

    def forward_post(self, from_chat, message_id, to_chat):
        self.forwards.append((str(getattr(from_chat, "key", from_chat)),
                              str(message_id),
                              str(getattr(to_chat, "key", to_chat))))
        if str(getattr(to_chat, "key", to_chat)) in self._fail:
            return SendResult(ok=False, platform=self.name,
                              error="chat_write_forbidden")
        return SendResult(ok=True, platform=self.name,
                          message_id=str(message_id))

    def send(self, chat, text, **kw):
        self.sends.append((str(getattr(chat, "key", chat)), str(text)))
        if str(getattr(chat, "key", chat)) in self._fail:
            return SendResult(ok=False, platform=self.name,
                              error="chat_write_forbidden")
        return SendResult(ok=True, platform=self.name,
                          message_id=f"s{len(self.sends)}")

    def send_media(self, chat, media, *, caption="", **kw):
        self.media_sends.append(
            (str(getattr(chat, "key", chat)), str(media.path), str(caption)))
        if str(getattr(chat, "key", chat)) in self._fail:
            return SendResult(ok=False, platform=self.name,
                              error="chat_write_forbidden")
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.media_sends)}")


def _plan_targets():
    from nomorals.social.cross_relay import RelayTarget

    return [
        RelayTarget(platform="whatsapp", key=WA_GROUP_B,
                    title="Signals 2", kind="group"),
        RelayTarget(platform="telegram-bot", key=TG_CHANNEL,
                    title="Signals", kind="channel"),
    ]


class OrganPlanTests(unittest.TestCase):
    def test_empty_targets_refuses(self):
        from nomorals.social.cross_relay import RelayRefused, plan_relay

        with self.assertRaisesRegex(RelayRefused, "no target"):
            plan_relay(_src(), [], pace_seconds=3.0)

    def test_missing_message_id_refuses(self):
        from nomorals.social.cross_relay import RelayRefused, plan_relay

        with self.assertRaisesRegex(RelayRefused, "message"):
            plan_relay(_src(message_id=""), _plan_targets(),
                       pace_seconds=3.0)

    def test_pace_below_floor_refuses(self):
        from nomorals.social.cross_relay import RelayRefused, plan_relay

        with self.assertRaisesRegex(RelayRefused, "3s"):
            plan_relay(_src(), _plan_targets(), pace_seconds=1.0)

    def test_modes_split_by_platform(self):
        from nomorals.social.cross_relay import plan_relay

        plan = plan_relay(_src(), _plan_targets(), pace_seconds=3.0)
        self.assertTrue(plan.ok)
        self.assertEqual(plan.modes[WA_GROUP_B], "forward")
        self.assertEqual(plan.modes[TG_CHANNEL], "repost")
        self.assertEqual(plan.targets[0].key, WA_GROUP_B)

    def test_stopped_platform_is_skipped_not_fatal(self):
        from nomorals.social.cross_relay import plan_relay

        plan = plan_relay(_src(), _plan_targets(), pace_seconds=3.0,
                          stopped_platforms={"telegram-bot"})
        self.assertTrue(plan.ok)
        self.assertEqual([t.key for t in plan.targets], [WA_GROUP_B])
        self.assertEqual(len(plan.skipped), 1)
        self.assertEqual(plan.skipped[0]["key"], TG_CHANNEL)
        self.assertIn("standing stop", plan.skipped[0]["reason"])

    def test_all_targets_stopped_refuses(self):
        from nomorals.social.cross_relay import RelayRefused, plan_relay

        with self.assertRaisesRegex(RelayRefused, "standing stop"):
            plan_relay(_src(), _plan_targets(), pace_seconds=3.0,
                       stopped_platforms={"whatsapp", "telegram-bot"})

    def test_cross_platform_without_text_is_skipped(self):
        from nomorals.social.cross_relay import plan_relay

        plan = plan_relay(_src(text=""), _plan_targets(),
                          pace_seconds=3.0)
        self.assertTrue(plan.ok)
        self.assertEqual([t.key for t in plan.targets], [WA_GROUP_B])
        self.assertEqual(plan.skipped[0]["key"], TG_CHANNEL)
        self.assertIn("read the message text", plan.skipped[0]["reason"])

    def test_media_rides_when_on_disk_and_drops_when_missing(self):
        from nomorals.social.cross_relay import plan_relay

        with tempfile.TemporaryDirectory() as tmp:
            pic = Path(tmp) / "pic.jpg"
            pic.write_bytes(b"fake-jpeg")
            media = (MediaRef(path=str(pic), mime="image/jpeg",
                              kind="image", name="pic.jpg"),
                     MediaRef(path=str(Path(tmp) / "gone.mp4"),
                              mime="video/mp4", kind="video",
                              name="gone.mp4"))
            plan = plan_relay(_src(media=media), _plan_targets(),
                              pace_seconds=3.0)
            self.assertEqual(len(plan.media_carried), 1)
            self.assertEqual(plan.media_carried[0].path, str(pic))
            self.assertIn("gone.mp4", plan.media_note)

    def test_preview_names_source_targets_modes_and_skips(self):
        from nomorals.social.cross_relay import plan_relay, preview_relay

        plan = plan_relay(_src(), _plan_targets(), pace_seconds=3.0,
                          stopped_platforms={"telegram-bot"})
        out = preview_relay(plan)
        self.assertIn("Gold Signals", out)
        self.assertIn("m42", out)
        self.assertIn(WA_GROUP_B, out)
        self.assertIn("forward", out)
        self.assertIn("Signals", out)
        self.assertIn("standing stop", out)


class OrganRunTests(unittest.TestCase):
    def _run(self, *, fail_keys=(), **plan_kw):
        from nomorals.social.cross_relay import plan_relay, run_relay

        wa = FakeAdapter("whatsapp", fail_keys=fail_keys)
        tg = FakeAdapter("telegram-bot", fail_keys=fail_keys)
        plan = plan_relay(_src(), _plan_targets(), pace_seconds=3.0,
                          **plan_kw)
        sleeps: list = []
        record = run_relay(plan=plan,
                           adapters={"whatsapp": wa, "telegram-bot": tg},
                           sleep=sleeps.append, pace_seconds=3.0)
        return record, wa, tg, sleeps

    def test_same_platform_uses_native_forward(self):
        record, wa, tg, _sleeps = self._run()
        self.assertEqual(record["sent"], 2)
        self.assertEqual(record["failed"], 0)
        self.assertEqual(wa.forwards,
                         [(WA_GROUP, "m42", WA_GROUP_B)])
        # The cross-platform target is a repost, not a forward.
        self.assertEqual(len(tg.sends), 1)
        self.assertIn("forwarded from WhatsApp", tg.sends[0][1])

    def test_cross_platform_repost_carries_attribution_and_text(self):
        record, wa, tg, _sleeps = self._run()
        self.assertEqual(len(tg.sends), 1)
        dest, text = tg.sends[0]
        self.assertEqual(dest, TG_CHANNEL)
        self.assertIn("forwarded from WhatsApp", text)
        self.assertIn("Gold Signals", text)
        self.assertIn(SRC_TEXT, text)
        row = [r for r in record["results"]
               if r["target"] == TG_CHANNEL][0]
        self.assertEqual(row["mode"], "repost")
        self.assertTrue(row["ok"])

    def test_repost_carries_media_with_attribution_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            pic = Path(tmp) / "pic.jpg"
            pic.write_bytes(b"fake-jpeg")
            media = (MediaRef(path=str(pic), mime="image/jpeg",
                              kind="image", name="pic.jpg"),)
            record, _wa, tg, _sleeps = self._run()
            # Re-run with media on the source.
            from nomorals.social.cross_relay import plan_relay, run_relay

            wa2 = FakeAdapter("whatsapp")
            tg2 = FakeAdapter("telegram-bot")
            plan = plan_relay(_src(media=media), _plan_targets(),
                              pace_seconds=3.0)
            record2 = run_relay(
                plan=plan, adapters={"whatsapp": wa2, "telegram-bot": tg2},
                sleep=lambda s: None, pace_seconds=3.0)
            self.assertEqual(record2["sent"], 2)
            self.assertEqual(len(tg2.media_sends), 1)
            dest, path, caption = tg2.media_sends[0]
            self.assertEqual(dest, TG_CHANNEL)
            self.assertEqual(path, str(pic))
            self.assertIn("forwarded from WhatsApp", caption)

    def test_one_dead_target_does_not_kill_the_rest(self):
        record, wa, tg, _sleeps = self._run(fail_keys={TG_CHANNEL})
        self.assertEqual(record["sent"], 1)
        self.assertEqual(record["failed"], 1)
        failed = [r for r in record["results"] if not r["ok"]]
        self.assertEqual(failed[0]["target"], TG_CHANNEL)
        self.assertIn("chat_write_forbidden", failed[0]["error"])
        self.assertEqual(wa.forwards, [(WA_GROUP, "m42", WA_GROUP_B)])

    def test_raised_adapter_call_becomes_a_failed_row(self):
        from nomorals.social.cross_relay import plan_relay, run_relay

        wa = FakeAdapter("whatsapp")
        tg = FakeAdapter("telegram-bot")
        tg.send = mock.Mock(side_effect=RuntimeError("socket died"))
        plan = plan_relay(_src(), _plan_targets(), pace_seconds=3.0)
        record = run_relay(plan=plan,
                           adapters={"whatsapp": wa, "telegram-bot": tg},
                           sleep=lambda s: None, pace_seconds=3.0)
        self.assertEqual(record["sent"], 1)
        self.assertEqual(record["failed"], 1)
        row = [r for r in record["results"]
               if r["target"] == TG_CHANNEL][0]
        self.assertIn("socket died", row["error"])

    def test_missing_adapter_is_a_failed_row(self):
        from nomorals.social.cross_relay import plan_relay, run_relay

        wa = FakeAdapter("whatsapp")
        plan = plan_relay(_src(), _plan_targets(), pace_seconds=3.0)
        record = run_relay(plan=plan, adapters={"whatsapp": wa},
                           sleep=lambda s: None, pace_seconds=3.0)
        self.assertEqual(record["sent"], 1)
        self.assertEqual(record["failed"], 1)
        row = [r for r in record["results"]
               if r["target"] == TG_CHANNEL][0]
        self.assertIn("no live telegram-bot surface", row["error"])

    def test_pacing_gaps_between_deliveries_only(self):
        _record, _wa, _tg, sleeps = self._run()
        self.assertEqual(sleeps, [3.0])


# ── the handler ────────────────────────────────────────────────


def _bare_runtime(tmp_path: Path) -> PartnerRuntime:
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(
            community_dir=str(tmp_path),
            workspace_dir=str(tmp_path)),
    )
    rt._owner_chats = {OWNER_ID}
    rt._owner_sender_ids = {OWNER_ID}
    return rt


def _seed_surface(tmp_path: Path) -> GroupSurface:
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    for uid, name in (("1", "Ada"), ("2", "Bola")):
        surface.record_join(TG_GROUP, uid, username=name.lower(),
                            name=name, ts=NOW)
    surface.store.group(TG_GROUP)["title"] = "Group A"
    centry = surface.store.group(TG_CHANNEL)
    centry.setdefault("extras", {})["channel"] = {"title": "Signals"}
    surface.store.group(FOREIGN)["title"] = "Foreign"
    surface.store.save()
    return surface


class FakeGateway:
    """Gateway contract: adapters dict + send, with a call log."""

    def __init__(self):
        self.adapters: dict = {}
        self.sends: list = []

    def send(self, platform, chat, text, **kw):
        self.sends.append((platform, chat, text))
        return SimpleNamespace(ok=True, message_id="1")


class _RelayCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-xrelay-")
        self.tmp_path = Path(self._tmp.name)
        self.rt = _bare_runtime(self.tmp_path)
        self.surface = _seed_surface(self.tmp_path)
        self.rt._group_surface_obj = self.surface
        self.wa = FakeAdapter("whatsapp")
        self.tg = FakeAdapter("telegram-bot")
        self.gateway = FakeGateway()
        self.gateway.adapters = {"whatsapp": self.wa,
                                 "telegram-bot": self.tg}
        self.rt.gateway = self.gateway
        self.rt._owner_registry().seed(telegram_ids=[OWNER_ID])
        # Paired accounts, so the source-platform gate passes.
        from nomorals.social.pairing import AccountRecord

        svc = self.rt._pairing_personal()
        svc["registry"].upsert(AccountRecord(
            id="whatsapp:2348012345678", platform="whatsapp",
            identity="2348012345678", status="linked", created_at=1.0))
        svc["registry"].upsert(AccountRecord(
            id="telegram:111", platform="telegram",
            identity="111", status="linked", created_at=1.0))
        # Source history: the replied-to message, with text.
        src_chat = _ref("whatsapp", "120363111111@g.us",
                        ChatKind.GROUP, "Gold Signals")
        self.wa.set_history(WA_GROUP, [
            ChatMessage(chat=src_chat, incoming=False, text=SRC_TEXT,
                        message_id="m42", ts=NOW)])

    def tearDown(self):
        self._tmp.cleanup()

    def _msg(self, text: str, *, reply_to: str = "m42"):
        chat = SimpleNamespace(key=WA_GROUP, platform="whatsapp",
                               kind=ChatKind.GROUP,
                               chat_id="120363111111@g.us",
                               title="Gold Signals")
        return SimpleNamespace(chat=chat, text=text, meta={},
                               sender="owner", sender_id=OWNER_ID,
                               sender_username="owner",
                               reply_to=reply_to)

    def _reply(self, text: str, **kw):
        return self.rt.handle_control(text, WA_GROUP,
                                      message=self._msg(text, **kw))

    def _journal_rows(self):
        svc = self.rt._relay_service()
        rows = svc["journal"].tail(20, tool="relay")
        return rows

    def _text(self, out) -> str:
        return getattr(out, "text", out)


class HandlerTests(_RelayCase):
    def test_bare_relay_is_usage_and_names_rooms(self):
        out = self._text(self._reply("/relay", reply_to=""))
        self.assertIn("usage", out.lower())
        self.assertIn("Group A", out)
        self.assertIn("Signals", out)
        self.assertEqual(self.wa.forwards, [])
        self.assertEqual(self.tg.sends, [])

    def test_reply_without_targets_is_usage(self):
        out = self._text(self._reply("/relay"))
        self.assertIn("usage", out.lower())
        self.assertIn("Group A", out)
        self.assertEqual(self.wa.forwards, [])
        self.assertEqual(self.tg.sends, [])

    def test_dry_run_previews_and_sends_nothing(self):
        out = self._text(self._reply(f"/relay {TG_CHANNEL}"))
        self.assertIn("dry-run", out)
        self.assertIn(TG_CHANNEL, out)
        self.assertIn("repost", out)
        self.assertEqual(self.wa.forwards, [])
        self.assertEqual(self.tg.sends, [])
        self.assertEqual(len(self._journal_rows()), 1)
        self.assertTrue(self._journal_rows()[0].get("dry_run"))

    def test_live_run_refuses_until_armed(self):
        out = self._text(self._reply(f"/relay {TG_CHANNEL} | live"))
        self.assertIn("arm", out.lower())
        self.assertEqual(self.wa.forwards, [])
        self.assertEqual(self.tg.sends, [])
        self.assertEqual(len(self._journal_rows()), 1)

    def test_armed_live_run_relays_and_reports_modes(self):
        self._reply("/relay arm")
        out = self._text(self._reply(
            f"/relay {WA_GROUP_B} | {TG_CHANNEL} | live"))
        self.assertIn("— forward", out)
        self.assertIn("— repost", out)
        self.assertEqual(self.wa.forwards,
                         [(WA_GROUP, "m42", WA_GROUP_B)])
        self.assertEqual(len(self.tg.sends), 1)
        self.assertIn("forwarded from WhatsApp", self.tg.sends[0][1])
        rows = self._journal_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("sent"), 2)

    def test_scope_all_reaches_claimed_rooms_only(self):
        self._reply("/relay arm")
        out = self._text(self._reply("/relay all | live"))
        self.assertNotIn(FOREIGN, out)
        self.assertEqual(self.tg.forwards, [])
        # TG_GROUP is same-platform? No — source is WhatsApp, so the
        # telegram-bot rooms get reposts.
        dests = [d for d, _t in self.tg.sends]
        self.assertIn(TG_GROUP, dests)
        self.assertIn(TG_CHANNEL, dests)
        self.assertEqual(self.wa.forwards, [])

    def test_unknown_named_target_refuses_before_any_send(self):
        self._reply("/relay arm")
        out = self._text(self._reply("/relay No Such Chat | live"))
        self.assertIn("No Such Chat", out)
        self.assertEqual(self.wa.forwards, [])
        self.assertEqual(self.tg.sends, [])

    def test_standing_stop_halts_that_platform(self):
        svc = self.rt._pairing_personal()
        svc["registry"].set_owner_stopped("telegram-bot", True)
        self._reply("/relay arm")
        out = self._text(self._reply(
            f"/relay {WA_GROUP_B} | {TG_CHANNEL} | live"))
        self.assertIn("standing stop", out)
        self.assertEqual(self.wa.forwards,
                         [(WA_GROUP, "m42", WA_GROUP_B)])
        self.assertEqual(self.tg.sends, [])

    def test_explicit_source_form(self):
        self._reply("/relay arm")
        out = self._text(self._reply(
            f"/relay from {WA_GROUP} m42 | {TG_CHANNEL} | live",
            reply_to=""))
        self.assertIn("— repost", out)
        self.assertEqual(len(self.tg.sends), 1)
        self.assertIn(SRC_TEXT, self.tg.sends[0][1])

    def test_arm_status_runs_pacing_verbs(self):
        out = self._text(self._reply("/relay status"))
        self.assertIn("disarmed", out)
        self._reply("/relay arm")
        out = self._text(self._reply("/relay status"))
        self.assertIn("ARMED", out)
        out = self._text(self._reply("/relay pacing"))
        self.assertIn("3s", out)
        out = self._text(self._reply("/relay pacing 1"))
        self.assertIn("3s", out)
        self._reply("/relay disarm")
        out = self._text(self._reply("/relay status"))
        self.assertIn("disarmed", out)

    def test_runs_lists_the_journal(self):
        self._reply("/relay arm")
        self._reply(f"/relay {TG_CHANNEL} | live")
        out = self._text(self._reply("/relay runs"))
        self.assertIn("relay", out.lower())
        self.assertIn("1/1", out)


class RegistryTests(unittest.TestCase):
    def test_relay_is_registered_end_to_end(self):
        self.assertIn("relay", CONTROL_COMMANDS)
        page = COMMAND_DETAILS.get("relay") or {}
        for field in ("what", "usage", "example"):
            self.assertTrue(str(page.get(field, "")).strip(),
                            f"relay DETAILS.{field} is empty")
        self.assertTrue(str(LIST_ONELINERS.get("relay", "")).strip())
        grouped = [k for _t, kinds in LIST_GROUPS for k in kinds]
        self.assertEqual(grouped.count("relay"), 1)
        self.assertNotIn("relay", __import__(
            "nomorals.social.chat.control", fromlist=["PUBLIC_COMMANDS"]
        ).PUBLIC_COMMANDS)

    def test_relay_has_a_dispatch_branch(self):
        from pathlib import Path as _Path

        src = (_Path(__file__).resolve().parent.parent
               / "nomorals" / "agents" / "partner" / "runtime.py").read_text()
        self.assertIn('if kind == "relay"', src)


if __name__ == "__main__":
    unittest.main()
