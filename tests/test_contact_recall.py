"""Per-contact total recall briefs (census §6 candidate #5).

"Everything between me and X": messages, media, statuses, calls,
across platforms, from the vault + history verbs — as one brief.

Discipline under test: the brief assembles only what is recorded
(vault journal rows, DM cache rings, recoveries, live per-chat
history, group ledger records) and names every source; a contact
matches by exact identity (digits-tail id, @handle, display name —
never substring, so "Ada" never matches "Adaeze"); an ambiguous
name lists its candidates instead of guessing; an unknown contact
briefs honestly instead of inventing a record.
"""
from __future__ import annotations

import datetime
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.frozen_clock import FrozenClockMixin

OWNER_ID = "111"

NOW = datetime.datetime(2026, 10, 14, 12, 0,
                        tzinfo=datetime.UTC).timestamp()
HOUR = 3600


def _row(poster: str, ts: float, *, kind: str = "status_arrival",
         platform: str = "whatsapp", text: str = "",
         source_id: str = "", media_kind: str = "",
         counterpart_id: str = "") -> dict:
    return {
        "ts": ts, "kind": kind,
        "account": f"{platform}:primary", "platform": platform,
        "counterpart_id": counterpart_id or f"id-{poster.lower()}",
        "poster_name": poster, "media_kind": media_kind,
        "source_id": source_id or f"{poster}-{kind}-{int(ts)}",
        "path": "", "bytes": 0, "post_ts": ts, "text": text,
        "media": [], "media_missing": False,
        "note": f"{kind} row",
    }


def _call(poster: str, ts: float, *,
          platform: str = "whatsapp") -> dict:
    return {
        "ts": ts, "kind": "call_offer",
        "account": f"{platform}:primary", "platform": platform,
        "counterpart_id": f"id-{poster.lower()}",
        "poster_name": poster, "media_kind": "",
        "source_id": f"call-{int(ts)}", "path": "", "bytes": 0,
        "note": "rejected (anticall on)",
    }


def _recovery(poster: str, ts: float, *, kind: str = "delete",
              text: str = "hello there") -> dict:
    return {
        "ts": ts, "kind": kind, "account": "whatsapp:primary",
        "platform": "whatsapp", "chat_key": "whatsapp:id-ada",
        "counterpart_id": f"id-{poster.lower()}",
        "message_id": f"m-{int(ts)}", "recovered": True,
        "original_text": text, "new_text": "",
        "original_ts": ts - 60,
    }


# ── organ ────────────────────────────────────────────────────────────

class OrganTests(unittest.TestCase):
    def test_digits_tail_matches_jid_and_lid(self):
        from nomorals.social.contact_recall import matches_contact

        self.assertTrue(matches_contact(
            "2348065574087", "2348065574087@s.whatsapp.net"))
        self.assertTrue(matches_contact(
            "2348065574087", "2348065574087@lid"))
        self.assertFalse(matches_contact(
            "2348065574087", "2348065574088@s.whatsapp.net"))

    def test_username_case_insensitive(self):
        from nomorals.social.contact_recall import matches_contact

        self.assertTrue(matches_contact(
            "@Adaeze", "999", username="adaeze", name="Ada Eze"))
        self.assertTrue(matches_contact(
            "adaeze", "999", username="Adaeze", name="Ada Eze"))

    def test_name_exact_never_substring(self):
        from nomorals.social.contact_recall import matches_contact

        self.assertTrue(matches_contact("Ada", "1", name="ada"))
        self.assertFalse(matches_contact("Ada", "2", name="Adaeze"))
        self.assertFalse(matches_contact("", "1", name="Ada"))

    def test_index_merges_kinds_and_names(self):
        from nomorals.social.contact_recall import build_contact_index

        rows = [
            _row("Ada", NOW - HOUR, text="morning"),
            _row("Ada", NOW - 2 * HOUR, kind="story",
                 platform="whatsapp", media_kind="photo"),
            _call("Ada", NOW - 3 * HOUR),
            _row("Chidi", NOW - HOUR, platform="telegram",
                 kind="status_arrival"),
        ]
        index = build_contact_index(rows)
        self.assertEqual(len(index), 2)
        ada = [e for e in index
               if e["counterpart_id"] == "id-ada"][0]
        self.assertEqual(ada["platform"], "whatsapp")
        self.assertEqual(ada["names"], ["Ada"])
        self.assertEqual(ada["kinds"]["status_arrival"], 1)
        self.assertEqual(ada["kinds"]["story"], 1)
        self.assertEqual(ada["kinds"]["call_offer"], 1)
        self.assertEqual(ada["total"], 3)
        self.assertEqual(ada["first_ts"], NOW - 3 * HOUR)
        self.assertEqual(ada["last_ts"], NOW - HOUR)

    def test_index_ignores_channel_and_own_rows(self):
        from nomorals.social.contact_recall import build_contact_index

        rows = [
            _row("SomeChannel", NOW - HOUR, kind="channel_post"),
            _row("", NOW - HOUR, kind="profile_changed"),
            _row("", NOW - HOUR, kind="status_posted"),
            _row("Ada", NOW - HOUR),
        ]
        index = build_contact_index(rows)
        self.assertEqual(
            [e["counterpart_id"] for e in index], ["id-ada"])

    def test_resolve_ambiguous_across_platforms(self):
        from nomorals.social.contact_recall import (
            build_contact_index,
            resolve_contact,
        )

        rows = [
            _row("Ada", NOW - HOUR, platform="whatsapp"),
            _row("Ada", NOW - HOUR, platform="telegram"),
        ]
        index = build_contact_index(rows)
        matches = resolve_contact("Ada", index)
        self.assertEqual(len(matches), 2)
        wa_only = resolve_contact("Ada", index, platform="whatsapp")
        self.assertEqual(len(wa_only), 1)
        self.assertEqual(wa_only[0]["platform"], "whatsapp")
        self.assertEqual(resolve_contact("Nobody", index), [])

    def test_dedupe_posts_counts_arrival_and_save_once(self):
        from nomorals.social.contact_recall import dedupe_posts

        rows = [
            _row("Ada", NOW - HOUR, source_id="s1"),
            _row("Ada", NOW - HOUR, kind="status",
                 source_id="s1", media_kind="image"),
            _row("Ada", NOW - 2 * HOUR, source_id="s2"),
        ]
        self.assertEqual(len(dedupe_posts(rows)), 2)

    def test_assemble_sections_cover_every_source(self):
        from nomorals.social.contact_recall import assemble_brief

        contact = {"platform": "whatsapp",
                   "counterpart_id": "id-ada", "names": ["Ada"]}
        brief = assemble_brief(
            contact,
            journal_rows=[
                _row("Ada", NOW - HOUR, text="morning",
                     source_id="s1"),
                _row("Ada", NOW - HOUR, kind="status",
                     source_id="s1", media_kind="image"),
                _call("Ada", NOW - 2 * HOUR),
                _row("Ada", NOW - 3 * HOUR, kind="status_react",
                     source_id="s1"),
            ],
            recoveries=[_recovery("Ada", NOW - 4 * HOUR)],
            cached_messages=[{
                "message_id": "m1", "sender_id": "id-ada",
                "sender_name": "Ada", "text": "see you at 5",
                "media": [], "incoming": True, "ts": NOW - HOUR,
            }],
            live_history=[{
                "message_id": "m2", "text": "on my way",
                "incoming": False, "ts": NOW - 1800,
            }],
            groups=[{
                "key": "telegram:1", "kind": "group",
                "title": "Devon", "message_count": 42,
                "last_seen": NOW - HOUR, "joined_ts": NOW - 86400,
                "role": "", "wallet": 100, "banked": 0,
                "warns": 0, "strikes": 0, "bans": 0,
                "reputation_records": 0, "identity_history": [],
            }],
            live_notes=[],
        )
        self.assertEqual(len(brief["statuses"]["posts"]), 1)
        self.assertEqual(len(brief["calls"]), 1)
        self.assertEqual(len(brief["statuses"]["reacted"]), 1)
        self.assertEqual(len(brief["recoveries"]), 1)
        self.assertEqual(len(brief["messages"]["cached"]), 1)
        self.assertEqual(len(brief["messages"]["live"]), 1)
        self.assertEqual(len(brief["groups"]), 1)

    def test_render_names_every_source(self):
        from nomorals.social.contact_recall import (
            assemble_brief,
            render_brief,
        )

        contact = {"platform": "whatsapp",
                   "counterpart_id": "id-ada", "names": ["Ada"]}
        brief = assemble_brief(
            contact,
            journal_rows=[_row("Ada", NOW - HOUR, text="morning")],
            recoveries=[], cached_messages=[], live_history=[],
            groups=[], live_notes=["WhatsApp — no live surface here"],
        )
        out = render_brief(brief)
        self.assertIn("Ada", out)
        self.assertIn("vault journal", out)
        self.assertIn("no live surface", out)

    def test_render_empty_brief_is_honest(self):
        from nomorals.social.contact_recall import (
            assemble_brief,
            render_brief,
        )

        contact = {"platform": "telegram",
                   "counterpart_id": "999", "names": ["Ghost"]}
        brief = assemble_brief(
            contact, journal_rows=[], recoveries=[],
            cached_messages=[], live_history=[], groups=[],
            live_notes=["Telegram — no live surface here"],
        )
        out = render_brief(brief)
        self.assertIn("nothing recorded", out)
        self.assertNotIn("0 messages from", out)


class VaultCacheTests(unittest.TestCase):
    def test_cached_chat_returns_ring(self):
        from nomorals.social.personal import PersonalVault

        with tempfile.TemporaryDirectory() as tmp:
            vault = PersonalVault(Path(tmp) / "personal")
            cache = {
                "whatsapp:primary": {
                    "whatsapp:id-ada": [{
                        "message_id": "m1", "sender_id": "id-ada",
                        "sender_name": "Ada", "text": "hi",
                        "media": [], "incoming": True,
                        "ts": NOW - HOUR,
                    }],
                },
            }
            (Path(tmp) / "personal" / "dm_cache.json").write_text(
                json.dumps(cache), encoding="utf-8")
            vault = PersonalVault(Path(tmp) / "personal")
            rows = vault.cached_chat("whatsapp:primary",
                                     "whatsapp:id-ada")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["text"], "hi")
            self.assertEqual(
                vault.cached_chat("whatsapp:primary",
                                  "whatsapp:id-nobody"), [])
            self.assertEqual(
                vault.cached_chat("telegram:primary",
                                  "whatsapp:id-ada"), [])


# ── handler ──────────────────────────────────────────────────────────

def _bare_runtime(tmp_path: Path):
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.storage.db import Database

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


class _ContactCase(FrozenClockMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-contact-")
        self.tmp_path = Path(self._tmp.name)
        self.rt = _bare_runtime(self.tmp_path)
        self.rt._owner_registry().seed(telegram_ids=[OWNER_ID])

        class _Gw:
            def __init__(self):
                self.adapters = {}

            def send(self, platform, chat, text, **kw):
                return SimpleNamespace(ok=True, message_id="1")

        self.rt.gateway = _Gw()
        self._clock_patch = mock.patch(
            "nomorals.agents.partner.runtime_provision.time")
        mocked = self._clock_patch.start()
        mocked.time.return_value = self.now
        self.addCleanup(self._clock_patch.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def _vault_dir(self) -> Path:
        vault_dir = self.tmp_path / "data" / "social" / "personal"
        vault_dir.mkdir(parents=True, exist_ok=True)
        return vault_dir

    def _seed(self, rows, *, recoveries=None, cache=None):
        vault_dir = self._vault_dir()
        path = vault_dir / "journal.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        if recoveries:
            rpath = vault_dir / "recoveries.jsonl"
            with rpath.open("a", encoding="utf-8") as fh:
                for row in recoveries:
                    fh.write(json.dumps(row) + "\n")
        if cache is not None:
            (vault_dir / "dm_cache.json").write_text(
                json.dumps(cache), encoding="utf-8")

    def _msg(self, text: str):
        from nomorals.social.chat.base import ChatKind

        chat = SimpleNamespace(key="telegram:111", platform="telegram",
                               kind=ChatKind.DM, chat_id="111",
                               title="")
        return SimpleNamespace(chat=chat, text=text, meta={},
                               sender="owner", sender_id=OWNER_ID,
                               sender_username="owner", reply_to="")

    def _reply(self, text: str):
        out = self.rt.handle_control(text, "telegram:111",
                                     message=self._msg(text))
        return getattr(out, "text", out)


class HandlerTests(_ContactCase):
    def test_contact_brief_over_seeded_vault(self):
        self._seed(
            [_row("Ada", NOW - HOUR, text="morning"),
             _call("Ada", NOW - 2 * HOUR)],
            recoveries=[_recovery("Ada", NOW - 4 * HOUR)],
            cache={"whatsapp:primary": {
                "whatsapp:id-ada": [{
                    "message_id": "m1", "sender_id": "id-ada",
                    "sender_name": "Ada", "text": "see you at 5",
                    "media": [], "incoming": True,
                    "ts": NOW - HOUR,
                }],
            }},
        )
        out = self._reply("/contact ada")
        self.assertIn("Ada", out)
        self.assertIn("see you at 5", out)
        self.assertIn("Calls", out)
        self.assertIn("Recovered", out)
        self.assertIn("vault", out.lower())

    def test_contact_unknown_is_honest(self):
        self._seed([_row("Ada", NOW - HOUR)])
        out = self._reply("/contact nobody")
        self.assertIn("no record of nobody", out)

    def test_contact_bare_is_usage(self):
        out = self._reply("/contact")
        self.assertIn("/contact", out)
        self.assertIn("usage", out.lower())

    def test_contact_ambiguous_lists_candidates(self):
        self._seed([
            _row("Ada", NOW - HOUR, platform="whatsapp"),
            _row("Ada", NOW - HOUR, platform="telegram"),
        ])
        out = self._reply("/contact ada")
        self.assertIn("WhatsApp", out)
        self.assertIn("Telegram", out)
        self.assertIn("wa:", out)

    def test_contact_platform_prefix_narrows(self):
        self._seed([
            _row("Ada", NOW - HOUR, platform="whatsapp"),
            _row("Ada", NOW - HOUR, platform="telegram"),
        ])
        out = self._reply("/contact tg:ada")
        self.assertIn("Telegram", out)
        self.assertNotIn("WhatsApp", out)

    def test_contact_by_digits(self):
        self._seed([_row("Ada", NOW - HOUR,
                         counterpart_id="2348065574087@s.whatsapp.net")])
        out = self._reply("/contact 2348065574087")
        self.assertIn("Ada", out)

    def test_contact_names_down_live_surface(self):
        self._seed([_row("Ada", NOW - HOUR)])
        out = self._reply("/contact ada")
        self.assertIn("no live surface", out)

    def test_contact_live_history_when_adapter_present(self):
        self._seed([_row("Ada", NOW - HOUR, platform="telegram")])

        class _Adapter:
            def history(self, chat, limit=20):
                return [SimpleNamespace(
                    text="on my way", ts=NOW - 1800,
                    incoming=False, message_id="m9")]

        self.rt.gateway.adapters = {"telegram": _Adapter()}
        out = self._reply("/contact ada")
        self.assertIn("on my way", out)


class RegistryTests(unittest.TestCase):
    def test_contact_registered_everywhere(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            help_text,
            list_catalog,
        )

        self.assertIn("contact", CONTROL_COMMANDS)
        page = COMMAND_DETAILS["contact"]
        self.assertTrue(page["what"])
        self.assertTrue(page["usage"])
        self.assertTrue(page["example"])
        self.assertIn("contact", LIST_ONELINERS)
        slots = [slot for group in LIST_GROUPS
                 for slot in group[1] if slot == "contact"]
        self.assertEqual(len(slots), 1)
        self.assertIn("/contact", help_text())
        self.assertIn("/contact", list_catalog())


if __name__ == "__main__":
    unittest.main()
