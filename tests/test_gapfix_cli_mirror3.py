"""CLI-mirror bundle #3: ``nm community`` + ``nm groups`` + ``nm digest``.

Failing-first: on the pristine base none of the three surfaces exists
(no ``nomorals.cmdline.commands.community`` / ``groups`` / ``digest``
modules, no parsers), so every test here fails there.

Each command mirrors its chat verb over the same durable store:

1. ``nm groups`` — the chat /groups board (runtime_group._control_groups
   + _groups_board + _groups_group_lines) over the same GroupSurface
   (``social.chat.group_surface.default_surface`` from the console
   workspace dir): every group/channel, member count, claim state,
   economy health, pending actions. The console is the owner's seat,
   so it shows every group — no admin gate, no Card/buttons, no
   console-session map.
2. ``nm community`` — the chat /community read verbs (list/show/status)
   over the same communities index (``social.chat.group_community``)
   and the same durable arms store. Post/mirror/arm/disarm/link/
   unlink/sync/member change state or send, so they stay chat-only
   and are refused in words from the console; ``nm community mirror``
   is deliberately not built (R2-26).
3. ``nm digest`` — the chat /digest read verbs over the same
   ``whatsapp_digest.build_group_digest_text`` + the same
   ConversationLog: the board over every WhatsApp group, or one group
   by name. ``policy`` sets state, so it stays chat-only and is
   refused in words from the console.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from nomorals.storage.conversations import ConversationLog
from nomorals.storage.db import Database

OWNER_ID = "5478650254"
GROUP_A = "telegram-bot:-100111"
GROUP_B = "telegram-bot:-100222"
CHANNEL = "telegram-bot:-100333"
WA = "whatsapp:-100999"
NOW = time.time()


def _ctx(home: Path, db=None):
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(
            home=str(home), home_path=Path(home),
            workspace_dir=home / "workspace"),
    )


def _ns(**kwargs):
    args = argparse.Namespace(
        groups_action=None, groups_rest=[],
        community_action=None, community_rest=[],
        digest_action=None, digest_rest=[], json=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def _run(fn, args, ctx):
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = fn(args, ctx)
    return rc, buf.getvalue(), err.getvalue()


def _seed_surface(ws: Path):
    """Two groups + one channel + one WhatsApp group, as chat sees them."""
    from nomorals.social.chat.group_surface import default_surface

    surface = default_surface(ws)
    # Group A: four members, a live economy, one of everything waiting.
    for uid, name in (("u1", "Ada"), ("u2", "Bo"), ("u3", "Cy"),
                      ("u11", "Dee")):
        surface.record_join(GROUP_A, uid, username=name.lower(),
                            name=name, ts=NOW)
    surface.economy.award(GROUP_A, "u1", 100, "seed", now=NOW)
    surface.economy.treasury_bonus(GROUP_A, 50, "seed", now=NOW)
    surface.propose_spend(GROUP_A, kind="grant",
                          payload={"recipient": "u2", "amount": 600},
                          by="admin1", now=NOW)
    surface.stage_flag(GROUP_A, user_id="u9",
                       person={"id": "u9", "username": "bad"},
                       category="spam", note="seed", by="admin1", now=NOW)
    entry = surface.store.group(GROUP_A)
    entry.setdefault("join_requests", {})["req-1"] = {
        "id": "req-1", "state": "pending", "ts": NOW, "user_id": "u10"}
    entry["members"]["u11"]["acceptance"] = {
        "state": "restricted", "missing": ["username"],
        "reasons": ["username required"], "via": "join",
        "action": "restricted", "since": NOW, "grace_until": None,
        "welcome_pending": True}
    ban = surface.bans.ban("u12", reason="seed", group_id=GROUP_A,
                           banned_by="admin1")
    surface.bans.file_appeal("u12", ban.id, "sorry")
    # Group B: one member, nothing waiting, economy untouched.
    surface.record_join(GROUP_B, "u1", username="ada", name="Ada", ts=NOW)
    # Channel: claimed-channel marker + title, nothing queued/posted.
    centry = surface.store.group(CHANNEL)
    centry.setdefault("extras", {})["channel"] = {"title": "Signals"}
    # WhatsApp group: two members, for the digest board.
    for uid, name in (("w1", "Ada"), ("w2", "Ben")):
        surface.record_join(WA, uid, username=name.lower(),
                            name=name, ts=NOW)
    surface.store.save()
    return surface


def _seed_log(db, rows, *, key=WA):
    """rows: (sender_name, text, ts_offset_s). ConversationLog rows."""
    log = ConversationLog(db)
    log.ensure_conversation(key, title=key, agent="partner",
                            channel=key.split(":", 1)[0], now=NOW)
    for i, (sender, text, off) in enumerate(rows):
        log.append_message(row_id=f"m{i}", conversation_id=key,
                           role="user", content=text, name=sender,
                           model="", created_at=NOW + off)
    return log


class GroupsCliTests(unittest.TestCase):
    def test_board_lists_groups_channels_and_waiting(self):
        from nomorals.cmdline.commands.groups import _cmd_groups

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            rc, out, _ = _run(_cmd_groups, _ns(), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn(GROUP_A, out)
            self.assertIn("4 in the ledger", out)
            self.assertIn("treasury", out)
            self.assertIn("waiting: join 1", out)
            self.assertIn("restricted 1", out)
            self.assertIn("flags 1", out)
            self.assertIn("proposals 1", out)
            self.assertIn("appeals 1", out)
            self.assertIn(GROUP_B, out)
            self.assertIn("nothing waiting", out)
            self.assertIn("Signals", out)
            self.assertIn("0 queued", out)

    def test_board_json_parses(self):
        from nomorals.cmdline.commands.groups import _cmd_groups

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            rc, out, _ = _run(_cmd_groups, _ns(json=True), _ctx(home))
            self.assertEqual(rc, 0)
            data = json.loads(out)
            keys = [g["key"] for g in data["groups"]]
            self.assertIn(GROUP_A, keys)
            self.assertIn(GROUP_B, keys)
            self.assertEqual([c["key"] for c in data["channels"]],
                             [CHANNEL])

    def test_board_empty_store_says_so(self):
        from nomorals.cmdline.commands.groups import _cmd_groups

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            rc, out, _ = _run(_cmd_groups, _ns(), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("no groups", out)

    def test_unknown_action_is_usage_error(self):
        from nomorals.cmdline.commands.groups import _cmd_groups

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            rc, _out, err = _run(_cmd_groups,
                                 _ns(groups_action="explode"), _ctx(home))
            self.assertEqual(rc, 2)
            self.assertIn("nm groups", err)


class CommunityCliTests(unittest.TestCase):
    def _seed_community(self, ws: Path):
        surface = _seed_surface(ws)
        from nomorals.social.chat.group_community import set_community_tag

        out = set_community_tag(surface, GROUP_A, "c1", "Testers")
        self.assertTrue(out.get("ok"))
        out = set_community_tag(surface, GROUP_B, "c1", "Testers")
        self.assertTrue(out.get("ok"))
        surface.store.save()
        return surface

    def test_list_names_linked_community(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_community(home / "workspace")
            rc, out, _ = _run(_cmd_community, _ns(), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("Testers (c1)", out)
            self.assertIn("2 group(s)", out)

    def test_show_lists_member_groups(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_community(home / "workspace")
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="show",
                                  community_rest=["Testers"]), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("community Testers (c1)", out)
            self.assertIn(GROUP_A, out)
            self.assertIn(GROUP_B, out)
            self.assertIn("tag: manual", out)

    def test_show_unknown_names_the_fix(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_community(home / "workspace")
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="show",
                                  community_rest=["nope"]), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("no community", out)
            self.assertIn("/community lists", out)

    def test_status_reports_arm_state(self):
        from nomorals.cmdline.commands.community import _cmd_community
        from nomorals.social.mass_actions import MassArms, default_mass_root

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_community(home / "workspace")
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="status"), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("/community post: disarmed", out)
            self.assertIn("/community mirror: disarmed", out)
            MassArms(default_mass_root(home / "workspace")
                     / "arms.json").arm("community-post")
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="status"), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("/community post: armed", out)

    def test_post_and_mirror_refused_in_words(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_community(home / "workspace")
            for verb in ("post", "mirror"):
                rc, out, _ = _run(_cmd_community,
                                  _ns(community_action=verb,
                                      community_rest=["c1", "|", "hi"]),
                                  _ctx(home))
                self.assertEqual(rc, 0, verb)
                self.assertIn("live chat", out, verb)
            self.assertIn("not built", out)  # the mirror refusal

    def test_link_sync_member_arm_refused(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_community(home / "workspace")
            for verb, rest in (("link", ["g1", "c1"]),
                               ("unlink", ["g1"]),
                               ("sync", ["g1"]),
                               ("arm", []),
                               ("disarm", []),
                               ("member", ["add", "g1", "Eve"]),
                               ("member", ["link", "g1", "m1",
                                           "telegram", "eve"]),
                               ("member", ["unlink", "g1", "m1",
                                           "telegram"])):
                # NOTE: ("member", ["list", "g1"]) was refused here
                # until the member-list lane made the roster read a
                # console verb — see test_gapfix_cli_member_list.py.
                rc, out, _ = _run(_cmd_community,
                                  _ns(community_action=verb,
                                      community_rest=rest), _ctx(home))
                self.assertEqual(rc, 0, verb)
                self.assertIn("live chat", out, verb)
            # Nothing was linked/unlinked by the refusals.
            from nomorals.social.chat.group_community import community_of
            from nomorals.social.chat.group_surface import default_surface

            surface = default_surface(home / "workspace")
            self.assertIsNotNone(community_of(surface, GROUP_A))


class DigestCliTests(unittest.TestCase):
    def _ctx_with_log(self, home: Path, rows):
        db = Database(":memory:")
        db.migrate()
        _seed_log(db, rows)
        return _ctx(home, db=db)

    def test_board_includes_needing_group(self):
        from nomorals.cmdline.commands.digest import _cmd_digest

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            ctx = self._ctx_with_log(home, [
                ("Ada", "@Death can you look at this", -3600),
                ("Ben", "should we launch Friday?", -1800),
                ("Ada", "sorted, launching Friday ✅", -900),
            ])
            rc, out, _ = _run(_cmd_digest, _ns(), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("decisions needing you", out)

    def test_single_group_by_name(self):
        from nomorals.cmdline.commands.digest import _cmd_digest

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            ctx = self._ctx_with_log(home, [
                ("Ben", "should we launch Friday?", -1800),
            ])
            rc, out, _ = _run(_cmd_digest,
                              _ns(digest_action="-100999"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("decisions needing you", out)

    def test_no_need_says_so(self):
        from nomorals.cmdline.commands.digest import _cmd_digest

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            ctx = self._ctx_with_log(home, [
                ("Cid", "lunch at noon", -300),
            ])
            rc, out, _ = _run(_cmd_digest,
                              _ns(digest_action="-100999"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("nothing needs you", out)

    def test_policy_refused_in_words_and_unchanged(self):
        from nomorals.cmdline.commands.digest import _cmd_digest
        from nomorals.social.chat.group_surface import default_surface
        from nomorals.social.chat.whatsapp_digest import (
            digest_policy_for_group,
        )

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            surface = _seed_surface(home / "workspace")
            before = digest_policy_for_group(surface, WA)
            ctx = self._ctx_with_log(home, [])
            rc, out, _ = _run(_cmd_digest,
                              _ns(digest_action="policy",
                                  digest_rest=["always"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("live chat", out)
            surface = default_surface(home / "workspace")
            self.assertEqual(digest_policy_for_group(surface, WA), before)

    def test_unknown_group_names_it(self):
        from nomorals.cmdline.commands.digest import _cmd_digest

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            ctx = self._ctx_with_log(home, [])
            rc, out, _ = _run(_cmd_digest,
                              _ns(digest_action="no-such-group"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("no WhatsApp group matching", out)

    def test_board_json_parses(self):
        from nomorals.cmdline.commands.digest import _cmd_digest

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            ctx = self._ctx_with_log(home, [
                ("Ben", "should we launch Friday?", -1800),
            ])
            rc, out, _ = _run(_cmd_digest, _ns(json=True), ctx)
            self.assertEqual(rc, 0)
            data = json.loads(out)
            self.assertEqual(len(data["groups"]), 1)
            self.assertTrue(data["groups"][0]["digest"]["included"])


class ParserTests(unittest.TestCase):
    def test_parsers_exist_and_dispatch_covers_them(self):
        from nomorals.cmdline.dispatch import _COMMANDS
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        for name in ("community", "groups", "digest"):
            args = parser.parse_args([name])
            self.assertEqual(args.command, name)
            self.assertIn(name, _COMMANDS)

    def test_sibling_verbs_untouched(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        args = parser.parse_args(["museum"])
        self.assertEqual(args.command, "museum")
        args = parser.parse_args(["focus", "status"])
        self.assertEqual(args.command, "focus")
        self.assertEqual(args.focus_action, "status")
