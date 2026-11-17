"""``nm community member list`` — the console twin of chat
``/community member list``.

Failing-first: on the pristine base ``member`` is a flat refusal
(``_REFUSALS`` in ``cmdline/commands/community.py``), so every
list test here fails there — the refusal text carries no roster.

The read mirrors chat's ``list`` branch
(``runtime_group._community_member``) over the same per-group
``BridgeStore``: same words, same roster rendering, minus the
admin gate (the console is the owner's seat) and minus the
console session-map leg of ``_community_resolve`` (no session on
the console — the cold gN/cN ordinal, the exact key and the unique
substring match resolve directly). ``add``/``link``/``unlink``
change the roster, so they stay chat-only and are refused in
words, the way bundle #3 refused the other write verbs.
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
from unittest.mock import patch

from nomorals.community.bridges import BridgeStore

GROUP_A = "telegram-bot:-100111"
GROUP_B = "telegram-bot:-100222"
CHANNEL = "telegram-bot:-100333"
NOW = time.time()


def _ctx(home: Path):
    return SimpleNamespace(
        db=None,
        settings=SimpleNamespace(
            home=str(home), home_path=Path(home),
            workspace_dir=home / "workspace"),
    )


def _ns(**kwargs):
    args = argparse.Namespace(
        community_action=None, community_rest=[], json=False)
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
    """Two groups + one channel, as chat sees them."""
    from nomorals.social.chat.group_surface import default_surface

    surface = default_surface(ws)
    surface.record_join(GROUP_A, "u1", username="ada",
                        name="Ada", ts=NOW)
    surface.record_join(GROUP_B, "u1", username="ada",
                        name="Ada", ts=NOW)
    centry = surface.store.group(CHANNEL)
    centry.setdefault("extras", {})["channel"] = {"title": "Signals"}
    surface.store.save()
    return surface


def _seed_roster(bridge_dir: Path, key: str, members):
    """members: (display_name, {platform: platform_user_id})."""
    store = BridgeStore(str(bridge_dir))
    roster = store.load(key)
    for name, links in members:
        member = roster.add_member(name)
        for platform, platform_user_id in links.items():
            roster.link(member.id, platform, platform_user_id)
    store.save(key, roster)
    return store


class MemberListTests(unittest.TestCase):
    def _bridge(self, tmp: Path):
        """Patch the module seam onto a tmp store; returns the dir."""
        bridge_dir = tmp / "bridges"
        patcher = patch(
            "nomorals.cmdline.commands.community._bridge_store",
            lambda: BridgeStore(str(bridge_dir)))
        patcher.start()
        self.addCleanup(patcher.stop)
        return bridge_dir

    def test_list_renders_linked_and_unlinked(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            bridge_dir = self._bridge(home)
            _seed_roster(bridge_dir, GROUP_A, [
                ("Ada", {"telegram": "ada_tg",
                         "whatsapp": "+234800"}),
                ("Bo", {}),
            ])
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list", "g1"]),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn(f"members in {GROUP_A}:", out)
            self.assertIn("Ada", out)
            self.assertIn("telegram:ada_tg", out)
            self.assertIn("whatsapp:+234800", out)
            self.assertIn("Bo", out)
            self.assertIn("no linked identities", out)

    def test_exact_key_resolves(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            bridge_dir = self._bridge(home)
            _seed_roster(bridge_dir, GROUP_B, [("Cy", {})])
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list", GROUP_B]),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn(f"members in {GROUP_B}:", out)
            self.assertIn("Cy", out)

    def test_unique_substring_resolves(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            bridge_dir = self._bridge(home)
            _seed_roster(bridge_dir, GROUP_A, [("Ada", {})])
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list", "100111"]),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn(f"members in {GROUP_A}:", out)

    def test_channel_ordinal_resolves(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            bridge_dir = self._bridge(home)
            _seed_roster(bridge_dir, CHANNEL, [("Dee", {})])
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list", "c1"]),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn(f"members in {CHANNEL}:", out)
            self.assertIn("Dee", out)

    def test_empty_roster_names_the_add(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            self._bridge(home)
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list", "g2"]),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn(f"no members in {GROUP_B} yet", out)
            self.assertIn("/community member add", out)

    def test_unknown_group_in_chat_words(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            self._bridge(home)
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list", "nope"]),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("that group handle expired", out)
            self.assertIn("community board", out)

    def test_ambiguous_substring_names_the_fix(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            self._bridge(home)
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list",
                                                  "telegram-bot:-100"]),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("more than one group matches", out)
            self.assertIn("g1, g2", out)

    def test_json_shape(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            bridge_dir = self._bridge(home)
            _seed_roster(bridge_dir, GROUP_A, [
                ("Ada", {"telegram": "ada_tg"}),
                ("Bo", {}),
            ])
            rc, out, _ = _run(_cmd_community,
                              _ns(community_action="member",
                                  community_rest=["list", "g1"],
                                  json=True),
                              _ctx(home))
            self.assertEqual(rc, 0)
            data = json.loads(out)
            self.assertEqual(data["group"], GROUP_A)
            by_name = {m["display_name"]: m for m in data["members"]}
            self.assertEqual(by_name["Ada"]["platforms"],
                             {"telegram": "ada_tg"})
            self.assertEqual(by_name["Bo"]["platforms"], {})
            self.assertTrue(all(m["id"] for m in data["members"]))

    def test_add_link_unlink_refused_and_roster_unchanged(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            bridge_dir = self._bridge(home)
            _seed_roster(bridge_dir, GROUP_A, [("Ada", {})])
            for rest in (["add", "g1", "Eve"],
                         ["link", "g1", "member-x", "telegram", "eve"],
                         ["unlink", "g1", "member-x", "telegram"]):
                rc, out, _ = _run(_cmd_community,
                                  _ns(community_action="member",
                                      community_rest=rest),
                                  _ctx(home))
                self.assertEqual(rc, 0, rest)
                self.assertIn("live chat", out, rest)
            roster = BridgeStore(str(bridge_dir)).load(GROUP_A)
            self.assertEqual(
                [m.display_name for m in roster.members], ["Ada"])

    def test_no_token_is_usage_error(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            self._bridge(home)
            rc, _out, err = _run(_cmd_community,
                                 _ns(community_action="member",
                                     community_rest=["list"]),
                                 _ctx(home))
            self.assertEqual(rc, 2)
            self.assertIn("nm community member list", err)

    def test_bare_member_is_usage_error(self):
        from nomorals.cmdline.commands.community import _cmd_community

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_surface(home / "workspace")
            self._bridge(home)
            rc, _out, err = _run(_cmd_community,
                                 _ns(community_action="member",
                                     community_rest=[]),
                                 _ctx(home))
            self.assertEqual(rc, 2)
            self.assertIn("nm community member list", err)


class ParserTests(unittest.TestCase):
    def test_member_list_parses_and_dispatch_covers_it(self):
        from nomorals.cmdline.dispatch import _COMMANDS
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        args = parser.parse_args(
            ["community", "member", "list", "g1"])
        self.assertEqual(args.command, "community")
        self.assertEqual(args.community_action, "member")
        self.assertEqual(args.community_rest, ["list", "g1"])
        self.assertIn("community", _COMMANDS)


if __name__ == "__main__":
    unittest.main()
