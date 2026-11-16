"""Status/story radar (census §6 candidate #3).

Pattern briefs from the status/story vault: who posts a lot, who went
quiet, what themes run through statuses, what is worth a look. The
vault already journals every observed status (``status_arrival``) and
every saved story (``story``) — the radar is pure organ work on those
rows, plus a ``/radar`` chat surface and a morning-briefing section.

Discipline under test: the brief counts each post once (an arrival
and its saved ``status`` row are the same post); quiet means posted
in the baseline week and silent in the window; a pattern needs a
real concentration (modal weekday + part of day, >=3 posts, >=50%);
a theme word must show up across >=2 posters; an empty vault briefs
honestly instead of inventing activity.
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

# Fixed instants (explicit UTC datetimes — never a clock read).
NOW = datetime.datetime(2026, 10, 14, 12, 0,
                        tzinfo=datetime.UTC).timestamp()
FRIDAY = datetime.datetime(2026, 10, 9, 18, 0,
                           tzinfo=datetime.UTC).timestamp()


def _arrival(poster: str, ts: float, *, platform: str = "whatsapp",
             text: str = "", source_id: str = "") -> dict:
    return {
        "ts": ts, "kind": "status_arrival",
        "account": f"{platform}:primary", "platform": platform,
        "counterpart_id": f"id-{poster.lower()}",
        "poster_name": poster, "media_kind": "",
        "source_id": source_id or f"{poster}-{int(ts)}",
        "path": "", "bytes": 0, "post_ts": ts, "text": text,
        "media": [{"kind": "image"}], "media_missing": False,
        "note": "status post observed",
    }


def _story(poster: str, ts: float, *, text: str = "",
           source_id: str = "") -> dict:
    return {
        "ts": ts, "kind": "story", "account": "telegram:111",
        "platform": "telegram",
        "counterpart_id": f"id-{poster.lower()}",
        "poster_name": poster, "media_kind": "photo",
        "source_id": source_id or f"{poster}-{int(ts)}",
        "path": "/vault/x.jpg", "bytes": 10, "text": text,
    }


def _status_row(poster: str, ts: float, *, source_id: str) -> dict:
    return {
        "ts": ts, "kind": "status", "account": "whatsapp:primary",
        "platform": "whatsapp",
        "counterpart_id": f"id-{poster.lower()}",
        "poster_name": poster, "media_kind": "image",
        "source_id": source_id, "path": "/vault/y.jpg", "bytes": 10,
    }


class OrganTests(unittest.TestCase):
    def test_counts_and_platforms(self):
        from nomorals.social.status_radar import build_brief

        rows = [_arrival("Ada", NOW - 3600 * i, text="hi")
                for i in range(5)]
        rows += [_story("Chidi", NOW - 7200 * i) for i in range(3)]
        brief = build_brief(rows, now=NOW)
        self.assertEqual(brief.total, 8)
        self.assertEqual(brief.posters, 2)
        self.assertEqual(brief.per_platform,
                         {"whatsapp": 5, "telegram": 3})
        self.assertEqual([(p.name, p.count) for p in brief.top],
                         [("Ada", 5), ("Chidi", 3)])

    def test_dedupe_arrival_beats_saved_status(self):
        from nomorals.social.status_radar import build_brief

        rows = [_arrival("Ada", NOW - 3600, source_id="s1"),
                _status_row("Ada", NOW - 3600, source_id="s1")]
        brief = build_brief(rows, now=NOW)
        self.assertEqual(brief.total, 1)

    def test_quiet_poster(self):
        from nomorals.social.status_radar import build_brief

        rows = [_arrival("Ada", NOW - 3600)]
        # Bola posted 3 times in the baseline week, nothing since.
        rows += [_arrival("Bola", NOW - 8 * 86400),
                 _arrival("Bola", NOW - 9 * 86400),
                 _arrival("Bola", NOW - 10 * 86400)]
        # One baseline post is not a pattern — not quiet-listed.
        rows += [_arrival("Solo", NOW - 8 * 86400)]
        brief = build_brief(rows, now=NOW)
        quiet = [q.name for q in brief.quiet]
        self.assertEqual(quiet, ["Bola"])
        bola = brief.quiet[0]
        self.assertEqual(bola.baseline_count, 3)
        self.assertGreaterEqual(bola.days_quiet, 7)

    def test_weekday_pattern(self):
        from nomorals.social.status_radar import build_brief

        # Four Friday-evening (WAT) posts, one Monday post.
        rows = [_arrival("Ada", FRIDAY + 3600 * i,
                         source_id=f"fri-{i}") for i in range(4)]
        rows += [_arrival("Ada", NOW - 2 * 86400, source_id="mon-1")]
        brief = build_brief(rows, now=NOW)
        self.assertEqual(len(brief.patterns), 1)
        pat = brief.patterns[0]
        self.assertEqual(pat.name, "Ada")
        self.assertEqual(pat.weekday, "Friday")
        self.assertEqual(pat.part, "evening")
        self.assertEqual((pat.count, pat.total), (4, 5))

    def test_no_pattern_when_spread(self):
        from nomorals.social.status_radar import build_brief

        rows = [_arrival("Ada", NOW - 86400 * i,
                         source_id=f"d{i}") for i in range(1, 6)]
        brief = build_brief(rows, now=NOW)
        self.assertEqual(brief.patterns, [])

    def test_themes_across_posters(self):
        from nomorals.social.status_radar import build_brief

        rows = [
            _arrival("Ada", NOW - 3600, text="church service was good"),
            _arrival("Bola", NOW - 7200, text="at church with family"),
            _arrival("Ada", NOW - 10800, text="my own private words"),
        ]
        brief = build_brief(rows, now=NOW)
        words = [t.word for t in brief.themes]
        self.assertIn("church", words)
        church = next(t for t in brief.themes if t.word == "church")
        self.assertEqual(sorted(church.posters), ["Ada", "Bola"])
        # A word from one poster only is not a theme.
        self.assertNotIn("private", words)

    def test_worth_a_look(self):
        from nomorals.social.status_radar import build_brief

        rows = [
            _story("Chidi", NOW - 3600, text="morning run done"),
            _arrival("Ada", NOW - 3 * 86400, text="old news"),
        ]
        brief = build_brief(rows, now=NOW)
        self.assertEqual([w.name for w in brief.worth], ["Chidi"])
        self.assertIn("morning run", brief.worth[0].snippet)

    def test_new_poster(self):
        from nomorals.social.status_radar import build_brief

        rows = [_arrival("Ada", NOW - 3600),
                _arrival("Newbie", NOW - 7200)]
        brief = build_brief(rows, now=NOW)
        self.assertEqual([p.name for p in brief.new],
                         ["Ada", "Newbie"])

    def test_empty_vault(self):
        from nomorals.social.status_radar import build_brief, render_brief

        brief = build_brief([], now=NOW)
        self.assertEqual(brief.total, 0)
        self.assertIn("No statuses", render_brief(brief))

    def test_window_param(self):
        from nomorals.social.status_radar import build_brief

        rows = [_arrival("Ada", NOW - 10 * 86400),
                _arrival("Ada", NOW - 3600)]
        self.assertEqual(build_brief(rows, now=NOW).total, 1)
        self.assertEqual(
            build_brief(rows, now=NOW, window_days=14).total, 2)

    def test_render_shape(self):
        from nomorals.social.status_radar import build_brief, render_brief

        rows = [_arrival("Ada", NOW - 3600, text="hello there")]
        text = render_brief(build_brief(rows, now=NOW))
        self.assertIn("last 7 days", text)
        self.assertIn("Ada", text)
        self.assertIn("Most active", text)

    def test_load_journal_rows_skips_bad_lines(self):
        from nomorals.social.status_radar import load_journal_rows

        with tempfile.TemporaryDirectory(prefix="nm-test-radar-") as d:
            path = Path(d) / "journal.jsonl"
            path.write_text(
                json.dumps({"kind": "status_arrival", "ts": 1.0}) + "\n"
                + "not json\n"
                + json.dumps({"kind": "story", "ts": 2.0}) + "\n",
                encoding="utf-8")
            rows = load_journal_rows(path)
            self.assertEqual(len(rows), 2)


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


class _RadarCase(FrozenClockMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-radar-")
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

    def _seed(self, rows):
        vault_dir = self.tmp_path / "data" / "social" / "personal"
        vault_dir.mkdir(parents=True, exist_ok=True)
        path = vault_dir / "journal.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

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


class HandlerTests(_RadarCase):
    def test_radar_brief_over_seeded_vault(self):
        rows = [_arrival("Ada", self.now - 3600, text="hello there",
                         source_id="a1"),
                _arrival("Ada", self.now - 7200, source_id="a2"),
                _story("Chidi", self.now - 10800, text="morning run")]
        self._seed(rows)
        out = self._reply("/radar")
        self.assertIn("Ada", out)
        self.assertIn("Chidi", out)
        self.assertIn("Most active", out)

    def test_radar_today_window(self):
        rows = [_arrival("Ada", self.now - 3600, source_id="a1"),
                _arrival("Ada", self.now - 3 * 86400, source_id="a2")]
        self._seed(rows)
        out = self._reply("/radar today")
        self.assertIn("last 24 hours", out)
        self.assertIn("1 post", out)

    def test_radar_name_filter(self):
        rows = [_arrival("Ada", self.now - 3600, source_id="a1"),
                _arrival("Bola", self.now - 7200, source_id="b1")]
        self._seed(rows)
        out = self._reply("/radar ada")
        self.assertIn("Ada", out)
        self.assertNotIn("Bola", out)

    def test_radar_unknown_name(self):
        self._seed([_arrival("Ada", self.now - 3600, source_id="a1")])
        out = self._reply("/radar zzz")
        self.assertIn("No one matching", out)

    def test_radar_empty_vault(self):
        out = self._reply("/radar")
        self.assertIn("No statuses", out)

    def test_radar_window_days(self):
        rows = [_arrival("Ada", self.now - 10 * 86400, source_id="a1"),
                _arrival("Ada", self.now - 3600, source_id="a2")]
        self._seed(rows)
        out = self._reply("/radar 14d")
        self.assertIn("last 14 days", out)
        self.assertIn("2 posts", out)


class RegistryTests(unittest.TestCase):
    def test_radar_registered(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
        )

        self.assertIn("radar", CONTROL_COMMANDS)
        page = COMMAND_DETAILS["radar"]
        for key in ("what", "usage", "example"):
            self.assertTrue(page[key], key)
        self.assertIn("radar", LIST_ONELINERS)
        slots = [g for g, kinds in LIST_GROUPS if "radar" in kinds]
        self.assertEqual(len(slots), 1)
        self.assertNotIn("radar",
                         __import__("nomorals.social.chat.control",
                                    fromlist=["PUBLIC_COMMANDS"])
                         .PUBLIC_COMMANDS)


class ProviderTests(unittest.TestCase):
    def _ctx(self, tmp_path: Path):
        return SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(tmp_path)))

    def test_provider_section(self):
        from nomorals.agents.morning_briefing import (
            StatusRadarProvider,
        )

        with tempfile.TemporaryDirectory(prefix="nm-test-radar-") as d:
            tmp = Path(d)
            vault_dir = tmp / "data" / "social" / "personal"
            vault_dir.mkdir(parents=True, exist_ok=True)
            rows = [_arrival("Ada", NOW - 3600, text="hello there",
                             source_id="a1"),
                    _arrival("Ada", NOW - 7200, source_id="a2")]
            with (vault_dir / "journal.jsonl").open(
                    "a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            provider = StatusRadarProvider()
            with mock.patch(
                    "nomorals.agents.morning_briefing.time") as mocked:
                mocked.time.return_value = NOW
                section = provider.collect(self._ctx(tmp), since=0.0)
            self.assertIsNotNone(section)
            assert section is not None
            self.assertEqual(section.name, "status_radar")
            self.assertIn("Ada", "\n".join(section.lines))

    def test_provider_empty_returns_none(self):
        from nomorals.agents.morning_briefing import (
            StatusRadarProvider,
        )

        with tempfile.TemporaryDirectory(prefix="nm-test-radar-") as d:
            provider = StatusRadarProvider()
            section = provider.collect(self._ctx(Path(d)), since=0.0)
            self.assertIsNone(section)


if __name__ == "__main__":
    unittest.main()
