"""``/airports`` chat verb (airport-refresh console-only remainder).

``nm airports`` (ship 0bd2bd1f) served status/refresh from the
console, but no ``/airports`` chat command existed — while the
functions it needs (``travel/refresh.py``'s ``refresh_airports`` /
``read_refresh_state``) were already proven by the console verb.
These tests pin the thin owner-only verb in the exact ``/analytics``
pattern: registration everywhere, status over a seeded refresh
state, refresh on demand reporting honestly, never raising, and no
live chat needed — nothing here sends a file back.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    PUBLIC_COMMANDS,
    caller_may_run,
    list_catalog,
    parse_control,
)
from nomorals.social.chat.telegram_menu import validate_menu
from tests.test_gapfix_airport_refresh import _big_dump
from tests.test_gapfix_museum_command import _msg


class AirportsRegistrationTests(unittest.TestCase):
    def test_registered_everywhere(self):
        self.assertIn("airports", CONTROL_COMMANDS)
        self.assertIn("airports", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["airports"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("airports", grouped)
        self.assertEqual(grouped.count("airports"), 1)
        self.assertIn("/airports refresh",
                      COMMAND_DETAILS["airports"]["usage"])

    def test_owner_only(self):
        self.assertNotIn("airports", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("airports", False))
        self.assertTrue(caller_may_run("airports", True))

    def test_parse_control(self):
        cmd = parse_control("/airports refresh")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "airports")
        self.assertEqual(cmd.tail, "refresh")

    def test_listed_and_in_menu(self):
        self.assertIn("/airports", list_catalog())
        ok, problems = validate_menu()
        self.assertTrue(ok, problems)


class NextRefreshRunTests(unittest.TestCase):
    @staticmethod
    def _next(*args: Any) -> str:
        # Function-level import: on the pristine base this name does
        # not exist, so these tests fail (rather than erroring at
        # collection) until the helper lands.
        from nomorals.travel.refresh import next_refresh_run

        return next_refresh_run(*args)

    @staticmethod
    def _epoch(year: int, month: int, day: int,
               hour: int = 0, minute: int = 0) -> float:
        return time.mktime((year, month, day, hour, minute, 0, 0, 0, -1))

    def test_mid_month_goes_to_next_first(self):
        self.assertEqual(
            self._next(self._epoch(2026, 11, 17, 12)),
            "2026-12-01 04:00")

    def test_first_before_0400_is_today(self):
        self.assertEqual(
            self._next(self._epoch(2026, 11, 1, 3)),
            "2026-11-01 04:00")

    def test_first_after_0400_goes_to_next_month(self):
        self.assertEqual(
            self._next(self._epoch(2026, 11, 1, 5)),
            "2026-12-01 04:00")

    def test_first_at_0400_goes_to_next_month(self):
        self.assertEqual(
            self._next(self._epoch(2026, 11, 1, 4)),
            "2026-12-01 04:00")

    def test_december_rolls_to_january(self):
        self.assertEqual(
            self._next(self._epoch(2026, 12, 15, 10)),
            "2027-01-01 04:00")

    def test_default_is_a_first_at_0400(self):
        nxt = self._next()
        self.assertTrue(nxt.endswith("01 04:00"), nxt)


class AirportsChatTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / "workspace" / "airportsverb-verify"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-airports-chat-", dir=str(scratch))
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(
            settings, with_executor=False, with_router=False,
            with_tools=True)
        self.context.db.migrate()

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.runtime = PartnerRuntime(self.context, gateway=None)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True,
             chat_key: str = "telegram:999") -> str:
        return self.runtime.handle_control(
            text, chat_key, message=_msg(text, chat_key=chat_key,
                                         owner=owner))

    def _seed_state(self, state: dict[str, Any]) -> None:
        path = Path(self.tmp.name) / "data" / "airports_refresh.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")

    def test_status_never_refreshed(self):
        reply = self._run("/airports")
        self.assertIn("airport table:", reply)
        self.assertIn("IATA codes", reply)
        self.assertIn("last refresh: never", reply)
        self.assertIn("next run:", reply)
        self.assertIn("schedule: cron 0 4 1 * *", reply)

    def test_status_seeded_ok_state(self):
        ran_at = 1763366400.0
        self._seed_state({"ran_at": ran_at, "ok": True, "rows": 9053,
                          "countries": 239, "dropped": 12,
                          "source": "https://example.invalid/dump.csv",
                          "source_sha256": "", "error": ""})
        reply = self._run("/airports status")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ran_at))
        self.assertIn(f"last refresh: {when}", reply)
        self.assertIn("9053 rows", reply)
        self.assertIn("239 countries", reply)
        self.assertIn("next run:", reply)

    def test_status_seeded_failed_state(self):
        ran_at = 1763366400.0
        self._seed_state({"ran_at": ran_at, "ok": False, "rows": 0,
                          "countries": 0, "dropped": 0,
                          "source": "https://example.invalid/dump.csv",
                          "source_sha256": "",
                          "error": "fetch failed: boom"})
        reply = self._run("/airports")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ran_at))
        self.assertIn(f"last refresh: {when}", reply)
        self.assertIn("failed: fetch failed: boom", reply)

    def test_refresh_runs_now(self):
        from nomorals.travel import airports

        data = Path(self.tmp.name) / "airports_iata_country.csv"
        with (mock.patch("nomorals.travel.refresh._fetch_dump",
                         return_value=_big_dump(6000)),
              mock.patch("nomorals.travel.refresh.DATA_PATH", data),
              mock.patch.object(airports, "AIRPORTS_DATA_PATH", data)):
            try:
                reply = self._run("/airports refresh")
            finally:
                airports.invalidate_cache()
        self.assertIn("refreshed: 6000 IATA codes", reply)
        self.assertIn("676 countries", reply)
        self.assertIn("0 rows dropped", reply)
        # ... and the status afterwards reads the new state.
        status = self._run("/airports")
        self.assertIn("6000 rows", status)
        self.assertIn("676 countries", status)

    def test_refresh_failure_reports_and_keeps_old_table(self):
        from nomorals.travel.airports import airport_country_count

        before = airport_country_count()
        with mock.patch("nomorals.travel.refresh._fetch_dump",
                        side_effect=RuntimeError("network down")):
            reply = self._run("/airports refresh")
        self.assertIn("refresh failed: fetch failed: network down", reply)
        self.assertEqual(airport_country_count(), before)

    def test_refresh_never_raises(self):
        with mock.patch(
                "nomorals.travel.refresh.run_airport_refresh_tick_once",
                side_effect=RuntimeError("tick exploded")):
            reply = self.runtime._control_airports(
                "refresh", message=None)
        # The tick is documented never-raise; if it ever does, the
        # handler still answers in words instead of crashing chat.
        self.assertIn("refresh failed: tick exploded", reply)

    def test_unknown_verb_usage(self):
        reply = self._run("/airports frobnicate")
        self.assertIn("usage: /airports [status | refresh]", reply)

    def test_non_owner_refused(self):
        reply = self._run("/airports", owner=False)
        self.assertNotIn("airport table:", reply)
        reply = self._run("/airports refresh", owner=False)
        self.assertNotIn("refreshed:", reply)
        self.assertNotIn("refresh failed:", reply)

    def test_no_live_chat_needed(self):
        # Unlike /analytics export, nothing here sends a file back,
        # so the handler answers with message=None instead of
        # refusing — status and refresh both work without a chat.
        reply = self.runtime._control_airports("status", message=None)
        self.assertIn("airport table:", reply)
        self.assertIn("last refresh: never", reply)
        with mock.patch("nomorals.travel.refresh._fetch_dump",
                        side_effect=RuntimeError("network down")):
            reply = self.runtime._control_airports("refresh",
                                                   message=None)
        self.assertIn("refresh failed:", reply)


if __name__ == "__main__":
    unittest.main()
