"""Tests for monitor target validation and auto-disable on permanent failure.

Covers:
- `MonitorAgent.add()` rejects file watches for nonexistent paths (fail fast)
- `MonitorAgent.add()` still allows URL watches without validation
- `_note_error` auto-disables after _ERROR_DISABLE_STREAK consecutive failures
- Auto-disable publishes a notification
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from nomorals.agents import monitor as M
from nomorals.agents.monitor import MonitorAgent


def _make_agent():
    """Minimal MonitorAgent with in-memory DB."""
    from nomorals.storage.db import Database
    db = Database(":memory:")
    # Create the monitors table (minimal schema for our tests)
    db.execute("""
        CREATE TABLE IF NOT EXISTS monitors (
            id TEXT PRIMARY KEY, target TEXT, kind TEXT, watch TEXT,
            interval_s REAL, enabled INTEGER, created_at REAL, last_ts REAL,
            last_hash TEXT, last_size INTEGER, last_ok INTEGER,
            error_streak INTEGER DEFAULT 0, last_content TEXT,
            last_change_ts REAL, webhook_url TEXT, webhook_secret TEXT,
            min_alert_gap_s REAL, auto_decode INTEGER, volatile TEXT,
            last_alert_ts REAL DEFAULT 0
        )
    """)
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()))
    agent = MonitorAgent.__new__(MonitorAgent)
    agent.context = ctx
    agent.db = db
    agent.notifier = MagicMock()
    return agent


class TestFileValidation(unittest.TestCase):
    def test_rejects_nonexistent_file(self):
        agent = _make_agent()
        with self.assertRaises(ValueError) as cm:
            agent.add("/nonexistent/path/to/file.txt", kind="file")
        self.assertIn("not a readable workspace file", str(cm.exception))

    def test_rejects_inferred_file_kind(self):
        # No kind specified + not a URL = inferred as file
        agent = _make_agent()
        with self.assertRaises(ValueError) as cm:
            agent.add("btc price and xauusd market direction")
        self.assertIn("not a readable workspace file", str(cm.exception))

    def test_accepts_existing_file(self):
        agent = _make_agent()
        workspace = Path(agent.context.settings.workspace_dir)
        path = workspace / "watchme.txt"
        path.write_text("test")
        result = agent.add("watchme.txt", kind="file")
        self.assertEqual(result["kind"], "file")
        self.assertTrue(result["created"])

    def test_accepts_url_without_validation(self):
        agent = _make_agent()
        result = agent.add("https://example.com/status", kind="url")
        self.assertEqual(result["kind"], "url")

    def test_rejects_page_without_url(self):
        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent.add("not a url", kind="page")


class TestAutoDisable(unittest.TestCase):
    def test_disables_after_streak(self):
        agent = _make_agent()
        # Insert a monitor directly (bypassing add validation)
        agent.db.execute(
            "INSERT INTO monitors (id, target, kind, watch, interval_s, enabled, "
            "created_at, last_ts, error_streak) VALUES (?,?,?,?,?,?,?,?,?)",
            ("test1", "https://dead.example.com", "url", "content", 60, 1,
             time.time(), 0, M._ERROR_DISABLE_STREAK - 1),
        )
        row = agent.db.query_one("SELECT * FROM monitors WHERE id=?", ("test1",))
        agent._note_error(dict(row), "connection refused")

        updated = agent.db.query_one("SELECT enabled FROM monitors WHERE id=?", ("test1",))
        self.assertEqual(updated["enabled"], 0)
        # Should have published an auto-disabled notification
        agent.notifier.publish.assert_called_once()
        call_kwargs = agent.notifier.publish.call_args[1]
        self.assertIn("auto-disabled", call_kwargs["title"])

    def test_alerts_at_three_but_not_disabled(self):
        agent = _make_agent()
        agent.db.execute(
            "INSERT INTO monitors (id, target, kind, watch, interval_s, enabled, "
            "created_at, last_ts, error_streak) VALUES (?,?,?,?,?,?,?,?,?)",
            ("test2", "https://flaky.example.com", "url", "content", 60, 1,
             time.time(), 0, 2),  # 2 failures, this is #3
        )
        row = agent.db.query_one("SELECT * FROM monitors WHERE id=?", ("test2",))
        agent._note_error(dict(row), "timeout")

        updated = agent.db.query_one(
            "SELECT enabled, error_streak FROM monitors WHERE id=?", ("test2",))
        self.assertEqual(updated["enabled"], 1)  # still enabled
        self.assertEqual(updated["error_streak"], 3)
        # Should have published "watch down" (not auto-disabled)
        agent.notifier.publish.assert_called_once()
        call_kwargs = agent.notifier.publish.call_args[1]
        self.assertIn("watch down", call_kwargs["title"])


if __name__ == "__main__":
    unittest.main()
