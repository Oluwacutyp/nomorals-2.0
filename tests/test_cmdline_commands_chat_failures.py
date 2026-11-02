"""Tail-100 pins: nomorals/cmdline/commands/chat.py census sites.

This module is the read side of `nm chat` — setup doctor + outbox
viewer. Its probes are honest by construction: a check that cannot
run reports failure/DOWN/empty rather than dressing uncertainty up
as health. All four sites are held as exclusions:

107 — ``_check_module``: find_spec crash -> False. An import probe
that explodes is a module the caller cannot rely on; the doctor row
built on it reads "not installed", the fail-closed direction for a
setup check.

167 — ``_doctor`` whatsapp rung: bridge socket connect raises
OSError -> reachable stays False, the row reports bridge DOWN and
ok=False. A connect error *is* the probe's negative result, not
missing data. PARENT: flagged-accepted remainder (no noqa in code).

213 — ``_outbox_path``: settings.resolve() crash -> the raw
(unresolved) path. Degradation to the documented default value;
the viewer then reads that path or reports it empty.

223 — ``_read_outbox``: a corrupt/unreadable outbox file -> []
— the same value a missing outbox returns. Read-side display
degrade on a queue the command only views; it claims nothing was
sent and fabricates no entries.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.cmdline.commands import chat as C


class TestCheckModule(unittest.TestCase):
    """107: find_spec crash -> False (fail-closed import probe)."""

    def test_find_spec_crash_is_false(self):
        with mock.patch("importlib.util.find_spec",
                        side_effect=RuntimeError("spec boom")):
            self.assertFalse(C._check_module("nomorals"))

    def test_missing_module_is_false(self):
        self.assertFalse(C._check_module("nomorals.no_such_module_xyz"))

    def test_present_module_is_true(self):
        # Control: the healthy probe still answers True.
        self.assertTrue(C._check_module("json"))


class TestDoctorWhatsappDown(unittest.TestCase):
    """167: bridge connect OSError -> row says DOWN, ok=False."""

    def _settings(self) -> SimpleNamespace:
        chat = SimpleNamespace(
            whatsapp_enabled=True, whatsapp_host="127.0.0.1",
            whatsapp_port=8787)
        return SimpleNamespace(chat=chat)

    def test_connect_crash_reports_down(self):
        with mock.patch.object(
                C.socket, "create_connection",
                side_effect=OSError("connection refused")):
            rows = C._doctor(self._settings())
        wa = [r for r in rows if r["platform"] == "whatsapp"]
        self.assertEqual(len(wa), 1)
        self.assertFalse(wa[0]["ok"])
        self.assertIn("DOWN", wa[0]["detail"])

    def test_disabled_whatsapp_row_is_ok(self):
        # Control: disabled is a clean, passing row — the DOWN above
        # comes from the probe, not from the row machinery.
        rows = C._doctor(SimpleNamespace(chat=None))
        wa = [r for r in rows if r["platform"] == "whatsapp"]
        self.assertEqual(len(wa), 1)
        self.assertTrue(wa[0]["ok"])


class TestOutboxPath(unittest.TestCase):
    """213: resolve crash -> the raw default path."""

    def test_resolve_crash_returns_raw(self):
        def boom(raw):
            raise RuntimeError("resolve boom")

        settings = SimpleNamespace(chat=None, resolve=boom)
        self.assertEqual(C._outbox_path(settings), "data/chat/outbox")

    def test_resolve_success_is_used(self):
        # Control: a working resolver's answer is the path.
        settings = SimpleNamespace(
            chat=None, resolve=lambda raw: "/abs/" + raw)
        self.assertEqual(C._outbox_path(settings),
                         "/abs/data/chat/outbox")

    def test_no_settings_returns_raw(self):
        self.assertEqual(C._outbox_path(None), "data/chat/outbox")


class TestReadOutbox(unittest.TestCase):
    """223: corrupt outbox -> [] (same as a missing outbox)."""

    def test_corrupt_file_reads_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "outbox"
            path.write_text("{not json\n", encoding="utf-8")
            self.assertEqual(C._read_outbox(str(path)), [])

    def test_missing_file_reads_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(
                C._read_outbox(str(Path(td) / "absent")), [])

    def test_valid_file_reads_entries(self):
        # Control: real entries come back as themselves.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "outbox"
            rows = [{"to": "a", "text": "hi"}, {"to": "b", "text": "yo"}]
            path.write_text(
                "".join(json.dumps(r) + "\n" for r in rows),
                encoding="utf-8")
            self.assertEqual(C._read_outbox(str(path)), rows)


if __name__ == "__main__":
    unittest.main()
