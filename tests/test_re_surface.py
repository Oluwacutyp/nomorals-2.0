"""RE Pass 1 — surfaces: engine wiring to the journal + memory
seams, the registry tools, the /re chat handler, `nm re`, and the
doctor matrix.

The journal/memory assertions pin the outdo requirement: a finding
lands in Devon's decision journal and memory as an Evidence record
— never as bare text, and never silently dropped.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from nomorals.agents.reverse import ReEngine
from nomorals.storage.db import Database
from tests.test_re_inspectors import build_elf64, build_pe


class _FakeJournal:
    def __init__(self):
        self.entries = []

    def record(self, decision, context="", signals=None, tags=None,
               **kw):
        self.entries.append({"decision": decision, "context": context,
                             "tags": list(tags or [])})
        return len(self.entries)


class _FakeMemory:
    def __init__(self):
        self.items = []

    def remember(self, text, kind="episode", source=""):
        self.items.append({"text": text, "kind": kind,
                           "source": source})
        return len(self.items)


class _TmpHome:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="re-surface-")
        return self.dir

    def __exit__(self, *exc):
        return False


class EngineWiringTest(unittest.TestCase):
    def _engine(self):
        journal, memory = _FakeJournal(), _FakeMemory()
        engine = ReEngine(db=Database(":memory:"), journal=journal,
                          memory=memory)
        return engine, journal, memory

    def test_inspect_journals_and_remembers_evidence(self):
        engine, journal, memory = self._engine()
        with _TmpHome() as home:
            path = os.path.join(home, "sample")
            with open(path, "wb") as fh:
                fh.write(build_elf64())
            res = engine.inspect_path(path)
        self.assertEqual(res["format"], "elf")
        self.assertGreaterEqual(len(res["evidence_ids"]), 5)
        self.assertTrue(res["unknowns"])
        # journal: ONE entry, an Evidence-rooted decision, tagged re
        self.assertTrue(res["journaled"])
        self.assertEqual(len(journal.entries), 1)
        entry = journal.entries[0]
        self.assertIn("RE inspection", entry["decision"])
        self.assertIn(res["evidence_ids"][0], entry["context"])
        self.assertEqual(entry["tags"], ["re", "elf"])
        # memory: remembered under the root evidence address
        self.assertTrue(res["remembered"])
        self.assertEqual(len(memory.items), 1)
        self.assertEqual(memory.items[0]["kind"], "episode")
        self.assertEqual(memory.items[0]["source"],
                         f"re:{res['evidence_ids'][0]}")
        # every inspector pass became its own Evidence record
        kinds = [engine.store.get(e).predicate_type
                 for e in res["evidence_ids"]]
        self.assertEqual(kinds[0], "format.identify")
        self.assertIn("elf.structure", kinds)
        self.assertIn("binary.strings", kinds)

    def test_inspect_missing_file_is_an_honest_error(self):
        engine, journal, _ = self._engine()
        res = engine.inspect_path("/nonexistent/nope.bin")
        self.assertIn("no readable file", res["error"])
        self.assertEqual(journal.entries, [])

    def test_close_through_engine_surfaces_store_rules(self):
        engine, _, _ = self._engine()
        with _TmpHome() as home:
            path = os.path.join(home, "sample.exe")
            with open(path, "wb") as fh:
                fh.write(build_pe())
            res = engine.inspect_path(path)
        signer = [u for u in res["unknowns"]
                  if "signer" in u["question"]]
        self.assertTrue(signer)
        uid = signer[0]["unknown_id"]
        # the signer unknown demands shipped-artifact evidence: an
        # analyst-inference close is refused, with the reason named
        out = engine.close_unknown(uid, status="resolved",
                                   resolution="because I said so")
        self.assertIn("requires authority", out["error"])
        out = engine.close_unknown(
            uid, status="resolved",
            resolution="certificate chain decoded: self-signed test "
                       "certificate",
            resolution_authority="shipped-artifact",
            resolution_confidence="derived")
        self.assertNotIn("error", out)
        self.assertEqual(out["unknown"]["status"], "resolved")

    def test_status_reports_ephemeral_db_honestly(self):
        engine, _, _ = self._engine()
        status = engine.status()
        self.assertTrue(status["store"].startswith("ephemeral"))
        self.assertTrue(any(e["engine"] == "elf" and e["available"]
                            for e in status["engines"]))
        self.assertTrue(any(e["engine"] == "ghidra-headless"
                            and not e["available"]
                            for e in status["engines"]))


class RegistryToolsTest(unittest.TestCase):
    def test_tools_register_and_answer(self):
        from nomorals.agents import reverse
        from nomorals.tools.registry import ToolRegistry

        context = SimpleNamespace(db=Database(":memory:"),
                                  memory=None, settings=None)
        registry = ToolRegistry(context, enforce=False)
        reverse.register(registry)
        self.assertIn("re", registry.names())
        self.assertIn("re_inspect", registry.names())
        outcome = registry.call("re", action="status")
        self.assertTrue(outcome.ok, outcome)
        self.assertIn("engines", outcome.value)
        with _TmpHome() as home:
            path = os.path.join(home, "sample")
            with open(path, "wb") as fh:
                fh.write(build_elf64())
            outcome = registry.call("re_inspect", action="inspect",
                                    path=path)
        self.assertTrue(outcome.ok, outcome)
        self.assertEqual(outcome.value["format"], "elf")
        listed = registry.call("re", action="evidence")
        self.assertTrue(listed.ok, listed)
        self.assertGreaterEqual(len(listed.value["evidence"]), 5)


class ChatHandlerTest(unittest.TestCase):
    def _runtime(self):
        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

        class _RT(RuntimeGamesMixin):
            pass

        rt = _RT()
        rt.context = SimpleNamespace(db=Database(":memory:"),
                                     memory=None, settings=None)
        return rt

    def test_re_verbs(self):
        rt = self._runtime()
        status = rt._control_re("status")
        self.assertIn("reverse engineering", status)
        with _TmpHome() as home:
            path = os.path.join(home, "sample")
            with open(path, "wb") as fh:
                fh.write(build_elf64())
            card = rt._control_re(f"inspect {path}")
        self.assertIn("elf", card)
        self.assertIn("evidence:", card)
        unknowns = rt._control_re("unknowns")
        self.assertIn("debug sections", unknowns)
        audit = rt._control_re("audit")
        self.assertIn("unknowns audit", audit)
        evidence = rt._control_re("evidence")
        self.assertIn("format.identify", evidence)


class CliTest(unittest.TestCase):
    def _context(self):
        return SimpleNamespace(db=Database(":memory:"), memory=None,
                               settings=None)

    def _run(self, args) -> tuple[int, str]:
        from nomorals.cmdline.commands.re import _cmd_re
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_re(args, self._context())
        return rc, buf.getvalue()

    def test_cli_status_inspect_evidence(self):
        rc, out = self._run(SimpleNamespace(re_action="status"))
        self.assertEqual(rc, 0)
        self.assertIn("ghidra-headless", out)
        with _TmpHome() as home:
            path = os.path.join(home, "sample.exe")
            with open(path, "wb") as fh:
                fh.write(build_pe())
            rc, out = self._run(SimpleNamespace(
                re_action="inspect", path=path))
            self.assertEqual(rc, 0)
            self.assertIn("pe", out)
            self.assertIn("kernel32.dll", out)

    def test_parser_parses_re_commands(self):
        from nomorals.cmdline.parser import _parser
        parser = _parser()
        args = parser.parse_args(["re", "inspect", "/tmp/x"])
        self.assertEqual(args.command, "re")
        self.assertEqual(args.re_action, "inspect")
        self.assertEqual(args.path, "/tmp/x")
        args = parser.parse_args(["re", "close", "un:abc", "it",
                                  "is", "fine"])
        self.assertEqual(args.re_action, "close")
        self.assertEqual(args.resolution, ["it", "is", "fine"])

    def test_dispatch_has_re(self):
        from nomorals.cmdline import dispatch
        self.assertIn("re", dispatch._COMMANDS)


class DoctorRowsTest(unittest.TestCase):
    def test_capability_matrix_probes(self):
        from nomorals.agents.reverse import capability_matrix, lightbin
        rows = {row["engine"]: row for row in capability_matrix()}
        for name in ("elf", "pe", "mach-o"):
            self.assertTrue(rows[name]["available"], name)
        # Pass 2: the deep rows carry REAL probe state instead of a
        # blanket "not built yet" — available means ready NOW, and
        # the state names the rest (provisionable / unavailable).
        gh = rows["ghidra-headless"]
        self.assertIn(gh["state"],
                      ("ready", "provisionable", "unavailable"))
        self.assertEqual(gh["available"], gh["state"] == "ready")
        self.assertTrue(gh["note"])
        self.assertIn("temurin-jdk", rows)
        light = rows["rizin-objdump"]
        self.assertEqual(light["available"],
                         lightbin.probe()["available"])
        self.assertTrue(rows["android-graph"]["available"])
        # Pass 3 supersedes the Pass 1 placeholder rows: the
        # "binwalk-unblob" and "tshark" not-built rows are replaced
        # by the real firmware/traffic rows with probe-true states
        # (frida-gdb stays an unbuilt gated row — Pass 4).
        self.assertNotIn("binwalk-unblob", rows)
        for name in ("firmware-census", "traffic-native",
                     "mitm-addon"):
            self.assertTrue(rows[name]["available"], name)
            self.assertTrue(rows[name]["note"], name)
        for name in ("unblob", "binwalk", "tshark", "frida-gdb"):
            self.assertIn(name, rows)
            self.assertTrue(rows[name]["note"], name)
            if name in ("binwalk", "frida-gdb"):
                self.assertFalse(rows[name]["available"], name)
            else:
                # probe-true: available exactly when the engine
                # state says ready on THIS machine
                self.assertEqual(rows[name]["available"],
                                 rows[name].get("state") == "ready",
                                 name)


if __name__ == "__main__":
    unittest.main()
