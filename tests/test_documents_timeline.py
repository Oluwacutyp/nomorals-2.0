"""Timeline wiring for the documents organ (Wave K).

Attaches a :class:`~nomorals.os.timeline.Timeline` to the process bus,
runs ``parse_bytes`` / ``parse_path``, and asserts ``document.parsed``
events land with format, title, and section counts.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.core.events import global_bus
from nomorals.documents import DocumentError, parse_bytes, parse_path
from nomorals.os.timeline import Timeline


class DocumentsTimelineTestCase(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline()  # in-memory
        self.addCleanup(self.timeline.close)
        self.timeline.attach(global_bus, sync=True)
        self.addCleanup(self.timeline.detach, global_bus)

    def _parsed(self):
        return self.timeline.query(topic="document.parsed", limit=50)

    def test_parse_bytes_txt_emits_event(self):
        doc = parse_bytes(b"hello world", filename="note.txt")
        rows = self._parsed()
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["doc_id"], doc.id)
        self.assertEqual(data["format"], "txt")
        self.assertEqual(data["title"], "note")
        self.assertEqual(data["sections"], 1)
        self.assertEqual(data["tables"], 0)
        self.assertEqual(data["source"], "note.txt")
        self.assertEqual(rows[0]["source"], "nomorals.documents.parsers")

    def test_parse_bytes_markdown_section_count(self):
        md = b"# Title\n\nbody text\n\n## Second\n\nmore body\n"
        parse_bytes(md, filename="guide.md")
        rows = self._parsed()
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["format"], "markdown")
        self.assertEqual(data["title"], "Title")  # first heading is promoted
        self.assertEqual(data["sections"], 2)

    def test_parse_bytes_csv_reports_table(self):
        parse_bytes(b"a,b\n1,2\n", filename="data.csv")
        rows = self._parsed()
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["format"], "csv")
        self.assertEqual(data["tables"], 1)

    def test_parse_path_emits_event(self):
        with tempfile.TemporaryDirectory(prefix="nm-doc-timeline-") as tmp:
            path = Path(tmp) / "report.txt"
            path.write_text("some content", encoding="utf-8")
            parse_path(path)
        rows = self._parsed()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["data"]["format"], "txt")
        self.assertEqual(rows[0]["data"]["title"], "report")

    def test_failed_parse_emits_nothing(self):
        with self.assertRaises(DocumentError):
            parse_bytes(b"", filename="empty.txt")
        self.assertEqual(self._parsed(), [])

    def test_no_timeline_works_fine(self):
        # Fail-open telemetry: parsing must work with no subscriber.
        self.timeline.detach(global_bus)
        doc = parse_bytes(b"plain", filename="p.txt")
        self.assertEqual(doc.format, "txt")


if __name__ == "__main__":
    unittest.main()
