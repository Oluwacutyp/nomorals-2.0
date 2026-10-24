"""Pass #103 pins: the voice failure ledger records instead of vanishing.

Covers the ledger itself (basics, bound, unprintable exceptions, render,
package exports) and real converted sites across the subsystem: a corrupt
segment-cache file, a failing music detector, a bad melody modifier, a
corrupt stats file, and a cleanup that cannot unlink. Fallback behavior
is asserted alongside every record: recording must never change the
safe default.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.voice import failures as vf
from nomorals.voice import format_failures


class VoiceFailuresTest(unittest.TestCase):
    def setUp(self) -> None:
        vf.clear_failures()

    def tearDown(self) -> None:
        vf.clear_failures()

    def test_ledger_basics_newest_first(self) -> None:
        vf.record_failure("voice.test:a", ValueError("first"))
        vf.record_failure("voice.test:b", OSError("second"))
        self.assertEqual(vf.failure_count(), 2)
        recent = vf.recent_failures()
        self.assertEqual([f.site for f in recent],
                         ["voice.test:b", "voice.test:a"])
        self.assertEqual(recent[0].exc_type, "OSError")
        self.assertEqual(recent[0].message, "second")

    def test_ledger_bounded_ring(self) -> None:
        for i in range(250):
            vf.record_failure("voice.test:loop", RuntimeError(str(i)))
        self.assertEqual(vf.failure_count(), 250)
        self.assertEqual(len(vf.recent_failures(limit=1000)), 200)
        # Oldest dropped, newest kept.
        self.assertEqual(vf.recent_failures(limit=1)[0].message, "249")

    def test_ledger_unprintable_exception(self) -> None:
        class BadStr(Exception):
            def __str__(self) -> str:
                raise RuntimeError("nope")

        vf.record_failure("voice.test:badstr", BadStr())
        entry = vf.recent_failures(limit=1)[0]
        self.assertIn("unprintable", entry.message)

    def test_format_failures_empty_and_rendered(self) -> None:
        self.assertEqual(format_failures(), "no voice failures recorded.")
        vf.record_failure("voice.test:site", ValueError("boom"))
        rendered = format_failures()
        self.assertIn("voice failures (1 total", rendered)
        self.assertIn("voice.test:site: ValueError: boom", rendered)

    def test_package_exports(self) -> None:
        import nomorals.voice as voice_pkg

        for name in ("Failure", "record_failure", "recent_failures",
                     "failure_count", "clear_failures", "format_failures"):
            self.assertIn(name, voice_pkg.__all__)
            self.assertTrue(hasattr(voice_pkg, name))

    def test_segment_cache_corrupt_file_records(self) -> None:
        import nomorals.voice.efficiency as eff

        with tempfile.TemporaryDirectory() as tmp:
            original = eff._cache_dir
            eff._cache_dir = lambda: tmp  # type: ignore[attr-defined]
            try:
                cache = eff.SegmentCache()
                key = eff.SegmentCache.key("hello", "v1", "sys")
                Path(cache._disk_path(key)).write_bytes(b"not a wav file")
                self.assertIsNone(cache.get(key))
            finally:
                eff._cache_dir = original
        sites = [f.site for f in vf.recent_failures()]
        self.assertIn("efficiency.SegmentCache.get:92", sites)

    def test_detect_music_analyze_failure_records(self) -> None:
        from nomorals.voice.music_note import detect_music

        def boom(_path: str):
            raise RuntimeError("fingerprint down")

        out = detect_music("clip.ogg", analyze=boom)
        self.assertEqual(out["is_music"], False)
        sites = [f.site for f in vf.recent_failures()]
        self.assertIn("music_note.detect_music:128", sites)

    def test_parse_melody_bad_modifier_records(self) -> None:
        from nomorals.voice.singing import parse_melody

        notes = parse_melody("E4:1.2:love:vib=abc")
        self.assertIsInstance(notes, list)
        sites = [f.site for f in vf.recent_failures()]
        self.assertIn("singing.parse_melody:105", sites)

    def test_stats_store_corrupt_file_records(self) -> None:
        from nomorals.voice.session import StatsStore

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "stats.json").write_text("{not json")
            store = StatsStore(tmp)
        self.assertEqual(store._data["sessions"], 0)
        sites = [f.site for f in vf.recent_failures()]
        self.assertIn("session.StatsStore.__init__:794", sites)

    def test_bridge_cleanup_directory_records(self) -> None:
        from nomorals.voice.bridge import VoiceBridge

        with tempfile.TemporaryDirectory() as d1, \
                tempfile.TemporaryDirectory() as d2:
            self.assertIsNone(VoiceBridge._cleanup(d1, d2))
        sites = [f.site for f in vf.recent_failures()]
        self.assertEqual(sites.count("bridge.VoiceBridge._cleanup:149"), 2)

    def test_error_scan_subsystem_clean(self) -> None:
        from nomorals.tools.error_scan import scan

        report = scan(["nomorals/voice"], tracked_only=True)
        self.assertEqual(
            len(report.findings), 0,
            [(f.file, f.line, f.rule) for f in report.findings])


if __name__ == "__main__":
    unittest.main()
