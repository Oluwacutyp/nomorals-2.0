"""SongRecon surface reachability (doc 24 §16 residual).

The recognition engine (fingerprint/bench/melodic/lineage + AudD)
shipped under Doc 24, and /sham already identified songs in chat —
but the brain could not call it, the CLI had no full-pipeline verb,
and the system had no /songrecon name anywhere. This file pins the
surface this lane wired: the /songrecon chat alias, the songrecon
brain tool, nm audio identify (+ the nm songrecon alias), and the
shared audio.songrecon pipeline the /sham rewire now runs through.

Vocabulary rules (SR1): a local hit reports its coherent-offset
fraction — an uncalibrated alignment measure, never a probability —
and the word "confidence" never appears in recognition copy.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.audio.edit import control_audio
from nomorals.audio.fingerprint import FingerprintDB
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    parse_control,
)
from nomorals.storage.db import Database


def _tone_wav(path: str, freqs: list[float], secs: int = 6) -> str:
    sr = 11025
    n = int(sr * secs)
    vals = array("d", [0.0]) * n
    for i in range(n):
        t = i / sr
        vals[i] = sum(
            math.sin(2 * math.pi * f * t) * 0.25 for f in freqs) * 0.9
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in vals))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class SongReconSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.d = tempfile.mkdtemp()
        self.song_a = _tone_wav(
            os.path.join(self.d, "a.wav"), [220.0, 277.18, 329.63])
        self.song_b = _tone_wav(
            os.path.join(self.d, "b.wav"), [196.0, 246.94, 293.66])
        self.db = FingerprintDB(os.path.join(self.d, "fp.db"))
        enrolled = self.db.add_track(
            self.song_a, title="Song A", artist="Devon")
        self.assertTrue(enrolled["ok"], enrolled)

    def tearDown(self) -> None:
        self.db.close()

    def _mixin(self) -> Any:
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        class T(RuntimeMediaMixin):
            def __init__(self) -> None:
                self.context = mock.Mock()
                self.context.db = Database(":memory:")
                self.gateway = None

        return T()

    @staticmethod
    def _msg(path: str) -> Any:
        media = mock.Mock()
        media.kind = "audio"
        media.path = path
        msg = mock.Mock()
        msg.media = [media]
        msg.meta = {}
        return msg

    def _registry(self):
        from nomorals.tools.registry import ToolRegistry

        ctx = SimpleNamespace(
            db=None,
            settings=SimpleNamespace(workspace_dir=self.d,
                                     home_path=Path(self.d)))
        reg = ToolRegistry(ctx)
        reg.register_builtins()
        return reg

    # ── chat alias ────────────────────────────────────────────────

    def test_songrecon_parses_as_sham_alias(self) -> None:
        cmd = parse_control("/songrecon")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "songrecon")
        self.assertEqual(CONTROL_COMMANDS["songrecon"], (0, 0))
        # /sham itself is unchanged.
        sham = parse_control("/sham")
        self.assertIsNotNone(sham)
        assert sham is not None
        self.assertEqual(sham.kind, "sham")
        # Same access as /sham: not a public command, so the 279 pin
        # does not move.
        self.assertNotIn("songrecon", PUBLIC_COMMANDS)
        self.assertEqual(len(PUBLIC_COMMANDS), 279)

    def test_songrecon_details_stay_honest(self) -> None:
        details = COMMAND_DETAILS["songrecon"]
        text = " ".join(str(v) for v in details.values())
        self.assertIn("coherent-offset fraction", text)
        self.assertIn("not a probability", text)
        self.assertIn("AudD is the labelled external fallback", text)
        self.assertIn("melodic hypotheses", text)
        self.assertNotIn("confidence", text.lower())
        self.assertIn("/songrecon", str(details.get("usage") or ""))

    def test_sham_still_identifies_after_rewire(self) -> None:
        mixin = self._mixin()
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db), mock.patch(
                "nomorals.connectors.audd.open_audd",
                return_value=(None, "not_configured")):
            out = mixin._sham_identify(self._msg(self.song_a))
        self.assertIn("I know this one — Devon — Song A", out)
        self.assertIn("coherent-offset fraction", out)
        self.assertIn("/play Devon Song A", out)
        self.assertNotIn("confidence", out.lower())

    def test_sham_miss_still_honest_after_rewire(self) -> None:
        mixin = self._mixin()
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db), mock.patch(
                "nomorals.connectors.audd.open_audd",
                return_value=(None, "not_configured")):
            out = mixin._sham_identify(self._msg(self.song_b))
        self.assertIn("couldn't name that track", out)
        self.assertIn("AudD external catalogue was not searched", out)
        self.assertNotIn("confidence", out.lower())

    # ── brain tool ────────────────────────────────────────────────

    def test_registry_counts_787_with_songrecon(self) -> None:
        # 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist.
        reg = self._registry()
        self.assertEqual(len(reg._tools), 805)
        self.assertIsNotNone(reg.get("songrecon"))
        self.assertEqual(reg.failed_modules(), [])

    def test_tool_identify_hit_and_miss(self) -> None:
        reg = self._registry()
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db), mock.patch(
                "nomorals.connectors.audd.open_audd",
                return_value=(None, "not_configured")):
            hit = reg.call("songrecon", action="identify",
                           audio=self.song_a)
            self.assertTrue(hit.ok, hit)
            self.assertTrue(hit.value["ok"], hit.value)
            self.assertEqual(hit.value["stage"], "local")
            self.assertIn("Song A", hit.value["text"])
            self.assertNotIn("confidence", hit.value["text"].lower())
            miss = reg.call("songrecon", action="identify",
                            audio=self.song_b)
            self.assertTrue(miss.ok, miss)
            self.assertFalse(miss.value["ok"])
            self.assertEqual(miss.value["stage"], "unmatched")
            self.assertIn("AudD external catalogue was not searched",
                          miss.value["text"])

    def test_tool_enroll_then_match_then_catalog(self) -> None:
        reg = self._registry()
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db):
            enrolled = reg.call("songrecon", action="enroll",
                                audio=self.song_b, title="Song B",
                                artist="Devon")
            self.assertTrue(enrolled.ok, enrolled)
            self.assertTrue(enrolled.value["ok"], enrolled.value)
            self.assertIn("fingerprinted 'Song B'",
                          enrolled.value["text"])
            matched = reg.call("songrecon", action="match",
                               audio=self.song_b)
            self.assertTrue(matched.ok, matched)
            self.assertTrue(matched.value["ok"], matched.value)
            self.assertIn("Song B", matched.value["text"])
            catalog = reg.call("songrecon", action="catalog")
            self.assertTrue(catalog.ok, catalog)
            self.assertEqual(catalog.value["tracks"], 2)

    def test_tool_rejects_unknown_action_and_missing_audio(self) -> None:
        reg = self._registry()
        bad = reg.call("songrecon", action="bogus", audio=self.song_a)
        self.assertTrue(bad.ok, bad)
        self.assertFalse(bad.value["ok"])
        missing = reg.call("songrecon", action="identify", audio="")
        self.assertTrue(missing.ok, missing)
        self.assertFalse(missing.value["ok"])

    # ── CLI ───────────────────────────────────────────────────────

    def test_cli_identify_hit_and_miss(self) -> None:
        from nomorals.cmdline.commands.audio import _cmd_audio

        ctx = SimpleNamespace(db=None)
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db), mock.patch(
                "nomorals.connectors.audd.open_audd",
                return_value=(None, "not_configured")):
            hit = _cmd_audio(argparse.Namespace(
                audio_action="identify", file=self.song_a,
                json=False, no_external=True), ctx)
            self.assertEqual(hit, 0)
            miss = _cmd_audio(argparse.Namespace(
                audio_action="identify", file=self.song_b,
                json=False, no_external=True), ctx)
            self.assertEqual(miss, 1)

    def test_cli_songrecon_alias_resolves_to_audio(self) -> None:
        from nomorals.cmdline.dispatch import _canonical_command
        from nomorals.cmdline.parser import CLI_ALIASES

        self.assertIn("songrecon", CLI_ALIASES["audio"])
        self.assertEqual(_canonical_command("songrecon"), "audio")
        self.assertEqual(_canonical_command("au"), "audio")

    # ── shared enrollment path ────────────────────────────────────

    def test_control_audio_fingerprint_still_enrolls(self) -> None:
        other_db = FingerprintDB(os.path.join(self.d, "other.db"))
        try:
            with mock.patch(
                    "nomorals.audio.fingerprint.default_fingerprint_db",
                    return_value=other_db):
                enrolled = control_audio(
                    f"fingerprint {self.song_a} Enrolled Song")
                matched = control_audio(f"match {self.song_a}")
        finally:
            other_db.close()
        self.assertIn("fingerprinted 'Enrolled Song'", enrolled)
        self.assertIn("coherent-offset fraction", matched)
        self.assertNotIn("confidence", matched.lower())


if __name__ == "__main__":
    unittest.main()
