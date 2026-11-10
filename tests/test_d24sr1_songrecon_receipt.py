"""Doc 24 SR1 — SongRecon receipts and honest /sham rendering.

The local score is a coherent-offset fraction: the share of query
landmark hashes that agree on one reference-offset bin. These tests
pin that name and its evidence, and pin the absence of the word
"confidence" from the /sham surface and catalog description.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.audio.edit import control_audio
from nomorals.audio.fingerprint import (
    FingerprintDB,
    external_recognition_receipt,
    identify,
)
from nomorals.social.chat.control import COMMAND_DETAILS
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


class SongReconReceiptTests(unittest.TestCase):
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

    def test_local_identify_emits_typed_receipt(self) -> None:
        result = identify(self.song_a, db=self.db)
        self.assertTrue(result["ok"], result)
        receipt = result["receipt"]
        self.assertEqual(receipt["schema"], "songrecon.receipt/v1")
        self.assertEqual(receipt["status"], "matched")
        self.assertEqual(receipt["source"], "devon_local_fingerprint")
        self.assertEqual(
            receipt["match_class"],
            "class_1_exact_or_near_exact_recording")
        self.assertEqual(receipt["catalog_track_count"], 1)
        self.assertGreater(receipt["catalog_hash_count"], 0)
        self.assertTrue(
            receipt["catalog_version"].startswith(
                "local-fingerprint-v1:sha256:"))
        self.assertEqual(len(receipt["query_hash"]), 64)
        self.assertGreater(receipt["query_fingerprint_count"], 0)

        strength = receipt["match_strength"]
        self.assertEqual(strength["name"], "coherent_offset_fraction")
        self.assertEqual(strength["value"], result["score"])
        self.assertGreater(strength["numerator"], 0)
        self.assertEqual(
            strength["denominator"],
            receipt["query_fingerprint_count"])
        self.assertIn("50 ms", strength["basis"])
        self.assertIn("not a probability", strength["calibration"])
        self.assertTrue(receipt["evidence_windows"])
        window = receipt["evidence_windows"][0]
        self.assertIn("query_start_s", window)
        self.assertIn("reference_start_s", window)
        self.assertIn("reference_minus_query_ms",
                      receipt["offset_map"])
        self.assertNotIn("confidence", json.dumps(receipt).lower())

    def test_local_no_match_receipt_names_search_and_reason(self) -> None:
        result = identify(self.song_b, db=self.db)
        self.assertFalse(result["ok"])
        receipt = result["receipt"]
        self.assertEqual(receipt["status"], "no_match")
        self.assertEqual(receipt["match_class"], "no_reliable_match")
        self.assertEqual(receipt["catalog_track_count"], 1)
        self.assertIn("searched 1 indexed track(s)",
                      receipt["no_match_reason"])
        self.assertIn("coherent-offset fraction",
                      receipt["no_match_reason"])

    def test_sham_local_match_renders_measured_quantity(self) -> None:
        mixin = self._mixin()
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db):
            out = mixin._sham_identify(self._msg(self.song_a))
        self.assertIn("Devon — Song A", out)
        self.assertIn("coherent-offset fraction", out)
        self.assertIn("query landmark hashes aligned at one "
                      "reference offset", out)
        self.assertIn("uncalibrated; not a probability", out)
        self.assertIn("evidence window:", out)
        self.assertNotIn("confidence", out.lower())

    def test_audio_surface_enrollment_then_match(self) -> None:
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
        self.assertIn("not a probability", matched)
        self.assertNotIn("confidence", matched.lower())

    def test_sham_no_match_names_catalogs_searched(self) -> None:
        mixin = self._mixin()
        connector = mock.Mock()
        connector.recognize = mock.Mock(return_value={
            "ok": False,
            "reason": "couldn't identify that audio — no match in "
                      "AudD's catalog. Try a clearer/longer clip.",
        })
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db), mock.patch(
                "nomorals.accounts.vault.open_vault",
                return_value=object()), mock.patch(
                "nomorals.connectors.registry.create_connector",
                return_value=connector):
            out = mixin._sham_identify(self._msg(self.song_b))
        self.assertIn("no reliable match found", out)
        self.assertIn("Devon's local fingerprint catalog "
                      "(1 indexed track(s))", out)
        self.assertIn("AudD external catalogue was searched", out)
        self.assertNotIn("confidence", out.lower())

    def test_unmatched_reply_does_not_claim_unsearched_local(self) -> None:
        mixin = self._mixin()
        local = {
            "ok": False,
            "reason": "audio could not be decoded",
            "receipt": {
                "status": "unavailable",
                "reason": "audio could not be decoded",
                "catalog_track_count": 1,
            },
        }
        out = mixin._sham_unmatched_reply(
            local,
            lead="couldn't name that track",
            reason="AudD is not configured",
            external_status="AudD external catalogue was not searched",
            heard="a clipped tone")
        self.assertIn("local fingerprint catalog was not searched", out)
        self.assertIn("audio could not be decoded", out)
        self.assertNotIn("Searched Devon's local fingerprint catalog",
                         out)

    def test_sham_audd_match_is_labelled_as_fallback(self) -> None:
        mixin = self._mixin()
        connector = mock.Mock()
        connector.recognize = mock.Mock(return_value={
            "ok": True,
            "artist": "Ayo Maff",
            "title": "Lifestyle",
            "album": "EP",
            "release_date": "2024-01-01",
            "label": "Test Label",
            "timecode": "00:12",
            "song_link": "https://lis.tn/test",
            "spotify": None,
            "apple_music": None,
        })
        with mock.patch(
                "nomorals.audio.fingerprint.default_fingerprint_db",
                return_value=self.db), mock.patch(
                "nomorals.accounts.vault.open_vault",
                return_value=object()), mock.patch(
                "nomorals.connectors.registry.create_connector",
                return_value=connector):
            out = mixin._sham_identify(self._msg(self.song_b))
        self.assertIn("Ayo Maff — Lifestyle", out)
        self.assertIn("Identified by AudD — external catalogue "
                      "fallback", out)
        self.assertIn("not Devon's local fingerprint catalog", out)
        self.assertIn("AudD timecode: 00:12", out)
        self.assertNotIn("confidence", out.lower())

    def test_audd_receipt_preserves_query_and_local_search(self) -> None:
        local = identify(self.song_b, db=self.db)
        receipt = external_recognition_receipt(
            {
                "artist": "Ayo Maff",
                "title": "Lifestyle",
                "album": "EP",
                "release_date": "2024-01-01",
                "label": "Test Label",
                "timecode": "00:12",
                "song_link": "https://lis.tn/test",
            },
            local_receipt=local["receipt"])
        self.assertEqual(receipt["source"], "audd")
        self.assertEqual(receipt["status"], "matched")
        self.assertEqual(receipt["query_hash"],
                         local["receipt"]["query_hash"])
        self.assertEqual(receipt["local_search"]["status"], "no_match")
        self.assertEqual(receipt["match_strength"], {})
        self.assertEqual(
            receipt["catalog_version"],
            "audd:provider-managed (version not exposed)")
        self.assertNotIn("confidence", json.dumps(receipt).lower())

    def test_sham_catalog_copy_uses_alignment_vocabulary(self) -> None:
        sham = COMMAND_DETAILS["sham"]
        text = " ".join(str(value) for value in sham.values())
        self.assertIn("coherent-offset fraction", text)
        self.assertIn("not a probability", text)
        self.assertIn("AudD is the labelled external fallback", text)
        self.assertNotIn("confidence", text.lower())


if __name__ == "__main__":
    unittest.main()
