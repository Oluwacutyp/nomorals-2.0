"""D1 decompression guard — the reworked shape (owner correction
2026-11-10: a fixed output ceiling is a limitation, not a guard).

The decoder quarantine exists to be the safe place to open
untrusted data.  Its compressed-stream decoders (gzip/zlib, bzip2,
xz) used to decompress with NO output cap at all, then (02df4f91)
under a fixed 256 MiB in-memory ceiling.  The shipped shape now:

* output STREAMS — RAM up to MATERIALIZE_LIMIT, then a staging
  file on disk — so a legitimate expansion of any size decodes
  COMPLETELY (proven at ~600 MB and >1 GB in
  test_d1921d1_decoder_rework.py);
* there is NO fixed byte ceiling by default: a stream stops only
  at the owner's own ceiling (NM_DECODER_MAX_DECOMPRESSED_BYTES,
  when he sets it) or at the staging volume's free-disk reserve
  floor — each stop a refusal on the report naming the limit kind,
  the effective cap, and the size reached, with nothing partial
  surviving anywhere;
* the ceiling, when set, binds the TOTAL across a whole run:
  concatenated bzip2/xz streams that are each under it trip it
  together (a per-entry cap is the classic bypass);
* an incompressible control (ratio ~1:1) decodes fine — the guard
  counts decompressed BYTES, not ratio;
* invalid/truncated streams stay plain no-hits with NO refusal
  (invalid is not capped, and a refusal is never an error
  swallowed as success).

Hermetic: fixtures are generated in-process; the compressed
forms stay kilobytes on disk no matter how large they expand.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import tempfile
import unittest
import zlib
from typing import Any
from unittest import mock

from nomorals.core.decoder import (
    MATERIALIZE_LIMIT,
    MIN_FREE_DISK_BYTES,
    DecompressionCapExceeded,
    analyze,
    min_free_disk_bytes,
    owner_ceiling_bytes,
)

CAP = 1_048_576  # the test-lowered owner ceiling (1 MiB)
CHUNK = 1 << 20  # the pump's output chunk: reached <= cap + one chunk


def _cap_env() -> Any:
    return mock.patch.dict(
        os.environ, {"NM_DECODER_MAX_DECOMPRESSED_BYTES": str(CAP)})


def _no_cap_env() -> Any:
    env = dict(os.environ)
    env.pop("NM_DECODER_MAX_DECOMPRESSED_BYTES", None)
    return mock.patch.dict(os.environ, env, clear=True)


def _gzip_bytes(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as fh:
        fh.write(payload)
    return buf.getvalue()


def _gzip_zeros(total: int) -> bytes:
    """A real zip-bomb shape, streamed so the fixture stays tiny."""
    buf = io.BytesIO()
    block = b"\x00" * 65_536
    with gzip.GzipFile(fileobj=buf, mode="wb") as fh:
        left = total
        while left:
            take = min(left, len(block))
            fh.write(block[:take])
            left -= take
    return buf.getvalue()


def _hits(report: Any, name: str) -> list[Any]:
    return [h for h in report.hits if h.decoder == name]


def _as_bytes(out: Any) -> bytes:
    return out.encode() if isinstance(out, str) else bytes(out)


# ─────────────────── ceiling resolution ────────────────────────

class CeilingResolutionTests(unittest.TestCase):
    def test_default_is_no_fixed_ceiling(self) -> None:
        # the rework's core fact: unset, there is NO builder-picked
        # byte ceiling — output is bounded by the disk reserve.
        with _no_cap_env():
            self.assertIsNone(owner_ceiling_bytes())

    def test_disk_floor_default_is_the_named_constant(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(min_free_disk_bytes(), MIN_FREE_DISK_BYTES)

    def test_default_decodes_the_d1_bomb_fixture_size(self) -> None:
        # the shipped D1 fixture decodes a 5,000,000-byte gzip in
        # full; the default shape has no ceiling below it (or at
        # any other size) — it decodes, full stop.
        payload = b"fixture " * (5_000_000 // 8)
        with _no_cap_env():
            report = analyze(_gzip_bytes(payload))
        found = _hits(report, "gzip")
        self.assertEqual(len(found), 1)
        self.assertEqual(len(_as_bytes(found[0].output)), 5_000_000)
        self.assertEqual(report.refusals, [])

    def test_env_override_is_respected(self) -> None:
        with _cap_env():
            self.assertEqual(owner_ceiling_bytes(), CAP)

    def test_bad_env_values_set_no_ceiling(self) -> None:
        # an unparsable owner setting must not INVENT a ceiling the
        # owner never chose; the disk reserve still binds output.
        for bad in ("abc", "0", "-7", "", "1.5"):
            with mock.patch.dict(
                    os.environ,
                    {"NM_DECODER_MAX_DECOMPRESSED_BYTES": bad}):
                self.assertIsNone(owner_ceiling_bytes(), bad)

    def test_limits_record_states_the_shape(self) -> None:
        from nomorals.tools.decoder import decoder_limits

        with _cap_env():
            limits = decoder_limits()
        self.assertEqual(limits["max_decompressed_bytes"], CAP)
        self.assertIn("decompressed output is capped at 1,048,576 "
                      "bytes", limits["text"])
        self.assertIn("never silently truncated", limits["text"])
        with _no_cap_env():
            limits = decoder_limits()
        self.assertIsNone(limits["max_decompressed_bytes"])
        self.assertIn("no fixed byte ceiling", limits["text"])
        self.assertEqual(limits["min_free_disk_bytes"],
                         MIN_FREE_DISK_BYTES)
        self.assertEqual(limits["max_materialized_input_bytes"],
                         MATERIALIZE_LIMIT)


# ─────────────────────── bomb refusals (owner ceiling) ─────────

class BombRefusalTests(unittest.TestCase):
    def _assert_refusal(self, report: Any, *, fmt: str, decoder: str,
                        full_size: int) -> dict[str, Any]:
        self.assertEqual(_hits(report, decoder), [])
        self.assertEqual(len(report.refusals), 1)
        refusal = report.refusals[0]
        self.assertEqual(refusal["reason"], "decompressed-size-cap")
        self.assertEqual(refusal["limit_kind"], "owner-ceiling")
        self.assertEqual(refusal["format"], fmt)
        self.assertEqual(refusal["decoder"], decoder)
        self.assertEqual(refusal["cap_bytes"], CAP)
        # mid-stream abort: past the cap, at most one pump chunk
        # beyond it, and nowhere near the full decompressed size —
        # the extraction stopped while it was happening, it did
        # not decompress fully and then measure.
        self.assertGreater(refusal["reached_bytes"], CAP)
        self.assertLessEqual(refusal["reached_bytes"], CAP + CHUNK)
        self.assertLess(refusal["reached_bytes"], full_size)
        self.assertIn(fmt, refusal["detail"])
        self.assertIn(str(CAP), refusal["detail"])
        # the refusal rides the report dict (the receipt shape)
        self.assertEqual(report.to_dict()["refusals"], report.refusals)
        # no half-extracted output anywhere on the report
        for hit in report.hits:
            if isinstance(hit.output, (bytes, bytearray)):
                self.assertLessEqual(len(hit.output), CAP)
        if report.best and isinstance(report.best.get("output"),
                                      (bytes, bytearray)):
            self.assertLessEqual(len(report.best["output"]), CAP)
        return refusal

    def test_gzip_bomb_refused_mid_stream(self) -> None:
        bomb = _gzip_zeros(4 * CAP)
        self.assertLess(len(bomb), 100_000)  # ~1000:1 expansion
        with _cap_env():
            report = analyze(bomb)
        self._assert_refusal(report, fmt="gzip", decoder="gzip",
                             full_size=4 * CAP)

    def test_zlib_bomb_refused_mid_stream(self) -> None:
        bomb = zlib.compress(b"\x00" * (4 * CAP))
        with _cap_env():
            report = analyze(bomb)
        self._assert_refusal(report, fmt="zlib", decoder="gzip",
                             full_size=4 * CAP)

    def test_bz2_bomb_refused_mid_stream(self) -> None:
        bomb = bz2.compress(b"\x00" * (4 * CAP))
        with _cap_env():
            report = analyze(bomb)
        self._assert_refusal(report, fmt="bzip2", decoder="bz2",
                             full_size=4 * CAP)

    def test_xz_bomb_refused_mid_stream(self) -> None:
        bomb = lzma.compress(b"\x00" * (4 * CAP))
        with _cap_env():
            report = analyze(bomb)
        self._assert_refusal(report, fmt="xz", decoder="xz",
                             full_size=4 * CAP)

    def test_direct_decoder_call_is_capped_too(self) -> None:
        # the ceiling binds the seam itself, not just analyze(): a
        # standalone try_bytes on a bomb raises the typed refusal
        from nomorals.core.decoder import _GzipDecoder

        with _cap_env(), self.assertRaises(DecompressionCapExceeded) as cm:
            _GzipDecoder().try_bytes(_gzip_zeros(4 * CAP))
        self.assertEqual(cm.exception.cap_bytes, CAP)
        self.assertEqual(cm.exception.kind, "owner-ceiling")
        self.assertGreater(cm.exception.reached_bytes, CAP)

    def test_summary_names_the_refusal(self) -> None:
        from nomorals.tools.decoder import _summarize

        with _cap_env():
            report = analyze(_gzip_zeros(4 * CAP))
        summary = _summarize(report)
        self.assertIn("refused:", summary)
        self.assertIn("gzip stream refused", summary)
        self.assertIn("cap of 1048576 bytes", summary)

    def test_refused_stream_quarantines_nothing(self) -> None:
        # end to end through the agent: a refused stream produces
        # no artifact, no saved_to, and the store stays empty —
        # nothing half-extracted remains in the quarantine.
        from nomorals.agents.context import build_context
        from nomorals.agents.decoder import DecoderAgent, _artifact_store_for
        from nomorals.core.config import load_settings

        tmp = tempfile.TemporaryDirectory(prefix="deccap-")
        try:
            settings = load_settings(
                overrides={"home": tmp.name,
                           "partner.platforms": "local",
                           "chat.local_enabled": "true"})
            ctx = build_context(settings, with_executor=False,
                                with_tools=True, with_router=False)
            try:
                agent = DecoderAgent(context=ctx, name="t")
                with _cap_env():
                    r = agent.run({"data": _gzip_zeros(4 * CAP),
                                   "save": True, "explain": False})
                self.assertTrue(r.ok, r.error)
                self.assertIsNone(r.output["artifact"])
                self.assertIsNone(r.output["saved_to"])
                self.assertEqual(
                    len(r.output["report"]["refusals"]), 1)
                store = _artifact_store_for(ctx)
                self.assertIsNotNone(store)
                self.assertEqual(store.search(creator="decoder"), [])
            finally:
                ctx.close()
        finally:
            tmp.cleanup()


# ─────────────────── shared total across streams ───────────────

class SharedTotalTests(unittest.TestCase):
    """Each member under the ceiling, the TOTAL over it: the limit
    is shared across every stream one run decompresses."""

    def test_bz2_streams_trip_the_shared_total(self) -> None:
        part = bz2.compress(b"B" * 700_000)  # 700k < CAP alone
        with _cap_env():
            solo = analyze(part)
            self.assertEqual(len(_hits(solo, "bz2")), 1)
            self.assertEqual(solo.refusals, [])
            report = analyze(part + part)
        self.assertEqual(_hits(report, "bz2"), [])
        self.assertEqual(len(report.refusals), 1)
        self.assertEqual(report.refusals[0]["reached_bytes"],
                         1_400_000)

    def test_xz_streams_trip_the_shared_total(self) -> None:
        part = lzma.compress(b"C" * 700_000)
        with _cap_env():
            solo = analyze(part)
            self.assertEqual(len(_hits(solo, "xz")), 1)
            report = analyze(part + part)
        self.assertEqual(_hits(report, "xz"), [])
        self.assertEqual(len(report.refusals), 1)
        self.assertEqual(report.refusals[0]["reached_bytes"],
                         1_400_000)

    def test_three_small_bz2_streams_trip_together(self) -> None:
        part = bz2.compress(b"D" * 400_000)
        with _cap_env():
            report = analyze(part + part + part)
        self.assertEqual(_hits(report, "bz2"), [])
        self.assertEqual(len(report.refusals), 1)
        self.assertGreater(report.refusals[0]["reached_bytes"], CAP)


# ─────────────────── legitimate extraction ─────────────────────

class LegitExtractionTests(unittest.TestCase):
    PAYLOAD = b"hello code beast, this is a real document. " * 400

    def test_each_format_byte_identical_to_uncapped(self) -> None:
        payload = self.PAYLOAD
        cases = [
            (_gzip_bytes(payload), "gzip",
             zlib.decompress(_gzip_bytes(payload), 31)),
            (zlib.compress(payload), "gzip",
             zlib.decompress(zlib.compress(payload))),
            (bz2.compress(payload), "bz2", bz2.decompress(
                bz2.compress(payload))),
            (lzma.compress(payload), "xz", lzma.decompress(
                lzma.compress(payload))),
        ]
        with _cap_env():
            for blob, decoder, expected in cases:
                report = analyze(blob)
                found = _hits(report, decoder)
                self.assertEqual(len(found), 1, decoder)
                self.assertEqual(_as_bytes(found[0].output), expected)
                self.assertEqual(_as_bytes(found[0].output), payload)
                self.assertEqual(report.refusals, [])

    def test_stream_just_under_the_ceiling_extracts_completely(self) -> None:
        payload = b"word " * ((CAP - 5_000) // 5)
        with _cap_env():
            report = analyze(_gzip_bytes(payload))
        found = _hits(report, "gzip")
        self.assertEqual(len(found), 1)
        self.assertEqual(_as_bytes(found[0].output), payload)
        self.assertEqual(report.refusals, [])

    def test_stream_exactly_at_the_ceiling_extracts_completely(self) -> None:
        payload = b"abcd" * (CAP // 4)  # exactly CAP bytes out
        with _cap_env():
            report = analyze(_gzip_bytes(payload))
        found = _hits(report, "gzip")
        self.assertEqual(len(found), 1)
        self.assertEqual(len(_as_bytes(found[0].output)), CAP)
        self.assertEqual(report.refusals, [])

    def test_incompressible_control_decodes(self) -> None:
        # ratio ~1:1 — the guard counts decompressed bytes, not the
        # compression ratio, so honest dense data passes untouched
        payload = os.urandom(CAP // 2)
        with _cap_env():
            report = analyze(_gzip_bytes(payload))
        found = _hits(report, "gzip")
        self.assertEqual(len(found), 1)
        self.assertEqual(_as_bytes(found[0].output), payload)
        self.assertEqual(report.refusals, [])

    def test_multi_stream_bz2_under_ceiling_sums_like_uncapped(self) -> None:
        blob = bz2.compress(b"one ") + bz2.compress(b"two")
        with _cap_env():
            report = analyze(blob)
        found = _hits(report, "bz2")
        self.assertEqual(len(found), 1)
        self.assertEqual(_as_bytes(found[0].output),
                         bz2.decompress(blob))

    def test_gzip_multi_member_keeps_shipped_semantics(self) -> None:
        # the shipped gzip path is a zlib call: it reads the FIRST
        # member and ignores the rest.  The rework does not change
        # success-path semantics — output stays byte-identical to
        # that uncapped call.
        blob = _gzip_bytes(b"G" * 600_000) + _gzip_bytes(b"H" * 600_000)
        with _cap_env():
            report = analyze(blob)
        found = _hits(report, "gzip")
        self.assertEqual(len(found), 1)
        self.assertEqual(_as_bytes(found[0].output),
                         zlib.decompress(blob, 31))
        self.assertEqual(report.refusals, [])

    def test_nested_chain_lands_on_the_guarded_seam(self) -> None:
        # the engine's real nesting (the D1 fixture pattern): a
        # gzip wrapping a gzip bomb.  The outer stage decodes to
        # the inner stream's bytes; the stage that would actually
        # expand the bomb is the one that refuses.
        inner = _gzip_zeros(4 * CAP)
        outer = _gzip_bytes(inner)
        with _cap_env():
            stage = analyze(outer)
            found = _hits(stage, "gzip")
            self.assertEqual(len(found), 1)
            self.assertEqual(_as_bytes(found[0].output), inner)
            self.assertEqual(stage.refusals, [])
            deep = analyze(_as_bytes(found[0].output))
        self.assertEqual(_hits(deep, "gzip"), [])
        self.assertEqual(len(deep.refusals), 1)


# ─────────────────── invalid is not capped ─────────────────────

class InvalidStreamTests(unittest.TestCase):
    def test_truncated_gzip_is_a_no_hit_not_a_refusal(self) -> None:
        blob = _gzip_bytes(b"hello world" * 100)[:20]
        with _cap_env():
            report = analyze(blob)
        self.assertEqual(_hits(report, "gzip"), [])
        self.assertEqual(report.refusals, [])

    def test_corrupt_streams_are_no_hits_not_refusals(self) -> None:
        cases = [
            (b"\x1f\x8b" + b"\xff" * 40, "gzip"),
            (b"BZh9" + b"\xff" * 40, "bz2"),
            (b"\xfd7zXZ\x00" + b"\xff" * 40, "xz"),
        ]
        with _cap_env():
            for blob, decoder in cases:
                report = analyze(blob)
                self.assertEqual(_hits(report, decoder), [], decoder)
                self.assertEqual(report.refusals, [], decoder)

    def test_empty_gzip_still_decodes_to_an_empty_hit(self) -> None:
        with _cap_env():
            report = analyze(_gzip_bytes(b""))
        found = _hits(report, "gzip")
        self.assertEqual(len(found), 1)
        self.assertEqual(_as_bytes(found[0].output), b"")
        self.assertEqual(report.refusals, [])


if __name__ == "__main__":
    unittest.main()
