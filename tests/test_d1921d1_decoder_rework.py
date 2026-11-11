"""D1 decoder rework (2026-11-10) — the shape the owner ordered
when he called the fixed 256 MiB ceiling a limitation:

* decompression STREAMS to disk staging past MATERIALIZE_LIMIT,
  so a legit expansion of any size decodes completely (proven
  here at ~600 MB and >1 GB) with RAM bounded by construction;
* the budget is the disk: an injected free-space reading below
  the reserve floor refuses with disk numbers (the probe seam is
  the run's constructor, not a monkeypatched df);
* staged files are working paper — gone after success, refusal,
  exception, and orphaned files from dead runs are swept;
* file inputs stream (analyze_path): a file larger than the
  old 2,000,000-byte door — and larger than RAM's materialize
  limit — is analyzed whole, never truncated;
* the quarantine sweep reclaims decoded-untrusted residents past
  their budget, oldest first, journaled on the record, and never
  touches released/kept or general artifacts.

Every staging/store path here is a per-test tmp path.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import os
import shutil
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import nomorals.core.decoder as dec
from nomorals.core.decoder import (
    MATERIALIZE_LIMIT,
    MIN_FREE_DISK_BYTES,
    StagedOutput,
    analyze,
    analyze_path,
    decompression_run,
)

FOX = b"The quick brown fox jumps over the lazy dog 0123456789. "


def _stream_gzip(path: Path, total: int, block: bytes) -> str:
    """Write a gzip file expanding to exactly ``total`` bytes of
    ``block`` repeated; return the payload's sha256 — both built
    by streaming, so the fixture never sits in RAM either."""
    digest = hashlib.sha256()
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as fh:
        left = total
        while left:
            take = block[:min(left, len(block))]
            fh.write(take)
            digest.update(take)
            left -= len(take)
    return digest.hexdigest()


class _StagingCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(dir=os.environ.get(
            "NM_TEST_TMPDIR") or None)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.stage = os.path.join(self.tmp, "staging")
        os.makedirs(self.stage)

    def staged_files(self) -> list[str]:
        return sorted(os.listdir(self.stage))


# ─────────────────── disk-streamed expansion ───────────────────

class DiskStreamingTests(_StagingCase):
    def _decode_big(self, total: int, block: bytes) -> None:
        src = Path(self.tmp) / "big.gz"
        expected_sha = _stream_gzip(src, total, block)
        blob = src.read_bytes()
        tracemalloc.start()
        try:
            with decompression_run(stage_dir=self.stage):
                report = analyze(blob)
                _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        found = [h for h in report.hits if h.decoder == "gzip"]
        self.assertEqual(len(found), 1)
        out = found[0].output
        self.assertIsInstance(out, StagedOutput)
        self.assertEqual(out.size, total)
        self.assertEqual(out.sha256, expected_sha)
        self.assertEqual(report.refusals, [])
        # RAM is bounded by construction: the whole 600 MB+ output
        # never materialized — traced Python allocations stay far
        # under the output size (the spool is one buffered chunk).
        self.assertLess(peak, 128 * 1024 * 1024)
        self.assertEqual(self.staged_files(), [])

    def test_600mb_legit_expansion_decodes_completely(self) -> None:
        self._decode_big(600 * 1024 * 1024, FOX)

    def test_over_1gb_expansion_decodes_completely(self) -> None:
        self._decode_big(1_100_000_000, b"\x00" * 65_536)


# ─────────────────── the disk reserve ──────────────────────────

class DiskReserveTests(_StagingCase):
    def test_disk_floor_refusal_names_the_numbers(self) -> None:
        free = MIN_FREE_DISK_BYTES + 300_000
        blob = gzip.compress(b"D" * (4 * 1024 * 1024))
        with mock.patch.object(dec, "MATERIALIZE_LIMIT", 1 << 20), decompression_run(
                stage_dir=self.stage,
                free_space_probe=lambda _p: free) as run:
            report = analyze(blob)
            self.assertEqual(
                [h for h in report.hits if h.decoder == "gzip"], [])
            self.assertEqual(len(report.refusals), 1)
            refusal = report.refusals[0]
            self.assertEqual(refusal["reason"],
                             "decompressed-size-cap")
            self.assertEqual(refusal["limit_kind"], "disk-reserve")
            self.assertEqual(refusal["free_disk_bytes"], free)
            self.assertEqual(refusal["reserve_floor_bytes"],
                             MIN_FREE_DISK_BYTES)
            self.assertEqual(refusal["cap_bytes"],
                             (1 << 20) + 300_000)
            self.assertEqual(refusal["reached_bytes"], 2 << 20)
            self.assertIn("disk reserve", refusal["detail"])
            self.assertIsNotNone(run)
        self.assertEqual(self.staged_files(), [])

    def test_roomy_probe_lets_the_spool_run(self) -> None:
        blob = gzip.compress(b"R" * (4 * 1024 * 1024))
        with mock.patch.object(dec, "MATERIALIZE_LIMIT", 1 << 20), decompression_run(
                stage_dir=self.stage,
                free_space_probe=lambda _p: 1 << 40):
            report = analyze(blob)
        found = [h for h in report.hits if h.decoder == "gzip"]
        self.assertEqual(len(found), 1)
        self.assertIsInstance(found[0].output, StagedOutput)
        self.assertEqual(found[0].output.size, 4 * 1024 * 1024)
        self.assertEqual(report.refusals, [])
        self.assertEqual(self.staged_files(), [])


# ─────────────────── staging lifecycle ─────────────────────────

class StagingLifecycleTests(_StagingCase):
    def _env(self) -> Any:
        return mock.patch.dict(
            os.environ, {"NM_DECODER_STAGING_DIR": self.stage})

    def test_refusal_leaves_no_staged_files(self) -> None:
        blob = gzip.compress(b"Z" * (8 * 1024 * 1024))
        env = {"NM_DECODER_MAX_DECOMPRESSED_BYTES": "1048576"}
        with self._env(), mock.patch.dict(os.environ, env), \
                mock.patch.object(dec, "MATERIALIZE_LIMIT", 1 << 20):
            report = analyze(blob)
        self.assertEqual(len(report.refusals), 1)
        self.assertEqual(self.staged_files(), [])

    def test_exception_inside_a_run_cleans_staging(self) -> None:
        blob = gzip.compress(b"E" * (4 * 1024 * 1024))
        with mock.patch.object(dec, "MATERIALIZE_LIMIT", 1 << 20), \
                self.assertRaises(RuntimeError), \
                decompression_run(stage_dir=self.stage):
            report = analyze(blob)
            found = [h for h in report.hits
                     if h.decoder == "gzip"]
            self.assertIsInstance(found[0].output, StagedOutput)
            raise RuntimeError("boom mid-run")
        self.assertEqual(self.staged_files(), [])

    def test_corrupt_tail_after_spool_cleans_and_no_hits(self) -> None:
        blob = bz2.compress(b"C" * (2 * 1024 * 1024)) + b"garbage!!"
        with self._env(), mock.patch.object(
                dec, "MATERIALIZE_LIMIT", 1 << 20):
            report = analyze(blob)
        self.assertEqual(
            [h for h in report.hits if h.decoder == "bz2"], [])
        self.assertEqual(self.staged_files(), [])

    def test_orphan_sweep(self) -> None:
        pid = os.getpid()
        now = time.time()

        def plant(name: str, age: float) -> Path:
            p = Path(self.stage) / name
            p.write_bytes(b"x")
            os.utime(p, (now - age, now - age))
            return p

        dead_old = plant(f"decstage-{pid}-deadold-1.tmp", 120)
        dead_fresh = plant(f"decstage-{pid}-deadfresh-1.tmp", 5)
        foreign_old = plant("decstage-999999-ancient-1.tmp", 7_200)
        foreign_fresh = plant("decstage-999999-recent-1.tmp", 30)
        unrelated = plant("keepme.txt", 10_000)
        with decompression_run(stage_dir=self.stage):
            pass
        self.assertFalse(dead_old.exists())
        self.assertTrue(dead_fresh.exists())
        self.assertFalse(foreign_old.exists())
        self.assertTrue(foreign_fresh.exists())
        self.assertTrue(unrelated.exists())


# ─────────────────── the input door ────────────────────────────

class InputDoorTests(_StagingCase):
    def test_small_file_report_matches_analyze(self) -> None:
        blob = gzip.compress(b"small file body " * 10)
        path = Path(self.tmp) / "small.gz"
        path.write_bytes(blob)
        via_path = analyze_path(path)
        direct = analyze(blob)
        self.assertEqual(via_path.forensics["bytes"], len(blob))
        self.assertNotIn("input_mode", via_path.forensics)
        a = [h for h in via_path.hits if h.decoder == "gzip"]
        b = [h for h in direct.hits if h.decoder == "gzip"]
        self.assertEqual(len(a), 1)

        def as_bytes(out: Any) -> bytes:
            return out.encode() if isinstance(out, str) else bytes(out)

        self.assertEqual(as_bytes(a[0].output), as_bytes(b[0].output))

    def test_file_over_materialize_limit_streams_whole(self) -> None:
        payload = os.urandom(40 * 1024 * 1024)
        path = Path(self.tmp) / "stored.gz"
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb",
                           compresslevel=0) as fh:
            fh.write(payload)
        self.assertGreater(path.stat().st_size, MATERIALIZE_LIMIT)
        with decompression_run(stage_dir=self.stage):
            report = analyze_path(path)
        self.assertEqual(report.forensics["input_mode"], "streamed")
        self.assertEqual(report.forensics["file_bytes"],
                         path.stat().st_size)
        self.assertEqual(report.forensics["sample_bytes"], 2_000_000)
        found = [h for h in report.hits if h.decoder == "gzip"]
        self.assertEqual(len(found), 1)
        out = found[0].output
        self.assertIsInstance(out, StagedOutput)
        self.assertEqual(out.size, len(payload))
        self.assertEqual(out.sha256,
                         hashlib.sha256(payload).hexdigest())
        self.assertEqual(self.staged_files(), [])

    def test_big_plain_file_forensics_are_honest(self) -> None:
        path = Path(self.tmp) / "plain.txt"
        block = b"plain text line\n" * 4096
        total = 0
        with open(path, "wb") as fh:
            while total <= MATERIALIZE_LIMIT:
                fh.write(block)
                total += len(block)
        report = analyze_path(path)
        self.assertEqual(report.forensics["input_mode"], "streamed")
        self.assertEqual(report.forensics["file_bytes"], total)
        # "bytes" names the subject (the whole file); sample_bytes
        # names what was actually materialized for inspection
        self.assertEqual(report.forensics["bytes"], total)
        self.assertEqual(report.forensics["sample_bytes"], 2_000_000)
        self.assertIn("string_analysis", report.forensics)


# ─────────────────── the quarantine sweep ──────────────────────

class SweepTests(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.storage.artifacts import ArtifactStore
        from nomorals.storage.blob import BlobStore
        from nomorals.storage.db import Database

        self.tmp = tempfile.mkdtemp(dir=os.environ.get(
            "NM_TEST_TMPDIR") or None)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = Database(":memory:")
        self.db.migrate()
        self.addCleanup(self.db.close)
        blobs = BlobStore(self.db, Path(self.tmp) / "blobs")
        self.store = ArtifactStore(self.db, blobs)

    def _quarantined(self, data: bytes, *, age: float = 0.0,
                     **put_kw: Any) -> Any:
        from nomorals.agents.decoder import _quarantine_metadata

        sha = hashlib.sha256(data).hexdigest()
        art = self.store.put(
            data, type="file", creator="decoder",
            mime="application/octet-stream",
            metadata=_quarantine_metadata("decoded.bin", "binary", sha),
            **put_kw)
        if age:
            self.db.execute(
                "UPDATE artifacts SET created_at = created_at - ? "
                "WHERE id = ?", (age, art.id))
        return art

    def test_oldest_evicted_first_journaled_bytes_reclaimed(self) -> None:
        from nomorals.agents.decoder import sweep_quarantine

        old = self._quarantined(b"O" * 600, age=300)
        mid = self._quarantined(b"M" * 500, age=200)
        new = self._quarantined(b"N" * 400, age=100)
        result = sweep_quarantine(self.store, budget=1_000)
        self.assertEqual(
            [e["artifact_id"] for e in result["evicted"]], [old.id])
        self.assertEqual(result["bytes_reclaimed"], 600)
        self.assertEqual(result["resident_bytes_after"], 900)
        record = self.store.get(old.id)
        self.assertIsNotNone(record)
        evicted = (record.metadata or {}).get("evicted")
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted["reason"], "quarantine-budget")
        self.assertEqual(evicted["size"], 600)
        self.assertFalse(self.store.blobs.exists(old.content_hash))
        self.assertTrue(self.store.blobs.exists(mid.content_hash))
        self.assertTrue(self.store.blobs.exists(new.content_hash))

    def test_under_budget_is_a_noop(self) -> None:
        from nomorals.agents.decoder import sweep_quarantine

        self._quarantined(b"small")
        result = sweep_quarantine(self.store, budget=10_000)
        self.assertEqual(result["evicted"], [])
        self.assertIsNone(result["error"])

    def test_released_and_kept_artifacts_are_never_evicted(self) -> None:
        from nomorals.agents.decoder import (
            _live_quarantine_row,
            sweep_quarantine,
        )

        aliased = self._quarantined(b"A" * 900, age=400)
        self.store.set_alias("released-copy", aliased.id)
        linked = self._quarantined(b"L" * 900, age=300,
                                   mission_id="m-1")
        parent = self._quarantined(b"P" * 100, age=50)
        child = self.store.put(
            b"derived", type="note", creator="analyst",
            provenance={"derived_from": [parent.id]})
        self.assertIsNotNone(child)
        plain = self._quarantined(b"X" * 900, age=200)
        general = self.store.put(b"G" * 5_000, type="report",
                                 creator="analyst")
        result = sweep_quarantine(self.store, budget=100)
        # only the unprotected plain resident is evictable; the
        # total stays over budget and the sweep says so honestly
        self.assertEqual(
            [e["artifact_id"] for e in result["evicted"]], [plain.id])
        self.assertGreater(result["resident_bytes_after"], 100)
        for kept in (aliased, linked, parent, general):
            self.assertIsNotNone(self.store.get(kept.id))
            self.assertNotIn("evicted",
                             (self.store.get(kept.id).metadata or {}))
            self.assertTrue(self.store.blobs.exists(kept.content_hash))
        # the evicted row no longer dedups: the same bytes would
        # quarantine fresh, while a live row still dedups
        self.assertIsNone(
            _live_quarantine_row(self.store, plain.content_hash))
        self.assertEqual(
            _live_quarantine_row(self.store, aliased.content_hash).id,
            aliased.id)


# ─────────────────── staged quarantine (agent) ─────────────────

class StagedQuarantineTests(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.core.config import load_settings

        self.tmp = tempfile.TemporaryDirectory(prefix="decstage-")
        self.addCleanup(self.tmp.cleanup)
        self.stage = os.path.join(self.tmp.name, "staging")
        os.makedirs(self.stage)
        settings = load_settings(
            overrides={"home": self.tmp.name,
                       "partner.platforms": "local",
                       "chat.local_enabled": "true"})
        self.ctx = build_context(settings, with_executor=False,
                                 with_tools=True, with_router=False)
        self.addCleanup(self.ctx.close)

    def test_big_binary_decode_quarantines_from_staging(self) -> None:
        from nomorals.agents.decoder import (
            DecoderAgent,
            _artifact_store_for,
        )

        payload = b"\x89PNG\r\n\x1a\n" + os.urandom(2 * 1024 * 1024)
        blob = gzip.compress(payload)
        agent = DecoderAgent(context=self.ctx, name="t")
        env = {"NM_DECODER_STAGING_DIR": self.stage}
        with mock.patch.dict(os.environ, env), mock.patch.object(
                dec, "MATERIALIZE_LIMIT", 1 << 20):
            r = agent.run({"data": blob, "save": True,
                           "explain": False})
            self.assertTrue(r.ok, r.error)
            art = r.output["artifact"]
            self.assertIsNotNone(art)
            self.assertEqual(art["size"], len(payload))
            self.assertEqual(art["sha256"],
                             hashlib.sha256(payload).hexdigest())
            store = _artifact_store_for(self.ctx)
            self.assertIsNotNone(store)
            self.assertEqual(
                store.get(art["id"]).metadata.get("staged"), "streamed")
            self.assertEqual(store.read(art["id"]), payload)
            # the staging file went to the store, not left behind
            self.assertEqual(sorted(os.listdir(self.stage)), [])
            # dedup: the same bytes again reuse the same row
            r2 = agent.run({"data": blob, "save": True,
                            "explain": False})
            self.assertTrue(r2.ok, r2.error)
            self.assertEqual(r2.output["artifact"]["id"], art["id"])


if __name__ == "__main__":
    unittest.main()
