"""WS2 P7 — GoldenSwing bundle evidence + the digest goals section.

Pins:

* the trading provider's ``gs-evidence`` opt-in: a tagged goal is
  measured from the trade-journal MEMBER of the newest GoldenSwing
  state bundle in the artifact store — a REAL GS-shaped bundle
  built in-test the way GS's own writer builds it (stdlib zip,
  family ``manifest.json``, STATE.md prose, the WS1 contract
  manifest written last) — with GS record semantics (``type`` /
  ``pnl_money`` / ISO-8601 ``ts``), the goal's ``pnl-target:`` tag
  still supplying the denominator, and the source labelled
  ``goldenswing bundle``;
* the verification gate: a bundle that fails the seam's
  verification yields NO reading, the failure is named in the log
  (artifact id + the seam's problems), and the reader does NOT
  fall back to an older bundle or to the Devon desk journal;
* absence is honest: a tagged goal with no bundle available
  measures nothing (self-reported fallback intact), with the
  absence logged — never a silent source mix;
* precedence: without the tag, P3 behaviour is untouched — the
  bundle is ignored and the desk journal is the only source;
* corrupt journal lines / unusable records are COUNTED in the
  evidence (GS's own health-read semantics), never absorbed;
  STATE.md prose is never parsed into a number; GSJ1 frame counts
  come from a conservative structural walk only (exact-landing or
  None);
* the finance weekly digest gains a goals section: finance-class
  tracker goals with their readings, basis labels verbatim
  (``measured · savings ledger`` / ``self-reported``), absent when
  no finance goals exist, and contained when the tracker breaks.
"""

from __future__ import annotations

import binascii
import hashlib
import io
import json
import os
import struct
import tempfile
import time
import unittest
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.archives import create_bundle
from nomorals.finance.digest import compose_weekly_digest
from nomorals.finance.goals import Goal as SavingsGoal
from nomorals.finance.goals import GoalStore
from nomorals.finance.ledger import Ledger
from nomorals.goals import GoalTracker, providers, surface
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

UID = "owner"
T0 = 1_767_225_600.0  # 2026-01-01T00:00:00Z — the goal's filing time

K500K = 50_000_000   # ₦500,000 in kobo
K150K = 15_000_000   # ₦150,000 in kobo


def _iso(ts: float) -> str:
    """GS's journal timestamp shape: ISO-8601 UTC with +00:00."""
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _goal(**kwargs: Any) -> SimpleNamespace:
    base = dict(goal_id="goal_test", user_id=UID, title="Grow the GS book",
                tags=[], created_at=T0, goal_class="trading",
                progress=0.0, status="active")
    base.update(kwargs)
    return SimpleNamespace(**base)


# ── a real GS-shaped state bundle, built the way GS builds it ─────

_GSJ1_HEADER = struct.Struct(">4sIQQQII")
_GSJ1_GENESIS = hashlib.sha256(
    b"goldenswing-deterministic-journal-v1").digest()


def _gsj1_bytes(payloads: list[bytes]) -> bytes:
    """GSJ1 frames exactly as GoldenSwing's engine.journal writes."""
    out = bytearray()
    chain = _GSJ1_GENESIS
    for seq, payload in enumerate(payloads):
        header = _GSJ1_HEADER.pack(
            b"GSJ1", len(payload), seq, 1_767_225_600 + seq, 1, 1,
            len(payload))
        crc = struct.pack(
            ">I", binascii.crc32(header + payload) & 0xFFFFFFFF)
        chain = hashlib.sha256(chain + header + payload + crc).digest()
        out += header + payload + crc + chain
    return bytes(out)


def build_gs_state_bundle(
    journal_lines: list[Any],
    *,
    created_epoch: float,
    frames: int = 2,
    journal_records_declared: int | None = None,
    tamper_journal: bool = False,
) -> bytes:
    """A GoldenSwing state bundle, byte for byte GS's writer shape.

    Members under data-dir-relative arcnames; the family
    ``manifest.json`` (GS's own export manifest, declared counts
    included) and STATE.md prose ride as ordinary members; the WS1
    contract manifest (``bundle.manifest.json``, created_by
    goldenswing, kind journal-state) is written LAST and records
    every other member's size + SHA-256.
    """
    journal = ("\n".join(
        line if isinstance(line, str) else json.dumps(line)
        for line in journal_lines) + "\n").encode("utf-8")
    gsj = _gsj1_bytes([b'{"seq": 0}', b'{"seq": 1}'][:frames])
    members: dict[str, bytes] = {
        "trade_journal.jsonl": journal,
        "deterministic/deterministic.gsj": gsj,
        "risk_state.json": b'{"daily_pnl": -12.5, "open_risk": 40.0}',
        "STATE.md": (
            b"# GoldenSwing state export\n\n"
            b"The desk made 999999.99 today, probably. Prose is not "
            b"evidence and no reader may turn this sentence into a "
            b"progress number.\n"),
    }
    readable = 0
    for line in journal_lines:
        if isinstance(line, dict):
            readable += 1
    family = {
        "bundle": "goldenswing state export",
        "exported_utc": _iso(created_epoch),
        "journal": "/goldenswing/data/trade_journal.jsonl",
        "journal_records": (readable if journal_records_declared is None
                            else journal_records_declared),
        "deterministic_frames": frames,
        "deterministic_head_chain": "ab" * 32,
        "files": [
            {"arcname": name, "bytes": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
        ],
        "zip_bytes": 0,
    }
    members["manifest.json"] = json.dumps(
        family, indent=2, ensure_ascii=False).encode("utf-8")
    contract = {
        "format": "nomorals-bundle",
        "version": 1,
        "created_by": "goldenswing",
        "created_at": created_epoch,
        "kind": "journal-state",
        "members": [
            {"name": name, "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest(),
             "is_dir": False}
            for name, data in members.items()
        ],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            if tamper_journal and name == "trade_journal.jsonl":
                # Post-manifest tamper: same length, different
                # bytes — the recorded size still matches, the
                # recorded SHA-256 does not.
                data = data.replace(b"250.0", b"950.0", 1)
            zf.writestr(name, data)
        zf.writestr("bundle.manifest.json", json.dumps(
            contract, indent=2, ensure_ascii=False).encode("utf-8"))
    return buf.getvalue()


def standard_journal() -> list[Any]:
    """GS journal lines around the goal window at T0.

    Readable records: 8.  In-window closes: +250.0 and −50.0
    (realized 200.0).  A pre-goal close (+999.0) must not count;
    an open and a note carry no P&L; one close's ``pnl_money`` is
    unusable; one close's ``ts`` is unusable; one line is torn.
    """
    return [
        {"ts": _iso(T0 - 3600), "type": "open", "alert_id": "a0"},
        {"ts": _iso(T0 - 100), "type": "close", "alert_id": "a0",
         "pnl_money": 999.0},
        {"ts": _iso(T0 + 10), "type": "close", "alert_id": "a1",
         "pnl_money": 250.0, "r_net": 1.5},
        {"ts": _iso(T0 + 20), "type": "close", "alert_id": "a2",
         "pnl_money": -50.0, "r_net": -0.3},
        {"ts": _iso(T0 + 30), "type": "open", "alert_id": "a3"},
        {"ts": _iso(T0 + 40), "type": "close", "alert_id": "a3",
         "pnl_money": "abc"},
        {"ts": "not-a-timestamp", "type": "close", "alert_id": "a4",
         "pnl_money": 500.0},
        "{torn journal line",
        {"ts": _iso(T0 + 50), "type": "note", "text": "session end"},
    ]


class _BundleCase(unittest.TestCase):
    """File-backed context db + artifact store + patched settings."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.home = Path(self._tmp.name)
        self.fin_dir = self.home / "finance"
        self.fin_dir.mkdir(parents=True)
        self.db = Database(self.home / "context.db")
        self.db.migrate()  # the runtime's db is migrated at startup
        self.store = ArtifactStore(
            self.db, BlobStore(self.db, self.home / "blobs"))
        self._real_get_settings = providers.get_settings
        providers.get_settings = lambda: SimpleNamespace(
            home_path=self.home)

    def tearDown(self) -> None:
        providers.get_settings = self._real_get_settings
        self._tmp.cleanup()

    # seeding helpers -------------------------------------------------
    def register(self, data: bytes, *, created_at: float | None = None):
        art = self.store.put(data, type="file", creator="goldenswing",
                             mime="application/zip")
        if created_at is not None:
            self.db.execute(
                "UPDATE artifacts SET created_at = ? WHERE id = ?",
                (created_at, art.id))
            art = self.store.get(art.id)
        return art

    def register_bundle(self, journal_lines: list[Any] | None = None,
                        **kwargs: Any):
        data = build_gs_state_bundle(
            standard_journal() if journal_lines is None else journal_lines,
            created_epoch=kwargs.pop("created_epoch", T0 + 90000),
            **kwargs)
        return self.register(data)

    def write_desk_journal(self, records: list[Any]) -> None:
        lines = [r if isinstance(r, str) else json.dumps(r)
                 for r in records]
        (self.fin_dir / "trading_journal.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")


class BundleEvidenceTests(_BundleCase):

    def test_tagged_goal_reads_bundle_exact(self) -> None:
        art = self.register_bundle()
        goal = _goal(tags=["pnl-target:1000", "gs-evidence"])
        reading = providers.trading_progress(self.db, goal)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 20.0)
        self.assertEqual(reading.basis, "measured")
        self.assertEqual(reading.source, "goldenswing bundle")
        ev = reading.evidence
        self.assertEqual(ev["artifact_id"], art.id)
        self.assertEqual(ev["artifact_uri"], art.uri)
        self.assertEqual(ev["bundle_kind"], "journal-state")
        self.assertIs(ev["bundle_verified"], True)
        self.assertEqual(ev["journal_member"], "trade_journal.jsonl")
        self.assertEqual(ev["journal_records"], 8)
        self.assertEqual(ev["closed_trades"], 2)
        self.assertEqual(ev["realized_pnl"], 200.0)
        self.assertEqual(ev["target_pnl"], 1000.0)
        self.assertEqual(ev["unreadable_journal_lines"], 1)
        self.assertEqual(ev["unreadable_pnl_records"], 1)
        self.assertEqual(ev["records_with_unreadable_ts"], 1)
        self.assertEqual(ev["journal_span"]["first"], _iso(T0 - 3600))
        self.assertEqual(ev["journal_span"]["last"], _iso(T0 + 50))
        self.assertEqual(ev["deterministic_frames"], 2)
        # GS's declared counts ride labelled as declared — the
        # reading's own numbers above were counted from the members.
        self.assertEqual(ev["gs_declared"]["journal_records"], 8)
        self.assertEqual(ev["gs_declared"]["deterministic_frames"], 2)
        day = datetime.fromtimestamp(
            T0 + 90000, UTC).strftime("%Y-%m-%d")
        self.assertIn("+200.00", reading.headline)
        self.assertIn(day, reading.headline)
        # STATE.md's prose number (999999.99) is nowhere in the math.
        self.assertNotIn("999999", reading.headline)

    def test_gs_evidence_tag_with_value_also_opts_in(self) -> None:
        self.register_bundle()
        goal = _goal(tags=["pnl-target:1000", "gs-evidence:latest"])
        reading = providers.trading_progress(self.db, goal)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.source, "goldenswing bundle")

    def test_tagged_goal_without_target_measures_nothing(self) -> None:
        self.register_bundle()
        goal = _goal(tags=["gs-evidence"])
        self.assertIsNone(providers.trading_progress(self.db, goal))

    def test_tampered_bundle_yields_no_reading_and_is_named(self) -> None:
        art = self.register_bundle(tamper_journal=True)
        # A rich Devon desk journal exists — the tagged goal must
        # NOT fall back to it under any label.
        self.write_desk_journal([
            {"ts": T0 + 5, "event": "paper_close", "pnl": 5000.0}])
        goal = _goal(tags=["pnl-target:1000", "gs-evidence"])
        with self.assertLogs("nomorals.goals.providers",
                             level="WARNING") as caught:
            reading = providers.trading_progress(self.db, goal)
        self.assertIsNone(reading)
        joined = "\n".join(caught.output)
        self.assertIn("FAILED verification", joined)
        self.assertIn(art.id, joined)
        self.assertIn("sha256 mismatch: trade_journal.jsonl", joined)

    def test_tagged_goal_with_no_bundle_measures_nothing(self) -> None:
        self.store.put(b"just a note, not a bundle", type="text",
                       creator="owner", mime="text/plain")
        self.write_desk_journal([
            {"ts": T0 + 5, "event": "paper_close", "pnl": 5000.0}])
        goal = _goal(tags=["pnl-target:1000", "gs-evidence"])
        with self.assertLogs("nomorals.goals.providers",
                             level="WARNING") as caught:
            reading = providers.trading_progress(self.db, goal)
        self.assertIsNone(reading)
        self.assertIn("no GoldenSwing state bundle", "\n".join(caught.output))

    def test_untagged_goal_ignores_bundle_reads_desk_journal(self) -> None:
        self.register_bundle()
        self.write_desk_journal([
            {"ts": T0 + 5, "event": "paper_close", "pnl": 100.0}])
        goal = _goal(tags=["pnl-target:1000"])
        reading = providers.trading_progress(self.db, goal)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 10.0)
        self.assertEqual(reading.source, "trading journal")

    def test_newest_verified_bundle_wins(self) -> None:
        # Registered newest FIRST: precedence is the artifacts'
        # created_at order, not insertion order.
        newer = self.register_bundle(
            [{"ts": _iso(T0 + 10), "type": "close", "pnl_money": 100.0}],
            created_epoch=T0 + 2 * 90000)
        older = self.register_bundle(
            [{"ts": _iso(T0 + 10), "type": "close", "pnl_money": 900.0}],
            created_epoch=T0 + 90000)
        self.db.execute(
            "UPDATE artifacts SET created_at = ? WHERE id = ?",
            (T0 + 200000, newer.id))
        self.db.execute(
            "UPDATE artifacts SET created_at = ? WHERE id = ?",
            (T0 + 100000, older.id))
        goal = _goal(tags=["pnl-target:1000", "gs-evidence"])
        reading = providers.trading_progress(self.db, goal)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 10.0)
        self.assertEqual(reading.evidence["artifact_id"], newer.id)

    def test_non_state_bundles_are_skipped(self) -> None:
        state = self.register_bundle()
        other_path = self.home / "other.zip"
        create_bundle(
            other_path,
            [("trade_journal.jsonl",
              b'{"ts": 1, "type": "close", "pnl_money": 777.0}\n')],
            created_by="nomorals", kind="artifact-export",
            container="zip")
        newer_other = self.register(other_path.read_bytes())
        self.db.execute(
            "UPDATE artifacts SET created_at = ? WHERE id = ?",
            (T0 + 300000, newer_other.id))
        self.db.execute(
            "UPDATE artifacts SET created_at = ? WHERE id = ?",
            (T0 + 100000, state.id))
        goal = _goal(tags=["pnl-target:1000", "gs-evidence"])
        reading = providers.trading_progress(self.db, goal)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.evidence["artifact_id"], state.id)
        self.assertEqual(reading.percent, 20.0)

    def test_bundle_with_no_in_window_closes_reads_zero(self) -> None:
        self.register_bundle(
            [{"ts": _iso(T0 - 50), "type": "close", "pnl_money": 42.0}])
        goal = _goal(tags=["pnl-target:1000", "gs-evidence"])
        reading = providers.trading_progress(self.db, goal)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.percent, 0.0)
        self.assertEqual(reading.evidence["closed_trades"], 0)
        self.assertIn("no closed trades", reading.headline)

    def test_surface_renders_bundle_reading_measured(self) -> None:
        tracker = GoalTracker(self.db)
        goal = surface.run_sync(tracker.create(
            UID, "Grow the GS book", goal_class="trading",
            tags=["pnl-target:1000", "gs-evidence"]))
        self.register_bundle([
            {"ts": _iso(goal.created_at + 10), "type": "close",
             "pnl_money": 125.0},
            {"ts": _iso(goal.created_at + 20), "type": "close",
             "pnl_money": 75.0},
        ])
        text = surface.render_goal_progress(tracker, UID, "Grow the GS book")
        self.assertIn("20% (measured · goldenswing bundle)", text)


class Gsj1WalkTests(unittest.TestCase):

    def test_walk_counts_real_frames(self) -> None:
        data = _gsj1_bytes([b"a", b"bb", b"ccc"])
        self.assertEqual(providers._gsj1_frame_count(data), 3)

    def test_walk_empty_member_is_zero(self) -> None:
        self.assertEqual(providers._gsj1_frame_count(b""), 0)

    def test_walk_torn_tail_is_none_not_a_guess(self) -> None:
        data = _gsj1_bytes([b"a", b"bb"])[:-5]
        self.assertIsNone(providers._gsj1_frame_count(data))

    def test_walk_bad_magic_is_none(self) -> None:
        data = b"XXXX" + _gsj1_bytes([b"a"])[4:]
        self.assertIsNone(providers._gsj1_frame_count(data))


class DigestGoalsSectionTests(_BundleCase):

    def _ctx(self, *, with_db: bool = True) -> SimpleNamespace:
        ctx = SimpleNamespace(
            settings=SimpleNamespace(home_path=str(self.home)))
        if with_db:
            ctx.db = self.db
        return ctx

    def _tracker(self) -> GoalTracker:
        return GoalTracker(self.db)

    def _create(self, tracker: GoalTracker, title: str, **kwargs: Any):
        return surface.run_sync(tracker.create(UID, title, **kwargs))

    def test_digest_renders_finance_goals_with_basis_labels(self) -> None:
        saved = SavingsGoal(
            id="goal_seed0001", name="Emergency Fund",
            target_kobo=K500K, created_at=time.time() - 86400)
        GoalStore(self.fin_dir / "goals.json").add(saved)
        Ledger(self.fin_dir / "ledger.jsonl").log(
            K150K, category="savings", kind="income",
            ts=saved.created_at + 60)
        tracker = self._tracker()
        self._create(tracker, "Emergency fund", goal_class="finance")
        manual = self._create(tracker, "House deposit",
                              goal_class="finance")
        surface.run_sync(tracker.update_progress(manual.goal_id, 40.0))
        self._create(tracker, "Grow the desk", goal_class="trading",
                     tags=["pnl-target:1000"])
        result = compose_weekly_digest(self._ctx())
        self.assertTrue(result["ok"])
        text = result["text"]
        self.assertIn("🎯 finance goals:", text)
        self.assertIn(
            "Emergency fund — 30% (measured · savings ledger)", text)
        self.assertIn("House deposit — 40% (self-reported)", text)
        # The trading goal is not a finance goal: no section line.
        self.assertNotIn("Grow the desk —", text)

    def test_digest_without_finance_goals_has_no_section(self) -> None:
        tracker = self._tracker()
        self._create(tracker, "Grow the desk", goal_class="trading",
                     tags=["pnl-target:1000"])
        result = compose_weekly_digest(self._ctx())
        self.assertTrue(result["ok"])
        self.assertNotIn("🎯 finance goals:", result["text"])

    def test_digest_without_db_has_no_section(self) -> None:
        result = compose_weekly_digest(self._ctx(with_db=False))
        self.assertTrue(result["ok"])
        self.assertNotIn("🎯 finance goals:", result["text"])

    def test_digest_goals_section_is_contained_when_tracker_breaks(
            self) -> None:
        ctx = self._ctx()
        ctx.db = object()  # no query/execute: the tracker cannot read
        result = compose_weekly_digest(ctx)
        self.assertTrue(result["ok"])
        self.assertNotIn("🎯 finance goals:", result["text"])


if __name__ == "__main__":
    unittest.main()
