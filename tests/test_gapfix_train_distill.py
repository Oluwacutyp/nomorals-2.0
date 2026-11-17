"""Gapfix: the periodic distillation loop actually runs.

``training/distillation.py`` was complete and test-pinned but had zero
production callers: no kept cloud turn ever became a training pair, no
scheduled distill ever fired, and no distilled adapter ever reached the
promotion gate. This file pins the wiring: a durable pair store fed by
the live conversation log, a daily driver that harvests kept cloud
turns, trains the local adapter when enough pairs exist, and promotes
through the TrainingRegistry gate — plus the scheduler job registered
at boot and the ``nm train --distill`` handle.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.storage.conversations import ConversationLog
from nomorals.storage.db import Database

CLOUD_MODEL = "llama-3.3-70b-versatile"
OWNER_MODEL = "Cutyp/codebeast-7b-vl"
# Fixture clock: inside the driver's 30-day first-tick lookback of the
# ``now`` the driver tests pass (2_000_000_000.0). Real 2023 timestamps
# would fall outside the window and never harvest.
BASE_TS = 1_999_990_000.0


def _driver_module():
    from nomorals.training import distill_driver

    return distill_driver


def _ctx(home, db):
    return SimpleNamespace(
        settings=SimpleNamespace(home=str(home)), db=db)


def _home_db(tmp):
    home = Path(tmp) / "home"
    home.mkdir()
    db = Database(":memory:")
    db.migrate()
    return home, db


def _seed_kept(db, n, *, model=CLOUD_MODEL, base_ts=BASE_TS, prefix="c"):
    """``n`` kept cloud turns, one conversation each.

    Each turn: owner prompt -> cloud answer -> owner follow-up, so the
    turn is provably kept (the owner continued past it). Returns the
    newest timestamp seeded.
    """
    log = ConversationLog(db)
    for i in range(n):
        conv = f"{prefix}-conv-{i}"
        log.ensure_conversation(
            conv, title=f"fixture {i}", agent="partner",
            channel="telegram", now=base_ts)
        ts = base_ts + i * 1000
        log.append_message(
            row_id=f"{prefix}u{i}", conversation_id=conv, role="user",
            content=(f"Explain concept number {i} in full detail please, "
                     f"with examples and reasons"),
            name="owner", model="", created_at=ts)
        log.append_message(
            row_id=f"{prefix}a{i}", conversation_id=conv, role="assistant",
            content=(f"Concept number {i} works like this in full detail, "
                     f"with examples and reasons why it matters for the work."),
            name="Devon", model=model, created_at=ts + 10)
        log.append_message(
            row_id=f"{prefix}f{i}", conversation_id=conv, role="user",
            content=f"Got it, thanks — moving on to the next question {i}",
            name="owner", model="", created_at=ts + 20)
    return base_ts + n * 1000


def _seed_unkept(db, *, model=CLOUD_MODEL, base_ts=BASE_TS):
    """One cloud turn with no owner reply after it — not yet kept."""
    log = ConversationLog(db)
    log.ensure_conversation(
        "conv-pending", title="pending fixture", agent="partner",
        channel="telegram", now=base_ts)
    log.append_message(
        row_id="u0", conversation_id="conv-pending", role="user",
        content="Explain concept zero in full detail please, with examples",
        name="owner", model="", created_at=base_ts)
    log.append_message(
        row_id="a0", conversation_id="conv-pending", role="assistant",
        content=("Concept zero works like this in full detail, with "
                 "examples and reasons why it matters for the work."),
        name="Devon", model=model, created_at=base_ts + 10)
    return base_ts + 10


class HarvestTests(unittest.TestCase):
    def test_kept_cloud_turn_harvests_as_pair(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            _home, db = _home_db(tmp)
            _seed_kept(db, 1)
            result = mod.harvest_kept_turns(db, since=0.0, pending=[])
            self.assertEqual(len(result["pairs"]), 1)
            pair = result["pairs"][0]
            self.assertEqual(pair.source, "kept")
            self.assertEqual(pair.teacher, CLOUD_MODEL)
            self.assertIn("concept number 0", pair.prompt)
            self.assertIn("Concept number 0", pair.answer)
            self.assertEqual(result["pending"], [])
            self.assertEqual(result["stats"]["kept"], 1)

    def test_turn_without_continuation_stays_pending_then_harvests(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            _home, db = _home_db(tmp)
            newest = _seed_unkept(db)
            first = mod.harvest_kept_turns(db, since=0.0, pending=[])
            self.assertEqual(first["pairs"], [])
            self.assertEqual(len(first["pending"]), 1)
            self.assertEqual(first["stats"]["pending"], 1)
            # The owner replies: the same turn is kept on the next tick.
            log = ConversationLog(db)
            log.append_message(
                row_id="f0", conversation_id="conv-pending", role="user",
                content="Got it, thanks — moving on now",
                name="owner", model="", created_at=newest + 10)
            second = mod.harvest_kept_turns(
                db, since=first["seen_at"], pending=first["pending"])
            self.assertEqual(len(second["pairs"]), 1)
            self.assertEqual(second["pending"], [])

    def test_owner_model_turns_never_harvest(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            _home, db = _home_db(tmp)
            _seed_kept(db, 2, model=OWNER_MODEL)
            result = mod.harvest_kept_turns(db, since=0.0, pending=[])
            self.assertEqual(result["pairs"], [])
            self.assertEqual(result["pending"], [])
            self.assertEqual(result["stats"]["owner_model"], 2)

    def test_synthetic_markers_never_harvest(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            _home, db = _home_db(tmp)
            _seed_kept(db, 1, model="tool-loop", prefix="t")
            _seed_kept(db, 1, model="stt-failure", prefix="s")
            result = mod.harvest_kept_turns(db, since=0.0, pending=[])
            self.assertEqual(result["pairs"], [])
            self.assertEqual(result["pending"], [])

    def test_command_turns_are_noise(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            _home, db = _home_db(tmp)
            log = ConversationLog(db)
            log.ensure_conversation(
                "conv-cmd", title="cmd", agent="partner",
                channel="telegram", now=BASE_TS)
            log.append_message(
                row_id="u0", conversation_id="conv-cmd", role="user",
                content="/play some music now please", name="owner",
                model="", created_at=BASE_TS)
            log.append_message(
                row_id="a0", conversation_id="conv-cmd", role="assistant",
                content=("Playing your music now in full detail with "
                         "examples and reasons why it matters."),
                name="Devon", model=CLOUD_MODEL, created_at=BASE_TS + 10)
            log.append_message(
                row_id="f0", conversation_id="conv-cmd", role="user",
                content="Got it, thanks — moving on now",
                name="owner", model="", created_at=BASE_TS + 20)
            result = mod.harvest_kept_turns(db, since=0.0, pending=[])
            self.assertEqual(result["pairs"], [])
            self.assertEqual(result["stats"]["noise_dropped"], 1)

    def test_pii_is_scrubbed_before_capture(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            _home, db = _home_db(tmp)
            log = ConversationLog(db)
            log.ensure_conversation(
                "conv-pii", title="pii", agent="partner",
                channel="telegram", now=BASE_TS)
            log.append_message(
                row_id="u0", conversation_id="conv-pii", role="user",
                content=("My email is owner@example.com — explain concept "
                         "zero in full detail please"),
                name="owner", model="", created_at=BASE_TS)
            log.append_message(
                row_id="a0", conversation_id="conv-pii", role="assistant",
                content=("Concept zero works like this in full detail, with "
                         "examples and reasons why it matters for the work."),
                name="Devon", model=CLOUD_MODEL, created_at=BASE_TS + 10)
            log.append_message(
                row_id="f0", conversation_id="conv-pii", role="user",
                content="Got it, thanks — moving on now",
                name="owner", model="", created_at=BASE_TS + 20)
            result = mod.harvest_kept_turns(db, since=0.0, pending=[])
            self.assertEqual(len(result["pairs"]), 1)
            self.assertNotIn("owner@example.com", result["pairs"][0].prompt)


class StoreTests(unittest.TestCase):
    def test_store_dedupes_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            from nomorals.training.distillation import DistillationStore, capture_owner_kept

            path = Path(tmp) / "distillation.jsonl"
            store = DistillationStore(path)
            pair = capture_owner_kept(
                "Explain concept zero", "", "Concept zero in full detail.",
                teacher=CLOUD_MODEL)
            self.assertTrue(store.add(pair))
            self.assertFalse(store.add(pair))
            self.assertEqual(len(store), 1)
            reloaded = DistillationStore(path)
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded.pairs()[0].teacher, CLOUD_MODEL)

    def test_corrupt_state_degrades_to_empty(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "distillation.json"
            path.write_text("{not json", encoding="utf-8")
            state = mod.DistillState(path)
            self.assertEqual(state.last_harvest_at, 0.0)
            self.assertEqual(state.pending, [])
            self.assertTrue(state.degraded_reason)


class DriverTests(unittest.TestCase):
    def test_tick_trains_and_promotes_through_gate(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 40)
            ctx = _ctx(home, db)
            result = mod.run_distill_once(ctx, now=2_000_000_000.0)
            self.assertEqual(result["status"], "trained")
            self.assertTrue(result["gate_passed"])
            self.assertTrue(result["promoted"])
            self.assertGreaterEqual(result["pairs"], 32)
            checkpoint = Path(str(result["checkpoint"]))
            self.assertTrue(checkpoint.exists())
            from nomorals.llm.registry import ModelRegistry

            active = ModelRegistry(db).active()
            self.assertIsNotNone(active)
            self.assertEqual(active.name, "distillation")

    def test_too_few_pairs_defers_and_says_so(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 5)
            ctx = _ctx(home, db)
            result = mod.run_distill_once(ctx, now=2_000_000_000.0)
            self.assertEqual(result["status"], "deferred")
            self.assertEqual(result["pairs"], 5)
            self.assertIn("5", result["reason"])
            self.assertIn("32", result["reason"])

    def test_second_immediate_run_is_not_due_then_force_runs(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 40)
            ctx = _ctx(home, db)
            first = mod.run_distill_once(ctx, now=2_000_000_000.0)
            self.assertEqual(first["status"], "trained")
            second = mod.run_distill_once(ctx, now=2_000_000_000.0 + 60)
            self.assertEqual(second["status"], "not_due")
            # Harvest still ran on the not-due tick: pairs accrue anyway.
            self.assertGreaterEqual(second["pairs"], 32)
            third = mod.run_distill_once(
                ctx, now=2_000_000_000.0 + 120, force=True)
            self.assertEqual(third["status"], "trained")

    def test_failed_train_leaves_previous_checkpoint_in_place(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 40)
            ctx = _ctx(home, db)
            first = mod.run_distill_once(ctx, now=2_000_000_000.0)
            checkpoint = Path(str(first["checkpoint"]))
            self.assertTrue(checkpoint.exists())
            model_file = checkpoint / "model.json"
            before = model_file.read_bytes() if model_file.is_file() else None
            with mock.patch.object(
                    mod, "periodic_distill",
                    side_effect=RuntimeError("trainer exploded")):
                result = mod.run_distill_once(
                    ctx, now=2_000_000_000.0 + 90000, force=True)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(checkpoint.exists())
            if before is not None:
                self.assertEqual(model_file.read_bytes(), before)

    def test_promotion_notifies_once_and_deferral_notifies_nobody(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 40)
            ctx = _ctx(home, db)
            calls: list[tuple[str, str, str]] = []

            class _Notifier:
                def publish(self, kind, title, body, **_kw):
                    calls.append((kind, title, body))

            result = mod.run_distill_once(
                ctx, now=2_000_000_000.0, notifier=_Notifier())
            self.assertEqual(result["status"], "trained")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "training")
            self.assertIn("distill", calls[0][1].lower())
            calls.clear()
            deferred = mod.run_distill_once(
                ctx, now=2_000_000_000.0 + 60, notifier=_Notifier())
            self.assertEqual(deferred["status"], "not_due")
            self.assertEqual(calls, [])

    def test_reharvest_dedupes_instead_of_double_counting(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 5)
            ctx = _ctx(home, db)
            first = mod.run_distill_once(ctx, now=2_000_000_000.0)
            self.assertEqual(first["pairs"], 5)
            # A re-scan of the same history adds nothing new.
            state = mod.DistillState(mod.default_distill_state_path(home))
            state.last_harvest_at = 0.0
            state.save()
            second = mod.run_distill_once(
                ctx, now=2_000_000_000.0 + 60, force=True)
            self.assertEqual(second["pairs"], 5)
            self.assertEqual(second["harvested"], 0)


class JobTests(unittest.TestCase):
    def test_ensure_distill_job_is_idempotent(self):
        mod = _driver_module()
        sched = _FakeScheduler()
        first = mod.ensure_distill_job(sched)
        second = mod.ensure_distill_job(sched)
        self.assertEqual(first, second)
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(first["name"], mod.DISTILL_JOB)
        self.assertEqual(first["payload"]["tool"], mod.DISTILL_TOOL)
        stale = mod.ensure_distill_job(sched, spec="every 1h")
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(stale["spec"], "every 1h")

    def test_ensure_returns_none_without_a_scheduler(self):
        mod = _driver_module()
        self.assertIsNone(mod.ensure_distill_job(None))


class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"id": f"job-{len(self.jobs)}", "name": name,
               "spec": spec, "kind": kind, "payload": payload,
               "routine": routine}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


class CliTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            backends=False, status=False, export_telegram=False,
            continual=False, distill=True, quick=False, run=False,
            force=True, json=False, backend="", base_model="",
            dataset="", no_register=False)

    def test_nm_train_distill_runs_one_tick(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 40)
            ctx = _ctx(home, db)
            from nomorals.cmdline.commands.train import _cmd_train

            rc = _cmd_train(self._args(), ctx)
            self.assertEqual(rc, 0)
            state = mod.DistillState(mod.default_distill_state_path(home))
            self.assertTrue(state.runs())
            self.assertEqual(state.runs()[-1]["status"], "trained")

    def test_nm_train_distill_defers_with_few_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, db = _home_db(tmp)
            _seed_kept(db, 5)
            ctx = _ctx(home, db)
            from nomorals.cmdline.commands.train import _cmd_train

            rc = _cmd_train(self._args(), ctx)
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
