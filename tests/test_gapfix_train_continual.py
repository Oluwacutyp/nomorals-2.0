"""Gapfix R2-32: the continual-update loop actually runs.

The confined continual-update loop (`training/continual.py`) was complete
and test-pinned but had zero production callers: no durable replay buffer
ever accrued and no scheduled update ever fired. This file pins the
wiring: a durable store, a daily driver fed by the live representation
ledger, an accepted candidate saved as the live preference adapter and
served by fresh brains, and a scheduler job registered at boot.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.cognition.representation import RepresentationLedger
from nomorals.storage.db import Database
from nomorals.training.preference import PreferenceAdapter, PreferencePair

CHOSEN = "concise summary cite source verified"
REJECTED = "verbose guess no citation rambling"


def _driver_module():
    from nomorals.training import continual_driver

    return continual_driver


def _ctx(home, db):
    return SimpleNamespace(
        settings=SimpleNamespace(home=str(home)), db=db)


def _seed_ledger(db, n=12):
    """n owner-correction pairs in the live ledger (real feed)."""
    ledger = RepresentationLedger(db)
    for i in range(n):
        rec = ledger.log_action(
            "message", f"draft answer number {i} was too wordy",
            outcome="success")
        assert ledger.record_outcome(
            rec.id, "success", owner_feedback=-1,
            feedback_text=f"{CHOSEN} case {i}")
    return ledger


def _seed_swapped_ledger(db, n=20):
    """n pairs that are the TRUE swap of the phase-1 pairs: the owner
    correction now prefers the wordy draft over the concise summary.
    A batch dominated by these must regress the frozen harness (which
    was frozen on the good pairs), so the update is rejected."""
    ledger = RepresentationLedger(db)
    for i in range(n):
        rec = ledger.log_action(
            "message", f"{CHOSEN} case {i}",
            outcome="success")
        assert ledger.record_outcome(
            rec.id, "success", owner_feedback=-1,
            feedback_text=f"draft answer number {i} was too wordy")
    return ledger


def _ledger_pair_count(db):
    from nomorals.training.preference import build_preference_pairs

    pairs, _stats = build_preference_pairs(RepresentationLedger(db))
    return len(pairs)


class StoreTests(unittest.TestCase):
    def test_buffer_accrues_durably_across_runs(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "continual.json"
            store = mod.ContinualStore(path)
            pairs = [PreferencePair(prompt=f"p{i}", chosen=CHOSEN,
                                    rejected=REJECTED) for i in range(5)]
            self.assertEqual(store.add_pairs(pairs), 5)
            store.save()
            # A fresh store over the same file still holds them.
            reopened = mod.ContinualStore(path)
            self.assertEqual(len(reopened.pairs()), 5)
            self.assertEqual(
                {p.prompt for p in reopened.pairs()},
                {f"p{i}" for i in range(5)})
            # Re-adding the same pairs adds nothing (dedupe by pair_id).
            self.assertEqual(reopened.add_pairs(pairs), 0)

    def test_corrupt_store_degrades_to_empty(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "continual.json"
            path.write_text("{not json", encoding="utf-8")
            store = mod.ContinualStore(path)
            self.assertTrue(store.degraded_reason)
            self.assertEqual(store.pairs(), [])
            self.assertEqual(store.last_run_at, 0.0)
            quarantined = list(Path(tmp).glob("continual.json.*"))
            self.assertEqual(len(quarantined), 1)
            # The store still works after degrading.
            store.add_pairs([PreferencePair(prompt="p", chosen=CHOSEN,
                                            rejected=REJECTED)])
            store.save()
            self.assertEqual(len(mod.ContinualStore(path).pairs()), 1)

    def test_harness_freezes_and_never_resplits(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "continual.json"
            store = mod.ContinualStore(path)
            pairs = [PreferencePair(prompt=f"p{i}", chosen=CHOSEN,
                                    rejected=REJECTED,
                                    created_at=float(i)) for i in range(15)]
            store.add_pairs(pairs)
            store.save()
            fp1 = store.freeze_harness()
            self.assertTrue(fp1)
            self.assertEqual(len(store.harness_pairs()), 3)
            # More pairs arrive; the frozen harness does not move.
            store.add_pairs([PreferencePair(
                prompt="p-new", chosen=CHOSEN, rejected=REJECTED,
                created_at=100.0)])
            store.save()
            reopened = mod.ContinualStore(path)
            self.assertEqual(reopened.harness_fingerprint(), fp1)
            self.assertEqual(len(reopened.harness_pairs()), 3)


class DriverTests(unittest.TestCase):
    def _home_db(self, tmp):
        home = Path(tmp) / "home"
        home.mkdir()
        db = Database(str(Path(tmp) / "test.db"))
        return home, db

    def test_update_fires_when_due_and_skips_when_not_due(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = self._home_db(tmp)
            _seed_ledger(db, 20)
            ctx = _ctx(home, db)
            first = mod.run_continual_update_once(ctx, now=1_000_000.0)
            self.assertIn(first["status"], ("accepted", "rejected"))
            self.assertEqual(first["buffer"], 20)
            # Immediately again: not due (24h interval).
            second = mod.run_continual_update_once(ctx, now=1_000_000.0 + 60)
            self.assertEqual(second["status"], "not_due")
            # Forced: runs again.
            third = mod.run_continual_update_once(
                ctx, force=True, now=1_000_000.0 + 120)
            self.assertIn(third["status"], ("accepted", "rejected"))

    def test_not_enough_data_defers(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = self._home_db(tmp)
            ctx = _ctx(home, db)
            result = mod.run_continual_update_once(ctx, force=True)
            self.assertEqual(result["status"], "deferred")
            self.assertIn("not enough", result["reason"])

    def test_failed_update_leaves_previous_adapter_in_place(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = self._home_db(tmp)
            _seed_ledger(db, 20)
            ctx = _ctx(home, db)
            first = mod.run_continual_update_once(ctx, force=True,
                                                  now=1_000_000.0)
            self.assertEqual(first["status"], "accepted")
            adapter_path = (mod.default_preference_adapter_dir(home)
                            / "preference_adapter.json")
            before_bytes = adapter_path.read_bytes()
            # A store whose frozen harness is good pairs but whose
            # buffer is ALL swapped pairs: every replay sample trains
            # the wrong direction, so the fixed harness must refuse
            # the candidate — deterministically, whatever the random
            # sample draws. Built as a store file directly; the
            # durable format is the store's contract.
            from nomorals.training.continual import FixedEvalHarness

            good = [PreferencePair(
                prompt=("Act on the owner's behalf (message): draft "
                        f"answer number {i} was too wordy"),
                chosen="short",
                rejected=f"draft answer number {i} was too wordy",
                created_at=float(i)) for i in range(4)]
            harness = FixedEvalHarness(good)
            swapped = [PreferencePair(
                prompt="Act on the owner's behalf (message): short",
                chosen=f"draft answer number {i} was too wordy",
                rejected="short",
                created_at=float(100 + i)) for i in range(20)]
            store_path = home / "data" / "continual.json"
            store_path.write_text(json.dumps({
                "buffer": [{"pair": p.to_dict(), "kind": "pair",
                            "added_at": p.created_at} for p in swapped],
                "last_run_at": 0.0,
                "harness": {"pairs": [p.to_dict() for p in good],
                            "fingerprint": harness.fingerprint},
                "runs": [],
            }), encoding="utf-8")
            # A fresh, empty ledger: nothing new is harvested, so the
            # run trains on exactly the swapped buffer.
            fresh_db = Database(str(Path(tmp) / "fresh.db"))
            second = mod.run_continual_update_once(
                _ctx(home, fresh_db), now=2_000_000.0)
            self.assertEqual(second["status"], "rejected")
            self.assertEqual(second["reason"],
                             "fixed-harness regression (forgetting)")
            self.assertEqual(adapter_path.read_bytes(), before_bytes)

    def test_update_is_confined_to_its_own_artifacts(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = self._home_db(tmp)
            _seed_ledger(db, 20)
            ctx = _ctx(home, db)
            result = mod.run_continual_update_once(ctx, force=True,
                                                  now=1_000_000.0)
            self.assertIn(result["status"], ("accepted", "rejected"))
            # Only the driver's own files exist under the home: the
            # store and (on accept) the adapter. No training run, no
            # dataset, no model dir, no mission was created.
            created = sorted(p.relative_to(home).as_posix()
                             for p in home.rglob("*") if p.is_file())
            self.assertEqual(created, ["data/continual.json"]
                             + (["training/preference-adapter/"
                                 "preference_adapter.json"]
                                if result["status"] == "accepted" else []))
            from nomorals.missions import MissionStore

            try:
                self.assertEqual(MissionStore(db).list(), [])
            except Exception as exc:  # noqa: BLE001 - table absent = none
                # No missions table at all: no mission was ever created.
                self.assertIn("no such table: missions", str(exc))
            from nomorals.training.registry import TrainingRegistry

            try:
                total = TrainingRegistry(db).stats().get("total", 0)
            except Exception as exc:  # noqa: BLE001 - table absent = none
                # No training_runs table at all: no run was registered.
                self.assertIn("no such table: training_runs", str(exc))
                total = 0
            self.assertEqual(total, 0)

    def test_harness_fingerprint_stable_across_runs(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = self._home_db(tmp)
            _seed_ledger(db, 20)
            ctx = _ctx(home, db)
            first = mod.run_continual_update_once(ctx, force=True,
                                                  now=1_000_000.0)
            fp1 = first["fingerprint"]
            self.assertTrue(fp1)
            _seed_ledger(db, 6)
            second = mod.run_continual_update_once(ctx, force=True,
                                                   now=1_000_000.0 + 90000)
            self.assertEqual(second["fingerprint"], fp1)

    def test_accepted_update_notifies_once(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = self._home_db(tmp)
            _seed_ledger(db, 20)
            ctx = _ctx(home, db)
            notes: list[tuple[str, str, str]] = []

            class _FakeNotifier:
                def publish(self, kind, title="", body="", **_kw):
                    notes.append((kind, title, body))

            result = mod.run_continual_update_once(
                ctx, force=True, now=1_000_000.0,
                notifier=_FakeNotifier())
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0][0], "training")
            # A not-due run notifies nobody.
            mod.run_continual_update_once(ctx, now=1_000_000.0 + 60,
                                          notifier=_FakeNotifier())
            self.assertEqual(len(notes), 1)


class ServingTests(unittest.TestCase):
    def test_saved_adapter_loads_and_brain_serves_it(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            self.assertIsNone(mod.load_saved_adapter(home))
            adapter = PreferenceAdapter(
                name="preference-v1",
                token_weights={"concise": 2.0},
                lessons=["prefer concise summaries"],
                pair_count=12)
            adapter.save(mod.default_preference_adapter_dir(home))
            loaded = mod.load_saved_adapter(home)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.token_weights, {"concise": 2.0})
            self.assertEqual(loaded.lessons, ["prefer concise summaries"])
            # A fresh brain built for this home serves the saved adapter.
            from nomorals.llm.brain import brain_for

            ctx = _ctx(home, Database(str(Path(tmp) / "test.db")))
            ctx.router = object()
            brain = brain_for(ctx)
            self.assertIsNotNone(brain.preference_adapter)
            self.assertEqual(
                brain.preference_adapter.token_weights, {"concise": 2.0})
            self.assertEqual(
                brain.preference_lessons(), ["prefer concise summaries"])

    def test_brain_without_saved_adapter_serves_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            from nomorals.llm.brain import brain_for

            ctx = _ctx(home, Database(str(Path(tmp) / "test.db")))
            ctx.router = object()
            brain = brain_for(ctx)
            self.assertIsNone(brain.preference_adapter)
            self.assertEqual(brain.preference_lessons(), [])


class JobTests(unittest.TestCase):
    def test_ensure_continual_update_job_is_idempotent(self):
        mod = _driver_module()
        sched = _FakeScheduler()
        first = mod.ensure_continual_update_job(sched)
        second = mod.ensure_continual_update_job(sched)
        self.assertEqual(first, second)
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(first["name"], mod.CONTINUAL_JOB)
        self.assertEqual(first["payload"]["tool"], mod.CONTINUAL_TOOL)
        stale = mod.ensure_continual_update_job(sched, spec="every 1h")
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(stale["spec"], "every 1h")

    def test_ensure_returns_none_without_a_scheduler(self):
        mod = _driver_module()
        self.assertIsNone(mod.ensure_continual_update_job(None))


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
    def test_nm_train_continual_runs_one_update(self):
        mod = _driver_module()
        with tempfile.TemporaryDirectory() as tmp:
            home, db = DriverTests._home_db(self, tmp)
            _seed_ledger(db, 20)
            ctx = _ctx(home, db)
            from nomorals.cmdline.commands.train import _cmd_train

            args = SimpleNamespace(backends=False, run=False,
                                   backend="", base_model="",
                                   continual=True, force=True, json=False)
            rc = _cmd_train(args, ctx)
            self.assertEqual(rc, 0)
            store = mod.ContinualStore(mod.default_continual_path(home))
            self.assertEqual(len(store.pairs()), 20)
            self.assertTrue(store.runs())


if __name__ == "__main__":
    unittest.main()
