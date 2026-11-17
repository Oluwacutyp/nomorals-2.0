"""Gapfix R3-46: task-scoped agent identities.

A delegated task mints ONE identity from IdentityStore at spawn,
carries it on the run record, lends it to the booking/posting paths
it drives, and retires it when the task ends — no live identity
outlives its task. All offline: fake roster workers, scripted Duffel
HTTP, fake social adapters.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.delegation import DelegationStore
from nomorals.agents.delegation_manager import DelegationManager
from nomorals.agents.identities import IdentityStore
from nomorals.core.config import Settings
from nomorals.finance.mandate import (
    SCOPE_TRAVEL,
    MandateStore,
    issue_mandate,
)
from nomorals.social import (
    PlatformAdapter,
    PostResult,
    SocialManager,
)
from nomorals.storage.db import Database
from tests.test_delegation_manager import _FakeAgent, _FakeNotifier
from tests.test_duffel import (
    OFFER_JSON,
    ORDER_JSON,
    PAX,
    FakeResponse,
    _connected,
)


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _identities() -> tuple[IdentityStore, str]:
    path = tempfile.mktemp(prefix="nm-ident46-", suffix=".db")
    return IdentityStore(db_path=path), path


def _manager(db, fake, notifier, store, **kw):
    return DelegationManager(
        DelegationStore(db), db=db, run_agent=fake,
        notifier=notifier, identity_store=store, **kw)


class SpawnIdentityTests(unittest.TestCase):
    def test_spawn_mints_task_identity(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        store, _path = _identities()
        manager = _manager(db, fake, _FakeNotifier(), store)
        try:
            run = manager.spawn(role="tester",
                                goal="cover the new retry path")
            self.assertTrue(run.identity_id.startswith("ident_"))
            self.assertEqual(run.identity_label, "Devon Tester")
            # Live while the task runs: resolvable in the store.
            ident = store.get(run.identity_id)
            self.assertIsNotNone(ident)
            self.assertEqual(ident.name, "Devon Tester")
            self.assertEqual(ident.handle, "@devon_tester")
            # The durable row carries it too.
            self.assertEqual(manager.get_run(run.id).identity_id,
                             run.identity_id)
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_two_tasks_get_different_identities(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event), ("block", event)])
        store, _path = _identities()
        manager = _manager(db, fake, _FakeNotifier(), store)
        try:
            first = manager.spawn(role="tester",
                                  goal="cover the new retry path")
            second = manager.spawn(role="tester",
                                   goal="cover the new retry path")
            self.assertNotEqual(first.identity_id, second.identity_id)
            self.assertIsNotNone(store.get(first.identity_id))
            self.assertIsNotNone(store.get(second.identity_id))
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_identity_retired_when_task_completes(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        store, _path = _identities()
        manager = _manager(db, fake, _FakeNotifier(), store)
        run = manager.spawn(role="tester",
                            goal="cover the new retry path")
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "completed")
        # No live identity outlives its task …
        self.assertIsNone(store.get(run.identity_id))
        # … but the run record still names who acted.
        self.assertEqual(done.identity_id, run.identity_id)
        self.assertEqual(done.identity_label, "Devon Tester")
        manager.shutdown(wait=False)

    def test_identity_retired_when_task_fails(self):
        db = _db()
        fake = _FakeAgent([("fail", "the worker gave up")])
        store, _path = _identities()
        manager = _manager(db, fake, _FakeNotifier(), store)
        run = manager.spawn(role="tester",
                            goal="cover the new retry path")
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "failed")
        self.assertIsNone(store.get(run.identity_id))
        self.assertEqual(done.identity_label, "Devon Tester")
        manager.shutdown(wait=False)

    def test_resume_mints_a_fresh_identity(self):
        db = _db()
        fake = _FakeAgent([("fail", "first try died"),
                           ("ok", {"done": True})])
        store, _path = _identities()
        manager = _manager(db, fake, _FakeNotifier(), store)
        run = manager.spawn(role="tester",
                            goal="cover the new retry path")
        failed = manager.wait(run.id, timeout=10)
        self.assertEqual(failed.status, "failed")
        self.assertIsNone(store.get(run.identity_id))
        resumed = manager.resume(run.id)
        self.assertNotEqual(resumed.identity_id, run.identity_id)
        self.assertTrue(resumed.identity_id.startswith("ident_"))
        self.assertIsNotNone(store.get(resumed.identity_id))
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "completed")
        self.assertIsNone(store.get(resumed.identity_id))
        manager.shutdown(wait=False)

    def test_collect_names_the_identity(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        notifier = _FakeNotifier()
        store, _path = _identities()
        manager = _manager(db, fake, notifier, store)
        run = manager.spawn(role="tester",
                            goal="cover the new retry path")
        done = manager.wait(run.id, timeout=10)
        self.assertEqual(done.status, "completed")
        out = manager.collect(run.id)
        self.assertTrue(out["ok"])
        self.assertTrue(any("Devon Tester" in p["body"]
                            for p in notifier.published))
        manager.shutdown(wait=False)

    def test_spawn_without_a_store_still_runs(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        home = tempfile.mkdtemp(prefix="nm-ident46-home-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        with mock.patch.dict("os.environ", {"HOME": home}):
            manager = DelegationManager(
                DelegationStore(db), db=db, run_agent=fake,
                notifier=_FakeNotifier())
            run = manager.spawn(role="tester",
                                goal="cover the new retry path")
            done = manager.wait(run.id, timeout=10)
            self.assertEqual(done.status, "completed")
            self.assertTrue(done.identity_id.startswith("ident_"))
            manager.shutdown(wait=False)


class BookingIdentityTests(unittest.TestCase):
    def test_create_order_carries_task_identity(self):
        store, _path = _identities()
        ident = store.create("Devon Travel", handle="@devon_travel",
                             purpose="book the owner's flight")
        c, http = _connected()
        http.route("GET", "/air/offers/off_1",
                   FakeResponse(200, {"data": OFFER_JSON}))
        http.route("POST", "/air/orders",
                   FakeResponse(200, {"data": ORDER_JSON}))
        mandates = MandateStore(path=tempfile.mktemp(suffix=".json"))
        issue_mandate(mandates, scope=SCOPE_TRAVEL,
                      cap_per_txn=500_000_00, cap_per_day=1_000_000_00)
        seen: list[dict] = []

        def _log(action, text, **kw):
            seen.append({"action": action, "text": text, **kw})

        with mock.patch(
                "nomorals.cognition.representation"
                ".log_representation_action", _log):
            order = c.create_order("off_1", PAX, confirmed=True,
                                   mandate_store=mandates,
                                   identity=ident)
        self.assertEqual(order.identity_id, ident.id)
        self.assertEqual(order.identity_label, "Devon Travel")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["action"], "purchase")
        self.assertEqual(seen[0]["metadata"]["identity_id"], ident.id)
        self.assertEqual(seen[0]["metadata"]["identity_label"],
                         "Devon Travel")

    def test_create_order_without_identity_is_devon_herself(self):
        c, http = _connected()
        http.route("GET", "/air/offers/off_1",
                   FakeResponse(200, {"data": OFFER_JSON}))
        http.route("POST", "/air/orders",
                   FakeResponse(200, {"data": ORDER_JSON}))
        mandates = MandateStore(path=tempfile.mktemp(suffix=".json"))
        issue_mandate(mandates, scope=SCOPE_TRAVEL,
                      cap_per_txn=500_000_00, cap_per_day=1_000_000_00)
        order = c.create_order("off_1", PAX, confirmed=True,
                               mandate_store=mandates)
        self.assertEqual(order.identity_id, "")
        self.assertEqual(order.identity_label, "")


class _FakeSocialAdapter(PlatformAdapter):
    def __init__(self, name: str):
        self.name = name
        self.calls: list[str] = []

    def post(self, account, content, **kwargs):
        self.calls.append(content)
        return PostResult(platform=self.name, ok=True,
                          external_id=f"{self.name}-1",
                          url=f"https://{self.name}.test/1",
                          status_code=200)


class PostingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-social46-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.social = SocialManager(self.context)
        self.adapter = _FakeSocialAdapter("alpha")
        self.social.register_adapter(self.adapter)
        self.social.connect("alpha", "@a@alpha.test")

    def test_publish_carries_task_identity(self):
        store, _path = _identities()
        ident = store.create("Devon Social", handle="@devon_social",
                             purpose="post the owner's update")
        seen: list[dict] = []

        def _log(action, text, **kw):
            seen.append({"action": action, "text": text, **kw})

        with mock.patch(
                "nomorals.cognition.representation"
                ".log_representation_action", _log):
            outcome = self.social.publish(
                "hello from the task", platforms=["alpha"],
                identity=ident)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.identity_id, ident.id)
        self.assertEqual(outcome.identity_label, "Devon Social")
        self.assertEqual(outcome.to_dict()["identity_id"], ident.id)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["action"], "post")
        self.assertEqual(seen[0]["metadata"]["identity_id"], ident.id)
        self.assertEqual(seen[0]["metadata"]["identity_label"],
                         "Devon Social")

    def test_publish_without_identity_is_devon_herself(self):
        outcome = self.social.publish("hello", platforms=["alpha"])
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.identity_id, "")
        self.assertEqual(outcome.identity_label, "")

    def test_execute_post_passes_identity_through(self):
        from nomorals.social.drafts import execute_post

        store, _path = _identities()
        ident = store.create("Devon Social", handle="@devon_social",
                             purpose="post the owner's update")
        seen: dict = {}

        class _RecordingManager:
            context = None

            def publish(self, content, **kw):
                seen.update(kw)
                return SimpleNamespace(ok=True, posted=[object()],
                                       failed=[])

        execute_post(_RecordingManager(), "hello from the task",
                     identity=ident)
        self.assertIs(seen.get("identity"), ident)


if __name__ == "__main__":
    unittest.main()
