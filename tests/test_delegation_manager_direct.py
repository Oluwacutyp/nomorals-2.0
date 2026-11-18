"""Delegation collect: direct-to-origin-chat delivery + per-worker timeout.

Lane 71. The gap, re-verified from code at 24e1a9d0:

* ``collect()`` delivers only through the notifier (the owner's
  resolved channels) and never reads the run's stored ``chat_key`` —
  a delegation started in a group chat reports back to the owner's
  channels, not to the chat it started from.
* Workers run on in-process threads with no per-worker timeout:
  ``poll()`` skips not-done futures forever and ``recover()`` leaves
  rows with a live future alone, so a stuck worker hangs the
  delegation forever.

Tests use the same fakes as ``tests/test_delegation_manager.py``
(scripted roster worker, recording notifier) plus a fake gateway
recording direct sends to a chat key.
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.delegation import DelegationStore
from nomorals.agents.delegation_manager import DelegationManager

try:  # the feature under test; None on pristine so tests fail
    # behaviorally (wrong delivery / no reap) instead of at import.
    from nomorals.agents.delegation_manager import (
        DEFAULT_WORKER_TIMEOUT_S,
    )
except ImportError:  # pragma: no cover - pristine base only
    DEFAULT_WORKER_TIMEOUT_S = None
from nomorals.core.config import _ENV_MAP, Settings
from nomorals.storage.db import Database


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _FakeAgent:
    """One scripted behavior per invocation; ("block", event) waits."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, role, input_dict):
        self.calls.append({"role": role,
                           "input": dict(input_dict or {})})
        behavior = self.script[min(len(self.calls) - 1,
                                   len(self.script) - 1)]
        kind = behavior[0]
        if kind == "ok":
            return {"ok": True, "result": behavior[1]}
        if kind == "fail":
            return {"ok": False, "error": behavior[1]}
        if kind == "block":
            behavior[1].wait(timeout=30)
            return {"ok": True, "result": {"unblocked": True}}
        raise AssertionError(f"unknown behavior {behavior!r}")


class _FakeNotifier:
    """Records every publish; delivery always succeeds."""

    def __init__(self):
        self.published: list[dict] = []

    def __call__(self, kind, title, body, **_kw):
        self.published.append({"kind": kind, "title": title,
                               "body": body})
        return {"delivered": True, "delivery_state": "sent"}


class _FakeGateway:
    """Records direct sends; ``ok`` controls the reported outcome."""

    def __init__(self, ok=True):
        self.sent: list[dict] = []
        self.ok = ok

    def send(self, platform, chat, text, **_kw):
        self.sent.append({"platform": platform, "chat": chat,
                          "text": text})
        return SimpleNamespace(ok=self.ok, platform=platform,
                               message_id="fake-1")


def _manager(db, fake, notifier, **kw):
    store = DelegationStore(db)
    return DelegationManager(store, db=db, run_agent=fake,
                             notifier=notifier, **kw)


def _spawn(manager, **kw):
    base = dict(role="tester", goal="cover the new retry path")
    base.update(kw)
    return manager.spawn(**base)


def _context_with_gateway(gateway):
    return SimpleNamespace(db=None, extras={"gateway": gateway},
                           settings=Settings())


class DirectDeliveryTests(unittest.TestCase):
    def test_collect_delivers_direct_to_origin_chat(self):
        db = _db()
        fake = _FakeAgent([("ok", {"tests": 12, "passed": 12})])
        notifier = _FakeNotifier()
        gateway = _FakeGateway()
        context = _context_with_gateway(gateway)
        manager = _manager(db, fake, notifier, context=context)
        run = _spawn(manager, chat_key="telegram:123")
        manager.wait(run.id, timeout=10)
        outcome = manager.collect(run.id)
        self.assertTrue(outcome["ok"])
        self.assertTrue(outcome["delivered"])
        self.assertEqual(outcome["delivery"], "direct")
        self.assertEqual(outcome["delivery_target"], "telegram:123")
        # The result lands in the origin chat — and nowhere else.
        self.assertEqual(len(gateway.sent), 1)
        sent = gateway.sent[0]
        self.assertEqual(sent["platform"], "telegram")
        self.assertEqual(sent["chat"], "telegram:123")
        self.assertIn(run.contract_id, sent["text"])
        self.assertEqual(len(notifier.published), 0)
        self.assertEqual(manager.get_run(run.id).status, "collected")
        # A second collect replays without a second direct send.
        again = manager.collect(run.id)
        self.assertTrue(again["replayed"])
        self.assertEqual(len(gateway.sent), 1)
        self.assertEqual(len(notifier.published), 0)
        manager.shutdown(wait=False)

    def test_collect_falls_back_to_notifier_without_origin(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        notifier = _FakeNotifier()
        gateway = _FakeGateway()
        manager = _manager(db, fake, notifier,
                           context=_context_with_gateway(gateway))
        run = _spawn(manager)  # no chat_key recorded
        manager.wait(run.id, timeout=10)
        outcome = manager.collect(run.id)
        self.assertTrue(outcome["ok"])
        self.assertTrue(outcome["delivered"])
        self.assertEqual(outcome["delivery"], "notifier")
        self.assertEqual(len(gateway.sent), 0)
        self.assertEqual(len(notifier.published), 1)
        manager.shutdown(wait=False)

    def test_collect_falls_back_to_notifier_when_direct_fails(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        notifier = _FakeNotifier()
        gateway = _FakeGateway(ok=False)
        manager = _manager(db, fake, notifier,
                           context=_context_with_gateway(gateway))
        run = _spawn(manager, chat_key="telegram:123")
        manager.wait(run.id, timeout=10)
        outcome = manager.collect(run.id)
        # Direct failed, so the result still reaches the owner
        # through the notifier — named, never silently dropped.
        self.assertTrue(outcome["ok"])
        self.assertTrue(outcome["delivered"])
        self.assertEqual(outcome["delivery"], "notifier")
        self.assertEqual(len(gateway.sent), 1)
        self.assertEqual(len(notifier.published), 1)
        self.assertTrue(any("telegram:123" in r
                            for r in outcome["reasons"]))
        manager.shutdown(wait=False)

    def test_collect_falls_back_without_any_gateway(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier)  # no context at all
        run = _spawn(manager, chat_key="telegram:123")
        manager.wait(run.id, timeout=10)
        outcome = manager.collect(run.id)
        self.assertTrue(outcome["ok"])
        self.assertTrue(outcome["delivered"])
        self.assertEqual(outcome["delivery"], "notifier")
        self.assertEqual(len(notifier.published), 1)
        manager.shutdown(wait=False)

    def test_collect_direct_with_explicit_gateway_kwarg(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        notifier = _FakeNotifier()
        gateway = _FakeGateway()
        manager = _manager(db, fake, notifier, gateway=gateway)
        run = _spawn(manager, chat_key="whatsapp:999")
        manager.wait(run.id, timeout=10)
        outcome = manager.collect(run.id)
        self.assertEqual(outcome["delivery"], "direct")
        self.assertEqual(len(gateway.sent), 1)
        self.assertEqual(gateway.sent[0]["platform"], "whatsapp")
        self.assertEqual(len(notifier.published), 0)
        manager.shutdown(wait=False)

    def test_poll_collects_to_origin_chat(self):
        db = _db()
        fake = _FakeAgent([("ok", {"done": True})])
        notifier = _FakeNotifier()
        gateway = _FakeGateway()
        manager = _manager(db, fake, notifier,
                           context=_context_with_gateway(gateway))
        run = _spawn(manager, chat_key="telegram:123")
        manager.wait(run.id, timeout=10)
        out = manager.poll()
        self.assertIn(run.id, out["collected"])
        self.assertEqual(len(gateway.sent), 1)
        self.assertEqual(len(notifier.published), 0)
        manager.shutdown(wait=False)


class WorkerTimeoutTests(unittest.TestCase):
    def test_timeout_default_is_sane_and_configured(self):
        self.assertEqual(DEFAULT_WORKER_TIMEOUT_S, 1800.0)
        self.assertEqual(Settings().delegation.worker_timeout_s,
                         1800.0)
        self.assertEqual(
            _ENV_MAP.get("NM_DELEGATION_WORKER_TIMEOUT_S"),
            "delegation.worker_timeout_s")

    def test_manager_reads_timeout_from_env(self):
        db = _db()
        with mock.patch.dict(os.environ,
                             {"NM_DELEGATION_WORKER_TIMEOUT_S": "42"}):
            manager = _manager(db, _FakeAgent([]), _FakeNotifier())
        self.assertEqual(manager.worker_timeout_s, 42.0)
        manager.shutdown(wait=False)

    def test_manager_kwarg_beats_env(self):
        db = _db()
        with mock.patch.dict(os.environ,
                             {"NM_DELEGATION_WORKER_TIMEOUT_S": "42"}):
            manager = _manager(db, _FakeAgent([]), _FakeNotifier(),
                               worker_timeout_s=7.0)
        self.assertEqual(manager.worker_timeout_s, 7.0)
        manager.shutdown(wait=False)

    def test_stuck_worker_is_reaped_and_reported(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier, worker_timeout_s=0.2)
        run = _spawn(manager, chat_key="telegram:123")
        try:
            time.sleep(0.6)
            out = manager.poll()
            self.assertIn(run.id, [r["run_id"]
                                   for r in out["timed_out"]])
            reaped = manager.get_run(run.id)
            self.assertEqual(reaped.status, "failed")
            self.assertIn("timed out", reaped.error)
            self.assertIn("0.2", reaped.error)
            # The failure is reported through the delivery path
            # (no gateway here, so the notifier) naming the timeout.
            self.assertEqual(len(notifier.published), 1)
            self.assertIn("timed out", notifier.published[0]["title"])
            self.assertIn("timed out", notifier.published[0]["body"])
            # The failure is recorded on the contract (status failed,
            # failure claim on the ledger) — and the contract stays
            # open by design, so the run can resume: a timeout is a
            # failed attempt, not a dead contract.
            store = DelegationStore(db)
            contract = store.get_contract(run.contract_id)
            self.assertEqual(contract.status, "failed")
            self.assertFalse(contract.terminal)
            failures = [c for c in store.claims(contract.id)
                        if c.kind == "failure"]
            self.assertTrue(failures)
            self.assertIn("timed out", failures[-1].claim)
            # The late worker's output is fenced: when the stuck
            # thread finally returns, the run stays failed and the
            # late result is never recorded.
            event.set()
            time.sleep(0.5)
            after = manager.get_run(run.id)
            self.assertEqual(after.status, "failed")
            self.assertNotIn("unblocked", after.result_json)
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_timeout_notice_goes_direct_to_origin_chat(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        notifier = _FakeNotifier()
        gateway = _FakeGateway()
        manager = _manager(db, fake, notifier, worker_timeout_s=0.2,
                           context=_context_with_gateway(gateway))
        _spawn(manager, chat_key="telegram:123")
        try:
            time.sleep(0.6)
            out = manager.poll()
            self.assertEqual(len(out["timed_out"]), 1)
            self.assertEqual(out["timed_out"][0]["delivery"], "direct")
            self.assertEqual(len(gateway.sent), 1)
            self.assertIn("timed out", gateway.sent[0]["text"])
            self.assertEqual(len(notifier.published), 0)
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_timeout_disabled_when_zero(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event)])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier, worker_timeout_s=0)
        run = _spawn(manager)
        try:
            time.sleep(0.3)
            out = manager.poll()
            self.assertEqual(out["timed_out"], [])
            self.assertEqual(manager.get_run(run.id).status, "running")
        finally:
            event.set()
            manager.shutdown(wait=False)

    def test_resume_gets_a_fresh_deadline(self):
        db = _db()
        event = threading.Event()
        fake = _FakeAgent([("block", event), ("block", event)])
        notifier = _FakeNotifier()
        manager = _manager(db, fake, notifier, worker_timeout_s=0.2)
        run = _spawn(manager)
        try:
            time.sleep(0.6)
            manager.poll()
            self.assertEqual(manager.get_run(run.id).status, "failed")
            resumed = manager.resume(run.id, by="owner")
            self.assertEqual(resumed.status, "running")
            # Fresh deadline: a poll right after resume must not
            # reap the new attempt.
            out = manager.poll()
            self.assertEqual(out["timed_out"], [])
            self.assertEqual(manager.get_run(run.id).status, "running")
        finally:
            event.set()
            manager.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
