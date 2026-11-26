"""Inference-job system (census §3B.1, artcraft-services pattern).

Proves the mined design end to end: the eight-state machine with
every legal/illegal edge, the poll contract (single + batch), the
wallet/402 gate, provider webhooks in (signed, idempotent) and
client webhooks out (signed, retried, dead-lettered), and the API
routes that expose it all. The tool count stays 748 — no new brain
tool was added; the job system is exercised through the existing
API server surface.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.api.server import APIServer
from nomorals.core.errors import (
    PaymentRequired,
    ValidationError,
)
from nomorals.core.inference_jobs import (
    CATEGORIES,
    COSTS,
    TERMINAL_STATUSES,
    TRANSITIONS,
    DuplicateIdempotencyToken,
    IllegalTransition,
    InferenceJobStatus,
    InferenceJobStore,
    InsufficientBalance,
    WalletStore,
    estimate_cost,
    webhook_retry_delay,
)
from nomorals.core.policy import CapabilitySet
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

S = InferenceJobStatus
LIVE = (S.PENDING, S.STARTED, S.ATTEMPT_FAILED)


def _token() -> str:
    return str(uuid.uuid4())


def _result(url: str = "https://cdn.example/x.mp4") -> dict[str, Any]:
    return {"entity_token": "m_test", "entity_type": "media_file",
            "media_links": {"cdn_url": url}}


class StatusTests(unittest.TestCase):
    def test_eight_states_with_mined_wire_values(self) -> None:
        self.assertEqual(
            list(S.ALL),
            ["pending", "started", "complete_success", "complete_failure",
             "attempt_failed", "dead", "cancelled_by_user",
             "cancelled_by_system"])

    def test_terminal_set_is_the_mined_five(self) -> None:
        self.assertEqual(
            TERMINAL_STATUSES,
            {"complete_success", "complete_failure", "dead",
             "cancelled_by_user", "cancelled_by_system"})
        for state in LIVE:
            self.assertFalse(S.is_terminal(state))
        for state in TERMINAL_STATUSES:
            self.assertTrue(S.is_terminal(state))

    def test_from_str_round_trip_and_rejection(self) -> None:
        for state in S.ALL:
            self.assertEqual(S.from_str(state), state)
        with self.assertRaises(ValidationError):
            S.from_str("running")
        with self.assertRaises(ValidationError):
            S.from_str("done")


class TransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InferenceJobStore()

    def _job_in(self, status: str) -> Any:
        job = self.store.submit("image_generation", "devon-local",
                                "test", idempotency_token=_token())
        if status == S.PENDING:
            return job
        if status == S.STARTED:
            return self.store.start(job.token)
        if status == S.ATTEMPT_FAILED:
            self.store.start(job.token)
            return self.store.record_attempt_failure(job.token)
        if status == S.COMPLETE_SUCCESS:
            self.store.start(job.token)
            return self.store.complete(job.token, _result())
        if status == S.COMPLETE_FAILURE:
            self.store.start(job.token)
            return self.store.fail(job.token, "model_error")
        if status == S.DEAD:
            job2 = self.store.submit(
                "image_generation", "devon-local", "test",
                idempotency_token=_token(), max_attempts=1)
            self.store.start(job2.token)
            return self.store.record_attempt_failure(job2.token)
        if status == S.CANCELLED_BY_USER:
            return self.store.cancel(job.token, by="user")
        if status == S.CANCELLED_BY_SYSTEM:
            return self.store.cancel(job.token, by="system")
        raise AssertionError(status)

    def test_every_legal_edge_moves(self) -> None:
        for source, targets in TRANSITIONS.items():
            for target in sorted(targets):
                with self.subTest(source=source, target=target):
                    job = self._job_in(source)
                    moved = self.store.transition(job.token, target)
                    self.assertEqual(moved.status, target)

    def test_every_illegal_edge_raises_naming_both_ends(self) -> None:
        for source in S.ALL:
            for target in S.ALL:
                if target in TRANSITIONS.get(source, frozenset()):
                    continue
                with self.subTest(source=source, target=target):
                    job = self._job_in(source)
                    with self.assertRaises(IllegalTransition) as ctx:
                        self.store.transition(job.token, target)
                    self.assertIn(source, str(ctx.exception))
                    self.assertIn(target, str(ctx.exception))
                    # The failed move left the job exactly where it was.
                    self.assertEqual(self.store.get(job.token).status, source)

    def test_terminal_states_never_move(self) -> None:
        for terminal in sorted(TERMINAL_STATUSES):
            job = self._job_in(terminal)
            for target in S.ALL:
                with self.subTest(terminal=terminal, target=target), \
                        self.assertRaises(IllegalTransition):
                    self.store.transition(job.token, target)


class SubmitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InferenceJobStore()

    def test_token_shape_is_jinf_plus_25_crockford_chars(self) -> None:
        job = self.store.submit("image_generation", "devon-local", "x",
                                idempotency_token=_token())
        self.assertTrue(job.token.startswith("jinf_"))
        self.assertEqual(len(job.token), 30)
        alphabet = set("0123456789abcdefghjkmnpqrstvwxyz")
        self.assertTrue(set(job.token[5:]) <= alphabet)
        self.assertNotEqual(
            job.token,
            self.store.submit("image_generation", "devon-local", "y",
                              idempotency_token=_token()).token)

    def test_cost_table_is_devons_own_and_scales_with_duration(self) -> None:
        self.assertEqual(estimate_cost("image_generation"), 10)
        self.assertEqual(estimate_cost("text_generation"), 5)
        self.assertEqual(
            estimate_cost("video_generation", {"duration_seconds": 6}), 120)
        self.assertEqual(
            estimate_cost("audio_generation", {"duration_seconds": 30}), 80)
        for category in CATEGORIES:
            self.assertIn(category, COSTS)
        with self.assertRaises(ValidationError):
            estimate_cost("teleportation")
        with self.assertRaises(ValidationError):
            estimate_cost("video_generation", {"duration_seconds": -1})

    def test_submit_without_wallet_runs_unbilled_and_says_so(self) -> None:
        job = self.store.submit("image_generation", "devon-local", "x",
                                idempotency_token=_token())
        self.assertEqual(job.cost_credits, 10)
        self.assertFalse(job.billed)
        self.assertIsNone(job.bill)
        self.assertEqual(job.wallet, "unavailable")
        self.assertEqual(job.status, S.PENDING)

    def test_submit_requires_fresh_uuid_idempotency_token(self) -> None:
        with self.assertRaises(ValidationError):
            self.store.submit("image_generation", "devon-local", "x")
        with self.assertRaises(ValidationError):
            self.store.submit("image_generation", "devon-local", "x",
                              idempotency_token="not-a-uuid")

    def test_reused_idempotency_token_is_rejected_naming_the_original(self) -> None:
        token = _token()
        first = self.store.submit("image_generation", "devon-local", "x",
                                  idempotency_token=token)
        with self.assertRaises(DuplicateIdempotencyToken) as ctx:
            self.store.submit("image_generation", "devon-local", "y",
                              idempotency_token=token)
        self.assertEqual(ctx.exception.details["existing_job_token"],
                         first.token)
        self.assertIn(first.token, str(ctx.exception))

    def test_submit_rejects_bad_category_and_missing_model(self) -> None:
        with self.assertRaises(ValidationError):
            self.store.submit("teleportation", "devon-local", "x",
                              idempotency_token=_token())
        with self.assertRaises(ValidationError):
            self.store.submit("image_generation", "  ",
                              "x", idempotency_token=_token())


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InferenceJobStore()
        self.job = self.store.submit("image_generation", "devon-local",
                                     "a puffin", idempotency_token=_token())

    def test_happy_path_pending_started_done_with_first_started_at(self) -> None:
        self.assertIsNone(self.job.first_started_at)
        self.store.start(self.job.token)
        started = self.store.get(self.job.token)
        self.assertEqual(started.status, S.STARTED)
        self.assertIsNotNone(started.first_started_at)
        first = started.first_started_at
        self.store.report_progress(self.job.token, 42, "rendering")
        mid = self.store.get(self.job.token)
        self.assertEqual(mid.progress_percentage, 42)
        self.assertEqual(mid.last_progress_note, "rendering")
        # The result stays null until success — a poller never sees a
        # half-finished result.
        self.assertIsNone(mid.result)
        done = self.store.complete(self.job.token, _result())
        self.assertEqual(done.status, S.COMPLETE_SUCCESS)
        self.assertEqual(done.progress_percentage, 100)
        self.assertEqual(done.first_started_at, first)
        self.assertEqual(
            done.result["media_links"]["cdn_url"], "https://cdn.example/x.mp4")

    def test_complete_requires_a_result(self) -> None:
        self.store.start(self.job.token)
        with self.assertRaises(ValidationError):
            self.store.complete(self.job.token, {})

    def test_progress_clamps_and_needs_a_started_job(self) -> None:
        with self.assertRaises(IllegalTransition):
            self.store.report_progress(self.job.token, 50)
        self.store.start(self.job.token)
        self.store.report_progress(self.job.token, 250)
        self.assertEqual(self.store.get(self.job.token).progress_percentage,
                         100)
        self.store.report_progress(self.job.token, -5)
        self.assertEqual(self.store.get(self.job.token).progress_percentage,
                         0)

    def test_fail_carries_the_category(self) -> None:
        self.store.start(self.job.token)
        failed = self.store.fail(self.job.token, "model_error",
                                 "the model refused the prompt")
        self.assertEqual(failed.status, S.COMPLETE_FAILURE)
        self.assertEqual(failed.failure_category, "model_error")
        self.assertIn("refused", failed.result["error"])

    def test_attempt_failed_retries_then_goes_dead(self) -> None:
        job = self.store.submit("image_generation", "devon-local", "x",
                                idempotency_token=_token(), max_attempts=2)
        self.store.start(job.token)
        again = self.store.record_attempt_failure(job.token, "timeout")
        self.assertEqual(again.status, S.ATTEMPT_FAILED)
        self.assertEqual(again.attempts, 1)
        self.assertEqual(again.failure_category, "timeout")
        # The retry re-enters started; the first-started time is kept.
        self.store.start(job.token)
        dead = self.store.record_attempt_failure(job.token, "timeout")
        self.assertEqual(dead.status, S.DEAD)
        self.assertEqual(dead.attempts, 2)

    def test_cancel_by_user_and_by_system(self) -> None:
        by_user = self.store.cancel(self.job.token, by="user")
        self.assertEqual(by_user.status, S.CANCELLED_BY_USER)
        job2 = self.store.submit("image_generation", "devon-local", "x",
                                 idempotency_token=_token())
        self.store.start(job2.token)
        by_system = self.store.cancel(job2.token, by="system")
        self.assertEqual(by_system.status, S.CANCELLED_BY_SYSTEM)
        with self.assertRaises(ValidationError):
            self.store.cancel(job2.token, by="the moon")

    def test_run_local_drives_an_executor_end_to_end(self) -> None:
        seen: list[str] = []

        def executor(job: Any) -> dict[str, Any]:
            seen.append(job.token)
            return _result("https://cdn.example/local.png")

        done = self.store.run_local(self.job.token, executor)
        self.assertEqual(done.status, S.COMPLETE_SUCCESS)
        self.assertEqual(seen, [self.job.token])
        self.assertEqual(
            done.result["media_links"]["cdn_url"],
            "https://cdn.example/local.png")

    def test_run_local_executor_failure_is_an_attempt_failure(self) -> None:
        def boom(job: Any) -> dict[str, Any]:
            raise RuntimeError("the renderer crashed")

        out = self.store.run_local(self.job.token, boom)
        self.assertEqual(out.status, S.ATTEMPT_FAILED)
        self.assertEqual(out.failure_category, "executor_error")

    def test_run_local_needs_a_pending_job(self) -> None:
        self.store.start(self.job.token)
        with self.assertRaises(IllegalTransition):
            self.store.run_local(self.job.token, lambda j: _result())


class PollTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InferenceJobStore()
        self.job = self.store.submit(
            "video_generation", "devon-local", "a puffin",
            params={"duration_seconds": 6}, idempotency_token=_token())

    def test_single_poll_shape_is_the_mined_omni_shape(self) -> None:
        answer = self.store.job_status(self.job.token)
        self.assertTrue(answer["success"])
        state = answer["state"]
        self.assertEqual(state["job_token"], self.job.token)
        self.assertEqual(state["request"]["inference_category"],
                         "video_generation")
        self.assertEqual(state["request"]["maybe_model_type"], "devon-local")
        self.assertEqual(state["request"]["prompt"], "a puffin")
        status = state["status"]
        self.assertEqual(status["status"], "pending")
        self.assertIsNone(status["maybe_first_started_at"])
        self.assertIsNone(status["maybe_failure_category"])
        self.assertEqual(status["progress_percentage"], 0)
        self.assertIsNone(state["maybe_result"])
        self.assertIn("created_at", state)
        self.assertIn("updated_at", state)
        self.assertEqual(state["cost_credits"], 120)

    def test_batch_poll_returns_one_state_per_token_in_order(self) -> None:
        second = self.store.submit("image_generation", "devon-local", "x",
                                   idempotency_token=_token())
        answer = self.store.job_status_batch(
            [second.token, "jinf_nope", self.job.token])
        self.assertTrue(answer["success"])
        states = answer["job_states"]
        self.assertEqual(len(states), 3)
        self.assertEqual(states[0]["job_token"], second.token)
        self.assertIn("unknown job token", states[1]["error"])
        self.assertEqual(states[2]["job_token"], self.job.token)

    def test_unknown_token_raises(self) -> None:
        with self.assertRaises(ValidationError):
            self.store.job_status("jinf_nope")


class WalletTests(unittest.TestCase):
    def test_wallet_credit_charge_and_ledger(self) -> None:
        wallet = WalletStore()
        self.assertEqual(wallet.balance(), 0)
        wallet.credit(100, note="owner top-up")
        self.assertEqual(wallet.balance(), 100)
        bill = wallet.charge(30, job_token="jinf_test")
        self.assertEqual(bill["amount"], 30)
        self.assertEqual(bill["balance_before"], 100)
        self.assertEqual(bill["balance_after"], 70)
        self.assertEqual(wallet.balance(), 70)
        self.assertEqual(len(wallet.bills()), 2)

    def test_wallet_rejects_bad_amounts(self) -> None:
        wallet = WalletStore()
        for bad in (0, -5, "10", 10.5, True):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                wallet.credit(bad)  # type: ignore[arg-type]

    def test_charge_never_goes_negative_and_names_the_amounts(self) -> None:
        wallet = WalletStore()
        wallet.credit(50)
        with self.assertRaises(InsufficientBalance) as ctx:
            wallet.charge(60, job_token="jinf_test")
        self.assertEqual(ctx.exception.details["required"], 60)
        self.assertEqual(ctx.exception.details["available"], 50)
        self.assertEqual(ctx.exception.details["shortfall"], 10)
        self.assertEqual(wallet.balance(), 50)

    def test_submit_with_funded_wallet_bills_and_402s_when_empty(self) -> None:
        wallet = WalletStore()
        wallet.credit(15)
        store = InferenceJobStore(wallet=wallet)
        job = store.submit("image_generation", "devon-local", "x",
                           idempotency_token=_token())
        self.assertTrue(job.billed)
        self.assertEqual(job.wallet, "local")
        self.assertEqual(job.bill["amount"], 10)
        self.assertEqual(wallet.balance(), 5)
        with self.assertRaises(PaymentRequired) as ctx:
            store.submit("image_generation", "devon-local", "y",
                         idempotency_token=_token())
        self.assertEqual(ctx.exception.details["required"], 10)
        self.assertEqual(ctx.exception.details["available"], 5)
        self.assertEqual(ctx.exception.details["shortfall"], 5)
        # The failed submit billed nothing and recorded no job.
        self.assertEqual(wallet.balance(), 5)
        self.assertEqual(len(store.list_jobs()), 1)

    def test_wallet_persists_across_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallet.json"
            wallet = WalletStore(path)
            wallet.credit(100, note="top-up")
            wallet.charge(25, job_token="jinf_x")
            again = WalletStore(path)
            self.assertEqual(again.balance(), 75)
            self.assertEqual(len(again.bills()), 2)


class ProviderWebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InferenceJobStore()
        self.job = self.store.submit("video_generation", "devon-local",
                                     "a puffin",
                                     idempotency_token=_token())
        # Bind the provider's job id to Devon's job the way the
        # router does when it submits to a provider.
        self.store.bind_provider(self.job.token, "acme", "prov_1")

    def test_unsigned_callback_accepted_when_no_secret_registered(self) -> None:
        out = self.store.receive_provider_webhook("acme", {
            "provider_job_id": "prov_1", "event": "started"})
        self.assertTrue(out["success"])
        self.assertFalse(out["signature_verified"])
        self.assertEqual(self.store.get(self.job.token).status, S.STARTED)

    def test_signed_callback_verifies_and_bad_signature_rejected(self) -> None:
        self.store.register_provider_secret("acme", "s3cr3t")
        body = json.dumps({"provider_job_id": "prov_1",
                           "event": "started"}).encode()
        good = hmac.new(b"s3cr3t", body, hashlib.sha256).hexdigest()
        out = self.store.receive_provider_webhook(
            "acme", {"provider_job_id": "prov_1", "event": "started"},
            signature=good, raw_body=body)
        self.assertTrue(out["signature_verified"])
        with self.assertRaises(ValidationError):
            self.store.receive_provider_webhook(
                "acme", {"provider_job_id": "prov_1", "event": "started"},
                signature="0" * 64, raw_body=body)

    def test_replay_is_idempotent_and_moves_nothing_twice(self) -> None:
        event = {"provider_job_id": "prov_1", "event": "started",
                 "event_id": "evt_1"}
        first = self.store.receive_provider_webhook("acme", event)
        second = self.store.receive_provider_webhook("acme", event)
        self.assertEqual(first, second)
        self.assertEqual(self.store.get(self.job.token).status, S.STARTED)

    def test_full_provider_cycle_started_progress_succeeded(self) -> None:
        self.store.receive_provider_webhook(
            "acme", {"provider_job_id": "prov_1", "event": "started"})
        self.store.receive_provider_webhook(
            "acme", {"provider_job_id": "prov_1", "event": "progress",
                     "progress": 55})
        self.assertEqual(
            self.store.get(self.job.token).progress_percentage, 55)
        out = self.store.receive_provider_webhook(
            "acme", {"provider_job_id": "prov_1", "event": "succeeded",
                     "result": _result()})
        self.assertEqual(out["status"], S.COMPLETE_SUCCESS)
        state = self.store.job_status(self.job.token)["state"]
        self.assertEqual(state["maybe_result"]["media_links"]["cdn_url"],
                         "https://cdn.example/x.mp4")

    def test_provider_failure_is_an_attempt_failure(self) -> None:
        self.store.receive_provider_webhook(
            "acme", {"provider_job_id": "prov_1", "event": "started"})
        out = self.store.receive_provider_webhook(
            "acme", {"provider_job_id": "prov_1", "event": "failed",
                     "failure_category": "provider_timeout"})
        self.assertEqual(out["status"], S.ATTEMPT_FAILED)
        self.assertEqual(
            self.store.get(self.job.token).failure_category,
            "provider_timeout")

    def test_bind_provider_refuses_terminal_jobs_and_unknown_tokens(self) -> None:
        self.store.start(self.job.token)
        self.store.complete(self.job.token, _result())
        with self.assertRaises(IllegalTransition):
            self.store.bind_provider(self.job.token, "acme", "prov_9")
        with self.assertRaises(ValidationError):
            self.store.bind_provider("jinf_nope", "acme", "prov_9")

    def test_unknown_provider_job_and_bad_events_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.store.receive_provider_webhook(
                "acme", {"provider_job_id": "prov_nope",
                         "event": "started"})
        with self.assertRaises(ValidationError):
            self.store.receive_provider_webhook(
                "acme", {"provider_job_id": "prov_1", "event": "exploded"})
        with self.assertRaises(ValidationError):
            self.store.receive_provider_webhook(
                "acme", {"provider_job_id": "prov_1",
                         "event": "succeeded", "result": {}})


class OutboundWebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.statuses: list[int] = []
        self.store = InferenceJobStore(post=self._fake_post)

    def _fake_post(self, url: str, body: bytes,
                   headers: dict[str, str]) -> int:
        self.calls.append({"url": url, "body": body, "headers": headers})
        return self.statuses.pop(0) if self.statuses else 200

    def _finished_job(self, secret: str | None = None) -> Any:
        job = self.store.submit(
            "image_generation", "devon-local", "x",
            idempotency_token=_token(),
            webhook_url="https://client.example/hook",
            webhook_secret=secret)
        self.store.start(job.token)
        return self.store.complete(job.token, _result())

    def test_terminal_job_posts_signed_state_once(self) -> None:
        self._finished_job(secret="hook-secret")
        outcomes = self.store.deliver_webhooks(now=1000.0)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "delivered")
        self.assertEqual(outcomes[0]["status_code"], 200)
        call = self.calls[0]
        self.assertEqual(call["url"], "https://client.example/hook")
        self.assertEqual(call["headers"]["X-Devon-Event"], "job.completed")
        self.assertTrue(call["headers"]["X-Devon-Delivery-Id"])
        body = json.loads(call["body"])
        self.assertEqual(
            body["state"]["status"]["status"], "complete_success")
        expected = hmac.new(b"hook-secret", call["body"],
                            hashlib.sha256).hexdigest()
        self.assertEqual(call["headers"]["X-Devon-Signature"], expected)

    def test_failed_delivery_retries_on_schedule_then_dead_letters(self) -> None:
        self.statuses = [500, 500, 500, 500]
        self._finished_job()
        first = self.store.deliver_webhooks(now=1000.0)
        self.assertEqual(first[0]["status"], "retrying")
        # Too early: the 60s backoff has not elapsed.
        self.assertEqual(self.store.deliver_webhooks(now=1050.0), [])
        second = self.store.deliver_webhooks(now=1061.0)
        self.assertEqual(second[0]["attempt"], 2)
        self.assertEqual(second[0]["status"], "retrying")
        third = self.store.deliver_webhooks(now=1061.0 + 120.0)
        self.assertEqual(third[0]["attempt"], 3)
        fourth = self.store.deliver_webhooks(now=1061.0 + 120.0 + 240.0)
        self.assertEqual(fourth[0]["attempt"], 4)
        self.assertEqual(fourth[0]["status"], "failed")
        # Dead-lettered: no further attempts, ever.
        self.assertEqual(
            self.store.deliver_webhooks(now=100000.0), [])
        job = self.store.list_jobs()[0]
        self.assertEqual(job.deliveries[0]["status"], "failed")
        self.assertEqual(len(job.deliveries[0]["attempt_log"]), 4)

    def test_retry_delay_schedule_is_60_120_240_then_none(self) -> None:
        self.assertEqual(webhook_retry_delay(1), 60.0)
        self.assertEqual(webhook_retry_delay(2), 120.0)
        self.assertEqual(webhook_retry_delay(3), 240.0)
        self.assertIsNone(webhook_retry_delay(4))
        self.assertIsNone(webhook_retry_delay(0))

    def test_delivery_id_is_stable_across_retries(self) -> None:
        self.statuses = [500, 200]
        self._finished_job()
        self.store.deliver_webhooks(now=1000.0)
        self.store.deliver_webhooks(now=1061.0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(
            self.calls[0]["headers"]["X-Devon-Delivery-Id"],
            self.calls[1]["headers"]["X-Devon-Delivery-Id"])


class PersistenceTests(unittest.TestCase):
    def test_jobs_survive_reload_and_secrets_never_hit_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            store = InferenceJobStore(path)
            job = store.submit(
                "image_generation", "devon-local", "x",
                idempotency_token=_token(),
                webhook_url="https://client.example/hook",
                webhook_secret="do-not-persist")
            store.start(job.token)
            store.complete(job.token, _result())
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("do-not-persist", raw)
            again = InferenceJobStore(path)
            reloaded = again.get(job.token)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.status, S.COMPLETE_SUCCESS)
            self.assertEqual(
                reloaded.result["media_links"]["cdn_url"],
                "https://cdn.example/x.mp4")
            self.assertEqual(reloaded.webhook_url,
                             "https://client.example/hook")


# ── API surface (/v1/jobs) ─────────────────────────────────────────────

OWNER_TOKEN = "owner-test-token"
EMPTY_TOKEN = "empty-test-token"


def _context(tmpdir: str) -> Any:
    db = Database(":memory:")
    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0),
        backup_dir=tmpdir,
        workspace_dir=tmpdir,
    )
    return SimpleNamespace(
        db=db, settings=settings, tools=ToolRegistry(), bus=None,
        memory=None)


def _server(context: Any) -> APIServer:
    return APIServer(
        context, token=OWNER_TOKEN, max_body_bytes=64 * 1024,
        principals={
            OWNER_TOKEN: ("owner", CapabilitySet.all()),
            EMPTY_TOKEN: ("empty", CapabilitySet.of()),
        },
    )


class ApiJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.api = _server(_context(self.tmp.name))
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _submit(self, **over: Any) -> tuple[int, dict[str, Any]]:
        body = {"category": "image_generation", "model": "devon-local",
                "prompt": "a puffin", "idempotency_token": _token()}
        body.update(over)
        return self.api.dispatch("POST", "/v1/jobs", body, {},
                                 principal=self.owner)

    def test_submit_and_poll_over_dispatch(self) -> None:
        status, payload = self._submit()
        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        token = payload["inference_job_token"]
        self.assertTrue(token.startswith("jinf_"))
        self.assertEqual(payload["all_job_tokens"], [token])
        status, payload = self.api.dispatch(
            "GET", "/v1/jobs", {}, {"token": token}, principal=self.owner)
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"]["status"]["status"], "pending")
        self.assertEqual(payload["state"]["request"]["prompt"], "a puffin")

    def test_poll_batch_over_dispatch(self) -> None:
        _, first = self._submit()
        _, second = self._submit()
        status, payload = self.api.dispatch(
            "GET", "/v1/jobs", {},
            {"tokens": f"{first['inference_job_token']},"
                       f"{second['inference_job_token']}"},
            principal=self.owner)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["job_states"]), 2)

    def test_poll_needs_a_token(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/v1/jobs", {}, {}, principal=self.owner)
        self.assertEqual(status, 400)
        self.assertIn("token is required", payload["error"])

    def test_duplicate_idempotency_token_is_400(self) -> None:
        token = _token()
        self._submit(idempotency_token=token)
        status, payload = self._submit(idempotency_token=token)
        self.assertEqual(status, 400)
        self.assertIn("already used", payload["error"])

    def test_wallet_starts_unattached_then_gates_after_funding(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/v1/jobs/wallet", {}, {}, principal=self.owner)
        self.assertEqual(status, 200)
        self.assertFalse(payload["attached"])
        self.assertIsNone(payload["balance"])
        # Unbilled while no wallet is attached.
        status, payload = self._submit()
        self.assertEqual(status, 200)
        # Funding attaches the wallet (owner capability).
        status, payload = self.api.dispatch(
            "POST", "/v1/jobs/wallet/credit",
            {"amount": 15, "note": "test top-up"}, {},
            principal=self.owner)
        self.assertEqual(status, 200)
        self.assertEqual(payload["balance"], 15)
        status, payload = self.api.dispatch(
            "GET", "/v1/jobs/wallet", {}, {}, principal=self.owner)
        self.assertTrue(payload["attached"])
        self.assertEqual(payload["balance"], 15)
        # One image job bills 10; the next 402s with the amounts named.
        status, payload = self._submit()
        self.assertEqual(status, 200)
        status, payload = self._submit()
        self.assertEqual(status, 402)
        self.assertEqual(payload["kind"], "PaymentRequired")
        self.assertEqual(payload["required"], 10)
        self.assertEqual(payload["available"], 5)
        self.assertEqual(payload["shortfall"], 5)

    def test_wallet_credit_needs_sys_config(self) -> None:
        empty = self.api.resolve_principal(f"Bearer {EMPTY_TOKEN}")
        status, payload = self.api.dispatch(
            "POST", "/v1/jobs/wallet/credit", {"amount": 10}, {},
            principal=empty)
        self.assertEqual(status, 403)

    def test_cancel_over_dispatch(self) -> None:
        _, payload = self._submit()
        token = payload["inference_job_token"]
        status, payload = self.api.dispatch(
            "POST", "/v1/jobs/cancel", {"token": token, "by": "user"}, {},
            principal=self.owner)
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"]["status"]["status"],
                         "cancelled_by_user")

    def test_provider_webhook_over_dispatch(self) -> None:
        _, payload = self._submit()
        token = payload["inference_job_token"]
        # Bind the provider job the way the router does on submit.
        from nomorals.api.server import _inference_job_store
        store = _inference_job_store(self.api)
        store.bind_provider(token, "acme", "prov_9")
        body = {"provider": "acme", "provider_job_id": "prov_9",
                "event": "started"}
        status, payload = self.api.dispatch(
            "POST", "/v1/jobs/webhook", body, {}, principal=self.owner,
            raw_body=json.dumps(body).encode())
        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(store.get(token).status, S.STARTED)

    def test_docs_lists_the_job_routes_with_descriptions(self) -> None:
        status, payload = self.api.dispatch(
            "GET", "/docs", {}, {}, principal=self.owner)
        self.assertEqual(status, 200)
        routes = {(r["method"], r["path"]): r for r in payload["routes"]}
        for method, path in (
                ("POST", "/v1/jobs"), ("GET", "/v1/jobs"),
                ("POST", "/v1/jobs/cancel"), ("POST", "/v1/jobs/webhook"),
                ("GET", "/v1/jobs/wallet"),
                ("POST", "/v1/jobs/wallet/credit")):
            self.assertIn((method, path), routes)
            self.assertTrue(routes[(method, path)]["description"])

    def test_tool_count_stays_807(self) -> None:
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 807)


if __name__ == "__main__":
    unittest.main()
