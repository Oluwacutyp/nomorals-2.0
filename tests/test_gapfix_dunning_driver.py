"""Lane 72: the dunning scheduled driver — a real tick, not a record.

Register §8 L969: "subscriptions are records, not live billing, dunning
has no scheduled driver yet." ``dunning_due()`` exists on the gateway
but nothing ever calls it. This suite pins the driver end to end:

* the tick charges due subscriptions through the real gateway
  (fake HTTP in tests; the live HttpClient adapter in production),
  resets the retry state on success, advances the 1/3/7-day schedule
  on a failed charge, and parks the subscription past_due after the
  third failed retry;
* a subscription whose provider key is missing is SKIPPED with the
  wall named (the env var) — its retry is NOT burned on a config
  problem — and the skip is journaled like every other outcome;
* every attempt lands one durable row in the commerce store's dunning
  journal, and the tick explains itself into the why store when the
  boot-built autonomy stack is on the context;
* ``ensure_dunning_job`` registers one durable "dunning tick" job,
  idempotent by name, replacing a stale spec;
* the ``__dunning_tick__`` tool is registered and binds with no args
  (the scheduler's exact call shape).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from nomorals.commerce.dunning import (
    DUNNING_JOB,
    DUNNING_SPEC,
    DUNNING_TOOL,
    ensure_dunning_job,
    run_dunning_tick_once,
)
from nomorals.commerce.store import CommerceStore, default_commerce_path

T0 = 1_790_000_000.0
DAY = 86_400.0


# ── harness ──────────────────────────────────────────────────────────

def _http_ok(method, url, headers, json_body):
    return 200, {"status": "success", "reference": json_body.get("reference")}


def _http_declined(method, url, headers, json_body):
    return 402, {"message": "card declined"}


def _make_sub(store, *, now, provider="paystack", amount=50_000,
              customer="ada", http=None):
    """A subscription with one failed charge behind it (retry due +1d)."""
    gw = store.gateway(http=http, now=lambda: now)
    sub = gw.create_subscription(provider, customer, amount)
    gw.record_failed_charge(sub)
    store.save_gateway(gw)
    return sub


def _ctx(tmp_path, *, extras=None):
    return SimpleNamespace(
        settings=SimpleNamespace(home=str(tmp_path)),
        extras=dict(extras or {}),
        db=None,
    )


def _reload(tmp_path):
    """A fresh store from disk — the driver writes through its own
    store instance, so a test's store object is stale after a tick."""
    return CommerceStore(default_commerce_path(tmp_path))


class _FakeWhy:
    def __init__(self):
        self.records = []

    def record(self, what_fired, **kwargs):
        self.records.append({"what_fired": what_fired, **kwargs})
        return f"fire-{len(self.records)}"


class _FakeScheduler:
    """Minimal stand-in for the durable Scheduler (list/add/remove)."""

    def __init__(self, jobs=None):
        self.jobs = list(jobs or [])

    def list_jobs(self):
        return [dict(j) for j in self.jobs]

    def add(self, name, spec, kind, payload, **kwargs):
        job = {"id": f"job-{len(self.jobs) + 1}", "name": name,
               "spec": spec, "payload_kind": kind, "payload": payload,
               **kwargs}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


# ── the driver ───────────────────────────────────────────────────────

def test_charges_due_subscription_and_resets_retries(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    sub = _make_sub(store, now=T0, http=_http_ok)
    assert sub.failed_attempts == 1
    assert sub.next_retry_at == T0 + DAY

    out = run_dunning_tick_once(
        _ctx(tmp_path), http=_http_ok, now=lambda: T0 + DAY + 1)

    assert out["ok"] is True, out
    assert out["attempted"] == 1, out
    assert out["charged"] == 1, out
    assert out["failed"] == 0 and out["past_due"] == 0
    assert out["skipped_no_key"] == 0
    gw = _reload(tmp_path).gateway()
    (after,) = gw.subscriptions
    assert after.failed_attempts == 0
    assert after.next_retry_at == 0.0
    assert after.status == "active"
    assert len(gw.payments) == 1
    assert gw.payments[0].status == "succeeded"
    attempts = _reload(tmp_path).list_dunning_attempts()
    assert len(attempts) == 1
    assert attempts[0]["outcome"] == "charged"
    assert attempts[0]["subscription_id"] == sub.id
    assert attempts[0]["reference"] == f"{sub.id}-retry-1"


def test_failed_charge_advances_the_retry_schedule(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    _make_sub(store, now=T0, http=_http_declined)

    out = run_dunning_tick_once(
        _ctx(tmp_path), http=_http_declined, now=lambda: T0 + DAY + 1)

    assert out["ok"] is True, out
    assert out["attempted"] == 1 and out["charged"] == 0
    assert out["failed"] == 1 and out["past_due"] == 0
    gw = _reload(tmp_path).gateway()
    (after,) = gw.subscriptions
    assert after.failed_attempts == 2
    assert after.next_retry_at == T0 + DAY + 1 + 3 * DAY
    assert after.status == "active"
    attempts = _reload(tmp_path).list_dunning_attempts()
    assert attempts[0]["outcome"] == "failed_retry"
    assert "card declined" in attempts[0]["reason"]


def test_third_failed_retry_parks_subscription_past_due(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    sub = _make_sub(store, now=T0, http=_http_declined)
    # Burn retries 2 and 3 on the organ itself, so this tick is the last.
    # (Rehydrate first: the saved sub is a different object from the one
    # _make_sub returned — mutating that one would never reach the file.)
    gw = store.gateway(http=_http_declined, now=lambda: T0)
    sub = gw.subscriptions[0]
    gw.record_failed_charge(sub)
    gw.record_failed_charge(sub)
    store.save_gateway(gw)
    assert sub.failed_attempts == 3
    assert sub.next_retry_at == T0 + 7 * DAY

    out = run_dunning_tick_once(
        _ctx(tmp_path), http=_http_declined, now=lambda: T0 + 7 * DAY + 1)

    assert out["ok"] is True, out
    assert out["past_due"] == 1 and out["failed"] == 0
    gw = _reload(tmp_path).gateway()
    (after,) = gw.subscriptions
    assert after.failed_attempts == 4
    assert after.status == "past_due"
    assert after.next_retry_at == 0.0
    assert gw.dunning_due() == []
    attempts = _reload(tmp_path).list_dunning_attempts()
    assert attempts[0]["outcome"] == "past_due"


def test_missing_key_skips_without_burning_the_retry(
        tmp_path, monkeypatch):
    monkeypatch.delenv("PAYSTACK_SECRET_KEY", raising=False)
    store = CommerceStore(default_commerce_path(tmp_path))
    # The key was present at creation time; it is gone at tick time.
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    _make_sub(store, now=T0, http=_http_ok)
    monkeypatch.delenv("PAYSTACK_SECRET_KEY", raising=False)

    out = run_dunning_tick_once(
        _ctx(tmp_path), http=_http_ok, now=lambda: T0 + DAY + 1)

    assert out["ok"] is True, out
    assert out["attempted"] == 1 and out["skipped_no_key"] == 1
    assert out["charged"] == 0 and out["failed"] == 0
    gw = _reload(tmp_path).gateway()
    (after,) = gw.subscriptions
    assert after.failed_attempts == 1, "a config wall must not burn a retry"
    assert after.next_retry_at == T0 + DAY, "still due next tick"
    assert after.status == "active"
    assert gw.payments == []
    attempts = _reload(tmp_path).list_dunning_attempts()
    assert attempts[0]["outcome"] == "skipped_no_key"
    assert "PAYSTACK_SECRET_KEY" in attempts[0]["reason"]


def test_bad_amount_skips_without_burning_the_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    # Hand-edited record: create_subscription refuses amount <= 0.
    gw = store.gateway(http=_http_ok, now=lambda: T0)
    sub = gw.create_subscription("paystack", "ada", 50_000)
    sub.amount_minor = 0
    gw.record_failed_charge(sub)
    store.save_gateway(gw)

    out = run_dunning_tick_once(
        _ctx(tmp_path), http=_http_ok, now=lambda: T0 + DAY + 1)

    assert out["ok"] is True, out
    assert out["skipped_bad_record"] == 1
    gw = _reload(tmp_path).gateway()
    assert gw.subscriptions[0].failed_attempts == 1
    assert _reload(tmp_path).list_dunning_attempts()[0]["outcome"] == (
        "skipped_bad_record")


def test_nothing_due_is_a_quiet_success(tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    _make_sub(store, now=T0, http=_http_ok)

    out = run_dunning_tick_once(_ctx(tmp_path), http=_http_ok,
                                now=lambda: T0 + 60)

    assert out == {"ok": True, "attempted": 0, "charged": 0, "failed": 0,
                   "past_due": 0, "skipped_no_key": 0,
                   "skipped_bad_record": 0, "outcomes": []}
    assert store.list_dunning_attempts() == []


def test_tick_explains_itself_into_the_why_store(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    _make_sub(store, now=T0, http=_http_ok)
    why = _FakeWhy()
    stack = SimpleNamespace(why_store=why)
    ctx = _ctx(tmp_path, extras={"autonomy_stack": stack})

    out = run_dunning_tick_once(ctx, http=_http_ok,
                                now=lambda: T0 + DAY + 1)

    assert out["ok"] is True, out
    assert len(why.records) == 1
    record = why.records[0]
    assert record["what_fired"] == "dunning tick"
    assert record["evidence"]["attempted"] == 1
    assert record["evidence"]["charged"] == 1
    assert "dunning" in record["did"]


def test_tick_without_a_stack_still_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    _make_sub(store, now=T0, http=_http_ok)

    out = run_dunning_tick_once(_ctx(tmp_path), http=_http_ok,
                                now=lambda: T0 + DAY + 1)

    assert out["ok"] is True and out["charged"] == 1, out


def test_corrupt_store_is_an_honest_failure_not_a_crash(tmp_path):
    path = default_commerce_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    out = run_dunning_tick_once(_ctx(tmp_path), http=_http_ok)

    assert out["ok"] is False
    assert "commerce store" in out["error"]


# ── ensure ───────────────────────────────────────────────────────────

def test_ensure_registers_one_dunning_job_idempotently():
    sched = _FakeScheduler()
    job = ensure_dunning_job(sched)
    assert job is not None
    assert job["name"] == DUNNING_JOB
    assert job["spec"] == DUNNING_SPEC
    assert job["payload"] == {"tool": DUNNING_TOOL, "args": {}}
    again = ensure_dunning_job(sched)
    assert again["id"] == job["id"]
    assert len(sched.list_jobs()) == 1


def test_ensure_replaces_a_stale_spec():
    sched = _FakeScheduler([{"id": "job-old", "name": DUNNING_JOB,
                             "spec": "every 1h"}])
    job = ensure_dunning_job(sched)
    assert job["spec"] == DUNNING_SPEC
    assert len(sched.list_jobs()) == 1


def test_ensure_is_best_effort_when_the_scheduler_is_broken():
    class _Broken:
        def list_jobs(self):
            raise RuntimeError("no scheduler here")

    assert ensure_dunning_job(_Broken()) is None
    assert ensure_dunning_job(None) is None


# ── tool registration ────────────────────────────────────────────────

def test_dunning_tick_tool_is_registered_and_binds_no_args():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace(db=None, settings=None))
    registry.register_builtins()
    spec = registry._tools.get(DUNNING_TOOL)
    assert spec is not None, f"{DUNNING_TOOL} not in the tool registry"
    import inspect

    inspect.signature(spec.fn).bind()


def test_dunning_job_runs_end_to_end_through_the_real_scheduler(
        tmp_path, monkeypatch):
    """The full binding: ensure → scheduler → tool → driver.

    The due subscription's key is absent, so the tick takes the
    honest skip path — no HTTP client is ever built, no network.
    """
    monkeypatch.delenv("PAYSTACK_SECRET_KEY", raising=False)
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    store = CommerceStore(default_commerce_path(tmp_path))
    _make_sub(store, now=T0, http=_http_ok)
    monkeypatch.delenv("PAYSTACK_SECRET_KEY", raising=False)
    from nomorals.agents.scheduler import Scheduler
    from nomorals.storage.db import Database
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home=str(tmp_path),
                                 home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={},
        tools=None,
    )
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    ctx.tools = registry
    sched = Scheduler(ctx)

    class _NoopNotifier:
        def publish(self, kind, title, body="", **kw):
            return {"delivered": True, "channel": "test"}

    sched.notifier = _NoopNotifier()
    ensure_dunning_job(sched)
    (row,) = [j for j in sched.list_jobs() if j.get("name") == DUNNING_JOB]
    out = sched.run_now(row["id"])
    assert out["ok"] is True, out
    rows = db.query(
        "SELECT ok, result FROM schedule_runs WHERE job_id = ?",
        (row["id"],))
    assert len(rows) == 1 and rows[0]["ok"] == 1, rows
    assert "skipped_no_key" in json.dumps(rows[0]["result"]), rows
    assert _reload(tmp_path).list_dunning_attempts()[0]["outcome"] == (
        "skipped_no_key")


# ── chat surface ─────────────────────────────────────────────────────

def test_subscribe_dunning_lists_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_x")
    from nomorals.commerce.commands import handle_store_commerce

    store = CommerceStore(default_commerce_path(tmp_path))
    _make_sub(store, now=T0, http=_http_ok)
    run_dunning_tick_once(_ctx(tmp_path), http=_http_ok,
                          now=lambda: T0 + DAY + 1)

    reply = handle_store_commerce("subscribe dunning", home=tmp_path)
    assert "charged" in reply
    assert "ada" in reply


def test_subscribe_dunning_empty_is_honest(tmp_path):
    from nomorals.commerce.commands import handle_store_commerce

    reply = handle_store_commerce("subscribe dunning", home=tmp_path)
    assert "no dunning attempts yet" in reply
