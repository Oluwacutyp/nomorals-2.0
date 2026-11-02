"""Tail-90 pins: nomorals/llm/learning.py census sites.

The module's design law is written at its head: "Best-effort,
always. If the trajectory DB is missing, corrupt, or
unwritable, the attachment degrades to a no-op and the router
behaves exactly as before (logged, never raised)." Every site
sits inside that law; all are held as exclusions:

74 — atexit hook: ``list(_live_attachments)`` crash -> return.
If the WeakSet registry itself cannot be iterated during
interpreter teardown, no worker is stopped — the tension is
stated plainly: the segfault risk the hook exists to prevent
returns for every attachment. Held: any retry iterates the
same corrupted registry mid-teardown; there is no
within-contract second channel, and nothing consumes a return
value from an atexit hook.
79 — atexit hook: one attachment's ``_stop.set()`` crash ->
pass. Per-attachment isolation: one broken attachment cannot
prevent the rest being signalled (probe: the sibling's stop
IS set).
87 — atexit hook: worker ``join`` crash -> pass. Teardown
must never raise and must never hang; the stop signal from
the first loop has already been delivered.
246 — ``classify_failure`` crash -> ``failure_class = ""``.
The typed class is explicitly a bonus ("the class is a
bonus" at the sibling guard in trajectories.record): the
trajectory row is still recorded with its success/latency/
error intact; only failure-cluster grouping misses the kind.
300 — ``router.broker`` read crash -> treated as absent, a
broker is built and attached. WEAKEST EXCLUSION of the
batch, tension stated plainly: None is byte-identical to
"no broker attached" on a decision-consumed surface (the
broker steers provider selection), and a falsely-absent read
makes ``set_broker`` REPLACE the router's broker for the
rest of the process. Held: the production LLMRouter getter
is a plain locked ``return self._broker`` — it cannot raise
while a broker exists short of interpreter teardown; the
raising getters that exist are duck-typed fakes with no
broker to lose; and the outcome is recovery (a working
broker, recording attached), not a false success claim.
Narrowing the guard to AttributeError-only is shelved for
parent adjudication (it changes duck-typed-router behavior
estate-wide).
"""
from __future__ import annotations

from types import SimpleNamespace

import nomorals.llm.failures as failures_mod
from nomorals.agency import failures as agency_failures
from nomorals.cognition.trajectories import TrajectoryStore
from nomorals.llm import learning as L
from nomorals.llm.benchmarks import BenchmarkDB
from nomorals.llm.broker import ModelBroker
from nomorals.storage.db import Database


def _not_recorded(before: int) -> None:
    assert agency_failures.failure_count() == before


def _boom(*a, **k):
    raise RuntimeError("probe boom")


class _FakeStop:
    def __init__(self, raises=False):
        self.raises = raises
        self.set_calls = 0

    def set(self):
        self.set_calls += 1
        if self.raises:
            raise RuntimeError("stop boom")


class _FakeWorker:
    def __init__(self, join_raises=False):
        self.join_raises = join_raises
        self.join_calls = 0

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_calls += 1
        if self.join_raises:
            raise RuntimeError("join boom")


class _FakeAttachment:
    def __init__(self, stop_raises=False, join_raises=False):
        self._stop = _FakeStop(stop_raises)
        self._worker = _FakeWorker(join_raises)


# ── 74: registry iteration crash -> quiet return ────────────────

class _BoomSet:
    def __iter__(self):
        raise RuntimeError("weakset boom")


def test_74_registry_iteration_crash_returns_quietly(monkeypatch):
    before = agency_failures.failure_count()
    monkeypatch.setattr(L, "_live_attachments", _BoomSet())
    assert L._stop_all_learning_workers() is None
    _not_recorded(before)


# ── 79: one bad _stop.set cannot block the others ───────────────

def test_79_stop_crash_isolated_per_attachment():
    before = agency_failures.failure_count()
    bad = _FakeAttachment(stop_raises=True)
    good = _FakeAttachment()
    L._live_attachments.add(bad)
    L._live_attachments.add(good)
    assert L._stop_all_learning_workers() is None
    assert good._stop.set_calls == 1
    _not_recorded(before)


# ── 87: join crash cannot break teardown ────────────────────────

def test_87_join_crash_returns_quietly():
    before = agency_failures.failure_count()
    att = _FakeAttachment(join_raises=True)
    L._live_attachments.add(att)
    assert L._stop_all_learning_workers() is None
    assert att._stop.set_calls == 1
    assert att._worker.join_calls == 1
    _not_recorded(before)


# ── 246: classify crash -> row recorded with empty class ────────

class _FakeRouter:
    def set_learning(self, fn):
        self.learning = fn


def test_246_classify_crash_records_row_with_empty_class(monkeypatch):
    before = agency_failures.failure_count()
    db = Database(":memory:")
    att = L.LearningAttachment(_FakeRouter(), TrajectoryStore(db),
                               BenchmarkDB(db), None)
    try:
        monkeypatch.setattr(failures_mod, "classify_failure", _boom)
        att._write(("chat", "prov-x", False, 0.25,
                    "rate limited by upstream"))
        rows = db.query(
            "SELECT failure_class, success FROM cog_trajectories")
        assert len(rows) == 1
        assert rows[0]["failure_class"] == ""
        assert rows[0]["success"] == 0
        # control: classifier restored -> the kind lands
        monkeypatch.undo()
        att._write(("chat", "prov-x", False, 0.25,
                    "rate limited by upstream"))
        rows = db.query(
            "SELECT failure_class FROM cog_trajectories ORDER BY rowid")
        assert rows[1]["failure_class"] == "rate_limited"
    finally:
        att.detach()
    _not_recorded(before)


# ── 300: broker getter crash -> broker built and attached ───────

class _RaisingBrokerRouter:
    def __init__(self):
        self.set_calls = []

    @property
    def broker(self):
        raise RuntimeError("broker getter boom")

    def providers(self):
        return []

    def get(self, name):
        return None

    def set_broker(self, broker):
        self.set_calls.append(broker)


def test_300_broker_getter_crash_builds_and_attaches():
    before = agency_failures.failure_count()
    db = Database(":memory:")
    ts, bd = TrajectoryStore(db), BenchmarkDB(db)
    router = _RaisingBrokerRouter()
    built = L._ensure_broker(router, ts, bd)
    assert isinstance(built, ModelBroker)
    assert router.set_calls == [built]
    # control: an existing broker is kept as-is; the trajectory
    # store is injected only because it was missing
    existing = SimpleNamespace(trajectories=None)
    out = L._ensure_broker(SimpleNamespace(broker=existing), ts, bd)
    assert out is existing
    assert existing.trajectories is ts
    _not_recorded(before)
