"""Download swarm boot wiring: daemon pump auto-start + per-job cancel.

Pins (fake range transports only — no network):

* boot: the runtime's swarm-pool helper starts the daemon pump and a
  queued job drains to done with no /download call at all;
* resume: resume_stale re-queues a crash-orphaned running job;
* cancel: a queued job is cancelled durably and the pump never runs
  it; a running job stops at its next segment boundary and its part
  files are discarded; unknown/finished ids refuse in words;
* shutdown: stopping the pool leaves no orphan swarm-pump thread;
* lifecycle: start() calls the boot helper, stop() calls the
  teardown helper (source pin — the helpers are only reachable
  from the lifecycle).
"""
import contextlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.media import download_swarm as ds
from tests.test_download_swarm import (
    DATA,
    URL,
    FakeRangeServer,
    _handler,
    _sha,
    _swarm,
)


def _wait_until(pred, *, timeout=15.0, desc="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {desc}")


def _ctx(tmp_path, queue_path):
    return SimpleNamespace(
        extras={
            "swarm_queue_path": str(queue_path),
            "tune": SimpleNamespace(max_concurrent_downloads=2),
        })


def _runtime(tmp_path, queue_path):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = _ctx(tmp_path, queue_path)
    rt.dry_run = False
    rt._swarm_pool = None
    return rt


@contextlib.contextmanager
def _patch_pool(server):
    """Build the runtime's pool with the fake range transport."""
    real = ds.SwarmPool

    def _factory(queue, **kw):
        return real(queue, downloader=_swarm(server), **kw)

    with mock.patch("nomorals.media.download_swarm.SwarmPool",
                    _factory):
        yield


def _pump_threads():
    return [t for t in threading.enumerate()
            if t.name == "swarm-pump" and t.is_alive()]


class _SlowServer(FakeRangeServer):
    """A fake range server that drips chunks slowly, so a download
    is visibly in flight when the test cancels it."""

    def get_range_stream(self, url, start, end, headers=None):
        for chunk in super().get_range_stream(url, start, end, headers):
            time.sleep(0.02)
            yield chunk


def test_boot_starts_pump_and_drains_queued_job(tmp_path):
    """The boot helper starts the daemon pump; a queued job lands
    with no /download call at all."""
    server = FakeRangeServer(DATA)
    qpath = tmp_path / "q.json"
    queue = ds.SwarmQueue(qpath)
    dest = tmp_path / "big.mp4"
    jid = queue.enqueue(URL, dest)
    rt = _runtime(tmp_path, qpath)
    try:
        with _patch_pool(server):
            rt._start_download_swarm_pool()
            assert _pump_threads(), "no swarm-pump thread after boot"
            _wait_until(
                lambda: (queue.get(jid) or {}).get("status") == "done",
                timeout=15.0, desc="queued job drained by the pump")
            assert dest.read_bytes() == DATA
            assert _sha(dest.read_bytes()) == _sha(DATA)
    finally:
        stop = getattr(rt, "_stop_download_swarm_pool", None)
        if stop is not None:
            stop()


def test_resume_stale_revives_crash_orphaned_running_job(tmp_path):
    """A running job untouched since a crash is re-queued and then
    drains to done instead of restarting from byte 0."""
    server = FakeRangeServer(DATA)
    qpath = tmp_path / "q.json"
    queue = ds.SwarmQueue(qpath)
    dest = tmp_path / "big.mp4"
    jid = queue.enqueue(URL, dest)
    assert queue.mark(jid, "running")
    rows = json.loads(qpath.read_text(encoding="utf-8"))
    rows[0]["updated_ts"] = time.time() - 400.0
    qpath.write_text(json.dumps(rows), encoding="utf-8")
    pool = ds.SwarmPool(ds.SwarmQueue(qpath), downloader=_swarm(server))
    try:
        assert pool.resume_stale() == 1
        assert pool.queue.get(jid)["status"] == "queued"
        done = pool.process_one()
        assert done is not None and done["status"] == "done"
        assert dest.read_bytes() == DATA
    finally:
        pool.close()


def test_cancel_queued_job_is_durable_and_never_runs(tmp_path):
    """Cancelling a queued job marks it cancelled durably and the
    pump never runs it."""
    server = FakeRangeServer(DATA)
    qpath = tmp_path / "q.json"
    queue = ds.SwarmQueue(qpath)
    dest = tmp_path / "big.mp4"
    jid = queue.enqueue(URL, dest)
    assert queue.cancel(jid) == "cancelled"
    assert ds.SwarmQueue(qpath).get(jid)["status"] == "cancelled"
    pool = ds.SwarmPool(ds.SwarmQueue(qpath), downloader=_swarm(server))
    try:
        pool.start()
        time.sleep(2.0)
        assert pool.queue.get(jid)["status"] == "cancelled"
        assert server.fetched == []
        assert not dest.exists()
        assert pool.process_one() is None
    finally:
        pool.close()


def test_cancel_running_job_stops_at_segment_boundary(tmp_path):
    """Cancelling a running job stops it at the next segment
    boundary: in-flight segments finish, none start after the
    cancel, and the part files + state are discarded."""
    server = _SlowServer(DATA)
    qpath = tmp_path / "q.json"
    queue = ds.SwarmQueue(qpath)
    dest = tmp_path / "big.mp4"
    jid = queue.enqueue(URL, dest, workers=4)
    pool = ds.SwarmPool(ds.SwarmQueue(qpath), downloader=_swarm(server),
                        context=_ctx(tmp_path, qpath))
    try:
        pool.start()
        _wait_until(lambda: len(server.fetched) >= 1,
                    timeout=10.0, desc="first segment fetch in flight")
        # Cancel through a FRESH queue instance — the command layer's
        # shape, not the pool's own instance.
        assert ds.SwarmQueue(qpath).cancel(jid) == "cancelled"
        started = len(server.fetched)
        assert started >= 1
        # The status alone proves nothing — the cancel call itself
        # set it.  The state file disappearing proves the download
        # actually took the cancel path (it discards state there);
        # a failure would have kept the partial state for resume.
        _wait_until(
            lambda: not (tmp_path / "big.mp4.swarm.json").exists()
            and not dest.exists(),
            timeout=15.0, desc="cancelled download discarded its state")
        assert (pool.queue.get(jid) or {}).get("status") == "cancelled"
        assert not dest.exists()
        assert list(tmp_path.glob("big.mp4.part-*")) == []
        time.sleep(0.5)
        assert len(server.fetched) == started, \
            "a segment started after the cancel landed"
    finally:
        pool.close()


def test_cancel_unknown_and_finished_ids_refuse_in_words(tmp_path):
    """The cancel verb refuses unknown and finished ids in words,
    and resolves a queued job by its file name."""
    rt = _handler(tmp_path)
    out = rt._control_download("swarm cancel sw-nope")
    assert "no swarm job matching" in out
    server = FakeRangeServer(DATA)
    qpath = tmp_path / "q.json"
    queue = ds.SwarmQueue(qpath)
    dest = tmp_path / "big.mp4"
    jid = queue.enqueue(URL, dest)
    pool = ds.SwarmPool(ds.SwarmQueue(qpath), downloader=_swarm(server))
    try:
        done = pool.process_one()
    finally:
        pool.close()
    assert done is not None and done["status"] == "done"
    out = rt._control_download(f"swarm cancel {jid}")
    assert "already finished" in out
    jid2 = queue.enqueue(URL, tmp_path / "other.mp4")
    out = rt._control_download("swarm cancel other.mp4")
    assert "cancelled" in out
    assert queue.get(jid2)["status"] == "cancelled"


def test_stop_stops_pump_thread(tmp_path):
    """Stopping the pool leaves no orphan swarm-pump thread, and a
    second stop is a no-op."""
    server = FakeRangeServer(DATA)
    qpath = tmp_path / "q.json"
    rt = _runtime(tmp_path, qpath)
    with _patch_pool(server):
        rt._start_download_swarm_pool()
        assert _pump_threads(), "no swarm-pump thread after boot"
        rt._stop_download_swarm_pool()
        _wait_until(lambda: not _pump_threads(),
                    timeout=10.0, desc="pump thread gone after stop")
        assert rt._swarm_pool is None
        rt._stop_download_swarm_pool()


def test_start_and_stop_call_the_swarm_pool_helpers():
    """Source pin: the lifecycle actually calls the helpers — the
    boot test above drives the helper, this pins the wiring."""
    root = Path(__file__).resolve().parent.parent
    src = (root / "nomorals" / "agents" / "partner"
           / "runtime.py").read_text(encoding="utf-8")
    start = src[src.index("    def start(self)"):src.index(
        "    def _recover_interrupted_update")]
    assert "self._start_download_swarm_pool()" in start
    stop = src[src.index("    def stop(self)"):src.index(
        "    def status(self)")]
    assert "self._stop_download_swarm_pool()" in stop
