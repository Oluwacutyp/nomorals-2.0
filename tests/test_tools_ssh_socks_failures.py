"""Pins for nomorals/tools/ssh_socks.py (tail-116): behavior + NOT-recording.

Sites pinned:

* 226 (exclusion) — ``_supervisor``: a raising ``proc.wait()`` ends
  the supervisor loop cleanly instead of killing the thread with a
  traceback. Nothing fabricates liveness afterwards: ``alive()`` and
  ``status()`` keep deriving from the real process handle, so the
  degradation is absence-shaped (no further auto-reconnect), never a
  false "running".
* 282 (exclusion) — ``stop()`` kill ladder, first rung: killpg
  failure (incl. Windows' missing killpg) falls back to
  ``proc.terminate()``; a failing terminate is the already-gone case.
* 295 (exclusion) — ``stop()`` kill ladder, last rung: a process
  still unwaitable after SIGKILL leaves ``stop()`` reporting its
  printed contract (handle released, ``{"stopped": True, "pid"}``);
  the pid was SIGTERMed then SIGKILLed before this point.
* 330 (exclusion in-batch; DATA-LOSS SHELF) — ``_load_json``: a
  crashing KV read yields the caller's default. For the profiles key
  that default is load-bearing: the manager loads the whole profile
  set at init and every ``upsert_profile``/``remove`` rewrites the
  whole set, so a transiently crashed init load makes the next
  routine mutation overwrite every stored profile (byte-proven in
  ``test_crashed_load_then_upsert_rewrites_whole_set``). The honest
  repair — distinguish "unreadable" from "empty" before a wholesale
  rewrite may proceed — changes ``_load_json``'s contract for every
  caller, so it belongs to the targeted data-loss pass (alerts-162 /
  goals-93 / trust-152 family), not a batch convert.

NOT-recording: none of these guards record or log; ``ssh_socks._log``
is patched with a MagicMock around the supervisor/stop triggers and
``method_calls`` must stay empty. This file contains NO except
handlers: triggers are mock side_effects raised inside the target's
own handlers.
"""
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.storage.db import Database
from nomorals.storage.kv import KVStore
from nomorals.storage.migrations import MIGRATIONS
from nomorals.storage.schema import MigrationRunner
from nomorals.tools import ssh_socks as ss
from nomorals.tools.ssh_socks import (
    SshSocksManager,
    SshSocksProfile,
    SshSocksTunnel,
)


def _db():
    db = Database(":memory:")
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


def _profile(name="t1", **kw):
    return SshSocksProfile(name=name, host="example.invalid", user="u", **kw)


class _WaitBoom:
    pid = 4242

    def wait(self, timeout=None):
        raise RuntimeError("wait exploded")

    def poll(self):
        return 0


class TestSupervisor226(unittest.TestCase):
    def test_wait_crash_ends_supervisor_cleanly(self):
        tun = SshSocksTunnel(_profile(max_reconnects=0))
        tun.proc = _WaitBoom()
        with mock.patch.object(ss, "_log") as log:
            result = tun._supervisor()
        self.assertIsNone(result)
        self.assertEqual(log.method_calls, [])

    def test_status_stays_honest_after_supervisor_death(self):
        tun = SshSocksTunnel(_profile(max_reconnects=0))
        tun.proc = _WaitBoom()
        tun._supervisor()
        # liveness derives from the process handle, not the supervisor
        self.assertFalse(tun.alive())
        self.assertEqual(tun.status()["pid"], 0)


class _StubbornProc:
    pid = 4243

    def __init__(self):
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("ssh", timeout)


class TestStopLadder282and295(unittest.TestCase):
    def test_killpg_failure_falls_back_and_stop_reports(self):
        tun = SshSocksTunnel(_profile())
        proc = _StubbornProc()
        tun.proc = proc

        def _killpg_boom(pgid, sig):
            raise OSError("no process group")

        with mock.patch.object(ss.os, "killpg", _killpg_boom), \
                mock.patch.object(ss.os, "getpgid", lambda pid: pid), \
                mock.patch.object(ss, "_log") as log:
            result = tun.stop()
        self.assertEqual(proc.terminated, 1)   # 282 fallback rung ran
        self.assertEqual(proc.killed, 1)       # SIGKILL rung ran
        self.assertEqual(result, {"stopped": True, "pid": 4243})
        self.assertEqual(log.method_calls, [])


class TestLoadJson330(unittest.TestCase):
    def test_healthy_load_control(self):
        db = _db()
        seed = {"old": _profile("old").to_dict()}
        KVStore(db).set_raw(ss._PROFILES_KEY, json.dumps(seed), "json")
        mgr = SshSocksManager(SimpleNamespace(db=db))
        self.assertEqual(set(mgr._profiles), {"old"})

    def test_crashed_load_yields_default(self):
        db = _db()
        seed = {"old": _profile("old").to_dict()}
        KVStore(db).set_raw(ss._PROFILES_KEY, json.dumps(seed), "json")
        with mock.patch.object(KVStore, "get",
                               side_effect=RuntimeError("kv read dead")):
            mgr = SshSocksManager(SimpleNamespace(db=db))
        self.assertEqual(mgr._profiles, {})

    def test_crashed_load_then_upsert_rewrites_whole_set(self):
        # The shelf shape, byte-proven: the seeded profile is destroyed
        # by one routine upsert after a transiently crashed init load.
        db = _db()
        seed = {"old": _profile("old").to_dict()}
        KVStore(db).set_raw(ss._PROFILES_KEY, json.dumps(seed), "json")
        with mock.patch.object(KVStore, "get",
                               side_effect=RuntimeError("kv read dead")):
            mgr = SshSocksManager(SimpleNamespace(db=db))
        mgr.upsert_profile(name="new", host="new.invalid", user="y")
        after = KVStore(db).get(ss._PROFILES_KEY)
        self.assertEqual(set(after), {"new"})


if __name__ == "__main__":
    unittest.main()
