"""Orphan-bridge fix (2026-11-09) — identity-based bridge ownership.

The live incident: on the owner's EC2 host, /update died at the
quiesce step — "a bridge is listening on port 8787 but is not a
process this Devon started (it survived a restart) — Devon cannot
stop it, and it may hold its program file" — so the update stopped
BEFORE pulling and the owner was told to fix it by hand.  The
bridge WAS Devon's own; the ownership test was spawn parentage,
and a restart had thrown the process handle away.

The pins here:

1. Quiesce (stop_all_processes) stops an identity-verified
   survivor — the bridge's own status payload names its program
   and the account it serves, and the OS process image agrees —
   exactly as it stops a self-started bridge.
2. A listener whose identity does NOT verify is a stranger: the
   quiesce fails closed with the precise reason and NEVER signals
   the process.  Wrong account, no program identity, no binding —
   all refuse, none kill.
3. Adoption: reconcile / start / ensure_running take a verified
   survivor into supervision (tracked, stoppable) instead of
   leaving it unmanaged.  The owner's sticky stop beats adoption:
   a stopped account's survivor is STOPPED by the sweep, never
   resurrected — and an unverifiable listener on a stopped
   account is left untouched.
4. Real shape: the committed Go bridge binary, started OUTSIDE
   the supervisor on the account's port, is adopted and stopped
   through the real OS process inspector (no mocks).
5. The /update announcement is exactly-once: one living card per
   operation — redelivery of the same command message reuses it,
   and the failed outcome closes it with a single revision (the
   incident's twin identical "Devon restart" messages).
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ACCT = "whatsapp:2348012345678"
DIGITS = "2348012345678"
GO_PROGRAM = "gobridge/1.0.0"


class _FakeProc:
    """A spawned bridge handle for paths that never spawn here."""

    def __init__(self):
        self._poll = None
        self.stdout = iter(())

    def poll(self):
        return self._poll

    def terminate(self):
        self._poll = 0

    def kill(self):
        self._poll = 0


class _FakeBridge:
    """A real TCP listener standing on the account's port, answering
    the bridge protocol's status request with a staged payload —
    or with garbage, like any foreign server would."""

    def __init__(self, payload):
        self.payload = payload
        self._stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.sock.settimeout(0.25)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                conn.settimeout(2.0)
                conn.recv(4096)
                if isinstance(self.payload, bytes):
                    conn.sendall(self.payload)
                elif self.payload is not None:
                    conn.sendall(
                        (json.dumps(self.payload) + "\n").encode())
            except OSError:
                pass
            finally:
                conn.close()

    def close(self):
        self._stop.set()
        with contextlib.suppress(OSError):
            self.sock.close()
        self.thread.join(timeout=2)


class _FakeInspector:
    """The staged OS view: which pid listens, what it is, whether
    signals land.  Records every signal so tests can prove a
    stranger is never touched."""

    def __init__(self):
        self.pids: dict[int, int] = {}
        self.images: dict[int, tuple[str, str] | None] = {}
        self.alive: set[int] = set()
        self.stubborn: set[int] = set()
        self.unkillable: set[int] = set()
        self.terminated: list[int] = []
        self.killed: list[int] = []

    def listener_pid(self, port):
        pid = self.pids.get(port)
        return pid if pid in self.alive else None

    def process_image(self, pid):
        return self.images.get(pid)

    def pid_alive(self, pid):
        return pid in self.alive

    def terminate_pid(self, pid):
        self.terminated.append(pid)
        if pid not in self.stubborn:
            self.alive.discard(pid)
        return True

    def kill_pid(self, pid):
        self.killed.append(pid)
        if pid not in self.unkillable:
            self.alive.discard(pid)
        return True


def _stage(tmp, inspector, *, payload, pid=4242, exe="", cmdline="",
           status="linked"):
    """A supervisor + registry whose one account's port is served
    by a staged bridge the supervisor never spawned."""
    from nomorals.social.pairing import AccountRecord, BridgeSupervisor, PairingRegistry

    bridge = _FakeBridge(payload)
    registry = PairingRegistry(Path(tmp) / "paired.json")
    spawns: list = []
    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=Path(tmp) / "sessions",
        spawn=lambda argv, env: spawns.append(argv) or _FakeProc(),
        binary_resolver=lambda: "/staged/whatsapp-bridge",
        process_inspector=inspector)
    creds = str(Path(tmp) / "sessions" / f"wa-{DIGITS}")
    rec = AccountRecord(
        id=ACCT, platform="whatsapp", identity=DIGITS,
        status=status, port=bridge.port, creds_dir=creds)
    registry.upsert(rec)
    if pid is not None:
        inspector.pids[bridge.port] = pid
        inspector.alive.add(pid)
        inspector.images[pid] = (
            exe or "/staged/bridge/bin/whatsapp-bridge-linux",
            cmdline or (f"/staged/bridge/bin/whatsapp-bridge-linux "
                        f"--port {bridge.port} --creds {creds} "
                        f"--media {creds}-media"))
    return sup, registry, rec, bridge, spawns


def _verified_payload(port, **over):
    payload = {"type": "status", "program": GO_PROGRAM, "port": port,
               "state": "open", "user": f"{DIGITS}@s.whatsapp.net",
               "lid": ""}
    payload.update(over)
    return payload


class TestQuiesceIdentity(unittest.TestCase):
    """stop_all_processes: identity decides, not parentage."""

    def test_verified_survivor_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(bridge.port)
            try:
                ok, detail = sup.stop_all_processes()
            finally:
                bridge.close()
            self.assertTrue(ok, detail)
            self.assertIn("surviving bridge", detail)
            self.assertEqual(insp.terminated, [4242])
            self.assertNotIn(4242, insp.alive)

    def test_wrong_account_bridge_blocks_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(
                bridge.port, user="15551234567@s.whatsapp.net")
            try:
                ok, detail = sup.stop_all_processes()
            finally:
                bridge.close()
            self.assertFalse(ok)
            self.assertIn(ACCT, detail)
            self.assertIn("not a process this Devon started", detail)
            self.assertIn("different account", detail)
            self.assertEqual(insp.terminated, [])
            self.assertEqual(insp.killed, [])
            self.assertIn(4242, insp.alive)

    def test_foreign_server_blocks_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=b"HTTP/1.1 200 OK\r\n\r\n",
                exe="/usr/bin/python3", cmdline="python3 -m http.server")
            try:
                ok, detail = sup.stop_all_processes()
            finally:
                bridge.close()
            self.assertFalse(ok)
            self.assertIn("not a process this Devon started", detail)
            self.assertEqual(insp.terminated, [])
            self.assertEqual(insp.killed, [])
            self.assertIn(4242, insp.alive)

    def test_legacy_bridge_verified_by_process_image(self):
        """A pre-marker bridge (status payload without a program
        identity) still verifies when the account matches AND the
        listening process IS her bridge binary."""
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            payload = _verified_payload(bridge.port)
            del payload["program"]
            bridge.payload = payload
            try:
                ok, detail = sup.stop_all_processes()
            finally:
                bridge.close()
            self.assertTrue(ok, detail)
            self.assertEqual(insp.terminated, [4242])

    def test_markerless_unreadable_image_blocks(self):
        """Account matches but neither the payload nor the process
        proves the program is hers — fail closed, never signal."""
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None, exe="/usr/bin/python3",
                cmdline="python3 /tmp/impostor.py")
            payload = _verified_payload(bridge.port)
            del payload["program"]
            bridge.payload = payload
            try:
                ok, detail = sup.stop_all_processes()
            finally:
                bridge.close()
            self.assertFalse(ok)
            self.assertIn("no program identity", detail)
            self.assertEqual(insp.terminated, [])
            self.assertIn(4242, insp.alive)

    def test_unlinked_bridge_binds_by_cmdline(self):
        """Pairing in progress (no account served yet): the marker
        plus a cmdline binding this account's port + creds dir is
        identity enough."""
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(
                bridge.port, user="", state="closed")
            try:
                ok, detail = sup.stop_all_processes()
            finally:
                bridge.close()
            self.assertTrue(ok, detail)
            self.assertEqual(insp.terminated, [4242])

    def test_unlinked_bridge_without_binding_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None,
                cmdline=("/staged/bridge/bin/whatsapp-bridge-linux "
                         f"--port {0} --creds /elsewhere/wa-999 "
                         "--media /elsewhere/media"))
            bridge.payload = _verified_payload(
                bridge.port, user="", state="closed")
            try:
                ok, detail = sup.stop_all_processes()
            finally:
                bridge.close()
            self.assertFalse(ok)
            self.assertIn("not a process this Devon started", detail)
            self.assertEqual(insp.terminated, [])
            self.assertIn(4242, insp.alive)

    def test_stubborn_survivor_escalates_to_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            insp.stubborn.add(4242)
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(bridge.port)
            try:
                ok, detail = sup.stop_all_processes(timeout=0.3)
            finally:
                bridge.close()
            self.assertTrue(ok, detail)
            self.assertEqual(insp.terminated, [4242])
            self.assertEqual(insp.killed, [4242])
            self.assertNotIn(4242, insp.alive)

    def test_unkillable_survivor_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            insp.stubborn.add(4242)
            insp.unkillable.add(4242)
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(bridge.port)
            try:
                ok, detail = sup.stop_all_processes(timeout=0.3)
            finally:
                bridge.close()
            self.assertFalse(ok)
            self.assertIn("still running", detail)
            self.assertIn(4242, insp.alive)


class TestAdoption(unittest.TestCase):
    """A verified survivor joins supervision; the owner's stop
    always wins over reconcile."""

    def test_adopted_survivor_is_tracked_and_stoppable(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, registry, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(bridge.port)
            try:
                ident = sup.adopt(_rec)
                self.assertIsNotNone(ident)
                self.assertTrue(ident.verified)
                self.assertEqual(ident.pid, 4242)
                self.assertTrue(sup.is_running(ACCT))
                self.assertEqual(sup.status_all()[0]["process"],
                                 "running")
                # The owner's stop reaches the adopted orphan too.
                self.assertTrue(sup.stop(ACCT))
            finally:
                bridge.close()
            self.assertEqual(insp.terminated, [4242])
            self.assertEqual(registry.get(ACCT).status, "stopped")

    def test_start_adopts_instead_of_double_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, rec, bridge, spawns = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(bridge.port)
            try:
                out = sup.start(ACCT)
            finally:
                bridge.close()
            self.assertIs(out, rec)
            self.assertEqual(spawns, [])
            self.assertTrue(sup.is_running(ACCT))

    def test_ensure_running_adopts(self):
        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, _reg, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(bridge.port)
            try:
                ok, why = sup.ensure_running(ACCT)
            finally:
                bridge.close()
            self.assertTrue(ok, why)
            self.assertTrue(sup.is_running(ACCT))

    def test_sweep_adopts_linked_survivor(self):
        from nomorals.social.pairing import PoolSupervision

        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, registry, _rec, bridge, _s = _stage(
                tmp, insp, payload=None)
            bridge.payload = _verified_payload(bridge.port)
            pool = PoolSupervision(registry, supervisor=sup)
            try:
                pool.sweep()
            finally:
                bridge.close()
            self.assertTrue(sup.is_running(ACCT))
            self.assertEqual(registry.get(ACCT).status, "linked")

    def test_sticky_stop_beats_adoption(self):
        """An owner-stopped account whose bridge survived is
        STOPPED by the sweep — never adopted, never resurrected
        (personal-surface live gate)."""
        from nomorals.social.pairing import PoolSupervision

        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, registry, _rec, bridge, _s = _stage(
                tmp, insp, payload=None, status="stopped")
            bridge.payload = _verified_payload(bridge.port)
            pool = PoolSupervision(registry, supervisor=sup)
            try:
                changes = pool.sweep()
            finally:
                bridge.close()
            self.assertEqual(insp.terminated, [4242])
            self.assertNotIn(4242, insp.alive)
            self.assertFalse(sup.is_running(ACCT))
            self.assertEqual(registry.get(ACCT).status, "stopped")
            self.assertTrue(
                any("surviving bridge stopped" in str(c.get("change"))
                    for c in changes),
                changes)
            ok, why = sup.ensure_running(ACCT)
            self.assertFalse(ok)
            self.assertIn("stopped", why)

    def test_stopped_account_stranger_left_alone(self):
        from nomorals.social.pairing import PoolSupervision

        with tempfile.TemporaryDirectory() as tmp:
            insp = _FakeInspector()
            sup, registry, _rec, bridge, _s = _stage(
                tmp, insp, payload=b"HTTP/1.1 200 OK\r\n\r\n",
                status="stopped", exe="/usr/bin/python3",
                cmdline="python3 -m http.server")
            pool = PoolSupervision(registry, supervisor=sup)
            try:
                pool.sweep()
            finally:
                bridge.close()
            self.assertEqual(insp.terminated, [])
            self.assertEqual(insp.killed, [])
            self.assertIn(4242, insp.alive)
            self.assertEqual(registry.get(ACCT).status, "stopped")


_REPO = Path(__file__).resolve().parents[1]
_GO_BIN = _REPO / "bridge" / "bin" / "whatsapp-bridge-linux"


@unittest.skipUnless(
    sys.platform.startswith("linux") and _GO_BIN.exists(),
    "needs the committed linux Go bridge binary")
class TestRealBridgeProcess(unittest.TestCase):
    """REAL shapes: the actual Go bridge binary, started outside
    the supervisor, adopted and stopped through the real OS
    process inspector (/proc) — the EC2 orphan's exact shape."""

    def _free_port(self) -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_real_orphan_adopted_and_quiesced(self):
        from nomorals.social.pairing import AccountRecord, BridgeSupervisor, PairingRegistry

        with tempfile.TemporaryDirectory() as tmp:
            port = self._free_port()
            creds = Path(tmp) / "sessions" / f"wa-{DIGITS}"
            creds.mkdir(parents=True)
            # Held for the child's lifetime, closed in the finally.
            log = open(Path(tmp) / "orphan-bridge.log", "wb")  # noqa: SIM115
            proc = subprocess.Popen(
                [str(_GO_BIN), "--port", str(port),
                 "--creds", str(creds), "--media", str(creds) + "-media"],
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                deadline = time.time() + 15
                while time.time() < deadline:
                    try:
                        with socket.create_connection(
                                ("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        if proc.poll() is not None:
                            self.fail("real bridge exited early")
                        time.sleep(0.2)
                else:
                    self.fail("real bridge never listened")

                registry = PairingRegistry(Path(tmp) / "paired.json")
                sup = BridgeSupervisor(   # NO inspector injected:
                    registry,               # the real /proc view.
                    bridge_path="/repo/bridge/whatsapp-bridge.mjs",
                    sessions_root=Path(tmp) / "sessions",
                    spawn=lambda argv, env: _FakeProc(),
                    binary_resolver=lambda: str(_GO_BIN))
                rec = AccountRecord(
                    id=ACCT, platform="whatsapp", identity=DIGITS,
                    status="linked", port=port, creds_dir=str(creds))
                registry.upsert(rec)

                ident = sup.adopt(rec)
                self.assertIsNotNone(ident)
                self.assertTrue(ident.verified)
                self.assertEqual(ident.pid, proc.pid)
                self.assertIn("process", ident.via)
                self.assertTrue(sup.is_running(ACCT))

                ok, detail = sup.stop_all_processes()
                self.assertTrue(ok, detail)
                for _ in range(50):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                self.assertIsNotNone(proc.poll(),
                                     "the orphan bridge is still alive")
                self.assertFalse(sup._port_listening(rec))
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                log.close()


# ── announcement exactly-once ────────────────────────────────────────

SHA_A = "a" * 40
SHA_B = "b" * 40


def _failed_update_report():
    from nomorals.os.update import UpdateReport

    rep = UpdateReport(ok=False, from_sha=SHA_A, seconds=7.1)
    rep.record("disk_space", True, "free 11.8GB, needs ~472.6MB")
    rep.record("quiesce", False,
               "bridge processes still running: "
               "whatsapp:2348012345678: a bridge is listening but "
               "is not a process this Devon started")
    rep.error = "update stopped before pulling"
    return rep


class TestAnnouncementOnce(unittest.TestCase):
    """The incident's twin: ONE /update delivered the "Devon
    restart" card twice as two identical messages.  One operation,
    one living card — published once, advanced by edits."""

    def _make_runtime(self):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner.runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        sends: list[str] = []
        edits: list[str] = []

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")
                self.name = "telegram-personal"

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to="", buttons=None,
                     **kw):
                sends.append(text)
                return SendResult(ok=True, platform=self.name,
                                  message_id="card-1")

            def edit_message(self, chat, message_id, text, **kw):
                edits.append(text)
                return {"ok": True}

        tmp = tempfile.TemporaryDirectory(prefix="nm-test-orphanfix-")
        settings = load_settings(
            overrides={"home": tmp.name,
                       "partner.platforms": "telegram-personal"})
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        adapter = FakeAdapter()
        gateway = ChatGateway({"telegram-personal": adapter},
                              db=context.db)
        runtime = PartnerRuntime(context, gateway=gateway)
        return runtime, context, tmp, sends, edits

    def _msg(self, message_id: str):
        from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

        return ChatMessage(
            chat=ChatRef(platform="telegram-personal", chat_id="c1",
                         kind=ChatKind.DM, peer="someone"),
            incoming=True, text="/update", sender="someone",
            message_id=message_id, meta={"is_owner": True})

    def _run_update(self, runtime, message):
        class FakeUpdater:
            def __init__(self, repo, snapshots, quiesce=None):
                pass

            def run(self, pull=True):
                return _failed_update_report()

        with mock.patch("nomorals.os.update.UpdateManager",
                        FakeUpdater):
            return runtime.handle_control(
                "/update", "telegram-personal:c1", message=message)

    def _drive(self, messages):
        runtime, context, tmp, sends, edits = self._make_runtime()
        try:
            replies = [self._run_update(runtime, m) for m in messages]
        finally:
            with contextlib.suppress(Exception):
                runtime.stop()
            with contextlib.suppress(Exception):
                context.close()
            tmp.cleanup()
        cards = [t for t in sends if "Devon restart" in t]
        return replies, cards, edits

    def test_failed_update_publishes_one_card(self):
        replies, cards, edits = self._drive([self._msg("m-1")])
        self.assertIn("update failed", replies[0])
        self.assertEqual(len(cards), 1, cards)
        self.assertTrue(edits, "stages advance by editing the card")

    def test_redelivered_command_reuses_the_card(self):
        msg = self._msg("m-1")
        _replies, cards, _edits = self._drive([msg, msg])
        self.assertEqual(len(cards), 1, cards)

    def test_new_command_opens_its_own_card(self):
        _replies, cards, _edits = self._drive(
            [self._msg("m-1"), self._msg("m-2")])
        self.assertEqual(len(cards), 2, cards)


if __name__ == "__main__":
    unittest.main()
