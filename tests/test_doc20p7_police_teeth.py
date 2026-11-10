"""Doc 20 P7 — police teeth, own-process scope only.

Before this pass the police's enforcement rungs were honest but
toothless: ``Policy.revoke_grant`` took a grant OBJECT, nothing
recorded issued grants, so the police's ``revoke_fn`` was wired
nowhere and "grant revoked" graded G0/unwired; and no kill
function existed that could name a process as Devon's own.

What is pinned here:

* ISSUANCE IS RECORDED. A grant minted through the real
  production seam (``EgressToken.mint`` → ``Policy.issue_grant``)
  lands in the grant registry with its scope / subject / source /
  expiry; the record survives a reload; expiry is honored by the
  registry's checker AND the live object's own ``grants()`` —
  the two never disagree about time.
* REVOCATION BITES BY ID. The police disarm, wired with
  ``police_revoke_fn``, revokes the registry row AND the live
  grant object (the egress token starts denying immediately).
  ``Policy.revoke_issued_grant`` is the identity-based resolution
  and lands its AUDIT_REVOKE. An unknown id fails in words
  (GrantNotFound) and records nothing as performed; re-revoking
  is idempotent.
* THE TICKET READS THE REGISTRY. A performed disarm's healing
  ticket grades G5 on its effect record (P2's rule, untouched)
  and its evidence field now cites the registry's own record;
  a legacy bare-claim receipt for the same grant STILL grades
  G0 (the reader changes no grading rule) while its evidence
  cites the registry record.
* THE KILL IS FENCED. A sleeper spawned through the own-process
  registry is terminated by the police arrest, with the row and
  the event journal recording it. A process Devon did NOT spawn
  (a bare subprocess, or this very test process) is refused in
  words and left alive; a tampered start token (the
  recycled-pid shape) is refused; an unverifiable identity is
  refused; an already-exited process is refused and marked.
* THE SPAWN SEAMS REGISTER. The repair pipeline's subprocess
  runner and the sandbox runner record their children on the
  default registry and mark them exited when they end.
* THE CHAT SURFACE ENFORCES. Through ``control_immune`` with the
  teeth injected exactly as the composition callers inject them
  (the partner runtime and the nm CLI — immune/ imports no core
  module, so injection is the wiring) and the registry defaults
  redirected to temp files: a real grant disarms, a real
  registered sleeper is arrested, and unknown targets are
  NOT-performed answers in words — never success wording. A
  bare, uninjected call keeps the pre-P7 unwired posture
  byte-for-byte.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from nomorals.agents.healing_ticket import healing_ticket_for
from nomorals.core import grant_registry as grants_mod
from nomorals.core import own_processes as procs_mod
from nomorals.core.egress import EgressDenied, EgressToken
from nomorals.core.grant_registry import (
    GrantNotFound,
    GrantRegistry,
    police_revoke_fn,
)
from nomorals.core.own_processes import (
    OwnProcessRefusal,
    OwnProcessRegistry,
    pid_alive,
    police_kill_fn,
    process_start_token,
)
from nomorals.core.policy import AUDIT_REVOKE, Policy
from nomorals.immune.chat import (
    control_immune,
    get_immune_state,
    reset_immune_state,
)
from nomorals.immune.police import Police, Receipt

CHAT_KEY = "p7-suite"
SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


class _FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


# ── issuance ─────────────────────────────────────────────────────


class GrantIssuanceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "grants.json"
        self.registry = GrantRegistry(self.path)

    def test_egress_mint_registers_the_grant(self) -> None:
        token = EgressToken.mint(
            "task-p7", domains=("example.com",), ttl_s=600,
            registry=self.registry)
        row = self.registry.get(token.token_id)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["source"], "egress")
        self.assertEqual(row["subject"], "task-p7")
        self.assertEqual(row["actor"], "task-p7")
        self.assertEqual(row["patterns"], ["net.out"])
        self.assertEqual(row["state"], "active")
        self.assertEqual(
            GrantRegistry.standing(row), "active")
        self.assertAlmostEqual(
            row["expires_at"] - row["issued_at"], 600, delta=2)
        # The registry checker agrees with the live object.
        self.assertTrue(self.registry.grants(token.token_id, "net.out"))
        self.assertFalse(self.registry.grants(token.token_id, "fs.read"))
        self.assertTrue(token.grant.grants("net.out"))
        events = self.registry.events()
        self.assertEqual([e["kind"] for e in events], ["issued"])
        self.assertEqual(events[0]["grant_id"], token.token_id)

    def test_registry_record_survives_reload(self) -> None:
        token = EgressToken.mint(
            "task-p7", domains=("example.com",), ttl_s=600,
            registry=self.registry)
        reloaded = GrantRegistry(self.path)
        row = reloaded.get(token.token_id)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["subject"], "task-p7")
        self.assertTrue(reloaded.grants(token.token_id, "net.out"))

    def test_expiry_honored_by_registry_and_live_object(self) -> None:
        clock = _FakeClock()
        policy = Policy(clock=clock)
        grant = policy.issue_grant(
            "net.out", ttl_s=100, actor="task-clock",
            registry=self.registry)
        self.assertEqual(
            self.registry.get(grant.grant_id)["source"],
            "policy.issue_grant")
        self.assertTrue(grant.grants("net.out"))
        self.assertTrue(self.registry.grants(
            grant.grant_id, "net.out", now=clock.now()))
        clock.advance(101)
        # The real checker (the live object) and the registry's
        # checker move together: an expired grant authorizes
        # nothing, read either way. (The registry reads are taken
        # at the grant's own clock — the registry's standing is a
        # pure function of the record and the moment asked about.)
        self.assertFalse(grant.grants("net.out"))
        self.assertFalse(self.registry.grants(
            grant.grant_id, "net.out", now=clock.now()))
        row = self.registry.get(grant.grant_id)
        assert row is not None
        self.assertEqual(
            GrantRegistry.standing(row, now=clock.now()), "expired")

    def test_egress_token_denies_after_expiry(self) -> None:
        token = EgressToken.mint(
            "task-short", domains=("example.com",), ttl_s=0.25,
            registry=self.registry)
        time.sleep(0.35)
        with self.assertRaises(EgressDenied):
            token.check_url("https://example.com/x")
        row = self.registry.get(token.token_id)
        assert row is not None
        self.assertEqual(GrantRegistry.standing(row), "expired")


# ── revocation ───────────────────────────────────────────────────


class RevocationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = GrantRegistry(Path(self.tmp.name) / "g.json")
        self.token = EgressToken.mint(
            "task-p7", domains=("example.com",), ttl_s=600,
            registry=self.registry)
        self.police = Police(revoke_fn=police_revoke_fn(self.registry))

    def test_disarm_performs_and_bites_the_live_token(self) -> None:
        receipt = self.police.disarm(self.token.token_id)
        self.assertEqual(receipt.details["effect"], "performed")
        self.assertEqual(
            receipt.reason,
            f"capability grant {self.token.token_id} revoked")
        row = self.registry.get(self.token.token_id)
        assert row is not None
        self.assertEqual(row["state"], "revoked")
        self.assertEqual(row["revoked_by"], "immune.police")
        self.assertIsNotNone(row["revoked_at"])
        # The live object flipped too: the egress checker denies.
        self.assertTrue(self.token.grant.revoked)
        self.assertFalse(self.registry.grants(self.token.token_id,
                                              "net.out"))
        with self.assertRaises(EgressDenied):
            self.token.check_url("https://example.com/x")
        kinds = [e["kind"] for e in self.registry.events()]
        self.assertEqual(kinds, ["issued", "revoked"])

    def test_revoke_issued_grant_by_id_records_policy_audit(self) -> None:
        policy = Policy()
        row = policy.revoke_issued_grant(
            self.token.token_id, actor="owner", registry=self.registry)
        self.assertEqual(row["state"], "revoked")
        entries = policy.audit_log(kind=AUDIT_REVOKE)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["context"]["grant_id"],
                         self.token.token_id)
        self.assertEqual(entries[0]["context"]["via"], "grant_registry")

    def test_unknown_grant_disarm_fails_in_words(self) -> None:
        receipt = self.police.disarm("grt_nope")
        self.assertEqual(receipt.details["effect"], "failed")
        self.assertIn("GrantNotFound", receipt.reason)
        self.assertIn("no grant with id", receipt.reason)
        with self.assertRaises(GrantNotFound):
            Policy().revoke_issued_grant(
                "grt_nope", registry=self.registry)
        # Nothing was recorded as revoked.
        self.assertEqual(
            [e["kind"] for e in self.registry.events()], ["issued"])

    def test_double_revoke_is_idempotent(self) -> None:
        first = self.registry.revoke(self.token.token_id, actor="owner")
        self.assertFalse(first["already_revoked"])
        second = self.registry.revoke(self.token.token_id, actor="owner")
        self.assertTrue(second["already_revoked"])
        self.assertEqual(
            [e["kind"] for e in self.registry.events()],
            ["issued", "revoked"])


# ── the ticket reads the registry ────────────────────────────────


class TicketRegistryReaderTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        reset_immune_state(CHAT_KEY)
        self.state = get_immune_state(CHAT_KEY)
        self.registry = GrantRegistry(Path(self.tmp.name) / "g.json")
        self.token = EgressToken.mint(
            "task-p7", domains=("example.com",), ttl_s=600,
            registry=self.registry)

    def tearDown(self) -> None:
        reset_immune_state(CHAT_KEY)

    def _ticket(self, receipt: object, **kwargs: Any) -> dict:
        self.state._record(receipt)
        return healing_ticket_for(
            f"receipt:{receipt.id}",  # type: ignore[attr-defined]
            immune_states={CHAT_KEY: self.state}, **kwargs)

    def test_performed_disarm_grades_g5_and_cites_the_registry(
            self) -> None:
        receipt = Police(
            revoke_fn=police_revoke_fn(self.registry)).disarm(
                self.token.token_id)
        ticket = self._ticket(receipt, grant_registry=self.registry)
        self.assertEqual(ticket["claim"]["grade"], "G5")
        evidence = ticket["evidence"]
        self.assertEqual(evidence["grant_registry"]["grant_id"],
                         self.token.token_id)
        self.assertEqual(evidence["grant_registry"]["standing"],
                         "revoked")
        self.assertEqual(evidence["grant_registry"]["revoked_by"],
                         "immune.police")
        self.assertEqual(ticket["sources"]["evidence"],
                         "core.grant_registry")
        self.assertIn(f"grant:{self.token.token_id}",
                      ticket["related_refs"])

    def test_legacy_claim_still_g0_but_evidence_cites_registry(
            self) -> None:
        # The grant really was revoked (through the wired path)…
        self.registry.revoke(self.token.token_id, actor="immune.police")
        # …but THIS receipt is a pre-P2 bare claim: no effect
        # record. The grading rule is untouched — the claim grades
        # on what the receipt cites: nothing. G0. The reader adds
        # the registry's record to the evidence field only.
        legacy = Receipt(
            rung=4, action="disarm", target=self.token.token_id,
            reason=f"capability grant {self.token.token_id} revoked",
            details={"grant_id": self.token.token_id})
        ticket = self._ticket(legacy, grant_registry=self.registry)
        self.assertEqual(ticket["claim"]["grade"], "G0")
        self.assertEqual(
            ticket["evidence"]["grant_registry"]["standing"],
            "revoked")

    def test_without_a_registry_the_projection_is_the_p2_shape(
            self) -> None:
        receipt = Police(
            revoke_fn=police_revoke_fn(self.registry)).disarm(
                self.token.token_id)
        ticket = self._ticket(receipt)  # no grant_registry supplied
        self.assertEqual(ticket["claim"]["grade"], "G5")
        self.assertIn("evidence", ticket["absent"])


# ── the fenced kill ──────────────────────────────────────────────


@unittest.skipUnless(process_start_token(os.getpid()) is not None,
                     "platform exposes no process start token")
class KillFencingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "procs.json"
        self.registry = OwnProcessRegistry(self.path)
        self.held: list[subprocess.Popen] = []
        self.addCleanup(self._reap)

    def _reap(self) -> None:
        for proc in self.held:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - cleanup only
                pass

    def _spawn_sleeper(self, label: str = "p7 sleeper"):
        proc, row = self.registry.spawn(
            SLEEPER, label=label, spawned_by="test_doc20p7")
        self.held.append(proc)
        return proc, row

    def test_arrest_terminates_a_registered_sleeper(self) -> None:
        proc, row = self._spawn_sleeper()
        self.assertTrue(pid_alive(row["pid"]))
        self.assertTrue(row["start_token"])
        receipt = Police(
            kill_fn=police_kill_fn(self.registry)).arrest(row["proc_id"])
        kill_outcome = receipt.details["effect_outcomes"][0]
        self.assertEqual(kill_outcome["fn"], "kill_fn")
        self.assertTrue(kill_outcome["ok"])
        # The outcome result rides the receipt as a bounded repr
        # (the receipt discipline) — the proc id is on it.
        self.assertIn(row["proc_id"], kill_outcome["result"])
        # The cancel half is unwired in this construction, so the
        # arrest as a whole is honestly PARTIAL — and says so.
        self.assertEqual(receipt.details["effect"], "partial")
        self.assertIn("killed", receipt.reason)
        self.assertTrue(_wait_dead(row["pid"]))
        proc.wait(timeout=5)
        stored = self.registry.get(row["proc_id"])
        assert stored is not None
        self.assertEqual(stored["state"], "killed")
        self.assertEqual(stored["killed_by"], "immune.police")
        self.assertEqual(
            [e["kind"] for e in self.registry.events()],
            ["spawned", "killed"])

    def test_kill_by_pid_resolves_the_running_row(self) -> None:
        proc, row = self._spawn_sleeper()
        outcome = self.registry.kill(str(row["pid"]), actor="owner")
        self.assertTrue(outcome["killed"])
        self.assertEqual(outcome["proc_id"], row["proc_id"])
        proc.wait(timeout=5)

    def test_unregistered_process_is_refused_and_untouched(self) -> None:
        bare = subprocess.Popen(SLEEPER)
        self.held.append(bare)
        with self.assertRaises(OwnProcessRefusal) as ctx:
            self.registry.kill(str(bare.pid))
        self.assertIn("no record of process", str(ctx.exception))
        self.assertTrue(pid_alive(bare.pid))
        # Through the police the refusal is a failed effect, in
        # words — never a claimed kill.
        receipt = Police(
            kill_fn=police_kill_fn(self.registry)).arrest(str(bare.pid))
        self.assertFalse(receipt.details["effect_outcomes"][0]["ok"])
        self.assertIn("no record of process", receipt.reason)
        self.assertTrue(pid_alive(bare.pid))

    def test_this_process_is_refused(self) -> None:
        # The test process itself is not in Devon's spawn records.
        with self.assertRaises(OwnProcessRefusal):
            self.registry.kill(str(os.getpid()))
        self.assertTrue(pid_alive(os.getpid()))

    def test_recycled_pid_token_mismatch_is_refused(self) -> None:
        proc, row = self._spawn_sleeper()
        # The recycled-pid shape: the pid now answers with a
        # different start token than the registered process had.
        self.registry._processes[row["proc_id"]]["start_token"] = "1"
        with self.assertRaises(OwnProcessRefusal) as ctx:
            self.registry.kill(row["proc_id"])
        self.assertIn("recycled", str(ctx.exception))
        self.assertTrue(pid_alive(row["pid"]))

    def test_unverifiable_identity_is_refused(self) -> None:
        proc, row = self._spawn_sleeper()
        self.registry._processes[row["proc_id"]]["start_token"] = None
        with self.assertRaises(OwnProcessRefusal) as ctx:
            self.registry.kill(row["proc_id"])
        self.assertIn("unverifiable", str(ctx.exception))
        self.assertTrue(pid_alive(row["pid"]))

    def test_already_exited_is_refused_and_marked(self) -> None:
        proc, row = self.registry.spawn(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            label="p7 short sleeper", spawned_by="test_doc20p7")
        proc.wait(timeout=5)
        self.assertTrue(_wait_dead(row["pid"]))
        with self.assertRaises(OwnProcessRefusal) as ctx:
            self.registry.kill(row["proc_id"])
        self.assertIn("already exited", str(ctx.exception))
        stored = self.registry.get(row["proc_id"])
        assert stored is not None
        self.assertEqual(stored["state"], "exited")

    def test_double_kill_is_refused(self) -> None:
        proc, row = self._spawn_sleeper()
        self.registry.kill(row["proc_id"], actor="owner")
        proc.wait(timeout=5)
        with self.assertRaises(OwnProcessRefusal) as ctx:
            self.registry.kill(row["proc_id"])
        self.assertIn("already recorded killed", str(ctx.exception))

    def test_rows_survive_reload(self) -> None:
        proc, row = self._spawn_sleeper()
        self.registry.kill(row["proc_id"], actor="owner")
        proc.wait(timeout=5)
        reloaded = OwnProcessRegistry(self.path)
        stored = reloaded.get(row["proc_id"])
        assert stored is not None
        self.assertEqual(stored["state"], "killed")
        self.assertEqual(stored["pid"], row["pid"])
        self.assertEqual(stored["start_token"], row["start_token"])


# ── the spawn seams register ─────────────────────────────────────


class SpawnSeamTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.procs_path = Path(self.tmp.name) / "own-processes.json"
        patcher = mock.patch.object(
            procs_mod, "default_processes_path",
            lambda: self.procs_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        procs_mod.reset_shared_process_registries()
        self.addCleanup(procs_mod.reset_shared_process_registries)

    def _rows(self) -> list[dict]:
        return procs_mod.default_own_process_registry().all()

    def test_repair_pipeline_runner_registers_its_child(self) -> None:
        from nomorals.agents.repair_pipeline import _subprocess_runner

        rc, out, _err = _subprocess_runner(
            [sys.executable, "-c", "print('p7 seam')"],
            cwd=Path(self.tmp.name), timeout=60)
        self.assertEqual(rc, 0)
        self.assertIn("p7 seam", out)
        rows = [r for r in self._rows()
                if r["spawned_by"] == "repair_pipeline"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "exited")
        self.assertEqual(rows[0]["label"], "repair pipeline command")

    def test_sandbox_runner_registers_its_child(self) -> None:
        from nomorals.tools.shell import run_sandboxed

        result = run_sandboxed(
            "true", cwd=self.tmp.name, timeout=60, backend="rlimit")
        self.assertEqual(result["exit_code"], 0)
        rows = [r for r in self._rows()
                if r["spawned_by"] == "tools.shell.run_sandboxed"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "exited")
        self.assertEqual(rows[0]["label"], "sandboxed command")


# ── the chat surface ─────────────────────────────────────────────


class ChatSurfaceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        reset_immune_state(CHAT_KEY)
        self.addCleanup(reset_immune_state, CHAT_KEY)
        grants_path = Path(self.tmp.name) / "grant-registry.json"
        procs_path = Path(self.tmp.name) / "own-processes.json"
        for patcher in (
                mock.patch.object(
                    grants_mod, "default_grants_path",
                    lambda: grants_path),
                mock.patch.object(
                    procs_mod, "default_processes_path",
                    lambda: procs_path)):
            patcher.start()
            self.addCleanup(patcher.stop)
        grants_mod.reset_shared_grant_registries()
        procs_mod.reset_shared_process_registries()
        self.addCleanup(grants_mod.reset_shared_grant_registries)
        self.addCleanup(procs_mod.reset_shared_process_registries)
        self.held: list[subprocess.Popen] = []
        self.addCleanup(self._reap)

    def _reap(self) -> None:
        for proc in self.held:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - cleanup only
                pass

    def _control(self, tail: str) -> str:
        """The production call shape: the composition callers
        (partner runtime / nm CLI) inject the police teeth into
        control_immune — these tests call it exactly as they do."""
        return control_immune(
            tail, chat_key=CHAT_KEY,
            revoke_fn=police_revoke_fn(), kill_fn=police_kill_fn())

    def test_disarm_a_real_grant_through_chat(self) -> None:
        from nomorals.core.grant_registry import default_grant_registry

        token = EgressToken.mint(
            "task-chat", domains=("example.com",), ttl_s=600)
        reply = self._control(f"disarm {token.token_id}")
        self.assertTrue(reply.startswith("disarmed:"), reply)
        self.assertTrue(token.grant.revoked)
        row = default_grant_registry().get(token.token_id)
        assert row is not None
        self.assertEqual(row["state"], "revoked")
        state = get_immune_state(CHAT_KEY)
        self.assertTrue(any(
            r.get("kind") == "grant_revoked"
            for r in state.observations.records()))

    def test_disarm_an_unknown_grant_through_chat(self) -> None:
        reply = self._control("disarm grt_nope")
        self.assertTrue(reply.startswith("disarm NOT performed"), reply)
        self.assertIn("GrantNotFound", reply)

    def test_arrest_a_registered_sleeper_through_chat(self) -> None:
        from nomorals.core.own_processes import (
            default_own_process_registry,
        )

        registry = default_own_process_registry()
        proc, row = registry.spawn(
            SLEEPER, label="chat sleeper", spawned_by="test_doc20p7")
        self.held.append(proc)
        reply = self._control(f"arrest {row['proc_id']} system")
        self.assertTrue(reply.startswith("arrested:"), reply)
        self.assertTrue(_wait_dead(row["pid"]))
        proc.wait(timeout=5)
        stored = registry.get(row["proc_id"])
        assert stored is not None
        self.assertEqual(stored["state"], "killed")
        state = get_immune_state(CHAT_KEY)
        self.assertTrue(any(
            r.get("kind") == "process_killed"
            for r in state.observations.records()))

    def test_arrest_an_unknown_process_through_chat(self) -> None:
        reply = self._control("arrest 424242 system")
        self.assertTrue(reply.startswith("arrest NOT performed"), reply)
        self.assertIn("no record of process", reply)

    def test_arrest_without_system_stays_report_only(self) -> None:
        reply = self._control("arrest 424242")
        self.assertTrue(reply.startswith("arrest refused"), reply)

    def test_uninjected_surface_keeps_the_unwired_posture(self) -> None:
        # Parity where P7 is not engaged: a bare control_immune
        # call (no teeth injected) produces exactly the pre-P7
        # honest receipts — not performed, not wired, in words.
        reply = control_immune("disarm grt_nope", chat_key=CHAT_KEY)
        self.assertTrue(reply.startswith("disarm NOT performed"), reply)
        self.assertIn("not wired", reply)
        reply = control_immune("arrest 424242 system", chat_key=CHAT_KEY)
        self.assertTrue(reply.startswith("arrest NOT performed"), reply)
        self.assertIn("no kill function wired", reply)


if __name__ == "__main__":
    unittest.main()
