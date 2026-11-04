"""Server-confirmed pairing codes (nodefix) — the phantom-code fix.

The defect (Baileys issue #2737, protocol-captured): the
library's ``requestPairingCode()`` resolves and returns a code
BEFORE WhatsApp's server evaluates the companion_hello IQ — it
sends the IQ with the fire-and-forget ``sendNode`` and returns
the locally-minted code at once, so a server ``400 bad-request``
arrives after the code is already on its way to the owner, who
types a code that was never registered and gets WhatsApp's
"Couldn't link device".  That is the owner's live failure of
2026-11-04.  Verified 2026-11-04: still unfixed in every
published build (6.7.24 legacy AND 7.0.0-rc14 ship the same
``await sendNode(...)`` + immediate return, and master has no
fix), so the guarantee is built in OUR bridge instead of
waited for: bridge/whatsapp-bridge.mjs sends stage 1 through
the socket's awaited ``query()`` with a byte-identical node,
and prints/broadcasts the code only after the server accepts —
a refusal rejects with the server's own text and no code line
exists.  The bridge pin is pinned at 6.7.24 (exact) so the
mirrored node is reviewed against a fixed library, not a
drifting caret range.

The bridge half is verified by ``node --check`` plus a build-
time extraction harness (the shipped function run against the
real library utils with a fake socket: accept → code, refuse →
rejection, no code) — the repo has no JS test harness.  What
pytest CAN pin is the supervisor-visible contract, which is
what lives here:

* a request-stage refusal that lands AFTER a code line (the
  phantom sequence, replayed at the seam) leaves NO displayable
  code behind — the refusal clears the buffered code, the
  reason is what /pair code answers, and the record stays
  "pairing" because the bridge lives;
* the clearing is proc-checked: a stale process's refusal can
  never kill a fresh attempt's live code;
* a fresh /pair after a refusal recovers cleanly — new code,
  old error gone;
* the live entry names its bridge engine ("node" here — the
  harness selects the node rung explicitly, as on main the
  supervisor's default rung is "go" and stamps that instead).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from nomorals.social.pairing import (
    BridgeSupervisor,
    PairingRegistry,
)


# ── fakes (the bridgefix harness shape) ────────────────────────────

class FakeProc:
    """A bridge process: yields its lines, then either dies
    (poll() → 0, like a process whose stdout closed) or parks
    alive on a gate when block=True (a running bridge)."""

    def __init__(self, lines=(), die=True, block=False):
        self.terminated = False
        self._poll = None
        self._gate = threading.Event()
        self.stdout = self._stream(list(lines), die, block)

    def _stream(self, lines, die, block):
        for line in lines:
            yield line
        if block:
            self._gate.wait(30)
        elif die:
            self._poll = 0

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0
        self._gate.set()


def wait_for(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return bool(pred())


def bare_bridge(tmp_path) -> Path:
    path = tmp_path / "whatsapp-bridge.mjs"
    path.write_text("// bridge\n", encoding="utf-8")
    return path


def make_sup(tmp_path, proc_factory, **kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = proc_factory()
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path=bare_bridge(tmp_path),
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        engine="node",
        node_resolver=lambda: "/staged/node", **kw)
    return sup, registry, spawns


def bare_runtime(sup):
    from nomorals.agents.partner.runtime_provision import \
        RuntimeProvisionMixin

    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = sup
    return rt


CODE_LINE = ("2026-11-04T02:20:00.000Z [bridge] PAIRING CODE for "
             "+2348012345678: WXYZ-1234 (type into WhatsApp > "
             "Linked Devices)")
NEW_CODE_LINE = ("2026-11-04T02:25:00.000Z [bridge] PAIRING CODE "
                 "for +2348012345678: ABCD-9876 (type into "
                 "WhatsApp > Linked Devices)")
# The confirmed bridge's refusal line: the text is the server's
# own error text, carried by the awaited query's rejection.
REFUSAL_LINE = ("2026-11-04T02:20:03.000Z [bridge] pairing code "
                "request failed: bad-request")


# ── a refusal after a code leaves nothing displayable ─────────────

def test_refusal_after_code_line_leaves_no_displayable_code(
        tmp_path):
    # The phantom sequence, replayed at the supervisor seam: a
    # code line arrived, THEN the request-stage refusal lands.
    # Whatever the bridge did, once the refusal is in, the code
    # must stop being displayable — the reason is the answer.
    lines = [CODE_LINE, REFUSAL_LINE]
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc(lines, block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: bool(sup.pairing_error(rec.id)))
    assert sup.pairing_error(rec.id) == (
        "WhatsApp refused the code request: bad-request")
    assert registry.get(rec.id).last_error == (
        "WhatsApp refused the code request: bad-request")
    # no displayable code survives the refusal…
    assert sup.pairing_code(rec.id) is None
    # …and the /pair code reply tells the refusal story — it
    # neither shows the dead code nor falls back to "no live code"
    reply = bare_runtime(sup)._control_pair(
        "code whatsapp:2348012345678", chat_key="owner")
    assert "WhatsApp refused the code request: bad-request" \
        in reply
    assert "WXYZ-1234" not in reply
    assert "no live code" not in reply
    # a refusal is not a death: the bridge lives, still pairing
    assert registry.get(rec.id).status == "pairing"


def test_stale_procs_refusal_cannot_clear_the_live_code(tmp_path):
    # The code-clearing is proc-checked: a refusal attributed to
    # a process that is not the live one (an old bridge's late
    # line) must not kill a fresh attempt's displayable code.
    # (The registry error note itself follows the module's
    # standing last-writer convention, shared with the death and
    # link-failure notes; the proc guard protects the live
    # entry — its state and its code.)
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc([CODE_LINE], block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: sup.pairing_code(rec.id) is not None)
    sup._note_request_refused(
        rec.id, object(),
        "WhatsApp refused the code request: bad-request")
    code, _age = sup.pairing_code(rec.id)
    assert code == "WXYZ-1234"
    assert sup._live[rec.id].last_error == ""
    # …while the live process's own refusal clears it.
    sup._note_request_refused(
        rec.id, sup._live[rec.id].proc,
        "WhatsApp refused the code request: bad-request")
    assert sup.pairing_code(rec.id) is None
    assert sup.pairing_error(rec.id) == (
        "WhatsApp refused the code request: bad-request")


def test_fresh_pair_after_refusal_recovers_with_a_new_code(
        tmp_path):
    procs = [FakeProc([CODE_LINE, REFUSAL_LINE], die=True),
             FakeProc([NEW_CODE_LINE], block=True)]
    sup, registry, _ = make_sup(
        tmp_path, lambda: procs.pop(0))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: bool(sup.pairing_error(rec.id)))
    assert sup.pairing_code(rec.id) is None
    rec2 = sup.pair_whatsapp("2348012345678")
    assert rec2.id == rec.id
    assert wait_for(lambda: sup.pairing_code(rec.id) is not None)
    code, _age = sup.pairing_code(rec.id)
    assert code == "ABCD-9876"
    # the fresh launch superseded the refusal
    assert sup.pairing_error(rec.id) == ""
    assert registry.get(rec.id).last_error == ""


# ── the engine marker ──────────────────────────────────────────────

def test_live_entry_names_the_node_engine(tmp_path):
    sup, _registry, _ = make_sup(
        tmp_path, lambda: FakeProc([CODE_LINE], block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: sup.pairing_code(rec.id) is not None)
    assert sup._live[rec.id].bridge_engine == "node"
