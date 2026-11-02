"""Pass tail-96 pins: nomorals/agency/expansion.py — 4 sites, all exclusions.

143 — ``_organ_pending`` probe crash -> return 0. EXCLUSION: the probe
walks duck-typed organ APIs (pending_count / open_items / queue_list /
pending attr) where partial implementations are expected; the broad
handler is the probe's floor. The sole consumer, ``has_work``, treats
0 (probe says empty) and None (unknowable) IDENTICALLY — both fall to
the standing-initiative check — so a crashed probe changes no outcome,
and the next cycle re-probes.

177 — ``_gated_capabilities`` declaration crash -> treated as no
declaration. EXCLUSION (the batch's weakest — see the dedicated pin):
the function's stated contract is "No declaration = no gated actions
= no gate", and an unreadable declaration lands on the same path.
Counter-argument, recorded for adjudication: this fails OPEN on the
authority gate — an organ whose ``gated_capabilities()`` raises runs
its tick with NO standing-authority check, while a missing authority
object fails closed in ``_authority_allows``. Converting would require
inventing policy the contract does not state (gate on which
capabilities when the declaration itself is unreadable?), and this
module carries no logger at all, so the failure is currently invisible
— an observability gap, shelved as a sketch, not a convert.

245 — ``organ.tick()`` crash -> ``WorkResult(useful=False)``.
EXCLUSION: the agency loop's isolation boundary by design ("organ
failure = empty turn"). The outcome is honestly shaped — useful=False
is recorded by the scheduler as a not-useful turn — never a fabricated
success, and nothing is deposited on the failure path.

284 — ``_registry_collect()`` crash in ``_tick_sourcehealth`` ->
``WorkResult(useful=False)``. EXCLUSION: same isolation pattern; the
sourcehealth turn simply does not happen this cycle and no summary is
deposited (nothing fabricated about source health).
"""

import pytest

from nomorals.agency.expansion import EXPANSION_SYSTEMS, ExpansionSystems
from nomorals.storage.db import Database


def _systems(**kw):
    return ExpansionSystems(db=Database(":memory:"), **kw)


class _CrashPending:
    def pending_count(self):
        raise RuntimeError("queue backend down")


class _NoQueueApi:
    pass


class _CountingOrgan:
    def __init__(self, pending=0):
        self._pending = pending
        self.ticks = 0

    def pending_count(self):
        return self._pending

    def tick(self):
        self.ticks += 1
        return {"done": self._pending}


# ── 143: probe crash -> 0, indistinguishable from unknowable ─────

def test_143_pending_probe_crash_returns_zero():
    sys_ = _systems(organs={"games": _CrashPending()})
    assert sys_._organ_pending("games") == 0
    # control: a working probe reports the real count
    sys2 = _systems(organs={"games": _CountingOrgan(pending=3)})
    assert sys2._organ_pending("games") == 3


def test_143_crash_and_unknowable_have_identical_has_work():
    crashed = _systems(organs={"games": _CrashPending()})
    unknown = _systems(organs={"games": _NoQueueApi()})
    # no standing initiative: both stand down
    assert crashed.has_work("games") is False
    assert unknown.has_work("games") is False
    # with a standing initiative: both report work via the initiative
    crashed.seed_initiatives(now=1000.0)
    unknown.seed_initiatives(now=1000.0)
    assert crashed.has_work("games") is True
    assert unknown.has_work("games") is True


# ── 177: declaration crash -> no gate (weakest exclusion) ────────

class _RaisingDeclaration:
    def __init__(self):
        self.ticks = 0

    def gated_capabilities(self):
        raise RuntimeError("declaration machinery broken")

    def tick(self):
        self.ticks += 1
        return {"done": 1}


class _SingleAndRaising:
    gated_capability = "send_email"

    def gated_capabilities(self):
        raise RuntimeError("declaration machinery broken")


def test_177_declaration_crash_yields_no_caps():
    assert ExpansionSystems._gated_capabilities(_RaisingDeclaration()) == []
    # control: a readable declaration is honored in full
    class _Declares:
        gated_capability = "send_email"

        def gated_capabilities(self):
            return ["post", "pay"]
    assert ExpansionSystems._gated_capabilities(_Declares()) == [
        "send_email", "post", "pay"]


def test_177_single_declaration_survives_callable_crash():
    assert ExpansionSystems._gated_capabilities(_SingleAndRaising()) == [
        "send_email"]


def test_177_declaration_crash_runs_tick_ungated_today():
    # HONEST PIN of the fail-open shape: with no authority configured a
    # readable declaration would block the tick at the gate; a crashed
    # declaration skips the gate entirely and the organ runs.
    organ = _RaisingDeclaration()
    sys_ = _systems(organs={"games": organ})
    result = sys_.tick("games", now=1000.0)
    assert organ.ticks == 1
    assert result.useful is True


# ── 245: organ tick crash -> empty turn, nothing deposited ───────

class _CrashTick:
    def tick(self):
        raise RuntimeError("organ exploded")


def test_245_organ_tick_crash_is_empty_turn():
    sys_ = _systems(organs={"games": _CrashTick()})
    result = sys_.tick("games", now=1000.0)
    assert result.useful is False
    # non-recording: the failure deposits no knowledge item
    assert sys_.knowledge.count() == 0
    # control: a working organ's tick is useful and deposits
    ok = _systems(organs={"games": _CountingOrgan(pending=2)})
    res2 = ok.tick("games", now=1000.0)
    assert res2.useful is True
    assert ok.knowledge.count() == 1


# ── 284: registry collect crash -> sourcehealth empty turn ───────

def _raise_collect():
    raise RuntimeError("registry unavailable")


def test_284_registry_collect_crash_is_empty_turn():
    sys_ = _systems(registry_collect=_raise_collect)
    result = sys_.tick("sourcehealth", now=1000.0)
    assert result.useful is False
    assert sys_.knowledge.count() == 0
    # control: an empty-but-working registry still runs the turn and
    # deposits the (zero-count) summary
    ok = _systems(registry_collect=lambda: [])
    res2 = ok.tick("sourcehealth", now=1000.0)
    assert res2.useful is True
    assert ok.knowledge.count() == 1
