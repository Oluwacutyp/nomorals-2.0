"""Tail-98 pins — nomorals/agents/watcher.py silent-failure sites.

All four live in / around ``WatchLoop.tick()`` — a resident daemon
sweep whose sub-duties are documented best-effort ("delivery is
best-effort", "learning is best-effort", "the farm report is
best-effort"). Each crash degrades its own counter to the honest
zero value; the ops report itself (paged / pending / findings) comes
from ``ops_alerts`` and is never touched by these swallows, so a
failed redelivery still shows as pending rather than as delivered.

46 E103 — Notifier.redeliver() crash -> redelivered stays 0.
EXCLUSION.
51 E103 — ReasoningAgent.mine_preventions() crash -> mined stays 0.
EXCLUSION.
59 E103 — workspace summary/autoscale crash -> workspace stays "".
EXCLUSION.
114 E106 — run()'s KeyboardInterrupt handler: the deliberate
top-level shutdown of the watch daemon (prints the stop line,
returns 0). EXCLUSION.
"""
from types import SimpleNamespace

import nomorals.agents.notifier as notifier_mod
import nomorals.agents.ops_alerts as ops_mod
import nomorals.agents.reasoning as reasoning_mod
import nomorals.agents.watcher as W

_REPORT = {"sent": [], "self_healed": [], "cooldown": [],
           "findings": [], "pending": 2}


def _patch_ops(monkeypatch):
    monkeypatch.setattr(ops_mod, "ops_alerts",
                        lambda ctx, heal_background=True: dict(_REPORT))


class _Notifier:
    def __init__(self, result=0, boom=False):
        self._result = result
        self._boom = boom

    def __call__(self, ctx):
        outer = self

        class _N:
            def redeliver(self):
                if outer._boom:
                    raise RuntimeError("delivery boom")
                return outer._result

        return _N()


class _Reasoner:
    def __init__(self, result=(), boom=False):
        self._result = list(result)
        self._boom = boom

    def __call__(self, ctx):
        outer = self

        class _R:
            def mine_preventions(self):
                if outer._boom:
                    raise RuntimeError("mining boom")
                return list(outer._result)

        return _R()


def _loop():
    return W.WatchLoop(SimpleNamespace(workspace=None))


# ── 46: redeliver crash -> 0, pending still reported ──────────────

def test_46_redeliver_crash_reports_zero_and_pending(monkeypatch):
    _patch_ops(monkeypatch)
    monkeypatch.setattr(notifier_mod, "Notifier", _Notifier(boom=True))
    monkeypatch.setattr(reasoning_mod, "ReasoningAgent",
                        _Reasoner(result=["a", "b"]))
    out = _loop().tick()
    assert out["redelivered"] == 0
    assert out["pending"] == 2  # queue state not hidden by the failure
    assert out["mined"] == 2


# ── 51: mining crash -> 0 ─────────────────────────────────────────

def test_51_mining_crash_reports_zero(monkeypatch):
    _patch_ops(monkeypatch)
    monkeypatch.setattr(notifier_mod, "Notifier", _Notifier(result=3))
    monkeypatch.setattr(reasoning_mod, "ReasoningAgent",
                        _Reasoner(boom=True))
    out = _loop().tick()
    assert out["mined"] == 0
    assert out["redelivered"] == 3


# ── 59: workspace crash -> "" ─────────────────────────────────────

class _BoomWorkspace:
    def summary_line(self):
        raise RuntimeError("farm boom")

    def autoscale_tick(self):
        pass


def test_59_workspace_crash_reports_empty(monkeypatch):
    _patch_ops(monkeypatch)
    monkeypatch.setattr(notifier_mod, "Notifier", _Notifier(result=0))
    monkeypatch.setattr(reasoning_mod, "ReasoningAgent", _Reasoner())
    loop = W.WatchLoop(SimpleNamespace(workspace=_BoomWorkspace()))
    out = loop.tick()
    assert out["workspace"] == ""


# ── 114: KeyboardInterrupt -> clean shutdown, return 0 ────────────

def test_114_keyboard_interrupt_returns_zero(monkeypatch):
    loop = _loop()

    def _ki():
        raise KeyboardInterrupt

    monkeypatch.setattr(loop, "tick", _ki)
    assert loop.run(once=True) == 0
