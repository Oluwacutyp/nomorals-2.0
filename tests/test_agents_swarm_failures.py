"""Tail-108 pins: nomorals/agents/swarm.py census sites.

All three sites are exclusions — degradation is fail-closed or
named in-band, and nothing is fabricated:

* 149 — ``_model_available``: a crashing router stats snapshot ->
  False, byte-identical to the no-router and mock/offline answers;
  the swarm takes its heuristic decomposition path instead of
  planning against a model it cannot confirm exists.
* 214 — ``_prior_digests``: a crashing blackboard read -> "", the
  documented best-effort value ("awareness is a bonus"); no digest
  is invented, legs simply run without cross-leg context.
* 502 — ``render_swarm``: any rendering crash -> the fallback
  string ``"swarm {run_id} (render failed)"`` — the failure is
  NAMED in the output, never dressed as a report ("rendering never
  breaks callers" is the printed contract).

Shelved sketch: 149's guard is silent while the module logs its
other degradations; a ``_log.debug`` there would make "why did the
swarm go heuristic" diagnosable. Observability only.
"""

import types

from nomorals.agents import swarm as sw


def _agent_with_snapshot(snapshot):
    ctx = types.SimpleNamespace(
        router=types.SimpleNamespace(stats_snapshot=snapshot))
    return sw.SwarmAgent(ctx)


# ── 149: _model_available ────────────────────────────────────────

def test_model_available_snapshot_crash_is_false():
    def _snap():
        raise RuntimeError("snapshot boom")

    assert _agent_with_snapshot(_snap)._model_available() is False


def test_model_available_controls():
    assert _agent_with_snapshot(lambda: {"active": "gpt-x"})._model_available() is True
    assert _agent_with_snapshot(lambda: {"active": "mock"})._model_available() is False
    assert sw.SwarmAgent(types.SimpleNamespace())._model_available() is False


# ── 214: _prior_digests ──────────────────────────────────────────

class _BoardBoom:
    def topic(self, run_id):
        raise RuntimeError("board boom")


class _BoardOk:
    def topic(self, run_id):
        return {"swarm.r1.leg.0": {"digest": "DIGEST0", "subtask": "sub0"},
                "zz": {}}


def test_prior_digests_board_crash_returns_empty():
    agent = sw.SwarmAgent(types.SimpleNamespace(), blackboard=_BoardBoom())
    assert agent._prior_digests("r1") == ""


def test_prior_digests_controls():
    agent = sw.SwarmAgent(types.SimpleNamespace(), blackboard=_BoardOk())
    assert "DIGEST0" in agent._prior_digests("r1")
    assert sw.SwarmAgent(types.SimpleNamespace())._prior_digests("r1") == ""


# ── 502: render_swarm ────────────────────────────────────────────

class _BadStr:
    def __str__(self):
        raise RuntimeError("str boom")


def test_render_swarm_crash_returns_named_fallback():
    result = sw.SwarmResult(
        goal="g", subtasks=["s"],
        legs=[{"ok": True, "subtask": _BadStr(), "digest": "d", "seconds": 1}],
        synthesis="syn", run_id="rX")
    assert sw.render_swarm(result) == "swarm rX (render failed)"


def test_render_swarm_healthy_renders_synthesis():
    result = sw.SwarmResult(
        goal="g", subtasks=["s"],
        legs=[{"ok": True, "subtask": "sub", "digest": "d", "seconds": 1}],
        synthesis="the fused answer", run_id="rY")
    out = sw.render_swarm(result)
    assert "the fused answer" in out
    assert "render failed" not in out
