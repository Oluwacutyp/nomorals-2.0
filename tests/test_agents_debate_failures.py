"""Tail-64 pins for nomorals/agents/debate.py —
0 converts / 6 exclusions. Each test pins one site with a
control half proving the guarded operation was actually
ATTEMPTED via the double's own call log. The file contains
no exception handlers, so it adds zero census sites.

debate.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against
the agents package ledger (nomorals/agents/failures.py):
recent_failures() snapshot before/after every trigger.

Verdict notes (full grounds in the tail-64 SURVEY.md):
- 158 / 405 / 537 / 649: context.emit is telemetry; the
  inline contract is "telemetry never breaks debate /
  arbitration" and every result is computed independently
  of the emit.
- 676: a future whose result() raises lands on the
  function's own designed "(no answer)" sentinel (the same
  value used for empty answers and missing labels);
  _converged explicitly refuses to count it as consensus.
- 795: render_debate's contract is "Never raises"; the
  fallback is a degraded but honest one-line rendering.
"""

from unittest import mock

from nomorals.agents import debate as DB
from nomorals.agents import failures as agent_fail


def _snap():
    return list(agent_fail.recent_failures(200))


class BadContext:
    def __init__(self):
        self.calls = []

    def emit(self, kind, **kw):
        self.calls.append(kind)
        raise RuntimeError("emit dead")


def _artifact(content="work"):
    return DB.WorkArtifact(content=content, summary="s")


def test_158_debate_round_emit_failure_completes():
    before = _snap()
    ctx = BadContext()
    debate = DB.Debate(
        coder_fn=lambda b, f, a: _artifact(),
        critic_fn=lambda a, r: DB.Critique(verdict="approve", score=90.0),
        context=ctx)
    res = debate.run("brief")
    assert res.verdict == "approved"
    assert ctx.calls == ["swarm.debate_round"]
    assert _snap() == before


def test_405_panel_round_emit_failure_completes():
    before = _snap()
    ctx = BadContext()
    panel = DB.PanelDebate(
        position_fns={
            "a": lambda: DB.Position(label="a", artifact=_artifact())},
        critic_fn=lambda a, r: DB.Critique(verdict="approve", score=90.0),
        context=ctx, max_rounds=1)
    res = panel.run("brief")
    assert res.verdict == "decided"
    assert ctx.calls == ["swarm.panel_round"]
    assert _snap() == before


def test_537_arbitration_emit_failure_keeps_decision():
    before = _snap()
    ctx = BadContext()
    out = DB.arbitrate(
        [{"label": "x", "decision": "D1", "evidence": "e"}],
        context=ctx, autonomous=True)
    assert out.decision == "D1"
    assert ctx.calls == ["swarm.arbitration"]
    assert _snap() == before


def test_649_symmetric_round_emit_failure_keeps_consensus():
    before = _snap()
    ctx = BadContext()

    def deb(first):
        def fn(brief, others):
            return "same" if others else first
        return fn

    debate = DB.SymmetricDebate(
        debater_fns={"a": deb("A"), "b": deb("B")}, context=ctx)
    res = debate.run("q")
    assert res.verdict == "consensus"
    assert ctx.calls == ["swarm.symmetric_round"]
    assert _snap() == before


def test_676_collect_future_failure_yields_no_answer():
    before = _snap()
    debate = DB.SymmetricDebate(
        debater_fns={"a": lambda b, o: "x", "b": lambda b, o: "y"})
    calls = []
    real = debate._safe_call

    def flaky(label, brief, others):
        calls.append(label)
        if label == "b":
            raise RuntimeError("escaped _safe_call")
        return real(label, brief, others)

    debate._safe_call = flaky
    out = debate._collect("brief", {}, round_no=0)
    assert out == {"a": "x", "b": "(no answer)"}
    assert sorted(calls) == ["a", "b"]
    assert _snap() == before


def test_795_render_failure_yields_fallback_string():
    before = _snap()
    calls = []

    def bad_banner(*a, **k):
        calls.append(1)
        raise RuntimeError("render dead")

    res = DB.DebateResult(debate_id="d-1", verdict="approved", rounds=1,
                          score=90.0, final_artifact=_artifact(),
                          transcript=[], reason="r")
    with mock.patch("nomorals.agents.render.banner", bad_banner):
        out = DB.render_debate(res)
    assert out == "debate result (DebateResult): approved"
    assert calls == [1]
    assert _snap() == before
