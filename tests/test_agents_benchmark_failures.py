"""Tail-63 pins for nomorals/agents/benchmark.py —
1 convert (884, parent amendment) / 5 exclusions. Each test
pins one site with a control half proving the guarded
operation was actually ATTEMPTED via the double's own call
log. The file contains no exception handlers, so it adds
zero census sites.

NOT-recording is pinned against the agents package ledger
(nomorals/agents/failures.py): recent_failures() snapshot
before/after every trigger. The 884 convert is the one site
that DOES record there now (via .failures.record_failure,
the agents/ standard surface benchmark.py imports for it).

Verdict notes (full grounds in the tail-63 SURVEY.md):
- 102: _provider_name failure -> "" -> measurable() reads
  the context as unmeasurable. measurable()'s whole job is
  defensive capability detection over arbitrary context
  objects; "" is its designed "cannot measure" value.
- 397: temp-file cleanup OSError in _run_python's finally
  -> swallowed; the sandbox result was already decided and
  is returned intact. Teardown, not the operation.
- 884 CONVERTED (parent overrule, amendment): a claim check
  that CRASHES now returns None (unverifiable, per the
  function's own docstring and the module design "excluded
  from the denominator ... never silently counted either
  way"), records the failure on the agents ledger, and
  _score_claim_labels excludes the claim from the scored
  denominator into the unverifiable tally. Pin asserts the
  NEW behavior end-to-end, including a working control
  claim scoring normally in the same task.
- 1356: warmup decide() failure -> pass; the warmup is an
  un-timed prelude, explicitly "not a result".
- 1363 / 1372: a failed timed rep -> pass, and the rep's
  elapsed time is STILL appended and counted ("a failed rep
  counts, slowly"). The recorded datum (elapsed time) is
  truthful on the failure path.
"""

import types
from pathlib import Path

import pytest

from nomorals.agents import benchmark as BM
from nomorals.agents import coremind as CM
from nomorals.agents import failures as agent_fail


def _snap():
    return list(agent_fail.recent_failures(200))


# 102 — provider-name read failure -> "" / measurable False
class _RaisingLLM:
    def __init__(self):
        self.accesses = 0

    @property
    def provider(self):
        self.accesses += 1
        raise RuntimeError("settings boom")


def test_102_provider_name_failure_reads_unmeasurable():
    before = _snap()
    llm = _RaisingLLM()
    settings = types.SimpleNamespace(llm=llm, offline=False)
    ctx = types.SimpleNamespace(router=object(), settings=settings)
    assert BM._provider_name(ctx) == ""
    assert llm.accesses >= 1  # the read was attempted
    assert BM.measurable(ctx) is False
    good = types.SimpleNamespace(
        router=object(),
        settings=types.SimpleNamespace(
            llm=types.SimpleNamespace(provider="groq"), offline=False))
    assert BM._provider_name(good) == "groq"  # control: healthy read
    assert BM.measurable(good) is True
    assert _snap() == before  # NOT recorded


# 397 — temp cleanup failure does not mask the sandbox result
def test_397_cleanup_oserror_result_intact(monkeypatch):
    before = _snap()
    real_unlink = Path.unlink
    attempted = []

    def flaky_unlink(self, missing_ok=False):
        if self.name == "script.py" and "nm-bench-" in str(self):
            attempted.append(str(self))
            raise OSError("locked")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    res = BM._run_python("print('hello-bench')")
    assert res["exit_code"] == 0
    assert "hello-bench" in res.get("stdout", "")
    assert attempted  # cleanup was attempted
    assert _snap() == before


# 884 — CONVERTED: a crashed claim check is unverifiable (None),
# recorded on the agents ledger, and excluded from the scorer's
# denominator (it joins the unverifiable tally instead).
def test_884_crashed_check_is_unverifiable_and_excluded(monkeypatch):
    before = _snap()
    real_read_text = Path.read_text
    attempted = []

    def spy_read_text(self, *a, **k):
        if "no_such_file_xyz" in str(self):
            attempted.append(str(self))
            raise FileNotFoundError(str(self))
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy_read_text)
    # (a) the crashed check returns None — unverifiable, never a
    # false verdict — and the failure IS recorded (site + type).
    crashed = BM._run_claim_check(
        ("line_count_ge", "nomorals/no_such_file_xyz.py", 5))
    assert crashed is None
    assert attempted  # the check's read was attempted
    after = _snap()
    site = "benchmark.claim_check"
    assert len([f for f in after if f.site == site]) == len(
        [f for f in before if f.site == site]) + 1
    assert after[0].site == site
    assert after[0].exc_type == "FileNotFoundError"
    # (b)+(c) end-to-end scoring: a task mixing a working claim,
    # the crashed claim and a check-less claim scores ONLY the
    # working claim; the crashed claim leaves the verifiable
    # denominator, joins the unverifiable tally, and keeps
    # exactly one per_claim row with expected "unverifiable".
    task = BM.ResearchTask(
        id="pin-884", question="pin task",
        claims=[
            {"id": "ok1", "text": "benchmark defines DimensionScore",
             "check": ("symbol", "nomorals.agents.benchmark",
                       "DimensionScore")},
            {"id": "bad1", "text": "check crashes on a missing file",
             "check": ("line_count_ge",
                       "nomorals/no_such_file_xyz.py", 5)},
            {"id": "none1", "text": "no harness check", "check": None},
        ])
    labels = {"ok1": "true", "bad1": "false",
              "none1": "unverifiable"}
    scored = BM._score_claim_labels(task, labels)
    assert scored["verifiable"] == 1
    assert scored["unverifiable"] == 2
    assert scored["correct"] == 1  # only genuinely scored claims
    rows = {row["id"]: row for row in scored["per_claim"]}
    assert set(rows) == {"ok1", "bad1"}
    assert rows["bad1"] == {"id": "bad1", "expected": "unverifiable",
                            "got": "false", "hit": False}
    assert rows["ok1"]["expected"] == "true"
    assert rows["ok1"]["hit"] is True
    # controls: genuine verdicts from the same function unchanged
    assert BM._run_claim_check(
        ("symbol", "nomorals.agents.benchmark", "DimensionScore")) is True
    assert BM._run_claim_check(
        ("line_count_ge", "nomorals/agents/benchmark.py", 100000)) is False
    assert BM._run_claim_check(None) is None


# 1356 / 1363 / 1372 — latency reps
class _FakeMind:
    decide_calls = 0
    check_calls = 0
    decide_mode = "ok"   # ok | first | all
    check_mode = "ok"    # ok | all

    def __init__(self, context):
        self.context = context

    def decide(self, text, allow_model=True):
        type(self).decide_calls += 1
        mode = type(self).decide_mode
        if mode == "all" or (mode == "first" and type(self).decide_calls == 1):
            raise RuntimeError("decide boom")
        return None

    def _model_check(self, text, intent):
        type(self).check_calls += 1
        if type(self).check_mode == "all":
            raise RuntimeError("model_check boom")
        return None


@pytest.fixture()
def fake_mind(monkeypatch):
    _FakeMind.decide_calls = 0
    _FakeMind.check_calls = 0
    _FakeMind.decide_mode = "ok"
    _FakeMind.check_mode = "ok"
    monkeypatch.setattr(CM, "CoreMind", _FakeMind)
    return _FakeMind


def _latency_ctx():
    return types.SimpleNamespace(
        settings=types.SimpleNamespace(), router=None,
        extras={}, memory=None)


# 1356 — warmup failure is not a result; measurement proceeds
def test_1356_warmup_failure_not_a_result(fake_mind):
    before = _snap()
    fake_mind.decide_mode = "first"  # only the warmup call raises
    score = BM._dim_latency(_latency_ctx(), 1, mode="selftest")
    # reps = 5: warmup(1) + fast reps(5) decide calls, all attempted
    assert fake_mind.decide_calls == 6
    assert fake_mind.check_calls == 5
    assert score.total == 5
    assert _snap() == before


# 1363 — every fast rep fails; each still counts with its time
def test_1363_failed_fast_reps_still_count(fake_mind):
    before = _snap()
    fake_mind.decide_mode = "all"  # warmup + every fast rep raises
    score = BM._dim_latency(_latency_ctx(), 1, mode="selftest")
    assert fake_mind.decide_calls == 6  # every rep was attempted
    fast = [d for d in score.details if "fast-path" in d["task"]]
    assert fast and "over 5 reps" in fast[0]["detail"]
    assert _snap() == before


# 1372 — every heavy rep fails; each still counts with its time
def test_1372_failed_heavy_reps_still_count(fake_mind):
    before = _snap()
    fake_mind.check_mode = "all"  # every heavy rep raises
    score = BM._dim_latency(_latency_ctx(), 1, mode="selftest")
    assert fake_mind.check_calls == 5  # every rep was attempted
    heavy = [d for d in score.details if "heavy-path" in d["task"]]
    assert heavy and "over 5 reps" in heavy[0]["detail"]
    assert _snap() == before
