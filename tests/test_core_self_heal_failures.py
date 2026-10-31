"""Failure pins for nomorals/core/self_heal.py (tail-72).

One pin per census site. All sites are adjudicated EXCLUSION except
164, which is FLAGGED for parent adjudication (pin asserts current
behavior; see SURVEY.md in the tail-72 verify dir):

- 104 _safe: failure -> the caller-chosen default (the never-raises
  primitive; every consumer handles its default explicitly)
- 164 _probe_inner outer guard: probe-machinery failure -> broken=True,
  exc=None, "probe machinery failed: …" (report-only; CONVERTED by
  parent amendment — was FLAGGED: it read byte-identical to healthy)
- 714 _fingerprint_of: failure -> "" (the designed no-fingerprint signal)
- 726 _report_outcome: learning side-channel failure -> swallowed by design
- 854 heal_all per-kind guard: crash -> HealResult with named skip_reason
- 858 heal_all outer guard: loop failure -> partial results, never raises
"""

import types

import nomorals.core.error_intelligence as ei
from nomorals.core import self_heal as sh


# --------------------------------------------------------------------- 104
def test_safe_returns_default_on_failure():
    def boom():
        raise ValueError("probe-boom")

    assert sh._safe(boom) is None
    assert sh._safe(boom, default="d") == "d"
    assert sh._safe(lambda: 7) == 7  # success path (control)
    # Consumer-level: the defaults are the designed signals downstream —
    # _read_source's None becomes an honest "could not read" skip_reason.
    assert sh._read_source("/no/such/file_xyz.py") is None
    assert sh._write_source("/no/such/dir/file.py", ["x"]) is False
    assert sh._compiles("/no/such/file_xyz.py") is False


# --------------------------------------------------------------------- 164
def test_probe_machinery_failure_reports_broken_not_healthy(monkeypatch):
    """CONVERTED (tail-72 parent amendment): if the probe machinery
    itself dies (here the ThreadPoolExecutor constructor raises), the
    command was never probed — reporting healthy would be a lie on
    the module's central decision surface. The honest shape is the
    module's own report-only one: broken=True with exc=None (never
    auto-fixed; the caller reports) and the machinery failure named
    in err. "Never raises" still holds."""
    made = []

    class RaisingTP:
        def __init__(self, *a, **k):
            made.append((a, k))
            raise RuntimeError("no threads")

    fake_cf = types.SimpleNamespace(
        ThreadPoolExecutor=RaisingTP, TimeoutError=sh._cf.TimeoutError)
    monkeypatch.setattr(sh, "_cf", fake_cf)
    rep = sh._probe_inner(lambda *a, **k: "fine", "whatever")
    assert made  # the executor construction was attempted (control)
    assert rep.broken is True
    assert rep.exc is None  # report-only: no exception to auto-fix from
    assert rep.err == "probe machinery failed: RuntimeError: no threads"
    assert rep.loc == ""
    assert sh.heal_probe(lambda *a, **k: "fine", "whatever") == (
        True, None, "probe machinery failed: RuntimeError: no threads", "")
    monkeypatch.undo()
    healthy = sh._probe_inner(lambda *a, **k: "a real answer", "whatever")
    # Healthy control unchanged — and now distinguishable from the
    # machinery-failure report above.
    assert (healthy.broken, healthy.exc, healthy.err) == (False, None, "")


# --------------------------------------------------------------------- 714
def test_fingerprint_failure_returns_empty(monkeypatch):
    assert sh._fingerprint_of(ValueError("x"))  # healthy control: non-empty

    class BadFP:
        @staticmethod
        def of(exc):
            raise RuntimeError("fp down")

    monkeypatch.setattr(ei, "ErrorFingerprint", BadFP)
    # "" is the designed no-fingerprint signal: _report_outcome guards on
    # it and HealResult.fingerprint simply claims nothing.
    assert sh._fingerprint_of(ValueError("x")) == ""


# --------------------------------------------------------------------- 726
def test_report_outcome_swallows_intelligence_failure():
    class Intel:
        def __init__(self, fail):
            self.fail, self.calls = fail, []

        def record_outcome(self, fp, fixed=False, fix=""):
            self.calls.append((fp, fixed, fix))
            if self.fail:
                raise RuntimeError("kb down")

    bad = Intel(True)
    # Learning never breaks healing: no raise, no return value, and the
    # heal outcome itself is reported through HealResult, not this call.
    assert sh._report_outcome(bad, "fp1", True, "fixed it") is None
    assert bad.calls == [("fp1", True, "fixed it")]  # attempted (control)
    good = Intel(False)
    sh._report_outcome(good, "fp1", True, "fixed it")
    assert good.calls == [("fp1", True, "fixed it")]  # healthy control
    # Guard cases: no intelligence / no fingerprint -> no call at all.
    sh._report_outcome(None, "fp1", True, "x")
    sh._report_outcome(good, "", True, "x")
    assert good.calls == [("fp1", True, "fixed it")]


# ------------------------------------------------------------- 854 and 858
def _flaky_heal_one(handle, kind, chat_key="", dry_run=False,
                    intelligence=None):
    if kind == "boom":
        raise RuntimeError("heal_one exploded")
    return sh.HealResult(kind)


def test_heal_all_records_crashed_kind_with_skip_reason(monkeypatch):
    monkeypatch.setattr(sh, "heal_one", _flaky_heal_one)
    summary = sh.heal_all(lambda *a, **k: "ok", ["fine", "boom"])
    assert summary["total"] == 2
    by_kind = {r["kind"]: r for r in summary["results"]}
    # The crash does not vanish: the result carries a named skip_reason
    # on the module's standard surface (HealResult -> to_dict -> report).
    assert by_kind["boom"]["skip_reason"] == "heal probe crashed unexpectedly"
    assert by_kind["boom"]["broken"] is False  # a crash is not a verdict
    assert by_kind["fine"]["skip_reason"] == ""


def test_heal_all_survives_raising_kinds_iterable(monkeypatch):
    monkeypatch.setattr(sh, "heal_one", _flaky_heal_one)
    raised = []

    def gen():
        yield "fine"
        raised.append(True)
        raise RuntimeError("iterable died")

    summary = sh.heal_all(lambda *a, **k: "ok", gen())
    assert raised  # the iterable really died mid-loop (control)
    # Never raises; the partial results are returned and counted as-is.
    assert summary["total"] == 1
    assert [r["kind"] for r in summary["results"]] == ["fine"]
