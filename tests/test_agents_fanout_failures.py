"""Pass tail-8 pins: agents/fanout failures stay honest, ledger intact.

Target: nomorals/agents/fanout.py — 2 converts / 8 exclusions.

Converts (parent-adjudication note in SURVEY.md):
- 433 sources coercion now records itself in errors (was: empty run,
  ok=True, zero errors — indistinguishable from a legitimate empty run).
- 579 render failure now returns an explicit failure notice (was: "",
  identical to the documented empty-table value — failure
  indistinguishable from empty; the sibling renderer at 653 already
  uses an explicit notice). The documented "" for an empty table is
  preserved via the early return. 579 stays E104-counted by design
  (broad handler, no exception-value use — the notice is a constant,
  so no exception text leaks into user-facing output); see tail-3
  precedent for converted-but-counted.

Exclusions: every other site maps a failure to a genuine degraded
value (documented default, miss value, recorded error, honest
fallback) and the run proceeds honestly.

NOT-recording: the target never imports the agents failures ledger,
so each test snapshots recent_failures() before its trigger and
asserts it is unchanged after (pollution-proof: other tests in the
same session may have recorded before this file ran; what matters is
that THIS trigger records nothing of its own). Ledger REUSED:
nomorals/agents/failures.py (same package — read-only for pins; no
record call, no import added to the target).

No except handlers in this file at all, so the scanner counts 0
sites here (stubs raise, the code under test catches).
"""

from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as lf
from nomorals.agents import fanout as fo


class BoomStr(str):
    def __str__(self, *a, **k):
        raise RuntimeError("str boom")


class BoomInt:
    def __int__(self, *a, **k):
        raise RuntimeError("int boom")


def _silent():
    before = lf.recent_failures()
    return before


def _assert_silent(before):
    assert lf.recent_failures() == before


def test_058_profile_detection_failure_keeps_laptop_default():
    before = _silent()
    with mock.patch("nomorals.core.profiles.get_profile_kind",
                    side_effect=RuntimeError("profile boom")):
        assert fo.fanout_cap() == 12
    _assert_silent(before)


def test_215_telemetry_failure_never_breaks_fan_out():
    before = _silent()
    ctx = SimpleNamespace(emit=mock.Mock(side_effect=RuntimeError("emit boom")))
    r = fo.fan_out("g", ["a1", "a2"], worker_fn=lambda a, ag: {"v": a},
                   context=ctx)
    assert len(r.results) == 2 and r.results[0]["v"] == "a1"
    assert r.errors == [] and ctx.emit.called
    _assert_silent(before)


def test_349_progress_failure_never_breaks_map_reduce():
    before = _silent()
    r = fo.map_reduce([1, 2, 3, 4], lambda c, i: sum(c), sum, k=2,
                      on_progress=mock.Mock(side_effect=RuntimeError("prog boom")))
    assert r.reduced == 10
    _assert_silent(before)


def test_396_telemetry_failure_never_breaks_map_reduce():
    before = _silent()
    ctx = SimpleNamespace(emit=mock.Mock(side_effect=RuntimeError("emit boom")))
    r = fo.map_reduce([1, 2], lambda c, i: sum(c), sum, k=1, context=ctx)
    assert r.reduced == 3
    _assert_silent(before)


def test_433_sources_coercion_failure_is_recorded_not_silent():
    # Tail-8 convert: the coercion failure used to vanish into an
    # empty ok=True run (0 sources, 0 errors). It now records itself.
    before = _silent()
    res = fo.fan_out_compare("g", [BoomStr("x")],
                             worker_fn=lambda s, i: {"price": 1})
    assert res.sources == [] and res.results == []
    assert res.errors == ["sources: str boom"]
    assert res.ok is False
    _assert_silent(before)


def test_441_max_workers_failure_keeps_profile_cap():
    before = _silent()
    res = fo.fan_out_compare("g", ["a.com", "b.com"],
                             worker_fn=lambda s, i: {"price": 1},
                             max_workers=BoomInt(), profile="termux")
    assert len(res.results) == 2 and res.workers_used <= 4
    _assert_silent(before)


def test_463_bad_source_is_recorded_and_isolated():
    before = _silent()

    def worker(s, i):
        if s == "bad.com":
            raise RuntimeError("fetch exploded")
        return {"price": 99}

    res = fo.fan_out_compare("g", ["good.com", "bad.com"], worker_fn=worker)
    assert res.results[0]["price"] == 99
    assert res.results[1].get("error")
    assert len(res.errors) == 1 and "bad.com" in res.errors[0]
    _assert_silent(before)


def test_487_pool_failure_is_recorded():
    before = _silent()

    class FakePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, *a, **k):
            raise RuntimeError("submit boom")

    with mock.patch.object(fo, "ThreadPoolExecutor", FakePool):
        res = fo.fan_out_compare("g", ["a.com"],
                                 worker_fn=lambda s, i: {"price": 1})
    assert res.errors == ["fan-out failed: submit boom"]
    _assert_silent(before)


def test_579_render_failure_is_noticed_not_blank():
    # Tail-8 convert: a malformed row used to render as "", identical
    # to the documented empty-table value. It now says so explicitly.
    # The legitimate empty table still renders "" (early return).
    before = _silent()
    assert fo.render_comparison_table(["not-a-dict"]) == (
        "comparison table render failed")
    assert fo.render_comparison_table([]) == ""
    _assert_silent(before)


def test_579_synthesize_keeps_data_and_flags_render_failure():
    # Consumer-visible effect of the 579 convert: the table data and
    # the success summary survive, but markdown no longer silently
    # blanks out when a value cannot be stringified.
    before = _silent()
    synth = fo.synthesize_comparison([{"price": BoomStr("x")}])
    assert synth["table"] != []
    assert synth["markdown"] == "comparison table render failed"
    assert synth["summary"].startswith("1/1 sources compared")
    _assert_silent(before)


def test_653_render_failure_is_noticed():
    before = _silent()
    bad = fo.MapReduceResult(n_items=5, n_workers=2, reduced=BoomStr("x"),
                             errors=[], seconds=0.0)
    out = fo.render_mapreduce(bad)
    assert "render failed" in out and "5 items" in out
    _assert_silent(before)
