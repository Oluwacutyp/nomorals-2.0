"""Failure pins for nomorals/core/tool_render.py (tail-112).

One pin per census site. All sites adjudicated EXCLUSION:

- 176 render_tool_value outer guard: a render crash -> str(value)
  truncated to 1000 chars. The docstring prints "Never raises"; the
  fallback returns the value's own text — the structured rendering
  is what is lost, nothing is fabricated.
- 179 render_tool_value inner fallback: str(value) itself raises ->
  "(unprintable result)", an honest label.
- 206 emit_progress: a raising progress callback is swallowed. The
  contract is printed on ProgressCallback itself: "Devon swallows
  callback errors so progress can never break the run." Telemetry
  by design; the run's outcome never depends on the callback.

The module has no logger: non-recording is structural (pinned below).
"""

from unittest import mock

from nomorals.core import tool_render as tr


class _BadStr:
    def __str__(self):
        raise RuntimeError("str boom")


# --------------------------------------------------------------------- 176
def test_render_tool_value_crash_falls_back_to_str():
    with mock.patch.object(tr, "_render",
                           side_effect=RuntimeError("render boom")):
        assert tr.render_tool_value({"a": 1}) == "{'a': 1}"


# --------------------------------------------------------------------- 179
def test_render_tool_value_unprintable():
    with mock.patch.object(tr, "_render",
                           side_effect=RuntimeError("render boom")):
        assert tr.render_tool_value(_BadStr()) == "(unprintable result)"


def test_render_tool_value_controls():
    assert tr.render_tool_value(None) == "(no output)"
    assert tr.render_tool_value({"a": 1}) == "a: 1"
    long = "\n".join(f"line{i}" for i in range(7))
    assert tr.render_tool_value(long, max_lines=5).endswith(
        "… (+2 more lines)")


# --------------------------------------------------------------------- 206
def test_emit_progress_swallows_callback_errors():
    def bad_cb(*args):
        raise RuntimeError("cb boom")

    assert tr.emit_progress(bad_cb, 1, 2, "t", "done") is None
    assert tr.emit_progress(None, 1, 2, "t", "done") is None


def test_emit_progress_control_delivers_coerced_args():
    calls = []
    tr.emit_progress(lambda *a: calls.append(a), "2", "5", 7, "done")
    assert calls == [(2, 5, "7", "done")]


# ------------------------------------------------------- non-recording
def test_module_has_no_logger():
    assert not hasattr(tr, "_log")
