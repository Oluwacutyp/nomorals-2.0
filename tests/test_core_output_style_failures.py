"""Failure pins for nomorals/core/output_style.py (tail-112).

One pin per census site. All sites adjudicated EXCLUSION:

- 168 render_output outer guard: any rendering crash -> str(payload)
  truncated to max_chars. The function's printed contract is "never
  raises"; the fallback returns the payload's own text, unstyled —
  nothing is fabricated, the styling is what is lost.
- 171 render_output inner fallback: str(payload) itself raises ->
  "(unprintable output)", an honest label for an unrenderable payload.
- 205 style_output_text outer guard: crash -> the caller's own text
  truncated to max_chars, byte-identical to the legitimate input.

The module has no logger: non-recording is structural (pinned below).
"""

from unittest import mock

from nomorals.core import output_style as os_


class _BadTitle:
    def __str__(self):
        raise RuntimeError("title boom")

    def __format__(self, spec=""):
        raise RuntimeError("title boom")


class _BadStr:
    def __str__(self):
        raise RuntimeError("str boom")


# --------------------------------------------------------------------- 168
def test_render_output_crash_falls_back_to_payload_text():
    out = os_.render_output(kind="info", title=_BadTitle(),
                            payload="hello world", max_chars=5)
    assert out == "hello"


def test_render_output_healthy_control():
    out = os_.render_output(kind="info", title="T", payload={"a": 1})
    assert "T" in out and "a: 1" in out
    out = os_.render_output(kind="info", title="T", payload=None)
    assert "(no output)" in out


# --------------------------------------------------------------------- 171
def test_render_output_unprintable_payload():
    out = os_.render_output(kind="info", title=_BadTitle(),
                            payload=_BadStr(), max_chars=10)
    assert out == "(unprintable output)"


# --------------------------------------------------------------------- 205
def test_style_output_text_crash_returns_raw_text_truncated():
    with mock.patch.object(os_, "_render_body",
                           side_effect=RuntimeError("body boom")):
        out = os_.style_output_text("abcdefgh", kind="k", title="T",
                                    max_chars=3)
    assert out == "abc"


def test_style_output_text_healthy_control():
    out = os_.style_output_text("hello", kind="k", title="T")
    assert "hello" in out and "T" in out


# ------------------------------------------------------- non-recording
def test_module_has_no_logger():
    # The three guards bind nothing and record nothing; the module
    # carries no logging surface at all.
    assert not hasattr(os_, "_log")
