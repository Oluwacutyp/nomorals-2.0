"""Failure pins for nomorals/browser/slider_solver.py (tail-69).

Every site is an adjudicated EXCLUSION: the module's documented
posture is "absence is a result, not an exception" — probes, decoder
rungs, pacing bookkeeping and status snapshots degrade to the honest
unavailable / None / {} value while the decision gates that matter
(limiter.check, gap confidence, capture verdicts) all bind their
exceptions into named outcomes. Each pin proves the guarded operation
was attempted (spy), the degraded value that follows, an honest
control, and that nothing was recorded on the browser failure ledger.
"""
import builtins
import io
from unittest import mock

import numpy as np
from PIL import Image

from nomorals.browser import failures as browser_failures
from nomorals.browser import slider_solver as ss

_REAL_IMPORT = builtins.__import__


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (6, 4), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


# 101 — _probe: an import that raises is an unusable module -> False,
# the probe's contract ("absence is data, not an error"). Consumers
# (vision_status and everything above it) report absence by name.
def test_101_probe_failure_is_absence():
    before = browser_failures.failure_count()

    def raising_import(name, *a, **k):
        if name == "boommod":
            raise RuntimeError("import boom")
        return _REAL_IMPORT(name, *a, **k)

    with mock.patch("builtins.__import__", raising_import):
        assert ss._probe("boommod") is False
    assert ss._probe("os") is True  # honest control
    assert ss._probe("definitely_missing_mod_xyz") is False
    assert browser_failures.failure_count() == before


# 143 — _decode_image PIL rung: PIL raises -> the ladder falls
# through to the next decoder instead of failing the decode.
def test_143_pil_failure_falls_to_next_decoder():
    before = browser_failures.failure_count()
    png = _png_bytes()
    pil_calls = []

    def pil_boom(*a, **k):
        pil_calls.append(1)
        raise RuntimeError("pil boom")

    def no_cv2(name, *a, **k):
        if name == "cv2":
            raise ImportError("no cv2")
        return _REAL_IMPORT(name, *a, **k)

    with mock.patch.object(Image, "open", pil_boom), \
            mock.patch("builtins.__import__", no_cv2):
        assert ss._decode_image(png) is None  # no decoder left
    assert pil_calls  # the PIL rung was really attempted
    out = ss._decode_image(png)
    assert out is not None and out[0].shape == (4, 6, 3)  # control
    assert browser_failures.failure_count() == before


# 159 — _decode_image terminal rung: no decoder can read the bytes
# -> None, the documented "undecodable" result that detect_gap and
# _attempt_once report by name.
def test_159_undecodable_bytes_are_none():
    before = browser_failures.failure_count()
    assert ss._decode_image(b"not an image") is None
    assert ss._decode_image(b"") is None
    out = ss._decode_image(_png_bytes())
    assert out is not None and out[0].shape == (4, 6, 3)  # control
    assert browser_failures.failure_count() == before


class FakeLimiter:
    def __init__(self, *, submit_exc=None, record_exc=None):
        self.submit_calls = 0
        self.success_calls = 0
        self.failure_calls = 0
        self.submit_exc = submit_exc
        self.record_exc = record_exc

    def check(self):
        return None

    def record_submit(self):
        self.submit_calls += 1
        if self.submit_exc is not None:
            raise self.submit_exc

    def record_success(self):
        self.success_calls += 1
        if self.record_exc is not None:
            raise self.record_exc

    def record_failure(self):
        self.failure_calls += 1
        if self.record_exc is not None:
            raise self.record_exc


INFO = ss.SliderChallengeInfo(
    vendor="generic", background_selector=".bg", handle_selector=".h",
    shape="single_image")


def _solver(limiter):
    return ss.LocalSliderSolver(limiter=limiter,
                                vision_check=lambda: (True, "ok"))


def _ok_attempt(self, tab, info, n):
    return ss.SliderOutcome(True, attempts=1, detail="cleared")


# 752 — solve_tab: limiter.record_submit raises -> swallowed ("pacing
# never sinks a solve"); the gate check() above it is the decision,
# and the production limiter's record paths are internally guarded.
def test_752_record_submit_failure_never_sinks_solve():
    before = browser_failures.failure_count()
    lim = FakeLimiter(submit_exc=RuntimeError("submit boom"))
    with mock.patch.object(ss.LocalSliderSolver, "_attempt_once",
                           _ok_attempt):
        res = _solver(lim).solve_tab(object(), info=INFO)
    assert lim.submit_calls == 1  # the stamp was really attempted
    assert res.ok is True
    lim2 = FakeLimiter()
    with mock.patch.object(ss.LocalSliderSolver, "_attempt_once",
                           _ok_attempt):
        res2 = _solver(lim2).solve_tab(object(), info=INFO)
    assert res2.ok is True and lim2.submit_calls == 1  # honest control
    assert browser_failures.failure_count() == before


# 797 — _record: limiter.record_success raises -> swallowed; the
# solve outcome was already decided and is returned regardless.
def test_797_record_outcome_failure_never_sinks_solve():
    before = browser_failures.failure_count()
    lim = FakeLimiter(record_exc=RuntimeError("record boom"))
    with mock.patch.object(ss.LocalSliderSolver, "_attempt_once",
                           _ok_attempt):
        res = _solver(lim).solve_tab(object(), info=INFO)
    assert lim.success_calls == 1  # recording was really attempted
    assert res.ok is True
    assert browser_failures.failure_count() == before


# 1015 — local_slider_status: the rate-limit snapshot raises -> that
# one sub-key degrades to {}; availability/vision honesty fields are
# computed outside the guard and stay true.
def test_1015_rate_limit_snapshot_degrades_to_empty():
    before = browser_failures.failure_count()
    with mock.patch.object(ss, "rate_limiter",
                           side_effect=RuntimeError("rl boom")):
        st = ss.local_slider_status()
    assert st["rate_limit"] == {}
    assert st["backend"] == "local_slider"
    assert st["available"] is True
    st2 = ss.local_slider_status()
    assert "submits_last_24h" in st2["rate_limit"]  # honest control
    assert browser_failures.failure_count() == before
