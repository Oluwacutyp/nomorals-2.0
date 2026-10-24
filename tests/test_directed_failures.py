"""Pins for Pass #106: media/directed failure ledger.

The directed engine is best-effort by design — backend probes,
pipeline loads, and score parses fall back to safe defaults instead
of raising. These pins prove the fallbacks now leave a trace: the
fallback value is unchanged AND the failure is recorded under its
site. Ledger state is cleared per test; sys.modules injections are
restored so no test leaks into the next.
"""

import os
import sys
import tempfile

import pytest

from nomorals.media.directed import failures
from nomorals.media.directed.failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures()]


def test_ledger_basics_and_order():
    record_failure("a.site", ValueError("bad"))
    record_failure("b.site", RuntimeError("worse"))
    assert failure_count() == 2
    recent = recent_failures()
    assert recent[0].site == "b.site"  # newest first
    assert recent[1].site == "a.site"
    assert recent[0].exc_type == "RuntimeError"
    assert recent[0].message == "worse"


def test_ledger_bounded_ring():
    for _ in range(250):
        record_failure("s", ValueError("x"))
    assert failure_count() == 250
    assert len(recent_failures(limit=1000)) == 200


def test_unprintable_exception_never_raises():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    record_failure("bad.str", BadStr())
    assert failure_count() == 1
    assert "unprintable" in recent_failures()[0].message


def test_format_empty_and_render():
    assert format_failures() == "no directed failures recorded."
    record_failure("ai_edit.faceswap_status", ImportError("no ort"))
    rendered = format_failures()
    assert "ai_edit.faceswap_status" in rendered
    assert "ImportError" in rendered
    assert "1 total" in rendered


def test_package_all_unchanged():
    """The #102 lesson: no __init__ surface change in this pass."""
    import nomorals.media.directed as directed

    assert directed.__all__ == ["pose_rig", "animator", "camera"]


class TestStatusProbesRecord:
    """Backend probes fail closed AND record, via import blocking.

    sys.modules None-injection makes `import torch` / `import
    onnxruntime` raise ImportError deterministically, whether or not
    the package is installed on the machine running the test.
    """

    @staticmethod
    def _block(*names):
        saved = {}
        for name in names:
            saved[name] = sys.modules.get(name, "ABSENT")
            sys.modules[name] = None
        return saved

    @staticmethod
    def _restore(saved):
        for name, value in saved.items():
            if value == "ABSENT":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def test_faceswap_status_records(self):
        from nomorals.media.directed import ai_edit

        saved = self._block("onnxruntime")
        try:
            status = ai_edit.faceswap_status()
        finally:
            self._restore(saved)
        assert status["available"] is False
        assert "ai_edit.faceswap_status" in _sites()

    def test_expression_status_records(self):
        from nomorals.media.directed import ai_edit

        saved = self._block("torch")
        try:
            status = ai_edit.expression_status()
        finally:
            self._restore(saved)
        assert status["available"] is False
        assert "ai_edit.expression_status" in _sites()

    def test_mimicmotion_status_records(self):
        from nomorals.media.directed import animator

        saved = self._block("torch")
        try:
            status = animator.mimicmotion_status()
        finally:
            self._restore(saved)
        assert status["torch"] is False
        assert "animator.mimicmotion_status" in _sites()

    def test_kimodo_status_records(self):
        from nomorals.media.directed import motion_score

        saved = self._block("torch")
        try:
            status = motion_score.kimodo_status()
        finally:
            self._restore(saved)
        assert status["torch"] is False
        assert "motion_score.kimodo_status" in _sites()

    def test_has_torch_cuda_records(self):
        from nomorals.media.directed import lipsync

        saved = self._block("torch")
        try:
            result = lipsync._has_torch_cuda()
        finally:
            self._restore(saved)
        assert result == (False, False)
        assert "lipsync._has_torch_cuda" in _sites()


def test_finger_spread_parse_records_and_valid_still_parses():
    from nomorals.media.directed import motion_score

    fingers, spread, tips = motion_score._parse_finger_state(
        {"spread": "not-a-number"})
    assert (fingers, spread, tips) == (None, None, None)
    assert "motion_score._parse_finger_state" in _sites()

    clear_failures()
    _, spread, _ = motion_score._parse_finger_state({"spread": 1.5})
    assert spread == 1.5
    assert failure_count() == 0


def test_find_pipeline_garbage_checkpoint_records():
    from nomorals.media.directed import filler

    with tempfile.NamedTemporaryFile(
            suffix=".safetensors", delete=False) as tf:
        tf.write(b"garbage")
        path = tf.name
    old = os.environ.get("DEVON_SD_CKPT")
    os.environ["DEVON_SD_CKPT"] = path
    try:
        assert filler._find_pipeline() is None
    finally:
        if old is None:
            os.environ.pop("DEVON_SD_CKPT", None)
        else:
            os.environ["DEVON_SD_CKPT"] = old
        os.unlink(path)
    assert "filler._find_pipeline" in _sites()


def test_error_scan_subsystem_clean():
    from nomorals.tools.error_scan import scan

    report = scan(["nomorals/media/directed"])
    assert report.findings == [], report.findings
