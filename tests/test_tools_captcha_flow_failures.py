"""Tail-82 failure pins — nomorals/tools/captcha_flow.py.

All six sites are EXCLUSIONS, probed live first
(nm-tail82-verify/probes.py, 22/22 behavior confirmations
across both batch modules):

- 286: OriginCostCeiling._load treats a corrupt or stale-day
  store as a fresh day. The estate-standard JSON store-loader
  contract (SuccessLedger._load in the same module reads the
  same way); zero-spend is also the genuine state every
  midnight rollover, so the failure value is not a false
  claim — and _load is read-only (pin asserts the corrupt
  file is left byte-identical). Tension: a mid-day corruption
  makes the day's real spend invisible to can_spend until the
  next record_spend rewrites the file whole; the remedy is
  observability only (shelved sketch in the tail-82 SURVEY).
- 371: _default_ocr_fn returns None when the browser estate's
  OCR core is unimportable — the documented "no estate, no
  default" rung (its docstring); None is the same value as
  "nothing registered", and stage 2 consumes it explicitly
  (`ocr_fn is not None`).
- 404: _local_asr's finally-block temp-file unlink is pure
  teardown — the transcription result is already computed.
- 505: _local_stage returns None when the browser spine import
  fails — None is the designed fall-through value, consumed
  explicitly by solve() (`if local_out is not None`); later
  stages run unchanged. ("Spine import is additive.")
- 598: slider detection in the passive stage is additive —
  its failure leaves `kind` untouched and the question
  detection + passive verdict still run.
- 608: question detection is the same additive rung.
"""
import json
import sys
from unittest import mock

from nomorals.tools import captcha_flow as cf


def test_286_corrupt_ceiling_store_loads_as_fresh_day(tmp_path):
    path = tmp_path / "ceiling.json"
    path.write_text("{corrupt json", "utf-8")
    ceiling = cf.OriginCostCeiling(path, daily_cap_usd=0.50)
    data = ceiling._load()
    assert data["origins"] == {}
    assert data["day"] == cf.OriginCostCeiling._today()
    assert ceiling.spent_today("https://x.example") == 0.0
    assert ceiling.can_spend("https://x.example")[0] is True
    # NOT-recording: the read never rewrites/repairs the file
    assert path.read_text("utf-8") == "{corrupt json"
    # a stale-day store is the same fresh-day state
    path.write_text(json.dumps(
        {"day": "1999-01-01",
         "origins": {"x.example": {"spend": 9.9, "solves": 3}}}), "utf-8")
    assert ceiling.spent_today("https://x.example") == 0.0


def test_371_default_ocr_fn_none_when_estate_unimportable():
    cf.register_default_ocr_fn(None)
    with mock.patch.dict(sys.modules,
                         {"nomorals.browser.ocr_solver": None}):
        assert cf._default_ocr_fn() is None
    sentinel = object()
    cf.register_default_ocr_fn(sentinel)
    try:
        assert cf._default_ocr_fn() is sentinel
    finally:
        cf.register_default_ocr_fn(None)
    from nomorals.browser.ocr_solver import flow_ocr_fn
    assert cf._default_ocr_fn() is flow_ocr_fn


class _FakeBackend:
    def transcribe(self, path, language="en"):
        return {"text": " 12345 "}


def test_404_local_asr_survives_tmp_unlink_failure():
    with mock.patch("nomorals.voice.stt.WhisperCppBackend", _FakeBackend):
        assert cf._local_asr(b"RIFF....") == "12345"
        with mock.patch("os.unlink", side_effect=OSError("locked")):
            assert cf._local_asr(b"RIFF....") == "12345"


def test_505_local_stage_falls_through_when_spine_unimportable():
    flow = cf.CaptchaFlow()
    with mock.patch.dict(sys.modules,
                         {"nomorals.browser.kind_solvers": None,
                          "nomorals.browser.captcha_estate": None}):
        out = flow._local_stage({"kind": "slider"}, kind="slider",
                                site="x.example", tried=[])
    assert out is None


_PLAIN_HTML = "<html><body>hello world, nothing here</body></html>"


def test_598_slider_detection_failure_keeps_passive_verdict():
    with mock.patch.dict(sys.modules,
                         {"nomorals.browser.slider_solver": None}):
        res = cf.CaptchaFlow().solve({"kind": "", "html": _PLAIN_HTML},
                                     url="https://x.example/")
    assert res["ok"] is True
    assert res["stage"] == "passive"
    assert res["kind"] == "no_challenge"


def test_608_question_detection_failure_keeps_passive_verdict():
    with mock.patch.dict(sys.modules,
                         {"nomorals.browser.question_solver": None}):
        res = cf.CaptchaFlow().solve({"kind": "", "html": _PLAIN_HTML},
                                     url="https://x.example/")
    assert res["ok"] is True
    assert res["stage"] == "passive"
    assert res["kind"] == "no_challenge"
