"""Failure pins for nomorals/browser/ocr_solver.py (tail-68).

Every site is an adjudicated EXCLUSION: the module's contract is
"absence is a result, not an exception" — a probe, fetch, capture or
frame failure degrades to the honest unavailable / no-bytes / False
value, and every decision surface above it (ocr_status, solve_tab,
the backend SolveResult) reports that outcome by name instead of
claiming success. Each pin proves the guarded operation was
attempted (spy), the degraded value that follows, and that nothing
was recorded on the browser failure ledger.
"""
from types import SimpleNamespace

from nomorals.browser import failures as browser_failures
from nomorals.browser import ocr_solver as ocr
from nomorals.tools.captcha import CaptchaKind

HTML = ('<html><body><img src="https://x.test/cap.png">'
        '<input name="captcha_answer" type="text"></body></html>')
CH = SimpleNamespace(kind=CaptchaKind.IMAGE_CAPTCHA, image_bytes=b"",
                     image_url="https://x.test/cap.png")


class FakeTab:
    url = "https://x.test/page"

    def html(self):
        return HTML


# 76 — _tesseract_binary: the availability probe raises -> None =
# "not available", the documented probe contract; ocr_status reports
# unavailable and routing falls through to the next backend.
def test_76_probe_failure_is_unavailable(monkeypatch):
    before = browser_failures.failure_count()
    calls = []

    def raising_probe():
        calls.append(1)
        raise RuntimeError("probe boom")

    monkeypatch.setattr("nomorals.vision.native._tesseract_binary",
                        raising_probe)
    assert ocr._tesseract_binary() is None
    assert calls  # the probe was really attempted
    status = ocr.ocr_status()
    assert status["available"] is False
    assert status["tesseract_binary"] == ""
    monkeypatch.setattr("nomorals.vision.native._tesseract_binary",
                        lambda: "/usr/bin/tesseract")
    status2 = ocr.ocr_status()
    assert status2["available"] is True  # honest control
    assert status2["tesseract_binary"] == "/usr/bin/tesseract"
    assert browser_failures.failure_count() == before


# 265 — detect_image_challenge: captcha.detect raises on one document
# -> that document contributes no challenges ("detection never
# blocks"); with no other document the result is None, the same value
# as a page with no captcha.
def test_265_detect_failure_skips_document(monkeypatch):
    before = browser_failures.failure_count()
    calls = []

    def raising_detect(html, url):
        calls.append((html, url))
        raise RuntimeError("detect boom")

    monkeypatch.setattr("nomorals.tools.captcha.detect", raising_detect)
    assert ocr.detect_image_challenge(FakeTab()) is None
    assert len(calls) == 1  # detection was really attempted
    challenge = SimpleNamespace(kind=CaptchaKind.IMAGE_CAPTCHA,
                                image_bytes=b"IMG", image_url="")
    monkeypatch.setattr("nomorals.tools.captcha.detect",
                        lambda html, url: [challenge])
    out = ocr.detect_image_challenge(FakeTab())
    assert out is not None and out["image_bytes"] == b"IMG"
    assert browser_failures.failure_count() == before


class FetchTab(FakeTab):
    def __init__(self):
        self.fetch_calls, self.capture_calls = [], []

    def session_fetch_bytes(self, url):
        self.fetch_calls.append(url)
        raise RuntimeError("fetch boom")

    def element_image_bytes(self, selector):
        self.capture_calls.append(selector)
        return b"CAP"


# 277 — detect_image_challenge: the session fetch of the image URL
# raises -> the next rung of the documented resolution order
# (element capture) serves the bytes instead.
def test_277_fetch_failure_falls_to_capture(monkeypatch):
    before = browser_failures.failure_count()
    tab = FetchTab()
    monkeypatch.setattr("nomorals.tools.captcha.detect",
                        lambda html, url: [CH])
    out = ocr.detect_image_challenge(tab)
    assert tab.fetch_calls == ["https://x.test/cap.png"]
    assert tab.capture_calls == ["img"]  # capture attempted next
    assert out is not None and out["image_bytes"] == b"CAP"
    assert browser_failures.failure_count() == before


class CaptureBoomTab(FakeTab):
    def __init__(self):
        self.capture_calls = []

    def element_image_bytes(self, selector):
        self.capture_calls.append(selector)
        raise RuntimeError("capture boom")


# 296 — detect_image_challenge: element capture raises -> the bytes
# stay empty; the challenge is still returned and solve_tab refuses
# honestly ("bytes could not be captured") rather than solving blind.
def test_296_capture_failure_leaves_empty_bytes(monkeypatch):
    before = browser_failures.failure_count()
    tab = CaptureBoomTab()  # no session_fetch_bytes: capture is the rung
    monkeypatch.setattr("nomorals.tools.captcha.detect",
                        lambda html, url: [CH])
    out = ocr.detect_image_challenge(tab)
    assert tab.capture_calls == ["img"]  # capture was really attempted
    assert out is not None and out["image_bytes"] == b""
    assert browser_failures.failure_count() == before


class FrameTab:
    def __init__(self, exc=None):
        self.exc, self.calls = exc, []

    def frame_evaluate(self, needle, js, arg=None):
        self.calls.append((needle, js, arg))
        if self.exc is not None:
            raise self.exc
        return True


# 336 — _type_in_frame: frame_evaluate raises -> False ("frame
# typing failure is data"); solve_tab turns False into an honest
# ok=False ("could not type it"), never a claimed success.
def test_336_frame_typing_failure_is_false():
    before = browser_failures.failure_count()
    tab = FrameTab(exc=RuntimeError("frame boom"))
    got = ocr._type_in_frame(tab, "https://x.test/frame?a=1", "#inp",
                             "txt")
    assert got is False
    assert len(tab.calls) == 1  # typing was really attempted
    assert tab.calls[0][0] == "https://x.test/frame"  # the needle
    assert ocr._type_in_frame(FrameTab(), "https://x.test/frame",
                              "#inp", "t") is True  # honest control
    assert browser_failures.failure_count() == before


# 356 — _submit_in_frame: frame_evaluate raises -> False. solve_tab
# discards this return, but its success is decided by re-detecting
# the challenge afterwards, so a failed submit cannot read as a
# cleared captcha.
def test_356_frame_submit_failure_is_false():
    before = browser_failures.failure_count()
    tab = FrameTab(exc=RuntimeError("submit boom"))
    got = ocr._submit_in_frame(tab, "https://x.test/frame?a=1", "#inp")
    assert got is False
    assert len(tab.calls) == 1  # submit was really attempted
    assert ocr._submit_in_frame(FrameTab(), "https://x.test/frame",
                                "#inp") is True  # honest control
    assert browser_failures.failure_count() == before
