"""Tail-87 pins: nomorals/browser/challenge.py census sites.

All five are E104 guards in the wall-detection / solve-ladder
machinery, held as exclusions:

164/177/187 — the layered sub-detectors inside
``detect_challenge_signal`` (slider, question, token-captcha).
The primary detector (``errors.detect_challenge``) is deliberately
unguarded and runs first; each optional detector that crashes
degrades to "that detector found nothing" and the ladder moves to
the next detector — the module's documented "detection never
blocks" posture. Tension, stated plainly: if a page's ONLY wall
evidence is one kind and that kind's detector crashes, the
composite result is None and ``solve()`` reports
cleared/"no_challenge". The returned value is never falsified for
the detectors that did run; the remedy is observability only
(sibling guards in this module bind + ``_log.debug(exc_info=True)``
— a bind+debug sketch is shelved as a follow-up feature).
421 — ``_local_applicable`` registry crash -> False: the local
rung is skipped as ABSENT, not failed ("the ladder is unchanged");
the solve continues to the remaining strategies and takeover.
Fail-closed: the solver under-claims its own capability.
502 — ``_recheck`` unreadable page -> False: explicitly
documented "A recheck that errors is NOT a clear". Fail-closed —
a crash can never manufacture a cleared outcome.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import nomorals.browser.kind_solvers as kind_mod
import nomorals.browser.question_solver as question_mod
import nomorals.browser.slider_solver as slider_mod
import nomorals.tools.captcha as captcha_mod
from nomorals.browser.challenge import (
    ChallengeSignal,
    ChallengeSolver,
    detect_challenge_signal,
)

CLEAN_HTML = "<html><title>Hi</title><body>hello world</body></html>"


def _raise(exc):
    raise exc


# ── 164: slider detector crash ──────────────────────────────────

def test_164_slider_detector_crash_is_no_slider_wall(monkeypatch):
    monkeypatch.setattr(slider_mod, "detect_slider_challenge",
                        lambda html, url="": _raise(RuntimeError("boom")))
    # crash degrades to "no slider wall"; detection completes -> None
    assert detect_challenge_signal(CLEAN_HTML, "http://x/") is None
    # control: a working detector's hit is returned verbatim
    info = SimpleNamespace(vendor="acme", kind="slider",
                           evidence="drag handle")
    monkeypatch.setattr(slider_mod, "detect_slider_challenge",
                        lambda html, url="": info)
    sig = detect_challenge_signal(CLEAN_HTML, "http://x/")
    assert sig is not None and sig.detection == "slider-wall"


# ── 177: question detector crash ────────────────────────────────

def test_177_question_detector_crash_is_no_question_wall(monkeypatch):
    monkeypatch.setattr(slider_mod, "detect_slider_challenge",
                        lambda html, url="": None)
    monkeypatch.setattr(question_mod, "detect_question_challenge",
                        lambda html: _raise(RuntimeError("boom")))
    assert detect_challenge_signal(CLEAN_HTML, "http://x/") is None
    # control: a working detector's hit is returned verbatim
    monkeypatch.setattr(question_mod, "detect_question_challenge",
                        lambda html: {"question": "2+2?"})
    sig = detect_challenge_signal(CLEAN_HTML, "http://x/")
    assert sig is not None and sig.kind == question_mod.QUESTION_KIND


# ── 187: token-captcha detector crash ───────────────────────────

def test_187_captcha_detector_crash_is_no_captcha_wall(monkeypatch):
    monkeypatch.setattr(slider_mod, "detect_slider_challenge",
                        lambda html, url="": None)
    monkeypatch.setattr(question_mod, "detect_question_challenge",
                        lambda html: None)
    monkeypatch.setattr(captcha_mod, "detect",
                        lambda html, url="", **kw: _raise(RuntimeError("boom")))
    assert detect_challenge_signal(CLEAN_HTML, "http://x/") is None
    # control: a working detector's hit is returned verbatim
    monkeypatch.setattr(
        captcha_mod, "detect",
        lambda html, url="", **kw: [SimpleNamespace(kind="recaptcha")])
    sig = detect_challenge_signal(CLEAN_HTML, "http://x/")
    assert sig is not None and sig.detection == "captcha-wall"


# ── 421: solver-registry crash -> rung absent ───────────────────

def test_421_registry_crash_means_no_local_solver():
    solver = ChallengeSolver()
    sig = ChallengeSignal(kind="recaptcha", vendor="captcha",
                          detection="captcha-wall", url="http://x/",
                          evidence="pin")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kind_mod, "kind_solver_for",
                   lambda kind: _raise(RuntimeError("boom")))
        assert solver._local_applicable(sig) is False
    # control: a registered solver makes the rung applicable
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kind_mod, "kind_solver_for", lambda kind: (lambda *a: None))
        assert solver._local_applicable(sig) is True


# ── 502: unreadable page is not a clear ─────────────────────────

class _BadTab:
    url = "http://x/"
    title = "Hi"

    def html(self):
        raise RuntimeError("boom")


class _GoodTab:
    url = "http://x/"
    title = "Hi"

    def html(self):
        return CLEAN_HTML


def test_502_unreadable_page_is_not_cleared():
    solver = ChallengeSolver()
    assert solver._recheck(_BadTab(), "") is False
    # control: a readable, wall-free page IS a clear
    assert solver._recheck(_GoodTab(), "") is True
