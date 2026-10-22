"""Local audio-captcha solver: the page's own audio, her own ears.

An audio captcha (reCAPTCHA's "audio challenge" fallback) is a
sound file and a box to type the words into. The transcription core
already existed — :func:`nomorals.tools.captcha_flow.solve_audio_local_first`
(local whisper.cpp first, paid service only after) — but nothing
could *reach* it from a live page: the audio payload lives inside
the challenge frame, and the rendered tab had no session-fetch
primitive. Pass #79 adds the primitive
(:meth:`nomorals.browser.service.RenderedTab.session_fetch_bytes`,
context-level request — session cookies ride along, no CORS gate)
and this solver around it:

    find the payload URL (challenge dict → frame documents → main
    document) → fetch its bytes through the session → transcribe
    locally → type the words into the response input → submit →
    read the outcome from the page.

Availability is probed honestly: the local path is whisper.cpp via
the ``pywhispercpp`` package (the exact backend
``solve_audio_local_first`` constructs); when the package is absent
the solver reports itself unavailable and routing falls through —
it does not fetch audio it cannot transcribe.
"""
from __future__ import annotations

import importlib.util
import re
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..tools.captcha import (
    CaptchaBackend,
    CaptchaKind,
    SolveResult,
    register_backend,
)
from .kind_solvers import register_kind_solver
from .tab_tools import docs_for_tab, press_key, type_answer

_log = get_logger(__name__)

__all__ = [
    "audio_asr_status",
    "find_audio_challenge",
    "AudioCaptchaSolver",
    "solve_audio_kind",
    "LocalAudioBackend",
    "register_local_backend",
]

_PAYLOAD_RE = re.compile(
    r"(?:https?://[^\s\"'<>]+)?/?recaptcha/api2/payload/audio[^\s\"'<>]*",
    re.I)
_AUDIO_TAG_RE = re.compile(r"<audio\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']",
                           re.I)
_DOWNLOAD_LINK_RE = re.compile(
    r"<a\b[^>]*href\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.I)
_INPUT_HINTS = ("audio", "response", "captcha", "answer")


def audio_asr_status() -> dict[str, Any]:
    """Can the local ASR path run at all right now?

    The probe matches the real backend: ``solve_audio_local_first``
    constructs ``voice.stt.WhisperCppBackend``, which needs the
    ``pywhispercpp`` package (its model loads at solve time).
    """
    have = importlib.util.find_spec("pywhispercpp") is not None
    return {
        "available": have,
        "backend": "whisper.cpp (pywhispercpp)",
        "detail": ("pywhispercpp present — local transcription ready "
                   "(model loads at solve time)" if have else
                   "local audio solving unavailable — pywhispercpp "
                   "not installed (pip install pywhispercpp)"),
    }


def _default_transcribe(audio: Any) -> dict[str, Any]:
    """The estate's own audio core: local ASR first, no service."""
    from ..tools.captcha_flow import solve_audio_local_first
    return solve_audio_local_first(audio)


def find_audio_challenge(tab: Any,
                         challenge: dict | None = None
                         ) -> dict[str, Any] | None:
    """Locate the audio challenge across the tab's documents.

    Returns ``{"audio_url", "audio_bytes", "input_selector",
    "frame_url", "page_url"}`` or None. The payload URL hides in
    the challenge frame's document (a download link), so frames are
    searched before the main document.
    """
    from .ocr_solver import find_answer_input
    if challenge:
        url = str(challenge.get("audio_url")
                  or challenge.get("image_url") or "")
        data = bytes(challenge.get("audio_bytes")
                     or challenge.get("image_bytes") or b"")
        if url or data:
            return {"audio_url": url, "audio_bytes": data,
                    "input_selector": "",
                    "frame_url": str(getattr(tab, "url", "") or ""),
                    "page_url": str(getattr(tab, "url", "") or "")}
    docs = docs_for_tab(tab)
    main_url = str(getattr(tab, "url", "") or "")
    ordered = docs[1:] + docs[:1]  # frames first: the payload hides there
    for doc in ordered:
        html = doc["html"] or ""
        if not html:
            continue
        url = ""
        m = _PAYLOAD_RE.search(html)
        if m:
            url = m.group(0)
            if url.startswith("/"):
                from urllib.parse import urljoin
                url = urljoin(doc["url"] or main_url, url)
        if not url:
            m = _AUDIO_TAG_RE.search(html)
            if m:
                from urllib.parse import urljoin
                url = urljoin(doc["url"] or main_url, m.group(1))
        if not url:
            for m in _DOWNLOAD_LINK_RE.finditer(html):
                href = m.group(1)
                if "audio" in href.lower() and "payload" in href.lower():
                    from urllib.parse import urljoin
                    url = urljoin(doc["url"] or main_url, href)
                    break
        if not url:
            continue
        return {"audio_url": url, "audio_bytes": b"",
                "input_selector": find_answer_input(html),
                "frame_url": doc["url"],
                "page_url": main_url}
    return None


_SET_VALUE_JS = """
({selector, text}) => {
  const el = document.querySelector(selector);
  if (!el) return false;
  const proto = el instanceof HTMLTextAreaElement
    ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
  setter.call(el, text);
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return true;
}
"""

_VERIFY_JS = """
() => {
  const sels = ['#recaptcha-verify-button', '#audio-verify-button',
                'button[id*=verify]', 'input[type=submit]'];
  for (const sel of sels) {
    const el = document.querySelector(sel);
    if (el) { el.click(); return sel; }
  }
  return '';
}
"""


class AudioCaptchaSolver:
    """Solve audio captchas on a live tab, in its own session."""

    def __init__(self, *,
                 transcribe_fn: Callable[[Any], Any] | None = None,
                 probe: Callable[[], tuple[bool, str]] | None = None,
                 sleeper: Callable[[float], None] | None = None) -> None:
        self._transcribe_fn = transcribe_fn
        self._probe = probe
        self._sleep = sleeper or time.sleep

    def available(self) -> tuple[bool, str]:
        if self._probe is not None:
            return self._probe()
        st = audio_asr_status()
        return bool(st["available"]), str(st["detail"])

    def transcribe(self, audio: Any) -> dict[str, Any]:
        fn = self._transcribe_fn or _default_transcribe
        try:
            return dict(fn(audio) or {})
        except Exception as exc:  # noqa: BLE001 - ASR failure is data
            return {"ok": False, "backend": "none", "text": "",
                    "detail": f"{type(exc).__name__}: {exc}"}

    def solve_tab(self, tab: Any, challenge: dict | None = None
                  ) -> dict[str, Any]:
        ok_avail, reason = self.available()
        if not ok_avail:
            return {"ok": False, "backend": "local_asr",
                    "detail": reason}
        info = find_audio_challenge(tab, challenge)
        if info is None:
            return {"ok": False, "backend": "local_asr",
                    "detail": "no audio challenge found on this page"}
        audio: Any = info["audio_bytes"]
        if not audio and info["audio_url"]:
            fetch = getattr(tab, "session_fetch_bytes", None)
            if not callable(fetch):
                return {"ok": False, "backend": "local_asr",
                        "detail": "audio payload URL found but the "
                                  "tab cannot fetch subresources "
                                  "through its session"}
            try:
                audio = bytes(fetch(info["audio_url"]))
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "backend": "local_asr",
                        "detail": f"could not fetch the audio "
                                  f"payload: {exc}"}
        if not audio:
            return {"ok": False, "backend": "local_asr",
                    "detail": "audio challenge found but no audio "
                              "bytes could be obtained"}
        res = self.transcribe(audio)
        text = str(res.get("text") or "").strip()
        if not text:
            return {"ok": False, "backend": "local_asr",
                    "detail": "local ASR produced no transcript — "
                              "falling through"}
        selector = info["input_selector"]
        main_url = str(getattr(tab, "url", "") or "")
        in_frame = bool(info["frame_url"]) and \
            info["frame_url"] != main_url
        typed = False
        if selector and in_frame:
            fn = getattr(tab, "frame_evaluate", None)
            if callable(fn):
                needle = info["frame_url"].split("?")[0][-48:]
                try:
                    typed = bool(fn(needle, _SET_VALUE_JS,
                                    {"selector": selector,
                                     "text": text}))
                    if typed:
                        fn(needle, _VERIFY_JS)
                except Exception:  # noqa: BLE001
                    typed = False
        elif selector:
            typed = type_answer(tab, selector, text)
            if typed:
                press_key(tab, "Enter")
        if not typed:
            return {"ok": False, "backend": "local_asr",
                    "detail": f"transcribed {text!r} but could not "
                              f"type it into the response input"}
        self._sleep(0.3)
        if find_audio_challenge(tab) is None:
            return {"ok": True, "backend": "whisper.cpp",
                    "detail": f"audio captcha cleared in-session: "
                              f"local ASR transcript typed",
                    "text": text}
        return {"ok": False, "backend": "whisper.cpp",
                "detail": "transcript typed but the audio challenge "
                          "is still present — likely misheard"}


def solve_audio_kind(tab: Any, challenge: dict) -> dict[str, Any]:
    """Kind-router entry point for ``audio_captcha``."""
    if tab is None:
        data = bytes((challenge or {}).get("audio_bytes")
                     or (challenge or {}).get("image_bytes") or b"")
        url = str((challenge or {}).get("audio_url")
                  or (challenge or {}).get("image_url") or "")
        if not data and not url:
            return {"ok": False, "backend": "local_asr",
                    "detail": "audio solving needs a live tab or "
                              "challenge audio bytes"}
        res = AudioCaptchaSolver().transcribe(data or url)
        return {"ok": bool(res.get("text")),
                "backend": str(res.get("backend") or "local_asr"),
                "text": str(res.get("text") or ""),
                "detail": "transcribed locally"
                          if res.get("text") else
                          "local ASR produced no transcript"}
    return AudioCaptchaSolver().solve_tab(tab, challenge)


# ── captcha backend registration (tools/captcha.py untouched) ────────────

class LocalAudioBackend(CaptchaBackend):
    """Local ASR as a captcha backend (``local_audio``)."""

    name = "local_audio"

    def __init__(self, settings: Any = None, *,
                 solver: AudioCaptchaSolver | None = None,
                 **_: Any) -> None:
        self._settings = settings
        self._solver = solver

    def available(self) -> bool:
        return bool(audio_asr_status()["available"])

    def solve(self, challenge: Any) -> SolveResult:
        started = time.time()
        kind = str(getattr(challenge, "kind", "") or "")
        if kind != CaptchaKind.AUDIO_CAPTCHA:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail=f"local audio backend solves audio captchas, "
                       f"not {kind!r}")
        solver = self._solver or AudioCaptchaSolver()
        audio: Any = bytes(getattr(challenge, "image_bytes", b"") or b"")
        if not audio:
            audio = str(getattr(challenge, "image_url", "") or "")
        if not audio:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail="audio challenge carries no audio bytes or URL")
        res = solver.transcribe(audio)
        text = str(res.get("text") or "")
        return SolveResult(
            ok=bool(text), kind=kind, backend=self.name, text=text,
            elapsed_ms=int((time.time() - started) * 1000),
            detail=("local ASR transcript" if text else
                    "local ASR produced no transcript"))


_REGISTERED = False


def register_local_backend() -> bool:
    """Register ``local_audio`` in the captcha backend registry."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        register_backend("local_audio", LocalAudioBackend)
        _REGISTERED = True
        return True
    except Exception as exc:  # noqa: BLE001 - registration is additive
        _log.warning("local audio backend registration failed: %s", exc)
        return False


register_local_backend()
register_kind_solver(CaptchaKind.AUDIO_CAPTCHA, solve_audio_kind)
