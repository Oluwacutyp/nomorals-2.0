"""Local image-captcha solver: tesseract reads, vision confirms.

An image captcha is a picture of text and a box to type it into. The
estate's answer (Pass #79) is a real local mechanism, in two readers:

1. **Tesseract** — :func:`nomorals.vision.native.ocr_text` drives the
   tesseract binary directly (subprocess TSV) and returns the text
   WITH a mean word confidence (0–100). That confidence is the
   hand-off contract: at or above :data:`MIN_OCR_CONFIDENCE` the
   read stands on its own.
2. **Vision (the Seer)** — below the threshold, the image goes to
   the vision stack (:func:`nomorals.vision.seer.see_bytes`) with a
   strict transcription prompt. The Seer is text-only by design —
   perfect for reading, useless for coordinates, which is all this
   solver needs from it. When the Seer is unavailable the hand-off
   simply doesn't happen and a low-confidence tesseract read is
   reported as a failure, never typed on a guess.

Availability is probed, not assumed: no tesseract binary → the
solver reports itself unavailable with the reason, and routing
falls through to the service backend / owner takeover. (The
``documents.ocr`` stack is NOT the probe: its ``ocr_available()``
also demands pdf2image/poppler, which image captchas don't need;
its ``ocr_image`` returns bare text with no confidence, so the
confidence hand-off is defined on the native wrapper's
``mean_conf`` — the #79 survey correction to the #78 map.)

Three registrations, mirroring the slider solver: the
``image_captcha`` kind solver (in-session, for browser flows), the
``local_ocr`` captcha backend (solves from challenge image bytes —
no tab needed), and the default ``ocr_fn`` for
:mod:`nomorals.tools.captcha_flow` stage 2, so non-browser flows
finally get a real OCR stage instead of a silently skipped slot.
"""
from __future__ import annotations

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
    "MIN_OCR_CONFIDENCE",
    "ocr_status",
    "read_captcha_text",
    "detect_image_challenge",
    "ImageCaptchaSolver",
    "solve_image_kind",
    "flow_ocr_fn",
    "LocalOcrBackend",
    "register_local_backend",
]

#: Mean tesseract word confidence (0–100) at which a read stands on
#: its own; below it the vision stack gets the image instead.
MIN_OCR_CONFIDENCE = 55.0


# ── availability ─────────────────────────────────────────────────────────

def _tesseract_binary() -> str | None:
    try:
        from ..vision.native import _tesseract_binary as _probe
        return _probe()
    except Exception:  # noqa: BLE001 - probe failure = not available
        return None


def ocr_status() -> dict[str, Any]:
    """What the local OCR stack can do right now, honestly."""
    binary = _tesseract_binary()
    return {
        "available": binary is not None,
        "tesseract_binary": binary or "",
        "min_confidence": MIN_OCR_CONFIDENCE,
        "second_reader": "vision (Seer) — probed at solve time",
        "detail": ("tesseract binary found — local OCR ready"
                   if binary else
                   "local OCR unavailable — no tesseract binary "
                   "(install tesseract or set NM_OCR_BINARY)"),
    }


# ── readers ──────────────────────────────────────────────────────────────

def _native_read(data: bytes) -> dict[str, Any]:
    """Primary read: tesseract via the native wrapper.

    Tries the single-line segmentation first (captcha shape), then
    sparse-text. Returns ``{"text", "mean_conf"}``; raises the
    wrapper's own error when the binary/stack is missing.
    """
    from ..vision.native import ocr_text
    best: dict[str, Any] = {"text": "", "mean_conf": None}
    for psm in (7, 11, 6):
        out = ocr_text(data, psm=psm)
        text = str(out.get("text") or "").strip()
        if text:
            best = {"text": text, "mean_conf": out.get("mean_conf")}
            break
    return best


def _seer_read(data: bytes) -> str:
    """Second read: the vision stack transcribes the image."""
    from ..vision.seer import UNTRUSTED_VISION_PREFIX, see_bytes
    raw = see_bytes(
        data,
        "Transcribe the text in this captcha image exactly, character "
        "by character. Reply with ONLY the characters you see — no "
        "explanation, no quotes, no punctuation added.")
    text = str(raw or "").replace(UNTRUSTED_VISION_PREFIX, "").strip()
    # a vision answer is prose-shaped despite the prompt; the captcha
    # answer is its first line / first token-shaped run.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _normalize(text: str) -> str:
    """Captcha answers are single tokens: whitespace never survives."""
    return "".join(str(text or "").split())


def read_captcha_text(data: bytes, *,
                      read_fn: Callable[[bytes], Any] | None = None,
                      second_read_fn: Callable[[bytes], str] | None = None
                      ) -> dict[str, Any]:
    """Read a captcha image: tesseract first, vision on low confidence.

    Returns ``{"ok", "text", "confidence", "backend", "detail"}``.
    Never raises on bad input or missing stacks — absence is a
    result, not an exception.
    """
    if not data:
        return {"ok": False, "text": "", "confidence": None,
                "backend": "none", "detail": "no image bytes to read"}
    primary = read_fn or _native_read
    text, conf = "", None
    primary_error = ""
    try:
        out = primary(data) or {}
        if isinstance(out, dict):
            text = str(out.get("text") or "").strip()
            conf = out.get("mean_conf")
        else:
            text = str(out or "").strip()
    except Exception as exc:  # noqa: BLE001 - reader absence is data
        primary_error = f"{type(exc).__name__}: {exc}"
    conf_f = None
    if conf is not None:
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = None
    if text and (conf_f is None or conf_f >= MIN_OCR_CONFIDENCE):
        return {"ok": True, "text": _normalize(text),
                "confidence": conf_f, "backend": "tesseract",
                "detail": f"tesseract read at confidence "
                          f"{conf_f if conf_f is not None else 'n/a'}"}
    # Low confidence (or no read): the vision stack is the second reader.
    second = second_read_fn or _seer_read
    try:
        vision_text = _normalize(second(data))
    except Exception as exc:  # noqa: BLE001 - no second reader is data
        vision_text = ""
        vision_error = f"{type(exc).__name__}: {exc}"
    else:
        vision_error = ""
    if vision_text:
        return {"ok": True, "text": vision_text, "confidence": conf_f,
                "backend": "vision",
                "detail": f"tesseract confidence "
                          f"{conf_f if conf_f is not None else 'n/a'} "
                          f"below {MIN_OCR_CONFIDENCE:g} — vision read "
                          f"used instead"}
    reasons = []
    if primary_error:
        reasons.append(f"tesseract: {primary_error}")
    elif text:
        reasons.append(f"tesseract read {text!r} at confidence "
                       f"{conf_f} — below {MIN_OCR_CONFIDENCE:g}, "
                       f"not typed on a guess")
    else:
        reasons.append("tesseract read no text")
    if vision_error:
        reasons.append(f"vision: {vision_error}")
    else:
        reasons.append("vision read no text")
    return {"ok": False, "text": "", "confidence": conf_f,
            "backend": "none", "detail": "; ".join(reasons)}


# ── page-side detection (which image, which input) ───────────────────────

_TAG_RE = re.compile(r"<(img|input)\b([^>]*)>", re.I | re.S)
_ATTR_RE = re.compile(r"([\w-]+)\s*=\s*[\"']([^\"']*)[\"']")

_INPUT_HINTS = ("captcha", "answer", "code", "verify", "security",
                "human", "challenge")


def _attrs(raw: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in _ATTR_RE.finditer(raw)}


def _selector_for(attrs: dict[str, str], tag: str) -> str:
    if attrs.get("id"):
        return f"#{attrs['id']}"
    if attrs.get("name"):
        return f"{tag}[name='{attrs['name']}']"
    classes = (attrs.get("class") or "").split()
    if classes:
        return f"{tag}.{classes[0]}"
    return tag


def find_answer_input(html: str) -> str:
    """Selector of the captcha answer input in one document. "" = none."""
    fallback = ""
    for m in _TAG_RE.finditer(html or ""):
        if m.group(1).lower() != "input":
            continue
        attrs = _attrs(m.group(2))
        itype = (attrs.get("type") or "text").lower()
        if itype not in ("text", "", "number", "tel"):
            continue
        identity = " ".join(str(attrs.get(k) or "") for k in
                            ("name", "id", "placeholder", "aria-label",
                             "class")).lower()
        if any(h in identity for h in _INPUT_HINTS):
            return _selector_for(attrs, "input")
        if not fallback:
            fallback = _selector_for(attrs, "input")
    return fallback


def detect_image_challenge(tab: Any) -> dict[str, Any] | None:
    """Find the image-captcha on a live tab, across its documents.

    Returns ``{"image_bytes", "image_url", "image_selector",
    "input_selector", "frame_url", "page_url"}`` or None. Image
    bytes resolve in order: bytes the detector already decoded
    (data: URIs) → session fetch of the image URL (any origin) →
    element capture (main document only).
    """
    from ..tools.captcha import detect as _detect
    docs = docs_for_tab(tab)
    for doc in docs:
        html = doc["html"]
        if not html:
            continue
        try:
            found = _detect(html, doc["url"])
        except Exception:  # noqa: BLE001 - detection never blocks
            found = []
        for ch in found:
            if ch.kind != CaptchaKind.IMAGE_CAPTCHA:
                continue
            img_bytes = bytes(ch.image_bytes or b"")
            img_selector = ""
            if not img_bytes and ch.image_url:
                fetch = getattr(tab, "session_fetch_bytes", None)
                if callable(fetch):
                    try:
                        img_bytes = bytes(fetch(ch.image_url))
                    except Exception:  # noqa: BLE001 - try capture next
                        img_bytes = b""
            if not img_bytes and doc["url"] == str(
                    getattr(tab, "url", "") or ""):
                for m in _TAG_RE.finditer(html):
                    if m.group(1).lower() != "img":
                        continue
                    attrs = _attrs(m.group(2))
                    src = attrs.get("src") or ""
                    if ch.image_url and src and \
                            src.split("?")[0] not in ch.image_url and \
                            ch.image_url.split("?")[0] not in src:
                        continue
                    img_selector = _selector_for(attrs, "img")
                    break
                if img_selector:
                    try:
                        img_bytes = bytes(
                            tab.element_image_bytes(img_selector))
                    except Exception:  # noqa: BLE001 - no bytes, no solve
                        img_bytes = b""
            input_selector = find_answer_input(html)
            if not input_selector:
                input_selector = find_answer_input(docs[0]["html"])
            return {"image_bytes": img_bytes,
                    "image_url": ch.image_url or "",
                    "image_selector": img_selector,
                    "input_selector": input_selector,
                    "frame_url": doc["url"],
                    "page_url": str(getattr(tab, "url", "") or "")}
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


def _type_in_frame(tab: Any, frame_url: str, selector: str,
                   text: str) -> bool:
    """Type into an input that lives inside a frame document."""
    fn = getattr(tab, "frame_evaluate", None)
    if not callable(fn) or not frame_url:
        return False
    # the needle: enough of the frame URL to pick it out uniquely
    needle = frame_url.split("?")[0][-48:]
    try:
        return bool(fn(needle, _SET_VALUE_JS,
                       {"selector": selector, "text": text}))
    except Exception:  # noqa: BLE001 - frame typing failure is data
        return False


def _submit_in_frame(tab: Any, frame_url: str, selector: str) -> bool:
    fn = getattr(tab, "frame_evaluate", None)
    if not callable(fn) or not frame_url:
        return False
    needle = frame_url.split("?")[0][-48:]
    js = """
    (selector) => {
      const el = document.querySelector(selector);
      const form = el && el.form;
      if (form) { form.requestSubmit ? form.requestSubmit()
                                     : form.submit(); return true; }
      return false;
    }
    """
    try:
        return bool(fn(needle, js, selector))
    except Exception:  # noqa: BLE001
        return False


# ── the solver ───────────────────────────────────────────────────────────

class ImageCaptchaSolver:
    """Solve image captchas on a live tab, in its own session."""

    def __init__(self, *, read_fn: Callable[[bytes], Any] | None = None,
                 second_read_fn: Callable[[bytes], str] | None = None,
                 probe: Callable[[], tuple[bool, str]] | None = None,
                 sleeper: Callable[[float], None] | None = None) -> None:
        self._read_fn = read_fn
        self._second_read_fn = second_read_fn
        self._probe = probe
        self._sleep = sleeper or time.sleep

    def available(self) -> tuple[bool, str]:
        if self._probe is not None:
            return self._probe()
        st = ocr_status()
        return bool(st["available"]), str(st["detail"])

    def solve_tab(self, tab: Any, challenge: dict | None = None
                  ) -> dict[str, Any]:
        ok_avail, reason = self.available()
        if not ok_avail:
            return {"ok": False, "backend": "local_ocr",
                    "detail": reason}
        info = None
        pre_bytes = b""
        pre_url = ""
        if challenge:
            pre_bytes = bytes(challenge.get("image_bytes") or b"")
            pre_url = str(challenge.get("image_url") or "")
        if pre_bytes or pre_url:
            info = {"image_bytes": pre_bytes, "image_url": pre_url,
                    "image_selector": "",
                    "input_selector": find_answer_input(tab.html()),
                    "frame_url": str(getattr(tab, "url", "") or ""),
                    "page_url": str(getattr(tab, "url", "") or "")}
        if info is None or (not info["image_bytes"] and not pre_url):
            info = detect_image_challenge(tab)
        if info is None:
            return {"ok": False, "backend": "local_ocr",
                    "detail": "no image captcha detected on this page"}
        if not info["image_bytes"]:
            return {"ok": False, "backend": "local_ocr",
                    "detail": "captcha image found but its bytes could "
                              "not be captured in-session"}
        if not info["input_selector"]:
            return {"ok": False, "backend": "local_ocr",
                    "detail": "captcha image found but no answer "
                              "input could be located"}
        read = read_captcha_text(info["image_bytes"],
                                 read_fn=self._read_fn,
                                 second_read_fn=self._second_read_fn)
        if not read["ok"]:
            return {"ok": False, "backend": "local_ocr",
                    "detail": read["detail"]}
        text = read["text"]
        main_url = str(getattr(tab, "url", "") or "")
        in_frame = bool(info["frame_url"]) and \
            info["frame_url"] != main_url
        if in_frame:
            typed = _type_in_frame(tab, info["frame_url"],
                                   info["input_selector"], text)
            if typed:
                _submit_in_frame(tab, info["frame_url"],
                                 info["input_selector"])
        else:
            typed = type_answer(tab, info["input_selector"], text)
            if typed:
                press_key(tab, "Enter")
        if not typed:
            return {"ok": False, "backend": "local_ocr",
                    "detail": f"read {text!r} but could not type it "
                              f"into {info['input_selector']!r}"}
        self._sleep(0.3)
        if detect_image_challenge(tab) is None:
            return {"ok": True, "backend": read["backend"],
                    "detail": f"image captcha cleared in-session "
                              f"({read['detail']})",
                    "confidence": read["confidence"]}
        return {"ok": False, "backend": read["backend"],
                "detail": "answer typed but the captcha is still "
                          "present — the read was likely wrong",
                "confidence": read["confidence"]}


def solve_image_kind(tab: Any, challenge: dict) -> dict[str, Any]:
    """Kind-router entry point for ``image_captcha``."""
    if tab is None:
        # Offline shape: bytes on the challenge are still readable —
        # the caller (a form flow) types the text itself.
        data = bytes((challenge or {}).get("image_bytes") or b"")
        if data:
            read = read_captcha_text(data)
            return {"ok": read["ok"], "backend": read["backend"],
                    "text": read["text"], "detail": read["detail"]}
        return {"ok": False, "backend": "local_ocr",
                "detail": "image solving needs a live tab or "
                          "challenge image bytes"}
    return ImageCaptchaSolver().solve_tab(tab, challenge)


def flow_ocr_fn(ch: dict[str, Any]) -> dict[str, Any]:
    """The default ``ocr_fn`` for captcha_flow stage 2.

    Works on the challenge dict alone (image bytes, or the image URL
    fetched through captcha.py's own fetcher) — the non-browser
    flows' OCR core.
    """
    data = bytes(ch.get("image_bytes") or b"")
    if not data and ch.get("image_url"):
        try:
            from ..tools.captcha import fetch_image_bytes
            data = fetch_image_bytes(str(ch["image_url"]))
        except Exception as exc:  # noqa: BLE001 - no bytes, honest fail
            return {"ok": False, "text": "",
                    "detail": f"could not fetch captcha image: {exc}"}
    read = read_captcha_text(data)
    return {"ok": read["ok"], "text": read["text"],
            "backend": read["backend"],
            "confidence": read["confidence"],
            "detail": read["detail"]}


# ── captcha backend registration (tools/captcha.py untouched) ────────────

class LocalOcrBackend(CaptchaBackend):
    """Local OCR as a captcha backend (``local_ocr``).

    Solves ``image_captcha`` challenges from their bytes — the
    pipeline shape of the same two readers the tab solver uses.
    """

    name = "local_ocr"

    def __init__(self, settings: Any = None, *,
                 solver: ImageCaptchaSolver | None = None,
                 **_: Any) -> None:
        self._settings = settings
        self._solver = solver

    def available(self) -> bool:
        solver = self._solver or ImageCaptchaSolver()
        return bool(solver.available()[0])

    def solve(self, challenge: Any) -> SolveResult:
        started = time.time()
        kind = str(getattr(challenge, "kind", "") or "")
        if kind != CaptchaKind.IMAGE_CAPTCHA:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail=f"local OCR backend solves image captchas, "
                       f"not {kind!r}")
        data = bytes(getattr(challenge, "image_bytes", b"") or b"")
        if not data and getattr(challenge, "image_url", ""):
            try:
                from ..tools.captcha import fetch_image_bytes
                data = fetch_image_bytes(str(challenge.image_url))
            except Exception as exc:  # noqa: BLE001
                return SolveResult(
                    ok=False, kind=kind, backend=self.name,
                    detail=f"could not fetch captcha image: {exc}")
        read = read_captcha_text(
            data,
            read_fn=(self._solver._read_fn if self._solver else None),
            second_read_fn=(self._solver._second_read_fn
                            if self._solver else None))
        return SolveResult(
            ok=read["ok"], kind=kind, backend=self.name,
            text=read["text"] if read["ok"] else "",
            elapsed_ms=int((time.time() - started) * 1000),
            detail=read["detail"])


_REGISTERED = False


def register_local_backend() -> bool:
    """Register ``local_ocr`` + the flow default. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        register_backend("local_ocr", LocalOcrBackend)
        from ..tools.captcha_flow import register_default_ocr_fn
        register_default_ocr_fn(flow_ocr_fn)
        _REGISTERED = True
        return True
    except Exception as exc:  # noqa: BLE001 - registration is additive
        _log.warning("local OCR registration failed: %s", exc)
        return False


register_local_backend()
register_kind_solver(CaptchaKind.IMAGE_CAPTCHA, solve_image_kind)
