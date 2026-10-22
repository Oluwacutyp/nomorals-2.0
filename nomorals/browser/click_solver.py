"""Local click-selection solver: vision picks the cells, the tab clicks.

reCAPTCHA v2 image grids, hCaptcha click challenges and GeeTest v4
click-word/icon widgets share one shape: a picture, an instruction
("select all images with buses"), and cells to click. There is no
token to buy and no slider to drag — the solve is *grounding*:
which cells match the instruction?

The grounding contract here is strict JSON from the vision brain
(:meth:`nomorals.llm.brain.Brain.describe_image` — the same
normalized 0–1000 coordinate convention as
:func:`nomorals.tools.vision.locate`): grid geometry, the matching
cells, and the verify button's position. The Seer is deliberately
NOT used — its system prompt forbids coordinates by design. If the
vision stack cannot ground (no vision-capable provider, unparseable
reply), the solver clicks NOTHING and says so; a guessed click is
worse than an honest fall-through to the service/takeover rungs.

The frame story, proven against the tab primitives: the grid lives
in a cross-origin iframe whose DOM the main document never shows —
but the iframe *element* is in the parent DOM, so its composited
pixels are captured with ``element_image_bytes`` on the iframe
itself, and mouse clicks at viewport coordinates land inside the
frame without any frame switching. Frame documents are read via
``frames_html()`` for the prompt text and the outcome.

A fresh grid after a verify (new image, no error) is a RETRY with
the new image — dynamic challenges replace tiles as you click —
capped at :data:`MAX_GRID_ROUNDS`. An error banner from the widget,
or a grid that comes back byte-identical, is a rejection: a
failure, reported as one.

GeeTest routing: kind ``geetest`` is a DISPATCHER — slider markup
goes to the slider solver (Pass #78), click markup comes here;
this module's registration replaces the slider's own ``geetest``
registration, which is why it imports the slider module first and
delegates back to it.
"""
from __future__ import annotations

import hashlib
import json
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
from . import slider_solver as _slider  # noqa: F401 - base registration first
from .kind_solvers import register_kind_solver
from .tab_tools import click_px, docs_for_tab

_log = get_logger(__name__)

__all__ = [
    "MAX_GRID_ROUNDS",
    "ClickChallengeInfo",
    "detect_click_challenge",
    "grounding_status",
    "ClickSolver",
    "solve_click_kind",
    "solve_geetest_kind",
    "LocalClickBackend",
    "register_local_backend",
]

#: Initial round + retries on freshly issued grids.
MAX_GRID_ROUNDS = 3

_REJECTION_MARKERS = (
    "please select all", "please try again", "incorrect",
    "verification failed", "invalid captcha", "please also check",
)

_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_IFRAME_RE = re.compile(r"<iframe\b([^>]*)>", re.I | re.S)
_ATTR_RE = re.compile(r"([\w-]+)\s*=\s*[\"']([^\"']*)[\"']")


# ── detection ────────────────────────────────────────────────────────────

class ClickChallengeInfo:
    def __init__(self, *, vendor: str, prompt: str,
                 widget_selector: str, frame_url: str = "") -> None:
        self.vendor = vendor
        self.prompt = prompt
        self.widget_selector = widget_selector
        self.frame_url = frame_url

    def to_dict(self) -> dict[str, Any]:
        return {"vendor": self.vendor, "prompt": self.prompt,
                "widget_selector": self.widget_selector,
                "frame_url": self.frame_url}


def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_STRIP_RE.sub(" ", html or "")).strip()


def _prompt_from(html: str, patterns: tuple[str, ...]) -> str:
    for pat in patterns:
        m = re.search(pat, html or "", re.I | re.S)
        if m:
            text = _strip_tags(m.group(1))
            if text:
                return text[:300]
    return ""


def _iframe_selector(main_html: str, *src_tokens: str) -> str:
    """Selector for the iframe element carrying the challenge frame."""
    for m in _IFRAME_RE.finditer(main_html or ""):
        attrs = {a.group(1).lower(): a.group(2)
                 for a in _ATTR_RE.finditer(m.group(1))}
        src = attrs.get("src") or ""
        if src_tokens and not any(t in src for t in src_tokens):
            continue
        if attrs.get("id"):
            return f"#{attrs['id']}"
        if attrs.get("name"):
            return f"iframe[name='{attrs['name']}']"
        for token in src_tokens:
            if token in src:
                return f"iframe[src*='{token}']"
        classes = (attrs.get("class") or "").split()
        if classes:
            return f"iframe.{classes[0]}"
        return "iframe"
    return ""


def detect_click_challenge(docs: list[dict[str, str]]
                           ) -> ClickChallengeInfo | None:
    """Find a click-selection widget across the page's documents.

    ``docs`` is :func:`nomorals.browser.tab_tools.docs_for_tab`
    output (main document first). Grid documents are frames for
    reCAPTCHA/hCaptcha; GeeTest click widgets live in the main
    document. Returns None when no click widget is present — a
    checkbox-only widget is a token flow, not this solver's.
    """
    if not docs:
        return None
    main_html = docs[0]["html"] or ""
    for doc in docs:
        html = doc["html"] or ""
        low = html.lower()
        is_frame = doc["url"] != docs[0]["url"]
        # — reCAPTCHA v2 image grid (inside the bframe document) —
        if "rc-imageselect" in low or "rc-image-tile-wrapper" in low:
            prompt = _prompt_from(html, (
                r"rc-imageselect-desc-no-canonical[^>]*>(.*?)</div>",
                r"rc-imageselect-desc[^>]*>(.*?)</div>",
                r"rc-imageselect-instructions[^>]*>(.*?)</div>",
            )) or "Select all matching images"
            selector = _iframe_selector(main_html, "bframe", "recaptcha")
            if selector:
                return ClickChallengeInfo(
                    vendor="recaptcha", prompt=prompt,
                    widget_selector=selector, frame_url=doc["url"])
        # — hCaptcha click challenge (inside its challenge frame) —
        if ("task-grid" in low or "h-captcha" in low) and \
                ("prompt" in low or "task" in low) and is_frame:
            prompt = _prompt_from(html, (
                r"prompt-text[^>]*>(.*?)</",
                r"<h2[^>]*>(.*?)</h2>",
            )) or "Click the matching images"
            selector = _iframe_selector(main_html, "hcaptcha", "newassets")
            if selector:
                return ClickChallengeInfo(
                    vendor="hcaptcha", prompt=prompt,
                    widget_selector=selector, frame_url=doc["url"])
        # — GeeTest v4 click (main document widget) —
        if not is_frame and "geetest" in low and (
                "geetest_item" in low or "geetest_square" in low
                or "geetest_click" in low):
            prompt = _prompt_from(html, (
                r"geetest_tip[^>]*>(.*?)</",
                r"(Please click[^<]*)",
            )) or "Click the matching icons in order"
            selector = ""
            m = re.search(r"class\s*=\s*[\"']([^\"']*geetest_holder"
                          r"[^\"']*)[\"']", html, re.I)
            if m:
                selector = "." + m.group(1).split()[0]
            else:
                m = re.search(r"class\s*=\s*[\"']([^\"']*geetest_box"
                              r"[^\"']*)[\"']", html, re.I)
                if m:
                    selector = "." + m.group(1).split()[0]
            if selector:
                return ClickChallengeInfo(
                    vendor="geetest", prompt=prompt,
                    widget_selector=selector, frame_url="")
    return None


# ── grounding ────────────────────────────────────────────────────────────

_GROUND_PROMPT = """You are solving a click-selection captcha.
Instruction: {prompt}

Look at the widget image. It contains a grid of image cells plus,
usually, a prompt strip and a verify button. Reply with ONLY a JSON
object, no other text:
{{"rows": <int>, "cols": <int>,
  "grid": {{"x": <0-1000>, "y": <0-1000>, "w": <0-1000>, "h": <0-1000>}},
  "cells": [{{"row": <0-based int>, "col": <0-based int>}}, ...],
  "verify": {{"x": <0-1000>, "y": <0-1000>}} or null}}
"grid" is the bounding box of the cell area only (not the prompt or
button), in 0-1000 normalized coordinates of this image. "cells"
lists EVERY cell matching the instruction. "verify" is the centre
of the verify/submit button if one is visible, else null. If you
cannot tell which cells match, reply {{"cells": []}}."""


def grounding_status() -> dict[str, Any]:
    """Can the vision brain be reached for grounding right now?

    Structural probe: the brain resolves without invoking a
    provider. Whether a vision-capable provider answers is only
    knowable at solve time — a grounder failure there is reported
    as unavailability with zero clicks, never papered over.
    """
    try:
        from ..llm.brain import brain_for
        brain_for(None)
        return {"available": True,
                "detail": "vision brain resolvable — provider "
                          "support confirmed on first solve"}
    except Exception as exc:  # noqa: BLE001 - no brain, no grounding
        return {"available": False,
                "detail": f"vision brain unresolvable: {exc}"}


def _parse_grounding(text: str) -> dict[str, Any] | None:
    """Parse the grounder's JSON reply. None = unusable reply."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        rows = int(data.get("rows") or 0)
        cols = int(data.get("cols") or 0)
        grid = data.get("grid") or {}
        gx, gy = float(grid["x"]), float(grid["y"])
        gw, gh = float(grid["w"]), float(grid["h"])
        cells = []
        for c in data.get("cells") or []:
            r, col = int(c["row"]), int(c["col"])
            if 0 <= r < rows and 0 <= col < cols:
                cells.append((r, col))
        verify = data.get("verify")
        vpoint = None
        if isinstance(verify, dict):
            vpoint = (float(verify["x"]), float(verify["y"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not (2 <= rows <= 6 and 2 <= cols <= 6):
        return None
    if gw <= 0 or gh <= 0 or not cells:
        return None
    # dedupe, keep order
    seen: set[tuple[int, int]] = set()
    unique = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return {"rows": rows, "cols": cols,
            "grid": {"x": gx, "y": gy, "w": gw, "h": gh},
            "cells": unique, "verify": vpoint}


def _brain_ground(image: bytes, prompt: str) -> dict[str, Any] | None:
    """Default grounder: the vision brain, strict JSON out."""
    from ..llm.brain import brain_for
    brain = brain_for(None)
    resp = brain.describe_image(
        image, _GROUND_PROMPT.format(prompt=prompt or ""))
    if getattr(resp, "error", ""):
        raise RuntimeError(f"vision brain: {resp.error}")
    return _parse_grounding(str(getattr(resp, "text", "") or ""))


_VERIFY_JS = """
() => {
  const sels = ['#recaptcha-verify-button', '.rc-button-default',
                'button[id*=verify]', 'button[class*=verify]',
                '.verify-button', 'input[type=submit]'];
  for (const sel of sels) {
    const el = document.querySelector(sel);
    if (el) { el.click(); return sel; }
  }
  return '';
}
"""


# ── the solver ───────────────────────────────────────────────────────────

class ClickSolver:
    """Solve click-selection challenges on a live tab."""

    def __init__(self, *,
                 ground_fn: Callable[[bytes, str], Any] | None = None,
                 probe: Callable[[], tuple[bool, str]] | None = None,
                 sleeper: Callable[[float], None] | None = None,
                 max_rounds: int = MAX_GRID_ROUNDS) -> None:
        self._ground_fn = ground_fn
        self._probe = probe
        self._sleep = sleeper or time.sleep
        self._max_rounds = max(1, int(max_rounds))

    def available(self) -> tuple[bool, str]:
        if self._probe is not None:
            return self._probe()
        st = grounding_status()
        return bool(st["available"]), str(st["detail"])

    def _ground(self, image: bytes, prompt: str) -> dict[str, Any] | None:
        fn = self._ground_fn or _brain_ground
        return fn(image, prompt)

    def solve_tab(self, tab: Any, challenge: dict | None = None
                  ) -> dict[str, Any]:
        ok_avail, reason = self.available()
        if not ok_avail:
            return {"ok": False, "backend": "local_click",
                    "detail": reason, "rounds": 0}
        info = detect_click_challenge(docs_for_tab(tab))
        if info is None:
            return {"ok": False, "backend": "local_click",
                    "detail": "no click-selection widget on this "
                              "page (a checkbox-only widget is a "
                              "token flow, not a grid)",
                    "rounds": 0}
        for round_no in range(1, self._max_rounds + 1):
            try:
                image = bytes(tab.element_image_bytes(
                    info.widget_selector))
                box = tab.element_box(info.widget_selector)
            except Exception as exc:  # noqa: BLE001 - capture is data
                return {"ok": False, "backend": "local_click",
                        "detail": f"could not capture the widget: "
                                  f"{exc}",
                        "rounds": round_no - 1}
            try:
                ground = self._ground(image, info.prompt)
            except Exception as exc:  # noqa: BLE001 - no ground, no clicks
                return {"ok": False, "backend": "local_click",
                        "detail": f"vision grounding failed: {exc}",
                        "rounds": round_no - 1}
            if ground is None:
                return {"ok": False, "backend": "local_click",
                        "detail": "vision grounding returned nothing "
                                  "usable — no cells clicked",
                        "rounds": round_no - 1}
            grid = ground["grid"]
            bw, bh = float(box["width"]), float(box["height"])
            bx, by = float(box["x"]), float(box["y"])
            for (r, c) in ground["cells"]:
                fx = (grid["x"] + grid["w"] * (c + 0.5)
                      / ground["cols"]) / 1000.0
                fy = (grid["y"] + grid["h"] * (r + 0.5)
                      / ground["rows"]) / 1000.0
                if not click_px(tab, bx + fx * bw, by + fy * bh):
                    return {"ok": False, "backend": "local_click",
                            "detail": "the tab cannot click at "
                                      "computed cell points",
                            "rounds": round_no}
                self._sleep(0.12)
            verify_method = ""
            if ground.get("verify"):
                vx, vy = ground["verify"]
                if click_px(tab, bx + vx / 1000.0 * bw,
                            by + vy / 1000.0 * bh):
                    verify_method = "vision point"
            if not verify_method and info.frame_url:
                fn = getattr(tab, "frame_evaluate", None)
                if callable(fn):
                    needle = info.frame_url.split("?")[0][-48:]
                    try:
                        if fn(needle, _VERIFY_JS):
                            verify_method = "frame button"
                    except Exception:  # noqa: BLE001 - settle decides
                        pass
            self._sleep(0.4)
            docs2 = docs_for_tab(tab)
            info2 = detect_click_challenge(docs2)
            if info2 is None:
                return {"ok": True, "backend": "local_click",
                        "detail": f"click challenge cleared in-session "
                                  f"in {round_no} round(s) "
                                  f"({info.vendor}, verify via "
                                  f"{verify_method or 'widget'})",
                        "rounds": round_no}
            grid_html = ""
            for doc in docs2:
                if doc["url"] == info2.frame_url or \
                        info2.vendor == "geetest":
                    grid_html = (doc["html"] or "").lower()
                    break
            if any(m in grid_html for m in _REJECTION_MARKERS):
                return {"ok": False, "backend": "local_click",
                        "detail": "the widget rejected the selection "
                                  "(error banner after verify)",
                        "rounds": round_no}
            try:
                image2 = bytes(tab.element_image_bytes(
                    info2.widget_selector))
            except Exception:  # noqa: BLE001
                image2 = b""
            hash2 = hashlib.sha256(image2).hexdigest() if image2 else ""
            hash1 = hashlib.sha256(image).hexdigest()
            if hash2 == hash1 and info2.prompt == info.prompt:
                return {"ok": False, "backend": "local_click",
                        "detail": "grid unchanged after verify — "
                                  "the attempt was rejected",
                        "rounds": round_no}
            # A fresh grid (new image and/or new prompt): retry with it.
            info = info2
        return {"ok": False, "backend": "local_click",
                "detail": f"still challenged after {self._max_rounds} "
                          f"grid round(s) — attempt cap reached",
                "rounds": self._max_rounds}


def solve_click_kind(tab: Any, challenge: dict) -> dict[str, Any]:
    """Kind-router entry point for ``recaptcha_v2`` / ``hcaptcha``."""
    if tab is None:
        return {"ok": False, "backend": "local_click",
                "detail": "click-selection solving needs a live tab"}
    return ClickSolver().solve_tab(tab, challenge)


def solve_geetest_kind(tab: Any, challenge: dict) -> dict[str, Any]:
    """Kind-router entry point for ``geetest`` — the dispatcher.

    Slider markup → the Pass #78 slider solver. Click markup → the
    click solver. Anything else (v4 behavioral with no widget yet)
    is an honest local failure; the service/takeover rungs decide
    what happens next.
    """
    if tab is None:
        return {"ok": False, "backend": "local_click",
                "detail": "geetest solving needs a live tab"}
    try:
        html = str(tab.html() or "")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "backend": "local_click",
                "detail": f"could not read the page: {exc}"}
    if _slider.detect_slider_challenge(html, str(
            getattr(tab, "url", "") or "")) is not None:
        return _slider.solve_slider_kind(tab, challenge)
    info = detect_click_challenge(docs_for_tab(tab))
    if info is not None and info.vendor == "geetest":
        return ClickSolver().solve_tab(tab, challenge)
    return {"ok": False, "backend": "local_click",
            "detail": "geetest wall is neither a slider nor a "
                      "click widget this estate recognises"}


# ── captcha backend registration (tools/captcha.py untouched) ────────────

class LocalClickBackend(CaptchaBackend):
    """Vision-grounded clicking as a captcha backend (``local_click``).

    Needs the live tab in ``metadata["session"]`` — a grid cannot
    be clicked from a sitekey.
    """

    name = "local_click"
    _KINDS = (CaptchaKind.RECAPTCHA_V2, CaptchaKind.HCAPTCHA,
              CaptchaKind.GEETEST)

    def __init__(self, settings: Any = None, *,
                 solver: ClickSolver | None = None,
                 **_: Any) -> None:
        self._settings = settings
        self._solver = solver

    def available(self) -> bool:
        solver = self._solver or ClickSolver()
        return bool(solver.available()[0])

    def solve(self, challenge: Any) -> SolveResult:
        started = time.time()
        kind = str(getattr(challenge, "kind", "") or "")
        if kind not in self._KINDS:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail=f"local click backend solves click-selection "
                       f"challenges, not {kind!r}")
        metadata = getattr(challenge, "metadata", None) or {}
        session = metadata.get("session") or metadata.get("tab")
        if session is None:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail="click solving needs the live browser session "
                       "in challenge.metadata['session']")
        solver = self._solver or ClickSolver()
        if kind == CaptchaKind.GEETEST:
            out = solve_geetest_kind(
                session, {"kind": kind,
                          "page_url": getattr(challenge, "page_url",
                                              "")})
        else:
            out = solver.solve_tab(session)
        return SolveResult(
            ok=bool(out.get("ok")), kind=kind, backend=self.name,
            elapsed_ms=int((time.time() - started) * 1000),
            detail=str(out.get("detail") or ""))


_REGISTERED = False


def register_local_backend() -> bool:
    """Register ``local_click`` in the captcha backend registry."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        register_backend("local_click", LocalClickBackend)
        _REGISTERED = True
        return True
    except Exception as exc:  # noqa: BLE001 - registration is additive
        _log.warning("local click backend registration failed: %s", exc)
        return False


register_local_backend()
register_kind_solver(CaptchaKind.RECAPTCHA_V2, solve_click_kind)
register_kind_solver(CaptchaKind.HCAPTCHA, solve_click_kind)
# The geetest dispatcher replaces the slider module's own geetest
# registration (slider markup still routes to the slider solver).
register_kind_solver(CaptchaKind.GEETEST, solve_geetest_kind)
