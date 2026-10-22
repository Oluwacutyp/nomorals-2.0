"""Shared live-tab helpers for the per-kind captcha solvers.

The kind solvers (slider, OCR, audio, click, question, Cloudflare)
all drive the same duck-typed tab protocol that
:mod:`nomorals.browser.slider_solver` established — ``html()``,
``url``, ``element_image_bytes``, ``element_box``, ``mouse_drag_path``
— plus the Pass #79 primitives (``click_at_px``,
``session_fetch_bytes``, ``frames_html``, ``frame_evaluate``). This
module holds the small shared moves so each solver doesn't grow its
own copy:

* :func:`docs_for_tab` — the page as a list of documents (main frame
  first, then every readable frame). Challenge widgets live in
  iframes; a solver that only reads the main document is blind to
  them.
* :func:`click_px` — one real mouse click at CSS pixels, through
  whichever click primitive the tab actually has.
* :func:`type_answer` — type text into an element: the tab's own
  human-cadence ``type_text`` when it takes a target, else a click to
  focus + focused typing.
* :func:`press_key` — best-effort single key press.

Everything here is probe-and-degrade: a tab that lacks a primitive
gets an honest ``False``/empty result, never an exception leaking
into a solver and never a faked success.
"""
from __future__ import annotations

from typing import Any

__all__ = ["docs_for_tab", "click_px", "type_answer", "press_key"]


def docs_for_tab(tab: Any) -> list[dict[str, str]]:
    """The page as documents: main frame first, then its frames.

    Returns ``[{"url", "html"}, ...]``. The main document always
    comes from ``tab.html()``; frames come from ``tab.frames_html()``
    when the tab can read them (its main-frame entry is de-duplicated
    against the ``html()`` one). A tab with no frame primitive yields
    just the main document.
    """
    docs: list[dict[str, str]] = []
    try:
        main_html = str(tab.html() or "")
    except Exception:  # noqa: BLE001 - unreadable page = no documents
        main_html = ""
    main_url = str(getattr(tab, "url", "") or "")
    docs.append({"url": main_url, "html": main_html})
    frames_fn = getattr(tab, "frames_html", None)
    if callable(frames_fn):
        try:
            frames = frames_fn() or []
        except Exception:  # noqa: BLE001 - frames are a bonus view
            frames = []
        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            html = str(frame.get("html") or "")
            url = str(frame.get("url") or "")
            if i == 0 and url == main_url and html == main_html:
                continue  # frames_html's own main-frame entry
            docs.append({"url": url, "html": html})
    return docs


def click_px(tab: Any, x: float, y: float) -> bool:
    """One real mouse click at viewport CSS pixels. False = the tab
    cannot click there (no primitive), never a silent pretend."""
    fn = getattr(tab, "click_at_px", None)
    if callable(fn):
        try:
            fn(float(x), float(y))
            return True
        except Exception:  # noqa: BLE001 - fall through to other hands
            pass
    # Normalized-click tabs (takeover-style): convert via the viewport.
    click_norm = getattr(tab, "click_at", None) or \
        getattr(tab, "click_norm", None)
    evaluate = getattr(tab, "evaluate", None)
    if callable(click_norm) and callable(evaluate):
        try:
            size = evaluate(
                "() => ({w: window.innerWidth, h: window.innerHeight})")
            w = float(size["w"])
            h = float(size["h"])
            if w > 0 and h > 0:
                click_norm(float(x) / w, float(y) / h)
                return True
        except Exception:  # noqa: BLE001 - no other hand available
            return False
    return False


def type_answer(tab: Any, selector: str, text: str) -> bool:
    """Type ``text`` into the element ``selector`` names.

    Prefers the tab's targeted ``type_text(target, text)`` (human
    cadence, verified read-back on a real tab). Falls back to
    clicking the element's centre and typing into focus — the shape
    the takeover adapter exposes. False when neither hand exists or
    the typing itself failed.
    """
    text = str(text or "")
    if not text or not selector:
        return False
    type_text = getattr(tab, "type_text", None)
    if callable(type_text):
        try:
            type_text(selector, text)
            return True
        except TypeError:
            pass  # single-arg (focused) type_text — use the fallback
        except Exception:  # noqa: BLE001 - typing failed; try fallback
            pass
    box_fn = getattr(tab, "element_box", None)
    focused = getattr(tab, "type_text", None)
    if callable(box_fn):
        try:
            box = box_fn(selector)
            cx = float(box["x"]) + float(box["width"]) / 2.0
            cy = float(box["y"]) + float(box["height"]) / 2.0
            if click_px(tab, cx, cy) and callable(focused):
                focused(text)
                return True
        except Exception:  # noqa: BLE001 - no other hand available
            return False
    return False


def press_key(tab: Any, key: str) -> bool:
    """Press one key (``Enter`` submits most captcha forms)."""
    fn = getattr(tab, "press_key", None)
    if not callable(fn):
        return False
    try:
        fn(key)
        return True
    except Exception:  # noqa: BLE001 - a failed key press is data
        return False
