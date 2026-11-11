"""Navigation-aware page pipeline (§18/§20 god-tier).

One pipeline behind /webshot and /webrec: open the URL in
Devon's real browser session (a persistent profile, so
cookies and gate clearances survive across captures) →
settle → run the interstitial pass (cookie banners, consent
walls, age gates) → work the bot check (managed wait-out,
checkbox click, owner takeover for hard walls) → capture
or record.  The session's presentation is a ladder of
coherent device fingerprints (mobile first): a wall or a
gate that refuses one fingerprint rotates to the next,
and owner takeover is the last rung, not the first
response.  A page that refuses to clear is reported
honestly — the gate or the challenge page is never handed
back as if it were the result.

The detection half is pure (HTML in, verdicts out) and the
driving half sits behind the PageProbe seam, so the whole
pass is pinned offline with a scripted probe; the real
Chromium suite drives the same code through Playwright.

Layer: social/chat is L4; the browser organs it reuses
(challenge detection, wall typing) live in L4's browser/
package — no new layering pairs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_PRESENTATION",
    "PRESENTATIONS",
    "GateAction",
    "GateBlocked",
    "GateCandidate",
    "GateMemory",
    "NavError",
    "NavOutcome",
    "NavSession",
    "NavTabAdapter",
    "NavUnavailable",
    "PageProbe",
    "RecordOutcome",
    "ScrollPlan",
    "TakeoverAsk",
    "TakeoverBridge",
    "TakeoverHook",
    "format_takeover_message",
    "WallBlocked",
    "apply_cookies_to_live",
    "classify_wall",
    "clear_cookies_from_live",
    "context_kwargs",
    "detect_gates",
    "imported_cookies_path",
    "max_nav_sessions",
    "navigate",
    "open_with_ladder",
    "presentation_ladder",
    "playwright_cookie_payload",
    "read_imported_cookie_records",
    "record_page",
    "run_interstitial_pass",
    "run_wall_pass",
    "scroll_plan",
]

# Bounded budgets — a capture never waits forever.
SETTLE_BUDGET_S = 8.0
MANAGED_BUDGET_S = 25.0
TAKEOVER_BUDGET_S = 120.0
REC_MIN_S = 5.0
REC_MAX_S = 60.0

StepHook = Callable[[str], None]


# ── errors ────────────────────────────────────────────────


class NavError(RuntimeError):
    """The navigation pipeline failed honestly."""


class NavUnavailable(NavError):
    """No browser engine is available on this host.  The
    /webshot ladder may still fall back to provider seeds;
    a gate/wall refusal is NOT this error."""


class GateBlocked(NavError):
    """An interstitial gate could not be cleared."""

    def __init__(self, message: str, *,
                 gates: list[GateCandidate] | None = None):
        super().__init__(message)
        self.gates = list(gates or [])


class WallBlocked(NavError):
    """A bot wall could not be cleared (or needs a hand
    that never came)."""

    def __init__(self, message: str, *, wall: str = "",
                 attempts: list[str] | None = None):
        super().__init__(message)
        self.wall = wall
        self.attempts = list(attempts or [])


# ── interstitial gates: detection (pure) ──────────────────


@dataclass
class GateCandidate:
    """One detected gate: what kind, which control clears
    it, and the hint the driver needs to click it."""

    kind: str                # cookie | consent | age | age_dob
    label: str               # the control's visible text
    selector: str = ""       # css hint, when the page gives one
    host: str = ""
    #: The control sits in the layer that is actually
    #: BLOCKING the page (a modal / overlay / dialog),
    #: not in an inline banner or a footer — the pass
    #: works the blocking layer's control first, because
    #: a covered control cannot be clicked at all.
    blocking: bool = False


# Age-confirmation wording (matched case-insensitively).
_AGE_LABELS = (
    "i'm 18 or older", "i am 18 or older", "i'm over 18",
    "i am over 18", "18 or older - enter", "18+ enter",
    "enter - i am 18", "yes, i am 18", "yes i'm 18",
    "i confirm i am 18", "agree and enter",
)
# Bare labels that only count as an age gate when the page
# itself carries age markers.
_AGE_BARE_LABELS = ("enter", "yes", "continue", "i agree")
_AGE_MARKERS = (
    "18 or older", "over 18", "age verification",
    "verify your age", "confirm your age", "adult website",
    "adult content", "sexually explicit", "you must be 18",
    "must be at least 18", "age-gate", "age_gate", "agegate",
    "date of birth", "dob",
)
# Known per-site shapes where the generic heuristics need
# help — seeds beside the heuristics, never a closed list.
SITE_AGE_LABELS: dict[str, tuple[str, ...]] = {
    "pornhub.com": ("i'm 18 or older", "i am 18 or older",
                    "18 or older - enter", "enter"),
}

_CONSENT_MARKERS = (
    # Strong CMP/consent signals only: the bare word
    # "consent" also appears in cookie names and scripts
    # on plain cookie banners, which are cookie gates.
    "onetrust", "qc-cmp", "sp_message", "trustarc",
    "gdpr", "we value your privacy",
    "your privacy choices", "tcf", "privacy preference",
    "consent manager", "consent banner", "consent wall",
)
_CONSENT_LABELS = (
    "accept all", "accept all cookies", "i accept",
    "accept & continue", "accept and continue", "agree",
    "i agree", "allow all", "allow all cookies", "got it",
    "ok", "continue",
)
_COOKIE_MARKERS = (
    "cookie", "we use cookies", "this site uses cookies",
    "this website uses cookies",
)
_COOKIE_LABELS = (
    "accept all cookies", "accept cookies", "accept all",
    "allow cookies", "allow all cookies", "i accept",
    "got it", "ok", "agree",
)


_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img",
    "input", "link", "meta", "source", "track", "wbr",
})
_LAYER_TOKENS = frozenset({
    "modal", "overlay", "dialog", "popup", "lightbox",
    "interstitial", "wall", "gate", "fullscreen", "agegate",
})


def _is_layer(a: dict[str, str]) -> bool:
    """True when an element presents as the layer BLOCKING
    the page: a dialog/modal by role, by class/id shape,
    or a fixed/absolute full-viewport box by style.  An
    inline banner (a bar across one edge) is not a layer —
    the content behind it is reachable."""
    if (a.get("role") or "").lower() == "dialog":
        return True
    if (a.get("aria-modal") or "").lower() == "true":
        return True
    ident = f"{a.get('class', '')} {a.get('id', '')}".lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", ident) if t]
    if any(t in _LAYER_TOKENS or "modal" in t
           or "overlay" in t for t in tokens):
        return True
    style = (a.get("style") or "").lower().replace(" ", "")
    return (
        ("position:fixed" in style
         or "position:absolute" in style)
        and ("inset:0" in style or (
            "top:0" in style and "left:0" in style
            and "bottom:0" in style)))


class _ClickScanner(HTMLParser):
    """Collects clickable controls (buttons, links, inputs)
    with their text, classes and ids — and whether each
    sits inside the blocking layer (see :func:`_is_layer`),
    so the pass can aim at the control that actually
    covers the page instead of the first gate-shaped
    button in DOM order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.controls: list[dict[str, str]] = []
        self.forms_with_dob = False
        self._open: dict[str, str] | None = None
        self._text: list[str] = []
        self._stack: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]
                        ) -> None:
        a = {k: (v or "") for k, v in attrs}
        layer = _is_layer(a)
        if tag in ("button", "a") or (
                tag == "input"
                and a.get("type", "").lower()
                in ("button", "submit", "")):
            self._open = {
                "tag": tag,
                "id": a.get("id", ""),
                "class": a.get("class", ""),
                "role": a.get("role", ""),
                "value": a.get("value", ""),
                "text": "",
                "blocking": "1" if layer or any(
                    up for _, up in self._stack) else "",
            }
            self._text = []
            if tag == "input" and a.get("value"):
                self._finish()
        if tag not in _VOID_TAGS:
            self._stack.append((tag, layer))
        if tag == "input":
            hint = " ".join((a.get("name", ""), a.get("id", ""),
                             a.get("placeholder", ""))).lower()
            if any(k in hint for k in
                   ("birth", "dob", "age", "year")) and \
                    a.get("type", "").lower() in (
                        "date", "number", "text", ""):
                self.forms_with_dob = True

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._open is not None and \
                tag == self._open["tag"]:
            self._finish()
        # Pop the element stack up to the matching open
        # tag (tolerant of the mis-nesting real pages ship).
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    def _finish(self) -> None:
        assert self._open is not None
        self._open["text"] = " ".join(
            " ".join(self._text).split())
        self.controls.append(self._open)
        self._open = None
        self._text = []


def _selector_for(control: dict[str, str]) -> str:
    if control.get("id"):
        return f"#{control['id']}"
    classes = (control.get("class") or "").split()
    if classes:
        return "." + ".".join(classes[:2])
    return ""


def _host_of(url: str) -> str:
    with suppress(Exception):
        return (urlsplit(url).hostname or "").lower()
    return ""


def detect_gates(html: str, url: str = "") -> list[GateCandidate]:
    """Detect interstitial gates on a page.  Pure: the same
    HTML always gives the same candidates.  Order matters:
    the BLOCKING layer's control first (a covered control
    cannot be clicked at all), and within a tier the
    age-gate confirm outranks generic consent/cookie
    controls — a page can stack them, and the pass works
    them in this order."""
    text = str(html or "")
    low = text.lower()
    host = _host_of(url)
    scanner = _ClickScanner()
    # A torn page scans as whatever was collected.
    with suppress(Exception):
        scanner.feed(text)
    controls = scanner.controls

    def norm(s: str) -> str:
        # Fold typographic quotes/dashes: real gate buttons
        # ship "I’m 18 or older" with a curly apostrophe,
        # which must still read as the age confirm it is.
        s = str(s).lower()
        for fancy, plain in (("’", "'"), ("‘", "'"),
                             ("“", '"'), ("”", '"'),
                             ("–", "-"), ("—", "-")):
            s = s.replace(fancy, plain)
        return " ".join(s.split())

    labels = [(c, norm(c["text"] or c["value"]))
              for c in controls]

    by_kind: dict[str, GateCandidate] = {}

    def add(kind: str, control: dict[str, str],
            label: str) -> None:
        cand = GateCandidate(
            kind=kind, label=label,
            selector=_selector_for(control), host=host,
            blocking=bool(control.get("blocking")))
        old = by_kind.get(kind)
        # One candidate per kind; the blocking layer's
        # control replaces a same-kind control that merely
        # sits on the page (a footer age link is not the
        # age gate when a modal is standing).
        if old is None or (cand.blocking
                           and not old.blocking):
            by_kind[kind] = cand

    # ── age gates ──
    age_context = any(m in low for m in _AGE_MARKERS)
    site_labels = SITE_AGE_LABELS.get(host, ())
    for control, label in labels:
        if not label:
            continue
        if label in _AGE_LABELS or label in site_labels or \
                any(a in label for a in _AGE_LABELS) or (
                age_context and label in _AGE_BARE_LABELS):
            add("age", control, label)
    if "age" not in by_kind and age_context and \
            scanner.forms_with_dob:
        # A date-of-birth form is an age gate Devon will not
        # invent an answer for.
        by_kind["age_dob"] = GateCandidate(
            kind="age_dob", label="date-of-birth form",
            host=host)

    # ── consent walls ──
    consent_context = any(m in low for m in _CONSENT_MARKERS)
    if consent_context:
        for control, label in labels:
            if label in _CONSENT_LABELS or \
                    label.startswith("accept"):
                add("consent", control, label)

    # ── cookie banners ──
    cookie_context = any(m in low for m in _COOKIE_MARKERS)
    if cookie_context and "consent" not in by_kind:
        for control, label in labels:
            if label in _COOKIE_LABELS:
                add("cookie", control, label)

    order = {"age": 0, "age_dob": 0, "consent": 1, "cookie": 2}
    found = sorted(
        by_kind.values(),
        key=lambda g: (not g.blocking,
                       order.get(g.kind, 3)))
    return found


# ── bot walls: typing (pure) ──────────────────────────────

_CHALLENGE_TEXT = re.compile(
    r"(verify you are human|verify that you are human|"
    r"are you a robot|i'm not a robot|prove you are human|"
    r"security check|bot check|unusual traffic|"
    r"just a moment|checking your browser|attention required)",
    re.IGNORECASE)
_IMAGE_GRID = re.compile(
    r"(rc-imageselect|imageselect|image-grid|"
    r"select all images|click (on )?all images|"
    r"hcaptcha-challenge|challenge-container.*image)",
    re.IGNORECASE)
_SLIDER = re.compile(
    r"(slider|slide to (verify|unlock|confirm)|"
    r"drag the (slider|piece)|puzzle piece)",
    re.IGNORECASE)
_CHECKBOX_WIDGET = re.compile(
    r"(challenges\.cloudflare\.com/turnstile|cf-turnstile|"
    r"google\.com/recaptcha/api2/anchor|recaptcha\.net/"
    r"recaptcha/api2/anchor|hcaptcha\.com/.*checkbox|"
    r"recaptcha-checkbox|challenge-checkbox|"
    r"class=\"[^\"]*checkbox[^\"]*\"[^>]*>\s*<(span|div))",
    re.IGNORECASE)


def classify_wall(signal: Any, html: str, title: str) -> str:
    """Type a detected challenge: 'managed' (wait it out),
    'checkbox' (one click), 'hard' (a human puzzle), or ''
    (an incidental widget on an open page — not a wall)."""
    if signal is None:
        return ""
    kind = str(getattr(signal, "kind", "") or "")
    detection = str(getattr(signal, "detection", "") or "")
    text = str(html or "")
    low = (text + " " + str(title or "")).lower()
    if "just a moment" in low or \
            "checking your browser" in low or \
            detection == "managed" or kind == "managed":
        return "managed"
    if _IMAGE_GRID.search(text) or _SLIDER.search(text):
        return "hard"
    has_challenge_text = bool(_CHALLENGE_TEXT.search(low))
    if kind == "turnstile" or _CHECKBOX_WIDGET.search(text):
        # A widget alone on an otherwise open page is
        # incidental — a blog with a comment-form CAPTCHA
        # is not a wall.  It becomes one when the page
        # itself is a challenge page (challenge text in the
        # body or title, or a turnstile interstitial).
        if has_challenge_text or kind == "turnstile":
            return "checkbox"
        return ""
    if has_challenge_text:
        return "hard"
    return ""


# ── the probe seam ────────────────────────────────────────


class PageProbe(Protocol):
    """What the passes need from a live page.  Playwright
    implements it in production; tests script it."""

    def url(self) -> str: ...
    def title(self) -> str: ...
    def html(self) -> str: ...
    def reload(self) -> None: ...
    def click_gate(self, gate: GateCandidate) -> bool: ...
    def click_checkbox(self) -> bool: ...
    def wait(self, seconds: float) -> None: ...


@dataclass
class GateAction:
    """A gate the pass cleared — journaled by callers."""

    kind: str
    label: str
    host: str = ""


class GateMemory:
    """Session memory of cleared gates, keyed by host+kind,
    so repeat captures don't re-fight a gate the browser
    already carries clearance for."""

    def __init__(self) -> None:
        self._cleared: set[tuple[str, str]] = set()

    def remember(self, action: GateAction) -> None:
        self._cleared.add((action.host, action.kind))

    def knows(self, host: str, kind: str) -> bool:
        return (host, kind) in self._cleared


def _emit(on_step: StepHook | None, text: str) -> None:
    if on_step is not None:
        with suppress(Exception):
            on_step(text)


def run_interstitial_pass(
        probe: PageProbe, *,
        memory: GateMemory | None = None,
        on_step: StepHook | None = None,
        rounds: int = 3) -> list[GateAction]:
    """Clear cookie/consent/age gates, then VERIFY none is
    still standing.  A click counts as handled only when
    the gate verifiably clears; a click that refuses or
    doesn't clear is a failed attempt on THAT control, and
    the pass moves to the next candidate (the blocking
    layer's control) instead of dying on — or retrying —
    a covered, unrelated button.  Raises GateBlocked
    honestly when no candidate clears — including
    date-of-birth forms, which Devon does not invent an
    answer for."""
    actions: list[GateAction] = []
    failed: dict[tuple[str, str], str] = {}
    for _round in range(max(1, rounds)):
        gates = detect_gates(probe.html(), probe.url())
        if not gates:
            return actions
        progressed = False
        for gate in gates:
            if gate.kind == "age_dob":
                raise GateBlocked(
                    "the page asks for a date of birth to "
                    "verify age — Devon does not invent one, "
                    "so the gate stays up",
                    gates=gates)
            if memory is not None and memory.knows(
                    gate.host or _host_of(probe.url()),
                    gate.kind):
                continue
            key = (gate.kind, gate.label)
            if key in failed:
                continue
            _emit(on_step, f"clearing a {gate.kind} gate")
            if not probe.click_gate(gate):
                failed[key] = (
                    f"the {gate.kind} gate "
                    f"({gate.label!r}) would not click")
                continue
            probe.wait(0.6)
            if any(g.kind == gate.kind for g in
                   detect_gates(probe.html(), probe.url())):
                failed[key] = (
                    f"the {gate.kind} gate "
                    f"({gate.label!r}) was clicked but "
                    "stayed up")
                continue
            action = GateAction(
                kind=gate.kind, label=gate.label,
                host=gate.host or _host_of(probe.url()))
            actions.append(action)
            if memory is not None:
                memory.remember(action)
            progressed = True
        if not progressed:
            break
    remaining = detect_gates(probe.html(), probe.url())
    if remaining:
        kinds = ", ".join(g.kind for g in remaining)
        message = (
            f"gate(s) still standing after the pass: "
            f"{kinds} — the capture would be a photo of "
            "the gate, so nothing was shot")
        if failed:
            message = (
                f"gate(s) still standing after the pass: "
                f"{kinds} — tried: "
                f"{'; '.join(failed.values())} — the "
                "capture would be a photo of the gate, so "
                "nothing was shot")
        raise GateBlocked(message, gates=remaining)
    return actions


@dataclass
class TakeoverAsk:
    """What the owner is told when a hard wall needs their
    hand: which wall, where, and what was already tried."""

    url: str
    wall: str
    vendor: str = ""
    attempts: list[str] = field(default_factory=list)
    hold_s: float = TAKEOVER_BUDGET_S


TakeoverHook = Callable[[TakeoverAsk], bool]


def _signal_for(probe: PageProbe) -> Any:
    from ...browser.challenge import detect_challenge_signal

    return detect_challenge_signal(
        probe.html(), probe.url(), title=probe.title())


def _wall_now(probe: PageProbe) -> tuple[str, Any]:
    signal = _signal_for(probe)
    return classify_wall(signal, probe.html(),
                         probe.title()), signal


def run_wall_pass(
        probe: PageProbe, *,
        url: str,
        takeover_hook: TakeoverHook | None = None,
        takeover_done: Callable[[], bool] | None = None,
        on_step: StepHook | None = None,
        managed_budget_s: float = MANAGED_BUDGET_S,
        takeover_budget_s: float = TAKEOVER_BUDGET_S,
        clock: Callable[[], float] = time.monotonic,
        ) -> str | None:
    """Work a bot check.  Returns how it cleared
    ('managed_wait' | 'checkbox_click' | 'owner_takeover')
    or None when the page is open.  Raises WallBlocked —
    a challenge page is never the result.

    During the takeover hold, ``takeover_done`` (when given)
    is consulted every poll: the owner's explicit hand-back
    ends the hold — the pass re-reads the page state and
    resumes as 'owner_takeover' when the wall is actually
    gone, and raises the wall honestly when it is still
    up — instead of holding the session hostage until the
    budget dies."""
    wall, signal = _wall_now(probe)
    if not wall:
        return None
    vendor = str(getattr(signal, "vendor", "") or "")
    attempts: list[str] = []

    if wall == "managed":
        _emit(on_step, "bot check — waiting it out")
        attempts.append("managed wait-out")
        deadline = clock() + max(1.0, managed_budget_s)
        while clock() < deadline:
            probe.wait(2.0)
            wall, signal = _wall_now(probe)
            if not wall:
                return "managed_wait"
            if wall != "managed":
                break          # reclassified mid-wait
        if not wall:
            return "managed_wait"

    if wall == "checkbox":
        _emit(on_step, "bot check — ticking the checkbox")
        for _try in range(3):
            attempts.append("checkbox click")
            if probe.click_checkbox():
                probe.wait(1.2)
                wall, signal = _wall_now(probe)
                if not wall:
                    return "checkbox_click"
                if wall != "checkbox":
                    break      # it escalated; handle below
            else:
                probe.wait(0.8)
        wall, signal = _wall_now(probe)
        if not wall:
            return "checkbox_click"

    if wall:
        # Hard wall (or an escalation into one): the owner's
        # hand in the live session is the only honest path.
        ask = TakeoverAsk(url=url, wall=wall, vendor=vendor,
                          attempts=attempts,
                          hold_s=takeover_budget_s)
        if takeover_hook is None:
            raise WallBlocked(
                f"bot wall ({wall}) at {url} needs a human "
                "hand and no takeover route is attached — "
                "nothing was captured",
                wall=wall, attempts=attempts)
        _emit(on_step, "bot wall — asking for your hand")
        try:
            delivered = bool(takeover_hook(ask))
        except Exception:  # noqa: BLE001 - a failed notice
            delivered = False      # is a fact, not a crash
        if not delivered:
            raise WallBlocked(
                f"bot wall ({wall}) at {url}: the takeover "
                "notice could not be delivered, so the "
                "session did not hold",
                wall=wall, attempts=attempts)
        deadline = clock() + max(1.0, takeover_budget_s)
        while clock() < deadline:
            if takeover_done is not None and takeover_done():
                # Hand-back: re-read the page state before
                # resuming.  Done alone is not a clear — if
                # the wall is still up, the honest verdict
                # is the wall, never a blind continue.
                wall, _sig = _wall_now(probe)
                if not wall:
                    return "owner_takeover"
                raise WallBlocked(
                    f"bot wall ({wall}) at {url}: the owner "
                    "handed the session back with the wall "
                    "still up — nothing was captured",
                    wall=wall, attempts=attempts)
            probe.wait(2.0)
            wall, _sig = _wall_now(probe)
            if not wall:
                return "owner_takeover"
        raise WallBlocked(
            f"bot wall ({wall}) at {url} did not clear "
            f"within {takeover_budget_s:.0f}s of the "
            "takeover hold",
            wall=wall, attempts=attempts)
    return None


# ── scroll planning (pure) ────────────────────────────────

_SCROLL_MAX_PX_S = 1600.0
_SCROLL_MIN_DESCENT_S = 1.2


def _ease(f: float) -> float:
    """easeInOutCubic."""
    f = min(1.0, max(0.0, f))
    if f < 0.5:
        return 4 * f * f * f
    return 1 - pow(-2 * f + 2, 3) / 2


@dataclass
class ScrollPlan:
    """Keyframe plan for one recording: DENSE (offset_s, y)
    samples of the eased scroll curve (~5 per second), so
    stepping through them is a glide, not a jump."""

    duration_s: float
    distance: float
    keyframes: list[tuple[float, float]]
    reached_bottom: bool
    cycles: int

    def y_at(self, t: float) -> float:
        frames = self.keyframes
        if not frames:
            return 0.0
        if t <= frames[0][0]:
            return frames[0][1]
        for (t0, y0), (t1, y1) in zip(frames, frames[1:],
                                      strict=False):
            if t <= t1:
                if t1 <= t0:
                    return y1
                f = (t - t0) / (t1 - t0)
                return y0 + (y1 - y0) * f
        return frames[-1][1]


def scroll_plan(duration_s: float, page_height: float,
                viewport_height: float, *,
                hold_head_s: float = 0.8,
                hold_tail_s: float = 0.8) -> ScrollPlan:
    """Plan an eased scroll for one recording.

    Short pages: descend once, hold the tail.  Pages that
    fit the descent but leave time over: glide back up and
    descend again (loop-pan) rather than sitting still.
    Pages too long for the duration at the speed ceiling:
    a partial descent that ends mid-page — reported via
    reached_bottom=False, never dressed up."""
    duration = max(1.0, float(duration_s))
    distance = max(0.0, float(page_height)
                   - float(viewport_height))
    if distance <= 0:
        return ScrollPlan(duration_s=duration, distance=0.0,
                          keyframes=[(0.0, 0.0),
                                     (duration, 0.0)],
                          reached_bottom=True, cycles=1)
    descent_s = max(_SCROLL_MIN_DESCENT_S,
                    distance / _SCROLL_MAX_PX_S)
    # Segments: (t_start, t_end, y_start, y_target,
    # full_span).  A motion's easing runs over its FULL
    # span; when the time budget truncates the segment, it
    # simply stops partway along the same curve (a partial
    # descent ends mid-page at the honest position).
    segments: list[tuple[float, float, float, float, float]] \
        = []
    t = 0.0
    head = min(hold_head_s, duration * 0.2)
    segments.append((t, t + head, 0.0, 0.0, 1.0))
    t = segments[-1][1]
    budget_end = duration - min(hold_tail_s, duration * 0.2)
    cycles = 0
    reached = False
    while t < budget_end - 0.05:
        span = min(descent_s, budget_end - t)
        y1 = distance * _ease(span / descent_s)
        segments.append((t, t + span, 0.0, distance,
                         descent_s))
        t += span
        cycles += 1
        if span >= descent_s - 1e-9:
            reached = True
        # Another loop-pan cycle starts only when a FULL
        # cycle (glide up + full descent) fits the remaining
        # budget — otherwise the tail holds at the bottom.
        up_s = min(0.9, (budget_end - t) / 2)
        if up_s <= 0.05 or \
                (budget_end - t) < up_s + descent_s - 1e-9:
            break
        segments.append((t, t + up_s, y1, 0.0, up_s))
        t += up_s

    def seg_y(tau: float) -> float:
        for t0, t1, y0, y_target, full_span in segments:
            if tau <= t1:
                if y_target == y0 or full_span <= 0:
                    return y0
                return y0 + (y_target - y0) * _ease(
                    (tau - t0) / full_span)
        last = segments[-1]
        return last[2] + (last[3] - last[2]) * _ease(
            (tau - last[0]) / last[4]) if last[4] > 0 \
            else last[2]

    end_y = seg_y(t)
    segments.append((t, duration, end_y, end_y, 1.0))
    keyframes: list[tuple[float, float]] = []
    tau = 0.0
    while tau < duration:
        keyframes.append((round(tau, 3), seg_y(tau)))
        tau += 0.2
    keyframes.append((duration, seg_y(duration)))
    return ScrollPlan(duration_s=duration, distance=distance,
                      keyframes=keyframes,
                      reached_bottom=reached,
                      cycles=max(1, cycles))


# ── the Playwright probe ──────────────────────────────────

_GATE_FRAME_HINTS = ("onetrust", "qc-cmp", "sp_message",
                     "trustarc", "consent", "gdpr")
_CHECKBOX_FRAMES = ("recaptcha/api2/anchor",
                    "challenges.cloudflare.com",
                    "hcaptcha")
_CHECKBOX_SELECTORS = (
    ".recaptcha-checkbox-border",
    "#recaptcha-anchor",
    "input[type='checkbox']",
    "[role='checkbox']",
    "[class*='checkbox']",
)


class _PlaywrightProbe:
    """PageProbe over a live Playwright page."""

    def __init__(self, page: Any) -> None:
        self._page = page

    def url(self) -> str:
        return str(self._page.url or "")

    def title(self) -> str:
        with suppress(Exception):
            return str(self._page.title() or "")
        return ""

    def html(self) -> str:
        with suppress(Exception):
            return str(self._page.content() or "")
        return ""

    def reload(self) -> None:
        with suppress(Exception):
            self._page.reload(wait_until="load",
                              timeout=30_000)

    def wait(self, seconds: float) -> None:
        self._page.wait_for_timeout(
            int(max(0.0, seconds) * 1000))

    def click_gate(self, gate: GateCandidate) -> bool:
        page = self._page
        label = gate.label.strip()
        # 1. The selector hint the detector collected.
        if gate.selector:
            with suppress(Exception):
                loc = page.locator(gate.selector).first
                if loc.count() and loc.is_visible():
                    loc.click(timeout=3_000)
                    return True
        # 2. Text/role lookup on the main frame.
        if label:
            for finder in (
                    lambda: page.get_by_role(
                        "button", name=label, exact=False),
                    lambda: page.locator(
                        f"text={label}").first):
                with suppress(Exception):
                    loc = finder()
                    if loc.count() and loc.is_visible():
                        loc.click(timeout=3_000)
                        return True
        # 3. Frames — consent managers live in iframes.
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            with suppress(Exception):
                loc = frame.locator(
                    f"text={label}").first if label else None
                if loc is not None and loc.count() and \
                        loc.is_visible():
                    loc.click(timeout=3_000)
                    return True
        return False

    def click_checkbox(self) -> bool:
        page = self._page
        # Frame-carried widgets first (reCAPTCHA anchor,
        # Turnstile, hCaptcha).
        for frame in page.frames:
            furl = str(getattr(frame, "url", "") or "")
            if not any(h in furl for h in _CHECKBOX_FRAMES):
                continue
            for selector in _CHECKBOX_SELECTORS[:2]:
                with suppress(Exception):
                    loc = frame.locator(selector).first
                    if loc.count():
                        loc.click(timeout=3_000)
                        return True
        for selector in _CHECKBOX_SELECTORS:
            with suppress(Exception):
                loc = page.locator(selector).first
                if loc.count() and loc.is_visible():
                    loc.click(timeout=3_000)
                    return True
        return False


# ── presentations: the fingerprint ladder ─────────────────
#
# A capture session's presentation is a LADDER of coherent
# device fingerprints, not one fixed shape: the desktop /
# headless-default presentation is what bot walls are tuned
# to catch, while a phone-shaped presentation passes the
# common walls — so the ladder starts mobile and rotates
# (Android phone → iOS phone → tablet → coherent desktop)
# only when a wall or a gate refuses the current rung.
# Owner takeover is the LAST rung, after the ladder is
# exhausted.  Every rung is internally consistent: its UA,
# client hints, viewport, touch and scale all describe the
# same device — a fingerprint and a form factor never
# disagree.

_CHROME_HINTS_VERSION = "141"


def _chrome_hints(*, platform: str, model: str,
                  mobile: bool) -> dict[str, Any]:
    """Sec-CH-UA metadata (the CDP userAgentMetadata
    payload) coherent with a Chrome UA on ``platform``."""
    return {
        "brands": [
            {"brand": "Chromium",
             "version": _CHROME_HINTS_VERSION},
            {"brand": "Google Chrome",
             "version": _CHROME_HINTS_VERSION},
            {"brand": "Not?A_Brand", "version": "99"},
        ],
        "fullVersionList": [
            {"brand": "Chromium",
             "version": f"{_CHROME_HINTS_VERSION}.0.0.0"},
            {"brand": "Google Chrome",
             "version": f"{_CHROME_HINTS_VERSION}.0.0.0"},
            {"brand": "Not?A_Brand", "version": "99.0.0.0"},
        ],
        "platform": platform,
        "model": model,
        "mobile": mobile,
        "architecture": "",
        "bitness": "",
        "platformVersion": "",
    }


def _safari_hints(*, model: str,
                  mobile: bool) -> dict[str, Any]:
    """Client-hint metadata for the WebKit rungs: Safari
    sends no brand list, so the coherent metadata carries
    the device facts without Chromium brands."""
    return {
        "brands": [],
        "fullVersionList": [],
        "platform": "iOS",
        "model": model,
        "mobile": mobile,
        "architecture": "",
        "bitness": "",
        "platformVersion": "",
    }


#: The named presentations.  Keys beyond the geometry the
#: session has always carried (width/height/scale/mobile):
#: ``name``, ``user_agent``, ``client_hints``.
PRESENTATIONS: dict[str, dict[str, Any]] = {
    "android-phone": {
        "name": "android-phone",
        "width": 390, "height": 844, "scale": 2,
        "mobile": True,
        "user_agent": (
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{_CHROME_HINTS_VERSION}.0.0.0 "
            "Mobile Safari/537.36"),
        "client_hints": _chrome_hints(
            platform="Android", model="Pixel 8",
            mobile=True),
    },
    "ios-phone": {
        "name": "ios-phone",
        "width": 390, "height": 844, "scale": 3,
        "mobile": True,
        "user_agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 "
            "like Mac OS X) AppleWebKit/605.1.15 (KHTML, "
            "like Gecko) Version/18.1 Mobile/15E148 "
            "Safari/604.1"),
        "client_hints": _safari_hints(
            model="iPhone", mobile=True),
    },
    "ipad": {
        "name": "ipad",
        "width": 768, "height": 1024, "scale": 2,
        "mobile": True,
        "user_agent": (
            "Mozilla/5.0 (iPad; CPU OS 18_1 like Mac OS "
            "X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/18.1 Mobile/15E148 Safari/604.1"),
        "client_hints": _safari_hints(
            model="iPad", mobile=True),
    },
    "android-tablet": {
        "name": "android-tablet",
        "width": 800, "height": 1280, "scale": 2,
        "mobile": True,
        "user_agent": (
            "Mozilla/5.0 (Linux; Android 14; SM-X710) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{_CHROME_HINTS_VERSION}.0.0.0 "
            "Safari/537.36"),
        "client_hints": _chrome_hints(
            platform="Android", model="SM-X710",
            mobile=True),
    },
    "desktop-windows": {
        "name": "desktop-windows",
        "width": 1440, "height": 900, "scale": 1,
        "mobile": False,
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{_CHROME_HINTS_VERSION}.0.0.0 "
            "Safari/537.36"),
        "client_hints": _chrome_hints(
            platform="Windows", model="", mobile=False),
    },
    "desktop-macos": {
        "name": "desktop-macos",
        "width": 1440, "height": 900, "scale": 1,
        "mobile": False,
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X "
            "10_15_7) AppleWebKit/537.36 (KHTML, like "
            f"Gecko) Chrome/{_CHROME_HINTS_VERSION}.0.0.0 "
            "Safari/537.36"),
        "client_hints": _chrome_hints(
            platform="macOS", model="", mobile=False),
    },
}

#: The default presentation: the first rung of the
#: default ladder (mobile — his steer).
DEFAULT_PRESENTATION = "android-phone"

#: Ladders by requested variant.  A variant's ladder
#: stays inside its form factor (a phone request never
#: rotates onto a desktop viewport); the no-variant
#: ladder walks phone → tablet → desktop.
_PRESENTATION_LADDERS: dict[str, tuple[str, ...]] = {
    "default": ("android-phone", "ios-phone", "ipad",
                "desktop-windows"),
    "phone": ("android-phone", "ios-phone"),
    "tablet": ("ipad", "android-tablet"),
    "desktop": ("desktop-windows", "desktop-macos"),
}


def presentation_ladder(
        variant: str | None = None,
        ) -> list[dict[str, Any]]:
    """The ordered fingerprint ladder for a variant
    (``phone`` / ``tablet`` / ``desktop``), or the default
    ladder when no variant was chosen.  Fresh dicts —
    callers may annotate them."""
    names = _PRESENTATION_LADDERS.get(
        str(variant or "default").lower(),
        _PRESENTATION_LADDERS["default"])
    return [dict(PRESENTATIONS[name]) for name in names]


def context_kwargs(device: dict[str, Any], *,
                   scheme: str = "light",
                   record_dir: Path | None = None,
                   ) -> dict[str, Any]:
    """The context facts the pipeline REQUESTS from the
    driver for one presentation — pure, so the seam tests
    pin exactly what a session asks for: viewport, scale,
    mobile/touch flags and the user agent, all from the
    same device dict (one coherent fingerprint)."""
    viewport = {"width": int(device["width"]),
                "height": int(device["height"])}
    kwargs: dict[str, Any] = {
        "viewport": viewport,
        "color_scheme": scheme,
        "device_scale_factor": float(device.get("scale", 1)),
        "is_mobile": bool(device.get("mobile", False)),
        "has_touch": bool(device.get("mobile", False)),
    }
    ua = str(device.get("user_agent") or "")
    if ua:
        kwargs["user_agent"] = ua
    if record_dir is not None:
        kwargs["record_video_dir"] = str(record_dir)
        kwargs["record_video_size"] = dict(viewport)
    return kwargs


# ── the session ───────────────────────────────────────────

_PROFILE_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _profile_lock(profile_dir: Path) -> threading.Lock:
    key = str(profile_dir)
    with _LOCKS_GUARD:
        return _PROFILE_LOCKS.setdefault(
            key, threading.Lock())


# ── imported cookies (owner cookie bundles, /cookies) ────
#
# The owner hands Devon his logged-in sessions as a cookie
# bundle (cookies.txt / JSON export — Android Chrome cannot
# export, so the bundle is how a phone's sessions reach her).
# Since Navigator A6 the imported set lives in the UNIFIED
# cookie estate (storage/cookie_estate.py), the one store
# every browser surface reads and writes: every NavSession
# over the profile applies the estate at start (so captures
# begin signed-in), and an import that lands while a session
# is live applies to that session's context too.  The old
# sidecar (imported_cookies.json in the profile dir) is the
# legacy location — its contents were imported once into the
# estate and it is never written again; it is read only as a
# fallback when the estate cannot be opened, and as a
# migration source.  The write side (parsing, merging,
# clearing, journaling) lives in cookie_bundle.py; this seam
# owns only the profile-side mechanics.  Cookie VALUES are
# credentials: nothing in this seam logs, prints or journals
# them — records move as data, never as text.

#: Sidecar filename inside the browser profile dir.
IMPORTED_COOKIES_FILENAME = "imported_cookies.json"


def imported_cookies_path(profile_dir: Path) -> Path:
    return Path(profile_dir) / IMPORTED_COOKIES_FILENAME


def read_imported_cookie_records(
        profile_dir: Path) -> list[dict[str, Any]]:
    """The stored imported-cookie records for a profile
    (normalized dicts: name/value/domain/path/expires/
    httpOnly/secure/sameSite).  LEGACY read since Navigator
    A6: the estate holds the live set; this reads the old
    sidecar — the migration source and the fallback when
    the estate cannot be opened.  A missing or unreadable
    sidecar reads as no imports — the profile simply starts
    without them.  Never raises."""
    try:
        raw = json.loads(imported_cookies_path(
            profile_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = raw.get("cookies") if isinstance(raw, dict) \
        else raw
    if not isinstance(rows, list):
        return []
    return [dict(r) for r in rows if isinstance(r, dict)
            and r.get("name") and r.get("domain")]


def playwright_cookie_payload(
        records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stored records → ``add_cookies`` arguments — the
    canonical rendering, owned by the estate (Navigator A6).
    Session cookies (expires <= 0) go out with no expires
    key at all — the driver's word for a session cookie is
    the absence of one."""
    out: list[dict[str, Any]] = []
    for rec in records:
        item: dict[str, Any] = {
            "name": str(rec.get("name") or ""),
            "value": str(rec.get("value") or ""),
            "domain": str(rec.get("domain") or ""),
            "path": str(rec.get("path") or "/"),
            "httpOnly": bool(rec.get("httpOnly")),
            "secure": bool(rec.get("secure")),
        }
        expires = float(rec.get("expires") or 0)
        if expires > 0:
            item["expires"] = expires
        same = str(rec.get("sameSite") or "")
        if same in ("Strict", "Lax", "None"):
            item["sameSite"] = same
        if item["name"] and item["domain"]:
            out.append(item)
    return out


_LIVE_CONTEXTS: dict[str, set[Any]] = {}
_LIVE_CONTEXTS_GUARD = threading.Lock()


def _register_live_context(profile_dir: Path,
                           context: Any) -> None:
    with _LIVE_CONTEXTS_GUARD:
        _LIVE_CONTEXTS.setdefault(
            str(profile_dir), set()).add(context)


def _unregister_live_context(profile_dir: Path,
                             context: Any) -> None:
    with _LIVE_CONTEXTS_GUARD:
        live = _LIVE_CONTEXTS.get(str(profile_dir))
        if live is not None:
            live.discard(context)
            if not live:
                del _LIVE_CONTEXTS[str(profile_dir)]


def _live_contexts(profile_dir: Path) -> list[Any]:
    with _LIVE_CONTEXTS_GUARD:
        return list(_LIVE_CONTEXTS.get(
            str(profile_dir), ()))


def apply_cookies_to_live(
        profile_dir: Path,
        records: list[dict[str, Any]]) -> int:
    """Add cookie records to every live session context
    over this profile (a capture running right now picks
    the import up without waiting for its next start).
    Returns how many contexts took them.  A context that
    refuses (closing, driver drift) is skipped, never an
    import failure — the sidecar still lands the cookies
    at the next session start."""
    payload = playwright_cookie_payload(records)
    if not payload:
        return 0
    applied = 0
    for context in _live_contexts(profile_dir):
        with suppress(Exception):
            context.add_cookies(payload)
            applied += 1
    return applied


def clear_cookies_from_live(
        profile_dir: Path,
        domains: list[str]) -> int:
    """Best-effort removal of cookies for *domains* from
    live session contexts (the durable removal is the
    sidecar — this only shortens the tail on sessions
    already running).  Returns how many contexts were
    asked.  Never raises."""
    asked = 0
    for context in _live_contexts(profile_dir):
        clear = getattr(context, "clear_cookies", None)
        if not callable(clear):
            continue
        asked += 1
        for domain in domains:
            with suppress(Exception):
                clear(domain=domain)
    return asked


@dataclass
class NavOutcome:
    """The open page, after the passes."""

    final_url: str
    title: str
    gates: list[GateAction] = field(default_factory=list)
    wall: str | None = None
    steps: list[str] = field(default_factory=list)
    #: The fingerprint that cleared (presentation name),
    #: and every presentation tried before it, in order —
    #: the journal of the ladder walk.
    presentation: str = ""
    presentations_tried: list[str] = field(
        default_factory=list)
    #: Per-rung verdicts of the ladder race (name →
    #: cleared | gate_blocked | wall_blocked | dob_refused
    #: | error | superseded): every rung's outcome is
    #: journaled, never just the winner's — a job that
    #: loses every rung reports every rung it tried.
    rung_outcomes: dict[str, str] = field(
        default_factory=dict)

    @property
    def touched(self) -> bool:
        """The pass had to clear something to get in."""
        return bool(self.gates or self.wall)


# ── the session bound ─────────────────────────────────────
#
# Live Chromium sessions are the browser organ's scarce
# resource: concurrent captures race their ladders in
# parallel, so the bound is on SESSIONS, not jobs — one
# job's wall or takeover hold occupies only its own
# session's slot while other jobs proceed on theirs.

#: Default live-session bound (a 2-vCPU / 2 GB host runs
#: ~3 headless Chromiums comfortably); the profile
#: runtime lowers it on small devices, and
#: NM_WEBNAV_MAX_SESSIONS overrides it outright.
_MAX_SESSIONS_DEFAULT = 3
_MAX_SESSIONS_BY_PROFILE = {"termux": 2}
_SLOT_ACQUIRE_TIMEOUT_S = 180.0

_nav_slot_semaphores: dict[int, threading.BoundedSemaphore] \
    = {}
_nav_slot_lock = threading.Lock()


def max_nav_sessions() -> int:
    """The live-session bound: env override, else the
    profile runtime's word, else the default."""
    raw = os.environ.get("NM_WEBNAV_MAX_SESSIONS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    kind = ""
    try:
        from ...core.profiles import get_profile_kind
        kind = (get_profile_kind() or "").strip().lower()
    except Exception:  # noqa: BLE001 - profile lookup
        kind = ""              # never blocks a capture
    return _MAX_SESSIONS_BY_PROFILE.get(
        kind, _MAX_SESSIONS_DEFAULT)


def _nav_slot_semaphore() -> threading.BoundedSemaphore:
    size = max_nav_sessions()
    with _nav_slot_lock:
        sem = _nav_slot_semaphores.get(size)
        if sem is None:
            sem = threading.BoundedSemaphore(size)
            _nav_slot_semaphores[size] = sem
        return sem


class NavSession:
    """One browser session over Devon's persistent
    profile (or an ephemeral context when no profile dir
    is given)."""

    def __init__(self, *, profile_dir: Path | None = None,
                 device: dict[str, Any] | None = None,
                 scheme: str = "light",
                 record_dir: Path | None = None,
                 cookie_estate: Any = None) -> None:
        self._profile_dir = profile_dir
        self._device = dict(
            device or PRESENTATIONS[DEFAULT_PRESENTATION])
        self._scheme = scheme
        self._record_dir = record_dir
        #: The unified cookie estate (Navigator A6), injected
        #: by callers that hold one (tests); None = opened
        #: lazily from the home DB on first use.
        self._cookie_estate = cookie_estate
        self._estate_failed = False
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._lock: threading.Lock | None = None
        self._started_at: float | None = None
        self._slot_held = False
        self.probe: _PlaywrightProbe | None = None

    def _estate(self) -> Any:
        """The session's cookie estate, or None when the home
        DB cannot be opened — the session then starts
        logged-out and harvests nothing, exactly as a
        missing sidecar behaves today.  Never raises."""
        if self._cookie_estate is not None:
            return self._cookie_estate
        if self._estate_failed:
            return None
        try:
            from ...storage.cookie_estate import home_estate

            self._cookie_estate = home_estate(
                legacy_roots={"profile_dir": self._profile_dir})
            return self._cookie_estate
        except Exception:  # noqa: BLE001 - the estate is an
            self._estate_failed = True  # enhancement, never
            return None  # a capture failure

    # -- lifecycle --
    def _acquire_slot(self) -> None:
        """Take one live-session slot (bounded across the
        whole organ: raced rungs and concurrent jobs all
        draw from the same bound).  A held session — even
        one parked on an owner takeover — holds only its
        own slot; the rest keep serving other jobs."""
        sem = _nav_slot_semaphore()
        if not sem.acquire(timeout=_SLOT_ACQUIRE_TIMEOUT_S):
            raise NavUnavailable(
                "the browser organ is at its session bound "
                f"({max_nav_sessions()} live sessions) — "
                "every slot is busy")
        self._slot_held = True

    def _release_slot(self) -> None:
        if self._slot_held:
            self._slot_held = False
            with suppress(Exception):
                _nav_slot_semaphore().release()

    def __enter__(self) -> NavSession:
        self._acquire_slot()
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            # Self-provision (owner doctrine): a missing engine is
            # Devon's job to fix — one bounded install attempt, then
            # the import is retried before the honest failure stands.
            provisioned = False
            try:
                from ...browser.provision import ensure_playwright

                provisioned = ensure_playwright()
            except Exception:  # noqa: BLE001 - provisioning never gates
                provisioned = False
            if provisioned:
                try:
                    from playwright.sync_api import sync_playwright
                except Exception as exc2:  # noqa: BLE001
                    self._release_slot()
                    raise NavUnavailable(
                        "no browser engine on this host "
                        f"(playwright import failed: {exc2})") from exc2
            else:
                self._release_slot()
                raise NavUnavailable(
                    "no browser engine on this host "
                    f"(playwright import failed: {exc}; Devon tried "
                    "to install it herself and could not)") from exc
        if self._profile_dir is not None:
            lock = _profile_lock(self._profile_dir)
            if not lock.acquire(timeout=90):
                raise NavError(
                    "the browser profile is busy (another "
                    "capture is holding it)")
            self._lock = lock
        try:
            self._pw = sync_playwright().start()
            kwargs = context_kwargs(
                self._device, scheme=self._scheme,
                record_dir=self._record_dir)
            if self._profile_dir is not None:
                self._profile_dir.mkdir(parents=True,
                                        exist_ok=True)
                self._context = \
                    self._pw.chromium.launch_persistent_context(
                        str(self._profile_dir), headless=True,
                        **kwargs)
                self._page = (self._context.pages[0]
                              if self._context.pages
                              else self._context.new_page())
            else:
                self._browser = self._pw.chromium.launch(
                    headless=True)
                self._context = self._browser.new_context(
                    **kwargs)
                self._page = self._context.new_page()
            if self._profile_dir is not None \
                    and self._context is not None:
                _register_live_context(self._profile_dir,
                                       self._context)
                self._apply_imported_cookies()
            self._apply_client_hints()
        except NavError:
            self._teardown()
            raise
        except Exception as exc:  # noqa: BLE001
            self._teardown()
            raise NavUnavailable(
                f"the browser engine failed to start: {exc}"
            ) from exc
        self._started_at = time.monotonic()
        self.probe = _PlaywrightProbe(self._page)
        return self

    def _apply_client_hints(self) -> None:
        """Carry the presentation's client hints (Sec-CH-UA
        metadata) on the live context, where the driver
        exposes a CDP session: UA and hints must describe
        the same device, or the pair is itself a bot tell.
        Best-effort in the file's idiom — the user agent
        itself rides the context kwargs, never this call."""
        hints = self._device.get("client_hints")
        if not hints or self._context is None \
                or self._page is None:
            return
        with suppress(Exception):
            cdp = self._context.new_cdp_session(self._page)
            cdp.send(
                "Emulation.setUserAgentOverride",
                {"userAgent": str(
                    self._device.get("user_agent") or ""),
                 "userAgentMetadata": dict(hints)})

    def _apply_imported_cookies(self) -> None:
        """Start this session signed-in: the unified estate's
        records go onto the fresh context before the first
        navigation, so a gate cleared by identity stays
        cleared and a signed-in page renders as the owner's
        view.  Best-effort in the file's idiom — a context
        that refuses the records leaves the session working
        logged-out, and the estate keeps them for the next
        start.  When the estate cannot be opened, the legacy
        sidecar is read instead (read-only — it is never
        written again)."""
        if self._profile_dir is None \
                or self._context is None:
            return
        records: list[dict[str, Any]] | None = None
        estate = self._estate()
        if estate is not None:
            with suppress(Exception):
                records = estate.cookies_for()
        if records is None:
            records = read_imported_cookie_records(
                self._profile_dir)
        payload = playwright_cookie_payload(records)
        if payload:
            with suppress(Exception):
                self._context.add_cookies(payload)

    def _harvest_cookies(self) -> None:
        """Land this session's cookies in the unified estate,
        so a login earned during a capture counts for the
        next drive, replay and download — the half of the
        unification the old profile never had (its Chromium
        DB was a dead end no other surface could read).
        Best-effort: a harvest failure never fails the
        capture that just ran."""
        if self._profile_dir is None \
                or self._context is None:
            return
        estate = self._estate()
        if estate is None:
            return
        getter = getattr(self._context, "cookies", None)
        if not callable(getter):
            return
        with suppress(Exception):
            from ...storage.cookie_estate import SOURCE_WEBNAV

            cookies = getter()
            if cookies:
                estate.upsert(cookies, source=SOURCE_WEBNAV)

    def _teardown(self) -> None:
        if self._profile_dir is not None \
                and self._context is not None:
            _unregister_live_context(self._profile_dir,
                                     self._context)
        if self._context is not None:
            with suppress(Exception):
                self._context.close()
        if self._browser is not None:
            with suppress(Exception):
                self._browser.close()
        if self._pw is not None:
            with suppress(Exception):
                self._pw.stop()
        self._context = self._browser = self._pw = None
        if self._lock is not None:
            with suppress(RuntimeError):
                self._lock.release()
            self._lock = None
        self._release_slot()

    def __exit__(self, *exc: Any) -> None:
        self._harvest_cookies()
        self._teardown()

    @property
    def page(self) -> Any:
        if self._page is None:
            raise NavError("the session is not open")
        return self._page

    @property
    def device(self) -> dict[str, Any]:
        """The presentation this session is running."""
        return dict(self._device)

    @property
    def started_at(self) -> float | None:
        """When this session's page came into existence
        (monotonic clock) — the recording's t=0."""
        return self._started_at

    # -- the pipeline --
    def settle(self, *, budget_s: float = SETTLE_BUDGET_S
               ) -> None:
        """Load is the contract; then wait for the render to
        stop changing (bounded), with a best-effort network
        idle first — the discipline the old rung had, kept."""
        page = self.page
        with suppress(Exception):  # best effort only
            page.wait_for_load_state("networkidle",
                                     timeout=3_000)
        deadline = time.monotonic() + max(1.0, budget_s)
        last: Any = None
        while time.monotonic() < deadline:
            try:
                cur = page.evaluate(
                    "() => [document.documentElement"
                    ".outerHTML.length, document"
                    ".documentElement.scrollHeight]")
            except Exception:  # noqa: BLE001 - mid-navigation
                cur = None
            if cur is not None and cur == last:
                return
            last = cur
            page.wait_for_timeout(350)

    def open(self, url: str, *,
             takeover_hook: TakeoverHook | None = None,
             takeover_bridge: "TakeoverBridge | None" = None,
             memory: GateMemory | None = None,
             on_step: StepHook | None = None,
             managed_budget_s: float = MANAGED_BUDGET_S,
             takeover_budget_s: float = TAKEOVER_BUDGET_S
             ) -> NavOutcome:
        """goto → settle → interstitial pass → bot-check
        pass → interstitial pass once more (sites gate
        behind the check).

        ``takeover_bridge`` (a :class:`TakeoverBridge` bound
        to THIS session) takes precedence over the plain
        ``takeover_hook``: its grant registers the live
        session with the takeover manager, so the owner's
        hand-back ends the hold and the pipeline re-reads
        the page from where it stopped."""
        assert self.probe is not None
        probe = self.probe
        steps: list[str] = []
        try:
            self.page.goto(url, wait_until="load",
                           timeout=45_000)
        except Exception as exc:  # noqa: BLE001
            raise NavError(
                f"the page would not load: {exc}") from exc
        self.settle()
        gates = run_interstitial_pass(
            probe, memory=memory, on_step=on_step)
        if gates:
            steps.extend(g.kind for g in gates)
            self.settle()
        wall = run_wall_pass(
            probe, url=url,
            takeover_hook=(takeover_bridge.hook
                           if takeover_bridge is not None
                           else takeover_hook),
            takeover_done=(takeover_bridge.done
                           if takeover_bridge is not None
                           else None),
            on_step=on_step, managed_budget_s=managed_budget_s,
            takeover_budget_s=takeover_budget_s)
        if wall:
            steps.append(wall)
            self.settle()
            more = run_interstitial_pass(
                probe, memory=memory, on_step=on_step)
            if more:
                gates = gates + more
                steps.extend(g.kind for g in more)
                self.settle()
        return NavOutcome(
            final_url=probe.url(), title=probe.title(),
            gates=gates, wall=wall, steps=steps)

    # -- capture + record primitives --
    def capture_still(self, *, full_page: bool = False,
                      clip: dict[str, Any] | None = None,
                      pdf: bool = False) -> bytes:
        page = self.page
        if pdf:
            return bytes(page.pdf(
                format="A4", print_background=True))
        kwargs: dict[str, Any] = {"type": "png",
                                  "full_page": full_page}
        if clip:
            kwargs["clip"] = clip
        return bytes(page.screenshot(**kwargs))

    def run_scroll(self, duration_s: float, *,
                   clock: Callable[[], float] = time.monotonic
                   ) -> ScrollPlan:
        """Scroll the page along the plan for the
        duration, stepping at ~10 Hz."""
        page = self.page
        metrics = page.evaluate(
            "() => [document.documentElement.scrollHeight,"
            " window.innerHeight]")
        plan = scroll_plan(duration_s, float(metrics[0]),
                           float(metrics[1]))
        start = clock()
        for offset, y in plan.keyframes:
            while True:
                remaining = (start + offset) - clock()
                if remaining <= 0:
                    break
                page.wait_for_timeout(
                    int(min(remaining, 0.1) * 1000))
            # A mid-navigation frame can refuse one step;
            # the next keyframe re-asserts the position.
            with suppress(Exception):
                page.evaluate(
                    f"window.scrollTo(0, {y:.1f})")
        tail = (start + duration_s) - clock()
        if tail > 0:
            page.wait_for_timeout(int(tail * 1000))
        return plan

    def video_path(self) -> Path | None:
        """The recorded file's path.  Readable while the
        driver is alive; the file itself is finalized when
        the context closes."""
        try:
            video = getattr(self.page, "video", None)
            if video is None:
                return None
            path = video.path()
            return Path(path) if path else None
        except Exception:  # noqa: BLE001
            return None

    def stop_recording(self) -> Path | None:
        """Finalize the recording and return its file.

        Order matters: closing the context writes the .webm
        out, and the path must be read BEFORE the driver
        stops (afterwards the Video handle is dead).  Falls
        back to the recording directory itself — one page
        records exactly one file there."""
        if self._context is not None:
            with suppress(Exception):
                self._context.close()
            self._context = None
        path = self.video_path()
        if path is None and self._record_dir is not None:
            found = sorted(self._record_dir.glob("*.webm"))
            if found:
                path = found[-1]
        return path


# ── owner takeover: the handoff link ──────────────────────


class NavTabAdapter:
    """The takeover surface for one live NavSession — the
    ``TakeoverTabAdapter`` protocol (browser/takeover.py)
    over this session's real Playwright page.  Frames and
    inputs act on the SAME page the pipeline is driving:
    same cookies, same login, same wall."""

    def __init__(self, session: NavSession) -> None:
        self._session = session

    @property
    def _probe(self) -> PageProbe:
        probe = self._session.probe
        if probe is None:
            raise NavError("the session is not open")
        return probe

    @property
    def url(self) -> str:
        return self._probe.url()

    @property
    def title(self) -> str:
        return self._probe.title()

    def html(self) -> str:
        return self._probe.html()

    def reload(self) -> None:
        self._probe.reload()

    def screenshot_png(self) -> bytes:
        return self._session.capture_still()

    def click_norm(self, x: float, y: float) -> None:
        """Click at normalized (0..1) viewport coordinates —
        the live view's tap, landed on the real page."""
        page = self._session.page
        size = page.evaluate(
            "() => ({w: window.innerWidth,"
            " h: window.innerHeight})")
        page.mouse.click(
            int(float(size["w"]) * max(0.0, min(1.0, float(x)))),
            int(float(size["h"]) * max(0.0, min(1.0, float(y)))))

    def type_text(self, text: str) -> None:
        self._session.page.keyboard.type(text or "")

    def press_key(self, key: str) -> None:
        self._session.page.keyboard.press(key or "")

    def scroll_norm(self, dx: float, dy: float) -> None:
        self._session.page.mouse.wheel(float(dx or 0.0),
                                       float(dy or 0.0))


def format_takeover_message(ask: TakeoverAsk, grant: Any) -> str:
    """The plain-language takeover message: what wall, on
    what site, and the handoff link — the link IS the
    feature; without it there is nothing the owner can
    open.  ``grant`` is a browser/takeover TakeoverGrant
    (duck-typed: ``url`` + ``needs_public_base``)."""
    host = _host_of(ask.url) or ask.url
    tried = "; ".join(ask.attempts) or "the standard passes"
    text = (
        f"I hit a wall on {host} ({ask.wall}) and I can't "
        "clear it myself — open this link to take over my "
        "browser and clear it; hand it back when you're "
        f"done:\n{grant.url}\n"
        f"Already tried: {tried}. The session stays live "
        "while you work; when you hand it back I re-read "
        "the page and carry on from there.")
    if getattr(grant, "needs_public_base", False):
        text += (
            "\n\n⚠️ No public takeover URL is configured "
            "(NM_TAKEOVER_BASE_URL) — this link works on "
            "the host network only until a tunnel / "
            "port-forward is set up.")
    return text


class TakeoverBridge:
    """Binds the browser organ's TakeoverManager to one
    live NavSession for one wall.

    ``hook`` is the webnav takeover hook: on a hard wall it
    registers THIS session with the manager (a one-time,
    expiring, token-gated grant whose tab adapter drives
    the real page) and delivers the plain-language message
    carrying the handoff link through ``notify`` (returns
    True when the message landed).  ``done`` is the wall
    pass's hand-back check: the owner's Done ends the
    hold, and the pipeline re-reads the page state and
    resumes.  The manager is duck-typed (``request_takeover``
    returning a grant with ``url`` / ``done`` /
    ``needs_public_base``) so the seam fakes cleanly."""

    def __init__(self, manager: Any, session: NavSession, *,
                 notify: Callable[[str], bool],
                 session_id: str = "",
                 owner_chat: str = "",
                 task_label: str = "browsing",
                 arm: "Callable[[], bool] | None" = None) -> None:
        self._manager = manager
        self._session = session
        self._notify = notify
        self._session_id = session_id
        self._owner_chat = owner_chat
        self._task_label = task_label
        #: When set, the hook consults it BEFORE registering
        #: anything: the ladder race arms the takeover only
        #: once every other fingerprint has failed, so a
        #: rung that clears can never trigger a spurious
        #: takeover for a job already won.  ``arm()``
        #: returning False = the race was decided elsewhere;
        #: the hook then reports "not delivered" and no
        #: grant, no link, no message ever exists.
        self.arm = arm
        self.grant: Any = None
        self.error: str = ""

    def hook(self, ask: TakeoverAsk) -> bool:
        if self.arm is not None and not self.arm():
            return False
        try:
            grant = self._manager.request_takeover(
                session_id=self._session_id
                or f"webnav-{time.time_ns():x}",
                url=ask.url, task_label=self._task_label,
                signal_kind=ask.wall,
                attempts=list(ask.attempts), resume_state={},
                tab=NavTabAdapter(self._session),
                owner_chat=self._owner_chat)
        except Exception as exc:  # noqa: BLE001 - a grant that
            # cannot start is a fact, not a crash
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        self.grant = grant
        try:
            return bool(self._notify(
                format_takeover_message(ask, grant)))
        except Exception:  # noqa: BLE001 - a failed notice is
            return False           # a fact the pipeline needs

    def done(self) -> bool:
        return bool(self.grant is not None
                    and getattr(self.grant, "done", False))


#: Identity sentinel: when a caller supplies a takeover
#: FACTORY but no plain hook, navigate passes this stand-in
#: into the ladder so the last rung is still marked (the
#: ladder attaches takeover help to the last rung only).
#: open_one substitutes the factory-built bridge whenever
#: a factory is in play; the mark itself only ever
#: answers "not delivered" (a factory that declined to
#: build a bridge for its session).
_FACTORY_MARK: TakeoverHook = lambda ask: False  # noqa: E731


# ── entry points ──────────────────────────────────────────


class _LadderRace:
    """Shared state for one ladder race: which rungs
    failed, which (if any) won.  The takeover rung's
    hook/bridge ARMS only when every other rung has
    failed and no winner exists — takeover stays the
    last resort even though the rungs now run in
    parallel, and a rung that clears can never trigger
    a spurious takeover for a job already won."""

    def __init__(self, rungs: int) -> None:
        self._cond = threading.Condition()
        self._rungs = max(1, rungs)
        self._failed: set[int] = set()
        self._winner: int | None = None

    def note_failure(self, idx: int) -> None:
        with self._cond:
            self._failed.add(idx)
            self._cond.notify_all()

    def note_winner(self, idx: int) -> bool:
        """Record a clearing rung.  True when it is THE
        winner (the first to clear); a later clearer is
        superseded and closes its own session."""
        with self._cond:
            if self._winner is None:
                self._winner = idx
                self._cond.notify_all()
                return True
            return False

    def winner(self) -> int | None:
        with self._cond:
            return self._winner

    def arm(self, *, except_idx: int) -> bool:
        """Wait until the takeover may engage: every
        other rung failed and nobody won.  False when
        the race was won elsewhere — the takeover then
        never registers, never messages, never holds."""
        with self._cond:
            while self._winner is None and len(
                    self._failed - {except_idx}) \
                    < self._rungs - 1:
                self._cond.wait(timeout=0.25)
            return self._winner is None


#: How long the race gives losing rungs to record their
#: outcomes after a winner is known, before returning
#: the winner (stragglers' sessions close themselves
#: out; their verdicts land as 'superseded').
_RACE_GRACE_S = 2.0


def _rung_verdict(exc: BaseException) -> str:
    if isinstance(exc, GateBlocked):
        if any(g.kind == "age_dob" for g in exc.gates):
            return "dob_refused"
        return "gate_blocked"
    if isinstance(exc, WallBlocked):
        return "wall_blocked"
    if isinstance(exc, NavUnavailable):
        return "unavailable"
    return "error"


def open_with_ladder(
        url: str,
        presentations: list[dict[str, Any]],
        open_one: Callable[
            [dict[str, Any], TakeoverHook | None],
            tuple[Any, NavOutcome]],
        *, takeover_hook: TakeoverHook | None = None,
        on_step: StepHook | None = None,
        race: _LadderRace | None = None,
        ) -> tuple[Any, NavOutcome]:
    """Race the fingerprint ladder for one navigation.

    Every presentation runs CONCURRENTLY, each in its
    own session (``open_one(presentation, hook)`` opens
    one session under one presentation and runs the
    pipeline, raising GateBlocked/WallBlocked when that
    presentation is refused, having cleaned up its own
    session).  The first rung to clear wins; losing
    sessions are closed immediately, and every rung's
    verdict is journaled on the outcome — a job that
    loses every rung reports every rung tried, never
    just the first failure.

    The owner's takeover is attached ONLY to the last
    rung, and it is GATED: the hook fires only once
    every other rung has failed (see :class:`_LadderRace`),
    so a parallel clearer never produces a spurious
    takeover.  A date-of-birth refusal is never answered
    on any rung; when nothing clears, a DOB refusal
    among the failures is the verdict that surfaces —
    the page's demand, not a fingerprint's, is the
    refusal."""
    names = [str(p.get("name") or f"presentation-{i + 1}")
             for i, p in enumerate(presentations)]
    last_idx = len(presentations) - 1
    if race is None:
        race = _LadderRace(len(presentations))

    def gated_hook(ask: TakeoverAsk) -> bool:
        if not race.arm(except_idx=last_idx):
            return False
        return bool(takeover_hook(ask)) \
            if takeover_hook is not None else False

    results: dict[int, tuple[str, Any]] = {}
    results_lock = threading.Lock()
    all_done = threading.Event()

    def _record(idx: int, kind: str, payload: Any) -> None:
        with results_lock:
            results[idx] = (kind, payload)
            if len(results) >= len(presentations):
                all_done.set()

    def run(idx: int, pres: dict[str, Any]) -> None:
        hook = gated_hook if idx == last_idx \
            and takeover_hook is not None else None
        try:
            session, outcome = open_one(pres, hook)
        except BaseException as exc:  # noqa: BLE001 - a
            # rung's failure is data for the race
            race.note_failure(idx)
            _record(idx, "err", exc)
            return
        if race.note_winner(idx):
            _record(idx, "ok", (session, outcome))
        else:
            # Superseded: another rung cleared first —
            # this session closes immediately.
            with suppress(Exception):
                session.__exit__()
            _record(idx, "late", outcome)

    threads = [
        threading.Thread(target=run, args=(idx, pres),
                         name=f"webnav-rung-{names[idx]}",
                         daemon=True)
        for idx, pres in enumerate(presentations)]
    for thread in threads:
        thread.start()

    # The race resolves when a winner is known and the
    # field has had a brief grace to journal its
    # outcomes, or when every rung has reported.
    winner_seen_at: float | None = None
    while not all_done.is_set():
        if race.winner() is not None:
            if winner_seen_at is None:
                winner_seen_at = time.monotonic()
            elif time.monotonic() - winner_seen_at \
                    >= _RACE_GRACE_S:
                break
        all_done.wait(timeout=0.05)

    with results_lock:
        snapshot = dict(results)
    verdicts: dict[str, str] = {}
    for idx, name in enumerate(names):
        entry = snapshot.get(idx)
        if entry is None:
            verdicts[name] = "superseded"
        elif entry[0] == "ok":
            verdicts[name] = "cleared"
        elif entry[0] == "late":
            verdicts[name] = "cleared_late"
        else:
            verdicts[name] = _rung_verdict(entry[1])

    winner_idx = race.winner()
    if winner_idx is not None \
            and snapshot.get(winner_idx, (None,))[0] == "ok":
        session, outcome = snapshot[winner_idx][1]
        outcome.presentations_tried = list(names)
        outcome.presentation = names[winner_idx]
        outcome.rung_outcomes = verdicts
        return session, outcome

    # No rung cleared: wait out the field (a takeover
    # hold on the last rung is a live outcome, not a
    # straggler) and report EVERY rung, honestly.
    for thread in threads:
        thread.join()
    with results_lock:
        snapshot = dict(results)
    # A winner that recorded after the grace snapshot
    # still wins — never raise over a cleared page.
    for idx, (kind, payload) in snapshot.items():
        if kind == "ok":
            session, outcome = payload
            outcome.presentations_tried = list(names)
            outcome.presentation = names[idx]
            outcome.rung_outcomes = {
                names[i]: (
                    "cleared" if snap[0] == "ok"
                    else "cleared_late" if snap[0] == "late"
                    else _rung_verdict(snap[1]))
                for i, snap in snapshot.items()}
            return session, outcome
    errors = {idx: payload for idx, (kind, payload)
              in snapshot.items() if kind == "err"}
    names_csv = ", ".join(names)
    for idx in sorted(errors):
        exc = errors[idx]
        if isinstance(exc, GateBlocked) and any(
                g.kind == "age_dob" for g in exc.gates):
            raise exc
    base_exc = errors.get(last_idx) or (
        errors[max(errors)] if errors else None)
    if isinstance(base_exc, NavUnavailable):
        # No engine on this host is a fact about the
        # host, not the ladder — callers branch on it
        # (seed fall-through, the fetch rung), so it
        # surfaces unwrapped.
        raise base_exc
    if isinstance(base_exc, WallBlocked):
        raise WallBlocked(
            f"{base_exc} — fingerprints tried: "
            f"{names_csv}; none cleared",
            wall=base_exc.wall,
            attempts=[*base_exc.attempts,
                      *(f"fingerprint {t}" for t in names)])
    if isinstance(base_exc, GateBlocked):
        raise GateBlocked(
            f"{base_exc} — fingerprints tried: "
            f"{names_csv}; none cleared",
            gates=base_exc.gates)
    if base_exc is not None:
        raise NavError(
            f"no presentation could open the page — "
            f"fingerprints tried: {names_csv}; "
            f"last failure: {base_exc}")
    raise NavError(
        "no presentation could open the page")


def navigate(url: str, *, device: dict[str, Any] | None = None,
             ladder: list[dict[str, Any]] | None = None,
             scheme: str = "light",
             profile_dir: Path | None = None,
             takeover_hook: TakeoverHook | None = None,
             takeover_factory: "Callable[[NavSession], "
             "TakeoverBridge | None] | None" = None,
             memory: GateMemory | None = None,
             on_step: StepHook | None = None,
             managed_budget_s: float = MANAGED_BUDGET_S,
             takeover_budget_s: float = TAKEOVER_BUDGET_S
             ) -> tuple[NavSession, NavOutcome]:
    """Open a session and run the pipeline, RACING the
    fingerprint ladder when one is given: every
    presentation opens concurrently in its own session,
    the first to clear wins, and the owner's takeover
    stays the last resort (attached to the last rung,
    armed only when every other rung has failed).  The
    CALLER owns the returned session (capture from it,
    then close it — it is a context manager).

    Racing rungs run on EPHEMERAL contexts: a persistent
    profile is one browser's state and cannot be raced
    against itself.  A single-presentation navigation
    (no ladder) keeps the caller's ``profile_dir``
    exactly as before.  ``takeover_factory`` builds a
    :class:`TakeoverBridge` for the session that reaches
    the last rung (the factory receives that session —
    the bridge must bind the live session, which only
    exists inside the race).  A bridge takes precedence
    over the plain hook."""
    presentations = [dict(p) for p in ladder] if ladder \
        else [dict(device or PRESENTATIONS[
            DEFAULT_PRESENTATION])]
    ladder_hook = takeover_hook
    if takeover_factory is not None and ladder_hook is None:
        ladder_hook = _FACTORY_MARK
    race = _LadderRace(len(presentations))
    racing = len(presentations) > 1
    by_name = {
        str(p.get("name") or f"presentation-{i + 1}"): i
        for i, p in enumerate(presentations)}

    def open_one(pres: dict[str, Any],
                 hook: TakeoverHook | None,
                 ) -> tuple[NavSession, NavOutcome]:
        session = NavSession(
            profile_dir=None if racing else profile_dir,
            device=pres, scheme=scheme)
        try:
            session.__enter__()
            bridge = takeover_factory(session) \
                if takeover_factory is not None \
                and hook is not None else None
            if bridge is not None:
                idx = by_name.get(
                    str(pres.get("name") or ""),
                    len(presentations) - 1)
                bridge.arm = (
                    lambda idx=idx: race.arm(except_idx=idx))
            plain = None if hook is _FACTORY_MARK else hook
            outcome = session.open(
                url,
                takeover_hook=None if bridge is not None
                else plain,
                takeover_bridge=bridge, memory=memory,
                on_step=on_step,
                managed_budget_s=managed_budget_s,
                takeover_budget_s=takeover_budget_s)
        except BaseException:  # noqa: BLE001 — cleanup, then re-raise
            # Deliberate catch-all: a session that fails to open — on any
            # interruption, including KeyboardInterrupt/SystemExit — must
            # release its browser profile before the failure propagates,
            # or the profile lock strands every later capture.
            session.__exit__()
            raise
        return session, outcome

    return open_with_ladder(
        url, presentations, open_one,
        takeover_hook=ladder_hook, on_step=on_step,
        race=race)


@dataclass
class RecordOutcome:
    """One finished recording."""

    path: Path
    fmt: str                    # mp4 | webm
    duration_s: float
    plan: ScrollPlan | None
    nav: NavOutcome | None
    includes_clearing: bool
    converted: bool


def _probe_duration(path: Path) -> float:
    """Container duration via ffprobe when present."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, timeout=20)
        return float((out.stdout or "0").strip() or 0)
    except Exception:  # noqa: BLE001 - ffprobe is a rung
        return 0.0


def _convert_to_mp4(webm: Path) -> Path | None:
    """webm → mp4 (H.264/yuv420p/faststart) via ffmpeg when
    available; None when it is not or the conversion fails
    — the webm is then delivered honestly as webm."""
    ffmpeg = "ffmpeg"
    target = webm.with_suffix(".mp4")
    try:
        out = subprocess.run(
            [ffmpeg, "-y", "-i", str(webm), "-c:v",
             "libx264", "-pix_fmt", "yuv420p", "-movflags",
             "+faststart", "-an", str(target)],
            capture_output=True, timeout=120)
        if out.returncode == 0 and target.exists() \
                and target.stat().st_size > 0:
            return target
    except Exception:  # noqa: BLE001 - a rung, not a wall
        return None
    return None


def record_page(url: str, *, seconds: float,
                out_dir: Path,
                device: dict[str, Any] | None = None,
                ladder: list[dict[str, Any]] | None = None,
                scheme: str = "light",
                profile_dir: Path | None = None,
                takeover_hook: TakeoverHook | None = None,
                takeover_factory: "Callable[[NavSession], "
                "TakeoverBridge | None] | None" = None,
                on_step: StepHook | None = None,
                managed_budget_s: float = MANAGED_BUDGET_S,
                takeover_budget_s: float = TAKEOVER_BUDGET_S
                ) -> RecordOutcome:
    """Record a page: warm-up pass first (gates/walls
    cleared into the persistent profile, unrecorded —
    walking the fingerprint ladder, so the recording runs
    under the presentation that actually cleared), then
    the recording pass — eased scroll for the requested
    duration → mp4 when ffmpeg allows, webm otherwise."""
    seconds = max(REC_MIN_S, min(REC_MAX_S, float(seconds)))
    memory = GateMemory()
    # Phase 1 — warm-up: clear whatever stands in the way,
    # and learn which fingerprint clears it.
    warm_session, warm_nav = navigate(
        url, device=device, ladder=ladder, scheme=scheme,
        profile_dir=profile_dir, takeover_hook=takeover_hook,
        takeover_factory=takeover_factory,
        memory=memory, on_step=on_step,
        managed_budget_s=managed_budget_s,
        takeover_budget_s=takeover_budget_s)
    device = dict(warm_session.device)
    warm_session.__exit__()
    # Phase 2 — record.
    out_dir.mkdir(parents=True, exist_ok=True)
    includes_clearing = False
    outcome: RecordOutcome | None = None
    for attempt in range(2):
        rec_dir = out_dir / f"rec-{int(time.time() * 1000)}"
        rec_dir.mkdir(parents=True, exist_ok=True)
        session = NavSession(profile_dir=profile_dir,
                             device=device, scheme=scheme,
                             record_dir=rec_dir)
        session.__enter__()
        video: Path | None = None
        try:
            bridge = takeover_factory(session) \
                if takeover_factory is not None else None
            nav = session.open(
                url,
                takeover_hook=None if bridge is not None
                else takeover_hook,
                takeover_bridge=bridge, memory=memory,
                on_step=on_step,
                managed_budget_s=managed_budget_s,
                takeover_budget_s=takeover_budget_s)
            # The recording ran under the warm-up's winning
            # fingerprint; the journal says so.
            nav.presentation = warm_nav.presentation
            nav.presentations_tried = list(
                warm_nav.presentations_tried)
            if nav.touched and attempt == 0:
                # The opening frames would show the clearing:
                # discard and re-record on the warm profile.
                continue
            includes_clearing = bool(nav.touched)
            if on_step is not None:
                on_step(f"recording {seconds:.0f}s — scrolling")
            started = session.started_at or time.monotonic()
            elapsed = time.monotonic() - started
            plan = session.run_scroll(
                max(1.0, seconds - elapsed))
            # Guarantee the full requested duration of video.
            while time.monotonic() - started < seconds:
                session.page.wait_for_timeout(200)
            title = nav.title
            final_url = nav.final_url
            # Read the path before the driver stops (the
            # Video handle dies with it).
            video = session.stop_recording()
        finally:
            session.__exit__()
        if video is None or not video.exists():
            raise NavError(
                "the recorder produced no video file")
        duration = _probe_duration(video)
        fmt = "webm"
        converted = False
        final_path = video
        mp4 = _convert_to_mp4(video)
        if mp4 is not None:
            final_path = mp4
            fmt = "mp4"
            converted = True
            if duration <= 0:
                duration = _probe_duration(mp4)
        outcome = RecordOutcome(
            path=final_path, fmt=fmt,
            duration_s=duration or seconds, plan=plan,
            nav=NavOutcome(final_url=final_url, title=title,
                           gates=nav.gates, wall=nav.wall,
                           steps=nav.steps),
            includes_clearing=includes_clearing,
            converted=converted)
        break
    if outcome is None:  # pragma: no cover - loop always sets
        raise NavError("the recording did not complete")
    return outcome
