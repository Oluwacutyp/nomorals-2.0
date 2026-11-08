"""Navigation-aware page pipeline (§18/§20 god-tier).

One pipeline behind /webshot and /webrec: open the URL in
Devon's real browser session (a persistent profile, so
cookies and gate clearances survive across captures) →
settle → run the interstitial pass (cookie banners, consent
walls, age gates) → work the bot check (managed wait-out,
checkbox click, owner takeover for hard walls) → capture
or record.  A page that refuses to clear is reported
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
    "GateAction",
    "GateBlocked",
    "GateCandidate",
    "GateMemory",
    "NavError",
    "NavOutcome",
    "NavSession",
    "NavUnavailable",
    "PageProbe",
    "RecordOutcome",
    "ScrollPlan",
    "TakeoverAsk",
    "TakeoverHook",
    "WallBlocked",
    "classify_wall",
    "detect_gates",
    "navigate",
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


class _ClickScanner(HTMLParser):
    """Collects clickable controls (buttons, links, inputs)
    with their text, classes and ids."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.controls: list[dict[str, str]] = []
        self.forms_with_dob = False
        self._open: dict[str, str] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]
                        ) -> None:
        a = {k: (v or "") for k, v in attrs}
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
            }
            self._text = []
            if tag == "input" and a.get("value"):
                self._finish()
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
    age before consent before cookie (a page can stack
    them; the pass works them in this order)."""
    text = str(html or "")
    low = text.lower()
    host = _host_of(url)
    scanner = _ClickScanner()
    # A torn page scans as whatever was collected.
    with suppress(Exception):
        scanner.feed(text)
    controls = scanner.controls

    def norm(s: str) -> str:
        return " ".join(str(s).lower().split())

    labels = [(c, norm(c["text"] or c["value"]))
              for c in controls]

    found: list[GateCandidate] = []
    seen: set[str] = set()

    def add(kind: str, control: dict[str, str],
            label: str) -> None:
        if kind in seen:
            return
        seen.add(kind)
        found.append(GateCandidate(
            kind=kind, label=label,
            selector=_selector_for(control), host=host))

    # ── age gates ──
    age_context = any(m in low for m in _AGE_MARKERS)
    site_labels = SITE_AGE_LABELS.get(host, ())
    for control, label in labels:
        if not label:
            continue
        if label in _AGE_LABELS or label in site_labels or \
                any(a in label for a in _AGE_LABELS):
            add("age", control, label)
            break
        if age_context and label in _AGE_BARE_LABELS:
            add("age", control, label)
            break
    if "age" not in seen and age_context and \
            scanner.forms_with_dob:
        # A date-of-birth form is an age gate Devon will not
        # invent an answer for.
        found.append(GateCandidate(
            kind="age_dob", label="date-of-birth form",
            host=host))
        seen.add("age_dob")

    # ── consent walls ──
    consent_context = any(m in low for m in _CONSENT_MARKERS)
    if consent_context:
        for control, label in labels:
            if label in _CONSENT_LABELS or \
                    label.startswith("accept"):
                add("consent", control, label)
                break

    # ── cookie banners ──
    cookie_context = any(m in low for m in _COOKIE_MARKERS)
    if cookie_context and "consent" not in seen:
        for control, label in labels:
            if label in _COOKIE_LABELS:
                add("cookie", control, label)
                break

    order = {"age": 0, "age_dob": 0, "consent": 1, "cookie": 2}
    found.sort(key=lambda g: order.get(g.kind, 3))
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
    still standing.  Raises GateBlocked honestly when a
    gate won't clear — including date-of-birth forms, which
    Devon does not invent an answer for."""
    actions: list[GateAction] = []
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
            _emit(on_step, f"clearing a {gate.kind} gate")
            if probe.click_gate(gate):
                action = GateAction(
                    kind=gate.kind, label=gate.label,
                    host=gate.host or _host_of(probe.url()))
                actions.append(action)
                if memory is not None:
                    memory.remember(action)
                progressed = True
                probe.wait(0.6)
            else:
                raise GateBlocked(
                    f"the {gate.kind} gate "
                    f"({gate.label!r}) would not click — "
                    "the capture would be a photo of the "
                    "gate, so nothing was shot",
                    gates=gates)
        if not progressed:
            break
    remaining = detect_gates(probe.html(), probe.url())
    if remaining:
        kinds = ", ".join(g.kind for g in remaining)
        raise GateBlocked(
            f"gate(s) still standing after the pass: "
            f"{kinds} — the capture would be a photo of "
            "the gate, so nothing was shot",
            gates=remaining)
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
        on_step: StepHook | None = None,
        managed_budget_s: float = MANAGED_BUDGET_S,
        takeover_budget_s: float = TAKEOVER_BUDGET_S,
        clock: Callable[[], float] = time.monotonic,
        ) -> str | None:
    """Work a bot check.  Returns how it cleared
    ('managed_wait' | 'checkbox_click' | 'owner_takeover')
    or None when the page is open.  Raises WallBlocked —
    a challenge page is never the result."""
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


# ── the session ───────────────────────────────────────────

_PROFILE_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _profile_lock(profile_dir: Path) -> threading.Lock:
    key = str(profile_dir)
    with _LOCKS_GUARD:
        return _PROFILE_LOCKS.setdefault(
            key, threading.Lock())


@dataclass
class NavOutcome:
    """The open page, after the passes."""

    final_url: str
    title: str
    gates: list[GateAction] = field(default_factory=list)
    wall: str | None = None
    steps: list[str] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        """The pass had to clear something to get in."""
        return bool(self.gates or self.wall)


class NavSession:
    """One browser session over Devon's persistent
    profile (or an ephemeral context when no profile dir
    is given)."""

    def __init__(self, *, profile_dir: Path | None = None,
                 device: dict[str, Any] | None = None,
                 scheme: str = "light",
                 record_dir: Path | None = None) -> None:
        self._profile_dir = profile_dir
        self._device = dict(device or {
            "width": 1440, "height": 900, "scale": 1,
            "mobile": False})
        self._scheme = scheme
        self._record_dir = record_dir
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._lock: threading.Lock | None = None
        self._started_at: float | None = None
        self.probe: _PlaywrightProbe | None = None

    # -- lifecycle --
    def __enter__(self) -> NavSession:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            raise NavUnavailable(
                "no browser engine on this host "
                f"(playwright import failed: {exc})") from exc
        if self._profile_dir is not None:
            lock = _profile_lock(self._profile_dir)
            if not lock.acquire(timeout=90):
                raise NavError(
                    "the browser profile is busy (another "
                    "capture is holding it)")
            self._lock = lock
        try:
            self._pw = sync_playwright().start()
            viewport = {"width": int(self._device["width"]),
                        "height": int(self._device["height"])}
            kwargs: dict[str, Any] = {
                "viewport": viewport,
                "color_scheme": self._scheme,
                "device_scale_factor": float(
                    self._device.get("scale", 1)),
                "is_mobile": bool(self._device.get(
                    "mobile", False)),
                "has_touch": bool(self._device.get(
                    "mobile", False)),
            }
            if self._record_dir is not None:
                kwargs["record_video_dir"] = str(
                    self._record_dir)
                kwargs["record_video_size"] = dict(viewport)
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

    def _teardown(self) -> None:
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

    def __exit__(self, *exc: Any) -> None:
        self._teardown()

    @property
    def page(self) -> Any:
        if self._page is None:
            raise NavError("the session is not open")
        return self._page

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
             memory: GateMemory | None = None,
             on_step: StepHook | None = None,
             managed_budget_s: float = MANAGED_BUDGET_S,
             takeover_budget_s: float = TAKEOVER_BUDGET_S
             ) -> NavOutcome:
        """goto → settle → interstitial pass → bot-check
        pass → interstitial pass once more (sites gate
        behind the check)."""
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
            probe, url=url, takeover_hook=takeover_hook,
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


# ── entry points ──────────────────────────────────────────


def navigate(url: str, *, device: dict[str, Any] | None = None,
             scheme: str = "light",
             profile_dir: Path | None = None,
             takeover_hook: TakeoverHook | None = None,
             memory: GateMemory | None = None,
             on_step: StepHook | None = None,
             managed_budget_s: float = MANAGED_BUDGET_S,
             takeover_budget_s: float = TAKEOVER_BUDGET_S
             ) -> tuple[NavSession, NavOutcome]:
    """Open a session and run the pipeline.  The CALLER
    owns the returned session (capture from it, then close
    it — it is a context manager)."""
    session = NavSession(profile_dir=profile_dir,
                         device=device, scheme=scheme)
    try:
        session.__enter__()
        outcome = session.open(
            url, takeover_hook=takeover_hook, memory=memory,
            on_step=on_step, managed_budget_s=managed_budget_s,
            takeover_budget_s=takeover_budget_s)
    except BaseException:
        session.__exit__()
        raise
    return session, outcome


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
                scheme: str = "light",
                profile_dir: Path | None = None,
                takeover_hook: TakeoverHook | None = None,
                on_step: StepHook | None = None,
                managed_budget_s: float = MANAGED_BUDGET_S,
                takeover_budget_s: float = TAKEOVER_BUDGET_S
                ) -> RecordOutcome:
    """Record a page: warm-up pass first (gates/walls
    cleared into the persistent profile, unrecorded), then
    the recording pass — eased scroll for the requested
    duration → mp4 when ffmpeg allows, webm otherwise."""
    seconds = max(REC_MIN_S, min(REC_MAX_S, float(seconds)))
    device = dict(device or {"width": 1440, "height": 900,
                             "scale": 1, "mobile": False})
    memory = GateMemory()
    # Phase 1 — warm-up: clear whatever stands in the way.
    with NavSession(profile_dir=profile_dir, device=device,
                    scheme=scheme) as warm:
        warm.open(url, takeover_hook=takeover_hook,
                  memory=memory, on_step=on_step,
                  managed_budget_s=managed_budget_s,
                  takeover_budget_s=takeover_budget_s)
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
            nav = session.open(
                url, takeover_hook=takeover_hook, memory=memory,
                on_step=on_step,
                managed_budget_s=managed_budget_s,
                takeover_budget_s=takeover_budget_s)
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
