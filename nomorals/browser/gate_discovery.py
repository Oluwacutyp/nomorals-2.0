"""Gate discovery + interaction receipts — one live contract.

The browser package owns gate discovery: what a gate IS on a
page, what was actually done to it, and whether the page is
open afterwards.  ``social/chat/webnav.py`` (the navigation
pipeline) consumes this contract; it does not re-implement it.

The contract closes the WEBSHOT_CONSENT_CODE_FINDINGS handoff
(doc-queue §39) and the BROWSER_CAPTURE spec (doc-queue §52) —
one build for both.  The handoff's eight findings, checked
against the code at 9bce92b6:

1. Consent click mis-targeting — the detector picked the
   blocking layer's control, but the clicker re-resolved by
   class/text and could land on a different control.  The
   clicker now resolves from the observation itself (its
   selector, scoped to its frame) and the receipt records
   the target actually used.
2. Age-detector/clicker apostrophe disagreement — the
   detector folded curly quotes, the clicker did not, so a
   found age gate could not be clicked.  The observation
   carries both the folded label and the raw label, and the
   clicker tries the raw form first, then the folded one.
   The folding lives here, once, shared by both sides.
3. Hidden dialog treated as standing — the scanner recorded
   no visibility, so a ``display:none`` dialog's controls
   counted as standing gates.  Controls now carry
   ``visible``; hidden ones never become observations.
4. Fixed 600 ms verification — a click was followed by one
   fixed wait and one re-read.  Verification is now a settle
   loop: re-discover until the gate kind is gone or the
   deadline (5 s) hits, and the receipt records the checks.
5. Missing frame discovery — detection read the top document
   only, so a gate living in an iframe was never seen.  The
   probe gains a ``frames()`` seam (tolerant: probes without
   it just report the main document) and every observation
   records which frame it came from.
6. Ignored ARIA controls — only ``button``/``a``/``input``
   were collected.  Elements with ``role="button"`` and
   ``aria-label`` controls are collected too.
7. Remembered-consent suppression — session memory SKIPPED
   a detected gate.  Memory is now a journal only: it
   records what was cleared, and a detected gate is always
   verified gone, memory or not.
8. Swallowed failures — failed attempts vanished when the
   pass ended clean.  Every attempt lands in an
   :class:`InteractionReceipt`, and the :class:`JobProjection`
   carries them all — a success never drops a failure.

A receipt is honest by construction: ``outcome`` is the
measured outcome (re-discovery after the click), never a
claim from HTML presence.  ``JobProjection.diagnostics()``
is the P0 honest text: what was actually clicked/found and
what was not.
"""

from __future__ import annotations

import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "GateDiscovery",
    "GateObservation",
    "InteractionReceipt",
    "JobProjection",
    "discover_gates",
    "fold_text",
    "host_of",
    "norm_text",
    "run_gate_pass",
    "settle_kind",
]


# ── text folding (shared by detector AND clicker) ──────────


def fold_text(s: str) -> str:
    """Fold typographic quotes/dashes and collapse whitespace.
    Real gate buttons ship "I'm 18 or older" with a curly
    apostrophe — the folded form is what both sides match on."""
    out = str(s or "")
    for fancy, plain in (("’", "'"), ("‘", "'"),
                         ("“", '"'), ("”", '"'),
                         ("–", "-"), ("—", "-")):
        out = out.replace(fancy, plain)
    return " ".join(out.split())


def norm_text(s: str) -> str:
    """The folded, lowercased form labels are matched on."""
    return fold_text(s).lower()


# ── gate markers (moved from social/chat/webnav.py) ─────────
#
# The marker tables are seeds beside the heuristics, never a
# closed list: a per-site shape can be added without touching
# the discovery algorithm.


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


def _is_hidden(a: dict[str, str]) -> bool:
    """True when the element itself is hidden: the hidden
    attribute, or display:none / visibility:hidden in its
    inline style.  A hidden dialog is not a standing gate —
    its controls cannot be clicked and must never be
    reported as blocking the page."""
    if "hidden" in a:
        return True
    style = (a.get("style") or "").lower().replace(" ", "")
    return ("display:none" in style
            or "visibility:hidden" in style
            or "visibility:collapse" in style)


class _ClickScanner(HTMLParser):
    """Collects clickable controls (buttons, links, inputs,
    and elements with ``role="button"``) with their text,
    classes and ids — whether each sits inside the blocking
    layer (see :func:`_is_layer`), and whether it is
    actually visible (see :func:`_is_hidden`).  A control
    inside a hidden subtree is collected with
    ``visible=False`` and never becomes a gate candidate."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.controls: list[dict[str, str]] = []
        self.forms_with_dob = False
        self._open: dict[str, str] | None = None
        self._text: list[str] = []
        # Stack of (tag, is_layer, is_visible) for ancestors.
        self._stack: list[tuple[str, bool, bool]] = []

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]
                        ) -> None:
        a = {k: (v or "") for k, v in attrs}
        layer = _is_layer(a)
        ancestors_visible = all(
            vis for _, _, vis in self._stack)
        visible = ancestors_visible and not _is_hidden(a)
        is_clickable = tag in ("button", "a") or (
            tag == "input"
            and a.get("type", "").lower()
            in ("button", "submit", "")) or (
            (a.get("role") or "").lower() == "button")
        if is_clickable:
            self._open = {
                "tag": tag,
                "id": a.get("id", ""),
                "class": a.get("class", ""),
                "role": a.get("role", ""),
                "aria-label": a.get("aria-label", ""),
                "value": a.get("value", ""),
                "text": "",
                "blocking": "1" if layer or any(
                    up for _, up, _ in self._stack) else "",
                "visible": "1" if visible else "",
            }
            self._text = []
            if tag == "input" and a.get("value"):
                self._finish()
        if tag not in _VOID_TAGS:
            self._stack.append((tag, layer, visible))
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
        text = " ".join(" ".join(self._text).split())
        if not text or not any(ch.isalnum() for ch in text):
            # An icon-only control names itself by aria-label
            # ("✓" says nothing; "Accept all" does).
            labelled = " ".join(
                (self._open.get("aria-label") or "").split())
            if labelled:
                text = labelled
        self._open["text"] = text
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


def host_of(url: str) -> str:
    with suppress(Exception):
        return (urlsplit(url).hostname or "").lower()
    return ""


# ── the contract: observation, receipt, projection ──────────


@dataclass
class GateObservation:
    """One observed gate: what kind, which control clears
    it, where it lives, and whether it is actually
    visible.  ``frame`` is "" for the main document, else
    the frame's name/URL; ``source`` names the document
    ("main" or "frame:<name>")."""

    kind: str                # cookie | consent | age | age_dob
    label: str               # the control's visible text, folded
    selector: str = ""       # css hint, when the page gives one
    host: str = ""
    #: The control sits in the layer that is actually
    #: BLOCKING the page (a modal / overlay / dialog),
    #: not in an inline banner or a footer — the pass
    #: works the blocking layer's control first, because
    #: a covered control cannot be clicked at all.
    blocking: bool = False
    #: False when the control sits in a hidden subtree —
    #: a hidden dialog is not a standing gate.
    visible: bool = True
    #: "" = main document; else the frame's name/URL.
    frame: str = ""
    source: str = "main"
    #: The control's text exactly as the page spells it
    #: (case + curly quotes kept) — the clicker tries
    #: this form first, then the folded label, so the
    #: detector and the clicker can never disagree about
    #: an apostrophe again.
    raw_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label,
                "selector": self.selector, "host": self.host,
                "blocking": self.blocking,
                "visible": self.visible, "frame": self.frame,
                "source": self.source,
                "raw_label": self.raw_label}


@dataclass
class InteractionReceipt:
    """One attempted interaction with a gate — the actual
    target, the measured outcome, and how it was verified.
    The outcome is measured (re-discovery after the click),
    never assumed from HTML presence."""

    observation: GateObservation
    action: str = "click"    # click | refuse
    #: What was actually used to act: the selector, the
    #: role+name form, or the frame search — never the
    #: label the detector guessed at.
    target: str = ""
    #: cleared | stayed | would_not_click | refused | error
    outcome: str = ""
    #: rediscovered_absent | rediscovered_present |
    #: no_click | no_reader
    verification: str = ""
    #: Settle checks run for this attempt.
    attempts: int = 1
    duration_ms: float = 0.0
    #: Human detail (the refusal reason, the click error).
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"observation": self.observation.to_dict(),
                "action": self.action, "target": self.target,
                "outcome": self.outcome,
                "verification": self.verification,
                "attempts": self.attempts,
                "duration_ms": self.duration_ms,
                "detail": self.detail}

    def honest_line(self) -> str:
        """One honest line: what was actually clicked/found
        and what was not."""
        obs = self.observation
        where = obs.source
        if self.outcome == "cleared":
            return (f"cleared {obs.kind} gate "
                    f"({obs.label!r} in {where}, {self.target}, "
                    f"verified gone after {self.attempts} "
                    f"check(s) in {self.duration_ms:.0f}ms)")
        if self.outcome == "stayed":
            return (f"the {obs.kind} gate "
                    f"({obs.label!r} in {where}, {self.target}) "
                    f"was clicked but stayed up after "
                    f"{self.attempts} checks")
        if self.outcome == "would_not_click":
            return (f"the {obs.kind} gate "
                    f"({obs.label!r} in {where}) "
                    f"would not click")
        if self.outcome == "refused":
            return f"{obs.kind} gate refused: {self.detail}"
        if self.outcome == "error":
            return (f"the {obs.kind} gate "
                    f"({obs.label!r} in {where}) click raised: "
                    f"{self.detail}")
        return f"{obs.kind} gate: {self.outcome}"


@dataclass
class JobProjection:
    """The job-level projection: every gate observed, every
    attempt receipted, the final state of the page.  Failed
    attempts are carried here — a success never drops a
    failure."""

    observations: list[GateObservation] = field(
        default_factory=list)
    receipts: list[InteractionReceipt] = field(
        default_factory=list)
    cleared: list[str] = field(default_factory=list)
    refused: list[GateObservation] = field(
        default_factory=list)
    remaining: list[GateObservation] = field(
        default_factory=list)
    #: open | gate_standing | refused
    final_state: str = "open"

    def diagnostics(self) -> str:
        """The P0 honest text: what was actually clicked /
        found and what was not."""
        lines: list[str] = []
        if self.final_state == "refused":
            lines.append(
                "the page asks for a date of birth to "
                "verify age — Devon does not invent one, "
                "so the gate stays up")
        elif self.final_state == "gate_standing":
            kinds = ", ".join(o.kind for o in self.remaining)
            lines.append(
                f"gate(s) still standing after the pass: "
                f"{kinds} — the capture would be a photo "
                "of the gate, so nothing was shot")
        elif self.cleared:
            lines.append("cleared: " + ", ".join(self.cleared))
        else:
            lines.append("no gates standing")
        failed = [r.honest_line() for r in self.receipts
                  if r.outcome in ("stayed", "would_not_click",
                                   "error")]
        if failed:
            if self.final_state == "gate_standing":
                lines.append("tried: " + "; ".join(failed))
            else:
                lines.append("failed along the way: "
                             + "; ".join(failed))
        if self.final_state != "open":
            cleared_lines = [
                r.honest_line() for r in self.receipts
                if r.outcome == "cleared"]
            if cleared_lines:
                lines.append("cleared: "
                             + "; ".join(cleared_lines))
        return " — ".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"observations": [o.to_dict() for o in
                                 self.observations],
                "receipts": [r.to_dict() for r in self.receipts],
                "cleared": list(self.cleared),
                "refused": [o.to_dict() for o in self.refused],
                "remaining": [o.to_dict() for o in
                              self.remaining],
                "final_state": self.final_state}

    @classmethod
    def merge(cls, *projections: JobProjection
              ) -> JobProjection:
        """Combine successive passes (before + after a wall
        check) into one projection: every observation and
        receipt is kept; the final state is the last pass's."""
        observations: list[GateObservation] = []
        receipts: list[InteractionReceipt] = []
        cleared: list[str] = []
        refused: list[GateObservation] = []
        remaining: list[GateObservation] = []
        final_state = "open"
        for proj in projections:
            for obs in proj.observations:
                if obs not in observations:
                    observations.append(obs)
            receipts.extend(proj.receipts)
            cleared.extend(proj.cleared)
            refused.extend(proj.refused)
        if projections:
            last = projections[-1]
            remaining = list(last.remaining)
            final_state = last.final_state
        return cls(observations=observations, receipts=receipts,
                   cleared=cleared, refused=refused,
                   remaining=remaining, final_state=final_state)


# ── discovery (pure) ────────────────────────────────────────


class GateDiscovery:
    """Pure gate discovery over documents: the main HTML
    plus any frames.  The same HTML always gives the same
    observations.  Order matters: the BLOCKING layer's
    control first (a covered control cannot be clicked at
    all), and within a tier the age-gate confirm outranks
    generic consent/cookie controls — a page can stack
    them, and the pass works them in this order."""

    @staticmethod
    def discover(main_html: str, url: str = "",
                 frames: Any = ()) -> list[GateObservation]:
        """``frames`` is a sequence of (name, html) for the
        page's iframes — a gate living in a frame is a real
        gate, with its frame recorded on the observation."""
        docs: list[tuple[str, str]] = [
            ("", str(main_html or ""))]
        for name, fhtml in (frames or ()):
            docs.append((str(name or ""), str(fhtml or "")))
        host = host_of(url)
        by_key: dict[tuple[str, str], GateObservation] = {}
        # The control behind each collected observation, by
        # the same key — the cookie scan must skip the exact
        # control already collected as consent (the label
        # tables overlap), never the whole kind.
        collected: dict[tuple[str, str], dict[str, str]] = {}

        def add(kind: str, control: dict[str, str],
                label: str, raw_label: str,
                frame: str) -> None:
            obs = GateObservation(
                kind=kind, label=label, raw_label=raw_label,
                selector=_selector_for(control), host=host,
                blocking=bool(control.get("blocking")),
                visible=bool(control.get("visible")),
                frame=frame,
                source=("main" if not frame
                        else f"frame:{frame}"))
            key = (kind, frame)
            old = by_key.get(key)
            # One candidate per kind per frame; the blocking
            # layer's control replaces a same-kind control
            # that merely sits on the page (a footer age
            # link is not the age gate when a modal is
            # standing).
            if old is None or (obs.blocking
                               and not old.blocking):
                by_key[key] = obs
                collected[key] = control

        for frame_name, text in docs:
            low = text.lower()
            scanner = _ClickScanner()
            # A torn page scans as whatever was collected.
            with suppress(Exception):
                scanner.feed(text)
            controls = [c for c in scanner.controls
                        if c.get("visible")]
            labels = [(c, norm_text(c["text"] or c["value"]),
                       " ".join(
                           (c["text"] or c["value"]).split()))
                      for c in controls]

            # ── age gates ──
            age_context = any(m in low for m in _AGE_MARKERS)
            site_labels = SITE_AGE_LABELS.get(host, ())
            for control, label, raw in labels:
                if not label:
                    continue
                if label in _AGE_LABELS or \
                        label in site_labels or \
                        any(a in label for a in _AGE_LABELS) or (
                        age_context and label in _AGE_BARE_LABELS):
                    add("age", control, label, raw, frame_name)
            if ("age", frame_name) not in by_key and \
                    age_context and scanner.forms_with_dob:
                # A date-of-birth form is an age gate Devon
                # will not invent an answer for.
                by_key[("age_dob", frame_name)] = GateObservation(
                    kind="age_dob", label="date-of-birth form",
                    host=host, frame=frame_name,
                    source=("main" if not frame_name else
                            f"frame:{frame_name}"),
                    raw_label="date-of-birth form")

            # ── consent walls ──
            consent_context = any(
                m in low for m in _CONSENT_MARKERS)
            if consent_context:
                for control, label, raw in labels:
                    if label in _CONSENT_LABELS or \
                            label.startswith("accept"):
                        add("consent", control, label, raw,
                            frame_name)

            # ── cookie banners ──
            cookie_context = any(
                m in low for m in _COOKIE_MARKERS)
            if cookie_context:
                # Control-level dedup, not kind-level: the
                # consent and cookie label tables overlap
                # ("accept all" matches both), so the button
                # already collected as consent must not be
                # collected again as a cookie — but a distinct
                # cookie button in the same frame is a real
                # second gate (a page can stack them).
                consent_control = collected.get(
                    ("consent", frame_name))
                for control, label, raw in labels:
                    if control is consent_control:
                        continue
                    if label in _COOKIE_LABELS:
                        add("cookie", control, label, raw,
                            frame_name)

        order = {"age": 0, "age_dob": 0, "consent": 1,
                 "cookie": 2}
        return sorted(
            by_key.values(),
            key=lambda g: (not g.blocking,
                           order.get(g.kind, 3),
                           0 if g.frame == "" else 1))


def discover_gates(html: str, url: str = "",
                   frames: Any = ()) -> list[GateObservation]:
    """Pure discovery, one call: the main document plus any
    frames, in pass order."""
    return GateDiscovery.discover(html, url, frames)


# ── the pass (over a live probe) ────────────────────────────


def _emit(on_step: Any, text: str) -> None:
    if on_step is not None:
        with suppress(Exception):
            on_step(text)


def _probe_frames(probe: Any) -> list[tuple[str, str]]:
    """Frames the probe can see: (name, html) per iframe.
    Tolerant — a probe without the seam just reports the
    main document, and a frame that will not give its HTML
    is skipped, never a reason to stop the pass."""
    frames_fn = getattr(probe, "frames", None)
    if not callable(frames_fn):
        return []
    try:
        out = frames_fn()
    except Exception:  # noqa: BLE001 - frames are best effort
        return []
    cleaned: list[tuple[str, str]] = []
    for item in (out or []):
        try:
            name, html = item
        except (TypeError, ValueError):
            continue
        cleaned.append((str(name or ""), str(html or "")))
    return cleaned


def _discover_on_probe(probe: Any) -> list[GateObservation]:
    """Re-discover from the live probe: main document plus
    its frames."""
    return discover_gates(probe.html(), probe.url(),
                          _probe_frames(probe))


def _target_text(probe: Any, obs: GateObservation) -> str:
    """What was actually used to act: the clicker's own
    record when it has one, else the observation's hint."""
    last = getattr(probe, "last_click", None)
    if isinstance(last, dict) and last.get("summary"):
        return str(last["summary"])
    if obs.selector:
        target = f"selector {obs.selector}"
    elif obs.raw_label:
        target = f"text {obs.raw_label!r}"
    else:
        target = f"text {obs.label!r}"
    if obs.frame:
        target += f" in frame {obs.frame!r}"
    return target


def settle_kind(probe: Any, kind: str, *,
                deadline_s: float = 5.0,
                poll_s: float = 0.25
                ) -> tuple[bool, int, float]:
    """Verify a gate kind is gone: re-discover until no
    observation of that kind remains, or the deadline
    hits.  Returns (gone, checks, elapsed_ms).  The wait
    is driven by disappearance — never a fixed sleep."""
    deadline = time.monotonic() + max(0.0, deadline_s)
    started = time.monotonic()
    checks = 0
    while True:
        checks += 1
        if not any(o.kind == kind for o in
                   _discover_on_probe(probe)):
            return (True, checks,
                    (time.monotonic() - started) * 1000.0)
        if time.monotonic() >= deadline:
            return (False, checks,
                    (time.monotonic() - started) * 1000.0)
        probe.wait(poll_s)


def run_gate_pass(probe: Any, *, memory: Any = None,
                  on_step: Any = None, rounds: int = 3,
                  settle_deadline_s: float = 5.0,
                  settle_poll_s: float = 0.25
                  ) -> JobProjection:
    """Clear cookie/consent/age gates, then VERIFY none is
    still standing — every attempt receipted.  A click
    counts as handled only when the gate verifiably
    clears; a click that refuses or doesn't clear is a
    failed attempt on THAT control, and the pass moves to
    the next candidate instead of dying on — or retrying —
    a covered, unrelated button.

    Session ``memory`` is a journal only: it records what
    was cleared, and a detected gate is ALWAYS verified
    gone, memory or not — memory never skips a gate.  A
    date-of-birth form ends the pass at once with
    final_state "refused": Devon does not invent an age,
    and nothing is clicked on such a page.  The caller
    turns a non-open final state into its own refusal
    error; this pass reports truth and raises nothing."""
    observations: list[GateObservation] = []
    receipts: list[InteractionReceipt] = []
    cleared: list[str] = []
    refused: list[GateObservation] = []
    failed: dict[tuple[str, str, str], str] = {}
    for _round in range(max(1, rounds)):
        found = _discover_on_probe(probe)
        if not found:
            break
        for obs in found:
            if obs not in observations:
                observations.append(obs)
        progressed = False
        for obs in found:
            if obs.kind == "age_dob":
                receipts.append(InteractionReceipt(
                    observation=obs, action="refuse",
                    outcome="refused",
                    verification="no_reader",
                    detail="the page asks for a date of "
                           "birth to verify age — Devon does "
                           "not invent one, so the gate "
                           "stays up"))
                refused.append(obs)
                remaining = _discover_on_probe(probe)
                return JobProjection(
                    observations=observations, receipts=receipts,
                    cleared=cleared, refused=refused,
                    remaining=remaining, final_state="refused")
            key = (obs.kind, obs.label, obs.frame)
            if key in failed:
                continue
            _emit(on_step, f"clearing a {obs.kind} gate")
            started = time.monotonic()
            clicked = False
            click_error = ""
            try:
                clicked = bool(probe.click_gate(obs))
            except Exception as exc:  # noqa: BLE001 - click
                click_error = str(exc)[:300]  # failure is data
            target = _target_text(probe, obs)
            duration_ms = ((time.monotonic() - started)
                           * 1000.0)
            if click_error:
                receipts.append(InteractionReceipt(
                    observation=obs, target=target,
                    outcome="error", verification="no_click",
                    duration_ms=duration_ms,
                    detail=f"the {obs.kind} gate "
                           f"({obs.label!r}) click raised: "
                           f"{click_error}"))
                failed[key] = "click raised"
                continue
            if not clicked:
                receipts.append(InteractionReceipt(
                    observation=obs, target=target,
                    outcome="would_not_click",
                    verification="no_click",
                    duration_ms=duration_ms,
                    detail=f"the {obs.kind} gate "
                           f"({obs.label!r}) would not click"))
                failed[key] = "would not click"
                continue
            gone, checks, _elapsed = settle_kind(
                probe, obs.kind,
                deadline_s=settle_deadline_s,
                poll_s=settle_poll_s)
            duration_ms = ((time.monotonic() - started)
                           * 1000.0)
            if not gone:
                receipts.append(InteractionReceipt(
                    observation=obs, target=target,
                    outcome="stayed",
                    verification="rediscovered_present",
                    attempts=checks, duration_ms=duration_ms,
                    detail=f"the {obs.kind} gate "
                           f"({obs.label!r}) was clicked but "
                           "stayed up"))
                failed[key] = "stayed up"
                continue
            receipts.append(InteractionReceipt(
                observation=obs, target=target,
                outcome="cleared",
                verification="rediscovered_absent",
                attempts=checks, duration_ms=duration_ms))
            cleared.append(obs.kind)
            if memory is not None:
                with suppress(Exception):
                    memory.remember(obs)
            progressed = True
        if not progressed:
            break
    remaining = _discover_on_probe(probe)
    final_state = "gate_standing" if remaining else "open"
    return JobProjection(
        observations=observations, receipts=receipts,
        cleared=cleared, refused=refused, remaining=remaining,
        final_state=final_state)
