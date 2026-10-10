"""The core agent loop — one integrated decision path for owner DM input.

Every non-command message in the owner's DM flows through six explicit steps:

1. CONTEXT PACK — cheap, local: who/where/what's open, no model calls.
2. GOAL INFERENCE — what does "done" look like? (CoreMind.decide)
3. PLAN — choose the organ matching the deliverable. (CoreMind._dispatch)
4. EXECUTE — the organ runs against real tools.
5. VERIFY — per-organ acceptance gates; fake success is failure.
6. REPLY — human-readable text back in the same chat.

This module owns steps 1 and 5 as explicit, testable units. Steps 2–4 live
in ``nomorals.agents.coremind`` (CoreMind.decide/_dispatch); step 6 is the
chat gateway's ``send``. The loop is wired together in
``CoreMind.handle`` — see the ``run_loop`` entry point below.
"""

from __future__ import annotations

import abc
import logging
import re
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


# ── step 1: context pack ──────────────────────────────────────────────────

@dataclass
class LoopContext:
    """Everything the loop needs to decide, gathered cheaply and locally.

    No model calls, no network, no disk scans beyond tiny DB reads. Built
    once per inbound message and threaded through inference → plan →
    execute → verify.
    """

    # origin stamp — the reply MUST go back here
    platform: str = ""
    chat_id: str = ""
    chat_kind: str = ""
    chat_key: str = ""
    thread_id: str = ""

    # who's talking
    is_owner_dm: bool = False
    owner_mode: bool = False  # owner identity asserted + accepted this session

    # what's already happening
    live_game: str | None = None
    open_jobs: int = 0
    pending_clarification: bool = False

    # the message itself
    text: str = ""
    text_len: int = 0
    has_media: bool = False
    is_command: bool = False

    # machine we're on
    resource_profile: str = "workstation"  # workstation | laptop | termux

    # recent dialogue (tiny — last few turns, truncated)
    recent_turns: list[str] = field(default_factory=list)

    def origin(self) -> dict[str, str]:
        """The stamp every WORK job carries so notify/deliver returns here."""
        return {
            "platform": self.platform,
            "chat_id": self.chat_id,
            "chat_key": self.chat_key,
            "thread_id": self.thread_id,
        }


# ── dynamic context sizing ──────────────────────────────────────────────
# The owner's principle: dynamic where it matters. Dialogue depth adapts
# to the message via a STRATEGY CHAIN — each strategy sees the signals
# and the running (turns, chars) depth and returns an adjusted depth.
# The default chain reproduces the old length bands and then layers
# ambiguity/media/command/profile adjustments on top; custom chains can
# be injected for per-deployment tuning.

#: (turns, chars-per-turn)
Depth = tuple[int, int]


@dataclass
class DepthSignals:
    """Everything the depth chain may consider.  Cheap, local, no model."""

    text_len: int = 0
    ambiguous: bool = False
    has_media: bool = False
    is_command: bool = False
    resource_profile: str = "workstation"  # workstation | laptop | termux


class DepthStrategy(abc.ABC):
    """One adjustable link in the dialogue-depth chain."""

    name: str = "base"

    @abc.abstractmethod
    def adjust(self, depth: Depth, signals: DepthSignals) -> Depth:
        """Return the depth after this strategy's consideration."""


class LengthBandDepth(DepthStrategy):
    """Base depth from text-length bands.  Bands are data, not branches."""

    name = "length_band"

    def __init__(
        self,
        bands: tuple[tuple[float, Depth], ...] = (
            (30, (2, 120)),
            (120, (4, 160)),
            (300, (6, 180)),
            (float("inf"), (8, 200)),
        ),
    ) -> None:
        self.bands = bands

    def adjust(self, depth: Depth, signals: DepthSignals) -> Depth:
        for limit, band_depth in self.bands:
            if signals.text_len < limit:
                return band_depth
        return self.bands[-1][1]


class AmbiguityBoost(DepthStrategy):
    """Questions / alternatives / hedges deserve a wider window."""

    name = "ambiguity_boost"

    def __init__(self, extra_turns: int = 2, extra_chars: int = 20) -> None:
        self.extra_turns = extra_turns
        self.extra_chars = extra_chars

    def adjust(self, depth: Depth, signals: DepthSignals) -> Depth:
        if signals.ambiguous:
            return (depth[0] + self.extra_turns, depth[1] + self.extra_chars)
        return depth


class MediaBoost(DepthStrategy):
    """Media messages usually refer back to earlier context."""

    name = "media_boost"

    def __init__(self, extra_turns: int = 2) -> None:
        self.extra_turns = extra_turns

    def adjust(self, depth: Depth, signals: DepthSignals) -> Depth:
        if signals.has_media:
            return (depth[0] + self.extra_turns, depth[1])
        return depth


class CommandNarrow(DepthStrategy):
    """Commands are self-contained — don't drag history along."""

    name = "command_narrow"

    def __init__(self, max_turns: int = 2, max_chars: int = 120) -> None:
        self.max_turns = max_turns
        self.max_chars = max_chars

    def adjust(self, depth: Depth, signals: DepthSignals) -> Depth:
        if signals.is_command:
            return (min(depth[0], self.max_turns), min(depth[1], self.max_chars))
        return depth


class ProfileCap(DepthStrategy):
    """Small machines get a smaller window — a cap, not a redesign."""

    name = "profile_cap"

    def __init__(self, caps: dict[str, Depth] | None = None) -> None:
        self.caps = caps or {"termux": (4, 160)}

    def adjust(self, depth: Depth, signals: DepthSignals) -> Depth:
        cap = self.caps.get(signals.resource_profile)
        if cap is not None:
            return (min(depth[0], cap[0]), min(depth[1], cap[1]))
        return depth


def default_depth_strategies() -> list[DepthStrategy]:
    """The stock chain: band → ambiguity → media → command → profile cap."""
    return [
        LengthBandDepth(),
        AmbiguityBoost(),
        MediaBoost(),
        CommandNarrow(),
        ProfileCap(),
    ]


def dialogue_depth(
    signals: DepthSignals,
    strategies: list[DepthStrategy] | None = None,
) -> Depth:
    """Run the depth chain over ``signals``.  Never raises."""
    depth: Depth = (4, 160)
    try:
        for strategy in strategies if strategies is not None else default_depth_strategies():
            depth = strategy.adjust(depth, signals)
    except Exception:  # noqa: BLE001 — depth must never break the loop
        _log.debug("depth chain degraded", exc_info=True)
    return depth


def _dialogue_depth(text_len: int, ambiguous: bool = False) -> Depth:
    """Backward-compatible entry: the old two-argument call delegates to
    the chain with only length/ambiguity signals."""
    return dialogue_depth(DepthSignals(text_len=text_len, ambiguous=ambiguous))


def _looks_ambiguous(text: str) -> bool:
    """Cheap ambiguity signal: questions, alternatives, hedges."""
    t = text.lower()
    return bool(
        t.count("?") >= 1
        or " or " in t
        or "maybe" in t
        or "not sure" in t
        or t.startswith(("what about", "how about"))
    )


def build_loop_context(
    message: Any,
    *,
    mind: Any = None,
    runtime: Any = None,
    chat_key: str = "",
) -> LoopContext:
    """Build the context pack. Cheap and local — never raises.

    ``message`` is a ChatMessage (or a duck-typed stand-in in tests).
    ``mind`` is the CoreMind (for live-game + pending lookups).
    ``runtime`` is the PartnerRuntime (for owner-mode + profile).
    """
    ctx = LoopContext()
    try:
        chat = getattr(message, "chat", None)
        if chat is not None:
            ctx.platform = str(getattr(chat, "platform", "") or "")
            ctx.chat_id = str(getattr(chat, "chat_id", "") or "")
            ctx.chat_kind = str(getattr(chat, "kind", "") or "")
            ctx.thread_id = str(getattr(chat, "thread_id", "") or "")
            try:
                ctx.chat_key = chat.key
            except Exception:  # noqa: BLE001
                ctx.chat_key = chat_key
        if not ctx.chat_key:
            ctx.chat_key = chat_key

        text = str(getattr(message, "text", "") or "")
        ctx.text = text
        ctx.text_len = len(text)
        ctx.has_media = bool(getattr(message, "media", None))
        ctx.is_command = text.strip().startswith("/")

        # live game for this chat (priority 1 in the decision path)
        if mind is not None:
            try:
                ctx.live_game = mind._live_game(ctx.chat_key)
            except Exception:  # noqa: BLE001
                ctx.live_game = None
            try:
                ctx.pending_clarification = mind._get_pending(ctx.chat_key) is not None
            except Exception:  # noqa: BLE001
                pass
            try:
                jobs = getattr(mind, "_jobs", None)
                if isinstance(jobs, dict):
                    ctx.open_jobs = sum(
                        1 for j in jobs.values()
                        if isinstance(j, dict) and not j.get("done")
                    )
            except Exception:  # noqa: BLE001
                pass

        # owner identity / mode
        if runtime is not None:
            try:
                brain = getattr(runtime, "brain", None)
                rel = getattr(brain, "relationship", None)
                if rel is not None:
                    facts = rel.get_facts() if hasattr(rel, "get_facts") else {}
                    ctx.owner_mode = bool(
                        (facts.get("identity_confirmed") if isinstance(facts, dict) else None)
                    )
            except Exception:  # noqa: BLE001
                pass

        # resource profile: termux vs workstation
        try:
            settings = getattr(runtime, "settings", None) or getattr(
                getattr(mind, "context", None), "settings", None)
            if settings is not None:
                prof = getattr(settings, "profile", "") or ""
                if prof:
                    ctx.resource_profile = str(prof)
        except Exception:  # noqa: BLE001
            pass
        # termux heuristic when profile unset: well-known termux home prefix
        if ctx.resource_profile == "workstation":
            try:
                import os
                if os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
                    ctx.resource_profile = "termux"
            except Exception:  # noqa: BLE001
                pass

        # recent dialogue — last 4 turns, truncated (history lives on brain)
        if runtime is not None:
            try:
                brain = getattr(runtime, "brain", None)
                if brain is not None and hasattr(brain, "_history"):
                    # dynamic depth: the strategy chain adapts the window
                    # to the message (length, ambiguity, media, command,
                    # machine profile) instead of a hardcoded branch.
                    n_turns, chars = dialogue_depth(DepthSignals(
                        text_len=ctx.text_len,
                        ambiguous=_looks_ambiguous(ctx.text),
                        has_media=ctx.has_media,
                        is_command=ctx.is_command,
                        resource_profile=ctx.resource_profile,
                    ))
                    hist = brain._history(ctx.chat_key, limit=n_turns)
                    for m in hist[-n_turns:]:
                        t = getattr(m, "text", "")
                        if t:
                            who = "owner" if getattr(m, "incoming", True) else "devon"
                            ctx.recent_turns.append(f"{who}: {t[:chars]}")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 - context pack never breaks the loop
        _log.debug("loop context build degraded", exc_info=True)
    return ctx


# ── step 5: verify ────────────────────────────────────────────────────────

# Signatures of fake success — a dispatch that returns one of these has NOT
# actually done the work. The loop converts them to honest failures.
_FAKE_SUCCESS_PATTERNS = (
    re.compile(r"0 passed,\s*0 failed", re.IGNORECASE),
    re.compile(r"no tests? (directory|found).*nothing ran", re.IGNORECASE),
    re.compile(r"nothing to do", re.IGNORECASE),
)


def _looks_fake_success(reply: str, intent_kind: str) -> str | None:
    """Return a reason when a dispatch reply smells like fake success."""
    if not reply:
        return None
    # coding/build: empty or trivial output with no artifact is failure
    if intent_kind in ("build",):
        for pat in _FAKE_SUCCESS_PATTERNS:
            if pat.search(reply):
                return f"coding reported no real outcome ({pat.pattern[:40]}…)"
        # "code done" with no file, no output, no tests mentioned
        if re.search(r"code done|build complete|success", reply, re.IGNORECASE):
            has_artifact = bool(re.search(
                r"\.(py|js|ts|sh|html|md|pdf|zip)\b|/[\w\-./]+|artifact|saved to|wrote ",
                reply))
            has_output = bool(re.search(
                r"output:|result:|ran in|passed", reply, re.IGNORECASE))
            if not has_artifact and not has_output:
                return "coding claimed success with no artifact or output"
    return None


def verify_dispatch(intent_kind: str, reply: str | None) -> tuple[bool, str | None]:
    """Step 5 of the loop: verify the organ actually did the work.

    Returns (ok, reply). When verification fails, ok=False and reply is an
    honest failure message — never the fake success text.
    """
    if reply is None:
        # None means "fall through to conversation" — not a failure.
        return True, None
    reason = _looks_fake_success(reply, intent_kind)
    if reason is not None:
        _log.warning("loop verify rejected fake success for %s: %s",
                     intent_kind, reason)
        return False, (
            "that didn't actually complete — the work organ reported "
            f"success with no real outcome ({reason}). I've logged it; "
            "tell me what you wanted and I'll route it properly."
        )
    return True, reply


# ── the integrated loop ───────────────────────────────────────────────────

def run_loop(
    mind: Any,
    text: str,
    *,
    message: Any,
    chat_key: str,
    runtime: Any = None,
) -> tuple[str | None, LoopContext]:
    """Run the full six-step loop for one inbound owner-DM message.

    THE single entry point — ``CoreMind.handle`` delegates here after its
    structural gate.  There is no second path.

    Returns (reply_or_None, context). ``None`` reply means "fall through to
    the companion conversation path" — not an error.
    """
    # 1 — context pack (cheap, local, never raises)
    ctx = build_loop_context(message, mind=mind, runtime=runtime,
                             chat_key=chat_key)
    ctx.is_owner_dm = True  # run_loop is only called past the owner-DM gate
    ctx.text = text or ""
    ctx.text_len = len(ctx.text)

    # an open clarification? this message may be the answer
    pending = mind._get_pending(chat_key)
    if pending is not None:
        from .coremind import _RE_CANCEL
        if _RE_CANCEL.match(text):
            mind._clear_pending(chat_key)
            return "ok — scrapped. what's next?", ctx
        resolved = mind._pending_resolves(pending, text)
        if resolved is not None:
            mind._clear_pending(chat_key)
            reply = mind._dispatch(resolved, chat_key, message) or None
            ok, reply = verify_dispatch(resolved.kind, reply)
            return reply, ctx
        # not an answer — clear the stale question and fall through
        mind._clear_pending(chat_key)

    # 2 — goal inference, 3 — plan, 4 — execute
    from .coremind import fast_path, Intent
    live_game = ctx.live_game
    if live_game is None:
        try:
            live_game = mind._live_game(chat_key)
        except Exception:  # noqa: BLE001
            live_game = None
    # fast path for trivial chat sits ABOVE the heavy path — no router
    # model call, no swarms, no research loops for greetings/thanks.
    fast = fast_path(text)
    if fast is not None:
        reply, why = fast
        try:
            mind._record_route(Intent("fastchat", 1.0, route="fastchat", why=why))
        except Exception:  # noqa: BLE001
            pass
        return reply, ctx

    intent = mind.decide(text, live_game=live_game, allow_model=True)
    if intent.kind == "chat":
        return None, ctx
    if intent.action == "ask":
        question = mind._question_for(intent)
        mind._set_pending(chat_key, intent, question)
        return question, ctx
    reply = mind._dispatch(intent, chat_key, message) or None

    # 5 — verify: fake success becomes honest failure, never green
    _ok, reply = verify_dispatch(intent.kind, reply)

    # 6 — reply happens at the call site (gateway.send to ctx.origin())
    return reply, ctx


def _last_intent_kind(mind: Any) -> str:
    try:
        routes = getattr(mind, "_route_log", None)
        if routes:
            last = routes[-1]
            if isinstance(last, dict):
                return str(last.get("kind", ""))
            kind = getattr(last, "kind", "")
            if kind:
                return str(kind)
    except Exception:  # noqa: BLE001
        pass
    return ""
