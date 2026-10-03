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
# to input complexity instead of a fixed 4-turn window.

def _dialogue_depth(text_len: int, ambiguous: bool = False) -> tuple[int, int]:
    """Return (turn_count, chars_per_turn) adapted to input complexity.

    Short greetings need almost no history; long or ambiguous messages
    benefit from a wider window.
    """
    if text_len < 30:
        return (2, 120)
    if text_len < 120:
        return (4, 160)
    if ambiguous or text_len >= 300:
        return (8, 200)
    return (6, 180)


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
                    # dynamic depth: adapt to input complexity
                    n_turns, chars = _dialogue_depth(
                        ctx.text_len, _looks_ambiguous(ctx.text))
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

    Returns (reply_or_None, context). ``None`` reply means "fall through to
    the companion conversation path" — not an error.
    """
    # 1 — context pack
    ctx = build_loop_context(message, mind=mind, runtime=runtime,
                             chat_key=chat_key)
    ctx.is_owner_dm = True  # run_loop is only called past the owner-DM gate

    # 2 — goal inference (fast path + decide live inside mind.handle today;
    #     kept here as the explicit seam for the next refactor)
    # 3 — plan, 4 — execute: delegated to the mind's dispatch
    reply = mind._dispatch_from_loop(ctx, text, message=message)

    # 5 — verify
    # (intent kind is recovered from the job the dispatch just recorded)
    kind = _last_intent_kind(mind)
    ok, reply = verify_dispatch(kind, reply)

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
