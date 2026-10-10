"""Context management: fit prompts to a model's window, per task-kind.

Nothing in the serving path used to manage context at all — a prompt that
exceeded the model's window failed over through the whole chain and came
back as a generic error, parking healthy providers along the way.  This
module owns the recovery: strategies that shrink a message list to a
token budget, chosen per task-kind instead of one hardcoded truncation.

Strategies
----------
* :class:`TruncateOldest` — drop the oldest non-system turns until the
  budget fits.  The system prompt and the latest turn are sacred.
* :class:`SummarizeMiddle` — keep the system prompt, the first turn and
  the last N turns; compress everything between them with a summarizer
  callable (the brain's own summarizer in the serving path, a stub in
  tests).  When no summarizer is available it degrades to truncation.
* :class:`Compact` — collapse whitespace-runs in every message; a cheap
  first pass that costs no meaning.

:class:`fit_messages` runs the task-kind's strategy chain in order until
the estimate fits or the chain is exhausted.  :class:`fit_prompt` is the
single-string equivalent.

Budgets come from the serving provider's advertised context length when
known (the broker card), else :data:`DEFAULT_CONTEXT_TOKENS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger
from .base import Message, estimate_messages

__all__ = [
    "DEFAULT_CONTEXT_TOKENS",
    "TASK_FIT_CHAINS",
    "CaseFacts",
    "Compact",
    "ContextStrategy",
    "FitResult",
    "PrioritizeEnds",
    "SummarizeMiddle",
    "TruncateOldest",
    "fit_messages",
    "fit_prompt",
    "strategy_for",
    "trim_tool_results",
    "with_cache_breakpoints",
]

_log = get_logger(__name__)

#: Fallback window when the serving provider's is unknown.
DEFAULT_CONTEXT_TOKENS = 8192

#: Keep this fraction of the window for the prompt; the rest is headroom
#: for the completion.
_BUDGET_FRACTION = 0.85

#: task-kind → strategy chain, most conservative first.  "judge" and
#: "research" synthesis genuinely need the middle, so they summarize it;
#: everything else drops old turns — cheaper and usually sufficient.
TASK_FIT_CHAINS: dict[str, tuple[str, ...]] = {
    "chat": ("compact", "truncate_oldest"),
    "intent": ("compact", "truncate_oldest"),
    "summarize": ("compact", "truncate_oldest"),
    "judge": ("compact", "prioritize_ends", "summarize_middle", "truncate_oldest"),
    "research": ("compact", "prioritize_ends", "summarize_middle", "truncate_oldest"),
    "reasoning": ("compact", "prioritize_ends", "summarize_middle", "truncate_oldest"),
    "plan": ("compact", "prioritize_ends", "summarize_middle", "truncate_oldest"),
    "code": ("compact", "truncate_oldest"),
    "creative": ("compact", "truncate_oldest"),
    "vision": ("compact", "truncate_oldest"),
    "embed": ("truncate_oldest",),
}
_DEFAULT_CHAIN = ("compact", "truncate_oldest")

#: Summarized-middle keeps this many recent turns verbatim.
_SUMMARY_KEEP_TURNS = 4

_RE_WS_RUN = re.compile(r"[ \t]{2,}|\n{3,}")


@dataclass
class FitResult:
    """What fitting did."""

    messages: list[Message]
    #: Estimated tokens after fitting.
    tokens: int
    #: Strategies that actually changed something, in order.
    applied: list[str] = field(default_factory=list)
    #: How many non-system turns were dropped.
    dropped_turns: int = 0
    #: True when the middle was summarized (not just dropped).
    summarized: bool = False    #: True when the result still exceeds the budget (chain exhausted).
    overflow: bool = False
    #: Pinned facts that survived verbatim (case-facts block).
    facts_kept: int = 0

    @property
    def ok(self) -> bool:
        return not self.overflow


class ContextStrategy:
    """One way to shrink a message list.  Returns the new list, or the
    input unchanged when the strategy does not apply."""

    name: str = "base"

    def apply(
        self,
        messages: list[Message],
        budget: int,
        ctx: dict[str, Any],
    ) -> list[Message]:
        raise NotImplementedError


class Compact(ContextStrategy):
    """Collapse whitespace runs.  Never drops content — the cheap first pass."""

    name = "compact"

    def apply(self, messages: list[Message], budget: int,
              ctx: dict[str, Any]) -> list[Message]:
        out: list[Message] = []
        changed = False
        for m in messages:
            compacted = _RE_WS_RUN.sub(
                lambda mm: "\n\n" if "\n" in mm.group(0) else " ", m.content)
            if compacted != m.content:
                changed = True
            out.append(Message(role=m.role, content=compacted, name=m.name,
                               tool_call_id=m.tool_call_id, images=list(m.images)))
        if changed:
            _log.debug("context_fit: compact saved ~%d tokens",
                       estimate_messages(messages) - estimate_messages(out))
        return out


class TruncateOldest(ContextStrategy):
    """Drop the oldest non-system turns until the budget fits.

    Every system turn (the system prompt and the case-facts block) and
    the latest user turn are never dropped — they carry the instruction,
    the pinned facts, and the actual request.
    """

    name = "truncate_oldest"

    def apply(self, messages: list[Message], budget: int,
              ctx: dict[str, Any]) -> list[Message]:
        if estimate_messages(messages) <= budget or len(messages) <= 2:
            return messages
        # Every system turn is sacred — the case-facts block rides here.
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        # keep the latest turn sacred; drop from the oldest of the rest
        sacred = rest[-1:] if rest else []
        droppable = rest[:-1] if rest else []
        kept: list[Message] = list(system)
        tail: list[Message] = []
        # rebuild from the newest droppable turn backwards
        for m in reversed(droppable):
            candidate = kept + [m] + tail + sacred
            if estimate_messages(candidate) <= budget:
                tail.insert(0, m)
            # else: dropped
        result = kept + tail + sacred
        dropped = len(messages) - len(result)
        if dropped:
            ctx["dropped_turns"] = ctx.get("dropped_turns", 0) + dropped
            _log.info("context_fit: truncate_oldest dropped %d turn(s)", dropped)
        return result


class SummarizeMiddle(ContextStrategy):
    """Compress the middle of a long conversation into one summary turn.

    Keeps the system prompt, the first turn, and the last
    ``keep_turns`` turns verbatim; everything between them is replaced by
    a single assistant turn carrying the summary.  Needs a ``summarizer``
    callable in ``ctx`` (``fn(text) -> str``); without one it degrades to
    :class:`TruncateOldest`.
    """

    name = "summarize_middle"

    def __init__(self, keep_turns: int = _SUMMARY_KEEP_TURNS) -> None:
        self.keep_turns = max(1, keep_turns)

    def apply(self, messages: list[Message], budget: int,
              ctx: dict[str, Any]) -> list[Message]:
        if estimate_messages(messages) <= budget:
            return messages
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        if len(rest) <= self.keep_turns + 2:
            # too short to have a real middle — let truncation handle it
            return messages
        head = rest[:1]
        tail = rest[-(self.keep_turns):]
        middle = rest[1:-(self.keep_turns)]
        summarizer = ctx.get("summarizer")
        if not callable(summarizer) or not middle:
            return TruncateOldest().apply(messages, budget, ctx)
        middle_text = "\n".join(
            f"{m.role}: {m.content}" for m in middle)[:12000]
        try:
            summary = (summarizer(middle_text) or "").strip()
        except Exception as exc:  # noqa: BLE001 — summarizer is best-effort
            _log.debug("context_fit: summarizer failed (%s); truncating", exc)
            return TruncateOldest().apply(messages, budget, ctx)
        if not summary:
            return TruncateOldest().apply(messages, budget, ctx)
        summary_turn = Message(
            role="assistant",
            content=f"[earlier conversation summarized: {summary}]",
        )
        result = system + head + [summary_turn] + tail
        ctx["summarized"] = True
        ctx["dropped_turns"] = ctx.get("dropped_turns", 0) + len(middle)
        _log.info("context_fit: summarize_middle compressed %d turn(s)",
                  len(middle))
        if estimate_messages(result) > budget:
            # Still over budget: truncate, but the summary turn is sacred —
            # it is the compressed memory of everything dropped.  Dropping
            # it would silently discard the summarizer's work.
            return self._truncate_keep_summary(result, budget, ctx,
                                               summary_turn)
        return result

    @staticmethod
    def _truncate_keep_summary(
        messages: list[Message], budget: int, ctx: dict[str, Any],
        summary_turn: Message,
    ) -> list[Message]:
        """Truncate oldest, keeping every system turn AND the summary."""
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        sacred_tail = rest[-1:] if rest else []
        middle = [m for m in rest[:-1] if m is not summary_turn]
        kept: list[Message] = list(system) + [summary_turn]
        tail: list[Message] = []
        for m in reversed(middle):
            candidate = kept + [m] + tail + sacred_tail
            if estimate_messages(candidate) <= budget:
                tail.insert(0, m)
        result = kept + tail + sacred_tail
        dropped = len(messages) - len(result)
        if dropped:
            ctx["dropped_turns"] = ctx.get("dropped_turns", 0) + dropped
        return result


class CaseFacts(ContextStrategy):
    """Pin facts that must survive fitting verbatim.

    Progressive summarization silently strips exactly the details production
    systems need kept exact — numbers, dates, ids, names (Anthropic's
    context-management guidance calls this the case-facts block).  Facts
    arrive via ``ctx["pinned_facts"]`` (a list of strings); they are
    rendered as one verbatim system-adjacent turn right after the system
    prompt, immune to every other strategy in the chain.  Never drops
    content on its own — run it first.
    """

    name = "case_facts"

    def apply(self, messages: list[Message], budget: int,
              ctx: dict[str, Any]) -> list[Message]:
        facts = [str(f).strip() for f in (ctx.get("pinned_facts") or [])
                 if str(f).strip()]
        if not facts:
            return messages
        # Don't double-pin when the block is already present.
        if any(m.role == "system" and m.content.startswith(_FACTS_HEADER)
               for m in messages):
            ctx["facts_kept"] = len(facts)
            return messages
        block = Message(
            role="system",
            content=_FACTS_HEADER + "\n" + "\n".join(f"• {f}" for f in facts),
        )
        ctx["facts_kept"] = len(facts)
        system_idx = next(
            (i for i, m in enumerate(messages) if m.role == "system"), None)
        if system_idx is None:
            return [block] + messages
        return messages[: system_idx + 1] + [block] + messages[system_idx + 1:]


class PrioritizeEnds(ContextStrategy):
    """Lost-in-the-middle reorder: long reference up top, request at bottom.

    Models attend to the start and end of a long context far more than the
    interior; Anthropic's own testing shows placing long reference material
    at the top and the question/instructions at the bottom improves response
    quality by up to 30% on complex multi-document tasks at zero token cost.
    This strategy moves the longest non-system message (the reference) to
    just after the system prompt and keeps the last user message (the
    request) at the very bottom.  Content-preserving — it only reorders.
    """

    name = "prioritize_ends"

    def apply(self, messages: list[Message], budget: int,
              ctx: dict[str, Any]) -> list[Message]:
        if len(messages) < 4:
            return messages
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        if len(rest) < 3:
            return messages
        # The request: last user message stays last.
        request_idx = next(
            (i for i in range(len(rest) - 1, -1, -1)
             if rest[i].role == "user"), None)
        # The reference: longest non-request message goes to the top.
        pool = [m for j, m in enumerate(rest) if j != request_idx]
        if not pool:
            return messages
        reference = max(pool, key=lambda m: len(m.content))
        if len(reference.content) < 500:
            return messages  # nothing long enough to matter
        middle = [m for m in pool if m is not reference]
        request = [rest[request_idx]] if request_idx is not None else []
        return system + [reference] + middle + request


def trim_tool_results(messages: Sequence[Message],
                      max_chars: int = 2000) -> list[Message]:
    """Cap verbose tool outputs before they enter context.

    Tool results are the noisiest context filler: a directory listing or a
    stack trace that was never going to be read in full.  Truncation keeps
    head and tail (the error line usually lives at the end) and marks the
    cut honestly.  Content-safe: pure function, never raises.
    """
    out: list[Message] = []
    for m in messages:
        if m.role == "tool" and len(m.content) > max_chars:
            head = max_chars * 2 // 3
            tail = max_chars - head
            cut = (m.content[:head] + "\n…[tool output trimmed]…\n"
                   + m.content[-tail:])
            out.append(Message(role=m.role, content=cut, name=m.name,
                               tool_call_id=m.tool_call_id,
                               images=list(m.images)))
        else:
            out.append(m)
    return out


def with_cache_breakpoints(
    messages: Sequence[Message], *, window: int = 3,
) -> list[dict[str, Any]]:
    """Annotate provider-agnostic prompt-cache breakpoints (X19 pattern).

    Breakpoint 1: the system prompt (stable across turns — never mutate it
    mid-conversation).  Breakpoints 2..N: the last ``window`` non-system
    messages (the rolling window that re-establishes caching within a turn
    or two after compression).  Returns plain dicts with a
    ``cache_breakpoint: True`` marker; provider adapters translate the
    marker into their native form (e.g. Anthropic ``cache_control``).
    """
    msgs = list(messages)
    if not msgs:
        return []
    marked: set[int] = set()
    sys_idx = next((i for i, m in enumerate(msgs) if m.role == "system"),
                   None)
    if sys_idx is not None:
        marked.add(sys_idx)
    non_system = [i for i, m in enumerate(msgs) if m.role != "system"]
    marked.update(non_system[-max(1, window):])
    out: list[dict[str, Any]] = []
    for i, m in enumerate(msgs):
        row: dict[str, Any] = {
            "role": m.role, "content": m.content,
            "cache_breakpoint": i in marked,
        }
        if m.name:
            row["name"] = m.name
        out.append(row)
    return out


_FACTS_HEADER = "[facts — keep verbatim, never summarize]"


_STRATEGIES: dict[str, ContextStrategy] = {
    "compact": Compact(),
    "case_facts": CaseFacts(),
    "prioritize_ends": PrioritizeEnds(),
    "truncate_oldest": TruncateOldest(),
    "summarize_middle": SummarizeMiddle(),
}


def strategy_for(name: str) -> ContextStrategy | None:
    """Look up a strategy by name (None when unknown)."""
    return _STRATEGIES.get((name or "").lower())


def fit_messages(
    messages: Sequence[Message],
    max_tokens: int,
    task_kind: str = "",
    *,
    summarizer: Callable[[str], str] | None = None,
    chain: Sequence[str] | None = None,
    floor: int = 256,
    pinned_facts: Sequence[str] | None = None,
) -> FitResult:
    """Shrink ``messages`` to ``max_tokens`` via the task-kind's strategy chain.

    ``floor`` is the smallest budget ever attempted — fitting below a
    couple hundred tokens rarely leaves anything usable.  Callers doing
    aggressive post-overflow recovery may pass a smaller floor.
    ``pinned_facts`` threads through ``ctx`` to :class:`CaseFacts`: facts
    that survive every strategy verbatim.  Never raises: on any internal
    failure the original messages come back with ``overflow`` set honestly.
    """
    msgs = list(messages)
    budget = max(floor, int(max_tokens * _BUDGET_FRACTION))
    names = tuple(chain) if chain is not None else \
        TASK_FIT_CHAINS.get((task_kind or "").lower(), _DEFAULT_CHAIN)
    ctx: dict[str, Any] = {
        "summarizer": summarizer,
        "pinned_facts": list(pinned_facts or []),
    }
    applied: list[str] = []
    try:
        # Pin facts even when the budget already fits — they are cheap and
        # must be present on every turn, not just when we are shrinking.
        case_facts = strategy_for("case_facts")
        if case_facts is not None and ctx["pinned_facts"]:
            msgs = case_facts.apply(msgs, budget, ctx)
            if ctx.get("facts_kept"):
                applied.append(case_facts.name)
        if estimate_messages(msgs) <= budget:
            return FitResult(messages=msgs, tokens=estimate_messages(msgs),
                             applied=applied,
                             facts_kept=int(ctx.get("facts_kept", 0)))
        for name in names:
            strategy = strategy_for(name)
            if strategy is None:
                continue
            before = len(msgs)
            msgs = strategy.apply(msgs, budget, ctx)
            if len(msgs) != before or estimate_messages(msgs) < estimate_messages(messages):
                applied.append(strategy.name)
            if estimate_messages(msgs) <= budget:
                break
        tokens = estimate_messages(msgs)
        return FitResult(
            messages=msgs, tokens=tokens, applied=applied,
            dropped_turns=int(ctx.get("dropped_turns", 0)),
            summarized=bool(ctx.get("summarized", False)),
            overflow=tokens > budget,
            facts_kept=int(ctx.get("facts_kept", 0)),
        )
    except Exception as exc:  # noqa: BLE001 — fitting must never break the call
        _log.debug("context_fit failed (%s); returning original", exc)
        return FitResult(messages=list(messages),
                         tokens=estimate_messages(messages),
                         overflow=True)


def fit_prompt(prompt: str, max_tokens: int, task_kind: str = "") -> str:
    """Single-string equivalent: head-truncate to the budget.

    Keeps the *tail* of the prompt (the actual request usually lives at
    the end of an assembled prompt) — the opposite of naive [:n].
    Never raises.
    """
    try:
        from .base import estimate_tokens

        budget = max(256, int(max_tokens * _BUDGET_FRACTION))
        if estimate_tokens(prompt) <= budget:
            return prompt
        # binary-search the longest fitting tail
        text = prompt
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if estimate_tokens(text[mid:]) <= budget:
                hi = mid
            else:
                lo = mid + 1
        cut = max(0, lo - 200)  # small overlap so we don't start mid-word
        return "…[truncated]…\n" + text[cut:].lstrip()
    except Exception:  # noqa: BLE001
        return prompt[: max(256, int(max_tokens * _BUDGET_FRACTION)) * 4]
