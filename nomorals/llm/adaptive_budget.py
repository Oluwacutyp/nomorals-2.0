"""Cost-adaptive token budgeting.

Distilled spec — Research-loop proposal #3 Adaptive Agents, cost/token
model §4.4 of arXiv 2508.11126v1 ("Token Economics for LLM Agents"):
§4.4 optimises on the *intensive margin*: each message is made more
informative per token via (a) format redesign / compact protocols,
(b) cost-aware model cascading, (c) runtime supervision.  Distilled
into Devon behaviour:

1. TRACK per operation: prompt + completion tokens and USD cost, per
   operation and in total, against configured daily + per-operation
   budgets (tokens AND cost).
2. ADAPT as utilisation rises (thresholds from config):
   - >= downgrade (default 50%): cascade to the cheapest tier —
     route ``sort=price`` + ``max_price`` cap.
   - >= truncate (default 80%): shrink context to a capped budget
     (compact + truncate-oldest; system prompt + latest turn sacred)
     and cap the completion length.  Critical work truncates only
     at >= 95%.
   - >= defer (default 80%): background / non-urgent work is deferred,
     never executed — with an honest "deferred" verdict, not a drop.
3. BLOCK at >= 100%: honest "budget exceeded" naming the budget and
   the spend.  No call is made.
4. HONEST, NEVER SILENT: every adaptation is carried on the verdict
   (``Adaptation.note``) and, through the router, on the response
   (``adaptation_note`` + ``degraded=True`` / ``deferred=True``).
   Unlimited (no budgets configured) == pure allow.

Verdicts use *spent* utilisation only — a call is never blocked for
its own estimated size; an oversized call gets truncated instead, and
only spend already past the cap blocks.  Day boundaries are UTC; the
ledger rolls over automatically.  All methods never raise on bad
input: a broken budget must never break routing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.logging_setup import get_logger
from .base import Message, estimate_messages

_log = get_logger(__name__)

__all__ = [
    "Adaptation",
    "AdaptiveBudget",
    "AdaptiveBudgetConfig",
    "URGENCIES",
]

#: Accepted urgency levels.  ``critical`` is never deferred and
#: truncates only at >= 95% utilisation; ``background`` is deferred
#: once the defer threshold is reached; ``normal`` is served degraded.
URGENCIES = ("critical", "normal", "background")

#: Operations that count as non-urgent unless the caller says otherwise.
DEFAULT_BACKGROUND_OPERATIONS: tuple[str, ...] = (
    "research", "background", "summarize", "judge", "index",
)

#: When truncation fires, the context budget is halved (never below the
#: min-context floor) and the completion cap is halved (never below 64).
_TRUNCATE_FRACTION = 0.5
_MIN_COMPLETION_TOKENS = 64


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


@dataclass
class AdaptiveBudgetConfig:
    """Budget + adaptation thresholds.  0 / empty == unlimited."""

    daily_token_budget: int = 0
    daily_cost_usd: float = 0.0
    #: Per-operation budgets: {"research": {"tokens": 5000, "cost_usd": 0.5}}.
    #: A per-operation cap blocks/degrades only that operation.
    per_operation: dict[str, dict[str, float]] = field(default_factory=dict)
    warn_threshold: float = 0.5
    downgrade_threshold: float = 0.5
    truncate_threshold: float = 0.8
    defer_threshold: float = 0.8
    #: Critical work truncates only at this utilisation.
    critical_truncate_threshold: float = 0.95
    #: Context window assumed when no broker card says otherwise.
    full_context_tokens: int = 8192
    #: Context budget never shrinks below this (tokens).
    min_context_tokens: int = 1024
    #: Cheapest-tier price cap applied when downgrading (USD / 1M input).
    cheap_price_cap_per_1m: float = 1.0
    background_operations: tuple[str, ...] = DEFAULT_BACKGROUND_OPERATIONS

    def __post_init__(self) -> None:
        self.daily_token_budget = max(0, int(self.daily_token_budget or 0))
        self.daily_cost_usd = max(0.0, _as_float(self.daily_cost_usd))
        for name in ("warn_threshold", "downgrade_threshold",
                     "truncate_threshold", "defer_threshold",
                     "critical_truncate_threshold"):
            val = _as_float(getattr(self, name))
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {val}")
            setattr(self, name, val)
        if not (self.warn_threshold <= self.downgrade_threshold
                <= self.truncate_threshold):
            raise ValueError(
                "require warn <= downgrade <= truncate thresholds, got "
                f"{self.warn_threshold}/{self.downgrade_threshold}/"
                f"{self.truncate_threshold}")
        self.full_context_tokens = max(1, int(self.full_context_tokens or 8192))
        self.min_context_tokens = max(256, int(self.min_context_tokens or 1024))
        self.cheap_price_cap_per_1m = max(0.0, _as_float(self.cheap_price_cap_per_1m))
        self.background_operations = tuple(
            str(o).lower() for o in (self.background_operations or ()))
        norm: dict[str, dict[str, float]] = {}
        for op, spec in (self.per_operation or {}).items():
            if isinstance(spec, Mapping):
                norm[str(op)] = {
                    "tokens": max(0.0, _as_float(spec.get("tokens"))),
                    "cost_usd": max(0.0, _as_float(spec.get("cost_usd"))),
                }
        self.per_operation = norm

    @property
    def has_budget(self) -> bool:
        """True when any daily or per-operation budget is configured."""
        if self.daily_token_budget > 0 or self.daily_cost_usd > 0:
            return True
        return any(v.get("tokens", 0) > 0 or v.get("cost_usd", 0) > 0
                   for v in self.per_operation.values())

    @classmethod
    def from_settings(cls, settings: Any) -> "AdaptiveBudgetConfig":
        """Build from a root :class:`Settings` (or anything duck-typed).

        Prefers ``settings.adaptive_budget`` when present; falls back
        to the legacy ``settings.budget`` (tokens / cost_usd) as the
        daily budget with default thresholds.
        """
        try:
            section = getattr(settings, "adaptive_budget", None)
            if section is not None:
                return cls(
                    daily_token_budget=int(getattr(
                        section, "daily_token_budget", 0) or 0),
                    daily_cost_usd=_as_float(
                        getattr(section, "daily_cost_usd", 0.0)),
                    per_operation=dict(
                        getattr(section, "per_operation", {}) or {}),
                    warn_threshold=_as_float(
                        getattr(section, "warn_threshold", 0.5), 0.5),
                    downgrade_threshold=_as_float(
                        getattr(section, "downgrade_threshold", 0.5), 0.5),
                    truncate_threshold=_as_float(
                        getattr(section, "truncate_threshold", 0.8), 0.8),
                    defer_threshold=_as_float(
                        getattr(section, "defer_threshold", 0.8), 0.8),
                    critical_truncate_threshold=_as_float(
                        getattr(section, "critical_truncate_threshold", 0.95),
                        0.95),
                    full_context_tokens=int(getattr(
                        section, "full_context_tokens", 8192) or 8192),
                    min_context_tokens=int(getattr(
                        section, "min_context_tokens", 1024) or 1024),
                    cheap_price_cap_per_1m=_as_float(
                        getattr(section, "cheap_price_cap_per_1m", 1.0), 1.0),
                    background_operations=tuple(getattr(
                        section, "background_operations",
                        DEFAULT_BACKGROUND_OPERATIONS) or ()),
                )
            legacy = getattr(settings, "budget", None)
            if legacy is not None:
                return cls(
                    daily_token_budget=int(getattr(legacy, "tokens", 0) or 0),
                    daily_cost_usd=_as_float(getattr(legacy, "cost_usd", 0.0)),
                )
        except Exception:  # noqa: BLE001 — config must never break routing
            _log.debug("adaptive budget from_settings failed", exc_info=True)
        return cls()


@dataclass
class Adaptation:
    """One verdict from :meth:`AdaptiveBudget.decide` (pure data)."""

    action: str = "allow"  # allow | downgrade | truncate | defer | block
    downgrade: bool = False
    truncate: bool = False
    deferred: bool = False
    blocked: bool = False
    #: True when the call still runs but degraded (downgrade / truncate).
    degraded: bool = False
    #: Context budget (tokens) when truncating; None otherwise.
    context_budget_tokens: int | None = None
    #: Fraction of the full context budget to keep when truncating.
    context_fraction: float = 1.0
    #: Cheapest-tier price cap (USD / 1M input) when downgrading.
    max_price_per_1m: float | None = None
    #: Fraction of the requested completion cap to keep when truncating.
    completion_fraction: float = 1.0
    #: Honest human-readable reason — surfaced on the response, never dropped.
    note: str = ""
    utilisation: float = 0.0
    spent_tokens: float = 0.0
    spent_cost_usd: float = 0.0

    def apply_to_route(self, route: Mapping[str, Any] | None) -> dict[str, Any]:
        """Route dict with the downgrade applied — caller's explicit
        ``sort`` / ``max_price`` always win over the adaptation."""
        out = dict(route or {})
        if self.downgrade:
            out.setdefault("sort", "price")
            if self.max_price_per_1m is not None:
                out.setdefault("max_price", self.max_price_per_1m)
        return out

    def apply_to_messages(self, messages: list[Message],
                          *, task_kind: str = "") -> list[Message]:
        """Truncate ``messages`` to the adaptation's context budget.

        No-op unless this verdict truncates.  Uses the existing
        ``context_fit`` chain (compact + truncate-oldest): the system
        prompt and the latest turn are sacred and always survive.
        """
        if not self.truncate or self.context_budget_tokens is None:
            return list(messages)
        try:
            from .context_fit import fit_messages

            result = fit_messages(messages, self.context_budget_tokens,
                                  task_kind=task_kind)
            if result.overflow:
                _log.warning(
                    "adaptive budget: context still over budget after fit "
                    "(%d tokens vs %d cap)", result.tokens,
                    self.context_budget_tokens)
            return result.messages
        except Exception:  # noqa: BLE001 — fitting never breaks the call
            _log.debug("adaptive budget fit_messages failed", exc_info=True)
            return list(messages)

    def cap_completion(self, max_tokens: int) -> int:
        """Completion cap after the adaptation (floor 64 when truncating)."""
        if not self.truncate:
            return max(1, int(max_tokens))
        return max(_MIN_COMPLETION_TOKENS,
                   int(max(1, int(max_tokens)) * self.completion_fraction))


class AdaptiveBudget:
    """Per-operation token/cost ledger + cost-adaptive verdicts.

    The ledger is in-memory per UTC day (the router's ``CostLog`` is
    the durable audit trail; this is the live gate).  :meth:`decide`
    is pure — it never mutates — so it is safe to consult twice for
    the same call (pre-call gate + dispatch safety net).
    """

    def __init__(self, config: AdaptiveBudgetConfig | None = None, *,
                 clock: Callable[[], float] | None = None) -> None:
        self.config = config or AdaptiveBudgetConfig()
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._day = self._today()
        self._totals: dict[str, float] = {"tokens": 0.0, "cost_usd": 0.0,
                                           "calls": 0.0}
        self._per_op: dict[str, dict[str, float]] = {}

    # ── day rollover ───────────────────────────────────────────────────────
    def _today(self) -> str:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            self._clock(), tz=_dt.timezone.utc).strftime("%Y-%m-%d")

    def _rollover(self) -> None:
        today = self._today()
        if today != self._day:
            _log.info("adaptive budget: day rollover %s -> %s (spend reset)",
                      self._day, today)
            self._day = today
            self._totals = {"tokens": 0.0, "cost_usd": 0.0, "calls": 0.0}
            self._per_op = {}

    # ── ledger ─────────────────────────────────────────────────────────────
    def record(self, operation: str, *, prompt_tokens: int = 0,
               completion_tokens: int = 0, cost_usd: float = 0.0,
               **_ignored: Any) -> dict[str, Any]:
        """Record one completed call's actuals.  Never raises."""
        try:
            with self._lock:
                self._rollover()
                op = str(operation or "chat")
                tokens = max(0, int(prompt_tokens or 0)) \
                    + max(0, int(completion_tokens or 0))
                cost = max(0.0, _as_float(cost_usd))
                slot = self._per_op.setdefault(
                    op, {"tokens": 0.0, "cost_usd": 0.0, "calls": 0.0})
                slot["tokens"] += tokens
                slot["cost_usd"] += cost
                slot["calls"] += 1
                self._totals["tokens"] += tokens
                self._totals["cost_usd"] += cost
                self._totals["calls"] += 1
                return {"operation": op, "day": self._day,
                        **dict(slot), "totals": dict(self._totals)}
        except Exception:  # noqa: BLE001 — accounting never breaks routing
            _log.debug("adaptive budget record failed", exc_info=True)
            return {"operation": str(operation or "chat"), "day": self._day}

    def spent(self, operation: str = "") -> dict[str, Any]:
        """Spend so far today: per-operation, or totals when blank."""
        with self._lock:
            self._rollover()
            if operation:
                slot = self._per_op.get(str(operation), {})
                return {"operation": str(operation), "day": self._day,
                        "tokens": slot.get("tokens", 0.0),
                        "cost_usd": slot.get("cost_usd", 0.0),
                        "calls": slot.get("calls", 0.0)}
            return {"operation": "*", "day": self._day, **dict(self._totals)}

    def remaining(self, operation: str = "") -> dict[str, Any]:
        """Budget minus spend.  ``None`` where no budget is set."""
        spent = self.spent(operation)
        cfg = self.config
        if operation:
            spec = cfg.per_operation.get(str(operation), {})
            tok = spec.get("tokens") or None
            cost = spec.get("cost_usd") or None
            # Per-operation remaining is against the tighter of the
            # per-op budget and the daily budget.
            daily_tok = cfg.daily_token_budget or None
            daily_cost = cfg.daily_cost_usd or None
            if tok is not None and daily_tok is not None:
                tok = min(tok, daily_tok)
            elif tok is None:
                tok = daily_tok
            if cost is not None and daily_cost is not None:
                cost = min(cost, daily_cost)
            elif cost is None:
                cost = daily_cost
        else:
            tok = cfg.daily_token_budget or None
            cost = cfg.daily_cost_usd or None
        return {
            "operation": operation or "*",
            "tokens": None if tok is None else tok - spent["tokens"],
            "cost_usd": None if cost is None else cost - spent["cost_usd"],
        }

    # ── utilisation ────────────────────────────────────────────────────────
    def utilisation(self, operation: str = "") -> float:
        """0.0..inf: the binding utilisation (max over token/cost, daily
        and per-operation).  0.0 when nothing is budgeted."""
        cfg = self.config
        if not cfg.has_budget:
            return 0.0
        with self._lock:
            self._rollover()
            ratios: list[float] = []
            total = self._totals
            if cfg.daily_token_budget > 0:
                ratios.append(total["tokens"] / cfg.daily_token_budget)
            if cfg.daily_cost_usd > 0:
                ratios.append(total["cost_usd"] / cfg.daily_cost_usd)
            if operation:
                spec = cfg.per_operation.get(str(operation), {})
                slot = self._per_op.get(str(operation), {})
                if spec.get("tokens", 0) > 0:
                    ratios.append(slot.get("tokens", 0.0) / spec["tokens"])
                if spec.get("cost_usd", 0) > 0:
                    ratios.append(slot.get("cost_usd", 0.0) / spec["cost_usd"])
            return max(ratios) if ratios else 0.0

    def _effective_urgency(self, operation: str,
                           urgency: str | None) -> str:
        if urgency:
            u = str(urgency).lower()
            if u in URGENCIES:
                return u
            return "normal"
        if str(operation or "").lower() in self.config.background_operations:
            return "background"
        return "normal"

    # ── verdict ────────────────────────────────────────────────────────────
    def decide(self, operation: str, *, urgency: str | None = None,
               estimated_tokens: int = 0) -> Adaptation:
        """Verdict for one call.  Pure (no mutation).  Never raises.

        ``estimated_tokens`` is accepted for forward-compatibility
        (callers may pass the pre-call estimate); the verdict uses
        *spent* utilisation — a call is never blocked for its own
        size, it gets truncated instead.
        """
        try:
            with self._lock:
                self._rollover()
                cfg = self.config
                op = str(operation or "chat")
                util = self.utilisation(op)
                spent = self.spent()
                op_spent = self.spent(op)
                eff = self._effective_urgency(op, urgency)
                spend_txt = (
                    f"spent {spent['tokens']:.0f}"
                    f"{f'/{cfg.daily_token_budget}' if cfg.daily_token_budget else ''}"
                    f" tokens and ${spent['cost_usd']:.4f}"
                    f"{f'/${cfg.daily_cost_usd:.2f}' if cfg.daily_cost_usd else ''}"
                    f" (utilisation {util:.0%})"
                )
                if not cfg.has_budget or util < cfg.downgrade_threshold:
                    return Adaptation(action="allow", utilisation=util,
                                      spent_tokens=spent["tokens"],
                                      spent_cost_usd=spent["cost_usd"])
                if util >= 1.0:
                    return Adaptation(
                        action="block", blocked=True,
                        note=(f"budget exceeded: {spend_txt} — no further "
                              f"{op} calls until the budget resets"),
                        utilisation=util, spent_tokens=spent["tokens"],
                        spent_cost_usd=spent["cost_usd"])
                if eff == "background" and util >= cfg.defer_threshold:
                    return Adaptation(
                        action="defer", deferred=True,
                        note=(f"deferred: {spend_txt} — non-urgent {op} work "
                              f"deferred until the budget recovers"),
                        utilisation=util, spent_tokens=spent["tokens"],
                        spent_cost_usd=spent["cost_usd"])
                truncate_at = (cfg.critical_truncate_threshold
                               if eff == "critical" else cfg.truncate_threshold)
                if util >= truncate_at:
                    ctx_budget = max(cfg.min_context_tokens, int(
                        cfg.full_context_tokens * _TRUNCATE_FRACTION))
                    return Adaptation(
                        action="truncate", downgrade=True, truncate=True,
                        degraded=True,
                        context_budget_tokens=ctx_budget,
                        context_fraction=_TRUNCATE_FRACTION,
                        max_price_per_1m=cfg.cheap_price_cap_per_1m,
                        completion_fraction=_TRUNCATE_FRACTION,
                        note=(f"token budget {util:.0%} used ({spend_txt}): "
                              f"truncating context to {ctx_budget} tokens, "
                              f"capping completions, downgrading to cheapest tier"),
                        utilisation=util, spent_tokens=spent["tokens"],
                        spent_cost_usd=spent["cost_usd"])
                # downgrade band: warn <= util < truncate
                return Adaptation(
                    action="downgrade", downgrade=True, degraded=True,
                    max_price_per_1m=cfg.cheap_price_cap_per_1m,
                    note=(f"token budget {util:.0%} used ({spend_txt}): "
                          f"downgrading to cheapest tier"),
                    utilisation=util, spent_tokens=spent["tokens"],
                    spent_cost_usd=spent["cost_usd"])
        except Exception:  # noqa: BLE001 — verdict never breaks routing
            _log.debug("adaptive budget decide failed", exc_info=True)
            return Adaptation(action="allow")

    # ── reporting ──────────────────────────────────────────────────────────
    def status(self, operation: str = "chat") -> dict[str, Any]:
        """Snapshot for dashboards / stats: spend, remaining, verdict."""
        try:
            util = self.utilisation(operation)
            verdict = self.decide(operation)
            cfg = self.config
            alert = "ok"
            if util >= 1.0:
                alert = "exceeded"
            elif util >= cfg.truncate_threshold:
                alert = "warning"
            elif util >= cfg.warn_threshold:
                alert = "watch"
            return {
                "day": self._day,
                "spent": self.spent(),
                "spent_operation": self.spent(operation),
                "remaining": self.remaining(operation),
                "utilisation": round(util, 4),
                "action": verdict.action,
                "alert": alert,
                "thresholds": {
                    "warn": cfg.warn_threshold,
                    "downgrade": cfg.downgrade_threshold,
                    "truncate": cfg.truncate_threshold,
                    "defer": cfg.defer_threshold,
                },
                "budgets": {
                    "daily_token_budget": cfg.daily_token_budget,
                    "daily_cost_usd": cfg.daily_cost_usd,
                    "per_operation": {k: dict(v)
                                      for k, v in cfg.per_operation.items()},
                },
            }
        except Exception:  # noqa: BLE001
            return {"day": self._day, "utilisation": 0.0,
                    "action": "allow", "alert": "ok"}

    def reset(self) -> None:
        """Clear today's ledger (tests / operator reset)."""
        with self._lock:
            self._totals = {"tokens": 0.0, "cost_usd": 0.0, "calls": 0.0}
            self._per_op = {}
            self._day = self._today()
