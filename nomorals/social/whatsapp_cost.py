"""WhatsApp cost-awareness layer — cost transparency as a feature.

Every WhatsApp-facing automation must be cost-aware: message budgets,
batching, and the "is this ₦14 message worth it?" calculation built into
the agent.

Rates (Meta Nigeria billing, effective Oct 1 2026):
    - service conversation:   ~₦14 per message
    - marketing conversation: ~₦84 per message

These are module constants because Meta reprices periodically — update
RATE_SERVICE_KOBO / RATE_MARKETING_KOBO and the accounting follows.

What this gives you:
- ``CostTracker`` — per-message accounting, per-client weekly budgets
  (hard blocks + soft warnings), weekly spend reports.
- ``CostAwareSender`` — wraps ANY ``sender(phone, text) -> bool`` callable
  (this is #66's sender seam: ``CartRecovery(sender=CostAwareSender(...))``).
  Blocks when a hard budget is exceeded, batches queued updates, and
  estimates campaign cost before the owner approves.
- ``worth_it()`` — the agent's "is this message worth it?" calculation.

The transparency pitch (build-map #68): "Devon spent ₦1,240 on WhatsApp
this week and recovered ₦47,000." Nobody else shows you the bill next to
the revenue.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "RATE_SERVICE_KOBO",
    "RATE_MARKETING_KOBO",
    "CATEGORIES",
    "CostTracker",
    "CostAwareSender",
    "CampaignEstimate",
    "worth_it",
    "naira",
]

# ── rates (Meta Nigeria, Oct 1 2026 billing) ──────────────────────────────
# VERIFY PERIODICALLY: Meta reprices WhatsApp conversations; when they do,
# update these two constants and every report, estimate, and budget check
# follows automatically. Values are in kobo (₦14 = 1_400 kobo).
RATE_SERVICE_KOBO = 1_400      # ~₦14 per service-category message
RATE_MARKETING_KOBO = 8_400    # ~₦84 per marketing-category message

CATEGORIES = ("service", "marketing")

#: Warn the owner when a budget is this fraction consumed (soft warning).
BUDGET_WARN_FRACTION = 0.8


def naira(kobo: int) -> str:
    """₦1,240 formatting."""
    return f"₦{kobo / 100:,.0f}"


def _rate_for(category: str) -> int:
    cat = (category or "service").strip().lower()
    if cat not in CATEGORIES:
        raise ValueError(f"unknown WhatsApp category {category!r} — {CATEGORIES}")
    return RATE_SERVICE_KOBO if cat == "service" else RATE_MARKETING_KOBO


# ── CostTracker ───────────────────────────────────────────────────────────


class CostTracker:
    """Per-message WhatsApp accounting.

    SQLite-backed, never raises. Budgets are weekly (rolling 7 days):
    ``set_budget(client, amount_kobo)`` then ``check_budget(client)`` tells
    you whether sends may continue.

    A budget is *hard* by default — exceeding it blocks sends. Pass
    ``hard=False`` for a soft budget that warns instead of blocking.
    """

    def __init__(
        self,
        db_path: str = "",
        *,
        now: Callable[[], float] | None = None,
    ) -> None:
        import os
        path = db_path or os.path.expanduser("~/.nomorals/whatsapp_cost.db")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
        self.db_path = path
        self._now = now or time.time
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS wa_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'service',
                    cost_kobo INTEGER NOT NULL DEFAULT 0,
                    client TEXT NOT NULL DEFAULT 'default',
                    purpose TEXT NOT NULL DEFAULT '',
                    ts REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wa_budgets (
                    client TEXT PRIMARY KEY,
                    amount_kobo INTEGER NOT NULL,
                    hard INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                );
            """)
            self._db.commit()
        except Exception:  # noqa: BLE001 — accounting must never break sends
            _log.debug("cost schema init failed", exc_info=True)

    # ── accounting ──

    def track(
        self,
        phone: str,
        category: str,
        *,
        client: str = "default",
        purpose: str = "",
    ) -> int:
        """Record one message. Returns the cost in kobo. Never raises."""
        try:
            cost = _rate_for(category)
            self._db.execute(
                """INSERT INTO wa_costs
                   (phone, category, cost_kobo, client, purpose, ts)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (phone or "", (category or "service").lower(), cost,
                 client or "default", purpose or "", self._now()),
            )
            self._db.commit()
            return cost
        except ValueError:
            raise  # unknown category is a caller bug — surface it
        except Exception:  # noqa: BLE001
            _log.debug("cost track failed", exc_info=True)
            return 0

    def spent_since(
        self, since: float, *, client: str = "default"
    ) -> int:
        """Kobo spent since a timestamp. Never raises."""
        try:
            row = self._db.execute(
                """SELECT COALESCE(SUM(cost_kobo), 0) AS t FROM wa_costs
                   WHERE ts >= ? AND client = ?""",
                (since, client or "default"),
            ).fetchone()
            return int(row["t"] or 0)
        except Exception:  # noqa: BLE001
            return 0

    def spent_week(self, *, client: str = "default") -> int:
        """Kobo spent in the rolling 7 days. Never raises."""
        return self.spent_since(self._now() - 7 * 86400, client=client)

    def message_count(self, *, client: str = "default") -> int:
        """Messages sent in the rolling 7 days. Never raises."""
        try:
            row = self._db.execute(
                """SELECT COUNT(*) AS n FROM wa_costs
                   WHERE ts >= ? AND client = ?""",
                (self._now() - 7 * 86400, client or "default"),
            ).fetchone()
            return int(row["n"] or 0)
        except Exception:  # noqa: BLE001
            return 0

    # ── budgets ──

    def set_budget(
        self, client: str, amount_kobo: int, *, hard: bool = True
    ) -> None:
        """Set a weekly budget for a client. Never raises."""
        try:
            self._db.execute(
                """INSERT OR REPLACE INTO wa_budgets
                   (client, amount_kobo, hard, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (client or "default", int(amount_kobo or 0),
                 1 if hard else 0, self._now()),
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("set_budget failed", exc_info=True)

    def get_budget(self, client: str = "default") -> dict[str, Any] | None:
        """The budget row, or None when no budget is set. Never raises."""
        try:
            row = self._db.execute(
                "SELECT * FROM wa_budgets WHERE client = ?",
                (client or "default",),
            ).fetchone()
            if row is None:
                return None
            return {
                "client": row["client"],
                "amount_kobo": int(row["amount_kobo"] or 0),
                "hard": bool(row["hard"]),
            }
        except Exception:  # noqa: BLE001
            return None

    def check_budget(
        self, client: str = "default"
    ) -> tuple[bool, int, int]:
        """(may_send, spent_kobo, budget_kobo).

        No budget set → (True, spent, 0): unlimited.
        Hard budget exceeded → (False, spent, budget): BLOCK.
        Soft budget exceeded → (True, spent, budget): warn, don't block.
        Never raises.
        """
        try:
            budget = self.get_budget(client)
            spent = self.spent_week(client=client)
            if budget is None:
                return True, spent, 0
            amount = int(budget["amount_kobo"] or 0)
            if spent >= amount:
                return (False if budget["hard"] else True), spent, amount
            return True, spent, amount
        except Exception:  # noqa: BLE001
            return True, 0, 0  # fail open on accounting errors, never block

    def budget_warning(self, client: str = "default") -> str:
        """Soft warning text when a budget is nearly consumed. '' when fine."""
        try:
            budget = self.get_budget(client)
            if budget is None:
                return ""
            amount = int(budget["amount_kobo"] or 0)
            if amount <= 0:
                return ""
            spent = self.spent_week(client=client)
            if spent >= amount * BUDGET_WARN_FRACTION:
                return (
                    f"⚠️ WhatsApp budget {naira(spent)} of {naira(amount)} "
                    f"used for '{client}' this week."
                )
            return ""
        except Exception:  # noqa: BLE001
            return ""

    # ── reports ──

    def weekly_spend(self, *, client: str = "default") -> str:
        """'Devon spent ₦1,240 on WhatsApp this week.' Never raises."""
        spent = self.spent_week(client=client)
        n = self.message_count(client=client)
        return (
            f"💸 Devon spent {naira(spent)} on WhatsApp this week "
            f"({n} message{'s' if n != 1 else ''})."
        )

    def spend_vs_revenue(
        self, recovered_kobo: int, *, client: str = "default"
    ) -> str:
        """Cost transparency next to revenue: the #68 pitch.

        'Devon spent ₦1,240 on WhatsApp this week and recovered ₦47,000.'
        Pass #66's ``recovered_kobo_7d`` from ``CartRecovery.stats()``.
        Never raises.
        """
        spent = self.spent_week(client=client)
        roi = (recovered_kobo / spent) if spent > 0 else 0.0
        line = (
            f"Devon spent {naira(spent)} on WhatsApp this week "
            f"and recovered {naira(int(recovered_kobo or 0))}."
        )
        if spent > 0 and recovered_kobo > 0:
            line += f" That's a {roi:.0f}× return on messaging spend."
        return line


# ── campaign estimates ──────────────────────────────────────────────────


@dataclass
class CampaignEstimate:
    count: int
    category: str
    per_message_kobo: int
    total_kobo: int
    within_budget: bool = True
    budget_kobo: int = 0
    client: str = "default"

    @property
    def text(self) -> str:
        return (
            f"this campaign will cost {naira(self.total_kobo)} "
            f"for {self.count} customer{'s' if self.count != 1 else ''} "
            f"({self.category}, {naira(self.per_message_kobo)} each)"
        )


# ── CostAwareSender ─────────────────────────────────────────────────────


class CostAwareSender:
    """Wraps any ``sender(phone, text) -> bool`` with cost-awareness.

    This is the seam #66 was built for::

        tracker = CostTracker()
        aware = CostAwareSender(default_whatsapp_sender, tracker=tracker,
                                client="suyaspot", category="marketing")
        recovery = CartRecovery(db_path=..., sender=aware)

    - Blocks sends when a HARD budget is exceeded (returns False, logs why).
    - Tracks the cost of every successful send.
    - ``queue()`` / ``flush()`` batch multiple updates to the same
      recipient into one message.
    - ``estimate_campaign()`` prices a broadcast before the owner approves.
    """

    def __init__(
        self,
        sender: Callable[[str, str], bool],
        *,
        tracker: CostTracker | None = None,
        db_path: str = "",
        client: str = "default",
        category: str = "marketing",
        purpose: str = "",
    ) -> None:
        self.sender = sender
        self.tracker = tracker or CostTracker(db_path)
        self.client = client or "default"
        self.category = (category or "marketing").lower()
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r} — {CATEGORIES}")
        self.purpose = purpose or ""
        self._batch: dict[str, list[str]] = {}

    def __call__(self, phone: str, text: str) -> bool:
        """Send one message, cost-aware. Never raises."""
        try:
            may_send, spent, budget = self.tracker.check_budget(self.client)
            if not may_send:
                _log.warning(
                    "WhatsApp send BLOCKED for '%s': budget %s spent of %s",
                    self.client, naira(spent), naira(budget),
                )
                return False
            ok = bool(self.sender(phone, text))
            if ok:
                self.tracker.track(
                    phone, self.category,
                    client=self.client, purpose=self.purpose,
                )
            return ok
        except Exception:  # noqa: BLE001 — the seam must never raise
            _log.debug("cost-aware send failed", exc_info=True)
            return False

    # ── campaigns ──

    def estimate_campaign(
        self, count: int, category: str | None = None
    ) -> CampaignEstimate:
        """Price a broadcast before the owner approves it.

        Returns a CampaignEstimate whose ``.text`` reads like:
        'this campaign will cost ₦8,400 for 100 customers'.
        """
        cat = (category or self.category).lower()
        per = _rate_for(cat)
        total = int(count or 0) * per
        may_send, spent, budget = self.tracker.check_budget(self.client)
        within = budget == 0 or (spent + total) < budget
        return CampaignEstimate(
            count=int(count or 0), category=cat,
            per_message_kobo=per, total_kobo=total,
            within_budget=within, budget_kobo=budget,
            client=self.client,
        )

    def request_campaign_approval(
        self,
        estimate: CampaignEstimate,
        approval_fn: Callable[[CampaignEstimate], bool] | None = None,
    ) -> bool:
        """Owner gate: returns True only when the owner approves.

        ``approval_fn`` receives the estimate and returns True/False.
        No approval_fn → returns False (fail closed: never send a priced
        campaign without an explicit yes).
        """
        try:
            if approval_fn is None:
                _log.info("campaign needs owner approval: %s", estimate.text)
                return False
            return bool(approval_fn(estimate))
        except Exception:  # noqa: BLE001
            _log.debug("campaign approval failed", exc_info=True)
            return False

    # ── batching ──

    def queue(self, phone: str, text: str) -> None:
        """Stage an update; ``flush()`` combines per-recipient updates."""
        if phone and text:
            self._batch.setdefault(phone, []).append(text)

    def flush(self) -> int:
        """Send all queued updates, one combined message per recipient.

        Returns the number of messages actually sent. Never raises.
        """
        sent = 0
        try:
            for phone, texts in list(self._batch.items()):
                combined = "\n\n".join(texts)
                if self(phone, combined):
                    sent += 1
            return sent
        except Exception:  # noqa: BLE001
            _log.debug("batch flush failed", exc_info=True)
            return sent
        finally:
            self._batch.clear()

    @property
    def queued(self) -> int:
        return sum(len(v) for v in self._batch.values())


# ── agent integration ───────────────────────────────────────────────────


def worth_it(
    expected_value_kobo: int,
    cost_kobo: int,
    *,
    min_roi: float = 2.0,
) -> bool:
    """The "is this ₦14 message worth it?" calculation.

    ``expected_value_kobo`` is the expected return — e.g. for cart
    recovery, ``cart_total_kobo * recovery_probability`` (≈0.30 for the
    25–35% WhatsApp recovery rate). Sends when the expected value clears
    ``min_roi`` × the cost (default: expected ≥ 2× cost).

    Example: a ₦84 marketing message chasing a ₦47,000 cart at 30%
    recovery → expected ₦14,100 — obviously worth it.
    """
    try:
        expected = int(expected_value_kobo or 0)
        cost = int(cost_kobo or 0)
        if cost <= 0:
            return expected > 0
        return expected >= cost * float(min_roi or 0)
    except Exception:  # noqa: BLE001
        return False
