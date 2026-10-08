"""Cart recovery via WhatsApp — abandoned-cart → 3-step sequence → recovered revenue.

The highest-ROI business automation in the commerce phase: abandoned carts
get a 3-message WhatsApp sequence (30min → 24h → 72h), and every recovery is
tracked so the business sees "Devon recovered ₦X for you."

Hard rules (test-enforced):
- Explicit opt-in REQUIRED. Only carts with opted_in=True enter the
  sequence. No opt-in → no messages, ever.
- Max 3 messages per cart. Recovery cancels the remaining steps.
- Never raises on send failure — a dead bridge must not break the pipeline.

Sender seam (for #68's cost-aware layer):
    Messages go through ``sender(phone, text) -> bool``. The default sender
    uses the WhatsApp bridge adapter. #68 will wrap this callable with
    cost-awareness — the seam is the contract, not the implementation.

Business-scoped: this is a business automation, not owner-personal chat and
never community. It lives in nomorals/commerce/, away from community scope.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "CartRecovery",
    "Cart",
    "RecoveryStep",
    "STEPS",
    "normalize_webhook",
    "default_whatsapp_sender",
]

#: The 3-step sequence: (step_number, delay_seconds, template_key).
STEPS: tuple[tuple[int, int, str], ...] = (
    (1, 30 * 60, "reminder"),      # 30 min — "forgot something?"
    (2, 24 * 3600, "nudge"),       # 24 h  — "still thinking?"
    (3, 72 * 3600, "last_call"),   # 72 h  — "last call"
)
MAX_MESSAGES = 3


def _naira(kobo: int) -> str:
    return f"₦{kobo / 100:,.0f}"


def _items_summary(items: list[dict[str, Any]]) -> str:
    names = [str(i.get("name") or i.get("title") or "item") for i in items]
    if len(names) <= 2:
        return " and ".join(names)
    return f"{names[0]}, {names[1]} (+{len(names) - 2} more)"


TEMPLATES: dict[str, str] = {
    "reminder": (
        "👋 Hi {name}, you left something in your cart at {store} — "
        "{items} ({total}).\nStill want it? {link}"
    ),
    "nudge": (
        "Still thinking about it, {name}? Your cart is waiting:\n"
        "{items} ({total}) at {store}.\n{link}"
    ),
    "last_call": (
        "⏰ Last call, {name} — your cart at {store} ({items}, {total}) "
        "expires soon.\nGrab it while it's there: {link}"
    ),
}


@dataclass
class Cart:
    cart_id: str
    phone: str
    items: list[dict[str, Any]] = field(default_factory=list)
    total_kobo: int = 0
    opted_in: bool = False
    status: str = "abandoned"  # abandoned | recovered | expired
    store: str = "unknown"
    customer_name: str = ""
    checkout_link: str = ""
    created_at: float = 0.0


@dataclass
class RecoveryStep:
    cart_id: str
    step: int
    run_at: float
    sent_at: float = 0.0
    status: str = "pending"  # pending | sent | cancelled


def default_whatsapp_sender(phone: str, text: str) -> bool:
    """Default sender: WhatsApp bridge adapter. Never raises.

    Lazily imports the adapter so this module loads without the bridge.
    Returns False when the bridge is down — the step stays failed, not lost.
    """
    try:
        from ..social.chat.whatsapp import WhatsAppAdapter
        from ..social.chat.base import ChatRef, ChatKind
        adapter = WhatsAppAdapter()
        result = adapter.send(
            ChatRef(platform="whatsapp", chat_id=phone, kind=ChatKind.DM),
            text,
        )
        return bool(result.ok)
    except Exception:  # noqa: BLE001 — sender seam must never raise
        _log.debug("whatsapp sender failed for %s", phone, exc_info=True)
        return False


class CartRecovery:
    """Abandoned-cart recovery engine.

    Args:
        db_path: SQLite file for carts/steps/recoveries.
        sender: ``sender(phone, text) -> bool``. Defaults to the WhatsApp
            bridge. #68's cost-aware layer wraps this callable.
        schedule_fn: ``schedule_fn(task_id, run_at, action, parameters)``.
            When None, steps are stored as pending and ``run_due_steps()``
            executes them (the host wires the real scheduler).
        now: injectable clock for tests.
    """

    def __init__(
        self,
        db_path: str,
        *,
        sender: Callable[[str, str], bool] | None = None,
        schedule_fn: Callable[..., Any] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.db_path = db_path
        self.sender = sender or default_whatsapp_sender
        self.schedule_fn = schedule_fn
        self._now = now or time.time
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS carts (
                cart_id TEXT PRIMARY KEY,
                phone TEXT NOT NULL,
                items_json TEXT NOT NULL DEFAULT '[]',
                total_kobo INTEGER NOT NULL DEFAULT 0,
                opted_in INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'abandoned',
                store TEXT NOT NULL DEFAULT 'unknown',
                customer_name TEXT NOT NULL DEFAULT '',
                checkout_link TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cart_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cart_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                run_at REAL NOT NULL,
                sent_at REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                UNIQUE(cart_id, step)
            );
            CREATE TABLE IF NOT EXISTS recoveries (
                cart_id TEXT PRIMARY KEY,
                recovered_kobo INTEGER NOT NULL,
                recovered_at REAL NOT NULL,
                store TEXT NOT NULL DEFAULT 'unknown'
            );
        """)
        self._db.commit()

    # ── intake ──────────────────────────────────────────────────────────

    def cart_abandoned(
        self,
        cart_id: str,
        phone: str,
        items: list[dict[str, Any]],
        total_kobo: int,
        opted_in: bool,
        *,
        store: str = "unknown",
        customer_name: str = "",
        checkout_link: str = "",
    ) -> Cart:
        """Register an abandoned cart. Starts the sequence ONLY when opted_in.

        Never raises. No opt-in → cart is recorded (for reporting) but no
        messages are ever scheduled.
        """
        try:
            now = self._now()
            cart = Cart(
                cart_id=cart_id, phone=phone, items=items,
                total_kobo=int(total_kobo or 0), opted_in=bool(opted_in),
                store=store, customer_name=customer_name,
                checkout_link=checkout_link, created_at=now,
            )
            self._db.execute(
                """INSERT OR REPLACE INTO carts
                   (cart_id, phone, items_json, total_kobo, opted_in, status,
                    store, customer_name, checkout_link, created_at)
                   VALUES (?, ?, ?, ?, ?, 'abandoned', ?, ?, ?, ?)""",
                (cart_id, phone, json.dumps(items), cart.total_kobo,
                 1 if cart.opted_in else 0, store, customer_name,
                 checkout_link, now),
            )
            if cart.opted_in:
                for step_no, delay, _template in STEPS:
                    run_at = now + delay
                    self._db.execute(
                        """INSERT OR REPLACE INTO cart_steps
                           (cart_id, step, run_at, status)
                           VALUES (?, ?, ?, 'pending')""",
                        (cart_id, step_no, run_at),
                    )
                    if self.schedule_fn is not None:
                        try:
                            self.schedule_fn(
                                f"cart-recovery-{cart_id}-step{step_no}",
                                run_at, "cart_recovery_step",
                                {"cart_id": cart_id, "step": step_no},
                            )
                        except Exception:  # noqa: BLE001 — scheduling is best-effort
                            _log.debug("schedule_fn failed", exc_info=True)
            self._db.commit()
            return cart
        except Exception:  # noqa: BLE001 — intake never raises
            _log.debug("cart_abandoned failed", exc_info=True)
            return Cart(cart_id=cart_id, phone=phone, opted_in=False)

    def watch_abandonments(self, source: Any) -> None:
        """Subscribe to a cart-event source.

        The source must expose ``on_cart_abandoned(callback)``; the callback
        receives the same kwargs as :meth:`cart_abandoned`.
        """
        try:
            source.on_cart_abandoned(self.cart_abandoned)
        except Exception:  # noqa: BLE001
            _log.debug("watch_abandonments failed", exc_info=True)

    def handle_webhook(self, event: dict[str, Any]) -> Cart | None:
        """Normalize a store webhook (Medusa / Shopify / WooCommerce) to a cart.

        Returns the registered Cart, or None when the event isn't a cart
        abandonment. Unknown shapes are ignored, never raised on.
        """
        try:
            normalized = normalize_webhook(event or {})
            if normalized is None:
                return None
            return self.cart_abandoned(**normalized)
        except Exception:  # noqa: BLE001
            _log.debug("handle_webhook failed", exc_info=True)
            return None

    # ── sequence ────────────────────────────────────────────────────────

    def run_step(self, cart_id: str, step: int) -> bool:
        """Send one sequence step. Returns True when a message went out.

        Checks the cart is still abandoned first — recovery cancels.
        """
        try:
            row = self._db.execute(
                "SELECT * FROM carts WHERE cart_id = ?", (cart_id,)
            ).fetchone()
            if row is None or row["status"] != "abandoned" or not row["opted_in"]:
                self._cancel_step(cart_id, step)
                return False
            template_key = next(
                (t for n, _d, t in STEPS if n == step), None
            )
            if template_key is None:
                return False
            cart = self._row_to_cart(row)
            text = self._render(template_key, cart)
            ok = self.sender(cart.phone, text)
            now = self._now()
            self._db.execute(
                """UPDATE cart_steps SET sent_at = ?, status = ?
                   WHERE cart_id = ? AND step = ?""",
                (now if ok else 0, "sent" if ok else "failed", cart_id, step),
            )
            self._db.commit()
            return bool(ok)
        except Exception:  # noqa: BLE001 — steps never raise
            _log.debug("run_step failed", exc_info=True)
            return False

    def run_due_steps(self) -> int:
        """Send all pending steps whose run_at has passed. Returns count sent."""
        try:
            now = self._now()
            rows = self._db.execute(
                """SELECT cart_id, step FROM cart_steps
                   WHERE status = 'pending' AND run_at <= ?""",
                (now,),
            ).fetchall()
            sent = 0
            for row in rows:
                if self.run_step(row["cart_id"], row["step"]):
                    sent += 1
            return sent
        except Exception:  # noqa: BLE001
            _log.debug("run_due_steps failed", exc_info=True)
            return 0

    def mark_recovered(
        self, cart_id: str, *, recovered_kobo: int | None = None
    ) -> bool:
        """Mark a cart recovered: stops the sequence, records revenue."""
        try:
            row = self._db.execute(
                "SELECT * FROM carts WHERE cart_id = ?", (cart_id,)
            ).fetchone()
            if row is None:
                return False
            now = self._now()
            amount = int(recovered_kobo) if recovered_kobo is not None \
                else int(row["total_kobo"] or 0)
            self._db.execute(
                "UPDATE carts SET status = 'recovered' WHERE cart_id = ?",
                (cart_id,),
            )
            self._db.execute(
                """UPDATE cart_steps SET status = 'cancelled'
                   WHERE cart_id = ? AND status = 'pending'""",
                (cart_id,),
            )
            self._db.execute(
                """INSERT OR REPLACE INTO recoveries
                   (cart_id, recovered_kobo, recovered_at, store)
                   VALUES (?, ?, ?, ?)""",
                (cart_id, amount, now, row["store"]),
            )
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            _log.debug("mark_recovered failed", exc_info=True)
            return False

    # ── reporting ───────────────────────────────────────────────────────

    def weekly_report(self) -> str:
        """White-label revenue report: 'Devon recovered ₦X for you.'"""
        try:
            since = self._now() - 7 * 86400
            row = self._db.execute(
                """SELECT COUNT(*) AS n, COALESCE(SUM(recovered_kobo), 0) AS total
                   FROM recoveries WHERE recovered_at >= ?""",
                (since,),
            ).fetchone()
            n, total = int(row["n"] or 0), int(row["total"] or 0)
            if n == 0:
                return "No carts recovered this week yet — the sequence is watching. 👀"
            return (
                f"💰 Devon recovered {_naira(total)} for you this week "
                f"({n} cart{'s' if n != 1 else ''})."
            )
        except Exception:  # noqa: BLE001
            _log.debug("weekly_report failed", exc_info=True)
            return "Recovery report unavailable."

    def stats(self) -> dict[str, Any]:
        """Raw numbers for dashboards/tools."""
        try:
            since = self._now() - 7 * 86400
            carts = self._db.execute(
                "SELECT COUNT(*) AS n FROM carts WHERE created_at >= ?",
                (since,),
            ).fetchone()["n"]
            rec = self._db.execute(
                """SELECT COUNT(*) AS n, COALESCE(SUM(recovered_kobo), 0) AS t
                   FROM recoveries WHERE recovered_at >= ?""",
                (since,),
            ).fetchone()
            sent = self._db.execute(
                "SELECT COUNT(*) AS n FROM cart_steps WHERE status = 'sent'"
            ).fetchone()["n"]
            return {
                "abandoned_7d": int(carts or 0),
                "recovered_7d": int(rec["n"] or 0),
                "recovered_kobo_7d": int(rec["t"] or 0),
                "messages_sent": int(sent or 0),
            }
        except Exception:  # noqa: BLE001
            return {}

    # ── internals ───────────────────────────────────────────────────────

    def _cancel_step(self, cart_id: str, step: int) -> None:
        try:
            self._db.execute(
                """UPDATE cart_steps SET status = 'cancelled'
                   WHERE cart_id = ? AND step = ? AND status = 'pending'""",
                (cart_id, step),
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def _row_to_cart(self, row: sqlite3.Row) -> Cart:
        try:
            items = json.loads(row["items_json"] or "[]")
        except Exception:  # noqa: BLE001
            items = []
        return Cart(
            cart_id=row["cart_id"], phone=row["phone"], items=items,
            total_kobo=int(row["total_kobo"] or 0),
            opted_in=bool(row["opted_in"]), status=row["status"],
            store=row["store"], customer_name=row["customer_name"],
            checkout_link=row["checkout_link"], created_at=row["created_at"],
        )

    def _render(self, template_key: str, cart: Cart) -> str:
        name = cart.customer_name.strip() or "there"
        return TEMPLATES[template_key].format(
            name=name,
            store=cart.store,
            items=_items_summary(cart.items),
            total=_naira(cart.total_kobo),
            link=cart.checkout_link or "(checkout link)",
        )


def normalize_webhook(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a store webhook event to :meth:`CartRecovery.cart_abandoned` kwargs.

    Recognized shapes:
    - Medusa: ``{"type": "cart.updated", "data": {"cart": {...}}}``
    - Shopify: ``{"topic": "carts/update", "cart": {...}}`` or ``{"cart_token": ...}``
    - WooCommerce: ``{"action": "woocommerce_cart_updated", "cart": {...}}``
    - Generic: ``{"event": "cart.abandoned", "cart": {...}}``

    The inner cart dict may carry: id/token, phone/customer_phone,
    items/line_items, total/total_price (kobo or major units — see below),
    opted_in/marketing_opt_in, store/shop, customer_name, checkout_url.

    Amounts: pass kobo. If the payload carries major units (naira/dollars),
    the ``amount_unit`` key ("major") converts ×100.

    Returns None for unrecognized shapes.
    """
    if not isinstance(event, dict):
        return None
    kind = str(
        event.get("type") or event.get("topic") or event.get("action")
        or event.get("event") or ""
    ).lower()
    if "cart" not in kind and "checkout" not in kind:
        return None
    payload = event.get("data", event)
    cart = payload.get("cart") or payload.get("checkout") or payload
    if not isinstance(cart, dict):
        return None

    cart_id = str(
        cart.get("id") or cart.get("token") or cart.get("cart_token") or ""
    )
    if not cart_id:
        return None
    customer = cart.get("customer") if isinstance(cart.get("customer"), dict) else {}
    phone = str(
        cart.get("phone") or cart.get("customer_phone")
        or customer.get("phone") or ""
    ).strip()
    raw_items = cart.get("items") or cart.get("line_items") or []
    items = [
        {"name": str(i.get("title") or i.get("name") or "item"),
         "qty": int(i.get("quantity") or i.get("qty") or 1)}
        for i in raw_items if isinstance(i, dict)
    ]
    total = cart.get("total_kobo", cart.get("total_price", cart.get("total", 0)))
    try:
        total_f = float(total or 0)
    except (TypeError, ValueError):
        total_f = 0.0
    if str(cart.get("amount_unit") or event.get("amount_unit") or "").lower() == "major":
        total_f *= 100
    opted_in = bool(
        cart.get("opted_in") or cart.get("marketing_opt_in")
        or cart.get("accepts_marketing") or False
    )
    return {
        "cart_id": cart_id,
        "phone": phone,
        "items": items,
        "total_kobo": int(total_f),
        "opted_in": opted_in,
        "store": str(event.get("store") or cart.get("store") or cart.get("shop") or "unknown"),
        "customer_name": str(
            cart.get("customer_name") or customer.get("name")
            or f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
        ),
        "checkout_link": str(
            cart.get("checkout_url") or cart.get("abandoned_checkout_url") or ""
        ),
    }
