"""Cart recovery via WhatsApp — abandoned-cart → 3-step sequence → recovered revenue.

The highest-ROI business automation in the commerce phase: abandoned carts
get a 3-message WhatsApp sequence (30min → 24h → 72h), and every recovery is
tracked so the business sees "Devon recovered ₦X for you."

Cadence follows the industry consensus (Baymard / Klaviyo): the first touch
is a plain reminder with NO discount, the second handles the real objection
(unexpected costs at checkout are the #1 abandonment reason), and only the
final nudge may carry an incentive — reserved at intake and never shown in
steps 1-2, so shoppers don't learn to abandon for a discount.

Hard rules (test-enforced):
- Explicit opt-in REQUIRED. Only carts with opted_in=True enter the
  sequence. No opt-in → no messages, ever. The opt-in *source* is recorded
  (Meta requires documented consent naming the business).
- Max 3 messages per cart. Recovery cancels the remaining steps.
- Never raises on send failure — a dead bridge must not break the pipeline.
  Failed sends retry once with a 15-minute backoff.

Sender seam (for #68's cost-aware layer):
    Messages go through ``sender(phone, text) -> bool``. The default sender
    uses the WhatsApp bridge adapter. #68 will wrap this callable with
    cost-awareness — the seam is the contract, not the implementation.

Coupon seam:
    ``coupon_fn(cart) -> str | None`` mints an incentive code for the final
    nudge. Wire it to ``StoreManager.create_discount`` so the code is a real
    coupon in the store's own backend. The code is reserved at intake and
    rendered ONLY in step 3.

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

#: The default 3-step sequence: (step_number, delay_seconds, template_key).
#: 30 min is right for WhatsApp (read within ~90s); 24h/72h follow the
#: email-proven cadence. Override via CartRecovery(steps=...).
STEPS: tuple[tuple[int, int, str], ...] = (
    (1, 30 * 60, "reminder"),      # 30 min — "forgot something?", no discount
    (2, 24 * 3600, "nudge"),       # 24 h  — objection handling, no discount
    (3, 72 * 3600, "last_call"),   # 72 h  — urgency + optional incentive
)
MAX_MESSAGES = 3
MAX_RETRIES = 2
RETRY_BACKOFF_S = 15 * 60  # 15 minutes


def _naira(kobo: int) -> str:
    return f"₦{kobo / 100:,.0f}"


def _items_summary(items: list[dict[str, Any]]) -> str:
    items = items or []
    names = [str(i.get("name") or i.get("title") or "item") for i in items]
    if len(names) <= 2:
        return " and ".join(names) if names else "your items"
    return f"{names[0]}, {names[1]} (+{len(names) - 2} more)"


TEMPLATES: dict[str, str] = {
    "reminder": (
        "👋 Hi {name}, you left something in your cart at {store} — "
        "{items} ({total}).\nStill want it? {link}{opt_out}"
    ),
    # Step 2 handles the #1 abandonment reason: unexpected costs at
    # checkout (Baymard: 39-48%). Reassure, don't just repeat the nudge.
    "nudge": (
        "Still thinking about it, {name}? {items} ({total}) at {store} "
        "is still waiting.\nNo hidden fees at checkout, easy returns, and "
        "real humans on support if you need help.\n{link}{opt_out}"
    ),
    # Step 3 is the only step that may carry an incentive.
    "last_call": (
        "⏰ Last call, {name} — your cart at {store} ({items}, {total}) "
        "expires soon.{incentive}\nGrab it while it's there: {link}{opt_out}"
    ),
}


@dataclass
class Cart:
    cart_id: str
    phone: str
    items: list[dict[str, Any]] = field(default_factory=list)
    total_kobo: int = 0
    opted_in: bool = False
    opt_in_source: str = ""       # where documented consent came from
    status: str = "abandoned"     # abandoned | recovered | expired
    store: str = "unknown"
    customer_name: str = ""
    checkout_link: str = ""
    coupon_code: str = ""         # incentive reserved for step 3 only
    discount_pct: int = 0
    created_at: float = 0.0


@dataclass
class RecoveryStep:
    cart_id: str
    step: int
    run_at: float
    sent_at: float = 0.0
    status: str = "pending"  # pending | sent | failed | cancelled
    attempts: int = 0
    incentive_code: str = ""


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


#: The store's own DDL, in each table's FINAL shape (the columns
#: the in-file ALTER migrations add are included) — the native
#: path executes it and the seam mapping is derived from it, so
#: the two can never drift. The native path still runs its ALTER
#: loop for files created before those columns existed.
_DDL = (
    "CREATE TABLE IF NOT EXISTS carts ("
    " cart_id TEXT PRIMARY KEY,"
    " phone TEXT NOT NULL,"
    " items_json TEXT NOT NULL DEFAULT '[]',"
    " total_kobo INTEGER NOT NULL DEFAULT 0,"
    " opted_in INTEGER NOT NULL DEFAULT 0,"
    " status TEXT NOT NULL DEFAULT 'abandoned',"
    " store TEXT NOT NULL DEFAULT 'unknown',"
    " customer_name TEXT NOT NULL DEFAULT '',"
    " checkout_link TEXT NOT NULL DEFAULT '',"
    " created_at REAL NOT NULL,"
    " opt_in_source TEXT NOT NULL DEFAULT '',"
    " coupon_code TEXT NOT NULL DEFAULT '',"
    " discount_pct INTEGER NOT NULL DEFAULT 0);"
    "CREATE TABLE IF NOT EXISTS cart_steps ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " cart_id TEXT NOT NULL,"
    " step INTEGER NOT NULL,"
    " run_at REAL NOT NULL,"
    " sent_at REAL NOT NULL DEFAULT 0,"
    " status TEXT NOT NULL DEFAULT 'pending',"
    " attempts INTEGER NOT NULL DEFAULT 0,"
    " incentive_code TEXT NOT NULL DEFAULT '',"
    " UNIQUE(cart_id, step));"
    "CREATE TABLE IF NOT EXISTS recoveries ("
    " cart_id TEXT PRIMARY KEY,"
    " recovered_kobo INTEGER NOT NULL,"
    " recovered_at REAL NOT NULL,"
    " store TEXT NOT NULL DEFAULT 'unknown',"
    " recovered_via_step INTEGER NOT NULL DEFAULT 0)")


class CartRecovery:
    """Abandoned-cart recovery engine.

    Args:
        db_path: SQLite file for carts/steps/recoveries.
        sender: ``sender(phone, text) -> bool``. Defaults to the WhatsApp
            bridge. #68's cost-aware layer wraps this callable.
        schedule_fn: ``schedule_fn(task_id, run_at, action, parameters)``.
            When None, steps are stored as pending and ``run_due_steps()``
            executes them (the host wires the real scheduler).
        steps: custom (step_number, delay_seconds, template_key) sequence.
            Defaults to STEPS.
        coupon_fn: ``coupon_fn(cart) -> str | None`` — mints an incentive
            code reserved for the final nudge only. Wire to
            ``StoreManager.create_discount``.
        opt_out_text: appended to marketing steps so recipients can reply
            STOP (reduces Meta spam reports). Empty string disables.
        now: injectable clock for tests.
        store: a resolved document store (Mongo program P5) — when
            given (or the boot ambient resolves one), carts and
            recoveries are documents in
            ``commerce-cart-recovery__carts`` / ``__recoveries`` and
            cart_steps ride ``put_raw`` under caller-shaped
            ``<cart_id>:<step>`` ids (registered in storage.sidecars
            as commerce-cart-recovery). The file's existing rows are
            imported once, through the same mapping, at construction.
    """

    def __init__(
        self,
        db_path: str,
        *,
        sender: Callable[[str, str], bool] | None = None,
        schedule_fn: Callable[..., Any] | None = None,
        steps: tuple[tuple[int, int, str], ...] | None = None,
        coupon_fn: Callable[[Cart], str | None] | None = None,
        opt_out_text: str = "Reply STOP to opt out.",
        now: Callable[[], float] | None = None,
        store: Any = None,
    ) -> None:
        self.db_path = db_path
        self.sender = sender or default_whatsapp_sender
        self.schedule_fn = schedule_fn
        self.steps = steps or STEPS
        self.coupon_fn = coupon_fn
        self.opt_out_text = opt_out_text
        self._now = now or time.time
        self._port: Any = None
        from ..storage.sidecars import sidecar_port
        port = sidecar_port(
            "commerce-cart-recovery", _DDL, ("carts", "recoveries"),
            None if db_path == ":memory:" else db_path, store=store)
        if port is not None:
            self._db: sqlite3.Connection | None = None
            self._import_into(port)
            self._port = port
            return
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    def _import_into(self, port: Any) -> None:
        """The one-time lazy file import (see SidecarPort): carts and
        recoveries go through their RowTables; cart_steps are child
        rows addressed by (cart_id, step), so they ride put_raw with
        caller-shaped ``<cart_id>:<step>`` ids (the file's
        AUTOINCREMENT id is never read and is not carried)."""

        def write_carts(rows: list[dict[str, Any]]) -> int:
            table = port.tables["carts"]
            for row in rows:
                table.put(row)
            return len(rows)

        def write_steps(rows: list[dict[str, Any]]) -> int:
            for row in rows:
                body = {k: v for k, v in row.items() if k != "id"}
                port.put_raw(
                    "cart_steps", f"{row['cart_id']}:{row['step']}", body)
            return len(rows)

        def write_recoveries(rows: list[dict[str, Any]]) -> int:
            table = port.tables["recoveries"]
            for row in rows:
                table.put(row)
            return len(rows)

        port.ensure_imported({"carts": write_carts,
                              "cart_steps": write_steps,
                              "recoveries": write_recoveries})

    def _init_schema(self) -> None:
        self._db.executescript(_DDL)
        # Migrate older DBs forward (new columns since the first version).
        for table, col, ddl in (
            ("carts", "opt_in_source", "TEXT NOT NULL DEFAULT ''"),
            ("carts", "coupon_code", "TEXT NOT NULL DEFAULT ''"),
            ("carts", "discount_pct", "INTEGER NOT NULL DEFAULT 0"),
            ("cart_steps", "attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("cart_steps", "incentive_code", "TEXT NOT NULL DEFAULT ''"),
            ("recoveries", "recovered_via_step", "INTEGER NOT NULL DEFAULT 0"),
        ):
            try:
                self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError as exc:
                _log.debug("cart_recovery._init_schema failed", exc_info=True)
        self._db.commit()

    # ── rung helpers (native sqlite / document seam) ─────────────────

    def _commit(self) -> None:
        if self._db is not None:
            self._db.commit()

    def _cart_row(self, cart_id: str) -> dict[str, Any] | None:
        """One cart as a plain dict, from whichever rung serves."""
        if self._port is not None:
            return self._port.tables["carts"].get(cart_id)
        row = self._db.execute(
            "SELECT * FROM carts WHERE cart_id = ?", (cart_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def _step_docs(self, cart_id: str = "") -> list[dict[str, Any]]:
        """cart_steps rows as plain dicts (seam docs keep ``_id``)."""
        if self._port is not None:
            docs = self._port.find_raw("cart_steps")
            if cart_id:
                docs = [d for d in docs if d.get("cart_id") == cart_id]
            return docs
        if cart_id:
            rows = self._db.execute(
                "SELECT * FROM cart_steps WHERE cart_id = ?",
                (cart_id,)).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM cart_steps").fetchall()
        return [dict(r) for r in rows]

    def _step_row(self, cart_id: str, step: int) -> dict[str, Any] | None:
        for doc in self._step_docs(cart_id):
            if doc.get("step") == step:
                return doc
        return None

    def _put_step(self, fields: dict[str, Any]) -> None:
        """INSERT OR REPLACE one cart_steps row (full-row write)."""
        if self._port is not None:
            body = {"cart_id": fields["cart_id"], "step": fields["step"],
                    "run_at": fields["run_at"],
                    "sent_at": fields.get("sent_at", 0),
                    "status": fields.get("status", "pending"),
                    "attempts": fields.get("attempts", 0),
                    "incentive_code": fields.get("incentive_code", "")}
            self._port.put_raw(
                "cart_steps", f"{fields['cart_id']}:{fields['step']}",
                body)
            return
        self._db.execute(
            """INSERT OR REPLACE INTO cart_steps
               (cart_id, step, run_at, sent_at, status, attempts,
                incentive_code)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (fields["cart_id"], fields["step"], fields["run_at"],
             fields.get("sent_at", 0), fields.get("status", "pending"),
             fields.get("attempts", 0), fields.get("incentive_code", "")),
        )

    def _update_step(self, cart_id: str, step: int,
                     **fields: Any) -> None:
        """Targeted cart_steps UPDATE by (cart_id, step)."""
        if self._port is not None:
            doc = self._step_row(cart_id, step)
            if doc is None:
                return
            body = {k: v for k, v in doc.items()
                    if k not in ("id", "_id")}
            body.update(fields)
            self._port.put_raw("cart_steps", doc["_id"], body)
            return
        cols = ", ".join(f"{name} = ?" for name in fields)
        self._db.execute(
            f"UPDATE cart_steps SET {cols} WHERE cart_id = ? AND step = ?",
            (*fields.values(), cart_id, step),
        )

    def _update_cart(self, cart_id: str, **fields: Any) -> None:
        """Targeted carts UPDATE by cart_id."""
        if self._port is not None:
            self._port.tables["carts"].update(cart_id, fields)
            return
        cols = ", ".join(f"{name} = ?" for name in fields)
        self._db.execute(
            f"UPDATE carts SET {cols} WHERE cart_id = ?",
            (*fields.values(), cart_id),
        )

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
        opt_in_source: str = "",
        coupon_code: str = "",
        discount_pct: int = 0,
    ) -> Cart:
        """Register an abandoned cart. Starts the sequence ONLY when opted_in.

        Never raises. No opt-in → cart is recorded (for reporting) but no
        messages are ever scheduled.

        When ``coupon_fn`` is set (or coupon_code/discount_pct passed), the
        incentive is reserved for the FINAL step only.
        """
        try:
            now = self._now()
            items = list(items or [])
            cart = Cart(
                cart_id=cart_id, phone=phone, items=items,
                total_kobo=int(total_kobo or 0), opted_in=bool(opted_in),
                opt_in_source=str(opt_in_source or ""),
                store=store, customer_name=customer_name,
                checkout_link=checkout_link,
                coupon_code=str(coupon_code or ""),
                discount_pct=int(discount_pct or 0),
                created_at=now,
            )
            if self._port is not None:
                self._port.tables["carts"].put({
                    "cart_id": cart_id, "phone": phone,
                    "items_json": json.dumps(items),
                    "total_kobo": cart.total_kobo,
                    "opted_in": 1 if cart.opted_in else 0,
                    "opt_in_source": cart.opt_in_source,
                    "status": "abandoned", "store": store,
                    "customer_name": customer_name,
                    "checkout_link": checkout_link,
                    "coupon_code": cart.coupon_code,
                    "discount_pct": cart.discount_pct,
                    "created_at": now})
            else:
                self._db.execute(
                    """INSERT OR REPLACE INTO carts
                       (cart_id, phone, items_json, total_kobo, opted_in,
                        opt_in_source, status, store, customer_name,
                        checkout_link, coupon_code, discount_pct, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'abandoned', ?, ?, ?, ?, ?, ?)""",
                    (cart_id, phone, json.dumps(items), cart.total_kobo,
                     1 if cart.opted_in else 0, cart.opt_in_source,
                     store, customer_name, checkout_link,
                     cart.coupon_code, cart.discount_pct, now),
                )
            if cart.opted_in:
                incentive = self._reserve_incentive(cart)
                last_step_no = max(n for n, _d, _t in self.steps)
                for step_no, delay, _template in self.steps:
                    run_at = now + delay
                    if self._port is not None:
                        self._put_step({
                            "cart_id": cart_id, "step": step_no,
                            "run_at": run_at, "status": "pending",
                            "incentive_code":
                                incentive if step_no == last_step_no
                                else ""})
                    else:
                        self._db.execute(
                            """INSERT OR REPLACE INTO cart_steps
                               (cart_id, step, run_at, status, incentive_code)
                               VALUES (?, ?, ?, 'pending', ?)""",
                            (cart_id, step_no, run_at,
                             incentive if step_no == last_step_no else ""),
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
            self._commit()
            return cart
        except Exception:  # noqa: BLE001 — intake never raises
            _log.debug("cart_abandoned failed", exc_info=True)
            return Cart(cart_id=cart_id, phone=phone, opted_in=False)

    def _reserve_incentive(self, cart: Cart) -> str:
        """Mint (or reuse) the step-3-only incentive code. Never raises."""
        if cart.coupon_code:
            return cart.coupon_code
        if self.coupon_fn is None:
            return ""
        try:
            code = self.coupon_fn(cart)
            return str(code or "")
        except Exception:  # noqa: BLE001 — no incentive beats a crash
            _log.debug("coupon_fn failed", exc_info=True)
            return ""

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
        Failed sends retry once after a backoff (never raise).
        """
        try:
            row = self._cart_row(cart_id)
            if row is None or row["status"] != "abandoned" or not row["opted_in"]:
                self._cancel_step(cart_id, step)
                return False
            template_key = next(
                (t for n, _d, t in self.steps if n == step), None
            )
            if template_key is None:
                return False
            step_row = self._step_row(cart_id, step)
            attempts = int(step_row["attempts"] or 0) if step_row else 0
            incentive = (step_row["incentive_code"] or "") if step_row else ""
            cart = self._row_to_cart(row)
            if incentive and not cart.coupon_code:
                cart.coupon_code = incentive
            text = self._render(template_key, cart, step)
            try:
                ok = bool(self.sender(cart.phone, text))
            except Exception:  # noqa: BLE001 — sender seam never raises out
                _log.debug("sender raised for %s", cart_id, exc_info=True)
                ok = False
            now = self._now()
            if ok:
                self._update_step(cart_id, step, sent_at=now,
                                  status="sent", attempts=attempts + 1)
            else:
                attempts += 1
                if attempts < MAX_RETRIES:
                    # Retry once after a backoff — a flapping bridge
                    # shouldn't cost the whole sequence.
                    self._update_step(cart_id, step, status="failed",
                                      attempts=attempts,
                                      run_at=now + RETRY_BACKOFF_S)
                else:
                    self._update_step(cart_id, step, status="failed",
                                      attempts=attempts)
            self._commit()
            return ok
        except Exception:  # noqa: BLE001 — steps never raise
            _log.debug("run_step failed", exc_info=True)
            return False

    def run_due_steps(self) -> int:
        """Send all pending steps whose run_at has passed. Returns count sent.

        Also picks up failed steps that are still within their retry budget.
        """
        try:
            now = self._now()
            if self._port is not None:
                rows = [d for d in self._step_docs()
                        if d["status"] in ("pending", "failed")
                        and (d["attempts"] or 0) < MAX_RETRIES
                        and d["run_at"] <= now]
            else:
                rows = self._db.execute(
                    """SELECT cart_id, step FROM cart_steps
                       WHERE status IN ('pending', 'failed')
                         AND attempts < ? AND run_at <= ?""",
                    (MAX_RETRIES, now),
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
        """Mark a cart recovered: stops the sequence, records revenue.

        Attribution: ``recovered_via_step`` records the last step that was
        sent before recovery (0 when recovery happened with no message).
        """
        try:
            row = self._cart_row(cart_id)
            if row is None:
                return False
            now = self._now()
            amount = int(recovered_kobo) if recovered_kobo is not None \
                else int(row["total_kobo"] or 0)
            if self._port is not None:
                sent_steps = [d["step"] for d in self._step_docs(cart_id)
                              if d["status"] == "sent"]
                via = max(sent_steps) if sent_steps else 0
                self._update_cart(cart_id, status="recovered")
                for doc in self._step_docs(cart_id):
                    if doc["status"] in ("pending", "failed"):
                        self._update_step(cart_id, doc["step"],
                                          status="cancelled")
                self._port.tables["recoveries"].put({
                    "cart_id": cart_id, "recovered_kobo": amount,
                    "recovered_at": now, "store": row["store"],
                    "recovered_via_step": int(via or 0)})
                return True
            via = self._db.execute(
                """SELECT COALESCE(MAX(step), 0) AS m FROM cart_steps
                   WHERE cart_id = ? AND status = 'sent'""",
                (cart_id,),
            ).fetchone()["m"]
            self._db.execute(
                "UPDATE carts SET status = 'recovered' WHERE cart_id = ?",
                (cart_id,),
            )
            self._db.execute(
                """UPDATE cart_steps SET status = 'cancelled'
                   WHERE cart_id = ? AND status IN ('pending', 'failed')""",
                (cart_id,),
            )
            self._db.execute(
                """INSERT OR REPLACE INTO recoveries
                   (cart_id, recovered_kobo, recovered_at, store,
                    recovered_via_step)
                   VALUES (?, ?, ?, ?, ?)""",
                (cart_id, amount, now, row["store"], int(via or 0)),
            )
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            _log.debug("mark_recovered failed", exc_info=True)
            return False

    def mark_expired(self, cart_id: str) -> bool:
        """Age a dead cart out of the funnel (never recovered, sequence done)."""
        try:
            row = self._cart_row(cart_id)
            if row is None or row["status"] != "abandoned":
                return False
            if self._port is not None:
                open_steps = sum(
                    1 for d in self._step_docs(cart_id)
                    if d["status"] in ("pending", "failed")
                    and (d["attempts"] or 0) < MAX_RETRIES)
            else:
                open_steps = self._db.execute(
                    """SELECT COUNT(*) AS n FROM cart_steps
                       WHERE cart_id = ? AND status IN ('pending', 'failed')
                         AND attempts < ?""",
                    (cart_id, MAX_RETRIES),
                ).fetchone()["n"]
            if open_steps:
                return False  # sequence still has life
            self._update_cart(cart_id, status="expired")
            self._commit()
            return True
        except Exception:  # noqa: BLE001
            _log.debug("mark_expired failed", exc_info=True)
            return False

    def expire_dead_carts(self, *, grace_days: float = 7.0) -> int:
        """Expire abandoned carts whose whole sequence finished long ago."""
        try:
            cutoff = self._now() - grace_days * 86400
            if self._port is not None:
                rows = [r for r in self._port.tables["carts"].find()
                        if r["status"] == "abandoned"
                        and r["created_at"] < cutoff]
            else:
                rows = self._db.execute(
                    """SELECT cart_id FROM carts
                       WHERE status = 'abandoned' AND created_at < ?""",
                    (cutoff,),
                ).fetchall()
            expired = sum(1 for r in rows
                          if self.mark_expired(r["cart_id"]))
            return expired
        except Exception:  # noqa: BLE001
            _log.debug("expire_dead_carts failed", exc_info=True)
            return 0

    # ── reporting ───────────────────────────────────────────────────────

    def weekly_report(self) -> str:
        """White-label revenue report: 'Devon recovered ₦X for you.'"""
        try:
            since = self._now() - 7 * 86400
            if self._port is not None:
                recs = [r for r in self._port.tables["recoveries"].find()
                        if r["recovered_at"] >= since]
                n = len(recs)
                total = sum(int(r["recovered_kobo"] or 0) for r in recs)
            else:
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

    def funnel_report(self, days: float = 7.0) -> str:
        """Funnel: abandoned → recovered | expired, with rate vs benchmark."""
        try:
            since = self._now() - days * 86400
            if self._port is not None:
                counts: dict[str, int] = {}
                for r in self._port.tables["carts"].find():
                    if r["created_at"] >= since:
                        counts[r["status"]] = counts.get(r["status"], 0) + 1
                total = sum(
                    int(r["recovered_kobo"] or 0)
                    for r in self._port.tables["recoveries"].find()
                    if r["recovered_at"] >= since)
            else:
                counts = dict(self._db.execute(
                    """SELECT status, COUNT(*) AS n FROM carts
                       WHERE created_at >= ? GROUP BY status""",
                    (since,),
                ).fetchall() or [])
                rec = self._db.execute(
                    """SELECT COALESCE(SUM(recovered_kobo), 0) AS t
                       FROM recoveries WHERE recovered_at >= ?""",
                    (since,),
                ).fetchone()
                total = int(rec["t"] or 0)
            abandoned = sum(int(v or 0) for v in counts.values())
            recovered = int(counts.get("recovered") or 0)
            rate = (100.0 * recovered / abandoned) if abandoned else 0.0
            # Klaviyo benchmark: 5-15% cart recovery rate is healthy.
            verdict = ("🟢 above" if rate >= 5 else "🟡 below") + \
                " the 5-15% industry benchmark"
            return (
                f"🛒 Cart funnel ({int(days)}d): {abandoned} abandoned → "
                f"{recovered} recovered ({rate:.1f}% — {verdict}), "
                f"{counts.get('expired', 0)} expired. "
                f"Revenue recovered: {_naira(total)}."
            )
        except Exception:  # noqa: BLE001
            _log.debug("funnel_report failed", exc_info=True)
            return "Funnel report unavailable."

    def stats(self) -> dict[str, Any]:
        """Raw numbers for dashboards/tools."""
        try:
            since = self._now() - 7 * 86400
            if self._port is not None:
                carts = sum(
                    1 for r in self._port.tables["carts"].find()
                    if r["created_at"] >= since)
                recs = [r for r in self._port.tables["recoveries"].find()
                        if r["recovered_at"] >= since]
                rec = {"n": len(recs),
                       "t": sum(int(r["recovered_kobo"] or 0)
                                for r in recs)}
                steps = self._step_docs()
                sent = sum(1 for d in steps if d["status"] == "sent")
                failed = sum(1 for d in steps if d["status"] == "failed")
                expired = sum(
                    1 for r in self._port.tables["carts"].find()
                    if r["status"] == "expired"
                    and r["created_at"] >= since)
                per_step: dict[str, int] = {}
                for r in recs:
                    key = str(r["recovered_via_step"])
                    per_step[key] = per_step.get(key, 0) + 1
                recovered_n = int(rec["n"] or 0)
                return {
                    "abandoned_7d": int(carts or 0),
                    "recovered_7d": recovered_n,
                    "recovered_kobo_7d": int(rec["t"] or 0),
                    "messages_sent": int(sent or 0),
                    "messages_failed": int(failed or 0),
                    "expired_7d": int(expired or 0),
                    "recovered_via_step_7d": per_step,
                    "revenue_per_recipient_kobo_7d": (
                        int(rec["t"] // recovered_n) if recovered_n else 0
                    ),
                }
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
            failed = self._db.execute(
                "SELECT COUNT(*) AS n FROM cart_steps WHERE status = 'failed'"
            ).fetchone()["n"]
            expired = self._db.execute(
                """SELECT COUNT(*) AS n FROM carts
                   WHERE status = 'expired' AND created_at >= ?""",
                (since,),
            ).fetchone()["n"]
            per_step = {
                str(r["recovered_via_step"]): int(r["n"])
                for r in self._db.execute(
                    """SELECT recovered_via_step, COUNT(*) AS n
                       FROM recoveries WHERE recovered_at >= ?
                       GROUP BY recovered_via_step""",
                    (since,),
                ).fetchall()
            }
            recovered_n = int(rec["n"] or 0)
            return {
                "abandoned_7d": int(carts or 0),
                "recovered_7d": recovered_n,
                "recovered_kobo_7d": int(rec["t"] or 0),
                "messages_sent": int(sent or 0),
                "messages_failed": int(failed or 0),
                "expired_7d": int(expired or 0),
                "recovered_via_step_7d": per_step,
                "revenue_per_recipient_kobo_7d": (
                    int(rec["t"] // recovered_n) if recovered_n else 0
                ),
            }
        except Exception:  # noqa: BLE001
            return {}

    # ── internals ───────────────────────────────────────────────────────

    def _cancel_step(self, cart_id: str, step: int) -> None:
        try:
            if self._port is not None:
                doc = self._step_row(cart_id, step)
                if doc is not None \
                        and doc["status"] in ("pending", "failed"):
                    self._update_step(cart_id, step, status="cancelled")
                return
            self._db.execute(
                """UPDATE cart_steps SET status = 'cancelled'
                   WHERE cart_id = ? AND step = ?
                     AND status IN ('pending', 'failed')""",
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
            cart_id=row["cart_id"], phone=row["phone"], items=items or [],
            total_kobo=int(row["total_kobo"] or 0),
            opted_in=bool(row["opted_in"]),
            opt_in_source=row["opt_in_source"] or "",
            status=row["status"],
            store=row["store"], customer_name=row["customer_name"],
            checkout_link=row["checkout_link"],
            coupon_code=row["coupon_code"] or "",
            discount_pct=int(row["discount_pct"] or 0),
            created_at=row["created_at"],
        )

    def _checkout_link(self, cart: Cart, step: int) -> str:
        """Checkout link tagged for step-level attribution."""
        base = cart.checkout_link or "(checkout link)"
        if base == "(checkout link)":
            return base
        sep = "&" if "?" in base else "?"
        return (f"{base}{sep}utm_source=devon-recovery"
                f"&utm_medium=whatsapp&utm_campaign=cart-step-{step}")

    def _render(self, template_key: str, cart: Cart, step: int) -> str:
        name = (cart.customer_name or "").strip() or "there"
        incentive = ""
        if template_key == "last_call" and cart.coupon_code:
            pct = f" {cart.discount_pct}%" if cart.discount_pct else ""
            incentive = (
                f"\n🎁 Here's{pct} off to finish it — use code "
                f"{cart.coupon_code} at checkout."
            )
        opt_out = (f"\n\n{self.opt_out_text}" if self.opt_out_text else "")
        return TEMPLATES[template_key].format(
            name=name,
            store=cart.store,
            items=_items_summary(cart.items),
            total=_naira(cart.total_kobo),
            link=self._checkout_link(cart, step),
            incentive=incentive,
            opt_out=opt_out,
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
    opted_in/marketing_opt_in, store/shop, customer_name, checkout_url,
    opt_in_source, coupon_code, discount_pct.

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
    try:
        discount_pct = int(cart.get("discount_pct") or 0)
    except (TypeError, ValueError):
        discount_pct = 0
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
        "opt_in_source": str(cart.get("opt_in_source") or ""),
        "coupon_code": str(cart.get("coupon_code") or ""),
        "discount_pct": discount_pct,
    }
