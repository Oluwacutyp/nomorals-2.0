"""Chat verbs for the dormant commerce organs, behind /store.

Audit gap R1-5: payments, logistics, reconciliation, support,
analytics, catalog, storefront and tax existed and were unit-pinned,
but nothing in production reached them — only medusa.py +
cart_recovery.py were live via /store. This module is the surface:
``handle_store_commerce`` dispatches the commerce first-words, and
the runtime's ``_control_store`` calls it before the
``/store <id> <instruction>`` fallthrough.

Money is integer minor units (kobo) everywhere — the organs' own
contract (``price_minor``, ``amount_minor``). Usage lines say so.

Honesty rules, inherited from the organs:
- A live charge/refund needs the provider's real secret key AND a
  real HTTP round-trip (``core.http.HttpClient``). Without either it
  refuses in words — never a fabricated success.
- Quotes and ETAs are labelled estimates from the offline tariff
  table, never a live courier quote.
- Tracking returns ``not_available`` unless a live tracker is wired
  — a waybill is never invented, a location never guessed.
- A storefront page is brief-only unless an imggen callable is
  supplied — no image path is ever fabricated.
- Reconciliation posts to the real finance ledger (the organ's
  designed purpose) and proves ``balanced``; a payment with no
  matching invoice is flagged unreconciled, never dropped.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analytics import monthly_client_report
from .logistics import CARRIERS, rate_quote, tracking
from .logistics import eta as _eta
from .payments import PROVIDERS, PaymentError, provider_for
from .store import CommerceStore, default_commerce_path
from .storefront import Product, product_page_data
from .support import PRIORITIES
from .tax import compute_tax

__all__ = ["COMMERCE_VERBS", "handle_store_commerce"]

#: First words ``_control_store`` hands to this module (before the
#: ``/store <id> <instruction>`` fallthrough).
COMMERCE_VERBS: set[str] = {
    "products", "page", "pay", "invoice", "subscribe", "ship",
    "reconcile", "ticket", "report", "tax",
}

_USAGE = (
    "usage: /store products list | add <id> <name> <price_kobo> "
    "[| currency | category | stock | description | tags,a,b] | remove <id> "
    "| feed [json|csv] · /store page <product_id> · /store pay providers | "
    "quote <provider> <amount_kobo> | charge <provider> <amount_kobo> "
    "[| customer | reference | channel] | record <provider> <amount_kobo> "
    "<reference> [| customer | channel] | list [provider] | refund "
    "<reference> [amount_kobo] | dispute <reference> <amount_kobo> <reason> "
    "| disputes | resolve <dispute_id> <won|lost|accepted> [resolution] · "
    "/store invoice create <customer> <amount_kobo> [| currency] | list | "
    "pay <invoice_id> <payment_reference> · /store subscribe create "
    "<provider> <customer> <amount_kobo> [| interval] | list | cancel <id> "
    "| dunning [n] · "
    "/store ship quote <carrier> <weight_kg> [| distance_km] | eta <carrier> "
    "[| distance_km | same-city] | send <carrier> <waybill> [| order_ref | "
    "recipient] | track <carrier> <waybill> | list · /store reconcile run | "
    "status | unreconciled · /store ticket open <customer> <message> "
    "[| priority | channel | phone] | list [open|all] | show <id> | assign "
    "<id> <assignee> | reply <id> [template] | close <id> [resolution] | "
    "stats · /store report [YYYY-MM] · /store tax <amount_kobo> "
    "[jurisdiction]"
)


def _fmt(minor: int, currency: str = "NGN") -> str:
    """Minor units → human money (naira via the finance formatter)."""
    if (currency or "NGN").upper() == "NGN":
        from ..finance.ledger import format_naira

        return format_naira(int(minor))
    return f"{int(minor) / 100:,.2f} {(currency or '').upper()}"


def _amount(text: str) -> int | None:
    """Minor units, plain integer. None when unparseable / negative."""
    try:
        value = int(str(text).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _split_pipes(rest: str) -> tuple[str, list[str]]:
    """``head | field | field`` → (head, [fields])."""
    parts = [p.strip() for p in (rest or "").split("|")]
    return parts[0], parts[1:]


def _gateway_http() -> Any:
    """The payments module's ``http`` shape over core HttpClient."""
    from ..core.http import HttpClient

    client = HttpClient()

    def _http(method: str, url: str, headers: dict[str, str],
              json_body: dict[str, Any]) -> tuple[int, Any]:
        resp = client.request(
            method, url,
            data=json.dumps(json_body).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"})
        try:
            payload: Any = resp.json()
        except Exception:  # noqa: BLE001 — non-JSON body is still a payload
            payload = {"raw": resp.text}
        return resp.status, payload

    return _http


def _ledger_for(settings: Any) -> Any:
    """The real finance ledger for this home (reconcile/report)."""
    from ..finance.budgets import finance_paths
    from ..finance.ledger import Ledger

    ledger_path, _ = finance_paths(settings)
    return Ledger(ledger_path)


def handle_store_commerce(tail: str, *, home: Path | str,
                          settings: Any = None,
                          ledger: Any = None,
                          http: Any = None) -> str:
    """Dispatch one commerce first-word. Owner-only (callers gate first).

    ``home`` is Devon's home dir; the store lives at
    ``<home>/data/commerce.json``. ``settings`` (when given) resolves
    the finance ledger the same way /spend does; ``ledger`` overrides
    it (tests pass an isolated one). ``http`` overrides the live-charge
    transport (tests pass a fake; production builds the HttpClient
    adapter on first use).
    """
    store = CommerceStore(default_commerce_path(home))
    parts = (tail or "").strip().split(None, 1)
    verb = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    try:
        if verb == "products":
            return _products(store, rest)
        if verb == "page":
            return _page(store, rest)
        if verb == "pay":
            return _pay(store, rest, http=http)
        if verb == "invoice":
            return _invoice(store, rest)
        if verb == "subscribe":
            return _subscribe(store, rest)
        if verb == "ship":
            return _ship(store, rest)
        if verb == "reconcile":
            return _reconcile(store, rest, settings=settings,
                              ledger=ledger)
        if verb == "ticket":
            return _ticket(store, rest)
        if verb == "report":
            return _report(store, rest, settings=settings, ledger=ledger)
        if verb == "tax":
            return _tax(rest)
    except (ValueError, PaymentError) as exc:
        return str(exc)
    return _USAGE


# ── catalog ────────────────────────────────────────────────────────


def _products(store: CommerceStore, rest: str) -> str:
    parts = (rest or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    arg = parts[1] if len(parts) > 1 else ""
    catalog = store.catalog()
    if sub == "list":
        products = catalog.list()
        if not products:
            return "no products yet — /store products add <id> <name> <price_kobo>."
        lines = ["🛍️ products:"]
        for p in products:
            stock = f" — stock {p.get('stock')}" if p.get("stock") else ""
            lines.append(
                f"• {p.get('id')}: {p.get('name')} — "
                f"{_fmt(p.get('price_minor', 0), p.get('currency', 'NGN'))}"
                f"{stock}")
        return "\n".join(lines)
    if sub == "add":
        head, fields = _split_pipes(arg)
        tokens = head.split()
        if len(tokens) < 3:
            return ("usage: /store products add <id> <name> <price_kobo> "
                    "[| currency | category | stock | description | tags,a,b]")
        price = _amount(tokens[-1])
        if price is None:
            return f"price must be integer kobo, got {tokens[-1]!r}"
        product: dict[str, Any] = {
            "id": tokens[0],
            "name": " ".join(tokens[1:-1]),
            "price_minor": price,
            "currency": "NGN",
        }
        if len(fields) > 0 and fields[0]:
            product["currency"] = fields[0].upper()
        if len(fields) > 1 and fields[1]:
            product["category"] = fields[1]
        if len(fields) > 2 and fields[2]:
            try:
                product["stock"] = int(fields[2])
            except ValueError:
                return f"stock must be an integer, got {fields[2]!r}"
        if len(fields) > 3 and fields[3]:
            product["description"] = fields[3]
        if len(fields) > 4 and fields[4]:
            product["tags"] = [t.strip() for t in fields[4].split(",")
                               if t.strip()]
        record = catalog.upsert(product)
        store.save_catalog(catalog)
        return (f"product added: {record['name']} ({record['id']}) — "
                f"{_fmt(record['price_minor'], record.get('currency', 'NGN'))}")
    if sub == "remove":
        pid = arg.strip().split(None, 1)[0] if arg.strip() else ""
        if not pid:
            return "usage: /store products remove <id>"
        if catalog.remove(pid):
            store.save_catalog(catalog)
            return f"removed product: {pid}"
        return f"no product '{pid}'. Use /store products list."
    if sub == "feed":
        fmt = (arg.strip().lower() or "json")
        if fmt not in ("json", "csv"):
            return "usage: /store products feed [json|csv]"
        return catalog.feed_export(fmt)
    return _USAGE


def _page(store: CommerceStore, rest: str) -> str:
    pid = (rest or "").strip().split(None, 1)[0] if rest.strip() else ""
    if not pid:
        return "usage: /store page <product_id>"
    product = store.catalog().get(pid)
    if product is None:
        return f"no product '{pid}'. Use /store products list."
    item = Product(
        id=str(product.get("id")), name=str(product.get("name", "")),
        price_minor=int(product.get("price_minor", 0) or 0),
        currency=str(product.get("currency", "NGN") or "NGN"),
        description=str(product.get("description", "") or ""),
        tags=list(product.get("tags") or []),
        category=str(product.get("category", "") or ""),
        stock=int(product.get("stock", 0) or 0))
    data = product_page_data(item)
    cover = data.get("cover") or {}
    lines = [
        f"🛍️ {data['title']} — {_fmt(data['price_minor'], data['currency'])}",
        f"stock: {data['stock']} ({'in stock' if data['in_stock'] else 'out of stock'})",
    ]
    if data.get("description"):
        lines.append(data["description"])
    if data.get("tags"):
        lines.append("tags: " + ", ".join(data["tags"]))
    lines.append("")
    lines.append("cover brief (brief only — no image generator wired, "
                 "so no image is claimed):")
    lines.append(f"  subject: {cover.get('subject', '')}")
    lines.append(f"  setting: {cover.get('setting', '')}")
    lines.append(f"  prompt: {cover.get('prompt', '')}")
    return "\n".join(lines)


# ── payments ───────────────────────────────────────────────────────


def _pay(store: CommerceStore, rest: str, *, http: Any = None) -> str:
    parts = (rest or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    arg = parts[1] if len(parts) > 1 else ""
    gw = store.gateway(http=http)
    if sub == "providers":
        lines = ["💳 payment rails (local first):"]
        for key, provider in PROVIDERS.items():
            lines.append(
                f"• {provider.name} ({key}) — {provider.currency} · "
                f"{provider.fee_note} Key: {provider.env_key}")
        return "\n".join(lines)
    if sub == "quote":
        tokens = arg.split()
        if len(tokens) < 2:
            return "usage: /store pay quote <provider> <amount_kobo>"
        amount = _amount(tokens[1])
        if amount is None or amount <= 0:
            return f"amount must be a positive integer kobo, got {tokens[1]!r}"
        provider = provider_for(tokens[0])
        fee = provider.fee_for(amount)
        return (f"{provider.name} on {_fmt(amount, provider.currency)}: "
                f"fee {_fmt(fee, provider.currency)} · "
                f"net {_fmt(amount - fee, provider.currency)} "
                f"({provider.fee_note})")
    if sub == "charge":
        head, fields = _split_pipes(arg)
        tokens = head.split()
        if len(tokens) < 2:
            return ("usage: /store pay charge <provider> <amount_kobo> "
                    "[| customer | reference | channel]")
        amount = _amount(tokens[1])
        if amount is None or amount <= 0:
            return f"amount must be a positive integer kobo, got {tokens[1]!r}"
        customer = fields[0] if len(fields) > 0 else ""
        reference = fields[1] if len(fields) > 1 else ""
        channel = fields[2] if len(fields) > 2 else "online"
        if http is None:
            gw = store.gateway(http=_gateway_http())
        payment = gw.charge(tokens[0], amount, customer=customer,
                            reference=reference, channel=channel)
        store.save_gateway(gw)
        return (f"charged {_fmt(payment.amount_minor, payment.currency)} "
                f"via {payment.provider} — ref {payment.reference} · "
                f"fee {_fmt(payment.fee_minor, payment.currency)} · "
                f"net {_fmt(payment.net_minor, payment.currency)} "
                f"(payment {payment.id})")
    if sub == "record":
        head, fields = _split_pipes(arg)
        tokens = head.split()
        if len(tokens) < 3:
            return ("usage: /store pay record <provider> <amount_kobo> "
                    "<reference> [| customer | channel]")
        amount = _amount(tokens[1])
        if amount is None or amount <= 0:
            return f"amount must be a positive integer kobo, got {tokens[1]!r}"
        provider = provider_for(tokens[0])
        reference = tokens[2]
        if gw.find_payment(reference) is not None:
            return (f"payment {reference!r} already recorded — refusing "
                    "to double-count it")
        from .payments import Payment

        payment = Payment(
            id="pay_" + uuid.uuid4().hex[:12],
            provider=provider.name.lower(), amount_minor=amount,
            currency=provider.currency, reference=reference,
            status="succeeded",
            customer=fields[0] if len(fields) > 0 else "",
            channel=fields[1] if len(fields) > 1 else "online",
            fee_minor=provider.fee_for(amount),
            created_at=time.time())
        gw.payments.append(payment)
        store.save_gateway(gw)
        return (f"recorded payment {reference} — "
                f"{_fmt(amount, provider.currency)} via {provider.name} · "
                f"fee {_fmt(payment.fee_minor, provider.currency)} · "
                f"net {_fmt(payment.net_minor, provider.currency)}")
    if sub == "list":
        wanted = (arg.strip().lower() or "")
        payments = [p for p in gw.payments
                    if not wanted or p.provider == wanted]
        if not payments:
            return "no payments recorded yet — /store pay record <provider> <amount_kobo> <reference>."
        lines = ["💳 payments:"]
        for p in payments[-15:]:
            lines.append(
                f"• {p.reference}: {_fmt(p.amount_minor, p.currency)} "
                f"via {p.provider} [{p.status}]"
                + (f" — {p.customer}" if p.customer else ""))
        return "\n".join(lines)
    if sub == "refund":
        tokens = arg.split()
        if not tokens:
            return "usage: /store pay refund <reference> [amount_kobo]"
        payment = gw.find_payment(tokens[0])
        if payment is None:
            return f"no payment with reference {tokens[0]!r}."
        amount = None
        if len(tokens) > 1:
            amount = _amount(tokens[1])
            if amount is None:
                return f"amount must be integer kobo, got {tokens[1]!r}"
        if http is None:
            gw = store.gateway(http=_gateway_http())
            payment = gw.find_payment(tokens[0])
            assert payment is not None
        refund = gw.refund(payment, amount_minor=amount)
        store.save_gateway(gw)
        return (f"refunded {_fmt(-refund.amount_minor, refund.currency)} "
                f"for {payment.reference} (refund {refund.id})")
    if sub == "dispute":
        head, _fields = _split_pipes(arg)
        tokens = head.split(None, 2)
        if len(tokens) < 3:
            return ("usage: /store pay dispute <reference> <amount_kobo> "
                    "<reason>")
        amount = _amount(tokens[1])
        if amount is None or amount <= 0:
            return f"amount must be a positive integer kobo, got {tokens[1]!r}"
        dispute = gw.record_dispute(tokens[0], amount, tokens[2])
        store.save_gateway(gw)
        return (f"dispute recorded: {dispute.id} — {tokens[0]} "
                f"{_fmt(amount)} ({tokens[2]})")
    if sub == "disputes":
        if not gw.disputes:
            return "no disputes recorded."
        lines = ["⚖️ disputes:"]
        for d in gw.disputes:
            lines.append(
                f"• {d.id}: {d.payment_reference} "
                f"{_fmt(d.amount_minor)} [{d.status}] — {d.reason}")
        return "\n".join(lines)
    if sub == "resolve":
        tokens = arg.split(None, 2)
        if len(tokens) < 2:
            return ("usage: /store pay resolve <dispute_id> "
                    "<won|lost|accepted> [resolution]")
        dispute = next((d for d in gw.disputes if d.id == tokens[0]), None)
        if dispute is None:
            return f"no dispute {tokens[0]!r}. Use /store pay disputes."
        gw.resolve_dispute(dispute, tokens[1],
                           resolution=tokens[2] if len(tokens) > 2 else "")
        store.save_gateway(gw)
        return f"dispute {dispute.id} resolved: {tokens[1]}"
    return _USAGE


# ── invoices + subscriptions ───────────────────────────────────────


def _invoice(store: CommerceStore, rest: str) -> str:
    parts = (rest or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    arg = parts[1] if len(parts) > 1 else ""
    gw = store.gateway()
    if sub == "create":
        head, fields = _split_pipes(arg)
        tokens = head.split(None, 1)
        if len(tokens) < 2:
            return ("usage: /store invoice create <customer> "
                    "<amount_kobo> [| currency]")
        amount = _amount(tokens[1])
        if amount is None or amount <= 0:
            return f"amount must be a positive integer kobo, got {tokens[1]!r}"
        currency = fields[0].upper() if fields and fields[0] else "NGN"
        invoice = gw.create_invoice(tokens[0], amount, currency=currency)
        store.save_gateway(gw)
        return (f"invoice created: {invoice.id} — {tokens[0]} "
                f"{_fmt(amount, currency)} [unpaid]")
    if sub == "list":
        if not gw.invoices:
            return "no invoices yet — /store invoice create <customer> <amount_kobo>."
        lines = ["🧾 invoices:"]
        for inv in gw.invoices:
            lines.append(
                f"• {inv.id}: {inv.customer} "
                f"{_fmt(inv.amount_minor, inv.currency)} [{inv.status}]")
        return "\n".join(lines)
    if sub == "pay":
        tokens = arg.split()
        if len(tokens) < 2:
            return ("usage: /store invoice pay <invoice_id> "
                    "<payment_reference>")
        invoice = next((i for i in gw.invoices if i.id == tokens[0]), None)
        if invoice is None:
            return f"no invoice {tokens[0]!r}. Use /store invoice list."
        payment = gw.find_payment(tokens[1])
        if payment is None:
            return (f"no payment with reference {tokens[1]!r}. "
                    "Record it first: /store pay record ...")
        gw.mark_paid(invoice, payment)
        store.save_gateway(gw)
        return f"invoice {invoice.id} marked paid by {payment.reference}."
    return _USAGE


def _subscribe(store: CommerceStore, rest: str) -> str:
    parts = (rest or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    arg = parts[1] if len(parts) > 1 else ""
    gw = store.gateway()
    if sub == "create":
        head, fields = _split_pipes(arg)
        tokens = head.split(None, 2)
        if len(tokens) < 3:
            return ("usage: /store subscribe create <provider> <customer> "
                    "<amount_kobo> [| interval]")
        amount = _amount(tokens[2])
        if amount is None or amount <= 0:
            return f"amount must be a positive integer kobo, got {tokens[2]!r}"
        interval = fields[0].lower() if fields and fields[0] else "monthly"
        sub_obj = gw.create_subscription(
            tokens[0], tokens[1], amount, interval=interval)
        store.save_gateway(gw)
        return (f"subscription created: {sub_obj.id} — {tokens[1]} "
                f"{_fmt(amount)} / {interval} via {sub_obj.provider}")
    if sub == "list":
        if not gw.subscriptions:
            return "no subscriptions yet."
        lines = ["🔁 subscriptions:"]
        for s in gw.subscriptions:
            lines.append(
                f"• {s.id}: {s.customer} {_fmt(s.amount_minor, s.currency)} "
                f"/ {s.interval} [{s.status}] via {s.provider}")
        return "\n".join(lines)
    if sub == "cancel":
        sub_id = arg.strip().split(None, 1)[0] if arg.strip() else ""
        sub_obj = next((s for s in gw.subscriptions if s.id == sub_id),
                       None)
        if sub_obj is None:
            return f"no subscription {sub_id!r}. Use /store subscribe list."
        gw.cancel_subscription(sub_obj)
        store.save_gateway(gw)
        return f"subscription {sub_id} cancelled."
    if sub == "dunning":
        try:
            limit = int(arg.strip().split(None, 1)[0]) if arg.strip() else 15
        except ValueError:
            return "usage: /store subscribe dunning [n]"
        attempts = store.list_dunning_attempts(limit=limit)
        if not attempts:
            return ("no dunning attempts yet — the daily dunning tick "
                    "records every retry here.")
        lines = ["🔁 dunning attempts (most recent first):"]
        for a in attempts:
            # timezone.utc, not datetime.UTC: EC2 runs Python <= 3.11.
            when = datetime.fromtimestamp(
                float(a.get("ts") or 0), tz=timezone.utc).strftime(  # noqa: UP017
                    "%Y-%m-%d %H:%M")
            lines.append(
                f"• {when} {a.get('customer')} — {a.get('outcome')} "
                f"(attempt {a.get('attempt')}) via {a.get('provider')}"
                + (f" — {a.get('reason')}" if a.get("reason") else ""))
        return "\n".join(lines)
    return _USAGE


# ── logistics ──────────────────────────────────────────────────────


def _ship(store: CommerceStore, rest: str) -> str:
    parts = (rest or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    arg = parts[1] if len(parts) > 1 else ""
    if sub == "quote":
        head, fields = _split_pipes(arg)
        tokens = head.split()
        if len(tokens) < 2:
            return "usage: /store ship quote <carrier> <weight_kg> [| distance_km]"
        try:
            weight = float(tokens[1])
        except ValueError:
            return f"weight_kg must be a number, got {tokens[1]!r}"
        distance = None
        if fields and fields[0]:
            try:
                distance = float(fields[0])
            except ValueError:
                return f"distance_km must be a number, got {fields[0]!r}"
        quote = rate_quote(tokens[0], origin=_empty_address(),
                           destination=_empty_address(), weight_kg=weight,
                           distance_km=distance)
        return (f"{quote['carrier']}: {_fmt(quote['fee_minor'])} "
                f"for {weight}kg"
                + (f" over {quote['distance_km']}km" if distance else "")
                + " — estimated: True (offline tariff table — planning "
                  "figure, not a live quote)")
    if sub == "eta":
        head, fields = _split_pipes(arg)
        tokens = head.split()
        if not tokens:
            return "usage: /store ship eta <carrier> [| distance_km | same-city]"
        distance = None
        same_city = False
        if fields and fields[0]:
            try:
                distance = float(fields[0])
            except ValueError:
                return f"distance_km must be a number, got {fields[0]!r}"
        if len(fields) > 1 and fields[1].strip().lower() in ("same-city", "same", "yes"):
            same_city = True
        result = _eta(tokens[0], distance_km=distance, same_city=same_city)
        return (f"{result['carrier']}: {result['eta']} — estimated: True "
                "(offline distance bands — planning figure)")
    if sub == "send":
        head, fields = _split_pipes(arg)
        tokens = head.split()
        if len(tokens) < 2:
            return ("usage: /store ship send <carrier> <waybill> "
                    "[| order_ref | recipient]")
        carrier = tokens[0].lower()
        if carrier not in CARRIERS:
            return (f"unknown carrier {tokens[0]!r} — known: "
                    f"{sorted(CARRIERS)}")
        shipment = store.record_shipment({
            "id": "shp_" + uuid.uuid4().hex[:10],
            "carrier": carrier,
            "waybill": tokens[1],
            "order_ref": fields[0] if len(fields) > 0 else "",
            "recipient": fields[1] if len(fields) > 1 else "",
            "status": "recorded",
            "created_at": time.time(),
        })
        return (f"shipment recorded: {shipment['id']} — "
                f"{CARRIERS[carrier].name} waybill {tokens[1]}"
                + (f" (order {shipment['order_ref']})"
                   if shipment["order_ref"] else ""))
    if sub == "track":
        tokens = arg.split()
        if len(tokens) < 2:
            return "usage: /store ship track <carrier> <waybill>"
        result = tracking(tokens[0], tokens[1])
        stored = store.find_shipment(tokens[0], tokens[1])
        lines = [f"{result['carrier']} waybill {tokens[1]}: "
                 f"status {result['status']}"]
        lines.append(f"reason: {result['reason']}")
        if stored is not None:
            detail = ""
            if stored.get("order_ref"):
                detail += f"order {stored.get('order_ref')} "
            if stored.get("recipient"):
                detail += f"for {stored.get('recipient')}"
            lines.append("on record: " + (detail.strip() or "recorded shipment"))
        return "\n".join(lines)
    if sub == "list":
        shipments = store.list_shipments()
        if not shipments:
            return "no shipments recorded — /store ship send <carrier> <waybill>."
        lines = ["📦 shipments:"]
        for s in shipments:
            lines.append(
                f"• {s.get('waybill')}: {s.get('carrier')} "
                f"[{s.get('status', 'recorded')}]"
                + (f" — order {s.get('order_ref')}" if s.get("order_ref") else "")
                + (f" — {s.get('recipient')}" if s.get("recipient") else ""))
        return "\n".join(lines)
    return _USAGE


def _empty_address() -> Any:
    from .logistics import Address

    return Address()


# ── reconciliation ─────────────────────────────────────────────────


def _reconcile(store: CommerceStore, rest: str, *, settings: Any = None,
               ledger: Any = None) -> str:
    sub = (rest or "").strip().split(None, 1)[0].lower() or "status"
    gw = store.gateway()
    if ledger is None:
        ledger = _ledger_for(settings)
    rec = store.reconciler(ledger=ledger)
    if sub == "run":
        done = [e.payment_reference for e in rec.journal]
        paid = {i.id: i for i in gw.invoices if i.status == "paid"}
        used: set[str] = set()
        count = 0
        flagged = 0
        for payment in gw.payments:
            if payment.status != "succeeded":
                continue
            if payment.reference in done:
                continue
            invoice = None
            for inv_id, inv in paid.items():
                if inv_id in used:
                    continue
                if (inv.amount_minor == payment.amount_minor
                        and inv.currency == payment.currency):
                    invoice = inv
                    used.add(inv_id)
                    break
            entry = rec.reconcile(payment, invoice)
            count += 1
            if not entry.invoice_id:
                flagged += 1
        store.save_reconciler(rec)
        if not count:
            return "nothing new to reconcile — every succeeded payment is already in the journal."
        return (f"reconciled {count} payment(s) — balanced: "
                f"{'yes' if rec.balanced else 'NO'}"
                + (f" · {flagged} flagged unreconciled (no matching invoice)"
                   if flagged else ""))
    if sub == "status":
        summary = rec.summary()
        return (
            f"reconciliation — balanced: {'yes' if summary['balanced'] else 'NO'}\n"
            f"gross: {_fmt(summary['gross_minor'])} · "
            f"fees: {_fmt(summary['fees_minor'])} · "
            f"refunds: {_fmt(summary['refunds_minor'])} · "
            f"net: {_fmt(summary['net_minor'])}\n"
            f"entries: {summary['entries']} · "
            f"unreconciled: {summary['unreconciled_count']}")
    if sub == "unreconciled":
        return rec.unreconciled_report()
    return _USAGE


# ── support ────────────────────────────────────────────────────────


def _ticket(store: CommerceStore, rest: str) -> str:
    parts = (rest or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    arg = parts[1] if len(parts) > 1 else ""
    inbox = store.inbox()
    if sub == "open":
        head, fields = _split_pipes(arg)
        tokens = head.split(None, 1)
        if len(tokens) < 2:
            return ("usage: /store ticket open <customer> <message> "
                    "[| priority | channel | phone]")
        priority = fields[0].lower() if fields and fields[0] else "medium"
        if priority not in PRIORITIES:
            return (f"unknown priority {priority!r} — "
                    f"{list(PRIORITIES)}")
        ticket = inbox.open(
            tokens[0], tokens[1],
            channel=fields[1] if len(fields) > 1 else "whatsapp",
            customer_phone=fields[2] if len(fields) > 2 else "",
            priority=priority)
        store.save_inbox(inbox)
        return (f"ticket opened: {ticket.id} — {tokens[0]} "
                f"[{priority}] (SLA "
                f"{int((ticket.sla_deadline() - ticket.created_at) // 3600)}h "
                f"to first response)")
    if sub == "list":
        wanted = (arg.strip().lower() or "open")
        tickets = (inbox.open_tickets() if wanted == "open"
                   else inbox.tickets)
        if not tickets:
            return "no open tickets." if wanted == "open" else "no tickets yet."
        lines = ["🎫 tickets:"]
        for t in tickets:
            lines.append(
                f"• {t.id}: {t.customer or t.customer_phone} "
                f"[{t.priority}] [{t.status}] — {t.message[:60]}")
        return "\n".join(lines)
    if sub == "show":
        ticket = _find_ticket(inbox, arg)
        if ticket is None:
            return f"no ticket {arg.strip()!r}. Use /store ticket list."
        lines = [f"🎫 {ticket.id} [{ticket.status}] [{ticket.priority}]",
                 f"customer: {ticket.customer or ticket.customer_phone}",
                 f"channel: {ticket.channel}",
                 f"message: {ticket.message}"]
        if ticket.assignee:
            lines.append(f"assignee: {ticket.assignee}")
        if ticket.resolution:
            lines.append(f"resolution: {ticket.resolution}")
        for reply in ticket.replies:
            lines.append(f"reply ({reply.get('template', '')}): "
                         f"{reply.get('text', '')}")
        return "\n".join(lines)
    if sub == "assign":
        tokens = arg.split(None, 1)
        if len(tokens) < 2:
            return "usage: /store ticket assign <id> <assignee>"
        ticket = _find_ticket(inbox, tokens[0])
        if ticket is None:
            return f"no ticket {tokens[0]!r}. Use /store ticket list."
        inbox.assign(ticket, tokens[1])
        store.save_inbox(inbox)
        return f"ticket {ticket.id} assigned to {tokens[1]}."
    if sub == "reply":
        tokens = arg.split(None, 1)
        if not tokens:
            return "usage: /store ticket reply <id> [template]"
        ticket = _find_ticket(inbox, tokens[0])
        if ticket is None:
            return f"no ticket {tokens[0]!r}. Use /store ticket list."
        template = tokens[1].strip() if len(tokens) > 1 else "greeting"
        draft = inbox.reply_draft(ticket, template)
        store.save_inbox(inbox)
        return (f"reply draft for {ticket.id} (draft — not sent):\n"
                f"{draft['text']}")
    if sub == "close":
        head, fields = _split_pipes(arg)
        ticket = _find_ticket(inbox, head.strip())
        if ticket is None:
            return f"no ticket {head.strip()!r}. Use /store ticket list."
        inbox.close(ticket, resolution=fields[0] if fields else "")
        store.save_inbox(inbox)
        return f"ticket {ticket.id} closed."
    if sub == "stats":
        stats = inbox.stats()
        return (
            f"support — total: {stats['total']} · open: {stats['open']} · "
            f"closed: {stats['closed']} · sla breached: "
            f"{stats['sla_breached']} · avg first response: "
            f"{stats['avg_first_response_hours']}h")
    return _USAGE


def _find_ticket(inbox: Any, ref: str) -> Any | None:
    ref = (ref or "").strip()
    ticket = inbox.get(ref)
    if ticket is not None:
        return ticket
    matches = [t for t in inbox.tickets if t.id.startswith(ref)]
    return matches[0] if len(matches) == 1 else None


# ── analytics + tax ────────────────────────────────────────────────


def _report(store: CommerceStore, rest: str, *, settings: Any = None,
            ledger: Any = None) -> str:
    arg = (rest or "").strip()
    if arg:
        try:
            year_s, month_s = arg.split("-", 1)
            year, month = int(year_s), int(month_s)
        except ValueError:
            return "usage: /store report [YYYY-MM]"
        if not 1 <= month <= 12:
            return "usage: /store report [YYYY-MM]"
    else:
        # timezone.utc, not datetime.UTC: EC2 runs Python <= 3.11.
        now = datetime.now(timezone.utc)  # noqa: UP017
        year, month = now.year, now.month
    if ledger is None:
        ledger = _ledger_for(settings)
    gw = store.gateway()
    return monthly_client_report(ledger, year, month,
                                 payments=gw.payments)


def _tax(rest: str) -> str:
    tokens = (rest or "").strip().split()
    if not tokens:
        return "usage: /store tax <amount_kobo> [jurisdiction]"
    amount = _amount(tokens[0])
    if amount is None:
        return f"amount must be a non-negative integer kobo, got {tokens[0]!r}"
    jurisdiction = tokens[1].upper() if len(tokens) > 1 else "NG"
    result = compute_tax(amount, jurisdiction)
    return (
        f"{result['jurisdiction_name']} {result['tax_name']} "
        f"{result['rate_pct']}% on {_fmt(result['net_minor'])}: "
        f"tax {_fmt(result['tax_minor'])} · "
        f"total {_fmt(result['total_minor'])}")
