"""Commerce, wired into production.

Audit gap R1-5: eight commerce modules (payments, logistics,
reconciliation, support, analytics, catalog, storefront, tax) existed
and were unit-pinned, but nothing in production reached them — only
medusa.py + cart_recovery.py were live via /store. Every test here
drives the PRODUCTION wiring: the durable CommerceStore under the
home's data/ directory and the /store sub-verbs — never the dormant
organ in isolation.

Teeth: on the pristine base each test fails for the missing wiring
(no CommerceStore, no commerce sub-verbs — /store falls through to
"no store 'products'").

Money is integer minor units (kobo) everywhere. A live charge needs
the provider's real secret key and a real HTTP round-trip — without
either it refuses in words, never a fabricated success. Quotes and
ETAs are labelled estimates from the offline tariff table.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nomorals.commerce.commands import COMMERCE_VERBS, handle_store_commerce
from nomorals.commerce.store import CommerceStore, default_commerce_path
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    caller_may_run,
)


def _ctx(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))


def _store(tmp_path):
    return CommerceStore(default_commerce_path(tmp_path))


def _run(tmp_path, tail, **kwargs):
    return handle_store_commerce(
        tail, home=Path(tmp_path), settings=_ctx(tmp_path).settings,
        **kwargs)


def _mem_host(tmp_path):
    """Production control host on a throwaway home."""
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin

    host = RuntimeMemoryMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


# ── durable store ───────────────────────────────────────────────────


def test_store_persists_across_instances(tmp_path):
    store = _store(tmp_path)
    catalog = store.catalog()
    catalog.upsert(
        {"id": "yam-1", "name": "Yam tuber", "price_minor": 250000,
         "currency": "NGN", "stock": 12})
    store.save_catalog(catalog)
    gw = store.gateway()
    gw.create_invoice("Ada", 250000)
    store.save_gateway(gw)
    inbox = store.inbox()
    inbox.open("Ada", "where is my yam?", priority="high")
    store.save_inbox(inbox)

    fresh = _store(tmp_path)
    assert [p["name"] for p in fresh.catalog().list()] == ["Yam tuber"]
    assert len(fresh.gateway().invoices) == 1
    assert len(fresh.inbox().tickets) == 1
    assert (tmp_path / "data" / "commerce.json").exists()


def test_store_corrupt_file_quarantined_and_raises(tmp_path):
    path = default_commerce_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    try:
        CommerceStore(path)
    except ValueError as exc:
        assert "quarantined" in str(exc)
    else:
        raise AssertionError("corrupt store must raise")
    assert not path.exists()
    quarantined = list(path.parent.glob("commerce.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"
    # A bad row inside valid JSON is skipped, not fatal.
    path.write_text(json.dumps(
        {"products": [{"id": "bad"},
                      {"id": "ok", "name": "Good", "price_minor": 100}],
         "payments": [], "invoices": [], "subscriptions": [],
         "disputes": [], "tickets": [], "shipments": [],
         "journal": [], "unreconciled": []}), encoding="utf-8")
    store = CommerceStore(path)
    assert [p["id"] for p in store.catalog().list()] == ["ok"]


# ── catalog + storefront ────────────────────────────────────────────


def test_products_add_list_feed_remove(tmp_path):
    out = _run(tmp_path, "products add yam-1 Yam tuber 250000")
    assert out.startswith("product added: Yam tuber"), out
    out = _run(tmp_path,
               "products add yam-1 Yam tuber 250000 | NGN | food | 12")
    assert "product added" in out
    listed = _run(tmp_path, "products list")
    assert "Yam tuber" in listed and "yam-1" in listed
    assert "₦2,500" in listed
    feed = _run(tmp_path, "products feed json")
    assert "yam-1" in feed
    csv_feed = _run(tmp_path, "products feed csv")
    assert "id,name,price_minor" in csv_feed
    assert "yam-1" in csv_feed
    out = _run(tmp_path, "products remove yam-1")
    assert "removed product: yam-1" in out
    assert "no products yet" in _run(tmp_path, "products list")
    assert "no product" in _run(tmp_path, "products remove nope")
    assert "usage:" in _run(tmp_path, "products add yam-1")
    assert "usage:" in _run(tmp_path, "products frobnicate")


def test_page_brief_only_no_fake_image(tmp_path):
    _run(tmp_path, "products add yam-1 Yam tuber 250000")
    out = _run(tmp_path, "page yam-1")
    assert "Yam tuber" in out
    assert "₦2,500" in out
    # Brief-only: no image is claimed to exist.
    assert "brief only" in out
    assert "no image" in out
    assert "no product" in _run(tmp_path, "page nope")


# ── payments + invoices + reconciliation ────────────────────────────


def test_pay_providers_and_quote(tmp_path):
    out = _run(tmp_path, "pay providers")
    for name in ("Paystack", "Flutterwave", "Monnify", "M-Pesa", "Stripe"):
        assert name in out, out
    out = _run(tmp_path, "pay quote paystack 100000")
    # 1.5% + ₦100 → 1500 + 10000 = ₦115 fee, net ₦885.
    assert "₦115" in out and "₦885" in out, out
    assert "unknown provider" in _run(tmp_path, "pay quote nope 1000")


def test_pay_charge_refuses_without_key(tmp_path):
    # No provider key on this box and no HTTP client is faked in:
    # a live charge must refuse in words, never fake success.
    out = _run(tmp_path, "pay charge paystack 100000")
    assert "PAYSTACK_SECRET_KEY" in out, out
    assert "succeeded" not in out


def test_pay_record_invoice_reconcile_report(tmp_path):
    out = _run(tmp_path, "pay record paystack 250000 ref-001 | Ada")
    assert "recorded payment" in out and "ref-001" in out, out
    assert "₦2,500" in out
    listed = _run(tmp_path, "pay list")
    assert "ref-001" in listed and "succeeded" in listed
    # Duplicate reference is refused — no double-counting.
    assert "already recorded" in _run(tmp_path,
                                      "pay record paystack 250000 ref-001")
    inv = _run(tmp_path, "invoice create Ada 250000")
    assert inv.startswith("invoice created: inv_"), inv
    inv_id = inv.split(":")[1].strip().split()[0]
    out = _run(tmp_path, f"invoice pay {inv_id} ref-001")
    assert "marked paid" in out, out
    out = _run(tmp_path, "reconcile run")
    assert "reconciled 1 payment" in out, out
    assert "balanced" in out
    status = _run(tmp_path, "reconcile status")
    assert "balanced: yes" in status
    assert "₦2,500" in status  # gross
    report = _run(tmp_path, "report")
    assert "₦2,500" in report
    assert "Gross revenue" in report
    # The ledger really got the income row.
    ledger_path = tmp_path / "finance" / "ledger.jsonl"
    assert ledger_path.exists()
    assert "ref-001" in ledger_path.read_text(encoding="utf-8")


def test_reconcile_unmatched_payment_flagged(tmp_path):
    _run(tmp_path, "pay record flutterwave 99000 ref-002 | Bola")
    out = _run(tmp_path, "reconcile run")
    assert "reconciled 1 payment" in out
    assert "unreconciled" in _run(tmp_path, "reconcile status")
    assert "no matching invoice" in _run(
        tmp_path, "reconcile unreconciled")


def test_subscriptions_and_disputes(tmp_path, monkeypatch):
    # Subscription creation needs the provider key (the organ's own
    # honest rule) — a test key is enough, since creating a
    # subscription moves no money and makes no HTTP call.
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "test-key")
    out = _run(tmp_path,
               "subscribe create paystack Ada 500000 | monthly")
    assert out.startswith("subscription created: sub_"), out
    sub_id = out.split(":")[1].strip().split()[0]
    assert "Ada" in _run(tmp_path, "subscribe list")
    out = _run(tmp_path, f"subscribe cancel {sub_id}")
    assert "cancelled" in out
    _run(tmp_path, "pay record paystack 500000 ref-003 | Ada")
    out = _run(tmp_path, "pay dispute ref-003 500000 charged twice")
    assert out.startswith("dispute recorded: dsp_"), out
    dsp_id = out.split(":")[1].strip().split()[0]
    assert "charged twice" in _run(tmp_path, "pay disputes")
    out = _run(tmp_path, f"pay resolve {dsp_id} won | bank confirmed")
    assert "won" in out
    assert "no dispute" in _run(tmp_path, "pay resolve nope won")


# ── tax ─────────────────────────────────────────────────────────────


def test_tax_nigeria_vat_750bps(tmp_path):
    out = _run(tmp_path, "tax 100000")
    # Nigeria VAT 7.5% on ₦1,000 → ₦75 tax, ₦1,075 total.
    assert "7.5%" in out and "₦75" in out and "₦1,075" in out, out
    assert "Nigeria" in out
    out = _run(tmp_path, "tax 100000 GH")
    assert "15.0%" in out and "₦150" in out, out
    assert "unknown jurisdiction" in _run(tmp_path, "tax 1000 XX")


# ── logistics ───────────────────────────────────────────────────────


def test_ship_quote_eta_send_track(tmp_path):
    out = _run(tmp_path, "ship quote gig 2")
    assert "GIG Logistics" in out
    assert "estimated" in out  # offline tariff table, labelled
    assert "not a live quote" in out
    out = _run(tmp_path, "ship eta kwik")
    assert "2–6 hours" in out
    out = _run(tmp_path, "ship send gig GIG123456 | order-9 | Ada")
    assert "shipment recorded" in out and "GIG123456" in out, out
    out = _run(tmp_path, "ship track gig GIG123456")
    # Honest: no live tracker wired, so no invented location/status.
    assert "not_available" in out
    assert "GIG123456" in out
    assert "Ada" in _run(tmp_path, "ship list")
    assert "unknown carrier" in _run(tmp_path, "ship quote nope 1")


# ── support ─────────────────────────────────────────────────────────


def test_ticket_open_reply_close(tmp_path):
    out = _run(tmp_path,
               "ticket open Ada my yam never arrived | urgent | whatsapp")
    assert out.startswith("ticket opened: tkt_"), out
    tkt_id = out.split(":")[1].strip().split()[0]
    assert "Ada" in _run(tmp_path, "ticket list")
    out = _run(tmp_path, f"ticket reply {tkt_id} greeting")
    assert "draft" in out
    assert "Ada" in out  # the draft is addressed to her
    out = _run(tmp_path, f"ticket close {tkt_id} | replaced the yam")
    assert "closed" in out
    stats = _run(tmp_path, "ticket stats")
    assert "total: 1" in stats and "closed: 1" in stats
    assert "no ticket" in _run(tmp_path, "ticket show nope")
    assert "usage:" in _run(tmp_path, "ticket open Ada")


# ── tool family: same organs, agent surface ─────────────────────────


def _tool_call(monkeypatch, tmp_path, **kwargs):
    """Call the registered `commerce` tool with home redirected to tmp."""
    monkeypatch.setattr(
        "os.path.expanduser",
        lambda p: str(tmp_path) if p.startswith("~") else p)
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry().register_builtins()
    outcome = registry.call("commerce", **kwargs)
    assert outcome.ok, outcome.error
    return outcome.value


def test_tool_tax_and_products(tmp_path, monkeypatch):
    out = _tool_call(monkeypatch, tmp_path, action="tax",
                     amount_minor=100000)
    assert out["ok"] is True
    assert "7.5%" in out["reply"] and "₦75" in out["reply"], out
    out = _tool_call(monkeypatch, tmp_path, action="products", sub="add",
                     product_id="yam-1", name="Yam tuber",
                     price_minor=250000)
    assert out["ok"] is True
    assert "product added: Yam tuber" in out["reply"], out
    out = _tool_call(monkeypatch, tmp_path, action="products", sub="list")
    assert "yam-1" in out["reply"]
    assert (tmp_path / "data" / "commerce.json").exists()


# ── /store control: dispatch + registry ─────────────────────────────


def test_store_dispatch_reaches_commerce_verbs(tmp_path):
    host = _mem_host(tmp_path)
    assert "Yam tuber" in host._control_store(
        "products add yam-1 Yam tuber 250000")
    assert "yam-1" in host._control_store("products list")
    assert "7.5%" in host._control_store("tax 100000")
    # Existing verbs are byte-identical: list still lists stores.
    assert "no stores yet" in host._control_store("list")
    assert "no store 'nope'" in host._control_store("nope do a thing")


def test_store_commerce_verbs_registered_owner_only():
    assert "store" in CONTROL_COMMANDS
    assert "store" not in PUBLIC_COMMANDS
    assert not caller_may_run("store", is_owner=False)
    assert caller_may_run("store", is_owner=True)
    assert {
        "products", "page", "pay", "invoice", "subscribe", "ship",
        "reconcile", "ticket", "report", "tax",
    } == COMMERCE_VERBS
