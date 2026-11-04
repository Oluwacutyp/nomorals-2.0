"""Devon's own pairing page (§15 slice 2).

The in-chat ``/pair`` flow is the primary surface; this page is the
same state rendered for a browser — the pool header the reference
bots show (Registered · Connected), every account with its status
and personal-surface toggles, any live WhatsApp pairing code, and
any Telegram flow waiting on its login code.  It follows the
codebase's page precedent (``wa_provisioner.provisioning_page_html``):
a pure render of current state, no secrets beyond the pairing codes
the owner himself just requested, everything HTML-escaped.

Serving: the API server serves ``GET /pair`` as a special HTML
path (the same precedent as its SSE paths, which also cannot ride
the JSON dispatch), gated on the ``sys.config`` capability — in
practice the owner principal, since the bounded default grant does
not carry it.  The page shows live pairing codes, which are
credentials while they live; anyone less than the owner gets a
403.  When the API is not running, ``/pair`` in chat still carries
the whole flow and this module renders the identical state.
"""
from __future__ import annotations

import html
import time
from typing import Any

from .pairing import PERSONAL_TOGGLES, PairingRegistry


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def pairing_page_html(registry: PairingRegistry, *,
                      codes: dict[str, tuple[str, float]] | None = None,
                      flows: dict[str, dict[str, Any]] | None = None,
                      counts: dict[str, int] | None = None,
                      now: float | None = None) -> str:
    """Render the pool page from live state.

    ``codes``: account id → (pairing code, age seconds) for WhatsApp
    accounts with a live code.  ``flows``: account id → pending
    Telegram flow state (``TelegramPairer.pending``).  ``counts``:
    the pool header; computed from the registry when not supplied
    (without probing — the page must render even when a bridge is
    down, indeed especially then)."""
    now = time.time() if now is None else now
    codes = codes or {}
    flows = flows or {}
    rows = registry.all()
    if counts is None:
        counts = {
            "registered": len(rows),
            "connected": sum(1 for r in rows if r.status == "linked"),
            "dropped": sum(1 for r in rows if r.status == "down"),
        }
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,"
        "initial-scale=1'>"
        "<title>Devon — paired accounts</title></head><body>",
        "<h1>📟 Paired accounts</h1>",
        f"<p>Registered: <b>{counts.get('registered', 0)}</b> · "
        f"Connected: <b>{counts.get('connected', 0)}</b> · "
        f"Dropped: <b>{counts.get('dropped', 0)}</b></p>",
        "<p>Pairing is approved on the phone, always: WhatsApp → "
        "Linked Devices → Link with phone number, or the login code "
        "Telegram sends the account itself. A web page can show a "
        "code — it can never approve one.</p>",
    ]
    if not rows:
        parts.append(
            "<p>No accounts paired yet. In Devon's DM: "
            "<code>/pair whatsapp &lt;number&gt;</code> or "
            "<code>/pair telegram &lt;number&gt;</code>.</p>")
    for rec in rows:
        label = f" “{_esc(rec.label)}”" if rec.label else ""
        parts.append(
            f"<h2>{_esc(rec.id)}{label}</h2>"
            f"<p>{_esc(rec.platform)} · status: "
            f"<b>{_esc(rec.status)}</b>"
            + (f" · primary for {_esc(rec.platform)}"
               if rec.primary else "")
            + "</p>")
        toggles = ", ".join(
            f"{name} {'on' if rec.toggle(name) else 'off'}"
            for name in PERSONAL_TOGGLES)
        parts.append(f"<p>personal surfaces: {_esc(toggles)}</p>")
        code = codes.get(rec.id)
        if code:
            parts.append(
                f"<p>pairing code: <b>{_esc(code[0])}</b> "
                f"({int(code[1])}s old — type it into Linked "
                "Devices now, codes expire in minutes)</p>")
        flow = flows.get(rec.id)
        if flow:
            parts.append(
                f"<p>Telegram login code sent — answer in chat: "
                f"<code>/pair tgcode {_esc(rec.id)} "
                f"&lt;code&gt;</code> (flow "
                f"{_esc(flow.get('state', ''))}, "
                f"{int(flow.get('expires_in_s', 0))}s left)</p>")
    parts.append(
        "<p>Manage the pool in Devon's DM with <code>/pair</code>. "
        "This page never pairs anything by itself.</p>"
        "</body></html>")
    return "".join(parts)


def pairing_page_for_context(context: Any) -> str:
    """Render the page for an API/runtime context: build the
    production supervisor + pairer from the context's workspace and
    chat settings, read the live codes and pending flows, render."""
    from pathlib import Path

    from .pairing import default_supervisor
    from .telegram_pairing import TelegramPairer

    settings = getattr(context, "settings", None)
    ws = getattr(settings, "workspace_dir", ".")
    sup = default_supervisor(ws)
    chat = getattr(settings, "chat", None)
    pairer = TelegramPairer(
        sup.registry,
        sessions_root=Path(ws) / "data" / "telegram_sessions",
        api_id=getattr(chat, "telegram_api_id", ""),
        api_hash=getattr(chat, "telegram_api_hash", ""))
    codes: dict[str, tuple[str, float]] = {}
    flows: dict[str, dict[str, Any]] = {}
    for rec in sup.registry.all():
        if rec.platform == "whatsapp":
            got = sup.pairing_code(rec.id)
            if got:
                codes[rec.id] = got
        elif rec.platform == "telegram":
            pending = pairer.pending(rec.id)
            if pending:
                flows[rec.id] = pending
    return pairing_page_html(
        sup.registry, codes=codes, flows=flows,
        counts=sup.counts(probe=False))
