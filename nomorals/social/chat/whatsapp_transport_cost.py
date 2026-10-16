"""Per-transport WhatsApp cost: Cloud API vs Baileys.

The two transports bill nothing alike, and picking blind is how a bot
burns money (or an account):

* **Cloud API (Meta)** — billed per *conversation* (a 24h window opened
  by the first message), by category: service / utility / authentication
  / marketing. Rates come from :mod:`nomorals.social.whatsapp_cost`
  (``RATE_CARD_2026`` + the naira constants) — this module never
  hardcodes a second price list. Within an open 24h window, extra
  messages are free; the cost lever is *how many windows you open*, not
  how many messages you send.
* **Baileys (personal account)** — no per-message Meta fee (the phone
  number + data you already pay for). The cost is *account risk*: bulk
  or cold-outreach behaviour gets the number banned (see
  :mod:`baileys_guard`). Marginal cash cost ≈ ₦0, risk cost = high if
  misused.

:func:`estimate` prices a planned send on each transport; :func:`compare`
says which transport to use and why. All money figures come from the
single rate-card source.
"""

from __future__ import annotations

from typing import Any

from ...social.whatsapp_cost import RATE_CARD_2026, RATE_SERVICE_KOBO, naira

__all__ = ["estimate", "compare", "CLOUD_CATEGORIES"]

#: Cloud API categories (Meta's four).
CLOUD_CATEGORIES: tuple[str, ...] = ("service", "utility", "authentication", "marketing")

#: Approx. USD→NGN used only for the Cloud-API naira display. The kobo
#: constants in whatsapp_cost are the source of truth for service;
#: this converts the USD card for the other categories.
_USD_TO_NGN = 1500.0


def _cloud_conversation_cost_kobo(category: str) -> int:
    """One Cloud-API conversation in kobo, from the single rate card."""
    cat = (category or "service").strip().lower()
    if cat == "service":
        return int(RATE_SERVICE_KOBO)
    info = RATE_CARD_2026.get(cat)
    if info is None:
        raise ValueError(f"unknown WhatsApp category {category!r} — one of {CLOUD_CATEGORIES}")
    return int(round(float(info["usd"]) * _USD_TO_NGN * 100))


def estimate(*, transport: str, category: str = "service",
             conversations: int = 1, messages: int = 1) -> dict[str, Any]:
    """Price a planned send on one transport.

    ``conversations`` = distinct 24h windows opened (Cloud API bills
    per conversation, not per message). ``messages`` is informational
    for Cloud (extra messages inside a window are free) and the pacing
    input for Baileys.
    """
    t = (transport or "").strip().lower()
    conv = max(0, int(conversations))
    msgs = max(0, int(messages))
    if t in ("cloud", "cloud-api", "meta"):
        cat = (category or "service").strip().lower()
        if cat not in CLOUD_CATEGORIES:
            raise ValueError(f"unknown Cloud category {category!r}")
        per = _cloud_conversation_cost_kobo(cat)
        total = per * conv
        return {"transport": "cloud-api", "category": cat,
                "conversations": conv, "messages": msgs,
                "per_conversation_kobo": per, "total_kobo": total,
                "total": naira(total),
                "note": ("Cloud bills per 24h conversation, not per message — "
                         "extra messages inside an open window are free. "
                         "Service replies inside the 1,000/month free allowance "
                         "cost nothing; this estimate is the beyond-allowance price.")}
    if t in ("baileys", "personal", "bridge"):
        return {"transport": "baileys", "category": category,
                "conversations": conv, "messages": msgs,
                "per_conversation_kobo": 0, "total_kobo": 0,
                "total": naira(0),
                "note": ("No per-message Meta fee — but the cost is account "
                         "risk: bulk/cold outreach gets the number banned. "
                         "Use Baileys for warm, paced, personal traffic only "
                         "(see baileys_guard).")}
    raise ValueError(f"unknown WhatsApp transport {transport!r} — 'cloud-api' or 'baileys'")


def compare(*, category: str = "service", conversations: int = 1,
            messages: int = 1, cold_outreach: bool = False) -> dict[str, Any]:
    """Recommend a transport for this send, with priced reasoning."""
    cloud = estimate(transport="cloud-api", category=category,
                     conversations=conversations, messages=messages)
    baileys = estimate(transport="baileys", category=category,
                       conversations=conversations, messages=messages)
    if cold_outreach:
        pick, why = ("cloud-api",
                     "cold/bulk outreach over Baileys risks a number ban — "
                     "the ban costs more than any per-conversation fee. "
                     "Cloud API with an approved template is the safe route.")
    elif category == "marketing":
        pick, why = ("cloud-api",
                     "marketing needs an approved template + opt-in; "
                     "Baileys marketing blasts are the fastest way to lose a number.")
    else:
        pick, why = ("baileys",
                     "warm, personal traffic: no per-conversation fee, "
                     "and the account is already warmed for it.")
    return {"recommendation": pick, "why": why,
            "cloud": cloud, "baileys": baileys}
