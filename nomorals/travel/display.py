"""Travel display rules — budget-in-first-reply, points-vs-cash, multi-origin.

Build-map #75. These are OUTPUT-FORMAT RULES, not new modules.
Every travel output shows total cost from message one — never a
dead-end list without prices (Wonderplan/SearchSpot pattern).

Three rules:
1. Budget-in-first-reply: every itinerary/offer list carries the
   total and, when a budget is known, where it lands against it.
2. Points-vs-cash: loyalty comparison in the same view
   ("₦450k cash or 65k miles + ₦80k").
3. Multi-origin fallback: alternative-airport chains (LOS → ABV)
   checked in every flight query.

All functions are pure formatting — never raise, never do I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "format_with_budget",
    "points_vs_cash",
    "multi_origin_search",
    "LoyaltyProgram",
    "enrich_offers_text",
    "enrich_alert_text",
    "enrich_itinerary_text",
]


def _naira(kobo: int) -> str:
    """₦450,000 — full precision for budgets."""
    return f"₦{kobo / 100:,.0f}"


def _short(kobo: int) -> str:
    """₦450k — compact for inline lists."""
    n = kobo / 100
    if n >= 1_000_000:
        return f"₦{n / 1_000_000:.1f}m"
    if n >= 1_000:
        return f"₦{n / 1_000:.0f}k"
    return f"₦{n:,.0f}"


def format_with_budget(
    body: str,
    total_kobo: int,
    *,
    budget_kobo: int | None = None,
) -> str:
    """Prepend the cost summary to any travel output.

    The total always comes first — never a price-less list.
    When a budget is known, say where the total lands against it.
    """
    lines = [f"💰 Total: {_naira(total_kobo)}"]
    if budget_kobo:
        if total_kobo <= budget_kobo:
            lines.append(f"   within your {_short(budget_kobo)} budget ✅")
        else:
            over = total_kobo - budget_kobo
            lines.append(f"   over budget by {_naira(over)} ⚠️")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


@dataclass
class LoyaltyProgram:
    """One of the owner's loyalty programs (manual config to start).

    ``value_per_point_kobo``: what a point is worth in kobo when
    redeemed for flights (e.g. 100 = ₦1/point). ``balance``: points
    held. ``topup_per_point_kobo``: cash cost of the co-pay leg.
    """

    name: str
    balance: int = 0
    value_per_point_kobo: float = 100.0
    # Typical award shape: points + cash co-pay for this route.
    award_points: int = 0
    award_cash_kobo: int = 0


def points_vs_cash(cash_kobo: int, programs: list[LoyaltyProgram]) -> str:
    """Loyalty comparison in the same view as the cash price.

    "₦450k cash or 65k miles + ₦80k." Only programs with enough
    balance for the award are shown. Empty programs → "".
    """
    options: list[str] = []
    for p in programs:
        if p.award_points <= 0 or p.balance < p.award_points:
            continue
        cash_bit = f" + {_short(p.award_cash_kobo)}" if p.award_cash_kobo else ""
        options.append(f"{p.award_points:,} {p.name}{cash_bit}")
    if not options:
        return ""
    return f"{_short(cash_kobo)} cash or " + " / ".join(options)


def multi_origin_search(
    origins: list[str],
    destination: str,
    search_fn: Callable[[str, str], list[Any]],
    *,
    amount_of: Callable[[Any], int] | None = None,
) -> dict[str, Any]:
    """Check every origin airport, return the cheapest per origin.

    ``search_fn(origin, destination)`` returns offers; ``amount_of``
    extracts minor units from an offer (defaults to Duffel Offer
    shape: total_amount/total_currency). Never raises — a failed
    origin is reported, not fatal.
    """
    results: dict[str, Any] = {"checked": [], "cheapest": None}
    best: tuple[str, int, Any] | None = None
    for origin in origins:
        origin = (origin or "").upper().strip()
        if not origin:
            continue
        try:
            offers = search_fn(origin, destination) or []
        except Exception:  # noqa: BLE001 — one airport down ≠ all down
            results["checked"].append({"origin": origin, "status": "failed"})
            continue
        cheapest_kobo = 0
        cheapest_offer = None
        for o in offers:
            try:
                kobo = amount_of(o) if amount_of else _offer_minor(o)
            except Exception:  # noqa: BLE001
                continue
            if kobo > 0 and (cheapest_kobo == 0 or kobo < cheapest_kobo):
                cheapest_kobo = kobo
                cheapest_offer = o
        results["checked"].append({
            "origin": origin,
            "status": "ok" if cheapest_kobo else "no_offers",
            "cheapest_kobo": cheapest_kobo,
        })
        if cheapest_kobo and (best is None or cheapest_kobo < best[1]):
            best = (origin, cheapest_kobo, cheapest_offer)
    if best:
        results["cheapest"] = {
            "origin": best[0], "kobo": best[1], "offer": best[2],
        }
    return results


def _offer_minor(offer: Any) -> int:
    """Minor units from a Duffel-shaped Offer (or dict)."""
    if isinstance(offer, dict):
        amount = offer.get("total_amount", "0")
        currency = offer.get("total_currency", "NGN")
    else:
        amount = getattr(offer, "total_amount", "0")
        currency = getattr(offer, "total_currency", "NGN")
    try:
        return int(round(float(amount) * 100))
    except (TypeError, ValueError):
        return 0


def format_multi_origin(results: dict[str, Any], primary: str) -> str:
    """'also checked ABV: ₦380k (save ₦70k).'"""
    cheapest = results.get("cheapest")
    lines: list[str] = []
    for c in results.get("checked", []):
        origin = c["origin"]
        if origin == primary:
            continue
        if c["status"] != "ok" or not c["cheapest_kobo"]:
            continue
        bit = f"also checked {origin}: {_short(c['cheapest_kobo'])}"
        if cheapest and origin != cheapest["origin"]:
            save = c["cheapest_kobo"] - cheapest["kobo"]
            if save > 0:
                bit += f" (save {_short(save)} via {cheapest['origin']})"
        lines.append(bit)
    if cheapest and cheapest["origin"] != primary:
        lines.append(
            f"cheapest is {cheapest['origin']} at "
            f"{_short(cheapest['kobo'])}"
        )
    return "\n".join(lines)


# ── wiring into #70 / #71 / #72 ─────────────────────────────────────────────


def enrich_offers_text(
    lines: list[str],
    offers: list[Any],
    *,
    budget_kobo: int | None = None,
    programs: list[LoyaltyProgram] | None = None,
    origins: list[str] | None = None,
    destination: str = "",
    search_fn: Callable[[str, str], list[Any]] | None = None,
) -> str:
    """#70 offer lists: every line keeps its price; header gets the
    budget line; footer gets points-vs-cash and multi-origin."""
    body = "\n".join(lines)
    total_kobo = 0
    for o in offers:
        kobo = _offer_minor(o)
        if kobo > 0 and (total_kobo == 0 or kobo < total_kobo):
            total_kobo = kobo
    out = format_with_budget(body, total_kobo, budget_kobo=budget_kobo)
    extras: list[str] = []
    if programs and total_kobo:
        pvc = points_vs_cash(total_kobo, programs)
        if pvc:
            extras.append(f"🎖️ {pvc}")
    if origins and len(origins) > 1 and search_fn and destination:
        mo = multi_origin_search(origins, destination, search_fn)
        mot = format_multi_origin(mo, origins[0])
        if mot:
            extras.append(f"🛫 {mot}")
    if extras:
        out += "\n\n" + "\n".join(extras)
    return out


def enrich_alert_text(
    alert_text: str,
    current_kobo: int,
    *,
    budget_kobo: int | None = None,
    programs: list[LoyaltyProgram] | None = None,
) -> str:
    """#71 price alerts: the drop already has prices; add the budget
    line and points alternative."""
    out = alert_text
    bits: list[str] = []
    if budget_kobo:
        if current_kobo <= budget_kobo:
            bits.append(f"within your {_short(budget_kobo)} budget ✅")
        else:
            bits.append(
                f"over budget by {_naira(current_kobo - budget_kobo)} ⚠️")
    if programs:
        pvc = points_vs_cash(current_kobo, programs)
        if pvc:
            bits.append(f"🎖️ {pvc}")
    if bits:
        out += "\n" + "\n".join(bits)
    return out


def enrich_itinerary_text(
    summary_text: str,
    total_kobo: int,
    *,
    budget_kobo: int | None = None,
    programs: list[LoyaltyProgram] | None = None,
) -> str:
    """#72 itinerary summaries: cost summary first, points after."""
    out = format_with_budget(summary_text, total_kobo,
                             budget_kobo=budget_kobo)
    if programs and total_kobo:
        pvc = points_vs_cash(total_kobo, programs)
        if pvc:
            out += f"\n\n🎖️ {pvc}"
    return out
