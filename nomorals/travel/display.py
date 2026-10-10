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
    "cpp",
    "rate_redemption",
    "best_redemption",
    "TRANSFER_PARTNERS",
    "bank_partners",
    "OutputTheme",
    "THEMES",
    "themed",
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
    balance for the award are shown. Every option carries its
    cents-per-point so you can see if it's actually a good deal
    (point.me/pointsyeah rule: ≥1.5¢ is good, ≥2.0¢ is great).
    Empty programs → "".
    """
    options: list[str] = []
    for p in programs:
        if p.award_points <= 0 or p.balance < p.award_points:
            continue
        cash_bit = f" + {_short(p.award_cash_kobo)}" if p.award_cash_kobo else ""
        value = cpp(cash_kobo, p.award_points, p.award_cash_kobo)
        star = _cpp_star(value)
        options.append(f"{p.award_points:,} {p.name}{cash_bit}"
                       f" ({value:.0f} kobo/pt{star})")
    if not options:
        return ""
    return f"{_short(cash_kobo)} cash or " + " / ".join(options)


# ── award valuation (point.me / pointsyeah gold) ────────────────────────────

def cpp(cash_kobo: int, award_points: int, award_cash_kobo: int = 0) -> float:
    """Kobo-per-point: (cash − taxes/fees) / points. Pure.

    The naira-native analog of the US cents-per-point rule — the single
    number that says whether a redemption is actually worth it.
    """
    if award_points <= 0:
        return 0.0
    net_kobo = max(0, cash_kobo - award_cash_kobo)
    return net_kobo / award_points


def rate_redemption(value_kobo_pt: float) -> str:
    """great ≥ ₦5/pt · good ≥ ₦3/pt · fair ≥ ₦1.50/pt · poor below."""
    if value_kobo_pt >= 500:
        return "great"
    if value_kobo_pt >= 300:
        return "good"
    if value_kobo_pt >= 150:
        return "fair"
    return "poor"


def _cpp_star(value_cpp: float) -> str:
    return {"great": " ⭐", "good": " ✅"}.get(rate_redemption(value_cpp), "")


#: Transferable bank currencies → airline programs (curated subset).
#: Lets the recommendation name a *move*, not just a program.
TRANSFER_PARTNERS: dict[str, list[str]] = {
    "Amex Membership Rewards": ["Flying Blue", "Virgin Atlantic",
                                 "British Airways Avios", "Delta SkyMiles",
                                 "Etihad Guest"],
    "Chase Ultimate Rewards": ["United MileagePlus", "British Airways Avios",
                               "Virgin Atlantic", "Flying Blue"],
    "Capital One miles": ["Flying Blue", "Virgin Atlantic",
                          "British Airways Avios", "Etihad Guest"],
    "Citi ThankYou": ["Virgin Atlantic", "Flying Blue", "Qatar Privilege"],
}


def bank_partners(program_name: str) -> list[str]:
    """Banks whose points transfer into this program. Pure."""
    want = (program_name or "").lower()
    return [bank for bank, progs in TRANSFER_PARTNERS.items()
            if any(want in p.lower() or p.lower() in want for p in progs)]


def best_redemption(cash_kobo: int,
                    programs: list[LoyaltyProgram]) -> dict[str, Any]:
    """The single best award across programs — with the point.me rule.

    Returns {} when nothing is redeemable. Includes the transfer warning:
    never move points speculatively — confirm the award seat first.
    """
    best: dict[str, Any] | None = None
    for p in programs:
        if p.award_points <= 0 or p.balance < p.award_points:
            continue
        value = cpp(cash_kobo, p.award_points, p.award_cash_kobo)
        if best is None or value > best["cpp"]:
            best = {"program": p.name, "points": p.award_points,
                    "cash_kobo": p.award_cash_kobo, "cpp": round(value, 2),
                    "rating": rate_redemption(value)}
    if best is None:
        return {}
    banks = bank_partners(best["program"])
    best["transfer_from"] = banks
    best["warning"] = ("confirm the award seat is still there before "
                       "transferring — transfers are irreversible")
    return best


def format_best_redemption(best: dict[str, Any]) -> str:
    if not best:
        return ""
    star = _cpp_star(best["cpp"])
    move = ""
    if best.get("transfer_from"):
        move = f" (move {best['transfer_from'][0]} → {best['program']})"
    return (f"🎖️ best redemption: {best['points']:,} {best['program']}"
            f"{move} — {best['cpp']:.0f} kobo/pt, {best['rating']}{star}."
            f" {best['warning']}.")


# ── output themes ───────────────────────────────────────────────────────────

_THEME_NAMES = ("rich", "compact", "minimal")


class OutputTheme:
    """A named presentation style for travel output.

    rich:    emoji + structure + verdict banners (default for chat)
    compact: one-liners, prices only
    minimal: bare numbers — for embedding in other flows
    """

    def __init__(self, name: str = "rich") -> None:
        self.name = name if name in _THEME_NAMES else "rich"

    def header(self, text: str) -> str:
        if self.name == "rich":
            return f"✈️ {text}"
        if self.name == "compact":
            return f"» {text}"
        return text

    def price(self, kobo: int) -> str:
        return _naira(kobo) if self.name != "minimal" else str(kobo // 100)

    def verdict(self, text: str) -> str:
        return text if self.name == "rich" else text.split(" — ")[0]

    def ok(self) -> str:
        return "✅" if self.name == "rich" else ("ok" if self.name == "compact"
                                                else "")

    def warn(self) -> str:
        return "⚠️" if self.name == "rich" else ("!" if self.name == "compact"
                                                else "")


THEMES = {"rich": OutputTheme("rich"), "compact": OutputTheme("compact"),
          "minimal": OutputTheme("minimal")}


def themed(name: str = "rich") -> OutputTheme:
    """Pick an output theme by name. Unknown → rich."""
    return THEMES.get(name, THEMES["rich"])


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
