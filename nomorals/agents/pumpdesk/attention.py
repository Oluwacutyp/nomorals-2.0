"""The attention verifier — is the noise around this token real?

On-chain momentum is only half the story callers sell; the other
half is attention, and attention is the cheapest thing to fake on
the platform (comment bots, bumped threads, recycled socials —
design §3.2/§8).  This verifier cross-checks the two against each
other and returns a verdict WITH its evidence:

* **ORGANIC** — off-chain mentions exist, come from varied
  sources, read as independently written, and sit in proportion
  to the on-chain buyer base;
* **MANUFACTURED** — attention is templated (the same text
  recycled across results), or wildly disproportionate to the
  real buyer count, or the token's social identity is recycled
  from another launch;
* **MIXED** — some evidence each way;
* **UNKNOWN** — the desk could not gather off-chain evidence at
  all.  Fail-closed (the P1 idiom): UNKNOWN carries NO quality
  component and NO score effect — a named gap, never a dressed-up
  guess, and never a boost.

Manufactured attention DISCOUNTS through the manipulation
component (design §4.4 rule 3): the score raises manipulation on
positive manufactured evidence only.  Absence of attention data
is not evidence of manufacture.

Evidence gathering uses the repo's registry-backed federated
search (web sources) through an injectable ``search_fn`` — the
desk supplies the real one; tests supply fixtures.  Social links
declared in the token's own metadata ride the launch record's
``uri``; two different mints sharing one metadata URI is the
recycled-identity signal the desk can prove from its own ledger.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

__all__ = [
    "verify_attention", "default_search_fn", "render_attention",
    "MANUFACTURED", "ORGANIC", "MIXED", "UNKNOWN",
]

ORGANIC = "ORGANIC"
MANUFACTURED = "MANUFACTURED"
MIXED = "MIXED"
UNKNOWN = "UNKNOWN"

#: Manufactured-evidence thresholds (named, tunable, in the open).
TEMPLATED_RATIO_FLAG = 0.5     # >= half the snippets near-identical
TEMPLATED_MIN_MENTIONS = 6
MISMATCH_RATIO_FLAG = 3.0      # mentions per unique buyer
MISMATCH_MIN_MENTIONS = 10


def default_search_fn(context: Any) -> Callable:
    """The production search: registry-backed federated web
    search.  Any failure yields no evidence (the verifier then
    reports UNKNOWN with the gap named)."""
    def _search(query: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            from ...search.federated import federated_search
            resp = federated_search(query, context=context,
                                    sources=["web"], limit=limit)
            out = []
            for r in getattr(resp, "results", []) or []:
                prov = getattr(r, "provenance", {}) or {}
                out.append({
                    "title": getattr(r, "title", ""),
                    "snippet": getattr(r, "snippet", ""),
                    "url": prov.get("url", ""),
                    "source": getattr(r, "source", ""),
                    "timestamp": getattr(r, "timestamp", None),
                })
            return out
        except Exception:  # noqa: BLE001 - no evidence, honestly
            return []
    return _search


def _domain(url: str) -> str:
    m = re.search(r"://([^/]+)", url or "")
    return m.group(1).lower() if m else ""


def _snippet_key(text: str) -> str:
    """Near-duplicate key: the first 60 normalized chars — comment
    bots recycle openers even when they vary the tail."""
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower())[:60].strip()


def verify_attention(dossier: dict[str, Any],
                     search_fn: Optional[Callable] = None,
                     store: Any = None,
                     now: Optional[float] = None) -> dict[str, Any]:
    """Cross-check one dossier's off-chain attention against its
    on-chain flow.  Pure over (dossier, search results, store) —
    no clock/network of its own beyond the injected search."""
    now = time.time() if now is None else now
    mint = dossier.get("mint", "")
    symbol = (dossier.get("symbol") or "").strip()
    gaps: list[str] = []
    evidence: dict[str, Any] = {"mint": mint, "symbol": symbol}
    verdict = UNKNOWN
    quality: Optional[float] = None

    if search_fn is None:
        gaps.append("no attention search bound — off-chain leg "
                    "not gathered")
        return {"verdict": verdict, "quality": None,
                "evidence": evidence, "gaps": gaps, "checked_at": now}
    queries = []
    if symbol:
        queries.append(f"{symbol} pump.fun")
        queries.append(f"{symbol} solana memecoin")
    if mint:
        queries.append(mint)
    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for q in queries:
        try:
            rows = search_fn(q, 10) or []
        except Exception as exc:  # noqa: BLE001 - a dead leg is a gap
            gaps.append(f"attention search failed: {exc}")
            rows = []
        for r in rows:
            url = (r.get("url") or "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(r)
    if not results:
        gaps.append("off-chain attention evidence unreachable "
                    "(search returned nothing) — verdict UNKNOWN, "
                    "no score effect")
        evidence["mentions"] = 0
        return {"verdict": verdict, "quality": None,
                "evidence": evidence, "gaps": gaps, "checked_at": now}

    mentions = len(results)
    domains = sorted({d for d in (_domain(r.get("url", ""))
                                  for r in results) if d})
    keys = [_snippet_key(r.get("snippet") or r.get("title") or "")
            for r in results]
    keys = [k for k in keys if k]
    dupes = len(keys) - len(set(keys))
    templated_ratio = round(dupes / len(keys), 3) if keys else 0.0
    stamps = [float(r["timestamp"]) for r in results
              if r.get("timestamp")]
    velocity = None
    if len(stamps) >= 3:
        span_h = max((max(stamps) - min(stamps)) / 3600.0, 0.25)
        velocity = round(mentions / span_h, 2)
    else:
        gaps.append("mention timestamps too sparse — velocity "
                    "unmeasured")
    trades = dossier.get("trades") or {}
    unique_buyers = int(trades.get("unique_buyers", 0))
    mismatch = (round(mentions / unique_buyers, 2)
                if unique_buyers else None)
    if mismatch is None:
        gaps.append("no on-chain buyer base to proportion "
                    "attention against")

    # recycled identity: another mint in the desk's own ledger
    # wearing the same metadata URI
    recycled = False
    uri = dossier.get("uri") or ""
    if store is not None and uri:
        for launch in store.launches():
            if launch.get("mint") != mint and launch.get("uri") == uri:
                recycled = True
                evidence["recycled_uri_with"] = launch.get("mint", "")
                break

    evidence.update({
        "mentions": mentions, "domains": domains,
        "domain_count": len(domains),
        "templated_ratio": templated_ratio,
        "mentions_per_hour": velocity,
        "unique_buyers": unique_buyers,
        "mentions_per_buyer": mismatch,
        "recycled_socials": recycled,
    })

    manufactured_points = []
    if (templated_ratio >= TEMPLATED_RATIO_FLAG
            and mentions >= TEMPLATED_MIN_MENTIONS):
        manufactured_points.append(
            f"templated text: {templated_ratio:.0%} of mentions share "
            f"an opener (comment-bot shape)")
    if (mismatch is not None and mismatch >= MISMATCH_RATIO_FLAG
            and mentions >= MISMATCH_MIN_MENTIONS):
        manufactured_points.append(
            f"attention/flow mismatch: {mentions} mentions against "
            f"{unique_buyers} real buyers ({mismatch:.1f} per buyer)")
    if recycled:
        manufactured_points.append(
            "recycled identity: this token's metadata URI is shared "
            "with another launch in the desk's ledger")
    evidence["manufactured_points"] = manufactured_points

    if len(manufactured_points) >= 2 or (
            manufactured_points and recycled):
        verdict, quality = MANUFACTURED, 0.1
    elif manufactured_points:
        verdict, quality = MIXED, 0.45
    elif mentions >= 3 and templated_ratio < 0.3:
        verdict = ORGANIC
        quality = round(0.8 + 0.2 * min(1.0, len(domains) / 5.0), 3)
    elif mentions >= 1:
        verdict, quality = MIXED, 0.5
        gaps.append("attention exists but is too thin/varied to "
                    "grade organic — MIXED by default, not by proof")
    else:
        gaps.append("no mentions found — attention unproven either "
                    "way")
    return {"verdict": verdict, "quality": quality,
            "evidence": evidence, "gaps": gaps, "checked_at": now}


def render_attention(att: dict[str, Any]) -> str:
    ev = att.get("evidence") or {}
    L = [f"📣 attention: {att.get('verdict')} "
         f"(quality {att.get('quality')})"]
    if ev.get("mentions") is not None:
        L.append(f"  mentions: {ev.get('mentions', 0)} across "
                 f"{ev.get('domain_count', 0)} domain(s); templated "
                 f"ratio {ev.get('templated_ratio', 0):.0%}; "
                 f"{ev.get('mentions_per_buyer')} mentions per real "
                 f"buyer")
    for p in ev.get("manufactured_points") or []:
        L.append(f"  ⚠️ {p}")
    for g in att.get("gaps") or []:
        L.append(f"  gap: {g}")
    return "\n".join(L)
