"""The attention verifier — is the noise around this token real?

On-chain momentum is only half the story callers sell; the other
half is attention, and attention is the cheapest thing to fake on
the platform (comment bots, bumped threads, recycled socials —
design §3.2/§8).  This verifier cross-checks the two against each
other and returns a verdict WITH its evidence:

* **ORGANIC** — attention exists, comes from varied sources,
  reads as independently written, and sits in proportion to the
  on-chain buyer base;
* **MANUFACTURED** — attention is templated (the same text
  recycled across results), or wildly disproportionate to the
  real buyer count, or the token's social identity is recycled
  from another launch, or its platform comment count floods a
  tiny buyer base, or its visibility is quite literally bought
  (active DexScreener boosts);
* **MIXED** — some evidence each way;
* **UNKNOWN** — the desk could not gather attention evidence at
  all.  Fail-closed (the P1 idiom): UNKNOWN carries NO quality
  component and NO score effect — a named gap, never a dressed-up
  guess, and never a boost.

Manufactured attention DISCOUNTS through the manipulation
component (design §4.4 rule 3): the score raises manipulation on
positive manufactured evidence only.  Absence of attention data
is not evidence of manufacture.

Four evidence legs, each genuinely readable by the desk:

1. **Web mentions** — the repo's registry-backed federated search
   through an injectable ``search_fn``.
2. **Platform comments** — pump.fun's own frontend API carries a
   per-coin ``reply_count`` (the desk reads the coin row live via
   ``platform_fn``, or the launch ledger's copy when the row came
   through the frontend poll / PumpPortal create).  Comment
   VELOCITY (count over the token's life) and its proportion to
   the real buyer base are measurable; commenter IDENTITIES are
   not exposed by the API (no replies route answers on the
   current frontend API — verified live 2026-10-22), so comment
   diversity is a named gap, never an invented number.
3. **DexScreener pair surface** — declared social links, website
   count, and paid-boost state from the token's best pair
   (``dex_fn``).  Fields the pair does not populate stay None —
   a gap, not a zero.
4. **The desk's own launch ledger** — two different mints sharing
   one metadata URI, or one X/Telegram handle declared by two
   different launches, is recycled identity the desk can prove
   from records it alone keeps.

Linked-social PRESENCE and handle recurrence are measured;
follower counts and off-platform velocity are not claimed —
the desk does not grade what it cannot read.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional
from .failures import record_failure

__all__ = [
    "verify_attention", "default_search_fn", "default_platform_fn",
    "default_dex_fn", "render_attention",
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
#: Platform comment flood: a comment count this large against a
#: buyer base this disproportionate is a comment-bot shape.
COMMENT_FLOOD_MIN = 50
COMMENT_PER_BUYER_FLAG = 10.0
#: Platform comments alone can support ORGANIC at this count when
#: their proportion to the buyer base is sane — the platform's own
#: thread is real attention surface, not a proxy for it.
PLATFORM_ORGANIC_MIN_COMMENTS = 5


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
        except Exception as exc:  # noqa: BLE001 - no evidence, honestly
            record_failure("attention.default_search_fn._search", exc)
            return []
    return _search


def default_platform_fn(http_get: Callable = None) -> Callable:
    """The production platform leg: the token's current coin row
    from the pump.fun frontend API (bounded list scan — the API
    has no by-mint route).  A failure yields None (a named gap).
    The desk injects its budgeted transport as ``http_get`` so
    the scan shares the frontend source's cycle budget."""
    def _platform(mint: str) -> Optional[dict[str, Any]]:
        try:
            from .firehose import _urllib_get, fetch_coin_row
            return fetch_coin_row(http_get or _urllib_get, mint)
        except Exception as exc:  # noqa: BLE001 - no evidence, honestly
            record_failure("attention.default_platform_fn._platform", exc)
            return None
    return _platform


def default_dex_fn(http_get: Callable = None) -> Callable:
    """The production DexScreener leg: the best pair's declared
    socials + paid-boost state.  A failure yields None.  The desk
    injects its recording transport as ``http_get`` so the read
    lands in the source registry; the plain default serves
    context-free callers."""
    def _dex(mint: str) -> Optional[dict[str, Any]]:
        try:
            from .outcomes import default_http_get, token_attention
            return token_attention(mint, http_get or default_http_get)
        except Exception as exc:  # noqa: BLE001 - no evidence, honestly
            record_failure("attention.default_dex_fn._dex", exc)
            return None
    return _dex


def _domain(url: str) -> str:
    m = re.search(r"://([^/]+)", url or "")
    return m.group(1).lower() if m else ""


def _snippet_key(text: str) -> str:
    """Near-duplicate key: the first 60 normalized chars — comment
    bots recycle openers even when they vary the tail."""
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower())[:60].strip()


def _handle(value: Any, kind: str) -> str:
    """Normalize a declared social (URL or bare handle) to the
    bare account name two launches can be compared on."""
    v = str(value or "").strip().lower()
    if not v:
        return ""
    v = re.sub(r"^https?://(www\.)?", "", v)
    if kind == "telegram":
        v = v.replace("t.me/", "").replace("telegram.me/", "")
    elif kind == "twitter":
        v = v.replace("twitter.com/", "").replace("x.com/", "")
    return v.strip("/").lstrip("@").strip()


def _own_socials(dossier: dict[str, Any], launch: Optional[dict],
                 row: Optional[dict]) -> dict[str, str]:
    """This token's declared handles, from the freshest source the
    desk has: the live coin row, else its launch record, else the
    dossier itself."""
    out: dict[str, str] = {}
    for src in (row, launch, dossier):
        if not isinstance(src, dict):
            continue
        for kind in ("twitter", "telegram"):
            if kind not in out:
                h = _handle(src.get(kind), kind)
                if h:
                    out[kind] = h
    return out


def verify_attention(dossier: dict[str, Any],
                     search_fn: Optional[Callable] = None,
                     store: Any = None,
                     now: Optional[float] = None,
                     platform_fn: Optional[Callable] = None,
                     dex_fn: Optional[Callable] = None) -> dict[str, Any]:
    """Cross-check one dossier's attention against its on-chain
    flow, over every evidence leg the desk can bind.  Pure over
    (dossier, injected legs, store) — no clock/network of its own
    beyond the injected functions."""
    now = time.time() if now is None else now
    mint = dossier.get("mint", "")
    symbol = (dossier.get("symbol") or "").strip()
    gaps: list[str] = []
    evidence: dict[str, Any] = {"mint": mint, "symbol": symbol}
    trades = dossier.get("trades") or {}
    unique_buyers = int(trades.get("unique_buyers", 0))

    if search_fn is None and platform_fn is None and dex_fn is None \
            and store is None:
        gaps.append("no attention legs bound — off-chain evidence "
                    "not gathered")
        return {"verdict": UNKNOWN, "quality": None,
                "evidence": evidence, "gaps": gaps, "checked_at": now}

    # ── leg 2/4 inputs: the launch record + the live coin row ────
    launch = None
    if store is not None:
        try:
            launch = store.launch(mint)
        except Exception as exc:  # noqa: BLE001
            record_failure("attention.verify_attention.launch", exc)
            launch = None
    row: Optional[dict[str, Any]] = None
    if platform_fn is not None and mint:
        try:
            row = platform_fn(mint)
        except Exception as exc:  # noqa: BLE001 - a dead leg is a gap
            gaps.append(f"platform coin row unreachable: {exc}")
            row = None

    # ── leg 1: web mentions ──────────────────────────────────────
    results: list[dict[str, Any]] = []
    mentions = 0
    templated_ratio = 0.0
    velocity = None
    mismatch = None
    if search_fn is None:
        gaps.append("no attention search bound — web mention leg "
                    "not gathered")
        evidence["mentions"] = 0
    else:
        queries = []
        if symbol:
            queries.append(f"{symbol} pump.fun")
            queries.append(f"{symbol} solana memecoin")
        if mint:
            queries.append(mint)
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
        if len(stamps) >= 3:
            span_h = max((max(stamps) - min(stamps)) / 3600.0, 0.25)
            velocity = round(mentions / span_h, 2)
        elif results:
            gaps.append("mention timestamps too sparse — web "
                        "velocity unmeasured")
        mismatch = (round(mentions / unique_buyers, 2)
                    if unique_buyers else None)
        if results and mismatch is None:
            gaps.append("no on-chain buyer base to proportion "
                        "attention against")
        evidence.update({
            "mentions": mentions, "domains": domains,
            "domain_count": len(domains),
            "templated_ratio": templated_ratio,
            "mentions_per_hour": velocity,
            "unique_buyers": unique_buyers,
            "mentions_per_buyer": mismatch,
        })

    # ── leg 2: platform comments (count + velocity) ──────────────
    comments: Optional[int] = None
    created = 0.0
    for src in (row, launch):
        if isinstance(src, dict):
            if comments is None and src.get("reply_count") is not None:
                try:
                    comments = int(src.get("reply_count") or 0)
                except (TypeError, ValueError):
                    comments = None
            if not created:
                ts = src.get("created_timestamp")
                if ts:
                    created = float(ts) / 1000.0
                elif src.get("created_ts"):
                    created = float(src.get("created_ts") or 0)
    if not created:
        created = float(dossier.get("created_at") or 0)
    comments_per_hour = None
    comments_per_buyer = None
    if comments is not None:
        if created:
            age_h = max((now - created) / 3600.0, 0.25)
            comments_per_hour = round(comments / age_h, 2)
        comments_per_buyer = (round(comments / unique_buyers, 2)
                              if unique_buyers else None)
        evidence["platform_comments"] = comments
        evidence["platform_comments_per_hour"] = comments_per_hour
        evidence["platform_comments_per_buyer"] = comments_per_buyer
        gaps.append("commenter identities are not exposed by the "
                    "frontend API — comment diversity unmeasured")
    elif platform_fn is not None or launch is not None:
        gaps.append("platform comment count unreadable (coin row "
                    "not found in the scanned frontend list and "
                    "no launch-record copy)")

    # ── leg 3: DexScreener pair surface ──────────────────────────
    dex: Optional[dict[str, Any]] = None
    if dex_fn is not None and mint:
        try:
            dex = dex_fn(mint)
        except Exception as exc:  # noqa: BLE001 - a dead leg is a gap
            gaps.append(f"DexScreener attention read failed: {exc}")
            dex = None
    boosts_active: Optional[int] = None
    if dex is not None:
        socials = dex.get("socials")
        evidence["dex_socials"] = (
            [s.get("type", "") for s in socials]
            if isinstance(socials, list) else None)
        evidence["dex_websites"] = dex.get("websites")
        boosts_active = dex.get("boosts_active")
        evidence["dex_boosts_active"] = boosts_active
        if socials is None:
            gaps.append("DexScreener pair carries no socials block "
                        "— declared-socials surface unmeasured")
    elif dex_fn is not None:
        gaps.append("no DexScreener pair for this mint yet — "
                    "pair-surface leg unmeasured")

    # ── leg 4: recycled identity in the desk's own ledger ────────
    recycled = False
    uri = dossier.get("uri") or ""
    if store is not None and uri:
        for other in store.launches():
            if other.get("mint") != mint and other.get("uri") == uri:
                recycled = True
                evidence["recycled_uri_with"] = other.get("mint", "")
                break
    own = _own_socials(dossier, launch, row)
    recycled_handles: dict[str, list[str]] = {}
    if store is not None and own:
        for other in store.launches():
            if other.get("mint") == mint:
                continue
            for kind, handle in own.items():
                if _handle(other.get(kind), kind) == handle:
                    recycled_handles.setdefault(kind, []).append(
                        other.get("mint", ""))
    if recycled_handles:
        evidence["recycled_handles"] = recycled_handles
    evidence["recycled_socials"] = bool(recycled or recycled_handles)
    evidence["declared_socials"] = own

    # ── verdict: manufactured points across every leg ────────────
    points: list[tuple[str, str]] = []
    if (templated_ratio >= TEMPLATED_RATIO_FLAG
            and mentions >= TEMPLATED_MIN_MENTIONS):
        points.append((
            f"templated text: {templated_ratio:.0%} of mentions share "
            f"an opener (comment-bot shape)", "templated"))
    if (mismatch is not None and mismatch >= MISMATCH_RATIO_FLAG
            and mentions >= MISMATCH_MIN_MENTIONS):
        points.append((
            f"attention/flow mismatch: {mentions} mentions against "
            f"{unique_buyers} real buyers ({mismatch:.1f} per buyer)",
            "mismatch"))
    if recycled:
        points.append((
            "recycled identity: this token's metadata URI is shared "
            "with another launch in the desk's ledger", "recycled"))
    for kind, mints in recycled_handles.items():
        points.append((
            f"recycled {kind} identity: handle @{own.get(kind, '')} "
            f"is also declared by {len(mints)} other launch(es) in "
            f"the desk's ledger", "recycled"))
    if (comments is not None and comments >= COMMENT_FLOOD_MIN
            and comments_per_buyer is not None
            and comments_per_buyer >= COMMENT_PER_BUYER_FLAG):
        points.append((
            f"platform comment flood: {comments} comments "
            f"({comments_per_hour}/hour) against {unique_buyers} "
            f"real buyers ({comments_per_buyer:.1f} per buyer) — "
            f"comment-bot shape", "comment_flood"))
    if boosts_active:
        points.append((
            f"paid promotion: {boosts_active} DexScreener boost(s) "
            f"active — visibility bought, not earned", "boosts"))
    evidence["manufactured_points"] = [p for p, _k in points]
    kinds = {k for _p, k in points}

    if points and (len(points) >= 2 or "recycled" in kinds):
        verdict, quality = MANUFACTURED, 0.1
    elif points:
        verdict, quality = MIXED, 0.45
    elif mentions >= 3 and templated_ratio < 0.3:
        verdict = ORGANIC
        quality = round(0.8 + 0.2 * min(
            1.0, int(evidence.get("domain_count", 0)) / 5.0), 3)
        evidence["organic_basis"] = "varied web mentions"
    elif (comments is not None
            and comments >= PLATFORM_ORGANIC_MIN_COMMENTS
            and (comments_per_buyer is None
                 or comments_per_buyer < COMMENT_PER_BUYER_FLAG)):
        verdict, quality = ORGANIC, 0.7
        evidence["organic_basis"] = ("platform comment thread in "
                                     "proportion to the buyer base")
    elif mentions >= 1 or (comments is not None and comments >= 1) \
            or evidence.get("dex_socials"):
        verdict, quality = MIXED, 0.5
        gaps.append("attention exists but is too thin/varied to "
                    "grade organic — MIXED by default, not by proof")
    else:
        verdict, quality = UNKNOWN, None
        gaps.append("no attention evidence gathered on any leg — "
                    "verdict UNKNOWN, no score effect")
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
    if ev.get("platform_comments") is not None:
        L.append(f"  platform: {ev['platform_comments']} comments "
                 f"({ev.get('platform_comments_per_hour')}/hour, "
                 f"{ev.get('platform_comments_per_buyer')} per real "
                 f"buyer)")
    if ev.get("dex_socials") is not None or \
            ev.get("dex_boosts_active") is not None:
        L.append(f"  dexscreener: socials "
                 f"{ev.get('dex_socials')}, websites "
                 f"{ev.get('dex_websites')}, boosts active "
                 f"{ev.get('dex_boosts_active')}")
    for p in ev.get("manufactured_points") or []:
        L.append(f"  ⚠️ {p}")
    for g in att.get("gaps") or []:
        L.append(f"  gap: {g}")
    return "\n".join(L)
