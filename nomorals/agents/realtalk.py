"""Pulse real-talk — one shared talk builder for every briefing slot.

Owner's program (2026-10-22, M2): every pulse — morning, afternoon,
evening, midnight — carries one *real talk*: a short teaching in
Devon's voice on something worth thinking about.  Three sources feed
it, co-equal (the owner's have-it-all rule — no source retires
another):

* **trending** — the highest-interest story on the news delivery
  ledger (the same undelivered set F6 put every slot on), with the
  article body fetched and readability-extracted so the talk is
  grounded in the piece itself, never in a ~200-char feed summary;
* **beyond-trending (the wheel)** — a rotating domain wheel (science,
  history, philosophy, …) hunted live over the registry-backed search
  class, the winning page fetched and extracted the same way;
* **Library** — a passage she actually holds (WisdomKeeper digests /
  keeper passages / BookForge holdings) drawn from the shared
  Teachings ledger M1 built.

One builder, every consumer: the morning briefing's
``RealTalkProvider`` and the afternoon/evening/midnight slot composers
all call :func:`compose_realtalk`, so sourcing and dedupe can never
diverge between slots.  Every served talk is marked in the Teachings
ledger (``surface="pulse-<slot>"``, ``kind="realtalk"``), which is
what makes a repeat across pulses — or against the Motivation class —
structurally impossible: the ledger's 30-day item cooldown and 48h
theme cooldown apply to talks exactly as they do to motivations.

Mix discipline (design §3.2, enforced from the ledger's own served
record, not a rota table): at most one slot in any rolling 24h leads
with trending, and across a rolling week roughly half the talks are
beyond-trending or Library.  Both rules are preferences with a full
fallback — a preference that would silence the segment is dropped,
never the other way round.

Failure honesty: extraction is fail-closed (a story whose page cannot
be extracted is not talked about from its summary), hunts that find
nothing yield nothing, and a dead brain degrades the talk to the
passage itself as a teaching card (M1's pattern) — never a canned
talk, never a fabricated citation.  No material anywhere → ``None``
and the pulse simply carries no talk that slot.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

__all__ = [
    "DOMAINS",
    "compose_realtalk",
]

_log = logging.getLogger(__name__)

#: The beyond-trending domain wheel.  Seeds, not a rota: the day's
#: domain is the least-recently-talked one (from the ledger's served
#: record), and the 48h theme cooldown vetoes repeats — the wheel
#: grows by what the hunts find, per the open-sources law.
DOMAINS: tuple[str, ...] = (
    "science", "history", "philosophy", "technology", "economics",
    "the body", "space", "language", "craft", "the mind",
    "how things work", "the natural world",
)

#: A fetched page must yield at least this much extracted text to
#: ground a talk — below it, the source fails closed rather than
#: letting a snippet or a feed summary pretend to be the article.
MIN_GROUNDING_CHARS = 400
#: Grounding passage stored on the ledger item / fed to the brain.
PASSAGE_CHARS = 1500
#: The degraded card excerpts the passage to about this many chars.
CARD_PASSAGE_CHARS = 700

_WEEK_S = 7 * 86400.0
_DAY_S = 86400.0

#: Stocking runs the ledger's producers (both feeds, co-equal — M1)
#: and is LAZY: only the Library source consults the pool, so only it
#: pays for a stock.  Pulses are hours apart, so a slot that reaches
#: the Library always re-stocks; the memo only collapses repeat
#: composes inside one process (a manual /briefing right after the
#: scheduled one) onto the pool they already stocked.
_STOCK_TTL_S = 1800.0
_stock_memo: dict[int, float] = {}


# ── seams (tests patch these; production uses the real organs) ──────

def _hunt(context: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """One live-net hunt via the registry-backed search class."""
    from .teachings import _web_hunt

    return _web_hunt(context, query, limit=limit)


def _extract(url: str) -> str:
    """Fetch + readability-extract one page.  "" on ANY failure —
    robots, walls, dead hosts all read the same: no grounding."""
    try:
        from ..tools.web import web_extract

        out = web_extract(url, max_chars=20000)
        return str((out or {}).get("text") or "").strip()
    except Exception:  # noqa: BLE001 - fail-closed by design
        _log.debug("realtalk: extract failed for %s", url, exc_info=True)
        return ""


# ── item shape (mirrors teachings._base_item; the ledger's contract) ─

def _make_item(*, title: str, theme: str, source_type: str,
               source_ref: str, source_label: str,
               passage: str) -> dict[str, Any]:
    from ..core.ids import new_id

    return {
        "id": new_id("teach"),
        "kind": "realtalk",
        "title": title or "",
        "body": "",
        "theme": theme or "",
        "tradition": "",
        "weight": 1.0,
        "source": {
            "type": source_type,
            "ref": source_ref or "",
            "label": source_label or "",
            "passage": passage or "",
        },
        "created_at": time.time(),
        "served": [],
    }


# ── stocking ─────────────────────────────────────────────────────────

def _ensure_stocked(ledger: Any, context: Any, now: float) -> None:
    """Stock the ledger's pool unless this process stocked it recently."""
    key = id(context)
    last = _stock_memo.get(key, 0.0)
    if now - last < _STOCK_TTL_S and ledger.candidates(now=now):
        return
    try:
        ledger.stock(now=now)
    except Exception:  # noqa: BLE001 - producers are fail-soft anyway
        _log.debug("realtalk: stock failed", exc_info=True)
    _stock_memo[key] = now


# ── mix discipline (from the ledger's served record) ────────────────

def _talk_history(ledger: Any, now: float) -> list[dict[str, Any]]:
    """Real talks served by any pulse in the trailing week."""
    try:
        rows = ledger.served_items(since=now - _WEEK_S,
                                   surface_prefix="pulse-")
    except Exception:  # noqa: BLE001
        return []
    return [r for r in rows if str(r.get("kind") or "") == "realtalk"]


def _source_order(ledger: Any, now: float) -> list[str]:
    """Preference order over the three sources, from served history.

    * a trending talk inside the trailing 24h pushes trending last
      (at most one trending-led slot per rolling day);
    * if beyond-trending + Library talks are under half the week's
      talks, they lead until the mix rebalances.
    The order is a preference, never a gate: the builder walks it
    until a source yields grounded material.
    """
    history = _talk_history(ledger, now)
    trending_recent = False
    beyond = 0
    for it in history:
        src_type = str((it.get("source") or {}).get("type") or "")
        served_at = max(
            (float(s.get("at") or 0) for s in it.get("served") or []),
            default=0.0)
        if src_type == "news":
            if now - served_at < _DAY_S:
                trending_recent = True
        else:
            beyond += 1
    total = len(history)
    beyond_short = total >= 2 and beyond * 2 < total
    if trending_recent or beyond_short:
        return ["wheel", "library", "trending"]
    return ["trending", "wheel", "library"]


# ── the three sources ────────────────────────────────────────────────
# Each returns (item, talk_kind) or None, and never raises.

def _trending_item(context: Any, ledger: Any,
                   now: float) -> tuple[dict[str, Any], str] | None:
    """The highest-interest undelivered story, grounded in its own
    extracted article body.  Fail-closed: no extraction, no talk."""
    try:
        from .news import NewsAgent, publisher_of

        rows = NewsAgent(context).undelivered(limit=30)
    except Exception:  # noqa: BLE001
        _log.debug("realtalk: news read failed", exc_info=True)
        return None
    if not rows:
        return None
    # A story any surface already taught from is spent, whichever
    # passage variant stocked it — refs are the story's identity here.
    talked_refs: set[str] = set()
    try:
        from .teachings import ITEM_COOLDOWN_S

        for it in ledger.served_items(since=now - ITEM_COOLDOWN_S):
            src = it.get("source") or {}
            if str(src.get("type") or "") == "news" and src.get("ref"):
                talked_refs.add(str(src["ref"]))
    except Exception:  # noqa: BLE001
        pass

    def _rank(row: dict[str, Any]) -> tuple[int, float, float]:
        interest = row.get("interest")
        return (1 if interest is not None else 0,
                float(interest) if interest is not None else 0.0,
                float(row.get("published") or 0))

    fresh = [r for r in rows
             if str(r.get("url") or "") not in talked_refs
             and str(r.get("url") or "").startswith("http")]
    fresh.sort(key=_rank, reverse=True)
    for row in fresh[:3]:
        url = str(row.get("url") or "")
        text = _extract(url)
        if len(text) < MIN_GROUNDING_CHARS:
            continue  # walled / thin — never talked from a summary
        try:
            label = publisher_of(row) or str(
                row.get("source_name") or "news")
        except Exception:  # noqa: BLE001
            label = str(row.get("source_name") or "news")
        item = _make_item(
            title=str(row.get("title") or "").strip(),
            theme="",
            source_type="news", source_ref=url, source_label=label,
            passage=text[:PASSAGE_CHARS])
        return item, "trending"
    return None


def _wheel_item(context: Any, ledger: Any,
                now: float) -> tuple[dict[str, Any], str] | None:
    """The day's beyond-trending domain: hunted live, the winning page
    fetched and extracted.  Fail-closed the same way as trending."""
    # Domain choice: least-recently talked wins; the shared 48h theme
    # cooldown vetoes a domain outright (motivation included).
    last_by_theme: dict[str, float] = {}
    try:
        for it in _talk_history(ledger, now):
            theme = str(it.get("theme") or "").strip().lower()
            if not theme:
                continue
            at = max((float(s.get("at") or 0)
                      for s in it.get("served") or []), default=0.0)
            last_by_theme[theme] = max(last_by_theme.get(theme, 0.0), at)
    except Exception:  # noqa: BLE001
        pass
    domains = []
    for d in DOMAINS:
        try:
            if ledger.theme_in_cooldown(d, now=now):
                continue
        except Exception:  # noqa: BLE001
            pass
        domains.append(d)
    if not domains:
        return None
    domains.sort(key=lambda d: last_by_theme.get(d, 0.0))
    domain = domains[0]

    queries = [
        f"what is newly understood about {domain}",
        f"{domain} explained — a discovery or idea worth knowing",
    ]
    for query in queries:
        try:
            results = _hunt(context, query, limit=5)
        except Exception:  # noqa: BLE001 - offline yields no hunt
            _log.debug("realtalk: wheel hunt failed", exc_info=True)
            return None
        if not results:
            # Every backend answered empty: a rephrased query against
            # the same silence is not a second chance, it's a second
            # wait.  The Library (reached next) is the better use of
            # the slot's time.  A query that DID return results but
            # none extractable still earns the rephrase below.
            break
        for r in results[:3]:
            url = str(r.get("url") or "").strip()
            if not url.startswith("http"):
                continue
            text = _extract(url)
            if len(text) < MIN_GROUNDING_CHARS:
                continue
            item = _make_item(
                title=str(r.get("title") or "").strip() or domain,
                theme=domain,
                source_type="web", source_ref=url,
                source_label=str(r.get("title") or "").strip() or domain,
                passage=text[:PASSAGE_CHARS])
            return item, "wheel"
    return None


def _library_item(context: Any, ledger: Any, now: float,
                  rng: random.Random) -> tuple[dict[str, Any], str] | None:
    """A passage she holds, drawn from the ledger's stocked pool —
    the pool already carries the 30-day item cooldown; the 48h theme
    cooldown is applied here exactly as ledger.pick applies it."""
    try:
        pool = ledger.candidates(now=now)
    except Exception:  # noqa: BLE001
        return None
    eligible: list[dict[str, Any]] = []
    for cand in pool:
        src = cand.get("source") or {}
        if str(src.get("type") or "") not in ("wisdom", "book"):
            continue
        theme = str(cand.get("theme") or "").strip().lower()
        if theme:
            try:
                if ledger.theme_in_cooldown(theme, now=now):
                    continue
            except Exception:  # noqa: BLE001
                pass
        eligible.append(cand)
    if not eligible:
        return None
    weights = [max(0.1, float(c.get("weight") or 1.0)) for c in eligible]
    return rng.choices(eligible, weights=weights, k=1)[0], "library"


# ── synthesis + degraded card ────────────────────────────────────────

def _synthesize_talk(context: Any, item: dict[str, Any], slot: str,
                     talk_kind: str) -> str | None:
    """The brain's talk: 150–250 words, grounded in the passage only.

    Returns None when no brain answers or the answer is off-size —
    the caller degrades to the teaching card, never to canned text.
    """
    try:
        from ..llm.brain import brain_for

        brain = brain_for(context)
        if brain is None:
            return None
        src = item.get("source") or {}
        passage = str(src.get("passage") or "").strip()
        label = str(src.get("label") or "").strip()
        angle = {
            "trending": "This is a live news story. Say what happened "
                        "first, plainly.",
            "wheel": "This is a found piece on a bigger idea. Say what "
                     "the idea is first, plainly.",
            "library": "This is a passage from a work she holds. Say "
                       "what it teaches first, plainly.",
        }.get(talk_kind, "")
        prompt = (
            f"Write the real-talk segment of a {slot} briefing for "
            "your owner, in your own voice — direct, curious, "
            "plain-spoken, teaching something real. 150 to 250 words. "
            "Shape: what it is → why it matters → the wider thought it "
            f"opens. {angle} Use ONLY facts from the material below — "
            "invent nothing, add no quotes or numbers that are not "
            "there. No greeting, no sign-off, no headline voice.\n\n"
            f"Material (from {label or 'her sources'}):\n{passage}"
        )
        resp = brain.chat([{"role": "user", "content": prompt}])
        text = resp.text if hasattr(resp, "text") else str(resp)
        text = (text or "").strip()
        words = len(text.split())
        if not (100 <= words <= 340):
            _log.info("realtalk: synthesis off-size (%d words); "
                      "degrading to card", words)
            return None
        return text
    except Exception:  # noqa: BLE001 - no brain == degraded card
        _log.debug("realtalk: synthesis unavailable", exc_info=True)
        return None


_CARD_FRAMING = {
    "trending": "No spin from me on this one — the story itself:",
    "wheel": "Found this today — the piece itself, worth the read:",
    "library": "Straight from her shelves — the passage itself:",
}


def _card(item: dict[str, Any], talk_kind: str) -> str:
    """The degraded path: the grounding passage itself as the talk.

    Real material with its provenance — the M1 teaching-card shape.
    Never a stored rotation, never invented framing facts.
    """
    src = item.get("source") or {}
    passage = str(src.get("passage") or "").strip()
    if len(passage) > CARD_PASSAGE_CHARS:
        cut = passage[:CARD_PASSAGE_CHARS]
        passage = cut.rsplit(" ", 1)[0].rstrip() + "…"
    title = str(item.get("title") or "").strip()
    lines = [_CARD_FRAMING.get(talk_kind, _CARD_FRAMING["library"])]
    if title and title.lower() not in passage.lower():
        lines.append(f"**{title}**")
    lines.append(f"“{passage}”")
    return "\n".join(lines)


def _footer(item: dict[str, Any]) -> str:
    """Provenance, in the same words M1's motivation pieces use."""
    src = item.get("source") or {}
    kind = str(src.get("type") or "")
    label = str(src.get("label") or "").strip()
    origin = ("From her library" if kind in ("wisdom", "book")
              else "Found today" if kind in ("web", "news")
              else "Source")
    out = f"— {origin}: {label}" if label else ""
    ref = str(src.get("ref") or "")
    if kind in ("web", "news") and ref.startswith("http"):
        out = f"{out}\n{ref}" if out else ref
    return out


# ── the builder ──────────────────────────────────────────────────────

def compose_realtalk(context: Any, slot: str, *,
                     now: float | None = None,
                     rng: random.Random | None = None
                     ) -> dict[str, Any] | None:
    """Build one real talk for ``slot`` and mark it served.

    Returns the ledger item with ``body`` (the talk prose or card),
    ``text`` (body + provenance footer — the embeddable segment),
    ``talk_kind`` (trending | wheel | library) and ``degraded``
    set — or None when no source yields grounded material.  The
    served-mark lands BEFORE the caller delivers: a composed talk is
    spent even if its pulse's delivery later fails, so no surface
    ever repeats it (the M1 motivation semantics).  Never raises.
    """
    slot = (slot or "morning").strip().lower()
    now = time.time() if now is None else now
    rng = rng or random.Random()
    try:
        from .teachings import TeachingsLedger

        ledger = TeachingsLedger(context)

        built: tuple[dict[str, Any], str] | None = None
        for source in _source_order(ledger, now):
            if source == "trending":
                built = _trending_item(context, ledger, now)
            elif source == "wheel":
                built = _wheel_item(context, ledger, now)
            else:
                # Only the Library source reads the stocked pool, so
                # stocking (which runs every producer, hunts included)
                # happens here, at the moment it is actually needed —
                # a trending- or wheel-decided talk never pays for it.
                _ensure_stocked(ledger, context, now)
                built = _library_item(context, ledger, now, rng)
            if built is not None:
                break
        if built is None:
            _log.info("realtalk/%s: no grounded material this slot",
                      slot)
            return None
        item, talk_kind = built

        synthesized = _synthesize_talk(context, item, slot, talk_kind)
        body = synthesized if synthesized else _card(item, talk_kind)
        footer = _footer(item)
        text = f"{body}\n\n{footer}" if footer else body

        served = ledger.mark_served(
            item, f"pulse-{slot}", body=text, kind="realtalk", now=now)
        out = dict(item)
        out["id"] = served.get("id") or item.get("id")
        out["kind"] = "realtalk"
        out["talk_kind"] = talk_kind
        out["body"] = body
        out["footer"] = footer
        out["text"] = text
        out["degraded"] = synthesized is None
        out["surface"] = f"pulse-{slot}"
        return out
    except Exception:  # noqa: BLE001 - a talk must never sink a pulse
        _log.warning("realtalk/%s: builder failed", slot, exc_info=True)
        return None
