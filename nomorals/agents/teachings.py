"""The Teachings ledger — one spine under motivation, pulses, trivia.

The owner's program (2026-10-22): everything works hand in hand.  A
single store of *teaching items* — a source passage, its provenance,
and Devon's synthesis — that every teaching surface draws from and
marks as served: the Motivation class (``agents/motivation.py``), the
pulse real-talk segments (M2), and the Library-fed trivia bank (M3).
Built separately, those three would be three silos repeating each
other; the ledger is what prevents it.

Sourcing is **both-feeds, co-equal** (owner's correction, 2026-10-22:
"It can use both though. I'm not the type to give up one thing for
another."): the Library (WisdomKeeper passages + WisdomOrgan digests
+ BookForge holdings) and the live net (trending news + hunted finds)
both stock the candidate pool continuously, each item carrying its
provenance.  Generation draws from the combined store.  There is no
first/fallback ladder and no pre-baked inventory — producers run at
stocking time, the pool is a ≤24h candidate cache, and the only
long-lived rows are *served* items.  A canned rotation is therefore
structurally impossible, not just discouraged.

Selection guarantees (the anti-repeat machinery):
* an item served on ANY surface is invisible to every surface for
  30 days (item cooldown);
* a theme served anywhere is invisible for 48h (theme cooldown);
* source-mix floors for the motivation surfaces, tracked from served
  history: net-sourced pieces are owed ≥2 mornings a week, and the
  week's teachings may not collapse into a single tradition — both
  enforced as soft preferences that never empty the pool.

Storage follows the wisdom-organ precedent: small tables via
``context.db`` (``ensure_schema``), created lazily.  With no db the
ledger degrades to a per-process in-memory store — producers still
run and served-marks still hold for the process lifetime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from typing import Any

__all__ = [
    "ITEM_COOLDOWN_S",
    "THEME_COOLDOWN_S",
    "CANDIDATE_TTL_S",
    "BASE_THEMES",
    "TeachingsLedger",
    "ensure_schema",
    "item_key",
    "theme_wheel",
]

_log = logging.getLogger(__name__)

#: A served item is invisible to every surface for 30 days.
ITEM_COOLDOWN_S = 30 * 86400.0
#: A served theme is invisible to every surface for 48 hours.
THEME_COOLDOWN_S = 48 * 3600.0
#: Candidate pool entries older than this are pruned — the pool is a
#: cache of fresh stock, never an inventory.
CANDIDATE_TTL_S = 24 * 3600.0
#: Served rows older than this are pruned (cooldown data outlives its
#: own window by half, then goes).
PRUNE_AFTER_S = 45 * 86400.0

#: Seed themes for the wheel.  Seeds, not a rota: the wheel re-weights
#: them from Library holdings, interest pins, and serve staleness, and
#: traditions found in the corpus join as themes of their own.
BASE_THEMES: tuple[str, ...] = (
    "discipline", "mortality", "money", "wonder", "courage",
    "patience", "gratitude", "craft", "adversity", "attention",
    "ambition", "stillness",
)

_WEEK_S = 7 * 86400.0


def ensure_schema(db: Any) -> None:
    """Create the teachings tables when a db exists.  Never raises."""
    if db is None:
        return
    try:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS teachings_items (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'teaching',
                title TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                theme TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '',
                source_ref TEXT NOT NULL DEFAULT '',
                source_label TEXT NOT NULL DEFAULT '',
                passage TEXT NOT NULL DEFAULT '',
                item_key TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                served_json TEXT NOT NULL DEFAULT '[]'
            )
            """)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS teachings_candidates (
                item_key TEXT PRIMARY KEY,
                item_json TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """)
    except Exception:  # noqa: BLE001 - schema is best-effort at import
        _log.debug("teachings: schema ensure failed", exc_info=True)


def item_key(source_type: str, source_ref: str, passage: str) -> str:
    """Stable identity of one teaching item across producers/surfaces."""
    raw = f"{source_type}\x1f{source_ref}\x1f{(passage or '')[:400]}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()


def _passage_text(raw: Any) -> str:
    """Duck-type a digest passage (str | dict) into plain text."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for key in ("text", "snippet", "passage", "content"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


# ── producers ────────────────────────────────────────────────────────
# Every producer returns candidate item dicts in the ledger shape and
# NEVER raises: a dead source yields fewer candidates, not a crash.
# Producers are asked for a spread of themes so the pool stocks
# broadly; an item's theme is a property of the item, not of the ask.

def _base_item(*, kind: str, title: str, theme: str, source_type: str,
               source_ref: str, source_label: str, passage: str,
               tradition: str = "", weight: float = 1.0) -> dict[str, Any]:
    from ..core.ids import new_id

    return {
        "id": new_id("teach"),
        "kind": kind,
        "title": title or "",
        "body": "",
        "theme": theme or "",
        "tradition": tradition or "",
        "weight": float(weight),
        "source": {
            "type": source_type,
            "ref": source_ref or "",
            "label": source_label or "",
            "passage": passage or "",
        },
        "created_at": time.time(),
        "served": [],
    }


def library_candidates(context: Any, themes: list[str]) -> list[dict[str, Any]]:
    """Candidates from her Library: digests, keeper passages, books.

    Co-equal feed #1 (owner's correction): this stocks the pool; it is
    not a first rung anything falls back from.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item: dict[str, Any]) -> None:
        src = item["source"]
        key = item_key(src["type"], src["ref"], src["passage"])
        if key in seen or not src["passage"]:
            return
        seen.add(key)
        out.append(item)

    # (a) WisdomOrgan digests — "what she has been learning lately".
    try:
        from ..wisdom.autonomy import WisdomOrgan

        organ = WisdomOrgan(context)
        for dig in organ.recent_digests(limit=6):
            slug = str(dig.get("slug") or "")
            if not slug:
                continue
            full = organ.get_digest(slug) or dig
            title = str(full.get("title") or slug)
            tradition = str(full.get("tradition") or "")
            terms = [str(t) for t in (full.get("key_terms") or [])]
            theme = tradition or (terms[0] if terms else "")
            for raw in (full.get("passages") or [])[:4]:
                text = _passage_text(raw)
                if len(text) < 40:
                    continue
                _add(_base_item(
                    kind="teaching", title=title, theme=theme,
                    source_type="wisdom", source_ref=slug,
                    source_label=f"{title} — WisdomKeeper",
                    passage=text[:1200], tradition=tradition,
                    weight=1.2))
    except Exception:  # noqa: BLE001 - a thin corpus yields nothing
        _log.debug("teachings: digest producer failed", exc_info=True)

    # (b) Keeper passages per theme (the corpus asked directly).
    try:
        from ..wisdom.keeper import WisdomKeeper

        keeper = WisdomKeeper(context)
        for theme in themes[:2]:
            try:
                ans = keeper.ask(theme, top=3)
            except Exception:  # noqa: BLE001
                continue
            for p in (getattr(ans, "passages", None) or []):
                text = (getattr(p, "snippet", "") or "").strip()
                if len(text) < 40:
                    continue
                work = getattr(p, "work", "") or "WisdomKeeper"
                section = getattr(p, "section", "") or ""
                label = f"{work}{(' — ' + section) if section else ''}"
                _add(_base_item(
                    kind="teaching", title=work, theme=theme,
                    source_type="wisdom",
                    source_ref=getattr(p, "slug", "") or work,
                    source_label=f"{label} — WisdomKeeper",
                    passage=text[:1200],
                    tradition=getattr(p, "canon_status", "") or ""))
    except Exception:  # noqa: BLE001
        _log.debug("teachings: keeper producer failed", exc_info=True)

    # (c) BookForge holdings — fiction and non-fiction she holds.
    try:
        from ..books.library import Library

        library = Library(context)
        for theme in themes[:2]:
            try:
                hits = library.search(theme, top=3)
            except Exception:  # noqa: BLE001
                continue
            for hit in hits or []:
                text = ""
                for attr in ("context", "text", "snippet", "passage"):
                    val = getattr(hit, attr, "")
                    if isinstance(val, str) and val.strip():
                        text = val.strip()
                        break
                if len(text) < 40:
                    continue
                title = str(getattr(hit, "title", "") or
                            getattr(hit, "slug", "") or "Her library")
                _add(_base_item(
                    kind="teaching", title=title, theme=theme,
                    source_type="book",
                    source_ref=str(getattr(hit, "slug", "") or title),
                    source_label=f"{title} — her library",
                    passage=text[:1200]))
    except Exception:  # noqa: BLE001
        _log.debug("teachings: book producer failed", exc_info=True)
    return out


def _web_hunt(context: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """One live-net hunt via the federated search spine.  Seam for tests."""
    from ..search.federated import federated_search

    resp = federated_search(query, context=context, limit=limit)
    results = getattr(resp, "results", None) or []
    out: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict):
            out.append(r)
            continue
        out.append({
            "title": getattr(r, "title", "") or "",
            "url": getattr(r, "url", "") or "",
            "snippet": (getattr(r, "snippet", "") or
                        getattr(r, "text", "") or ""),
        })
    return out


def net_candidates(context: Any, themes: list[str]) -> list[dict[str, Any]]:
    """Candidates from the live net: trending news + hunted finds.

    Co-equal feed #2 (owner's correction): this stocks the pool
    alongside the Library — it is not a fallback for a thin Library.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item: dict[str, Any]) -> None:
        src = item["source"]
        key = item_key(src["type"], src["ref"], src["passage"])
        if key in seen or not src["passage"]:
            return
        seen.add(key)
        out.append(item)

    # (a) Trending: the news ledger's undelivered backlog (post-#85) —
    # operationally, what is hot in the owner's world right now.
    try:
        from .news import NewsAgent

        agent = NewsAgent(context)
        for row in agent.undelivered(limit=25):
            title = str(row.get("title") or "").strip()
            summary = str(row.get("summary") or "").strip()
            if not title:
                continue
            label = (str(row.get("source_name") or
                         row.get("publisher") or "news")).strip()
            _add(_base_item(
                kind="teaching", title=title, theme="",
                source_type="news",
                source_ref=str(row.get("url") or row.get("id") or title),
                source_label=label,
                passage=(summary or title)[:1200]))
    except Exception:  # noqa: BLE001
        _log.debug("teachings: news producer failed", exc_info=True)

    # (b) The hunt: question-shaped queries on the day's themes over
    # the registry-backed search class (Pass #90).
    for theme in themes[:2]:
        try:
            results = _web_hunt(
                context, f"{theme}: an idea worth knowing, explained",
                limit=5)
        except Exception:  # noqa: BLE001 - offline yields no hunt
            _log.debug("teachings: web hunt failed", exc_info=True)
            continue
        for r in results:
            title = str(r.get("title") or "").strip()
            url = str(r.get("url") or "").strip()
            snippet = str(r.get("snippet") or "").strip()
            if not url or len(snippet) < 40:
                continue
            _add(_base_item(
                kind="teaching", title=title or url, theme=theme,
                source_type="web", source_ref=url,
                source_label=title or url,
                passage=snippet[:1200]))
    return out


def memory_candidates(context: Any) -> list[dict[str, Any]]:
    """Occasional feed: the owner's own past, resurfaced (due cards)."""
    out: list[dict[str, Any]] = []
    try:
        from ..memory.repetition import RepetitionScheduler

        for card in RepetitionScheduler().due(limit=3):
            text = (getattr(card, "text", "") or "").strip()
            if len(text) < 20:
                continue
            out.append(_base_item(
                kind="teaching", title="From your past",
                theme="from your past", source_type="memory",
                source_ref=str(getattr(card, "memory_id", "") or text[:40]),
                source_label="From your past — your own notes",
                passage=text[:1200], weight=0.7))
    except Exception:  # noqa: BLE001
        _log.debug("teachings: memory producer failed", exc_info=True)
    return out


# ── the theme wheel ──────────────────────────────────────────────────

def theme_wheel(context: Any, ledger: "TeachingsLedger | None" = None,
                *, now: float | None = None) -> list[tuple[str, float]]:
    """Dynamic theme weights: seeds re-weighted by what she holds,
    what the owner pinned, and how long a theme has gone unserved.

    No static rota (owner's dynamics doctrine): a theme nobody has
    served in a week outweighs one served yesterday, corpus traditions
    join as themes, and interest pins pull their themes up.
    """
    now = time.time() if now is None else now
    weights: dict[str, float] = {t: 1.0 for t in BASE_THEMES}

    # Corpus traditions join the wheel as themes of their own.
    try:
        from ..wisdom.keeper import WisdomKeeper

        for trad in WisdomKeeper(context).traditions() or []:
            if isinstance(trad, dict):
                name = str(trad.get("tradition") or trad.get("name") or
                           trad.get("label") or "").strip().lower()
            else:
                name = str(getattr(trad, "name", "") or "").strip().lower()
            if name:
                weights.setdefault(name, 0.9)
    except Exception:  # noqa: BLE001
        _log.debug("teachings: traditions read failed", exc_info=True)

    # Library richness: recently digested traditions/key terms pull up.
    try:
        from ..wisdom.autonomy import WisdomOrgan

        for dig in WisdomOrgan(context).recent_digests(limit=6):
            trad = str(dig.get("tradition") or "").strip().lower()
            if trad and trad in weights:
                weights[trad] += 0.5
            for term in (dig.get("key_terms") or [])[:3]:
                t = str(term).strip().lower()
                if t in weights:
                    weights[t] += 0.25
    except Exception:  # noqa: BLE001
        _log.debug("teachings: digest weighting failed", exc_info=True)

    # Interest pins pull matching themes up.
    try:
        from .news import NewsAgent

        profile = NewsAgent(context).interest_profile() or {}
        pins: list[Any] = []
        for key in ("pins", "pinned", "topics", "interests"):
            val = profile.get(key)
            if isinstance(val, (list, tuple)):
                pins.extend(val)
        for pin in pins:
            p = str(pin).strip().lower()
            for theme in list(weights):
                if p and (p in theme or theme in p):
                    weights[theme] += 1.0
    except Exception:  # noqa: BLE001
        _log.debug("teachings: pin weighting failed", exc_info=True)

    # Staleness pressure: unserved themes rise (rotation by evidence,
    # not by table position).
    if ledger is not None:
        for theme in list(weights):
            last = ledger.theme_last_served(theme)
            age_days = (now - last) / 86400.0 if last else 14.0
            weights[theme] += min(2.0, max(0.0, age_days) / 3.5)

    return sorted(weights.items(), key=lambda kv: kv[1], reverse=True)


# ── the ledger ───────────────────────────────────────────────────────

class TeachingsLedger:
    """The shared store: candidate pool + served record + selection.

    One instance per context is enough; every method is fail-soft
    (returns empty/None rather than raising) so a consumer's fire path
    degrades to "no material today" instead of crashing.
    """

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        if self.db is not None:
            ensure_schema(self.db)
        # in-memory fallback (db-less contexts)
        self._mem_candidates: dict[str, tuple[dict[str, Any], float]] = {}
        self._mem_served: list[dict[str, Any]] = []

    # ── storage primitives ──
    def _candidate_rows(self) -> list[tuple[dict[str, Any], float]]:
        if self.db is not None:
            try:
                rows = self.db.query(
                    "SELECT item_json, fetched_at FROM teachings_candidates"
                ) or []
                out = []
                for r in rows:
                    try:
                        out.append((json.loads(r["item_json"]),
                                    float(r["fetched_at"])))
                    except (TypeError, ValueError):
                        continue
                return out
            except Exception:  # noqa: BLE001
                return []
        return list(self._mem_candidates.values())

    def _store_candidates(self, items: list[dict[str, Any]],
                          now: float) -> int:
        stored = 0
        for item in items:
            src = item.get("source") or {}
            key = item_key(str(src.get("type") or ""),
                           str(src.get("ref") or ""),
                           str(src.get("passage") or ""))
            item = dict(item)
            item["_key"] = key
            if self.db is not None:
                try:
                    self.db.execute(
                        "INSERT OR REPLACE INTO teachings_candidates "
                        "(item_key, item_json, fetched_at) VALUES (?, ?, ?)",
                        (key, json.dumps(item), now))
                    stored += 1
                except Exception:  # noqa: BLE001
                    _log.debug("teachings: candidate store failed",
                               exc_info=True)
            else:
                self._mem_candidates[key] = (item, now)
                stored += 1
        return stored

    def _prune(self, now: float) -> None:
        if self.db is not None:
            try:
                self.db.execute(
                    "DELETE FROM teachings_candidates WHERE fetched_at < ?",
                    (now - CANDIDATE_TTL_S,))
                self.db.execute(
                    "DELETE FROM teachings_items WHERE created_at < ?",
                    (now - PRUNE_AFTER_S,))
            except Exception:  # noqa: BLE001
                _log.debug("teachings: prune failed", exc_info=True)
        else:
            self._mem_candidates = {
                k: v for k, v in self._mem_candidates.items()
                if v[1] >= now - CANDIDATE_TTL_S}
            self._mem_served = [
                it for it in self._mem_served
                if float(it.get("created_at") or 0) >= now - PRUNE_AFTER_S]

    def served_items(self, *, since: float | None = None,
                     surface_prefix: str = "") -> list[dict[str, Any]]:
        """Served items (newest first), optionally filtered."""
        items: list[dict[str, Any]] = []
        if self.db is not None:
            try:
                rows = self.db.query(
                    "SELECT * FROM teachings_items "
                    "ORDER BY created_at DESC") or []
                for r in rows:
                    try:
                        served = json.loads(r.get("served_json") or "[]")
                    except (TypeError, ValueError):
                        served = []
                    items.append({
                        "id": r["id"], "kind": r["kind"],
                        "title": r["title"], "body": r["body"],
                        "theme": r["theme"],
                        "source": {
                            "type": r["source_type"], "ref": r["source_ref"],
                            "label": r["source_label"],
                            "passage": r["passage"]},
                        "created_at": float(r["created_at"]),
                        "served": served,
                        "_key": r.get("item_key") or "",
                    })
            except Exception:  # noqa: BLE001
                items = []
        else:
            items = [dict(it) for it in self._mem_served]
        out = []
        for it in items:
            served = it.get("served") or []
            if since is not None and not any(
                    float(s.get("at") or 0) >= since for s in served):
                continue
            if surface_prefix and not any(
                    str(s.get("surface") or "").startswith(surface_prefix)
                    for s in served):
                continue
            out.append(it)
        return out

    # ── stocking: BOTH feeds, co-equal ──
    def stock(self, *, now: float | None = None,
              themes: list[str] | None = None) -> dict[str, int]:
        """Run every producer and merge its finds into the pool.

        Library and net stock side by side on every call — neither is
        consulted only when the other runs dry.  Memory joins as an
        occasional third feed.  Returns per-source counts stocked.
        """
        now = time.time() if now is None else now
        self._prune(now)
        if themes is None:
            wheel = theme_wheel(self.context, self, now=now)
            themes = [t for t, _ in wheel[:4]] or list(BASE_THEMES[:4])
        counts: dict[str, int] = {}
        batches: list[tuple[str, list[dict[str, Any]]]] = []
        try:
            batches.append(("library", library_candidates(self.context,
                                                           themes)))
        except Exception:  # noqa: BLE001 - producers are fail-soft
            batches.append(("library", []))
        try:
            batches.append(("net", net_candidates(self.context, themes)))
        except Exception:  # noqa: BLE001
            batches.append(("net", []))
        try:
            batches.append(("memory", memory_candidates(self.context)))
        except Exception:  # noqa: BLE001
            batches.append(("memory", []))
        for name, items in batches:
            counts[name] = self._store_candidates(items, now)
        counts["pool"] = len(self._candidate_rows())
        return counts

    # ── cooldown views ──
    def _served_keys_since(self, since: float) -> set[str]:
        keys: set[str] = set()
        for it in self.served_items(since=since):
            if it.get("_key"):
                keys.add(str(it["_key"]))
            else:
                src = it.get("source") or {}
                keys.add(item_key(str(src.get("type") or ""),
                                  str(src.get("ref") or ""),
                                  str(src.get("passage") or "")))
        return keys

    def theme_last_served(self, theme: str) -> float:
        theme = (theme or "").strip().lower()
        last = 0.0
        for it in self.served_items():
            if str(it.get("theme") or "").strip().lower() != theme:
                continue
            for s in it.get("served") or []:
                last = max(last, float(s.get("at") or 0))
        return last

    def theme_in_cooldown(self, theme: str, *, now: float | None = None
                          ) -> bool:
        now = time.time() if now is None else now
        last = self.theme_last_served(theme)
        return bool(last and (now - last) < THEME_COOLDOWN_S)

    # ── selection ──
    def candidates(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """The stocked pool with the 30-day served filter applied."""
        now = time.time() if now is None else now
        served_keys = self._served_keys_since(now - ITEM_COOLDOWN_S)
        out = []
        for item, fetched_at in self._candidate_rows():
            if now - fetched_at > CANDIDATE_TTL_S:
                continue
            if item.get("_key") in served_keys:
                continue
            out.append(item)
        return out

    def eligible_themes(self, *, now: float | None = None) -> list[str]:
        """Wheel themes not inside the 48h theme cooldown, best first."""
        now = time.time() if now is None else now
        return [t for t, _ in theme_wheel(self.context, self, now=now)
                if not self.theme_in_cooldown(t, now=now)]

    def _mix_preferred(self, pool: list[dict[str, Any]], surface: str,
                       now: float) -> list[dict[str, Any]]:
        """Source-mix floors for motivation surfaces, as soft filters.

        * net pieces are owed ≥2 mornings a week: if the trailing
          week's motivation serves are short on net sources and the
          pool holds net candidates, prefer them;
        * the week may not collapse into one tradition: if every
          motivation serve this week shares a tradition and the pool
          holds another tradition, prefer the other.
        A preference that would empty the pool is dropped.
        """
        if not surface.startswith("motivation"):
            return pool
        week = self.served_items(since=now - _WEEK_S,
                                 surface_prefix="motivation")
        net_serves = sum(
            1 for it in week
            if (it.get("source") or {}).get("type") in ("web", "news"))
        if net_serves < 2:
            net_pool = [it for it in pool
                        if (it.get("source") or {}).get("type")
                        in ("web", "news")]
            if net_pool:
                return net_pool
        traditions = {str(it.get("tradition") or "")
                      for it in week if it.get("tradition")}
        if len(traditions) == 1:
            other = [it for it in pool
                     if it.get("tradition")
                     and it.get("tradition") not in traditions]
            if other:
                return other
        return pool

    def pick(self, *, surface: str, theme: str | None = None,
             now: float | None = None,
             rng: random.Random | None = None) -> dict[str, Any] | None:
        """Draw one item from the combined pool under the guarantees.

        Both cooldowns are enforced here, for every surface: a theme
        inside its 48h cooldown yields None (callers pick another
        theme); an item served inside 30 days is never re-drawn.
        Returns the candidate dict (unmarked — call :meth:`mark_served`
        once it is actually used).
        """
        now = time.time() if now is None else now
        rng = rng or random.Random()
        pool = self.candidates(now=now)
        if not pool:
            return None
        if theme is None:
            eligible = [t for t in self.eligible_themes(now=now)]
            # weighted theme choice among eligible themes that have stock
            stocked = {str(it.get("theme") or "") for it in pool}
            choices = [t for t in eligible if t in stocked]
            if choices:
                wheel = dict(theme_wheel(self.context, self, now=now))
                total = sum(max(0.1, wheel.get(t, 1.0)) for t in choices)
                roll = rng.uniform(0, total)
                theme = choices[-1]
                for t in choices:
                    roll -= max(0.1, wheel.get(t, 1.0))
                    if roll <= 0:
                        theme = t
                        break
            else:
                theme = ""  # only wildcard stock exists
        theme = (theme or "").strip().lower()
        if theme and self.theme_in_cooldown(theme, now=now):
            return None
        themed = [it for it in pool
                  if not theme
                  or str(it.get("theme") or "").strip().lower() == theme
                  or not str(it.get("theme") or "").strip()]
        if not themed:
            return None
        themed = self._mix_preferred(themed, surface, now)
        weights = [max(0.1, float(it.get("weight") or 1.0)) for it in themed]
        return rng.choices(themed, weights=weights, k=1)[0]

    def mark_served(self, item: dict[str, Any], surface: str, *,
                    body: str = "", kind: str = "",
                    now: float | None = None) -> dict[str, Any]:
        """Persist a served item (the ledger's long-lived record)."""
        now = time.time() if now is None else now
        src = item.get("source") or {}
        key = item.get("_key") or item_key(
            str(src.get("type") or ""), str(src.get("ref") or ""),
            str(src.get("passage") or ""))
        row = {
            "id": str(item.get("id") or key),
            "kind": kind or str(item.get("kind") or "teaching"),
            "title": str(item.get("title") or ""),
            "body": body or str(item.get("body") or ""),
            "theme": str(item.get("theme") or ""),
            "tradition": str(item.get("tradition") or ""),
            "source": {
                "type": str(src.get("type") or ""),
                "ref": str(src.get("ref") or ""),
                "label": str(src.get("label") or ""),
                "passage": str(src.get("passage") or ""),
            },
            "created_at": now,
            "served": [{"surface": surface, "at": now}],
            "_key": key,
        }
        if self.db is not None:
            try:
                existing = self.db.query_one(
                    "SELECT served_json FROM teachings_items WHERE id = ?",
                    (row["id"],))
                served = row["served"]
                if existing:
                    try:
                        served = (json.loads(existing.get("served_json")
                                             or "[]") + served)
                    except (TypeError, ValueError):
                        pass
                    self.db.execute(
                        "UPDATE teachings_items SET body = ?, "
                        "served_json = ? WHERE id = ?",
                        (row["body"], json.dumps(served), row["id"]))
                else:
                    self.db.execute(
                        "INSERT INTO teachings_items (id, kind, title, "
                        "body, theme, source_type, source_ref, "
                        "source_label, passage, item_key, created_at, "
                        "served_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                        "?, ?, ?, ?)",
                        (row["id"], row["kind"], row["title"], row["body"],
                         row["theme"], row["source"]["type"],
                         row["source"]["ref"], row["source"]["label"],
                         row["source"]["passage"], key, now,
                         json.dumps(served)))
                row["served"] = served
                return row
            except Exception:  # noqa: BLE001
                _log.debug("teachings: mark_served failed", exc_info=True)
        for it in self._mem_served:
            if it.get("id") == row["id"]:
                it.setdefault("served", []).extend(row["served"])
                if body:
                    it["body"] = body
                return it
        self._mem_served.append(row)
        return row

    def stats(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        week = self.served_items(since=now - _WEEK_S)
        by_source: dict[str, int] = {}
        for it in week:
            t = str((it.get("source") or {}).get("type") or "?")
            by_source[t] = by_source.get(t, 0) + 1
        return {
            "pool": len(self.candidates(now=now)),
            "served_total": len(self.served_items()),
            "served_this_week": len(week),
            "week_by_source": by_source,
        }
