"""Source trust for web sources (wave 85) — core layer.

Moved to ``nomorals.core`` so the lowest layers (the browser tool) can use
the domain ladder without importing the agents layer;
``nomorals.agents.search.trust`` re-exports it for the search engine.

Every URL the search engine returns is scored 0.0–1.0 from three
model-free signals, so a dead forum link stops outranking a working
press article:

* **domain tier** — a curated reputation ladder (government/academic →
  official project docs → established press → general web → forums and
  aggregators → low-reputation TLDs).
* **fetch feedback** — URLs that repeatedly fail to read (dead links,
  bot walls, paywalls) earn a *persistent* penalty in the kv store, so
  the learning survives restarts and a flaky source keeps sinking.
* **corroboration** — the same finding reported by more independent
  domains is worth more; callers pass how many other distinct domains
  agree.

The score is attached to every result (``trust`` + ``trust_note``) and
``ranked`` re-orders a result set by it while keeping sources within
the same trust band in their original relevance order.
"""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from .logging_setup import get_logger
from ..storage.kv import KVStore

__all__ = ["SourceTrust", "domain_tier"]

_log = get_logger(__name__)

#: (host fragment, tier, note).  First match wins; matched against the
#: hostname (exact or as a suffix, e.g. ".gov").
_DOMAIN_LADDER: tuple[tuple[str, float, str], ...] = (
    # government & academic
    (".gov.uk", 0.95, "government"),
    (".gov", 0.95, "government"),
    (".ac.uk", 0.90, "academic"),
    (".edu", 0.90, "academic"),
    ("arxiv.org", 0.90, "academic"),
    ("doi.org", 0.90, "academic"),
    ("nature.com", 0.90, "academic"),
    ("sciencedirect.com", 0.85, "academic"),
    # official project surfaces
    ("github.com", 0.85, "source / project"),
    ("gitlab.com", 0.80, "source / project"),
    ("docs.", 0.85, "official docs"),
    # established press & reference
    ("nytimes.com", 0.80, "established press"),
    ("reuters.com", 0.80, "established press"),
    ("apnews.com", 0.80, "established press"),
    ("bbc.com", 0.75, "established press"),
    ("bbc.co.uk", 0.75, "established press"),
    ("theguardian.com", 0.75, "established press"),
    ("arstechnica.com", 0.75, "tech press"),
    ("stackoverflow.com", 0.75, "established Q&A"),
    ("wikipedia.org", 0.70, "reference (crowd)"),
    ("theverge.com", 0.70, "tech press"),
    ("techcrunch.com", 0.70, "tech press"),
    ("wired.com", 0.70, "tech press"),
    # long-form blogs, forums, aggregators
    ("medium.com", 0.45, "blog / aggregator"),
    ("news.ycombinator.com", 0.45, "forum"),
    ("reddit.com", 0.40, "forum"),
    ("quora.com", 0.40, "forum"),
    ("youtube.com", 0.40, "video"),
    ("twitter.com", 0.35, "social"),
    ("x.com", 0.35, "social"),
    ("facebook.com", 0.30, "social"),
    ("tumblr.com", 0.30, "blog"),
)

_LOW_REP_TLDS = {".xyz", ".top", ".click", ".loan", ".gq", ".cf", ".tk",
                 ".zip", ".rest", ".cyou", ".quest"}

_DEFAULT_TIER = (0.55, "general web")
_UNRESOLVED = (0.30, "unresolved host")


def _hostname(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        return host
    except Exception:  # noqa: BLE001 - garbage in, low score out
        return ""


def _matches(host: str, frag: str) -> bool:
    """Ladder fragment matching:
    * ``.gov``      — TLD suffix: host == "gov" or ends with ".gov"
    * ``github.com``— domain: host == frag or ends with "." + frag
    * ``docs.``     — label prefix: host starts with "docs."
    """
    if frag.startswith("."):
        return host == frag[1:] or host.endswith(frag)
    if frag.endswith("."):
        return host.startswith(frag)
    return host == frag or host.endswith("." + frag)


def domain_tier(url: str) -> tuple[float, str]:
    """The curated reputation tier for a URL (0.0–1.0, note)."""
    host = _hostname(url)
    if not host:
        return _UNRESOLVED
    for frag, tier, note in _DOMAIN_LADDER:
        if _matches(host, frag):
            return tier, note
    if any(host.endswith(t) for t in _LOW_REP_TLDS):
        return 0.20, "low-reputation TLD"
    return _DEFAULT_TIER


class SourceTrust:
    """Domain tiers + persistent fetch learning + corroboration, with
    kv-store persistence so penalties survive restarts.

    ``allow(url)`` / ``block(url)`` pin hosts to the top / bottom of the
    scale (persisted); ``explain(url)`` shows the full score breakdown.
    """

    _KV = "search.trust.v1"
    _ALLOW_KV = "search.trust.allow.v1"
    _BLOCK_KV = "search.trust.block.v1"

    def __init__(self, context: Any) -> None:
        self.context = context
        self._cache: dict[str, tuple[int, int, float]] = {}  # host → (ok, fail, last)
        # in-memory fallback when no db is attached (persistence no-ops then)
        self._allow_mem: set[str] = set()
        self._block_mem: set[str] = set()

    # ── persistence ────────────────────────────────────────────────────────
    def _load(self) -> dict[str, dict[str, Any]]:
        db = getattr(self.context, "db", None)
        if db is None:
            return {}
        try:
            data = KVStore(db).get(self._KV)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            import json

            KVStore(db).set_raw(self._KV,
                                json.dumps(data, ensure_ascii=False), "json")
        except Exception:  # noqa: BLE001
            _log.debug("source-trust persist failed", exc_info=True)

    # ── learning ───────────────────────────────────────────────────────────
    def feedback(self, url: str, ok: bool) -> None:
        """Record a read attempt: failures sink the source, successes
        claw back a little of the tier."""
        host = _hostname(url)
        if not host:
            return
        data = self._load()
        entry = data.get(host) or {"ok": 0, "fail": 0, "last": 0.0}
        entry["ok"] = int(entry.get("ok", 0)) + (1 if ok else 0)
        entry["fail"] = int(entry.get("fail", 0)) + (0 if ok else 1)
        entry["last"] = time.time()
        # prune to the 400 most-touched hosts
        if len(data) > 400:
            keep = sorted(data.items(), key=lambda kv: -float(kv[1].get("last", 0)))[:400]
            data = dict(keep)
        data[host] = entry
        self._save(data)
        self._cache.pop(host, None)  # fresh learning, drop the stale view

    def _history(self, url: str) -> tuple[int, int]:
        host = _hostname(url)
        if not host:
            return 0, 0
        cached = self._cache.get(host)
        if cached is not None and time.time() - cached[2] < 30:
            return cached[0], cached[1]
        data = self._load()
        entry = data.get(host) or {}
        ok, fail = int(entry.get("ok", 0)), int(entry.get("fail", 0))
        self._cache[host] = (ok, fail, time.time())
        return ok, fail

    # ── scoring ───────────────────────────────────────────────────────────
    def _host_set(self, kv_key: str) -> set[str]:
        mem = self._allow_mem if kv_key == self._ALLOW_KV else self._block_mem
        db = getattr(self.context, "db", None)
        if db is None:
            return set(mem)
        try:
            data = KVStore(db).get(kv_key)
            if isinstance(data, list):
                return {str(h) for h in data} | mem
        except Exception:  # noqa: BLE001
            pass
        return set(mem)

    def _save_host_set(self, kv_key: str, hosts: set[str]) -> None:
        if kv_key == self._ALLOW_KV:
            self._allow_mem = set(hosts)
        else:
            self._block_mem = set(hosts)
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            import json

            KVStore(db).set_raw(kv_key, json.dumps(sorted(hosts)), "json")
        except Exception:  # noqa: BLE001
            _log.debug("source-trust persist failed", exc_info=True)

    # ── allow / block lists ──
    def allow(self, url: str) -> None:
        """Pin a host to full trust (also removes it from the block list)."""
        host = _hostname(url)
        if not host:
            return
        allowed = self._host_set(self._ALLOW_KV)
        allowed.add(host)
        self._save_host_set(self._ALLOW_KV, allowed)
        self.unblock(url)

    def block(self, url: str) -> None:
        """Pin a host to zero trust (also removes it from the allow list)."""
        host = _hostname(url)
        if not host:
            return
        blocked = self._host_set(self._BLOCK_KV)
        blocked.add(host)
        self._save_host_set(self._BLOCK_KV, blocked)
        self.unallow(url)

    def unallow(self, url: str) -> None:
        host = _hostname(url)
        if not host:
            return
        allowed = self._host_set(self._ALLOW_KV)
        allowed.discard(host)
        self._save_host_set(self._ALLOW_KV, allowed)

    def unblock(self, url: str) -> None:
        host = _hostname(url)
        if not host:
            return
        blocked = self._host_set(self._BLOCK_KV)
        blocked.discard(host)
        self._save_host_set(self._BLOCK_KV, blocked)

    def allowed(self) -> list[str]:
        """Pinned-trust hosts."""
        return sorted(self._host_set(self._ALLOW_KV))

    def blocked(self) -> list[str]:
        """Zero-trust hosts."""
        return sorted(self._host_set(self._BLOCK_KV))

    def is_allowed(self, url: str) -> bool:
        host = _hostname(url)
        return bool(host) and host in self._host_set(self._ALLOW_KV)

    def is_blocked(self, url: str) -> bool:
        host = _hostname(url)
        return bool(host) and host in self._host_set(self._BLOCK_KV)

    def _breakdown(self, url: str, *, corroborated_by: int = 0,
                   age_hours: float = 0.0) -> dict[str, Any]:
        tier, tier_note = domain_tier(url)
        ok, fail = self._history(url)
        learned = tier + ok * 0.02 - fail * 0.15
        learned = max(0.05, min(tier + 0.05, learned))
        corroboration = 0.05 * max(0, min(int(corroborated_by), 4))
        score = learned + corroboration
        decay = 1.0
        if age_hours > 48:
            decay = max(0.5, 1.0 - age_hours / 720.0)
            score *= decay
        pinned = None
        if self.is_blocked(url):
            pinned, score = "blocked", 0.0
        elif self.is_allowed(url):
            pinned, score = "allowed", 0.99
        return {
            "host": _hostname(url),
            "tier": tier,
            "tier_note": tier_note,
            "history": {"ok": ok, "fail": fail},
            "learned": round(learned, 3),
            "corroboration_bonus": round(corroboration, 3),
            "staleness_decay": round(decay, 3),
            "pinned": pinned,
            "score": round(max(0.0, min(0.99, score)), 3),
        }

    def score(self, url: str, *, corroborated_by: int = 0,
              age_hours: float = 0.0) -> float:
        """The 0.0–1.0 trust score for one source."""
        return self._breakdown(url, corroborated_by=corroborated_by,
                               age_hours=age_hours)["score"]

    def explain(self, url: str, *, corroborated_by: int = 0,
                age_hours: float = 0.0) -> dict[str, Any]:
        """The full score breakdown: tier, history, adjustments, pins.

        Also renders a one-line human summary under ``"summary"``.
        """
        parts = self._breakdown(url, corroborated_by=corroborated_by,
                                age_hours=age_hours)
        h = parts["history"]
        summary = (f"{parts['host'] or url}: tier {parts['tier']:.2f} "
                   f"({parts['tier_note']}), history ok={h['ok']} "
                   f"fail={h['fail']}")
        if parts["pinned"]:
            summary += f", PINNED {parts['pinned'].upper()}"
        summary += f" → {parts['score']:.2f}"
        parts["summary"] = summary
        parts["url"] = url
        return parts

    def note(self, url: str) -> str:
        tier, note = domain_tier(url)
        ok, fail = self._history(url)
        if fail >= 2:
            return f"{note} — failing source ({fail} failed reads)"
        return note

    # ── batch ──────────────────────────────────────────────────────────────
    def annotate(self, results: list[dict[str, Any]], *,
                 corroborated_by: int = 0,
                 age_hours: float = 0.0) -> list[dict[str, Any]]:
        """Attach ``trust`` + ``trust_note`` to each result (in place,
        idempotent).  Returns the same list."""
        for r in results:
            url = str(r.get("url") or "")
            if not url:
                continue
            if "trust" not in r:
                r["trust"] = self.score(url, corroborated_by=corroborated_by,
                                        age_hours=age_hours)
            r.setdefault("trust_note", self.note(url))
        return results

    def ranked(self, results: list[dict[str, Any]],
               band: float = 0.25) -> list[dict[str, Any]]:
        """Re-order by trust without destroying relevance: results within
        ``band`` of each other's trust keep their original order (stable
        sort on a *quantized* trust key)."""
        if len(results) < 2:
            return self.annotate(results)
        quant = lambda r: int(round(float(r.get("trust", 0.5)) / band)) * band
        return sorted(self.annotate(results), key=lambda r: -quant(r))
