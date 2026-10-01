"""Watchers — general background monitors with smart alerting.

The owner says "watch X for Y and tell me when Z" in plain language; Devon
creates a durable watcher, checks it on a schedule, and alerts ONLY when
something worth knowing happens — no spam, no alert storms, full history.

**Improve-vs-build decision (Prompt 03, 2026-10-01)** — surveyed the existing
monitoring bones first, per standing rule:

* **IMPROVED (not duplicated):** ``MonitorAgent``'s fetching/hashing logic was
  extracted to module-level functions in ``nomorals/agents/monitor.py``
  (``fetch_url_bytes`` / ``fetch_file_bytes`` / ``fetch_page_bytes`` /
  ``hash_bytes`` / ``unified_content_diff``); the ``url`` and ``file`` watcher
  kinds call those directly, and ``MonitorAgent``'s public API is unchanged.
* **REUSED as-is:** ``Notifier`` — every watcher alert goes through
  ``Notifier.publish`` (durable row first, then send); no second sending path.
  ``Scheduler`` — watchers run as scheduled work through it (see below).
  Prompt 02's ``check_spec_call`` — ``condition``-kind checks are enforced
  through a read-only ``RoleSpec``, so a watcher check can never invoke a
  write/mutating tool.
* **BUILT NEW (this module):** the general watcher model (``Watcher``,
  six kind checkers, structured conditions, ``WatchResult``), the smart
  alerting layer (cooldowns, digest batching, quiet hours, flap suppression),
  natural-language creation (``WatcherAgent.parse``), watcher history and the
  alert audit log.

**Scheduler: one sweeper job, not one job per watcher.** A single
``watchers.sweep`` cron job (every minute) checks every due watcher.  N jobs
would mean N cron rows and N wake-ups for the same work; the sweeper is one
row, one wake-up, and it checks only due watchers.  Restart recovery is
trivial because all state lives in the DB.

**DB: a new ``watchers`` table (migration 54), not an extension of
``monitors``.** The ``monitors`` table is narrowly shaped for MonitorAgent's
URL/file/page model (``webhook_url``, ``volatile``, ``auto_decode``, content
vs size).  Shoehorning six watcher kinds plus structured conditions, severity,
channels, cooldowns and flap state into it would leave a dozen nullable
columns and confuse both systems; ``MonitorAgent`` keeps working untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ids import new_id, new_short_id
from ..core.logging_setup import get_logger

__all__ = [
    "Watcher",
    "WatchResult",
    "Condition",
    "WatcherAgent",
    "WatcherStore",
    "AlertEngine",
    "WatcherContext",
    "parse_condition",
    "evaluate_condition",
    "WATCHER_KINDS",
    "WATCHER_READONLY_SPEC",
    "register",
    "ensure_sweeper_job",
]

_log = get_logger(__name__)

#: watcher kinds; each maps to a checker class below
WATCHER_KINDS = ("url", "file", "price", "repo", "keyword", "condition")
#: severity levels, in escalating order
SEVERITIES = ("info", "important", "urgent")
_SEVERITY_RANK = {"info": 0, "important": 1, "urgent": 2}
#: watcher states
WATCHER_STATES = ("active", "paused", "error", "expired")

#: structured-condition operators
CONDITION_OPS = (
    "eq", "ne", "lt", "lte", "gt", "gte",
    "changed", "changed_by_pct", "contains", "matches",
)

#: default check intervals per kind (seconds)
DEFAULT_INTERVALS = {
    "url": 300.0, "file": 300.0, "price": 3600.0,
    "repo": 3600.0, "keyword": 3600.0, "condition": 900.0,
}
#: cooldown multiplier when the watcher does not set one explicitly
DEFAULT_COOLDOWN_MULT = 6.0
#: info-severity hits are batched into one digest per this window
DEFAULT_DIGEST_WINDOW_S = 6 * 3600.0
#: flap suppression: this many changed/unchanged transitions inside the
#: window auto-pauses the watcher with a single notice
FLAP_TRANSITIONS = 4
FLAP_WINDOW_S = 3600.0
#: check history kept per watcher
HISTORY_CAP = 50
#: consecutive check errors before the single "watcher is erroring" notice
_ERROR_NOTICE_STREAK = 3
#: per-host minimum gap between URL/repo/keyword polls (rate-limit respect)
PER_HOST_MIN_GAP_S = 60.0


# ── structured conditions ────────────────────────────────────────────────────

@dataclass
class Condition:
    """A structured predicate, parsed ONCE at creation and evaluated
    deterministically on every tick.  The model never re-interprets it."""
    op: str
    field: str = "value"
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "field": self.field, "value": self.value}


def parse_condition(data: Any) -> Condition:
    """Parse a stored/plain condition dict into a :class:`Condition`.

    ``None``/empty → ``changed`` on the whole value (the classic "tell me
    when this changes" watcher).  Raises :class:`ValueError` on bad input —
    fail fast at creation, never at tick time.
    """
    if not data:
        return Condition(op="changed", field="value", value=None)
    if isinstance(data, Condition):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"condition must be a dict, got {type(data).__name__}")
    op = str(data.get("op", "changed")).strip().lower()
    if op not in CONDITION_OPS:
        raise ValueError(
            f"unknown condition op {op!r}; expected one of {CONDITION_OPS}")
    return Condition(op=op, field=str(data.get("field", "value")),
                     value=data.get("value"))


def _as_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("₦", "").replace("$", "") \
            .replace("€", "").replace("£", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _resolve_operand(record: Any, field: str) -> Any:
    """Pull ``field`` out of a check value; ``field == "value"`` means the
    whole value."""
    if field == "value":
        return record
    if isinstance(record, dict):
        return record.get(field)
    return None


def evaluate_condition(cond: Condition, *, old: Any,
                       new: Any) -> tuple[bool, str]:
    """Evaluate a structured condition against old/new check values.

    Returns ``(triggered, note)``.  Threshold-style ops (``lt``/``gt``/…)
    evaluate on the first check too — if the price is *already* below the
    threshold, the owner wants to know now.  ``changed``/``changed_by_pct``
    need a baseline, so the first check only establishes it.
    """
    op = cond.op
    current = _resolve_operand(new, cond.field)
    previous = _resolve_operand(old, cond.field)

    if op in ("changed", "changed_by_pct"):
        if old is None:
            return False, "baseline established"
        if op == "changed":
            hit = previous != current
            return hit, (f"{cond.field} changed" if hit
                         else f"{cond.field} unchanged")
        # changed_by_pct
        po, co = _as_number(previous), _as_number(current)
        if po is None or co is None or po == 0:
            return False, "changed_by_pct needs numeric non-zero baseline"
        pct = abs(co - po) / abs(po) * 100.0
        thresh = _as_number(cond.value)
        if thresh is None:
            return False, "changed_by_pct needs a numeric threshold"
        hit = pct >= thresh
        return hit, (f"{cond.field} moved {pct:.1f}% (≥ {thresh}%)" if hit
                     else f"{cond.field} moved {pct:.1f}%")

    if op in ("lt", "lte", "gt", "gte", "eq", "ne"):
        cn, vn = _as_number(current), _as_number(cond.value)
        if cn is not None and vn is not None:
            hit = {"lt": cn < vn, "lte": cn <= vn, "gt": cn > vn,
                   "gte": cn >= vn, "eq": cn == vn, "ne": cn != vn}[op]
            return hit, f"{cond.field}={cn:g} {op} {vn:g}" if hit else \
                f"{cond.field}={cn:g} not {op} {vn:g}"
        # string fallback for eq/ne only
        if op in ("eq", "ne"):
            hit = (str(current) == str(cond.value)) == (op == "eq")
            return hit, f"{cond.field} {op} {cond.value!r}"
        return False, f"{op} needs numeric values"

    if op == "contains":
        if isinstance(current, (list, tuple, set)):
            hit = cond.value in current
        else:
            hit = str(cond.value) in str(current)
        return hit, (f"{cond.field} contains {cond.value!r}" if hit
                     else f"{cond.field} does not contain {cond.value!r}")

    if op == "matches":
        try:
            hit = re.search(str(cond.value), str(current)) is not None
        except re.error:
            return False, f"bad regex {cond.value!r}"
        return hit, (f"{cond.field} matches {cond.value!r}" if hit
                     else f"{cond.field} does not match {cond.value!r}")

    return False, f"unreachable op {op!r}"  # pragma: no cover


# ── the watcher model ────────────────────────────────────────────────────────

@dataclass
class WatchResult:
    """One check of one watcher."""
    changed: bool
    old_value: Any = None
    new_value: Any = None
    summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class Watcher:
    id: str
    name: str
    kind: str
    target: dict[str, Any] = field(default_factory=dict)
    condition: Condition = field(default_factory=lambda: Condition(op="changed"))
    interval_s: float = 3600.0
    cooldown_s: float = 0.0  # 0 → default: interval_s * 6
    severity: str = "info"
    channels: list[str] = field(default_factory=list)
    quiet_hours: dict[str, Any] = field(default_factory=dict)
    expires_at: float = 0.0
    state: str = "active"
    last_check: float = 0.0
    last_value: Any = None
    last_alert_ts: float = 0.0
    last_alert_severity: str = ""
    error_streak: int = 0
    last_changed: bool = False
    flip_times: list[float] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def cooldown(self) -> float:
        return self.cooldown_s if self.cooldown_s > 0 \
            else self.interval_s * DEFAULT_COOLDOWN_MULT

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "target": json.dumps(self.target or {}),
            "condition": json.dumps(self.condition.to_dict()),
            "interval_s": self.interval_s,
            "cooldown_s": self.cooldown_s,
            "severity": self.severity,
            "channels": json.dumps(self.channels or []),
            "quiet_hours": json.dumps(self.quiet_hours or {}),
            "expires_at": self.expires_at,
            "state": self.state,
            "last_check": self.last_check,
            "last_value": json.dumps(self.last_value),
            "last_alert_ts": self.last_alert_ts,
            "last_alert_severity": self.last_alert_severity,
            "error_streak": self.error_streak,
            "last_changed": 1 if self.last_changed else 0,
            "flip_times": json.dumps(self.flip_times or []),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def watcher_from_row(row: dict[str, Any]) -> Watcher:
    def _j(key: str, default: Any) -> Any:
        try:
            return json.loads(row.get(key) or "") if row.get(key) else default
        except (ValueError, TypeError):
            return default

    return Watcher(
        id=row["id"],
        name=row.get("name", "") or "",
        kind=row.get("kind", "") or "",
        target=_j("target", {}),
        condition=parse_condition(_j("condition", {})),
        interval_s=float(row.get("interval_s") or 3600.0),
        cooldown_s=float(row.get("cooldown_s") or 0.0),
        severity=row.get("severity", "info") or "info",
        channels=_j("channels", []),
        quiet_hours=_j("quiet_hours", {}),
        expires_at=float(row.get("expires_at") or 0.0),
        state=row.get("state", "active") or "active",
        last_check=float(row.get("last_check") or 0.0),
        last_value=_j("last_value", None),
        last_alert_ts=float(row.get("last_alert_ts") or 0.0),
        last_alert_severity=row.get("last_alert_severity", "") or "",
        error_streak=int(row.get("error_streak") or 0),
        last_changed=bool(row.get("last_changed", 0)),
        flip_times=_j("flip_times", []),
        created_at=float(row.get("created_at") or 0.0),
        updated_at=float(row.get("updated_at") or 0.0),
    )

# ── read-only enforcement for watcher checks ─────────────────────────────────
# Watchers are READ-ONLY by construction: every ``condition``-kind tool call
# (and the ``price``/``keyword`` kinds, which also go through the registry)
# is checked against this locked spec BEFORE dispatch, using Prompt 02's
# shared ``check_spec_call``.  A malicious watcher target like
# ``{tool: "shell_run", ...}`` is denied and the denial is recorded.


def _watcher_readonly_spec():  # lazy: role_specs is heavyweight to import
    from .role_specs import RoleSpec

    return RoleSpec(
        name="watcher",
        description="Read-only watcher checks. May observe, never mutate.",
        system_prompt="You are a watcher check. Observe only; never mutate.",
        tool_allowlist=(
            # read-only observation tools a watcher check may use
            "web_search", "web_fetch", "deals", "monitor",
            "api_call", "api_list", "fs_read", "http_get",
        ),
        read_only=True,
    )


WATCHER_READONLY_SPEC = None  # replaced by _watcher_readonly_spec() on first use


def _read_only_spec():
    global WATCHER_READONLY_SPEC
    if WATCHER_READONLY_SPEC is None:
        WATCHER_READONLY_SPEC = _watcher_readonly_spec()
    return WATCHER_READONLY_SPEC


def _deny_unless_read_only(context: Any, tool_name: str,
                           call_kwargs: dict[str, Any]) -> str:
    """Return '' when the tool call is allowed, else the denial reason.

    Two layers: the locked allowlist, then Prompt 02's read-only
    mutating-tool defense (catches a mutating tool even if it were ever
    allowlisted by mistake).  Denials are recorded through the shared
    ``record_tool_denial`` telemetry.
    """
    from .role_specs import check_spec_call, record_tool_denial

    denied = check_spec_call(_read_only_spec(), tool_name, call_kwargs)
    if denied is not None:
        record_tool_denial(context, "watcher", tool_name,
                           denied.reason, denied.message)
        return denied.message
    return ""


class _HostRateLimited(Exception):
    """Raised when a per-host minimum poll gap would be violated."""


_ROBOTS_CACHE: Any = None


def _robots_cache_singleton() -> Any:
    """Shared robots.txt cache (10-min TTL) — the same class the web tools
    use. Lazily created so importing watchers stays cheap."""
    from ..tools.web import RobotsCache

    global _ROBOTS_CACHE
    if _ROBOTS_CACHE is None:
        _ROBOTS_CACHE = RobotsCache()
    return _ROBOTS_CACHE


class _LazyRobots:
    """Module-level ``_robots_cache`` proxy: ``_robots_cache.allowed(url)``."""

    def allowed(self, url: str, user_agent: str = "*",
                client: Any = None) -> bool:
        return _robots_cache_singleton().allowed(url, user_agent, client)


_robots_cache = _LazyRobots()


@dataclass
class WatcherContext:
    """What a kind checker needs: the app context, the tool registry, and a
    clock override for tests."""
    context: Any
    registry: Any = None
    store: Any = None
    now: Callable[[], float] = time.time
    _host_last_fetch: dict[str, float] = field(default_factory=dict)

    @property
    def db(self) -> Any:
        return getattr(self.context, "db", None)

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a registry tool through the read-only gate.

        Raises :class:`PermissionError` when the watcher spec denies the
        call, :class:`RuntimeError` when the tool is missing or fails.
        """
        reason = _deny_unless_read_only(self.context, tool_name, kwargs)
        if reason:
            raise PermissionError(f"watcher check denied: {reason}")
        registry = self.registry
        if registry is None:
            raise RuntimeError("no tool registry available for watcher check")
        outcome = registry.call(tool_name, **kwargs)
        if not outcome.ok:
            raise RuntimeError(f"tool {tool_name!r} failed: {outcome.error}")
        return outcome.value

    def check_host_gap(self, host: str) -> None:
        """Enforce the per-host minimum poll interval; raises
        :class:`_HostRateLimited` when the gap would be violated so the
        sweeper can defer this watcher to the next sweep instead of
        hammering the host.

        The last-fetch timestamp is durable (``watcher_state``) so the
        minimum interval holds across sweeps and restarts; the in-memory
        dict additionally collapses duplicate hosts within one sweep.
        """
        host = (host or "").lower()
        if not host:
            return
        now = self.now()
        last = self._host_last_fetch.get(host)
        if last is None and self.store is not None:
            raw = self.store.get_state(f"hostfetch:{host}")
            try:
                last = float(raw) if raw else 0.0
            except (TypeError, ValueError):
                last = 0.0
        last = last or 0.0
        if now - last < PER_HOST_MIN_GAP_S:
            raise _HostRateLimited(
                f"host {host!r} polled {now - last:.0f}s ago "
                f"(min gap {PER_HOST_MIN_GAP_S:.0f}s)")
        self._host_last_fetch[host] = now
        if self.store is not None:
            self.store.set_state(f"hostfetch:{host}", str(now))


def _semantic_hash(body: bytes) -> tuple[str, bool]:
    """Content-aware hash of an HTML body: strip scripts/styles/boilerplate
    via the existing DOM → markdown renderer, then hash the visible text.

    Returns ``(hex_digest, semantic_ok)``; ``semantic_ok`` is False when
    extraction failed and the caller should fall back to the raw hash.
    """
    try:
        from ..tools.browser import parse_html, node_to_markdown

        text = body.decode("utf-8", "ignore")
        if not text.strip():
            return "", False
        rendered = node_to_markdown(parse_html(text))
        lines = [ln.rstrip() for ln in rendered.splitlines() if ln.strip()]
        tight = "\n".join(lines)
        if not tight:
            return "", False
        return hashlib.sha256(tight.encode("utf-8")).hexdigest(), True
    except Exception:  # noqa: BLE001 — fall back to raw hash
        return "", False


class _KindChecker:
    """One watcher kind: ``check(wctx, watcher) -> WatchResult``."""
    name = ""

    def check(self, wctx: WatcherContext,
              watcher: Watcher) -> WatchResult:  # pragma: no cover
        raise NotImplementedError

    def describe(self, watcher: Watcher) -> str:
        return f"{self.name} watcher"


class UrlKind(_KindChecker):
    """Watch a URL's *visible content* (semantic hash, boilerplate stripped),
    falling back to the raw-body hash when extraction fails."""

    name = "url"

    def check(self, wctx: WatcherContext, watcher: Watcher) -> WatchResult:
        from .monitor import fetch_url_bytes, hash_bytes
        from ..tools.web import RobotsCache

        url = (watcher.target.get("url") or "").strip()
        if not url:
            return WatchResult(changed=False, error="url watcher has no target url")
        # respect robots.txt (shared RobotsCache, 10-min TTL — the same one
        # the web tools use). A disallowed URL is *skipped*, not errored:
        # erroring would trip the flaky-watcher quiet logic for a policy
        # decision, and the skip is visible in history.
        if not _robots_cache.allowed(url):
            return WatchResult(
                changed=False, old_value=watcher.last_value,
                new_value=watcher.last_value,
                summary=f"skipped: robots.txt disallows {url}",
                evidence={"url": url, "robots_skipped": True})
        host = re.sub(r"^https?://", "", url).split("/")[0]
        wctx.check_host_gap(host)
        body = fetch_url_bytes(url)
        sem_hash, semantic = _semantic_hash(body)
        digest = sem_hash if semantic else hash_bytes(body)
        new_value = {"hash": digest, "bytes": len(body), "semantic": semantic}
        old = watcher.last_value if isinstance(watcher.last_value, dict) else None
        if old is None:
            return WatchResult(changed=False, old_value=None,
                               new_value=new_value,
                               summary="baseline established",
                               evidence={"url": url})
        changed = old.get("hash") != digest
        delta = len(body) - int(old.get("bytes") or 0)
        summary = (f"content changed at {url}"
                   + (f" ({delta:+,d} bytes)" if delta else "")) \
            if changed else "no change"
        return WatchResult(
            changed=changed, old_value=old, new_value=new_value,
            summary=summary,
            evidence={"url": url, "old_hash": old.get("hash"),
                      "new_hash": digest, "byte_delta": delta,
                      "semantic": semantic})

    def describe(self, watcher: Watcher) -> str:
        return f"URL content at {watcher.target.get('url', '?')}"


class FileKind(_KindChecker):
    """Watch a workspace file's bytes (hash comparison)."""

    name = "file"

    def check(self, wctx: WatcherContext, watcher: Watcher) -> WatchResult:
        from .monitor import fetch_file_bytes, hash_bytes

        path = (watcher.target.get("path") or "").strip()
        if not path:
            return WatchResult(changed=False, error="file watcher has no target path")
        body = fetch_file_bytes(wctx.context, path)
        digest = hash_bytes(body)
        new_value = {"hash": digest, "bytes": len(body)}
        old = watcher.last_value if isinstance(watcher.last_value, dict) else None
        if old is None:
            return WatchResult(changed=False, old_value=None,
                               new_value=new_value,
                               summary="baseline established",
                               evidence={"path": path})
        changed = old.get("hash") != digest
        return WatchResult(
            changed=changed, old_value=old, new_value=new_value,
            summary=(f"{path} changed" if changed else "no change"),
            evidence={"path": path, "old_hash": old.get("hash"),
                      "new_hash": digest})

    def describe(self, watcher: Watcher) -> str:
        return f"file {watcher.target.get('path', '?')}"


def _extract_price_number(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        m = re.search(r"[\d][\d,]*\.?\d*", raw.replace("₦", "").replace("$", ""))
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


def _iter_products(result: Any) -> list[dict[str, Any]]:
    """Best-effort product list extraction from a ``deals`` tool result."""
    if isinstance(result, dict):
        for key in ("products", "results", "items", "deals"):
            items = result.get(key)
            if isinstance(items, list) and items:
                return [i for i in items if isinstance(i, dict)]
        # single-product shape
        if "price" in result:
            return [result]
    if isinstance(result, list):
        return [i for i in result if isinstance(i, dict)]
    return []


class PriceKind(_KindChecker):
    """Watch a price and fire when the structured condition triggers.

    Two sources, both read-only:
    * ``deals`` — a product query on Nigerian marketplaces (Jumia, Konga,
      Jiji…) via the ``deals`` tool's ``scan`` action (never the mutating
      watchlist/track actions).
    * ``market`` — a crypto/fiat symbol via CoinGecko's keyless
      ``simple/price`` endpoint through the proxy-aware HttpClient (the same
      pattern ``payment_integration.get_price`` already uses in-repo);
      stocks via Yahoo and fiat FX via Frankfurter (both keyless) when the
      watcher target sets ``market`` to ``stocks``/``forex``.
    """

    name = "price"

    #: common ticker → CoinGecko id
    _SYMBOL_MAP = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
        "DOGE": "dogecoin", "TON": "the-open-network", "TRX": "tron",
        "USDT": "tether", "USDC": "usd-coin",
    }
    _CURRENCY_MAP = {"$": "usd", "₦": "ngn", "€": "eur", "£": "gbp"}

    def check(self, wctx: WatcherContext, watcher: Watcher) -> WatchResult:
        target = watcher.target or {}
        source = (target.get("source") or "deals").strip().lower()
        if source == "market":
            return self._check_market(wctx, watcher)
        return self._check_deals(wctx, watcher)

    def _check_deals(self, wctx: WatcherContext,
                     watcher: Watcher) -> WatchResult:
        target = watcher.target or {}
        query = (target.get("query") or "").strip()
        if not query:
            return WatchResult(changed=False,
                               error="price watcher needs a product query")
        result = wctx.call_tool("deals", action="scan", query=query)
        products = _iter_products(result)
        priced = [(p, _extract_price_number(p.get("price")))
                  for p in products]
        priced = [(p, v) for p, v in priced if v is not None]
        if not priced:
            return WatchResult(changed=False,
                               error=f"no priced results for {query!r}",
                               evidence={"query": query})
        best, price = min(priced, key=lambda pv: pv[1])
        title = str(best.get("title") or best.get("name") or query)
        new_value = {"price": price, "currency": "NGN", "title": title,
                     "url": str(best.get("url") or best.get("link") or "")}
        cond = watcher.condition
        if cond.field == "value":
            # bare threshold on the price itself
            old_price = (watcher.last_value or {}).get("price") \
                if isinstance(watcher.last_value, dict) else None
            triggered, note = evaluate_condition(cond, old=old_price,
                                                new=price)
        else:
            triggered, note = evaluate_condition(cond, old=watcher.last_value,
                                                new=new_value)
        summary = f"₦{price:,.0f} · {title[:70]} — {note}"
        return WatchResult(
            changed=triggered, old_value=watcher.last_value,
            new_value=new_value, summary=summary,
            evidence={"query": query, "product": best})

    def _check_market(self, wctx: WatcherContext,
                      watcher: Watcher) -> WatchResult:
        from ..core.http import HttpClient

        target = watcher.target or {}
        symbol = (target.get("symbol") or "").strip().upper()
        if not symbol:
            return WatchResult(changed=False,
                               error="market price watcher needs a symbol")
        market = (target.get("market") or "crypto").strip().lower()
        if market in ("stocks", "forex"):
            return self._check_market_free(wctx, watcher, market, symbol)
        coin_id = self._SYMBOL_MAP.get(symbol, symbol.lower())
        currency = (target.get("currency") or "usd").strip().lower()
        wctx.check_host_gap("api.coingecko.com")
        resp = HttpClient().get(
            "https://api.coingecko.com/api/v3/simple/price"
            f"?ids={coin_id}&vs_currencies={currency}",
            headers={"User-Agent": "nomorals-watcher/1.0"})
        status = int(getattr(resp, "status", 0) or 0)
        if status == 429:
            raise _HostRateLimited("coingecko rate limit (429)")
        if status >= 400:
            return WatchResult(changed=False,
                               error=f"price API HTTP {status}")
        body = getattr(resp, "body", b"") or b""
        try:
            data = json.loads(body.decode("utf-8", "ignore") or "{}")
            price = float(data[coin_id][currency])
        except Exception:  # noqa: BLE001 — bad payload → error streak
            return WatchResult(changed=False,
                               error=f"no {currency} quote for {symbol}")
        new_value = {"price": price, "currency": currency.upper(),
                     "symbol": symbol}
        cond = watcher.condition
        if cond.field == "value":
            old_price = (watcher.last_value or {}).get("price") \
                if isinstance(watcher.last_value, dict) else None
            triggered, note = evaluate_condition(cond, old=old_price,
                                                new=price)
        else:
            triggered, note = evaluate_condition(cond, old=watcher.last_value,
                                                new=new_value)
        sym = "$" if currency == "usd" else (currency.upper() + " ")
        summary = f"{symbol} {sym}{price:,.2f} — {note}"
        return WatchResult(changed=triggered, old_value=watcher.last_value,
                           new_value=new_value, summary=summary,
                           evidence={"symbol": symbol})

    def _check_market_free(self, wctx: WatcherContext, watcher: Watcher,
                           market: str, symbol: str) -> WatchResult:
        """Stocks via Yahoo / fiat FX via Frankfurter — both keyless.

        Same condition semantics as the crypto path; the quote comes from
        :mod:`nomorals.integrations.market_data` (stdlib-only, no pandas).
        """
        from ..integrations import market_data

        host = "stooq.com" if market == "stocks" else "api.frankfurter.dev"
        wctx.check_host_gap(host)
        try:
            q = market_data.quote(symbol, market=market)
            price = float(q.get("price"))
        except Exception as exc:  # noqa: BLE001 — bad payload → error streak
            return WatchResult(changed=False, error=str(exc)[:160])
        currency = str(q.get("currency") or "USD").upper()
        new_value = {"price": price, "currency": currency,
                     "symbol": symbol, "market": market}
        cond = watcher.condition
        if cond.field == "value":
            old_price = (watcher.last_value or {}).get("price") \
                if isinstance(watcher.last_value, dict) else None
            triggered, note = evaluate_condition(cond, old=old_price,
                                                new=price)
        else:
            triggered, note = evaluate_condition(cond, old=watcher.last_value,
                                                new=new_value)
        summary = f"{symbol} {currency} {price:,.4g} — {note}"
        return WatchResult(changed=triggered, old_value=watcher.last_value,
                           new_value=new_value, summary=summary,
                           evidence={"symbol": symbol, "market": market})

    def describe(self, watcher: Watcher) -> str:
        t = watcher.target or {}
        if (t.get("source") or "") == "market":
            return f"market price of {t.get('symbol', '?')}"
        return f"price of {t.get('query', '?')!r}"


class _GitHubMixin:
    def _gh_get(self, wctx: WatcherContext, path: str) -> Any:
        """GET the GitHub REST API through the proxy-aware HttpClient (the
        in-repo client — never shelling out to ``gh``)."""
        import os

        from ..core.http import HttpClient

        host = "api.github.com"
        wctx.check_host_gap(host)
        headers = {"Accept": "application/vnd.github+json",
                   "User-Agent": "nomorals-watcher/1.0"}
        token = os.environ.get("NM_API_GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = HttpClient().get(f"https://{host}{path}", headers=headers)
        status = int(getattr(resp, "status", 0) or 0)
        if status == 403:
            raise RuntimeError("GitHub API rate limit hit (403)")
        if status >= 400:
            raise RuntimeError(f"GitHub API HTTP {status}")
        body = getattr(resp, "body", b"") or b""
        if not body and getattr(resp, "text", ""):
            body = resp.text.encode("utf-8", "ignore")
        return json.loads(body.decode("utf-8", "ignore") or "[]")


class RepoKind(_GitHubMixin, _KindChecker):
    """Watch a GitHub repo for new issues / PRs / releases matching a filter."""

    name = "repo"
    _FEED_PATHS = {"issues": "issues", "prs": "pulls", "releases": "releases"}

    def check(self, wctx: WatcherContext, watcher: Watcher) -> WatchResult:
        target = watcher.target or {}
        owner = (target.get("owner") or "").strip()
        repo = (target.get("repo") or "").strip()
        feed = (target.get("feed") or "releases").strip().lower()
        if not owner or not repo:
            return WatchResult(changed=False,
                               error="repo watcher needs owner and repo")
        if feed not in self._FEED_PATHS:
            return WatchResult(changed=False,
                               error=f"unknown repo feed {feed!r}")
        path = (f"/repos/{owner}/{repo}/{self._FEED_PATHS[feed]}"
                f"?per_page=20&sort=created&direction=desc")
        if feed != "releases":
            path += "&state=open"
        items = self._gh_get(wctx, path)
        if not isinstance(items, list):
            return WatchResult(changed=False,
                               error="unexpected GitHub API response")
        labels = {str(l_).lower() for l_ in (target.get("labels") or [])}
        seen: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            if feed == "issues" and "pull_request" in it:
                continue  # the issues endpoint also returns PRs
            if labels:
                it_labels = {str(l.get("name", "")).lower()
                             for l in (it.get("labels") or [])
                             if isinstance(l, dict)}
                if not (labels & it_labels):
                    continue
            seen.append({
                "id": str(it.get("id") or it.get("node_id") or it.get("tag_name")),
                "title": str(it.get("title") or it.get("name") or ""),
                "url": str(it.get("html_url") or ""),
            })
        old = watcher.last_value if isinstance(watcher.last_value, dict) else None
        old_ids = set(old.get("ids", [])) if old else set()
        fresh = [s for s in seen if s["id"] not in old_ids]
        new_value = {"ids": [s["id"] for s in seen], "count": len(seen)}
        if old is None:
            return WatchResult(changed=False, old_value=None,
                               new_value=new_value,
                               summary=f"baseline: {len(seen)} {feed}",
                               evidence={"owner": owner, "repo": repo})
        titles = ", ".join(s["title"][:50] for s in fresh[:3])
        summary = (f"{len(fresh)} new {feed} in {owner}/{repo}: {titles}"
                   if fresh else f"no new {feed} in {owner}/{repo}")
        return WatchResult(changed=bool(fresh), old_value=old,
                           new_value=new_value, summary=summary,
                           evidence={"owner": owner, "repo": repo,
                                     "items": fresh})

    def describe(self, watcher: Watcher) -> str:
        t = watcher.target or {}
        return (f"github {t.get('feed', 'releases')} in "
                f"{t.get('owner', '?')}/{t.get('repo', '?')}")


class KeywordKind(_KindChecker):
    """Watch a search/news source for new results matching keywords."""

    name = "keyword"

    def check(self, wctx: WatcherContext, watcher: Watcher) -> WatchResult:
        target = watcher.target or {}
        keywords = (target.get("keywords") or "").strip()
        if not keywords:
            return WatchResult(changed=False,
                               error="keyword watcher needs keywords")
        result = wctx.call_tool("web_search", query=keywords)
        links: list[dict[str, str]] = []
        items = result if isinstance(result, list) else (
            result.get("results") if isinstance(result, dict) else [])
        for it in (items or []):
            if not isinstance(it, dict):
                continue
            url = str(it.get("url") or it.get("link") or "")
            if url:
                links.append({"url": url,
                              "title": str(it.get("title") or "")})
        # de-dupe by URL, keep order
        uniq, seen_urls = [], set()
        for l_ in links:
            if l_["url"] not in seen_urls:
                seen_urls.add(l_["url"])
                uniq.append(l_)
        old = watcher.last_value if isinstance(watcher.last_value, dict) else None
        old_urls = set(old.get("urls", [])) if old else set()
        fresh = [l_ for l_ in uniq if l_["url"] not in old_urls]
        new_value = {"urls": [l_["url"] for l_ in uniq],
                     "count": len(uniq)}
        if old is None:
            return WatchResult(changed=False, old_value=None,
                               new_value=new_value,
                               summary=f"baseline: {len(uniq)} results",
                               evidence={"keywords": keywords})
        titles = ", ".join(l_["title"][:50] for l_ in fresh[:3])
        summary = (f"{len(fresh)} new results for {keywords!r}: {titles}"
                   if fresh else f"no new results for {keywords!r}")
        return WatchResult(changed=bool(fresh), old_value=old,
                           new_value=new_value, summary=summary,
                           evidence={"keywords": keywords, "items": fresh})

    def describe(self, watcher: Watcher) -> str:
        return f"new results for {watcher.target.get('keywords', '?')!r}"


class ConditionKind(_KindChecker):
    """Run an arbitrary READ-ONLY tool query and evaluate the structured
    predicate on the result.  The tool call goes through the read-only gate
    (allowlist + mutating-tool defense); anything else is denied and logged."""

    name = "condition"

    def check(self, wctx: WatcherContext, watcher: Watcher) -> WatchResult:
        target = watcher.target or {}
        tool_name = (target.get("tool") or "").strip()
        args = target.get("args") or {}
        if not tool_name:
            return WatchResult(changed=False,
                               error="condition watcher needs a tool")
        if not isinstance(args, dict):
            return WatchResult(changed=False,
                               error="condition watcher args must be a dict")
        try:
            result = wctx.call_tool(tool_name, **args)
        except PermissionError as exc:
            return WatchResult(changed=False, error=str(exc),
                               evidence={"tool": tool_name, "denied": True})
        except Exception as exc:  # noqa: BLE001 — tool failure → error streak
            return WatchResult(changed=False, error=f"{tool_name}: {exc}",
                               evidence={"tool": tool_name})
        cond = watcher.condition
        triggered, note = evaluate_condition(cond, old=watcher.last_value,
                                            new=result)
        summary = f"{tool_name}: {note}"
        if triggered and isinstance(result, dict):
            summary += f" — {json.dumps(result)[:120]}"
        return WatchResult(changed=triggered, old_value=watcher.last_value,
                           new_value=result, summary=summary,
                           evidence={"tool": tool_name, "note": note})

    def describe(self, watcher: Watcher) -> str:
        t = watcher.target or {}
        return f"condition on {t.get('tool', '?')}"


KIND_CHECKERS: dict[str, _KindChecker] = {
    "url": UrlKind(),
    "file": FileKind(),
    "price": PriceKind(),
    "repo": RepoKind(),
    "keyword": KeywordKind(),
    "condition": ConditionKind(),
}

# ── persistence ──────────────────────────────────────────────────────────────

_WATCHER_COLUMNS = (
    "id", "name", "kind", "target", "condition", "interval_s", "cooldown_s",
    "severity", "channels", "quiet_hours", "expires_at", "state",
    "last_check", "last_value", "last_alert_ts", "last_alert_severity",
    "error_streak", "last_changed", "flip_times", "created_at", "updated_at",
)


class WatcherStore:
    """DB persistence for watchers, check history, and the alert audit log."""

    def __init__(self, db: Any) -> None:
        if db is None:
            raise RuntimeError("WatcherStore needs a database")
        self.db = db

    # -- watchers ------------------------------------------------------
    def create(self, watcher: Watcher) -> Watcher:
        row = watcher.to_row()
        cols = ", ".join(_WATCHER_COLUMNS)
        placeholders = ", ".join("?" for _ in _WATCHER_COLUMNS)
        self.db.execute(
            f"INSERT INTO watchers ({cols}) VALUES ({placeholders})",
            tuple(row[c] for c in _WATCHER_COLUMNS))
        return watcher

    def get(self, ref: str) -> Watcher | None:
        ref = (ref or "").strip()
        if not ref:
            return None
        row = self.db.query_one("SELECT * FROM watchers WHERE id=?", (ref,))
        if row is None:
            row = self.db.query_one("SELECT * FROM watchers WHERE name=?",
                                    (ref,))
        return watcher_from_row(row) if row else None

    def list(self, state: str = "") -> list[Watcher]:
        if state:
            rows = self.db.query(
                "SELECT * FROM watchers WHERE state=? ORDER BY created_at DESC",
                (state,))
        else:
            rows = self.db.query(
                "SELECT * FROM watchers ORDER BY created_at DESC")
        return [watcher_from_row(r) for r in rows]

    def due(self, now: float) -> list[Watcher]:
        rows = self.db.query(
            "SELECT * FROM watchers WHERE state='active' "
            "AND (expires_at=0 OR expires_at > ?) "
            "AND (last_check=0 OR last_check + interval_s <= ?) "
            "ORDER BY last_check ASC", (now, now))
        return [watcher_from_row(r) for r in rows]

    def save(self, watcher: Watcher) -> None:
        watcher.updated_at = time.time()
        row = watcher.to_row()
        sets = ", ".join(f"{c}=?" for c in _WATCHER_COLUMNS if c != "id")
        self.db.execute(
            f"UPDATE watchers SET {sets} WHERE id=?",
            tuple(row[c] for c in _WATCHER_COLUMNS if c != "id")
            + (watcher.id,))

    def delete(self, ref: str) -> bool:
        watcher = self.get(ref)
        if watcher is None:
            return False
        self.db.execute("DELETE FROM watcher_alerts WHERE watcher_id=?",
                        (watcher.id,))
        self.db.execute("DELETE FROM watcher_checks WHERE watcher_id=?",
                        (watcher.id,))
        self.db.execute("DELETE FROM watchers WHERE id=?", (watcher.id,))
        return True

    def purge_expired(self, now: float) -> list[str]:
        """Auto-remove watchers past their ``expires_at``.

        Returns the removed watcher ids so the sweep can report them.
        """
        rows = self.db.query(
            "SELECT id, name FROM watchers "
            "WHERE expires_at > 0 AND expires_at <= ?", (now,))
        removed = []
        for row in rows:
            if self.delete(row["id"]):
                removed.append(row["id"])
                _log.info("watcher expired and auto-removed: %s (%s)",
                          row["id"], row.get("name"))
        return removed

    # -- check history (capped at HISTORY_CAP per watcher) --------------
    def record_check(self, watcher_id: str, *, changed: bool,
                     summary: str, value_hash: str,
                     now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self.db.execute(
            "INSERT INTO watcher_checks (id, watcher_id, checked_at, "
            "changed, summary, value_hash) VALUES (?,?,?,?,?,?)",
            (new_id(), watcher_id, now, 1 if changed else 0,
             summary[:500], value_hash))
        self.db.execute(
            "DELETE FROM watcher_checks WHERE watcher_id=? AND id NOT IN "
            "(SELECT id FROM watcher_checks WHERE watcher_id=? "
            "ORDER BY checked_at DESC LIMIT ?)",
            (watcher_id, watcher_id, HISTORY_CAP))

    def check_history(self, watcher_id: str,
                      limit: int = HISTORY_CAP) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT checked_at, changed, summary, value_hash "
            "FROM watcher_checks WHERE watcher_id=? "
            "ORDER BY checked_at DESC LIMIT ?", (watcher_id, limit))

    # -- alert audit log ------------------------------------------------
    def record_alert(self, watcher_id: str, *, severity: str, channel: str,
                     status: str, title: str, body: str,
                     now: float | None = None) -> str:
        """Every sent/held/digested/suppressed alert is recorded, so the
        owner can audit 'why didn't you tell me' and 'why did you spam me'."""
        now = now if now is not None else time.time()
        aid = new_id()
        self.db.execute(
            "INSERT INTO watcher_alerts (id, watcher_id, severity, channel, "
            "status, title, body, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (aid, watcher_id, severity, channel, status, title[:300],
             body[:2000], now))
        return aid

    def held_alerts(self) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM watcher_alerts WHERE status='held' "
            "ORDER BY created_at ASC")

    def mark_digested(self, alert_ids: list[str]) -> None:
        for aid in alert_ids:
            self.db.execute(
                "UPDATE watcher_alerts SET status='digested' WHERE id=?",
                (aid,))

    def alert_log(self, watcher_id: str = "",
                  limit: int = 50) -> list[dict[str, Any]]:
        if watcher_id:
            return self.db.query(
                "SELECT * FROM watcher_alerts WHERE watcher_id=? "
                "ORDER BY created_at DESC LIMIT ?", (watcher_id, limit))
        return self.db.query(
            "SELECT * FROM watcher_alerts ORDER BY created_at DESC LIMIT ?",
            (limit,))

    # -- sweeper-level state (digest bookkeeping) ------------------------
    def get_state(self, key: str, default: str = "") -> str:
        row = self.db.query_one(
            "SELECT value FROM watcher_state WHERE key=?", (key,))
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO watcher_state (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))


# ── smart alerting (the anti-spam core) ──────────────────────────────────────

def _owner_timezone(context: Any) -> str:
    """Owner's timezone from settings; never hardcoded.  Falls back to the
    TZ environment variable, then UTC."""
    import os

    settings = getattr(context, "settings", None)
    for attr in ("timezone", "tz", "owner_timezone"):
        tz = getattr(settings, attr, None) if settings else None
        if tz:
            return str(tz)
    partner = getattr(settings, "partner", None) if settings else None
    for attr in ("timezone", "tz"):
        tz = getattr(partner, attr, None) if partner else None
        if tz:
            return str(tz)
    return os.environ.get("TZ", "UTC") or "UTC"


def _in_quiet_hours(quiet_hours: dict[str, Any], now: float,
                    context: Any) -> bool:
    """True when ``now`` falls inside the watcher's quiet window.

    ``quiet_hours`` is ``{"start": "22:00", "end": "07:00", "tz": ...}``;
    empty/absent → no quiet hours.  The tz defaults to the owner's
    timezone (never a hardcoded one); an unparseable tz falls back to UTC
    rather than failing.
    """
    if not quiet_hours:
        return False
    start = str(quiet_hours.get("start", "") or "")
    end = str(quiet_hours.get("end", "") or "")
    if not start or not end:
        return False
    tz_name = str(quiet_hours.get("tz") or _owner_timezone(context) or "UTC")
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001 — bad tz never breaks alerting
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("UTC")
    import datetime as _dt

    local = _dt.datetime.fromtimestamp(now, tz=tz)
    cur = local.hour * 60 + local.minute
    try:
        sh, sm = (int(x) for x in start.split(":")[:2])
        eh, em = (int(x) for x in end.split(":")[:2])
    except ValueError:
        return False
    s, e = sh * 60 + sm, eh * 60 + em
    if s <= e:
        return s <= cur < e
    return cur >= s or cur < e  # overnight window


class AlertEngine:
    """Routes watcher hits through the anti-spam pipeline.

    Order of checks per hit: expiry → error streak → flap suppression →
    cooldown (with severity-escalation escape) → quiet hours → severity
    routing (urgent = immediate, important = immediate unless quiet hours,
    info = digest).  Every alert goes through ``Notifier`` (durable row
    first, then send) and is recorded in the alert audit log.
    """

    def __init__(self, store: WatcherStore, context: Any,
                 notifier: Any = None,
                 digest_window_s: float = DEFAULT_DIGEST_WINDOW_S) -> None:
        self.store = store
        self.context = context
        self._notifier = notifier
        self.digest_window_s = digest_window_s

    @property
    def notifier(self) -> Any:
        if self._notifier is None:
            from .notifier import Notifier

            # Notifier resolves the live gateway from the context itself
            # (context.extras["gateway"] in the runtime).
            self._notifier = Notifier(self.context)
        return self._notifier

    def _send(self, watcher: Watcher, title: str, body: str,
              now: float) -> dict[str, Any]:
        critical = watcher.severity == "urgent"
        res = self.notifier.publish(kind="watcher", title=title, body=body,
                                    critical=critical,
                                    channels=watcher.channels or None)
        channel = ",".join(watcher.channels) if watcher.channels else "notifier"
        # audit log stays honest: only "sent" when a channel was reached
        audit_status = (res.get("delivery_state")
                        or ("sent" if res.get("delivered") else "failed"))
        self.store.record_alert(watcher.id, severity=watcher.severity,
                                channel=channel, status=audit_status,
                                title=title, body=body, now=now)
        watcher.last_alert_ts = now
        watcher.last_alert_severity = watcher.severity
        self.store.save(watcher)  # persist last_alert_* (set after the
        # earlier save in handle())
        return {"action": "sent", "critical": critical,
                "delivered": res.get("delivered")}

    def _hold(self, watcher: Watcher, title: str, body: str, reason: str,
              now: float) -> dict[str, Any]:
        self.store.record_alert(watcher.id, severity=watcher.severity,
                                channel="digest", status="held",
                                title=f"[{reason}] {title}", body=body,
                                now=now)
        return {"action": "held", "reason": reason}

    def handle(self, watcher: Watcher, result: WatchResult,
               now: float) -> dict[str, Any]:
        """Process one check result; returns ``{"action": ...}``."""
        # 1) expiry
        if watcher.expires_at and now >= watcher.expires_at:
            watcher.state = "expired"
            self.store.save(watcher)
            return {"action": "none", "reason": "expired"}

        # 2) error streak (MonitorAgent's quiet-on-flaky philosophy)
        if result.error:
            watcher.error_streak += 1
            watcher.last_check = now
            self.store.save(watcher)
            if watcher.error_streak == _ERROR_NOTICE_STREAK:
                self.notifier.publish(
                    kind="watcher",
                    title=f"watcher erroring: {watcher.name}",
                    body=f"{watcher.error_streak} consecutive check "
                         f"failures; latest: {result.error[:200]}. "
                         f"It stays quiet until it recovers.",
                    critical=False)
                self.store.record_alert(
                    watcher.id, severity="important", channel="notifier",
                    status="sent", title=f"watcher erroring: {watcher.name}",
                    body=result.error[:300], now=now)
            return {"action": "none", "reason": "error",
                    "streak": watcher.error_streak}
        watcher.error_streak = 0

        # 3) flap suppression — repeated changed/unchanged flips in an hour
        if result.changed != watcher.last_changed:
            flips = [t for t in watcher.flip_times
                     if now - t <= FLAP_WINDOW_S] + [now]
            watcher.flip_times = flips
            watcher.last_changed = result.changed
            if len(flips) >= FLAP_TRANSITIONS:
                watcher.state = "paused"
                self.store.save(watcher)
                notice = (f"⏸ {watcher.name}: flapping "
                          f"({len(flips)} flips in the last hour) — paused. "
                          f"Say `nm watch resume {watcher.id}` to re-enable.")
                self.notifier.publish(kind="watcher",
                                      title=f"watcher paused: {watcher.name}",
                                      body=notice, critical=False)
                self.store.record_alert(
                    watcher.id, severity="important", channel="notifier",
                    status="sent", title=f"watcher paused: {watcher.name}",
                    body=notice, now=now)
                return {"action": "none", "reason": "flapping"}
        watcher.last_check = now
        self.store.save(watcher)

        if not result.changed:
            return {"action": "none", "reason": "no change"}

        # 4) cooldown — with a severity-escalation escape hatch
        escalated = (_SEVERITY_RANK.get(watcher.severity, 0)
                     > _SEVERITY_RANK.get(watcher.last_alert_severity, 0)
                     and watcher.last_alert_severity != "")
        if not escalated and (now - watcher.last_alert_ts) < watcher.cooldown:
            self.store.record_alert(
                watcher.id, severity=watcher.severity, channel="notifier",
                status="suppressed",
                title=f"[cooldown] {watcher.name}", body=result.summary,
                now=now)
            return {"action": "suppressed", "reason": "cooldown"}

        # 5) severity routing
        title = f"👁 {watcher.name}: {result.summary[:120]}"
        body = result.summary
        if watcher.severity == "urgent":
            return self._send(watcher, title, body, now)
        if _in_quiet_hours(watcher.quiet_hours, now, self.context):
            return self._hold(watcher, title, body, "quiet hours", now)
        if watcher.severity == "info":
            return self._hold(watcher, title, body, "digest", now)
        return self._send(watcher, title, body, now)

    def _held_in_quiet_hours(self, held_row: dict[str, Any],
                             now: float) -> bool:
        """True when a held alert's watcher is still in quiet hours."""
        watcher = self.store.get(held_row.get("watcher_id") or "")
        if watcher is None or not watcher.quiet_hours:
            return False
        return _in_quiet_hours(watcher.quiet_hours, now, self.context)

    def flush_digest(self, now: float) -> dict[str, Any]:
        """Send one digest for all held info alerts when the window is due."""
        held = self.store.held_alerts()
        if not held:
            return {"action": "none", "reason": "nothing held"}
        last = self.store.get_state("last_digest_ts", "0")
        try:
            last_ts = float(last)
        except ValueError:
            last_ts = 0.0
        if now - last_ts < self.digest_window_s:
            return {"action": "none", "reason": "window not due",
                    "held": len(held)}
        # Don't fire the digest while EVERY held alert is still inside its
        # watcher's quiet hours — that's the "morning digest", not a 3am one.
        if all(self._held_in_quiet_hours(h, now) for h in held):
            return {"action": "none", "reason": "quiet hours",
                    "held": len(held)}
        lines = []
        by_watcher: dict[str, list[str]] = {}
        for h in held:
            by_watcher.setdefault(h["watcher_id"], []).append(
                f"• [{h['severity']}] {h['title']}")
        for wid, items in by_watcher.items():
            watcher = self.store.get(wid)
            name = watcher.name if watcher else wid[:8]
            lines.append(f"**{name}**")
            lines.extend(items[:5])
        body = "\n".join(lines)[:3500]
        res = self.notifier.publish(kind="watcher",
                                    title=f"👁 watcher digest ({len(held)})",
                                    body=body, critical=False)
        self.store.mark_digested([h["id"] for h in held])
        self.store.set_state("last_digest_ts", str(now))
        return {"action": "digested", "count": len(held),
                "delivered": res.get("delivered")}

# ── natural-language watcher creation ────────────────────────────────────────

_PRICE_RE = re.compile(
    r"(?P<sym>[₦$€£])\s*(?P<amt>[\d][\d,]*(?:\.\d+)?)")
_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_GH_REPO_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
_INTERVAL_RE = re.compile(
    r"every\s+(?P<n>\d+)\s*(?P<unit>second|minute|hour|day)s?")
_CURRENCY_TO_FIAT = {"$": "usd", "₦": "ngn", "€": "eur", "£": "gbp"}


def _parse_interval(text: str, default: float) -> float:
    low = text.lower()
    m = _INTERVAL_RE.search(low)
    if m:
        n = int(m.group("n"))
        unit = m.group("unit")
        mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
        return max(60.0, float(n * mult))
    if "hourly" in low:
        return 3600.0
    if "daily" in low or "every day" in low:
        return 86400.0
    return default


def _parse_severity(text: str, default: str) -> str:
    low = text.lower()
    if any(w in low for w in ("urgent", "asap", "immediately", "right away")):
        return "urgent"
    if any(w in low for w in ("important",)):
        return "important"
    return default


_EXPIRY_RE = re.compile(
    r"\bfor\s+(?P<n>\d+)\s*(?P<unit>minutes?|hours?|days?|weeks?)\b")


def _parse_expiry(text: str) -> float:
    """\"for 3 days\" → seconds until expiry; 0.0 when no expiry given."""
    m = _EXPIRY_RE.search(text.lower())
    if not m:
        return 0.0
    n = int(m.group("n"))
    unit = m.group("unit")
    mult = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}
    key = unit.rstrip("s")
    return float(n * mult.get(key, 86400))


def _stamp_draft(draft: dict, text: str) -> dict:
    """Stamp derived fields (``expires_in_s``, ``channels``) on a draft
    so the echo and ``add()`` agree."""
    draft["expires_in_s"] = _parse_expiry(text)
    chans = _parse_channels(text)
    if chans:
        draft["channels"] = chans
    return draft


_CHANNEL_RE = re.compile(
    r"\b(?:only\s+)?(?:on|via|through|in)\s+"
    r"(?P<chans>(?:telegram|whatsapp|discord|sms|email)"
    r"(?:\s*(?:,|and|\+)\s*(?:telegram|whatsapp|discord|sms|email))*)")


def _parse_channels(text: str) -> list[str]:
    """\"alert me only on telegram\" / \"via whatsapp and email\" → platforms."""
    m = _CHANNEL_RE.search(text.lower())
    if not m:
        return []
    seen: list[str] = []
    for part in re.split(r"\s*(?:,|and|\+)\s*", m.group("chans")):
        part = part.strip()
        if part and part not in seen:
            seen.append(part)
    return seen


class WatcherAgent:
    """Owns watcher lifecycle: NL creation, checks, alerting, history."""

    role = "watcher"

    def __init__(self, context: Any, notifier: Any = None,
                 registry: Any = None,
                 now: Callable[[], float] | None = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        if self.db is None:
            raise RuntimeError("WatcherAgent needs a context with a database")
        self.store = WatcherStore(self.db)
        self.registry = registry or getattr(context, "tools", None)
        self._engine: AlertEngine | None = None
        self._notifier = notifier
        self._now = now or time.time

    @property
    def engine(self) -> AlertEngine:
        if self._engine is None:
            self._engine = AlertEngine(self.store, self.context,
                                       notifier=self._notifier)
        return self._engine

    def _wctx(self) -> WatcherContext:
        return WatcherContext(context=self.context, registry=self.registry,
                              store=self.store, now=self._now)

    # -- natural-language creation --------------------------------------
    def parse(self, text: str) -> dict[str, Any]:
        """Parse plain language into a watcher draft.

        Returns ``{"ok": True, "draft": {...}, "echo": "..."}`` or
        ``{"ok": False, "needs_clarification": True, "question": "..."}``.
        Ambiguity policy: ask ONE clarifying question; never create a
        watcher on a guessed product/symbol/threshold.
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "needs_clarification": True,
                    "question": "What should I watch for you?"}

        # 1) GitHub repo watches
        m = _GH_REPO_RE.search(text)
        if m:
            owner, repo = m.group(1), m.group(2)
            low = text.lower()
            feed = "releases"
            if "pull request" in low or re.search(r"\bprs?\b", low):
                feed = "prs"
            elif "issue" in low:
                feed = "issues"
            draft = {
                "kind": "repo",
                "name": f"github {feed} in {owner}/{repo}",
                "target": {"owner": owner, "repo": repo, "feed": feed},
                "condition": {"op": "changed", "field": "value"},
                "interval_s": _parse_interval(text, 3600.0),
                "severity": _parse_severity(text, "info"),
            }
            return {"ok": True, "draft": draft,
                    "echo": self._echo(_stamp_draft(draft, text))}

        # 2) price watches — "watch the Jumia price of X ... below ₦Y"
        #    or "tell me if BTC drops below $60000"
        low = text.lower()
        price_hit = ("price of" in low or "drops below" in low
                     or "rises above" in low or "drop below" in low
                     or "goes below" in low or "goes above" in low)
        if price_hit:
            return self._parse_price(text)

        # 3) keyword watches — "news about X", "articles mentioning X"
        kw = re.search(
            r"(?:news|articles|posts|results?)\s+(?:about|on|mentioning|for)\s+"
            r"(?P<kw>[\"']?[^\"']+[\"']?)", low)
        if kw or "keyword" in low:
            keywords = (kw.group("kw").strip().strip("\"'") if kw else "")
            keywords = re.split(r"\s+(?:and|or)\s+tell me|\s+and\s+alert",
                                keywords)[0].strip()
            if not keywords:
                return {"ok": False, "needs_clarification": True,
                        "question": "What keywords should I watch for?"}
            draft = {
                "kind": "keyword",
                "name": f"news: {keywords}",
                "target": {"keywords": keywords, "source": "web"},
                "condition": {"op": "changed", "field": "value"},
                "interval_s": _parse_interval(text, 3600.0),
                "severity": _parse_severity(text, "info"),
            }
            return {"ok": True, "draft": draft, "echo": self._echo(_stamp_draft(draft, text))}

        # 4) plain URL watches — "watch https://example.com for changes"
        urls = _URL_RE.findall(text)
        if urls:
            url = urls[0].rstrip(".,;")
            draft = {
                "kind": "url",
                "name": f"url: {url[:60]}",
                "target": {"url": url},
                "condition": {"op": "changed", "field": "value"},
                "interval_s": _parse_interval(text, 300.0),
                "severity": _parse_severity(text, "important"),
            }
            return {"ok": True, "draft": draft, "echo": self._echo(_stamp_draft(draft, text))}

        # 5) file watches — quoted path or path-like token
        fmatch = re.search(r"['\"]([^'\"]+\.[A-Za-z0-9]+)['\"]", text)
        if not fmatch:
            fmatch = re.search(
                r"(?:file|path)\s+([~\w./-]+\.[A-Za-z0-9]+)", low)
        if fmatch:
            path = fmatch.group(1)
            draft = {
                "kind": "file",
                "name": f"file: {path}",
                "target": {"path": path},
                "condition": {"op": "changed", "field": "value"},
                "interval_s": _parse_interval(text, 300.0),
                "severity": _parse_severity(text, "important"),
            }
            return {"ok": True, "draft": draft, "echo": self._echo(_stamp_draft(draft, text))}

        return {"ok": False, "needs_clarification": True,
                "question": ("I couldn't tell what to watch. Try e.g. "
                             "'watch the Jumia price of X and tell me if it "
                             "drops below ₦Y', or 'watch "
                             "https://example.com for changes'.")}

    def _parse_price(self, text: str) -> dict[str, Any]:
        low = text.lower()
        # comparator + threshold
        op: str | None = None
        if re.search(r"drops?\s+below|goes\s+below|falls?\s+below|\bbelow\b"
                     r"|\bunder\b", low):
            op = "lt"
        elif re.search(r"rises?\s+above|goes\s+above|\babove\b|\bover\b",
                       low):
            op = "gt"
        pm = _PRICE_RE.search(text)
        if op is None or pm is None:
            if pm is None and op is None:
                return {"ok": False, "needs_clarification": True,
                        "question": ("What price threshold should trigger "
                                     "the alert? e.g. 'below ₦50,000'.")}
            return {"ok": False, "needs_clarification": True,
                    "question": ("Should I alert when the price goes above "
                                 "or below that threshold?")}
        amount = float(pm.group("amt").replace(",", ""))
        currency = pm.group("sym")

        # symbol (BTC drops below $60000) vs product query
        sym_m = re.search(r"\b([A-Z]{2,10})\s+(?:drops?|rises?|goes|falls?)\s+"
                          r"(?:below|above)", text)
        if sym_m and sym_m.group(1) not in ("THE", "AND"):
            symbol = sym_m.group(1)
            draft = {
                "kind": "price",
                "name": f"{symbol} {op} {currency}{amount:,.0f}",
                "target": {"source": "market", "symbol": symbol,
                           "currency": _CURRENCY_TO_FIAT.get(currency, "usd")},
                "condition": {"op": op, "field": "price", "value": amount},
                "interval_s": _parse_interval(text, 3600.0),
                "severity": _parse_severity(text, "important"),
            }
            return {"ok": True, "draft": draft, "echo": self._echo(_stamp_draft(draft, text))}

        # product query: text between "price of" and the threshold clause
        qm = re.search(r"price of\s+(?P<q>.+?)\s+(?:and\s+tell me|if it|"
                       r"drops?|rises?|goes|falls?|below|above|under|over"
                       r"|[₦$€£])", low)
        query = qm.group("q").strip() if qm else ""
        query = re.sub(r"^(the|a|an)\s+", "", query)
        if not query:
            return {"ok": False, "needs_clarification": True,
                    "question": "Which product should I watch the price of?"}
        draft = {
            "kind": "price",
            "name": f"price of {query} {op} {currency}{amount:,.0f}",
            "target": {"source": "deals", "query": query},
            "condition": {"op": op, "field": "price", "value": amount},
            "interval_s": _parse_interval(text, 3600.0),
            "severity": _parse_severity(text, "important"),
        }
        return {"ok": True, "draft": draft, "echo": self._echo(_stamp_draft(draft, text))}

    @staticmethod
    def _echo(draft: dict[str, Any]) -> str:
        """Plain-language echo of the parsed spec so the owner can spot
        misparses before the watcher is created."""
        kind = draft["kind"]
        t = draft.get("target", {})
        c = draft.get("condition", {})
        interval = draft.get("interval_s", 3600)
        sev = draft.get("severity", "info")
        if interval >= 86400:
            every = f"every {interval / 86400:g} day(s)"
        elif interval >= 3600:
            every = f"every {interval / 3600:g} hour(s)"
        else:
            every = f"every {interval / 60:g} minute(s)"
        if kind == "price":
            what = (f"the market price of {t.get('symbol')}"
                    if t.get("source") == "market"
                    else f"the price of {t.get('query')!r}")
            direction = "drops below" if c.get("op") == "lt" else \
                "rises above" if c.get("op") == "gt" else c.get("op")
            cond_s = f"{direction} {c.get('value')}"
        elif kind == "repo":
            cond_s = f"new {t.get('feed')}"
            what = f"github.com/{t.get('owner')}/{t.get('repo')}"
        elif kind == "keyword":
            cond_s = "new results appear"
            what = f"news/results for {t.get('keywords')!r}"
        elif kind == "url":
            cond_s = "its visible content changes"
            what = t.get("url", "?")
        elif kind == "file":
            cond_s = "it changes"
            what = f"file {t.get('path', '?')}"
        else:
            cond_s, what = c.get("op"), t.get("tool", "?")
        exp = ""
        exp_s = draft.get("expires_in_s") or 0.0
        if exp_s >= 86400:
            exp = f", auto-removing after {exp_s / 86400:g} day(s)"
        elif exp_s > 0:
            exp = f", auto-removing after {exp_s / 3600:g} hour(s)"
        chans = draft.get("channels") or []
        chan_s = f" on {', '.join(chans)}" if chans else ""
        return (f"I'll watch {what}, {every}, and alert you ({sev}) when "
                f"{cond_s}{chan_s}{exp}.")

    # -- lifecycle ------------------------------------------------------
    def add(self, text: str, **overrides: Any) -> dict[str, Any]:
        """Parse NL, echo the draft, and create the watcher (or ask one
        clarifying question when ambiguous)."""
        parsed = self.parse(text)
        if not parsed.get("ok"):
            return parsed
        draft = dict(parsed["draft"])
        draft.update({k: v for k, v in overrides.items()
                      if v is not None})
        kind = draft["kind"]
        if kind not in WATCHER_KINDS:
            raise ValueError(f"unknown watcher kind {kind!r}")
        cond = parse_condition(draft.get("condition"))
        now = self._now()
        # expiry: explicit expires_at wins; else "for 3 days" → expires_in_s
        expires_at = float(draft.get("expires_at") or 0.0)
        if not expires_at:
            exp_s = float(draft.get("expires_in_s")
                          or _parse_expiry(text))
            if exp_s > 0:
                expires_at = now + exp_s
        watcher = Watcher(
            id=new_short_id("w"),
            name=str(draft.get("name") or f"{kind} watcher"),
            kind=kind,
            target=dict(draft.get("target") or {}),
            condition=cond,
            interval_s=max(60.0, float(draft.get("interval_s")
                                      or DEFAULT_INTERVALS[kind])),
            cooldown_s=float(draft.get("cooldown_s") or 0.0),
            severity=draft.get("severity") or "info",
            channels=list(draft.get("channels") or []),
            quiet_hours=dict(draft.get("quiet_hours") or {}),
            expires_at=expires_at,
            created_at=now, updated_at=now,
        )
        if watcher.severity not in SEVERITIES:
            watcher.severity = "info"
        self.store.create(watcher)
        _log.info("watcher created: %s (%s)", watcher.id, watcher.name)
        return {"ok": True, "watcher": self._public(watcher),
                "echo": parsed["echo"]}

    def pause(self, ref: str) -> dict[str, Any] | None:
        watcher = self.store.get(ref)
        if watcher is None:
            return None
        watcher.state = "paused"
        self.store.save(watcher)
        return self._public(watcher)

    def resume(self, ref: str) -> dict[str, Any] | None:
        watcher = self.store.get(ref)
        if watcher is None:
            return None
        watcher.state = "active"
        watcher.flip_times = []  # fresh start after a flap pause
        self.store.save(watcher)
        return self._public(watcher)

    def remove(self, ref: str) -> bool:
        return self.store.delete(ref)

    def list(self, state: str = "") -> list[dict[str, Any]]:
        return [self._public(w) for w in self.store.list(state)]

    def history(self, ref: str,
                limit: int = HISTORY_CAP) -> dict[str, Any] | None:
        watcher = self.store.get(ref)
        if watcher is None:
            return None
        return {"watcher": self._public(watcher),
                "checks": self.store.check_history(watcher.id, limit)}

    def alert_log(self, ref: str = "", limit: int = 50) -> list[dict[str, Any]]:
        wid = ""
        if ref:
            watcher = self.store.get(ref)
            if watcher is None:
                return []
            wid = watcher.id
        return self.store.alert_log(wid, limit)

    @staticmethod
    def _public(w: Watcher) -> dict[str, Any]:
        return {
            "id": w.id, "name": w.name, "kind": w.kind,
            "target": w.target, "condition": w.condition.to_dict(),
            "interval_s": w.interval_s, "cooldown_s": w.cooldown,
            "severity": w.severity, "channels": w.channels,
            "quiet_hours": w.quiet_hours,
            "expires_at": w.expires_at, "state": w.state,
            "last_check": w.last_check, "last_alert_ts": w.last_alert_ts,
            "error_streak": w.error_streak, "created_at": w.created_at,
        }

    # -- the check loop --------------------------------------------------
    def sweep(self, *, now: float | None = None) -> dict[str, Any]:
        """Check every due watcher once, route hits through the alert
        engine, flush the digest when due.

        Restart/catch-up: watcher state lives in the DB, so a restart
        simply resumes — any watcher whose interval elapsed during downtime
        is due and gets exactly ONE catch-up check (never a storm of
        backfilled checks).
        """
        now = now if now is not None else self._now()
        expired_removed = self.store.purge_expired(now)
        due = self.store.due(now)
        wctx = self._wctx()
        checked, changed, errors = 0, 0, 0
        actions: list[dict[str, Any]] = []
        for watcher in due:
            # expiry is handled even for watchers that are not "due"
            checked += 1
            checker = KIND_CHECKERS.get(watcher.kind)
            if checker is None:
                result = WatchResult(changed=False,
                                     error=f"unknown kind {watcher.kind!r}")
            else:
                try:
                    result = checker.check(wctx, watcher)
                except _HostRateLimited as exc:
                    # deferred to the next sweep, not an error
                    self.store.record_check(
                        watcher.id, changed=False,
                        summary=f"deferred: {exc}", value_hash="", now=now)
                    continue
                except Exception as exc:  # noqa: BLE001 — one bad watcher
                    # must not kill the sweep
                    result = WatchResult(changed=False,
                                         error=f"{type(exc).__name__}: {exc}")
            try:
                value_hash = self._value_hash(result.new_value)
            except Exception:  # noqa: BLE001
                value_hash = ""
            self.store.record_check(watcher.id, changed=result.changed,
                                    summary=result.summary or result.error,
                                    value_hash=value_hash, now=now)
            # the check established a new baseline — persist it
            watcher.last_value = result.new_value
            outcome = self.engine.handle(watcher, result, now)
            if result.changed:
                changed += 1
            if result.error:
                errors += 1
            actions.append({"watcher": watcher.id, "name": watcher.name,
                            "changed": result.changed,
                            "action": outcome.get("action"),
                            "reason": outcome.get("reason", "")})
            _log.info("watcher sweep: %s changed=%s action=%s", watcher.id,
                      result.changed, outcome.get("action"))
        digest = self.engine.flush_digest(now)
        return {"checked": checked, "changed": changed, "errors": errors,
                "actions": actions, "digest": digest, "now": now,
                "expired_removed": expired_removed}

    def tick(self, *, now: float | None = None) -> dict[str, Any]:
        """Autonomy-loop entry point (same contract as MonitorAgent.tick):
        drive the sweep from the loop as well as from the scheduler."""
        return self.sweep(now=now)

    @staticmethod
    def _value_hash(value: Any) -> str:
        try:
            return hashlib.sha256(
                json.dumps(value, sort_keys=True,
                           default=str).encode("utf-8")).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            return ""


# ── scheduler integration ────────────────────────────────────────────────────

SWEEPER_JOB_NAME = "watchers sweep"


def ensure_sweeper_job(context: Any) -> dict[str, Any]:
    """Register the single watchers sweeper job on the agent scheduler
    (idempotent by name).

    One durable job — ``watchers sweep``, every 1m — calls the ``watch``
    tool's sweep action, which checks only due watchers.  Never one job per
    watcher.  Follows the Prompt-01 ``ensure_improvement_schedule`` pattern;
    safe to call on every boot.  The job row is durable, so watchers resume
    after a restart with one catch-up check each.
    """
    from .scheduler import Scheduler

    sched = Scheduler(context)
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == SWEEPER_JOB_NAME]
    except Exception:  # noqa: BLE001 — scheduler table may not exist yet
        have = []
    if have:
        return {"name": SWEEPER_JOB_NAME, "already_scheduled": True,
                "job_id": have[0].get("id")}
    job = sched.add(SWEEPER_JOB_NAME, "every 1m", "tool",
                    {"tool": "watch", "args": {"action": "sweep"}})
    _log.info("scheduled sweeper job: %s", SWEEPER_JOB_NAME)
    return {"name": SWEEPER_JOB_NAME, "scheduled": True,
            "job_id": job.get("id")}


# ── tool registration ────────────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Register the ``watch`` tool (agent-callable watcher management)."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "watch",
        description=(
            "Create and manage background watchers: 'watch X for Y and tell "
            "me when Z'. action=add (text, plain-language, e.g. 'watch the "
            "Jumia price of X and tell me if it drops below ₦50000') | "
            "list | pause (ref) | resume (ref) | rm (ref) | history (ref) | "
            "alerts (ref, limit) | tick (run due checks now). Kinds: url, "
            "file, price, repo (github), keyword (news), condition "
            "(read-only tool query + structured predicate). Alerts are "
            "smart: cooldowns, digest batching for info, quiet hours, flap "
            "auto-pause. Checks are READ-ONLY — a watcher can never invoke "
            "a write/mutating tool."
        ),
        capability=Capability.MEM_WRITE,
    )
    def watch(action: str = "list", text: str = "", ref: str = "",
              limit: int = 50) -> dict[str, Any]:
        from .notifier import Notifier

        agent = WatcherAgent(
            context,
            notifier=Notifier(context),
            registry=registry)
        if action == "add":
            return agent.add(text)
        if action == "pause":
            row = agent.pause(ref)
            return {"watcher": row, "found": row is not None}
        if action == "resume":
            row = agent.resume(ref)
            return {"watcher": row, "found": row is not None}
        if action == "rm":
            return {"removed": agent.remove(ref)}
        if action == "history":
            row = agent.history(ref, limit=limit)
            return {"history": row, "found": row is not None}
        if action == "alerts":
            return {"alerts": agent.alert_log(ref, limit=limit)}
        if action == "tick":
            return agent.tick()
        if action == "sweep":
            # the scheduler sweeper job's entry point (same as tick)
            return agent.sweep()
        return {"watchers": agent.list()}
