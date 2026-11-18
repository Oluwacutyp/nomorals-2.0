"""Price-watch tool: the CamelCamelCamel rung for the brain.

"Watch this on Amazon, tell me under $50" over the price-history
organ — not a standalone tracker. The organ
(:mod:`nomorals.commerce.price_history`) owns the watches, the
journalled points, the alert discipline and the chart; this
module is only the surface between the brain and the organ, plus
the live price fetcher built on the shopping engine (the organ
itself never imports the L5 engine — the fetcher is injected).

The fetcher: a product URL goes through the deal hunter's
``get_price_detail`` (price + stock sniffing); a search phrase
goes through the shopping engine's ``scan_category`` (cheapest
snapshot on the marketplace). Either way a price the fetcher
cannot observe is reported as no observation, never invented.

``action=history`` renders the chart from the watch's own journal
(matplotlib first, PIL where it is absent) and writes the PNG
under ``<home>/commerce/price_charts/`` — the chart carries the
true start date, the day watching began.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..commerce.price_history import (
    MARKETPLACES,
    PriceHistoryStore,
    _db_path_for,
    format_history,
    format_watches,
    parse_price_request,
)
from ..core.policy import Capability

__all__ = ["register", "build_fetcher", "init_price_history_tool"]

_engine: Any = None
_hunter: Any = None
_lazy_context: Any = None


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store_of(context: Any) -> PriceHistoryStore:
    home = _home_of(context)
    if home:
        return PriceHistoryStore(_db_path_for(home))
    return PriceHistoryStore()


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _build_engine() -> Any:
    from ..integrations import expose as _expose
    from ..storage.db import Database
    from ..tools.browser import BrowserSession

    db = getattr(_lazy_context, "db", None) if _lazy_context else None
    if db is None:
        home = os.path.expanduser("~/.nomorals")
        os.makedirs(home, exist_ok=True)
        db = Database(os.path.join(home, "shopping.db"))
    browser = BrowserSession(name="price_history")
    return _expose.make_naija_shopping(db, browser)


def _get_hunter() -> Any:
    global _hunter
    if _hunter is None:
        _hunter = _build_hunter()
    return _hunter


def _build_hunter() -> Any:
    from ..accounts.manager import AccountManager
    from ..accounts.vault import VaultLockedError, open_vault
    from ..integrations.naija_deals import NaijaDealHunter
    from ..storage.db import Database

    db = getattr(_lazy_context, "db", None) if _lazy_context else None
    if db is None:
        home = os.path.expanduser("~/.nomorals")
        os.makedirs(home, exist_ok=True)
        db = Database(os.path.join(home, "shopping.db"))
    try:
        vault = open_vault(db, context=_lazy_context,
                           actor="price_history")
    except VaultLockedError as exc:
        raise RuntimeError(
            "Price history needs the credential vault.\n"
            f"{exc}\n"
            "Or call init_price_history_tool() with a ready hunter."
        ) from exc
    return NaijaDealHunter(AccountManager(vault), db)


def init_price_history_tool(*, engine: Any = None,
                            hunter: Any = None) -> None:
    """Inject a ready engine/hunter (tests and explicit setup)."""
    global _engine, _hunter
    if engine is not None:
        _engine = engine
    if hunter is not None:
        _hunter = hunter


def _run_async(coro: Any) -> Any:
    """Run one async fetch from sync code (the deals-tool shape)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def build_fetcher(*, engine: Any = None, hunter: Any = None
                  ) -> Callable[[str, str], list[dict[str, Any]] | None]:
    """Live price fetcher over the shopping engine.

    URL → the deal hunter's page fetch (price + stock sniffing);
    search phrase → the engine's category scan (cheapest snapshot
    on the marketplace). Returns a list of per-series observations
    (``series``/``price`` or ``price_minor``/``currency``/
    ``in_stock``), ``[]`` when nothing was observed, None only
    when the engine itself is unavailable. Never raises.
    """
    eng = engine if engine is not None else _get_engine()
    hun = hunter

    def fetch(product_ref: str, marketplace: str
              ) -> list[dict[str, Any]] | None:
        ref = (product_ref or "").strip()
        market = (marketplace or "amazon").strip().lower()
        if not ref:
            return []
        try:
            if ref.startswith("http"):
                hunter = hun if hun is not None else _get_hunter()
                vendor = (hunter._detect_vendor_from_url(ref)
                          or market)
                detail = _run_async(
                    hunter.get_price_detail(ref, vendor))
                price = float(detail.get("price") or 0)
                if price <= 0:
                    return []
                series = ("amazon" if market == "amazon" else market)
                return [{
                    "series": series, "price": price,
                    "currency": MARKETPLACES.get(market, {}).get(
                        "currency", ""),
                    "in_stock": bool(detail.get("in_stock", True)),
                }]
            snaps = _run_async(
                eng.scan_category(ref, sites=[market]))
            mine = [s for s in snaps if s.site == market]
            if not mine:
                mine = list(snaps)
            if not mine:
                return []
            best = min(mine, key=lambda s: s.price_ngn or float("inf"))
            if not best.price_ngn:
                return []
            return [{
                "series": market,
                "price_minor": int(round(best.price_ngn * 100)),
                "currency": "NGN",
                "in_stock": bool(best.in_stock),
            }]
        except Exception:
            return []

    return fetch


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:60] or "product"


def _chart_path(home: str | None, watch_id: str, product_ref: str) -> str:
    base = (Path(home) / "commerce" / "price_charts"
            if home else Path("commerce") / "price_charts")
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    return str(base / f"{_slug(product_ref)}-{watch_id[-6:]}-{stamp}.png")


def register(registry: Any) -> None:
    """Attach the ``price_watch`` tool."""
    global _lazy_context
    _lazy_context = getattr(registry, "context", None)
    context = registry.context

    @registry.register(
        "price_watch",
        description=(
            "Watch a retail product's price and alert on drops: "
            "action=watch takes product_ref=<URL or product name>, "
            "marketplace=amazon (default) | jumia | konga | jiji | "
            "kara | slot | temu | aliexpress | ebay | banggood, "
            "series=amazon (default) | tp_new | tp_used (Amazon's "
            "three price series; other marketplaces follow their "
            "own single series), target_minor=<target in minor "
            "units, 0 = alert on any drop ≥ threshold> or "
            "target_text='under $50' / 'under ₦200k', "
            "threshold_pct=<drop % that fires, default 10>. "
            "action=history takes product_ref=<name or URL> and "
            "renders the price chart from the watch's own journal "
            "(true start date on the chart). action=list lists the "
            "active watches; action=check runs one check now for "
            "watch_id=<id>; action=unwatch drops watch_id=<id>; "
            "action=status reports the seeded marketplaces and "
            "whether the live price fetcher is available. Alerts "
            "fire once per drop level and once per back-in-stock — "
            "a second tick below target stays silent."
        ),
        capability=Capability.NET_OUT,
    )
    def price_watch_tool(action: str = "list", **kwargs: Any
                         ) -> dict[str, Any]:
        verb = (action or "list").strip().lower()
        store = _store_of(context)
        if verb == "status":
            try:
                build_fetcher()
                fetcher_ok: str = "available"
            except Exception as exc:  # noqa: BLE001 - status reports, never raises
                fetcher_ok = f"unavailable: {exc}"
            return {
                "ok": True, "action": "status",
                "marketplaces": sorted(MARKETPLACES),
                "fetcher": fetcher_ok,
                "active_watches": len(store.list_watches()),
            }
        if verb == "list":
            watches = store.list_watches()
            return {"ok": True, "action": "list",
                    "count": len(watches),
                    "text": format_watches(watches),
                    "watches": [
                        {"id": w.id, "product_ref": w.product_ref,
                         "marketplace": w.marketplace, "series": w.series,
                         "target_minor": w.target_minor,
                         "last_minor": w.last_minor,
                         "currency": w.currency}
                        for w in watches]}
        if verb == "watch":
            product_ref = str(kwargs.get("product_ref") or "").strip()
            if not product_ref:
                raise ValueError(
                    "price_watch watch needs product_ref=<URL or "
                    "product name>")
            marketplace = str(kwargs.get("marketplace") or "amazon")
            series = str(kwargs.get("series") or "")
            target_minor = int(kwargs.get("target_minor") or 0)
            currency = str(kwargs.get("currency") or "")
            target_text = str(kwargs.get("target_text") or "")
            if target_text and not target_minor:
                parsed = parse_price_request(
                    f"watch {product_ref} on {marketplace} {target_text}")
                if parsed is not None:
                    target_minor = int(parsed.get("target_minor") or 0)
                    currency = str(parsed.get("currency") or "")
            watch = store.watch(
                product_ref, marketplace, series=series,
                target_minor=target_minor, currency=currency,
                threshold_pct=float(kwargs.get("threshold_pct") or 10.0))
            try:
                fetcher = build_fetcher()
            except Exception:  # noqa: BLE001 - watch stands without a live fetcher
                fetcher = None
            result = store.check(watch, fetcher)
            alert = result.get("alert")
            return {"ok": True, "action": "watch",
                    "watch_id": watch.id,
                    "product_ref": watch.product_ref,
                    "marketplace": watch.marketplace,
                    "series": watch.series,
                    "target_minor": watch.target_minor,
                    "check": {k: v for k, v in result.items()
                              if k != "alert"},
                    "alert": alert.text if alert is not None else ""}
        if verb == "unwatch":
            watch_id = str(kwargs.get("watch_id") or "")
            if not watch_id:
                raise ValueError("price_watch unwatch needs watch_id=<id>")
            return {"ok": True, "action": "unwatch",
                    "removed": store.unwatch(watch_id)}
        if verb == "check":
            watch_id = str(kwargs.get("watch_id") or "")
            product_ref = str(kwargs.get("product_ref") or "")
            watch = None
            if watch_id:
                watch = store.get(watch_id)
            elif product_ref:
                found = store.find_by_product(
                    product_ref, str(kwargs.get("marketplace") or ""))
                watch = found[0] if found else None
            if watch is None:
                raise ValueError("no such price watch")
            try:
                fetcher = build_fetcher()
            except Exception:  # noqa: BLE001 - check reports, never raises
                fetcher = None
            result = store.check(watch, fetcher)
            alert = result.get("alert")
            return {"ok": True, "action": "check",
                    "watch_id": watch.id,
                    "check": {k: v for k, v in result.items()
                              if k != "alert"},
                    "alert": alert.text if alert is not None else ""}
        if verb == "history":
            product_ref = str(kwargs.get("product_ref") or "").strip()
            if not product_ref:
                raise ValueError(
                    "price_watch history needs product_ref=<name or URL>")
            found = store.find_by_product(
                product_ref, str(kwargs.get("marketplace") or ""))
            if not found:
                raise ValueError(
                    f"no price watch for {product_ref!r} — say "
                    f"'watch {product_ref} on amazon under $50' first")
            watch = found[0]
            points = store.history(watch.id)
            if not points:
                return {"ok": True, "action": "history",
                        "watch_id": watch.id,
                        "text": format_history(watch, points),
                        "chart": "", "n_points": 0}
            chart = store.chart_for(watch.id)
            home = _home_of(context)
            chart_path = ""
            if chart is not None:
                chart_path = _chart_path(home, watch.id,
                                         watch.product_ref)
                Path(chart_path).write_bytes(chart.png)
            return {"ok": True, "action": "history",
                    "watch_id": watch.id,
                    "text": format_history(watch, points,
                                           chart_path=chart_path),
                    "chart": chart_path,
                    "chart_meta": chart.to_dict() if chart else None,
                    "n_points": len(points)}
        raise ValueError(
            f"unknown price_watch action {action!r}; use "
            "watch|unwatch|list|history|check|status")
