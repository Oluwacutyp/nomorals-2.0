"""Deals tool - exposes Naija Shopping Engine as a tool for the agent.

This is the tool interface layer. The actual scraping engine lives in
nomorals.integrations.naija_shopping.

Usage from agent:
    deals(action="scan", query="iphone 15", max_price=500000)
    deals(action="steals", threshold=70)
    deals(action="track", url="https://...", target_price=100000)
    deals(action="watchlist")
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..integrations.naija_shopping import NaijaShoppingEngine

__all__ = ["register", "init_deals_tool", "init_deal_hunter"]

_log = get_logger(__name__)

_engine: NaijaShoppingEngine | None = None
_hunter: Any = None
_lazy_context: Any = None


def _get_engine() -> NaijaShoppingEngine:
    """Get or create the shopping engine singleton.

    Lazily builds the engine from the registry context on first use, so
    the tool works in chat without an explicit init call. An explicitly
    injected engine (init_deals_tool) still wins.
    """
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _build_engine() -> NaijaShoppingEngine:
    """Build a NaijaShoppingEngine from the captured registry context."""
    from ..storage.db import Database
    from ..tools.browser import BrowserSession

    db = getattr(_lazy_context, "db", None) if _lazy_context else None
    if db is None:
        # Standalone engine DB under the nomorals home.
        import os
        home = os.path.expanduser("~/.nomorals")
        os.makedirs(home, exist_ok=True)
        db = Database(os.path.join(home, "shopping.db"))
    browser = BrowserSession(name="deals")
    engine = NaijaShoppingEngine(db, browser)
    _log.info("deals engine lazily initialized")
    return engine


def _get_hunter() -> Any:
    """Get or create the NaijaDealHunter singleton (price-drop alerts).

    Lazily builds from the registry context when a vault-backed
    AccountManager is available; otherwise raises an actionable error.
    """
    global _hunter
    if _hunter is None:
        _hunter = _build_hunter()
    return _hunter


def _build_hunter() -> Any:
    import os
    from ..accounts.manager import AccountManager
    from ..accounts.vault import CredentialVault
    from ..integrations.naija_deals import NaijaDealHunter
    from ..storage.db import Database

    passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
    if not passphrase:
        raise RuntimeError(
            "Deal hunter needs the credential vault: set "
            "NM_VAULT_PASSPHRASE, or call init_deal_hunter() with a "
            "ready NaijaDealHunter.")
    db = getattr(_lazy_context, "db", None) if _lazy_context else None
    if db is None:
        import os as _os
        home = _os.path.expanduser("~/.nomorals")
        _os.makedirs(home, exist_ok=True)
        db = Database(_os.path.join(home, "shopping.db"))
    vault = CredentialVault(db, master_passphrase=passphrase)
    hunter = NaijaDealHunter(AccountManager(vault), db)
    _log.info("deal hunter lazily initialized")
    return hunter


def init_deal_hunter(hunter: Any) -> None:
    """Initialize the deal hunter (price-drop monitoring/alerts)."""
    global _hunter
    _hunter = hunter
    _log.info("Deal hunter initialized")


def init_deals_tool(engine: NaijaShoppingEngine) -> None:
    """Initialize the deals tool with a shopping engine instance."""
    global _engine
    _engine = engine
    _log.info("Deals tool initialized")


async def deals(action: str, **kwargs: Any) -> dict[str, Any]:
    """Deals tool - search, track, and find steals across Nigerian marketplaces.
    
    Args:
        action: One of: scan, compare, steals, track, watchlist, check_watchlists
        **kwargs: Action-specific parameters
        
    Returns:
        Dict with results
    """
    engine = _get_engine()
    
    if action == "scan":
        query = kwargs.get("query", "")
        max_price = kwargs.get("max_price", float("inf"))
        sites = kwargs.get("sites", None)
        
        snapshots = await engine.scan_category(
            query,
            sites=sites,
            max_price=max_price,
        )
        
        return {
            "action": "scan",
            "query": query,
            "results_count": len(snapshots),
            "results": [s.to_dict() for s in snapshots[:20]],
        }
    
    elif action == "compare":
        query = kwargs.get("query", "")
        sites = kwargs.get("sites", None)
        max_price = kwargs.get("max_price", float("inf"))
        
        report = await engine.compare_prices(
            query,
            sites=sites,
            max_price=max_price,
        )
        
        return {"action": "compare", **report}
    
    elif action == "steals":
        threshold = kwargs.get("threshold", 70)
        limit = kwargs.get("limit", 20)
        
        steals = await engine.get_steals(threshold=threshold, limit=limit)
        
        return {
            "action": "steals",
            "threshold": threshold,
            "count": len(steals),
            "steals": [s.to_dict() for s in steals],
        }
    
    elif action == "track":
        url = kwargs.get("url", "")
        user_id = kwargs.get("user_id", "")
        target_price = kwargs.get("target_price", 0)
        
        if not url or not user_id:
            return {"error": "url and user_id required"}
        
        watchlist = await engine.track_url(url, user_id, target_price=target_price)
        
        return {
            "action": "track",
            "watchlist_id": watchlist.watchlist_id,
            "product_url": watchlist.product_url,
            "target_price": watchlist.target_price,
        }
    
    elif action == "watchlist":
        user_id = kwargs.get("user_id", "")
        if not user_id:
            return {"error": "user_id required"}
        
        watchlists = engine.db.query(
            "SELECT * FROM watchlists WHERE user_id = ? AND is_triggered = 0",
            (user_id,)
        )
        
        return {
            "action": "watchlist",
            "count": len(watchlists),
            "items": [dict(w) for w in watchlists],
        }
    
    elif action == "check_watchlists":
        triggered = await engine.check_watchlists()
        
        return {
            "action": "check_watchlists",
            "triggered_count": len(triggered),
            "triggered": [w.to_dict() for w in triggered],
        }

    elif action == "hunt":
        # NaijaDealHunter: find deals with discount scoring
        hunter = _get_hunter()
        category = kwargs.get("category", "")
        max_price = kwargs.get("max_price", float("inf"))
        min_discount = kwargs.get("min_discount", 0)
        deals_found = await hunter.find_deals(
            category=category, max_price=max_price, min_discount=min_discount
        )
        return {
            "action": "hunt",
            "count": len(deals_found),
            "deals": [d.to_dict() for d in deals_found],
        }

    elif action == "flash":
        hunter = _get_hunter()
        limit = kwargs.get("limit", 10)
        sales = await hunter.find_flash_sales(limit=limit)
        return {
            "action": "flash",
            "count": len(sales),
            "sales": [s.to_dict() for s in sales],
        }

    elif action == "deal_alerts":
        hunter = _get_hunter()
        user_id = kwargs.get("user_id", "")
        if user_id:
            alerts = await hunter.get_user_alerts(user_id)
        else:
            alerts = await hunter.check_alerts()
        return {
            "action": "deal_alerts",
            "count": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
        }

    elif action == "price_history":
        hunter = _get_hunter()
        url = kwargs.get("url", "")
        history = await hunter.price_history(url)
        return {
            "action": "price_history",
            "history": history.to_dict() if hasattr(history, "to_dict") else str(history),
        }

    else:
        return {"error": f"Unknown action: {action}"}


def register(registry: Any) -> None:
    """Register the deals tool with the tool registry."""
    global _lazy_context
    _lazy_context = getattr(registry, "context", None)
    def _deals_sync(action: str, **kwargs: Any) -> dict[str, Any]:
        """Sync wrapper — the registry dispatches synchronously."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # Already in a loop (shouldn't happen via registry, but be safe):
            # run in a fresh thread with its own loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, deals(action, **kwargs)).result()
        return asyncio.run(deals(action, **kwargs))

    registry.register(
        name="deals",
        fn=_deals_sync,
        description="Search, track, compare, and find steals across Nigerian marketplaces (Jumia, Konga, Jiji, Kara, SLOT, Temu, AliExpress, eBay, Banggood, Amazon)",
        capability="net.out",
        parameters={
            "action": {
                "type": "string",
                "enum": ["scan", "compare", "steals", "track", "watchlist", "check_watchlists",
                         "hunt", "flash", "deal_alerts", "price_history"],
                "description": "Action to perform (compare = cross-site cheapest-price comparison; hunt/flash/deal_alerts/price_history use the DealHunter price-drop engine)",
            },
            "query": {"type": "string", "description": "Search query (for scan/compare)"},
            "sites": {"type": "array", "description": "Sites to scan (for scan/compare)"},
            "max_price": {"type": "number", "description": "Maximum price in Naira"},
            "threshold": {"type": "number", "description": "Steal score threshold (0-100)"},
            "url": {"type": "string", "description": "Product URL to track"},
            "user_id": {"type": "string", "description": "User ID for watchlist"},
            "target_price": {"type": "number", "description": "Target price for alert"},
        },
    )
