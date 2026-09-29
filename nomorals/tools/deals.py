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

__all__ = ["register"]

_log = get_logger(__name__)

_engine: NaijaShoppingEngine | None = None


def _get_engine() -> NaijaShoppingEngine:
    """Get or create the shopping engine singleton."""
    global _engine
    if _engine is None:
        raise RuntimeError("NaijaShoppingEngine not initialized. Call init_deals_tool() first.")
    return _engine


def init_deals_tool(engine: NaijaShoppingEngine) -> None:
    """Initialize the deals tool with a shopping engine instance."""
    global _engine
    _engine = engine
    _log.info("Deals tool initialized")


async def deals(action: str, **kwargs: Any) -> dict[str, Any]:
    """Deals tool - search, track, and find steals across Nigerian marketplaces.
    
    Args:
        action: One of: scan, steals, track, watchlist, check_watchlists
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
    
    else:
        return {"error": f"Unknown action: {action}"}


def register(registry: Any) -> None:
    """Register the deals tool with the tool registry."""
    registry.register(
        name="deals",
        func=deals,
        description="Search, track, and find steals across Nigerian marketplaces (Jumia, Konga, Jiji, Temu, AliExpress)",
        parameters={
            "action": {
                "type": "string",
                "enum": ["scan", "steals", "track", "watchlist", "check_watchlists"],
                "description": "Action to perform",
            },
            "query": {"type": "string", "description": "Search query (for scan)"},
            "max_price": {"type": "number", "description": "Maximum price in Naira"},
            "threshold": {"type": "number", "description": "Steal score threshold (0-100)"},
            "url": {"type": "string", "description": "Product URL to track"},
            "user_id": {"type": "string", "description": "User ID for watchlist"},
            "target_price": {"type": "number", "description": "Target price for alert"},
        },
    )
