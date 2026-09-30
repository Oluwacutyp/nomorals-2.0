"""Deal hunter scheduled skill - runs scans and alerts on steals.

This skill runs the Naija Shopping Engine on a schedule (twice daily + flash sale windows),
detects steals, and sends alerts via Telegram/WhatsApp.

Usage:
    skill = DealHunterSkill(shopping_engine, scheduler, chat_gateway)
    await skill.setup_schedule()  # Creates cron jobs
    
    # Or run manually:
    await skill.execute_scan(categories=["phones", "laptops"], steal_threshold=70)
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..integrations.naija_shopping import NaijaShoppingEngine
from ..scheduler import Scheduler
from ..skills import SkillManager

__all__ = ["DealHunterSkill"]

_log = get_logger(__name__)

# Default scan schedule
DEFAULT_SCANS = [
    # Morning scan (8 AM)
    {"cron_expr": "0 8 * * *", "categories": ["phones", "laptops", "electronics"]},
    # Evening scan (6 PM)
    {"cron_expr": "0 18 * * *", "categories": ["fashion", "home", "appliances"]},
    # Jumia flash sale window (12 PM - 2 PM)
    {"cron_expr": "0 12 * * *", "categories": ["flash_sale"], "sites": ["jumia"]},
]


class DealHunterSkill:
    """Scheduled deal hunter that scans and alerts on steals."""
    
    def __init__(
        self,
        shopping_engine: NaijaShoppingEngine,
        scheduler: Scheduler,
        chat_gateway: Any = None,  # Telegram/WhatsApp gateway
        user_id: str = "",
    ) -> None:
        self.engine = shopping_engine
        self.scheduler = scheduler
        self.chat = chat_gateway
        self.user_id = user_id
        _log.info("Deal Hunter Skill initialized")
    
    async def setup_schedule(self) -> list[str]:
        """Set up scheduled scan jobs.
        
        Returns:
            List of cron job IDs
        """
        job_ids = []
        
        for scan_config in DEFAULT_SCANS:
            job_id = await self.scheduler.add_cron_job(
                cron_expr=scan_config["cron_expr"],
                action="deal_hunter_scan",
                params={
                    "categories": scan_config["categories"],
                    "sites": scan_config.get("sites"),
                    "steal_threshold": 70,
                },
                handler=self._handle_scheduled_scan,
            )
            job_ids.append(job_id)
            _log.info(f"Scheduled deal scan: {scan_config['cron_expr']} for {scan_config['categories']}")
        
        return job_ids
    
    async def _handle_scheduled_scan(self, params: dict[str, Any]) -> None:
        """Handler for scheduled scans."""
        categories = params.get("categories", [])
        sites = params.get("sites")
        threshold = params.get("steal_threshold", 70)
        
        await self.execute_scan(
            categories=categories,
            sites=sites,
            steal_threshold=threshold,
        )
    
    async def execute_scan(
        self,
        *,
        categories: list[str],
        sites: list[str] | None = None,
        steal_threshold: float = 70,
        max_price: float = float("inf"),
    ) -> dict[str, Any]:
        """Execute a deal scan and alert on steals.
        
        Args:
            categories: Categories to scan
            sites: Sites to scan (default: all)
            steal_threshold: Minimum steal score to alert on
            max_price: Maximum price filter
            
        Returns:
            Dict with scan results
        """
        _log.info(f"Starting deal scan: {categories}")
        
        all_steals = []
        
        for category in categories:
            try:
                # Scan category
                snapshots = await self.engine.scan_category(
                    category,
                    sites=sites,
                    max_price=max_price,
                )
                
                _log.info(f"Scanned {category}: {len(snapshots)} products")
                
                # Compute steal scores for each product
                seen_urls = set()
                for snapshot in snapshots:
                    if snapshot.product_url in seen_urls:
                        continue
                    seen_urls.add(snapshot.product_url)
                    
                    steal = await self.engine.compute_steal_score(snapshot.product_url)
                    
                    if steal and steal.steal_score >= steal_threshold:
                        all_steals.append(steal)
                        _log.info(f"🔥 Steal detected: {steal.title} (score: {steal.steal_score:.0f})")
                        
                        # Send alert
                        await self._send_steal_alert(steal)
                
            except Exception as e:
                _log.error(f"Failed to scan {category}: {e}")
        
        # Check watchlists
        triggered = await self.engine.check_watchlists()
        for watchlist in triggered:
            await self._send_watchlist_alert(watchlist)
        
        return {
            "categories_scanned": categories,
            "steals_found": len(all_steals),
            "watchlists_triggered": len(triggered),
            "steals": [s.to_dict() for s in all_steals],
        }
    
    async def _send_steal_alert(self, steal: Any) -> None:
        """Send steal alert via chat gateway."""
        if not self.chat or not self.user_id:
            return
        
        message = steal.to_message()
        
        try:
            await self.chat.send_message(self.user_id, message)
            _log.info(f"Sent steal alert to {self.user_id}")
        except Exception as e:
            _log.error(f"Failed to send steal alert: {e}")
    
    async def _send_watchlist_alert(self, watchlist: Any) -> None:
        """Send watchlist price drop alert."""
        if not self.chat:
            return
        
        message = (
            f"🎯 **Price Drop Alert!**\n\n"
            f"**{watchlist.title or watchlist.product_url}**\n"
            f"💰 Now: ₦{watchlist.last_price:,.0f}\n"
            f"🎯 Target: ₦{watchlist.target_price:,.0f}\n"
            f"🏪 {watchlist.site}\n"
            f"🔗 {watchlist.product_url}"
        )
        
        try:
            await self.chat.send_message(watchlist.user_id, message)
            _log.info(f"Sent watchlist alert to {watchlist.user_id}")
        except Exception as e:
            _log.error(f"Failed to send watchlist alert: {e}")
    
    async def get_recent_steals(self, *, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent steals from the database."""
        since = time.time() - (days * 24 * 3600)
        
        rows = self.engine.db.query("""
            SELECT * FROM steals
            WHERE discovered_at >= ?
            ORDER BY steal_score DESC
            LIMIT ?
        """, (since, limit))
        
        return [dict(r) for r in rows]
    
    async def add_to_watchlist(
        self,
        product_url: str,
        target_price: float,
        *,
        user_id: str = "",
    ) -> dict[str, Any]:
        """Add a product to the watchlist."""
        uid = user_id or self.user_id
        if not uid:
            return {"error": "user_id required"}
        
        watchlist = await self.engine.track_url(product_url, uid, target_price=target_price)
        
        return {
            "watchlist_id": watchlist.watchlist_id,
            "product_url": watchlist.product_url,
            "target_price": watchlist.target_price,
            "status": "tracking",
        }
    
    async def get_watchlist(self, *, user_id: str = "") -> list[dict[str, Any]]:
        """Get user's watchlist."""
        uid = user_id or self.user_id
        if not uid:
            return []
        
        rows = self.engine.db.query(
            "SELECT * FROM watchlists WHERE user_id = ? ORDER BY created_at DESC",
            (uid,)
        )
        
        return [dict(r) for r in rows]


def register_deal_hunter_skill(
    skill_manager: SkillManager,
    deal_hunter: DealHunterSkill,
) -> None:
    """Register the deal hunter as a reusable skill."""
    
    async def skill_handler(params: dict[str, Any]) -> dict[str, Any]:
        action = params.get("action", "scan")
        
        if action == "scan":
            return await deal_hunter.execute_scan(
                categories=params.get("categories", ["electronics"]),
                sites=params.get("sites"),
                steal_threshold=params.get("threshold", 70),
            )
        elif action == "steals":
            steals = await deal_hunter.get_recent_steals(
                days=params.get("days", 7),
                limit=params.get("limit", 10),
            )
            return {"steals": steals}
        elif action == "track":
            return await deal_hunter.add_to_watchlist(
                product_url=params.get("url", ""),
                target_price=params.get("target_price", 0),
            )
        elif action == "watchlist":
            items = await deal_hunter.get_watchlist()
            return {"watchlist": items}
        else:
            return {"error": f"Unknown action: {action}"}
    
    skill_manager.register_skill(
        skill_id="deal_hunter",
        name="Naija Deal Hunter",
        description="Find steals across Nigerian marketplaces (Jumia, Konga, Jiji, Temu, AliExpress)",
        handler=skill_handler,
        parameters={
            "action": {
                "type": "string",
                "enum": ["scan", "steals", "track", "watchlist"],
                "description": "Action to perform",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categories to scan",
            },
            "threshold": {
                "type": "number",
                "description": "Steal score threshold (0-100)",
            },
            "url": {
                "type": "string",
                "description": "Product URL to track",
            },
            "target_price": {
                "type": "number",
                "description": "Target price for alert",
            },
        },
    )
    
    _log.info("Registered deal_hunter skill")
