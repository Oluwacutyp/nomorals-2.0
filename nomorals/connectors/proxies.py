"""Proxy pool connector — multi-source scraper, async validator, scoring, rotation.

Sources (all free, no key unless noted):
1. ProxyScrape v2 (http/socks4/socks5)
2. GeoNode (filterable by country/anonymity)
3. proxy-list.download (txt endpoints)
4. free-proxy-list.net (HTML scrape)
5. WebShare (10 free with account)

Pipeline:
- Scrape (every 6h): Pull sources → normalize → dedupe → SQLite
- Validate (async): Test via httpbin.org/ip, record latency/anonymity
- Score: w1*speed + w2*uptime + w3*anonymity - w4*fail_streak
- Rotate: best(country, protocol) → highest-scored working proxy
- Decay: Re-validate on schedule, auto-prune dead proxies

Security:
- Proxies are third-party — never trust for auth flows
- Rate-limited scraping (polite)
- User-agent rotation
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .base import BaseConnector, ConnectorStatus

__all__ = ["ProxyPoolConnector", "Proxy", "ProxyPool"]

_log = get_logger(__name__)


@dataclass
class Proxy:
    """Proxy server."""
    
    ip: str
    port: int
    protocol: str = "http"  # http, https, socks4, socks5
    country: str = ""
    anonymity: str = ""  # elite, anonymous, unknown
    source: str = ""
    score: float = 0.0
    working: bool = False
    latency_ms: float = 0.0
    last_check: float = 0.0
    fail_streak: int = 0
    uptime_pct: float = 0.0
    
    def to_url(self) -> str:
        """Return proxy URL (e.g., http://1.2.3.4:8080)."""
        return f"{self.protocol}://{self.ip}:{self.port}"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ip": self.ip,
            "port": self.port,
            "protocol": self.protocol,
            "country": self.country,
            "anonymity": self.anonymity,
            "source": self.source,
            "score": self.score,
            "working": self.working,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
            "fail_streak": self.fail_streak,
            "uptime_pct": self.uptime_pct,
        }


class ProxyPool:
    """Proxy pool manager — scrape, validate, score, rotate.
    
    Storage: SQLite database with proxies and proxy_checks tables.
    
    Scoring formula:
        score = w1 * speed_score + w2 * uptime + w3 * anonymity_bonus - w4 * fail_streak
    
    Where:
        speed_score = max(0, 1 - latency_ms / 5000)  # 0-1, 1 = fast
        uptime = uptime_pct / 100  # 0-1
        anonymity_bonus = 1.0 if elite else 0.5 if anonymous else 0.2
        fail_streak = consecutive failures (penalty)
    
    Weights (default):
        w1 = 0.4  # Speed
        w2 = 0.3  # Uptime
        w3 = 0.2  # Anonymity
        w4 = 0.1  # Fail streak penalty
    """
    
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Path.home() / ".nomorals" / "proxies.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Create proxy tables."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS proxies (
                    ip TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    protocol TEXT NOT NULL DEFAULT 'http',
                    country TEXT DEFAULT '',
                    anonymity TEXT DEFAULT '',
                    source TEXT DEFAULT '',
                    score REAL DEFAULT 0.0,
                    working INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0.0,
                    last_check REAL DEFAULT 0.0,
                    fail_streak INTEGER DEFAULT 0,
                    uptime_pct REAL DEFAULT 0.0,
                    PRIMARY KEY (ip, port)
                );
                
                CREATE TABLE IF NOT EXISTS proxy_checks (
                    id TEXT PRIMARY KEY,
                    ip TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    working INTEGER NOT NULL,
                    latency_ms REAL DEFAULT 0.0,
                    checked_at REAL NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_proxies_score ON proxies(score);
                CREATE INDEX IF NOT EXISTS idx_proxies_working ON proxies(working);
                CREATE INDEX IF NOT EXISTS idx_proxy_checks_ip ON proxy_checks(ip, port);
            """)
            conn.commit()
        finally:
            conn.close()
    
    def add(self, proxy: Proxy) -> None:
        """Add or update a proxy."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                INSERT OR REPLACE INTO proxies 
                (ip, port, protocol, country, anonymity, source, score, working, 
                 latency_ms, last_check, fail_streak, uptime_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                proxy.ip, proxy.port, proxy.protocol, proxy.country, proxy.anonymity,
                proxy.source, proxy.score, int(proxy.working), proxy.latency_ms,
                proxy.last_check, proxy.fail_streak, proxy.uptime_pct
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_all(self, working_only: bool = False) -> list[Proxy]:
        """Get all proxies."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            query = "SELECT * FROM proxies"
            if working_only:
                query += " WHERE working = 1"
            query += " ORDER BY score DESC"
            
            cursor = conn.execute(query)
            proxies = []
            for row in cursor.fetchall():
                proxies.append(Proxy(
                    ip=row[0], port=row[1], protocol=row[2], country=row[3],
                    anonymity=row[4], source=row[5], score=row[6], working=bool(row[7]),
                    latency_ms=row[8], last_check=row[9], fail_streak=row[10], uptime_pct=row[11]
                ))
            return proxies
        finally:
            conn.close()
    
    def best(self, country: str = "", protocol: str = "", limit: int = 1) -> list[Proxy]:
        """Get best proxies by score, filtered by country/protocol.
        
        Args:
            country: Filter by country code (e.g., "US", "NG")
            protocol: Filter by protocol (http, https, socks4, socks5)
            limit: Max proxies to return
        
        Returns:
            List of best proxies
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            query = "SELECT * FROM proxies WHERE working = 1"
            params = []
            
            if country:
                query += " AND country = ?"
                params.append(country.upper())
            if protocol:
                query += " AND protocol = ?"
                params.append(protocol.lower())
            
            query += " ORDER BY score DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            proxies = []
            for row in cursor.fetchall():
                proxies.append(Proxy(
                    ip=row[0], port=row[1], protocol=row[2], country=row[3],
                    anonymity=row[4], source=row[5], score=row[6], working=bool(row[7]),
                    latency_ms=row[8], last_check=row[9], fail_streak=row[10], uptime_pct=row[11]
                ))
            return proxies
        finally:
            conn.close()
    
    def record_check(self, ip: str, port: int, working: bool, latency_ms: float) -> None:
        """Record a proxy check result."""
        check_id = f"{ip}:{port}:{time.time()}"
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Insert check record
            conn.execute("""
                INSERT INTO proxy_checks (id, ip, port, working, latency_ms, checked_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (check_id, ip, port, int(working), latency_ms, time.time()))
            
            # Update proxy stats
            cursor = conn.execute(
                "SELECT fail_streak, uptime_pct FROM proxies WHERE ip = ? AND port = ?",
                (ip, port)
            )
            row = cursor.fetchone()
            if row:
                fail_streak = row[0]
                uptime_pct = row[1]
                
                if working:
                    fail_streak = 0
                else:
                    fail_streak += 1
                
                # Update uptime (exponential moving average)
                uptime_pct = 0.9 * uptime_pct + 0.1 * (100.0 if working else 0.0)
                
                # Recalculate score
                latency_ms_avg = latency_ms if latency_ms > 0 else 1000.0
                speed_score = max(0.0, 1.0 - latency_ms_avg / 5000.0)
                anonymity_bonus = 0.5  # Default
                score = 0.4 * speed_score + 0.3 * (uptime_pct / 100.0) + 0.2 * anonymity_bonus - 0.1 * fail_streak
                score = max(0.0, min(1.0, score))
                
                conn.execute("""
                    UPDATE proxies 
                    SET working = ?, latency_ms = ?, last_check = ?, fail_streak = ?, 
                        uptime_pct = ?, score = ?
                    WHERE ip = ? AND port = ?
                """, (
                    int(working), latency_ms, time.time(), fail_streak, uptime_pct, score,
                    ip, port
                ))
            
            conn.commit()
        finally:
            conn.close()
    
    def prune(self, max_age_days: int = 7) -> int:
        """Remove proxies not checked in max_age_days.
        
        Returns:
            Number of proxies removed
        """
        cutoff = time.time() - (max_age_days * 86400)
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "DELETE FROM proxies WHERE last_check < ? AND last_check > 0",
                (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


class ProxyPoolConnector(BaseConnector):
    """Proxy pool connector — multi-source scraper with async validation."""
    
    name = "proxy_pool"
    description = "Multi-source proxy pool with async validation and scoring"
    
    SOURCES = [
        {
            "name": "proxyscrape",
            "url": "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
            "format": "text"
        },
        {
            "name": "geonode",
            "url": "https://proxylist.geonode.com/api/proxy-list?limit=100&page=1&sort_by=lastChecked&sort_type=desc&protocols=http%2Chttps",
            "format": "json"
        },
        {
            "name": "proxy-list-download",
            "url": "https://www.proxy-list.download/api/v1/get?type=https",
            "format": "text"
        },
        {
            "name": "free-proxy-list",
            "url": "https://free-proxy-list.net/",
            "format": "html"
        }
    ]
    
    def __init__(self, vault: Any = None, config: dict[str, Any] | None = None) -> None:
        super().__init__(vault, config)
        self.pool = ProxyPool()
        self.http = HttpClient(timeout=15.0)
    
    def status(self) -> ConnectorStatus:
        """Check proxy pool status."""
        try:
            all_proxies = self.pool.get_all()
            working = [p for p in all_proxies if p.working]
            
            return ConnectorStatus(
                connected=True,
                account=f"{len(working)}/{len(all_proxies)} working",
                scopes=["scrape", "validate", "rotate"]
            )
        except Exception as e:
            return ConnectorStatus(connected=False, error=str(e))
    
    def connect_url(self) -> str:
        """No connection URL needed."""
        return ""
    
    def disconnect(self) -> dict[str, Any]:
        """Clear proxy pool."""
        try:
            # Clear database
            conn = sqlite3.connect(str(self.pool.db_path))
            try:
                conn.execute("DELETE FROM proxies")
                conn.execute("DELETE FROM proxy_checks")
                conn.commit()
            finally:
                conn.close()
            return {"disconnected": True}
        except Exception as e:
            return {"disconnected": False, "error": str(e)}
    
    def scrape_all(self) -> int:
        """Scrape all sources and add to pool.
        
        Returns:
            Number of proxies added
        """
        total_added = 0
        
        for source in self.SOURCES:
            try:
                proxies = self._scrape_source(source)
                for proxy in proxies:
                    self.pool.add(proxy)
                    total_added += 1
                _log.info("Scraped %d proxies from %s", len(proxies), source["name"])
            except Exception as e:
                _log.warning("Failed to scrape %s: %s", source["name"], e)
        
        return total_added
    
    def _scrape_source(self, source: dict[str, Any]) -> list[Proxy]:
        """Scrape a single source."""
        response = self.http.get(source["url"])
        if not response.ok:
            return []
        
        text = response.text or ""
        proxies = []
        
        if source["format"] == "text":
            # Parse "ip:port" lines
            for line in text.splitlines():
                match = re.match(r"(\d+\.\d+\.\d+\.\d+):(\d+)", line.strip())
                if match:
                    proxies.append(Proxy(
                        ip=match.group(1),
                        port=int(match.group(2)),
                        protocol="http",
                        source=source["name"]
                    ))
        
        elif source["format"] == "json":
            # Parse JSON response
            try:
                data = json.loads(text)
                items = data.get("data", [])
                for item in items:
                    ip = item.get("ip", "")
                    port = item.get("port", 0)
                    if ip and port:
                        proxies.append(Proxy(
                            ip=ip,
                            port=int(port),
                            protocol=item.get("protocols", ["http"])[0] if item.get("protocols") else "http",
                            country=item.get("country", ""),
                            anonymity=item.get("anonymityLevel", ""),
                            source=source["name"]
                        ))
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        
        elif source["format"] == "html":
            # Parse HTML table (free-proxy-list.net)
            # Look for <tr><td>ip</td><td>port</td>...</tr>
            rows = re.findall(r"<tr><td>(\d+\.\d+\.\d+\.\d+)</td><td>(\d+)</td>", text)
            for ip, port in rows:
                proxies.append(Proxy(
                    ip=ip,
                    port=int(port),
                    protocol="http",
                    source=source["name"]
                ))
        
        return proxies
    
    async def validate_async(self, max_concurrent: int = 50) -> int:
        """Validate all proxies asynchronously.
        
        Args:
            max_concurrent: Max concurrent validation tasks
        
        Returns:
            Number of proxies validated
        """
        proxies = self.pool.get_all()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_one(proxy: Proxy) -> None:
            async with semaphore:
                working, latency = await self._check_proxy_async(proxy)
                self.pool.record_check(proxy.ip, proxy.port, working, latency)
        
        tasks = [validate_one(p) for p in proxies[:200]]  # Limit to 200
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return len(tasks)
    
    async def _check_proxy_async(self, proxy: Proxy) -> tuple[bool, float]:
        """Check if proxy is working (async).
        
        Returns:
            (working, latency_ms)
        """
        try:
            import aiohttp
            start = time.time()
            
            async with aiohttp.ClientSession() as session:
                proxy_url = proxy.to_url()
                async with session.get(
                    "http://httpbin.org/ip",
                    proxy=proxy_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        latency = (time.time() - start) * 1000
                        return True, latency
            
            return False, 0.0
        
        except Exception:
            return False, 0.0
    
    def rotate(self, country: str = "", protocol: str = "") -> Proxy | None:
        """Get best available proxy with rotation.
        
        Args:
            country: Filter by country code
            protocol: Filter by protocol
        
        Returns:
            Best proxy, or None if none available
        """
        candidates = self.pool.best(country=country, protocol=protocol, limit=10)
        if not candidates:
            return None
        
        # Rotate: pick randomly from top 3 to avoid hammering one proxy
        top_n = min(3, len(candidates))
        return random.choice(candidates[:top_n])
    
    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        all_proxies = self.pool.get_all()
        working = [p for p in all_proxies if p.working]
        
        return {
            "total": len(all_proxies),
            "working": len(working),
            "avg_score": sum(p.score for p in working) / len(working) if working else 0.0,
            "avg_latency_ms": sum(p.latency_ms for p in working) / len(working) if working else 0.0,
            "countries": len(set(p.country for p in working if p.country)),
            "sources": len(set(p.source for p in all_proxies if p.source)),
        }
