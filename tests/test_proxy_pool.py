"""Tests for proxy pool connector — scraper, validator, scoring, rotation."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nomorals.connectors.proxies import Proxy, ProxyPool, ProxyPoolConnector


class TestProxyPool:
    """Test proxy pool manager."""
    
    def test_proxy_pool_add_and_get(self, tmp_path):
        """Proxy pool adds and retrieves proxies."""
        db_path = tmp_path / "test_proxies.db"
        pool = ProxyPool(db_path=db_path)
        
        # Add proxy
        proxy = Proxy(
            ip="192.168.1.1",
            port=8080,
            protocol="http",
            country="US",
            anonymity="elite",
            source="test",
            score=0.8,
            working=True,
            latency_ms=200.0
        )
        pool.add(proxy)
        
        # Retrieve
        all_proxies = pool.get_all()
        assert len(all_proxies) == 1
        assert all_proxies[0].ip == "192.168.1.1"
        assert all_proxies[0].port == 8080
        assert all_proxies[0].score == 0.8
    
    def test_proxy_pool_best_by_score(self, tmp_path):
        """Proxy pool best() returns highest-scored proxies."""
        db_path = tmp_path / "test_proxies.db"
        pool = ProxyPool(db_path=db_path)
        
        # Add proxies with different scores
        for i in range(5):
            proxy = Proxy(
                ip=f"10.0.0.{i}",
                port=8080,
                protocol="http",
                score=0.5 + (i * 0.1),
                working=True,
                latency_ms=100.0 + (i * 50)
            )
            pool.add(proxy)
        
        # Get best 2
        best = pool.best(limit=2)
        assert len(best) == 2
        assert best[0].score >= best[1].score
        assert best[0].ip == "10.0.0.4"  # Highest score
    
    def test_proxy_pool_best_filter_country(self, tmp_path):
        """Proxy pool best() filters by country."""
        db_path = tmp_path / "test_proxies.db"
        pool = ProxyPool(db_path=db_path)
        
        # Add proxies from different countries
        pool.add(Proxy(ip="1.1.1.1", port=80, country="US", working=True, score=0.9))
        pool.add(Proxy(ip="2.2.2.2", port=80, country="NG", working=True, score=0.8))
        pool.add(Proxy(ip="3.3.3.3", port=80, country="US", working=True, score=0.7))
        
        # Filter by US
        us_proxies = pool.best(country="US", limit=10)
        assert len(us_proxies) == 2
        assert all(p.country == "US" for p in us_proxies)
    
    def test_proxy_pool_best_filter_protocol(self, tmp_path):
        """Proxy pool best() filters by protocol."""
        db_path = tmp_path / "test_proxies.db"
        pool = ProxyPool(db_path=db_path)
        
        pool.add(Proxy(ip="1.1.1.1", port=80, protocol="http", working=True, score=0.9))
        pool.add(Proxy(ip="2.2.2.2", port=1080, protocol="socks5", working=True, score=0.8))
        pool.add(Proxy(ip="3.3.3.3", port=443, protocol="https", working=True, score=0.7))
        
        # Filter by socks5
        socks_proxies = pool.best(protocol="socks5", limit=10)
        assert len(socks_proxies) == 1
        assert socks_proxies[0].protocol == "socks5"
    
    def test_proxy_pool_record_check_working(self, tmp_path):
        """Proxy pool record_check updates working proxy."""
        db_path = tmp_path / "test_proxies.db"
        pool = ProxyPool(db_path=db_path)
        
        # Add proxy
        pool.add(Proxy(ip="5.5.5.5", port=8080, working=False, fail_streak=3))
        
        # Record successful check
        pool.record_check("5.5.5.5", 8080, working=True, latency_ms=150.0)
        
        # Verify updated
        proxies = pool.get_all()
        assert len(proxies) == 1
        assert proxies[0].working is True
        assert proxies[0].fail_streak == 0  # Reset on success
        assert proxies[0].latency_ms == 150.0
    
    def test_proxy_pool_record_check_failure(self, tmp_path):
        """Proxy pool record_check increments fail streak on failure."""
        db_path = tmp_path / "test_proxies.db"
        pool = ProxyPool(db_path=db_path)
        
        # Add working proxy
        pool.add(Proxy(ip="6.6.6.6", port=8080, working=True, fail_streak=0))
        
        # Record failed check
        pool.record_check("6.6.6.6", 8080, working=False, latency_ms=0.0)
        
        # Verify updated
        proxies = pool.get_all()
        assert proxies[0].working is False
        assert proxies[0].fail_streak == 1
    
    def test_proxy_pool_prune_old(self, tmp_path):
        """Proxy pool prune removes old proxies."""
        db_path = tmp_path / "test_proxies.db"
        pool = ProxyPool(db_path=db_path)
        
        # Add old proxy (checked 10 days ago)
        old_time = time.time() - (10 * 86400)
        pool.add(Proxy(ip="7.7.7.7", port=8080, last_check=old_time))
        
        # Add recent proxy
        pool.add(Proxy(ip="8.8.8.8", port=8080, last_check=time.time()))
        
        # Prune proxies older than 7 days
        pruned = pool.prune(max_age_days=7)
        assert pruned == 1
        
        # Verify only recent proxy remains
        proxies = pool.get_all()
        assert len(proxies) == 1
        assert proxies[0].ip == "8.8.8.8"


class TestProxyPoolConnector:
    """Test proxy pool connector with mocked scraping."""
    
    def test_connector_status(self, tmp_path):
        """Proxy pool connector returns status."""
        connector = ProxyPoolConnector()
        connector.pool = ProxyPool(db_path=tmp_path / "test.db")
        
        # Add some proxies
        connector.pool.add(Proxy(ip="1.1.1.1", port=80, working=True, score=0.8))
        connector.pool.add(Proxy(ip="2.2.2.2", port=80, working=False, score=0.3))
        connector.pool.add(Proxy(ip="3.3.3.3", port=80, working=True, score=0.9))
        
        status = connector.status()
        assert status.connected
        assert "2/3 working" in status.account
    
    def test_connector_scrape_text_source(self, tmp_path):
        """Proxy pool scrapes text format sources."""
        connector = ProxyPoolConnector()
        connector.pool = ProxyPool(db_path=tmp_path / "test.db")
        
        # Mock text response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.text = "192.168.1.1:8080\n192.168.1.2:3128\n192.168.1.3:1080\n"
        
        source = {"name": "test", "url": "http://test.com", "format": "text"}
        
        with patch.object(connector.http, "get", return_value=mock_response):
            proxies = connector._scrape_source(source)
        
        assert len(proxies) == 3
        assert proxies[0].ip == "192.168.1.1"
        assert proxies[0].port == 8080
        assert proxies[1].ip == "192.168.1.2"
        assert proxies[1].port == 3128
    
    def test_connector_scrape_json_source(self, tmp_path):
        """Proxy pool scrapes JSON format sources."""
        connector = ProxyPoolConnector()
        connector.pool = ProxyPool(db_path=tmp_path / "test.db")
        
        # Mock JSON response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.text = '{"data": [{"ip": "10.0.0.1", "port": "8080", "protocols": ["http"], "country": "US"}, {"ip": "10.0.0.2", "port": "3128", "protocols": ["https"], "country": "NG"}]}'
        
        source = {"name": "test", "url": "http://test.com", "format": "json"}
        
        with patch.object(connector.http, "get", return_value=mock_response):
            proxies = connector._scrape_source(source)
        
        assert len(proxies) == 2
        assert proxies[0].ip == "10.0.0.1"
        assert proxies[0].port == 8080
        assert proxies[0].country == "US"
        assert proxies[1].ip == "10.0.0.2"
        assert proxies[1].country == "NG"
    
    def test_connector_rotate_top_proxies(self, tmp_path):
        """Proxy pool rotate picks from top proxies."""
        connector = ProxyPoolConnector()
        connector.pool = ProxyPool(db_path=tmp_path / "test.db")
        
        # Add proxies with different scores
        for i in range(10):
            connector.pool.add(Proxy(
                ip=f"10.0.0.{i}",
                port=8080,
                working=True,
                score=0.5 + (i * 0.05)
            ))
        
        # Rotate 10 times
        selected = set()
        for _ in range(10):
            proxy = connector.rotate()
            assert proxy is not None
            selected.add(proxy.ip)
        
        # Should pick from top 3 (rotation)
        assert len(selected) <= 3
    
    def test_connector_stats(self, tmp_path):
        """Proxy pool stats returns pool statistics."""
        connector = ProxyPoolConnector()
        connector.pool = ProxyPool(db_path=tmp_path / "test.db")
        
        # Add proxies
        connector.pool.add(Proxy(ip="1.1.1.1", port=80, country="US", source="src1", working=True, score=0.8, latency_ms=200))
        connector.pool.add(Proxy(ip="2.2.2.2", port=80, country="NG", source="src2", working=True, score=0.9, latency_ms=150))
        connector.pool.add(Proxy(ip="3.3.3.3", port=80, country="US", source="src1", working=False, score=0.3, latency_ms=0))
        
        stats = connector.stats()
        assert stats["total"] == 3
        assert stats["working"] == 2
        assert stats["countries"] == 2
        assert stats["sources"] == 2
        assert 0 < stats["avg_score"] <= 1.0
        assert stats["avg_latency_ms"] > 0


class TestProxy:
    """Test Proxy dataclass."""
    
    def test_proxy_to_url(self):
        """Proxy.to_url() returns correct URL."""
        proxy = Proxy(ip="192.168.1.1", port=8080, protocol="http")
        assert proxy.to_url() == "http://192.168.1.1:8080"
        
        proxy_socks = Proxy(ip="10.0.0.1", port=1080, protocol="socks5")
        assert proxy_socks.to_url() == "socks5://10.0.0.1:1080"
    
    def test_proxy_to_dict(self):
        """Proxy.to_dict() returns complete dict."""
        proxy = Proxy(
            ip="192.168.1.1",
            port=8080,
            protocol="http",
            country="US",
            anonymity="elite",
            source="test",
            score=0.85,
            working=True,
            latency_ms=200.0,
            last_check=time.time(),
            fail_streak=0,
            uptime_pct=95.0
        )
        
        data = proxy.to_dict()
        assert data["ip"] == "192.168.1.1"
        assert data["port"] == 8080
        assert data["protocol"] == "http"
        assert data["country"] == "US"
        assert data["anonymity"] == "elite"
        assert data["score"] == 0.85
        assert data["working"] is True
        assert data["latency_ms"] == 200.0
        assert data["uptime_pct"] == 95.0
