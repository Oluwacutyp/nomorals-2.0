#!/usr/bin/env python3
"""Simple test runner for connector system (no pytest required)."""

import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nomorals.connectors.base import BaseConnector, ConnectorStatus
from nomorals.connectors.vault import CredentialVault
from nomorals.connectors.patterns import ConnectionPattern, PatternRegistry
from nomorals.connectors.finance import MonoConnector, Finance
from nomorals.connectors.cards import PrivacyCardsConnector, VirtualCard
from nomorals.connectors.proxies import ProxyPool, Proxy
from nomorals.connectors.commerce_ng import DealHunter, ProductListing


def test_vault():
    """Test credential vault."""
    print("Testing vault...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        vault = CredentialVault(db_path=db_path)
        
        # Store and retrieve
        secret = {"api_key": "sk_test_123456", "account_id": "acc_789"}
        vault.store("test", "api_key", secret)
        retrieved = vault.get("test", "api_key")
        
        assert retrieved == secret, f"Expected {secret}, got {retrieved}"
        print("  ✓ Vault encrypt/decrypt works")
        
        # List labels (no values)
        labels = vault.list_labels("test")
        assert labels == ["api_key"], f"Expected ['api_key'], got {labels}"
        assert "sk_test_123456" not in str(labels), "Secret leaked in labels!"
        print("  ✓ Vault list_labels() never leaks values")
        
        # Delete
        deleted = vault.delete("test", "api_key")
        assert deleted is True
        assert vault.get("test", "api_key") is None
        print("  ✓ Vault delete works")
    
    print("✓ Vault tests passed\n")


def test_patterns():
    """Test connection patterns."""
    print("Testing patterns...")
    
    registry = PatternRegistry()
    patterns = registry.list()
    
    assert len(patterns) == 10, f"Expected 10 patterns, got {len(patterns)}"
    print(f"  ✓ Registry has all 10 patterns")
    
    # Check descriptions
    for pattern in ConnectionPattern:
        desc = registry.describe(pattern)
        assert desc, f"Pattern {pattern} has no description"
        assert len(desc) > 10, f"Pattern {pattern} description too short"
    
    print("  ✓ All patterns have descriptions")
    print("✓ Pattern tests passed\n")


def test_connector_status():
    """Test connector status."""
    print("Testing connector status...")
    
    class TestConnector(BaseConnector):
        name = "test"
        
        def status(self):
            return ConnectorStatus(connected=True, account="test_account")
        
        def connect_url(self):
            return "https://example.com/connect"
        
        def disconnect(self):
            return {"disconnected": True}
    
    connector = TestConnector()
    status = connector.status()
    
    assert status.connected is True
    assert status.account == "test_account"
    assert status.last_check  # Auto-generated timestamp
    
    data = status.to_dict()
    assert data["connected"] is True
    assert "last_check" in data
    
    print("  ✓ Connector status works")
    print("✓ Connector status tests passed\n")


def test_proxy_pool():
    """Test proxy pool."""
    print("Testing proxy pool...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        pool = ProxyPool(db_path=db_path)
        
        # Add proxies
        pool.add(Proxy(ip="1.1.1.1", port=80, score=0.9, working=True))
        pool.add(Proxy(ip="2.2.2.2", port=80, score=0.7, working=True))
        pool.add(Proxy(ip="3.3.3.3", port=80, score=0.5, working=False))
        
        # Get all
        all_proxies = pool.get_all()
        assert len(all_proxies) == 3
        print(f"  ✓ Pool stores {len(all_proxies)} proxies")
        
        # Get best
        best = pool.best(limit=2)
        assert len(best) == 2
        assert best[0].score >= best[1].score
        print(f"  ✓ Pool best() returns highest-scored proxies")
        
        # Record check
        pool.record_check("3.3.3.3", 80, working=True, latency_ms=100.0)
        proxies = pool.get_all()
        updated = [p for p in proxies if p.ip == "3.3.3.3"][0]
        assert updated.working is True
        assert updated.fail_streak == 0
        print("  ✓ Pool record_check updates proxy stats")
    
    print("✓ Proxy pool tests passed\n")


def test_deal_hunter():
    """Test deal hunter."""
    print("Testing deal hunter...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        hunter = DealHunter(db_path=db_path)
        
        # Create listings
        listings = [
            ProductListing(listing_id="1", marketplace="jumia", title="iPhone 15", url="https://...", price_ngn=800000),
            ProductListing(listing_id="2", marketplace="jumia", title="iPhone 15", url="https://...", price_ngn=750000),
            ProductListing(listing_id="3", marketplace="jumia", title="iPhone 15", url="https://...", price_ngn=450000),  # Steal! (41% below)
            ProductListing(listing_id="4", marketplace="konga", title="iPhone 15", url="https://...", price_ngn=780000),
        ]
        
        # Find steals (median ~765000, 35% below = <497250)
        steals = hunter.find_steals(listings, median_price_ngn=765000)
        
        assert len(steals) >= 1, f"Expected at least 1 steal, got {len(steals)}"
        assert steals[0]["listing"]["price_ngn"] == 450000
        assert steals[0]["discount_pct"] >= 35.0
        print(f"  ✓ Deal hunter found {len(steals)} steal(s)")
        
        # Track price
        watch_id = hunter.track_price("https://jumia.com.ng/product", "user_123", target_price=700000)
        assert watch_id.startswith("watch_")
        print(f"  ✓ Deal hunter tracks price (watch_id: {watch_id})")
    
    print("✓ Deal hunter tests passed\n")


def test_virtual_card():
    """Test virtual card."""
    print("Testing virtual card...")
    
    card = VirtualCard(
        token="card_test",
        provider="privacy_com",
        card_type="UNLOCKED",
        state="OPEN",
        last_four="1234",
        masked_pan="****1234",
        spend_limit=10000,
        spend_limit_duration="MONTHLY",
        memo="Test card",
        pan="4111111111111234",
        cvv="123",
        exp_month="12",
        exp_year="2028"
    )
    
    # Masked
    data_masked = card.to_dict(reveal=False)
    assert data_masked["pan"] == "****1234"
    assert data_masked["cvv"] == "***"
    assert data_masked["exp_month"] == "**"
    print("  ✓ Virtual card masks sensitive fields")
    
    # Revealed
    data_revealed = card.to_dict(reveal=True)
    assert data_revealed["pan"] == "4111111111111234"
    assert data_revealed["cvv"] == "123"
    assert data_revealed["exp_month"] == "12"
    print("  ✓ Virtual card reveals fields when requested")
    
    print("✓ Virtual card tests passed\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CONNECTOR SYSTEM TEST SUITE")
    print("="*60 + "\n")
    
    try:
        test_vault()
        test_patterns()
        test_connector_status()
        test_proxy_pool()
        test_deal_hunter()
        test_virtual_card()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
