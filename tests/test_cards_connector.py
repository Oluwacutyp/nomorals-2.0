"""Tests for virtual cards connector — Privacy.com with mocked HTTP."""

from unittest.mock import Mock, patch

import pytest

from nomorals.connectors.cards import PrivacyCardsConnector, VirtualCard
from nomorals.connectors.vault import CredentialVault


class TestPrivacyCardsConnector:
    """Test Privacy.com virtual cards connector."""
    
    def test_cards_status_no_key(self, tmp_path):
        """Cards status returns disconnected when no API key."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        status = connector.status()
        assert not status.connected
        assert "No Privacy.com API key" in status.error
    
    def test_cards_status_connected(self, tmp_path):
        """Cards status returns connected when API key configured."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        status = connector.status()
        assert status.connected
        assert "Privacy.com" in status.account
    
    def test_cards_create_unlocked(self, tmp_path):
        """Cards create returns unlocked card."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "card": {
                "token": "card_token_123",
                "type": "UNLOCKED",
                "state": "OPEN",
                "pan": "4111111111111234",
                "cvv": "123",
                "exp_month": 12,
                "exp_year": 2028,
                "spend_limit": 0,
                "memo": "Test card"
            }
        }
        
        with patch.object(connector.http, "post_json", return_value=mock_response):
            card = connector.create_card(
                card_type="UNLOCKED",
                spend_limit=0,
                memo="Test card"
            )
        
        assert isinstance(card, VirtualCard)
        assert card.token == "card_token_123"
        assert card.card_type == "UNLOCKED"
        assert card.state == "OPEN"
        assert card.last_four == "1234"
        assert card.masked_pan == "****1234"
        assert card.memo == "Test card"
    
    def test_cards_create_single_use(self, tmp_path):
        """Cards create_purchase returns SINGLE_USE card."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "card": {
                "token": "card_su_456",
                "type": "SINGLE_USE",
                "state": "OPEN",
                "pan": "4222222222222567",
                "cvv": "456",
                "exp_month": 6,
                "exp_year": 2027,
                "spend_limit": 9999,  # $99.99
                "spend_limit_duration": "TRANSACTION",
                "memo": "Purchase: Amazon"
            }
        }
        
        with patch.object(connector.http, "post_json", return_value=mock_response):
            card = connector.create_for_purchase("Amazon", 9999)
        
        assert card.card_type == "SINGLE_USE"
        assert card.spend_limit == 9999
        assert card.spend_limit_duration == "TRANSACTION"
        assert "Amazon" in card.memo
    
    def test_cards_create_subscription(self, tmp_path):
        """Cards create_subscription returns MERCHANT_LOCKED card."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "card": {
                "token": "card_sub_789",
                "type": "MERCHANT_LOCKED",
                "state": "OPEN",
                "pan": "4333333333333890",
                "cvv": "789",
                "exp_month": 3,
                "exp_year": 2029,
                "spend_limit": 1500,  # $15.00/month
                "spend_limit_duration": "MONTHLY",
                "memo": "Subscription: Netflix"
            }
        }
        
        with patch.object(connector.http, "post_json", return_value=mock_response):
            card = connector.create_for_subscription("Netflix", 1500)
        
        assert card.card_type == "MERCHANT_LOCKED"
        assert card.spend_limit == 1500
        assert card.spend_limit_duration == "MONTHLY"
        assert "Netflix" in card.memo
    
    def test_cards_pan_masked_in_storage(self, tmp_path):
        """Cards PAN is masked when stored in vault."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response with full PAN
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "card": {
                "token": "card_mask_test",
                "type": "UNLOCKED",
                "state": "OPEN",
                "pan": "4444555566667777",
                "cvv": "999",
                "exp_month": 1,
                "exp_year": 2030,
                "spend_limit": 0,
                "memo": "Mask test"
            }
        }
        
        with patch.object(connector.http, "post_json", return_value=mock_response):
            card = connector.create_card()
        
        # Verify card object has full PAN
        assert card.pan == "4444555566667777"
        assert card.masked_pan == "****7777"
        
        # Verify stored version is masked
        stored = vault.get("privacy_cards", "card_card_mask_test")
        assert stored is not None
        assert stored["pan"] == "****7777"  # Masked
        assert "4444555566667777" not in stored["pan"]
        assert stored["cvv"] == "***"  # Masked
    
    def test_cards_get_reveal_false(self, tmp_path):
        """Cards get with reveal=False masks PAN/CVV."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "card": {
                "token": "card_get_test",
                "type": "UNLOCKED",
                "state": "OPEN",
                "pan": "4555666677778888",
                "cvv": "111",
                "exp_month": 5,
                "exp_year": 2028,
                "spend_limit": 0,
                "memo": ""
            }
        }
        
        with patch.object(connector.http, "get", return_value=mock_response):
            card = connector.get_card("card_get_test", reveal=False)
        
        data = card.to_dict(reveal=False)
        assert data["pan"] == "****8888"
        assert data["cvv"] == "***"
        assert data["exp_month"] == "**"
        assert data["exp_year"] == "**"
    
    def test_cards_get_reveal_true(self, tmp_path):
        """Cards get with reveal=True shows full PAN/CVV (checkout only)."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "card": {
                "token": "card_reveal_test",
                "type": "UNLOCKED",
                "state": "OPEN",
                "pan": "4666777788889999",
                "cvv": "222",
                "exp_month": 7,
                "exp_year": 2029,
                "spend_limit": 0,
                "memo": ""
            }
        }
        
        with patch.object(connector.http, "get", return_value=mock_response):
            card = connector.get_card("card_reveal_test", reveal=True)
        
        data = card.to_dict(reveal=True)
        assert data["pan"] == "4666777788889999"
        assert data["cvv"] == "222"
        assert data["exp_month"] == "7"
        assert data["exp_year"] == "2029"
    
    def test_cards_pause(self, tmp_path):
        """Cards pause updates state to PAUSED."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        
        with patch.object(connector.http, "put_json", return_value=mock_response):
            success = connector.pause_card("card_pause_test")
        
        assert success is True
    
    def test_cards_close(self, tmp_path):
        """Cards close updates state to CLOSED."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        
        with patch.object(connector.http, "put_json", return_value=mock_response):
            success = connector.close_card("card_close_test")
        
        assert success is True
    
    def test_cards_list(self, tmp_path):
        """Cards list returns all cards."""
        vault = CredentialVault(db_path=tmp_path / "test.db")
        connector = PrivacyCardsConnector(vault=vault)
        
        vault.store("privacy_cards", "api_key", {"key": "api_key_test_123"})
        
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "data": [
                {
                    "token": "card_1",
                    "type": "UNLOCKED",
                    "state": "OPEN",
                    "pan": "4111111111111111",
                    "spend_limit": 0,
                    "memo": "Card 1"
                },
                {
                    "token": "card_2",
                    "type": "SINGLE_USE",
                    "state": "CLOSED",
                    "pan": "4222222222222222",
                    "spend_limit": 5000,
                    "memo": "Card 2"
                }
            ]
        }
        
        with patch.object(connector.http, "get", return_value=mock_response):
            cards = connector.list_cards(limit=50)
        
        assert len(cards) == 2
        assert cards[0].token == "card_1"
        assert cards[0].card_type == "UNLOCKED"
        assert cards[1].token == "card_2"
        assert cards[1].card_type == "SINGLE_USE"


class TestVirtualCard:
    """Test VirtualCard dataclass."""
    
    def test_virtual_card_to_dict_masked(self):
        """VirtualCard.to_dict(reveal=False) masks sensitive fields."""
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
        
        data = card.to_dict(reveal=False)
        assert data["token"] == "card_test"
        assert data["card_type"] == "UNLOCKED"
        assert data["pan"] == "****1234"
        assert data["cvv"] == "***"
        assert data["exp_month"] == "**"
        assert data["exp_year"] == "**"
    
    def test_virtual_card_to_dict_revealed(self):
        """VirtualCard.to_dict(reveal=True) shows full details."""
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
        
        data = card.to_dict(reveal=True)
        assert data["pan"] == "4111111111111234"
        assert data["cvv"] == "123"
        assert data["exp_month"] == "12"
        assert data["exp_year"] == "2028"
