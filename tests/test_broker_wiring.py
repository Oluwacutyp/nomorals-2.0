"""Test that build_context wires the ModelBroker for capability routing."""
import unittest

from nomorals.agents.context import build_context


class BrokerWiringTest(unittest.TestCase):
    def test_broker_attached_to_router(self):
        """build_context attaches a ModelBroker to the LLMRouter."""
        ctx = build_context(with_executor=False)
        self.assertIsNotNone(ctx.router)
        broker = ctx.router.broker
        self.assertIsNotNone(broker, "ModelBroker not attached to router")

    def test_broker_has_cards_from_providers(self):
        """Broker builds cards from the router's registered providers."""
        ctx = build_context(with_executor=False)
        broker = ctx.router.broker
        self.assertIsNotNone(broker)
        cards = broker.cards()
        # At least the mock provider should be registered
        self.assertGreater(len(cards), 0, "Broker has no model cards")
        for card in cards:
            self.assertTrue(card.id, "Card missing id")
            self.assertTrue(card.provider, "Card missing provider")

    def test_broker_select_returns_card(self):
        """Broker can select a model for a capability."""
        from nomorals.llm.capabilities import Capability
        ctx = build_context(with_executor=False)
        broker = ctx.router.broker
        self.assertIsNotNone(broker)
        # Select should return a card or None (not raise)
        card = broker.select(Capability.CHAT)
        # May be None if no provider supports CHAT, but should not raise
        if card is not None:
            self.assertTrue(card.id)


if __name__ == "__main__":
    unittest.main()
