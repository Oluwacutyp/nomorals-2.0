"""Adapter conformance tests (G5): the base send() signature is the oracle."""

import unittest
from typing import Any

from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
from nomorals.social.chat.conformance import (
    all_adapter_classes,
    check_adapter_conformance,
)


class GoodAdapter(ChatAdapter):
    name = "good"

    def run(self, handler) -> None:  # pragma: no cover - never started
        raise NotImplementedError

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        return SendResult(ok=True, platform=self.name, message_id="m1")


class BadAdapter(ChatAdapter):
    """Narrows the contract: missing buttons / reply_keyboard / parse_mode."""

    name = "bad"

    def run(self, handler) -> None:  # pragma: no cover - never started
        raise NotImplementedError

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        return SendResult(ok=True, platform=self.name, message_id="m1")


class TestAdapterConformance(unittest.TestCase):
    def test_production_adapters_conform(self) -> None:
        classes = all_adapter_classes()
        # The package ships 8 production adapters; the test must see them.
        names = sorted(c.__name__ for c in classes)
        self.assertGreaterEqual(len(classes), 8, f"found: {names}")
        failures = {c.__name__: check_adapter_conformance(c) for c in classes}
        bad = {k: v for k, v in failures.items() if v}
        self.assertEqual(bad, {}, f"non-conformant adapters: {bad}")

    def test_good_double_passes(self) -> None:
        self.assertEqual(check_adapter_conformance(GoodAdapter), [])

    def test_bad_double_is_flagged(self) -> None:
        violations = check_adapter_conformance(BadAdapter)
        self.assertTrue(violations)
        joined = " ".join(violations)
        for missing in ("buttons", "reply_keyboard", "parse_mode"):
            self.assertIn(missing, joined)

    def test_non_adapter_is_flagged(self) -> None:
        violations = check_adapter_conformance(object)  # type: ignore[arg-type]
        self.assertTrue(violations)


if __name__ == "__main__":
    unittest.main()
