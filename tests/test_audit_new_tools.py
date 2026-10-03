"""Tests for audit-added tools: finance and side_chats registration."""

import unittest

from nomorals.tools.registry import ToolRegistry


class TestFinanceToolRegistration(unittest.TestCase):
    def test_finance_tools_registered(self) -> None:
        from nomorals.tools import finance

        reg = ToolRegistry()
        finance.register(reg)
        for name in ("finance_analyze", "finance_signal", "finance_backtest"):
            self.assertIn(name, reg._tools, f"{name} not registered")


class TestSideChatsToolRegistration(unittest.TestCase):
    def test_side_chats_tool_registered(self) -> None:
        from nomorals.tools import side_chats

        reg = ToolRegistry()
        side_chats.register(reg)
        self.assertIn("side_chat", reg._tools)


if __name__ == "__main__":
    unittest.main()
