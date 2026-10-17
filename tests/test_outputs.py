"""Infra spec P2 (§5 + §6): commands + outputs.

- Truncation is always noted WITH a count (§6.1).
- Honest partials: thin zone book says so (§6.2).
- Reverse orphan audit: every _control_* handler is reachable (§5.1).
- Confirmations are exact-bound: one act, those arguments, single-use (§5.3).
- Tool results never render as raw dicts (§6.1).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parent.parent


class TestTruncationCounts(unittest.TestCase):
    def test_prev_placeholder_truncation_names_count(self):
        from nomorals.social.chat.control import (
            PREV_PLACEHOLDER_MAX_CHARS,
            expand_prev_placeholders,
        )
        long_out = "x" * (PREV_PLACEHOLDER_MAX_CHARS + 250)
        expanded = expand_prev_placeholders("use {prev} here", [long_out])
        self.assertIn("truncated, 250 more chars", expanded)
        self.assertNotIn("x" * (PREV_PLACEHOLDER_MAX_CHARS + 1), expanded)


class TestHonestPartials(unittest.TestCase):
    def _expert(self, n_bars: int):
        from nomorals.agents.financial_expert import FinancialExpert

        expert = FinancialExpert(SimpleNamespace(db=None, settings=None))
        fake_df = list(range(n_bars))  # len() is all analyze() needs

        class FakePipeline:
            @staticmethod
            def analyze(_df):
                return {
                    "bars": n_bars, "regime_label": "trend",
                    "bias": 0.5, "agreement": 0.8, "position_now": 0.0,
                    "approved": False, "size_fraction": 0.0,
                    "stop_distance_pct": 100.0, "ranked": [],
                    "n_strategies": 0,
                }

        expert._load_bars = lambda *a, **k: fake_df  # noqa: ARG005
        expert._ta = lambda: (FakePipeline(), None, None, None)
        expert._room_log = lambda *a, **k: None  # noqa: ARG005
        return expert

    def test_thin_zone_book_labelled(self):
        out = self._expert(729).analyze("XAUUSD", "forex", "1h", 2000)
        self.assertIn("729/2000", out.bars_note)
        self.assertIn("1h", out.bars_note)
        self.assertIn("729/2000", out.to_dict()["bars_note"])

    def test_full_zone_book_no_note(self):
        out = self._expert(2000).analyze("XAUUSD", "forex", "1h", 2000)
        self.assertEqual(out.bars_note, "")


class TestReverseOrphanAudit(unittest.TestCase):
    def test_every_control_handler_is_referenced(self):
        """§5.1 reverse orphan audit: every _control_* handler must be
        reachable (referenced as self._control_x somewhere in nomorals/).
        A handler nobody calls is an orphan, even if the registry is fine.
        """
        runtime_files = sorted((REPO / "nomorals" / "agents" / "partner").glob("runtime*.py"))
        defs: dict[str, str] = {}
        for path in runtime_files:
            for m in re.finditer(r"def (_control_\w+)\(", path.read_text()):
                defs[m.group(1)] = str(path)
        blob = ""
        for path in (REPO / "nomorals").rglob("*.py"):
            blob += path.read_text(errors="replace") + "\n"
        orphans = [name for name in defs
                   if not re.search(r"\." + re.escape(name) + r"\b", blob)]
        self.assertEqual(orphans, [], f"unreachable _control_* handlers: {orphans}")


class TestConfirmGateExactBound(unittest.TestCase):
    def _runtime(self):
        from nomorals.agents.partner.runtime import PartnerRuntime
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt._pending_confirms = {}
        return rt

    def test_confirm_runs_stored_tail_only(self):
        rt = self._runtime()
        reply, proceed, _ = rt._confirm_gate("forget", "abc", "chat1")
        self.assertFalse(proceed)
        self.assertIn("confirm", reply.lower())
        reply, proceed, effective = rt._confirm_gate("forget", "confirm", "chat1")
        self.assertTrue(proceed)
        self.assertEqual(effective, "abc")
        # Single-use: a second confirm finds nothing.
        reply, proceed, _ = rt._confirm_gate("forget", "confirm", "chat1")
        self.assertFalse(proceed)
        self.assertIn("nothing to confirm", reply)

    def test_confirm_wrong_kind_does_not_authorize(self):
        rt = self._runtime()
        rt._confirm_gate("forget", "abc", "chat1")
        reply, proceed, _ = rt._confirm_gate("quit", "confirm", "chat1")
        self.assertFalse(proceed)
        # The original pending confirm is untouched (kind-mismatch pops nothing).
        reply, proceed, effective = rt._confirm_gate("forget", "confirm", "chat1")
        self.assertTrue(proceed)
        self.assertEqual(effective, "abc")

    def test_cancel_clears_pending(self):
        rt = self._runtime()
        rt._confirm_gate("forget", "abc", "chat1")
        reply, proceed, _ = rt._confirm_gate("forget", "cancel", "chat1")
        self.assertFalse(proceed)
        self.assertIn("cancelled", reply)
        reply, proceed, _ = rt._confirm_gate("forget", "confirm", "chat1")
        self.assertFalse(proceed)

    def test_expired_confirm_refused(self):
        import time
        from nomorals.social.chat.control import CONFIRM_TTL_S
        rt = self._runtime()
        rt._confirm_gate("forget", "abc", "chat1")
        kind, tail, _exp = rt._pending_confirms["chat1"]
        rt._pending_confirms["chat1"] = (kind, tail, time.time() - CONFIRM_TTL_S - 1)
        reply, proceed, _ = rt._confirm_gate("forget", "confirm", "chat1")
        self.assertFalse(proceed)
        self.assertIn("expired", reply)


class TestNoRawDicts(unittest.TestCase):
    def test_render_tool_value_never_raw_dict(self):
        from nomorals.core.tool_render import render_tool_value
        payload = {"a": 1, "nested": {"b": [1, 2, {"c": "x"}]}, "bars_note": ""}
        out = render_tool_value(payload)
        self.assertNotIn("{'", out)
        self.assertNotIn('{"', out)
        self.assertIn("a: 1", out)
        self.assertIn("c: x", out)

    def test_render_scalar_never_raw(self):
        from nomorals.core.tool_render import render_tool_value
        self.assertEqual(render_tool_value(None), "(no output)")
        self.assertEqual(render_tool_value({}), "(empty)")


if __name__ == "__main__":
    unittest.main()
