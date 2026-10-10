"""Story continuation: bible-from-text, in-voice chapter composing,
bible digestion of written chapters.  No model in the test context —
the heuristic composer is exercised.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.bible import BibleBuilder
from nomorals.books.continuation import StoryContinuer


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


PASTED = """Chapter 1 — The Bite

Quinn woke with the taste of copper in his mouth. "You died last night,"
Mara said, not looking up from her book. "I must find out what I am now,"
Quinn said. The old rule was simple: the thirst comes first, and it does
not negotiate.

Chapter 2 — The Thirst

"I will not drink from the living," Quinn vowed. Mara laughed, a dry
sound. "We need to leave the city before the Vigil finds you," she said.
The Vigil cannot be reasoned with; it can only be outrun.
"""


class TestContinuation(unittest.TestCase):
    def test_prepare_from_text_builds_bible(self) -> None:
        ctx, _ = _ctx()
        cont = StoryContinuer(ctx)
        bible = cont.prepare_from_text("The Night Ledger", PASTED)
        names = {c.name for c in bible.characters}
        self.assertIn("Quinn", names)
        self.assertIn("Mara", names)
        self.assertTrue(bible.open_threads())
        self.assertTrue(bible.world_rules)

    def test_compose_continuation_advances_thread(self) -> None:
        ctx, _ = _ctx()
        cont = StoryContinuer(ctx)
        bible = cont.prepare_from_text("The Night Ledger", PASTED)
        text = cont._compose_continuation(bible, 3, [], words=600,
                                          direction="")
        words = len(text.split())
        self.assertGreater(words, 150)
        # the hottest open thread is being advanced, not dropped
        hot = max(bible.open_threads(), key=lambda t: t.heat)
        key = set(w for w in hot.summary.lower().split() if len(w) > 3)
        text_words = set(text.lower().split())
        self.assertTrue(key & text_words)
        # no placeholder leakage
        self.assertNotIn("TODO", text)
        self.assertNotIn("lorem", text.lower())

    def test_compose_is_deterministic(self) -> None:
        ctx, _ = _ctx()
        cont = StoryContinuer(ctx)
        bible = cont.prepare_from_text("The Night Ledger", PASTED)
        a = cont._compose_continuation(bible, 5, [], words=400, direction="")
        b = cont._compose_continuation(bible, 5, [], words=400, direction="")
        self.assertEqual(a, b)

    def test_direction_is_woven_in(self) -> None:
        ctx, _ = _ctx()
        cont = StoryContinuer(ctx)
        bible = cont.prepare_from_text("The Night Ledger", PASTED)
        text = cont._compose_continuation(bible, 3, [], words=400,
                                          direction="Quinn meets the Vigil's captain")
        self.assertIn("Vigil", text)


if __name__ == "__main__":
    unittest.main()
