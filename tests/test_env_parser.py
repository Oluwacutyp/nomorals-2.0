"""Regression tests for the .env parser's comment handling.

The incident: ``NM_LLM_LOCAL_MODEL=            # .gguf path, name in cache,
family, or HF repo id`` (a template line uncommented by hand) parsed to the
value ``# .gguf path, name in cache, family, or HF repo id`` — the comment
became the model name, because the value was whitespace-stripped *before*
the inline-comment check, hiding the leading ``#``.  A comment after ``=``
carries no value.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.core.config import _parse_env_file


def _parse(text: str) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".env"
        path.write_text(text, encoding="utf-8")
        return _parse_env_file(path)


class TestEnvCommentHandling(unittest.TestCase):
    def test_value_that_is_only_a_comment_is_empty(self) -> None:
        self.assertEqual(_parse("KEY=   # just a comment\n"), {"KEY": ""})

    def test_inline_comment_after_real_value(self) -> None:
        self.assertEqual(_parse("KEY=real-value   # trailing note\n"), {"KEY": "real-value"})

    def test_plain_value(self) -> None:
        self.assertEqual(_parse("KEY=/data/model.gguf\n"), {"KEY": "/data/model.gguf"})

    def test_full_line_comment_is_skipped(self) -> None:
        self.assertEqual(_parse("# KEY=x\nOTHER=y\n"), {"OTHER": "y"})

    def test_hash_without_preceding_space_stays_in_value(self) -> None:
        # no space before the hash: it is part of the value (not a comment)
        self.assertEqual(_parse("KEY=real#note\n"), {"KEY": "real#note"})

    def test_quoted_value_keeps_inner_hash(self) -> None:
        self.assertEqual(_parse('KEY="a # b"\n'), {"KEY": "a # b"})

    def test_export_prefix(self) -> None:
        self.assertEqual(_parse("export KEY=v\n"), {"KEY": "v"})

    def test_duplicate_key_last_wins(self) -> None:
        # the documented behavior the CLI's upsert relies on
        self.assertEqual(_parse("KEY=first\nKEY=second\n"), {"KEY": "second"})


if __name__ == "__main__":
    unittest.main()
