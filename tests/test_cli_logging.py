"""CLI logging wiring: NM_LOG_FILE must actually configure a log file.

Regression: the CLI used to call setup_logging() without the file argument,
so NM_LOG_FILE (default logs/nomorals.log) was silently ignored and no log
file was ever written — diagnostics pointed users at a file that could not
exist.
"""

from __future__ import annotations

import logging
import logging.handlers
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from nomorals.cli import _configure_log_file
from nomorals.core.config import load_settings


def _file_handlers() -> list[logging.handlers.RotatingFileHandler]:
    return [h for h in logging.getLogger().handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)]


class ConfigureLogFileTest(unittest.TestCase):
    def test_resolves_against_nm_home(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-cfg-") as tmp:
            settings = load_settings(
                env={"NM_HOME": tmp, "NM_LOG_FILE": "logs/nomorals.log"},
                use_env_file=False,
            )
            _configure_log_file(Namespace(log_level="INFO"), settings)
        handlers = _file_handlers()
        self.assertTrue(handlers, "expected a RotatingFileHandler on the root logger")
        self.assertEqual(Path(handlers[-1].baseFilename), Path(tmp) / "logs" / "nomorals.log")

    def test_writes_are_landed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-cfg-") as tmp:
            settings = load_settings(
                env={"NM_HOME": tmp, "NM_LOG_FILE": "logs/nomorals.log"},
                use_env_file=False,
            )
            _configure_log_file(Namespace(log_level="INFO"), settings)
            logging.getLogger("test.log.wiring").info("selftest-line-12345")
            for h in _file_handlers():
                h.flush()
            log = Path(tmp) / "logs" / "nomorals.log"
            self.assertTrue(log.exists(), "log file was never created")
            self.assertIn("selftest-line-12345", log.read_text())

    def test_empty_file_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-cfg-") as tmp:
            settings = load_settings(
                env={"NM_HOME": tmp, "NM_LOG_FILE": ""},
                use_env_file=False,
            )
            before = _file_handlers()
            _configure_log_file(Namespace(log_level="INFO"), settings)
            self.assertEqual(_file_handlers(), before)


if __name__ == "__main__":
    unittest.main()
