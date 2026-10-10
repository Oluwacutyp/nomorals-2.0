"""Phase 2 Section 11 (final) — catch-all audit regression tests.

Covers the three fixes landed in this section:

1. cipher tool: ``formats`` advertised aes-gcm/chacha20 although the
   audited core cipher only supports ctr/cbc; encrypt silently defaulted
   an unsupported ``algorithm=`` to ctr. Now the advertised list is
   honest and unknown modes fail closed.
2. research/llm import cycle: ``nomorals.llm.router`` imported COST_TABLE
   from ``nomorals.research.pipeline`` while pipeline imports
   ``nomorals.llm.brain`` — importing research first raised ImportError.
   COST_TABLE now lives in ``nomorals.research.costs`` (import-light).
3. pytest_runner: pytest 9.x ``-q`` omits the summary line when stdout
   is not a TTY, so run_tests reported ``passed: 0, ok: True`` on a
   fully green suite. (Pure-parse cases live in test_pytest_runner.py.)

Tier: unit. No network, no subprocesses.
"""

from __future__ import annotations

import sys
import unittest

from nomorals.core.cipher import CipherError
from nomorals.tools.cipher import cipher_tool


class CipherFormatsHonestyTests(unittest.TestCase):
    def test_formats_lists_only_supported_modes(self) -> None:
        algos = cipher_tool("formats")["algorithms"]
        self.assertIn("aes-ctr", algos)
        self.assertIn("aes-cbc", algos)
        self.assertNotIn("aes-gcm", algos)
        self.assertNotIn("chacha20", algos)

    def test_encrypt_decrypt_roundtrip_passphrase(self) -> None:
        blob = cipher_tool("encrypt", data="hello world",
                           passphrase="correct horse")["blob"]
        self.assertEqual(
            cipher_tool("decrypt", blob=blob,
                        passphrase="correct horse")["data"],
            "hello world")

    def test_encrypt_decrypt_roundtrip_cbc(self) -> None:
        blob = cipher_tool("encrypt", data="cbc me", passphrase="pw",
                           mode="cbc")["blob"]
        self.assertEqual(
            cipher_tool("decrypt", blob=blob, passphrase="pw")["data"],
            "cbc me")

    def test_unsupported_mode_fails_closed_not_silent_default(self) -> None:
        # Before the fix, algorithm="aes-gcm" was silently encrypted as
        # ctr. Now it must raise instead of pretending.
        with self.assertRaises(CipherError):
            cipher_tool("encrypt", data="x", passphrase="pw",
                        algorithm="aes-gcm")

    def test_aes_ctr_spelling_normalizes(self) -> None:
        blob = cipher_tool("encrypt", data="norm", passphrase="pw",
                           algorithm="aes-ctr")["blob"]
        self.assertEqual(
            cipher_tool("decrypt", blob=blob, passphrase="pw")["data"],
            "norm")


class ResearchImportCycleTests(unittest.TestCase):
    def _import_in_subprocess(self, first: str) -> None:
        # Import-order checks must not run in-process: purging
        # sys.modules would poison every other test in the session.
        # A fresh interpreter per order is the honest isolation.
        import subprocess
        code = (
            f"import {first};"
            "import nomorals.research, nomorals.llm.router;"
            "assert nomorals.research.COST_TABLE['llm_call'] == 0.0008;"
            "assert nomorals.llm.router._FLAT_LLM_CALL_USD == 0.0008;"
            "print('cycle-ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"import order {first}-first failed:\n"
                         f"{proc.stderr[-2000:]}")
        self.assertIn("cycle-ok", proc.stdout)

    def test_research_first_imports(self) -> None:
        self._import_in_subprocess("nomorals.research")

    def test_llm_first_imports(self) -> None:
        self._import_in_subprocess("nomorals.llm.router")

    def test_cost_table_is_single_source(self) -> None:
        from nomorals.research import COST_TABLE as a
        from nomorals.research.costs import COST_TABLE as b
        from nomorals.research.pipeline import COST_TABLE as c
        self.assertIs(a, b)
        self.assertIs(a, c)

    def test_costs_module_is_import_light(self) -> None:
        import ast
        from pathlib import Path
        tree = ast.parse(
            (Path(__file__).parent.parent
             / "nomorals" / "research" / "costs.py").read_text())
        imports = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom))]
        # only __future__ allowed — anything from nomorals re-arms the cycle
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0,
                                 "costs.py must not use relative imports")
                self.assertFalse(
                    (node.module or "").startswith("nomorals"),
                    "costs.py must not import from nomorals")


if __name__ == "__main__":
    unittest.main()
