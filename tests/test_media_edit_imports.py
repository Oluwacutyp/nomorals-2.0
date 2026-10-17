"""R8: ``nomorals.media_edit`` must import without optional deps installed.

Regression test for the ``cv_ops`` hard PIL import: the package ``__init__``
imported ``cv_ops`` unguarded while ``cv_ops`` does ``from PIL import ...``
at module top level — so ``import nomorals.media_edit`` crashed on a bare
box without Pillow, violating the zero-mandatory-deps promise
(``pyproject.toml`` declares ``dependencies = []``).  The import now
degrades like ``cv_video``: the package loads, ``cv_ops`` loads,
``backend_status()["pillow"]`` is False, and touching an op raises an
honest ``MediaEditError`` naming Pillow.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

_BLOCK_PIL = (
    "import importlib.abc, sys\n"
    "class _Block(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'PIL' or name.startswith('PIL.'):\n"
    "            raise ImportError(\"No module named 'PIL' (blocked for test)\")\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Block())\n"
)


class MediaEditImportTests(unittest.TestCase):
    def _run(self, snippet: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", _BLOCK_PIL + snippet],
            capture_output=True, text=True, timeout=180,
        )

    def test_package_imports_without_pillow(self):
        proc = self._run("import nomorals.media_edit; print('import ok')")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("import ok", proc.stdout)

    def test_cv_ops_imports_without_pillow_and_reports_it(self):
        # New contract: cv_ops imports fine without Pillow, reports
        # pillow: False, and any op raises an honest MediaEditError.
        proc = self._run(
            "from nomorals.media_edit import cv_ops as C\n"
            "print('pillow:', C.backend_status()['pillow'])\n"
            "try:\n"
            "    C.denoise(None)\n"
            "    print('NO ERROR - BUG')\n"
            "except Exception as exc:\n"
            "    print('error type:', type(exc).__name__)\n"
            "    print('error:', exc)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("pillow: False", proc.stdout)
        self.assertIn("MediaEditError", proc.stdout)
        self.assertIn("unavailable", proc.stdout)
        self.assertIn("Pillow", proc.stdout)

    def test_cv_video_names_still_degrade(self):
        # The pre-existing cv_video guard keeps working alongside the fix.
        proc = self._run(
            "import nomorals.media_edit as me\n"
            "print('cv2_available' in dir(me))\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
