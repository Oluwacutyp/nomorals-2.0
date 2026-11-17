"""Gapfix R3-54: the dead `nomorals/media/cv_ops.py` copy is gone.

The module was a duplicate of image-repair capability that already
exists live and strictly better in two places:

* `nomorals/media/imggen/edit.py` — `_inpaint_numpy` (Voronoi-seeded
  diffusion fill + feather blend) and `feather_mask` (same GaussianBlur
  one-liner, same default radius 8);
* `nomorals/media_edit/cv_ops.py` — `inpaint_cv` (OpenCV Telea/NS >
  scikit-image biharmonic > Pillow onion-peel diffusion fallback),
  with a strictly more flexible mask input (PIL / path / bytes / box).

Wiring the copy in would have added a third copy of the same algorithm
and, in `imggen/edit.py`, downgraded the live numpy path (mean-seeded,
unblended fill). So the copy was deleted and the audit test's three
references to it removed; these tests pin the deletion and guard the
two live paths.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent

# Needles built from parts so this file itself contains no literal
# match for the scan in test_no_reference_to_dead_module_in_python_tree.
_NEEDLE_DOTTED = ".".join(["media", "cv_ops"])
_NEEDLE_SLASH = "/".join(["media", "cv_ops"])


class DeadCopyGoneTests(unittest.TestCase):
    def test_dead_module_is_gone(self):
        self.assertIsNone(
            importlib.util.find_spec("nomorals.media.cv_ops"),
            "nomorals.media.cv_ops must not exist after R3-54",
        )

    def test_no_reference_to_dead_module_in_python_tree(self):
        # This file itself is skipped: it must name the module to assert
        # its absence (find_spec + the audit-test check below).
        _self = Path(__file__).resolve()
        hits = []
        for root in (_ROOT / "nomorals", _ROOT / "tests"):
            for path in sorted(root.rglob("*.py")):
                if path.resolve() == _self:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if _NEEDLE_DOTTED in text or _NEEDLE_SLASH in text:
                    hits.append(str(path.relative_to(_ROOT)))
        self.assertEqual(hits, [], f"references to the dead module remain: {hits}")

    def test_audit_optional_imports_no_longer_references_dead_module(self):
        text = (_ROOT / "tests" / "test_audit_optional_imports.py").read_text(
            encoding="utf-8")
        self.assertNotIn(_NEEDLE_DOTTED, text)
        self.assertNotIn(_NEEDLE_SLASH, text)


class WiredPathsStillWorkTests(unittest.TestCase):
    def _rgb(self, size=(32, 32), color=(200, 40, 90)):
        return Image.new("RGB", size, color)

    def test_wired_inpaint_cv_still_works(self):
        from nomorals.media_edit.cv_ops import inpaint_cv

        out = inpaint_cv(self._rgb(), "8,8,24,24")
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (32, 32))

    def test_live_cpu_inpaint_still_works(self):
        from nomorals.media.imggen.edit import inpaint_cpu

        mask = Image.new("L", (32, 32), 0)
        from PIL import ImageDraw

        ImageDraw.Draw(mask).rectangle([8, 8, 24, 24], fill=255)
        out, label = inpaint_cpu(self._rgb(), mask, method="numpy")
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (32, 32))
        self.assertIn("numpy", label)

    def test_feather_mask_still_available(self):
        from nomorals.media.imggen.edit import feather_mask

        mask = Image.new("L", (16, 16), 0)
        out = feather_mask(mask, radius=8)
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (16, 16))


if __name__ == "__main__":
    unittest.main()
