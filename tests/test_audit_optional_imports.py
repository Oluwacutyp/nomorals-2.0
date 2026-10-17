"""Audit: optional numpy / Pillow / curses must degrade honestly, never crash
at import.

With numpy + Pillow + curses blocked (clean subprocess, meta-path blocker
following the ``test_media_edit_imports.py`` convention):

* every media module in the cascade still imports;
* every feature that needs a blocked dep raises an honest ``* unavailable``
  error naming the missing dep — never a bare ImportError/AttributeError;
* pure-Python entry points keep working (``list_actions()``,
  ``HighlightMLP()`` construction, ``backend_status()`` reporting).

Plus PIL-only and numpy-only variants: the surviving dep's features still
work, the blocked dep's features raise honest errors.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_BLOCK_TMPL = (
    "import importlib.abc, sys\n"
    "class _Block(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name in {names!r} or name.startswith({prefixes!r}):\n"
    "            raise ImportError(f\"No module named '{{name}}' (blocked for test)\".format(name=name))\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Block())\n"
)


def _blocker(names: tuple[str, ...]) -> str:
    prefixes = tuple(f"{n}." for n in names)
    return _BLOCK_TMPL.format(names=names, prefixes=prefixes)


_BLOCK_ALL = _blocker(("numpy", "PIL", "curses"))
_BLOCK_PIL = _blocker(("PIL",))
_BLOCK_NUMPY = _blocker(("numpy",))

_IMPORTS = """
import nomorals.media.videogen
import nomorals.media.motion_studio
import nomorals.media.contentops
import nomorals.media_edit.cv_ops
import nomorals.media.directed.ai_edit
import nomorals.media.directed.animator
import nomorals.media.directed.camera
import nomorals.media.directed.filler
import nomorals.media.directed.lipsync
import nomorals.media.directed.motion_score
import nomorals.media.directed.pose_rig
import nomorals.media.directed.validate
import nomorals.media.imggen.diffusion
import nomorals.media.imggen.masks
import nomorals.media.imggen.edit
import nomorals.media.highlight_model.model
import nomorals.media.cv_ops
import nomorals.tui
import nomorals.tui.app
import nomorals.cmdline
import nomorals.tools.directed
import nomorals.tools.media_edit
print("all imports ok")
"""

# Every feature probe: (callable-expression, expected substring).
# The snippet defines _expect() + _FAKE image stub first.
_PROBES_ALL = """
def _expect(label, fn, *needles):
    try:
        fn()
    except Exception as exc:
        msg = str(exc)
        assert "unavailable" in msg, (label, type(exc).__name__, msg)
        for n in needles:
            assert n in msg, (label, msg)
        print("honest:", label, "->", msg[:100])
        return
    raise AssertionError(f"{label}: no error raised")

class _FakeImg:
    size = (10, 10)
    def convert(self, mode): return self
    def resize(self, size, *a): return self
    def copy(self): return self
    def filter(self, f): return self

from nomorals.media.directed.pose_rig import list_actions, rest_pose
assert list_actions(), "list_actions must still work without deps"
_expect("pose_rig.rest_pose", rest_pose, "numpy")
from nomorals.media.directed.camera import Perlin1D
_expect("camera.Perlin1D", Perlin1D, "numpy")
from nomorals.media.highlight_model.model import HighlightMLP
_mlp = HighlightMLP()  # construction must still work
assert not _mlp.loaded
_expect("HighlightMLP.score", lambda: _mlp.score([[0.0] * 8]), "numpy")
from nomorals.media_edit.cv_ops import backend_status, denoise
assert backend_status()["pillow"] is False
_expect("media_edit.denoise", lambda: denoise(None), "Pillow")
from nomorals.media.contentops.beats import _beats_numpy, beats_available
assert beats_available()["numpy_fallback"] is False
_expect("beats._beats_numpy", lambda: _beats_numpy([], 22050), "numpy")
from nomorals.media.videogen.consistency import boundary_metric, reinhard_match
_expect("consistency.boundary_metric", lambda: boundary_metric(None, None), "numpy")
_expect("consistency.reinhard_match", lambda: reinhard_match(None, None), "numpy")
from nomorals.media.imggen.diffusion import linear_beta_schedule, BetaSchedule
_expect("diffusion.linear_beta_schedule", lambda: linear_beta_schedule(10), "numpy")
_expect("diffusion.BetaSchedule", lambda: BetaSchedule(timesteps=3, betas=[0.1, 0.2, 0.3]), "numpy")
from nomorals.media.imggen.masks import box_mask, point_flood_mask
_expect("masks.box_mask", lambda: box_mask((10, 10), (0, 0, 5, 5)), "Pillow")
_expect("masks.point_flood_mask", lambda: point_flood_mask(None, (5, 5)))
from nomorals.media.imggen.edit import feather_mask, inpaint_cpu
_expect("imggen.edit.feather_mask", lambda: feather_mask(None), "Pillow")
_expect("imggen.edit.inpaint_cpu", lambda: inpaint_cpu(None, None))
from nomorals.media.cv_ops import inpaint_diffuse, feather
_expect("media.cv_ops.inpaint_diffuse", lambda: inpaint_diffuse(_FakeImg(), _FakeImg()))
_expect("media.cv_ops.feather", lambda: feather(None), "Pillow")
from nomorals.tui.app import TuiApp, run as tui_run
TuiApp()  # construction must still work
_expect("tui.run", tui_run, "curses")
from nomorals.cmdline.commands.tui import _cmd_tui
from types import SimpleNamespace
assert _cmd_tui(None, SimpleNamespace(settings=SimpleNamespace(profile="x"))) == 1
from nomorals.media.motion_studio.kenburns import add_grain
_expect("kenburns.add_grain", lambda: add_grain(None), "numpy")
from nomorals.media.motion_studio._core import decode_audio_mono
import wave as _wave, tempfile, os
_fd, _wav = tempfile.mkstemp(suffix=".wav"); os.close(_fd)
with _wave.open(_wav, "wb") as _w:
    _w.setnchannels(1); _w.setsampwidth(2); _w.setframerate(22050)
    _w.writeframes(b"\\x00\\x00" * 2205)
_expect("motion_studio.decode_audio_mono", lambda: decode_audio_mono(_wav), "numpy")
from nomorals.media.motion_studio.typography import _LyricRenderer, STYLE_PRESETS
_expect("typography._LyricRenderer",
         lambda: _LyricRenderer([], STYLE_PRESETS["neon_pop"], 100, 100, 24.0, None, 0),
         "Pillow")
from nomorals.media.motion_studio.visualizer import analyze_audio
_expect("visualizer.analyze_audio", lambda: analyze_audio(_wav, 24.0), "numpy")
from nomorals.media.directed.filler import _phase_shift
_expect("filler._phase_shift", lambda: _phase_shift(None, None), "numpy")
from nomorals.media.directed.motion_score import compile_score
_expect("motion_score.compile_score", lambda: compile_score({"phases": []}), "numpy")
from nomorals.media.directed.validate import validate_score
_expect("validate.validate_score",
         lambda: validate_score({"phases": [{"name": "x", "t": [0, 1],
                                             "moves": [{"joint": "nose", "to": [0.5, 0.5]}]}]}),
         "numpy")
from nomorals.media.directed.lipsync import _warp_mouth
class _Arr:
    shape = (10, 10, 3)
    def __getitem__(self, k): return self
    def copy(self): return self
_expect("lipsync._warp_mouth", lambda: _warp_mouth(_Arr(), (0, 0, 5, 5), (0.5, 1.0, 0.0)), "Pillow")
from nomorals.media.directed.animator import _bilinear_sample
_expect("animator._bilinear_sample", lambda: _bilinear_sample(_Arr(), None, None), "numpy")
from nomorals.media.directed.ai_edit import relight_photo
_expect("ai_edit.relight_photo", lambda: relight_photo(_FakeImg()), "numpy")
from nomorals.media.directed.ai_edit import background_replace
_expect("ai_edit.background_replace",
         lambda: background_replace(_FakeImg(), _FakeImg(), _FakeImg()), "Pillow")
print("all probes honest")
"""

_PROBES_PIL_ONLY = """
import numpy as np
from nomorals.media.imggen.diffusion import linear_beta_schedule
assert linear_beta_schedule(3).shape == (3,)
from nomorals.media.contentops.beats import _beats_numpy, beats_available
assert beats_available()["numpy_fallback"] is True
assert _beats_numpy(np.zeros(22050), 22050) == (0.0, [])
from nomorals.media.directed.pose_rig import rest_pose
assert rest_pose().shape[0] > 0
from nomorals.media.directed.motion_score import compile_score
compile_score({"phases": []})
from nomorals.media.highlight_model.model import HighlightMLP
_m = HighlightMLP(); _m._init_random(0)
assert len(_m.score([[0.0] * 8])) == 1
from nomorals.media_edit.cv_ops import backend_status, denoise
assert backend_status()["pillow"] is False
try:
    denoise(None)
except Exception as exc:
    assert "unavailable" in str(exc) and "Pillow" in str(exc), str(exc)
else:
    raise AssertionError("denoise without Pillow did not raise")
from nomorals.media.imggen.masks import box_mask
try:
    box_mask((10, 10), (0, 0, 5, 5))
except Exception as exc:
    assert "unavailable" in str(exc), str(exc)
else:
    raise AssertionError("box_mask without Pillow did not raise")
print("pil-only probes ok")
"""

_PROBES_NUMPY_ONLY = """
from PIL import Image
from nomorals.media.imggen.masks import box_mask
m = box_mask((10, 10), (0, 0, 5, 5))
assert m.size == (10, 10)
from nomorals.media.imggen.edit import feather_mask
assert feather_mask(Image.new("L", (8, 8), 0)).size == (8, 8)
from nomorals.media.directed.filler import _morph_blend_frames
a = Image.new("RGB", (8, 8), (255, 0, 0))
b = Image.new("RGB", (8, 8), (0, 0, 255))
assert len(_morph_blend_frames(a, b, 2)) == 2
from nomorals.media.cv_ops import feather
assert feather(Image.new("L", (8, 8), 0)).size == (8, 8)
from nomorals.media_edit.cv_ops import backend_status, denoise
assert backend_status()["pillow"] is True
assert denoise(Image.new("RGB", (8, 8), (10, 20, 30))).size == (8, 8)
from nomorals.media.videogen.consistency import boundary_metric
try:
    boundary_metric(a, b)
except Exception as exc:
    assert "unavailable" in str(exc) and "numpy" in str(exc), str(exc)
else:
    raise AssertionError("boundary_metric without numpy did not raise")
from nomorals.media.imggen.diffusion import linear_beta_schedule
try:
    linear_beta_schedule(10)
except Exception as exc:
    assert "unavailable" in str(exc), str(exc)
else:
    raise AssertionError("linear_beta_schedule without numpy did not raise")
print("numpy-only probes ok")
"""


class AuditOptionalImportTests(unittest.TestCase):
    def _run(self, blocker: str, snippet: str) -> subprocess.CompletedProcess[str]:
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-c", blocker + snippet],
            capture_output=True, text=True, timeout=180, env=env,
        )

    def test_all_blocked_imports_and_honest_errors(self):
        proc = self._run(_BLOCK_ALL, _IMPORTS + _PROBES_ALL)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        self.assertIn("all imports ok", proc.stdout)
        self.assertIn("all probes honest", proc.stdout)

    def test_pil_only_blocked(self):
        proc = self._run(_BLOCK_PIL, _IMPORTS + _PROBES_PIL_ONLY)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        self.assertIn("pil-only probes ok", proc.stdout)

    def test_numpy_only_blocked(self):
        proc = self._run(_BLOCK_NUMPY, _IMPORTS + _PROBES_NUMPY_ONLY)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        self.assertIn("numpy-only probes ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
