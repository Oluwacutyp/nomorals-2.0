"""Tests for new editor ops: blur, border, video speed/fade/overlay."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from nomorals.media_edit.images import op_blur, op_border


def _img(w=64, h=64, color="blue"):
    return Image.new("RGB", (w, h), color)


class BlurOpTests(unittest.TestCase):
    def test_blur_changes_pixels(self):
        # Checkerboard: blur must actually blend.
        img = Image.new("RGB", (16, 16))
        px = img.load()
        for y in range(16):
            for x in range(16):
                px[x, y] = (255, 255, 255) if (x + y) % 2 else (0, 0, 0)
        out = op_blur(img, radius=2.0)
        self.assertEqual(out.size, img.size)
        # Blurred checkerboard has gray pixels; original has none.
        self.assertNotEqual(list(out.getdata()), list(img.getdata()))

    def test_blur_zero_is_copy(self):
        img = _img()
        out = op_blur(img, radius=0)
        self.assertEqual(list(out.getdata()), list(img.getdata()))
        self.assertIsNot(out, img)

    def test_blur_negative_raises(self):
        with self.assertRaises(Exception):
            op_blur(_img(), radius=-1)


class BorderOpTests(unittest.TestCase):
    def test_border_expands(self):
        out = op_border(_img(32, 32), width=10, color="black")
        self.assertEqual(out.size, (52, 52))

    def test_border_color(self):
        out = op_border(_img(16, 16, "red"), width=4, color="white")
        # Corner pixel should be the border color.
        self.assertEqual(out.getpixel((0, 0)), (255, 255, 255))
        # Center should still be red.
        self.assertEqual(out.getpixel((12, 12)), (255, 0, 0))

    def test_border_zero_is_copy(self):
        img = _img()
        out = op_border(img, width=0)
        self.assertEqual(out.size, img.size)

    def test_border_negative_raises(self):
        with self.assertRaises(Exception):
            op_border(_img(), width=-5)


class VideoOpTests(unittest.TestCase):
    def _mock_run(self):
        return patch("nomorals.media_edit.videos.run_ffmpeg")

    def _mock_probe(self):
        m = MagicMock(return_value={"duration": 10.0})
        return patch("nomorals.media_edit.videos.video_probe", m)

    def test_speed_builds_filters(self):
        from nomorals.media_edit import videos as V
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4",
                                         delete=False) as f:
            src = f.name
        try:
            with self._mock_run() as run, self._mock_probe():
                # Fake the output file so stat() works.
                with patch.object(V, "_out") as out_fn:
                    out_path = src + ".out.mp4"
                    Path = __import__("pathlib").Path
                    open(out_path, "wb").write(b"x" * 100)
                    out_fn.return_value = Path(out_path)
                    with patch.object(Path, "stat") as st:
                        st.return_value = MagicMock(st_size=100)
                        res = V.speed(src, 2.0)
                    self.assertEqual(res["factor"], 2.0)
                    args = run.call_args[0][0]
                    vf = args[args.index("-vf") + 1]
                    self.assertIn("setpts=PTS/2.0", vf)
        finally:
            os.unlink(src)
            try:
                os.unlink(src + ".out.mp4")
            except OSError:
                pass

    def test_speed_invalid_factor_raises(self):
        from nomorals.media_edit import videos as V
        with self.assertRaises(Exception):
            V.speed("/nonexistent.mp4", 0)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with self.assertRaises(Exception):
                V.speed(f.name, -1.0)

    def test_fade_builds_filters(self):
        from nomorals.media_edit import videos as V
        import tempfile, os
        from pathlib import Path as P
        with tempfile.NamedTemporaryFile(suffix=".mp4",
                                         delete=False) as f:
            src = f.name
        try:
            with self._mock_run() as run, self._mock_probe():
                out_path = src + ".out.mp4"
                open(out_path, "wb").write(b"x" * 100)
                with patch.object(V, "_out", return_value=P(out_path)):
                    with patch.object(P, "stat") as st:
                        st.return_value = MagicMock(st_size=100)
                        res = V.fade(src, fade_in=1.0, fade_out=2.0)
                    self.assertEqual(res["fade_in"], 1.0)
                    args = run.call_args[0][0]
                    vf = args[args.index("-vf") + 1]
                    self.assertIn("fade=t=in", vf)
                    self.assertIn("fade=t=out", vf)
        finally:
            os.unlink(src)
            try:
                os.unlink(src + ".out.mp4")
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
