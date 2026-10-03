"""Tests for new image-gen capabilities: txt2img, upscale, size/batch params."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from nomorals.media_edit.generate import (
    GenerativeEditError,
    get_backend,
    op_txt2img,
    op_upscale,
)


def _img(w=64, h=64, color="red"):
    return Image.new("RGB", (w, h), color)


class Txt2ImgOpTests(unittest.TestCase):
    def test_txt2img_calls_backend_generate(self):
        fake_backend = MagicMock()
        fake_backend.generate.return_value = [_img()]
        with patch("nomorals.media_edit.generate.get_backend",
                   return_value=fake_backend):
            out = op_txt2img("a sunset", seed=42, width=512, height=512)
        self.assertIsInstance(out, Image.Image)
        _, kwargs = fake_backend.generate.call_args
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["height"], 512)

    def test_txt2img_batch_returns_list(self):
        fake_backend = MagicMock()
        fake_backend.generate.return_value = [_img(), _img(), _img()]
        with patch("nomorals.media_edit.generate.get_backend",
                   return_value=fake_backend):
            out = op_txt2img("a sunset", n=3)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)

    def test_txt2img_empty_prompt_raises(self):
        fake_backend = MagicMock()
        with patch("nomorals.media_edit.generate.get_backend",
                   return_value=fake_backend):
            with self.assertRaises(GenerativeEditError):
                fake_backend.generate.side_effect = GenerativeEditError("x")
                op_txt2img("")

    def test_txt2img_passes_negative_prompt(self):
        fake_backend = MagicMock()
        fake_backend.generate.return_value = [_img()]
        with patch("nomorals.media_edit.generate.get_backend",
                   return_value=fake_backend):
            op_txt2img("a cat", negative_prompt="blurry, low quality")
        _, kwargs = fake_backend.generate.call_args
        self.assertEqual(kwargs["negative_prompt"], "blurry, low quality")


class UpscaleOpTests(unittest.TestCase):
    def test_upscale_doubles_size(self):
        out = op_upscale(_img(32, 32), scale=2.0)
        self.assertEqual(out.size, (64, 64))

    def test_upscale_fractional(self):
        out = op_upscale(_img(100, 50), scale=1.5)
        self.assertEqual(out.size, (150, 75))

    def test_upscale_invalid_scale_raises(self):
        with self.assertRaises(Exception):
            op_upscale(_img(), scale=0)
        with self.assertRaises(Exception):
            op_upscale(_img(), scale=-1)

    def test_upscale_preserves_mode(self):
        out = op_upscale(_img(16, 16), scale=3.0)
        self.assertEqual(out.mode, "RGB")


class BackendGenerateTests(unittest.TestCase):
    def test_base_generate_raises(self):
        from nomorals.media_edit.generate import GenerativeBackend
        be = GenerativeBackend()
        with self.assertRaises(NotImplementedError):
            be.generate("test")

    def test_hf_generate_builds_params(self):
        from nomorals.media_edit.generate import HFInferenceBackend
        be = HFInferenceBackend(model="test-model", token="fake")
        fake_client = MagicMock()
        fake_client.text_to_image.return_value = _img()
        with patch.object(be, "_client", return_value=fake_client):
            out = be.generate("a dog", seed=7, width=256, height=256,
                              negative_prompt="ugly", steps=20, n=1)
        self.assertEqual(len(out), 1)
        _, kwargs = fake_client.text_to_image.call_args
        self.assertEqual(kwargs["seed"], 7)
        self.assertEqual(kwargs["width"], 256)
        self.assertEqual(kwargs["model"], "test-model")

    def test_hf_generate_batch_varies_seed(self):
        from nomorals.media_edit.generate import HFInferenceBackend
        be = HFInferenceBackend(model="test-model", token="fake")
        fake_client = MagicMock()
        fake_client.text_to_image.return_value = _img()
        with patch.object(be, "_client", return_value=fake_client):
            be.generate("a dog", seed=10, n=3)
        self.assertEqual(fake_client.text_to_image.call_count, 3)
        seeds = [c.kwargs["seed"]
                 for c in fake_client.text_to_image.call_args_list]
        self.assertEqual(seeds, [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
