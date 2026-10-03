"""Tests for /image smart routing: lookup vs generation."""
import unittest

from nomorals.agents.partner.runtime_media import _looks_like_path_or_url
from nomorals.social.chat.control import parse_control


class LooksLikePathOrUrlTests(unittest.TestCase):
    def test_urls(self):
        self.assertTrue(_looks_like_path_or_url("https://example.com/pic.jpg"))
        self.assertTrue(_looks_like_path_or_url("http://example.com/a.png"))
        self.assertTrue(_looks_like_path_or_url("file:///tmp/x.jpg"))

    def test_paths(self):
        self.assertTrue(_looks_like_path_or_url("/sdcard/pic.jpg"))
        self.assertTrue(_looks_like_path_or_url("~/photos/cat.png"))
        self.assertTrue(_looks_like_path_or_url("./img/photo.jpg"))
        self.assertTrue(_looks_like_path_or_url("C:\\pics\\dog.jpg"))

    def test_filenames_with_ext(self):
        self.assertTrue(_looks_like_path_or_url("photo.jpg"))
        self.assertTrue(_looks_like_path_or_url("screenshot.PNG"))

    def test_prompts_are_not_paths(self):
        self.assertFalse(_looks_like_path_or_url("a cyberpunk city"))
        self.assertFalse(_looks_like_path_or_url("a cyberpunk city at night"))
        self.assertFalse(_looks_like_path_or_url("lucid dream"))
        self.assertFalse(_looks_like_path_or_url("portrait of a cat wearing a hat"))

    def test_empty(self):
        self.assertFalse(_looks_like_path_or_url(""))
        self.assertFalse(_looks_like_path_or_url("   "))


class ImageArgLimitTests(unittest.TestCase):
    def test_many_words_allowed(self):
        # "/image a cyber punk city" is 4 words — must not hit "at most 3"
        cmd = parse_control("/image a cyber punk city")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "image")
        self.assertNotEqual(cmd.kind, "error")

    def test_single_word_prompt(self):
        cmd = parse_control("/image sunset")
        self.assertEqual(cmd.kind, "image")

    def test_path_still_works(self):
        cmd = parse_control("/image /sdcard/pic.jpg")
        self.assertEqual(cmd.kind, "image")
        self.assertEqual(cmd.arg, "/sdcard/pic.jpg")


if __name__ == "__main__":
    unittest.main()
