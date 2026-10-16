"""Chunking tests (G3): never truncate, paragraph → sentence → hard."""

import unittest

from nomorals.social.chat.chunking import (
    chunk_for_platform,
    chunk_message,
    label_chunks,
    media_kind_for,
    media_plan,
    split_caption,
)
from nomorals.social.chat.platform_limits import for_platform


class TestChunking(unittest.TestCase):
    def test_short_text_is_one_chunk(self) -> None:
        self.assertEqual(chunk_message("hello", max_text=4096), ["hello"])

    def test_empty_text(self) -> None:
        self.assertEqual(chunk_message("", max_text=100), [""])

    def test_twenty_k_never_truncates(self) -> None:
        text = ("Lorem ipsum dolor sit amet. " * 700) + "\n\n" + ("Second para. " * 300)
        chunks = chunk_message(text, max_text=4096)
        self.assertTrue(len(chunks) > 1)
        self.assertEqual("".join(chunks), text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 4096)

    def test_paragraph_boundary_preferred(self) -> None:
        text = "AAA\n\n" + "B" * 100
        chunks = chunk_message(text, max_text=10)
        self.assertEqual("".join(chunks), text)
        # The first chunk must not contain the second paragraph's B's.
        self.assertNotIn("B", chunks[0])

    def test_sentence_boundary_preferred(self) -> None:
        text = "First sentence here. Second sentence here. Third one."
        chunks = chunk_message(text, max_text=30)
        self.assertEqual("".join(chunks), text)
        # Chunks split at sentence ends, not mid-sentence.
        self.assertTrue(chunks[0].rstrip().endswith("."))
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)

    def test_single_huge_word_hard_splits_exactly(self) -> None:
        text = "x" * 1000
        # Content budget is max_text minus the " (i/n)" label room (6 chars
        # at this count): 94-char content chunks → 11 chunks, labelled ≤100.
        chunks = chunk_message(text, max_text=100)
        self.assertEqual("".join(chunks), text)
        self.assertEqual(len(chunks), 11)
        for chunk in label_chunks(chunks):
            self.assertLessEqual(len(chunk), 100)

    def test_unicode_and_emoji_survive(self) -> None:
        text = ("héllo wörld 🎉 " * 500) + "\n\nfin 🎉"
        chunks = chunk_message(text, max_text=200)
        self.assertEqual("".join(chunks), text)

    def test_labels_present_and_strippable(self) -> None:
        text = "Word. " * 500
        chunks = chunk_message(text, max_text=100)
        labelled = label_chunks(chunks)
        self.assertEqual(len(labelled), len(chunks))
        n = len(chunks)
        for i, part in enumerate(labelled):
            self.assertTrue(part.endswith(f" ({i + 1}/{n})"),
                            f"chunk {i} missing label: {part[-20:]!r}")
            self.assertEqual(part[: -len(f" ({i + 1}/{n})")], chunks[i])
            self.assertLessEqual(len(part), 100)

    def test_single_chunk_is_unlabelled(self) -> None:
        self.assertEqual(label_chunks(["hello"]), ["hello"])

    def test_sms_returns_single_chunk(self) -> None:
        # The SMS adapter owns segmentation; the gateway must not chunk.
        text = "x" * 5000
        self.assertEqual(chunk_for_platform(text, "sms"), [text])

    def test_chunk_for_platform_uses_descriptor(self) -> None:
        text = "y" * 5000
        chunks = chunk_for_platform(text, "discord")
        self.assertTrue(len(chunks) > 1)
        self.assertEqual("".join(chunks), text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 2000)

    def test_media_plan(self) -> None:
        tg = for_platform("telegram")
        self.assertEqual(media_plan(10 * 1024 * 1024, tg), "inline")
        self.assertEqual(media_plan(60 * 1024 * 1024, tg), "document")
        self.assertEqual(media_plan(60 * 1024 * 1024, for_platform("local")), "inline")
        self.assertEqual(media_kind_for(60 * 1024 * 1024, "image", tg), "document")
        self.assertEqual(media_kind_for(10, "image", tg), "image")

    def test_split_caption_never_truncates(self) -> None:
        caption = "Caption words here. " * 100
        chunks = split_caption(caption, max_caption=1024)
        self.assertEqual("".join(chunks), caption)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1024)


if __name__ == "__main__":
    unittest.main()
