"""Tests for nomorals.shows.composer."""

import unittest
import xml.etree.ElementTree as ET

from nomorals.shows.composer import Show, compose_episode, publish_episode


def _show():
    return Show(
        id="show-1", title="Morning Pulse", schedule="daily 23:00",
        voices=["Devon", "Pulse"], bed_track="bed.mp3",
        rss_url="https://example.invalid/feed.xml",
    )


class TestShowsComposer(unittest.TestCase):
    def test_compose_joins_segments(self):
        episode = compose_episode(_show(), [
            {"kind": "script", "text": "Good morning. Here is the news."},
            {"kind": "voice", "text": "And now the markets."},
            {"kind": "bed", "ref": "bed.mp3"},
            {"kind": "ad", "text": "Sponsored by nobody."},
        ], title="Ep 1")
        self.assertIn("Good morning. Here is the news.", episode.script_text)
        self.assertIn("And now the markets.", episode.script_text)
        self.assertIn("Sponsored by nobody.", episode.script_text)
        # Bed segments contribute no spoken words.
        self.assertNotIn("bed.mp3", episode.script_text)
        self.assertEqual(episode.show_id, "show-1")
        self.assertEqual(len(episode.segments), 4)

    def test_duration_from_word_count(self):
        words = " ".join(["word"] * 150)  # 150 wpm → 60s
        episode = compose_episode(_show(), [{"kind": "script", "text": words}])
        self.assertEqual(episode.word_count, 150)
        self.assertAlmostEqual(episode.duration_seconds, 60.0)
        short = compose_episode(_show(), [{"kind": "script", "text": "hello world"}])
        self.assertEqual(short.word_count, 2)
        self.assertAlmostEqual(short.duration_seconds, 0.8)

    def test_rss_valid_xml(self):
        episode = compose_episode(
            _show(), [{"kind": "script", "text": "Hello listeners."}],
            title="Ep 1", created_ts=1_800_000_000.0)
        root = ET.fromstring(episode.rss_item)
        self.assertEqual(root.tag, "item")
        self.assertEqual(root.findtext("title"), "Ep 1")
        self.assertEqual(root.findtext("guid"), episode.id)
        self.assertIn("Hello listeners.", root.findtext("description"))
        self.assertIsNotNone(root.findtext("pubDate"))

    def test_publish_draft_without_approval(self):
        episode = compose_episode(
            _show(), [{"kind": "script", "text": "Hello listeners."}])
        result = publish_episode(episode, owner_approved=False)
        self.assertFalse(result.published)
        self.assertEqual(result.page.status, "draft")
        self.assertIsNone(result.link)

    def test_publish_public_with_approval_and_clean_content(self):
        episode = compose_episode(
            _show(), [{"kind": "script", "text": "Hello listeners, here is the news."}])
        result = publish_episode(episode, owner_approved=True)
        self.assertTrue(result.published)
        self.assertEqual(result.page.status, "public")
        self.assertEqual(result.page.pack, "show")
        assert result.link is not None
        self.assertIn("show-", result.link.url_path)

    def test_publish_blocked_by_private_data_even_with_approval(self):
        episode = compose_episode(
            _show(), [{"kind": "script", "text": "The password is hunter2, do not share."}])
        result = publish_episode(episode, owner_approved=True)
        self.assertFalse(result.published)
        self.assertEqual(result.page.status, "draft")
        self.assertTrue(result.violations)

    def test_composer_does_not_generate_scripts(self):
        # Empty segments → empty script: the composer never invents one.
        episode = compose_episode(_show(), [])
        self.assertEqual(episode.script_text, "")
        self.assertEqual(episode.word_count, 0)
        self.assertEqual(episode.duration_seconds, 0.0)
        with self.assertRaises(ValueError):
            publish_episode(episode, owner_approved=True)

    def test_unknown_segment_kind_rejected(self):
        with self.assertRaises(ValueError):
            compose_episode(_show(), [{"kind": "jingle", "text": "la la"}])


if __name__ == "__main__":
    unittest.main()
