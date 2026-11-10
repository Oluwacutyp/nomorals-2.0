"""Doc 16 Pass 2 — the /books surfaces over the generalized reader.

The defect this pins shut (evaluation Finding A): on no-pager
surfaces (WhatsApp), /books read used to show text[:3500] of
read()'s already-capped 12000-char preview and point at the NEXT
CHAPTER as "the next part" — chars 3500→end of the current chapter
were unreachable and the pointer lied about its destination.  Now:

- /books read on a no-pager surface serves the chapter's FULL text
  through the reader session, one part at a time; the mid-chapter
  pointer names the next part of THIS chapter, and the next chapter
  is named only once this chapter has ended;
- the /novel fallback (and the /books pager fallback) sends the
  full chapter_text through the chunked send, never the preview;
- /books contents shows held vs not-held truthfully;
- /books go actuates reader-session tokens with the stale-token
  discipline;
- pager surfaces keep the live pager render (existing grammar).

Fake adapter only — no network.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import reader as reader_mod
from nomorals.books.reader import StoryReader
from nomorals.books.sources import Chapter, StoryMeta
from nomorals.social.chat.reader_service import ReaderService


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings, extras={}), tmp


def _para(tag: str, chars: int) -> str:
    seed = f"{tag} unfolds slowly, sentence over sentence. "
    return (seed * (chars // len(seed) + 1))[:chars]


LONG_CH1 = "\n\n".join(
    [_para("Opening", 4000), _para("Middle", 4500),
     _para("Late", 4200), "The final breath. FINAL-BREATH-MARKER"])
MID_CH3 = "\n\n".join([_para("Third", 2600), _para("Third-b", 2400)])


class _Adapter:
    name = "fake"
    verified = True
    bodies = {1: LONG_CH1, 2: "Short second chapter.",
              3: MID_CH3, 4: "Fourth and last."}
    total = 4

    def novel(self, url: str) -> StoryMeta:
        return StoryMeta(title="The Long Tale", url=url, source=self.name,
                         author="A. Teller", status="Ongoing",
                         total_chapters=self.total)

    def chapter_list(self, novel_url: str, *, limit: int = 0):
        out = [(n, f"Chapter {n}", f"https://fake/ch/{n}")
               for n in range(1, self.total + 1)]
        return out[:limit] if limit else out

    def chapter_url(self, novel_url: str, number: int) -> str:
        return f"https://fake/ch/{number}" if 1 <= number <= self.total else ""

    def fetch_chapter(self, url: str) -> Chapter:
        n = int(url.rsplit("/", 1)[-1])
        return Chapter(
            number=n, title=f"Chapter {n}", url=url,
            paragraphs=self.bodies[n].split("\n\n"),
            prev_url=f"https://fake/ch/{n - 1}" if n > 1 else "",
            next_url=f"https://fake/ch/{n + 1}" if n < self.total else "",
            source=self.name)


_FAST_PROFILE = {"sync_chapters": 3, "list_limit": 50,
                 "pause_s": 0.0, "timeout": 5.0}
_URL = "https://fake/novel/the-long-tale"


def _served(reader, slug: str, number: int) -> str:
    """The chapter's served canonical text: chapter_text() minus
    its presentation header line (the surfaces strip it and add
    their own head — the shipped convention)."""
    full = reader.chapter_text(slug, number)
    if full.startswith("# "):
        full = full.split("\n", 1)[1].lstrip("\n")
    return full


def _body_of(reply: str) -> str:
    """The page body out of a plain part render (head, body, footer)."""
    after_head = reply.split("\n\n", 1)[1]
    return after_head.rsplit("\n\n— part ", 1)[0]


class _SurfaceCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.adapter = _Adapter()
        self._orig_adapter = reader_mod.adapter_for_url
        reader_mod.adapter_for_url = lambda url: self.adapter
        self._orig_profile = StoryReader._profile
        StoryReader._profile = lambda self: dict(_FAST_PROFILE)
        self.r = StoryReader(self.ctx)
        self.slug = self.r.follow(_URL).slug
        from nomorals.agents.partner.runtime_search import RuntimeSearchMixin
        m = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
        m.context = self.ctx
        self.m = m

    def tearDown(self) -> None:
        reader_mod.adapter_for_url = self._orig_adapter
        StoryReader._profile = self._orig_profile


class WhatsAppReadTests(_SurfaceCase):
    def test_long_chapter_fully_reachable_part_by_part(self):
        self.assertGreater(len(LONG_CH1), 12000)
        reply = self.m._control_book(f"read {self.slug} 1", "whatsapp:123")
        self.assertNotIn("truncated", reply)
        bodies = [_body_of(reply)]
        guard = 0
        while "next part of this chapter" in reply:
            reply = self.m._control_book(f"part {self.slug}",
                                         "whatsapp:123")
            bodies.append(_body_of(reply))
            guard += 1
            self.assertLess(guard, 20)
        # the walk ended because the chapter ended, and the pointer
        # now names the NEXT CHAPTER as the next chapter
        self.assertIn("end of chapter 1 — next chapter", reply)
        self.assertIn(f"/books read {self.slug} 2", reply)
        canonical = _served(self.r, self.slug, 1)
        self.assertEqual("".join(bodies), canonical)
        self.assertIn("FINAL-BREATH-MARKER", bodies[-1])

    def test_mid_chapter_pointer_never_names_the_next_chapter(self):
        reply = self.m._control_book(f"read {self.slug} 1", "whatsapp:123")
        self.assertIn("next part of this chapter", reply)
        self.assertNotIn("next chapter", reply)

    def test_5000_char_chapter_fully_reachable(self):
        # the old path stranded everything past char 3500
        self.assertGreater(len(MID_CH3), 3500)
        reply = self.m._control_book(f"read {self.slug} 3", "whatsapp:123")
        bodies = [_body_of(reply)]
        while "next part of this chapter" in reply:
            reply = self.m._control_book(f"part {self.slug}",
                                         "whatsapp:123")
            bodies.append(_body_of(reply))
        canonical = _served(self.r, self.slug, 3)
        self.assertEqual("".join(bodies), canonical)
        self.assertIn("next chapter", reply)  # chapter 4 named at the end

    def test_part_back_and_forward_cross_chapters(self):
        self.m._control_book(f"read {self.slug} 2", "whatsapp:123")
        reply = self.m._control_book(f"part {self.slug}", "whatsapp:123")
        self.assertIn("chapter 3", reply.split("\n", 1)[0])
        back = self.m._control_book(f"part {self.slug} back",
                                    "whatsapp:123")
        self.assertIn("Short second chapter.", back)

    def test_part_without_a_read_answers_honestly(self):
        reply = self.m._control_book(f"part {self.slug}", "whatsapp:123")
        self.assertIn("no reading in progress", reply)


class ContentsAndTokenTests(_SurfaceCase):
    def test_contents_names_held_and_unheld(self):
        self.m._control_book(f"read {self.slug} 1", "whatsapp:123")
        reply = self.m._control_book(f"contents {self.slug}",
                                     "whatsapp:123")
        self.assertIn("1 of 4 held", reply)
        self.assertIn("not held", reply)
        self.assertIn("Chapter 1", reply)

    def test_books_go_moves_then_stale_tap_is_a_noop(self):
        self.m._control_book(f"read {self.slug} 1", "whatsapp:123")
        svc = ReaderService.for_context(self.ctx)
        session = svc.session_for("chat:whatsapp:123",
                                  f"story:{self.slug}")
        self.assertIsNotNone(session)
        token = session.token
        moved = self.m._control_book(f"go {token} n", "whatsapp:123")
        self.assertIn("— part 2 of", moved)
        after = ReaderService.for_context(self.ctx).session_for(
            "chat:whatsapp:123", f"story:{self.slug}")
        self.assertGreater(after.offset, 0)
        stale = self.m._control_book(f"go {token} n", "whatsapp:123")
        self.assertIn("older page", stale)
        stayed = ReaderService.for_context(self.ctx).session_for(
            "chat:whatsapp:123", f"story:{self.slug}")
        self.assertEqual(stayed.offset, after.offset)
        unknown = self.m._control_book("go ffffffffffff n", "whatsapp:123")
        self.assertIn("expired", unknown)


class FallbackAndPagerTests(_SurfaceCase):
    def test_novel_whatsapp_fallback_sends_full_chapter(self):
        sent: list = []
        self.m._ref_from_key = lambda key: SimpleNamespace(
            platform="whatsapp")
        self.m._send_long_checked = (
            lambda platform, chat, text: sent.append(text) or "")
        reply = self.m._control_novel(f"read {self.slug} 1",
                                      "whatsapp:123")
        self.assertEqual(reply, "")
        # sends: the cover (short), then the chapter — the chapter
        # send carries the FULL text, tail marker included, not the
        # 12000-char preview
        chapter_sends = [t for t in sent if "FINAL-BREATH-MARKER" in t]
        self.assertEqual(len(chapter_sends), 1)
        self.assertGreater(len(chapter_sends[0]), 12000)

    def test_telegram_read_still_renders_the_live_pager(self):
        reply = self.m._control_book(f"read {self.slug} 1",
                                     "telegram:123")
        self.assertTrue(hasattr(reply, "text") and
                        hasattr(reply, "buttons"),
                        f"expected an actionable pager render, got {reply!r}")
        self.assertEqual(reply.session, "books")
        self.assertIn("chapter 1", reply.text)


if __name__ == "__main__":
    unittest.main()
