"""Continuation grounding — the owner's defect, pinned.

Following a novel and continuing it produced a chapter that read like
a NEW novel from the start: the bible's memory was frozen at the
story's opening (arc summary from a chapters 1–6 sample, character
descriptions from first mentions), the prompt carried only a sliver
of the final chapter, and nothing anchored the model at the final
scene.  These tests pin the fixed behaviour on a synthetic followed
novel with specific late-story facts:

* ch.5 — Quinn takes the Ember Key (a timeline fact);
* ch.6 — Mara is captured by the Vigil (her current state);
* ch.7 — Quinn's left hand is crushed (his current state);
* ch.8 — the final chapter stops mid-scene at the ember gate, with an
  open vow: "I will open the ember gate before the Vigil returns".

No model in the test context: a recording fake stands in for the
model path and captures the exact prompt the continuer assembles.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import reader as reader_mod
from nomorals.books.bible import BibleBuilder
from nomorals.books.continuation import StoryContinuer
from nomorals.books.reader import StoryReader
from nomorals.books.sources import Chapter, SourceError, StoryMeta


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


_CHAPTERS: dict[int, list[str]] = {
    1: ["Quinn woke with the taste of copper in his mouth. "
        "\"You died last night,\" Mara said, not looking up from her "
        "book. The old rule was simple: the thirst comes first, and "
        "it does not negotiate.",
        "\"I must find out what I am now,\" Quinn said. Mara "
        "shrugged. \"Then stop bleeding on my floor,\" Mara said."],
    2: ["\"I will not drink from the living,\" Quinn vowed. Mara "
        "laughed, a dry sound. \"We need to leave the city before "
        "the Vigil finds you,\" Mara said.",
        "The Vigil cannot be reasoned with; it can only be outrun. "
        "Quinn nodded, and Mara began to pack."],
    3: ["They descended into the Sunken Archive beneath the city, "
        "where the old maps slept under glass. \"The gate is real,\" "
        "Mara said. \"The maps do not lie.\"",
        "Quinn traced the route with one finger. \"Then we go "
        "through the archive,\" Quinn said."],
    4: ["Inside the Sunken Archive, dust lay thick on the reading "
        "desks. Something moved between the stacks. \"Quinn, "
        "behind you,\" Mara said, and Quinn turned with the knife "
        "already out.",
        "The thing in the stacks fled upward. Mara let out a slow "
        "breath. \"It knows your name,\" Mara said."],
    5: ["At the altar stone Quinn stopped. \"That is it,\" Mara "
        "said. Quinn took the Ember Key from the altar, and its "
        "teeth bit his palm.",
        "\"It chose you,\" Mara said. Quinn closed his fingers "
        "around the Ember Key and did not answer."],
    6: ["The trap was patient and the bridge was narrow. The Vigil "
        "captured Mara at the bridge and dragged her into the "
        "dark before Quinn could cross.",
        "\"Run,\" Mara shouted, but Quinn did not run. Quinn "
        "watched the dark take her and memorised every face."],
    7: ["The second trap was cruder and faster. The trap crushed "
        "Quinn's left hand; the bones ground together and would "
        "never set right.",
        "Quinn bound the hand without a sound. \"It still holds a "
        "key,\" Quinn said. \"That is enough.\""],
    8: ["Quinn stood before the ember gate with the Ember Key cold "
        "in his ruined left hand. The gate's seam glowed like a "
        "held breath.",
        "\"I will open the ember gate before the Vigil returns,\" "
        "Quinn said. Then the seam split wide and a bell began to "
        "toll beneath the world."],
    9: ["Quinn crossed the ash plain alone, the Ember Key heavy "
        "against his ribs. \"One more gate,\" Quinn said to the "
        "empty road.",
        "The ash gate rose out of the murk ahead, black and "
        "patient."],
    10: ["At the ash gate Quinn lifted the Ember Key with his "
         "ruined left hand. \"Open,\" Quinn said, and somewhere "
         "below the world a second bell answered the first."],
}


class _FactAdapter:
    """Fake archive whose chapters carry the facts above."""

    name = "freewebnovel"
    verified = True

    def __init__(self, total: int = 8) -> None:
        self.total = total
        self.title = "My Vampire System"
        self.fetch_log: list[int] = []

    def novel(self, url: str) -> StoryMeta:
        return StoryMeta(title=self.title, url=url, source=self.name,
                         author="JKSManga", synopsis="A vampire system.",
                         status="Ongoing", total_chapters=self.total)

    def chapter_list(self, novel_url: str, *, limit: int = 0):
        out = [(n, f"Chapter {n}", f"https://fake/ch/{n}")
               for n in range(1, self.total + 1)]
        return out[:limit] if limit else out

    def chapter_url(self, novel_url: str, number: int) -> str:
        return f"https://fake/ch/{number}" \
            if 1 <= number <= self.total else ""

    def fetch_chapter(self, url: str) -> Chapter:
        n = int(url.rsplit("/", 1)[-1])
        self.fetch_log.append(n)
        if n not in _CHAPTERS or n > self.total:
            raise SourceError(f"no chapter {n}")
        return Chapter(
            number=n, title=f"Chapter {n}", url=url,
            paragraphs=list(_CHAPTERS[n]),
            prev_url=f"https://fake/ch/{n - 1}" if n > 1 else "",
            next_url=f"https://fake/ch/{n + 1}" if n < self.total else "",
            source=self.name)


_FAST_PROFILE = {"sync_chapters": 3, "list_limit": 3,
                 "pause_s": 0.0, "timeout": 5.0}

_TEXT9 = (
    "DEVON-CH9-MARKER. Quinn now stood with the Ember Key pressed "
    "to the seam, his ruined left hand shaking against the heat. "
    "The gate drank the light and gave back a low, pleased hum. "
    "\"I will burn the Vigil archive before dawn,\" Quinn said, "
    "and the seam widened another inch. Behind him the tolling "
    "bell climbed toward a note it had never reached, and Quinn "
    "felt the old thirst rise and wait, patient as debt. He thought "
    "of Mara in the dark below the bridge and set his jaw. "
    "\"One lock at a time,\" Quinn said. The Ember Key turned a "
    "fraction, and heat licked between his fingers.")

_TEXT10 = (
    "DEVON-CH10-MARKER. The seam became a doorway and the doorway "
    "became a road of cinders. Quinn stepped through with the Ember "
    "Key raised, counting bells. \"Mara first,\" Quinn said. "
    "\"Then the archive burns.\" The dark ahead of him arranged "
    "itself into corridors he half remembered from the maps, and "
    "every corridor smelled of rain on hot stone. He walked "
    "carefully, the way a man walks through a house where someone "
    "is sleeping, and the house was sleeping, and it was dreaming "
    "of him.")


class _GroundingCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.adapter = _FactAdapter()
        self._orig_adapter = reader_mod.adapter_for_url
        reader_mod.adapter_for_url = lambda url: self.adapter
        self._orig_profile = StoryReader._profile
        StoryReader._profile = lambda self: dict(_FAST_PROFILE)
        self.r = StoryReader(self.ctx)
        story = self.r.follow("https://fake/novel/my-vampire-system")
        self.slug = story.slug

    def tearDown(self) -> None:
        reader_mod.adapter_for_url = self._orig_adapter
        StoryReader._profile = self._orig_profile

    def _continuer_with_recording_model(self):
        cont = StoryContinuer(self.ctx)
        calls: list[dict] = []

        def fake_model(bible, chapter_no, tails, words, direction,
                       temperature=0.75, scene=None):
            system, user = StoryContinuer._assemble_prompt(
                cont, bible, chapter_no, tails, words, direction,
                scene=scene)
            calls.append({"chapter_no": chapter_no,
                          "tails": list(tails), "system": system,
                          "user": user, "scene": scene})
            return _TEXT9 if chapter_no == 9 else _TEXT10

        cont._model_continuation = fake_model
        return cont, calls


class TestBibleDigestsTheWholeStory(_GroundingCase):
    def test_prepare_records_late_story_facts(self) -> None:
        cont = StoryContinuer(self.ctx)
        bible = cont.prepare(self.slug)
        self.assertEqual(bible.chapters_digested, 8)
        by_name = {c.name: c for c in bible.characters}
        self.assertIn("Quinn", by_name)
        self.assertIn("Mara", by_name)
        # current states come from the LATE chapters, not the opening
        self.assertIn("crushed", by_name["Quinn"].state)
        self.assertIn("left hand", by_name["Quinn"].state)
        self.assertEqual(by_name["Quinn"].state_chapter, 7)
        self.assertIn("captured Mara", by_name["Mara"].state)
        self.assertEqual(by_name["Mara"].state_chapter, 6)
        # timeline facts were recorded as progressions
        details = " ".join(p.detail for p in bible.progressions)
        self.assertIn("Ember Key", details)
        self.assertIn("crushed", details)
        # places, recaps and the closing situation all exist and are
        # anchored at the story's end, not its beginning
        self.assertIn("Sunken Archive",
                      {x.name for x in bible.locations})
        self.assertTrue(bible.recaps)
        self.assertEqual(bible.recaps[-1].end, 8)
        self.assertIn("End of chapter 8", bible.current_situation)
        self.assertIn("bell began to toll", bible.current_situation)
        # the voice target is measured from the digested story
        self.assertIsNotNone(bible.voice)
        self.assertGreater(bible.voice.words, 0)


class TestContinuePromptGrounding(_GroundingCase):
    def test_continue_prompt_carries_the_true_end(self) -> None:
        cont, calls = self._continuer_with_recording_model()
        result = cont.continue_story(self.slug, n=1)
        self.assertEqual([c["number"] for c in result["chapters"]], [9])
        self.assertEqual(result["continued_from_chapter"], 8)
        self.assertEqual(len(calls), 1)
        user = calls[0]["user"]
        # the bible's late facts are IN the prompt
        self.assertIn("crushed", user)               # Quinn's state
        self.assertIn("left hand", user)
        self.assertIn("captured Mara", user)         # Mara's state
        self.assertIn("Ember Key", user)             # the item
        self.assertIn("open the ember gate", user)   # the open vow
        # timeline position: this is chapter 9 of a long story
        self.assertIn("TIMELINE", user)
        self.assertIn("writing chapter 9", user)
        # the final scene anchors it: closing lines + on-stage cast
        self.assertIn("WHERE THE STORY STOPPED", user)
        self.assertIn("bell began to toll", user)
        self.assertIn("Quinn", (calls[0]["scene"] or {}).get(
            "present", []))
        # the anti-restart law is stated, not hinted
        self.assertIn("Do NOT re-introduce", user)
        self.assertIn("Do NOT recap", user)
        # drift + AI-tell reporting ride along, never blocking
        ch = result["chapters"][0]
        self.assertIn("voice_drift", ch)
        self.assertIn("ai_tells", ch)
        self.assertFalse(ch["ai_tells"]["blocked"])
        # persisted as the new last chapter; bible digested past it
        path = (self.r.story_dir(self.slug) / "continuations"
                / "chapter-00009.md")
        self.assertTrue(path.exists())
        bible = BibleBuilder(self.ctx).load(self.slug)
        self.assertGreaterEqual(bible.chapters_digested, 9)

    def test_second_continue_builds_on_the_first(self) -> None:
        cont, calls = self._continuer_with_recording_model()
        cont.continue_story(self.slug, n=1)
        result = cont.continue_story(self.slug, n=1)
        self.assertEqual([c["number"] for c in result["chapters"]],
                         [10])
        self.assertEqual(result["continued_from_chapter"], 9)
        second = calls[-1]
        # the first written chapter is in the second's tails
        self.assertIn("DEVON-CH9-MARKER", "\n".join(second["tails"]))
        # and its facts were digested: Quinn's state moved to ch.9
        self.assertIn("Ember Key pressed", second["user"])
        bible = BibleBuilder(self.ctx).load(self.slug)
        quinn = next(c for c in bible.characters if c.name == "Quinn")
        self.assertEqual(quinn.state_chapter, 9)
        self.assertGreaterEqual(bible.chapters_digested, 10)


class TestIncrementalDigestion(_GroundingCase):
    def test_prepare_digests_new_chapters_incrementally(self) -> None:
        cont = StoryContinuer(self.ctx)
        bible = cont.prepare(self.slug)
        self.assertEqual(bible.chapters_digested, 8)
        self.assertEqual(sorted(self.adapter.fetch_log),
                         list(range(1, 9)))
        # the source publishes two more chapters
        self.adapter.total = 10
        self.adapter.fetch_log = []
        bible = cont.prepare(self.slug)
        self.assertEqual(bible.chapters_digested, 10)
        # only the new chapters were fetched
        self.assertEqual(sorted(self.adapter.fetch_log), [9, 10])
        # the bible's memory advanced to the new end
        self.assertEqual(bible.recaps[-1].end, 10)
        self.assertIn("End of chapter 10", bible.current_situation)
        self.assertIn("second bell", bible.current_situation)


if __name__ == "__main__":
    unittest.main()
