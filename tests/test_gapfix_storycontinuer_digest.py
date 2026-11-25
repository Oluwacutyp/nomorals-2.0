"""Story continuer, made real: read + digest + continue.

The owner's verdict (real-build queue §5a): the novel continuer was
"still mid" — it should actually READ the story, DIGEST it, and
CONTINUE where it stopped, not write a new story from the start.
Reproduced 2026-11-24 on a 3-chapter vampire story: the System quest
("Drink the blood of the Seventh Elder before the third night ends,
or the thirst will take me instead") was never mined as a thread, the
heuristic composer's chapter 4 mentioned neither Dren, nor the
chapel, nor the silvered blade, nor the third night, opened "Mara to
catch his breath", and repeated the same paragraph three times while
padding.  These tests pin the fixed behaviour.  No model in the test
context — the heuristic composer is exercised.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.continuation import StoryContinuer


def _ctx():
    tmp = tempfile.mkdtemp(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


STORY = """Chapter 1 — The Bite

Quinn woke with the taste of copper in his mouth. "You died last night,"
Mara said, not looking up from her book. "The System has marked you, Quinn.
Drink the blood of the Seventh Elder before the third night ends, or the
thirst will take me instead." "I must find out what I am now," Quinn said.
The old rule was simple: the thirst comes first, and it does not negotiate.
The System chimed: [Quest: Drink the blood of the Seventh Elder. Time
remaining: three nights. Failure: the thirst claims Mara.]

Chapter 2 — The Thirst

"I will not drink from the living," Quinn vowed. Mara laughed, a dry
sound. "The Seventh Elder is Dren, and Dren is my own sire," she said.
"We need to leave the city before the Vigil finds you," she said. The
Vigil cannot be reasoned with; it can only be outrun. At the north gate
the Vigil's hunters closed in, and Quinn lost his left hand to a silvered
axe before Mara dragged him clear. They fled to the Blackspire chapel,
where Dren was said to wait.

Chapter 3 — The Chapel Door

Dawn was an hour away when Quinn reached the chapel door. Inside, Dren
waited with a silvered blade across his knees, patient as weather. Mara
stood behind Quinn, bleeding from a cut on her shoulder, her hand pressed
hard against it. "One hour left on the third night," Dren said, not
rising. "Drink, boy, or watch her fade. The System does not bargain."
"""


def _chapters(text):
    import re
    parts = re.split(r"(?im)^Chapter \d+ — [^\n]*$", "\n" + text)
    return [p.strip() for p in parts if p.strip()]


class TestQuestDigestion(unittest.TestCase):
    def test_system_quest_is_the_hottest_thread(self) -> None:
        ctx, _ = _ctx()
        bible = StoryContinuer(ctx).prepare_from_text(
            "My Vampire System", STORY)
        quest = [t for t in bible.open_threads()
                 if "Seventh Elder" in t.summary]
        # the System notification and Mara's spoken order are ONE
        # quest, not two near-duplicate threads
        self.assertEqual(len(quest), 1)
        quest = quest[0]
        self.assertIn("third night", quest.summary)
        self.assertIn("Mara", quest.summary)
        self.assertNotIn(" me ", f" {quest.summary} ")
        # hotter than every first-person goal statement
        goals = [t for t in bible.open_threads()
                 if t is not quest]
        self.assertTrue(all(quest.heat > t.heat for t in goals))

    def test_bracket_quest_notification_is_mined(self) -> None:
        ctx, _ = _ctx()
        bible = StoryContinuer(ctx).prepare_from_text(
            "Quest Only",
            "Chapter 1 — The Call\n\n"
            "The System chimed: [Quest: Find the Ember Crown before "
            "dawn, or the city burns.] Mara stared at the light.\n")
        quest = [t for t in bible.open_threads()
                 if "Ember Crown" in t.summary]
        self.assertEqual(len(quest), 1)
        self.assertGreaterEqual(quest[0].heat, 5.0)

    def test_state_sentence_belongs_to_its_subject(self) -> None:
        ctx, _ = _ctx()
        bible = StoryContinuer(ctx).prepare_from_text(
            "My Vampire System", STORY)
        by_name = {c.name: c for c in bible.characters}
        # Quinn keeps HIS injury: the ch.3 transit sentence ("reached
        # the chapel door") must not bury it, and Mara's bleeding
        # shoulder must not be assigned to Quinn as well.
        self.assertIn("left hand", by_name["Quinn"].state)
        self.assertEqual(by_name["Quinn"].state_chapter, 2)
        self.assertIn("bleeding", by_name["Mara"].state)
        self.assertIn("shoulder", by_name["Mara"].state)
        self.assertEqual(by_name["Mara"].state_chapter, 3)

    def test_capture_state_belongs_to_the_captive_not_the_captor(
            self) -> None:
        # regression (2026-11-24): a purely positional subject rule
        # handed "The Vigil captured Mara" to the Vigil — a mined
        # character name — and Mara kept a chapter-1 state instead.
        ctx, _ = _ctx()
        bible = StoryContinuer(ctx).prepare_from_text(
            "Capture Test",
            "Chapter 1 — The Trap\n\n"
            "The Vigil captured Mara at the bridge and dragged her "
            "into the dark before Quinn could cross. Mara had crossed "
            "that bridge a hundred times, and never once been caught. "
            "Quinn watched the dark take her and memorised every "
            "face.\n")
        by_name = {c.name: c for c in bible.characters}
        self.assertIn("captured Mara", by_name["Mara"].state)
        self.assertEqual(by_name["Mara"].state_chapter, 1)
        if "Vigil" in by_name:
            self.assertNotIn("captured Mara",
                             by_name["Vigil"].state)

    def test_location_mined_from_single_motion_mention(self) -> None:
        ctx, _ = _ctx()
        bible = StoryContinuer(ctx).prepare_from_text(
            "My Vampire System", STORY)
        places = {x.name for x in bible.locations}
        self.assertIn("Blackspire chapel", places)


class TestComposerContinuesTheStory(unittest.TestCase):
    def _chapter4(self):
        ctx, _ = _ctx()
        cont = StoryContinuer(ctx)
        bible = cont.prepare_from_text("My Vampire System", STORY)
        chaps = _chapters(STORY)
        tails = [chaps[0][-3000:], chaps[1][-3000:], chaps[2][-8000:]]
        scene = cont._final_scene_anchor(chaps[2], bible)
        text = cont._compose_continuation(
            bible, 4, tails, words=600, direction="", scene=scene)
        return text, scene, bible, cont, tails

    def test_composer_carries_the_scene_forward(self) -> None:
        text, _, _, _, _ = self._chapter4()
        # every fact the story stopped on is carried forward
        for needle in ("Dren", "Blackspire chapel", "silvered blade",
                       "shoulder", "left hand", "third night", "thirst",
                       "Seventh Elder"):
            self.assertIn(needle, text)
        # the closing lines are continued FROM, never echoed
        self.assertNotIn("One hour left on the third night", text)
        # the old composer's defects are gone
        self.assertNotIn("Mara to catch his breath", text)
        self.assertNotIn("mara's", text.lower())
        self.assertNotIn("Dren fled", text)
        self.assertNotIn("..", text)
        # no paragraph ever repeats (the old padding loop repeated
        # the same mid-beat three times)
        paras = [p for p in text.split("\n\n") if p.strip()]
        self.assertEqual(len(paras), len(set(paras)))
        self.assertGreater(len(text.split()), 150)

    def test_composer_opens_on_the_onstage_cast(self) -> None:
        text, scene, _, _, _ = self._chapter4()
        first = text.split("\n\n")[0]
        self.assertIn(scene["present"][0], first)

    def test_composer_advances_the_hottest_thread(self) -> None:
        text, _, bible, _, _ = self._chapter4()
        hot = max(bible.open_threads(), key=lambda t: t.heat)
        key = set(w for w in hot.summary.lower().split() if len(w) > 3)
        self.assertTrue(key & set(text.lower().split()))

    def test_composer_is_deterministic_and_takes_vary(self) -> None:
        _, _, bible, cont, tails = self._chapter4()
        chaps = _chapters(STORY)
        scene = cont._final_scene_anchor(chaps[2], bible)
        a = cont._compose_continuation(bible, 4, tails, words=600,
                                       direction="", scene=scene)
        b = cont._compose_continuation(bible, 4, tails, words=600,
                                       direction="", scene=scene)
        self.assertEqual(a, b)
        c = cont._compose_continuation(bible, 4, tails, words=600,
                                       direction="", scene=scene,
                                       seed_salt=1)
        self.assertNotEqual(a, c)

    def test_direction_still_woven_in(self) -> None:
        ctx, _ = _ctx()
        cont = StoryContinuer(ctx)
        bible = cont.prepare_from_text("My Vampire System", STORY)
        text = cont._compose_continuation(
            bible, 4, [], words=400,
            direction="Quinn meets the Vigil's captain")
        self.assertIn("Vigil", text)


if __name__ == "__main__":
    unittest.main()
