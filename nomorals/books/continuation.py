"""Story continuation — keep a story going in its own voice.

Given a story Devon has read (fetched chapters via the reader, or text
pasted by the owner), ``StoryContinuer`` writes the next chapter(s):

* the story bible (auto-built/updated from the read chapters) steers
  voice, cast, open threads, and world rules;
* the last two chapters ride along for immediate continuity;
* the model writes when one is answering; the heuristic composer — real
  prose assembled from the bible's threads and beats, never placeholder —
  is the floor;
* every written chapter is digested back into the bible, so a 50-chapter
  continuation never forgets chapter 1.

The user's canonical example: "My Vampire System" by JKSManga — they
dislike the ending and want Devon to keep it going.  That is exactly
``continue_story("my-vampire-system-freewebnovel", n=...)`` after
following + reading the final chapters.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .bible import BibleBuilder, StoryBible
from .model import count_words, slugify
from .reader import ReaderError, StoryReader

_log = get_logger(__name__)

__all__ = ["StoryContinuer", "ContinuationError"]


class ContinuationError(Exception):
    """Continuation could not proceed."""


def _tail(text: str, max_chars: int = 3500) -> str:
    text = (text or "").strip()
    return text[-max_chars:] if len(text) > max_chars else text


class StoryContinuer:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.reader = StoryReader(context)
        self.bibles = BibleBuilder(context)

    # ── setup ─────────────────────────────────────────────────────────────
    def _story_slug(self, slug_or_url: str) -> str:
        return slugify(slug_or_url.split("/")[-1] or slug_or_url,
                       fallback="story")[:80]

    def prepare(self, slug: str, *,
                chapters: int = 6) -> StoryBible:
        """Ensure a bible exists: build from the last ``chapters`` read
        chapters (fetching them when the cache is cold)."""
        story = self.reader._load(slug)
        bible = self.bibles.load(slug)
        have = bible.chapters_digested if bible else 0
        # gather the most recent chapters Devon has (or can fetch)
        end = story.current_chapter or story.total_chapters or chapters
        start = max(1, end - chapters + 1)
        batch: list[tuple[int, str, str]] = []
        for n in range(start, end + 1):
            if n <= have:
                continue
            try:
                text = self.reader.chapter_text(slug, n)
            except ReaderError as exc:
                _log.warning("continuation prepare: ch%s: %s", n, exc)
                continue
            title = text.split("\n", 1)[0].lstrip("# ").strip()[:120]
            batch.append((n, title, text))
        if not batch and bible is None:
            raise ContinuationError(
                f"no chapters readable for {slug!r} — read some first")
        if bible is None:
            bible = self.bibles.build(slug, story.title, batch)
        elif batch:
            bible = self.bibles.update(slug, batch)
        return bible

    def prepare_from_text(self, title: str, text: str) -> StoryBible:
        """Build a bible from pasted text (owner pastes the story so far)."""
        slug = slugify(title, fallback="story")
        # split pasted text into rough chapters on chapter headings
        parts = re.split(r"(?im)^(?:chapter|chap\.?)\s+(\d+)[^\n]*$",
                         "\n" + text)
        chapters: list[tuple[int, str, str]] = []
        # re.split with a group: [pre, num1, body1, num2, body2, ...]
        if len(parts) >= 3:
            for i in range(1, len(parts) - 1, 2):
                try:
                    n = int(parts[i])
                except ValueError:
                    continue
                body = parts[i + 1].strip()
                if len(body) > 100:
                    first_line = body.split("\n", 1)[0][:120]
                    chapters.append((n, first_line, body))
        if not chapters:
            chapters = [(1, title, text)]
        return self.bibles.build(slug, title, chapters)

    # ── writing ───────────────────────────────────────────────────────────
    def continue_story(self, slug: str, n: int = 1, *,
                       words: int = 1500,
                       direction: str = "") -> dict[str, Any]:
        """Write ``n`` new chapters continuing the story."""
        story = self.reader._load(slug)
        bible = self.prepare(slug)
        written: list[dict[str, Any]] = []
        for _ in range(max(1, n)):
            chapter_no = (story.current_chapter or bible.chapters_digested) + 1
            result = self._write_one(slug, story.title, bible, chapter_no,
                                     words=words, direction=direction)
            written.append(result)
            # digest back into the bible so continuity compounds
            bible = self.bibles.update(
                slug, [(chapter_no, result["title"], result["text"])])
            story.current_chapter = chapter_no
            story.last_read_at = self.reader._load(slug).last_read_at
            # persist progress without clobbering the cache count
            cur = self.reader._load(slug)
            cur.current_chapter = chapter_no
            self.reader._save(cur)
            direction = ""  # only the first chapter takes a direction
        return {"slug": slug, "title": story.title,
                "chapters": written,
                "bible": self.bibles.bible_path(slug).as_posix()}

    def _write_one(self, slug: str, title: str, bible: StoryBible,
                   chapter_no: int, *, words: int,
                   direction: str) -> dict[str, Any]:
        # immediate continuity: tails of the last two chapters
        tails: list[str] = []
        for n in (chapter_no - 2, chapter_no - 1):
            if n < 1:
                continue
            try:
                tails.append(_tail(self.reader.chapter_text(slug, n)))
            except ReaderError:
                continue
        text = self._model_continuation(
            bible, chapter_no, tails, words, direction)
        if not text:
            text = self._compose_continuation(
                bible, chapter_no, tails, words, direction)
        ch_title = f"Chapter {chapter_no}"
        first_line = text.split("\n", 1)[0].strip()
        if first_line and len(first_line) < 120 and not first_line.endswith(
                (".", "!", "?", '"', "\u201c", "\u201d")):
            ch_title = f"Chapter {chapter_no}: {first_line[:80]}"
        # save alongside the fetched chapters, clearly marked
        out_dir = self.reader.story_dir(slug) / "continuations"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"chapter-{chapter_no:05d}.md"
        path.write_text(f"# {ch_title}\n\n*Devon continuation — chapter {chapter_no}*\n\n{text}\n",
                        encoding="utf-8")
        return {"number": chapter_no, "title": ch_title,
                "words": count_words(text), "path": path.as_posix(),
                "text": text}

    # -- model path -----------------------------------------------------------
    def _model_continuation(self, bible: StoryBible, chapter_no: int,
                            tails: list[str], words: int,
                            direction: str) -> str:
        try:
            from ..llm.base import Message, SamplingParams
            from ..llm.brain import brain_for
            from .write import model_available
        except Exception:  # noqa: BLE001
            return ""
        if not model_available(self.context):
            return ""
        brief = bible.brief()
        tail_text = "\n\n---\n\n".join(tails[-2:])
        user = (
            f"{brief}\n\n"
            f"END OF PREVIOUS CHAPTER(S):\n{tail_text}\n\n"
            f"Write CHAPTER {chapter_no} continuing this story RIGHT where it "
            f"left off — same voice ({bible.pov}-person, {bible.tense} "
            f"tense), same characters behaving like themselves, open threads "
            f"advanced not dropped, world rules obeyed. "
            f"~{words} words of real prose with dialogue and action. "
            "Do NOT summarize, do NOT break character, do NOT moralize.")
        if direction:
            user += f"\nStory direction for this chapter: {direction}"
        try:
            response = brain_for(self.context).chat(
                [Message.system(
                    "You are the original author's ghostwriter. You continue "
                    "the story seamlessly — the reader must not feel a "
                    "change of hand. Complete prose only, no meta-commentary."),
                 Message.user(user)],
                SamplingParams(temperature=0.75,
                               max_tokens=min(12000, int(words * 1.8) + 400)),
                task_kind="creative")
            text = (getattr(response, "text", "") or "").strip()
            if getattr(response, "ok", False) and count_words(text) >= 150:
                return text
        except Exception as exc:  # noqa: BLE001
            _log.debug("model continuation failed: %s", exc)
        return ""

    # -- heuristic composer (the floor) ---------------------------------------
    def _compose_continuation(self, bible: StoryBible, chapter_no: int,
                              tails: list[str], words: int,
                              direction: str) -> str:
        """Real prose from the bible: advance the hottest open thread,
        keep the cast in character, land on a hook.  Deterministic per
        chapter number (seeded) so re-runs are stable."""
        rng = random.Random(hash((bible.story_slug, chapter_no)) & 0xFFFFFFFF)
        threads = bible.open_threads()
        thread = max(threads, key=lambda t: t.heat) if threads else None
        chars = bible.characters
        protag = next((c for c in chars if c.role == "protagonist"), None)
        if protag is None and chars:
            protag = chars[0]
        deut = next((c for c in chars
                     if c is not protag and c.role != "protagonist"), None)
        p_name = protag.name if protag else "the protagonist"
        p_desc = (protag.description[:160] + "." if protag and protag.description
                  else "")
        d_name = deut.name if deut else None

        beats: list[str] = []
        # 1. cold open — resume mid-motion from the tail
        beats.append(
            f"The moment did not wait for {p_name} to catch {self._possessive(p_name)} breath.")
        # 2. thread pressure
        if thread:
            beats.append(
                f"Every instinct said the same thing: {thread.summary.rstrip('.')}, "
                f"and time was already running short.")
        # 3. character beat
        if d_name:
            beats.append(
                f"{d_name} was watching, the way {d_name} always watched — "
                f"measuring, waiting for the crack to show.")
        # 4. world texture
        if bible.world_rules:
            rule = rng.choice(bible.world_rules)
            beats.append(f"The old rule held, as it always did: {rule.rstrip('.')}.")
        # 5. action turn
        beats.append(
            f"{p_name} moved before doubt could finish its sentence — "
            f"{self._action(rng, p_name)}.")
        # 6. complication
        beats.append(self._complication(rng, p_name, d_name, thread))
        # 7. hook ending
        beats.append(self._hook(rng, p_name, thread))

        if direction:
            beats.insert(3, direction.rstrip(".") + ".")
        paras: list[str] = []
        for beat in beats:
            paras.append(self._expand(beat, rng, bible, p_name, d_name))
        # pad toward the word target with a second movement when very short
        text = "\n\n".join(paras)
        while count_words(text) < words * 0.5:
            paras.append(self._expand(self._mid_beat(rng, p_name, d_name, thread),
                                      rng, bible, p_name, d_name))
            text = "\n\n".join(paras)
            if len(paras) > 14:
                break
        return text

    @staticmethod
    def _possessive(name: str) -> str:
        return "their" if " " in name else "his"

    @staticmethod
    def _action(rng: random.Random, p_name: str) -> str:
        return rng.choice([
            f"across the broken ground toward the sound",
            f"hand closing around the hilt of {p_name.split()[0].lower()}'s blade",
            f"into the dark with nothing but momentum and spite",
            f"toward the light, because standing still had stopped being an option",
            f"with the old precision, every step a decision",
        ])

    @staticmethod
    def _complication(rng: random.Random, p_name: str, d_name: str | None,
                      thread: Any) -> str:
        who = d_name or "the shadows"
        goal = thread.summary.rstrip(".") if thread else "what had to be done"
        return rng.choice([
            f"But {who} had been waiting — of course {who} had been waiting — "
            f"and {goal} would not come cheap.",
            f"The plan survived exactly until contact. Then {who} moved, and "
            f"{p_name} understood, too late, the shape of the trap.",
            f"What {p_name} had not counted on was {who}: a variable with "
            f"teeth, and {goal} hanging in the balance.",
        ])

    @staticmethod
    def _hook(rng: random.Random, p_name: str, thread: Any) -> str:
        goal = thread.summary.rstrip(".") if thread else "the truth"
        return rng.choice([
            f"And in the distance, something answered — a sound {p_name} "
            f"recognized, and wished {p_name.split()[0].lower()} did not.",
            f"{p_name} smiled then, the thin dangerous smile of someone who "
            f"has just realized {goal} was never the real prize.",
            f"The chapter of hesitation was over. What came next would not "
            f"ask permission.",
        ])

    @staticmethod
    def _mid_beat(rng: random.Random, p_name: str, d_name: str | None,
                  thread: Any) -> str:
        goal = thread.summary.rstrip(".") if thread else "the task"
        partner = f" {d_name} kept pace without a word." if d_name else ""
        return rng.choice([
            f"They covered ground in silence, {p_name} turning over {goal} "
            f"like a stone with something moving underneath it.{partner}",
            f"Somewhere behind them the world went on arguing with itself. "
            f"Ahead, {goal} waited, patient as weather.{partner}",
            f"{p_name} checked {p_name.split()[0].lower()}'s reserves — "
            f"strength, nerve, options — and found all three thinner than "
            f"honesty allowed.{partner}",
        ])

    def _expand(self, beat: str, rng: random.Random, bible: StoryBible,
                p_name: str, d_name: str | None) -> str:
        """Grow one beat into a full paragraph with voice texture."""
        textures = [
            f"The air had the particular weight of {rng.choice(['evening', 'the hour before dawn', 'a held breath'])}.",
            f"Somewhere, {rng.choice(['a bell', 'water', 'a distant engine', 'insects'])} kept the time nobody else would keep.",
            f"{p_name} had learned young that {rng.choice(['hesitation', 'mercy', 'noise'])} got people buried.",
        ]
        para = beat + " " + rng.choice(textures)
        if bible.tone and "dialogue-driven" in bible.tone and rng.random() < 0.5:
            speaker = d_name or p_name
            line = rng.choice([
                "“We don't have long.”",
                "“Then we stop asking for more time.”",
                "“You feel that too, don't you.”",
                "“Stay close. Whatever happens next, stay close.”",
            ])
            para += f"\n\n{line} — {speaker.split()[0]}"
        return para
