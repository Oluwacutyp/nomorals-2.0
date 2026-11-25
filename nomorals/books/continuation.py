"""Story continuation — keep a story going in its own voice.

Given a story Devon has read (fetched chapters via the reader, or text
pasted by the owner), ``StoryContinuer`` writes the next chapter(s):

* the story bible (auto-built/updated from the read chapters) steers
  voice, cast, open threads, and world rules — and digestion keeps
  its memory CURRENT: character states, places, per-arc recaps, the
  story's closing situation, and a voiceprint measured from the
  latest chapters, so chapter 2,546 is steered by the story as it
  stands, not by its opening;
* the last three chapters ride along for immediate continuity, the
  final one near-full, plus an explicit final-scene anchor (who is on
  stage, the literal closing lines) with a no-recap, no-re-introduction
  instruction;
* the model writes when one is answering; the heuristic composer — real
  prose assembled from the bible's threads and beats, never placeholder —
  is the floor;
* every written chapter is digested back into the bible, so a 50-chapter
  continuation never forgets chapter 1.

The user's canonical example: "My Vampire System" by JKSManga — they
dislike the ending and want Devon to keep it going.  That is exactly
``continue_story("my-vampire-system-freewebnovel", n=...)`` after
following: the follow ingests the whole story into its library, and
the continuation anchors at the story's true final chapter — never
at the owner's reading position.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .bible import (DIGESTION_VERSION, BibleBuilder, StoryBible,
                    _STATE_MARKERS, bible_needs_redigest)
from .model import count_words, slugify
from .voiceprint import compare, fingerprint
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

    def _ensure_library(self, slug: str) -> int:
        """Ingest the story's library to the source's current end.

        Returns the final source chapter number.  Raises
        ContinuationError — naming how far the library got vs the
        source's end — when ingestion cannot complete: a continuation
        is NEVER written from a partial library.
        """
        try:
            report = self.reader.ingest(slug)
        except ReaderError as exc:
            raise ContinuationError(str(exc)) from exc
        if not report["complete"]:
            detail = ""
            if report.get("failed"):
                first = report["failed"][0]
                detail = (f"; first failure at chapter {first['number']}: "
                          f"{first['reason']}")
            raise ContinuationError(
                f"cannot continue {slug!r} from a partial library: "
                f"{report['ingested']} of {report['total']} chapters "
                f"are ingested{detail}. Finish the ingestion "
                f"(/novel sync {slug}) and continue again.")
        return int(report["total"])

    def _written_continuations(self, slug: str) -> list[int]:
        """Chapter numbers Devon has already written for this story."""
        out_dir = self.reader.story_dir(slug) / "continuations"
        nums: list[int] = []
        if out_dir.exists():
            for p in out_dir.glob("chapter-*.md"):
                m = re.fullmatch(r"chapter-(\d+)", p.stem)
                if m:
                    nums.append(int(m.group(1)))
        return sorted(nums)

    def _continuation_text(self, slug: str, number: int) -> str:
        """Text of a chapter DEVON wrote (continuations/), in the same
        shape reader.chapter_text returns for source chapters."""
        path = (self.reader.story_dir(slug) / "continuations"
                / f"chapter-{number:05d}.md")
        if not path.exists():
            raise ReaderError(f"no written chapter {number} for {slug!r}")
        text = path.read_text(encoding="utf-8")
        parts = text.split("\n\n", 2)
        body = parts[2] if len(parts) > 2 else text
        title = parts[0].lstrip("# ").strip() if parts else ""
        return f"# {title}\n\n{body}" if title else body

    def _any_chapter_text(self, slug: str, number: int, final: int) -> str:
        """Chapter text across the seam: source chapters up to the
        story's end, Devon's written chapters beyond it."""
        if final and number > final:
            return self._continuation_text(slug, number)
        try:
            return self.reader.chapter_text(slug, number)
        except ReaderError:
            if number > (final or 0):
                return self._continuation_text(slug, number)
            raise

    def prepare(self, slug: str, *,
                chapters: int = 6) -> StoryBible:
        """Ensure a bible exists, digested through the story's TRUE END.

        The library is first ingested to the source's current end
        (failing honestly when it cannot be), then every undigested
        chapter — from wherever an existing bible stopped, all the
        way to the final chapter — is folded in, in batches.  The old
        behaviour anchored the bible at the owner's reading position,
        so continuing a story read only at chapter 1 produced a
        "Chapter 2" from a near-empty bible.  ``chapters`` is retained
        for call compatibility; digestion now always runs to the end.

        A LEGACY bible — one written before the grounding fold
        existed — already claims the end, so the batch loop above
        would never touch it and its grounding would stay empty
        forever; :meth:`_ensure_grounding` re-digests it once.
        """
        story = self.reader._load(slug)
        slug = story.slug
        final = self._ensure_library(slug)
        bible = self.bibles.load(slug)
        if bible is not None:
            bible = self._ensure_grounding(slug, bible)
        have = bible.chapters_digested if bible else 0
        if have < final:
            batch: list[tuple[int, str, str]] = []

            def _flush() -> None:
                nonlocal bible, batch
                if not batch:
                    return
                if bible is None:
                    bible = self.bibles.build(slug, story.title, batch)
                else:
                    bible = self.bibles.update(slug, batch)
                batch = []

            try:
                for n in range(have + 1, final + 1):
                    text = self.reader.chapter_text(slug, n)
                    title = text.split("\n", 1)[0].lstrip("# ").strip()[:120]
                    batch.append((n, title, text))
                    if len(batch) >= 24:
                        _flush()
                _flush()
            except ReaderError as exc:
                raise ContinuationError(str(exc)) from exc
        if bible is None:
            raise ContinuationError(
                f"no chapters readable for {slug!r} — read some first")
        return bible

    def _ensure_grounding(self, slug: str,
                          bible: StoryBible) -> StoryBible:
        """Bring a LEGACY bible's grounding current — once.

        Bibles written before the grounding fold claim every chapter
        digested while their character states, arc recaps, locations
        and current situation stand empty: prepare's incremental loop
        keys off chapters_digested, so without this the books the
        owner has followed longest would keep grounding
        continuations on their openings forever.

        The re-digest folds the already-ingested LOCAL chapters
        through the fold in prepare's usual batches — no chapter is
        refetched, the reader's position is untouched, and the
        claimed digested range does not move.  Progress is saved per
        batch (``grounding_through``), so an interrupted rebuild
        resumes where it stopped and is only stamped current
        (``digestion_version``) when the whole claimed range is
        folded.  A bible stamped current is never re-digested; an
        unstamped one whose grounding demonstrably reaches its end
        (written by the fold before the stamp existed) is only
        stamped.  If a chapter cannot be read locally the rebuild
        PAUSES and the bible serves as it stands — older fields plus
        the prompt's true-end anchors — rather than blocking the
        continuation; the next prepare resumes it.
        """
        if bible.digestion_version >= DIGESTION_VERSION:
            return bible
        if not bible_needs_redigest(bible):
            # grounding folded before the stamp existed: mark
            # current from the data, never re-digest
            bible.digestion_version = DIGESTION_VERSION
            bible.grounding_through = bible.chapters_digested
            self.bibles.save(bible)
            return bible
        start = (bible.grounding_through or 0) + 1
        end = bible.chapters_digested
        batch: list[tuple[int, str, str]] = []
        try:
            for n in range(start, end + 1):
                text = self.reader.chapter_text(slug, n)
                title = text.split("\n", 1)[0].lstrip("# ").strip()[:120]
                batch.append((n, title, text))
                if len(batch) >= 24:
                    bible = self.bibles.refold(slug, batch)
                    batch = []
            if batch:
                bible = self.bibles.refold(slug, batch)
        except ReaderError as exc:
            _log.warning(
                "story bible re-digest for %s paused after chapter "
                "%d of %d: %s — serving the bible as it stands; the "
                "rebuild resumes on the next prepare",
                slug, bible.grounding_through, end, exc)
            return bible
        bible.digestion_version = DIGESTION_VERSION
        self.bibles.save(bible)
        _log.info(
            "story bible rebuilt for %s: re-digested chapters %d–%d "
            "through the grounding fold (legacy bible predated "
            "grounding digestion)", slug, start, end)
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

    # ── voice ─────────────────────────────────────────────────────────────
    def validate_voice(self, text: str, bible: StoryBible) -> dict[str, Any]:
        """Compare a new chapter against the bible's voice target.

        Reported, never blocking: a drift score, the largest deltas,
        and ``blocked: False``.  A bible with no measured voice yet
        reports drift 0 with an explicit note instead of inventing a
        target.
        """
        try:
            target = getattr(bible, "voice", None)
            if target is None or getattr(target, "words", 0) == 0:
                return {
                    "drift": 0.0,
                    "verdict": ("no voice target yet — the bible has no "
                                "measured voiceprint, nothing to drift from"),
                    "blocked": False,
                    "owner_may_overrule": True,
                    "largest_deltas": [],
                    "note": "Voice drift reported, never blocking.",
                }
            names = [c.name for c in (getattr(bible, "characters", []) or [])
                     if getattr(c, "name", "")]
            return compare(fingerprint(text or "", names), target)
        except Exception as exc:  # noqa: BLE001 - voice check is advisory
            return {"drift": 0.0, "blocked": False,
                    "owner_may_overrule": True, "error": str(exc),
                    "verdict": "voice check failed — reported, not blocking"}

    # ── writing ───────────────────────────────────────────────────────────
    def continue_story(self, slug: str, n: int = 1, *,
                       words: int = 1500,
                       direction: str = "",
                       takes: int = 1,
                       from_chapter: int = 0) -> dict[str, Any]:
        """Write ``n`` new chapters continuing the story.

        Anchored at the story's TRUE END: the library is ingested to
        the source's current end and the bible digested through the
        final chapter (see :meth:`prepare`) before anything is
        written.  The first new chapter is (final source chapter)+1;
        later continues increment from the last chapter Devon wrote
        (persisted under the story's ``continuations/``), so a second
        continue builds on the first.  The owner's reading position
        (``current_chapter``) is never moved by writing.

        ``takes`` = alternate versions per chapter (AI-Dungeon retry
        style); the best-scored take becomes the live chapter, the rest
        are kept beside it for the author to pick.  ``from_chapter``
        is the explicit fork override (SerialManager's continue-from-
        any-chapter): numbering and tails anchor there instead.
        """
        story = self.reader._load(slug)
        slug = story.slug
        bible = self.prepare(slug)
        state = self.reader.ingest_state(slug)
        final = int(state["total"])
        written = self._written_continuations(slug)
        if from_chapter and from_chapter >= 1:
            anchor = from_chapter
        else:
            anchor = max([final, *written])
        written_out: list[dict[str, Any]] = []
        last = anchor
        for _ in range(max(1, n)):
            chapter_no = last + 1
            result = self._write_one(slug, story.title, bible, chapter_no,
                                     words=words, direction=direction,
                                     takes=takes, final=final)
            written_out.append(result)
            last = chapter_no
            # digest back into the bible so continuity compounds
            bible = self.bibles.update(
                slug, [(chapter_no, result["title"], result["text"])])
            self._update_summary(bible, chapter_no, result["text"])
            self.bibles.save(bible)
            direction = ""  # only the first chapter takes a direction
        return {"slug": slug, "title": story.title,
                "chapters": written_out,
                "continued_from_chapter": anchor,
                "final_chapter": final,
                "bible": self.bibles.bible_path(slug).as_posix()}

    @staticmethod
    def _final_scene_anchor(text: str, bible: StoryBible) -> dict[str, Any]:
        """Where the story physically STOPPED at the end of ``text``.

        Cast members present in the closing stretch, and the literal
        closing lines — the anchor the continuation must pick up
        mid-flow instead of re-staging the story from its beginning.
        """
        tail = (text or "")[-1800:]
        present: list[str] = []
        for c in bible.characters:
            forms = [c.name] + ([c.name.split()[0]]
                                if len(c.name.split()) > 1 else [])
            if any(re.search(r"\b" + re.escape(f) + r"\b", tail)
                   for f in forms if len(f) > 2):
                present.append(c.name)
        paras = [p.strip() for p in re.split(r"\n\s*\n", text or "")
                 if p.strip() and not p.strip().startswith("#")]
        last_lines = re.sub(r"\s+", " ", paras[-1]).strip()[:700] \
            if paras else ""
        return {"present": present[:8], "last_lines": last_lines}

    def _write_one(self, slug: str, title: str, bible: StoryBible,
                   chapter_no: int, *, words: int,
                   direction: str, takes: int = 1,
                   final: int = 0) -> dict[str, Any]:
        # immediate continuity: tails of the last three chapters — the
        # story's actual last chapters (source tail, then any chapters
        # Devon already wrote beyond it).  The final chapter rides
        # near-full: its scene setup (who is present, where, mid-action)
        # is exactly what a starved tail used to cut away.
        tails: list[str] = []
        anchor_text = ""
        for n in (chapter_no - 3, chapter_no - 2, chapter_no - 1):
            if n < 1:
                continue
            try:
                text_n = self._any_chapter_text(slug, n, final)
            except ReaderError:
                continue
            if n == chapter_no - 1:
                anchor_text = text_n
                tails.append(_tail(text_n, 8000))
            else:
                tails.append(_tail(text_n, 3000))
        scene = self._final_scene_anchor(anchor_text, bible) \
            if anchor_text else None
        candidates: list[str] = []
        for take_i in range(max(1, takes)):
            text = self._model_continuation(
                bible, chapter_no, tails, words, direction,
                temperature=0.7 + 0.1 * take_i, scene=scene)
            if not text:
                text = self._compose_continuation(
                    bible, chapter_no, tails, words, direction,
                    seed_salt=take_i, scene=scene)
            if text:
                candidates.append(text)
        if not candidates:
            raise ContinuationError(
                f"could not compose chapter {chapter_no} for {slug!r}")
        scored = [(self._score_take(t, bible, tails), t) for t in candidates]
        scored.sort(key=lambda s: s[0], reverse=True)
        text = scored[0][1]
        alt_takes = [
            {"label": f"take-{i + 2}", "text": t,
             "words": count_words(t), "score": round(s, 3)}
            for i, (s, t) in enumerate(scored[1:])]
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
        # alternate takes live next to the chapter, pickable later
        for alt in alt_takes:
            alt_path = out_dir / f"chapter-{chapter_no:05d}.{alt['label']}.md"
            alt_path.write_text(
                f"# {ch_title} ({alt['label']})\n\n{text and '' or ''}"
                f"*alternate take — score {alt['score']}*\n\n{alt['text']}\n",
                encoding="utf-8")
        from .ai_tells import scan_ai_tells
        return {"number": chapter_no, "title": ch_title,
                "words": count_words(text), "path": path.as_posix(),
                "text": text, "takes": alt_takes,
                "take_scores": [round(s, 3) for s, _ in scored],
                # voice drift vs the bible's target, on every chapter
                "voice_drift": self.validate_voice(text, bible),
                # AI-tell scan (fiction.validate's machinery): counts +
                # locations, reported, never blocking
                "ai_tells": scan_ai_tells(text)}

    def _assemble_prompt(self, bible: StoryBible, chapter_no: int,
                         tails: list[str], words: int,
                         direction: str,
                         scene: dict[str, Any] | None = None
                         ) -> tuple[str, str]:
        """Build (system, user) prompt in the canonical AI-Dungeon order.

        AI Instructions → Plot Essentials (always-on) → Story Cards
        (keyed lorebook hits) → Story Summary → timeline + recent arcs →
        hot threads (memory) → current situation (cast states, places) →
        voice target → History (tails) → final-scene anchor →
        Author's Note (voice lock) → the ask → buffer.
        """
        recent = "\n\n".join(tails)
        # 1. plot essentials: always-on lorebook entries only
        essentials = bible.inject("", max_chars=1200)
        # 2. story cards: keyed hits from the recent tail
        cards = bible.inject(recent, max_chars=1500)
        # 3. rolling summary
        summary = (bible.arc_summary or "").strip()[:2400]
        # 4. memory: hottest open threads + the most recently touched
        open_t = bible.open_threads()
        hot = sorted(open_t, key=lambda t: t.heat, reverse=True)[:4]
        seen_ids = {t.id for t in hot}
        for t in sorted(open_t, key=lambda t: t.last_seen,
                        reverse=True)[:2]:
            if t.id not in seen_ids and len(hot) < 5:
                hot.append(t)
                seen_ids.add(t.id)
        memory = "\n".join(f"- {t.summary[:180]}" for t in hot)
        system = (
            "You are the original author's ghostwriter. You continue the "
            "story seamlessly — the reader must not feel a change of hand. "
            "Complete prose only, no meta-commentary, no summary in place "
            "of scenes, no moralizing.")
        parts = []
        if essentials:
            parts.append(f"PLOT ESSENTIALS (always true):\n{essentials}")
        if cards and cards != essentials:
            parts.append(f"WORLD LORE (relevant now):\n{cards}")
        if summary:
            parts.append(f"STORY SO FAR:\n{summary}")
        # timeline position + the latest arcs — the story's memory of
        # where it has actually reached, not only how it began
        timeline = []
        if bible.chapters_digested:
            timeline.append(
                f"TIMELINE: this story has run {bible.chapters_digested} "
                f"chapters. You are writing chapter {chapter_no} — deep "
                f"in the story, NOT a new beginning.")
        if bible.recaps:
            timeline.append("RECENT ARCS (what actually happened "
                            "lately):")
            timeline += [f"  - ch.{r.start}–{r.end}: {r.text[:260]}"
                         for r in bible.recaps[-3:]]
        if timeline:
            parts.append("\n".join(timeline))
        if memory:
            parts.append(f"OPEN THREADS (advance, don't drop):\n{memory}")
        # current situation: where everyone is NOW — latest recorded
        # character states (scene cast first), the closing beat of the
        # last digested chapter, and the latest places
        situation: list[str] = []
        present = list((scene or {}).get("present") or [])
        stated = [c for c in bible.characters if c.state]
        stated.sort(key=lambda c: (c.name not in present,
                                   -c.state_chapter))
        for c in stated[:6]:
            situation.append(
                f"  - {c.name} — {c.state[:220]} "
                f"(as of ch.{c.state_chapter})")
        if bible.current_situation:
            situation.append(f"  {bible.current_situation[:420]}")
        if bible.locations:
            recent_places = sorted(bible.locations,
                                   key=lambda x: x.last_seen,
                                   reverse=True)[:3]
            situation.append("  Places lately: " + "; ".join(
                f"{x.name} (ch.{x.last_seen})" for x in recent_places))
        if situation:
            parts.append("CURRENT SITUATION — where everyone is NOW "
                         "(these facts outrank anything older):\n"
                         + "\n".join(situation))
        # voice-target lines: the measured voice steers generation
        voice_lines = (bible.voice_target()
                       if hasattr(bible, "voice_target") else [])
        if voice_lines:
            parts.append("\n".join(voice_lines))
        parts.append(f"RECENT STORY:\n{recent[-14000:] or '(beginning)'}")
        # final-scene anchor: the exact moment to pick up, with the
        # anti-restart instruction stated as law, not a hint
        if scene and (scene.get("last_lines") or scene.get("present")):
            on_stage = ", ".join(scene.get("present") or []) \
                or "the cast of the recent story"
            parts.append(
                f"WHERE THE STORY STOPPED — end of chapter "
                f"{chapter_no - 1}. On stage: {on_stage}. The final "
                f"lines were:\n\u201c{scene.get('last_lines', '')}\u201d\n"
                f"Continue from that exact moment, mid-flow. Do NOT "
                f"recap what has already happened. Do NOT re-introduce "
                f"established characters or explain who anyone is — "
                f"the reader has read all {chapter_no - 1} chapters. "
                f"Do NOT re-stage, echo, or re-write earlier scenes.")
        # author's note: short, by position — right before the ask
        parts.append(
            f"[Author's note: write in {bible.pov}-person, {bible.tense} "
            f"tense; tone: {', '.join(bible.tone[:3]) or 'as established'}. "
            f"Voice: {(bible.voice_notes or 'as established')[:200]}]")
        parts.append(
            f"Write CHAPTER {chapter_no} continuing RIGHT where the recent "
            f"story left off — begin inside the ongoing action, not with "
            f"scene-setting or backstory. Same voice, characters behaving "
            f"like themselves as they are NOW, open threads advanced, "
            f"world rules obeyed. "
            f"~{words} words of real prose with dialogue and action.")
        if direction:
            parts.append(f"Story direction for this chapter: {direction}")
        return system, "\n\n".join(parts)

    @staticmethod
    def _score_take(text: str, bible: StoryBible,
                    tails: list[str]) -> float:
        """Rank alternate takes: voice-match + hook + freshness."""
        low = text.lower()
        score = 0.5
        # voice lock: tense consistency
        past_hits = sum(low.count(w) for w in
                        (" was ", " were ", " had ", " said "))
        pres_hits = sum(low.count(w) for w in
                        (" is ", " are ", " has ", " says "))
        if bible.tense == "past" and past_hits >= pres_hits:
            score += 0.15
        elif bible.tense == "present" and pres_hits >= past_hits:
            score += 0.15
        # hook strength: ends on tension
        last = text.strip().split("\n")[-1].strip()
        if last.endswith(("?", "!", "…", "...")):
            score += 0.15
        if re.search(r"\b(suddenly|too late|behind them|the door|"
                     r"footsteps|a scream|darkness)\b", last.lower()):
            score += 0.1
        # freshness: penalize 5-gram overlap with recent tails
        def _grams(s: str) -> set[str]:
            words = re.findall(r"[a-z']+", s.lower())
            return {" ".join(words[i:i + 5]) for i in
                    range(max(0, len(words) - 5))}
        tail_grams: set[str] = set()
        for t in tails:
            tail_grams |= _grams(t[-2000:])
        own = _grams(text)
        if own:
            overlap = len(own & tail_grams) / len(own)
            score -= min(0.3, overlap * 2)
        # dialogue presence
        if text.count('"') >= 4 or "“" in text:
            score += 0.1
        return round(score, 3)

    def _update_summary(self, bible: StoryBible, chapter_no: int,
                        text: str) -> None:
        """Rolling story summary: compress the new chapter into the bible.

        Auto-maintained (AI-Dungeon Auto-Summary pattern) — manual edits to
        arc_summary feed back into later summaries, so this appends short
        recaps and trims the oldest when the budget is exceeded.
        """
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) > 40]
        recap = " ".join(sents[:2])[:280]
        if not recap:
            return
        entry = f"[ch.{chapter_no}] {recap}"
        parts = [p for p in (bible.arc_summary or "").split(" // ")
                 if p.strip()]
        parts.append(entry)
        # keep the last ~8 recaps (budget); drop oldest first
        bible.arc_summary = " // ".join(parts[-8:])

    # -- model path -----------------------------------------------------------
    def _model_continuation(self, bible: StoryBible, chapter_no: int,
                            tails: list[str], words: int,
                            direction: str, temperature: float = 0.75,
                            scene: dict[str, Any] | None = None) -> str:
        try:
            from ..llm.base import Message, SamplingParams
            from ..llm.brain import brain_for
            from .write import model_available
        except Exception:  # noqa: BLE001
            return ""
        if not model_available(self.context):
            return ""
        system, user = self._assemble_prompt(bible, chapter_no, tails,
                                             words, direction, scene=scene)
        try:
            response = brain_for(self.context).chat(
                [Message.system(system), Message.user(user)],
                SamplingParams(temperature=temperature,
                               max_tokens=min(12000, int(words * 1.8) + 400)),
                task_kind="creative")
            text = (getattr(response, "text", "") or "").strip()
            if getattr(response, "ok", False) and count_words(text) >= 150:
                return text
        except Exception as exc:  # noqa: BLE001
            _log.debug("model continuation failed: %s", exc)
        return ""
    # -- heuristic composer (the floor) ---------------------------------------
    #: Voice texture sentences.  Rotated WITHOUT replacement inside one
    #: chapter: the old composer drew them with replacement, so a
    #: chapter could repeat "Somewhere, water kept the time nobody
    #: else would keep" three times and the same mid-beat three times.
    _TEXTURES = (
        "The air had the particular weight of evening.",
        "The air had the particular weight of a held breath.",
        "The air had the particular weight of the hour before dawn.",
        "Somewhere, water kept the time nobody else would keep.",
        "Somewhere, a distant engine kept the time nobody else would keep.",
        "Somewhere, insects kept the time nobody else would keep.",
        "Somewhere, a bell kept the time nobody else would keep.",
        "{name} had learned young that hesitation got people buried.",
        "{name} had learned young that mercy got people buried.",
        "{name} had learned young that noise got people buried.",
    )

    @staticmethod
    def _closing_facts(last_lines: str, present: list[str]
                       ) -> tuple[list[str], list[tuple[str, str]]]:
        """Concrete facts from the closing text: time pressure and held
        objects.

        Only NARRATIVE sentences (never quoted dialogue) and only short
        fact phrases — the continuation writes NEW sentences around
        them; it never echoes the closing lines.  Returns (time nouns,
        [(holder, object phrase)]): e.g. (["hour"], [("Dren",
        "a silvered blade")]).
        """
        time_nouns: list[str] = []
        held: list[tuple[str, str]] = []
        for sent in re.split(r"(?<=[.!?])\s+", last_lines or ""):
            sent = sent.strip()
            if not sent or sent[0] in "\"“":
                continue
            low = sent.lower()
            for noun in ("hour", "dawn", "dusk", "midnight"):
                if re.search(r"\b" + noun + r"\b", low) \
                        and noun not in time_nouns:
                    time_nouns.append(noun)
            m = re.search(r"\bwith\s+(a|an|the)\s+([a-z]+(?:\s+[a-z]+){0,1})\b",
                          low)
            if m:
                holder = next(
                    (n for n in present
                     if re.search(r"\b" + re.escape(n) + r"\b", sent)), "")
                if holder:
                    obj = f"{m.group(1)} {m.group(2)}"
                    if (holder, obj) not in held:
                        held.append((holder, obj))
        return time_nouns[:2], held[:2]

    @staticmethod
    def _last_speaker(last_lines: str, present: list[str]) -> str:
        """Who spoke last at the close — the voice the chapter answers.

        The last on-stage name mentioned in the closing text: in the
        repro story that is Dren ("One hour left on the third night,"
        Dren said), the one who issued the ultimatum.
        """
        best = ""
        best_pos = -1
        for name in present:
            pos = (last_lines or "").rfind(name)
            if pos > best_pos:
                best_pos = pos
                best = name
        return best

    @staticmethod
    def _state_fact(sentence: str, name: str) -> str:
        """Turn a recorded state sentence into a NEW sentence carrying
        the same fact forward — never the old sentence echoed.

        The segment is scored: +2 for a state marker, +1 for naming
        the character, -2 when its subject is a pronoun (They/He/She)
        — a "They fled" segment belongs to whoever fled, not to a
        character named later in the sentence.  The winning segment is
        rebuilt from the character's name on: "Mara stood behind
        Quinn, bleeding from a cut on her shoulder" -> "Mara was
        still bleeding from a cut on her shoulder."; "At the north
        gate ... Quinn lost his left hand to a silvered axe" ->
        "Quinn had lost his left hand to a silvered axe."; "They fled
        to the Blackspire chapel, where Dren was said to wait" ->
        "Dren was said to wait."  Returns "" when no clean rebuild
        exists — a skipped beat beats broken or misattributed prose.
        """
        first = name.split()[0]
        segments = [s.strip() for s in re.split(r"[,;]", sentence or "")
                    if s.strip()]
        best = ""
        best_score = 0
        for seg in segments:
            low = seg.lower()
            score = 0
            if any(m in low for m in _STATE_MARKERS):
                score += 2
            if re.search(r"\b" + re.escape(first) + r"\b", seg):
                score += 1
            if re.match(r"(They|He|She)\s", seg):
                score -= 2
            if score > best_score:
                best_score = score
                best = seg
        if not best or best_score <= 0:
            return ""
        # rebuild from the character's name on
        m = re.search(r"\b" + re.escape(first) + r"\b", best)
        if m is not None:
            rest = best[m.end():].strip().rstrip(".")
            if not rest or rest.startswith("'s"):
                return ""
            if rest.split()[0].lower() in (
                    "at", "in", "on", "to", "from", "with", "by", "for",
                    "through", "into", "after", "before", "under",
                    "over", "against", "between", "within"):
                return ""
            if rest.split()[0].lower() in (
                    "was", "were", "is", "are", "had", "has", "been"):
                return f"{name} {rest}."
            return f"{name} had {rest}."
        if re.match(r"[a-z]+ing\b", best):
            return f"{name} was still {best.rstrip('.')}."
        return ""

    @staticmethod
    def _thread_first_clause(summary: str) -> str:
        """The quest's imperative core: "Drink the blood of the
        Seventh Elder. Time remaining: three nights. Failure: the
        thirst claims Mara" -> "Drink the blood of the Seventh Elder";
        "Drink the blood of the Seventh Elder before the third night
        ends, or the thirst will take Mara instead" -> the same."""
        first = summary.split(". ")[0]
        first = first.split(",")[0]
        first = first.split(" before ")[0]
        first = first.split("Failure:")[0]
        first = first.strip().strip(".")
        return first[:120] if first else summary[:120]

    @staticmethod
    def _thread_consequence(summary: str) -> str:
        """The stakes clause: "..., or the thirst will take Mara
        instead" -> "the thirst will take Mara instead"; "Failure:
        the thirst claims Mara" -> "the thirst claims Mara"."""
        if ", or " in summary:
            return summary.split(", or ", 1)[1].strip().strip('."“”')
        if "Failure:" in summary:
            return summary.split("Failure:", 1)[1].strip().strip('."“”')
        return ""

    def _compose_continuation(self, bible: StoryBible, chapter_no: int,
                              tails: list[str], words: int,
                              direction: str, seed_salt: int = 0,
                              scene: dict[str, Any] | None = None) -> str:
        """Real prose from the bible: the chapter CONTINUES the digest.

        Every paragraph carries a fact the digest actually measured —
        the hottest thread with its deadline and stakes, the on-stage
        cast's recorded states, the place the story stopped, an object
        held at the close — so the chapter reads as the next chapter
        of THIS story, not a new story wearing its names.  The old
        composer slotted names into fixed templates and never used the
        scene anchor's content, the character states, or the thread
        specifics: its chapter 4 of the repro story mentioned neither
        Dren, nor the chapel, nor the silvered blade, nor the third
        night, opened "Mara to catch his breath", and repeated the
        same paragraph three times while padding.  Deterministic per
        chapter number (seeded with a stable hash, so re-runs are
        stable across processes too); ``seed_salt`` varies alternate
        takes.  No beat is ever repeated inside one chapter: textures
        and padding beats rotate without replacement, and padding
        stops when the facts run out rather than looping.
        """
        seed = int.from_bytes(hashlib.md5(
            f"{bible.story_slug}|{chapter_no}|{seed_salt}".encode("utf-8")
        ).digest()[:4], "little")
        rng = random.Random(seed)
        threads = bible.open_threads()
        thread = max(threads, key=lambda t: t.heat) if threads else None
        chars = bible.characters
        protag = next((c for c in chars if c.role == "protagonist"), None)
        if protag is None and chars:
            protag = chars[0]
        present = list((scene or {}).get("present") or [])
        lead = present[0] if present else (
            protag.name if protag else "the protagonist")
        by_name = {c.name: c for c in chars}
        places = sorted(bible.locations, key=lambda x: x.last_seen,
                        reverse=True)
        place = places[0].name if places else ""
        last_lines = (scene or {}).get("last_lines") or ""
        time_nouns, held = self._closing_facts(last_lines, present)
        speaker = self._last_speaker(last_lines, present)
        opponent = ""
        if speaker and speaker != lead:
            opponent = speaker
        elif len(present) > 1:
            opponent = present[1]
        textures = list(self._TEXTURES)
        rng.shuffle(textures)

        def _with_texture(para: str) -> str:
            if textures and rng.random() < 0.5:
                return para + " " + textures.pop().format(name=lead)
            return para

        paras: list[str] = []
        # 1. cold open — pick up where the story stopped: the lead, the
        # place, the live task with its deadline and stakes, the time
        # pressure from the closing scene.
        place_bit = f" at {place}" if place else ""
        if thread:
            open_text = (f"{lead} had not moved{place_bit}. "
                         f"The task was plain: {thread.summary.rstrip('.')}.")
        else:
            open_text = f"{lead} had not moved{place_bit}."
        if time_nouns:
            open_text += f" The {time_nouns[0]} was nearly gone."
        paras.append(_with_texture(open_text))
        # 2. state beats — the on-stage cast as they ARE now.
        for name in present[:3]:
            char = by_name.get(name)
            if char is None or not char.state:
                continue
            fact = self._state_fact(char.state, name)
            if fact:
                paras.append(_with_texture(fact))
        # 3. held objects — what the closing scene put in someone's hands.
        for holder, obj in held:
            paras.append(_with_texture(f"{holder} still had {obj}."))
        # 4. dialogue — the thread spoken aloud by the one who issued
        # the ultimatum at the close, else by the lead.
        if thread:
            clause = self._thread_first_clause(thread.summary)
            who = (speaker or lead).split()[0]
            line = clause[0].upper() + clause[1:] if clause else clause
            paras.append(f"\"{line}.\" — {who}")
        # 5. action turn — the lead closes on the opposing force.
        if opponent:
            paras.append(_with_texture(
                f"{lead} closed the distance to {opponent}."))
        else:
            paras.append(_with_texture(
                f"{lead} moved before doubt could finish its sentence — "
                f"{self._action(rng, lead)}."))
        # 6. complication — the opposing force answers.
        if opponent and thread:
            clause = self._thread_first_clause(thread.summary)
            lowered = clause[0].lower() + clause[1:] if clause else clause
            paras.append(_with_texture(
                f"But {opponent} had been waiting — of course {opponent} "
                f"had been waiting — and {lowered} would not come cheap."))
        else:
            paras.append(_with_texture(self._complication(
                rng, lead, opponent or None, thread)))
        # 7. hook — end on the stakes.
        consequence = self._thread_consequence(thread.summary) \
            if thread else ""
        if consequence:
            paras.append(f"One way or another, {consequence}.")
        else:
            paras.append(_with_texture(self._hook(rng, lead, thread)))
        if direction:
            paras.insert(3, direction.rstrip(".") + ".")
        # pad toward the word target with unused fact-beats, never repeats
        text = "\n\n".join(paras)
        padding: list[str] = []
        for other in threads[1:3]:
            padding.append(
                f"{lead} had not forgotten: {other.summary.rstrip('.')}.")
        for rule in bible.world_rules[:2]:
            padding.append(
                f"The old rule held, as it always did: {rule.rstrip('.')}.")
        for entry in bible.lore:
            if entry.kind == "rule" and entry.text:
                if any(entry.text[:60].lower()
                       in r.lower() or r[:60].lower()
                       in entry.text.lower()
                       for r in bible.world_rules):
                    continue
                padding.append(
                    "The old rule held, as it always did: "
                    f"{entry.text.rstrip('.')[:200]}.")
                break
        for beat in padding:
            if count_words(text) >= words * 0.5:
                break
            text = text + "\n\n" + _with_texture(beat)
        return text

    @staticmethod
    def _action(rng: random.Random, p_name: str) -> str:
        return rng.choice([
            "across the broken ground toward the sound",
            "hand closing around the hilt of their blade",
            "into the dark with nothing but momentum and spite",
            "toward the light, because standing still had stopped being an option",
            "with the old precision, every step a decision",
        ])

    def _complication(self, rng: random.Random, p_name: str,
                      d_name: str | None, thread: Any) -> str:
        who = d_name or "the shadows"
        if thread is not None:
            clause = self._thread_first_clause(thread.summary)
            goal = clause[0].lower() + clause[1:] if clause else clause
        else:
            goal = "what had to be done"
        return rng.choice([
            f"But {who} had been waiting — of course {who} had been waiting — "
            f"and {goal} would not come cheap.",
            f"The plan survived exactly until contact. Then {who} moved, and "
            f"{p_name} understood, too late, the shape of the trap.",
            f"What {p_name} had not counted on was {who}: a variable with "
            f"teeth, and {goal} hanging in the balance.",
        ])

    def _hook(self, rng: random.Random, p_name: str, thread: Any) -> str:
        if thread is not None:
            clause = self._thread_first_clause(thread.summary)
            goal = clause[0].lower() + clause[1:] if clause else clause
        else:
            goal = "the truth"
        return rng.choice([
            f"And in the distance, something answered — a sound {p_name} "
            f"recognized, and wished they did not.",
            f"{p_name} smiled then, the thin dangerous smile of someone who "
            f"has just realized {goal} was never the real prize.",
            f"The chapter of hesitation was over. What came next would not "
            f"ask permission.",
        ])
