"""Interactive Audio Overviews — the podcast you can talk to.

NotebookLM Audio Overview + Interactive Mode, Devon-native: the
research pipeline's source-grounded session + the dual voice stack,
no Google needed.

Flow::

    ov = make_overview(
        [{"title": "Lagos rents", "text": "..."}, ...],
        format="deep-dive",
        script_fn=my_llm,          # injectable
        voice_fn=my_two_voice_tts, # injectable (distinct cloned voices)
        lang="yo-ekiti",
    )
    # ov.audio_path — the discussion as audio, with chapter markers.

    session = InteractiveSession(ov, llm_fn=my_llm, voice_fn=my_tts)
    # user interrupts mid-playback with a voice note:
    answer = session.ask("what was the cheapest area again?")
    # answer["audio"] — spoken from the SAME sources, grounded.

Rules:

* GROUNDED OR REFUSED.  Script lines must carry [S1]/[S2] source markers.
  Lines without citations are dropped; if fewer than half survive, the
  overview is refused rather than confabulated.  Interactive answers go
  through the research pipeline's ``GroundedSession`` — CANNOT_ANSWER
  becomes an honest refusal, never a hallucination.
* TWO DISTINCT VOICES.  The hosts are spoken by two separate cloned
  voices (private XTTS stack).  One voice_fn call per speaker.
* FULL LANGUAGE SCOPE.  Ekiti/Ilawe Ekiti flagship plus Hausa, Igbo,
  Pidgin, standard Yoruba, English — via the same alias map as
  :mod:`nomorals.audio.edit`.
* Never raises.  No LLM → no script (honest).  No TTS → script without
  audio (honest).  No sources → refused.

Chat (owner-only): /overview make <format> <sources> | list | ask <q>
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .edit import _LANG_ALIASES

_log = get_logger(__name__)

__all__ = [
    "FORMATS", "HOSTS",
    "ScriptLine", "Chapter", "OverviewScript", "AudioOverview",
    "OverviewStore", "InteractiveSession",
    "make_overview", "control_overview",
    "suggest_questions", "export_markdown",
]

#: script formats
FORMATS = ("deep-dive", "debate", "brief", "interview")

#: the two hosts — distinct voices, distinct display names
HOSTS = ("A", "B")
_HOST_NAMES = {"A": "Adaeze", "B": "Tunde"}

#: minimum share of cited lines before we ship the script
_MIN_CITED_SHARE = 0.5

#: [S1], [S2], ... citation markers the script LLM must emit
_CITE_RE = re.compile(r"\[S(\d+)\]")

#: script line header: "A: ..." or "B: ..."
_LINE_RE = re.compile(r"^\s*([AB])\s*[:\-–—]\s*(.+)$", re.DOTALL)

#: "CHAPTER: title" chapter markers
_CHAPTER_RE = re.compile(r"^\s*chapter\s*[:\-–—]\s*(.+)$", re.IGNORECASE)


def _norm_lang(lang: str = "en") -> str:
    key = (lang or "en").strip().lower()
    return _LANG_ALIASES.get(key, key)


# ── script model ──────────────────────────────────────────────────────────


@dataclass
class ScriptLine:
    """One spoken line. ``cites`` = source indices backing the claims."""
    speaker: str          # "A" or "B"
    text: str
    cites: list[int] = field(default_factory=list)
    chapter: str = ""

    @property
    def cited(self) -> bool:
        return bool(self.cites)


@dataclass
class Chapter:
    title: str
    start_s: float = 0.0


@dataclass
class OverviewScript:
    format: str
    lang: str
    lines: list[ScriptLine] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    dropped: int = 0       # uncited lines removed before shipping

    @property
    def cited_share(self) -> float:
        total = len(self.lines) + self.dropped
        return (len(self.lines) / total) if total else 0.0


@dataclass
class AudioOverview:
    overview_id: str
    title: str
    format: str
    lang: str
    script: OverviewScript
    sources: list[dict[str, str]] = field(default_factory=list)
    audio_path: str = ""
    created_at: float = 0.0

    def has_audio(self) -> bool:
        return bool(self.audio_path) and Path(self.audio_path).exists()


# ── script generation ─────────────────────────────────────────────────────


def _script_prompt(sources: list[dict[str, str]], fmt: str,
                   lang: str) -> str:
    labeled = []
    for i, s in enumerate(sources, 1):
        title = (s.get("title") or f"source {i}").strip()
        text = (s.get("text") or "").strip()
        labeled.append(f"[S{i}] {title}:\n{text[:4000]}")
    persona = {
        "deep-dive": ("a relaxed deep-dive discussion", "Adaeze and Tunde explore the material together, building on each other's points"),
        "debate": ("a friendly debate", "Adaeze and Tunde take opposing angles on the material and push back on each other"),
        "brief": ("a short briefing", "Adaeze gives the key points, Tunde asks the sharp follow-ups"),
        "interview": ("an interview", "Tunde interviews Adaeze as the domain expert: sharp questions, concrete examples, no jargon without explanation"),
    }.get(fmt, ("a discussion", "Adaeze and Tunde discuss the material"))
    return (
        f"You are writing {persona[0]} of the sources below as a two-host podcast script.\n"
        f"{persona[1]}. Keep it natural, spoken English"
        + (f" — the discussion is in {lang}" if lang != "en" else "")
        + ".\n\nRULES:\n"
        "1. Every factual claim MUST carry a source marker like [S1], [S2]. "
        "Lines with no factual content (greetings, transitions) may omit markers.\n"
        "2. Never invent facts, numbers, or quotes not in the sources.\n"
        "3. Format: each line starts with 'A: ' or 'B: '. "
        "Mark major sections with a line 'CHAPTER: <title>'.\n"
        "4. 12–20 lines total.\n\n"
        "SOURCES:\n\n" + "\n\n".join(labeled)
    )


def _parse_script(raw: str) -> list[ScriptLine]:
    """Parse 'A: ...' / 'B: ...' / 'CHAPTER: ...' into lines."""
    lines: list[ScriptLine] = []
    chapter = ""
    for raw_line in (raw or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        ch = _CHAPTER_RE.match(line)
        if ch:
            chapter = ch.group(1).strip()[:80]
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue  # stage directions, numbering — not spoken
        speaker, text = m.group(1).upper(), m.group(2).strip()
        if speaker not in HOSTS or not text:
            continue
        cites = sorted({int(n) for n in _CITE_RE.findall(text)})
        lines.append(ScriptLine(speaker=speaker, text=text,
                               cites=cites, chapter=chapter))
    return lines


def _ground_script(lines: list[ScriptLine],
                   n_sources: int) -> OverviewScript | None:
    """Drop uncited lines; refuse if fewer than half survive."""
    kept: list[ScriptLine] = []
    dropped = 0
    for ln in lines:
        valid = [c for c in ln.cites if 1 <= c <= n_sources]
        if valid:
            ln.cites = valid
            kept.append(ln)
        else:
            dropped += 1
    if not kept or len(kept) / (len(kept) + dropped) < _MIN_CITED_SHARE:
        return None
    chapters: list[Chapter] = []
    seen = set()
    for ln in kept:
        if ln.chapter and ln.chapter not in seen:
            seen.add(ln.chapter)
            chapters.append(Chapter(title=ln.chapter))
    return OverviewScript(format="", lang="", lines=kept,
                          chapters=chapters, dropped=dropped)


# ── two-voice TTS ─────────────────────────────────────────────────────────


def _default_voice_fn(speaker: str, text: str, *,
                      lang: str = "en",
                      voice_refs: dict[str, str] | None = None) -> str | None:
    """Speak ``text`` as host ``speaker`` with a distinct cloned voice.

    PRIVATE stack only (XTTS v2 zero-shot).  Requires reference audio per
    host via ``voice_refs={"A": path, "B": path}``; returns None (never
    fake audio) when anything is missing.
    """
    try:
        ref = (voice_refs or {}).get(speaker, "")
        if not ref or not Path(ref).exists():
            return None
        from ..voice.tts import UniversalTTS
        tmp = tempfile.mkdtemp(prefix="overview-voice-")
        tts = UniversalTTS(backend="auto", voices_dir=tmp, audience="private")
        voice_name = f"overview-host-{speaker.lower()}"
        tts.voices.upload_voice(voice_name, ref, language=lang or "en")
        out = tts.speak(text, voice_name=voice_name, audience="private")
        path = (out or {}).get("path", "")
        return path if path and Path(path).exists() else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("overview TTS failed for host %s: %s", speaker, exc)
        return None


def _render_audio(script: OverviewScript, *,
                  lang: str,
                  voice_fn: Callable[..., str | None] | None = None,
                  voice_refs: dict[str, str] | None = None,
                  out_dir: str | os.PathLike[str] | None = None
                  ) -> dict[str, Any]:
    """Render the script to one audio file; returns {ok, path, chapters}."""
    try:
        pieces: list[str] = []
        line_durs: list[float] = []   # measured duration per script line
        tmp = Path(tempfile.mkdtemp(prefix="overview-render-"))
        for i, ln in enumerate(script.lines):
            clean = _CITE_RE.sub("", ln.text).strip()
            if not clean:
                line_durs.append(0.0)
                continue
            wav = None
            if voice_fn is not None:
                try:
                    wav = voice_fn(ln.speaker, clean, lang=lang,
                                   voice_refs=voice_refs)
                except TypeError:
                    wav = voice_fn(ln.speaker, clean)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("voice_fn failed: %s", exc)
                    wav = None
            else:
                wav = _default_voice_fn(ln.speaker, clean, lang=lang,
                                        voice_refs=voice_refs)
            if not wav or not Path(wav).exists():
                return {"ok": False,
                        "reason": f"no TTS audio for host {ln.speaker} "
                                  "(line %d) — script kept, audio not rendered" % (i + 1)}
            from .edit import _normalize_wav
            norm = _normalize_wav(wav, 24000, tmp)
            if norm:
                pieces.append(norm)
                line_durs.append(_audio_duration(Path(norm)) or 0.0)
            else:
                line_durs.append(0.0)
        if not pieces:
            return {"ok": False, "reason": "no audio pieces rendered"}
        from ..media_edit.videos import concat
        out = concat(pieces, out_dir=out_dir or str(tmp), suffix="overview",
                     ext=".wav")
        if not out or not Path(out).exists():
            return {"ok": False, "reason": "ffmpeg concat failed"}
        # chapter timestamps: measured from each rendered line's real
        # audio duration (not word-proportional estimates)
        total = sum(line_durs)
        chapters: list[Chapter] = []
        elapsed = 0.0
        seen: set[str] = set()
        for ln, d in zip(script.lines, line_durs):
            if ln.chapter and ln.chapter not in seen:
                seen.add(ln.chapter)
                chapters.append(Chapter(title=ln.chapter, start_s=elapsed))
            elapsed += d
        if total <= 0:
            # measurement unavailable — fall back to word-proportional
            words = [len(_CITE_RE.sub("", ln.text).split())
                     for ln in script.lines]
            total_words = sum(words) or 1
            dur = _audio_duration(Path(out)) or sum(words) * 0.4
            chapters = []
            elapsed = 0.0
            seen = set()
            for ln, w in zip(script.lines, words):
                if ln.chapter and ln.chapter not in seen:
                    seen.add(ln.chapter)
                    chapters.append(Chapter(title=ln.chapter, start_s=elapsed))
                elapsed += dur * w / total_words
        return {"ok": True, "path": str(out), "chapters": chapters}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"render failed: {exc}"}


def _audio_duration(p: Path) -> float:
    try:
        import shutil, subprocess
        if shutil.which("ffprobe") is None:
            return 0.0
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, timeout=30)
        return float((out.stdout or "").strip() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


# ── store ─────────────────────────────────────────────────────────────────


def _default_db() -> str:
    d = Path.home() / ".nomorals" / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "overviews.db")


class OverviewStore:
    """SQLite persistence for overviews. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS overviews(
                    id TEXT PRIMARY KEY, title TEXT, format TEXT, lang TEXT,
                    script_json TEXT, sources_json TEXT, audio_path TEXT,
                    created_at REAL)""")
            self._db.commit()
        except Exception as exc:  # noqa: BLE001
            _log.warning("overview store unavailable: %s", exc)
            self._db = None

    # -- persistence helpers ------------------------------------------
    @staticmethod
    def _script_to_json(script: OverviewScript) -> str:
        import json
        return json.dumps({
            "format": script.format, "lang": script.lang,
            "dropped": script.dropped,
            "chapters": [{"title": c.title, "start_s": c.start_s}
                         for c in script.chapters],
            "lines": [{"speaker": ln.speaker, "text": ln.text,
                       "cites": ln.cites, "chapter": ln.chapter}
                      for ln in script.lines],
        })

    @staticmethod
    def _script_from_json(raw: str) -> OverviewScript:
        import json
        d = json.loads(raw or "{}")
        return OverviewScript(
            format=d.get("format", ""), lang=d.get("lang", ""),
            dropped=int(d.get("dropped", 0)),
            chapters=[Chapter(title=c.get("title", ""),
                              start_s=float(c.get("start_s", 0)))
                      for c in d.get("chapters", [])],
            lines=[ScriptLine(speaker=ln.get("speaker", "A"),
                              text=ln.get("text", ""),
                              cites=list(ln.get("cites", [])),
                              chapter=ln.get("chapter", ""))
                   for ln in d.get("lines", [])],
        )

    def save(self, ov: AudioOverview) -> bool:
        try:
            if self._db is None:
                return False
            import json
            self._db.execute(
                "INSERT OR REPLACE INTO overviews VALUES (?,?,?,?,?,?,?,?)",
                (ov.overview_id, ov.title, ov.format, ov.lang,
                 self._script_to_json(ov.script),
                 json.dumps(ov.sources), ov.audio_path, ov.created_at))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def get(self, overview_id: str) -> AudioOverview | None:
        try:
            if self._db is None:
                return None
            import json
            row = self._db.execute(
                "SELECT * FROM overviews WHERE id = ?",
                (overview_id,)).fetchone()
            if not row:
                return None
            return AudioOverview(
                overview_id=row["id"], title=row["title"],
                format=row["format"], lang=row["lang"],
                script=self._script_from_json(row["script_json"]),
                sources=json.loads(row["sources_json"] or "[]"),
                audio_path=row["audio_path"] or "",
                created_at=row["created_at"])
        except Exception:  # noqa: BLE001
            return None

    def list(self, limit: int = 20) -> list[AudioOverview]:
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT id FROM overviews ORDER BY created_at DESC LIMIT ?",
                (int(limit) or 20,)).fetchall()
            return [o for o in (self.get(r["id"]) for r in rows) if o]
        except Exception:  # noqa: BLE001
            return []

    def latest(self) -> AudioOverview | None:
        items = self.list(1)
        return items[0] if items else None


# ── make_overview ─────────────────────────────────────────────────────────


def make_overview(sources: list[dict[str, Any]] | None, fmt: str = "deep-dive",
                  *, title: str = "",
                  lang: str = "en",
                  script_fn: Callable[[str], str] | None = None,
                  voice_fn: Callable[..., str | None] | None = None,
                  voice_refs: dict[str, str] | None = None,
                  render_audio: bool = True,
                  store: OverviewStore | None = None,
                  ) -> dict[str, Any]:
    """Sources → grounded two-host script → two-voice audio.

    Returns ``{"ok", "overview", "note"}`` or ``{"ok": False, "reason"}``.
    Never raises; refuses rather than confabulates.
    """
    try:
        lang = _norm_lang(lang)
        fmt = (fmt or "deep-dive").strip().lower()
        if fmt not in FORMATS:
            return {"ok": False,
                    "reason": f"unknown format {fmt!r} — pick: {', '.join(FORMATS)}"}
        srcs = [{"title": str(s.get("title", "") or f"source {i + 1}"),
                 "text": str(s.get("text", "") or "")}
                for i, s in enumerate(sources or [])
                if isinstance(s, dict) and (s.get("text") or "").strip()]
        if not srcs:
            return {"ok": False,
                    "reason": "no sources — I can't discuss material I haven't seen. "
                              "Give me notes, documents, or pasted text first."}
        if script_fn is None:
            return {"ok": False,
                    "reason": "no script generator available (pass script_fn) — "
                              "refusing to invent a discussion without a model"}
        try:
            raw = script_fn(_script_prompt(srcs, fmt, lang))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"script generation failed: {exc}"}
        lines = _parse_script(raw)
        if not lines:
            return {"ok": False,
                    "reason": "the model returned no usable dialogue lines — "
                              "try again with a different prompt"}
        script = _ground_script(lines, len(srcs))
        if script is None:
            return {"ok": False,
                    "reason": "too few lines carried source citations "
                              f"({len(lines)} parsed) — refusing to ship an "
                              "ungrounded discussion"}
        script.format, script.lang = fmt, lang
        ov = AudioOverview(
            overview_id="ov_" + uuid.uuid4().hex[:8],
            title=title or f"{fmt} overview",
            format=fmt, lang=lang, script=script,
            sources=srcs, created_at=time.time())
        note = (f"{len(script.lines)} lines, {script.dropped} uncited dropped, "
                f"{len(script.chapters)} chapters")
        if render_audio:
            res = _render_audio(script, lang=lang, voice_fn=voice_fn,
                                voice_refs=voice_refs)
            if res.get("ok"):
                ov.audio_path = res["path"]
                ov.script.chapters = res.get("chapters") or ov.script.chapters
                note += f" — audio rendered: {ov.audio_path}"
            else:
                note += f" — audio not rendered ({res.get('reason')})"
        else:
            note += " — audio rendering skipped"
        if store is not None:
            store.save(ov)
        return {"ok": True, "overview": ov, "note": note}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"overview failed: {exc}"}


# ── interactive mode ──────────────────────────────────────────────────────


def suggest_questions(overview: AudioOverview, *,
                      llm_fn: Callable[[str], str] | None = None,
                      n: int = 5) -> list[str]:
    """Suggest sharp follow-up questions for the interactive session
    (the NotebookLM "what to ask next" pattern).

    With ``llm_fn`` the suggestions come from the model, grounded in
    the script; without one, honest structural fallbacks are built from
    the chapter titles. Never raises.
    """
    try:
        if overview is None or not overview.script.lines:
            return []
        chapters = [c.title for c in overview.script.chapters if c.title]
        topics = chapters[:6] or [
            ln.text[:60] for ln in overview.script.lines[:6]]
        if llm_fn is None:
            out = []
            for t in topics[:n]:
                out.append(f"Can you give a concrete example of {t}?")
        else:
            script_txt = "\n".join(
                f"{ln.speaker}: {_CITE_RE.sub('', ln.text).strip()}"
                for ln in overview.script.lines[:40])
            prompt = (
                "You are helping a listener of a two-host podcast discussion. "
                "Based on the script excerpt below, suggest %d sharp, specific "
                "follow-up questions the listener could ask the hosts to go "
                "deeper. Each question must be answerable from the material. "
                "One question per line, no numbering, no extra text.\n\n%s"
                % (n, script_txt[:4000]))
            raw = (llm_fn(prompt) or "").strip()
            out = [q.strip(" -•\t") for q in raw.splitlines() if q.strip()]
            if not out:
                out = [f"Can you give a concrete example of {t}?"
                       for t in topics[:n]]
        # pad to n with grounded generic follow-ups
        generics = ["What surprised you most in this discussion?",
                    "How does this connect to the bigger picture?",
                    "What would you tell a skeptic about this?"]
        i = 0
        while len(out) < n and i < len(generics):
            if generics[i] not in out:
                out.append(generics[i])
            i += 1
        return out[:n]
    except Exception:  # noqa: BLE001
        return []


def export_markdown(ov: AudioOverview) -> str:
    """The overview as a readable Markdown document (script + chapters +
    sources). Never raises."""
    try:
        lines = [f"# {ov.title or 'Audio overview'}", "",
                 f"_Format: {ov.format} · Language: {ov.lang} · "
                 f"{len(ov.script.lines)} lines_"]
        if ov.script.dropped:
            lines.append(f"_{ov.script.dropped} uncited lines dropped_")
        lines.append("")
        cur_chapter = ""
        for ln in ov.script.lines:
            if ln.chapter and ln.chapter != cur_chapter:
                cur_chapter = ln.chapter
                ch_start = next((c.start_s for c in ov.script.chapters
                                 if c.title == cur_chapter), 0.0)
                m, s = divmod(ch_start, 60)
                lines.append(f"\n## {cur_chapter} "
                             f"`{int(m):02d}:{s:04.1f}`\n")
            name = _HOST_NAMES.get(ln.speaker, ln.speaker)
            lines.append(f"**{name}:** {ln.text}")
        if ov.sources:
            lines.append("\n---\n\n### Sources\n")
            for i, s in enumerate(ov.sources, 1):
                lines.append(f"{i}. **{s.get('title', f'source {i}')}**")
        return "\n".join(lines).strip() + "\n"
    except Exception:  # noqa: BLE001
        return ""


class InteractiveSession:
    """Interrupt the overview, ask a question, get a grounded spoken answer.

    Questions are answered from the SAME sources via the research
    pipeline's ``GroundedSession``.  CANNOT_ANSWER → honest refusal, never
    confabulation.  The discussion then resumes where it left off.
    """

    def __init__(self, overview: AudioOverview, *,
                 llm_fn: Callable[[str], str] | None = None,
                 voice_fn: Callable[..., str | None] | None = None,
                 lang: str = "") -> None:
        self.overview = overview
        self.lang = _norm_lang(lang or overview.lang)
        self._llm_fn = llm_fn
        self._voice_fn = voice_fn
        self._session = None
        self._history: list[dict[str, Any]] = []
        try:
            from ..research.grounded import GroundedSession
            gs = GroundedSession()
            for s in (overview.sources or []):
                gs.add_text(s.get("text", ""),
                            title=s.get("title", "source"))
            self._session = gs
        except Exception as exc:  # noqa: BLE001
            _log.warning("interactive grounded session unavailable: %s", exc)
            self._session = None

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def ask(self, question: str) -> dict[str, Any]:
        """Answer ``question`` from the overview's sources.

        Steering with continuity: prior Q&A pairs are injected into the
        session so follow-ups reference the discussion so far (the
        NotebookLM continuity pattern).

        Returns ``{"ok", "answer", "refused", "audio", "sources"}`` —
        or ``{"ok": False, "reason"}``.  Never raises.
        """
        try:
            question = (question or "").strip()
            if not question:
                return {"ok": False, "reason": "no question asked"}
            if self._session is None or self._llm_fn is None:
                return {"ok": False,
                        "reason": "interactive Q&A unavailable (no grounded "
                                  "session or LLM) — refusing to guess"}
            q = question
            if self._history:
                ctx = "\n".join(
                    f"Q: {h['question']}\nA: {h['answer'][:400]}"
                    for h in self._history[-4:])
                q = (f"Conversation so far:\n{ctx}\n\n"
                     f"Follow-up question (answer in the same grounded way, "
                     f"referencing the discussion when relevant):\n{question}")
            ans = self._session.ask(q, llm_fn=self._llm_fn)
            refused = bool(ans.refused) or "CANNOT_ANSWER" in ans.text.upper()
            text = ans.text
            if refused:
                text = ("I can't answer that from your sources — it's not "
                        "covered in the material this overview was built on.")
            audio = None
            if self._voice_fn is not None and not refused:
                try:
                    audio = self._voice_fn("A", text, lang=self.lang)
                except TypeError:
                    audio = self._voice_fn("A", text)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("interactive answer TTS failed: %s", exc)
                    audio = None
            rec = {"question": question, "answer": text, "refused": refused,
                   "audio": audio or "",
                   "sources": [s.title for s in (ans.sources or [])]}
            self._history.append(rec)
            return {"ok": True, **rec}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"Q&A failed: {exc}"}


# ── chat ──────────────────────────────────────────────────────────────────


def _get_store() -> OverviewStore:
    return OverviewStore()


_SESSIONS: dict[str, InteractiveSession] = {}


def _parse_sources(text: str) -> list[dict[str, str]]:
    """'Title :: text ;; Title2 :: text2' → source dicts."""
    out = []
    for chunk in (text or "").split(";;"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "::" in chunk:
            title, body = chunk.split("::", 1)
        else:
            title, body = "source", chunk
        if body.strip():
            out.append({"title": title.strip() or "source",
                        "text": body.strip()})
    return out


def _usage() -> str:
    return (
        "🎙️ /overview make <deep-dive|debate|brief|interview> [lang] <Title :: text ;; Title2 :: text2>\n"
        "🎙️ /overview list — saved overviews\n"
        "🎙️ /overview ask <question> — interrupt the latest overview, get a grounded answer\n"
        "🎙️ /overview questions — suggested follow-ups to ask the hosts\n"
        "🎙️ /overview export <id> — the script as Markdown\n"
        "🎙️ /overview voices — the two hosts"
    )


def control_overview(tail: str, context=None, chat=None,
                     **kwargs) -> str:
    """Chat entry: /overview …  Never raises."""
    try:
        store: OverviewStore = kwargs.get("store") or _get_store()
        tail = (tail or "").strip()
        if not tail or tail.startswith("help"):
            return _usage()
        head, _, rest = tail.partition(" ")
        head = head.lower()

        if head == "list":
            items = store.list()
            if not items:
                return "🎙️ no overviews yet — /overview make … to build one."
            lines = ["🎙️ saved overviews:"]
            for o in items:
                mark = "🔊" if o.has_audio() else "📝"
                lines.append(f"{mark} {o.overview_id} — {o.title} "
                             f"({o.format}, {o.lang}, {len(o.script.lines)} lines)")
            return "\n".join(lines)

        if head == "voices":
            return ("🎙️ the hosts: Adaeze (host A) and Tunde (host B) — "
                    "two distinct cloned voices on the private stack "
                    "(custom voices per host — beyond NotebookLM's two "
                    "fixed defaults).")

        if head == "questions":
            ov = store.latest()
            if ov is None:
                return "🎙️ no overview yet — /overview make … first."
            qs = suggest_questions(ov, llm_fn=kwargs.get("llm_fn"))
            if not qs:
                return "🎙️ no suggestions — the overview has no script."
            lines = ["🎙️ try asking the hosts:"]
            lines += [f"  {i}. {q}" for i, q in enumerate(qs, 1)]
            lines.append("  (ask with /overview ask <question>)")
            return "\n".join(lines)

        if head == "export":
            oid = rest.strip()
            if not oid:
                return "🎙️ export which? /overview export <id>"
            ov = store.get(oid)
            if ov is None:
                return f"🎙️ no overview {oid!r} — /overview list."
            return export_markdown(ov)

        if head == "make":
            # make <format> [lang] <Title :: text ;; ...>
            parts = rest.split(None, 2)
            fmt = (parts[0] if parts else "deep-dive").lower()
            lang, src_text = "en", ""
            if len(parts) == 3:
                lang, src_text = _norm_lang(parts[1]), parts[2]
            elif len(parts) == 2:
                src_text = parts[1]
            sources = _parse_sources(src_text)
            llm_fn = kwargs.get("script_fn") or kwargs.get("llm_fn")
            res = make_overview(sources, fmt, lang=lang, script_fn=llm_fn,
                                voice_fn=kwargs.get("voice_fn"),
                                voice_refs=kwargs.get("voice_refs"),
                                store=store)
            if not res.get("ok"):
                return f"🎙️ couldn't build it: {res.get('reason')}"
            ov: AudioOverview = res["overview"]
            lines = [f"🎙️ overview ready: {ov.overview_id}", res["note"]]
            for c in ov.script.chapters:
                lines.append(f"  📑 {c.title} — {c.start_s:.0f}s")
            return "\n".join(lines)

        if head == "ask":
            question = rest.strip()
            if not question:
                return "🎙️ ask what? /overview ask <your question>"
            ov = store.latest()
            if ov is None:
                return "🎙️ no overview to interrupt yet — /overview make … first."
            sess = _SESSIONS.get(ov.overview_id)
            if sess is None:
                sess = InteractiveSession(
                    ov, llm_fn=kwargs.get("llm_fn"),
                    voice_fn=kwargs.get("voice_fn"), lang=ov.lang)
                _SESSIONS[ov.overview_id] = sess
            ans = sess.ask(question)
            if not ans.get("ok"):
                return f"🎙️ {ans.get('reason')}"
            out = [f"🎙️ {ans['answer']}"]
            if ans.get("audio"):
                out.append(f"🔊 {ans['audio']}")
            out.append("— discussion resumes where it left off.")
            return "\n".join(out)

        return _usage()
    except Exception as exc:  # noqa: BLE001
        return f"🎙️ overview error: {exc}"
