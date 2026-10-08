"""EPUB → audiobook, one click, with store-compliant AI disclosure.

``epub_to_audiobook(epub, voice_cast, target_stores)`` → chapter split
(from the EPUB TOC or heading structure) → per-chapter TTS (private
XTTS stack for the user's own books; multi-voice casting optional) →
LUFS master (Auphonic pattern via ffmpeg ``loudnorm``) → packaged
audiobook with AI-disclosure metadata attached per target store.

Disclosure rules are a versioned config (``DISCLOSURE_RULES``); they
are re-checked before every publish run because store rules shift.
Every method never raises and refuses rather than producing fake
audio or wrong metadata.

Pairs with #44 (creator voice): pass the author's cloned voice ref
as the narrator and the audiobook speaks in their voice.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("nomorals.audio.audiobook")

# ── disclosure rules (versioned — checked before every publish) ────────────

DISCLOSURE_RULES: dict[str, Any] = {
    "version": "2026-10",
    "checked": "2026-10-08",
    # ACX: AI narration must be disclosed at distribution; some rights
    # holders exclude AI narration — the metadata flags the review need.
    "acx": {
        "requires_disclosure": True,
        "disclosure_text": "This audiobook was narrated with synthetic voice technology.",
        "field": "ai_narration_disclosure",
        "note": "ACX requires AI-narration disclosure at title setup.",
    },
    # Spotify (Findaway Voices): free production, non-exclusive, AI
    # disclosure checkbox at upload.
    "spotify": {
        "requires_disclosure": True,
        "disclosure_text": "AI-generated narration.",
        "field": "ai_generated",
        "note": "Spotify/Findaway requires the AI-narration checkbox at upload.",
    },
    # Kobo (Kobo Writing Life audiobooks): AI narration disclosure in
    # the title metadata.
    "kobo": {
        "requires_disclosure": True,
        "disclosure_text": "Narrated using text-to-speech technology.",
        "field": "ai_narration",
        "note": "Kobo requires AI-narration disclosure in title metadata.",
    },
}

KNOWN_STORES = tuple(s for s in DISCLOSURE_RULES if s not in ("version", "checked"))

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9'’]+")


# ── model ──────────────────────────────────────────────────────────────────


@dataclass
class BookChapter:
    """One chapter of the audiobook."""

    index: int = 0
    title: str = ""
    text: str = ""
    audio_path: str = ""
    duration_s: float = 0.0
    voice: str = "narrator"


@dataclass
class Audiobook:
    """A produced audiobook package."""

    book_id: str = ""
    title: str = ""
    author: str = ""
    chapters: list[BookChapter] = field(default_factory=list)
    master_path: str = ""
    duration_s: float = 0.0
    lufs: float = -16.0
    disclosure: dict[str, dict[str, str]] = field(default_factory=dict)
    target_stores: list[str] = field(default_factory=list)
    created_at: float = 0.0


# ── chapter split ──────────────────────────────────────────────────────────


def split_chapters(epub_path: str, *,
                   parse_fn: Callable[[str], Any] | None = None
                   ) -> list[BookChapter]:
    """EPUB → chapters (TOC/heading structure, else length-based split).

    Never raises; returns [] when the book cannot be parsed.
    """
    try:
        path = Path(epub_path or "")
        if not path.exists():
            return []
        if parse_fn is not None:
            doc = parse_fn(str(path))
        else:
            from ..documents import parse_path
            doc = parse_path(str(path))
        sections = getattr(doc, "sections", None) or []
        chapters: list[BookChapter] = []
        idx = 0
        for sec in sections:
            heading = (getattr(sec, "heading", "") or "").strip()
            text = (getattr(sec, "text", "") or "").strip()
            if not text:
                continue
            level = int(getattr(sec, "level", 2) or 2)
            if level <= 1 or len(chapters) == 0 and not heading:
                # new chapter at every h1; first section starts chapter 0
                if heading or not chapters:
                    idx += 1
                    chapters.append(BookChapter(
                        index=idx, title=heading or f"Chapter {idx}",
                        text=text))
                elif chapters:
                    chapters[-1].text += "\n\n" + text
            else:
                chapters[-1].text += ("\n\n" + heading + "\n" if heading else "\n\n") + text
        if not chapters:
            # fallback: length-based split of full text
            full = ""
            try:
                from ..documents.model import full_text
                full = full_text(doc) or ""
            except Exception:  # noqa: BLE001
                pass
            if not full.strip():
                return []
            words = full.split()
            size = 2500
            for i in range(0, len(words), size):
                idx += 1
                chunk = " ".join(words[i:i + size])
                chapters.append(BookChapter(index=idx, title=f"Part {idx}",
                                            text=chunk))
        return [c for c in chapters if c.text.strip()]
    except Exception as exc:  # noqa: BLE001
        _log.warning("split_chapters failed: %s", exc)
        return []


def _book_title_author(epub_path: str, *,
                       parse_fn: Callable[[str], Any] | None = None
                       ) -> tuple[str, str]:
    try:
        if parse_fn is not None:
            doc = parse_fn(epub_path)
        else:
            from ..documents import parse_path
            doc = parse_path(epub_path)
        meta = getattr(doc, "metadata", None) or {}
        title = str(meta.get("title", "") or "").strip() or \
            Path(epub_path).stem.replace("_", " ")
        author = str(meta.get("author", "") or meta.get("creator", "") or
                     "").strip()
        return title, author
    except Exception:  # noqa: BLE001
        return Path(epub_path).stem.replace("_", " "), ""


# ── TTS ────────────────────────────────────────────────────────────────────


def _default_tts_fn(text: str, voice_ref: str, *,
                    lang: str = "en") -> str | None:
    """Speak ``text`` in the cloned voice at ``voice_ref`` (private XTTS).

    Returns the WAV path, or None (never fake audio) when anything is
    missing — the caller refuses the chapter rather than faking it.
    """
    try:
        if not voice_ref or not Path(voice_ref).exists():
            return None
        from ..voice.tts import UniversalTTS
        tmp = tempfile.mkdtemp(prefix="audiobook-tts-")
        tts = UniversalTTS(backend="auto", voices_dir=tmp, audience="private")
        voice_name = "audiobook-narrator"
        tts.voices.upload_voice(voice_name, voice_ref, language=lang or "en")
        # chunk long text so no single TTS call explodes
        chunks: list[str] = []
        words = text.split()
        size = 400
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i:i + size]))
        pieces: list[str] = []
        for chunk in chunks:
            out = tts.speak(chunk, voice_name=voice_name, audience="private")
            p = (out or {}).get("path", "")
            if not p or not Path(p).exists():
                return None
            pieces.append(p)
        if len(pieces) == 1:
            return pieces[0]
        from ..media_edit.videos import concat
        res = concat(pieces, out_dir=tmp, suffix="chapter", ext=".wav")
        out = (res or {}).get("output", "") if isinstance(res, dict) else ""
        return out if out and Path(out).exists() else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("audiobook TTS failed: %s", exc)
        return None


def _audio_duration(p: Path) -> float:
    """Duration in seconds via ffprobe. Never raises."""
    try:
        if shutil.which("ffprobe") is None:
            return 0.0
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, timeout=30)
        return float((out.stdout or "").strip() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def master_lufs(wav_path: str, *, target_lufs: float = -16.0,
                out_dir: str | None = None) -> str | None:
    """LUFS-normalize a WAV (Auphonic pattern) via ffmpeg loudnorm.

    Returns the mastered path, or None when ffmpeg is unavailable.
    Never raises.
    """
    try:
        p = Path(wav_path or "")
        if not p.exists():
            return None
        if shutil.which("ffmpeg") is None:
            return None
        tmp = Path(out_dir or tempfile.mkdtemp(prefix="audiobook-master-"))
        out = tmp / (p.stem + f"-lufs{int(target_lufs)}.wav")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(p),
               "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
               "-ar", "44100", "-ac", "2", str(out)]
        subprocess.run(cmd, capture_output=True, timeout=600)
        return str(out) if out.exists() else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("master_lufs failed: %s", exc)
        return None


# ── the one-click pipeline ─────────────────────────────────────────────────


def epub_to_audiobook(epub_path: str, voice_cast: dict[str, str] | None = None,
                      target_stores: list[str] | None = None, *,
                      lang: str = "en",
                      tts_fn: Callable[..., str | None] | None = None,
                      parse_fn: Callable[[str], Any] | None = None,
                      out_dir: str | None = None,
                      store: "AudiobookStore | None" = None,
                      ) -> dict[str, Any]:
    """EPUB in → chaptered, mastered audiobook out.

    ``voice_cast`` maps ``{"narrator": <voice ref path>}`` (the #44
    creator-voice pairing: the author's own cloned voice); optional
    per-chapter voices via ``{"chapter:<n>": ref}``.

    Returns ``{"ok", "audiobook", "note"}`` or ``{"ok": False,
    "reason"}``. Never raises; refuses rather than faking audio.
    """
    try:
        path = Path(epub_path or "")
        if not path.exists():
            return {"ok": False, "reason": "no EPUB found — nothing to read"}
        chapters = split_chapters(str(path), parse_fn=parse_fn)
        if not chapters:
            return {"ok": False,
                    "reason": "could not read any chapters from the EPUB"}
        cast = dict(voice_cast or {})
        narrator_ref = cast.get("narrator", "")
        if not narrator_ref:
            return {"ok": False,
                    "reason": "no narrator voice — pass voice_cast={'narrator': <voice reference audio>}"}
        stores = [s.strip().lower() for s in (target_stores or [])
                  if s and s.strip().lower() in KNOWN_STORES]
        if not stores:
            stores = ["spotify"]  # sensible default target
        title, author = _book_title_author(str(path), parse_fn=parse_fn)
        work = Path(out_dir or tempfile.mkdtemp(prefix="audiobook-"))
        rendered: list[BookChapter] = []
        for ch in chapters:
            ref = cast.get(f"chapter:{ch.index}", narrator_ref)
            wav = None
            if tts_fn is not None:
                try:
                    wav = tts_fn(ch.text, ref, lang=lang)
                except TypeError:
                    wav = tts_fn(ch.text, ref)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("tts_fn failed for chapter %d: %s",
                                 ch.index, exc)
                    wav = None
            else:
                wav = _default_tts_fn(ch.text, ref, lang=lang)
            if not wav or not Path(wav).exists():
                return {"ok": False,
                        "reason": f"no TTS audio for chapter {ch.index} "
                                  f"({ch.title}) — stopping rather than "
                                  "shipping a broken audiobook"}
            from .edit import _normalize_wav
            norm = _normalize_wav(wav, 24000, work)
            ch.audio_path = norm or wav
            ch.duration_s = _audio_duration(Path(ch.audio_path))
            rendered.append(ch)
        # concat chapters → master
        from ..media_edit.videos import concat
        if len(rendered) == 1:
            combined = rendered[0].audio_path
        else:
            res = concat([c.audio_path for c in rendered],
                         out_dir=str(work), suffix="audiobook",
                         ext=".wav")
            combined = (res or {}).get("output", "") \
                if isinstance(res, dict) else ""
        if not combined or not Path(combined).exists():
            return {"ok": False, "reason": "chapter concat failed"}
        mastered = master_lufs(combined, out_dir=str(work))
        if not mastered:
            return {"ok": False,
                    "reason": "LUFS mastering failed (ffmpeg unavailable?)"}
        disclosure = {
            s: {"text": DISCLOSURE_RULES[s]["disclosure_text"],
                "field": DISCLOSURE_RULES[s]["field"],
                "note": DISCLOSURE_RULES[s]["note"],
                "rules_version": DISCLOSURE_RULES["version"]}
            for s in stores
        }
        book = Audiobook(
            book_id="ab_" + uuid.uuid4().hex[:8],
            title=title, author=author, chapters=rendered,
            master_path=mastered,
            duration_s=_audio_duration(Path(mastered)),
            disclosure=disclosure, target_stores=stores,
            created_at=time.time())
        if store is not None:
            try:
                store.save(book)
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "audiobook": book,
                "note": f"{len(rendered)} chapters, "
                        f"{book.duration_s / 3600:.1f}h, mastered to "
                        f"-16 LUFS, disclosure attached for "
                        f"{', '.join(stores)} (rules v{DISCLOSURE_RULES['version']})"}
    except Exception as exc:  # noqa: BLE001
        _log.warning("epub_to_audiobook failed: %s", exc)
        return {"ok": False, "reason": f"audiobook build failed: {exc}"}


# ── store ──────────────────────────────────────────────────────────────────


def _default_db() -> str:
    base = Path(os.environ.get("NOMORALS_HOME", Path.home() / ".nomorals"))
    base = base / "audio" / "audiobooks"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return str(base / "audiobooks.db")


class AudiobookStore:
    """SQLite persistence for audiobook runs. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        try:
            p = Path(db_path or _default_db())
            self._db = sqlite3.connect(str(p), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS audiobooks(
                book_id TEXT PRIMARY KEY, title TEXT, author TEXT,
                master_path TEXT, duration_s REAL, target_stores TEXT,
                disclosure TEXT, created_at REAL)""")
            self._db.commit()
        except Exception as exc:  # noqa: BLE001
            _log.warning("AudiobookStore init failed: %s", exc)
            self._db = None

    def save(self, book: Audiobook) -> bool:
        try:
            if self._db is None or not book or not book.book_id:
                return False
            import json
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO audiobooks VALUES (?,?,?,?,?,?,?,?)",
                    (book.book_id, book.title, book.author, book.master_path,
                     book.duration_s, ",".join(book.target_stores),
                     json.dumps(book.disclosure), book.created_at))
                self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT book_id, title, author, master_path, duration_s,"
                " target_stores, created_at FROM audiobooks"
                " ORDER BY created_at DESC LIMIT ?", (max(1, limit),))
            return [dict(r) for r in rows.fetchall()]
        except Exception:  # noqa: BLE001
            return []

    def get(self, book_id: str) -> dict[str, Any] | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM audiobooks WHERE book_id = ?",
                ((book_id or "").strip(),)).fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
            return None


# ── chat ───────────────────────────────────────────────────────────────────


def _get_store() -> AudiobookStore:
    return AudiobookStore()


def _usage() -> str:
    return ("🎧 /audiobook make <epub> [voices...] [stores...] — EPUB in, "
            "chaptered mastered audiobook out.\n"
            "🎧 /audiobook status — list produced audiobooks.\n"
            "Voices: narrator=<ref audio> [chapter:2=<ref>] · stores: acx, "
            "spotify, kobo. AI disclosure is attached automatically "
            f"(rules v{DISCLOSURE_RULES['version']}).")


def control_audiobook(tail: str, context=None, chat=None, **kwargs) -> str:
    """Chat entry point. Owner-only (wired at dispatch). Never raises."""
    try:
        tail = (tail or "").strip()
        store = _get_store()
        if not tail or tail.lower() in ("help", "?"):
            return _usage()
        low = tail.lower()
        if low.startswith("status"):
            books = store.list()
            if not books:
                return "🎧 no audiobooks produced yet."
            lines = ["🎧 audiobooks:"]
            for b in books:
                hrs = (b.get("duration_s") or 0) / 3600
                lines.append(f"• {b.get('title','?')} — {hrs:.1f}h "
                             f"→ {b.get('target_stores','')} "
                             f"({b.get('book_id','')})")
            return "\n".join(lines)
        if low.startswith("make "):
            rest = tail[5:].strip()
            parts = rest.split()
            epub = parts[0] if parts else ""
            cast: dict[str, str] = {}
            stores: list[str] = []
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip().lower()
                    if k in ("narrator",) or k.startswith("chapter:"):
                        cast[k] = v.strip()
                elif p.strip().lower() in KNOWN_STORES:
                    stores.append(p.strip().lower())
            res = epub_to_audiobook(epub, cast or None, stores or None,
                                    store=store)
            if res.get("ok"):
                book: Audiobook = res["audiobook"]
                return (f"🎧 audiobook ready: {book.title}\n"
                        f"📖 {len(book.chapters)} chapters · "
                        f"⏱️ {book.duration_s / 3600:.1f}h · mastered "
                        f"-16 LUFS\n📦 {res['note']}")
            return f"🎧 couldn't make the audiobook — {res.get('reason', '?')}"
        return _usage()
    except Exception as exc:  # noqa: BLE001
        return f"🎧 audiobook hiccup: {exc}"
