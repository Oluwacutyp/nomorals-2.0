"""Voice notes + the voice trust layer (build-map #82).

Hinge's data point: voice lifts date-conversion 41%; 65% of singles say
voice helps them assess interest. For Devon: freelancer voice intros
(#81's ``voice_intro``), community voice notes (#12), gig-messaging
voice notes (#1), business voice greetings (#73) — in the full language
scope: English, Pidgin, Ekiti/Ilawe Ekiti Yoruba (flagship), standard
Yoruba, Hausa, Igbo.

Design rules:
- **Audio for tone, transcript for search.** Every voice note keeps its
  audio file and (when STT is available) a transcript that is indexed,
  so "what did she say about the price?" is answerable.
- **STT is a production seam, not a dependency.** Pass an ``stt``
  callable ``(audio_path, language) -> dict`` — in production that's
  ``nomorals.voice.stt.UniversalSTT``; in tests a mock. ``None`` means
  Devon stores the audio honestly without a transcript.
- **Stack routing per #30.** Public surfaces synthesize with Chatterbox
  (MIT, streaming-capable); private surfaces with XTTS (best cloning,
  non-commercial). The ``tts`` seam is ``(text, language) -> path``.
- Never raises: a broken note is a missing note, not a crash.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "VOICE_LANGUAGES",
    "VOICE_LANGUAGE_NAMES",
    "STACK_PUBLIC",
    "STACK_PRIVATE",
    "VoiceNote",
    "VoiceNoteStore",
    "normalize_voice_language",
]

#: The voice language scope. ``yo-ekiti`` is the Ekiti/Ilawe Ekiti
#: flagship; plain ``yo`` is standard Yoruba. Both map to the
#: underlying model's Yoruba handling, but the code travels with the
#: note so the agent answers in the right register.
VOICE_LANGUAGES = ("en", "pcm", "yo", "yo-ekiti", "ha", "ig")

VOICE_LANGUAGE_NAMES = {
    "en": "English",
    "pcm": "Pidgin",
    "yo": "Yoruba",
    "yo-ekiti": "Yoruba (Ekiti/Ilawe Ekiti)",
    "ha": "Hausa",
    "ig": "Igbo",
}

#: TTS stack per #30 — the public/private split, by license.
STACK_PUBLIC = "chatterbox"    # MIT — community bot, shared surfaces
STACK_PRIVATE = "xtts"         # best cloning — owner's private Devon

_DEFAULT_DB = os.path.expanduser("~/.nomorals/social/voice_notes.db")
_DEFAULT_VAULT = os.path.expanduser("~/.nomorals/social/voice_notes")


def normalize_voice_language(lang: str | None) -> str:
    """Map a language tag into the voice scope. Unknown → 'en'."""
    l = (lang or "en").strip().lower().replace("_", "-")
    if l in VOICE_LANGUAGES:
        return l
    # Friendly aliases.
    aliases = {
        "english": "en", "pidgin": "pcm", "nigerian-pidgin": "pcm",
        "yoruba": "yo", "ekiti": "yo-ekiti", "ilawe": "yo-ekiti",
        "hausa": "ha", "igbo": "ig",
    }
    return aliases.get(l, "en")


@dataclass
class VoiceNote:
    """One voice note in a chat: audio kept, transcript searchable."""

    note_id: str = ""
    chat_id: str = ""           # community chat / gig thread / DM
    sender: str = ""
    audio_path: str = ""        # vault copy — playable
    transcript: str = ""        # STT text; "" when STT wasn't available
    language: str = "en"
    stt_backend: str = ""       # which STT produced the transcript
    created_at: float = 0.0


class VoiceNoteStore:
    """SQLite voice-note storage + STT indexing. Never raises."""

    def __init__(self, db_path: str = "",
                 vault_dir: str = "",
                 stt: object = None,
                 tts: object = None) -> None:
        #: ``stt(audio_path, language) -> {"text": ..., "backend": ...}``
        self._stt = stt
        #: ``tts(text, language) -> audio_path`` (a factory producing audio)
        self._tts = tts
        self._vault = vault_dir or _DEFAULT_VAULT
        self._db_path = db_path or _DEFAULT_DB
        try:
            os.makedirs(os.path.dirname(db_path or _DEFAULT_DB),
                        exist_ok=True)
            os.makedirs(self._vault, exist_ok=True)
            self._db = sqlite3.connect(db_path or _DEFAULT_DB)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS voice_notes (
                       note_id TEXT PRIMARY KEY, chat_id TEXT,
                       sender TEXT, audio_path TEXT, transcript TEXT,
                       language TEXT, stt_backend TEXT, created_at REAL)""")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS vn_chat ON voice_notes(chat_id)")
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty store
            _log.warning("voice_notes: db unavailable, running empty",
                         exc_info=True)
            self._db = None

    # — sending —

    def send_voice_note(self, chat_id: str, sender: str, audio_path: str,
                        *, language: str = "en") -> VoiceNote | None:
        """Store a voice note: audio for tone, transcript for search.

        Copies the audio into Devon's vault, runs STT when the seam is
        wired (empty transcript + honest flag when it isn't). None when
        invalid/down — never raises.
        """
        if self._db is None:
            return None
        if not (chat_id or "").strip():
            return None
        try:
            from .profiles import AUDIO_EXTENSIONS
        except Exception:  # noqa: BLE001
            AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".oga", ".m4a",
                                ".opus", ".aac", ".flac", ".wma")
        if not audio_path or not os.path.isfile(audio_path):
            return None
        if os.path.splitext(audio_path)[1].lower() not in AUDIO_EXTENSIONS:
            return None
        lang = normalize_voice_language(language)
        try:
            import shutil
            ext = os.path.splitext(audio_path)[1].lower()
            note_id = "vn_" + uuid.uuid4().hex[:8]
            dest = os.path.join(self._vault, f"{note_id}{ext}")
            shutil.copy2(audio_path, dest)
        except Exception:  # noqa: BLE001
            _log.debug("voice_notes: vault copy failed", exc_info=True)
            return None
        transcript, backend = "", ""
        if self._stt is not None:
            try:
                out = self._stt(dest, lang) or {}
                transcript = str(out.get("text", "") or "").strip()
                backend = str(out.get("backend", "") or "").strip()
            except Exception:  # noqa: BLE001 — STT failure isn't fatal
                _log.debug("voice_notes: STT failed", exc_info=True)
        try:
            now = time.time()
            self._db.execute(
                """INSERT INTO voice_notes
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (note_id, chat_id.strip(), (sender or "").strip(), dest,
                 transcript, lang, backend, now))
            self._db.commit()
            return VoiceNote(note_id=note_id, chat_id=chat_id.strip(),
                             sender=(sender or "").strip(),
                             audio_path=dest, transcript=transcript,
                             language=lang, stt_backend=backend,
                             created_at=now)
        except Exception:  # noqa: BLE001
            _log.debug("voice_notes: insert failed", exc_info=True)
            return None

    def stats(self, chat_id: str = "") -> dict[str, object]:
        """Voice-note activity: counts, languages, transcript coverage.

        Never raises. Powers the "how much voice is this community"
        answer without dumping every note.
        """
        try:
            if self._db is None:
                return {}
            where = "WHERE chat_id = ?" if chat_id else ""
            params: tuple = (chat_id,) if chat_id else ()
            total = self._db.execute(
                f"SELECT COUNT(*) AS n FROM voice_notes {where}", params).fetchone()
            with_text = self._db.execute(
                f"SELECT COUNT(*) AS n FROM voice_notes {where}"
                + (" AND " if where else " WHERE ")
                + "transcript != ''",
                params).fetchone()
            langs = self._db.execute(
                f"SELECT language, COUNT(*) AS n FROM voice_notes {where} "
                f"GROUP BY language ORDER BY n DESC LIMIT 8", params).fetchall()
            n_total = int(total["n"] or 0)
            n_text = int(with_text["n"] or 0)
            return {
                "total": n_total,
                "transcribed": n_text,
                "transcript_coverage": round(n_text / n_total, 2) if n_total else 0.0,
                "languages": {r["language"]: int(r["n"]) for r in langs},
            }
        except Exception:  # noqa: BLE001
            return {}

    def get(self, note_id: str) -> VoiceNote | None:
        """One note by id. None when missing/down."""
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT * FROM voice_notes WHERE note_id = ?",
                (note_id,)).fetchone()
            return self._row(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def list(self, chat_id: str = "", limit: int = 50) -> list[VoiceNote]:
        """Notes in a chat (or everywhere), newest last."""
        if self._db is None:
            return []
        try:
            q = "SELECT * FROM voice_notes"
            args: tuple = ()
            if chat_id:
                q += " WHERE chat_id = ?"
                args = (chat_id,)
            q += " ORDER BY created_at LIMIT ?"
            return [self._row(r) for r in
                    self._db.execute(q, args + (max(1, int(limit or 50)),))]
        except Exception:  # noqa: BLE001
            return []

    # — search —

    def search_voice_notes(self, query: str,
                           chat_id: str = "") -> list[VoiceNote]:
        """Find notes by their transcripts — 'what did she say about X?'"""
        if self._db is None:
            return []
        q = (query or "").strip().lower()
        if not q:
            return []
        try:
            sql = "SELECT * FROM voice_notes WHERE LOWER(transcript) LIKE ?"
            args: tuple = (f"%{q}%",)
            if chat_id:
                sql += " AND chat_id = ?"
                args += (chat_id,)
            sql += " ORDER BY created_at"
            return [self._row(r) for r in self._db.execute(sql, args)]
        except Exception:  # noqa: BLE001
            return []

    # — TTS routing (#30) —

    @staticmethod
    def stack_for(public: bool = True) -> str:
        """Which TTS stack for this surface: chatterbox vs xtts."""
        return STACK_PUBLIC if public else STACK_PRIVATE

    def synthesize_preview(self, text: str, *, public: bool = True,
                           language: str = "en") -> str | None:
        """Render text to speech, routed by surface.

        Returns the audio path, or None when no TTS seam is wired —
        never a fabricated path.
        """
        if self._tts is None:
            return None
        t = (text or "").strip()
        if not t:
            return None
        try:
            out = self._tts(t, normalize_voice_language(language),
                            stack=self.stack_for(public))
            path = str(out or "").strip()
            return path if path and os.path.isfile(path) else None
        except Exception:  # noqa: BLE001
            _log.debug("voice_notes: TTS preview failed", exc_info=True)
            return None

    # — internals —

    @staticmethod
    def _row(r: sqlite3.Row) -> VoiceNote:
        return VoiceNote(
            note_id=r["note_id"], chat_id=r["chat_id"],
            sender=r["sender"], audio_path=r["audio_path"],
            transcript=r["transcript"] or "",
            language=r["language"] or "en",
            stt_backend=r["stt_backend"] or "",
            created_at=r["created_at"] or 0.0)


# ── chat control ──────────────────────────────────────────────────────────

def _usage() -> str:
    return (
        "🎙️ /vnote — voice notes in chat. Audio for tone, transcript for search.\n"
        "  /vnote send <chat_id> <audio_path> [language] — send a voice note\n"
        "  /vnote search <query> [chat_id] — find by what was said\n"
        "  /vnote play <note_id> — where the audio lives\n"
        "  /vnote preview <public|private> <language> <text> — TTS preview\n"
        "  languages: en, pcm (Pidgin), yo (Yoruba), yo-ekiti (Ekiti),\n"
        "             ha (Hausa), ig (Igbo)"
    )


def control_vnote(tail: str, context=None, chat=None,
                  sender_id: str = "", sender: str = "") -> str:
    """/vnote — voice notes. Never raises."""
    try:
        rest = (tail or "").strip()
        store = getattr(context, "voice_note_store", None) \
            if context is not None else None
        s = store if isinstance(store, VoiceNoteStore) else VoiceNoteStore()
        parts = rest.split()
        if not parts or parts[0].lower() in ("help", "?"):
            return _usage()
        cmd = parts[0].lower()

        if cmd == "send":
            if len(parts) < 3:
                return "usage: /vnote send <chat_id> <audio_path> [language]"
            lang = parts[3] if len(parts) > 3 else "en"
            stt = getattr(context, "stt", None) if context else None
            real = s if stt is None else VoiceNoteStore(
                db_path=s._db_path, vault_dir=s._vault, stt=stt)
            n = real.send_voice_note(parts[1], sender or sender_id or "anon",
                                     parts[2], language=lang)
            if n is None:
                return ("couldn't send — check the chat id and that the "
                        "file is real audio (wav/mp3/ogg/m4a/opus).")
            note = (f"\n  📝 “{n.transcript[:160]}”" if n.transcript
                    else "\n  (audio saved; transcription not available yet)")
            return f"🎙️ voice note sent [{n.note_id}, {n.language}].{note}"

        if cmd == "search":
            if len(parts) < 2:
                return "usage: /vnote search <query> [chat_id]"
            chat_id = parts[-1] if len(parts) > 2 else ""
            hits = s.search_voice_notes(" ".join(parts[1:-1] or parts[1:]),
                                        chat_id)
            if not hits:
                return "no voice notes match that — try different words."
            lines = [f"🎙️ {len(hits)} match(es):"]
            for h in hits[-10:]:
                snippet = (h.transcript[:80] + "…"
                           if len(h.transcript) > 80 else h.transcript)
                lines.append(f"  [{h.note_id}] {h.sender} ({h.language}): "
                             f"{snippet or '(no transcript)'}")
            return "\n".join(lines)

        if cmd == "play":
            if len(parts) < 2:
                return "usage: /vnote play <note_id>"
            n = s.get(parts[1])
            if n is None:
                return "no voice note with that id."
            line = f"🎙️ [{n.note_id}] {n.sender} → {n.chat_id}: {n.audio_path}"
            if n.transcript:
                line += f"\n  📝 “{n.transcript[:160]}”"
            return line

        if cmd == "preview":
            if len(parts) < 4:
                return ("usage: /vnote preview <public|private> <language> "
                        "<text>")
            public = parts[1].lower() != "private"
            stack = VoiceNoteStore.stack_for(public)
            path = s.synthesize_preview(" ".join(parts[3:]),
                                        public=public, language=parts[2])
            if path is None:
                return (f"no TTS wired up right now — would have rendered "
                        f"with {stack}.")
            return f"🔊 preview ready ({stack}): {path}"

        return _usage()
    except Exception:  # noqa: BLE001 — chat never sees a traceback
        _log.debug("vnote control failed", exc_info=True)
        return "voice note command hit a snag — try again."
