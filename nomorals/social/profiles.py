"""Prompt-scaffolded profiles + element-level likes (build-map #81).

The Hinge pattern, portable to every profile surface Devon has:

1. **Prompt-scaffolded profiles** — curated prompts solve the blank-bio
   problem everywhere: freelancer profiles ("my best work was..."),
   community member cards ("ask me about..."), business listings
   ("what we do best is..."). No one stares at an empty bio box.

2. **Element-level likes** — react to a specific project/photo/answer,
   not the whole profile. "Like this project" gives the opener and the
   signal; the optional comment is the conversation starter.

Applies to (integration points):
- #1 gig marketplace — freelancer profiles (`surface="gig"`)
- #12 community — member profiles (`surface="community"`)
- #73 white-label — business listings (`surface="business"`)

``ProfileStore`` is SQLite and never raises: a broken profile lookup is
an empty result, not a crash mid-conversation.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "PROMPTS",
    "ELEMENT_TYPES",
    "SURFACES",
    "Profile",
    "ProfileElement",
    "ElementLike",
    "VoiceIntro",
    "ProfileStore",
    "suggested_prompts",
    "format_profile",
]

#: Profile surfaces Devon knows how to scaffold.
SURFACES = ("gig", "community", "business")

#: Audio formats Devon accepts for voice intros / voice notes.
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".oga", ".m4a",
                    ".opus", ".aac", ".flac", ".wma")

#: The 30-second guidance for a spoken intro (advisory, not a hard cap —
#: a great 45s intro beats a rushed 30s one, and the UI says so).
VOICE_INTRO_SECONDS = 30

#: Element types a profile can hold.
ELEMENT_TYPES = ("project", "photo", "answer", "link")

#: Curated prompts per surface. The blank-bio fix: a profile always has
#: something worth reading, and every answer is a conversation hook.
PROMPTS: dict[str, list[tuple[str, str]]] = {
    "gig": [
        ("best_work", "My best work was..."),
        ("looking_for", "I'm looking for clients who..."),
        ("ask_me", "Ask me about..."),
        ("superpower", "The thing I'm weirdly good at is..."),
    ],
    "community": [
        ("intro", "A little about me..."),
        ("ask_me", "Ask me about..."),
        ("here_for", "I'm here because..."),
        ("offer", "I can help others with..."),
    ],
    "business": [
        ("best", "What we do best is..."),
        ("customers", "Our customers usually come to us for..."),
        ("different", "What makes us different..."),
        ("ask_us", "Ask us about..."),
    ],
}

_DEFAULT_DB = os.path.expanduser("~/.nomorals/social/profiles.db")


def suggested_prompts(surface: str, answered: set[str] | None = None) -> list[tuple[str, str]]:
    """Prompts for a surface, skipping ones already answered."""
    answered = answered or set()
    return [(pid, text) for pid, text in PROMPTS.get(surface, PROMPTS["gig"])
            if pid not in answered]


# ── data ────────────────────────────────────────────────────────────────

@dataclass
class ProfileElement:
    """One likeable thing on a profile: a project, photo, answer, link."""

    element_id: str = ""
    type: str = "project"          # project | photo | answer | link
    title: str = ""
    content: str = ""
    url: str = ""
    like_count: int = 0


@dataclass
class ElementLike:
    """A like on a specific element. The comment is the opener."""

    profile_id: str = ""
    element_id: str = ""
    liker: str = ""
    comment: str = ""
    created_at: float = 0.0


@dataclass
class VoiceIntro:
    """A profile's 30-second spoken 'about me' — audio + transcript."""

    profile_id: str = ""
    audio_path: str = ""        # vault copy, playable
    transcript: str = ""        # STT text (empty when no STT was available)
    language: str = "en"        # en | pcm | yo | yo-ekiti | ha | ig
    stt_backend: str = ""       # which STT produced the transcript
    created_at: float = 0.0


@dataclass
class Profile:
    """One profile on one surface, scaffolded by prompts."""

    profile_id: str = ""
    owner: str = ""
    surface: str = "gig"           # gig | community | business
    display_name: str = ""
    prompts: dict[str, str] = field(default_factory=dict)
    elements: list[ProfileElement] = field(default_factory=list)
    voice_intro: VoiceIntro | None = None
    created_at: float = 0.0

    def completeness(self) -> dict[str, object]:
        """How finished this profile is: 0-100 score + per-part breakdown.

        Parts: display name, prompt answers (vs the surface's prompt list),
        a voice intro, and at least one element. The score drives the
        onboarding brain — it knows exactly what's missing instead of
        asking "is your profile done?" into the void.
        """
        parts: dict[str, float] = {}
        parts["display_name"] = 1.0 if (self.display_name or "").strip() else 0.0
        total_prompts = len(PROMPTS.get(self.surface, PROMPTS["gig"]))
        answered = sum(1 for v in (self.prompts or {}).values() if (v or "").strip())
        parts["prompts"] = min(1.0, answered / max(1, total_prompts))
        parts["voice_intro"] = 1.0 if self.voice_intro is not None else 0.0
        parts["elements"] = min(1.0, len(self.elements or []) / 3.0)
        weights = {"display_name": 0.15, "prompts": 0.45,
                   "voice_intro": 0.15, "elements": 0.25}
        score = round(sum(parts[p] * weights[p] for p in parts) * 100)
        return {"score": score,
                "parts": {p: round(v * 100) for p, v in parts.items()},
                "done": score >= 90}

    def missing_for_complete(self) -> list[str]:
        """Plain-language checklist of what's missing. Empty when done."""
        missing: list[str] = []
        if not (self.display_name or "").strip():
            missing.append("add a display name")
        total_prompts = len(PROMPTS.get(self.surface, PROMPTS["gig"]))
        answered = sum(1 for v in (self.prompts or {}).values() if (v or "").strip())
        if answered < total_prompts:
            missing.append(f"answer {total_prompts - answered} more prompt(s)")
        if self.voice_intro is None:
            missing.append("record a 30-second voice intro")
        if len(self.elements or []) < 3:
            missing.append(f"add {3 - len(self.elements or [])} more profile element(s)")
        return missing


# ── store ───────────────────────────────────────────────────────────────

class ProfileStore:
    """SQLite profile + like storage. Every method never raises."""

    def __init__(self, db_path: str = "") -> None:
        path = db_path or _DEFAULT_DB
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS profiles (
                       profile_id TEXT PRIMARY KEY, owner TEXT, surface TEXT,
                       display_name TEXT, prompts_json TEXT, created_at REAL)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS elements (
                       element_id TEXT PRIMARY KEY, profile_id TEXT, type TEXT,
                       title TEXT, content TEXT, url TEXT)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS likes (
                       profile_id TEXT, element_id TEXT, liker TEXT,
                       comment TEXT, created_at REAL,
                       PRIMARY KEY (profile_id, element_id, liker))""")
            self._db.commit()
            self._ensure_voice_columns()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty store
            _log.warning("profiles: db unavailable, running empty", exc_info=True)
            self._db = None

    def _ensure_voice_columns(self) -> None:
        """Voice-intro columns, added to DBs created before #82."""
        cols = {r["name"] for r in
                self._db.execute("PRAGMA table_info(profiles)")}
        for col in ("voice_intro_path TEXT",
                    "voice_intro_transcript TEXT",
                    "voice_intro_language TEXT",
                    "voice_intro_stt TEXT",
                    "voice_intro_at REAL"):
            name = col.split()[0]
            if name not in cols:
                self._db.execute(f"ALTER TABLE profiles ADD COLUMN {col}")
        self._db.commit()

    # — profiles —

    def create(self, owner: str, surface: str = "gig",
               display_name: str = "") -> Profile | None:
        """Create a profile. Returns None only when the store is down."""
        import json
        if self._db is None:
            return None
        if surface not in SURFACES:
            surface = "gig"
        p = Profile(
            profile_id="prof_" + uuid.uuid4().hex[:8],
            owner=(owner or "").strip(),
            surface=surface,
            display_name=(display_name or "").strip() or (owner or "").strip(),
            created_at=time.time(),
        )
        try:
            self._db.execute(
                """INSERT INTO profiles
                   (profile_id, owner, surface, display_name, prompts_json,
                    created_at, voice_intro_path, voice_intro_transcript,
                    voice_intro_language, voice_intro_stt, voice_intro_at)
                   VALUES (?, ?, ?, ?, ?, ?, '', '', '', '', 0)""",
                (p.profile_id, p.owner, p.surface, p.display_name,
                 json.dumps({}), p.created_at))
            self._db.commit()
            return p
        except Exception:  # noqa: BLE001
            _log.debug("profiles: create failed", exc_info=True)
            return None

    def get(self, profile_id: str) -> Profile | None:
        """Load a profile with its elements. None when missing/down."""
        import json
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT * FROM profiles WHERE profile_id = ?",
                (profile_id,)).fetchone()
            if row is None:
                return None
            p = Profile(
                profile_id=row["profile_id"], owner=row["owner"],
                surface=row["surface"], display_name=row["display_name"],
                prompts=json.loads(row["prompts_json"] or "{}"),
                created_at=row["created_at"])
            p.voice_intro = self._row_voice_intro(row)
            for erow in self._db.execute(
                    "SELECT * FROM elements WHERE profile_id = ?",
                    (profile_id,)):
                likes = self._db.execute(
                    "SELECT COUNT(*) c FROM likes WHERE element_id = ?",
                    (erow["element_id"],)).fetchone()["c"]
                p.elements.append(ProfileElement(
                    element_id=erow["element_id"], type=erow["type"],
                    title=erow["title"], content=erow["content"],
                    url=erow["url"] or "", like_count=likes))
            return p
        except Exception:  # noqa: BLE001
            _log.debug("profiles: get failed", exc_info=True)
            return None

    @staticmethod
    def _row_voice_intro(row: sqlite3.Row) -> VoiceIntro | None:
        """VoiceIntro from a profiles row, or None when unset."""
        try:
            path = row["voice_intro_path"] or ""
        except (IndexError, KeyError):
            return None
        if not path:
            return None
        return VoiceIntro(
            profile_id=row["profile_id"], audio_path=path,
            transcript=row["voice_intro_transcript"] or "",
            language=row["voice_intro_language"] or "en",
            stt_backend=row["voice_intro_stt"] or "",
            created_at=row["voice_intro_at"] or 0.0)

    def list(self, surface: str = "") -> list[Profile]:
        """All profiles, optionally filtered by surface."""
        import json
        if self._db is None:
            return []
        try:
            q = "SELECT * FROM profiles"
            args: tuple = ()
            if surface:
                q += " WHERE surface = ?"
                args = (surface,)
            out = []
            for row in self._db.execute(q, args):
                out.append(Profile(
                    profile_id=row["profile_id"], owner=row["owner"],
                    surface=row["surface"], display_name=row["display_name"],
                    prompts=json.loads(row["prompts_json"] or "{}"),
                    created_at=row["created_at"]))
            return out
        except Exception:  # noqa: BLE001
            return []

    # — prompts —

    def answer_prompt(self, profile_id: str, prompt_id: str,
                      answer: str) -> bool:
        """Answer one scaffold prompt. False when profile/prompt invalid."""
        import json
        if self._db is None:
            return False
        p = self.get(profile_id)
        if p is None:
            return False
        valid = {pid for pid, _ in PROMPTS.get(p.surface, [])}
        if prompt_id not in valid:
            return False
        p.prompts[prompt_id] = (answer or "").strip()
        try:
            self._db.execute(
                "UPDATE profiles SET prompts_json = ? WHERE profile_id = ?",
                (json.dumps(p.prompts), profile_id))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def unanswered(self, profile_id: str) -> list[tuple[str, str]]:
        """Prompts this profile hasn't answered yet — the nudge list."""
        p = self.get(profile_id)
        if p is None:
            return []
        return suggested_prompts(p.surface, set(p.prompts))

    # — voice intro (#82) —

    @staticmethod
    def voice_vault_dir() -> str:
        """Where Devon keeps its copy of voice-intro audio."""
        path = os.path.expanduser("~/.nomorals/social/voice_intros")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _valid_audio(path: str) -> bool:
        """An audio file Devon will accept: exists + known extension."""
        if not path or not os.path.isfile(path):
            return False
        return os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS

    @staticmethod
    def _wav_duration_hint(path: str) -> float | None:
        """Best-effort duration for WAV files (stdlib only)."""
        if os.path.splitext(path)[1].lower() != ".wav":
            return None
        try:
            import wave
            with wave.open(path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return frames / rate if rate else None
        except Exception:  # noqa: BLE001 — hint only, never blocks
            return None

    def set_voice_intro(self, profile_id: str, audio_path: str, *,
                        language: str = "en",
                        stt: object = None) -> VoiceIntro | None:
        """Attach a spoken 30s 'about me' to a profile.

        ``stt`` is the production seam: any callable
        ``(audio_path, language) -> dict`` with a ``"text"`` key.
        None → no transcription is attempted and the transcript stays
        empty (honest: Devon says the intro isn't transcribed yet,
        never fabricates one). None returned when invalid/down.
        """
        if self._db is None:
            return None
        if self.get(profile_id) is None:
            return None
        if not self._valid_audio(audio_path):
            return None
        try:
            from .voice_notes import normalize_voice_language
            lang = normalize_voice_language(language)
        except Exception:  # noqa: BLE001 — voice_notes import problems
            lang = "en"
        try:
            import shutil
            ext = os.path.splitext(audio_path)[1].lower()
            dest = os.path.join(
                self.voice_vault_dir(),
                f"{profile_id}{ext}")
            shutil.copy2(audio_path, dest)
        except Exception:  # noqa: BLE001
            _log.debug("profiles: voice intro copy failed", exc_info=True)
            return None
        transcript, backend = "", ""
        if stt is not None:
            try:
                out = stt(dest, lang) or {}
                transcript = str(out.get("text", "") or "").strip()
                backend = str(out.get("backend", "") or "").strip()
            except Exception:  # noqa: BLE001 — STT failure is not fatal
                _log.debug("profiles: voice intro STT failed", exc_info=True)
        try:
            now = time.time()
            self._db.execute(
                """UPDATE profiles
                   SET voice_intro_path = ?, voice_intro_transcript = ?,
                       voice_intro_language = ?, voice_intro_stt = ?,
                       voice_intro_at = ?
                   WHERE profile_id = ?""",
                (dest, transcript, lang, backend, now, profile_id))
            self._db.commit()
            return VoiceIntro(profile_id=profile_id, audio_path=dest,
                              transcript=transcript, language=lang,
                              stt_backend=backend, created_at=now)
        except Exception:  # noqa: BLE001
            _log.debug("profiles: voice intro save failed", exc_info=True)
            return None

    def get_voice_intro(self, profile_id: str) -> VoiceIntro | None:
        """A profile's voice intro, or None when unset."""
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT * FROM profiles WHERE profile_id = ?",
                (profile_id,)).fetchone()
            return self._row_voice_intro(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def delete_voice_intro(self, profile_id: str) -> bool:
        """Remove a voice intro (and its vault copy). False on failure."""
        vi = self.get_voice_intro(profile_id)
        if vi is None:
            return False
        try:
            self._db.execute(
                """UPDATE profiles
                   SET voice_intro_path = '', voice_intro_transcript = '',
                       voice_intro_language = '', voice_intro_stt = '',
                       voice_intro_at = 0
                   WHERE profile_id = ?""", (profile_id,))
            self._db.commit()
            if vi.audio_path and os.path.isfile(vi.audio_path):
                os.remove(vi.audio_path)
            return True
        except Exception:  # noqa: BLE001
            return False

    # — elements —

    def add_element(self, profile_id: str, type: str, title: str,
                    content: str = "", url: str = "") -> ProfileElement | None:
        """Add a likeable element. None when invalid/down."""
        if self._db is None:
            return None
        if self.get(profile_id) is None:
            return None
        if type not in ELEMENT_TYPES:
            type = "project"
        el = ProfileElement(
            element_id="elem_" + uuid.uuid4().hex[:8], type=type,
            title=(title or "").strip()[:120], content=(content or "").strip(),
            url=(url or "").strip())
        try:
            self._db.execute(
                "INSERT INTO elements VALUES (?, ?, ?, ?, ?, ?)",
                (el.element_id, profile_id, el.type, el.title,
                 el.content, el.url))
            self._db.commit()
            return el
        except Exception:  # noqa: BLE001
            return None

    # — likes —

    def like(self, profile_id: str, element_id: str, liker: str,
             comment: str = "") -> ElementLike | None:
        """Like an element. Optional comment = the conversation starter.

        Idempotent per (profile, element, liker): liking twice keeps the
        first like (and its comment) — the signal is the first touch.
        """
        if self._db is None:
            return None
        if self.get(profile_id) is None:
            return None
        try:
            exists = self._db.execute(
                "SELECT 1 FROM elements WHERE element_id = ? AND profile_id = ?",
                (element_id, profile_id)).fetchone()
            if exists is None:
                return None
            lk = ElementLike(
                profile_id=profile_id, element_id=element_id,
                liker=(liker or "").strip(), comment=(comment or "").strip(),
                created_at=time.time())
            self._db.execute(
                "INSERT OR IGNORE INTO likes VALUES (?, ?, ?, ?, ?)",
                (lk.profile_id, lk.element_id, lk.liker, lk.comment,
                 lk.created_at))
            self._db.commit()
            return lk
        except Exception:  # noqa: BLE001
            return None

    def likes_for(self, element_id: str) -> list[ElementLike]:
        """All likes on an element, oldest first."""
        if self._db is None:
            return []
        try:
            return [ElementLike(
                        profile_id=r["profile_id"], element_id=r["element_id"],
                        liker=r["liker"], comment=r["comment"],
                        created_at=r["created_at"])
                    for r in self._db.execute(
                        "SELECT * FROM likes WHERE element_id = ? "
                        "ORDER BY created_at", (element_id,))]
        except Exception:  # noqa: BLE001
            return []

    def matches(self, owner: str) -> list[ElementLike]:
        """Likes on any of this owner's elements — the match inbox.

        Each like with a comment is a warm opener waiting for a reply.
        """
        if self._db is None:
            return []
        try:
            rows = self._db.execute(
                """SELECT l.* FROM likes l
                   JOIN elements e ON e.element_id = l.element_id
                   JOIN profiles p ON p.profile_id = e.profile_id
                   WHERE p.owner = ? ORDER BY l.created_at""",
                (owner,)).fetchall()
            return [ElementLike(
                        profile_id=r["profile_id"], element_id=r["element_id"],
                        liker=r["liker"], comment=r["comment"],
                        created_at=r["created_at"])
                    for r in rows]
        except Exception:  # noqa: BLE001
            return []


# ── display ─────────────────────────────────────────────────────────────

# ── chat control ──────────────────────────────────────────────────────────

def _usage() -> str:
    return (
        "👤 /uprofile — prompt-scaffolded profiles + element-level likes.\n"
        "  /uprofile create [gig|community|business] [name] — new profile\n"
        "  /uprofile show <id> — view a profile\n"
        "  /uprofile prompt <id> <prompt_id> <answer> — answer a prompt\n"
        "  /uprofile prompts <id> — what to answer next\n"
        "  /uprofile add <id> <type> <title> | [content] — add an element\n"
        "  /uprofile like <profile_id> <element_id> [comment] — like it\n"
        "  /uprofile matches — likes on your elements (the warm openers)\n"
        "  /uprofile voice <id> <audio_path> [language] — 30s spoken intro 🎙️\n"
        "  /uprofile list [surface]"
    )


def control_uprofile(tail: str, context=None, chat=None,
                     sender_id: str = "", sender: str = "") -> str:
    """/uprofile — social profiles. Never raises."""
    try:
        rest = (tail or "").strip()
        store = getattr(context, "profile_store", None) \
            if context is not None else None
        s = store if isinstance(store, ProfileStore) else ProfileStore()
        parts = rest.split()
        if not parts or parts[0].lower() in ("help", "?"):
            return _usage()
        cmd = parts[0].lower()

        if cmd == "create":
            surface = parts[1].lower() if len(parts) > 1 else "gig"
            name = " ".join(parts[2:]) if len(parts) > 2 else ""
            p = s.create(sender or sender_id or "owner", surface, name)
            if p is None:
                return "couldn't create the profile — try again."
            nxt = ", ".join(t for _, t in suggested_prompts(p.surface)[:3])
            return (f"👤 profile created: {p.display_name} ({p.profile_id})\n"
                    f"  answer a prompt to bring it alive: {nxt}\n"
                    f"  /uprofile prompt {p.profile_id} <prompt_id> <answer>")

        if cmd == "show":
            if len(parts) < 2:
                return "usage: /uprofile show <profile_id>"
            p = s.get(parts[1])
            return format_profile(p) if p else "no profile with that id."

        if cmd == "prompts":
            if len(parts) < 2:
                return "usage: /uprofile prompts <profile_id>"
            left = s.unanswered(parts[1])
            if not left:
                return "all prompts answered — this profile is alive. 🎉"
            return ("💡 answer these next:\n" +
                    "\n".join(f"  {pid}: {ptext}" for pid, ptext in left))

        if cmd == "prompt":
            if len(parts) < 4:
                return "usage: /uprofile prompt <profile_id> <prompt_id> <answer>"
            ok = s.answer_prompt(parts[1], parts[2].lower(),
                                 " ".join(parts[3:]))
            return ("✅ saved." if ok
                    else "couldn't save — check the profile id and prompt id "
                         "(/uprofile prompts <id>).")

        if cmd == "add":
            # /uprofile add <id> <type> <title> | [content]
            if len(parts) < 4:
                return ("usage: /uprofile add <profile_id> <project|photo|"
                        "answer|link> <title> | [content]")
            body = " ".join(parts[3:])
            title, _, content = body.partition("|")
            el = s.add_element(parts[1], parts[2].lower(),
                               title.strip(), content.strip())
            return (f"📌 added [{el.element_id}] {el.title}" if el
                    else "couldn't add — check the profile id.")

        if cmd == "like":
            if len(parts) < 3:
                return ("usage: /uprofile like <profile_id> <element_id> "
                        "[comment]")
            lk = s.like(parts[1], parts[2], sender or sender_id or "anon",
                        " ".join(parts[3:]))
            if lk is None:
                return "couldn't like that — check the ids."
            note = (f"\n  💬 your opener: \"{lk.comment}\""
                    if lk.comment else "")
            return f"❤️ liked.{note}"

        if cmd == "matches":
            got = s.matches(sender or sender_id or "owner")
            if not got:
                return "no likes on your elements yet — go be interesting. 😉"
            lines = ["💘 people liked your stuff:"]
            for lk in got[-10:]:
                c = f" — \"{lk.comment}\"" if lk.comment else ""
                lines.append(f"  {lk.liker} liked [{lk.element_id}]{c}")
            return "\n".join(lines)

        if cmd == "voice":
            # /uprofile voice <id> <audio_path> [language] — set intro;
            # /uprofile voice <id> — show the intro's transcript.
            if len(parts) < 2:
                return "usage: /uprofile voice <profile_id> [audio_path] [language]"
            if len(parts) < 3:
                vi = s.get_voice_intro(parts[1])
                if vi is None:
                    return ("no voice intro yet — record ~30s 'about me' "
                            "and /uprofile voice <id> <audio_path>.")
                line = f"🎙️ voice intro [{vi.language}] — {vi.audio_path}"
                if vi.transcript:
                    line += f"\n  “{vi.transcript[:160]}”"
                else:
                    line += "\n  (not transcribed yet)"
                return line
            lang = parts[3].lower() if len(parts) > 3 else "en"
            stt = getattr(context, "stt", None) if context is not None else None
            vi = s.set_voice_intro(parts[1], parts[2], language=lang,
                                   stt=stt)
            if vi is None:
                return ("couldn't set the intro — check the profile id and "
                        "that the file is real audio (wav/mp3/ogg/m4a/opus).")
            note = (f"\n  📝 “{vi.transcript[:160]}”" if vi.transcript
                    else "\n  (audio saved; transcription not available yet)")
            return f"🎙️ voice intro set [{vi.language}].{note}"

        if cmd == "list":
            surface = parts[1].lower() if len(parts) > 1 else ""
            profs = s.list(surface)
            if not profs:
                return "no profiles yet — /uprofile create to start."
            return "\n".join(
                f"👤 {p.display_name} [{p.profile_id}] ({p.surface})"
                for p in profs)

        return _usage()
    except Exception:  # noqa: BLE001 — chat never sees a traceback
        _log.debug("uprofile control failed", exc_info=True)
        return "profile command hit a snag — try again."


def format_profile(p: Profile) -> str:
    """Render a profile as chat text — prompts first, elements likeable."""
    lines = [f"👤 {p.display_name}"]
    if p.voice_intro is not None:
        vi = p.voice_intro
        tag = f" 🎙️ voice intro [{vi.language}]"
        if vi.transcript:
            tag += f" — “{vi.transcript[:80]}”"
        lines.append(tag)
    prompt_list = PROMPTS.get(p.surface, [])
    for pid, ptext in prompt_list:
        ans = p.prompts.get(pid)
        if ans:
            lines.append(f"  • {ptext} → {ans}")
    if p.elements:
        lines.append("")
        for el in p.elements:
            heart = f" ❤️×{el.like_count}" if el.like_count else ""
            lines.append(f"  📌 [{el.element_id}] {el.title}{heart}")
            if el.content:
                lines.append(f"     {el.content[:120]}")
    missing = [pt for pid, pt in prompt_list if pid not in p.prompts]
    if missing:
        lines.append(f"\n  💡 Unanswered: {', '.join(missing[:2])}")
    return "\n".join(lines)
