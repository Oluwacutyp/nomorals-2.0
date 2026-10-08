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
    "ProfileStore",
    "suggested_prompts",
    "format_profile",
]

#: Profile surfaces Devon knows how to scaffold.
SURFACES = ("gig", "community", "business")

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
class Profile:
    """One profile on one surface, scaffolded by prompts."""

    profile_id: str = ""
    owner: str = ""
    surface: str = "gig"           # gig | community | business
    display_name: str = ""
    prompts: dict[str, str] = field(default_factory=dict)
    elements: list[ProfileElement] = field(default_factory=list)
    created_at: float = 0.0


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
        except Exception:  # noqa: BLE001 — a bad DB path is an empty store
            _log.warning("profiles: db unavailable, running empty", exc_info=True)
            self._db = None

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
                "INSERT INTO profiles VALUES (?, ?, ?, ?, ?, ?)",
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
