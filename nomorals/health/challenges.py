"""Photo-proof challenges + streak/squad mechanics (build-map #87).

Proven retention, trivial to build (Proovit, FitSquads, Workout Quest):
- **Photo-proof challenges** — submit a workout photo; Seer verifies
  "that's a gym, that's a person exercising"; leaderboards rank by
  verified completions.
- **Streaks** — daily check-ins; loss-aversion nudges ("3-day streak
  at risk!").
- **Squads** — small groups (3–8) with shared challenges, living in
  the #12 community namespace.
- **#14 features-as-loot** — challenge completion awards badges via
  the achievements system.
- **Explicit override** — "log my workout" logs it, no interrogation.

Seer verification is injectable (``verifier(image_path, question) -> str``).
Without one, proofs are accepted but flagged ``verified=False`` — never
blocks the user, always honest. Every method never raises.

Chat: ``/challenge`` (owner + community, not owner-gated).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
import uuid

_log = logging.getLogger("nomorals.health.challenges")

_DEFAULT_DB = os.path.expanduser("~/.nomorals/health/challenges.db")

CHALLENGE_TYPES = ("workout-count", "streak-days", "distance", "custom")
PROOF_METHODS = ("photo", "honor", "none")

SQUAD_MIN = 3
SQUAD_MAX = 8

_STREAK_RISK_DAYS = 1  # check-in missing today → at risk


# ── dataclasses ──────────────────────────────────────────────────────

class Challenge:
    """One challenge: a goal, a duration, and a proof method."""

    def __init__(self, challenge_id: str = "", name: str = "",
                 ctype: str = "workout-count", duration_days: int = 7,
                 target: int = 5, proof_method: str = "photo",
                 created_by: str = "", created_at: float = 0.0,
                 badge: str = "") -> None:
        self.id = challenge_id
        self.name = name
        self.type = ctype if ctype in CHALLENGE_TYPES else "custom"
        self.duration_days = max(1, int(duration_days or 7))
        self.target = max(1, int(target or 1))
        self.proof_method = (proof_method if proof_method in PROOF_METHODS
                             else "photo")
        self.created_by = created_by
        self.created_at = created_at or time.time()
        self.badge = badge or f"challenge:{challenge_id or 'new'}"

    @property
    def ends_at(self) -> float:
        return self.created_at + self.duration_days * 86400.0

    @property
    def active(self) -> bool:
        return time.time() < self.ends_at


class Proof:
    """One submitted proof of a workout."""

    def __init__(self, proof_id: str = "", challenge_id: str = "",
                 member: str = "", photo_path: str = "",
                 verified: bool = False, verdict: str = "",
                 created_at: float = 0.0) -> None:
        self.id = proof_id
        self.challenge_id = challenge_id
        self.member = member
        self.photo_path = photo_path
        self.verified = verified
        self.verdict = verdict
        self.created_at = created_at or time.time()


class Squad:
    """A small workout squad (3–8 members) in the community namespace."""

    def __init__(self, squad_id: str = "", name: str = "",
                 members: list | None = None,
                 created_at: float = 0.0) -> None:
        self.id = squad_id
        self.name = name
        self.members = list(members or [])
        self.created_at = created_at or time.time()


# ── photo verification ───────────────────────────────────────────────

_VERIFY_PROMPT = (
    "Does this photo show a person exercising or working out "
    "(gym, home workout, running, sports)? Answer with one word: "
    "YES or NO, then one short sentence explaining what you see."
)

_YES_RE = re.compile(r"\b(yes|yeah|yep|correct|true)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|nope|not|false)\b", re.IGNORECASE)


def verify_proof_photo(photo_path: str, verifier=None) -> tuple[bool, str]:
    """Verify a workout photo via Seer. Returns (accepted, verdict).

    ``verifier(image_path, question) -> str`` is the injectable Seer seam.
    Without a verifier the photo is accepted but flagged unverified —
    never blocks the user, always honest. Never raises.
    """
    try:
        if not photo_path or not os.path.exists(photo_path):
            return False, "no photo found — submit a real photo"
        if verifier is None:
            return True, "accepted (unverified — vision not available)"
        text = (verifier(photo_path, _VERIFY_PROMPT) or "").strip()
        if not text:
            return True, "accepted (unverified — vision returned nothing)"
        if _NO_RE.search(text) and not _YES_RE.search(text):
            return False, f"rejected: {text[:140]}"
        return True, f"verified: {text[:140]}"
    except Exception:  # noqa: BLE001 — never raises
        _log.debug("proof verification failed", exc_info=True)
        return True, "accepted (verification errored — flagged unverified)"


# ── stores ───────────────────────────────────────────────────────────

def _connect(db_path: str):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute(
        """CREATE TABLE IF NOT EXISTS challenges (
               id TEXT PRIMARY KEY, name TEXT, ctype TEXT,
               duration_days INTEGER, target INTEGER,
               proof_method TEXT, created_by TEXT,
               created_at REAL, badge TEXT)""")
    db.execute(
        """CREATE TABLE IF NOT EXISTS challenge_members (
               challenge_id TEXT, member TEXT,
               joined_at REAL,
               PRIMARY KEY (challenge_id, member))""")
    db.execute(
        """CREATE TABLE IF NOT EXISTS proofs (
               id TEXT PRIMARY KEY, challenge_id TEXT, member TEXT,
               photo_path TEXT, verified INTEGER, verdict TEXT,
               created_at REAL)""")
    db.execute(
        """CREATE TABLE IF NOT EXISTS completions (
               challenge_id TEXT, member TEXT, count INTEGER,
               completed INTEGER DEFAULT 0,
               PRIMARY KEY (challenge_id, member))""")
    db.execute(
        """CREATE TABLE IF NOT EXISTS streaks (
               member TEXT PRIMARY KEY, count INTEGER,
               last_checkin REAL)""")
    db.execute(
        """CREATE TABLE IF NOT EXISTS squads (
               id TEXT PRIMARY KEY, name TEXT, members_json TEXT,
               created_at REAL)""")
    db.commit()
    return db


class ChallengeStore:
    """Challenges, proofs, completions, streaks, squads. Never raises."""

    def __init__(self, db_path: str = "", verifier=None) -> None:
        self._db_path = db_path or _DEFAULT_DB
        self._verifier = verifier
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._db = _connect(self._db_path)
        except Exception:  # noqa: BLE001 — bad path → empty store
            _log.warning("challenges: db unavailable, running empty",
                         exc_info=True)
            self._db = None

    # — challenges —

    def create_challenge(self, name: str, ctype: str = "workout-count",
                         duration_days: int = 7, target: int = 5,
                         proof_method: str = "photo",
                         created_by: str = "") -> Challenge | None:
        """Create a challenge. Never raises."""
        try:
            name = (name or "").strip()[:120]
            if not name:
                return None
            c = Challenge(
                challenge_id="ch_" + uuid.uuid4().hex[:8],
                name=name, ctype=ctype, duration_days=duration_days,
                target=target, proof_method=proof_method,
                created_by=created_by or "owner")
            c.badge = f"challenge:{c.id}"
            self._db.execute(
                "INSERT INTO challenges VALUES (?,?,?,?,?,?,?,?,?)",
                (c.id, c.name, c.type, c.duration_days, c.target,
                 c.proof_method, c.created_by, c.created_at, c.badge))
            self._db.commit()
            return c
        except Exception:  # noqa: BLE001
            _log.debug("create_challenge failed", exc_info=True)
            return None

    def get(self, challenge_id: str) -> Challenge | None:
        try:
            row = self._db.execute(
                "SELECT * FROM challenges WHERE id = ?",
                (challenge_id,)).fetchone()
            if row is None:
                return None
            return Challenge(
                challenge_id=row["id"], name=row["name"],
                ctype=row["ctype"], duration_days=row["duration_days"],
                target=row["target"], proof_method=row["proof_method"],
                created_by=row["created_by"], created_at=row["created_at"],
                badge=row["badge"])
        except Exception:  # noqa: BLE001
            return None

    def list_challenges(self, active_only: bool = True) -> list[Challenge]:
        try:
            out = []
            for row in self._db.execute("SELECT * FROM challenges"):
                c = self.get(row["id"])
                if c is None:
                    continue
                if active_only and not c.active:
                    continue
                out.append(c)
            return out
        except Exception:  # noqa: BLE001
            return []

    def join_challenge(self, challenge_id: str, member: str) -> bool:
        try:
            if self.get(challenge_id) is None:
                return False
            member = (member or "").strip()[:80]
            if not member:
                return False
            self._db.execute(
                "INSERT OR IGNORE INTO challenge_members VALUES (?,?,?)",
                (challenge_id, member, time.time()))
            self._db.execute(
                "INSERT OR IGNORE INTO completions VALUES (?,?,0,0)",
                (challenge_id, member))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def members(self, challenge_id: str) -> list[str]:
        try:
            return [r["member"] for r in self._db.execute(
                "SELECT member FROM challenge_members WHERE challenge_id = ?",
                (challenge_id,))]
        except Exception:  # noqa: BLE001
            return []

    # — workout logging (explicit override: "log my workout") —

    def log_workout(self, challenge_id: str, member: str) -> int:
        """Log a workout, no interrogation. Returns the new count.

        This is the explicit-override path: the member said "log my
        workout" so it gets logged. Proof (photo) is a separate,
        optional step.
        """
        try:
            c = self.get(challenge_id)
            if c is None:
                return -1
            member = (member or "").strip()[:80]
            if not member:
                return -1
            self.join_challenge(challenge_id, member)
            self._db.execute(
                "UPDATE completions SET count = count + 1 "
                "WHERE challenge_id = ? AND member = ?",
                (challenge_id, member))
            self._db.commit()
            row = self._db.execute(
                "SELECT count FROM completions WHERE challenge_id = ? "
                "AND member = ?",
                (challenge_id, member)).fetchone()
            count = int(row["count"]) if row else 0
            self._maybe_complete(c, member, count)
            return count
        except Exception:  # noqa: BLE001
            _log.debug("log_workout failed", exc_info=True)
            return -1

    # — photo proof —

    def submit_proof(self, challenge_id: str, member: str,
                     photo_path: str) -> Proof | None:
        """Submit photo proof → Seer verification → accepted/rejected."""
        try:
            c = self.get(challenge_id)
            if c is None:
                return None
            member = (member or "").strip()[:80]
            if not member:
                return None
            self.join_challenge(challenge_id, member)
            if c.proof_method == "none":
                accepted, verdict = True, "no proof required"
            elif c.proof_method == "honor":
                accepted, verdict = True, "honor system — trusted"
            else:
                accepted, verdict = verify_proof_photo(
                    photo_path, self._verifier)
            p = Proof(
                proof_id="proof_" + uuid.uuid4().hex[:8],
                challenge_id=challenge_id, member=member,
                photo_path=photo_path or "",
                verified=accepted and "unverified" not in verdict,
                verdict=verdict)
            self._db.execute(
                "INSERT INTO proofs VALUES (?,?,?,?,?,?,?)",
                (p.id, p.challenge_id, p.member, p.photo_path,
                 int(p.verified), p.verdict, p.created_at))
            self._db.commit()
            if accepted:
                count = self.log_workout(challenge_id, member)
                if count >= 0:
                    p.verdict += f" (workout #{count} logged)"
            return p
        except Exception:  # noqa: BLE001
            _log.debug("submit_proof failed", exc_info=True)
            return None

    # — completion + #14 loot —

    def _maybe_complete(self, c: Challenge, member: str, count: int) -> None:
        """Mark completion and award the badge (#14 features-as-loot)."""
        try:
            if count < c.target:
                return
            self._db.execute(
                "UPDATE completions SET completed = 1 "
                "WHERE challenge_id = ? AND member = ?",
                (c.id, member))
            self._db.commit()
            try:
                from ...games.achievements import unlock_achievement
                from ...games.database import Database
                db = Database()
                unlock_achievement(db, member, c.badge)
            except Exception:  # noqa: BLE001 — loot is best-effort
                _log.debug("challenge badge unlock failed", exc_info=True)
        except Exception:  # noqa: BLE001
            _log.debug("_maybe_complete failed", exc_info=True)

    def progress(self, challenge_id: str, member: str) -> tuple[int, int]:
        """(count, target) for a member."""
        try:
            c = self.get(challenge_id)
            if c is None:
                return (0, 0)
            row = self._db.execute(
                "SELECT count FROM completions WHERE challenge_id = ? "
                "AND member = ?",
                (challenge_id, member)).fetchone()
            return (int(row["count"]) if row else 0, c.target)
        except Exception:  # noqa: BLE001
            return (0, 0)

    def leaderboard(self, challenge_id: str,
                    limit: int = 10) -> list[tuple[str, int, bool]]:
        """Ranked (member, count, completed). Never raises."""
        try:
            rows = self._db.execute(
                "SELECT member, count, completed FROM completions "
                "WHERE challenge_id = ? ORDER BY completed DESC, count DESC "
                "LIMIT ?",
                (challenge_id, max(1, int(limit or 10)))).fetchall()
            return [(r["member"], int(r["count"]), bool(r["completed"]))
                    for r in rows]
        except Exception:  # noqa: BLE001
            return []

    # — streaks —

    def checkin(self, member: str, *, now: float | None = None) -> int:
        """Daily check-in. Returns the streak count. Never raises."""
        try:
            member = (member or "").strip()[:80]
            if not member:
                return 0
            now = now if now is not None else time.time()
            row = self._db.execute(
                "SELECT count, last_checkin FROM streaks WHERE member = ?",
                (member,)).fetchone()
            today = time.strftime("%Y-%m-%d", time.localtime(now))
            if row is None:
                self._db.execute(
                    "INSERT INTO streaks VALUES (?,?,?)",
                    (member, 1, now))
                self._db.commit()
                return 1
            last_day = time.strftime(
                "%Y-%m-%d", time.localtime(float(row["last_checkin"] or 0)))
            if last_day == today:
                return int(row["count"])
            # consecutive if yesterday, else reset
            yday = time.strftime(
                "%Y-%m-%d", time.localtime(now - 86400))
            count = int(row["count"]) + 1 if last_day == yday else 1
            self._db.execute(
                "UPDATE streaks SET count = ?, last_checkin = ? "
                "WHERE member = ?",
                (count, now, member))
            self._db.commit()
            return count
        except Exception:  # noqa: BLE001
            _log.debug("checkin failed", exc_info=True)
            return 0

    def streak(self, member: str) -> int:
        try:
            row = self._db.execute(
                "SELECT count FROM streaks WHERE member = ?",
                ((member or "").strip()[:80],)).fetchone()
            return int(row["count"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    def at_risk(self, *, now: float | None = None) -> list[tuple[str, int]]:
        """Members whose streak is at risk (no check-in today)."""
        try:
            now = now if now is not None else time.time()
            today = time.strftime("%Y-%m-%d", time.localtime(now))
            out = []
            for row in self._db.execute("SELECT * FROM streaks"):
                last_day = time.strftime(
                    "%Y-%m-%d",
                    time.localtime(float(row["last_checkin"] or 0)))
                if last_day != today and int(row["count"]) >= 2:
                    out.append((row["member"], int(row["count"])))
            return sorted(out, key=lambda x: -x[1])
        except Exception:  # noqa: BLE001
            return []

    def risk_nudge(self, member: str) -> str:
        """Loss-aversion nudge for a member's streak."""
        try:
            n = self.streak(member)
            if n >= 2:
                return (f"🔥 {member}: your {n}-day streak is at risk! "
                        "Check in today to keep it alive.")
            return ""
        except Exception:  # noqa: BLE001
            return ""

    # — squads (#12 community namespace) —

    def create_squad(self, name: str,
                     members: list[str]) -> Squad | None:
        """Create a squad of 3–8 members. Never raises."""
        try:
            name = (name or "").strip()[:80]
            clean = [str(m).strip()[:80] for m in (members or [])
                     if str(m).strip()]
            # dedupe, preserve order
            seen, deduped = set(), []
            for m in clean:
                if m not in seen:
                    seen.add(m)
                    deduped.append(m)
            if not name or not (SQUAD_MIN <= len(deduped) <= SQUAD_MAX):
                return None
            import json as _json
            s = Squad(squad_id="sq_" + uuid.uuid4().hex[:8],
                      name=name, members=deduped)
            self._db.execute(
                "INSERT INTO squads VALUES (?,?,?,?)",
                (s.id, s.name, _json.dumps(s.members), s.created_at))
            self._db.commit()
            return s
        except Exception:  # noqa: BLE001
            _log.debug("create_squad failed", exc_info=True)
            return None

    def get_squad(self, squad_id: str) -> Squad | None:
        try:
            import json as _json
            row = self._db.execute(
                "SELECT * FROM squads WHERE id = ?",
                (squad_id,)).fetchone()
            if row is None:
                return None
            return Squad(squad_id=row["id"], name=row["name"],
                         members=_json.loads(row["members_json"] or "[]"),
                         created_at=row["created_at"])
        except Exception:  # noqa: BLE001
            return None

    def squad_leaderboard(self, squad_id: str,
                          challenge_id: str = "") -> list[tuple[str, int]]:
        """Rank squad members. Uses streaks when no challenge given."""
        try:
            s = self.get_squad(squad_id)
            if s is None:
                return []
            if challenge_id:
                board = {m: n for m, n, _ in
                         self.leaderboard(challenge_id, limit=1000)}
                scored = [(m, board.get(m, 0)) for m in s.members]
            else:
                scored = [(m, self.streak(m)) for m in s.members]
            return sorted(scored, key=lambda x: -x[1])
        except Exception:  # noqa: BLE001
            return []


# ── chat ─────────────────────────────────────────────────────────────

def _store(context) -> ChallengeStore:
    store = getattr(context, "challenge_store", None)
    if not isinstance(store, ChallengeStore):
        verifier = getattr(context, "challenge_verifier", None)
        store = ChallengeStore(verifier=verifier)
    return store


def _usage() -> str:
    return (
        "🏋️ /challenge — photo-proof challenges + streaks + squads\n"
        "create <name> | <workout-count|streak-days|distance|custom> | <days> [target N]\n"
        "list · join <id> · log <id> (log my workout, no interrogation)\n"
        "proof <id> <photo-path> — Seer-verified workout photo\n"
        "board <id> — leaderboard\n"
        "streak — your streak · risk — who's at risk\n"
        "squad create <name> | <m1,m2,m3+> · squad board <squad_id> [challenge_id]"
    )


def control_challenge(tail: str, context=None, chat=None,
                      sender_id: str = "", sender: str = "") -> str:
    """/challenge — photo-proof challenges + streak/squad mechanics.

    Owner + community (not owner-gated). Never raises.
    """
    try:
        rest = (tail or "").strip()
        store = _store(context)
        who = (sender or sender_id or "owner").strip() or "owner"
        if not rest or rest.split()[0] in ("help", "?"):
            return _usage()
        parts = rest.split(None, 1)
        action, args = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if action == "create":
            segs = [s.strip() for s in args.split("|")]
            name = segs[0] if len(segs) > 0 else ""
            ctype = segs[1].lower() if len(segs) > 1 else "workout-count"
            days = int(segs[2]) if len(segs) > 2 and segs[2].isdigit() else 7
            target = 5
            m = re.search(r"target\s+(\d+)", args, re.IGNORECASE)
            if m:
                target = int(m.group(1))
            c = store.create_challenge(name, ctype, days, target,
                                       created_by=who)
            if c is None:
                return "couldn't create that challenge — need a name."
            return (f"🏋️ challenge created: *{c.name}* ({c.id})\n"
                    f"{c.type} · {c.target} in {c.duration_days} days · "
                    f"proof: {c.proof_method}\n"
                    f"others join with: /challenge join {c.id}")

        if action == "list":
            cs = store.list_challenges()
            if not cs:
                return "no active challenges. /challenge create to start one."
            return "\n".join(
                f"• *{c.name}* ({c.id}) — {c.type}, {c.target} in "
                f"{c.duration_days}d, {len(store.members(c.id))} in"
                for c in cs)

        if action == "join":
            cid = args.strip().split()[0] if args.strip() else ""
            if store.join_challenge(cid, who):
                return f"you're in! 💪 (/challenge log {cid} to log a workout)"
            return "couldn't join — check the challenge id."

        if action == "log":
            # Explicit override: "log my workout" → logged, no interrogation.
            cid = args.strip().split()[0] if args.strip() else ""
            n = store.log_workout(cid, who)
            if n < 0:
                return "couldn't log — check the challenge id."
            c = store.get(cid)
            done = " 🏆 COMPLETE!" if n >= (c.target if c else 0) else ""
            return f"logged! workout #{n} for *{c.name if c else cid}*.{done}"

        if action == "proof":
            segs = args.split(None, 1)
            if len(segs) < 2:
                return "usage: /challenge proof <id> <photo-path>"
            cid, photo = segs[0], segs[1].strip()
            p = store.submit_proof(cid, who, photo)
            if p is None:
                return "couldn't submit — check the challenge id."
            icon = "✅" if "verified" in p.verdict else "📝"
            return f"{icon} {p.verdict}"

        if action == "board":
            cid = args.strip().split()[0] if args.strip() else ""
            board = store.leaderboard(cid)
            if not board:
                return "no scores yet — be the first. 💪"
            lines = [f"{i+1}. {m} — {n} {'🏆' if done else ''}"
                     for i, (m, n, done) in enumerate(board)]
            return "🏆 leaderboard:\n" + "\n".join(lines)

        if action == "streak":
            store.checkin(who)
            return f"🔥 {who}: {store.streak(who)}-day streak."

        if action == "risk":
            risk = store.at_risk()
            if not risk:
                return "no streaks at risk right now."
            return "\n".join(f"⚠️ {m}: {n}-day streak at risk!"
                             for m, n in risk)

        if action == "squad":
            sub = args.strip().split(None, 1)
            if not sub:
                return ("usage: /challenge squad create <name> | <m1,m2,…> "
                        "· /challenge squad board <squad_id> [challenge_id]")
            if sub[0].lower() == "create":
                segs = [s.strip() for s in (sub[1] if len(sub) > 1
                                           else "").split("|")]
                name = segs[0] if segs else ""
                members = ([m.strip() for m in segs[1].split(",")]
                           if len(segs) > 1 else [])
                if who not in members:
                    members.append(who)
                s = store.create_squad(name, members)
                if s is None:
                    return (f"squads need {SQUAD_MIN}–{SQUAD_MAX} members "
                            "and a name.")
                return (f"👥 squad *{s.name}* created ({s.id}): "
                        + ", ".join(s.members))
            if sub[0].lower() == "board":
                segs = (sub[1] if len(sub) > 1 else "").split()
                if not segs:
                    return "usage: /challenge squad board <squad_id>"
                cid = segs[1] if len(segs) > 1 else ""
                board = store.squad_leaderboard(segs[0], cid)
                if not board:
                    return "no squad scores yet."
                return ("👥 squad leaderboard:\n" + "\n".join(
                    f"{i+1}. {m} — {n}" for i, (m, n) in enumerate(board)))
            return _usage()

        return _usage()
    except Exception:  # noqa: BLE001 — chat never raises
        _log.debug("control_challenge failed", exc_info=True)
        return "challenge hiccup — try again."
