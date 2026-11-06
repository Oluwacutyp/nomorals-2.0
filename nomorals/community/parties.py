"""AI-run parties for group chats.

One party, end to end, driven by the bot inside the group:

    INVITE → START → MUSIC → GAMES → SCORE → WALL → END → NEXT DAY

* **INVITE** — a party is a recurring-able :mod:`nomorals.community.events`
  event with RSVPs, so a weekly Friday party is one ritual, not a new
  invite every week.
* **START** — the bot pings the group and the DJ opens (an injectable
  ``dj_opener`` callable; the default opener is an honest text opener,
  not a fake audio stream).
* **MUSIC** — members request tracks and vote; the queue is
  vote-ordered, ties break by earliest request.
* **GAMES** — trivia rounds between tracks via an injectable
  ``trivia_provider`` (any callable returning a question dict, or
  ``None``). Quiz points + streaks live here.
* **SCORE** — achievements (first win, streaks, top voter, MVP) and an
  injectable ``economy_hook`` that credits group-economy points.
* **WALL** — members post text/photo/video moments; the wall feeds the recap.
* **END** — auto-recap: top tracks, top players, top wall moments.
* **NEXT DAY** — a "last night" card for the most recent ended party.

**Fail-soft is the whole contract.** A party is a live social moment;
one broken step (a trivia provider raising, a DJ opener timing out,
an economy hook failing) must never kill the party or raise into the
chat layer. Every step runs through :meth:`PartyManager._safe`:
failures are recorded on ``party.failures``, the party keeps going,
and the chat sees silence — never an exception, never a traceback.

**Rehearsals.** The first three parties in a group are flagged
``rehearsal=True``: they run the full show, but economy credits are
dry-run only (computed, reported, never credited) and the recap says
so. Rehearsals let a group learn the format before points are real.

**Isolation.** stdlib + ``nomorals.core`` + sibling community modules
only (``events``). No owner data, no memory, no connectors. The DJ,
trivia, and economy hooks are all injected callables — this module
never reaches outside the group boundary to find them.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .events import EventStore

_log = get_logger("nomorals.community.parties")

__all__ = [
    "Party", "PartyManager", "PartyStore", "PartyRecap",
    "PARTY_STATES", "REHEARSAL_COUNT", "TRIVIA_POINTS", "STREAK_BONUS_CAP",
    "format_recap", "format_last_night_card",
    "control_party", "party_economy_hook",
]

#: First N parties per group run as rehearsals (dry-run economy).
REHEARSAL_COUNT = 3

#: Points for a correct trivia answer, plus streak bonus.
TRIVIA_POINTS = 10
STREAK_BONUS_PER = 2
STREAK_BONUS_CAP = 10

PARTY_STATES = ("planned", "invited", "live", "paused", "ended")
TRACK_REQUEST_LIMIT = 5  # per member per party — one person can't flood the queue

_DATA_DIR = Path.home() / ".devon" / "community" / "parties"


# ── data model ─────────────────────────────────────────────────────────────


@dataclass
class PartyRecap:
    """The auto-recap produced at END. Plain data, no side effects."""

    party_id: str = ""
    title: str = ""
    rehearsal: bool = False
    started_at: float = 0.0
    ended_at: float = 0.0
    attendees: list[str] = field(default_factory=list)
    top_tracks: list[dict[str, Any]] = field(default_factory=list)
    top_players: list[dict[str, Any]] = field(default_factory=list)
    top_wall: list[dict[str, Any]] = field(default_factory=list)
    achievements: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    economy_credited: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartyRecap":
        return cls(
            party_id=str(data.get("party_id", "")),
            title=str(data.get("title", "")),
            rehearsal=bool(data.get("rehearsal")),
            started_at=float(data.get("started_at") or 0.0),
            ended_at=float(data.get("ended_at") or 0.0),
            attendees=[str(x) for x in (data.get("attendees") or [])],
            top_tracks=list(data.get("top_tracks") or []),
            top_players=list(data.get("top_players") or []),
            top_wall=list(data.get("top_wall") or []),
            achievements=list(data.get("achievements") or []),
            failures=[str(x) for x in (data.get("failures") or [])],
            economy_credited={str(k): int(v)
                              for k, v in (data.get("economy_credited") or {}).items()},
        )


@dataclass
class Party:
    """One party: state machine + queue + scores + wall."""

    id: str = ""
    group_id: str = ""
    title: str = ""
    host: str = ""
    theme: str = ""
    starts_at: float = 0.0
    state: str = "planned"  # planned → invited → live → paused ↔ live → ended
    rehearsal: bool = False
    event_id: str = ""  # linked events.py event (INVITE + RSVP source of truth)
    created_at: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    outbox: list[str] = field(default_factory=list)  # messages for the host to deliver
    queue: list[dict[str, Any]] = field(default_factory=list)  # track requests
    played: list[dict[str, Any]] = field(default_factory=list)  # played tracks
    current_track: dict[str, Any] | None = None
    votes: dict[str, list[str]] = field(default_factory=dict)  # track_id → voters
    trivia_scores: dict[str, int] = field(default_factory=dict)
    trivia_streaks: dict[str, int] = field(default_factory=dict)
    trivia_rounds: int = 0
    wall: list[dict[str, Any]] = field(default_factory=list)
    achievements: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    economy_credited: dict[str, int] = field(default_factory=dict)
    recap: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Party":
        p = cls(
            id=str(data.get("id", "")),
            group_id=str(data.get("group_id", "")),
            title=str(data.get("title", "")),
            host=str(data.get("host", "")),
            theme=str(data.get("theme", "")),
            starts_at=float(data.get("starts_at") or 0.0),
            state=str(data.get("state", "planned")),
            rehearsal=bool(data.get("rehearsal")),
            event_id=str(data.get("event_id", "")),
            created_at=float(data.get("created_at") or 0.0),
            started_at=float(data.get("started_at") or 0.0),
            ended_at=float(data.get("ended_at") or 0.0),
            outbox=[str(x) for x in (data.get("outbox") or [])],
            queue=list(data.get("queue") or []),
            played=list(data.get("played") or []),
            current_track=data.get("current_track"),
            votes={str(k): [str(v) for v in vs]
                   for k, vs in (data.get("votes") or {}).items()},
            trivia_scores={str(k): int(v)
                           for k, v in (data.get("trivia_scores") or {}).items()},
            trivia_streaks={str(k): int(v)
                            for k, v in (data.get("trivia_streaks") or {}).items()},
            trivia_rounds=int(data.get("trivia_rounds") or 0),
            wall=list(data.get("wall") or []),
            achievements=list(data.get("achievements") or []),
            failures=[str(x) for x in (data.get("failures") or [])],
            economy_credited={str(k): int(v)
                              for k, v in (data.get("economy_credited") or {}).items()},
            recap=data.get("recap"),
        )
        return p

    # ── derived ──
    def attendees(self) -> list[str]:
        seen: list[str] = []
        for src in (self.queue, self.wall):
            for item in src:
                who = str(item.get("by") or item.get("author") or "")
                if who and who not in seen:
                    seen.append(who)
        for who in list(self.trivia_scores) + list(self.votes_voters()):
            if who not in seen:
                seen.append(who)
        if self.host and self.host not in seen:
            seen.append(self.host)
        return seen

    def votes_voters(self) -> list[str]:
        out: list[str] = []
        for voters in self.votes.values():
            for v in voters:
                if v not in out:
                    out.append(v)
        return out

    def queue_ranked(self) -> list[dict[str, Any]]:
        """Queue ordered by votes desc, earliest request first."""

        def key(t: dict[str, Any]) -> tuple[int, float]:
            return (-len(self.votes.get(str(t.get("id", "")), [])),
                    float(t.get("requested_at") or 0.0))

        return sorted(self.queue, key=key)


class PartyStore:
    """JSON persistence for parties, per group. Never raises."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR

    def _path(self, group_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_"
                       for c in str(group_id)).strip("_") or "group"
        return self.data_dir / f"{safe}.json"

    def load(self, group_id: str) -> list[Party]:
        try:
            raw = json.loads(self._path(group_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out: list[Party] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                p = Party.from_dict(item)
                if p.id:
                    out.append(p)
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    def save(self, group_id: str, parties: list[Party]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path(group_id).with_suffix(".tmp")
            tmp.write_text(json.dumps([p.to_dict() for p in parties],
                                     ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self._path(group_id))
        except OSError as exc:
            _log.warning("party save failed for %s: %s", group_id, exc)

    def get(self, group_id: str, party_id: str) -> Party | None:
        for p in self.load(group_id):
            if p.id == party_id or p.id.startswith(party_id):
                return p
        return None

    def put(self, party: Party) -> None:
        parties = [p for p in self.load(party.group_id) if p.id != party.id]
        parties.append(party)
        self.save(party.group_id, parties)

    def live(self, group_id: str) -> Party | None:
        for p in self.load(group_id):
            if p.state == "live":
                return p
        return None


# ── manager ────────────────────────────────────────────────────────────────


class PartyManager:
    """Runs parties. Hooks are injected; every step is fail-soft.

    ``dj_opener``: ``Callable[[dict], str | None]`` — called with the
    opening track, returns the opener line (or None to use the default).
    ``trivia_provider``: ``Callable[[Party], dict | None]`` — returns
    ``{question, options, answer_index}`` or None when out of questions.
    ``economy_hook``: ``Callable[[group_id, member, amount, reason], bool]``
    — credits group-economy points; its return is ignored on failure.
    ``emit``: ``Callable[[str], None]`` — delivers a message to the
    group. Default collects into ``party.outbox``.
    """

    def __init__(self, store: PartyStore | None = None,
                 event_store: EventStore | None = None,
                 dj_opener: Callable[[dict[str, Any]], str | None] | None = None,
                 trivia_provider: Callable[[Party], dict[str, Any] | None] | None = None,
                 economy_hook: Callable[[str, str, int, str], bool] | None = None,
                 emit: Callable[[str], None] | None = None) -> None:
        self.store = store or PartyStore()
        self.events = event_store or EventStore()
        self.dj_opener = dj_opener
        self.trivia_provider = trivia_provider
        self.economy_hook = economy_hook
        self._emit = emit

    # ── fail-soft core ──
    def _safe(self, party: Party, step: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn``; on failure record it and return None. Never raises."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — fail-soft is the contract
            _log.debug("party %s step %s failed: %s", party.id, step, exc)
            party.failures.append(f"{step}: {type(exc).__name__}")
            return None

    def _say(self, party: Party, text: str) -> None:
        if self._emit is not None:
            self._safe(party, "emit", lambda: self._emit(text))  # type: ignore[misc]
        else:
            party.outbox.append(text)

    # ── INVITE ──
    def create_party(self, group_id: str, title: str, starts_at: float,
                     host: str = "", theme: str = "",
                     recurrence: str = "") -> Party | None:
        """Create a party + its linked event (INVITE). Never raises."""
        try:
            title = (title or "").strip()[:120] or "Party"
            if not group_id or starts_at <= 0:
                return None
            past = self.store.load(group_id)
            rehearsal = len(past) < REHEARSAL_COUNT
            party = Party(
                id="pty_" + new_short_id(),
                group_id=group_id,
                title=title,
                host=(host or "").strip() or "host",
                theme=(theme or "").strip()[:80],
                starts_at=starts_at,
                state="planned",
                rehearsal=rehearsal,
                created_at=time.time(),
            )
            ev = self._safe(party, "event.create", lambda: self.events.create_event(
                group_id, f"🎉 {title}", starts_at,
                where="group chat", created_by=party.host,
                recurrence=recurrence,
                description=f"Party: {title}" + (f" · theme: {theme}" if theme else "")))
            if ev is not None:
                party.event_id = ev.id
            party.state = "invited"
            self.store.put(party)
            flag = " 🧪 rehearsal — points are dry-run" if rehearsal else ""
            self._say(party, f"🎉 **{title}** is on!{flag}\n"
                             f"Theme: {theme or 'open'} · "
                             f"RSVP: /cgroup event rsvp {party.event_id} yes")
            self.store.put(party)
            return party
        except Exception as exc:  # noqa: BLE001
            _log.warning("create_party failed: %s", exc)
            return None

    def rsvp(self, party: Party, member: str, choice: str) -> str:
        """RSVP through the linked event. '' when no linked event."""
        if not party.event_id:
            return ""
        result = self._safe(party, "rsvp",
                            lambda: self.events.rsvp(party.event_id, member, choice))
        return str(result or "")

    # ── START ──
    def start_party(self, party: Party, opener_track: dict[str, Any] | None = None) -> bool:
        """START: ping the group, DJ opens. Returns False when not startable."""
        if party.state not in ("invited", "planned", "paused"):
            return False
        party.state = "live"
        party.started_at = time.time()
        self._say(party, f"🎉 **{party.title}** is LIVE! "
                         f"Theme: {party.theme or 'open'} — drop requests, vote, play.")
        opener = opener_track or {"id": "trk_opener", "title": "Party opener",
                                "by": party.host}
        line = self._safe(party, "dj.open", lambda: self.dj_opener(opener)
                          if self.dj_opener is not None else None)
        if not line:
            line = f"🎧 DJ opens with **{opener.get('title', 'the opener')}** — let's go!"
        self._say(party, str(line))
        party.current_track = dict(opener)
        self.store.put(party)
        return True

    def pause_party(self, party: Party) -> bool:
        if party.state != "live":
            return False
        party.state = "paused"
        self._say(party, "⏸️ Party paused — the music waits. Say go to resume.")
        self.store.put(party)
        return True

    def resume_party(self, party: Party) -> bool:
        if party.state != "paused":
            return False
        return self.start_party(party)

    # ── MUSIC ──
    def request_track(self, party: Party, member: str, title: str) -> dict[str, Any] | None:
        """MUSIC: queue a track request. Per-member flood cap enforced."""
        if party.state != "live" or not (title or "").strip():
            return None
        mine = [t for t in party.queue if t.get("by") == member]
        if len(mine) >= TRACK_REQUEST_LIMIT:
            self._say(party, f"@{member} — request cap reached ({TRACK_REQUEST_LIMIT}). "
                             "Vote on what's queued instead! 🗳️")
            return None
        track = {"id": "trk_" + new_short_id(),
                 "title": title.strip()[:160],
                 "by": member,
                 "requested_at": time.time(),
                 "votes": 0}
        party.queue.append(track)
        party.votes.setdefault(track["id"], [])
        self.store.put(party)
        return track

    def vote_track(self, party: Party, member: str, track_id: str) -> bool:
        """Vote for a queued track. One vote per member per track."""
        if party.state != "live":
            return False
        for t in party.queue:
            if t["id"] == track_id or t["id"].startswith(track_id):
                voters = party.votes.setdefault(t["id"], [])
                if member not in voters:
                    voters.append(member)
                    t["votes"] = len(voters)
                    self.store.put(party)
                return True
        return False

    def next_track(self, party: Party) -> dict[str, Any] | None:
        """Advance the queue: highest-voted track plays next."""
        if party.state != "live":
            return None
        ranked = party.queue_ranked()
        if not ranked:
            return None
        nxt = ranked[0]
        if party.current_track:
            party.played.append(party.current_track)
        party.queue = [t for t in party.queue if t["id"] != nxt["id"]]
        party.current_track = dict(nxt)
        self._say(party, f"🎵 Now playing: **{nxt.get('title')}** "
                         f"(requested by {nxt.get('by')}, "
                         f"{len(party.votes.get(nxt['id'], []))} votes)")
        self.store.put(party)
        return dict(nxt)

    # ── GAMES ──
    def play_trivia_round(self, party: Party) -> dict[str, Any] | None:
        """GAMES: one trivia round between tracks. Fail-soft by design.

        Returns the question dict (with ``round`` + ``party_id``) or None
        when no provider / no question / provider failed. A raising
        provider is recorded in ``party.failures`` and the party lives on.
        """
        if party.state != "live":
            return None
        if self.trivia_provider is None:
            return None
        question = self._safe(party, "trivia.provide",
                              lambda: self.trivia_provider(party))
        if not isinstance(question, dict) or not question.get("question"):
            return None
        party.trivia_rounds += 1
        question = dict(question)
        question["round"] = party.trivia_rounds
        question["party_id"] = party.id
        self._say(party, f"🧠 Trivia round {party.trivia_rounds}: "
                         f"{question.get('question')}\n"
                         + "\n".join(f"  {i + 1}. {opt}" for i, opt in
                                     enumerate(question.get("options") or [])))
        self.store.put(party)
        return question

    def answer_trivia(self, party: Party, member: str,
                      question: dict[str, Any], choice: int) -> dict[str, Any]:
        """Score a trivia answer. Streak bonus, achievements, economy hook.

        ``question`` is the dict returned by :meth:`play_trivia_round`.
        Never raises; a bad question just scores nothing.
        """
        out: dict[str, Any] = {"correct": False, "points": 0, "streak": 0}
        if party.state != "live":
            return out
        try:
            correct_idx = question.get("answer_index")
            if correct_idx is None or int(choice) != int(correct_idx):
                party.trivia_streaks[member] = 0
                self._say(party, f"❌ Not quite, {member}.")
                self.store.put(party)
                return out
            streak = party.trivia_streaks.get(member, 0) + 1
            party.trivia_streaks[member] = streak
            bonus = min(STREAK_BONUS_CAP, (streak - 1) * STREAK_BONUS_PER)
            points = TRIVIA_POINTS + bonus
            party.trivia_scores[member] = party.trivia_scores.get(member, 0) + points
            out = {"correct": True, "points": points, "streak": streak}
            msg = f"✅ {member} +{points} pts" + (f" 🔥 {streak} streak!" if streak >= 2 else "")
            self._say(party, msg)
            self._check_achievements(party, member, streak)
            if not party.rehearsal and self.economy_hook is not None:
                credited = self._safe(party, "economy.credit", lambda: self.economy_hook(
                    party.group_id, member, points, f"party trivia: {party.title}"))
                if credited:
                    party.economy_credited[member] = \
                        party.economy_credited.get(member, 0) + points
            self.store.put(party)
            return out
        except Exception as exc:  # noqa: BLE001 — scoring never raises
            _log.debug("answer_trivia failed: %s", exc)
            party.failures.append(f"trivia.answer: {type(exc).__name__}")
            return out

    def _check_achievements(self, party: Party, member: str, streak: int) -> None:
        """SCORE: achievements from trivia play. Pure + fail-soft."""
        def _grant(kind: str, label: str) -> None:
            if not any(a.get("member") == member and a.get("kind") == kind
                       for a in party.achievements):
                party.achievements.append(
                    {"member": member, "kind": kind, "label": label,
                     "at": time.time()})
                self._say(party, f"🏅 {member} unlocked **{label}**!")

        try:
            # Any correct answer: first one in this party earns First Win.
            _grant("first_win", "First Win")
            if streak >= 3:
                _grant("streak_3", "On Fire — 3 streak")
            if streak >= 5:
                _grant("streak_5", "Unstoppable — 5 streak")
        except Exception as exc:  # noqa: BLE001
            party.failures.append(f"achievements: {type(exc).__name__}")

    # ── WALL ──
    def post_to_wall(self, party: Party, member: str, text: str = "",
                     kind: str = "text", media_ref: str = "") -> dict[str, Any] | None:
        """WALL: a moment for the recap. ``kind``: text | photo | video | voice."""
        if party.state not in ("live", "paused"):
            return None
        if not (text or "").strip() and not media_ref:
            return None
        post = {"id": "wall_" + new_short_id(),
                "author": member, "by": member,
                "kind": kind if kind in ("text", "photo", "video", "voice") else "text",
                "text": (text or "").strip()[:1000],
                "media_ref": (media_ref or "").strip()[:300],
                "reactions": {},
                "at": time.time()}
        party.wall.append(post)
        self.store.put(party)
        return post

    def react_to_wall(self, party: Party, member: str, post_id: str,
                      emoji: str = "❤️") -> bool:
        for post in party.wall:
            if post["id"] == post_id or post["id"].startswith(post_id):
                post.setdefault("reactions", {}).setdefault(emoji, [])
                if member not in post["reactions"][emoji]:
                    post["reactions"][emoji].append(member)
                self.store.put(party)
                return True
        return False

    # ── END + NEXT DAY ──
    def end_party(self, party: Party) -> PartyRecap:
        """END: auto-recap, achievements for MVP/top voter, then close."""
        if party.state == "ended" and party.recap:
            return PartyRecap.from_dict(party.recap)
        # MVP + top voter achievements (fail-soft: best effort)
        self._safe(party, "score.mvp", lambda: self._final_achievements(party))
        recap = PartyRecap(
            party_id=party.id,
            title=party.title,
            rehearsal=party.rehearsal,
            started_at=party.started_at,
            ended_at=time.time(),
            attendees=party.attendees(),
            top_tracks=[{"title": t.get("title"), "by": t.get("by"),
                         "votes": len(party.votes.get(t.get("id", ""), []))}
                        for t in sorted(party.played + ([party.current_track]
                                        if party.current_track else []),
                                        key=lambda t: -len(party.votes.get(t.get("id", ""), [])))[:5]],
            top_players=[{"member": m, "points": pts,
                          "streak": party.trivia_streaks.get(m, 0)}
                         for m, pts in sorted(party.trivia_scores.items(),
                                              key=lambda kv: -kv[1])[:5]],
            top_wall=[{"author": w.get("author"), "text": (w.get("text") or "")[:140],
                       "kind": w.get("kind"),
                       "reactions": sum(len(v) for v in (w.get("reactions") or {}).values())}
                      for w in sorted(party.wall,
                                      key=lambda w: -sum(len(v) for v in (w.get("reactions") or {}).values()))[:5]],
            achievements=list(party.achievements),
            failures=list(party.failures),
            economy_credited=dict(party.economy_credited),
        )
        party.state = "ended"
        party.ended_at = recap.ended_at
        party.recap = recap.to_dict()
        self._say(party, format_recap(recap))
        self.store.put(party)
        return recap

    def _final_achievements(self, party: Party) -> None:
        if party.trivia_scores:
            mvp = max(party.trivia_scores.items(), key=lambda kv: kv[1])[0]
            if not any(a.get("member") == mvp and a.get("kind") == "mvp"
                       for a in party.achievements):
                party.achievements.append({"member": mvp, "kind": "mvp",
                                           "label": "Party MVP", "at": time.time()})
        if party.votes:
            voter_counts: dict[str, int] = {}
            for voters in party.votes.values():
                for v in voters:
                    voter_counts[v] = voter_counts.get(v, 0) + 1
            if voter_counts:
                top = max(voter_counts.items(), key=lambda kv: kv[1])[0]
                if not any(a.get("member") == top and a.get("kind") == "top_voter"
                           for a in party.achievements):
                    party.achievements.append(
                        {"member": top, "kind": "top_voter",
                         "label": "Top Voter — the people's DJ", "at": time.time()})

    def last_night_card(self, group_id: str, now: float | None = None) -> str:
        """NEXT DAY: the last-night card for the most recent ended party."""
        now = now if now is not None else time.time()
        ended = [p for p in self.store.load(group_id) if p.state == "ended" and p.recap]
        if not ended:
            return "No party last night — the dance floor is waiting. 🪩"
        latest = max(ended, key=lambda p: p.ended_at or 0.0)
        if now - latest.ended_at > 36 * 3600:
            return "No party last night — the dance floor is waiting. 🪩"
        return format_last_night_card(PartyRecap.from_dict(latest.recap))


# ── formatting ─────────────────────────────────────────────────────────────


def format_recap(recap: PartyRecap) -> str:
    """The END message: top tracks, top players, top wall moments."""
    lines = [f"🎉 **{recap.title}** — that's a wrap!"]
    if recap.rehearsal:
        lines.append("🧪 Rehearsal — scores counted, points were dry-run.")
    if recap.top_tracks:
        lines.append("\n🎵 **Top tracks:**")
        for i, t in enumerate(recap.top_tracks, 1):
            lines.append(f"  {i}. {t.get('title')} — {t.get('by')} ({t.get('votes')} votes)")
    if recap.top_players:
        lines.append("\n🧠 **Top players:**")
        medals = ["🥇", "🥈", "🥉"]
        for i, pl in enumerate(recap.top_players, 1):
            medal = medals[i - 1] if i <= 3 else f"{i}."
            lines.append(f"  {medal} {pl.get('member')} — {pl.get('points')} pts")
    if recap.top_wall:
        lines.append("\n🧱 **Wall moments:**")
        for w in recap.top_wall:
            txt = (w.get("text") or f"[{w.get('kind')}]")[:120]
            lines.append(f"  {w.get('author')}: {txt} ({w.get('reactions')} ❤️)")
    if recap.achievements:
        lines.append("\n🏅 **Achievements:** " +
                     ", ".join(f"{a.get('member')} — {a.get('label')}"
                               for a in recap.achievements))
    if recap.attendees:
        lines.append(f"\n👥 {len(recap.attendees)} in the room.")
    # The recap is the party's loudest group surface: it renders in
    # the hype_man register (group_extras, AA-40) — greeting, emoji
    # frame, sign-off — while the recap body itself is untouched.
    from .group_extras import render_in_register

    return render_in_register("\n".join(lines), "hype_man")


def format_last_night_card(recap: PartyRecap) -> str:
    """The NEXT DAY card: one compact look back at last night."""
    when = time.strftime("%a %H:%M", time.localtime(recap.ended_at)) if recap.ended_at else ""
    lines = [f"🌅 **Last night: {recap.title}** ({when})"]
    if recap.top_players:
        top = recap.top_players[0]
        lines.append(f"🏆 MVP: {top.get('member')} — {top.get('points')} pts")
    if recap.top_tracks:
        lines.append(f"🎵 Most loved track: {recap.top_tracks[0].get('title')}")
    if recap.top_wall:
        lines.append(f"🧱 Moment of the night: {recap.top_wall[0].get('text') or '[media]'} "
                     f"— {recap.top_wall[0].get('author')}")
    lines.append(f"👥 {len(recap.attendees)} showed up. Next one soon? 🪩")
    return "\n".join(lines)


# ── chat surface: /party (AA-05) ─────────────────────────────────────────
#
# The PartyManager above (and the PartyQuick fast variant) shipped with
# no caller — this control is their production surface, following the
# community control contract (never raises, group-only, state under
# the community dir). Two production hooks are wired by default:
#
# * trivia — the group's OWN quiz mini-apps are the question bank
#   (a party round is a deterministic pick over the bank, so an
#   answer can be re-scored later without storing the question);
# * economy — trivia credits land in the ONE group economy (§36)
#   through the TokenEconomy policy facade, at the same data root
#   the runtime's economy surface uses. Rehearsal parties (the first
#   REHEARSAL_COUNT per group) stay dry-run, as the manager intends.

_PARTY_HELP = (
    "/party — group parties (DJ queue, votes, trivia, wall, recap)\n"
    "/party new <title> [| <theme>] [in 2h]   •   /party list\n"
    "/party show   •   /party rsvp yes|no|maybe   •   /party start\n"
    "/party pause   •   /party resume   •   /party end\n"
    "/party request <track title>   •   /party vote <track id>\n"
    "/party next   •   /party trivia   •   /party answer <number>\n"
    "/party wall <text>   •   /party react <post id> <emoji>\n"
    "/party lastnight\n"
    "Fast variant: /party quick new <title> · start · request · "
    "vote <n> · next · wall · end")


def _party_root(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    root = getattr(settings, "community_dir", None)
    return Path(root) if root else None


def _economy_root(context: Any) -> Path | None:
    """Where THIS deployment's group economy lives (one purse, §36)."""
    root = _party_root(context)
    if root is not None:
        return root / "group_economy"
    settings = getattr(context, "settings", None)
    workspace = getattr(settings, "workspace_dir", None)
    if workspace:
        # The root GroupSurface itself uses: <workspace>/data/social.
        return Path(workspace) / "data" / "social" / "group_economy"
    return None


def party_economy_hook(context: Any = None):
    """The production economy hook for PartyManager / PartyQuick.

    Credits through :func:`persistent_token_economy` so party points
    are group-economy coins — same ledger, same caps, no parallel
    balance. Returns False (no credit) on any failure; the managers
    treat the hook as best-effort by design.
    """
    root = _economy_root(context)

    def hook(group_id: str, member: str, amount: int, reason: str) -> bool:
        try:
            from .token_economy import persistent_token_economy

            economy = persistent_token_economy(group_id, data_dir=root)
            return economy.earn(member, int(amount), "party",
                                reason=reason) is not None
        except Exception as exc:  # best-effort by the manager's design
            _log.warning("party economy hook failed: %s", exc)
            return False

    return hook


def _quiz_bank(context: Any, group_id: str) -> list[dict[str, Any]]:
    """The group's quiz mini-apps as a trivia bank (options + answer)."""
    try:
        from .miniapps import MiniAppStore

        root = _party_root(context)
        store = MiniAppStore(data_dir=root) if root else MiniAppStore()
        bank: list[dict[str, Any]] = []
        for app in store.load(group_id):
            if app.kind != "quiz":
                continue
            options = [str(o) for o in (app.state.get("options") or [])]
            correct = app.state.get("correct")
            if (len(options) >= 2 and isinstance(correct, int)
                    and 0 <= correct < len(options)):
                bank.append({"question": app.title, "options": options,
                             "answer_index": correct})
        return bank
    except Exception as exc:  # trivia is an enhancement, never a gate
        _log.debug("party quiz bank unavailable: %s", exc)
        return []


def _quiz_question(bank: list[dict[str, Any]], party_id: str,
                   round_no: int) -> dict[str, Any] | None:
    """Deterministic pick: the same (party, round) always re-derives
    the same question, so /party answer needs no stored question."""
    if not bank or round_no < 1:
        return None
    digest = hashlib.sha256(f"{party_id}:{round_no}".encode()).hexdigest()
    return bank[int(digest[:8], 16) % len(bank)]


def _party_manager(context: Any) -> PartyManager:
    root = _party_root(context)
    store = PartyStore(root / "parties") if root else PartyStore()
    events = EventStore(root / "events") if root else EventStore()

    def trivia(party: Party) -> dict[str, Any] | None:
        bank = _quiz_bank(context, party.group_id)
        return _quiz_question(bank, party.id, party.trivia_rounds + 1)

    return PartyManager(store=store, event_store=events,
                        trivia_provider=trivia,
                        economy_hook=party_economy_hook(context))


def _active_party(store: PartyStore, group_id: str) -> Party | None:
    live = store.live(group_id)
    if live is not None:
        return live
    open_parties = [p for p in store.load(group_id) if p.state != "ended"]
    return open_parties[-1] if open_parties else None


def _drain(store: PartyStore, party: Party) -> list[str]:
    lines = [str(m) for m in party.outbox]
    if party.outbox:
        party.outbox = []
        store.put(party)
    return lines


def _render_party(party: Party) -> str:
    lines = [f"🎉 **{party.title}** — {party.state}"
             + (" (rehearsal)" if party.rehearsal else ""),
             f"Host: {party.host}"
             + (f" · theme: {party.theme}" if party.theme else "")]
    if party.current_track:
        lines.append(f"🎵 Now: {party.current_track.get('title')}")
    ranked = party.queue_ranked()
    if ranked:
        lines.append("Queue:")
        for track in ranked[:6]:
            votes = len(party.votes.get(track["id"], []))
            lines.append(f"• {track.get('title')} — {votes} vote(s) "
                         f"· `{track['id']}`")
    if party.trivia_scores:
        top = sorted(party.trivia_scores.items(), key=lambda kv: -kv[1])[:5]
        lines.append("🏆 Trivia: " + ", ".join(f"{m} {s}" for m, s in top))
    lines.append(f"🧱 Wall: {len(party.wall)} post(s) · `{party.id}`")
    return "\n".join(lines)


def control_party(tail: str = "", context: Any = None, chat: Any = None,
                  sender_id: str = "", sender: str = "") -> str:
    """`/party …` — the community party surface. Never raises."""
    if chat is None or getattr(chat, "kind", "dm") != "group":
        return "Parties live in group chats — run /party in a group. 🎉"
    try:
        return _control_party((tail or "").strip(), context, chat,
                              sender_id, sender)
    except Exception as exc:  # the surface never raises into chat
        _log.warning("party control failed: %s", exc)
        return "The party hit a snag — try again."


def _control_party(tail: str, context: Any, chat: Any,
                   sender_id: str, sender: str) -> str:
    from .miniapps import _group_key

    group_id = _group_key(chat)
    who = sender_id or sender or "anon"
    words = tail.split()
    sub = words[0].lower() if words else "help"
    rest = words[1:]

    if sub in ("help", ""):
        return _PARTY_HELP
    if sub == "quick":
        return _control_party_quick(" ".join(rest), context, group_id, who)

    manager = _party_manager(context)
    store = manager.store

    if sub == "new":
        body = " ".join(rest)
        delay = 3600.0
        match = re.search(r"\bin\s+(\d+)\s*(h|m)\s*$", body, re.IGNORECASE)
        if match:
            delay = int(match.group(1)) * (3600 if match.group(2).lower() == "h" else 60)
            body = body[:match.start()].strip()
        title, _, theme = body.partition("|")
        title = title.strip()
        if not title:
            return "Name the party: /party new <title> [| <theme>] [in 2h]"
        party = manager.create_party(group_id, title, time.time() + delay,
                                     who, theme=theme.strip())
        if party is None:
            return "Couldn't create that party — try a different title."
        lines = _drain(store, party)
        return "\n".join(lines) or f"🎉 **{title}** is on the calendar."

    if sub == "list":
        parties = store.load(group_id)
        if not parties:
            return "No parties here yet — /party new <title> starts one."
        lines = ["🎉 Parties in this group:"]
        for party in parties[-8:]:
            lines.append(f"• {party.title} — {party.state} · `{party.id}`")
        return "\n".join(lines)

    if sub == "lastnight":
        return manager.last_night_card(group_id)

    party = _active_party(store, group_id)
    if party is None:
        return "No party running here — /party new <title> starts one."

    if sub == "show":
        return _render_party(party)
    if sub == "rsvp":
        choice = rest[0].lower() if rest else ""
        if choice not in ("yes", "no", "maybe"):
            return "RSVP with: /party rsvp yes|no|maybe"
        result = manager.rsvp(party, who, choice)
        lines = _drain(store, party)
        return "\n".join(lines + ([result] if result else [])) or \
            f"RSVP noted: {choice}."
    if sub == "start":
        if not manager.start_party(party):
            return f"Can't start from '{party.state}' — see /party show."
        return "\n".join(_drain(store, party))
    if sub == "pause":
        if not manager.pause_party(party):
            return "Only a live party can pause."
        return "\n".join(_drain(store, party))
    if sub == "resume":
        if not manager.resume_party(party):
            return "Only a paused party can resume."
        return "\n".join(_drain(store, party))
    if sub == "request":
        title = " ".join(rest).strip()
        if not title:
            return "Request what? /party request <track title>"
        track = manager.request_track(party, who, title)
        lines = _drain(store, party)
        if track is None:
            return "\n".join(lines) or "Requests open when the party is live."
        return (f"🎶 Queued **{track['title']}** — vote with "
                f"/party vote {track['id']}")
    if sub == "vote":
        if not rest:
            return "Vote with: /party vote <track id> (ids in /party show)"
        if not manager.vote_track(party, who, rest[0]):
            return "No such queued track here — /party show lists them."
        return "🗳️ Vote counted."
    if sub == "next":
        track = manager.next_track(party)
        lines = _drain(store, party)
        if track is None:
            return "\n".join(lines) or "Queue's empty — /party request a track."
        return "\n".join(lines)
    if sub == "trivia":
        question = manager.play_trivia_round(party)
        lines = _drain(store, party)
        if question is None:
            return ("No trivia bank here yet — party trivia draws on this "
                    "group's quiz mini-apps (/miniapp new quiz …).")
        return "\n".join(lines)
    if sub == "answer":
        if not rest or not rest[0].isdigit():
            return "Answer with: /party answer <number>"
        bank = _quiz_bank(context, group_id)
        question = _quiz_question(bank, party.id, party.trivia_rounds)
        if question is None:
            return "No trivia round is open — /party trivia starts one."
        result = manager.answer_trivia(party, who, question,
                                       int(rest[0]) - 1)
        lines = _drain(store, party)
        return "\n".join(lines) or ("✅ Correct!" if result.get("correct")
                                    else "❌ Not that one.")
    if sub == "wall":
        text = " ".join(rest).strip()
        if not text:
            return "Post with: /party wall <text>"
        post = manager.post_to_wall(party, who, text)
        if post is None:
            return "The wall opens when the party is live or paused."
        return f"🧱 Posted to the wall · `{post['id']}`"
    if sub == "react":
        if len(rest) < 2:
            return "React with: /party react <post id> <emoji>"
        if not manager.react_to_wall(party, who, rest[0], rest[1]):
            return "No wall post with that id here."
        return "❤️ Reaction added."
    if sub == "end":
        recap = manager.end_party(party)
        lines = _drain(store, party)
        return "\n".join(lines) or format_recap(recap)
    return _PARTY_HELP


def _control_party_quick(tail: str, context: Any, group_id: str,
                         who: str) -> str:
    """`/party quick …` — the PartyQuick fast variant, same surface."""
    from .parties_quick import PartyQuick, QuickPartyStore

    root = _party_root(context)
    store = QuickPartyStore(root / "parties_quick") if root else QuickPartyStore()
    outbox: list[str] = []
    cell: dict[str, Any] = {}

    def trivia() -> dict[str, Any] | None:
        party = cell.get("party")
        if party is None:
            return None
        bank = _quiz_bank(context, group_id)
        question = _quiz_question(bank, party.party_id,
                                  party.trivia_rounds + 1)
        if question is None:
            return None
        return {"q": question["question"], "options": question["options"]}

    quick = PartyQuick(trivia_provider=trivia,
                       economy_hook=party_economy_hook(context),
                       say=outbox.append)
    words = tail.split()
    sub = words[0].lower() if words else "help"
    rest = words[1:]

    if sub in ("help", ""):
        return ("Quick party: /party quick new <title> · start · "
                "request <track> · vote <number> · next · trivia · "
                "answer <number> · wall <text> · end")

    if sub == "new":
        title = " ".join(rest).strip()
        if not title:
            return "Name it: /party quick new <title>"
        rehearsal = len(store.load(group_id)) < REHEARSAL_COUNT
        party = quick.create(group_id, title, who, mixes=[],
                             rehearsal=rehearsal)
        # PartyQuick numbers parties per manager instance; the store
        # is the durable namespace, so the persisting caller allocates
        # a collision-free id (two chats must never overwrite parties).
        party.party_id = "qp-" + new_short_id()
        store.put(party)
        note = " (rehearsal — economy dry-run)" if rehearsal else ""
        return (f"⚡ Quick party **{title}** created{note} — "
                f"/party quick start when ready.")

    party = store.live(group_id)
    if party is None:
        return "No quick party here — /party quick new <title> starts one."
    cell["party"] = party

    if sub == "start":
        if not quick.start(party):
            return "That quick party can't start now."
        store.put(party)
        return f"⚡ **{party.title}** is LIVE — request tracks!"
    if sub == "request":
        title = " ".join(rest).strip()
        track = quick.request_track(party, who, title) if title else None
        if track is None:
            return "Requests open when the quick party is live."
        store.put(party)
        return (f"🎶 Queued **{title}** — it's #{len(party.queue)}; "
                f"vote with /party quick vote {len(party.queue)}")
    if sub == "vote":
        if not rest or not rest[0].isdigit():
            return "Vote with: /party quick vote <number> (queue order)"
        if not quick.vote_track(party, who, int(rest[0]) - 1):
            return "No track at that queue position."
        store.put(party)
        return "🗳️ Vote counted."
    if sub == "next":
        track = quick.next_track(party)
        if track is None:
            return "Queue's empty — /party quick request a track."
        store.put(party)
        lines = [f"🎵 Now playing: **{track['title']}**"]
        lines.extend(outbox)  # an auto trivia round announces itself here
        return "\n".join(lines)
    if sub == "trivia":
        question = quick.play_trivia_round(party)
        if question is None:
            return ("No trivia bank here yet — quick-party trivia draws on "
                    "this group's quiz mini-apps (/miniapp new quiz …).")
        store.put(party)
        opts = "\n".join(f"{i}. {o}" for i, o in
                         enumerate(question["options"], 1))
        return f"🧠 Round {question['round']}: {question['question']}\n{opts}"
    if sub == "answer":
        if not rest or not rest[0].isdigit():
            return "Answer with: /party quick answer <number>"
        bank = _quiz_bank(context, group_id)
        question = _quiz_question(bank, party.party_id, party.trivia_rounds)
        if question is None:
            return "No trivia round is open — /party quick trivia starts one."
        picked = int(rest[0]) - 1
        options = question["options"]
        answer_text = options[picked] if 0 <= picked < len(options) else ""
        correct_text = options[question["answer_index"]]
        result = quick.answer_trivia(party, who, answer_text, correct_text)
        store.put(party)
        if not result.get("ok"):
            return "The quick party isn't live."
        return ("✅ Correct! +10" if result.get("correct")
                else f"❌ It was: {correct_text}")
    if sub == "wall":
        text = " ".join(rest).strip()
        if not text:
            return "Post with: /party quick wall <text>"
        quick.post_to_wall(party, who, text)
        store.put(party)
        return "🧱 Posted to the wall."
    if sub == "end":
        recap = quick.end(party)
        store.put(party)
        lines = [f"⚡ **{recap.title}** — that's a wrap."]
        if recap.top_players:
            lines.append("🏆 Top players: " + ", ".join(
                f"{p['member']} {p['score']}" for p in recap.top_players))
        if recap.trivia_rounds:
            lines.append(f"🧠 {recap.trivia_rounds} trivia round(s) played.")
        if recap.economy_dry_run:
            lines.append("Rehearsal party — economy was dry-run.")
        return "\n".join(lines)
    return "Quick party: /party quick new <title> · start · request · end"
