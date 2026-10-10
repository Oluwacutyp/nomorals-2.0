"""Live audio rooms for communities — host → speakers → hand-raise → listeners.

Clubhouse/Spaces-style rooms scoped to existing communities (FB Live Audio
Rooms pattern). Each community gets its own audio space; rooms never cross
the community isolation boundary.

ISOLATION CONTRACT (same as the rest of ``nomorals.community``):
* No imports of ``nomorals.memory``, ``nomorals.accounts``, any vault, or
  any private connector. Participants are platform user ids / display names
  only.
* Room state lives under ``~/.devon/community/audio_rooms/``.
* Money (tips) goes through an *injectable* ``tip_fn`` seam — the module
  never touches Paystack or any money primitive directly. Without a
  configured tip function, tipping honestly reports "not wired up".

Platform audio (Telegram voice chats, web-room fallback) goes through an
injectable ``platform`` seam. The default seam is honest: it records intent
and reports what the platform would need, never pretending to control a
voice chat it cannot reach.

Recording is opt-in per room with always-visible state (🔴 REC in every
render). When a recording ends, a transcript is generated through an
injectable ``transcriber`` seam (WhisperX → #103 transcript editing in
production). Live captions are ON by default (accessibility, not a feature).

Every public function never raises — chat must never blow up.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)

# ── storage ────────────────────────────────────────────────────────────────


def _default_dir() -> Path:
    d = Path.home() / ".devon" / "community" / "audio_rooms"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return d


def _default_data_dir() -> Path:
    return _default_dir()


# ── models ─────────────────────────────────────────────────────────────────


@dataclass
class Participant:
    """One person in a room. Identity = platform user id + display name only."""

    user_id: str = ""
    name: str = ""
    muted: bool = False
    hand_raised_at: float = 0.0
    note: str = ""  # hand-raise note ("Q about the venue") shown to hosts

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "muted": self.muted,
            "hand_raised_at": self.hand_raised_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Participant":
        data = data or {}
        return cls(
            user_id=str(data.get("user_id", "") or ""),
            name=str(data.get("name", "") or ""),
            muted=bool(data.get("muted", False)),
            hand_raised_at=float(data.get("hand_raised_at", 0.0) or 0.0),
            note=str(data.get("note", "") or ""),
        )


@dataclass
class Tip:
    """One tip. Amount in kobo. Recorded, never moved here."""

    tip_id: str = ""
    room_id: str = ""
    from_id: str = ""
    from_name: str = ""
    to_id: str = ""
    to_name: str = ""
    amount_kobo: int = 0
    status: str = "recorded"  # recorded | sent | failed
    note: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tip_id": self.tip_id,
            "room_id": self.room_id,
            "from_id": self.from_id,
            "from_name": self.from_name,
            "to_id": self.to_id,
            "to_name": self.to_name,
            "amount_kobo": self.amount_kobo,
            "status": self.status,
            "note": self.note,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tip":
        data = data or {}
        return cls(
            tip_id=str(data.get("tip_id", "") or ""),
            room_id=str(data.get("room_id", "") or ""),
            from_id=str(data.get("from_id", "") or ""),
            from_name=str(data.get("from_name", "") or ""),
            to_id=str(data.get("to_id", "") or ""),
            to_name=str(data.get("to_name", "") or ""),
            amount_kobo=max(0, int(data.get("amount_kobo", 0) or 0)),
            status=str(data.get("status", "recorded") or "recorded"),
            note=str(data.get("note", "") or ""),
            created_at=float(data.get("created_at", 0.0) or 0.0),
        )


@dataclass
class Room:
    """A live audio room. Scoped to one community (``community_id``)."""

    room_id: str = ""
    community_id: str = ""
    title: str = ""
    host_id: str = ""
    host_name: str = ""
    cohosts: list[Participant] = field(default_factory=list)  # trusted speakers w/ host powers
    speakers: list[Participant] = field(default_factory=list)
    hand_raise_queue: list[Participant] = field(default_factory=list)
    listeners: list[Participant] = field(default_factory=list)
    lobby: list[Participant] = field(default_factory=list)  # green-room pre-check-ins
    blocked: list[str] = field(default_factory=list)  # user_ids
    locked: bool = False  # lock rejects new joins (sabha- pattern)
    invite_codes: dict[str, str] = field(default_factory=dict)  # code → "speaker"|"listener"
    reactions: dict[str, int] = field(default_factory=dict)  # emoji → count
    agenda: str = ""  # what this room is about / running order
    reports: list[dict[str, Any]] = field(default_factory=list)
    recording: bool = False          # opt-in
    recording_visible: bool = False  # always True when recording
    captions: bool = True            # on by default (accessibility)
    caption_log: list[str] = field(default_factory=list)
    tips: list[Tip] = field(default_factory=list)
    platform: str = "telegram"       # telegram | web
    platform_ref: str = ""           # voice-chat id / web room url
    state: str = "live"              # live | ended | scheduled
    hand_raise_expiry: float = 1800.0  # stale hand-raises auto-expire (0 = never)
    starts_at: float = 0.0  # scheduled start epoch; 0 = starts immediately
    announced: bool = False  # scheduled-start reminder already queued
    peak_headcount: int = 0  # analytics: max concurrent participants
    transcript: str = ""
    created_at: float = 0.0
    ended_at: float = 0.0

    # ── derived ──

    def is_cohost(self, user_id: str) -> bool:
        return any(p.user_id == (user_id or "").strip() for p in self.cohosts)

    def find(self, user_id: str) -> tuple[str, Participant | None]:
        """Return (role, participant) — role in host/cohost/speaker/queue/listener/lobby/''."""
        uid = (user_id or "").strip()
        if not uid:
            return "", None
        if uid == self.host_id:
            return "host", Participant(user_id=self.host_id, name=self.host_name)
        for p in self.cohosts:
            if p.user_id == uid:
                return "cohost", p
        for p in self.speakers:
            if p.user_id == uid:
                return "speaker", p
        for p in self.hand_raise_queue:
            if p.user_id == uid:
                return "queue", p
        for p in self.listeners:
            if p.user_id == uid:
                return "listener", p
        for p in self.lobby:
            if p.user_id == uid:
                return "lobby", p
        return "", None

    def is_blocked(self, user_id: str) -> bool:
        return (user_id or "").strip() in (self.blocked or [])

    def headcount(self) -> int:
        return 1 + len(self.speakers) + len(self.listeners)

    def bump_peak(self) -> None:
        h = self.headcount()
        if h > self.peak_headcount:
            self.peak_headcount = h

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "community_id": self.community_id,
            "title": self.title,
            "host_id": self.host_id,
            "host_name": self.host_name,
            "cohosts": [p.to_dict() for p in self.cohosts],
            "speakers": [p.to_dict() for p in self.speakers],
            "hand_raise_queue": [p.to_dict() for p in self.hand_raise_queue],
            "listeners": [p.to_dict() for p in self.listeners],
            "lobby": [p.to_dict() for p in self.lobby],
            "blocked": list(self.blocked),
            "locked": self.locked,
            "invite_codes": dict(self.invite_codes),
            "reactions": dict(self.reactions),
            "agenda": self.agenda,
            "reports": list(self.reports),
            "recording": self.recording,
            "recording_visible": self.recording_visible,
            "captions": self.captions,
            "caption_log": list(self.caption_log[-50:]),
            "tips": [t.to_dict() for t in self.tips],
            "platform": self.platform,
            "platform_ref": self.platform_ref,
            "state": self.state,
            "hand_raise_expiry": self.hand_raise_expiry,
            "starts_at": self.starts_at,
            "announced": self.announced,
            "peak_headcount": self.peak_headcount,
            "transcript": self.transcript,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Room":
        data = data or {}
        return cls(
            room_id=str(data.get("room_id", "") or ""),
            community_id=str(data.get("community_id", "") or ""),
            title=str(data.get("title", "") or ""),
            host_id=str(data.get("host_id", "") or ""),
            host_name=str(data.get("host_name", "") or ""),
            cohosts=[Participant.from_dict(p) for p in (data.get("cohosts") or [])],
            speakers=[Participant.from_dict(p) for p in (data.get("speakers") or [])],
            hand_raise_queue=[Participant.from_dict(p) for p in (data.get("hand_raise_queue") or [])],
            listeners=[Participant.from_dict(p) for p in (data.get("listeners") or [])],
            lobby=[Participant.from_dict(p) for p in (data.get("lobby") or [])],
            blocked=[str(u) for u in (data.get("blocked") or [])],
            locked=bool(data.get("locked", False)),
            invite_codes={str(k): str(v) for k, v in (data.get("invite_codes") or {}).items()},
            reactions={str(k): int(v) for k, v in (data.get("reactions") or {}).items()},
            agenda=str(data.get("agenda", "") or ""),
            reports=list(data.get("reports") or []),
            recording=bool(data.get("recording", False)),
            recording_visible=bool(data.get("recording_visible", False)),
            captions=bool(data.get("captions", True)),
            caption_log=[str(c) for c in (data.get("caption_log") or [])],
            tips=[Tip.from_dict(t) for t in (data.get("tips") or [])],
            platform=str(data.get("platform", "telegram") or "telegram"),
            platform_ref=str(data.get("platform_ref", "") or ""),
            state=str(data.get("state", "live") or "live"),
            hand_raise_expiry=float(data.get("hand_raise_expiry", 1800.0) or 0.0),
            starts_at=float(data.get("starts_at", 0.0) or 0.0),
            announced=bool(data.get("announced", False)),
            peak_headcount=int(data.get("peak_headcount", 0) or 0),
            transcript=str(data.get("transcript", "") or ""),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            ended_at=float(data.get("ended_at", 0.0) or 0.0),
        )


# ── store ──────────────────────────────────────────────────────────────────


class RoomStore:
    """JSON-file room storage under the community data dir. Never raises."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        try:
            self._dir = Path(data_dir) if data_dir else _default_dir()
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            self._dir = Path("/tmp") / "devon_audio_rooms"

    def _path(self, room_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "", room_id or "")[:64]
        return self._dir / f"{safe or 'room'}.json"

    def _index_path(self) -> Path:
        return self._dir / "_index.json"

    def _load_index(self) -> list[str]:
        try:
            raw = self._index_path().read_text(encoding="utf-8")
            ids = json.loads(raw)
            return [str(i) for i in ids] if isinstance(ids, list) else []
        except Exception:  # noqa: BLE001
            return []

    def _save_index(self, ids: list[str]) -> None:
        try:
            self._index_path().write_text(json.dumps(ids), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def put(self, room: Room) -> bool:
        try:
            if not room or not room.room_id:
                return False
            self._path(room.room_id).write_text(
                json.dumps(room.to_dict(), ensure_ascii=False), encoding="utf-8")
            ids = self._load_index()
            if room.room_id not in ids:
                ids.append(room.room_id)
                self._save_index(ids)
            return True
        except Exception:  # noqa: BLE001
            return False

    def get(self, room_id: str) -> Room | None:
        try:
            raw = self._path(room_id).read_text(encoding="utf-8")
            return Room.from_dict(json.loads(raw))
        except Exception:  # noqa: BLE001
            return None

    def list(self, community_id: str = "", live_only: bool = True) -> list[Room]:
        rooms: list[Room] = []
        try:
            for rid in self._load_index():
                room = self.get(rid)
                if room is None:
                    continue
                if community_id and room.community_id != community_id:
                    continue
                if live_only and room.state != "live":
                    continue
                rooms.append(room)
        except Exception:  # noqa: BLE001
            pass
        rooms.sort(key=lambda r: r.created_at or 0.0, reverse=True)
        return rooms

    def remove(self, room_id: str) -> bool:
        try:
            self._path(room_id).unlink(missing_ok=True)
            ids = [i for i in self._load_index() if i != room_id]
            self._save_index(ids)
            return True
        except Exception:  # noqa: BLE001
            return False


# ── platform + money + transcription seams ───────────────────────────────────


def _default_platform() -> dict[str, Any]:
    """Honest default platform seam: records intent, controls nothing real.

    A real deployment injects a Telegram voice-chat adapter (or web-room
    adapter) via the ``platform=`` kwarg on the room functions.
    """
    return {"name": "default", "connected": False}


def _platform_start_voice_chat(platform: Any, room: Room) -> dict[str, Any]:
    """Ask the platform seam to start backing audio. Never raises."""
    try:
        fn = getattr(platform, "start_voice_chat", None)
        if callable(fn):
            return fn(room) or {}
        name = platform.get("name") if isinstance(platform, dict) else ""
        if name == "default" or not platform:
            return {
                "ok": False,
                "reason": "no voice-chat platform connected — room runs in announce mode",
            }
        return {"ok": True, "note": "platform acknowledged"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def _platform_mute(platform: Any, room: Room, user_id: str, muted: bool) -> bool:
    try:
        fn = getattr(platform, "set_muted", None)
        if callable(fn):
            return bool(fn(room, user_id, muted))
        return True  # default seam: state tracked locally, platform best-effort
    except Exception:  # noqa: BLE001
        return False


def _default_transcriber(audio_ref: str) -> str:
    """Honest default: no transcription without a wired STT backend."""
    return ""


def _transcribe(audio_ref: str,
                transcriber: Callable[[str], str] | None = None) -> str:
    try:
        fn = transcriber or _default_transcriber
        return str(fn(audio_ref) or "")
    except Exception:  # noqa: BLE001
        return ""


# ── room lifecycle ───────────────────────────────────────────────────────────


def _community_exists(community_id: str, group_store: Any = None) -> bool:
    """Rooms must belong to a real community. Lazy import avoids cycles."""
    try:
        if group_store is None:
            from .groups import GroupStore
            group_store = GroupStore()
        return group_store.get(community_id or "") is not None
    except Exception:  # noqa: BLE001
        return False


def create_room(community_id: str, title: str, host_id: str, host_name: str,
                *, platform: Any = None, store: RoomStore | None = None,
                data_dir: Any = None, group_store: Any = None,
                starts_at: float = 0.0, agenda: str = "") -> dict[str, Any]:
    """Create a live room in a community. Returns {"ok", "room"|"reason"}.

    ``starts_at`` > now → a scheduled room (Resonate pattern): it sits in
    ``scheduled`` state until its time arrives; ``due_starts()`` lists
    rooms ready to go live.
    """
    try:
        community_id = (community_id or "").strip()
        title = (title or "").strip()[:120]
        host_id = (host_id or "").strip()
        host_name = (host_name or host_id or "host").strip()[:60]
        if not community_id or not title or not host_id:
            return {"ok": False, "reason": "need a community, a title, and a host"}
        if not _community_exists(community_id, group_store=group_store):
            return {"ok": False, "reason": f"no such community: {community_id}"}
        st = store or RoomStore(data_dir=data_dir)
        now = time.time()
        scheduled = starts_at > now + 1
        room = Room(
            room_id="room_" + uuid.uuid4().hex[:10],
            community_id=community_id,
            title=title,
            host_id=host_id,
            host_name=host_name,
            platform="telegram",
            state="scheduled" if scheduled else "live",
            starts_at=starts_at if scheduled else 0.0,
            agenda=(agenda or "").strip()[:500],
            created_at=now,
        )
        room.bump_peak()
        plat = platform if platform is not None else _default_platform()
        started = _platform_start_voice_chat(plat, room)
        if started.get("ok"):
            room.platform_ref = str(started.get("voice_chat_id", "") or "")
        elif isinstance(plat, dict) and plat.get("name") == "web":
            room.platform = "web"
            room.platform_ref = str(started.get("room_url", "") or "")
        st.put(room)
        return {"ok": True, "room": room,
                "platform_note": started.get("reason", "") or started.get("note", "")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"create failed: {exc}"[:160]}


def go_live(room_id: str, actor_id: str,
            store: RoomStore | None = None, data_dir: Any = None
            ) -> dict[str, Any]:
    """Start a scheduled room (or re-announce a live one)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        room.state = "live"
        room.starts_at = 0.0
        room.announced = True
        _save(st, room)
        return {"ok": True, "room": room}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def due_starts(now: float | None = None,
               store: RoomStore | None = None, data_dir: Any = None
               ) -> list[Room]:
    """Scheduled rooms whose start time has arrived and aren't announced.

    The host delivers a 'going live' ping and flips them with go_live.
    Marks them announced so the ping fires once.
    """
    try:
        now = now if now is not None else time.time()
        st = store or RoomStore(data_dir=data_dir)
        due: list[Room] = []
        for room in st.list(community_id="", live_only=False):
            if room.state == "scheduled" and not room.announced \
                    and room.starts_at and room.starts_at <= now:
                room.announced = True
                _save(st, room)
                due.append(room)
        return due
    except Exception:  # noqa: BLE001
        return []


def get_room(room_id: str, store: RoomStore | None = None,
             data_dir: Any = None) -> Room | None:
    try:
        return (store or RoomStore(data_dir=data_dir)).get(room_id or "")
    except Exception:  # noqa: BLE001
        return None


def _save(st: RoomStore, room: Room) -> None:
    try:
        st.put(room)
    except Exception:  # noqa: BLE001
        pass


def join_room(room_id: str, user_id: str, name: str,
              store: RoomStore | None = None, data_dir: Any = None
              ) -> dict[str, Any]:
    """Join as a listener. Never raises."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        if room.state != "live":
            return {"ok": False, "reason": "room has ended"}
        uid, nm = (user_id or "").strip(), (name or user_id or "anon").strip()[:60]
        if not uid:
            return {"ok": False, "reason": "who are you?"}
        if room.is_blocked(uid):
            return {"ok": False, "reason": "you're blocked from this room"}
        if room.locked:
            return {"ok": False, "reason": "this room is locked — no new joins"}
        role, _ = room.find(uid)
        if role:
            return {"ok": True, "room": room, "role": role, "note": "already in"}
        room.listeners.append(Participant(user_id=uid, name=nm))
        room.bump_peak()
        _save(st, room)
        return {"ok": True, "room": room, "role": "listener"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def join_with_code(code: str, user_id: str, name: str,
                   store: RoomStore | None = None, data_dir: Any = None
                   ) -> dict[str, Any]:
    """Join via an invite code — speaker codes land as speakers, listener
    codes as listeners (Telegram's speaker/listener invite-link pattern)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        code = (code or "").strip()
        target = None
        for room in st.list(community_id="", live_only=False):
            if room.state == "live" and code in room.invite_codes:
                target = room
                break
        if target is None:
            return {"ok": False, "reason": "bad or expired invite code"}
        role = target.invite_codes.get(code, "listener")
        uid, nm = (user_id or "").strip(), (name or user_id or "anon").strip()[:60]
        if not uid:
            return {"ok": False, "reason": "who are you?"}
        if target.is_blocked(uid):
            return {"ok": False, "reason": "you're blocked from this room"}
        existing, _ = target.find(uid)
        if existing:
            return {"ok": True, "room": target, "role": existing,
                    "note": "already in"}
        if role == "speaker":
            target.speakers.append(Participant(user_id=uid, name=nm))
        else:
            target.listeners.append(Participant(user_id=uid, name=nm))
        target.bump_peak()
        _save(st, target)
        return {"ok": True, "room": target, "role": role}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def lobby_checkin(room_id: str, user_id: str, name: str,
                  store: RoomStore | None = None, data_dir: Any = None
                  ) -> dict[str, Any]:
    """Green-room lobby: check in before the room goes live."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        uid, nm = (user_id or "").strip(), (name or user_id or "anon").strip()[:60]
        if not uid:
            return {"ok": False, "reason": "who are you?"}
        if room.is_blocked(uid):
            return {"ok": False, "reason": "you're blocked from this room"}
        if not any(p.user_id == uid for p in room.lobby):
            room.lobby.append(Participant(user_id=uid, name=nm))
            _save(st, room)
        return {"ok": True, "room": room,
                "lobby_count": len(room.lobby)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def leave_room(room_id: str, user_id: str,
               store: RoomStore | None = None, data_dir: Any = None
               ) -> dict[str, Any]:
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        uid = (user_id or "").strip()
        before = room.headcount()
        room.speakers = [p for p in room.speakers if p.user_id != uid]
        room.hand_raise_queue = [p for p in room.hand_raise_queue if p.user_id != uid]
        room.listeners = [p for p in room.listeners if p.user_id != uid]
        if uid == room.host_id:
            return {"ok": False, "reason": "host can't leave — end the room instead"}
        _save(st, room)
        return {"ok": True, "room": room,
                "left": before != room.headcount()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def raise_hand(room_id: str, user_id: str, name: str,
               store: RoomStore | None = None, data_dir: Any = None,
               note: str = ""
               ) -> dict[str, Any]:
    """Listener asks to speak — joins the hand-raise queue (FIFO).

    ``note`` is shown to the host/co-hosts for triage ("Q about the
    venue") — Telegram shows bios for the same reason. Stale raises
    auto-expire per ``room.hand_raise_expiry``.
    """
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        if room.state != "live":
            return {"ok": False, "reason": "room has ended"}
        uid, nm = (user_id or "").strip(), (name or user_id or "anon").strip()[:60]
        if not uid:
            return {"ok": False, "reason": "who are you?"}
        if room.is_blocked(uid):
            return {"ok": False, "reason": "you're blocked from this room"}
        _expire_hands(room)
        role, _ = room.find(uid)
        if role in ("host", "cohost", "speaker"):
            return {"ok": False, "reason": "you can already speak"}
        if role == "queue":
            return {"ok": True, "room": room, "position": _queue_pos(room, uid),
                    "note": "already in queue"}
        if role == "":
            room.listeners.append(Participant(user_id=uid, name=nm))
        room.hand_raise_queue.append(
            Participant(user_id=uid, name=nm, hand_raised_at=time.time(),
                        note=(note or "").strip()[:120]))
        room.bump_peak()
        _save(st, room)
        return {"ok": True, "room": room, "position": _queue_pos(room, uid)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def _expire_hands(room: Room, now: float | None = None) -> int:
    """Drop stale hand-raises. Returns the number expired."""
    try:
        if not room.hand_raise_expiry:
            return 0
        now = now if now is not None else time.time()
        before = len(room.hand_raise_queue)
        room.hand_raise_queue = [
            p for p in room.hand_raise_queue
            if not p.hand_raised_at or (now - p.hand_raised_at) < room.hand_raise_expiry
        ]
        return before - len(room.hand_raise_queue)
    except Exception:  # noqa: BLE001
        return 0


def _queue_pos(room: Room, user_id: str) -> int:
    for i, p in enumerate(room.hand_raise_queue, 1):
        if p.user_id == user_id:
            return i
    return 0


def lower_hand(room_id: str, user_id: str,
               store: RoomStore | None = None, data_dir: Any = None
               ) -> dict[str, Any]:
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        uid = (user_id or "").strip()
        before = len(room.hand_raise_queue)
        room.hand_raise_queue = [p for p in room.hand_raise_queue if p.user_id != uid]
        _save(st, room)
        return {"ok": True, "room": room,
                "was_queued": len(room.hand_raise_queue) != before}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def _require_host(room: Room, actor_id: str) -> dict[str, Any] | None:
    """Host or co-host required. Co-hosts are trusted speakers (Reddit Talk)."""
    actor = (actor_id or "").strip()
    if actor != room.host_id and not room.is_cohost(actor):
        return {"ok": False, "reason": "only the host or a co-host can do that"}
    if room.state not in ("live", "scheduled"):
        return {"ok": False, "reason": "room has ended"}
    return None


def promote_speaker(room_id: str, user_id: str, actor_id: str,
                    store: RoomStore | None = None, data_dir: Any = None,
                    platform: Any = None) -> dict[str, Any]:
    """Host brings someone up to speak (from queue or listeners)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        uid = (user_id or "").strip()
        target = None
        for lst in (room.hand_raise_queue, room.listeners):
            for p in lst:
                if p.user_id == uid:
                    target = p
                    break
            if target:
                break
        if target is None:
            return {"ok": False, "reason": "that person isn't in the room"}
        room.hand_raise_queue = [p for p in room.hand_raise_queue if p.user_id != uid]
        room.listeners = [p for p in room.listeners if p.user_id != uid]
        if not any(p.user_id == uid for p in room.speakers):
            room.speakers.append(Participant(user_id=uid, name=target.name))
        _save(st, room)
        return {"ok": True, "room": room, "speaker": target.name}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def demote_speaker(room_id: str, user_id: str, actor_id: str,
                   store: RoomStore | None = None, data_dir: Any = None
                   ) -> dict[str, Any]:
    """Host sends a speaker back to the audience."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        uid = (user_id or "").strip()
        moved = [p for p in room.speakers if p.user_id == uid]
        if not moved:
            return {"ok": False, "reason": "not a speaker"}
        room.speakers = [p for p in room.speakers if p.user_id != uid]
        room.listeners.append(Participant(user_id=uid, name=moved[0].name))
        room.bump_peak()
        _save(st, room)
        return {"ok": True, "room": room}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


# ── co-hosts / room control ────────────────────────────────────────────────


def add_cohost(room_id: str, user_id: str, name: str, actor_id: str,
               store: RoomStore | None = None, data_dir: Any = None
               ) -> dict[str, Any]:
    """Host invites a trusted speaker as co-host (Reddit Talk pattern).

    Co-hosts get host powers: promote/demote, mute, lock, record, end.
    """
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        if (actor_id or "").strip() != room.host_id:
            return {"ok": False, "reason": "only the host can name co-hosts"}
        uid, nm = (user_id or "").strip(), (name or user_id or "co-host").strip()[:60]
        if not uid:
            return {"ok": False, "reason": "who?"}
        if uid == room.host_id:
            return {"ok": False, "reason": "that's the host already"}
        if not room.is_cohost(uid):
            room.cohosts.append(Participant(user_id=uid, name=nm))
            _save(st, room)
        return {"ok": True, "room": room, "cohost": nm}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def remove_cohost(room_id: str, user_id: str, actor_id: str,
                  store: RoomStore | None = None, data_dir: Any = None
                  ) -> dict[str, Any]:
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        if (actor_id or "").strip() != room.host_id:
            return {"ok": False, "reason": "only the host can remove co-hosts"}
        uid = (user_id or "").strip()
        before = len(room.cohosts)
        room.cohosts = [p for p in room.cohosts if p.user_id != uid]
        _save(st, room)
        return {"ok": True, "room": room, "removed": len(room.cohosts) != before}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def mute_all(room_id: str, actor_id: str, muted: bool = True,
             store: RoomStore | None = None, data_dir: Any = None,
             platform: Any = None) -> dict[str, Any]:
    """Host/co-host mutes (or unmutes) every speaker at once."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        n = 0
        for p in room.speakers:
            if p.muted != muted:
                p.muted = muted
                _platform_mute(platform, room, p.user_id, muted)
                n += 1
        _save(st, room)
        return {"ok": True, "room": room,
                "action": "muted" if muted else "unmuted", "count": n}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def lock_room(room_id: str, actor_id: str, locked: bool = True,
              store: RoomStore | None = None, data_dir: Any = None
              ) -> dict[str, Any]:
    """Lock rejects all new joins (sabha- 'Lock Sabha' pattern)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        room.locked = bool(locked)
        _save(st, room)
        return {"ok": True, "room": room, "locked": room.locked}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def mint_invite(room_id: str, actor_id: str, role: str = "listener",
                store: RoomStore | None = None, data_dir: Any = None
                ) -> dict[str, Any]:
    """Mint a speaker or listener invite code (Telegram VC 2.0 pattern).

    Speaker codes let the holder join straight onto the stage.
    """
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        role = (role or "listener").lower()
        if role not in ("speaker", "listener"):
            return {"ok": False, "reason": "role must be speaker or listener"}
        code = uuid.uuid4().hex[:8]
        room.invite_codes[code] = role
        _save(st, room)
        return {"ok": True, "room": room, "code": code, "role": role}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def react(room_id: str, user_id: str, emoji: str,
          store: RoomStore | None = None, data_dir: Any = None
          ) -> dict[str, Any]:
    """Emoji reaction to the room (Reddit Talk: react during talks)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        if room.state != "live":
            return {"ok": False, "reason": "room has ended"}
        e = (emoji or "").strip()[:8]
        if not e:
            return {"ok": False, "reason": "which emoji?"}
        room.reactions[e] = room.reactions.get(e, 0) + 1
        _save(st, room)
        return {"ok": True, "room": room, "reactions": dict(room.reactions)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def set_agenda(room_id: str, actor_id: str, text: str,
               store: RoomStore | None = None, data_dir: Any = None
               ) -> dict[str, Any]:
    """Set the room's agenda / running order (rendered on the room card)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        room.agenda = (text or "").strip()[:500]
        _save(st, room)
        return {"ok": True, "room": room}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def room_stats(room_id: str, store: RoomStore | None = None,
               data_dir: Any = None) -> dict[str, Any]:
    """Analytics card for a room: duration, peak, speakers, tips, reactions."""
    try:
        room = (store or RoomStore(data_dir=data_dir)).get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        end = room.ended_at or time.time()
        duration = max(0.0, end - (room.created_at or end))
        tips_total = sum(t.amount_kobo for t in room.tips
                         if t.status in ("sent", "recorded"))
        return {
            "ok": True,
            "room_id": room.room_id,
            "title": room.title,
            "state": room.state,
            "duration_s": duration,
            "peak_headcount": room.peak_headcount or room.headcount(),
            "speakers": len(room.speakers) + len(room.cohosts),
            "listeners": len(room.listeners),
            "reactions": dict(room.reactions),
            "tips_kobo": tips_total,
            "tips_naira": _fmt_naira(tips_total),
            "reports": len(room.reports),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def export_captions_vtt(room_id: str, store: RoomStore | None = None,
                        data_dir: Any = None) -> dict[str, Any]:
    """Export the live-caption log as WebVTT (accessibility artifact)."""
    try:
        room = (store or RoomStore(data_dir=data_dir)).get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        lines = ["WEBVTT", ""]
        step = 5.0  # captions carry no timestamps; 5s slots keep it honest
        base = room.created_at or time.time()
        for i, cap in enumerate(room.caption_log):
            s, e = base + i * step, base + (i + 1) * step
            lines.append(f"{_vtt_ts(s)} --> {_vtt_ts(e)}")
            lines.append(cap)
            lines.append("")
        return {"ok": True, "vtt": "\n".join(lines),
                "lines": len(room.caption_log)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def _vtt_ts(ts: float) -> str:
    try:
        t = time.gmtime(max(0.0, ts))
        ms = int((ts % 1) * 1000)
        return time.strftime("%H:%M:%S", t) + f".{ms:03d}"
    except Exception:  # noqa: BLE001
        return "00:00:00.000"


# ── moderation ───────────────────────────────────────────────────────────────


def mute_user(room_id: str, user_id: str, actor_id: str, muted: bool = True,
              store: RoomStore | None = None, data_dir: Any = None,
              platform: Any = None) -> dict[str, Any]:
    """Host mic control. Live audio can't be pre-screened — host tools are
    the minimum moderation layer."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        uid = (user_id or "").strip()
        if uid == room.host_id:
            return {"ok": False, "reason": "can't mute the host"}
        hit = False
        for lst in (room.speakers, room.listeners):
            for p in lst:
                if p.user_id == uid:
                    p.muted = muted
                    hit = True
        if not hit:
            return {"ok": False, "reason": "that person isn't in the room"}
        _platform_mute(platform, room, uid, muted)
        _save(st, room)
        return {"ok": True, "room": room,
                "action": "muted" if muted else "unmuted"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def block_user(room_id: str, user_id: str, actor_id: str,
               store: RoomStore | None = None, data_dir: Any = None
               ) -> dict[str, Any]:
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        uid = (user_id or "").strip()
        if uid == room.host_id:
            return {"ok": False, "reason": "can't block the host"}
        if uid not in room.blocked:
            room.blocked.append(uid)
        room.speakers = [p for p in room.speakers if p.user_id != uid]
        room.hand_raise_queue = [p for p in room.hand_raise_queue if p.user_id != uid]
        room.listeners = [p for p in room.listeners if p.user_id != uid]
        _save(st, room)
        return {"ok": True, "room": room}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def report_user(room_id: str, user_id: str, reporter_id: str, reason: str,
                store: RoomStore | None = None, data_dir: Any = None
                ) -> dict[str, Any]:
    """Anyone can report. Reports are logged for the host to review."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        uid = (user_id or "").strip()
        rep = (reporter_id or "").strip()
        why = (reason or "").strip()[:200]
        if not uid or not rep:
            return {"ok": False, "reason": "need a user and a reporter"}
        room.reports.append({
            "user_id": uid, "reporter_id": rep, "reason": why,
            "at": time.time(),
        })
        _save(st, room)
        return {"ok": True, "room": room,
                "note": "reported — the host has been notified"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


# ── recording / captions ─────────────────────────────────────────────────────


def set_recording(room_id: str, on: bool, actor_id: str,
                  store: RoomStore | None = None, data_dir: Any = None
                  ) -> dict[str, Any]:
    """Recording is opt-in per room; state is always visible (🔴 REC)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        room.recording = bool(on)
        room.recording_visible = bool(on)  # visible state is mandatory
        _save(st, room)
        return {"ok": True, "room": room,
                "recording": room.recording}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def set_captions(room_id: str, on: bool, actor_id: str,
                 store: RoomStore | None = None, data_dir: Any = None
                 ) -> dict[str, Any]:
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        room.captions = bool(on)
        _save(st, room)
        return {"ok": True, "room": room, "captions": room.captions}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


def add_caption(room_id: str, text: str,
                store: RoomStore | None = None, data_dir: Any = None
                ) -> dict[str, Any]:
    """Append a live-caption line (from the platform's caption feed)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        line = (text or "").strip()[:280]
        if not line:
            return {"ok": False, "reason": "empty caption"}
        if not room.captions:
            return {"ok": False, "reason": "captions are off in this room"}
        room.caption_log.append(line)
        room.caption_log = room.caption_log[-50:]
        _save(st, room)
        return {"ok": True, "room": room}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


# ── tips ─────────────────────────────────────────────────────────────────────


def _fmt_naira(kobo: int) -> str:
    try:
        n = int(kobo or 0)
        if n % 100 == 0:
            return f"₦{n // 100:,}"
        return f"₦{n / 100:,.2f}"
    except Exception:  # noqa: BLE001
        return "₦0"


def tip_user(room_id: str, from_id: str, from_name: str, to_id: str,
             amount_kobo: int, tip_fn: Callable[..., Any] | None = None,
             store: RoomStore | None = None, data_dir: Any = None
             ) -> dict[str, Any]:
    """Tip a speaker/host. Money moves through the injected ``tip_fn``
    (#65 Paystack / #49 money primitive, wired by the host app) — this
    module never touches money rails directly."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        if room.state != "live":
            return {"ok": False, "reason": "room has ended"}
        fid, tid = (from_id or "").strip(), (to_id or "").strip()
        amt = max(0, int(amount_kobo or 0))
        if not fid or not tid:
            return {"ok": False, "reason": "need a sender and a recipient"}
        if amt <= 0:
            return {"ok": False, "reason": "tip amount must be positive"}
        _, recip = room.find(tid)
        recip_name = recip.name if recip else (tid or "them")
        tip = Tip(
            tip_id="tip_" + uuid.uuid4().hex[:10],
            room_id=room.room_id,
            from_id=fid,
            from_name=(from_name or fid or "anon").strip()[:60],
            to_id=tid,
            to_name=recip_name[:60],
            amount_kobo=amt,
            created_at=time.time(),
        )
        if tip_fn is None:
            tip.status = "recorded"
            tip.note = "tips aren't wired up in this community yet — recorded for the host"
            room.tips.append(tip)
            _save(st, room)
            return {"ok": True, "room": room, "tip": tip,
                    "note": tip.note}
        try:
            result = tip_fn(from_id=fid, to_id=tid, amount_kobo=amt,
                            room_id=room.room_id)
        except Exception as exc:  # noqa: BLE001
            tip.status = "failed"
            tip.note = f"tip processor failed: {exc}"[:160]
            room.tips.append(tip)
            _save(st, room)
            return {"ok": False, "reason": tip.note, "tip": tip}
        if isinstance(result, dict) and result.get("ok"):
            tip.status = "sent"
        else:
            tip.status = "failed"
            tip.note = str((result or {}).get("reason", "processor declined"))[:160]
        room.tips.append(tip)
        _save(st, room)
        return {"ok": tip.status == "sent", "room": room, "tip": tip,
                "reason": tip.note if tip.status != "sent" else ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


# ── ending / transcript ──────────────────────────────────────────────────────


def end_room(room_id: str, actor_id: str,
             transcriber: Callable[[str], str] | None = None,
             store: RoomStore | None = None, data_dir: Any = None
             ) -> dict[str, Any]:
    """Host ends the room. If it was recorded, a transcript is generated
    through the ``transcriber`` seam (WhisperX → #103 in production)."""
    try:
        st = store or RoomStore(data_dir=data_dir)
        room = st.get(room_id or "")
        if room is None:
            return {"ok": False, "reason": "no such room"}
        denied = _require_host(room, actor_id)
        if denied:
            return denied
        room.state = "ended"
        room.ended_at = time.time()
        room.recording = False
        room.recording_visible = False
        note = ""
        if room.transcript:
            note = "transcript already exists"
        elif transcriber is not None or room.caption_log:
            # Prefer the real transcriber; fall back to caption log.
            text = _transcribe(room.platform_ref or room.room_id, transcriber)
            if not text and room.caption_log:
                text = "\n".join(room.caption_log)
                note = "transcript from live captions (no audio STT wired)"
            elif text:
                note = "transcript generated"
            room.transcript = text[:20000]
        _save(st, room)
        return {"ok": True, "room": room, "transcript_note": note}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:120]}


# ── rendering ────────────────────────────────────────────────────────────────


def render_room(room: Room) -> str:
    """Chat-ready room card. Recording state is always visible."""
    try:
        if room is None:
            return "no such room"
        rec = " 🔴 REC" if room.recording and room.recording_visible else ""
        lock = " 🔒" if room.locked else ""
        state_glyph = {"live": "🟢 LIVE", "scheduled": "⏰ SCHEDULED",
                       "ended": "⚫ ENDED"}.get(room.state, room.state.upper())
        lines = [f"🎙️ **{room.title or 'Room'}**{rec}{lock}",
                 f"   {state_glyph} · {room.headcount()} in right now"]
        host_line = f"👑 {room.host_name or room.host_id}"
        if room.cohosts:
            host_line += "  ·  🤝 " + ", ".join(p.name for p in room.cohosts)
        lines.append(f"   {host_line}")
        if room.agenda:
            lines.append(f"   📝 {room.agenda[:140]}")
        if room.state == "scheduled" and room.starts_at:
            when = time.strftime("%a %H:%M", time.localtime(room.starts_at))
            lines.append(f"   ⏰ starts {when}")
            if room.lobby:
                lines.append(f"   🟢 lobby: {len(room.lobby)} waiting — "
                             + ", ".join(p.name for p in room.lobby[:8]))
        spk = ", ".join(
            f"{p.name}{' 🔇' if p.muted else ''}" for p in room.speakers) or "—"
        lines.append(f"   🎤 speakers ({len(room.speakers)}): {spk}")
        q = ", ".join(
            f"{i}. {p.name}" + (f" _“{p.note}”_" if p.note else "")
            for i, p in enumerate(room.hand_raise_queue, 1)) or "—"
        lines.append(f"   ✋ hand-raise ({len(room.hand_raise_queue)}): {q}")
        lines.append(f"   🎧 listeners: {len(room.listeners)} · "
                     f"captions: {'on' if room.captions else 'off'}")
        if room.reactions:
            top = " ".join(f"{e}×{c}"
                           for e, c in sorted(room.reactions.items(),
                                              key=lambda kv: -kv[1])[:6])
            lines.append(f"   ❤️ reactions: {top}")
        if room.tips:
            total = sum(t.amount_kobo for t in room.tips if t.status in ("sent", "recorded"))
            lines.append(f"   💸 tips: {_fmt_naira(total)} ({len(room.tips)})")
        if room.reports:
            lines.append(f"   ⚠️ {len(room.reports)} report(s) for the host")
        if room.state == "ended" and room.transcript:
            lines.append(f"   📝 transcript: {room.transcript[:120]}…")
        lines.append(f"   `{room.room_id}`")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return "room card failed to render"


# ── chat ─────────────────────────────────────────────────────────────────────


def _parse_room_when(text: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM' (local) → epoch. 0.0 when unparseable."""
    m = re.fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", (text or "").strip())
    if not m:
        return 0.0
    try:
        return time.mktime((
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), 0, 0, 0, -1))
    except (ValueError, OverflowError):
        return 0.0


def _usage() -> str:
    return (
        "🎙️ /room — live audio rooms (community-scoped)\n"
        "  /room create <community_id> | <title> [| at:YYYY-MM-DD HH:MM] [| agenda:…]\n"
        "  /room list [community_id] · join <room_id> · code <invite_code> · leave <room_id>\n"
        "  /room lobby <room_id> — green-room check-in before go-live\n"
        "  /room raise <room_id> [note] — ask to speak (hand-raise)\n"
        "  /room lower <room_id>\n"
        "  /room speak <room_id> <user_id> — host brings someone up\n"
        "  /room drop <room_id> <user_id> — host sends back to audience\n"
        "  /room cohost <room_id> <user_id> — trusted speaker w/ host powers\n"
        "  /room mute|unmute <room_id> <user_id> · /room muteall <room_id>\n"
        "  /room lock|unlock <room_id> — stop/start new joins\n"
        "  /room invite <room_id> <speaker|listener> — mint an invite code\n"
        "  /room block <room_id> <user_id> · report <room_id> <user_id> <reason>\n"
        "  /room record <room_id> on|off — opt-in, always visible\n"
        "  /room captions <room_id> on|off · vtt <room_id> — caption export\n"
        "  /room agenda <room_id> <text> · react <room_id> <emoji>\n"
        "  /room tip <room_id> <user_id> <amount_naira>\n"
        "  /room stats <room_id> · /room show <room_id> · /room end <room_id> · /room golive <room_id>"
    )


def control_room(tail: str, context=None, chat=None,
                 sender: str = "", sender_id: str = "",
                 store: RoomStore | None = None,
                 platform: Any = None,
                 tip_fn: Callable[..., Any] | None = None,
                 transcriber: Callable[[str], str] | None = None,
                 group_store: Any = None) -> str:
    """Chat entry point for audio rooms. Never raises. Community-scoped —
    anyone in the community drives it (not owner-gated)."""
    try:
        return _control_room(tail, context, chat, sender, sender_id,
                             store, platform, tip_fn, transcriber, group_store)
    except Exception as exc:  # noqa: BLE001
        _log.warning("room control failed: %s", exc)
        return "Audio rooms hit a snag — try again."


def _control_room(tail, context, chat, sender, sender_id,
                  store, platform, tip_fn, transcriber, group_store) -> str:
    st = store or RoomStore()
    who = (sender or sender_id or "anon").strip() or "anon"
    who_id = (sender_id or sender or "anon").strip() or "anon"
    parts = (tail or "").strip().split(None, 1)
    if not parts:
        return _usage()
    cmd, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

    if cmd == "create":
        segs = [b.strip() for b in rest.split("|")]
        if len(segs) < 2 or not segs[0]:
            return "usage: /room create <community_id> | <title> [| at:YYYY-MM-DD HH:MM] [| agenda:…]"
        starts_at = 0.0
        agenda = ""
        title_parts: list[str] = []
        for extra in segs[1:]:
            low = extra.lower()
            if low.startswith("at:"):
                starts_at = _parse_room_when(extra[3:])
            elif low.startswith("agenda:"):
                agenda = extra[7:].strip()
            else:
                title_parts.append(extra)
        title = " | ".join(title_parts).strip()
        if not title:
            return "usage: /room create <community_id> | <title>"
        res = create_room(segs[0], title, who_id, who,
                          platform=platform, store=st, group_store=group_store,
                          starts_at=starts_at, agenda=agenda)
        if not res.get("ok"):
            return f"couldn't create the room: {res.get('reason')}"
        room = res["room"]
        note = f"\n_{res['platform_note']}_" if res.get("platform_note") else ""
        sched = ""
        if room.state == "scheduled":
            when = time.strftime("%a %H:%M", time.localtime(room.starts_at))
            sched = f"\n⏰ scheduled for {when} — I'll ping when it's time"
        return f"🎙️ room is live!\n{render_room(room)}{sched}{note}"

    if cmd == "golive":
        res = go_live(rest.strip(), who_id, store=st)
        if not res.get("ok"):
            return f"couldn't go live: {res.get('reason')}"
        return f"🎙️ we're live!\n{render_room(res['room'])}"

    if cmd == "list":
        rooms = st.list(community_id=rest.strip(), live_only=True)
        sched = [r for r in st.list(community_id=rest.strip(), live_only=False)
                 if r.state == "scheduled"]
        if not rooms and not sched:
            return "no live rooms right now."
        out = []
        if rooms:
            out.append("🎙️ live rooms:")
            for r in rooms[:15]:
                rec = " 🔴" if r.recording else ""
                lk = " 🔒" if r.locked else ""
                out.append(f"· {r.title}{rec}{lk} — {r.headcount()} in · `{r.room_id}`")
        if sched:
            out.append("⏰ scheduled:")
            for r in sched[:10]:
                when = time.strftime("%a %H:%M", time.localtime(r.starts_at)) \
                    if r.starts_at else "?"
                out.append(f"· {r.title} — {when} · `{r.room_id}`")
        return "\n".join(out)

    if cmd == "show":
        room = st.get(rest.strip())
        return render_room(room) if room else "no such room"

    if cmd == "stats":
        s = room_stats(rest.strip(), store=st)
        if not s.get("ok"):
            return f"couldn't: {s.get('reason')}"
        dur = int(s["duration_s"])
        dh, dm = divmod(dur // 60, 60)
        reac = " ".join(f"{e}×{c}" for e, c in
                        sorted(s["reactions"].items(), key=lambda kv: -kv[1])[:5])
        return (f"📊 **{s['title']}** — {s['state']}\n"
                f"  ⏱️ {dh}h{dm:02d}m · 👥 peak {s['peak_headcount']} · "
                f"🎤 {s['speakers']} speakers\n"
                f"  💸 tips {s['tips_naira']} · ❤️ {reac or '—'} · "
                f"⚠️ {s['reports']} reports")

    if cmd == "vtt":
        res = export_captions_vtt(rest.strip(), store=st)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        return (f"📝 WebVTT captions ({res['lines']} lines):\n"
                f"```\n{res['vtt'][:1500]}\n```")

    if cmd == "join":
        res = join_room(rest.strip(), who_id, who, store=st)
        if not res.get("ok"):
            return f"couldn't join: {res.get('reason')}"
        return f"you're in as a listener 🎧\n{render_room(res['room'])}"

    if cmd == "code":
        res = join_with_code(rest.strip(), who_id, who, store=st)
        if not res.get("ok"):
            return f"couldn't join: {res.get('reason')}"
        return f"you're in as a {res.get('role')} 🎉\n{render_room(res['room'])}"

    if cmd == "lobby":
        res = lobby_checkin(rest.strip(), who_id, who, store=st)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        return (f"🟢 checked into the green room ({res['lobby_count']} waiting). "
                "the host will bring you in when it's time.")

    if cmd == "leave":
        res = leave_room(rest.strip(), who_id, store=st)
        return "you left the room." if res.get("ok") else f"couldn't leave: {res.get('reason')}"

    if cmd == "raise":
        bits = rest.split(None, 1)
        if not bits:
            return "usage: /room raise <room_id> [note]"
        res = raise_hand(bits[0], who_id, who, store=st,
                         note=bits[1] if len(bits) > 1 else "")
        if not res.get("ok"):
            return f"couldn't raise your hand: {res.get('reason')}"
        return f"✋ hand raised — you're #{res.get('position')} in the queue"

    if cmd == "react":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room react <room_id> <emoji>"
        res = react(bits[0], who_id, bits[1], store=st)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        top = " ".join(f"{e}×{c}" for e, c in
                       sorted(res["reactions"].items(), key=lambda kv: -kv[1])[:5])
        return f"{bits[1]} noted! {top}"

    if cmd == "agenda":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room agenda <room_id> <text>"
        res = set_agenda(bits[0], who_id, bits[1], store=st)
        return "📝 agenda set." if res.get("ok") else f"couldn't: {res.get('reason')}"

    if cmd == "cohost":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room cohost <room_id> <user_id>"
        res = add_cohost(bits[0], bits[1], bits[1], who_id, store=st)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        return f"🤝 {res.get('cohost')} is now a co-host"

    if cmd == "uncohost":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room uncohost <room_id> <user_id>"
        res = remove_cohost(bits[0], bits[1], who_id, store=st)
        return "co-host removed." if res.get("ok") else f"couldn't: {res.get('reason')}"

    if cmd == "muteall":
        bits = rest.split(None, 1)
        on = not (len(bits) > 1 and bits[1].lower() == "off")
        res = mute_all(bits[0] if bits else "", who_id, muted=on, store=st,
                       platform=platform)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        return f"🔇 all speakers {res.get('action')} ({res.get('count')})."

    if cmd in ("lock", "unlock"):
        res = lock_room(rest.strip(), who_id, locked=(cmd == "lock"), store=st)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        return "🔒 room locked — no new joins." if res.get("locked") else "🔓 room unlocked."

    if cmd == "invite":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room invite <room_id> <speaker|listener>"
        res = mint_invite(bits[0], who_id, bits[1], store=st)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        return (f"🎟️ {res.get('role')} invite code: `{res.get('code')}`\n"
                f"join with: /room code {res.get('code')}")

    if cmd == "lower":
        res = lower_hand(rest.strip(), who_id, store=st)
        return "hand lowered." if res.get("ok") else f"couldn't: {res.get('reason')}"

    if cmd == "speak":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room speak <room_id> <user_id>"
        res = promote_speaker(bits[0], bits[1], who_id, store=st, platform=platform)
        if not res.get("ok"):
            return f"couldn't bring them up: {res.get('reason')}"
        return f"🎤 {res.get('speaker')} is now speaking"

    if cmd == "drop":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room drop <room_id> <user_id>"
        res = demote_speaker(bits[0], bits[1], who_id, store=st)
        return "back to the audience." if res.get("ok") else f"couldn't: {res.get('reason')}"

    if cmd in ("mute", "unmute"):
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return f"usage: /room {cmd} <room_id> <user_id>"
        res = mute_user(bits[0], bits[1], who_id, muted=(cmd == "mute"),
                        store=st, platform=platform)
        if not res.get("ok"):
            return f"couldn't {cmd}: {res.get('reason')}"
        return f"{bits[1]} {res.get('action')}."

    if cmd == "block":
        bits = rest.split(None, 1)
        if len(bits) < 2:
            return "usage: /room block <room_id> <user_id>"
        res = block_user(bits[0], bits[1], who_id, store=st)
        return "blocked from this room." if res.get("ok") else f"couldn't block: {res.get('reason')}"

    if cmd == "report":
        bits = rest.split(None, 2)
        if len(bits) < 3:
            return "usage: /room report <room_id> <user_id> <reason>"
        res = report_user(bits[0], bits[1], who_id, bits[2], store=st)
        return res.get("note", "reported.") if res.get("ok") else f"couldn't report: {res.get('reason')}"

    if cmd == "record":
        bits = rest.split(None, 1)
        if len(bits) < 2 or bits[1].lower() not in ("on", "off"):
            return "usage: /room record <room_id> on|off"
        res = set_recording(bits[0], bits[1].lower() == "on", who_id, store=st)
        if not res.get("ok"):
            return f"couldn't: {res.get('reason')}"
        state = "🔴 recording — visible to everyone" if res.get("recording") else "recording off"
        return state

    if cmd == "captions":
        bits = rest.split(None, 1)
        if len(bits) < 2 or bits[1].lower() not in ("on", "off"):
            return "usage: /room captions <room_id> on|off"
        res = set_captions(bits[0], bits[1].lower() == "on", who_id, store=st)
        return f"captions {'on' if res.get('captions') else 'off'}." if res.get("ok") else f"couldn't: {res.get('reason')}"

    if cmd == "tip":
        bits = rest.split(None, 2)
        if len(bits) < 3:
            return "usage: /room tip <room_id> <user_id> <amount_naira>"
        try:
            kobo = int(round(float(bits[2].replace(",", "").replace("₦", "")) * 100))
        except Exception:  # noqa: BLE001
            return "tip amount must be a number (naira)"
        res = tip_user(bits[0], who_id, who, bits[1], kobo,
                       tip_fn=tip_fn, store=st)
        if res.get("ok") and res.get("tip"):
            t = res["tip"]
            if t.status == "sent":
                return f"💸 {_fmt_naira(t.amount_kobo)} sent to {t.to_name}!"
            return f"💸 tip recorded ({_fmt_naira(t.amount_kobo)} → {t.to_name}): {res.get('note')}"
        return f"couldn't tip: {res.get('reason')}"

    if cmd == "end":
        res = end_room(rest.strip(), who_id, transcriber=transcriber, store=st)
        if not res.get("ok"):
            return f"couldn't end: {res.get('reason')}"
        note = f"\n_{res.get('transcript_note')}_" if res.get("transcript_note") else ""
        return f"room ended. thanks for coming 🎙️{note}"

    return _usage()
