"""LiveLocationBroker: consent-gated live shares of SUPPLIED updates.

The broker (docs 19+21 M6) manages time-boxed live-location shares
on top of :class:`~nomorals.location.privacy.ConsentStore`.  It is
an extension of this package's shipped shape, not a new organ:

* **Supplied updates only.**  The broker's one intake is
  :meth:`LiveLocationBroker.evaluate`, handed a position a person
  (in practice the owner) *sent* Devon — a location share in chat.
  The broker never polls, never opens GPS, never runs a timer that
  acquires a position, and holds no watcher of any kind.  An expiry
  "sweep" here is a lazy state flip inside calls the owner already
  made; it touches records, never a location source.  ``never
  granted`` is a fully supported steady state: with no grant and no
  session, every evaluation is one withheld decision record.

* **Consent is checked, never cached.**  A session starts only on a
  live grant under :data:`SCOPE_LIVE_SHARE` (``ConsentStore.check``
  raises :class:`ConsentError` naming the missing grant), and every
  forward re-checks ``has_consent`` at evaluation time — an external
  revoke lands on the very next update, mid-session or not.

* **The owner's stop is sticky.**  Session states persist to disk
  on every transition; a broker re-instantiated over the same root
  (a restart) reloads them, and terminal states (ended / revoked /
  expired) never resurrect — starting a share always mints a NEW
  session, it never revives an old one.

* **Precision is part of the share.**  Each session carries a
  precision tier (``exact`` / ``neighborhood`` / ``city``).  The
  forwarded payload is coarsened to the tier BEFORE it reaches the
  sender, and the decision record stores only the coarsened
  coordinates — the audit trail never holds a finer fix than what
  recipients actually saw.  An evaluation matching no session
  records NO coordinates at all (nothing was shared; the record
  says an update arrived and matched nothing).

* **Every evaluation is a durable decision.**  Forwarded or
  withheld, each (update, session) pair appends one JSONL record
  with the reason — grant-live, no-session, expired, revoked,
  ended, consent-denied, stale-update, send-failed, no-sender.
  A fix whose supplied timestamp is older than :data:`STALE_AFTER_S`
  at evaluation time is stale, not live, and is withheld as such.

State lives beside the consent records (the location root):
``live_sessions.json`` + ``live_decisions.jsonl``, in the same
file-store idiom as ``ConsentStore``.  Sending is an injected
callable (``sender(chat_key, loc) -> bool``): this layer owns no
gateway — the runtime wires the platform adapters in.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .primitives import LatLon
from .privacy import ConsentStore, default_location_root

__all__ = [
    "SCOPE_LIVE_SHARE", "PRECISIONS", "STALE_AFTER_S", "MAX_SHARE_S",
    "LiveLocationError", "ShareSession", "LiveLocationBroker", "coarsen",
]

#: The ConsentStore scope every live share rides.  A grant for any
#: other scope ("geofence", "reminders", …) never implies this one.
SCOPE_LIVE_SHARE = "live-share"

#: Precision tiers a share can declare (doc 21 §5: exact /
#: neighborhood / city).  Coarsening is decimal-place rounding:
#: neighborhood ≈ 2 places (~1.1 km), city ≈ 1 place (~11 km).
PRECISIONS = ("exact", "neighborhood", "city")
_PRECISION_PLACES = {"exact": None, "neighborhood": 2, "city": 1}

#: A supplied fix older than this at evaluation time is stale.
STALE_AFTER_S = 900.0

#: One share is time-boxed; a day is the longest single box.
MAX_SHARE_S = 86_400.0

#: Session states.  Everything but "active" is terminal.
STATE_ACTIVE = "active"
STATE_EXPIRED = "expired"
STATE_REVOKED = "revoked"
STATE_ENDED = "ended"
TERMINAL_STATES = (STATE_EXPIRED, STATE_REVOKED, STATE_ENDED)

SenderFn = Callable[[str, LatLon], bool]
ClockFn = Callable[[], float]


class LiveLocationError(Exception):
    """A typed refusal: the share as asked cannot exist."""


def coarsen(loc: LatLon, precision: str) -> LatLon:
    """The position at ``precision``'s tier — what recipients see.

    Exact returns the fix itself; the coarser tiers round the
    decimal places away.  Rounding can only move a coordinate
    within its tier's cell, never out of range.
    """
    places = _PRECISION_PLACES[precision]
    if places is None:
        return loc
    return LatLon(lat=round(loc.lat, places), lon=round(loc.lon, places))


@dataclass
class ShareSession:
    """One time-boxed live share: whose updates, to whom, how fine."""

    id: str
    subject: str
    recipients: tuple[str, ...]
    precision: str = "exact"
    purpose: str = ""
    started_at: float = 0.0
    expires_at: float = 0.0
    state: str = STATE_ACTIVE
    scope: str = SCOPE_LIVE_SHARE

    @property
    def grant_ref(self) -> str:
        """The ConsentStore key this session rides (``person:scope``)."""
        return f"{self.subject}:{self.scope}"

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "subject": self.subject,
                "recipients": list(self.recipients),
                "precision": self.precision, "purpose": self.purpose,
                "started_at": self.started_at,
                "expires_at": self.expires_at, "state": self.state,
                "scope": self.scope}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShareSession:
        return cls(
            id=str(data["id"]), subject=str(data["subject"]),
            recipients=tuple(str(r) for r in data.get("recipients", ())),
            precision=str(data.get("precision", "exact")),
            purpose=str(data.get("purpose", "")),
            started_at=float(data.get("started_at", 0.0)),
            expires_at=float(data.get("expires_at", 0.0)),
            state=str(data.get("state", STATE_ACTIVE)),
            scope=str(data.get("scope", SCOPE_LIVE_SHARE)))


class LiveLocationBroker:
    """Sessions + supplied-update evaluation over a ConsentStore."""

    def __init__(self, root: Path | str | None = None, *,
                 consent: ConsentStore | None = None,
                 clock: ClockFn | None = None,
                 sender: SenderFn | None = None) -> None:
        self.root = (Path(root) if root is not None
                     else default_location_root())
        self.root.mkdir(parents=True, exist_ok=True)
        self._consent = consent if consent is not None \
            else ConsentStore(self.root)
        self._clock = clock if clock is not None else time.time
        self._sender = sender

    # -- persistence ---------------------------------------------------

    def _sessions_path(self) -> Path:
        return self.root / "live_sessions.json"

    def _decisions_path(self) -> Path:
        return self.root / "live_decisions.jsonl"

    def _load_sessions(self) -> dict[str, ShareSession]:
        path = self._sessions_path()
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, ShareSession] = {}
        for sid, rec in raw.items():
            out[sid] = ShareSession.from_dict(rec)
        return out

    def _save_sessions(self, sessions: dict[str, ShareSession]) -> None:
        data = {sid: s.to_dict() for sid, s in sessions.items()}
        self._sessions_path().write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _append_decision(self, record: dict[str, Any]) -> None:
        with self._decisions_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _settle(self, sessions: dict[str, ShareSession],
                now: float) -> bool:
        """Flip active-but-past-expiry sessions to ``expired``.

        Returns True when anything changed (caller persists).  This
        is the only "sweep" the broker has: a state flip over
        records, run lazily inside calls — it acquires nothing.
        """
        changed = False
        for session in sessions.values():
            if session.state == STATE_ACTIVE \
                    and session.expires_at <= now:
                session.state = STATE_EXPIRED
                changed = True
        return changed

    # -- sessions ------------------------------------------------------

    def start_session(self, subject: str, recipients: Any, *,
                      ttl_s: float, precision: str = "exact",
                      purpose: str = "") -> ShareSession:
        """Open a share for ``subject``'s supplied updates.

        Refuses (typed) on a bad shape — no recipients, a precision
        outside the tiers, a non-positive or over-cap TTL — and on
        missing consent: ``ConsentStore.check`` raises
        ``ConsentError`` naming the person and scope, which IS the
        refusal.  Starting never revives a terminal session; every
        start mints a fresh id.
        """
        recips = tuple(str(r).strip() for r in (recipients or ())
                       if str(r).strip())
        if not str(subject or "").strip():
            raise LiveLocationError("a live share needs a subject")
        if not recips:
            raise LiveLocationError(
                "a live share needs at least one recipient")
        if precision not in PRECISIONS:
            raise LiveLocationError(
                f"unknown precision {precision!r} — one of "
                + ", ".join(PRECISIONS))
        ttl = float(ttl_s)
        if ttl <= 0:
            raise LiveLocationError("a live share needs a positive TTL")
        if ttl > MAX_SHARE_S:
            raise LiveLocationError(
                f"a single live share is capped at {MAX_SHARE_S:.0f}s "
                f"(24h); asked for {ttl:.0f}s")
        self._consent.check(str(subject), SCOPE_LIVE_SHARE)
        now = self._clock()
        session = ShareSession(
            id="ls-" + uuid.uuid4().hex[:12], subject=str(subject),
            recipients=recips, precision=precision, purpose=purpose,
            started_at=now, expires_at=now + ttl)
        sessions = self._load_sessions()
        if self._settle(sessions, now):
            pass  # persisted together with the new session below
        sessions[session.id] = session
        self._save_sessions(sessions)
        return session

    def stop_session(self, session_id: str) -> ShareSession | None:
        """End one session (owner stop).  Terminal states are final:
        stopping an ended/revoked/expired session returns it as-is;
        an unknown id returns None.  The stop persists immediately,
        so no later broker instance can resurrect the share."""
        sessions = self._load_sessions()
        session = sessions.get(session_id)
        if session is None:
            return None
        now = self._clock()
        self._settle(sessions, now)
        if session.state == STATE_ACTIVE:
            session.state = STATE_ENDED
        self._save_sessions(sessions)
        return session

    def revoke_subject(self, subject: str) -> list[ShareSession]:
        """Revoke every live session for ``subject`` (the hard stop).

        The sessions flip to ``revoked`` and persist; pair this with
        ``ConsentStore.revoke`` for the full stop — either one alone
        already halts forwarding (revoked state here, consent check
        there)."""
        sessions = self._load_sessions()
        now = self._clock()
        self._settle(sessions, now)
        flipped: list[ShareSession] = []
        for session in sessions.values():
            if session.subject == subject \
                    and session.state == STATE_ACTIVE:
                session.state = STATE_REVOKED
                flipped.append(session)
        self._save_sessions(sessions)
        return flipped

    def sessions(self, subject: str | None = None, *,
                 include_terminal: bool = False) -> list[ShareSession]:
        """Current sessions (expiry settled), oldest first."""
        stored = self._load_sessions()
        if self._settle(stored, self._clock()):
            self._save_sessions(stored)
        out = [s for s in stored.values()
               if (subject is None or s.subject == subject)
               and (include_terminal or s.state == STATE_ACTIVE)]
        out.sort(key=lambda s: s.started_at)
        return out

    # -- evaluation ----------------------------------------------------

    def evaluate(self, subject: str, loc: LatLon, *,
                 ts: float | None = None,
                 source: str = "") -> list[dict[str, Any]]:
        """Evaluate ONE supplied update against ``subject``'s shares.

        ``loc`` arrived because somebody sent it; ``ts`` is when the
        fix was supplied (defaults to now on the broker clock).  One
        decision record per matching session is appended + returned;
        with no sessions at all, a single ``no-session`` withheld
        record (carrying no coordinates) is the audit entry.
        """
        now = self._clock()
        upd_ts = float(ts) if ts is not None else now
        age_s = round(now - upd_ts, 3)
        stored = self._load_sessions()
        if self._settle(stored, now):
            self._save_sessions(stored)
        mine = [s for s in stored.values() if s.subject == subject]
        mine.sort(key=lambda s: s.started_at)
        decisions: list[dict[str, Any]] = []
        if not mine:
            decisions.append(self._record(
                now, subject, None, "withheld", "no-session",
                None, age_s, source, ()))
            return decisions
        for session in mine:
            decisions.append(self._evaluate_one(
                session, loc, now, age_s, source))
        return decisions

    def _evaluate_one(self, session: ShareSession, loc: LatLon,
                      now: float, age_s: float,
                      source: str) -> dict[str, Any]:
        payload = coarsen(loc, session.precision)
        if session.state == STATE_EXPIRED:
            return self._record(now, session.subject, session,
                                "withheld", "expired", payload,
                                age_s, source, ())
        if session.state == STATE_REVOKED:
            return self._record(now, session.subject, session,
                                "withheld", "revoked", payload,
                                age_s, source, ())
        if session.state == STATE_ENDED:
            return self._record(now, session.subject, session,
                                "withheld", "ended", payload,
                                age_s, source, ())
        if not self._consent.has_consent(session.subject, session.scope):
            return self._record(now, session.subject, session,
                                "withheld", "consent-denied", payload,
                                age_s, source, ())
        if age_s > STALE_AFTER_S:
            return self._record(now, session.subject, session,
                                "withheld", "stale-update", payload,
                                age_s, source, ())
        if self._sender is None:
            return self._record(now, session.subject, session,
                                "withheld", "no-sender", payload,
                                age_s, source, session.recipients)
        failed: list[str] = []
        for chat_key in session.recipients:
            try:
                ok = bool(self._sender(chat_key, payload))
            except Exception:  # noqa: BLE001 - one bad recipient
                ok = False  # never stops the others being tried
            if not ok:
                failed.append(chat_key)
        if len(failed) == len(session.recipients):
            return self._record(now, session.subject, session,
                                "withheld", "send-failed", payload,
                                age_s, source, tuple(failed))
        return self._record(now, session.subject, session,
                            "forwarded", "grant-live", payload,
                            age_s, source, tuple(failed))

    def _record(self, now: float, subject: str,
                session: ShareSession | None, decision: str,
                reason: str, payload: LatLon | None, age_s: float,
                source: str,
                failed: tuple[str, ...]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "at": now, "subject": subject,
            "session_id": session.id if session else "",
            "recipients": (list(session.recipients)
                           if session else []),
            "precision": session.precision if session else "",
            "decision": decision, "reason": reason,
            "lat": payload.lat if payload else None,
            "lon": payload.lon if payload else None,
            "update_age_s": age_s, "source": source,
            "failed_recipients": list(failed),
        }
        self._append_decision(record)
        return record

    def recent_decisions(self, limit: int = 20,
                         subject: str | None = None,
                         ) -> list[dict[str, Any]]:
        """The newest decision records (audit read-back)."""
        path = self._decisions_path()
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn tail line is skipped, never fatal
            if subject is None or rec.get("subject") == subject:
                rows.append(rec)
        return rows[-max(0, int(limit)):] if limit else rows
