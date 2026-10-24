"""Durable outbox for chat sends (G2 + G4).

Every durable send becomes a row before it becomes a network call.
The row is the unit of work; the row is also the receipt.

Storage is the existing :class:`WorkQueue` (``storage/queue.py``) —
lease / reclaim / backoff / dead-letter — not a new scheduler and not a
parallel table. One WorkQueue topic per chat
(``chat_outbox:<platform>:<chat_key>``) so that:

* ordering is per-chat: a chat's jobs drain in ``order_seq`` order and
  the head-of-line rule holds — a later job is never sent before an
  earlier job's outcome is known (a transiently-failed head blocks its
  chat; a terminally-failed head kills only its own logical message's
  remaining chunks);
* one dead platform cannot jam the queue: drain iterates topics and
  isolates failures per topic.

States (receipt view) map from WorkQueue statuses:

* ``ready`` → ``queued``, ``leased`` → ``sending``, ``done`` → ``sent``,
  ``cancelled`` → ``dropped``, ``dead`` → ``failed_terminal`` /
  ``failed_transient`` (classified from the stored error).

Idempotency: re-submitting with the same ``idempotency_key`` returns the
existing job id — including jobs that already completed, so a caller
retry after a timeout can never double-send. Without an explicit key,
each submit gets a unique key (no false dedup).

The gateway wraps ``gateway.send()`` with this outbox when durability
is enabled (flag default OFF; per-send ``durable=True`` overrides).
Typing indicators never go through the outbox — fire-and-forget.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Callable

from ...core.ids import new_id
from ...core.logging_setup import get_logger
from ...storage.queue import WorkQueue
from .send_retry import classify_send_error
from .failures import record_failure

__all__ = ["Outbox", "parse_since", "OUTBOX_TOPIC_PREFIX"]

_log = get_logger(__name__)

OUTBOX_TOPIC_PREFIX = "chat_outbox"

#: WorkQueue statuses → receipt states (dead is classified at read time).
_STATE_MAP: dict[str, str] = {
    "ready": "queued",
    "leased": "sending",
    "done": "sent",
    "cancelled": "dropped",
}

_SINCE_RELATIVE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdwSMHDW])\s*$")
_SINCE_MULTIPLIER: dict[str, float] = {
    "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0,
}


def parse_since(value: Any) -> float:
    """Parse a ``--since`` value to a unix timestamp.

    Accepts: relative durations (``30m``, ``1h``, ``7d``, ``1w`` — seconds
    back from now), ISO dates / datetimes (``2026-10-16``,
    ``2026-10-16T10:00:00``; naive values are local time), epoch seconds,
    or a number (passed through). Raises :class:`ValueError` on garbage.
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError(f"cannot parse --since {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    match = _SINCE_RELATIVE.match(text)
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        return time.time() - amount * _SINCE_MULTIPLIER[unit]
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"cannot parse --since {value!r} (want e.g. 1h, 7d, "
            "2026-10-16, 2026-10-16T10:00:00, or epoch seconds)"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.timestamp()


def _topic_for(platform: str, chat_key: str) -> str:
    return f"{OUTBOX_TOPIC_PREFIX}:{platform}:{chat_key}"


def _split_topic(topic: str) -> tuple[str, str]:
    """``chat_outbox:<platform>:<chat_key>`` → ``(platform, chat_key)``."""
    parts = str(topic).split(":", 2)
    if len(parts) != 3 or parts[0] != OUTBOX_TOPIC_PREFIX:
        return "", ""
    return parts[1], parts[2]


def _parse_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class Outbox:
    """Durable chat outbox backed by a :class:`WorkQueue`."""

    def __init__(
        self,
        db: Any,
        *,
        queue: WorkQueue | None = None,
        worker: str = "outbox",
        max_attempts: int = 3,
    ) -> None:
        self.db = db
        self.queue = queue if queue is not None else WorkQueue(db)
        self.worker = worker
        #: Unique to this Outbox instance. Auto-generated idempotency
        #: keys carry it (see :meth:`submit`): two instances over one
        #: database can compute the same ``order_seq`` for a topic (each
        #: scans before the other submits), and without the instance
        #: component their auto keys collide — WorkQueue.enqueue then
        #: returns the *first* job's id for the second submit and that
        #: message is silently never enqueued.
        self._instance = new_id("obx")
        #: Outer attempts per job (WorkQueue requeues). The *inner* retry
        #: (classified, backoff+jitter) happens inside one lease via
        #: send_with_retry; the outer layer only covers crashes and
        #: inner-exhausted transients.
        self.max_attempts = int(max_attempts)
        #: topic → highest order_seq submitted (scanned from the DB once
        #: per topic so a restarted worker resumes the sequence).
        self._seq_cache: dict[str, int] = {}
        self._seq_scanned: set[str] = set()

    # ── submission ─────────────────────────────────────────────────────
    def _next_seq(self, topic: str) -> int:
        if topic not in self._seq_scanned:
            highest = 0
            for row in self._rows_for_topic(topic):
                seq = _parse_json(row.get("payload")).get("order_seq")
                if isinstance(seq, (int, float)) and int(seq) > highest:
                    highest = int(seq)
            self._seq_cache[topic] = highest
            self._seq_scanned.add(topic)
        self._seq_cache[topic] = self._seq_cache.get(topic, 0) + 1
        return self._seq_cache[topic]

    def _find_by_key(self, topic: str, idempotency_key: str) -> str | None:
        """Job id for a prior submit with this key, in ANY status."""
        if not idempotency_key:
            return None
        for row in self._rows_for_topic(topic):
            payload = _parse_json(row.get("payload"))
            if payload.get("idempotency_key") == idempotency_key:
                return str(row.get("id") or "")
        return None

    def submit(
        self,
        *,
        platform: str,
        chat: Any,
        text: str = "",
        reply_to: str = "",
        buttons: Any = None,
        reply_keyboard: Any = None,
        parse_mode: str = "",
        kind: str = "chat_send",
        durable_kind: str = "chat",
        group_id: str = "",
        idempotency_key: str = "",
        media: dict[str, Any] | None = None,
        caption_rest: list[str] | None = None,
    ) -> str:
        """Enqueue one send. Returns the job id (the receipt handle).

        Idempotent on ``idempotency_key``: a repeat submit returns the
        original job id without creating a second row.
        """
        chat_key = str(getattr(chat, "key", "") or chat or "")
        chat_id = str(getattr(chat, "chat_id", "") or "")
        chat_kind = str(getattr(chat, "kind", "") or "")
        thread_id = str(getattr(chat, "thread_id", "") or "")
        topic = _topic_for(platform, chat_key)
        if idempotency_key:
            existing = self._find_by_key(topic, idempotency_key)
            if existing:
                return existing
        order_seq = self._next_seq(topic)
        # Without an explicit key the queueing lock only has to be
        # unique per submit (never a false dedup) — the instance suffix
        # is what guarantees that across Outbox instances/processes
        # sharing one database (see __init__). Explicit keys keep their
        # exact cross-instance dedup semantics.
        key = idempotency_key or (
            f"seq:{platform}:{chat_key}:{order_seq}:{self._instance}")
        payload: dict[str, Any] = {
            "kind": kind,
            "platform": platform,
            "chat_key": chat_key,
            "chat_id": chat_id,
            "chat_kind": chat_kind,
            "thread_id": thread_id,
            "text": text or "",
            "reply_to": reply_to or "",
            "buttons": json.loads(json.dumps(buttons)) if buttons is not None else None,
            "reply_keyboard": reply_keyboard,
            "parse_mode": parse_mode or "",
            "durable_kind": durable_kind,
            "group_id": group_id or new_id("send"),
            "order_seq": order_seq,
            "idempotency_key": key,
        }
        if media is not None:
            payload["media"] = media
        if caption_rest:
            payload["caption_rest"] = list(caption_rest)
        job_id = self.queue.enqueue(
            topic, payload,
            max_attempts=self.max_attempts,
            queueing_lock=key,
        )
        return job_id

    def job_result(self, job_id: str) -> dict[str, Any] | None:
        """Terminal outcome for ``job_id`` (None when not terminal)."""
        job = self.queue.get(job_id)
        if job is None:
            return None
        if job.status == "done":
            row = self.db.query_one(
                f"SELECT result, error FROM {self.queue.TABLE} WHERE id = ?",
                (job_id,),
            )
            stored = _parse_json((row or {}).get("result"))
            return {"ok": True, "message_id": str(stored.get("message_id") or ""),
                    "attempts": job.attempts, "error": ""}
        if job.status == "dead":
            row = self.db.query_one(
                f"SELECT error FROM {self.queue.TABLE} WHERE id = ?",
                (job_id,),
            )
            return {"ok": False, "message_id": "",
                    "attempts": job.attempts,
                    "error": str((row or {}).get("error") or "")}
        return None

    # ── draining ───────────────────────────────────────────────────────
    def _rows_for_topic(self, topic: str) -> list[dict[str, Any]]:
        try:
            return self.db.query(
                f"SELECT * FROM {self.queue.TABLE} WHERE topic = ? "
                "ORDER BY created_at, id",
                (topic,),
            )
        except Exception as exc:  # noqa: BLE001 - receipts/drain never crash the caller
            _log.debug("outbox: topic query failed for %s: %s", topic, exc)
            return []

    def _live_jobs(self, topic: str) -> list[dict[str, Any]]:
        """Live (ready/leased) rows for one chat, in send order."""
        rows = [
            r for r in self._rows_for_topic(topic)
            if str(r.get("status")) in ("ready", "leased")
        ]
        rows.sort(key=lambda r: (
            int(_parse_json(r.get("payload")).get("order_seq") or 0),
            float(r.get("created_at") or 0),
        ))
        return rows

    def _head(self, topic: str) -> dict[str, Any] | None:
        live = self._live_jobs(topic)
        return live[0] if live else None

    def deliver_chat(
        self,
        chat_key: str,
        platform: str,
        send_fn: Callable[[dict[str, Any]], Any],
        *,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Drain one chat's queue in order. Returns per-job outcomes.

        Stops at the first head that is leased elsewhere, in backoff, or
        transiently failed (its outcome is not yet known, so later jobs
        must wait). A terminally failed head kills only its own logical
        message's remaining chunks (same ``group_id``); later messages
        still drain.
        """
        topic = _topic_for(platform, chat_key)
        moment = time.time() if now is None else now
        outcomes: list[dict[str, Any]] = []
        try:
            self.queue.reclaim_expired()
        except Exception as exc:  # noqa: BLE001 - reclaim is best-effort
            _log.debug("outbox: reclaim failed: %s", exc)
        while True:
            head = self._head(topic)
            if head is None:
                break
            if str(head.get("status")) == "leased":
                break  # another worker owns the head — do not jump ahead
            if float(head.get("available_at") or 0) > moment:
                break  # head is in backoff — do not jump ahead
            try:
                leased = self.queue.lease(topic, self.worker, batch=1)
            except Exception as exc:  # noqa: BLE001 - per-chat isolation
                _log.warning("outbox: lease failed for %s: %s", topic, exc)
                break
            if not leased:
                break
            job = leased[0]
            # Belt-and-braces: the leased job must be the head. If a race
            # handed us a later job, put it back and stop.
            fresh_head = self._head(topic)
            if fresh_head is not None and str(fresh_head.get("id")) != str(job.id):
                try:
                    self.queue.requeue(job.id)
                except Exception as exc:  # noqa: BLE001 - best-effort
                    record_failure('outbox.Outbox.deliver_chat', exc)
                break
            payload = dict(job.payload or {})
            try:
                raw = send_fn(payload)
            except Exception as exc:  # noqa: BLE001 - send_fn failure is a send failure
                raw = None
                send_error = f"{type(exc).__name__}: {exc}"
            else:
                send_error = ""
            if isinstance(raw, tuple):
                result, _inner_attempts = raw[0], raw[1] if len(raw) > 1 else 0
            else:
                result = raw
            ok = bool(getattr(result, "ok", False))
            message_id = str(getattr(result, "message_id", "") or "")
            error = str(getattr(result, "error", "") or "") or send_error
            if ok:
                try:
                    self.queue.complete(job.id, result={"message_id": message_id})
                except Exception as exc:  # noqa: BLE001
                    _log.warning("outbox: complete failed for %s: %s", job.id, exc)
                outcomes.append({"job_id": job.id, "ok": True, "state": "sent",
                                 "attempts": job.attempts,
                                 "message_id": message_id, "error": ""})
                continue
            kind = classify_send_error(error or "unknown")
            if kind == "terminal":
                try:
                    self.queue.fail(job.id, error, retry=False)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("outbox: fail failed for %s: %s", job.id, exc)
                outcomes.append({"job_id": job.id, "ok": False,
                                 "state": "failed_terminal",
                                 "attempts": job.attempts,
                                 "message_id": "", "error": error})
                self._cancel_group(topic, str(payload.get("group_id") or ""),
                                   exclude_id=job.id)
                continue
            # Transient, inner retries exhausted: requeue with WorkQueue
            # backoff (or dead when outer attempts are spent), then stop —
            # later jobs wait for this head's outcome.
            try:
                self.queue.fail(job.id, error, retry=True)
            except Exception as exc:  # noqa: BLE001
                _log.warning("outbox: fail failed for %s: %s", job.id, exc)
            outcomes.append({"job_id": job.id, "ok": False,
                             "state": "failed_transient",
                             "attempts": job.attempts,
                             "message_id": "", "error": error})
            break
        return outcomes

    def _cancel_group(self, topic: str, group_id: str, *, exclude_id: str) -> None:
        """Cancel a terminally-failed message's remaining chunks."""
        if not group_id:
            return
        for row in self._live_jobs(topic):
            if str(row.get("id")) == exclude_id:
                continue
            payload = _parse_json(row.get("payload"))
            if str(payload.get("group_id") or "") == group_id:
                try:
                    self.queue.cancel(str(row.get("id")))
                except Exception as exc:  # noqa: BLE001
                    _log.debug("outbox: cancel failed for %s: %s",
                               row.get("id"), exc)

    def drain_all(
        self,
        send_fn: Callable[[dict[str, Any]], Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Drain every outbox topic. One dead platform cannot jam the
        queue: failures are isolated per topic and draining continues."""
        results: dict[str, list[dict[str, Any]]] = {}
        try:
            topics = [
                str(r.get("topic") or "")
                for r in self.db.query(
                    f"SELECT DISTINCT topic FROM {self.queue.TABLE} "
                    f"WHERE topic LIKE ?",
                    (f"{OUTBOX_TOPIC_PREFIX}:%",),
                )
            ]
        except Exception as exc:  # noqa: BLE001
            _log.warning("outbox: topic scan failed: %s", exc)
            return results
        for topic in sorted(topics):
            platform, chat_key = _split_topic(topic)
            if not platform or not chat_key:
                continue
            try:
                results[topic] = self.deliver_chat(chat_key, platform, send_fn)
            except Exception as exc:  # noqa: BLE001 - per-topic isolation
                _log.warning("outbox: drain failed for %s: %s", topic, exc)
                results[topic] = [{"job_id": "", "ok": False,
                                   "state": "failed_transient",
                                   "attempts": 0, "message_id": "",
                                   "error": f"{type(exc).__name__}: {exc}"}]
        return results

    # ── receipts (G4) ──────────────────────────────────────────────────
    def receipts(
        self,
        *,
        chat_key: str = "",
        platform: str = "",
        since: float = 0.0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Delivery receipts, newest first.

        ``chat_key`` matches a full ``platform:chat_id`` key exactly, or a
        bare chat id (matched against the stored chat id). ``since`` is a
        unix timestamp; rows older than it are excluded.
        """
        try:
            rows = self.db.query(
                f"SELECT * FROM {self.queue.TABLE} "
                f"WHERE topic LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"{OUTBOX_TOPIC_PREFIX}:%", max(int(limit) * 5, 200)),
            )
        except Exception as exc:  # noqa: BLE001 - receipts never crash the CLI
            _log.debug("outbox: receipts query failed: %s", exc)
            return []
        want_chat = (chat_key or "").strip()
        want_platform = (platform or "").strip().lower()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = _parse_json(row.get("payload"))
            row_platform = str(payload.get("platform") or "")
            row_chat_key = str(payload.get("chat_key") or "")
            row_chat_id = str(payload.get("chat_id") or "")
            if want_platform and row_platform.lower() != want_platform:
                continue
            if want_chat:
                if ":" in want_chat:
                    if row_chat_key != want_chat:
                        continue
                elif row_chat_id != want_chat and row_chat_key != want_chat:
                    continue
            created = float(row.get("created_at") or 0)
            if since and created < since:
                continue
            status = str(row.get("status") or "")
            if status == "dead":
                error = str(row.get("error") or "")
                state = ("failed_terminal"
                         if classify_send_error(error) == "terminal"
                         else "failed_transient")
            else:
                state = _STATE_MAP.get(status, status or "unknown")
            stored = _parse_json(row.get("result"))
            out.append({
                "id": str(row.get("id") or ""),
                "platform": row_platform,
                "chat_key": row_chat_key,
                "state": state,
                "attempts": int(row.get("attempts") or 0),
                "message_id": str(stored.get("message_id") or ""),
                "error": str(row.get("error") or ""),
                "created_at": created,
                "sent_at": float(row.get("updated_at") or 0),
                "durable_kind": str(payload.get("durable_kind") or ""),
                "text_preview": (str(payload.get("text") or "")[:120]),
            })
            if len(out) >= max(1, int(limit)):
                break
        return out
