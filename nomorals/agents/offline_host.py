"""Offline fallback host (audit gap R2-42).

The offline package (``nomorals/offline/``) was complete — a persisted
FIFO queue, a merge engine with conflict flagging, a per-feature
degraded-capability matrix — but it had ZERO production callers: the
docstring says "the host calls ``sync()`` when it knows it is back
online" and no host did. A failed send lost the action, reconnecting
synced nothing, and the feature gates never consulted the matrix.

This module is that host. It lives in ``agents/`` (layer 5), not in
``offline/`` (layer 0, stdlib only): it needs the gateway, the
notifier, and the email applier, all above layer 0. It wires ONE
integration, not a second system:

* enqueue on send failure — the one genuinely-lost action today is
  an approved email whose transport died (see
  ``agents/email_triage.py``: ``DraftQueue.send``). Chat replies are
  NOT enqueued here: the gateway's durable outbox already persists
  them, and a second queue would double-deliver.
* ``sync()`` on reconnect — a durable 5-minute tick
  (``__offline_sync_tick__``) plus a fast path on the first
  successful chat send after a failure. Both call ``drain()``.
* matrix lookup in the feature gates — ``feature_status()`` is the
  consult ``/search`` / ``/news`` use through
  ``PartnerRuntime._offline_feature_refusal``.

Connectivity is OBSERVED, never probed (the 2026-11-15 offline-fallback
survey's deliberate design): a send failure marks the host offline, a
send success clears it. There is no central ``is_online`` check.

Delivery is ``Notifier.publish(kind, title, body)`` — the Notifier
has no ``send`` method. Only NEW conflicts publish, once each; a
flushed queue is silent (the deliveries speak for themselves).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..offline import (
    OFFLINE_MATRIX,
    OfflineQueue,
    SyncEngine,
    matrix_table,
    offline_level,
)

_log = logging.getLogger("nomorals.offline_host")

#: Durable scheduler job + routine tool for the reconnect drain.
OFFLINE_SYNC_JOB = "offline sync tick"
OFFLINE_SYNC_TOOL = "__offline_sync_tick__"
OFFLINE_SYNC_SPEC = "every 5m"

#: Kinds whose applier performs a network call. A drain that applies
#: at least one action proves the transport is alive again.
_NETWORK_KINDS = frozenset({"email_send"})

#: The process-wide default host (see ``get_offline_host``).
_DEFAULT_HOST: OfflineHost | None = None


class OfflineHost:
    """The host for the offline store-and-forward package.

    Owns one :class:`OfflineQueue` (default root ``~/.devon/offline/``),
    an applier registry keyed by action kind, the persisted version
    map the sync engine merges against, the set of conflicts already
    published to the owner, and the observed-connectivity state.
    """

    def __init__(self, root: Path | str | None = None,
                 queue: OfflineQueue | None = None) -> None:
        self.queue = queue if queue is not None else OfflineQueue(root)
        self._appliers: dict[str, Callable[[Any], Any]] = {}
        self._versions: dict[str, int] = self._load_json(
            self._versions_path(), {})
        self._notified: set[str] = set(self._load_json(
            self._notified_path(), []))
        #: Timestamp of the first send failure of the current episode,
        #: or None when the transport was last observed healthy.
        self._offline_since: float | None = None

    # -- persistence ----------------------------------------------------------

    def _versions_path(self) -> Path:
        return self.queue.root / "versions.json"

    def _notified_path(self) -> Path:
        return self.queue.root / "notified.json"

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - corrupt state degrades, never crashes
            _log.warning("offline host: could not load %s: %s", path, exc)
        return default

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, sort_keys=True),
                            encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - state save is best-effort
            _log.warning("offline host: could not save %s: %s", path, exc)

    # -- versions --------------------------------------------------------------

    def set_version(self, entity: str, version: int) -> None:
        """Record the live version of ``entity`` (the sync baseline)."""
        self._versions[str(entity)] = int(version)
        self._save_json(self._versions_path(), self._versions)

    def get_version(self, entity: str) -> int | None:
        return self._versions.get(str(entity))

    # -- connectivity observation ----------------------------------------------

    @property
    def is_offline(self) -> bool:
        return self._offline_since is not None

    @property
    def offline_since(self) -> float | None:
        return self._offline_since

    def note_send_failure(self, error: str) -> None:
        """A send just failed: the transport is observed down."""
        if self._offline_since is None:
            self._offline_since = time.time()
            _log.warning("offline host: send failed, marking offline: %s",
                         error)

    def note_send_success(self) -> None:
        """A send just succeeded: the transport is observed healthy."""
        if self._offline_since is not None:
            _log.info("offline host: send succeeded, back online")
        self._offline_since = None

    # -- enqueue ----------------------------------------------------------------

    def enqueue_failed(self, kind: str, payload: dict[str, Any] | None = None,
                       *, entity: str = "",
                       base_version: int | None = None):
        """Queue an action whose send failed. Returns the stored action.

        An action only reaches here because a send failed, so the host
        is marked offline as well — the gates must not claim the
        network is fine while a queued action waits for it.
        """
        action = self.queue.enqueue(kind, payload, entity=entity,
                                    base_version=base_version)
        self.note_send_failure(f"queued {kind} after a failed send")
        return action

    # -- appliers -----------------------------------------------------------------

    def register_applier(self, kind: str,
                         fn: Callable[[Any], Any]) -> None:
        """Register how a queued action of ``kind`` is performed."""
        self._appliers[str(kind)] = fn

    def _dispatch(self, action: Any) -> Any:
        fn = self._appliers.get(action.kind)
        if fn is None:
            raise RuntimeError(
                f"no applier for offline action kind {action.kind!r}")
        return fn(action)

    # -- matrix consult -------------------------------------------------------------

    def feature_status(self, feature: str) -> dict[str, Any]:
        """How ``feature`` behaves right now, per the offline matrix.

        Online: allowed, level ``"online"``. Offline: the matrix level
        (``full`` / ``degraded`` / ``unavailable``) plus its note;
        ``allowed`` is False only for ``unavailable`` — a degraded
        feature queues instead of refusing.
        """
        if not self.is_offline:
            return {"feature": feature, "offline": False,
                    "level": "online", "allowed": True, "note": ""}
        level = offline_level(feature)
        note = str(OFFLINE_MATRIX.get(feature, {}).get("notes", ""))
        return {"feature": feature, "offline": True, "level": level,
                "allowed": level != "unavailable", "note": note}

    # -- drain ----------------------------------------------------------------------

    def drain(self, *, gateway: Any = None,
              notifier: Any = None) -> dict[str, Any]:
        """Drain the queue through the sync engine; pump the outbox.

        Requeues ``failed`` actions as ``pending`` first — the engine
        never retries a ``failed`` row on its own, and a transport
        failure is exactly the case a later drain exists for. New
        conflicts publish once via the notifier; a flushed queue is
        silent. Never raises: the summary always reports what
        happened.
        """
        summary: dict[str, Any] = {
            "ok": True, "applied": [], "conflicts": [], "failed": [],
            "outbox": {}, "pending_after": 0,
        }
        try:
            for action in self.queue.all():
                if action.status == "failed":
                    action.status = "pending"
                    action.error = ""
                    self.queue.update(action)
            engine = SyncEngine(self.queue, self._dispatch, self._versions)
            pending_kinds = {a.id: a.kind for a in self.queue.pending()}
            result = engine.sync()
            self._versions = dict(engine.versions)
            self._save_json(self._versions_path(), self._versions)
            summary["applied"] = list(result.applied)
            summary["failed"] = list(result.failed)
            summary["conflicts"] = [c.to_dict() for c in result.conflicts]
            summary["ok"] = result.ok
            if notifier is not None and result.conflicts:
                for conflict in result.conflicts:
                    if conflict.action_id in self._notified:
                        continue
                    try:
                        self._publish_conflict(notifier, conflict)
                    except Exception as exc:  # noqa: BLE001 - publish is best-effort
                        _log.warning("offline host: conflict publish failed: %s",
                                     exc)
                        continue
                    self._notified.add(conflict.action_id)
                self._save_json(self._notified_path(),
                                sorted(self._notified))
            # A drain that applied a network action proves the
            # transport is alive again. A local-only drain says
            # nothing about the network — the observation stands
            # until the next send proves otherwise.
            if result.applied and any(
                    pending_kinds.get(aid) in _NETWORK_KINDS
                    for aid in result.applied):
                self.note_send_success()
        except Exception as exc:  # noqa: BLE001 - a drain never raises
            _log.warning("offline host: drain failed: %s", exc)
            summary["ok"] = False
            summary["error"] = f"{type(exc).__name__}: {exc}"
        if gateway is not None:
            try:
                summary["outbox"] = gateway.drain_queued()
            except Exception as exc:  # noqa: BLE001 - one bad pump never hides the queue result
                _log.warning("offline host: outbox pump failed: %s", exc)
                summary["outbox"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        with contextlib.suppress(Exception):
            summary["pending_after"] = self.queue.count_pending()
        return summary

    def _publish_conflict(self, notifier: Any, conflict: Any) -> None:
        title = f"Offline sync conflict — {conflict.entity}"
        body = (
            f"Your offline change to {conflict.entity} was based on "
            f"version {conflict.base_version}, but the live version is "
            f"now {conflict.current_version}. Neither was dropped: your "
            f"queued change is held as a conflict and the live version "
            f"is untouched."
        )
        notifier.publish("offline-sync", title, body)


def get_offline_host(root: Path | str | None = None) -> OfflineHost:
    """The process-wide default host.

    The runtime's send observations, the email seam's enqueue, and
    the reconnect tick all share this one host — one queue file, one
    offline flag. Tests build fresh ``OfflineHost(root=tmp)`` instead.
    """
    global _DEFAULT_HOST
    if _DEFAULT_HOST is None:
        _DEFAULT_HOST = OfflineHost(root=root)
    return _DEFAULT_HOST


# ── default appliers ──────────────────────────────────────────────────────────


def _apply_email_send(action: Any, *, context: Any) -> Any:
    """Send a queued approved email, exactly as approved.

    Guards: a draft already marked ``sent`` is skipped (never
    double-sent); a draft the owner discarded meanwhile is skipped
    (the owner's later decision wins). The send itself re-uses the
    triage connector — approval is durable in the draft store, so no
    new confirmation is invented. ``mark_sent`` runs only after the
    send succeeds, and never raises: a post-send bookkeeping failure
    must not mark the action failed and re-send it next drain.
    """
    from .email_triage import DraftQueue, _get_gmail

    payload = dict(action.payload or {})
    draft_id = str(payload.get("draft_id") or "")
    gmail = _get_gmail(context)
    if gmail is None:
        raise RuntimeError(
            "gmail isn't connected — queued email can't send yet")
    if draft_id:
        queue = DraftQueue(context=context)
        draft = queue.get(draft_id)
        if draft is not None and draft.status == "sent":
            return {"skipped": "already sent"}
        if draft is not None and draft.status == "discarded":
            return {"skipped": "discarded"}
    result = gmail.send_message(
        payload.get("to", ""), payload.get("subject", ""),
        payload.get("body", ""), confirmed=True)
    if draft_id:
        with contextlib.suppress(Exception):
            DraftQueue(context=context).mark_sent(draft_id)
    return result


def register_default_appliers(host: OfflineHost, context: Any) -> None:
    """Register the production appliers on ``host`` (idempotent)."""
    host.register_applier(
        "email_send", lambda a: _apply_email_send(a, context=context))


# ── tick driver ───────────────────────────────────────────────────────────────


def run_offline_sync_tick_once(context: Any, *,
                               host: OfflineHost | None = None,
                               gateway: Any = None,
                               notifier: Any = None) -> dict[str, Any]:
    """One reconnect drain. Never raises — a tick that fails reports.

    The 5-minute durable job calls this; the gateway comes from the
    live context's extras when the caller does not pass one.
    """
    if host is None:
        host = get_offline_host()
    if gateway is None:
        extras = getattr(context, "extras", None) or {}
        if isinstance(extras, dict):
            gateway = extras.get("gateway")
    try:
        register_default_appliers(host, context)
        return host.drain(gateway=gateway, notifier=notifier)
    except Exception as exc:  # noqa: BLE001 - a tick never raises
        return {"ok": False,
                "error": f"offline sync tick failed: {exc}"}


def ensure_offline_sync_job(scheduler: Any, *,
                            spec: str = OFFLINE_SYNC_SPEC) -> Any:
    """Register the durable offline sync tick on the live scheduler.

    Mirror of the render / swarm-board / digest / watch-loop / drift /
    workflow / continual ensures: a routine tool job, one per
    scheduler, replaced when its spec goes stale. Never raises —
    scheduling is best-effort; a missing or broken scheduler yields
    None.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name != OFFLINE_SYNC_JOB:
                continue
            job_spec = (job.get("spec") if isinstance(job, dict)
                        else getattr(job, "spec", None))
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                job_id = (job.get("id") if isinstance(job, dict)
                          else getattr(job, "id", None))
                scheduler.remove(job_id)
            break
        return scheduler.add(OFFLINE_SYNC_JOB, spec, "tool",
                             {"tool": OFFLINE_SYNC_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


#: Failure-ledger site for BrainGate held/recovered episodes (see
#: runtime.PartnerRuntime._send_reply — the gate records the held and
#: recovered events there).
BRAINGATE_SITE = "runtime.PartnerRuntime._send_reply:braingate"

#: The held count a Recovered entry reports, e.g.
#: "(3 held during the episode opened on telegram, 600s)".
_HELD_COUNT_RE = re.compile(r"\((\d+) held during")


def summarize_offline_state(
    *,
    queue: Any = None,
    scheduler: Any = None,
    failure_entries: list[dict[str, Any]] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """The durable offline state, read-only. Never raises.

    Four sections, all from real durable state — nothing invented:

    * ``queue`` — the persisted offline queue (rows exist only after a
      failed send; ``enqueue_failed`` is the only writer). Counts by
      status and kind, plus the oldest pending action's age.
    * ``sync_job`` — the durable 5-minute drain job (name/spec/enabled/
      next run) when a scheduler is given.
    * ``braingate`` — held/recovered episodes from the failure ledger:
      a Held opens an episode, a later Recovered closes it; an open
      episode's held count is unknown and is said so.
    * ``matrix`` — the degraded-capability matrix (feature/level/notes).

    What this read CANNOT say: whether the live process is offline
    right now. Connectivity is observed in memory by the live runtime
    (a send failure marks it offline, a send success clears it); a
    durable read sees only the aftermath — a non-empty queue means a
    send failed and the action is still waiting. That limit is stated
    in ``notes``, not hidden.
    """
    now_f = float(now if now is not None else time.time())

    q: dict[str, Any] = {
        "pending": 0, "failed": 0, "synced": 0, "conflict": 0, "total": 0,
        "by_kind": {}, "oldest_pending_id": None,
        "oldest_pending_age_s": None, "rows": [], "error": "",
    }
    actions: list[Any] = []
    try:
        if queue is None:
            queue = OfflineQueue()
        actions = list(queue.all())
    except Exception as exc:  # noqa: BLE001 - a broken queue reads as empty
        q["error"] = f"{type(exc).__name__}: {exc}"
    for action in actions:
        q["total"] += 1
        kind = getattr(action, "kind", "") or ""
        q["by_kind"][kind] = q["by_kind"].get(kind, 0) + 1
        status = getattr(action, "status", "") or ""
        if status in ("pending", "failed", "synced", "conflict"):
            q[status] += 1
        if status in ("pending", "failed", "conflict"):
            # Identity only — never the payload (it may hold a draft
            # body); the queue verb lists the same rows.
            q["rows"].append({
                "id": getattr(action, "id", None),
                "kind": kind,
                "entity": getattr(action, "entity", "") or "",
                "status": status,
                "error": getattr(action, "error", "") or "",
            })
    pending = [a for a in actions
               if (getattr(a, "status", "") or "") == "pending"]
    if pending:
        oldest = min(pending, key=lambda a: float(getattr(a, "created_ts", 0) or 0))
        q["oldest_pending_id"] = getattr(oldest, "id", None)
        q["oldest_pending_age_s"] = max(
            0.0, now_f - float(getattr(oldest, "created_ts", 0) or 0))

    job: dict[str, Any] = {
        "scheduled": False, "spec": "", "enabled": False,
        "next_run_iso": None, "error": "",
    }
    if scheduler is not None:
        try:
            for item in scheduler.list_jobs() or []:
                name = (item.get("name") if isinstance(item, dict)
                        else getattr(item, "name", None))
                if name != OFFLINE_SYNC_JOB:
                    continue
                if isinstance(item, dict):
                    job = {
                        "scheduled": True,
                        "spec": str(item.get("spec") or ""),
                        "enabled": bool(item.get("enabled", True)),
                        "next_run_iso": item.get("next_run"),
                        "error": "",
                    }
                else:
                    job = {
                        "scheduled": True,
                        "spec": str(getattr(item, "spec", "") or ""),
                        "enabled": bool(getattr(item, "enabled", True)),
                        "next_run_iso": getattr(item, "next_run", None),
                        "error": "",
                    }
                break
        except Exception as exc:  # noqa: BLE001 - scheduler read is best-effort
            job["error"] = f"{type(exc).__name__}: {exc}"

    bg: dict[str, Any] = {
        "episodes": 0, "open": 0, "held_closed": 0,
        "held_open_unknown": False, "last_episode_ts": None, "error": "",
    }
    entries = failure_entries
    if entries is None:
        try:
            from ..core.failure_store import Store

            entries, _total = Store("core").load()
        except Exception as exc:  # noqa: BLE001 - ledger read is best-effort
            entries = []
            bg["error"] = f"{type(exc).__name__}: {exc}"
    try:
        gate = sorted(
            (e for e in (entries or [])
             if isinstance(e, dict) and e.get("site") == BRAINGATE_SITE),
            key=lambda e: float(e.get("ts") or 0.0))
        open_count = 0
        for entry in gate:
            exc_type = entry.get("exc_type") or ""
            if exc_type == "BrainGateHeld":
                bg["episodes"] += 1
                open_count += 1
                bg["last_episode_ts"] = float(entry.get("ts") or 0.0)
            elif exc_type == "BrainGateRecovered":
                open_count = max(0, open_count - 1)
                match = _HELD_COUNT_RE.search(str(entry.get("message") or ""))
                if match:
                    bg["held_closed"] += int(match.group(1))
        bg["open"] = open_count
        bg["held_open_unknown"] = open_count > 0
    except Exception as exc:  # noqa: BLE001 - episode pairing never fails the read
        bg["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "queue": q,
        "sync_job": job,
        "braingate": bg,
        "matrix": matrix_table(),
        "notes": [
            "the live runtime observes connectivity in memory — this read "
            "sees the durable aftermath only. A non-empty queue means a "
            "send failed and the action is still waiting.",
        ],
    }
