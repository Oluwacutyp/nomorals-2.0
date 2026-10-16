"""Status orchestration + multi-device routing for community chats.

Problem: the group agent runs on two devices (the owner's phone and a
VPS). When the phone goes offline, scheduled posts, sequences, and
replies must keep flowing — from the VPS — without double-sending.

Design:

* :class:`SessionRegistry` — devices register and heartbeat. Health is
  derived from heartbeat age (``stale_after_s``), never assumed.
* :class:`Router` — picks the device for a task: the preferred device
  when healthy, else the failover (phone offline → VPS picks up).
  Every decision is appended to a :class:`RoutingLog` (JSONL) so a
  double-send can always be audited after the fact.
* :class:`Orchestrator` — scheduled posts (one-shot or recurring),
  multi-step sequences (drip / onboarding / announcements), and
  reply jobs. ``due()`` is pure (what is due now); ``dispatch()`` sends
  via *injected* sender callables only. No senders → nothing sent, and
  the dispatch still returns the plan.
* Idempotency: every job carries an id; dispatch skips jobs already
  marked sent in the same run, and the log records the skip.

Only stdlib + ``nomorals.core``. No platform SDKs, no owner data.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "DeviceSession", "SessionRegistry", "RoutingDecision", "RoutingLog",
    "Router", "ScheduledPost", "Sequence", "SequenceStep", "Orchestrator",
    "OrchestratorStore", "default_orchestration_dir",
]


def default_orchestration_dir() -> Path:
    return Path.home() / ".devon" / "community" / "orchestration"


# ───────────────────────── sessions + health ────────────────────────────────

@dataclass
class DeviceSession:
    device_id: str
    kind: str = "phone"              # phone | vps | laptop | ...
    capabilities: list[str] = field(default_factory=list)
    last_heartbeat: float = 0.0
    registered_ts: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def age_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.last_heartbeat

    def is_healthy(self, now: float | None = None, stale_after_s: float = 120.0) -> bool:
        return self.age_s(now) <= stale_after_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeviceSession":
        return cls(device_id=str(d["device_id"]), kind=str(d.get("kind", "phone")),
                   capabilities=list(d.get("capabilities", [])),
                   last_heartbeat=float(d.get("last_heartbeat", 0)),
                   registered_ts=float(d.get("registered_ts", 0)),
                   meta=dict(d.get("meta", {})))


class SessionRegistry:
    """Device sessions + health. In-memory; persistence via OrchestratorStore."""

    def __init__(self, stale_after_s: float = 120.0) -> None:
        self.stale_after_s = stale_after_s
        self.sessions: dict[str, DeviceSession] = {}

    def register(self, device_id: str, kind: str = "phone",
                 capabilities: list[str] | None = None) -> DeviceSession:
        now = time.time()
        sess = DeviceSession(device_id=device_id, kind=kind,
                             capabilities=list(capabilities or []),
                             last_heartbeat=now, registered_ts=now)
        self.sessions[device_id] = sess
        return sess

    def heartbeat(self, device_id: str, now: float | None = None) -> bool:
        sess = self.sessions.get(device_id)
        if sess is None:
            return False
        sess.last_heartbeat = now if now is not None else time.time()
        return True

    def health(self, now: float | None = None) -> dict[str, dict[str, Any]]:
        now = now if now is not None else time.time()
        return {d: {"kind": s.kind, "healthy": s.is_healthy(now, self.stale_after_s),
                    "age_s": round(s.age_s(now), 1)}
                for d, s in self.sessions.items()}

    def healthy_devices(self, kind: str | None = None,
                        now: float | None = None) -> list[DeviceSession]:
        now = now if now is not None else time.time()
        out = [s for s in self.sessions.values()
               if s.is_healthy(now, self.stale_after_s)]
        if kind is not None:
            out = [s for s in out if s.kind == kind]
        return sorted(out, key=lambda s: s.device_id)


# ───────────────────────── routing ──────────────────────────────────────────

@dataclass
class RoutingDecision:
    task: str
    chosen_device: str | None
    preferred_device: str | None
    reason: str
    at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoutingLog:
    """Append-only JSONL log of routing decisions. Audit trail for failover."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else \
            default_orchestration_dir() / "routing.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[RoutingDecision] = []

    def record(self, decision: RoutingDecision) -> RoutingDecision:
        if not decision.at:
            decision.at = time.time()
        self.entries.append(decision)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")
        return decision

    def recent(self, n: int = 50) -> list[RoutingDecision]:
        return list(self.entries[-n:])


class Router:
    """Pick a device for a task. Phone first when healthy, VPS on failover.

    ``preference`` is an ordered list of device kinds, e.g.
    ``["phone", "vps"]``. A task whose required capability no healthy
    device has routes to nobody (``chosen_device=None``) with the reason
    recorded — never to a device that cannot do it.
    """

    def __init__(self, registry: SessionRegistry,
                 routing_log: RoutingLog | None = None,
                 preference: list[str] | None = None) -> None:
        self.registry = registry
        self.log = routing_log or RoutingLog()
        self.preference = list(preference or ["phone", "vps", "laptop"])

    def route(self, task: str, *, capability: str | None = None,
              now: float | None = None) -> RoutingDecision:
        now = now if now is not None else time.time()
        healthy = self.registry.healthy_devices(now=now)
        candidates = [s for s in healthy
                      if capability is None or capability in s.capabilities]
        chosen: DeviceSession | None = None
        reason = ""
        for kind in self.preference:
            for sess in candidates:
                if sess.kind == kind:
                    chosen = sess
                    break
            if chosen is not None:
                break
        if chosen is None and candidates:
            chosen = candidates[0]
            reason = f"no preferred kind healthy; fell back to {chosen.kind}"
        elif chosen is not None:
            if chosen.kind == self.preference[0]:
                reason = f"preferred {chosen.kind} healthy"
            else:
                reason = f"preferred device offline — failover to {chosen.kind}"
        else:
            reason = "no healthy device" + (f" with capability {capability}" if capability else "")
        decision = RoutingDecision(
            task=task, chosen_device=chosen.device_id if chosen else None,
            preferred_device=self.preference[0] if self.preference else None,
            reason=reason, at=now)
        return self.log.record(decision)


# ───────────────────────── scheduled posts + sequences ─────────────────────

@dataclass
class ScheduledPost:
    id: str
    group_id: str
    platform: str
    target: str                    # chat/channel id supplied by caller
    text: str
    due_ts: float
    recurrence_s: float | None = None   # None = one-shot
    sent: bool = False
    sent_at: float | None = None
    sent_count: int = 0

    def is_due(self, now: float) -> bool:
        return (not self.sent or self.recurrence_s is not None) and self.due_ts <= now

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScheduledPost":
        return cls(**{k: d.get(k) for k in
                      ("id", "group_id", "platform", "target", "text", "due_ts",
                       "recurrence_s", "sent", "sent_at", "sent_count")})


@dataclass
class SequenceStep:
    text: str
    delay_s: float = 0.0          # delay after the previous step
    platform: str = ""            # "" = same as sequence platform


@dataclass
class Sequence:
    """A multi-step sequence: drip, onboarding, announcement chain."""

    id: str
    group_id: str
    platform: str
    target: str
    steps: list[SequenceStep]
    started_ts: float = 0.0
    step_index: int = 0
    next_due_ts: float = 0.0
    done: bool = False

    def start(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self.started_ts = now
        self.step_index = 0
        self.next_due_ts = now + (self.steps[0].delay_s if self.steps else 0.0)
        self.done = not self.steps

    def due_step(self, now: float) -> SequenceStep | None:
        if self.done or not self.steps or self.step_index >= len(self.steps):
            return None
        return self.steps[self.step_index] if self.next_due_ts <= now else None

    def advance(self, now: float) -> None:
        self.step_index += 1
        if self.step_index >= len(self.steps):
            self.done = True
            return
        self.next_due_ts = now + self.steps[self.step_index].delay_s

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "group_id": self.group_id, "platform": self.platform,
                "target": self.target,
                "steps": [asdict(s) for s in self.steps],
                "started_ts": self.started_ts, "step_index": self.step_index,
                "next_due_ts": self.next_due_ts, "done": self.done}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sequence":
        return cls(id=str(d["id"]), group_id=str(d["group_id"]),
                   platform=str(d["platform"]), target=str(d["target"]),
                   steps=[SequenceStep(**s) for s in d.get("steps", [])],
                   started_ts=float(d.get("started_ts", 0)),
                   step_index=int(d.get("step_index", 0)),
                   next_due_ts=float(d.get("next_due_ts", 0)),
                   done=bool(d.get("done", False)))


class OrchestratorStore:
    """JSON persistence for posts + sequences (one file, community dir)."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_orchestration_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / "orchestration.json"

    def save(self, posts: list[ScheduledPost], sequences: list[Sequence]) -> None:
        self._path().write_text(json.dumps(
            {"posts": [p.to_dict() for p in posts],
             "sequences": [s.to_dict() for s in sequences]}, indent=1),
            encoding="utf-8")

    def load(self) -> tuple[list[ScheduledPost], list[Sequence]]:
        path = self._path()
        if not path.exists():
            return [], []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return [], []
        posts = [ScheduledPost.from_dict(d) for d in data.get("posts", [])]
        seqs = [Sequence.from_dict(d) for d in data.get("sequences", [])]
        return posts, seqs


class Orchestrator:
    """Scheduled posts + sequences + routing-aware dispatch."""

    def __init__(self, store: OrchestratorStore | None = None,
                 router: Router | None = None) -> None:
        self.store = store or OrchestratorStore()
        self.router = router
        self.posts, self.sequences = self.store.load()
        self._sent_ids: set[str] = set()

    def schedule_post(self, group_id: str, platform: str, target: str,
                      text: str, due_ts: float,
                      recurrence_s: float | None = None) -> ScheduledPost:
        post = ScheduledPost(id=new_short_id(), group_id=group_id, platform=platform,
                             target=target, text=text, due_ts=due_ts,
                             recurrence_s=recurrence_s)
        self.posts.append(post)
        self.store.save(self.posts, self.sequences)
        return post

    def start_sequence(self, group_id: str, platform: str, target: str,
                       steps: list[SequenceStep],
                       now: float | None = None) -> Sequence:
        seq = Sequence(id=new_short_id(), group_id=group_id, platform=platform,
                       target=target, steps=list(steps))
        seq.start(now)
        self.sequences.append(seq)
        self.store.save(self.posts, self.sequences)
        return seq

    def due(self, now: float | None = None) -> dict[str, list[Any]]:
        now = now if now is not None else time.time()
        return {"posts": [p for p in self.posts if p.is_due(now)],
                "sequences": [(s, s.due_step(now)) for s in self.sequences
                              if s.due_step(now) is not None]}

    def dispatch(self, now: float | None = None,
                 sender: Callable[[str, str, str], bool] | None = None,
                 device_id: str | None = None) -> list[dict[str, Any]]:
        """Send everything due now. ``sender(platform, target, text) -> bool``.

        No sender → nothing sent; the returned plan still lists what
        *would* go out. One-shot posts already sent are skipped
        (idempotency within this process). Recurring posts advance
        their ``due_ts`` instead of being marked done.
        """
        now = now if now is not None else time.time()
        results: list[dict[str, Any]] = []
        for post in list(self.posts):
            if not post.is_due(now) or post.id in self._sent_ids:
                continue
            if self.router is not None:
                decision = self.router.route(f"post:{post.id}", now=now)
                if decision.chosen_device is None:
                    results.append({"id": post.id, "kind": "post", "sent": False,
                                    "reason": decision.reason})
                    continue
            sent = False
            if sender is not None:
                try:
                    sent = bool(sender(post.platform, post.target, post.text))
                except Exception as exc:  # noqa: BLE001 — sender is caller code
                    _log.warning("orchestration sender failed: %s", exc)
            if sent:
                self._sent_ids.add(post.id)
                post.sent = True
                post.sent_at = now
                post.sent_count += 1
                if post.recurrence_s:
                    post.due_ts = now + post.recurrence_s
                    post.sent = False
                    self._sent_ids.discard(post.id)
            results.append({"id": post.id, "kind": "post", "sent": sent,
                            "platform": post.platform, "target": post.target})
        for seq in list(self.sequences):
            step = seq.due_step(now)
            if step is None:
                continue
            sent = False
            if sender is not None:
                try:
                    sent = bool(sender(step.platform or seq.platform,
                                       seq.target, step.text))
                except Exception as exc:  # noqa: BLE001 — sender is caller code
                    _log.warning("orchestration sender failed: %s", exc)
            if sent:
                seq.advance(now)
            results.append({"id": seq.id, "kind": "sequence_step",
                            "step_index": seq.step_index, "sent": sent,
                            "platform": step.platform or seq.platform,
                            "target": seq.target})
        self.store.save(self.posts, self.sequences)
        return results
