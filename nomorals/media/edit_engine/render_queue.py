"""Render queue first-class (EDIT_GEN_UPGRADE 2.4).

* **Resumable at segment level** — a render is a list of segments
  (cuts / chapters / time ranges). Segment state persists to a JSON
  sidecar, so a crashed / interrupted render resumes at the first
  unfinished segment instead of restarting.
* **Profile-aware, never silently downgraded** — on ``termux`` /
  ``mobile`` / ``embedded`` a full-res render is refused unless the
  caller explicitly passes ``allow_full_res=True``; the default path
  renders a proxy/thumbnail and says so in the result
  (``degraded=True`` + ``reason``). No silent downgrade anywhere.
* **Chunked + progress-streamed** — :meth:`RenderQueue.run` yields
  progress events (``{"segment", "state", "progress"}``) the chat layer
  can stream; :meth:`RenderQueue.run_all` collects them.
* **Honest failure** — :class:`RenderFailure` names the model / VRAM /
  degraded path that failed, and which segment failed, so the owner
  sees *why*, not just "render failed".
* **Location-aware / work-stealing** — segments are claimed by workers
  (``worker_id`` + ``location``). A stale claim (worker gone longer
  than ``claim_ttl_s``) is stolen by another worker instead of
  blocking the queue forever.

The queue orchestrates *callables* (segment renderers) — it never
re-implements rendering. Pass the real engine renderer
(:func:`edit_engine.spec.render`, :func:`pure_render.render_pure`,
or a segment ffmpeg call) as ``segment_fn``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from ...media_edit.videos import MediaEditError

__all__ = [
    "RenderFailure",
    "Segment",
    "PROXY_HEIGHT",
    "TERMUX_PROFILES",
    "resolve_render_profile",
    "RenderQueue",
]

PROXY_HEIGHT = 480
TERMUX_PROFILES = frozenset({"termux", "mobile", "embedded"})

#: profiles that may render full-res without an explicit opt-in.
FULL_RES_PROFILES = frozenset({"workstation", "pc", "vps", "laptop"})


@dataclass
class RenderFailure(Exception):
    """Honest render failure: what failed, where, and on what hardware."""
    message: str
    segment_id: str = ""
    model: str = ""
    vram: str = ""
    degraded_path: str = ""
    profile: str = ""
    detail: str = ""

    def __str__(self) -> str:  # noqa: D105
        parts = [self.message]
        if self.segment_id:
            parts.append(f"segment={self.segment_id}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.vram:
            parts.append(f"vram={self.vram}")
        if self.degraded_path:
            parts.append(f"degraded_path={self.degraded_path}")
        if self.profile:
            parts.append(f"profile={self.profile}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "segment_id": self.segment_id,
                "model": self.model, "vram": self.vram,
                "degraded_path": self.degraded_path,
                "profile": self.profile, "detail": self.detail}


def resolve_render_profile(*, profile: str, height: int,
                           allow_full_res: bool = False) -> dict[str, Any]:
    """Profile gate for a render request. Never silently downgrades.

    Returns ``{"height", "degraded", "reason", "profile"}``. On a
    small profile without ``allow_full_res`` the height is clamped to
    :data:`PROXY_HEIGHT` with ``degraded=True`` and an explicit reason.
    With ``allow_full_res=True`` the full height is kept and the
    explicit opt-in is recorded in the reason.
    """
    prof = (profile or "workstation").strip().lower()
    height = int(height)
    if prof in TERMUX_PROFILES and height > PROXY_HEIGHT:
        if not allow_full_res:
            return {"height": PROXY_HEIGHT, "degraded": True,
                    "profile": prof,
                    "reason": (
                        f"profile {prof!r}: full-res {height}p refused "
                        f"without allow_full_res=True — rendering "
                        f"{PROXY_HEIGHT}p proxy instead (explicit, "
                        "not silent). Pass allow_full_res=True to "
                        "render full-res on this device.")}
        return {"height": height, "degraded": False, "profile": prof,
                "reason": f"profile {prof!r}: full-res explicitly allowed"}
    return {"height": height, "degraded": False, "profile": prof,
            "reason": ""}


@dataclass
class Segment:
    """One resumable render unit."""
    id: str
    start: float
    end: float
    state: str = "pending"  # pending|claimed|done|failed
    worker: str = ""
    location: str = ""
    attempts: int = 0
    claimed_at: float = 0.0
    output: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "start": self.start, "end": self.end,
                "state": self.state, "worker": self.worker,
                "location": self.location, "attempts": self.attempts,
                "claimed_at": self.claimed_at, "output": self.output,
                "error": self.error}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            id=str(data["id"]), start=float(data["start"]),
            end=float(data["end"]),
            state=str(data.get("state", "pending")),
            worker=str(data.get("worker", "")),
            location=str(data.get("location", "")),
            attempts=int(data.get("attempts", 0)),
            claimed_at=float(data.get("claimed_at", 0.0)),
            output=str(data.get("output", "")),
            error=str(data.get("error", "")))


class RenderQueue:
    """Segment-level resumable render queue with work-stealing."""

    def __init__(self, segments: list[Segment] | None = None, *,
                 state_path: str | Path | None = None,
                 claim_ttl_s: float = 300.0,
                 max_attempts: int = 3) -> None:
        self.segments: list[Segment] = list(segments or [])
        self.state_path = Path(state_path) if state_path else None
        self.claim_ttl_s = float(claim_ttl_s)
        self.max_attempts = int(max_attempts)
        self._lock = threading.Lock()
        if self.state_path and self.state_path.exists():
            self.load()

    # -- persistence ----------------------------------------------------
    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"segments": [s.to_dict() for s in self.segments],
                   "saved_at": time.time()}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.segments = [Segment.from_dict(s)
                         for s in data.get("segments", [])]

    # -- claiming / work-stealing ---------------------------------------
    def claim(self, worker_id: str, *, location: str = "") -> Segment | None:
        """Claim the next pending segment — or steal a stale claim."""
        now = time.time()
        with self._lock:
            for seg in self.segments:
                if seg.state == "pending":
                    seg.state = "claimed"
                    seg.worker = worker_id
                    seg.location = location
                    seg.claimed_at = now
                    seg.attempts += 1
                    self.save()
                    return seg
            for seg in self.segments:
                if (seg.state == "claimed"
                        and now - seg.claimed_at > self.claim_ttl_s
                        and seg.attempts < self.max_attempts):
                    seg.worker = worker_id
                    seg.location = location
                    seg.claimed_at = now
                    seg.attempts += 1
                    self.save()
                    return seg
        return None

    def complete(self, segment_id: str, *, output: str = "") -> None:
        with self._lock:
            for seg in self.segments:
                if seg.id == segment_id:
                    seg.state = "done"
                    seg.output = output
                    seg.error = ""
                    break
        self.save()

    def fail(self, segment_id: str, error: str) -> None:
        with self._lock:
            for seg in self.segments:
                if seg.id == segment_id:
                    seg.error = str(error)[:500]
                    seg.state = ("failed" if seg.attempts >= self.max_attempts
                                 else "pending")
                    break
        self.save()

    @property
    def pending(self) -> list[Segment]:
        return [s for s in self.segments if s.state in ("pending", "claimed")]

    @property
    def done(self) -> list[Segment]:
        return [s for s in self.segments if s.state == "done"]

    @property
    def failed(self) -> list[Segment]:
        return [s for s in self.segments if s.state == "failed"]

    @property
    def progress(self) -> float:
        if not self.segments:
            return 1.0
        return len(self.done) / len(self.segments)

    # -- execution ------------------------------------------------------
    def run(self, segment_fn: Callable[[Segment], str], *,
            worker_id: str = "worker-0", location: str = "",
            profile: str = "workstation") -> Iterator[dict[str, Any]]:
        """Run pending segments, yielding progress events for chat.

        ``segment_fn`` renders one segment → output path (or raises).
        A raised exception is recorded honestly per segment; segments
        that exhaust ``max_attempts`` end ``failed`` and the run stops
        with a :class:`RenderFailure` naming the segment + profile.
        Already-done segments are skipped — that is the resume path.
        """
        while True:
            seg = self.claim(worker_id, location=location)
            if seg is None:
                break
            yield {"segment": seg.id, "state": "started",
                   "progress": round(self.progress, 3),
                   "worker": worker_id, "location": location}
            try:
                out = segment_fn(seg)
            except Exception as exc:  # noqa: BLE001 - per-segment honesty
                self.fail(seg.id, str(exc))
                yield {"segment": seg.id, "state": "error",
                       "error": str(exc)[:300],
                       "progress": round(self.progress, 3)}
                if seg.attempts >= self.max_attempts:
                    raise RenderFailure(
                        f"render failed on segment {seg.id}: {exc}",
                        segment_id=seg.id, profile=profile,
                        detail=str(exc)[:500]) from exc
                continue
            self.complete(seg.id, output=str(out))
            yield {"segment": seg.id, "state": "done", "output": str(out),
                   "progress": round(self.progress, 3)}
        yield {"state": "finished", "progress": round(self.progress, 3),
               "done": len(self.done), "failed": len(self.failed),
               "total": len(self.segments)}

    def run_all(self, segment_fn: Callable[[Segment], str], **kw: Any
                ) -> dict[str, Any]:
        """Collect :meth:`run` events → final summary dict."""
        events = list(self.run(segment_fn, **kw))
        return {"events": events, "done": len(self.done),
                "failed": len(self.failed), "total": len(self.segments),
                "outputs": [s.output for s in self.done],
                "progress": round(self.progress, 3)}


def segments_for_ranges(ranges: list[tuple[float, float]], *,
                        prefix: str = "seg") -> list[Segment]:
    """Ranges → segments (the usual way to build a queue)."""
    return [Segment(id=f"{prefix}-{i}", start=float(s), end=float(e))
            for i, (s, e) in enumerate(ranges)]
