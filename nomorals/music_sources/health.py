"""Health prober for the music registry.

Probing is honest by construction:

* A source is probed only when it is *due* — ``health_due`` from the
  registry, driven by volatility (institutional 720h / commercial
  168h / indie 24h / platform_api 6h).
* The actual ping is an injected ``ping_fn(source) -> bool``. With no
  pinger wired, the status is ``unknown`` with the reason "no pinger
  wired" — never a fabricated up/down.
* A pinger raising is recorded as ``down`` with the exception type,
  never raised to the caller.

Results are kept in-memory with timestamps so ``probe_all`` can skip
sources that are not due yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .registry import MusicSource, all_sources, health_due

__all__ = ["HealthStatus", "HealthProber"]


@dataclass
class HealthStatus:
    source_id: str
    status: str = "unchecked"  # up | down | unknown | unchecked
    checked_ts: Optional[float] = None
    detail: str = ""
    next_due_hours: Optional[int] = None


class HealthProber:
    """Probe registry sources on their volatility-driven cadence."""

    def __init__(
        self,
        ping_fn: Optional[Callable[[MusicSource], bool]] = None,
    ) -> None:
        self.ping_fn = ping_fn
        self.results: Dict[str, HealthStatus] = {}

    def probe(
        self, source: MusicSource, *, now: Optional[float] = None,
        force: bool = False,
    ) -> HealthStatus:
        """Probe one source (skipped when not due, unless ``force``)."""
        at = now if now is not None else time.time()
        previous = self.results.get(source.id)
        if not force and previous is not None and previous.checked_ts is not None:
            if not health_due(source, previous.checked_ts, at):
                return previous
        if self.ping_fn is None:
            status = HealthStatus(
                source_id=source.id, status="unknown", checked_ts=at,
                detail="no pinger wired — status unknown, never fabricated",
                next_due_hours=source.health_check_hours,
            )
        else:
            try:
                up = bool(self.ping_fn(source))
            except Exception as exc:  # noqa: BLE001 — ping failure is data
                status = HealthStatus(
                    source_id=source.id, status="down", checked_ts=at,
                    detail=f"ping raised {type(exc).__name__}: {exc}",
                    next_due_hours=source.health_check_hours,
                )
                self.results[source.id] = status
                return status
            status = HealthStatus(
                source_id=source.id, status="up" if up else "down",
                checked_ts=at,
                detail="ping ok" if up else "ping returned false",
                next_due_hours=source.health_check_hours,
            )
        self.results[source.id] = status
        return status

    def probe_all(
        self, sources: Optional[List[MusicSource]] = None, *,
        now: Optional[float] = None, force: bool = False,
    ) -> List[HealthStatus]:
        """Probe every due source (all registry rows by default)."""
        at = now if now is not None else time.time()
        out: List[HealthStatus] = []
        for source in (sources if sources is not None else all_sources()):
            previous = self.results.get(source.id)
            if not force and previous is not None and previous.checked_ts is not None:
                if not health_due(source, previous.checked_ts, at):
                    continue
            out.append(self.probe(source, now=at, force=True))
        return out

    def status(self, source_id: str) -> HealthStatus:
        """Last known status, or ``unchecked`` when never probed."""
        found = self.results.get(source_id)
        if found is not None:
            return found
        return HealthStatus(source_id=source_id, status="unchecked",
                            detail="never probed")
