"""Tier 2 moat — Learned model routing (same shape as LLM health-scored
fallback): per-task model table with health scores, latency, and cost.
Routing picks the best healthy model for the task; on failure the
caller falls back to the next healthy entry. Scores are updated from
observed outcomes (success/failure, quality) — never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TASKS = ("photoreal", "anime", "text", "video", "upscale", "inpaint", "audio")


@dataclass
class ModelRoute:
    task: str
    model: str
    health: float = 1.0  # 0..1, decays on failure, recovers on success
    latency_s: float = 0.0
    cost_per_call: float = 0.0
    quality: float = 0.5  # observed mean quality 0..1

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown routing task {self.task!r}")
        if not 0.0 <= self.health <= 1.0:
            raise ValueError("health must be 0..1")

    def observe(self, *, success: bool, quality: float | None = None) -> None:
        if success:
            self.health = min(1.0, self.health + 0.05)
            if quality is not None:
                self.quality = 0.8 * self.quality + 0.2 * quality
        else:
            self.health = max(0.0, self.health - 0.25)

    @property
    def score(self) -> float:
        """Health-weighted routing score; unhealthy models sink."""
        return self.health * (0.7 * self.quality + 0.3 * (1.0 - min(1.0, self.latency_s / 60.0)))

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "model": self.model, "health": self.health,
                "latency_s": self.latency_s, "cost_per_call": self.cost_per_call,
                "quality": self.quality}


class RouteTable:
    """Per-task ordered routes; best = highest score among healthy."""

    def __init__(self, routes: list[ModelRoute] | None = None) -> None:
        self._routes: list[ModelRoute] = list(routes or [])

    def add(self, route: ModelRoute) -> None:
        self._routes.append(route)

    def route_for_task(self, task: str, *, min_health: float = 0.3) -> ModelRoute | None:
        cands = [r for r in self._routes if r.task == task and r.health >= min_health]
        if not cands:
            return None
        return max(cands, key=lambda r: r.score)

    def fallback_chain(self, task: str) -> list[ModelRoute]:
        return sorted((r for r in self._routes if r.task == task),
                      key=lambda r: r.score, reverse=True)


def route_for_task(table: RouteTable, task: str) -> ModelRoute | None:
    return table.route_for_task(task)
