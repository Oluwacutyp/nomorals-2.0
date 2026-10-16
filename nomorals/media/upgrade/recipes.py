"""Tier 1 — Branching history + recipes.

A :class:`Recipe` is everything needed to reproduce a render: the
brief, the spec dict, the model names, the seeds, the reference
fingerprints, and the op stack. A :class:`RecipeHistory` holds
branches: each branch is a named line of recipes; branching copies the
line, so experiments never destroy the shipped line. Reproducibility
invariant: same recipe dict -> same fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Recipe:
    brief: str
    spec: dict[str, Any] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    refs_fingerprint: str = ""
    ops: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"brief": self.brief, "spec": self.spec,
                "models": dict(self.models), "seeds": dict(self.seeds),
                "refs_fingerprint": self.refs_fingerprint,
                "ops": list(self.ops)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        return cls(brief=data.get("brief", ""), spec=dict(data.get("spec", {})),
                   models=dict(data.get("models", {})), seeds=dict(data.get("seeds", {})),
                   refs_fingerprint=data.get("refs_fingerprint", ""),
                   ops=list(data.get("ops", [])))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def recipe_from_spec(brief: str, spec: Any, *, models: dict[str, str] | None = None,
                     seeds: dict[str, int] | None = None,
                     refs_fingerprint: str = "",
                     ops: list[dict[str, Any]] | None = None) -> Recipe:
    """Build a recipe from an EditSpec (or a plain spec dict)."""
    spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    return Recipe(brief=brief, spec=spec_dict, models=dict(models or {}),
                  seeds=dict(seeds or {}), refs_fingerprint=refs_fingerprint,
                  ops=list(ops or []))


class RecipeHistory:
    """Branching history: named lines of recipes."""

    def __init__(self) -> None:
        self._branches: dict[str, list[Recipe]] = {"main": []}

    def commit(self, recipe: Recipe, branch: str = "main") -> None:
        self._branches.setdefault(branch, []).append(recipe)

    def branch(self, name: str, from_branch: str = "main") -> None:
        if name in self._branches:
            raise ValueError(f"branch {name!r} already exists")
        self._branches[name] = list(self._branches.get(from_branch, []))

    def head(self, branch: str = "main") -> Recipe | None:
        line = self._branches.get(branch, [])
        return line[-1] if line else None

    def branches(self) -> list[str]:
        return sorted(self._branches)

    def log(self, branch: str = "main") -> list[Recipe]:
        return list(self._branches.get(branch, []))
