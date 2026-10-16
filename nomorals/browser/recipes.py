"""Browser recipes: run once with the LLM, compile to a deterministic replay.

Recipes ARE skills — a named, versioned chain of steps with declared
inputs, selectors, waits, and assertions. The shape matches
``skills/manifest.py`` (plain dicts), so a recipe converts to a skill
manifest with :func:`to_skill_manifest` and benefits from the skills
registry / versioning / bench / repair machinery without this module
importing it.

Flow:
  run 1 — LLM exploratory: the agent loop runs, its trace is recorded.
  compile — trace -> RECIPE: deterministic steps (goto/click/fill/submit/
  wait/assert), selectors grounded on the live page, version 1.0.0.
  run 2+ — REPLAY, no LLM: ~100x faster, same assertions.
  site changed — recipe fails at step K -> LLM re-explores ONLY step K
  -> patch that step -> version-bump (never silently edit the rest).

Retrieval is by layout similarity: a recipe matches a new page when its
recorded (url pattern, element labels) overlap the live page above
threshold — never by blind name match.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["Recipe", "RecipeStep", "RecipeStore", "compile_recipe",
           "layout_similarity", "to_skill_manifest"]


@dataclass
class RecipeStep:
    """One deterministic replay step."""

    kind: str  # goto | click | fill | submit | select | check | wait | assert
    target: str = ""
    label: str = ""
    payload: dict[str, str] = field(default_factory=dict)
    wait_ms: int = 0
    assertion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target,
                "label": self.label, "payload": dict(self.payload),
                "wait_ms": self.wait_ms, "assertion": dict(self.assertion)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecipeStep":
        return cls(kind=str(raw.get("kind", "")),
                   target=str(raw.get("target", "")),
                   label=str(raw.get("label", "")),
                   payload=dict(raw.get("payload", {}) or {}),
                   wait_ms=int(raw.get("wait_ms", 0) or 0),
                   assertion=dict(raw.get("assertion", {}) or {}))


@dataclass
class Recipe:
    """A compiled deterministic replay for one task on one site."""

    name: str
    version: str = "1.0.0"
    site: str = ""
    goal: str = ""
    steps: list[RecipeStep] = field(default_factory=list)
    url_pattern: str = ""
    element_labels: list[str] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "site": self.site, "goal": self.goal,
                "steps": [s.to_dict() for s in self.steps],
                "url_pattern": self.url_pattern,
                "element_labels": list(self.element_labels),
                "successes": self.successes, "failures": self.failures,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Recipe":
        return cls(name=str(raw.get("name", "")),
                   version=str(raw.get("version", "1.0.0")),
                   site=str(raw.get("site", "")),
                   goal=str(raw.get("goal", "")),
                   steps=[RecipeStep.from_dict(s)
                          for s in raw.get("steps", []) or []],
                   url_pattern=str(raw.get("url_pattern", "")),
                   element_labels=list(raw.get("element_labels", []) or []),
                   successes=int(raw.get("successes", 0) or 0),
                   failures=int(raw.get("failures", 0) or 0),
                   created_at=float(raw.get("created_at", 0) or 0))

    def bump_version(self) -> str:
        """Patch-level bump on self-heal: 1.0.0 -> 1.0.1."""
        parts = (self.version or "1.0.0").split(".")
        try:
            major, minor, patch = (int(parts[0]), int(parts[1]),
                                   int(parts[2]) if len(parts) > 2 else 0)
        except ValueError:
            major, minor, patch = 1, 0, 0
        self.version = f"{major}.{minor}.{patch + 1}"
        return self.version


def _trace_to_steps(trace: list[dict[str, Any]]) -> list[RecipeStep]:
    steps: list[RecipeStep] = []
    for entry in trace:
        action = entry.get("action", {}) or {}
        steps.append(RecipeStep(
            kind=str(action.get("kind", "")),
            target=str(action.get("target", "")),
            label=str(action.get("label", "")),
            payload=dict(action.get("payload", {}) or {}),
            assertion={"after_url": entry.get("after_url", "")} if entry.get("after_url") else {},
        ))
    return steps


def compile_recipe(*, name: str, goal: str, trace: list[dict[str, Any]],
                   site: str = "", url_pattern: str = "",
                   element_labels: list[str] | None = None) -> Recipe:
    """Compile an exploratory agent trace into a deterministic recipe."""
    if not trace:
        raise ValueError("cannot compile a recipe from an empty trace")
    return Recipe(name=name, goal=goal, site=site,
                  steps=_trace_to_steps(trace),
                  url_pattern=url_pattern or site,
                  element_labels=list(element_labels or []))


def layout_similarity(recipe: Recipe, *, url: str,
                      labels: list[str]) -> float:
    """0..1: how well a recipe's recorded layout matches a live page."""
    score = 0.0
    if recipe.url_pattern and recipe.url_pattern in url:
        score += 0.5
    elif recipe.site and recipe.site in url:
        score += 0.5
    if recipe.element_labels and labels:
        wanted = {str(x).lower() for x in recipe.element_labels}
        have = {str(x).lower() for x in labels}
        score += 0.5 * (len(wanted & have) / max(1, len(wanted)))
    return round(min(1.0, score), 3)


def to_skill_manifest(recipe: Recipe) -> dict[str, Any]:
    """Recipe -> skills/manifest.py-shaped dict (plain data, no import)."""
    return {
        "name": f"browser.{recipe.name}",
        "version": recipe.version,
        "description": recipe.goal,
        "owner": "devon",
        "tools": ["browser." + (s.kind or "goto") for s in recipe.steps],
        "input_schema": {"start_url": "str"},
        "output_schema": {"final_url": "str", "steps": "int"},
        "wiring": [s.to_dict() for s in recipe.steps],
    }


class RecipeStore:
    """In-memory recipe registry with replay + self-heal."""

    def __init__(self) -> None:
        self._recipes: dict[str, Recipe] = {}

    def add(self, recipe: Recipe) -> None:
        self._recipes[recipe.name] = recipe

    def get(self, name: str) -> Recipe | None:
        return self._recipes.get(name)

    def find_for(self, *, url: str, labels: list[str],
                 threshold: float = 0.5) -> Recipe | None:
        best: Recipe | None = None
        best_score = 0.0
        for recipe in self._recipes.values():
            score = layout_similarity(recipe, url=url, labels=labels)
            if score >= threshold and score > best_score:
                best, best_score = recipe, score
        return best

    def replay(self, recipe: Recipe, *,
               execute: Callable[[RecipeStep, int], dict[str, Any]],
               reexplore_step: Callable[[RecipeStep, int, str], RecipeStep] | None = None) -> dict[str, Any]:
        """Replay without an LLM. On step-K failure, re-explore ONLY step K.

        Returns {ok, failed_step, version, steps_run}. A self-healed step
        patches that step only and bumps the version — the rest is untouched.
        """
        for i, step in enumerate(list(recipe.steps)):
            result = execute(step, i)
            if result.get("ok"):
                continue
            error = str(result.get("error", "step failed"))
            if reexplore_step is None:
                recipe.failures += 1
                return {"ok": False, "failed_step": i, "error": error,
                        "version": recipe.version, "steps_run": i}
            patched = reexplore_step(step, i, error)
            recipe.steps[i] = patched
            recipe.bump_version()
            result2 = execute(patched, i)
            if not result2.get("ok"):
                recipe.failures += 1
                return {"ok": False, "failed_step": i,
                        "error": str(result2.get("error", error)),
                        "version": recipe.version, "steps_run": i}
        recipe.successes += 1
        return {"ok": True, "failed_step": None, "version": recipe.version,
                "steps_run": len(recipe.steps)}

    @staticmethod
    def fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
