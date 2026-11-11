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
  site changed — recipe fails at step K -> LOUD mismatch naming the
  step, the recipe is marked stale and never retried blind (the old
  self-heal re-explore lives on only in the in-memory
  :class:`RecipeStore.replay` for callers that explicitly opt in).

Production wiring (navigator A5): a successful ``/browser run`` /
``drive`` job auto-compiles its trace into a
:class:`DurableRecipeStore` (runtime-created tables, the Batch B
pattern — no migrations edits); a later drive to the same site/goal
replays the recipe when it has earned trust, or is offered it.
Typed values a run typed (search text, form input) become named
SLOTS — the literal value is never stored, so a replay cannot leak
a secret the first run typed. Consequential steps carry their
approval outcome from the run that earned them, and a replay still
routes them through the run's approver — a recipe never smuggles a
consequential action past the gate.

Retrieval is by layout similarity: a recipe matches a new page when its
recorded (url pattern, element labels) overlap the live page above
threshold — never by blind name match.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

from ..core.logging_setup import get_logger
from .agent import CONSEQUENTIAL_KINDS

_log = get_logger(__name__)

__all__ = ["Recipe", "RecipeStep", "RecipeStore", "DurableRecipeStore",
           "compile_recipe", "layout_similarity", "recipe_name_for",
           "run_compilable", "to_skill_manifest", "TRUSTED_SUCCESSES",
           "SLOT_REF"]

#: Consecutive successful replays that earn a recipe TRUST — the
#: autonomy ladder's "act_silent is EARNED per recipe (N consecutive
#: successes), never assumed" (browser/agent.py). Below this a
#: matching recipe is OFFERED, never applied unasked.
TRUSTED_SUCCESSES = 2

#: Replay starts only when the live page matches the recorded layout
#: at least this well (same threshold as the in-memory store).
MATCH_THRESHOLD = 0.5

#: A payload value that is a slot reference, not a literal.
SLOT_REF = re.compile(r"^\{\{([A-Za-z0-9_]+)\}\}$")

#: Trace outcomes that bar a run from auto-compiling: the run did not
#: complete its goal (blocked / cancelled / denied / failed), so there
#: is nothing worth distilling. Anything else with a non-empty trace
#: of verified steps compiles.
_NO_COMPILE_PREFIXES = ("blocked:", "cancelled:", "approval denied",
                        "action failed:", "job failed:")


def run_compilable(trace: list[dict[str, Any]] | None,
                   stopped_reason: str) -> bool:
    """Whether a finished run's trace is worth distilling into a
    recipe: a non-empty trace, and a stopped reason that is not a
    block / cancel / denial / failure."""
    if not isinstance(trace, list) or not trace:
        return False
    stopped = str(stopped_reason or "")
    return not stopped.startswith(_NO_COMPILE_PREFIXES)


@dataclass
class RecipeStep:
    """One deterministic replay step.

    ``slot`` names the recipe parameter a ``fill``/``select`` value
    stands for: the payload carries ``{{slot}}`` and the literal the
    first run typed is never stored. ``consequential`` marks a step
    whose kind is in :data:`agent.CONSEQUENTIAL_KINDS`; ``approval``
    is the verdict the run that earned the step recorded for it
    (``{}`` when the step was compiled without one).
    """

    kind: str  # goto | click | fill | submit | select | check | wait | assert
    target: str = ""
    label: str = ""
    payload: dict[str, str] = field(default_factory=dict)
    wait_ms: int = 0
    assertion: dict[str, Any] = field(default_factory=dict)
    consequential: bool = False
    approval: dict[str, Any] = field(default_factory=dict)
    slot: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target,
                "label": self.label, "payload": dict(self.payload),
                "wait_ms": self.wait_ms, "assertion": dict(self.assertion),
                "consequential": bool(self.consequential),
                "approval": dict(self.approval), "slot": self.slot}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecipeStep":
        return cls(kind=str(raw.get("kind", "")),
                   target=str(raw.get("target", "")),
                   label=str(raw.get("label", "")),
                   payload=dict(raw.get("payload", {}) or {}),
                   wait_ms=int(raw.get("wait_ms", 0) or 0),
                   assertion=dict(raw.get("assertion", {}) or {}),
                   consequential=bool(raw.get("consequential", False)),
                   approval=dict(raw.get("approval", {}) or {}),
                   slot=str(raw.get("slot", "") or ""))


@dataclass
class Recipe:
    """A compiled deterministic replay for one task on one site.

    ``slots`` maps each parameter name to its human label and the
    step kind that consumes it (``{"search": {"label": "Search",
    "kind": "fill"}}``) — values are supplied at replay time, never
    stored. ``stale`` means a replay failed against the live page:
    the recipe is never offered or applied again until a fresh
    successful run re-compiles it. ``consecutive_successes`` is the
    trust counter the auto-apply gate reads.
    """

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
    slots: dict[str, dict[str, str]] = field(default_factory=dict)
    origin: str = ""
    stale: bool = False
    consecutive_successes: int = 0
    unmatched_approvals: list[dict[str, Any]] = field(default_factory=list)
    last_error: str = ""
    #: The run that earned this recipe was authenticated on
    #: the site — a replay needs the estate to hold a live
    #: session for the site, and fails loudly before the
    #: first step when it does not (Navigator A6).  The
    #: session itself is never stored in the recipe: the
    #: replay reuses the estate, it does not embed secrets.
    needs_session: bool = False

    @property
    def trusted(self) -> bool:
        """Earned auto-apply: not stale and enough consecutive
        successful replays (TRUSTED_SUCCESSES)."""
        return (not self.stale
                and self.consecutive_successes >= TRUSTED_SUCCESSES)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "site": self.site, "goal": self.goal,
                "steps": [s.to_dict() for s in self.steps],
                "url_pattern": self.url_pattern,
                "element_labels": list(self.element_labels),
                "successes": self.successes, "failures": self.failures,
                "created_at": self.created_at,
                "slots": {k: dict(v) for k, v in self.slots.items()},
                "origin": self.origin, "stale": bool(self.stale),
                "consecutive_successes": self.consecutive_successes,
                "unmatched_approvals": [dict(a) for a in
                                        self.unmatched_approvals],
                "last_error": self.last_error,
                "needs_session": bool(self.needs_session)}

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
                   created_at=float(raw.get("created_at", 0) or 0),
                   slots={str(k): dict(v)
                          for k, v in (raw.get("slots", {}) or {}).items()},
                   origin=str(raw.get("origin", "") or ""),
                   stale=bool(raw.get("stale", False)),
                   consecutive_successes=int(
                       raw.get("consecutive_successes", 0) or 0),
                   unmatched_approvals=[dict(a) for a in
                                        raw.get("unmatched_approvals", [])
                                        or []],
                   last_error=str(raw.get("last_error", "") or ""),
                   needs_session=bool(raw.get("needs_session", False)))

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


def _slug(text: str) -> str:
    """A name-safe slug: lowercase, runs of non-alphanumerics to one
    underscore, no leading/trailing underscores."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return slug or "step"


def recipe_name_for(url: str, goal: str) -> str:
    """The deterministic auto-compile name for a site+goal pair:
    ``<origin-slug>:<goal-slug>`` — so a later drive to the same
    site/goal finds the recipe its earlier run compiled."""
    try:
        origin = urlsplit(str(url or "")).netloc or str(url or "")
    except ValueError:
        origin = str(url or "")
    return _slug(origin) + ":" + _slug(goal or "browse")


def _slot_name(step: RecipeStep, taken: set[str]) -> str:
    """A unique slot name for a parameterised step, from its label
    then its target (``Search`` -> ``search``; a second ``search``
    becomes ``search_2``)."""
    base = _slug(step.label or step.target or "value")
    name, n = base, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    taken.add(name)
    return name


def _trace_to_steps(trace: list[dict[str, Any]]) -> list[RecipeStep]:
    """Trace entries -> steps. Only steps the run actually VERIFIED
    become replay steps: an entry whose result says ``ok: False``
    (or ``verified: False``) is a failed move the run recovered
    from, and replaying it would re-enact the failure. Entries
    without a result at all (older traces, hand-built traces) are
    kept — the filter only ever drops an explicit failure."""
    steps: list[RecipeStep] = []
    for entry in trace:
        result = entry.get("result", {}) or {}
        if result.get("ok") is False or entry.get("verified") is False:
            continue
        action = entry.get("action", {}) or {}
        assertion: dict[str, Any] = {}
        if entry.get("before_url"):
            assertion["before_url"] = entry.get("before_url", "")
        if entry.get("after_url"):
            assertion["after_url"] = entry.get("after_url", "")
        steps.append(RecipeStep(
            kind=str(action.get("kind", "")),
            target=str(action.get("target", "")),
            label=str(action.get("label", "")),
            payload=dict(action.get("payload", {}) or {}),
            assertion=assertion,
        ))
    return steps


def _parameterize(steps: list[RecipeStep]) -> dict[str, dict[str, str]]:
    """Turn typed values into slots IN PLACE: a ``fill``/``select``
    payload value becomes ``{{slot}}`` and the literal is dropped —
    a recipe must never store what the first run typed (it may be a
    password, a search, a name). Returns the slot catalogue."""
    slots: dict[str, dict[str, str]] = {}
    taken: set[str] = set()
    for step in steps:
        if step.kind not in ("fill", "select"):
            continue
        value = str(step.payload.get("value", "") or "")
        if not value:
            continue
        name = _slot_name(step, taken)
        step.payload["value"] = "{{" + name + "}}"
        step.slot = name
        slots[name] = {"label": step.label or step.target,
                       "kind": step.kind}
    return slots


def _mark_consequential(steps: list[RecipeStep],
                        approvals: list[dict[str, Any]]
                        ) -> list[dict[str, Any]]:
    """Mark the steps that mutate state and attach each one's
    approval outcome from the run that earned it. An approval
    matches a step on (kind, url): the approval's url is the page
    the step ran on (before_url or after_url). Approvals no step
    claims are returned, so the caller can keep them on the recipe
    instead of dropping them silently."""
    remaining = [dict(a) for a in approvals]
    for step in steps:
        if step.kind not in CONSEQUENTIAL_KINDS:
            continue
        step.consequential = True
        for approval in remaining:
            if str(approval.get("kind", "")) != step.kind:
                continue
            url = str(approval.get("url", "") or "")
            # The approval's url is the page the step ran ON: the
            # step's before_url (after_url is where it landed).
            pages = {str(step.assertion.get("before_url", "") or ""),
                     str(step.assertion.get("after_url", "") or "")}
            if not url or url in pages:
                step.approval = dict(approval)
                remaining.remove(approval)
                break
    return remaining


#: A cookie value shorter than this cannot be told apart
#: from an ordinary typed word (``"en"``, ``"1"``, ``"true"``),
#: so the secret scan below only covers values at least this
#: long — a session token is.  Fill/select values are stripped
#: by :func:`_parameterize` regardless of length; the scan
#: covers the paths parameterization does not reach (goto URLs,
#: other payloads, assertions).
_SECRET_MIN_LEN = 16


def _recipe_secret_hit(recipe: Recipe, secrets: set[str]
                       ) -> tuple[int, str] | None:
    """First stored literal that carries a cookie value, as
    (step index, step kind) — or None.  Slot references
    (``{{name}}``) are not literals and never match.  The
    VALUE is never returned: a hit is named by step, never
    by secret."""
    long_secrets = [s for s in secrets if len(s) >= _SECRET_MIN_LEN]
    if not long_secrets:
        return None

    def _hit(text: str) -> bool:
        if not text or SLOT_REF.match(text):
            return False
        return any(secret in text for secret in long_secrets)

    for i, step in enumerate(recipe.steps or []):
        if _hit(step.target) or _hit(step.label):
            return (i, step.kind)
        for value in (step.payload or {}).values():
            if _hit(str(value or "")):
                return (i, step.kind)
        for value in (step.assertion or {}).values():
            if isinstance(value, str) and _hit(value):
                return (i, step.kind)
    for text in (recipe.goal, recipe.url_pattern, recipe.site,
                 recipe.origin, *recipe.element_labels):
        if _hit(str(text or "")):
            return (-1, "recipe")
    return None


def compile_recipe(*, name: str, goal: str, trace: list[dict[str, Any]],
                   site: str = "", url_pattern: str = "",
                   element_labels: list[str] | None = None,
                   start_url: str = "",
                   approvals: list[dict[str, Any]] | None = None,
                   needs_session: bool = False,
                   cookie_estate: Any = None,
                   ) -> Recipe:
    """Compile an exploratory agent trace into a deterministic recipe.

    ``start_url`` is the URL the run began on: the loop fetches it
    before deciding, so the trace carries no ``goto`` for it — the
    compile synthesises step 0 (a replay must start by navigating
    there). ``approvals`` is the run's consequential-approval list
    (``job.approvals`` / the runner's ``consequential_approvals``);
    each entry lands on the step it approved. ``needs_session``
    declares the run was authenticated on the site (set by the
    caller from the estate — a replay then fails loudly when the
    session is gone). ``cookie_estate`` is the unified cookie
    estate: when given, the compiled recipe is scanned against
    its live cookie values and the compile FAILS (ValueError)
    rather than store a recipe that carries a secret — a recipe
    never embeds session material, it reuses the estate at
    replay time. Raises ValueError on an empty trace or a trace
    with no verified step.
    """
    if not trace:
        raise ValueError("cannot compile a recipe from an empty trace")
    steps = _trace_to_steps(trace)
    if not steps:
        raise ValueError("cannot compile a recipe: the trace has no "
                         "verified step to replay")
    origin = ""
    first_url = str(start_url or "") or str(
        trace[0].get("before_url", "") or "")
    try:
        origin = urlsplit(first_url).netloc if first_url else ""
    except ValueError:
        origin = ""
    if steps[0].kind != "goto" and first_url:
        steps.insert(0, RecipeStep(
            kind="goto", target=first_url, label="",
            assertion={"after_url": first_url}))
    slots = _parameterize(steps)
    unmatched = _mark_consequential(steps, list(approvals or []))
    labels = list(element_labels or [])
    for step in steps:
        if step.label and step.label not in labels:
            labels.append(step.label)
    recipe = Recipe(name=name, goal=goal, site=site or origin,
                    steps=steps, url_pattern=url_pattern or site or origin,
                    element_labels=labels, slots=slots, origin=origin,
                    unmatched_approvals=unmatched,
                    needs_session=bool(needs_session))
    if cookie_estate is not None:
        try:
            secrets = cookie_estate.secret_values()
        except Exception:  # noqa: BLE001 - an unreadable
            secrets = set()  # estate never blocks a compile
        hit = _recipe_secret_hit(recipe, secrets) if secrets else None
        if hit is not None:
            step_index, kind = hit
            raise ValueError(
                f"cannot compile recipe {name!r}: step {step_index + 1} "
                f"({kind}) carries a live session value — a recipe "
                "never stores cookie material, it reuses the cookie "
                "estate at replay time")
    return recipe


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


#: Durable recipe table (Batch B pattern: DDL in code, runtime-created
#: over the shared ``Database`` — migrations untouched). One row per
#: recipe name; ``data`` is the full recipe JSON, the indexed columns
#: are what lookup filters on. A re-compile of the same name bumps
#: the version and clears ``stale`` — the fresh run proved the site
#: again — while the counters keep the recipe's history.
_RECIPE_DDL = """
CREATE TABLE IF NOT EXISTS browser_recipes (
  name TEXT PRIMARY KEY,
  site TEXT NOT NULL DEFAULT '',
  goal TEXT NOT NULL DEFAULT '',
  origin TEXT NOT NULL DEFAULT '',
  url_pattern TEXT NOT NULL DEFAULT '',
  version TEXT NOT NULL DEFAULT '1.0.0',
  stale INTEGER NOT NULL DEFAULT 0,
  successes INTEGER NOT NULL DEFAULT 0,
  failures INTEGER NOT NULL DEFAULT 0,
  consecutive_successes INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NOT NULL DEFAULT '',
  data TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_browser_recipes_site_goal
  ON browser_recipes(site, goal);
"""


def _norm_goal(goal: str) -> str:
    return " ".join(str(goal or "").split()).casefold()


class DurableRecipeStore:
    """Recipes that survive the process: save / list / get / delete /
    find, over the shared home database.

    ``find_for`` is the drive-time lookup: same site AND same goal
    (normalised) and a layout score at or above threshold — stale
    recipes never match, so a failed recipe is never retried blind.
    The store never raises for a caller that cannot afford it: the
    auto-compile hook treats a store failure as evidence lost, not
    as a failed run (see jobs.py).
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_RECIPE_DDL)

    def save(self, recipe: Recipe) -> Recipe:
        """Upsert by name. A re-compile bumps the version, clears
        ``stale`` and resets the trust counter — the new run proved
        the site again — and keeps the lifetime counters."""
        existing = self.get(recipe.name)
        if existing is not None:
            recipe.version = existing.bump_version()
            recipe.successes = existing.successes
            recipe.failures = existing.failures
            recipe.stale = False
            recipe.consecutive_successes = 0
            recipe.last_error = ""
            recipe.created_at = existing.created_at
        now = time.time()
        self.db.execute(
            "INSERT INTO browser_recipes (name, site, goal, origin,"
            " url_pattern, version, stale, successes, failures,"
            " consecutive_successes, last_error, data, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
            " ?, ?, ?) ON CONFLICT(name) DO UPDATE SET site ="
            " excluded.site, goal = excluded.goal, origin ="
            " excluded.origin, url_pattern = excluded.url_pattern,"
            " version = excluded.version, stale = excluded.stale,"
            " successes = excluded.successes, failures ="
            " excluded.failures, consecutive_successes ="
            " excluded.consecutive_successes, last_error ="
            " excluded.last_error, data = excluded.data, updated_at"
            " = excluded.updated_at",
            (recipe.name, recipe.site, recipe.goal, recipe.origin,
             recipe.url_pattern, recipe.version, 1 if recipe.stale
             else 0, recipe.successes, recipe.failures,
             recipe.consecutive_successes, recipe.last_error,
             json.dumps(recipe.to_dict(), sort_keys=True),
             recipe.created_at, now))
        return recipe

    def get(self, name: str) -> Recipe | None:
        row = self.db.query_one(
            "SELECT data FROM browser_recipes WHERE name = ?",
            (str(name or ""),))
        if row is None:
            return None
        try:
            return Recipe.from_dict(json.loads(row["data"] or "{}"))
        except (ValueError, TypeError, AttributeError):
            return None

    def delete(self, name: str) -> bool:
        cur = self.db.execute(
            "DELETE FROM browser_recipes WHERE name = ?",
            (str(name or ""),))
        return bool(getattr(cur, "rowcount", 0))

    def list_all(self) -> list[Recipe]:
        rows = self.db.query(
            "SELECT data FROM browser_recipes ORDER BY updated_at DESC")
        out: list[Recipe] = []
        for row in rows:
            try:
                out.append(Recipe.from_dict(json.loads(row["data"] or "{}")))
            except (ValueError, TypeError, AttributeError):
                continue
        return out

    def find_for(self, *, url: str, labels: list[str],
                 goal: str = "", threshold: float = MATCH_THRESHOLD
                 ) -> Recipe | None:
        """Best matching recipe for a drive: same site (origin or
        url pattern) and same goal, layout score >= threshold, never
        stale. ``labels`` may be empty before the page is fetched —
        the replay itself re-checks the live labels and fails loudly
        when they no longer match."""
        best: Recipe | None = None
        best_score = 0.0
        for recipe in self.list_all():
            if recipe.stale:
                continue
            if goal and _norm_goal(recipe.goal) != _norm_goal(goal):
                continue
            score = layout_similarity(recipe, url=url, labels=labels)
            if score >= threshold and score > best_score:
                best, best_score = recipe, score
        return best

    def mark_stale(self, name: str, reason: str) -> bool:
        """A replay failed against the live page: the recipe is
        stale (failures +1, trust reset) and carries the reason, so
        the next drive sees WHY it stopped matching instead of a
        bare flag. Returns False when no such recipe exists."""
        recipe = self.get(name)
        if recipe is None:
            return False
        recipe.stale = True
        recipe.failures += 1
        recipe.consecutive_successes = 0
        recipe.last_error = str(reason or "")[:500]
        self.db.execute(
            "UPDATE browser_recipes SET stale = 1, failures = ?,"
            " consecutive_successes = 0, last_error = ?,"
            " data = ?, updated_at = ? WHERE name = ?",
            (recipe.failures, recipe.last_error,
             json.dumps(recipe.to_dict(), sort_keys=True),
             time.time(), recipe.name))
        return True

    def note_replay_success(self, name: str) -> bool:
        """A replay completed: lifetime successes +1, trust +1.
        Returns False when no such recipe exists."""
        recipe = self.get(name)
        if recipe is None:
            return False
        recipe.successes += 1
        recipe.consecutive_successes += 1
        self.db.execute(
            "UPDATE browser_recipes SET successes = ?,"
            " consecutive_successes = ?, data = ?, updated_at = ?"
            " WHERE name = ?",
            (recipe.successes, recipe.consecutive_successes,
             json.dumps(recipe.to_dict(), sort_keys=True),
             time.time(), recipe.name))
        return True
