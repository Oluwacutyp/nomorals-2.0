"""Replay a compiled browser recipe against the live page — no LLM.

The production replay path for navigator A5. A recipe is a
deterministic step chain compiled from a run that already worked;
replaying it spends no model time. Two rules from the owner's
standing orders govern every replay:

* **A stale path is never walked silently.** The live page must
  match the recipe's recorded layout (origin + element labels) at
  :data:`recipes.MATCH_THRESHOLD` before the first step, and every
  step's target must still be present on the page it runs on. A
  mismatch raises :class:`RecipeReplayError` naming the step, the
  expected target/label and what the page actually shows — and the
  recipe is marked stale in the store, so it is never retried blind.
  A missing slot value is the caller's fault, not the recipe's: it
  fails loudly too, but does NOT mark the recipe stale.
* **A recipe never smuggles a consequential step past the gate.**
  Every consequential step goes through the SAME approver the
  model-driven loop hands its decisions to (ActingApprover by
  default, the approval broker under
  ``partner.browser_require_approval``). Denied / timed out /
  no-channel stops the replay at that step, exactly like the loop.

The return shape mirrors :meth:`agent.AgentLoop.run` (goal, steps,
stopped_reason, final_url, trace, consequential_approvals) so a
replay job and a model-driven job read identically downstream.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..core.logging_setup import get_logger
from .agent import (
    Action,
    AgentTrace,
    Decision,
    Observation,
    Page,
)
from .recipes import (
    MATCH_THRESHOLD,
    SLOT_REF,
    Recipe,
    RecipeStep,
    layout_similarity,
)

_log = get_logger(__name__)

__all__ = ["RecipeReplayError", "replay_recipe"]

#: A wait step never sleeps longer than this, however the compiled
#: step reads — a stale wait is a stall, not a nap.
_MAX_WAIT_S = 30.0


class RecipeReplayError(RuntimeError):
    """The replay cannot proceed — the mismatch is the message.

    ``step_index`` is the recipe step that failed (``-1`` for the
    pre-flight layout check); ``stale`` says whether the recipe was
    marked stale for it (a page mismatch yes, a missing caller
    parameter no).
    """

    def __init__(self, message: str, *, step_index: int = -1,
                 stale: bool = False) -> None:
        super().__init__(message)
        self.step_index = step_index
        self.stale = stale


def _page_labels(page: Page) -> list[str]:
    return [str(el.get("label", "") or "")
            for el in (page.elements or []) if el.get("label")]


def _resolve_slots(step: RecipeStep,
                   slots: dict[str, str]) -> dict[str, str]:
    """Substitute ``{{name}}`` payload values from the caller's
    parameters. A missing value fails LOUDLY naming the slot —
    never a guessed or empty value typed into a live form."""
    resolved: dict[str, str] = {}
    for key, value in step.payload.items():
        match = SLOT_REF.match(str(value or ""))
        if match is None:
            resolved[key] = value
            continue
        name = match.group(1)
        if name not in slots:
            raise RecipeReplayError(
                f"recipe step needs parameter {name!r} "
                f"({step.kind} {step.label or step.target}) and no "
                "value was supplied — pass it as a replay parameter, "
                "it is never guessed",
                step_index=-1, stale=False)
        resolved[key] = str(slots[name])
    return resolved


def _target_present(step: RecipeStep, page: Page) -> bool:
    """The step's target (or label) still exists on the live page.
    ``goto``/``wait``/``assert`` have no element target and always
    pass this check."""
    if step.kind in ("goto", "wait", "assert", "back"):
        return True
    if not step.target and not step.label:
        return True
    for el in page.elements or []:
        if step.target and str(el.get("id", "")) == step.target:
            return True
        if step.label and str(el.get("label", "")) == step.label:
            return True
    return False


def replay_recipe(recipe: Recipe, *, driver: Any,
                  approve: Callable[[Decision, Observation], bool],
                  slots: dict[str, str] | None = None,
                  goal: str = "",
                  on_progress: Callable[[dict[str, Any]], None] | None = None,
                  cancel_requested: Callable[[], bool] | None = None,
                  store: Any = None,
                  match_threshold: float = MATCH_THRESHOLD
                  ) -> dict[str, Any]:
    """Replay ``recipe`` on the live page through ``driver``.

    ``approve`` is the run's approver (same seam the loop uses);
    ``slots`` supplies the recipe's parameters; ``store`` (a
    :class:`recipes.DurableRecipeStore`, optional) is where a
    mismatch marks the recipe stale and a success records trust.
    Raises :class:`RecipeReplayError` on a page mismatch or a step
    failure; an approval refusal instead stops the replay honestly
    (``stopped_reason`` set, trace kept) like the loop's own denial.
    """
    slots = dict(slots or {})
    steps = list(recipe.steps or [])
    if not steps:
        raise RecipeReplayError(
            f"recipe {recipe.name!r} has no steps to replay",
            step_index=-1, stale=False)
    first = steps[0]
    start_url = first.target if first.kind == "goto" and first.target \
        else recipe.url_pattern or recipe.site
    if not start_url:
        raise RecipeReplayError(
            f"recipe {recipe.name!r} names no start URL",
            step_index=-1, stale=False)
    page = driver.fetch(start_url)
    labels = _page_labels(page)
    score = layout_similarity(recipe, url=page.url, labels=labels)
    if score < match_threshold:
        missing = [label for label in recipe.element_labels
                   if label.lower() not in {x.lower() for x in labels}]
        reason = (f"recipe {recipe.name!r} v{recipe.version} no "
                  f"longer matches {page.url}: layout similarity "
                  f"{score} < {match_threshold}"
                  + (f"; missing elements: {', '.join(missing[:5])}"
                     if missing else "")
                  + (f"; page shows: {', '.join(labels[:5])}"
                     if labels else "; page shows no labelled elements"))
        if store is not None:
            try:
                store.mark_stale(recipe.name, reason)
            except Exception:  # noqa: BLE001 - stale-mark never gates
                _log.warning("recipe stale-mark failed for %s",
                             recipe.name, exc_info=True)
        raise RecipeReplayError(reason, step_index=-1, stale=True)

    trace: list[dict[str, Any]] = []
    stopped_reason = ""
    for i, step in enumerate(steps):
        if cancel_requested is not None:
            try:
                if cancel_requested():
                    stopped_reason = "cancelled: owner request"
                    break
            except Exception:  # noqa: BLE001 - a broken poll never
                # kills a healthy replay; the loop's own rule.
                cancel_requested = None
        if step.kind == "wait" and step.wait_ms > 0:
            time.sleep(min(step.wait_ms, int(_MAX_WAIT_S * 1000)) / 1000.0)
        try:
            payload = _resolve_slots(step, slots)
        except RecipeReplayError as exc:
            exc.step_index = i
            raise
        if not _target_present(step, page):
            shown = ", ".join(labels[:8]) if labels else "(none)"
            reason = (f"recipe {recipe.name!r} v{recipe.version} "
                      f"mismatch at step {i + 1} ({step.kind} "
                      f"{step.label or step.target}): the target is "
                      f"no longer on {page.url} — page shows: {shown}")
            if store is not None:
                try:
                    store.mark_stale(recipe.name, reason)
                except Exception:  # noqa: BLE001 - stale-mark never gates
                    _log.warning("recipe stale-mark failed for %s",
                                 recipe.name, exc_info=True)
            raise RecipeReplayError(reason, step_index=i, stale=True)
        action = Action(kind=step.kind, target=step.target,
                        label=step.label, payload=payload)
        if step.consequential:
            decision = Decision(
                action=action,
                reason=(f"recipe {recipe.name} v{recipe.version} "
                        f"step {i + 1}: {step.kind} "
                        f"{step.label or step.target}"),
                irreversible=False)
            obs = Observation(page=page, step=i + 1,
                              goal=goal or recipe.goal)
            if not approve(decision, obs):
                stopped_reason = "approval denied / no approver"
                break
        before_url, before_hash = page.url, page.snapshot_hash()
        started = time.monotonic()
        try:
            new_page = driver.act(action, page)
        except Exception as exc:  # noqa: BLE001 - the step failed
            reason = (f"recipe {recipe.name!r} v{recipe.version} "
                      f"failed at step {i + 1} ({step.kind} "
                      f"{step.label or step.target}): {exc}")
            if store is not None:
                try:
                    store.mark_stale(recipe.name, reason)
                except Exception:  # noqa: BLE001 - stale-mark never gates
                    _log.warning("recipe stale-mark failed for %s",
                                 recipe.name, exc_info=True)
            raise RecipeReplayError(reason, step_index=i,
                                    stale=True) from exc
        ms = (time.monotonic() - started) * 1000.0
        entry = AgentTrace(step=i + 1, before_url=before_url,
                           before_hash=before_hash,
                           action=action.to_dict(),
                           after_url=new_page.url,
                           after_hash=new_page.snapshot_hash(),
                           result={"ok": True, "verified": True},
                           ms=ms, verified=True)
        trace.append(entry.to_dict())
        page = new_page
        labels = _page_labels(page)
        if on_progress is not None:
            try:
                on_progress({"step": i + 1, "goal": goal or recipe.goal,
                             "url": page.url, "recipe": recipe.name})
            except Exception:  # noqa: BLE001 - progress never gates
                _log.debug("recipe replay progress note failed",
                           exc_info=True)
    approvals = list(getattr(approve, "outcomes", None) or [])
    out: dict[str, Any] = {
        "goal": goal or recipe.goal,
        "steps": len(trace),
        "stopped_reason": stopped_reason,
        "final_url": page.url,
        "trace": trace,
        "consequential_approvals": approvals,
        "recipe_name": recipe.name,
        "recipe_version": recipe.version,
        "replayed": True,
    }
    if cancel_requested is not None:
        out["cancel_poll_armed"] = True
    if not stopped_reason and store is not None:
        try:
            store.note_replay_success(recipe.name)
        except Exception:  # noqa: BLE001 - trust-note never gates
            _log.warning("recipe success note failed for %s",
                         recipe.name, exc_info=True)
    return out
