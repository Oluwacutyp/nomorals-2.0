"""Models-into-classes ladder (Phase 1).

The doctrine (owner, 2026-10-25): classes STAY, models wire IN, behavior
becomes DYNAMIC. The pieces already existed — :class:`Brain` fronts the
providers, ``SuggestFn`` is the ``(prompt) -> text`` bridge, every game
and character class already takes ``suggest``-or-None and falls back to
a full deterministic code rung. What did NOT exist was the DECISION: one
registry mapping JOB -> rungs, selected per device by the profile
runtime, with an honest unavailable path.

This module is that decision layer (L3, so characters/games/media/agents
may import it; it imports only core + llm):

* :func:`resolve` — per job, per profile: ``"model"`` when the model rung
  can actually run here, ``"code"`` otherwise, with a ``reason`` naming
  the deciding factor. The model rung is the brain when the job needs
  language (the brain fronts local + remote providers); for jobs whose
  model is data (the DJ's learned taste) the caller supplies
  ``model_available`` and owns the model object — the ladder decides the
  rung, the caller owns the rung.
* :func:`recommend_local_model` — which catalog entry fits this device
  (1.5B on Termux, 3.8B on a PC, 8B on a workstation), from
  :data:`MODEL_CATALOG`, never hardcoded.
* :func:`local_model_available` — are those weights really on disk
  (``find_gguf`` + ``gguf_check``) and does the box have the RAM? An
  absent local model is reported, never assumed.
* Per-profile model timeouts: a model call on Termux gets 20s, a PC 30s,
  a workstation 60s — the suggest fn carries the profile's budget.

No single model is ever "her brain": the brain directs, the ladder
selects, the specialists do the how. A rung that cannot run reports
``"code"`` with its reason — never a fake model result.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.profile import (
    EnvironmentProfile,
    detect_profile,
    memory_mb,
    resolve_profile,
)
from .base import Message

__all__ = [
    "JobSpec",
    "Resolution",
    "JOB_NAMES",
    "MODEL_TIMEOUT_S",
    "list_jobs",
    "register_job",
    "resolve",
    "recommend_local_model",
    "local_model_available",
]

_log = get_logger(__name__)

#: Model-call budget per profile kind (seconds). Constrained devices get
#: a shorter leash so a slow provider can never stall a game turn.
MODEL_TIMEOUT_S: dict[str, float] = {
    "termux": 20.0,
    "mobile": 20.0,
    "embedded": 15.0,
    "pc": 30.0,
    "vps": 30.0,
    "workstation": 60.0,
}

#: Local-model recommendation per profile kind: (catalog repo_id,
#: max params). Termux runs the 1.5B mobile Dolphin; a PC the 3.8B Phi;
#: a workstation the 8B Dolphin. All three are MODEL_CATALOG entries.
_LOCAL_MODEL_BY_KIND: dict[str, tuple[str, int]] = {
    "termux": ("cognitivecomputations/dolphin-2.9.3-qwen2.5-1.5b",
               2_000_000_000),
    "mobile": ("cognitivecomputations/dolphin-2.9.3-qwen2.5-1.5b",
               2_000_000_000),
    "embedded": ("cognitivecomputations/dolphin-2.9.3-qwen2.5-1.5b",
                 2_000_000_000),
    "pc": ("microsoft/Phi-3.5-mini-instruct", 8_000_000_000),
    "vps": ("microsoft/Phi-3.5-mini-instruct", 8_000_000_000),
    "workstation": ("cognitivecomputations/dolphin-2.9.1-llama-3-8b",
                    16_000_000_000),
}


@dataclass(frozen=True)
class JobSpec:
    """One job on the ladder: its two full rungs, in words."""

    name: str
    description: str
    model_rung: str  # what the model does for this job
    code_rung: str  # what the deterministic code does (always works)
    model_needs_brain: bool = True  # False when the model is data-owned


@dataclass
class Resolution:
    """What :func:`resolve` decided, and why."""

    job: str
    rung: str  # "model" | "code"
    reason: str
    suggest: Callable[[str], str] | None = None
    timeout_s: float = 30.0
    local_model: str = ""  # recommended local weights for this device
    local_available: bool = False


_JOBS: dict[str, JobSpec] = {}


def register_job(spec: JobSpec) -> JobSpec:
    """Register (or replace) a job on the ladder. Returns the spec."""
    _JOBS[spec.name] = spec
    return spec


def list_jobs() -> list[str]:
    """Registered job names, in registration order."""
    return list(_JOBS)


# ── the four Phase-1 jobs ─────────────────────────────────────────────────

register_job(JobSpec(
    name="character_dialogue",
    description="NPCs talk dynamically, in their own voice, from persona "
                "+ context + memory — not canned lines.",
    model_rung="Brain chat via SuggestFn (GameMaster.npc_speak / "
               "Character.speak)",
    code_rung="Template forge: mood-band lines parameterized by name "
              "and situation (GameMaster._NPC_LINES, Character fallback)",
))

register_job(JobSpec(
    name="character_tactics",
    description="NPCs choose battle actions dynamically — the model may "
                "override the math, but never invents an action.",
    model_rung="Brain picks one legal action (attack/focus/fury/defend/"
               "potion) from the live state; validated, else math",
    code_rung="GameMind.combat_move math: potion/focus/defend/fury "
              "thresholds by hunter rank (deterministic)",
))

register_job(JobSpec(
    name="dj_sound_selection",
    description="The DJ model: which sound comes next, learned from the "
                "sounds the DJ actually plays and the owner keeps/skips.",
    model_rung="LearnedTaste — online logistic ranker over the DJ "
               "transition features, trained on log_feedback rows",
    code_rung="HeuristicTaste — harmonic + tempo + energy-flow "
              "deterministic scoring (the content-based ranker)",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="arena_narration",
    description="Dynamic arena battle narration illustrating the actual "
                "battle, in the table's DM mood — like a movie.",
    model_rung="Brain narrates the drained DM feed (GameMaster.narrate / "
               "announce) in the DM's mood",
    code_rung="GameMaster template bank, mood-parameterized, no immediate "
              "repeats",
))

JOB_NAMES: tuple[str, ...] = tuple(_JOBS)


def _coerce_profile(profile: Any) -> EnvironmentProfile:
    if profile is None:
        return detect_profile()
    if isinstance(profile, str):
        return resolve_profile(profile)
    return profile


def _model_timeout(profile: EnvironmentProfile) -> float:
    return MODEL_TIMEOUT_S.get((profile.kind or "").lower(), 30.0)


def _brain_suggest(brain: Any, timeout_s: float) -> Callable[[str], str]:
    """Wrap a Brain (or anything with .chat) as a SuggestFn.

    Never raises: a failed model call returns "" and the caller falls
    through to its code rung — exactly the contract GameMind.ask and
    GameMaster._ask already rely on.
    """

    def _call(prompt: str) -> str:
        try:
            resp = brain.chat(
                [Message(role="user", content=prompt)],
                task_kind="chat", timeout_s=timeout_s)
            return getattr(resp, "text", "") or ""
        except Exception:  # noqa: BLE001 - the model must never kill a job
            _log.debug("ladder model call failed", exc_info=True)
            return ""

    return _call


def recommend_local_model(profile: Any = None):
    """The catalog entry that fits this device, or None.

    Sizes come from :data:`MODEL_CATALOG` — the recommendation is a
    lookup, never a hardcoded guess. Falls back to the smallest catalog
    entry when the preferred one is absent (defensive, never raises).
    """
    from .registry import MODEL_CATALOG

    prof = _coerce_profile(profile)
    kind = (prof.kind or "").lower()
    repo_id, _cap = _LOCAL_MODEL_BY_KIND.get(
        kind, _LOCAL_MODEL_BY_KIND["pc"])
    by_id = {entry.repo_id: entry for entry in MODEL_CATALOG}
    entry = by_id.get(repo_id)
    if entry is not None:
        return entry
    smallest = min(MODEL_CATALOG, key=lambda e: e.params)
    return smallest


def local_model_available(repo_id: str,
                          cache_dir: str | None = None
                          ) -> tuple[bool, str]:
    """Are ``repo_id``'s weights really on disk, with RAM to run them?

    Checks, in order: a GGUF file resolvable by name (``find_gguf``),
    its integrity (``gguf_check``: magic + non-trivial size), and a RAM
    fit against the catalog size hint. Returns ``(ok, reason)`` — an
    absent local model is reported, never assumed.
    """
    from .local_server import find_gguf, gguf_check
    from .registry import MODEL_CATALOG

    repo_id = (repo_id or "").strip()
    if not repo_id:
        return False, "no model id given"
    by_id = {entry.repo_id: entry for entry in MODEL_CATALOG}
    entry = by_id.get(repo_id)
    dirs: list[Path] = []
    if cache_dir:
        dirs.append(Path(os.path.expanduser(cache_dir)))
    env_dir = (os.environ.get("NM_MODEL_CACHE") or "").strip()
    if env_dir:
        dirs.append(Path(os.path.expanduser(env_dir)))
    dirs.append(Path.home() / "models")
    dirs.append(Path("models"))
    found = ""
    for d in dirs:
        found = find_gguf(repo_id, d)
        if found:
            break
    if not found:
        checked = ", ".join(str(d) for d in dirs)
        return False, f"no {repo_id} weights on disk (checked {checked})"
    ok, detail = gguf_check(found)
    if not ok:
        return False, f"weights at {found} fail integrity: {detail}"
    if entry is not None:
        total_gb = max(0.0, memory_mb() / 1024.0)
        if total_gb and entry.size_hint_gb > total_gb * 0.8:
            return False, (
                f"{repo_id} needs ~{entry.size_hint_gb:.1f} GB but this "
                f"box has {total_gb:.1f} GB total")
    return True, f"weights present: {found} ({detail})"


def resolve(job: str, *, profile: Any = None, brain: Any = None,
            model_available: Callable[[], bool] | None = None
            ) -> Resolution:
    """Decide which rung runs ``job`` on this device.

    Rung order: model first when it can actually run, code otherwise.
    ``model_available`` overrides the brain check for jobs whose model
    is data-owned (the DJ's learned taste); for brain jobs the check is
    ``brain.available()``. The returned ``suggest`` is a brain-bound
    SuggestFn when the model rung runs through the brain, else None —
    the caller owns its code rung and any data-owned model object.
    """
    spec = _JOBS.get(job)
    if spec is None:
        known = ", ".join(_JOBS)
        raise ValueError(f"unknown ladder job: {job!r} (known: {known})")
    prof = _coerce_profile(profile)
    timeout = _model_timeout(prof)
    recommended = recommend_local_model(prof)
    local_id = recommended.repo_id if recommended is not None else ""
    local_ok, _local_reason = (
        local_model_available(local_id)
        if local_id else (False, "no local model recommended"))
    if model_available is not None:
        try:
            model_ok = bool(model_available())
        except Exception:  # noqa: BLE001 - availability check must not raise
            model_ok = False
        why = ("learned model available" if model_ok
               else "learned model not available")
    elif spec.model_needs_brain:
        if brain is None:
            model_ok = False
            why = "no brain available on this device"
        else:
            try:
                model_ok = bool(brain.available())
            except Exception:  # noqa: BLE001 - availability check must not raise
                model_ok = False
            if model_ok:
                try:
                    active = brain.active_model() or "model"
                except Exception:  # noqa: BLE001 - name lookup must not raise
                    active = "model"
                why = f"brain available ({active})"
            else:
                why = "brain unavailable"
    else:
        model_ok = False
        why = "no model availability supplied for a data-owned model job"
    if model_ok:
        suggest = None
        if brain is not None and spec.model_needs_brain:
            suggest = _brain_suggest(brain, timeout)
        return Resolution(job=job, rung="model",
                          reason=f"model rung: {why}",
                          suggest=suggest, timeout_s=timeout,
                          local_model=local_id, local_available=local_ok)
    return Resolution(job=job, rung="code",
                      reason=f"code rung: {spec.code_rung} "
                      f"(model rung: {why})",
                      suggest=None, timeout_s=timeout,
                      local_model=local_id, local_available=local_ok)
