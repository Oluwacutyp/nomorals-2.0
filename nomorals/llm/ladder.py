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

* :func:`resolve` — per job, per profile: ``"model"`` when the job's
  REAL specialist can actually run here, ``"code"`` otherwise, with a
  ``reason`` naming the deciding factor. **The brain is never a rung:**
  the brain directs and (for dialogue jobs) generates candidates, but
  the rung is decided by the specialist's availability — a learned
  scorer trained on the journal, a learned policy, a trained taste
  model. A caller that cannot say whether the specialist is trained
  gets the code rung. The returned ``suggest`` is the candidate
  GENERATOR (brain-bound) when the model rung runs through it, else
  None — the caller owns its code rung and any data-owned model object.
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

Phase 2 (2026-11-12, owner correction): Phase 1 registered the brain
itself as the model rung for dialogue/tactics/narration — a design-down.
The four Phase-1 jobs now name their REAL specialist as the model rung
(the character's learned line scorer, the learned combat policy, the
trained taste model, and — for narration — a specialist that does NOT
exist yet, so the template bank carries and the reason says so). Nine
further specialist jobs are registered below.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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
    "JOBS",
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
    #: False when no specialist model exists for this job yet — the
    #: model rung is then honestly unavailable and the code rung carries.
    specialist_exists: bool = True


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
    model_rung: str = ""  # the specialist named as the model rung


_JOBS: dict[str, JobSpec] = {}

#: Live read-only view of the registry (new jobs appear here).
JOBS = MappingProxyType(_JOBS)


def register_job(spec: JobSpec) -> JobSpec:
    """Register (or replace) a job on the ladder. Returns the spec."""
    _JOBS[spec.name] = spec
    return spec


def list_jobs() -> list[str]:
    """Registered job names, in registration order."""
    return list(_JOBS)


# ── the four Phase-1 jobs ─────────────────────────────────────────────────
# Phase 2 (2026-11-12): the model rung is the REAL specialist, never the
# brain. The brain generates candidates (dialogue) or directs; the
# specialist decides. A job whose specialist does not exist yet
# (narration) names that wall and the code rung carries.

register_job(JobSpec(
    name="character_dialogue",
    description="NPCs talk dynamically, in their own voice, from persona "
                "+ context + memory — not canned lines.",
    model_rung="The character's own learned line scorer — batch fit on "
               "the journal (characters/learn.py fit_scorer); the brain "
               "only generates candidates, the scorer picks the line",
    code_rung="Template forge: mood-band lines parameterized by name "
              "and situation (GameMaster._NPC_LINES, Character fallback)",
))

register_job(JobSpec(
    name="character_tactics",
    description="NPCs choose battle actions dynamically — the model may "
                "override the math, but never invents an action.",
    model_rung="Learned combat policy — journal → grade → retrain "
               "(games/npc_learn.py fit_tactics_policy); picks among "
               "legal actions from the live state",
    code_rung="GameMind.combat_move math: potion/focus/defend/fury "
              "thresholds by hunter rank (deterministic)",
    model_needs_brain=False,
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
    model_rung="Narration specialist — NOT BUILT: no narration model "
               "exists yet; the template bank carries until one lands",
    code_rung="GameMaster template bank, mood-parameterized, no immediate "
              "repeats",
    model_needs_brain=False,
    specialist_exists=False,
))

JOB_NAMES: tuple[str, ...] = tuple(_JOBS)

# ── Phase 2 specialist jobs ──────────────────────────────────────────────
# Each names its real specialist as the model rung and its deterministic
# fallback as the code rung. All are data-owned (model_needs_brain=False):
# the specialist is a learned model on journal rows, never the brain.

register_job(JobSpec(
    name="npc_line_scorer",
    description="An NPC's own line scorer: generated candidates, the "
                "scorer picks the most in-character, outcomes journaled "
                "and graded, retrained from the journal.",
    model_rung="NPC line scorer — sklearn regressor on the NPC journal "
               "(games/npc_learn.py fit_npc_scorer)",
    code_rung="Hand-weight NPC scorer over the same five features "
              "(trait overlap, mood fit, situation fit, novelty, "
              "length fit)",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="npc_tactics",
    description="The NPC combat policy: which legal action wins fights, "
                "learned from journaled rounds graded by fight outcome.",
    model_rung="Learned combat policy — classifier on journaled rounds "
               "(games/npc_learn.py fit_tactics_policy)",
    code_rung="GameMind.combat_move math thresholds by hunter rank",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="pumpdesk_win",
    description="PumpDesk call scorer: the probability a call wins, "
                "learned from the graded call journal — evidence on the "
                "call, never a silent override of the gates.",
    model_rung="Win-probability classifier on graded calls "
               "(agents/pumpdesk/learn.py fit_win_model)",
    code_rung="The incumbent composite score + CALL_THRESHOLD gate "
              "(agents/pumpdesk/score.py evaluate)",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="error_triage",
    description="Error catcher: which failures are real vs noise, "
                "learned from operator-graded failure rows.",
    model_rung="Error classifier on graded failures "
               "(agents/failure_models.py fit_error_classifier)",
    code_rung="Surface every failure (the failures ledger, "
              "agents/failures.py recent_failures)",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="morning_rank",
    description="Morning-pulse ranker: which briefing sections he "
                "actually reads, learned from engagement history.",
    model_rung="Learned ranker on briefing engagement history "
               "(agents/pulse_rank.py fit_pulse_ranker)",
    code_rung="Hand rule: views>=7 with followups/views<0.2 boosts "
              "priority (agents/morning_briefing.py _apply_engagement)",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="character_voice_xtts",
    description="A character's voice: XTTS cloning speaks their lines "
                "in their own voice.",
    model_rung="XTTS voice clone via the voice catalogue "
               "(characters/specialists.py character_voice)",
    code_rung="No voice audio — the line is delivered as text",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="character_voice_rvc",
    description="A character's voice swap: RVC converts a rendered "
                "line into the character's voice identity.",
    model_rung="RVC voice-swap bridge per character "
               "(characters/specialists.py swap_voice)",
    code_rung="The unswapped render (or text) is delivered as-is",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="character_lora",
    description="A character's persona: a LoRA adapter per character on "
                "one shared base, implanted when the character speaks.",
    model_rung="Per-character LoRA adapter slot "
               "(characters/specialists.py lora_for; local_server --lora)",
    code_rung="The shared base model with the character's persona "
              "prompt (no adapter implanted)",
    model_needs_brain=False,
))

register_job(JobSpec(
    name="character_face",
    description="A character's face, only where the character appears "
                "outside the game — guarded face swap under the vault "
                "lock. NPCs in-game do NOT get faces.",
    model_rung="Guarded face swap from the character's face reference "
               "(characters/specialists.py swap_face)",
    code_rung="No swapped image — the target image is delivered as-is",
    model_needs_brain=False,
))


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
            model_available: Callable[[], bool] | None = None,
            specialist_available: Callable[[], bool] | bool | None = None
            ) -> Resolution:
    """Decide which rung runs ``job`` on this device.

    The rung is decided by the job's REAL specialist, never the brain:

    1. A job with no specialist (``specialist_exists=False``) is always
       the code rung — the reason names the missing specialist.
    2. ``specialist_available`` (a bool, or a callable returning one)
       decides the rung when supplied — this is the specialist's own
       availability (a learned scorer trained, a policy fitted, a taste
       model with enough rows). This is the Phase 2 correction: Phase 1
       let brain availability decide the rung.
    3. ``model_available`` is the legacy override for data-owned jobs
       whose model the caller owns (the DJ's learned taste).
    4. With neither supplied the rung is "code": the brain is NOT a
       rung, so a bare brain never promotes a job to the model rung.

    The returned ``suggest`` is the candidate GENERATOR (brain-bound)
    and is set whenever the job needs the brain as its generator
    (dialogue candidates) and a brain is supplied — on EITHER rung.
    The generator is not the rung: on the code rung the brain's
    candidates are picked by the job's code-rung scorer (the hand
    weights), and when the brain is down the template forge supplies
    the candidates instead, so the code rung always works. Jobs whose
    model rung does not use the brain as a generator (tactics, the
    data-owned jobs) never get a suggest. The caller owns its code
    rung and any data-owned model object.
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
    if not spec.specialist_exists:
        model_ok = False
        why = (f"no specialist model exists for {job} yet — the "
               f"model rung is honestly unavailable")
    elif specialist_available is not None:
        try:
            model_ok = bool(specialist_available()
                            if callable(specialist_available)
                            else specialist_available)
        except Exception:  # noqa: BLE001 - availability check must not raise
            model_ok = False
        why = ("learned specialist available" if model_ok
               else "learned specialist not trained yet")
    elif model_available is not None:
        try:
            model_ok = bool(model_available())
        except Exception:  # noqa: BLE001 - availability check must not raise
            model_ok = False
        why = ("learned model available" if model_ok
               else "learned model not available")
    else:
        model_ok = False
        why = ("no specialist availability supplied — the directing "
               "layer is not a rung")
    # The generator flows whenever the job uses the brain to generate
    # candidates, on either rung — it is not the rung. (Tactics and the
    # data-owned jobs never get one: model_needs_brain is False.)
    suggest = None
    if brain is not None and spec.model_needs_brain:
        suggest = _brain_suggest(brain, timeout)
    if model_ok:
        return Resolution(job=job, rung="model",
                          reason=f"model rung: {why}",
                          suggest=suggest, timeout_s=timeout,
                          local_model=local_id, local_available=local_ok,
                          model_rung=spec.model_rung)
    return Resolution(job=job, rung="code",
                      reason=f"code rung: {spec.code_rung} "
                      f"(model rung: {why})",
                      suggest=suggest, timeout_s=timeout,
                      local_model=local_id, local_available=local_ok,
                      model_rung=spec.model_rung)
