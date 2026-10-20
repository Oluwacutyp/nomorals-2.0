"""Immune system — defensive-only police for Devon, like an immune system.

* Thymus: negative selection — antibodies must not fire on self.
* Innate: fast pattern scanning, no learning, no model, no network.
* Adaptive: antigen presentation + INJECTED labelling (never an LLM itself).
* Antibodies: versioned immune memory, exportable across tissues.
* Police: seven-rung ladder. CONSTITUTION — revoke-only, never acquire;
  owner-directed work gets a report, never a block; every action leaves
  a receipt. Teeth apply only to the system's own runaway processes.
* Fever: bounded, announced self-throttle.
* Case file: append-only evidence, sealable.
* Drift: 3-day behaviour watch — suggestions, never actions.
* Pathogens / red-team: the pathogen vocabulary and the immune-evasion
  attack class, as data.
* Events: organ-bus event kinds + injected-sink emit.

Layer 0: standard library only. Integration is via injected callables
and plain data — this package imports nothing from ``nomorals.*``.
Surfaces: ``nm immune`` (cmdline/commands/immune.py) and ``/immune``
in chat (agents/partner/runtime.py → immune/chat.py control_immune).
Wiring (immune/wiring.py): the adaptive cancer gate is LIVE — a
durable sampler job (``__immune_sampler__``, every 1m) feeds real
process samples into a persisted, recency-windowed baseline;
incident samples quarantine instead of teaching the baseline that a
runaway is normal.
"""

from __future__ import annotations

from .adaptive import AntigenBundle, LabelledAntigen, label_antigen, present, should_create_antibody
from .baseline import (
    MIN_SAMPLES,
    AdaptiveBaseline,
    Thresholds,
    adaptive_consult_then_enforce,
    collect_process_sample,
    sample_over_thresholds,
)
from .wiring import (
    BASELINE_WINDOW,
    IMMUNE_SAMPLER_INTERVAL_S,
    IMMUNE_SAMPLER_JOB,
    IMMUNE_SAMPLER_SPEC,
    IMMUNE_SAMPLER_TOOL,
    BaselineStore,
    ImmuneMonitor,
    default_store_path,
    ensure_immune_sampler_job,
    reset_shared_monitors,
    shared_monitor,
)
from .antibodies import AntibodyStore
from .casefile import CaseFile, build_case
from .drift import BehaviourDriftMonitor, DriftReport
from .events import EVENT_KINDS, emit_immune_event, immune_event
from .fever import FEVER_MAX_SECONDS, Fever, FeverReport
from .innate import BARRIERS, InnateHit, evaluate_live, scan_content
from .pathogens import ATTACK_SCENARIOS, PATHOGEN_CLASSES, REDTEAM_IMMUNE_CLASS
from .police import (
    CANCER_MAX_PROCESSES,
    RUNGS,
    Receipt,
    Police,
    consult_then_enforce,
    mark_tainted,
    redteam_selftest,
)
from .thymus import Antibody, SelfCorpus, SelfExemplar, Thymus, test_antibody

__all__ = [
    # thymus
    "Antibody",
    "SelfExemplar",
    "SelfCorpus",
    "test_antibody",
    "Thymus",
    # innate
    "BARRIERS",
    "InnateHit",
    "scan_content",
    "evaluate_live",
    # adaptive
    "AntigenBundle",
    "LabelledAntigen",
    "present",
    "label_antigen",
    "should_create_antibody",
    # antibodies
    "AntibodyStore",
    # police
    "RUNGS",
    "CANCER_MAX_PROCESSES",
    "Receipt",
    "mark_tainted",
    "Police",
    "consult_then_enforce",
    "redteam_selftest",
    # fever
    "FEVER_MAX_SECONDS",
    "Fever",
    "FeverReport",
    # casefile
    "CaseFile",
    "build_case",
    # drift
    "BehaviourDriftMonitor",
    "DriftReport",
    # pathogens / redteam
    "PATHOGEN_CLASSES",
    "REDTEAM_IMMUNE_CLASS",
    "ATTACK_SCENARIOS",
    # events
    "EVENT_KINDS",
    "immune_event",
    "emit_immune_event",
    # adaptive baseline (percentile cancer thresholds)
    "MIN_SAMPLES",
    "AdaptiveBaseline",
    "Thresholds",
    "adaptive_consult_then_enforce",
    "collect_process_sample",
    "sample_over_thresholds",
    # wiring (persisted baseline, live sampler, incident quarantine)
    "BASELINE_WINDOW",
    "BaselineStore",
    "ImmuneMonitor",
    "default_store_path",
    "shared_monitor",
    "reset_shared_monitors",
    "ensure_immune_sampler_job",
    "IMMUNE_SAMPLER_JOB",
    "IMMUNE_SAMPLER_TOOL",
    "IMMUNE_SAMPLER_SPEC",
    "IMMUNE_SAMPLER_INTERVAL_S",
    # chat control
    "control_immune",
    "IMMUNE_USAGE",
]

from .chat import IMMUNE_USAGE, control_immune
