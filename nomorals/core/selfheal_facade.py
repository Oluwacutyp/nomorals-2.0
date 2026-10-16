"""Facade over the two self-healing modules — neither is edited.

There are two self-healing modules with complementary jobs, and both
stay exactly as they are:

- :mod:`nomorals.core.selfheal` — the runtime recovery executor:
  ``SelfHealingExecutor`` (analyze → consult memory → recover →
  verify → record), ``RecoveryOutcome`` / ``RecoveryStatus``,
  ``SubsystemSupervisor`` / ``RestartStrategy``.
- :mod:`nomorals.core.self_heal` — the broken-chat-command healer:
  probe → diagnose → conservative auto-fix → re-probe → report →
  learn (``heal_one`` / ``heal_all`` / ``fix_unbound_local`` /
  ``fix_dead_method`` / ``FixStrategy`` / ``FIX_STRATEGIES``).

Their ``__all__`` lists are disjoint (verified: no name overlap), so
this facade re-exports both namespaces under one import without
shadowing anything.  Import this module when a caller wants "the
self-healing system" without caring which file owns which half::

    from nomorals.core.selfheal_facade import SelfHealingExecutor, heal_one

Nothing here deletes, renames, or re-implements a function from
either original — this file is pure re-export plus a small
introspection helper (:func:`describe`).
"""

from __future__ import annotations

from . import self_heal as self_heal_module
from . import selfheal as selfheal_module
from .self_heal import (
    FIX_STRATEGIES,
    FixStrategy,
    HealResult,
    diff_preview,
    fix_dead_method,
    fix_unbound_local,
    heal_all,
    heal_one,
    heal_probe,
    register_fix_strategy,
)
from .selfheal import (
    RecoveryOutcome,
    RecoveryStatus,
    RestartStrategy,
    SelfHealingExecutor,
    SubsystemSupervisor,
)

__all__ = [
    # selfheal (runtime recovery executor)
    "RecoveryOutcome",
    "RecoveryStatus",
    "SelfHealingExecutor",
    "SubsystemSupervisor",
    "RestartStrategy",
    # self_heal (broken-command healer)
    "heal_probe",
    "fix_unbound_local",
    "fix_dead_method",
    "heal_one",
    "heal_all",
    "HealResult",
    "FixStrategy",
    "register_fix_strategy",
    "FIX_STRATEGIES",
    "diff_preview",
    # facade itself
    "selfheal_module",
    "self_heal_module",
    "describe",
    "healing_names",
]

#: The two modules this facade unifies, in canonical order.
SELFHEAL_MODULES = (selfheal_module, self_heal_module)


def healing_names() -> list[str]:
    """Every public name the facade re-exports, sorted."""
    return sorted(set(__all__) - {"selfheal_module", "self_heal_module",
                                  "describe", "healing_names"})


def describe() -> dict[str, object]:
    """Introspection: which names come from which original module.

    Returns ``{"selfheal": [...], "self_heal": [...], "overlap": []}``
    — ``overlap`` must stay empty; if the two modules ever gain a
    shared name, the facade would need an explicit disambiguation
    instead of silently shadowing one.
    """
    selfheal_names = sorted(set(getattr(selfheal_module, "__all__", [])))
    self_heal_names = sorted(set(getattr(self_heal_module, "__all__", [])))
    return {
        "selfheal": selfheal_names,
        "self_heal": self_heal_names,
        "overlap": sorted(set(selfheal_names) & set(self_heal_names)),
        "facade_exports": healing_names(),
    }
