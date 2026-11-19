"""Conditioning-data errors.

Upstream FineTrainers raises plain ``ValueError`` with exact messages for
dataset problems; this package keeps those messages verbatim but raises
them as :class:`ConditioningError` (a ``ValueError`` subclass) so callers
can catch conditioning failures without catching every value error in
the process. :class:`ConditioningUnavailable` is the honest failure for
the GPU-only half: torch / diffusers / transformers / model weights that
are not present on this machine.
"""

from __future__ import annotations

__all__ = ["ConditioningError", "ConditioningUnavailable"]


class ConditioningError(ValueError):
    """A conditioning dataset, config, or precompute store is unusable.

    The message is upstream FineTrainers' wording wherever the failure
    mirrors an upstream check, so a test that pins the message pins the
    upstream behaviour, not a Devon paraphrase.
    """


class ConditioningUnavailable(ConditioningError):
    """The requested step needs a package or weight that is not here.

    Raised (never silently skipped) when torch, diffusers, transformers,
    or a model checkpoint is required and missing. The message names
    exactly what is missing and what provides it.
    """
