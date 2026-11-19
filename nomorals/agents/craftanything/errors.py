"""Typed failures for the Craft-Anything organ.

Upstream shapes these as ``ToolError`` with an ``ErrorCode``
(``crates/evidence/src/error.rs``); Devon already has the typed error
family in :mod:`nomorals.core.errors`, so this module only adds the
two refusals the upstream engine can return that Devon has no exact
name for. Everything else reuses the core family.
"""

from __future__ import annotations

from ...core.errors import ToolError

__all__ = ["EffectRefused", "ConsentRequiredError"]


class EffectRefused(ToolError):
    """A call asked for an effect outside the tool's declared effects.

    Raised (as a result, never a crash) before anything executes, with
    the missing effect named.
    """


class ConsentRequiredError(ToolError):
    """A call needs the owner's OK first (upstream ``ConsentRequired``).

    The message names the tool and the effect; the remediation says how
    to approve it or allow it for the tool in standing rules.
    """
