"""Declarative provider specs (Craft-Libs port of ai-core
``spec.rs``).

A spec is a small data file that describes how to call a provider
whose API is not OpenAI-shaped. A spec runs no code. It names the
calls to make, where the key goes and where in each reply to find
the result. The types live in :mod:`types`, the checker in
:mod:`check`, the substitutions in :mod:`template`.

Version 1 leaves out, on purpose: uploads, streaming, signing,
webhooks, request parameters and a second input kind next to the
prompt. A spec that needs one is refused with its name.
"""

from __future__ import annotations

from .check import SpecError, parse
from .limits import (
                     MAX_MODELS,
                     MAX_PATH_BYTES,
                     MAX_POINTER_DEPTH,
                     MAX_RENDERED_BYTES,
                     MAX_SPEC_BYTES,
                     MAX_STATES,
                     MAX_TEMPLATE_BYTES,
                     SPEC_VERSION,
)
from .template import render_text, render_value
from .types import (
                     Call,
                     Extract,
                     ExtractAs,
                     Flow,
                     Method,
                     Spec,
                     SpecAuth,
                     SpecLimits,
                     SpecModel,
                     StateWord,
)

__all__ = [
    "Call",
    "Extract",
    "ExtractAs",
    "Flow",
    "MAX_MODELS",
    "MAX_PATH_BYTES",
    "MAX_POINTER_DEPTH",
    "MAX_RENDERED_BYTES",
    "MAX_SPEC_BYTES",
    "MAX_STATES",
    "MAX_TEMPLATE_BYTES",
    "Method",
    "SPEC_VERSION",
    "Spec",
    "SpecAuth",
    "SpecError",
    "SpecLimits",
    "SpecModel",
    "StateWord",
    "parse",
    "render_text",
    "render_value",
]
