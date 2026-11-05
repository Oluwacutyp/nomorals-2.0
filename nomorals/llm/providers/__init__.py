"""Provider backends.

Construction is data-driven (Echelon Wave 4): :func:`build_provider`
walks the declarative registry in :mod:`.registry`, where every
provider — kind, aliases, class path, transport, auth shape,
capabilities, roster hook — is a :class:`ProviderSpec` entry.  Adding
a provider is a data-only change; see the registry module.
"""

from __future__ import annotations

from ..base import LLMProvider
from .registry import (
    AuthSpec,
    ProviderSpec,
    Verification,
    build_provider,
    canonical_kind,
    expected_kinds,
    get_provider_spec,
    list_provider_specs,
    register_provider_spec,
    render_status_markdown,
    require_provider_spec,
    unregister_provider_spec,
)

__all__ = [
    "AuthSpec",
    "LLMProvider",
    "ProviderSpec",
    "Verification",
    "build_provider",
    "canonical_kind",
    "expected_kinds",
    "get_provider_spec",
    "list_provider_specs",
    "register_provider_spec",
    "render_status_markdown",
    "require_provider_spec",
    "unregister_provider_spec",
]
