"""Preference adapters — domain -> descriptor, nothing more.

An adapter translates between a domain's items and its descriptor:
it names the axes for the domain and extracts axis values from an
item.  Built-in adapters are pure descriptor passthroughs: they carry
no taste, no defaults, no keyword lists.  All taste lives in the
learned weights in :mod:`nomorals.preferences.store`.

Custom adapters register via :func:`register_adapter`; a custom
adapter may teach the system how a new domain's items are shaped, but
it must not bake in a preference (the registry does not check taste,
the design does — adapters return descriptors and extractions only).

Only the standard library is imported.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .store import DOMAIN_DESCRIPTORS, extract_axis_values

__all__ = [
    "Adapter",
    "DescriptorAdapter",
    "PassthroughAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "reset_adapters",
]


@runtime_checkable
class Adapter(Protocol):
    """Domain -> descriptor translation.  No taste, no ranking."""

    domain: str

    def describe(self) -> dict[str, Any]:
        """The descriptor for this adapter's domain."""
        ...

    def extract(self, item: Any) -> dict[str, list[str]]:
        """Axis values present on ``item``: ``{axis: [value, ...]}``."""
        ...


class DescriptorAdapter:
    """Pure descriptor passthrough for one domain in DOMAIN_DESCRIPTORS."""

    def __init__(self, domain: str) -> None:
        if domain not in DOMAIN_DESCRIPTORS:
            raise ValueError(f"unknown domain {domain!r}; "
                             f"known: {sorted(DOMAIN_DESCRIPTORS)}")
        self.domain = domain

    def describe(self) -> dict[str, Any]:
        return dict(DOMAIN_DESCRIPTORS[self.domain])

    def extract(self, item: Any) -> dict[str, list[str]]:
        return extract_axis_values(item, self.domain)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DescriptorAdapter(domain={self.domain!r})"


# Alias: the built-in adapters are passthroughs, nothing else.
PassthroughAdapter = DescriptorAdapter

_REGISTRY: dict[str, Any] = {}


def _register_builtins() -> None:
    for domain in DOMAIN_DESCRIPTORS:
        _REGISTRY.setdefault(domain, DescriptorAdapter(domain))


_register_builtins()


def register_adapter(domain: str, adapter: Any) -> None:
    """Register (or replace) the adapter for ``domain``.

    ``adapter`` must provide ``describe()`` and ``extract(item)`` and
    expose ``domain``.  Registration is the only way a new domain's
    item shape is taught; it never carries taste.
    """
    if not hasattr(adapter, "describe") or not hasattr(adapter, "extract"):
        raise TypeError("adapter must provide describe() and extract(item)")
    _REGISTRY[domain] = adapter


def get_adapter(domain: str) -> Any | None:
    """The adapter for ``domain``, or ``None`` when unknown."""
    if domain in _REGISTRY:
        return _REGISTRY[domain]
    if domain in DOMAIN_DESCRIPTORS:
        adapter = DescriptorAdapter(domain)
        _REGISTRY[domain] = adapter
        return adapter
    return None


def list_adapters() -> list[str]:
    """Domains with a registered adapter."""
    return sorted(_REGISTRY)


def reset_adapters() -> None:
    """Restore the built-in passthrough registry (tests / fresh state)."""
    _REGISTRY.clear()
    _register_builtins()
