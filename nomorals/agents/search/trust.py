"""Wave 85 source trust — re-exported from the core layer.

The implementation lives in :mod:`nomorals.core.trust` so the browser tool
(low layer) can use the domain ladder without an agents→tools inversion.
This shim keeps the search engine's import path stable.
"""
from ...core.trust import SourceTrust, domain_tier

__all__ = ["SourceTrust", "domain_tier"]
