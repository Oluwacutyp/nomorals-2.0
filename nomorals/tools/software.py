"""Software tool: ``software_alternatives`` — the brain's alternatives.

Devon asks "what are the alternatives to X" and gets a ranked list
with the reason each one matches (license, platform, shared task,
crowd likes where the live site carried them). The tool is a thin
surface over :mod:`nomorals.sources.software`; the organ owns the
catalogue, the live fetch, and the ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..sources.software import AlternativeToSource, SoftwareCatalog

__all__ = ["register"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home", None)
    if home:
        return str(home)
    home_path = getattr(settings, "home_path", None)
    if home_path:
        return str(Path(home_path))
    return None


def register(registry: Any) -> None:
    """Attach the ``software_alternatives`` tool to a registry."""
    context = registry.context

    @registry.register(
        "software_alternatives",
        description=(
            "Find software alternatives to a named app (AlternativeTo "
            "crowd-ranked catalogue, grown by live fetch, seeded so it "
            "answers offline too). Devon herself goes first where she "
            "does the same job. software=<app name>; platform=<linux|"
            "windows|mac|online|...> filters by platform; license_tier="
            "<free|freemium|paid|...> filters by license; open_source="
            "true|false filters to open-source only; limit caps results. "
            "Returns ranked alternatives with the reason each matches "
            "(license, platform, shared task, likes where the live site "
            "carried them)."
        ),
        capability=Capability.NET_OUT,
    )
    def software_alternatives_tool(
        software: str = "", platform: str = "", license_tier: str = "",
        open_source: bool | None = None, limit: int = 10,
    ) -> dict[str, Any]:
        if not software or not str(software).strip():
            raise ValueError("software_alternatives needs a software name")
        home = _home_from_context(context)
        catalog = SoftwareCatalog(home)
        source = AlternativeToSource(catalog=catalog)
        results = source.alternatives(
            str(software).strip(),
            platform=platform.strip() or None,
            license_tier=license_tier.strip() or None,
            open_source=open_source, limit=limit)
        return {
            "ok": True,
            "software": str(software).strip(),
            "source": "alternativeto",
            "results": [
                {
                    "name": r.name, "slug": r.slug, "url": r.url,
                    "license_tier": r.license_tier,
                    "open_source": r.open_source,
                    "license_id": r.license_id,
                    "categories": list(r.categories),
                    "platforms": list(r.platforms),
                    "likes": r.likes, "is_devon": r.is_devon,
                    "reason": r.reason,
                    "description": r.description,
                }
                for r in results
            ],
        }
