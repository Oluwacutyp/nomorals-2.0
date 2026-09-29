"""Archive tool for managing archives."""

from __future__ import annotations

from typing import Any


def archive(action: str = "list", path: str = "", **kwargs) -> dict[str, Any]:
    """Manage archives."""
    return {
        "action": action,
        "path": path,
        "status": "ok",
        **kwargs
    }


def register(registry: Any) -> None:
    """Register archive tool with the registry."""
    registry.register(
        "archive",
        archive,
        description="Manage archives",
        capability="archive",
        parameters={
            "action": {"type": "string", "description": "Action to perform"},
            "path": {"type": "string", "description": "Archive path"}
        }
    )
