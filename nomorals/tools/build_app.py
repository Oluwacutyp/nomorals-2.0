"""Build app tool for creating applications."""

from __future__ import annotations

from typing import Any


def build_app(name: str = "", template: str = "basic", **kwargs) -> dict[str, Any]:
    """Build an application from a template."""
    return {
        "name": name,
        "template": template,
        "status": "built",
        "path": f"/tmp/{name}"
    }


def register(registry: Any) -> None:
    """Register build_app tool with the registry."""
    registry.register(
        "build_app",
        build_app,
        description="Build an application from a template",
        capability="build",
        parameters={
            "name": {"type": "string", "description": "Application name"},
            "template": {"type": "string", "description": "Template to use"}
        }
    )
