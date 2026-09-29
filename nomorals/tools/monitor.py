"""Monitor tool for file and URL change detection."""

from __future__ import annotations

import os
import time
from typing import Any


def monitor_tool(action: str = "", path: str = "", interval: int = 60,
                 url: str = "", **kwargs) -> dict[str, Any]:
    """Monitor files or URLs for changes."""
    
    if action == "add":
        return _add_monitor(path, url, interval)
    elif action == "remove":
        return _remove_monitor(path, url)
    elif action == "list":
        return _list_monitors()
    elif action == "status":
        return _monitor_status()
    elif action == "tick":
        return _tick_monitors()
    
    return {"error": f"Unknown action: {action}"}


def _add_monitor(path: str, url: str, interval: int) -> dict[str, Any]:
    """Add a monitor."""
    target = path or url
    if not target:
        return {"error": "No path or URL specified"}
    
    # Store monitor in a simple dict (in production, use database)
    monitors = _get_monitors()
    monitors[target] = {
        "path": path,
        "url": url,
        "interval": interval,
        "last_check": 0,
        "last_hash": "",
        "last_size": 0,
        "error_streak": 0,
        "created": time.time()
    }
    _save_monitors(monitors)
    
    return {"status": "added", "target": target, "interval": interval}


def _remove_monitor(path: str, url: str) -> dict[str, Any]:
    """Remove a monitor."""
    target = path or url
    monitors = _get_monitors()
    
    if target in monitors:
        del monitors[target]
        _save_monitors(monitors)
        return {"status": "removed", "target": target}
    
    return {"error": f"Monitor not found: {target}"}


def _list_monitors() -> dict[str, Any]:
    """List all monitors."""
    monitors = _get_monitors()
    return {"monitors": list(monitors.keys()), "count": len(monitors)}


def _monitor_status() -> dict[str, Any]:
    """Get monitor status."""
    monitors = _get_monitors()
    active = sum(1 for m in monitors.values() if m.get("last_check", 0) > 0)
    return {
        "total": len(monitors),
        "active": active,
        "status": f"{active}/{len(monitors)}"
    }


def _tick_monitors() -> dict[str, Any]:
    """Check all monitors for changes."""
    monitors = _get_monitors()
    checked = 0
    changed = 0
    
    for target, monitor in monitors.items():
        now = time.time()
        if now - monitor.get("last_check", 0) >= monitor.get("interval", 60):
            monitor["last_check"] = now
            checked += 1
            
            # Check for changes
            if monitor.get("path"):
                try:
                    if os.path.exists(monitor["path"]):
                        size = os.path.getsize(monitor["path"])
                        if size != monitor.get("last_size", 0):
                            monitor["last_size"] = size
                            changed += 1
                except Exception:
                    monitor["error_streak"] = monitor.get("error_streak", 0) + 1
    
    _save_monitors(monitors)
    return {"checked": checked, "changed": changed}


def _get_monitors() -> dict[str, Any]:
    """Get monitors from storage."""
    # In production, use database
    # For now, return empty dict
    return {}


def _save_monitors(monitors: dict[str, Any]) -> None:
    """Save monitors to storage."""
    # In production, use database
    pass


def register(registry: Any) -> None:
    """Register monitor tool with the registry."""
    registry.register(
        "monitor",
        monitor_tool,
        description="Monitor files or URLs for changes",
        capability="monitor",
        parameters={
            "action": {"type": "string", "description": "Action: add, remove, list, status, tick"},
            "path": {"type": "string", "description": "File path to monitor"},
            "url": {"type": "string", "description": "URL to monitor"},
            "interval": {"type": "integer", "description": "Check interval in seconds"}
        }
    )
