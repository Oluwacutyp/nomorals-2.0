"""Errors for the plugin package model."""

from __future__ import annotations

__all__ = [
    "PluginError",
    "ManifestError",
    "AlreadyInstalled",
    "NotInstalled",
    "PermissionDenied",
    "LoadError",
]


class PluginError(Exception):
    """Base error for the plugin system."""


class ManifestError(PluginError):
    """A plugin manifest is missing or invalid."""


class AlreadyInstalled(PluginError):
    """That plugin name+version is already installed."""


class NotInstalled(PluginError):
    """No such plugin is installed."""


class PermissionDenied(PluginError):
    """A plugin requested a permission it was not granted."""


class LoadError(PluginError):
    """A plugin module could not be imported or initialized."""
