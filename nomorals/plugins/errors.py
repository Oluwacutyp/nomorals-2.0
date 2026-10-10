"""Errors for the plugin package model."""

from __future__ import annotations

__all__ = [
    "PluginError",
    "ManifestError",
    "AlreadyInstalled",
    "NotInstalled",
    "PermissionDenied",
    "LoadError",
    "PluginTimeout",
    "DependencyError",
    "IncompatibleEngine",
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


class PluginTimeout(PluginError):
    """A plugin entry-point call exceeded its time budget.

    Raised by :func:`loader.call_entry` when the plugin does not return
    within ``timeout`` seconds. The worker thread is abandoned, not
    killed (CPython can't kill threads) — the *caller* gets control
    back; the plugin's thread keeps running in the background until it
    finishes on its own.
    """


class DependencyError(PluginError):
    """A plugin's declared dependencies are not satisfied.

    Raised at install/upgrade time when a manifest ``dependencies``
    entry names a plugin that isn't installed, or the installed version
    doesn't satisfy the declared version spec. The message names exactly
    which dependency failed and why — install the missing plugin (or a
    newer version) and retry.
    """


class IncompatibleEngine(PluginError):
    """A plugin's ``requires_devon`` floor is newer than this engine.

    Raised at install/upgrade time. The message names the required spec
    and the running engine version so the user knows to upgrade Devon,
    not the plugin.
    """
