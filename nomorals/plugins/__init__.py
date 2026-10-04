"""Plugin package model: manifests, registry, sandboxed loading."""

from __future__ import annotations

from .errors import (
    AlreadyInstalled,
    LoadError,
    ManifestError,
    NotInstalled,
    PermissionDenied,
    PluginError,
)
from .loader import LoadedPlugin, PluginCapabilities, load_plugin, unload_plugin
from .manifest import (
    KNOWN_PERMISSIONS,
    MANIFEST_FILENAME,
    PluginManifest,
    load_manifest,
    load_manifest_file,
)
from .registry import InstalledPlugin, PluginRegistry
from .wiring import PluginKV, make_fetcher, wire_capabilities

__all__ = [
    "AlreadyInstalled",
    "InstalledPlugin",
    "KNOWN_PERMISSIONS",
    "LoadError",
    "LoadedPlugin",
    "MANIFEST_FILENAME",
    "ManifestError",
    "NotInstalled",
    "PermissionDenied",
    "PluginCapabilities",
    "PluginError",
    "PluginKV",
    "PluginManifest",
    "PluginRegistry",
    "load_manifest",
    "load_manifest_file",
    "load_plugin",
    "make_fetcher",
    "unload_plugin",
    "wire_capabilities",
]
