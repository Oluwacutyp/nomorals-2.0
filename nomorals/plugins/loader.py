"""Sandboxed plugin loading.

Plugins run in-process but isolated: each plugin's modules load under a
private ``nomorals.plugins._loaded.<name>`` namespace (never injected
into ``sys.modules`` under their bare name), and the *only* Devon
surface a plugin receives is the :class:`PluginCapabilities` object
handed to its entry point. Every capability method checks the granted
permission set first — ungranted access raises
:exc:`PermissionDenied`.

This is cooperative sandboxing (CPython has no true sandbox): it stops
accidental overreach and makes every capability use explicit and
auditable. It does not defend against actively malicious code — don't
install plugins you don't trust.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from ..core.logging_setup import get_logger
from .errors import LoadError, PermissionDenied
from .manifest import PluginManifest

__all__ = [
    "PluginCapabilities",
    "LoadedPlugin",
    "load_plugin",
]

_log = get_logger(__name__)


class PluginCapabilities:
    """The gated Devon surface handed to a plugin entry point.

    ``granted`` is the permission set from the manifest/install record.
    Each accessor checks its permission before returning anything.
    """

    def __init__(self, *, granted: frozenset[str],
                 artifacts: Any = None, kv: Any = None,
                 fetcher: Any = None, chatter: Any = None) -> None:
        self._granted = granted
        self._artifacts = artifacts
        self._kv = kv
        self._fetcher = fetcher
        self._chatter = chatter

    def _check(self, perm: str) -> None:
        if perm not in self._granted:
            raise PermissionDenied(
                f"plugin lacks permission {perm!r}; granted: "
                f"{sorted(self._granted)}")

    @property
    def granted(self) -> frozenset[str]:
        return self._granted

    def artifacts(self) -> Any:
        """Artifact store (put/get). Needs ``artifacts.read`` to read,
        ``artifacts.write`` to write — checked per call."""
        return _GuardedArtifacts(self)

    def kv(self) -> Any:
        """Private key-value store for this plugin. Needs ``storage.kv``."""
        self._check("storage.kv")
        if self._kv is None:
            raise PermissionDenied("storage.kv: no store wired")
        return self._kv

    def fetch(self, url: str, *, timeout: int = 30) -> bytes:
        """Fetch a URL. Needs ``network.fetch``."""
        self._check("network.fetch")
        if self._fetcher is None:
            raise PermissionDenied("network.fetch: no fetcher wired")
        return self._fetcher(url, timeout=timeout)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Chat completion via the model broker. Needs ``llm.chat``."""
        self._check("llm.chat")
        if self._chatter is None:
            raise PermissionDenied("llm.chat: no model wired")
        return self._chatter(messages)


class _GuardedArtifacts:
    def __init__(self, caps: PluginCapabilities) -> None:
        self._caps = caps

    def _store(self, need: str) -> Any:
        self._caps._check(need)
        store = self._caps._artifacts
        if store is None:
            raise PermissionDenied(f"{need}: no artifact store wired")
        return store

    def put(self, data: bytes, **kwargs: Any) -> Any:
        return self._store("artifacts.write").put(data, **kwargs)

    def put_text(self, text: str, **kwargs: Any) -> Any:
        return self._store("artifacts.write").put_text(text, **kwargs)

    def get(self, artifact_id: str) -> Any:
        self._caps._check("artifacts.read")
        store = self._caps._artifacts
        if store is None:
            raise PermissionDenied("artifacts.read: no store wired")
        return store.get(artifact_id)


class LoadedPlugin:
    """A plugin whose modules are imported and entry points resolved."""

    def __init__(self, manifest: PluginManifest, path: Path,
                 modules: dict[str, ModuleType]) -> None:
        self.manifest = manifest
        self.path = path
        self._modules = modules

    @property
    def name(self) -> str:
        return self.manifest.name

    def entry(self, key: str, caps: PluginCapabilities) -> Any:
        """Resolve an entry point and call it with capabilities.

        Entry point targets look like ``"module:attr"``. The attribute
        must be callable; it is invoked as ``fn(caps)``.
        """
        try:
            target = self.manifest.entry_points[key]
        except KeyError:
            raise LoadError(
                f"plugin {self.name!r} has no entry point {key!r}") from None
        module_name, _, attr = target.partition(":")
        full = f"nomorals.plugins._loaded.{self.manifest.name}.{module_name}"
        module = self._modules.get(full)
        if module is None:
            raise LoadError(
                f"plugin {self.name!r}: module {module_name!r} not loaded")
        fn = getattr(module, attr, None)
        if not callable(fn):
            raise LoadError(
                f"plugin {self.name!r}: {target} is not callable")
        return fn(caps)


def _import_tree(root: Path, namespace: str) -> dict[str, ModuleType]:
    """Import every ``*.py`` under ``root`` into an isolated namespace."""
    modules: dict[str, ModuleType] = {}
    # Import parents before children so relative imports work.
    files = sorted(root.rglob("*.py"),
                   key=lambda p: len(p.relative_to(root).parts))
    for py in files:
        rel = py.relative_to(root)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mod_name = namespace + ("." + ".".join(parts) if parts else "")
        spec = importlib.util.spec_from_file_location(mod_name, py)
        if spec is None or spec.loader is None:
            raise LoadError(f"cannot create spec for {py}")
        module = importlib.util.module_from_spec(spec)
        # Register under the isolated name BEFORE exec so dataclasses and
        # relative imports resolve.
        sys.modules[mod_name] = module
        modules[mod_name] = module
        # Make it a package if it has children.
        if (py.parent / "__init__.py").exists() or py.name == "__init__.py":
            module.__path__ = [str(py.parent)]  # type: ignore[attr-defined]
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            # Roll back partial imports — never leave a half-loaded plugin.
            for name in modules:
                sys.modules.pop(name, None)
            raise LoadError(f"plugin module {mod_name} failed: {exc}") from exc
    return modules


def load_plugin(manifest: PluginManifest, path: str | Path) -> LoadedPlugin:
    """Import a plugin directory in isolation. Fail fast on any problem."""
    root = Path(path)
    if not root.is_dir():
        raise LoadError(f"plugin path not a directory: {root}")
    namespace = f"nomorals.plugins._loaded.{manifest.name}"
    modules = _import_tree(root, namespace)
    if not modules:
        raise LoadError(f"plugin {manifest.name!r} contains no Python modules")
    _log.info("loaded plugin %s (%d modules)",
              manifest.name, len(modules))
    return LoadedPlugin(manifest, root, modules)
