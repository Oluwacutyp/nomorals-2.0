"""Sandboxed plugin loading.

Plugins run in-process but isolated: each plugin's modules load under a
private ``nomorals.plugins._loaded.<name>`` namespace (never injected
into ``sys.modules`` under their bare name), and the *only* Devon
surface a plugin receives is the :class:`PluginCapabilities` object
handed to its entry point. Every capability method checks the granted
permission set first — ungranted access raises
:exc:`PermissionDenied` — and reports the call to the audit sink when
one is wired, so every capability use is explicit and auditable.

Beyond one-shot entry points, plugins can declare **hooks**
(``manifest["hooks"]``): named extension points the host emits and every
attached plugin can implement, in priority order, with per-plugin error
isolation (pluggy's semantics, minus the decorator framework). See
:class:`PluginHookBus`.

This is cooperative sandboxing (CPython has no true sandbox): it stops
accidental overreach and makes every capability use explicit and
auditable. It does not defend against actively malicious code — don't
install plugins you don't trust. Long-running or untrusted entry points
should be invoked through :func:`call_entry` with a timeout so a hung
plugin can't hang the host.
"""

from __future__ import annotations

import importlib.util
import inspect
import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .errors import LoadError, PermissionDenied, PluginTimeout
from .manifest import DEFAULT_LIMITS, PluginManifest

__all__ = [
    "PluginCapabilities",
    "PluginConfig",
    "LoadedPlugin",
    "PluginHookBus",
    "HookCall",
    "HookResults",
    "load_plugin",
    "unload_plugin",
    "call_entry",
]

_log = get_logger(__name__)

#: ``audit(action, permission, detail)`` — wired by ``wiring.py`` to the
#: event bus; ``None`` disables auditing.
AuditSink = Callable[[str, str, str], None]


class PluginConfig:
    """Schema-validated settings for one plugin.

    Reads merge ``manifest.default_config`` with stored overrides;
    writes validate against ``manifest.config_schema`` and persist into
    the plugin's kv store under a reserved key (so they survive
    upgrades — kv is namespaced per plugin *name*, not version).

    Without a kv store wired (``storage.kv`` not granted), reads still
    work from defaults but writes raise :exc:`PermissionDenied`.
    """

    _KV_KEY = "__plugin_config__"

    def __init__(self, manifest: PluginManifest, kv: Any | None = None
                 ) -> None:
        self._manifest = manifest
        self._kv = kv

    def _overrides(self) -> dict[str, Any]:
        if self._kv is None:
            return {}
        try:
            stored = self._kv.get(self._KV_KEY, {})
        except Exception:  # noqa: BLE001 - kv backends vary; read defensively
            return {}
        return stored if isinstance(stored, dict) else {}

    def all(self) -> dict[str, Any]:
        """Defaults merged with stored overrides (overrides win)."""
        from .manifest import apply_schema_defaults
        merged = dict(self._manifest.default_config)
        merged.update(self._overrides())
        return apply_schema_defaults(merged, self._manifest.config_schema)

    def get(self, key: str, default: Any = None) -> Any:
        return self.all().get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Validate ``value`` and persist it as an override."""
        self.update({key: value})

    def update(self, mapping: dict[str, Any]) -> None:
        """Validate a batch of overrides and persist them."""
        if self._kv is None:
            raise PermissionDenied(
                "plugin config: no store wired (grant 'storage.kv' to "
                "persist settings)")
        if not isinstance(mapping, dict):
            raise PermissionDenied(
                f"plugin config: update needs a dict, got "
                f"{type(mapping).__name__}")
        from .manifest import validate_against_schema
        schema = self._manifest.config_schema
        merged = self.all()
        merged.update(mapping)
        if schema:
            props = schema.get("properties", {})
            # Validate per-key when the schema describes the key —
            # sharper errors than whole-object validation.
            for key, value in mapping.items():
                sub = props.get(key)
                if isinstance(sub, dict):
                    validate_against_schema(value, sub, f"config.{key}")
            validate_against_schema(merged, schema, "config")
        overrides = self._overrides()
        overrides.update(mapping)
        self._kv.set(self._KV_KEY, overrides)

    def reset(self) -> None:
        """Drop all stored overrides, back to manifest defaults."""
        if self._kv is None:
            raise PermissionDenied(
                "plugin config: no store wired (grant 'storage.kv' to "
                "persist settings)")
        self._kv.delete(self._KV_KEY)


class PluginCapabilities:
    """The gated Devon surface handed to a plugin entry point.

    ``granted`` is the permission set from the manifest/install record.
    Each accessor checks its permission before returning anything.

    ``audit`` is an optional ``(action, permission, detail)`` sink —
    every capability method reports through it. ``limits`` carries the
    manifest's per-run budgets (``fetch_calls``, ``chat_calls`` …);
    exceeding one raises :exc:`PermissionDenied` with the budget named.
    """

    def __init__(self, *, granted: frozenset[str],
                 plugin_name: str = "",
                 manifest: PluginManifest | None = None,
                 artifacts: Any = None, kv: Any = None,
                 fetcher: Any = None, chatter: Any = None,
                 notifier: Any = None,
                 audit: AuditSink | None = None,
                 limits: dict[str, float] | None = None) -> None:
        self._granted = granted
        self._plugin_name = plugin_name
        self._manifest = manifest
        self._artifacts = artifacts
        self._kv = kv
        self._fetcher = fetcher
        self._chatter = chatter
        self._notifier = notifier
        self._audit = audit
        self._limits = dict(DEFAULT_LIMITS)
        if limits:
            self._limits.update(limits)
        self._usage: dict[str, int] = {"fetch_calls": 0, "chat_calls": 0}

    def _check(self, perm: str) -> None:
        if perm not in self._granted:
            raise PermissionDenied(
                f"plugin lacks permission {perm!r}; granted: "
                f"{sorted(self._granted)}")

    def _record(self, action: str, permission: str, detail: str = "") -> None:
        if self._audit is not None:
            try:
                self._audit(action, permission, detail)
            except Exception:  # noqa: BLE001 - audit is fail-open
                _log.debug("capability audit failed", exc_info=True)

    def _budget(self, kind: str) -> None:
        limit = self._limits.get(kind, 0)
        used = self._usage.get(kind, 0)
        if limit and used >= limit:
            raise PermissionDenied(
                f"plugin {self._plugin_name!r} exceeded its {kind} budget "
                f"({int(limit)} per run)")
        self._usage[kind] = used + 1

    @property
    def granted(self) -> frozenset[str]:
        return self._granted

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    @property
    def usage(self) -> dict[str, int]:
        """Per-run capability counters (fetch_calls, chat_calls)."""
        return dict(self._usage)

    def artifacts(self) -> Any:
        """Artifact store (put/get). Needs ``artifacts.read`` to read,
        ``artifacts.write`` to write — checked per call."""
        return _GuardedArtifacts(self)

    def kv(self) -> Any:
        """Private key-value store for this plugin. Needs ``storage.kv``."""
        self._check("storage.kv")
        self._record("kv", "storage.kv")
        if self._kv is None:
            raise PermissionDenied("storage.kv: no store wired")
        return self._kv

    def fetch(self, url: str, *, timeout: int = 30) -> bytes:
        """Fetch a URL. Needs ``network.fetch``."""
        self._check("network.fetch")
        self._budget("fetch_calls")
        self._record("fetch", "network.fetch", str(url)[:200])
        if self._fetcher is None:
            raise PermissionDenied("network.fetch: no fetcher wired")
        return self._fetcher(url, timeout=timeout)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Chat completion via the model broker. Needs ``llm.chat``."""
        self._check("llm.chat")
        self._budget("chat_calls")
        self._record("chat", "llm.chat",
                     f"{len(messages) if isinstance(messages, list) else '?'} msgs")
        if self._chatter is None:
            raise PermissionDenied("llm.chat: no model wired")
        return self._chatter(messages)

    def notify(self, text: str, *, title: str = "") -> None:
        """Send a message back to the owner. Needs ``notify.send``.

        The default backend publishes a ``plugin.notify`` event on the
        host bus — owner surfaces (chat adapters) deliver it.
        """
        self._check("notify.send")
        self._record("notify", "notify.send", str(text)[:200])
        if self._notifier is None:
            raise PermissionDenied("notify.send: no notifier wired")
        self._notifier(str(text), title=str(title))

    def config(self) -> PluginConfig:
        """This plugin's settings (schema-validated, persisted in kv).

        Reads always work (manifest defaults); writes need
        ``storage.kv`` granted so overrides can persist.
        """
        self._record("config", "", "")
        manifest = self._manifest or PluginManifest(
            name=self._plugin_name or "unknown", version="0.0.0")
        kv = self._kv if "storage.kv" in self._granted else None
        return PluginConfig(manifest, kv)


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
        self._caps._record("artifacts.put", "artifacts.write")
        return self._store("artifacts.write").put(data, **kwargs)

    def put_text(self, text: str, **kwargs: Any) -> Any:
        self._caps._record("artifacts.put_text", "artifacts.write")
        return self._store("artifacts.write").put_text(text, **kwargs)

    def get(self, artifact_id: str) -> Any:
        self._caps._record("artifacts.get", "artifacts.read", artifact_id)
        self._caps._check("artifacts.read")
        store = self._caps._artifacts
        if store is None:
            raise PermissionDenied("artifacts.read: no store wired")
        return store.get(artifact_id)


@dataclass
class HookCall:
    """One plugin's answer to a hook emission."""
    plugin: str
    hook: str
    ok: bool
    result: Any = None
    error: str = ""


@dataclass
class HookResults:
    """Everything that happened during one :meth:`PluginHookBus.emit`."""
    calls: list[HookCall] = field(default_factory=list)

    @property
    def results(self) -> list[tuple[str, Any]]:
        """``(plugin_name, result)`` for every successful impl, in order."""
        return [(c.plugin, c.result) for c in self.calls if c.ok]

    @property
    def errors(self) -> list[tuple[str, str]]:
        """``(plugin_name, error)`` for every failed impl, in order."""
        return [(c.plugin, c.error) for c in self.calls if not c.ok]

    def first(self) -> Any:
        """First non-None result (pluggy ``firstresult`` semantics)."""
        for _, result in self.results:
            if result is not None:
                return result
        return None


@dataclass
class _HookImpl:
    plugin: str
    hook: str
    priority: int
    fn: Callable[..., Any]
    caps: PluginCapabilities


class PluginHookBus:
    """Emit named hooks to every attached plugin that implements them.

    A plugin declares hooks in its manifest::

        "hooks": {
          "on_message": "my_plugin:on_message",
          "on_boot": {"entry": "my_plugin:on_boot", "priority": 10}
        }

    :meth:`attach` resolves each declared entry point and validates the
    implementation **at attach time** (must be callable and accept the
    capabilities object as its first argument) — a bad impl raises
    :exc:`LoadError` immediately, not when the hook fires.

    :meth:`emit` calls impls in priority order (highest first, then plugin
    name for determinism), each as ``fn(caps, *args, **kwargs)``. One
    plugin's exception is captured in the returned :class:`HookResults`
    and never stops the others. :meth:`emit_first` returns the first
    non-None result, for resolution-style hooks.
    """

    def __init__(self) -> None:
        self._impls: dict[str, list[_HookImpl]] = {}

    def attach(self, loaded: LoadedPlugin,
               caps: PluginCapabilities) -> int:
        """Register every hook impl declared by a loaded plugin.

        Returns the number of hooks attached.
        """
        attached = 0
        for hook in loaded.manifest.hooks:
            fn = loaded.resolve(hook.entry)
            self._validate_impl(loaded.name, hook.name, fn)
            impl = _HookImpl(plugin=loaded.name, hook=hook.name,
                             priority=hook.priority, fn=fn, caps=caps)
            bucket = self._impls.setdefault(hook.name, [])
            bucket.append(impl)
            bucket.sort(key=lambda i: (-i.priority, i.plugin))
            attached += 1
        _log.info("attached %d hooks for plugin %s",
                  attached, loaded.name)
        return attached

    def detach(self, plugin_name: str) -> int:
        """Remove every hook impl belonging to a plugin."""
        removed = 0
        for hook, bucket in list(self._impls.items()):
            kept = [i for i in bucket if i.plugin != plugin_name]
            removed += len(bucket) - len(kept)
            if kept:
                self._impls[hook] = kept
            else:
                self._impls.pop(hook, None)
        return removed

    def hooks(self) -> dict[str, list[str]]:
        """Map of hook name → plugin names implementing it, in call order."""
        return {hook: [i.plugin for i in bucket]
                for hook, bucket in sorted(self._impls.items())}

    def emit(self, hook: str, *args: Any, **kwargs: Any) -> HookResults:
        """Call every impl of ``hook``; isolate per-plugin failures."""
        results = HookResults()
        for impl in list(self._impls.get(hook, [])):
            try:
                result = impl.fn(impl.caps, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - isolate, record
                _log.warning("plugin %s hook %s failed: %s",
                             impl.plugin, hook, exc)
                results.calls.append(HookCall(
                    plugin=impl.plugin, hook=hook, ok=False,
                    error=f"{type(exc).__name__}: {exc}"))
            else:
                results.calls.append(HookCall(
                    plugin=impl.plugin, hook=hook, ok=True, result=result))
        return results

    def emit_first(self, hook: str, *args: Any, **kwargs: Any) -> Any:
        """First non-None result wins; failures are skipped silently
        (they're still logged)."""
        return self.emit(hook, *args, **kwargs).first()

    @staticmethod
    def _validate_impl(plugin_name: str, hook: str,
                       fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise LoadError(
                f"plugin {plugin_name!r}: hook {hook!r} target is not "
                f"callable")
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise LoadError(
                f"plugin {plugin_name!r}: hook {hook!r} has no "
                f"inspectable signature: {exc}") from exc
        params = list(sig.parameters.values())
        positional = [p for p in params
                      if p.kind in (p.POSITIONAL_ONLY,
                                    p.POSITIONAL_OR_KEYWORD)]
        accepts_var = any(p.kind == p.VAR_POSITIONAL for p in params)
        if not accepts_var and len(positional) < 1:
            raise LoadError(
                f"plugin {plugin_name!r}: hook {hook!r} must accept the "
                f"capabilities object as its first argument")


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

    def resolve(self, target: str) -> Callable[..., Any]:
        """Resolve a ``"module:attr"`` target to a callable, without
        calling it. Raises :exc:`LoadError` when anything is off."""
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
        return fn

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
        return self.resolve(target)(caps)

    def lifecycle(self, hook: str, caps: PluginCapabilities) -> Any:
        """Run a lifecycle entry (``on_install`` …) when declared.

        Returns ``None`` without calling anything when the manifest
        doesn't declare that lifecycle hook.
        """
        target = self.manifest.lifecycle.get(hook)
        if not target:
            return None
        return self.resolve(target)(caps)


def call_entry(loaded: LoadedPlugin, key: str, caps: PluginCapabilities,
               *, timeout: float | None = None) -> Any:
    """Call an entry point with a wall-clock timeout.

    Without ``timeout`` this is exactly ``loaded.entry(key, caps)``. With
    one, the call runs on a daemon worker thread; when the budget
    expires the *caller* gets :exc:`PluginTimeout` immediately while the
    worker thread is abandoned (CPython cannot kill threads — the plugin
    code keeps running in the background until it returns on its own).
    Exceptions raised by the plugin propagate unchanged.
    """
    if timeout is None or timeout <= 0:
        return loaded.entry(key, caps)
    box: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            box.put((True, loaded.entry(key, caps)))
        except Exception as exc:  # noqa: BLE001 - ferried back to caller
            box.put((False, exc))

    worker = threading.Thread(
        target=_run, name=f"plugin-{loaded.name}-{key}", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise PluginTimeout(
            f"plugin {loaded.name!r} entry {key!r} exceeded {timeout:g}s "
            f"(worker abandoned, not killed)")
    ok, payload = box.get_nowait()
    if ok:
        return payload
    raise payload


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
    # Purge stale modules from a previous load of the same plugin (e.g.
    # after an upgrade) so the new code actually runs.
    _purge_namespace(namespace)
    modules = _import_tree(root, namespace)
    if not modules:
        raise LoadError(f"plugin {manifest.name!r} contains no Python modules")
    _log.info("loaded plugin %s (%d modules)",
              manifest.name, len(modules))
    return LoadedPlugin(manifest, root, modules)


def unload_plugin(loaded: LoadedPlugin) -> None:
    """Purge a loaded plugin's isolated modules from ``sys.modules``.

    Use after ``run`` when the plugin won't be needed again, or before
    re-loading an upgraded copy. Idempotent.
    """
    _purge_namespace(f"nomorals.plugins._loaded.{loaded.manifest.name}")
    loaded._modules.clear()
    _log.info("unloaded plugin %s", loaded.manifest.name)


def _purge_namespace(namespace: str) -> None:
    prefix = namespace + "."
    for name in [n for n in sys.modules
                 if n == namespace or n.startswith(prefix)]:
        sys.modules.pop(name, None)
