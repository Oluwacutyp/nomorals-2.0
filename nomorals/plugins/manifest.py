"""Plugin manifests: name, version, entry points, permissions, and more.

A plugin is a directory containing a ``plugin.json`` manifest plus Python
modules. The manifest declares everything the loader needs::

    {
      "name": "my_plugin",
      "version": "1.0.0",
      "display_name": "My Plugin",
      "description": "Does useful things.",
      "author": "owner",
      "license": "MIT",
      "homepage": "https://example.com/my_plugin",
      "tags": ["tools", "web"],
      "requires_devon": ">=0.1.0",
      "dependencies": ["helper-lib>=1.2"],
      "requirements": ["requests>=2.0"],
      "entry_points": {"main": "my_plugin:run", "hooks": "my_plugin:hooks"},
      "lifecycle": {"on_install": "my_plugin:setup"},
      "hooks": {"on_message": {"entry": "my_plugin:on_msg", "priority": 10}},
      "contributes": {
        "commands": [{"name": "shout", "title": "Shout", "description": "..."}],
        "tools": [{"name": "shout", "description": "...",
                   "schema": {"type": "object", "properties": {...}},
                   "entry": "my_plugin:shout_tool"}]
      },
      "config_schema": {"type": "object",
                        "properties": {"loud": {"type": "boolean"}},
                        "required": ["loud"]},
      "default_config": {"loud": true},
      "permissions": ["artifacts.write", "network.fetch", "notify.send"],
      "limits": {"entry_timeout_s": 30, "fetch_calls": 20, "chat_calls": 10},
      "purge_data_on_remove": false
    }

Permission names are dotted capability strings. The loader grants exactly
the permissions listed at install time; anything else the plugin tries
to use raises :exc:`PermissionDenied`. Known permissions:

* ``artifacts.read`` / ``artifacts.write`` — read/write Devon's artifacts
* ``network.fetch`` — outbound HTTP(S) fetches
* ``storage.kv`` — a private key-value table for the plugin
* ``llm.chat`` — call the model broker for chat completions
* ``notify.send`` — send a message back to the owner (via the host bus)

Unknown permission strings are rejected at install — fail fast, no
silent grants. Unknown *top-level* manifest keys are kept in
:attr:`PluginManifest.extra` (forward-compat; the future may define
them) rather than rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ManifestError

__all__ = [
    "KNOWN_PERMISSIONS",
    "MANIFEST_FILENAME",
    "LIFECYCLE_HOOKS",
    "PluginManifest",
    "PluginDependency",
    "PluginHook",
    "ContributedCommand",
    "ContributedTool",
    "load_manifest",
    "load_manifest_file",
    "parse_version_spec",
    "satisfies_version",
    "version_key",
]

MANIFEST_FILENAME = "plugin.json"

KNOWN_PERMISSIONS = frozenset({
    "artifacts.read",
    "artifacts.write",
    "network.fetch",
    "storage.kv",
    "llm.chat",
    "notify.send",
})

#: Lifecycle entry-point names. May appear under ``lifecycle`` or as
#: plain ``entry_points`` keys (the latter wins — explicit beats nested).
LIFECYCLE_HOOKS = (
    "on_install",
    "on_upgrade",
    "on_enable",
    "on_disable",
    "on_uninstall",
)

#: Defaults applied to a manifest's ``limits`` block. Budgets are per
#: capabilities instance (i.e. per run), not global.
DEFAULT_LIMITS: dict[str, float] = {
    "entry_timeout_s": 60.0,
    "fetch_bytes": 10 * 1024 * 1024,
    "fetch_calls": 100,
    "chat_calls": 50,
}

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(([ab]\d+)|([-+][0-9A-Za-z.-]+))?$")
_SPEC_OP_RE = re.compile(r"^(>=|<=|==|~=|!=|>|<)\s*(.+)$")
# Spec versions may be partial ("1", "1.2") — unlike manifest versions,
# which must be full semver.
_SPEC_VERSION_RE = re.compile(
    r"^\d+(\.\d+){0,2}(([ab]\d+)|([-+][0-9A-Za-z.-]+))?$")
_HOOK_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# ── version ordering + spec satisfaction ────────────────────────────

def version_key(version: str) -> tuple:
    """Ordering key so 1.10.0 > 1.9.0 and 1.0.0 > 1.0.0-beta.

    Compares the numeric core first; a release (no suffix) sorts above
    the same core with a prerelease suffix, per semver.
    """
    parts = re.split(r"[.\-+]", str(version))
    nums: list[int] = []
    suffix: list[str] = []
    for p in parts:
        if p.isdigit() and not suffix:
            nums.append(int(p))
        else:
            suffix.append(p)
    return (tuple(nums), (1,) if not suffix else (0, tuple(suffix)))


def parse_version_spec(spec: str) -> list[tuple[str, str]]:
    """Parse ``">=1.2, <2.0"`` into ``[(">=", "1.2"), ("<", "2.0")]``.

    Raises :exc:`ManifestError` on anything unparsable. An empty spec
    means "any version" and parses to ``[]``.
    """
    spec = str(spec or "").strip()
    if not spec:
        return []
    clauses: list[tuple[str, str]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        m = _SPEC_OP_RE.match(raw)
        if not m:
            raise ManifestError(
                f"bad version spec clause {raw!r} in {spec!r}: want "
                f"op+version, e.g. '>=1.2' (ops: >=, <=, ==, ~=, !=, >, <)")
        op, ver = m.group(1), m.group(2).strip()
        if not _SPEC_VERSION_RE.match(ver):
            raise ManifestError(
                f"bad version {ver!r} in spec {spec!r}: want like 1, 1.2, "
                f"or 1.2.3")
        clauses.append((op, ver))
    return clauses


def satisfies_version(version: str, spec: str) -> bool:
    """True when ``version`` satisfies every clause of ``spec``."""
    key = version_key(version)
    for op, ver in parse_version_spec(spec):
        other = version_key(ver)
        if op == "==":
            ok = key == other
        elif op == "!=":
            ok = key != other
        elif op == ">=":
            ok = key >= other
        elif op == "<=":
            ok = key <= other
        elif op == ">":
            ok = key > other
        elif op == "<":
            ok = key < other
        elif op == "~=":  # compatible release: >= ver, == ver.* (drop last)
            nums = other[0]
            prefix = nums[:-1] if len(nums) > 1 else nums
            ok = key >= other and key[0][:len(prefix)] == prefix
        else:  # pragma: no cover - parse_version_spec guards this
            raise ManifestError(f"unknown version op {op!r}")
        if not ok:
            return False
    return True


# ── small dependency-free JSON-schema subset validator ──────────────

_SCHEMA_TYPES = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "array": list, "object": dict, "null": type(None),
}


def _schema_error(where: str, msg: str) -> ManifestError:
    return ManifestError(f"config schema error at {where}: {msg}")


def validate_against_schema(value: Any, schema: dict[str, Any],
                            where: str = "value") -> None:
    """Validate ``value`` against a small JSON-schema subset.

    Supports ``type``, ``enum``, ``required``, ``properties``,
    ``additionalProperties`` (bool), ``items``, ``minimum``/``maximum``
    (and exclusives), ``minLength``/``maxLength``, ``pattern``.
    Raises :exc:`ManifestError` describing the first problem found.
    Dependency-free on purpose — no ``jsonschema`` needed.
    """
    if not isinstance(schema, dict):
        raise _schema_error(where, "schema must be an object")
    want = schema.get("type")
    if want is not None:
        py = _SCHEMA_TYPES.get(want)
        if py is None:
            raise _schema_error(where, f"unknown type {want!r}")
        # bool is a subclass of int — "integer" must not accept True.
        if want == "integer" and isinstance(value, bool):
            raise _schema_error(where, f"want integer, got boolean")
        if not isinstance(value, py) or (
                want == "number" and isinstance(value, bool)):
            raise _schema_error(
                where, f"want {want}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise _schema_error(where, f"{value!r} not in enum {schema['enum']!r}")
    if want == "string" and isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise _schema_error(where, "string too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise _schema_error(where, "string too long")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise _schema_error(
                where, f"does not match pattern {schema['pattern']!r}")
    if want in ("integer", "number") and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        for bound, op in (("minimum", lambda v, b: v >= b),
                          ("maximum", lambda v, b: v <= b),
                          ("exclusiveMinimum", lambda v, b: v > b),
                          ("exclusiveMaximum", lambda v, b: v < b)):
            if bound in schema and not op(value, schema[bound]):
                raise _schema_error(
                    where, f"{value!r} violates {bound}={schema[bound]!r}")
    if want == "array" and isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                validate_against_schema(item, items, f"{where}[{i}]")
    if want == "object" and isinstance(value, dict):
        required = schema.get("required", [])
        for req in required:
            if req not in value:
                raise _schema_error(where, f"missing required key {req!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value and isinstance(sub, dict):
                validate_against_schema(value[key], sub, f"{where}.{key}")
        if schema.get("additionalProperties") is False:
            unknown = [k for k in value if k not in props]
            if unknown:
                raise _schema_error(
                    where, f"additional properties not allowed: {unknown}")


def apply_schema_defaults(value: Any, schema: dict[str, Any]) -> Any:
    """Return ``value`` with schema ``default``\\ s filled in (recursive)."""
    if not isinstance(schema, dict):
        return value
    if schema.get("type") == "object" and isinstance(value, dict):
        out = dict(value)
        for key, sub in schema.get("properties", {}).items():
            if key in out:
                out[key] = apply_schema_defaults(out[key], sub)
            elif isinstance(sub, dict) and "default" in sub:
                out[key] = sub["default"]
        return out
    if schema.get("type") == "array" and isinstance(value, list) \
            and isinstance(schema.get("items"), dict):
        return [apply_schema_defaults(v, schema["items"]) for v in value]
    return value


# ── manifest dataclasses ────────────────────────────────────────────

@dataclass
class PluginDependency:
    """One ``dependencies`` entry: another plugin + version spec."""
    name: str
    spec: str = ""

    def satisfied_by(self, version: str) -> bool:
        return satisfies_version(version, self.spec)


@dataclass
class PluginHook:
    """One ``hooks`` entry: hook name → entry point + call priority."""
    name: str
    entry: str
    priority: int = 0


@dataclass
class ContributedCommand:
    """A command the plugin declares (visible without loading code)."""
    name: str
    title: str = ""
    description: str = ""


@dataclass
class ContributedTool:
    """An agent tool the plugin contributes, with a JSON input schema."""
    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    entry: str = ""


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str = ""
    author: str = ""
    entry_points: dict[str, str] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)
    # ── extended metadata ──
    display_name: str = ""
    license: str = ""
    homepage: str = ""
    icon: str = ""
    tags: list[str] = field(default_factory=list)
    requires_devon: str = ""
    dependencies: list[PluginDependency] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    default_config: dict[str, Any] = field(default_factory=dict)
    contributes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    hooks: list[PluginHook] = field(default_factory=list)
    lifecycle: dict[str, str] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)
    purge_data_on_remove: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name
        merged = dict(DEFAULT_LIMITS)
        merged.update(self.limits)
        self.limits = merged

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "version": self.version,
            "display_name": self.display_name,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "icon": self.icon,
            "tags": list(self.tags),
            "requires_devon": self.requires_devon,
            "dependencies": [
                {"name": d.name, "spec": d.spec} for d in self.dependencies
            ],
            "requirements": list(self.requirements),
            "entry_points": dict(self.entry_points),
            "lifecycle": dict(self.lifecycle),
            "hooks": [
                {"name": h.name, "entry": h.entry, "priority": h.priority}
                for h in self.hooks
            ],
            "contributes": {k: [dict(c) for c in v]
                            for k, v in self.contributes.items()},
            "config_schema": dict(self.config_schema),
            "default_config": dict(self.default_config),
            "permissions": list(self.permissions),
            "limits": dict(self.limits),
            "purge_data_on_remove": self.purge_data_on_remove,
        }
        # Unknown keys round-trip at top level (forward-compat), so a
        # manifest survives to_dict() → load_manifest() unchanged.
        out.update(self.extra)
        return out

    def contribution_summary(self) -> dict[str, list[str]]:
        """What this plugin adds, without loading any code."""
        return {
            "commands": [c.get("name", "") for c in
                         self.contributes.get("commands", [])],
            "tools": [t.get("name", "") for t in
                      self.contributes.get("tools", [])],
            "hooks": [h.name for h in self.hooks],
        }


# ── parsing helpers ─────────────────────────────────────────────────

def _req_str(data: dict[str, Any], key: str, default: str = "") -> str:
    val = data.get(key, default)
    if not isinstance(val, str):
        raise ManifestError(f"manifest field {key!r} must be a string")
    return val


def _opt_str_list(data: dict[str, Any], key: str) -> list[str]:
    val = data.get(key, [])
    if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
        raise ManifestError(f"manifest field {key!r} must be a list of strings")
    return list(val)


def _entry_target(target: Any, where: str) -> str:
    if not isinstance(target, str) or ":" not in target:
        raise ManifestError(
            f"{where} must look like 'module:attr', got {target!r}")
    module, _, attr = target.partition(":")
    if not module.strip() or not attr.strip():
        raise ManifestError(f"{where} has an empty module or attr: {target!r}")
    return target


def _parse_dependencies(raw: Any) -> list[PluginDependency]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ManifestError("dependencies must be a list")
    out: list[PluginDependency] = []
    for item in raw:
        if isinstance(item, str):
            m = re.match(r"^([a-z0-9][a-z0-9_-]{1,63})(.*)$", item.strip())
            if not m:
                raise ManifestError(
                    f"bad dependency {item!r}: want 'name' or "
                    f"'name>=1.2' (lowercase alnum/-/_ )")
            name, spec = m.group(1), m.group(2).strip()
            parse_version_spec(spec)  # fail fast on garbage specs
            out.append(PluginDependency(name=name, spec=spec))
        elif isinstance(item, dict):
            name = item.get("name", "")
            if not isinstance(name, str) or not _NAME_RE.match(name):
                raise ManifestError(
                    f"bad dependency name {name!r} in {item!r}")
            # "version" is the manifest spelling; "spec" is the to_dict()
            # round-trip spelling.
            spec = str(item.get("version", "") or item.get("spec", "") or "")
            parse_version_spec(spec)
            out.append(PluginDependency(name=name, spec=spec))
        else:
            raise ManifestError(
                f"bad dependency {item!r}: want 'name>=1.2' or "
                f"{{name, version}}")
    return out


def _parse_hooks(raw: Any) -> list[PluginHook]:
    if raw is None:
        return []
    # Accept the to_dict() list form too, so manifests round-trip.
    if isinstance(raw, list):
        items: list[tuple[str, Any]] = []
        for i, h in enumerate(raw):
            if not isinstance(h, dict) or "name" not in h:
                raise ManifestError(
                    f"hooks[{i}] must be a {{name, entry, priority?}} object")
            items.append((h["name"],
                          {"entry": h.get("entry", ""),
                           "priority": h.get("priority", 0)}))
    elif isinstance(raw, dict):
        items = list(raw.items())
    else:
        raise ManifestError("hooks must be a map of hook name to entry point")
    out: list[PluginHook] = []
    for name, target in items:
        if not isinstance(name, str) or not _HOOK_NAME_RE.match(name):
            raise ManifestError(
                f"bad hook name {name!r}: lowercase alnum/_ starting "
                f"with a letter")
        priority = 0
        if isinstance(target, dict):
            priority = target.get("priority", 0)
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise ManifestError(
                    f"hook {name!r}: priority must be an integer")
            target = target.get("entry", "")
        entry = _entry_target(target, f"hook {name!r}")
        out.append(PluginHook(name=name, entry=entry, priority=priority))
    return out


def _parse_lifecycle(raw: Any,
                     entry_points: dict[str, str]) -> dict[str, str]:
    lifecycle: dict[str, str] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            raise ManifestError("lifecycle must be a map")
        for key, target in raw.items():
            if key not in LIFECYCLE_HOOKS:
                raise ManifestError(
                    f"unknown lifecycle hook {key!r}; want one of "
                    f"{list(LIFECYCLE_HOOKS)}")
            lifecycle[key] = _entry_target(target, f"lifecycle {key!r}")
    # Plain entry_points keys act as lifecycle hooks too (back-compat);
    # explicit lifecycle entries are overridden by nothing — entry_points
    # wins because it's the older, more visible spelling.
    for key in LIFECYCLE_HOOKS:
        if key in entry_points:
            lifecycle[key] = _entry_target(
                entry_points[key], f"entry point {key!r}")
    return lifecycle


def _parse_contributes(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError("contributes must be a map")
    out: dict[str, list[dict[str, Any]]] = {}
    commands = raw.get("commands", [])
    if not isinstance(commands, list):
        raise ManifestError("contributes.commands must be a list")
    parsed_commands: list[dict[str, Any]] = []
    for cmd in commands:
        if not isinstance(cmd, dict) or not cmd.get("name"):
            raise ManifestError(
                f"contributes.commands entries need at least a name: "
                f"{cmd!r}")
        parsed_commands.append({
            "name": str(cmd["name"]),
            "title": str(cmd.get("title", "")),
            "description": str(cmd.get("description", "")),
        })
    if parsed_commands:
        out["commands"] = parsed_commands
    tools = raw.get("tools", [])
    if not isinstance(tools, list):
        raise ManifestError("contributes.tools must be a list")
    parsed_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            raise ManifestError(
                f"contributes.tools entries need at least a name: {tool!r}")
        schema = tool.get("schema", {})
        if not isinstance(schema, dict):
            raise ManifestError(
                f"contributes.tools[{tool.get('name')!r}].schema must be "
                f"an object")
        entry = tool.get("entry", "")
        if entry:
            _entry_target(entry,
                          f"contributes.tools[{tool.get('name')!r}].entry")
        parsed_tools.append({
            "name": str(tool["name"]),
            "description": str(tool.get("description", "")),
            "schema": schema,
            "entry": str(entry),
        })
    if parsed_tools:
        out["tools"] = parsed_tools
    return out


def _parse_limits(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError("limits must be a map")
    out: dict[str, float] = {}
    for key, val in raw.items():
        if key not in DEFAULT_LIMITS:
            raise ManifestError(
                f"unknown limit {key!r}; known: {sorted(DEFAULT_LIMITS)}")
        if isinstance(val, bool) or not isinstance(val, (int, float)) \
                or val <= 0:
            raise ManifestError(
                f"limit {key!r} must be a positive number, got {val!r}")
        out[key] = float(val)
    return out


_KNOWN_TOP_LEVEL = {
    "name", "version", "display_name", "description", "author", "license",
    "homepage", "icon", "tags", "requires_devon", "dependencies",
    "requirements", "entry_points", "lifecycle", "hooks", "contributes",
    "config_schema", "default_config", "permissions", "limits",
    "purge_data_on_remove",
}


def load_manifest(data: dict[str, Any]) -> PluginManifest:
    """Validate a manifest dict. Raises :exc:`ManifestError` on any problem."""
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    name = data.get("name", "")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ManifestError(
            f"bad plugin name {name!r}: 2-64 chars, lowercase alnum/-/_")
    version = data.get("version", "")
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise ManifestError(
            f"bad version {version!r}: expected semver like 1.0.0 "
            f"(prerelease/build suffixes like 1.0.0-beta or 1.0.0+build "
            f"are allowed)")
    entry_points = data.get("entry_points", {})
    if not isinstance(entry_points, dict) or not entry_points:
        raise ManifestError("manifest needs a non-empty entry_points map")
    for key, target in entry_points.items():
        _entry_target(target, f"entry point {key!r}")
    permissions = data.get("permissions", [])
    if not isinstance(permissions, list):
        raise ManifestError("permissions must be a list")
    unknown = [p for p in permissions if p not in KNOWN_PERMISSIONS]
    if unknown:
        raise ManifestError(
            f"unknown permissions {unknown}; known: {sorted(KNOWN_PERMISSIONS)}")
    requires_devon = _req_str(data, "requires_devon")
    if requires_devon:
        parse_version_spec(requires_devon)  # fail fast on garbage
    config_schema = data.get("config_schema", {})
    if not isinstance(config_schema, dict):
        raise ManifestError("config_schema must be an object")
    default_config = data.get("default_config", {})
    if not isinstance(default_config, dict):
        raise ManifestError("default_config must be an object")
    if config_schema:
        try:
            validate_against_schema(default_config, config_schema,
                                    "default_config")
        except ManifestError as exc:
            raise ManifestError(f"default_config invalid: {exc}") from exc
        default_config = apply_schema_defaults(default_config, config_schema)
    purge = data.get("purge_data_on_remove", False)
    if not isinstance(purge, bool):
        raise ManifestError("purge_data_on_remove must be a boolean")
    extra = {k: v for k, v in data.items() if k not in _KNOWN_TOP_LEVEL}
    return PluginManifest(
        name=name,
        version=str(version),
        display_name=_req_str(data, "display_name", name),
        description=_req_str(data, "description"),
        author=_req_str(data, "author"),
        license=_req_str(data, "license"),
        homepage=_req_str(data, "homepage"),
        icon=_req_str(data, "icon"),
        tags=_opt_str_list(data, "tags"),
        requires_devon=requires_devon,
        dependencies=_parse_dependencies(data.get("dependencies")),
        requirements=_opt_str_list(data, "requirements"),
        entry_points={str(k): str(v) for k, v in entry_points.items()},
        lifecycle=_parse_lifecycle(data.get("lifecycle"), entry_points),
        hooks=_parse_hooks(data.get("hooks")),
        contributes=_parse_contributes(data.get("contributes")),
        config_schema=config_schema,
        default_config=default_config,
        permissions=[str(p) for p in permissions],
        limits=_parse_limits(data.get("limits")),
        purge_data_on_remove=purge,
        extra=extra,
    )


def load_manifest_file(path: str | Path) -> PluginManifest:
    """Read and validate ``plugin.json`` from a plugin directory."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    if not p.is_file():
        raise ManifestError(f"no manifest at {p} (expected plugin.json)")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"could not read manifest {p}: {exc}") from exc
    return load_manifest(data)
