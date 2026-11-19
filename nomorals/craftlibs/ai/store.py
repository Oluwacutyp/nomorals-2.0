"""The store file (Craft-Libs port of ai-core ``store.rs``).

One JSON file holds the connections, the launcher defaults, the
per-app overrides and the custom tools. The version is read first,
so a newer file is refused even if its other fields no longer fit
this build's types. Serialized form holds secret *references*,
never secret values.
"""

from __future__ import annotations

import json

from .connection import Connection, ConnectionError
from .ids import AppId, RoleId
from .model import InputRole
from .role import Binding, ResolveError, Role, resolve

__all__ = ["MAX_STORE_BYTES", "SCHEMA_VERSION", "StoreError",
           "StoreFile"]

SCHEMA_VERSION = 1
MAX_STORE_BYTES = 1024 * 1024
MAX_CUSTOM_TOOLS = 64


class StoreError(ValueError):
    """Why a store file was refused. ``kind`` is one of
    ``too_large``, ``syntax``, ``too_new``, ``bad_version``,
    ``duplicate_connection``, ``duplicate_tool``, ``too_many_tools``,
    ``invalid_tool``, ``invalid``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class CustomTool:
    """A tool an app registered by id: one suite-wide contract
    (an app may localise its label, but cannot change the
    capability or input set while using the same id)."""

    def __init__(self, tool_id, label: str, capability,
                 inputs: list) -> None:
        from .capability import CapabilityId
        self.id = tool_id if isinstance(tool_id, RoleId) \
            else RoleId.new(tool_id)
        self.label = label
        self.capability = capability if isinstance(capability,
                                                   CapabilityId) \
            else CapabilityId.parse(capability)
        self.inputs = [i if isinstance(i, InputRole)
                       else InputRole.parse(i) for i in inputs]

    def to_dict(self) -> dict:
        return {"id": self.id.as_str(), "label": self.label,
                "capability": self.capability.as_str(),
                "inputs": [i.as_str() for i in self.inputs]}

    @classmethod
    def from_dict(cls, data: dict) -> CustomTool:
        return cls(RoleId.new(data["id"]), str(data["label"]),
                   data["capability"],
                   [InputRole.parse(i) for i in data.get("inputs", [])])

    def __eq__(self, other) -> bool:
        return (isinstance(other, CustomTool)
                and self.to_dict() == other.to_dict())


class StoreFile:
    """The parsed store."""

    def __init__(self, connections: list | None = None,
                 defaults: list | None = None,
                 apps: dict | None = None,
                 custom_tools: list | None = None,
                 schema_version: int = SCHEMA_VERSION) -> None:
        self.schema_version = schema_version
        self.connections: list[Connection] = list(connections or [])
        self.defaults: list[Binding] = list(defaults or [])
        self.apps: dict = dict(apps or {})
        self.custom_tools: list[CustomTool] = list(custom_tools or [])

    @classmethod
    def parse(cls, text: str) -> StoreFile:
        """Parse and validate a store file. Never panics on bad
        input: every refusal is a :class:`StoreError`."""
        if len(text.encode("utf-8")) > MAX_STORE_BYTES:
            raise StoreError(
                "too_large",
                f"store file is {len(text.encode('utf-8'))} bytes "
                f"(max {MAX_STORE_BYTES})")
        try:
            head = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StoreError("syntax", f"not JSON: {exc}") from exc
        if not isinstance(head, dict):
            raise StoreError("syntax", "a store file is a JSON object")
        version = head.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise StoreError("bad_version",
                             "a store file names its schema_version")
        if version < 1:
            raise StoreError(
                "bad_version",
                f"schema_version {version} is not read")
        if version > SCHEMA_VERSION:
            raise StoreError(
                "too_new",
                f"schema_version {version} is newer than the supported "
                f"{SCHEMA_VERSION}")
        try:
            store = cls.from_dict(head)
        except (ValueError, KeyError, TypeError) as exc:
            raise StoreError("syntax", f"not a store file: {exc}") from exc
        store.validate()
        return store

    @classmethod
    def from_dict(cls, data: dict) -> StoreFile:
        apps = {}
        for app_id, settings in (data.get("apps") or {}).items():
            overrides = [Binding.from_dict(b)
                         for b in (settings or {}).get("overrides", [])]
            apps[str(app_id)] = {"overrides": overrides}
        return cls(
            [Connection.from_dict(c)
             for c in data.get("connections", [])],
            [Binding.from_dict(b) for b in data.get("defaults", [])],
            apps,
            [CustomTool.from_dict(t)
             for t in data.get("custom_tools", [])],
            int(data.get("schema_version", SCHEMA_VERSION)))

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version,
                "connections": [c.to_dict() for c in self.connections],
                "defaults": [b.to_dict() for b in self.defaults],
                "apps": {app_id: {"overrides": [b.to_dict() for b in
                                                 settings["overrides"]]}
                         for app_id, settings in self.apps.items()},
                "custom_tools": [t.to_dict() for t in self.custom_tools]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def validate(self) -> None:
        """Check every connection and custom tool before anything
        is written."""
        if len(self.custom_tools) > MAX_CUSTOM_TOOLS:
            raise StoreError(
                "too_many_tools",
                f"{len(self.custom_tools)} custom tools "
                f"(max {MAX_CUSTOM_TOOLS})")
        for i, tool in enumerate(self.custom_tools):
            if any(other.id == tool.id
                   for other in self.custom_tools[i + 1:]):
                raise StoreError(
                    "duplicate_tool",
                    f"custom tool {tool.id.as_str()} is listed twice")
            if (not tool.label.strip() or len(tool.label) > 120
                    or any(ord(c) < 0x20 for c in tool.label)
                    or not tool.inputs or len(tool.inputs) > 8
                    or len(set(tool.inputs)) != len(tool.inputs)):
                raise StoreError(
                    "invalid_tool",
                    f"custom tool {tool.id.as_str()} is not valid")
        for i, conn in enumerate(self.connections):
            if any(other.id == conn.id
                   for other in self.connections[i + 1:]):
                raise StoreError(
                    "duplicate_connection",
                    f"connection {conn.id.as_str()} is listed twice")
            try:
                conn.validate()
            except ConnectionError as exc:
                raise StoreError(
                    "invalid",
                    f"connection {conn.id.as_str()}: {exc}") from exc

    def check_tool(self, role: Role) -> None:
        """A custom id is one suite-wide contract."""
        saved = next((t for t in self.custom_tools if t.id == role.id),
                     None)
        if saved is not None and (
                saved.capability != role.capability
                or len(saved.inputs) != len(role.inputs)
                or not all(i in role.inputs for i in saved.inputs)):
            raise ResolveError(
                "tool_mismatch",
                f"custom tool {role.id.as_str()} does not match the "
                "saved contract")

    def resolve(self, app, role: Role):
        """Resolve an app's role through its overrides, then the
        launcher defaults."""
        self.check_tool(role)
        app_id = app if isinstance(app, str) else AppId.new(app).as_str()
        overrides = self.apps.get(app_id, {}).get("overrides", [])
        return resolve(role, [overrides, self.defaults], self.connections)

    def dangling_bindings(self) -> list[Binding]:
        """Bindings (defaults and app overrides) whose connection
        or model is not in the store."""
        offered = {(c.id, m.id) for c in self.connections
                   for m in c.models}
        dangling = []
        bindings = list(self.defaults)
        for settings in self.apps.values():
            bindings.extend(settings.get("overrides", []))
        for binding in bindings:
            if (binding.connection, binding.model) not in offered:
                dangling.append(binding)
        return dangling
