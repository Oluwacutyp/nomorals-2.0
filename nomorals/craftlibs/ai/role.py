"""Roles (Craft-Libs port of ai-core ``role.rs``).

An app declares a role — one feature's AI job: its capability and
the inputs the feature can supply. A :class:`Binding` chooses a
connection and a model for that role. :func:`resolve` walks the
layers (app overrides first, then the launcher defaults) and the
first binding wins; a broken override is an error, never a silent
fall-through to the default.
"""

from __future__ import annotations

from .capability import CapabilityId
from .ids import ConnectionId, ModelId, RoleId
from .model import InputRole

__all__ = ["Binding", "ResolveError", "Resolved", "Role", "eligible",
           "fits", "resolve"]


class ResolveError(ValueError):
    """Why a role did not resolve. ``kind`` is one of
    ``unbound``, ``connection_missing``, ``connection_disabled``,
    ``model_missing``, ``capability_mismatch``,
    ``input_unsupported``, ``input_required``, ``tool_mismatch``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class Role:
    """One feature's AI role."""

    def __init__(self, id, label: str, capability,
                 inputs: list | None = None) -> None:
        self.id = id if isinstance(id, RoleId) \
            else RoleId.new(id)
        self.label = label
        self.capability = capability if isinstance(capability,
                                                   CapabilityId) \
            else CapabilityId.parse(capability)
        self.inputs: list[InputRole] = [
            i if isinstance(i, InputRole) else InputRole.parse(i)
            for i in (inputs or [])]

    def to_dict(self) -> dict:
        return {"id": self.id.as_str(), "label": self.label,
                "capability": self.capability.as_str(),
                "inputs": [i.as_str() for i in self.inputs]}

    @classmethod
    def from_dict(cls, data: dict) -> Role:
        return cls(RoleId.new(data["id"]), str(data["label"]),
                   CapabilityId.parse(data["capability"]),
                   [InputRole.parse(i) for i in data.get("inputs", [])])

    def __eq__(self, other) -> bool:
        return (isinstance(other, Role)
                and self.to_dict() == other.to_dict())


class Binding:
    """A role's chosen connection and model."""

    def __init__(self, role, connection, model) -> None:
        self.role = role if isinstance(role, RoleId) else RoleId.new(role)
        self.connection = connection if isinstance(connection,
                                                 ConnectionId) \
            else ConnectionId.new(connection)
        self.model = model if isinstance(model, ModelId) \
            else ModelId.new(model)

    def to_dict(self) -> dict:
        return {"role": self.role.as_str(),
                "connection": self.connection.as_str(),
                "model": self.model.as_str()}

    @classmethod
    def from_dict(cls, data: dict) -> Binding:
        return cls(RoleId.new(data["role"]),
                   ConnectionId.new(data["connection"]),
                   ModelId.new(data["model"]))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Binding)
                and self.to_dict() == other.to_dict())


class Resolved:
    """A role, resolved to a live connection, model and spec."""

    def __init__(self, connection, model, spec) -> None:
        self.connection = connection
        self.model = model
        self.spec = spec


def fits(role: Role, model):
    """The model's spec for this role's capability, if the model
    can take every input the role supplies."""
    spec = model.spec_for(role.capability)
    if spec is None:
        return None
    for role_input in role.inputs:
        if role_input not in spec.accepts:
            return None
    return spec


def _resolve_binding(role: Role, binding: Binding, connections):
    connection = next((c for c in connections
                       if c.id == binding.connection), None)
    if connection is None:
        raise ResolveError(
            "connection_missing",
            f"connection {binding.connection.as_str()} is not configured")
    if not connection.enabled:
        raise ResolveError(
            "connection_disabled",
            f"connection {binding.connection.as_str()} is disabled")
    model = next((m for m in connection.models
                  if m.id == binding.model), None)
    if model is None:
        raise ResolveError(
            "model_missing",
            f"model {binding.model.as_str()} is not offered by "
            f"connection {binding.connection.as_str()}")
    spec = model.spec_for(role.capability)
    if spec is None:
        raise ResolveError(
            "capability_mismatch",
            f"model {binding.model.as_str()} cannot do "
            f"{role.capability.as_str()}")
    for role_input in role.inputs:
        if role_input not in spec.accepts:
            raise ResolveError(
                "input_unsupported",
                f"model {binding.model.as_str()} does not take a "
                f"'{role_input.as_str()}' input")
    for required in spec.requires:
        if required not in role.inputs:
            raise ResolveError(
                "input_required",
                f"model {binding.model.as_str()} requires a "
                f"'{required.as_str()}' input that this tool does not "
                "provide")
    return Resolved(connection, model, spec)


def resolve(role: Role, layers, connections) -> Resolved:
    """Resolve a role through the layers (app overrides first).
    The first binding for this role wins; a broken override is
    an error, never a fall-through."""
    for layer in layers:
        for binding in layer:
            if binding.role == role.id:
                return _resolve_binding(role, binding, connections)
    raise ResolveError(
        "unbound",
        f"no model is chosen for role '{role.id.as_str()}'")


def eligible(role: Role, connections) -> list[Resolved]:
    """Every live (connection, model) pair that can run this role,
    in order. Disabled connections and models that cannot take the
    role's inputs are left out."""
    found: list[Resolved] = []
    for connection in connections:
        if not connection.enabled:
            continue
        for model in connection.models:
            spec = fits(role, model)
            if spec is not None:
                found.append(Resolved(connection, model, spec))
    return found
