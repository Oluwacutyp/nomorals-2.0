"""Models and what each one can do (Craft-Libs port of ai-core
``model.rs``).

A provider offers models. A model advertises
:class:`CapabilitySpec`\\ s: for each capability, which inputs it
accepts and how many of each (so "takes a mask" and "takes at most
one reference" are data, not guesses), how many outputs it can
return, and whether it can stream, call tools or return structured
output. Requests are validated against the spec before anything
is sent.
"""

from __future__ import annotations

from .capability import CapabilityId
from .ids import ModelId

__all__ = [
    "CapabilitySpec",
    "Features",
    "InputRole",
    "ModelInfo",
    "Pricing",
]

_KNOWN_ROLES = ("prompt", "system", "source", "mask", "reference",
                "context")


class InputRole:
    """What an input is for. Edits need typed inputs: a mask is
    not just another image. An open set: the six known roles, plus
    anything else a provider reports."""

    def __init__(self, name: str) -> None:
        self._name = name

    @classmethod
    def parse(cls, text: str) -> InputRole:
        return cls(str(text))

    def as_str(self) -> str:
        return self._name

    @property
    def value(self) -> str:
        return self._name

    def is_known(self) -> bool:
        return self._name in _KNOWN_ROLES

    def __eq__(self, other) -> bool:
        return isinstance(other, InputRole) and self._name == other._name

    def __hash__(self):
        return hash(self._name)

    def __repr__(self) -> str:
        return f"InputRole({self._name!r})"


# The known roles, as module constants (upstream's enum variants).
InputRole.PROMPT = InputRole("prompt")
InputRole.SYSTEM = InputRole("system")
InputRole.SOURCE = InputRole("source")
InputRole.MASK = InputRole("mask")
InputRole.REFERENCE = InputRole("reference")
InputRole.CONTEXT = InputRole("context")


class Features:
    """Optional abilities beyond the basic request and result."""

    def __init__(self, streaming: bool = False, tools: bool = False,
                 structured_output: bool = False) -> None:
        self.streaming = streaming
        self.tools = tools
        self.structured_output = structured_output

    def to_dict(self) -> dict:
        return {"streaming": self.streaming, "tools": self.tools,
                "structured_output": self.structured_output}

    @classmethod
    def from_dict(cls, data: dict) -> Features:
        return cls(bool(data.get("streaming", False)),
                   bool(data.get("tools", False)),
                   bool(data.get("structured_output", False)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Features)
                and self.to_dict() == other.to_dict())


class CapabilitySpec:
    """What one model can do for one capability."""

    def __init__(self, capability: CapabilityId,
                 accepts: dict | None = None,
                 requires: list | None = None,
                 max_outputs: int = 1,
                 features: Features | None = None) -> None:
        self.capability = capability
        self.accepts: dict[InputRole, int] = dict(accepts or {})
        self.requires: list[InputRole] = list(requires or [])
        self.max_outputs = max_outputs
        self.features = features or Features()

    @classmethod
    def new(cls, capability: CapabilityId) -> CapabilitySpec:
        return cls(capability)

    def accepting(self, role: InputRole, max_count: int) -> CapabilitySpec:
        self.accepts[role] = max_count
        return self

    def requiring(self, role: InputRole) -> CapabilitySpec:
        if role not in self.requires:
            self.requires.append(role)
        return self

    def with_max_outputs(self, n: int) -> CapabilitySpec:
        """Set the output cap. (Upstream calls this
        ``max_outputs``; that name is the attribute here.)"""
        self.max_outputs = n
        return self

    def with_features(self, features: Features) -> CapabilitySpec:
        self.features = features
        return self

    def to_dict(self) -> dict:
        return {"capability": self.capability.as_str(),
                "accepts": {r.as_str(): n for r, n in self.accepts.items()},
                "requires": [r.as_str() for r in self.requires],
                "max_outputs": self.max_outputs,
                "features": self.features.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> CapabilitySpec:
        return cls(CapabilityId.parse(data["capability"]),
                   {InputRole.parse(r): n
                    for r, n in data.get("accepts", {}).items()},
                   [InputRole.parse(r) for r in data.get("requires", [])],
                   int(data.get("max_outputs", 1)),
                   Features.from_dict(data.get("features", {})))

    def __eq__(self, other) -> bool:
        return (isinstance(other, CapabilitySpec)
                and self.to_dict() == other.to_dict())


class Pricing:
    """What one model costs, in millionths of a US dollar. Every
    field is optional: providers report some, one or none, and a
    missing price is never filled in by guessing."""

    def __init__(self, input_micros_per_mtok: int | None = None,
                 output_micros_per_mtok: int | None = None,
                 per_image_micros: int | None = None) -> None:
        self.input_micros_per_mtok = input_micros_per_mtok
        self.output_micros_per_mtok = output_micros_per_mtok
        self.per_image_micros = per_image_micros

    def to_dict(self) -> dict:
        data: dict = {}
        if self.input_micros_per_mtok is not None:
            data["input_micros_per_mtok"] = self.input_micros_per_mtok
        if self.output_micros_per_mtok is not None:
            data["output_micros_per_mtok"] = self.output_micros_per_mtok
        if self.per_image_micros is not None:
            data["per_image_micros"] = self.per_image_micros
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Pricing:
        return cls(data.get("input_micros_per_mtok"),
                   data.get("output_micros_per_mtok"),
                   data.get("per_image_micros"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Pricing)
                and self.to_dict() == other.to_dict())


class ModelInfo:
    """One model as offered by one connection."""

    def __init__(self, model_id: ModelId, label: str,
                 capabilities: list | None = None,
                 pricing: Pricing | None = None,
                 released: int | None = None,
                 tags: list | None = None,
                 local_file: str | None = None) -> None:
        self.id = model_id if isinstance(model_id, ModelId) \
            else ModelId.new(model_id)
        self.label = label
        self.capabilities: list[CapabilitySpec] = list(capabilities or [])
        self.pricing = pricing
        self.released = released
        self.tags: list[str] = list(tags or [])
        self.local_file = local_file

    @classmethod
    def new(cls, model_id: ModelId, label: str) -> ModelInfo:
        return cls(model_id, label)

    def with_spec(self, spec: CapabilitySpec) -> ModelInfo:
        """Attach a capability spec. (Upstream calls this
        ``with``; that is a keyword in Python.)"""
        self.capabilities.append(spec)
        return self

    def spec_for(self, capability: CapabilityId) -> CapabilitySpec | None:
        for spec in self.capabilities:
            if spec.capability == capability:
                return spec
        return None

    def to_dict(self) -> dict:
        data: dict = {"id": self.id.as_str(), "label": self.label,
                      "capabilities": [s.to_dict()
                                       for s in self.capabilities]}
        if self.pricing is not None:
            data["pricing"] = self.pricing.to_dict()
        if self.released is not None:
            data["released"] = self.released
        if self.tags:
            data["tags"] = list(self.tags)
        if self.local_file is not None:
            data["local_file"] = self.local_file
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ModelInfo:
        return cls(ModelId.new(data["id"]), str(data["label"]),
                   [CapabilitySpec.from_dict(s)
                    for s in data.get("capabilities", [])],
                   Pricing.from_dict(data["pricing"])
                   if data.get("pricing") is not None else None,
                   data.get("released"),
                   list(data.get("tags", [])),
                   data.get("local_file"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, ModelInfo)
                and self.to_dict() == other.to_dict())

    def __repr__(self) -> str:
        return f"ModelInfo({self.id.as_str()!r}, {self.label!r})"
