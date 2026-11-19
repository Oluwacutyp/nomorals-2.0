"""Spec data types (Craft-Libs port of ai-core ``spec.rs``).

The types live in this leaf module so the checker can import
them without a circular import through the package ``__init__``.
"""

from __future__ import annotations

from ..capability import CapabilityId
from ..ids import ModelId
from ..model import CapabilitySpec, InputRole, ModelInfo


class SpecAuth:
    """How the key is sent. The key itself comes from the
    connection's secret reference — never from the spec."""

    def __init__(self, kind: str, name: str | None = None) -> None:
        self.kind = kind
        self.name = name

    @classmethod
    def bearer(cls) -> SpecAuth:
        return cls("bearer")

    @classmethod
    def header(cls, name: str) -> SpecAuth:
        return cls("header", name=name)

    @classmethod
    def none(cls) -> SpecAuth:
        return cls("none")

    def to_dict(self) -> dict:
        if self.kind == "header":
            return {"kind": "header", "name": self.name}
        return {"kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict) -> SpecAuth:
        if data.get("kind") == "header":
            return cls.header(str(data.get("name", "")))
        return cls(str(data.get("kind", "bearer")))

    def __eq__(self, other) -> bool:
        return (isinstance(other, SpecAuth)
                and self.to_dict() == other.to_dict())


class Method:
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"


class Call:
    """One call. ``path`` is joined to ``base``; ``query`` and
    ``body`` may use the allowed substitutions."""

    def __init__(self, method: str, path: str, query: dict | None = None,
                 body=None) -> None:
        self.method = method
        self.path = path
        self.query: dict = dict(query or {})
        self.body = body

    def to_dict(self) -> dict:
        data: dict = {"method": self.method, "path": self.path}
        if self.query:
            data["query"] = self.query
        if self.body is not None:
            data["body"] = self.body
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Call:
        return cls(str(data["method"]), str(data["path"]),
                   dict(data.get("query", {})), data.get("body"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Call)
                and self.to_dict() == other.to_dict())


class ExtractAs:
    BASE64 = "base64"
    URL = "url"
    TEXT = "text"


class Extract:
    """What a result is, and how to read it from the reply."""

    def __init__(self, at: str, kind: str, media_type: str | None = None
                 ) -> None:
        self.at = at
        self.kind = kind
        self.media_type = media_type

    def to_dict(self) -> dict:
        data: dict = {"at": self.at, "as": self.kind}
        if self.media_type is not None:
            data["media_type"] = self.media_type
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Extract:
        return cls(str(data["at"]), str(data["as"]),
                   data.get("media_type"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Extract)
                and self.to_dict() == other.to_dict())


class StateWord:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Flow:
    """One call whose reply holds the result, or submit-then-poll."""

    def __init__(self, kind: str, call: Call | None = None,
                 result: Extract | None = None, submit=None,
                 poll=None, cancel: Call | None = None) -> None:
        self.kind = kind
        self.call = call
        self.result = result
        self.submit = submit
        self.poll = poll
        self.cancel = cancel

    def to_dict(self) -> dict:
        if self.kind == "request_response":
            return {"kind": "request_response",
                    "call": self.call.to_dict(),
                    "result": self.result.to_dict()}
        data: dict = {"kind": "queue",
                      "submit": self.submit,
                      "poll": self.poll}
        if self.cancel is not None:
            data["cancel"] = self.cancel.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Flow:
        if data.get("kind") == "request_response":
            return cls("request_response",
                       call=Call.from_dict(data["call"]),
                       result=Extract.from_dict(data["result"]))
        return cls("queue", submit=data.get("submit"),
                   poll=data.get("poll"),
                   cancel=Call.from_dict(data["cancel"])
                   if data.get("cancel") is not None else None)

    def __eq__(self, other) -> bool:
        return (isinstance(other, Flow)
                and self.to_dict() == other.to_dict())


class SpecModel:
    """A model the spec offers. Nothing is offered that is not
    declared here."""

    def __init__(self, model_id: str, label: str, capability: str,
                 accepts: dict, requires: list | None = None,
                 max_outputs: int = 1) -> None:
        self.id = model_id
        self.label = label
        self.capability = capability
        self.accepts = dict(accepts)
        self.requires = list(requires or [])
        self.max_outputs = max_outputs

    def to_dict(self) -> dict:
        data: dict = {"id": self.id, "label": self.label,
                      "capability": self.capability,
                      "accepts": self.accepts,
                      "requires": self.requires,
                      "max_outputs": self.max_outputs}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SpecModel:
        return cls(str(data["id"]), str(data["label"]),
                   str(data["capability"]), dict(data.get("accepts", {})),
                   list(data.get("requires", [])),
                   int(data.get("max_outputs", 1)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, SpecModel)
                and self.to_dict() == other.to_dict())


class SpecLimits:
    """Bounds on one job."""

    def __init__(self, max_polls: int = 600, max_seconds: int = 600
                 ) -> None:
        self.max_polls = max_polls
        self.max_seconds = max_seconds

    def to_dict(self) -> dict:
        return {"max_polls": self.max_polls,
                "max_seconds": self.max_seconds}

    @classmethod
    def from_dict(cls, data: dict) -> SpecLimits:
        return cls(int(data.get("max_polls", 600)),
                   int(data.get("max_seconds", 600)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, SpecLimits)
                and self.to_dict() == other.to_dict())


class Spec:
    """A checked spec. Build one only through :meth:`Spec.parse`."""

    def __init__(self, spec_version: int, spec_id: str, label: str,
                 base: str, auth: SpecAuth, models: list, flow: Flow,
                 probe: Call | None = None,
                 limits: SpecLimits | None = None) -> None:
        self.spec_version = spec_version
        self.id = spec_id
        self.label = label
        self.base = base
        self.auth = auth
        self.probe = probe
        self.models = models
        self.flow = flow
        self.limits = limits or SpecLimits()

    @classmethod
    def parse(cls, text: str) -> Spec:
        """Read and check a spec. Nothing is returned that has
        not passed every rule."""
        from .check import parse as _parse

        return _parse(text)

    def model_infos(self) -> list[ModelInfo]:
        """The models the spec offers, as the rest of the layer
        sees them."""
        infos = []
        for model in self.models:
            try:
                capability = CapabilitySpec.new(
                    CapabilityId.parse(model.capability))
                capability = capability.with_max_outputs(model.max_outputs)
                for role, count in model.accepts.items():
                    capability = capability.accepting(
                        InputRole.parse(role), int(count))
                for role in model.requires:
                    capability = capability.requiring(InputRole.parse(role))
                infos.append(ModelInfo.new(ModelId.new(model.id),
                                           model.label)
                               .with_spec(capability))
            except ValueError:
                continue
        return infos

    def to_dict(self) -> dict:
        data: dict = {"spec_version": self.spec_version, "id": self.id,
                      "label": self.label, "base": self.base,
                      "auth": self.auth.to_dict(),
                      "models": [m.to_dict() for m in self.models],
                      "flow": self.flow.to_dict(),
                      "limits": self.limits.to_dict()}
        if self.probe is not None:
            data["probe"] = self.probe.to_dict()
        return data

    def __eq__(self, other) -> bool:
        return (isinstance(other, Spec)
                and self.to_dict() == other.to_dict())
