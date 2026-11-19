"""Requests (Craft-Libs port of ai-core ``request.rs``).

A request names a capability, carries typed inputs, provider-neutral
options, an output count and a streaming flag. :meth:`Request.validate`
checks it against the model's :class:`CapabilitySpec` before
anything is sent, in this order: capability mismatch, payload
limits, unsupported input, too many of one input, missing required
input, output count, streaming needs the feature, params size.
"""

from __future__ import annotations

import json

from .capability import CapabilityId
from .error import AiError
from .model import InputRole
from .payload import Limits, Payload

__all__ = ["MAX_PARAMS_BYTES", "Request"]

MAX_PARAMS_BYTES = 64 * 1024


class Request:
    """One job to run, before it is sent anywhere."""

    def __init__(self, capability: CapabilityId,
                 inputs: list | None = None, params: dict | None = None,
                 outputs: int = 1, stream: bool = False) -> None:
        self.capability = capability
        self.inputs: list[tuple[InputRole, Payload]] = list(inputs or [])
        self.params: dict = dict(params or {})
        self.outputs = outputs
        self.stream = stream

    @classmethod
    def new(cls, capability: CapabilityId) -> Request:
        return cls(capability)

    def with_input(self, role: InputRole, payload: Payload) -> Request:
        self.inputs.append((role, payload))
        return self

    def with_param(self, name: str, value) -> Request:
        self.params[name] = value
        return self

    def with_outputs(self, n: int) -> Request:
        self.outputs = n
        return self

    def streaming(self) -> Request:
        self.stream = True
        return self

    def validate(self, spec, limits: Limits) -> None:
        """Check this request against a model's capability spec.
        Raises :class:`AiError` on the first problem."""
        if self.capability != spec.capability:
            raise AiError.invalid(
                f"request is for {self.capability.as_str()}, the model "
                f"offers {spec.capability.as_str()}")
        for _role, payload in self.inputs:
            payload.check(limits)
        counts: dict[InputRole, int] = {}
        for role, _payload in self.inputs:
            counts[role] = counts.get(role, 0) + 1
        for role in counts:
            if role not in spec.accepts:
                raise AiError.unsupported(
                    f"this model does not take a '{role.as_str()}' input")
        for role, count in counts.items():
            if count > spec.accepts[role]:
                raise AiError.unsupported(
                    f"this model takes at most {spec.accepts[role]} "
                    f"'{role.as_str()}' input(s), got {count}")
        present = set(counts)
        for role in spec.requires:
            if role not in present:
                raise AiError.invalid(
                    f"a '{role.as_str()}' input is required")
        if self.outputs < 1 or self.outputs > spec.max_outputs:
            raise AiError.invalid(
                f"outputs must be 1 to {spec.max_outputs}, "
                f"got {self.outputs}")
        if self.stream and not spec.features.streaming:
            raise AiError.unsupported(
                "this model cannot stream")
        params_len = len(json.dumps(self.params,
                                    separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8"))
        if params_len > MAX_PARAMS_BYTES:
            raise AiError.invalid(
                f"params are over {MAX_PARAMS_BYTES} bytes")

    def to_dict(self) -> dict:
        return {"capability": self.capability.as_str(),
                "inputs": [{"role": r.as_str(), "payload": p.to_dict()}
                           for r, p in self.inputs],
                "params": self.params,
                "outputs": self.outputs,
                "stream": self.stream}

    @classmethod
    def from_dict(cls, data: dict) -> Request:
        req = cls(CapabilityId.parse(data["capability"]))
        for item in data.get("inputs", []):
            req.inputs.append((InputRole.parse(item["role"]),
                               Payload.from_dict(item["payload"])))
        req.params = dict(data.get("params", {}))
        req.outputs = int(data.get("outputs", 1))
        req.stream = bool(data.get("stream", False))
        return req

    def __eq__(self, other) -> bool:
        return (isinstance(other, Request)
                and self.to_dict() == other.to_dict())
