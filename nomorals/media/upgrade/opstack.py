"""Tier 0 — Non-destructive op stack wired to EditSpec.

Every edit is an :class:`EditOp` (name + deterministic params + seed).
The stack is the history: ``undo`` pops, ``branch`` copies the stack,
``explain`` shows the stack. ``apply_stack_to_spec`` replays the ops
onto a base EditSpec deterministically — same stack + same base =
same spec dict, which is what makes renders reproducible.

Supported op names (all pure spec transforms, no rendering here):
crop, grade (brightness/contrast/saturation), text (append TextLayer),
trim (shorten clip ends), transition (set transition), reorder
(clip permutation), speed (clip playback factor via end scaling),
mask (attach mask descriptor to a clip effect), regional
(attach a regional prompt to a clip effect), camera (attach a camera
move to a clip effect). Unknown ops raise — no silent no-op edits.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_OPS = ("crop", "grade", "text", "trim", "transition",
                 "reorder", "speed", "mask", "regional", "camera")


@dataclass(frozen=True)
class EditOp:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_OPS:
            raise ValueError(f"unknown op {self.name!r}; supported: {SUPPORTED_OPS}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params), "seed": self.seed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditOp":
        return cls(name=data["name"], params=dict(data.get("params", {})),
                   seed=int(data.get("seed", 0)))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class OpStack:
    """Ordered, branchable, explainable edit history."""

    def __init__(self, ops: list[EditOp] | None = None) -> None:
        self._ops: list[EditOp] = list(ops or [])

    def push(self, op: EditOp) -> "OpStack":
        self._ops.append(op)
        return self

    def undo(self) -> EditOp | None:
        return self._ops.pop() if self._ops else None

    def branch(self) -> "OpStack":
        return OpStack(ops=list(self._ops))

    def explain(self) -> list[str]:
        lines = []
        for i, op in enumerate(self._ops):
            params = ", ".join(f"{k}={v!r}" for k, v in sorted(op.params.items()))
            lines.append(f"{i}: {op.name}({params}) seed={op.seed}")
        return lines

    def __len__(self) -> int:
        return len(self._ops)

    def __iter__(self):
        return iter(list(self._ops))

    def to_dict(self) -> dict[str, Any]:
        return {"ops": [op.to_dict() for op in self._ops]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpStack":
        return cls(ops=[EditOp.from_dict(o) for o in data.get("ops", [])])

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _effect_dict(name: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "params": dict(params)}


def apply_stack_to_spec(base_spec: Any, stack: OpStack) -> Any:
    """Replay ops onto a copy of ``base_spec`` (an EditSpec).

    Uses only EditSpec's public surface (to_dict / from_dict plus the
    Clip/Effect dataclasses) so the stack stays decoupled from the
    renderer's internals. The base spec is never mutated.
    """
    from dataclasses import replace as _replace

    from ..edit_engine.timeline import Effect

    spec = copy.deepcopy(base_spec)
    for op in stack:
        p = op.params
        if op.name == "crop":
            idx = int(p.get("clip", 0))
            spec.clips[idx].effects = tuple(list(spec.clips[idx].effects) + [
                Effect(name="crop", params={k: p[k] for k in ("x", "y", "w", "h") if k in p})])
        elif op.name == "grade":
            for clip in spec.clips:
                clip.effects = tuple(list(clip.effects) + [
                    Effect(name="grade", params={k: p[k] for k in
                            ("brightness", "contrast", "saturation", "temperature") if k in p})])
        elif op.name == "text":
            from ..edit_engine.text import TextLayer
            spec.text_layers = list(spec.text_layers) + [
                TextLayer(text=str(p.get("text", "")),
                          start=float(p.get("start", 0.0)),
                          end=float(p.get("end", 0.0)) or None)]
        elif op.name == "trim":
            idx = int(p.get("clip", 0))
            clip = spec.clips[idx]
            if clip.end is not None and "end" in p:
                spec.clips[idx] = _replace(clip, end=float(p["end"]))
        elif op.name == "transition":
            spec.transition = p.get("kind", spec.transition)
        elif op.name == "reorder":
            order = [int(i) for i in p.get("order", [])]
            if sorted(order) != list(range(len(spec.clips))):
                raise ValueError("reorder order must be a permutation of clip indices")
            spec.clips = [spec.clips[i] for i in order]
        elif op.name == "speed":
            idx = int(p.get("clip", 0))
            factor = float(p.get("factor", 1.0))
            if factor <= 0:
                raise ValueError("speed factor must be > 0")
            clip = spec.clips[idx]
            if clip.end is not None:
                spec.clips[idx] = _replace(clip, end=clip.end / factor)
        elif op.name in ("mask", "regional", "camera"):
            idx = int(p.get("clip", 0))
            clip = spec.clips[idx]
            clip.effects = tuple(list(clip.effects) + [Effect(name=op.name, params=dict(p))])
    return spec
