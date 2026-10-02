"""Executable skill manifests.

A *skill* in this package is an executable unit: a named, versioned chain
of tool calls with declared input/output schemas and explicit data wiring
between steps.  This is the L4 execution substrate — distinct from the
knowledge-skill library in ``nomorals/agents/skills.py`` (L5), whose
skills are strategy texts and playbooks, not things you run.

A manifest is a plain dict, so skills can be authored as JSON files::

    {
      "name": "summarize_then_save",
      "version": "1.0.0",
      "tools": ["summarizer", "file_write"],
      "input_schema": {"text": "str", "path": "str"},
      "output_schema": {"path": "str", "bytes": "int?"},
      "wiring": [
        {},
        {"path": "$input.path", "content": "$0.summary"}
      ],
      "owner": "devon",
      "description": "Summarize text and save it to a file."
    }

Wiring expressions (strings starting with ``$``):
  * ``$input.<dotted.path>`` — from the skill run's input
  * ``$<n>.<dotted.path>``   — from step *n*'s output (n < current step)
  * ``$last.<dotted.path>``   — from the previous step's output
Anything else is a literal passed through unchanged.

Threading rule: a step with no wiring entry declared receives the run
input unchanged; a step *with* wiring receives exactly its resolved
mapping (no silent merge).

Schema entries are either a type-name string (``"str"``, ``"int?"`` —
trailing ``?`` marks the key optional) or a dict
``{"type": "str", "required": false, "default": ...}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ValidationError

__all__ = [
    "SkillManifest",
    "ManifestError",
    "WiringError",
    "validate_schema",
    "resolve_expression",
    "SEMVER_RE",
    "NAME_RE",
    "TYPE_NAMES",
]

#: "semver-ish": 1.0, 1.0.0, 2.1.0-rc1 are all fine.
SEMVER_RE = re.compile(r"^\d+\.\d+(\.\d+)?(-[0-9A-Za-z.-]+)?$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TYPE_NAMES = {"str", "int", "float", "bool", "list", "dict", "any"}
_REF_RE = re.compile(r"^\$(input|last|\d+)(?P<path>(\.[A-Za-z0-9_]+|\[\d+\])*)$")


class ManifestError(ValidationError):
    """A manifest (or a manifest-shaped dict) failed validation."""

    def __init__(self, errors: list[str] | str) -> None:
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("invalid skill manifest: " + "; ".join(self.errors))


class WiringError(ValueError):
    """A wiring expression could not be resolved at run time."""


def _parse_type_spec(spec: Any, where: str) -> tuple[str, bool, Any]:
    """Normalize a schema entry → (type_name, required, default).

    Raises ManifestError listing the problem; never returns garbage.
    """
    has_default = False
    default: Any = None
    if isinstance(spec, str):
        text = spec.strip().lower()
        required = not text.endswith("?")
        type_name = text[:-1] if text.endswith("?") else text
    elif isinstance(spec, dict):
        type_name = str(spec.get("type", "")).strip().lower()
        required = bool(spec.get("required", True))
        has_default = "default" in spec
        default = spec.get("default")
        if has_default:
            required = False
    else:
        raise ManifestError(
            f"{where}: schema entry must be a type-name string or a dict, "
            f"got {type(spec).__name__}")
    if type_name not in TYPE_NAMES:
        raise ManifestError(
            f"{where}: unknown type {type_name!r}; "
            f"expected one of {sorted(TYPE_NAMES)}")
    return type_name, required, default if has_default else None


def _type_matches(type_name: str, value: Any) -> bool:
    if type_name == "any":
        return True
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "list":
        return isinstance(value, list)
    if type_name == "dict":
        return isinstance(value, dict)
    return False  # unreachable — _parse_type_spec guards the names


def validate_schema(schema: dict[str, Any], data: Any,
                     *, label: str = "input") -> list[str]:
    """Check ``data`` against a dict-based schema.

    Returns a list of human-readable problems; empty means valid.  Never
    raises on bad data — a non-dict ``data`` is itself one problem.
    """
    errors: list[str] = []
    if not isinstance(schema, dict):
        return [f"{label}: schema must be a dict, got {type(schema).__name__}"]
    if not isinstance(data, dict):
        return [f"{label}: expected an object, got {type(data).__name__}"]
    for key, spec in schema.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{label}: schema key must be a non-empty string, "
                          f"got {key!r}")
            continue
        try:
            type_name, required, _default = _parse_type_spec(
                spec, f"{label} schema key {key!r}")
        except ManifestError as exc:
            errors.extend(exc.errors)
            continue
        if key not in data:
            if required:
                errors.append(f"{label}: missing required key {key!r} "
                              f"(expected {type_name})")
            continue
        value = data[key]
        if not _type_matches(type_name, value):
            errors.append(f"{label}: key {key!r} expected {type_name}, "
                          f"got {type(value).__name__}")
    return errors


def resolve_expression(expr: Any, *, skill_input: dict[str, Any],
                       step_outputs: list[Any], step_index: int) -> Any:
    """Resolve one wiring value: literals pass through, ``$...`` refs are
    looked up.  Raises :class:`WiringError` with a concrete message when a
    reference cannot be resolved."""
    if not isinstance(expr, str) or not expr.startswith("$"):
        return expr
    match = _REF_RE.match(expr)
    if not match:
        raise WiringError(
            f"bad wiring expression {expr!r} at step {step_index}: expected "
            f"$input.<path>, $<n>.<path> or $last.<path>")
    root, path = match.group(1), match.group("path")
    if root == "input":
        base: Any = skill_input
    elif root == "last":
        if step_index == 0:
            raise WiringError(
                f"$last used at step 0: there is no previous step")
        base = step_outputs[step_index - 1]
    else:
        ref = int(root)
        if ref >= step_index:
            raise WiringError(
                f"step {step_index} references ${ref}, which has not run "
                f"yet — wiring may only point backwards")
        base = step_outputs[ref]
    current = base
    for part in re.findall(r"\.([A-Za-z0-9_]+)|\[(\d+)\]", path):
        key = part[0] or int(part[1])
        if isinstance(key, int):
            if not isinstance(current, (list, tuple)) or not (
                    0 <= key < len(current)):
                raise WiringError(
                    f"{expr!r} at step {step_index}: index [{key}] out of "
                    f"range")
            current = current[key]
        else:
            if not isinstance(current, dict) or key not in current:
                raise WiringError(
                    f"{expr!r} at step {step_index}: key {key!r} not present "
                    f"in the referenced output")
            current = current[key]
    return current


@dataclass
class SkillManifest:
    """The validated shape of an executable skill."""

    name: str
    version: str = "1.0.0"
    tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    wiring: list[dict[str, Any]] = field(default_factory=list)
    owner: str = ""
    description: str = ""

    # ── validation ──────────────────────────────────────────────────────
    def validate(self) -> list[str]:
        """Return every problem with this manifest; empty means valid."""
        errors: list[str] = []
        if not isinstance(self.name, str) or not NAME_RE.match(self.name):
            errors.append(
                f"name {self.name!r} is invalid: 1-64 chars, letters/digits "
                f"plus _ . -, must start with a letter or digit")
        if not isinstance(self.version, str) or not SEMVER_RE.match(
                self.version):
            errors.append(
                f"version {self.version!r} is invalid: expected semver-ish "
                f"like '1.0.0' or '2.1-rc1'")
        if not isinstance(self.tools, list) or not self.tools:
            errors.append("tools must be a non-empty list of tool names in "
                          "execution order")
        else:
            for i, tool in enumerate(self.tools):
                if not isinstance(tool, str) or not tool.strip():
                    errors.append(f"tools[{i}] must be a non-empty tool name, "
                                  f"got {tool!r}")
        for label in ("input_schema", "output_schema"):
            schema = getattr(self, label)
            if not isinstance(schema, dict):
                errors.append(f"{label} must be a dict, got "
                              f"{type(schema).__name__}")
                continue
            for key, spec in schema.items():
                if not isinstance(key, str) or not key:
                    errors.append(f"{label}: schema key must be a non-empty "
                                  f"string, got {key!r}")
                    continue
                try:
                    _parse_type_spec(spec, f"{label} key {key!r}")
                except ManifestError as exc:
                    errors.extend(exc.errors)
        errors.extend(self._validate_wiring())
        if not isinstance(self.owner, str):
            errors.append(f"owner must be a string, got "
                          f"{type(self.owner).__name__}")
        elif self.owner and len(self.owner) > 128:
            errors.append("owner must be at most 128 chars")
        if not isinstance(self.description, str):
            errors.append(f"description must be a string, got "
                          f"{type(self.description).__name__}")
        return errors

    def _validate_wiring(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.wiring, list):
            return ["wiring must be a list with one entry per tool step "
                    f"(got {type(self.wiring).__name__})"]
        if len(self.wiring) > len(self.tools):
            errors.append(f"wiring has {len(self.wiring)} entries but tools "
                          f"has only {len(self.tools)} steps")
        for i, entry in enumerate(self.wiring):
            if not isinstance(entry, dict):
                errors.append(f"wiring[{i}] must be a dict mapping "
                              f"parameter names to expressions, got "
                              f"{type(entry).__name__}")
                continue
            for param, expr in entry.items():
                if not isinstance(param, str) or not param:
                    errors.append(f"wiring[{i}]: parameter name must be a "
                                  f"non-empty string, got {param!r}")
                    continue
                if isinstance(expr, str) and expr.startswith("$"):
                    match = _REF_RE.match(expr)
                    if not match:
                        errors.append(
                            f"wiring[{i}][{param!r}]: bad expression {expr!r}; "
                            f"expected $input.<path>, $<n>.<path> or "
                            f"$last.<path>")
                        continue
                    root = match.group(1)
                    if root == "last" and i == 0:
                        errors.append(f"wiring[0][{param!r}]: $last used at "
                                      f"step 0 — there is no previous step")
                    elif root.isdigit() and int(root) >= i:
                        errors.append(
                            f"wiring[{i}][{param!r}]: ${root} points at step "
                            f"{root}, which has not run yet — wiring may "
                            f"only point backwards")
        return errors

    def validate_or_raise(self) -> "SkillManifest":
        errors = self.validate()
        if errors:
            raise ManifestError(errors)
        return self

    # ── schema helpers ──────────────────────────────────────────────────
    def validate_input(self, data: Any) -> list[str]:
        return validate_schema(self.input_schema, data, label="input")

    def validate_output(self, data: Any) -> list[str]:
        return validate_schema(self.output_schema, data, label="output")

    def step_input(self, step_index: int, skill_input: dict[str, Any],
                   step_outputs: list[Any]) -> dict[str, Any]:
        """Build the kwargs for one step.

        Threading rule: a step with no wiring entry declared receives the
        run input unchanged (the entry point needs the raw input); a step
        *with* wiring receives exactly its resolved mapping — never a
        silent merge, so a strict tool signature never gets surprise
        kwargs from an unrelated key.
        """
        entry: dict[str, Any] = {}
        if 0 <= step_index < len(self.wiring):
            entry = self.wiring[step_index]
        if not entry:
            return dict(skill_input)
        return {
            param: resolve_expression(
                expr, skill_input=skill_input,
                step_outputs=step_outputs, step_index=step_index)
            for param, expr in entry.items()
        }

    # ── serialization ───────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "tools": list(self.tools),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "wiring": [dict(w) for w in self.wiring],
            "owner": self.owner,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillManifest":
        if not isinstance(data, dict):
            raise ManifestError(
                f"manifest must be a dict, got {type(data).__name__}")
        manifest = cls(
            name=data.get("name", ""),
            version=str(data.get("version", "1.0.0")),
            tools=list(data.get("tools", []) or []),
            input_schema=dict(data.get("input_schema", {}) or {}),
            output_schema=dict(data.get("output_schema", {}) or {}),
            wiring=[dict(w) for w in (data.get("wiring", []) or [])],
            owner=data.get("owner", "") or "",
            description=data.get("description", "") or "",
        )
        return manifest.validate_or_raise()
