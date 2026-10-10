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
  * ``$<step_id>.<dotted.path>`` — from a named step's output (needs
    ``step_ids``; the GitHub Actions ``steps.<id>.outputs`` shape)
  * ``$last.<dotted.path>``  — from the previous step's output
  * ``$env.VARNAME``         — from the process environment (runner must
    opt in with ``allow_env=True``)
Anything else is a literal passed through unchanged.

Threading rule: a step with no wiring entry declared receives the run
input unchanged; a step *with* wiring receives exactly its resolved
mapping (no silent merge).

Schema entries are either a type-name string (``"str"``, ``"int?"`` —
trailing ``?`` marks the key optional) or a dict
``{"type": "str", "required": false, "default": ...}``.

Execution policy (all optional, all validated):
  * ``retries`` — Temporal-style retry policy for flaky steps:
    ``{"max_attempts": 3, "initial_backoff_s": 1.0,
    "backoff_multiplier": 2.0, "max_backoff_s": 30.0,
    "retryable_errors": [...], "non_retryable_errors": [...]}``.
    Error classification is substring matching against the failure
    message; permanent failures (wiring, validation, capability,
    unknown tool) never retry.
  * ``timeout_s`` — per-step wall-clock ceiling; a step that exceeds it
    fails with a timeout error instead of hanging the run.
  * ``on_error`` — per-step ``continue-on-error`` (GitHub Actions):
    ``[{"continue": true, "fallback": "$input.default"}]`` — the step's
    failure is recorded but the chain keeps going with the fallback as
    the step's output.
  * ``coerce_inputs`` — when true, the runner coerces run inputs toward
    the declared types (``"5"`` → ``5``, ``"true"`` → ``True``,
    JSON strings → list/dict), GitHub Actions ``fromJSON`` style.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    "parse_wiring_root",
    "sanitize_description",
    "diff_manifests",
    "SEMVER_RE",
    "NAME_RE",
    "TYPE_NAMES",
    "RESERVED_NAMES",
]

#: "semver-ish": 1.0, 1.0.0, 2.1.0-rc1 are all fine.
SEMVER_RE = re.compile(r"^\d+\.\d+(\.\d+)?(-[0-9A-Za-z.-]+)?$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TYPE_NAMES = {"str", "int", "float", "bool", "list", "dict", "any"}
#: Names that may never be skill names — the Agent Skills standard
#: reserves vendor/role words, and role-adjacent words invite confusion.
RESERVED_NAMES = frozenset({
    "anthropic", "claude", "system", "developer", "assistant", "user",
    "admin", "root", "devon",
})
#: Wiring roots with built-in meaning; step_ids may not shadow them.
RESERVED_ROOTS = frozenset({"input", "last", "env"})
_STEP_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_REF_RE = re.compile(
    r"^\$(input|last|env|[A-Za-z_][A-Za-z0-9_-]*|\d+)"
    r"(?P<path>(\.[A-Za-z0-9_]+|\[\d+\])*)$")

_JSON_SCHEMA_TYPES = {
    "str": "string", "int": "integer", "float": "number",
    "bool": "boolean", "list": "array", "dict": "object",
}

# ── description sanitization (sup-skill-poison) ─────────────────────────
# A skill description steers the agent ("when to use this skill"), so a
# poisoned description is a prompt-injection vector.  Two-tier cleaning:
#   * strip: hidden/zero-width unicode, ANSI escapes, control characters.
#     Hidden unicode is replaced with a space first, so it cannot split a
#     marker phrase apart to dodge detection.
#   * reject: instruction-override markers fail the install outright.
# The marker list is deliberately tight: none of these phrases has a
# legitimate use in a one-line "when to use this skill" description.
# Research grounding: Invariant Labs' tool-poisoning demos, the OWASP
# ASI02 (2026) tool-poisoning class, and the documented defenses — review
# and pin descriptions, diff them on every change, scan for imperative
# sequencing text ("before using any other tool, call …").


def _hidden_unicode_chars() -> str:
    points = [0x00AD, 0xFEFF]
    points += list(range(0x200B, 0x2010))  # zero-width space/joiners, LRM/RLM
    points += list(range(0x202A, 0x202F))  # bidi embeddings and overrides
    points += list(range(0x2060, 0x2065))  # word joiner, invisible operators
    points += list(range(0x2066, 0x2070))  # isolates
    return "".join(chr(c) for c in points)


_HIDDEN_UNICODE_RE = re.compile("[" + re.escape(_hidden_unicode_chars()) + "]")
_ANSI_ESCAPE_RE = re.compile(re.escape(chr(0x1B)) + "\\[[0-9;?]*[A-Za-z]")
_CONTROL_CHARS_RE = re.compile(
    "[" + "".join(chr(c) for c in
                   list(range(0x00, 0x09)) + [0x0B, 0x0C] +
                   list(range(0x0E, 0x20)) + list(range(0x7F, 0xA0))) + "]")

#: (pattern, human-readable reason) — a match REJECTS the description.
#: All patterns are case-insensitive.
_INJECTION_MARKERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"\b(ignore|disregard|forget|overlook)\s+"
        r"((all|any|your|the|previous|prior|earlier|above)\s+)*"
        r"instructions?\b", re.IGNORECASE),
     "instruction-override phrase ('ignore ... instructions')"),
    (re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
     "identity-override phrase ('you are now')"),
    (re.compile(r"\b(hidden|secret|embedded|concealed)\s+instructions?\b",
                re.IGNORECASE),
     "hidden-instruction marker"),
    (re.compile(r"(^|[.\n])\s*(new\s+|updated\s+|additional\s+|revised\s+)?"
                r"instructions?\s*:", re.IGNORECASE),
     "embedded instruction block"),
    (re.compile(r"(^|[.\n])\s*(system|developer|assistant)\s*:",
                re.IGNORECASE),
     "role-header marker"),
    (re.compile(r"\[/?INST\]|<<SYS>>|<\|\s*(system|im_start|assistant|user)\s*\|>",
                re.IGNORECASE),
     "chat-template marker"),
    (re.compile(r"\bexfiltrat\w*\b", re.IGNORECASE),
     "'exfiltrate' directive"),
    (re.compile(r"\bjailbreak\b", re.IGNORECASE),
     "'jailbreak' directive"),
    (re.compile(
        r"\boverride\s+(\w+\s+){0,2}"
        r"(instructions?|rules?|polic(ies|y)|guardrails|safeguards)\b",
        re.IGNORECASE),
     "override directive"),
    (re.compile(
        r"\b(do\s+not|don't|never)\s+(mention|reveal|tell|disclose|expose|show)\b",
        re.IGNORECASE),
     "secrecy directive"),
    # ── tool-poisoning phrasing (Invariant Labs demos / ASI02) ──
    (re.compile(r"<\s*/?\s*important\s*>", re.IGNORECASE),
     "hidden-instruction delimiter ('<IMPORTANT>')"),
    (re.compile(
        r"\bbefore\s+(using|calling|running|invoking)\s+"
        r"(any\s+)?(other\s+|this\s+|that\s+|the\s+)?tools?\b",
        re.IGNORECASE),
     "sequencing directive ('before using ... tool')"),
    (re.compile(
        r"\bdo\s+not\s+(tell|inform|notify)\s+(the\s+)?users?\b",
        re.IGNORECASE),
     "secrecy directive ('do not tell the user')"),
    (re.compile(
        r"\b(you\s+must|make\s+sure\s+you)\s+(first|always)\s+"
        r"(call|run|execute|invoke)\b", re.IGNORECASE),
     "forced tool-call directive"),
]


def _rejection_reason(text: str) -> str | None:
    """First matching injection marker's reason, or None when clean."""
    for pattern, reason in _INJECTION_MARKERS:
        if pattern.search(text):
            return reason
    return None


def sanitize_description(text: Any, max_len: int = 200
                         ) -> tuple[str | None, str | None]:
    """Clean a skill description for install.  Never raises.

    Returns ``(cleaned, None)`` when the description is safe, or
    ``(None, reason)`` when it carries instruction-override markers and
    the install must be refused honestly.

    Cleaning (always applied): hidden/zero-width unicode becomes a space
    (so it cannot split a marker phrase to dodge detection), ANSI escape
    sequences and control characters are stripped, whitespace collapses
    to single spaces, then the text is stripped and truncated to
    ``max_len``.  Marker detection runs on the cleaned text.
    """
    try:
        if not isinstance(text, str):
            return None, "description must be a string, got %s" % (
                type(text).__name__,)
        cleaned = _HIDDEN_UNICODE_RE.sub(" ", text)
        cleaned = _ANSI_ESCAPE_RE.sub("", cleaned)
        cleaned = _CONTROL_CHARS_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if max_len and len(cleaned) > max_len:
            cleaned = cleaned[:max_len].rstrip()
        reason = _rejection_reason(cleaned)
        if reason is not None:
            return None, "rejected: " + reason
        return cleaned, None
    except Exception as exc:  # noqa: BLE001 — never raises by contract
        return None, "sanitize failed: %s" % exc


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


def _coerce_value(type_name: str, value: Any) -> tuple[Any, bool]:
    """Try to coerce ``value`` toward ``type_name`` (fromJSON style).

    Returns ``(new_value, changed)``.  Never raises; uncoercible values
    come back unchanged so validation can report them honestly.
    """
    if _type_matches(type_name, value):
        return value, False
    try:
        if type_name == "int" and isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?\d+", text):
                return int(text), True
        elif type_name == "float" and isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text):
                return float(text), True
        elif type_name == "bool" and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True, True
            if low in ("false", "0", "no", "off"):
                return False, True
        elif type_name in ("list", "dict") and isinstance(value, str):
            parsed = json.loads(value)
            if _type_matches(type_name, parsed):
                return parsed, True
    except (ValueError, TypeError):
        pass
    return value, False


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
                       step_outputs: list[Any], step_index: int,
                       step_ids: list[str] | None = None,
                       allow_env: bool = False,
                       env: dict[str, str] | None = None) -> Any:
    """Resolve one wiring value: literals pass through, ``$...`` refs are
    looked up.  Raises :class:`WiringError` with a concrete message when a
    reference cannot be resolved.

    Roots: ``$input``, ``$last``, ``$<n>``, ``$<step_id>`` (needs
    ``step_ids``), ``$env`` (needs ``allow_env=True``; reads exactly one
    ``$env.VARNAME`` segment).
    """
    if not isinstance(expr, str) or not expr.startswith("$"):
        return expr
    match = _REF_RE.match(expr)
    if not match:
        raise WiringError(
            f"bad wiring expression {expr!r} at step {step_index}: expected "
            f"$input.<path>, $<n>.<path>, $<step_id>.<path>, $last.<path> "
            f"or $env.VARNAME")
    root, path = match.group(1), match.group("path")
    if root == "input":
        base: Any = skill_input
    elif root == "env":
        if not allow_env:
            raise WiringError(
                f"{expr!r} at step {step_index}: $env references are "
                f"disabled for this run (runner allow_env=False)")
        var = path[1:] if path.startswith(".") else ""
        if not var or "." in var or "[" in var:
            raise WiringError(
                f"{expr!r} at step {step_index}: $env takes exactly one "
                f"variable name, e.g. $env.HOME")
        source = os.environ if env is None else env
        if var not in source:
            raise WiringError(
                f"{expr!r} at step {step_index}: environment variable "
                f"{var!r} is not set")
        return source[var]
    elif root == "last":
        if step_index == 0:
            raise WiringError(
                f"$last used at step 0: there is no previous step")
        base = step_outputs[step_index - 1]
    elif root.isdigit():
        ref = int(root)
        if ref >= step_index:
            raise WiringError(
                f"step {step_index} references ${ref}, which has not run "
                f"yet — wiring may only point backwards")
        base = step_outputs[ref]
    else:
        ids = step_ids or []
        if root not in ids:
            raise WiringError(
                f"{expr!r} at step {step_index}: unknown step id {root!r} "
                f"(known: {', '.join(ids) if ids else 'none'})")
        ref = ids.index(root)
        if ref >= step_index:
            raise WiringError(
                f"step {step_index} references step id {root!r} (step "
                f"{ref}), which has not run yet — wiring may only point "
                f"backwards")
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


def parse_wiring_root(expr: Any) -> tuple[str | None, str | None]:
    """Split a wiring expression into ``(root, path)``.

    Returns ``(None, None)`` when ``expr`` is not a ``$``-reference
    (a literal), and ``(None, <reason>)`` when it starts with ``$`` but
    is malformed.  Roots: ``input``, ``last``, ``env``, a step index,
    or a step id.
    """
    if not isinstance(expr, str) or not expr.startswith("$"):
        return None, None
    match = _REF_RE.match(expr)
    if not match:
        return None, (f"bad wiring expression {expr!r}: expected "
                      f"$input.<path>, $<n>.<path>, $<step_id>.<path>, "
                      f"$last.<path> or $env.VARNAME")
    return match.group(1), match.group("path")


def diff_manifests(old: "SkillManifest",
                   new: "SkillManifest") -> list[str]:
    """Human-readable diff between two manifests of the same skill.

    The review aid for version bumps: what changed in the tool chain,
    the wiring, the schemas, and the execution policy.  Empty means
    "no effective change" (ignoring the version string itself).
    """
    changes: list[str] = []
    if old.tools != new.tools:
        removed = [t for t in old.tools if t not in new.tools]
        added = [t for t in new.tools if t not in old.tools]
        if removed:
            changes.append("tools removed: " + ", ".join(removed))
        if added:
            changes.append("tools added: " + ", ".join(added))
        if not removed and not added and old.tools != new.tools:
            changes.append("tool order changed: %s → %s" % (
                " → ".join(old.tools), " → ".join(new.tools)))
    for i in range(max(len(old.wiring), len(new.wiring))):
        ow = old.wiring[i] if i < len(old.wiring) else {}
        nw = new.wiring[i] if i < len(new.wiring) else {}
        if ow != nw:
            tool = new.tools[i] if i < len(new.tools) else f"step {i}"
            changes.append(f"wiring changed at step {i} ({tool})")
    for label in ("input_schema", "output_schema"):
        o_schema, n_schema = getattr(old, label), getattr(new, label)
        for key in sorted(set(o_schema) | set(n_schema)):
            if key not in o_schema:
                changes.append(f"{label}: key {key!r} added")
            elif key not in n_schema:
                changes.append(f"{label}: key {key!r} removed")
            elif o_schema[key] != n_schema[key]:
                changes.append(f"{label}: key {key!r} spec changed "
                                f"({o_schema[key]!r} → {n_schema[key]!r})")
    for attr in ("retries", "timeout_s", "on_error", "coerce_inputs",
                 "step_ids", "tags", "license"):
        if getattr(old, attr) != getattr(new, attr):
            changes.append(f"policy changed: {attr}")
    if old.description != new.description:
        changes.append("description changed")
    if old.owner != new.owner:
        changes.append(f"owner changed: {old.owner!r} → {new.owner!r}")
    return changes


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
    # ── execution policy ──
    retries: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 0.0
    on_error: list[dict[str, Any]] = field(default_factory=list)
    coerce_inputs: bool = False
    step_ids: list[str] = field(default_factory=list)
    # ── Agent Skills standard metadata ──
    tags: list[str] = field(default_factory=list)
    license: str = ""
    short_description: str = ""

    # ── validation ──────────────────────────────────────────────────────
    def validate(self) -> list[str]:
        """Return every problem with this manifest; empty means valid."""
        errors: list[str] = []
        if not isinstance(self.name, str) or not NAME_RE.match(self.name):
            errors.append(
                f"name {self.name!r} is invalid: 1-64 chars, letters/digits "
                f"plus _ . -, must start with a letter or digit")
        elif self.name.lower() in RESERVED_NAMES:
            errors.append(f"name {self.name!r} is reserved and cannot be "
                          f"used as a skill name")
        elif "<" in self.name or ">" in self.name:
            errors.append(f"name {self.name!r} must not contain XML tags")
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
        errors.extend(self._validate_policy())
        errors.extend(self._validate_metadata())
        if not isinstance(self.owner, str):
            errors.append(f"owner must be a string, got "
                          f"{type(self.owner).__name__}")
        elif self.owner and len(self.owner) > 128:
            errors.append("owner must be at most 128 chars")
        if not isinstance(self.description, str):
            errors.append(f"description must be a string, got "
                          f"{type(self.description).__name__}")
        else:
            # sup-skill-poison: a description that steers the agent with
            # instruction-override markers fails validation outright.
            reason = _rejection_reason(self.description)
            if reason is not None:
                errors.append(f"description rejected: {reason}")
        return errors

    def _validate_wiring(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.wiring, list):
            return ["wiring must be a list with one entry per tool step "
                    f"(got {type(self.wiring).__name__})"]
        if len(self.wiring) > len(self.tools):
            errors.append(f"wiring has {len(self.wiring)} entries but tools "
                          f"has only {len(self.tools)} steps")
        if self.step_ids:
            if len(self.step_ids) != len(self.tools):
                errors.append(f"step_ids has {len(self.step_ids)} entries "
                              f"but tools has {len(self.tools)} steps")
            for i, sid in enumerate(self.step_ids):
                if not isinstance(sid, str) or not _STEP_ID_RE.match(sid):
                    errors.append(
                        f"step_ids[{i}] {sid!r} is invalid: letters/digits "
                        f"plus _ -, must start with a letter or _ "
                        f"(dots are reserved as the path separator)")
                elif sid in RESERVED_ROOTS:
                    errors.append(f"step_ids[{i}] {sid!r} shadows the "
                                  f"reserved wiring root ${sid}")
            if len(set(self.step_ids)) != len(self.step_ids):
                errors.append("step_ids must be unique")
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
                            f"expected $input.<path>, $<n>.<path>, "
                            f"$<step_id>.<path>, $last.<path> or $env.VAR")
                        continue
                    root = match.group(1)
                    if root == "last" and i == 0:
                        errors.append(f"wiring[0][{param!r}]: $last used at "
                                      f"step 0 — there is no previous step")
                    elif root == "env":
                        var = match.group("path")
                        if not var.startswith(".") or "." in var[1:]:
                            errors.append(
                                f"wiring[{i}][{param!r}]: $env takes exactly "
                                f"one variable name, e.g. $env.HOME")
                    elif root.isdigit() and int(root) >= i:
                        errors.append(
                            f"wiring[{i}][{param!r}]: ${root} points at step "
                            f"{root}, which has not run yet — wiring may "
                            f"only point backwards")
                    elif (not root.isdigit() and root not in RESERVED_ROOTS
                          and self.step_ids):
                        if root not in self.step_ids:
                            errors.append(
                                f"wiring[{i}][{param!r}]: unknown step id "
                                f"{root!r}")
                        elif self.step_ids.index(root) >= i:
                            errors.append(
                                f"wiring[{i}][{param!r}]: step id {root!r} "
                                f"has not run yet — wiring may only point "
                                f"backwards")
        return errors

    def _validate_policy(self) -> list[str]:
        errors: list[str] = []
        # retries: Temporal-style policy, always bounded.
        if not isinstance(self.retries, dict):
            errors.append(f"retries must be a dict, got "
                          f"{type(self.retries).__name__}")
        elif self.retries:
            rp = self.retries
            ma = rp.get("max_attempts", 3)
            if not isinstance(ma, int) or isinstance(ma, bool) \
                    or not 1 <= ma <= 10:
                errors.append("retries.max_attempts must be an int in "
                              f"1..10, got {ma!r}")
            for key in ("initial_backoff_s", "max_backoff_s"):
                val = rp.get(key, 0.0)
                if not isinstance(val, (int, float)) \
                        or isinstance(val, bool) or val < 0:
                    errors.append(f"retries.{key} must be a non-negative "
                                  f"number, got {val!r}")
            mult = rp.get("backoff_multiplier", 2.0)
            if not isinstance(mult, (int, float)) or isinstance(mult, bool) \
                    or mult < 1.0:
                errors.append("retries.backoff_multiplier must be a number "
                              f">= 1.0, got {mult!r}")
            for key in ("retryable_errors", "non_retryable_errors"):
                val = rp.get(key, [])
                if not isinstance(val, list) or not all(
                        isinstance(v, str) for v in val):
                    errors.append(f"retries.{key} must be a list of "
                                  f"strings, got {val!r}")
        # timeout: always set a maximum (Temporal rule #2).
        if not isinstance(self.timeout_s, (int, float)) \
                or isinstance(self.timeout_s, bool) or self.timeout_s < 0:
            errors.append(f"timeout_s must be a non-negative number, got "
                          f"{self.timeout_s!r}")
        # on_error: GitHub Actions continue-on-error per step.
        if not isinstance(self.on_error, list):
            errors.append(f"on_error must be a list, got "
                          f"{type(self.on_error).__name__}")
        else:
            if len(self.on_error) > len(self.tools):
                errors.append(f"on_error has {len(self.on_error)} entries "
                              f"but tools has only {len(self.tools)} steps")
            for i, entry in enumerate(self.on_error):
                if not isinstance(entry, dict):
                    errors.append(f"on_error[{i}] must be a dict, got "
                                  f"{type(entry).__name__}")
                    continue
                cont = entry.get("continue", False)
                if not isinstance(cont, bool):
                    errors.append(f"on_error[{i}].continue must be a bool, "
                                  f"got {cont!r}")
                if "fallback" in entry:
                    fb = entry["fallback"]
                    if isinstance(fb, str) and fb.startswith("$"):
                        if not _REF_RE.match(fb):
                            errors.append(
                                f"on_error[{i}].fallback: bad expression "
                                f"{fb!r}")
                if set(entry) - {"continue", "fallback"}:
                    errors.append(f"on_error[{i}]: unknown keys "
                                  f"{sorted(set(entry) - {'continue', 'fallback'})}")
        if not isinstance(self.coerce_inputs, bool):
            errors.append(f"coerce_inputs must be a bool, got "
                          f"{self.coerce_inputs!r}")
        return errors

    def _validate_metadata(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.tags, list) or not all(
                isinstance(t, str) and t for t in self.tags):
            errors.append("tags must be a list of non-empty strings")
        elif len(self.tags) > 16:
            errors.append("tags must have at most 16 entries")
        if not isinstance(self.license, str):
            errors.append(f"license must be a string, got "
                          f"{type(self.license).__name__}")
        elif len(self.license) > 64:
            errors.append("license must be at most 64 chars")
        if not isinstance(self.short_description, str):
            errors.append(f"short_description must be a string, got "
                          f"{type(self.short_description).__name__}")
        elif len(self.short_description) > 200:
            errors.append("short_description must be at most 200 chars")
        else:
            reason = _rejection_reason(self.short_description)
            if reason is not None:
                errors.append(f"short_description rejected: {reason}")
        return errors

    def validate_or_raise(self) -> "SkillManifest":
        errors = self.validate()
        if errors:
            raise ManifestError(errors)
        return self

    # ── policy helpers ──────────────────────────────────────────────────
    def retry_policy(self) -> dict[str, Any] | None:
        """Normalized retry policy, or None when the manifest sets none."""
        if not self.retries:
            return None
        rp = self.retries
        return {
            "max_attempts": int(rp.get("max_attempts", 3)),
            "initial_backoff_s": float(rp.get("initial_backoff_s", 1.0)),
            "backoff_multiplier": float(rp.get("backoff_multiplier", 2.0)),
            "max_backoff_s": float(rp.get("max_backoff_s", 30.0)),
            "retryable_errors": [str(v).lower()
                                 for v in rp.get("retryable_errors", [])],
            "non_retryable_errors": [str(v).lower()
                                     for v in rp.get("non_retryable_errors",
                                                     [])],
        }

    def on_error_for(self, step_index: int) -> dict[str, Any]:
        """The on_error entry for a step ({} when none declared)."""
        if 0 <= step_index < len(self.on_error):
            entry = self.on_error[step_index]
            if isinstance(entry, dict):
                return entry
        return {}

    # ── schema helpers ──────────────────────────────────────────────────
    def validate_input(self, data: Any) -> list[str]:
        return validate_schema(self.input_schema, data, label="input")

    def validate_output(self, data: Any) -> list[str]:
        return validate_schema(self.output_schema, data, label="output")

    def apply_defaults(self, data: dict[str, Any]) -> dict[str, Any]:
        """Fill schema-declared defaults for missing input keys.

        Returns a new dict; never mutates the caller's.  Declared
        defaults were previously parsed but never applied — a gap.
        """
        filled = dict(data)
        for key, spec in self.input_schema.items():
            if key in filled:
                continue
            try:
                _type, _required, default = _parse_type_spec(
                    spec, f"input schema key {key!r}")
            except ManifestError:
                continue
            if default is not None:
                filled[key] = default
        return filled

    def coerce_input(self, data: dict[str, Any]) -> tuple[dict[str, Any],
                                                          list[str]]:
        """Coerce input values toward declared types (fromJSON style).

        Returns ``(coerced, notes)`` where notes name each coercion, e.g.
        ``"key 'count': '5' → 5"``.  Uncoercible values pass through so
        validation can report them honestly.  The runner applies this
        only when ``coerce_inputs`` is true.
        """
        coerced = dict(data)
        notes: list[str] = []
        for key, spec in self.input_schema.items():
            if key not in coerced:
                continue
            try:
                type_name, _required, _default = _parse_type_spec(
                    spec, f"input schema key {key!r}")
            except ManifestError:
                continue
            new_value, changed = _coerce_value(type_name, coerced[key])
            if changed:
                notes.append(f"key {key!r}: {coerced[key]!r} → "
                             f"{new_value!r}")
                coerced[key] = new_value
        return coerced, notes

    def step_input(self, step_index: int, skill_input: dict[str, Any],
                   step_outputs: list[Any], *, allow_env: bool = False,
                   env: dict[str, str] | None = None) -> dict[str, Any]:
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
                step_outputs=step_outputs, step_index=step_index,
                step_ids=self.step_ids or None,
                allow_env=allow_env, env=env)
            for param, expr in entry.items()
        }

    def manifest_hash(self) -> str:
        """Short sha256 over the canonical manifest JSON.

        The rug-pull detector: the registry stores this per installed
        version and warns when a re-install changes a pinned version's
        bytes (per the MCP rug-pull research, nobody deploys this check
        — we do).
        """
        canonical = json.dumps(self.to_dict(), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_json_schema(self) -> dict[str, Any]:
        """The input/output schemas as standard JSON Schema (2020-12).

        Interop export: external validators and UI form-builders speak
        JSON Schema, not our compact type strings.
        """
        def _convert(schema: dict[str, Any]) -> dict[str, Any]:
            properties: dict[str, Any] = {}
            required: list[str] = []
            for key, spec in schema.items():
                try:
                    type_name, is_required, default = _parse_type_spec(
                        spec, f"schema key {key!r}")
                except ManifestError:
                    continue
                prop: dict[str, Any] = {}
                json_type = _JSON_SCHEMA_TYPES.get(type_name)
                if json_type:
                    prop["type"] = json_type
                if default is not None:
                    prop["default"] = default
                properties[key] = prop
                if is_required:
                    required.append(key)
            out: dict[str, Any] = {"type": "object",
                                   "properties": properties}
            if required:
                out["required"] = required
            return out

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.name,
            "description": self.description,
            "input": _convert(self.input_schema),
            "output": _convert(self.output_schema),
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
            "retries": dict(self.retries),
            "timeout_s": self.timeout_s,
            "on_error": [dict(e) for e in self.on_error],
            "coerce_inputs": self.coerce_inputs,
            "step_ids": list(self.step_ids),
            "tags": list(self.tags),
            "license": self.license,
            "short_description": self.short_description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillManifest":
        if not isinstance(data, dict):
            raise ManifestError(
                f"manifest must be a dict, got {type(data).__name__}")
        # sup-skill-poison: sanitize at the install boundary.  Poisoned
        # descriptions fail the install honestly instead of being stored.
        cleaned, reason = sanitize_description(data.get("description", "") or "")
        if reason is not None:
            raise ManifestError(f"description {reason}")
        short_raw = data.get("short_description", "") or ""
        short_cleaned, short_reason = sanitize_description(
            short_raw, max_len=200) if short_raw else ("", None)
        if short_reason is not None:
            raise ManifestError(f"short_description {short_reason}")
        try:
            timeout_s = float(data.get("timeout_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            raise ManifestError(
                f"timeout_s must be a non-negative number, got "
                f"{data.get('timeout_s')!r}")
        manifest = cls(
            name=data.get("name", ""),
            version=str(data.get("version", "1.0.0")),
            tools=list(data.get("tools", []) or []),
            input_schema=dict(data.get("input_schema", {}) or {}),
            output_schema=dict(data.get("output_schema", {}) or {}),
            wiring=[dict(w) for w in (data.get("wiring", []) or [])],
            owner=data.get("owner", "") or "",
            description=cleaned,
            retries=dict(data.get("retries", {}) or {}),
            timeout_s=timeout_s,
            on_error=[dict(e) for e in (data.get("on_error", []) or [])],
            coerce_inputs=bool(data.get("coerce_inputs", False)),
            step_ids=list(data.get("step_ids", []) or []),
            tags=list(data.get("tags", []) or []),
            license=data.get("license", "") or "",
            short_description=short_cleaned,
        )
        return manifest.validate_or_raise()
