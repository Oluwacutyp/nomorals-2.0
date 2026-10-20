"""Argument-level oversight classification for tool calls.

The policy layer used to see only a capability string (``fs.write``,
``exec.shell``) — never the arguments. That made two bypasses possible:

* a confirmation token minted for one file authorized *any* file with
  the same capability (token redirection), and
* an explicit-override phrase ("do it") skipped confirmation even when
  the action targeted the oversight machinery itself.

This module classifies a tool call's *arguments* so the policy can
tell an ordinary write from a write that tampers with its own gates.
It is pure (no imports from the rest of nomorals) so the policy and
the registry can both use it without import cycles, and so it is
trivially unit-testable.

What counts as an oversight target
----------------------------------
* Files that implement the enforcement itself: the capability policy,
  the audit logger, the tool registry, the trust store, the social
  gate, the credential vault, and this classifier. Editing any of
  them edits what the agent is allowed to do, what gets audited, or
  who it may present as.
* Files the owner's standing rules mark untouchable: the partner
  persona/mood blobs (byte-identical by rule) and the social-gate
  actor matrix.
* Secret material: private keys and the secret stores. Writing them
  plants credentials the agent can later read back or exfiltrate.

Matching is by path *suffix*: a tool may address a target as
``nomorals/core/policy.py`` (repo-relative), ``core/policy.py``
(package-relative), or an absolute path. A bare basename such as
``policy.py`` is deliberately NOT enough — other projects have one.

Reads are never oversight: transparency (reading the gates) is
allowed; only writes, deletes, copies-onto, and exec naming a target
are classified. For exec tools the check is a tripwire, not a
containment proof — a command that literally names an oversight
target plus a write indicator is flagged; an obfuscated command that
reconstructs the path at runtime is not caught here and remains
covered by the capability grant, the sandbox, and the audit trail.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["args_digest", "is_oversight_target"]

#: Suffixes (as ``/``-joined part tuples) of files whose modification
#: tampers with the agent's own oversight. Each entry is accepted both
#: repo-relative (``nomorals/...``) and package-relative (without the
#: leading ``nomorals``).
_OVERSIGHT_SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("nomorals", "core", "policy.py"),
    ("core", "policy.py"),
    ("nomorals", "core", "trust.py"),
    ("core", "trust.py"),
    ("nomorals", "core", "logging_setup.py"),
    ("core", "logging_setup.py"),
    ("nomorals", "core", "oversight.py"),
    ("core", "oversight.py"),
    ("nomorals", "tools", "registry.py"),
    ("tools", "registry.py"),
    ("nomorals", "partner", "social_gate.py"),
    ("partner", "social_gate.py"),
    ("nomorals", "partner", "persona.py"),
    ("partner", "persona.py"),
    ("nomorals", "partner", "mood.py"),
    ("partner", "mood.py"),
    ("nomorals", "accounts", "vault.py"),
    ("accounts", "vault.py"),
)

#: Path components that mark secret material wherever they appear.
_SECRET_COMPONENTS: frozenset[str] = frozenset({".ssh", ".devon-secrets"})

#: Private-key filenames (the ``.pub`` halves are public; writing a
#: ``.pub`` is not secret-planting, so they are not listed).
_PRIVATE_KEY_NAMES: frozenset[str] = frozenset(
    {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
)

#: Capabilities whose calls can *change* something. Reads (fs.read,
#: mem.read, db.read, ...) are never oversight.
_WRITE_CAPABILITIES: frozenset[str] = frozenset(
    {
        "fs.write",
        "fs.delete",
        "exec.shell",
        "exec.code",
        "exec.install",
        "db.write",
        "db.admin",
        "sys.config",
    }
)

#: Kwargs that carry a filesystem target. ``fs_copy`` is special-cased
#: to ``destination`` only: copying an oversight file OUT is
#: read-equivalent, not tampering.
_PATH_KWARGS: tuple[str, ...] = (
    "path", "destination", "target", "file", "filepath", "filename",
    "output", "out_path", "source",
)

#: Kwargs that carry code / a shell command for exec tools.
_CODE_KWARGS: tuple[str, ...] = ("command", "code", "script")

#: Tokens that indicate an exec string *writes* rather than reads.
#: ``cat``/``grep``/``diff``/``less`` naming a target stay unflagged.
_WRITE_INDICATORS: tuple[str, ...] = (
    ">", "tee", "sed", "awk", "rm ", "mv ", "cp ", "install", "chmod",
    "chown", "truncate", "dd ", "patch", "git apply", "git checkout",
    "git restore", "git clean", "open(", "write_text", "write(", "unlink",
    "remove(", "rename", "shutil", "os.remove", "os.rename", "pathlib",
    ">>",
)


def args_digest(kwargs: dict[str, Any] | None) -> str:
    """Stable digest of a tool call's arguments.

    Canonical JSON (sorted keys, ``str`` fallback) hashed with SHA-256.
    Two calls with the same arguments in any order produce the same
    digest; any argument change produces a different one. Used to bind
    confirmation tokens to the exact action the owner approved.
    """
    try:
        payload = json.dumps(kwargs or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(kwargs or {})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parts(value: str) -> tuple[str, ...]:
    """Split a path into components, normalizing separators."""
    return tuple(p for p in value.replace("\\", "/").split("/") if p not in ("", "."))


def _is_oversight_path(value: str) -> bool:
    """True when ``value`` names an oversight file or secret material.

    Matching is case-insensitive: on a case-insensitive filesystem
    (Windows — the deployment host) ``NOMORALS/CORE/POLICY.PY`` IS
    the policy file, and a gate that only recognises the exact-case
    spelling would wave the write through. Case-folding the compare
    can only widen matches to case variants of the exact oversight
    suffixes / secret names, which the suffix design already treats
    conservatively (any project with a ``core/policy.py`` matches).
    """
    parts = tuple(p.casefold() for p in _parts(value))
    if not parts:
        return False
    for suffix in _OVERSIGHT_SUFFIXES:
        folded = tuple(p.casefold() for p in suffix)
        if len(parts) >= len(folded) and parts[-len(folded):] == folded:
            return True
    if _SECRET_COMPONENTS.intersection(parts):
        return True
    name = parts[-1]
    if name in _PRIVATE_KEY_NAMES or name.endswith(".pem"):
        return True
    return False


def _is_oversight_exec_text(text: str) -> bool:
    """Tripwire for exec strings: oversight target + write indicator."""
    if not isinstance(text, str) or not text:
        return False
    lowered = text.lower()
    if not any(ind in lowered for ind in _WRITE_INDICATORS):
        return False
    # Reassemble candidate paths from the text: any oversight suffix
    # appearing as a substring of the (separator-normalized, case-folded)
    # command — case variants name the same file on Windows.
    normalized = lowered.replace("\\", "/")
    for suffix in _OVERSIGHT_SUFFIXES:
        if "/".join(suffix) in normalized:
            return True
    for component in _SECRET_COMPONENTS:
        if component in normalized:
            return True
    return False


def is_oversight_target(tool_name: str, capability: str,
                        kwargs: dict[str, Any] | None) -> bool:
    """True when this tool call would write/delete/exec against an
    oversight target.

    ``capability`` decides whether the call can change anything at
    all; read-capability calls are never oversight. ``tool_name``
    special-cases ``fs_copy`` (destination only). Never raises: an
    unparseable argument is treated as non-oversight here — the
    policy's capability check and the tool's own validation still
    apply; this classifier only *adds* the oversight gate, it never
    substitutes for the other checks.
    """
    try:
        if capability not in _WRITE_CAPABILITIES:
            return False
        args = kwargs or {}
        if tool_name == "fs_copy":
            dest = args.get("destination")
            return isinstance(dest, str) and _is_oversight_path(dest)
        for key in _PATH_KWARGS:
            value = args.get(key)
            if isinstance(value, str) and value and _is_oversight_path(value):
                return True
        for key in _CODE_KWARGS:
            value = args.get(key)
            if isinstance(value, str) and _is_oversight_exec_text(value):
                return True
        return False
    except Exception:  # noqa: BLE001 - classifier never breaks a tool call
        return False
