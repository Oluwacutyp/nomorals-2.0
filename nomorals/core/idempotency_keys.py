"""Deterministic idempotency keys for registry tool calls.

The tool registry (``nomorals/tools/registry.py``) can replay a
recorded result instead of re-executing when a call carries an
``idempotency_key`` — but a key only protects anything when callers
derive it from the *intended effect*, never from the attempt. This
module is the one place that derivation lives:

* A **scope** names the unit of work that owns the effect: one inbound
  chat message, one mission step, one plan step, one orchestration run,
  one MCP request. Scopes are carried on a contextvar so the runners
  that own the unit of work set it once and every registry call made
  underneath — on whatever thread the work lands — derives its key
  from it.
* A **key** is ``<scope-kind>:<tool>:<digest>`` where the digest
  covers the scope, the tool name and the canonical arguments. Two
  distinct intended effects (different scope, different tool, or
  different arguments) therefore always get different keys and both
  execute; a retry of the SAME effect (same scope, same tool, same
  arguments) gets the same key and replays. The registry's conflict
  error can never fire from a key derived here, by construction.

No scope → no key (``""``). Callers pass the key only when one is
derived; an absent scope means the layer above could not name the
intended effect (e.g. a chat message with no platform message id),
and inventing one could collide two distinct effects into a false
replay — so the honest degrade is to run unkeyed, exactly as before.

Pure stdlib, layer 0: every caller layer may import this.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

__all__ = [
    "args_fingerprint",
    "canonical_args",
    "current_idempotency_scope",
    "derive_idempotency_key",
    "idempotency_scope",
    "scoped_idempotency_key",
    "set_idempotency_scope",
    "reset_idempotency_scope",
]

_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nomorals_idempotency_scope", default="")


def current_idempotency_scope() -> str:
    """The scope naming the unit of work in flight (``""`` when none)."""
    return _scope.get()


def set_idempotency_scope(scope: str) -> contextvars.Token:
    """Set the current scope; returns the token for
    :func:`reset_idempotency_scope`. Prefer the
    :func:`idempotency_scope` context manager where a ``with`` block
    fits; the token form exists for runners that must set a scope
    around a background-job *submission* (the job registry copies the
    context at submit time, so the scope must be visible exactly then
    and restored immediately after)."""
    return _scope.set(str(scope or ""))


def reset_idempotency_scope(token: contextvars.Token) -> None:
    _scope.reset(token)


@contextmanager
def idempotency_scope(scope: str) -> Iterator[None]:
    """Run a block under ``scope`` as the idempotency scope."""
    token = _scope.set(str(scope or ""))
    try:
        yield
    finally:
        _scope.reset(token)


def canonical_args(args: Mapping[str, Any] | None) -> str:
    """Deterministic JSON for an argument mapping: same logical args →
    same bytes, in any process. Non-serializable values stringify."""
    try:
        return json.dumps(dict(args or {}), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - fingerprinting must never raise
        return repr(dict(args or {}))


def args_fingerprint(args: Mapping[str, Any] | None) -> str:
    """SHA-256 hex digest of :func:`canonical_args`."""
    return hashlib.sha256(
        canonical_args(args).encode("utf-8")).hexdigest()


def derive_idempotency_key(scope: str, tool: str,
                           args: Mapping[str, Any] | None) -> str:
    """The registry key for one intended effect.

    ``""`` when there is no scope or no tool — the caller then runs
    unkeyed rather than inventing a key that could collide.
    """
    scope = str(scope or "")
    tool = str(tool or "")
    if not scope or not tool:
        return ""
    digest = hashlib.sha256(
        f"{scope}\x00{tool}\x00{canonical_args(args)}".encode("utf-8")
    ).hexdigest()[:32]
    kind = scope.split(":", 1)[0] or "scoped"
    return f"{kind}:{tool}:{digest}"


def scoped_idempotency_key(tool: str,
                           args: Mapping[str, Any] | None) -> str:
    """Derive a key from the *current* contextvar scope (``""`` when
    no scope is active). The funnel every scope-consuming caller uses.
    """
    return derive_idempotency_key(current_idempotency_scope(), tool, args)
