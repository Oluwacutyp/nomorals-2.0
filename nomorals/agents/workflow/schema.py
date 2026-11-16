"""Schema-typed handoffs for workflow agents.

This is a WRAPPER, not a new validator: the dependency-free
JSON-schema subset validator already lives at
``nomorals/plugins/manifest.py:validate_against_schema`` (type,
enum, required, properties, additionalProperties, items,
minimum/maximum + exclusives, minLength/maxLength, pattern) and
covers exactly the subset workflow handoffs need. A second
validator would be a duplicate system (SOUL §5). The wrapper only
translates the plugin error into a workflow error, so callers see
one error type.
"""

from __future__ import annotations

from typing import Any

from ...plugins.errors import ManifestError
from ...plugins.manifest import validate_against_schema
from .errors import WorkflowError

__all__ = ["validate_result"]


def validate_result(value: Any, schema: dict[str, Any],
                    where: str = "result") -> Any:
    """Validate ``value`` against ``schema``; return it unchanged.

    Raises :exc:`WorkflowError` naming the first problem, with the
    ``where`` prefix the caller chose (usually the agent call's
    role or the result's field). A missing schema is a caller bug,
    not a validation failure — it raises too.
    """
    if not isinstance(schema, dict):
        raise WorkflowError(
            f"{where}: schema must be an object, got "
            f"{type(schema).__name__}")
    try:
        validate_against_schema(value, schema, where)
    except ManifestError as exc:
        raise WorkflowError(str(exc)) from exc
    return value
