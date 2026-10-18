"""Typed encoding for idempotent replay.

Plain ``json.dumps(value, default=str)`` destroys non-JSON-native
types on replay: tuples come back as lists, sets/bytes/custom
objects as strings. These helpers tag non-JSON-native values so a
replayed outcome restores the ACTUAL types, not JSON shapes.
Tagged forms are dicts with an exact key set (``{"__type__", ...}``)
so a plain user dict that merely contains a ``"__type__"`` key is
never mis-decoded: such dicts go through the tagged-dict form.
"""

from __future__ import annotations

import base64
import datetime
import decimal
import importlib
from typing import Any


_TYPED_EXACT_KEYS = {
    "tuple": {"__type__", "items"},
    "set": {"__type__", "items"},
    "frozenset": {"__type__", "items"},
    "bytes": {"__type__", "data"},
    "bytearray": {"__type__", "data"},
    "datetime": {"__type__", "data"},
    "decimal": {"__type__", "data"},
    "dict": {"__type__", "items"},
    "custom": {"__type__", "class", "state"},
}


def _encode_typed(value: Any) -> Any:
    """Encode ``value`` for JSON storage, preserving non-JSON types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"__type__": "tuple",
                "items": [_encode_typed(v) for v in value]}
    if isinstance(value, list):
        return [_encode_typed(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {"__type__": "frozenset"
                if isinstance(value, frozenset) else "set",
                "items": [_encode_typed(v) for v in value]}
    if isinstance(value, (bytes, bytearray)):
        return {"__type__": "bytearray"
                if isinstance(value, bytearray) else "bytes",
                "data": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, datetime.datetime):
        return {"__type__": "datetime", "data": value.isoformat()}
    if isinstance(value, decimal.Decimal):
        return {"__type__": "decimal", "data": str(value)}
    if isinstance(value, dict):
        if "__type__" in value or not all(
                isinstance(k, str) for k in value):
            # Tagged-dict form: preserves non-str keys AND a literal
            # "__type__" key without ambiguity.
            return {"__type__": "dict",
                    "items": [[_encode_typed(k), _encode_typed(v)]
                              for k, v in value.items()]}
        return {k: _encode_typed(v) for k, v in value.items()}
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict) and not isinstance(value, type):
        cls = type(value)
        return {"__type__": "custom",
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "state": _encode_typed(dict(state))}
    # Exotic types with no faithful encoding: honest lossy fallback
    # (a string), same as the old ``default=str`` behaviour.
    return str(value)


def _decode_typed(value: Any) -> Any:
    """Inverse of :func:`_encode_typed`. Never raises on malformed data:
    an unresolvable tag degrades to the decoded payload."""
    if isinstance(value, list):
        return [_decode_typed(v) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("__type__")
    if isinstance(tag, str) and tag in _TYPED_EXACT_KEYS \
            and set(value) == _TYPED_EXACT_KEYS[tag]:
        if tag == "tuple":
            return tuple(_decode_typed(v) for v in value["items"])
        if tag in ("set", "frozenset"):
            items = [_decode_typed(v) for v in value["items"]]
            return frozenset(items) if tag == "frozenset" else set(items)
        if tag in ("bytes", "bytearray"):
            try:
                raw = base64.b64decode(str(value["data"]))
            except Exception:  # noqa: BLE001 - corrupt payload
                return b""
            return raw if tag == "bytes" else bytearray(raw)
        if tag == "datetime":
            try:
                return datetime.datetime.fromisoformat(str(value["data"]))
            except Exception:  # noqa: BLE001 - corrupt payload
                return value["data"]
        if tag == "decimal":
            try:
                return decimal.Decimal(str(value["data"]))
            except Exception:  # noqa: BLE001 - corrupt payload
                return value["data"]
        if tag == "dict":
            return {_decode_typed(k): _decode_typed(v)
                    for k, v in value["items"]}
        if tag == "custom":
            try:
                dotted = str(value["class"])
                module_name, _, qualname = dotted.rpartition(".")
                cls = importlib.import_module(module_name)
                for part in qualname.split("."):
                    cls = getattr(cls, part)
                obj = cls.__new__(cls)
                obj.__dict__.update(_decode_typed(value["state"]))
                return obj
            except Exception:  # noqa: BLE001 - class gone: state dict
                return _decode_typed(value["state"])
    return {k: _decode_typed(v) for k, v in value.items()}
