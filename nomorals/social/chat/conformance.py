"""Adapter contract conformance (G5).

The :class:`ChatAdapter` ABC declares the full ``send()`` signature, but
Python does not enforce override-signature compatibility — a subclass
that narrows the signature (a test double missing ``buttons``) silently
breaks every send that forwards those kwargs. This module turns the
convention into a checkable invariant: the base signature is the oracle,
so the check cannot drift from the contract.

Used by ``tests/test_adapter_conformance.py`` and by any harness that
builds adapters dynamically.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from .base import ChatAdapter

__all__ = ["check_adapter_conformance", "all_adapter_classes", "base_send_signature"]


def base_send_signature() -> inspect.Signature:
    """The oracle: the ABC's own ``send()`` signature."""
    return inspect.signature(ChatAdapter.send)


def check_adapter_conformance(cls: type) -> list[str]:
    """Violations of the base ``send()`` contract for ``cls``.

    Empty means conformant. Checks: ``cls`` is a ``ChatAdapter``
    subclass; its ``send()`` accepts every base parameter with the same
    kind (positional vs keyword-only) and the same default; and it adds
    no new *required* positional parameter (which would break every
    existing caller).
    """
    violations: list[str] = []
    name = getattr(cls, "__name__", repr(cls))
    if not (inspect.isclass(cls) and issubclass(cls, ChatAdapter)):
        return [f"{name} is not a ChatAdapter subclass"]
    if cls is ChatAdapter:
        return []
    try:
        sig = inspect.signature(cls.send)
    except (TypeError, ValueError):
        return [f"{name}.send() cannot be inspected"]
    base = base_send_signature()
    params = sig.parameters
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    for pname, base_param in base.parameters.items():
        if pname == "self":
            continue
        if pname not in params:
            violations.append(f"{name}.send() is missing parameter {pname!r}")
            continue
        got = params[pname]
        if got.kind is not base_param.kind:
            violations.append(
                f"{name}.send() parameter {pname!r} has kind {got.kind} "
                f"(base: {base_param.kind})"
            )
        if base_param.default is not inspect.Parameter.empty:
            if got.default is inspect.Parameter.empty:
                violations.append(
                    f"{name}.send() parameter {pname!r} lost its default "
                    f"({base_param.default!r})"
                )
            elif got.default != base_param.default:
                violations.append(
                    f"{name}.send() parameter {pname!r} default "
                    f"{got.default!r} != base {base_param.default!r}"
                )
    if not has_var_keyword:
        for pname, param in params.items():
            if pname in ("self", *base.parameters):
                continue
            if (
                param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and param.default is inspect.Parameter.empty
            ):
                violations.append(
                    f"{name}.send() adds required parameter {pname!r} "
                    "— existing callers cannot satisfy it"
                )
    return violations


def all_adapter_classes() -> list[type[ChatAdapter]]:
    """Every concrete ``ChatAdapter`` subclass defined in this package.

    Modules are imported (all adapter modules keep optional dependencies
    lazy, so this is safe). A class counts once, under the module that
    defines it — re-exports do not duplicate it.
    """
    import nomorals.social.chat as pkg

    out: list[type[ChatAdapter]] = []
    seen: set[type[ChatAdapter]] = set()
    for info in pkgutil.iter_modules(pkg.__path__):
        try:
            module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        except Exception:  # noqa: BLE001 - a broken adapter module is reported by its own tests
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, ChatAdapter)
                and cls is not ChatAdapter
                and cls.__module__ == module.__name__
                and cls not in seen
            ):
                seen.add(cls)
                out.append(cls)
    return sorted(out, key=lambda c: f"{c.__module__}.{c.__name__}")
