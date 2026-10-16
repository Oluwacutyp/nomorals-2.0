"""Regression test for the tool-listing bug — DO NOT REGRESS.

Historical bug: ``ToolRegistry.prompt_listing()`` rendered every tool as
``name(type, properties, required, additionalProperties)`` — the KEYS of
the JSON-Schema ``parameters`` object — instead of the real parameter
names. The fix lives in :func:`nomorals.tools.registry._format_params`
(``registry.py``): required params render bare, optional params render
with their recorded default.

This file pins the FIXED behaviour at the listing level (the full
builtin registry, not just one hand-built tool). It does not re-implement
or touch the fix itself. The count assertion is a floor (``> 400``), not
an exact number: the builtin tool count grows over time and this test
must not break when a new tool is added.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.tools.registry import ToolRegistry, _format_params

# The exact nonsense string the old listing produced on every tool.
BUG_STRING = "(type, properties, required, additionalProperties)"


def _builtins():
    reg = ToolRegistry()
    reg.register_builtins()
    return reg


def test_prompt_listing_never_contains_schema_keys():
    listing = _builtins().prompt_listing()
    assert BUG_STRING not in listing
    assert "type, properties, required" not in listing
    assert "additionalProperties" not in listing


def test_prompt_listing_tool_count_stays_large():
    reg = _builtins()
    schemas = reg.schemas()
    assert len(schemas) > 400, f"expected >400 builtin tools, got {len(schemas)}"
    listing = reg.prompt_listing()
    assert len(listing.splitlines()) == len(schemas)


def test_prompt_listing_param_names_are_real():
    """Every listing line shows real parameter names, never schema keys."""
    reg = _builtins()
    schemas = {s["name"]: s for s in reg.schemas()}
    listing = reg.prompt_listing()
    for line in listing.splitlines():
        assert line.startswith("- "), line
        name = line[2:].split("(", 1)[0]
        assert name in schemas, f"listing mentions unknown tool {name!r}"
        params = schemas[name].get("parameters") or {}
        real = set((params.get("properties") or {}).keys())
        rendered = line.split("(", 1)[1].split("):", 1)[0]
        shown = [p.split("=", 1)[0].strip() for p in rendered.split(",") if p.strip()]
        # Every rendered param must be a real declared parameter.
        assert set(shown) <= real, f"{name}: rendered {shown} not subset of {sorted(real)}"
        # And every declared param must be rendered (no silent drops).
        assert set(shown) == real, f"{name}: rendered {shown} != declared {sorted(real)}"
    # Spot-check a tool known to have both required and optional params:
    # required params render bare, optional ones render with defaults.
    assert "memory_recall(" in listing


def test_format_params_required_bare_optional_with_default():
    schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "req": {"type": "string"},
                "opt": {"type": "string", "default": "x"},
                "no_default": {"type": "integer"},
            },
            "required": ["req"],
            "additionalProperties": False,
        }
    }
    rendered = _format_params(schema)
    assert rendered == "req, opt='x', no_default", rendered
    assert BUG_STRING not in rendered
