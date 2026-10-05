"""``nm tools`` — list available tools and capabilities."""

from __future__ import annotations

import argparse
import json
from typing import Any



def _cmd_tools(args: argparse.Namespace, context: Any) -> int:
    tools = context.tools.register_builtins()
    if args.schema:
        print(json.dumps(tools.schemas(), indent=2))
        return 0
    for schema in tools.schemas():
        params = ", ".join(schema["parameters"])
        print(f"{schema['name']:<20} [{schema['capability'] or '-'}]  {schema['description']}")
        if params:
            print(f"{'':<20} ({params})")
    # Surface tool modules that failed to register — a broken tool must
    # never vanish invisibly.
    failed = tools.failed_modules()
    if failed:
        print(f"\n{len(failed)} tool module(s) failed to register:")
        for entry in failed:
            print(f"  ✗ {entry['module']}: {entry['error']}")
    return 0
