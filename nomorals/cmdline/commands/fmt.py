"""``nm fmt`` — the text-transformation family on the command line.

    nm fmt '{"a": 1}' [--json]
    nm fmt json '{"a": 1}' [--json]
    nm fmt --format json '{"a": }' [--json]
    nm fmt validate json '{"a": 1}' [--json]
    nm fmt minify css 'a { color: red; }' [--json]
    nm fmt convert json yaml '{"a": 1}' [--json]
    nm fmt fix "{'a': 1,}" [--json]
    nm fmt detect '{"a": 1}' [--json]

The same textfmt organ chat /fmt uses: the input is parsed by a real
parser first, so a malformed input returns a located error (line,
column, machine-readable kind), never a guess; convert∘convert-back
returns equal data; damaged JSON is repaired and the repair is proved
with a real parse. A format she cannot detect is refused by name.
Text comes from the argument, or from stdin when no text is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...codews.textfmt import FORMATS, format_transform_reply, transform

__all__ = ["_cmd_fmt"]


def _read_text(words: list[str]) -> str:
    if words:
        return " ".join(words)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _cmd_fmt(args: argparse.Namespace, context: Any) -> int:
    words = list(getattr(args, "fmt_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    fmt_format = (getattr(args, "fmt_format", None) or "").strip().lower()
    to_format = (getattr(args, "to_format", None) or "").strip().lower()
    op_attr = (getattr(args, "operation", None) or "").strip().lower()
    verb = (words[0].lower() if words else "")

    def emit(result: Any) -> int:
        if as_json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print(format_transform_reply(result))
        return 0 if result.ok else 1

    def usage(msg: str) -> int:
        print(msg, file=sys.stderr)
        return 2

    # An explicit verb in the words wins; otherwise the --operation
    # attribute (the test seam) wins; otherwise beautify.
    if verb in {"detect", "fix", "validate", "minify", "convert"}:
        operation, rest = verb, words[1:]
    elif op_attr in {"beautify", "minify", "validate", "convert",
                     "fix", "detect"}:
        operation, rest = op_attr, words
    else:
        operation, rest = "beautify", words

    if operation in {"detect", "fix"}:
        text = _read_text(rest)
        if not text.strip():
            return usage(f"usage: nm fmt {operation} <text>")
        return emit(transform(text, operation=operation))
    if operation in {"validate", "minify"}:
        if fmt_format:
            fmt_in, text = fmt_format, _read_text(rest)
        elif len(rest) >= 2:
            fmt_in, text = rest[0].lower(), _read_text(rest[1:])
        else:
            return usage(f"usage: nm fmt {operation} <format> <text>")
        if not text.strip():
            return usage(f"usage: nm fmt {operation} <format> <text>")
        return emit(transform(text, operation=operation, format_in=fmt_in))
    if operation == "convert":
        if fmt_format and to_format:
            fmt_in, fmt_out, text = fmt_format, to_format, _read_text(rest)
        elif len(rest) >= 3:
            fmt_in = rest[0].lower()
            fmt_out = rest[1].lower()
            text = _read_text(rest[2:])
        else:
            return usage("usage: nm fmt convert <from> <to> <text>")
        if not text.strip():
            return usage("usage: nm fmt convert <from> <to> <text>")
        return emit(transform(text, operation="convert", format_in=fmt_in,
                              format_out=fmt_out))
    # beautify
    if fmt_format:
        fmt_in, text = fmt_format, _read_text(rest)
    elif len(rest) >= 2 and rest[0].lower() in FORMATS:
        fmt_in, text = rest[0].lower(), _read_text(rest[1:])
    else:
        fmt_in, text = "auto", _read_text(rest)
    if not text.strip():
        return usage("usage: nm fmt <text> | <format> <text> | validate "
                     "<format> <text> | minify <format> <text> | convert "
                     "<from> <to> <text> | fix <text> | detect <text>")
    return emit(transform(text, operation="beautify", format_in=fmt_in))
