"""Datatools: ``format_text`` / ``validate_text`` / ``convert_data``.

The brain's text-transformation family (CodeBeautify core). A thin
surface over :mod:`nomorals.codews.textfmt` — the organ owns the
parse -> validate -> transform loop with located errors; the tools
only route arguments and shape the reply. Pure local work, no
network, no writes: text in, text out, nothing stored.
"""

from __future__ import annotations

from typing import Any

from ..codews.textfmt import (
    FORMATS,
    convert,
    format_transform_reply,
    transform,
    validate,
)
from ..core.policy import Capability

__all__ = ["register"]


def _reply(result: Any) -> dict[str, Any]:
    row = result.to_dict()
    row["reply"] = format_transform_reply(result)
    return row


def register(registry: Any) -> None:
    """Attach the datatools family to a registry."""
    @registry.register(
        "format_text",
        description=(
            "Format text for real — beautify or minify JSON, JSONL, "
            "YAML, TOML, XML, CSV, SQL, CSS, HTML (and JS via prettier "
            "when installed). The input is parsed by a real parser "
            "first, so a malformed input returns a located error "
            "(line, column, machine-readable kind), never a guess. "
            "text=<the text>; format=<json|jsonl|yaml|toml|xml|csv|sql|"
            "css|html|js> or auto to detect; operation=<beautify|minify>."
        ),
        capability=Capability.FS_READ,
    )
    def format_text_tool(text: str = "", format: str = "auto",
                         operation: str = "beautify") -> dict[str, Any]:
        if not text or not str(text).strip():
            raise ValueError("format_text needs text")
        result = transform(str(text), operation=operation,
                           format_in=format or "auto")
        return _reply(result)

    @registry.register(
        "validate_text",
        description=(
            "Validate text as a format — JSON, JSONL, YAML, TOML, XML, "
            "CSV, SQL, CSS, HTML, JS — with the real parser, returning "
            "located diagnostics (line, column, kind) for every defect. "
            "The text is echoed back unchanged; nothing is transformed. "
            "text=<the text>; format=<the format to check>."
        ),
        capability=Capability.FS_READ,
    )
    def validate_text_tool(text: str = "", format: str = "") -> dict[str, Any]:
        if not text or not str(text).strip():
            raise ValueError("validate_text needs text")
        if not format or str(format).strip().lower() not in FORMATS:
            raise ValueError(
                "validate_text needs format, one of: "
                + ", ".join(FORMATS))
        result = validate(str(text), str(format).strip().lower())
        return _reply(result)

    @registry.register(
        "convert_data",
        description=(
            "Convert data between formats through a real shared data "
            "model — JSON, JSONL, YAML, TOML, XML, CSV — plus JSON to "
            "TypeScript interfaces or PHP types, hex to UTF-8 and back, "
            "and HTML to JSX. convert∘convert-back returns equal data. "
            "A pair with no real conversion is refused by name. "
            "text=<the data>; from_format=<source>; to_format=<target>."
        ),
        capability=Capability.FS_READ,
    )
    def convert_data_tool(text: str = "", from_format: str = "",
                          to_format: str = "") -> dict[str, Any]:
        if not text or not str(text).strip():
            raise ValueError("convert_data needs text")
        result = convert(str(text), str(from_format or "").strip(),
                         str(to_format or "").strip())
        return _reply(result)
