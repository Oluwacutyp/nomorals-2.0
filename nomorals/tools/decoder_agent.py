"""Decoder agent tool for automated decoding operations."""

from __future__ import annotations

from typing import Any


def decoder_agent(data: str = "", format: str = "auto", **kwargs) -> dict[str, Any]:
    """Decode data using various methods."""
    result = {"input": data, "format": format, "decoded": ""}
    
    if format == "auto" or format == "base64":
        try:
            import base64
            result["decoded"] = base64.b64decode(data.encode()).decode()
            result["format"] = "base64"
            return result
        except Exception:
            pass
    
    if format == "auto" or format == "hex":
        try:
            result["decoded"] = bytes.fromhex(data).decode()
            result["format"] = "hex"
            return result
        except Exception:
            pass
    
    result["decoded"] = data
    return result


def register(registry: Any) -> None:
    """Register decoder agent tools with the registry."""
    registry.register(
        "decoder_agent",
        decoder_agent,
        description="Decode data using various methods",
        capability="decoder",
        parameters={
            "data": {"type": "string", "description": "Data to decode"},
            "format": {"type": "string", "description": "Format (auto, base64, hex)"}
        }
    )
