"""Vision: image understanding.

Description comes from whichever registered provider supports vision. Image
metadata (dimensions, format, size) is parsed from headers directly, so the tool
still reports something useful when no VLM is configured — which is the offline
and Termux case.

Results are cached by content hash: describing the same screenshot twice should
not cost two model calls.
"""

from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.policy import Capability

__all__ = ["describe", "image_metadata", "register"]


def image_metadata(data: bytes) -> dict[str, Any]:
    """Format, dimensions, and size — parsed from headers, no PIL needed."""
    meta: dict[str, Any] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        meta["format"] = "png"
        if len(data) >= 24:
            meta["width"] = int.from_bytes(data[16:20], "big")
            meta["height"] = int.from_bytes(data[20:24], "big")
            meta["bit_depth"] = data[24] if len(data) > 24 else 0
    elif data.startswith(b"\xff\xd8\xff"):
        meta["format"] = "jpeg"
        width = height = 0
        index = 2
        while index + 9 <= len(data):  # need 9 bytes from index; '<' dropped the last frame
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3}:
                height = int.from_bytes(data[index + 5 : index + 7], "big")
                width = int.from_bytes(data[index + 7 : index + 9], "big")
                break
            length = int.from_bytes(data[index + 2 : index + 4], "big")
            index += 2 + max(2, length)
        meta["width"], meta["height"] = width, height
    elif data.startswith(b"GIF8"):
        meta["format"] = "gif"
        if len(data) >= 10:
            meta["width"] = int.from_bytes(data[6:8], "little")
            meta["height"] = int.from_bytes(data[8:10], "little")
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        meta["format"] = "webp"
    elif data.startswith(b"BM"):
        meta["format"] = "bmp"
        if len(data) >= 26:
            meta["width"] = int.from_bytes(data[18:22], "little")
            meta["height"] = abs(int.from_bytes(data[22:26], "little", signed=True))
    else:
        meta["format"] = "unknown"
    return meta


def describe(
    context: Any,
    data: bytes,
    prompt: str = "",
    *,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe an image with the best available vision provider."""
    metadata = image_metadata(data)
    key = metadata["sha256"]
    if cache is not None and key in cache:
        return {**cache[key], "cached": True}

    instruction = prompt or (
        "Describe this image thoroughly: layout, text, objects, and anything notable. "
        "Transcribe any visible text verbatim."
    )
    started = time.perf_counter()
    router = getattr(context, "router", None) if context is not None else None
    description = ""
    provider = ""
    if router is not None:
        try:
            response = router.describe_image(data, instruction)
            description = response.text
            provider = response.provider
        except Exception as exc:  # noqa: BLE001 - vision is optional, metadata is not
            description = f"[vision unavailable: {type(exc).__name__}]"

    result = {
        **metadata,
        "description": description,
        "provider": provider,
        "prompt": instruction,
        "seconds": round(time.perf_counter() - started, 3),
    }
    if cache is not None:
        cache[key] = result
    return result


def register(registry: Any) -> None:
    """Attach the vision tools to a registry."""
    context = registry.context
    cache: dict[str, dict[str, Any]] = {}

    def _load(path: str = "", url: str = "") -> bytes:
        if path:
            from .filesystem import safe_path

            target = safe_path(context, path, must_exist=True)
            return target.read_bytes()
        if url:
            from ..core.http import HttpClient

            return HttpClient(timeout=60.0).get(url).body
        raise ToolError("vision_describe needs a path or a url")

    @registry.register(
        "vision_describe",
        description="Describe an image (from a workspace path or URL) using a vision model.",
        capability=Capability.MODEL_CALL,
    )
    def vision_describe(path: str = "", url: str = "", prompt: str = "") -> dict[str, Any]:
        return describe(context, _load(path, url), prompt, cache=cache)

    @registry.register(
        "vision_metadata",
        description="Read image format, dimensions, and size without calling a model.",
        capability=Capability.FS_READ,
    )
    def vision_metadata(path: str = "", url: str = "") -> dict[str, Any]:
        return image_metadata(_load(path, url))

    @registry.register(
        "vision_to_base64",
        description="Encode an image as a data URL, for passing to an API by hand.",
        capability=Capability.FS_READ,
    )
    def vision_to_base64(path: str = "", url: str = "", mime: str = "image/png") -> dict[str, Any]:
        data = _load(path, url)
        encoded = base64.b64encode(data).decode("ascii")
        return {"mime": mime, "bytes": len(data), "data_url": f"data:{mime};base64,{encoded}"}
