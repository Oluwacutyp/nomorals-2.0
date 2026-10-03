"""Nano Banana connector — paid Google image generation/editing via AI Studio.

Drives Google's Gemini image models (the "Nano Banana" family) through the
Gemini Developer API (https://ai.google.dev/gemini-api/docs/image-generation)
with the framework HttpClient — no ``google-genai`` SDK dependency.

Models:
    ``gemini-2.5-flash-image``      Nano Banana — fast, 1024px
    ``gemini-3-pro-image-preview``  Nano Banana Pro — up to 4K, better text

Both endpoints are ``POST
https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent``
with ``x-goog-api-key`` auth. Text-to-image sends a plain text part; editing
sends the text part plus one or more ``inlineData`` image parts. Generation
config asks for ``responseModalities: ["TEXT","IMAGE"]`` and images come back
as base64 ``inlineData`` parts on the candidates.

Auth: API key (``API_KEY``) from Google AI Studio
(https://aistudio.google.com/apikey). Env var ``NANO_BANANA_API_KEY`` (falls
back to ``GEMINI_API_KEY`` — it is the same Google key). Paid-tier pricing
applies; generation costs real money per image and is confirmation-gated.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from ..core.errors import NoMoralsError, RateLimited
from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["NanoBananaConnector", "NanoBananaError"]

_log = get_logger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY_ENV = "NANO_BANANA_API_KEY"
FALLBACK_KEY_ENV = "GEMINI_API_KEY"
DOCS_URL = "https://ai.google.dev/gemini-api/docs/image-generation"

#: The two Nano Banana image models.
MODEL_FLASH = "gemini-2.5-flash-image"
MODEL_PRO = "gemini-3-pro-image-preview"

ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
}
DEFAULT_TIMEOUT_S = 120.0


class NanoBananaError(ConnectorError):
    """A Nano Banana (Gemini Developer API) call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class NanoBananaConnector(Connector):
    """Devon's paid Google image generation/editing adapter."""

    id = "nano_banana"
    name = "Nano Banana"
    description = (
        "Google AI Studio image generation and editing (gemini-2.5-flash-image "
        "/ gemini-3-pro-image-preview): text-to-image plus image editing via "
        "the Gemini Developer API. Opt-in paid upgrade — costs per image."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        api_key: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate an AI Studio API key and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "nano_banana is already connected — one account per service. "
                "Disconnect first to switch keys."
            )
        key = self._resolve_key(api_key)
        self._validate_key(key)
        self._store_credential(
            "ai-studio-key",
            key,
            credential_type="api_key",
            scopes=["image.generate", "image.edit"],
            metadata={"models": [MODEL_FLASH, MODEL_PRO]},
        )
        _log.info("nano_banana connected")
        return ConnectResult(
            ok=True,
            account="ai-studio-key",
            scopes=["image.generate", "image.edit"],
            message=(
                "connected to Google AI Studio (Nano Banana). The key is in "
                "the encrypted vault. Image generation is PAID — every "
                "generate/edit call is gated behind explicit owner "
                "confirmation."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — set NANO_BANANA_API_KEY and run "
                       "`nm connectors connect --name nano_banana`",
            )
        try:
            self._validate_key(cred.password)
        except NanoBananaError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"API key rejected ({exc}): mint a new key at "
                       "aistudio.google.com/apikey and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account=cred.username,
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="key accepted; flash + pro image models reachable",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._validate_key(cred.password)
            return True
        except ConnectorError:
            return False

    # ── capability map ─────────────────────────────────────────

    def capabilities(self) -> dict[str, Any]:
        """Exactly what this connector can and cannot do, in one place."""
        return {
            "side": "paid image generation/editing",
            "auth": "AI Studio API key (x-goog-api-key header)",
            "can": {
                "generate_image": "text-to-image; aspect ratio + model choice",
                "edit_image": "text + image → image (up to 4 input images)",
                "status": "live key validation via GET /v1beta/models/{model}",
            },
            "cannot": {
                "upscale": "no dedicated upscale endpoint — ask Pro for larger",
                "batch_discount": "every image bills individually",
            },
            "cost_note": "paid per image on the owner's Google AI Studio key",
        }

    # ── Nano Banana API ────────────────────────────────────────

    def generate_image(
        self,
        prompt: str,
        *,
        model: str = MODEL_FLASH,
        aspect_ratio: str = "1:1",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> list[dict[str, Any]]:
        """Generate images from a text prompt.

        Returns a list of ``{"image_bytes": bytes, "mime_type": str,
        "text": str}`` (one entry per image part the model returned).
        Consequential (costs money): gated behind explicit owner
        confirmation.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ConnectorError("a prompt is required")
        if aspect_ratio not in ASPECT_RATIOS:
            raise ConnectorError(
                f"aspect_ratio must be one of {sorted(ASPECT_RATIOS)}, "
                f"got {aspect_ratio!r}"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="generate_image",
            title="Generate image via Nano Banana (paid)",
            instructions="\n".join([
                "Devon wants to generate an image via your paid Nano Banana "
                "(Google AI Studio) key.",
                "Review it — generation costs real money per image.",
                f"Model: {model}",
                f"Aspect ratio: {aspect_ratio}",
                "",
                prompt,
            ]),
            resume_state={"prompt": prompt, "model": model,
                          "aspect_ratio": aspect_ratio},
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio},
            },
        }
        data = self._post_generate(model, body)
        return self._extract_images(data)

    def edit_image(
        self,
        image: bytes | str | Path,
        prompt: str,
        *,
        model: str = MODEL_FLASH,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> list[dict[str, Any]]:
        """Edit an image with a text instruction (image-to-image).

        ``image`` may be raw bytes, a path string, or a ``Path``. Extra
        images can be composed with ``compose_images`` (up to 4 total).
        Returns the same shape as :meth:`generate_image`. Consequential
        (costs money): gated behind explicit owner confirmation.
        """
        parts: list[dict[str, Any]] = [
            {"text": (prompt or "").strip() or "Edit this image."}
        ]
        parts.append(self._image_part(image))
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="edit_image",
            title="Edit image via Nano Banana (paid)",
            instructions="\n".join([
                "Devon wants to edit an image via your paid Nano Banana "
                "(Google AI Studio) key.",
                "Review it — editing costs real money per image.",
                f"Model: {model}",
                "",
                parts[0]["text"],
            ]),
            resume_state={"prompt": parts[0]["text"], "model": model},
        )
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        data = self._post_generate(model, body)
        return self._extract_images(data)

    def compose_images(
        self,
        images: list[bytes | str | Path],
        prompt: str,
        *,
        model: str = MODEL_FLASH,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> list[dict[str, Any]]:
        """Compose up to 4 input images with a text instruction.

        Same shape as :meth:`edit_image`; the first input image part is
        replaced by the composed set.
        """
        if not images:
            raise ConnectorError("at least one input image is required")
        if len(images) > 4:
            raise ConnectorError(
                "Nano Banana composes at most 4 input images, "
                f"got {len(images)}"
            )
        parts: list[dict[str, Any]] = [
            {"text": (prompt or "").strip() or "Compose these images."}
        ]
        parts.extend(self._image_part(img) for img in images)
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="compose_images",
            title=f"Compose {len(images)} images via Nano Banana (paid)",
            instructions="\n".join([
                "Devon wants to compose images via your paid Nano Banana "
                "(Google AI Studio) key.",
                "Review it — generation costs real money per image.",
                f"Model: {model} · input images: {len(images)}",
                "",
                parts[0]["text"],
            ]),
            resume_state={"prompt": parts[0]["text"], "model": model,
                          "image_count": len(images)},
        )
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        data = self._post_generate(model, body)
        return self._extract_images(data)

    # ── HTTP plumbing ──────────────────────────────────────────

    def _require_credential(self):  # type: ignore[no-untyped-def]
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "nano_banana is not connected — set NANO_BANANA_API_KEY and "
                "run `nm connectors connect --name nano_banana` first"
            )
        return cred

    @staticmethod
    def _resolve_key(override: str | None) -> str:
        key = (override or "").strip() or os.environ.get(KEY_ENV, "").strip()
        if not key:
            key = os.environ.get(FALLBACK_KEY_ENV, "").strip()
        if not key:
            raise ConnectorError(
                f"no API key: set the {KEY_ENV} environment variable (or "
                f"{FALLBACK_KEY_ENV}) to your Google AI Studio key from "
                "https://aistudio.google.com/apikey"
            )
        return key

    def _validate_key(self, key: str) -> None:
        """A key is valid when the model lookup accepts it."""
        url = f"{API_BASE}/models/{MODEL_FLASH}"
        try:
            resp = self.http.get(url, headers=self._headers(key))
        except RateLimited as exc:
            raise NanoBananaError(
                "google rate limited (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise NanoBananaError(
                f"key validation request failed: {exc}"
            ) from exc
        if resp.status in (400, 401, 403):
            raise NanoBananaError(
                "google rejected the API key "
                f"({resp.status}) — mint a fresh key at "
                "https://aistudio.google.com/apikey",
                status_code=resp.status,
            )
        if not resp.ok:
            raise NanoBananaError(
                f"key validation failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )

    def _post_generate(self, model: str, body: dict[str, Any]) -> Any:
        cred = self._require_credential()
        url = f"{API_BASE}/models/{model}:generateContent"
        try:
            resp = self.http.post_json(
                url, body, headers=self._headers(cred.password)
            )
        except RateLimited as exc:
            raise NanoBananaError(
                "google rate limited (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise NanoBananaError(
                f"nano banana request failed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise NanoBananaError(
                f"nano banana request failed: {exc}"
            ) from exc
        if resp.status in (400, 401, 403):
            detail = self._error_detail(resp)
            raise NanoBananaError(
                f"google rejected the request ({resp.status}): {detail} — "
                "check the key and its paid-tier quota",
                status_code=resp.status,
            )
        if not resp.ok:
            raise NanoBananaError(
                f"nano banana generate failed ({resp.status}): "
                f"{self._error_detail(resp) or resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise NanoBananaError(
                "nano banana returned invalid JSON"
            ) from exc

    @staticmethod
    def _headers(key: str) -> dict[str, str]:
        return {"x-goog-api-key": key}

    @staticmethod
    def _error_detail(resp: Any) -> str:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            return resp.text[:200]
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return str(error.get("message", ""))[:200]
        return resp.text[:200]

    @staticmethod
    def _image_part(image: bytes | str | Path) -> dict[str, Any]:
        """One ``inlineData`` part from bytes or a file path."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            raw = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
        else:
            raw = bytes(image)
            mime = "image/png"
        if not raw:
            raise ConnectorError("empty image bytes — nothing to edit")
        return {
            "inlineData": {
                "mimeType": mime,
                "data": base64.b64encode(raw).decode("ascii"),
            }
        }

    @staticmethod
    def _extract_images(data: Any) -> list[dict[str, Any]]:
        """Pull every generated image out of a generateContent response."""
        images: list[dict[str, Any]] = []
        texts: list[str] = []
        candidates = data.get("candidates", []) if isinstance(data, dict) \
            else []
        for candidate in candidates:
            parts = (candidate.get("content") or {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData")
                if isinstance(inline, dict) and inline.get("data"):
                    try:
                        images.append({
                            "image_bytes": base64.b64decode(inline["data"]),
                            "mime_type": str(inline.get("mimeType", "image/png")),
                            "text": " ".join(texts),
                        })
                    except Exception as exc:  # noqa: BLE001 - bad payload
                        raise NanoBananaError(
                            f"model returned undecodable image data: {exc}"
                        ) from exc
                elif isinstance(part.get("text"), str):
                    texts.append(part["text"])
        if not images:
            raise NanoBananaError(
                "model returned no image — the prompt may have been blocked: "
                f"{' '.join(texts)[:300] or data}"
            )
        return images
