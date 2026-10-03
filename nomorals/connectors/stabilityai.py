"""Stability AI connector — paid Stable Diffusion image generation.

Drives the v2beta stable-image endpoints
(https://platform.stability.ai/docs/api-reference#tag/Generate) with the
framework HttpClient — no SDK dependency.

* text-to-image: ``POST
  https://api.stability.ai/v2beta/stable-image/generate/sd3``
  (multipart form: ``prompt``, ``model``, ``aspect_ratio`` or
  ``height``/``width``, ``output_format``, ``seed``) with
  ``Accept: application/json`` — returns ``{"image": <base64>, "seed": ...,
  "finish_reason": "SUCCESS"}``. If the provider answers with raw bytes
  instead, those are used directly.
* image-to-image: same endpoint with ``mode=image-to-image``, the ``image``
  file field, and ``strength``.

Auth: API key (``API_KEY``) from the Stability AI platform.
Env var ``STABILITY_API_KEY``. Paid-tier pricing applies; generation costs
real credits and is confirmation-gated.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import tempfile
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

__all__ = ["StabilityAIConnector", "StabilityAIError"]

_log = get_logger(__name__)

API_BASE = "https://api.stability.ai"
KEY_ENV = "STABILITY_API_KEY"
DOCS_URL = "https://platform.stability.ai/docs/api-reference"

MODELS = ("sd3.5-large", "sd3.5-large-turbo", "sd3.5-medium", "sd3-large-turbo")
ASPECT_RATIOS = {
    "1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"
}
OUTPUT_FORMATS = {"png", "jpeg", "webp"}


class StabilityAIError(ConnectorError):
    """A Stability AI call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class StabilityAIConnector(Connector):
    """Devon's paid Stability AI (Stable Diffusion) adapter."""

    id = "stability_ai"
    name = "Stability AI"
    description = (
        "Stability AI Stable Diffusion (SD 3.x): text-to-image and "
        "image-to-image via the v2beta stable-image API. Opt-in paid "
        "upgrade — costs credits; designed as the fallback image provider."
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
        """Validate a Stability AI API key and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "stability_ai is already connected — one account per "
                "service. Disconnect first to switch keys."
            )
        key = self._resolve_key(api_key)
        balance = self._api("GET", "/v1/user/balance", key=key)
        account = f"stability-user ({balance.get('credits', '?')} credits)"
        self._store_credential(
            account,
            key,
            credential_type="api_key",
            scopes=["image.generate", "image.edit"],
            metadata={"models": list(MODELS)},
        )
        _log.info("stability_ai connected")
        return ConnectResult(
            ok=True,
            account=account,
            scopes=["image.generate", "image.edit"],
            message=(
                f"connected to Stability AI ({account}). The key is in the "
                "encrypted vault. Generation is PAID — every generate/edit "
                "call is gated behind explicit owner confirmation."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — set STABILITY_API_KEY and run "
                       "`nm connectors connect --name stability_ai`",
            )
        try:
            balance = self._api("GET", "/v1/user/balance")
        except StabilityAIError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"API key rejected ({exc}): create a new key on "
                       "platform.stability.ai and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account=cred.username,
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"key accepted; balance "
                   f"{balance.get('credits', '?')} credits",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/v1/user/balance")
            return True
        except ConnectorError:
            return False

    # ── capability map ─────────────────────────────────────────

    def capabilities(self) -> dict[str, Any]:
        """Exactly what this connector can and cannot do, in one place."""
        return {
            "side": "paid image generation",
            "auth": "Bearer API key (Authorization: Bearer)",
            "can": {
                "generate_image": "text-to-image on SD 3.x; aspect ratio or "
                                  "explicit size, seed, negative prompt",
                "edit_image": "image-to-image via mode=image-to-image + "
                              "strength",
                "balance": "live credit balance via /v1/user/balance",
            },
            "cannot": {
                "video": "images only — no video models here",
                "negative_on_default_cfg": "negative_prompt needs "
                                           "guidance_scale > 1 to take "
                                           "effect",
            },
            "cost_note": "paid credits on the owner's Stability AI account",
        }

    # ── Stability AI API ───────────────────────────────────────

    def generate_image(
        self,
        prompt: str,
        *,
        model: str = "sd3.5-large",
        aspect_ratio: str = "16:9",
        output_format: str = "png",
        seed: int = 0,
        negative_prompt: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Text-to-image (``POST /v2beta/stable-image/generate/sd3``).

        Returns ``{"image_bytes": bytes, "mime_type": str, "seed": int,
        "finish_reason": str}``. Synchronous — the API answers with the
        finished image. Consequential (costs credits): gated behind explicit
        owner confirmation.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ConnectorError("a prompt is required")
        if model not in MODELS:
            raise ConnectorError(
                f"model must be one of {list(MODELS)}, got {model!r}"
            )
        if aspect_ratio not in ASPECT_RATIOS:
            raise ConnectorError(
                f"aspect_ratio must be one of {sorted(ASPECT_RATIOS)}, "
                f"got {aspect_ratio!r}"
            )
        if output_format not in OUTPUT_FORMATS:
            raise ConnectorError(
                f"output_format must be one of {sorted(OUTPUT_FORMATS)}, "
                f"got {output_format!r}"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="generate_image",
            title="Generate image via Stability AI (paid)",
            instructions="\n".join([
                "Devon wants to generate an image via your paid Stability AI "
                "key.",
                "Review it — generation costs real credits.",
                f"Model: {model} · {aspect_ratio} · {output_format}",
                "",
                prompt,
            ]),
            resume_state={"prompt": prompt, "model": model,
                          "aspect_ratio": aspect_ratio,
                          "output_format": output_format, "seed": seed},
        )
        fields = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "seed": str(seed),
        }
        if negative_prompt.strip():
            fields["negative_prompt"] = negative_prompt.strip()
        return self._generate(fields, files=None, mime=output_format)

    def edit_image(
        self,
        image: bytes | str | Path,
        prompt: str,
        *,
        model: str = "sd3.5-large",
        strength: float = 0.5,
        output_format: str = "png",
        seed: int = 0,
        negative_prompt: str = "",
        mime_type: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Image-to-image (``mode=image-to-image`` + ``strength``).

        ``image`` may be raw bytes, a path string, or a ``Path``.
        ``strength`` (0-1): higher deviates more from the input. Returns
        the same shape as :meth:`generate_image`. Consequential (costs
        credits): gated behind explicit owner confirmation.
        """
        if not 0.0 <= strength <= 1.0:
            raise ConnectorError(f"strength must be 0-1, got {strength}")
        if model not in MODELS:
            raise ConnectorError(
                f"model must be one of {list(MODELS)}, got {model!r}"
            )
        if output_format not in OUTPUT_FORMATS:
            raise ConnectorError(
                f"output_format must be one of {sorted(OUTPUT_FORMATS)}, "
                f"got {output_format!r}"
            )
        prompt = (prompt or "").strip()
        if not prompt:
            raise ConnectorError("a prompt is required")
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="edit_image",
            title="Edit image via Stability AI (paid)",
            instructions="\n".join([
                "Devon wants to edit an image via your paid Stability AI "
                "key (image-to-image).",
                "Review it — generation costs real credits.",
                f"Model: {model} · strength: {strength} · {output_format}",
                "",
                prompt,
            ]),
            resume_state={"prompt": prompt, "model": model,
                          "strength": strength},
        )
        raw, mime = self._read_image(image, mime_type)
        fields = {
            "prompt": prompt,
            "mode": "image-to-image",
            "model": model,
            "strength": str(strength),
            "output_format": output_format,
            "seed": str(seed),
        }
        if negative_prompt.strip():
            fields["negative_prompt"] = negative_prompt.strip()
        return self._generate(
            fields, files=(raw, mime), mime=output_format
        )

    def balance(self) -> dict[str, Any]:
        """Live credit balance (``GET /v1/user/balance``)."""
        data = self._api("GET", "/v1/user/balance")
        return data if isinstance(data, dict) else {}

    # ── HTTP plumbing ──────────────────────────────────────────

    def _require_credential(self):  # type: ignore[no-untyped-def]
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "stability_ai is not connected — set STABILITY_API_KEY and "
                "run `nm connectors connect --name stability_ai` first"
            )
        return cred

    @staticmethod
    def _resolve_key(override: str | None) -> str:
        key = (override or "").strip() or os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise ConnectorError(
                f"no API key: set the {KEY_ENV} environment variable to "
                "your Stability AI API key (from platform.stability.ai)"
            )
        return key

    def _api(
        self,
        method: str,
        path: str,
        *,
        key: str | None = None,
    ) -> Any:
        """One Stability AI REST call; failures become StabilityAIError."""
        cred = self._require_credential() if key is None else None
        api_key = key if key is not None else cred.password
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers)
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise StabilityAIError(
                "stability rate limited (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise StabilityAIError(
                f"stability request failed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise StabilityAIError(f"stability request failed: {exc}") from exc
        self._check(resp, method, path)
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise StabilityAIError(
                f"stability {method} {path} returned invalid JSON"
            ) from exc

    def _generate(
        self,
        fields: dict[str, str],
        files: tuple[bytes, str] | None,
        *,
        mime: str,
    ) -> dict[str, Any]:
        """One v2beta generate call; returns decoded image bytes."""
        cred = self._require_credential()
        url = f"{API_BASE}/v2beta/stable-image/generate/sd3"
        headers = {
            "Authorization": f"Bearer {cred.password}",
            "Accept": "application/json",
        }
        tmp: str | None = None
        try:
            files_arg: list[tuple[str, str, str]] = []
            if files is not None:
                raw, fmime = files
                extension = (fmime.split("/")[-1] or "png").lower()
                with tempfile.NamedTemporaryFile(
                    suffix=f".{extension}", delete=False
                ) as handle:
                    handle.write(raw)
                    tmp = handle.name
                files_arg = [("image", tmp, fmime)]
            try:
                resp = self.http.post_multipart(
                    url, fields=fields, files=files_arg, headers=headers
                )
            except RateLimited as exc:
                raise StabilityAIError(
                    "stability rate limited (429) — back off "
                    f"~{exc.retry_after:.0f}s before retrying",
                    status_code=429,
                ) from exc
            except NoMoralsError as exc:
                raise StabilityAIError(
                    f"stability generate failed: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - network opaque
                raise StabilityAIError(
                    f"stability generate failed: {exc}"
                ) from exc
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        self._check(resp, "POST", "/v2beta/stable-image/generate/sd3")
        content_type = str(
            (resp.headers or {}).get("content-type", "")
        ).lower()
        if "json" in content_type:
            try:
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 - invalid JSON error
                raise StabilityAIError(
                    "stability returned invalid JSON"
                ) from exc
            if not isinstance(data, dict) or not data.get("image"):
                raise StabilityAIError(
                    f"stability returned no image: {resp.text[:200]}"
                )
            try:
                image_bytes = base64.b64decode(data["image"])
            except Exception as exc:  # noqa: BLE001 - bad payload
                raise StabilityAIError(
                    f"stability returned undecodable image data: {exc}"
                ) from exc
            return {
                "image_bytes": image_bytes,
                "mime_type": f"image/{mime}",
                "seed": int(data.get("seed", 0) or 0),
                "finish_reason": str(data.get("finish_reason", "")),
            }
        if not resp.body:
            raise StabilityAIError("stability returned empty image bytes")
        return {
            "image_bytes": bytes(resp.body),
            "mime_type": content_type.split(";")[0].strip() or f"image/{mime}",
            "seed": 0,
            "finish_reason": "SUCCESS",
        }

    def _check(self, resp: Any, method: str, path: str) -> None:
        """Raise StabilityAIError on bad status; unwraps the error JSON."""
        if resp.status == 401:
            raise StabilityAIError(
                "stability rejected the API key (401): create a new key on "
                "platform.stability.ai and reconnect",
                status_code=401,
            )
        if resp.status == 402:
            raise StabilityAIError(
                "stability refused (402): out of credits — top up at "
                "platform.stability.ai",
                status_code=402,
            )
        if resp.status == 429:
            raise StabilityAIError(
                "stability rate limited (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            raise StabilityAIError(
                f"stability {method} {path} failed ({resp.status}): "
                f"{self._error_detail(resp) or resp.text[:200]}",
                status_code=resp.status,
            )

    @staticmethod
    def _error_detail(resp: Any) -> str:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            return resp.text[:200]
        if isinstance(body, dict):
            for key in ("message", "error"):
                value = body.get(key)
                if isinstance(value, str) and value:
                    return value[:200]
        return resp.text[:200]

    @staticmethod
    def _read_image(
        image: bytes | str | Path, mime_type: str
    ) -> tuple[bytes, str]:
        """Raw bytes + mime from bytes or a file path."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            raw = path.read_bytes()
            mime = mime_type or mimetypes.guess_type(path.name)[0] or \
                "image/png"
        else:
            raw = bytes(image)
            mime = mime_type or "image/png"
        if not raw:
            raise ConnectorError("empty image bytes — nothing to edit")
        return raw, mime
