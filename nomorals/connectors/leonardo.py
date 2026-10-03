"""Leonardo AI connector — paid image generation via the Leonardo API.

Drives https://cloud.leonardo.ai/api/rest/v1 with the framework HttpClient
— no SDK dependency.

Flow (all JSON, ``Authorization: Bearer <key>``):

* text-to-image: ``POST /generations`` (``prompt``, ``num_images``,
  ``width``, ``height``, ``modelId``) → ``sdGenerationJob.generationId``,
  then poll ``GET /generations/{id}`` until
  ``generations_by_pk.generated_images`` is populated with ``url``s.
* image-to-image: ``POST /init-image`` (``{"extension": "png"}``) →
  ``uploadInitImage {id, fields (JSON string), url}``; multipart-upload the
  file to that presigned URL (fields + ``file`` part); then ``POST
  /generations`` with ``initImageId`` and ``initStrength``.
* key test: ``GET /me`` returns the user record.

Auth: API key (``API_KEY``) minted on the Leonardo API Access page.
Env var ``LEONARDO_API_KEY``. Paid-tier pricing applies; generation costs
real API credits and is confirmation-gated.
"""

from __future__ import annotations

import json
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

__all__ = ["LeonardoConnector", "LeonardoError"]

_log = get_logger(__name__)

API_BASE = "https://cloud.leonardo.ai/api/rest/v1"
KEY_ENV = "LEONARDO_API_KEY"
DOCS_URL = "https://docs.leonardo.ai"

POLL_INTERVAL_S = 8.0
DEFAULT_TIMEOUT_S = 600.0
MIN_DIMENSION = 32
MAX_DIMENSION = 2048


class LeonardoError(ConnectorError):
    """A Leonardo API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class LeonardoConnector(Connector):
    """Devon's paid Leonardo AI image-generation adapter."""

    id = "leonardo"
    name = "Leonardo AI"
    description = (
        "Leonardo AI image generation: text-to-image plus image-to-image via "
        "init-image upload, polled until the job completes. Opt-in paid "
        "upgrade — costs API credits."
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
        """Validate a Leonardo API key and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "leonardo is already connected — one account per service. "
                "Disconnect first to switch keys."
            )
        key = self._resolve_key(api_key)
        me = self._api("GET", "/me", key=key)
        username = str(me.get("username") or me.get("id") or "leonardo-user")
        self._store_credential(
            username,
            key,
            credential_type="api_key",
            scopes=["image.generate", "image.edit"],
            metadata={"user_id": str(me.get("id", ""))},
        )
        _log.info("leonardo connected for user %s", username)
        return ConnectResult(
            ok=True,
            account=username,
            scopes=["image.generate", "image.edit"],
            message=(
                f"connected to Leonardo AI as {username}. The key is in the "
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
                detail="not connected — set LEONARDO_API_KEY and run "
                       "`nm connectors connect --name leonardo`",
            )
        try:
            self._api("GET", "/me")
        except LeonardoError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"API key rejected ({exc}): create a new key on the "
                       "Leonardo API Access page and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account=cred.username,
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="key accepted; /me reachable",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/me")
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
                "generate_image": "text-to-image; size, count, model choice",
                "edit_image": "image-to-image via init-image presigned "
                              "upload + initStrength",
                "get_generation": "fetch a past generation by id",
            },
            "cannot": {
                "canvas": "AI Canvas inpainting is web-app only — not in "
                          "this connector",
                "sync_render": "generations are async — poll until done",
            },
            "cost_note": "paid API credits on the owner's Leonardo plan",
        }

    # ── Leonardo API ─────────────────────────────────────────

    def generate_image(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        model_id: str = "",
        negative_prompt: str = "",
        seed: int = 0,
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> list[dict[str, Any]]:
        """Submit a text-to-image job and wait for it (``POST /generations``).

        Returns a list of ``{"id": str, "url": str}``. Consequential (costs
        credits): gated behind explicit owner confirmation.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ConnectorError("a prompt is required")
        for label, value in (("width", width), ("height", height)):
            if not (MIN_DIMENSION <= value <= MAX_DIMENSION):
                raise ConnectorError(
                    f"{label} must be {MIN_DIMENSION}-{MAX_DIMENSION}, "
                    f"got {value}"
                )
        if not 1 <= num_images <= 8:
            raise ConnectorError(
                f"num_images must be 1-8, got {num_images}"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="generate_image",
            title="Generate image via Leonardo AI (paid)",
            instructions="\n".join([
                "Devon wants to generate an image via your paid Leonardo AI "
                "key.",
                "Review it — generation costs real API credits.",
                f"{num_images}× {width}×{height}"
                + (f" · model {model_id}" if model_id else ""),
                "",
                prompt,
            ]),
            resume_state={"prompt": prompt, "width": width, "height": height,
                          "num_images": num_images, "model_id": model_id},
        )
        payload: dict[str, Any] = {
            "prompt": prompt,
            "num_images": num_images,
            "width": width,
            "height": height,
        }
        if model_id:
            payload["modelId"] = model_id
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()
        if seed:
            payload["seed"] = seed
        data = self._api("POST", "/generations", payload=payload)
        generation_id = self._generation_id(data)
        return self._wait_for_generation(generation_id, timeout_seconds)

    def edit_image(
        self,
        image: bytes | str | Path,
        prompt: str,
        *,
        init_strength: float = 0.5,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        model_id: str = "",
        mime_type: str = "",
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> list[dict[str, Any]]:
        """Image-to-image: upload an init image, then generate from it.

        ``image`` may be raw bytes, a path string, or a ``Path``.
        ``init_strength`` (0-1): higher keeps more of the input. Returns
        the same shape as :meth:`generate_image`. Consequential (costs
        credits): gated behind explicit owner confirmation.
        """
        if not 0.0 <= init_strength <= 1.0:
            raise ConnectorError(
                f"init_strength must be 0-1, got {init_strength}"
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
            title="Edit image via Leonardo AI (paid)",
            instructions="\n".join([
                "Devon wants to edit an image via your paid Leonardo AI "
                "key (init-image upload).",
                "Review it — generation costs real API credits.",
                f"init strength: {init_strength} · {width}×{height}",
                "",
                prompt,
            ]),
            resume_state={"prompt": prompt,
                          "init_strength": init_strength},
        )
        init_image_id = self._upload_init_image(image, mime_type)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "num_images": max(1, min(num_images, 8)),
            "width": width,
            "height": height,
            "initImageId": init_image_id,
            "initStrength": init_strength,
        }
        if model_id:
            payload["modelId"] = model_id
        data = self._api("POST", "/generations", payload=payload)
        generation_id = self._generation_id(data)
        return self._wait_for_generation(generation_id, timeout_seconds)

    def get_generation(self, generation_id: str) -> list[dict[str, Any]]:
        """Fetch one generation by id (``GET /generations/{id}``)."""
        generation_id = (generation_id or "").strip()
        if not generation_id:
            raise ConnectorError("a generation id is required")
        data = self._api("GET", f"/generations/{generation_id}")
        return self._image_entries(data)

    # ── HTTP plumbing ──────────────────────────────────────────

    def _require_credential(self):  # type: ignore[no-untyped-def]
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "leonardo is not connected — set LEONARDO_API_KEY and run "
                "`nm connectors connect --name leonardo` first"
            )
        return cred

    @staticmethod
    def _resolve_key(override: str | None) -> str:
        key = (override or "").strip() or os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise ConnectorError(
                f"no API key: set the {KEY_ENV} environment variable to "
                "your Leonardo API key (create one on the Leonardo API "
                "Access page)"
            )
        return key

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        key: str | None = None,
    ) -> Any:
        """One Leonardo REST call; failures become LeonardoError."""
        cred = self._require_credential() if key is None else None
        api_key = key if key is not None else cred.password
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers)
            elif method == "POST":
                resp = self.http.post_json(url, payload or {}, headers=headers)
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise LeonardoError(
                "leonardo rate limited (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise LeonardoError(
                f"leonardo request failed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise LeonardoError(f"leonardo request failed: {exc}") from exc
        if resp.status == 401:
            raise LeonardoError(
                "leonardo rejected the API key (401): create a new key on "
                "the Leonardo API Access page and reconnect",
                status_code=401,
            )
        if resp.status == 403:
            raise LeonardoError(
                "leonardo refused the request (403): out of API credits or "
                "plan limits reached",
                status_code=403,
            )
        if resp.status == 429:
            raise LeonardoError(
                "leonardo rate limited (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            raise LeonardoError(
                f"leonardo {method} {path} failed ({resp.status}): "
                f"{self._error_detail(resp) or resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise LeonardoError(
                f"leonardo {method} {path} returned invalid JSON"
            ) from exc

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
    def _generation_id(data: Any) -> str:
        try:
            gid = data["sdGenerationJob"]["generationId"]
        except (KeyError, TypeError):
            raise LeonardoError(
                f"leonardo returned no generation id: {str(data)[:200]}"
            ) from None
        return str(gid)

    def _wait_for_generation(
        self, generation_id: str, timeout_seconds: float
    ) -> list[dict[str, Any]]:
        """Poll ``GET /generations/{id}`` until images land."""
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while True:
            data = self._api("GET", f"/generations/{generation_id}")
            entries = self._image_entries(data)
            if entries:
                return entries
            if time.monotonic() > deadline:
                raise LeonardoError(
                    f"leonardo generation {generation_id} timed out after "
                    f"{timeout_seconds:.0f}s; fetch it later with "
                    "get_generation()"
                )
            time.sleep(POLL_INTERVAL_S)

    @staticmethod
    def _image_entries(data: Any) -> list[dict[str, Any]]:
        """Extract ``{"id", "url"}`` from a generation response."""
        try:
            images = data["generations_by_pk"]["generated_images"]
        except (KeyError, TypeError):
            return []
        entries = []
        for image in images if isinstance(images, list) else []:
            if isinstance(image, dict) and image.get("url"):
                entries.append({
                    "id": str(image.get("id", "")),
                    "url": str(image["url"]),
                })
        return entries

    def _upload_init_image(
        self, image: bytes | str | Path, mime_type: str
    ) -> str:
        """Upload an init image; returns the init image id."""
        raw: bytes
        if isinstance(image, (str, Path)):
            path = Path(image)
            raw = path.read_bytes()
            mime = mime_type or mimetypes.guess_type(path.name)[0] or \
                "image/png"
        else:
            raw = bytes(image)
            mime = mime_type or "image/png"
        if not raw:
            raise ConnectorError("empty image bytes — nothing to upload")
        extension = (mime.split("/")[-1] or "png").lower()
        data = self._api(
            "POST", "/init-image", payload={"extension": extension}
        )
        try:
            upload = data["uploadInitImage"]
            upload_url = str(upload["url"])
            fields = json.loads(str(upload["fields"]))
            init_id = str(upload["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LeonardoError(
                f"leonardo init-image response had no upload data: "
                f"{str(data)[:200]}"
            ) from exc
        if not isinstance(fields, dict):
            raise LeonardoError(
                "leonardo init-image fields were not an object"
            )
        string_fields = {str(k): str(v) for k, v in fields.items()}
        tmp: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{extension}", delete=False
            ) as handle:
                handle.write(raw)
                tmp = handle.name
            resp = self.http.post_multipart(
                upload_url,
                fields=string_fields,
                files=[("file", tmp, mime)],
            )
        except Exception as exc:  # noqa: BLE001 - upload is opaque
            raise LeonardoError(
                f"init image upload failed: {exc}"
            ) from exc
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        if not resp.ok:
            raise LeonardoError(
                f"init image upload failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        return init_id
