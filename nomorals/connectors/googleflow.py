"""Google Flow connector — paid Google video generation via Veo.

Drives Google's Veo models (the engine behind Google Flow) through the
Gemini Developer API (https://ai.google.dev/gemini-api/docs/video) with the
framework HttpClient — no ``google-genai`` SDK dependency.

Video generation is asynchronous:

1. ``POST
   https://generativelanguage.googleapis.com/v1beta/models/{model}:predictLongRunning``
   with ``{"instances": [{"prompt": ...}], "parameters": {...}}`` — returns a
   long-running operation name.
2. ``GET https://generativelanguage.googleapis.com/v1beta/{operationName}``
   until ``done`` is true.
3. The video URI lives at
   ``response.generateVideoResponse.generatedSamples[0].video.uri`` — the
   file is downloaded with the ``x-goog-api-key`` header (server keeps it for
   2 days).

Models: ``veo-3.1-fast-generate-preview`` (default, balanced),
``veo-3.1-lite-generate-preview`` (cheapest), ``veo-3.1-generate-preview``
(highest quality), plus the non-preview ``veo-3.0-generate-001`` stable id.
Clips are 4/6/8 seconds; 1080p/4K need 8s. Veo is PAID — billed per second
of output — and requires an API key with paid-tier access (a billing-enabled
Google Cloud project linked in AI Studio).

Auth: API key (``API_KEY``) from Google AI Studio
(https://aistudio.google.com/apikey). Env var ``GOOGLE_FLOW_API_KEY`` (falls
back to ``GEMINI_API_KEY`` — the same Google key).
"""

from __future__ import annotations

import base64
import os
import time
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

__all__ = ["GoogleFlowConnector", "GoogleFlowError"]

_log = get_logger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY_ENV = "GOOGLE_FLOW_API_KEY"
FALLBACK_KEY_ENV = "GEMINI_API_KEY"
DOCS_URL = "https://ai.google.dev/gemini-api/docs/video"

MODEL_FAST = "veo-3.1-fast-generate-preview"
MODEL_LITE = "veo-3.1-lite-generate-preview"
MODEL_FULL = "veo-3.1-generate-preview"
MODEL_STABLE = "veo-3.0-generate-001"

ASPECT_RATIOS = {"16:9", "9:16"}
DURATIONS = {4, 6, 8}
POLL_INTERVAL_S = 10.0
DEFAULT_TIMEOUT_S = 600.0


class GoogleFlowError(ConnectorError):
    """A Google Flow (Veo) call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class GoogleFlowConnector(Connector):
    """Devon's paid Google video-generation adapter (Flow / Veo)."""

    id = "google_flow"
    name = "Google Flow"
    description = (
        "Google's video generation (Veo 3.x, the engine behind Flow): "
        "text-to-video and image-to-video via the Gemini Developer API. "
        "Opt-in paid upgrade — billed per second of output."
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
                "google_flow is already connected — one account per service. "
                "Disconnect first to switch keys."
            )
        key = self._resolve_key(api_key)
        self._validate_key(key)
        self._store_credential(
            "ai-studio-key",
            key,
            credential_type="api_key",
            scopes=["video.generate"],
            metadata={"models": [MODEL_FAST, MODEL_LITE, MODEL_FULL,
                                 MODEL_STABLE]},
        )
        _log.info("google_flow connected")
        return ConnectResult(
            ok=True,
            account="ai-studio-key",
            scopes=["video.generate"],
            message=(
                "connected to Google AI Studio (Flow/Veo). The key is in the "
                "encrypted vault. Video generation is PAID — every render is "
                "gated behind explicit owner confirmation. Veo also needs a "
                "billing-enabled key; if renders fail with 403, link a "
                "billing project in AI Studio."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — set GOOGLE_FLOW_API_KEY and run "
                       "`nm connectors connect --name google_flow`",
            )
        try:
            self._validate_key(cred.password)
        except GoogleFlowError as exc:
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
            detail="key accepted; Veo predictLongRunning models reachable",
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
            "side": "paid video generation",
            "auth": "AI Studio API key (x-goog-api-key header)",
            "can": {
                "generate_video": "text-to-video; aspect ratio, resolution, "
                                  "duration, negative prompt",
                "image_to_video": "animate from a first-frame image",
                "status": "live key validation via GET /v1beta/models/{model}",
            },
            "cannot": {
                "sync_render": "all renders are async — poll until done",
                "over_8s": "clips are 4/6/8 seconds max per render",
            },
            "cost_note": "billed per second of output on the owner's Google "
                         "key; requires paid-tier access",
        }

    # ── Veo API ────────────────────────────────────────────────

    def generate_video(
        self,
        prompt: str,
        *,
        model: str = MODEL_FAST,
        duration_seconds: int = 4,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: str = "",
        first_frame: bytes | None = None,
        first_frame_mime: str = "image/jpeg",
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Render a video (``POST ...:predictLongRunning`` + poll).

        Returns ``{"video_bytes": bytes, "video_uri": str, "model": str,
        "duration_seconds": int, "elapsed_seconds": float}``. Blocks while
        polling the operation (``timeout_seconds``). Consequential (costs
        money per second of output): gated behind explicit owner
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
        if duration_seconds not in DURATIONS:
            raise ConnectorError(
                f"duration_seconds must be one of {sorted(DURATIONS)}, "
                f"got {duration_seconds}"
            )
        if resolution == "4K" and duration_seconds != 8:
            raise ConnectorError(
                "4K renders require duration_seconds=8"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="generate_video",
            title="Render video via Google Flow (paid)",
            instructions="\n".join([
                "Devon wants to render a video via your paid Google Flow "
                "(Veo) key.",
                "Review it — Veo bills real money PER SECOND of output.",
                f"Model: {model} · {duration_seconds}s · {aspect_ratio} · "
                f"{resolution}" + (" · image-to-video" if first_frame else ""),
                "",
                prompt,
            ]),
            resume_state={"prompt": prompt, "model": model,
                          "duration_seconds": duration_seconds,
                          "aspect_ratio": aspect_ratio,
                          "resolution": resolution},
        )
        instance: dict[str, Any] = {"prompt": prompt}
        if first_frame:
            if not first_frame_mime.startswith("image/"):
                raise ConnectorError(
                    "first_frame_mime must be an image/* type"
                )
            instance["image"] = {
                "inlineData": {
                    "mimeType": first_frame_mime,
                    "data": base64.b64encode(first_frame).decode("ascii"),
                }
            }
        parameters: dict[str, Any] = {
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "durationSeconds": str(duration_seconds),
        }
        if negative_prompt.strip():
            parameters["negativePrompt"] = negative_prompt.strip()
        started = time.monotonic()
        operation_name = self._submit(model, instance, parameters)
        uri = self._poll_until_done(operation_name, timeout_seconds, started)
        video_bytes = self._download_video(uri)
        return {
            "video_bytes": video_bytes,
            "video_uri": uri,
            "model": model,
            "duration_seconds": duration_seconds,
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }

    # ── HTTP plumbing ──────────────────────────────────────────

    def _require_credential(self):  # type: ignore[no-untyped-def]
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "google_flow is not connected — set GOOGLE_FLOW_API_KEY and "
                "run `nm connectors connect --name google_flow` first"
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
        """A key is valid when the Veo model lookup accepts it."""
        url = f"{API_BASE}/models/{MODEL_FAST}"
        try:
            resp = self.http.get(url, headers=self._headers(key))
        except RateLimited as exc:
            raise GoogleFlowError(
                "google rate limited (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise GoogleFlowError(
                f"key validation request failed: {exc}"
            ) from exc
        if resp.status in (400, 401, 403):
            raise GoogleFlowError(
                "google rejected the API key "
                f"({resp.status}) — mint a fresh key at "
                "https://aistudio.google.com/apikey",
                status_code=resp.status,
            )
        if not resp.ok:
            raise GoogleFlowError(
                f"key validation failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )

    def _submit(
        self,
        model: str,
        instance: dict[str, Any],
        parameters: dict[str, Any],
    ) -> str:
        """Start a render; returns the operation name."""
        cred = self._require_credential()
        url = f"{API_BASE}/models/{model}:predictLongRunning"
        try:
            resp = self.http.post_json(
                url,
                {"instances": [instance], "parameters": parameters},
                headers=self._headers(cred.password),
            )
        except RateLimited as exc:
            raise GoogleFlowError(
                "google rate limited (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise GoogleFlowError(f"render submit failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GoogleFlowError(f"render submit failed: {exc}") from exc
        if resp.status == 403:
            raise GoogleFlowError(
                "google refused the render (403) — Veo needs a billing-"
                "enabled AI Studio key; link a billing project in AI Studio "
                "and try again",
                status_code=403,
            )
        if resp.status in (400, 401):
            raise GoogleFlowError(
                f"google rejected the render ({resp.status}): "
                f"{self._error_detail(resp)}",
                status_code=resp.status,
            )
        if not resp.ok:
            raise GoogleFlowError(
                f"render submit failed ({resp.status}): "
                f"{self._error_detail(resp) or resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            name = resp.json().get("name", "")
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise GoogleFlowError(
                "veo returned invalid JSON on submit"
            ) from exc
        if not name:
            raise GoogleFlowError(
                f"veo submit returned no operation name: {resp.text[:200]}"
            )
        return str(name)

    def _poll_until_done(
        self,
        operation_name: str,
        timeout_seconds: float,
        started: float,
    ) -> str:
        """Poll the operation until done; returns the video URI."""
        cred = self._require_credential()
        url = f"{API_BASE}/{operation_name.lstrip('/')}"
        headers = self._headers(cred.password)
        deadline = started + max(1.0, timeout_seconds)
        while True:
            try:
                resp = self.http.get(url, headers=headers)
            except RateLimited as exc:
                raise GoogleFlowError(
                    "veo poll rate limited (429) — back off "
                    f"~{exc.retry_after:.0f}s before retrying",
                    status_code=429,
                ) from exc
            except NoMoralsError as exc:
                raise GoogleFlowError(
                    f"operation poll failed: {exc}"
                ) from exc
            if not resp.ok:
                raise GoogleFlowError(
                    f"operation poll failed ({resp.status}): "
                    f"{self._error_detail(resp) or resp.text[:200]}",
                    status_code=resp.status,
                )
            try:
                op = resp.json()
            except Exception as exc:  # noqa: BLE001 - invalid JSON is error
                raise GoogleFlowError(
                    "veo returned invalid JSON while polling"
                ) from exc
            if op.get("done"):
                if isinstance(op.get("error"), dict):
                    message = str(
                        op["error"].get("message", "render failed")
                    )[:300]
                    raise GoogleFlowError(
                        f"veo render failed: {message}"
                    )
                uri = self._extract_uri(op)
                if not uri:
                    raise GoogleFlowError(
                        "veo operation finished without a video URI: "
                        f"{resp.text[:300]}"
                    )
                return uri
            if time.monotonic() > deadline:
                raise GoogleFlowError(
                    f"veo render timed out after "
                    f"{timeout_seconds:.0f}s (operation {operation_name}); "
                    "poll the operation again later with a longer timeout"
                )
            time.sleep(POLL_INTERVAL_S)

    @staticmethod
    def _extract_uri(op: dict[str, Any]) -> str:
        try:
            samples = op["response"]["generateVideoResponse"][
                "generatedSamples"]
            uri = samples[0]["video"]["uri"]
        except (KeyError, IndexError, TypeError):
            return ""
        return str(uri or "")

    def _download_video(self, uri: str) -> bytes:
        """Download the finished video (auth header, redirects followed)."""
        cred = self._require_credential()
        try:
            resp = self.http.get(uri, headers=self._headers(cred.password))
        except NoMoralsError as exc:
            raise GoogleFlowError(
                f"video download failed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GoogleFlowError(
                f"video download failed: {exc}"
            ) from exc
        if not resp.ok:
            raise GoogleFlowError(
                f"video download failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        if not resp.body:
            raise GoogleFlowError("video download returned empty bytes")
        return bytes(resp.body)

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
