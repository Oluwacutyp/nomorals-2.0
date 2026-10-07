"""The Seer: vision as a tool, not an agent.

``see(image_path, question)`` takes an image and a question, returns the
vision model's answer as text. The vision model NEVER makes decisions —
it only describes. This is enforced in the system prompt.

Loading strategy:
- Primary: Groq vision API (Llama 3.2 Vision) via the LLM router. No local
  install needed, works on the phone.
- Fallback: local GGUF vision model via Devon's lifecycle/provisioner.
  Lazy-loaded on first use, unloaded after idle timeout (default 60s).

Fail fast: if neither path is available, raises VisionUnavailable with a
clear message — never a silent empty result.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["Seer", "see", "VisionUnavailable", "DEFAULT_IDLE_TIMEOUT"]

#: Seconds of inactivity before the local vision model is unloaded.
DEFAULT_IDLE_TIMEOUT = 60.0

#: System prompt that enforces the sensor role. The vision model describes;
#: it does not decide, plan, or act.
SEER_SYSTEM_PROMPT = (
    "You are a visual sensor, not an agent. Your ONLY job is to describe "
    "what you see in the image and answer the specific question asked. "
    "Do NOT make decisions. Do NOT suggest actions. Do NOT plan. Do NOT "
    "express opinions about what should be done. Describe precisely and "
    "factually: positions, colors, text visible, UI elements, their states. "
    "If asked where something is, give its location descriptively "
    "(e.g. 'top-right corner', 'center of the screen')."
)


class VisionUnavailable(ToolError):
    """Raised when no vision path is available."""
    code = "vision.unavailable"


class Seer:
    """Vision as a callable tool.

    Usage:
        seer = Seer()
        description = seer.see("/path/to/screenshot.png",
                               "Where is the attack button?")
    """

    def __init__(
        self,
        *,
        idle_timeout: float | None = None,
        router: Any | None = None,
    ) -> None:
        from ..core.profiles import profile_value
        self.idle_timeout = (
            idle_timeout if idle_timeout is not None
            else float(profile_value("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        )
        self._router = router
        self._lock = threading.Lock()
        # Local-model state (lazy)
        self._local_loaded = False
        self._local_last_used = 0.0
        self._local_unload_timer: threading.Timer | None = None

    # ── public API ────────────────────────────────────────────────────

    def see(self, image_path: str | Path, question: str = "") -> str:
        """Describe an image and answer a question about it.

        Returns the vision model's answer as text. Raises VisionUnavailable
        if no vision path works.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise ToolError(f"image not found: {image_path}")

        image_bytes = image_path.read_bytes()
        prompt = self._build_prompt(question)

        # Primary: Groq vision via router
        try:
            return self._see_via_router(image_bytes, prompt)
        except VisionUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            _log.warning("router vision failed: %s; trying local", exc)

        # Fallback: local GGUF vision model
        try:
            return self._see_via_local(image_bytes, prompt)
        except VisionUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VisionUnavailable(
                "vision unavailable: Groq vision failed and no local "
                f"vision model is usable ({exc})"
            ) from exc

    def unload_local(self) -> None:
        """Immediately unload the local vision model if loaded."""
        with self._lock:
            self._cancel_unload_timer()
            if self._local_loaded:
                self._do_unload_local()
                self._local_loaded = False

    # ── router (Groq) path ────────────────────────────────────────────

    def _get_router(self) -> Any:
        if self._router is not None:
            return self._router
        # Deferred import: the router pulls in providers.
        from ..llm.router import get_router
        return get_router()

    def _see_via_router(self, image_bytes: bytes, prompt: str) -> str:
        router = self._get_router()
        try:
            response = router.describe_image(image_bytes, prompt)
        except Exception as exc:  # noqa: BLE001
            # Distinguish "no provider supports vision" (fail fast) from
            # transient errors (fall through to local).
            msg = str(exc).lower()
            if "no registered provider supports vision" in msg:
                raise VisionUnavailable(
                    "vision unavailable: no registered LLM provider supports "
                    "vision (need Groq with vision capability or a local "
                    "vision model)"
                ) from exc
            raise
        text = (response.text or "").strip()
        if not text:
            raise ToolError("vision model returned an empty description")
        return text

    # ── local GGUF path ───────────────────────────────────────────────

    def _see_via_local(self, image_bytes: bytes, prompt: str) -> str:
        self._ensure_local_loaded()
        with self._lock:
            self._local_last_used = time.time()
            self._schedule_unload()
        try:
            from ..llm.lifecycle import get_lifecycle
            lifecycle = get_lifecycle()
            result = lifecycle.describe_image_local(image_bytes, prompt)
        except ImportError as exc:
            raise VisionUnavailable(
                "vision unavailable: local vision model support not present "
                "in the lifecycle manager"
            ) from exc
        except AttributeError as exc:
            raise VisionUnavailable(
                "vision unavailable: lifecycle manager has no local vision "
                "support"
            ) from exc
        text = (result or "").strip()
        if not text:
            raise ToolError("local vision model returned an empty description")
        return text

    def _ensure_local_loaded(self) -> None:
        with self._lock:
            if self._local_loaded:
                return
            # Ask the lifecycle/provisioner to load a vision-capable model.
            try:
                from ..llm.lifecycle import get_lifecycle
                lifecycle = get_lifecycle()
                loaded = lifecycle.ensure_vision_model()
            except ImportError as exc:
                raise VisionUnavailable(
                    "vision unavailable: no lifecycle manager for local models"
                ) from exc
            except AttributeError:
                raise VisionUnavailable(
                    "vision unavailable: lifecycle manager cannot provision "
                    "a vision model"
                )
            if not loaded:
                raise VisionUnavailable(
                    "vision unavailable: no local vision GGUF model found; "
                    "place one where the lifecycle manager expects it or "
                    "configure Groq vision"
                )
            self._local_loaded = True
            self._local_last_used = time.time()
            _log.info("local vision model loaded (lazy)")

    def _schedule_unload(self) -> None:
        self._cancel_unload_timer()
        timer = threading.Timer(self.idle_timeout, self._idle_unload)
        timer.daemon = True
        self._local_unload_timer = timer
        timer.start()

    def _cancel_unload_timer(self) -> None:
        timer = self._local_unload_timer
        self._local_unload_timer = None
        if timer is not None:
            timer.cancel()

    def _idle_unload(self) -> None:
        with self._lock:
            idle_for = time.time() - self._local_last_used
            if idle_for < self.idle_timeout:
                # Used recently; reschedule instead of unloading.
                self._schedule_unload()
                return
            self._do_unload_local()

    def _do_unload_local(self) -> None:
        try:
            from ..llm.lifecycle import get_lifecycle
            get_lifecycle().release_vision_model()
        except Exception as exc:  # noqa: BLE001
            _log.warning("local vision unload failed: %s", exc)
        finally:
            self._local_loaded = False
            _log.info("local vision model unloaded after idle timeout")

    # ── prompt ────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(question: str) -> str:
        question = (question or "").strip()
        if question:
            return f"{SEER_SYSTEM_PROMPT}\n\nQuestion: {question}"
        return (
            f"{SEER_SYSTEM_PROMPT}\n\nDescribe this image in detail: layout, "
            "visible text, UI elements and their states, colors, and anything "
            "notable."
        )


# ── convenience ──────────────────────────────────────────────────────

_default_seer: Seer | None = None
_default_lock = threading.Lock()


def get_seer() -> Seer:
    """Process-wide default Seer (lazy singleton)."""
    global _default_seer
    with _default_lock:
        if _default_seer is None:
            _default_seer = Seer()
        return _default_seer


def see(image_path: str | Path, question: str = "") -> str:
    """Describe an image and answer a question about it.

    Tool-registry-friendly wrapper around :meth:`Seer.see`.
    """
    return get_seer().see(image_path, question)
