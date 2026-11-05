"""Named errors for the native home/hardware backends.

The wave's honesty rule (INTEGRATION_PLAN Wave 6 acceptance): discovery
and device failures end in *named* errors from the core code system —
``network.error`` when nothing answered, ``dependency.failed`` when a
dependency (multicast stack, controller, protocol peer) failed — never
an empty list or a bare ``False`` that reads as "no devices".

The gate refusal is its own named class so the chat layer can say
exactly what happened: control was asked for on a device the owner has
not enabled.
"""

from __future__ import annotations

from typing import Any

from ...core.errors import CapabilityDenied, NotFound

__all__ = ["ControlNotEnabledError", "HomeDeviceNotFoundError"]


class ControlNotEnabledError(CapabilityDenied):
    """A control action was attempted on a device whose owner enable
    flag is off (the default).  Raised by the control seam *before* any
    transport call — a refused control never reaches the device.

    ``details`` carries ``backend``, ``device_id`` and ``action`` so the
    refusal names the device, and ``how_to_enable`` so the surface can
    tell the owner the exact enabling step.
    """

    code = "home.control_not_enabled"
    retryable = False

    def __init__(
        self,
        message: str = "",
        *,
        backend: str = "",
        device_id: str = "",
        action: str = "",
        **kw: Any,
    ) -> None:
        if not message:
            message = (
                f"control is not enabled for device {device_id!r} on "
                f"backend {backend!r} — the owner must enable control "
                f"for this device first (discovery and reads stay open)")
        super().__init__(message, **kw)
        self.backend = backend
        self.device_id = device_id
        self.action = action
        if backend:
            self.details.setdefault("backend", backend)
        if device_id:
            self.details.setdefault("device_id", device_id)
        if action:
            self.details.setdefault("action", action)
        self.details.setdefault(
            "how_to_enable",
            "SmartHomeIntegration.enable_device_control(backend, device_id)")


class HomeDeviceNotFoundError(NotFound):
    """The named device is not in the backend's discovered registry."""

    code = "home.device_not_found"

    def __init__(
        self,
        message: str = "",
        *,
        backend: str = "",
        device_id: str = "",
        **kw: Any,
    ) -> None:
        if not message:
            message = (
                f"device {device_id!r} not found on backend {backend!r} "
                f"(not discovered — run discovery first)")
        super().__init__(message, **kw)
        self.backend = backend
        self.device_id = device_id
        if backend:
            self.details.setdefault("backend", backend)
        if device_id:
            self.details.setdefault("device_id", device_id)
