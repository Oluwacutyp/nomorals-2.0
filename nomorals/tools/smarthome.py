"""Smarthome tool — Devon's home surface as a brain tool.

The native home backends (WeMo over UPnP/SOAP + SSDP, Z-Wave over
RaZberry/Z-Way) that :mod:`nomorals.integrations.smarthome_integration`
already wires, now reachable from the brain, the ``nm`` CLI and chat:

* discovery is a fresh scan (SSDP → setup.xml, with the known-host
  ladder); ``list`` reads the cached registry without scanning;
* reads are always open; **control is off per device until the owner
  enables it** (``enable``) — the gate lives in the backends'
  control seam, so a refused control never reaches a device;
* ``subscribe`` listens to a WeMo device's state changes (GENA
  SUBSCRIBE/NOTIFY, renewed every 30s) — listening, never actuating;
* no device, no backend, no network → a named error, never an empty
  list or fake success.

Usage from agent:
    smarthome(action="discover")
    smarthome(action="status", device_id="Kitchen Plug")
    smarthome(action="on", device_id="Kitchen Plug")
    smarthome(action="subscribe", device_id="Kitchen Plug")
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import NoMoralsError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = ["register", "init_smarthome", "smarthome", "smarthome_sync"]

_log = get_logger(__name__)

_integration: Any | None = None
_lazy_context: Any = None


class _NoCredentials:
    """Credential manager with nothing stored — env-only config
    (NM_WEMO_HOSTS / NM_WEMO_ENABLED) still works without a vault."""

    def get_credential(self, *args: Any, **kwargs: Any) -> Any:
        raise KeyError("no credential")


class _NoSessions:
    """Session manager stand-in; the home integration never uses it."""


def init_smarthome(integration: Any | None) -> None:
    """Inject the integration (tests, or a pre-built one)."""
    global _integration
    _integration = integration
    _log.info("smarthome tool initialized")


def _build_integration() -> Any:
    """Build the integration from the registry context, the deals-tool
    way: a vault-backed AccountManager when the vault opens, else
    no-credential managers so env-configured backends still work."""
    from ..accounts.manager import AccountManager
    from ..accounts.sessions import SessionManager
    from ..accounts.vault import open_vault
    from ..integrations.smarthome_integration import SmartHomeIntegration
    from ..storage.db import Database

    db = getattr(_lazy_context, "db", None) if _lazy_context else None
    if db is None:
        import os
        home = os.path.expanduser("~/.nomorals")
        os.makedirs(home, exist_ok=True)
        db = Database(os.path.join(home, "smarthome.db"))
    try:
        vault = open_vault(db, context=_lazy_context, actor="smarthome")
        managers: Any = (AccountManager(vault), SessionManager(vault))
    except Exception as exc:  # noqa: BLE001 — no vault = env-only config
        _log.debug("smarthome: vault unavailable (%s); env-only", exc)
        managers = (_NoCredentials(), _NoSessions())
    return SmartHomeIntegration(*managers)


def _get_integration() -> Any:
    global _integration
    if _integration is None:
        _integration = _build_integration()
    return _integration


def _device_id(kwargs: dict[str, Any]) -> str:
    device_id = str(kwargs.get("device_id") or kwargs.get("device") or "")
    if not device_id.strip():
        raise ValueError("smarthome needs device_id (id or name)")
    return device_id.strip()


def _gate_enabled(integration: Any, device_id: str) -> bool:
    """Per-device control-enable state, from the public home_status."""
    status = integration.home_status()
    for record in status.get("control_gate", []):
        if record.get("device_id") == device_id:
            return bool(record.get("enabled"))
    return False


async def _status_payload(integration: Any,
                          device_id: str) -> dict[str, Any]:
    """One device's status, with the RAW backend state at the top
    (binary_state/state/on/...) — the DeviceState wrapper's
    ``attributes`` nesting is an integration detail, not the tool's
    shape."""
    state = await integration.get_device_status(device_id)
    device = await integration.get_device(device_id)
    return {
        "device_id": state.device_id,
        "device": device.to_dict() if device is not None else None,
        "is_on": state.is_on,
        "brightness": state.brightness,
        "state": dict(state.attributes),
        "control_enabled": _gate_enabled(integration, state.device_id),
    }


async def smarthome(action: str, **kwargs: Any) -> dict[str, Any]:
    """Smarthome tool — discover, read, control (gated) and subscribe.

    Args:
        action: discover, list, status, on, off, toggle, brightness,
            enable, disable, gate, subscribe, unsubscribe,
            subscriptions, events, health
        device_id: device id or friendly name (status/on/off/toggle/
            brightness/enable/disable/subscribe)
        value: brightness percent 0-100 (brightness)
        sid: subscription id (unsubscribe/events)
        timeout_s: subscription TTL seconds (subscribe)
    """
    try:
        integration = _get_integration()

        if action == "discover":
            devices = await integration.list_devices()
            return {"ok": True, "devices": [d.to_dict() for d in devices]}

        if action == "list":
            status = await asyncio.to_thread(integration.home_status)
            devices: list[dict[str, Any]] = []
            for backend in status.get("backends", []):
                for device in backend.get("devices", []):
                    devices.append(device)
            return {"ok": True, "devices": devices}

        if action == "health":
            status = await asyncio.to_thread(integration.home_status)
            return {"ok": True, **status}

        if action == "gate":
            status = await asyncio.to_thread(integration.home_status)
            return {"ok": True, "gate": status.get("control_gate", [])}

        if action == "status":
            device_id = _device_id(kwargs)
            payload = await _status_payload(integration, device_id)
            return {"ok": True, **payload}

        if action in ("on", "off", "toggle"):
            device_id = _device_id(kwargs)
            if action == "on":
                done = await integration.turn_on(device_id)
            elif action == "off":
                done = await integration.turn_off(device_id)
            else:
                done = await integration.toggle(device_id)
            payload = await _status_payload(integration, device_id)
            return {"ok": bool(done), **payload}

        if action == "brightness":
            device_id = _device_id(kwargs)
            try:
                value = int(kwargs.get("value", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"brightness needs an integer value, got "
                    f"{kwargs.get('value')!r}") from exc
            # Percent 0-100 — the integration converts to the
            # device's 0-255 for WeMo dimmers.
            value = max(0, min(100, value))
            done = await integration.set_light(
                device_id, brightness=value)
            payload = await _status_payload(integration, device_id)
            return {"ok": bool(done), **payload}

        if action in ("enable", "disable"):
            device_id = _device_id(kwargs)
            backend = str(kwargs.get("backend") or "wemo")
            fn = (integration.enable_device_control
                  if action == "enable"
                  else integration.disable_device_control)
            record = await asyncio.to_thread(
                fn, backend, device_id)
            return {"ok": True, "device_id": device_id,
                    "backend": backend, "record": record}

        if action == "subscribe":
            device_id = _device_id(kwargs)
            timeout_s = int(kwargs.get("timeout_s", 300))
            sub = await asyncio.to_thread(
                integration.subscribe_wemo, device_id,
                timeout_s=timeout_s)
            return {"ok": True, **sub}

        if action == "unsubscribe":
            sid = str(kwargs.get("sid") or "").strip()
            if not sid:
                raise ValueError("unsubscribe needs sid")
            removed = await asyncio.to_thread(
                integration.unsubscribe_wemo, sid)
            return {"ok": True, "sid": sid, "removed": bool(removed)}

        if action == "subscriptions":
            subs = await asyncio.to_thread(
                integration.wemo_subscriptions)
            return {"ok": True, "subscriptions": subs}

        if action == "events":
            sid = str(kwargs.get("sid") or "").strip() or None
            events = await asyncio.to_thread(
                integration.wemo_subscription_events, sid)
            return {"ok": True, "events": events}

        return {"ok": False,
                "error": f"unknown action {action!r} (discover, list, "
                         f"status, on, off, toggle, brightness, enable, "
                         f"disable, gate, subscribe, unsubscribe, "
                         f"subscriptions, events, health)"}
    except NoMoralsError as exc:
        return {"ok": False, "error": str(exc),
                "code": getattr(exc, "code", "")}
    except (ValueError, KeyError) as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — the tool never raises out
        _log.warning("smarthome action %r failed: %s", action, exc)
        return {"ok": False, "error": str(exc)}


def smarthome_sync(action: str, **kwargs: Any) -> dict[str, Any]:
    """Sync entry — the registry and the chat handler dispatch
    synchronously; the async tool runs in its own loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, smarthome(action, **kwargs)).result()
    return asyncio.run(smarthome(action, **kwargs))


def register(registry: Any) -> None:
    """Register the smarthome tool with the tool registry."""
    global _lazy_context
    _lazy_context = getattr(registry, "context", None)

    registry.register(
        name="smarthome",
        fn=smarthome_sync,
        description=(
            "Smart home devices — WeMo (UPnP/SOAP + SSDP, wemo.rs port) "
            "and Z-Wave: discover devices, read state, control (gated "
            "per device by the owner), subscribe to state changes. "
            "Discovery and reads are open; control stays off until the "
            "owner enables it per device."),
        capability=Capability.NET_OUT,
        parameters={
            "action": {
                "type": "string",
                "enum": ["discover", "list", "status", "on", "off",
                         "toggle", "brightness", "enable", "disable",
                         "gate", "subscribe", "unsubscribe",
                         "subscriptions", "events", "health"],
                "description": "Action to perform",
            },
            "device_id": {
                "type": "string",
                "description": "Device id or friendly name (status/on/"
                               "off/toggle/brightness/enable/disable/"
                               "subscribe)",
            },
            "value": {
                "type": "integer",
                "description": "Brightness percent 0-100 (brightness)",
            },
            "sid": {
                "type": "string",
                "description": "Subscription id (unsubscribe/events)",
            },
            "timeout_s": {
                "type": "integer",
                "description": "Subscription TTL seconds (subscribe)",
            },
            "backend": {
                "type": "string",
                "description": "Backend for enable/disable "
                               "(default 'wemo')",
            },
        },
    )
