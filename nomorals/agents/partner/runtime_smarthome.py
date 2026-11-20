"""RuntimeSmarthomeMixin: PartnerRuntime command group (smart home)."""

from __future__ import annotations

from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)


class RuntimeSmarthomeMixin:
    """RuntimeSmarthomeMixin for :class:`PartnerRuntime`."""

    # ── smart home: /smarthome ────────────────────────────────────────
    def _control_smarthome(self, tail: str, chat_key: str,
                           message: Any = None) -> str:
        """Smart home devices — WeMo + Z-Wave, live from any chat.

        /smarthome discover               scan for devices (SSDP)
        /smarthome list                   cached devices, no scan
        /smarthome status <device>        one device's state
        /smarthome on|off|toggle <device> control (owner-enabled only)
        /smarthome brightness <device> <0-100>
                                          dimmer brightness, percent
        /smarthome enable|disable <device>
                                          owner: allow/forbid control
        /smarthome gate                   recorded control enables
        /smarthome subscribe <device>     listen to state changes
        /smarthome unsubscribe <sid>      end one subscription
        /smarthome subscriptions          live subscriptions
        /smarthome events [sid]           recorded state-change events
        /smarthome health                 backends + gate status
        """
        from ...tools.smarthome import smarthome_sync

        parts = (tail or "").split()
        if not parts:
            return self._control_smarthome_help()
        verb = parts[0]
        rest = parts[1:]
        try:
            if verb == "discover":
                result = smarthome_sync(action="discover")
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                return self._smarthome_devices_text(
                    result.get("devices", []))
            if verb == "list":
                result = smarthome_sync(action="list")
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                return self._smarthome_devices_text(
                    result.get("devices", []))
            if verb == "health":
                result = smarthome_sync(action="health")
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                lines = ["smarthome health:"]
                for backend in result.get("backends", []):
                    lines.append(
                        f"  {backend.get('backend')}: "
                        f"{backend.get('devices_known', '?')} device(s)"
                        f"{'' if backend.get('configured', True) else ' (not configured)'}")
                return "\n".join(lines)
            if verb == "gate":
                result = smarthome_sync(action="gate")
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                gate = result.get("gate", [])
                if not gate:
                    return "smarthome gate: no device has control enabled"
                lines = ["smarthome gate:"]
                for record in gate:
                    lines.append(
                        f"  {record.get('backend')} "
                        f"{record.get('device_id')}: "
                        f"{'enabled' if record.get('enabled') else 'disabled'}")
                return "\n".join(lines)
            if verb == "subscriptions":
                result = smarthome_sync(action="subscriptions")
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                subs = result.get("subscriptions", [])
                if not subs:
                    return "smarthome: no live subscriptions"
                lines = ["smarthome subscriptions:"]
                for sub in subs:
                    lines.append(
                        f"  {sub.get('sid')} — {sub.get('device_name')} "
                        f"({sub.get('events', 0)} events, "
                        f"{sub.get('renewals', 0)} renewals)")
                return "\n".join(lines)
            if verb == "events":
                sid = rest[0] if rest else None
                result = smarthome_sync(action="events", sid=sid)
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                events = result.get("events", [])
                if not events:
                    return "smarthome: no subscription events yet"
                lines = ["smarthome events:"]
                for event in events[-10:]:
                    lines.append(
                        f"  {event.get('device_id')}: "
                        f"{event.get('state')} "
                        f"({event.get('binary_state')})")
                return "\n".join(lines)
            if verb == "unsubscribe":
                if not rest:
                    return self._control_smarthome_help()
                result = smarthome_sync(action="unsubscribe", sid=rest[0])
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                return (f"smarthome: subscription {rest[0]} "
                        f"{'ended' if result.get('removed') else 'not found'}")
            if verb in ("status", "on", "off", "toggle", "subscribe",
                        "enable", "disable"):
                if not rest:
                    return self._control_smarthome_help()
                device = " ".join(rest)
                result = smarthome_sync(action=verb, device_id=device)
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                if verb == "status":
                    attrs = result.get("state", {})
                    line = (f"{result.get('device_id')}: "
                            f"{attrs.get('state', '')}")
                    if result.get("is_on") is True:
                        line += " (on)"
                    elif result.get("is_on") is False:
                        line += " (off)"
                    if attrs.get("brightness") is not None:
                        line += f" brightness {attrs['brightness']}"
                    if attrs.get("current_power_mw") is not None:
                        line += (f" {attrs['current_power_mw']} mW")
                    line += (" — control "
                             + ("enabled" if result.get("control_enabled")
                                else "off"))
                    return line
                if verb == "subscribe":
                    return (f"smarthome: subscribed to "
                            f"{result.get('device_name')} "
                            f"(sid {result.get('sid')}) — state changes "
                            f"will be recorded")
                if verb in ("enable", "disable"):
                    return (f"smarthome: control {verb}d for "
                            f"{result.get('device_id')}")
                attrs = result.get("state", {})
                return (f"{result.get('device_id')}: "
                        f"{attrs.get('state', '')}"
                        f"{' (on)' if result.get('is_on') is True else ''}"
                        f"{' (off)' if result.get('is_on') is False else ''}")
            if verb == "brightness":
                if len(rest) < 2:
                    return self._control_smarthome_help()
                try:
                    value = int(rest[-1])
                except ValueError:
                    return self._control_smarthome_help()
                device = " ".join(rest[:-1])
                result = smarthome_sync(action="brightness",
                                        device_id=device, value=value)
                if not result.get("ok"):
                    return f"smarthome: {result.get('error')}"
                return (f"{result.get('device_id')}: brightness "
                        f"{result.get('brightness', value)}%")
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_smarthome.RuntimeSmarthomeMixin._control_smarthome",
                exc)
            return f"smarthome failed: {exc}"
        return self._control_smarthome_help()

    def _smarthome_devices_text(self, devices: list[dict]) -> str:
        if not devices:
            return ("smarthome: no devices "
                    "(run /smarthome discover to scan)")
        lines = ["smarthome devices:"]
        for device in devices:
            state = device.get("state", {})
            line = (f"  {device.get('name')} "
                    f"({device.get('device_id')}) — {device.get('type')}")
            if isinstance(state, dict) and state.get("state"):
                line += f" {state['state']}"
            if device.get("control_enabled") is True:
                line += " [control enabled]"
            lines.append(line)
        return "\n".join(lines)

    def _control_smarthome_help(self) -> str:
        return ("/smarthome discover | list | status <device> | "
                "on|off|toggle <device> | brightness <device> <0-100> | "
                "enable|disable <device> | gate | subscribe <device> | "
                "unsubscribe <sid> | subscriptions | events [sid] | "
                "health")
