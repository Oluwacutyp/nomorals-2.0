"""``nm smarthome`` — the home surface, live from the terminal.

Thin CLI over the ``smarthome`` brain tool: the same devices the
tool discovers, from the terminal. Discovery and reads are open;
control stays off per device until the owner enables it
(``nm smarthome enable``) — the gate lives in the backends' control
seam, so a refused control never reaches a device.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from ...tools.smarthome import smarthome


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            print(item)
    else:
        print(obj)
    return 0


def _devices_text(devices: list[dict[str, Any]]) -> str:
    if not devices:
        return "no devices (run `nm smarthome discover` to scan)"
    lines = []
    for device in devices:
        state = device.get("state", {})
        on = state.get("on")
        state_text = ""
        if isinstance(state, dict) and state:
            state_text = f" — {state.get('state', '')}" if state.get(
                "state") else ""
            if on is True:
                state_text += " (on)"
            elif on is False:
                state_text += " (off)"
        enabled = device.get("control_enabled")
        gate = ""
        if enabled is True:
            gate = " [control enabled]"
        elif enabled is False:
            gate = " [control off]"
        lines.append(
            f"{device.get('name')} ({device.get('device_id')}) — "
            f"{device.get('type')}{state_text}{gate}")
    return "\n".join(lines)


def _cmd_smarthome(args: argparse.Namespace, context: Any) -> int:
    """Route `nm smarthome` subcommands."""
    action = args.smarthome_action
    device = (getattr(args, "device", "") or "").strip()
    try:
        if action == "discover":
            result = asyncio.run(smarthome(action="discover"))
            if not result.get("ok"):
                print(f"discover failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(_devices_text(result.get("devices", [])))
            return 0
        if action == "list":
            result = asyncio.run(smarthome(action="list"))
            if not result.get("ok"):
                print(f"list failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(_devices_text(result.get("devices", [])))
            return 0
        if action == "health":
            return _out(args, asyncio.run(smarthome(action="health")))
        if action == "gate":
            return _out(args, asyncio.run(smarthome(action="gate")))
        if action == "subscriptions":
            return _out(args, asyncio.run(
                smarthome(action="subscriptions")))
        if action == "events":
            sid = (getattr(args, "sid", "") or "").strip() or None
            return _out(args, asyncio.run(
                smarthome(action="events", sid=sid)))
        if action == "unsubscribe":
            sid = (getattr(args, "sid", "") or "").strip()
            if not sid:
                print("unsubscribe needs --sid SID", file=sys.stderr)
                return 2
            return _out(args, asyncio.run(
                smarthome(action="unsubscribe", sid=sid)))
        if action in ("status", "on", "off", "toggle", "subscribe",
                      "enable", "disable"):
            if not device:
                print(f"{action} needs --device ID|NAME",
                      file=sys.stderr)
                return 2
            result = asyncio.run(smarthome(action=action, device_id=device))
            if not result.get("ok"):
                print(f"{action} failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "brightness":
            if not device:
                print("brightness needs --device ID|NAME",
                      file=sys.stderr)
                return 2
            value = getattr(args, "value", None)
            if value is None:
                print("brightness needs --value 0-100", file=sys.stderr)
                return 2
            result = asyncio.run(smarthome(
                action="brightness", device_id=device, value=value))
            if not result.get("ok"):
                print(f"brightness failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            return _out(args, result)
        print(f"unknown smarthome action {action!r}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI prints, never traces
        print(f"smarthome {action} failed: {exc}", file=sys.stderr)
        return 1
