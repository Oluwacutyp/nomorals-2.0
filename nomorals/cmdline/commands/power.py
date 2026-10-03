"""``nm power`` — power mode controls."""

from __future__ import annotations

import argparse
import getpass
import sys
from typing import Any



def _cmd_power(args: argparse.Namespace, context: Any) -> int:
    """Power mode via the owner seal (secure prompts, nothing echoed)."""
    from ...agents.power import power_mode_for

    power = power_mode_for(context)
    action = args.power_action or "status"
    if action == "status":
        s = power.status()
        print("power mode: " + ("ACTIVE" if s["active"] else "locked"))
        if s["active"]:
            print(f"unlocked by {s['unlocked_by']}")
        return 0
    if action == "battery":
        return _power_battery(args)
    if action == "lock":
        result = power.lock(actor="cli")
        print(result.get("message", ""))
        return 0
    if action == "unlock":
        identity = input("identity: ").strip()
        secret = getpass.getpass("passphrase: ")
        result = power.unlock(secret, actor="cli", identity=identity)
        print(result.get("message", ""))
        return 0 if result.get("ok") else 1
    print(f"unknown power action: {action}", file=sys.stderr)
    return 2


def _power_battery(args: argparse.Namespace) -> int:
    """Battery/thermal status from the power monitor (nomorals.power)."""
    import json

    from ...os.resources import ResourceManager
    from ...power import PowerMonitor

    # L7 (CLI) wires the L6 sampler into the L5 monitor.
    monitor = PowerMonitor(sampler=ResourceManager)
    st = monitor.status()
    as_json = bool(getattr(args, "json", False))
    if as_json:
        print(json.dumps(st.to_dict(), indent=2, default=str))
        return 0
    d = st.to_dict()
    bat = d["battery_pct"]
    print(f"battery: {bat:.0f}%" if bat is not None else "battery: unknown")
    print(f"thermal: {d['thermal_state'] or 'unknown'}")
    print(f"throttled: {d['throttled']}")
    print(f"ok: {d['ok']}")
    print(f"defer heavy work: {st.should_defer_heavy}")
    for r in d["reasons"]:
        print(f"  - {r}")
    return 0
