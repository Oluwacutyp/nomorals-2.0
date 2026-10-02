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
