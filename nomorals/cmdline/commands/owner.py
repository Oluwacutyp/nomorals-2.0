"""``nm owner`` — owner identity seal."""

from __future__ import annotations

import argparse
import getpass
import hmac
import sys
from typing import Any



def _cmd_owner(args: argparse.Namespace, context: Any) -> int:
    """Owner identity seal: bake / verify the passphrase proof in code."""
    from ...core.owner import (OWNER_IDENTITIES, bake_seal, make_seal,
                             seal_configured, verify_owner)

    action = args.owner_action or "whoami"
    if action == "whoami":
        print("ingrained owner identities: " + ", ".join(OWNER_IDENTITIES))
        print("passphrase seal: " + ("baked in code" if seal_configured()
                                     else "not set — run `nm owner seal`"))
        return 0
    identity = input("identity: ").strip()
    secret = getpass.getpass("passphrase: ")
    if action == "seal":
        if not secret:
            print("empty passphrase — nothing sealed.", file=sys.stderr)
            return 1
        confirm = getpass.getpass("passphrase (again): ")
        if not hmac.compare_digest(secret, confirm):
            print("passphrases do not match.", file=sys.stderr)
            return 1
        try:
            seal = make_seal(secret)
        except ValueError as exc:
            print(f"too weak: {exc}", file=sys.stderr)
            return 1
        path = bake_seal(seal)
        print(f"seal baked into {path}")
        print("the passphrase itself was never written anywhere.")
        return 0
    if action == "verify":
        ok = verify_owner(identity, secret)
        print("owner verified." if ok else "not verified.")
        return 0 if ok else 1
    print(f"unknown owner action: {action}", file=sys.stderr)
    return 2
