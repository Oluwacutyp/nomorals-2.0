"""``nm consent`` — record and inspect LLM provider-consent grants."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from ..emit import _emit

_DATA_KINDS = ("chat_history", "images", "embeddings_text")


def _ledger_path(settings: Any) -> Path:
    return Path(settings.home).expanduser() / "consent.json"


def _cmd_consent(args: argparse.Namespace, settings: Any) -> int:
    """Provider-consent ledger surface (AA-34).

    The router enforces these grants per provider attempt once the
    ledger file exists (``build_router`` attaches it): local providers
    never need a grant; a hosted provider may receive a data kind only
    under a recorded, unexpired, unrevoked grant.

    ``nm consent``                      status + every recorded grant
    ``nm consent grant <provider>``     record a grant (creates the
                                        ledger file — this is what arms
                                        enforcement on the next boot)
    ``nm consent revoke <record-id>``   revoke one grant
    ``nm consent check <provider>``     would this provider be allowed?
    """
    from ...core.consent import ConsentLedger

    path = _ledger_path(settings)
    action = getattr(args, "action", "status") or "status"
    target = getattr(args, "target", "") or ""

    if action == "status":
        records = []
        if path.is_file():
            try:
                records = ConsentLedger(path).records()
            except ValueError as exc:
                print(f"consent: ledger unreadable — {exc}", file=sys.stderr)
                return 1
        payload = {
            "ledger": str(path),
            "armed": path.is_file(),
            "grants": [r.to_dict() for r in records],
        }
        lines = [f"consent ledger: {path}",
                 "enforcement: armed (the router consent-checks every "
                 "provider attempt)" if path.is_file() else
                 "enforcement: not armed — no grants recorded yet; "
                 "routing is unchanged until the first `nm consent "
                 "grant` creates the ledger"]
        for r in records:
            state = "active" if r.active(time.time()) else "inactive"
            lines.append(
                f"  {r.id}  {r.destination}  {r.capability}  "
                f"kinds={','.join(r.data_kinds)}  by {r.granted_by}  "
                f"[{state}]")
        if not records and path.is_file():
            lines.append("  (no grants recorded — every hosted provider "
                         "is denied while enforcement is armed)")
        _emit(args, payload, "\n".join(lines))
        return 0

    if action == "grant":
        if not target:
            print("usage: nm consent grant <provider> "
                  "[--data-kind chat_history,images,embeddings_text] "
                  "[--capability llm.chat] [--ttl-hours N]",
                  file=sys.stderr)
            return 2
        from ...llm.providers.registry import get_provider_spec

        spec = get_provider_spec(target)
        if spec is None:
            print(f"consent: unknown provider {target!r} — consent "
                  "cannot be granted to a destination the registry "
                  "does not know", file=sys.stderr)
            return 2
        if spec.local:
            print(f"consent: {spec.id} is a local provider — data never "
                  "leaves the machine, so no grant is needed (the "
                  "router always allows it)")
            return 0
        kinds_raw = getattr(args, "data_kind", "") or "chat_history"
        kinds = [k.strip() for k in str(kinds_raw).split(",") if k.strip()]
        bad = [k for k in kinds if k not in _DATA_KINDS]
        if bad:
            print(f"consent: unknown data kind(s): {', '.join(bad)} — "
                  f"use one of {', '.join(_DATA_KINDS)}", file=sys.stderr)
            return 2
        ttl_s = None
        ttl_hours = getattr(args, "ttl_hours", 0.0) or 0.0
        if ttl_hours:
            ttl_s = float(ttl_hours) * 3600.0
        ledger = ConsentLedger(path)
        record = ledger.grant(
            destination=spec.id, data_kinds=kinds,
            granted_by="owner (nm consent)",
            capability=getattr(args, "capability", "") or "llm.chat",
            ttl_s=ttl_s)
        payload = {"record": record.to_dict(), "ledger": str(path)}
        _emit(args, payload,
              f"consent granted: {record.id} — {spec.id} may receive "
              f"{', '.join(kinds)} (capability {record.capability}); "
              f"recorded in {path} — enforcement is armed from the "
              "next boot")
        return 0

    if action == "revoke":
        if not target:
            print("usage: nm consent revoke <record-id>", file=sys.stderr)
            return 2
        if not path.is_file():
            print("consent: no ledger yet — nothing to revoke",
                  file=sys.stderr)
            return 1
        ledger = ConsentLedger(path)
        if ledger.revoke(target, revoked_by="owner (nm consent)"):
            _emit(args, {"revoked": target}, f"consent revoked: {target}")
            return 0
        print(f"consent: no live grant with id {target!r}", file=sys.stderr)
        return 1

    if action == "check":
        if not target:
            print("usage: nm consent check <provider> "
                  "[--data-kind chat_history]", file=sys.stderr)
            return 2
        from ...llm.consent import provider_consent_decision

        ledger = ConsentLedger(path)
        kind = getattr(args, "data_kind", "") or "chat_history"
        capability = getattr(args, "capability", "") or "llm.chat"
        decision = provider_consent_decision(
            ledger, target, data_kind=kind, capability=capability)
        _emit(args, {"provider": target, "data_kind": kind,
                     "capability": capability,
                     "allowed": decision.allowed,
                     "reason": decision.reason},
              f"{'allowed' if decision.allowed else 'DENIED'}: "
              f"{decision.reason}")
        return 0 if decision.allowed else 1

    print(f"consent: unknown action {action!r} "
          "(status|grant|revoke|check)", file=sys.stderr)
    return 2
