"""``nm account`` — account creation flows with CAPTCHA solver wiring."""

from __future__ import annotations

import argparse
import asyncio
import json as _json
import os as _os
import sys
from typing import Any


def _vault(context: Any):
    """The accounts credential vault, unlocked via NM_VAULT_PASSPHRASE."""
    from ...accounts.vault import CredentialVault

    passphrase = _os.environ.get("NM_VAULT_PASSPHRASE", "")
    if not passphrase:
        raise _AccountCliError(
            "vault is locked: set the NM_VAULT_PASSPHRASE environment variable "
            "to create accounts"
        )
    return CredentialVault(context.db, master_passphrase=passphrase)


class _AccountCliError(Exception):
    pass


def _cmd_account(args: argparse.Namespace, context: Any) -> int:
    """Route `nm account` subcommands."""
    from ...accounts.creator import (
        AccountCheckpointPending,
        AccountCreator,
        AccountExistsError,
    )
    from ...tools.captcha import creator_solver_adapter

    action = args.account_action
    settings = getattr(context, "settings", None)

    if action == "create":
        # --solver/--no-solver mirror `nm captcha solve`; None → env.
        solver_on = args.solver_enabled
        try:
            vault = _vault(context)
        except _AccountCliError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        creator = AccountCreator(
            vault,
            db=context.db,
            captcha_solver=creator_solver_adapter(
                solver_enabled=solver_on, settings=settings),
            solver_enabled=solver_on,
        )
        try:
            account = asyncio.run(creator.create_account(
                args.service,
                username=args.username,
                email=args.email,
            ))
        except AccountExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except AccountCheckpointPending as pending:
            cp = pending.checkpoint
            out = {
                "status": "checkpoint",
                "checkpoint_id": cp.id,
                "kind": cp.kind.value,
                "title": cp.title,
                "instructions": cp.instructions,
                "resume": f"nm account resume --id {cp.id}",
            }
            print(_json.dumps(out, indent=2))
            return 3
        out = {
            "status": account.status,
            "service": account.service,
            "username": account.username,
            "email": account.email,
            "notes": account.notes,
        }
        print(_json.dumps(out, indent=2))
        return 0 if account.status == "created" else 1

    if action == "resume":
        try:
            vault = _vault(context)
        except _AccountCliError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        creator = AccountCreator(vault, db=context.db)
        try:
            result = creator.resume_checkpoint(args.id, note=args.note or "")
        except Exception as exc:  # noqa: BLE001 — surface cleanly
            print(f"resume failed: {exc}", file=sys.stderr)
            return 1
        from ...accounts.creator import CreatedAccount
        if isinstance(result, CreatedAccount):
            print(_json.dumps({
                "status": result.status,
                "service": result.service,
                "username": result.username,
                "email": result.email,
            }, indent=2))
        else:
            print(_json.dumps({
                "status": "resolved",
                "checkpoint_id": result.id,
            }, indent=2))
        return 0

    if action == "pending":
        try:
            vault = _vault(context)
        except _AccountCliError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        creator = AccountCreator(vault, db=context.db)
        pending = creator.get_pending_checkpoints(service=args.service or None)
        print(_json.dumps([c.to_dict() for c in pending], indent=2))
        return 0

    if action == "sms":
        # Free temp number for SMS verification; stashed for `sms-code`.
        from ...agents.trial.flow import TrialFlow

        flow = TrialFlow(context)
        if getattr(args, "json", False):
            from ...accounts.temp_sms import grab_number

            info = grab_number(country=args.country or "us",
                               provider=args.provider or "simcodes")
            if info.get("status") == "ok":
                flow._stash_temp_number(info)
            print(_json.dumps(info, indent=2))
            return 0 if info.get("status") == "ok" else 1
        print(flow.temp_number(args.country or "us",
                               args.provider or "simcodes"))
        return 0

    if action == "sms-code":
        # Blocking poll for the verification code (CLI context: the user
        # is watching, so blocking is the honest behavior here).
        from ...agents.trial.flow import TrialFlow

        flow = TrialFlow(context)
        print(flow.temp_sms_code(sender_hint=getattr(args, "sender_hint", "")
                                 or "",
                                 timeout=float(getattr(args, "timeout", 180)
                                               or 180)))
        return 0

    if action == "inbox":
        # Poll a stored disposable-email inbox.
        from ...agents.trial.flow import TrialFlow

        flow = TrialFlow(context)
        username, msgs, error = flow.disposable_inbox_messages(
            args.service, limit=int(getattr(args, "limit", 10) or 10))
        if error:
            print(error, file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(_json.dumps({"address": username, "messages": msgs},
                              indent=2))
        else:
            print(flow.disposable_inbox(args.service,
                                        limit=int(getattr(args, "limit", 10)
                                                  or 10)))
        return 0

    print(f"unknown account action: {action}", file=sys.stderr)
    return 2
