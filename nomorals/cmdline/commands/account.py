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

    if action == "login":
        # Session-restore-aware login: when the stored session is still
        # valid, no browser opens and no login form is touched
        # (--force bypasses the restore check and re-logs in anyway).
        from ...accounts import (
            AccountManager,
            LoginCaptchaRequired,
            LoginConfig,
            LoginFailed,
            SessionManager,
            ensure_login,
        )
        try:
            vault = _vault(context)
        except _AccountCliError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        manager = AccountManager(vault)
        sessions = SessionManager(vault)

        def _open_tab():
            from ...browser.service import BrowserService
            svc = BrowserService()
            try:
                return svc.open_rendered_tab(args.service)
            except Exception:
                # rendered tabs need playwright+chromium; fall back to the
                # plain-HTTP tab (works for simple login forms).
                handle = svc.open_session(args.service)
                return handle.open_tab()

        solver = None
        solver_on = getattr(args, "solver_enabled", None)
        if solver_on is not False:
            solver = creator_solver_adapter(
                solver_enabled=solver_on, settings=settings)
        cfg = LoginConfig(
            login_url=getattr(args, "login_url", "") or "",
            success_text=getattr(args, "success_text", "") or "",
        )
        try:
            result = ensure_login(
                manager, sessions, _open_tab,
                service=args.service,
                username=getattr(args, "username", None),
                config=cfg,
                captcha_solver=solver,
                force=bool(getattr(args, "force", False)),
            )
        except (LoginFailed, LoginCaptchaRequired) as exc:
            print(f"login failed: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(_json.dumps(result, indent=2))
        else:
            print(result["note"])
            if result.get("from_session"):
                print("(restored from the stored session — no re-login)")
            else:
                print(f"cookies saved: {result['cookies_saved']}")
        return 0

    if action == "default":
        from ...accounts import AccountManager
        try:
            vault = _vault(context)
        except _AccountCliError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        manager = AccountManager(vault)
        service = args.service
        if getattr(args, "clear", False):
            cleared = manager.clear_default(service)
            out = {"service": service, "cleared": cleared}
            if getattr(args, "json", False):
                print(_json.dumps(out, indent=2))
            else:
                print(f"default for {service}: "
                      f"{'cleared' if cleared else 'was not set'}")
            return 0
        username = getattr(args, "username", None)
        if not username:
            current = manager.get_default(service)
            out = {"service": service, "default": current}
            if getattr(args, "json", False):
                print(_json.dumps(out, indent=2))
            else:
                print(f"default for {service}: {current or '(not set)'}")
            return 0
        try:
            manager.set_default(service, username)
        except Exception as exc:  # noqa: BLE001 — surface cleanly
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out = {"service": service, "default": username}
        if getattr(args, "json", False):
            print(_json.dumps(out, indent=2))
        else:
            print(f"default for {service} -> {username}")
        return 0

    if action == "rotate":
        from ...accounts import AccountManager
        try:
            vault = _vault(context)
        except _AccountCliError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        manager = AccountManager(vault)

        if getattr(args, "on_site", False):
            # Full rotation: change the password on the site itself, then
            # in the vault. Needs the site's password-change page.
            from ...accounts import (
                LoginCaptchaRequired,
                LoginFailed,
                PasswordChangeConfig,
                SessionManager,
                change_password_on_site,
            )
            sessions = SessionManager(vault)

            def _open_tab():
                from ...browser.service import BrowserService
                svc = BrowserService()
                try:
                    return svc.open_rendered_tab(args.service)
                except Exception:
                    handle = svc.open_session(args.service)
                    return handle.open_tab()

            solver = None
            solver_on = getattr(args, "solver_enabled", None)
            if solver_on is not False:
                solver = creator_solver_adapter(
                    solver_enabled=solver_on, settings=settings)
            pw_cfg = PasswordChangeConfig(
                change_password_url=getattr(
                    args, "change_password_url", "") or "",
                success_text=getattr(args, "success_text", "") or "",
            )
            try:
                result = change_password_on_site(
                    manager, sessions, _open_tab,
                    service=args.service,
                    username=args.username,
                    length=int(getattr(args, "length", 32) or 32),
                    config=pw_cfg,
                    captcha_solver=solver,
                )
            except (LoginFailed, LoginCaptchaRequired) as exc:
                print(f"on-site rotation failed: {exc}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                print(_json.dumps(result, indent=2))
            else:
                print(result["note"])
            return 0

        try:
            cred = manager.rotate_credential_auto(
                args.service, args.username,
                length=int(getattr(args, "length", 32) or 32))
        except Exception as exc:  # noqa: BLE001 — surface cleanly
            print(f"rotate failed: {exc}", file=sys.stderr)
            return 1
        out = {"service": cred.service, "username": cred.username,
               "rotated": True,
               "note": "new password stored in the vault — apply it on the "
                       "service's own password-change page"}
        if getattr(args, "json", False):
            print(_json.dumps(out, indent=2))
        else:
            print(f"rotated {cred.service}/{cred.username} "
                  f"({len(cred.password)}-char password in vault)")
            print("apply the new password on the service's own "
                  "password-change page — the vault copy is updated.")
        return 0

    if action == "health":
        # Account health: vault state + session health + optional
        # server-side probe. One account or the full sweep.
        from ...accounts import AccountManager, SessionManager, check_all_health
        try:
            vault = _vault(context)
        except _AccountCliError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        manager = AccountManager(vault)
        sessions = SessionManager(vault)
        service = getattr(args, "service", None)
        username = getattr(args, "username", None)
        probe_url = getattr(args, "probe_url", "") or ""
        if service:
            from ...accounts import check_account_health
            report = check_account_health(
                manager, sessions, service, username,
                probe_url=probe_url)
            if getattr(args, "json", False):
                print(_json.dumps(report.to_dict(), indent=2))
            else:
                print(f"{report.service}/{report.username}: {report.status}")
                for issue in report.issues:
                    print(f"  - {issue}")
            return 0 if report.ok else 1
        # Sweep mode: probe URLs come from each credential's metadata
        # (health_probe_url), so configure once and sweep forever.
        sweep = check_all_health(manager, sessions)
        if getattr(args, "json", False):
            print(_json.dumps(sweep, indent=2))
        else:
            print(f"checked {len(sweep['accounts'])} account(s):")
            for entry in sweep["accounts"]:
                print(f"  {entry['service']}/{entry['username']}: "
                      f"{entry['status']}")
            print(f"summary: {_json.dumps(sweep['summary'])}")
        return 0 if not sweep["needs_attention"] else 1

    print(f"unknown account action: {action}", file=sys.stderr)
    return 2
