"""``nm commands`` / ``deliver`` / ``zip`` / ``connectors`` / ``help`` — meta surfaces."""

from __future__ import annotations

import argparse
import json
from typing import Any
from pathlib import Path
from ..parser import _parser
from ..emit import _emit



def _reply_path_report(settings_or_args, path: str = ""):
    """Return a reply path report for the given settings or args.
    
    Args:
        settings_or_args: Settings object or args namespace
        path: Optional path string
    
    Returns:
        A dict with provider and model status information
    """
    import os
    from nomorals.core.config import get_settings
    
    settings = settings_or_args
    if hasattr(settings_or_args, 'settings'):
        settings = settings_or_args.settings
    
    llm = getattr(settings, 'llm', None)
    if llm is None:
        llm = get_settings().llm
    
    provider = getattr(llm, 'provider', 'mock')
    local_model = getattr(llm, 'local_model', '')
    local_port = getattr(llm, 'local_port', 8080)
    
    # Check if model file is valid
    model_valid = bool(local_model) and os.path.exists(local_model)
    
    # Check local server state
    import socket
    server_state = "not running"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        result = s.connect_ex(('127.0.0.1', local_port))
        s.close()
        if result == 0:
            server_state = "running on port " + str(local_port)
    except Exception:  # noqa: E103 - local server probe is best-effort
        pass
    
    # Build providers list
    providers = []
    if provider:
        providers.append({
            "name": provider,
            "responds": False
        })
    
    return {
        "active_provider": provider,
        "model_file": {
            "path": local_model,
            "valid": model_valid
        },
        "local_server": {
            "state": server_state,
            "port": local_port
        },
        "providers": providers
    }


def _cmd_commands(args: argparse.Namespace, context: Any) -> int:
    """List the command catalog — the same surface chat and the console share.

    An optional filter matches group names or command names (e.g. ``nm
    commands building`` → the building group only)."""
    from ...social.chat.control import LIST_GROUPS, LIST_ONELINERS

    flt = (getattr(args, "filter", "") or "").strip().lower()
    groups: list[tuple[str, list[str]]] = []
    for name, kinds in LIST_GROUPS:
        if not flt:
            groups.append((name, list(kinds)))
            continue
        if flt in name.lower():
            groups.append((name, list(kinds)))
        else:
            hits = [k for k in kinds if flt in k
                    or flt in LIST_ONELINERS.get(k, "").lower()]
            if hits:
                groups.append((name, hits))
    payload = {"groups": [{"name": n, "commands": k} for n, k in groups]}
    lines = []
    for name, kinds in groups:
        lines.append(f"— {name} —")
        for k in kinds:
            lines.append(f"  /{k:<12} {LIST_ONELINERS.get(k, '')}")
    if not lines:
        lines = [f"no commands match {flt!r} — `nm commands` for the catalog"]
    _emit(args, payload, "\n".join(lines))
    return 0


def _cmd_deliver(args: argparse.Namespace, context: Any) -> int:
    """`nm deliver report <topic>` — create-and-deliver: generate → zip → send.

    `nm deliver send <zip> --to platform:chat` — resend a previously built
    archive (the recovery path when a deliver built the artifact but the
    send failed).
    """
    from ...core.errors import ToolError
    from ...tools.deliver_report import (deliver_report, generate_report,
                                       resend_report)

    action = getattr(args, "deliver_action", "") or ""

    def _split_target() -> tuple[str, str]:
        to = (getattr(args, "to", "") or "").strip()
        platform = (getattr(args, "platform", "") or "").strip()
        chat = to
        if to and ":" in to:
            platform, chat = to.split(":", 1)
            platform, chat = platform.strip(), chat.strip()
        return platform, chat

    def _deliver_warnings(out: dict) -> str:
        notes = out.get("warnings") or []
        if not notes:
            return ""
        return "\n" + "\n".join(f"⚠️ {note}" for note in notes)

    if action == "send":
        platform, chat = _split_target()
        path = (getattr(args, "path", "") or "").strip()
        if not path:
            _emit(args, {"error": "path required"},
                  "usage: nm deliver send <zip-path> --to platform:chat")
            return 2
        if not platform or not chat:
            _emit(args, {"error": "destination required"},
                  "usage: nm deliver send <zip-path> --to platform:chat "
                  "(e.g. --to telegram:123456) or --to <chat> --platform <name>")
            return 2
        caption = (getattr(args, "caption", "") or "").strip()
        try:
            out = resend_report(context, path, platform, chat,
                                caption=caption)
        except ToolError as exc:
            _emit(args, {"error": str(exc)}, f"deliver send: {exc}")
            return 1
        name = out["zip_path"].rsplit("/", 1)[-1]
        _emit(args, out,
              f"📄 resent {name} → {out.get('platform', platform)}:"
              f"{out.get('chat_id', chat)} "
              f"({out['zip_bytes']} B, message {out['message_id']})")
        return 0

    if action != "report":
        _emit(args, {"error": "unknown deliver action"},
              "usage: nm deliver report <topic> --section 'Title::markdown body' "
              "[--to platform:chat] [--title T] [--no-pdf]\n"
              "       nm deliver send <zip-path> --to platform:chat")
        return 2
    topic = (getattr(args, "topic", "") or "").strip()
    if not topic:
        _emit(args, {"error": "topic required"},
              "usage: nm deliver report <topic> --section 'Title::markdown body' …")
        return 2
    sections: list[tuple[str, str]] = []
    for raw in getattr(args, "section", []) or []:
        sec_title, sep, sec_body = raw.partition("::")
        if not sep or not sec_title.strip():
            _emit(args, {"error": f"bad --section {raw!r}"},
                  f"deliver: bad --section {raw!r} — "
                  "use 'Title::markdown body'")
            return 2
        sections.append((sec_title.strip(), sec_body.strip()))
    if not sections:
        _emit(args, {"error": "sections required"},
              "deliver: at least one --section 'Title::markdown body' "
              "is required — e.g.\n"
              "  nm deliver report \"Q3 markets\" "
              "--section \"Overview::The quarter was **volatile**…\" "
              "--section \"Risks::- rate hikes\\n- liquidity\"")
        return 2
    title = (getattr(args, "title", "") or "").strip()
    include_pdf = not getattr(args, "no_pdf", False)
    platform, chat = _split_target()
    to = (getattr(args, "to", "") or "").strip()

    try:
        if to:
            if not platform:
                _emit(args, {"error": "platform required"},
                      "deliver: --to needs platform:chat "
                      "(e.g. --to telegram:123456) or --platform <name>")
                return 2
            out = deliver_report(
                context, topic, sections, platform, chat, title=title,
                include_pdf=include_pdf)
            name = out["zip_path"].rsplit("/", 1)[-1]
            _emit(args, out,
                  f"📄 delivered {name} → {out.get('platform', platform)}:"
                  f"{out.get('chat_id', chat)} "
                  f"({out['zip_bytes']} B, message {out['message_id']})"
                  + _deliver_warnings(out))
        else:
            bundle = generate_report(context, topic, sections, title=title,
                                     include_pdf=include_pdf)
            out = bundle.to_dict()
            files = ", ".join(
                p.rsplit("/", 1)[-1] for p in
                ([out["html_path"]] + ([out["pdf_path"]] if out["pdf_path"] else [])))
            _emit(args, out,
                  f"📄 report built: {files} → {out['zip_path']} "
                  f"({out['zip_bytes']} B, {len(out['sections'])} sections)\n"
                  f"add --to platform:chat to send it"
                  + _deliver_warnings(out))
        return 0
    except ToolError as exc:
        _emit(args, {"error": str(exc)}, f"deliver: {exc}")
        return 1


def _cmd_zip(args: argparse.Namespace, context: Any) -> int:
    """Create and manage archives — delegated to the real Archivist system."""
    from ...archives import Archivist

    a = Archivist(context)
    action = getattr(args, "action", "list") or "list"
    path = getattr(args, "path", "") or ""
    dest = getattr(args, "dest", "") or ""
    verbs = {"info", "list", "create", "extract", "compress", "digest"}
    if action not in verbs:
        # shorthand: `nm zip <file> --dest <archive>` == create
        action, path, dest = "create", action, dest

    try:
        if action == "info":
            out = a.info(path)
        elif action == "list":
            out = a.list(path)
        elif action == "create":
            if not dest:
                _emit(args, {"error": "--dest required"},
                      "usage: nm zip <path> --dest <archive.zip>")
                return 1
            out = a.create([path], dest)
            _emit(args, {"created": dest, **out},
                  f"created {dest} ({out.get('entries', 0)} entries, "
                  f"{out.get('bytes', 0)} bytes)")
            return 0
        elif action == "extract":
            out = a.extract(path)
            n = int(out.get("extracted") or 0)
            _emit(args, out, f"extracted {n} files to {out.get('dest', '')}")
            return 0
        elif action == "compress":
            out = a.compress(path)
        elif action == "digest" and path and Path(path).is_dir():
            out = a.digest_directory(path)
        else:
            out = a.digest(path)
    except Exception as exc:  # noqa: BLE001
        _emit(args, {"error": str(exc)}, f"zip: {exc}")
        return 1

    if action == "list":
        entries = out.get("entries") or []
        names = [e if isinstance(e, str) else str(e.get("name", ""))
                 for e in entries]
        _emit(args, out, "\n".join(names) if names
              else f"{out.get('format', 'archive')}: {out.get('path', path)} "
                   f"({out.get('count', len(names))} entries)")
    else:
        _emit(args, out, json.dumps(out, indent=2, default=str))
    return 0


def _connector_vault(context: Any):
    """The encrypted vault, unlocked via NM_VAULT_PASSPHRASE. Fail fast."""
    import os

    from ...accounts.vault import CredentialVault

    passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
    if not passphrase:
        raise _ConnectorCliError(
            "vault is locked: set the NM_VAULT_PASSPHRASE environment variable "
            "to manage connector credentials"
        )
    return CredentialVault(context.db, master_passphrase=passphrase)


class _ConnectorCliError(Exception):
    """User-facing CLI error for the connectors command."""


def _cmd_connectors(args: argparse.Namespace, context: Any) -> int:
    """Manage external service connectors (connect, status, disconnect)."""
    from ...connectors import create_connector, list_connectors
    from ...connectors.base import ConnectorError

    action = getattr(args, "action", "list") or "list"
    name = getattr(args, "name", "") or ""

    if action == "list":
        infos = list_connectors()
        _emit(
            args,
            {"connectors": infos, "count": len(infos)},
            "\n".join(
                f"  {c['id']:20s} {c['description']}" for c in infos
            )
            or "no connectors registered",
        )
        return 0

    if action not in ("status", "connect", "disconnect", "provision",
                      "checkpoint", "health"):
        _emit(args, {"error": f"unknown action: {action}"},
              f"Unknown action: {action} (list, status, connect, disconnect, provision, checkpoint, health)")
        return 1
    if not name and action != "checkpoint":
        _emit(args, {"error": "name required"},
              f"Usage: nm connectors {action} --name <connector>")
        return 1

    if action == "checkpoint":
        return _cmd_connector_checkpoint(args, context)

    try:
        vault = _connector_vault(context)
    except _ConnectorCliError as exc:
        _emit(args, {"error": str(exc)}, str(exc))
        return 1
    try:
        connector = create_connector(name, vault)
    except ConnectorError as exc:
        _emit(args, {"error": str(exc)}, str(exc))
        return 1

    try:
        if action == "status":
            st = connector.status()
            _emit(
                args,
                st.to_dict(),
                f"{connector.name}: "
                f"{'connected' + (f' as {st.account}' if st.account else '') if st.connected else 'not connected'}"
                + (f" — {st.detail}" if st.detail else ""),
            )
        elif action == "connect":
            result = connector.connect()
            _emit(args, result.to_dict(), result.message or
                  (f"connected as {result.account}" if result.ok else "connect failed"))
            return 0 if result.ok else 1
        elif action == "disconnect":
            connector.disconnect()
            _emit(args, {"disconnected": True, "name": name},
                  f"{connector.name}: disconnected")
        elif action == "health":
            return _cmd_connector_health(args, connector)
        elif action == "provision":
            kind = getattr(args, "kind", "") or ""
            if not kind:
                _emit(args, {"error": "kind required"},
                      "Usage: nm connectors provision --name <connector> --kind <kind> [--params-json '{...}']")
                return 1
            if not connector.can_provision(kind):
                _emit(args, {"error": f"{name} cannot provision {kind!r}"},
                      f"{connector.name} cannot provision {kind!r}")
                return 1
            import json as _json

            raw = getattr(args, "params_json", "") or "{}"
            try:
                params = _json.loads(raw)
            except ValueError as exc:
                _emit(args, {"error": f"bad --params-json: {exc}"},
                      f"bad --params-json: {exc}")
                return 1
            # Provisioning acts on the owner's explicit command here.
            # db/context ride along for flows with human checkpoints.
            out = connector.provision(kind, db=context.db, context=context,
                                      **params)
            _emit(args, {"provisioned": kind, "result": out},
                  f"provisioned {kind}: {out}")
    except ConnectorError as exc:
        _emit(args, {"error": str(exc)}, str(exc))
        return 1
    return 0


def _cmd_connector_health(args: argparse.Namespace,
                            connector: Any) -> int:
    """``nm connectors health --name <id>`` — token/credential health check.

    Connectors that implement ``token_health()`` (e.g. x) get a real
    check with an actionable nudge; everything else falls back to the
    generic status + connectivity probe.
    """
    from ...connectors.base import ConnectorError
    try:
        health_fn = getattr(connector, "token_health", None)
        if callable(health_fn):
            report = dict(health_fn())
        else:
            st = connector.status()
            ok = connector.test_connection()
            report = {
                "connected": bool(st.connected and ok),
                "account": st.account,
                "status": ("healthy" if st.connected and ok
                           else "unknown"),
                "last_checked": st.last_checked,
                "last_ok": None,
                "consecutive_failures": 0,
                "nudge": ("" if st.connected and ok
                          else (st.detail or "connection check failed")),
            }
    except ConnectorError as exc:
        _emit(args, {"error": str(exc)}, str(exc))
        return 1
    status = report.get("status", "unknown")
    account = report.get("account") or ""
    nudge = report.get("nudge") or ""
    human = f"{connector.name}: token {status}" + (f" as {account}"
                                                  if account else "")
    if nudge:
        human += f" — {nudge}"
    _emit(args, report, human)
    return 0 if report.get("connected") else 1


def _cmd_connector_checkpoint(args: argparse.Namespace, context: Any) -> int:
    """List, resolve, or cancel human-in-the-loop checkpoints."""
    from ...connectors import create_connector
    from ...connectors.base import ConnectorError
    from ...connectors.checkpoints import (
        CheckpointStore,
        HumanCheckpointPending,
    )
    from ...core.errors import NoMoralsError

    op = getattr(args, "cop", "list") or "list"
    store = CheckpointStore(context.db)

    if op == "list":
        name = getattr(args, "name", "") or ""
        cps = store.list_pending(name or None)
        _emit(
            args,
            {"checkpoints": [c.to_dict() for c in cps], "count": len(cps)},
            "\n".join(f"  {c.id}  {c.connector_id}  {c.kind.value}  {c.title}"
                       for c in cps) or "no pending checkpoints",
        )
        return 0

    cid = getattr(args, "id", "") or ""
    if not cid:
        _emit(args, {"error": "id required"},
              f"Usage: nm connectors checkpoint --cop {op} --id <checkpoint-id>")
        return 1
    note = getattr(args, "note", "") or ""
    try:
        if op == "cancel":
            cp = store.cancel(cid, note)
            _emit(args, {"cancelled": cp.id}, f"cancelled {cp.id}")
            return 0
        if op != "resolve":
            _emit(args, {"error": f"unknown checkpoint op: {op}"},
                  f"Unknown checkpoint op: {op} (list, resolve, cancel)")
            return 1
        cp = store.resolve(cid, note or "resolved by owner")
        # The owner's resolve IS the attestation for human-only steps.
        # Hand the flow back to the connector to continue.
        vault = _connector_vault(context)
        connector = create_connector(cp.connector_id, vault)
        try:
            result = connector.resume_checkpoint(cp, db=context.db,
                                                context=context)
        except HumanCheckpointPending as pending:
            nxt = pending.checkpoint
            _emit(args, {
                "resolved": cid,
                "resumed": False,
                "next_checkpoint": nxt.to_dict(),
            }, f"resolved {cid}; next human step: {nxt.title} "
               f"(id {nxt.id})")
            return 0
        _emit(args, {"resolved": cid, "resumed": True, "result": result},
              f"resolved {cid}; flow continued: {result}")
    except (ConnectorError, NoMoralsError) as exc:
        _emit(args, {"error": str(exc)}, str(exc))
        return 1
    return 0



def _cli_subparsers() -> Any | None:
    """The top-level subparsers action of a fresh parser (for help rendering)."""
    parser = _parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _cli_command_help(topic: str) -> str | None:
    """argparse help for one nm command (alias-aware). None if not a command."""
    sub = _cli_subparsers()
    if sub is None or not topic:
        return None
    target = sub.choices.get(topic)
    if target is None:
        return None
    return target.format_help()


def _cli_overview() -> str:
    """Every top-level nm command with its aliases — generated, never stale."""
    sub = _cli_subparsers()
    rows: list[tuple[str, str, str]] = []
    if sub is not None:
        seen: set[int] = set()
        for name, target in sub.choices.items():
            if id(target) in seen:
                continue
            seen.add(id(target))
            canonical = target.prog.split()[-1]
            aliases = sorted(k for k, v in sub.choices.items()
                             if v is target and k != canonical)
            alias_txt = f" [{', '.join(aliases)}]" if aliases else ""
            help_txt = ""
            for choice_action in sub._choices_actions:
                if choice_action.dest.strip() == canonical:
                    help_txt = choice_action.help or ""
                    break
            rows.append((canonical, alias_txt, help_txt))
    rows.sort()
    lines = ["nm — the command center. Top-level commands and their aliases:",
             ""]
    for name, alias_txt, help_txt in rows:
        lines.append(f"  {name}{alias_txt}")
        if help_txt:
            lines.append(f"      {help_txt}")
    lines += ["",
              "detail for one command:  nm help <command>   (aliases work too)",
              "chat command catalog:    nm help (no topic)"]
    return "\n".join(lines)


def _cmd_help(args: argparse.Namespace, context: Any) -> int:
    """`nm help [topic]` — CLI command help first, chat pages as fallback.

    * no topic — the chat command catalog, plus a pointer to `nm help cli`.
    * ``nm help cli`` — every nm command with its alias map (generated).
    * ``nm help <command>`` — argparse help for that nm command
      (aliases accepted, e.g. ``nm help st``).
    * anything else — the same pages the chat /help renders.
    """
    from ...social.chat.control import detailed_help

    topic = (getattr(args, "topic", "") or "").strip().lstrip("/").lower()
    if topic in ("cli", "nm"):
        page = _cli_overview()
    elif not topic:
        page = (detailed_help("")
                + "\n\n— command center —\n"
                + "  `nm help cli` lists every nm command with its aliases;\n"
                + "  `nm help <command>` shows one command's full help.")
    else:
        page = _cli_command_help(topic) or detailed_help(topic)
    _emit(args, {"help": page}, page)
    return 0
