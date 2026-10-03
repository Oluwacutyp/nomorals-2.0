"""``nm trigger`` — event-condition-action automation.

    nm trigger add --name "btc dip" --source price \\
        --symbol BTC --op lt --value 60000 \\
        --action notify --title "BTC dip" --body "below 60k"
    nm trigger add --name "morning" --source schedule --cron "0 9 * * *" \\
        --action notify --title "good morning"
    nm trigger add --name "deploy log" --source file --path ./deploy.log \\
        --action message --chat telegram:123 --text "deploy log changed"
    nm trigger list [--json]
    nm trigger remove <id>
    nm trigger enable|disable <id>
    nm trigger history [<id>] [--limit N] [--json]
    nm trigger run <id>

``--condition`` / ``--params`` take raw JSON for anything the sugar
flags don't cover; sugar flags merge into (never silently override)
them.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _engine(context: Any):
    from ...triggers import TriggerEngine
    return TriggerEngine(context.db, context)


def _cmd_trigger(args: Any, context: Any) -> int:
    """Route ``nm trigger <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm trigger add --name NAME --source SRC --action ACT [...]\n"
              "       nm trigger list [--json]\n"
              "       nm trigger remove <id>\n"
              "       nm trigger enable|disable <id>\n"
              "       nm trigger history [<id>] [--limit N] [--json]\n"
              "       nm trigger run <id>\n"
              "sources: schedule|file|price|message|webhook\n"
              "actions: notify|message|command|mission\n"
              "add flags: --condition JSON --params JSON --cooldown SECONDS\n"
              "  schedule: --cron EXPR | --interval 30m | --daily HH:MM |\n"
              "            --weekly 'MON 09:30' | --once ISO\n"
              "  file: --path PATH [--on change|create|delete]\n"
              "  price: --symbol SYM [--market crypto|stocks|forex]\n"
              "         [--op lt|gt|...] [--value N]\n"
              "  message: --pattern REGEX [--chat KEY] [--sender S]\n"
              "  webhook: [--secret S]\n"
              "  notify: [--title T] [--body B]\n"
              "  message action: --chat KEY --text T\n"
              "  command: --argv CMD... | --command \"...\"\n"
              "  mission: --goal G [--max-iterations N]",
              file=sys.stderr)
        return 2
    verb = words[0]
    try:
        if verb == "add":
            return _trigger_add(args, context, words[1:])
        if verb == "list":
            return _trigger_list(args, context)
        if verb == "remove":
            return _trigger_remove(args, context, words[1:])
        if verb in ("enable", "disable"):
            return _trigger_enable(args, context, words[1:], verb == "enable")
        if verb == "history":
            return _trigger_history(args, context, words[1:])
        if verb == "run":
            return _trigger_run(args, context, words[1:])
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown trigger verb: {verb}", file=sys.stderr)
    return 2


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _load_json(text: str, flag: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"error: {flag} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(data, dict):
        print(f"error: {flag} must be a JSON object", file=sys.stderr)
        raise SystemExit(2)
    return data


def _num(text: str) -> Any:
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _build_condition(args: Any) -> dict[str, Any]:
    raw = getattr(args, "condition", "") or ""
    cond = _load_json(raw, "--condition") if raw else {}
    src = (getattr(args, "source", "") or "").strip()
    sugar: dict[str, Any] = {}
    if src == "schedule":
        for flag, key in (("cron", "cron"), ("interval", "interval"),
                          ("daily", "daily"), ("weekly", "weekly"),
                          ("once", "once")):
            val = getattr(args, flag, "") or ""
            if val:
                sugar[key] = val
    elif src == "file":
        if getattr(args, "path", ""):
            sugar["path"] = args.path
        if getattr(args, "on", ""):
            sugar["on"] = args.on
    elif src == "price":
        if getattr(args, "symbol", ""):
            sugar["symbol"] = args.symbol
        if getattr(args, "market", ""):
            sugar["market"] = args.market
        if getattr(args, "op", ""):
            sugar["op"] = args.op
        if getattr(args, "value", None) not in (None, ""):
            sugar["value"] = _num(str(args.value))
    elif src == "message":
        if getattr(args, "pattern", ""):
            sugar["pattern"] = args.pattern
        if getattr(args, "chat", ""):
            sugar["chat"] = args.chat
        if getattr(args, "sender", ""):
            sugar["sender"] = args.sender
    elif src == "webhook":
        if getattr(args, "secret", ""):
            sugar["secret"] = args.secret
    for key, val in sugar.items():
        if key in cond and cond[key] != val:
            print(f"error: --{key} conflicts with --condition JSON",
                  file=sys.stderr)
            raise SystemExit(2)
        cond[key] = val
    return cond


def _build_params(args: Any) -> dict[str, Any]:
    raw = getattr(args, "params", "") or ""
    params = _load_json(raw, "--params") if raw else {}
    sugar: dict[str, Any] = {}
    if getattr(args, "title", ""):
        sugar["title"] = args.title
    if getattr(args, "body", ""):
        sugar["body"] = args.body
    if getattr(args, "chat", ""):
        sugar["chat"] = args.chat
    if getattr(args, "text", ""):
        sugar["text"] = args.text
    if getattr(args, "command_text", ""):
        sugar["command"] = args.command_text
    if getattr(args, "argv", None):
        sugar["argv"] = list(args.argv)
    if getattr(args, "goal", ""):
        sugar["goal"] = args.goal
    if getattr(args, "max_iterations", None):
        sugar["max_iterations"] = args.max_iterations
    for key, val in sugar.items():
        if key in params and params[key] != val:
            print(f"error: --{key} conflicts with --params JSON",
                  file=sys.stderr)
            raise SystemExit(2)
        params[key] = val
    return params


def _trigger_add(args: Any, context: Any, rest: list[str]) -> int:
    _ = rest
    name = getattr(args, "name", "") or ""
    source = (getattr(args, "source", "") or "").strip()
    action = (getattr(args, "action", "") or "").strip()
    if not name or not source or not action:
        print("usage: nm trigger add --name NAME --source SRC --action ACT "
              "[--condition JSON] [--params JSON] [sugar flags...]",
              file=sys.stderr)
        return 2
    eng = _engine(context)
    cooldown = getattr(args, "cooldown", 0.0) or 0.0
    trigger = eng.add(name, source, _build_condition(args), action,
                      _build_params(args), cooldown_s=cooldown)
    if _as_json(args):
        print(json.dumps(trigger.to_dict(), indent=2, default=str))
    else:
        print(f"added trigger {trigger.id} ({trigger.source} -> "
              f"{trigger.action})")
    return 0


def _trigger_list(args: Any, context: Any) -> int:
    eng = _engine(context)
    triggers = eng.list()
    if _as_json(args):
        print(json.dumps([t.to_dict() for t in triggers], indent=2,
                         default=str))
        return 0
    if not triggers:
        print("no triggers")
        return 0
    for t in triggers:
        state = "on" if t.enabled else "off"
        last = (f" last_fired={t.last_fired:.0f}" if t.last_fired else "")
        print(f"{t.id} [{state}] {t.name}: "
              f"{t.source} -> {t.action} (fires={t.fire_count}{last})")
    return 0


def _trigger_remove(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm trigger remove <id>", file=sys.stderr)
        return 2
    eng = _engine(context)
    if not eng.remove(rest[0]):
        print(f"unknown trigger: {rest[0]}", file=sys.stderr)
        return 1
    print(f"removed trigger {rest[0]}")
    return 0


def _trigger_enable(args: Any, context: Any, rest: list[str],
                    enabled: bool) -> int:
    if not rest:
        print(f"usage: nm trigger {'enable' if enabled else 'disable'} <id>",
              file=sys.stderr)
        return 2
    eng = _engine(context)
    trigger = eng.set_enabled(rest[0], enabled)
    print(f"trigger {trigger.id} "
          f"{'enabled' if trigger.enabled else 'disabled'}")
    return 0


def _trigger_history(args: Any, context: Any, rest: list[str]) -> int:
    eng = _engine(context)
    trigger_id = rest[0] if rest else None
    limit = getattr(args, "limit", 0) or 50
    rows = eng.history(trigger_id, limit=limit)
    if _as_json(args):
        print(json.dumps(rows, indent=2, default=str))
        return 0
    if not rows:
        print("no history")
        return 0
    for r in rows:
        import datetime as _dt
        when = _dt.datetime.fromtimestamp(r["at"]).strftime(
            "%m-%d %H:%M:%S")
        err = f" error={r['error']}" if r["error"] else ""
        print(f"{when} {r['trigger_id']} {r['outcome']}{err}")
    return 0


def _trigger_run(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm trigger run <id>", file=sys.stderr)
        return 2
    eng = _engine(context)
    result = eng.manual_fire(rest[0])
    if _as_json(args):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"trigger {rest[0]}: {result['outcome']}")
        if result.get("error"):
            print(f"  error: {result['error']}")
    return 0 if result["fired"] else 1
