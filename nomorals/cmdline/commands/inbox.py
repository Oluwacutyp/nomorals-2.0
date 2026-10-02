"""``nm inbox`` — owner inbox."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from .improve import _inbox_obj



def _cmd_inbox(args: argparse.Namespace, context: Any) -> int:
    """Route `nm inbox` to list / show / retry / release / add-link / sweep."""
    as_json = getattr(args, "json", False)
    try:
        inbox = _inbox_obj(context)
    except Exception as exc:  # noqa: BLE001
        print(f"inbox unavailable: {exc}", file=sys.stderr)
        return 1
    action = args.inbox_action
    try:
        if action == "list":
            items = inbox.list_items(
                status=args.status or None, room=args.room or None,
                limit=args.limit)
            payload = [i.to_dict() for i in items]
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                if not payload:
                    print("inbox is empty")
                for i in payload:
                    print(f"{i['id']}  [{i['status']}] {i['kind']}: {i['name']}"
                          + (f"  → {i['intent']}" if i["intent"] else ""))
            return 0
        if action == "show":
            item = inbox.get_item(args.id).to_dict()
            if as_json:
                print(json.dumps(item, indent=2, default=str))
            else:
                for k, v in item.items():
                    print(f"{k}: {v}")
                hist = inbox.history(args.id, limit=10)
                if hist:
                    print("history:")
                    for h in hist:
                        print(f"  {h['intent']} → {h['outcome']}: "
                              f"{h['detail'][:100]}")
            return 0
        if action == "retry":
            item = inbox.retry(args.id)
            print(f"{item.id} re-queued (status={item.status})")
            return 0
        if action == "release":
            item = inbox.release(args.id)
            print(f"{item.id} released from quarantine (status={item.status})")
            return 0
        if action == "add-link":
            item = inbox.add_link(args.url, note=args.note,
                                  room=args.room or None)
            print(f"link queued: {item.id} → "
                  f"{'room ' + args.room if args.room else 'global inbox'}")
            return 0
        if action == "sweep":
            report = inbox.sweep()
            if as_json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print(f"swept: {report.get('pending_found', 0)} pending, "
                      f"{report.get('counts', {})}")
            return 0
    except (KeyError, ValueError) as exc:
        print(f"inbox: {exc}", file=sys.stderr)
        return 1
    print(f"unknown inbox action: {action}", file=sys.stderr)
    return 2
