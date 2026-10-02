"""``nm briefing`` — morning briefing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any



def _cmd_briefing(args: argparse.Namespace, context: Any) -> int:
    """Route `nm briefing` to now / today / retry / config / status /
    topics / symbols / sections / followup."""
    from ...agents import morning_briefing as mb

    as_json = getattr(args, "json", False)
    action = args.briefing_action
    try:
        if action == "status":
            payload = mb.proactive_status(context)
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                s = payload["settings"]
                print("proactive delivery:")
                print(f"  master:   {'ON' if s['proactive_enabled'] else 'OFF'}"
                      "  (NM_PARTNER_PROACTIVE_ENABLED=0 to silence)")
                print(f"  briefing: {'ON' if s['proactive_briefing'] else 'OFF'}"
                      "  (NM_PARTNER_PROACTIVE_BRIEFING)")
                print(f"  watchers: {'ON' if s['proactive_watchers'] else 'OFF'}"
                      "  (NM_PARTNER_PROACTIVE_WATCHERS)")
                print(f"  quiet hours: {s['quiet_hours']} ({s['timezone']})")
                print(f"  briefing time: {s['briefing_time']}")
                counts = payload.get("counts") or {}
                if counts:
                    print("delivery counts (last 24h):")
                    for state_name in ("sent", "failed", "pending",
                                       "held-quiet-hours", "disabled",
                                       "muted", "deduped"):
                        n = counts.get(state_name, 0)
                        if n:
                            print(f"  {state_name}: {n}")
                health = payload.get("health") or {}
                degraded = health.get("degraded") or []
                if degraded:
                    print("health: DEGRADED")
                    for reason in degraded:
                        print(f"  ! {reason}")
                elif health:
                    live = ", ".join(health.get("live_channels") or []) or "none"
                    print(f"health: ok (live channels: {live})")
                recent = payload["recent"]
                if not recent:
                    print("no proactive sends recorded yet")
                else:
                    print("recent sends:")
                    for r in recent:
                        when = time.strftime(
                            "%m-%d %H:%M",
                            time.localtime(r.get("created_at", 0)))
                        mark = {"sent": "✓", "failed": "✗",
                                "pending": "…",
                                "held-quiet-hours": "⏸",
                                "disabled": "⊘",
                                "muted": "⊘"}.get(
                                    r.get("delivery_state"), "?")
                        print(f"  {when} [{r.get('kind')}] {mark} "
                              f"{r.get('delivery_state')} — "
                              f"{r.get('title', '')[:60]}")
            return 0
        if action in ("now", "retry"):
            result = mb.run_briefing(context)
            if as_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(result.get("text", ""))
                if not result.get("delivered"):
                    print("(notifier off — printed locally only)")
            return 0 if result.get("ok") else 1
        if action == "today":
            b = mb.latest_briefing(context)
            if b is None:
                print("no briefing stored yet — run `nm briefing now`",
                      file=sys.stderr)
                return 1
            if as_json:
                print(json.dumps(b, indent=2, default=str))
            else:
                print(f"☀️ Morning briefing — {b['date']}"
                      + (" (late)" if b.get("late") else ""))
                if not b["sections"]:
                    print("(quiet night — nothing was reported.)")
                for s in b["sections"]:
                    print(f"\n— {s['title']} —")
                    for line in s.get("lines", []):
                        print(line)
            return 0
        if action == "config":
            prefs = mb._prefs(context)
            payload = {
                "time": mb.briefing_time(context),
                "timezone": mb._owner_tz(context),
                "topics": prefs.get("topics", []),
                "symbols": prefs.get("symbols", []),
                "pinned_sections": prefs.get("pinned_sections", []),
                "max_words": mb.MAX_WORDS,
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                for k, v in payload.items():
                    print(f"{k}: {v}")
            return 0
        if action == "topics":
            prefs = mb._prefs(context)
            topic = args.topic.strip()
            if args.op == "add":
                if topic not in prefs["topics"]:
                    prefs["topics"].append(topic)
            else:
                prefs["topics"] = [t for t in prefs["topics"] if t != topic]
            mb._save_prefs(context, prefs)
            print(f"topics: {prefs['topics']}")
            return 0
        if action == "symbols":
            prefs = mb._prefs(context)
            sym = args.symbol.strip().upper()
            if args.op == "add":
                if sym not in prefs["symbols"]:
                    prefs["symbols"].append(sym)
            else:
                prefs["symbols"] = [s for s in prefs["symbols"] if s != sym]
            mb._save_prefs(context, prefs)
            print(f"symbols: {prefs['symbols']}")
            return 0
        if action == "sections":
            prefs = mb._prefs(context)
            name = args.name.strip()
            if args.op == "pin":
                if name not in prefs["pinned_sections"]:
                    prefs["pinned_sections"].append(name)
            else:
                prefs["pinned_sections"] = [
                    s for s in prefs["pinned_sections"] if s != name]
            mb._save_prefs(context, prefs)
            store = mb._engagement_store(context)
            if store is not None:
                store.set_pinned(name, args.op == "pin")
            print(f"pinned sections: {prefs['pinned_sections']}")
            return 0
        if action == "followup":
            item = mb.followup_item(context, args.n)
            if item is None:
                print(f"no item {args.n} in the latest briefing",
                      file=sys.stderr)
                return 1
            if as_json:
                print(json.dumps(item, indent=2, default=str))
            else:
                it = item["item"]
                print(f"[{item['section']}] {it.get('title', '')}")
                if it.get("body"):
                    print(it["body"])
                if it.get("url"):
                    print(it["url"])
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"briefing error: {exc}", file=sys.stderr)
        return 1
    print(f"unknown briefing action: {action}", file=sys.stderr)
    return 2
