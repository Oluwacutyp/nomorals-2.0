"""``nm weather`` — weather surfaces."""

from __future__ import annotations

import argparse
from typing import Any



def _cmd_weather(args: argparse.Namespace, context: Any) -> int:
    """Route `nm weather` subcommands — keyless live weather + tz utilities."""
    from ...agents.weather import (USASituations, Weather, convert_time, owner_tz,
                                 tz_note)

    action = args.weather_action
    if action == "now":
        print(Weather().now(args.place)["text"])
    elif action == "forecast":
        print(Weather().forecast(args.place, days=args.days)["text"])
    elif action == "alerts":
        res = Weather().now(args.place)
        alerts = res.get("alerts") or []
        if alerts:
            print("\n".join(f"• [{a.get('severity', '?')}] "
                            f"{a.get('title', '')}" for a in alerts[:10]))
        else:
            print(f"no active alerts — {res['text']}")
    elif action == "usa":
        print(USASituations(Weather()).overview()["text"])
    elif action == "tz":
        if not args.when:
            print(tz_note(owner_tz()))
        else:
            res = convert_time(args.when,
                               args.from_zone or owner_tz(),
                               args.to_zone or owner_tz())
            print(res.get("text") or f"couldn't parse: {res.get('error', '')}")
    return 0
