"""``nm airports`` — the airport→country table from the console.

The mapping (``travel/airports.py``) answers visa questions from IATA
codes, and the monthly refresh job (``travel/refresh.py``) keeps it
current. This command surfaces the real state: how many codes and
countries the loaded table knows, when it was last refreshed, and a
refresh on demand.

* ``nm airports`` / ``nm airports status`` — rows, countries, last
  refresh (or an honest "never")
* ``nm airports refresh`` — run one refresh tick now (the monthly
  1st 04:00 job, on demand); a failed fetch or validation reports
  the reason and leaves the old table in place
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["_cmd_airports"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _when(epoch: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))
    except Exception:  # noqa: BLE001 - a bad timestamp is data
        return str(epoch)


def _status_payload(context: Any) -> dict[str, Any]:
    """Rows, countries, last refresh, as one JSON-plain dict."""
    from ...travel.airports import (
        airport_countries_count,
        airport_country_count,
    )
    from ...travel.refresh import (
        AIRPORTS_REFRESH_SPEC,
        AIRPORTS_SOURCE_URL,
        read_refresh_state,
    )

    state = read_refresh_state(context)
    return {
        "rows": airport_country_count(),
        "countries": airport_countries_count(),
        "source": AIRPORTS_SOURCE_URL,
        "spec": AIRPORTS_REFRESH_SPEC,
        "last_refresh": state,
    }


def _cmd_airports(args: argparse.Namespace, context: Any) -> int:
    """``nm airports [status|refresh]``."""
    action = (getattr(args, "airports_action", None)
              or "status").strip().lower()
    as_json = bool(getattr(args, "json", False))

    if action == "status":
        payload = _status_payload(context)
        if as_json:
            print(json.dumps(payload, indent=1, default=str))
            return 0
        lines = [f"airport table: {payload['rows']} IATA codes, "
                 f"{payload['countries']} countries"]
        state = payload["last_refresh"]
        if state is None:
            lines.append("last refresh: never")
        elif state.get("ok"):
            lines.append(
                f"last refresh: {_when(float(state.get('ran_at') or 0.0))}"
                f" — {state.get('rows', 0)} rows, "
                f"{state.get('countries', 0)} countries")
        else:
            lines.append(
                f"last refresh: {_when(float(state.get('ran_at') or 0.0))}"
                f" — failed: {state.get('error') or 'unknown error'}")
        lines.append(f"source: {payload['source']}")
        lines.append(f"schedule: {payload['spec']}")
        print("\n".join(lines))
        return 0

    if action == "refresh":
        from ...travel.refresh import run_airport_refresh_tick_once

        out = run_airport_refresh_tick_once(context=context)
        if as_json:
            print(json.dumps(out, indent=1, default=str))
            return 0
        if out.get("ok"):
            print(f"refreshed: {out.get('rows', 0)} IATA codes, "
                  f"{out.get('countries', 0)} countries, "
                  f"{out.get('dropped', 0)} rows dropped")
        else:
            print(f"refresh failed: {out.get('error') or 'unknown error'}")
        return 0

    return _usage_error("usage: nm airports [status|refresh] [--json]")
