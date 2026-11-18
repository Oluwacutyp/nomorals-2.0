"""``nm av`` — the antivirus scanner from the console.

The scanner itself (``core/antivirus.py``) and the ``av_scan`` tool
shipped 2026-11-15, but the console had no verb for it. This command
surfaces the real state: the engine on this box, the last scheduled
sweep, the quarantine, and a scan on demand.

* ``nm av`` / ``nm av status`` — engine + last sweep + quarantine
  + per-engine scan timeouts
* ``nm av scan <path>`` — scan a file or folder now (quarantines
  hits, unless ``--no-quarantine``)
* ``nm av sweep`` — run one sweep tick now (the daily 03:00 job, on
  demand)
* ``nm av quarantine`` — list quarantined files from the sidecars
* ``nm av provision`` — install ClamAV via the platform package
  manager and verify it (``/models`` backend ``clamav``); the run
  itself on EC2 stays the owner's hand
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["_cmd_av"]


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
    """Engine + last sweep + quarantine + timeouts, as one JSON-plain
    dict."""
    from ...core.antivirus import (
        detect_antivirus,
        engine_timeouts,
        quarantine_dir_for,
    )
    from ...security.av_sweep import list_quarantine, read_sweep_state

    info = detect_antivirus()
    state = read_sweep_state(context)
    qdir = quarantine_dir_for(context)
    rows = list_quarantine(context)
    return {
        "engines": list(info.engines),
        "primary": info.primary,
        "detail": info.detail,
        "timeouts": engine_timeouts(),
        "last_sweep": state,
        "quarantine_dir": str(qdir) if qdir is not None else "",
        "quarantined": rows,
    }


def _cmd_av(args: argparse.Namespace, context: Any) -> int:
    """``nm av [status|scan <path>|sweep|quarantine|provision]``."""
    action = (getattr(args, "av_action", None) or "status").strip().lower()
    as_json = bool(getattr(args, "json", False))

    if action == "status":
        payload = _status_payload(context)
        if as_json:
            print(json.dumps(payload, indent=1, default=str))
            return 0
        lines = [f"engine: {payload['primary'] or 'none'}"
                 + (f" ({payload['detail']})" if payload["detail"] else "")]
        timeouts = payload.get("timeouts") or {}
        lines.append(
            "timeouts: defender "
            f"{float(timeouts.get('defender') or 0):g}s, clamav "
            f"{float(timeouts.get('clamav') or 0):g}s")
        state = payload["last_sweep"]
        if state is None:
            lines.append("last sweep: never run")
        else:
            lines.append(
                f"last sweep: {_when(float(state.get('ran_at') or 0.0))} — "
                f"scanned {state.get('scanned', 0)}, "
                f"{state.get('infected', 0)} infected, "
                f"{state.get('unknown', 0)} unknown")
        rows = payload["quarantined"]
        qdir = payload["quarantine_dir"]
        if rows:
            lines.append(f"quarantine: {len(rows)} file(s)"
                         + (f" in {qdir}" if qdir else ""))
            for row in rows[:10]:
                lines.append(
                    f"  {row.get('threat') or 'unknown threat'} — "
                    f"{row.get('original_path') or ''} → "
                    f"{row.get('quarantined_to') or ''}")
        else:
            lines.append("quarantine: empty"
                         + (f" ({qdir})" if qdir else ""))
        print("\n".join(lines))
        return 0

    if action == "scan":
        rest = list(getattr(args, "av_rest", []) or [])
        if not rest:
            return _usage_error("usage: nm av scan <path> "
                                "[--no-quarantine]")
        from ...core.antivirus import quarantine_dir_for, scan_path

        target = rest[0]
        qdir = None if getattr(args, "no_quarantine", False) \
            else quarantine_dir_for(context)
        out = scan_path(target, quarantine_dir=qdir)
        if as_json:
            print(json.dumps(out.to_dict(), indent=1, default=str))
            return 0
        lines = [f"{out.verdict}: {target} "
                 f"(engine: {out.engine or 'none'})",
                 f"  {out.detail}"]
        if out.threat:
            lines.append(f"  threat: {out.threat}")
        if out.quarantined_to:
            lines.append(f"  quarantined to: {out.quarantined_to}")
        print("\n".join(lines))
        return 0

    if action == "sweep":
        from ...security.av_sweep import run_av_sweep_tick_once

        out = run_av_sweep_tick_once(context=context)
        if as_json:
            print(json.dumps(out, indent=1, default=str))
            return 0 if out.get("ok") else 1
        if not out.get("ok"):
            print(f"sweep failed: {out.get('error') or 'unknown error'}",
                  file=sys.stderr)
            return 1
        if out.get("skipped"):
            print(f"sweep skipped: {out['skipped']}")
            return 0
        print(f"swept {out.get('scanned', 0)} file(s): "
              f"{out.get('clean', 0)} clean, "
              f"{out.get('infected', 0)} infected, "
              f"{out.get('unknown', 0)} unknown")
        for dest in out.get("quarantined") or []:
            print(f"  quarantined: {dest}")
        return 0

    if action == "quarantine":
        from ...security.av_sweep import list_quarantine

        rows = list_quarantine(context)
        if as_json:
            print(json.dumps(rows, indent=1, default=str))
            return 0
        if not rows:
            print("quarantine is empty")
            return 0
        for row in rows:
            print(f"{row.get('threat') or 'unknown threat'} — "
                  f"{row.get('original_path') or ''} → "
                  f"{row.get('quarantined_to') or ''} "
                  f"({_when(float(row.get('scanned_at') or 0.0))})")
        return 0

    if action == "provision":
        from ...tools.models_provision import (
            models_report_text,
            provision_models,
        )

        result = provision_models(["clamav"])
        if as_json:
            print(json.dumps(result, indent=1, default=str))
            return 0 if result.get("ok") else 1
        print(models_report_text(result))
        return 0 if result.get("ok") else 1

    return _usage_error(
        "usage: nm av [status|scan <path>|sweep|quarantine|provision]")
