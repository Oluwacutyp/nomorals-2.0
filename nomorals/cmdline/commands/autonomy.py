"""``nm autonomy`` — autonomous operations."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from pathlib import Path
from ..emit import _emit



def _cmd_autonomy(args: argparse.Namespace, context: Any) -> int:
    """Autonomy dial: status/on/off/tick/report/budget.

    ``on``/``off`` persist through the durable kv override (wave 51), so the
    next process — and the scheduler boot path — see the same dial state
    without touching config.toml.
    """
    from ...agents.cognition import (CognitiveLoop, ModelBudget,
                                  autonomy_enabled, set_autonomy_enabled)

    action = getattr(args, "action", "status") or "status"
    if action == "status":
        enabled = autonomy_enabled(context)
        loop_status = CognitiveLoop(context).status()
        payload = {"enabled": enabled, **loop_status}
        _emit(args, payload,
              f"autonomy: {'on' if enabled else 'off'} "
              f"(interval {loop_status.get('interval_hours', 0)}h)")
        return 0
    if action in ("on", "enable"):
        set_autonomy_enabled(context, True)
        _emit(args, {"ok": True, "enabled": True, "persisted": True},
              "autonomy: enabled (persisted for the next boot)")
        return 0
    if action in ("off", "disable"):
        set_autonomy_enabled(context, False)
        _emit(args, {"ok": True, "enabled": False, "persisted": True},
              "autonomy: disabled (persisted for the next boot)")
        return 0
    if action == "tick":
        tick = CognitiveLoop(context).tick()
        stages = tick.get("stages", {})
        bits = ", ".join(
            f"{name}:{next(iter(st.keys()))}" if st else f"{name}:ok"
            for name, st in stages.items())
        _emit(args, tick,
              f"cognitive loop tick in {tick.get('seconds', 0)}s — {bits}")
        return 0
    if action == "report":
        report = CognitiveLoop(context).report()
        ticks = report.get("ticks", 0)
        lines = [f"cognitive loop report: {ticks} heartbeat(s), "
                 f"total {report.get('total_seconds', 0)}s "
                 f"(avg {report.get('avg_seconds', 0)}s)"]
        for name, counters in (report.get("stages") or {}).items():
            bits = " ".join(f"{k}={v}" for k, v in sorted(counters.items()))
            lines.append(f"{name}: {bits}")
        _emit(args, report, "\n".join(lines))
        return 0
    if action == "budget":
        cap_arg = str(getattr(args, "cap", "") or "")
        if getattr(args, "unlimited", False):
            _set_daily_call_cap(context, 0)
            _emit(args, {"ok": True, "cap": 0},
                  "model-call budget: set to 0 (unlimited) — persisted to the "
                  "home .env as NM_AUTONOMY_DAILY_MODEL_CALLS")
            return 0
        if cap_arg:
            try:
                n = int(cap_arg)
            except ValueError:
                print("autonomy budget: --cap needs an integer", file=sys.stderr)
                return 2
            if n < 0:
                print("autonomy budget: --cap must be >= 0 (0 = unlimited)",
                      file=sys.stderr)
                return 2
            _set_daily_call_cap(context, n)
            _emit(args, {"ok": True, "cap": n},
                  f"model-call budget: set to {n} calls/day — persisted to "
                  "the home .env as NM_AUTONOMY_DAILY_MODEL_CALLS")
            return 0
        budget = ModelBudget(context).report()
        if budget.get("unlimited"):
            text = (f"model budget: unlimited — used {budget.get('used', 0)} calls, "
                    f"{budget.get('tokens', 0)} tokens today, "
                    f"reserved {budget.get('reserved', 0)} in flight")
        else:
            text = (f"model budget: cap {budget.get('cap', 0)} calls/day — "
                    f"used {budget.get('used', 0)}, "
                    f"remaining {budget.get('remaining', 0)}, "
                    f"reserved {budget.get('reserved', 0)} "
                    f"(pressure {budget.get('pressure', 0.0):.0%}"
                    + (", throttled" if budget.get("throttled") else "")
                    + ") — lift with `nm autonomy budget --unlimited`")
        _emit(args, budget, text)
        return 0
    print(f"autonomy: unknown action {action}", file=sys.stderr)
    return 2


def _set_daily_call_cap(context: Any, calls: int) -> None:
    """Persist the daily model-call budget as NM_AUTONOMY_DAILY_MODEL_CALLS.

    Written to the home ``.env`` (the file users already maintain), replacing
    any existing line so one knob = exactly one line. Also applied to the
    in-memory settings so the current process agrees.
    """
    import os
    from pathlib import Path

    home = Path(os.path.expanduser(os.environ.get("NM_HOME")
                                    or getattr(context.settings, "home", "~/.nomorals")))
    key = "NM_AUTONOMY_DAILY_MODEL_CALLS"
    home.mkdir(parents=True, exist_ok=True)
    env_path = home / ".env"
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text().splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(key + "="):
            lines[i] = f"{key}={int(calls)}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={int(calls)}")
    env_path.write_text("\n".join(lines) + "\n")
    try:
        context.settings.autonomy.daily_model_calls = int(calls)
    except Exception:  # noqa: BLE001 — the file is the durable truth
        pass
