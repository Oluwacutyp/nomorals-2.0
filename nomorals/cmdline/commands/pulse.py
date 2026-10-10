"""``nm pulse`` — the autonomous morning-pulse pipeline from the CLI.

``nm pulse run`` runs the full pipeline now (news → briefing → voice →
deliver), ``nm pulse status`` shows the scheduler job + prefs,
``nm pulse config`` shows time/timezone/hosts.  Thin wrappers over
``nomorals.agents.morning_pulse`` — no duplicated pipeline logic.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from ..emit import _emit


def _cmd_pulse(args: Any, context: Any) -> int:
    """Route ``nm pulse <action>``."""
    from ...agents import morning_pulse as mp

    action = str(getattr(args, "action", "") or "").strip().lower() or "status"
    as_json = bool(getattr(args, "json", False))
    if action == "run":
        try:
            res = mp.run_pulse(context)
        except Exception as exc:  # noqa: BLE001 - fail fast with the real error
            print(f"pulse run failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(res, indent=2, default=str))
            return 0
        stages = ",".join(res.get("stages", []))
        print(f"morning pulse done in {res.get('elapsed_s', '?')}s "
              f"(stages: {stages})")
        print(f"  text: {'delivered' if res.get('delivered_text') else 'NOT delivered'}")
        audio = res.get("audio_path")
        print(f"  audio: {'delivered' if res.get('delivered_audio') else ('not sent' if not audio else 'NOT delivered')}")
        return 0
    if action == "status":
        from ...agents.scheduler import Scheduler

        sched = Scheduler(context)
        jobs = [j for j in sched.list_jobs()
                if j.get("name") == mp.PULSE_JOB_NAME]
        job = jobs[0] if jobs else None
        payload = {
            "job": job,
            "time": mp.pulse_time(context),
            "timezone": mp.pulse_timezone(context),
            "enabled": mp.pulse_enabled(context),
        }
        if as_json:
            print(json.dumps(payload, indent=2, default=str))
            return 0
        if job:
            nxt = job.get("next_run_iso") or "—"
            print(f"morning pulse: {'on' if job.get('enabled') else 'off'} "
                  f"(daily {payload['time']} {payload['timezone']}, next {nxt})")
            print(f"  policies: missed={job.get('missed_fire_policy')} "
                  f"overlap={job.get('overlap_policy')} "
                  f"timeout={job.get('run_timeout_s')}s")
            if job.get("last_result"):
                print(f"  last: {job['last_result'][:100]}")
        else:
            print("morning pulse: no scheduler job "
                  "(starts automatically with the runtime)")
        _emit(args, payload, "")
        return 0
    if action == "config":
        payload = {
            "time": mp.pulse_time(context),
            "timezone": mp.pulse_timezone(context),
            "enabled": mp.pulse_enabled(context),
            "hosts": [mp.HOST_1_NAME, mp.HOST_2_NAME],
        }
        if as_json:
            print(json.dumps(payload, indent=2, default=str))
            return 0
        print(f"pulse time: {payload['time']} ({payload['timezone']})")
        print(f"enabled: {payload['enabled']}")
        print(f"hosts: {', '.join(payload['hosts'])}")
        return 0
    if action == "ensure":
        res = mp.ensure_pulse_job(context)
        if as_json:
            print(json.dumps(res, indent=2, default=str))
            return 0
        if res.get("already_scheduled"):
            print(f"morning pulse already scheduled ({res.get('job_id', '')[:8]})")
        else:
            print(f"morning pulse scheduled ({res.get('job_id', '')[:8]})")
        return 0
    print(f"unknown pulse action: {action} (run|status|config|ensure)",
          file=sys.stderr)
    return 2
