"""The status beacon: how a separate process learns the bot is alive.

``nm status`` (or any other CLI invocation) runs in a NEW process — it
cannot read the running bot's in-memory router state.  So the running
:mod:`partner_runtime` writes a small heartbeat file while it lives:

* every ~10 seconds from its main loop (and once on shutdown), it atomically
  writes ``<home>/state/status.json`` — pid, uptime, platform state, the
  last reply (which model answered, and whether it was a fallback), the
  last model error, and the router's health snapshot.

* a reader in another process calls :func:`read_status` and gets the state
  plus its age.  "Alive" means the file is fresh (< :data:`ALIVE_WINDOW_S`)
  and not marked stopped — exactly how you check any daemon: by the
  freshness of what it wrote, never by assuming.

The beacon is best-effort: a failed write is swallowed (logging state
must never take the bot down with it), and the CLI treats a stale/missing
beacon as "not running" with the last known line shown.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = [
    "ALIVE_WINDOW_S",
    "BEACON_INTERVAL_S",
    "build_beacon_state",
    "last_log_error",
    "read_status",
    "status_path",
    "write_status",
]

#: a beacon older than this means the process is gone (writes every 10s)
ALIVE_WINDOW_S = 45.0
#: how often the runtime rewrites the beacon
BEACON_INTERVAL_S = 10.0


def status_path(home: str | os.PathLike[str]) -> Path:
    return Path(home).expanduser() / "state" / "status.json"


def write_status(home: str | os.PathLike[str], state: dict[str, Any]) -> bool:
    """Atomically write the beacon.  Returns True on success."""
    path = status_path(home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, default=str),
                       encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def read_status(home: str | os.PathLike[str]) -> tuple[dict[str, Any] | None,
                                                      float | None]:
    """The beacon state and its age in seconds, or (None, None)."""
    path = status_path(home)
    try:
        raw = path.read_text(encoding="utf-8")
        state = json.loads(raw)
    except (OSError, ValueError):
        return None, None
    if not isinstance(state, dict):
        return None, None
    try:
        age = time.time() - float(state.get("ts") or 0.0)
    except (TypeError, ValueError):
        return state, None
    return state, max(0.0, age)


def build_beacon_state(runtime: Any) -> dict[str, Any]:
    """Assemble the beacon from a running PartnerRuntime.

    Reply/error bookkeeping lives on the BRAIN (it generates the replies);
    uptime and process stats live on the runtime.  Never raises — a status
    line that crashes is worse than none.
    """
    brain = getattr(runtime, "brain", None)
    started = getattr(runtime, "_started", time.time())
    state: dict[str, Any] = {
        "ts": time.time(),
        "pid": os.getpid(),
        "uptime_s": round(time.time() - started, 1),
        "stats": dict(getattr(runtime, "stats", {}) or {}),
        "last_reply": dict(getattr(brain, "_last_reply", {}) or {}),
        "last_error": getattr(brain, "_last_error", "") or "",
        "stopped": False,
    }
    try:
        state["platforms"] = runtime.gateway.status()
    except Exception:  # noqa: BLE001
        pass
    try:
        router = getattr(getattr(runtime, "context", None), "router", None)
        if router is not None and hasattr(router, "stats_snapshot"):
            state["router"] = router.stats_snapshot()
    except Exception:  # noqa: BLE001
        pass
    return state


def last_log_error(log_path: str | os.PathLike[str],
                   max_bytes: int = 256 * 1024) -> str:
    """The most recent ERROR line from the rotating diag log, if any.

    Reads only the tail (the log rotates at 10MB; 256KB is far more than
    one error's worth) so a big log never costs the status command.
    """
    path = Path(log_path).expanduser()
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        if " ERROR " in line or line.startswith("ERROR "):
            # keep the message part: "… | nome.log | msg" style or "ERROR msg"
            if " | " in line:
                tail = line.rsplit(" | ", 1)[-1]
            else:
                tail = line.split(" ERROR ", 1)[-1]
            return tail.strip()[:200]
    return ""
