"""Scheduled antivirus sweep (AV wiring bundle).

The scanner itself (``core/antivirus.py``, register §11(1)) and the
``av_scan`` tool shipped 2026-11-15 — but nothing ever ran a scan on
its own clock. This module is the scheduled half: a durable daily job
(03:00 local) whose tick scans the media dir, quarantines hits,
publishes through the Notifier only when there is something to
report, and writes durable last-sweep state for ``nm av status``.

Guarantees, in order:

* never raises — the tick is a scheduler routine; any failure is
  returned as ``ok: False`` with the error named, never raised into
  the scheduler;
* routine and silent — a clean sweep publishes nothing (the state
  file still records it);
* the quarantine dir itself is never re-scanned (``scan_path`` skips
  it), and a missing media dir is a skip, not an unknown verdict.

Boot wiring lives beside the other jobs in
``agents/partner/runtime.py``; the tool itself is registered in
``tools/autonomy.py`` beside the weekly-review tick.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

__all__ = [
    "AV_SWEEP_JOB",
    "AV_SWEEP_SPEC",
    "AV_SWEEP_TOOL",
    "ensure_av_sweep_job",
    "list_quarantine",
    "read_sweep_state",
    "run_av_sweep_tick_once",
]

AV_SWEEP_JOB = "av sweep"
AV_SWEEP_TOOL = "__av_sweep_tick__"
#: Daily 03:00 local — the box is quiet, and a Defender custom scan
#: of a fresh download dir is not competing with the owner's use.
AV_SWEEP_SPEC = "cron 0 3 * * *"
_STATE_NAME = "av_sweep.json"


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _media_roots(context: Any) -> list[Path]:
    """Where downloads land — the media dir under the workspace.

    Resolved from ``settings.workspace_dir`` directly: the literal
    "media" can never escape the workspace, so the tools-layer
    ``safe_path`` (L4 — this module is L3 and must not import
    upward) adds nothing here.
    """
    if context is None:
        return []
    try:
        settings = getattr(context, "settings", None)
        if settings is None:
            return []
        root = Path(settings.workspace_dir).resolve()
        return [root / "media"]
    except Exception:  # noqa: BLE001 - no resolvable media dir
        return []


def _state_path(context: Any) -> Path | None:
    home = _home_path(context)
    if home is None:
        return None
    return home / "data" / _STATE_NAME


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_sweep_state(context: Any) -> dict[str, Any] | None:
    """The last sweep's durable state, or None when never run."""
    path = _state_path(context)
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no state is a fact, not an error
        return None
    return raw if isinstance(raw, dict) else None


def list_quarantine(context: Any) -> list[dict[str, Any]]:
    """Quarantined files, newest first — from the sidecar records."""
    from ..core.antivirus import quarantine_dir_for

    qdir = quarantine_dir_for(context)
    if qdir is None or not qdir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for sidecar in sorted(qdir.glob("*.json"), reverse=True):
        if sidecar.name == "scan-journal.jsonl":
            continue
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a bad sidecar is skipped
            continue
        if isinstance(raw, dict):
            rows.append(raw)
    rows.sort(key=lambda r: float(r.get("scanned_at") or 0.0),
              reverse=True)
    return rows


def ensure_av_sweep_job(scheduler: Any, *,
                        spec: str = AV_SWEEP_SPEC) -> Any | None:
    """Register the daily AV sweep job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job that
    already matches is returned untouched. Best-effort — a scheduler
    that cannot store the job returns None rather than raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"), job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != AV_SWEEP_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            AV_SWEEP_JOB, spec, "tool",
            {"tool": AV_SWEEP_TOOL, "args": {}}, routine=True)
    except Exception:
        return None


def run_av_sweep_tick_once(
    *,
    context: Any = None,
    notifier: Any = None,
    now: float | None = None,
    roots: list[str | Path] | None = None,
    quarantine_dir: str | Path | None = None,
    state_path: str | Path | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    """One AV sweep tick. Never raises.

    Scans every root (default: the media dir), quarantines hits, and
    writes durable state. Publishes through the Notifier only when
    something was found (infected or unknown files); a clean sweep is
    silent. A root that does not exist is skipped, not counted.
    """
    now_f = float(now if now is not None else time.time())
    summary: dict[str, Any] = {"ok": True, "scanned": 0, "clean": 0,
                               "infected": 0, "unknown": 0,
                               "quarantined": [], "skipped": "",
                               "error": ""}
    try:
        from ..core.antivirus import quarantine_dir_for, scan_path

        if roots is None:
            if context is None:
                return {"ok": False, "scanned": 0, "clean": 0,
                        "infected": 0, "unknown": 0, "quarantined": [],
                        "skipped": "",
                        "error": "no context and no scan roots"}
            scan_roots = _media_roots(context)
        else:
            scan_roots = [Path(r) for r in roots]
        qdir = (Path(quarantine_dir) if quarantine_dir is not None
                else quarantine_dir_for(context))
        live = [r for r in scan_roots if r.exists()]
        if not live:
            summary["skipped"] = "no media dir to sweep yet"
            return summary
        threats: list[str] = []
        for root in live:
            verdict = scan_path(
                root, quarantine_dir=qdir,
                **({"max_files": max_files} if max_files else {}))
            results = verdict.results or [verdict.to_dict()]
            for row in results:
                summary["scanned"] += 1
                v = row.get("verdict")
                if v == "infected":
                    summary["infected"] += 1
                    if row.get("quarantined_to"):
                        summary["quarantined"].append(
                            row["quarantined_to"])
                    if row.get("threat"):
                        threats.append(str(row["threat"]))
                elif v == "unknown":
                    summary["unknown"] += 1
                else:
                    summary["clean"] += 1
        state = {"ran_at": now_f, "scanned": summary["scanned"],
                 "clean": summary["clean"],
                 "infected": summary["infected"],
                 "unknown": summary["unknown"],
                 "quarantined": summary["quarantined"],
                 "roots": [str(r) for r in live]}
        mark = (Path(state_path) if state_path is not None
                else _state_path(context))
        if mark is not None:
            _write_state(mark, state)
        if notifier is not None and (summary["infected"]
                                     or summary["unknown"]):
            lines = [f"antivirus sweep: {summary['scanned']} file(s), "
                     f"{summary['infected']} infected, "
                     f"{summary['unknown']} unknown"]
            for threat in sorted(set(threats)):
                lines.append(f"  threat: {threat}")
            for dest in summary["quarantined"]:
                lines.append(f"  quarantined: {dest}")
            notifier.publish("av_sweep",
                             title="antivirus sweep",
                             body="\n".join(lines))
        return summary
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        return {"ok": False, "scanned": 0, "clean": 0, "infected": 0,
                "unknown": 0, "quarantined": [], "skipped": "",
                "error": f"{type(exc).__name__}: {exc}"}
