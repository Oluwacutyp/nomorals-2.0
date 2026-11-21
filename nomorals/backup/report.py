"""Report: backup outcomes through the notifier organ.

Wires into :class:`nomorals.agents.notifier.Notifier` — the same funnel
every proactive send uses — so backup failures get the owner's attention
instead of sitting in a log file. Reporting never raises: a failed
notification must not mask a failed (or successful) backup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "report_backup_result",
    "report_backup_failure",
    "report_drill",
    "report_queued",
    "clear_backup_failure_streak",
]

#: Durable streak marker: written beside the backup generations.
#: One JSON file, one error class at a time — the marker is the
#: memory that turns "page every run" into "page once per streak".
FAILURE_STREAK_FILE = ".backup_failure_streak.json"
#: A streak older than this pages again — a stale page is worse than
#: a repeated one.
STREAK_REPAGE_DAYS = 7

_log = get_logger(__name__)

KIND = "backup"


def _publish(target: Any, title: str, body: str, *,
             critical: bool = False,
             doc: Any = None,
             alert_hints: Any = None) -> dict[str, Any]:
    """Publish via a Notifier, a context, or nowhere — never raises.

    Formatting F5: callers pass a presentation ``doc`` (built in
    ``agents/alert_templates`` — backup is L5, a peer of agents); the
    stored body is the Doc's canonical body and delivery renders the
    Doc per platform, under the exact historical kind + title.

    Pass 7 completion (F7): callers also pass ``alert_hints`` — the
    family's classification (see the report functions below)."""
    if target is None:
        _log.info("backup report (no notifier): %s — %s", title, body)
        return {"delivered": False, "reason": "no notifier"}
    try:
        kw: dict[str, Any] = {}
        if alert_hints is not None:
            kw["alert_hints"] = alert_hints
        if doc is not None:
            from ..social.chat.present import body_text

            body = body_text(doc)
            kw["doc"] = doc
        if hasattr(target, "publish"):
            return dict(target.publish(
                KIND, title, body, critical=critical, **kw
            ))
        from ..agents.notifier import Notifier

        notifier = Notifier(target)
        return dict(notifier.publish(
            KIND, title, body, critical=critical, **kw
        ))
    except Exception as exc:  # noqa: BLE001 - reporting never raises
        _log.warning("backup report publish failed: %s", exc)
        return {"delivered": False, "reason": str(exc)[:300]}


def report_backup_result(
    target: Any, result: dict[str, Any]
) -> dict[str, Any]:
    """Report a completed backup run (shipped or queued)."""
    backup_id = result.get("backup_id", "unknown")
    if result.get("queued"):
        return report_queued(target, result)
    title = f"backup shipped: {backup_id}"
    s3_note = ""
    if "s3" in result:
        s3_note = (f" S3 copy: uploaded "
                   f"{len((result.get('s3') or {}).get('objects', []))} "
                   f"objects.")
    elif "s3_error" in result:
        s3_note = f" S3 copy failed: {result['s3_error']}."
    session_note = ""
    written = ((result.get("whatsapp_session") or {}).get("written") or [])
    if written:
        paths = ", ".join(str(w.get("path") or "") for w in written
                          if w.get("path"))
        session_note = (f" WhatsApp session: sealed copy written to "
                        f"{paths}.")
    body = (
        f"Backup {backup_id} shipped to GitHub Releases "
        f"({result.get('assets', []) and len(result['assets'])} assets, "
        f"{result.get('total_encrypted_bytes', 0)} encrypted bytes). "
        f"Release: {result.get('tag', '')}. "
        f"Manifest committed: {result.get('manifest_pushed', False)}."
        f"{s3_note}{session_note}"
    )
    from ..agents.alert_templates import backup_result_doc

    return _publish(target, title, body, doc=backup_result_doc(result),
                    alert_hints={"notice": True, "subject": backup_id})


def _read_failure_marker(state_dir: str | Path) -> dict[str, Any] | None:
    """Read the streak marker; None when absent or unreadable."""
    try:
        raw = (Path(state_dir) / FAILURE_STREAK_FILE).read_text(
            encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _write_failure_marker(state_dir: str | Path,
                          marker: dict[str, Any]) -> None:
    """Write the streak marker; never raises."""
    try:
        path = Path(state_dir) / FAILURE_STREAK_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    except OSError:
        _log.warning("could not write backup failure streak marker",
                     exc_info=True)


def clear_backup_failure_streak(state_dir: str | Path | None) -> None:
    """Clear the streak marker after a successful backup run.

    A success ends the streak, so the NEXT failure pages again.
    Never raises; a missing marker is already clear.
    """
    if state_dir is None:
        return
    try:
        (Path(state_dir) / FAILURE_STREAK_FILE).unlink(missing_ok=True)
    except OSError:
        _log.warning("could not clear backup failure streak marker",
                     exc_info=True)


def report_backup_failure(
    target: Any, backup_id: str, error: Exception, *,
    state_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Report a failed backup run — critical, bypasses quiet gates.

    With ``state_dir`` given, a failure pages ONCE PER STREAK: the
    first failure of a streak pages and the marker is written; later
    failures with the same error class stay silent and only bump the
    marker's ``repeats`` count; a different error class pages again
    (a new problem); a failure more than ``STREAK_REPAGE_DAYS`` after
    the last page pages again.  Without a state dir the old behaviour
    holds (page every call) so one-off reporting sites are unchanged.
    A backup that fails every morning used to page every run — the
    owner woke to a fresh critical alert for a failure he already
    knew about (2026-11-21).
    """
    from datetime import UTC, datetime

    error_key = type(error).__name__
    if state_dir is not None:
        now = datetime.now(UTC)
        marker = _read_failure_marker(state_dir)
        last_paged = None
        if (marker is not None and marker.get("error") == error_key
                and marker.get("last_paged_at")):
            try:
                last_paged = datetime.fromisoformat(
                    marker["last_paged_at"])
            except ValueError:
                last_paged = None
        if last_paged is not None \
                and (now - last_paged).days < STREAK_REPAGE_DAYS:
            marker["repeats"] = int(marker.get("repeats", 0)) + 1
            marker["last_failed_at"] = now.isoformat()
            marker["backup_id"] = backup_id
            _write_failure_marker(state_dir, marker)
            return {"ok": False, "paged": False, "reason": "same-streak",
                    "backup_id": backup_id}
        _write_failure_marker(state_dir, {
            "error": error_key,
            "backup_id": backup_id,
            "first_failed_at": (
                marker.get("first_failed_at")
                if marker is not None
                and marker.get("error") == error_key
                and marker.get("first_failed_at")
                else now.isoformat()),
            "last_paged_at": now.isoformat(),
            "repeats": 0,
            "message": str(error)[:500],
        })
    title = f"BACKUP FAILED: {backup_id}"
    body = (
        f"Backup {backup_id} failed: {type(error).__name__}: {error}. "
        "The previous generation is intact. Check the run log and re-run; "
        "a queued snapshot (if any) will retry with backoff."
    )
    from ..agents.alert_templates import backup_failure_doc

    return _publish(target, title, body, critical=True,
                    doc=backup_failure_doc(backup_id, error),
                    alert_hints={"failed": True, "subject": backup_id})


def report_drill(target: Any, drill_report: Any) -> dict[str, Any]:
    """Report a restore-drill outcome — critical on FAIL."""
    summary = drill_report.summary() if hasattr(drill_report, "summary") \
        else str(drill_report)
    steps = drill_report.to_dict().get("steps", []) \
        if hasattr(drill_report, "to_dict") else []
    failed = [s["name"] for s in steps if not s.get("ok")]
    from ..agents.alert_templates import backup_drill_doc

    if getattr(drill_report, "ok", False):
        return _publish(target, f"restore drill PASS: {drill_report.backup_id}",
                        summary, doc=backup_drill_doc(drill_report),
                        alert_hints={"notice": True,
                                     "subject": drill_report.backup_id})
    body = (
        f"{summary}\nFailed steps: {', '.join(failed) or 'unknown'}.\n"
        "A backup never restored is a hypothesis — investigate before "
        "the next generation ships."
    )
    return _publish(target, f"RESTORE DRILL FAIL: {drill_report.backup_id}",
                    body, critical=True,
                    doc=backup_drill_doc(drill_report),
                    alert_hints={"failed": True,
                                 "subject": drill_report.backup_id})


def report_queued(target: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Report a snapshot queued in the outbox (network down, etc.)."""
    backup_id = result.get("backup_id", "unknown")
    title = f"backup queued (offline): {backup_id}"
    session_note = ""
    written = ((result.get("whatsapp_session") or {}).get("written") or [])
    if written:
        paths = ", ".join(str(w.get("path") or "") for w in written
                          if w.get("path"))
        session_note = (f" WhatsApp session: sealed copy written to "
                        f"{paths}.")
    body = (
        f"Backup {backup_id} snapshotted + encrypted locally but NOT "
        f"shipped: {result.get('reason', 'network down')}. It waits in the "
        "local outbox and retries with backoff — this is not a silent skip."
        f"{session_note}"
    )
    from ..agents.alert_templates import backup_queued_doc

    return _publish(target, title, body, doc=backup_queued_doc(result),
                    alert_hints={"self_resolving": True,
                                 "subject": backup_id})
