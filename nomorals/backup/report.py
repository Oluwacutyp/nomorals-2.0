"""Report: backup outcomes through the notifier organ.

Wires into :class:`nomorals.agents.notifier.Notifier` — the same funnel
every proactive send uses — so backup failures get the owner's attention
instead of sitting in a log file. Reporting never raises: a failed
notification must not mask a failed (or successful) backup.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "report_backup_result",
    "report_backup_failure",
    "report_drill",
    "report_queued",
]

_log = get_logger(__name__)

KIND = "backup"


def _publish(target: Any, title: str, body: str, *,
             critical: bool = False,
             doc: Any = None) -> dict[str, Any]:
    """Publish via a Notifier, a context, or nowhere — never raises.

    Formatting F5: callers pass a presentation ``doc`` (built in
    ``agents/alert_templates`` — backup is L5, a peer of agents); the
    stored body is the Doc's canonical body and delivery renders the
    Doc per platform, under the exact historical kind + title."""
    if target is None:
        _log.info("backup report (no notifier): %s — %s", title, body)
        return {"delivered": False, "reason": "no notifier"}
    try:
        kw: dict[str, Any] = {}
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
    body = (
        f"Backup {backup_id} shipped to GitHub Releases "
        f"({result.get('assets', []) and len(result['assets'])} assets, "
        f"{result.get('total_encrypted_bytes', 0)} encrypted bytes). "
        f"Release: {result.get('tag', '')}. "
        f"Manifest committed: {result.get('manifest_pushed', False)}."
    )
    from ..agents.alert_templates import backup_result_doc

    return _publish(target, title, body, doc=backup_result_doc(result))


def report_backup_failure(
    target: Any, backup_id: str, error: Exception
) -> dict[str, Any]:
    """Report a failed backup run — critical, bypasses quiet gates."""
    title = f"BACKUP FAILED: {backup_id}"
    body = (
        f"Backup {backup_id} failed: {type(error).__name__}: {error}. "
        "The previous generation is intact. Check the run log and re-run; "
        "a queued snapshot (if any) will retry with backoff."
    )
    from ..agents.alert_templates import backup_failure_doc

    return _publish(target, title, body, critical=True,
                    doc=backup_failure_doc(backup_id, error))


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
                        summary, doc=backup_drill_doc(drill_report))
    body = (
        f"{summary}\nFailed steps: {', '.join(failed) or 'unknown'}.\n"
        "A backup never restored is a hypothesis — investigate before "
        "the next generation ships."
    )
    return _publish(target, f"RESTORE DRILL FAIL: {drill_report.backup_id}",
                    body, critical=True,
                    doc=backup_drill_doc(drill_report))


def report_queued(target: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Report a snapshot queued in the outbox (network down, etc.)."""
    backup_id = result.get("backup_id", "unknown")
    title = f"backup queued (offline): {backup_id}"
    body = (
        f"Backup {backup_id} snapshotted + encrypted locally but NOT "
        f"shipped: {result.get('reason', 'network down')}. It waits in the "
        "local outbox and retries with backoff — this is not a silent skip."
    )
    from ..agents.alert_templates import backup_queued_doc

    return _publish(target, title, body, doc=backup_queued_doc(result))
