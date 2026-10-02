"""``nm timeline`` — the persistent event timeline."""

from __future__ import annotations

import json



def _render_timeline_row(row: dict) -> str:
    """One human-readable timeline line: ts, topic, ids, key facts."""
    import datetime as _dt

    ts = _dt.datetime.fromtimestamp(row.get("ts") or 0,
                                    tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    topic = str(row.get("topic") or "?")
    ids = " ".join(f"{k}={row[k]}" for k in
                   ("session_id", "project_id", "mission_id", "artifact_id")
                   if row.get(k))
    data = row.get("data") or {}
    facts: list[str] = []
    for key in ("from_state", "to_state", "verdict", "frontend", "principal",
                "type", "creator"):
        if data.get(key):
            facts.append(f"{key}={data[key]}")
    note = data.get("note") or data.get("details") or ""
    detail = (" | " + " ".join(facts)) if facts else ""
    if note:
        detail += f" — {note}"
    return f"{ts}  {topic:<28} {ids}{detail}".rstrip()


def _cmd_timeline(args, context) -> int:
    """Read the persisted event log (``nm timeline``). Never attaches to the
    bus — it only reads what Timeline.attach() persisted earlier."""
    from ...os.timeline import Timeline

    db_path = getattr(getattr(context, "db", None), "path", None)
    tl = Timeline(db_path)
    try:
        rows = tl.query(
            session_id=getattr(args, "session", "") or None,
            project_id=getattr(args, "project", "") or None,
            mission_id=getattr(args, "mission", "") or None,
            artifact_id=getattr(args, "artifact", "") or None,
            topic=getattr(args, "topic", "") or None,
            limit=getattr(args, "limit", 50) or 50,
        )
    finally:
        tl.close()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, default=str, ensure_ascii=False))
        return 0
    if not rows:
        print("no events recorded yet")
        return 0
    for row in rows:
        print(_render_timeline_row(row))
    return 0
