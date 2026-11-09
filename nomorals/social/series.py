"""Series, campaigns and the unified content calendar (WS3 P6).

The cooperative shipped drafting, review, measurement and the day
and idea cells — but content had no *shape across days*: no named
run of pieces, no one board showing what exists, what's waiting
and what's due. This module is that shape.

* A **series** is a named, ordered run of pieces. A **campaign**
  is a series with an objective — same entity, one extra field.
* Pieces are the three kinds content already has: review-queue
  **drafts**, planned **shorts** (the contentops calendar) and
  **scheduled posts** (social_posts rows). Membership lives in
  this module's own runtime tables as an ordered (kind, ref) list;
  a draft also carries the series name in its own metadata
  (``draft.metadata["series"]``) so the piece knows its home.
  There is NO parallel content store — the pieces themselves stay
  exactly where their organs keep them.
* :func:`render_calendar` is the ONE board: drafts awaiting a
  decision, the forward timeline (scheduled drafts + due shorts +
  scheduled posts together, overdue first), and the series with
  their shapes. The chat surface (``/calendar``) renders it; the
  text path is complete without buttons.
"""

from __future__ import annotations

import json
import time
from typing import Any

__all__ = [
    "SeriesStore",
    "stamp_draft",
    "render_calendar",
    "KIND_DRAFT",
    "KIND_SHORT",
    "KIND_POST",
]

KIND_DRAFT = "draft"
KIND_SHORT = "short"
KIND_POST = "post"
_KINDS = (KIND_DRAFT, KIND_SHORT, KIND_POST)

_DDL = """
CREATE TABLE IF NOT EXISTS content_series (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    objective TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS content_series_members (
    id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    position INTEGER NOT NULL,
    added_at REAL NOT NULL,
    UNIQUE(series_id, kind, ref)
);
"""


class SeriesStore:
    """Series + membership over the app's Database (runtime DDL,
    migrations.py untouched — the Batch B pattern)."""

    def __init__(self, db: Any) -> None:
        self._db = db
        db.executescript(_DDL)

    # ── series ────────────────────────────────────────────────────
    def create_series(self, name: str, *, objective: str = "") -> dict:
        from ..core.ids import new_id

        name = (name or "").strip()
        if not name:
            raise ValueError("a series needs a name")
        if self.get_by_name(name) is not None:
            raise ValueError(f"series {name!r} already exists")
        row = {"id": new_id(), "name": name,
               "objective": (objective or "").strip(),
               "status": "active", "created_at": time.time()}
        self._db.execute(
            "INSERT INTO content_series (id, name, objective, status,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (row["id"], row["name"], row["objective"], row["status"],
             row["created_at"]))
        return row

    def get(self, series_id: str) -> dict | None:
        rows = self._db.query(
            "SELECT * FROM content_series WHERE id = ?", (series_id,))
        return rows[0] if rows else None

    def get_by_name(self, name: str) -> dict | None:
        rows = self._db.query(
            "SELECT * FROM content_series WHERE name = ?", (name,))
        return rows[0] if rows else None

    def list_series(self) -> list[dict]:
        return self._db.query(
            "SELECT * FROM content_series ORDER BY created_at")

    # ── membership ────────────────────────────────────────────────
    def attach(self, series_id: str, kind: str, ref: str) -> dict:
        from ..core.ids import new_id

        if kind not in _KINDS:
            raise ValueError(
                f"unknown piece kind {kind!r} (draft | short | post)")
        ref = str(ref or "").strip()
        if not ref:
            raise ValueError("a piece needs a reference")
        existing = self._db.query(
            "SELECT * FROM content_series_members WHERE series_id = ?"
            " AND kind = ? AND ref = ?", (series_id, kind, ref))
        if existing:
            return existing[0]
        top = self._db.query(
            "SELECT MAX(position) AS p FROM content_series_members"
            " WHERE series_id = ?", (series_id,))
        position = int((top[0].get("p") if top else None) or 0) + 1
        row = {"id": new_id(), "series_id": series_id, "kind": kind,
               "ref": ref, "position": position,
               "added_at": time.time()}
        self._db.execute(
            "INSERT INTO content_series_members (id, series_id,"
            " kind, ref, position, added_at) VALUES (?, ?, ?, ?, ?, ?)",
            (row["id"], row["series_id"], row["kind"], row["ref"],
             row["position"], row["added_at"]))
        return row

    def members(self, series_id: str) -> list[dict]:
        return self._db.query(
            "SELECT * FROM content_series_members WHERE series_id = ?"
            " ORDER BY position", (series_id,))

    def series_of(self, kind: str, ref: str) -> dict | None:
        rows = self._db.query(
            "SELECT s.* FROM content_series_members m"
            " JOIN content_series s ON s.id = m.series_id"
            " WHERE m.kind = ? AND m.ref = ?",
            (kind, str(ref)))
        return rows[0] if rows else None


def stamp_draft(queue: Any, draft: Any, series_name: str) -> Any:
    """Write the series name into a draft's own metadata, through
    whichever write path the queue is on (the document-store port
    or the SQLite file) — the add_variant pattern, completed for
    both paths."""
    meta = dict(getattr(draft, "metadata", None) or {})
    meta["series"] = series_name
    if getattr(queue, "_port", None) is not None:
        queue._port.tables["social_drafts"].update(
            draft.id, {"metadata": json.dumps(meta)})
    elif getattr(queue, "_db", None) is not None:
        queue._db.execute(
            "UPDATE social_drafts SET metadata = ? WHERE id = ?",
            (json.dumps(meta), draft.id))
        queue._db.commit()
    draft.metadata = meta
    return draft


# ── the board ─────────────────────────────────────────────────────────

def _fmt_when(ts: float, now: float) -> str:
    from datetime import datetime

    dt = datetime.fromtimestamp(ts)
    label = dt.strftime("%a %d %b %H:%M")
    if ts < now:
        return f"{label} — overdue"
    if ts - now < 86400:
        return f"{label} — today" if dt.date() == datetime.fromtimestamp(now).date() else label
    return label


def _preview(text: str, limit: int = 64) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_calendar(db: Any, queue: Any, *, shorts: Any = (),
                    now: float | None = None,
                    store: SeriesStore | None = None) -> str:
    """The unified board: awaiting decision + the forward timeline
    (drafts, shorts and scheduled posts together) + the series.
    ``shorts`` is an iterable of planned-post-shaped objects
    (id/topic/scheduled_for/status) supplied by the caller — this
    L4 module never imports the media factory."""
    from .drafts import reviewable_drafts

    now = time.time() if now is None else now
    lines = ["🗓 **content calendar**", ""]

    # Awaiting a decision.
    waiting = reviewable_drafts(queue)
    if waiting:
        lines.append(f"📥 **awaiting your decision ({len(waiting)})**")
        for i, d in enumerate(waiting[:8], 1):
            state = {"draft": "draft", "pending_review": "in review",
                     "scheduled": "scheduled"}.get(d.status, d.status)
            series = (d.metadata or {}).get("series")
            tag = f" · 📚 {series}" if series else ""
            lines.append(
                f"  {i}. [{state}{tag}] {_preview(d.content)}")
        lines.append("  decide with /drafts — post, schedule or drop.")
        lines.append("")

    # The forward timeline: three kinds, one list, time-ordered.
    events: list[tuple[float, str, str]] = []
    for d in waiting:
        if d.status == "scheduled" and d.scheduled_time:
            events.append((float(d.scheduled_time), "📝",
                           f"draft — {_preview(d.content, 48)}"))
    for p in shorts or ():
        if getattr(p, "status", "") in ("queued", "rendering") \
                and getattr(p, "scheduled_for", ""):
            try:
                from datetime import datetime

                ts = datetime.fromisoformat(
                    str(p.scheduled_for).replace("Z", "+00:00")
                ).timestamp()
            except (TypeError, ValueError):
                continue
            events.append((ts, "🎬",
                           f"short — {_preview(p.topic, 48)}"
                           f" ({p.niche})"))
    if db is not None:
        try:
            rows = db.query(
                "SELECT platform, content, scheduled_at FROM"
                " social_posts WHERE status = 'scheduled'"
                " AND scheduled_at IS NOT NULL"
                " ORDER BY scheduled_at LIMIT 20")
        except Exception:  # noqa: BLE001 - a thin db may lack it
            rows = []
        for r in rows:
            events.append((float(r["scheduled_at"]), "📤",
                           f"{r['platform']} —"
                           f" {_preview(r['content'], 48)}"))
    events.sort(key=lambda e: e[0])
    horizon = now + 14 * 86400
    upcoming = [e for e in events if e[0] <= horizon][:20]
    if upcoming:
        lines.append("⏭ **coming up**")
        for ts, glyph, label in upcoming:
            lines.append(f"  {glyph} {_fmt_when(ts, now)} — {label}")
        lines.append("")
    elif not waiting:
        lines.append("nothing scheduled in the next two weeks.")
        lines.append("")

    # The series.
    store = store or (SeriesStore(db) if db is not None else None)
    if store is not None:
        series = store.list_series()
        if series:
            lines.append("📚 **series**")
            for s in series:
                members = store.members(s["id"])
                kinds: dict[str, int] = {}
                for m in members:
                    kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
                shape = ", ".join(
                    f"{n} {k}{'s' if n != 1 else ''}"
                    for k, n in sorted(kinds.items())) or "empty"
                obj = (f" — 🎯 {s['objective']}"
                       if s.get("objective") else "")
                lines.append(
                    f"  • **{s['name']}**{obj} — {shape}")
            lines.append("")
    lines.append(
        "series: `/calendar new <name> [| <objective>]` ·"
        " `/calendar add <series> draft|short|post <n>`")
    return "\n".join(lines).rstrip()
