"""Operator-facing knowledge base over trajectory failure clusters.

:class:`FailureKB` is thin by design: clustering lives in
:mod:`nomorals.cognition.trajectories`; this module adds human
operator notes on top of the clusters and surfaces the recurring
ones (the "fix me" list).

Layer rule: L3 — only :mod:`nomorals.core` (layer 1) and
:mod:`nomorals.storage` (layer 2) are imported.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ..core.ids import new_short_id
from ..storage.db import Database
from .trajectories import TrajectoryStore, cluster_key_for, normalize_error

_log = logging.getLogger(__name__)

__all__ = ["FailureKB", "guidance_for"]

_NOTES_DDL = """
CREATE TABLE IF NOT EXISTS cog_cluster_notes (
    id          TEXT PRIMARY KEY,
    cluster_key TEXT NOT NULL DEFAULT '',
    note_text   TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cog_notes_cluster
    ON cog_cluster_notes(cluster_key, created_at);
"""

#: Cluster lifecycle (Sentry-style issue states): new → resolved, and
#: "regressed" when failures recur after a resolution.
_RESOLUTIONS_DDL = """
CREATE TABLE IF NOT EXISTS cog_cluster_resolutions (
    cluster_key     TEXT PRIMARY KEY,
    resolved_at     REAL NOT NULL,
    resolution_text TEXT NOT NULL DEFAULT '',
    reopened_at     REAL
);
"""

_REPEATED_THRESHOLD = 3

#: Deterministic first-response hints per failure class.  Heuristic, not
#: diagnosis — a starting point the operator (or the brain) refines.
_GUIDANCE_BY_CLASS = {
    "rate_limited": "back off with jitter before retrying; spread load "
                    "across providers instead of hammering one.",
    "auth": "check credentials / token expiry before retrying — this "
            "class never heals on its own.",
    "context_overflow": "shrink the context (summarize, drop history) "
                         "before retrying; do not resend the same payload.",
    "timeout": "retry with a longer timeout and backoff; treat "
               "repeats as provider degradation, not a flake.",
    "network": "retry; on repeats fail over to an alternate provider "
               "or endpoint.",
    "not_found": "verify the referenced id / path exists; cache the "
                 "lookup instead of retrying blindly.",
    "invalid_request": "the request itself is malformed — fix the "
                       "payload, do not retry it.",
}


def guidance_for(failure_class: str, error_signature: str = "") -> str:
    """Heuristic first-response hint for a failure class.

    Keyword-matched against the class name so unlisted-but-related
    classes (e.g. ``"auth_expired"``) still hit.  Empty string when
    nothing matches — silence beats a wrong hint.
    """
    fc = (failure_class or "").lower()
    for key, hint in _GUIDANCE_BY_CLASS.items():
        if key in fc:
            return hint
    sig = (error_signature or "").lower()
    for key, hint in _GUIDANCE_BY_CLASS.items():
        if key in sig:
            return hint
    return ""


class FailureKB:
    """Attach operator notes to failure clusters and surface recurring ones.

    Beyond notes this class now runs the cluster lifecycle (Sentry-style
    issue states: new → resolved → regressed), priority triage, note
    search, and Voyager-style lesson cards distilled for the skill
    system.
    """

    def __init__(
        self,
        store: TrajectoryStore | None = None,
        db: Database | str | Path | None = None,
    ) -> None:
        if store is not None and db is not None:
            raise ValueError(
                "pass either store or db, not both — they would point at "
                "different databases and notes would land in the wrong one"
            )
        self.store = store if store is not None else TrajectoryStore(db)
        self.db = self.store.db
        self.db.executescript(_NOTES_DDL)
        self.db.executescript(_RESOLUTIONS_DDL)
        self._lock = threading.RLock()

    # ── notes ────────────────────────────────────────────────────────────
    def note(self, cluster_key: str, note_text: str) -> str:
        """Attach an operator note to a failure cluster.

        ``cluster_key`` is the ``task_kind::error_signature`` key (see
        :func:`nomorals.cognition.trajectories.cluster_key_for`).  An
        identical note on the same cluster is deduplicated — the
        existing note id is returned instead of writing a twin.  Returns
        the note id.
        """
        cluster_key = cluster_key or ""
        note_text = note_text or ""
        with self._lock:
            existing = self.db.query(
                "SELECT id FROM cog_cluster_notes"
                " WHERE cluster_key = ? AND note_text = ?"
                " ORDER BY created_at DESC LIMIT 1",
                (cluster_key, note_text),
            )
            if existing:
                return existing[0]["id"]
            note_id = new_short_id("note_")
            self.db.execute(
                "INSERT INTO cog_cluster_notes (id, cluster_key, note_text,"
                " created_at) VALUES (?, ?, ?, ?)",
                (note_id, cluster_key, note_text, time.time()),
            )
        _log.info("failure-kb note %s on %s", note_id, cluster_key)
        return note_id

    def note_for_error(self, task_kind: str, error: str,
                       note_text: str) -> str:
        """Attach a note given a raw error string (signature normalized).

        Broker convenience: pass the raw ``error`` exactly as recorded;
        the cluster key is derived with the same rule-aware fingerprint
        used by :meth:`TrajectoryStore.failure_clusters`, so the note
        lands on the right cluster even when operator fingerprint
        rules remap the signature.
        """
        from .trajectories import cluster_key_for

        return self.note(
            cluster_key_for(
                task_kind, self.store.fingerprint_for(task_kind, error)),
            note_text,
        )

    def search_notes(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text-ish search over operator notes (LIKE match).

        Returns newest first: ``cluster_key``, ``note_text``,
        ``created_at``.
        """
        if not query:
            return []
        rows = self.db.query(
            "SELECT id, cluster_key, note_text, created_at"
            " FROM cog_cluster_notes WHERE note_text LIKE ?"
            " ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", max(0, int(limit))),
        )
        return [
            {
                "id": row["id"],
                "cluster_key": row["cluster_key"],
                "note_text": row["note_text"],
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    # ── lifecycle (Sentry-style issue states) ──────────────────────────
    def resolve(self, cluster_key: str, resolution_text: str = "") -> bool:
        """Mark a cluster resolved — it drops off the fix-me list.

        If failures recur afterwards it shows up in :meth:`regressed`.
        """
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO cog_cluster_resolutions"
                " (cluster_key, resolved_at, resolution_text, reopened_at)"
                " VALUES (?, ?, ?, NULL)",
                (cluster_key or "", time.time(), resolution_text or ""),
            )
        _log.info("failure-kb resolved %s", cluster_key)
        return True

    def reopen(self, cluster_key: str) -> bool:
        """Reopen a resolved cluster (back to "new").  Returns False
        when the cluster was never resolved."""
        with self._lock:
            cur = self.db.execute(
                "UPDATE cog_cluster_resolutions SET reopened_at = ?"
                " WHERE cluster_key = ? AND reopened_at IS NULL",
                (time.time(), cluster_key or ""),
            )
            return (cur.rowcount or 0) > 0

    def state(self, cluster_key: str) -> str:
        """Cluster lifecycle state: ``"resolved"`` or ``"new"``."""
        rows = self.db.query(
            "SELECT reopened_at FROM cog_cluster_resolutions"
            " WHERE cluster_key = ?",
            (cluster_key or "",),
        )
        if rows and rows[0].get("reopened_at") is None:
            return "resolved"
        return "new"

    def _decorate(self, cluster: dict) -> dict:
        """Attach cluster_key, notes, lifecycle state and priority."""
        from .trajectories import cluster_key_for

        key = cluster_key_for(
            cluster["task_kind"], cluster["error_signature"])
        cluster["cluster_key"] = key
        cluster["notes"] = self._notes_for(key)
        cluster["state"] = self.state(key)
        cluster["priority"] = TrajectoryStore.cluster_priority(cluster)
        return cluster

    def _notes_for(self, cluster_key: str) -> list[dict]:
        rows = self.db.query(
            "SELECT id, note_text, created_at FROM cog_cluster_notes"
            " WHERE cluster_key = ? ORDER BY created_at DESC",
            (cluster_key or "",),
        )
        return [
            {
                "id": row["id"],
                "note_text": row["note_text"],
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    # ── lookups ──────────────────────────────────────────────────────────
    def lookup(self, task_kind: str) -> list[dict]:
        """Failure clusters for ``task_kind``, each with attached notes.

        Each dict carries the store's cluster fields (``task_kind``,
        ``failure_class``, ``exception_class``, ``error_signature``,
        ``count``, ``first_seen``, ``last_seen``, ``models_affected``,
        ``example_ids``) plus ``cluster_key``, ``notes`` (newest first),
        ``state`` (``"new"``/``"resolved"``) and ``priority``.  Empty
        when the store has no failures for the kind.
        """
        clusters = self.store.failure_clusters(task_kind, limit=10_000)
        return [self._decorate(cluster) for cluster in clusters]

    def repeated(self, limit: int = 10) -> list[dict]:
        """Clusters with ``count >= 3`` — the "fix me" list.

        Ordered most-frequent first, truncated to ``limit``.  Notes,
        state and priority are included via :meth:`lookup` semantics.
        (Prefer :meth:`triage` for the priority-ordered, open-only
        view.)
        """
        clusters = self.store.failure_clusters(None, limit=10_000)
        out = []
        for cluster in clusters:
            if cluster["count"] < _REPEATED_THRESHOLD:
                continue
            out.append(self._decorate(cluster))
            if len(out) >= max(0, int(limit)):
                break
        return out

    def triage(self, limit: int = 10) -> list[dict]:
        """The fix-me list that matters: open (unresolved) clusters
        ordered by :meth:`TrajectoryStore.cluster_priority` — count ×
        recency decay, with a bump for clusters that fired in the last
        24h.  Resolved clusters are excluded; see :meth:`regressed`."""
        clusters = self.store.failure_clusters(None, limit=10_000)
        decorated = [self._decorate(c) for c in clusters]
        open_only = [c for c in decorated if c["state"] != "resolved"]
        open_only.sort(key=lambda c: -c["priority"])
        return open_only[: max(0, int(limit))]

    def regressed(self, limit: int = 10) -> list[dict]:
        """Resolved clusters that failed again afterwards — the "it
        came back" list (Sentry's Regressed state).  Each dict carries
        the cluster fields plus ``resolved_at``, ``resolution_text``
        and ``post_resolution_count`` (failures seen after the
        resolution timestamp).
        """
        resolutions = {
            row["cluster_key"]: row
            for row in self.db.query(
                "SELECT cluster_key, resolved_at, resolution_text"
                " FROM cog_cluster_resolutions WHERE reopened_at IS NULL")
        }
        if not resolutions:
            return []
        clusters = self.store.failure_clusters(None, limit=10_000)
        out = []
        for cluster in clusters:
            key = self._decorate(cluster)["cluster_key"]
            res = resolutions.get(key)
            if res is None:
                continue
            if float(cluster["last_seen"]) > float(res["resolved_at"]):
                out.append({
                    **cluster,
                    "resolved_at": float(res["resolved_at"]),
                    "resolution_text": res.get("resolution_text") or "",
                    "post_resolution_count": self._post_resolution_count(
                        cluster, float(res["resolved_at"])),
                })
                if len(out) >= max(0, int(limit)):
                    break
        out.sort(key=lambda c: -c["post_resolution_count"])
        return out

    def _post_resolution_count(self, cluster: dict,
                               resolved_at: float) -> int:
        """Failures in this cluster's bucket seen after ``resolved_at``."""
        rows = self.db.query(
            "SELECT error, failure_class, created_at FROM cog_trajectories"
            " WHERE success = 0 AND task_kind = ? AND created_at > ?",
            (cluster.get("task_kind") or "", resolved_at),
        )
        rules = self.store._rules()
        want_class = cluster.get("failure_class") or ""
        want_fp = cluster.get("error_signature") or ""
        n = 0
        for row in rows:
            err = row.get("error") or ""
            hit = self.store._apply_rules(err, rules)
            fp = hit if hit is not None else normalize_error(err)
            if (row.get("failure_class") or "") == want_class and fp == want_fp:
                n += 1
        return n

    # ── lessons → the skill-distillation feed ──────────────────────────
    def lessons(self, limit: int = 5) -> list[dict]:
        """Voyager-style verified lesson cards for the skill system.

        One card per top triage cluster: the failure fingerprint, the
        evidence (count, blast radius), the operator notes, and a
        first-response hint.  Cards are prompt-ready text the brain can
        index and retrieve before retrying the same task kind.
        """
        cards = []
        for cluster in self.triage(limit=limit):
            hint = guidance_for(cluster.get("failure_class") or "",
                                cluster.get("error_signature") or "")
            notes = [n["note_text"] for n in cluster.get("notes", [])]
            cards.append({
                "cluster_key": cluster["cluster_key"],
                "task_kind": cluster["task_kind"],
                "exception_class": cluster.get("exception_class") or "",
                "failure_class": cluster.get("failure_class") or "",
                "error_signature": cluster["error_signature"],
                "count": cluster["count"],
                "models_affected": cluster.get("models_affected") or [],
                "priority": cluster["priority"],
                "notes": notes,
                "guidance": hint,
                "lesson_text": self._lesson_text(cluster, notes, hint),
            })
        return cards

    @staticmethod
    def _lesson_text(cluster: dict, notes: list[str], hint: str) -> str:
        bits = [
            f"LESSON — {cluster['task_kind']}"
            f"::{cluster['error_signature']}",
            f"Seen {cluster['count']}×"
            + (f" across {', '.join(cluster.get('models_affected') or [])}"
               if cluster.get("models_affected") else ""),
        ]
        if notes:
            bits.append("Operator notes: " + " | ".join(notes[:3]))
        if hint:
            bits.append("First response: " + hint)
        return "\n".join(bits)

    # ── operator surface ───────────────────────────────────────────────
    def report(self, limit: int = 10) -> str:
        """Markdown fix-me list for the operator ("how broken are we")."""
        triage = self.triage(limit=limit)
        reg = self.regressed(limit=limit)
        lines = ["🔧 Failure triage"]
        if not triage and not reg:
            lines.append("No failure clusters on record — nothing to fix.")
            return "\n".join(lines)
        if reg:
            lines.append("")
            lines.append(f"🚨 Regressed ({len(reg)} — fixed, then broke again):")
            for c in reg:
                lines.append(
                    f"• {c['task_kind']}::{c['error_signature']}"
                    f" — {c['post_resolution_count']}× since fix"
                    f" (was: {c['resolution_text'][:60]})")
        if triage:
            lines.append("")
            lines.append(f"⚠️ Open clusters ({len(triage)}), fix first:")
            for c in triage:
                exc = f" [{c['exception_class']}]" if c.get(
                    "exception_class") else ""
                hint = guidance_for(c.get("failure_class") or "",
                                    c.get("error_signature") or "")
                lines.append(
                    f"• {c['task_kind']}::{c['error_signature']}{exc}"
                    f" — {c['count']}×, priority {c['priority']}")
                if c["notes"]:
                    lines.append(f"  ↳ note: {c['notes'][0]['note_text'][:100]}")
                if hint:
                    lines.append(f"  ↳ try: {hint[:100]}")
        return "\n".join(lines)
