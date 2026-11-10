"""Sensor feeds — the immune system's inbound senses (doc 20 P5).

Before P5 the immune organs could only see what the chat surface
typed at them: the adaptive baseline had its live process sampler,
but the rest of Devon's failure experience never became immune
evidence. Two organs already recorded exactly the signals an
immune system needs — the tool registry's failure ledger (every
real tool error, attributed with a fingerprint) and the per-chat
drift/fever monitors — and their observations evaporated: ledger
rows were read by nobody immune, and drift/fever state died with
the chat state at restart. P5 is the feed layer that connects
them, under the programme's standing rules:

* **Feeds observe and case-build ONLY.** A feed opens (or extends)
  a case file citing its source rows; it never creates an
  antibody, never promotes anything, never enforces. Learning
  still flows through the shipped scan → thymus → probation path
  (P3), and enforcement through the shipped organs.

* **The failure-ledger feed** reads the app DB's ``failures``
  table (``source='tool'`` rows — the registry's
  ``_record_failure_ledger`` shape) over a trailing window and
  groups by (tool, signature): the row's ``skill`` is the tool,
  its ``fingerprint`` the signature, with a ``nofp:`` fallback
  derived from the normalized error text for fingerprint-less
  rows. A group at or over the recurrence threshold opens ONE
  case (idempotent per tool + signature + window bucket);
  further failures in the same bucket extend that case with an
  evidence item citing only the new ids; a SEALED case is never
  extended — a fresh case (``#2`` feed key) opens beside it.
  Evaluation is LAZY, on the immune surface's status / casefile /
  feeds reads: the sampler's ``monitor.tick()`` context carries
  no db and no case collection, so riding the tick doesn't fit,
  and no new thread is started (P3's lazy precedent).

* **The observation journal** (:class:`ObservationLog`) is the
  durable, append-only record of what the drift and fever organs
  observed and what false-positive verdicts landed: ``drift_day``,
  ``fever_started``, ``fever_ended``, ``false_positive`` records
  in a home-scoped JSONL file (wiring-shared, one per path).
  Appends never raise — a broken log degrades with the reason
  recorded, never sinks the surface that observed something.

Layer 0: standard library only, like the rest of this package.
The ledger is read through whatever db-shaped object the caller
injects (anything with ``execute(sql) -> cursor``); this module
imports nothing from ``nomorals.*``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .casefile import CaseFile, build_case

__all__ = [
    "FEED_MIN_FAILURES",
    "FEED_WINDOW_S",
    "FEED_ENV_MIN_FAILURES",
    "FEED_ENV_WINDOW_S",
    "OBSERVATION_KINDS",
    "ObservationLog",
    "feed_policy",
    "read_failure_rows",
    "evaluate_failure_feed",
]

_log = logging.getLogger(__name__)

#: Recurrence policy defaults. Five failures of the same tool with
#: the same signature inside an hour is a recurrence worth a case
#: file; the floor sits above the incident journal's chronic-1h
#: floor of 3 so recovered retry storms (a tool that failed twice
#: and then healed) do not open cases — only persistence does.
FEED_MIN_FAILURES = 5
FEED_WINDOW_S = 3600.0

#: Env overrides, in the package's ``NM_IMMUNE_*`` convention.
FEED_ENV_MIN_FAILURES = "NM_IMMUNE_FEED_MIN_FAILURES"
FEED_ENV_WINDOW_S = "NM_IMMUNE_FEED_WINDOW_S"

#: The observation kinds the journal records. ``drift_day`` — one
#: drift-monitor day (metrics + verdict); ``fever_started`` /
#: ``fever_ended`` — a fever window on a chat surface;
#: ``false_positive`` — an owner's verdict on an antibody (the
#: ``/immune mark-fp`` seat, doc 20 P5).
OBSERVATION_KINDS = (
    "drift_day", "fever_started", "fever_ended", "false_positive")

#: Detailed per-row citations kept on a feed case. The summary
#: item cites EVERY failure id in the group; the detail items are
#: the first rows spelled out (id, time, error) so a reader can
#: open the ledger at the right places without a second query.
_MAX_DETAIL_CITATIONS = 10


def feed_policy() -> tuple[int, float]:
    """The recurrence policy: ``(min_failures, window_s)``.

    Env-overridable (``NM_IMMUNE_FEED_MIN_FAILURES`` /
    ``NM_IMMUNE_FEED_WINDOW_S``); malformed or non-positive
    overrides fall back to the defaults — the promotion-policy
    discipline: a typo in the environment must never silently
    weaken or break the gate.
    """
    min_failures = FEED_MIN_FAILURES
    raw = os.environ.get(FEED_ENV_MIN_FAILURES, "")
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            min_failures = parsed
    window_s = FEED_WINDOW_S
    raw = os.environ.get(FEED_ENV_WINDOW_S, "")
    if raw:
        try:
            parsed_f = float(raw)
        except ValueError:
            parsed_f = -1.0
        if parsed_f > 0:
            window_s = parsed_f
    return min_failures, window_s


# ── the observation journal ─────────────────────────────────────

class ObservationLog:
    """Append-only JSONL journal of immune observations.

    One shared instance per file in production (see
    ``immune/wiring.shared_observation_log``). ``path=None`` is an
    in-memory log (the records live for the session only — the
    chat surface's fallback when the shared log cannot load, and
    the shape tests use). Records are plain dicts:
    ``{"kind", "ts", ...payload}``.

    Discipline: appends NEVER raise (a failed append records
    :attr:`degraded_reason` and returns False — the observation
    mattered less than the surface that made it), and reads skip
    torn lines (a crash mid-append leaves at most one partial
    line; the journal stays readable).
    """

    def __init__(self, path: Any = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._memory: list[dict[str, Any]] = []
        #: Why the on-disk journal could not be written/read, when
        #: it could not. Empty while the log is healthy: a degraded
        #: journal says so on the read surface, never silently.
        self.degraded_reason = ""

    def append(self, kind: str, payload: dict[str, Any] | None = None,
               *, now: float | None = None) -> bool:
        """Append one observation. Returns True when it landed."""
        record = dict(payload or {})
        record["kind"] = str(kind)
        record["ts"] = time.time() if now is None else float(now)
        with self._lock:
            if self.path is None:
                self._memory.append(record)
                return True
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    # A torn previous write (crash mid-append) can
                    # leave the file without a trailing newline;
                    # appending directly would glue this record onto
                    # the torn line and lose both. Terminate the
                    # torn line first — readers skip it, this
                    # record stays clean.
                    if self.path.exists() and \
                            self.path.stat().st_size > 0:
                        with open(self.path, "rb") as tail:
                            tail.seek(-1, os.SEEK_END)
                            if tail.read(1) != b"\n":
                                fh.write("\n")
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
            except OSError as exc:
                self.degraded_reason = (
                    f"observation log {self.path} unwritable: {exc}")
                _log.warning("immune observation append failed (%s): %s",
                             self.path, exc)
                return False
            return True

    def records(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Every readable record, oldest first (one kind, if given).

        Torn lines — partial JSON from a crash mid-append, or a
        line that is not an object — are skipped, never fatal.
        """
        with self._lock:
            if self.path is None:
                out = list(self._memory)
            else:
                out = []
                try:
                    lines = self.path.read_text(
                        encoding="utf-8").splitlines()
                except FileNotFoundError:
                    lines = []
                except OSError as exc:
                    self.degraded_reason = (
                        f"observation log {self.path} unreadable: {exc}")
                    _log.warning("immune observation read failed (%s): %s",
                                 self.path, exc)
                    lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue  # torn line — skip, never fatal
                    if isinstance(record, dict):
                        out.append(record)
        if kind is not None:
            out = [r for r in out if r.get("kind") == kind]
        return out

    def count(self, kind: str | None = None) -> int:
        return len(self.records(kind))

    def clear(self) -> None:
        """Empty the journal (the reset seam behind
        ``immune/chat.reset_immune_state`` — test/support only,
        the corpus/antibody clear discipline)."""
        with self._lock:
            self._memory.clear()
            self.degraded_reason = ""
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self.path.write_text("", encoding="utf-8")
                except OSError as exc:
                    self.degraded_reason = (
                        f"observation log {self.path} uncleanable: {exc}")
                    _log.warning(
                        "immune observation clear failed (%s): %s",
                        self.path, exc)


# ── the failure-ledger feed ─────────────────────────────────────

def _row_field(row: Any, key: str, index: int) -> Any:
    """One field from a ledger row, dict-shaped or positional."""
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None


def _effective_ts(row: dict[str, Any]) -> float:
    """The row's failure time: ``ts`` when the writer stamped it,
    else ``created_at`` (the registry's ledger rows carry ``ts``;
    other writers' rows may only carry the table's own column)."""
    for value in (row.get("ts"), row.get("created_at")):
        try:
            ts = float(value or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts > 0:
            return ts
    return 0.0


def _signature_for(row: dict[str, Any]) -> str:
    """The recurrence signature: the row's fingerprint, or a
    ``nofp:`` digest of the normalized error text for rows that
    carry none — fingerprint-less rows still recur visibly, under
    a signature that names its own derivation."""
    fingerprint = str(row.get("fingerprint") or "").strip()
    if fingerprint:
        return fingerprint
    normalized = " ".join(
        str(row.get("error") or "").lower().split())[:500]
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"nofp:{digest}"


def read_failure_rows(db: Any, *, now: float | None = None,
                      window_s: float | None = None,
                      ) -> tuple[list[dict[str, Any]] | None, str]:
    """The ``source='tool'`` failure rows inside the trailing
    window, normalized to plain dicts.

    Returns ``(rows, "")`` on success and ``(None, reason)`` when
    the ledger cannot be read at all — no database attached, or
    the query failed. UNAVAILABLE is a stated fact with its
    reason, never an exception and never a fabricated empty read.
    """
    if db is None:
        return None, ("no database attached to this surface — "
                      "the failure ledger cannot be read")
    if not hasattr(db, "execute"):
        return None, "the attached database cannot execute queries"
    _min, default_window = feed_policy()
    window = default_window if window_s is None else float(window_s)
    now = time.time() if now is None else float(now)
    try:
        cursor = db.execute(
            "SELECT id, skill, fingerprint, error, ts, created_at "
            "FROM failures WHERE source = 'tool'")
        raw_rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001 — UNAVAILABLE, never a crash
        return None, f"failure ledger unreadable: {exc}"
    rows: list[dict[str, Any]] = []
    for raw in raw_rows or []:
        row = {
            "id": str(_row_field(raw, "id", 0) or ""),
            "tool": str(_row_field(raw, "skill", 1) or ""),
            "fingerprint": str(_row_field(raw, "fingerprint", 2) or ""),
            "error": str(_row_field(raw, "error", 3) or ""),
            "ts": _row_field(raw, "ts", 4),
            "created_at": _row_field(raw, "created_at", 5),
        }
        row["effective_ts"] = _effective_ts(row)
        if row["effective_ts"] >= now - window:
            rows.append(row)
    rows.sort(key=lambda r: r["effective_ts"])
    return rows, ""


def _feed_cases(casefiles: Iterable[CaseFile], base_key: str,
                ) -> list[tuple[int, CaseFile]]:
    """The feed cases for one base key, as (suffix, case) sorted:
    the original case is suffix 1; successors opened beside a
    sealed case carry ``#2``, ``#3``, …"""
    out: list[tuple[int, CaseFile]] = []
    for case in casefiles:
        event = getattr(case, "event", None)
        if not isinstance(event, dict):
            continue
        if event.get("kind") != "tool_failure_recurrence":
            continue
        key = str(event.get("feed_key") or "")
        if key == base_key:
            out.append((1, case))
        elif key.startswith(base_key + "#"):
            suffix = key[len(base_key) + 1:]
            if suffix.isdigit():
                out.append((int(suffix), case))
    out.sort(key=lambda pair: pair[0])
    return out


def _cited_ids(case: CaseFile) -> list[str]:
    """Every failure id a feed case already cites: the opening
    event's ids plus each extension item's new ids, in order."""
    ids: list[str] = []
    seen: set[str] = set()
    event = case.event if isinstance(case.event, dict) else {}
    for fid in event.get("failure_ids") or []:
        if fid not in seen:
            seen.add(fid)
            ids.append(fid)
    for item in case.evidence:
        if isinstance(item, dict) \
                and item.get("kind") == "failure_ledger_update":
            for fid in item.get("new_failure_ids") or []:
                if fid not in seen:
                    seen.add(fid)
                    ids.append(fid)
    return ids


def _open_feed_case(casefiles: list[CaseFile], feed_key: str,
                    tool: str, signature: str,
                    group: list[dict[str, Any]]) -> CaseFile:
    """Open the case for one recurrence group: the event IS the
    recurrence (kind, feed key, tool, signature, count, span, the
    full id list), the evidence is one summary item citing every
    id plus up to ``_MAX_DETAIL_CITATIONS`` detailed row
    citations."""
    ids = [row["id"] for row in group]
    event = {
        "kind": "tool_failure_recurrence",
        "source": "failure_ledger_feed",
        "feed_key": feed_key,
        "tool": tool,
        "signature": signature,
        "count": len(group),
        "first_ts": group[0]["effective_ts"],
        "last_ts": group[-1]["effective_ts"],
        "failure_ids": ids,
    }
    case = build_case(event)
    case.add_evidence({
        "kind": "failure_ledger_summary",
        "source": "failure_ledger_feed",
        "feed_key": feed_key,
        "tool": tool,
        "signature": signature,
        "count": len(group),
        "first_ts": event["first_ts"],
        "last_ts": event["last_ts"],
        "failure_ids": ids,
    })
    for row in group[:_MAX_DETAIL_CITATIONS]:
        case.add_evidence({
            "kind": "failure_ledger_row",
            "source": "failure_ledger_feed",
            "failure_id": row["id"],
            "tool": tool,
            "signature": signature,
            "ts": row["effective_ts"],
            "error": row["error"][:200],
        })
    casefiles.append(case)
    return case


def evaluate_failure_feed(db: Any, casefiles: list[CaseFile], *,
                          now: float | None = None,
                          ) -> dict[str, Any]:
    """One lazy evaluation of the failure-ledger feed.

    Groups the window's tool failures by (tool, signature); each
    group at or over the threshold opens its case or extends the
    open one. Idempotent per (tool, signature, window bucket): a
    re-evaluation that finds no new failure ids changes nothing.
    A sealed case is never extended — the recurrence opens a
    fresh case beside it under the next ``#n`` feed key.

    The report is the feed's honesty surface: ``available`` with
    the real counts, or ``available: False`` with the reason —
    and zero cases either way it cannot read.
    """
    min_failures, window_s = feed_policy()
    now = time.time() if now is None else float(now)
    report: dict[str, Any] = {
        "available": False,
        "reason": "",
        "rows": 0,
        "groups": 0,
        "groups_over_threshold": 0,
        "opened": [],
        "extended": [],
        "min_failures": min_failures,
        "window_s": window_s,
    }
    rows, reason = read_failure_rows(db, now=now, window_s=window_s)
    if rows is None:
        report["reason"] = reason
        return report
    report["available"] = True
    report["rows"] = len(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["tool"], _signature_for(row))
        groups.setdefault(key, []).append(row)
    report["groups"] = len(groups)
    bucket = int(now // window_s) if window_s > 0 else 0
    for (tool, signature), group in sorted(groups.items()):
        if len(group) < min_failures:
            continue
        report["groups_over_threshold"] += 1
        base_key = f"tool_failure:{tool}:{signature}:{bucket}"
        existing = _feed_cases(casefiles, base_key)
        if existing:
            suffix, case = existing[-1]
            if not case.sealed:
                cited = set(_cited_ids(case))
                new_rows = [r for r in group if r["id"] not in cited]
                if not new_rows:
                    continue  # no change — a true no-op
                case.add_evidence({
                    "kind": "failure_ledger_update",
                    "source": "failure_ledger_feed",
                    "feed_key": str(case.event.get("feed_key") or ""),
                    "new_failure_ids": [r["id"] for r in new_rows],
                    "count": len(cited) + len(new_rows),
                    "first_ts": new_rows[0]["effective_ts"],
                    "last_ts": new_rows[-1]["effective_ts"],
                })
                report["extended"].append(
                    str(case.event.get("feed_key") or ""))
                continue
            feed_key = f"{base_key}#{suffix + 1}"
        else:
            feed_key = base_key
        _open_feed_case(casefiles, feed_key, tool, signature, group)
        report["opened"].append(feed_key)
    return report
