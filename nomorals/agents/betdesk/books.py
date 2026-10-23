"""Session-book collector — the desk's owner-session data door.

The session-shaped books (Stake, SportyBet, MSport, Football.com —
see :mod:`.collectors`) keep their round history behind the owner's
account session: there is no endpoint the desk can poll for it.  What
the owner CAN do is export a history file from the book and drop it
in the desk's inbox.  This collector is the organ that turns those
drops into the desk's SESSION BOOK:

* on the desk's study cadence it sweeps ``<desk>/inbox/`` for new
  export files (the naming convention the inbox has always used:
  ``<provider>__<game>.csv`` or ``.jsonl``);
* each file is imported through the EXISTING import path
  (:func:`.collectors.import_rounds_file`), whose dedupe is the
  crash ledger's own identity — (provider, game, round_id) — so a
  re-imported round can never double-count;
* every import outcome is filed as a session-book record in
  ``<desk>/session_books.jsonl`` (append-only, the crash ledger's
  posture): source file, content fingerprint, provider/game,
  rounds parsed / landed / duplicate-skipped, or the named failure;
* a file whose exact content was already booked — success OR
  failure — is a true no-op on later sweeps: never re-imported,
  never re-recorded, so a corrupt export is reported once, not
  re-failed every cycle, and fixing the file (new content, new
  fingerprint) is picked up as a fresh import;
* a successfully booked file moves to ``inbox/done/``; a failed
  file stays where the owner can fix or remove it, and the ledger
  is never touched by a failed import (the import parses fully
  before anything is appended).

The book is what ``BettingDesk.status()`` shows and the study cycle
reports: what session exports the desk holds, what they landed, and
what failed — by name.  No bet placement anywhere, as everywhere in
the desk.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from .collectors import import_rounds_file

__all__ = ["SessionBookCollector", "BOOKS_FILE"]

#: The session-book ledger, beside the crash ledger in the desk dir.
BOOKS_FILE = "session_books.jsonl"


class SessionBookCollector:
    """Sweeps the desk inbox and keeps the session book."""

    def __init__(self, desk_dir: str, store: Any):
        self.dir = desk_dir
        self.store = store
        self.inbox = os.path.join(desk_dir, "inbox")
        self.done_dir = os.path.join(self.inbox, "done")
        self.path = os.path.join(desk_dir, BOOKS_FILE)

    # ── the book ledger ─────────────────────────────────────────────

    def records(self) -> list[dict[str, Any]]:
        """Every filed session-book record, oldest first.  A torn
        last line never kills the book (the crash ledger's rule)."""
        out: list[dict[str, Any]] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    def _append_record(self, rec: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _booked(self, name: str, sha: str) -> dict[str, Any] | None:
        """The record for this exact file content, if it was
        already processed — success or failure alike."""
        found: dict[str, Any] | None = None
        for rec in self.records():
            if rec.get("file") == name and rec.get("sha256") == sha:
                found = rec
        return found

    @staticmethod
    def _fingerprint(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── the sweep ───────────────────────────────────────────────────

    def sweep(self, now: float | None = None) -> list[dict[str, Any]]:
        """One collector pass over the inbox.  Returns one result
        row per file found — the same row shape the desk's collect
        report has always carried (file / ok / detail / added)."""
        now = time.time() if now is None else now
        if not os.path.isdir(self.inbox):
            return []
        results: list[dict[str, Any]] = []
        for name in sorted(os.listdir(self.inbox)):
            path = os.path.join(self.inbox, name)
            if not os.path.isfile(path):
                continue
            results.append(self._sweep_one(name, path, now))
        return results

    def _sweep_one(self, name: str, path: str,
                   now: float) -> dict[str, Any]:
        try:
            sha = self._fingerprint(path)
        except OSError as exc:
            return {"file": name, "ok": False,
                    "detail": f"unreadable export: {exc}"}
        prior = self._booked(name, sha)
        if prior is not None:
            # This exact content was already processed: a true
            # no-op.  A booked-ok file still sitting in the inbox
            # (its move to done/ was interrupted) is moved now; a
            # booked-failed file is left for the owner, unfetched
            # and un-recorded a second time.
            if prior.get("ok"):
                self._move_done(name, path)
            return {
                "file": name, "ok": bool(prior.get("ok")),
                "added": 0,
                "duplicates": int(prior.get("duplicates") or 0),
                "parsed": int(prior.get("parsed") or 0),
                "already_booked": True,
                "detail": "already in the session book — "
                          + str(prior.get("detail")
                                or f"+{int(prior.get('added') or 0)} "
                                   "rounds landed"),
            }
        stem = name.rsplit(".", 1)[0]
        parts = stem.split("__", 1)
        if len(parts) != 2:
            detail = "name must be <provider>__<game>.csv"
            self._append_record({
                "at": now, "file": name, "sha256": sha,
                "provider": "", "game": "", "ok": False,
                "parsed": 0, "added": 0, "duplicates": 0,
                "detail": detail})
            return {"file": name, "ok": False, "detail": detail}
        provider, game = parts
        try:
            res = import_rounds_file(self.store, path, provider, game)
        except Exception as exc:  # noqa: BLE001 - the failure is filed
            res = {"ok": False,
                   "detail": f"{type(exc).__name__}: {exc}",
                   "added": 0}
        rec = {
            "at": now, "file": name, "sha256": sha,
            "provider": provider, "game": game,
            "ok": bool(res.get("ok")),
            "parsed": int(res.get("parsed") or 0),
            "added": int(res.get("added") or 0),
            "duplicates": int(res.get("duplicates") or 0),
            "detail": str(res.get("detail") or ""),
        }
        self._append_record(rec)
        out = dict(res)
        out["file"] = name
        if res.get("ok"):
            self._move_done(name, path)
        return out

    def _move_done(self, name: str, path: str) -> None:
        """A booked file leaves the inbox.  Best-effort: if the move
        fails the file stays, and the book record still makes every
        later sweep a no-op for its content."""
        try:
            os.makedirs(self.done_dir, exist_ok=True)
            os.replace(path, os.path.join(self.done_dir, name))
        except OSError:
            pass

    # ── the book view ───────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """What the book holds: per-file latest state + the totals
        across every filed record (a fixed-and-reimported file
        contributes both of its records' landed rounds)."""
        recs = self.records()
        latest: dict[str, dict[str, Any]] = {}
        for rec in recs:
            latest[str(rec.get("file") or "")] = rec
        failures = [
            {"file": name, "detail": str(rec.get("detail") or "")}
            for name, rec in sorted(latest.items())
            if not rec.get("ok")]
        return {
            "files_booked": sum(1 for r in latest.values()
                                if r.get("ok")),
            "files_failed": len(failures),
            "imports_recorded": len(recs),
            "rounds_landed": sum(int(r.get("added") or 0)
                                 for r in recs),
            "duplicates_skipped": sum(int(r.get("duplicates") or 0)
                                      for r in recs),
            "last_at": max((float(r.get("at") or 0) for r in recs),
                           default=0.0),
            "failures": failures,
        }
