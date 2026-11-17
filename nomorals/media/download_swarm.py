"""Parallel segmented downloading — the download swarm.

Today Devon downloads one file at a time on a single stream: the
kit's :class:`ResumeDownloader` resumes, but it fetches one range
from byte 0 (or from the resume point) to the end, and nothing in
the repo runs two ranges at once.  This module is the parallel
worker pool the profile budget already names: ``runtune`` sizes
``max_concurrent_downloads`` per machine (1 on a phone, 2 on a
small cloud box, 4 on a laptop, 8 on a workstation) and
``runtime.py`` prints it in a status line — but no worker pool
consumed it.  This one does.

Per file the swarm:

* HEADs for size + ``Accept-Ranges`` + validators (ETag /
  Last-Modified) — the same validator discipline the kit's
  single-stream resume already proves;
* splits the file into contiguous byte ranges (one per worker,
  never smaller than 1 MiB a segment) and fetches them
  concurrently on a ``ThreadPoolExecutor`` sized by the profile
  budget;
* writes each segment to its own part file, hashing as it lands,
  and reports progress on the tracker's two honest axes — bytes
  for the transfer, ``report_segments`` for the segment count
  (never mixed);
* reassembles by concatenating the part files in order, with a
  whole-file sha256 over the result;
* resumes from durable state (``<dest>.swarm.json`` + part files):
  completed segments are re-hashed before they are trusted, only
  the missing ranges are fetched, and a validator change discards
  everything and restarts from byte 0 — never a splice across a
  changed file, the kit's invariant kept;
* REFUSES cleanly — never a half-download presented as a plan —
  when the server will not serve ranges, answers a range request
  with the whole file, hides the size, or the file is below the
  swarm threshold: the caller falls back to the single-stream kit.

Layering: media is L4.  Stdlib + core only; no tools, no
connectors, no agents.  Transports are injectable (``head`` /
``get_range_stream``) so tests drive the whole state machine
without a network, exactly like the kit.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .download_kit_failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "DEFAULT_MIN_SWARM_BYTES",
    "MIN_SEGMENT_BYTES",
    "SEGMENT_RETRIES",
    "DEFAULT_WORKERS",
    "MAX_WORKERS",
    "SwarmReport",
    "SwarmDownloader",
    "SwarmQueue",
    "SwarmPool",
    "plan_segments",
    "resolve_workers",
    "default_queue_path",
]

#: A file smaller than this is not worth segmenting — the
#: single-stream kit is cheaper (connection setup dominates).
DEFAULT_MIN_SWARM_BYTES = 8 << 20
#: The smallest range one worker fetches — smaller segments would
#: spend more time handshaking than transferring.
MIN_SEGMENT_BYTES = 1 << 20
#: Extra fetches of one segment after a transport failure.
SEGMENT_RETRIES = 2
#: Worker-count bounds.  The top is the workstation profile's
#: ``max_concurrent_downloads`` (runtune); the floor is one worker,
#: which is the single-stream shape with swarm bookkeeping.
MIN_WORKERS = 1
MAX_WORKERS = 8
#: config.py:305's default when no tune is reachable.
DEFAULT_WORKERS = 4

#: How long a ``running`` queue job must be untouched before the
#: pool treats it as orphaned (crashed mid-download) and re-queues
#: it for resume.
STALE_RUNNING_S = 300.0

#: One lock for every queue instance in this process: the runtime's
#: pool and the per-call command queues open the SAME file through
#: different instances, so a cancel issued by one must be visible
#: to the other — the lock makes reload → mutate → save atomic
#: across instances (the two runtimes on the box use different
#: homes, hence different queue files, so this stays in-process).
_QUEUE_LOCK = threading.Lock()


class _RangeRefused(Exception):
    """The server will not honour byte ranges for this file."""


def _call_progress(progress: Callable[[int, int | None], None] | None,
                   received: int, expected: int | None) -> None:
    """Invoke a progress sink; a broken sink never kills a download."""
    if progress is None:
        return
    try:
        progress(received, expected)
    except Exception as e:  # noqa: BLE001 — reporting is not the payload
        record_failure("SwarmDownloader.progress", e)


def _call_tracker(tracker: Any, fn: str, *args: Any) -> None:
    """Invoke a tracker mutator; a broken tracker never kills a
    download (the tracker is a render surface, not the payload)."""
    if tracker is None:
        return
    try:
        getattr(tracker, fn)(*args)
    except Exception as e:  # noqa: BLE001
        record_failure(f"SwarmDownloader.tracker.{fn}", e)


def plan_segments(total: int, workers: int, *,
                  min_segment_bytes: int = MIN_SEGMENT_BYTES
                  ) -> list[tuple[int, int]]:
    """Split *total* bytes into contiguous inclusive ranges.

    One range per worker, never smaller than ``min_segment_bytes``
    — a file too small for the requested workers simply uses
    fewer.  Ranges tile the file exactly: first starts at 0, last
    ends at ``total - 1``, no gaps, no overlap.  Unknown or empty
    totals plan nothing.
    """
    total = int(total or 0)
    if total <= 0:
        return []
    workers = max(MIN_WORKERS, min(MAX_WORKERS, int(workers or 1)))
    workers = min(workers,
                  max(MIN_WORKERS, total // max(1, int(min_segment_bytes))))
    if workers <= 1:
        return [(0, total - 1)]
    base, rem = divmod(total, workers)
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size - 1))
        start += size
    return ranges


def resolve_workers(context: Any = None,
                    explicit: int | None = None) -> int:
    """How many range workers a swarm may run.

    Explicit wins; then the runtime tune bound to the context's
    extras (the same ``max_concurrent_downloads`` the profile
    status line prints); then the config default.  Always clamped
    to 1..8.  Never raises — a missing tune is a default, not a
    failure.
    """
    if explicit is not None:
        try:
            return int(max(MIN_WORKERS,
                           min(MAX_WORKERS, int(explicit))))
        except (TypeError, ValueError):
            pass
    try:
        extras = getattr(context, "extras", None) or {}
        tune = extras.get("tune") if hasattr(extras, "get") else None
        if tune is not None:
            return int(max(MIN_WORKERS, min(
                MAX_WORKERS, int(tune.max_concurrent_downloads))))
    except Exception as e:  # noqa: BLE001 — a tune problem is a default
        record_failure("SwarmDownloader.resolve_workers", e)
    return DEFAULT_WORKERS


def default_queue_path(context: Any = None) -> Path:
    """Where the durable swarm queue lives.

    ``$NM_HOME/data/download_swarm_queue.json`` (``~/.nomorals``
    when NM_HOME is unset) — the same data-home shape the cookie
    store uses.  A context carrying an explicit ``swarm_queue_path``
    in extras wins (tests, one-off callers).
    """
    try:
        extras = getattr(context, "extras", None) or {}
        pinned = extras.get("swarm_queue_path") \
            if hasattr(extras, "get") else None
        if pinned:
            return Path(str(pinned)).expanduser()
    except Exception as e:  # noqa: BLE001 — a pinned path is optional
        record_failure("SwarmDownloader.default_queue_path", e)
    home = Path(os.environ.get("NM_HOME") or "~/.nomorals").expanduser()
    return home / "data" / "download_swarm_queue.json"


@dataclass
class SwarmReport:
    """The outcome of one swarm download.  Never raises — a refusal
    or a failure is a report, and the caller decides the fallback."""

    ok: bool = False
    path: str = ""
    size_bytes: int = 0
    segments_total: int = 0
    segments_done: int = 0
    resumed: bool = False
    restarted: bool = False
    sha256: str = ""
    error: str = ""
    refused_reason: str = ""
    segment_attempts: int = 0
    cancelled: bool = False


@dataclass
class _Segment:
    index: int
    start: int
    end: int
    sha256: str = ""
    complete: bool = False

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start": self.start, "end": self.end,
                "sha256": self.sha256, "complete": bool(self.complete)}


class SwarmDownloader:
    """Fetch one direct file URL with parallel byte-range workers.

    Transports are injectable (``head`` / ``get_range_stream``) so
    tests drive the validator, resume and refusal logic without a
    network; the defaults are plain urllib, the kit's own shape.
    """

    def __init__(self, *,
                 head: Callable[[str], dict[str, Any]] | None = None,
                 get_range_stream: Callable[..., Any] | None = None
                 ) -> None:
        self._head = head or self._default_head
        self._get_range_stream = (
            get_range_stream or self._default_get_range_stream)

    # ── default transports (urllib) ──────────────────────────────────

    @staticmethod
    def _default_head(url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "DevonBot/1.0"})
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            return {"status": resp.status,
                    "etag": resp.headers.get("ETag", ""),
                    "last_modified": resp.headers.get("Last-Modified", ""),
                    "content_length": resp.headers.get("Content-Length", ""),
                    "accept_ranges": resp.headers.get("Accept-Ranges", "")}

    @staticmethod
    def _default_get_range_stream(url: str, start: int, end: int | None,
                                  headers: dict[str, str] | None = None):
        """Yield one range in bounded pieces.  A non-206 answer to a
        range request is a refusal, raised here — the length check
        in the worker is the second line of defence for transports
        that cannot see the status."""
        hdrs = {"User-Agent": "DevonBot/1.0"}
        if start or end is not None:
            if end is None:
                hdrs["Range"] = f"bytes={start}-"
            else:
                hdrs["Range"] = f"bytes={start}-{end}"
        hdrs.update(headers or {})
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            if getattr(resp, "status", 200) != 206:
                raise _RangeRefused(
                    f"server answered a range request with HTTP "
                    f"{getattr(resp, 'status', '?')} instead of 206 — "
                    "byte ranges are not honoured")
            while True:
                chunk = resp.read(1 << 18)
                if not chunk:
                    break
                yield chunk

    # ── state on disk ────────────────────────────────────────────────

    @staticmethod
    def _state_path(dest: Path) -> Path:
        return dest.with_suffix(dest.suffix + ".swarm.json")

    @staticmethod
    def _segment_path(dest: Path, index: int) -> Path:
        return dest.parent / (dest.name + f".part-{index}")

    def _load_state(self, dest: Path) -> dict[str, Any]:
        path = self._state_path(dest)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as e:  # noqa: BLE001 — corrupt state restarts
            record_failure("SwarmDownloader.load_state", e)
            return {}

    def _save_state(self, dest: Path, state: dict[str, Any]) -> None:
        path = self._state_path(dest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            record_failure("SwarmDownloader.save_state", e)

    def _discard_state(self, dest: Path, segments: int) -> None:
        """Remove every part file and the state file — the whole
        attempt is void (validators changed, or ranges refused)."""
        for i in range(segments):
            try:
                self._segment_path(dest, i).unlink(missing_ok=True)
            except OSError as e:
                record_failure("SwarmDownloader.discard_state", e)
        try:
            self._state_path(dest).unlink(missing_ok=True)
        except OSError as e:
            record_failure("SwarmDownloader.discard_state", e)

    def discard(self, dest: str | Path) -> None:
        """Drop a swarm attempt's leftovers for *dest* (part files +
        state).  Best-effort: a caller that landed the file another
        way calls this so no stale part files outlive the download."""
        dest_p = Path(dest)
        try:
            for part in dest_p.parent.glob(dest_p.name + ".part-*"):
                part.unlink(missing_ok=True)
        except OSError as e:
            record_failure("SwarmDownloader.discard", e)
        try:
            self._state_path(dest_p).unlink(missing_ok=True)
        except OSError as e:
            record_failure("SwarmDownloader.discard", e)

    # ── one segment ──────────────────────────────────────────────────

    def _fetch_segment(self, url: str, dest: Path, seg: _Segment,
                       segment_retries: int,
                       should_stop: Callable[[], bool] | None = None
                       ) -> str:
        """Fetch one range into its part file.

        Returns ``(sha256, attempts_made)``.  Retries a transport
        failure; a range refusal is not a retry — the server will
        not honour the range on the next try either.  A short or
        over-long landing is the same refusal (the length is the
        range contract, checked here so transports that cannot see
        the HTTP status are still caught).  ``should_stop`` is the
        cooperative cancel: a retry never starts after it fires.
        """
        part = self._segment_path(dest, seg.index)
        attempts = 0
        while True:
            attempts += 1
            digest = hashlib.sha256()
            written = 0
            try:
                with part.open("wb") as fh:
                    for chunk in self._get_range_stream(
                            url, seg.start, seg.end):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != seg.size:
                    raise _RangeRefused(
                        f"range bytes={seg.start}-{seg.end} landed "
                        f"{written} bytes, expected {seg.size} — the "
                        "server did not honour the range")
                return digest.hexdigest(), attempts
            except _RangeRefused:
                raise
            except Exception as exc:  # noqa: BLE001 — retry, then fail
                if should_stop is not None and should_stop():
                    # Cancelled while retrying: stop here, do not
                    # start another fetch after the stop.  The
                    # cancellation is deliberate, not caused by the
                    # transport failure, so it carries no cause.
                    raise CancelledError("swarm cancelled") from None
                if attempts > segment_retries:
                    raise
                _log.info("swarm segment %d attempt %d failed (%s) — "
                          "retrying", seg.index, attempts, exc)

    # ── main path ────────────────────────────────────────────────────

    def download(self, url: str, dest: str | Path, *,
                 workers: int | None = None,
                 min_swarm_bytes: int = DEFAULT_MIN_SWARM_BYTES,
                 segment_retries: int = SEGMENT_RETRIES,
                 progress: Callable[[int, int | None], None] | None = None,
                 tracker: Any | None = None,
                 executor: ThreadPoolExecutor | None = None,
                 should_stop: Callable[[], bool] | None = None
                 ) -> SwarmReport:
        """Fetch *url* to *dest* with parallel range workers.

        Returns a report, never raises.  ``refused_reason`` names
        why the swarm declined (the caller falls back to the
        single-stream kit); ``error`` names why a started swarm
        failed (the partial state stays on disk for resume).
        ``should_stop`` is cooperative cancellation: when it
        returns True the download stops at the next segment
        boundary — in-flight segments finish, none start after —
        the part files are discarded, and the report carries
        ``cancelled=True``.
        """
        dest_p = Path(dest)
        report = SwarmReport()

        def _stopped() -> bool:
            # A broken stop check never stops a download.
            if should_stop is None:
                return False
            try:
                return bool(should_stop())
            except Exception as e:  # noqa: BLE001
                record_failure("SwarmDownloader.should_stop", e)
                return False

        try:
            head = self._head(url)
        except Exception as exc:  # noqa: BLE001 — no HEAD, no swarm
            report.refused_reason = f"HEAD failed: {exc}"
            return report
        validators = {"etag": str(head.get("etag") or ""),
                      "last_modified": str(head.get("last_modified") or "")}
        try:
            total = int(head.get("content_length") or 0)
        except (TypeError, ValueError):
            total = 0
        if total <= 0:
            report.refused_reason = (
                "server did not report a content length — a swarm "
                "cannot plan segments without a known size")
            return report
        accept = str(head.get("accept_ranges") or "").lower()
        if "bytes" not in accept:
            report.refused_reason = (
                "server does not serve byte ranges "
                f"(Accept-Ranges: {accept or 'missing'})")
            return report
        if total < min_swarm_bytes:
            report.refused_reason = (
                f"file is {total} bytes, below the "
                f"{min_swarm_bytes} swarm threshold — single-stream "
                "is cheaper")
            return report

        n_workers = max(MIN_WORKERS, min(
            MAX_WORKERS, int(workers) if workers is not None
            else DEFAULT_WORKERS))
        ranges = plan_segments(total, n_workers)
        report.segments_total = len(ranges)

        # ── resume: trust nothing without its validators ──────────
        saved = self._load_state(dest_p)
        saved_segments = saved.get("segments") or []
        validators_changed = bool(saved) and (
            saved.get("url") != url
            or saved.get("total") != total
            or saved.get("etag") != validators["etag"]
            or saved.get("last_modified") != validators["last_modified"])
        segments = [_Segment(index=i, start=s, end=e)
                    for i, (s, e) in enumerate(ranges)]
        resumed_any = False
        if validators_changed:
            _log.info("swarm: validators changed for %s — restarting, "
                      "not splicing", url[:80])
            self._discard_state(dest_p, len(ranges))
            report.restarted = True
        elif saved:
            by_index = {int(s.get("index", -1)): s
                        for s in saved_segments
                        if isinstance(s, dict)}
            for seg in segments:
                prior = by_index.get(seg.index)
                if not prior or not prior.get("complete"):
                    continue
                part = self._segment_path(dest_p, seg.index)
                if not part.is_file():
                    continue
                digest = hashlib.sha256()
                try:
                    with part.open("rb") as fh:
                        for block in iter(lambda: fh.read(1 << 20), b""):
                            digest.update(block)
                except OSError:
                    continue
                if digest.hexdigest() == prior.get("sha256"):
                    seg.sha256 = str(prior.get("sha256") or "")
                    seg.complete = True
                    resumed_any = True
                else:
                    _log.info("swarm segment %d hash mismatch — "
                              "re-fetching", seg.index)
        report.resumed = resumed_any

        state: dict[str, Any] = {
            "url": url, "total": total,
            "etag": validators["etag"],
            "last_modified": validators["last_modified"],
            "segments": [s.as_dict() for s in segments],
        }
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        self._save_state(dest_p, state)

        received = sum(s.size for s in segments if s.complete)
        done_count = sum(1 for s in segments if s.complete)
        lock = threading.Lock()
        attempts = [0]
        noted: set[int] = set()

        def _emit_progress() -> None:
            with lock:
                got = received
            _call_progress(progress, got, total)
            _call_tracker(tracker, "report_bytes", got, total)

        def _emit_segments() -> None:
            with lock:
                done = done_count
            _call_tracker(tracker, "report_segments", done, len(segments))

        def _note_completed(seg: _Segment) -> None:
            # Counter + durable state move together under one lock:
            # a crash between them must never record a segment the
            # state file does not know about (or vice versa).
            # Idempotent: a segment the loop already noted (or one
            # swept up after an early break) is never counted twice.
            nonlocal received, done_count
            with lock:
                if seg.index in noted:
                    return
                noted.add(seg.index)
                received += seg.size
                done_count += 1
                state["segments"][seg.index] = seg.as_dict()
                self._save_state(dest_p, state)
            _emit_progress()
            _emit_segments()

        _emit_progress()
        _emit_segments()

        if _stopped():
            # Cancelled before the first segment started — nothing
            # to resume, so the attempt leaves nothing behind.
            self._discard_state(dest_p, len(ranges))
            report.cancelled = True
            report.error = "swarm cancelled"
            return report

        own_executor = executor is None
        pool = executor or ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="swarm")
        refused: str | None = None
        failed: Exception | None = None
        cancelled = False
        futures: dict = {}
        try:
            todo = [s for s in segments if not s.complete]

            def _run(seg: _Segment) -> _Segment:
                if _stopped():
                    # The stop landed before this segment started:
                    # never fetch it.  CancelledError is the loop's
                    # existing "skip this segment" signal — and the
                    # check lives here, in the worker, because the
                    # executor promotes a queued segment the moment
                    # a worker frees up, before the controlling
                    # loop can cancel it.
                    raise CancelledError("swarm cancelled")
                digest, made = self._fetch_segment(
                    url, dest_p, seg, segment_retries,
                    should_stop=_stopped)
                attempts[0] += made
                seg.sha256 = digest
                seg.complete = True
                return seg

            futures = {pool.submit(_run, seg): seg for seg in todo}
            for fut in as_completed(futures):
                if _stopped():
                    # Cooperative stop: the next segment boundary
                    # is here — in-flight segments finish in the
                    # finally below, none start after.
                    cancelled = True
                    break
                try:
                    seg = fut.result()
                except _RangeRefused as exc:
                    refused = str(exc)
                    break
                except CancelledError:
                    continue
                except Exception as exc:  # noqa: BLE001 — one segment
                    failed = exc  # failing breaks the swarm; the
                    break  # partial state stays for resume
                _note_completed(seg)
        finally:
            # Let every in-flight segment write land before any
            # cleanup deletes part files.  On a cooperative cancel
            # a segment that has not started yet is cancelled here
            # (it never starts); one already in flight runs to its
            # boundary and its landing is recorded — the whole
            # attempt is discarded below either way, so the
            # recording is only bookkeeping.  Without a cancel
            # nothing is cancelled: a segment that has not started
            # yet still lands bytes the partial state wants for
            # resume, and a refused range fails fast on its own.  A
            # caller-owned executor is never shut down.  Segments
            # that finished after the loop broke early on a
            # refusal/failure are swept up here, so the partial
            # state records everything that actually landed — the
            # wait is the point: a done() peek here would run
            # before the in-flight segments finish and record
            # nothing.
            for fut in futures:
                if fut.cancelled():
                    continue
                if cancelled and fut.cancel():
                    # Never started, and now never will: the
                    # boundary for this segment is "not started".
                    continue
                try:
                    seg = fut.result()
                except Exception:  # noqa: BLE001 — already recorded
                    pass
                else:
                    _note_completed(seg)
            if own_executor:
                pool.shutdown(wait=True)
        report.segment_attempts = attempts[0]
        report.segments_done = sum(1 for s in segments if s.complete)
        if cancelled:
            # The part files are resume bait no longer — the job
            # was cancelled, so the attempt leaves nothing behind.
            self._discard_state(dest_p, len(ranges))
            report.cancelled = True
            report.error = "swarm cancelled"
            return report
        if refused is not None:
            # Ranges are not honoured at all — the part files are
            # useless, so the attempt leaves nothing behind.
            self._discard_state(dest_p, len(ranges))
            report.refused_reason = refused
            return report
        if failed is not None:
            # The partial state stays on disk: the next run resumes
            # from the completed segments instead of starting over.
            report.error = f"segment download failed: {failed}"
            return report

        # ── reassembly: part files in order, one streaming hash ──
        if _stopped():
            # Cancelled while the segments were landing: the file
            # must not land after the owner stopped it.
            self._discard_state(dest_p, len(ranges))
            report.cancelled = True
            report.error = "swarm cancelled"
            return report
        digest = hashlib.sha256()
        try:
            with dest_p.open("wb") as out:
                for seg in segments:
                    part = self._segment_path(dest_p, seg.index)
                    with part.open("rb") as fh:
                        for block in iter(lambda: fh.read(1 << 20), b""):
                            digest.update(block)
                            out.write(block)
        except OSError as exc:
            report.error = f"reassembly failed: {exc}"
            return report
        try:
            landed = dest_p.stat().st_size
        except OSError as exc:
            report.error = f"reassembly failed: {exc}"
            return report
        if landed != total:
            report.error = (f"reassembled file is {landed} bytes, "
                            f"expected {total}")
            return report
        self._discard_state(dest_p, len(ranges))
        report.ok = True
        report.path = str(dest_p)
        report.size_bytes = total
        report.sha256 = digest.hexdigest()
        _emit_progress()
        _emit_segments()
        return report


# ── durable queue ────────────────────────────────────────────────────

class SwarmQueue:
    """The swarm's durable job queue.

    One JSON file (atomic tmp + replace on every mutation), so a
    queued job survives a restart and a crashed mid-download job
    keeps its segment state beside the file for resume.  A corrupt
    queue file is quarantined beside itself and the queue starts
    empty — losing the queue never loses a download, only the
    record of it.
    """

    def __init__(self, path: str | Path | None = None,
                 *, context: Any = None) -> None:
        self.path = Path(path).expanduser() if path is not None \
            else default_queue_path(context)
        # One lock for every instance in this process (see
        # _QUEUE_LOCK): the pool and the command layer open the same
        # file through different instances.
        self._lock = _QUEUE_LOCK
        self._jobs: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Re-read the queue file into memory.

        Called by every public method before it touches ``_jobs``:
        the pool and the command layer open the same file through
        different instances, so a mutation by one (a cancel, a
        mark) must be visible to the other without a restart.  A
        missing file keeps the in-memory jobs; a corrupt file is
        quarantined beside itself and the queue starts empty."""
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._jobs = [j for j in data if isinstance(j, dict)]
                return
            raise ValueError("queue file is not a JSON list")
        except Exception as e:  # noqa: BLE001 — quarantine, start empty
            record_failure("SwarmQueue.load", e)
            try:
                bad = self.path.with_suffix(
                    f".corrupt-{int(time.time())}")
                self.path.replace(bad)
            except OSError:
                pass
            self._jobs = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._jobs), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            record_failure("SwarmQueue.save", e)

    def enqueue(self, url: str, dest: str | Path, *,
                audio_only: bool = False,
                workers: int | None = None,
                min_swarm_bytes: int | None = None) -> str:
        """Queue a swarm job.  The same url + dest already queued
        (or running) returns its existing id — no duplicates."""
        url = str(url or "")
        dest_s = str(dest)
        now = time.time()
        with self._lock:
            self._load()
            for job in self._jobs:
                if job.get("url") == url and job.get("dest") == dest_s \
                        and job.get("status") in ("queued", "running"):
                    return str(job["id"])
            job: dict[str, Any] = {
                "id": new_short_id("sw"), "url": url, "dest": dest_s,
                "audio_only": bool(audio_only),
                "workers": workers, "min_swarm_bytes": min_swarm_bytes,
                "status": "queued", "segments_done": 0,
                "segments_total": 0, "size_bytes": 0, "sha256": "",
                "error": "", "created_ts": now, "updated_ts": now,
            }
            self._jobs.append(job)
            self._save()
            return str(job["id"])

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._load()
            for job in self._jobs:
                if job.get("id") == job_id:
                    return dict(job)
        return None

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self._load()
            jobs = [dict(j) for j in self._jobs]
        if status:
            jobs = [j for j in jobs if j.get("status") == status]
        return jobs

    def mark(self, job_id: str, status: str, **fields: Any) -> bool:
        with self._lock:
            self._load()
            for job in self._jobs:
                if job.get("id") == job_id:
                    # Cancelled is terminal: a job the owner stopped
                    # never moves back to running/done/failed — a
                    # mark that lands after a cancel is a no-op, so
                    # a cancelled job is never drained or reported
                    # as finished by a stale caller.
                    if job.get("status") == "cancelled" \
                            and status != "cancelled":
                        return False
                    job["status"] = status
                    job["updated_ts"] = time.time()
                    for key, value in fields.items():
                        job[key] = value
                    self._save()
                    return True
        return False

    def cancel(self, job_id: str) -> str:
        """Cancel a queued or running job.

        Queued → ``cancelled`` (the pump never runs it); running →
        ``cancelled`` (the in-flight download stops at its next
        segment boundary — see ``SwarmPool``).  Durable: the new
        status is written atomically, so a fresh queue instance
        sees it.  Returns ``"cancelled"``, ``"not-found"``,
        ``"already-cancelled"`` or ``"finished"`` (done/failed) —
        the caller renders the outcome in words.
        """
        with self._lock:
            self._load()
            for job in self._jobs:
                if job.get("id") == job_id:
                    status = str(job.get("status") or "")
                    if status == "cancelled":
                        return "already-cancelled"
                    if status in ("done", "failed"):
                        return "finished"
                    job["status"] = "cancelled"
                    job["updated_ts"] = time.time()
                    self._save()
                    return "cancelled"
        return "not-found"

    def status_text(self) -> str:
        """One plain-text rendering of the queue for /download swarm."""
        jobs = self.list_jobs()
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job.get("status", "?")] = \
                counts.get(job.get("status", "?"), 0) + 1
        lines = []
        summary = (f"download swarm — {len(jobs)} job(s): "
                   f"{counts.get('queued', 0)} queued, "
                   f"{counts.get('running', 0)} running, "
                   f"{counts.get('done', 0)} done, "
                   f"{counts.get('failed', 0)} failed")
        if counts.get("cancelled"):
            summary += f", {counts['cancelled']} cancelled"
        lines.append(summary)
        if not jobs:
            lines.append("nothing queued — /download <url> --swarm "
                         "queues a large file here")
            return "\n".join(lines)
        for job in jobs:
            seg = ""
            if job.get("segments_total"):
                seg = (f" [{job.get('segments_done', 0)}/"
                       f"{job.get('segments_total')} segments]")
            err = f" — {job['error']}" if job.get("error") else ""
            lines.append(f"• {job['id']} {job.get('status')} "
                         f"{job.get('url', '')}{seg}{err}")
        return "\n".join(lines)


# ── worker pool ──────────────────────────────────────────────────────

class SwarmPool:
    """The worker pool that drains the durable queue.

    One shared ``ThreadPoolExecutor`` sized by the profile budget
    (``max_concurrent_downloads``): a queued job's segments fan out
    over it.  ``process_one`` drains a single job synchronously
    (the unit tests drive); :meth:`start` runs a daemon pump that
    drains continuously, re-queuing jobs orphaned by a crash first
    so their segment state resumes instead of restarting.
    """

    def __init__(self, queue: SwarmQueue, *,
                 downloader: SwarmDownloader | None = None,
                 workers: int | None = None,
                 context: Any = None) -> None:
        self.queue = queue
        self._downloader = downloader or SwarmDownloader()
        self._workers = resolve_workers(context, workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="swarm")
        self._stop = threading.Event()
        self._pump: threading.Thread | None = None

    def resume_stale(self, older_than_s: float = STALE_RUNNING_S
                     ) -> int:
        """Re-queue ``running`` jobs untouched for longer than
        *older_than_s* — their process died mid-download; their
        segment state is still on disk, so the next run resumes
        instead of restarting.  Returns how many were re-queued."""
        now = time.time()
        revived = 0
        for job in self.queue.list_jobs("running"):
            if now - float(job.get("updated_ts") or 0) > older_than_s \
                    and self.queue.mark(str(job["id"]), "queued"):
                revived += 1
        return revived

    def process_one(self, *, min_swarm_bytes: int | None = None
                    ) -> dict[str, Any] | None:
        """Drain the oldest queued job to done/failed.  None when
        the queue is empty.  The job's own worker count wins over
        the pool's when it names one.  A job cancelled between the
        listing and the mark is skipped, not drained — ``mark``
        refuses to move a cancelled job to running, so the race
        closes here."""
        for job in self.queue.list_jobs("queued"):
            job_id = str(job["id"])
            if not self.queue.mark(job_id, "running"):
                continue
            break
        else:
            return None
        report = self._downloader.download(
            str(job["url"]), Path(str(job["dest"])),
            workers=job.get("workers") or self._workers,
            min_swarm_bytes=(job.get("min_swarm_bytes")
                             if job.get("min_swarm_bytes") is not None
                             else (min_swarm_bytes
                                   if min_swarm_bytes is not None
                                   else DEFAULT_MIN_SWARM_BYTES)),
            executor=self._executor,
            should_stop=lambda: self._is_cancelled(job_id))
        if report.cancelled:
            # The downloader already discarded the part files; the
            # job just needs its terminal status.
            self.queue.mark(job_id, "cancelled", error="")
        elif report.ok:
            self.queue.mark(
                job_id, "done", segments_done=report.segments_done,
                segments_total=report.segments_total,
                size_bytes=report.size_bytes, sha256=report.sha256,
                error="")
        elif report.refused_reason:
            self.queue.mark(job_id, "failed",
                            error=f"swarm refused: "
                                  f"{report.refused_reason}")
        else:
            self.queue.mark(job_id, "failed",
                            segments_done=report.segments_done,
                            segments_total=report.segments_total,
                            error=report.error or "swarm failed")
        return self.queue.get(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        """True when the job has been cancelled — read fresh from
        the durable queue, so a cancel issued through another
        queue instance (the command layer) is seen here."""
        job = self.queue.get(job_id)
        return bool(job) and job.get("status") == "cancelled"

    def start(self) -> None:
        """Start the daemon pump.  A second start is a no-op."""
        if self._pump is not None and self._pump.is_alive():
            return
        self.resume_stale()

        def _pump() -> None:
            while not self._stop.is_set():
                try:
                    if self.process_one() is None:
                        self._stop.wait(1.0)
                except Exception as e:  # noqa: BLE001 — the pump
                    record_failure("SwarmPool.pump", e)  # never dies
                    self._stop.wait(1.0)

        self._stop.clear()
        self._pump = threading.Thread(
            target=_pump, name="swarm-pump", daemon=True)
        self._pump.start()

    def stop(self) -> None:
        """Stop the pump and release the pool.  Idempotent."""
        self._stop.set()
        if self._pump is not None:
            self._pump.join(timeout=5.0)
            self._pump = None
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:  # noqa: BLE001 — shutdown is best-effort
            record_failure("SwarmPool.stop", e)

    def close(self) -> None:
        self.stop()
