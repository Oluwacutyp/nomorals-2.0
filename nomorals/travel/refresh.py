"""Monthly refresh for the airport→country mapping (R3-67 remainder).

``travel/airports.py`` loads ``data/airports_iata_country.csv`` (seed
shipped 1812774a: 86,371 dump rows → 9,053 distinct IATA codes → 239
countries) but nothing ever refreshed it — the table silently went
stale forever, and the only refresh was a human re-running a
generator script that lived outside the repo. This module ports that
generator's rules in-repo and runs them on a monthly durable job.

Guarantees, in order:

* the old table is never touched until the new one is fetched, built,
  and validated — a failed fetch or a bad dump leaves the old file
  byte-identical;
* the replacement is atomic (temp file in the same dir + ``os.replace``)
  and the running process sees it at once (the ``airports`` module
  cache is invalidated right after the replace);
* the tick never raises — any failure is returned as ``ok: False``
  with the reason named, written to durable state, and published to
  the owner (a stale table is the one thing worth paging about; a
  routine monthly success stays silent).

The fetch goes through the repo's normal HTTP path
(``core.http.HttpClient``) against the same fetched copy the seed
used — no key, no new client. The CSV itself is data, not code
(gitignored by the bare ``data/`` rule); this lane ships the job +
generator, the table refreshes on EC2.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .airports import AIRPORTS_DATA_PATH

_log = get_logger(__name__)

__all__ = [
    "AIRPORTS_REFRESH_JOB",
    "AIRPORTS_REFRESH_SPEC",
    "AIRPORTS_REFRESH_TOOL",
    "AIRPORTS_SOURCE_URL",
    "DATA_PATH",
    "build_mapping",
    "ensure_airport_refresh_job",
    "next_refresh_run",
    "read_refresh_state",
    "refresh_airports",
    "render_mapping",
    "run_airport_refresh_tick_once",
    "validate_mapping",
]

#: The fetched copy of the canonical dump — the same file OurAirports
#: publishes (https://ourairports.com/data/airports.csv), mirrored on
#: GitHub. Public domain, no key.
AIRPORTS_SOURCE_URL = (
    "https://raw.githubusercontent.com/davidmegginson/"
    "ourairports-data/main/airports.csv"
)

#: The table this job refreshes. Tests point it at a temp file.
DATA_PATH = AIRPORTS_DATA_PATH

AIRPORTS_REFRESH_JOB = "airport dataset refresh"
AIRPORTS_REFRESH_TOOL = "__airport_refresh_tick__"
#: Monthly, 1st 04:00 local. The 03:00 slot on the 1st is owned by the
#: backup restore drill (backup/coordinator.py MONTHLY_DRILL_CRON),
#: so the refresh takes the next hour rather than colliding with it.
AIRPORTS_REFRESH_SPEC = "cron 0 4 1 * *"

_STATE_NAME = "airports_refresh.json"


def next_refresh_run(now: float | None = None) -> str:
    """Next monthly run, from AIRPORTS_REFRESH_SPEC: 04:00 on the 1st.

    The spec is monthly by construction ("cron 0 4 1 * *"), so the
    next run is the 1st of this month at 04:00 local when that is
    still ahead, else the 1st of next month. Returned as
    "YYYY-MM-DD HH:MM" local time, the same shape the status lines
    use for the last run.
    """
    current = time.localtime(now if now is not None else time.time())
    year, month = current.tm_year, current.tm_mon
    if current.tm_mday > 1 or (current.tm_mday == 1
                               and (current.tm_hour, current.tm_min) >= (4, 0)):
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return time.strftime("%Y-%m-%d %H:%M",
                         (year, month, 1, 4, 0, 0, 0, 0, -1))

#: Validation floors, set well below the seed table (9,053 rows / 239
#: countries) and far above any degenerate dump: a dump that lost
#: most of its rows, or one that is mostly junk, must not replace a
#: good table.
_MIN_ROWS = 5000
_MIN_COUNTRIES = 100
_MAX_ROWS = 50000
_MAX_DROPPED_FRACTION = 0.05


def build_mapping(text: str) -> tuple[dict[str, str], dict[str, int]]:
    """Build the IATA→country mapping from a dump's CSV text.

    The generator's rules, ported verbatim: a row with no IATA code
    is skipped; a bad code (not 3 letters) or a bad country (not 2
    letters) is dropped; a code seen before with a DIFFERENT country
    is dropped (first wins); a code seen before with the SAME country
    collapses silently. Returns ``(mapping, stats)``.
    """
    mapping: dict[str, str] = {}
    total = 0
    dropped = 0
    try:
        rows = csv.DictReader(io.StringIO(text))
        for row in rows:
            total += 1
            code = (row.get("iata_code") or "").strip().upper()
            country = (row.get("iso_country") or "").strip().upper()
            if not code:
                continue
            if len(code) != 3 or not code.isalpha():
                dropped += 1
                continue
            if len(country) != 2 or not country.isalpha():
                dropped += 1
                continue
            if code in mapping and mapping[code] != country:
                dropped += 1
                continue
            mapping[code] = country
    except Exception:  # noqa: BLE001 - an unreadable dump builds nothing
        _log.debug("airport dump unreadable", exc_info=True)
        return {}, {"total_rows": total, "dropped": dropped,
                    "codes": 0, "countries": 0}
    return mapping, {"total_rows": total, "dropped": dropped,
                     "codes": len(mapping),
                     "countries": len(set(mapping.values()))}


def render_mapping(mapping: dict[str, str]) -> str:
    """The mapping as CSV text: header + one row per code, sorted."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["iata_code", "iso_country"])
    for code in sorted(mapping):
        w.writerow([code, mapping[code]])
    return buf.getvalue()


def validate_mapping(mapping: dict[str, str],
                     stats: dict[str, int]) -> str:
    """Sanity-check a built mapping. "" means valid, else the reason."""
    total = stats.get("total_rows", 0)
    dropped = stats.get("dropped", 0)
    codes = stats.get("codes", len(mapping))
    countries = stats.get("countries", len(set(mapping.values())))
    if total == 0:
        return "dump has no rows"
    if codes < _MIN_ROWS:
        return f"too few IATA codes: {codes} < {_MIN_ROWS}"
    if codes > _MAX_ROWS:
        return f"too many IATA codes: {codes} > {_MAX_ROWS}"
    if countries < _MIN_COUNTRIES:
        return f"too few countries: {countries} < {_MIN_COUNTRIES}"
    if dropped / total > _MAX_DROPPED_FRACTION:
        return (f"dropped {dropped} of {total} rows "
                f"({dropped / total:.1%} > {_MAX_DROPPED_FRACTION:.0%})")
    for code, country in mapping.items():
        if len(code) != 3 or not code.isalpha():
            return f"bad IATA code in mapping: {code!r}"
        if len(country) != 2 or not country.isalpha():
            return f"bad country in mapping: {country!r}"
    return ""


def _fetch_dump() -> str:
    """Fetch the canonical dump through the repo's HTTP client."""
    client = HttpClient(timeout=120.0)
    response = client.get(AIRPORTS_SOURCE_URL)
    response.raise_for_status()
    return response.text


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


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


def read_refresh_state(context: Any) -> dict[str, Any] | None:
    """The last refresh's durable state, or None when never run."""
    path = _state_path(context)
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no state is a fact, not an error
        return None
    return raw if isinstance(raw, dict) else None


def refresh_airports(
        *,
        fetcher: Any = None,
        data_path: str | Path | None = None,
        state_path: str | Path | None = None,
        now: float | None = None) -> dict[str, Any]:
    """Fetch, build, validate, atomically replace the airport table.

    Never raises. A failed fetch or a failed validation returns
    ``ok: False`` with the reason — and the old file is byte-identical
    afterwards, because nothing is written until validation passes.
    On success the ``airports`` module cache is invalidated so the
    running process serves the new table at once.
    """
    from .airports import invalidate_cache

    now_f = float(now if now is not None else time.time())
    target = Path(data_path) if data_path is not None else DATA_PATH
    state: dict[str, Any] = {"ran_at": now_f, "ok": False,
                             "rows": 0, "countries": 0, "dropped": 0,
                             "source": AIRPORTS_SOURCE_URL,
                             "source_sha256": "", "error": ""}
    try:
        fetch = fetcher if fetcher is not None else _fetch_dump
        text = fetch()
    except Exception as exc:  # noqa: BLE001 - fetch failure is a result
        state["error"] = f"fetch failed: {exc}"
        if state_path is not None:
            with contextlib.suppress(Exception):
                _write_state(Path(state_path), state)
        return state
    mapping, stats = build_mapping(text)
    reason = validate_mapping(mapping, stats)
    if reason:
        state["error"] = f"validation failed: {reason}"
        state["dropped"] = stats.get("dropped", 0)
        if state_path is not None:
            with contextlib.suppress(Exception):
                _write_state(Path(state_path), state)
        return state
    rendered = render_mapping(mapping)
    source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(rendered, encoding="utf-8")
        os.replace(tmp, target)
    except Exception as exc:  # noqa: BLE001 - a failed replace is a result
        state["error"] = f"replace failed: {exc}"
        if state_path is not None:
            with contextlib.suppress(Exception):
                _write_state(Path(state_path), state)
        return state
    invalidate_cache()
    state.update({"ok": True, "rows": stats["codes"],
                  "countries": stats["countries"],
                  "dropped": stats["dropped"],
                  "source_sha256": source_sha})
    if state_path is not None:
        with contextlib.suppress(Exception):
            _write_state(Path(state_path), state)
    return state


def run_airport_refresh_tick_once(
        *,
        context: Any = None,
        notifier: Any = None,
        fetcher: Any = None,
        data_path: str | Path | None = None,
        state_path: str | Path | None = None,
        now: float | None = None) -> dict[str, Any]:
    """One airport-refresh tick. Never raises.

    A routine monthly success is silent. A failure is published
    through the Notifier — a table going stale is the one thing worth
    paging about — and recorded in durable state either way.
    """
    try:
        mark = (Path(state_path) if state_path is not None
                else _state_path(context))
        out = refresh_airports(fetcher=fetcher, data_path=data_path,
                               state_path=mark, now=now)
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        out = {"ran_at": float(now if now is not None else time.time()),
               "ok": False, "rows": 0, "countries": 0, "dropped": 0,
               "source": AIRPORTS_SOURCE_URL, "source_sha256": "",
               "error": f"tick failed: {exc}"}
    if not out.get("ok") and notifier is not None:
        with contextlib.suppress(Exception):
            notifier.publish(
                "airports_refresh",
                title="airport dataset refresh failed",
                body=(f"airport table refresh failed: "
                      f"{out.get('error') or 'unknown error'}\n"
                      f"the old table is still in place; the next "
                      f"monthly run retries."))
    return out


def ensure_airport_refresh_job(scheduler: Any, *,
                               spec: str = AIRPORTS_REFRESH_SPEC
                               ) -> Any | None:
    """Register the monthly airport-refresh job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job that
    already matches is returned untouched. Best-effort — a scheduler
    that cannot store the job returns None rather than raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"),
                                          job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != AIRPORTS_REFRESH_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            AIRPORTS_REFRESH_JOB, spec, "tool",
            {"tool": AIRPORTS_REFRESH_TOOL, "args": {}}, routine=True)
    except Exception:
        return None
