"""Solved-session store: cookies + storage saved when a wall clears.

When Devon (or the owner, via takeover) clears a challenge, the tab's
cookies + storage are worth more than the single page load: the
clearance (e.g. a ``cf_clearance`` cookie) proves the session passed
the wall. This store persists that state per-site, so the next visit
to the same site reuses the solved session instead of facing the same
wall again.

Rules:
- save ONLY on a verified clear — still-blocked challenges save
  nothing, ever;
- one JSON file per site (``<data_dir>/solved_sessions/<site>.json``),
  written atomically, so a crash never leaves a half-written jar;
- expired cookies are dropped on load — a stale clearance is not
  restored and cannot mask a fresh wall;
- sites never leak into each other: a load is keyed by the exact
  registrable site of the URL being visited;
- cookie VALUES live only in these files, never in logs or audit
  entries — callers audit the save event (site + count + via) without
  values.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .challenge_memory import _site_of

_log = get_logger(__name__)

__all__ = [
    "SolvedSessionStore",
    "restore_solved_session",
    "save_solved_session",
]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.\-]+")


def _clean_cookie(raw: Any) -> dict[str, Any] | None:
    """Normalize one cookie dict. ``None`` = unusable, skip it."""
    if not isinstance(raw, dict):
        return None
    name, value, domain = (str(raw.get("name") or ""),
                           str(raw.get("value") or ""),
                           str(raw.get("domain") or ""))
    if not name or not domain:
        return None
    out: dict[str, Any] = {
        "name": name,
        "value": value,
        "domain": domain,
        "path": str(raw.get("path") or "/") or "/",
    }
    expires = raw.get("expires")
    if isinstance(expires, (int, float)) and expires > 0:
        out["expires"] = float(expires)
    if raw.get("secure"):
        out["secure"] = True
    http_only = raw.get("httpOnly", raw.get("http_only"))
    if http_only:
        out["http_only"] = True
    same_site = raw.get("sameSite", raw.get("same_site"))
    if same_site in ("Strict", "Lax", "None"):
        out["same_site"] = same_site
    return out


def _to_playwright_cookies(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stored shape -> Playwright ``add_cookies`` shape."""
    out: list[dict[str, Any]] = []
    for cookie in cookies:
        cleaned = _clean_cookie(cookie)
        if cleaned is None:
            continue
        entry: dict[str, Any] = {
            "name": cleaned["name"],
            "value": cleaned["value"],
            "domain": cleaned["domain"],
            "path": cleaned["path"],
        }
        if "expires" in cleaned:
            entry["expires"] = cleaned["expires"]
        if cleaned.get("secure"):
            entry["secure"] = True
        if cleaned.get("http_only"):
            entry["httpOnly"] = True
        if cleaned.get("same_site"):
            entry["sameSite"] = cleaned["same_site"]
        out.append(entry)
    return out


class SolvedSessionStore:
    """Per-site solved-session files under ``<data_dir>/solved_sessions``."""

    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir).expanduser() / "solved_sessions"
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- paths ----------------------------------------------------------------
    def _path_for(self, site_or_url: str) -> Path:
        safe = _SAFE_NAME.sub("_", _site_of(site_or_url))
        return self._dir / f"{safe}.json"

    # -- writes -----------------------------------------------------------------
    def save(self, site_or_url: str, url: str, cookies: list[dict[str, Any]],
             *, origins: list[dict[str, Any]] | None = None,
             cleared_via: str = "") -> dict[str, Any]:
        """Persist one site's solved state. Returns the stored record."""
        site = _site_of(site_or_url)
        record = {
            "site": site,
            "url": url or "",
            "saved_at": time.time(),
            "cleared_via": cleared_via or "",
            "cookies": [c for c in (_clean_cookie(c) for c in (cookies or []))
                        if c is not None],
            "origins": list(origins or []),
        }
        path = self._path_for(site)
        tmp = path.with_suffix(f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(record, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, path)
        return record

    # -- reads ------------------------------------------------------------------
    def load(self, site_or_url: str) -> dict[str, Any] | None:
        """Saved state for a site, expired cookies dropped.

        ``None`` = nothing saved (or nothing still valid). Expired
        cookies never come back: a stale clearance cannot mask a
        fresh wall.
        """
        path = self._path_for(site_or_url)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            _log.debug("solved session %s unreadable", path, exc_info=True)
            return None
        if not isinstance(record, dict):
            return None
        now = time.time()
        fresh = [c for c in (record.get("cookies") or [])
                 if isinstance(c, dict)
                 and not ("expires" in c
                          and isinstance(c["expires"], (int, float))
                          and c["expires"] <= now)]
        if not fresh:
            return None
        record["cookies"] = fresh
        return record

    def has(self, site_or_url: str) -> bool:
        return self.load(site_or_url) is not None

    def clear(self, site_or_url: str) -> bool:
        """Delete one site's saved state. True when a file was removed."""
        path = self._path_for(site_or_url)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:
            _log.debug("solved session %s clear failed", path, exc_info=True)
            return False

    def sites(self) -> list[str]:
        """Sites with a saved state file (validity not checked)."""
        return sorted(p.stem for p in self._dir.glob("*.json"))


def save_solved_session(tab: Any, url: str, store: SolvedSessionStore,
                        *, cleared_via: str = "") -> int:
    """Persist a tab's cookies + storage. Returns cookies saved (0 = none).

    Best effort by design: a save failure must never break the resume
    that just cleared the wall. Prefers the tab's full
    ``storage_state()`` (cookies + origins) and falls back to
    ``cookies()``. Returns 0 when the tab exposes neither, or when
    there is nothing worth saving.
    """
    state: dict[str, Any] | None = None
    getter = getattr(tab, "storage_state", None)
    if callable(getter):
        try:
            candidate = getter()
            if isinstance(candidate, dict):
                state = candidate
        except Exception:  # noqa: BLE001 - save is best effort
            state = None
    cookies: list[dict[str, Any]] = []
    origins: list[dict[str, Any]] = []
    if state is not None:
        cookies = state.get("cookies") or []
        origins = state.get("origins") or []
    if not cookies:
        getter = getattr(tab, "cookies", None)
        if callable(getter):
            try:
                cookies = getter() or []
            except Exception:  # noqa: BLE001 - save is best effort
                cookies = []
    if not cookies:
        return 0
    try:
        record = store.save(url, url, cookies, origins=origins,
                            cleared_via=cleared_via)
    except Exception:  # noqa: BLE001 - save is best effort
        _log.debug("save_solved_session failed", exc_info=True)
        return 0
    return len(record.get("cookies") or [])


def restore_solved_session(tab: Any, url: str,
                           store: SolvedSessionStore) -> int:
    """Apply a site's saved cookies to a tab. Returns cookies applied.

    0 = nothing saved, nothing valid, or the tab cannot take cookies.
    Only the visiting site's cookies are applied — never another
    site's. Failures are silent (0): a restore that cannot apply is
    not a clear, and the caller falls through to the normal ladder.
    """
    try:
        saved = store.load(url)
    except Exception:  # noqa: BLE001 - restore is best effort
        return 0
    if not saved:
        return 0
    cookies = _to_playwright_cookies(saved.get("cookies") or [])
    if not cookies:
        return 0
    adder = getattr(tab, "add_cookies", None)
    if not callable(adder):
        return 0
    try:
        adder(cookies)
    except Exception:  # noqa: BLE001 - restore is best effort
        _log.debug("restore_solved_session apply failed", exc_info=True)
        return 0
    return len(cookies)
