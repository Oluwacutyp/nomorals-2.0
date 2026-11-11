"""The ONE cookie estate (Navigator A6, webdrive census §4 A6).

Before this module Devon kept the same logins in several places
that could not see each other:

* the webnav persistent profile — Chromium's own ``Cookies`` DB
  inside the profile dir, plus the owner-bundle sidecar
  ``imported_cookies.json`` (``social/chat/cookie_bundle.py``);
* the browser-service store — HTTP jars
  (``data/browser/sessions/<name>.json`` and
  ``<browser-service>/cookies/<session>/<session>.json``) and
  rendered tabs' ``playwright-storage.json`` per session
  (``browser/service.py``);
* solved-session files — per-site JSON under
  ``<browser-service>/takeover/solved_sessions/``
  (``browser/solved_sessions.py``).

A login earned in one estate did not count in the others, and a
clear in one left the others holding the cookie.  This module is
the single estate every consumer reads and writes: two
runtime-created tables in the home database (the Batch-B pattern
— no migration-file edits, the file is frozen), keyed by
(domain, name, path), newest write winning.

Rules, the same as the bundle organ's:

* cookie VALUES are credentials — they live in these tables and
  nowhere else.  Logs, journals, replies and recipes carry
  domains and counts only, never a value.  The one exception is
  the Netscape export (:meth:`CookieEstate.export_netscape`),
  which exists because yt-dlp only reads files; it is a
  derivation regenerated from the estate, never a source, and
  it is written 0600.
* an expired cookie is dropped at the door (write and read);
* a record the estate already holds newer wins over an older
  write — "newest wins" is by ``updated_at``;
* the legacy files are imported ONCE, per source and root,
  tracked in ``browser_cookie_meta``.  After the import the old
  stores are never written again — the writers were rewired to
  this estate, and the old files survive only as read-only
  history until the owner deletes them.
* ``source`` is the LAST writer of a record (for the audit
  trail and for scoped clears).  ``imported`` is stickier: a
  key that ever arrived as an owner bundle stays flagged
  imported even when a later harvest refreshes it — so
  /cookies clear removes what the owner handed over (and a
  refreshed copy of it), while /cookies clear all still
  leaves the sessions Devon earned herself alone, exactly
  as the old sidecar clear did.

Layer: storage (L2) — imports core (L1) only.  Every browser
consumer (browser, social/chat, media, tools, accounts) sits
above it and reaches the estate through here.
"""

from __future__ import annotations

from contextlib import suppress
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from ..core.config import load_settings
from ..core.logging_setup import get_logger
from .db import Database, open_database

_log = get_logger(__name__)

__all__ = [
    "CookieEstate",
    "CookieEstateError",
    "COOKIE_ESTATE_DDL",
    "SOURCE_IMPORTED",
    "SOURCE_LOGIN",
    "SOURCE_SERVICE",
    "SOURCE_SOLVED",
    "SOURCE_WEBNAV",
    "default_legacy_roots",
    "home_estate",
    "migrate_legacy",
    "normalize_record",
    "playwright_cookie_payload",
]

#: Sources — the last writer of a record, for the audit trail
#: and for scoped clears (a solved-session clear removes only
#: the cookies a solved save wrote).
SOURCE_IMPORTED = "imported"
SOURCE_WEBNAV = "webnav"
SOURCE_SERVICE = "service"
SOURCE_SOLVED = "solved"
SOURCE_LOGIN = "login"

#: Chromium's epoch: microseconds since 1601-01-01.  The
#: profile's ``Cookies`` DB stores expiry/last-access in it.
_CHROME_EPOCH_DELTA_US = 11_644_473_600_000_000

#: SameSite spellings Devon accepts, normalised to the
#: Playwright spelling.
_SAME_SITE = {"strict": "Strict", "lax": "Lax", "none": "None",
              "no_restriction": "None"}

#: Chromium samesite column: -1 unspecified, 0 no restriction,
#: 1 lax, 2 strict.
_CHROME_SAME_SITE = {-1: "", 0: "None", 1: "Lax", 2: "Strict"}

#: A cookie value shorter than this is never matched by
#: substring in the recipe guard — an exact payload equality
#: still counts, but a 3-letter value inside a URL is not
#: evidence of a leak.
_MIN_SUBSTRING_SECRET_LEN = 8


class CookieEstateError(ValueError):
    """The estate cannot do what was asked (bad record, bad
    root).  The message never carries a cookie value."""


COOKIE_ESTATE_DDL = """
CREATE TABLE IF NOT EXISTS browser_cookies (
  domain TEXT NOT NULL,
  name TEXT NOT NULL,
  path TEXT NOT NULL DEFAULT '/',

  value TEXT NOT NULL DEFAULT '',
  expires REAL NOT NULL DEFAULT -1,
  secure INTEGER NOT NULL DEFAULT 0,
  http_only INTEGER NOT NULL DEFAULT 0,
  same_site TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL DEFAULT 0,
  imported INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (domain, name, path)
);
CREATE INDEX IF NOT EXISTS idx_browser_cookies_domain
  ON browser_cookies(domain);
CREATE TABLE IF NOT EXISTS browser_cookie_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL DEFAULT 0
);
"""


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _same_site(value: Any) -> str:
    return _SAME_SITE.get(str(value or "").strip().lower(), "")


def _chrome_ts_to_unix(micros: Any) -> float:
    """Chromium microseconds-since-1601 → unix seconds.  0 or
    unusable = no timestamp."""
    try:
        raw = int(micros)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if raw <= 0:
        return 0.0
    return (raw - _CHROME_EPOCH_DELTA_US) / 1_000_000.0


def _expires_seconds(value: Any) -> float:
    """Any expiry spelling → unix seconds.  -1 = session cookie
    (no expiry).  Milliseconds collapse to seconds."""
    if value is None:
        return -1.0
    try:
        exp = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1.0
    if exp == 0:
        return -1.0
    if exp > 1e11:      # milliseconds, not seconds
        exp /= 1000.0
    return exp


def normalize_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Any cookie shape Devon stores → the canonical estate
    record, or None when the row is unusable (no name, no
    domain, no value key).

    Shapes accepted: the webnav/Playwright record
    (``httpOnly``/``sameSite``), the solved-session cleaned
    shape (``http_only``/``same_site``), an ``http.cookiejar``
    ``__getstate__`` dump (``domain_initial_dot``/``rest``/
    ``discard``), and a Chromium profile-DB row (``host_key``,
    ``last_access_utc``).  Domains fold to lowercase — the
    (domain, name, path) key is case-insensitive on the domain.
    The returned record never carries its source; the caller
    stamps that at write time."""
    if not isinstance(rec, dict):
        return None
    name = str(rec.get("name") or "")
    domain = str(rec.get("domain") or rec.get("host_key") or "")
    if not name or not domain:
        return None
    if rec.get("domain_initial_dot") and not domain.startswith("."):
        domain = "." + domain
    domain = domain.lower()
    if "value" not in rec:
        return None
    expires = _expires_seconds(rec.get("expires"))
    rest = rec.get("rest") if isinstance(rec.get("rest"), dict) else {}
    http_only = _truthy(rec.get("httpOnly", rec.get("http_only",
                        rest.get("HttpOnly"))))
    same_site = _same_site(rec.get("sameSite")
                           or rec.get("same_site")
                           or rest.get("SameSite"))
    updated_at = 0.0
    if rec.get("updated_at") is not None:
        try:
            updated_at = float(rec["updated_at"])
        except (TypeError, ValueError):
            updated_at = 0.0
    if not updated_at and rec.get("last_access_utc") is not None:
        updated_at = _chrome_ts_to_unix(rec.get("last_access_utc"))
    return {"name": name,
            "value": str(rec.get("value") or ""),
            "domain": domain,
            "path": str(rec.get("path") or "/") or "/",
            "expires": expires,
            "secure": bool(_truthy(rec.get("secure"))),
            "httpOnly": http_only,
            "sameSite": same_site,
            "updated_at": updated_at}


def playwright_cookie_payload(
        records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stored records → ``add_cookies`` arguments.  Session
    cookies (expires <= 0) go out with no expires key at
    all — the driver's word for a session cookie is the
    absence of one."""
    out: list[dict[str, Any]] = []
    for rec in records:
        item: dict[str, Any] = {
            "name": str(rec.get("name") or ""),
            "value": str(rec.get("value") or ""),
            "domain": str(rec.get("domain") or ""),
            "path": str(rec.get("path") or "/"),
            "httpOnly": bool(rec.get("httpOnly")),
            "secure": bool(rec.get("secure")),
        }
        expires = float(rec.get("expires") or 0)
        if expires > 0:
            item["expires"] = expires
        same = str(rec.get("sameSite") or "")
        if same in ("Strict", "Lax", "None"):
            item["sameSite"] = same
        if item["name"] and item["domain"]:
            out.append(item)
    return out


def _cookie_key(rec: dict[str, Any]) -> tuple[str, str, str]:
    return (str(rec.get("domain") or "").lower(),
            str(rec.get("name") or ""),
            str(rec.get("path") or "/"))


def _domain_covers(cookie_domain: str, host: str) -> bool:
    """True when a cookie for *cookie_domain* is sent to *host*:
    the host is the cookie's domain or a subdomain of it —
    ``.google.com`` covers ``mail.google.com``."""
    domain = str(cookie_domain or "").lower().lstrip(".")
    want = str(host or "").strip().lower().lstrip(".")
    return bool(want) and (want == domain
                           or want.endswith("." + domain))


def _domain_within(cookie_domain: str, target: str) -> bool:
    """True when a stored cookie's domain sits inside
    *target*'s tree — the clear direction (the same rule
    ``cookie_bundle`` clears by): a clear for ``google.com``
    takes ``.google.com``, ``google.com`` and
    ``mail.google.com``."""
    domain = str(cookie_domain or "").lower().lstrip(".")
    want = str(target or "").strip().lower().lstrip(".")
    return bool(want) and (domain == want
                           or domain.endswith("." + want))


def _netscape_line(rec: dict[str, Any]) -> str | None:
    """One canonical record → a Netscape cookies.txt line.
    None when the name or value carries a tab/newline the
    format cannot represent (the cookie stays in the estate;
    it simply cannot ride a yt-dlp file)."""
    domain = str(rec.get("domain") or "").strip()
    name = str(rec.get("name") or "").strip()
    value = str(rec.get("value") or "")
    if not domain or not name:
        return None
    if any(ch in name or ch in value for ch in ("\t", "\n", "\r")):
        return None
    tailmatch = "TRUE" if domain.startswith(".") else "FALSE"
    path = str(rec.get("path") or "/")
    secure = "TRUE" if rec.get("secure") else "FALSE"
    try:
        expires = int(rec.get("expires") or 0)
    except (TypeError, ValueError):
        expires = 0
    if expires < 0:
        expires = 0
    return "\t".join([domain, tailmatch, path, secure,
                      str(expires), name, value])


def default_legacy_roots() -> dict[str, Path | None]:
    """Where the pre-unification stores live on this host, by
    the same paths their own modules resolve: the webnav
    profile under the workspace, the browser-service data dir
    under the home, the brain/downloader session jars under
    the home's data tree, solved sessions under the service's
    takeover dir.  Resolved through :func:`load_settings`
    (see :func:`home_estate`) so ``$NM_HOME`` is honored."""
    settings = load_settings()
    home = settings.home_path
    try:
        workspace = settings.workspace_dir
    except Exception:  # noqa: BLE001 - a settings without a
        workspace = None  # workspace still has a home
    profile = None
    if workspace:
        profile = Path(workspace) / "data" / "social" \
            / "browser_profile"
    service = home / "browser-service"
    return {
        "profile_dir": profile,
        "service_data_dir": service,
        "sessions_dir": home / "data" / "browser" / "sessions",
        "solved_dir": service / "takeover" / "solved_sessions",
    }


class CookieEstate:
    """The single cookie store.  Constructed over a
    :class:`Database` (injected — tests open one on a tmp
    file); :func:`home_estate` builds one over the home DB.
    ``legacy_roots`` overrides where the one-time migration
    looks for the old stores (default: the production layout
    from :func:`default_legacy_roots`)."""

    def __init__(self, db: Database, *,
                 legacy_roots: dict[str, Any] | None = None) -> None:
        self.db = db
        self._roots: dict[str, Path | None] = {}
        defaults = default_legacy_roots()
        for key, value in (legacy_roots or {}).items():
            self._roots[key] = (Path(value).expanduser()
                                if value is not None else None)
        for key, value in defaults.items():
            self._roots.setdefault(key, value)
        self._ensured = False

    # -- setup -----------------------------------------------------

    def ensure(self) -> None:
        """Create the tables (if absent), drop expired rows,
        and run the one-time legacy migration for every root
        that has not migrated yet.  Safe to call repeatedly;
        a migration that finds nothing still records its
        flag, so a later call is four cheap SELECTs."""
        self.db.executescript(COOKIE_ESTATE_DDL)
        self._purge_expired()
        self._migrate_pending()
        self._ensured = True

    def _purge_expired(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        cur = self.db.execute(
            "DELETE FROM browser_cookies "
            "WHERE expires > 0 AND expires <= ?",
            (now,))
        return int(cur.rowcount or 0)

    def _migrate_pending(self, *, now: float | None = None) -> dict:
        return migrate_legacy(
            self,
            profile_dir=self._roots.get("profile_dir"),
            service_data_dir=self._roots.get("service_data_dir"),
            sessions_dir=self._roots.get("sessions_dir"),
            solved_dir=self._roots.get("solved_dir"),
            now=now)

    # -- meta ------------------------------------------------------

    def _meta_get(self, key: str) -> str | None:
        try:
            cur = self.db.execute(
                "SELECT value FROM browser_cookie_meta "
                "WHERE key = ?", (str(key),))
            row = cur.fetchone()
        except Exception:  # noqa: BLE001 - a table that does
            return None  # not exist yet reads as unset
        return None if row is None else str(row[0])

    def _meta_set(self, key: str, value: str,
                  *, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        self.db.execute(
            "INSERT INTO browser_cookie_meta "
            "(key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (str(key), str(value), now))

    # -- writes ----------------------------------------------------

    def upsert(self, records: list[dict[str, Any]] | None, *,
               source: str, now: float | None = None
               ) -> dict[str, Any]:
        """Fold records in, newest winning.  Returns
        ``{added, replaced, kept_newer, skipped}`` — kept_newer
        is an older write losing to the record already held,
        skipped is an expired or unusable row.  One
        transaction: a batch lands whole or not at all.  The
        log carries the source, the counts and the domains —
        never a value."""
        now = time.time() if now is None else float(now)
        source = str(source or "")
        added = replaced = kept = skipped = 0
        domains: set[str] = set()
        with self.db.transaction():
            for raw in records or []:
                rec = normalize_record(raw)
                if rec is None:
                    skipped += 1
                    continue
                if 0 < float(rec["expires"]) <= now:
                    skipped += 1
                    continue
                stamp = float(rec.get("updated_at") or 0) or now
                rec["updated_at"] = stamp
                rec["source"] = source
                domains.add(rec["domain"])
                row = self.db.execute(
                    "SELECT updated_at FROM browser_cookies "
                    "WHERE domain = ? AND name = ? AND path = ?",
                    (rec["domain"], rec["name"], rec["path"])
                    ).fetchone()
                if row is None:
                    self.db.execute(
                        "INSERT INTO browser_cookies "
                        "(domain, name, path, value, expires, "
                        " secure, http_only, same_site, source, "
                        " updated_at, imported) VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (rec["domain"], rec["name"], rec["path"],
                         rec["value"], float(rec["expires"]),
                         1 if rec["secure"] else 0,
                         1 if rec["httpOnly"] else 0,
                         rec["sameSite"], source, stamp,
                         1 if source == SOURCE_IMPORTED else 0))
                    added += 1
                elif stamp >= float(row[0]):
                    self.db.execute(
                        "UPDATE browser_cookies SET value = ?, "
                        "expires = ?, secure = ?, http_only = ?, "
                        "same_site = ?, source = ?, "
                        "updated_at = ?, "
                        "imported = MAX(imported, ?) "
                        "WHERE domain = ? "
                        "AND name = ? AND path = ?",
                        (rec["value"], float(rec["expires"]),
                         1 if rec["secure"] else 0,
                         1 if rec["httpOnly"] else 0,
                         rec["sameSite"], source, stamp,
                         1 if source == SOURCE_IMPORTED else 0,
                         rec["domain"], rec["name"], rec["path"]))
                    replaced += 1
                else:
                    kept += 1
        _log.info("cookie estate upsert (%s): %d added, %d replaced, "
                  "%d kept newer, %d skipped across %d domains",
                  source, added, replaced, kept, skipped,
                  len(domains))
        return {"added": added, "replaced": replaced,
                "kept_newer": kept, "skipped": skipped,
                "domains": sorted(domains)}

    # -- reads -----------------------------------------------------

    def _rows(self) -> list[dict[str, Any]]:
        cur = self.db.execute(
            "SELECT domain, name, path, value, expires, secure, "
            "http_only, same_site, source, updated_at, imported "
            "FROM browser_cookies "
            "ORDER BY domain, name, path")
        out: list[dict[str, Any]] = []
        for row in cur.fetchall():
            out.append({
                "domain": str(row[0]), "name": str(row[1]),
                "path": str(row[2]), "value": str(row[3]),
                "expires": float(row[4]),
                "secure": bool(row[5]), "httpOnly": bool(row[6]),
                "sameSite": str(row[7] or ""),
                "source": str(row[8] or ""),
                "updated_at": float(row[9]),
                "imported": bool(row[10]),
            })
        return out

    def cookies_for(self, host: str = "", *,
                    now: float | None = None) -> list[dict[str, Any]]:
        """Live (non-expired) records — all of them, or only
        the ones whose domain covers *host* (the site and its
        subdomains).  Canonical shape, the same records the
        webnav seam hands Playwright."""
        now = time.time() if now is None else float(now)
        host = str(host or "").strip().lower()
        out = []
        for rec in self._rows():
            if 0 < float(rec["expires"]) <= now:
                continue
            if host and not _domain_covers(rec["domain"], host):
                continue
            out.append(rec)
        return out

    def status(self) -> list[dict[str, Any]]:
        """Per-domain counts + nearest expiry (no values) —
        the shape /cookies status renders."""
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for rec in self.cookies_for():
            by_domain.setdefault(rec["domain"], []).append(rec)
        out: list[dict[str, Any]] = []
        for domain in sorted(by_domain):
            rows = by_domain[domain]
            dated = [float(r["expires"]) for r in rows
                     if float(r["expires"]) > 0]
            out.append({"domain": domain, "count": len(rows),
                        "nearest_expiry": min(dated)
                        if dated else None})
        return out

    def domains(self) -> list[str]:
        """Domains holding at least one live cookie."""
        return [row["domain"] for row in self.status()]

    def has_domain(self, host: str) -> bool:
        """True when a live cookie covers *host* (site or
        subdomain) — the fact a recipe's ``needs_session``
        flag is built from."""
        return bool(self.cookies_for(host))

    def newest_updated_at(self) -> float:
        """The estate's high-water mark — the Netscape export
        cache compares against it."""
        cur = self.db.execute(
            "SELECT MAX(updated_at) FROM browser_cookies")
        row = cur.fetchone()
        if row is None or row[0] is None:
            return 0.0
        return float(row[0])

    def secret_values(self) -> set[str]:
        """Every live cookie value, in memory only — the
        recipe guard's comparison set.  Never logged, never
        returned to a chat surface."""
        return {str(r["value"]) for r in self.cookies_for()
                if str(r.get("value") or "")}

    # -- deletes ---------------------------------------------------

    def clear(self, target: str, *,
              only_imported: bool = False) -> tuple[int, list[str]]:
        """Remove one domain's cookies (the site and its
        subdomains), or every cookie for ``all``.  Returns
        (removed, affected domains).  With *only_imported*,
        only records that arrived as an owner bundle go —
        the /cookies clear contract: the sessions Devon
        earned herself survive a clear-all."""
        want = str(target or "").strip().lower()
        with self.db.transaction():
            rows = self._rows()
            if only_imported:
                rows = [r for r in rows if r.get("imported")]
            if want == "all":
                doomed = rows
            else:
                doomed = [r for r in rows
                          if _domain_within(r["domain"], want)]
            affected = sorted({r["domain"] for r in doomed})
            for rec in doomed:
                self.db.execute(
                    "DELETE FROM browser_cookies "
                    "WHERE domain = ? AND name = ? AND path = ?",
                    (rec["domain"], rec["name"], rec["path"]))
        _log.info("cookie estate clear (%s): %d removed across "
                  "%d domains", target, len(doomed), len(affected))
        return len(doomed), affected

    def clear_source(self, site: str, source: str) -> int:
        """Remove only the records *source* last wrote for a
        site — a solved-session clear must not take a login
        another surface earned for the same site.  *site* may
        be a bare domain or a URL.  Returns how many went."""
        site = str(site or "").strip()
        host = site
        if "://" in site:
            from urllib.parse import urlparse

            host = urlparse(site).hostname or site
        host = str(host or "").strip().lower().lstrip(".")
        source = str(source or "")
        removed = 0
        with self.db.transaction():
            for rec in self._rows():
                if rec["source"] != source:
                    continue
                if not host or not _domain_within(
                        rec["domain"], host):
                    continue
                self.db.execute(
                    "DELETE FROM browser_cookies "
                    "WHERE domain = ? AND name = ? AND path = ?",
                    (rec["domain"], rec["name"], rec["path"]))
                removed += 1
        return removed

    # -- the yt-dlp derivation -------------------------------------

    def export_netscape(self, dest: str | Path, *,
                        host: str = "") -> dict[str, Any]:
        """Write the live cookies as a Netscape cookies.txt
        for yt-dlp (the only consumer that reads files).
        A derivation, regenerated from the estate — never a
        source, never read back.  Written 0600.  Returns
        ``{path, count, skipped}``."""
        dest = Path(dest)
        records = self.cookies_for(host)
        lines = ["# Netscape HTTP Cookie File",
                 "# Exported from Devon's unified cookie estate "
                 "for yt-dlp."]
        skipped = 0
        for rec in records:
            line = _netscape_line(rec)
            if line is None:
                skipped += 1
            else:
                lines.append(line)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, dest)
        with suppress(OSError):
            os.chmod(dest, 0o600)
        return {"path": str(dest), "count": len(lines) - 2,
                "skipped": skipped}


def home_estate(*, db: Database | None = None,
                legacy_roots: dict[str, Any] | None = None
                ) -> CookieEstate:
    """The estate over the home database (``NM_HOME`` /
    ``~/.nomorals`` — the same home every other organ
    resolves).  An explicit *db* (tests) skips the open.
    Ensures tables + the one-time migration before returning.
    The home is resolved through :func:`load_settings`, not
    a bare :class:`Settings` — the bare default hardcodes
    ``~/.nomorals`` and ignores ``$NM_HOME``, which is the
    seam tests (and any redirected home) rely on."""
    if db is None:
        db, _recovered, _note = open_database(
            load_settings().db_path)
    estate = CookieEstate(db, legacy_roots=legacy_roots)
    estate.ensure()
    return estate


# ── the one-time legacy migration ─────────────────────────────

def _migration_flag(source: str, root: Any) -> str:
    digest = hashlib.sha256(
        str(root or "").encode("utf-8")).hexdigest()[:16]
    return f"migrated:{source}:{digest}"


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(payload, dict):
        rows = payload.get("cookies")
        return [r for r in (rows or []) if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _read_sidecar(profile_dir: Path | None) -> tuple[list[dict], int]:
    """The webnav imported-cookie sidecar."""
    if profile_dir is None:
        return [], 0
    path = Path(profile_dir) / "imported_cookies.json"
    if not path.is_file():
        return [], 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    records = _read_json_list(path)
    for rec in records:
        rec.setdefault("updated_at", mtime)
    return records, 1 if records else 0


def _read_profile_db(profile_dir: Path | None
                     ) -> tuple[list[dict], int]:
    """Chromium's own Cookies DB inside the webnav profile —
    the logins earned during captures, which never reached
    any sidecar.  Read off a temp copy (the live DB may be
    locked by a running Chromium); a DB that cannot be read
    is skipped, never fatal."""
    if profile_dir is None:
        return [], 0
    path = Path(profile_dir) / "Cookies"
    if not path.is_file():
        return [], 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    tmpdir = tempfile.mkdtemp(prefix="cookie-estate-")
    try:
        copy = Path(tmpdir) / "Cookies"
        shutil.copyfile(path, copy)
        conn = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT host_key, name, value, path, "
                "expires_utc, is_secure, is_httponly, "
                "samesite, last_access_utc FROM cookies"
                ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        _log.warning("cookie estate: profile Cookies DB "
                     "unreadable (%s) — skipped",
                     type(exc).__name__)
        return [], 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    records: list[dict] = []
    for (host_key, name, value, cpath, expires_utc, secure,
         http_only, samesite, last_access) in rows:
        expires = _chrome_ts_to_unix(expires_utc)
        records.append({
            "name": name, "value": value, "domain": host_key,
            "path": cpath or "/",
            "expires": expires if expires > 0 else -1.0,
            "secure": bool(secure), "httpOnly": bool(http_only),
            "sameSite": _CHROME_SAME_SITE.get(
                int(samesite) if samesite is not None else -1, ""),
            "updated_at": _chrome_ts_to_unix(last_access) or mtime,
        })
    return records, 1 if records else 0


def _read_service_jars(sessions_dir: Path | None,
                       service_data_dir: Path | None
                       ) -> tuple[list[dict], int]:
    """HTTP jars (the brain/downloader flat session dir and
    the service's per-session cookie dirs) plus rendered
    tabs' playwright-storage.json files."""
    records: list[dict] = []
    files = 0
    for directory in (sessions_dir,):
        if directory is None:
            continue
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name == "browser-cookies.txt":
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            rows = _read_json_list(path)
            for rec in rows:
                rec.setdefault("updated_at", mtime)
            if rows:
                files += 1
            records.extend(rows)
    if service_data_dir is not None:
        cookies_root = Path(service_data_dir) / "cookies"
        if cookies_root.is_dir():
            for path in sorted(
                    cookies_root.glob("*/*.json")):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                if path.name == "playwright-storage.json":
                    try:
                        payload = json.loads(
                            path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    rows = payload.get("cookies") \
                        if isinstance(payload, dict) else []
                    rows = [r for r in (rows or [])
                            if isinstance(r, dict)]
                else:
                    rows = _read_json_list(path)
                for rec in rows:
                    rec.setdefault("updated_at", mtime)
                if rows:
                    files += 1
                records.extend(rows)
    return records, files


def _read_solved(solved_dir: Path | None) -> tuple[list[dict], int]:
    """Solved-session files: the clearance cookies."""
    if solved_dir is None:
        return [], 0
    directory = Path(solved_dir)
    if not directory.is_dir():
        return [], 0
    records: list[dict] = []
    files = 0
    for path in sorted(directory.glob("*.json")):
        try:
            mtime = path.stat().st_mtime
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        rows = [r for r in (payload.get("cookies") or [])
                if isinstance(r, dict)]
        saved_at = payload.get("saved_at")
        try:
            stamp = float(saved_at) if saved_at else mtime
        except (TypeError, ValueError):
            stamp = mtime
        for rec in rows:
            rec.setdefault("updated_at", stamp)
        if rows:
            files += 1
        records.extend(rows)
    return records, files


def migrate_legacy(estate: CookieEstate, *,
                   profile_dir: Any = None,
                   service_data_dir: Any = None,
                   sessions_dir: Any = None,
                   solved_dir: Any = None,
                   now: float | None = None,
                   force: bool = False) -> dict[str, Any]:
    """Import the pre-unification stores into *estate*,
    once per source and root.  Import order is oldest-trust
    last: service jars, solved clearances, the webnav
    profile DB, and the owner-handed sidecar last, so an
    owner-handed record wins a recency tie.  Returns
    ``{migrated, sources: {name: {...}}}`` — a source that
    already migrated reports ``already`` and reads nothing.
    Never raises: one unreadable source is reported, not
    fatal, and the others still import."""
    now = time.time() if now is None else float(now)
    roots = {
        "service": (sessions_dir, service_data_dir),
        "solved": (solved_dir,),
        "webnav": (profile_dir,),
        "imported": (profile_dir,),
    }
    readers = {
        "service": lambda: _read_service_jars(
            Path(sessions_dir) if sessions_dir is not None else None,
            Path(service_data_dir)
            if service_data_dir is not None else None),
        "solved": lambda: _read_solved(
            Path(solved_dir) if solved_dir is not None else None),
        "webnav": lambda: _read_profile_db(
            Path(profile_dir) if profile_dir is not None else None),
        "imported": lambda: _read_sidecar(
            Path(profile_dir) if profile_dir is not None else None),
    }
    sources: dict[str, Any] = {}
    for name in ("service", "solved", "webnav", "imported"):
        root_key = ":".join(str(r or "") for r in roots[name])
        flag = _migration_flag(name, root_key)
        if not force and estate._meta_get(flag):
            sources[name] = {"already": True}
            continue
        try:
            records, files = readers[name]()
        except Exception as exc:  # noqa: BLE001 - one bad
            # source never blocks the others
            _log.warning("cookie estate migration (%s) failed: %s",
                         name, type(exc).__name__)
            sources[name] = {"error": type(exc).__name__}
            continue
        result = estate.upsert(records, source=name, now=now)
        estate._meta_set(flag, json.dumps({
            "files": files, "added": result["added"],
            "replaced": result["replaced"],
            "at": now}, sort_keys=True), now=now)
        sources[name] = {"files": files, **result}
        _log.info("cookie estate migration (%s): %d files, "
                  "%d added, %d replaced, %d kept newer, "
                  "%d skipped", name, files, result["added"],
                  result["replaced"], result["kept_newer"],
                  result["skipped"])
    return {"migrated": True, "sources": sources}
