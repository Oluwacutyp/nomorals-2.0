"""Cookie bundles: the owner's logged-in sessions, imported
into Devon's own browser profile (/cookies import|status|clear).

Android Chrome cannot export cookies, so the owner's phone
sessions reach Devon as a file: a bundle exported from a
browser that can (cookies.txt from a Firefox/extension
exporter, or a JSON cookie export).  This module parses the
two real exporter shapes —

* Netscape cookies.txt (tab-separated, ``#HttpOnly_`` lines
  honored), and
* JSON — a bare array of cookie objects, or a Playwright
  storage-state file (``{"cookies": [...]}``) —

normalizes them to one record shape, and merges them into
the unified cookie estate (``storage/cookie_estate.py``) —
the one store every browser surface reads and writes since
Navigator A6.  The old profile sidecar
(``imported_cookies.json``) is the legacy location: its
contents were imported once into the estate and it is never
written again; the webnav seam still applies the estate at
every session start, and to live sessions immediately.

Secret discipline is the organ's hard law: cookie VALUES
are credentials.  They are stored in the estate (the home
database) and nowhere else — never in a chat reply, never in
a log line, never in a journal row.  The journal records
domains, counts, the expired-dropped count and a SHA-256
fingerprint of the bundle bytes.  The uploaded bundle file
is deleted after the import attempt, success or failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...storage.cookie_estate import (
    SOURCE_IMPORTED,
    CookieEstate,
    home_estate,
)
from . import webnav

__all__ = [
    "BundleFormatError",
    "ImportOutcome",
    "ImportedCookiesStore",
    "ParseResult",
    "bundle_fingerprint",
    "clear_imported",
    "import_bundle_file",
    "parse_bundle",
    "render_import_reply",
    "render_status",
]

_log = logging.getLogger(__name__)

#: No real cookie bundle is anywhere near this size; a
#: bigger file is something else wearing the name.
MAX_BUNDLE_BYTES = 16 * 1024 * 1024


class BundleFormatError(ValueError):
    """The file is not a cookie bundle Devon can read.  The
    message is a plain sentence about the FORMAT — it never
    carries any of the file's content."""


@dataclass
class ParseResult:
    cookies: list[dict[str, Any]] = field(default_factory=list)
    dropped_expired: int = 0
    skipped_rows: int = 0
    format: str = ""


@dataclass
class ImportOutcome:
    ok: bool = False
    error: str = ""
    count: int = 0
    domains: list[str] = field(default_factory=list)
    dropped_expired: int = 0
    live_sessions: int = 0
    file_deleted: bool = False
    fingerprint: str = ""


# ── parsing ──────────────────────────────────────────────

def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _same_site(value: Any) -> str:
    return {"strict": "Strict", "lax": "Lax", "none": "None",
            "no_restriction": "None"}.get(
                str(value or "").strip().lower(), "")


def _record(*, name: str, value: str, domain: str,
            path: str, expires: float, http_only: bool,
            secure: bool, same_site: str) -> dict[str, Any]:
    return {"name": name, "value": value, "domain": domain,
            "path": path or "/", "expires": float(expires),
            "httpOnly": bool(http_only),
            "secure": bool(secure), "sameSite": same_site}


def _parse_netscape(text: str) -> tuple[list[dict[str, Any]], int]:
    """cookies.txt rows → unified records (expires 0, the
    format's session marker, becomes -1).  Comment lines are
    skipped; ``#HttpOnly_`` lines are cookies.  Rows that
    cannot be read are counted, never guessed at."""
    records: list[dict[str, Any]] = []
    skipped = 0
    for raw in text.splitlines():
        line = raw.strip("\r\n")
        if not line.strip():
            continue
        http_only = False
        if line.startswith("#HttpOnly_"):
            http_only = True
            line = line[len("#HttpOnly_"):]
        elif line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            skipped += 1
            continue
        domain, _flag, path, secure, expires_raw, name = \
            parts[0], parts[1], parts[2], parts[3], parts[4], \
            parts[5]
        value = "\t".join(parts[6:])
        if not name or not domain:
            skipped += 1
            continue
        try:
            expires = int(str(expires_raw).strip())
        except ValueError:
            skipped += 1
            continue
        records.append(_record(
            name=name, value=value, domain=domain,
            path=path or "/",
            expires=-1.0 if expires == 0 else float(expires),
            http_only=http_only,
            secure=str(secure).strip().upper() == "TRUE",
            same_site=""))
    return records, skipped


def _json_record(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "")
    domain = str(item.get("domain")
                 or item.get("host_key")
                 or item.get("hostKey")
                 or item.get("host") or "")
    if not name or not domain or "value" not in item:
        return None
    expires = -1.0
    if not _truthy(item.get("session")):
        raw_exp: Any = None
        for key in ("expires", "expirationDate", "expiry",
                    "expiresUtc", "expiration"):
            if item.get(key) is not None:
                raw_exp = item.get(key)
                break
        if raw_exp is not None:
            try:
                exp = float(raw_exp)
            except (TypeError, ValueError):
                exp = -1.0
            if exp > 1e11:      # milliseconds, not seconds
                exp /= 1000.0
            expires = exp
    return _record(
        name=name, value=str(item.get("value") or ""),
        domain=domain, path=str(item.get("path") or "/"),
        expires=expires,
        http_only=_truthy(item.get("httpOnly")),
        secure=_truthy(item.get("secure")),
        same_site=_same_site(item.get("sameSite")
                             or item.get("same_site")))


def _parse_json(data: bytes) -> tuple[list[dict[str, Any]], int, str]:
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleFormatError(
            "the JSON in that file could not be read — it "
            "is not a cookie export I can use") from exc
    fmt = "json"
    if isinstance(payload, dict):
        rows = payload.get("cookies")
        if not isinstance(rows, list):
            raise BundleFormatError(
                "that JSON holds no cookie list — it is not "
                "a cookie export I can use")
        fmt = "storage-state"
    elif isinstance(payload, list):
        rows = payload
    else:
        raise BundleFormatError(
            "that JSON is not a cookie list or a "
            "storage-state file — nothing in it reads as "
            "cookies")
    records: list[dict[str, Any]] = []
    skipped = 0
    for item in rows:
        rec = _json_record(item)
        if rec is None:
            skipped += 1
        else:
            records.append(rec)
    return records, skipped, fmt


def parse_bundle(data: bytes, *,
                 now: float | None = None) -> ParseResult:
    """Parse one bundle (either exporter shape) into
    normalized cookie records.  Expired cookies are dropped
    and COUNTED — never silently, never kept.  Raises
    :class:`BundleFormatError` with a content-free sentence
    when the file is not a readable bundle."""
    now = time.time() if now is None else float(now)
    if len(data) > MAX_BUNDLE_BYTES:
        raise BundleFormatError(
            "that file is far larger than any cookie bundle "
            "— it is not one I can read")
    if not data.strip():
        raise BundleFormatError(
            "the file is empty — there are no cookies in it")
    head = data.lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if head in (b"{", b"["):
        records, skipped, fmt = _parse_json(data)
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BundleFormatError(
                "that file is neither a cookies.txt bundle "
                "nor a JSON cookie export — nothing in it "
                "could be read as cookies") from exc
        records, skipped = _parse_netscape(text)
        fmt = "netscape"
        if not records:
            raise BundleFormatError(
                "that file is neither a cookies.txt bundle "
                "nor a JSON cookie export — nothing in it "
                "could be read as cookies")
    kept: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        if 0 < float(rec["expires"]) <= now:
            dropped += 1
        else:
            kept.append(rec)
    return ParseResult(cookies=kept, dropped_expired=dropped,
                       skipped_rows=skipped, format=fmt)


def bundle_fingerprint(data: bytes) -> str:
    """SHA-256 of the bundle bytes — the only bundle-derived
    string a journal row or log line may carry."""
    return hashlib.sha256(data).hexdigest()


# ── the profile's imported set ───────────────────────────
#
# The set lives in the unified estate now
# (storage/cookie_estate.py); this class is the bundle
# organ's view onto it.  The old sidecar-merge helpers are
# gone with the sidecar — dedupe/newest-wins is the estate's
# upsert.


@dataclass
class DomainStat:
    domain: str
    count: int
    nearest_expiry: float | None


class ImportedCookiesStore:
    """The imported-cookie set — the owner's bundles, held in
    the unified cookie estate (Navigator A6), not in a
    per-profile file.  ``profile_dir`` is kept as the legacy
    migration root (the old sidecar's home) and as the key
    live sessions are registered under; the records themselves
    live in the estate.  An explicit *db* (tests) or *estate*
    pins the backing store; otherwise the home estate is
    opened lazily."""

    def __init__(self, profile_dir: str | Path, *,
                 db: Any = None,
                 estate: CookieEstate | None = None) -> None:
        self.profile_dir = Path(profile_dir)
        self._db = db
        self._estate = estate

    def _store(self) -> CookieEstate:
        if self._estate is None:
            self._estate = home_estate(
                db=self._db,
                legacy_roots={"profile_dir": self.profile_dir})
        return self._estate

    @property
    def path(self) -> Path:
        """The legacy sidecar's path (migration source and
        diagnostics only — never written since A6)."""
        return webnav.imported_cookies_path(self.profile_dir)

    def load(self) -> list[dict[str, Any]]:
        return self._store().cookies_for()

    def save(self, records: list[dict[str, Any]]) -> None:
        """Replace the imported set with *records* (the old
        sidecar write, re-homed: the estate is the store
        now).  Only the imported flag is touched — the
        sessions Devon earned herself are never cleared by
        an import."""
        store = self._store()
        store.clear("all", only_imported=True)
        store.upsert(records, source=SOURCE_IMPORTED)

    def merge(self, records: list[dict[str, Any]]) -> tuple[int, int]:
        """Fold new records in.  Returns (added, replaced)."""
        result = self._store().upsert(
            records, source=SOURCE_IMPORTED)
        return int(result["added"]), int(result["replaced"])

    def status(self) -> list[DomainStat]:
        out: list[DomainStat] = []
        for row in self._store().status():
            out.append(DomainStat(
                domain=str(row["domain"]),
                count=int(row["count"]),
                nearest_expiry=(float(row["nearest_expiry"])
                                if row["nearest_expiry"] is not None
                                else None)))
        return out

    def clear(self, target: str) -> tuple[int, list[str]]:
        """Remove one domain's imported cookies (or every
        imported cookie for ``all``).  Only the imported set
        goes — the sessions Devon earned herself survive,
        exactly as the old sidecar clear left the profile's
        own sessions alone.  Returns (removed, affected
        domains)."""
        return self._store().clear(target, only_imported=True)


# ── journal ──────────────────────────────────────────────

def _append_journal(journal_path: Path | None,
                    entry: dict[str, Any]) -> None:
    """One row in the personal surfaces' journal (the
    /pair journal store) — domains, counts, fingerprint;
    never a value.  Best-effort: a journal that cannot be
    written never fails the import it describes."""
    if journal_path is None:
        return
    try:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False)
                     + "\n")
    except OSError:
        _log.debug("cookie bundle journal write failed",
                   exc_info=True)


# ── the import itself ────────────────────────────────────

def import_bundle_file(path: str | Path,
                       profile_dir: str | Path, *,
                       journal_path: str | Path | None = None,
                       now: float | None = None,
                       db: Any = None,
                       estate: CookieEstate | None = None
                       ) -> ImportOutcome:
    """Read one uploaded bundle, merge it into the estate,
    apply it to any live capture session, delete the
    uploaded file, and journal the outcome.  Never raises
    for an unreadable or malformed bundle — the outcome
    carries a plain sentence instead."""
    src = Path(path)
    profile = Path(profile_dir)
    outcome = ImportOutcome()
    data: bytes | None = None
    error_kind = ""
    try:
        data = src.read_bytes()
    except OSError:
        error_kind = "missing"
        outcome.error = ("the bundle file was gone before I "
                         "could read it — send it again.")
    if data is not None:
        outcome.fingerprint = bundle_fingerprint(data)
        try:
            result = parse_bundle(data, now=now)
            store = ImportedCookiesStore(
                profile, db=db, estate=estate)
            store.merge(result.cookies)
            outcome.live_sessions = \
                webnav.apply_cookies_to_live(
                    profile, result.cookies)
            outcome.ok = True
            outcome.count = len(result.cookies)
            outcome.domains = sorted(
                {str(r.get("domain") or "")
                 for r in result.cookies})
            outcome.dropped_expired = result.dropped_expired
            _log.info(
                "cookie bundle imported: %d cookies across "
                "%d domains, %d expired dropped, %d live "
                "sessions updated",
                outcome.count, len(outcome.domains),
                outcome.dropped_expired, outcome.live_sessions)
        except BundleFormatError as exc:
            error_kind = "format"
            outcome.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - an import
            # failure is a sentence, never a crash in chat
            error_kind = type(exc).__name__
            outcome.error = ("the import failed partway — "
                             "send the bundle again and I "
                             "will retry it whole.")
            _log.warning("cookie bundle import failed (%s)",
                         error_kind, exc_info=True)
    try:
        src.unlink()
        outcome.file_deleted = True
    except FileNotFoundError:
        outcome.file_deleted = error_kind == "missing"
    except OSError:
        outcome.file_deleted = False
    entry: dict[str, Any] = {
        "ts": time.time(), "kind": "cookies_import",
        "ok": outcome.ok, "fingerprint": outcome.fingerprint,
        "file_deleted": outcome.file_deleted,
    }
    if outcome.ok:
        entry.update({
            "domains": outcome.domains,
            "count": outcome.count,
            "dropped_expired": outcome.dropped_expired,
            "live_sessions": outcome.live_sessions})
    else:
        entry["error_kind"] = error_kind
    _append_journal(
        Path(journal_path) if journal_path else None, entry)
    return outcome


# ── replies ──────────────────────────────────────────────

def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def render_import_reply(outcome: ImportOutcome) -> str:
    if not outcome.ok:
        return f"🍪 {outcome.error}"
    if outcome.count == 0:
        return (f"🍪 that bundle's {_plural(outcome.dropped_expired, 'cookie')} "
                "had all already expired — nothing was "
                "imported. Export a fresh bundle and send "
                "it again.")
    parts = [
        f"🍪 imported {_plural(outcome.count, 'cookie')} "
        f"across {_plural(len(outcome.domains), 'domain')} "
        "into my browser sessions"]
    if outcome.dropped_expired:
        parts.append(
            f"{_plural(outcome.dropped_expired, 'expired one')} "
            "dropped")
    live = ""
    if outcome.live_sessions:
        live = (" — and applied to the capture session "
                "running right now")
    return (" · ".join(parts) + live + ". They ride every "
            "capture from now on — /webshot a page that "
            "looks different when you're signed in and "
            "you'll see your view, not the logged-out one.")


def _expiry_phrase(stat: DomainStat) -> str:
    if stat.nearest_expiry is None:
        return "session cookies only"
    when = time.strftime("%d %b %Y",
                         time.gmtime(stat.nearest_expiry))
    return f"nearest expiry {when}"


def render_status(profile_dir: str | Path, *,
                  db: Any = None,
                  estate: CookieEstate | None = None) -> str:
    stats = ImportedCookiesStore(
        profile_dir, db=db, estate=estate).status()
    if not stats:
        return ("🍪 no imported cookies yet — send a bundle "
                "with /cookies import and your signed-in "
                "sessions ride my captures.")
    total = sum(s.count for s in stats)
    lines = [f"🍪 imported cookies in my browser sessions "
             f"— {_plural(len(stats), 'domain')}, "
             f"{_plural(total, 'cookie')}:"]
    for stat in stats:
        lines.append(f"· {stat.domain} — "
                     f"{_plural(stat.count, 'cookie')} · "
                     f"{_expiry_phrase(stat)}")
    lines.append("Clear one with /cookies clear <domain>, "
                 "or everything with /cookies clear all.")
    return "\n".join(lines)


def clear_imported(profile_dir: str | Path, target: str, *,
                   journal_path: str | Path | None = None,
                   db: Any = None,
                   estate: CookieEstate | None = None
                   ) -> str:
    """Clear one domain (or all) from the imported set,
    journal it, ask live sessions to drop them too, and
    render the answer.  Only the imported set goes — the
    sessions Devon earned herself are left alone."""
    profile = Path(profile_dir)
    store = ImportedCookiesStore(profile, db=db, estate=estate)
    removed, affected = store.clear(target)
    if affected:
        webnav.clear_cookies_from_live(profile, affected)
    _append_journal(
        Path(journal_path) if journal_path else None,
        {"ts": time.time(), "kind": "cookies_clear",
         "target": target.strip().lower()
         if target.strip().lower() == "all" else target,
         "removed": removed})
    if target.strip().lower() == "all":
        if not removed:
            return ("🍪 there were no imported cookies to "
                    "clear — my browser sessions hold only "
                    "their own logins.")
        return (f"🍪 removed all {_plural(removed, 'imported cookie')} "
                "— my browser sessions are back to their own "
                "logins only.")
    if not removed:
        return (f"🍪 no imported cookies matched {target} "
                "— nothing was removed.")
    return (f"🍪 removed {_plural(removed, 'imported cookie')} "
            f"for {target} — captures for that site start "
            "logged-out again.")
