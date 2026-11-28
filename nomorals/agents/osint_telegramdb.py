"""osint_telegramdb — the Telegram user database behind /lookup's
fourth phone route.

The database is the Telegram user corpus: account_id, phone and
username for the accounts in it, collected by scraping Telegram's
contact-import discovery and circulated as a leak.  It is mirrored
publicly on Hugging Face as ``rehuuuu/Telegram-DB`` (187.7M rows,
public, ungated, free to download — no key, no account, verified
live 2026-11-28: the datasets-server answers rows for it and a
downloaded shard answers username → phone lookups 3/3, with an
honest miss returning nothing).

Devon does NOT query Hugging Face at lookup time: the hosted
query endpoints do not answer for a dataset this size (the
datasets-server /filter endpoint rejects where clauses and the
/search index never finishes loading).  The provision downloads
the parquet set once and builds a local indexed SQLite copy;
lookups then run offline against that copy with the standard
library only (sqlite3) — no new runtime dependency.  DuckDB is
needed only at build time and the build installs it itself.

Honesty rules, same as the other phone routes:
- a number is returned only from a row in the local copy;
  absence is "not visible", never a guess;
- the data is STALE (collected 2019–2024): absence means
  nothing, presence is a signal to cross-check, never proof
  the number is still the account's;
- the row names its source: "Telegram user database
  (leaked corpus)".

``python -m nomorals.agents.osint_telegramdb build|verify|status``
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "DATASET",
    "DB_FILENAME",
    "NEED_BYTES",
    "TABLE",
    "build_db",
    "db_status",
    "ensure_telegramdb",
    "lookup_telegramdb",
    "telegramdb_path",
    "verify_db",
]

#: The public mirror of the Telegram user corpus.
DATASET = "rehuuuu/Telegram-DB"
HF_API_TREE = (
    "https://huggingface.co/api/datasets/rehuuuu/Telegram-DB/tree/main")
HF_RESOLVE = (
    "https://huggingface.co/datasets/rehuuuu/Telegram-DB/resolve/main")

#: The local copy: one SQLite file, two indexes.
DB_FILENAME = "telegramdb.sqlite"
TABLE = "telegram_users"

#: Free space the build needs: the SQLite copy holds ~9.7M usable
#: rows (~2 GB with indexes) plus working room.  The parquet set
#: itself (~4.6 GB) is streamed, never stored.
NEED_BYTES = 3 * 1024 ** 3

#: A usable row: a real account, a handle, a digit phone.
_PHONE_RE = re.compile(r"^[0-9]{7,15}$")


def _default_home() -> Path:
    env = os.environ.get("NM_HOME", "")
    if env.strip():
        return Path(env).expanduser()
    return Path.home() / ".nomorals"


def telegramdb_path(home: Any = None) -> Path:
    """Where the local copy lives in Devon's data estate."""
    base = Path(home).expanduser() if home else _default_home()
    return base / "osint" / DB_FILENAME


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n //= 1024
    return f"{n} GB"


def db_status(home: Any = None) -> dict[str, Any]:
    """Whether the local copy is stocked, and what it holds."""
    path = telegramdb_path(home)
    if not path.is_file():
        return {"present": False, "path": str(path), "rows": None,
                "bytes": None, "dataset": DATASET}
    rows = None
    dataset = DATASET
    try:
        con = sqlite3.connect(path)
        try:
            got = con.execute(
                "SELECT value FROM meta WHERE key = 'rows'"
                ).fetchone()
            rows = int(got[0]) if got else None
            got = con.execute(
                "SELECT value FROM meta WHERE key = 'dataset'"
                ).fetchone()
            dataset = str(got[0]) if got else DATASET
        finally:
            con.close()
    except sqlite3.Error:
        rows = None
    return {"present": True, "path": str(path), "rows": rows,
            "bytes": path.stat().st_size, "dataset": dataset}


def lookup_telegramdb(account_id: Any = None, username: Any = None,
                      *, home: Any = None,
                      db_path: Any = None) -> str | None:
    """The number for one account out of the local copy.

    The ID is tried first (the ID is the identity; a handle can
    move), then the username, case-insensitively.  When an ID or
    handle has several rows with different numbers — the corpus
    scraped accounts more than once — the most frequent number
    wins.  None when the copy is absent, unreadable, or holds no
    row: never a guess.  Read-only; never raises.
    """
    path = Path(db_path) if db_path else telegramdb_path(home)
    if not path.is_file():
        return None
    try:
        con = sqlite3.connect(path)
    except sqlite3.Error:
        return None
    try:
        aid = str(account_id or "").strip()
        if aid:
            row = con.execute(
                "SELECT phone, COUNT(*) AS c FROM telegram_users "
                "WHERE account_id = ? GROUP BY phone "
                "ORDER BY c DESC, phone ASC LIMIT 1",
                (aid,)).fetchone()
            if row:
                return str(row[0])
        uname = str(username or "").strip().lower()
        if uname:
            row = con.execute(
                "SELECT phone, COUNT(*) AS c FROM telegram_users "
                "WHERE username = ? GROUP BY phone "
                "ORDER BY c DESC, phone ASC LIMIT 1",
                (uname,)).fetchone()
            if row:
                return str(row[0])
    except sqlite3.Error:
        return None
    finally:
        con.close()
    return None


def _usable(row: tuple) -> tuple | None:
    """One raw row → a stored row, or None when it is junk."""
    if len(row) != 3:
        return None
    aid, uname, phone = (str(row[0] or "").strip(),
                         str(row[1] or "").strip().lower(),
                         str(row[2] or "").strip())
    if not aid or aid == "0" or not uname or not phone:
        return None
    if not _PHONE_RE.match(phone):
        return None
    return (aid, uname, phone)


def _write_sqlite(rows: Any, path: Any, *,
                  dataset: str = DATASET) -> dict[str, Any]:
    """Store usable rows into a SQLite file, atomically.

    Rows land in a temp file beside the target with both
    indexes, then the temp file replaces the target — a
    killed build never leaves a half-written database in
    place.  Returns the stored row count.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    count = 0
    con = sqlite3.connect(tmp)
    try:
        con.execute(
            f"CREATE TABLE {TABLE} (account_id TEXT NOT NULL, "
            "username TEXT NOT NULL, phone TEXT NOT NULL)")
        batch: list[tuple] = []
        for raw in rows:
            kept = _usable(raw)
            if kept is None:
                continue
            batch.append(kept)
            if len(batch) >= 100000:
                con.executemany(
                    f"INSERT INTO {TABLE} VALUES (?, ?, ?)", batch)
                count += len(batch)
                batch = []
        if batch:
            con.executemany(
                f"INSERT INTO {TABLE} VALUES (?, ?, ?)", batch)
            count += len(batch)
        con.execute(
            f"CREATE INDEX idx_tdb_account ON {TABLE}(account_id)")
        con.execute(
            f"CREATE INDEX idx_tdb_username ON {TABLE}(username)")
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, "
                    "value TEXT NOT NULL)")
        con.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [("dataset", dataset),
             ("rows", str(count)),
             ("built_at", str(int(time.time()))),
             ("source", "Telegram user database (leaked corpus)")])
        con.commit()
    finally:
        con.close()
    os.replace(tmp, path)
    return {"rows": count, "path": str(path)}


def verify_db(path: Any) -> dict[str, Any]:
    """Is this file a usable copy: table, both indexes, rows."""
    path = Path(path)
    if not path.is_file():
        return {"ok": False, "rows": 0, "detail": "no such file"}
    try:
        con = sqlite3.connect(path)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table'")}
            if TABLE not in tables or "meta" not in tables:
                return {"ok": False, "rows": 0,
                        "detail": "missing tables"}
            idx = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index'")}
            if not {"idx_tdb_account",
                    "idx_tdb_username"} <= idx:
                return {"ok": False, "rows": 0,
                        "detail": "missing indexes"}
            rows = con.execute(
                f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {"ok": False, "rows": 0, "detail": str(exc)}
    if rows <= 0:
        return {"ok": False, "rows": 0, "detail": "empty"}
    return {"ok": True, "rows": int(rows), "detail": "usable"}


def _parquet_urls() -> list[str]:
    """The corpus's parquet files, largest first — from the
    Hugging Face file list (public, no key).  Zero-byte
    placeholders are skipped: DuckDB cannot read them."""
    req = urllib.request.Request(HF_API_TREE, headers={
        "User-Agent": "Devon/lookup-telegramdb"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        import json

        files = json.loads(resp.read().decode("utf-8"))
    urls = []
    for entry in files:
        name = str(entry.get("path") or "")
        size = int(entry.get("size") or 0)
        if name.endswith(".parquet") and size > 0:
            urls.append(f"{HF_RESOLVE}/{name}")
    urls.sort(key=lambda u: u, reverse=True)
    if not urls:
        raise ValueError("no parquet files listed for " + DATASET)
    return urls


def _ensure_duckdb() -> dict[str, Any]:
    """DuckDB, the build-time parquet reader: import it, or
    install it (Devon provisions her own tools)."""
    try:
        import duckdb  # noqa: F401

        return {"ok": True, "detail": "duckdb present"}
    except ImportError:
        pass
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "duckdb"],
            capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "detail": f"pip install failed: {exc}"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-300:]
        return {"ok": False,
                "detail": f"pip install duckdb failed: {tail}"}
    try:
        import duckdb  # noqa: F401

        return {"ok": True, "detail": "duckdb installed"}
    except ImportError:
        return {"ok": False, "detail": "duckdb still unimportable"}


def _read_parquet_rows(urls: list[str]) -> Any:
    """Stream usable rows out of the parquet set, in batches.

    DuckDB reads the files over HTTPS — the 4.6 GB set is
    streamed, never stored.  Yields (account_id, username,
    phone) tuples; the junk filter lives in the SQL.
    """
    import duckdb

    quoted = []
    for url in urls:
        # An https:// mirror URL, a file:// URL, or a plain local
        # path (verification) — anything else is refused.
        local = not url.startswith(("https://", "file://"))
        if local and not os.path.isfile(url):
            raise ValueError(f"refusing parquet URL: {url!r}")
        if "'" in url:
            raise ValueError(f"refusing parquet URL: {url!r}")
        quoted.append(f"'{url}'")
    sql = (
        "SELECT account_id, lower(username) AS username, phone "
        f"FROM read_parquet([{', '.join(quoted)}]) "
        "WHERE account_id <> '0' AND account_id <> '' "
        "AND username <> '' AND phone <> '' "
        "AND regexp_matches(phone, '^[0-9]{7,15}$')")
    con = duckdb.connect()
    try:
        cur = con.execute(sql)
        while True:
            batch = cur.fetchmany(100000)
            if not batch:
                return
            yield from batch
    finally:
        con.close()


def build_db(home: Any = None, *, parquet_files: list[str] | None = None,
             need_bytes: int = NEED_BYTES,
             log: Any = None) -> dict[str, Any]:
    """Build the local copy: stream the corpus, keep usable
    rows, index, verify, put in place.

    ``parquet_files`` overrides the Hugging Face file list
    (local paths for verification).  Returns a status dict —
    never raises.  A killed build leaves the previous copy
    (or none) untouched: the write is atomic.
    """
    path = telegramdb_path(home)
    ready = verify_db(path)
    if ready["ok"]:
        return {"status": "ready", "rows": ready["rows"],
                "detail": "already stocked", "path": str(path)}
    home_dir = path.parent.parent
    home_dir.mkdir(parents=True, exist_ok=True)
    try:
        free = shutil.disk_usage(home_dir).free
    except OSError as exc:
        return {"status": "failed",
                "detail": f"cannot measure free disk: {exc}"}
    if free < int(need_bytes):
        return {"status": "blocked",
                "detail": f"needs {_human(int(need_bytes))} free "
                          f"to build, has {_human(free)}"}
    duck = _ensure_duckdb()
    if not duck["ok"]:
        return {"status": "failed", "detail": duck["detail"]}
    try:
        urls = list(parquet_files) if parquet_files else \
            _parquet_urls()
    except Exception as exc:  # noqa: BLE001 - honest status
        return {"status": "failed",
                "detail": f"cannot list the corpus files: {exc}"}
    if log:
        log("telegramdb: streaming %d parquet files", len(urls))
    try:
        written = _write_sqlite(_read_parquet_rows(urls), path)
    except Exception as exc:  # noqa: BLE001 - honest status
        _log.exception("telegramdb build failed")
        return {"status": "failed",
                "detail": f"build failed: {exc}"}
    checked = verify_db(path)
    if not checked["ok"]:
        return {"status": "failed",
                "detail": f"built file did not verify: "
                          f"{checked['detail']}"}
    return {"status": "ready", "rows": written["rows"],
            "detail": f"stocked {written['rows']} usable rows "
                      f"from {DATASET}",
            "path": str(path)}


def ensure_telegramdb(home: Any = None) -> dict[str, Any]:
    """The copy's status — no download.  The provision is
    explicit (a 4.6 GB corpus is not a quiet-boot act)."""
    st = db_status(home)
    if not st["present"]:
        return {"status": "missing",
                "detail": "not stocked — /lookup provision "
                          "telegramdb stocks it (downloads the "
                          "public 4.6 GB corpus and builds a "
                          "local index; needs ~3 GB free)",
                "path": st["path"]}
    checked = verify_db(st["path"])
    if not checked["ok"]:
        return {"status": "failed",
                "detail": f"copy present but unusable: "
                          f"{checked['detail']}",
                "path": st["path"]}
    return {"status": "ready", "rows": checked["rows"],
            "detail": f"stocked, {checked['rows']} usable rows",
            "path": st["path"]}


def main(argv: list[str] | None = None) -> int:
    """``python -m nomorals.agents.osint_telegramdb
    build|verify|status [--home DIR]``."""
    import argparse

    parser = argparse.ArgumentParser(prog="osint_telegramdb")
    parser.add_argument("command", choices=["build", "verify",
                                            "status"])
    parser.add_argument("--home", default=None)
    args = parser.parse_args(argv)
    if args.command == "status":
        st = ensure_telegramdb(home=args.home)
        print(f"{st['status']}: {st['detail']}")
        return 0 if st["status"] == "ready" else 1
    if args.command == "verify":
        res = verify_db(telegramdb_path(args.home))
        print(f"{'ok' if res['ok'] else 'bad'}: "
              f"{res['rows']} rows — {res['detail']}")
        return 0 if res["ok"] else 1
    res = build_db(home=args.home, log=_log.info)
    print(f"{res['status']}: {res.get('detail', '')}")
    return 0 if res["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
