"""Drop-in inbox (Prompt 06): drop a file, link, or note — Devon acts.

The owner drops files into ``inbox/`` (global) or ``rooms/<slug>/inbox/`` and
Devon figures out what to do: read it, classify the intent, act (summarize /
file / remind / watch / research / describe media / apply media directives),
and report back with a one-line notification. Zero chat needed for the simple
cases; one clarifying question max for the ambiguous ones.

Design notes:

* **Deterministic first.** Explicit ``@directives`` always win; the fallback
  classifier is extension/MIME based and works with zero model calls. A real
  model classifier can be injected via ``Inbox(classifier=...)``.
* **Handlers are injectable.** Every intent maps to a handler
  ``(inbox, item) -> ActionResult``; pass ``handlers={...}`` to override or
  extend. Unit tests inject fakes — no network egress in tests.
* **The inbox is an attack surface.** Dropped files are never executed.
  Executables, archives, and double-extension decoys (``report.pdf.exe``) are
  quarantined before any model sees them. Size caps apply. Secret-shaped
  content is redacted in every summary/notification.
* **Crash-safe.** Items flip to ``processing`` inside a transaction (only one
  sweeper wins); on startup anything stuck in ``processing`` for >30 min goes
  back to ``pending``.
* **Room-aware, with fallback.** A ``RoomProvider`` supplies
  ``rooms/<slug>/inbox/`` and ``rooms/<slug>/files/`` paths (the interface
  Prompt 05 needs); without one everything falls back to the global inbox.
"""

from __future__ import annotations

import asyncio
import inspect
import mimetypes
import os
import re
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from ..agents.notifier import Notifier
from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..storage.db import Database

log = get_logger(__name__)

# ── tuning ───────────────────────────────────────────────────────────────────

#: background sweep cadence for InboxWatcher
SWEEP_INTERVAL_S = 120.0
#: a `processing` item older than this is presumed crashed → back to `pending`
STALE_PROCESSING_S = 30 * 60.0
#: intake caps (spec §3)
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_INBOX_BYTES = 2 * 1024 * 1024 * 1024
#: never read more than this of a dropped file as text
READ_CAP_BYTES = 200 * 1024
#: default reminder horizon when a time can't be parsed → ask, don't guess
DEFAULT_REMIND_DELTA_S = 24 * 3600.0

STATUSES = ("pending", "processing", "done", "needs_input", "quarantined", "failed")
KINDS = ("file", "link", "note")

INTENTS = (
    "summarize", "describe", "file", "remind", "watch", "research",
    "transcribe", "media_square", "media_trim", "media_gif", "needs_input",
    "image",  # Prompt 09: vision intent for dropped images
)

#: quarantined before any model sees the content (spec §3)
EXECUTABLE_EXTS = {
    ".exe", ".bat", ".cmd", ".com", ".msi", ".sh", ".ps1", ".vbs",
    ".scr", ".jar", ".dmg", ".pkg", ".run", ".bin",
}
ARCHIVE_EXTS = {
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".rar", ".7z", ".iso",
}
#: magic bytes → quarantine even when the extension lies
QUARANTINE_MAGIC = (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"MZ\x90\x00")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".opus"}
TEXT_EXTS = {
    ".md", ".txt", ".rst", ".log",
    # documents we can safely extract printable text from → summarize
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".epub",
    ".csv", ".json", ".yaml", ".yml", ".toml", ".org", ".markdown",
}

#: explicit directives — these ALWAYS beat model classification (spec §2)
DIRECTIVE_INTENTS = {
    "summarize": "summarize",
    "describe": "describe",
    "remind": "remind",
    "watch": "watch",
    "room": "file",          # @room <slug> → file into that room
    "research": "research",
    "transcribe": "transcribe",
    "square": "media_square",  # Prompt 15: image/video edit directives
    "trim": "media_trim",
    "gif": "media_gif",
    "read-text": "image",  # Prompt 09: vision directives for dropped images
    "locate": "image",
}

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[bap]-?[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(api[_-]?key|apikey|auth[_-]?token|client[_-]?secret|secret|"
        r"password|passwd)\b\s*[:=]\s*['\"]?([^\s'\"]{6,})"
    ),
]


def redact_secrets(text: str) -> str:
    """Replace secret-shaped content with [redacted]. Never echoes secrets."""
    out = text or ""
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[redacted]", out)
    return out


def sanitize_name(name: str) -> str:
    """Make a dropped filename safe: no traversal, no absolute, no controls."""
    name = (name or "").strip().replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = name.strip().strip(".")
    if not name or name in {".", ".."}:
        name = f"item-{new_short_id()}"
    return name[:200]


def _looks_executable(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return head.startswith(QUARANTINE_MAGIC) or head.startswith(b"#!")

# ── text helpers ─────────────────────────────────────────────────────────────


def read_dropped_text(path: str | os.PathLike[str],
                      cap: int = READ_CAP_BYTES) -> tuple[str, bool]:
    """Read a dropped file as text. Returns (text, is_binary).

    Never executes anything; binary files fall back to printable-run
    extraction (like `strings`) so PDFs still yield something useful.
    """
    p = Path(path)
    try:
        raw = p.read_bytes()[:cap]
    except OSError:
        return "", True
    if not raw:
        return "", False
    # binary test: NUL byte or >30% non-printable
    text_chars = bytes({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0})
    nontext = sum(b not in text_chars for b in raw)
    if b"\x00" in raw or nontext > len(raw) * 0.30:
        runs = re.findall(rb"[ -~]{6,}", raw)
        text = "\n".join(r.decode("ascii", "replace") for r in runs[:400])
        return text, True
    return raw.decode("utf-8", "replace"), False


def extractive_summary(text: str, max_chars: int = 600) -> str:
    """Naive model-free summary: first sentences up to max_chars."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return "empty file"
    # prefer whole sentences
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    out: list[str] = []
    total = 0
    for s in sentences:
        if total + len(s) > max_chars and out:
            break
        out.append(s)
        total += len(s) + 1
    summary = " ".join(out).strip()
    return summary if len(summary) <= max_chars else summary[:max_chars] + "…"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> tuple[str, str]:
    """(title, body text) from HTML. Crude but dependency-free."""
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
    ext = _TextExtractor()
    try:
        ext.feed(html[:500_000])
    except Exception:  # noqa: BLE001 - malformed HTML still yields partial text
        pass
    return title, re.sub(r"\s+", " ", " ".join(ext.parts)).strip()


def default_fetch(url: str, timeout: float = 20.0) -> str:
    """Fetch a URL, return text. Injectable — tests pass a fake."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "DevonInbox/1.0 (+drop-in inbox)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read(1_000_000)
    try:
        html = raw.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        html = ""
    if "<html" in html[:2000].lower():
        title, body = html_to_text(html)
        return f"{title}\n\n{body}" if title else body
    return html


# ── @directive parsing ────────────────────────────────────────────────────────


def parse_directive(line: str) -> tuple[str, str] | None:
    """Parse `@name arg...` from a note's first line. None if not a directive."""
    line = (line or "").strip()
    if not line.startswith("@"):
        return None
    m = re.match(r"@([A-Za-z_][\w-]*)\s*(.*)$", line)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).strip()


def parse_when(text: str, now: float | None = None) -> float | None:
    """Parse `@remind <when>`-style times. Returns epoch seconds or None.

    Handles: `friday 9am`, `next monday 17:30`, `tomorrow`, `today 5pm`,
    `in 2 hours`, `in 30 minutes`, `in 3 days`, ISO datetimes.
    """
    now = now if now is not None else time.time()
    s = (text or "").strip().lower()
    if not s:
        return None
    base = datetime.fromtimestamp(now)

    m = re.match(r"in\s+(\d+)\s*(minute|hour|day|week)s?", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}[unit]
        return now + n * delta

    try:  # ISO first: 2026-10-05 09:00
        dt = datetime.fromisoformat(s)
        ts = dt.timestamp()
        return ts if ts > now - 60 else None
    except ValueError:  # noqa: E103 - not ISO format, try other date formats below
        pass

    # optional time-of-day: 9am, 5:30pm, 17:30
    tm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", s)
    hour, minute = 9, 0
    if tm:
        hour = int(tm.group(1))
        minute = int(tm.group(2) or 0)
        ampm = tm.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0

    if "tomorrow" in s:
        day = base + timedelta(days=1)
        return day.replace(hour=hour, minute=minute, second=0,
                           microsecond=0).timestamp()
    if "today" in s:
        dt = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        ts = dt.timestamp()
        return ts if ts > now else ts + 86400

    for name, wd in _WEEKDAYS.items():
        if re.search(rf"\bnext\s+{name}\b", s) or re.search(rf"\b{name}\b", s):
            days_ahead = (wd - base.weekday()) % 7
            if "next" in s and days_ahead == 0:
                days_ahead = 7
            if days_ahead == 0:  # later today, else next week
                probe = base.replace(hour=hour, minute=minute, second=0,
                                     microsecond=0).timestamp()
                if probe > now:
                    return probe
                days_ahead = 7
            day = base + timedelta(days=days_ahead)
            return day.replace(hour=hour, minute=minute, second=0,
                               microsecond=0).timestamp()
    return None

# ── model ────────────────────────────────────────────────────────────────────


@dataclass
class InboxItem:
    """One dropped thing being (or having been) processed."""

    id: str
    name: str
    path: str
    kind: str = "file"          # file | link | note
    mime: str = ""
    size_bytes: int = 0
    received_at: float = field(default_factory=time.time)
    status: str = "pending"     # pending|processing|done|needs_input|quarantined|failed
    room: str | None = None
    directive: str = ""         # raw "@name arg" when an explicit directive applied
    directive_target: str = ""  # inbox-relative file the directive acts on
    intent: str = ""
    action_taken: str = ""
    result_summary: str = ""
    error: str = ""
    attempts: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name)
                for f in self.__dataclass_fields__.values()}


@dataclass
class ActionResult:
    """What a handler did with an item."""

    summary: str
    notify: str = ""            # one-liner for the owner; "" = silent
    disposition: str = "processed"  # processed | archive | room_files | stay
    room: str | None = None


class RoomProvider(Protocol):
    """Interface Prompt 05's RoomManager will satisfy.

    Return None from either method to fall back to the global inbox.
    """

    def inbox_path(self, slug: str) -> Path | None: ...
    def files_path(self, slug: str) -> Path | None: ...


class FilesystemRooms:
    """Default RoomProvider: rooms/<slug>/inbox|files under the workspace root."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def inbox_path(self, slug: str) -> Path | None:
        p = self.root / "rooms" / sanitize_name(slug) / "inbox"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def files_path(self, slug: str) -> Path | None:
        p = self.root / "rooms" / sanitize_name(slug) / "files"
        p.mkdir(parents=True, exist_ok=True)
        return p


class Inbox:
    """The drop-in inbox. See module docstring for the design."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        db: Database | None = None,
        *,
        notifier: Any = None,
        scheduler: Any = None,
        classifier: Callable[["Inbox", InboxItem], str | None] | None = None,
        handlers: dict[str, Callable[["Inbox", InboxItem], ActionResult]] | None = None,
        fetcher: Callable[[str], str] | None = None,
        room_provider: RoomProvider | None = None,
        vision: Callable[[bytes, str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.dir = self.root / "inbox"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._processed = self.dir / "processed"
        self._quarantine = self.dir / "quarantine"
        self._archive = self.dir / "archive"
        for d in (self._processed, self._quarantine, self._archive):
            d.mkdir(parents=True, exist_ok=True)

        self.db = db or Database(self.dir / "inbox.db")
        self._ensure_schema()
        self._recover_stale()  # crash recovery, once per startup

        self._notifier = notifier
        self._scheduler = scheduler
        self._classifier = classifier or _default_classify
        self._fetch = fetcher or default_fetch
        #: Prompt 09 vision hook: ``(image_bytes, action, prompt) -> dict``.
        #: action is "describe" | "read_text" | "locate" (prompt = locate
        #: target). None → the image intent degrades to probe-only, honestly.
        self._vision = vision
        self.rooms = room_provider or FilesystemRooms(self.root)
        self.handlers: dict[str, Callable[[Inbox, InboxItem], ActionResult]] = {
            **_default_handlers(), **(handlers or {}),
        }
        self._sweep_lock = threading.Lock()

    # ── schema / recovery ────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS inbox_items (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'file', mime TEXT DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    received_at REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    room TEXT, directive TEXT DEFAULT '', directive_target TEXT DEFAULT '',
                    intent TEXT DEFAULT '', action_taken TEXT DEFAULT '',
                    result_summary TEXT DEFAULT '', error TEXT DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS inbox_history (
                    id TEXT PRIMARY KEY, item_id TEXT NOT NULL, intent TEXT DEFAULT '',
                    action TEXT DEFAULT '', started_at REAL NOT NULL,
                    ended_at REAL NOT NULL, outcome TEXT NOT NULL, detail TEXT DEFAULT ''
                )
            """)

    def _recover_stale(self) -> None:
        """Crash recovery: `processing` older than 30 min → `pending`, once."""
        cutoff = time.time() - STALE_PROCESSING_S
        with self.db.transaction():
            cur = self.db.execute(
                "UPDATE inbox_items SET status='pending', attempts=attempts+1,"
                " updated_at=? WHERE status='processing' AND updated_at < ?",
                (time.time(), cutoff),
            )
        if cur.rowcount:
            log.warning("inbox: recovered %d stale processing item(s) to pending",
                        cur.rowcount)

    # ── locations ──────────────────────────────────────────────────────

    def inbox_dir(self, room: str | None = None) -> Path:
        if room:
            p = self.rooms.inbox_path(room)
            if p is not None:
                return p
        return self.dir

    def _row_to_item(self, row: dict[str, Any]) -> InboxItem:
        return InboxItem(**{k: row[k] for k in InboxItem.__dataclass_fields__
                            if k in row})

    # ── intake ─────────────────────────────────────────────────────────

    def _quarantine_reason(self, path: Path) -> str | None:
        ext = path.suffix.lower()
        stem_exts = path.name.lower().split(".")[1:]
        if ext in EXECUTABLE_EXTS:
            if len(stem_exts) > 1:
                return f"double extension decoy ({path.name})"
            return f"executable type ({ext})"
        if ext in ARCHIVE_EXTS:
            return f"archive type ({ext}); no auto-extract in v1"
        if _looks_executable(path):
            return "executable magic bytes"
        return None

    def _inbox_size(self) -> int:
        total = 0
        for p in self.dir.rglob("*"):
            if p.is_file() and self._quarantine not in p.parents \
                    and self._processed not in p.parents:
                try:
                    total += p.stat().st_size
                except OSError:  # noqa: E103 - file vanished mid-scan
                    pass
        return total

    def intake(self, source: str | os.PathLike[str], *,
               room: str | None = None,
               kind_hint: str | None = None) -> InboxItem:
        """Register a dropped file/dir with the inbox.

        The owner moves things into the inbox dir; intake also accepts paths
        from outside and moves them in. Directories are flattened with
        ``subdir__file`` namespacing. Returns the registered InboxItem
        (directories register each contained file and return the first).
        """
        src = Path(source)
        target_dir = self.inbox_dir(room)
        items: list[InboxItem] = []

        def _register_file(f: Path, *, already_inside: bool) -> InboxItem:
            name = sanitize_name(f.name)
            dest = target_dir / name
            if not already_inside:
                if dest.exists():
                    dest = target_dir / f"{dest.stem}__{new_short_id()}{dest.suffix}"
                shutil.move(str(f), str(dest))
            else:
                if f.name != name or f.parent != target_dir:
                    new_dest = target_dir / name
                    if new_dest != f:
                        f.rename(new_dest)
                    dest = new_dest
                else:
                    dest = f
            return self._register(dest, room=room, kind_hint=kind_hint)

        if src.is_dir():
            for child in sorted(src.iterdir()):
                if child.is_file():
                    ns = sanitize_name(f"{src.name}__{child.name}")
                    moved = target_dir / ns
                    if moved.exists():
                        moved = target_dir / f"{moved.stem}__{new_short_id()}{moved.suffix}"
                    shutil.move(str(child), str(moved))
                    items.append(self._register(moved, room=room,
                                                kind_hint=kind_hint))
            try:
                src.rmdir()
            except OSError:  # noqa: E103 - source dir may not be empty, leave it
                pass
            if not items:
                raise ValueError(f"nothing to intake in {src}")
            return items[0]

        if not src.exists():
            raise FileNotFoundError(f"no such file: {src}")
        inside = target_dir in src.resolve().parents \
            or src.resolve().parent == target_dir.resolve()
        return _register_file(src, already_inside=inside)

    def _register(self, dest: Path, *, room: str | None,
                  kind_hint: str | None) -> InboxItem:
        # quarantine BEFORE anything (model included) sees the content
        reason = self._quarantine_reason(dest)
        if reason:
            q = self._quarantine / dest.name
            if q.exists():
                q = self._quarantine / f"{q.stem}__{new_short_id()}{q.suffix}"
            shutil.move(str(dest), str(q))
            item = self._insert(dest.name, str(q), room, kind="file",
                                status="quarantined",
                                result_summary=f"quarantined: {reason}")
            self._notify("inbox", f"inbox: quarantined {dest.name}",
                         f"{reason}; release with `nm inbox release {item.id}`",
                         critical=True)
            log.warning("inbox: quarantined %s (%s)", dest.name, reason)
            return item

        size = dest.stat().st_size
        if size > MAX_FILE_BYTES:
            item = self._insert(dest.name, str(dest), room, kind="file",
                                status="needs_input",
                                result_summary=f"over size cap ({size} bytes)",
                                error="too large — confirm processing")
            self._notify("inbox", f"inbox: {dest.name} is too large",
                         f"{size} bytes exceeds the {MAX_FILE_BYTES}-byte cap. "
                         f"Reply to confirm or `nm inbox retry {item.id}`.")
            return item
        if self._inbox_size() > MAX_INBOX_BYTES:
            item = self._insert(dest.name, str(dest), room, kind="file",
                                status="needs_input",
                                result_summary="inbox total over 2GB cap")
            self._notify("inbox", "inbox: total size over cap",
                         "Inbox exceeds 2GB; oldest items should be archived.")
            return item

        kind = kind_hint or self._detect_kind(dest)
        return self._insert(dest.name, str(dest), room, kind=kind)

    def _detect_kind(self, dest: Path) -> str:
        ext = dest.suffix.lower()
        if ext in (".link", ".url"):
            return "link"
        if ext in TEXT_EXTS:
            try:
                head = dest.read_bytes()[:4096].decode("utf-8", "replace")
            except OSError:
                head = ""
            if head.lstrip().lower().startswith("link:"):
                return "link"
            return "note"
        return "file"

    def _insert(self, name: str, path: str, room: str | None, *,
                kind: str = "file", status: str = "pending",
                result_summary: str = "", error: str = "") -> InboxItem:
        item = InboxItem(
            id=new_short_id("inbox"), name=name, path=path, kind=kind,
            mime=mimetypes.guess_type(name)[0] or "",
            size_bytes=Path(path).stat().st_size if Path(path).exists() else 0,
            status=status, room=room, result_summary=result_summary, error=error,
        )
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO inbox_items (id, name, path, kind, mime, size_bytes,"
                " received_at, status, room, directive, directive_target, intent,"
                " action_taken, result_summary, error, attempts, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item.id, item.name, item.path, item.kind, item.mime,
                 item.size_bytes, item.received_at, item.status, item.room,
                 item.directive, item.directive_target, item.intent,
                 item.action_taken, item.result_summary, item.error,
                 item.attempts, item.updated_at),
            )
        return item

    def add_link(self, url: str, *, note: str = "",
                 room: str | None = None) -> InboxItem:
        """`nm inbox add-link <url> [--room slug] [--note ...]`."""
        url = (url or "").strip()
        if not re.match(r"https?://", url, re.I):
            raise ValueError(f"not an http(s) URL: {url!r}")
        target_dir = self.inbox_dir(room)
        dest = target_dir / f"link-{new_short_id()}.link"
        content = url + ("\n" + note.strip() if note.strip() else "") + "\n"
        dest.write_text(content, encoding="utf-8")
        return self._register(dest, room=room, kind_hint="link")

    # ── queries / lifecycle ────────────────────────────────────────────

    def list_items(self, status: str | None = None, room: str | None = None,
                   limit: int = 50) -> list[InboxItem]:
        sql = "SELECT * FROM inbox_items"
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if room:
            clauses.append("room = ?")
            params.append(room)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY received_at DESC LIMIT ?"
        params.append(limit)
        return [self._row_to_item(r) for r in self.db.query(sql, params)]

    def get_item(self, item_id: str) -> InboxItem:
        row = self.db.query_one("SELECT * FROM inbox_items WHERE id = ?",
                                (item_id,))
        if row is None:
            raise KeyError(f"no inbox item {item_id}")
        return self._row_to_item(row)

    def retry(self, item_id: str) -> InboxItem:
        """Re-queue a failed / needs_input item."""
        item = self.get_item(item_id)
        if item.status not in ("failed", "needs_input"):
            raise ValueError(f"item {item_id} is {item.status}; nothing to retry")
        with self.db.transaction():
            self.db.execute(
                "UPDATE inbox_items SET status='pending', error='',"
                " updated_at=? WHERE id=?", (time.time(), item_id))
        return self.get_item(item_id)

    def release(self, item_id: str) -> InboxItem:
        """Release a quarantined item back to the inbox (explicit, logged)."""
        item = self.get_item(item_id)
        if item.status != "quarantined":
            raise ValueError(f"item {item_id} is {item.status}; not quarantined")
        dest = self.inbox_dir(item.room) / item.name
        if dest.exists():
            dest = dest.parent / f"{dest.stem}__{new_short_id()}{dest.suffix}"
        shutil.move(item.path, str(dest))
        log.warning("inbox: quarantine released by owner: %s (%s)",
                    item.name, item.id)
        with self.db.transaction():
            self.db.execute(
                "UPDATE inbox_items SET status='pending', path=?, error='',"
                " result_summary='released from quarantine by owner',"
                " updated_at=? WHERE id=?", (str(dest), time.time(), item_id))
            self.db.execute(
                "INSERT INTO inbox_history (id, item_id, intent, action,"
                " started_at, ended_at, outcome, detail)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (new_short_id("hx"), item_id, "quarantined", "release",
                 time.time(), time.time(), "released",
                 f"released from quarantine to {dest.name} by owner"))
        return self.get_item(item_id)

    def history(self, item_id: str | None = None,
                limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM inbox_history"
        params: list[Any] = []
        if item_id:
            sql += " WHERE item_id = ?"
            params.append(item_id)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return self.db.query(sql, params)

    # ── classification ─────────────────────────────────────────────────

    def _directive_for(self, item: InboxItem) -> tuple[str, str, str] | None:
        """Resolve an explicit @directive for an item.

        Returns (name, arg, target_path). Notes carry the directive on line 1
        (optionally naming a target file: `@square photo.jpg`); media files
        pick up a sibling `<stem>.md`/`.txt` sidecar's first line.
        """
        if item.kind == "note":
            try:
                first = Path(item.path).read_text(
                    encoding="utf-8", errors="replace").splitlines()
                first_line = first[0] if first else ""
            except OSError:
                return None
            parsed = parse_directive(first_line)
            if not parsed:
                return None
            name, arg = parsed
            target = ""
            if arg:
                cand = Path(item.path).parent / sanitize_name(arg.split()[0])
                if cand.exists() and cand.is_file():
                    target = str(cand)
                    arg = " ".join(arg.split()[1:])
            if not target and name in ("square", "trim", "gif", "read-text", "locate"):
                # media directive without a target: am I a sidecar note for a
                # sibling media file? If so, defer — the media file claims it.
                p = Path(item.path)
                for mext in IMAGE_EXTS | VIDEO_EXTS:
                    if (p.parent / f"{p.stem}{mext}").exists():
                        return None
            return name, arg, target or item.path
        # media / other files: sidecar directive
        p = Path(item.path)
        for ext in (".md", ".txt"):
            sidecar = p.parent / f"{p.stem}{ext}"
            if sidecar.exists():
                try:
                    first_line = sidecar.read_text(
                        encoding="utf-8", errors="replace").splitlines()[0]
                except (OSError, IndexError):
                    continue
                parsed = parse_directive(first_line)
                if parsed:
                    return parsed[0], parsed[1], str(p)
        return None

    def classify(self, item: InboxItem) -> str:
        """Intent for an item. Explicit directives ALWAYS win (spec §2)."""
        directive = self._directive_for(item)
        if directive:
            name, arg, target = directive
            with self.db.transaction():
                self.db.execute(
                    "UPDATE inbox_items SET directive=?, directive_target=?,"
                    " updated_at=? WHERE id=?",
                    (f"@{name} {arg}".strip(), target, time.time(), item.id))
            item.directive = f"@{name} {arg}".strip()
            item.directive_target = target
            intent = DIRECTIVE_INTENTS.get(name)
            if intent is None:
                item.error = f"unknown directive @{name}"
                return "needs_input"
            if name == "room" and arg:
                item.room = sanitize_name(arg.split()[0])
                with self.db.transaction():
                    self.db.execute(
                        "UPDATE inbox_items SET room=?, updated_at=? WHERE id=?",
                        (item.room, time.time(), item.id))
            return intent
        custom = self._classifier(self, item)
        if custom:
            # fixed taxonomy for the default classifier; injected handlers
            # may define their own intents as long as a handler exists
            if custom in INTENTS or custom in self.handlers:
                return custom
            item.error = f"unknown intent {custom!r}"
            return "needs_input"
        return "needs_input"

    # ── processing ─────────────────────────────────────────────────────

    def _claim(self, item_id: str) -> bool:
        """Flip pending→processing; False if another sweeper won the race."""
        with self.db.transaction():
            cur = self.db.execute(
                "UPDATE inbox_items SET status='processing',"
                " attempts=attempts+1, updated_at=? WHERE id=? AND status='pending'",
                (time.time(), item_id))
        return cur.rowcount == 1

    def _processed_dest(self, item: InboxItem) -> Path:
        """Where _finish() will move this item (mirrors its dedup logic)."""
        dest_dir = self._processed / datetime.now().strftime("%Y-%m-%d")
        dest = dest_dir / item.name
        if dest.exists():
            dest = dest_dir / f"{dest.stem}__{new_short_id()}{dest.suffix}"
        return dest

    def _finish(self, item: InboxItem, intent: str, result: ActionResult,
              outcome: str) -> InboxItem:
        now = time.time()
        src = Path(item.path)
        if result.disposition == "processed":
            dest = self._processed_dest(item)
            dest.parent.mkdir(parents=True, exist_ok=True)
        elif result.disposition == "archive":
            dest = self._archive / item.name
        elif result.disposition == "room_files":
            room = result.room or item.room or ""
            rp = self.rooms.files_path(room) if room else None
            dest_dir = rp or self._archive
            dest = dest_dir / item.name
        else:  # stay — parked for input
            dest = None
        new_path = item.path
        if dest is not None and src.exists():
            if dest.exists():
                dest = dest.parent / f"{dest.stem}__{new_short_id()}{dest.suffix}"
            shutil.move(str(src), str(dest))
            new_path = str(dest)
        status = "done" if outcome == "processed" else outcome
        summary = redact_secrets(result.summary)
        with self.db.transaction():
            self.db.execute(
                "UPDATE inbox_items SET status=?, path=?, intent=?,"
                " action_taken=?, result_summary=?, error='', updated_at=?"
                " WHERE id=?",
                (status, new_path, intent, intent, summary, now, item.id))
            self.db.execute(
                "INSERT INTO inbox_history (id, item_id, intent, action,"
                " started_at, ended_at, outcome, detail)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (new_short_id("hx"), item.id, intent, intent,
                 item.updated_at, now, outcome, summary[:2000]))
        item.status, item.path, item.intent = status, new_path, intent
        item.action_taken, item.result_summary = intent, summary
        return item

    def _fail(self, item: InboxItem, intent: str, error: str) -> InboxItem:
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE inbox_items SET status='failed', intent=?, error=?,"
                " updated_at=? WHERE id=?", (intent, error[:2000], now, item.id))
            self.db.execute(
                "INSERT INTO inbox_history (id, item_id, intent, action,"
                " started_at, ended_at, outcome, detail)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (new_short_id("hx"), item.id, intent, intent,
                 item.updated_at, now, "failed", error[:2000]))
        log.error("inbox: item %s (%s) failed: %s", item.id, item.name, error)
        item.status, item.error = "failed", error[:2000]
        return item

    def process(self, item: InboxItem) -> InboxItem:
        """Classify + act on one item. Idempotent via _claim()."""
        if not self._claim(item.id):
            return self.get_item(item.id)  # another sweeper is on it
        started = time.time()
        try:
            intent = self.classify(item)
        except Exception as exc:  # noqa: BLE001 - classification must not kill sweeps
            log.error("inbox: classify failed for %s: %s", item.id, exc)
            return self._fail(item, "", f"classify: {exc}")
        handler = self.handlers.get(intent)
        if handler is None:
            return self._fail(item, intent, f"no handler for intent {intent!r}")
        try:
            result = handler(self, item)
        except Exception as exc:  # noqa: BLE001 - one bad item never kills a sweep
            log.error("inbox: handler %s failed for %s: %s",
                      intent, item.id, exc)
            return self._fail(item, intent, f"{intent}: {exc}")
        outcome = "processed"
        if intent == "needs_input" or result.disposition == "stay":
            outcome = "needs_input"
        item = self._finish(item, intent, result, outcome)
        item._notify_line = redact_secrets(result.notify)  # noqa: SLF001
        log.info("inbox: %s → %s (%s, %.1fs)", item.name, intent,
                 item.status, time.time() - started)
        return item

    def sweep(self) -> dict[str, Any]:
        """One sweep cycle: oldest pending first. Returns a report."""
        if not self._sweep_lock.acquire(blocking=False):
            return {"swept": False, "reason": "sweep already running"}
        try:
            pending = [self._row_to_item(r) for r in self.db.query(
                "SELECT * FROM inbox_items WHERE status='pending'"
                " ORDER BY received_at ASC")]
            # a note that names another pending file must run BEFORE that
            # file's default intent consumes it (e.g. `@square photo.jpg`)
            by_path = {it.path: it for it in pending}
            targeted: set[str] = set()
            for it in pending:
                if it.kind == "note":
                    try:
                        d = self._directive_for(it)
                    except Exception:  # noqa: BLE001 - never break a sweep
                        d = None
                    if d and d[2] and d[2] != it.path and d[2] in by_path:
                        targeted.add(d[2])
            pending.sort(key=lambda it: (it.path in targeted, it.received_at))
            lines: list[str] = []
            counts: dict[str, int] = {}
            intents: dict[str, int] = {}
            for row_item in pending:
                item = self.process(row_item)
                counts[item.status] = counts.get(item.status, 0) + 1
                key = item.intent or item.status
                intents[key] = intents.get(key, 0) + 1
                line = getattr(item, "_notify_line", "")
                if line:
                    lines.append(f"• {item.name}: {line}")
            if len(lines) > 1:
                self._notify(
                    "inbox", f"inbox: {len(lines)} items processed",
                    "\n".join(lines[:10]))
            elif lines:
                title, _, body = lines[0][2:].partition(": ")
                self._notify("inbox", f"inbox: {title}", body)
            return {"swept": True, "pending_found": len(pending),
                    "counts": counts, "intents": intents}
        finally:
            self._sweep_lock.release()

    # ── notifications ──────────────────────────────────────────────────

    def _default_notifier(self) -> Notifier:
        ctx = SimpleNamespace(db=self.db, settings=SimpleNamespace())
        return Notifier(ctx, gateway=None)  # CLI context: persist only

    def _notify(self, kind: str, title: str, body: str = "",
                *, critical: bool = False) -> None:
        notifier = self._notifier or self._default_notifier()
        try:
            notifier.publish(kind, title, redact_secrets(body),
                             critical=critical)
        except Exception as exc:  # noqa: BLE001 - notify never breaks intake
            log.warning("inbox: notify failed (%s): %s", title, exc)


class InboxWatcher:
    """Scheduler-driven sweep (spec §1): every 2 min, idempotent."""

    def __init__(self, inbox: Inbox, interval_s: float = SWEEP_INTERVAL_S) -> None:
        self.inbox = inbox
        self.interval_s = max(10.0, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sweep_now(self) -> dict[str, Any]:
        return self.inbox.sweep()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="inbox-watcher",
                                        daemon=True)
        self._thread.start()
        log.info("inbox watcher started (every %.0fs)", self.interval_s)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.inbox.sweep()
            except Exception as exc:  # noqa: BLE001 - watcher never dies
                log.error("inbox watcher sweep failed: %s", exc)

# ── deterministic fallback classifier ────────────────────────────────────────


def _default_classify(inbox: Inbox, item: InboxItem) -> str | None:
    """Zero-model intent mapping. Returns None when genuinely ambiguous."""
    ext = Path(item.name).suffix.lower()
    if item.kind == "link":
        return "research"
    if ext in IMAGE_EXTS:
        return "image"  # Prompt 09: dropped images go to the vision intent
    if ext in VIDEO_EXTS:
        return "describe"  # Prompt 15: default (no directive) is probe + describe
    if ext in AUDIO_EXTS:
        return "transcribe"
    if ext in TEXT_EXTS or item.kind == "note" \
            or (item.mime or "").startswith("text/"):
        return "summarize"
    return None


# ── handlers ─────────────────────────────────────────────────────────────────

def _handle_summarize(inbox: Inbox, item: InboxItem) -> ActionResult:
    text, is_binary = read_dropped_text(item.path)
    if is_binary or not text.strip():
        summary = (f"binary file ({item.mime or 'unknown type'}, "
                   f"{item.size_bytes} bytes) — no text to summarize")
    else:
        n_lines = text.count("\n") + 1
        summary = f"{n_lines} lines, {item.size_bytes} bytes: " \
                  f"{extractive_summary(text)}"
    summary = redact_secrets(summary)
    return ActionResult(summary=summary,
                        notify=f"summarized → {summary[:140]}",
                        disposition="processed")


def _handle_describe(inbox: Inbox, item: InboxItem) -> ActionResult:
    """Prompt 15 default for media drops: probe + describe."""
    try:
        from ..media_edit import images as _images
        from ..media_edit import videos as _videos
    except ImportError:
        return ActionResult(
            summary="media probe unavailable (Pillow/ffmpeg not installed)",
            notify="media tools missing — install Pillow + ffmpeg to probe",
            disposition="processed")
    ext = Path(item.name).suffix.lower()
    try:
        if ext in VIDEO_EXTS:
            info = _videos.video_probe(item.path)
            dur = info.get("duration") or 0
            detail = (f"{info.get('width')}x{info.get('height')}, "
                      f"{dur:.1f}s, {info.get('video_codec') or '?'}")
        else:
            info = _images.image_probe(item.path)
            detail = (f"{info.get('width')}x{info.get('height')} "
                      f"{info.get('format') or ''}".strip())
    except Exception as exc:  # noqa: BLE001 - unreadable media isn't fatal
        detail = f"unreadable: {exc}"
    summary = f"{item.mime or 'media'}: {detail}"
    return ActionResult(summary=summary, notify=f"probed → {detail}",
                        disposition="processed")


def _handle_file(inbox: Inbox, item: InboxItem) -> ActionResult:
    room = item.room or None
    where = f"room '{room}' files" if room else "archive"
    return ActionResult(summary=f"filed to {where}",
                        notify=f"filed → {where}",
                        disposition="room_files" if room else "archive",
                        room=room)


def _schedule_reminder(inbox: Inbox, text: str, due: float,
                       item: InboxItem) -> Any:
    """Create a scheduler reminder; works with the real Scheduler or a fake."""
    sched = inbox._scheduler  # noqa: SLF001
    if sched is None:
        from ..scheduler.scheduler import Scheduler
        sched = Scheduler(inbox.db)
    res = sched.create_reminder(
        text=text, due_at=due, user_id="owner",
        parameters={"inbox_item": item.id, "item_path": item.path,
                    "item_name": item.name})
    if inspect.isawaitable(res):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            loop.create_task(res)  # fire-and-forget inside a live agent
            return {"scheduled": "background"}
        return asyncio.run(res)
    return res


def _handle_remind(inbox: Inbox, item: InboxItem) -> ActionResult:
    if item.directive.startswith("@remind"):
        when_text = item.directive[len("@remind"):].strip()
        what = f"inbox note {item.name}"
    else:
        try:
            body = Path(item.path).read_text(
                encoding="utf-8", errors="replace")[:2000]
        except OSError:
            body = ""
        m = re.search(r"remind me (?:about|to) (.+)", body, re.I | re.S)
        when_text = m.group(1).strip() if m else ""
        what = (m.group(0).strip()[:120] if m else item.name)
    due = parse_when(when_text)
    if due is None:
        item.error = f"couldn't parse a time from {when_text!r}"
        return ActionResult(
            summary="reminder needs a time",
            notify="when should I remind you? e.g. `@remind friday 9am`",
            disposition="stay")
    try:
        _schedule_reminder(inbox, f"{what} (from inbox: {item.name})",
                           due, item)
    except Exception as exc:  # noqa: BLE001 - scheduler failure isn't fatal
        item.error = f"scheduler: {exc}"
        return ActionResult(summary=f"reminder failed: {exc}",
                            notify="couldn't set that reminder — try again?",
                            disposition="stay")
    when = datetime.fromtimestamp(due).strftime("%a %Y-%m-%d %H:%M")
    return ActionResult(summary=f"reminder set for {when}: {what}",
                        notify=f"reminder set → {when}",
                        disposition="processed")


def _handle_watch(inbox: Inbox, item: InboxItem) -> ActionResult:
    if item.kind == "link":
        try:
            target = Path(item.path).read_text(
                encoding="utf-8", errors="replace").splitlines()[0].strip()
        except OSError:
            target = ""
    else:
        target = str(inbox._processed_dest(item))  # noqa: SLF001
    kind = "url" if target.startswith(("http://", "https://")) else "file"
    try:
        from ..agents.monitor import MonitorAgent
        mon = MonitorAgent(SimpleNamespace(db=inbox.db), notifier=None)
        res = mon.add(target, kind=kind)
        wid = res.get("id", "?")
    except Exception as exc:  # noqa: BLE001 - degraded mode → ask, don't fake
        item.error = f"watch backend unavailable: {exc}"
        return ActionResult(
            summary="watch backend unavailable",
            notify="watch backend isn't available in this build — "
                   "want a reminder to check it manually instead?",
            disposition="stay")
    return ActionResult(summary=f"watch {wid} on {target}",
                        notify=f"watching → {target}",
                        disposition="processed")


def _handle_research(inbox: Inbox, item: InboxItem) -> ActionResult:
    if item.kind != "link":
        item.error = "research needs a link"
        return ActionResult(
            summary="research needs a link",
            notify="drop a link and I'll research it — what URL?",
            disposition="stay")
    try:
        url = Path(item.path).read_text(
            encoding="utf-8", errors="replace").splitlines()[0].strip()
    except OSError:
        url = ""
    if not url.startswith(("http://", "https://")):
        item.error = "link file has no URL"
        return ActionResult(
            summary="link file empty",
            notify="that link file looks empty — what URL did you mean?",
            disposition="stay")
    try:
        text = redact_secrets(inbox._fetch(url))  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - fetch failure parks, doesn't crash
        item.error = f"fetch failed: {exc}"
        return ActionResult(
            summary=f"fetch failed: {exc}",
            notify="couldn't fetch that link — still want me to dig in?",
            disposition="stay")
    brief = extractive_summary(text, 800) or "no readable text found"
    return ActionResult(summary=f"researched {url}: {brief}",
                        notify=f"researched → {brief[:140]}",
                        disposition="processed")


def _handle_transcribe(inbox: Inbox, item: InboxItem) -> ActionResult:
    # Spec §2: only if transcription tooling exists; otherwise needs_input.
    item.error = "no transcription backend configured in this build"
    return ActionResult(
        summary="transcription unavailable",
        notify="no transcription backend in this build — "
               "want a file summary instead?",
        disposition="stay")


def _handle_needs_input(inbox: Inbox, item: InboxItem) -> ActionResult:
    # Exactly ONE question: the item leaves `pending`, so a later sweep
    # never re-asks. It sits in `needs_input` until retry/release.
    question = item.error or f"what should I do with {item.name}?"
    return ActionResult(summary=f"waiting on owner: {question}",
                        notify=question, disposition="stay")


# ── media directives (Prompt 15 loop-closer) ─────────────────────────────────


def _run_media_edit(src: Path, instruction: str, out_dir: Path) -> dict[str, Any]:
    """Run a media edit the same way the media_edit tools do (engine-direct;
    the inbox already sandboxes paths to the workspace)."""
    from ..media_edit import intent as _intent
    from ..media_edit import images as _images
    from ..media_edit import videos as _videos
    ext = src.suffix.lower()
    kind = "video" if ext in VIDEO_EXTS else "image"
    plan = _intent.parse_instruction(instruction, kind=kind)
    if kind == "image":
        return _images.edit_image(src, plan.ops, out_dir=out_dir)
    action = plan.action
    name = action.get("video_op")
    kw = {k: v for k, v in action.items() if k != "video_op"}
    kw.setdefault("out_dir", out_dir)
    if name == "trim":
        return _videos.trim(src, kw.pop("start", 0), kw.pop("end", None), **kw)
    if name == "extract_audio":
        return _videos.extract_audio(src, **kw)
    if name == "make_gif":
        return _videos.make_gif(src, **kw)
    if name == "extract_frames":
        return _videos.extract_frames(src, **kw)
    if name == "transcode":
        return _videos.transcode(src, **kw)
    raise ValueError(f"unsupported video op {name!r}")


def _handle_media(inbox: Inbox, item: InboxItem, instruction: str) -> ActionResult:
    target = Path(item.directive_target or item.path)
    if not target.exists():
        item.error = f"directive target missing: {target.name}"
        return ActionResult(
            summary="directive target missing",
            notify=f"couldn't find {target.name} — re-drop it?",
            disposition="stay")
    out_dir = inbox._processed / datetime.now().strftime("%Y-%m-%d")  # noqa: SLF001
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_media_edit(target, instruction, out_dir)
    except Exception as exc:  # noqa: BLE001 - a failed edit parks, doesn't crash
        item.error = str(exc)[:500]
        return ActionResult(
            summary=f"media edit failed: {exc}",
            notify="that edit failed — want me to try it differently?",
            disposition="stay")
    artifact = Path(result.get("output", "?")).name
    # if a note directed this at another pending inbox file, don't process
    # that file a second time — mark it done as handled-via-note.
    if target != Path(item.path):
        row = inbox.db.query_one(
            "SELECT id FROM inbox_items WHERE path=? AND status='pending'",
            (str(target),))
        if row:
            with inbox.db.transaction():
                inbox.db.execute(
                    "UPDATE inbox_items SET status='done', intent=?,"
                    " action_taken=?, result_summary=?, updated_at=?"
                    " WHERE id=?",
                    (item.intent, item.intent,
                     f"edited via {item.directive} from {item.name}",
                     time.time(), row["id"]))
    # consume sidecar notes (<stem>.md/.txt) so they aren't swept separately
    for ext in (".md", ".txt"):
        sc = target.parent / f"{target.stem}{ext}"
        if sc == Path(item.path):
            continue
        row = inbox.db.query_one(
            "SELECT id FROM inbox_items WHERE path=? AND status='pending'",
            (str(sc),))
        if row:
            with inbox.db.transaction():
                inbox.db.execute(
                    "UPDATE inbox_items SET status='done', intent='consumed',"
                    " action_taken='consumed', result_summary=?, updated_at=?"
                    " WHERE id=?",
                    (f"sidecar for {item.directive}", time.time(), row["id"]))
    return ActionResult(summary=f"edited → {artifact} ({instruction})",
                        notify=f"edited → {artifact}",
                        disposition="processed")


def _handle_media_trim(inbox: Inbox, item: InboxItem) -> ActionResult:
    arg = item.directive[len("@trim"):].strip() \
        if item.directive.startswith("@trim") else ""
    if "-" in arg:
        start, end = arg.split("-", 1)
        instruction = f"trim from {start.strip()} to {end.strip()}"
    elif arg:
        instruction = f"trim {arg}"
    else:
        instruction = "trim the first 30 seconds"
    return _handle_media(inbox, item, instruction)


def _handle_image(inbox: Inbox, item: InboxItem) -> ActionResult:
    """Prompt 09 image intent: probe + vision, directive-aware.

    Default (no directive) is ``describe`` + file. ``@read-text`` transcribes,
    ``@locate <target>`` returns an approximate region. Without a vision hook
    the handler degrades to probe-only and says so — it never pretends to
    have seen the image.
    """
    try:
        from ..media_edit import images as _images
        info = _images.image_probe(item.path)
        detail = (f"{info.get('width')}x{info.get('height')} "
                  f"{info.get('format') or ''}".strip())
    except Exception as exc:  # noqa: BLE001 - unreadable media isn't fatal
        detail = f"unreadable: {exc}"

    action, prompt = "describe", ""
    if item.directive:
        parsed = parse_directive(item.directive)
        if parsed:
            name, arg = parsed
            if name == "read-text":
                action = "read_text"
            elif name == "locate":
                action, prompt = "locate", arg

    hook = getattr(inbox, "_vision", None)
    if hook is None:
        summary = (f"{item.mime or 'image'}: {detail} — vision not wired, "
                   "probe only")
        return ActionResult(summary=summary, notify=f"image → {detail}",
                            disposition="processed")
    try:
        data = Path(item.path).read_bytes()
    except OSError as exc:
        item.error = f"can't read image: {exc}"
        return ActionResult(summary=item.error, notify=item.error,
                            disposition="failed")
    try:
        seen = hook(data, action, prompt)
    except Exception as exc:  # noqa: BLE001 - vision failure parks, doesn't crash
        item.error = f"vision {action} failed: {exc}"
        return ActionResult(summary=item.error,
                            notify="vision hiccup — I'll retry next sweep",
                            disposition="stay")
    if action == "read_text":
        seen_text = (seen.get("text") or "").strip()
        summary = f"read-text {item.name}: {seen_text[:400]}"
        notify = f"transcribed → {seen_text[:140]}"
    elif action == "locate":
        if seen.get("found"):
            summary = (f"locate '{prompt}' in {item.name}: "
                       f"{seen.get('x')},{seen.get('y')} "
                       f"{seen.get('w')}x{seen.get('h')} "
                       f"(confidence {seen.get('confidence')}) — approximate")
        else:
            summary = f"locate '{prompt}' in {item.name}: not found"
        notify = summary[:140]
    else:
        desc = (seen.get("description") or "").strip()
        summary = f"image {item.name} ({detail}): {desc[:400]}"
        notify = f"described → {desc[:140]}"
    return ActionResult(summary=redact_secrets(summary),
                        notify=redact_secrets(notify),
                        disposition="processed")


def _default_handlers() -> dict[str, Callable[[Inbox, InboxItem], ActionResult]]:
    return {
        "summarize": _handle_summarize,
        "describe": _handle_describe,
        "image": _handle_image,
        "file": _handle_file,
        "remind": _handle_remind,
        "watch": _handle_watch,
        "research": _handle_research,
        "transcribe": _handle_transcribe,
        "media_square": lambda i, it: _handle_media(
            i, it, "make it square for instagram"),
        "media_trim": _handle_media_trim,
        "media_gif": lambda i, it: _handle_media(i, it, "make a gif"),
        "needs_input": _handle_needs_input,
    }
