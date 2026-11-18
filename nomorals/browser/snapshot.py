"""Page snapshots (52-websites #27, Archive.ph): her own receipts.

Devon reads the web constantly but kept text and citations, not
receipts — no organ captured a page *as it rendered* into her own
store with provenance. This one does: render via her browser
machinery (playwright-backed rendered tab when it exists, plain
HTTP as an honest labelled fallback), save the rendered DOM + an
asset bundle + a screenshot, and record a ``Snapshot`` carrying
the capture-time hash + timestamp + source URL in her journal.

Trust comes from the capture being hers: ``content_hash`` is the
sha256 of the stored DOM bytes, ``render_hash`` covers the DOM +
screenshot together, and :func:`verify` re-hashes the files on
disk and says whether the receipt still holds. A snapshot proves
what *her browser was served* — personalised or geo-varied pages
snapshot as her vantage saw them, and the record says so.

Seed sources: pages she did not capture herself can be fetched
from the Wayback Machine (§8) or archive.ph and recorded as
``kind="reference"`` — reference copies, never sole evidence (the
record's standing conclusion: third-party archives can no longer
guarantee their snapshots, and English Wikipedia has banned
archive.today links).

Retention is the owner's call: the store keeps ``retention_days``
(default 90); :func:`prune` and the daily prune tick delete
snapshots older than that, files and record together, and report
what went.
"""

from __future__ import annotations

import contextlib
import hashlib
import html as html_mod
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..core.ids import ulid_now
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Snapshot",
    "SnapshotStore",
    "BrowserRenderer",
    "capture",
    "seed",
    "verify",
    "prune",
    "set_retention",
    "format_snapshot_reply",
    "cite_line",
    "snapshot_text",
    "parse_snapshot_request",
    "ensure_snapshot_prune_job",
    "SNAPSHOT_PRUNE_JOB",
    "SNAPSHOT_PRUNE_SPEC",
    "DEFAULT_RETENTION_DAYS",
]

STORE_FILENAME = "snapshots.json"
SNAPSHOT_PRUNE_JOB = "snapshot prune tick"
SNAPSHOT_PRUNE_SPEC = "0 3 * * *"
DEFAULT_RETENTION_DAYS = 90

MAX_ASSETS = 40
MAX_ASSET_BYTES = 2_000_000
MAX_ASSET_TOTAL_BYTES = 20_000_000
FETCH_TIMEOUT_S = 20.0
ASSET_TIMEOUT_S = 10.0

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# The vantage label every snapshot carries: a snapshot proves what
# her browser was served on this host, nothing more.
VANTAGE = "devon-browser (this host)"


def _default_home(home: str | Path | None) -> Path:
    if home:
        return Path(home)
    return Path("~/.nomorals").expanduser()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)


def _http_get(url: str, *, timeout: float = FETCH_TIMEOUT_S) -> bytes:
    """One plain GET with a browser UA. Raises on any failure."""
    request = urllib.request.Request(
        url, headers={"User-Agent": _BROWSER_UA,
                      "Accept": "text/html,*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# ── shapes ────────────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """One captured page — her receipt, with its hashes."""

    id: str
    url: str
    captured_at: float
    captured_iso: str
    title: str
    kind: str                       # capture | reference
    source: str                     # devon-browser | wayback | archive.ph
    rendered: bool
    render_reason: str              # "" when rendered; else why not
    content_hash: str               # sha256 of the stored DOM bytes
    render_hash: str                # sha256 of DOM bytes + screenshot bytes
    dom_path: str
    screenshot_path: str            # "" when no screenshot was taken
    assets: list[dict[str, Any]] = field(default_factory=list)
    vantage: str = VANTAGE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        return cls(
            id=str(data.get("id", "")),
            url=str(data.get("url", "")),
            captured_at=float(data.get("captured_at", 0.0)),
            captured_iso=str(data.get("captured_iso", "")),
            title=str(data.get("title", "")),
            kind=str(data.get("kind", "capture")),
            source=str(data.get("source", "devon-browser")),
            rendered=bool(data.get("rendered", False)),
            render_reason=str(data.get("render_reason", "")),
            content_hash=str(data.get("content_hash", "")),
            render_hash=str(data.get("render_hash", "")),
            dom_path=str(data.get("dom_path", "")),
            screenshot_path=str(data.get("screenshot_path", "")),
            assets=list(data.get("assets") or []),
            vantage=str(data.get("vantage", VANTAGE)),
        )


class SnapshotStore:
    """The snapshot journal: ``<home>/data/snapshots.json`` + files
    under ``<home>/snapshots/<id>/``.

    Atomic writes; a corrupt index is quarantined aside
    (``snapshots.json.corrupt-<ts>``) and the store starts empty
    rather than dying — the files on disk survive, the index is
    the only thing rebuilt.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        base = _default_home(home)
        self.path = base / "data" / STORE_FILENAME
        self.dir = base / "snapshots"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, Snapshot] = {}
        self.retention_days = DEFAULT_RETENTION_DAYS
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            stamp = int(time.time())
            aside = self.path.with_suffix(f".corrupt-{stamp}")
            with contextlib.suppress(OSError):
                os.replace(self.path, aside)
            _log.warning("snapshot index corrupt, quarantined aside: %s",
                         exc)
            return
        if not isinstance(payload, dict):
            return
        try:
            self.retention_days = int(
                payload.get("retention_days", DEFAULT_RETENTION_DAYS))
        except (TypeError, ValueError):
            self.retention_days = DEFAULT_RETENTION_DAYS
        for raw in payload.get("snapshots", []) or []:
            try:
                snap = Snapshot.from_dict(raw)
            except (TypeError, ValueError):
                continue
            if snap.id:
                self._records[snap.id] = snap

    def _save(self) -> None:
        payload = {
            "retention_days": self.retention_days,
            "snapshots": [snap.to_dict()
                          for snap in self._records.values()],
        }
        _atomic_write_json(self.path, payload)

    # -- records ----------------------------------------------------------

    def add(self, snap: Snapshot) -> Snapshot:
        self._records[snap.id] = snap
        self._save()
        return snap

    def get(self, snapshot_id: str) -> Snapshot | None:
        return self._records.get((snapshot_id or "").strip())

    def find_by_url(self, url: str) -> Snapshot | None:
        want = (url or "").strip()
        for snap in self._records.values():
            if snap.url == want:
                return snap
        return None

    def list(self) -> list[Snapshot]:
        return sorted(self._records.values(),
                      key=lambda snap: snap.captured_at, reverse=True)

    def remove(self, snapshot_id: str) -> Snapshot | None:
        snap = self._records.pop((snapshot_id or "").strip(), None)
        if snap is not None:
            self._save()
        return snap

    def snapshot_dir(self, snapshot_id: str) -> Path:
        return self.dir / snapshot_id


# ── DOM helpers ───────────────────────────────────────────────────────────


class _AssetParser(HTMLParser):
    """Collect asset URLs: img src, script src, stylesheet href."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        mapping = dict(attrs)
        if tag in ("img", "script") and mapping.get("src"):
            self.urls.append(str(mapping["src"]))
        elif (tag == "link"
              and (mapping.get("rel") or "").lower() == "stylesheet"
              and mapping.get("href")):
            self.urls.append(str(mapping["href"]))


class _TextParser(HTMLParser):
    """Plain text out of stored DOM, for citation registration."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _extract_asset_urls(dom: str, base_url: str) -> list[str]:
    parser = _AssetParser()
    try:
        parser.feed(dom)
    except Exception:  # noqa: BLE001 - a hostile DOM never kills capture
        return []
    seen: list[str] = []
    for raw in parser.urls:
        resolved = urllib.parse.urljoin(base_url, raw.strip())
        if resolved.startswith(("http://", "https://")) \
                and resolved not in seen:
            seen.append(resolved)
    return seen[:MAX_ASSETS]


def snapshot_text(dom: str) -> str:
    """Readable text from stored DOM (citation source text)."""
    parser = _TextParser()
    try:
        parser.feed(dom)
    except Exception:  # noqa: BLE001 - a hostile DOM never kills capture
        return ""
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return html_mod.unescape(text)


def _safe_asset_name(index: int, url: str) -> str:
    base = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1] or "asset"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:60] or "asset"
    return f"{index:02d}_{base}"


def _fetch_assets(urls: list[str], dest_dir: Path, *,
                  fetcher: Any = None) -> tuple[list[dict[str, Any]], int]:
    """Fetch the asset bundle. Every failure is recorded, never fatal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []
    total = 0
    for index, url in enumerate(urls):
        entry: dict[str, Any] = {"url": url, "ok": False,
                                 "path": "", "bytes": 0,
                                 "sha256": "", "error": ""}
        try:
            data = (fetcher(url) if fetcher is not None
                    else _http_get(url, timeout=ASSET_TIMEOUT_S))
            if not isinstance(data, bytes):
                data = str(data).encode("utf-8", "replace")
            if len(data) > MAX_ASSET_BYTES:
                entry["error"] = (f"asset over {MAX_ASSET_BYTES} bytes, "
                                  "skipped")
            elif total + len(data) > MAX_ASSET_TOTAL_BYTES:
                entry["error"] = "asset bundle budget reached, skipped"
            else:
                name = _safe_asset_name(index, url)
                path = dest_dir / name
                path.write_bytes(data)
                entry.update({"ok": True, "path": str(path),
                              "bytes": len(data),
                              "sha256": _sha256_bytes(data)})
                total += len(data)
        except Exception as exc:  # noqa: BLE001 - one dead asset is a row
            entry["error"] = str(exc)[:200]
        assets.append(entry)
    return assets, total


# ── rendering ─────────────────────────────────────────────────────────────


class BrowserRenderer:
    """The live renderer: a playwright-backed rendered tab from her
    own browser service — the same session space her other browsing
    uses, so logged-in state and JS mutations are captured.

    Raises on any failure; :func:`capture` treats that as the raw
    fallback, never as a dead capture.
    """

    def __init__(self, *, data_dir: str | Path | None = None) -> None:
        self._data_dir = data_dir
        self._svc: Any = None

    def _service(self) -> Any:
        if self._svc is not None:
            return self._svc
        from .service import BrowserService

        kwargs: dict[str, Any] = {}
        if self._data_dir is not None:
            kwargs["data_dir"] = self._data_dir
        self._svc = BrowserService(**kwargs)
        return self._svc

    def render(self, url: str, *, screenshot_path: str | Path
               ) -> dict[str, Any]:
        """Render ``url``; returns {dom, title, screenshot_path}."""
        svc = self._service()
        tab = svc.open_rendered_tab("snapshot", url)
        try:
            rendered = tab.html()
            dom = str(rendered.get("html", ""))
            shot = tab.screenshot(path=str(screenshot_path))
            return {"dom": dom,
                    "title": str(rendered.get("title", "")),
                    "screenshot_path": str(shot.get("path", ""))}
        finally:
            # Tab cleanup never kills the capture.
            with contextlib.suppress(Exception):
                svc.close_rendered_tab(tab.tab_id)


def _default_renderer() -> Any | None:
    """A live renderer when playwright is installed, else None —
    the caller falls back to raw HTTP and says so."""
    import importlib.util

    if importlib.util.find_spec("playwright") is None:
        return None
    try:
        return BrowserRenderer()
    except Exception as exc:  # noqa: BLE001 - renderer build never kills capture
        _log.debug("snapshot renderer unavailable: %s", exc)
        return None


def _validate_url(url: str) -> str:
    cleaned = (url or "").strip().strip("<>")
    if not cleaned.startswith(("http://", "https://")):
        raise ValueError("snapshot needs an http(s) URL — "
                         f"got {cleaned[:80]!r}")
    return cleaned


def _captured_iso(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(ts))


# ── capture ───────────────────────────────────────────────────────────────


def capture(url: str, *, store: SnapshotStore | None = None,
            home: str | Path | None = None,
            renderer: Any = None, fetcher: Any = None) -> Snapshot:
    """Capture ``url`` as it renders, into the store.

    ``renderer`` is the injected render rung (an object with
    ``render(url, *, screenshot_path)`` returning
    ``{dom, title, screenshot_path}``); None builds the live
    playwright renderer when it exists. Any render failure falls
    through to a plain HTTP fetch, labelled ``rendered=False``
    with the reason named — a receipt that says what it is, never
    a rendered claim on raw HTML.
    """
    target = _validate_url(url)
    own_store = store if store is not None else SnapshotStore(home)
    snap_id = ulid_now()
    snap_dir = own_store.snapshot_dir(snap_id)
    snap_dir.mkdir(parents=True, exist_ok=True)
    dom_path = snap_dir / "dom.html"
    shot_path = snap_dir / "screenshot.png"

    dom = ""
    title = ""
    rendered = False
    render_reason = ""
    screenshot_file = ""

    rung = renderer if renderer is not None else _default_renderer()
    if rung is not None:
        try:
            result = rung.render(target, screenshot_path=str(shot_path))
            dom = str(result.get("dom", ""))
            title = str(result.get("title", ""))
            if result.get("screenshot_path") and Path(
                    str(result["screenshot_path"])).is_file():
                screenshot_file = str(result["screenshot_path"])
            rendered = True
        except Exception as exc:  # noqa: BLE001 - render failure is a fallback
            render_reason = f"render failed ({exc}); raw HTML instead"
            _log.debug("snapshot render fell through for %s: %s",
                       target, exc)
    if not rendered and not dom:
        if rung is None and not render_reason:
            render_reason = ("playwright not installed — raw HTML only, "
                             "no screenshot")
        try:
            raw = (fetcher(target) if fetcher is not None
                   else _http_get(target))
            dom = raw.decode("utf-8", "replace") if isinstance(
                raw, bytes) else str(raw)
        except Exception as exc:  # noqa: BLE001 - the failure is the answer
            raise ValueError(f"could not fetch {target}: {exc}") from exc
    if not dom:
        raise ValueError(f"capture of {target} returned no content")

    dom_bytes = dom.encode("utf-8")
    dom_path.write_bytes(dom_bytes)
    content_hash = _sha256_bytes(dom_bytes)
    if screenshot_file:
        render_hash = _sha256_bytes(
            dom_bytes + Path(screenshot_file).read_bytes())
    else:
        render_hash = content_hash

    assets, _total = _fetch_assets(
        _extract_asset_urls(dom, target), snap_dir / "assets",
        fetcher=fetcher)
    now = time.time()
    snap = Snapshot(
        id=snap_id, url=target, captured_at=now,
        captured_iso=_captured_iso(now), title=title,
        kind="capture", source="devon-browser", rendered=rendered,
        render_reason="" if rendered else render_reason,
        content_hash=content_hash, render_hash=render_hash,
        dom_path=str(dom_path),
        screenshot_path=screenshot_file,
        assets=assets, vantage=VANTAGE)
    return own_store.add(snap)


# ── seed sources ──────────────────────────────────────────────────────────


def _wayback_fetch(url: str, *, fetcher: Any = None) -> tuple[str, str]:
    """Closest Wayback snapshot: (snapshot_url, html)."""
    get = fetcher if fetcher is not None else _http_get
    raw = get("https://archive.org/wayback/available?url="
              + urllib.parse.quote(url, safe=""))
    data = json.loads(raw.decode("utf-8", "replace")
                      if isinstance(raw, bytes) else str(raw))
    snap = (data.get("archived_snapshots") or {}).get("closest", {})
    snap_url = str(snap.get("url", ""))
    if not snap_url:
        raise ValueError(f"Wayback holds no snapshot of {url}")
    page = get(snap_url)
    html = page.decode("utf-8", "replace") if isinstance(
        page, bytes) else str(page)
    return snap_url, html


def _archiveph_fetch(url: str, *, fetcher: Any = None) -> tuple[str, str]:
    """Newest archive.ph copy: (archive_url, html)."""
    get = fetcher if fetcher is not None else _http_get
    archive_url = ("https://archive.ph/newest/"
                   + urllib.parse.quote(url, safe=""))
    page = get(archive_url)
    html = page.decode("utf-8", "replace") if isinstance(
        page, bytes) else str(page)
    if not html.strip():
        raise ValueError(f"archive.ph returned nothing for {url}")
    return archive_url, html


def seed(url: str, *, source: str, store: SnapshotStore | None = None,
         home: str | Path | None = None, fetcher: Any = None) -> Snapshot:
    """Record a page Devon did not capture herself, from a seed
    source — Wayback or archive.ph — as a reference copy.

    The record is labelled ``kind="reference"`` with the source
    named: a reference copy is never sole evidence, and a seeded
    snapshot never claims to be her own capture.
    """
    target = _validate_url(url)
    origin = (source or "").strip().lower()
    if origin not in ("wayback", "archive.ph"):
        raise ValueError("seed source is wayback or archive.ph — "
                         f"got {source!r}")
    if origin == "wayback":
        archive_url, html = _wayback_fetch(target, fetcher=fetcher)
    else:
        archive_url, html = _archiveph_fetch(target, fetcher=fetcher)
    own_store = store if store is not None else SnapshotStore(home)
    snap_id = ulid_now()
    snap_dir = own_store.snapshot_dir(snap_id)
    snap_dir.mkdir(parents=True, exist_ok=True)
    dom_path = snap_dir / "dom.html"
    dom_bytes = html.encode("utf-8")
    dom_path.write_bytes(dom_bytes)
    assets, _total = _fetch_assets(
        _extract_asset_urls(html, target), snap_dir / "assets",
        fetcher=fetcher)
    now = time.time()
    snap = Snapshot(
        id=snap_id, url=target, captured_at=now,
        captured_iso=_captured_iso(now), title="",
        kind="reference", source=origin, rendered=False,
        render_reason=(f"reference copy from {origin} "
                       f"({archive_url}) — never sole evidence"),
        content_hash=_sha256_bytes(dom_bytes),
        render_hash=_sha256_bytes(dom_bytes),
        dom_path=str(dom_path), screenshot_path="",
        assets=assets, vantage=f"{origin} (third-party archive)")
    return own_store.add(snap)


# ── verify / prune / retention ────────────────────────────────────────────


def verify(store: SnapshotStore, snapshot_id: str) -> dict[str, Any]:
    """Re-hash the stored files and say whether the receipt holds."""
    snap = store.get(snapshot_id)
    if snap is None:
        return {"ok": False, "error": f"no snapshot {snapshot_id!r}"}
    dom_path = Path(snap.dom_path)
    if not dom_path.is_file():
        return {"ok": False, "id": snap.id, "url": snap.url,
                "error": "stored DOM is gone from disk"}
    actual_content = _sha256_file(dom_path)
    if actual_content != snap.content_hash:
        return {"ok": False, "id": snap.id, "url": snap.url,
                "error": "stored DOM no longer matches its recorded hash"}
    if snap.screenshot_path:
        shot_path = Path(snap.screenshot_path)
        if not shot_path.is_file():
            return {"ok": False, "id": snap.id, "url": snap.url,
                    "error": "stored screenshot is gone from disk"}
        actual_render = _sha256_bytes(
            dom_path.read_bytes() + shot_path.read_bytes())
        if actual_render != snap.render_hash:
            return {"ok": False, "id": snap.id, "url": snap.url,
                    "error": "stored files no longer match the render hash"}
    return {"ok": True, "id": snap.id, "url": snap.url,
            "captured_iso": snap.captured_iso,
            "content_hash": snap.content_hash,
            "render_hash": snap.render_hash,
            "rendered": snap.rendered, "kind": snap.kind,
            "source": snap.source}


def prune(store: SnapshotStore, *,
          older_than_days: int | None = None) -> dict[str, Any]:
    """Delete snapshots older than the retention window — files and
    record together — and report what went."""
    days = (store.retention_days if older_than_days is None
            else int(older_than_days))
    cutoff = time.time() - max(days, 0) * 86400
    removed: list[str] = []
    for snap in store.list():
        if snap.captured_at >= cutoff:
            continue
        store.remove(snap.id)
        snap_dir = store.snapshot_dir(snap.id)
        if snap_dir.is_dir():
            for path in sorted(snap_dir.rglob("*"), reverse=True):
                with contextlib.suppress(OSError):
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
            with contextlib.suppress(OSError):
                snap_dir.rmdir()
        removed.append(snap.id)
    return {"ok": True, "removed": removed, "removed_count": len(removed),
            "kept": len(store.list()), "retention_days": days}


def set_retention(store: SnapshotStore, days: int) -> dict[str, Any]:
    """The owner's call: how long snapshots are kept (days)."""
    if int(days) < 1:
        raise ValueError("retention is at least 1 day")
    store.retention_days = int(days)
    store._save()
    return {"ok": True, "retention_days": store.retention_days}


def ensure_snapshot_prune_job(scheduler: Any, *,
                              spec: str = SNAPSHOT_PRUNE_SPEC
                              ) -> Any | None:
    """Register the daily snapshot-prune job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
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
            if name != SNAPSHOT_PRUNE_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            SNAPSHOT_PRUNE_JOB, spec, "tool",
            {"tool": "__snapshot_prune_tick__", "args": {}}, routine=True)
    except Exception:  # noqa: BLE001 - job registration is best effort
        _log.debug("snapshot prune job not registered", exc_info=True)
        return None


# ── replies ───────────────────────────────────────────────────────────────


def _short_hash(value: str) -> str:
    return value[:16] + "…" if len(value) > 16 else value


def format_snapshot_reply(snap: Snapshot) -> str:
    """The chat receipt: what was captured, when, and its hashes."""
    lines = [f"snapshot saved — {snap.title or snap.url}",
             f"as captured on {snap.captured_iso} "
             f"({snap.vantage})",
             f"content hash: {_short_hash(snap.content_hash)}",
             f"render hash: {_short_hash(snap.render_hash)}"]
    if snap.kind == "reference":
        lines.append(f"reference copy from {snap.source} — "
                     "never sole evidence")
    elif snap.rendered:
        lines.append("rendered: yes (JavaScript applied)")
    else:
        lines.append(f"rendered: no — {snap.render_reason}")
    kept = sum(1 for asset in snap.assets if asset.get("ok"))
    missing = sum(1 for asset in snap.assets if not asset.get("ok"))
    lines.append(f"files: DOM + "
                 f"{'screenshot + ' if snap.screenshot_path else ''}"
                 f"{kept} assets"
                 + (f" ({missing} missing)" if missing else ""))
    lines.append(f"snapshot id: {snap.id}")
    return "\n".join(lines)


def cite_line(snap: Snapshot) -> str:
    """The research citation: \"as captured on <date>\"."""
    date = snap.captured_iso.split(" ")[0] if snap.captured_iso else "?"
    return (f"as captured on {date} — {snap.url} "
            f"(snapshot {snap.id}, "
            f"content {_short_hash(snap.content_hash)})")


_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def parse_snapshot_request(text: str) -> dict[str, Any] | None:
    """\"snapshot this page <url>\" / \"snapshot <url>\" / \"take a
    snapshot of <url>\" → {url} — else None (not a snapshot ask)."""
    cleaned = (text or "").strip()
    if not cleaned or cleaned.startswith("/"):
        return None
    lowered = cleaned.lower()
    if not re.search(r"\bsnapshot\b", lowered):
        return None
    match = _URL_RE.search(cleaned)
    if match is None:
        return None
    return {"url": match.group(0).rstrip(".,;:!?")}
