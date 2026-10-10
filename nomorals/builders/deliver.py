"""Build -> zip -> deliver: package a project as a ``.zip`` and send it to chat.

:func:`zip_project` builds a ``<name>.zip`` from a project directory with
the same exclusions as :mod:`nomorals.builders.export` (``.git``,
``__pycache__``, ``.venv``, ``node_modules``, ...).  :func:`deliver_project`
zips and then sends the archive through a chat gateway's ``send_file``
path (e.g. :meth:`nomorals.social.chat.gateway.ChatGateway.send_file`).
:func:`build_zip_and_deliver` runs the whole pipeline end to end --
the :func:`nomorals.builders.verify.build_and_verify` lifecycle
(scaffold, install, tests, smoke, export), then zip, then deliver --
capturing every step in a :class:`DeliveryReport` instead of raising.

The gateway is duck-typed (only ``send_file`` is used) so this module
never imports the social layer.  Pass the gateway explicitly, or pass an
agent context and it is resolved from ``context.extras["gateway"]`` --
the runtime's canonical location (``context.gateway`` does not exist).
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from .export import EXCLUDE_DIRS
from .verify import BuildStep, build_and_verify

_log = get_logger(__name__)

__all__ = [
    "ZipResult", "DeliverResult", "DeliveryReport",
    "zip_project", "deliver_project", "build_zip_and_deliver",
]


def _excluded(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(
        part in EXCLUDE_DIRS or part.endswith(".egg-info")
        for part in parts[:-1]
    )


@dataclass
class ZipResult:
    """Outcome of :func:`zip_project`."""

    archive: Path
    files: list[str] = field(default_factory=list)
    bytes: int = 0
    #: Multi-part pieces when split_mb was used (else [archive]).
    parts: list[Path] = field(default_factory=list)
    #: sha256 of the archive (lets receivers verify integrity).
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"archive": str(self.archive), "files": list(self.files),
                "bytes": self.bytes,
                "parts": [str(p) for p in self.parts],
                "sha256": self.sha256}


@dataclass
class DeliverResult:
    """Outcome of :func:`deliver_project` (zip + send)."""

    ok: bool
    zip: ZipResult | None = None
    message_id: str = ""
    detail: str = ""
    problems: list[str] = field(default_factory=list)
    #: How many send attempts were made (1 when the first try worked).
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "zip": self.zip.to_dict() if self.zip else None,
            "message_id": self.message_id,
            "detail": self.detail,
            "problems": list(self.problems),
            "attempts": self.attempts,
        }


@dataclass
class DeliveryReport:
    """Full build -> zip -> deliver pipeline report."""

    kind: str
    name: str
    project_dir: Path | None = None
    steps: list[BuildStep] = field(default_factory=list)
    zip_path: Path | None = None
    message_id: str = ""
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def broken(self) -> list[str]:
        return [s.name for s in self.steps if not s.ok]

    def summary(self) -> str:
        lines = [f"build_zip_and_deliver {self.kind}/{self.name}: "
                 f"{'OK' if self.ok else 'BROKEN'} ({self.elapsed:.1f}s)"]
        for step in self.steps:
            mark = "ok" if step.ok else "FAIL"
            first = step.detail.splitlines()[0] if step.detail else ""
            lines.append(f"  [{mark}] {step.name} ({step.elapsed:.1f}s) {first}")
        if self.zip_path:
            lines.append(f"  zip: {self.zip_path}")
        if self.message_id:
            lines.append(f"  message_id: {self.message_id}")
        if self.broken:
            lines.append("broken: " + ", ".join(self.broken))
        return "\n".join(lines)

    def fancy_summary(self, theme: str | None = None) -> str:
        """God-tier styled rendering of the delivery report."""
        from .style import banner, render_kv, render_steps, resolve_theme

        th = resolve_theme(theme)
        head = banner(f"deliver {self.kind}/{self.name}",
                      "OK" if self.ok else "BROKEN — see failed steps",
                      theme=th)
        body = render_steps([s.to_dict() for s in self.steps], theme=th)
        kv = render_kv([
            ("elapsed", f"{self.elapsed:.1f}s"),
            ("zip", str(self.zip_path) if self.zip_path else "—"),
            ("message_id", self.message_id or "—"),
            ("broken", ", ".join(self.broken) or "none"),
        ], theme=th)
        return "\n".join([head, body, kv])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name,
            "project_dir": str(self.project_dir) if self.project_dir else None,
            "ok": self.ok, "broken": self.broken,
            "zip_path": str(self.zip_path) if self.zip_path else None,
            "message_id": self.message_id,
            "elapsed": round(self.elapsed, 3),
            "steps": [s.to_dict() for s in self.steps],
        }


def _split_archive(archive: Path, split_mb: float) -> list[Path]:
    """Split ``archive`` into ``split_mb``-sized ``.partNN`` pieces.

    Chat platforms cap file uploads (Telegram bots: 50MB) — multi-part
    archives are the honest answer, not silent truncation.
    """
    chunk = int(split_mb * 1024 * 1024)
    data = archive.read_bytes()
    if len(data) <= chunk:
        return [archive]
    parts: list[Path] = []
    total = (len(data) + chunk - 1) // chunk
    for i in range(total):
        piece = archive.with_name(f"{archive.name}.part{i:02d}of{total:02d}")
        piece.write_bytes(data[i * chunk:(i + 1) * chunk])
        parts.append(piece)
    _log.info("split %s into %d parts of ~%.1fMB", archive.name, total, split_mb)
    return parts


def zip_project(project_dir: str | Path,
                dest: str | Path | None = None,
                split_mb: float = 0.0) -> ZipResult:
    """Create ``<name>.zip`` from ``project_dir``; return a :class:`ZipResult`.

    ``dest`` may be a directory (archive lands inside it) or a full file
    path.  Defaults to ``<parent>/<name>.zip``.  Excludes the same junk
    as :func:`nomorals.builders.export.export_project` plus the archive
    itself.

    ``split_mb > 0`` splits the finished zip into ``split_mb``-sized
    ``.partNNofMM`` pieces for chat size limits; ``parts`` lists them
    (reassemble with ``cat``).
    """
    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise ToolError(f"not a directory: {project_dir}")

    if dest is None:
        archive = project_dir.parent / f"{project_dir.name}.zip"
    else:
        dest = Path(dest).expanduser()
        archive = dest / f"{project_dir.name}.zip" if dest.suffix != ".zip" else dest
    if archive.exists():
        raise ToolError(f"archive already exists: {archive}")

    entries: list[tuple[str, Path]] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file() or _excluded(path, project_dir):
            continue
        if path.resolve() == archive.resolve():
            continue
        entries.append((path.relative_to(project_dir).as_posix(), path))
    if not entries:
        raise ToolError(f"nothing to zip in {project_dir}")

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, path in entries:
            zf.write(path, f"{project_dir.name}/{rel}")

    size = archive.stat().st_size
    files = [rel for rel, _ in entries]
    digest = _sha256_file(archive)
    parts = _split_archive(archive, split_mb) if split_mb > 0 else [archive]
    _log.info("zipped %s -> %s (%d files, %d bytes%s)",
              project_dir, archive, len(files), size,
              f", {len(parts)} parts" if len(parts) > 1 else "")
    return ZipResult(archive=archive, files=files, bytes=size,
                     parts=parts, sha256=digest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_caption(template: str, *, name: str, kind: str,
                    archive: Path, size: int, parts: int) -> str:
    """Caption templating: {name} {kind} {archive} {bytes} {mb} {parts}."""
    try:
        return template.format(name=name, kind=kind, archive=archive.name,
                               bytes=size, mb=f"{size / 1048576:.1f}",
                               parts=parts)
    except (KeyError, ValueError, IndexError):
        return template


def _resolve_gateway(gateway: Any = None, context: Any = None) -> Any:
    """Return a gateway object with a ``send_file`` method.

    Fails fast: a missing gateway, or one without ``send_file``, is a
    :class:`ToolError` -- never a silent no-send.
    """
    resolved = gateway
    if resolved is None and context is not None:
        extras = getattr(context, "extras", None) or {}
        resolved = extras.get("gateway")
    if resolved is None:
        raise ToolError(
            "deliver needs a gateway: pass gateway= explicitly or a context "
            "whose extras['gateway'] holds one")
    if not callable(getattr(resolved, "send_file", None)):
        raise ToolError(
            f"gateway {type(resolved).__name__!r} has no send_file method")
    return resolved


def _send_zip(gw: Any, zipped: ZipResult, *, platform: str, chat: str,
              caption: str, max_send_mb: float,
              retries: int = 2, backoff: float = 1.0) -> DeliverResult:
    """Send an already-built zip through the gateway; captures send failures.

    Transient send failures are retried with exponential backoff
    (``retries`` extra attempts); the attempt count lands in the result.
    When the zip was split, every part is sent in order and the message
    ids are joined.
    """
    attempts = 0
    problems: list[str] = []
    message_ids: list[str] = []
    targets = zipped.parts or [zipped.archive]
    for idx, piece in enumerate(targets):
        piece_caption = (caption if len(targets) == 1
                         else f"{caption} (part {idx + 1}/{len(targets)})")
        sent = False
        last_error = ""
        for attempt in range(retries + 1):
            attempts += 1
            try:
                result = gw.send_file(platform, chat, str(piece),
                                      caption=piece_caption,
                                      max_send_mb=max_send_mb)
            except Exception as exc:  # noqa: BLE001 -- captured, never raised
                last_error = f"send_file raised {type(exc).__name__}: {exc}"
                _log.warning("deliver send attempt %d failed: %s",
                             attempt + 1, last_error)
            else:
                if bool(getattr(result, "ok", False)):
                    message_ids.append(
                        str(getattr(result, "message_id", "") or ""))
                    sent = True
                    break
                last_error = (f"send_file failed: "
                              f"{getattr(result, 'error', '') or 'send failed'}")
                _log.warning("deliver send attempt %d failed: %s",
                             attempt + 1, last_error)
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
        if not sent:
            problems.append(f"{piece.name}: {last_error} "
                            f"({retries + 1} attempts)")
    ok = not problems
    detail = (f"sent {zipped.archive.name} ({zipped.bytes} bytes, "
              f"{len(targets)} part(s)) to {chat} as "
              f"{', '.join(message_ids)}" if ok else "; ".join(problems))
    _log.info("deliver_project %s -> %s (%d attempts)", zipped.archive,
              "OK" if ok else "FAIL", attempts)
    return DeliverResult(ok=ok, zip=zipped,
                         message_id=",".join(message_ids),
                         detail=detail, problems=problems,
                         attempts=attempts)


def deliver_project(project_dir: str | Path, *,
                    platform: str, chat: str,
                    gateway: Any = None, context: Any = None,
                    caption: str = "",
                    max_send_mb: float = 0.0,
                    dest: str | Path | None = None,
                    split_mb: float = 0.0,
                    retries: int = 2) -> DeliverResult:
    """Zip ``project_dir`` and send the archive to ``chat`` via the gateway.

    ``caption`` supports ``{name}`` ``{kind}`` ``{archive}`` ``{bytes}``
    ``{mb}`` ``{parts}`` templating.  ``split_mb`` splits the zip for
    chat size limits; ``retries`` controls send retry attempts.

    Returns a :class:`DeliverResult`; raises :class:`ToolError` only for a
    bad project dir, an existing archive, or a missing/broken gateway.  A
    failed *send* is captured in the result, not raised.
    """
    gw = _resolve_gateway(gateway=gateway, context=context)
    project_dir = Path(project_dir).expanduser().resolve()
    zipped = zip_project(project_dir, dest=dest, split_mb=split_mb)
    caption = _render_caption(
        caption or "{name} — built project archive",
        name=project_dir.name, kind="", archive=zipped.archive,
        size=zipped.bytes, parts=len(zipped.parts))
    return _send_zip(gw, zipped, platform=platform, chat=chat,
                     caption=caption, max_send_mb=max_send_mb,
                     retries=retries)


def build_zip_and_deliver(kind: str, name: str, dest: str | Path, *,
                          platform: str, chat: str,
                          gateway: Any = None, context: Any = None,
                          caption: str = "",
                          max_send_mb: float = 0.0,
                          export_dir: str | Path | None = None,
                          startup_timeout: float = 10.0,
                          policy: Any = None,
                          confirmation: str | None = None,
                          split_mb: float = 0.0,
                          retries: int = 2,
                          steps: list[str] | None = None,
                          skip: list[str] | None = None) -> DeliveryReport:
    """Scaffold, verify, zip, and deliver -- the full pipeline.

    The build/verify portion (scaffold -> install_deps -> project tests
    -> serve + smoke -> export + verify_export) is delegated to
    :func:`nomorals.builders.verify.build_and_verify` -- the canonical
    "after each coding mission verify and report" primitive -- so the
    delivery pipeline and the verify module can never drift apart.
    Then the verified project is zipped and delivered to chat.

    Steps: scaffold -> install_deps -> tests -> serve+smoke/smoke ->
    export -> zip -> deliver.  Every step's outcome lands in the report;
    nothing raises.
    """
    started = time.monotonic()
    verified = build_and_verify(
        kind, name, dest, policy=policy, confirmation=confirmation,
        export_dir=export_dir, startup_timeout=startup_timeout,
        steps=steps, skip=skip)
    report = DeliveryReport(kind=kind, name=name,
                            project_dir=verified.project_dir,
                            steps=list(verified.steps))
    if not verified.ok or verified.project_dir is None:
        # Verification failed -- the project is NOT zipped or delivered.
        # The report names the broken step(s); nothing raises.
        report.elapsed = time.monotonic() - started
        _log.info("\n%s", report.summary())
        return report

    # zip the verified project (reuse its dir -- never re-scaffold)
    step_started = time.monotonic()
    zipped: ZipResult | None = None
    try:
        zipped = zip_project(verified.project_dir,
                             dest=export_dir or verified.project_dir.parent,
                             split_mb=split_mb)
        report.zip_path = zipped.archive
        report.steps.append(BuildStep(
            "zip", True,
            f"{zipped.archive.name} ({zipped.bytes} bytes, "
            f"{len(zipped.files)} files"
            + (f", {len(zipped.parts)} parts" if len(zipped.parts) > 1 else "")
            + f", sha256 {zipped.sha256[:16]}…",
            time.monotonic() - step_started))
    except Exception as exc:  # noqa: BLE001
        report.steps.append(BuildStep("zip", False,
                                      f"{type(exc).__name__}: {exc}",
                                      time.monotonic() - step_started))
        report.elapsed = time.monotonic() - started
        return report

    # deliver (reuse the zip from the zip step -- never re-zip)
    step_started = time.monotonic()
    try:
        gw = _resolve_gateway(gateway=gateway, context=context)
        delivered = _send_zip(
            gw, zipped, platform=platform, chat=chat,
            caption=_render_caption(
                caption or "{name} ({kind}) — built project archive",
                name=name, kind=kind, archive=zipped.archive,
                size=zipped.bytes, parts=len(zipped.parts)),
            max_send_mb=max_send_mb, retries=retries)
    except Exception as exc:  # noqa: BLE001
        report.steps.append(BuildStep("deliver", False,
                                      f"{type(exc).__name__}: {exc}",
                                      time.monotonic() - step_started))
        report.elapsed = time.monotonic() - started
        return report
    if delivered.ok:
        report.message_id = delivered.message_id
    report.steps.append(BuildStep("deliver", delivered.ok, delivered.detail,
                                  time.monotonic() - step_started))

    report.elapsed = time.monotonic() - started
    _log.info("\n%s", report.summary())
    return report
