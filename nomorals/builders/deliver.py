"""Build -> zip -> deliver: package a project as a ``.zip`` and send it to chat.

:func:`zip_project` builds a ``<name>.zip`` from a project directory with
the same exclusions as :mod:`nomorals.builders.export` (``.git``,
``__pycache__``, ``.venv``, ``node_modules``, ...).  :func:`deliver_project`
zips and then sends the archive through a chat gateway's ``send_file``
path (e.g. :meth:`nomorals.social.chat.gateway.ChatGateway.send_file`).
:func:`build_zip_and_deliver` runs the whole pipeline end to end --
scaffold, tests, smoke, zip, deliver -- capturing every step in a
:class:`DeliveryReport` instead of raising.

The gateway is duck-typed (only ``send_file`` is used) so this module
never imports the social layer.  Pass the gateway explicitly, or pass an
agent context and it is resolved from ``context.extras["gateway"]`` --
the runtime's canonical location (``context.gateway`` does not exist).
"""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from .export import EXCLUDE_DIRS
from .run import ServeError, run_config, serve
from .scaffold import scaffold
from .smoke import SmokeResult, smoke_test
from .verify import BuildStep, run_project_tests

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

    def to_dict(self) -> dict[str, Any]:
        return {"archive": str(self.archive), "files": list(self.files),
                "bytes": self.bytes}


@dataclass
class DeliverResult:
    """Outcome of :func:`deliver_project` (zip + send)."""

    ok: bool
    zip: ZipResult | None = None
    message_id: str = ""
    detail: str = ""
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "zip": self.zip.to_dict() if self.zip else None,
            "message_id": self.message_id,
            "detail": self.detail,
            "problems": list(self.problems),
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


def zip_project(project_dir: str | Path,
                dest: str | Path | None = None) -> ZipResult:
    """Create ``<name>.zip`` from ``project_dir``; return a :class:`ZipResult`.

    ``dest`` may be a directory (archive lands inside it) or a full file
    path.  Defaults to ``<parent>/<name>.zip``.  Excludes the same junk
    as :func:`nomorals.builders.export.export_project` plus the archive
    itself.
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
    _log.info("zipped %s -> %s (%d files, %d bytes)",
              project_dir, archive, len(files), size)
    return ZipResult(archive=archive, files=files, bytes=size)


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
              caption: str, max_send_mb: float) -> DeliverResult:
    """Send an already-built zip through the gateway; captures send failures."""
    try:
        result = gw.send_file(platform, chat, str(zipped.archive),
                              caption=caption, max_send_mb=max_send_mb)
    except Exception as exc:  # noqa: BLE001 -- captured, never raised
        problems = [f"send_file raised {type(exc).__name__}: {exc}"]
        _log.warning("deliver_project send failed: %s", problems[0])
        return DeliverResult(ok=False, zip=zipped, problems=problems,
                             detail=problems[0])

    ok = bool(getattr(result, "ok", False))
    message_id = str(getattr(result, "message_id", "") or "")
    problems = []
    if not ok:
        error = str(getattr(result, "error", "") or "send failed")
        problems.append(f"send_file failed: {error}")
    detail = (f"sent {zipped.archive.name} ({zipped.bytes} bytes) "
              f"to {chat} as {message_id}" if ok
              else "; ".join(problems))
    _log.info("deliver_project %s -> %s", zipped.archive,
              "OK" if ok else "FAIL")
    return DeliverResult(ok=ok, zip=zipped, message_id=message_id,
                         detail=detail, problems=problems)


def deliver_project(project_dir: str | Path, *,
                    platform: str, chat: str,
                    gateway: Any = None, context: Any = None,
                    caption: str = "",
                    max_send_mb: float = 0.0,
                    dest: str | Path | None = None) -> DeliverResult:
    """Zip ``project_dir`` and send the archive to ``chat`` via the gateway.

    Returns a :class:`DeliverResult`; raises :class:`ToolError` only for a
    bad project dir, an existing archive, or a missing/broken gateway.  A
    failed *send* is captured in the result, not raised.
    """
    gw = _resolve_gateway(gateway=gateway, context=context)
    project_dir = Path(project_dir).expanduser().resolve()
    zipped = zip_project(project_dir, dest=dest)
    return _send_zip(gw, zipped, platform=platform, chat=chat,
                     caption=caption or f"{project_dir.name} -- built project archive",
                     max_send_mb=max_send_mb)


def build_zip_and_deliver(kind: str, name: str, dest: str | Path, *,
                          platform: str, chat: str,
                          gateway: Any = None, context: Any = None,
                          caption: str = "",
                          max_send_mb: float = 0.0,
                          export_dir: str | Path | None = None,
                          startup_timeout: float = 10.0) -> DeliveryReport:
    """Scaffold, test, smoke, zip, and deliver -- the full pipeline.

    Steps: scaffold -> run project tests -> serve + smoke (HTTP kinds)
    or --help smoke (CLI/console kinds) -> zip -> deliver.  Every step's
    outcome lands in the report; nothing raises.
    """
    started = time.monotonic()
    report = DeliveryReport(kind=kind, name=name)
    dest = Path(dest).expanduser()

    # 1. scaffold
    step_started = time.monotonic()
    try:
        result = scaffold(kind, name, dest)
    except Exception as exc:  # noqa: BLE001 -- captured into the report
        report.steps.append(BuildStep("scaffold", False, f"{type(exc).__name__}: {exc}",
                                      time.monotonic() - step_started))
        report.elapsed = time.monotonic() - started
        return report
    report.project_dir = result.project_dir
    report.steps.append(BuildStep(
        "scaffold", True,
        f"{len(result.files)} files -> {result.project_dir}",
        time.monotonic() - step_started))

    # 2. project tests
    report.steps.append(run_project_tests(result))

    # 3. serve + smoke (HTTP) or --help smoke (CLI/console)
    step_started = time.monotonic()
    try:
        config = run_config(result.project_dir)
    except Exception as exc:  # noqa: BLE001
        report.steps.append(BuildStep("smoke", False, f"run_config: {exc}",
                                      time.monotonic() - step_started))
        config = None
    if config is not None:
        if config.kind == "http":
            try:
                with serve(result.project_dir, port=0,
                           startup_timeout=startup_timeout) as handle:
                    smoke: SmokeResult = smoke_test(handle, timeout=startup_timeout)
                detail = "; ".join(
                    f"{c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}".strip()
                    for c in smoke.checks)
                report.steps.append(BuildStep("serve+smoke", smoke.ok, detail,
                                              time.monotonic() - step_started))
            except ServeError as exc:
                report.steps.append(BuildStep("serve+smoke", False,
                                              f"{exc}\nstderr:\n{exc.stderr}",
                                              time.monotonic() - step_started))
            except Exception as exc:  # noqa: BLE001
                report.steps.append(BuildStep("serve+smoke", False,
                                              f"{type(exc).__name__}: {exc}",
                                              time.monotonic() - step_started))
        else:
            try:
                smoke = smoke_test(result.project_dir, timeout=30.0)
                detail = "; ".join(
                    f"{c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}".strip()
                    for c in smoke.checks)
                report.steps.append(BuildStep("smoke", smoke.ok, detail,
                                              time.monotonic() - step_started))
            except Exception as exc:  # noqa: BLE001
                report.steps.append(BuildStep("smoke", False,
                                              f"{type(exc).__name__}: {exc}",
                                              time.monotonic() - step_started))

    # 4. zip
    step_started = time.monotonic()
    zipped: ZipResult | None = None
    try:
        zipped = zip_project(result.project_dir,
                             dest=export_dir or result.project_dir.parent)
        report.zip_path = zipped.archive
        report.steps.append(BuildStep(
            "zip", True,
            f"{zipped.archive.name} ({zipped.bytes} bytes, "
            f"{len(zipped.files)} files)",
            time.monotonic() - step_started))
    except Exception as exc:  # noqa: BLE001
        report.steps.append(BuildStep("zip", False,
                                      f"{type(exc).__name__}: {exc}",
                                      time.monotonic() - step_started))
        report.elapsed = time.monotonic() - started
        return report

    # 5. deliver (reuse the zip from step 4 -- never re-zip)
    step_started = time.monotonic()
    try:
        gw = _resolve_gateway(gateway=gateway, context=context)
        delivered = _send_zip(
            gw, zipped, platform=platform, chat=chat,
            caption=caption or f"{name} ({kind}) -- built project archive",
            max_send_mb=max_send_mb)
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
