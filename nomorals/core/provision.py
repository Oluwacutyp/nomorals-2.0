"""Self-provision + fallback — the ONE shared seam every organ rides.

Any organ that needs a resource it does not have — a model file, a
Python package, a rung that can do the job — comes here instead of
telling the owner to do the setup himself:

- :func:`ensure_resource`: fetch a file (free/open, no login) into
  place — idempotent, resumable, validated, with real progress.
  A file that already sits there and validates is never
  re-downloaded.
- :func:`ensure_package`: pip-install a Python package when its
  import is missing, then re-probe. Small packages only — a
  heavy runtime (torch) is never installed behind the owner's
  back; that stays an honest gate.
- :func:`attempt_chain`: try ordered rungs (local → provisioned →
  remote); the first that produces wins; when all shed, one honest
  error naming every rung's failure. Never a fake result.

An organ declares what it needs (a :class:`ResourceSpec`: what,
where, how to tell a good landing) and/or a rung list, and calls
these two functions. No per-organ download code, no per-organ
fallback loop — that is the point: the seam is general, the
knowledge (URL, validator, rung order) stays with the organ.

Imggen is the first rider (stock SD1.5 checkpoint); the dead-end
queue is surveyed in ~/workspace/imggen-verify/SURVEY.md §5.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import NoMoralsError

__all__ = [
    "ProvisionError",
    "ResourceSpec",
    "ensure_resource",
    "ensure_package",
    "attempt_chain",
]

_DOWNLOAD_TIMEOUT_S = 600.0
_PIP_TIMEOUT_S = 300.0
_CHUNK = 1 << 20


class ProvisionError(NoMoralsError):
    """Self-provision or the whole fallback chain failed.

    Carries ``attempts`` — one receipt per thing tried, so the
    caller (and the owner) sees exactly what was attempted and
    why each step shed, instead of a bare "not available".
    """

    code = "provision.failed"

    def __init__(self, message: str = "",
                 attempts: list[str] | None = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])
        self.details.setdefault("attempts", self.attempts)


@dataclass
class ResourceSpec:
    """One fetchable resource: what, where, how to tell it landed well.

    ``urls`` are tried in order. ``validate`` takes the landed path
    and returns True, or an error string naming what is wrong with
    it — a failed validation deletes the file and raises, never a
    corrupt landing passed off as provisioned.
    """

    name: str
    urls: tuple[str, ...]
    dest: str | Path
    expected_bytes: int | None = None
    sha256: str | None = None
    validate: Callable[[Path], bool | str] | None = None
    user_agent: str = "devon-provision"
    timeout: float = _DOWNLOAD_TIMEOUT_S


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _validated(spec: ResourceSpec, path: Path) -> str | None:
    """None when ``path`` is a good landing, else the reason."""
    if spec.expected_bytes is not None:
        try:
            size = path.stat().st_size
        except OSError as exc:
            return f"cannot stat {path}: {exc}"
        if size != spec.expected_bytes:
            return (f"{spec.name}: {size} bytes landed, expected "
                    f"{spec.expected_bytes}")
    if spec.sha256 is not None:
        try:
            got = _sha256_of(path)
        except OSError as exc:
            return f"cannot hash {path}: {exc}"
        if got.lower() != spec.sha256.lower():
            return (f"{spec.name}: sha256 mismatch "
                    f"(got {got[:16]}…, want {spec.sha256[:16]}…)")
    if spec.validate is not None:
        try:
            verdict = spec.validate(path)
        except Exception as exc:  # noqa: BLE001 - validator failure is a reason
            return f"{spec.name}: validation raised: {exc}"
        if verdict is not True:
            return f"{spec.name}: invalid landing: {verdict}"
    return None


def _download_url(url: str, dest: Path, *,
                  progress: Callable[[int, int | None], None] | None = None,
                  user_agent: str = "devon-provision",
                  timeout: float = _DOWNLOAD_TIMEOUT_S) -> dict[str, Any]:
    """Stream ``url`` to ``dest`` with Range resume + progress.

    A ``.part`` sibling resumes from its size when the server
    answers 206; a server that ignores Range restarts cleanly.
    Returns measured facts about what landed.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": user_agent}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if have and status != 206:
                # The server ignored Range: restart from zero rather
                # than appending onto a stale prefix.
                have = 0
                part.unlink(missing_ok=True)
            total = resp.headers.get("Content-Length")
            total_n: int | None = None
            if total is not None:
                try:
                    total_n = int(total) + have
                except ValueError:
                    total_n = None
            mode = "ab" if have else "wb"
            downloaded = have
            with open(part, mode) as fh:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total_n)
            size = part.stat().st_size
            if total_n is not None and size != total_n:
                raise ProvisionError(
                    f"download of {url} was truncated: {size} of "
                    f"{total_n} bytes landed",
                    attempts=[f"download {url}: truncated "
                              f"{size}/{total_n}"])
    except ProvisionError:
        raise
    except BaseException as exc:  # noqa: BLE001 - the receipt names it
        raise ProvisionError(
            f"download of {url} failed: {exc}",
            attempts=[f"download {url}: {exc}"]) from exc
    os.replace(part, dest)
    return {"path": str(dest), "bytes": size,
            "sha256": _sha256_of(dest), "resumed": have > 0,
            "downloaded_bytes": size - have, "url": url}


def ensure_resource(
        spec: ResourceSpec, *,
        progress: Callable[[int, int | None], None] | None = None,
        downloader: Callable[..., dict[str, Any]] | None = None
        ) -> dict[str, Any]:
    """Fetch ``spec`` into place (or return it when already there).

    Idempotent: a destination that exists and validates returns at
    once with ``already: True`` — no download, no re-validation
    cost beyond the validator itself. Otherwise each URL is tried
    in order; the first landing that validates wins. A landing
    that fails validation is deleted and the next URL is tried.
    All URLs exhausted (or all validations failed) raises
    :class:`ProvisionError` with every attempt recorded.
    """
    dest = Path(spec.dest)
    if dest.exists():
        bad = _validated(spec, dest)
        if bad is None:
            return {"path": str(dest), "bytes": dest.stat().st_size,
                    "sha256": _sha256_of(dest), "resumed": False,
                    "downloaded_bytes": 0, "url": "",
                    "already": True}
        # A corrupt landing is not a landing: remove it and fetch
        # fresh, rather than serving it or failing on it.
        dest.unlink(missing_ok=True)
    download = downloader or _download_url
    attempts: list[str] = []
    for url in spec.urls:
        try:
            facts = download(
                url, dest, progress=progress,
                user_agent=spec.user_agent, timeout=spec.timeout)
        except ProvisionError as exc:
            attempts.extend(exc.attempts or [str(exc)])
            continue
        except Exception as exc:  # noqa: BLE001 - a rung sheds, named
            attempts.append(f"download {url}: {exc}")
            continue
        bad = _validated(spec, dest)
        if bad is None:
            facts["already"] = False
            return facts
        dest.unlink(missing_ok=True)
        attempts.append(f"landing from {url} rejected: {bad}")
    raise ProvisionError(
        f"could not provision {spec.name}: " + "; ".join(attempts),
        attempts=attempts)


def _run_capture(argv: list[str], timeout: float) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def ensure_package(package: str, *, import_name: str | None = None,
                   runner: Callable[..., tuple[int, str]] | None = None,
                   timeout: float = _PIP_TIMEOUT_S) -> dict[str, Any]:
    """pip-install ``package`` when its import is missing.

    Small packages only: a heavy runtime (torch) is never installed
    behind the owner's back — that stays an honest gate for the
    caller to name. Returns ``{"already": True}`` when the import
    already resolves, else the install receipt. Raises
    :class:`ProvisionError` with the pip tail on failure.
    """
    mod = import_name or package
    if importlib.util.find_spec(mod) is not None:
        return {"package": package, "already": True}
    run = runner or _run_capture
    argv = [sys.executable, "-m", "pip", "install", package]
    try:
        rc, output = run(argv, timeout)
    except subprocess.TimeoutExpired:
        raise ProvisionError(
            f"pip install of {package} timed out after "
            f"{timeout:.0f}s",
            attempts=[f"pip install {package}: timeout"]) from None
    except OSError as exc:
        raise ProvisionError(
            f"pip install of {package} could not run ({exc})",
            attempts=[f"pip install {package}: {exc}"]) from exc
    if rc != 0:
        tail = (output or "").strip().splitlines()[-1:] or [""]
        raise ProvisionError(
            f"pip install of {package} failed (exit {rc}): "
            f"{tail[0]}",
            attempts=[f"pip install {package}: exit {rc}: "
                      f"{tail[0]}"])
    if importlib.util.find_spec(mod) is None:
        raise ProvisionError(
            f"pip install of {package} reported success but "
            f"{mod} still does not import",
            attempts=[f"pip install {package}: installed but "
                      f"unimportable"])
    return {"package": package, "already": False,
            "receipt": (output or "").strip().splitlines()[-1:]}


def attempt_chain(what: str,
                  rungs: list[tuple[str, Callable[[], Any]]],
                  ) -> tuple[str, Any, list[str]]:
    """Try ordered rungs; the first that produces wins.

    Each rung is ``(name, callable)``. A rung sheds by raising —
    the failure is recorded as ``"name: <error>"`` and the next
    rung runs. Returns ``(rung_name, result, attempts)``. When
    every rung sheds, raises :class:`ProvisionError` naming every
    rung's failure — the honest error, only when truly nothing in
    the system can produce ``what``.
    """
    attempts: list[str] = []
    for name, fn in rungs:
        try:
            return name, fn(), attempts
        except Exception as exc:  # noqa: BLE001 - a rung sheds, named
            attempts.append(f"{name}: {exc}")
    raise ProvisionError(
        f"nothing in the system could produce {what}: "
        + "; ".join(attempts),
        attempts=attempts)
