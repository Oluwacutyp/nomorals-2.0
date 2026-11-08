"""Consent-to-download flow for the SAM mask rung (plan §1.1d).

LightCraft's governing pattern: ``ErrorKind::ModelMissing`` → the
app **offers** a consent-to-download; models are user-downloaded,
never bundled. Devon's shape of the same law, through her
self-provisioning idiom (probe → state reason → install on the
owner's standing grant → re-probe):

- :meth:`MaskProvisionFlow.assess` probes :func:`mask_status`.
  SAM usable → ``available``. Caller already has a working mask
  from a lower rung → ``not_needed`` (no offer is made). SAM
  missing but installable → ``offered``, with the one plain
  message the surface shows the owner. A recorded refusal →
  ``refused`` — she does not nag; the offer returns only when
  the owner asks again (:meth:`reopen`) or SAM lands by another
  route (probe wins over the record).
- :meth:`MaskProvisionFlow.accept` (the owner's yes) installs:
  the ``segment-anything`` package via pip when the package is
  what's missing, and the ViT-B checkpoint — the smallest
  official SAM weights, ~375 MB — into ``~/.nomorals/models/``
  when the checkpoint is what's missing. Then it **re-probes in
  a fresh process** (a pip install cannot change this process's
  import cache, so an in-process re-check would lie). Usable →
  ``installed``; anything else → ``failed`` with the attempt
  receipts attached, in plain words — never a chore list while
  a route remains untried.
- :meth:`MaskProvisionFlow.refuse` records the owner's no.

Every seam is injectable (``probe`` / ``installer`` /
``reprobe`` / ``routes`` / ``state_path``) so tests stage whole
machines with fakes — no network, no real download, no real
install ever runs in a test. The download verifies its byte
count against the server's Content-Length when one is given,
lands via a ``.part`` file + atomic rename, and the flow record
keeps the landed file's size and SHA-256 as measured facts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import ImgGenError
from .failures import record_failure
from .masks import mask_status

__all__ = [
    "MaskProvisionError",
    "MaskProvisionFlow",
    "default_flow",
    "STATES",
    "SAM_PACKAGE",
    "SAM_CHECKPOINT_NAME",
    "SAM_CHECKPOINT_URL",
    "SAM_CHECKPOINT_MB",
]

#: The flow's honest states.
STATES = ("not_needed", "available", "offered", "installed",
          "refused", "failed")

SAM_PACKAGE = "segment-anything"
SAM_PIP_MODULE = "segment_anything"
#: The name sam_status() looks for — the flow lands the official
#: file under the name the probe already knows.
SAM_CHECKPOINT_NAME = "sam_vit_b.pth"
SAM_CHECKPOINT_URL = ("https://dl.fbaipublicfiles.com/"
                      "segment_anything/sam_vit_b_01ec64.pth")
SAM_CHECKPOINT_MB = 375  # official ViT-B size, for the offer text

_MASK_MODULE = "nomorals.media.imggen.masks"
_DOWNLOAD_TIMEOUT_S = 1800.0
_PIP_TIMEOUT_S = 1200.0


class MaskProvisionError(ImgGenError):
    """An install ran and failed; ``attempts`` is the plain list
    of what was tried and how each route ended (the
    NodeProvisionFailed receipts shape)."""

    def __init__(self, attempts: list[str]) -> None:
        self.attempts = list(attempts)
        super().__init__("; ".join(self.attempts)
                         or "no install route applies")


def models_dir() -> Path:
    return Path.home() / ".nomorals" / "models"


def default_state_path() -> Path:
    return Path.home() / ".nomorals" / "mask_provision.json"


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _output_tail(output: str, limit: int = 280) -> str:
    lines = [ln.strip() for ln in (output or "").splitlines()
             if ln.strip()]
    tail = " · ".join(lines[-3:]) if lines else "no output"
    if len(tail) > limit:
        tail = "…" + tail[-(limit - 1):]
    return tail


def _run_capture(argv: list[str], timeout: float) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def _download_file(url: str, dest: Path) -> dict[str, Any]:
    """Stream ``url`` to ``dest`` via a ``.part`` sibling + atomic
    rename; byte count verified against Content-Length when the
    server sends one. Returns measured facts about what landed."""
    import urllib.request

    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "devon-mask-provision"})
    try:
        with urllib.request.urlopen(
                req, timeout=_DOWNLOAD_TIMEOUT_S) as resp, \
                open(part, "wb") as fh:
            expected = resp.headers.get("Content-Length")
            shutil.copyfileobj(resp, fh, length=1 << 20)
        size = part.stat().st_size
        if expected is not None and size != int(expected):
            raise MaskProvisionError([
                f"download of {url} was truncated: "
                f"{size} of {expected} bytes landed"])
        os.replace(part, dest)
    except BaseException:  # noqa: BLE001 — cleanup, then re-raise
        # Deliberate catch-all: whatever interrupted the download
        # (error, KeyboardInterrupt, SystemExit), the partial file
        # must not survive it — and the interruption itself always
        # propagates unchanged.
        part.unlink(missing_ok=True)
        raise
    return {"path": str(dest), "bytes": size,
            "sha256": _sha256_of(dest)}


def _fresh_mask_status() -> dict:
    """``mask_status()`` from a *fresh* interpreter.

    A pip install cannot change this process's import cache, so
    after an install the only honest re-probe is a new process
    (the provision_node re-resolve idiom, import-shaped). Raises
    with the child's own words when the probe cannot run.
    """
    code = (
        "import json;"
        f"from {_MASK_MODULE} import mask_status;"
        "print(json.dumps(mask_status()))")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p])
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True,
        text=True, timeout=120, env=env)
    if done.returncode != 0:
        raise MaskProvisionError([
            "fresh-process re-probe failed: "
            + _output_tail(done.stderr or done.stdout)])
    return json.loads(done.stdout.strip().splitlines()[-1])


class _SamInstaller:
    """The real install routes for the SAM rung, built from
    injectable ``runner`` / ``downloader`` so tests stage them.
    ``install()`` returns the landed checkpoint facts (or None);
    raises :class:`MaskProvisionError` with receipts on failure."""

    def __init__(self, *, runner=None, downloader=None,
                 dest_dir: Path | None = None) -> None:
        self._runner = runner or (
            lambda argv: _run_capture(argv, _PIP_TIMEOUT_S))
        self._downloader = downloader or _download_file
        self._dest_dir = (Path(dest_dir) if dest_dir is not None
                          else models_dir())

    def install(self, sam: dict) -> dict | None:
        attempts: list[str] = []
        facts: dict | None = None
        if not sam.get("package"):
            try:
                rc, output = self._runner(
                    [sys.executable, "-m", "pip", "install",
                     SAM_PACKAGE])
            except subprocess.TimeoutExpired:
                attempts.append(
                    f"pip install of {SAM_PACKAGE} timed out "
                    f"after {_PIP_TIMEOUT_S:.0f}s")
            except OSError as exc:
                attempts.append(
                    f"pip install of {SAM_PACKAGE} could not "
                    f"run ({exc})")
            else:
                if rc != 0:
                    attempts.append(
                        f"pip install of {SAM_PACKAGE} failed "
                        f"(exit {rc}): {_output_tail(output)}")
        if not sam.get("checkpoint"):
            dest = self._dest_dir / SAM_CHECKPOINT_NAME
            try:
                self._dest_dir.mkdir(parents=True, exist_ok=True)
                facts = self._downloader(SAM_CHECKPOINT_URL, dest)
            except MaskProvisionError as exc:
                attempts.extend(exc.attempts)
            except Exception as exc:  # noqa: BLE001 - named attempt
                attempts.append(
                    f"download of the SAM checkpoint "
                    f"({SAM_CHECKPOINT_URL}) failed: {exc}")
        if attempts:
            raise MaskProvisionError(attempts)
        return facts


def _default_routes(sam: dict) -> list[str]:
    """Which install routes apply to what the probe says is
    missing — the honest precondition for making an offer."""
    routes: list[str] = []
    if not sam.get("package") and \
            importlib.util.find_spec("pip") is not None:
        routes.append("package")
    if not sam.get("checkpoint"):
        routes.append("checkpoint")
    return routes


class MaskProvisionFlow:
    """Probe → offer → (owner's yes) install → re-probe, with
    the owner's no remembered. One instance is cheap and
    stateless apart from the JSON record at ``state_path``."""

    def __init__(self, *, probe: Callable[[], dict] | None = None,
                 installer: Any = None,
                 reprobe: Callable[[], dict] | None = None,
                 routes: Callable[[dict], list[str]] | None = None,
                 state_path: str | Path | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self._probe = probe or mask_status
        self._installer = installer or _SamInstaller()
        self._reprobe = reprobe or _fresh_mask_status
        self._routes = routes or _default_routes
        self._state_path = (Path(state_path) if state_path
                            is not None else default_state_path())
        self._clock = clock or time.time

    # -- record plumbing -------------------------------------------
    def _record(self, state: str, *, reason: str, detail: str = "",
                offer: str = "", attempts: list[str] | None = None,
                checkpoint: dict | None = None) -> dict:
        return {
            "state": state,
            "rung": "sam",
            "reason": reason,
            "detail": detail,
            "offer": offer,
            "attempts": list(attempts or []),
            "checkpoint": checkpoint,
            "updated_at": self._clock(),
        }

    def _persist(self, record: dict) -> dict:
        try:
            self._state_path.parent.mkdir(parents=True,
                                          exist_ok=True)
            tmp = self._state_path.with_suffix(
                self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(record, indent=1),
                           encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:
            # The flow's answer stands even when the record
            # cannot be kept (read-only home); only persistence
            # of the owner's refusal is lost, never the truth of
            # this call — but the loss itself is recorded.
            record_failure("mask_provision._persist", exc)
        return record

    def _saved(self) -> dict | None:
        try:
            data = json.loads(
                self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _sam(self, status: dict) -> dict:
        sam = status.get("sam") if isinstance(status, dict) \
            else None
        if not isinstance(sam, dict):
            raise MaskProvisionError(
                ["mask_status() returned no 'sam' rung"])
        return sam

    def _offer_text(self, sam: dict, routes: list[str]) -> str:
        steps = []
        if "package" in routes:
            steps.append(f"pip install {SAM_PACKAGE}")
        if "checkpoint" in routes:
            steps.append(
                f"download the SAM ViT-B checkpoint "
                f"(~{SAM_CHECKPOINT_MB} MB) into "
                f"{models_dir()}")
        return (
            f"The best mask rung (SAM) isn't available: "
            f"{sam['reason']}. I can fix that myself — "
            + " and ".join(steps)
            + ". Say yes and I'll install it and re-check; "
              "the GrabCut/flood/box rungs keep working "
              "meanwhile. Say no and I won't ask again.")

    # -- the flow ----------------------------------------------------
    def assess(self, *, need: bool = True) -> dict:
        """Probe and place the flow — no side effects beyond
        returning an ``offered`` record (the offer is *data*;
        showing it to the owner is the surface's job)."""
        try:
            sam = self._sam(self._probe())
        except MaskProvisionError as exc:
            return self._record("failed", reason=str(exc),
                                attempts=exc.attempts)
        except Exception as exc:  # noqa: BLE001 - probe honesty
            return self._record(
                "failed",
                reason=f"mask probe failed: {exc}")
        if sam.get("usable"):
            return self._record("available",
                                reason=str(sam.get("reason", "ok")))
        if not need:
            return self._record(
                "not_needed",
                reason=str(sam.get("reason", "")),
                detail="the caller's mask already came from a "
                       "working lower rung; no offer made")
        saved = self._saved()
        if saved and saved.get("state") == "refused" and \
                saved.get("reason") == sam.get("reason"):
            return self._record(
                "refused", reason=str(sam.get("reason", "")),
                detail="the owner already declined this offer; "
                       "it stays closed until they ask again")
        missing = [k for k in ("package", "checkpoint")
                   if not sam.get(k)]
        routes = self._routes(sam)
        needed = (["package"] if not sam.get("package") else []) \
            + (["checkpoint"] if not sam.get("checkpoint")
               else [])
        if sorted(routes) != sorted(needed):
            return self._record(
                "failed",
                reason=str(sam.get("reason", "")),
                detail="SAM is missing "
                       f"({', '.join(missing) or 'unknown part'}) "
                       "and no install route applies on this "
                       "machine")
        return self._record(
            "offered", reason=str(sam.get("reason", "")),
            offer=self._offer_text(sam, routes))

    def accept(self) -> dict:
        """The owner's yes: install, then re-probe in a fresh
        process. Only a usable re-probe earns ``installed``."""
        try:
            sam = self._sam(self._probe())
        except Exception as exc:  # noqa: BLE001 - honest failure
            return self._persist(self._record(
                "failed", reason=f"mask probe failed: {exc}"))
        if sam.get("usable"):
            return self._persist(self._record(
                "available", reason=str(sam.get("reason", "ok")),
                detail="SAM was already usable; nothing "
                       "installed"))
        try:
            facts = self._installer.install(sam)
        except MaskProvisionError as exc:
            return self._persist(self._record(
                "failed", reason=str(sam.get("reason", "")),
                attempts=exc.attempts))
        except Exception as exc:  # noqa: BLE001 - named failure
            return self._persist(self._record(
                "failed", reason=str(sam.get("reason", "")),
                attempts=[f"install failed: {exc}"]))
        try:
            after = self._sam(self._reprobe())
        except MaskProvisionError as exc:
            return self._persist(self._record(
                "failed", reason=str(sam.get("reason", "")),
                detail="the install landed but could not be "
                       "verified",
                attempts=exc.attempts, checkpoint=facts))
        except Exception as exc:  # noqa: BLE001 - named failure
            return self._persist(self._record(
                "failed", reason=str(sam.get("reason", "")),
                detail="the install landed but the re-probe "
                       f"failed: {exc}",
                checkpoint=facts))
        if after.get("usable"):
            return self._persist(self._record(
                "installed", reason="ok",
                detail="installed and verified by a fresh "
                       "re-probe",
                checkpoint=facts))
        return self._persist(self._record(
            "failed", reason=str(after.get("reason", "")),
            detail="the install landed, but a fresh re-probe "
                   "still reports SAM unusable",
            checkpoint=facts))

    def refuse(self) -> dict:
        """The owner's no, remembered against the probe's
        current reason — a changed situation re-opens the
        question on its own."""
        try:
            sam = self._sam(self._probe())
            reason = str(sam.get("reason", ""))
        except Exception:  # noqa: BLE001 - refusal still records
            reason = ""
        return self._persist(self._record(
            "refused", reason=reason,
            detail="the owner declined the SAM download; "
                   "lower rungs keep serving masks"))

    def reopen(self) -> dict:
        """The owner asked again after refusing: clear the
        record and assess fresh."""
        try:
            self._state_path.unlink(missing_ok=True)
        except OSError as exc:
            # Reopening still assesses fresh; a state file that
            # refuses to clear is recorded, not silently kept.
            record_failure("mask_provision.reopen", exc)
        return self.assess()


def default_flow() -> MaskProvisionFlow:
    """The flow over the real machine, real state path."""
    return MaskProvisionFlow()
