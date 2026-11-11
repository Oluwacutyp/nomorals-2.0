"""Self-provisioning for the browser engine (the yt-dlp pattern).

Playwright used to be a lazy, optional dependency everywhere in the
browser organ: when it was missing, every rendered surface failed
fast with an install hint — a command handed to the owner. That is
not how Devon treats a missing tool (owner doctrine: self-provision,
mirroring :func:`nomorals.media.resolver.ensure_ytdlp` for yt-dlp).
This module is the browser engine's half of that promise:

* :func:`playwright_usable` — package importable AND a Chromium
  build present in the browsers root;
* :func:`ensure_playwright` — when it is missing, ONE bounded
  install attempt per process (``pip install playwright`` then
  ``python -m playwright install chromium``), re-verified after.
  On profiles that cannot host a ~170MB browser (Termux / mobile /
  embedded) it does not thrash the device: it reports unavailable
  honestly and the callers keep their plain-words failure.
* Every failure is journaled in the browser failure ledger and
  logged; nothing here ever fakes success or raises into a caller.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "browsers_root",
    "chromium_present",
    "ensure_playwright",
    "playwright_package_present",
    "playwright_usable",
]

#: One self-provision attempt per process — a missing engine is
#: Devon's job to fix, but a broken network must not turn every
#: rendered open into a pip run.
_PROVISION_ATTEMPTED = False

#: Profiles that cannot host a Chromium build. Provisioning there
#: would burn a constrained device for an install that can never
#: serve; those profiles report the engine unavailable, honestly.
_NO_INSTALL_PROFILES = frozenset({"termux", "mobile", "embedded"})

_PIP_TIMEOUT_S = 300
_BROWSER_TIMEOUT_S = 900

_CHROMIUM_EXE_NAMES = frozenset({
    "chrome", "headless_shell", "chrome.exe", "headless_shell.exe",
})


def playwright_package_present() -> bool:
    """True when the ``playwright`` package is importable here."""
    try:
        return importlib.util.find_spec("playwright") is not None
    except Exception:  # noqa: BLE001 - a broken spec lookup is "absent"
        return False


def browsers_root() -> Path:
    """The directory Playwright keeps its browser builds in."""
    pinned = (os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if pinned:
        return Path(pinned).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform.startswith("win"):
        local = (os.environ.get("LOCALAPPDATA") or "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def chromium_present(root: Path | None = None) -> bool:
    """True when a Chromium build with a runnable binary is on disk."""
    base = root if root is not None else browsers_root()
    try:
        if not base.is_dir():
            return False
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if not (child.name.startswith("chromium-")
                    or child.name.startswith("chromium_headless_shell-")):
                continue
            for dirpath, _dirnames, filenames in os.walk(child):
                if _CHROMIUM_EXE_NAMES.intersection(filenames):
                    return True
                # Browser builds nest the binary a couple of levels
                # down at most; don't wander the whole tree.
                if Path(dirpath).relative_to(child).parts \
                        and len(Path(dirpath).relative_to(child).parts) > 3:
                    _dirnames[:] = []
    except OSError:
        return False
    return False


def playwright_usable() -> bool:
    """True when a rendered tab can actually be opened right now."""
    return playwright_package_present() and chromium_present()


def _profile_allows_install(profile_kind: str | None) -> bool:
    kind = (profile_kind or "").strip().lower()
    if not kind:
        try:
            from ..core.profiles import get_profile_kind

            kind = str(get_profile_kind() or "").strip().lower()
        except Exception:  # noqa: BLE001 - unknown profile: allow
            kind = ""
    return kind not in _NO_INSTALL_PROFILES


def _run(cmd: list[str], timeout_s: float) -> Any:
    """One bounded provisioning command (the tests' seam)."""
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_s)


def _record(site: str, exc: BaseException) -> None:
    try:
        from .failures import record_failure

        record_failure(site, exc)
    except Exception:  # noqa: BLE001 - the ledger never gates
        pass


def ensure_playwright(*, runner: Callable[..., Any] | None = None,
                      profile_kind: str | None = None) -> bool:
    """Make the browser engine usable, installing it when missing.

    Mirrors ``ensure_ytdlp``: one bounded attempt per process,
    re-verified, never raises. Returns True when Playwright +
    Chromium are usable afterwards. When every route fails the
    honest not-installed failure stands at the caller — never a
    silent downgrade, never a fake success.

    Under a test runner the environment is never mutated (the
    availability state is simulated by the tests there instead).
    """
    global _PROVISION_ATTEMPTED
    if playwright_usable():
        return True
    if _PROVISION_ATTEMPTED:
        return False
    _PROVISION_ATTEMPTED = True
    if "pytest" in sys.modules:
        return False
    if not _profile_allows_install(profile_kind):
        _log.info(
            "browser engine self-provision skipped: this profile "
            "cannot host a Chromium build; rendered tabs report "
            "unavailable here")
        _record("browser.provision.profile",
                RuntimeError("profile cannot host a Chromium build"))
        return False
    run = runner or _run
    steps = [
        ([sys.executable, "-m", "pip", "install", "playwright"],
         _PIP_TIMEOUT_S),
        ([sys.executable, "-m", "playwright", "install", "chromium"],
         _BROWSER_TIMEOUT_S),
    ]
    for cmd, timeout_s in steps:
        try:
            proc = run(cmd, timeout_s)
        except Exception as exc:  # noqa: BLE001 - provisioning is
            # best-effort; the honest failure stands when it can't run
            _log.info("browser engine self-provision step failed to "
                      "run (%s): %s", cmd[2:], exc)
            _record("browser.provision.playwright", exc)
            return False
        if getattr(proc, "returncode", 1) != 0:
            tail = (getattr(proc, "stderr", "")
                    or getattr(proc, "stdout", "") or "")[-300:]
            _log.info("browser engine self-provision step failed "
                      "(%s rc=%s): %s", cmd[2:],
                      getattr(proc, "returncode", "?"), tail)
            _record("browser.provision.playwright",
                    RuntimeError(
                        f"install step {' '.join(cmd[1:3])} rc="
                        f"{getattr(proc, 'returncode', '?')}: {tail}"))
            return False
    ok = playwright_usable()
    _log.info("browser engine self-provision %s",
              "succeeded" if ok else
              "ran but playwright/chromium is still not usable")
    if not ok:
        _record("browser.provision.playwright",
                RuntimeError("install ran but the engine is still "
                             "not usable"))
    return ok
