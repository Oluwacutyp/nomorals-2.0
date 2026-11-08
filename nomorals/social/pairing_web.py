"""The pairing pool's web contract + the PHP pairing page (§15).

The pool already had two surfaces: the in-chat ``/pair`` flow (the
primary) and a read-only Python render at ``GET /pair`` on the API.
This module is the third piece the owner's polyglot order adds:

* **The contract** — ``pairing_state`` / ``pairing_action``: the
  registry's machine-readable seam.  State is the same truth the
  chat summary and the Python page render (counts, every account
  with its real toggle states, live pairing codes, pending Telegram
  flows); actions dispatch to the SAME supervisor / pairer /
  personal-surfaces calls ``_control_pair`` makes, with the same
  owner-stop bookkeeping, so the page and the chat can never drift
  into two different pools.  The API server exposes the contract at
  ``GET /pair/state`` and ``POST /pair/action``, both gated on the
  ``sys.config`` capability — the owner principal, exactly like the
  HTML page: a live pairing code is a credential while it lives.

* **The PHP rung** — Devon's own pairing page as a standalone PHP
  page (``web/pairing.php`` in the repo), served by PHP's built-in
  server on loopback.  PHP is provisioned the way Node is
  (``provision_php`` mirrors ``provision_node``: resolve the ladder
  first, then pinned downloads / platform package routes, every
  attempt recorded, an honest failure when no route lands).  The
  page itself is owner-gated by a per-workspace page token Devon
  generates and stores under ``data/social/``; it talks to the
  contract above with the owner's API token and renders nothing it
  did not just read — when the API is down or refuses, the page
  says so instead of inventing a pool.

* **The rung registry** — ``pairing_surface_rungs`` states, per
  device, which pairing surfaces exist: the PHP web rung (probe:
  a PHP runtime resolves and the page file is present), the API
  page rung, and the in-chat rung (always).  ``nm doctor`` prints
  it; ``/pair page`` launches the best web rung available.

Cross-instance honesty: supervisors built by different processes
share the registry file and adopt — never double-launch — a bridge
already listening on an account's port.  A pairing code is known
only to the supervisor instance that launched that bridge; a
surface that did not launch it reports the account as pairing
with no code visible, never a fabricated one.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .pairing import PERSONAL_TOGGLES, PairingRegistry, PairingUnavailable, default_supervisor

_log = get_logger(__name__)

#: The PHP page, relative to the repo root (this module's
#: great-grandparent).  One file, no framework.
PAGE_FILE = Path(__file__).resolve().parents[2] / "web" / "pairing.php"

#: Pinned PHP for the self-provisioning download route.  Windows
#: gets the official windows.php.net zip; POSIX gets the
#: static-php-cli common build (a single self-contained CLI
#: binary, which is all the built-in server needs).
PHP_VERSION = "8.3.13"
_PHP_WIN_URL = ("https://windows.php.net/downloads/releases/"
                f"php-{PHP_VERSION}-Win32-vs16-x64.zip")
_PHP_STATIC_BASE = "https://dl.static-php.dev/static-php-cli/common"
PHP_PROVISION_TIMEOUT_S = 600.0


class PhpProvisionFailed(RuntimeError):
    """No PHP provisioning route landed a runtime; carries the
    attempts, like :class:`NodeProvisionFailed`."""

    def __init__(self, attempts: list[str]) -> None:
        self.attempts = attempts
        super().__init__(
            "PHP could not be provisioned on this machine. "
            "Attempts: " + " | ".join(attempts))


# ── the service bundle (one per context, cached) ────────────────────

def _bundle_for_context(context: Any) -> dict[str, Any]:
    """The contract's service bundle over the context's workspace:
    the production supervisor, the Telegram pairer and the personal
    surfaces organ — built once per context and cached on it, the
    way the partner runtime caches its own bundle.  Actions taken
    through the API therefore share ONE supervisor inside the API
    process: a bridge it launches is a bridge it can read codes
    back from."""
    svc = getattr(context, "_pairing_web_svc", None)
    if svc is not None:
        return svc
    from .personal import PersonalSurfaces, PersonalVault
    from .telegram_pairing import TelegramPairer

    settings = getattr(context, "settings", None)
    ws = Path(getattr(settings, "workspace_dir", "."))
    sup = default_supervisor(ws)
    chat = getattr(settings, "chat", None)
    pairer = TelegramPairer(
        sup.registry,
        sessions_root=ws / "data" / "telegram_sessions",
        api_id=getattr(chat, "telegram_api_id", ""),
        api_hash=getattr(chat, "telegram_api_hash", ""))
    personal = PersonalSurfaces(
        sup.registry,
        PersonalVault(ws / "data" / "social" / "personal"))
    svc = {"supervisor": sup, "registry": sup.registry,
           "pairer": pairer, "personal": personal, "workspace": ws}
    # A frozen context still gets a working (uncached) bundle.
    with suppress(Exception):
        context._pairing_web_svc = svc
    return svc


def _find(registry: PairingRegistry, ref: str) -> Any | None:
    """Resolve an account reference the way chat does: the exact
    id, else a unique identity / id-suffix match."""
    ref = (ref or "").strip()
    if not ref:
        return None
    rec = registry.get(ref)
    if rec is not None:
        return rec
    digits = "".join(c for c in ref if c.isdigit())
    matches = [r for r in registry.all()
               if r.identity == digits or r.id.endswith(ref)]
    return matches[0] if len(matches) == 1 else None


# ── the contract: state ─────────────────────────────────────────────

def pairing_state(context: Any, *, svc: dict[str, Any] | None = None,
                  rungs: list[dict[str, Any]] | None = None
                  ) -> dict[str, Any]:
    """The pool as data: counts, accounts with real toggle states,
    live codes and pending Telegram flows, owner-stop intent, and
    the surface rungs this device carries.  Counts are read without
    probing (``counts(probe=False)``, the Python page's posture):
    the state must render even when a bridge is down."""
    svc = svc if svc is not None else _bundle_for_context(context)
    sup = svc["supervisor"]
    registry = svc["registry"]
    pairer = svc["pairer"]
    personal = svc.get("personal")
    from .pairing import toggle_live_verdicts

    accounts: list[dict[str, Any]] = []
    for rec in registry.all():
        # The evidence leg (2026-11-08): the organ's last
        # activity per toggle from the journal, so the web
        # card's "✓ live" carries the same fact as chat /pair.
        activity = None
        if personal is not None:
            try:
                activity = personal.toggle_activity(rec.id)
            except Exception:  # noqa: BLE001 - one leg only
                activity = None
        row: dict[str, Any] = {
            "id": rec.id,
            "platform": rec.platform,
            "identity": rec.identity,
            "label": rec.label,
            "status": rec.status,
            "primary": bool(rec.primary),
            "port": rec.port,
            "toggles": {name: bool(rec.toggle(name))
                        for name in PERSONAL_TOGGLES},
            # Live verdicts on the legs THIS surface can see
            # (registry stop intent + supervisor process): a
            # toggle that is on but not live reads as a warning
            # here exactly as it does in chat /pair — set-state
            # alone cannot tell "off" from "broken".
            "live": toggle_live_verdicts(
                rec,
                stopped=bool(registry.owner_stopped(rec.platform)),
                running=bool(sup.is_running(rec.id)),
                attached=None, connected=None,
                activity=activity),
            "last_error": rec.last_error or "",
            "process": ("running" if sup.is_running(rec.id)
                        else "—"),
            "created_at": rec.created_at,
            "last_seen": rec.last_seen,
        }
        if rec.platform == "whatsapp":
            got = sup.pairing_code(rec.id)
            if got:
                row["pairing_code"] = {"code": got[0],
                                       "age_s": round(got[1], 1)}
            err = sup.pairing_error(rec.id)
            if err:
                row["pairing_error"] = err
        elif rec.platform == "telegram":
            pending = pairer.pending(rec.id)
            if pending:
                row["tg_flow"] = {
                    "state": str(pending.get("state", "")),
                    "expires_in_s": int(
                        pending.get("expires_in_s", 0) or 0),
                }
        accounts.append(row)
    if rungs is None:
        settings = getattr(context, "settings", None)
        rungs = pairing_surface_rungs(
            workspace=getattr(settings, "workspace_dir", None))
    return {
        "ok": True,
        "counts": sup.counts(probe=False),
        "accounts": accounts,
        "owner_stopped": {
            "whatsapp": bool(registry.owner_stopped("whatsapp")),
            "telegram": bool(registry.owner_stopped("telegram")),
        },
        "toggles": list(PERSONAL_TOGGLES),
        "rungs": rungs,
        "generated_at": time.time(),
    }


# ── the contract: actions ───────────────────────────────────────────

def pairing_action(context: Any, payload: dict[str, Any], *,
                   svc: dict[str, Any] | None = None
                   ) -> dict[str, Any]:
    """One pool action, dispatched to the same calls the chat
    handler makes (see ``_control_pair`` — the semantics mirrored
    there are the contract here, including the owner-stop
    bookkeeping: a WhatsApp stop records the platform stop intent,
    a start / restart / fresh pair revokes it, a removal retires
    it).  Every result carries the fresh state, so a caller
    re-renders the truth in one round trip.  Failures come back as
    ``{"ok": False, "error": <plain words>}`` — the supervisor's
    own words, the same ones chat would show the owner."""
    svc = svc if svc is not None else _bundle_for_context(context)
    sup = svc["supervisor"]
    registry = svc["registry"]
    pairer = svc["pairer"]
    action = str((payload or {}).get("action") or "").strip().lower()

    def done(**extra: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": True, "action": action}
        out.update(extra)
        out["state"] = pairing_state(context, svc=svc)
        return out

    def fail(message: str) -> dict[str, Any]:
        return {"ok": False, "action": action, "error": message}

    try:
        if action == "pair_whatsapp":
            number = str(payload.get("number") or "")
            label = str(payload.get("label") or "")
            if not number.strip():
                return fail("usage: pair_whatsapp needs the number "
                            "in international format, e.g. "
                            "2348012345678")
            rec = sup.pair_whatsapp(number, label)
            # A fresh pair is an owner start: any earlier stop
            # order for the platform is revoked (chat parity).
            registry.set_owner_stopped("whatsapp", False)
            return done(account=rec.id,
                        note="bridge launched — the pairing code "
                             "appears in state as soon as WhatsApp "
                             "issues it; approve it on the phone")
        if action == "pair_telegram":
            number = str(payload.get("number") or "")
            label = str(payload.get("label") or "")
            if not number.strip():
                return fail("usage: pair_telegram needs the number "
                            "in international format")
            rec = pairer.start(number, label)
            return done(account=rec.id,
                        note="login code sent to that account — "
                             "answer it here with tgcode")
        if action in ("tgcode", "tgpass"):
            rec = _find(registry, str(payload.get("id") or ""))
            if rec is None:
                return fail("which account? the state lists them")
            secret = str(payload.get("code") or
                         payload.get("password") or "")
            if not secret:
                return fail(f"usage: {action} needs the "
                            f"{'code' if action == 'tgcode' else 'password'}")
            if action == "tgcode":
                out = pairer.confirm_code(rec.id, secret)
            else:
                # The 2FA password rides this one call into the
                # sign-in and nowhere else — never stored, never
                # logged (chat parity).
                out = pairer.confirm_password(rec.id, secret)
            if out.get("status") == "password_needed":
                return done(account=rec.id, password_needed=True,
                            note="that account has two-factor auth "
                                 "on — send its password with tgpass")
            return done(account=rec.id,
                        note="linked — its session file is saved "
                             "under its own name")
        if action in ("start", "stop", "restart", "remove"):
            rec = _find(registry, str(payload.get("id") or ""))
            if rec is None:
                return fail("which account? the state lists them")
            return _lifecycle(svc, action, rec, done, fail)
        if action == "use":
            rec = _find(registry, str(payload.get("id") or ""))
            if rec is None:
                return fail("which account? the state lists them")
            registry.set_primary(rec.id)
            return done(account=rec.id,
                        note=f"{rec.id} is now the default "
                             f"{rec.platform} account")
        if action == "toggle":
            rec = _find(registry, str(payload.get("id") or ""))
            if rec is None:
                return fail("which account? the state lists them")
            name = str(payload.get("name") or "").strip().lower()
            on = payload.get("on")
            if isinstance(on, str):
                on = on.strip().lower() in ("on", "true", "yes", "1")
            svc["personal"].set_toggle(rec.id, name, bool(on))
            return done(account=rec.id,
                        note=f"{name} {'on' if on else 'off'} "
                             f"for {rec.id}")
        return fail(f"unknown pairing action {action!r} — known: "
                    "pair_whatsapp, pair_telegram, tgcode, tgpass, "
                    "start, stop, restart, remove, use, toggle")
    except (PairingUnavailable, ValueError) as exc:
        # The supervisor's / pairer's plain words are the answer,
        # exactly as chat relays them.
        return fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - the flow's own words
        return fail(str(exc))   # are the honest answer


def _lifecycle(svc: dict[str, Any], action: str, rec: Any,
               done: Callable[..., dict[str, Any]],
               fail: Callable[[str], dict[str, Any]]
               ) -> dict[str, Any]:
    """start / stop / restart / remove for one record — the
    ``_control_pair`` branches, verbatim in effect."""
    sup = svc["supervisor"]
    registry = svc["registry"]
    pairer = svc["pairer"]
    if rec.platform == "telegram":
        if action == "remove":
            registry.set_owner_stopped(rec.platform, False)
            sup.remove(rec.id)
            return done(account=rec.id,
                        note="removed from the pool — its session "
                             "file stays on disk")
        if action == "stop":
            registry.set_status(rec.id, "stopped")
            return done(account=rec.id,
                        note="marked stopped — Devon will not "
                             "treat it as connected")
        # start/restart on Telegram = an authorization probe:
        # there is no process to launch.
        state = pairer.probe(rec)
        if state == "linked":
            registry.set_status(rec.id, "linked")
            return done(account=rec.id,
                        note="session authorized, live")
        registry.set_status(rec.id, "down")
        return fail(f"{rec.id} — the saved session no longer "
                    f"authorizes. Re-pair it "
                    f"(pair_telegram {rec.identity})")
    # WhatsApp lifecycle rides the bridge supervisor.
    if action == "start":
        registry.set_owner_stopped("whatsapp", False)
        sup.start(rec.id)
        return done(account=rec.id, note="bridge starting")
    if action == "stop":
        # An account stop is an owner stop of the surface too:
        # the intent is recorded so no reconcile resurrects it.
        registry.set_owner_stopped("whatsapp", True)
        sup.stop(rec.id)
        return done(account=rec.id, note="bridge stopped")
    if action == "restart":
        registry.set_owner_stopped("whatsapp", False)
        sup.stop(rec.id)
        sup.start(rec.id)
        return done(account=rec.id, note="bridge restarted")
    # remove: retire the account AND any stop order that named
    # its platform — a future pair starts clean.
    registry.set_owner_stopped(rec.platform, False)
    sup.remove(rec.id)
    return done(account=rec.id,
                note="removed from the pool — its saved session "
                     "stays on disk")


# ── the rung registry ───────────────────────────────────────────────

def php_page_path() -> Path:
    return PAGE_FILE


def pairing_surface_rungs(*, env: dict | None = None,
                          platform: str | None = None,
                          home: str | Path | None = None,
                          workspace: str | Path | None = None,
                          php: str | None = None,
                          page_file: Path | None = None,
                          runner: Callable[..., tuple[int, str]] | None
                          = None) -> list[dict[str, Any]]:
    """Which pairing surfaces this device carries, best first.
    The profile runtime picks the best AVAILABLE rung; every rung
    states why it is or isn't here — an unavailable rung is a
    fact, never a fake page."""
    page = page_file if page_file is not None else PAGE_FILE
    if php is None:
        php = resolve_php("", env=env, platform=platform, home=home,
                          workspace=workspace)
    rungs: list[dict[str, Any]] = []
    if php and page.is_file():
        version = _php_version(php, runner=runner)
        rungs.append({
            "name": "php-web", "surface": "web", "language": "php",
            "available": True,
            "detail": f"Devon's pairing page in PHP ({version or 'php'}"
                      f" at {php}), served on loopback by Devon",
        })
    else:
        why = ("the page file is missing from this install"
               if php else
               "no PHP runtime resolves on this device — Devon "
               "provisions one when the page is launched "
               "(/pair page), or reports why it cannot")
        rungs.append({
            "name": "php-web", "surface": "web", "language": "php",
            "available": bool(php and page.is_file()),
            "detail": (f"php at {php}, but {why}" if php else why),
        })
    rungs.append({
        "name": "api-page", "surface": "web", "language": "python",
        "available": True,
        "detail": "the read-only render at GET /pair on Devon's "
                  "API, behind the owner token, when the API runs",
    })
    rungs.append({
        "name": "in-chat", "surface": "chat", "language": "python",
        "available": True,
        "detail": "/pair in the owner's DM — the primary surface; "
                  "the whole flow works there with no web rung",
    })
    return rungs


# ── PHP resolution (mirrors resolve_node) ───────────────────────────

def _php_ladder_paths(env: dict, platform: str,
                      home: Path) -> list[Path]:
    """Absolute php candidates outside PATH, most standard first."""
    if platform.startswith("win"):
        out: list[Path] = [Path("C:/php/php.exe")]
        pf = env.get("ProgramFiles")
        if pf:
            out.append(Path(pf) / "PHP" / "php.exe")
        local = env.get("LOCALAPPDATA")
        if local:
            out.append(Path(local) / "Programs" / "PHP" / "php.exe")
        return out
    return [
        Path("/usr/local/bin/php"),
        Path("/usr/bin/php"),
        home / ".local" / "bin" / "php",
        Path("/data/data/com.termux/files/usr/bin/php"),
    ]


def php_managed_root(workspace: str | Path) -> Path:
    """Where Devon's own provisioned PHP lives for a workspace."""
    return Path(workspace) / "tools" / "php"


def php_managed_binary(workspace: str | Path,
                       platform: str | None = None) -> Path:
    platform = ("windows" if os.name == "nt" else "posix") \
        if platform is None else platform
    name = "php.exe" if platform.startswith("win") else "php"
    return php_managed_root(workspace) / name


def resolve_php(explicit: str = "", *,
                env: dict | None = None,
                platform: str | None = None,
                home: str | Path | None = None,
                workspace: str | Path | None = None,
                ladder: list[Path] | None = None) -> str | None:
    """Resolve the php executable, mirroring ``resolve_node``: an
    explicit path (``NM_PHP_BINARY``) wins outright and fails
    honestly when wrong; then PATH; then Devon's managed
    provisioned copy for the workspace; then the platform ladder.
    ``env`` / ``platform`` / ``home`` / ``ladder`` are injectable
    so tests stage a whole machine in a tmp dir."""
    env = os.environ if env is None else env
    platform = ("windows" if os.name == "nt" else "posix") \
        if platform is None else platform
    home = Path.home() if home is None else Path(home)
    explicit = (explicit or env.get("NM_PHP_BINARY") or "").strip()
    if explicit and explicit != "php":
        if Path(explicit).is_file():
            return str(Path(explicit).absolute())
        return shutil.which(explicit, path=env.get("PATH"))
    found = shutil.which("php", path=env.get("PATH"))
    if found:
        return found
    if workspace is not None:
        managed = php_managed_binary(workspace, platform)
        if managed.is_file():
            return str(managed.absolute())
    if ladder is None:
        ladder = _php_ladder_paths(env, platform, home)
    for cand in ladder:
        try:
            if cand.is_file():
                return str(cand.absolute())
        except OSError:
            continue
    return None


def _default_php_runner(argv: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=PHP_PROVISION_TIMEOUT_S)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def _php_version(php: str, *,
                 runner: Callable[..., tuple[int, str]] | None = None
                 ) -> str:
    runner = runner or _default_php_runner
    try:
        rc, out = runner([php, "-v"])
    except OSError:
        return ""
    if rc != 0:
        return ""
    m = re.search(r"PHP (\d+\.\d+\.\d+)", out or "")
    return f"PHP {m.group(1)}" if m else ""


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": f"nomorals-php-provision/"
                                   f"{PHP_VERSION}"})
    with urllib.request.urlopen(req, timeout=300) as resp, \
            open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def _static_php_url(platform: str, machine: str) -> str | None:
    if platform.startswith("win"):
        return _PHP_WIN_URL
    if platform == "darwin":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return (f"{_PHP_STATIC_BASE}/php-{PHP_VERSION}-cli-macos-{arch}"
                ".tar.gz")
    if machine in ("x86_64", "amd64"):
        return (f"{_PHP_STATIC_BASE}/php-{PHP_VERSION}-cli-linux-x86_64"
                ".tar.gz")
    if machine in ("aarch64", "arm64"):
        return (f"{_PHP_STATIC_BASE}/php-{PHP_VERSION}-cli-linux-aarch64"
                ".tar.gz")
    return None


def provision_php(*, env: dict | None = None,
                  platform: str | None = None,
                  home: str | Path | None = None,
                  workspace: str | Path,
                  runner: Callable[..., tuple[int, str]] | None = None,
                  downloader: Callable[[str, Path], None] | None = None,
                  resolver: Callable[..., str | None] | None = None,
                  euid: Any = None,
                  machine: str | None = None) -> str:
    """Install PHP on this machine — Devon's job, not a chore
    handed to the owner (self-provisioning doctrine; mirrors
    ``provision_node``).

    Routes, in order: the pinned official download extracted
    into Devon's managed tools dir for the workspace (Windows:
    the windows.php.net zip; POSIX/macOS: the static-php-cli
    build); winget on Windows; Termux ``pkg`` on a Termux
    profile; ``apt-get`` only when already root.  After any
    route the ladder is re-resolved and the found path returned.
    Every failed or inapplicable route is recorded; when none
    lands a runtime, :class:`PhpProvisionFailed` carries the
    attempts.  All side-effecting seams are injectable so tests
    stage whole machines — no real install ever runs in a test."""
    env = os.environ if env is None else env
    platform = ("windows" if os.name == "nt" else "posix") \
        if platform is None else platform
    home = Path.home() if home is None else Path(home)
    runner = runner or _default_php_runner
    downloader = downloader or _download_file
    resolver = resolver or resolve_php
    if machine is None:
        import platform as _platform_mod
        machine = _platform_mod.machine().lower()
    attempts: list[str] = []

    def _reresolve() -> str | None:
        try:
            return resolver("", env=env, platform=platform,
                            home=home, workspace=workspace)
        except Exception:  # noqa: BLE001 - a resolver quirk is
            return None     # not an install verdict

    def _attempt(label: str, argv: list[str]) -> str | None:
        try:
            rc, output = runner(argv)
        except subprocess.TimeoutExpired:
            attempts.append(f"{label} timed out")
            return None
        except OSError as exc:  # noqa: BLE001 - named attempt
            attempts.append(f"{label} could not run ({exc})")
            return None
        if rc != 0:
            tail = " ".join((output or "").split())[-200:]
            attempts.append(f"{label} failed (exit {rc}): {tail}")
            return None
        found = _reresolve()
        if found:
            return found
        attempts.append(f"{label} ran, but php is still not "
                        "resolvable")
        return None

    # Route 1: the pinned download into Devon's managed dir.
    url = _static_php_url(platform, machine)
    if url is None:
        attempts.append(f"no pinned PHP build for "
                        f"{platform}/{machine}")
    else:
        root = php_managed_root(workspace)
        try:
            root.mkdir(parents=True, exist_ok=True)
            archive = root / ("php-download.zip"
                              if platform.startswith("win")
                              else "php-download.tar.gz")
            downloader(url, archive)
            if platform.startswith("win"):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(root)
            else:
                with tarfile.open(archive) as tf:
                    tf.extractall(root, filter="data")
                binary = php_managed_binary(workspace, platform)
                if binary.is_file():
                    binary.chmod(0o755)
            archive.unlink(missing_ok=True)
            found = _reresolve()
            if found:
                return found
            attempts.append("the pinned download extracted, but "
                            "no php binary resolved from it")
        except Exception as exc:  # noqa: BLE001 - named route
            attempts.append(f"pinned download route failed: {exc}")

    # Route 2: platform package managers (node-provision parity).
    if platform.startswith("win"):
        got = _attempt("winget PHP", ["winget", "install", "-e",
                                      "--id", "PHP.PHP.8.3",
                                      "--accept-source-agreements",
                                      "--accept-package-agreements"])
        if got:
            return got
    elif Path("/data/data/com.termux/files/usr/bin/pkg").is_file() \
            or env.get("TERMUX_VERSION"):
        got = _attempt("termux pkg php", ["pkg", "install", "-y", "php"])
        if got:
            return got
    elif platform == "posix":
        uid = os.geteuid() if euid is None else euid
        if uid == 0:
            got = _attempt("apt-get php-cli",
                           ["apt-get", "install", "-y", "php-cli"])
            if got:
                return got
        else:
            attempts.append("apt-get route skipped — Devon is not "
                            "root here and never sudo-prompts")
    raise PhpProvisionFailed(attempts)


# ── serving the PHP page ────────────────────────────────────────────

def _page_state_path(workspace: str | Path) -> Path:
    return Path(workspace) / "data" / "social" / "pairing_page_server.json"


def _page_token_path(workspace: str | Path) -> Path:
    return Path(workspace) / "data" / "social" / "pairing_page_token"


def _port_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _alloc_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def pairing_page_status(workspace: str | Path) -> dict[str, Any]:
    """Is the PHP pairing page serving for this workspace, and
    where — read from the launch record, probed live.  A stale
    record (server gone) reads as not serving; nothing is
    claimed from the file alone."""
    path = _page_state_path(workspace)
    if not path.is_file():
        return {"serving": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"serving": False}
    port = int(data.get("port") or 0)
    if not port or not _port_listening(port):
        return {"serving": False}
    return {"serving": True, "url": data.get("url", ""),
            "port": port, "php": data.get("php", ""),
            "started_at": data.get("started_at", 0)}


def _read_or_make_page_token(workspace: str | Path) -> str:
    path = _page_token_path(workspace)
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    with suppress(OSError):
        path.chmod(0o600)
    return token


def launch_pairing_page(context: Any, *,
                        spawn: Callable[..., Any] | None = None,
                        env: dict | None = None,
                        platform: str | None = None,
                        home: str | Path | None = None,
                        provision: bool = True) -> dict[str, Any]:
    """Serve the PHP pairing page for the context's workspace on
    loopback, or report the one already serving.  Raises
    :class:`PairingUnavailable` with plain words when the rung
    cannot be served here (no PHP and provisioning failed, the
    page file missing) — the caller relays them, chat-style.

    The spawned server gets three env facts: where Devon's API
    lives (``DEVON_API_BASE``, from this install's api.port),
    the owner's API token (``DEVON_API_TOKEN`` — possibly empty,
    in which case the contract refuses and the page says so),
    and the page's own gate token (``DEVON_PAIR_PAGE_TOKEN``)."""
    settings = getattr(context, "settings", None)
    ws = Path(getattr(settings, "workspace_dir", "."))
    status = pairing_page_status(ws)
    if status.get("serving"):
        return {"url": status["url"], "already": True,
                "token": _read_or_make_page_token(ws),
                "php": status.get("php", "")}
    if not PAGE_FILE.is_file():
        raise PairingUnavailable(
            "the PHP pairing page is missing from this install "
            f"({PAGE_FILE}) — the web rung cannot be served")
    php = resolve_php("", env=env, platform=platform, home=home,
                      workspace=ws)
    if php is None and provision:
        try:
            php = provision_php(env=env, platform=platform,
                                home=home, workspace=ws)
        except PhpProvisionFailed as exc:
            raise PairingUnavailable(
                "no PHP runtime on this device, and Devon's "
                f"provisioning could not land one — {exc}") from exc
    if php is None:
        raise PairingUnavailable(
            "no PHP runtime resolves on this device — the PHP "
            "pairing page cannot be served here yet")
    token = _read_or_make_page_token(ws)
    api = getattr(settings, "api", None)
    api_port = int(getattr(api, "port", 8787) or 8787)
    api_token = str(getattr(api, "token", "") or "")
    port = _alloc_free_port()
    child_env = dict(os.environ)
    child_env.update({
        "DEVON_API_BASE": f"http://127.0.0.1:{api_port}",
        "DEVON_API_TOKEN": api_token,
        "DEVON_PAIR_PAGE_TOKEN": token,
    })
    log_path = ws / "data" / "social" / "pairing_page_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The page's login rides PHP sessions; a bare runtime (the
    # static builds Devon provisions have no session.save_path
    # configured) would fail to store them, so Devon names a
    # workspace-owned session dir explicitly.
    sess_dir = ws / "data" / "social" / "php_sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    spawn = spawn or subprocess.Popen
    with open(log_path, "ab") as log_fh:
        proc = spawn([php, "-d", f"session.save_path={sess_dir}",
                      "-S", f"127.0.0.1:{port}", str(PAGE_FILE)],
                     env=child_env, stdout=log_fh,
                     stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/"
    record = {"pid": getattr(proc, "pid", 0), "port": port,
              "url": url, "php": php, "started_at": time.time()}
    _page_state_path(ws).write_text(
        json.dumps(record, indent=1), encoding="utf-8")
    return {"url": url, "already": False, "token": token,
            "php": php, "pid": record["pid"]}


# ── CLI: python -m nomorals.social.pairing_web ──────────────────────

def _cli(argv: list[str]) -> int:
    """Devon's own launch path outside chat: state the rungs,
    provision PHP, or serve the page for a workspace.  Prints
    JSON — the same shapes the contract returns."""
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(prog="pairing_web")
    parser.add_argument("command",
                        choices=("rungs", "provision-php", "serve",
                                 "status"))
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--api-port", type=int, default=8787)
    parser.add_argument("--api-token", default="")
    args = parser.parse_args(argv)
    ws = Path(args.workspace)
    if args.command == "rungs":
        print(json.dumps(pairing_surface_rungs(workspace=ws),
                         indent=1))
        return 0
    if args.command == "provision-php":
        try:
            php = provision_php(workspace=ws)
        except PhpProvisionFailed as exc:
            print(json.dumps({"ok": False, "error": str(exc),
                              "attempts": exc.attempts}, indent=1))
            return 1
        print(json.dumps({"ok": True, "php": php,
                          "version": _php_version(php)}, indent=1))
        return 0
    context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(ws),
        api=SimpleNamespace(port=args.api_port, token=args.api_token),
        chat=SimpleNamespace(telegram_api_id="",
                             telegram_api_hash="")))
    if args.command == "status":
        print(json.dumps(pairing_page_status(ws), indent=1))
        return 0
    try:
        out = launch_pairing_page(context)
    except PairingUnavailable as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=1))
        return 1
    print(json.dumps({"ok": True, **out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
