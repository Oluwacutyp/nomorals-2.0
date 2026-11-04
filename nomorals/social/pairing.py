"""Pairing-with-code system (group build §15): the account registry
and the WhatsApp bridge supervisor.

Devon's pool of personal accounts — the reach behind the group and
channel tools — is built by pairing accounts in, one explicit act at
a time:

- ``/pair whatsapp <number>`` registers the number, spawns a
  dedicated bridge process for it (own port, own creds dir) in
  pairing-code mode, and returns the 8-character code WhatsApp
  issues for that number. The owner types it into WhatsApp →
  Linked Devices → Link with phone number instead; the account is
  then a live Devon session.
- Every account lives in ONE registry (WhatsApp numbers today,
  Telegram userbots as they pair) with registered vs connected
  counts — the header the §14 mass tools and the social status
  surface read.
- Supervision is per account: start / stop / restart from chat,
  health probed over the bridge's own protocol.
- TWO bridge engines speak that one protocol: the Go program
  (``bridge/bin/whatsapp-bridge-*``, whatsmeow — the default)
  and the Node/Baileys bridge (``bridge/whatsapp-bridge.mjs``)
  as the second rung.  The supervisor's ``engine`` (or
  ``NM_WHATSAPP_BRIDGE_ENGINE``) picks which one launches; each
  live entry remembers its engine.  Both engines may only ever
  display a pairing code the server confirmed.

Store discipline: the registry is a JSON file written atomically;
an unreadable registry RAISES (quarantined aside) — it is never
silently reset to empty, because an empty registry would orphan
live sessions and forget which numbers are whose.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

_ACCOUNT_ID_RE = re.compile(r"[^a-z0-9]+")
_CODE_LINE_RE = re.compile(r"PAIRING CODE for \+(\d+):\s*(\S+)")
#: The bridge logs a refused pairing-code request in this shape
#: (bridge/whatsapp-bridge.mjs) — the refusal reason rides the
#: line, and the drain captures it instead of swallowing it.
_REFUSAL_RE = re.compile(r"pairing code request failed:\s*(.+)$")
#: The confirmed-code bridges (the Go/whatsmeow program, and the
#: hardened Node bridge) report a refusal in this shape instead:
#: the code request was answered by WhatsApp's server and the
#: answer was no — printed only when no code was registered.
_REFUSED_RE = re.compile(r"PAIRING REFUSED:\s*(.+)$")
#: …and a pairing that was issued but died before the link
#: completed (disconnect / pair failure) in this shape: the
#: confirmed-code bridges print it when an issued code's link
#: dies, and the Node bridge prints it when a socket with a
#: code outstanding closes terminally — the classified reason
#: rides the line, and the drain captures it instead of
#: leaving a corpse code live.
_LINK_FAILED_RE = re.compile(r"PAIRING LINK FAILED:\s*(.+)$")
#: The bridge's log() prefixes an ISO timestamp; display cleaning
#: strips it (and the [bridge] tag) from lines shown to the owner.
_TS_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?\s+")

#: Bounded waits for Devon's self-provisioning: installing the
#: bridge's npm packages gets five minutes; installing Node.js
#: itself (a ~30 MB download plus a system installer) gets ten.
BRIDGE_DEPS_TIMEOUT_S = 300.0
NODE_INSTALL_TIMEOUT_S = 600.0

#: The Node.js LTS the self-installer falls back to when winget
#: cannot do it: a version-pinned official dist build (v22 Jod).
NODE_LTS_VERSION = "22.14.0"
NODE_MSI_URL = (f"https://nodejs.org/dist/v{NODE_LTS_VERSION}/"
                f"node-v{NODE_LTS_VERSION}-x64.msi")


#: How long the bridge-program probe (``--version``) may take
#: before the Go rung is declared not-runnable on this machine.
BRIDGE_PROBE_TIMEOUT_S = 10.0


def bridge_binary_name(os_name: str | None = None) -> str:
    """The committed bridge program for a platform: the Windows
    build for the EC2 host, the Linux build everywhere else.
    (Termux/arm64 is a later rung — not built yet.)"""
    return "whatsapp-bridge.exe" if (os_name or os.name) == "nt" \
        else "whatsapp-bridge-linux"


def bridge_binary_path(repo_root: str | Path | None = None,
                       os_name: str | None = None) -> Path:
    """``<repo>/bridge/bin/<binary>`` — the Go bridge ships as
    ONE self-contained program per platform, committed in the
    repo: no Node, no npm, no install step, nothing to resolve
    at runtime beyond "the file is there and it runs"."""
    root = Path(repo_root) if repo_root is not None else Path(
        __file__).resolve().parents[2]
    return root / "bridge" / "bin" / bridge_binary_name(os_name)


def _binary_missing_message(binary: str | Path) -> str:
    return (
        "WhatsApp pairing is unavailable on this machine: the "
        f"bridge program is missing ({binary}) — it ships "
        "inside Devon as one self-contained file (no Node.js, "
        "no npm, nothing to install). Restore bridge/bin/ "
        "(re-pull the repo) and run /pair again. Telegram "
        "pairing does not need the bridge and still works.")


def _binary_probe_failed_message(binary: str | Path,
                                 detail: str) -> str:
    return (
        "WhatsApp pairing is unavailable on this machine: the "
        f"bridge program exists but does not run here "
        f"({detail}) — {binary}. It is one self-contained file; "
        "if it was damaged in transit, re-pull the repo and run "
        "/pair again. Telegram pairing does not need the bridge "
        "and still works.")


def _default_probe_runner(binary: str) -> tuple[int, str]:
    """Run the bridge program's ``--version`` once: exit code +
    combined output.  Never raises — a spawn failure is a probe
    failure with the OS error as its text."""
    try:
        done = subprocess.run(
            [binary, "--version"], capture_output=True, text=True,
            timeout=BRIDGE_PROBE_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - the failure IS the
        return 1, f"{type(exc).__name__}: {exc}"  # probe's answer
    return done.returncode, (done.stdout or "") + (done.stderr
                                                  or "")


class PairingStoreError(RuntimeError):
    """The registry file exists but cannot be read — never a reset."""


class PairingUnavailable(RuntimeError):
    """WhatsApp pairing cannot run on this machine right now — the
    message IS the user-facing guidance (what is missing, how to
    fix it), so surfaces render it as a plain reply, never as a
    traceback digest.  Raised BEFORE any account record claims to
    be pairing, so the pool never shows a phantom session."""


#: The one plain sentence every surface may show when the bridge
#: cannot launch for want of Node.js.
NODE_MISSING_MESSAGE = (
    "WhatsApp pairing needs Node.js on this machine — no node "
    "executable was found in PATH or in the standard install "
    "locations. Install Node.js LTS (https://nodejs.org) and run "
    "/pair again. Telegram pairing does not need Node and still "
    "works.")


class NodeProvisionFailed(RuntimeError):
    """Devon tried to install Node.js herself and every route
    failed.  ``attempts`` is the plain list of what was tried and
    how each route ended — the supervisor folds it into the
    PairingUnavailable guidance, so the owner is pinged only at
    this genuine wall, with the receipts attached."""

    def __init__(self, attempts: list[str]) -> None:
        self.attempts = list(attempts)
        super().__init__("; ".join(self.attempts))


def _node_install_failed_message(attempts: list[str]) -> str:
    """PairingUnavailable text for a machine with no node where
    the self-install was attempted and failed: what was tried is
    stated, because 'install Node' as a bare chore is exactly
    what the self-provisioning doctrine forbids handing back."""
    tried = "; ".join(attempts) if attempts else \
        "no install route applies on this machine"
    return (
        "WhatsApp pairing needs Node.js on this machine. Devon "
        f"tried to install it herself and could not: {tried}. "
        "Install Node.js LTS (https://nodejs.org) by hand and "
        "run /pair again. Telegram pairing does not need Node "
        "and still works.")


def _output_tail(output: str, limit: int = 280) -> str:
    """The last meaningful lines of a captured command output,
    trimmed for a chat reply — enough to name the failure,
    never a dump."""
    lines = [ln.strip() for ln in (output or "").splitlines()
             if ln.strip()]
    tail = " · ".join(lines[-3:]) if lines else "no output"
    if len(tail) > limit:
        tail = "…" + tail[-(limit - 1):]
    return tail


def _run_capture(argv: list[str], cwd: str | Path | None,
                 timeout: float) -> tuple[int, str]:
    """Run a provisioning command to completion, output captured.
    TimeoutExpired and OSError propagate — callers turn them
    into named attempts / guidance."""
    done = subprocess.run(
        argv, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, timeout=timeout)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def _default_deps_runner(argv: list[str],
                         cwd: Path) -> tuple[int, str]:
    return _run_capture(argv, cwd, BRIDGE_DEPS_TIMEOUT_S)


def _default_install_runner(argv: list[str],
                            cwd: Path | None) -> tuple[int, str]:
    return _run_capture(argv, cwd, NODE_INSTALL_TIMEOUT_S)


def _download_file(url: str, dest: Path) -> None:
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "devon-pairing"})
    with urllib.request.urlopen(req, timeout=180) as resp, \
            open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def provision_node(*, env: dict | None = None,
                   platform: str | None = None,
                   home: str | Path | None = None,
                   runner: Callable[..., tuple[int, str]] | None
                   = None,
                   downloader: Callable[[str, Path], None] | None
                   = None,
                   resolver: Callable[..., str | None] | None
                   = None,
                   euid: Any = None) -> str:
    """Install Node.js on this machine — Devon's job, not a chore
    handed to the owner (self-provisioning doctrine).

    Routes, in order.  Windows: winget when it resolves, then the
    official version-pinned LTS MSI run silently.  POSIX: Termux
    ``pkg`` on a Termux profile (no root needed); ``apt-get`` only
    when the process already runs as root — without privilege the
    route is SKIPPED and recorded, never a sudo prompt.  After
    any successful install the node ladder is re-resolved (the
    MSI lands in C:\\Program Files\\nodejs, ladder location #1)
    and the found path returned.  Every route that fails or does
    not apply is recorded; when none lands a node,
    NodeProvisionFailed carries the attempts.  ``runner`` /
    ``downloader`` / ``resolver`` / ``euid`` are injectable so
    tests stage whole machines — no real install ever runs in a
    test."""
    env = os.environ if env is None else env
    platform = ("windows" if os.name == "nt" else "posix") \
        if platform is None else platform
    home = Path.home() if home is None else Path(home)
    runner = runner or _default_install_runner
    downloader = downloader or _download_file
    resolver = resolver or resolve_node
    attempts: list[str] = []

    def _reresolve() -> str | None:
        try:
            return resolver("", env=env, platform=platform,
                            home=home)
        except Exception:  # noqa: BLE001 - a resolver quirk is
            return None     # not an install verdict

    def _attempt(label: str, argv: list[str]) -> str | None:
        try:
            rc, output = runner(argv, None)
        except subprocess.TimeoutExpired:
            attempts.append(
                f"{label} timed out after "
                f"{NODE_INSTALL_TIMEOUT_S:.0f}s")
            return None
        except OSError as exc:  # noqa: BLE001 - named attempt
            attempts.append(f"{label} could not run ({exc})")
            return None
        if rc != 0:
            attempts.append(
                f"{label} failed (exit {rc}): "
                f"{_output_tail(output)}")
            return None
        found = _reresolve()
        if found:
            return found
        attempts.append(
            f"{label} ran, but node is still not in PATH or any "
            "standard install location")
        return None

    if platform.startswith("win"):
        winget = shutil.which("winget", path=env.get("PATH"))
        if winget:
            found = _attempt(
                "winget install of Node.js LTS "
                "(OpenJS.NodeJS.LTS)",
                [winget, "install", "-e", "--id",
                 "OpenJS.NodeJS.LTS",
                 "--accept-source-agreements",
                 "--accept-package-agreements"])
            if found:
                return found
        else:
            attempts.append("winget is not available on this "
                            "machine")
        label = (f"the official Node.js installer "
                 f"(node-v{NODE_LTS_VERSION}-x64.msi)")
        msi = Path(tempfile.gettempdir()) / \
            f"node-v{NODE_LTS_VERSION}-x64.msi"
        try:
            downloader(NODE_MSI_URL, msi)
        except Exception as exc:  # noqa: BLE001 - named attempt
            attempts.append(
                f"{label} could not be downloaded ({exc})")
        else:
            found = _attempt(
                label, ["msiexec", "/i", str(msi), "/quiet",
                        "/norestart"])
            if found:
                return found
    else:
        prefix = env.get("PREFIX", "")
        if prefix.startswith("/data/data/com.termux") \
                or "TERMUX_VERSION" in env:
            pkg = shutil.which("pkg", path=env.get("PATH"))
            if pkg:
                found = _attempt(
                    "pkg install of nodejs",
                    [pkg, "install", "-y", "nodejs"])
                if found:
                    return found
            else:
                attempts.append("pkg is not available on this "
                                "machine")
        apt = shutil.which("apt-get", path=env.get("PATH"))
        if apt:
            uid = euid() if callable(euid) else euid
            if uid is None and hasattr(os, "geteuid"):
                uid = os.geteuid()
            if uid == 0:
                found = _attempt(
                    "apt-get install of nodejs",
                    [apt, "install", "-y", "nodejs"])
                if found:
                    return found
            else:
                attempts.append(
                    "apt-get install was skipped: Devon is not "
                    "running as root and never asks for a sudo "
                    "password")
    if not attempts:
        attempts.append("no install route applies on this machine")
    raise NodeProvisionFailed(attempts)


def _node_ladder_paths(env: dict, platform: str,
                       home: Path) -> list[Path]:
    """Absolute node candidates outside PATH, most standard first.

    Windows: the nodejs.org installer dirs, the per-user install,
    nvm-windows' ``current`` dir and the ``NVM_SYMLINK`` target.
    POSIX: system package dirs, the nvm shim under the user's
    home, and Termux's prefix (Devon also lives on phones)."""
    if platform.startswith("win"):
        out: list[Path] = []
        pf = env.get("ProgramFiles")
        if pf:
            out.append(Path(pf) / "nodejs" / "node.exe")
        pfx = env.get("ProgramFiles(x86)")
        if pfx:
            out.append(Path(pfx) / "nodejs" / "node.exe")
        local = env.get("LOCALAPPDATA")
        if local:
            out.append(Path(local) / "Programs" / "nodejs"
                       / "node.exe")
        appdata = env.get("APPDATA")
        if appdata:
            out.append(Path(appdata) / "nvm" / "current"
                       / "node.exe")
        symlink = env.get("NVM_SYMLINK")
        if symlink:
            out.append(Path(symlink) / "node.exe")
        return out
    return [
        Path("/usr/local/bin/node"),
        Path("/usr/bin/node"),
        home / ".nvm" / "current" / "bin" / "node",
        Path("/data/data/com.termux/files/usr/bin/node"),
    ]


def resolve_node(explicit: str = "", *,
                 env: dict | None = None,
                 platform: str | None = None,
                 home: str | Path | None = None,
                 ladder: list[Path] | None = None) -> str | None:
    """Resolve the node executable the bridge will launch with.

    An explicit (non-default) ``explicit`` wins outright: it is
    used when it names a real file or resolves on PATH, and the
    resolution fails (None) when it does not — a configured node
    that does not exist is a configuration fact to report, not a
    reason to silently fall back.  Otherwise: ``node`` on PATH,
    then the platform install ladder.  Returns an absolute path,
    or None when this machine has no node anywhere we looked —
    the caller turns that into PairingUnavailable guidance.
    ``env`` / ``platform`` / ``home`` / ``ladder`` are injectable
    so tests can stage a whole machine in a tmp dir."""
    env = os.environ if env is None else env
    explicit = (explicit or "").strip()
    if explicit and explicit != "node":
        if Path(explicit).is_file():
            return str(Path(explicit).absolute())
        return shutil.which(explicit, path=env.get("PATH"))
    found = shutil.which("node", path=env.get("PATH"))
    if found:
        return found
    if ladder is None:
        platform = ("windows" if os.name == "nt" else "posix") \
            if platform is None else platform
        home = Path.home() if home is None else Path(home)
        ladder = _node_ladder_paths(env, platform, home)
    for cand in ladder:
        try:
            if cand.is_file():
                return str(cand)
        except OSError:
            continue
    return None


def resolve_npm(node_path: str = "", *,
                env: dict | None = None) -> str | None:
    """Resolve npm for the bridge dependency install: beside the
    resolved node binary first (npm.cmd / npm.exe / npm — the
    nodejs.org installers always ship npm next to node), then
    PATH.  None when this machine has no npm anywhere we looked."""
    env = os.environ if env is None else env
    node_path = (node_path or "").strip()
    if node_path:
        bindir = Path(node_path).parent
        for name in ("npm.cmd", "npm.exe", "npm"):
            cand = bindir / name
            try:
                if cand.is_file():
                    return str(cand)
            except OSError:
                continue
    return shutil.which("npm", path=env.get("PATH"))


def _clean_bridge_line(line: str) -> str:
    """A bridge output line as the owner should see it: no ISO
    timestamp, no [bridge] tag, no surrounding whitespace."""
    text = _TS_PREFIX_RE.sub("", (line or "").strip())
    if text.startswith("[bridge]"):
        text = text[len("[bridge]"):].strip()
    return text


def _trim_line(text: str, limit: int = 220) -> str:
    if len(text) > limit:
        return text[:limit - 1].rstrip() + "…"
    return text


def _pick_error_line(lines: list[str]) -> str:
    """The one bridge line that names why the process died —
    the module-resolution error when there is one (a fresh
    checkout's missing node_modules speaks exactly this), else
    the last Error line, else the last line that is not stack
    noise.  Cleaned and trimmed; '' when nothing speaks."""
    cleaned = [c for c in (_clean_bridge_line(ln) for ln in lines)
               if c]
    for cand in reversed(cleaned):
        if "Cannot find module" in cand \
                or "Cannot find package" in cand:
            return _trim_line(cand)
    for cand in reversed(cleaned):
        if cand.startswith("panic:"):
            return _trim_line(cand)
    for cand in reversed(cleaned):
        if cand.startswith("Error"):
            return _trim_line(cand)
    for cand in reversed(cleaned):
        if cand.startswith(("at ", "goroutine ", "node:",
                            "Node.js v")):
            continue
        return _trim_line(cand)
    return ""


def account_id_for(platform: str, identity: str) -> str:
    slug = _ACCOUNT_ID_RE.sub("-", identity.strip().lower()).strip("-")
    return f"{platform}:{slug}"


#: Personal-surface toggles every account carries (§15 slices 2–3).
#: All default OFF — the owner enables each surface per account;
#: Devon never starts saving a man's statuses or shadowing his DMs
#: because a default said so.
PERSONAL_TOGGLES = ("status_saving", "view_once", "antidelete")


def default_toggles() -> dict[str, bool]:
    return {name: False for name in PERSONAL_TOGGLES}


@dataclass
class AccountRecord:
    id: str
    platform: str            # "whatsapp" | "telegram"
    identity: str            # WA: international digits; TG: user id/phone
    label: str = ""
    creds_dir: str = ""      # WA: this account's bridge creds directory
    port: int = 0            # WA: this account's bridge port
    status: str = "registered"   # registered|pairing|linked|down|stopped
    created_at: float = 0.0
    last_seen: float = 0.0
    # ── slices 2–3 ──
    session_path: str = ""   # TG: Telethon session file stem (no suffix)
    remote_id: str = ""      # TG: numeric user id once linked
    primary: bool = False    # routing: the platform's named default
    toggles: dict[str, bool] = field(default_factory=default_toggles)
    # ── bridgefix ──
    #: The last bridge failure in plain words ("bridge exited:
    #: Error: Cannot find package …", "WhatsApp refused the code
    #: request: …").  Additive with a default — registries written
    #: before this field load unchanged, and older code reading a
    #: newer file filters it out (from_dict keeps known keys).
    last_error: str = ""

    def toggle(self, name: str) -> bool:
        return bool((self.toggles or {}).get(name, False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccountRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class PairingRegistry:
    """The durable account registry (one JSON file, atomic writes)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._accounts: dict[str, AccountRecord] = {}
        #: Owner stop orders, per platform: platform → when the
        #: owner stopped it.  An account's liveness (linked, bridge
        #: running) is a fact about the pool; this is a fact about
        #: the OWNER's intent, and the two must never be conflated —
        #: the WhatsApp live-attach reconcile resurrecting a surface
        #: the owner had /stop'd (2026-11-04) is what happens when
        #: the reconcile can only see the pool.  Additive top-level
        #: payload key: registries written before it load with no
        #: stops recorded, and older code reading a newer file
        #: ignores it (only "accounts" is ever required).
        self._owner_stopped: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            rows = data.get("accounts") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise ValueError("registry has no accounts list")
            for row in rows:
                rec = AccountRecord.from_dict(row)
                self._accounts[rec.id] = rec
            stops = data.get("owner_stopped")
            if isinstance(stops, dict):
                for platform, ts in stops.items():
                    try:
                        self._owner_stopped[str(platform).strip(
                            ).lower()] = float(ts)
                    except (TypeError, ValueError):
                        continue
        except Exception as exc:  # noqa: BLE001 - quarantine, never reset
            aside = self.path.with_suffix(
                f".corrupt-{int(time.time())}.json")
            try:
                os.replace(self.path, aside)
            except OSError:
                aside = self.path  # type: ignore[assignment]
            raise PairingStoreError(
                f"paired-account registry unreadable ({exc}); "
                f"original moved to {aside} — no accounts were "
                "forgotten silently") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"accounts": [r.to_dict() for r in
                                sorted(self._accounts.values(),
                                       key=lambda r: r.created_at)],
                   "owner_stopped": dict(self._owner_stopped)}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def upsert(self, rec: AccountRecord) -> AccountRecord:
        with self._lock:
            self._accounts[rec.id] = rec
            self._save()
            return rec

    def get(self, account_id: str) -> AccountRecord | None:
        with self._lock:
            return self._accounts.get(account_id)

    def remove(self, account_id: str) -> bool:
        with self._lock:
            if account_id not in self._accounts:
                return False
            del self._accounts[account_id]
            self._save()
            return True

    def all(self) -> list[AccountRecord]:
        with self._lock:
            return sorted(self._accounts.values(),
                          key=lambda r: r.created_at)

    def set_status(self, account_id: str, status: str) -> None:
        with self._lock:
            rec = self._accounts.get(account_id)
            if rec is None:
                return
            rec.status = status
            rec.last_seen = time.time()
            self._save()

    def set_error(self, account_id: str, message: str) -> None:
        """Record (or clear, with '') the account's last bridge
        failure in plain words.  Deliberately separate from
        set_status: an error is evidence about the last attempt,
        not a lifecycle state, and it must not bump last_seen."""
        with self._lock:
            rec = self._accounts.get(account_id)
            if rec is None or rec.last_error == message:
                return
            rec.last_error = message
            self._save()

    def set_owner_stopped(self, platform: str, stopped: bool) -> None:
        """Record (or clear) the owner's explicit stop order for a
        platform.  Set by every owner stop path (/stop <platform>,
        /pair stop) and cleared by every owner start path — the
        durable intent the live-attach reconcile must honor so an
        owner's /stop STICKS across beats and restarts instead of
        being silently undone the next time the pool looks live.
        Persisted with the registry; a no-op write is skipped."""
        platform = (platform or "").strip().lower()
        if not platform:
            return
        with self._lock:
            if stopped:
                if platform not in self._owner_stopped:
                    self._owner_stopped[platform] = time.time()
                    self._save()
            elif platform in self._owner_stopped:
                del self._owner_stopped[platform]
                self._save()

    def owner_stopped(self, platform: str) -> bool:
        """True while an owner stop order stands for *platform*."""
        with self._lock:
            return (platform or "").strip().lower() in \
                self._owner_stopped

    def accounts_for(self, platform: str) -> list[AccountRecord]:
        with self._lock:
            return [r for r in self.all() if r.platform == platform]

    def set_toggles(self, account_id: str, **kw: bool) -> AccountRecord:
        """Flip personal-surface toggles on one account.  Unknown
        toggle names raise — a typo must never read as success."""
        unknown = set(kw) - set(PERSONAL_TOGGLES)
        if unknown:
            raise ValueError(
                f"unknown personal toggle(s): {', '.join(sorted(unknown))} "
                f"— known: {', '.join(PERSONAL_TOGGLES)}")
        with self._lock:
            rec = self._accounts.get(account_id)
            if rec is None:
                raise ValueError(f"no paired account {account_id}")
            toggles = default_toggles()
            toggles.update(rec.toggles or {})
            toggles.update({k: bool(v) for k, v in kw.items()})
            rec.toggles = toggles
            self._save()
            return rec

    def set_primary(self, account_id: str) -> AccountRecord:
        """Name one account its platform's routing default.  Exactly
        one primary per platform — setting a new one clears the old."""
        with self._lock:
            rec = self._accounts.get(account_id)
            if rec is None:
                raise ValueError(f"no paired account {account_id}")
            for other in self._accounts.values():
                if other.platform == rec.platform:
                    other.primary = other.id == rec.id
            self._save()
            return rec


def resolve_account(registry: PairingRegistry, platform: str,
                    account_id: str = "") -> AccountRecord | None:
    """The routing primitive (§15 slice 3): which account speaks for
    a platform.  An explicit id wins (and must belong to the
    platform); otherwise the account the owner named primary;
    otherwise the earliest-created LINKED account (first-connected);
    otherwise the earliest-registered account; otherwise None — the
    platform has no paired account and callers must say so, never
    silently borrow another platform's session."""
    rows = registry.accounts_for(platform)
    if not rows:
        return None
    if account_id:
        for rec in rows:
            if rec.id == account_id:
                return rec
        return None
    for rec in rows:
        if rec.primary:
            return rec
    for rec in rows:
        if rec.status == "linked":
            return rec
    return rows[0]


@dataclass
class _Live:
    proc: Any = None
    code: str = ""
    code_at: float = 0.0
    #: Which bridge engine this process is ("go" | "node") —
    #: relaunches of a live account keep its engine.
    bridge_engine: str = "go"
    reader: threading.Thread | None = field(default=None, repr=False)
    # ── bridgefix: the bridge speaks, the supervisor remembers ──
    #: Bounded tail of the bridge's own output lines — when the
    #: process dies, its last meaningful line is in here.
    lines: deque = field(default_factory=lambda: deque(maxlen=20))
    #: The last failure in display form ("bridge exited: …" /
    #: "WhatsApp refused the code request: …"), '' while healthy.
    last_error: str = ""
    #: The drain saw this process end (set once, under the lock).
    exited: bool = False
    #: When this process was launched (supervisor clock) — the
    #: throttle base for AUTOMATIC relaunches (ensure_running):
    #: a bridge that died seconds after launch is not fork-looped,
    #: while one that lived for hours earns an immediate retry.
    launched_at: float = 0.0


class BridgeSupervisor:
    """One bridge process per WhatsApp account, supervised.

    Spawning is injectable (``spawn``) so tests drive the whole
    lifecycle without node; production uses subprocess.Popen with
    the bridge's stdout drained on a reader thread (the pairing
    code arrives on stdout and on the bridge protocol alike).

    The node executable is RESOLVED, never assumed: PATH first,
    then the platform install ladder (``resolve_node``) — a Devon
    process whose PATH lacks node (a service account, a scheduled
    task) still finds a standard install.  A ``node`` constructor
    arg other than the default overrides the ladder outright.
    Resolution is cached once it succeeds; while it fails it is
    re-checked on every launch, so installing Node fixes /pair
    without restarting Devon.  ``node_resolver`` is injectable
    beside ``spawn`` so tests can stage node-less machines.

    Self-provisioning (the owner's doctrine: a missing tool is
    Devon's job to fix, never a chore handed back): when no node
    resolves, Devon installs Node.js herself once per missing
    state (``provision_node``); before every bridge launch she
    installs the bridge's npm packages when node_modules is
    absent (``_ensure_bridge_deps``).  And the bridge's output
    is never swallowed: the drain keeps a bounded tail, a death
    before linking flips the record to "down" with the bridge's
    own last line as the reason, and a WhatsApp refusal of the
    code request is recorded verbatim; a pairing WhatsApp
    terminates mid-flight (its PAIRING LINK FAILED line) drops
    the dead code and flips the record to "down" with the
    classified reason — ``pairing_error`` is how surfaces
    answer with the truth.  The Node bridge's pairing codes are
    server-confirmed before they are ever printed (the bridge
    sends the stage-1 request through an awaited query, not
    Baileys' fire-and-forget requestPairingCode — see
    bridge/whatsapp-bridge.mjs), and a stage-1 refusal clears
    any buffered code here too, so a code a surface shows is
    a code WhatsApp accepted.

    Engines: ``engine`` picks the bridge program — "go" (the
    default: the self-contained whatsmeow binary under
    bridge/bin/, gated by presence + a ``--version`` probe) or
    "node" (the Node/Baileys bridge, gated by everything above).
    ``NM_WHATSAPP_BRIDGE_ENGINE`` sets the default when no
    engine is passed; a single pairing can override per call.
    Each live entry names the engine that launched it
    (``bridge_engine``).
    """

    #: The bridge engines a launch can use.
    ENGINES = ("go", "node")

    @staticmethod
    def _clean_engine(engine: str | None) -> str:
        eng = (engine or "").strip().lower()
        return eng if eng in BridgeSupervisor.ENGINES else ""

    def __init__(self, registry: PairingRegistry, *,
                 bridge_path: str | Path,
                 sessions_root: str | Path,
                 base_port: int = 8787,
                 node: str = "node",
                 spawn: Callable[..., Any] | None = None,
                 node_resolver: Callable[[], str | None] | None
                 = None,
                 deps_runner: Callable[..., tuple[int, str]] | None
                 = None,
                 npm_resolver: Callable[[str], str | None] | None
                 = None,
                 node_provisioner: Callable[[], str] | None
                 = None,
                 engine: str | None = None,
                 binary_resolver: Callable[[], str | None] | None
                 = None,
                 probe_runner: Callable[..., tuple[int, str]]
                 | None = None) -> None:
        self.registry = registry
        self.bridge_path = Path(bridge_path)
        self.sessions_root = Path(sessions_root)
        self.base_port = int(base_port)
        self.node = node
        # ── engine selection (gobridge) ──
        # The default engine for launches: explicit arg, else
        # the env default, else "go" (the self-contained binary
        # is the primary rung; node is the alternate).
        self.engine = self._clean_engine(engine) \
            or self._clean_engine(
                os.environ.get("NM_WHATSAPP_BRIDGE_ENGINE")) \
            or "go"
        self._binary_resolver = binary_resolver
        self._probe_runner = probe_runner
        self._binary_cache: str | None = None
        self._binary_probe_ok: bool | None = None
        self._binary_probe_detail = ""
        self._spawn = spawn or self._default_spawn
        self._spawn_is_default = spawn is None
        self._node_resolver = node_resolver
        self._node_cache: str | None = None
        self._live: dict[str, _Live] = {}
        self._lock = threading.RLock()
        # ── self-provisioning seams (bridgefix) ──
        # deps_runner / npm_resolver stage the npm dependency
        # install; node_provisioner stages the Node.js install.
        # Injected seams opt INTO provisioning under an injected
        # spawn; production (all defaults) provisions for real.
        self._deps_runner = deps_runner
        self._npm_resolver = npm_resolver
        self._node_provisioner = node_provisioner
        self._deps_lock = threading.Lock()
        # One Node install attempt per supervisor per missing
        # state — a broken environment must not reinstall-loop
        # on every /pair; the recorded attempts ride the failure
        # message from then on.
        self._provision_tried = False
        self._provision_attempts: list[str] = []

    # ── node + launch preflight ─────────────────────────────────────
    def _provisioning_applies(self) -> bool:
        """Whether a missing node triggers Devon's self-install.

        Never over an explicit node configuration (a configured
        path that does not exist is a configuration fact, not a
        missing runtime).  An injected node_provisioner opts in
        explicitly (the test seam).  Otherwise production only:
        default spawn AND the default resolution machinery — an
        injected node_resolver stages the whole machine, install
        attempts included."""
        if (self.node or "").strip() not in ("", "node"):
            return False
        if self._node_provisioner is not None:
            return True
        return self._spawn_is_default and self._node_resolver is None

    def resolve_node_path(self) -> str | None:
        """The node executable launches will use — resolved fresh
        until it succeeds, cached from then on.

        A resolution miss is not the end: when provisioning
        applies, Devon installs Node.js herself ONCE
        (``provision_node`` — winget / official MSI / package
        manager) and the installed path is used from then on.
        A failed attempt is recorded, not retried on every call:
        its attempts ride the PairingUnavailable guidance."""
        if self._node_cache:
            return self._node_cache
        if self._node_resolver is not None:
            found = self._node_resolver()
        else:
            found = resolve_node(self.node)
        if found:
            self._node_cache = str(found)
            return self._node_cache
        if not self._provision_tried and self._provisioning_applies():
            self._provision_tried = True
            provisioner = self._node_provisioner or provision_node
            try:
                installed = provisioner()
            except NodeProvisionFailed as exc:
                self._provision_attempts = list(exc.attempts)
                installed = None
            if installed:
                self._node_cache = str(installed)
                return self._node_cache
            if not self._provision_attempts:
                self._provision_attempts = [
                    "the Node.js installer finished without "
                    "leaving a node Devon can find"]
        return None

    # ── go rung: the bridge program ──────────────────────────────
    def resolve_bridge_binary(self) -> str | None:
        """The bridge program launches will use — re-checked on
        every call while missing (restoring bridge/bin/ fixes
        /pair without restarting Devon), cached once found:
        the same resolution contract the node ladder has."""
        if self._binary_cache:
            return self._binary_cache
        if self._binary_resolver is not None:
            found = self._binary_resolver()
        else:
            path = bridge_binary_path()
            found = str(path) if path.is_file() else None
        if found:
            self._binary_cache = str(found)
            return self._binary_cache
        return None

    def _probe_binary(self, binary: str) -> bool:
        """The program runs: one ``--version`` invocation, its
        verdict cached for the supervisor's life (a program that
        ran once is not re-probed per account).  Production
        (default spawn) always probes; with an injected spawn
        the caller owns the process layer (the test seam) and
        the probe runs only when a probe_runner is injected
        explicitly — the same seam logic as node provisioning."""
        if self._binary_probe_ok is not None:
            return self._binary_probe_ok
        if not self._spawn_is_default and self._probe_runner is None:
            return True
        runner = self._probe_runner or _default_probe_runner
        code, output = runner(binary)
        self._binary_probe_ok = code == 0
        if code != 0:
            tail = _output_tail(output) or f"exit code {code}"
            self._binary_probe_detail = tail
        return self._binary_probe_ok

    def _require_launchable_go(self) -> str:
        """The bridge program argv[0] for a Go launch, or
        PairingUnavailable naming the true blocker: the program
        is missing, or it exists but does not run here.  No
        install step exists to attempt — the binary IS the
        provisioning.  With an injected spawn and no binary
        resolver the caller owns the process layer (the test
        seam): the staged bridge_path stands in as argv[0]."""
        binary = self.resolve_bridge_binary()
        if binary is None:
            if not self._spawn_is_default \
                    and self._binary_resolver is None:
                return str(self.bridge_path)
            raise PairingUnavailable(
                _binary_missing_message(bridge_binary_path()))
        if self._spawn_is_default \
                and not Path(binary).is_file():
            raise PairingUnavailable(
                _binary_missing_message(binary))
        if not self._probe_binary(binary):
            raise PairingUnavailable(_binary_probe_failed_message(
                binary, self._binary_probe_detail))
        return binary

    def _engine_of(self, account_id: str) -> str:
        """The engine a (re)launch of this account uses: the
        live entry's engine while one exists, else the
        supervisor's default."""
        with self._lock:
            live = self._live.get(account_id)
        if live is not None and live.bridge_engine:
            return live.bridge_engine
        return self.engine

    def _require_launchable(self, engine: str | None = None) -> str:
        """The argv[0] for a launch on ``engine`` (the account's
        engine, else the supervisor default), or
        PairingUnavailable naming the true blocker in plain
        words.  Called BEFORE any account record claims to be
        pairing, and again at launch."""
        eng = self._clean_engine(engine) or self.engine
        if eng == "node":
            return self._require_launchable_node()
        return self._require_launchable_go()

    def _require_launchable_node(self) -> str:
        """The node argv[0] for a launch, or PairingUnavailable
        naming the true blocker in plain words.

        Production (default spawn) is strict: no resolvable node
        even after the self-install attempt, or a missing bridge
        script, fails HERE — before any account record claims to
        be pairing.  With an injected spawn the caller owns the
        process layer (the test seam): the resolved path still
        lands in argv when one is found, but a miss falls back to
        the configured name rather than raising, exactly as
        launches behaved before resolution existed."""
        node = self.resolve_node_path()
        if node is None:
            if not self._spawn_is_default \
                    and self._node_provisioner is None:
                return self.node
            explicit = (self.node or "").strip()
            if explicit and explicit != "node":
                raise PairingUnavailable(
                    "WhatsApp pairing is unavailable on this "
                    f"machine: the bridge is configured to run "
                    f"with '{explicit}', and that executable was "
                    "not found. Fix the configured path or install "
                    "Node.js LTS (https://nodejs.org) and run "
                    "/pair again. Telegram pairing does not need "
                    "Node and still works.")
            if self._provision_attempts:
                raise PairingUnavailable(
                    _node_install_failed_message(
                        self._provision_attempts))
            raise PairingUnavailable(NODE_MISSING_MESSAGE)
        if self._spawn_is_default and not self.bridge_path.is_file():
            raise PairingUnavailable(
                "WhatsApp pairing is unavailable on this machine: "
                f"the bridge script is missing ({self.bridge_path}) "
                "— the Devon install is incomplete, this is not a "
                "Node problem. Restore bridge/whatsapp-bridge.mjs "
                "(re-pull the repo) and run /pair again.")
        return node

    # ── bridge dependencies (self-provisioned) ─────────────────────
    def _ensure_bridge_deps(self, node: str) -> None:
        """The bridge's npm packages, installed by Devon herself.

        bridge/node_modules is gitignored, so a fresh checkout
        has none and the bridge dies on its first import — the
        silent death this fixes at the root.  When node_modules
        is absent (and the bridge dir carries a package.json to
        install from), npm runs ONCE per missing state: ``npm
        ci`` when the lockfile exists, else ``npm install``,
        serialized on the deps lock so concurrent launches never
        double-install.  npm unresolvable, a timeout, or a failed
        install raises PairingUnavailable naming what failed
        plus the tail of npm's own output.  With an injected
        spawn and no injected deps_runner, the caller owns its
        staging and this is skipped — the same seam contract as
        node resolution."""
        bridge_dir = self.bridge_path.parent
        if (bridge_dir / "node_modules").is_dir():
            return
        if not (bridge_dir / "package.json").is_file():
            return
        if self._deps_runner is None and not self._spawn_is_default:
            return
        with self._deps_lock:
            if (bridge_dir / "node_modules").is_dir():
                return  # a concurrent launch installed them
            npm = self._npm_resolver(node) \
                if self._npm_resolver is not None \
                else resolve_npm(node)
            if not npm:
                raise PairingUnavailable(
                    "WhatsApp pairing is unavailable on this "
                    "machine: the bridge's packages are not "
                    "installed and npm was not found beside node "
                    f"({node}) or on PATH, so Devon cannot "
                    "install them herself. Run `npm ci` in "
                    f"{bridge_dir} by hand, or reinstall Node.js "
                    "LTS (https://nodejs.org), then run /pair "
                    "again. Telegram pairing does not need the "
                    "bridge and still works.")
            argv = [npm, "ci"] \
                if (bridge_dir / "package-lock.json").is_file() \
                else [npm, "install"]
            runner = self._deps_runner or _default_deps_runner
            try:
                rc, output = runner(argv, bridge_dir)
            except subprocess.TimeoutExpired:
                raise PairingUnavailable(
                    "WhatsApp pairing is unavailable on this "
                    "machine: installing the bridge's packages "
                    f"(`{' '.join(argv)}` in {bridge_dir}) timed "
                    f"out after {BRIDGE_DEPS_TIMEOUT_S:.0f}s. "
                    "Check the machine's network and run /pair "
                    "again. Telegram pairing does not need the "
                    "bridge and still works.") from None
            except OSError as exc:  # noqa: BLE001 - named, plain
                raise PairingUnavailable(
                    "WhatsApp pairing is unavailable on this "
                    "machine: Devon could not run npm to install "
                    f"the bridge's packages ({exc}). Run "
                    f"`{' '.join(argv)}` in {bridge_dir} by "
                    "hand, then run /pair again. Telegram "
                    "pairing does not need the bridge and still "
                    "works.") from exc
            if rc != 0 or not (bridge_dir / "node_modules").is_dir():
                raise PairingUnavailable(
                    "WhatsApp pairing is unavailable on this "
                    "machine: the bridge's packages are not "
                    "installed, and Devon's own install — "
                    f"`{' '.join(argv)}` in {bridge_dir} — "
                    f"failed (exit {rc}): {_output_tail(output)}. "
                    "Check the machine's network, or run that "
                    "command by hand, then run /pair again. "
                    "Telegram pairing does not need the bridge "
                    "and still works.")

    # ── process plumbing ────────────────────────────────────────────
    @staticmethod
    def _default_spawn(argv: list[str], env: dict[str, str]) -> Any:
        return subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _alloc_port(self) -> int:
        used = {r.port for r in self.registry.all() if r.port}
        port = self.base_port
        while port in used:
            port += 1
        return port

    def _launch(self, rec: AccountRecord, *, pairing_number: str = "",
                engine: str | None = None) -> _Live:
        eng = self._clean_engine(engine) or self._engine_of(rec.id)
        program = self._require_launchable(eng)
        creds = Path(rec.creds_dir)
        creds.mkdir(parents=True, exist_ok=True)
        media = creds.parent / (creds.name + "-media")
        if eng == "node":
            # The Node rung keeps its self-provisioning: npm
            # packages installed by Devon before the launch.
            self._ensure_bridge_deps(program)
            argv = [program, str(self.bridge_path),
                    "--port", str(rec.port),
                    "--creds", str(creds),
                    "--media", str(media)]
        else:
            # The Go rung IS the program: same flags, same env,
            # same JSON-lines contract — nothing to install.
            argv = [program,
                    "--port", str(rec.port),
                    "--creds", str(creds),
                    "--media", str(media)]
        env = dict(os.environ)
        if pairing_number:
            env["WA_PAIRING_CODE"] = "1"
            env["WA_PAIRING_NUMBER"] = pairing_number
        try:
            proc = self._spawn(argv, env)
        except OSError as exc:  # noqa: BLE001 - named, not a digest
            if eng == "node":
                raise PairingUnavailable(
                    "the WhatsApp bridge process could not be "
                    f"started ({exc}) — node resolved to "
                    f"{program}, bridge at {self.bridge_path}. "
                    "Nothing was paired; fix the cause and run "
                    "/pair again.") from exc
            raise PairingUnavailable(
                "the WhatsApp bridge process could not be "
                f"started ({exc}) — bridge program at {program}. "
                "Nothing was paired; fix the cause and run "
                "/pair again.") from exc
        live = _Live(proc=proc, bridge_engine=eng,
                     launched_at=time.time())
        # register BEFORE the reader starts: a fast bridge can emit
        # its pairing code immediately, and a code that arrives before
        # the live entry exists would be dropped on the floor.
        with self._lock:
            self._live[rec.id] = live
        # A fresh launch supersedes the last attempt's failure —
        # the stored error clears so surfaces report THIS attempt.
        current = self.registry.get(rec.id)
        if current is not None and current.last_error:
            self.registry.set_error(rec.id, "")
        reader = threading.Thread(
            target=self._drain, args=(rec.id, proc),
            name=f"pair-reader-{rec.id}", daemon=True)
        live.reader = reader
        reader.start()
        return live

    def _drain(self, account_id: str, proc: Any) -> None:
        stream = getattr(proc, "stdout", None)
        if stream is None:
            return
        try:
            for line in stream:
                text = (line or "").rstrip("\n")
                with self._lock:
                    live = self._live.get(account_id)
                    if live is not None and live.proc is proc:
                        live.lines.append(text)
                m = _CODE_LINE_RE.search(text)
                if m:
                    with self._lock:
                        live = self._live.get(account_id)
                        if live is not None:
                            live.code = m.group(2)
                            live.code_at = time.time()
                    self.registry.set_status(account_id, "pairing")
                f = _REFUSED_RE.search(text) \
                    or _REFUSAL_RE.search(text)
                if f:
                    # WhatsApp said no at stage 1 — the reason is
                    # the answer /pair code owes the owner, not
                    # "no live code".  Recorded on the live entry
                    # AND the registry, and any code this attempt
                    # buffered is dropped with it: a refusal means
                    # no code from this attempt is valid.  The
                    # status stays "pairing" (the bridge lives;
                    # a fresh /pair retries).
                    self._note_request_refused(
                        account_id, proc,
                        "WhatsApp refused the code request: "
                        + f.group(1).strip())
                lf = _LINK_FAILED_RE.search(text)
                if lf:
                    # The code was issued but the link died
                    # before WhatsApp confirmed it: WhatsApp
                    # closed the pairing itself, the issued code
                    # died with the session, and the bridge
                    # parks — no fresh code is coming from this
                    # attempt.  Record the reason verbatim, drop
                    # the corpse code, flip the record to "down"
                    # (linked/stopped records untouched); a
                    # fresh /pair whatsapp relaunches the bridge
                    # and is the recovery.
                    self._note_pairing_dead(
                        account_id, proc,
                        "WhatsApp closed the pairing: "
                        + lf.group(1).strip())
                if "connected as" in text:
                    self.registry.set_status(account_id, "linked")
        except Exception:  # noqa: BLE001 - a dead stream ends the
            pass                    # drain; the exit is handled
        self._finish_dead(account_id, proc)

    def _note_live_error(self, account_id: str, proc: Any,
                         text: str) -> None:
        """Record a bridge-reported failure on the live entry and
        the registry (surfaces read whichever survives)."""
        with self._lock:
            live = self._live.get(account_id)
            if live is not None and live.proc is proc:
                live.last_error = text
        self.registry.set_error(account_id, text)

    def _note_request_refused(self, account_id: str, proc: Any,
                              text: str) -> None:
        """A stage-1 code-request refusal: WhatsApp evaluated the
        request and said no, so NO code from this attempt is
        valid — clear any buffered code from the live entry
        before recording the reason, so a later /pair code can
        never display one.  The bridge's confirmed request path
        never prints a code ahead of the server's answer, so in
        the normal run there is nothing to clear; this is the
        supervisor-side half of the same guarantee, and it also
        covers a refusal that lands on an attempt which somehow
        did show one.  The record stays "pairing" — the bridge
        lives and a fresh /pair retries."""
        with self._lock:
            live = self._live.get(account_id)
            if live is not None and live.proc is proc:
                live.code = ""
                live.code_at = 0.0
        self._note_live_error(account_id, proc, text)

    def _note_pairing_dead(self, account_id: str, proc: Any,
                           text: str) -> None:
        """A pairing WhatsApp terminated mid-flight: the issued
        code is a corpse, so clear it from the live entry (a
        /pair code consult must fall through to the reason, not
        re-show the dead code), record the reason, and flip the
        record to "down" — the bridge is parked, so nothing
        about this attempt is still in flight.  A record that
        meanwhile linked or was stopped is not this line's
        business and keeps its state."""
        with self._lock:
            live = self._live.get(account_id)
            if live is not None and live.proc is proc:
                live.code = ""
                live.code_at = 0.0
        self._note_live_error(account_id, proc, text)
        rec = self.registry.get(account_id)
        if rec is not None and rec.status not in ("linked",
                                                  "stopped"):
            self.registry.set_status(account_id, "down")

    def _finish_dead(self, account_id: str, proc: Any) -> None:
        """The drain ended: if the process really exited before
        the account linked, stop pretending it is pairing.

        The record flips to "down" and the bridge's own last
        meaningful line becomes the stored reason — "bridge
        exited: Error: Cannot find package '@whiskeysockets/
        baileys'…", "bridge exited: panic: …" — instead of a
        silent record and a bare
        "no live code".  A stream that merely ended while the
        process lives on (poll says so) changes nothing.  A
        LINKED account's dead bridge is supervision's domain
        (probe + relaunch); its state and error are left alone
        here.  Stopped/replaced entries are not this proc's
        business — the identity check sees to that."""
        with self._lock:
            live = self._live.get(account_id)
            if live is None or live.proc is not proc \
                    or live.exited:
                return
        # The pipe closing precedes the exit code by a hair on
        # real processes; give poll() a short grace before
        # believing the process is gone.
        exited = False
        for _ in range(25):
            try:
                exited = proc.poll() is not None
            except Exception:  # noqa: BLE001 - unpollable = gone
                exited = True
            if exited:
                break
            time.sleep(0.02)
        if not exited:
            return
        with self._lock:
            live = self._live.get(account_id)
            if live is None or live.proc is not proc \
                    or live.exited:
                return
            live.exited = True
            lines = list(live.lines)
            have_error = bool(live.last_error)
        rec = self.registry.get(account_id)
        if rec is None or rec.status == "linked":
            return
        if not have_error:
            line = _pick_error_line(lines)
            if line:
                self._note_live_error(
                    account_id, proc, f"bridge exited: {line}")
        if rec.status != "down":
            self.registry.set_status(account_id, "down")

    def _reap(self, account_id: str) -> None:
        """Consult-time death check: a bridge found dead when its
        code/status is asked after gets the same truth-telling as
        a drain-observed death.  Idempotent — the exited flag and
        the proc identity check do the guarding."""
        with self._lock:
            live = self._live.get(account_id)
        if live is None or live.proc is None or live.exited:
            return
        try:
            if live.proc.poll() is None:
                return
        except Exception:  # noqa: BLE001
            return
        self._finish_dead(account_id, live.proc)

    def pairing_error(self, account_id: str) -> str:
        """The account's last bridge failure in plain words, or
        '' — the live entry's word first (freshest), then the
        registry's stored one (survives restarts)."""
        self._reap(account_id)
        with self._lock:
            live = self._live.get(account_id)
            if live is not None and live.last_error:
                return live.last_error
        rec = self.registry.get(account_id)
        return rec.last_error if rec is not None else ""

    def _alive(self, account_id: str) -> bool:
        with self._lock:
            live = self._live.get(account_id)
        if live is None or live.proc is None:
            return False
        try:
            return live.proc.poll() is None
        except Exception:  # noqa: BLE001
            return False

    def is_running(self, account_id: str) -> bool:
        """Public view of the process table (supervision reads it)."""
        return self._alive(account_id)

    def _port_listening(self, rec: AccountRecord) -> bool:
        """Something accepts TCP on the record's bridge port.

        A bridge Devon did not spawn — one that survived a Devon
        restart, or a hand-run bridge — still counts as serving:
        it is adopted, never double-launched (a second process
        could not bind the port anyway, and two bridges on one
        account would fight over the session)."""
        if not rec.port:
            return False
        try:
            with socket.create_connection(("127.0.0.1", rec.port),
                                          timeout=1.0):
                return True
        except OSError:
            return False

    def bridge_serving(self, rec: AccountRecord) -> bool:
        """Is this account's bridge reachable right now — a
        process this supervisor runs, or anything listening on
        the record's port?  The truth the honest surfaces read
        (/platforms, the live-attach reconcile) before anyone
        claims a WhatsApp surface is up."""
        return self._alive(rec.id) or self._port_listening(rec)

    def ensure_running(self, account_id: str, *,
                       relaunch_interval: float = 300.0
                       ) -> tuple[bool, str]:
        """Make sure a bridge process serves this account, or
        say why none can — the AUTOMATIC ensure the surface
        lifecycle rides (boot reconcile, supervision beats,
        surface starts).  Owner-explicit starts (``start()``)
        never pass through here and are never throttled.

        Returns ``(True, "")`` when a bridge serves: already
        running under this supervisor, listening on the record's
        port (a survivor — adopted, never double-launched), or
        launched here through the same ``start()`` every other
        path uses (per-engine preflight included: the Go binary
        probe, the node ladder + npm deps).

        Returns ``(False, reason)`` in plain words when none
        can: the launch blocker verbatim (PairingUnavailable),
        a stopped account (an owner's stop is never an automatic
        path's business to undo), or a death still inside its
        relaunch throttle window — a bridge that died seconds
        after launch is not fork-looped; its own last line (via
        pairing_error) is the reason when it left one."""
        rec = self.registry.get(account_id)
        if rec is None:
            return False, f"no paired account {account_id}"
        if rec.platform != "whatsapp":
            return False, (f"{account_id} is a {rec.platform} "
                           "account — it has no WhatsApp bridge")
        if self._alive(account_id):
            return True, ""
        if rec.status == "stopped":
            return False, (f"{rec.id} is stopped — start it "
                           f"with /pair start {rec.id}")
        if not rec.port:
            return False, (f"{rec.id} has no bridge port "
                           "recorded — re-pair the account")
        if self._port_listening(rec):
            return True, ""
        with self._lock:
            live = self._live.get(account_id)
        if live is not None and relaunch_interval > 0 and \
                time.time() - live.launched_at < relaunch_interval:
            reason = self.pairing_error(account_id)
            return False, reason or (
                "the bridge process exited and its relaunch is "
                "throttled — it will be retried shortly")
        try:
            self.start(account_id)
        except PairingUnavailable as exc:
            return False, str(exc)
        return True, ""

    # ── pairing ─────────────────────────────────────────────────────
    def pair_whatsapp(self, number: str, label: str = "",
                      *, engine: str | None = None) -> AccountRecord:
        digits = re.sub(r"\D", "", number or "")
        if len(digits) < 7:
            raise ValueError(
                "that doesn't look like a phone number — give the "
                "number in international format, e.g. 2348012345678")
        eng = self._clean_engine(engine) or self.engine
        # Preflight BEFORE the record exists: a machine that cannot
        # launch the chosen bridge engine fails here, so the
        # registry never gains a record claiming "pairing" with
        # no process behind it.
        self._require_launchable(eng)
        rec_id = account_id_for("whatsapp", digits)
        rec = self.registry.get(rec_id)
        created = rec is None
        prior_status = "" if created else rec.status
        if created:
            creds = self.sessions_root / f"wa-{digits}"
            rec = AccountRecord(
                id=rec_id, platform="whatsapp", identity=digits,
                label=(label or "").strip(), creds_dir=str(creds),
                port=self._alloc_port(), status="pairing",
                created_at=time.time())
            self.registry.upsert(rec)
        else:
            rec.status = "pairing"
            if label:
                rec.label = label.strip()
            self.registry.upsert(rec)
        was_running = self._alive(rec.id)
        if was_running:
            self.stop(rec.id)
        try:
            self._launch(rec, pairing_number=digits, engine=eng)
        except PairingUnavailable:
            # Roll the record back to the truth: a brand-new pairing
            # that never launched leaves no record at all; an
            # existing account returns to its prior state ("down"
            # when this attempt stopped its live bridge to re-pair
            # — the sweep's probes own its state from there).
            if created:
                self.registry.remove(rec.id)
            elif was_running:
                self.registry.set_status(rec.id, "down")
            else:
                self.registry.set_status(rec.id,
                                         prior_status or "registered")
            raise
        return rec

    def pairing_code(self, account_id: str) -> tuple[str, float] | None:
        """(code, age_seconds) once the bridge has issued one."""
        self._reap(account_id)
        with self._lock:
            live = self._live.get(account_id)
            if live is None or not live.code:
                return None
            return live.code, max(0.0, time.time() - live.code_at)

    # ── lifecycle ───────────────────────────────────────────────────
    def start(self, account_id: str) -> AccountRecord:
        rec = self.registry.get(account_id)
        if rec is None:
            raise ValueError(f"no paired account {account_id}")
        if rec.platform != "whatsapp":
            raise ValueError(
                f"{account_id} is a {rec.platform} account — its "
                "session is supervised by its own adapter, not the "
                "WhatsApp bridge")
        if not self._alive(account_id):
            self._launch(rec)
        return rec

    def stop(self, account_id: str) -> bool:
        with self._lock:
            live = self._live.pop(account_id, None)
        if live is None or live.proc is None:
            return False
        try:
            live.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        self.registry.set_status(account_id, "stopped")
        return True

    def remove(self, account_id: str) -> bool:
        self.stop(account_id)
        return self.registry.remove(account_id)

    # ── health ──────────────────────────────────────────────────────
    def probe(self, rec: AccountRecord, timeout: float = 2.0) -> str:
        """Live session state over the bridge protocol: the bridge
        answers a status request with its connection state."""
        if rec.platform != "whatsapp" or not rec.port:
            return rec.status
        try:
            with socket.create_connection(("127.0.0.1", rec.port),
                                          timeout=timeout) as sock:
                sock.sendall(b'{"id":"probe","cmd":"status"}\n')
                sock.settimeout(timeout)
                buf = b""
                while b"\n" not in buf and len(buf) < 65536:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            for line in buf.decode("utf-8", "replace").splitlines():
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("type") == "status":
                    return "linked" if obj.get("state") == "open" \
                        else "down"
                if obj.get("id") == "probe" and "state" in obj:
                    return "linked" if obj.get("state") == "open" \
                        else "down"
            return "down"
        except OSError:
            return "down"

    def status_all(self, *, probe: bool = True) -> list[dict[str, Any]]:
        for rec in self.registry.all():
            if rec.platform == "whatsapp":
                self._reap(rec.id)
        out = []
        for rec in self.registry.all():
            row = rec.to_dict()
            with self._lock:
                live = self._live.get(rec.id)
                if live is not None and live.last_error:
                    # the live entry's word is fresher than the
                    # stored one (a death this process witnessed)
                    row["last_error"] = live.last_error
            row["process"] = "running" if self._alive(rec.id) else "—"
            if probe and rec.platform == "whatsapp" \
                    and row["process"] == "running":
                row["status"] = self.probe(rec)
            out.append(row)
        return out

    def counts(self, *, probe: bool = True) -> dict[str, int]:
        rows = self.status_all(probe=probe)
        return {
            "registered": len(rows),
            "connected": sum(1 for r in rows
                             if r.get("status") == "linked"),
            # dropped: known to the registry, believed linked once,
            # currently unreachable — the supervision sweep's quarry.
            "dropped": sum(1 for r in rows
                           if r.get("status") == "down"),
        }


class PoolSupervision:
    """The supervision sweep (§15): one pass over the registry that
    flips account states to what is TRUE right now.

    - WhatsApp: an account believed linked whose bridge process is
      gone gets probed over the bridge protocol (a bridge Devon did
      not spawn — e.g. surviving a Devon restart — still answers).
      Probe says down → the record flips to ``down`` (dropped) and
      the bridge is relaunched through the supervisor's own start —
      the bridge's internal reconnect policy then owns the session;
      the sweep never writes "linked" itself, the next probe earns
      it back.  Relaunches are throttled per account so a bridge
      that keeps dying is not fork-bombed.
    - Telegram: a linked account is probed through the pairer's
      session prober (throttled — an MTProto authorization check is
      a network round-trip).  Unauthorized → ``down``.  A down
      Telegram account is NEVER auto-relaunched: re-login needs a
      fresh code from the owner, and pretending otherwise would be
      a fake "connected".
    - Pairing flows that outlived their TTL are expired by the
      pairer as part of the same pass.

    ``pairer`` is duck-typed (probe / expire_flows) so this module
    never imports the Telethon layer.
    """

    def __init__(self, registry: PairingRegistry,
                 supervisor: "BridgeSupervisor | None" = None,
                 pairer: Any = None, *,
                 clock: Callable[[], float] = time.time,
                 tg_probe_interval: float = 300.0,
                 relaunch_interval: float = 300.0) -> None:
        self.registry = registry
        self.supervisor = supervisor
        self.pairer = pairer
        self._clock = clock
        self.tg_probe_interval = float(tg_probe_interval)
        self.relaunch_interval = float(relaunch_interval)
        self._tg_probed_at: dict[str, float] = {}
        self._relaunched_at: dict[str, float] = {}

    def sweep(self) -> list[dict[str, Any]]:
        """Run one pass; returns the list of state changes made."""
        changes: list[dict[str, Any]] = []
        if self.pairer is not None:
            try:
                for account_id in self.pairer.expire_flows():
                    changes.append({"account": account_id,
                                    "change": "pairing flow expired"})
            except Exception:  # noqa: BLE001 - expiry is best-effort
                pass
        for rec in self.registry.all():
            if rec.platform == "whatsapp" and self.supervisor is not None:
                change = self._sweep_whatsapp(rec)
            elif rec.platform == "telegram" and self.pairer is not None:
                change = self._sweep_telegram(rec)
            else:
                change = None
            if change:
                changes.append(change)
        return changes

    def _sweep_whatsapp(self, rec: AccountRecord) -> dict[str, Any] | None:
        sup = self.supervisor
        assert sup is not None
        if rec.status == "stopped":
            return None
        if sup.is_running(rec.id):
            return None  # the bridge's own policy owns a live process
        probed = sup.probe(rec)
        if probed == "linked":
            if rec.status != "linked":
                self.registry.set_status(rec.id, "linked")
                return {"account": rec.id,
                        "change": "probe found the bridge linked"}
            return None
        change = None
        if rec.status != "down":
            self.registry.set_status(rec.id, "down")
            change = {"account": rec.id,
                      "change": f"{rec.status} → down (bridge gone)"}
        now = self._clock()
        last = self._relaunched_at.get(rec.id, 0.0)
        if now - last >= self.relaunch_interval:
            self._relaunched_at[rec.id] = now
            try:
                sup.start(rec.id)
                change = {"account": rec.id,
                          "change": "dropped — bridge relaunched; "
                                    "its reconnect policy owns the "
                                    "session until a probe earns "
                                    "'linked' back"}
            except PairingUnavailable as exc:  # noqa: BLE001
                # The machine cannot run the bridge at all (no
                # node): say so by name, once per throttled attempt
                # — never a crash-loop, never a bare "failed".
                change = {"account": rec.id,
                          "change": f"relaunch blocked: {exc}"}
            except Exception as exc:  # noqa: BLE001
                change = {"account": rec.id,
                          "change": f"relaunch failed: {exc}"}
        return change

    def _sweep_telegram(self, rec: AccountRecord) -> dict[str, Any] | None:
        if rec.status != "linked":
            return None
        now = self._clock()
        last = self._tg_probed_at.get(rec.id, 0.0)
        if now - last < self.tg_probe_interval:
            return None
        self._tg_probed_at[rec.id] = now
        try:
            state = self.pairer.probe(rec)
        except Exception:  # noqa: BLE001 - a failed probe is not a
            return None                 # verdict; state stays as-is
        if state != "linked" and rec.status != "down":
            self.registry.set_status(rec.id, "down")
            return {"account": rec.id,
                    "change": "linked → down (session probe: "
                              "not authorized)"}
        return None


def default_supervisor(workspace_dir: str | Path) -> BridgeSupervisor:
    """The production supervisor rooted at the runtime workspace:
    registry under data/social/, sessions under data/whatsapp_sessions/,
    bridge script at the repo's bridge/whatsapp-bridge.mjs."""
    root = Path(workspace_dir)
    registry = PairingRegistry(root / "data" / "social"
                               / "paired_accounts.json")
    repo_root = Path(__file__).resolve().parents[2]
    return BridgeSupervisor(
        registry,
        bridge_path=repo_root / "bridge" / "whatsapp-bridge.mjs",
        sessions_root=root / "data" / "whatsapp_sessions")
