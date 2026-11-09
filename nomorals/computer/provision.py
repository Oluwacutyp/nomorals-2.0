"""Self-provisioning for the computer organ (the pairfix pattern).

Whatever the host is missing is Devon's job to fix, never a
command handed to the owner: the desktop bridge binary is FOUND
in the repo's bridge/bin, and if it is absent it is BUILT from the
in-repo Go source with the host's Go toolchain, then verified by
running it. A virtual display (Xvfb) is found, or installed via
the system package manager when Devon has the rights, or reported
in plain words when every route fails. Nothing here instructs the
owner to install anything, ever.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from . import failures


class ProvisionError(Exception):
    """Every provisioning route failed; the message says why, plainly."""


def repo_root() -> Path:
    """The nomorals-2.0 checkout this organ ships in."""
    return Path(__file__).resolve().parents[2]


def bridge_source_dir() -> Path:
    return repo_root() / "bridge" / "desktopbridge"


def bridge_binary_path() -> Path:
    """The committed bridge binary for this platform."""
    name = "desktop-bridge.exe" if sys.platform.startswith("win") \
        else "desktop-bridge-linux"
    return repo_root() / "bridge" / "bin" / name


def find_go() -> str | None:
    """A Go toolchain: the workspace toolchain first, then PATH."""
    candidates = [
        Path.home() / "workspace" / "tools" / "go" / "bin" / "go",
        Path("/usr/local/go/bin/go"),
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return shutil.which("go")


def build_bridge(into: Path | None = None) -> Path:
    """Build the desktop bridge from in-repo source. Returns its path.

    Raises ProvisionError with the plain reason when the source,
    the toolchain, or the build itself is unavailable.
    """
    src = bridge_source_dir()
    if not (src / "go.mod").is_file():
        raise ProvisionError(
            f"the desktop bridge source is missing ({src}) — the repo "
            "checkout is incomplete.")
    go = find_go()
    if go is None:
        raise ProvisionError(
            "no Go toolchain on this host, so the desktop bridge "
            "cannot be built here.")
    target = into or bridge_binary_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CGO_ENABLED"] = "0"
    env["GOOS"] = "windows" if target.name.endswith(".exe") else "linux"
    env["GOARCH"] = "amd64"
    try:
        proc = subprocess.run(
            [go, "build", "-o", str(target), "."],
            cwd=str(src), env=env, capture_output=True, text=True,
            timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvisionError(f"the bridge build did not run: {exc}") from exc
    if proc.returncode != 0 or not target.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise ProvisionError(
            "the bridge build failed: " + (tail[-1] if tail else "no output"))
    os.chmod(target, 0o755)
    return target


@dataclass
class BridgeProvision:
    path: Path
    how: str  # "found" | "built"
    detail: str = ""


def ensure_bridge() -> BridgeProvision:
    """Find the bridge binary, building it when it is missing."""
    override = os.environ.get("NM_COMPUTER_BRIDGE", "").strip()
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return BridgeProvision(path, "found", "NM_COMPUTER_BRIDGE override")
        raise ProvisionError(
            f"NM_COMPUTER_BRIDGE points at {path}, which does not exist.")
    path = bridge_binary_path()
    if path.is_file() and os.access(path, os.X_OK):
        return BridgeProvision(path, "found", "repo bridge/bin")
    try:
        built = build_bridge()
    except ProvisionError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface plainly
        failures.record_failure("provision build_bridge", exc)
        raise ProvisionError(f"the bridge build failed: {exc}") from exc
    return BridgeProvision(built, "built", "built from bridge/desktopbridge")


# ---------------------------------------------------------------- Xvfb

@dataclass
class XvfbProvision:
    display: str
    number: int
    pid: int
    auth_path: Path
    proc: subprocess.Popen[bytes] = field(repr=False, compare=False,
                                          default=None)  # type: ignore[assignment]
    how: str = "started"


def find_xvfb() -> str | None:
    return shutil.which("Xvfb")


def ensure_xvfb_binary() -> str:
    """The Xvfb executable, installing it when Devon has the rights."""
    found = find_xvfb()
    if found:
        return found
    # Self-provisioning route: the system package manager, when we
    # are root (her own hosts) — never a command handed to the owner.
    apt = shutil.which("apt-get")
    if apt and hasattr(os, "geteuid") and os.geteuid() == 0:
        try:
            proc = subprocess.run(
                [apt, "install", "-y", "--no-install-recommends", "xvfb"],
                capture_output=True, text=True, timeout=600,
                env=dict(os.environ, DEBIAN_FRONTEND="noninteractive"))
            if proc.returncode == 0:
                found = find_xvfb()
                if found:
                    return found
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.record_failure("provision apt xvfb", exc)
    raise ProvisionError(
        "this host has no virtual display server (Xvfb) and Devon "
        "could not install one here (no package route with the "
        "rights to use it). The computer needs a display facility: "
        "a real X display, or Xvfb for a virtual one.")


def display_socket_path(number: int) -> Path:
    return Path(f"/tmp/.X11-unix/X{number}")


def find_free_display(start: int = 99, end: int = 199) -> int:
    """First display number with neither a socket nor a lock file."""
    for number in range(start, end + 1):
        if display_socket_path(number).exists():
            continue
        if Path(f"/tmp/.X{number}-lock").exists():
            continue
        return number
    raise ProvisionError("no free virtual display number in 99..199")


def write_xauthority(path: Path, number: int) -> str:
    """Write a one-entry Xauthority file; returns the cookie hex.

    Devon generates the cookie herself and hands it to both Xvfb
    (-auth) and the bridge (XAUTHORITY env): the virtual display
    is hers, credential and all — the FamilyLocal entry format is
    big-endian CARD16 lengths, per the Xauthority spec.
    """
    cookie = secrets.token_bytes(16)
    host = socket.gethostname().encode()

    def field(data: bytes) -> bytes:
        return struct.pack(">H", len(data)) + data

    entry = (
        struct.pack(">H", 256)  # FamilyLocal
        + field(host)
        + field(str(number).encode())
        + field(b"MIT-MAGIC-COOKIE-1")
        + field(cookie)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(entry)
    os.chmod(path, 0o600)
    return cookie.hex()


def _display_accepts(number: int) -> bool:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        sock.connect(str(display_socket_path(number)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def start_xvfb(number: int, width: int, height: int,
               auth_path: Path) -> XvfbProvision:
    """Start Xvfb on ``number`` and wait until it accepts clients."""
    xvfb = ensure_xvfb_binary()
    display = f":{number}"
    cmd = [xvfb, display, "-screen", "0", f"{width}x{height}x24",
           "-auth", str(auth_path), "-nolisten", "tcp"]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise ProvisionError(f"Xvfb would not start: {exc}") from exc
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ProvisionError(
                f"Xvfb exited at once (code {proc.returncode}) for "
                f"display {display} — the display number may be taken.")
        if _display_accepts(number):
            return XvfbProvision(display=display, number=number,
                                 pid=proc.pid, auth_path=auth_path,
                                 proc=proc)
        time.sleep(0.1)
    with suppress(OSError):
        proc.terminate()
    raise ProvisionError(
        f"Xvfb started for display {display} but never accepted "
        "connections within 20s.")


def stop_process(pid: int) -> bool:
    """Terminate a process Devon owns; True when it is gone."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not process_alive(pid):
            return True
        time.sleep(0.05)
    with suppress(OSError):
        os.kill(pid, 9)
    return not process_alive(pid)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    # Reap first when it is our own child: an unreaped zombie still
    # answers kill(pid, 0) but is dead for every practical purpose.
    try:
        done, _status = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except (ChildProcessError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        if stat.rsplit(")", 1)[-1].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return True
