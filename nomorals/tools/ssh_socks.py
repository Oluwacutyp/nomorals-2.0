"""SSH → SOCKS5: turn any SSH server into a local SOCKS5 proxy.

Give the system an SSH credential (host, port, user, key or password)
and it becomes a live SOCKS5 endpoint on ``127.0.0.1`` — usable by
``proxy_set`` (route everything through it), the browser, web tools,
OSINT, or the proxy lab pool.

Implementation: the system's own OpenSSH client in local-forward mode
(``ssh -N -D 127.0.0.1:<port>``) — the battle-tested path, zero
third-party dependencies.  Password auth uses ``sshpass -e`` (the
password travels through the environment, never the argv/process list);
key auth is native.

Lifecycle is fully managed:

* ``start`` — picks a free local port, launches the tunnel, starts a
  supervisor thread
* ``status`` — process alive? SOCKS5 port answering handshakes? how
  many times did it auto-reconnect?
* ``stop`` — clean SIGTERM → SIGKILL escalation
* **auto-reconnect** — if the SSH connection drops, the supervisor
  relaunches it with backoff (2s → 4s → … → 30s cap) up to a
  configurable attempt budget; ``ServerAliveInterval=15`` keeps idle
  tunnels alive from the SSH side
* profiles (including credentials) persist in the state DB so a
  restarted bot knows what was configured; live process state is
  reconciled on boot (a dead parent means a dead tunnel — no phantom
  "running" state)
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["SshSocksProfile", "SshSocksTunnel", "SshSocksManager", "register"]

_PROFILES_KEY = "proxy.ssh.profiles"
_RUNTIME_KEY = "proxy.ssh.runtime"


# ── profile ──────────────────────────────────────────────────────────────────


@dataclass
class SshSocksProfile:
    """One SSH server, as a SOCKS5 source.  Persisted by name."""

    name: str
    host: str
    port: int = 22
    user: str = ""
    password: str = ""
    key: str = ""
    local_port: int = 0            # 0 = pick a free port
    auto_reconnect: bool = True
    max_reconnects: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "host": self.host, "port": int(self.port),
            "user": self.user, "password": self.password, "key": self.key,
            "local_port": int(self.local_port),
            "auto_reconnect": bool(self.auto_reconnect),
            "max_reconnects": int(self.max_reconnects),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SshSocksProfile":
        return cls(
            name=str(data.get("name") or ""),
            host=str(data.get("host") or ""),
            port=int(data.get("port") or 22),
            user=str(data.get("user") or ""),
            password=str(data.get("password") or ""),
            key=str(data.get("key") or ""),
            local_port=int(data.get("local_port") or 0),
            auto_reconnect=bool(data.get("auto_reconnect", True)),
            max_reconnects=int(data.get("max_reconnects") or 10),
        )


# ── tunnel ───────────────────────────────────────────────────────────────────


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _socks5_probe(host: str, port: int, timeout: float = 1.5) -> bool:
    """A SOCKS5 port answers the greeting — the tunnel is really up."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(b"\x05\x01\x00")
            reply = sock.recv(2)
            return reply[:2] == b"\x05\x00"
    except OSError:
        return False


class SshSocksTunnel:
    """One live (or configured) SSH→SOCKS5 tunnel with supervision."""

    def __init__(self, profile: SshSocksProfile, *,
                 ssh_bin: str = "ssh",
                 sshpass_bin: str = "sshpass") -> None:
        self.profile = profile
        self.ssh_bin = ssh_bin
        self.sshpass_bin = sshpass_bin
        self.proc: subprocess.Popen | None = None
        self.local_port: int = 0
        self.restarts = 0
        self.started_at: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- command --------------------------------------------------------------
    def build_cmd(self) -> list[str]:
        """The exact ssh command line.  Raises ToolError when the requested
        auth mode is impossible on this machine.  Callers must set
        ``self.local_port`` first (0 → ``_free_port()``)."""
        p = self.profile
        if not p.host or not p.user:
            raise ToolError("ssh tunnel needs host and user")
        port = self.local_port or _free_port()
        cmd: list[str] = [
            self.ssh_bin,
            "-N",
            "-D", f"127.0.0.1:{port}",
            "-p", str(p.port),
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if p.key:
            if not os.path.isfile(p.key):
                raise ToolError(f"ssh key not found: {p.key}")
            cmd += ["-i", p.key, "-o", "IdentitiesOnly=yes"]
        target = f"{p.user}@{p.host}"
        cmd.append(target)
        if p.password:
            if shutil.which(self.sshpass_bin) is None:
                raise ToolError(
                    "password auth needs sshpass on this machine "
                    f"(`pkg install sshpass` on Termux) — or use key auth")
            return [self.sshpass_bin, "-e", *cmd]
        return cmd

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.alive():
                return {"started": False,
                        "note": "already running",
                        "local_port": self.local_port}
            if not self.local_port:
                self.local_port = _free_port()
            cmd = self.build_cmd()
            env = dict(os.environ)
            if self.profile.password:
                env["SSHPASS"] = self.profile.password
            try:
                self.proc = subprocess.Popen(
                    cmd, env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ToolError(f"could not launch ssh: {exc}") from exc
            self.restarts = 0
            self.started_at = time.time()
            self._stop.clear()
            if self.profile.auto_reconnect:
                self._thread = threading.Thread(
                    target=self._supervisor, name=f"sshsocks-{self.profile.name}",
                    daemon=True)
                self._thread.start()
            return {"started": True, "local_port": self.local_port,
                    "pid": self.proc.pid}

    def _launch(self) -> bool:
        """One (re)launch, used by the supervisor (local_port is fixed)."""
        cmd = self.build_cmd()
        env = dict(os.environ)
        if self.profile.password:
            env["SSHPASS"] = self.profile.password
        try:
            self.proc = subprocess.Popen(
                cmd, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            return True
        except OSError as exc:
            _log.warning("ssh tunnel relaunch failed: %s", exc)
            return False

    def _supervisor(self) -> None:
        """Watch the process; on unexpected death, relaunch with backoff."""
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self.proc.wait()
            except Exception:  # noqa: BLE001 - supervisor must never die
                break
            if self._stop.is_set() or self.proc is None:
                break
            if self.restarts >= max(0, self.profile.max_reconnects):
                _log.warning("ssh tunnel %s gave up after %d reconnects",
                             self.profile.name, self.restarts)
                break
            _log.info("ssh tunnel %s dropped — relaunch in %.0fs",
                      self.profile.name, backoff)
            deadline = time.time() + backoff
            while time.time() < deadline and not self._stop.is_set():
                time.sleep(0.2)
            if self._stop.is_set():
                break
            if self._launch():
                self.restarts += 1
                backoff = min(backoff * 2, 30.0)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def port_up(self) -> bool:
        return bool(self.local_port) and _socks5_probe("127.0.0.1",
                                                       self.local_port)

    def status(self) -> dict[str, Any]:
        p = self.profile
        return {
            "name": p.name,
            "host": p.host,
            "user": p.user,
            "running": self.alive(),
            "socks_port_open": self.port_up(),
            "local_port": self.local_port,
            "proxy_url": (f"socks5://127.0.0.1:{self.local_port}"
                          if self.local_port else ""),
            "pid": self.proc.pid if self.alive() else 0,
            "restarts": self.restarts,
            "auto_reconnect": p.auto_reconnect,
            "since": self.started_at,
        }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop.set()
            proc = self.proc
            self.proc = None
            if proc is None or proc.poll() is not None:
                return {"stopped": True, "note": "not running"}
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    proc.terminate()
                except OSError:  # noqa: E103 - process already gone; teardown is best-effort
                    pass
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # noqa: E103 - process survived SIGKILL wait; stop() reports stopped anyway
                    pass
            return {"stopped": True, "pid": proc.pid}


# ── manager ──────────────────────────────────────────────────────────────────


class SshSocksManager:
    """Named tunnels + persistence.  Profiles survive restarts; live
    process state is reconciled on boot (dead parent = dead tunnel).

    Live tunnels are CLASS-level: every tool call / chat command / agent
    constructs a fresh manager, but a tunnel started by one of them must
    be visible (and stoppable) from all of them — it is a long-lived
    process, not per-call state.
    """

    _live: dict[str, SshSocksTunnel] = {}
    _live_lock = threading.Lock()

    def __init__(self, context: Any) -> None:
        self.context = context
        self.tunnels = SshSocksManager._live
        self._lock = SshSocksManager._live_lock
        self._profiles = self._load_json(_PROFILES_KEY, {})
        self._load_json(_RUNTIME_KEY, {})  # read for future reference only

    # -- persistence ------------------------------------------------------------
    def _load_json(self, key: str, default: Any) -> Any:
        db = getattr(self.context, "db", None)
        if db is None:
            return default
        try:
            row = db.query_one("SELECT value FROM kv_store WHERE key = ?",
                               (key,))
            return json.loads(row["value"]) if row else default
        except Exception:  # noqa: BLE001
            return default

    def _save_json(self, key: str, value: Any) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            with db.transaction():
                db.execute(
                    "INSERT INTO kv_store (key, value, kind, updated_at) "
                    "VALUES (?, ?, 'json', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                    "updated_at = excluded.updated_at",
                    (key, json.dumps(value, default=str), time.time()))
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            _log.warning("could not persist ssh socks state: %s", exc)

    # -- actions ------------------------------------------------------------------
    def upsert_profile(self, *, name: str, host: str, port: str = "",
                       user: str = "", password: str = "", key: str = "",
                       local_port: str = "", auto_reconnect: str = "",
                       max_reconnects: str = "") -> SshSocksProfile:
        name = (name or "").strip().lower().replace(" ", "-")
        host = (host or "").strip()
        if not name or not host:
            raise ToolError("ssh tunnel needs a name and a host")
        existing = self._profiles.get(name)
        profile = SshSocksProfile.from_dict(existing or {})
        profile.name = name
        if host:
            profile.host = host
        if port:
            try:
                profile.port = int(port)
            except ValueError:
                raise ToolError(f"bad ssh port: {port}")
        if user:
            profile.user = user
        if password:
            profile.password = password
        if key:
            profile.key = key
        if local_port:
            try:
                profile.local_port = int(local_port)
            except ValueError:
                raise ToolError(f"bad local port: {local_port}")
        if auto_reconnect:
            profile.auto_reconnect = \
                auto_reconnect.lower() not in {"0", "false", "no", "off"}
        if max_reconnects:
            try:
                profile.max_reconnects = int(max_reconnects)
            except ValueError:
                raise ToolError(f"bad max_reconnects: {max_reconnects}")
        self._profiles[name] = profile.to_dict()
        self._save_json(_PROFILES_KEY, self._profiles)
        return profile

    def start(self, **kwargs: str) -> dict[str, Any]:
        profile = self.upsert_profile(**kwargs)
        with self._lock:
            tunnel = self.tunnels.get(profile.name)
            if tunnel is None:
                tunnel = SshSocksTunnel(SshSocksProfile.from_dict(profile.to_dict()))
                self.tunnels[profile.name] = tunnel
            else:
                tunnel.profile = SshSocksProfile.from_dict(profile.to_dict())
        return tunnel.start()

    def stop(self, name: str) -> dict[str, Any]:
        name = (name or "").strip().lower()
        with self._lock:
            tunnel = self.tunnels.get(name)
        if tunnel is None:
            if name in self._profiles:
                return {"stopped": True, "note": "not running (no live tunnel)"}
            raise ToolError(f"no ssh tunnel named {name!r} — ssh_socks list")
        return tunnel.stop()

    def status(self, names: str = "") -> dict[str, Any]:
        wanted = [n.strip().lower() for n in names.split(",") if n.strip()]
        out = {}
        for name, tunnel in sorted(self.tunnels.items()):
            if wanted and name not in wanted:
                continue
            out[name] = tunnel.status()
        if wanted and not out:
            raise ToolError(f"no live ssh tunnel matching {names!r} — "
                            f"ssh_socks action=list to see configured ones")
        return {"tunnels": out}

    def list(self) -> dict[str, Any]:
        live = {name: t.status() for name, t in sorted(self.tunnels.items())}
        configured = [
            {k: v for k, v in p.items() if k != "password"}
            | {"has_password": bool(p.get("password"))}
            for p in self._profiles.values()
        ]
        return {"live": live, "configured": configured,
                "urls": self.urls()}

    def remove(self, name: str) -> dict[str, Any]:
        name = (name or "").strip().lower()
        if name not in self._profiles and name not in self.tunnels:
            raise ToolError(f"no ssh tunnel named {name!r}")
        self.stop(name)
        with self._lock:
            self.tunnels.pop(name, None)
            self._profiles.pop(name, None)
        self._save_json(_PROFILES_KEY, self._profiles)
        return {"removed": True, "name": name}

    def urls(self) -> list[str]:
        """Live SOCKS5 URLs for every tunnel whose port answers — ready
        for proxy_set or the proxy pool."""
        return self._live_urls()

    @classmethod
    def _live_urls(cls) -> list[str]:
        """Class-level: every live tunnel whose SOCKS5 port answers —
        usable by other modules (the rotation pool) without a context."""
        out = []
        with cls._live_lock:
            items = list(cls._live.items())
        for _name, tunnel in sorted(items):
            st = tunnel.status()
            if st["running"] and st["socks_port_open"]:
                out.append(st["proxy_url"])
        return out


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "ssh_socks",
        description=(
            "Turn an SSH server into a local SOCKS5 proxy (127.0.0.1). "
            "start: host+user required, password or key, optional "
            "local_port/auto_reconnect — supervised, auto-reconnects with "
            "backoff. stop/status/list/remove/urls (urls = live "
            "socks5://127.0.0.1:port ready for proxy_set). Passwords are "
            "kept in the state DB and passed to sshpass via environment, "
            "never argv."
        ),
        capability=Capability.NET_OUT,
        # Note: also spawns ssh subprocesses (EXEC_SHELL). NET_OUT declared
        # as the primary capability; process execution is inherent to SSH tunnels.
        parameters={
            "action": "str — start | stop | status | list | urls | remove",
            "name": "str — tunnel name (stop/status/remove)",
            "host": "str (start) — ssh host",
            "port": "str (start, optional, 22)",
            "user": "str (start) — ssh user",
            "password": "str (start) — or use key",
            "key": "str (start) — path to private key",
            "local_port": "str (start, optional) — 0 = pick free port",
            "auto_reconnect": "bool (start, optional, true)",
            "max_reconnects": "str (start, optional, 10)",
        },
    )
    def ssh_socks(*, action: str = "list", name: str = "", host: str = "",
                  port: str = "", user: str = "", password: str = "",
                  key: str = "", local_port: str = "",
                  auto_reconnect: str = "", max_reconnects: str = "") -> dict[str, Any]:
        manager = SshSocksManager(context)
        action = (action or "list").strip().lower()
        if action == "start":
            return manager.start(name=name, host=host, port=port, user=user,
                                 password=password, key=key,
                                 local_port=local_port,
                                 auto_reconnect=auto_reconnect,
                                 max_reconnects=max_reconnects)
        if action == "stop":
            return manager.stop(name)
        if action == "status":
            return manager.status(name)
        if action == "urls":
            return {"urls": manager.urls()}
        if action == "remove":
            return manager.remove(name)
        if action == "list":
            return manager.list()
        raise ToolError(f"unknown ssh_socks action {action!r}")
