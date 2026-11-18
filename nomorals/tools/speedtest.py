"""Throughput measurement (Fast.com native build, 52-websites #47).

Fast.com is one big number: the download throughput on the road to
Netflix. Devon does the same job with her own vantage points and
says more: download + upload + loaded vs unloaded latency, journaled
history, and a baseline watch — so "is my connection slower than
last week?" is a baseline question, not a feeling.

Honest limits, stated on every result: Devon cannot borrow Netflix's
CDN. Her seed endpoints are smaller and their own bottlenecks cap
the measurable ceiling — a gigabit line cannot be proven through a
100 Mbps mirror. Every result is therefore "throughput to THESE
endpoints from THIS host", labelled with target + vantage. A number
with no measured leg is reported as not measured, never filled in.

Scope discipline (shared with :mod:`nomorals.tools.network`): a
public seed is only touched when its host is explicitly allowlisted
in ``NM_NET_ALLOWED_TARGETS``. Loopback and private ranges pass by
default, so the loopback fixture and LAN targets always work. A
measurement that cannot run refuses in words naming the reason.
"""

from __future__ import annotations

import json
import os
import re
import socket
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..core.profile import is_termux
from .network import is_scannable

_log = get_logger(__name__)

__all__ = [
    "SeedEndpoint",
    "SEED_ENDPOINTS",
    "SpeedResult",
    "SpeedHistoryStore",
    "default_budgets",
    "format_speed_reply",
    "is_degraded",
    "mbps",
    "parse_fastcom_targets",
    "run_speed_test",
    "seed_for",
    "register",
]

#: Desktop default: a full test may move tens of MB.
DESKTOP_MAX_BYTES = 32 * 1024 * 1024
DESKTOP_MAX_SECONDS = 30.0
#: Phone (Termux) default: the budget shrinks, and the result says so.
PHONE_MAX_BYTES = 8 * 1024 * 1024
PHONE_MAX_SECONDS = 15.0
#: Ramp sizes: grow until throughput plateaus or the budget is hit.
RAMP_SIZES = (256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024)
#: A ramp counts as a plateau when it improves on the previous by less
#: than this factor — sustained throughput, not burst.
PLATEAU_FACTOR = 1.10
#: Streams per ramp: parallel transfers, the Fast.com mechanic.
STREAMS = 4
#: Degraded when the latest run is below this fraction of baseline.
DEGRADED_RATIO = 0.6
#: Baseline: median down over this window, with at least this many samples.
BASELINE_DAYS = 14
BASELINE_MIN_SAMPLES = 3
#: History retention.
HISTORY_MAX_DAYS = 90
HISTORY_MAX_ROWS = 500
#: Per-request ceiling; the overall budget is enforced separately.
REQUEST_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class SeedEndpoint:
    """One measurement endpoint: named, with its own upload rung."""

    name: str
    host: str
    down_url: str
    up_url: str | None
    ping_url: str
    note: str
    kind: str = "file"


SEED_ENDPOINTS: tuple[SeedEndpoint, ...] = (
    SeedEndpoint(
        name="cloudflare",
        host="speed.cloudflare.com",
        down_url="https://speed.cloudflare.com/__down?bytes={n}",
        up_url="https://speed.cloudflare.com/__up",
        ping_url="https://speed.cloudflare.com/__down?bytes=0",
        note="Cloudflare speed-test backend — arbitrary byte counts, upload rung",
    ),
    SeedEndpoint(
        name="fast.com",
        host="api.fast.com",
        down_url="https://api.fast.com/netflix/speedtest?https=true",
        up_url=None,
        ping_url="https://api.fast.com/netflix/speedtest?https=true",
        note="Netflix Open Connect via the fast.com API — the road to Netflix itself",
        kind="fastcom",
    ),
    SeedEndpoint(
        name="hetzner",
        host="speed.hetzner.de",
        down_url="https://speed.hetzner.de/100MB.bin",
        up_url=None,
        ping_url="https://speed.hetzner.de/1MB.bin",
        note="Hetzner public test file — Range reads, no upload rung",
    ),
    SeedEndpoint(
        name="cachefly",
        host="cachefly.cachefly.net",
        down_url="https://cachefly.cachefly.net/100mb.test",
        up_url=None,
        ping_url="https://cachefly.cachefly.net/10mb.test",
        note="Cachefly public test file — Range reads, no upload rung",
    ),
)


def seed_for(name: str) -> SeedEndpoint | None:
    """The seed with this name, or None."""
    wanted = (name or "").strip().lower()
    for seed in SEED_ENDPOINTS:
        if seed.name == wanted:
            return seed
    return None


def mbps(bytes_moved: int, seconds: float) -> float:
    """Throughput in megabits per second. Zero time is zero, not a crash."""
    if seconds <= 0 or bytes_moved <= 0:
        return 0.0
    return bytes_moved * 8.0 / seconds / 1_000_000.0


def default_budgets(phone: bool) -> tuple[int, float]:
    """(max_bytes, max_seconds) for the profile. Phone shrinks."""
    if phone:
        return PHONE_MAX_BYTES, PHONE_MAX_SECONDS
    return DESKTOP_MAX_BYTES, DESKTOP_MAX_SECONDS


def _vantage() -> str:
    host = socket.gethostname() or "unknown host"
    return f"{host} ({'termux' if is_termux() else 'host'})"


def parse_fastcom_targets(payload: Any) -> list[str]:
    """Target download URLs out of a fast.com API payload.

    Only https URLs with a ``url`` field survive; anything else is
    dropped, never guessed at.
    """
    if not isinstance(payload, dict):
        return []
    targets = payload.get("targets")
    if not isinstance(targets, list):
        return []
    out: list[str] = []
    for row in targets:
        if not isinstance(row, dict):
            continue
        url = row.get("url")
        if isinstance(url, str) and url.startswith("https://"):
            out.append(url)
    return out


_RANGE_RE = re.compile(r"range/0-\d+")


def _fastcom_url_for(url: str, n: int) -> str:
    """Rewrite a fast.com range URL to fetch exactly ``n`` bytes."""
    return _RANGE_RE.sub(f"range/0-{max(n - 1, 0)}", url, count=1)


#: Every request carries an honest User-Agent: the Cloudflare backend
#: refuses urllib's default outright (403), any real UA passes.
_USER_AGENT = "Devon-speedtest/1.0"


def _open(url: str, *, timeout: float, data: Any = None,
          headers: dict[str, str] | None = None):
    merged = {"User-Agent": _USER_AGENT}
    merged.update(headers or {})
    request = urllib.request.Request(url, data=data, headers=merged)
    return urllib.request.urlopen(request, timeout=timeout)


def _read_n(response: Any, n: int, deadline: float) -> int:
    """Read up to ``n`` bytes, counting what actually arrived.

    Stops early at the deadline so one slow stream cannot blow the
    overall budget; a short read is counted honestly, not padded.
    """
    got = 0
    while got < n:
        if time.monotonic() >= deadline:
            break
        try:
            chunk = response.read(min(65536, n - got))
        except (OSError, urllib.error.URLError):
            break
        if not chunk:
            break
        got += len(chunk)
    return got


def _download_once(url: str, n: int, timeout: float,
                   deadline: float) -> tuple[int, float]:
    """One timed download of ``n`` bytes. Returns (bytes, seconds)."""
    if "{n}" in url:
        fetch_url = url.format(n=n)
        headers: dict[str, str] = {}
    else:
        fetch_url = url
        headers = {"Range": f"bytes=0-{max(n - 1, 0)}"}
    start = time.monotonic()
    try:
        with _open(fetch_url, timeout=timeout, headers=headers) as resp:
            got = _read_n(resp, n, deadline)
    except (OSError, urllib.error.URLError) as exc:
        raise _TransferError(f"download failed: {exc}") from exc
    return got, max(time.monotonic() - start, 0.0)


class _TransferError(Exception):
    """A transfer leg failed; the caller decides whether to fail over."""


class _RandomBody:
    """A file-like body of ``size`` random bytes for the upload leg."""

    def __init__(self, size: int) -> None:
        self._left = size

    def __len__(self) -> int:
        return self._left

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        take = self._left if n < 0 else min(n, self._left)
        self._left -= take
        return os.urandom(take)


def _measure_latency(url: str, count: int = 5,
                     timeout: float = 3.0) -> float | None:
    """Median round-trip for small requests, or None when all fail."""
    samples: list[float] = []
    for _ in range(count):
        start = time.monotonic()
        try:
            with _open(url, timeout=timeout) as resp:
                resp.read(4096)
        except (OSError, urllib.error.URLError):
            continue
        samples.append((time.monotonic() - start) * 1000.0)
    if not samples:
        return None
    return statistics.median(samples)


def _latency_sampler(url: str, stop: threading.Event,
                     samples: list[float]) -> None:
    """Background sampler: pings during the download leg (loaded latency)."""
    while not stop.is_set():
        start = time.monotonic()
        try:
            with _open(url, timeout=3.0) as resp:
                resp.read(4096)
            samples.append((time.monotonic() - start) * 1000.0)
        except (OSError, urllib.error.URLError):
            pass


def _measure_upload(up_url: str, size: int, timeout: float,
                    deadline: float) -> tuple[int, float]:
    """One timed upload of ``size`` bytes. Returns (bytes, seconds).

    The overall budget is enforced by the caller (it only starts an
    upload when enough time remains); the per-request timeout bounds
    this call itself.
    """
    body = _RandomBody(size)
    headers = {"User-Agent": _USER_AGENT,
               "Content-Type": "application/octet-stream",
               "Content-Length": str(size)}
    start = time.monotonic()
    try:
        request = urllib.request.Request(
            up_url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            resp.read(4096)
    except (OSError, urllib.error.URLError) as exc:
        raise _TransferError(f"upload failed: {exc}") from exc
    return size - body._left, max(time.monotonic() - start, 0.0)


@dataclass
class SpeedResult:
    """One measurement, labelled with target + vantage, always."""

    target: str
    down_mbps: float
    up_mbps: float | None
    latency_idle_ms: float
    latency_loaded_ms: float | None
    bytes_moved: int
    duration_s: float
    vantage: str
    ok: bool = True
    reason: str = ""
    budget_note: str = ""
    profile: str = "desktop"
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "down_mbps": self.down_mbps,
            "up_mbps": self.up_mbps,
            "latency_idle_ms": self.latency_idle_ms,
            "latency_loaded_ms": self.latency_loaded_ms,
            "bytes_moved": self.bytes_moved,
            "duration_s": self.duration_s,
            "vantage": self.vantage,
            "ok": self.ok,
            "reason": self.reason,
            "budget_note": self.budget_note,
            "profile": self.profile,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> SpeedResult:
        return cls(
            target=str(row.get("target", "")),
            down_mbps=float(row.get("down_mbps", 0.0) or 0.0),
            up_mbps=(None if row.get("up_mbps") is None
                     else float(row.get("up_mbps") or 0.0)),
            latency_idle_ms=float(row.get("latency_idle_ms", 0.0) or 0.0),
            latency_loaded_ms=(None if row.get("latency_loaded_ms") is None
                               else float(row.get("latency_loaded_ms") or 0.0)),
            bytes_moved=int(row.get("bytes_moved", 0) or 0),
            duration_s=float(row.get("duration_s", 0.0) or 0.0),
            vantage=str(row.get("vantage", "")),
            ok=bool(row.get("ok", True)),
            reason=str(row.get("reason", "") or ""),
            budget_note=str(row.get("budget_note", "") or ""),
            profile=str(row.get("profile", "desktop") or "desktop"),
            recorded_at=float(row.get("recorded_at", 0.0) or 0.0),
        )


def _store_path(home: str) -> Path:
    return Path(home) / "data" / "speed_history.json"


class SpeedHistoryStore:
    """Journaled history: every run lands here, baseline reads it back.

    Atomic writes (temp + replace), pruned to 90 days / 500 rows.
    The store never raises: a corrupt file reads as empty, a failed
    write is logged and dropped.
    """

    def __init__(self, home: str | None) -> None:
        self._home = str(home) if home else ""

    def _read(self) -> list[dict[str, Any]]:
        if not self._home:
            return []
        path = _store_path(self._home)
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return rows if isinstance(rows, list) else []

    def _write(self, rows: list[dict[str, Any]]) -> None:
        if not self._home:
            return
        path = _store_path(self._home)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rows), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:  # noqa: BLE001 — history is best-effort
            _log.debug("speed history write failed: %s", exc)

    def append(self, result: SpeedResult) -> None:
        rows = self._read()
        rows.append(result.to_dict())
        cutoff = time.time() - HISTORY_MAX_DAYS * 86400.0
        rows = [r for r in rows
                if float(r.get("recorded_at", 0.0) or 0.0) >= cutoff]
        rows = rows[-HISTORY_MAX_ROWS:]
        self._write(rows)

    def list(self, target: str | None = None) -> list[SpeedResult]:
        rows = self._read()
        out = [SpeedResult.from_dict(r) for r in rows if isinstance(r, dict)]
        if target:
            out = [r for r in out if r.target == target]
        return out

    def latest(self, target: str | None = None) -> SpeedResult | None:
        rows = self.list(target)
        return rows[-1] if rows else None

    def baseline(self, target: str, days: int = BASELINE_DAYS,
                 min_samples: int = BASELINE_MIN_SAMPLES) -> float | None:
        """Median download over the window — None when too thin to say.

        Too thin is honest: no baseline is invented from one or two runs.
        """
        cutoff = time.time() - days * 86400.0
        downs = [r.down_mbps for r in self.list(target)
                 if r.ok and r.recorded_at >= cutoff and r.down_mbps > 0]
        if len(downs) < min_samples:
            return None
        return statistics.median(downs)


def is_degraded(latest_mbps: float, baseline_mbps: float | None) -> bool:
    """Degraded when the latest run is below 60% of baseline."""
    if baseline_mbps is None or baseline_mbps <= 0:
        return False
    return latest_mbps < DEGRADED_RATIO * baseline_mbps


def format_speed_reply(result: SpeedResult,
                       baseline_mbps: float | None = None) -> str:
    """Plain reply, always labelled with target + vantage."""
    if not result.ok:
        return (f"speed test failed: {result.reason or 'unknown reason'} "
                f"(target {result.target or 'none'}, "
                f"vantage {result.vantage})")
    lines = [
        f"speed to {result.target} from {result.vantage}: "
        f"{result.down_mbps:.1f} Mbps down",
    ]
    if result.up_mbps is None:
        lines[0] += " — upload not measured for this target"
    else:
        lines.append(f"upload: {result.up_mbps:.1f} Mbps")
    lines.append(f"latency: {result.latency_idle_ms:.0f} ms idle"
                 + (f", {result.latency_loaded_ms:.0f} ms loaded"
                    if result.latency_loaded_ms is not None
                    else ", loaded latency not sampled"))
    if baseline_mbps is not None:
        if is_degraded(result.down_mbps, baseline_mbps):
            lines.append(
                f"down from a {baseline_mbps:.1f} Mbps baseline "
                f"({100.0 * result.down_mbps / baseline_mbps:.0f}% of usual)")
        else:
            lines.append(f"baseline: {baseline_mbps:.1f} Mbps — holding")
    else:
        lines.append("no baseline yet — too few past runs to compare")
    lines.append(
        f"moved {result.bytes_moved / 1_000_000:.1f} MB in "
        f"{result.duration_s:.1f} s")
    if result.budget_note:
        lines.append(result.budget_note)
    return "\n".join(lines)


def _resolve_seed(target: str, seeds: tuple[SeedEndpoint, ...]
                  ) -> SeedEndpoint | tuple[SeedEndpoint, ...] | None:
    """A named seed, a custom URL, or the whole seed list in order.

    Names are looked up in the seed list actually in use first (the
    loopback fixture injects its own), then in the global list.
    """
    target = (target or "").strip()
    if not target:
        return seeds
    wanted = target.lower()
    for seed in seeds:
        if seed.name.lower() == wanted:
            return seed
    named = seed_for(target)
    if named is not None:
        return named
    if target.startswith(("http://", "https://")):
        host = urllib.parse.urlparse(target).hostname or ""
        return SeedEndpoint(
            name=target, host=host, down_url=target, up_url=None,
            ping_url=target,
            note="explicit target — measured as given, gated like a seed")
    return None


def _fastcom_target_url(seed: SeedEndpoint, timeout: float) -> str:
    """Resolve a fast.com API seed into a concrete download URL."""
    try:
        with _open(seed.down_url, timeout=timeout) as resp:
            payload = json.loads(resp.read(65536).decode("utf-8", "replace"))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise _TransferError(f"fast.com API unreachable: {exc}") from exc
    targets = parse_fastcom_targets(payload)
    if not targets:
        raise _TransferError("fast.com API returned no targets")
    return targets[0]


def _run_one_seed(seed: SeedEndpoint, *, max_bytes: int, max_seconds: float,
                  down_only: bool, phone: bool) -> SpeedResult:
    """Measure against one seed. Raises _TransferError when it cannot."""
    t0 = time.monotonic()
    deadline = t0 + max_seconds
    vantage = _vantage()
    budget_note = (
        f"budget {max_bytes / 1_000_000:.0f} MB / {max_seconds:.0f} s"
        + (" (phone profile — smaller budget)"
           if phone else ""))
    idle = None
    if seed.kind == "fastcom":
        # Resolve the download target first: idle latency is probed
        # against the target that will carry the traffic, not the API.
        base_url = _fastcom_target_url(seed, REQUEST_TIMEOUT_S)
        ping_url = _fastcom_url_for(base_url, 1)
        idle = _measure_latency(ping_url)
    else:
        base_url = seed.down_url
        ping_url = seed.ping_url
        idle = _measure_latency(ping_url)
    if idle is None:
        raise _TransferError(
            f"{seed.name} did not answer a latency probe")

    # Download ramp: grow until throughput plateaus or the budget is hit.
    stop = threading.Event()
    loaded: list[float] = []
    sampler = threading.Thread(
        target=_latency_sampler, args=(ping_url, stop, loaded),
        daemon=True)
    sampler.start()
    total_bytes = 0
    total_seconds = 0.0
    last_mbps = 0.0
    try:
        for size in RAMP_SIZES:
            if total_bytes + size > max_bytes:
                break
            if time.monotonic() >= deadline:
                break
            per_stream = max(size // STREAMS, 1)
            # Each ramp needs its own byte range on the target.
            ramp_url = (_fastcom_url_for(base_url, size)
                        if seed.kind == "fastcom" else base_url)
            urls = [ramp_url] * STREAMS
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=STREAMS) as pool:
                futures = [
                    pool.submit(_download_once, url, per_stream,
                                REQUEST_TIMEOUT_S, deadline)
                    for url in urls]
                got = 0
                for fut in futures:
                    got += fut.result()[0]
            elapsed = max(time.monotonic() - start, 0.0)
            total_bytes += got
            total_seconds += elapsed
            # A short read means the server capped us — that IS the
            # ceiling for this target; record it and stop ramping.
            if got < size:
                last_mbps = mbps(got, elapsed)
                break
            this_mbps = mbps(got, elapsed)
            if last_mbps > 0 and this_mbps < PLATEAU_FACTOR * last_mbps:
                last_mbps = this_mbps
                break
            last_mbps = this_mbps
    finally:
        stop.set()
        sampler.join(timeout=5.0)
    if total_bytes <= 0 or last_mbps <= 0:
        raise _TransferError(f"{seed.name} moved no bytes")
    loaded_ms = statistics.median(loaded) if loaded else None

    up_mbps: float | None = None
    if not down_only and seed.up_url:
        remaining = max(deadline - time.monotonic(), 0.0)
        if remaining > 2.0:
            up_size = min(4 * 1024 * 1024, max_bytes // 4)
            try:
                got, spent = _measure_upload(
                    seed.up_url, up_size, REQUEST_TIMEOUT_S, deadline)
                up_mbps = mbps(got, spent) if got > 0 else None
            except _TransferError:
                up_mbps = None  # upload rung failed — down still stands

    return SpeedResult(
        target=seed.name,
        down_mbps=last_mbps,
        up_mbps=up_mbps,
        latency_idle_ms=idle,
        latency_loaded_ms=loaded_ms,
        bytes_moved=total_bytes,
        duration_s=max(time.monotonic() - t0, 0.0),
        vantage=vantage,
        budget_note=budget_note,
        profile="phone" if phone else "desktop",
    )


def run_speed_test(home: str | None = None, target: str = "",
                   down_only: bool = False, budget_mb: float = 0.0,
                   max_seconds: float = 0.0, allowed_targets: str = "",
                   seeds: tuple[SeedEndpoint, ...] | list[SeedEndpoint] | None = None,
                   ) -> SpeedResult:
    """Run a speed test. Never raises — failure is a result with a reason.

    ``target`` names a seed (``cloudflare``), a full URL, or "" for the
    first allowlisted seed that answers. ``budget_mb`` / ``max_seconds``
    of 0 mean the profile default. ``seeds`` overrides the seed list
    (the loopback fixture uses this).
    """
    phone = is_termux()
    default_bytes, default_seconds = default_budgets(phone)
    max_bytes = (int(budget_mb * 1024 * 1024) if budget_mb and budget_mb > 0
                 else default_bytes)
    max_secs = (float(max_seconds) if max_seconds and max_seconds > 0
                else default_seconds)
    vantage = _vantage()
    profile = "phone" if phone else "desktop"
    budget_note = (
        f"budget {max_bytes / 1_000_000:.0f} MB / {max_secs:.0f} s"
        + (" (phone profile — smaller budget)" if phone else ""))

    def _fail(reason: str, name: str = "") -> SpeedResult:
        return SpeedResult(
            target=name or (target or "none"),
            down_mbps=0.0, up_mbps=None, latency_idle_ms=0.0,
            latency_loaded_ms=None, bytes_moved=0, duration_s=0.0,
            vantage=vantage, ok=False, reason=reason,
            budget_note=budget_note, profile=profile)

    seed_list: tuple[SeedEndpoint, ...] = tuple(seeds) if seeds else SEED_ENDPOINTS
    resolved = _resolve_seed(target, seed_list)
    if resolved is None:
        return _fail(
            f"unknown target {target!r} — name a seed "
            f"({', '.join(s.name for s in seed_list)}) or a full URL")
    candidates: list[SeedEndpoint] = (
        [resolved] if isinstance(resolved, SeedEndpoint) else list(resolved))

    last_error = ""
    for seed in candidates:
        if not is_scannable(seed.host, allowed_targets):
            last_error = (
                f"{seed.name} ({seed.host}) is outside the allowed scope — "
                f"add it to NM_NET_ALLOWED_TARGETS to permit it explicitly")
            continue
        try:
            result = _run_one_seed(
                seed, max_bytes=max_bytes, max_seconds=max_secs,
                down_only=down_only, phone=phone)
        except _TransferError as exc:
            last_error = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001 — a seed never kills the run
            last_error = f"{seed.name} failed: {exc}"
            continue
        if home:
            store = SpeedHistoryStore(home)
            store.append(result)
        return result
    if not last_error:
        last_error = "no seed endpoint available"
    if "NM_NET_ALLOWED_TARGETS" not in last_error:
        last_error = (
            f"no allowlisted seed answered ({last_error}); "
            f"add a seed host to NM_NET_ALLOWED_TARGETS, or pass a "
            f"loopback/LAN target")
    return _fail(last_error)


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _allowed_from_context(context: Any) -> str:
    settings = getattr(context, "settings", None) if context else None
    net = getattr(settings, "net", None) if settings else None
    return str(getattr(net, "allowed_targets", "") or "")


def register(registry: Any) -> None:
    """Attach the ``speed_test`` tool to a registry."""
    context = registry.context

    @registry.register(
        "speed_test",
        description=(
            "Measure connection throughput — download, upload, idle and "
            "loaded latency — to a named measurement endpoint, from this "
            "host. Ask it as 'how fast is the connection?' or 'check the "
            "EC2 network'. target=<seed name (cloudflare, fast.com, "
            "hetzner, cachefly), a full URL, or empty for the first "
            "allowlisted seed that answers>; down_only=<true to skip the "
            "upload leg>; budget_mb=<cap on bytes moved, 0 = profile "
            "default>; max_seconds=<time cap, 0 = profile default>. "
            "Public seeds need NM_NET_ALLOWED_TARGETS; loopback and LAN "
            "targets always pass. Returns ok, target, down_mbps, "
            "up_mbps (null when not measured), latency_idle_ms, "
            "latency_loaded_ms, bytes_moved, duration_s, vantage, "
            "baseline_mbps, degraded and a plain reply — or ok false "
            "with the reason named, never an invented number."
        ),
        capability=Capability.NET_OUT,
    )
    def speed_test_tool(target: str = "", down_only: bool = False,
                        budget_mb: float = 0.0, max_seconds: float = 0.0,
                        allowed_targets: str | None = None,
                        seeds: Any = None) -> dict[str, Any]:
        home = _home_from_context(context)
        allowed = (allowed_targets if allowed_targets is not None
                   else _allowed_from_context(context))
        result = run_speed_test(
            home=home, target=str(target or ""), down_only=bool(down_only),
            budget_mb=float(budget_mb or 0.0),
            max_seconds=float(max_seconds or 0.0),
            allowed_targets=allowed,
            seeds=seeds)
        baseline = None
        degraded = False
        if home and result.ok:
            store = SpeedHistoryStore(home)
            baseline = store.baseline(result.target)
            degraded = is_degraded(result.down_mbps, baseline)
        reply = format_speed_reply(result, baseline_mbps=baseline)
        return {
            **result.to_dict(),
            "baseline_mbps": baseline,
            "degraded": degraded,
            "reply": reply,
        }
