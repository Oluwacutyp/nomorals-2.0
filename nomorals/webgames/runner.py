"""Supervisor: N slots, one state machine each, self-healing.

The target handle is the only durable setting. Everything else is
discovered. A slot that fails is recycled; a wall is met by a fresh mint
on a fresh profile and a rotated proxy; a run that loses every slot backs
off globally rather than spinning.

State is persisted per slot so a restart resumes instead of restarting
the counter.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.logging_setup import get_logger
from . import lagos
from .driver import Wall, mint

_log = get_logger(__name__)
__all__ = ["RunConfig", "Runner", "get_runner"]

MAX_PER_SLOT = 4


@dataclass
class RunConfig:
    target: str = "death4all"
    amount: int = lagos.GIFT_AMOUNT
    slots: int = 2
    headless: bool = True
    daily_cap: int = 0            # 0 = unlimited
    delay: tuple[float, float] = (2.0, 5.0)


class Runner:
    def __init__(self, home: Path, config: RunConfig) -> None:
        self.home = home
        self.config = config
        self.root = home / "data" / "webgames"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = self.root / "save_payload.json"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._stats = {"created": 0, "gifted": 0, "failed": 0,
                       "walls": 0, "value": 0, "target_id": ""}
        self._pool = None

    # ── proxy pool, optional ────────────────────────────────────────────
    def _acquire(self) -> str:
        if self._pool is None:
            try:
                from ..tools.proxy_rotate import pool_from_context
                class _Ctx:
                    home = self.home
                self._pool = pool_from_context(_Ctx())
                self._pool.probe_all()
            except Exception:  # noqa: BLE001
                self._pool = False
        return self._pool.acquire() or "" if self._pool else ""

    def _target_id(self) -> str:
        if self._stats["target_id"]:
            return self._stats["target_id"]
        for _ in range(3):
            found = lagos.resolve_target(self.config.target,
                                         proxy=self._acquire())
            if found:
                self._stats["target_id"] = found
                return found
            time.sleep(3)
        return ""

    # ── one slot ────────────────────────────────────────────────────────
    def _slot(self, index: int) -> None:
        while not self._stop.is_set():
            if self.config.daily_cap and \
                    self._stats["gifted"] >= self.config.daily_cap:
                return
            try:
                self._cycle(index)
            except Wall as wall:
                self._stats["walls"] += 1
                _log.info("slot %s hit %s", index, wall.kind)
                time.sleep(5 + random.random() * 10)
            except Exception as exc:  # noqa: BLE001
                _log.info("slot %s error: %s", index, exc)
                time.sleep(4)
            if self._stop.is_set():
                return
            time.sleep(random.uniform(*self.config.delay))

    def _cycle(self, index: int) -> None:
        target_id = self._target_id()
        if not target_id:
            raise RuntimeError("target handle not resolvable")
        proxy = self._acquire()
        account = lagos._random_account()
        lagos.register(account, root=self.root / "profiles", slot=index,
                       proxy=proxy, headless=self.config.headless)
        if account.state != "registered":
            self._stats["failed"] += 1
            return
        self._stats["created"] += 1

        payload = lagos.capture_payload(account, root=self.root / "profiles",
                                        slot=index, cache=self.cache,
                                        proxy=proxy,
                                        headless=self.config.headless)
        if not payload:
            self._stats["failed"] += 1
            return
        if not lagos.qualify(account, payload, proxy=proxy):
            self._stats["failed"] += 1
            return
        result = lagos.gift(account, target_id, self.config.amount,
                            proxy=proxy)
        if result.get("ok"):
            self._stats["gifted"] += 1
            self._stats["value"] += self.config.amount
            self._persist(account)
        else:
            self._stats["failed"] += 1

    def _persist(self, account: lagos.Account) -> None:
        path = self.root / "accounts.jsonl"
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(account.as_dict()) + "\n")

    # ── control ─────────────────────────────────────────────────────────
    def start(self) -> dict[str, Any]:
        if self._threads:
            return {"already": True, "slots": len(self._threads)}
        self._stop.clear()
        for i in range(max(1, self.config.slots)):
            thread = threading.Thread(target=self._slot, args=(i,), daemon=True)
            thread.start()
            self._threads.append(thread)
        return {"started": True, "slots": len(self._threads),
                "target": self.config.target}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        count = len(self._threads)
        self._threads = []
        return {"stopped": True, "slots": count}

    def status(self) -> dict[str, Any]:
        return {"running": bool(self._threads), "target": self.config.target,
                "amount": self.config.amount, "slots": self.config.slots,
                **self._stats}


_RUNNERS: dict[str, Runner] = {}


def get_runner(home: Path, config: Optional[RunConfig] = None) -> Runner:
    key = str(home)
    runner = _RUNNERS.get(key)
    if runner is None:
        runner = Runner(home, config or RunConfig())
        _RUNNERS[key] = runner
    elif config is not None:
        runner.config = config
    return runner
