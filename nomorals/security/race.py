"""Race conditions: N identical requests in one wave.

Concurrency alone is not a race — a wave has to land inside the server's
check-then-write window. Method: fire N, count how many succeed
identically, then fire again with a distinct value per request so
double-spend is visible (two different codes accepted, one balance move).

Budget costs N per wave, so a 20-way wave is 20 from the governor.
"""
from __future__ import annotations

import concurrent.futures
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger
from .scope import Governor

_log = get_logger(__name__)
__all__ = ["RaceResult", "wave", "register"]


@dataclass
class RaceResult:
    url: str
    sent: int = 0
    answered: int = 0
    statuses: dict[int, int] = field(default_factory=dict)
    identical: int = 0
    bodies: list[str] = field(default_factory=list)
    verdict: str = "no_race"

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "sent": self.sent, "answered": self.answered,
                "statuses": self.statuses, "identical": self.identical,
                "verdict": self.verdict}


def wave(url: str, *, governor: Governor, method: str = "POST",
         body_template: str = "", count: int = 20, unique: bool = True,
         timeout: float = 25.0) -> RaceResult:
    """Fire `count` requests as simultaneously as the box allows.

    `body_template` may contain `{n}` (index) or `{u}` (uuid) so each
    request carries a distinct value.
    """
    result = RaceResult(url=url, sent=count)
    payloads: list[Optional[str]] = []
    for i in range(count):
        if not body_template:
            payloads.append(None)
        else:
            payloads.append(body_template.replace("{n}", str(i)).replace(
                "{u}", uuid.uuid4().hex[:12]))

    def fire(payload: Optional[str]) -> tuple[int, str]:
        client = HttpClient(timeout=timeout)
        governor.spend(url)
        started = time.monotonic()
        if method.upper() == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, data=payload or "",
                               headers={"Content-Type":
                                        "application/x-www-form-urlencoded"})
        elapsed = time.monotonic() - started
        _log.debug("race %s %s %.2fs", resp.status, url, elapsed)
        return resp.status, (resp.text or "")[:400]

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        for status, body in pool.map(fire, payloads):
            result.answered += 1
            result.statuses[status] = result.statuses.get(status, 0) + 1
            result.bodies.append(body)

    if result.bodies:
        first = result.bodies[0]
        result.identical = sum(1 for b in result.bodies if b == first)
    ok = result.statuses.get(200, 0) + result.statuses.get(201, 0)
    if unique and ok >= count - 1:
        result.verdict = "all_accepted"
    elif ok >= 2 and len(result.statuses) > 1:
        result.verdict = "partial"
    elif ok >= 2:
        result.verdict = "serialised"
    else:
        result.verdict = "no_race"
    return result


def register(registry: Any) -> None:
    from .case import governor_for, home_of, resolve_scope

    home = home_of(registry.context)

    @registry.register("race_wave",
                       description="Fire N concurrent requests to test a "
                                   "check-then-write window. `scope` is a "
                                   "scope file; pass `case` instead to run "
                                   "under that case's scope.txt. Refuses "
                                   "without a scope.",
                       capability=Capability.NET_OUT,
                       parameters={"url": "str",
                                   "scope": "str — scope file (optional)",
                                   "count": "int (optional, 20)",
                                   "body": "str (optional) — {n}/{u} templated",
                                   "method": "str (optional, POST)",
                                   "case": "str — case name (optional)"})
    def race_wave(url: str, scope: str = "", *, count: str = "20",
                  body: str = "", method: str = "POST",
                  case: str = "") -> dict[str, Any]:
        n = int(count or 20)
        gov = governor_for(resolve_scope(home, scope, case),
                           max_total=n * 2, max_per_host=50.0)
        return wave(url, governor=gov, count=n, body_template=body,
                    method=method).as_dict()
