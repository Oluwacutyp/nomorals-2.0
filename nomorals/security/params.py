"""Parameter discovery: mine, then confirm.

Two stages, and the second is what makes the first worth anything:

* mine — parameter names from the harvest corpus (`seed.json` URLs),
  wordlist, and a built-in set of the names that actually carry bugs:
  id, redirect, file, url, next, callback, path, template.
* confirm — send a control value. A parameter is REAL only when the
  response departs from a no-param baseline by `diff.compare`; a name the
  server ignores is dropped rather than passed to the injectors.

Reflection is recorded separately: a reflected param is a different class
of lead from a parsed-but-silent one.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from . import diff
from .scope import Governor

_log = get_logger(__name__)
__all__ = ["PARAM_SEEDS", "Param", "mine", "confirm", "register"]

#: Names that carry a disproportionate share of real bugs, by class.
PARAM_SEEDS: tuple[str, ...] = tuple("""
id ids uid user userid account order invoice item product pid
page p offset limit start count size q query s search term keyword
file filename path dir folder doc document template view page include
url uri link next return returnto redirect redirect_uri callback
continue dest destination target goto ref referer source from
lang locale country currency format type mode action cmd do op
debug test admin token key api_key apikey secret sig signature hash
email mail name first last phone address city zip
price amount qty quantity total discount coupon code
sort order_by orderby filter where select fields expand embed
""".split())

_CONTROL = "venicecanary8471"
_ASSIGN = re.compile(r"[?&]([A-Za-z_][A-Za-z0-9_\-\[\]]{0,40})=")
_PATHSEG = re.compile(r"^[a-z][a-z0-9_\-]{1,30}$", re.I)


@dataclass
class Param:
    url: str
    name: str
    where: str            # query | path | body
    reflected: bool = False
    delta: int = 0
    axes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "name": self.name, "where": self.where,
                "reflected": self.reflected, "delta": self.delta,
                "axes": self.axes}


def _from_urls(urls: Iterable[str]) -> dict[str, set[str]]:
    """Names already in the corpus, per host."""
    out: dict[str, set[str]] = {}
    for url in urls:
        parts = urllib.parse.urlsplit(url)
        if not parts.hostname:
            continue
        names = out.setdefault(parts.hostname, set())
        names.update(m.group(1) for m in _ASSIGN.finditer(url))
        for segment in parts.path.strip("/").split("/"):
            if _PATHSEG.match(segment) and not segment.isdigit():
                names.add(segment)
    return out


def mine(case_dir: str | Path, *, extra: Iterable[str] = (),
         limit: int = 120) -> dict[str, list[str]]:
    """Candidate names per host, corpus first, seeds after."""
    case = Path(case_dir)
    urls: list[str] = []
    seed_file = case / "seed.json"
    if seed_file.is_file():
        data = json.loads(seed_file.read_text(encoding="utf-8"))
        urls = [i["value"] for i in data.get("url", [])]
    found = _from_urls(urls)
    seeds = list(PARAM_SEEDS) + [e.strip() for e in extra if e.strip()]
    for host in found:
        found[host].update(seeds)
    return {host: sorted(names)[:limit] for host, names in found.items()}


def _baseline(client: HttpClient, url: str, samples: int = 3) -> Optional[diff.Fingerprint]:
    prints = []
    for _ in range(samples):
        try:
            started = __import__("time").monotonic()
            resp = client.get(url)
        except Exception:  # noqa: BLE001
            continue
        prints.append(diff.fingerprint(
            resp.status, resp.text or "", resp.headers,
            (__import__("time").monotonic() - started) * 1000))
    if not prints:
        return None
    prints.sort(key=lambda f: f.length)
    return prints[len(prints) // 2]


def confirm(url: str, names: Iterable[str], *, governor: Governor,
            client: Optional[HttpClient] = None, workers: int = 1,
            timeout: float = 20.0) -> list[Param]:
    """Send a control value per name; keep only names the server reacts to."""
    client = client or HttpClient(timeout=timeout)
    governor.spend(url)
    base = _baseline(client, url)
    if base is None:
        raise ToolError(f"no baseline for {url}")

    out: list[Param] = []
    for name in names:
        probe_url = url + ("&" if "?" in url else "?") + \
            f"{urllib.parse.quote(name)}={_CONTROL}"
        try:
            governor.spend(probe_url)
            started = __import__("time").monotonic()
            resp = client.get(probe_url)
        except Exception as exc:  # noqa: BLE001
            _log.info("param %s failed: %s", name, exc)
            continue
        probe = diff.fingerprint(
            resp.status, resp.text or "", resp.headers,
            (__import__("time").monotonic() - started) * 1000,
            markers=[_CONTROL])
        delta = diff.compare(base, probe)
        if not delta.interesting and not probe.reflected:
            continue
        out.append(Param(url=url, name=name, where="query",
                         reflected=bool(probe.reflected),
                         delta=delta.score, axes=delta.axes))
    out.sort(key=lambda p: (-p.reflected, -p.delta))
    return out


def register(registry: Any) -> None:
    context = registry.context
    home = Path(getattr(context, "home", ".") or ".")

    @registry.register(
        "param_mine",
        description="Mine parameter-name candidates from a case corpus.",
        capability=Capability.DB_READ,
        parameters={"case": "str", "extra": "str (optional) — comma list"},
    )
    def param_mine(case: str, *, extra: str = "") -> dict[str, Any]:
        found = mine(home / "data" / "evidence" / case,
                     extra=[e.strip() for e in extra.split(",") if e.strip()])
        return {"hosts": len(found),
                "candidates": {h: len(n) for h, n in found.items()}}

    @registry.register(
        "param_confirm",
        description="Confirm parameter names against a URL by control value; "
                    "returns only names the server reacts to.",
        capability=Capability.NET_OUT,
        parameters={"url": "str", "names": "str — comma list",
                    "scope": "str — scope file"},
    )
    def param_confirm(url: str, names: str, scope: str) -> dict[str, Any]:
        from .scope import load_scope
        gov = Governor(load_scope(scope), max_total=len(names.split(",")) + 10)
        found = confirm(url, [n.strip() for n in names.split(",") if n.strip()],
                        governor=gov)
        return {"url": url, "confirmed": [p.as_dict() for p in found]}
