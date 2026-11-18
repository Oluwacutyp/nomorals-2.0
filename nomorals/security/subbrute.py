"""DNS subdomain brute with wildcard suppression and permutation.

Three rules make the difference between a usable list and noise:

* the wildcard is measured first; any name resolving to the wildcard IP
  is kept but tagged, never silently counted;
* permutations are generated from names that ALREADY resolved, so the
  second pass is targeted rather than blind;
* every candidate is deduped against the inventory before a query is
  spent on it.

Resolution is DNS over HTTPS — no resolver config, works behind a
blocked UDP/53.
"""
from __future__ import annotations

import random
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.errors import ToolError
from ..core.policy import Capability
from ..core.logging_setup import get_logger
from .recon import _SAFE, dns_lookup, wildcard_control

_log = get_logger(__name__)
__all__ = ["SubBrute", "BruteHit", "SEED_WORDS", "permutations", "register"]

#: Built-in seed list — the names that actually appear on real estates.
#: Order matters: the useful ones resolve early and feed permutations.
SEED_WORDS: tuple[str, ...] = tuple("""
www mail webmail smtp pop imap mx ns1 ns2 ns3 dns dns1 dns2
api api1 api2 apis gateway graphql rest rpc ws wss socket
dev develop development devel test testing tst qa stage staging stg
uat preprod preview demo sandbox lab labs beta alpha canary
admin administrator manage management panel cpanel whm portal
dashboard console control controlpanel internal intranet private
vpn remote access jump bastion gateway ssh sftp ftp files file
cdn static assets img images media upload uploads download downloads
blog news press docs doc wiki help support kb status monitor
monitoring metrics grafana prometheus kibana elastic logs logging
sentry ci cd jenkins build builds deploy git gitlab github bitbucket
svn repo registry docker k8s kubernetes swarm rancher
db database mysql postgres pgsql mssql mongo redis cache memcached
sql backup backups restore archive old new legacy v1 v2 v3
app apps mobile m android ios web www2 www3
shop store cart checkout pay payments billing invoice
crm erp hr payroll finance accounting reports analytics
mail2 email webdisk autodiscover autoconfig exchange owa
proxy lb loadbalancer haproxy nginx apache
staging2 dev2 test2 api-dev api-staging admin-dev
sso auth login id identity accounts oauth
""".split())

_PREFIX = ("dev", "staging", "test", "qa", "uat", "api", "admin", "new",
           "old", "beta", "demo", "internal", "vpn", "app", "mobile")
_SUFFIX = ("dev", "staging", "test", "qa", "uat", "api", "admin", "new",
           "old", "beta", "demo", "internal", "vpn", "app", "backup",
           "prod", "1", "2", "3")


@dataclass
class BruteHit:
    host: str
    addresses: list[str] = field(default_factory=list)
    cname: list[str] = field(default_factory=list)
    wildcard: bool = False
    via: str = "brute"

    def as_dict(self) -> dict[str, Any]:
        return {"host": self.host, "addresses": self.addresses,
                "cname": self.cname, "wildcard": self.wildcard, "via": self.via}


def permutations(names: Iterable[str], domain: str) -> list[str]:
    """Derive candidates from names that already resolved."""
    out: set[str] = set()
    for full in names:
        label = full[: -len(domain) - 1] if full.endswith("." + domain) else full
        if not label or "." in label:
            continue
        for p in _PREFIX:
            out.add(f"{p}-{label}.{domain}")
            out.add(f"{p}{label}.{domain}")
        for s in _SUFFIX:
            out.add(f"{label}-{s}.{domain}")
            out.add(f"{label}{s}.{domain}")
    return sorted(out)


class SubBrute:
    def __init__(self, domain: str, *, wordlist: str = "",
                 workers: int = 25, rate: float = 120.0,
                 timeout: float = 8.0, rounds: int = 2,
                 extra_words: Optional[Iterable[str]] = None,
                 stop_event: Optional[threading.Event] = None,
                 progress: bool = True) -> None:
        domain = (domain or "").strip().lower().strip(".")
        if not _SAFE.match(domain):
            raise ToolError(f"not a domain: {domain!r}")
        self.domain = domain
        self.workers = max(1, min(workers, 64))
        self.interval = 1.0 / max(1.0, rate)
        self.timeout = timeout
        self.rounds = max(1, rounds)
        self.progress = progress
        self._stop = stop_event or threading.Event()
        words = list(SEED_WORDS)
        if wordlist:
            words += self._load(wordlist)
        words += [w.strip() for w in (extra_words or []) if str(w).strip()]
        seen: set[str] = set()
        self._words = [w for w in words if w and w not in seen
                       and not seen.add(w)]
        self._lock = threading.Lock()
        self._next = time.monotonic()
        self._tried: set[str] = {domain}
        self._hits: list[BruteHit] = []
        self._queries = 0
        self.wildcard: Optional[str] = None

    @staticmethod
    def _load(path: str) -> list[str]:
        fp = Path(path)
        if not fp.is_absolute():
            fp = Path.cwd() / fp
        if not fp.is_file():
            raise ToolError(f"wordlist not found: {path}")
        return [ln.strip().lower() for ln in
                fp.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip() and not ln.startswith("#")]

    def _gate(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            time.sleep(wait)

    def _probe(self, host: str, via: str) -> Optional[BruteHit]:
        self._gate()
        with self._lock:
            self._queries += 1
        addresses = dns_lookup(host, "A", timeout=self.timeout)
        cname = dns_lookup(host, "CNAME", timeout=self.timeout)
        if not addresses and not cname:
            return None
        wildcard = bool(self.wildcard and addresses and
                        set(addresses) == {self.wildcard})
        return BruteHit(host=host, addresses=addresses, cname=cname,
                        wildcard=wildcard, via=via)

    def _sweep(self, candidates: list[str], via: str) -> int:
        fresh: list[str] = []
        with self._lock:
            for host in candidates:
                if host not in self._tried:
                    self._tried.add(host)
                    fresh.append(host)
        found = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for hit in pool.map(lambda h: self._probe(h, via), fresh):
                if hit is None:
                    continue
                with self._lock:
                    self._hits.append(hit)
                found += 1
                if self.progress:
                    tag = "~" if hit.wildcard else "+"
                    print(f"  {tag} {hit.host} "
                          f"{','.join(hit.addresses) or ','.join(hit.cname)}")
        return found

    def run(self) -> dict[str, Any]:
        self.wildcard = wildcard_control(self.domain)
        if self.progress:
            print(f"  wildcard: {self.wildcard or 'none (clean)'}")
        started = time.monotonic()
        round_hits = 0
        for rnd in range(1, self.rounds + 1):
            if self._stop.is_set():
                break
            if rnd == 1:
                candidates = [f"{w}.{self.domain}" for w in self._words]
            else:
                seed = [h.host for h in self._hits if not h.wildcard]
                candidates = permutations(seed, self.domain)
                if not candidates:
                    break
            round_hits += self._sweep(candidates, "brute" if rnd == 1 else "perm")
        real = [h for h in self._hits if not h.wildcard]
        return {
            "domain": self.domain,
            "wildcard": self.wildcard,
            "queries": self._queries,
            "seconds": round(time.monotonic() - started, 1),
            "found": len(real),
            "wildcard_matches": len(self._hits) - len(real),
            "hits": [h.as_dict() for h in self._hits],
        }


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "subbrute",
        description="DNS subdomain brute with wildcard suppression and "
                    "permutation rounds. Tags wildcard matches instead of "
                    "counting them as findings.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str — apex domain",
                    "wordlist": "str (optional) — path to extra words",
                    "workers": "int (optional, 25)",
                    "rate": "float (optional, 120) — queries/second",
                    "rounds": "int (optional, 2) — brute, then permute"},
    )
    def subbrute(domain: str, *, wordlist: str = "", workers: str = "",
                 rate: str = "", rounds: str = "") -> dict[str, Any]:
        def num(v: str, d: float) -> float:
            try:
                return float(v) if str(v).strip() else d
            except ValueError:
                return d
        brute = SubBrute(domain, wordlist=wordlist,
                         workers=int(num(workers, 25)),
                         rate=num(rate, 120.0), rounds=int(num(rounds, 2)))
        return brute.run()
