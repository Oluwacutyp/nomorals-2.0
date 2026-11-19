"""Deterministic scope safety for the BugHunter engagement engine.

Port of ``engine/scope.py`` from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma) — the safety core, kept
verbatim in semantics: deny wins, default deny, patterns are bare
domains (apex + subdomains), ``*.host`` wildcards, exact hosts,
IPv4 CIDR, or ``re:`` regexes, plus suffix-confusion guards.

Devon differences from upstream:

- No CLI ``_main`` (Devon surfaces this through the ``security``
  brain tool, not a ``scope.py`` command) and no ``sys.exit`` —
  callers get booleans and reasons.
- ``Scope.allows_private`` is new: True when an in-scope pattern
  is a CIDR or names a private/loopback host, so the engine may
  pass ``allow_private_ips=True`` to Devon's HttpClient for a
  scope that explicitly authorizes it — never by default.
- Errors go through :mod:`nomorals.core.logging_setup`, not prints.
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ...core.logging_setup import get_logger

__all__ = [
    "Scope",
    "scope_selftest",
]

_log = get_logger(__name__)


def _host_of(target: str) -> str:
    t = (target or "").strip()
    if "://" not in t:
        t = "//" + t
    try:
        parsed = urlparse(t)
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _match(pattern: str, host: str) -> bool:
    p = (pattern or "").strip()
    if not p or not host:
        return False
    if p[:3].lower() == "re:":
        try:
            return re.search(p[3:], host, re.IGNORECASE) is not None
        except re.error:
            return False
    p = p.lower()
    if "/" in p and p.replace(".", "").replace("/", "").isdigit():
        try:
            return ipaddress.ip_address(host) in ipaddress.ip_network(
                p, strict=False)
        except ValueError:
            return False
    if p.startswith("*."):
        base = p[2:]
        return host.endswith("." + base)
    # Bare domain: apex or any subdomain; or exact host.
    return host == p or host.endswith("." + p)


def _is_private_host(host: str) -> bool:
    if not host:
        return False
    if host in {"localhost"} or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


class Scope:
    """One engagement's in/out-of-scope rules. Deny wins, default deny."""

    def __init__(self, in_scope: Any, out_of_scope: Any = None,
                 seeds: Any = None, name: str = "engagement") -> None:
        self.in_scope = [p for p in (in_scope or []) if str(p).strip()]
        self.out_of_scope = [p for p in (out_of_scope or [])
                             if str(p).strip()]
        self.seeds = list(seeds or [])
        self.name = name

    @classmethod
    def load(cls, path: str | Path) -> Scope:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("in_scope", []), data.get("out_of_scope", []),
                   data.get("seeds", []), data.get("name", "engagement"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scope:
        return cls(data.get("in_scope", []), data.get("out_of_scope", []),
                   data.get("seeds", []),
                   data.get("name", "engagement"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "in_scope": list(self.in_scope),
                "out_of_scope": list(self.out_of_scope),
                "seeds": list(self.seeds)}

    def in_scope_host(self, target: str) -> bool:
        host = _host_of(target)
        if not host:
            return False
        if any(_match(p, host) for p in self.out_of_scope):
            return False  # deny wins
        return any(_match(p, host) for p in self.in_scope)

    def reject_reason(self, target: str) -> str | None:
        """None if in scope, else a human reason (for logging)."""
        host = _host_of(target)
        if not host:
            return "could not parse host"
        if any(_match(p, host) for p in self.out_of_scope):
            return f"{host} matches an out-of-scope rule"
        if not any(_match(p, host) for p in self.in_scope):
            return f"{host} matches no in-scope rule (default deny)"
        return None

    def allows_private(self) -> bool:
        """True when the scope itself authorizes private address space.

        A CIDR in-scope rule, or an in-scope pattern naming a
        private/loopback host, is an explicit authorization for the
        engine to let Devon's HttpClient reach private IPs for this
        engagement only. A scope of public domains never does.
        """
        for pattern in self.in_scope:
            p = str(pattern).strip()
            if "/" in p and p.replace(".", "").replace("/", "").isdigit():
                return True
            if _is_private_host(_host_of(p)):
                return True
        return False


def _rule_from_token(token: str) -> str:
    """Normalize one scope.md bullet token into a host rule.

    ``https://app.example.com/login?x=1`` -> ``app.example.com``.
    Explicit-rule forms (``re:``, CIDR, ``*.host``) and already-bare
    hosts are kept as-is.
    """
    t = token.strip()
    if t[:3].lower() == "re:" or ("://" not in t
                                  and not t.startswith("//")):
        return t
    try:
        host = urlparse(t if "://" in t else "//" + t).hostname or ""
    except ValueError:
        return t
    return host or t


def scope_from_md(path: str | Path) -> Scope:
    """Build a Scope from a scaffolded scope.md.

    Collects bullet items under an '## In scope' / '## Out of scope'
    heading. Takes the first whitespace token of each bullet (so
    '- api.example.com — note' yields 'api.example.com'). Skips
    placeholder bullets like '(paste ... here)'. A token written as
    a URL is reduced to its host, because every rule is matched
    against the host anyway.
    """
    in_scope, out_scope, section = [], [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith("#"):
                hashes = len(low) - len(low.lstrip("#"))
                if "out of scope" in low or "out-of-scope" in low:
                    section = "out"
                elif "in scope" in low or "in-scope" in low:
                    section = "in"
                elif hashes <= 2:
                    section = None  # a peer-level section
                # Deeper sub-headings keep the current section.
                continue
            s = line.strip()
            if section and (s.startswith("- ") or s.startswith("* ")):
                tok = s[2:].strip().split()[0:1]
                if not tok:
                    continue
                pat = tok[0].strip("`\"'")
                if pat.startswith("**") and pat.endswith("**") \
                        and len(pat) > 4:
                    pat = pat[2:-2]
                if not pat or pat.startswith("(") or pat.startswith("["):
                    continue  # placeholder text, not a real pattern
                (in_scope if section == "in"
                 else out_scope).append(_rule_from_token(pat))
    return Scope(in_scope=in_scope, out_of_scope=out_scope, name=str(path))


def scope_selftest() -> bool:
    """Upstream's built-in self-test, verbatim in cases."""
    s = Scope(in_scope=["example.com", "*.test.example.com",
                        "10.0.0.0/8", "re:^lab[0-9]+\\.acme\\.io$"],
              out_of_scope=["admin.example.com",
                            "internal.example.com"])
    ok = s.in_scope_host
    assert ok("https://example.com/login")           # apex
    assert ok("http://api.example.com/x")            # subdomain
    assert ok("https://a.test.example.com")          # wildcard
    assert ok("https://10.1.2.3:8080/")              # CIDR
    assert ok("https://lab42.acme.io")               # regex
    assert not ok("https://admin.example.com")       # deny wins
    assert not ok("https://internal.example.com/x")  # deny wins
    assert ok("https://test.example.com")            # bare rule
    assert not ok("https://evil.com")                # default deny
    assert not ok("https://notexample.com")          # suffix guard
    assert not ok("https://example.com.evil.com")    # suffix guard
    assert not ok("https://11.0.0.1")                # outside CIDR
    assert s.reject_reason("https://evil.com")
    assert s.reject_reason("https://example.com") is None
    s2 = Scope(in_scope=["*.test.example.com"])
    assert s2.in_scope_host("a.test.example.com")
    assert not s2.in_scope_host("test.example.com")
    # Devon's addition: private-space authorization is scope-bound.
    assert Scope(in_scope=["127.0.0.1"]).allows_private()
    assert Scope(in_scope=["10.0.0.0/8"]).allows_private()
    assert not Scope(in_scope=["example.com"]).allows_private()
    _log.info("bughunter scope self-test: PASS")
    return True
