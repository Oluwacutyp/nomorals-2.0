"""Egress auditing for sandbox L2: every outbound request, logged.

Sandbox L2 confines untrusted code, but confinement without a record
is un-auditable.  :class:`EgressAuditor` is the record: every egress
request (host, port) gets a decision — allowed or blocked, with a
reason — and the decision is appended to an append-only log that is
never pruned or rewritten by this module.

Default policy (fail-closed):

* **Private / loopback / link-local / reserved / unspecified /
  multicast** destinations are blocked.  That covers RFC 1918
  (10/8, 172.16/12, 192.168/16), loopback (127/8, ::1), link-local
  (169.254/16, fe80::/10), and the rest of the non-routable space —
  so a sandboxed process cannot probe the LAN, the host, or cloud
  metadata endpoints (169.254.169.254) even when it resolves a name
  to such an address.
* Hostnames resolve via ``socket.getaddrinfo`` and *every* resolved
  address is checked — a name that resolves to both a public and a
  private address is blocked on the private one (no DNS-rebinding
  hole).  An unresolvable name is blocked, not passed: fail-closed.
* An explicit ``allowlist`` of hosts (exact or ``.suffix`` match)
  bypasses resolution checks — the operator's opt-in, recorded per
  entry with ``reason="allowlisted"``.
* Ports outside 1–65535 are blocked; port 0 is never valid egress.

The log is append-only by construction: there is no delete, clear, or
rewrite method.  ``audit_report()`` summarises it (totals, blocked
entries, breakdown by reason) for the sandbox supervisor.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EgressDecision",
    "EgressRecord",
    "EgressAuditor",
    "is_blocked_network",
    "BLOCKED_NETWORKS",
]

#: Networks egress may never reach by default (private, loopback,
#: link-local, reserved, unspecified, multicast — v4 + v6).
BLOCKED_NETWORKS: tuple[ipaddress._BaseNetwork, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::ffff:0:0/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("100::/64"),
    ipaddress.ip_network("2001::/32"),
    ipaddress.ip_network("2001:db8::/32"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
)


def is_blocked_network(ip: str) -> tuple[bool, str]:
    """True + reason when an IP literal sits in blocked space."""
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        return False, "not an IP literal"
    for net in BLOCKED_NETWORKS:
        if addr.version == net.version and addr in net:
            return True, f"{ip} is in blocked network {net}"
    return False, ""


@dataclass
class EgressDecision:
    """The verdict on one egress request."""

    host: str = ""
    port: int = 0
    allowed: bool = False
    reason: str = ""
    resolved_ips: list[str] = field(default_factory=list)
    caller: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host, "port": self.port,
            "allowed": self.allowed, "reason": self.reason,
            "resolved_ips": list(self.resolved_ips),
            "caller": self.caller,
        }


@dataclass
class EgressRecord(EgressDecision):
    """A decision plus its timestamp — one append-only log line."""

    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["timestamp"] = self.timestamp
        return out


class EgressAuditor:
    """Decide + append-only-log every egress request.

    ``allowlist`` entries are hostnames: ``"api.example.com"`` matches
    exactly, ``".example.com"`` matches any subdomain.  ``log_path``
    (optional) mirrors the in-memory log to a JSONL file, one line per
    decision, opened in append mode — never truncated.
    """

    def __init__(self, *, allowlist: tuple[str, ...] | list[str] = (),
                 block_private: bool = True,
                 log_path: str | Path | None = None,
                 resolve: bool = True) -> None:
        self.allowlist = tuple(a.strip().lower() for a in allowlist if a.strip())
        self.block_private = bool(block_private)
        self.resolve = bool(resolve)
        self.log_path = Path(log_path) if log_path else None
        self._log: list[EgressRecord] = []
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── decision ────────────────────────────────────────────────────────

    def _allowlisted(self, host: str) -> bool:
        host = host.lower()
        for entry in self.allowlist:
            if entry.startswith("."):
                if host.endswith(entry) or host == entry[1:]:
                    return True
            elif host == entry:
                return True
        return False

    def decide(self, host: str, port: int, *,
               caller: str = "") -> EgressDecision:
        """Decide an egress request without logging it. Pure policy —
        the only side effect is a DNS lookup when the host is not an
        IP literal and resolution is enabled."""
        host = (host or "").strip()
        caller = caller or ""
        if not host:
            return EgressDecision(host=host, port=port, allowed=False,
                                  reason="empty host", caller=caller)
        if not (1 <= int(port or 0) <= 65535):
            return EgressDecision(host=host, port=port, allowed=False,
                                  reason=f"invalid port {port!r}",
                                  caller=caller)
        if self._allowlisted(host):
            return EgressDecision(host=host, port=port, allowed=True,
                                  reason="allowlisted", caller=caller)
        # IP literal: check directly, no DNS needed.
        try:
            ipaddress.ip_address(host)
            is_literal = True
        except ValueError:
            is_literal = False
        if is_literal:
            if self.block_private:
                blocked, reason = is_blocked_network(host)
                if blocked:
                    return EgressDecision(
                        host=host, port=port, allowed=False, reason=reason,
                        resolved_ips=[host], caller=caller)
            return EgressDecision(host=host, port=port, allowed=True,
                                  reason="public IP literal",
                                  resolved_ips=[host], caller=caller)
        # Hostname: resolve and check EVERY address.
        if not self.resolve:
            return EgressDecision(host=host, port=port, allowed=False,
                                  reason="hostname resolution disabled — "
                                  "fail-closed",
                                  caller=caller)
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, OSError) as exc:
            return EgressDecision(host=host, port=port, allowed=False,
                                  reason=f"dns-resolution-failed: {exc}",
                                  caller=caller)
        ips: list[str] = []
        for info in infos:
            sockaddr = info[4]
            if sockaddr and sockaddr[0] not in ips:
                ips.append(str(sockaddr[0]))
        if not ips:
            return EgressDecision(host=host, port=port, allowed=False,
                                  reason="dns-resolution-failed: no addresses",
                                  caller=caller)
        if self.block_private:
            for ip in ips:
                blocked, reason = is_blocked_network(ip)
                if blocked:
                    return EgressDecision(
                        host=host, port=port, allowed=False,
                        reason=f"{reason} (via {host})",
                        resolved_ips=ips, caller=caller)
        return EgressDecision(host=host, port=port, allowed=True,
                              reason="public hostname", resolved_ips=ips,
                              caller=caller)

    # ── audit (decide + append-only log) ─────────────────────────────────

    def audit(self, host: str, port: int, *,
              caller: str = "") -> EgressDecision:
        """Decide and append the decision to the log. The log is
        append-only: entries are never removed or edited."""
        decision = self.decide(host, port, caller=caller)
        record = EgressRecord(
            host=decision.host, port=decision.port,
            allowed=decision.allowed, reason=decision.reason,
            resolved_ips=list(decision.resolved_ips),
            caller=decision.caller, timestamp=time.time())
        self._log.append(record)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")
        return decision

    def entries(self) -> list[EgressRecord]:
        """The full log, oldest first. A copy — callers cannot rewrite
        the auditor's history through it."""
        return list(self._log)

    def audit_report(self) -> dict[str, Any]:
        """Summary: totals, blocked entries, breakdown by reason."""
        total = len(self._log)
        allowed = sum(1 for r in self._log if r.allowed)
        blocked = [r.to_dict() for r in self._log if not r.allowed]
        by_reason: dict[str, int] = {}
        by_host: dict[str, int] = {}
        for r in self._log:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
            by_host[r.host] = by_host.get(r.host, 0) + 1
        return {
            "total": total,
            "allowed": allowed,
            "blocked": total - allowed,
            "blocked_entries": blocked,
            "by_reason": by_reason,
            "by_host": by_host,
            "log_path": str(self.log_path) if self.log_path else "",
        }
