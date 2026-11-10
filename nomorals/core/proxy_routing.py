"""One route authority + the route-decision receipt journal.

Doc 17 Pass 2 (Proxy Sources Activation, gap G3).  Proxy selection
*logic* already exists and is not touched here: the vault pool's
``select()``/``rotate()``/``sticky_acquire()``
(:mod:`nomorals.connectors.proxypool`) is rich and tier-gated, and the
lab rotation manager (:mod:`nomorals.tools.proxylab`) ranks its scraped
pool with domain affinity.  What did not exist is (a) a single place
where a route decision is *minted* and (b) a durable receipt for it —
the §3 rule that every route decision, including a refusal, leaves a
record naming the route actually used, the source and tier, the real
reason, and the candidates that were considered.

This module is that place, and it is a facade: the authority delegates
every selection to the shipped machinery and journals the outcome.
It never invents a reason — pool receipts carry the strategy the pool
itself reports, lab receipts carry the manager's own rationale — and
it never sees a credential: receipts are built only from public
fields (id / host / port / protocol / tier / country, plus a boolean
``authenticated``), never from ``url_with_auth``/``proxy_url``/
``password``, and lab receipts record the endpoint host, never the
raw URL.

The journal is an append-only JSONL file living beside the rest of
the proxy state (``<home>/proxies/route_receipts.jsonl``), following
the doc-16 route-journal discipline: a journal write failure is
recorded through the failure ledger and NEVER breaks the fetch path
(fail-open — the decision is returned exactly as the underlying
machinery produced it).

Outcomes are separate appends referencing a ``decision_id``: the
authority cannot observe a request's fate at decision time, so a
decision row is minted with ``outcome: None`` and a later reporter
(core.http's success/error reporters, wired by the rotation manager)
appends the outcome row for the decision that handed out the endpoint
that was actually used.  No outcome is ever fabricated at decision
time.

Layer: core (L1).  Pools and lab managers are duck-typed/injected —
this module imports nothing above L1.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .failures import record_failure
from .logging_setup import get_logger
from .proxy_tiers import TierGate, normalize_tier

__all__ = [
    "ROUTE_DIRECT",
    "ROUTE_PAID",
    "ROUTE_FREE",
    "ROUTE_NONE",
    "OUTCOME_USED",
    "OUTCOME_FAILED",
    "OUTCOME_BYPASSED",
    "RouteReceipt",
    "RouteReceiptJournal",
    "RouteAuthority",
    "default_receipts_path",
]

_log = get_logger(__name__)

#: Route kinds (doc §3): the route actually used, not merely a
#: candidate that was supplied.
ROUTE_DIRECT = "direct"
ROUTE_PAID = "paid"
ROUTE_FREE = "free"
ROUTE_NONE = "none"

#: Outcome vocabulary for the separate outcome appends.
OUTCOME_USED = "used"
OUTCOME_FAILED = "failed"
OUTCOME_BYPASSED = "bypassed"

JOURNAL_FILENAME = "route_receipts.jsonl"


def default_receipts_path() -> Path:
    """The shared journal location: ``<home>/proxies/route_receipts.jsonl``.

    ``home`` resolves exactly as the proxy lab's does — ``$NM_HOME``
    when set, else ``~/.nomorals`` — so every consumer's receipts land
    in the one journal beside ``working.json``/``sources.json``.
    """
    home = Path(os.path.expanduser(os.environ.get("NM_HOME", "~/.nomorals")))
    return home / "proxies" / JOURNAL_FILENAME


@dataclass
class RouteReceipt:
    """One route decision, as journaled.  Credential-free by
    construction: there is no field that could hold one."""

    decision_id: str
    at: float
    consumer: str
    route: str
    reason_code: str
    reason: str = ""
    host: str = ""
    source: str = ""
    tier: str = ""
    proxy_id: str = ""
    protocol: str = ""
    country: str = ""
    candidates: int = 0
    candidate_tiers: dict[str, int] = field(default_factory=dict)
    sticky_key: str = ""
    expires_at: float | None = None
    selected_by: str = ""
    authenticated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "decision",
            "decision_id": self.decision_id,
            "at": self.at,
            "consumer": self.consumer,
            "host": self.host,
            "route": self.route,
            "source": self.source,
            "tier": self.tier,
            "proxy_id": self.proxy_id,
            "protocol": self.protocol,
            "country": self.country,
            "candidates": self.candidates,
            "candidate_tiers": dict(self.candidate_tiers),
            "reason_code": self.reason_code,
            "reason": self.reason,
            "sticky_key": self.sticky_key,
            "expires_at": self.expires_at,
            "selected_by": self.selected_by,
            "authenticated": self.authenticated,
            "outcome": None,
        }


class RouteReceiptJournal:
    """Append-only JSONL store for route receipts + outcome appends.

    Every write is fail-open: a broken journal returns ``False`` and
    records the failure; it never raises into a routing path.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def append(self, receipt: RouteReceipt) -> bool:
        return self._append_row(receipt.to_dict())

    def append_outcome(self, decision_id: str, outcome: str,
                       detail: str = "") -> bool:
        """Append the outcome of an earlier decision (separate row,
        folded back onto the decision by :meth:`recent`)."""
        if not decision_id:
            return False
        return self._append_row({
            "type": "outcome",
            "decision_id": decision_id,
            "at": time.time(),
            "outcome": outcome,
            "detail": (detail or "")[:200],
        })

    def _append_row(self, row: dict[str, Any]) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            return True
        except Exception as exc:  # noqa: BLE001 - journal must never break routing
            record_failure("proxy_routing.journal_append", exc)
            return False

    def recent(self, limit: int = 50, *, consumer: str = "",
               source: str = "") -> list[dict[str, Any]]:
        """The newest ``limit`` decisions, oldest first, with any
        outcome appends folded onto their decision row (``outcome`` /
        ``outcome_at`` / ``outcome_detail``).  Filters match the
        decision's ``consumer`` / ``source`` exactly.  A missing or
        partly corrupt journal reads as what could be parsed — never
        an exception."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001 - reads fail soft too
            record_failure("proxy_routing.journal_read", exc)
            return []
        decisions: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("type") == "decision":
                did = str(row.get("decision_id") or "")
                if not did:
                    continue
                if consumer and row.get("consumer") != consumer:
                    continue
                if source and row.get("source") != source:
                    continue
                decisions[did] = row
                order.append(did)
            elif row.get("type") == "outcome":
                target = decisions.get(str(row.get("decision_id") or ""))
                if target is not None:
                    target["outcome"] = row.get("outcome")
                    target["outcome_at"] = row.get("at")
                    target["outcome_detail"] = row.get("detail", "")
        keep = order[-max(0, int(limit)):] if limit else order
        return [decisions[did] for did in keep]


class RouteAuthority:
    """The one place route decisions are minted + journaled.

    Constructed with the journal to write to; pools are passed per
    call (duck-typed: ``select``/``rotate``/``sticky_acquire``/
    ``healthy_proxies`` as the connector exposes them).
    """

    def __init__(self, journal: RouteReceiptJournal) -> None:
        self.journal = journal

    # -- minting -----------------------------------------------------

    def record(self, *, consumer: str, route: str, reason_code: str,
               reason: str = "", host: str = "", source: str = "",
               tier: str = "", proxy_id: str = "", protocol: str = "",
               country: str = "", candidates: int = 0,
               candidate_tiers: dict[str, int] | None = None,
               sticky_key: str = "", expires_at: float | None = None,
               selected_by: str = "",
               authenticated: bool = False) -> RouteReceipt:
        """Mint + journal one decision made by another organ (the lab
        rotation manager's paths).  Never raises."""
        receipt = RouteReceipt(
            decision_id="pr-" + uuid.uuid4().hex[:16],
            at=time.time(),
            consumer=consumer,
            route=route,
            reason_code=reason_code,
            reason=(reason or "")[:300],
            host=host,
            source=source,
            tier=tier,
            proxy_id=proxy_id,
            protocol=protocol,
            country=country,
            candidates=int(candidates or 0),
            candidate_tiers=dict(candidate_tiers or {}),
            sticky_key=sticky_key,
            expires_at=expires_at,
            selected_by=selected_by,
            authenticated=bool(authenticated),
        )
        self.journal.append(receipt)
        return receipt

    def record_outcome(self, decision_id: str, outcome: str,
                       detail: str = "") -> bool:
        """Append the observed outcome of an earlier decision."""
        return self.journal.append_outcome(decision_id, outcome, detail)

    # -- the pool decision -------------------------------------------

    def decide_pool(self, pool: Any, *, consumer: str, host: str = "",
                    strategy: str | None = None, sticky_key: str = "",
                    sticky_ttl: float = 600.0,
                    require_tier: str | None = None,
                    carries_credentials: bool = False) -> dict[str, Any]:
        """Decide a route through a vault-pool-shaped object and
        journal the receipt.

        Delegation, exactly: a ``sticky_key`` goes to the pool's
        ``sticky_acquire``; otherwise the pool's ``select`` when it
        has one (its own default strategy unless ``strategy`` is
        given); a rotate-only pool gets ``rotate()`` + the TierGate
        composed here (the composition the browser service used to
        inline).  Returns the pool's details dict *unchanged* (the
        same object the pool produced).  A tier refusal or an empty
        pool journals a ``route="none"`` receipt and re-raises the
        original exception untouched, so callers keep the shipped
        typed-refusal shape.
        """
        candidates, candidate_tiers = self._pool_snapshot(pool)
        try:
            if sticky_key and callable(getattr(pool, "sticky_acquire", None)):
                details = pool.sticky_acquire(
                    sticky_key, ttl=sticky_ttl,
                    require_tier=require_tier,
                    carries_credentials=carries_credentials)
                if "sticky_ttl" in details:
                    reason_code = "sticky_new_pin"
                    reason = (f"new sticky pin for {sticky_key!r} "
                              f"(ttl {float(sticky_ttl):.0f}s)")
                    expires_at = time.time() + float(sticky_ttl)
                else:
                    reason_code = "sticky_hit"
                    reason = f"sticky pin for {sticky_key!r} still eligible"
                    expires_at = None
                selected_by = "sticky_acquire"
            elif callable(getattr(pool, "select", None)):
                if strategy:
                    details = pool.select(
                        strategy, require_tier=require_tier,
                        carries_credentials=carries_credentials)
                else:
                    details = pool.select(
                        require_tier=require_tier,
                        carries_credentials=carries_credentials)
                used = str(details.get("selection_strategy") or "p2c")
                reason_code = f"select:{used}"
                reason = f"pool select via {used} over {candidates} healthy"
                expires_at = None
                selected_by = "select"
            else:
                details = pool.rotate()
                TierGate.require(
                    normalize_tier((details or {}).get("tier", "free")),
                    carries_credentials=carries_credentials,
                    require_tier=require_tier)
                reason_code = "rotate:round_robin"
                reason = f"pool rotate over {candidates} healthy"
                expires_at = None
                selected_by = "rotate"
        except Exception as exc:
            from .proxy_tiers import TierGateRefusal

            code = ("tier_gate_refusal" if isinstance(exc, TierGateRefusal)
                    else "pool_no_route")
            self.record(
                consumer=consumer, route=ROUTE_NONE, reason_code=code,
                reason=str(exc), host=host, source="proxypool",
                sticky_key=sticky_key, selected_by="authority",
                authenticated=carries_credentials)
            raise
        details = details or {}
        tier = normalize_tier(details.get("tier", "free"))
        tags = [str(t) for t in (details.get("tags") or [])]
        self.record(
            consumer=consumer,
            route=tier if tier in (ROUTE_PAID, ROUTE_FREE) else ROUTE_DIRECT,
            reason_code=reason_code, reason=reason, host=host,
            source="proxylab" if "proxylab" in tags else "proxypool",
            tier=tier, proxy_id=str(details.get("id", "")),
            protocol=str(details.get("protocol", "")),
            country=str(details.get("country", "") or ""),
            candidates=candidates, candidate_tiers=candidate_tiers,
            sticky_key=sticky_key, expires_at=expires_at,
            selected_by=selected_by,
            authenticated=bool(details.get("has_auth")
                               or details.get("username")))
        return details

    @staticmethod
    def _pool_snapshot(pool: Any) -> tuple[int, dict[str, int]]:
        """Healthy-candidate count + tier histogram from the pool's
        own public listing.  Best-effort: any failure reads as an
        empty snapshot rather than breaking the decision."""
        try:
            healthy = pool.healthy_proxies()
        except Exception:  # noqa: BLE001 - snapshot is evidence, not a gate
            return 0, {}
        tiers: dict[str, int] = {}
        for row in healthy or []:
            t = normalize_tier((row or {}).get("tier", "free"))
            tiers[t] = tiers.get(t, 0) + 1
        return len(healthy or []), tiers
