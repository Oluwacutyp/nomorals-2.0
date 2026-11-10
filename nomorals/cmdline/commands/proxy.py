"""``nm proxy`` — proxy source activation + route receipts (doc 17 P4).

The CLI surface over the shipped proxy machinery — it adds no
second implementation of anything:

* ``nm proxy sources sync`` runs catalogue activation on demand
  through :meth:`ProxyLab.activate_sources` — the exact production
  path P1 wired into every refresh (fenced, recorded, honest about
  empties) — and prints the run's truth.
* ``nm proxy receipts`` renders the P2 route-receipt journal
  (:meth:`RouteReceiptJournal.recent`): every route decision the
  authority minted, outcomes folded, in operator text.

Operator text carries hosts, counts, reasons and times only. The
journal is credential-free by construction (P2), and the internal
decision/proxy ids never print here either — a receipt is read by
when, who, where and why, not by its row id.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..emit import _emit


def _home_of(context: Any) -> str:
    """The Devon home, resolved exactly as the ProxyLab resolves
    it: the context settings' home, else ``~/.nomorals``."""
    settings = getattr(context, "settings", None)
    if settings is not None:
        return str(getattr(settings, "home", "~/.nomorals"))
    return "~/.nomorals"


def _when(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S",
                             time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "never"


def _record_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """The activation record's operator-safe summary (names and
    counts only — the record carries nothing else by design)."""
    if not record:
        return None
    return {
        "epoch": record.get("epoch"),
        "catalog": record.get("catalog"),
        "catalog_version": record.get("catalog_version"),
        "catalog_size": record.get("catalog_size"),
        "first_activated_at": record.get("first_activated_at"),
        "last_run": record.get("last_run"),
    }


def _render_sync(result: dict[str, Any],
                 record: dict[str, Any] | None) -> str:
    """The sync run's truth, in operator text."""
    status = str(result.get("status") or "")
    lines: list[str] = []
    if status == "failed":
        lines.append("proxy sources sync: activation FAILED — "
                     f"{result.get('error', 'unknown error')}")
        if record:
            last = record.get("last_run") or {}
            lines.append(
                f"recorded state stands: epoch {record.get('epoch')} · "
                f"registry holds "
                f"{last.get('registry_after', '?')} sources "
                f"(last completed merge {_when(last.get('at'))})")
        else:
            lines.append("no activation has completed on this host yet")
        return "\n".join(lines)
    if status == "skipped":
        return ("proxy sources sync: skipped — another activation "
                "owns the fence right now; nothing was merged by "
                "this run (the owner's run records itself)")
    if status == "fenced":
        return ("proxy sources sync: fenced — a newer activation "
                "recorded first while this one merged; this run "
                "stood down rather than overwrite its record")
    catalog = (record or {}).get("catalog") or "SOURCES_2026"
    version = result.get("catalog_version") or \
        (record or {}).get("catalog_version") or "?"
    lines.append(f"proxy sources sync — catalogue {catalog} "
                 f"(version {version})")
    lines.append(f"status: {status} · epoch {result.get('epoch', '?')}")
    added = [str(n) for n in (result.get("added") or [])]
    methods = result.get("methods") or {}
    lines.append(f"added: {len(added)} source(s)")
    for name in added:
        method = str(methods.get(name) or "GET").upper()
        lines.append(f"  + {name}" + ("" if method == "GET"
                                      else f" ({method})"))
    lines.append(f"already present: "
                 f"{int(result.get('already_present') or 0)}")
    excluded = result.get("excluded_unverified") or []
    if excluded:
        names = ", ".join(str(e.get("name", "?")) for e in excluded)
        lines.append(f"excluded: {len(excluded)} — unverified or "
                     f"no URL ({names})")
    else:
        lines.append("excluded: 0")
    lines.append(f"registry: {result.get('registry_before', '?')} → "
                 f"{result.get('registry_after', '?')} sources")
    recovery = result.get("corrupt_recovery") or {}
    if recovery:
        lines.append(
            "corrupt registry recovered — the original bytes were "
            f"preserved as {recovery.get('backup', '?')} "
            f"({recovery.get('error', '')})")
    quarantined = result.get("yield_quarantined") or []
    if quarantined:
        lines.append("yield-quarantined (merge accounting only, "
                     "never probed): " + ", ".join(
                         str(n) for n in quarantined))
    if status == "empty":
        lines.append(
            "⚠ the registry holds NO sources after the merge — a "
            "named no-route stall, not a silent no-op: there is "
            "nothing to scrape until the catalogue or the registry "
            "changes")
    return "\n".join(lines)


def _receipt_view(row: dict[str, Any]) -> dict[str, Any]:
    """One journal decision as operator data: the journal's public
    fields only — decision/proxy ids (internal) are dropped."""
    keys = ("at", "consumer", "host", "route", "source", "tier",
            "protocol", "country", "candidates", "candidate_tiers",
            "reason_code", "reason", "selected_by", "authenticated",
            "outcome", "outcome_at", "outcome_detail")
    return {k: row.get(k) for k in keys}


def _render_receipts(rows: list[dict[str, Any]], *,
                     consumer: str = "") -> str:
    if not rows:
        note = f" for consumer {consumer!r}" if consumer else ""
        return (f"no route decisions recorded{note} yet — the "
                "receipt journal fills as traffic routes (rotation, "
                "downloads, browser, picks)")
    lines = [f"route receipts — {len(rows)} decision(s), oldest first"]
    for row in rows:
        route = str(row.get("route") or "")
        label = "NO ROUTE" if route == "none" else route
        tier = str(row.get("tier") or "")
        if tier and tier != route:
            label += f" ({tier})"
        host = str(row.get("host") or "") or "—"
        line = (f"  {_when(row.get('at'))}  "
                f"{str(row.get('consumer') or ''):<12} {label:<12} "
                f"{host}  {row.get('reason_code') or ''}")
        if row.get("outcome"):
            line += f"  → outcome: {row['outcome']}"
        lines.append(line)
    return "\n".join(lines)


def _sources_sync(args: argparse.Namespace, context: Any) -> int:
    from ...tools.proxy_sources import read_activation_record
    from ...tools.proxylab import ProxyLab

    lab = ProxyLab(context)
    result = lab.activate_sources()
    record = read_activation_record(lab.registry.path)
    payload = {"activation": result,
               "activation_record": _record_summary(record)}
    _emit(args, payload, _render_sync(result, record))
    return 1 if str(result.get("status") or "") in {"failed",
                                                    "empty"} else 0


def _receipts(args: argparse.Namespace, context: Any) -> int:
    from ...core.proxy_routing import RouteReceiptJournal

    home = Path(os.path.expanduser(_home_of(context)))
    journal = RouteReceiptJournal(home / "proxies"
                                  / "route_receipts.jsonl")
    try:
        limit = int(getattr(args, "limit", 10) or 10)
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 500))
    consumer = str(getattr(args, "consumer", "") or "")
    source = str(getattr(args, "source", "") or "")
    rows = journal.recent(limit, consumer=consumer, source=source)
    views = [_receipt_view(r) for r in rows]
    _emit(args, {"receipts": views, "count": len(views)},
          _render_receipts(rows, consumer=consumer))
    return 0


def _cmd_proxy(args: argparse.Namespace, context: Any) -> int:
    """Route ``nm proxy`` — sources sync + route receipts."""
    action = getattr(args, "proxy_action", "")
    try:
        if action == "sources":
            sub = getattr(args, "proxy_sources_action", "")
            if sub == "sync":
                return _sources_sync(args, context)
            print(f"unknown proxy sources action: {sub}",
                  file=sys.stderr)
            return 2
        if action == "receipts":
            return _receipts(args, context)
    except Exception as exc:  # noqa: BLE001 - CLI last line of defence
        print(f"proxy: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown proxy action: {action}", file=sys.stderr)
    return 2
