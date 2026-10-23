"""Library shelving — dossiers become knowledge, not desk files.

Design §2.1/§4.1: dev-wallet histories and token dossiers are
shelved as knowledge items with provenance, so the Library grows
the desk's memory between calls.  A dossier in ``dossiers.json``
is the desk's working paper; a shelved item is what Devon *knows*
— retrievable after the dossier rotates, quotable by other
surfaces, carrying where it came from.

The shelf follows the house shapes:

* items are written in the Teachings item shape (``id`` /
  ``kind`` / ``title`` / ``body`` / ``source{type, ref, label,
  passage}`` / ``created_at``) to the desk's append-only
  ``knowledge.jsonl`` (the betdesk-journal idiom: appends never
  rewrite; re-shelving appends a newer version and reads take
  the latest, so a token's record tracks its outcome);
* when the runtime context carries a db, the same record is
  upserted into the shared knowledge graph (``agents/kg.py``) —
  a node for the token, a node for its creator, a ``launched``
  edge between them, and a node per graded wallet — where
  Devon's recall can find it by name.  The mirror path is
  proven end-to-end against a live kg database by the burn-down
  integration test; in production it stays fail-soft (a dead
  db never sinks the shelf).

What gets shelved: dossiers that were **called**, that
**graduated**, or that the gate marked CALL_CANDIDATE, plus
wallet records the graph has actually graded.  The thousands of
junk launches stay in the launch ledger where they belong — a
Library that shelves everything knows nothing.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

__all__ = [
    "shelve_dossier", "shelve_wallet", "shelve_notable",
    "search_shelf", "render_shelf_hits",
]


def _key(kind: str, ref: str) -> str:
    return hashlib.sha1(f"pumpdesk\x1f{kind}\x1f{ref}".encode(
        "utf-8", "replace")).hexdigest()


def _item(kind: str, ref: str, title: str, body: str,
          label: str, now: float) -> dict[str, Any]:
    return {
        "id": f"pump-{kind}-{ref[:12]}",
        "_key": _key(kind, ref),
        "kind": kind,
        "title": title,
        "body": body,
        "theme": "pumpdesk",
        "source": {
            "type": "pumpdesk",
            "ref": ref,
            "label": label,
            "passage": body[:400],
        },
        "created_at": now,
        "served": [],
    }


def _kg_upsert(context: Any, dossier: dict[str, Any],
               verdict: Optional[dict[str, Any]]) -> bool:
    """Mirror the record into the shared knowledge graph when the
    context carries a db.  Fail-soft: the shelf is the record;
    the graph is a retrieval mirror."""
    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        return False
    try:
        from ..kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        curve = dossier.get("curve") or {}
        node = kg.upsert_node(
            dossier.get("mint", ""), type="entity",
            properties={
                "kind": "pump-token",
                "symbol": dossier.get("symbol", ""),
                "name": dossier.get("name", ""),
                "creator": dossier.get("creator", ""),
                "curve_pct": curve.get("curve_pct"),
                "market_cap_sol": curve.get("market_cap_sol"),
                "graduated": bool(curve.get("graduated")),
                "verdict": (verdict or {}).get("verdict", ""),
                "provenance": "pumpdesk dossier (chain evidence)",
            })
        creator = dossier.get("creator", "")
        if creator:
            dev = kg.upsert_node(creator, type="entity", properties={
                "kind": "pump-wallet",
                "provenance": "pumpdesk dossier (chain evidence)",
            })
            kg.link(dev.id, node.id, "launched")
        return True
    except Exception:  # noqa: BLE001 - mirror never sinks the shelf
        return False


def _kg_upsert_wallet(context: Any, rec: dict[str, Any]) -> bool:
    """Mirror one graded wallet record into the shared knowledge
    graph (same fail-soft contract as the dossier mirror: the
    shelf is the record; the graph is a retrieval mirror)."""
    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        return False
    try:
        from ..kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        kg.upsert_node(
            rec.get("wallet", ""), type="entity",
            properties={
                "kind": "pump-wallet",
                "verdict": rec.get("verdict", ""),
                "realized_sol": rec.get("realized_sol"),
                "realized_recent_sol": rec.get("realized_recent_sol"),
                "closed_trades": rec.get("closed_trades"),
                "win_rate": rec.get("win_rate"),
                "momentum_leads": rec.get("momentum_leads"),
                "provenance": "pumpdesk wallet graph (FIFO ledger "
                              "evidence)",
            })
        return True
    except Exception:  # noqa: BLE001 - mirror never sinks the shelf
        return False


def shelve_dossier(store: Any, dossier: dict[str, Any],
                   verdict: Optional[dict[str, Any]] = None,
                   context: Any = None,
                   now: Optional[float] = None) -> dict[str, Any]:
    """Shelve one dossier as a knowledge item (+ KG mirror).
    Returns the item."""
    now = time.time() if now is None else now
    from .dossier import render_dossier
    mint = dossier.get("mint", "")
    body = render_dossier(dossier)
    if verdict:
        body += (f"\ngate verdict: {verdict.get('verdict')} "
                 f"(composite {verdict.get('composite', 0):.2f}) — "
                 f"{verdict.get('evidence', '')}")
    dev = dossier.get("dev") or {}
    if dev.get("serial_rugger"):
        body += (f"\ndev is a flagged serial rugger: "
                 f"{dev.get('launches', 0)} launches, rug ratio "
                 f"{dev.get('rug_ratio', 0):.0%}")
    item = _item("pump-dossier", mint,
                 f"{dossier.get('symbol') or '?'} — "
                 f"{dossier.get('name') or 'unnamed'} (pump.fun)",
                 body,
                 f"pump desk dossier for {mint}", now)
    item["shelved_at"] = now
    # the mirror flag is set BEFORE the append: shelve_knowledge
    # serializes on write, so a flag set afterwards would live
    # only on the returned dict and never on the shelf record
    item["kg_mirrored"] = _kg_upsert(context, dossier, verdict)
    store.shelve_knowledge(item)
    return item


def shelve_wallet(store: Any, rec: dict[str, Any],
                  context: Any = None,
                  now: Optional[float] = None) -> dict[str, Any]:
    """Shelve one graded wallet record as knowledge (+ KG mirror
    when the context carries a db)."""
    now = time.time() if now is None else now
    from .graph import render_wallet
    wallet = rec.get("wallet", "")
    item = _item("pump-wallet", wallet,
                 f"wallet {wallet[:12]}… — {rec.get('verdict')} "
                 f"(pump desk graph)",
                 render_wallet(rec), "pump desk wallet graph", now)
    item["shelved_at"] = now
    item["kg_mirrored"] = _kg_upsert_wallet(context, rec)
    store.shelve_knowledge(item)
    return item


def shelve_notable(store: Any, context: Any = None,
                   now: Optional[float] = None) -> dict[str, Any]:
    """Shelve everything that earned it: called, graduated, or
    gate-candidate dossiers, and graded wallet records.  Called
    by the study cycle; idempotent by key + latest-wins reads."""
    now = time.time() if now is None else now
    called = set()
    try:
        from .calls import CallLog
        called = {c.mint for c in CallLog(store.dir).calls()}
    except Exception:  # noqa: BLE001
        called = set()
    shelved = {"dossiers": 0, "wallets": 0}
    for mint, d in store.dossiers().items():
        verdict = d.get("verdict") or {}
        graduated = bool((d.get("curve") or {}).get("graduated"))
        if (mint in called or graduated
                or verdict.get("verdict") == "CALL_CANDIDATE"):
            shelve_dossier(store, d, verdict=verdict or None,
                           context=context, now=now)
            shelved["dossiers"] += 1
    graph = store.graph()
    for wallet, rec in (graph.get("wallets") or {}).items():
        if rec.get("verdict") in ("smart", "losing") or \
                int(rec.get("momentum_leads", 0)) >= 2:
            shelve_wallet(store, rec, context=context, now=now)
            shelved["wallets"] += 1
    return shelved


def search_shelf(store: Any, query: str,
                 limit: int = 8) -> list[dict[str, Any]]:
    """Retrieve shelved knowledge: substring match over mint,
    symbol, name, title and body.  Newest first."""
    q = (query or "").strip().lower()
    if not q:
        return []
    hits = []
    for item in store.knowledge():
        hay = " ".join([
            item.get("title", ""), item.get("body", ""),
            (item.get("source") or {}).get("ref", ""),
            (item.get("source") or {}).get("label", ""),
        ]).lower()
        if q in hay:
            hits.append(item)
    hits.sort(key=lambda it: -(it.get("shelved_at")
                               or it.get("created_at") or 0))
    return hits[:limit]


def render_shelf_hits(hits: list[dict[str, Any]], query: str) -> str:
    if not hits:
        return (f"library shelf: nothing shelved matches {query!r} — "
                f"the desk shelves called, graduated and candidate "
                f"dossiers plus graded wallets; the rest stays in "
                f"the launch ledger")
    L = [f"📚 library shelf — {len(hits)} item(s) for {query!r}:"]
    for it in hits:
        src = it.get("source") or {}
        L.append(f"• {it.get('title', '')} "
                 f"[{it.get('kind', '')}, {src.get('label', '')}]")
        body = (it.get("body") or "").splitlines()
        for line in body[:3]:
            L.append(f"    {line}")
    return "\n".join(L)
