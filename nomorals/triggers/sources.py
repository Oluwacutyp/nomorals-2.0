"""Trigger source evaluators.

Each source knows how to decide "did it fire?" for one evaluation:

* ``schedule`` — not polled here; wired into the existing scheduler
  (``Scheduler.schedule_cron`` / ``schedule_once``) by the engine, which
  registers a ``__trigger_fire__`` action handler.  This module only
  renders the scheduler call plan.
* ``file`` — hash-baseline polling.  First sight establishes the
  baseline and never fires (same semantics as ``watchers.FileKind``);
  restarts re-baseline rather than firing spuriously.
* ``price`` — keyless quote via ``integrations.market_data`` (the same
  data plane the finance layer uses), evaluated with the watchers'
  tested ``parse_condition``/``evaluate_condition`` engine.
* ``message`` — evaluated by ``TriggerEngine.on_message`` (regex + optional
  chat/sender filters).
* ``webhook`` — evaluated by ``TriggerEngine.fire_webhook``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..agents.watchers import evaluate_condition, parse_condition
from ..core.logging_setup import get_logger
from ..integrations import market_data

_log = get_logger(__name__)

#: action name the engine registers on the scheduler for schedule triggers
SCHEDULER_ACTION = "__trigger_fire__"


def schedule_plan(condition: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Render a normalized schedule condition into a scheduler call plan.

    Returns ``("cron", {"cron_expr": ...})`` or ``("once", {"run_at": ...})``.
    """
    if "once" in condition:
        return "once", {"run_at": float(condition["once"])}
    return "cron", {"cron_expr": str(condition["cron"])}


def _sha256_file(path: Path, *, max_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        remaining = max_bytes
        while remaining > 0:
            chunk = fh.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def evaluate_file(
    trigger: Any, state: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Poll a path for change/create/delete. ``state`` is the engine-held
    per-trigger memory (``exists`` / ``digest``); mutated in place."""
    cond = trigger.condition
    path = Path(cond["path"]).expanduser()
    mode = cond.get("on", "change")
    exists = path.is_file()
    prev_exists: bool | None = state.get("exists")

    if not exists:
        state["exists"] = False
        if mode == "delete" and prev_exists:
            return True, {"path": str(path), "event": "deleted"}
        # first sight of a missing path (or still missing): no fire
        note = "path missing" if prev_exists is None else "still missing"
        return False, {"path": str(path), "event": "missing", "note": note}

    digest = _sha256_file(path)
    if prev_exists is None:
        # first sight — establish the baseline, never fire
        state["exists"] = True
        state["digest"] = digest
        return False, {"path": str(path), "event": "baseline",
                       "note": "baseline established"}
    if mode == "create" and not prev_exists:
        state["exists"] = True
        state["digest"] = digest
        return True, {"path": str(path), "event": "created"}
    state["exists"] = True
    if mode == "delete":
        state["digest"] = digest
        return False, {"path": str(path), "event": "present",
                       "note": "watching for deletion"}
    if mode == "create":
        state["digest"] = digest
        return False, {"path": str(path), "event": "present",
                       "note": "watching for creation"}
    # mode == "change"
    old_digest = state.get("digest")
    state["digest"] = digest
    if old_digest != digest:
        return True, {"path": str(path), "event": "changed",
                      "old_digest": (old_digest or "")[:12],
                      "new_digest": digest[:12]}
    return False, {"path": str(path), "event": "unchanged",
                   "digest": digest[:12]}


def evaluate_price(
    trigger: Any, state: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Poll a keyless market quote and evaluate the threshold condition.

    Uses ``integrations.market_data.quote`` — the same keyless data plane
    as the finance layer — and the watchers' condition engine, so
    threshold semantics match the rest of the system (``lt``/``gt`` fire
    on the first check when already beyond the threshold; ``changed`` /
    ``changed_by_pct`` first establish a baseline).
    """
    cond = trigger.condition
    symbol = cond["symbol"]
    market = cond.get("market", "crypto")
    quote = market_data.quote(symbol, market=market)
    if not quote or quote.get("price") is None:
        raise RuntimeError(
            f"no quote for {symbol} ({market})")
    price = float(quote["price"])
    parsed = parse_condition(
        {"op": cond.get("op", "lt"), "field": "value",
         "value": cond.get("value")})
    old = state.get("last_price")
    hit, note = evaluate_condition(parsed, old=old, new=price)
    state["last_price"] = price
    evidence: dict[str, Any] = {
        "symbol": symbol, "market": market, "price": price,
        "currency": quote.get("currency"), "source": quote.get("source"),
        "note": note,
    }
    return hit, evidence


def match_message(trigger: Any, text: str, chat_key: str,
                  sender: str = "") -> tuple[bool, dict[str, Any]]:
    """Check one inbound message against a message-source trigger."""
    import re as _re

    cond = trigger.condition
    want_chat = cond.get("chat")
    if want_chat and want_chat != chat_key:
        return False, {"reason": "chat_mismatch", "chat": chat_key}
    want_sender = cond.get("sender")
    if want_sender and want_sender != sender:
        return False, {"reason": "sender_mismatch", "sender": sender}
    m = _re.search(cond["pattern"], text or "")
    if not m:
        return False, {"reason": "no_match"}
    return True, {"pattern": cond["pattern"], "chat": chat_key,
                  "matched": m.group(0)[:200]}


def match_entity_state(trigger: Any, entity_id: str,
                       old_state: Any, new_state: Any,
                       attributes: dict[str, Any] | None = None
                       ) -> tuple[bool, dict[str, Any]]:
    """Check one HA ``state_changed`` event against an entity_state trigger.

    Filters (all ANDed): ``domain`` (entity_id prefix), ``entity_id``
    (exact), ``to`` (new state), ``from`` (old state).  An empty condition
    matches every state change.
    """
    cond = trigger.condition
    entity_id = str(entity_id or "").lower()
    want_domain = cond.get("domain")
    if want_domain:
        domain = entity_id.split(".", 1)[0]
        if domain != want_domain:
            return False, {"reason": "domain_mismatch",
                           "entity_id": entity_id}
    want_entity = cond.get("entity_id")
    if want_entity and want_entity != entity_id:
        return False, {"reason": "entity_mismatch", "entity_id": entity_id}
    want_to = cond.get("to")
    if want_to is not None and str(new_state) != want_to:
        return False, {"reason": "to_mismatch", "new_state": new_state}
    want_from = cond.get("from")
    if want_from is not None and str(old_state) != want_from:
        return False, {"reason": "from_mismatch", "old_state": old_state}
    return True, {"entity_id": entity_id, "old_state": old_state,
                  "new_state": new_state}
