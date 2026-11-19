"""``nm decisions`` — the decision journal from the console.

The top-level twin of chat /decisions (``agents/partner/
runtime_reach.py: _control_decisions``): same verbs, same parsing,
same rendering — list/add/resolve/review/calibration/whatif — over the
same journal at ``<home>/data/decisions.json``. The what-if
comparison renders through the shared
``planning.counterfactual.render_whatif_block``, never a second copy.
Only the command names in usage strings differ (``nm decisions`` for
``/decisions``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_decisions"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _journal(context: Any) -> Any:
    from ...decisions.journal import DecisionJournal

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return DecisionJournal(str(Path(home) / "data" / "decisions.json"))


def _rfc_store(context: Any) -> Any:
    from ...decisions.rfc import RfcStore

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return RfcStore(str(Path(home) / "data" / "rfcs.json"),
                    journal=_journal(context))


def _ledger(context: Any) -> Any:
    from ...commitments.ledger import CommitmentLedger

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return CommitmentLedger(str(Path(home) / "data" / "commitments.json"))


def _world_graph(context: Any) -> Any:
    """A fresh planning graph, synced from the commitment ledger —
    the same projection chat /decisions simulates against."""
    from ...planning.commitment_sync import sync_ledger_to_graph
    from ...planning.graph import WorldGraph

    graph = WorldGraph()
    with contextlib.suppress(Exception):
        sync_ledger_to_graph(_ledger(context), graph)
    return graph


def _whatif_block(context: Any, decision: str, options: list[str]) -> str:
    from ...planning.counterfactual import render_whatif_block, whatif_for_options

    try:
        graph = _world_graph(context)
        result = whatif_for_options(graph, options)
    except Exception as exc:  # noqa: BLE001 - simulation never fails a command
        return f"what-if unavailable: {exc}"
    return render_whatif_block(decision, graph, result)


def _resolve_decision(journal: Any, ref: str) -> Any | None:
    """Exact id, else unique id-prefix, else None — as chat does."""
    ref = (ref or "").strip()
    if ref in journal.decisions:
        return journal.decisions[ref]
    matches = [d for did, d in journal.decisions.items()
               if did.startswith(ref)]
    return matches[0] if len(matches) == 1 else None


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_decisions(args: argparse.Namespace, context: Any) -> int:
    """``nm decisions [list|add|resolve|review|calibration|whatif]``."""
    action = (getattr(args, "decisions_action", None) or "list").lower()
    rest = list(getattr(args, "decisions_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    try:
        journal = _journal(context)
    except Exception as exc:  # noqa: BLE001 - journal path is the whole command
        return _usage_error(f"decisions unavailable: {exc}")

    if action == "list":
        rows = journal.open_decisions()
        sentence = journal.calibration_sentence()
        if as_json:
            print(json.dumps({"open": [d.to_dict() for d in rows],
                              "calibration": sentence}, indent=2))
            return 0
        lines: list[str] = []
        if sentence:
            lines.append(f"🧭 {sentence}")
        if not rows:
            lines.append(
                "no open decisions — nm decisions add <decision> "
                "[| chosen | prediction | conf%] records one with "
                "its prediction, before the outcome.")
            print("\n".join(lines))
            return 0
        lines.append(f"open decisions ({len(rows)}):")
        for d in rows:
            conf = (f" · {d.confidence:.0%} sure"
                    if d.confidence is not None else "")
            chosen = f" → {d.chosen}" if d.chosen else ""
            lines.append(f"  {d.id}: {d.decision}{chosen}{conf}")
        print("\n".join(lines))
        return 0

    if action == "add":
        text = " ".join(rest).strip()
        if not text:
            return _usage_error(
                "usage: nm decisions add <decision> "
                "[| chosen | prediction | conf% | opt1; opt2; ...]")
        fields = [f.strip() for f in text.split("|")]
        decision = fields[0] if fields else ""
        if not decision:
            return _usage_error(
                "usage: nm decisions add <decision> "
                "[| chosen | prediction | conf% | opt1; opt2; ...]")
        chosen = fields[1] if len(fields) > 1 else ""
        prediction = fields[2] if len(fields) > 2 else ""
        options = ([o.strip() for o in fields[4].split(";") if o.strip()]
                   if len(fields) > 4 and fields[4] else [])
        confidence = None
        if len(fields) > 3 and fields[3]:
            raw_conf = fields[3].rstrip("%").strip()
            try:
                confidence = float(raw_conf)
            except ValueError:
                return _usage_error(
                    f"confidence {fields[3]!r} is not a number — "
                    "use a percent like 70 or a fraction like 0.7")
            if confidence > 1.0:
                confidence = confidence / 100.0
        try:
            d = journal.record(
                decision=decision, chosen=chosen,
                prediction=prediction, confidence=confidence,
                options=options)
        except ValueError as exc:
            return _usage_error(str(exc))
        reply = (f"recorded: {d.id}: {d.decision}"
                 + (f" → {d.chosen}" if d.chosen else "")
                 + " — resolve it when the outcome lands: "
                   f"nm decisions resolve {d.id} [good|bad] <outcome>")
        if options:
            # The comparison runs at record time, before the decision
            # is settled: the owner sees what each option would disturb
            # while their pick is still just a pick.
            reply += "\n" + _whatif_block(context, d.decision, options)
        print(reply)
        return 0

    if action == "resolve":
        if not rest:
            return _usage_error(
                "usage: nm decisions resolve <id> [good|bad] <outcome>")
        target = _resolve_decision(journal, rest[0])
        if target is None:
            return _usage_error(
                f"no decision matching {rest[0]!r} — "
                "nm decisions list shows ids.")
        tail_text = " ".join(rest[1:]).strip()
        success: bool | None = None
        outcome = tail_text
        lowered = tail_text.lower()
        for word, verdict in (("good", True), ("bad", False)):
            if lowered == word or lowered.startswith(word + " "):
                success = verdict
                outcome = tail_text[len(word):].strip()
                break
        try:
            d = journal.resolve(target.id, outcome=outcome, success=success)
        except KeyError as exc:
            return _usage_error(str(exc))
        lines = [f"resolved: {d.decision} → {d.outcome or 'no outcome noted'}"]
        sentence = journal.calibration_sentence()
        if sentence:
            lines.append(f"🧭 {sentence}")
        print("\n".join(lines))
        return 0

    if action == "review":
        period = (rest[0].lower() if rest else "weekly")
        if period not in ("weekly", "monthly"):
            return _usage_error(
                "usage: nm decisions review [weekly|monthly]")
        if as_json:
            print(json.dumps(journal.review(period), indent=2))
            return 0
        print(journal.render_review(period))
        return 0

    if action == "calibration":
        sentence = journal.calibration_sentence()
        if as_json:
            print(json.dumps({"calibration": sentence}, indent=2))
            return 0
        if sentence is None:
            print("no resolved decisions yet — nothing to calibrate. "
                  "Record one with nm decisions add, then resolve it "
                  "when the outcome lands.")
            return 0
        print(f"🧭 {sentence}")
        return 0

    if action == "rfc":
        from ...decisions.rfc import RfcError, handle_rfc

        try:
            store = _rfc_store(context)
            print(handle_rfc(store, list(rest), actor="owner"))
        except RfcError as exc:
            return _usage_error(str(exc))
        return 0

    if action == "whatif":
        if not rest:
            return _usage_error("usage: nm decisions whatif <id>")
        target = _resolve_decision(journal, rest[0])
        if target is None:
            return _usage_error(
                f"no decision matching {rest[0]!r} — "
                "nm decisions list shows ids.")
        if not target.options:
            print(f"no options recorded for {target.id} — record them "
                  "with nm decisions add <decision> | chosen | prediction "
                  "| conf% | opt1; opt2")
            return 0
        print(_whatif_block(context, target.decision, list(target.options)))
        return 0

    return _usage_error(
        "usage: nm decisions list · add <decision> "
        "[| chosen | prediction | conf% | opt1; opt2; ...]\n"
        "       resolve <id> [good|bad] <outcome> · "
        "review [weekly|monthly] · calibration · whatif <id>\n"
        "       rfc list · show <n> · open <title> | <kind> | <mod1; mod2> "
        "[| summary] · period <n> · comment <n> <text> · approve <n> · "
        "object <n> | <reason> | <resolves> · resolve <n> <objection-id> | "
        "<note> · accept <n> [| reason] · close <n> | <reason> · "
        "supersede <n> <title> | <kind> | <modules> · maintainers "
        "[add|remove <name>] · module <name>")
