"""Chat control for /immune — one front door over nomorals.immune.

Never raises. Owner-only for quarantine / disarm / arrest (enforced
by the partner runtime gate); ``status`` / ``scan`` / ``antibodies
list`` / ``drift report`` / ``casefile list`` are readable.

Usage:
    /immune status
    /immune scan <text>
    /immune antibodies list | add <pattern> <reason> | retire <id> <reason> | test <pattern>
    /immune quarantine <artifact text>
    /immune disarm <grant-id>
    /immune arrest <process-id> [system]
    /immune fever status | start <reason> <seconds> | end
    /immune drift add <tokens> <spawned> <grants> <errors> | report
    /immune casefile list | add <text> | seal <id-prefix>

State is in-memory per chat key (same as /tutor): antibody store,
police vault, fever, drift monitor, case files, receipts. The
constitution lives in police.py — this control only routes; it never
grants capability, and ``arrest`` without the ``system`` keyword
returns a report-only receipt.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import logging

_log = logging.getLogger(__name__)

__all__ = ["control_immune", "IMMUNE_USAGE", "get_immune_state", "reset_immune_state"]

#: Normal owner activity — the thymus self-corpus. Antibodies whose
#: pattern fires on any of these are rejected (autoimmune).
_SELF_EXEMPLARS = [
    "owner sends a message to a friend",
    "owner plays music",
    "owner searches the web",
    "owner reads email",
    "owner checks the calendar",
    "owner sets a reminder",
    "owner starts a game",
    "owner opens a document",
]


def IMMUNE_USAGE() -> str:
    return (
        "🛡️ /immune — defensive immune system (revoke-only, never acquire)\n"
        "  /immune status — antibodies, fever, drift, receipts\n"
        "  /immune scan <text> — innate scan of inbound content\n"
        "  /immune antibodies list | add <pattern> <reason> | retire <id> <reason> | test <pattern>\n"
        "  /immune quarantine <artifact> — vault, never delete (owner-only)\n"
        "  /immune disarm <grant-id> — revoke a capability grant (owner-only)\n"
        "  /immune arrest <process-id> [system] — system processes only (owner-only)\n"
        "  /immune fever status | start <reason> <seconds> | end\n"
        "  /immune drift add <tokens> <spawned> <grants> <errors> | report\n"
        "  /immune casefile list | add <text> | seal <id-prefix>"
    )


class _ImmuneState:
    """Per-chat in-memory immune state."""

    def __init__(self) -> None:
        from .antibodies import AntibodyStore
        from .casefile import CaseFile
        from .drift import BehaviourDriftMonitor
        from .fever import Fever
        from .police import Police
        from .thymus import SelfCorpus, Thymus

        corpus = SelfCorpus()
        for text in _SELF_EXEMPLARS:
            corpus.add(text=text)
        self.thymus = Thymus(corpus)
        self.store = AntibodyStore()
        self.vault: List[Any] = []
        self.police = Police(vault_fn=self.vault.append)
        self.fever = Fever()
        self.fever_announcements: List[str] = []
        self.drift = BehaviourDriftMonitor()
        self.casefiles: List[CaseFile] = []
        self.receipts: List[Any] = []

    def _record(self, receipt: Any) -> None:
        self.receipts.append(receipt)


_STATES: Dict[str, _ImmuneState] = {}


def get_immune_state(chat_key: str = "") -> _ImmuneState:
    key = chat_key or "default"
    if key not in _STATES:
        _STATES[key] = _ImmuneState()
    return _STATES[key]


def reset_immune_state(chat_key: str = "") -> None:
    _STATES.pop(chat_key or "default", None)


def control_immune(tail: str, context: Any = None, chat: Any = None,
                   sender_id: str = "", sender: str = "",
                   chat_key: str = "") -> str:
    """``/immune`` — one front door over nomorals.immune. Never raises."""
    try:
        return _control_immune(tail, chat_key=chat_key)
    except Exception as exc:  # noqa: BLE001 — never raise from chat
        return f"Immune hit an error ({exc})."


def _control_immune(tail: str, *, chat_key: str = "") -> str:
    state = get_immune_state(chat_key)
    raw = (tail or "").strip()
    if not raw or raw.lower() == "help":
        return IMMUNE_USAGE()
    verb, _, rest = raw.partition(" ")
    verb = verb.lower()
    rest = rest.strip()
    if verb == "status":
        return _status(state)
    if verb == "scan":
        return _scan(state, rest)
    if verb == "antibodies":
        return _antibodies(state, rest)
    if verb == "quarantine":
        return _quarantine(state, rest)
    if verb == "disarm":
        return _disarm(state, rest)
    if verb == "arrest":
        return _arrest(state, rest)
    if verb == "fever":
        return _fever(state, rest)
    if verb == "drift":
        return _drift(state, rest)
    if verb == "casefile":
        return _casefile(state, rest)
    return IMMUNE_USAGE()


def _status(state: _ImmuneState) -> str:
    from .baseline import AdaptiveBaseline  # noqa: F401 — documents the adaptive gate

    active = state.store.list_active()
    drift = state.drift.current_report()
    lines = [
        "🛡️ immune status",
        f"  antibodies: {len(active)} active",
        f"  fever: {'ACTIVE — ' + state.fever.reason if state.fever.active else 'off'}",
        f"  drift: {drift.verdict} ({drift.days_considered} days)",
        f"  quarantined artifacts: {len(state.vault)} (vault, never deleted)",
        f"  case files: {len(state.casefiles)}",
        f"  receipts: {len(state.receipts)}",
        "  constitution: revoke-only — no grant/acquire; owner work is report-only.",
        "  cancer gate: adaptive p95 baseline (report-only until 30 live samples).",
    ]
    return "\n".join(lines)


def _scan(state: _ImmuneState, text: str) -> str:
    if not text:
        return "usage: /immune scan <text>"
    # Second-exposure matching against accepted antibodies (innate
    # speed, compiled regexes). Catalog-pattern live evaluation needs
    # injected redteam patterns — see innate.evaluate_live.
    hits = state.store.match(text)
    state._record(state.police.observe({"target": "scan", "text": text}, "scan"))
    if not hits:
        return "🛡️ scan: no antibody matched — no known pathogen."
    lines = ["🛡️ scan: antibody match (second exposure, innate speed):"]
    for hit in hits:
        lines.append(
            f"  • {hit.get('id', '?')[:8]} pattern={hit.get('pattern', '?')!r} "
            f"reason={hit.get('reason', '')}"
        )
    return "\n".join(lines)


def _antibodies(state: _ImmuneState, rest: str) -> str:
    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()
    if verb == "list" or not verb:
        active = state.store.list_active()
        if not active:
            return "antibodies: none active."
        lines = ["antibodies (active):"]
        for entry in active:
            data = entry if isinstance(entry, dict) else {}
            lines.append(
                f"  • {data.get('id', '?')[:8]} pattern={data.get('pattern', '?')!r} "
                f"class={data.get('pathogen_class', '?')} reason={data.get('reason', '')}"
            )
        return "\n".join(lines)
    if verb == "add":
        pattern, _, reason = tail.partition(" ")
        pattern = pattern.strip()
        reason = reason.strip()
        if not pattern or not reason:
            return "usage: /immune antibodies add <pattern> <reason>"
        try:
            candidate = state.thymus.propose(pattern, "virus", reason)
            accepted = state.thymus.accept(candidate)
        except ValueError as exc:
            return f"antibody rejected: {exc}"
        state.store.add(accepted)
        return (
            f"antibody accepted: id={accepted.id[:8]} pattern={pattern!r} "
            f"reason={reason}"
        )
    if verb == "retire":
        ab_id, _, reason = tail.partition(" ")
        if not ab_id.strip() or not reason.strip():
            return "usage: /immune antibodies retire <id> <reason>"
        entry = state.store.get(ab_id.strip())
        if entry is None:
            # prefix match on the short id shown by list
            for candidate in state.store.all():
                if str(candidate.get("id", "")).startswith(ab_id.strip()):
                    entry = candidate
                    break
        if entry is None:
            return f"no antibody {ab_id!r}."
        from .thymus import Antibody

        antibody = Antibody(
            id=str(entry.get("id", "")),
            pattern=str(entry.get("pattern", "")),
            pathogen_class=str(entry.get("pathogen_class", "")),
            reason=str(entry.get("reason", "")),
        )
        state.thymus.retire(antibody, reason.strip())
        full = dict(entry)
        full["status"] = "retired"
        full["retire_reason"] = antibody.retire_reason
        state.store.update(full)
        return f"antibody {ab_id} retired: {reason.strip()}"
    if verb == "test":
        if not tail:
            return "usage: /immune antibodies test <pattern>"
        from .thymus import test_antibody

        try:
            fires, matched = test_antibody(tail, state.thymus.corpus)
        except ValueError as exc:
            return f"invalid pattern: {exc}"
        if fires:
            return (
                f"pattern {tail!r} FIRES on self ({len(matched)} exemplar(s)) — "
                "would be rejected (autoimmune)."
            )
        return f"pattern {tail!r} does not fire on self — thymus would accept it."
    return IMMUNE_USAGE()


def _quarantine(state: _ImmuneState, artifact: str) -> str:
    if not artifact:
        return "usage: /immune quarantine <artifact text>"
    receipt = state.police.quarantine({"id": artifact[:80], "text": artifact})
    state._record(receipt)
    return (
        f"quarantined: {artifact[:80]}\n"
        f"  vault now holds {len(state.vault)} artifact(s) — nothing deleted, evidence kept."
    )


def _disarm(state: _ImmuneState, grant_id: str) -> str:
    if not grant_id:
        return "usage: /immune disarm <grant-id>"
    receipt = state.police.disarm(grant_id.strip())
    state._record(receipt)
    return f"disarmed: grant {grant_id.strip()} revoked (receipt {receipt.id[:8]})."


def _arrest(state: _ImmuneState, rest: str) -> str:
    parts = rest.split()
    if not parts:
        return "usage: /immune arrest <process-id> [system]"
    process_id = parts[0]
    if "system" not in [p.lower() for p in parts[1:]]:
        receipt = state.police.respond(
            {"target": process_id, "process_id": process_id},
            owner_directed=False, target_kind="owner_work",
            reason="arrest without 'system' — treated as owner work: report-only",
        )
        state._record(receipt)
        return (
            f"arrest refused: {process_id} not marked as a system process. "
            "Report-only receipt left. Use `/immune arrest <id> system` for "
            "a system runaway — owner work is never arrested."
        )
    receipt = state.police.arrest(process_id)
    state._record(receipt)
    return (
        f"arrested: {process_id} (receipt {receipt.id[:8]}). "
        "No kill wired in chat — receipt only; wire kill_fn to enforce."
    )


def _fever(state: _ImmuneState, rest: str) -> str:
    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()
    if verb == "status" or not verb:
        if state.fever.active:
            return (
                f"fever ACTIVE: {state.fever.reason} "
                f"(max {state.fever.max_seconds:g}s, "
                f"throttled {state.fever.actions_throttled})"
            )
        return "fever: off."
    if verb == "start":
        # /immune fever start <reason...> <seconds> — seconds is the last token
        tokens = tail.split()
        if len(tokens) < 2:
            return "usage: /immune fever start <reason> <seconds>"
        try:
            seconds = float(tokens[-1])
        except ValueError:
            return "usage: /immune fever start <reason> <seconds>"
        reason = " ".join(tokens[:-1])
        try:
            message = state.fever.start(
                reason, seconds,
                announce_fn=state.fever_announcements.append,
            )
        except ValueError as exc:
            return f"fever not started: {exc}"
        return f"🌡️ {message}"
    if verb == "end":
        if not state.fever.active:
            return "no fever active."
        report = state.fever.end()
        return (
            f"fever ended: {report.note or 'throttled, did not diagnose or fix'} "
            f"(throttled {report.actions_throttled}, resolved={report.resolved})"
        )
    return IMMUNE_USAGE()


def _drift(state: _ImmuneState, rest: str) -> str:
    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()
    if verb == "add":
        parts = tail.split()
        if len(parts) != 4:
            return "usage: /immune drift add <tokens> <spawned> <grants> <errors>"
        try:
            tokens, spawned, grants, errors = (float(p) for p in parts)
        except ValueError:
            return "usage: /immune drift add <tokens> <spawned> <grants> <errors> (numbers)"
        report = state.drift.add_day(
            time.strftime("%Y-%m-%d"),
            {"tokens": tokens, "processes_spawned": spawned,
             "grants_held": grants, "errors": errors},
        )
        return f"drift day recorded: verdict={report.verdict} signals={report.signals or 'none'}"
    if verb == "report" or not verb:
        report = state.drift.current_report()
        lines = [f"drift: {report.verdict} ({report.days_considered} days)"]
        lines += [f"  • {reason}" for reason in report.reasons]
        return "\n".join(lines)
    return IMMUNE_USAGE()


def _casefile(state: _ImmuneState, rest: str) -> str:
    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()
    if verb == "list" or not verb:
        if not state.casefiles:
            return "case files: none."
        lines = ["case files:"]
        for case in state.casefiles:
            lines.append(
                f"  • {case.id[:8]} sealed={case.sealed} "
                f"evidence={len(case.evidence)} timeline={len(case.timeline)}"
            )
        return "\n".join(lines)
    if verb == "add":
        if not tail:
            return "usage: /immune casefile add <text>"
        from .casefile import build_case

        case = build_case({"text": tail})
        case.add_evidence(tail)
        state.casefiles.append(case)
        return f"case file {case.id[:8]} opened with 1 evidence item."
    if verb == "seal":
        if not tail:
            return "usage: /immune casefile seal <id-prefix>"
        for case in state.casefiles:
            if case.id.startswith(tail):
                case.seal()
                return f"case file {case.id[:8]} sealed — no further adds."
        return f"no case file starting with {tail!r}."
    return IMMUNE_USAGE()
