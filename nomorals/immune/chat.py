"""Chat control for /immune — one front door over nomorals.immune.

Never raises. Owner-only for quarantine / disarm / arrest (enforced
by the partner runtime gate); ``status`` / ``scan`` / ``antibodies
list`` / ``drift report`` / ``casefile list`` are readable.

Usage:
    /immune status
    /immune scan <text>
    /immune antibodies list | add <pattern> <reason> | promote <id> | retire <id> <reason> | test <pattern>
    /immune corpus | corpus harvest
    /immune feeds
    /immune mark-fp <antibody-id> [reason]
    /immune quarantine <artifact text>
    /immune disarm <grant-id>
    /immune arrest <process-id> [system]
    /immune fever status | start <reason> <seconds> | end
    /immune drift add <tokens> <spawned> <grants> <errors> | report
    /immune casefile list | add <text> | seal <id-prefix>
    /immune gate [spawned growth]

State is in-memory per chat key (same as /tutor), EXCEPT the
antibody store: since doc 20 P3 it is the shared, path-backed
store (immune/wiring.py ``shared_antibody_store``) — one immune
memory for the whole organism, durable across restarts, with the
P3 lifecycle on every record (learned antibodies enter on
probation, observe-only, and promote on evidence or the owner's
verb). The self-corpus is shared the same way since doc 20 P4
(immune/wiring.py ``shared_corpus_store``): versioned data —
the seed exemplars plus exemplars harvested from Devon's own
verified fixes and owner-directed police receipts, each with
provenance, and negative selection cites what a candidate
matched. Since doc 20 P5 the drift/fever/false-positive
observation record is shared and durable the same way (the
append-only journal in immune/feeds.py, wiring-shared): what
those organs observe survives restart, and the tool failure
ledger's recurrences arrive as case files through the failure
feed, evaluated lazily on the status / casefile / feeds reads.
Police vault, fever, drift monitor, case files and
receipts stay per-chat in-memory. The adaptive cancer gate is
also shared: its baseline, incident state and quarantine count
live in the shared persisted monitor (immune/wiring.py), fed by
the durable sampler job — ``status`` reports that live state, and
this chat's fever is registered with the monitor so its samples
quarantine like any incident. The constitution lives in
police.py — this control only routes; it never grants capability,
and ``arrest`` without the ``system`` keyword returns a
report-only receipt.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import logging

from .corpus import SEED_TEXTS

_log = logging.getLogger(__name__)

__all__ = ["control_immune", "IMMUNE_USAGE", "get_immune_state", "reset_immune_state"]

#: Normal owner activity — the thymus self-corpus SEEDS. Antibodies
#: whose pattern fires on any exemplar are rejected (autoimmune).
#: Since doc 20 P4 the list is owned by immune/corpus.py (the
#: corpus's floor); the corpus itself is versioned data that also
#: carries exemplars harvested from Devon's own records. This name
#: stays as the seed list's alias for existing readers.
_SELF_EXEMPLARS = SEED_TEXTS


def IMMUNE_USAGE() -> str:
    return (
        "🛡️ /immune — defensive immune system (revoke-only, never acquire)\n"
        "  /immune status — antibodies, fever, drift, receipts\n"
        "  /immune scan <text> — innate scan of inbound content\n"
        "  /immune antibodies list | add <pattern> <reason> | retire <id> <reason> | test <pattern>\n"
        "  /immune corpus — self-corpus: version + counts by provenance\n"
        "  /immune corpus harvest — grow the self-corpus from verified fixes + owner-directed receipts\n"
        "  /immune feeds — sensor feeds: failure-ledger recurrences + the observation journal\n"
        "  /immune mark-fp <antibody-id> [reason] — owner false-positive verdict on an antibody (owner-only)\n"
        "  /immune quarantine <artifact> — vault, never delete (owner-only)\n"
        "  /immune disarm <grant-id> — revoke a capability grant (owner-only)\n"
        "  /immune arrest <process-id> [system] — system processes only (owner-only)\n"
        "  /immune fever status | start <reason> <seconds> | end\n"
        "  /immune drift add <tokens> <spawned> <grants> <errors> | report\n"
        "  /immune casefile list | add <text> | seal <id-prefix>\n"
        "  /immune gate [spawned growth] — consult the adaptive cancer gate "
        "(live sample, or probe with numbers)"
    )


def _default_scan_patterns() -> list[dict[str, Any]]:
    """The live-scan catalog: the immune package's own pathogen data.

    Materializes :data:`immune.pathogens.PATHOGEN_CLASSES` (each class
    carries a content-regex hint) and the immune-evasion
    :data:`immune.pathogens.ATTACK_SCENARIOS` into the
    ``{id, class_id, pattern, severity}`` dicts
    :func:`immune.innate.evaluate_live` consumes. The red-team
    catalog in ``agents/`` holds scenario payloads, not content
    regexes; this in-package data is the catalog's immune class
    source (pathogens.py is written for the catalog to consume).
    A caller may inject a richer pattern set per state instead.
    """
    from .pathogens import ATTACK_SCENARIOS, PATHOGEN_CLASSES

    severities = {
        "virus": "high", "worm": "high", "trojan": "high",
        "cancer": "medium", "parasite": "medium", "prion": "medium",
        "opportunistic": "medium", "autoimmune": "medium",
        "allergy": "low",
    }
    patterns: list[dict[str, Any]] = []
    for cls in PATHOGEN_CLASSES:
        hint = cls.get("innate_pattern_hint") or ""
        if hint:
            patterns.append({
                "id": f"pathogen:{cls['id']}",
                "class_id": cls["id"],
                "pattern": hint,
                "severity": severities.get(cls["id"], "medium"),
            })
    for scenario in ATTACK_SCENARIOS:
        patterns.append({
            "id": scenario["id"],
            "class_id": scenario["class_id"],
            "pattern": scenario["pattern"],
            "severity": "high",
        })
    return patterns


def _severity_label_fn(hit: Any):
    """The deterministic default labeller for one innate hit.

    The adaptive layer never calls an LLM itself — it takes an
    INJECTED ``label_fn``. This is the chat surface's deterministic
    classifier (the contract allows "an LLM, a human, or a
    deterministic classifier"): verdict from the catalog hit's own
    severity — critical/high is a real threat, medium a variant, low
    (the allergy class: urgency bait) is noise, and noise never
    creates an antibody. A caller-injected ``label_fn`` on the state
    replaces this wholesale.
    """

    def _label(bundle_dict: dict[str, Any]) -> dict[str, Any]:
        sev = str(getattr(hit, "severity", "medium") or "medium")
        if sev in ("critical", "high"):
            verdict, confidence = "real_threat", 0.9
        elif sev == "medium":
            verdict, confidence = "variant", 0.6
        else:
            verdict, confidence = "noise", 0.4
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": (
                f"innate catalog hit {hit.pattern_id} ({hit.class_id}, "
                f"severity {sev}) on live-scanned content"),
        }

    return _label


def _emit_event(state: _ImmuneState, kind: str,
                payload: dict[str, Any]) -> None:
    """Emit one immune event (AA-42): recorded on the state and, when
    a sink is injected, pushed to the organ bus. Never raises — a
    broken sink must not break the scan; the event still lands in
    ``state.events`` and the failure is logged."""
    from .events import emit_immune_event, immune_event

    try:
        event = emit_immune_event(state.event_sink, kind, payload)
    except Exception:  # noqa: BLE001 — reporting never breaks a scan
        _log.warning("immune event sink failed for %s", kind,
                     exc_info=True)
        event = immune_event(kind, payload)
    state.events.append(event)
    del state.events[:-50]


class _ImmuneState:
    """Per-chat in-memory immune state."""

    def __init__(self) -> None:
        from .casefile import CaseFile
        from .drift import BehaviourDriftMonitor
        from .fever import Fever
        from .police import Police
        from .thymus import Thymus
        from .wiring import shared_antibody_store

        # Doc 20 P4: the self-corpus is the organism's shared,
        # persisted, VERSIONED data (immune/corpus.py) — the seed
        # exemplars plus whatever harvest has admitted from
        # Devon's own verified fixes and owner-directed receipts.
        # The thymus holds the store's live corpus object, so a
        # harvest on any surface counts on every surface at once.
        # A store that cannot load must not break the surface:
        # fall back to a private in-memory seed-only store and say
        # so in the log (same posture as the antibody store).
        try:
            from .wiring import shared_corpus_store

            self.corpus_store = shared_corpus_store()
        except Exception:  # noqa: BLE001 — surface never breaks
            _log.warning("immune self-corpus store unavailable — "
                         "falling back to in-memory seed-only",
                         exc_info=True)
            from .corpus import CorpusStore

            self.corpus_store = CorpusStore()
        self.thymus = Thymus(self.corpus_store.corpus())
        # Doc 20 P3: the antibody store is the organism's shared,
        # path-backed immune memory — NOT per-chat in-memory. A
        # store that cannot load must not break the surface: fall
        # back to a private in-memory store and say so in the log
        # (status still renders; learning this session is volatile,
        # never silently presented as durable).
        try:
            self.store = shared_antibody_store()
        except Exception:  # noqa: BLE001 — surface never breaks
            _log.warning("immune antibody store unavailable — "
                         "falling back to in-memory", exc_info=True)
            from .antibodies import AntibodyStore

            self.store = AntibodyStore()
        # Doc 20 P5: the observation journal is the organism's
        # shared, durable record of what the drift and fever
        # organs observed and what false-positive verdicts landed
        # (immune/feeds.py ObservationLog, wiring-shared). Same
        # fallback posture as the stores: a journal that cannot
        # load must not break the surface — fall back to a private
        # in-memory log and say so in the log.
        try:
            from .wiring import shared_observation_log

            self.observations = shared_observation_log()
        except Exception:  # noqa: BLE001 — surface never breaks
            _log.warning("immune observation log unavailable — "
                         "falling back to in-memory", exc_info=True)
            from .feeds import ObservationLog

            self.observations = ObservationLog()
        self.vault: List[Any] = []
        self.police = Police(vault_fn=self.vault.append)
        self.fever = Fever()
        self.fever_announcements: List[str] = []
        self.drift = BehaviourDriftMonitor()
        self.casefiles: List[CaseFile] = []
        self.receipts: List[Any] = []
        #: Live-scan wiring (AA-41/42): the innate catalog patterns
        #: scanned on every /immune scan, the injected adaptive
        #: labeller (None = the deterministic severity labeller), the
        #: injected organ-bus event sink, and the events emitted here.
        self.scan_patterns: list[dict[str, Any]] = _default_scan_patterns()
        self.label_fn: Any = None
        self.event_sink: Any = None
        self.events: list[dict[str, Any]] = []

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
    # The antibody store is shared organism memory (doc 20 P3), so
    # resetting a chat state also clears it — a reset that left
    # learned antibodies in place would not be a reset. Resetting
    # is a test/support operation; production code never calls it.
    try:
        from .wiring import clear_shared_antibody_store

        clear_shared_antibody_store()
    except Exception:  # noqa: BLE001 — reset stays best-effort
        _log.warning("immune antibody store reset failed", exc_info=True)
    # The self-corpus is shared organism data the same way (doc 20
    # P4): a reset returns it to its seed-only floor.
    try:
        from .wiring import clear_shared_corpus_store

        clear_shared_corpus_store()
    except Exception:  # noqa: BLE001 — reset stays best-effort
        _log.warning("immune self-corpus store reset failed",
                     exc_info=True)
    # The observation journal is shared organism data the same
    # way (doc 20 P5): a reset empties it.
    try:
        from .wiring import clear_shared_observation_log

        clear_shared_observation_log()
    except Exception:  # noqa: BLE001 — reset stays best-effort
        _log.warning("immune observation log reset failed",
                     exc_info=True)


def control_immune(tail: str, context: Any = None, chat: Any = None,
                   sender_id: str = "", sender: str = "",
                   chat_key: str = "", event_sink: Any = None,
                   scan_patterns: Any = None,
                   label_fn: Any = None) -> str:
    """``/immune`` — one front door over nomorals.immune. Never raises.

    Live-scan injections (AA-41/42), all optional: ``scan_patterns``
    replaces the innate catalog set for this chat's state,
    ``label_fn`` is the adaptive layer's injected labeller, and
    ``event_sink`` receives immune events (organ bus). They persist
    on the per-chat state so later scans in the same chat use them.
    """
    try:
        return _control_immune(tail, chat_key=chat_key, context=context,
                               event_sink=event_sink,
                               scan_patterns=scan_patterns,
                               label_fn=label_fn)
    except Exception as exc:  # noqa: BLE001 — never raise from chat
        return f"Immune hit an error ({exc})."


def _monitor_for(context: Any = None):
    """The shared live gate monitor for this surface.

    Resolved from the context's settings (same store the sampler job
    feeds). Returns None when the monitor cannot load — the surface
    then says so honestly instead of printing a gate claim with no
    state behind it.
    """
    try:
        from . import wiring

        return wiring.shared_monitor(wiring.default_store_path(context))
    except Exception:  # noqa: BLE001 — status degrades honestly
        _log.warning("immune monitor unavailable", exc_info=True)
        return None


def _control_immune(tail: str, *, chat_key: str = "",
                    context: Any = None, event_sink: Any = None,
                    scan_patterns: Any = None,
                    label_fn: Any = None) -> str:
    state = get_immune_state(chat_key)
    if event_sink is not None:
        state.event_sink = event_sink
    if scan_patterns is not None:
        state.scan_patterns = list(scan_patterns)
    if label_fn is not None:
        state.label_fn = label_fn
    monitor = _monitor_for(context)
    if monitor is not None:
        # This chat's fever is an incident like any other: while it
        # runs, the shared baseline quarantines instead of learning.
        try:
            monitor.attach_fever(state.fever)
        except Exception:  # noqa: BLE001 — never break the surface
            pass
    raw = (tail or "").strip()
    if not raw or raw.lower() == "help":
        return IMMUNE_USAGE()
    verb, _, rest = raw.partition(" ")
    verb = verb.lower()
    rest = rest.strip()
    if verb == "status":
        return _status(state, monitor, context)
    if verb == "scan":
        return _scan(state, rest)
    if verb == "antibodies":
        return _antibodies(state, rest)
    if verb == "corpus":
        return _corpus(state, rest, context)
    if verb == "feeds":
        return _feeds(state, context)
    if verb == "mark-fp":
        return _mark_fp(state, rest)
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
        return _casefile(state, rest, context)
    if verb == "gate":
        return _gate(state, monitor, rest)
    return IMMUNE_USAGE()


def _evaluate_feeds(state: _ImmuneState, context: Any = None,
                    ) -> dict[str, Any] | None:
    """One lazy failure-feed evaluation over this state's case
    files (doc 20 P5).

    Runs on the status / casefile / feeds reads — the surfaces
    where a human is looking at immune state. The sampler tick
    cannot carry it (its context has no db and no case
    collection), and no thread is started for it. Returns the
    feed's report, or None when evaluation itself failed — the
    failure is logged, never surfaced as a crash.
    """
    from .feeds import evaluate_failure_feed

    db = getattr(context, "db", None) if context is not None else None
    try:
        return evaluate_failure_feed(db, state.casefiles)
    except Exception:  # noqa: BLE001 — a feed never breaks a surface
        _log.warning("immune failure-feed evaluation failed",
                     exc_info=True)
        return None


def _status(state: _ImmuneState, monitor: Any = None,
            context: Any = None) -> str:
    _evaluate_feeds(state, context)
    active = state.store.list_active()
    probation = state.store.list_probation()
    antibody_line = f"  antibodies: {len(active)} active"
    if probation:
        antibody_line += (
            f" · {len(probation)} on probation (observe-only)")
    drift = state.drift.current_report()
    lines = [
        "🛡️ immune status",
        antibody_line,
        f"  fever: {'ACTIVE — ' + state.fever.reason if state.fever.is_active() else 'off'}",
        f"  drift: {drift.verdict} ({drift.days_considered} days)",
        f"  quarantined artifacts: {len(state.vault)} (vault, never deleted)",
        f"  case files: {len(state.casefiles)}",
        f"  receipts: {len(state.receipts)}",
        f"  live scan: {len(state.scan_patterns)} catalog patterns · "
        f"{len(state.events)} immune event(s) emitted",
        "  constitution: revoke-only — no grant/acquire; owner work is report-only.",
    ]
    corpus_store = getattr(state, "corpus_store", None)
    if corpus_store is not None:
        counts = corpus_store.counts_by_provenance()
        lines.append(
            f"  self-corpus: {sum(counts.values())} exemplars "
            f"(v{corpus_store.version} — seed {counts.get('seed', 0)} · "
            f"verified fixes {counts.get('verified_fix', 0)} · "
            f"owner receipts {counts.get('owner_receipt', 0)})")
    lines += _gate_status_lines(monitor)
    return "\n".join(lines)


def _gate_status_lines(monitor: Any) -> List[str]:
    """The adaptive gate's REAL live state — measured, never claimed.

    Samples recorded, baseline ready vs report-only with the count,
    current p95 thresholds when ready, quarantined incident samples,
    whether the sampler is actually ticking, and the store behind it.
    """
    if monitor is None:
        return ["  cancer gate: adaptive p95 baseline — monitor "
                "unavailable, no live state to report."]
    st = monitor.status()
    if st["ready"]:
        head = (
            f"  cancer gate: adaptive p95 baseline — {st['samples']} samples "
            f"· READY (spawned p95 {st['spawned_p95']:g} · "
            f"growth p95 {st['growth_p95']:g})"
        )
    else:
        head = (
            f"  cancer gate: adaptive p95 baseline — "
            f"{st['samples']}/{st['min_samples']} samples · report-only "
            f"({st['threshold_reason']})"
        )
    age = st["last_tick_age_s"]
    sampling_down = st.get("sampling_available") is False
    if sampling_down:
        # Never present a live sampler where no sample can land:
        # the tick may be running, but sampling itself is down.
        sampler = "ticking, but process sampling is UNAVAILABLE " \
                  "on this platform"
    elif st["sampler_running"]:
        sampler = "running"
    elif age is None:
        sampler = "no tick yet"
    else:
        sampler = f"stale (last tick {age:.0f}s ago)"
    lines = [
        head,
        f"  sampler: {sampler} (every {st['sampler_interval_s']:.0f}s) · "
        f"quarantined: {st['quarantined']} incident sample(s) · "
        f"store: {st['store_path']}",
    ]
    if sampling_down:
        lines.append(
            "  sampling: UNAVAILABLE — "
            f"{st.get('sampling_reason') or 'no backend could count processes'}; "
            "no process sample has ever been collected on this host, "
            "so the baseline cannot learn here")
    if st["incident_active"]:
        cause = "gate enforcement" if st["gate_incident"] else "fever"
        lines.append(
            f"  incident: ACTIVE ({cause}) — samples are quarantined, "
            "not learned, until it closes")
    return lines


def _gate(state: _ImmuneState, monitor: Any, rest: str) -> str:
    """``/immune gate [spawned growth]`` — the live decision path.

    No numbers: run one sampler tick NOW (collect the real process
    sample, consult the gate, learn or quarantine it). Two numbers:
    a probe — consult the gate on that sample as a system process;
    probes are never learned (typed numbers are not live behaviour).
    """
    if monitor is None:
        return "🛡️ gate: monitor unavailable — no live gate state."
    rest = rest.strip()
    if not rest:
        out = monitor.tick()
        if not out["sampled"]:
            if out.get("unavailable"):
                return ("🛡️ gate: process sampling unavailable on this "
                        f"platform — {out.get('error') or out.get('reason') or ''} "
                        "Nothing consulted, nothing learned.")
            return ("🛡️ gate: no live process sample available on this "
                    "platform — nothing consulted, nothing learned.")
        sample = out.get("sample") or {}
        desc = (f"spawned {float(sample.get('processes_spawned', 0) or 0):g}, "
                f"growth {float(sample.get('process_growth', 0) or 0):g}")
        if out.get("enforced"):
            return (f"🛡️ gate: ENFORCE — {out['reason']} · live sample "
                    f"({desc}) quarantined; incident open.")
        if out.get("incident"):
            return (f"🛡️ gate: incident active — live sample ({desc}) "
                    f"quarantined, not learned "
                    f"({out['quarantined']} quarantined total).")
        return (f"🛡️ gate: live sample ({desc}) — report-only, learned "
                f"({out['samples']} samples in the baseline).")
    parts = rest.split()
    if len(parts) != 2:
        return "usage: /immune gate [spawned growth]"
    try:
        spawned, growth = (float(p) for p in parts)
    except ValueError:
        return "usage: /immune gate [spawned growth] (numbers)"
    result = monitor.consult(
        {"process_id": "chat-gate-probe", "processes_spawned": spawned,
         "process_growth": growth},
        is_system_process=True)
    state._record(result["receipt"])
    if result["action"] == "enforce":
        return (f"🛡️ gate probe: ENFORCE — {result['reason']} · incident "
                "opened; the probe sample was not learned.")
    return (f"🛡️ gate probe: report-only — {result['reason']} · the probe "
            "sample was not learned.")


def _scan(state: _ImmuneState, text: str) -> str:
    if not text:
        return "usage: /immune scan <text>"
    # Second-exposure matching against stored antibodies (innate
    # speed, compiled regexes). Matching is complete — probation
    # antibodies match too, and every hit is recorded on its
    # record as evidence — but the RESPONSE seam reads active
    # only (doc 20 P3): a probation hit is reported as an
    # observation and drives nothing.
    hits = state.store.match(text)
    state._record(state.police.observe({"target": "scan", "text": text}, "scan"))
    lines: list[str] = []
    enforcing = [h for h in hits if h.get("status") == "active"]
    observing = [h for h in hits if h.get("status") != "active"]
    if enforcing:
        lines.append("🛡️ scan: antibody match (second exposure, innate speed):")
        for hit in enforcing:
            lines.append(
                f"  • {hit.get('id', '?')[:8]} pattern={hit.get('pattern', '?')!r} "
                f"reason={hit.get('reason', '')}"
            )
    if observing:
        from .antibodies import promotion_policy

        min_matches, min_window = promotion_policy()
        lines.append(
            "🛡️ scan: probation antibody observed (match recorded as "
            "evidence — NOT enforced; probation antibodies only "
            "observe until promoted):")
        for hit in observing:
            lines.append(
                f"  • {hit.get('id', '?')[:8]} pattern={hit.get('pattern', '?')!r} "
                f"matches={hit.get('probation_matches', 0)}/{min_matches} "
                f"toward promotion (clean window >={min_window:.0f}s) "
                f"reason={hit.get('reason', '')}"
            )
    # First exposure (AA-42): the innate layer evaluates the catalog
    # patterns LIVE on this content — not just in CI. Suspicious hits
    # are presented to the adaptive layer (AA-41): an INJECTED
    # labeller (or the deterministic severity labeller) returns a
    # verdict, and a threat verdict seeds a real antibody through the
    # thymus (negative selection still applies — a pattern that fires
    # on self is rejected, never stored).
    from .innate import evaluate_live

    innate_hits = evaluate_live(text, state.scan_patterns)
    if innate_hits:
        _emit_event(state, "threat_detected", {
            "source": "chat-scan",
            "hits": [{"pattern_id": h.pattern_id, "class_id": h.class_id,
                      "severity": h.severity} for h in innate_hits[:10]],
        })
        lines.append("🛡️ scan: innate catalog hit (first exposure, live):")
        for hit in innate_hits[:10]:
            lines.append(
                f"  • {hit.pattern_id} class={hit.class_id} "
                f"severity={hit.severity} matched={hit.matched_text[:60]!r}")
        lines.extend(_label_and_learn(state, text, innate_hits))
    if not lines:
        return "🛡️ scan: no antibody matched — no known pathogen."
    return "\n".join(lines)


#: Cap on new antibodies per scan — one pasted blob must not flood
#: the store; remaining hits stay reported, unlearned.
_MAX_LEARN_PER_SCAN = 3


def _label_and_learn(state: _ImmuneState, text: str,
                     innate_hits: list[Any]) -> list[str]:
    """Adaptive labelling of innate hits (AA-41).

    Each distinct catalog pattern that hit is presented as an antigen
    and labelled; ``real_threat`` / ``variant`` verdicts seed an
    antibody (the catalog pattern itself) via thymus → store, and an
    ``antibody_created`` event is emitted. ``noise`` never creates an
    antibody, by construction. Thymus rejections (the pattern fires
    on self) are reported honestly, never stored.
    """
    from .adaptive import label_antigen, present, should_create_antibody

    patterns_by_id = {str(p.get("id")): p for p in state.scan_patterns}
    # A pattern already in memory — active OR on probation — is
    # already covered: re-proposing it would mint a duplicate
    # record instead of letting the existing one accrue evidence.
    known_patterns = {
        str(entry.get("pattern", ""))
        for entry in state.store.all()
        if entry.get("status") != "retired"
    }
    lines: list[str] = []
    seen: set[str] = set()
    learned = 0
    for hit in innate_hits:
        if hit.pattern_id in seen:
            continue
        seen.add(hit.pattern_id)
        bundle = present({
            "content": text[:500],
            "organ": "immune.chat",
            "source": "chat-scan",
            "capability": f"innate:{hit.class_id}",
        })
        labeller = state.label_fn or _severity_label_fn(hit)
        try:
            labelled = label_antigen(bundle, labeller)
        except Exception as exc:  # noqa: BLE001 — a bad labeller must
            lines.append(  # not take the scan down with it
                f"  ⚠ adaptive labelling failed for {hit.pattern_id}: {exc}")
            continue
        lines.append(
            f"  • adaptive: {hit.pattern_id} → {labelled.verdict} "
            f"(confidence {labelled.confidence:.2f}) — {labelled.reasoning}")
        if not should_create_antibody(labelled):
            continue
        if learned >= _MAX_LEARN_PER_SCAN:
            continue
        pattern = str((patterns_by_id.get(hit.pattern_id) or {}).get(
            "pattern") or "")
        if not pattern or pattern in known_patterns:
            continue  # already covered by a stored antibody
        reason = (f"adaptive: innate hit {hit.pattern_id} labelled "
                  f"{labelled.verdict} — {labelled.reasoning}")
        try:
            candidate = state.thymus.propose(pattern, hit.class_id, reason)
            accepted = state.thymus.accept(candidate)
        except ValueError as exc:
            lines.append(
                f"  ⚠ thymus rejected antibody for {hit.pattern_id} "
                f"(fires on self): {exc}")
            continue
        # Doc 20 P3: a LEARNED antibody enters on probation. The
        # thymus's accept is its negative-selection verdict, not a
        # promotion — the store's add_learned records the standing.
        state.store.add_learned(accepted)
        known_patterns.add(pattern)
        learned += 1
        _emit_event(state, "antibody_created", {
            "antibody_id": accepted.id, "pattern": pattern,
            "pathogen_class": hit.class_id, "verdict": labelled.verdict,
            "source": "chat-scan", "status": "probation",
        })
        lines.append(
            f"  🧬 antibody created (probation — observe-only until "
            f"it earns promotion): {accepted.id[:8]} "
            f"pattern={pattern!r} class={hit.class_id}")
    return lines


def _antibodies(state: _ImmuneState, rest: str) -> str:
    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()
    if verb == "list" or not verb:
        active = state.store.list_active()
        probation = state.store.list_probation()
        if not active and not probation:
            return "antibodies: none active."
        lines: list[str] = []
        if active:
            lines.append("antibodies (active):")
            for entry in active:
                data = entry if isinstance(entry, dict) else {}
                lines.append(
                    f"  • {data.get('id', '?')[:8]} pattern={data.get('pattern', '?')!r} "
                    f"class={data.get('pathogen_class', '?')} reason={data.get('reason', '')}"
                )
        if probation:
            from .antibodies import promotion_policy

            min_matches, min_window = promotion_policy()
            lines.append("antibodies (probation — observe-only):")
            for entry in probation:
                data = entry if isinstance(entry, dict) else {}
                lines.append(
                    f"  • {data.get('id', '?')[:8]} pattern={data.get('pattern', '?')!r} "
                    f"class={data.get('pathogen_class', '?')} "
                    f"matches={data.get('probation_matches', 0)}/{min_matches} "
                    f"clean-window>={min_window:.0f}s "
                    f"reason={data.get('reason', '')}"
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
        # Owner-directed: the owner's own add IS the vetting (the
        # thymus still ran negative selection) — the record keeps
        # the accepted standing, stamped with its origin. Only
        # LEARNED antibodies enter on probation (doc 20 P3).
        state.store.add(accepted, origin="owner")
        return (
            f"antibody accepted: id={accepted.id[:8]} pattern={pattern!r} "
            f"reason={reason}"
        )
    if verb == "promote":
        if not tail:
            return "usage: /immune antibodies promote <id>"
        entry = state.store.get(tail.strip())
        if entry is None:
            # prefix match on the short id shown by list
            for candidate in state.store.all():
                if str(candidate.get("id", "")).startswith(tail.strip()):
                    entry = candidate
                    break
        if entry is None:
            return f"no antibody {tail.strip()!r}."
        try:
            record = state.store.promote(
                str(entry.get("id", "")),
                reason="owner promotion via /immune antibodies promote")
        except ValueError as exc:
            return f"promotion refused: {exc}"
        return (
            f"antibody {str(entry.get('id', ''))[:8]} promoted to active "
            f"(version {record.get('version', '?')}): {record.get('status_reason', '')}"
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
        # This surface's retire IS the owner's false-positive
        # verdict (its event below is named for it): record the
        # outcome on the antibody's own record (doc 20 P3), so
        # hit/miss evidence survives the retirement. demote=False
        # — the retirement in this same act supersedes demotion.
        state.store.record_false_positive(
            str(entry.get("id", "")), reason=reason.strip(),
            demote=False)
        _emit_event(state, "false_positive_retired", {
            "antibody_id": str(entry.get("id", "")),
            "reason": reason.strip(), "source": "chat",
        })
        return f"antibody {ab_id} retired: {reason.strip()}"
    if verb == "test":
        if not tail:
            return "usage: /immune antibodies test <pattern>"
        from .thymus import exemplar_citation, test_antibody

        try:
            fires, matched = test_antibody(tail, state.thymus.corpus)
        except ValueError as exc:
            return f"invalid pattern: {exc}"
        if fires:
            cited = "; ".join(exemplar_citation(e) for e in matched)
            return (
                f"pattern {tail!r} FIRES on self ({len(matched)} "
                f"exemplar(s): {cited}) — would be rejected (autoimmune)."
            )
        return f"pattern {tail!r} does not fire on self — thymus would accept it."
    return IMMUNE_USAGE()


def _corpus(state: _ImmuneState, rest: str, context: Any = None) -> str:
    """``/immune corpus`` — the self-corpus as versioned data.

    Read (no subverb): the corpus version and its exemplar counts
    by provenance — the corpus's size is a reported fact, never a
    silent constant. ``harvest``: one harvest rung over Devon's own
    records (the incident journal's verified fixes, read from the
    journal file this deployment's error system writes, plus THIS
    surface's recorded owner-directed police receipts), admitting
    what is new as one new corpus version. Idempotent — a second
    harvest over the same records adds zero.
    """
    verb = (rest or "").strip().lower()
    store = state.corpus_store
    if verb in ("", "status", "show"):
        counts = store.counts_by_provenance()
        lines = [
            f"🛡️ self-corpus v{store.version}: "
            f"{sum(counts.values())} exemplars — "
            f"seed {counts.get('seed', 0)} · "
            f"verified fixes {counts.get('verified_fix', 0)} · "
            f"owner receipts {counts.get('owner_receipt', 0)}",
            f"  store: {store.path or 'in-memory only (not persisted)'}",
            "  negative selection tests every candidate against all "
            "of these; a match is rejected with the exemplar cited.",
        ]
        if store.degraded_reason:
            lines.append(f"  ⚠ {store.degraded_reason}")
        return "\n".join(lines)
    if verb == "harvest":
        from .wiring import default_journal_path

        report = store.harvest(
            journal_path=default_journal_path(context),
            receipts=list(state.receipts))
        lines = [
            f"🛡️ corpus harvest: +{report['added']} exemplar(s) — "
            f"corpus v{report['version_before']} → "
            f"v{report['version_after']} ({report['total']} total)",
        ]
        labels = {
            "verified_fix": "verified fixes (incident journal)",
            "owner_receipt": "owner-directed receipts (this surface)",
        }
        for kind in ("verified_fix", "owner_receipt"):
            source = report["sources"][kind]
            line = (f"  {labels[kind]}: {source['found']} found, "
                    f"+{source['added']} admitted")
            if source["note"]:
                line += f" — {source['note']}"
            lines.append(line)
        return "\n".join(lines)
    return "usage: /immune corpus | corpus harvest"


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
    # The reply follows the receipt's recorded effect — never the
    # success wording for an effect that was not performed.
    if receipt.details.get("effect") == "performed":
        return (f"disarmed: grant {grant_id.strip()} revoked "
                f"(receipt {receipt.id[:8]}).")
    return (f"disarm NOT performed: {receipt.reason} "
            f"(receipt {receipt.id[:8]}).")


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
        if state.fever.is_active():
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
        # Doc 20 P5: the fever window lands in the durable
        # observation journal, start and end alike.
        state.observations.append("fever_started", {
            "reason": reason, "max_seconds": seconds,
        })
        return f"🌡️ {message}"
    if verb == "end":
        if not state.fever.active:
            return "no fever active."
        fever_reason = state.fever.reason
        report = state.fever.end()
        state.observations.append("fever_ended", {
            "reason": fever_reason, "resolved": report.resolved,
            "actions_throttled": report.actions_throttled,
            "note": report.note,
        })
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
        day = time.strftime("%Y-%m-%d")
        metrics = {"tokens": tokens, "processes_spawned": spawned,
                   "grants_held": grants, "errors": errors}
        report = state.drift.add_day(day, metrics)
        # Doc 20 P5: the day lands in the durable observation
        # journal — the drift monitor itself stays per-chat
        # in-memory, but what it observed no longer dies with it.
        state.observations.append("drift_day", {
            "date": day, "metrics": metrics,
            "verdict": report.verdict,
            "signals": list(report.signals),
            "days_considered": report.days_considered,
        })
        return f"drift day recorded: verdict={report.verdict} signals={report.signals or 'none'}"
    if verb == "report" or not verb:
        report = state.drift.current_report()
        lines = [f"drift: {report.verdict} ({report.days_considered} days)"]
        lines += [f"  • {reason}" for reason in report.reasons]
        return "\n".join(lines)
    return IMMUNE_USAGE()


def _feeds(state: _ImmuneState, context: Any = None) -> str:
    """``/immune feeds`` — the sensor feeds' read surface.

    The failure ledger's real counts (or UNAVAILABLE with its
    reason — an unreadable ledger opens zero cases and says so),
    what this read opened or extended, and the observation
    journal's state. Reading this surface IS an evaluation point
    (doc 20 P5's lazy cadence), so the counts are this moment's.
    """
    report = _evaluate_feeds(state, context)
    lines = ["🛡️ immune sensor feeds"]
    if report is None:
        lines.append(
            "  failure ledger: UNAVAILABLE — feed evaluation "
            "failed; no feed cases opened from it")
    elif not report["available"]:
        lines.append(
            f"  failure ledger: UNAVAILABLE — {report['reason']}; "
            "no feed cases opened from it")
    else:
        lines.append(
            f"  failure ledger: {report['rows']} tool failure "
            f"row(s) in the last {report['window_s']:.0f}s · "
            f"{report['groups_over_threshold']} recurrence "
            f"group(s) at >={report['min_failures']} · "
            f"opened {len(report['opened'])} case(s), "
            f"extended {len(report['extended'])} on this read")
    feed_cases = sum(
        1 for case in state.casefiles
        if isinstance(case.event, dict)
        and case.event.get("kind") == "tool_failure_recurrence")
    lines.append(
        f"  feed cases: {feed_cases} case file(s) from the "
        "failure-ledger feed")
    observations = state.observations
    lines.append(
        f"  observations: {observations.count()} record(s) — "
        f"{observations.path or 'in-memory only (not persisted)'}")
    if observations.degraded_reason:
        lines.append(f"  ⚠ {observations.degraded_reason}")
    return "\n".join(lines)


def _mark_fp(state: _ImmuneState, rest: str) -> str:
    """``/immune mark-fp <antibody-id> [reason]`` — the owner's
    false-positive verdict (doc 20 P5).

    The FP signal path's seat: no shipped organ produces a match
    verdict automatically (the retire verdict was the only
    production caller of ``record_false_positive``), so the
    verdict is the owner's to give — and it lands on the
    antibody's own record through P3's machinery: an ACTIVE
    antibody demotes to probation, a probation antibody's stint
    restarts, and the verdict journals as an observation.
    """
    ab_id, _, reason = rest.partition(" ")
    ab_id = ab_id.strip()
    reason = reason.strip()
    if not ab_id:
        return "usage: /immune mark-fp <antibody-id> [reason]"
    entry = state.store.get(ab_id)
    if entry is None:
        # prefix match on the short id shown by list
        for candidate in state.store.all():
            if str(candidate.get("id", "")).startswith(ab_id):
                entry = candidate
                break
    if entry is None:
        return f"no antibody {ab_id!r}."
    verdict_reason = reason or (
        "owner false-positive verdict via /immune mark-fp")
    before = str(entry.get("status", ""))
    record = state.store.record_false_positive(
        str(entry.get("id", "")), reason=verdict_reason)
    if record is None:
        return f"no antibody {ab_id!r}."
    state.observations.append("false_positive", {
        "antibody_id": str(entry.get("id", "")),
        "reason": verdict_reason,
        "status_before": before,
        "status": str(record.get("status", "")),
    })
    after = str(record.get("status", ""))
    if before == "active" and after == "probation":
        outcome = ("demoted to probation — it must re-earn "
                   "promotion over a clean window")
    elif before == "probation" and after == "probation":
        outcome = ("probation stint restarted — the clean window "
                   "begins again")
    else:
        outcome = f"verdict recorded (standing: {after})"
    return (
        f"false positive recorded on antibody "
        f"{str(entry.get('id', ''))[:8]}: {outcome}.")


def _casefile(state: _ImmuneState, rest: str,
              context: Any = None) -> str:
    verb, _, tail = rest.partition(" ")
    verb = verb.lower()
    tail = tail.strip()
    if verb == "list" or not verb:
        _evaluate_feeds(state, context)
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
