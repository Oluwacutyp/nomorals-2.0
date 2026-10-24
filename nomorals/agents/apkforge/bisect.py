"""ApkForge version bisect — the git-bisect contract over a
version series (design §3.4 / §4 A3).

Devon's git bisect organ (``agents/bisect.py``) hunts regressions
with a tri-state predicate: good / bad / skip (125 — "cannot judge
this one"), halving until the culprit is bracketed, never looping
forever.  The APK form keeps that exact contract; only the items
and the judge change:

* items are landed projects of one app, ordered by versionCode;
* the predicate is a *dossier-observable* property compiled from a
  spec — ``permission:``, ``class:``, ``tracker:``, ``sdk:``,
  ``exported:``, ``endpoint:``, ``string:``, ``cert:``, ``native:``;
* the judge answers ``found`` (the property is present — the
  "bad" side), ``absent`` (the "good" side), or ``skip`` when this
  version's dossier cannot judge it (unparseable, field missing),
  exactly like exit-125: a skip is never evidence either way.

Two A5 refinements of the judge:

* ``exported:`` is judged on a three-valued component state —
  ``absent`` / ``present-not-exported`` / ``present-exported``
  (:func:`component_state`, the same distinction the A1 diff
  draws between a component being added and its exported flag
  flipping).  A series can cross two state boundaries; the state
  driver (:func:`halve_states`) reports each crossing as its own
  named transition instead of conflating "the component appeared,
  unexported" with "nothing changed".
* A dossier whose dex census recorded parse errors is a *damaged
  specimen*: dissect keeps what survived (the manifest census is
  real), but a bisect never judges from damaged goods — every
  predicate skips it, with the parse error recorded as the skip's
  reason.  Partial evidence is not a verdict.

Halving assumes a single transition (a property appears once, or
disappears once) — stated on every verdict.  Sessions are journaled
per project set under ``<root>/bisects/<key>/`` (state.json +
append-only log.jsonl): every probe verdict is persisted as it is
made, so an interrupted bisect resumes from its recorded verdicts
instead of re-dissecting versions it already judged.
"""

from __future__ import annotations
from .failures import record_failure

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "BisectError",
    "FOUND", "ABSENT", "SKIP",
    "COMPONENT_ABSENT", "COMPONENT_NOT_EXPORTED", "COMPONENT_EXPORTED",
    "COMPONENT_STATES",
    "parse_predicate",
    "evaluate_predicate",
    "component_state",
    "dossier_damage",
    "halve",
    "halve_states",
    "session_key",
    "load_session",
    "record_verdict",
    "finish_session",
    "render_bisect_card",
]

FOUND = "found"
ABSENT = "absent"
SKIP = "skip"

#: The three-valued component state for ``exported:`` predicates,
#: ordered: a component appears before it can open up.
COMPONENT_ABSENT = "absent"
COMPONENT_NOT_EXPORTED = "present-not-exported"
COMPONENT_EXPORTED = "present-exported"
COMPONENT_STATES = (COMPONENT_ABSENT, COMPONENT_NOT_EXPORTED,
                    COMPONENT_EXPORTED)
_STATE_RANK = {name: i for i, name in enumerate(COMPONENT_STATES)}

#: Safety bound, mirroring the git organ's _MAX_STEPS discipline.
_MAX_PROBES = 128

_PREDICATE_KINDS = ("permission", "class", "tracker", "sdk",
                    "exported", "endpoint", "string", "cert",
                    "native")


class BisectError(RuntimeError):
    """A designed bisect refusal — names the exact cause."""


# ── predicate compiler ───────────────────────────────────────────────

def parse_predicate(spec: str) -> dict[str, Any]:
    """Compile ``"permission: android.permission.CAMERA"`` (or a
    bare value, which means ``class:``) into a predicate dict."""
    raw = (spec or "").strip()
    if not raw:
        raise BisectError("bisect needs a --check predicate, e.g. "
                          "'permission: android.permission.CAMERA'")
    kind, sep, value = raw.partition(":")
    if sep and kind.strip().lower() in _PREDICATE_KINDS:
        kind = kind.strip().lower()
        value = value.strip()
    else:
        kind, value = "class", raw
    if not value:
        raise BisectError(f"predicate {raw!r} has an empty value")
    return {"kind": kind, "value": value, "raw": raw}


def _dotted(name: str) -> str:
    d = name.strip()
    if d.startswith("L"):
        d = d[1:]
    if d.endswith(";"):
        d = d[:-1]
    return d.replace("/", ".")


def dossier_damage(dossier: dict[str, Any] | None) -> str:
    """The parse damage recorded in a dossier, or "" — dissect
    tolerates a dex it cannot parse (the census row is recorded in
    ``dex.errors`` and the rest of the dossier survives), but the
    specimen is damaged: which fields the damage touched is not
    knowable from the surviving half."""
    if not dossier:
        return ""
    errors = (dossier.get("dex") or {}).get("errors") or []
    return "; ".join(str(e) for e in errors)


def component_state(dossier: dict[str, Any] | None,
                    name: str) -> str | None:
    """One component's three-valued state in a dossier:
    COMPONENT_ABSENT / COMPONENT_NOT_EXPORTED /
    COMPONENT_EXPORTED — or None when the dossier carries no
    component census to judge from.

    ``exported`` may be None in a dossier row (the attribute is
    absent from the manifest): the platform default decides —
    a component with an intent filter defaults to exported when
    the app targets API 31+, and to not-exported otherwise; a
    component without a filter is never exported by default.
    That is the rule teach.py states and Android enforces."""
    if dossier is None:
        return None
    comps = dossier.get("components")
    if comps is None:
        return None
    row = None
    for rows in comps.values():
        for comp in rows or []:
            if comp.get("name") == name:
                row = comp
                break
        if row is not None:
            break
    if row is None:
        return COMPONENT_ABSENT
    exported = row.get("exported")
    if exported is True:
        return COMPONENT_EXPORTED
    if exported is False:
        return COMPONENT_NOT_EXPORTED
    target = (dossier.get("identity") or {}).get("target_sdk")
    modern = target is None or (isinstance(target, int)
                                and target >= 31)
    if row.get("intent_filters") and modern:
        return COMPONENT_EXPORTED
    return COMPONENT_NOT_EXPORTED


def evaluate_predicate(dossier: dict[str, Any] | None,
                       pred: dict[str, Any]) -> str:
    """Judge one dossier: FOUND / ABSENT / SKIP (cannot judge)."""
    if dossier is None:
        return SKIP
    if dossier_damage(dossier):
        # damaged specimen — never judged, on any predicate
        return SKIP
    kind, value = pred["kind"], pred["value"]
    dex = dossier.get("dex") or {}
    if kind == "permission":
        perms = dossier.get("permissions")
        if perms is None:
            return SKIP
        return FOUND if any(p.get("name") == value for p in perms) \
            else ABSENT
    if kind == "class":
        names = dex.get("class_names")
        hashes = dex.get("class_hashes")
        if names is None and hashes is None:
            return SKIP
        want = _dotted(value)
        pool = {_dotted(str(n)) for n in (names or [])}
        pool |= {_dotted(str(n)) for n in (hashes or {})}
        return FOUND if want in pool else ABSENT
    if kind == "tracker":
        hits = dossier.get("sdk_signatures")
        if hits is None:
            return SKIP
        low = value.lower()
        return FOUND if any(
            low in str(s.get("name", "")).lower()
            or str(s.get("package_prefix", "")) == value
            or str(s.get("package_prefix", "")).startswith(value + ".")
            for s in hits) else ABSENT
    if kind == "sdk":
        hits = dossier.get("sdk_signatures")
        if hits is None:
            return SKIP
        return FOUND if any(
            str(s.get("package_prefix", "")) == value
            or str(s.get("package_prefix", "")).startswith(value + ".")
            for s in hits) else ABSENT
    if kind == "exported":
        state = component_state(dossier, value)
        if state is None:
            return SKIP
        # the binary verdict is derived FROM the three-valued
        # state — present-but-not-exported is ABSENT here, but the
        # state itself is what halve_states bisects on, so the
        # distinction is never lost where it is reported.
        return FOUND if state == COMPONENT_EXPORTED else ABSENT
    if kind == "endpoint":
        urls = dex.get("urls")
        domains = dex.get("domains")
        if urls is None and domains is None:
            return SKIP
        pool = [str(u) for u in (urls or [])] \
            + [str(d) for d in (domains or [])]
        return FOUND if any(value in u for u in pool) else ABSENT
    if kind == "string":
        strings = dex.get("_all_strings")
        if strings is None:
            urls = dex.get("urls") or []
            domains = dex.get("domains") or []
            if not urls and not domains:
                return SKIP
            pool = [str(u) for u in urls] + [str(d) for d in domains]
            return FOUND if any(value in s for s in pool) else ABSENT
        return FOUND if any(value in str(s) for s in strings) \
            else ABSENT
    if kind == "cert":
        certs = (dossier.get("signing") or {}).get("certificates")
        if certs is None:
            return SKIP
        want = value.lower().replace(":", "")
        return FOUND if any(
            str(c.get("sha256", "")).lower().startswith(want)
            for c in certs) else ABSENT
    if kind == "native":
        libs = dossier.get("native")
        if libs is None:
            return SKIP
        return FOUND if any(
            value in str(n.get("path", ""))
            or value == str(n.get("name", "")) for n in libs) \
            else ABSENT
    return SKIP


# ── the halving driver (git-bisect contract) ─────────────────────────

def halve(items: list[str],
          judge: Callable[[str], str],
          known: dict[str, str] | None = None) -> dict[str, Any]:
    """Binary-search an ordered series for the single transition of
    a tri-state predicate.  ``judge`` is called at most once per
    item; verdicts in ``known`` are reused (session resume)."""
    verdicts: dict[str, str] = dict(known or {})
    probes = 0

    def verdict(item: str) -> str:
        nonlocal probes
        if item in verdicts:
            return verdicts[item]
        if probes >= _MAX_PROBES:
            raise BisectError(
                f"bisect did not converge in {_MAX_PROBES} probes")
        probes += 1
        v = judge(item)
        verdicts[item] = v
        return v

    n = len(items)
    if n < 2:
        raise BisectError("bisect needs at least two versions")

    # Boundary scan: first judgeable item from each end.
    left = right = None
    for i in range(n):
        if verdict(items[i]) != SKIP:
            left = i
            break
    for i in range(n - 1, -1, -1):
        if verdict(items[i]) != SKIP:
            right = i
            break
    skipped = [it for it in items if verdicts.get(it) == SKIP]

    def _base() -> dict[str, Any]:
        return {"verdicts": dict(verdicts), "probes": probes,
                "skipped": [it for it in items
                            if verdicts.get(it) == SKIP],
                "series": list(items)}

    if left is None or right is None or left >= right:
        return {**_base(), "status": "inconclusive",
                "transition": None,
                "note": "no two judgeable versions bracket a "
                        "transition — every version was skipped or "
                        "only one could be judged"}
    v_left, v_right = verdicts[items[left]], verdicts[items[right]]
    if v_left == v_right:
        status = "present_throughout" if v_left == FOUND \
            else "absent_throughout"
        note = (f"the property reads {v_left} at both ends of the "
                "series; under the single-transition assumption "
                "there is no flip inside it")
        if skipped:
            note += (f" — caveat: {len(skipped)} skipped version(s) "
                     "could not be judged and might hide one")
        return {**_base(), "status": status, "transition": None,
                "note": note}

    lo, hi = left, right
    while hi - lo > 1:
        # Fold already-known interior verdicts into the bracket
        # first — on a resumed session EVERYTHING may be known, and
        # known verdicts are evidence exactly like fresh probes.
        known_lo = max((i for i in range(lo + 1, hi)
                        if verdicts.get(items[i]) == v_left),
                       default=lo)
        known_hi = min((i for i in range(lo + 1, hi)
                        if verdicts.get(items[i]) == v_right),
                       default=hi)
        lo, hi = known_lo, known_hi
        if hi - lo <= 1:
            break
        unjudged = [i for i in range(lo + 1, hi)
                    if items[i] not in verdicts]
        if not unjudged:
            break  # everything between is a known skip
        mid = (lo + hi) // 2
        i = min(unjudged, key=lambda j: (abs(j - mid), j))
        v = verdict(items[i])
        if v == v_left:
            lo = i
        elif v == v_right:
            hi = i
        # SKIP: the index is consumed as judged; the bracket is
        # unchanged but the unjudged set shrank — progress either way.
    pair = (items[lo], items[hi])
    direction = "appears" if v_left == ABSENT else "disappears"
    exact = hi - lo == 1
    return {**_base(),
            "status": "flip" if exact else "flip_bracketed",
            "transition": {"older": pair[0], "newer": pair[1],
                           "older_verdict": v_left,
                           "newer_verdict": v_right},
            "direction": direction,
            "note": ("exact transition pair" if exact else
                     "transition bracketed between these versions; "
                     "skipped version(s) between them could not be "
                     "judged, so the pair cannot be narrowed further")}


def halve_states(items: list[str],
                 judge: Callable[[str], str],
                 known: dict[str, str] | None = None
                 ) -> dict[str, Any]:
    """Binary-search an ordered series for the transitions of a
    three-valued component state (the ``exported:`` driver).

    ``judge`` answers a COMPONENT_STATES value or SKIP per item.
    States are ordered (absent < present-not-exported <
    present-exported): a series may cross both boundaries, and
    every crossing is reported as its own named transition —
    absent→present-exported, absent→present-not-exported and
    present-not-exported→present-exported are different findings,
    and the driver never merges them into one "flip".  Same
    contract as :func:`halve` otherwise: each item is judged at
    most once, ``known`` verdicts resume a session, skips bracket
    a crossing they sit inside, and nothing is ever guessed."""
    states: dict[str, str] = {
        k: v for k, v in (known or {}).items()
        if v in COMPONENT_STATES or v == SKIP}
    probes = 0

    def verdict(item: str) -> str:
        nonlocal probes
        if item in states:
            return states[item]
        if probes >= _MAX_PROBES:
            raise BisectError(
                f"bisect did not converge in {_MAX_PROBES} probes")
        probes += 1
        v = judge(item)
        if v != SKIP and v not in COMPONENT_STATES:
            raise BisectError(
                f"state judge answered {v!r} for {item} — not a "
                "component state or skip")
        states[item] = v
        return v

    n = len(items)
    if n < 2:
        raise BisectError("bisect needs at least two versions")

    left = right = None
    for i in range(n):
        if verdict(items[i]) != SKIP:
            left = i
            break
    for i in range(n - 1, -1, -1):
        if verdict(items[i]) != SKIP:
            right = i
            break
    skipped = [it for it in items if states.get(it) == SKIP]

    def _base() -> dict[str, Any]:
        return {"verdicts": dict(states), "probes": probes,
                "skipped": [it for it in items
                            if states.get(it) == SKIP],
                "series": list(items)}

    if left is None or right is None or left >= right:
        return {**_base(), "status": "inconclusive",
                "transition": None, "transitions": [],
                "note": "no two judgeable versions bracket a "
                        "transition — every version was skipped or "
                        "only one could be judged"}
    v_left, v_right = states[items[left]], states[items[right]]
    if v_left == v_right:
        status = {COMPONENT_EXPORTED: "present_throughout",
                  COMPONENT_ABSENT: "absent_throughout",
                  COMPONENT_NOT_EXPORTED:
                      "not_exported_throughout"}[v_left]
        note = (f"the component reads {v_left} at both ends of "
                "the series; under the single-progression "
                "assumption its state does not change inside it")
        if skipped:
            note += (f" — caveat: {len(skipped)} skipped version(s) "
                     "could not be judged and might hide one")
        return {**_base(), "status": status, "transition": None,
                "transitions": [], "note": note}

    transitions: list[dict[str, Any]] = []
    all_exact = True
    lo = left
    while _STATE_RANK[states[items[lo]]] < _STATE_RANK[v_right]:
        threshold = _STATE_RANK[states[items[lo]]]
        window = items[lo:right + 1]

        def derived(it: str, _t: int = threshold) -> str:
            s = verdict(it)
            if s == SKIP:
                return SKIP
            return FOUND if _STATE_RANK[s] > _t else ABSENT

        sub = halve(window, derived)
        pair = sub["transition"]
        if pair is None:  # pragma: no cover — ends differ by design
            raise BisectError("state crossing vanished mid-search")
        older, newer = pair["older"], pair["newer"]
        exact = sub["status"] == "flip"
        all_exact = all_exact and exact
        transitions.append({
            "older": older, "newer": newer,
            "older_state": states[older],
            "newer_state": states[newer],
            "exact": exact})
        lo = items.index(newer)
    primary = transitions[-1]
    named = "; ".join(
        f"{t['older_state']} → {t['newer_state']}"
        for t in transitions)
    return {**_base(),
            "status": "flip" if all_exact else "flip_bracketed",
            "transition": {"older": primary["older"],
                           "newer": primary["newer"],
                           "older_verdict": primary["older_state"],
                           "newer_verdict": primary["newer_state"]},
            "transitions": transitions,
            "direction": "state-change",
            "note": (f"component state transitions: {named}"
                     if all_exact else
                     f"component state transitions: {named} — "
                     "at least one is bracketed, not exact: "
                     "skipped version(s) inside it could not be "
                     "judged, so the pair cannot be narrowed "
                     "further")}


# ── journaled sessions ───────────────────────────────────────────────

def session_key(series_ids: list[str], spec: str) -> str:
    blob = json.dumps({"series": series_ids, "spec": spec},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _session_dir(root: Path, key: str) -> Path:
    d = root / "bisects" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log(root: Path, key: str, op: str, **extra: Any) -> None:
    row = {"ts": time.time(), "op": op}
    row.update(extra)
    with open(_session_dir(root, key) / "log.jsonl", "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def load_session(root: Path, series_ids: list[str],
                 spec: str) -> dict[str, Any]:
    """Load (or create) the session state for one series+predicate.
    Existing state is how a bisect resumes: its recorded verdicts
    are fed back into ``halve`` as ``known``."""
    key = session_key(series_ids, spec)
    path = _session_dir(root, key) / "state.json"
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            state.setdefault("skip_details", {})
            state["resumed"] = True
            return state
        except (OSError, json.JSONDecodeError) as e:
            record_failure("bisect.load_session", e)
    state = {"key": key, "series": list(series_ids), "spec": spec,
             "verdicts": {}, "skip_details": {}, "result": None,
             "created_at": time.time(),
             "updated_at": time.time(), "resumed": False}
    _save_session(root, state)
    _log(root, key, "bisect_start", series=list(series_ids),
         spec=spec)
    return state


def _save_session(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    path = _session_dir(root, state["key"]) / "state.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)


def record_verdict(root: Path, state: dict[str, Any],
                   project_id: str, verdict: str,
                   detail: str = "") -> None:
    """Persist one probe verdict the moment it is made — an
    interruption after this line loses nothing.  ``detail`` is
    the skip's reason when the verdict is SKIP (a parse error, a
    failed dissection): journaled with the probe and kept in the
    session state, so a resumed run reports the same reasons."""
    state["verdicts"][project_id] = verdict
    if verdict == SKIP and detail:
        state.setdefault("skip_details", {})[project_id] = detail
    _save_session(root, state)
    _log(root, state["key"], "probe", project=project_id,
         verdict=verdict, **({"detail": detail} if detail else {}))


def finish_session(root: Path, state: dict[str, Any],
                   result: dict[str, Any]) -> None:
    state["result"] = {k: result.get(k) for k in
                       ("status", "transition", "transitions",
                        "direction", "note",
                        "probes", "skipped", "skip_reasons")}
    _save_session(root, state)
    _log(root, state["key"], "bisect_complete",
         status=result.get("status"),
         transition=result.get("transition"))


# ── rendering ────────────────────────────────────────────────────────

def render_bisect_card(res: dict[str, Any]) -> str:
    pred = res.get("predicate") or {}
    lines = [f"🪓 **bisect** — {pred.get('raw', '?')} "
             f"over {len(res.get('series') or [])} version(s)"]
    status = res.get("status")
    by_id = res.get("versions") or {}
    if status in ("flip", "flip_bracketed") and res.get("transitions"):
        lines.append("verdict: the component's state changed —")
        for t in res["transitions"]:
            older = by_id.get(t["older"]) or {}
            newer = by_id.get(t["newer"]) or {}
            lines.append(
                f"  {t['older_state']} → {t['newer_state']}: "
                f"v{older.get('version_name') or '?'} "
                f"(versionCode {older.get('version_code')})"
                f" → v{newer.get('version_name') or '?'} "
                f"(versionCode {newer.get('version_code')})"
                + ("" if t.get("exact") else
                   " (bracketed — skipped version(s) between the "
                   "pair could not be judged)"))
        if res.get("diff_card"):
            lines.append(res["diff_card"])
    elif status in ("flip", "flip_bracketed"):
        t = res["transition"]
        older = by_id.get(t["older"]) or {}
        newer = by_id.get(t["newer"]) or {}
        verb = "first appears in" if res.get("direction") == "appears" \
            else "disappears in"
        lines.append(
            f"verdict: the property {verb} "
            f"v{newer.get('version_name') or '?'} "
            f"(versionCode {newer.get('version_code')}) — "
            f"transition from v{older.get('version_name') or '?'} "
            f"(versionCode {older.get('version_code')})")
        if status == "flip_bracketed":
            lines.append("⚠ bracketed, not exact: skipped version(s) "
                         "between the pair could not be judged")
        if res.get("diff_card"):
            lines.append(res["diff_card"])
    elif status in ("present_throughout", "absent_throughout",
                    "not_exported_throughout"):
        lines.append(f"verdict: {res.get('note')}")
    else:
        lines.append(f"verdict: inconclusive — {res.get('note')}")
    if res.get("skipped"):
        lines.append(f"skipped (could not judge): "
                     f"{len(res['skipped'])}")
        reasons = res.get("skip_reasons") or {}
        for pid in res["skipped"]:
            if reasons.get(pid):
                v = by_id.get(pid) or {}
                lines.append(
                    f"  v{v.get('version_name') or pid} "
                    f"(versionCode {v.get('version_code')}): "
                    f"{str(reasons[pid])[:240]}")
    lines.append(f"probes this run: {res.get('probes', 0)}"
                 + (" · resumed from journaled state"
                    if res.get("resumed") else ""))
    return "\n".join(lines)
