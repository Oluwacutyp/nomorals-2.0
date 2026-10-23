"""ApkForge tracker/SDK signature growth — the open-sources law
applied to A1's seed table (design §4 A3).

A1 shipped a hand-seeded prefix table (``dissect.SDK_SIGNATURES``,
~28 entries) and every dossier labeled it a seed: a hit is evidence,
no hit is not proof of absence.  A static seed under-reports forever,
so the table now *grows* from the class's own evidence:

* **Discovery** — every dissection is observed.  Package prefixes
  that are not the app's own tree, not framework roots, and not
  covered by the active table are counted as *candidates*, once per
  dossier, with the dossiers that evidenced them.  Discovery reads
  BOTH kinds of census evidence (A5): the classes a dossier
  *defines* and the types it merely *references* — the common case
  is an app calling an SDK whose classes live in another dex or a
  shared runtime, so the reference table is where its signature
  actually shows up.  Occurrence counts are kept split
  (defined vs referenced) on every candidate, proposal and
  promotion, so the evidence never loses which kind it was.
* **Proposal** — a candidate seen in >= the recurrence threshold
  distinct dossiers becomes a proposal: a standing, evidence-backed
  suggestion, nothing more.  Proposals never match anything.  The
  threshold is config (``NOMORALS_APKFORGE_SIG_THRESHOLD``, read at
  discovery time; ``RECURRENCE_THRESHOLD`` is its default), not a
  buried constant.
* **Promotion** — an explicit, journaled act (surface op) moves a
  proposal into the active table with its evidence attached.  Only
  promoted entries join dissections, marked ``origin="grown"`` next
  to the seeds' ``origin="seed"``.
* **Demotion** — a grown entry can be demoted the same way; the
  event is journaled with its reason and the entry leaves the
  active table.  Seeds are code, not state — they cannot be
  demoted through this store, and the refusal says so.

No silent mutation anywhere: the state file only changes through
``observe_dossier`` (candidate census — evidence, not table),
``promote`` and ``demote``, and every promotion/demotion appends a
row to ``signatures.jsonl`` in the class root.  A grown entry that
stops matching observed dossiers is *flagged* stale in status —
flagging is a report, not a mutation; demotion stays a decision.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .dissect import SDK_SIGNATURES, _pkg_of

__all__ = [
    "SignatureError",
    "RECURRENCE_THRESHOLD",
    "recurrence_threshold",
    "active_signatures",
    "load_state",
    "observe_dossier",
    "proposals",
    "promote",
    "demote",
    "signatures_status",
    "render_signatures",
]

STATE_NAME = "signatures_state.json"
EVENTS_NAME = "signatures.jsonl"

#: Default recurrence threshold: a prefix must appear in this many
#: distinct dossiers before it is proposed for the active table.
#: The live value is config — see recurrence_threshold().
RECURRENCE_THRESHOLD = 3

#: Minimum classes a prefix must account for in one dossier to count
#: as an occurrence there (one stray class is noise, not an SDK).
_MIN_CLASSES_PER_DOSSIER = 2

#: Minimum referenced types a prefix must account for in one dossier
#: to count as a referenced occurrence there — the same noise
#: discipline as defined classes, applied to the reference table.
_MIN_REFS_PER_DOSSIER = 2


def recurrence_threshold() -> int:
    """The live recurrence threshold, read from config at call
    time: ``NOMORALS_APKFORGE_SIG_THRESHOLD`` (the class's
    NOMORALS_APKFORGE_* env config surface, intake.py's pattern)
    when it parses to an int >= 1, else RECURRENCE_THRESHOLD."""
    raw = os.environ.get("NOMORALS_APKFORGE_SIG_THRESHOLD",
                         "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 1:
            return value
    return RECURRENCE_THRESHOLD

#: Package roots that are the platform/framework, never an SDK.
_FRAMEWORK_ROOTS = (
    "android.", "androidx.", "java.", "javax.", "kotlin.",
    "kotlinx.", "dalvik.", "com.android.", "org.w3c.", "org.xml.",
    "org.json.", "sun.", "jdk.",
)


class SignatureError(RuntimeError):
    """A designed signature-store refusal — names the exact cause."""


# ── state IO ─────────────────────────────────────────────────────────

def _state_path(root: Path) -> Path:
    return root / STATE_NAME


def _events_path(root: Path) -> Path:
    return root / EVENTS_NAME


def load_state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    if not path.exists():
        return {"grown": [], "demoted": [], "candidates": {},
                "observed_projects": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"grown": [], "demoted": [], "candidates": {},
                "observed_projects": []}
    state.setdefault("grown", [])
    state.setdefault("demoted", [])
    state.setdefault("candidates", {})
    state.setdefault("observed_projects", [])
    return state


def _save_state(root: Path, state: dict[str, Any]) -> None:
    path = _state_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)


def _event(root: Path, op: str, **extra: Any) -> None:
    row = {"ts": time.time(), "op": op}
    row.update(extra)
    with open(_events_path(root), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def events(root: Path, limit: int = 100) -> list[dict[str, Any]]:
    path = _events_path(root)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


# ── the active table ─────────────────────────────────────────────────

def seed_entries() -> list[dict[str, Any]]:
    return [{"prefix": p, "label": label, "kind": kind,
             "origin": "seed"}
            for p, label, kind in SDK_SIGNATURES]


def active_entries(root: Path) -> list[dict[str, Any]]:
    """Seeds + promoted grown entries — the table dissections use."""
    state = load_state(root)
    out = seed_entries()
    out.extend(dict(e) for e in state["grown"])
    return out


def active_signatures(root: Path) -> tuple[tuple[str, str, str], ...]:
    """(prefix, label, kind) triples for ``dissect_apk``'s
    ``extra_signatures`` — grown entries only (seeds are already
    compiled into dissect)."""
    state = load_state(root)
    return tuple((str(e["prefix"]), str(e["label"]), str(e["kind"]))
                 for e in state["grown"])


def _covered(prefix: str, table: list[dict[str, Any]]) -> bool:
    """True when the active table already accounts for ``prefix``:
    it equals an entry, sits under one, or is an ancestor of one
    (an ancestor would swallow a known SDK whole)."""
    for entry in table:
        known = str(entry["prefix"])
        if prefix == known or prefix.startswith(known + ".") \
                or known.startswith(prefix + "."):
            return True
    return False


# ── discovery ────────────────────────────────────────────────────────

def _prefix_counts(per_package: dict[str, int],
                   dossier: dict[str, Any]) -> dict[str, int]:
    """Roll a package → count map up to candidate prefixes
    (2- and 3-segment prefixes + the full package), dropping the
    app's own tree and the framework roots."""
    own = str((dossier.get("identity") or {}).get("package") or "")
    counts: dict[str, int] = {}
    for pkg, n in per_package.items():
        if own and (pkg == own or pkg.startswith(own + ".")):
            continue
        if any(pkg == r[:-1] or pkg.startswith(r)
               for r in _FRAMEWORK_ROOTS):
            continue
        segs = pkg.split(".")
        prefixes = {".".join(segs[:d]) for d in (2, 3) if len(segs) >= d}
        if 2 <= len(segs) <= 5:
            prefixes.add(pkg)
        for prefix in prefixes:
            if len(prefix.split(".")) < 2:
                continue
            counts[prefix] = counts.get(prefix, 0) + n
    return counts


def _candidate_prefixes(dossier: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Unknown package prefixes in one dossier → split counts
    ``{"defined": n, "referenced": m}``.

    Defined counts come from the dex census of classes the app
    *defines*; referenced counts from the types it *references*
    without defining (dissect's referenced census — method/field
    tables and superclasses).  A prefix is a candidate when either
    source clears its per-dossier noise floor; the split is kept
    so a proposal's evidence says which kind carried it."""
    defined = _prefix_counts(_dossier_packages(dossier), dossier)
    referenced = _prefix_counts(
        _dossier_referenced_packages(dossier), dossier)
    out: dict[str, dict[str, int]] = {}
    for prefix in sorted(set(defined) | set(referenced)):
        d_n = defined.get(prefix, 0)
        r_n = referenced.get(prefix, 0)
        if d_n >= _MIN_CLASSES_PER_DOSSIER or \
                r_n >= _MIN_REFS_PER_DOSSIER:
            out[prefix] = {"defined": d_n, "referenced": r_n}
    return out


def _dossier_packages(dossier: dict[str, Any]) -> dict[str, int]:
    """package → defined-class count, from the dex census."""
    dex = dossier.get("dex") or {}
    per_package: dict[str, int] = {}
    for name in dex.get("class_names") or []:
        pkg = _pkg_of(str(name))
        if pkg:
            per_package[pkg] = per_package.get(pkg, 0) + 1
    if not per_package:
        for row in dex.get("package_rollup") or []:
            pkg = str(row.get("package") or "")
            if pkg:
                per_package[pkg] = int(row.get("classes") or 0)
    return per_package


def _dossier_referenced_packages(
        dossier: dict[str, Any]) -> dict[str, int]:
    """package → referenced-type count, from the dex census's
    referenced set (A5 dossiers carry ``referenced_packages``
    directly; the rollup is derivable from ``referenced_types``
    for dossiers that carry the list form)."""
    dex = dossier.get("dex") or {}
    rollup = dex.get("referenced_packages")
    if isinstance(rollup, dict) and rollup:
        return {str(pkg): int(n) for pkg, n in rollup.items()
                if pkg and int(n or 0) > 0}
    per_package: dict[str, int] = {}
    for name in dex.get("referenced_types") or []:
        text = str(name)
        if "/" in text or text.startswith("L"):
            pkg = _pkg_of(text)
        else:
            pkg = text.rsplit(".", 1)[0] if "." in text else ""
        if pkg:
            per_package[pkg] = per_package.get(pkg, 0) + 1
    return per_package


def observe_dossier(root: Path, project_id: str,
                    dossier: dict[str, Any], *,
                    threshold: int | None = None
                    ) -> dict[str, Any]:
    """Count one dissection's unknown prefixes into the candidate
    census, and record sightings of already-grown entries (their
    health evidence).  ``threshold`` defaults to the configured
    recurrence threshold, read at discovery time.  Returns what
    changed: new candidates and prefixes that crossed the proposal
    threshold with this dossier."""
    if threshold is None:
        threshold = recurrence_threshold()
    state = load_state(root)
    per_package = _dossier_packages(dossier)
    seen_packages = set(per_package) | set(
        _dossier_referenced_packages(dossier))
    if project_id not in state["observed_projects"]:
        state["observed_projects"].append(project_id)
    # grown-entry sightings: health-check evidence for the entries
    # already in the table — a sighting in either census counts
    # (an SDK later apps only reference is still alive).
    for entry in state["grown"]:
        prefix = str(entry["prefix"])
        if any(pkg == prefix or pkg.startswith(prefix + ".")
               for pkg in seen_packages):
            sightings = entry.setdefault("sightings", [])
            if project_id not in sightings:
                sightings.append(project_id)
    table = seed_entries() + [dict(e) for e in state["grown"]]
    demoted = {str(e["prefix"]) for e in state["demoted"]}
    counts = _candidate_prefixes(dossier)
    new_candidates: list[str] = []
    newly_proposed: list[str] = []
    now = time.time()
    for prefix, split in sorted(counts.items()):
        if _covered(prefix, table) or prefix in demoted:
            continue
        cand = state["candidates"].get(prefix)
        if cand is None:
            cand = {"dossiers": [], "classes_total": 0,
                    "defined_classes_total": 0,
                    "referenced_types_total": 0,
                    "first_seen": now, "last_seen": now,
                    "proposed": False}
            state["candidates"][prefix] = cand
            new_candidates.append(prefix)
        if project_id not in cand["dossiers"]:
            # legacy candidates (pre-A5 state) carry only the
            # combined total, which was defined-only evidence —
            # seed the split from it so totals never regress.
            cand.setdefault(
                "defined_classes_total",
                int(cand.get("classes_total") or 0))
            cand.setdefault("referenced_types_total", 0)
            cand["dossiers"].append(project_id)
            cand["defined_classes_total"] = int(
                cand.get("defined_classes_total") or 0) \
                + split["defined"]
            cand["referenced_types_total"] = int(
                cand.get("referenced_types_total") or 0) \
                + split["referenced"]
            cand["classes_total"] = int(
                cand["defined_classes_total"]
                + cand["referenced_types_total"])
            cand["last_seen"] = now
            if not cand["proposed"] \
                    and len(cand["dossiers"]) >= threshold:
                cand["proposed"] = True
                cand["proposed_at"] = now
                newly_proposed.append(prefix)
                _event(root, "signature_proposed", prefix=prefix,
                       dossiers=list(cand["dossiers"]),
                       classes_total=cand["classes_total"],
                       defined_classes_total=cand[
                           "defined_classes_total"],
                       referenced_types_total=cand[
                           "referenced_types_total"],
                       threshold=threshold)
    _save_state(root, state)
    return {"observed_project": project_id,
            "new_candidates": new_candidates,
            "newly_proposed": newly_proposed,
            "candidates_total": len(state["candidates"])}


def proposals(root: Path, *,
              threshold: int | None = None
              ) -> list[dict[str, Any]]:
    if threshold is None:
        threshold = recurrence_threshold()
    state = load_state(root)
    out = []
    for prefix, cand in sorted(state["candidates"].items()):
        if len(cand.get("dossiers") or []) >= threshold:
            out.append({"prefix": prefix,
                        "dossiers": list(cand["dossiers"]),
                        "occurrences": len(cand["dossiers"]),
                        "classes_total": cand["classes_total"],
                        "defined_classes_total": int(
                            cand.get("defined_classes_total") or 0),
                        "referenced_types_total": int(
                            cand.get("referenced_types_total") or 0),
                        "first_seen": cand["first_seen"],
                        "last_seen": cand["last_seen"]})
    out.sort(key=lambda r: (-r["occurrences"], r["prefix"]))
    return out


# ── promotion / demotion (journaled, reversible) ─────────────────────

def promote(root: Path, prefix: str, *, label: str = "",
            kind: str = "") -> dict[str, Any]:
    """Move a proposed candidate into the active table.  The event
    is journaled with the candidate's full evidence; the candidate
    record stays in state as the entry's provenance."""
    prefix = (prefix or "").strip()
    if not prefix:
        raise SignatureError("promote needs a package prefix")
    state = load_state(root)
    if any(e["prefix"] == prefix for e in state["grown"]):
        raise SignatureError(f"{prefix} is already in the active "
                             "table (grown) — nothing to promote")
    if any(p == prefix for p, _l, _k in SDK_SIGNATURES):
        raise SignatureError(f"{prefix} is a seed signature — "
                             "already active, and seeds are not "
                             "managed through the store")
    cand = state["candidates"].get(prefix)
    if cand is None:
        raise SignatureError(
            f"{prefix} has never been observed as a candidate — "
            "promotion needs discovery evidence (dissect apps "
            "containing it first); refusing to add an unevidenced "
            "signature")
    threshold = recurrence_threshold()
    if len(cand.get("dossiers") or []) < threshold:
        raise SignatureError(
            f"{prefix} has been seen in "
            f"{len(cand.get('dossiers') or [])} dossier(s), below "
            f"the recurrence threshold {threshold} — "
            "not yet a proposal")
    entry = {
        "prefix": prefix,
        "label": label.strip() or prefix,
        "kind": kind.strip() or "unknown",
        "origin": "grown",
        "promoted_at": time.time(),
        "observed_count_at_promotion":
            len(state["observed_projects"]),
        "sightings": [],
        "provenance": {
            "source": "apkforge-discovery",
            "dossiers": list(cand["dossiers"]),
            "occurrences": len(cand["dossiers"]),
            "classes_total": cand["classes_total"],
            "defined_classes_total": int(
                cand.get("defined_classes_total") or 0),
            "referenced_types_total": int(
                cand.get("referenced_types_total") or 0),
            "first_seen": cand["first_seen"],
            "last_seen": cand["last_seen"],
        },
    }
    state["grown"].append(entry)
    _save_state(root, state)
    _event(root, "signature_promoted", prefix=prefix,
           label=entry["label"], kind=entry["kind"],
           evidence=entry["provenance"])
    return dict(entry)


def demote(root: Path, prefix: str, *,
           reason: str = "") -> dict[str, Any]:
    """Remove a grown entry from the active table (journaled,
    reversible by re-promotion once it re-earns proposal standing —
    its candidate census is kept)."""
    prefix = (prefix or "").strip()
    if any(p == prefix for p, _l, _k in SDK_SIGNATURES):
        raise SignatureError(
            f"{prefix} is a seed signature compiled into the class "
            "— seeds cannot be demoted through the store")
    state = load_state(root)
    hit = next((e for e in state["grown"] if e["prefix"] == prefix),
               None)
    if hit is None:
        raise SignatureError(f"{prefix} is not a grown active "
                             "signature — nothing to demote")
    state["grown"] = [e for e in state["grown"]
                      if e["prefix"] != prefix]
    gone = dict(hit)
    gone["demoted_at"] = time.time()
    gone["demote_reason"] = reason
    state["demoted"].append(gone)
    _save_state(root, state)
    _event(root, "signature_demoted", prefix=prefix, reason=reason,
           was=hit)
    return gone


# ── status / health ──────────────────────────────────────────────────

def signatures_status(root: Path, *,
                      threshold: int | None = None
                      ) -> dict[str, Any]:
    if threshold is None:
        threshold = recurrence_threshold()
    state = load_state(root)
    observed = state["observed_projects"]
    grown = []
    for entry in state["grown"]:
        row = dict(entry)
        since = observed[int(entry.get(
            "observed_count_at_promotion") or 0):]
        sightings = set(entry.get("sightings") or [])
        # staleness: enough dossiers observed since promotion and
        # none of them contained this prefix — a flag for a human
        # decision, never an automatic demotion.
        row["stale"] = len(since) >= threshold and not \
            (sightings & set(since))
        row["sightings_count"] = len(sightings)
        grown.append(row)
    return {
        "seed_count": len(SDK_SIGNATURES),
        "grown": grown,
        "grown_count": len(state["grown"]),
        "demoted": state["demoted"],
        "candidates": [
            {"prefix": p, "occurrences": len(c["dossiers"]),
             "classes_total": c["classes_total"],
             "defined_classes_total": int(
                 c.get("defined_classes_total") or 0),
             "referenced_types_total": int(
                 c.get("referenced_types_total") or 0),
             "proposed": bool(c.get("proposed"))}
            for p, c in sorted(state["candidates"].items())],
        "proposals": proposals(root, threshold=threshold),
        "threshold": threshold,
        "dossiers_observed": len(observed),
        "events": len(events(root, limit=100000)),
    }


def render_signatures(st: dict[str, Any]) -> str:
    lines = ["🪓 **ApkForge tracker/SDK signatures**",
             f"active table: {st.get('seed_count', 0)} seed + "
             f"{st.get('grown_count', 0)} grown "
             f"(from {st.get('dossiers_observed', 0)} observed "
             "dossier(s))"]
    for e in st.get("grown") or []:
        lines.append(f"  grown: {e['prefix']} — {e['label']} "
                     f"({e['kind']})"
                     + (" · STALE (no recent dossier matches)"
                        if e.get("stale") else ""))
    props = st.get("proposals") or []
    if props:
        lines.append("proposals (recurrent, awaiting promotion):")
        for p in props:
            lines.append(f"  {p['prefix']} — seen in "
                         f"{p['occurrences']} dossiers, "
                         f"{p['classes_total']} types "
                         f"({p.get('defined_classes_total', 0)} "
                         f"defined, "
                         f"{p.get('referenced_types_total', 0)} "
                         "referenced)")
    cands = [c for c in st.get("candidates") or []
             if not c.get("proposed")]
    if cands:
        lines.append("candidates (still gathering evidence):")
        for c in cands[:10]:
            lines.append(f"  {c['prefix']} — "
                         f"{c['occurrences']} dossier(s)")
    if st.get("demoted"):
        lines.append("demoted: "
                     + ", ".join(str(d["prefix"])
                                 for d in st["demoted"]))
    lines.append("promotion/demotion are journaled events in "
                 "signatures.jsonl — the table never mutates "
                 "silently")
    return "\n".join(lines)
