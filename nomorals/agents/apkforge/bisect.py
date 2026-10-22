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

Halving assumes a single transition (a property appears once, or
disappears once) — stated on every verdict.  Sessions are journaled
per project set under ``<root>/bisects/<key>/`` (state.json +
append-only log.jsonl): every probe verdict is persisted as it is
made, so an interrupted bisect resumes from its recorded verdicts
instead of re-dissecting versions it already judged.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "BisectError",
    "FOUND", "ABSENT", "SKIP",
    "parse_predicate",
    "evaluate_predicate",
    "halve",
    "session_key",
    "load_session",
    "record_verdict",
    "finish_session",
    "render_bisect_card",
]

FOUND = "found"
ABSENT = "absent"
SKIP = "skip"

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


def evaluate_predicate(dossier: dict[str, Any] | None,
                       pred: dict[str, Any]) -> str:
    """Judge one dossier: FOUND / ABSENT / SKIP (cannot judge)."""
    if dossier is None:
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
        comps = dossier.get("components")
        if comps is None:
            return SKIP
        seen = False
        for rows in comps.values():
            for comp in rows or []:
                if comp.get("name") == value:
                    seen = True
                    if comp.get("exported"):
                        return FOUND
        return ABSENT if seen else ABSENT
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
            state["resumed"] = True
            return state
        except (OSError, json.JSONDecodeError):
            pass
    state = {"key": key, "series": list(series_ids), "spec": spec,
             "verdicts": {}, "result": None, "created_at": time.time(),
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
                   project_id: str, verdict: str) -> None:
    """Persist one probe verdict the moment it is made — an
    interruption after this line loses nothing."""
    state["verdicts"][project_id] = verdict
    _save_session(root, state)
    _log(root, state["key"], "probe", project=project_id,
         verdict=verdict)


def finish_session(root: Path, state: dict[str, Any],
                   result: dict[str, Any]) -> None:
    state["result"] = {k: result.get(k) for k in
                       ("status", "transition", "direction", "note",
                        "probes", "skipped")}
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
    if status in ("flip", "flip_bracketed"):
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
    elif status in ("present_throughout", "absent_throughout"):
        lines.append(f"verdict: {res.get('note')}")
    else:
        lines.append(f"verdict: inconclusive — {res.get('note')}")
    if res.get("skipped"):
        lines.append(f"skipped (could not judge): "
                     f"{len(res['skipped'])}")
    lines.append(f"probes this run: {res.get('probes', 0)}"
                 + (" · resumed from journaled state"
                    if res.get("resumed") else ""))
    return "\n".join(lines)
