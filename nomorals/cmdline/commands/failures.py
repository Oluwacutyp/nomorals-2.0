"""``nm failures`` — the failures-ledger triage surface.

``nm failures list`` reads the agents ledger through the learned
classifier when one is trained (``[real 0.87]`` / ``[noise 0.12]`` on
each line); with no classifier it reads exactly as before.
``nm failures grade <n|site> <real|noise>`` records an operator grade
durably in the grade journal. ``nm failures train`` re-fits the
classifier from the journal once the volume gate (30 graded rows) is
met — the promoted model is what the read path loads next.
``nm failures grades`` shows the count against the gate.
"""
from __future__ import annotations

import os
from typing import Any

from ..emit import _emit

_MAX_RING = 200

_USAGE = ("nm failures list [--json]\n"
          "nm failures grade <n|site> <real|noise> [--grader NAME]\n"
          "nm failures train [--json]\n"
          "nm failures grades [--json]")


def _words(args: Any) -> list[str]:
    return [str(w) for w in (getattr(args, "task", None) or [])]


def _grader(words: list[str]) -> str:
    for i, w in enumerate(words):
        if w == "--grader" and i + 1 < len(words):
            return words[i + 1].strip() or "operator"
    return "operator"


def _strip_flags(words: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for w in words:
        if skip:
            skip = False
            continue
        if w == "--grader":
            skip = True
            continue
        out.append(w)
    return out


def _gate_status() -> tuple[int, int]:
    from ...agents import failure_models as fm

    return len(fm.graded_rows()), fm.ERROR_MIN_ROWS


def _classifier_status() -> str:
    from ...agents.failure_models import load_error_classifier

    return "trained" if load_error_classifier() is not None else "none"


def _cmd_list(args: Any) -> int:
    from ...agents import failures as af

    rows = af.triaged_failures()
    payload = {
        "classifier": _classifier_status(),
        "total": af.failure_count(),
        "grades": _gate_status()[0],
        "failures": [{
            "ts": r.failure.ts, "site": r.failure.site,
            "exc_type": r.failure.exc_type, "message": r.failure.message,
            "p_real": round(r.p_real, 4), "verdict": r.verdict,
        } for r in rows],
    }
    _emit(args, payload, af.format_failures())
    return 0


def _resolve_target(target: str):
    """The ledger row a grade names: 1-based newest-first index, or a
    site substring that must match exactly one row in the ring."""
    from ...agents import failures as af

    ring = af.recent_failures(_MAX_RING)
    if target.isdigit():
        idx = int(target) - 1
        if 0 <= idx < len(ring):
            return ring[idx], None
        return None, (f"no failure #{target} — the ledger holds "
                      f"{len(ring)} (newest first)")
    matches = [f for f in ring if target in f.site]
    if not matches:
        return None, f"no failure matches site '{target}'"
    if len(matches) > 1:
        sites = ", ".join(sorted({f.site for f in matches}))
        return None, (f"{len(matches)} failures match '{target}' "
                      f"({sites}) — be more specific, or grade by number")
    return matches[0], None


def _cmd_grade(args: Any, words: list[str]) -> int:
    from ...agents import failure_models as fm

    rest = _strip_flags(words[1:])
    if len(rest) != 2:
        print("usage: nm failures grade <n|site> <real|noise> "
              "[--grader NAME]")
        return 1
    target, label = rest
    if label not in ("real", "noise"):
        print(f"grade: label must be 'real' or 'noise', not {label!r}")
        return 1
    row, err = _resolve_target(target)
    if err is not None:
        print(f"grade: {err}")
        return 1
    graded = fm.grade_failure(site=row.site, exc_type=row.exc_type,
                              message=row.message, real=(label == "real"),
                              grader=_grader(words))
    count, gate = _gate_status()
    if count >= gate:
        note = "gate met — run `nm failures train` to re-fit"
    else:
        note = f"{gate - count} more to train"
    text = (f"graded {graded['site']}: {graded['exc_type']} as {label} "
            f"({count}/{gate} grades — {note})")
    payload = {"graded": graded, "grades": count, "gate": gate,
               "gate_met": count >= gate}
    _emit(args, payload, text)
    return 0


def _cmd_train(args: Any) -> int:
    from ...agents import failure_models as fm

    outcome = fm.fit_error_classifier()
    count, gate = _gate_status()
    lines = [f"train: {outcome.status} — {outcome.reason}",
             f"grades: {count}/{gate}"]
    if outcome.report is not None:
        lines.append(f"held-out: {outcome.report.score} vs code rung "
                     f"{outcome.code_score}")
    text = "\n".join(lines)
    payload = {"status": outcome.status, "reason": outcome.reason,
               "grades": count, "gate": gate,
               "held_out": (outcome.report.score
                            if outcome.report is not None else None),
               "code_score": outcome.code_score}
    _emit(args, payload, text)
    return 0 if outcome.status == "trained" else 1


def _cmd_grades(args: Any) -> int:
    from ...agents import failure_models as fm

    count, gate = _gate_status()
    path = (os.environ.get("NM_FAILURE_GRADES") or "").strip() \
        or fm.DEFAULT_GRADES_PATH
    status = _classifier_status()
    if count >= gate:
        note = "gate met — run `nm failures train` to re-fit"
    else:
        note = f"{gate - count} more to train"
    text = (f"grades: {count}/{gate} ({note})\n"
            f"journal: {path}\n"
            f"classifier: {status}")
    payload = {"grades": count, "gate": gate, "gate_met": count >= gate,
               "journal": path, "classifier": status}
    _emit(args, payload, text)
    return 0


def _cmd_failures(args: Any, context: Any) -> int:
    words = _words(args)
    verb = words[0] if words else "list"
    if verb == "list":
        return _cmd_list(args)
    if verb == "grade":
        return _cmd_grade(args, words)
    if verb == "train":
        return _cmd_train(args)
    if verb == "grades":
        return _cmd_grades(args)
    print(_USAGE)
    return 1
