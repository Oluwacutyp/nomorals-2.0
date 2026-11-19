"""The 7-Question Gate — finding triage before any report is drafted.

Port of the ``triage`` half of ``cbh/cli.py`` from
elementalsouls/Claude-BugHunter (MIT, Copyright (c) 2026 Sachin
Sharma): ``TRIAGE_QUESTIONS`` verbatim, the PASS / DOWNGRADE /
KILL verdicts verbatim, Q7 inverted exactly as upstream.

Devon differences (the gate is better than the original):

- ``gate_finding`` applies the gate to ANY Devon finding dict —
  ``audit_web``, ``audit_repo`` or bughunter — not only bounty
  findings. The gate runs before any disclosure draft is produced.
- Q2 (impact on the program's accepted list) is wired to Devon's
  own severity rubric (``security/disclose.py`` ``score``) instead
  of keyword signals alone: a finding whose kind scores
  medium-or-better on Devon's deterministic rubric passes Q2 even
  when the finding text carries no severity keyword.
- The gate is pure: it returns answers + verdict. Journaling the
  outcome is the caller's job (the tool surface records it in the
  audit book).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TRIAGE_QUESTIONS",
    "gate_text",
    "gate_finding",
    "finding_gate_text",
]

#: The 7 questions, verbatim from upstream. Each entry is
#: (id, question, keyword signals looked for in the finding text).
#: Q7 is INVERTED: presence of a "never-submit" signal = NO (kill).
TRIAGE_QUESTIONS = [
    ("Q1", "Can an attacker use this RIGHT NOW with a real HTTP request?",
     ["curl ", "POST ", "GET ", "HTTP/1.1", "PUT ", "DELETE ", "PATCH "]),
    ("Q2", "Is the impact on the program's accepted-impact list?",
     ["impact:", "severity:", "p1", "p2", "p3", "p4", "critical",
      "high", "medium", "low"]),
    ("Q3", "Is the asset in scope?",
     ["scope", "in-scope", "in scope", "target:", "asset:"]),
    ("Q4", "Does it work without privileged access an attacker can't get?",
     ["attacker", "unauthenticated", "user-role", "low-priv",
      "any user", "session"]),
    ("Q5", "Is this not already known or documented behavior?",
     ["disclosed-reports", "h1 hacktivity", "not duplicate", "novel",
      "first reported", "previously unknown", "previously"]),
    ("Q6", "Can impact be proved beyond 'technically possible'?",
     ["leaked", "exfiltrated", "rce", "data:", "credential",
      "session-id", "cookie:", "admin email", "production",
      "oob callback", "interactsh"]),
    ("Q7", "Is this not on the never-submit list?",
     ["self-xss", "rate-limit only", "click-jacking",
      "csrf on logout", "missing security headers"]),
]


def gate_text(text: str) -> dict[str, Any]:
    """Run the 7-Question Gate over finding markdown/text.

    Returns ``{"answers": [...], "failed": [...], "verdict": str,
    "reason": str}``. Verdict: PASS (all 7 answered with evidence),
    DOWNGRADE (single Q2/Q5 fail — severity/duplication concerns,
    continue but tone down the severity claim), KILL (anything else).
    """
    lowered = (text or "").lower()
    answers = []
    for qid, question, signals in TRIAGE_QUESTIONS:
        # Q7 is INVERTED: presence of a "never-submit" signal means
        # the answer is "NO" (kill).
        hit = any(s.lower() in lowered for s in signals)
        if qid == "Q7":
            answer = ("NO — finding matches never-submit category"
                      if hit else "YES")
            ok = not hit
        else:
            answer = ("YES — evidence found" if hit
                      else "NO — no supporting evidence in finding")
            ok = hit
        answers.append({"id": qid, "question": question,
                        "answer": answer, "ok": ok})
    failed = [a["id"] for a in answers if not a["ok"]]
    if not failed:
        verdict, reason = "PASS", "all 7 questions answered with evidence"
    elif len(failed) == 1 and failed[0] in ("Q2", "Q5"):
        verdict = "DOWNGRADE"
        reason = (f"failed {failed[0]} (severity / duplication concerns); "
                  "continue but tone down the severity claim")
    else:
        verdict = "KILL"
        reason = (f"failed {','.join(failed)}; per triage-validation "
                  "discipline: do not draft the report")
    return {"answers": answers, "failed": failed, "verdict": verdict,
            "reason": reason}


def finding_gate_text(finding: dict[str, Any]) -> str:
    """Render a Devon finding dict as the gate's input text.

    Audit findings (web/repo) and bughunter candidates share no
    single schema, so every field that carries evidence is folded
    in: title, kind/check/class, location/URL, detail, severity,
    and any evidence request/response text. Nothing is invented —
    absent fields contribute nothing.
    """
    parts: list[str] = []
    for key in ("title", "kind", "check", "vuln_class", "severity",
                "location", "url", "target", "asset", "detail",
                "impact", "summary"):
        value = finding.get(key)
        if value:
            parts.append(f"{key}: {value}")
    evidence = finding.get("evidence")
    if isinstance(evidence, dict):
        for key in ("request", "response", "note", "proof_note"):
            value = evidence.get(key)
            if value:
                parts.append(f"evidence {key}: {value}")
    elif evidence:
        parts.append(f"evidence: {evidence}")
    # Devon audit findings name the asset ``location``, not
    # ``asset``/``target`` — Q3 ("is the asset in scope?") would
    # fail every one of them on a naming technicality. Carry the
    # location over as the asset line too, so the gate judges the
    # finding, not the field name. (The 7 questions themselves
    # stay verbatim upstream.)
    if not finding.get("asset") and not finding.get("target"):
        location = finding.get("location") or finding.get("url")
        if location:
            parts.append(f"asset: {location}")
    meta = finding.get("meta")
    if isinstance(meta, dict):
        for key, value in meta.items():
            if value:
                parts.append(f"meta {key}: {value}")
    return "\n".join(parts)


def _devon_q2_pass(finding: dict[str, Any]) -> bool:
    """Q2 via Devon's deterministic severity rubric.

    Maps the finding's kind/class onto ``disclose.score`` and
    passes when the rubric tier is medium or better. Unknown kinds
    score low by the rubric's own default and do not pass — the
    keyword signals in ``gate_text`` remain the other way through.
    """
    try:
        from ..disclose import Tier, score
    except Exception:  # noqa: BLE001 - no rubric, no Devon Q2 pass
        return False
    kind = str(finding.get("kind") or finding.get("vuln_class") or "")
    if not kind:
        return False
    meta = finding.get("meta") if isinstance(finding.get("meta"), dict) \
        else {}
    sensitivity = str(meta.get("sensitivity") or "none")
    reach = str(meta.get("reach") or "unauth")
    scale = str(meta.get("scale") or "single")
    try:
        result = score(kind, sensitivity=sensitivity, reach=reach,
                       scale=scale)
    except Exception:  # noqa: BLE001 - unscoreable kind, no pass
        return False
    tier = result.get("tier", "")
    return tier in (Tier.CRITICAL, Tier.HIGH, Tier.MEDIUM)


def gate_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """Run the 7-Question Gate over one Devon finding dict.

    Same verdicts as ``gate_text``; Q2 additionally passes when
    Devon's severity rubric scores the finding medium-or-better.
    The returned dict carries ``"devon_q2": True`` when the rubric
    (rather than keyword signals) carried Q2, so the caller can
    tell which evidence the verdict rests on.
    """
    result = gate_text(finding_gate_text(finding))
    devon_q2 = False
    if "Q2" in result["failed"] and _devon_q2_pass(finding):
        devon_q2 = True
        for answer in result["answers"]:
            if answer["id"] == "Q2":
                answer["ok"] = True
                answer["answer"] = ("YES — Devon severity rubric scores "
                                    "this medium-or-better")
        result["failed"] = [f for f in result["failed"] if f != "Q2"]
        if not result["failed"]:
            result["verdict"] = "PASS"
            result["reason"] = "all 7 questions answered with evidence"
        elif len(result["failed"]) == 1 \
                and result["failed"][0] in ("Q2", "Q5"):
            result["verdict"] = "DOWNGRADE"
            result["reason"] = (
                f"failed {result['failed'][0]} (severity / duplication "
                "concerns); continue but tone down the severity claim")
    result["devon_q2"] = devon_q2
    return result
