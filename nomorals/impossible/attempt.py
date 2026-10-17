"""Stage 9 — ATTEMPT. Standing queue + attempt execution + specification output.

- Software: the class builds it itself — attempt produces a build brief and,
  when a builders entry exists, hands it off; the attempt record carries the
  brief, the verification step, and the outcome. No fake 'built' claims.
- Hardware: the class's deliverable is a SPECIFICATION (drawings, BOM,
  tolerances, sourcing, test plan, failure modes, cost curve, experiment),
  not a device. The spec is written to disk as a real file. Delegation
  (human hands) is the missing half: attempt records the delegation brief;
  there is no delegation/ organ in-repo yet, so the brief is stored and
  reported honestly as 'awaiting delegation' instead of pretending hands exist.
- Dormancy: a problem worked N cycles with no delta goes COLD, not dead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_QUEUE = Path("data/impossible_queue.json")
DEFAULT_SPECS = Path("data/impossible_specs")


class AttemptQueue:
    def __init__(self, path: str | Path | None = None, spec_dir: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_QUEUE
        self.spec_dir = Path(spec_dir) if spec_dir else DEFAULT_SPECS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"items": []}, indent=2), encoding="utf-8")

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {"items": []}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def enqueue(self, claim_id: str, problem: str, dossier: dict | None = None) -> dict:
        item = {"claim_id": claim_id, "problem": problem,
                "dossier_ref": (dossier or {}).get("claim_id", claim_id),
                "status": "open", "cycles_no_delta": 0, "attempts": 0,
                "created_at": time.time(), "updated_at": time.time()}
        data = self._load()
        data["items"].append(item)
        self._save(data)
        return item

    def queue(self, status: str | None = None) -> list[dict]:
        items = self._load().get("items", [])
        return [i for i in items if status is None or i["status"] == status]

    def get(self, claim_id: str) -> dict | None:
        for i in self._load().get("items", []):
            if i["claim_id"] == claim_id or i["claim_id"].startswith(claim_id):
                return i
        return None

    def mark_cold(self, claim_id: str, cycles: int = 3) -> dict | None:
        """Dormancy rule: N cycles with no delta -> COLD, not dead."""
        data = self._load()
        for i in data.get("items", []):
            if i["claim_id"] == claim_id:
                i["cycles_no_delta"] = cycles
                i["status"] = "cold"
                i["updated_at"] = time.time()
                self._save(data)
                return i
        return None

    def attempt(self, claim_id: str, dossier: dict, ledger=None) -> dict:
        """Run the next cheapest attempt for a queued problem.
        Tries the best-ranked untried hypothesis first; a failed attempt
        moves to the next hypothesis in rank order, not back to guessing.
        Returns the attempt record + hardware spec file (if hardware) or
        software build brief. Never claims a device was built."""
        item = self.get(claim_id)
        if item is None:
            return {"ok": False, "error": f"No queued problem with id '{claim_id}'. Submit it first."}
        hyp = dossier.get("hypothesis") or {}
        # Pick the best-ranked hypothesis not yet tried for this claim.
        ranked = list(hyp.get("hypotheses") or [])
        tried = list(item.get("tried_hypotheses") or [])
        attempted_hyp = None
        for candidate in ranked:
            if candidate.get("angle") not in tried:
                attempted_hyp = candidate
                break
        if attempted_hyp is None:
            # All hypotheses tried: fall back to the primary (best-ranked).
            attempted_hyp = ranked[0] if ranked else hyp
        remaining = [c.get("angle") for c in ranked
                     if c.get("angle") not in tried
                     and c.get("angle") != attempted_hyp.get("angle")]
        gate = dossier.get("gate") or {}
        is_hardware = dossier.get("domain", "hardware") != "software"
        attempt_no = item["attempts"] + 1
        lesson_hint = ""
        if ledger is not None:
            entry = ledger.get(claim_id)
            if entry:
                from .human import learn_from_failure
                lesson_hint = learn_from_failure(entry)
        if is_hardware:
            spec = build_hardware_spec(claim_id, dossier, hypothesis=attempted_hyp)
            spec_path = self.spec_dir / f"{claim_id}_spec.md"
            spec_path.write_text(spec, encoding="utf-8")
            result = {"ok": True, "attempt": attempt_no, "kind": "hardware",
                      "action": "specification written — hardware deliverable is a spec, not a device",
                      "spec_file": str(spec_path),
                      "delegation": ("Delegation brief stored in the spec. No delegation/ organ exists in-repo yet — "
                                     "status is 'awaiting delegation', not 'built'."),
                      "hypothesis": attempted_hyp.get("angle", ""),
                      "next": attempted_hyp.get("cheapest_killing_experiment", ""),
                      "remaining_hypotheses": remaining,
                      "lesson_hint": lesson_hint}
        else:
            result = {"ok": True, "attempt": attempt_no, "kind": "software",
                      "action": "build brief produced — software attempt goes through builders/ with verify",
                      "build_brief": (
                          f"Build: {dossier.get('formulation', {}).get('goal_function', '')}. "
                          f"Falsifier first: {attempted_hyp.get('falsifier', '')}. "
                          f"Verify with builders/verify + smoke; promote only on passing evidence."),
                      "hypothesis": attempted_hyp.get("angle", ""),
                      "next": attempted_hyp.get("cheapest_killing_experiment", ""),
                      "remaining_hypotheses": remaining,
                      "lesson_hint": lesson_hint}
        data = self._load()
        for i in data.get("items", []):
            if i["claim_id"] == claim_id:
                i["attempts"] = attempt_no
                i["status"] = "attempted"
                i["updated_at"] = time.time()
                tried_now = list(i.get("tried_hypotheses") or [])
                angle = attempted_hyp.get("angle", "")
                if angle and angle not in tried_now:
                    tried_now.append(angle)
                i["tried_hypotheses"] = tried_now
        self._save(data)
        if ledger is not None:
            ledger.record_attempt(claim_id, outcome="attempted",
                                  evidence=result.get("spec_file", result.get("build_brief", "")),
                                  lesson=f"Attempt {attempt_no} (hypothesis: {attempted_hyp.get('angle', '')}): "
                                         f"{result['action']}. Next: {result['next']}. "
                                         f"Remaining hypotheses: {', '.join(remaining) or 'none'}.")
        return result


def build_hardware_spec(claim_id: str, dossier: dict, hypothesis: dict | None = None) -> str:
    """Hardware specification document — the class's hardware deliverable."""
    f = dossier.get("formulation") or {}
    g = dossier.get("gate") or {}
    hyp = hypothesis or dossier.get("hypothesis") or {}
    reframed = dossier.get("reframed") or {}
    transfers = dossier.get("transfer") or {}
    fp = dossier.get("first_principles") or {}
    lines = [
        f"# Hardware Specification — {claim_id}",
        "",
        f"**Goal (function):** {f.get('goal_function', '')}",
        f"**Raw wish:** {f.get('raw', '')}",
        f"**Gate verdict:** {g.get('verdict', '')} — {g.get('why', '')}",
        f"**Hypothesis attempted:** {hyp.get('angle', '')} (rank {hyp.get('rank', '')})",
        "",
        "## First principles",
        fp.get("rebuilt_solution", ""),
        "",
        "## Transfer moves used",
    ]
    for m in (transfers.get("triz") or [])[:4]:
        lines.append(f"- TRIZ #{m.get('principle')}: {m.get('name')} — {m.get('how', '')}")
    for m in (transfers.get("inversion") or []):
        lines.append(f"- {m.get('move')}: {m.get('apply', '')}")
    lines += [
        "",
        "## Reframed goal",
        reframed.get("reframed_goal", ""),
        "",
        "## Drawings / BOM / tolerances",
        "- Drawings: to be produced from this spec (dimensions from the gate's material/scale checks).",
        "- BOM: list every part with material (see gate materials check), quantity, supplier, cost.",
        "- Tolerances: from the manufacturability check — state per-part tolerance, not a global number.",
        "- Sourcing: prefer local suppliers; note lead times and alternates.",
        "",
        "## Test plan",
        f"- Killing experiment first: {hyp.get('cheapest_killing_experiment', '')}",
        f"- Falsifier: {hyp.get('falsifier', '')}",
        "- Then: measure the goal function once, honestly, and record the result in the honesty ledger.",
        "",
        "## Failure modes",
        "- Weakest link first: seal, joint, heat path, wear part, tired human.",
        "- Maintenance: MTBF, parts availability, skill required (see gate maintenance check).",
        "",
        "## Cost curve",
        f"- Gate cost check: {next((c.get('reason', '') for c in g.get('checks', []) if c.get('check') == 'cost'), 'not supplied')}",
        "",
        "## Delegation",
        "- This spec is the handoff to human hands. Delegation brief: build exactly this spec, "
        "photograph each step, measure the goal function, report the falsifier result.",
        "- Status: awaiting delegation. Only the bench speaks for the bench.",
        "",
    ]
    return "\n".join(lines)
