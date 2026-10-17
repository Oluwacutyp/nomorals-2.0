"""ImpossibleMadePossible — the class the user named. Full pipeline, every
problem, no degraded simple mode: intake -> formulate -> classify ->
prior art -> transfer -> constrain (gate) -> reframe -> hypothesize ->
attempt -> report. Profile-gating affects resources per tick only, never
capability.
"""

from __future__ import annotations

from . import classify as _classify
from . import first_principles as _fp
from . import formulate as _formulate
from . import feasible as _feasible
from . import human as _human
from . import hypothesis as _hypothesis
from . import lattice as _lattice
from . import prior_art as _prior_art
from . import transfer as _transfer
from .attempt import AttemptQueue
from .charter import charter as _charter
from .ledger import HonestyLedger


class ImpossibleMadePossible:
    """Always-on, cross-domain conceive-and-solve class."""

    def __init__(self, ledger_path=None, queue_path=None, spec_dir=None):
        self.ledger = HonestyLedger(ledger_path)
        self.queue = AttemptQueue(queue_path, spec_dir)

    # -- the interface from the ArenaAI §13, plus first_principles -----------
    def submit(self, problem: str, *, domain: str = "hardware",
               fetch_prior_art: bool = False, **gate_kwargs) -> dict:
        """Full dossier for a problem. Underspecified problems return
        clarification questions instead of a guessed answer."""
        formulation = _formulate.formulate(problem)
        if formulation.underspecified:
            return {"status": "needs_clarification",
                    "formulation": formulation.to_dict(),
                    "clarifications": formulation.clarifications,
                    "note": "Not guessing — answer the clarifications and resubmit."}
        classification = _classify.classify(formulation)
        art = _prior_art.prior_art(problem) if fetch_prior_art else {
            "query": problem, "results": {}, "failed_attempts": [], "key_gated": _prior_art.key_gated_sources(),
            "summary": "Prior-art fetch skipped (fetch_prior_art=False) — pass True for live sources.",
            "novelty_note": "No prior-art fetch run."}
        transfers = _transfer.transfer_all(formulation)
        fp = _fp.decompose(problem, requirements=list(formulation.constraints), domain=domain)
        gate_result = _feasible.gate(formulation, **gate_kwargs)
        hyp = _hypothesis.hypothesize(formulation, gate_result, transfers)
        reframed = _hypothesis.reframe(formulation, gate_result)
        common = _human.common_sense(problem)
        simulation = _human.mental_simulation(f"Build the reframed version: {reframed['reframed_goal'][:200]}")
        dossier = {
            "formulation": formulation.to_dict(),
            "classification": classification,
            "prior_art": art,
            "transfer": transfers,
            "first_principles": fp.to_dict(),
            "common_sense": common,
            "mental_simulation": simulation,
            "gate": gate_result,
            "hypothesis": hyp,
            "reframed": reframed,
            "domain": domain,
            "status": "hypothesis",
        }
        entry = self.ledger.record(claim=reframed["reframed_goal"], formulation=dossier["formulation"],
                                   classification=classification, prior_art=art, transfer=transfers,
                                   gate=gate_result, hypothesis=hyp, reframed=reframed,
                                   status="hypothesis")
        dossier["claim_id"] = entry["id"]
        self.queue.enqueue(entry["id"], problem, dossier)
        dossier["report"] = self._report(dossier)
        return dossier

    def dossier(self, claim_id: str) -> dict | None:
        entry = self.ledger.get(claim_id)
        if entry is None:
            return None
        return {"claim_id": entry["id"], "entry": entry,
                "next_hint": self.ledger.next_attempt_hint(claim_id)}

    def queue_list(self, status: str | None = None) -> list[dict]:
        return self.queue.queue(status)

    def attempt(self, claim_id: str) -> dict:
        entry = self.ledger.get(claim_id)
        if entry is None:
            return {"ok": False, "error": f"No claim with id '{claim_id}'."}
        dossier = {"formulation": entry.get("formulation", {}),
                   "gate": entry.get("gate", {}), "hypothesis": entry.get("hypothesis", {}),
                   "reframed": entry.get("reframed", {}),
                   "transfer": {"triz": [], "inversion": []},
                   "first_principles": {}, "domain": "hardware"}
        return self.queue.attempt(claim_id, dossier, ledger=self.ledger)

    def status(self, claim_id: str) -> dict:
        entry = self.ledger.get(claim_id)
        if entry is None:
            return {"ok": False, "error": f"No claim with id '{claim_id}'."}
        return {"claim_id": entry["id"], "status": entry["status"],
                "attempts": len(entry.get("attempts", [])),
                "conflicts": entry.get("conflicts", []),
                "falsifier": entry.get("falsifier", "")}

    def charter(self) -> dict:
        return _charter()

    def lattice(self) -> dict:
        return _lattice.lattice()

    def digest(self, **kwargs) -> dict:
        from .digest import digest_tick
        from .sources import digest_sources
        return digest_tick(queue=self.queue, ledger=self.ledger,
                           prior_art_fn=_prior_art.prior_art,
                           digestion_fn=digest_sources, **kwargs)

    def cross_query(self, problem: str, **kwargs) -> dict:
        """One problem, three knowledge systems hand in hand
        (Impossible + WisdomKeeper + BookForge). See bridges.py."""
        from .bridges import cross_query as _cross
        return _cross(problem, **kwargs)

    @staticmethod
    def _report(dossier: dict) -> str:
        f = dossier["formulation"]
        g = dossier["gate"]
        lines = [
            f"Problem: {f['raw']}",
            f"Goal (function): {f['goal_function']}",
            f"Classification: {dossier['classification']['primary']}",
            f"Gate verdict: {g['verdict']} — {g['why']}",
            f"Reframed: {dossier['reframed']['reframed_goal']}",
            f"First principles: {dossier['first_principles']['rebuilt_solution']}",
            f"Falsifier: {dossier['hypothesis']['falsifier']}",
            f"Cheapest killing experiment: {dossier['hypothesis']['cheapest_killing_experiment']}",
            f"Competing hypotheses: {len(dossier['hypothesis'].get('hypotheses', []) or [])} "
            f"(best first: {dossier['hypothesis'].get('primary_angle', '')}; "
            f"alternatives stay in the ledger — a failed attempt moves to the next)",
            f"Prior art: {dossier['prior_art']['summary']}",
            f"Claim id: {dossier.get('claim_id', '')}",
        ]
        return "\n".join(lines)
