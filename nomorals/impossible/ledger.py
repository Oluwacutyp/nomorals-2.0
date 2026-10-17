"""Stage 10 — HONESTY LEDGER. Every claim gets a record; conflicts surface,
never averaged. A claim is never promoted past its evidence.

Backed by a JSON file (durable, human-readable, no new DB). Mirrors the
ResearchClaim conflict discipline in agents/research_digest.py: contradictions
are detected and recorded, not smoothed away.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

DEFAULT_PATH = Path("data/impossible_ledger.json")


def _default_path() -> Path:
    return DEFAULT_PATH


class HonestyLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"claims": []}, indent=2), encoding="utf-8")

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {"claims": []}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, claim: str, formulation: dict | None = None, classification: dict | None = None,
               prior_art: dict | None = None, transfer: dict | None = None,
               gate: dict | None = None, hypothesis: dict | None = None,
               reframed: dict | None = None, status: str = "hypothesis") -> dict:
        entry = {
            "id": hashlib.sha256(f"{claim}|{time.time()}".encode()).hexdigest()[:12],
            "claim": claim,
            "formulation": formulation or {},
            "classification": classification or {},
            "prior_art_summary": (prior_art or {}).get("summary", ""),
            "transfer_methods": list((transfer or {}).keys()),
            "gate": {"verdict": (gate or {}).get("verdict"), "why": (gate or {}).get("why"),
                     "failed": (gate or {}).get("failed", [])},
            "hypothesis": hypothesis or {},
            "reframed": reframed or {},
            "status": status,
            "falsifier": (hypothesis or {}).get("falsifier", ""),
            "attempts": [],
            "conflicts": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "checked_by": "impossible-gate",
            "what_would_change_my_mind": "A test result contradicting the gate, or prior art showing the exact thing already built/failed.",
        }
        data = self._load()
        data["claims"].append(entry)
        self._save(data)
        return entry

    def get(self, claim_id: str) -> dict | None:
        for c in self._load().get("claims", []):
            if c["id"] == claim_id or c["id"].startswith(claim_id):
                return c
        return None

    def list(self, status: str | None = None) -> list[dict]:
        claims = self._load().get("claims", [])
        return [c for c in claims if status is None or c["status"] == status]

    def record_attempt(self, claim_id: str, outcome: str, evidence: str,
                       lesson: str, new_status: str | None = None) -> dict | None:
        """Log an attempt + its lesson. Failed attempts change the next attempt:
        the lesson is stored on the claim and surfaced by next_attempt_hint()."""
        data = self._load()
        for c in data.get("claims", []):
            if c["id"] == claim_id or c["id"].startswith(claim_id):
                c["attempts"].append({"outcome": outcome, "evidence": evidence,
                                      "lesson": lesson, "at": time.time()})
                if new_status:
                    c["status"] = new_status
                c["updated_at"] = time.time()
                self._save(data)
                return c
        return None

    def add_conflict(self, claim_id: str, contradicting: str, evidence: str) -> dict | None:
        """Surface a contradiction — never average it away."""
        data = self._load()
        for c in data.get("claims", []):
            if c["id"] == claim_id or c["id"].startswith(claim_id):
                c["conflicts"].append({"contradicting": contradicting, "evidence": evidence, "at": time.time()})
                self._save(data)
                return c
        return None

    def next_attempt_hint(self, claim_id: str) -> str:
        c = self.get(claim_id)
        if not c:
            return ""
        lessons = [a.get("lesson", "") for a in c.get("attempts", []) if a.get("lesson")]
        if not lessons:
            return "No attempts yet — run the cheapest killing experiment first."
        return "Learned from past attempts (do not repeat these): " + " | ".join(lessons)
