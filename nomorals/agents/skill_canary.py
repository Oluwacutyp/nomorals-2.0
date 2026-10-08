"""Canary rollouts for skill edits — ship to a fraction first, promote or
revert on the numbers.

When a skill edit passes the gate in autonomous mode, it can ride a canary
instead of going live to everyone at once:

  1. **start** — record both versions in the ``skill_versions`` hash chain,
     open a ``canary_runs`` row (default 20% of tasks see the new version).
  2. **observe** — each task using the skill logs success/failure against
     the version it saw.
  3. **evaluate** — once there is a minimum sample (>= 20 canary tasks or
     48 hours, whichever comes first), compare canary vs baseline success
     rates. Auto-promote when canary >= baseline, auto-revert when worse.
     The decision and the numbers are recorded on the run.

``choose_version`` / ``resolve_body`` are the routing seam: skill consumers
call them to decide which version a given task sees. ``decide`` is a pure
function so the promotion rule is unit-testable without a database.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["CanaryRollout", "CanaryRun", "decide", "register"]

DEFAULT_FRACTION = 0.2
DEFAULT_MIN_SAMPLE = 20
DEFAULT_MIN_HOURS = 48.0


def decide(canary_ok: int, canary_n: int, baseline_ok: int,
           baseline_n: int) -> str:
    """Pure promotion rule. Returns 'promote', 'revert', or 'inconclusive'."""
    if canary_n <= 0:
        return "inconclusive"
    canary_rate = canary_ok / canary_n
    baseline_rate = (baseline_ok / baseline_n) if baseline_n > 0 else 0.5
    if canary_rate >= baseline_rate:
        return "promote"
    return "revert"


@dataclass
class CanaryRun:
    id: str
    skill_name: str
    canary_hash: str
    baseline_hash: str
    fraction: float
    status: str = "running"      # running|promoted|reverted
    decision: str = ""           # promote|revert (once decided)
    decision_detail: dict | None = None
    started_at: float = 0.0
    decided_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "skill_name": self.skill_name,
            "canary_hash": self.canary_hash,
            "baseline_hash": self.baseline_hash, "fraction": self.fraction,
            "status": self.status, "decision": self.decision,
            "decision_detail": self.decision_detail or {},
            "started_at": self.started_at, "decided_at": self.decided_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CanaryRun":
        try:
            detail = json.loads(row.get("decision_detail") or "{}")
        except Exception:  # noqa: BLE001
            detail = {}
        return cls(
            id=row["id"], skill_name=row.get("skill_name", ""),
            canary_hash=row.get("canary_hash", ""),
            baseline_hash=row.get("baseline_hash", ""),
            fraction=float(row.get("fraction", DEFAULT_FRACTION)),
            status=row.get("status", "running"),
            decision=row.get("decision", ""), decision_detail=detail,
            started_at=float(row.get("started_at", 0)),
            decided_at=float(row.get("decided_at", 0)))


class CanaryRollout:
    def __init__(self, context: Any, *,
                 min_sample: int = DEFAULT_MIN_SAMPLE,
                 min_hours: float = DEFAULT_MIN_HOURS,
                 rng: Any | None = None) -> None:
        self.context = context
        self.db = context.db
        self.min_sample = min_sample
        self.min_hours = min_hours
        self._rng = rng or random.Random()

    # ── versions: hash chain per skill ──────────────────────────────────────
    @staticmethod
    def _hash(parent_hash: str, body: str) -> str:
        return hashlib.sha256(
            f"{parent_hash}\n".encode() + body.encode("utf-8")).hexdigest()[:16]

    def latest_hash(self, skill_name: str) -> str:
        try:
            row = self.db.query_one(
                "SELECT version_hash FROM skill_versions WHERE skill_name=? "
                "ORDER BY created_at DESC LIMIT 1", (skill_name,))
        except Exception:  # noqa: BLE001 — pre-migration-52 databases
            return ""
        return row["version_hash"] if row else ""

    def record_version(self, skill_name: str, body: str, *,
                       source: str = "") -> str:
        """Append a version to the skill's hash chain. Returns its hash."""
        parent = self.latest_hash(skill_name)
        vhash = self._hash(parent, body)
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO skill_versions (id, skill_name, "
                "version_hash, parent_hash, body, source, created_at) VALUES "
                "(?,?,?,?,?,?,?)",
                (new_short_id("skver"), skill_name, vhash, parent, body,
                 source[:80], time.time()))
        except Exception:  # noqa: BLE001
            _log.debug("skill_versions store failed", exc_info=True)
        return vhash

    def get_body(self, skill_name: str, version_hash: str) -> str | None:
        try:
            row = self.db.query_one(
                "SELECT body FROM skill_versions WHERE skill_name=? AND "
                "version_hash=?", (skill_name, version_hash))
        except Exception:  # noqa: BLE001
            return None
        return row["body"] if row else None

    def versions(self, skill_name: str, *,
                 limit: int = 20) -> list[dict[str, Any]]:
        try:
            rows = self.db.query(
                "SELECT version_hash, parent_hash, source, created_at FROM "
                "skill_versions WHERE skill_name=? ORDER BY created_at DESC "
                "LIMIT ?", (skill_name, limit))
        except Exception:  # noqa: BLE001
            return []
        return [dict(r) for r in rows]

    def restore_version(self, skill_name: str,
                        version_hash: str) -> dict[str, Any]:
        """Restore any recorded version as the live skill body (db skills)."""
        from .skills import SkillLibrary
        body = self.get_body(skill_name, version_hash)
        if body is None:
            return {"ok": False, "error": "no such version"}
        lib = SkillLibrary(self.db)
        skill = lib.get_by_name(skill_name)
        if skill is None:
            return {"ok": False, "error": "no such skill"}
        lib.save(skill.name, kind=skill.kind, body=body,
                 description=skill.description, tags=list(skill.tags),
                 source=skill.source, id=skill.id)
        self.record_version(skill_name, body, source="restore")
        return {"ok": True, "skill": skill_name,
                "version_hash": version_hash}

    # ── canary lifecycle ────────────────────────────────────────────────────
    def start(self, skill_name: str, canary_body: str, *,
              baseline_body: str = "",
              baseline_hash: str = "",
              fraction: float = DEFAULT_FRACTION) -> dict[str, Any]:
        """Open a canary run. The live skill keeps serving the baseline body;
        ``fraction`` of tasks are routed to the canary body."""
        if not baseline_hash:
            baseline_hash = self.record_version(skill_name, baseline_body,
                                                source="canary-baseline")
        canary_hash = self.record_version(skill_name, canary_body,
                                          source="canary")
        run_id = new_short_id("canary")
        try:
            self.db.execute(
                "INSERT INTO canary_runs (id, skill_name, canary_hash, "
                "baseline_hash, fraction, status, started_at) VALUES "
                "(?,?,?,?,?,'running',?)",
                (run_id, skill_name, canary_hash, baseline_hash, fraction,
                 time.time()))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:200]}
        _log.info("canary started for %s (%s -> %s, %.0f%%)",
                  skill_name, baseline_hash, canary_hash, fraction * 100)
        return {"ok": True, "id": run_id, "skill_name": skill_name,
                "canary_hash": canary_hash, "baseline_hash": baseline_hash,
                "fraction": fraction}

    def active(self, skill_name: str) -> CanaryRun | None:
        try:
            row = self.db.query_one(
                "SELECT * FROM canary_runs WHERE skill_name=? AND status="
                "'running' ORDER BY started_at DESC LIMIT 1", (skill_name,))
        except Exception:  # noqa: BLE001
            return None
        return CanaryRun.from_row(row) if row else None

    def observe(self, skill_name: str, version_hash: str,
                success: bool) -> None:
        """Log one task outcome against the version it saw. Never raises."""
        try:
            run = self.active(skill_name)
            if run is None:
                return
            if version_hash not in (run.canary_hash, run.baseline_hash):
                return
            self.db.execute(
                "INSERT INTO canary_observations (id, canary_id, "
                "version_hash, success, observed_at) VALUES (?,?,?,?,?)",
                (new_short_id("cob"), run.id, version_hash,
                 1 if success else 0, time.time()))
        except Exception:  # noqa: BLE001
            _log.debug("canary observe failed", exc_info=True)

    # ── routing seam ────────────────────────────────────────────────────────
    def choose_version(self, skill_name: str) -> str:
        """'canary' or 'baseline' — which version should this task see?"""
        run = self.active(skill_name)
        if run is None:
            return "baseline"
        return "canary" if self._rng.random() < run.fraction else "baseline"

    def resolve_body(self, skill_name: str, live_body: str) -> tuple[str, str]:
        """(body, version) a task should use for this skill right now."""
        run = self.active(skill_name)
        if run is None:
            return live_body, "baseline"
        if self._rng.random() < run.fraction:
            body = self.get_body(skill_name, run.canary_hash)
            if body is not None:
                return body, "canary"
        return live_body, "baseline"

    # ── decision ────────────────────────────────────────────────────────────
    def _counts(self, run: CanaryRun) -> dict[str, tuple[int, int]]:
        """(ok, n) per version for a run."""
        out = {"canary": (0, 0), "baseline": (0, 0)}
        try:
            rows = self.db.query(
                "SELECT version_hash, success, COUNT(*) AS n FROM "
                "canary_observations WHERE canary_id=? GROUP BY "
                "version_hash, success", (run.id,))
        except Exception:  # noqa: BLE001
            return out
        agg: dict[str, list[int]] = {}
        for r in rows:
            key = ("canary" if r["version_hash"] == run.canary_hash
                   else "baseline")
            ok, n = agg.get(key, [0, 0])
            if r["success"]:
                ok += r["n"]
            n += r["n"]
            agg[key] = [ok, n]
        for k, (ok, n) in agg.items():
            out[k] = (ok, n)
        return out

    def ready(self, run: CanaryRun) -> tuple[bool, str]:
        counts = self._counts(run)
        canary_n = counts["canary"][1]
        age_h = (time.time() - run.started_at) / 3600.0
        if canary_n >= self.min_sample:
            return True, f"min sample reached ({canary_n} canary tasks)"
        if age_h >= self.min_hours:
            return True, f"min age reached ({age_h:.1f}h)"
        return False, (f"waiting: {canary_n}/{self.min_sample} canary tasks, "
                       f"{age_h:.1f}/{self.min_hours}h")

    def evaluate(self, skill_name: str, *,
                 proof_check: Any = None) -> dict[str, Any]:
        """Decide a running canary: promote, revert, or keep waiting. The
        decision and the numbers are recorded on the run.

        ``proof_check`` is an optional ``(skill_name, version_hash) -> bool``
        callable (see ``skill_proving.canary_proof_check``).  When given, a
        "promote" decision is downgraded to "revert" unless the canary
        version has a passing proof test — the numbers can only promote
        what the tests have already proven.  Untested versions are
        refused, never promoted.
        """
        run = self.active(skill_name)
        if run is None:
            return {"ok": False, "error": "no running canary"}
        ok_ready, why = self.ready(run)
        counts = self._counts(run)
        c_ok, c_n = counts["canary"]
        b_ok, b_n = counts["baseline"]
        if not ok_ready:
            return {"ok": True, "decision": "waiting", "reason": why,
                    "canary": {"ok": c_ok, "n": c_n},
                    "baseline": {"ok": b_ok, "n": b_n}}
        decision = decide(c_ok, c_n, b_ok, b_n)
        detail = {
            "canary": {"ok": c_ok, "n": c_n,
                       "rate": round(c_ok / c_n, 3) if c_n else 0.0},
            "baseline": {"ok": b_ok, "n": b_n,
                         "rate": round(b_ok / b_n, 3) if b_n else 0.0},
            "ready_reason": why,
        }
        if decision == "promote" and proof_check is not None:
            try:
                proven = bool(proof_check(skill_name, run.canary_hash))
            except Exception:  # noqa: BLE001
                proven = False
            if not proven:
                decision = "revert"
                detail["proof_gate"] = (
                    "promotion refused: no passing proof test for the "
                    "canary version")
        if decision == "inconclusive":
            # aged out with no canary data: keep collecting, don't judge
            status = "running"
        else:
            status = "promoted" if decision == "promote" else "reverted"
        try:
            self.db.execute(
                "UPDATE canary_runs SET status=?, decision=?, "
                "decision_detail=?, decided_at=? WHERE id=?",
                (status, decision, json.dumps(detail), time.time(), run.id))
        except Exception:  # noqa: BLE001
            pass
        _log.info("canary %s for %s: %s %s", run.id, skill_name, decision,
                  detail)
        return {"ok": True, "decision": decision, "run_id": run.id,
                "status": status, **detail}

    def history(self, skill_name: str = "",
                limit: int = 20) -> list[CanaryRun]:
        try:
            q = "SELECT * FROM canary_runs"
            args: tuple = ()
            if skill_name:
                q += " WHERE skill_name=?"
                args = (skill_name,)
            q += " ORDER BY started_at DESC LIMIT ?"
            rows = self.db.query(q, (*args, limit))
        except Exception:  # noqa: BLE001
            return []
        return [CanaryRun.from_row(r) for r in rows]


# ── registry ────────────────────────────────────────────────────────────────
def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "skill_canary",
        description=(
            "Canary rollouts for skill edits: start a canary, log task "
            "outcomes, evaluate (auto promote/revert on the numbers), "
            "inspect version history, restore any version. action=start | "
            "observe | evaluate | history | versions | restore."
        ),
        capability="memory.write",
        parameters={
            "action": "str — start|observe|evaluate|history|versions|restore",
            "skill": "str — skill name",
            "version_hash": "str — for observe/restore",
            "success": "str — true|false, for observe",
            "limit": "int — for history/versions",
        },
    )
    def skill_canary(
        action: str = "history", *, skill: str = "", version_hash: str = "",
        success: str = "true", limit: str = "10",
    ) -> dict[str, Any]:
        rollout = CanaryRollout(context)
        action = (action or "history").strip().lower()
        if action == "observe":
            rollout.observe(skill, version_hash,
                            str(success).lower() not in ("false", "0", "no"))
            return {"ok": True}
        if action == "evaluate":
            return rollout.evaluate(skill)
        if action == "versions":
            try:
                n = int(limit or 10)
            except ValueError:
                n = 10
            return {"versions": rollout.versions(skill, limit=n)}
        if action == "restore":
            return rollout.restore_version(skill, version_hash)
        if action == "start":
            return {"ok": False,
                    "error": "start a canary via the skill_evolve loop "
                             "(autonomous mode) so the gate runs first"}
        try:
            n = int(limit or 10)
        except ValueError:
            n = 10
        return {"runs": [r.to_dict()
                         for r in rollout.history(skill, limit=n)]}
