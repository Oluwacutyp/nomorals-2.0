"""RE missions — Pass 4's top layer.

REA waits for a human's agent to ask one tool call at a time.
Devon *originates*: an owner- or brain-stated question plus a set
of targets becomes a **mission** — she decomposes the question
herself, works the rungs in dependency order, folds everything
into one evidence chain per target, and closes with a verified
reconstruction (:mod:`.verify`). This module is that runner.

The discipline, in code:

* **Deterministic-first planning.** The planner classifies the
  question into intents over an explicit signal table, and each
  intent maps to rungs in the capability matrix — nothing else.
  Brain-supplied sub-questions go through the same classifier.
  A question that matches no intent is not dressed up: the plan
  says so and runs the census rungs (inspect + reconstruct).
* **The unknowns drive.** After the first pass, each target's
  open residual unknowns are re-read: one a provisionable rung
  can close gets that rung scheduled (provision, then re-inspect);
  one no static rung can close is parked *in the mission record*
  with the reason, not silently dropped.
* **The dynamic rung is built.** A question that only running
  the target could answer (what it does *when it runs*) gets a
  dynamic step — the target runs as a copy, in a per-run
  workdir, under the namespace sandbox with the network dropped
  (:mod:`.dynamic`). When no dynamic rung can run on this
  device, the question opens a ``dynamic_required`` unknown
  naming the concrete missing engine (Frida, GDB, or the
  sandbox), never a gate — and the mission still works every
  static rung.
* **Journaled end-to-end.** Mission and step state live in
  organ-local tables (the EvidenceStore pattern) on the shared
  database: planned → per-step results → closed, inspectable and
  resumable. A step interrupted mid-run is re-walked on resume —
  steps are idempotent because evidence is content-addressed.
  Closed missions journal into the decision journal and memory
  through the engine's Pass 1 seams.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import dynamic, provision, verify
from .failures import record_failure

__all__ = [
    "PROVIDER_ID", "MissionStore", "MissionRunner",
    "classify_question", "closer_for", "mission_status",
    "list_missions", "render_mission", "render_mission_list",
    "render_mission_status", "dynamic_unavailable_reason",
]

PROVIDER_ID = "devon.re-missions"
PROVIDER_NAME = "Devon RE missions (Pass 4)"

#: The dynamic rung's engines, in the order the rung prefers
#: them (Frida traces calls; GDB stops and inspects).
DYNAMIC_ENGINES = ("frida", "gdb")


def dynamic_unavailable_reason(root: Any = None) -> str:
    """Why no dynamic rung can run on this device, probe-true —
    the concrete missing engines and the sandbox, named, so a
    ``dynamic_required`` unknown (or a parked reason) says what
    would close it instead of naming a gate."""
    state = dynamic.dynamic_state(root)
    return ("no dynamic rung can run on this device: frida — "
            + state["frida"]["detail"] + "; gdb — "
            + state["gdb"]["detail"] + "; sandbox — "
            + state["sandbox"]["detail"])

#: Question-shape signals → intents. Each intent maps to rungs in
#: the capability matrix (see ``plan_steps``); the table is the
#: whole decomposition vocabulary, deliberately small and stated.
#: "dynamic" is not a rung intent — it is the boundary
#: classification, checked first.
INTENT_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dynamic", ("when run", "when it runs", "when executed",
                 "when launched", "at runtime", "runtime behavior",
                 "runtime behaviour", "side effect",
                 "what happens when", "does it do when")),
    ("deep", ("how does", "how do", "function", "decompile",
              "pseudocode", "internals", "logic", "algorithm")),
    ("graph", ("who calls", "callers", "callees", "calls what",
               "what calls", "references", "xrefs", "who uses",
               "used by", "class graph")),
    ("contents", ("inside", "contains", "extract", "filesystem",
                  "rootfs", "files in", "embedded")),
    ("traffic", ("talks to", "connects", "hosts", "dns", "tls",
                 "http", "traffic", "servers", "endpoint",
                 "network")),
)

#: Bounded follow-up: unknowns may re-drive the plan at most this
#: many rounds — a mission converges or reports why it cannot.
_MAX_REPLAN_ROUNDS = 2


def classify_question(question: str) -> list[str]:
    """The intents a question decomposes into, in table order."""
    text = (question or "").lower()
    return [intent for intent, signals in INTENT_SIGNALS
            if any(sig in text for sig in signals)]


def closer_for(unknown: Any) -> tuple[str, str]:
    """Which rung could close this unknown — by the unknown's own
    words and requirements, not by guessing.

    Returns (disposition, detail): ``ghidra`` / ``unblob`` when the
    unknown names that self-provisioned rung (the organ's gap
    unknowns name their closer verbatim); ``dynamic`` when closing
    demands a controlled replay — running something — which the
    dynamic rung (Frida/GDB, sandboxed) answers; ``none`` when no
    rung in the matrix answers it statically.
    """
    text = (unknown.question or "").lower()
    if "ghidra" in text:
        return ("ghidra", "the self-provisioned Ghidra rung")
    if "unblob" in text:
        return ("unblob", "the self-provisioned unblob rung")
    if unknown.required_authority == "controlled-replay":
        return ("dynamic", "closing takes a controlled replay — "
                           "the dynamic rung (Frida/GDB, sandboxed) "
                           "answers it")
    return ("none", "no rung in the capability matrix closes this "
                    "statically")


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _mission_id(question: str, targets: list[str],
                ts: float) -> str:
    digest = hashlib.sha256(_canonical({
        "question": question, "targets": targets, "ts": ts,
    }).encode("utf-8")).hexdigest()
    return f"ms:{digest[:24]}"


class MissionStore:
    """Mission + step journal on the shared house database.

    Organ-local tables (``re_missions``, ``re_mission_steps``),
    created idempotently — the EvidenceStore discipline. This
    journal is the mission's memory: status reads and resumes run
    off it, never off process state.
    """

    def __init__(self, db: Any) -> None:
        if db is None:
            raise ValueError("MissionStore needs a database handle")
        self.db = db
        try:
            with self.db.transaction():
                self.db.execute(
                    """CREATE TABLE IF NOT EXISTS re_missions (
                        mission_id TEXT PRIMARY KEY,
                        question TEXT NOT NULL DEFAULT '',
                        targets TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'planned',
                        plan TEXT NOT NULL DEFAULT '{}',
                        report TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL DEFAULT 0,
                        closed_at REAL NOT NULL DEFAULT 0
                    )""")
                self.db.execute(
                    """CREATE TABLE IF NOT EXISTS re_mission_steps (
                        mission_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        rung TEXT NOT NULL DEFAULT '',
                        action TEXT NOT NULL DEFAULT '',
                        target TEXT NOT NULL DEFAULT '',
                        detail TEXT NOT NULL DEFAULT '{}',
                        status TEXT NOT NULL DEFAULT 'pending',
                        evidence_ids TEXT NOT NULL DEFAULT '[]',
                        unknown_ids TEXT NOT NULL DEFAULT '[]',
                        note TEXT NOT NULL DEFAULT '',
                        updated_at REAL NOT NULL DEFAULT 0,
                        PRIMARY KEY (mission_id, seq)
                    )""")
        except Exception as exc:  # noqa: BLE001 - recorded, surfaced
            record_failure("reverse.missions.ensure_tables", exc)
            raise

    # ── missions ─────────────────────────────────────────────────
    def create(self, *, question: str, targets: list[str],
               plan: dict[str, Any]) -> dict[str, Any]:
        ts = time.time()
        mission_id = _mission_id(question, targets, ts)
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO re_missions (mission_id, question, "
                "targets, status, plan, report, created_at, "
                "updated_at, closed_at) VALUES (?, ?, ?, "
                "'planned', ?, '{}', ?, ?, 0)",
                (mission_id, question, _canonical(targets),
                 _canonical(plan), ts, ts))
            for seq, step in enumerate(plan.get("steps") or []):
                self.db.execute(
                    "INSERT INTO re_mission_steps (mission_id, seq, "
                    "rung, action, target, detail, status, "
                    "evidence_ids, unknown_ids, note, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'pending', '[]', "
                    "'[]', '', ?)",
                    (mission_id, seq, step.get("rung", ""),
                     step.get("action", ""), step.get("target", ""),
                     _canonical(step.get("detail") or {}), ts))
        got = self.get(mission_id)
        assert got is not None  # the row was just written
        return got

    def get(self, mission_id: str) -> dict[str, Any] | None:
        """Exact id, else a unique id-prefix, else None."""
        ref = (mission_id or "").strip()
        if not ref:
            return None
        row = self.db.query_one(
            "SELECT * FROM re_missions WHERE mission_id = ?",
            (ref,))
        if row is None:
            rows = self.db.query(
                "SELECT * FROM re_missions WHERE mission_id LIKE ? "
                "ORDER BY created_at LIMIT 2",
                (f"ms:{ref.removeprefix('ms:')}%",))
            if len(rows) != 1:
                return None
            row = rows[0]
        return self._row_mission(row)

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM re_missions ORDER BY created_at DESC "
            "LIMIT ?", (max(0, int(limit)),))
        return [self._row_mission(r) for r in rows]

    def counts(self) -> dict[str, int]:
        rows = self.db.query(
            "SELECT status, COUNT(*) AS n FROM re_missions "
            "GROUP BY status")
        return {str(r["status"]): int(r["n"]) for r in rows}

    def set_status(self, mission_id: str, status: str, *,
                   report: dict[str, Any] | None = None) -> None:
        ts = time.time()
        with self.db.transaction():
            if report is not None:
                self.db.execute(
                    "UPDATE re_missions SET status = ?, report = ?, "
                    "updated_at = ?, closed_at = ? WHERE "
                    "mission_id = ?",
                    (status, _canonical(report), ts,
                     ts if status in ("closed", "failed") else 0,
                     mission_id))
            else:
                self.db.execute(
                    "UPDATE re_missions SET status = ?, "
                    "updated_at = ? WHERE mission_id = ?",
                    (status, ts, mission_id))

    def merge_report(self, mission_id: str,
                     patch: dict[str, Any]) -> dict[str, Any]:
        mission = self.get(mission_id)
        report = dict((mission or {}).get("report") or {})
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(
                    report.get(key), dict):
                report[key] = {**report[key], **value}
            else:
                report[key] = value
        self.set_status(mission_id,
                        (mission or {}).get("status", "running"),
                        report=report)
        return report

    @staticmethod
    def _row_mission(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for key in ("targets", "plan", "report"):
            raw = out.get(key)
            if isinstance(raw, str):
                try:
                    out[key] = json.loads(raw) if raw else (
                        [] if key == "targets" else {})
                except ValueError:
                    out[key] = [] if key == "targets" else {}
        return out

    # ── steps ────────────────────────────────────────────────────
    def steps(self, mission_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM re_mission_steps WHERE mission_id = ? "
            "ORDER BY seq", (mission_id,))
        out = []
        for row in rows:
            step = dict(row)
            for key in ("detail", "evidence_ids", "unknown_ids"):
                raw = step.get(key)
                if isinstance(raw, str):
                    try:
                        step[key] = json.loads(raw) if raw else (
                            {} if key == "detail" else [])
                    except ValueError:
                        step[key] = {} if key == "detail" else []
            out.append(step)
        return out

    def add_step(self, mission_id: str, *, rung: str, action: str,
                 target: str = "",
                 detail: dict[str, Any] | None = None) -> int:
        row = self.db.query_one(
            "SELECT COALESCE(MAX(seq), -1) AS m FROM "
            "re_mission_steps WHERE mission_id = ?", (mission_id,))
        seq = int(row["m"]) + 1
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO re_mission_steps (mission_id, seq, "
                "rung, action, target, detail, status, "
                "evidence_ids, unknown_ids, note, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', '[]', '[]', "
                "'', ?)",
                (mission_id, seq, rung, action, target,
                 _canonical(detail or {}), time.time()))
        return seq

    def set_step(self, mission_id: str, seq: int, status: str, *,
                 evidence_ids: list[str] | None = None,
                 unknown_ids: list[str] | None = None,
                 note: str | None = None,
                 detail_patch: dict[str, Any] | None = None) -> None:
        current = self.db.query_one(
            "SELECT detail FROM re_mission_steps WHERE mission_id "
            "= ? AND seq = ?", (mission_id, seq))
        detail: dict[str, Any] = {}
        if current is not None:
            try:
                detail = json.loads(current["detail"] or "{}")
            except ValueError:
                detail = {}
        if detail_patch:
            detail = {**detail, **detail_patch}
        with self.db.transaction():
            self.db.execute(
                "UPDATE re_mission_steps SET status = ?, "
                "evidence_ids = ?, unknown_ids = ?, note = ?, "
                "detail = ?, updated_at = ? WHERE mission_id = ? "
                "AND seq = ?",
                (status,
                 _canonical(evidence_ids or []),
                 _canonical(unknown_ids or []),
                 note if note is not None else "",
                 _canonical(detail), time.time(), mission_id, seq))


# ── planning ─────────────────────────────────────────────────────────

def plan_steps(question: str, targets: list[str], *,
               extra_questions: tuple[str, ...] | list[str] = (),
               graph_target: str = "",
               tools_root: Any = None) -> dict[str, Any]:
    """Decompose a mission into rung steps, deterministically.

    The plan is data before it is action: intents found, notes
    stated (including the honest ones — an undecomposable
    question, a deep rung that cannot answer from this device),
    and the step list the runner will journal. Provision steps
    come first when a provisionable rung is the only thing
    between the question and its deep answer.
    """
    intents = classify_question(
        " ".join([question, *(extra_questions or ())]))
    notes: list[str] = []
    steps: list[dict[str, Any]] = []
    wants_dynamic = "dynamic" in intents
    if tools_root is not None and not isinstance(tools_root, Path):
        tools_root = Path(tools_root)
    dyn_state: dict[str, Any] | None = None
    if wants_dynamic:
        # The dynamic intent is a rung intent now: a dynamic step
        # per file target when a dynamic rung is available, and
        # the concrete missing engines named when none is.
        dyn_state = dynamic.dynamic_state(tools_root)
        if dyn_state["available"]:
            notes.append(
                "part of this question is dynamic — it asks what "
                "the target does when it runs; a dynamic step "
                f"({dyn_state['rung']}) runs each file target as "
                "a copy under the namespace sandbox, network "
                "dropped")
        else:
            notes.append(
                "part of this question is dynamic_required — it "
                "asks what the target does when it runs; "
                + dynamic_unavailable_reason(tools_root)
                + ". A dynamic_required unknown is opened per "
                  "target; every static rung still runs")
    gh = provision.ghidra_state(tools_root)
    if "deep" in intents:
        if gh["state"] == "provisionable":
            steps.append({
                "rung": "provision", "action": "provision",
                "target": "",
                "detail": {"engines": ["ghidra"],
                           "reason": "the question asks for deep "
                                     "answers and the Ghidra rung "
                                     "is provisionable"}})
            notes.append("deep intent: the Ghidra rung provisions "
                         "first, then inspections route deep")
        elif not gh["available"]:
            notes.append(
                "deep intent: the Ghidra rung is "
                f"{gh['state']} on this device ({gh['detail']}) — "
                "inspections degrade to the rungs that can answer "
                "and the gap lands as a residual unknown")
    file_targets: list[str] = []
    tree_targets: list[str] = []
    for target in targets:
        if Path(target).expanduser().is_dir():
            tree_targets.append(target)
        else:
            file_targets.append(target)
    for target in file_targets:
        steps.append({
            "rung": "inspect", "action": "inspect", "target": target,
            "detail": {"deep": "auto"}})
    # directory targets are jadx trees, not files: the graph rung
    # is the only rung that reads a tree, so it censuses them
    # whether or not the question carried graph intent — and an
    # APK file gets graph steps when the question asks for them
    graph_targets = list(tree_targets)
    if "graph" in intents:
        graph_targets.extend(
            t for t in file_targets
            if Path(t).suffix.lower() in (".apk", ".xapk", ".apks"))
    elif tree_targets:
        notes.append("directory target(s) route to the Android "
                     "graph census — the only rung that reads a "
                     "sources tree")
    for target in graph_targets:
        steps.append({
            "rung": "graph", "action": "census",
            "target": target, "detail": {}})
        if graph_target and "graph" in intents:
            kind = ("callees" if "callees" in question
                    or "calls what" in question
                    else "callers")
            steps.append({
                "rung": "graph", "action": "query",
                "target": target,
                "detail": {"query": kind,
                           "graph_target": graph_target}})
    if "graph" in intents and not graph_targets:
        notes.append("graph intent: no target is an APK or a "
                     "jadx tree, so no graph step is planned — "
                     "the Android graph rung answers over "
                     "decompiled sources only")
    if wants_dynamic and dyn_state and dyn_state["available"]:
        # the dynamic step runs after the inspect step identified
        # the target and before the reconstruction reads the
        # chain — its evidence lands in the same chain
        for target in file_targets:
            steps.append({
                "rung": "dynamic", "action": "run",
                "target": target,
                "detail": {"rung": dyn_state["rung"]}})
    for target in targets:
        steps.append({
            "rung": "reconstruct", "action": "reconstruct",
            "target": target, "detail": {}})
    if not intents:
        notes.append("the question did not decompose into rung "
                     "intents over the signal table — census rungs "
                     "only (inspect + reconstruct); the brain may "
                     "add sub-questions to focus a later mission")
    return {"intents": intents, "dynamic": wants_dynamic,
            "dynamic_available": bool(
                dyn_state and dyn_state["available"]),
            "notes": notes, "steps": steps}


# ── the runner ───────────────────────────────────────────────────────

class MissionRunner:
    """Plans, journals and works one mission at a time.

    The runner holds no mission state in memory — every read and
    write goes to the MissionStore journal, so a mission survives
    the process that started it and resumes from its own record.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.store = engine.store
        self.missions = MissionStore(engine._db)

    # ── lifecycle ────────────────────────────────────────────────
    def start(self, question: str, targets: list[str], *,
              extra_questions: tuple[str, ...] | list[str] = (),
              graph_target: str = "",
              max_steps: int | None = None) -> dict[str, Any]:
        """Plan a mission into the journal, then work it."""
        question = (question or "").strip()
        targets = [str(t).strip() for t in targets if str(t).strip()]
        if not question:
            return {"error": "a mission needs a question"}
        if not targets:
            return {"error": "a mission needs at least one target"}
        plan = plan_steps(
            question, targets, extra_questions=extra_questions,
            graph_target=graph_target,
            tools_root=self.engine._tools_root)
        plan["extra_questions"] = list(extra_questions or ())
        mission = self.missions.create(
            question=question, targets=targets, plan=plan)
        return self.run(mission["mission_id"], max_steps=max_steps)

    def run(self, mission_id: str, *,
            max_steps: int | None = None) -> dict[str, Any]:
        """Work a mission's pending steps (and its replan rounds).

        ``max_steps`` bounds this call's work — the rest stays
        pending in the journal for a later resume; that bound is
        also what an interruption looks like from the journal's
        side, which is why resume needs no special machinery.
        """
        mission = self.missions.get(mission_id)
        if mission is None:
            return {"error": f"no mission matches {mission_id!r}"}
        if mission["status"] == "closed":
            return self._result(mission)
        self.missions.set_status(mission_id, "running")
        budget = max_steps
        rounds = 0
        while True:
            worked = self._work_pending(mission_id, budget)
            if budget is not None:
                budget -= worked
                if budget <= 0:
                    break
            pending = [s for s in self.missions.steps(mission_id)
                       if s["status"] == "pending"]
            if pending:
                break  # budget exhausted mid-plan; resume later
            if rounds >= _MAX_REPLAN_ROUNDS:
                break
            appended = self._replan(mission_id)
            rounds += 1
            if not appended:
                break
        mission = self.missions.get(mission_id)
        steps = self.missions.steps(mission_id)
        if any(s["status"] == "pending" for s in steps):
            return self._result(mission)  # still running (budget)
        failed = [s for s in steps if s["status"] == "failed"]
        inspects = [s for s in steps if s["rung"] == "inspect"]
        status = "closed"
        if inspects and all(s["status"] == "failed"
                            for s in inspects):
            status = "failed"
        report = dict(mission.get("report") or {})
        report["failures"] = [
            {"seq": s["seq"], "rung": s["rung"],
             "target": s["target"], "note": s["note"]}
            for s in failed]
        report["steps_total"] = len(steps)
        report["steps_failed"] = len(failed)
        self.missions.set_status(mission_id, status, report=report)
        mission = self.missions.get(mission_id)
        self._journal_close(mission)
        return self._result(mission)

    # ── step work ────────────────────────────────────────────────
    def _work_pending(self, mission_id: str,
                      budget: int | None) -> int:
        worked = 0
        for step in self.missions.steps(mission_id):
            if step["status"] not in ("pending", "running"):
                continue
            if budget is not None and worked >= budget:
                break
            self._execute(mission_id, step)
            worked += 1
        return worked

    def _execute(self, mission_id: str, step: dict[str, Any]) -> None:
        seq = step["seq"]
        self.missions.set_step(mission_id, seq, "running")
        try:
            evidence_ids, unknown_ids, note, patch = self._dispatch(
                mission_id, step)
        except Exception as exc:  # noqa: BLE001 - a failed rung is
            # a mission fact, journaled — never a silent swallow
            record_failure(
                f"reverse.missions.step_{step['rung']}", exc)
            self.missions.set_step(
                mission_id, seq, "failed",
                note=f"{type(exc).__name__}: {exc}")
            return
        self.missions.set_step(
            mission_id, seq, "done", evidence_ids=evidence_ids,
            unknown_ids=unknown_ids, note=note,
            detail_patch=patch or None)

    def _dispatch(self, mission_id: str, step: dict[str, Any]
                  ) -> tuple[list[str], list[str], str,
                             dict[str, Any]]:
        rung = step["rung"]
        target = step["target"]
        detail = step.get("detail") or {}
        engine = self.engine
        if rung == "provision":
            report = engine.provision_engines()
            states = {name: ("ready" if row.get("available")
                             else row.get("state", "unavailable"))
                      for name, row in
                      (report.get("engines") or {}).items()}
            return [], [], ("provisioning drive: " + ", ".join(
                f"{k}={v}" for k, v in states.items())), {}
        if rung == "inspect":
            res = engine.inspect_path(
                target, deep=detail.get("deep", "auto"))
            if res.get("error"):
                raise RuntimeError(str(res["error"]))
            unknown_ids = [u["unknown_id"]
                           for u in res.get("unknowns") or []]
            mission = self.missions.get(mission_id)
            plan = (mission or {}).get("plan") or {}
            has_dynamic_step = any(
                s["rung"] == "dynamic" and s["target"] == target
                for s in self.missions.steps(mission_id))
            if plan.get("dynamic") and not has_dynamic_step:
                # no dynamic rung could run for this target, so
                # the question opens a dynamic_required unknown
                # naming the concrete missing engines — never a
                # gate, and never a silent drop
                unk = self.store.open_unknown(
                    f"what does {res['subject']['name']} do when "
                    "it runs — its runtime behavior, side effects "
                    "and network actions in motion — is "
                    "unestablished by any static rung; only a "
                    "dynamic run could answer it, and "
                    + dynamic_unavailable_reason(
                        self.engine._tools_root),
                    subject_sha256=res["subject"]["sha256"],
                    raised_by=res["evidence_ids"][0],
                    supporting=[res["evidence_ids"][0]],
                    required_authority="controlled-replay",
                    required_confidence="observed")
                unknown_ids.append(unk.unknown_id)
            return (list(res["evidence_ids"]), unknown_ids,
                    f"{res['format']} via {res['rung']}",
                    {"sha256": res["subject"]["sha256"],
                     "format": res["format"], "rung": res["rung"]})
        if rung == "dynamic":
            res = engine.run_dynamic(
                target, rung=detail.get("rung", "auto"))
            if res.get("error"):
                raise RuntimeError(str(res["error"]))
            unknown_ids = [u["unknown_id"]
                           for u in res.get("unknowns") or []]
            return (list(res["evidence_ids"]), unknown_ids,
                    f"dynamic run via {res['rung']}: exit "
                    f"{res.get('exit_code')}"
                    + (" (timed out)" if res.get("timed_out")
                       else ""),
                    {"sha256": res["subject"]["sha256"],
                     "rung": res["rung"]})
        if rung == "graph":
            if step["action"] == "query":
                res = engine.android_query(
                    target, detail.get("query", "callers"),
                    detail.get("graph_target", ""))
                if res.get("error"):
                    raise RuntimeError(str(res["error"]))
                return [res["evidence_id"]], [], (
                    f"graph {detail.get('query')} over "
                    f"{detail.get('graph_target')}"), {}
            res = engine.android_graph_build(target)
            if res.get("error"):
                raise RuntimeError(str(res["error"]))
            return (list(res["evidence_ids"]), [],
                    f"graph census: {res['stats']}",
                    {"sha256": res.get("_sha", ""),
                     "sources": res.get("sources", "")})
        if rung == "reconstruct":
            sha = self._subject_sha(mission_id, target)
            if not sha:
                raise RuntimeError(
                    f"no inspected subject for target {target!r} — "
                    "the reconstruction needs its evidence chain")
            report = verify.build_reconstruction(
                self.store, sha, mission_id=mission_id)
            self.missions.merge_report(
                mission_id, {"targets": {target: report}})
            counts = report["counts"]
            return ([report["report_evidence_id"]],
                    [u["unknown_id"]
                     for u in report.get("conflict_unknowns") or []],
                    f"{counts['claims']} claim(s), "
                    f"{counts['corroborated']} corroborated, "
                    f"{counts['conflicts']} conflict(s)", {})
        raise RuntimeError(f"unknown mission rung {rung!r}")

    def _subject_sha(self, mission_id: str, target: str) -> str:
        for step in self.missions.steps(mission_id):
            if (step["rung"] in ("inspect", "graph")
                    and step["target"] == target
                    and step["status"] == "done"):
                sha = (step.get("detail") or {}).get("sha256")
                if sha:
                    return str(sha)
        return ""

    # ── unknown-driven replanning ────────────────────────────────
    def _replan(self, mission_id: str) -> int:
        """One follow-up round driven by the open unknowns.

        For each target: an unknown whose closer is a rung that
        has not yet answered gets that rung scheduled (provision
        first when the rung is provisionable); an unknown whose
        closer is dynamic or nothing is parked in the mission
        report with its reason. Returns steps appended.
        """
        mission = self.missions.get(mission_id)
        if mission is None:
            return 0
        steps = self.missions.steps(mission_id)
        appended = 0
        parked: list[dict[str, str]] = []
        report = dict(mission.get("report") or {})
        parked.extend(report.get("parked_unknowns") or [])
        seen_parked = {p["unknown_id"] for p in parked}
        scheduled: set[tuple[str, str]] = set()
        for target in mission["targets"]:
            sha = self._subject_sha(mission_id, target)
            if not sha:
                continue
            chain = self.store.list_for_subject(sha)
            predicates = {r.predicate_type for r in chain}
            inspects_done = sum(
                1 for s in steps if s["rung"] == "inspect"
                and s["target"] == target
                and s["status"] == "done")
            for unk in self.store.list_unknowns(
                    status="open", limit=100000):
                if unk.subject_sha256 != sha:
                    continue
                disposition, why = closer_for(unk)
                if disposition in ("ghidra", "unblob"):
                    answered = (
                        any(p.startswith("native.") for p in
                            predicates)
                        if disposition == "ghidra"
                        else "firmware.inventory" in predicates)
                    if (answered or inspects_done >= 2
                            or (target, disposition) in scheduled):
                        continue
                    scheduled.add((target, disposition))
                    state = (
                        provision.ghidra_state(
                            self.engine._tools_root)
                        if disposition == "ghidra"
                        else provision.unblob_state(
                            self.engine._tools_root))
                    if state["state"] == "provisionable" and not any(
                            s["rung"] == "provision"
                            and s["status"] in ("done", "pending")
                            for s in steps):
                        self.missions.add_step(
                            mission_id, rung="provision",
                            action="provision",
                            detail={"engines": [disposition],
                                    "reason": "scheduled by "
                                              "unknown "
                                              + unk.unknown_id})
                        appended += 1
                    if state["state"] in ("ready", "provisionable"):
                        self.missions.add_step(
                            mission_id, rung="inspect",
                            action="inspect", target=target,
                            detail={"deep": "auto",
                                    "reason": "unknown-driven: "
                                              + unk.unknown_id})
                        self.missions.add_step(
                            mission_id, rung="reconstruct",
                            action="reconstruct", target=target,
                            detail={"reason": "after unknown-driven "
                                              "re-inspection"})
                        appended += 2
                    elif unk.unknown_id not in seen_parked:
                        parked.append({
                            "unknown_id": unk.unknown_id,
                            "reason": f"{why} is {state['state']} "
                                      f"on this device "
                                      f"({state['detail']})"})
                        seen_parked.add(unk.unknown_id)
                    continue
                if disposition == "dynamic":
                    state = dynamic.dynamic_state(
                        self.engine._tools_root)
                    already = any(
                        s["rung"] == "dynamic"
                        and s["target"] == target
                        for s in steps)
                    if (state["available"] and not already
                            and (target, "dynamic") not in scheduled):
                        scheduled.add((target, "dynamic"))
                        self.missions.add_step(
                            mission_id, rung="dynamic",
                            action="run", target=target,
                            detail={"rung": state["rung"],
                                    "reason": "unknown-driven: "
                                              + unk.unknown_id})
                        self.missions.add_step(
                            mission_id, rung="reconstruct",
                            action="reconstruct", target=target,
                            detail={"reason": "after unknown-driven "
                                              "dynamic run"})
                        appended += 2
                    elif unk.unknown_id not in seen_parked:
                        if state["available"]:
                            reason = (
                                "a dynamic step already ran for "
                                "this target and the question "
                                "stands — the run answered what "
                                "it could")
                        else:
                            reason = dynamic_unavailable_reason(
                                self.engine._tools_root)
                        parked.append({"unknown_id": unk.unknown_id,
                                       "reason": reason})
                        seen_parked.add(unk.unknown_id)
                    continue
                if unk.unknown_id not in seen_parked:
                    parked.append({"unknown_id": unk.unknown_id,
                                   "reason": why})
                    seen_parked.add(unk.unknown_id)
        if parked != (report.get("parked_unknowns") or []):
            self.missions.merge_report(
                mission_id, {"parked_unknowns": parked})
        return appended

    # ── close-out ────────────────────────────────────────────────
    def _journal_close(self, mission: dict[str, Any]) -> None:
        """Findings land in Devon's systems (the Pass 1 seams):
        the decision journal and memory get the mission as an
        evidence-backed entry — counts and ids, never bare prose
        dressed as a finding."""
        steps = self.missions.steps(mission["mission_id"])
        evidence: list[Any] = []
        seen: set[str] = set()
        for step in steps:
            for eid in step.get("evidence_ids") or []:
                if eid in seen:
                    continue
                seen.add(eid)
                rec = self.store.get(eid)
                if rec is not None:
                    evidence.append(rec)
        unknowns: list[Any] = []
        for step in steps:
            for uid in step.get("unknown_ids") or []:
                unk = self.store.get_unknown(uid)
                if unk is not None and unk.status == "open":
                    unknowns.append(unk)
        engine = self.engine
        try:
            engine._journal_finding(
                f"RE mission: {mission['question'][:60]}",
                "mission", "", evidence[:50], unknowns)
            if evidence:
                engine._remember_finding(
                    f"RE mission: {mission['question'][:60]}",
                    "mission", "", evidence[0], evidence, unknowns)
        except Exception as exc:  # noqa: BLE001 - journaling is a
            # seam, not the mission; its failure is recorded
            record_failure("reverse.missions.journal_close", exc)

    def _result(self, mission: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission_id": mission["mission_id"],
            "question": mission["question"],
            "targets": mission["targets"],
            "status": mission["status"],
            "plan": mission.get("plan") or {},
            "steps": self.missions.steps(mission["mission_id"]),
            "report": mission.get("report") or {},
            "created_at": mission.get("created_at"),
            "closed_at": mission.get("closed_at"),
        }


# ── module-level reads (surfaces) ────────────────────────────────────

def mission_status(engine: Any, mission_id: str) -> dict[str, Any]:
    store = MissionStore(engine._db)
    mission = store.get(mission_id)
    if mission is None:
        return {"error": f"no mission matches {mission_id!r}"}
    return {
        "mission_id": mission["mission_id"],
        "question": mission["question"],
        "targets": mission["targets"],
        "status": mission["status"],
        "plan": mission.get("plan") or {},
        "steps": store.steps(mission["mission_id"]),
        "report": mission.get("report") or {},
        "created_at": mission.get("created_at"),
        "closed_at": mission.get("closed_at"),
    }


def list_missions(engine: Any, limit: int = 20) -> dict[str, Any]:
    store = MissionStore(engine._db)
    rows = store.list(limit=limit)
    return {"missions": [{
        "mission_id": m["mission_id"], "question": m["question"],
        "targets": m["targets"], "status": m["status"],
        "created_at": m.get("created_at"),
        "closed_at": m.get("closed_at"),
    } for m in rows], "counts": store.counts()}


# ── renders ──────────────────────────────────────────────────────────

def render_mission(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"⚠️ {result['error']}"
    plan = result.get("plan") or {}
    lines = [
        f"🎯 RE mission [{result['mission_id']}] — "
        f"{result['status']}",
        f"question: {result['question']}",
    ]
    if plan.get("intents"):
        lines.append("intents: " + ", ".join(plan["intents"]))
    for note in plan.get("notes") or []:
        lines.append(f"  · {note}")
    report = result.get("report") or {}
    for target, rep in (report.get("targets") or {}).items():
        counts = rep.get("counts") or {}
        lines.append(
            f"  {target}: {counts.get('claims', 0)} claim(s) "
            f"({counts.get('corroborated', 0)} corroborated), "
            f"{counts.get('conflicts', 0)} conflict(s), "
            f"{len(rep.get('unknowns_open') or [])} open unknown(s)")
    for parked in report.get("parked_unknowns") or []:
        lines.append(f"  ? parked [{parked['unknown_id']}]: "
                     f"{parked['reason']}")
    for failure in report.get("failures") or []:
        lines.append(f"  ✗ step {failure['seq']} "
                     f"({failure['rung']}) failed: {failure['note']}")
    steps = result.get("steps") or []
    done = sum(1 for s in steps if s["status"] == "done")
    lines.append(f"steps: {done}/{len(steps)} done")
    return "\n".join(lines)


def render_mission_list(payload: dict[str, Any]) -> str:
    rows = payload.get("missions") or []
    if not rows:
        return "RE missions: none yet"
    lines = [f"RE missions: {len(rows)}"]
    for row in rows:
        lines.append(
            f"  [{row['status']}] {row['mission_id']} — "
            f"{row['question'][:70]} "
            f"({len(row['targets'])} target(s))")
    return "\n".join(lines)


def render_mission_status(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"⚠️ {result['error']}"
    lines = [render_mission(result), "steps:"]
    for step in result.get("steps") or []:
        lines.append(
            f"  {step['seq']}. [{step['status']}] {step['rung']}"
            + (f" {step['target']}" if step["target"] else "")
            + (f" — {step['note']}" if step["note"] else ""))
    for _name, rep in ((result.get("report") or {}).get(
            "targets") or {}).items():
        lines.append("")
        lines.append(verify.render_reconstruction(rep))
    return "\n".join(lines)
