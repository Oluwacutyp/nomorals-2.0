"""WS3 P5 — the idea-development cell.

Raw ideas stop sitting verbatim and start growing.  An idea captured
in :class:`~nomorals.goals.IdeaTracker` (title + description, stored
byte-identical forever) can be *developed*: a delegation-contract cell
on the shipped substrate (:mod:`nomorals.agents.delegation`) runs

    EXPAND → CONNECT → RESEARCH → ICE rescore → VERDICT
    → skeptic → verifier → completion gate → exit

and leaves a human-readable development record beside the raw idea.

The raw capture is IMMUTABLE: development only ever reads the idea
row and writes its own ``idea_development`` table (runtime-created,
the IdeaTracker's own pattern — ``migrations.py`` stays frozen).
The record itself is a fenced artifact (``idea-dev:<idea_id>``): one
contract owns it at a time, revisions only move forward, and a stale
write fails the run in words instead of overwriting a newer record.

Delegation pieces used, honestly:

- USED: contracts (one per run, worker ≠ verifier enforced by the
  store), the claim ledger (``submit_evidence`` / ``verify_claim``),
  skeptic challenges (``challenge`` / ``resolve_challenge``), the
  verifier's ``sign_off``, the completion gate (``complete`` — the
  exit executes only when the gate passes), fenced artifacts
  (``claim_artifact`` / ``write_artifact``), and ``report_failure``
  so a refused run leaves the contract honestly failed.
- NOT used: ``replan`` (a failed run is re-run as a fresh contract —
  the idea never changes between runs, so there is nothing to
  re-plan), ``waive`` (no arbiter shortcut around the gate: a record
  the verifier refuses stays refused), the ``complete`` idempotency
  store (one completion per contract by construction, and
  ``complete`` itself is replay-safe), and ``contract_for_step``
  (the mission-step dispatcher seam — a development run is not a
  mission step).

Exits (exactly the census set): ``promote`` files a goal in its WS2
class through the real ``IdeaTracker.promote_to_goal`` path;
``draft`` queues a post draft in DraftQueue and proposes it, where
the /drafts review surface governs it (never a post — #43);
``mission`` persists a mission through ``MissionStore.create_new``;
``park`` leaves the idea active with a named revival trigger — a
parked record without a trigger is a defect, so the trigger is
structural: the owner's words, or the named default below.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Callable, Mapping

from ..goals import GoalTracker, Idea, IdeaTracker
from ..goals.classes import GoalClassError, get_class
from ..goals.surface import OWNER_USER_ID, run_sync, suggest_class
from .delegation import DelegationError, DelegationStore

__all__ = [
    "DEFAULT_TRIGGER",
    "EXITS",
    "Finding",
    "IdeaDevelopment",
    "compose_draft_text",
    "make_brain_expand_fn",
    "make_research_fn",
    "register",
    "render_record",
    "resolve_idea",
    "template_expansion",
]

#: The cell's roles on every contract.  Worker and verifier are
#: different holders and the store enforces it; the skeptic and the
#: researcher are separate producers so no claim verifies itself.
WORKER = "idea-cell:developer"
RESEARCHER = "idea-cell:researcher"
VERIFIER = "idea-cell:verifier"
SKEPTIC = "idea-cell:skeptic"
WITNESS = "idea-cell:witness"
STEWARD = "idea-cell:steward"
SUPERVISOR = "idea-cell:supervisor"
ARBITER = "owner"

#: The census exits, canonical names.
EXITS = ("promote", "draft", "mission", "park")

#: Surface spellings that mean the promote exit.
EXIT_ALIASES = {"goal": "promote"}

SHAPES = ("project", "content", "task", "someday")

#: A parked idea always carries a revival trigger.  When the owner
#: names none, this is the trigger — a real cadence (the weekly idea
#: review), recorded in words, never an empty string.
DEFAULT_TRIGGER = "the next weekly idea review"

#: Projects connection lookup names — matched against the idea's own
#: words (title, description, tags), never assumed.
KNOWN_PROJECTS = ("Devon", "GoldenSwing", "Think Creativity Think Peace")

_DDL = """
CREATE TABLE IF NOT EXISTS idea_development (
    dev_id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    contract_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    record TEXT NOT NULL DEFAULT '{}',
    revision INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_idea_development_idea
    ON idea_development(idea_id, created_at);
"""


class Finding:
    """One research finding as the pipeline accepts it.

    ``source`` is the provenance (a URL or a named source).  A finding
    whose source is empty is refused — counted in the record, never
    submitted to the claim ledger.
    """

    def __init__(self, title: str, source: str, snippet: str = "") -> None:
        self.title = str(title or "")
        self.source = str(source or "")
        self.snippet = str(snippet or "")


# ── expansion rungs ───────────────────────────────────────────────────────


def template_expansion(idea: Idea) -> dict[str, Any]:
    """The deterministic expansion rung.

    Used when no brain expander is attached (CLI, tests) or the brain
    rung fails.  Honest about its limits in the record: it names the
    shape from the idea's own words and says plainly what it cannot
    judge.  The basis is recorded as ``template`` — never dressed up
    as a model's work.
    """
    text = (idea.title + " " + (idea.description or "")).lower()
    tags = {str(t).lower() for t in idea.tags}
    if tags & {"post", "content", "announce", "article"} or any(
            w in text for w in ("post about", "article", "announce")):
        shape = "content"
    elif tags & {"someday", "later", "maybe"}:
        shape = "someday"
    elif tags & {"mission", "task", "todo"} or "task:" in text:
        shape = "task"
    else:
        shape = "project"
    description = (idea.description or "").strip()
    what = description if description else (
        "An idea captured as “" + idea.title + "” with no further "
        "detail yet — development starts from the title alone.")
    suggested = suggest_class(idea.title + " " + (idea.description or ""))
    return {
        "what": what,
        "why": ("Not yet argued — the template rung names the shape "
                "only; it does not invent an audience or a case."),
        "takes": ("Not yet scoped — no estimate from the template "
                  "rung."),
        "unknowns": ["who it is for", "what done looks like",
                     "what it takes"],
        "shape": shape,
        "basis": "template",
        "draft_text": "",
        "suggested_class": suggested.id if suggested else "",
    }


def _normalize_expansion(raw: Any, idea: Idea) -> dict[str, Any] | None:
    """Validate an injected expander's output; None → template rung."""
    if not isinstance(raw, Mapping):
        return None
    what = str(raw.get("what") or "").strip()
    why = str(raw.get("why") or "").strip()
    if not what or not why:
        return None
    shape = str(raw.get("shape") or "project").strip().lower()
    if shape not in SHAPES:
        shape = "project"
    unknowns = raw.get("unknowns") or []
    if isinstance(unknowns, str):
        unknowns = [unknowns]
    return {
        "what": what,
        "why": why,
        "takes": str(raw.get("takes") or "").strip() or "not scoped",
        "unknowns": [str(u).strip() for u in unknowns if str(u).strip()],
        "shape": shape,
        "basis": "brain",
        "draft_text": str(raw.get("draft_text") or "").strip(),
        "suggested_class": str(raw.get("suggested_class") or "").strip(),
    }


def make_brain_expand_fn(context: Any) -> Callable[..., dict[str, Any] | None]:
    """The brain-backed expander for the chat/tool surfaces.

    Asks the brain for a structured expansion and parses leniently;
    any failure returns None and the pipeline falls back to the
    template rung with the basis labeled — the weekly-content
    precedent.  Never raises.
    """
    def _expand(idea: Idea) -> dict[str, Any] | None:
        try:
            from ..llm.brain import brain_for

            prompt = (
                "Grow this raw idea into a shaped proposal. Reply with "
                "one JSON object only, keys: what (what it is, one or "
                "two sentences), why (who it is for and why it "
                "matters), takes (what doing it takes — effort, "
                "pieces, cost), unknowns (list of open questions), "
                "shape (one of project|content|task|someday), "
                "draft_text (only when shape is content: the post "
                "draft itself, else empty string).\n\n"
                "Idea title: " + idea.title + "\n"
                "Idea detail: " + (idea.description or "(none)") + "\n"
                "Tags: " + (", ".join(idea.tags) or "(none)"))
            resp = brain_for(context).complete(prompt,
                                               task_kind="creative")
            text = resp.text if hasattr(resp, "text") else str(resp or "")
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            parsed = json.loads(text[start:end + 1])
            return _normalize_expansion(parsed, idea)
        except Exception:  # noqa: BLE001 — fallback rung is the design
            return None

    return _expand


def make_research_fn(db: Any, registry: Any) -> Callable[..., list[Finding]] | None:
    """The production research seam over research.pipeline.run_job.

    Returns None when the registry cannot search the web — the run
    then records the honest unavailable statement instead of faking
    findings.  The job is bounded: findings capped by the run's
    budget, one page fetch.
    """
    try:
        getter = getattr(registry, "get", None)
        if registry is None or getter is None or getter("web_search") is None:
            return None
    except Exception:  # noqa: BLE001 — a registry that cannot answer
        return None

    def _research(idea: Idea, expansion: Mapping[str, Any],
                  budget: int) -> list[Finding]:
        from ..research.pipeline import (
            ResearchContext,
            ResearchJob,
            run_job,
        )

        job = ResearchJob(
            id="idea-dev-" + idea.idea_id,
            topic=idea.title,
            queries=[idea.title,
                     idea.title + " " + str(expansion.get("shape") or "")],
            max_results=max(1, int(budget or 5)),
            fetch_top=1)
        rctx = ResearchContext(db=db, registry=registry)
        findings = run_job(job, rctx)
        return [Finding(title=f.title, source=f.url, snippet=f.snippet)
                for f in findings]

    return _research


def compose_draft_text(idea: Idea, expansion: Mapping[str, Any]) -> str:
    """The draft body for the draft exit: the expander's own draft
    when it wrote one, else the shaped proposal composed plainly."""
    draft = str(expansion.get("draft_text") or "").strip()
    if draft:
        return draft
    parts = [idea.title, "", str(expansion.get("what") or "").strip(),
             "", str(expansion.get("why") or "").strip()]
    return "\n".join(parts).strip()


def resolve_idea(tracker: IdeaTracker, user_id: str,
                 ref: str) -> tuple[Idea | None, str]:
    """Resolve an idea reference: full id, ``#n`` ordinal (list
    order, newest first), or a unique title substring.

    Returns ``(idea, "")`` or ``(None, message)`` — an ambiguous or
    unknown ref is answered in words, never guessed.
    """
    ref = (ref or "").strip()
    if not ref:
        return None, "name the idea — a #number, its title, or its id"
    idea = run_sync(tracker.get(ref))
    if idea is not None and idea.user_id == user_id:
        return idea, ""
    ideas = run_sync(tracker.list_ideas(user_id, limit=100))
    ordinal = ref.lstrip("#")
    if ordinal.isdigit():
        n = int(ordinal)
        if 1 <= n <= len(ideas):
            return ideas[n - 1], ""
        return None, ("there is no idea #" + str(n) + " — the pile "
                      "has " + str(len(ideas)))
    needle = ref.lower()
    hits = [i for i in ideas if needle in i.title.lower()]
    if len(hits) == 1:
        return hits[0], ""
    if len(hits) > 1:
        names = "; ".join(
            "#" + str(ideas.index(h) + 1) + " " + h.title for h in hits[:5])
        return None, "that matches more than one idea: " + names
    return None, "no idea matches “" + ref + "”"


# ── the cell ──────────────────────────────────────────────────────────────


class IdeaDevelopment:
    """The development pipeline over one Database.

    Tracker coroutines are driven through ``goals.surface.run_sync``
    (the surface pattern); the delegation store, the record store and
    the exits are all synchronous.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_DDL)
        self.tracker = IdeaTracker(db)
        self.goals = GoalTracker(db)
        self.store = DelegationStore(db)

    # -- record store -------------------------------------------------

    def _save(self, record: dict[str, Any]) -> None:
        record["updated_at"] = time.time()
        self.db.execute(
            "INSERT OR REPLACE INTO idea_development (dev_id, idea_id,"
            " user_id, contract_id, status, record, revision,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (record["dev_id"], record["idea_id"], record["user_id"],
             record.get("contract_id", ""), record["status"],
             json.dumps(record), int(record.get("revision", 0)),
             record["created_at"], record["updated_at"]))

    def _row(self, row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        try:
            return json.loads(row["record"] or "{}")
        except (TypeError, ValueError):
            return None

    def record_for(self, dev_id: str) -> dict[str, Any] | None:
        return self._row(self.db.query_one(
            "SELECT * FROM idea_development WHERE dev_id = ?", (dev_id,)))

    def latest_for(self, idea_id: str) -> dict[str, Any] | None:
        return self._row(self.db.query_one(
            "SELECT * FROM idea_development WHERE idea_id = ?"
            " ORDER BY created_at DESC LIMIT 1", (idea_id,)))

    def developments_for(self, idea_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM idea_development WHERE idea_id = ?"
            " ORDER BY created_at DESC", (idea_id,))
        return [r for r in (self._row(row) for row in rows) if r]

    # -- the run --------------------------------------------------------

    def develop(
        self,
        idea_id: str,
        *,
        user_id: str = OWNER_USER_ID,
        exit_override: str = "",
        class_arg: str = "",
        trigger: str = "",
        expand_fn: Callable[..., Any] | None = None,
        research_fn: Callable[..., Any] | None = None,
        draft_queue: Any = None,
        mission_store: Any = None,
        budget: int = 5,
    ) -> dict[str, Any]:
        """Run one development end-to-end and return its record.

        Raises KeyError for an unknown idea and ValueError for an
        undischargeable request (dismissed/promoted idea, unknown
        exit or class).  Everything the run itself hits — a refused
        gate, a fenced artifact, a broken exit — lands in the record's
        outcome in words; the record is returned either way.
        """
        exit_name = EXIT_ALIASES.get(
            (exit_override or "").strip().lower(),
            (exit_override or "").strip().lower())
        if exit_name and exit_name not in EXITS:
            raise ValueError(
                "unknown exit " + repr(exit_override) + " — one of: "
                + ", ".join(EXITS))
        class_id_arg = (class_arg or "").strip().lower()
        if class_id_arg:
            try:
                class_id_arg = get_class(class_id_arg).id
            except GoalClassError as exc:
                raise ValueError(str(exc)) from exc
        idea = run_sync(self.tracker.get(idea_id))
        if idea is None or idea.user_id != user_id:
            raise KeyError("no idea " + repr(idea_id))
        if idea.status == "dismissed":
            raise ValueError(
                "that idea is dismissed — a dismissed idea is not "
                "developed")
        if idea.status == "promoted_to_goal":
            raise ValueError(
                "that idea already became a goal — development is "
                "done for it")

        now = time.time()
        record: dict[str, Any] = {
            "dev_id": "dev-" + uuid.uuid4().hex[:12],
            "idea_id": idea.idea_id,
            "user_id": user_id,
            "contract_id": "",
            "status": "running",
            "raw": {"title": idea.title, "description": idea.description},
            "expansion": {},
            "connections": {},
            "research": {"accepted": [], "refused": 0, "statement": ""},
            "ice": {},
            "verdict": {},
            "verifier": {"checks": [], "ok": False},
            "outcome": {"status": "", "detail": ""},
            "revision": 0,
            "created_at": now,
            "updated_at": now,
        }
        dev_ref = "idea_development:" + record["dev_id"]

        contract = self.store.create_contract(
            goal="develop the idea: " + idea.title,
            title="idea development — " + idea.title[:48],
            deliverable="a verified development record with one "
                        "executed exit",
            worker=WORKER, verifier=VERIFIER, skeptic=SKEPTIC,
            witness=WITNESS, integration_owner=STEWARD,
            recovery_supervisor=SUPERVISOR, arbiter=ARBITER,
            specialist_class="idea-development", consequential=True,
            required_evidence=("expansion", "connections", "research",
                               "verdict"),
            scope=("idea-dev:" + idea.idea_id,),
            step_budget=6)
        record["contract_id"] = contract.id
        cid = contract.id

        # The fenced artifact: this run owns the idea's development
        # record before it writes a word of it.
        artifact_key = "idea-dev:" + idea.idea_id
        try:
            owned = self.store.claim_artifact(cid, artifact_key)
            base_revision = int(owned["revision"])
            self._save(record)
            written = self.store.write_artifact(
                cid, artifact_key, base_revision=base_revision,
                payload_ref=dev_ref)
            record["revision"] = int(written["revision"])
        except DelegationError as exc:
            record["status"] = "failed"
            record["outcome"] = {
                "status": "failed",
                "detail": "the development record is fenced by "
                          "another run: " + str(exc)}
            self._save(record)
            self.store.report_failure(cid, by=STEWARD,
                                      detail=record["outcome"]["detail"])
            return record
        self._save(record)

        try:
            self._run_steps(record, idea, exit_name=exit_name,
                            class_id_arg=class_id_arg,
                            trigger=(trigger or "").strip(),
                            expand_fn=expand_fn, research_fn=research_fn,
                            draft_queue=draft_queue,
                            mission_store=mission_store,
                            budget=max(1, int(budget or 5)),
                            dev_ref=dev_ref)
        except Exception as exc:  # noqa: BLE001 — the record carries it
            record["status"] = "failed"
            record["outcome"] = {
                "status": "failed",
                "detail": "development broke mid-run: "
                          + type(exc).__name__ + ": " + str(exc)}
            self._save(record)
            try:
                self.store.report_failure(
                    cid, by=STEWARD, detail=record["outcome"]["detail"])
            except DelegationError:
                pass
        return record

    # -- steps ----------------------------------------------------------

    def _run_steps(self, record: dict[str, Any], idea: Idea, *,
                   exit_name: str, class_id_arg: str, trigger: str,
                   expand_fn: Any, research_fn: Any, draft_queue: Any,
                   mission_store: Any, budget: int, dev_ref: str) -> None:
        cid = record["contract_id"]
        user_id = record["user_id"]

        # EXPAND ------------------------------------------------------
        expansion: dict[str, Any] | None = None
        if expand_fn is not None:
            try:
                expansion = _normalize_expansion(expand_fn(idea), idea)
            except Exception:  # noqa: BLE001 — fallback rung
                expansion = None
        if expansion is None:
            expansion = template_expansion(idea)
        record["expansion"] = expansion
        self.store.submit_evidence(
            cid, kind="expansion", producer=WORKER,
            claim=("expansion (" + expansion["basis"] + "): "
                   + expansion["what"])[:400],
            evidence_ref=dev_ref + "#expansion")

        # CONNECT -----------------------------------------------------
        connections = self._connect(idea, expansion, class_id_arg)
        record["connections"] = connections
        bits = []
        if connections.get("goal_class"):
            bits.append("class " + connections["goal_class"]["title"])
        bits.append(str(len(connections["class_goals"])) + " class goals")
        bits.append(str(len(connections["kin_ideas"])) + " kin ideas")
        if connections.get("projects"):
            bits.append("projects: " + ", ".join(connections["projects"]))
        self.store.submit_evidence(
            cid, kind="connections", producer=WORKER,
            claim="connections — " + "; ".join(bits),
            evidence_ref=dev_ref + "#connections")

        # RESEARCH ----------------------------------------------------
        self._research(record, idea, expansion, research_fn, budget,
                       dev_ref)

        # ICE rescore ---------------------------------------------------
        record["ice"] = self._rescore(idea, connections, record["research"])

        # VERDICT -------------------------------------------------------
        verdict = self._verdict(idea, expansion, connections,
                                exit_name=exit_name,
                                class_id_arg=class_id_arg,
                                trigger=trigger)
        record["verdict"] = verdict
        verdict_claim = self.store.submit_evidence(
            cid, kind="verdict", producer=WORKER,
            claim=("verdict: " + verdict["exit"] + " — "
                   + " ".join(verdict["reasoning"]))[:400],
            evidence_ref=dev_ref + "#verdict")

        # SKEPTIC -------------------------------------------------------
        challenges = [self.store.challenge(
            cid, challenger=SKEPTIC,
            claim="the verdict's exit must be justified by the "
                  "recorded expansion, connections and research — an "
                  "exit the evidence does not support must not "
                  "execute",
            target_claim_id=verdict_claim.id)]
        if verdict["exit"] == "promote" and not connections["class_goals"] \
                and not class_id_arg:
            challenges.append(self.store.challenge(
                cid, challenger=SKEPTIC,
                claim="promotion files into class "
                      + verdict["class_id"] + " with no existing goals "
                      "there and no owner-named class — the class "
                      "connection is thin",
                target_claim_id=verdict_claim.id))

        # VERIFIER ------------------------------------------------------
        checks = self._verify(record, idea, connections)
        record["verifier"] = {
            "checks": checks,
            "ok": all(c["ok"] for c in checks),
        }
        scope_ok = {
            "expansion": _check(checks, "expansion_complete"),
            "connections": _check(checks, "connections_resolve"),
            "research": _check(checks, "research_provenance"),
            "verdict": _check(checks, "verdict_shape")
                       and _check(checks, "raw_unchanged"),
        }
        claims = self.store.claims(cid, limit=200)
        for entry in claims:
            if entry.kind == "evidence" and entry.scope in scope_ok:
                self.store.verify_claim(
                    entry.id, validator=VERIFIER,
                    verdict="supported" if scope_ok[entry.scope]
                    else "contradicted")
        verdict_now = self.store.get_claim(verdict_claim.id)
        verdict_supported = bool(verdict_now
                                 and verdict_now.verdict == "supported")
        for pos, challenge in enumerate(challenges):
            if pos == 0:
                self.store.resolve_challenge(
                    challenge.id, resolver=VERIFIER,
                    verdict="supported" if verdict_supported
                    else "contradicted",
                    note="standing challenge resolved by the verifier's "
                         "checks")
            else:
                self.store.resolve_challenge(
                    challenge.id, resolver=VERIFIER, verdict="partial",
                    note="the thin class connection stands as recorded "
                         "— the verdict survives, qualified")
        if record["verifier"]["ok"]:
            self.store.sign_off(
                cid, verifier=VERIFIER,
                note="expansion, connections, research and verdict "
                     "verified against the real stores")

        # COMPLETION GATE -------------------------------------------------
        decision = self.store.complete(cid, by=STEWARD)
        if not decision.ok:
            record["status"] = "failed"
            record["outcome"] = {
                "status": "failed",
                "detail": "the verification gate refused completion — "
                          "no exit executed: "
                          + "; ".join(decision.reasons)}
            self._save(record)
            self.store.report_failure(cid, by=STEWARD,
                                      detail=record["outcome"]["detail"])
            return

        # EXIT ------------------------------------------------------------
        outcome = self._execute_exit(record, idea, verdict, expansion,
                                     draft_queue=draft_queue,
                                     mission_store=mission_store)
        record["outcome"] = outcome
        record["status"] = ("completed" if outcome["status"] == "completed"
                             else "failed")
        self._save(record)

    def _connect(self, idea: Idea, expansion: Mapping[str, Any],
                 class_id_arg: str) -> dict[str, Any]:
        """Real lookups against the real stores — a connection is
        named only when the store returns it."""
        user_id = idea.user_id
        class_id = class_id_arg or str(
            expansion.get("suggested_class") or "")
        descriptor = None
        if class_id:
            try:
                descriptor = get_class(class_id)
            except GoalClassError:
                descriptor = None
                class_id = ""
        class_goals: list[dict[str, Any]] = []
        if descriptor is not None:
            goals = run_sync(self.goals.list_goals(
                user_id, status="active", goal_class=descriptor.id,
                limit=5))
            class_goals = [{"goal_id": g.goal_id, "title": g.title,
                            "status": g.status} for g in goals]
        kin = run_sync(self.tracker.find_similar_ideas(
            user_id, idea.idea_id, threshold=0.25, limit=5))
        kin_ideas = [{"idea_id": other.idea_id, "title": other.title,
                      "similarity": sim} for other, sim in kin]
        haystack = (idea.title + " " + (idea.description or "") + " "
                    + " ".join(idea.tags))
        projects = [name for name in KNOWN_PROJECTS
                    if re.search(r"\b" + re.escape(name) + r"\b",
                                 haystack, flags=re.IGNORECASE)]
        skills: Any = []
        try:
            from ..skills.filebook import FileSkillLibrary

            routed = FileSkillLibrary().route(
                idea.title + " " + (idea.description or ""), limit=3)
            skills = [{"id": doc.id, "title": doc.title}
                      for doc, _score in routed]
        except Exception as exc:  # noqa: BLE001 — named, not hidden
            skills = {"unavailable": type(exc).__name__ + ": " + str(exc)}
        return {
            "goal_class": (
                {"id": descriptor.id, "title": descriptor.title,
                 "progress_model": descriptor.progress_model}
                if descriptor is not None else None),
            "class_goals": class_goals,
            "kin_ideas": kin_ideas,
            "projects": projects,
            "skills": skills,
        }

    def _research(self, record: dict[str, Any], idea: Idea,
                  expansion: Mapping[str, Any], research_fn: Any,
                  budget: int, dev_ref: str) -> None:
        cid = record["contract_id"]
        accepted: list[dict[str, Any]] = []
        refused = 0
        statement = ""
        findings: list[Finding] = []
        if research_fn is None:
            statement = ("no research seam attached to this surface — "
                         "the research step is recorded as unavailable, "
                         "not faked")
        else:
            try:
                findings = list(research_fn(idea, expansion, budget) or [])
            except Exception as exc:  # noqa: BLE001 — named in the record
                statement = ("research step failed: "
                             + type(exc).__name__ + ": " + str(exc))
        for finding in findings[:budget]:
            source = str(getattr(finding, "source", "") or "").strip()
            title = str(getattr(finding, "title", "") or "").strip()
            if not source:
                refused += 1
                continue
            entry = self.store.submit_evidence(
                cid, kind="research", producer=RESEARCHER,
                claim=(title or "finding")[:400], evidence_ref=source)
            accepted.append({"claim_id": entry.id, "title": title,
                             "source": source})
        if not accepted and not statement:
            if refused:
                statement = ("all " + str(refused) + " finding(s) were "
                             "refused — none carried provenance")
            else:
                statement = "research returned no findings"
        if statement:
            entry = self.store.submit_evidence(
                cid, kind="research", producer=RESEARCHER,
                claim=statement, evidence_ref=dev_ref + "#research")
            accepted.append({"claim_id": entry.id, "title": statement,
                             "source": dev_ref + "#research"})
        record["research"] = {"accepted": accepted, "refused": refused,
                              "statement": statement}

    def _rescore(self, idea: Idea, connections: Mapping[str, Any],
                 research: Mapping[str, Any]) -> dict[str, Any]:
        """Named-rule ICE adjustments on an existing score.

        Development never invents a score for an unscored idea — the
        record says so instead.  Every adjustment names the rule that
        fired and the evidence behind it.
        """
        if idea.impact is None or idea.confidence is None \
                or idea.ease is None:
            return {"before": None, "after": None, "adjustments": [],
                    "note": "no ICE score on the raw idea — development "
                            "does not invent one"}
        before = {"impact": idea.impact, "confidence": idea.confidence,
                  "ease": idea.ease}
        after = dict(before)
        adjustments: list[str] = []
        if connections.get("kin_ideas"):
            after["confidence"] = min(10.0, after["confidence"] + 1)
            adjustments.append(
                "kin support: confidence +1 — kin ideas already in "
                "the pile")
        provenanced = [a for a in research.get("accepted", [])
                       if not str(a.get("source", "")).startswith(
                           "idea_development:")]
        if len(provenanced) >= 2:
            after["confidence"] = min(10.0, after["confidence"] + 1)
            adjustments.append(
                "provenanced research: confidence +1 — two or more "
                "findings with sources landed")
        if connections.get("class_goals"):
            after["impact"] = min(10.0, after["impact"] + 1)
            adjustments.append(
                "class traction: impact +1 — the idea's class already "
                "holds active goals")
        if adjustments:
            run_sync(self.tracker.score_idea(
                idea.idea_id, impact=after["impact"],
                confidence=after["confidence"], ease=after["ease"]))
        else:
            adjustments = []
        return {"before": before, "after": after,
                "adjustments": adjustments,
                "note": "" if adjustments else
                        "scored, and no adjustment rule fired"}

    def _verdict(self, idea: Idea, expansion: Mapping[str, Any],
                 connections: Mapping[str, Any], *, exit_name: str,
                 class_id_arg: str, trigger: str) -> dict[str, Any]:
        reasoning: list[str] = []
        if exit_name:
            chosen = exit_name
            reasoning.append("the owner named the exit: " + chosen)
        else:
            shape = str(expansion.get("shape") or "project")
            chosen = {"content": "draft", "task": "mission",
                      "project": "promote", "someday": "park"}[shape]
            reasoning.append(
                "shape " + shape + " routes to " + chosen
                + " by the standing verdict policy")
        verdict: dict[str, Any] = {"exit": chosen, "reasoning": reasoning,
                                   "class_id": "", "trigger": ""}
        if chosen == "promote":
            if class_id_arg:
                verdict["class_id"] = class_id_arg
                reasoning.append("class named by the owner: "
                                 + class_id_arg)
            elif connections.get("goal_class"):
                verdict["class_id"] = connections["goal_class"]["id"]
                reasoning.append(
                    "class from the connection step: "
                    + connections["goal_class"]["title"])
            else:
                verdict["class_id"] = "general"
                reasoning.append(
                    "no class connection found — files under general")
        if chosen == "park":
            verdict["trigger"] = trigger or DEFAULT_TRIGGER
            reasoning.append("revival trigger: " + verdict["trigger"])
        if chosen == "draft":
            reasoning.append(
                "the draft lands in the review queue — posting stays "
                "the owner's call there")
        if chosen == "mission":
            reasoning.append(
                "the mission persists for the runner to pick up — "
                "spawning it executes nothing by itself")
        return verdict

    def _verify(self, record: dict[str, Any], idea: Idea,
                connections: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The verifier's deterministic checks — run by the verifier
        role against the real stores, never by the worker."""
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str) -> None:
            checks.append({"name": name, "ok": bool(ok),
                           "detail": detail})

        fresh = run_sync(self.tracker.get(idea.idea_id))
        add("raw_unchanged",
            fresh is not None and fresh.title == record["raw"]["title"]
            and fresh.description == record["raw"]["description"],
            "the captured title and description are byte-identical "
            "to the run-start snapshot"
            if fresh is not None else "the idea vanished mid-run")
        expansion = record["expansion"]
        add("expansion_complete",
            bool(str(expansion.get("what") or "").strip())
            and bool(str(expansion.get("why") or "").strip())
            and bool(str(expansion.get("takes") or "").strip())
            and expansion.get("shape") in SHAPES,
            "what / why / takes are filled and the shape is one of "
            + ", ".join(SHAPES))
        claims = self.store.claims(record["contract_id"], limit=200)
        research_claims = [c for c in claims
                           if c.kind == "evidence"
                           and c.scope == "research"]
        add("research_provenance",
            bool(research_claims) and all(
                str(c.evidence_ref or "").strip() for c in research_claims),
            str(len(research_claims)) + " research claim(s) on the "
            "contract, every one carrying its provenance")
        problems: list[str] = []
        descriptor = (connections.get("goal_class") or {}).get("id")
        if descriptor:
            try:
                get_class(descriptor)
            except GoalClassError:
                problems.append("class " + str(descriptor) + " unknown")
        for goal in connections.get("class_goals", []):
            if run_sync(self.goals.get(goal["goal_id"])) is None:
                problems.append("goal “" + goal["title"] + "” not found")
        for kin in connections.get("kin_ideas", []):
            if run_sync(self.tracker.get(kin["idea_id"])) is None:
                problems.append("kin idea “" + kin["title"] + "” not found")
        add("connections_resolve", not problems,
            "every named connection resolves in its store"
            if not problems else "; ".join(problems))
        verdict = record["verdict"]
        ok = verdict.get("exit") in EXITS
        detail = "exit is one of the four census exits"
        if verdict.get("exit") == "park" and not verdict.get("trigger"):
            ok = False
            detail = "a parked verdict with no revival trigger"
        if verdict.get("exit") == "promote":
            try:
                get_class(verdict.get("class_id") or "general")
            except GoalClassError:
                ok = False
                detail = "promotion class is not in the registry"
        if verdict.get("exit") == "draft" and not compose_draft_text(
                idea, expansion):
            ok = False
            detail = "the draft exit has no draft text"
        add("verdict_shape", ok, detail)
        return checks

    def _execute_exit(self, record: dict[str, Any], idea: Idea,
                      verdict: Mapping[str, Any],
                      expansion: Mapping[str, Any], *,
                      draft_queue: Any,
                      mission_store: Any) -> dict[str, Any]:
        user_id = record["user_id"]
        chosen = verdict["exit"]
        try:
            if chosen == "promote":
                goal_id = run_sync(self.tracker.promote_to_goal(
                    idea.idea_id, self.goals,
                    goal_class=verdict["class_id"] or None))
                if goal_id is None:
                    return {"status": "failed",
                            "detail": "promotion found no idea to "
                                      "promote"}
                goal = run_sync(self.goals.get(goal_id))
                return {"status": "completed",
                        "detail": "promoted — goal “"
                                  + (goal.title if goal else idea.title)
                                  + "” filed in class "
                                  + verdict["class_id"],
                        "goal_id": goal_id}
            if chosen == "draft":
                if draft_queue is None:
                    return {"status": "failed",
                            "detail": "draft exit chosen but no draft "
                                      "queue is attached to this "
                                      "surface"}
                draft = draft_queue.create_draft(
                    compose_draft_text(idea, expansion),
                    platforms=["linkedin"],
                    metadata={"source": "idea-development",
                              "idea_id": idea.idea_id,
                              "dev_id": record["dev_id"]})
                draft_queue.propose(draft.id)
                landed = draft_queue.get(draft.id)
                status = landed.status if landed else ""
                if status != "pending_review":
                    return {"status": "failed",
                            "detail": "the draft did not land in "
                                      "review (status: "
                                      + str(status) + ")"}
                return {"status": "completed",
                        "detail": "draft queued for review — posting "
                                  "stays the owner's call in /drafts",
                        "draft_id": draft.id}
            if chosen == "mission":
                if mission_store is None:
                    return {"status": "failed",
                            "detail": "mission exit chosen but no "
                                      "mission store is attached to "
                                      "this surface"}
                mission = mission_store.create_new(
                    goal=idea.title + " — "
                         + str(expansion.get("what") or ""),
                    name=idea.title[:80],
                    state={"idea_id": idea.idea_id,
                           "development_id": record["dev_id"]},
                    metadata={"source": "idea-development"},
                    tags=["idea-development"])
                return {"status": "completed",
                        "detail": "mission spawned — “" + idea.title
                                  + "” waits in the mission store",
                        "mission_id": mission.id}
            # park
            return {"status": "completed",
                    "detail": "parked — the idea stays active; revival "
                              "trigger: " + str(verdict.get("trigger")
                                                or DEFAULT_TRIGGER)}
        except Exception as exc:  # noqa: BLE001 — named in the record
            return {"status": "failed",
                    "detail": "the " + chosen + " exit broke: "
                              + type(exc).__name__ + ": " + str(exc)}


def _check(checks: list[dict[str, Any]], name: str) -> bool:
    for entry in checks:
        if entry["name"] == name:
            return bool(entry["ok"])
    return False


# ── rendering ─────────────────────────────────────────────────────────────


def render_record(record: Mapping[str, Any]) -> str:
    """The development record as the owner reads it — end-to-end,
    titles and names only; internal ids never print."""
    raw = record.get("raw") or {}
    expansion = record.get("expansion") or {}
    connections = record.get("connections") or {}
    research = record.get("research") or {}
    ice = record.get("ice") or {}
    verdict = record.get("verdict") or {}
    verifier = record.get("verifier") or {}
    outcome = record.get("outcome") or {}
    lines = ["🧠 idea developed — " + str(raw.get("title") or "")]
    lines.append("raw (untouched): " + str(raw.get("title") or "")
                 + ((" — " + str(raw.get("description")))
                    if raw.get("description") else ""))
    if expansion:
        lines.append("")
        lines.append("expansion (" + str(expansion.get("basis") or "")
                     + " rung, shape: "
                     + str(expansion.get("shape") or "") + ")")
        lines.append("  what: " + str(expansion.get("what") or ""))
        lines.append("  why: " + str(expansion.get("why") or ""))
        lines.append("  takes: " + str(expansion.get("takes") or ""))
        unknowns = expansion.get("unknowns") or []
        if unknowns:
            lines.append("  unknowns: " + "; ".join(
                str(u) for u in unknowns))
    lines.append("")
    lines.append("connections")
    descriptor = connections.get("goal_class")
    if descriptor:
        lines.append("  class: " + str(descriptor.get("title"))
                     + " (progress: "
                     + str(descriptor.get("progress_model")) + ")")
    else:
        lines.append("  class: none found")
    for goal in connections.get("class_goals") or []:
        lines.append("  goal: " + str(goal.get("title"))
                     + " (" + str(goal.get("status")) + ")")
    for kin in connections.get("kin_ideas") or []:
        lines.append("  kin idea: " + str(kin.get("title")))
    if connections.get("projects"):
        lines.append("  projects: "
                     + ", ".join(connections["projects"]))
    skills = connections.get("skills")
    if isinstance(skills, Mapping):
        lines.append("  skills: lookup unavailable — "
                     + str(skills.get("unavailable")))
    elif skills:
        lines.append("  skills: " + ", ".join(
            str(s.get("title")) for s in skills))
    lines.append("")
    lines.append("research")
    for hit in research.get("accepted") or []:
        source = str(hit.get("source") or "")
        if source.startswith("idea_development:"):
            lines.append("  " + str(hit.get("title") or ""))
        else:
            lines.append("  " + str(hit.get("title") or "finding")
                         + " — " + source)
    if research.get("refused"):
        lines.append("  refused: " + str(research["refused"])
                     + " finding(s) with no provenance — never landed")
    if not research.get("accepted") and not research.get("statement"):
        lines.append("  (nothing recorded)")
    if ice:
        lines.append("")
        if ice.get("before"):
            lines.append(
                "ICE: " + _ice_text(ice["before"]) + " → "
                + _ice_text(ice.get("after") or {}))
            for adj in ice.get("adjustments") or []:
                lines.append("  " + str(adj))
        if ice.get("note"):
            lines.append("ICE: " + str(ice["note"]))
    if verdict:
        lines.append("")
        lines.append("verdict: " + str(verdict.get("exit") or ""))
        for reason in verdict.get("reasoning") or []:
            lines.append("  " + str(reason))
    checks = verifier.get("checks") or []
    if checks:
        lines.append("")
        lines.append("verifier (independent of the developer)")
        for entry in checks:
            mark = "✓" if entry.get("ok") else "✗"
            lines.append("  " + mark + " " + str(entry.get("name"))
                         + " — " + str(entry.get("detail")))
    if outcome:
        lines.append("")
        lines.append("outcome: " + str(outcome.get("status") or "")
                     + " — " + str(outcome.get("detail") or ""))
    return "\n".join(lines)


def _ice_text(scores: Mapping[str, Any]) -> str:
    return ("I " + str(scores.get("impact")) + " / C "
            + str(scores.get("confidence")) + " / E "
            + str(scores.get("ease")))


# ── brain tool ────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    """The brain's ``idea`` tool — capture by narration.

    Plain owner chat reaches capture through the brain's own tool
    choice (dynamic understanding — the same seam the ``goal`` tool
    uses), never a keyword router.  ``develop`` runs the full cell
    with the production rungs attached.
    """
    context = registry.context

    @registry.register(
        "idea",
        description=(
            "Raw ideas: capture an idea in the owner's words (they "
            "are stored verbatim and never rewritten), list the pile, "
            "show one with its development record, or develop one — "
            "development grows it through expansion, connection to "
            "goal classes/goals/kin ideas, bounded provenanced "
            "research, and a verified verdict that promotes it to a "
            "goal in its class, queues a content draft for review, "
            "spawns a mission, or parks it with a revival trigger."
        ),
        capability="memory.write",
        parameters={
            "action": "str — capture|list|show|develop",
            "text": "str — the idea in the owner's words (capture)",
            "ref": "str — idea #number, title words, or id "
                   "(show/develop)",
            "exit": "str — develop only: promote|draft|mission|park "
                    "(empty = the pipeline's verdict policy decides)",
            "goal_class": "str — develop only: WS2 class id to file "
                          "a promotion under",
            "trigger": "str — develop only, park exit: the revival "
                       "trigger in words",
        },
    )
    def idea(action: str, *, text: str = "", ref: str = "",
             exit: str = "", goal_class: str = "",
             trigger: str = "") -> dict[str, Any]:
        db = getattr(context, "db", None)
        if db is None:
            return {"ok": False, "error": "no database attached"}
        cell = IdeaDevelopment(db)
        tracker = cell.tracker
        action = (action or "list").strip().lower()
        if action == "capture":
            raw = (text or "").strip()
            if not raw:
                return {"ok": False, "error": "the idea text is empty"}
            if "\n" in raw:
                title, _, rest = raw.partition("\n")
                description = rest.strip()
            else:
                title, description = raw, ""
            created = run_sync(tracker.create(
                OWNER_USER_ID, title.strip(),
                description=description, source="chat"))
            return {"ok": True, "idea_id": created.idea_id,
                    "title": created.title, "status": created.status}
        if action == "list":
            ideas = run_sync(tracker.list_ideas(OWNER_USER_ID, limit=20))
            return {"ok": True, "ideas": [
                {"idea_id": i.idea_id, "title": i.title,
                 "status": i.status} for i in ideas]}
        if action in ("show", "develop"):
            found, problem = resolve_idea(tracker, OWNER_USER_ID, ref)
            if found is None:
                return {"ok": False, "error": problem}
            if action == "show":
                record = cell.latest_for(found.idea_id)
                return {"ok": True, "idea_id": found.idea_id,
                        "title": found.title, "status": found.status,
                        "record": render_record(record) if record
                        else "not developed yet"}
            from ..missions.mission import MissionStore
            from ..social.drafts import DraftQueue

            try:
                record = cell.develop(
                    found.idea_id, exit_override=exit,
                    class_arg=goal_class, trigger=trigger,
                    expand_fn=make_brain_expand_fn(context),
                    research_fn=make_research_fn(
                        db, getattr(context, "tools", None)),
                    draft_queue=DraftQueue(),
                    mission_store=MissionStore(db))
            except (KeyError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": record["status"] == "completed",
                    "idea_id": found.idea_id,
                    "exit": (record.get("verdict") or {}).get("exit", ""),
                    "record": render_record(record)}
        return {"ok": False, "error": "unknown action " + repr(action)}
