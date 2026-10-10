"""Failure analysis agent — study failures, extract lessons, prevent repeats.

The system already records its failures in several places:
  * ``coding_log``   — every failed write->run->fix attempt (with the error)
  * ``devon_memory`` — investigations that ended ``failed``
  * benchmark runs   — dimensions/tasks that scored below the bar
  * test runs        — failing suites (the evolution gate captures these)

The ``FailureAnalyzer`` turns that raw failure data into durable, reusable
*lessons*: a classification, a root cause, a concrete lesson, a fix, and a
*prevention* one-liner.  Each lesson is also captured as a ``prevention``
skill so any agent can recall "how to avoid this mistake" before making it.

``prevention_context(query)`` returns the lessons most relevant to a new
task — inject that into a planner prompt and the same mistake stops
happening.  ``learn_from_failure`` is called automatically after a failure
so the loop is continuous and self-correcting.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.brain import brain_for
from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .skills import SkillLibrary

_log = get_logger(__name__)

__all__ = ["FailureAnalyzer", "FailureCase", "Lesson",
           "failure_fingerprint", "enrich_with_lessons", "register"]


def enrich_with_lessons(context: Any, query: str, *,
                        limit: int = 4) -> str:
    """Single choke point for lesson injection into planner prompts.

    EVERY planning call should route through here instead of calling
    ``FailureAnalyzer(...).prevention_context(...)`` directly, so lessons
    are injected automatically without each call site remembering to do it.
    Never raises: lesson context is a bonus, never fatal.
    """
    try:
        return FailureAnalyzer(context).prevention_context(query, limit=limit)
    except Exception:  # noqa: BLE001
        return ""


@dataclass
class FailureCase:
    source: str            # coding|devon|benchmark|test|tool
    summary: str
    error: str = ""
    detail: str = ""
    ts: float = 0.0
    skill: str = ""        # skill/tool attribution, when known
    operation: str = ""    # the operation that failed, when known


# ── failure fingerprinting ──────────────────────────────────────────────────
# A stable, deterministic fingerprint (normalized error class + skill +
# operation) so repeat failures match lessons exactly instead of by fuzzy
# text. Numbers become "#", quoted literals become "?", so "timeout after
# 30s" and "timeout after 120s" fingerprint identically.

_FINGERPRINT_NOISE = re.compile(r"\d+")
_FINGERPRINT_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def failure_fingerprint(source: str, error: str, operation: str = "",
                        skill: str = "") -> str:
    """Deterministic fingerprint for a failure: source|category|skill|
    operation|normalized-first-line."""
    first = next((l.strip() for l in (error or "").splitlines() if l.strip()),
                 "")[:200].lower()
    first = _FINGERPRINT_QUOTED.sub("?", first)
    first = _FINGERPRINT_NOISE.sub("#", first)
    first = re.sub(r"\s+", " ", first).strip()[:120]
    return "|".join([
        (source or "").strip().lower()[:40],
        categorize(error),
        (skill or "").strip().lower()[:60],
        (operation or "").strip().lower()[:60],
        first,
    ])


@dataclass
class Lesson:
    id: str
    source: str
    category: str
    root_cause: str
    lesson: str
    fix: str
    prevention: str
    skill_id: str = ""
    times_seen: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    # ── lesson memory v2: usefulness scoring ──
    times_surfaced: int = 0
    times_prevented: int = 0
    fingerprint: str = ""
    demoted: bool = False

    @property
    def usefulness(self) -> float:
        """Fraction of surfacings after which the failure did NOT recur."""
        if not self.times_surfaced:
            return 0.0
        return self.times_prevented / self.times_surfaced

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "source": self.source, "category": self.category,
            "root_cause": self.root_cause, "lesson": self.lesson,
            "fix": self.fix, "prevention": self.prevention,
            "skill_id": self.skill_id, "times_seen": self.times_seen,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "times_surfaced": self.times_surfaced,
            "times_prevented": self.times_prevented,
            "usefulness": round(self.usefulness, 3),
            "fingerprint": self.fingerprint, "demoted": self.demoted,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Lesson":
        return cls(
            id=row["id"], source=row.get("source", ""),
            category=row.get("category", ""), root_cause=row.get("root_cause", ""),
            lesson=row.get("lesson", ""), fix=row.get("fix", ""),
            prevention=row.get("prevention", ""), skill_id=row.get("skill_id", ""),
            times_seen=int(row.get("times_seen", 1)),
            created_at=float(row.get("created_at", 0)),
            updated_at=float(row.get("updated_at", 0)),
            times_surfaced=int(row.get("times_surfaced", 0)),
            times_prevented=int(row.get("times_prevented", 0)),
            fingerprint=row.get("fingerprint", "") or "",
            demoted=bool(row.get("demoted", 0)),
        )


# Map a raw error string to a coarse category — deterministic, no model.
_CATEGORIES: list[tuple[str, str]] = [
    ("NameError|not defined", "undefined_name"),
    ("IndexError|index out of range", "bad_index"),
    ("ZeroDivisionError|division by zero", "division_by_zero"),
    ("KeyError", "missing_key"),
    ("TypeError", "type_mismatch"),
    ("AttributeError", "bad_attribute"),
    ("FileNotFound|No such file", "missing_file"),
    ("Connection|timeout|timed out|refused", "network"),
    ("assert|AssertionError", "assertion"),
    ("SyntaxError", "syntax"),
    ("ModuleNotFound|ImportError", "import"),
    ("permission|denied|401|403", "permission"),
    ("no such tool|unknown tool", "tool_selection"),
    ("timeout|wall-clock|budget", "budget"),
]


def categorize(error: str) -> str:
    low = (error or "").lower()
    # re.I: the patterns use exact Python exception casing
    # ("NameError", "KeyError") but the input is lowercased
    for pattern, category in _CATEGORIES:
        if re.search(pattern, low, re.I):
            return category
    return "other"


class FailureAnalyzer:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = context.db
        self.skills = SkillLibrary(context.db)

    # ── collect ─────────────────────────────────────────────────────────────
    def collect(self, *, limit: int = 20) -> list[FailureCase]:
        """Gather recent failures from every source the system tracks."""
        cases: list[FailureCase] = []
        # coding agent failures
        try:
            rows = self.db.query(
                "SELECT task, stderr, stdout, attempt, created_at FROM "
                "coding_log WHERE exit_code != 0 ORDER BY created_at DESC "
                "LIMIT ?", (limit,))
            for r in rows:
                cases.append(FailureCase(
                    source="coding", summary=(r["task"] or "")[:200],
                    error=(r["stderr"] or r["stdout"] or "")[-1500:],
                    ts=float(r["created_at"] or 0)))
        except Exception:  # noqa: BLE001 — no table yet, on
            pass
        # devon failed investigations
        try:
            rows = self.db.query(
                "SELECT task, digest, ts FROM devon_memory WHERE status="
                "'failed' ORDER BY ts DESC LIMIT ?", (limit,))
            for r in rows:
                cases.append(FailureCase(
                    source="devon", summary=(r["task"] or "")[:200],
                    error=(r["digest"] or "")[:1500], ts=float(r["ts"] or 0)))
        except Exception:  # noqa: BLE001
            pass
        return cases

    # ── record: the high-frequency, always-on entry point ───────────────────
    def record(self, source: str, summary: str, error: str, *,
               learn: bool = True, skill: str = "",
               operation: str = "") -> dict[str, Any]:
        """Record one failure the system actually hit (a tool call, a
        mission step, an evolution revert, an execbox red).

        Persists it to the ``failures`` ledger and — when the error maps to
        a known family, or when the same summary has failed before —
        distills/updates a durable lesson.  ``learn=False`` only logs.
        ``skill``/``operation`` attribute the failure for the skill
        self-rewrite loop and the failure fingerprint.  Never raises: a
        bookkeeping hiccup must never mask the original failure.  Returns
        ``{"id", "family", "lesson_id", "times_seen"}``.
        """
        error = (error or "")[:2000]
        family = categorize(error)
        summary = (summary or "")[:200]
        skill = (skill or "")[:80]
        operation = (operation or "")[:80]
        fingerprint = failure_fingerprint(source, error, operation, skill)
        try:
            fid = new_short_id("fail")
            self.db.execute(
                "INSERT INTO failures (id, source, summary, error, family, "
                "lesson, ts, fingerprint, skill) VALUES (?,?,?,?,?,?,?,?,?)",
                (fid, source, summary, error, family, "", time.time(),
                 fingerprint, skill))
            times_seen = 1
            lesson_id = ""
            if learn:
                repeat = self.db.query_one(
                    "SELECT COUNT(*) AS n FROM failures WHERE source=? AND "
                    "summary=? AND id<>?", (source, summary, fid))["n"]
                if family != "other" or repeat:
                    case = FailureCase(source=source, summary=summary,
                                       error=error, ts=time.time(),
                                       skill=skill, operation=operation)
                    lesson = self.learn_from_failure(case,
                                                     analysis=self._deterministic_analyze(case))
                    lesson_id = lesson.id
                    times_seen = lesson.times_seen
            return {"id": fid, "family": family, "lesson_id": lesson_id,
                    "times_seen": times_seen, "learned": bool(lesson_id),
                    "fingerprint": fingerprint}
        except Exception as exc:  # noqa: BLE001
            _log.debug("failure record failed: %s", exc)
            return {"id": "", "family": family, "lesson_id": "",
                    "times_seen": 0, "learned": False, "error": str(exc)}

    def recent_failures(self, *, limit: int = 10,
                        family: str = "") -> list[dict[str, Any]]:
        q = ("SELECT * FROM failures"
             + (" WHERE family=?" if family else "")
             + " ORDER BY ts DESC LIMIT ?")
        args = (family, limit) if family else (limit,)
        return [dict(r) for r in self.db.query(q, args)]

    # ── analyze (model) ─────────────────────────────────────────────────────
    def analyze(self, case: FailureCase) -> dict[str, str]:
        """Ask the model to classify a failure and extract a lesson.
        Falls back to a deterministic, model-free extraction when no usable
        model reply comes back, so failure learning never depends on a
        live provider."""
        model = self._model_analyze(case)
        if model:
            return model
        return self._deterministic_analyze(case)

    def _model_analyze(self, case: FailureCase) -> dict[str, str] | None:
        router = getattr(self.context, "router", None)
        if router is None:
            return None
        from ..llm.base import Message, SamplingParams

        prompt = (
            "You are a failure-analysis agent. Given a failure, extract "
            "durable, specific lessons.\n\n"
            f"Source: {case.source}\nTask: {case.summary}\n\n"
            f"ERROR OUTPUT:\n{case.error[:2500]}\n\n"
            'Respond with JSON ONLY: {"category": "<short_snake_case>", '
            '"root_cause": "<one sentence: what actually went wrong>", '
            '"lesson": "<one sentence: the general principle>", '
            '"fix": "<one sentence: the concrete fix>", '
            '"prevention": "<one short imperative: how to avoid this in '
            'the future>"}')
        try:
            resp = brain_for(self.context).chat([Message.user(prompt)],
                               SamplingParams(temperature=0.1, max_tokens=400), task_kind="judge")
            if not getattr(resp, "ok", False):
                return None
            data = json.loads(_first_json(resp.text or ""))
            if not isinstance(data, dict) or not data.get("lesson"):
                return None
            return {
                "category": str(data.get("category") or categorize(case.error))[:40],
                "root_cause": str(data.get("root_cause") or "")[:300],
                "lesson": str(data.get("lesson") or "")[:300],
                "fix": str(data.get("fix") or "")[:300],
                "prevention": str(data.get("prevention") or "")[:200],
            }
        except Exception:  # noqa: BLE001 — degrade, never crash the learner
            _log.debug("model failure analysis failed; using deterministic")
            return None

    def _deterministic_analyze(self, case: FailureCase) -> dict[str, str]:
        category = categorize(case.error)
        first_line = next((l.strip() for l in (case.error or "").splitlines()
                           if l.strip()), "")
        return {
            "category": category,
            "root_cause": first_line[:280] or f"{case.source} failure",
            "lesson": f"{case.source} run failed with a {category} error",
            "fix": f"address the {category} failure in {case.summary[:80]}",
            "prevention": f"guard against {category} errors before running",
        }

    # ── learn (persist + prevent) ───────────────────────────────────────────
    def learn_from_failure(self, case: FailureCase, *,
                           analysis: dict[str, str] | None = None) -> Lesson:
        """Analyze a failure and persist it as a durable lesson + a
        prevention skill. Deduplicates by (source, category, root_cause):
        a repeat failure increments times_seen instead of duplicating."""
        analysis = analysis or self.analyze(case)
        now = time.time()
        category = analysis.get("category") or categorize(case.error)
        root_cause = (analysis.get("root_cause") or "").strip()
        lesson = (analysis.get("lesson") or "").strip()
        prevention = (analysis.get("prevention") or "").strip()
        fingerprint = failure_fingerprint(case.source, case.error,
                                          case.operation, case.skill)
        existing = self.db.query_one(
            "SELECT * FROM lessons WHERE source=? AND category=? AND "
            "root_cause=?", (case.source, category, root_cause[:280]))
        if existing:
            self.db.execute(
                "UPDATE lessons SET times_seen=times_seen+1, updated_at=?, "
                "fingerprint=? WHERE id=?", (now, fingerprint, existing["id"]))
            l = Lesson.from_row(existing)
            l.times_seen += 1
            l.fingerprint = fingerprint
            return l
        lesson_id = new_short_id("lesson")
        # capture a prevention skill so any agent can recall the avoidance
        skill = self.skills.save(
            f"avoid {category}: {root_cause[:60]}",
            kind="prevention",
            body=f"Prevention: {prevention}\nLesson: {lesson}\n"
                 f"Fix: {analysis.get('fix', '')}",
            description=f"How to avoid a {category} failure",
            tags=[category, case.source], source="failure_analysis")
        self.db.execute(
            "INSERT INTO lessons (id, source, category, root_cause, lesson, "
            "fix, prevention, skill_id, times_seen, created_at, updated_at, "
            "fingerprint) "
            "VALUES (?,?,?,?,?,?,?,?,1,?,?,?)",
            (lesson_id, case.source, category, root_cause[:280], lesson[:300],
             analysis.get("fix", "")[:300], prevention[:200], skill.id, now,
             now, fingerprint))
        _log.info("learned lesson [%s/%s] (skill %s)", case.source, category,
                  skill.id)
        return Lesson(id=lesson_id, source=case.source, category=category,
                      root_cause=root_cause, lesson=lesson,
                      fix=analysis.get("fix", ""), prevention=prevention,
                      skill_id=skill.id, created_at=now, updated_at=now,
                      fingerprint=fingerprint)

    def learn_from_all(self, *, limit: int = 20) -> int:
        """Learn from every recent, not-yet-learned failure. Returns count."""
        seen = {r["id"] for r in self.db.query(
            "SELECT root_cause||source||category AS id FROM lessons")}
        learned = 0
        for case in self.collect(limit=limit):
            key = f"{case.source}{categorize(case.error)}"
            if key in seen:
                continue
            self.learn_from_failure(case)
            seen.add(key)
            learned += 1
        return learned

    def rank(self, query: str, *, limit: int = 4) -> list[Lesson]:
        """Lessons ranked by relevance to a query. Demoted lessons are
        penalized (surfaced less often, not never)."""
        try:
            rows = self.db.query(
                "SELECT * FROM lessons ORDER BY times_seen DESC, "
                "updated_at DESC LIMIT 100")
        except Exception:  # noqa: BLE001
            return []
        if not rows:
            return []
        q = (query or "").lower()
        qtok = set(re.findall(r"[a-z0-9_]{3,}", q))
        scored: list[tuple[Lesson, float]] = []
        for r in rows:
            l = Lesson.from_row(r)
            ltok = set(re.findall(r"[a-z0-9_]{3,}",
                                  (l.lesson + " " + l.category + " " +
                                   l.root_cause).lower()))
            overlap = len(qtok & ltok) / len(qtok) if qtok else 0.0
            score = (l.times_seen / 2.0) + overlap
            if l.demoted:
                score -= 5.0  # demoted: surfaced less often, not never
            scored.append((l, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [l for l, _ in scored[:limit]]

    # ── recall prevention ───────────────────────────────────────────────────
    def prevention_context(self, query: str, *, limit: int = 4) -> str:
        """Lessons most relevant to a new task, as a 'avoid these known
        failures' block. An empty string when there are no lessons.

        Every surfaced lesson is counted (``times_surfaced``) and logged in
        the surfacing ledger so the usefulness scorer can later tell whether
        the lesson actually prevented a repeat failure. Demoted lessons are
        surfaced only when nothing better matches.
        """
        top = self.rank(query, limit=limit)
        if not top:
            return ""
        self._record_surfacings(top)
        lines = [f"Known failure patterns to avoid for '{query[:60]}':"]
        for l in top:
            lines.append(f"  - [{l.category}] {l.prevention or l.lesson}"
                         f"  (seen {l.times_seen}x)")
        return "\n".join(lines)

    def _record_surfacings(self, lessons: list[Lesson]) -> None:
        """Count a surfacing for each lesson and log it for later
        usefulness evaluation. Never raises."""
        try:
            now = time.time()
            for l in lessons:
                self.db.execute(
                    "UPDATE lessons SET times_surfaced=times_surfaced+1 "
                    "WHERE id=?", (l.id,))
                self.db.execute(
                    "INSERT INTO lesson_surfacings (id, lesson_id, "
                    "fingerprint, surfaced_at, evaluated) VALUES (?,?,?,?,0)",
                    (new_short_id("surf"), l.id, l.fingerprint, now))
        except Exception:  # noqa: BLE001 — surfacing is bookkeeping
            _log.debug("surfacing record failed", exc_info=True)

    # ── lesson memory v2: usefulness scoring ──────────────────────────────────
    def evaluate_usefulness(self, *, window_days: float = 7.0,
                            floor: float = 0.2) -> dict[str, Any]:
        """Score every surfaced lesson: a surfacing counts as *prevented*
        when no failure with the lesson's fingerprint recurred within the
        window after surfacing.

        Demotes lessons below ``floor`` usefulness after >= 10 surfacings;
        archives (not deletes) lessons at 0 usefulness after >= 20
        surfacings. Returns counts.
        """
        now = time.time()
        cutoff = now - window_days * 86400.0
        evaluated = 0
        try:
            pending = self.db.query(
                "SELECT * FROM lesson_surfacings WHERE evaluated=0 AND "
                "surfaced_at < ?", (cutoff,))
        except Exception:  # noqa: BLE001 — pre-migration-52 databases
            return {"evaluated": 0, "demoted": 0, "archived": 0}
        for s in pending:
            try:
                recurred = self.db.query_one(
                    "SELECT 1 FROM failures WHERE fingerprint=? AND ts > ? "
                    "LIMIT 1", (s["fingerprint"], s["surfaced_at"]))
                if not recurred and s["fingerprint"]:
                    self.db.execute(
                        "UPDATE lessons SET times_prevented=times_prevented+1 "
                        "WHERE id=?", (s["lesson_id"],))
                self.db.execute(
                    "UPDATE lesson_surfacings SET evaluated=1 WHERE id=?",
                    (s["id"],))
                evaluated += 1
            except Exception:  # noqa: BLE001 — keep evaluating the rest
                continue
        demoted = 0
        archived = 0
        try:
            for r in self.db.query(
                    "SELECT * FROM lessons WHERE times_surfaced >= 10"):
                l = Lesson.from_row(r)
                u = l.usefulness
                if l.times_surfaced >= 20 and u <= 0.0 and not l.demoted:
                    self._archive_lesson(l, reason="usefulness 0 after 20+ surfacings")
                    archived += 1
                elif u < floor and not l.demoted:
                    self.db.execute(
                        "UPDATE lessons SET demoted=1 WHERE id=?", (l.id,))
                    demoted += 1
        except Exception:  # noqa: BLE001
            pass
        return {"evaluated": evaluated, "demoted": demoted,
                "archived": archived}

    def _archive_lesson(self, lesson: Lesson, *, reason: str) -> None:
        """Move a useless lesson to the archive table (never deleted
        outright)."""
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO lessons_archive (id, source, category, "
                "root_cause, lesson, fix, prevention, skill_id, times_seen, "
                "times_surfaced, times_prevented, fingerprint, archived_at, "
                "archive_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (lesson.id, lesson.source, lesson.category,
                 lesson.root_cause, lesson.lesson, lesson.fix,
                 lesson.prevention, lesson.skill_id, lesson.times_seen,
                 lesson.times_surfaced, lesson.times_prevented,
                 lesson.fingerprint, time.time(), reason))
            self.db.execute("DELETE FROM lessons WHERE id=?", (lesson.id,))
            _log.info("archived lesson %s (%s)", lesson.id, reason)
        except Exception:  # noqa: BLE001
            _log.debug("lesson archive failed", exc_info=True)

    def promotion_candidates(self, *, min_usefulness: float = 0.8,
                             min_surfaced: int = 15) -> list[Lesson]:
        """Lessons proven useful enough to fold into their skill instead of
        repeating as context forever."""
        out = []
        try:
            rows = self.db.query(
                "SELECT * FROM lessons WHERE demoted=0 AND skill_id<>'' AND "
                "times_surfaced >= ?", (min_surfaced,))
        except Exception:  # noqa: BLE001
            return []
        for r in rows:
            l = Lesson.from_row(r)
            if l.usefulness >= min_usefulness:
                out.append(l)
        return out

    def recent(self, *, limit: int = 10) -> list[Lesson]:
        rows = self.db.query("SELECT * FROM lessons ORDER BY updated_at DESC "
                             "LIMIT ?", (limit,))
        return [Lesson.from_row(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        total = self.db.query_one("SELECT COUNT(*) AS n FROM lessons")["n"]
        by_cat = {r["category"]: r["n"] for r in self.db.query(
            "SELECT category, COUNT(*) AS n FROM lessons GROUP BY category")}
        out: dict[str, Any] = {"total": total, "by_category": by_cat}
        # ── lesson memory v2: usefulness overview ──
        try:
            v2 = self.db.query_one(
                "SELECT COUNT(*) AS n, COALESCE(SUM(times_surfaced),0) AS s, "
                "COALESCE(SUM(times_prevented),0) AS p, "
                "COALESCE(SUM(demoted),0) AS d FROM lessons")
            surfaced = v2["s"] or 0
            out["usefulness"] = {
                "lessons": v2["n"],
                "surfaced": surfaced,
                "prevented": v2["p"] or 0,
                "avg_usefulness": round((v2["p"] or 0) / surfaced, 3)
                if surfaced else 0.0,
                "demoted": v2["d"] or 0,
            }
            arch = self.db.query_one(
                "SELECT COUNT(*) AS n FROM lessons_archive")["n"]
            out["usefulness"]["archived"] = arch
        except Exception:  # noqa: BLE001 - pre-migration-52 databases
            pass
        # the raw ledger: what the system actually hit, by family/source
        try:
            hits = self.db.query_one(
                "SELECT COUNT(*) AS n FROM failures")["n"]
            by_family = {r["family"]: r["n"] for r in self.db.query(
                "SELECT family, COUNT(*) AS n FROM failures "
                "GROUP BY family ORDER BY n DESC")}
            by_source = {r["source"]: r["n"] for r in self.db.query(
                "SELECT source, COUNT(*) AS n FROM failures "
                "GROUP BY source ORDER BY n DESC")}
            recent = self.db.query_one(
                "SELECT COUNT(*) AS n FROM failures WHERE ts > ?",
                (time.time() - 86400.0,))["n"]
            out.update({"failures": hits, "by_family": by_family,
                        "by_source": by_source, "failures_24h": recent})
        except Exception:  # noqa: BLE001 - pre-migration-27 databases
            pass
        return out


def _first_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


# ── registry ────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "failure_analyze",
        description=(
            "Failure analysis: study failures and extract durable lessons. "
            "action=collect (recent failures) | record (log a failure the "
            "system just hit, auto-learn a lesson for known families) | "
            "ledger (recent raw failures) | learn (learn from all recent "
            "failures) | analyze (analyze one failure inline) | context "
            "(prevention context for a task) | recent (recent lessons) | stats."
        ),
        capability="memory.write",
        parameters={
            "action": "str — collect|record|ledger|learn|analyze|context|recent|stats",
            "query": "str — the task/summary, for context and record",
            "error": "str — for 'analyze'/'record' inline",
            "source": "str — coding|devon|benchmark|test|tool|mission|evolution",
            "family": "str — filter for 'ledger'",
            "limit": "int",
        },
    )
    def failure_analyze(
        action: str, *, query: str = "", error: str = "", source: str = "",
        family: str = "", limit: str = "10",
    ) -> dict[str, Any]:
        analyzer = FailureAnalyzer(context)
        action = (action or "collect").strip().lower()
        try:
            n = int(limit or 10)
        except ValueError:
            n = 10
        if action == "collect":
            return {"failures": [vars(c) for c in analyzer.collect(limit=n)]}
        if action == "record":
            return {"recorded": analyzer.record(
                source or "tool", query, error)}
        if action == "ledger":
            return {"failures": analyzer.recent_failures(limit=n,
                                                         family=family)}
        if action == "learn":
            count = analyzer.learn_from_all(limit=n)
            return {"learned": count,
                    "recent": [l.to_dict() for l in analyzer.recent(limit=10)]}
        if action == "analyze":
            case = FailureCase(source=source or "tool",
                               summary=query, error=error)
            return {"analysis": analyzer.analyze(case)}
        if action == "context":
            return {"context": analyzer.prevention_context(query, limit=n)}
        if action == "stats":
            return analyzer.stats()
        return {"lessons": [l.to_dict() for l in analyzer.recent(limit=n)]}
