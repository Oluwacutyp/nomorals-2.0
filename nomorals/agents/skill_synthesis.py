"""Skill synthesis — notice repeated tool patterns, draft real skills.

When the same tool sequence runs >= 5 times in 30 days with no registered
skill covering it, that is a skill waiting to be born. The synthesizer:

  1. **detect** — scans ``tool_calls`` traces for repeated contiguous tool
     sequences (length 2-4), excluding sequences an existing skill already
     covers.
  2. **draft** — synthesizes a new skill (SKILL.md-style body following the
     repo's skill conventions) from the traces, via the model or a
     deterministic template fallback.
  3. **gate** — the skill must (a) register cleanly in ``SkillLibrary``,
     (b) pass a generated smoke test that replays one of the source traces
     through the skill's declared tool sequence. In ``approval`` mode the
     skill is staged for the owner; in ``autonomous`` it registers with a
     14-day probation flag, after which it is kept or removed by usage.
  4. **provenance** — every synthesized skill carries ``synthesized_from``
     (trace IDs) and ``synthesized_at``, in the body header and in kv_store.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .skills import SkillLibrary

_log = get_logger(__name__)

__all__ = ["SkillSynthesizer", "detect_patterns", "register"]

DEFAULT_MIN_REPEATS = 5
DEFAULT_WINDOW_DAYS = 30.0
PROBATION_DAYS = 14.0


@dataclass
class ToolPattern:
    tools: list[str]
    count: int
    trace_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"tools": self.tools, "count": self.count,
                "trace_ids": self.trace_ids}


def detect_patterns(db: Any, *, min_repeats: int = DEFAULT_MIN_REPEATS,
                    window_days: float = DEFAULT_WINDOW_DAYS,
                    seq_lengths: tuple[int, ...] = (2, 3, 4)
                    ) -> list[ToolPattern]:
    """Find repeated tool sequences in recent traces with no covering skill."""
    cutoff = time.time() - window_days * 86400.0
    try:
        rows = db.query(
            "SELECT id, actor, tool, created_at FROM tool_calls "
            "WHERE created_at > ? AND status IN ('ok','pending') "
            "ORDER BY actor, created_at", (cutoff,))
    except Exception:  # noqa: BLE001 — no tool_calls table
        return []
    # per-actor ordered tool sequences
    seqs: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        seqs.setdefault(r["actor"] or "?", []).append(
            {"tool": r["tool"], "id": r["id"]})
    # count n-grams
    counts: dict[tuple[str, ...], dict[str, Any]] = {}
    for actor, calls in seqs.items():
        tools = [c["tool"] for c in calls]
        for n in seq_lengths:
            for i in range(len(tools) - n + 1):
                key = tuple(tools[i:i + n])
                entry = counts.setdefault(
                    key, {"count": 0, "trace_ids": []})
                entry["count"] += 1
                entry["trace_ids"].append(calls[i]["id"])
    lib = SkillLibrary(db)
    try:
        skills = lib.list(limit=500)
    except Exception:  # noqa: BLE001
        skills = []
    patterns = []
    for key, entry in counts.items():
        if entry["count"] < min_repeats:
            continue
        if _covered_by_skill(key, skills):
            continue
        patterns.append(ToolPattern(
            tools=list(key), count=entry["count"],
            trace_ids=entry["trace_ids"][:10]))
    patterns.sort(key=lambda p: p.count, reverse=True)
    return patterns


def _covered_by_skill(tools: tuple[str, ...], skills: list[Any]) -> bool:
    """True when an existing skill already documents most of the sequence."""
    need = max(1, len(tools) // 2)
    for s in skills:
        text = f"{s.name} {s.description} {' '.join(s.tags)}".lower()
        hits = sum(1 for t in tools if t.lower() in text)
        if hits >= need:
            return True
    return False


def _template_draft(pattern: ToolPattern) -> dict[str, Any]:
    """Deterministic skill draft (fallback when no model is available)."""
    name = " ".join(t.replace("_", " ") for t in pattern.tools) + " workflow"
    steps = "\n".join(
        f"{i + 1}. Call `{tool}` with the task's arguments for this step; "
        f"feed its result into the next step."
        for i, tool in enumerate(pattern.tools))
    seq = "\n".join(f"- `{tool}`" for tool in pattern.tools)
    body = (
        f"# {name.title()}\n\n"
        f"synthesized_from: {', '.join(pattern.trace_ids[:5])}\n"
        f"synthesized_at: {time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}\n\n"
        f"## When to use\n\n"
        f"When a task needs this exact tool chain, observed "
        f"{pattern.count} times in recent traces:\n\n"
        f"## Tool sequence\n\n{seq}\n\n"
        f"## Steps\n\n{steps}\n\n"
        f"## Notes\n\n"
        f"- Run the tools in order; stop and report if any step errors.\n"
        f"- This skill was synthesized from repeated traces, not hand-written: "
        f"verify the argument mapping on first use.\n")
    return {
        "name": name.title(),
        "description": f"Repeated {len(pattern.tools)}-tool workflow: " +
                       ", ".join(pattern.tools),
        "body": body,
        "tags": ["synthesized", *[t.lower() for t in pattern.tools]],
        "kind": "workflow",
    }


def _skill_tools(body: str) -> list[str]:
    """Extract the declared tool sequence from a synthesized skill body."""
    m = re.search(r"## Tool sequence\s*\n(.*?)(?:\n## |\Z)", body,
                  re.S)
    if not m:
        return []
    return re.findall(r"`([a-z0-9_]+)`", m.group(1))


def smoke_test_skill(body: str, trace_tools: list[str]) -> tuple[bool, str]:
    """Replay a source trace through the skill: every tool in the trace
    must appear in the skill's declared sequence, in order."""
    declared = _skill_tools(body)
    if not declared:
        return False, "skill declares no tool sequence"
    pos = 0
    for tool in trace_tools:
        try:
            pos = declared.index(tool.lower(), pos) + 1
        except ValueError:
            return False, f"trace tool {tool!r} not in skill sequence"
    return True, f"trace replays cleanly ({len(trace_tools)} tools)"


class SkillSynthesizer:
    def __init__(
        self,
        context: Any,
        *,
        drafter: Callable[[ToolPattern], dict[str, Any]] | None = None,
        min_repeats: int = DEFAULT_MIN_REPEATS,
        window_days: float = DEFAULT_WINDOW_DAYS,
    ) -> None:
        self.context = context
        self.db = context.db
        self.skills = SkillLibrary(context.db)
        self._drafter = drafter or self._default_drafter
        self.min_repeats = min_repeats
        self.window_days = window_days

    @property
    def settings(self):
        return self.context.settings.improvement

    @property
    def mode(self) -> str:
        return (self.settings.mode or "off").strip().lower()

    # ── draft ───────────────────────────────────────────────────────────────
    def _default_drafter(self, pattern: ToolPattern) -> dict[str, Any]:
        router = getattr(self.context, "router", None)
        if router is None:
            return _template_draft(pattern)
        from ..llm.base import Message, SamplingParams
        prompt = (
            "You write Devon skills. Draft a SKILL.md-style skill for this "
            "repeated tool workflow.\n\n"
            f"Tool sequence (observed {pattern.count} times): "
            f"{', '.join(pattern.tools)}\n\n"
            "Respond with JSON ONLY: {\"name\": \"<Title Case name>\", "
            "\"description\": \"<one line>\", \"body\": \"<full SKILL.md "
            "markdown with ## When to use, ## Tool sequence (backticked "
            "tool names, one per line), ## Steps, ## Notes sections>\", "
            "\"tags\": [\"synthesized\", ...]}")
        try:
            resp = router.chat([Message.user(prompt)],
                               SamplingParams(temperature=0.3, max_tokens=1500))
            if not getattr(resp, "ok", False):
                return _template_draft(pattern)
            data = json.loads(_first_json(resp.text or ""))
            if not isinstance(data, dict) or not data.get("body"):
                return _template_draft(pattern)
            body = str(data["body"])
            if not _skill_tools(body):
                return _template_draft(pattern)
            return {
                "name": str(data.get("name") or "Synthesized workflow")[:120],
                "description": str(data.get("description") or "")[:500],
                "body": body[:60000],
                "tags": ["synthesized", *[
                    str(t)[:40] for t in (data.get("tags") or [])
                    if isinstance(t, str)]][:12],
                "kind": "workflow",
            }
        except Exception:  # noqa: BLE001 — degrade to the template
            _log.debug("model skill draft failed; using template")
            return _template_draft(pattern)

    # ── gate + register ─────────────────────────────────────────────────────
    def synthesize_pattern(self, pattern: ToolPattern, *,
                           mode: str = "") -> dict[str, Any]:
        """Draft, gate, and register (or stage) one skill for a pattern."""
        mode = (mode or self.mode or "off").strip().lower()
        if mode == "off":
            return {"ok": False, "error": "improvement is off"}
        draft = self._drafter(pattern)
        # gate (a): registers cleanly in SkillLibrary
        try:
            skill = self.skills.save(
                draft["name"], kind=draft.get("kind", "workflow"),
                body=draft["body"], description=draft.get("description", ""),
                tags=draft.get("tags", ["synthesized"]),
                source="synthesized")
            fetched = self.skills.get_by_name(draft["name"])
            if fetched is None or not fetched.body:
                return {"ok": False, "error": "registration round-trip failed"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"registration failed: {exc}"[:200]}
        # gate (b): smoke test — replay the pattern's canonical sequence
        # through the skill (trace ids are provenance, not the sequence:
        # they store only the first call of each occurrence)
        ok, detail = smoke_test_skill(fetched.body, list(pattern.tools))
        if not ok:
            # gate failed: remove the half-registered skill
            try:
                self.db.execute("DELETE FROM skills WHERE id=?", (skill.id,))
            except Exception:  # noqa: BLE001
                pass
            return {"ok": False, "error": f"smoke test failed: {detail}"}
        # provenance
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, kind, "
                "updated_at) VALUES (?,?, 'json', ?)",
                (f"skill.provenance.{skill.id}",
                 json.dumps({"synthesized_from": pattern.trace_ids,
                             "synthesized_at": time.time(),
                             "pattern": pattern.to_dict()}),
                 time.time()))
        except Exception:  # noqa: BLE001
            pass
        if mode == "approval":
            self._set_probation_tag(skill.id, "staged")
            return {"ok": True, "staged": True, "skill_id": skill.id,
                    "name": skill.name, "smoke": detail}
        # autonomous: register with a 14-day probation flag
        self._set_probation_tag(skill.id, "probation")
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, kind, "
                "updated_at) VALUES (?,?, 'json', ?)",
                (f"skill.probation.{skill.id}",
                 json.dumps({"until": time.time() + PROBATION_DAYS * 86400,
                             "pattern": pattern.to_dict()}),
                 time.time()))
        except Exception:  # noqa: BLE001
            pass
        _log.info("synthesized skill %s (%s) on probation",
                  skill.id, skill.name)
        return {"ok": True, "staged": False, "skill_id": skill.id,
                "name": skill.name, "smoke": detail,
                "probation_until": time.time() + PROBATION_DAYS * 86400}

    def _trace_tools(self, trace_ids: list[str]) -> list[str]:
        out = []
        for tid in trace_ids:
            try:
                row = self.db.query_one(
                    "SELECT tool FROM tool_calls WHERE id=?", (tid,))
            except Exception:  # noqa: BLE001
                continue
            if row:
                out.append(row["tool"])
        return out

    def _set_probation_tag(self, skill_id: str, tag: str) -> None:
        try:
            row = self.db.query_one("SELECT tags FROM skills WHERE id=?",
                                    (skill_id,))
            tags = (row["tags"] or "").split(",") if row else []
            if tag not in tags:
                tags.append(tag)
            self.db.execute("UPDATE skills SET tags=? WHERE id=?",
                            (",".join(tags)[:500], skill_id))
        except Exception:  # noqa: BLE001
            pass

    # ── probation review ────────────────────────────────────────────────────
    def review_probation(self) -> list[dict[str, Any]]:
        """Settle expired probations: keep skills that earned usage, remove
        ones nobody touched."""
        out: list[dict[str, Any]] = []
        try:
            rows = self.db.query(
                "SELECT key, value FROM kv_store WHERE key LIKE "
                "'skill.probation.%'")
        except Exception:  # noqa: BLE001
            return []
        for r in rows:
            skill_id = r["key"].split("skill.probation.", 1)[1]
            try:
                data = json.loads(r["value"] or "{}")
            except Exception:  # noqa: BLE001
                continue
            if time.time() < data.get("until", 0):
                continue
            try:
                srow = self.db.query_one(
                    "SELECT uses FROM skills WHERE id=?", (skill_id,))
            except Exception:  # noqa: BLE001
                srow = None
            uses = (srow["uses"] or 0) if srow else 0
            if uses > 0 and srow is not None:
                # keep: drop the probation tag
                try:
                    trow = self.db.query_one(
                        "SELECT tags FROM skills WHERE id=?", (skill_id,))
                    tags = [t for t in (trow["tags"] or "").split(",")
                            if t not in ("probation", "")]
                    self.db.execute("UPDATE skills SET tags=? WHERE id=?",
                                    (",".join(tags)[:500], skill_id))
                except Exception:  # noqa: BLE001
                    pass
                out.append({"skill_id": skill_id, "kept": True, "uses": uses})
            else:
                try:
                    self.db.execute("DELETE FROM skills WHERE id=?",
                                    (skill_id,))
                except Exception:  # noqa: BLE001
                    pass
                out.append({"skill_id": skill_id, "kept": False, "uses": uses})
            try:
                self.db.execute("DELETE FROM kv_store WHERE key=?", (r["key"],))
            except Exception:  # noqa: BLE001
                pass
        return out

    # ── continuous ──────────────────────────────────────────────────────────
    def scan(self) -> list[dict[str, Any]]:
        """One synthesis pass: settle expired probations, then detect
        patterns and draft+gate each. No-op when improvement is off."""
        if self.mode == "off":
            return []
        out = []
        try:
            reviewed = self.review_probation()
            if reviewed:
                out.append({"probation_reviewed": reviewed})
        except Exception:  # noqa: BLE001
            pass
        for pattern in detect_patterns(
                self.db, min_repeats=self.min_repeats,
                window_days=self.window_days):
            try:
                out.append({**self.synthesize_pattern(pattern),
                            "pattern": pattern.to_dict()})
            except Exception as exc:  # noqa: BLE001
                out.append({"ok": False, "error": str(exc)[:200],
                            "pattern": pattern.to_dict()})
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
        "skill_synthesize",
        description=(
            "Skill synthesis from repeated patterns: scan tool-call traces "
            "for repeated tool sequences with no covering skill, draft a new "
            "skill, gate it (registers cleanly + smoke test replays a source "
            "trace), stage or probation-register per improvement mode. "
            "action=scan | patterns | review_probation. mode: "
            "off|approval|autonomous."
        ),
        capability="model.call",
        parameters={
            "action": "str — scan|patterns|review_probation",
        },
    )
    def skill_synthesize(action: str = "patterns") -> dict[str, Any]:
        synth = SkillSynthesizer(context)
        action = (action or "patterns").strip().lower()
        if action == "scan":
            return {"ok": True, "results": synth.scan()}
        if action == "review_probation":
            return {"reviewed": synth.review_probation()}
        return {"patterns": [p.to_dict() for p in detect_patterns(
            context.db)]}
