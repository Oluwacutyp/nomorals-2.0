"""Reflective checkpointing (wave 64) — the loop that looks back.

When a goal finishes, ``GoalReflector`` runs a *reflection pass* over
everything that actually happened: which steps needed retries, what failed
and how it was recovered, how many times the autonomous loop had to
self-heal, and what upstream knowledge the goal built on.

The distilled outcome is written into three durable stores:

  * **the knowledge graph** — a ``fact`` node (``reflection:<title>``) per
    completed goal, linked ``reflects`` back to the goal node, so future
    KG-aware reasoning sees not just *that* a goal finished but *what it
    taught the system*;
  * **the skill library** — when the reflection identifies a reusable
    approach or a failure to prevent, it is saved as a real skill
    (strategy / prevention / solution / …) that ``recall`` will surface
    for future tasks;
  * **the reflection record** — the full distilled result, stored
    durably per goal (idempotent; re-run with ``force=True``).

When a model is available the distillation is a model pass (structured
JSON); when it is not, a deterministic heuristic extracts the same shape
from the step history — reflection ALWAYS happens, it never blocks on a
model, and it never fails a goal completion.

Modular + callable by the main AI and sub-agents: ``nm goal reflect <id>``,
the ``goal`` tool with ``action=reflect``, the ``reflection`` tool
(``action=reflect|list|last``), or ``GoalReflector(context).reflect(id)``
directly.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..llm.base import Message, SamplingParams
from ..llm.brain import brain_for
from ..storage.kv import KVStore

_log = get_logger(__name__)

__all__ = ["GoalReflector", "register"]

_SKILL_KINDS = {"strategy", "prevention", "solution", "code", "prompt"}


def _slug(text: str, limit: int = 40) -> str:
    from ..core.text import slugify

    # canonical: nomorals.core.text.slugify
    return slugify(text, limit=limit, fallback="goal", strip_after_limit=True)


def _tags(text: str, limit: int = 5) -> list[str]:
    words = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= limit:
            break
    return out


def _str_list(raw: Any, limit: int, maxlen: int = 300) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        s = str(item).strip()
        if s:
            out.append(s[:maxlen])
        if len(out) >= limit:
            break
    return out


class GoalReflector:
    """Distill a finished goal into durable lessons (KG + skills)."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = context.db

    # ── keying ─────────────────────────────────────────────────────────────
    @staticmethod
    def _key(goal_id: str) -> str:
        return f"goal.reflection.{goal_id}"

    def existing(self, goal_id: str) -> Optional[dict[str, Any]]:
        """The stored reflection record, or None."""
        try:
            data = KVStore(self.db).get(self._key(goal_id))
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """All stored reflections, newest first (goal_id recovered from key)."""
        try:
            pairs = KVStore(self.db).scan("goal.reflection.", limit=limit)
        except Exception:  # noqa: BLE001
            return []
        out = []
        for key, data in pairs:
            if isinstance(data, dict):
                out.append({"goal_id": key.split(".", 2)[-1],
                            "ts": float(data.get("ts", 0) or 0), **data})
        # Sort by ts DESC to match the original ORDER BY updated_at DESC
        out.sort(key=lambda r: r["ts"], reverse=True)
        return out[:limit]

    # ── the reflection pass ────────────────────────────────────────────────
    def reflect(self, goal_id: str, *, force: bool = False) -> dict[str, Any]:
        """Run (or read back) the reflection for a finished goal."""
        from .goals import GoalSystem

        gs = GoalSystem(self.context)
        goal = gs.get(goal_id)
        if goal is None:
            return {"ok": False, "error": f"no goal {goal_id!r}"}
        if goal.status != "done":
            return {"ok": False,
                    "error": f"goal {goal_id!r} is {goal.status!r}, not done"}
        stored = self.existing(goal_id)
        if stored is not None and not force:
            return {"ok": True, "goal_id": goal_id, "cached": True, **stored}

        evidence = self._evidence(gs, goal)
        reflection: Optional[dict[str, Any]] = None
        source = "heuristic"
        if getattr(self.context, "router", None) is not None:
            try:
                reflection = self._model_reflection(goal, evidence)
                if reflection is not None:
                    source = "model"
            except Exception as exc:  # noqa: BLE001 — heuristic below
                _log.debug("model reflection failed: %s", exc)
        if reflection is None:
            reflection = self._heuristic_reflection(goal)
        meta = self._persist(goal, reflection, source)
        return {"ok": True, "goal_id": goal_id, "cached": False,
                "source": source, **reflection, **meta}

    # ── evidence ──────────────────────────────────────────────────────────
    def _evidence(self, gs: Any, goal: Any) -> str:
        lines = [
            f"GOAL: {goal.title}",
            f"OBJECTIVE: {(goal.description or '')[:600]}",
            (f"ELAPSED: {int((goal.finished_at or goal.updated_at) - (goal.created_at or 0))}s"),
            f"SELF-HEALS BY AUTONOMOUS LOOP: {goal.heals}",
            "STEPS (status | attempts | result):",
        ]
        for s in sorted(goal.steps, key=lambda x: x.position):
            lines.append(f"  - [{s.status} | attempts={s.attempts}] "
                         f"{s.description[:200]}")
            if s.result:
                lines.append(f"      result: {s.result[:300]}")
        try:
            upstream = gs.upstream_knowledge(goal.id)
            if upstream:
                lines.append("UPSTREAM KNOWLEDGE THIS GOAL BUILT ON:\n" + upstream)
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines)

    # ── model pass ─────────────────────────────────────────────────────────
    def _model_reflection(self, goal: Any, evidence: str) -> Optional[dict[str, Any]]:
        router = getattr(self.context, "router", None)
        prompt = (
            "You are reviewing a goal that an autonomous system just "
            "completed. Distill what actually worked, what did not, and "
            "one reusable lesson. Be specific and evidence-based — quote "
            "the step history, do not invent outcomes.\n\n"
            f"{evidence}\n\n"
            "Respond with JSON ONLY:\n"
            '{"what_worked": ["<specific>"], '
            '"what_didnt_work": ["<specific, or empty>"], '
            '"lessons": ["<reusable lesson>"], '
            '"reusable_skill": null or '
            '{"name": "<short-kebab-name>", '
            '"kind": "strategy|prevention|solution|code|prompt", '
            '"body": "<the reusable approach or avoidance rule>"}}')
        response = brain_for(self.context).chat(
            [Message.system("You are a rigorous post-mortem reviewer. "
                            "No preamble, no apologies."),
             Message.user(prompt)],
            SamplingParams(temperature=0.2, max_tokens=700), task_kind="judge")
        if not response.ok:
            return None
        from .reasoning import _extract_json

        data = _extract_json(response.text or "")
        if not isinstance(data, dict):
            return None
        result = {
            "what_worked": _str_list(data.get("what_worked"), 6),
            "what_didnt_work": _str_list(data.get("what_didnt_work"), 6),
            "lessons": _str_list(data.get("lessons"), 5),
            "reusable_skill": self._normalize_skill(data.get("reusable_skill")),
        }
        # vacuous reply (any JSON dict with none of our fields) = the model
        # ignored the format — the deterministic pass is worth more
        if not (result["what_worked"] or result["what_didnt_work"]
                or result["lessons"] or result["reusable_skill"]):
            return None
        return result

    @staticmethod
    def _normalize_skill(raw: Any) -> Optional[dict[str, str]]:
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name", "")).strip()[:80]
        body = str(raw.get("body", "")).strip()
        kind = str(raw.get("kind", "")).strip().lower()
        if not name or not body:
            return None
        if kind not in _SKILL_KINDS:
            kind = "strategy"
        return {"name": name, "kind": kind, "body": body[:6000]}

    # ── deterministic fallback ─────────────────────────────────────────────
    def _heuristic_reflection(self, goal: Any) -> dict[str, Any]:
        steps = goal.steps
        done = [s for s in steps if s.status == "done"]
        blocked = [s for s in steps if s.status == "blocked"]
        retries = [s for s in steps if s.attempts > 1]

        worked: list[str] = []
        if done:
            worked.append(f"{len(done)}/{len(steps)} step(s) completed "
                          f"for '{goal.title[:60]}'")
        if retries:
            worked.append("recovered from mid-run failure: "
                          + "; ".join(s.description[:60] for s in retries[:3]))
        if goal.heals:
            worked.append(f"survived {goal.heals} self-heal cycle(s) "
                          "driven by the autonomous loop")
        if not worked:
            worked.append("goal completed")

        didnt: list[str] = []
        for s in blocked[:4]:
            didnt.append(f"step blocked: {s.description[:120]} "
                         f"({(s.result or '')[:120]})")
        if goal.heals:
            didnt.append(f"required {goal.heals} self-heal(s) — the first "
                         "plan did not hold on its own")

        lessons: list[str] = []
        if retries:
            lessons.append(f"retry-with-context recovered {len(retries)} "
                           "step(s) — keep a retry/reword path for this "
                           "class of goal")
        if goal.heals:
            lessons.append(f"this goal needed {goal.heals} heal(s); "
                           "verify assumptions earlier (or over-provision "
                           "steps) for similar goals")
        if not lessons:
            lessons.append(f"first-pass execution of {len(done)} step(s) "
                           "succeeded — this plan shape is reusable")

        skill: Optional[dict[str, str]] = None
        if blocked or goal.heals:
            skill = {
                "name": f"lesson-{_slug(goal.title)}",
                "kind": "prevention",
                "body": ("Failure evidence (goal: " + goal.title[:120] + "):\n"
                         + "\n".join(f"- {d}" for d in didnt)
                         + "\n\nAvoidance:\n"
                         + "\n".join(f"- {l}" for l in lessons)),
            }
        return {"what_worked": worked, "what_didnt_work": didnt,
                "lessons": lessons, "reusable_skill": skill}

    # ── persistence ────────────────────────────────────────────────────────
    def _persist(self, goal: Any, reflection: dict[str, Any],
                 source: str) -> dict[str, Any]:
        now = time.time()
        meta: dict[str, Any] = {"kg_node": "", "skill_id": "",
                                "skill_name": ""}
        skill = reflection.get("reusable_skill")
        if skill:
            meta["skill_name"] = skill["name"]
        try:
            KVStore(self.db).set(self._key(goal.id), {
                "source": source,
                "what_worked": reflection.get("what_worked", []),
                "what_didnt_work": reflection.get("what_didnt_work", []),
                "lessons": reflection.get("lessons", []),
                "skill_name": meta["skill_name"],
                "skill_id": "",  # filled below
                "ts": now,
            })
        except Exception as exc:  # noqa: BLE001
            _log.debug("reflection record write failed: %s", exc)

        try:
            from .kg import KnowledgeGraph

            kg = KnowledgeGraph(self.db)
            meta["kg_node"] = f"reflection:{goal.title}"
            kg.upsert_node(meta["kg_node"], type="fact", properties={
                "goal_id": goal.id, "source": source,
                "what_worked": "; ".join(
                    reflection.get("what_worked", []))[:800],
                "what_didnt_work": "; ".join(
                    reflection.get("what_didnt_work", []))[:800],
                "lessons": "; ".join(reflection.get("lessons", []))[:800],
                "finished_at": goal.finished_at,
            })
            kg.link(meta["kg_node"], goal.title, "reflects")
        except Exception as exc:  # noqa: BLE001
            _log.debug("reflection KG write failed: %s", exc)

        if skill:
            try:
                from .skills import SkillLibrary

                s = SkillLibrary(self.db).save(
                    skill["name"], kind=skill.get("kind", "strategy"),
                    body=skill.get("body", ""),
                    description=f"Distilled from completed goal: "
                                f"{goal.title[:120]}",
                    tags=_tags(goal.title), source="reflection")
                meta["skill_id"] = s.id
                meta["skill_name"] = s.name
            except Exception as exc:  # noqa: BLE001
                _log.debug("reflection skill save failed: %s", exc)
        # keep the record consistent with the real skill id
        try:
            stored = self.existing(goal.id) or {}
            if meta.get("skill_id"):
                stored["skill_id"] = meta["skill_id"]
                KVStore(self.db).set(self._key(goal.id), stored)
        except Exception:  # noqa: BLE001
            pass
        return meta


# ── registry ─────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "reflection",
        description=(
            "Reflective checkpointing: distill what worked / didn't from a "
            "finished goal into the knowledge graph + skill library. "
            "action=reflect <goal_id> (fresh pass) | list | last."
        ),
        capability="memory.write",
        parameters={
            "action": "str — reflect|list|last",
            "goal_id": "str — the finished goal to reflect on (reflect)",
            "limit": "int — for list (default 20)",
        },
    )
    def reflection(action: str = "reflect", *, goal_id: str = "",
                   limit: str = "20") -> dict[str, Any]:
        rf = GoalReflector(context)
        action = (action or "list").strip().lower()
        if action == "reflect":
            if not goal_id:
                return {"ok": False, "error": "goal_id required"}
            return rf.reflect(goal_id, force=True)
        if action == "last":
            rows = rf.list(limit=1)
            return {"ok": True, "last": rows[0] if rows else None}
        try:
            n = int(limit or 20)
        except ValueError:
            n = 20
        return {"ok": True, "reflections": rf.list(limit=n)}


# referenced so the id import is never flagged unused
_ = new_short_id
