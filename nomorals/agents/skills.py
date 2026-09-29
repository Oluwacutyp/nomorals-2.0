"""Skill library — proven approaches the system reuses and improves.

When an agent solves something well (a tool sequence that answered the
question, a code fix that ran, a strategy that worked), that success is
captured as a *skill*: a named, typed, tagged, versioned artifact stored
durally.  Any agent can then:

  * ``recall(query)`` — get the best prior art for a new task, ranked by
    real success rate + relevance + recency, before starting from scratch
  * ``record_use(skill_id, success)`` — reinforce or weaken a skill based
    on the actual outcome of using it
  * ``improve(skill_id, revision)`` — replace a skill's body with a better
    variant and bump its version (the old one is kept in history)

Kinds: strategy (an approach), tool_sequence (ordered tool calls),
solution (a finished answer), prompt (a reusable prompt), code (a working
snippet), prevention (a lesson that avoids a known failure).

Skills are the substrate the failure-analyst writes to, the goal system
consults, and the self-improvement loop strengthens — one memory of *how
to do things* shared by every agent.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["Skill", "SkillLibrary", "register"]

_SKILL_KINDS = {"strategy", "tool_sequence", "solution", "prompt", "code",
                "prevention", "routing", "plantmpl"}
_STOP = re.compile(
    r"\b(a|an|and|the|of|to|in|on|for|is|are|was|were|be|with|as|at|by|"
    r"from|that|this|it|its|or|not|no|so|if|then|than|into|about|how|what|"
    r"why|when|where|which|who|will|would|can|could|should|do|does|did|"
    r"have|has|had|i|you|he|she|we|they|my|your|our|their)\b")


def _tokens(text: str) -> set[str]:
    return {t for t in _STOP.sub("", (text or "").lower()).split() if len(t) >= 3}


@dataclass
class Skill:
    id: str
    name: str
    description: str = ""
    kind: str = "strategy"
    body: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    success_count: int = 0
    failure_count: int = 0
    uses: int = 0
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    last_used: float = 0.0
    pruned: bool = False  # wave 77 auto-pruning: used often, mostly fails

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 0.5

    @property
    def text(self) -> str:
        return f"{self.name} {self.description} {' '.join(self.tags)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "kind": self.kind, "body": self.body, "tags": self.tags,
            "source": self.source, "success_count": self.success_count,
            "failure_count": self.failure_count, "uses": self.uses,
            "version": self.version, "success_rate": round(self.success_rate, 3),
            "last_used": self.last_used, "pruned": self.pruned,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Skill":
        try:
            tags = [t.strip() for t in (row.get("tags") or "").split(",")
                    if t.strip()]
        except Exception:  # noqa: BLE001
            tags = []
        return cls(
            id=row["id"], name=row["name"],
            description=row.get("description", ""),
            kind=row.get("kind", "strategy"), body=row.get("body", ""),
            tags=tags, source=row.get("source", ""),
            success_count=int(row.get("success_count", 0)),
            failure_count=int(row.get("failure_count", 0)),
            uses=int(row.get("uses", 0)),
            version=int(row.get("version", 1)),
            created_at=float(row.get("created_at", 0)),
            updated_at=float(row.get("updated_at", 0)),
            last_used=float(row.get("last_used", 0)),
            pruned=bool(row.get("pruned", 0)),
        )


class SkillLibrary:
    """Durable, ranked, self-updating store of reusable skills."""

    def __init__(self, db: Any) -> None:
        self.db = db

    # ── write ───────────────────────────────────────────────────────────────
    def save(
        self,
        name: str,
        *,
        kind: str = "strategy",
        body: str = "",
        description: str = "",
        tags: list[str] | None = None,
        source: str = "",
        id: str = "",
    ) -> Skill:
        """Create (or upsert by name) a skill.  Upserting an existing name
        keeps its usage counters but refreshes the body."""
        kind = kind if kind in _SKILL_KINDS else "strategy"
        now = time.time()
        name = (name or "").strip() or "unnamed skill"
        skill_id = id or new_short_id("skill")
        tags_csv = ",".join(dict.fromkeys((tags or [])))[:500]
        description = (description or "")[:1000]
        body = (body or "")[:60000]
        row = self.db.query_one("SELECT * FROM agent_skills WHERE name = ?", (name,))
        if row:
            skill_id = row["id"]
            self.db.execute(
                "UPDATE agent_skills SET kind=?, body=?, description=?, tags=?, "
                "source=?, version=version+1, updated_at=? WHERE id=?",
                (kind, body, description, tags_csv, source, now, skill_id),
            )
        else:
            self.db.execute(
                "INSERT INTO agent_skills (id, name, description, kind, body, tags, "
                "source, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (skill_id, name, description, kind, body, tags_csv, source,
                 now, now),
            )
        return self.get(skill_id) or Skill(id=skill_id, name=name, kind=kind,
                                           body=body, created_at=now,
                                           updated_at=now)

    def capture_tool_sequence(
        self,
        name: str,
        steps: list[dict[str, Any]],
        *,
        task: str = "",
        success: bool = True,
        tags: list[str] | None = None,
        source: str = "devon",
    ) -> Skill | None:
        """Capture a *successful* tool sequence as a reusable skill. Only
        successful runs are captured — a failing sequence is not prior art."""
        if not success or not steps:
            return None
        body = json.dumps(steps, ensure_ascii=False, default=str)
        return self.save(
            name, kind="tool_sequence", body=body,
            description=f"Proven tool sequence for: {task[:300]}",
            tags=tags or [t for t in _tokens(task)][:6], source=source)

    def capture_from_ledger(self, *, limit: int = 200,
                            window_seconds: float = 300.0) -> list[str]:
        """Mine the ``tool_calls`` ledger for *recovered* patterns and turn
        them into durable skills — the skill library learning by watching
        the system actually work.

        A recovery is: a failing call T1 (tool X), then a succeeding call T2
        by the same actor within the window.  Two shapes are captured:
        * retry-of-same-tool   → "retry <X> after <error family>"
        * pivot-to-other-tool  → "recover <X> via <Y>" (T2 is a different tool)
        Only the first recovery per (tool, family, recovery-tool) is kept;
        repeats upsert the same skill.  Returns the captured skill names.
        """
        captured: list[str] = []
        try:
            rows = self.db.query(
                "SELECT actor, tool, status, error, created_at FROM "
                "tool_calls WHERE status IN ('error','ok') "
                "ORDER BY created_at DESC LIMIT ?", (limit,))
        except Exception:  # noqa: BLE001 - no ledger yet
            return captured
        rows.reverse()  # chronological
        from .failure import categorize
        i = 0
        while i < len(rows) - 1:
            fail = rows[i]
            if fail["status"] != "error":
                i += 1
                continue
            family = categorize(fail["error"])
            # find the next success by the same actor inside the window
            for j in range(i + 1, len(rows)):
                nxt = rows[j]
                if nxt["actor"] != fail["actor"]:
                    continue
                if (nxt["created_at"] - fail["created_at"]) > window_seconds:
                    break
                if nxt["status"] != "ok":
                    continue
                if nxt["tool"] == fail["tool"]:
                    name = (f"retry {fail['tool']} after {family} error")
                    body = (f"When {fail['tool']} fails with a {family} "
                            f"error, retry it after fixing the cause. "
                            f"Observed recovery: {fail['error'][:200]}")
                    tags = [fail["tool"], family, "retry"]
                else:
                    name = (f"recover {fail['tool']} via {nxt['tool']} "
                            f"({family})")
                    body = (f"{fail['tool']} hit a {family} failure; the "
                            f"call that actually worked afterwards was "
                            f"{nxt['tool']}. Consider that path before "
                            f"repeating the failing call.")
                    tags = [fail["tool"], nxt["tool"], family, "recovery"]
                skill = self.save(
                    name, kind="solution", body=body[:6000],
                    description=f"Auto-captured recovery for {fail['tool']}",
                    tags=tags, source="ledger")
                captured.append(skill.name)
                break
            i += 1
        return captured

    def operator_context(self, query: str, *, limit: int = 3) -> str:
        """The complete 'what already worked / what already failed' block
        for a task: top recalled skills + systemic traps + failure
        prevention lessons.  This is the canonical context an agent
        prepends before attempting anything non-trivial.  Never raises."""
        parts: list[str] = []
        try:
            block = self.context_block(query, limit=limit)
            if block:
                parts.append(block)
            traps = self.trap_block()
            if traps:
                parts.append(traps)
            try:
                from .failure import FailureAnalyzer
                prev = FailureAnalyzer(type("_Ctx", (), {"db": self.db})()). \
                    prevention_context(query)
                if prev:
                    parts.append(prev)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            return ""
        return "\n\n".join(parts)

    def record_use(self, skill_id: str, *, success: bool,
                   task: str = "", outcome: str = "") -> Skill | None:
        """Reinforce or weaken a skill based on the real outcome of using it."""
        skill = self.get(skill_id)
        if skill is None:
            return None
        now = time.time()
        if success:
            self.db.execute(
                "UPDATE agent_skills SET success_count=success_count+1, "
                "uses=uses+1, last_used=? WHERE id=?", (now, skill_id))
        else:
            self.db.execute(
                "UPDATE agent_skills SET failure_count=failure_count+1, "
                "uses=uses+1, last_used=? WHERE id=?", (now, skill_id))
        self.db.execute(
            "INSERT INTO skill_uses (id, skill_id, task, success, outcome, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (new_short_id("su"), skill_id, task[:500], int(success),
             outcome[:2000], now),
        )
        return self.get(skill_id)

    def improve(self, skill_id: str, *, body: str, note: str = "") -> Skill | None:
        """Replace a skill's body with a better variant; bump the version.

        Improving a skill is a *rescue*: it clears any prune flag, because
        a hand-revised body is a fresh claim worth testing again."""
        skill = self.get(skill_id)
        if skill is None:
            return None
        now = time.time()
        self.db.execute(
            "UPDATE agent_skills SET body=?, version=version+1, updated_at=?, "
            "pruned=0, pruned_at=0 WHERE id=?",
            (body[:60000], now, skill_id))
        _log.info("skill %s improved to v%d (%s)", skill.name,
                  skill.version + 1, note[:80])
        return self.get(skill_id)

    # ── auto-pruning (wave 77) ─────────────────────────────────────────────
    def prune(
        self,
        *,
        min_uses: int = 5,
        max_success_rate: float = 0.34,
        min_age_days: float = 7.0,
    ) -> dict[str, Any]:
        """Quarantine skills that are USED OFTEN and MOSTLY FAIL.

        A skill earns a prune flag when ALL of:
          * ``uses >= min_uses``            — it has been tried enough to judge
          * ``success_rate < max_success_rate`` — it mostly does not work
          * age  >= min_age_days            — it is not brand-new (give new
                                               skills a chance to prove out)

        Pruned skills are excluded from ``recall`` / ``operator_context`` /
        ``list`` (the recall paths all go through ``list``), so the system
        stops re-using a dead strategy — but they are NOT deleted: they stay
        queryable (``list(include_pruned=True)``) and restorable via
        ``restore`` / a fresh ``improve``.  This is decay, not amnesia: the
        record of "we tried this and it failed" is exactly the prior art
        ``stats`` and the failure ledger want to keep.

        Returns ``{"pruned": [names], "count": n, "total": N,
        "pruned_total": P, "criteria": {...}}``.
        """
        now = time.time()
        min_age_s = max(0.0, float(min_age_days)) * 86400.0
        rows = self.db.query(
            "SELECT * FROM agent_skills WHERE pruned=0 ORDER BY updated_at DESC")
        pruned: list[str] = []
        for r in rows:
            s = Skill.from_row(r)
            if s.uses < int(min_uses):
                continue
            if s.success_rate >= float(max_success_rate):
                continue
            age = now - (s.updated_at or s.created_at)
            if age < min_age_s:
                continue
            self.db.execute(
                "UPDATE agent_skills SET pruned=1, pruned_at=? WHERE id=?",
                (now, s.id))
            pruned.append(s.name)
        total = self.db.query_one("SELECT COUNT(*) n FROM agent_skills")["n"]
        pruned_total = self.db.query_one(
            "SELECT COUNT(*) n FROM agent_skills WHERE pruned=1")["n"]
        return {
            "pruned": pruned, "count": len(pruned), "total": int(total),
            "pruned_total": int(pruned_total),
            "criteria": {"min_uses": int(min_uses),
                         "max_success_rate": float(max_success_rate),
                         "min_age_days": float(min_age_days)},
        }

    def restore(self, skill_id: str) -> Skill | None:
        """Lift the prune flag off a skill (it re-enters recall)."""
        skill = self.get(skill_id)
        if skill is None:
            return None
        self.db.execute(
            "UPDATE agent_skills SET pruned=0, pruned_at=0 WHERE id=?", (skill_id,))
        return self.get(skill_id)

    def prune_stats(self) -> dict[str, Any]:
        """How many skills are alive vs quarantined, for the dashboard."""
        total = self.db.query_one("SELECT COUNT(*) n FROM agent_skills")["n"]
        pruned = self.db.query_one(
            "SELECT COUNT(*) n FROM agent_skills WHERE pruned=1")["n"]
        return {"total": int(total), "pruned": int(pruned),
                "active": int(total) - int(pruned)}

    # ── read / recall ───────────────────────────────────────────────────────
    def get(self, skill_id: str) -> Skill | None:
        row = self.db.query_one("SELECT * FROM agent_skills WHERE id=?", (skill_id,))
        return Skill.from_row(row) if row else None

    def get_by_name(self, name: str) -> Skill | None:
        row = self.db.query_one("SELECT * FROM agent_skills WHERE name=?", (name,))
        return Skill.from_row(row) if row else None

    def list(self, *, kind: str = "", limit: int = 50,
             include_pruned: bool = False) -> list[Skill]:
        """List skills. Pruned skills are EXCLUDED by default (they are
        quarantined from recall/context) — pass ``include_pruned=True``
        to see them (the ``nm skill list --pruned`` view)."""
        sql = "SELECT * FROM agent_skills"
        clauses: list[str] = []
        params: list[Any] = []
        if not include_pruned:
            clauses.append("pruned=0")
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, tuple(params))
        return [Skill.from_row(r) for r in rows]

    def recall(self, query: str, *, kind: str = "", limit: int = 5,
               min_uses: int = 0) -> list[tuple[Skill, float]]:
        """Rank skills for a query: 0.5*success_rate + 0.3*relevance +
        0.2*recency. Returns [(skill, score)] best-first."""
        query = (query or "").strip()
        qtok = _tokens(query)
        skills = self.list(kind=kind, limit=500)
        now = time.time()
        scored: list[tuple[Skill, float]] = []
        for s in skills:
            if s.uses < min_uses:
                continue
            if kind and s.kind != kind:
                continue
            stok = _tokens(s.text)
            relevance = (len(qtok & stok) / len(qtok)) if qtok else 0.0
            age = max(0.0, now - (s.last_used or s.updated_at or s.created_at))
            recency = 1.0 / (1.0 + age / (7 * 86400))  # half-life ~7 days
            score = 0.5 * s.success_rate + 0.3 * relevance + 0.2 * recency
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def context_block(self, query: str, *, limit: int = 3) -> str:
        """A compact 'prior art' block an agent can prepend to a task: the
        top recalled skills and what worked."""
        recalled = self.recall(query, limit=limit)
        if not recalled:
            return ""
        lines = [f"Relevant proven skills for '{query[:60]}':"]
        for s, score in recalled:
            preview = s.body[:180].replace("\n", " ")
            if s.kind == "tool_sequence":
                try:
                    tools = [st.get("tool", "?") for st in
                             json.loads(s.body) if isinstance(st, dict)]
                    preview = " → ".join(tools)
                except Exception:  # noqa: BLE001
                    pass
            lines.append(f"  - {s.name} [{s.kind}] (success "
                         f"{s.success_rate:.0%}, used {s.uses}x): {preview}")
        return "\n".join(lines)

    # ── systemic error memory (wave 67) ──────────────────────────────────
    def _error_family(self, skill: "Skill") -> str:
        """The error family a distilled coding-session skill is about:
        the first slug token of its 'fix-<slug>' name (e.g.
        'fix-modulenotfounderror-no-modul' -> 'modulenotfounderror')."""
        name = skill.name or ""
        slug = name[4:] if name.startswith("fix-") else name
        parts = [w for w in slug.split("-") if w]
        return (parts[0] if parts else slug).lower()

    def systemic_traps(self, limit: int = 4) -> list[dict[str, Any]]:
        """Cross-build error memory: error families that RECUR across
        sessions become systemic traps.

        A family (the first slug token of a distilled session skill)
        qualifies when it appears in >= 3 distinct distilled skills
        (same error, different tasks/fixes) OR a single skill was
        upserted >= 3 times (the same build kept hitting it).  Ranked by
        total hits, newest evidence last.  This is what new builds are
        warned about up front — the system's scar tissue, made usable.
        """
        try:
            rows = self.list(kind="code", limit=200)
        except Exception:  # noqa: BLE001 - traps are best-effort
            return []
        fam: dict[str, list["Skill"]] = {}
        for s in rows:
            if s.source != "coding_session":
                continue
            fam.setdefault(self._error_family(s), []).append(s)
        out: list[dict[str, Any]] = []
        for family, group in fam.items():
            repeat_hits = sum(max(0, s.version - 1) for s in group)
            total = len(group) + repeat_hits
            if len(group) < 3 and max(s.version for s in group) < 3:
                continue  # seen once or twice — anecdote, not a trap
            top = max(group, key=lambda s: (s.version, s.uses, s.updated_at))
            out.append({
                "family": family,
                "hits": total,
                "distinct_sessions": len(group),
                "skill": top.to_dict(),
                "tasks": sorted({(s.description or "")[:90] for s in group})[:3],
            })
        out.sort(key=lambda d: (-d["hits"], d["family"]))
        return out[:limit]

    def trap_block(self, limit: int = 4) -> str:
        """A compact 'known traps' block to prepend to a build's FIRST
        prompt — deterministic, no model call.  Empty when the system
        has not yet accumulated systemic error memory."""
        traps = self.systemic_traps(limit)
        if not traps:
            return ""
        lines = ["KNOWN TRAPS from past builds (avoid these patterns):"]
        for t in traps:
            body = t["skill"].get("body") or ""
            first_err = next((ln.strip() for ln in body.splitlines()
                              if ln.strip().startswith("- attempt")), "")
            detail = first_err or (t["skill"].get("description") or "")[:100]
            lines.append(f"- {t['family']} (seen {t['hits']}x across "
                         f"{t['distinct_sessions']} session(s)): {detail[:110]}")
        return "\n".join(lines)

    def match_errors(self, stderr_text: str, limit: int = 3) -> list["Skill"]:
        """Hard mid-loop recall (wave 67): which distilled fix skills
        match the CURRENT sandbox error?  Deterministic string matching
        on the error family + the skill's slug words — no model call.
        The coding loop injects the matches into the next fix prompt,
        so attempt N+1 sees the exact proven fix for the exact error it
        just hit."""
        low = " ".join(str(stderr_text or "").lower().split())
        compact = re.sub(r"[^a-z0-9]", "", low)
        if not low:
            return []
        try:
            rows = self.list(kind="code", limit=100)
        except Exception:  # noqa: BLE001 - recall is best-effort
            return []
        out: list["Skill"] = []
        for s in rows:
            if s.source != "coding_session":
                continue
            family = self._error_family(s)
            slug = (s.name or "")[4:] if (s.name or "").startswith("fix-") \
                else (s.name or "")
            words = [w for w in slug.split("-") if len(w) >= 4]
            hit = bool(family) and (family in low)
            if not hit and words and compact:
                hit = all(w in compact for w in words)
            if hit:
                out.append(s)
        out.sort(key=lambda s: -(s.version + s.uses))
        return out[:limit]

    def match_preventions(self, action_text: str, limit: int = 3) -> list["Skill"]:
        """Wave 82: prevention skills matching an action about to run.

        The mirror of :meth:`match_errors` for the pre-action side:
        deterministic token overlap between the action text and the
        skill's name/description/body — no model call.  A hit means this
        (or a very close) action was repeatedly ABORTED by the pre-action
        check before, and the system remembers why.
        """
        low = " ".join(str(action_text or "").lower().split())
        qtok = {t for t in low.split() if len(t) >= 4}
        if not qtok:
            return []
        try:
            rows = self.list(kind="prevention", limit=100)
        except Exception:  # noqa: BLE001 - recall is best-effort
            return []
        out: list["Skill"] = []
        for s in rows:
            stok = _tokens(f"{s.name} {s.description} {s.body}")
            overlap = qtok & stok
            if len(overlap) >= 2 or (len(qtok) == 1 and overlap):
                out.append(s)
        out.sort(key=lambda s: -(s.uses + s.version))
        return out[:limit]

    def delete(self, skill_id: str) -> bool:
        cur = self.db.execute("DELETE FROM agent_skills WHERE id=?", (skill_id,))
        return cur.rowcount > 0

    def stats(self) -> dict[str, Any]:
        total = self.db.query_one("SELECT COUNT(*) AS n FROM agent_skills")["n"]
        by_kind = {r["kind"]: r["n"] for r in self.db.query(
            "SELECT kind, COUNT(*) AS n FROM agent_skills GROUP BY kind")}
        out = {"total": total, "by_kind": by_kind}
        out.update(self.prune_stats())
        return out


# ── registry ─────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "skill",
        description=(
            "Skill library — reusable proven approaches. "
            "action=recall (find prior art for a task) | context (the full "
            "prior-art + traps + failure-prevention block for a task) | "
            "capture (mine the tool-call ledger for recovered patterns and "
            "turn them into skills) | save (store a new skill) | list "
            "(include_pruned to see quarantined ones) | use (record a "
            "skill was used, success/fail) | improve (replace a skill's "
            "body — also rescues a pruned skill) | prune (quarantine "
            "skills used often that mostly fail; min_uses, "
            "max_success_rate, min_age_days) | restore (skill_id — lift a "
            "prune) | stats (includes active/pruned counts)."
        ),
        capability="memory.write",
        parameters={
            "action": "str — recall|context|capture|save|list|use|improve|prune|restore|get|stats",
            "query": "str — the task to recall skills for",
            "name": "str — skill name (save/get)",
            "kind": "str — strategy|tool_sequence|solution|prompt|code|prevention",
            "body": "str — the skill artifact (code/sequence/prompt text)",
            "description": "str — what the skill does",
            "tags": "str — comma list",
            "skill_id": "str — for use/improve/restore/get",
            "success": "bool (str) — for 'use'",
            "limit": "int — for recall/list",
            "include_pruned": "bool (str) — for list: show quarantined skills",
            "min_uses": "int (str) — prune threshold (default 5)",
            "max_success_rate": "float (str) — prune threshold (default 0.34)",
            "min_age_days": "float (str) — prune threshold (default 7)",
        },
    )
    def skill(
        action: str, *, query: str = "", name: str = "", kind: str = "",
        body: str = "", description: str = "", tags: str = "",
        skill_id: str = "", success: str = "true", limit: str = "5",
        include_pruned: str = "false",
        min_uses: str = "5", max_success_rate: str = "0.34",
        min_age_days: str = "7",
    ) -> dict[str, Any]:
        library = SkillLibrary(context.db)
        action = (action or "list").strip().lower()
        try:
            n = int(limit or 5)
        except ValueError:
            n = 5
        show_pruned = include_pruned.strip().lower() in {"1", "true", "yes"}
        if action == "prune":
            try:
                mu = int(min_uses or 5)
            except ValueError:
                mu = 5
            try:
                msp = float(max_success_rate or 0.34)
            except ValueError:
                msp = 0.34
            try:
                mad = float(min_age_days or 7)
            except ValueError:
                mad = 7.0
            return library.prune(min_uses=mu, max_success_rate=msp,
                                 min_age_days=mad)
        if action == "restore":
            s = library.restore(skill_id)
            return {"ok": s is not None,
                    "skill": s.to_dict() if s else None}
        if action == "recall":
            recalled = library.recall(query, kind=kind, limit=n)
            return {"skills": [{"name": s.name, "kind": s.kind,
                                 "score": round(sc, 3),
                                 "body": s.body[:2000]}
                                for s, sc in recalled],
                    "context_block": library.context_block(query, limit=n)}
        if action == "context":
            # The canonical "what already worked / what already failed"
            # block — the context any agent prepends before real work.
            return {"ok": True,
                    "context": library.operator_context(query, limit=n)}
        if action == "capture":
            # Mine the tool-call ledger for recovered patterns → skills.
            names = library.capture_from_ledger(limit=max(20, n * 10))
            return {"ok": True, "captured": names,
                    "count": len(names)}
        if action == "save":
            s = library.save(name or query, kind=kind, body=body,
                             description=description,
                             tags=[t.strip() for t in tags.split(",")
                                   if t.strip()])
            return {"ok": True, "skill": s.to_dict()}
        if action == "use":
            s = library.record_use(skill_id, success=success.strip().lower()
                                   in {"1", "true", "yes"}, task=query)
            return {"ok": s is not None, "skill": s.to_dict() if s else None}
        if action == "improve":
            s = library.improve(skill_id, body=body, note=description)
            return {"ok": s is not None, "skill": s.to_dict() if s else None}
        if action == "get":
            s = (library.get(skill_id) if skill_id
                 else library.get_by_name(name))
            return {"ok": s is not None, "skill": s.to_dict() if s else None}
        if action == "stats":
            return library.stats()
        # default: list
        return {"skills": [s.to_dict() for s in
                           library.list(kind=kind, limit=n,
                                        include_pruned=show_pruned)]}
